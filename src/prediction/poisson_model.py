"""
poisson_model.py  —  MOTOR DE PREDICCIÓN PROBABILÍSTICO
=======================================================
Modelo bivariado de Poisson (con corrección tipo Dixon-Coles para marcadores
bajos) sobre una malla de resultados 0..MAX_GOALS.

MODELO MATEMÁTICO
-----------------
1) Se estiman fuerzas ofensivas/defensivas de cada equipo a partir de los xG/goles
   de sus últimos partidos, relativas al promedio de la liga:

       attack_i  = xG_for_i  / liga_media
       defense_i = xG_against_i / liga_media       (menor = mejor defensa)

2) Goles esperados (tasas de Poisson) del enfrentamiento:

       lambda_home = liga_media * attack_home * defense_away * ventaja_local * adj_home
       lambda_away = liga_media * attack_away * defense_home             * adj_away

   donde `adj_*` son los ajustes que aporta el análisis DOFA cruzado.

3) Probabilidad de un marcador (x, y) bajo independencia:

       P(x, y) = Pois(x; λ_home) * Pois(y; λ_away) * τ(x, y)

   τ es la corrección de Dixon-Coles que ajusta 0-0, 1-0, 0-1, 1-1
   (Poisson puro subestima empates de pocos goles).

4) Se agregan las celdas de la malla para obtener 1X2, Over/Under y BTTS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from scipy.stats import poisson

import config
from ..pipeline.team_report import TeamReport


# ------------------------------------------------------------------ #
def _dixon_coles_tau(x: int, y: int, lh: float, la: float, rho: float = -0.08) -> float:
    """Factor de corrección para marcadores bajos (rho negativo sube empates)."""
    if x == 0 and y == 0:
        return 1 - lh * la * rho
    if x == 0 and y == 1:
        return 1 + lh * rho
    if x == 1 and y == 0:
        return 1 + la * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def estimate_lambdas(
    home_rep: TeamReport,
    away_rep: TeamReport,
    adj: Dict[str, float] | None = None,
) -> Tuple[float, float]:
    """Calcula λ_home, λ_away a partir de los informes y ajustes DOFA."""
    adj = adj or {"home": 1.0, "away": 1.0}
    L = config.LEAGUE_AVG_GOALS

    # Fuerzas relativas (usa xG si está, cae a goles si xG=0)
    def atk(rep):  return (rep.p1_general["xg_for_avg"] or rep.p1_general["goals_for_avg"]) / L
    def dfn(rep):  return (rep.p1_general["xg_against_avg"] or rep.p1_general["goals_against_avg"]) / L

    lam_h = L * atk(home_rep) * dfn(away_rep) * config.HOME_ADVANTAGE * adj.get("home", 1.0)
    lam_a = L * atk(away_rep) * dfn(home_rep) * adj.get("away", 1.0)

    # Cota inferior para evitar lambdas degenerados
    return max(0.15, round(lam_h, 3)), max(0.15, round(lam_a, 3))


@dataclass
class MatchPrediction:
    home: str
    away: str
    lambda_home: float
    lambda_away: float
    score_matrix: np.ndarray
    p_home: float
    p_draw: float
    p_away: float
    p_over_2_5: float
    p_under_2_5: float
    p_btts_yes: float
    p_btts_no: float
    most_likely_score: Tuple[int, int]
    top_scores: list = field(default_factory=list)

    def probabilities(self) -> Dict[str, float]:
        """Diccionario plano de probabilidades para el motor de valor."""
        return {
            "home": self.p_home, "draw": self.p_draw, "away": self.p_away,
            "over_2_5": self.p_over_2_5, "under_2_5": self.p_under_2_5,
            "btts_yes": self.p_btts_yes, "btts_no": self.p_btts_no,
        }

    def report(self) -> str:
        s = self.most_likely_score
        return (
            f"{self.home} {s[0]}-{s[1]} {self.away}  "
            f"(λ {self.lambda_home}/{self.lambda_away})\n"
            f"  1X2:  L {self.p_home:.1%} | X {self.p_draw:.1%} | V {self.p_away:.1%}\n"
            f"  O/U 2.5:  Over {self.p_over_2_5:.1%} | Under {self.p_under_2_5:.1%}\n"
            f"  BTTS:  Sí {self.p_btts_yes:.1%} | No {self.p_btts_no:.1%}"
        )


def predict(
    home_rep: TeamReport,
    away_rep: TeamReport,
    adj: Dict[str, float] | None = None,
) -> MatchPrediction:
    """Genera la predicción completa del encuentro."""
    lam_h, lam_a = estimate_lambdas(home_rep, away_rep, adj)
    n = config.MAX_GOALS + 1

    # Distribuciones marginales de Poisson
    ph = poisson.pmf(np.arange(n), lam_h)
    pa = poisson.pmf(np.arange(n), lam_a)

    # Matriz conjunta con corrección Dixon-Coles
    M = np.outer(ph, pa)
    for x in range(min(2, n)):
        for y in range(min(2, n)):
            M[x, y] *= _dixon_coles_tau(x, y, lam_h, lam_a)
    M /= M.sum()  # renormaliza tras la corrección

    # Agregados de mercado
    p_home = float(np.tril(M, -1).sum())   # x > y
    p_away = float(np.triu(M, 1).sum())    # y > x
    p_draw = float(np.trace(M))            # x == y

    idx = np.indices((n, n))
    totals = idx[0] + idx[1]
    p_over = float(M[totals > 2.5].sum())
    p_btts = float(M[1:, 1:].sum())        # ambos anotan (>=1 cada uno)

    # Marcadores más probables
    flat = [((x, y), float(M[x, y])) for x in range(n) for y in range(n)]
    flat.sort(key=lambda t: -t[1])
    top = [{"score": f"{x}-{y}", "prob": round(p, 3)} for (x, y), p in flat[:5]]
    ml = flat[0][0]

    return MatchPrediction(
        home=home_rep.team, away=away_rep.team,
        lambda_home=lam_h, lambda_away=lam_a,
        score_matrix=M,
        p_home=p_home, p_draw=p_draw, p_away=p_away,
        p_over_2_5=p_over, p_under_2_5=1 - p_over,
        p_btts_yes=p_btts, p_btts_no=1 - p_btts,
        most_likely_score=ml, top_scores=top,
    )
