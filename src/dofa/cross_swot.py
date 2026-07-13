"""
cross_swot.py  —  MATRIZ DOFA CRUZADA COMPARATIVA
=================================================
Cruza dos TeamReport (Local vs Rival) y produce una matriz DOFA *cruzada*:
no describe a cada equipo por separado, sino la INTERACCIÓN entre ambos.

    Fortalezas  (F): virtud propia que apunta a una carencia rival -> 'hueco'
    Debilidades (D): carencia propia que el estilo rival puede explotar
    Oportunidades(O): rachas/estados de forma que desequilibran el partido
    Amenazas    (A): ajustes probables del DT rival (predicción de replanteo)

Cada hallazgo es un dict con {dimension, mensaje, severidad/edge}. La severidad
alimenta después los ajustes (lambda_adjust) del motor de predicción.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any

from ..pipeline.team_report import TeamReport


@dataclass
class CrossSWOT:
    home: str
    away: str
    fortalezas: List[Dict[str, Any]] = field(default_factory=list)
    debilidades: List[Dict[str, Any]] = field(default_factory=list)
    oportunidades: List[Dict[str, Any]] = field(default_factory=list)
    amenazas: List[Dict[str, Any]] = field(default_factory=list)
    # Ajuste neto a la fuerza ofensiva de cada equipo derivado del cruce
    lambda_adjust: Dict[str, float] = field(default_factory=lambda: {"home": 1.0, "away": 1.0})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matchup": f"{self.home} vs {self.away}",
            "Fortalezas": self.fortalezas,
            "Debilidades": self.debilidades,
            "Oportunidades": self.oportunidades,
            "Amenazas": self.amenazas,
            "lambda_adjust": self.lambda_adjust,
        }


def _edge(a: float, b: float) -> float:
    """Ventaja relativa de a sobre b, acotada a [-1, 1]."""
    denom = abs(a) + abs(b)
    return round((a - b) / denom, 2) if denom else 0.0


def cross_analyze(home_rep: TeamReport, away_rep: TeamReport) -> CrossSWOT:
    H, A = home_rep, away_rep
    swot = CrossSWOT(home=H.team, away=A.team)

    hg, ag = H.p1_general, A.p1_general
    ht, at = H.p6_tactical, A.p6_tactical
    hv, av = ht["vector"], at["vector"]

    # ---------------------------------------------------------- #
    # FORTALEZAS: mi ataque vs. su defensa (busca 'huecos')
    # ---------------------------------------------------------- #
    # Local ataca -> mide contra goles concedidos del rival
    atk_edge_h = _edge(hg["xg_for_avg"], ag["goals_against_avg"])
    if atk_edge_h > 0.1:
        swot.fortalezas.append({
            "dimension": "Ataque Local vs Defensa Rival",
            "mensaje": f"{H.team} genera {hg['xg_for_avg']} xG/pp frente a "
                       f"{ag['goals_against_avg']} goles concedidos/pp de {A.team}.",
            "edge": atk_edge_h,
        })
        swot.lambda_adjust["home"] *= 1 + 0.15 * atk_edge_h

    atk_edge_a = _edge(ag["xg_for_avg"], hg["goals_against_avg"])
    if atk_edge_a > 0.1:
        swot.fortalezas.append({
            "dimension": "Ataque Rival vs Defensa Local",
            "mensaje": f"{A.team} genera {ag['xg_for_avg']} xG/pp frente a "
                       f"{hg['goals_against_avg']} concedidos/pp de {H.team}.",
            "edge": atk_edge_a,
        })
        swot.lambda_adjust["away"] *= 1 + 0.15 * atk_edge_a

    # ---------------------------------------------------------- #
    # DEBILIDADES: mi carencia vs. el estilo ofensivo rival
    # ---------------------------------------------------------- #
    # Línea alta + rival directo/transiciones => zona de contraataque
    if hv["line"] > 0.6 and av["directness"] > 0.55:
        swot.debilidades.append({
            "dimension": "Espalda de la defensa Local",
            "mensaje": f"{H.team} juega con línea alta ({ht['avg_line_height']}) y "
                       f"{A.team} es directo: riesgo de contraataques a la espalda.",
            "severidad": round(hv["line"] * av["directness"], 2),
        })
        swot.lambda_adjust["away"] *= 1.10

    if av["line"] > 0.6 and hv["directness"] > 0.55:
        swot.debilidades.append({
            "dimension": "Espalda de la defensa Rival",
            "mensaje": f"{A.team} sube la línea y {H.team} ataca directo: "
                       f"rutas de contra para el local.",
            "severidad": round(av["line"] * hv["directness"], 2),
        })
        swot.lambda_adjust["home"] *= 1.10

    # Presión rival vs. precisión de pase propia (riesgo de pérdidas en salida)
    if av["press"] > 0.6 and hg.get("shot_accuracy", 0) and H.p2_variance["possession_cv"] > 0.1:
        swot.debilidades.append({
            "dimension": "Salida de balón Local bajo presión",
            "mensaje": f"{A.team} presiona alto (PPDA {at['avg_ppda']}); "
                       f"{H.team} muestra posesión inestable.",
            "severidad": av["press"],
        })

    # ---------------------------------------------------------- #
    # OPORTUNIDADES: rachas / estados de forma / jugadores calientes
    # ---------------------------------------------------------- #
    for rep, side in ((H, "home"), (A, "away")):
        # Forma reciente por encima del promedio de la ventana
        if rep.p1_general["form_trend"] > rep.p1_general["ppg"] + 0.3:
            swot.oportunidades.append({
                "dimension": f"Racha ascendente ({rep.team})",
                "mensaje": f"{rep.team} llega en alza: forma ponderada "
                           f"{rep.p1_general['form_trend']} > PPG {rep.p1_general['ppg']}.",
            })
            swot.lambda_adjust[side] *= 1.05
        # Delantero 'caliente' (sobre-rendimiento vs xG)
        fin = rep.p4_talent.get("finishers", [])
        if fin and fin[0].get("overperformance", 0) > 0.8:
            swot.oportunidades.append({
                "dimension": f"Delantero en racha ({rep.team})",
                "mensaje": f"{fin[0]['name']} supera su xG en "
                           f"{fin[0]['overperformance']} goles: momento fino de cara al gol.",
            })

    # ---------------------------------------------------------- #
    # AMENAZAS: ajustes/replanteos probables del DT rival
    # ---------------------------------------------------------- #
    # Flexibilidad de esquema alta => el DT probablemente ajustará según marcador
    for rep, opp in ((H, A), (A, H)):
        flex = rep.p6_tactical["formation_flexibility"]
        if flex >= 3:
            swot.amenazas.append({
                "dimension": f"Replanteo táctico ({rep.team})",
                "mensaje": f"{rep.team} rotó {flex} esquemas en 5 pp: alta probabilidad "
                           f"de ajustar el sistema en función de {opp.team}.",
                "severidad": round(min(1.0, flex / 5), 2),
            })
    # Banco/fatiga: si el rival tiene jugadores fatigados, amenaza de bajón físico
    for rep in (H, A):
        hf = rep.p3_behavior.get("high_fatigue_players", [])
        if hf:
            swot.amenazas.append({
                "dimension": f"Riesgo físico ({rep.team})",
                "mensaje": f"Jugadores con carga alta en {rep.team}: {', '.join(hf)}. "
                           f"Posible caída en el tramo final.",
            })

    # Redondeo de ajustes
    swot.lambda_adjust = {k: round(v, 3) for k, v in swot.lambda_adjust.items()}
    return swot
