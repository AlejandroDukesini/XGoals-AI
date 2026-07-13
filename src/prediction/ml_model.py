"""
ml_model.py  —  Modelo ML alternativo (scikit-learn)
====================================================
Alternativa/complemento al Poisson: un clasificador multinomial 1X2 entrenado
sobre features de los informes de equipo (forma, xG diff, vector táctico, etc.).

Uso típico: se entrena con un histórico de partidos etiquetados (H/D/A) y luego
se combina (ensemble) con el Poisson promediando probabilidades. Aquí se deja el
esqueleto entrenable + un ensamblador de probabilidades.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..pipeline.team_report import TeamReport


def build_feature_vector(home_rep: TeamReport, away_rep: TeamReport) -> np.ndarray:
    """Extrae features simétricas (diferencias local-visita) para el clasificador."""
    h, a = home_rep.p1_general, away_rep.p1_general
    hv, av = home_rep.p6_tactical["vector"], away_rep.p6_tactical["vector"]
    return np.array([
        h["ppg"] - a["ppg"],
        h["xg_for_avg"] - a["xg_for_avg"],
        h["xg_against_avg"] - a["xg_against_avg"],
        h["form_trend"] - a["form_trend"],
        h["possession_avg"] - a["possession_avg"],
        hv["press"] - av["press"],
        hv["directness"] - av["directness"],
        home_rep.p5_consistency["team_rating_std"] - away_rep.p5_consistency["team_rating_std"],
    ], dtype=float)


class OutcomeClassifier:
    """Envuelve un RandomForest para predecir H/D/A. Import perezoso de sklearn."""

    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, class_weight="balanced"
        )
        self.classes_ = ["home", "draw", "away"]
        self._fitted = False

    def fit(self, X: np.ndarray, y: List[str]) -> "OutcomeClassifier":
        self.model.fit(X, y)
        self.classes_ = list(self.model.classes_)
        self._fitted = True
        return self

    def predict_proba(self, home_rep, away_rep) -> Dict[str, float]:
        if not self._fitted:
            raise RuntimeError("Modelo sin entrenar: llama a fit() con histórico etiquetado.")
        x = build_feature_vector(home_rep, away_rep).reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        return {c: float(p) for c, p in zip(self.classes_, proba)}


def ensemble_1x2(poisson_probs: Dict[str, float],
                 ml_probs: Dict[str, float],
                 w_poisson: float = 0.6) -> Dict[str, float]:
    """Combina 1X2 de Poisson y ML por media ponderada, renormalizando."""
    keys = ("home", "draw", "away")
    mix = {k: w_poisson * poisson_probs.get(k, 0) + (1 - w_poisson) * ml_probs.get(k, 0)
           for k in keys}
    s = sum(mix.values()) or 1.0
    return {k: round(v / s, 3) for k, v in mix.items()}
