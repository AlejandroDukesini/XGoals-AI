"""
team_report.py  —  PIPELINE DE CIENCIA DE DATOS (los 6 puntos)
==============================================================
Procesa los últimos N partidos de UN equipo y produce un TeamReport con los
6 bloques analíticos. Cada punto es una función pura (recibe DataFrames,
devuelve dict) para poder testearla y reutilizarla de forma aislada.

    Punto 1  general_stats        -> tendencias globales (goles, xG, posesión)
    Punto 2  match_variance       -> varianza/estabilidad entre jornadas
    Punto 3  player_behavior      -> fatiga y zonas de movimiento (heatmap)
    Punto 4  talent_efficiency    -> eficiencia (pases clave, duelos, remate)
    Punto 5  performance_consistency -> desviación estándar de ratings
    Punto 6  tactical_dna         -> deducción del sistema del DT

La "ingeniería inversa" vive en los puntos 3 y 6: a partir de eventos y
agregados observados deducimos intención táctica y estado físico, sin que el
dato venga etiquetado explícitamente.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from ..ingestion.schemas import MatchData
from .features import matches_to_team_df, matches_to_player_df


# ================================================================== #
# PUNTO 1 — Estadística General
# ================================================================== #
def general_stats(tdf: pd.DataFrame) -> Dict[str, Any]:
    """Resumen de tendencias globales sobre la ventana de partidos."""
    n = len(tdf)
    return {
        "n_matches": n,
        "record": f"{(tdf.result=='W').sum()}-{(tdf.result=='D').sum()}-{(tdf.result=='L').sum()}",
        "ppg": round(tdf.points.mean(), 2),                 # puntos por partido
        "goals_for_avg": round(tdf.gf.mean(), 2),
        "goals_against_avg": round(tdf.ga.mean(), 2),
        "xg_for_avg": round(tdf.xg_for.mean(), 2),
        "xg_against_avg": round(tdf.xg_against.mean(), 2),
        # xG diff > goles reales => infra-rendimiento en la definición (regresa al alza)
        "finishing_delta": round(tdf.gf.mean() - tdf.xg_for.mean(), 2),
        "possession_avg": round(tdf.possession.mean(), 1),
        "shots_avg": round(tdf.shots.mean(), 1),
        "shot_accuracy": round(tdf.sot.sum() / max(tdf.shots.sum(), 1), 3),
        "clean_sheets": int((tdf.ga == 0).sum()),
        "btts_rate": round(((tdf.gf > 0) & (tdf.ga > 0)).mean(), 2),
        "over25_rate": round(((tdf.gf + tdf.ga) > 2.5).mean(), 2),
        "form_trend": _weighted_form(tdf),                  # forma reciente ponderada
    }


def _weighted_form(tdf: pd.DataFrame) -> float:
    """Puntos por partido ponderando más los partidos recientes (recencia lineal)."""
    pts = tdf.points.to_numpy()
    w = np.arange(1, len(pts) + 1)                          # 1..n, el último pesa más
    return round(float(np.average(pts, weights=w)), 2)


# ================================================================== #
# PUNTO 2 — Estadística por Partido (varianza entre jornadas)
# ================================================================== #
def match_variance(tdf: pd.DataFrame) -> Dict[str, Any]:
    """Mide qué tan estable/errático es el equipo jornada a jornada."""
    def cv(series):  # coeficiente de variación (std/media) — 0 = muy estable
        m = series.mean()
        return round(series.std(ddof=0) / m, 3) if m else 0.0

    return {
        "gf_std": round(tdf.gf.std(ddof=0), 2),
        "ga_std": round(tdf.ga.std(ddof=0), 2),
        "xg_std": round(tdf.xg_for.std(ddof=0), 2),
        "points_std": round(tdf.points.std(ddof=0), 2),
        "gf_cv": cv(tdf.gf),
        "possession_cv": cv(tdf.possession),
        "stability_index": round(1 / (1 + tdf.points.std(ddof=0)), 3),  # 1=perfecto
        "per_match": tdf[["match_id", "opponent", "gf", "ga",
                          "xg_for", "points"]].to_dict("records"),
        "best_match": tdf.loc[tdf.gd.idxmax(), "match_id"] if len(tdf) else None,
        "worst_match": tdf.loc[tdf.gd.idxmin(), "match_id"] if len(tdf) else None,
    }


# ================================================================== #
# PUNTO 3 — Comportamiento del Jugador (fatiga + zonas / heatmap invertido)
# ================================================================== #
def player_behavior(pdf: pd.DataFrame) -> Dict[str, Any]:
    """
    Ingeniería inversa del estado físico y posicional:
      - Fatiga: minutos acumulados y caída de rating en partidos recientes.
      - Zona habitual: centroide (avg_x, avg_y) => 'mapa de calor invertido'
        (deducimos dónde ACTÚA cada jugador a partir de eventos agregados).
    """
    if pdf.empty:
        return {"fatigue": [], "zones": []}

    # --- Fatiga: carga de minutos + tendencia de rating (pendiente por jugador)
    fatigue = []
    for pid, g in pdf.sort_values("date").groupby("player_id"):
        total_min = int(g.minutes.sum())
        trend = _slope(g.rating.to_numpy())                # <0 => rendimiento a la baja
        load = total_min / (90 * len(g))                   # 1.0 = siempre 90'
        fatigue.append({
            "player_id": pid,
            "name": g.name.iloc[0],
            "position": g["position"].iloc[0],
            "total_minutes": total_min,
            "load_ratio": round(load, 2),
            "rating_trend": round(trend, 3),
            # Riesgo de fatiga: mucha carga + rating decreciente
            "fatigue_risk": round(min(1.0, load * max(0, -trend) * 5 + load * 0.3), 2),
        })

    # --- Zonas de movimiento (centroide medio por jugador)
    zones = (pdf.groupby(["player_id", "name", "position"])
             .agg(avg_x=("avg_x", "mean"), avg_y=("avg_y", "mean"),
                  touches=("passes", "mean"))
             .reset_index().round(1).to_dict("records"))

    return {
        "fatigue": sorted(fatigue, key=lambda d: -d["fatigue_risk"]),
        "zones": zones,
        "high_fatigue_players": [f["name"] for f in fatigue if f["fatigue_risk"] > 0.6],
    }


def _slope(y: np.ndarray) -> float:
    """Pendiente de una regresión lineal simple sobre índice temporal."""
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


# ================================================================== #
# PUNTO 4 — Habilidad / Talento (métricas de eficiencia)
# ================================================================== #
def talent_efficiency(pdf: pd.DataFrame) -> Dict[str, Any]:
    """Ranking de eficiencia individual normalizada por minutos (por 90')."""
    if pdf.empty:
        return {"top_players": [], "key_creators": [], "finishers": []}

    agg = (pdf.groupby(["player_id", "name", "position"])
           .agg(minutes=("minutes", "sum"),
                goals=("goals", "sum"), assists=("assists", "sum"),
                xg=("xg", "sum"), xa=("xa", "sum"),
                key_passes=("key_passes", "sum"),
                shots=("shots", "sum"), sot=("sot", "sum"),
                duels=("duels", "sum"),
                duel_win_rate=("duel_win_rate", "mean"),
                pass_accuracy=("pass_accuracy", "mean"),
                rating=("rating", "mean"))
           .reset_index())

    p90 = agg.minutes.clip(lower=1) / 90
    agg["goals_p90"] = (agg.goals / p90).round(2)
    agg["ga_p90"] = ((agg.goals + agg.assists) / p90).round(2)      # G+A por 90
    agg["kp_p90"] = (agg.key_passes / p90).round(2)
    agg["shot_eff"] = (agg.goals / agg.shots.clip(lower=1)).round(2)
    agg["overperformance"] = (agg.goals - agg.xg).round(2)          # def. clínica
    # Índice de talento compuesto (ponderación editable)
    agg["talent_index"] = (
        agg.rating * 0.4
        + agg.ga_p90 * 2.0
        + agg.kp_p90 * 0.6
        + agg.duel_win_rate * 2.0
    ).round(2)

    top = agg.sort_values("talent_index", ascending=False)
    return {
        "top_players": top.head(5)[["name", "position", "talent_index",
                                    "rating", "ga_p90"]].to_dict("records"),
        "key_creators": agg.sort_values("kp_p90", ascending=False)
                          .head(3)[["name", "kp_p90", "xa"]].to_dict("records"),
        "finishers": agg.sort_values("overperformance", ascending=False)
                        .head(3)[["name", "goals", "xg", "overperformance"]].to_dict("records"),
        "squad_avg_rating": round(agg.rating.mean(), 2),
    }


# ================================================================== #
# PUNTO 5 — Consistencia de Rendimiento (std de calificaciones)
# ================================================================== #
def performance_consistency(pdf: pd.DataFrame) -> Dict[str, Any]:
    """Desviación estándar de ratings: baja std = jugador fiable/predecible."""
    if pdf.empty:
        return {"team_rating_std": 0.0, "players": []}

    per_player = (pdf.groupby(["player_id", "name"])
                  .agg(mean_rating=("rating", "mean"),
                       std_rating=("rating", "std"),
                       apps=("rating", "count"))
                  .reset_index()
                  .fillna(0.0))
    per_player["consistency"] = (1 / (1 + per_player.std_rating)).round(3)  # 1=constante
    per_player = per_player.round(2)

    return {
        # std de la media por partido del equipo => solidez colectiva jornada a jornada
        "team_rating_std": round(pdf.groupby("match_id").rating.mean().std(ddof=0), 3),
        "most_consistent": per_player.sort_values("consistency", ascending=False)
                             .head(3)[["name", "mean_rating", "consistency"]].to_dict("records"),
        "most_volatile": per_player.sort_values("std_rating", ascending=False)
                           .head(3)[["name", "mean_rating", "std_rating"]].to_dict("records"),
        "players": per_player.to_dict("records"),
    }


# ================================================================== #
# PUNTO 6 — Análisis Táctico del DT (deducción del sistema de juego)
# ================================================================== #
def tactical_dna(tdf: pd.DataFrame, pdf: pd.DataFrame) -> Dict[str, Any]:
    """
    Ingeniería inversa del plan del DT a partir de señales observables:
      - PPDA bajo + línea alta       => bloque alto / presión
      - directness alto + posesión baja => juego directo / transiciones
      - posesión alta + pass_acc alto   => juego de posesión
    Devuelve etiquetas legibles + vector numérico para el motor de predicción.
    """
    ppda = tdf.ppda.mean()
    line = tdf.line_height.mean()
    directness = tdf.directness.mean()
    possession = tdf.possession.mean()
    pass_acc = tdf.pass_accuracy.mean()

    # --- Reglas de deducción (fuzzy, umbrales calibrables)
    if ppda < 9 and line > 60:
        press = "Bloque alto / presión intensa"
    elif ppda > 13 and line < 45:
        press = "Bloque bajo / repliegue"
    else:
        press = "Bloque medio / presión selectiva"

    if directness > 0.6 and possession < 48:
        style = "Juego directo / transiciones rápidas"
    elif possession > 55 and pass_acc > 0.83:
        style = "Posesión / construcción elaborada"
    else:
        style = "Mixto / pragmático"

    # Formación modal y variabilidad de esquema (¿el DT rota de sistema?)
    formation_mode = tdf.formation.mode().iloc[0] if len(tdf) else "n/d"
    formation_changes = tdf.formation.nunique()

    # Ancho de campo usado (dispersión de avg_y) => juego por bandas vs. interior
    width = round(pdf.avg_y.std(ddof=0), 1) if not pdf.empty else 0.0

    return {
        "press_scheme": press,
        "play_style": style,
        "formation_mode": formation_mode,
        "formation_flexibility": formation_changes,   # 1=rígido, >1 adaptable
        "avg_ppda": round(ppda, 1),
        "avg_line_height": round(line, 1),
        "field_width_usage": width,
        # Vector táctico normalizado (para features del modelo ML/ajustes DOFA)
        "vector": {
            "press": round(np.clip((15 - ppda) / 11, 0, 1), 2),
            "line": round(line / 100, 2),
            "directness": round(directness, 2),
            "possession": round(possession / 100, 2),
        },
    }


# ================================================================== #
# Ensamblador
# ================================================================== #
@dataclass
class TeamReport:
    team: str
    is_home: bool
    p1_general: Dict[str, Any] = field(default_factory=dict)
    p2_variance: Dict[str, Any] = field(default_factory=dict)
    p3_behavior: Dict[str, Any] = field(default_factory=dict)
    p4_talent: Dict[str, Any] = field(default_factory=dict)
    p5_consistency: Dict[str, Any] = field(default_factory=dict)
    p6_tactical: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "team": self.team,
            "record": self.p1_general.get("record"),
            "ppg": self.p1_general.get("ppg"),
            "xg_for": self.p1_general.get("xg_for_avg"),
            "style": self.p6_tactical.get("play_style"),
            "press": self.p6_tactical.get("press_scheme"),
        }


def build_team_report(team: str, matches: List[MatchData]) -> TeamReport:
    """Ejecuta los 6 puntos y devuelve el informe consolidado del equipo."""
    tdf = matches_to_team_df(matches)
    pdf = matches_to_player_df(matches)
    return TeamReport(
        team=team,
        is_home=matches[0].is_home if matches else True,
        p1_general=general_stats(tdf),
        p2_variance=match_variance(tdf),
        p3_behavior=player_behavior(pdf),
        p4_talent=talent_efficiency(pdf),
        p5_consistency=performance_consistency(pdf),
        p6_tactical=tactical_dna(tdf, pdf),
    )
