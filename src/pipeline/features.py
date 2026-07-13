"""
features.py
Convierte List[MatchData] en DataFrames tabulares (nivel-partido y nivel-jugador)
para que los 6 puntos de análisis operen con pandas/numpy de forma vectorizada.
"""
from __future__ import annotations

from typing import List

import pandas as pd

from ..ingestion.schemas import MatchData


def matches_to_team_df(matches: List[MatchData]) -> pd.DataFrame:
    """Una fila por partido con las estadísticas agregadas del equipo."""
    rows = []
    for m in matches:
        ts = m.team_stats
        rows.append({
            "match_id": m.match_id,
            "date": m.match_date,
            "opponent": m.opponent,
            "is_home": m.is_home,
            "formation": m.formation,
            "result": m.result,
            "points": m.points,
            "gf": ts.goals_for,
            "ga": ts.goals_against,
            "gd": ts.goals_for - ts.goals_against,
            "possession": ts.possession,
            "shots": ts.shots,
            "sot": ts.shots_on_target,
            "xg_for": ts.xg_for,
            "xg_against": ts.xg_against,
            "corners": ts.corners,
            "fouls": ts.fouls,
            "pass_accuracy": ts.pass_accuracy,
            "ppda": ts.ppda,
            "line_height": ts.defensive_line_height,
            "directness": ts.directness,
        })
    return pd.DataFrame(rows)


def matches_to_player_df(matches: List[MatchData]) -> pd.DataFrame:
    """Una fila por (jugador, partido). Base para comportamiento/talento/consistencia."""
    rows = []
    for m in matches:
        for p in m.players:
            rows.append({
                "match_id": m.match_id,
                "date": m.match_date,
                "player_id": p.player_id,
                "name": p.name,
                "position": p.position,
                "minutes": p.minutes,
                "rating": p.rating,
                "goals": p.goals,
                "assists": p.assists,
                "shots": p.shots,
                "sot": p.shots_on_target,
                "xg": p.xg,
                "xa": p.xa,
                "key_passes": p.key_passes,
                "passes": p.passes,
                "pass_accuracy": p.pass_accuracy,
                "duels": p.duels,
                "duel_win_rate": p.duel_win_rate,
                "tackles": p.tackles,
                "interceptions": p.interceptions,
                "avg_x": p.avg_x,
                "avg_y": p.avg_y,
                "shot_conversion": p.shot_conversion,
            })
    return pd.DataFrame(rows)
