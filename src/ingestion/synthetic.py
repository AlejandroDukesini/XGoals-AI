"""
synthetic.py
Genera datos realistas de los últimos N partidos de un equipo sin depender de
la red. Cada equipo tiene un "ADN" (fuerza ofensiva/defensiva, estilo táctico)
y los partidos se muestrean alrededor de ese ADN con ruido controlado.

Sirve para: desarrollo offline, tests reproducibles y demos. El resto del
sistema no distingue si los datos vienen de aquí o de un scraper real: ambos
devuelven List[MatchData].
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from typing import List

import numpy as np

from .schemas import MatchData, TeamMatchStats, PlayerMatchStats

_POSITIONS = (["GK"] + ["DEF"] * 4 + ["MID"] * 3 + ["FWD"] * 3)  # XI base 4-3-3


def _make_players(team: str, quality: float, rng: random.Random) -> List[PlayerMatchStats]:
    players = []
    for i, pos in enumerate(_POSITIONS):
        base_rating = 6.0 + quality * 1.5 + rng.uniform(-0.6, 0.9)
        is_att = pos in ("MID", "FWD")
        p = PlayerMatchStats(
            player_id=f"{team[:3].upper()}_{i:02d}",
            name=f"{team} J{i+1}",
            position=pos,
            minutes=rng.choice([90, 90, 90, 78, 65, 30]),
            rating=round(np.clip(base_rating, 4.5, 9.5), 2),
            goals=rng.choices([0, 1, 2], weights=[85, 13, 2])[0] if is_att else 0,
            assists=rng.choices([0, 1], weights=[88, 12])[0] if is_att else 0,
            shots=rng.randint(0, 5) if is_att else rng.randint(0, 1),
            shots_on_target=rng.randint(0, 2) if is_att else 0,
            xg=round(rng.uniform(0, 0.9) if is_att else rng.uniform(0, 0.1), 2),
            xa=round(rng.uniform(0, 0.4) if is_att else rng.uniform(0, 0.05), 2),
            key_passes=rng.randint(0, 4) if is_att else rng.randint(0, 1),
            passes=rng.randint(20, 75),
            passes_completed=0,  # se rellena abajo
            duels=rng.randint(4, 18),
            duels_won=0,
            tackles=rng.randint(0, 5),
            interceptions=rng.randint(0, 4),
            avg_x=float(np.clip({"GK": 8, "DEF": 30, "MID": 55, "FWD": 78}[pos]
                                + rng.uniform(-8, 8), 0, 100)),
            avg_y=float(np.clip(50 + rng.uniform(-30, 30), 0, 100)),
            yellow=rng.choices([0, 1], weights=[80, 20])[0],
        )
        p.passes_completed = int(p.passes * rng.uniform(0.72, 0.93))
        p.duels_won = int(p.duels * rng.uniform(0.35, 0.65))
        players.append(p)
    return players


def generate_team_matches(
    team: str,
    n: int = 5,
    quality: float = 0.5,          # 0=flojo, 1=élite
    style_directness: float = 0.5, # 0=posesión, 1=directo
    press_intensity: float = 0.5,  # 0=bloque bajo, 1=presión alta
    seed: int | None = None,
) -> List[MatchData]:
    """Devuelve los últimos `n` partidos sintéticos de `team`."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    matches: List[MatchData] = []
    opponents = ["Rival A", "Rival B", "Rival C", "Rival D", "Rival E", "Rival F"]

    for j in range(n):
        is_home = j % 2 == 0
        atk_lambda = 0.7 + quality * 1.8 + (0.2 if is_home else 0)
        gf = int(np_rng.poisson(atk_lambda))
        ga = int(np_rng.poisson(max(0.3, 1.7 - quality * 1.2)))
        players = _make_players(team, quality, rng)

        ts = TeamMatchStats(
            goals_for=gf,
            goals_against=ga,
            possession=round(np.clip(50 + (0.5 - style_directness) * 40
                                     + rng.uniform(-6, 6), 30, 72), 1),
            shots=sum(p.shots for p in players),
            shots_on_target=sum(p.shots_on_target for p in players),
            xg_for=round(sum(p.xg for p in players), 2),
            xg_against=round(max(0.2, np_rng.normal(1.3 - quality, 0.4)), 2),
            corners=rng.randint(2, 9),
            fouls=rng.randint(7, 18),
            passes=sum(p.passes for p in players),
            pass_accuracy=round(rng.uniform(0.74, 0.90), 3),
            ppda=round(np.clip(15 - press_intensity * 8 + rng.uniform(-1.5, 1.5), 4, 18), 1),
            defensive_line_height=round(np.clip(press_intensity * 100
                                                + rng.uniform(-10, 10), 20, 90), 1),
            directness=round(np.clip(style_directness + rng.uniform(-0.1, 0.1), 0, 1), 2),
        )

        matches.append(MatchData(
            match_id=f"{team[:3].upper()}-{j}",
            match_date=(date.today() - timedelta(days=(n - j) * 7)).isoformat(),
            team=team,
            opponent=opponents[j % len(opponents)],
            is_home=is_home,
            formation=rng.choice(["4-3-3", "4-2-3-1", "4-4-2", "3-5-2"]),
            team_stats=ts,
            players=players,
        ))
    return matches


if __name__ == "__main__":
    for m in generate_team_matches("Nacional", quality=0.7, seed=1):
        print(m.match_id, m.result, m.team_stats.goals_for, "-", m.team_stats.goals_against)
