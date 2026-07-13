"""
schemas.py
Contratos de datos del sistema. Definen la "forma canónica" que todo módulo
espera. Usamos dataclasses (ligeras, sin dependencias externas) y ofrecemos
conversores a/desde dict (JSON) y a fila de DataFrame.

Jerarquía:
    MatchData  (1 partido de 1 equipo)
      ├── team_stats:   TeamMatchStats   (agregado del equipo en ese partido)
      └── players:      List[PlayerMatchStats]

Un equipo se analiza con una lista de 5 MatchData (últimos 5 partidos).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import List, Optional, Dict, Any


# ------------------------------------------------------------------ #
# Nivel jugador
# ------------------------------------------------------------------ #
@dataclass
class PlayerMatchStats:
    player_id: str
    name: str
    position: str                 # GK / DEF / MID / FWD
    minutes: int = 0
    rating: float = 6.0           # calificación estilo SofaScore (1-10)

    # Ofensiva
    goals: int = 0
    assists: int = 0
    shots: int = 0
    shots_on_target: int = 0
    xg: float = 0.0               # expected goals
    xa: float = 0.0               # expected assists
    key_passes: int = 0

    # Construcción / control
    passes: int = 0
    passes_completed: int = 0

    # Defensa / duelos
    duels: int = 0
    duels_won: int = 0
    tackles: int = 0
    interceptions: int = 0

    # Zona de acción (centroide de eventos, x/y en 0-100). Base del "mapa de calor".
    avg_x: float = 50.0
    avg_y: float = 50.0

    # Disciplina
    yellow: int = 0
    red: int = 0

    @property
    def pass_accuracy(self) -> float:
        return self.passes_completed / self.passes if self.passes else 0.0

    @property
    def duel_win_rate(self) -> float:
        return self.duels_won / self.duels if self.duels else 0.0

    @property
    def shot_conversion(self) -> float:
        return self.goals / self.shots if self.shots else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ #
# Nivel equipo (en un partido)
# ------------------------------------------------------------------ #
@dataclass
class TeamMatchStats:
    goals_for: int = 0
    goals_against: int = 0
    possession: float = 50.0      # %
    shots: int = 0
    shots_on_target: int = 0
    xg_for: float = 0.0
    xg_against: float = 0.0
    corners: int = 0
    fouls: int = 0
    passes: int = 0
    pass_accuracy: float = 0.0
    ppda: float = 12.0            # passes allowed per defensive action (presión)

    # Rasgos tácticos observados (se deducen en el pipeline, aquí opcionales crudos)
    defensive_line_height: float = 50.0   # 0=bloque bajo, 100=bloque alto
    directness: float = 0.5               # 0=posesión, 1=juego directo

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ #
# Nivel partido
# ------------------------------------------------------------------ #
@dataclass
class MatchData:
    match_id: str
    match_date: str               # ISO 'YYYY-MM-DD'
    team: str                     # equipo analizado
    opponent: str
    is_home: bool
    competition: str = "Liga"
    formation: str = "4-3-3"      # esquema del DT en ese partido
    team_stats: TeamMatchStats = field(default_factory=TeamMatchStats)
    players: List[PlayerMatchStats] = field(default_factory=list)

    @property
    def result(self) -> str:
        gf, ga = self.team_stats.goals_for, self.team_stats.goals_against
        return "W" if gf > ga else "D" if gf == ga else "L"

    @property
    def points(self) -> int:
        return {"W": 3, "D": 1, "L": 0}[self.result]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["result"] = self.result
        d["points"] = self.points
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MatchData":
        players = [PlayerMatchStats(**p) for p in d.get("players", [])]
        ts = TeamMatchStats(**d.get("team_stats", {}))
        base = {k: v for k, v in d.items()
                if k not in ("players", "team_stats", "result", "points")}
        return cls(team_stats=ts, players=players, **base)


# ------------------------------------------------------------------ #
# Cuotas de mercado
# ------------------------------------------------------------------ #
@dataclass
class MarketOdds:
    bookmaker: str = "BetPlay"
    match_id: str = ""
    # 1X2
    home: float = 0.0
    draw: float = 0.0
    away: float = 0.0
    # Totales
    over_2_5: float = 0.0
    under_2_5: float = 0.0
    # Ambos anotan
    btts_yes: float = 0.0
    btts_no: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "PlayerMatchStats",
    "TeamMatchStats",
    "MatchData",
    "MarketOdds",
]
