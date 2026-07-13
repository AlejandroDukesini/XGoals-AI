"""
value_bets.py  —  MOTOR DE VALOR ESPERADO (Value Bets)
======================================================
Contrasta las probabilidades del modelo (IA) contra las cuotas de la casa para
detectar apuestas con valor positivo.

MODELO MATEMÁTICO
-----------------
- Prob. implícita de la casa:  q = 1 / cuota   (incluye margen/overround).
- Edge (ventaja):              edge = p_modelo - q
- Valor Esperado por unidad:   EV = p_modelo * (cuota - 1) - (1 - p_modelo)
                                  = p_modelo * cuota - 1
  EV > 0  => apuesta con valor (a largo plazo, rentable).
- Stake (Kelly fraccionado):   f* = (b*p - (1-p)) / b ,  b = cuota - 1
  Se multiplica por KELLY_FRACTION para reducir varianza (Kelly conservador).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import config
from ..ingestion.schemas import MarketOdds


# Mapea cada mercado a (prob_key del modelo, atributo de cuota en MarketOdds)
_MARKETS = {
    "1 (Local)":     ("home", "home"),
    "X (Empate)":    ("draw", "draw"),
    "2 (Visita)":    ("away", "away"),
    "Over 2.5":      ("over_2_5", "over_2_5"),
    "Under 2.5":     ("under_2_5", "under_2_5"),
    "BTTS Sí":       ("btts_yes", "btts_yes"),
    "BTTS No":       ("btts_no", "btts_no"),
}


@dataclass
class ValueBet:
    market: str
    model_prob: float
    book_odds: float
    implied_prob: float
    edge: float
    ev: float                 # por unidad apostada
    kelly_stake: float        # fracción del bankroll sugerida
    is_value: bool

    def as_dict(self) -> Dict:
        return asdict(self)


def _kelly_fraction(p: float, odds: float) -> float:
    b = odds - 1
    if b <= 0:
        return 0.0
    f = (b * p - (1 - p)) / b
    return max(0.0, f) * config.KELLY_FRACTION


def evaluate(probs: Dict[str, float], odds: MarketOdds,
             min_edge: float = None) -> List[ValueBet]:
    """Evalúa todos los mercados y devuelve la lista ordenada por EV."""
    min_edge = config.MIN_EDGE if min_edge is None else min_edge
    out: List[ValueBet] = []

    for label, (pk, ok) in _MARKETS.items():
        p = probs.get(pk, 0.0)
        o = getattr(odds, ok, 0.0)
        if o <= 1.0 or p <= 0:
            continue
        implied = 1 / o
        edge = p - implied
        ev = p * o - 1
        out.append(ValueBet(
            market=label,
            model_prob=round(p, 3),
            book_odds=round(o, 2),
            implied_prob=round(implied, 3),
            edge=round(edge, 3),
            ev=round(ev, 3),
            kelly_stake=round(_kelly_fraction(p, o), 4),
            is_value=(edge >= min_edge and ev > 0),
        ))

    return sorted(out, key=lambda b: -b.ev)


def value_only(probs: Dict[str, float], odds: MarketOdds) -> List[ValueBet]:
    """Solo las apuestas que superan el umbral de valor."""
    return [b for b in evaluate(probs, odds) if b.is_value]
