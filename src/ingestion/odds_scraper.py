"""
odds_scraper.py  —  Ingesta de CUOTAS de mercado (BetPlay u otras)
==================================================================
Devuelve un objeto MarketOdds para un partido. En OFFLINE_MODE genera cuotas
coherentes (con margen/overround realista) para poder probar el motor de valor
sin conexión.

Nota legal: las casas de apuestas suelen prohibir el scraping en sus ToS y usan
protección anti-bot. Para producción usa un feed de odds con licencia
(The Odds API, OddsPortal API, feeds del propio operador). El backend de red
de abajo es un esqueleto ilustrativo.
"""
from __future__ import annotations

import random
from typing import Dict

import requests
from bs4 import BeautifulSoup

import config
from .schemas import MarketOdds


def fetch_odds(match_id: str, home: str = "Local", away: str = "Rival") -> MarketOdds:
    if config.OFFLINE_MODE:
        return _synthetic_odds(match_id)
    return _scrape_betplay(match_id, home, away)


# ------------------------------------------------------------------ #
def _apply_margin(probs: Dict[str, float], overround: float = 1.06) -> Dict[str, float]:
    """Convierte probabilidades 'justas' en cuotas con margen de la casa.
    cuota = 1 / (prob * overround)."""
    return {k: round(1.0 / (p * overround), 2) for k, p in probs.items()}


def _synthetic_odds(match_id: str) -> MarketOdds:
    rng = random.Random(hash(match_id) % 10_000)
    # Probabilidades base "verdaderas" con algo de aleatoriedad
    p_home = rng.uniform(0.38, 0.52)
    p_away = rng.uniform(0.20, 0.34)
    p_draw = max(0.10, 1 - p_home - p_away)
    s = p_home + p_draw + p_away
    o1x2 = _apply_margin({"home": p_home / s, "draw": p_draw / s, "away": p_away / s})

    p_over = rng.uniform(0.45, 0.62)
    ou = _apply_margin({"over": p_over, "under": 1 - p_over})

    p_btts = rng.uniform(0.48, 0.60)
    btts = _apply_margin({"yes": p_btts, "no": 1 - p_btts})

    return MarketOdds(
        bookmaker="BetPlay(sim)", match_id=match_id,
        home=o1x2["home"], draw=o1x2["draw"], away=o1x2["away"],
        over_2_5=ou["over"], under_2_5=ou["under"],
        btts_yes=btts["yes"], btts_no=btts["no"],
    )


def _scrape_betplay(match_id: str, home: str, away: str) -> MarketOdds:
    """Esqueleto de scraping real. Ajusta URL/selectores; respeta los ToS."""
    sess = requests.Session()
    sess.headers.update({"User-Agent": config.USER_AGENT})
    url = f"https://betplay.com.co/apuestas#event/{match_id}"  # placeholder
    soup = BeautifulSoup(sess.get(url, timeout=config.REQUEST_TIMEOUT).text, "lxml")

    def _odd(css: str) -> float:
        el = soup.select_one(css)
        try:
            return float(el.get_text(strip=True).replace(",", "."))
        except Exception:  # noqa: BLE001
            return 0.0

    return MarketOdds(
        bookmaker="BetPlay", match_id=match_id,
        home=_odd("[data-market='1x2'] [data-sel='1']"),
        draw=_odd("[data-market='1x2'] [data-sel='X']"),
        away=_odd("[data-market='1x2'] [data-sel='2']"),
        over_2_5=_odd("[data-market='ou2.5'] [data-sel='over']"),
        under_2_5=_odd("[data-market='ou2.5'] [data-sel='under']"),
        btts_yes=_odd("[data-market='btts'] [data-sel='yes']"),
        btts_no=_odd("[data-market='btts'] [data-sel='no']"),
    )


if __name__ == "__main__":
    print(fetch_odds("LOC-vs-RIV").to_dict())
