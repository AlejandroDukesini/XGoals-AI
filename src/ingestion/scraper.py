"""
scraper.py  —  MÓDULO DE INGESTIÓN (ETL)
=========================================
Extrae los últimos N partidos de un equipo. Estrategia de "adaptador":

    fetch_last_matches(team)  ->  List[MatchData]

Internamente decide la fuente según config.OFFLINE_MODE / USE_SELENIUM:
    - OFFLINE_MODE  -> datos sintéticos (synthetic.py). Reproducible, sin red.
    - requests+BS4  -> sitios estáticos (ESPN summary pages).
    - Selenium      -> sitios con render JS (Win Sports, widgets dinámicos).

IMPORTANTE (legal/ético): respeta robots.txt, los Términos de Servicio de cada
sitio y las leyes locales. Este código trae SELECTORES DE EJEMPLO: el HTML real
cambia y debes ajustarlos. Prefiere una API oficial/licenciada cuando exista
(API-Football, Sportmonks, Opta) — el parser de abajo es solo el "esqueleto".
"""
from __future__ import annotations

import time
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

import config
from .schemas import MatchData, TeamMatchStats, PlayerMatchStats
from .synthetic import generate_team_matches

# Perfiles de "ADN" para el modo offline (calibra a gusto por equipo real)
_TEAM_PROFILES = {
    "Local":  dict(quality=0.68, style_directness=0.35, press_intensity=0.70, seed=101),
    "Rival":  dict(quality=0.55, style_directness=0.60, press_intensity=0.45, seed=202),
}


# ------------------------------------------------------------------ #
# API pública del módulo
# ------------------------------------------------------------------ #
def fetch_last_matches(team: str, n: int = None) -> List[MatchData]:
    """Punto de entrada único del ETL. Devuelve List[MatchData] normalizada."""
    n = n or config.N_MATCHES

    if config.OFFLINE_MODE:
        profile = _TEAM_PROFILES.get(team, dict(quality=0.5, seed=hash(team) % 999))
        return generate_team_matches(team, n=n, **profile)

    if config.USE_SELENIUM:
        return _fetch_with_selenium(team, n)
    return _fetch_with_requests(team, n)


# ------------------------------------------------------------------ #
# Backend 1: requests + BeautifulSoup (sitios estáticos, p. ej. ESPN)
# ------------------------------------------------------------------ #
def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": config.USER_AGENT})
    return s


def _fetch_with_requests(team: str, n: int) -> List[MatchData]:
    """
    Esqueleto: obtiene la lista de partidos y parsea cada 'match summary'.
    Ajusta URLs y selectores a la fuente concreta.
    """
    sess = _session()
    match_urls = _discover_match_urls(sess, team, n)
    matches: List[MatchData] = []
    for url in match_urls[:n]:
        try:
            html = sess.get(url, timeout=config.REQUEST_TIMEOUT).text
            matches.append(_parse_match_summary(html, team))
        except Exception as exc:  # noqa: BLE001
            print(f"[scraper] fallo en {url}: {exc}")
        time.sleep(config.SCRAPE_DELAY)  # cortesía anti-rate-limit
    return matches


def _discover_match_urls(sess: requests.Session, team: str, n: int) -> List[str]:
    """Devuelve las URLs de los últimos partidos desde la ficha del equipo."""
    # EJEMPLO — reemplaza por la URL real de resultados del equipo
    team_page = f"https://www.espn.com/soccer/team/results/_/id/{team}"
    soup = BeautifulSoup(sess.get(team_page, timeout=config.REQUEST_TIMEOUT).text, "lxml")
    urls = []
    for a in soup.select("a.AnchorLink[href*='/match/']"):  # selector de ejemplo
        href = a.get("href", "")
        if href.startswith("http"):
            urls.append(href)
        elif href:
            urls.append("https://www.espn.com" + href)
    return list(dict.fromkeys(urls))  # dedup preservando orden


def _parse_match_summary(html: str, team: str) -> MatchData:
    """
    Traduce el HTML de un resumen de partido al esquema canónico.
    Los selectores de abajo son PLACEHOLDERS ilustrativos.
    """
    soup = BeautifulSoup(html, "lxml")

    def _txt(css: str, default: str = "0") -> str:
        el = soup.select_one(css)
        return el.get_text(strip=True) if el else default

    ts = TeamMatchStats(
        goals_for=int(_txt(".score.home")),
        goals_against=int(_txt(".score.away")),
        possession=float(_txt(".stat--possession .home").replace("%", "") or 50),
        shots=int(_txt(".stat--shots .home")),
        shots_on_target=int(_txt(".stat--sot .home")),
        # xG rara vez está en ESPN gratis -> se estima luego en el pipeline
    )

    players: List[PlayerMatchStats] = []
    for row in soup.select("table.player-stats tbody tr"):  # placeholder
        cells = [c.get_text(strip=True) for c in row.select("td")]
        if len(cells) < 3:
            continue
        players.append(PlayerMatchStats(
            player_id=cells[0], name=cells[0], position="MID",
            rating=float(cells[-1] or 6.0),
        ))

    return MatchData(
        match_id=_txt("meta[name='match-id']", "unknown"),
        match_date=_txt(".match-date", "1970-01-01"),
        team=team,
        opponent=_txt(".opponent-name", "Rival"),
        is_home=True,
        team_stats=ts,
        players=players,
    )


# ------------------------------------------------------------------ #
# Backend 2: Selenium (sitios con JavaScript, p. ej. Win Sports)
# ------------------------------------------------------------------ #
def _fetch_with_selenium(team: str, n: int) -> List[MatchData]:
    """
    Render dinámico. Requiere selenium + webdriver-manager instalados.
    Se importa perezosamente para no exigir Selenium en modo offline/estático.
    """
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument(f"user-agent={config.USER_AGENT}")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
    try:
        driver.get(f"https://www.winsports.co/equipo/{team}")
        time.sleep(config.SCRAPE_DELAY)
        html = driver.page_source
        # Reutiliza el parser estático sobre el DOM ya renderizado
        return [_parse_match_summary(html, team)]
    finally:
        driver.quit()


if __name__ == "__main__":
    for m in fetch_last_matches("Local"):
        print(m.match_id, m.result, m.team_stats.to_dict()["goals_for"])
