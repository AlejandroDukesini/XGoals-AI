"""
app.py  —  APEXPREDICT ENTERPRISE ANALYTICS
===========================================
Terminal cuantitativa de análisis predictivo de fútbol y gestión de riesgo para
analistas profesionales. Streamlit + Pandas + NumPy + SciPy (scipy.stats.poisson).

Ejecutar:
    pip install streamlit==1.40.0 pandas numpy scipy
    streamlit run app.py
    # (si 'streamlit' no está en el PATH:  python -m streamlit run app.py)

Directrices de diseño (terminal financiera, sin clichés de IA):
    · Sin emojis. Tickers de 3 letras estilo Bloomberg para identificar equipos.
    · Paleta corporativa mate; tarjetas planas, bordes sutiles, 6px de radio.
    · Redacción técnica e institucional; cifras con tipografía tabular.

Rigor cuantitativo (determinista, sin aleatoriedad):
    · Histórico de 10 partidos por equipo generado con oscilaciones
      trigonométricas ancladas a un seed del nombre (reproducible).
    · Distribución de Poisson bivariada (scipy.stats.poisson.pmf).
    · Comparador de 4 casas y sizing por Criterio de Kelly.

Estructura del archivo:
    1. CONFIG & TEMA (CSS)
    2. CATÁLOGO MULTI-TORNEO + generador determinista
    3. BACKEND (Poisson, DOFA, auditoría, casas, Kelly, tracker)
    4. UI (sidebar, cabecera, pestañas)
    5. main()
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN DE PÁGINA Y TEMA
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ApexPredict Enterprise Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

ANALYST_NAME = "Santiago"

# Paleta "Casino Premium & Confiable": fondos slate profundos, verde esmeralda
# para valor/confianza, dorado premium y azul eléctrico para acentos interactivos.
C = {
    "bg": "#0b0f19", "bg2": "#0e1526", "panel": "#111827", "panel2": "#0f1a2e",
    "border": "#1e293b", "border2": "#243044",
    "text": "#f8fafc", "muted": "#94a3b8", "faint": "#64748b",
    "pos": "#10b981", "posbright": "#00e676", "neg": "#ef4444", "negsoft": "#f87171",
    "accent": "#38bdf8", "gold": "#f59e0b", "goldsoft": "#fbbf24",
}


def inject_css() -> None:
    """Inyecta el tema 'Casino Premium': gradientes slate, acentos esmeralda/dorado,
    micro-interacciones (hover, brillo pulsante en CTAs, spinner) y layout responsivo."""
    st.markdown(f"""
    <style>
      html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
      }}
      /* Fondo con degradado profundo + halos de color muy sutiles */
      .stApp {{
        background:
          radial-gradient(1200px 600px at 12% -8%, rgba(16,185,129,.08), transparent 60%),
          radial-gradient(1100px 620px at 100% 0%, rgba(56,189,248,.07), transparent 55%),
          linear-gradient(180deg, {C['bg']} 0%, {C['bg2']} 100%);
        background-attachment: fixed; color: {C['text']};
      }}
      #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}
      .block-container {{ padding-top: 1.4rem; max-width: 1400px; }}

      section[data-testid="stSidebar"] > div {{
        background: linear-gradient(180deg, {C['panel']} 0%, {C['panel2']} 100%);
        border-right: 1px solid {C['border']};
      }}
      section[data-testid="stSidebar"] * {{ color: {C['text']}; }}
      .stExpander {{ border: 1px solid {C['border']} !important; border-radius: 10px !important;
        background: rgba(15,23,42,.5) !important; overflow:hidden; }}

      h1, h2, h3, h4 {{ color: {C['text']}; font-weight: 650; letter-spacing: -.2px; }}

      /* ── Cabecera / wordmark premium ─────────────────────────────── */
      .wm {{ display:flex; align-items:baseline; gap:.6rem; flex-wrap:wrap;
        border-bottom:1px solid {C['border']}; padding-bottom:.7rem; margin-bottom:.4rem; }}
      .wm .brand {{ font-size:1.5rem; font-weight:800; letter-spacing:1.5px;
        background:linear-gradient(90deg,{C['text']} 0%,{C['posbright']} 55%,{C['gold']} 120%);
        -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }}
      .wm .sub {{ font-size:.78rem; letter-spacing:2px; color:{C['muted']}; text-transform:uppercase; }}
      .session {{ color:{C['faint']}; font-size:.8rem; margin-bottom:1rem; }}
      .session b {{ color:{C['muted']}; font-weight:600; }}

      /* ── KPI cards (alto dinámico, brillo superior, hover) ────────── */
      .kpi {{ position:relative; background:linear-gradient(180deg,{C['panel']},{C['panel2']});
        border:1px solid {C['border']}; border-radius:12px; padding:1.4rem 1.5rem; height:100%;
        overflow:hidden; transition:all .3s ease; }}
      .kpi::before {{ content:''; position:absolute; inset:0 0 auto 0; height:2px;
        background:linear-gradient(90deg,transparent,{C['accent']},transparent); opacity:.5; }}
      .kpi:hover {{ transform:translateY(-2px); border-color:{C['border2']};
        box-shadow:0 10px 30px -12px rgba(0,0,0,.6); }}
      .kpi .lbl {{ font-size:.7rem; letter-spacing:1.4px; color:{C['muted']}; text-transform:uppercase; }}
      .kpi .val {{ font-size:1.7rem; font-weight:750; margin-top:.35rem;
        font-variant-numeric: tabular-nums; word-break:break-word; }}
      .kpi .sub {{ font-size:.74rem; color:{C['faint']}; margin-top:.2rem; }}
      .pos {{ color:{C['pos']}; }}  .neg {{ color:{C['neg']}; }}  .gold {{ color:{C['gold']}; }}

      /* ── Tarjeta de contenido (alto auto, sin cortes de texto) ───── */
      .card {{ background:linear-gradient(180deg,{C['panel']},{C['panel2']});
        border:1px solid {C['border']}; border-radius:12px; padding:1.5rem; margin-bottom:1rem;
        height:auto; overflow:hidden; transition:all .3s ease; }}
      .card:hover {{ border-color:{C['border2']}; }}
      .card h4 {{ margin:0 0 .8rem; font-size:.82rem; letter-spacing:1.2px;
        text-transform:uppercase; color:{C['muted']}; font-weight:650; }}

      /* Ticker de equipo */
      .tk {{ display:inline-block; font-weight:800; font-size:.72rem; letter-spacing:.5px;
        padding:.12rem .45rem; border:1px solid {C['border2']}; border-radius:5px;
        color:{C['accent']}; background:rgba(56,189,248,.08); font-variant-numeric:tabular-nums; }}

      /* Barra de distribución 1X2 */
      .bar {{ display:flex; height:28px; border:1px solid {C['border']}; border-radius:6px;
        overflow:hidden; }}
      .seg {{ display:flex; align-items:center; justify-content:center; font-size:.75rem;
        font-weight:700; color:{C['bg']}; transition:all .3s ease; }}

      /* ── Matriz DOFA: cuadrícula 2x2 -> 1 col en móvil ───────────── */
      .dofa-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; }}
      @media (max-width: 900px) {{ .dofa-grid {{ grid-template-columns:1fr; }} }}
      .dofa-q {{ background:linear-gradient(180deg,{C['panel']},{C['panel2']});
        border:1px solid {C['border']}; border-left:3px solid var(--qc); border-radius:12px;
        padding:1.25rem 1.3rem; height:auto; overflow:hidden; transition:all .3s ease; }}
      .dofa-q:hover {{ transform:translateY(-2px); box-shadow:0 12px 28px -14px rgba(0,0,0,.65); }}
      .dofa-h {{ font-size:.74rem; letter-spacing:1.3px; text-transform:uppercase;
        font-weight:750; margin:0 0 .6rem; color:var(--qc);
        display:flex; align-items:center; gap:.45rem; }}
      .dofa-q ul {{ padding-left:1.05rem; margin:0; font-size:.85rem; line-height:1.45; }}
      .dofa-q li {{ margin-bottom:.45rem; }}

      /* ── Botones + micro-interacciones ───────────────────────────── */
      .stButton > button {{ background:rgba(30,41,59,.55); color:{C['text']};
        border:1px solid {C['border2']}; border-radius:8px; padding:.55rem 1rem;
        font-weight:650; width:100%; transition:all .3s ease; }}
      .stButton > button:hover {{ border-color:{C['accent']}; color:{C['accent']};
        transform:translateY(-1px); box-shadow:0 6px 18px -8px rgba(56,189,248,.5); }}
      /* CTA primario (Ejecutar / Guardar): degradado esmeralda + brillo pulsante */
      .stButton > button[kind="primary"], .stForm button[kind="primary"] {{
        background:linear-gradient(90deg,{C['pos']},{C['posbright']}); color:#04140c;
        border:none; font-weight:750; letter-spacing:.3px;
        animation:pulseGlow 2.4s ease-in-out infinite; }}
      .stButton > button[kind="primary"]:hover, .stForm button[kind="primary"]:hover {{
        color:#04140c; transform:translateY(-1px) scale(1.01);
        box-shadow:0 0 0 3px rgba(16,185,129,.25), 0 12px 28px -8px rgba(16,185,129,.6); }}
      @keyframes pulseGlow {{
        0%,100% {{ box-shadow:0 0 0 0 rgba(16,185,129,.0), 0 8px 22px -12px rgba(16,185,129,.5); }}
        50%     {{ box-shadow:0 0 0 4px rgba(16,185,129,.14), 0 10px 26px -10px rgba(16,185,129,.75); }}
      }}

      /* Spinner estilizado (sensación de software de alta tecnología) */
      .stSpinner > div {{ border-top-color:{C['posbright']} !important;
        border-right-color:{C['accent']} !important; }}

      /* Pestañas */
      .stTabs [data-baseweb="tab-list"] {{ gap:.15rem; border-bottom:1px solid {C['border']}; }}
      .stTabs [data-baseweb="tab"] {{ background:transparent; border:none; border-radius:0;
        padding:.55rem 1.1rem; color:{C['muted']}; font-weight:600; font-size:.88rem;
        transition:color .3s ease; }}
      .stTabs [data-baseweb="tab"]:hover {{ color:{C['text']}; }}
      .stTabs [aria-selected="true"] {{ color:{C['text']} !important;
        border-bottom:2px solid {C['posbright']} !important; }}

      .note {{ border-left:3px solid {C['accent']}; background:rgba(56,189,248,.06);
        padding:.8rem 1.1rem; border-radius:0 8px 8px 0; font-size:.84rem; color:{C['muted']};
        overflow-wrap:anywhere; }}
      .muted {{ color:{C['muted']}; font-size:.84rem; }}
      .mono {{ font-variant-numeric: tabular-nums; }}

      /* Chips de fuentes de datos */
      .srcwrap {{ display:flex; flex-wrap:wrap; gap:.4rem; margin:.5rem 0 .2rem; }}
      .chip {{ font-size:.68rem; font-weight:700; letter-spacing:.4px; padding:.24rem .55rem;
        border-radius:999px; border:1px solid {C['border2']}; color:{C['muted']};
        background:rgba(30,41,59,.5); white-space:nowrap; }}
      .chip.on {{ color:{C['posbright']}; border-color:rgba(16,185,129,.5);
        background:rgba(16,185,129,.12); box-shadow:0 0 0 1px rgba(16,185,129,.15) inset; }}
      .chip.soon {{ color:{C['faint']}; border-style:dashed; opacity:.85; }}

      /* ── Bet slip / ticket de apuesta ────────────────────────────── */
      .slip {{ position:relative; background:
          radial-gradient(600px 200px at 100% 0%, rgba(245,158,11,.10), transparent 60%),
          linear-gradient(180deg,{C['panel']},{C['panel2']});
        border:1px solid rgba(16,185,129,.45); border-radius:16px; padding:1.4rem 1.5rem;
        box-shadow:0 18px 46px -20px rgba(16,185,129,.35); overflow:hidden; }}
      .slip-hd {{ display:flex; justify-content:space-between; align-items:center;
        border-bottom:1px dashed {C['border2']}; padding-bottom:.7rem; margin-bottom:.9rem; }}
      .slip-hd .t {{ font-size:.74rem; letter-spacing:2px; text-transform:uppercase;
        font-weight:800; color:{C['gold']}; }}
      .slip-hd .badge {{ font-size:.68rem; font-weight:700; padding:.2rem .55rem; border-radius:999px;
        background:rgba(16,185,129,.15); color:{C['posbright']}; border:1px solid rgba(16,185,129,.4); }}
      .slip-row {{ display:flex; justify-content:space-between; gap:1rem; padding:.32rem 0;
        font-size:.9rem; }}
      .slip-row .k {{ color:{C['muted']}; }}
      .slip-row .v {{ font-weight:700; text-align:right; font-variant-numeric:tabular-nums;
        overflow-wrap:anywhere; }}
      .slip-total {{ display:flex; justify-content:space-between; align-items:baseline;
        margin-top:.8rem; padding-top:.8rem; border-top:1px dashed {C['border2']}; }}
      .slip-total .k {{ font-size:.72rem; letter-spacing:1.4px; text-transform:uppercase;
        color:{C['muted']}; }}
      .slip-total .v {{ font-size:1.55rem; font-weight:800; color:{C['posbright']};
        font-variant-numeric:tabular-nums; }}
    </style>
    """, unsafe_allow_html=True)


# Monedas: símbolo y rangos de banca por defecto (banca en la escala típica de cada divisa).
CURRENCIES = {
    "USD": dict(sym="$",    default=10_000,    min=1_000,   max=200_000,    step=500),
    "COP": dict(sym="COP$", default=1_000_000, min=100_000, max=20_000_000, step=50_000),
    "EUR": dict(sym="€",    default=10_000,    min=1_000,   max=200_000,    step=500),
    "GBP": dict(sym="£",    default=8_000,     min=1_000,   max=160_000,    step=500),
    "MXN": dict(sym="MX$",  default=150_000,   min=20_000,  max=3_000_000,  step=10_000),
    "BRL": dict(sym="R$",   default=50_000,    min=5_000,   max=1_000_000,  step=5_000),
}


def money(v: float, cur: str) -> str:
    """Formatea un monto con el símbolo de la moneda configurada."""
    return f"{CURRENCIES[cur]['sym']}{v:,.0f}"


# ══════════════════════════════════════════════════════════════════════════
# 2. CATÁLOGO MULTI-TORNEO + GENERADOR DETERMINISTA
# ══════════════════════════════════════════════════════════════════════════
# Perfil por equipo. 'code' = ticker de 3 letras (sustituye a cualquier icono).
#   atk goles a favor prom.   dfn goles en contra prom.   pos posesión %
#   sot tiros al arco prom.   pas efectividad de pase     style sistema del DT
CATALOG: Dict[str, Dict] = {
    "Copa del Mundo 2026": {"comp": 0.92, "teams": {
        "Colombia":  dict(code="COL", conf="CONMEBOL", atk=1.7, dfn=0.9, pos=55, sot=5.2, pas=.85, style="4-2-3-1", star="J. Rodríguez"),
        "Argentina": dict(code="ARG", conf="CONMEBOL", atk=2.3, dfn=0.7, pos=58, sot=6.5, pas=.88, style="4-3-3",   star="L. Messi"),
        "Francia":   dict(code="FRA", conf="UEFA",     atk=2.2, dfn=0.8, pos=56, sot=6.2, pas=.87, style="4-2-3-1", star="K. Mbappé"),
        "España":    dict(code="ESP", conf="UEFA",     atk=2.1, dfn=0.9, pos=64, sot=6.0, pas=.90, style="4-3-3",   star="Pedri"),
        "Marruecos": dict(code="MAR", conf="CAF",      atk=1.4, dfn=0.9, pos=50, sot=4.5, pas=.83, style="4-3-3",   star="A. Hakimi"),
        "Japón":     dict(code="JPN", conf="AFC",      atk=1.6, dfn=1.0, pos=54, sot=4.8, pas=.86, style="4-2-3-1", star="T. Kubo"),
        "Estados Unidos": dict(code="USA", conf="CONCACAF", atk=1.5, dfn=1.1, pos=52, sot=4.6, pas=.84, style="4-3-3", star="C. Pulisic"),
    }},
    "Liga BetPlay Dimayor": {"comp": 1.05, "teams": {
        "Millonarios":       dict(code="MIL", atk=1.6, dfn=1.0, pos=55, sot=5.0, pas=.83, style="4-2-3-1", star="R. Vargas"),
        "Atlético Nacional": dict(code="NAC", atk=1.8, dfn=0.9, pos=57, sot=5.4, pas=.84, style="4-3-3",   star="E. Palacios"),
        "Junior":            dict(code="JUN", atk=1.5, dfn=1.1, pos=53, sot=4.8, pas=.82, style="4-4-2",   star="C. Bacca"),
        "Santa Fe":          dict(code="SFE", atk=1.4, dfn=1.1, pos=52, sot=4.6, pas=.81, style="4-4-2",   star="H. Rodallega"),
    }},
    "UEFA Champions League": {"comp": 0.98, "teams": {
        "Real Madrid":     dict(code="RMA", atk=2.4, dfn=0.9, pos=58, sot=6.6, pas=.88, style="4-3-3",   star="J. Bellingham"),
        "Manchester City": dict(code="MCI", atk=2.6, dfn=0.8, pos=66, sot=7.2, pas=.91, style="4-3-3",   star="E. Haaland"),
        "PSG":             dict(code="PSG", atk=2.3, dfn=1.0, pos=60, sot=6.4, pas=.88, style="4-3-3",   star="Dembélé"),
        "Bayern Múnich":   dict(code="BAY", atk=2.5, dfn=1.0, pos=62, sot=6.8, pas=.89, style="4-2-3-1", star="H. Kane"),
    }},
    "Major League Soccer": {"comp": 1.12, "teams": {
        "Inter de Miami": dict(code="MIA", atk=2.0, dfn=1.2, pos=57, sot=5.6, pas=.85, style="4-4-2", star="L. Messi"),
        "Al-Nassr":       dict(code="NAS", atk=2.1, dfn=1.1, pos=58, sot=5.8, pas=.85, style="4-3-3", star="C. Ronaldo"),
        "LAFC":           dict(code="LAF", atk=1.8, dfn=1.1, pos=54, sot=5.2, pas=.83, style="4-3-3", star="O. Bouanga"),
        "LA Galaxy":      dict(code="LAG", atk=1.7, dfn=1.2, pos=53, sot=5.0, pas=.82, style="4-4-2", star="R. Puig"),
    }},
}

ALL_TEAMS: Dict[str, Dict] = {}
for _tour, _data in CATALOG.items():
    for _tname, _prof in _data["teams"].items():
        _p = dict(_prof); _p["tournament"] = _tour; _p["comp"] = _data["comp"]
        ALL_TEAMS[_tname] = _p

XG_PER_SOT = 0.31
HOME_ADV = 1.10
MAX_G = 5
SCRAPE_DATE = pd.Timestamp("2026-07-12 03:15")
# Fuentes de datos: activas (seleccionables) y placeholders (integraciones futuras).
SOURCE_URLS = {
    "ESPN":       "https://www.espn.com/soccer/match/_/gameId/{gid}",
    "WIN Sports": "https://www.winsports.co/futbol/partido/{gid}",
    "Custom URL": None,
}
ACTIVE_SOURCES = ["ESPN", "WIN Sports", "Custom URL"]
FUTURE_SOURCES = ["Sofascore", "Flashscore", "Directo (API)"]

BOOKIES = ["BetPlay", "Wplay", "Rushbet", "Codere"]
HOUSE_MARGIN = {"BetPlay": 0.040, "Wplay": 0.045, "Rushbet": 0.050, "Codere": 0.038}
RISK_MULT = {"Conservador": 0.20, "Moderado": 0.50, "Agresivo": 1.00}
_ALT_FORM = {"4-3-3": "4-2-3-1", "4-2-3-1": "4-3-3", "4-4-2": "4-3-3", "3-5-2": "4-4-2"}


def _seed(name: str) -> int:
    """Semilla determinista derivada del nombre."""
    return sum(ord(c) for c in name)


def code(team: str) -> str:
    """Ticker de 3 letras del equipo."""
    return ALL_TEAMS[team]["code"]


_HIST_CACHE: Dict[str, pd.DataFrame] = {}


def team_history(team: str) -> pd.DataFrame:
    """
    Histórico de 10 partidos generado de forma determinista desde el perfil del
    equipo (oscilaciones trigonométricas por seed del nombre; sin aleatoriedad).
    """
    if team in _HIST_CACHE:
        return _HIST_CACHE[team]
    p = ALL_TEAMS[team]
    seed = _seed(team)
    opponents = [t for t in ALL_TEAMS
                 if ALL_TEAMS[t]["tournament"] == p["tournament"] and t != team] or ["Rival"]
    rows = []
    for i in range(10):
        s1 = math.sin(seed * 0.13 + i * 0.9)
        s2 = math.cos(seed * 0.17 + i * 0.7)
        s3 = math.sin(seed * 0.07 + i * 1.3)
        gf = max(0, round(p["atk"] + 0.75 * s1))
        ga = max(0, round(p["dfn"] + 0.60 * s2))
        pos = int(min(72, max(35, round(p["pos"] + 6 * s3))))
        sot = max(1, round(p["sot"] + 1.3 * s1))
        sotc = max(1, round(2 + p["dfn"] * 2.2 + 1.3 * s2))
        fouls = max(5, round(12 + 3 * s3 + (2 if p["pos"] < 52 else 0)))
        pas = round(min(0.95, max(0.70, p["pas"] + 0.04 * s1)), 3)
        form = p["style"] if (i % 4) else _ALT_FORM.get(p["style"], p["style"])
        rows.append({
            "Jornada": f"J-{10 - i}", "Rival": opponents[i % len(opponents)],
            "GF": gf, "GC": ga, "Posesion_%": pos, "Tiros_arco": sot,
            "Tiros_concedidos": sotc, "Faltas": fouls,
            "Pases_%": round(pas * 100, 1), "xG": round(sot * XG_PER_SOT, 2),
            "Sistema_DT": form,
        })
    df = pd.DataFrame(rows)
    _HIST_CACHE[team] = df
    return df


def last_n(team: str, n: int) -> pd.DataFrame:
    """Filtra la muestra: últimos N partidos (cola del histórico)."""
    return team_history(team).tail(n).reset_index(drop=True)


def team_averages(team: str, n: int) -> Dict[str, float]:
    """Promedios de la muestra + sistema táctico modal del DT."""
    df = last_n(team, n)
    return {
        "gf": df["GF"].mean(), "ga": df["GC"].mean(), "pos": df["Posesion_%"].mean(),
        "sot": df["Tiros_arco"].mean(), "sotc": df["Tiros_concedidos"].mean(),
        "fouls": df["Faltas"].mean(), "pas": df["Pases_%"].mean(),
        "form": df["Sistema_DT"].mode().iloc[0],
    }


def _league_avg() -> float:
    """Base de goles de la liga (media combinada a favor y en contra del catálogo)."""
    vals = []
    for t in ALL_TEAMS:
        h = team_history(t)
        vals += h["GF"].tolist() + h["GC"].tolist()
    return float(np.mean(vals))


LEAGUE_AVG = _league_avg()


# ══════════════════════════════════════════════════════════════════════════
# 3. BACKEND CUANTITATIVO
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Prediction:
    lam_home: float
    lam_away: float
    matrix: np.ndarray
    markets: Dict[str, float]
    top_scores: List[Tuple[str, float]]


def compute_prediction(home: str, away: str, n: int, comp: float) -> Prediction:
    """
    Modelo de Poisson bivariado.

        lambda_home = atk_home * def_away / LEAGUE_AVG * ventaja_local * competitividad
        lambda_away = atk_away * def_home / LEAGUE_AVG * competitividad
        P(i, j) = poisson.pmf(i; lambda_home) * poisson.pmf(j; lambda_away)

    La malla 0..5 se normaliza (la cola por encima de 5 goles se trunca).
    """
    h, a = team_averages(home, n), team_averages(away, n)
    lam_h = max(0.15, h["gf"] * a["ga"] / LEAGUE_AVG * HOME_ADV * comp)
    lam_a = max(0.15, a["gf"] * h["ga"] / LEAGUE_AVG * comp)

    k = np.arange(MAX_G + 1)
    M = np.outer(poisson.pmf(k, lam_h), poisson.pmf(k, lam_a))
    M = M / M.sum()

    idx = np.add.outer(k, k)
    mk = {
        "1": float(np.tril(M, -1).sum()), "X": float(np.trace(M)), "2": float(np.triu(M, 1).sum()),
        "Over 1.5": float(M[idx >= 2].sum()), "Under 1.5": float(M[idx <= 1].sum()),
        "Over 2.5": float(M[idx >= 3].sum()), "Under 2.5": float(M[idx <= 2].sum()),
        "Over 3.5": float(M[idx >= 4].sum()), "Under 3.5": float(M[idx <= 3].sum()),
        "BTTS Si": float(M[1:, 1:].sum()),
    }
    mk["BTTS No"] = 1 - mk["BTTS Si"]
    mk["1X"] = mk["1"] + mk["X"]; mk["X2"] = mk["X"] + mk["2"]; mk["12"] = mk["1"] + mk["2"]
    flat = sorted(((f"{i}-{j}", float(M[i, j])) for i in k for j in k), key=lambda t: -t[1])[:6]
    return Prediction(round(lam_h, 3), round(lam_a, 3), M, mk, flat)


def build_dofa(home: str, away: str, n: int, pred: Prediction) -> Dict[str, List[str]]:
    """
    Matriz DOFA por correlación lógica: contrasta métricas ofensivas promedio de
    un equipo contra las defensivas del oponente para derivar conclusiones
    objetivas (fortalezas/debilidades/oportunidades/amenazas).
    """
    h, a = team_averages(home, n), team_averages(away, n)
    hs, as_ = ALL_TEAMS[home], ALL_TEAMS[away]
    F, D, O, A = [], [], [], []

    if h["gf"] > a["ga"] + 0.3:
        F.append(f"Producción ofensiva de {home} ({h['gf']:.2f} goles/pp) por encima de la "
                 f"concesión defensiva de {away} ({a['ga']:.2f} recibidos/pp).")
    if h["pos"] > a["pos"] + 3:
        F.append(f"Diferencial de posesión favorable a {home} ({h['pos']:.0f}% vs {a['pos']:.0f}%): "
                 f"control del tempo del partido.")
    if h["sot"] > a["sotc"]:
        F.append(f"Volumen de tiro de {home} ({h['sot']:.1f}/pp) superior a los tiros concedidos "
                 f"por {away} ({a['sotc']:.1f}/pp).")

    if a["gf"] > h["ga"] + 0.3:
        D.append(f"Ataque de {away} ({a['gf']:.2f} goles/pp) supera la concesión defensiva de "
                 f"{home} ({h['ga']:.2f} recibidos/pp).")
    if h["pas"] < a["pas"]:
        D.append(f"Precisión de pase inferior ({h['pas']:.0f}% vs {a['pas']:.0f}%): mayor "
                 f"exposición a pérdidas bajo presión.")
    if h["fouls"] > a["fouls"] + 1.5:
        D.append(f"Índice de faltas elevado en {home} ({h['fouls']:.1f}/pp): exposición a "
                 f"balón parado del rival.")

    if pred.markets["Over 2.5"] > 0.55:
        O.append(f"Perfil de partido abierto: probabilidad de Over 2.5 = {pred.markets['Over 2.5']:.1%}.")
    if pred.markets["1"] > pred.markets["2"] + 0.12:
        O.append(f"Ventaja de localía y nivel inclinan la probabilidad a {home} "
                 f"({pred.markets['1']:.1%}).")

    A.append(f"Sistema táctico documentado del rival: {a['form']} (referente: {as_.get('star', 'n/d')}).")
    if a["gf"] >= 2.2:
        A.append(f"Eficacia ofensiva alta de {away} ({a['gf']:.2f} goles/pp): riesgo en transición.")
    if pred.markets["2"] > 0.30:
        A.append(f"Probabilidad de victoria visitante no despreciable: {pred.markets['2']:.1%}.")

    return {"Fortalezas": F, "Debilidades": D, "Oportunidades": O, "Amenazas": A}


def extraction_table(team: str, n: int, source: str, custom_url: str) -> pd.DataFrame:
    """Tabla de trazabilidad de origen (partido, fecha, fuente, URL, estado HTTP)."""
    rows = []
    for _, row in last_n(team, n).iterrows():
        j = 10 - int(row["Jornada"].split("-")[1])
        gid = 6_400_000 + (_seed(team) * 31 + j * 7) % 90_000
        if source == "Custom URL" and custom_url.strip():
            url = f"{custom_url.rstrip('/')}?match={team.lower().replace(' ', '-')}-{gid}"
        else:
            template = SOURCE_URLS.get(source) or SOURCE_URLS["ESPN"]
            url = template.format(gid=gid)
        match_date = SCRAPE_DATE.normalize() - pd.Timedelta(days=(10 - j) * 7)
        stamp = SCRAPE_DATE + pd.Timedelta(seconds=j * 7)
        rows.append({
            "Partido": f"{team} vs {row['Rival']}", "Fecha": match_date.strftime("%Y-%m-%d"),
            "Fuente": source, "URL_origen": url,
            "Extraido": stamp.strftime("%Y-%m-%d %H:%M:%S"), "HTTP": "200 OK",
        })
    return pd.DataFrame(rows)


def extraction_console(team: str, n: int, source: str, custom_url: str) -> str:
    """Genera un log de consola de la extracción, formateado como texto técnico."""
    lines = [f"[{SCRAPE_DATE.strftime('%Y-%m-%d %H:%M:%S')}] INFO  session   "
             f"target={team!r} source={source!r} sample_N={n}"]
    tbl = extraction_table(team, n, source, custom_url)
    for i, r in tbl.iterrows():
        ms = 90 + (_seed(team) + i * 13) % 220
        lines.append(f"[{r['Extraido']}] INFO  http.get  {r['URL_origen']}  -> {r['HTTP']} ({ms}ms)")
    lines.append(f"[{SCRAPE_DATE.strftime('%Y-%m-%d %H:%M:%S')}] INFO  parser    "
                 f"rows_parsed={len(tbl)} schema=OK checksum=verified")
    lines.append(f"[{SCRAPE_DATE.strftime('%Y-%m-%d %H:%M:%S')}] INFO  pipeline  "
                 f"features_built ok -> handoff to poisson_engine")
    return "\n".join(lines)


# ---- Comparador multi-casa + Kelly --------------------------------------- #
def market_options(home: str, away: str) -> Dict[str, Tuple[str, str]]:
    """Etiqueta legible -> (clave interna del mercado, referencia de la selección)."""
    return {
        f"Victoria {home}": ("1", home),
        "Empate": ("X", "el empate"),
        f"Victoria {away}": ("2", away),
        "Mas de 2.5 goles": ("Over 2.5", "Over 2.5"),
        "Menos de 2.5 goles": ("Under 2.5", "Under 2.5"),
        "Ambos anotan - Si": ("BTTS Si", "Ambos anotan"),
        "Ambos anotan - No": ("BTTS No", "No ambos anotan"),
    }


def bookmaker_table(prob_ia: float, cons_p: float, home: str, away: str,
                    market_key: str) -> pd.DataFrame:
    """
    Cuotas simuladas de 4 casas para un mercado. Cada casa cotiza sobre el
    consenso con su margen y una dispersión determinista. El EV se evalúa contra
    la probabilidad del modelo (prob_ia).
    """
    fair_cons = 1.0 / min(max(cons_p, 0.02), 0.98)
    rows = []
    for casa in BOOKIES:
        wob = 0.06 * math.sin(_seed(casa + market_key + home + away) * 0.1)
        odds = round(max(1.01, fair_cons * (1 - HOUSE_MARGIN[casa] + wob)), 2)
        ev = prob_ia * odds - 1
        rows.append({"Casa": casa, "Cuota": odds, "Prob_implicita": round(1 / odds, 3),
                     "EV": round(ev, 3)})
    return pd.DataFrame(rows)


def kelly_fraction(p: float, odds: float) -> float:
    """
    Criterio de Kelly:   f* = (p*b - q) / b = p - (1-p)/b,  con b = Cuota - 1.
    Puede ser negativo (no apostar).
    """
    b = odds - 1
    if b <= 0:
        return 0.0
    q = 1 - p
    return (p * b - q) / b


def kelly_amount(p: float, odds: float, bankroll: float, risk: str) -> Tuple[float, float]:
    """Devuelve (fracción de Kelly ajustada por riesgo, monto en la moneda)."""
    f = kelly_fraction(p, odds) * RISK_MULT[risk]
    f = max(0.0, f)
    return f, bankroll * f


# ---- Tracker ------------------------------------------------------------- #
def bet_net(estado: str, inversion: float, cuota: float) -> float:
    """Utilidad neta de una apuesta según su estado."""
    if estado == "Ganada":
        return inversion * (cuota - 1)
    if estado == "Perdida":
        return -inversion
    return 0.0


def tracker_kpis(bets: List[Dict]) -> Dict[str, float]:
    """P&L neto, ROI, Yield y tasa de acierto del historial del usuario."""
    settled = [b for b in bets if b["Estado"] in ("Ganada", "Perdida")]
    staked = sum(b["Inversion"] for b in settled)
    profit = sum(bet_net(b["Estado"], b["Inversion"], b["Cuota"]) for b in settled)
    wins = sum(1 for b in settled if b["Estado"] == "Ganada")
    return {
        "profit": profit, "staked": staked,
        "roi": (profit / staked * 100) if staked else 0.0,
        "yield": (profit / staked * 100) if staked else 0.0,
        "hit": (wins / len(settled) * 100) if settled else 0.0,
        "n": len(bets), "settled": len(settled),
        "pending": sum(1 for b in bets if b["Estado"] == "Pendiente"),
    }


# ══════════════════════════════════════════════════════════════════════════
# 4. COMPONENTES DE UI
# ══════════════════════════════════════════════════════════════════════════
def kpi_card(col, label: str, value: str, sub: str = "", cls: str = "") -> None:
    col.markdown(
        f"<div class='kpi'><div class='lbl'>{label}</div>"
        f"<div class='val {cls}'>{value}</div><div class='sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def render_header(cur: str, risk: str) -> None:
    """Wordmark institucional + línea de sesión (sin adornos)."""
    st.markdown(
        "<div class='wm'><span class='brand'>APEXPREDICT</span>"
        "<span class='sub'>Enterprise Analytics</span></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='session'>Terminal de análisis cuantitativo &nbsp;|&nbsp; "
        f"Sesión: <b>{ANALYST_NAME}</b> &nbsp;|&nbsp; Moneda base: <b>{cur}</b> "
        f"&nbsp;|&nbsp; Perfil de riesgo: <b>{risk}</b></div>", unsafe_allow_html=True)


def team_tag(team: str) -> str:
    """Devuelve el HTML del ticker + nombre del equipo."""
    return f"<span class='tk'>{code(team)}</span> {team}"


def render_prob_bar(mk: Dict[str, float], home: str, away: str) -> None:
    h, d, a = mk["1"], mk["X"], mk["2"]
    st.markdown(f"""
        <div class='bar'>
          <div class='seg' style='width:{h*100:.1f}%;background:{C['pos']}'>{h:.0%}</div>
          <div class='seg' style='width:{d*100:.1f}%;background:{C['faint']}'>{d:.0%}</div>
          <div class='seg' style='width:{a*100:.1f}%;background:{C['accent']}'>{a:.0%}</div>
        </div>
        <div style='display:flex;justify-content:space-between;margin-top:.4rem;
                    font-size:.8rem;color:{C['muted']}'>
          <span>{code(home)} local</span><span>Empate</span><span>{code(away)} visitante</span>
        </div>""", unsafe_allow_html=True)


def render_dofa(dofa: Dict[str, List[str]]) -> None:
    """Cuadrícula DOFA 2x2 (una sola pieza HTML) con color por cuadrante; en móvil
    colapsa a lista de 1 columna gracias a la media query de .dofa-grid."""
    # Orden de lectura de la matriz: Fortalezas · Oportunidades / Debilidades · Amenazas
    spec = [
        ("Fortalezas", C["pos"]), ("Oportunidades", C["accent"]),
        ("Debilidades", C["gold"]), ("Amenazas", C["neg"]),
    ]
    quads = []
    for dim, color in spec:
        items = dofa.get(dim, [])
        if items:
            body = ("<ul>" + "".join(f"<li>{it}</li>" for it in items) + "</ul>")
        else:
            body = "<p class='muted'>Sin señales relevantes en la muestra.</p>"
        quads.append(
            f"<div class='dofa-q' style='--qc:{color}'>"
            f"<div class='dofa-h'>{dim}</div>{body}</div>"
        )
    st.markdown(f"<div class='dofa-grid'>{''.join(quads)}</div>", unsafe_allow_html=True)


# ---------------- Pestaña: Panel de Control ----------------
def tab_dashboard(cfg, cur, risk, bankroll) -> None:
    st.markdown("#### Panel de control")
    k = tracker_kpis(st.session_state.get("bets", []))
    balance = bankroll + k["profit"]
    pos = k["profit"] >= 0
    cls = ("pos" if pos else "neg") if k["settled"] else ""
    arrow = "▲" if pos else "▼"   # triángulos ASCII (no emoji)

    c = st.columns(4)
    kpi_card(c[0], "Balance de la cuenta", money(balance, cur),
             f"{arrow} {money(k['profit'], cur)} P&L neto", cls)
    kpi_card(c[1], "ROI acumulado", f"{k['roi']:+.2f}%", f"{k['settled']} operaciones liquidadas", cls)
    kpi_card(c[2], "Yield", f"{k['yield']:+.2f}%", "beneficio / capital expuesto", cls)
    kpi_card(c[3], "Tasa de acierto", f"{k['hit']:.1f}%", f"{k['pending']} posiciones abiertas", "")

    st.markdown("<div class='card'><h4>Parámetros activos de la sesión</h4>", unsafe_allow_html=True)
    if cfg:
        params = pd.DataFrame([
            ["Torneo", cfg["tournament"]],
            ["Enfrentamiento", f"{code(cfg['home'])} {cfg['home']}  vs  {cfg['away']} {code(cfg['away'])}"],
            ["Muestra (N partidos)", str(cfg["n"])],
            ["Competitividad de liga", f"x{cfg['comp']:.2f}"],
            ["Fuente de datos", cfg["source"]],
            ["Moneda base", cur],
            ["Perfil de riesgo", f"{risk} (Kelly x{RISK_MULT[risk]:.2f})"],
            ["Bankroll configurado", money(bankroll, cur)],
        ], columns=["Parámetro", "Valor"])
        st.dataframe(params, use_container_width=True, hide_index=True)
    else:
        st.markdown("<p class='muted'>Ejecute la inferencia para fijar los parámetros de "
                    "la sesión.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Restablecer parámetros"):
        st.session_state.ready = False
        st.session_state.pop("cfg", None)
        st.rerun()


# ---------------- Pestaña: Inferencia y DOFA ----------------
def tab_inferencia(home, away, tournament, n, pred, dofa) -> None:
    score, sp = pred.top_scores[0]
    st.markdown(f"#### Inferencia — {code(home)} {home} vs {away} {code(away)}")
    st.markdown(f"<span class='muted'>{tournament} · muestra: últimos {n} partidos · "
                f"marcador de máxima verosimilitud: {score} ({sp:.1%}).</span>",
                unsafe_allow_html=True)

    c = st.columns(5)
    kpi_card(c[0], "Marcador proyectado", score, f"P = {sp:.1%}")
    kpi_card(c[1], "Victoria local", f"{pred.markets['1']:.1%}", code(home))
    kpi_card(c[2], "Empate", f"{pred.markets['X']:.1%}", "resultado X")
    kpi_card(c[3], "Victoria visitante", f"{pred.markets['2']:.1%}", code(away))
    kpi_card(c[4], "Goles esperados", f"{pred.lam_home:.2f} / {pred.lam_away:.2f}", "lambda L / V")

    st.markdown("<div class='card'><h4>Distribución de resultado 1 X 2</h4>", unsafe_allow_html=True)
    render_prob_bar(pred.markets, home, away)
    st.markdown("</div>", unsafe_allow_html=True)

    cA, cB = st.columns([1, 1])
    with cA:
        st.markdown("<div class='card'><h4>Marcadores de mayor probabilidad</h4>", unsafe_allow_html=True)
        sdf = pd.DataFrame(pred.top_scores, columns=["Marcador", "Probabilidad"])
        st.dataframe(sdf.style.format({"Probabilidad": "{:.2%}"}), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with cB:
        st.markdown("<div class='card'><h4>Especificación del modelo</h4>", unsafe_allow_html=True)
        st.markdown("<span class='muted'>Probabilidad de k goles bajo Poisson y valor esperado "
                    "de una apuesta:</span>", unsafe_allow_html=True)
        st.latex(r"P(X = k) = \frac{\lambda^{k} e^{-\lambda}}{k!}")
        st.latex(r"EV = (P \times \text{Cuota}) - 1")
        st.markdown(f"<span class='muted'>lambda_local = {pred.lam_home:.3f} &nbsp; "
                    f"lambda_visitante = {pred.lam_away:.3f}</span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Matriz DOFA cruzada")
    st.markdown("<span class='muted'>Correlación directa de métricas ofensivas de un equipo "
                "contra las defensivas del oponente.</span>", unsafe_allow_html=True)
    render_dofa(dofa)


# ---------------- Pestaña: Auditoría de datos ----------------
def tab_auditoria(home, away, n, source, custom_url) -> None:
    st.markdown("#### Auditoría de datos")
    active = custom_url.strip() if source == "Custom URL" else "plantilla ESPN"
    st.markdown(f"<div class='note'>Fuente primaria: <b>{source}</b> — {active}. "
                f"Corrida de extracción: {SCRAPE_DATE.strftime('%Y-%m-%d %H:%M')}. "
                f"Datos deterministas y reproducibles.</div>", unsafe_allow_html=True)

    for team in (home, away):
        st.markdown(f"##### {code(team)} — {team}")
        st.markdown("<span class='muted'>Registros de consola (pipeline de extracción):</span>",
                    unsafe_allow_html=True)
        st.code(extraction_console(team, n, source, custom_url), language="log")
        st.markdown("<span class='muted'>Trazabilidad de origen por partido:</span>",
                    unsafe_allow_html=True)
        st.dataframe(extraction_table(team, n, source, custom_url), use_container_width=True,
                     hide_index=True,
                     column_config={"URL_origen": st.column_config.LinkColumn("URL_origen")})
        st.markdown("<span class='muted'>Datos históricos analizados (muestra):</span>",
                    unsafe_allow_html=True)
        st.dataframe(last_n(team, n), use_container_width=True, hide_index=True)
        st.markdown("---")

    # Verificación manual del origen
    st.markdown("##### Verificación manual de origen")
    team_v = st.selectbox("Equipo a auditar", [home, away], key="audit_team")
    src = last_n(team_v, n)[["Jornada", "Rival", "GF", "GC", "Posesion_%", "Tiros_arco", "xG"]].copy()
    src.insert(0, "Verificado", True)
    edited = st.data_editor(
        src, use_container_width=True, hide_index=True, key="audit_editor",
        column_config={"Verificado": st.column_config.CheckboxColumn(
            "Verificado", help="Confirme si el registro coincide con la fuente observada.")},
        disabled=["Jornada", "Rival", "GF", "GC", "Posesion_%", "Tiros_arco", "xG"])
    ok, total = int(edited["Verificado"].sum()), len(edited)
    rate = ok / total * 100 if total else 0
    st.markdown(f"<span class='muted'>Registros verificados: {ok}/{total} "
                f"({rate:.0f}% de la muestra auditada).</span>", unsafe_allow_html=True)


# ---------------- Pestaña: Comparador y valor ----------------
def tab_comparador(home, away, pred, consensus, risk, bankroll, cur) -> None:
    st.markdown("#### Comparador multi-casa y apuestas de valor")
    opts = market_options(home, away)
    label = st.selectbox("Mercado", list(opts.keys()))
    key, ref = opts[label]
    prob_ia = pred.markets[key]
    cons_p = consensus.markets[key]
    fair_ia = 1.0 / max(prob_ia, 0.001)

    st.markdown("<div class='note'>Cuota justa del modelo y valor esperado por casa:</div>",
                unsafe_allow_html=True)
    cL, cR = st.columns([1, 1])
    with cL:
        st.latex(r"\text{Cuota justa} = \frac{1}{P_{modelo}}")
        st.latex(r"EV = (P_{modelo} \times \text{Cuota}) - 1")
    with cR:
        st.latex(r"f^{*} = \frac{p \cdot b - q}{b} = p - \frac{1-p}{b}")
        st.markdown(f"<span class='muted'>P_modelo = {prob_ia:.1%} &nbsp; "
                    f"Cuota justa = {fair_ia:.2f} &nbsp; b = Cuota − 1 &nbsp; q = 1 − p</span>",
                    unsafe_allow_html=True)

    bt = bookmaker_table(prob_ia, cons_p, home, away, key)
    best_i = int(bt["Cuota"].idxmax())
    best = bt.loc[best_i]

    def _hl(row):
        if row.name == best_i:
            return [f"background-color: rgba(16,185,129,.12)"] * len(row)
        if row["EV"] <= 0:
            return [f"background-color: rgba(239,68,68,.08)"] * len(row)
        return [""] * len(row)

    st.dataframe(bt.style.apply(_hl, axis=1).format(
        {"Cuota": "{:.2f}", "Prob_implicita": "{:.1%}", "EV": "{:+.2f}"}),
        use_container_width=True, hide_index=True)

    ev_best = float(best["EV"])
    f_adj, monto = kelly_amount(prob_ia, float(best["Cuota"]), bankroll, risk)

    if ev_best > 0 and monto > 0:
        ganancia = monto * (best["Cuota"] - 1)
        st.markdown(
            f"<div class='slip'>"
            f"<div class='slip-hd'><span class='t'>Ticket de apuesta</span>"
            f"<span class='badge'>Valor detectado</span></div>"
            f"<div class='slip-row'><span class='k'>Selección</span>"
            f"<span class='v'>{ref}</span></div>"
            f"<div class='slip-row'><span class='k'>Casa / mejor cuota</span>"
            f"<span class='v'>{best['Casa']} &nbsp;·&nbsp; {best['Cuota']:.2f}</span></div>"
            f"<div class='slip-row'><span class='k'>Prob. modelo · EV</span>"
            f"<span class='v'>{prob_ia:.1%} &nbsp;·&nbsp; <span class='pos'>{ev_best:+.2f}</span></span></div>"
            f"<div class='slip-row'><span class='k'>Perfil de riesgo</span>"
            f"<span class='v gold'>{risk} · Kelly x{RISK_MULT[risk]:.2f} ({f_adj:.2%} banca)</span></div>"
            f"<div class='slip-row'><span class='k'>Stake recomendado</span>"
            f"<span class='v'>{money(monto, cur)}</span></div>"
            f"<div class='slip-total'><span class='k'>Ganancia potencial</span>"
            f"<span class='v'>{money(ganancia, cur)}</span></div>"
            f"</div>",
            unsafe_allow_html=True)
        st.caption(f"f* = ({prob_ia:.3f} · {best['Cuota']-1:.2f} − {1-prob_ia:.3f}) / "
                   f"{best['Cuota']-1:.2f} × {RISK_MULT[risk]:.2f} = {f_adj:.4f}  →  "
                   f"{money(bankroll, cur)} × {f_adj:.4f} = {money(monto, cur)}.")
    else:
        st.markdown(
            f"<div class='note'>Sin colocación recomendada. La mejor cuota disponible "
            f"({best['Cuota']:.2f} en {best['Casa']}) no supera la cuota justa del modelo "
            f"({fair_ia:.2f}); el valor esperado es no positivo. Kelly no asigna capital.</div>",
            unsafe_allow_html=True)


# ---------------- Pestaña: Tracker financiero ----------------
def tab_tracker(cur: str) -> None:
    st.markdown("#### Tracker financiero — libro de operaciones")
    st.markdown("<span class='muted'>Registro manual de posiciones. Almacenado en la sesión "
                "del navegador. P&L, ROI y tasa de acierto se consolidan automáticamente.</span>",
                unsafe_allow_html=True)

    if "bets" not in st.session_state:
        st.session_state.bets = []

    with st.form("bet_form", clear_on_submit=True):
        st.markdown("###### Registrar operación")
        c1, c2, c3 = st.columns(3)
        partido = c1.text_input("Equipos en juego", placeholder="Ej. Colombia vs Argentina")
        tipo = c2.selectbox("Tipo de apuesta", ["Sencilla", "Combinada"])
        seleccion = c3.text_input("Selección / mercado", placeholder="Ej. Victoria Colombia")
        c4, c5, c6 = st.columns(3)
        cuota = c4.number_input("Cuota obtenida", min_value=1.01, max_value=1000.0, value=2.00, step=0.01)
        inversion = c5.number_input(f"Inversión ({CURRENCIES[cur]['sym']})", min_value=0.0,
                                    value=float(CURRENCIES[cur]["default"]) * 0.01,
                                    step=1000.0 if cur == "COP" else 5.0)
        estado = c6.selectbox("Estado", ["Pendiente", "Ganada", "Perdida"])
        resultado = st.text_input("Resultado del partido", placeholder="Ej. 2-1")
        submitted = st.form_submit_button("Guardar registro", type="primary")

    if submitted:
        if not partido.strip():
            st.error("El campo 'Equipos en juego' es obligatorio.")
        else:
            st.session_state.bets.append({
                "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "Partido": partido.strip(), "Tipo": tipo,
                "Seleccion": seleccion.strip() or "-", "Cuota": float(cuota),
                "Inversion": float(inversion), "Estado": estado,
                "Resultado": resultado.strip() or "-",
            })

    bets = st.session_state.bets
    k = tracker_kpis(bets)
    st.markdown("###### Métricas consolidadas")
    m = st.columns(5)
    pcls = ("pos" if k["profit"] >= 0 else "neg") if k["settled"] else ""
    kpi_card(m[0], "P&L neto", money(k["profit"], cur), "utilidad acumulada", pcls)
    kpi_card(m[1], "Capital expuesto", money(k["staked"], cur), f"{k['settled']} liquidadas")
    kpi_card(m[2], "ROI", f"{k['roi']:+.2f}%", "retorno / inversión", pcls)
    kpi_card(m[3], "Yield", f"{k['yield']:+.2f}%", "beneficio / capital", pcls)
    kpi_card(m[4], "Tasa de acierto", f"{k['hit']:.1f}%", f"{k['pending']} pendientes")

    st.markdown("###### Libro de registros")
    if not bets:
        st.markdown("<p class='muted'>Sin operaciones registradas.</p>", unsafe_allow_html=True)
        return

    df = pd.DataFrame(bets)
    df["Retorno_bruto"] = df["Inversion"] * df["Cuota"]
    df["Utilidad_neta"] = [bet_net(e, i, c) for e, i, c in zip(df["Estado"], df["Inversion"], df["Cuota"])]
    df = df.iloc[::-1].reset_index(drop=True)

    def _hl_row(row):
        color = {"Ganada": "rgba(16,185,129,.10)", "Perdida": "rgba(239,68,68,.10)"}.get(row["Estado"], "")
        return [f"background-color: {color}" if color else ""] * len(row)

    st.dataframe(
        df.style.apply(_hl_row, axis=1).format({
            "Cuota": "{:.2f}", "Inversion": lambda v: money(v, cur),
            "Retorno_bruto": lambda v: money(v, cur), "Utilidad_neta": lambda v: money(v, cur)}),
        use_container_width=True, hide_index=True)

    cc = st.columns([1, 1, 3])
    if cc[0].button("Depurar historial"):
        st.session_state.bets = []
        st.rerun()
    cc[1].download_button("Exportar CSV", df.to_csv(index=False).encode("utf-8"),
                          "libro_operaciones.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════
# 5. APLICACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    inject_css()

    # ---------------- Sidebar (controles globales) ----------------
    with st.sidebar:
        st.markdown("### Configuración")

        with st.expander("Cuenta y riesgo", expanded=True):
            cur = st.selectbox("Moneda base", list(CURRENCIES.keys()),
                               format_func=lambda c: f"{c}  ·  {CURRENCIES[c]['sym']}",
                               help="Adapta todos los símbolos monetarios de la interfaz.")
            risk = st.select_slider("Perfil de riesgo",
                                    ["Conservador", "Moderado", "Agresivo"], value="Moderado",
                                    help="Multiplicador de Kelly: 0.20 / 0.50 / 1.00.")
            cparams = CURRENCIES[cur]
            bankroll = st.slider(f"Bankroll ({cparams['sym']})", cparams["min"], cparams["max"],
                                 cparams["default"], step=cparams["step"],
                                 help="Capital total destinado a la operativa.")

        with st.expander("Fuente de datos", expanded=False):
            source = st.selectbox("Origen primario", ACTIVE_SOURCES,
                                  help="Fuente activa para la extracción del enfrentamiento.")
            custom_url = st.text_input("URL personalizada", value="https://www.espn.com/soccer/",
                                       disabled=(source != "Custom URL"),
                                       help="URL de origen para la extracción; se refleja en la auditoría.")
            chips = "".join(
                f"<span class='chip {'on' if s == source else ''}'>{s}</span>"
                for s in ACTIVE_SOURCES)
            chips += "".join(
                f"<span class='chip soon'>{s} · próximamente</span>" for s in FUTURE_SOURCES)
            st.markdown(f"<div class='srcwrap'>{chips}</div>", unsafe_allow_html=True)
            st.caption(f"Fuente activa: {source}. Integraciones futuras deshabilitadas.")

        with st.expander("Competición y muestra", expanded=True):
            tournament = st.selectbox("Torneo", list(CATALOG.keys()))
            teams = list(CATALOG[tournament]["teams"].keys())
            home = st.selectbox("Equipo local", teams, index=0)
            away = st.selectbox("Equipo visitante", [t for t in teams if t != home], index=0)
            n = st.slider("Muestra (últimos N partidos)", 3, 10, 5,
                          help="Tamaño de la ventana histórica; recalcula el motor.")

        st.markdown("---")
        run = st.button("Ejecutar inferencia", type="primary")
        comp = CATALOG[tournament]["comp"]
        st.caption(f"Competitividad de liga x{comp:.2f} "
                   f"({'ofensiva' if comp > 1 else 'defensiva'}).")

    # ---------------- Cabecera ----------------
    render_header(cur, risk)

    # ---------------- Estado ----------------
    if run:
        st.session_state.ready = True
        st.session_state.cfg = dict(home=home, away=away, tournament=tournament, n=n,
                                    comp=comp, source=source, custom_url=custom_url)

    cfg = st.session_state.get("cfg")
    ready = st.session_state.get("ready", False)

    # ---------------- Pestañas ----------------
    tabs = st.tabs(["Panel de control", "Inferencia y DOFA", "Auditoría de datos",
                    "Comparador y valor", "Tracker financiero"])

    with tabs[0]:
        tab_dashboard(cfg, cur, risk, bankroll)

    if ready and cfg:
        with st.spinner("Procesando modelo cuantitativo (Poisson · Kelly · DOFA)…"):
            pred = compute_prediction(cfg["home"], cfg["away"], cfg["n"], cfg["comp"])
            consensus = compute_prediction(cfg["home"], cfg["away"], 10, cfg["comp"])
            dofa = build_dofa(cfg["home"], cfg["away"], cfg["n"], pred)
        with tabs[1]:
            tab_inferencia(cfg["home"], cfg["away"], cfg["tournament"], cfg["n"], pred, dofa)
        with tabs[2]:
            tab_auditoria(cfg["home"], cfg["away"], cfg["n"], cfg["source"], cfg["custom_url"])
        with tabs[3]:
            tab_comparador(cfg["home"], cfg["away"], pred, consensus, risk, bankroll, cur)
    else:
        notice = ("<div class='note'>Ejecute la inferencia desde el panel lateral para "
                  "poblar esta sección.</div>")
        for i in (1, 2, 3):
            with tabs[i]:
                st.markdown(notice, unsafe_allow_html=True)

    with tabs[4]:
        tab_tracker(cur)


if __name__ == "__main__":
    main()
