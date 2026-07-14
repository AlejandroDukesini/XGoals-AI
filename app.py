"""
app.py  —  APEXPREDICT MULTI-ENGINE & TRACKER IA
================================================
Aplicación web de nivel producción para análisis deportivo predictivo, comparación
de cuotas multi-casa, optimización de stake (Kelly) y seguimiento financiero
personal. Streamlit + Pandas + NumPy + SciPy (scipy.stats.poisson).

Ejecutar:
    pip install streamlit==1.40.0 pandas numpy scipy
    streamlit run app.py
    # (si 'streamlit' no está en el PATH:  python -m streamlit run app.py)

Dos pantallas (navegación por st.session_state):
    A) Panel de Análisis Predictivo  — inferencia, datos, auditoría, apuestas.
    B) Historial & Tracker de Apuestas — bet tracker manual con ROI/Yield.

Diseño (cero aleatoriedad, todo determinista y reproducible):
    · Catálogo multi-torneo; histórico de 10 pp generado por oscilaciones
      trigonométricas ancladas a un seed del nombre del equipo.
    · Poisson bivariado (scipy.stats.poisson.pmf) ajustado por competitividad.
    · 4 casas (BetPlay, Wplay, Rushbet, Codere) con dispersión determinista.
    · Stake óptimo por Criterio de Kelly simplificado.

Estructura:
    1. CONFIG & TEMA (CSS azul profundo)
    2. CATÁLOGO + generador determinista
    3. BACKEND (Poisson, DOFA, xG, log, casas, Kelly, tracker)
    4. UI (sidebar, header, pantalla A con pestañas, pantalla B tracker)
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
    page_title="ApexPredict Multi-Engine & Tracker IA",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ANALYST_NAME = "Santiago"

C = {
    "bg": "#050c1a", "bg2": "#0d1b2a", "card": "#1b263b", "card2": "#22304a",
    "line": "rgba(0,180,216,0.16)", "primary": "#0077b6", "accent": "#00b4d8",
    "neon": "#39ff14", "coral": "#ff4d4d", "text": "#e8eef7", "muted": "#8ea3bf",
    "amber": "#ffd166",
}


def inject_css() -> None:
    """CSS del entorno premium: azul profundo, cian, verde neón / rojo coral."""
    st.markdown(f"""
    <style>
      .stApp {{
        background: radial-gradient(1200px 560px at 12% -10%, #0f2544 0%, {C['bg']} 58%), {C['bg']};
        color: {C['text']};
      }}
      #MainMenu, footer, header[data-testid="stHeader"] {{ visibility:hidden; }}
      .block-container {{ padding-top:1.2rem; }}

      section[data-testid="stSidebar"] > div {{
        background: linear-gradient(180deg, {C['card']} 0%, #08111f 100%);
        border-right: 1px solid {C['line']};
      }}
      section[data-testid="stSidebar"] * {{ color:{C['text']}; }}
      .stExpander {{ border:1px solid {C['line']} !important; border-radius:12px !important;
        background:{C['bg2']} !important; }}

      h1,h2,h3,h4 {{ color:{C['text']}; font-weight:800; letter-spacing:-.3px; }}

      .brand {{ display:flex; align-items:center; gap:.7rem; margin-bottom:.1rem; }}
      .brand .mark {{ font-size:2rem; filter:drop-shadow(0 0 12px rgba(0,180,216,.7)); }}
      .brand .name {{ font-size:1.7rem; font-weight:900; letter-spacing:.5px;
        background:linear-gradient(90deg,{C['accent']},{C['primary']});
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
      .brand .tag {{ color:{C['muted']}; font-weight:700; font-size:.9rem; }}
      .welcome {{ color:{C['muted']}; font-size:.9rem; margin:.15rem 0 1rem; }}
      .welcome b {{ color:{C['accent']}; }}

      .kpi {{ background:linear-gradient(160deg,{C['card2']},{C['card']});
        border:1px solid {C['line']}; border-radius:16px; padding:.85rem 1rem;
        box-shadow:0 10px 26px rgba(0,0,0,.4); position:relative; overflow:hidden; }}
      .kpi::before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
        background:linear-gradient(180deg,{C['accent']},{C['primary']}); }}
      .kpi .lbl {{ font-size:.66rem; letter-spacing:1.2px; color:{C['muted']}; text-transform:uppercase; }}
      .kpi .val {{ font-size:1.45rem; font-weight:900; margin-top:.12rem; }}
      .kpi .sub {{ font-size:.72rem; color:{C['muted']}; }}
      .neon {{ color:{C['neon']}; text-shadow:0 0 10px rgba(57,255,20,.55); }}
      .coral {{ color:{C['coral']}; text-shadow:0 0 10px rgba(255,77,77,.45); }}

      .card {{ background:{C['card']}; border:1px solid {C['line']}; border-radius:16px;
        padding:1.05rem 1.25rem; margin-bottom:1rem; box-shadow:0 8px 22px rgba(0,0,0,.35); }}
      .card h4 {{ margin-top:0; color:{C['accent']}; }}

      .chip {{ display:inline-block; padding:.18rem .7rem; border-radius:999px;
        font-size:.72rem; font-weight:800; letter-spacing:.5px; margin-bottom:.5rem; }}
      .cf {{ background:rgba(57,255,20,.15); color:{C['neon']}; }}
      .cd {{ background:rgba(255,77,77,.15); color:{C['coral']}; }}
      .co {{ background:rgba(0,180,216,.16); color:{C['accent']}; }}
      .ca {{ background:rgba(255,209,102,.15); color:{C['amber']}; }}

      .bar {{ background:rgba(255,255,255,.05); border-radius:9px; height:30px;
        display:flex; overflow:hidden; border:1px solid {C['line']}; }}
      .seg {{ display:flex; align-items:center; justify-content:center;
        font-size:.8rem; font-weight:800; color:#06101f; }}

      .stButton > button {{ background:linear-gradient(90deg,{C['primary']},{C['accent']});
        color:#fff; border:none; border-radius:12px; padding:.55rem 1rem; font-weight:800;
        width:100%; box-shadow:0 6px 18px rgba(0,119,182,.45); transition:all .15s ease; }}
      .stButton > button:hover {{ transform:translateY(-2px); box-shadow:0 8px 24px rgba(0,180,216,.6); }}
      .stForm .stButton > button {{ background:linear-gradient(90deg,#0a9,{C['neon']}); color:#06101f; }}

      .stTabs [data-baseweb="tab"] {{ background:{C['card']}; border:1px solid {C['line']};
        border-radius:10px; padding:.5rem 1rem; color:{C['muted']}; font-weight:700; }}
      .stTabs [aria-selected="true"] {{
        background:linear-gradient(90deg,{C['primary']},{C['accent']}) !important;
        color:#fff !important; border-color:transparent !important; }}

      .best {{ background:linear-gradient(90deg, rgba(57,255,20,.18), rgba(0,180,216,.12));
        border:1px solid rgba(57,255,20,.4); border-radius:12px; padding:.9rem 1.1rem; }}
      .muted {{ color:{C['muted']}; font-size:.82rem; }}
    </style>
    """, unsafe_allow_html=True)


# Monedas soportadas (adapta símbolos y rangos de banca).
CURRENCIES = {
    "COP": dict(sym="COP$", default=1_000_000, min=100_000, max=20_000_000, step=50_000),
    "USD": dict(sym="$",    default=10_000,    min=1_000,   max=200_000,    step=500),
}


def money(v: float, cur: str) -> str:
    """Formatea un monto con el símbolo de la moneda seleccionada."""
    return f"{CURRENCIES[cur]['sym']}{v:,.0f}"


# ══════════════════════════════════════════════════════════════════════════
# 2. CATÁLOGO MULTI-TORNEO + GENERADOR DETERMINISTA
# ══════════════════════════════════════════════════════════════════════════
CATALOG: Dict[str, Dict] = {
    "Mundial 2026": {"comp": 0.92, "teams": {
        "Colombia":  dict(badge="🇨🇴", conf="CONMEBOL", atk=1.7, dfn=0.9, pos=55, sot=5.2, pas=.85, style="4-2-3-1", star="J. Rodríguez"),
        "Argentina": dict(badge="🇦🇷", conf="CONMEBOL", atk=2.3, dfn=0.7, pos=58, sot=6.5, pas=.88, style="4-3-3",   star="L. Messi"),
        "Francia":   dict(badge="🇫🇷", conf="UEFA",     atk=2.2, dfn=0.8, pos=56, sot=6.2, pas=.87, style="4-2-3-1", star="K. Mbappé"),
        "España":    dict(badge="🇪🇸", conf="UEFA",     atk=2.1, dfn=0.9, pos=64, sot=6.0, pas=.90, style="4-3-3",   star="Pedri"),
        "Marruecos": dict(badge="🇲🇦", conf="CAF",      atk=1.4, dfn=0.9, pos=50, sot=4.5, pas=.83, style="4-3-3",   star="A. Hakimi"),
        "Japón":     dict(badge="🇯🇵", conf="AFC",      atk=1.6, dfn=1.0, pos=54, sot=4.8, pas=.86, style="4-2-3-1", star="T. Kubo"),
        "USA":       dict(badge="🇺🇸", conf="CONCACAF", atk=1.5, dfn=1.1, pos=52, sot=4.6, pas=.84, style="4-3-3",   star="C. Pulisic"),
    }},
    "Liga BetPlay": {"comp": 1.05, "teams": {
        "Millonarios":       dict(badge="🔵", atk=1.6, dfn=1.0, pos=55, sot=5.0, pas=.83, style="4-2-3-1", star="R. Vargas"),
        "Atlético Nacional": dict(badge="🟢", atk=1.8, dfn=0.9, pos=57, sot=5.4, pas=.84, style="4-3-3",   star="E. Palacios"),
        "Junior":            dict(badge="🔴", atk=1.5, dfn=1.1, pos=53, sot=4.8, pas=.82, style="4-4-2",   star="C. Bacca"),
        "Santa Fe":          dict(badge="⚪", atk=1.4, dfn=1.1, pos=52, sot=4.6, pas=.81, style="4-4-2",   star="H. Rodallega"),
    }},
    "Champions League": {"comp": 0.98, "teams": {
        "Real Madrid":     dict(badge="⚪", atk=2.4, dfn=0.9, pos=58, sot=6.6, pas=.88, style="4-3-3",   star="J. Bellingham"),
        "Manchester City": dict(badge="🔷", atk=2.6, dfn=0.8, pos=66, sot=7.2, pas=.91, style="4-3-3",   star="E. Haaland"),
        "PSG":             dict(badge="🔵", atk=2.3, dfn=1.0, pos=60, sot=6.4, pas=.88, style="4-3-3",   star="Dembélé"),
        "Bayern Múnich":   dict(badge="🔴", atk=2.5, dfn=1.0, pos=62, sot=6.8, pas=.89, style="4-2-3-1", star="H. Kane"),
    }},
    "MLS / Otros": {"comp": 1.12, "teams": {
        "Inter de Miami": dict(badge="🩷", atk=2.0, dfn=1.2, pos=57, sot=5.6, pas=.85, style="4-4-2", star="L. Messi"),
        "Al-Nassr":       dict(badge="🟡", atk=2.1, dfn=1.1, pos=58, sot=5.8, pas=.85, style="4-3-3", star="C. Ronaldo"),
        "LAFC":           dict(badge="⚫", atk=1.8, dfn=1.1, pos=54, sot=5.2, pas=.83, style="4-3-3", star="O. Bouanga"),
        "LA Galaxy":      dict(badge="⭐", atk=1.7, dfn=1.2, pos=53, sot=5.0, pas=.82, style="4-4-2", star="R. Puig"),
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
BOOKIES = ["BetPlay", "Wplay", "Rushbet", "Codere"]
HOUSE_MARGIN = {"BetPlay": 0.040, "Wplay": 0.045, "Rushbet": 0.050, "Codere": 0.038}
RISK_KELLY = {"Conservador": 0.25, "Moderado": 0.50, "Agresivo": 1.00}
_ALT_FORM = {"4-3-3": "4-2-3-1", "4-2-3-1": "4-3-3", "4-4-2": "4-3-3", "3-5-2": "4-4-2"}


def _seed(name: str) -> int:
    return sum(ord(c) for c in name)


_HIST_CACHE: Dict[str, pd.DataFrame] = {}


def team_history(team: str) -> pd.DataFrame:
    """Histórico de 10 partidos GENERADO de forma determinista desde el perfil."""
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
            "GF": gf, "GC": ga, "Posesión %": pos, "Tiros arco": sot,
            "Tiros concedidos": sotc, "Faltas": fouls,
            "Pases %": round(pas * 100, 1), "xG": round(sot * XG_PER_SOT, 2),
            "Formación DT": form,
        })
    df = pd.DataFrame(rows)
    _HIST_CACHE[team] = df
    return df


def last_n(team: str, n: int) -> pd.DataFrame:
    return team_history(team).tail(n).reset_index(drop=True)


def team_averages(team: str, n: int) -> Dict[str, float]:
    df = last_n(team, n)
    return {
        "gf": df["GF"].mean(), "ga": df["GC"].mean(), "pos": df["Posesión %"].mean(),
        "sot": df["Tiros arco"].mean(), "sotc": df["Tiros concedidos"].mean(),
        "fouls": df["Faltas"].mean(), "pas": df["Pases %"].mean(),
        "form": df["Formación DT"].mode().iloc[0],
    }


def _league_avg() -> float:
    vals = []
    for t in ALL_TEAMS:
        h = team_history(t)
        vals += h["GF"].tolist() + h["GC"].tolist()
    return float(np.mean(vals))


LEAGUE_AVG = _league_avg()


# ══════════════════════════════════════════════════════════════════════════
# 3. BACKEND
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Prediction:
    lam_home: float
    lam_away: float
    matrix: np.ndarray
    markets: Dict[str, float]
    top_scores: List[Tuple[str, float]]


def compute_prediction(home: str, away: str, n: int, comp: float) -> Prediction:
    """Poisson bivariado ajustado por competitividad (`comp`) de la liga."""
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
        "BTTS Sí": float(M[1:, 1:].sum()),
    }
    mk["BTTS No"] = 1 - mk["BTTS Sí"]
    mk["1X"] = mk["1"] + mk["X"]; mk["X2"] = mk["X"] + mk["2"]; mk["12"] = mk["1"] + mk["2"]
    flat = sorted(((f"{i}-{j}", float(M[i, j])) for i in k for j in k), key=lambda t: -t[1])[:6]
    return Prediction(round(lam_h, 3), round(lam_a, 3), M, mk, flat)


def build_dofa(home: str, away: str, n: int, pred: Prediction) -> Dict[str, List[str]]:
    h, a = team_averages(home, n), team_averages(away, n)
    hs, as_ = ALL_TEAMS[home], ALL_TEAMS[away]
    F, D, O, A = [], [], [], []
    if h["gf"] > a["ga"] + 0.3:
        F.append(f"Ataque de {home} ({h['gf']:.1f} goles/pp) supera la defensa de {away} ({a['ga']:.1f} recibidos/pp).")
    if h["pos"] > a["pos"] + 3:
        F.append(f"{home} domina la posesión ({h['pos']:.0f}% vs {a['pos']:.0f}%): impone ritmo.")
    if a["fouls"] > 13 and hs.get("star"):
        F.append(f"{away} comete muchas faltas ({a['fouls']:.1f}/pp): balón parado ideal para {hs['star']} y {home}.")
    if a["gf"] > h["ga"] + 0.3:
        D.append(f"Ataque de {away} ({a['gf']:.1f} goles/pp) penetra la defensa de {home} ({h['ga']:.1f} recibidos/pp).")
    if h["pas"] < a["pas"]:
        D.append(f"Menor precisión de pase ({h['pas']:.0f}% vs {a['pas']:.0f}%): riesgo de pérdidas bajo presión.")
    if h["fouls"] > a["fouls"] + 1.5:
        D.append(f"{home} comete más faltas ({h['fouls']:.1f}/pp): expone su área a balón parado de {as_.get('star', away)}.")
    if pred.markets["Over 2.5"] > 0.55:
        O.append(f"Partido abierto: {pred.markets['Over 2.5']:.0%} de probabilidad de Over 2.5.")
    if pred.markets["1"] > pred.markets["2"] + 0.12:
        O.append(f"Localía y nivel inclinan el favoritismo a {home} ({pred.markets['1']:.0%}).")
    A.append(f"Sistema del DT rival: «{a['form']}» — figura a vigilar: {as_.get('star', 'su referente')}.")
    if a["gf"] >= 2.2:
        A.append(f"{away} es prolífico ({a['gf']:.1f} goles/pp): atención a las transiciones.")
    if pred.markets["2"] > 0.30:
        A.append(f"{away} conserva {pred.markets['2']:.0%} de opciones: margen estrecho.")
    return {"Fortalezas": F, "Debilidades": D, "Oportunidades": O, "Amenazas": A}


def extraction_log(team: str, n: int, source: str, custom_url: str) -> pd.DataFrame:
    base = {"ESPN Scraper": "https://www.espn.com/soccer/match/_/gameId/{gid}",
            "Win Sports Analytica": "https://www.winsports.co/partido/{gid}"}
    rows = []
    for _, row in last_n(team, n).iterrows():
        j = 10 - int(row["Jornada"].split("-")[1])
        gid = 6_400_000 + (_seed(team) * 31 + j * 7) % 90_000
        if source == "Custom URL" and custom_url.strip():
            url = f"{custom_url.rstrip('/')}?match={team.lower().replace(' ', '-')}-{gid}"
            src_name = "Custom URL"
        else:
            url = base.get(source, base["ESPN Scraper"]).format(gid=gid)
            src_name = source
        match_date = SCRAPE_DATE.normalize() - pd.Timedelta(days=(10 - j) * 7)
        stamp = SCRAPE_DATE + pd.Timedelta(seconds=j * 7)
        rows.append({
            "Partido": f"{team} vs {row['Rival']}", "Fecha partido": match_date.strftime("%Y-%m-%d"),
            "Fuente": src_name, "URL de origen": url,
            "Extraído": stamp.strftime("%Y-%m-%d %H:%M:%S"), "Estado": "✅ 200 OK",
        })
    return pd.DataFrame(rows)


# ---- Comparador multi-casa + Kelly --------------------------------------- #
# Mercados ofrecidos en el comparador: etiqueta -> (clave interna, referencia).
def market_options(home: str, away: str) -> Dict[str, Tuple[str, str]]:
    return {
        f"Victoria {home}": ("1", home),
        "Empate": ("X", "el empate"),
        f"Victoria {away}": ("2", away),
        "Más de 2.5 goles": ("Over 2.5", "Over 2.5"),
        "Menos de 2.5 goles": ("Under 2.5", "Under 2.5"),
        "Ambos Anotan · Sí": ("BTTS Sí", "Ambos Anotan"),
        "Ambos Anotan · No": ("BTTS No", "No Ambos Anotan"),
    }


def bookmaker_table(prob_ia: float, cons_p: float, home: str, away: str,
                    market_key: str) -> pd.DataFrame:
    """
    Cuotas de 4 casas para un mercado. Cada casa cotiza sobre el consenso con su
    margen de casa y una dispersión determinista (sin aleatoriedad). El EV se
    calcula contra la probabilidad del MODELO (prob_ia).
    """
    fair_cons = 1.0 / min(max(cons_p, 0.02), 0.98)      # base de la casa (consenso)
    rows = []
    for casa in BOOKIES:
        wob = 0.06 * math.sin(_seed(casa + market_key + home + away) * 0.1)
        odds = round(max(1.01, fair_cons * (1 - HOUSE_MARGIN[casa] + wob)), 2)
        ev = prob_ia * odds - 1
        rows.append({"Casa": casa, "Cuota": odds, "Prob. implícita": round(1 / odds, 3),
                     "EV": round(ev, 3)})
    return pd.DataFrame(rows)


def kelly_amount(ev: float, odds: float, bankroll: float, risk: str) -> float:
    """
    Criterio de Kelly simplificado:
        Monto = Bankroll · (EV / (Cuota − 1)) · Factor de Perfil de Riesgo
    Solo con ventaja (EV>0); si no, no se apuesta.
    """
    if ev <= 0 or odds <= 1:
        return 0.0
    fraction = (ev / (odds - 1)) * RISK_KELLY[risk]
    return max(0.0, bankroll * fraction)


# ---- Tracker ------------------------------------------------------------- #
def bet_net(estado: str, inversion: float, cuota: float) -> float:
    """Utilidad neta de una apuesta según su estado."""
    if estado == "Ganada":
        return inversion * (cuota - 1)
    if estado == "Perdida":
        return -inversion
    return 0.0                                          # Pendiente / Anulada


def tracker_kpis(bets: List[Dict]) -> Dict[str, float]:
    """Balance neto, ROI y Yield del historial de apuestas del usuario."""
    settled = [b for b in bets if b["Estado"] in ("Ganada", "Perdida")]
    staked = sum(b["Inversión"] for b in settled)
    profit = sum(bet_net(b["Estado"], b["Inversión"], b["Cuota"]) for b in settled)
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


def render_header(bankroll: float, cur: str) -> None:
    """Logo + bienvenida + KPIs (derivados del tracker del usuario)."""
    st.markdown(
        "<div class='brand'><span class='mark'>🛰️</span>"
        "<span class='name'>APEXPREDICT MULTI-ENGINE & TRACKER</span>"
        "<span class='tag'>| Quant Sports IA</span></div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='welcome'>Bienvenido de nuevo, <b>{ANALYST_NAME}</b>. "
        f"Motor multi-torneo · comparador de 4 casas · optimización de stake "
        f"(Kelly) · moneda base <b>{cur}</b>.</div>", unsafe_allow_html=True)

    k = tracker_kpis(st.session_state.get("bets", []))
    balance = bankroll + k["profit"]
    pos = k["profit"] >= 0
    cls = "neon" if pos else "coral"
    arrow = "▲" if pos else "▼"
    cols = st.columns(4)
    kpi_card(cols[0], "Balance (Banca + P&L)", money(balance, cur),
             f"{arrow} {money(k['profit'], cur)} P&L", cls if k["settled"] else "")
    kpi_card(cols[1], "ROI acumulado", f"{k['roi']:+.1f}%", f"{k['settled']} liquidadas",
             cls if k["settled"] else "")
    kpi_card(cols[2], "Yield actual", f"{k['yield']:+.1f}%", "beneficio / stake",
             cls if k["settled"] else "")
    kpi_card(cols[3], "Apuestas en tracker", f"{k['n']}", f"{k['pending']} pendientes", "")


def render_prob_bar(mk: Dict[str, float], home: str, away: str) -> None:
    h, d, a = mk["1"], mk["X"], mk["2"]
    hb, ab = ALL_TEAMS[home]["badge"], ALL_TEAMS[away]["badge"]
    st.markdown(f"""
        <div class='bar'>
          <div class='seg' style='width:{h*100:.1f}%;background:{C['accent']}'>{h:.0%}</div>
          <div class='seg' style='width:{d*100:.1f}%;background:{C['muted']}'>{d:.0%}</div>
          <div class='seg' style='width:{a*100:.1f}%;background:{C['primary']}'>{a:.0%}</div>
        </div>
        <div style='display:flex;justify-content:space-between;margin-top:.35rem;
                    font-size:.82rem;color:{C['muted']}'>
          <span>{hb} {home}</span><span>Empate</span><span>{away} {ab}</span>
        </div>""", unsafe_allow_html=True)


def render_dofa(dofa: Dict[str, List[str]]) -> None:
    cls = {"Fortalezas": "cf", "Debilidades": "cd", "Oportunidades": "co", "Amenazas": "ca"}
    cols = st.columns(4)
    for col, (dim, items) in zip(cols, dofa.items()):
        html = f"<div class='card'><span class='chip {cls[dim]}'>{dim.upper()}</span>"
        if items:
            html += "<ul style='padding-left:1.05rem;margin:.2rem 0 0;font-size:.85rem'>"
            html += "".join(f"<li style='margin-bottom:.4rem'>{it}</li>" for it in items)
            html += "</ul>"
        else:
            html += "<p class='muted'>Sin señales relevantes en esta muestra.</p>"
        col.markdown(html + "</div>", unsafe_allow_html=True)


# ---------------- Pestaña: Inferencia & DOFA ----------------
def tab_inferencia(home, away, tournament, n, pred, dofa) -> None:
    hb, ab = ALL_TEAMS[home]["badge"], ALL_TEAMS[away]["badge"]
    score, sp = pred.top_scores[0]
    st.markdown(f"#### {hb} {home}  vs  {away} {ab}  ·  _{tournament}_  ·  muestra: últimos {n} pp")
    c = st.columns(5)
    kpi_card(c[0], "Marcador proyectado", score, f"prob. {sp:.1%}", "neon")
    kpi_card(c[1], "Victoria local", f"{pred.markets['1']:.0%}", home)
    kpi_card(c[2], "Empate", f"{pred.markets['X']:.0%}", "resultado X")
    kpi_card(c[3], "Victoria visitante", f"{pred.markets['2']:.0%}", away)
    kpi_card(c[4], "Goles esperados (λ)", f"{pred.lam_home:.2f} – {pred.lam_away:.2f}", "local – visita")

    st.markdown("<div class='card'><h4>Distribución 1X2</h4>", unsafe_allow_html=True)
    render_prob_bar(pred.markets, home, away)
    st.markdown("</div>", unsafe_allow_html=True)

    cA, cB = st.columns([1, 1])
    with cA:
        st.markdown("<div class='card'><h4>🎯 Marcadores más probables</h4>", unsafe_allow_html=True)
        sdf = pd.DataFrame(pred.top_scores, columns=["Marcador", "Probabilidad"])
        st.dataframe(sdf.style.format({"Probabilidad": "{:.1%}"}), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with cB:
        st.markdown("<div class='card'><h4>📈 Mercados de goles</h4>", unsafe_allow_html=True)
        g = st.columns(2)
        kpi_card(g[0], "Más de 2.5", f"{pred.markets['Over 2.5']:.0%}")
        kpi_card(g[1], "Menos de 2.5", f"{pred.markets['Under 2.5']:.0%}")
        g2 = st.columns(2)
        kpi_card(g2[0], "Ambos anotan Sí", f"{pred.markets['BTTS Sí']:.0%}")
        kpi_card(g2[1], "Ambos anotan No", f"{pred.markets['BTTS No']:.0%}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🧭 Matriz DOFA Cruzada (dinámica)")
    render_dofa(dofa)


# ---------------- Pestaña: Datos & Auditoría ----------------
def tab_datos(home, away, n, source, custom_url) -> None:
    st.markdown("#### 🔍 Datos crudos & Auditoría de origen")
    active = custom_url.strip() if source == "Custom URL" else "plantilla oficial"
    st.markdown(f"<span class='muted'>Origen: <b>{source}</b> · {active} · "
                f"corrida {SCRAPE_DATE.strftime('%Y-%m-%d %H:%M')}.</span>", unsafe_allow_html=True)
    for team in (home, away):
        st.markdown(f"##### {ALL_TEAMS[team]['badge']} {team} — log de extracción")
        st.dataframe(extraction_log(team, n, source, custom_url), use_container_width=True,
                     hide_index=True,
                     column_config={"URL de origen": st.column_config.LinkColumn("URL de origen")})
        df = last_n(team, n)
        st.dataframe(df, use_container_width=True, hide_index=True)
        cc = st.columns(2)
        with cc[0]:
            st.markdown("<span class='muted'>Goles: a favor vs en contra</span>", unsafe_allow_html=True)
            st.line_chart(df.set_index("Jornada")[["GF", "GC"]], height=190)
        with cc[1]:
            st.markdown("<span class='muted'>Tiros al arco vs concedidos</span>", unsafe_allow_html=True)
            st.bar_chart(df.set_index("Jornada")[["Tiros arco", "Tiros concedidos"]], height=190)
        st.markdown("---")


# ---------------- Pestaña: Apuestas inteligentes (multi-casa + Kelly) --------
def tab_apuestas(home, away, pred, consensus, risk, bankroll, cur, stake_pct) -> None:
    st.markdown("#### 💸 Apuestas Inteligentes · Comparador Multi-Casa & Valor (EV)")
    opts = market_options(home, away)
    label = st.selectbox("Mercado a analizar", list(opts.keys()))
    key, ref = opts[label]
    prob_ia = pred.markets[key]
    cons_p = consensus.markets[key]
    fair_ia = 1.0 / max(prob_ia, 0.001)

    # Explicación matemática transparente
    st.info(f"**Cálculo transparente** · Probabilidad IA = **{prob_ia:.1%}**  →  "
            f"Cuota Justa IA = 1 / Prob. IA = **{fair_ia:.2f}**.  "
            f"Valor Esperado por casa: EV = (Prob. IA × Cuota) − 1.")

    # Tabla de las 4 casas, mejor cuota resaltada
    bt = bookmaker_table(prob_ia, cons_p, home, away, key)
    best_i = bt["Cuota"].idxmax()
    best = bt.loc[best_i]

    def _hl(row):
        is_best = row.name == best_i
        bg = ("background-color: rgba(57,255,20,.18)" if is_best
              else ("background-color: rgba(255,77,77,.10)" if row["EV"] <= 0 else ""))
        return [bg] * len(row)

    st.dataframe(
        bt.style.apply(_hl, axis=1).format({"Cuota": "{:.2f}", "Prob. implícita": "{:.1%}", "EV": "{:+.2f}"}),
        use_container_width=True, hide_index=True)

    # Recomendación de compra + Kelly
    ev_best = float(best["EV"])
    monto_kelly = kelly_amount(ev_best, float(best["Cuota"]), bankroll, risk)
    monto_cap = bankroll * (stake_pct / 100.0)
    monto_final = min(monto_kelly, monto_cap) if monto_kelly > 0 else 0.0

    if ev_best > 0 and monto_final > 0:
        ganancia = monto_final * (best["Cuota"] - 1)
        st.markdown(
            f"<div class='best'>✅ <b>Recomendación de compra</b><br>"
            f"Apuesta a <b>{ref}</b> en <b>{best['Casa']}</b> "
            f"(mejor cuota: <b>{best['Cuota']:.2f}</b>) · EV <span class='neon'>{ev_best:+.2f}</span><br>"
            f"Monto óptimo (Kelly {risk}): <b class='neon'>{money(monto_final, cur)}</b> "
            f"→ ganancia potencial <b>{money(ganancia, cur)}</b>.<br>"
            f"<span class='muted'>Kelly puro sugiere {money(monto_kelly, cur)}; limitado a tu stake "
            f"máx. {stake_pct:.1f}% = {money(monto_cap, cur)}.</span></div>",
            unsafe_allow_html=True)
        st.caption(f"Kelly: Monto = Bankroll {money(bankroll, cur)} × (EV {ev_best:+.2f} / "
                   f"(Cuota {best['Cuota']:.2f} − 1)) × Factor {RISK_KELLY[risk]:.2f} "
                   f"({risk}) = {money(monto_kelly, cur)}.")
    else:
        st.warning(f"❌ Ninguna casa ofrece valor (EV>0) para «{label}» con la muestra actual. "
                   f"La mejor cuota es {best['Cuota']:.2f} en {best['Casa']}, pero no supera la "
                   f"Cuota Justa IA ({fair_ia:.2f}). No se recomienda apostar.")


# ══════════════════════════════════════════════════════════════════════════
# PANTALLA A — Panel de Análisis Predictivo
# ══════════════════════════════════════════════════════════════════════════
def screen_analisis(cfg) -> None:
    # Botón de retroceso / reset
    cols = st.columns([1, 3])
    if cols[0].button("↩️ Volver / Configurar Nuevo Análisis"):
        st.session_state.ready = False
        st.rerun()

    if not st.session_state.get("ready"):
        st.markdown(
            "<div class='card'><h4>👋 Configura tu análisis</h4><p>Elige en la barra "
            "lateral el motor de datos, torneo y equipos, la muestra y tu perfil de "
            "apuestas. Pulsa <b>Ejecutar Inferencia de IA</b> para desplegar la "
            "predicción, la auditoría de datos y el comparador de casas.</p></div>",
            unsafe_allow_html=True)
        return

    home, away = cfg["home"], cfg["away"]
    n, comp = cfg["n"], cfg["comp"]
    pred = compute_prediction(home, away, n, comp)
    consensus = compute_prediction(home, away, 10, comp)
    dofa = build_dofa(home, away, n, pred)

    t1, t2, t3 = st.tabs(["🧠 Inferencia IA & DOFA", "🔍 Datos & Auditoría", "💸 Apuestas Inteligentes"])
    with t1:
        tab_inferencia(home, away, cfg["tournament"], n, pred, dofa)
    with t2:
        tab_datos(home, away, n, cfg["source"], cfg["custom_url"])
    with t3:
        tab_apuestas(home, away, pred, consensus, cfg["risk"], cfg["bankroll"],
                     cfg["cur"], cfg["stake_pct"])

    st.caption("⚠️ Herramienta educativa/analítica con datos simulados y deterministas. "
               "Las apuestas implican riesgo financiero; ningún modelo garantiza resultados.")


# ══════════════════════════════════════════════════════════════════════════
# PANTALLA B — Historial & Tracker de Apuestas
# ══════════════════════════════════════════════════════════════════════════
def screen_tracker(cur: str) -> None:
    st.markdown("### 📒 Historial & Tracker de Apuestas Personal")
    st.markdown("<span class='muted'>Registra manualmente tus jugadas. El historial se "
                "guarda en la sesión del navegador. Utilidad y ROI se calculan solos.</span>",
                unsafe_allow_html=True)

    if "bets" not in st.session_state:
        st.session_state.bets = []

    # ---- Formulario de alta ----
    with st.form("bet_form", clear_on_submit=True):
        st.markdown("#### ➕ Registrar nueva apuesta")
        c1, c2, c3 = st.columns(3)
        partido = c1.text_input("Equipos / Partido", value="",
                                placeholder="Ej. Colombia vs Argentina")
        tipo = c2.selectbox("Tipo de apuesta", ["Sencilla", "Combinada"])
        seleccion = c3.text_input("Selección / Mercado", value="",
                                  placeholder="Ej. Gana Colombia, Over 2.5")
        c4, c5, c6 = st.columns(3)
        cuota = c4.number_input("Cuota / Tasa", min_value=1.01, max_value=1000.0,
                                value=2.00, step=0.01,
                                help="⚖️ La cuota pactada con la casa. Ganancia = Inversión × Cuota.")
        inversion = c5.number_input(f"Inversión ({CURRENCIES[cur]['sym']})", min_value=0.0,
                                    value=float(CURRENCIES[cur]["default"]) * 0.01, step=1000.0 if cur == "COP" else 5.0,
                                    help="🎯 Monto arriesgado en esta apuesta (tu stake).")
        c7, c8 = st.columns(2)
        estado = c7.selectbox("Estado", ["Pendiente", "Ganada", "Perdida"])
        resultado = c8.text_input("Resultado real", value="", placeholder="Ej. 2-1")
        submitted = st.form_submit_button("💾 Guardar apuesta")

    if submitted:
        if not partido.strip():
            st.error("Indica al menos el partido/equipos.")
        else:
            st.session_state.bets.append({
                "Fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "Partido": partido.strip(), "Tipo": tipo,
                "Selección": seleccion.strip() or "-", "Cuota": float(cuota),
                "Inversión": float(inversion), "Estado": estado,
                "Resultado": resultado.strip() or "-",
            })
            st.success(f"Apuesta registrada: {partido.strip()} @ {cuota:.2f}")

    bets = st.session_state.bets

    # ---- Cuadro de mando ----
    k = tracker_kpis(bets)
    st.markdown("#### 📊 Cuadro de mando financiero")
    m = st.columns(5)
    pcls = "neon" if k["profit"] >= 0 else "coral"
    kpi_card(m[0], "Balance neto", money(k["profit"], cur),
             "utilidad acumulada", pcls if k["settled"] else "")
    kpi_card(m[1], "Importe apostado", money(k["staked"], cur), f"{k['settled']} liquidadas")
    kpi_card(m[2], "ROI", f"{k['roi']:+.1f}%", "retorno / inversión", pcls if k["settled"] else "")
    kpi_card(m[3], "Yield", f"{k['yield']:+.1f}%", "beneficio / stake", pcls if k["settled"] else "")
    kpi_card(m[4], "Tasa de acierto", f"{k['hit']:.0f}%", f"{k['pending']} pendientes")

    # ---- Tabla del historial ----
    st.markdown("#### 🗂️ Registro histórico de jugadas")
    if not bets:
        st.info("Aún no hay apuestas. Registra la primera con el formulario de arriba.")
        return

    df = pd.DataFrame(bets)
    df["Ganancia si acierta"] = df["Inversión"] * df["Cuota"]
    df["Utilidad neta"] = [bet_net(e, i, c) for e, i, c in
                           zip(df["Estado"], df["Inversión"], df["Cuota"])]
    df = df.iloc[::-1].reset_index(drop=True)          # más recientes arriba

    def _hl_row(row):
        color = {"Ganada": "rgba(57,255,20,.14)", "Perdida": "rgba(255,77,77,.14)"}.get(row["Estado"], "")
        return [f"background-color: {color}" if color else ""] * len(row)

    st.dataframe(
        df.style.apply(_hl_row, axis=1).format({
            "Cuota": "{:.2f}", "Inversión": lambda v: money(v, cur),
            "Ganancia si acierta": lambda v: money(v, cur),
            "Utilidad neta": lambda v: money(v, cur)}),
        use_container_width=True, hide_index=True)

    cc = st.columns([1, 1, 3])
    if cc[0].button("🗑️ Limpiar historial"):
        st.session_state.bets = []
        st.rerun()
    csv = df.to_csv(index=False).encode("utf-8")
    cc[1].download_button("⬇️ Exportar CSV", csv, "historial_apuestas.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════
# 5. APLICACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    inject_css()

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown("## 🛰️ Panel de Control")

        screen = st.radio("Navegación", ["📊 Panel de Análisis", "📒 Historial & Tracker"],
                          label_visibility="collapsed")

        with st.expander("🌎 Moneda base", expanded=True):
            cur = st.selectbox("Moneda de trabajo", ["COP", "USD"],
                               help="Adapta todos los símbolos ($ o COP$) de la interfaz.")

        with st.expander("🔌 A · Motor de Datos (Sourcing)", expanded=False):
            source = st.selectbox("Origen primario",
                                  ["ESPN Scraper", "Win Sports Analytica", "Custom URL"])
            custom_url = st.text_input("URL personalizada", value="https://www.espn.com/soccer/",
                                       help="Pega la URL que la IA 'leerá'. Se refleja en el log.",
                                       disabled=(source != "Custom URL"))

        with st.expander("🏆 B · Competición y Equipos", expanded=True):
            tournament = st.selectbox("Torneo", list(CATALOG.keys()))
            teams = list(CATALOG[tournament]["teams"].keys())
            home = st.selectbox("🏠 Equipo Local", teams, index=0)
            away = st.selectbox("✈️ Equipo Rival", [t for t in teams if t != home], index=0)
            n = st.slider("Muestra (últimos N partidos)", 3, 10, 5,
                          help="El backend recalcula todo con estos N partidos.")

        cparams = CURRENCIES[cur]
        with st.expander("💰 C · Perfil de Apuestas", expanded=True):
            bankroll = st.slider(f"💰 Bankroll ({cparams['sym']})", cparams["min"], cparams["max"],
                                 cparams["default"], step=cparams["step"],
                                 help="💰 Bankroll: capital total destinado exclusivamente a apostar.")
            stake_pct = st.slider("🎯 Stake máximo (% del Bankroll)", 0.5, 10.0, 2.0, 0.5,
                                  help="🎯 Stake: % del Bankroll que arriesgas por apuesta "
                                       "(Stake 1 = 1% del capital).")
            risk = st.select_slider("⚖️ Aversión al Riesgo",
                                    ["Conservador", "Moderado", "Agresivo"], value="Moderado",
                                    help="⚖️ Ajusta la agresividad de la sugerencia de stake (Kelly).")

        with st.expander("📖 Diccionario financiero", expanded=False):
            st.markdown(
                "- **💰 Bankroll:** todo tu capital destinado a apostar.\n"
                "- **🎯 Stake:** % del Bankroll arriesgado por apuesta (Stake 1 = 1%).\n"
                "- **⚖️ Aversión al Riesgo:** perfil (Conservador/Moderado/Agresivo) que "
                "escala el Criterio de Kelly.\n"
                "- **📈 EV (Valor Esperado):** (Prob. IA × Cuota) − 1. Si es >0, hay valor.\n"
                "- **🧮 Kelly:** fracción óptima del capital a apostar según la ventaja.\n"
                "- **📊 ROI / Yield:** utilidad ÷ importe apostado (rentabilidad).")

        st.markdown("---")
        run = st.button("🚀 Ejecutar Inferencia de IA")
        comp = CATALOG[tournament]["comp"]
        st.caption(f"Competitividad de liga ×{comp:.2f} "
                   f"({'abierta' if comp > 1 else 'cerrada/defensiva'}).")

    # ---------------- Header ----------------
    render_header(bankroll, cur)

    # ---------------- Estado + routing ----------------
    if run:
        st.session_state.ready = True
        st.session_state.cfg = dict(home=home, away=away, tournament=tournament, n=n,
                                    comp=comp, source=source, custom_url=custom_url,
                                    risk=risk, bankroll=bankroll, cur=cur, stake_pct=stake_pct)

    if screen == "📒 Historial & Tracker":
        screen_tracker(cur)
        return

    # Pantalla A: usa la última config ejecutada (o la actual del sidebar como fallback).
    cfg = st.session_state.get("cfg", dict(
        home=home, away=away, tournament=tournament, n=n, comp=comp, source=source,
        custom_url=custom_url, risk=risk, bankroll=bankroll, cur=cur, stake_pct=stake_pct))
    # Mantén moneda/banca/riesgo sincronizados aunque no se re-ejecute la inferencia.
    cfg.update(cur=cur, bankroll=bankroll, risk=risk, stake_pct=stake_pct)
    screen_analisis(cfg)


if __name__ == "__main__":
    main()
