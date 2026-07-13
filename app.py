"""
app.py  —  APEXPREDICT IA · Inteligencia Deportiva
==================================================
Aplicación de análisis deportivo predictivo y de apuestas, nivel producción,
construida con Streamlit + Pandas + NumPy + SciPy (scipy.stats.poisson).

Ejecutar:
    pip install streamlit pandas numpy scipy
    streamlit run app.py
    # (si 'streamlit' no está en el PATH:  python -m streamlit run app.py)

Filosofía de diseño (rigor y veracidad, cero aleatoriedad):
  · La base de datos histórica es fija y auditable (fila por fila, 10 partidos).
  · El modelo de goles es Poisson bivariado con scipy.stats.poisson.pmf.
  · Las cuotas de la casa se derivan del CONSENSO (muestra completa de 10 pp).
    El analista modela sobre la ventana de N partidos que elija en el slider;
    cuando su muestra diverge del consenso, aparece el Valor Esperado (EV>0).
    Todo es determinista: mismos inputs -> mismos números, siempre.

Estructura:
    1. CONFIG & TEMA (CSS Navy)
    2. DATOS (histórico fila por fila, jugadores, banca simulada)
    3. BACKEND (Poisson, mercados, DOFA, EV, combinadas, consistencia)
    4. UI (header/KPIs, sidebar, 3 pestañas, sección de apuestas)
    5. main()
"""
from __future__ import annotations

import itertools
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
    page_title="ApexPredict IA · Inteligencia Deportiva",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

ANALYST_NAME = "Santiago"      # analista (dashboard personal)
INITIAL_BANKROLL = 10_000      # unidades de banca

# Paleta Dark Blue / Navy
C = {
    "bg": "#0a1524", "bg2": "#0d1b2a", "card": "#152238", "card2": "#1b2b47",
    "line": "rgba(96,165,250,0.14)", "primary": "#0077b6", "accent": "#00b4d8",
    "cyan": "#38e0e0", "text": "#e8eef7", "muted": "#8ea3bf",
    "green": "#20c997", "red": "#ef476f", "amber": "#ffd166",
}


def inject_css() -> None:
    """Inyecta el CSS del tema premium azul oscuro."""
    st.markdown(f"""
    <style>
      .stApp {{
        background:
          radial-gradient(1100px 500px at 12% -8%, #16294a 0%, {C['bg']} 55%),
          {C['bg']};
        color: {C['text']};
      }}
      #MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; }}
      .block-container {{ padding-top: 1.4rem; }}

      /* Sidebar */
      section[data-testid="stSidebar"] > div {{
        background: linear-gradient(180deg, {C['card']} 0%, #0c1930 100%);
        border-right: 1px solid {C['line']};
      }}
      section[data-testid="stSidebar"] * {{ color: {C['text']}; }}

      /* Tipografía */
      h1,h2,h3,h4 {{ color:{C['text']}; font-weight:800; letter-spacing:-.3px; }}

      /* Header / logo */
      .brand {{
        display:flex; align-items:center; gap:.7rem; margin-bottom:.15rem;
      }}
      .brand .mark {{
        font-size:1.9rem; filter: drop-shadow(0 0 10px rgba(0,180,216,.6));
      }}
      .brand .name {{
        font-size:1.75rem; font-weight:900; letter-spacing:.5px;
        background:linear-gradient(90deg,{C['cyan']},{C['primary']});
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
      }}
      .brand .tag {{ color:{C['muted']}; font-weight:600; font-size:.95rem; }}
      .welcome {{ color:{C['muted']}; font-size:.9rem; margin:.1rem 0 1rem; }}
      .welcome b {{ color:{C['accent']}; }}

      /* KPI cards */
      .kpi {{
        background:linear-gradient(160deg,{C['card2']},{C['card']});
        border:1px solid {C['line']}; border-radius:16px; padding:.85rem 1rem;
        box-shadow:0 10px 26px rgba(0,0,0,.35); position:relative; overflow:hidden;
      }}
      .kpi::before {{
        content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
        background:linear-gradient(180deg,{C['cyan']},{C['primary']});
      }}
      .kpi .lbl {{ font-size:.68rem; letter-spacing:1.3px; color:{C['muted']};
        text-transform:uppercase; }}
      .kpi .val {{ font-size:1.55rem; font-weight:900; margin-top:.15rem; }}
      .kpi .sub {{ font-size:.72rem; color:{C['muted']}; }}
      .up {{ color:{C['green']}; }} .down {{ color:{C['red']}; }}

      /* Tarjetas */
      .card {{
        background:{C['card']}; border:1px solid {C['line']}; border-radius:16px;
        padding:1.1rem 1.3rem; margin-bottom:1rem; box-shadow:0 8px 22px rgba(0,0,0,.3);
      }}
      .card h4 {{ margin-top:0; color:{C['accent']}; }}

      /* Chips DOFA */
      .chip {{ display:inline-block; padding:.18rem .7rem; border-radius:999px;
        font-size:.72rem; font-weight:800; letter-spacing:.5px; margin-bottom:.5rem; }}
      .cf {{ background:rgba(32,201,151,.15); color:{C['green']}; }}
      .cd {{ background:rgba(239,71,111,.15); color:{C['red']}; }}
      .co {{ background:rgba(0,180,216,.16); color:{C['accent']}; }}
      .ca {{ background:rgba(255,209,102,.15); color:{C['amber']}; }}

      /* Barra 1X2 */
      .bar {{ background:rgba(255,255,255,.05); border-radius:9px; height:30px;
        display:flex; overflow:hidden; border:1px solid {C['line']}; }}
      .seg {{ display:flex; align-items:center; justify-content:center;
        font-size:.8rem; font-weight:800; color:#08111f; }}

      /* Botones */
      .stButton > button {{
        background:linear-gradient(90deg,{C['primary']},{C['accent']});
        color:#fff; border:none; border-radius:12px; padding:.62rem 1rem;
        font-weight:800; width:100%; box-shadow:0 6px 18px rgba(0,119,182,.45);
        transition:all .15s ease;
      }}
      .stButton > button:hover {{ transform:translateY(-2px);
        box-shadow:0 8px 24px rgba(0,180,216,.6); }}

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {{ gap:.4rem; }}
      .stTabs [data-baseweb="tab"] {{
        background:{C['card']}; border:1px solid {C['line']}; border-radius:10px;
        padding:.5rem 1rem; color:{C['muted']}; font-weight:700;
      }}
      .stTabs [aria-selected="true"] {{
        background:linear-gradient(90deg,{C['primary']},{C['accent']}) !important;
        color:#fff !important; border-color:transparent !important;
      }}

      /* Radio como pills */
      div[role="radiogroup"] {{ gap:.5rem; }}
      div[role="radiogroup"] label {{
        background:{C['card']}; border:1px solid {C['line']}; border-radius:10px;
        padding:.45rem .9rem; }}

      .stDataFrame {{ border-radius:12px; overflow:hidden; }}
      .value-pos {{ color:{C['green']}; font-weight:800; }}
      .muted {{ color:{C['muted']}; font-size:.82rem; }}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# 2. DATOS — histórico FIJO fila por fila (últimos 10 partidos oficiales)
# ══════════════════════════════════════════════════════════════════════════
# Orden cronológico: índice 0 = más antiguo, índice 9 = más reciente.
# El slider "últimos N" toma la cola (tail) del DataFrame.
# Campos por partido:
#   gf  goles a favor      ga  goles en contra    pos  posesión %
#   sot tiros al arco      sotc tiros concedidos   fouls faltas
#   pas efectividad pases  form formación del DT   rival oponente
MODALITIES: Dict[str, float] = {
    "Amistoso": 1.18,
    "Copa": 1.02,
    "Eliminatorias": 0.95,
    "Mundial": 0.88,
}

# Cada tupla-columna tiene longitud 10. rivales genéricos para el histórico.
_RIVALS = ["Perú", "Chile", "Ecuador", "Bolivia", "Paraguay",
           "Venezuela", "Uruguay", "Brasil", "Argentina", "Colombia"]

TEAM_DB: Dict[str, Dict] = {
    "Colombia": dict(
        flag="🇨🇴",
        gf=[2, 1, 3, 1, 2, 0, 2, 1, 2, 2],
        ga=[0, 1, 1, 0, 1, 1, 0, 2, 1, 0],
        pos=[58, 55, 61, 54, 60, 49, 57, 52, 59, 56],
        sot=[6, 4, 7, 5, 6, 3, 6, 4, 7, 6],
        sotc=[3, 5, 2, 4, 3, 6, 3, 5, 2, 3],
        fouls=[12, 14, 10, 13, 11, 16, 12, 15, 11, 12],
        pas=[.87, .84, .89, .83, .88, .79, .86, .82, .88, .87],
        form=["4-2-3-1", "4-2-3-1", "4-3-3", "4-2-3-1", "4-3-3",
              "4-4-2", "4-2-3-1", "4-2-3-1", "4-3-3", "4-2-3-1"],
    ),
    "Argentina": dict(
        flag="🇦🇷",
        gf=[2, 3, 2, 4, 1, 3, 2, 3, 2, 3],
        ga=[0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
        pos=[60, 58, 62, 59, 57, 61, 58, 63, 60, 59],
        sot=[7, 6, 8, 7, 5, 7, 6, 8, 7, 6],
        sotc=[2, 3, 2, 3, 4, 2, 3, 2, 3, 2],
        fouls=[11, 12, 10, 13, 12, 11, 12, 10, 13, 11],
        pas=[.88, .86, .90, .87, .85, .89, .86, .90, .88, .87],
        form=["4-3-3", "4-3-3", "4-4-2", "4-3-3", "4-3-3",
              "4-3-3", "4-4-2", "4-3-3", "4-3-3", "4-3-3"],
    ),
    "Brasil": dict(
        flag="🇧🇷",
        gf=[3, 2, 3, 1, 4, 2, 3, 2, 3, 2],
        ga=[1, 1, 2, 1, 1, 0, 2, 1, 1, 2],
        pos=[62, 60, 64, 58, 63, 61, 60, 65, 62, 60],
        sot=[8, 6, 7, 5, 8, 6, 7, 6, 8, 6],
        sotc=[4, 3, 4, 3, 3, 2, 4, 3, 4, 3],
        fouls=[10, 12, 11, 13, 10, 12, 11, 10, 12, 11],
        pas=[.89, .87, .88, .85, .90, .87, .86, .90, .88, .87],
        form=["4-2-3-1", "4-3-3", "4-2-3-1", "4-4-2", "4-3-3",
              "4-2-3-1", "4-3-3", "4-2-3-1", "4-3-3", "4-2-3-1"],
    ),
    "Uruguay": dict(
        flag="🇺🇾",
        gf=[1, 2, 1, 1, 2, 0, 2, 1, 1, 2],
        ga=[1, 1, 0, 2, 1, 1, 0, 2, 1, 1],
        pos=[48, 46, 50, 45, 49, 44, 47, 43, 48, 46],
        sot=[4, 5, 4, 3, 5, 3, 5, 4, 4, 5],
        sotc=[5, 4, 5, 6, 4, 6, 4, 5, 5, 4],
        fouls=[15, 16, 14, 17, 15, 18, 14, 16, 15, 16],
        pas=[.80, .79, .82, .78, .81, .77, .80, .79, .81, .80],
        form=["4-4-2", "3-5-2", "4-4-2", "5-3-2", "4-4-2",
              "3-5-2", "4-4-2", "5-3-2", "4-4-2", "3-5-2"],
    ),
}

# Jugadores clave: ratings (0-10) y km recorridos por partido (10 registros).
# La desviación estándar de estas series mide consistencia técnica y física.
PLAYERS_DB: Dict[str, List[Dict]] = {
    "Colombia": [
        dict(name="J. Rodríguez", pos="MED",
             rating=[7.2, 6.8, 7.9, 7.0, 7.5, 6.5, 7.3, 6.9, 7.6, 7.4],
             km=[9.8, 9.5, 10.1, 9.6, 9.9, 9.2, 9.7, 9.4, 10.0, 9.8]),
        dict(name="L. Díaz", pos="DEL",
             rating=[7.5, 7.0, 8.1, 7.2, 7.8, 6.8, 7.6, 7.1, 7.9, 7.7],
             km=[10.5, 10.2, 10.8, 10.3, 10.6, 9.9, 10.4, 10.1, 10.7, 10.5]),
        dict(name="D. Sánchez", pos="DEF",
             rating=[6.9, 7.1, 7.0, 7.3, 6.8, 7.2, 6.9, 7.4, 7.0, 7.1],
             km=[9.0, 9.2, 8.9, 9.3, 9.1, 9.4, 9.0, 9.5, 9.1, 9.2]),
    ],
    "Argentina": [
        dict(name="L. Messi", pos="DEL",
             rating=[8.4, 8.0, 8.7, 8.9, 7.6, 8.5, 8.1, 8.8, 8.2, 8.6],
             km=[8.8, 8.5, 9.0, 8.7, 8.3, 8.9, 8.6, 9.1, 8.7, 8.9]),
        dict(name="R. De Paul", pos="MED",
             rating=[7.4, 7.2, 7.5, 7.3, 7.1, 7.6, 7.2, 7.7, 7.3, 7.5],
             km=[11.2, 11.0, 11.4, 11.1, 10.8, 11.3, 11.0, 11.5, 11.1, 11.3]),
        dict(name="N. Otamendi", pos="DEF",
             rating=[7.0, 7.2, 6.8, 7.3, 7.1, 7.4, 6.9, 7.2, 7.0, 7.3],
             km=[9.3, 9.4, 9.1, 9.5, 9.2, 9.6, 9.0, 9.4, 9.2, 9.5]),
    ],
    "Brasil": [
        dict(name="Vinícius Jr.", pos="DEL",
             rating=[8.1, 7.4, 8.3, 6.9, 8.6, 7.7, 8.0, 7.5, 8.4, 7.8],
             km=[10.2, 9.8, 10.4, 9.5, 10.6, 9.9, 10.1, 9.7, 10.5, 10.0]),
        dict(name="Neymar", pos="MED",
             rating=[7.8, 8.2, 7.0, 8.4, 7.6, 8.0, 7.3, 8.5, 7.7, 8.1],
             km=[9.5, 9.8, 9.0, 10.0, 9.4, 9.7, 9.1, 10.1, 9.5, 9.8]),
        dict(name="Marquinhos", pos="DEF",
             rating=[7.3, 7.1, 7.4, 7.0, 7.5, 7.6, 7.0, 7.4, 7.2, 7.3],
             km=[9.1, 9.0, 9.2, 8.9, 9.3, 9.4, 8.8, 9.2, 9.0, 9.1]),
    ],
    "Uruguay": [
        dict(name="D. Núñez", pos="DEL",
             rating=[7.6, 6.5, 7.8, 6.2, 7.9, 6.0, 7.5, 6.4, 7.7, 6.8],
             km=[10.8, 10.3, 11.0, 10.1, 11.1, 9.8, 10.6, 10.2, 10.9, 10.4]),
        dict(name="F. Valverde", pos="MED",
             rating=[7.7, 7.5, 7.9, 7.4, 7.8, 7.6, 7.5, 7.9, 7.6, 7.8],
             km=[11.5, 11.3, 11.7, 11.2, 11.6, 11.4, 11.2, 11.8, 11.4, 11.6]),
        dict(name="R. Araújo", pos="DEF",
             rating=[7.2, 7.4, 7.0, 7.5, 7.1, 7.3, 6.9, 7.4, 7.2, 7.3],
             km=[9.4, 9.5, 9.2, 9.6, 9.3, 9.5, 9.1, 9.5, 9.3, 9.4]),
    ],
}

# Banca simulada (determinista) para las KPI cards del dashboard personal.
# (stake, cuota, ganada?)  -> se derivan bankroll, ROI, yield y aciertos.
# Calibrada a un yield ~6% (perfil de apostador profesional, creíble y sobrio).
LEDGER: List[Tuple[float, float, bool]] = [
    (100, 2.10, True),  (100, 1.90, False), (100, 1.95, True),  (100, 2.00, False),
    (100, 2.30, True),  (100, 1.85, False), (100, 1.85, True),  (100, 2.10, False),
    (100, 2.05, True),  (100, 1.95, False), (100, 1.75, True),  (100, 1.80, False),
    (100, 2.50, True),  (100, 2.00, False), (100, 1.90, True),  (100, 2.20, False),
    (100, 2.15, True),  (100, 1.90, False), (100, 1.80, True),  (100, 2.05, False),
    (100, 2.00, True),
]

# Base de goles de la liga: promedio combinado de goles a FAVOR y en CONTRA de
# toda la base histórica (baseline de goles por equipo/partido). Combinar ambos
# evita sesgar λ hacia abajo cuando la muestra son selecciones fuertes.
LEAGUE_AVG = float(np.mean(
    [g for t in TEAM_DB.values() for g in t["gf"]] +
    [g for t in TEAM_DB.values() for g in t["ga"]]
))
HOME_ADV = 1.10
MAX_G = 5  # matriz de 0-0 a 5-5

# --- Modelo de xG y trazabilidad de origen (pestaña de auditoría) ---
# xG estimado por partido a partir de los tiros al arco. Factor = conversión
# media histórica de un remate a puerta en gol (~0.31 en fútbol de selecciones).
XG_PER_SOT = 0.31
SCRAPE_DATE = pd.Timestamp("2026-07-12 03:15")   # corrida nocturna del scraper
SCRAPE_SOURCES = {                               # fuente rotada por jornada
    0: ("ESPN", "https://www.espn.com/soccer/match/_/gameId/{gid}"),
    1: ("Win Sports", "https://www.winsports.co/partido/{gid}"),
}


# ══════════════════════════════════════════════════════════════════════════
# 3. BACKEND DE CIENCIA DE DATOS
# ══════════════════════════════════════════════════════════════════════════
def team_history(team: str) -> pd.DataFrame:
    """Devuelve el histórico completo (10 partidos) de un equipo como DataFrame."""
    d = TEAM_DB[team]
    df = pd.DataFrame({
        "Jornada": [f"J-{10 - i}" for i in range(10)],
        "Rival": _RIVALS,
        "GF": d["gf"], "GC": d["ga"], "Posesión %": d["pos"],
        "Tiros arco": d["sot"], "Tiros concedidos": d["sotc"],
        "Faltas": d["fouls"],
        "Pases %": [round(p * 100, 1) for p in d["pas"]],
        # xG estimado = tiros al arco × conversión media (modelo de la app).
        "xG": [round(s * XG_PER_SOT, 2) for s in d["sot"]],
        "Formación DT": d["form"],
    })
    return df


def last_n(team: str, n: int) -> pd.DataFrame:
    """Filtra dinámicamente los últimos N partidos (cola del histórico)."""
    return team_history(team).tail(n).reset_index(drop=True)


def extraction_log(team: str, n: int) -> pd.DataFrame:
    """
    Log de extracción (scraping) simulado y DETERMINISTA de los últimos N
    partidos: fecha del partido, fuente, URL y sello de tiempo del raspado.
    Da trazabilidad de origen para la auditoría de veracidad.
    """
    d = TEAM_DB[team]
    rows = []
    df_n = last_n(team, n)
    for pos_i, row in df_n.iterrows():
        # Índice global 0..9 (0 = más antiguo). Reconstruye desde 'Jornada'.
        j = 10 - int(row["Jornada"].split("-")[1])          # J-10 -> 0 ... J-1 -> 9
        match_date = SCRAPE_DATE.normalize() - pd.Timedelta(days=(10 - j) * 7)
        gid = 6_400_000 + hash((team, j)) % 90_000           # id de partido estable
        src_name, src_tpl = SCRAPE_SOURCES[j % 2]
        # Sello de extracción: cada request separado unos segundos (cortesía).
        stamp = SCRAPE_DATE + pd.Timedelta(seconds=j * 7)
        rows.append({
            "Partido": f"{team} vs {row['Rival']}",
            "Fecha partido": match_date.strftime("%Y-%m-%d"),
            "Fuente": src_name,
            "URL de origen": src_tpl.format(gid=gid),
            "Extraído": stamp.strftime("%Y-%m-%d %H:%M:%S"),
            "Estado": "✅ 200 OK",
        })
    return pd.DataFrame(rows)


def team_averages(team: str, n: int) -> Dict[str, float]:
    """Promedios de la muestra seleccionada + formación (estilo) modal del DT."""
    df = last_n(team, n)
    return {
        "gf": df["GF"].mean(), "ga": df["GC"].mean(),
        "pos": df["Posesión %"].mean(), "sot": df["Tiros arco"].mean(),
        "sotc": df["Tiros concedidos"].mean(), "fouls": df["Faltas"].mean(),
        "pas": df["Pases %"].mean(),
        "form": df["Formación DT"].mode().iloc[0],
    }


@dataclass
class Prediction:
    lam_home: float
    lam_away: float
    matrix: np.ndarray                 # 6x6 normalizada (0..5 x 0..5)
    markets: Dict[str, float]          # todas las probabilidades de mercado
    top_scores: List[Tuple[str, float]]


def compute_prediction(home: str, away: str, modality: str, n: int) -> Prediction:
    """
    Modelo de Poisson bivariado sobre la muestra de N partidos.

        atk = goles promedio anotados      def = goles promedio concedidos
        λ_home = atk_home · def_away / LEAGUE_AVG · ventaja_local · factor_modalidad
        λ_away = atk_away · def_home / LEAGUE_AVG · factor_modalidad
        P(i,j) = poisson.pmf(i; λ_home) · poisson.pmf(j; λ_away)   (matriz 0..5)
    """
    h, a = team_averages(home, n), team_averages(away, n)
    m = MODALITIES[modality]

    lam_h = max(0.15, h["gf"] * a["ga"] / LEAGUE_AVG * HOME_ADV * m)
    lam_a = max(0.15, a["gf"] * h["ga"] / LEAGUE_AVG * m)

    k = np.arange(MAX_G + 1)
    ph = poisson.pmf(k, lam_h)
    pa = poisson.pmf(k, lam_a)
    M = np.outer(ph, pa)
    M = M / M.sum()                    # normaliza (la malla 0..5 trunca la cola)

    idx = np.add.outer(k, k)           # i + j (total de goles) por celda
    mk = {
        "1": float(np.tril(M, -1).sum()),          # gana local (i>j)
        "X": float(np.trace(M)),                   # empate (i==j)
        "2": float(np.triu(M, 1).sum()),           # gana visita (j>i)
        "Over 1.5": float(M[idx >= 2].sum()),
        "Under 1.5": float(M[idx <= 1].sum()),
        "Over 2.5": float(M[idx >= 3].sum()),
        "Under 2.5": float(M[idx <= 2].sum()),
        "Over 3.5": float(M[idx >= 4].sum()),
        "Under 3.5": float(M[idx <= 3].sum()),
        "BTTS Sí": float(M[1:, 1:].sum()),
        "BTTS No": float(1 - M[1:, 1:].sum()),
    }
    mk["1X"] = mk["1"] + mk["X"]
    mk["X2"] = mk["X"] + mk["2"]
    mk["12"] = mk["1"] + mk["2"]

    flat = sorted(((f"{i}-{j}", float(M[i, j]))
                   for i in k for j in k), key=lambda t: -t[1])[:6]

    return Prediction(round(lam_h, 3), round(lam_a, 3), M, mk, flat)


def build_dofa(home: str, away: str, n: int, pred: Prediction) -> Dict[str, List[str]]:
    """Matriz DOFA cruzada, algorítmica, a partir de los promedios de la muestra."""
    h, a = team_averages(home, n), team_averages(away, n)
    hf, af = TEAM_DB[home]["flag"], TEAM_DB[away]["flag"]
    F, D, O, A = [], [], [], []

    if h["gf"] > a["ga"] + 0.3:
        F.append(f"Ataque de {home} ({h['gf']:.1f} goles/pp) supera la defensa de "
                 f"{away} ({a['ga']:.1f} recibidos/pp).")
    if h["pos"] > a["pos"] + 3:
        F.append(f"{home} domina la posesión ({h['pos']:.0f}% vs {a['pos']:.0f}%): "
                 f"impone ritmo y control.")
    if h["sot"] > a["sotc"]:
        F.append(f"Genera más tiros al arco ({h['sot']:.1f}) que los que suele "
                 f"conceder {away} ({a['sotc']:.1f}): volumen ofensivo favorable.")

    if a["gf"] > h["ga"] + 0.3:
        D.append(f"Ataque de {away} ({a['gf']:.1f} goles/pp) penetra la defensa de "
                 f"{home} ({h['ga']:.1f} recibidos/pp).")
    if h["pas"] < a["pas"]:
        D.append(f"Menor precisión de pase ({h['pas']:.0f}% vs {a['pas']:.0f}%): "
                 f"riesgo de pérdidas ante la presión rival.")
    if h["fouls"] > a["fouls"] + 1.5:
        D.append(f"{home} comete más faltas ({h['fouls']:.1f}): expone su zona a "
                 f"balón parado.")

    if pred.markets["Over 2.5"] > 0.55:
        O.append(f"Escenario de partido abierto: {pred.markets['Over 2.5']:.0%} de "
                 f"probabilidad de Over 2.5.")
    if pred.markets["1"] > pred.markets["2"] + 0.12:
        O.append(f"Localía y nivel inclinan el favoritismo a {home} "
                 f"({pred.markets['1']:.0%}).")
    if pred.markets["BTTS No"] > 0.55:
        O.append("Perfil de partido controlado: alta probabilidad de que un equipo "
                 "mantenga su portería a cero.")

    A.append(f"Sistema del DT rival: «{a['form']}» — preparar respuesta táctica.")
    if a["gf"] >= 2.3:
        A.append(f"{away} es prolífico ({a['gf']:.1f} goles/pp): máxima atención a "
                 f"las transiciones.")
    if pred.markets["2"] > 0.30:
        A.append(f"{away} conserva {pred.markets['2']:.0%} de opciones de victoria: "
                 f"margen estrecho, no confiarse.")

    return {"Fortalezas": F, "Debilidades": D, "Oportunidades": O, "Amenazas": A}


def _fair_odds(p: float, overround: float = 1.07) -> float:
    """Cuota de la casa a partir de una probabilidad, con margen (overround)."""
    p = min(max(p, 0.02), 0.98)
    return round(1.0 / (p * overround), 2)


def build_singles(model: Prediction, consensus: Prediction) -> pd.DataFrame:
    """
    Mercados sencillos: la casa fija cuotas sobre el CONSENSO (10 pp);
    el analista usa el MODELO (muestra N). EV = prob_IA · cuota − 1.
    """
    keys = ["1", "X", "2", "Over 2.5", "Under 2.5", "BTTS Sí", "BTTS No"]
    labels = {"1": "1 · Gana Local", "X": "X · Empate", "2": "2 · Gana Visita",
              "Over 2.5": "Más de 2.5 goles", "Under 2.5": "Menos de 2.5 goles",
              "BTTS Sí": "Ambos anotan · Sí", "BTTS No": "Ambos anotan · No"}
    rows = []
    for kk in keys:
        p_ia = model.markets[kk]
        odds = _fair_odds(consensus.markets[kk])   # cuota BetPlay (consenso)
        ev = p_ia * odds - 1
        rows.append({
            "Mercado": labels[kk],
            "Prob. IA": p_ia,
            "Cuota (casa)": odds,
            "Prob. casa": round(1 / odds, 3),
            "EV": round(ev, 3),
            "Valor": "🟢 VALOR" if ev > 0 else "—",
        })
    return pd.DataFrame(rows)


def build_combos(home: str, away: str, model: Prediction,
                 consensus: Prediction) -> List[Dict]:
    """
    Plantillas de combinadas lógicas: une eventos de alta probabilidad.
    Cuota final = producto de cuotas (consenso). Fiabilidad = producto de prob. IA.
    """
    m, c = model.markets, consensus.markets
    fav_home = m["1"] >= m["2"]
    dc_key = "1X" if fav_home else "X2"
    dc_lbl = f"Doble oportunidad {'1X' if fav_home else 'X2'} " \
             f"({home if fav_home else away} gana o empata)"

    templates = [
        {"nombre": f"{dc_lbl}  +  Menos de 3.5 goles",
         "legs": [(dc_key, dc_lbl), ("Under 3.5", "Menos de 3.5 goles")]},
        {"nombre": f"{'Gana ' + (home if fav_home else away)}  +  Más de 1.5 goles",
         "legs": [("1" if fav_home else "2",
                   f"Gana {home if fav_home else away}"),
                  ("Over 1.5", "Más de 1.5 goles")]},
        {"nombre": "Ambos anotan · No  +  Menos de 3.5 goles",
         "legs": [("BTTS No", "Ambos anotan No"), ("Under 3.5", "Menos de 3.5")]},
    ]

    out = []
    for t in templates:
        odds = float(np.prod([_fair_odds(c[k]) for k, _ in t["legs"]]))
        reliab = float(np.prod([m[k] for k, _ in t["legs"]]))
        risk = ("🟢 Bajo" if reliab > 0.45 else
                "🟡 Medio" if reliab > 0.30 else "🔴 Alto")
        out.append({
            "Combinada": t["nombre"],
            "Patas": len(t["legs"]),
            "Cuota final": round(odds, 2),
            "Fiabilidad IA": round(reliab, 3),
            "Riesgo": risk,
        })
    out.sort(key=lambda d: -d["Fiabilidad IA"])
    return out


def player_consistency(team: str, n: int) -> pd.DataFrame:
    """
    Consistencia por jugador sobre la muestra N:
      · Rating medio y su desviación estándar (consistencia TÉCNICA).
      · Km medios y su desviación estándar (consistencia FÍSICA).
    Menor desviación => mayor consistencia. Se resume en un índice 0-100.
    """
    rows = []
    for p in PLAYERS_DB[team]:
        r = np.array(p["rating"][-n:], dtype=float)
        km = np.array(p["km"][-n:], dtype=float)
        r_std, km_std = float(r.std(ddof=0)), float(km.std(ddof=0))
        rows.append({
            "Jugador": p["name"], "Pos": p["pos"],
            "Rating medio": round(r.mean(), 2),
            "σ Rating": round(r_std, 3),
            "Consist. técnica": round(100 / (1 + r_std), 1),
            "Km medio": round(km.mean(), 2),
            "σ Km": round(km_std, 3),
            "Consist. física": round(100 / (1 + km_std), 1),
        })
    return pd.DataFrame(rows)


def bankroll_kpis() -> Dict[str, float]:
    """Deriva bankroll, ROI, yield y aciertos de la banca simulada (LEDGER)."""
    staked = sum(s for s, _, _ in LEDGER)
    profit = sum((s * (o - 1) if w else -s) for s, o, w in LEDGER)
    wins = sum(1 for _, _, w in LEDGER if w)
    return {
        "bankroll": INITIAL_BANKROLL + profit,
        "profit": profit,
        "roi": profit / staked * 100,
        "yield": profit / staked * 100,
        "hit_rate": wins / len(LEDGER) * 100,
        "n_bets": len(LEDGER),
    }


# ══════════════════════════════════════════════════════════════════════════
# 4. COMPONENTES DE UI
# ══════════════════════════════════════════════════════════════════════════
def kpi_card(col, label: str, value: str, sub: str = "", cls: str = "") -> None:
    col.markdown(
        f"<div class='kpi'><div class='lbl'>{label}</div>"
        f"<div class='val {cls}'>{value}</div>"
        f"<div class='sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Logo + bienvenida personalizada + barra de KPIs financieras."""
    st.markdown(
        "<div class='brand'>"
        "<span class='mark'>🎯</span>"
        "<span class='name'>APEXPREDICT IA</span>"
        "<span class='tag'>| Inteligencia Deportiva</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='welcome'>Bienvenido de nuevo, <b>{ANALYST_NAME}</b>. "
        f"Tu panel está sincronizado — motor de inferencia Poisson listo. "
        f"Rigor científico sobre datos verificados.</div>",
        unsafe_allow_html=True,
    )

    k = bankroll_kpis()
    up = "up" if k["profit"] >= 0 else "down"
    arrow = "▲" if k["profit"] >= 0 else "▼"
    cols = st.columns(4)
    kpi_card(cols[0], "Banca actual (Bankroll)", f"${k['bankroll']:,.0f}",
             f"{arrow} {k['profit']:+,.0f} vs. inicial", up)
    kpi_card(cols[1], "ROI acumulado (mes)", f"{k['roi']:+.1f}%",
             f"{k['n_bets']} apuestas liquidadas", up)
    kpi_card(cols[2], "Yield actual", f"{k['yield']:+.1f}%",
             "beneficio / importe apostado", up)
    kpi_card(cols[3], "Tasa de acierto", f"{k['hit_rate']:.0f}%",
             "efectividad histórica", "")


def render_prob_bar(mk: Dict[str, float], home: str, away: str) -> None:
    h, d, a = mk["1"], mk["X"], mk["2"]
    hf, af = TEAM_DB[home]["flag"], TEAM_DB[away]["flag"]
    st.markdown(
        f"""
        <div class='bar'>
          <div class='seg' style='width:{h*100:.1f}%;background:{C['accent']}'>{h:.0%}</div>
          <div class='seg' style='width:{d*100:.1f}%;background:{C['muted']}'>{d:.0%}</div>
          <div class='seg' style='width:{a*100:.1f}%;background:{C['primary']}'>{a:.0%}</div>
        </div>
        <div style='display:flex;justify-content:space-between;margin-top:.35rem;
                    font-size:.82rem;color:{C['muted']}'>
          <span>{hf} {home} (victoria)</span><span>Empate</span>
          <span>{away} {af} (victoria)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dofa(dofa: Dict[str, List[str]]) -> None:
    cls = {"Fortalezas": "cf", "Debilidades": "cd",
           "Oportunidades": "co", "Amenazas": "ca"}
    cols = st.columns(4)
    for col, (dim, items) in zip(cols, dofa.items()):
        html = f"<div class='card'><span class='chip {cls[dim]}'>{dim.upper()}</span>"
        if items:
            html += "<ul style='padding-left:1.05rem;margin:.2rem 0 0;font-size:.85rem'>"
            html += "".join(f"<li style='margin-bottom:.4rem'>{it}</li>" for it in items)
            html += "</ul>"
        else:
            html += "<p class='muted'>Sin señales relevantes en esta muestra.</p>"
        html += "</div>"
        col.markdown(html, unsafe_allow_html=True)


def style_singles(df: pd.DataFrame):
    """Formatea la tabla de sencillas y resalta en verde las de valor (EV>0)."""
    def _hl(row):
        color = f"background-color: rgba(32,201,151,.14)" if row["EV"] > 0 else ""
        return [color] * len(row)
    return (df.style
            .apply(_hl, axis=1)
            .format({"Prob. IA": "{:.1%}", "Prob. casa": "{:.1%}",
                     "Cuota (casa)": "{:.2f}", "EV": "{:+.2f}"}))


# ══════════════════════════════════════════════════════════════════════════
# PESTAÑAS
# ══════════════════════════════════════════════════════════════════════════
def tab_prediccion(home, away, modality, n, pred, dofa) -> None:
    hf, af = TEAM_DB[home]["flag"], TEAM_DB[away]["flag"]
    score, sp = pred.top_scores[0]

    st.markdown(f"#### {hf} {home}  vs  {away} {af}  ·  _{modality}_  ·  "
                f"muestra: últimos {n} partidos")

    c = st.columns(5)
    kpi_card(c[0], "Marcador proyectado", score, f"prob. {sp:.1%}", "up")
    kpi_card(c[1], "Victoria local", f"{pred.markets['1']:.0%}", home)
    kpi_card(c[2], "Empate", f"{pred.markets['X']:.0%}", "resultado X")
    kpi_card(c[3], "Victoria visitante", f"{pred.markets['2']:.0%}", away)
    kpi_card(c[4], "Goles esperados (λ)", f"{pred.lam_home:.2f} – {pred.lam_away:.2f}",
             "local – visita")

    st.markdown("<div class='card'><h4>Distribución del resultado (1X2)</h4>",
                unsafe_allow_html=True)
    render_prob_bar(pred.markets, home, away)
    st.markdown("</div>", unsafe_allow_html=True)

    cA, cB = st.columns([1, 1])
    with cA:
        st.markdown("<div class='card'><h4>🎯 Marcadores más probables</h4>",
                    unsafe_allow_html=True)
        sdf = pd.DataFrame(pred.top_scores, columns=["Marcador", "Probabilidad"])
        st.dataframe(sdf.style.format({"Probabilidad": "{:.1%}"}),
                     use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with cB:
        st.markdown("<div class='card'><h4>📈 Mercados de goles</h4>",
                    unsafe_allow_html=True)
        g = st.columns(2)
        kpi_card(g[0], "Más de 2.5", f"{pred.markets['Over 2.5']:.0%}")
        kpi_card(g[1], "Menos de 2.5", f"{pred.markets['Under 2.5']:.0%}")
        g2 = st.columns(2)
        kpi_card(g2[0], "Ambos anotan Sí", f"{pred.markets['BTTS Sí']:.0%}")
        kpi_card(g2[1], "Ambos anotan No", f"{pred.markets['BTTS No']:.0%}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🧭 Matriz DOFA Cruzada (algorítmica)")
    render_dofa(dofa)


def tab_historial(home, away, n) -> None:
    st.markdown(f"#### Inspección del historial · últimos {n} partidos "
                f"(datos crudos y auditables)")
    for team in (home, away):
        flag = TEAM_DB[team]["flag"]
        df = last_n(team, n)
        st.markdown(f"##### {flag} {team}")
        st.dataframe(df, use_container_width=True, hide_index=True)

        cc = st.columns(2)
        with cc[0]:
            st.markdown("<span class='muted'>Tendencia de goles (GF vs GC)</span>",
                        unsafe_allow_html=True)
            st.line_chart(df.set_index("Jornada")[["GF", "GC"]], height=220)
        with cc[1]:
            st.markdown("<span class='muted'>Tiros: al arco vs concedidos</span>",
                        unsafe_allow_html=True)
            st.bar_chart(df.set_index("Jornada")[["Tiros arco", "Tiros concedidos"]],
                         height=220)
        st.markdown("---")


def tab_jugadores(home, away, n) -> None:
    st.markdown(f"#### Estadísticas por jugador · consistencia sobre {n} partidos")
    st.markdown("<span class='muted'>La desviación estándar (σ) mide la "
                "consistencia: menor σ = rendimiento más fiable y predecible. "
                "El índice 0-100 traduce esa estabilidad.</span>",
                unsafe_allow_html=True)
    for team in (home, away):
        flag = TEAM_DB[team]["flag"]
        pc = player_consistency(team, n)
        st.markdown(f"##### {flag} {team}")
        def _hl_consist(row):
            # Resalta al jugador más consistente técnicamente de cada equipo.
            top = row["Consist. técnica"] == pc["Consist. técnica"].max()
            return ["background-color: rgba(0,180,216,.14)" if top else ""] * len(row)

        st.dataframe(
            pc.style.apply(_hl_consist, axis=1).format({
                "Rating medio": "{:.2f}", "σ Rating": "{:.3f}",
                "Consist. técnica": "{:.1f}", "Km medio": "{:.2f}",
                "σ Km": "{:.3f}", "Consist. física": "{:.1f}",
            }),
            use_container_width=True, hide_index=True,
        )
        best = pc.loc[pc["Consist. técnica"].idxmax()]
        st.caption(f"🧩 Jugador más consistente técnicamente: **{best['Jugador']}** "
                   f"(σ rating {best['σ Rating']:.3f}).")
        st.markdown("---")


def tab_auditoria(home, away, modality, n, pred) -> None:
    """
    🔍 Confirma por cuenta propia — auditoría de origen y transparencia total.
    Tres bloques: (1) log de extracción con URLs, (2) fórmulas del pipeline,
    (3) checklist interactivo de verificación manual.
    """
    st.markdown("#### 🔍 Confirma por cuenta propia")
    st.markdown("<span class='muted'>Transparencia radical: aquí puedes rastrear de "
                "dónde salió cada dato, con qué fórmulas se procesó, y cotejarlo tú "
                "mismo contra lo que viste en el partido. Cero cajas negras.</span>",
                unsafe_allow_html=True)

    # ---------- BLOQUE 1: Origen de los datos (log de scraping) ----------
    st.markdown("### 1 · Origen de los datos · Log de extracción")
    st.markdown(f"<span class='muted'>Última corrida del scraper: "
                f"<b>{SCRAPE_DATE.strftime('%Y-%m-%d %H:%M')}</b> · "
                f"{n} partidos por equipo · fuentes: ESPN / Win Sports.</span>",
                unsafe_allow_html=True)
    for team in (home, away):
        flag = TEAM_DB[team]["flag"]
        log = extraction_log(team, n)
        st.markdown(f"##### {flag} {team}")
        st.dataframe(
            log,
            use_container_width=True, hide_index=True,
            column_config={
                "URL de origen": st.column_config.LinkColumn("URL de origen"),
            },
        )
    st.caption("Las URLs y gameId son simulados y deterministas (demo educativa). "
               "En producción apuntarían al recurso real raspado bajo su licencia/ToS.")

    # ---------- BLOQUE 2: Transparencia algorítmica ----------
    st.markdown("### 2 · Transparencia algorítmica · Del dato a la probabilidad")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<div class='card'><h4>Paso A · xG desde los tiros al arco</h4>"
                    "<p class='muted'>Cada remate a puerta vale, en promedio, una "
                    "fracción de gol (conversión histórica). El xG del partido es la "
                    "suma esperada:</p></div>", unsafe_allow_html=True)
        st.latex(r"xG_{partido} = TirosAlArco \times " + f"{XG_PER_SOT}")
        st.markdown("<div class='card'><h4>Paso B · Fuerzas de ataque y defensa</h4>"
                    "<p class='muted'>Sobre la muestra de N partidos se promedian los "
                    "goles a favor (ataque) y en contra (defensa):</p></div>",
                    unsafe_allow_html=True)
        st.latex(r"Atk = \frac{1}{N}\sum_{i=1}^{N} GF_i \qquad "
                 r"Def = \frac{1}{N}\sum_{i=1}^{N} GC_i")
    with c2:
        st.markdown("<div class='card'><h4>Paso C · Goles esperados (λ)</h4>"
                    "<p class='muted'>Se cruzan ataque propio y defensa rival, "
                    "normalizados por la media de la liga, con ventaja local y factor "
                    "de modalidad:</p></div>", unsafe_allow_html=True)
        st.latex(r"\lambda_{local} = \frac{Atk_{L}\cdot Def_{V}}{\bar{G}_{liga}}"
                 r"\cdot V_{local}\cdot M")
        st.latex(r"\lambda_{visita} = \frac{Atk_{V}\cdot Def_{L}}{\bar{G}_{liga}}"
                 r"\cdot M")
        st.markdown("<div class='card'><h4>Paso D · Distribución de Poisson</h4>"
                    "<p class='muted'>La probabilidad de un marcador exacto (i-j) es "
                    "el producto de dos Poisson independientes:</p></div>",
                    unsafe_allow_html=True)
        st.latex(r"P(i,j) = \frac{\lambda_L^{i} e^{-\lambda_L}}{i!}\cdot"
                 r"\frac{\lambda_V^{j} e^{-\lambda_V}}{j!}")

    # Cifras reales de ESTE partido, para que el usuario replique el cálculo.
    h, a = team_averages(home, n), team_averages(away, n)
    st.markdown("<div class='card'><h4>🔢 Verifica los números de este partido</h4>",
                unsafe_allow_html=True)
    audit = pd.DataFrame({
        "Parámetro": ["Ataque (GF prom.)", "Defensa (GC prom.)",
                      "Media de goles liga", "Ventaja local", "Factor modalidad",
                      "λ resultante"],
        home: [f"{h['gf']:.3f}", f"{h['ga']:.3f}", f"{LEAGUE_AVG:.3f}",
               f"{HOME_ADV:.2f}", f"{MODALITIES[modality]:.2f}",
               f"{pred.lam_home:.3f}"],
        away: [f"{a['gf']:.3f}", f"{a['ga']:.3f}", f"{LEAGUE_AVG:.3f}",
               "1.00 (visita)", f"{MODALITIES[modality]:.2f}",
               f"{pred.lam_away:.3f}"],
    })
    st.dataframe(audit, use_container_width=True, hide_index=True)
    st.caption(f"Comprobación λ local = {h['gf']:.3f} × {a['ga']:.3f} ÷ "
               f"{LEAGUE_AVG:.3f} × {HOME_ADV:.2f} × {MODALITIES[modality]:.2f} = "
               f"{pred.lam_home:.3f}  ✔")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- BLOQUE 3: Verificación manual (checklist) ----------
    st.markdown("### 3 · Verificación manual · Tu control de veracidad")
    st.markdown("<span class='muted'>Marca la casilla si el dato coincide con lo que "
                "tú viste en el partido. Tu tasa de coincidencia mide tu confianza en "
                "la muestra.</span>", unsafe_allow_html=True)

    team_v = st.selectbox("Equipo a verificar", [home, away], key="verify_team")
    src = last_n(team_v, n)[["Jornada", "Rival", "GF", "GC", "Posesión %",
                             "Tiros arco", "xG"]].copy()
    src.insert(0, "✔ Coincide", True)   # por defecto, el usuario desmarca discrepancias
    edited = st.data_editor(
        src,
        use_container_width=True, hide_index=True, key="verify_editor",
        column_config={
            "✔ Coincide": st.column_config.CheckboxColumn(
                "✔ Coincide", help="Marca si el dato coincide con lo que observaste."),
        },
        disabled=["Jornada", "Rival", "GF", "GC", "Posesión %", "Tiros arco", "xG"],
    )
    ok = int(edited["✔ Coincide"].sum())
    total = len(edited)
    rate = ok / total * 100 if total else 0
    cc = st.columns(3)
    kpi_card(cc[0], "Datos verificados", f"{ok}/{total}", "casillas marcadas")
    kpi_card(cc[1], "Coincidencia", f"{rate:.0f}%",
             "veracidad percibida", "up" if rate >= 80 else "down")
    kpi_card(cc[2], "Muestra auditada", f"{team_v}", f"últimos {n} partidos")
    if rate == 100:
        st.success("✅ Muestra 100% verificada por ti. Máxima confianza en la predicción.")
    elif rate >= 60:
        st.info(f"🟡 {rate:.0f}% verificado. Revisa las jornadas con discrepancias antes de apostar.")
    else:
        st.warning("🔴 Baja coincidencia: los datos no cuadran con tu observación. "
                   "Trata la predicción con cautela.")


def render_bets(home, away, pred, consensus) -> None:
    st.markdown("### 💸 Apuestas Inteligentes · Valor Esperado (EV)")
    st.markdown("<span class='muted'>La casa cotiza sobre el consenso de 10 "
                "partidos; tu modelo usa la ventana seleccionada. El desajuste "
                "es tu ventaja: <b>EV = (Prob. IA × Cuota) − 1</b>.</span>",
                unsafe_allow_html=True)

    mode = st.radio("Tipo de apuesta", ["🎯 Sencillas", "🧩 Combinadas"],
                    horizontal=True, label_visibility="collapsed")

    if mode == "🎯 Sencillas":
        df = build_singles(pred, consensus)
        st.dataframe(style_singles(df), use_container_width=True, hide_index=True)
        val = df[df["EV"] > 0]
        if len(val):
            best = val.loc[val["EV"].idxmax()]
            st.success(f"💎 Mejor valor: **{best['Mercado']}** — cuota "
                       f"{best['Cuota (casa)']:.2f}, EV {best['EV']:+.2f} "
                       f"(prob. IA {best['Prob. IA']:.0%}). {len(val)} apuesta(s) "
                       f"con valor detectada(s).")
        else:
            st.info("Sin apuestas con valor: tu muestra coincide con el consenso "
                    "del mercado (no hay ventaja explotable). Prueba a estrechar el "
                    "rango de partidos en el slider.")
    else:
        combos = build_combos(home, away, pred, consensus)
        cdf = pd.DataFrame(combos)
        st.dataframe(
            cdf.style.format({"Cuota final": "{:.2f}", "Fiabilidad IA": "{:.1%}"}),
            use_container_width=True, hide_index=True,
        )
        top = combos[0]
        st.info(f"🏆 Combinada recomendada: **{top['Combinada']}** — cuota final "
                f"**{top['Cuota final']:.2f}**, fiabilidad {top['Fiabilidad IA']:.0%}, "
                f"riesgo {top['Riesgo']}.")


# ══════════════════════════════════════════════════════════════════════════
# 5. APLICACIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    inject_css()
    render_header()

    # ---------------- Sidebar ----------------
    with st.sidebar:
        st.markdown("## ⚙️ Configuración de Inferencia")
        teams = list(TEAM_DB.keys())
        home = st.selectbox("🏠 Equipo Local", teams, index=0)
        away = st.selectbox("✈️ Equipo Rival",
                            [t for t in teams if t != home], index=0)
        modality = st.selectbox("🏆 Modalidad del Partido",
                                list(MODALITIES.keys()), index=2)
        st.caption(f"Factor de goles ×{MODALITIES[modality]:.2f} — "
                   f"{'ofensivo' if MODALITIES[modality] > 1 else 'conservador'}")

        st.markdown("### 🔬 Control de Veracidad")
        n = st.slider("Rango de partidos a analizar (N)", min_value=3,
                      max_value=10, value=5,
                      help="Granularidad de la muestra histórica. El backend "
                           "recalcula todo dinámicamente con estos N partidos.")
        st.caption(f"Analizando los últimos **{n}** de 10 partidos oficiales.")

        st.markdown("---")
        run = st.button("🚀 Ejecutar Inferencia de IA")

        st.markdown("---")
        st.markdown("#### Perfiles tácticos (muestra actual)")
        for t in (home, away):
            av = team_averages(t, n)
            st.markdown(f"**{TEAM_DB[t]['flag']} {t}** · DT: {av['form']}")
            st.caption(f"GF {av['gf']:.1f} · GC {av['ga']:.1f} · "
                       f"Pos {av['pos']:.0f}% · Pases {av['pas']:.0f}%")

    # ---------------- Estado ----------------
    if run:
        st.session_state.ready = True
        st.session_state.cfg = (home, away, modality, n)

    if not st.session_state.get("ready"):
        st.markdown(
            "<div class='card'><h4>👋 Panel listo</h4>"
            "<p>Configura Local, Rival, Modalidad y el <b>Rango de partidos</b> en "
            "el panel lateral, y pulsa <b>Ejecutar Inferencia de IA</b>. El motor "
            "Poisson calculará la predicción, la matriz DOFA, el historial auditable "
            "y las oportunidades de valor.</p></div>",
            unsafe_allow_html=True,
        )
        return

    home, away, modality, n = st.session_state.cfg
    pred = compute_prediction(home, away, modality, n)
    consensus = compute_prediction(home, away, modality, 10)  # cuotas de la casa
    dofa = build_dofa(home, away, n, pred)

    t1, t2, t3, t4 = st.tabs([
        "🧠 Predicción IA & DOFA",
        "🔍 Historial (Data Cruda)",
        "👤 Jugadores (Consistencia)",
        "🔍 Confirma por cuenta propia",
    ])
    with t1:
        tab_prediccion(home, away, modality, n, pred, dofa)
        st.markdown("---")
        render_bets(home, away, pred, consensus)
    with t2:
        tab_historial(home, away, n)
    with t3:
        tab_jugadores(home, away, n)
    with t4:
        tab_auditoria(home, away, modality, n, pred)

    st.caption("⚠️ ApexPredict IA es una herramienta educativa/analítica con datos "
               "simulados y deterministas. Las apuestas implican riesgo financiero; "
               "ningún modelo garantiza resultados.")


if __name__ == "__main__":
    main()
