"""
config.py
Configuración central del sistema RES-IA-PART-FUT.
Todos los módulos importan desde aquí para no acoplar rutas/constantes.
"""
from pathlib import Path

# ------------------------------------------------------------------ #
# Rutas
# ------------------------------------------------------------------ #
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"          # HTML/JSON crudo del scraping
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"    # datos sintéticos para pruebas offline
DB_PATH = DATA_DIR / "betting.db"   # SQLite para el P&L

for _d in (RAW_DIR, PROCESSED_DIR, SAMPLE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# Parámetros de análisis
# ------------------------------------------------------------------ #
N_MATCHES = 5                 # ventana de análisis: últimos N partidos
HOME_ADVANTAGE = 1.15         # factor multiplicativo de localía (calibrable)
LEAGUE_AVG_GOALS = 1.35       # goles promedio por equipo/partido (ajustar por liga)
MAX_GOALS = 6                 # tope de goles para la malla de Poisson

# ------------------------------------------------------------------ #
# Motor de apuestas
# ------------------------------------------------------------------ #
MIN_EDGE = 0.05               # value bet solo si EV/edge supera este umbral (5%)
KELLY_FRACTION = 0.25         # fracción de Kelly (conservador) para el stake
BANKROLL_INICIAL = 1_000_000  # COP (pesos) para el registro P&L

# ------------------------------------------------------------------ #
# Scraping
# ------------------------------------------------------------------ #
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
REQUEST_TIMEOUT = 20
SCRAPE_DELAY = 2.0            # segundos de cortesía entre requests
USE_SELENIUM = False         # True si el sitio requiere render de JS

# Modo offline: usa datos sintéticos en vez de golpear la red.
# Ideal para desarrollo, tests y demostraciones reproducibles.
OFFLINE_MODE = True
