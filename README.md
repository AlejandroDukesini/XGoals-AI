# RES-IA-PART-FUT
### Sistema de IA para análisis, predicción y valoración de apuestas de fútbol

Analiza los **últimos N partidos** de dos equipos (Local y Rival), aplica
*ingeniería inversa* sobre eventos y estadísticas para deducir patrones tácticos
del DT y estado de los jugadores, genera una **matriz DOFA cruzada**, predice el
resultado con **Poisson/Dixon-Coles**, detecta **value bets** frente a las cuotas
del mercado y lleva un **P&L** con ROI/Yield. Incluye una app web premium
(**ApexPredict Multi-Engine IA**, `app.py`).

> ⚠️ **Uso responsable.** Herramienta educativa/analítica. El scraping debe
> respetar los Términos de Servicio y `robots.txt` de cada sitio; para producción
> usa feeds de datos/cuotas **con licencia**. Las apuestas implican riesgo
> financiero: ningún modelo garantiza ganancias.

---

## 1. Arquitectura

```
                 +------------------------------------------------+
                 |                    main.py                     |
                 |                (orquestador)                   |
                 +------+----------------+----------------+-------+
                        |                |                |
        +---------------v----+   +-------v--------+   +---v-------------+
        |  1. INGESTION ETL  |   |  2. PIPELINE   |   |  5. PREDICCION  |
        |  src/ingestion     |   |  src/pipeline  |   |  src/prediction |
        |  - scraper.py      |-->|  6 puntos ->   |-->|  - poisson      |
        |  - odds_scraper.py |   |  TeamReport    |   |  - ml_model     |
        |  - synthetic.py    |   +-------+--------+   |  - value_bets   |
        |  - schemas.py      |           |            +---+-------------+
        +--------------------+   +-------v--------+       |
                                 |  3. DOFA CRUZ. |       |
                                 |  src/dofa      |-------+  (ajusta lambda)
                                 |  cross_swot.py |
                                 +----------------+       +-----------------+
                                                          | 6. P&L / KPIs   |
                                                          | src/performance |
                                                          | pnl.py (SQLite) |
                                                          +-----------------+
```

**Principio de diseño:** *adaptadores + contratos*. Todo el sistema habla el
mismo esquema canónico (`schemas.py`); da igual si los datos vienen del scraper
real, de una API licenciada o del generador sintético — el resto del pipeline no
cambia. Cada punto de análisis es una **función pura** (DataFrame → dict),
testeable en aislamiento.

---

## 2. Estructura de datos (contratos)

Jerarquía canónica (`src/ingestion/schemas.py`):

```
MatchData                        # 1 partido de 1 equipo
 |- team_stats: TeamMatchStats   # posesión, tiros, xG, ppda, línea, directness...
 \- players:   [PlayerMatchStats]  # rating, xG/xA, duelos, pases, avg_x/avg_y...

MarketOdds                       # cuotas 1X2 / Over-Under / BTTS
```

Un equipo = `List[MatchData]`. Se tabula a dos DataFrames: `team_df`
(1 fila/partido) y `player_df` (1 fila/jugador-partido) en `pipeline/features.py`.

---

## 3. Los 6 puntos de análisis (`pipeline/team_report.py`)

| # | Punto | Qué deduce | Técnica |
|---|-------|-----------|---------|
| 1 | Estadística general | Tendencias de goles/xG/posesión, forma | Medias, forma ponderada por recencia |
| 2 | Estadística por partido | Estabilidad vs. equipo errático | Desv. estándar, coef. de variación |
| 3 | Comportamiento jugador | **Fatiga** y zonas (heatmap invertido) | Carga de minutos + pendiente de rating; centroides (avg_x, avg_y) |
| 4 | Habilidad/talento | Eficiencia por 90' | G+A/90, pases clave/90, duelos, over-xG |
| 5 | Consistencia | Fiabilidad del rendimiento | Std de ratings → índice de consistencia |
| 6 | Táctica del DT | **Sistema deducido** | Reglas *fuzzy* sobre PPDA, línea, directness, posesión |

La *ingeniería inversa* vive en los puntos **3** (deducir estado físico y zona a
partir de eventos agregados) y **6** (deducir el plan del DT sin que venga
etiquetado).

---

## 4. Matriz DOFA cruzada (`dofa/cross_swot.py`)

No describe equipos por separado: cruza sus informes.

- **Fortalezas** — mi ataque (xG) vs. su defensa (goles concedidos) → *huecos*.
- **Debilidades** — mi línea alta vs. su juego directo → rutas de contraataque.
- **Oportunidades** — rachas de forma / delanteros sobre su xG.
- **Amenazas** — flexibilidad de esquema del DT → probabilidad de replanteo; fatiga.

Cada hallazgo produce un ajuste multiplicativo `lambda_adjust` que se inyecta al
motor de predicción, cerrando el bucle análisis → pronóstico.

---

## 5. Modelos matemáticos

### xG (expected goals)
Probabilidad de que un remate termine en gol dado su contexto. Se agrega por
equipo/jugador y se usa como **proxy de calidad de generación**, más estable que
los goles reales (que tienen mucha varianza en pocas jornadas).

### Poisson bivariado + corrección Dixon-Coles (`prediction/poisson_model.py`)
Fuerzas relativas a la media de liga:

```
attack_i  = xG_for_i / media_liga
defense_i = xG_against_i / media_liga
```

Tasas esperadas del partido:

```
lambda_home = media_liga * attack_home * defense_away * ventaja_local * adj_home
lambda_away = media_liga * attack_away * defense_home                 * adj_away
```

Probabilidad de marcador (x, y):

```
P(x, y) = Pois(x; lambda_home) * Pois(y; lambda_away) * tau(x, y)
```

`tau` (Dixon-Coles, rho<0) corrige la subestimación de empates de pocos goles
(0-0, 1-0, 0-1, 1-1). Sumando la malla de resultados se obtienen **1X2**,
**Over/Under 2.5** y **BTTS**.

### Valor esperado y staking (`prediction/value_bets.py`)
```
q     = 1 / cuota                     # prob. implícita (con margen de la casa)
edge  = p_modelo - q
EV    = p_modelo * cuota - 1          # >0  =>  value bet
Kelly = (b*p - (1-p)) / b * fraccion  # b = cuota - 1  (Kelly fraccionado)
```

### KPIs de P&L (`performance/pnl.py`)
`ROI = Yield = profit / stake_total`, hit rate, CLV (valor vs. cierre) y curva de
capital (bankroll).

---

## 6. Instalación y ejecución

### Requisitos previos
- **Python 3.11 o superior** (probado en 3.13). Comprueba con `python --version`.
- **pip** actualizado: `python -m pip install --upgrade pip`.
- (Opcional) **Google Chrome** solo si activas el scraping con Selenium.

### Instalación (recomendado: entorno virtual)

**Windows (PowerShell):**
```powershell
cd "RES-IA-PART-FUT"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux (bash):**
```bash
cd RES-IA-PART-FUT
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Mínimo solo para la **app web**: `pip install streamlit==1.40.0 pandas numpy scipy`

### A) Ejecutar el pipeline backend (CLI)
Flujo end-to-end (ETL → 6 puntos → DOFA → Poisson → value bets → P&L) con datos
sintéticos, sin conexión:
```bash
python main.py
```
Para datos reales: en `config.py` pon `OFFLINE_MODE = False` (y `USE_SELENIUM`
según el sitio) y ajusta URLs/selectores en `scraper.py` / `odds_scraper.py` a tu
fuente **con licencia**.

### B) Ejecutar la app web (ApexPredict Multi-Engine IA)
```bash
streamlit run app.py
# Si 'streamlit' no está en el PATH:
python -m streamlit run app.py
```
Se abre en `http://localhost:8501`. Configura torneo, equipos, origen de datos,
muestra (slider) y perfil de apuestas en la barra lateral, y pulsa **Ejecutar
Inferencia de IA**.

### Solución de problemas (Windows)
| Síntoma | Causa | Solución |
|---|---|---|
| `ModuleNotFoundError: streamlit.proto` | Streamlit roto por **rutas largas** | `pip install "streamlit==1.40.0"` |
| `'streamlit' no se reconoce...` | script fuera del `PATH` | usa `python -m streamlit run app.py` |
| `UnicodeEncodeError` en consola | consola en cp1252 | `set PYTHONIOENCODING=utf-8` o usa la app web |
| `background_gradient requires matplotlib` | falta matplotlib | ya evitado; si aparece, `pip install matplotlib` |

### Estructura de módulos
```
config.py            parámetros centrales (ventana, ligas, Kelly, rutas)
main.py              orquestador end-to-end (backend CLI)
app.py               ApexPredict Multi-Engine IA (app web Streamlit)
requirements.txt     dependencias
src/
  ingestion/         ETL: scraping, cuotas, esquemas, sintético
  pipeline/          6 puntos -> TeamReport
  dofa/              matriz DOFA cruzada
  prediction/        poisson + ml + value bets
  performance/       P&L en SQLite + KPIs
```

---

## 7. App web: ApexPredict Multi-Engine IA (`app.py`)

Interfaz empresarial en azul profundo con motor multi-torneo:

- **Catálogo global**: Mundial 2026 (por confederaciones), Liga BetPlay, Champions
  League y MLS / Otros. Los dropdowns de equipos se actualizan según el torneo.
- **Data Sourcing configurable**: ESPN Scraper, Win Sports Analytica o **Custom URL**
  (pega la URL que la IA "leerá"; se refleja en el log de extracción).
- **Perfil de apuestas**: stake, bankroll y aversión al riesgo (Conservador /
  Moderado / Agresivo) que filtra las recomendaciones.
- **Poisson ajustado por competitividad de la liga** (`scipy.stats.poisson.pmf`).
- **Pestañas de veracidad**: Inferencia & DOFA · Confirma por cuenta propia
  (log de extracción + fórmulas + checklist) · Centro de Apuestas (sencillas/parlays).

Ejecuta con `streamlit run app.py` (ver sección 6-B).

---

## 8. Roadmap sugerido
- Calibrar `HOME_ADVANTAGE`, `LEAGUE_AVG_GOALS` y `rho` con histórico real de la liga.
- Entrenar `OutcomeClassifier` (RandomForest) y hacer *ensemble* con Poisson.
- Conectar la app web al `BetLedger` real (SQLite) para P&L y curva de capital.
- Backtesting: replay de temporadas para medir ROI del sistema antes de operar.
