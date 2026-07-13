# RES-IA-PART-FUT
### Sistema de IA para análisis, predicción y valoración de apuestas de fútbol

Analiza los **últimos 5 partidos** de dos equipos (Local y Rival), aplica
*ingeniería inversa* sobre eventos y estadísticas para deducir patrones tácticos
del DT y estado de los jugadores, genera una **matriz DOFA cruzada**, predice el
resultado con **Poisson/Dixon-Coles**, detecta **value bets** frente a las cuotas
del mercado y lleva un **P&L** con ROI/Yield.

> ⚠️ **Uso responsable.** Herramienta educativa/analítica. El scraping debe
> respetar los Términos de Servicio y `robots.txt` de cada sitio; para producción
> usa feeds de datos/cuotas **con licencia**. Las apuestas implican riesgo
> financiero: ningún modelo garantiza ganancias.

---

## 1. Arquitectura

```
                 ┌────────────────────────────────────────────────┐
                 │                    main.py                       │
                 │                (orquestador)                     │
                 └───────┬───────────────┬───────────────┬─────────┘
                         │               │               │
        ┌────────────────▼───┐   ┌───────▼────────┐   ┌──▼──────────────┐
        │  1. INGESTIÓN ETL  │   │  2. PIPELINE   │   │  5. PREDICCIÓN  │
        │  src/ingestion     │   │  src/pipeline  │   │  src/prediction │
        │  · scraper.py      │──▶│  6 puntos →    │──▶│  · poisson      │
        │  · odds_scraper.py │   │  TeamReport    │   │  · ml_model     │
        │  · synthetic.py    │   └───────┬────────┘   │  · value_bets   │
        │  · schemas.py      │           │            └──┬──────────────┘
        └────────────────────┘   ┌───────▼────────┐      │
                                 │  3. DOFA CRUZ. │      │
                                 │  src/dofa      │──────┘  (ajusta λ)
                                 │  cross_swot.py │
                                 └────────────────┘      ┌─────────────────┐
                                                         │ 6. P&L / KPIs   │
                                                         │ src/performance │
                                                         │ pnl.py (SQLite) │
                                                         └─────────────────┘
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
MatchData                    # 1 partido de 1 equipo
├── team_stats: TeamMatchStats   # posesión, tiros, xG, ppda, línea, directness…
└── players:   [PlayerMatchStats]  # rating, xG/xA, duelos, pases, avg_x/avg_y…

MarketOdds                   # cuotas 1X2 / Over-Under / BTTS
```

Un equipo = `List[MatchData]` (5 partidos). Se tabula a dos DataFrames:
`team_df` (1 fila/partido) y `player_df` (1 fila/jugador-partido) en
`pipeline/features.py`.

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
Probabilidad de que un remate termine en gol dado su contexto. Aquí se agrega por
equipo/jugador y se usa como **proxy de calidad de generación**, más estable que
los goles reales (que tienen mucha varianza en 5 partidos).

### Poisson bivariado + corrección Dixon-Coles (`prediction/poisson_model.py`)
Fuerzas relativas a la media de liga:

```
attack_i  = xG_for_i / media_liga
defense_i = xG_against_i / media_liga
```

Tasas esperadas del partido:

```
λ_home = media_liga · attack_home · defense_away · ventaja_local · adj_home
λ_away = media_liga · attack_away · defense_home                · adj_away
```

Probabilidad de marcador (x, y):

```
P(x, y) = Pois(x; λ_home) · Pois(y; λ_away) · τ(x, y)
```

`τ` (Dixon-Coles, ρ<0) corrige la subestimación de empates de pocos goles
(0-0, 1-0, 0-1, 1-1). Sumando la malla de resultados se obtienen **1X2**,
**Over/Under 2.5** y **BTTS**.

### Valor esperado y staking (`prediction/value_bets.py`)
```
q     = 1 / cuota                     # prob. implícita (con margen de la casa)
edge  = p_modelo − q
EV    = p_modelo · cuota − 1          # >0 ⇒ value bet
Kelly = (b·p − (1−p)) / b · fracción  # b = cuota − 1  (Kelly fraccionado)
```

### KPIs de P&L (`performance/pnl.py`)
`ROI = Yield = profit / stake_total`, hit rate, CLV (valor vs. cierre) y curva de
capital (bankroll).

---

## 6. Cómo ejecutar

```bash
pip install -r requirements.txt
python main.py          # corre en OFFLINE_MODE con datos sintéticos
```

Para datos reales: en `config.py` pon `OFFLINE_MODE = False` (y `USE_SELENIUM`
según el sitio), y ajusta URLs/selectores en `scraper.py` / `odds_scraper.py` a
tu fuente **con licencia**.

### Estructura de módulos
```
config.py                    parámetros centrales (ventana, ligas, Kelly, rutas)
main.py                      orquestador end-to-end
src/
├── ingestion/               ETL: scraping, cuotas, esquemas, sintético
├── pipeline/                6 puntos → TeamReport
├── dofa/                    matriz DOFA cruzada
├── prediction/             poisson + ml + value bets
└── performance/            P&L en SQLite + KPIs
```

---

## 7. Roadmap sugerido
- Calibrar `HOME_ADVANTAGE`, `LEAGUE_AVG_GOALS` y `ρ` con histórico real de la liga.
- Entrenar `OutcomeClassifier` (RandomForest) y hacer *ensemble* con Poisson.
- Dashboard visual (Streamlit) sobre `BetLedger.equity_curve()`.
- Backtesting: replay de temporadas para medir ROI del sistema antes de operar.
#   X G o a l s - A I  
 