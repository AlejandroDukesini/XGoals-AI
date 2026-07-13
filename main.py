"""
main.py  —  ORQUESTADOR RES-IA-PART-FUT
=======================================
Flujo end-to-end:

    1. ETL          -> últimos 5 partidos de Local y Rival (+ cuotas BetPlay)
    2. Pipeline     -> 6 puntos de análisis por equipo (TeamReport)
    3. DOFA cruzada -> hallazgos + ajustes de lambda
    4. Predicción   -> Poisson/Dixon-Coles (1X2, O/U, BTTS, marcador)
    5. Valor        -> value bets vs. cuotas de mercado
    6. P&L          -> registro de las apuestas sugeridas

Ejecuta:  python main.py
(Con config.OFFLINE_MODE = True corre sin conexión, con datos sintéticos.)
"""
from __future__ import annotations

import json
import sys

# La consola de Windows suele usar cp1252; forzamos UTF-8 para los símbolos.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from tabulate import tabulate

import config
from src.ingestion import fetch_last_matches, fetch_odds
from src.pipeline import build_team_report
from src.dofa import cross_analyze
from src.prediction import predict, evaluate
from src.performance import BetLedger

HOME_TEAM = "Local"
AWAY_TEAM = "Rival"


def _hr(title: str):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)


def run():
    match_id = f"{HOME_TEAM}-vs-{AWAY_TEAM}"

    # ---------- 1. ETL ----------
    _hr("1. INGESTIÓN (ETL)")
    home_matches = fetch_last_matches(HOME_TEAM)
    away_matches = fetch_last_matches(AWAY_TEAM)
    odds = fetch_odds(match_id, HOME_TEAM, AWAY_TEAM)
    print(f"{HOME_TEAM}: {len(home_matches)} partidos | "
          f"{AWAY_TEAM}: {len(away_matches)} partidos")
    print(f"Cuotas {odds.bookmaker}: 1={odds.home}  X={odds.draw}  2={odds.away} | "
          f"O2.5={odds.over_2_5}  BTTS Sí={odds.btts_yes}")

    # ---------- 2. Pipeline (6 puntos) ----------
    _hr("2. PIPELINE DE CIENCIA DE DATOS (6 puntos por equipo)")
    home_rep = build_team_report(HOME_TEAM, home_matches)
    away_rep = build_team_report(AWAY_TEAM, away_matches)
    for rep in (home_rep, away_rep):
        g, t = rep.p1_general, rep.p6_tactical
        print(f"\n>> {rep.team}: {g['record']} | PPG {g['ppg']} | "
              f"xGf {g['xg_for_avg']} xGa {g['xg_against_avg']}")
        print(f"   Estilo: {t['play_style']} | Presión: {t['press_scheme']} "
              f"({t['formation_mode']})")
        print(f"   Top talento: "
              f"{', '.join(p['name'] for p in rep.p4_talent['top_players'][:3])}")
        if rep.p3_behavior["high_fatigue_players"]:
            print(f"   ⚠ Fatiga: {', '.join(rep.p3_behavior['high_fatigue_players'])}")

    # ---------- 3. DOFA cruzada ----------
    _hr("3. MATRIZ DOFA CRUZADA")
    swot = cross_analyze(home_rep, away_rep)
    for dim in ("Fortalezas", "Debilidades", "Oportunidades", "Amenazas"):
        items = swot.to_dict()[dim]
        print(f"\n[{dim}]")
        if not items:
            print("   (sin hallazgos relevantes)")
        for it in items:
            print(f"   • {it['dimension']}: {it['mensaje']}")
    print(f"\nAjuste λ por cruce: {swot.lambda_adjust}")

    # ---------- 4. Predicción ----------
    _hr("4. PREDICCIÓN (Poisson + Dixon-Coles)")
    pred = predict(home_rep, away_rep, adj=swot.lambda_adjust)
    print(pred.report())
    print("\nMarcadores más probables:")
    print(tabulate(pred.top_scores, headers="keys", tablefmt="simple"))

    # ---------- 5. Value bets ----------
    _hr("5. MOTOR DE VALOR ESPERADO (Value Bets)")
    bets = evaluate(pred.probabilities(), odds)
    rows = [[b.market, f"{b.model_prob:.0%}", b.book_odds, f"{b.implied_prob:.0%}",
             f"{b.edge:+.1%}", f"{b.ev:+.2f}", f"{b.kelly_stake:.1%}",
             "✅" if b.is_value else ""] for b in bets]
    print(tabulate(rows, headers=["Mercado", "P(IA)", "Cuota", "P(casa)",
                                  "Edge", "EV", "Kelly", "Valor"], tablefmt="github"))

    value = [b for b in bets if b.is_value]

    # ---------- 6. P&L ----------
    _hr("6. REGISTRO P&L")
    ledger = BetLedger()
    if value:
        for b in value:
            stake = round(config.BANKROLL_INICIAL * b.kelly_stake, 2)
            bet_id = ledger.log_bet(match_id, b.market, b.book_odds, stake,
                                    model_prob=b.model_prob, edge=b.edge)
            print(f"Registrada #{bet_id}: {b.market} @ {b.book_odds} "
                  f"stake ${stake:,.0f} (EV {b.ev:+.2f})")
    else:
        print("No hay apuestas con valor suficiente (edge <", config.MIN_EDGE, ")")
    print("\nKPIs actuales:", json.dumps(ledger.kpis(), indent=2, ensure_ascii=False))
    ledger.close()

    # ---------- Export JSON completo ----------
    out = config.PROCESSED_DIR / f"{match_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "home_report": home_rep.summary(),
            "away_report": away_rep.summary(),
            "dofa": swot.to_dict(),
            "prediction": {**{k: round(v, 3) for k, v in pred.probabilities().items()},
                           "score": f"{pred.most_likely_score[0]}-{pred.most_likely_score[1]}"},
            "value_bets": [b.as_dict() for b in value],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Análisis exportado a: {out}")


if __name__ == "__main__":
    run()
