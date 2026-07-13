"""
pnl.py  —  DASHBOARD DE RENDIMIENTO Y P&L
=========================================
Registra apuestas sugeridas vs. resultados reales en SQLite y calcula las
métricas financieras estándar del trading deportivo.

MÉTRICAS
--------
- Profit:   Σ (retorno - stake). Ganada: stake*(cuota-1). Perdida: -stake.
- ROI:      profit / stake_total           (retorno sobre lo invertido)
- Yield:    profit / stake_total * 100      (== ROI en %; KPI clásico de apuestas)
- Hit rate: apuestas ganadas / resueltas
- CLV:      (odds_tomada / odds_cierre - 1) — valor vs. cierre del mercado
- Bankroll: BANKROLL_INICIAL + profit acumulado (curva de capital)
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

import config


SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    placed_at     TEXT NOT NULL,
    match_id      TEXT NOT NULL,
    market        TEXT NOT NULL,
    model_prob    REAL,
    odds          REAL NOT NULL,
    closing_odds  REAL,
    stake         REAL NOT NULL,
    edge          REAL,
    status        TEXT DEFAULT 'pending',   -- pending | won | lost | void
    payout        REAL DEFAULT 0.0,
    settled_at    TEXT
);
"""


class BetLedger:
    """Libro de registro de apuestas con cálculo de KPIs."""

    def __init__(self, db_path=None):
        self.db_path = str(db_path or config.DB_PATH)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(SCHEMA)
        self.conn.commit()

    # ---------------- Registro ----------------
    def log_bet(self, match_id: str, market: str, odds: float, stake: float,
                model_prob: float = None, edge: float = None,
                closing_odds: float = None) -> int:
        cur = self.conn.execute(
            """INSERT INTO bets (placed_at, match_id, market, model_prob, odds,
                                 closing_odds, stake, edge)
               VALUES (?,?,?,?,?,?,?,?)""",
            (datetime.utcnow().isoformat(timespec="seconds"), match_id, market,
             model_prob, odds, closing_odds, stake, edge),
        )
        self.conn.commit()
        return cur.lastrowid

    def settle(self, bet_id: int, won: bool, void: bool = False) -> None:
        """Liquida una apuesta y calcula el payout neto."""
        row = self.conn.execute("SELECT odds, stake FROM bets WHERE id=?",
                                (bet_id,)).fetchone()
        if row is None:
            raise ValueError(f"Bet {bet_id} inexistente")
        if void:
            status, payout = "void", 0.0
        elif won:
            status, payout = "won", row["stake"] * (row["odds"] - 1)  # ganancia neta
        else:
            status, payout = "lost", -row["stake"]
        self.conn.execute(
            "UPDATE bets SET status=?, payout=?, settled_at=? WHERE id=?",
            (status, payout, datetime.utcnow().isoformat(timespec="seconds"), bet_id),
        )
        self.conn.commit()

    # ---------------- Consulta ----------------
    def dataframe(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM bets", self.conn)

    def kpis(self) -> Dict[str, float]:
        df = self.dataframe()
        settled = df[df.status.isin(["won", "lost"])]
        staked = settled.stake.sum()
        profit = settled.payout.sum()
        wins = int((settled.status == "won").sum())

        clv = None
        cdf = settled.dropna(subset=["closing_odds"])
        if not cdf.empty:
            clv = round(((cdf.odds / cdf.closing_odds) - 1).mean() * 100, 2)

        return {
            "bets_total": int(len(df)),
            "bets_settled": int(len(settled)),
            "pending": int((df.status == "pending").sum()),
            "staked": round(float(staked), 2),
            "profit": round(float(profit), 2),
            "roi_pct": round(float(profit / staked * 100), 2) if staked else 0.0,
            "yield_pct": round(float(profit / staked * 100), 2) if staked else 0.0,
            "hit_rate": round(wins / len(settled), 3) if len(settled) else 0.0,
            "avg_clv_pct": clv,
            "bankroll": round(config.BANKROLL_INICIAL + float(profit), 2),
        }

    def equity_curve(self) -> pd.DataFrame:
        """Curva de capital acumulado para el dashboard."""
        df = self.dataframe()
        settled = df[df.status.isin(["won", "lost"])].sort_values("settled_at").copy()
        settled["cum_profit"] = settled.payout.cumsum()
        settled["bankroll"] = config.BANKROLL_INICIAL + settled.cum_profit
        return settled[["settled_at", "match_id", "market", "payout",
                        "cum_profit", "bankroll"]]

    def close(self):
        self.conn.close()
