"""
Helper functions for signal and inefficiency calculations.
"""

import math
import pandas as pd
import sqlite3
from datetime import datetime, timezone

def now_ts():
    return int(datetime.now(tz=timezone.utc).timestamp())

def compute_momentum_from_history(conn: sqlite3.Connection, market_id: str, lookback_seconds=3600*24):
    """
    Simple momentum signal: compares last price to median of previous 24h prices.
    Returns value in [-1,1].
    """
    c = conn.cursor()
    cutoff = now_ts() - lookback_seconds
    rows = c.execute(
        "SELECT price FROM historical_prices WHERE market_id=? AND ts>=? ORDER BY ts ASC",
        (market_id, cutoff)
    ).fetchall()
    prices = [r[0] for r in rows if r[0] is not None]
    if len(prices) < 2:
        return 0.0
    last = prices[-1]
    median = float(pd.Series(prices).median())
    if median == 0:
        return 0.0
    dev = (last - median) / abs(median)
    return max(-1.0, min(1.0, dev))

def normalize_sentiment(sentiment):
    """Map [-1,1] sentiment → [0,1]."""
    return (sentiment + 1.0) / 2.0

def compute_inefficiency_score(market_prob, model_prob=None, sentiment=None, liquidity=0.0, momentum=0.0):
    """
    Compute a heuristic inefficiency score in [0, +inf).
    """
    w_model, w_sent, w_vol, w_mom = 0.6, 0.2, 0.1, 0.1

    model_gap = abs(model_prob - market_prob) if model_prob is not None else 0.0
    sent_signal = 0.0
    if sentiment is not None:
        s_norm = normalize_sentiment(sentiment)
        sent_signal = abs(s_norm - 0.5)

    try:
        liq_score = 1.0 / (1.0 + math.log1p(max(0.0, liquidity)))
    except Exception:
        liq_score = 0.0

    mom_score = abs(momentum)
    score = (
        w_model * model_gap +
        w_sent * sent_signal +
        w_vol * liq_score +
        w_mom * mom_score
    )
    details = {
        "model_gap": model_gap,
        "sent_signal": sent_signal,
        "liq_score": liq_score,
        "mom_score": mom_score,
        "weights": {"w_model": w_model, "w_sent": w_sent, "w_vol": w_vol, "w_mom": w_mom}
    }
    return score, details
