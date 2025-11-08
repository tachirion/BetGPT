#!/usr/bin/env python3
"""
main.py

Single-file ingestion for Polymarket and Manifold:
- fetch live market info (title/question, outcomes, current prices / implied probabilities)
- fetch historical price time series (when available)
- pull volume / liquidity indicators
- normalize and store to SQLite and CSV

Run:
  python main.py --once       # run one fetch cycle
  python main.py --daemon     # run continuously using APScheduler
"""

import os
import time
import json
import sqlite3
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import quote_plus
import copy


import requests
import pandas as pd
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


# --- Load config ---
load_dotenv()
POLY_BASE = os.getenv("POLYMARKET_BASE", "https://clob.polymarket.com")
MANIFOLD_BASE = os.getenv("MANIFOLD_BASE", "https://api.manifold.markets")
SQLITE_PATH = os.getenv("SQLITE_PATH", "betgpt_markets.db")
CSV_DUMP_DIR = Path(os.getenv("CSV_DUMP_DIR", "./dumps"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", "60"))

MANIFOLD_HEADERS = {}
manifold_api_key = os.getenv("MANIFOLD_API_KEY")
if manifold_api_key:
    MANIFOLD_HEADERS["Authorization"] = f"Bearer {manifold_api_key}"

CSV_DUMP_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("betgpt_ingest")


# --- Storage helpers (SQLite) ---
def init_db(conn: sqlite3.Connection):
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS markets (
        id TEXT PRIMARY KEY,
        source TEXT,
        title TEXT,
        question TEXT,
        created_at_ts INTEGER,
        closes_at_ts INTEGER,
        raw JSON,
        last_seen_ts INTEGER
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS market_snapshots (
        snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_id TEXT,
        source TEXT,
        snapshot_ts INTEGER,
        outcome TEXT,
        price REAL,
        implied_prob REAL,
        volume REAL,
        liquidity REAL,
        raw JSON
    );
    """)
    c.execute("""
    CREATE TABLE IF NOT EXISTS historical_prices (
        hist_id INTEGER PRIMARY KEY AUTOINCREMENT,
        market_id TEXT,
        source TEXT,
        ts INTEGER,
        outcome TEXT,
        price REAL,
        raw JSON
    );
    """)
    conn.commit()


# --- Utility functions ---
def to_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None

def to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def now_ts():
    return int(datetime.now(tz=timezone.utc).timestamp())

def to_int_ts(val):
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None

def safe_get(d, *keys, default=None):
    x = d
    try:
        for k in keys:
            x = x[k]
        return x
    except (KeyError, TypeError, IndexError):
        return default


# --- HTTP client with retries/backoff ---
session = requests.Session()
session.headers.update({"User-Agent": "BetGPT-Ingest/1.0 (+https://example.org)"})

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
       retry=retry_if_exception_type(requests.exceptions.RequestException))
def http_get(url: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None, timeout=10):
    h = copy.deepcopy(session.headers)
    if headers:
        h.update(headers)
    r = session.get(url, params=params, headers=h, timeout=timeout)
    r.raise_for_status()
    if r.text.strip() == "":
        return None
    try:
        return r.json()
    except Exception as e:
        logger.warning("Non-JSON response from %s: %s", url, e)
        return None


# --- Polymarket client ---
def fetch_polymarket_markets(limit: int = 100) -> List[Dict[str, Any]]:
    url = f"{POLY_BASE.rstrip('/')}/markets"
    logger.info("Fetching Polymarket markets...")
    try:
        data = http_get(url, params={"limit": limit})
        if isinstance(data, list):
            logger.debug("Polymarket returned list of length %d", len(data))
            return data[:limit]
        elif isinstance(data, dict):
            # new API returns wrapped 'data' key
            if "data" in data and isinstance(data["data"], list):
                logger.debug("Polymarket wrapped response keys: %s", list(data.keys()))
                return data["data"][:limit]
            elif "markets" in data and isinstance(data["markets"], list):
                return data["markets"][:limit]
            else:
                logger.warning("Unexpected keys in Polymarket dict: %s", list(data.keys()))
                return []
        else:
            logger.warning("Unexpected Polymarket response type: %s", type(data))
            return []
    except Exception as e:
        logger.exception("Polymarket fetch failed: %s", e)
        return []


def fetch_polymarket_market_candles(market_id: str, resolution: str = "1h") -> List[Dict[str, Any]]:
    """
    Fetch historical OHLC candles for a given Polymarket market.
    """
    url = f"{POLY_BASE.rstrip('/')}/markets/{quote_plus(market_id)}/candles"
    params = {"resolution": resolution}
    try:
        data = http_get(url, params=params)
        if not data:
            logger.debug("No candle data returned for market %s", market_id)
            return []
        return data
    except Exception as e:
        logger.debug("Polymarket candles not available for %s: %s", market_id, e)
        return []


# --- Manifold client ---
def fetch_manifold_markets(limit=100) -> List[Dict[str,Any]]:
    """
    Manifold markets endpoint v0: /v0/markets
    """
    url = MANIFOLD_BASE.rstrip("/") + "/v0/markets"
    params = {"limit": limit}
    data = http_get(url, params=params, headers=MANIFOLD_HEADERS)
    return data or []


def fetch_manifold_bets(market_id: str) -> List[Dict[str,Any]]:
    """
    Manifold bet history provides changes; useful to derive time series.
    """
    url = MANIFOLD_BASE.rstrip("/") + "/v0/bets"
    params = {"marketId": market_id, "limit": 1000}
    data = http_get(url, params=params, headers=MANIFOLD_HEADERS)
    return data or []


# --- Normalization layer ---
def normalize_polymarket_market(raw: Dict[str,Any]) -> Dict[str,Any]:
    """
    Convert to unified market schema
    """
    mid = safe_get(raw, "id") or safe_get(raw, "slug") or safe_get(raw, "marketId")
    title = safe_get(raw, "question") or safe_get(raw, "title")
    created = safe_get(raw, "createdAt") or safe_get(raw, "created")
    close_ts = safe_get(raw, "closeTime") or safe_get(raw, "closeTimestamp")
    # outcomes prices -> list of dicts
    outcomes = safe_get(raw, "outcomes") or []
    norm_outcomes = []
    for out in outcomes:
        name = safe_get(out, "name") or safe_get(out, "label")
        price = safe_get(out, "price") or safe_get(out, "probability") or None
        norm_outcomes.append({"name": name, "price": float(price) if price is not None else None})
    volume = safe_get(raw, "volume24h") or safe_get(raw, "volume") or 0
    return {
        "id": f"polymarket:{mid}",
        "source": "polymarket",
        "title": title,
        "question": title,
        "created_at_ts": to_int_ts(created),
        "closes_at_ts": to_int_ts(close_ts),
        "outcomes": norm_outcomes,
        "volume": float(volume or 0),
        "raw": raw
    }


def normalize_manifold_market(raw: Dict[str,Any]) -> Dict[str,Any]:
    mid = safe_get(raw, "id")
    title = safe_get(raw, "question") or safe_get(raw, "title")
    created = safe_get(raw, "createdTime")
    closes = safe_get(raw, "closeTime")
    # Manifold returns 'probability' or 'resolved' etc.
    outcomes = []
    # For binary markets, Manifold often uses 'probability' as a float 0-1
    if "probability" in raw:
        outcomes = [{"name": "YES", "price": float(raw["probability"])},
                    {"name": "NO", "price": max(0.0, 1.0 - float(raw["probability"]))}]
    # fallback: check 'outcomeType' or 'answers'
    for ans in safe_get(raw, "answers", []) or []:
        outcomes.append({"name": safe_get(ans, "text"), "price": safe_get(ans, "probability")})
    volume = safe_get(raw, "volume24Hours") or safe_get(raw, "volume") or safe_get(raw, "volume24h") or 0
    return {
        "id": f"manifold:{mid}",
        "source": "manifold",
        "title": title,
        "question": title,
        "created_at_ts": int(created) if created else None,
        "closes_at_ts": int(closes) if closes else None,
        "outcomes": outcomes,
        "volume": float(volume or 0),
        "raw": raw
    }


# --- Storage ingestion functions ---
def upsert_market(conn: sqlite3.Connection, market: Dict[str,Any]):
    c = conn.cursor()
    now = now_ts()
    c.execute("""
        INSERT INTO markets(id, source, title, question, created_at_ts, closes_at_ts, raw, last_seen_ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          title=excluded.title,
          question=excluded.question,
          created_at_ts=COALESCE(markets.created_at_ts, excluded.created_at_ts),
          closes_at_ts=excluded.closes_at_ts,
          raw=excluded.raw,
          last_seen_ts=excluded.last_seen_ts;
    """, (
        market["id"],
        market["source"],
        market["title"],
        market.get("question"),
        market.get("created_at_ts"),
        market.get("closes_at_ts"),
        json.dumps(market.get("raw", {})),
        now
    ))
    conn.commit()


def write_snapshot(conn: sqlite3.Connection, market: Dict[str,Any]):
    c = conn.cursor()
    ts = now_ts()
    for outcome in market.get("outcomes", []):
        price = outcome.get("price")
        implied = price  # in binary markets price == implied prob
        # liquidity - heuristic: use volume as proxy; APIs may expose better fields
        liquidity = market.get("volume", 0)
        c.execute("""
            INSERT INTO market_snapshots(market_id, source, snapshot_ts, outcome, price, implied_prob, volume, liquidity, raw)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market["id"], market["source"], ts, outcome.get("name"), price, implied, market.get("volume", 0), liquidity,
            json.dumps(market.get("raw", {}))
        ))
    conn.commit()


def write_historical_rows(conn: sqlite3.Connection, market_id: str, source: str, rows: List[Dict[str,Any]]):
    """
    rows: list of dicts with ts, outcome, price, raw if available
    """
    if not rows:
        return
    c = conn.cursor()
    for r in rows:
        c.execute("""
          INSERT INTO historical_prices(market_id, source, ts, outcome, price, raw)
          VALUES (?, ?, ?, ?, ?, ?)
        """, (market_id, source, r.get("ts"), r.get("outcome"), r.get("price"), json.dumps(r.get("raw", {}))))
    conn.commit()


# --- CSV dump helpers ---
def dump_snapshot_csv(conn: sqlite3.Connection):
    df = pd.read_sql_query("SELECT * FROM market_snapshots ORDER BY snapshot_ts DESC LIMIT 10000", conn)
    fname = CSV_DUMP_DIR / f"market_snapshots_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    df.to_csv(fname, index=False)
    logger.info("Wrote snapshots CSV to %s", fname)


def dump_markets_csv(conn: sqlite3.Connection):
    df = pd.read_sql_query("SELECT * FROM markets", conn)
    fname = CSV_DUMP_DIR / f"markets_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    df.to_csv(fname, index=False)
    logger.info("Wrote markets CSV to %s", fname)


# --- Orchestrator: one fetch cycle (safe, non-blocking) ---
def run_fetch_cycle(conn: sqlite3.Connection):
    logger.info("Starting fetch cycle")
    try:
        # --- Polymarket ---
        try:
            pm = fetch_polymarket_markets(limit=200)
            logger.info("Fetched %d Polymarket markets", len(pm))
        except Exception as e:
            logger.exception("Polymarket fetch failed: %s", e)
            pm = []

        for i, raw in enumerate(pm):
            try:
                norm = normalize_polymarket_market(raw)
                upsert_market(conn, norm)
                write_snapshot(conn, norm)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning("Failed to process Polymarket market %s: %s", raw.get("id"), e)
                continue  # skip this market

            # --- Polymarket historical candles (best-effort, safe) ---
            market_id = safe_get(raw, "id")
            if not market_id:
                continue
            try:
                candles = fetch_polymarket_market_candles(market_id, resolution="1h")
                if not candles:
                    continue
                hist_rows = []
                if isinstance(candles, list) and candles and isinstance(candles[0], (list, tuple)):
                    for row in candles:
                        ts = int(row[0]) // 1000 if row[0] > 1e10 else int(row[0])
                        price = float(row[4]) if len(row) > 4 else None
                        hist_rows.append({"ts": ts, "outcome": None, "price": price, "raw": row})
                elif isinstance(candles, list):
                    for obj in candles:
                        ts = safe_get(obj, "t") or safe_get(obj, "time") or safe_get(obj, "timestamp") or None
                        price = safe_get(obj, "c") or safe_get(obj, "close") or None
                        if ts and price is not None:
                            hist_rows.append({"ts": int(ts), "outcome": None, "price": float(price), "raw": obj})
                write_historical_rows(conn, norm["id"], norm["source"], hist_rows)
            except Exception as e:
                logger.warning("Failed to fetch/process candles for Polymarket %s: %s", market_id, e)
                continue  # continue to next market

        # --- Manifold ---
        try:
            mm = fetch_manifold_markets(limit=200)
            logger.info("Fetched %d Manifold markets", len(mm))
        except Exception as e:
            logger.exception("Manifold fetch failed: %s", e)
            mm = []

        for i, raw in enumerate(mm):
            try:
                norm = normalize_manifold_market(raw)
                upsert_market(conn, norm)
                write_snapshot(conn, norm)
            except Exception as e:
                logger.warning("Failed to process Manifold market %s: %s", raw.get("id"), e)
                continue  # skip this market

            # --- Manifold bet history ---
            market_id = safe_get(raw, "id")
            if not market_id:
                continue
            try:
                bets = fetch_manifold_bets(market_id)
                hist_rows = []
                for b in bets:
                    ts = safe_get(b, "createdTime") or safe_get(b, "createdAt") or safe_get(b, "ts")
                    price = safe_get(b, "probAfter") or safe_get(b, "probBefore")
                    if ts and price is not None:
                        hist_rows.append({"ts": int(ts), "outcome": None, "price": float(price), "raw": b})
                write_historical_rows(conn, norm["id"], norm["source"], hist_rows)
            except Exception as e:
                logger.warning("Manifold bets fetch failed for %s: %s", market_id, e)

        # --- Post-cycle CSV dumps ---
        try:
            dump_snapshot_csv(conn)
            dump_markets_csv(conn)
        except Exception as e:
            logger.warning("Failed to dump CSVs: %s", e)

        logger.info("Fetch cycle finished")
    except Exception as e:
        logger.exception("Unexpected error in fetch cycle: %s", e)


# --- CLI / main ---
def main_once():
    conn = sqlite3.connect(SQLITE_PATH)
    init_db(conn)
    run_fetch_cycle(conn)
    conn.close()


def main_daemon():
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    init_db(conn)
    scheduler = BackgroundScheduler()
    # schedule run_fetch_cycle every POLL_INTERVAL seconds
    scheduler.add_job(lambda: run_fetch_cycle(conn), 'interval', seconds=POLL_INTERVAL, max_instances=1, coalesce=True)
    scheduler.start()
    logger.info("Scheduler started with interval %ds. Press Ctrl+C to stop.", POLL_INTERVAL)
    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutting down scheduler...")
        scheduler.shutdown()
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one ingestion cycle and exit")
    parser.add_argument("--daemon", action="store_true", help="Run continuously (scheduler)")
    args = parser.parse_args()
    if args.once:
        main_once()
    elif args.daemon:
        main_daemon()
    else:
        # default: run once
        main_once()