#!/usr/bin/env python3
"""
fetch_data.py – optimized for Google Cloud NLP & embeddings
- Multi-sport, multi-page
- Robust HTTP retry/backoff
- CSV + optional SQLite output
- Ingestion timestamps and last_updated
- Google Cloud NLP for sentiment
- Simple embeddings similarity for model_prob
"""

import os
import time
import requests
import pandas as pd
import sqlite3
from datetime import datetime, timezone
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Google Cloud
from google.cloud import language_v1
from google.cloud import aiplatform
from google.cloud import aiplatform

# ----------------------
# LOAD CONFIG
# ----------------------
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", os.path.join("..", "data", "processed_events.csv"))
SQLITE_FILE = os.getenv("SQLITE_FILE", None)  # e.g. "../data/events.db"

SPORTS = [
    "americanfootball_nfl",
    "basketball_nba",
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "mma_mixed_martial_arts",
    "politics_us_presidential_election_winner"
]

REGION = "us"
MARKETS = "h2h"
PAGE_LIMIT = int(os.getenv("PAGE_LIMIT", 5))
WAIT_BETWEEN = float(os.getenv("WAIT_BETWEEN", 1.0))
MAX_NEWS = int(os.getenv("MAX_NEWS", 5))

REQUEST_TIMEOUT = 10
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5

# Initialize Google NLP client
nlp_client = language_v1.LanguageServiceClient()

# Initialize Vertex AI (for embeddings)
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")  # must match your GCP project
REGION_AI = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
aiplatform.init(project=PROJECT_ID, location=REGION_AI)

# ----------------------
# HELPERS
# ----------------------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def safe_request_get(url: str, params: dict, timeout: int = REQUEST_TIMEOUT) -> Optional[requests.Response]:
    backoff = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 429:
                wait = backoff + 1
                print(f"Rate limited (429). Sleeping {wait}s (attempt {attempt}).")
                time.sleep(wait)
            elif 500 <= resp.status_code < 600 and attempt < MAX_RETRIES:
                print(f"Server error {resp.status_code}. Backing off {backoff}s.")
                time.sleep(backoff)
            else:
                return resp
        except requests.exceptions.RequestException as e:
            print(f"Request exception (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= BACKOFF_FACTOR
            else:
                return None
    return None

def get_news_headlines(query: str, max_results: int = MAX_NEWS) -> List[str]:
    if not NEWS_API_KEY or not query:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": max_results,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }
    try:
        resp = safe_request_get(url, params)
        if resp is None:
            return []
        data = resp.json()
        articles = data.get("articles", [])
        return [a.get("title", "") for a in articles if a.get("title")]
    except Exception as e:
        print("get_news_headlines error:", e)
        return []

def sentiment_google(texts: List[str]) -> float:
    """Compute sentiment score in [-1,1] using Google NLP."""
    if not texts:
        return 0.0
    try:
        combined = "\n".join(texts)
        document = language_v1.Document(content=combined, type_=language_v1.Document.Type.PLAIN_TEXT)
        sentiment = nlp_client.analyze_sentiment(request={"document": document}).document_sentiment
        return max(-1.0, min(1.0, sentiment.score))
    except Exception as e:
        print("sentiment_google error:", e)
        return 0.0

def embeddings_google(text: str) -> List[float]:
    """Return embedding vector for text using Vertex AI embeddings."""
    try:
        model = aiplatform.models.TextEmbeddingModel("textembedding-gecko@001")
        response = model.get_embeddings([text])
        return response.embeddings[0].values
    except Exception as e:
        print("embeddings_google error:", e)
        return []

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    import math
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a*b for a,b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a*a for a in vec1))
    norm2 = math.sqrt(sum(b*b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return max(0.0, min(1.0, (dot / (norm1*norm2)+1)/2))  # normalize 0-1

def llm_model_prob_google(teams_label: str, headlines: List[str]) -> float:
    """Approximate model probability using embedding similarity."""
    if not headlines:
        return 0.5
    try:
        # Combine headlines into one text
        text = " ".join(headlines)
        event_emb = embeddings_google(teams_label)
        text_emb = embeddings_google(text)
        return cosine_similarity(event_emb, text_emb)
    except Exception:
        return 0.5

def safe_float(x, fallback=None):
    try:
        return float(x)
    except Exception:
        return fallback

# ----------------------
# FETCH EVENTS
# ----------------------
def fetch_sport_pages(sport: str, page_limit: int = PAGE_LIMIT) -> List[Dict[str, Any]]:
    events = []
    for page in range(1, page_limit + 1):
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        params = {"apiKey": ODDS_API_KEY, "regions": REGION, "markets": MARKETS, "page": page}
        print(f"Fetching {sport} page {page} ...")
        resp = safe_request_get(url, params)
        if resp is None:
            break
        try:
            data = resp.json()
        except:
            break
        if isinstance(data, dict) and "message" in data:
            break
        if not data:
            break
        events.extend(data)
        time.sleep(WAIT_BETWEEN)
    return events

def choose_h2h_outcomes(event: dict) -> Optional[Dict[str, Any]]:
    bookmakers = event.get("bookmakers", []) or []
    for bookmaker in bookmakers:
        for market in bookmaker.get("markets", []):
            mkey = market.get("key") or market.get("market_key") or market.get("market")
            if mkey and ("h2h" in str(mkey).lower() or "moneyline" in str(mkey).lower()):
                outcomes = market.get("outcomes", []) or []
                if len(outcomes) >= 2:
                    return {
                        "team1_name": outcomes[0].get("name"),
                        "team2_name": outcomes[1].get("name"),
                        "team1_price": safe_float(outcomes[0].get("price")),
                        "team2_price": safe_float(outcomes[1].get("price")),
                        "bookmaker": bookmaker.get("title") or bookmaker.get("key")
                    }
    return None

# ----------------------
# MAIN SCRIPT
# ----------------------
raw_events_all = []
for sport in SPORTS:
    sport_events = fetch_sport_pages(sport)
    raw_events_all.extend([(sport, e) for e in sport_events])

# Parse & normalize
parsed = []
for sport, event in raw_events_all:
    event_id = event.get("id") or event.get("event_id") or None
    commence_time = event.get("commence_time") or event.get("start_time") or None
    h2h = choose_h2h_outcomes(event)
    if not h2h:
        continue
    t1_odds = h2h.get("team1_price")
    t2_odds = h2h.get("team2_price")
    if t1_odds in (None, 0) or t2_odds in (None, 0):
        continue
    teams_label = f"{h2h.get('team1_name')} vs {h2h.get('team2_name')}"
    parsed.append({
        "sport": sport,
        "event_id": event_id,
        "commence_time": commence_time,
        "team1": h2h.get("team1_name"),
        "team2": h2h.get("team2_name"),
        "team1_odds": t1_odds,
        "team2_odds": t2_odds,
        "bookmaker": h2h.get("bookmaker"),
        "teams_label": teams_label
    })

# Deduplicate
seen = set()
deduped = []
for p in parsed:
    key = (p.get("sport"), p.get("event_id"))
    if key in seen:
        continue
    seen.add(key)
    deduped.append(p)

# Build dataframe
rows = []
for p in deduped:
    t1_odds = p["team1_odds"]
    t2_odds = p["team2_odds"]
    team1_prob = 1.0 / t1_odds
    team2_prob = 1.0 / t2_odds
    total = team1_prob + team2_prob
    team1_prob_norm = team1_prob / total
    team2_prob_norm = team2_prob / total

    # News & signals
    headlines = get_news_headlines(p["teams_label"], max_results=MAX_NEWS)
    sentiment = sentiment_google(headlines)
    sentiment_norm = (sentiment + 1) / 2.0
    model_prob = llm_model_prob_google(p["teams_label"], headlines)

    ingestion_ts = now_iso()
    last_updated = ingestion_ts

    discrepancy = abs(model_prob - team1_prob_norm)
    inefficiency_score = discrepancy + sentiment_norm

    rows.append({
        "sport": p.get("sport"),
        "event_id": p.get("event_id"),
        "commence_time": p.get("commence_time"),
        "team1": p.get("team1"),
        "team2": p.get("team2"),
        "teams_label": p.get("teams_label"),
        "bookmaker": p.get("bookmaker"),
        "team1_odds": t1_odds,
        "team2_odds": t2_odds,
        "team1_prob": team1_prob,
        "team2_prob": team2_prob,
        "team1_prob_norm": team1_prob_norm,
        "team2_prob_norm": team2_prob_norm,
        "market_prob": team1_prob_norm,
        "model_prob": model_prob,
        "sentiment_score": sentiment,
        "sentiment_score_norm": sentiment_norm,
        "discrepancy": discrepancy,
        "inefficiency_score": inefficiency_score,
        "news_headlines": " ||| ".join(headlines),
        "ingestion_ts": ingestion_ts,
        "last_updated": last_updated
    })

df = pd.DataFrame(rows)
expected_cols = [
    "sport","event_id","commence_time","team1","team2","teams_label","bookmaker",
    "team1_odds","team2_odds","team1_prob","team2_prob","team1_prob_norm","team2_prob_norm",
    "market_prob","model_prob","sentiment_score","sentiment_score_norm","discrepancy",
    "inefficiency_score","news_headlines","ingestion_ts","last_updated"
]
for c in expected_cols:
    if c not in df.columns:
        df[c] = pd.NA
df = df[expected_cols]

# Save CSV
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved processed events to {OUTPUT_FILE}")

# Optional SQLite
if SQLITE_FILE:
    os.makedirs(os.path.dirname(SQLITE_FILE), exist_ok=True)
    conn = sqlite3.connect(SQLITE_FILE)
    df.to_sql("events", conn, if_exists="replace", index=False)
    conn.close()
    print(f"Saved processed events to SQLite DB {SQLITE_FILE}")
