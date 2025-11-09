# #!/usr/bin/env python3
# """
# Optimized fetch_data.py
# - Multi-sport, multi-page
# - Concurrent HTTP requests with retry/backoff
# - Clean structured CSV/SQLite output
# - LLM sentiment & model probability
# - Consistent schema + timestamps
# """
#
# import os
# import time
# import json
# import requests
# import pandas as pd
# import sqlite3
# from datetime import datetime, timezone
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from dotenv import load_dotenv
# from openai import OpenAI
#
# # ---------------------- CONFIG ----------------------
# load_dotenv()
#
# ODDS_API_KEY = os.getenv("ODDS_API_KEY")
# NEWS_API_KEY = os.getenv("NEWS_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#
# SPORTS = [
#     "americanfootball_nfl",
#     "basketball_nba",
#     "soccer_epl",
#     "soccer_spain_la_liga",
#     "soccer_italy_serie_a",
#     "mma_mixed_martial_arts",
#     "politics_us_presidential_election_winner",
# ]
#
# REGION = "us"
# MARKETS = "h2h"
# PAGE_LIMIT = int(os.getenv("PAGE_LIMIT", 5))
# MAX_NEWS = int(os.getenv("MAX_NEWS", 5))
# WAIT_BETWEEN = float(os.getenv("WAIT_BETWEEN", 1.0))
# MAX_WORKERS = 6
#
# OUTPUT_FILE = os.getenv("OUTPUT_FILE", os.path.join("..", "data", "processed_events.csv"))
# SQLITE_FILE = os.getenv("SQLITE_FILE", None)  # optional
# TIMEOUT = 8
# RETRIES = 3
#
# # Initialize OpenAI client
# client = None
# if OPENAI_API_KEY:
#     try:
#         client = OpenAI(api_key=OPENAI_API_KEY)
#     except Exception as e:
#         print("⚠️ OpenAI init failed:", e)
#
# # ---------------------- HELPERS ----------------------
# def now_iso():
#     return datetime.now(timezone.utc).isoformat()
#
# def safe_float(x, fallback=None):
#     try:
#         return float(x)
#     except:
#         return fallback
#
# def safe_request(url, params):
#     """GET with retry/backoff"""
#     for attempt in range(RETRIES):
#         try:
#             r = requests.get(url, params=params, timeout=TIMEOUT)
#             if r.status_code == 200:
#                 return r.json()
#             elif r.status_code == 429:
#                 time.sleep((attempt + 1) * 2)
#             elif 500 <= r.status_code < 600:
#                 time.sleep((attempt + 1))
#         except Exception:
#             time.sleep(1)
#     return None
#
# # ---------------------- FETCH ODDS ----------------------
# def fetch_page(sport, page):
#     url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
#     params = {"apiKey": ODDS_API_KEY, "regions": REGION, "markets": MARKETS, "page": page}
#     data = safe_request(url, params)
#     if not data or isinstance(data, dict):
#         return []
#     return [(sport, e) for e in data]
#
# def fetch_all_sports():
#     """Concurrent fetch across sports and pages"""
#     tasks, results = [], []
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
#         for sport in SPORTS:
#             for page in range(1, PAGE_LIMIT + 1):
#                 tasks.append(ex.submit(fetch_page, sport, page))
#         for fut in as_completed(tasks):
#             results.extend(fut.result())
#     print(f"✅ Total fetched events: {len(results)}")
#     return results
#
# # ---------------------- PARSE ----------------------
# def parse_event(sport, event):
#     bookmakers = event.get("bookmakers", [])
#     if not bookmakers:
#         return None
#
#     # Find first bookmaker with H2H market
#     for bm in bookmakers:
#         for m in bm.get("markets", []):
#             key = (m.get("key") or "").lower()
#             if "h2h" in key:
#                 outs = m.get("outcomes", [])
#                 if len(outs) >= 2:
#                     o1, o2 = outs[0], outs[1]
#                     return {
#                         "sport": sport,
#                         "event_id": event.get("id"),
#                         "commence_time": event.get("commence_time"),
#                         "bookmaker": bm.get("title"),
#                         "team1": o1.get("name"),
#                         "team2": o2.get("name"),
#                         "team1_odds": safe_float(o1.get("price")),
#                         "team2_odds": safe_float(o2.get("price")),
#                     }
#     return None
#
# def build_dataframe(events):
#     parsed = [p for (s, e) in events if (p := parse_event(s, e))]
#     df = pd.DataFrame(parsed).drop_duplicates(subset=["sport", "event_id"])
#     if df.empty:
#         return pd.DataFrame(columns=[
#             "sport","event_id","commence_time","team1","team2","bookmaker",
#             "team1_odds","team2_odds","team1_prob","team2_prob",
#             "team1_prob_norm","team2_prob_norm","market_prob",
#             "model_prob","sentiment_score","sentiment_score_norm",
#             "discrepancy","inefficiency_score","news_headlines",
#             "ingestion_ts","last_updated"
#         ])
#
#     # Market probabilities
#     df["team1_prob"] = 1 / df["team1_odds"]
#     df["team2_prob"] = 1 / df["team2_odds"]
#     total = df["team1_prob"] + df["team2_prob"]
#     df["team1_prob_norm"] = df["team1_prob"] / total
#     df["team2_prob_norm"] = df["team2_prob"] / total
#     df["market_prob"] = df["team1_prob_norm"]
#     return df
#
# # ---------------------- NEWS + LLM ----------------------
# def get_news_headlines(query):
#     if not NEWS_API_KEY:
#         return []
#     url = "https://newsapi.org/v2/everything"
#     params = {"q": query, "pageSize": MAX_NEWS, "apiKey": NEWS_API_KEY, "sortBy": "publishedAt"}
#     data = safe_request(url, params)
#     return [a["title"] for a in data.get("articles", []) if "title" in a] if data else []
#
# def llm_eval(prompt, fallback):
#     if not client:
#         return fallback
#     try:
#         r = client.chat.completions.create(
#             model="gpt-4.1-mini",
#             messages=[{"role": "user", "content": prompt}],
#             max_tokens=10,
#         )
#         return float(r.choices[0].message.content.strip())
#     except Exception:
#         return fallback
#
# def enrich_with_sentiment(df):
#     headlines_list, sentiments, model_probs = [], [], []
#     for _, row in df.iterrows():
#         query = f"{row.team1} vs {row.team2}"
#         headlines = get_news_headlines(query)
#         headlines_list.append(" ||| ".join(headlines))
#         s_prompt = f"Rate sentiment (-1 to 1) of headlines: {headlines}"
#         m_prompt = f"Given event '{query}' and these headlines, estimate probability (0-1) that {row.team1} wins."
#         s_score = llm_eval(s_prompt, 0.0)
#         m_prob = llm_eval(m_prompt, 0.5)
#         sentiments.append(s_score)
#         model_probs.append(m_prob)
#         print(f"📰 {query}: {len(headlines)} headlines, sentiment={s_score:.2f}, model_prob={m_prob:.2f}")
#     df["news_headlines"] = headlines_list
#     df["sentiment_score"] = sentiments
#     df["sentiment_score_norm"] = (df["sentiment_score"] + 1) / 2
#     df["model_prob"] = model_probs
#     return df
#
# # ---------------------- COMPUTE FINAL ----------------------
# def compute_signals(df):
#     df["discrepancy"] = abs(df["model_prob"] - df["market_prob"])
#     df["inefficiency_score"] = df["discrepancy"] + df["sentiment_score_norm"]
#     ts = now_iso()
#     df["ingestion_ts"] = ts
#     df["last_updated"] = ts
#     return df
#
# # ---------------------- MAIN ----------------------
# def main():
#     events = fetch_all_sports()
#     df = build_dataframe(events)
#     if df.empty:
#         print("⚠️ No valid events parsed.")
#     else:
#         df = enrich_with_sentiment(df)
#         df = compute_signals(df)
#         os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
#         df.to_csv(OUTPUT_FILE, index=False)
#         print(f"✅ Saved to {OUTPUT_FILE}")
#         if SQLITE_FILE:
#             os.makedirs(os.path.dirname(SQLITE_FILE), exist_ok=True)
#             with sqlite3.connect(SQLITE_FILE) as conn:
#                 df.to_sql("events", conn, if_exists="replace", index=False)
#             print(f"💾 Saved to SQLite: {SQLITE_FILE}")
#
# if __name__ == "__main__":
#     main()


# from google.cloud import language_v1
#
# client = language_v1.LanguageServiceClient()
# doc = language_v1.Document(content="Hello world!", type_=language_v1.Document.Type.PLAIN_TEXT)
# sentiment = client.analyze_sentiment(request={"document": doc}).document_sentiment
# print(sentiment.score)


# from google.cloud import aiplatform
# import os
#
# # Initialize Vertex AI
# PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
# REGION_AI = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
# aiplatform.init(project=PROJECT_ID, location=REGION_AI)
#
# # Create embedding model object
# embedding_model = aiplatform.TextEmbeddingModel(
#     model_name="textembedding-gecko@001"  # or another available embedding model
# )


# from vertexai.language_models import TextEmbeddingModel
# import vertexai
# import os
#
# from dotenv import load_dotenv
# load_dotenv()
#
#
# PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
# REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
#
# vertexai.init(project=PROJECT_ID, location=REGION)
#
# # create embedding model
# embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
#
# def embeddings_google(text: str) -> list:
#     if not text:
#         return []
#     try:
#         emb = embedding_model.get_embeddings([text])
#         return emb.values
#     except Exception as e:
#         print("embeddings_google error:", e)
#         return []
#
#
# text = "Hello world"
# vector = embeddings_google(text)
# print(len(vector), vector[:5])

from vertexai.language_models import TextEmbeddingModel
import vertexai
import os

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=REGION)

# Pick a model your project has access to (lite version is safest)
embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko-lite@001")

def embeddings_google(text: str) -> list:
    if not text:
        return []
    try:
        emb = embedding_model.get_embeddings([text])
        return emb[0].values
    except Exception as e:
        print("embeddings_google error:", e)
        return []

vector = embeddings_google("Hello world")
print(len(vector), vector[:5])
