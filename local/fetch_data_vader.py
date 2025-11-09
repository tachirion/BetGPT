import requests
import pandas as pd
import time
import os
import feedparser
import urllib.parse
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv


# ----------------------
# CONFIG
# ----------------------
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

if not ODDS_API_KEY:
    raise ValueError("ODDS_API_KEY not found in environment variables!")

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
PAGE_LIMIT = 2
WAIT_BETWEEN = 1  # seconds to avoid rate limits
MAX_NEWS = 5      # headlines per event

# ----------------------
# HELPER: Google News RSS Headlines
# ----------------------
news_cache = {}

def get_google_news_headlines(query, max_results=5):
    """Fetch top headlines from Google News RSS for a query."""
    if query in news_cache:
        return news_cache[query]

    query_encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={query_encoded}&hl=en-US&gl=US&ceid=US:en"

    try:
        feed = feedparser.parse(url)
        articles = feed.get('entries', [])[:max_results]
        headlines = [entry['title'] for entry in articles if 'title' in entry]
        if not headlines:
            print(f"[INFO] No headlines found for '{query}'")
        news_cache[query] = headlines
        return headlines
    except Exception as e:
        print(f"[ERROR] Failed to fetch Google News for '{query}': {e}")
        return []

# ----------------------
# FETCH ODDS EVENTS
# ----------------------
all_events = []
for sport in SPORTS:
    print(f"\n[FETCHING] Sport: {sport}")
    for page in range(1, PAGE_LIMIT + 1):
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        params = {"apiKey": ODDS_API_KEY, "regions": REGION, "markets": MARKETS, "page": page}
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
        except Exception as e:
            print(f"[ODDSAPI ERROR] Failed page {page} of {sport}: {e}")
            break

        if isinstance(data, dict) and "message" in data:
            print(f"[ODDSAPI ERROR] {sport} page {page}: {data['message']}")
            break
        if not data:
            print(f"[ODDSAPI INFO] No data on page {page} for {sport}")
            break

        all_events.extend(data)
        print(f"[ODDSAPI INFO] Page {page} fetched: {len(data)} events")
        time.sleep(WAIT_BETWEEN)

print(f"\n[TOTAL EVENTS] Fetched: {len(all_events)}")

# ----------------------
# PARSE EVENTS
# ----------------------
events_parsed = []
for event in all_events:
    if not isinstance(event, dict):
        continue
    bookmakers = event.get('bookmakers', [])
    if not bookmakers:
        continue
    for bookmaker in bookmakers:
        markets = bookmaker.get('markets', [])
        if not markets:
            continue
        outcomes = markets[0].get('outcomes', [])
        if len(outcomes) < 2:
            continue
        events_parsed.append({
            "sport": event.get('sport_key'),
            "event_id": event.get('id'),
            "teams": f"{outcomes[0]['name']} vs {outcomes[1]['name']}",
            "team1_odds": outcomes[0]['price'],
            "team2_odds": outcomes[1]['price']
        })

df = pd.DataFrame(events_parsed)
print(f"[PARSED EVENTS] {len(df)} events")

if df.empty:
    raise ValueError("No events parsed. Check your ODDS_API_KEY or SPORT list.")

# ----------------------
# MARKET PROBABILITIES
# ----------------------
df['team1_prob'] = 1 / df['team1_odds']
df['team2_prob'] = 1 / df['team2_odds']
total = df['team1_prob'] + df['team2_prob']
df['team1_prob_norm'] = df['team1_prob'] / total
df['team2_prob_norm'] = df['team2_prob'] / total

# ----------------------
# NEWS & SENTIMENT
# ----------------------
analyzer = SentimentIntensityAnalyzer()

def compute_sentiment(headlines):
    if not headlines:
        return 0
    scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    print(f"[DEBUG] Headlines: {headlines}")
    print(f"[DEBUG] Scores: {scores}")
    avg_score = sum(scores) / len(scores)
    return avg_score

news_list = []
sentiments = []

for teams in df['teams']:
    headlines = get_google_news_headlines(teams, max_results=MAX_NEWS)
    sentiment_score = compute_sentiment(headlines)
    news_list.append(headlines)
    sentiments.append(sentiment_score)
    print(f"[SENTIMENT] {teams} -> {len(headlines)} headlines, sentiment {sentiment_score:.2f}")

df['news_headlines'] = news_list
df['sentiment_score'] = sentiments
df['sentiment_score_norm'] = (df['sentiment_score'] + 1) / 2  # normalize 0-1

# ----------------------
# SAVE DATASET
# ----------------------
output_dir = os.path.join("..", "data")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "expanded_multi_sports_events.csv")
df.to_csv(output_path, index=False)
print(f"\n[SAVED] Dataset saved to {output_path}")