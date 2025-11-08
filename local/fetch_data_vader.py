import requests
import pandas as pd
import time
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from dotenv import load_dotenv

# ----------------------
# LOAD CONFIG
# ----------------------
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

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
PAGE_LIMIT = 5
WAIT_BETWEEN = 1  # seconds to avoid rate limits
MAX_NEWS = 5      # headlines per event

# ----------------------
# HELPER: Fetch News Headlines
# ----------------------
def get_headlines(query, max_results=5):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": max_results,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    try:
        data = response.json()
    except:
        return []
    if "articles" not in data:
        return []
    return [article['title'] for article in data['articles']]

# ----------------------
# FETCH EVENTS
# ----------------------
all_events = []
for sport in SPORTS:
    print(f"Fetching {sport}...")
    for page in range(1, PAGE_LIMIT + 1):
        url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds"
        params = {"apiKey": ODDS_API_KEY, "regions": REGION, "markets": MARKETS, "page": page}
        try:
            response = requests.get(url, params=params)
            data = response.json()
        except Exception as e:
            print(f"Failed page {page} of {sport}: {e}")
            break

        if isinstance(data, dict) and "message" in data:
            print(f"API error {sport} page {page}: {data['message']}")
            break
        if not data:
            break

        all_events.extend(data)
        print(f"Page {page} fetched: {len(data)} events")
        time.sleep(WAIT_BETWEEN)

print(f"Total events fetched: {len(all_events)}")

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
print(f"Parsed events: {len(df)}")

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
    scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
    return sum(scores)/len(scores) if scores else 0

news_list = []
sentiments = []

for teams in df['teams']:
    headlines = get_headlines(teams, max_results=MAX_NEWS)
    news_list.append(headlines)
    sentiments.append(compute_sentiment(headlines))
    print(f"{teams} -> {len(headlines)} headlines, sentiment {sentiments[-1]:.2f}")

df['news_headlines'] = news_list
df['sentiment_score'] = sentiments
df['sentiment_score_norm'] = (df['sentiment_score'] + 1) / 2  # normalize 0-1

# ----------------------
# SAVE DATASET
# ----------------------
# TODO: change path to data\, add makeosdir if needed
df.to_csv("expanded_multi_sports_events.csv", index=False)
print("Saved expanded dataset to expanded_multi_sports_events.csv")
