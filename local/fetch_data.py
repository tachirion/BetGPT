import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ----------------------
# LOAD CONFIG
# ----------------------
load_dotenv()
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------
# CONFIG
# ----------------------
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
WAIT_BETWEEN = 1
MAX_NEWS = 5  # number of headlines per event

OUTPUT_FILE = ("..\data\processed_events.csv")  # relative to local\

# ----------------------
# HELPER FUNCTIONS
# ----------------------
def get_news_headlines(query, max_results=5):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "pageSize": max_results,
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        articles = data.get("articles", [])
        return [a['title'] for a in articles]
    except:
        return []

def llm_sentiment(headlines):
    if not headlines:
        return 0.0
    prompt = f"""
    You are a sentiment analysis engine.
    Rate the overall sentiment of the following headlines from -1 (very negative) to 1 (very positive). 
    Provide only a single float number:
    {headlines}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        score_text = response.choices[0].message.content.strip()
        score = float(score_text)
        return score
    except:
        return 0.0

def llm_model_prob(event_name, headlines):
    if not headlines:
        return 0.5
    prompt = f"""
    Given the event '{event_name}' and the following headlines:
    {headlines}
    Estimate the probability (0-1) that the first listed team/person wins.
    Only return a float between 0 and 1.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        prob = float(response.choices[0].message.content.strip())
        return prob
    except:
        return 0.5

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
# NEWS & LLM SENTIMENT / MODEL PROB
# ----------------------
news_list = []
sentiments = []
model_probs = []

for teams in df['teams']:
    headlines = get_news_headlines(teams, max_results=MAX_NEWS)
    news_list.append(headlines)
    sentiment = llm_sentiment(headlines)
    sentiments.append(sentiment)
    model_prob = llm_model_prob(teams, headlines)
    model_probs.append(model_prob)
    print(f"{teams} -> {len(headlines)} headlines, sentiment {sentiment:.2f}, model_prob {model_prob:.2f}")

df['news_headlines'] = news_list
df['sentiment_score'] = sentiments
df['sentiment_score_norm'] = (df['sentiment_score'] + 1) / 2  # normalize 0-1
df['model_prob'] = model_probs

# ----------------------
# COMPUTE INEFFICIENCY SCORE
# ----------------------
# Here we take abs discrepancy + sentiment as alpha signal
df['market_prob'] = df['team1_prob_norm']  # for first outcome
df['discrepancy'] = abs(df['model_prob'] - df['market_prob'])
df['inefficiency_score'] = df['discrepancy'] + df['sentiment_score_norm']

# ----------------------
# SAVE CSV
# ----------------------
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved processed events to {OUTPUT_FILE}")