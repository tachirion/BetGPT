import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Example: fetch market odds from TheOddsAPI
API_KEY = "d6a7da0e0747fccb941b245d5702ccfd"
SPORT = "soccer_epl"
REGION = "uk"

url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
params = {"apiKey": API_KEY, "regions": REGION, "markets": "h2h"}
response = requests.get(url, params=params)
data = response.json()

# Validate response
if not isinstance(data, list):
    print("Unexpected API response:", data)
    exit()

# Build dataframe
events = []
for event in data:
    if not isinstance(event, dict):
        continue  # skip invalid items
    bookmakers = event.get('bookmakers', [])
    if not bookmakers:
        continue  # skip if no bookmakers

    for bookmaker in bookmakers:
        markets = bookmaker.get('markets', [])
        if not markets:
            continue
        odds = markets[0].get('outcomes', [])
        if len(odds) < 2:
            continue
        events.append({
            "event_id": event.get('id'),
            "teams": f"{odds[0]['name']} vs {odds[1]['name']}",
            "team1_odds": odds[0]['price'],
            "team2_odds": odds[1]['price'],
        })


df_odds = pd.DataFrame(events)

# Compute market probabilities
df_odds['team1_prob'] = 1 / df_odds['team1_odds']
df_odds['team2_prob'] = 1 / df_odds['team2_odds']
total = df_odds['team1_prob'] + df_odds['team2_prob']
df_odds['team1_prob_norm'] = df_odds['team1_prob'] / total
df_odds['team2_prob_norm'] = df_odds['team2_prob'] / total

# Sentiment analysis (simulate or fetch real headlines)
analyzer = SentimentIntensityAnalyzer()
news = {
    "Manchester United vs Liverpool": [
        "Manchester United looks strong this season",
        "Liverpool struggling with injuries"
    ]
}
df_odds['sentiment_score'] = df_odds['teams'].apply(
    lambda x: sum([analyzer.polarity_scores(h)['compound'] for h in news.get(x, [])]) / max(1,len(news.get(x,[])))
)

# Save CSV to load in Colab
df_odds.to_csv("processed_events.csv", index=False)
print("Processed data saved to processed_events.csv")