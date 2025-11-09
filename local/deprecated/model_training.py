import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np


df = pd.read_csv("../../data/processed_events.csv")

# TODO: fake??
# Fake features for demonstration: sentiment only
X = df[['sentiment_score']]
y = np.random.randint(0,2,size=len(df))  # placeholder outcomes

model = LogisticRegression()
model.fit(X, y)

# Predict probabilities
df['model_team1_prob'] = model.predict_proba(X)[:,1]
df['discrepancy_team1'] = abs(df['team1_prob_norm'] - df['model_team1_prob'])

# Compute inefficiency score
df['inefficiency_score'] = df['discrepancy_team1'] + df['sentiment_score']

# Top alpha opportunities
leaderboard = df.sort_values('inefficiency_score', ascending=False).head(10)
print(leaderboard[['teams','inefficiency_score']])

# Save leaderboard to CSV
# TODO: change path to data\, add makeosdir if needed
# leaderboard.to_csv("leaderboard.csv", index=False)