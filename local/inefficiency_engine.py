# #!/usr/bin/env python3
# """
# inefficiency_engine.py
# Compute inefficiency scores between market implied probabilities
# and sentiment/model-based expectations.
#
# Reads:
#     ../data/expanded_multi_sports_events.csv
#
# Outputs:
#     ../data/inefficiency_leaderboard_<timestamp>.csv
# """
#
# import pandas as pd
# import numpy as np
# from datetime import datetime
# from pathlib import Path
# from prediction_model import estimate_real_probability, detect_domain
#
# DATA_PATH = Path("../data/expanded_multi_sports_events.csv")
# OUTPUT_DIR = Path("../data")
# OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
#
#
# def compute_inefficiency(df: pd.DataFrame) -> pd.DataFrame:
#     """Compute inefficiency scores using sentiment-adjusted model predictions."""
#     df = df.copy()
#
#     # --- Setup and defaults ---
#     if "teams" not in df.columns:
#         raise ValueError("Missing 'teams' column in dataset.")
#
#     df["market_title"] = df["sport_title"] if "sport_title" in df.columns else df["teams"]
#     df["market_type"] = df.get("sport_key", "sports").apply(lambda x: str(x).split("-")[0])
#
#     # --- Model probability via prediction_model ---
#     df["model_team1_prob"] = df.apply(
#         lambda r: np.clip(
#             estimate_real_probability(r["market_title"]) +
#             0.1 * (r.get("sentiment_score_norm", 0.5) - 0.5),
#             0, 1
#         ),
#         axis=1
#     )
#     df["model_team2_prob"] = 1 - df["model_team1_prob"]
#
#     # --- Inefficiency computation ---
#     df["inefficiency_team1"] = abs(df["model_team1_prob"] - df["team1_prob_norm"])
#     df["inefficiency_team2"] = abs(df["model_team2_prob"] - df["team2_prob_norm"])
#     df["inefficiency_score"] = df[["inefficiency_team1", "inefficiency_team2"]].max(axis=1)
#
#     # --- Confidence / momentum ---
#     df["confidence"] = df["inefficiency_score"] * 100
#     df["momentum"] = np.random.uniform(-0.05, 0.05, len(df))  # placeholder drift
#
#     # --- Recommendation logic ---
#     def recommend(row):
#         team1, team2 = row["teams"].split(" vs ") if " vs " in row["teams"] else (row["teams"], "")
#         return f"BUY YES on {team1}" if row["model_team1_prob"] > row["team1_prob_norm"] else f"BUY YES on {team2}"
#
#     df["recommendation"] = df.apply(recommend, axis=1)
#
#     # --- Rank ---
#     df["rank"] = df["inefficiency_score"].rank(ascending=False, method="first")
#
#     # --- Output-friendly renames ---
#     df.rename(columns={
#         "team1_prob_norm": "market_team1_prob",
#         "team2_prob_norm": "market_team2_prob"
#     }, inplace=True)
#
#     return df.sort_values("inefficiency_score", ascending=False)
#
#
# def main():
#     df = pd.read_csv(DATA_PATH)
#     if df.empty:
#         raise ValueError("No data found — run fetch_data_vader.py first.")
#
#     scored_df = compute_inefficiency(df)
#     out_path = OUTPUT_DIR / f"inefficiency_leaderboard_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv"
#     scored_df.to_csv(out_path, index=False)
#
#     print(f"[SAVED] Inefficiency leaderboard saved to {out_path}")
#     print(scored_df[[
#         "market_title", "inefficiency_score", "recommendation", "confidence"
#     ]].head(10))
#
#
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
inefficiency_engine.py
Compute inefficiency scores from expanded_multi_sports_events.csv
using:
- market implied probabilities
- sentiment signals
- model discrepancy via prediction_model.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from prediction_model import estimate_real_probability  # NEW

DATA_PATH = Path("../data/expanded_multi_sports_events.csv")
OUTPUT_DIR = Path("../data")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def compute_inefficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute inefficiency scores between market probabilities and sentiment-adjusted model.
    """
    model_probs_list = []
    for idx, row in df.iterrows():
        title = row["teams"]
        model_outcome = estimate_real_probability(title)
        model_probs_list.append(model_outcome["outcome_probs"])

    # Convert list of dicts into DataFrame
    model_df = pd.DataFrame(model_probs_list).fillna(0.5)
    # Align with team1/team2 columns
    team1_name = df["teams"].str.split(" vs ").str[0]
    team2_name = df["teams"].str.split(" vs ").str[1]

    df["model_team1_prob"] = [probs.get(t, 0.5) for probs, t in zip(model_probs_list, team1_name)]
    df["model_team2_prob"] = [probs.get(t, 0.5) for probs, t in zip(model_probs_list, team2_name)]

    # Inefficiency = absolute difference between market and model
    df["inefficiency_team1"] = abs(df["model_team1_prob"] - df["team1_prob_norm"])
    df["inefficiency_team2"] = abs(df["model_team2_prob"] - df["team2_prob_norm"])
    df["inefficiency_score"] = df[["inefficiency_team1", "inefficiency_team2"]].max(axis=1)

    # Rank by inefficiency (descending)
    df["rank"] = df["inefficiency_score"].rank(ascending=False, method="first")

    # Recommendation
    df["recommendation"] = np.where(
        df["model_team1_prob"] > df["team1_prob_norm"],
        f"BUY YES on {team1_name}",
        f"BUY YES on {team2_name}"
    )

    return df.sort_values("inefficiency_score", ascending=False)


def main():
    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("No data found — run fetch_data_vader.py first.")

    scored_df = compute_inefficiency(df)
    out_path = OUTPUT_DIR / f"inefficiency_leaderboard_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.csv"
    scored_df.to_csv(out_path, index=False)
    print(f"[SAVED] Inefficiency leaderboard saved to {out_path}")
    print(scored_df[["teams", "inefficiency_score", "recommendation"]].head(10))


if __name__ == "__main__":
    main()