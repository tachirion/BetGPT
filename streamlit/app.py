#!/usr/bin/env python3
"""
inefficiency_dashboard.py
Streamlit app to visualize inefficiency leaderboards.
"""

import streamlit as st
import pandas as pd
import glob
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from local.inefficiency_engine import main as recompute_leaderboard

DATA_DIR = Path("../data")

# --- Streamlit page config (handle older versions) ---
try:
    st.set_page_config(
        page_title="Market Inefficiency Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
except AttributeError:
    pass  # Older Streamlit versions

# --- Helpers ---
try:
    # @st.cache_data
    def load_latest_leaderboard():
        files = sorted(glob.glob(str(DATA_DIR / "inefficiency_leaderboard_*.csv")))
        if not files:
            st.error("No leaderboard files found. Run inefficiency_engine.py first.")
            return None, None
        latest = files[-1]
        df = pd.read_csv(latest)
        return df, Path(latest).name
except AttributeError:
    @st.cache
    def load_latest_leaderboard():
        files = sorted(glob.glob(str(DATA_DIR / "inefficiency_leaderboard_*.csv")))
        if not files:
            st.error("No leaderboard files found. Run inefficiency_engine.py first.")
            return None, None
        latest = files[-1]
        df = pd.read_csv(latest)
        return df, Path(latest).name

# --- UI ---
st.title("📊 Market Inefficiency Leaderboard")

# --- Sidebar ---
st.sidebar.header("Filters & Actions")
if st.sidebar.button("🔄 Recompute Inefficiency"):
    with st.spinner("Recomputing..."):
        recompute_leaderboard()

df, filename = load_latest_leaderboard()
if df is None:
    st.stop()

# --- Top N Filter ---
top_n = st.sidebar.slider("Top N Markets", min_value=5, max_value=50, value=10)
df_top = df.head(top_n)

# --- Summary metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Markets Loaded", len(df_top))
col2.metric("Top Inefficiency", f"{df_top['inefficiency_score'].max():.3f}")
col3.metric("Average Inefficiency", f"{df_top['inefficiency_score'].mean():.3f}")

st.caption(f"Loaded file: `{filename}` | Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}`")

# --- Leaderboard Table ---
st.subheader("🏁 Inefficiency Rankings")
st.dataframe(
    df_top[[
        "rank", "teams", "inefficiency_score", "recommendation",
        "team1_prob_norm", "model_team1_prob", "team2_prob_norm", "model_team2_prob"
    ]].sort_values("inefficiency_score", ascending=False).reset_index(drop=True),
    use_container_width=True
)

# --- Chart: Inefficiency Distribution ---
st.subheader("📈 Inefficiency Score Distribution")
fig, ax = plt.subplots()
ax.hist(df_top["inefficiency_score"], bins=30, color="skyblue", edgecolor="black")
ax.set_xlabel("Inefficiency Score")
ax.set_ylabel("Count")
st.pyplot(fig)

# --- Chart: Model vs Market Probabilities ---
st.subheader("⚖️ Model vs Market (Team1 Probability)")
fig2, ax2 = plt.subplots()
ax2.scatter(df_top["team1_prob_norm"], df_top["model_team1_prob"], alpha=0.6)
ax2.plot([0, 1], [0, 1], 'r--', linewidth=1)
ax2.set_xlabel("Market Probability (Team1)")
ax2.set_ylabel("Model Probability (Team1)")
st.pyplot(fig2)
