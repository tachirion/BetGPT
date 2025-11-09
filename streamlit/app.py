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

# --- Helpers ---
@st.cache_data
def load_latest_leaderboard():
    files = sorted(glob.glob(str(DATA_DIR / "inefficiency_leaderboard_*.csv")))
    if not files:
        st.error("No leaderboard files found. Run inefficiency_engine.py first.")
        return None
    latest = files[-1]
    df = pd.read_csv(latest)
    return df, Path(latest).name

# --- UI Layout ---
st.set_page_config(
    page_title="Market Inefficiency Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📊 Market Inefficiency Leaderboard")

# --- Sidebar ---
st.sidebar.header("Filters & Actions")
if st.sidebar.button("🔄 Recompute Inefficiency"):
    with st.spinner("Recomputing..."):
        recompute_leaderboard()

df, filename = load_latest_leaderboard()
if df is None:
    st.stop()

# --- Domain Filter ---
domains = sorted(df["market_type"].unique())
selected_domains = st.sidebar.multiselect("Market Type", domains, default=domains)
filtered = df[df["market_type"].isin(selected_domains)]

# --- Summary metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Markets Loaded", len(filtered))
col2.metric("Top Inefficiency", f"{filtered['inefficiency_score'].max():.3f}")
col3.metric("Average Inefficiency", f"{filtered['inefficiency_score'].mean():.3f}")

st.caption(f"Loaded file: `{filename}` | Last updated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}`")

# --- Leaderboard Table ---
st.subheader("🏁 Inefficiency Rankings")
st.dataframe(
    filtered[[
        "rank", "market_title", "market_type", "inefficiency_score",
        "recommendation", "confidence", "model_team1_prob", "market_team1_prob"
    ]].sort_values("inefficiency_score", ascending=False).reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)

# --- Chart: Inefficiency Distribution ---
st.subheader("📈 Inefficiency Score Distribution")
fig, ax = plt.subplots()
ax.hist(filtered["inefficiency_score"], bins=30)
ax.set_xlabel("Inefficiency Score")
ax.set_ylabel("Count")
st.pyplot(fig)

# --- Chart: Model vs Market Probabilities ---
st.subheader("⚖️ Model vs Market (Team1 Probability)")
fig2, ax2 = plt.subplots()
ax2.scatter(filtered["market_team1_prob"], filtered["model_team1_prob"], alpha=0.6)
ax2.plot([0, 1], [0, 1], 'r--', linewidth=1)
ax2.set_xlabel("Market Probability (Team1)")
ax2.set_ylabel("Model Probability (Team1)")
st.pyplot(fig2)
