import streamlit as st
import pandas as pd


st.title("Alpha Opportunities Leaderboard")

# Load CSV generated in Colab
df = pd.read_csv("..\data\leaderboard.csv")
st.table(df[['teams','team1_prob_norm','team2_prob_norm','sentiment_score','inefficiency_score']])
