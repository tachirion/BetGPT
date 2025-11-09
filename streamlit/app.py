import streamlit as st
import pandas as pd


# TODO: imporve the styling of the table, the app itself to include LLM insights, etc.

st.title("Alpha Opportunities Leaderboard")

df = pd.read_csv("..\data\leaderboard.csv")
st.table(df[['teams','team1_prob_norm','team2_prob_norm','sentiment_score','inefficiency_score']])