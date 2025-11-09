#!/usr/bin/env python3
"""
prediction_model.py
-------------------
Unified probability estimator for BetGPT.
Uses simple domain heuristics + sentiment blending.

Inputs:
    title : str
    sentiment : float  # optional, -1..1
Outputs:
    float in [0, 1] representing model-implied probability
"""

import random
import re

# ---------- Domain detection ----------
def detect_domain(title: str) -> str:
    t = title.lower()
    if any(k in t for k in ["election", "president", "poll", "vote"]):
        return "politics"
    if any(k in t for k in ["vs", "match", "game", "cup", "league", "tournament"]):
        return "sports"
    if any(k in t for k in ["rain", "temperature", "weather", "snow", "storm"]):
        return "weather"
    if any(k in t for k in ["stock", "price", "inflation", "gdp", "market"]):
        return "finance"
    return "generic"

# ---------- Domain models (simple placeholders) ----------
def model_politics(title: str) -> float:
    # pretend polls favour candidate A ~0.55
    return 0.55 + random.uniform(-0.05, 0.05)

def model_sports(title: str) -> float:
    # pseudo baseline 0.45–0.55 depending on neutral keywords
    base = 0.5 + random.uniform(-0.05, 0.05)
    if "vs" in title.lower():
        # slight random bias per team name length
        parts = re.split(r"\s+vs\.?\s+|\s+v\s+", title, flags=re.I)
        if len(parts) == 2:
            bias = (len(parts[0]) - len(parts[1])) / 100
            base = max(0, min(1, 0.5 + bias))
    return base

def model_weather(title: str) -> float:
    if "rain" in title.lower():
        return 0.3 + random.uniform(-0.05, 0.05)
    if "snow" in title.lower():
        return 0.2 + random.uniform(-0.05, 0.05)
    return 0.5

def model_finance(title: str) -> float:
    return 0.55 + random.uniform(-0.08, 0.08)

def model_generic(title: str) -> float:
    return 0.5

# ---------- Master estimator ----------
def estimate_real_probability(title: str) -> dict:
    """
    Returns per-outcome probabilities + confidence:
    {
        "outcome_probs": {"YES": 0.58, "NO": 0.42},
        "confidence": 0.7
    }
    """
    domain = detect_domain(title)
    if domain == "politics":
        p = model_politics(title)
        return {"outcome_probs": {"YES": p, "NO": 1-p}, "confidence": 0.8}
    elif domain == "sports":
        p = model_sports(title)
        teams = title.split(" vs ")
        if len(teams) < 2:
            teams = ["Team1", "Team2"]
        return {"outcome_probs": {teams[0]: p, teams[1]: 1-p}, "confidence": 0.7}
    elif domain == "weather":
        p = model_weather(title)
        return {"outcome_probs": {"Event": p, "No Event": 1-p}, "confidence": 0.6}
    elif domain == "finance":
        p = model_finance(title)
        return {"outcome_probs": {"Up": p, "Down": 1-p}, "confidence": 0.7}
    else:
        p = model_generic(title)
        return {"outcome_probs": {"YES": p, "NO": 1-p}, "confidence": 0.5}