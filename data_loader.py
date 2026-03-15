
import pandas as pd
import streamlit as st
import os

DATA_DIR = "data/processed"

@st.cache_data
def load_porto():
    df = pd.read_csv(os.path.join(DATA_DIR, "porto_clean.csv"), parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["game_number"] = range(1, len(df) + 1)
    if "xga" not in df.columns:
        df["xga"] = df["goals_conceded"] if "goals_conceded" in df.columns else df["goals_against"]
    return df

@st.cache_data
def load_liga_big3():
    df = pd.read_csv(os.path.join(DATA_DIR, "liga_big3.csv"), parse_dates=["date"])
    return df

@st.cache_data
def load_players():
    df = pd.read_csv(os.path.join(DATA_DIR, "porto_players_main.csv"))
    return df

@st.cache_data
def load_players_all():
    df = pd.read_csv(os.path.join(DATA_DIR, "porto_players_all.csv"))
    return df


@st.cache_data
def load_scout():
    import os
    path = os.path.join(DATA_DIR, "scout_players.csv")
    return pd.read_csv(path)
