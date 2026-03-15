
import streamlit as st
import sys
sys.path.insert(0, "/content")

from data_loader import load_porto, load_liga_big3, load_players, load_scout, load_scout
from tabs import tab1_season, tab2_tactical, tab3_context, tab5_players, tab5_scout, tab5_players, tab5_scout

st.set_page_config(page_title="FC Porto | Farioli 2025/26", page_icon="🔵", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
  html, body, [class*="css"] { background-color: #0d0d1a; color: #ffffff; font-family: 'Inter', sans-serif; }
  .stTabs [data-baseweb="tab-list"] { background-color: #12122a; border-radius: 12px; padding: 4px; gap: 4px; }
  .stTabs [data-baseweb="tab"] { background-color: transparent; color: #8888aa; border-radius: 8px; padding: 8px 20px; font-weight: 600; font-size: 13px; border: none; }
  .stTabs [aria-selected="true"] { background-color: #0033A0 !important; color: white !important; }
  .block-container { padding-top: 2rem; padding-bottom: 2rem; }
  hr { border-color: #1e1e3a; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style="display:flex;align-items:center;gap:16px;margin-bottom:8px;">
  <div>
    <div style="font-size:22px;font-weight:700;color:white;">FC Porto — Análise Táctica</div>
    <div style="font-size:13px;color:#8888aa;">Francesco Farioli · Liga Portugal + Europa League · 2025/26</div>
  </div>
</div>
""", unsafe_allow_html=True)

porto     = load_porto()
liga_big3 = load_liga_big3()
players   = load_players()
scout     = load_scout()
scout     = load_scout()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Season Overview",
    "🧩 Tactical Identity",
    "🔍 Context Analysis",
    "👤 Player Profiles",
    "🔎 Similarity Scout",
])

with tab1: tab1_season.render(porto)
with tab2: tab2_tactical.render(porto, liga_big3)
with tab3: tab3_context.render(porto, liga_big3)
with tab4: tab5_players.py.render(players, scout)
with tab5: tab5_scout.render(players, scout)
