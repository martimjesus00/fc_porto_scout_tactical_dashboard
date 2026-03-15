
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

BG      = "#0d0d1a"
CARD_BG = "#12122a"
GOLD    = "#FFD700"
GRID    = "#1e1e3a"
MUTED   = "#8888aa"
PORTO   = "#0033A0"

RADAR_METRICS = {
    "Goalkeeper": {
        "Save %":           "save_pct",
        "Goals Prevented":  "goals_prevented_p90",
        "Sweeper Actions":  "sweeper_actions_p90",
        "Aerial Duels Won": "aerial_duels_won_pct",
        "xGC /90":          "xgc_p90",
        "Clean Sheets":     "clean_sheets",
    },
    "Defender": {
        "Def. Actions /90": "def_actions_p90",
        "Def. Duels Won %": "def_duels_won_pct",
        "Aerial Won %":     "aerial_duels_won_pct",
        "Interceptions /90":"interceptions_p90",
        "Prog. Passes /90": "progressive_passes_p90",
        "Long Pass Acc %":  "long_passes_accurate_pct",
    },
    "Midfielder": {
        "Prog. Passes /90": "progressive_passes_p90",
        "Key Passes /90":   "key_passes_p90",
        "Pass Acc %":       "passes_accurate_pct",
        "Def. Actions /90": "def_actions_p90",
        "Dribbles /90":     "dribbles_p90",
        "xA /90":           "xa_p90",
    },
    "Forward": {
        "Goals /90":        "goals_p90",
        "xG /90":           "xg_p90",
        "Shots /90":        "shots_p90",
        "Shot on Tgt %":    "shots_on_target_pct",
        "Dribbles /90":     "dribbles_p90",
        "xA /90":           "xa_p90",
    },
}

DETAIL_METRICS = {
    "Goalkeeper": [
        ("Jogos","matches"),("Minutos","minutes"),("Save %","save_pct"),
        ("Golos Sofridos","goals_conceded"),("Golos Sofridos/90","goals_conceded_p90"),
        ("Goals Prevented","goals_prevented_p90"),("Clean Sheets","clean_sheets"),
        ("Sweeper /90","sweeper_actions_p90"),
    ],
    "Defender": [
        ("Jogos","matches"),("Minutos","minutes"),("Def. Actions/90","def_actions_p90"),
        ("Def. Duels Won%","def_duels_won_pct"),("Aerial Won%","aerial_duels_won_pct"),
        ("Interc./90","interceptions_p90"),("Prog. Passes/90","progressive_passes_p90"),
        ("Pass Acc%","passes_accurate_pct"),
    ],
    "Midfielder": [
        ("Jogos","matches"),("Minutos","minutes"),("Prog. Passes/90","progressive_passes_p90"),
        ("Key Passes/90","key_passes_p90"),("Pass Acc%","passes_accurate_pct"),
        ("xA/90","xa_p90"),("Def. Actions/90","def_actions_p90"),("Dribbles/90","dribbles_p90"),
    ],
    "Forward": [
        ("Jogos","matches"),("Minutos","minutes"),("Goals","goals"),
        ("Goals/90","goals_p90"),("xG/90","xg_p90"),("Shots/90","shots_p90"),
        ("Shot on Tgt%","shots_on_target_pct"),("xA/90","xa_p90"),
    ],
}

def _section(title):
    st.markdown(f'<div style="font-size:15px;font-weight:700;color:{GOLD};margin:24px 0 12px 0;border-left:3px solid {GOLD};padding-left:10px;">{title}</div>', unsafe_allow_html=True)

def _get_position_group(pos_group):
    return {"goalkeeper":"Goalkeeper","defender":"Defender","midfielder":"Midfielder","forward":"Forward"}.get(str(pos_group).lower(),"Midfielder")

def _percentile(value, series):
    clean = series.dropna()
    if len(clean) == 0 or pd.isna(value): return 50
    return round((clean < value).mean() * 100)

def _radar(player_row, group_df, pos_group, player_name):
    metrics = RADAR_METRICS.get(pos_group, RADAR_METRICS["Midfielder"])
    labels, player_vals, raw_player, raw_avg = [], [], [], []
    for label, col in metrics.items():
        if col not in player_row.index or col not in group_df.columns: continue
        p_val = player_row[col]
        col_series = group_df[col].dropna()
        if pd.isna(p_val) or len(col_series) == 0: continue
        mn, mx = col_series.min(), col_series.max()
        norm_p = (p_val - mn) / (mx - mn) if mx > mn else 0.5
        labels.append(label)
        player_vals.append(round(float(norm_p), 3))
        raw_player.append(round(float(p_val), 2))
        raw_avg.append(round(float(col_series.mean()), 2))
    if not labels: return go.Figure()
    lc = labels + [labels[0]]
    pvc = player_vals + [player_vals[0]]
    ac  = [0.5] * (len(labels) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=ac, theta=lc, fill="toself",
        fillcolor="rgba(255,255,255,0.05)",
        line=dict(color="rgba(255,255,255,0.3)", width=1.5, dash="dot"),
        name="Média posição", hoverinfo="skip"))
    fig.add_trace(go.Scatterpolar(r=pvc, theta=lc, fill="toself",
        fillcolor="rgba(0,51,160,0.35)", line=dict(color=PORTO, width=2.5),
        name=player_name,
        hovertemplate=[f"<b>{l}</b><br>{player_name}: {rv}<br>Média: {ra}<extra></extra>"
                       for l,rv,ra in zip(lc, raw_player+[raw_player[0]], raw_avg+[raw_avg[0]])]))
    fig.update_layout(
        polar=dict(bgcolor=CARD_BG,
            radialaxis=dict(visible=False, range=[0,1]),
            angularaxis=dict(tickfont=dict(color="white",size=11), linecolor=GRID, gridcolor=GRID)),
        paper_bgcolor=BG, height=420,
        margin=dict(l=60,r=60,t=60,b=40),
        legend=dict(font=dict(color="white",size=11), bgcolor=CARD_BG, bordercolor=GRID, orientation="h", x=0.3, y=-0.1),
        title=dict(text=f"{player_name} — {pos_group}", font=dict(color=GOLD,size=14), x=0.5, xanchor="center"))
    return fig

def _percentile_bars(player_row, group_df, pos_group):
    metrics = RADAR_METRICS.get(pos_group, RADAR_METRICS["Midfielder"])
    labels, pcts, vals = [], [], []
    for label, col in metrics.items():
        if col not in player_row.index or col not in group_df.columns: continue
        val = player_row[col]
        if pd.isna(val): continue
        pcts.append(_percentile(val, group_df[col]))
        labels.append(label)
        vals.append(round(float(val), 2))
    if not labels: return go.Figure()
    colors = ["#2ecc71" if p >= 70 else "#f39c12" if p >= 40 else "#e74c3c" for p in pcts]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pcts, y=labels, orientation="h", marker_color=colors,
        marker_line=dict(color=BG, width=1),
        text=[f"{p}th  ({v})" for p,v in zip(pcts,vals)],
        textposition="inside", textfont=dict(color="white",size=10),
        hovertemplate="%{y}<br>Percentil: %{x}<extra></extra>"))
    fig.add_vline(x=50, line_color="white", line_width=1, line_dash="dot", opacity=0.3)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, height=300,
        margin=dict(l=10,r=20,t=30,b=10),
        title=dict(text="Percentil no plantel (por posição)", font=dict(color=GOLD,size=13), x=0),
        xaxis=dict(range=[0,100], tickfont=dict(color=MUTED), gridcolor=GRID,
                   title=dict(text="Percentil", font=dict(color=MUTED))),
        yaxis=dict(tickfont=dict(color="white",size=11), showgrid=False))
    return fig

def _detail_table(player_row, group_df, pos_group):
    metrics = DETAIL_METRICS.get(pos_group, DETAIL_METRICS["Midfielder"])
    rows = []
    for label, col in metrics:
        p_val = player_row[col] if col in player_row.index else np.nan
        avg   = group_df[col].mean() if col in group_df.columns else np.nan
        rows.append({"Métrica": label,
                     "Jogador": round(float(p_val),2) if not pd.isna(p_val) else "—",
                     "Média Posição": round(float(avg),2) if not pd.isna(avg) else "—"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render(players):
    st.markdown(f'<h2 style="color:{GOLD};margin-bottom:4px;">Player Profiles</h2><p style="color:{MUTED};margin-top:0;">Análise individual · FC Porto 2025/26</p>', unsafe_allow_html=True)
    if players.empty:
        st.warning("Sem dados de jogadores."); return
    pos_order = {"Goalkeeper":0,"Defender":1,"Midfielder":2,"Forward":3}
    ps = players.copy()
    ps["_pos_order"] = ps["position_group"].map(pos_order).fillna(9)
    ps = ps.sort_values(["_pos_order","player"]).reset_index(drop=True)
    col_sel, col_info = st.columns([3,5])
    with col_sel:
        selected = st.selectbox("Selecciona jogador", ps["player"].tolist(), key="tab5_player")
    player_row = ps[ps["player"] == selected].iloc[0]
    pos_group  = _get_position_group(player_row.get("position_group","Midfielder"))
    same_pos   = ps[ps["position_group"].str.lower() == pos_group.lower()]
    with col_info:
        pos, age, mins, games = player_row.get("position","—"), player_row.get("age","—"), player_row.get("minutes","—"), player_row.get("matches","—")
        st.markdown(f'<div style="background:{CARD_BG};border:1px solid {GRID};border-radius:10px;padding:14px 20px;margin-top:4px;"><span style="color:{GOLD};font-size:18px;font-weight:700;">{selected}</span><span style="color:{MUTED};font-size:13px;margin-left:12px;">{pos} · {pos_group}</span><br><span style="color:{MUTED};font-size:12px;">Idade: {age} &nbsp;|&nbsp; Jogos: {games} &nbsp;|&nbsp; Minutos: {mins}</span></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    col_radar, col_pct = st.columns([1,1])
    with col_radar:
        st.plotly_chart(_radar(player_row, same_pos, pos_group, selected), use_container_width=True)
    with col_pct:
        st.markdown("<br>", unsafe_allow_html=True)
        st.plotly_chart(_percentile_bars(player_row, same_pos, pos_group), use_container_width=True)
    _section("Stats Detalhadas")
    _detail_table(player_row, same_pos, pos_group)
    st.markdown(f'<p style="color:{MUTED};font-size:11px;text-align:right;">Data: Wyscout | Analysis: Martim Jesus</p>', unsafe_allow_html=True)
