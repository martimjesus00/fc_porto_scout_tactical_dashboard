import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

BG = "#0d0d1a"
CARD_BG = "#12122a"
GOLD = "#FFD700"
GRID = "#1e1e3a"
MUTED = "#8888aa"
PORTO = "#0033A0"
SIM_COLORS = ["#FFD700", "#c0c0c0", "#cd7f32", "#4a90d9", "#7b68ee", "#e74c3c", "#2ecc71", "#e67e22", "#1abc9c", "#9b59b6"]

# ── Métricas por posição específica ──────────────────────────
# Chave = posições Wyscout (qualquer uma que o jogador tenha)
POSITION_METRICS = {
    # Guarda-redes
    "GK": {
        "metrics": ["save_pct", "goals_prevented_p90", "sweeper_actions_p90", "aerial_duels_won_pct", "xgc_p90"],
        "labels":  ["Save %", "Goals Prev/90", "Sweeper/90", "Aerial Won%", "xGC/90"],
        "group":   "Goalkeeper",
    },
    # Centrais
    "CB": {
        "metrics": ["def_actions_p90", "def_duels_won_pct", "aerial_duels_won_pct", "interceptions_p90", "tackles_p90", "passes_accurate_pct", "long_passes_accurate_pct"],
        "labels":  ["Def Act/90", "Def Duel%", "Aerial Won%", "Interc/90", "Tackles/90", "Pass Acc%", "Long Pass%"],
        "group":   "Defender",
    },
    # Laterais (esquerdo e direito)
    "LB": {
        "metrics": ["def_actions_p90", "def_duels_won_pct", "crosses_p90", "crosses_accurate_pct", "progressive_passes_p90", "xa_p90", "dribbles_p90"],
        "labels":  ["Def Act/90", "Def Duel%", "Crosses/90", "Cross Acc%", "Prog Pass/90", "xA/90", "Dribbles/90"],
        "group":   "Defender",
    },
    "RB": {
        "metrics": ["def_actions_p90", "def_duels_won_pct", "crosses_p90", "crosses_accurate_pct", "progressive_passes_p90", "xa_p90", "dribbles_p90"],
        "labels":  ["Def Act/90", "Def Duel%", "Crosses/90", "Cross Acc%", "Prog Pass/90", "xA/90", "Dribbles/90"],
        "group":   "Defender",
    },
    # Médio defensivo
    "DMF": {
        "metrics": ["def_actions_p90", "def_duels_won_pct", "interceptions_p90", "passes_accurate_pct", "progressive_passes_p90", "tackles_p90", "aerial_duels_won_pct"],
        "labels":  ["Def Act/90", "Def Duel%", "Interc/90", "Pass Acc%", "Prog Pass/90", "Tackles/90", "Aerial%"],
        "group":   "Midfielder",
    },
    # Médio centro
    "CMF": {
        "metrics": ["progressive_passes_p90", "key_passes_p90", "passes_accurate_pct", "def_actions_p90", "xa_p90", "dribbles_p90", "final_third_passes_p90"],
        "labels":  ["Prog Pass/90", "Key Pass/90", "Pass Acc%", "Def Act/90", "xA/90", "Dribbles/90", "F3 Pass/90"],
        "group":   "Midfielder",
    },
    # Médio ofensivo / meia
    "AMF": {
        "metrics": ["key_passes_p90", "xa_p90", "xg_p90", "dribbles_p90", "progressive_passes_p90", "box_touches_p90", "shots_p90"],
        "labels":  ["Key Pass/90", "xA/90", "xG/90", "Dribbles/90", "Prog Pass/90", "Box Touch/90", "Shots/90"],
        "group":   "Midfielder",
    },
    # Extremos
    "WF": {
        "metrics": ["goals_p90", "xg_p90", "xa_p90", "dribbles_p90", "crosses_p90", "shots_p90", "progressive_runs_p90"],
        "labels":  ["Goals/90", "xG/90", "xA/90", "Dribbles/90", "Crosses/90", "Shots/90", "Prog Runs/90"],
        "group":   "Forward",
    },
    # Avançado centro
    "CF": {
        "metrics": ["goals_p90", "xg_p90", "shots_p90", "shots_on_target_pct", "aerial_duels_won_pct", "box_touches_p90", "dribbles_p90"],
        "labels":  ["Goals/90", "xG/90", "Shots/90", "Shot Tgt%", "Aerial Won%", "Box Touch/90", "Dribbles/90"],
        "group":   "Forward",
    },
}

# Mapeamento posição Wyscout → chave de métricas
POSITION_KEY_MAP = {
    "GK": "GK", "GKP": "GK",
    "LCB": "CB", "RCB": "CB", "CB": "CB",
    "LB": "LB", "LWB": "LB",
    "RB": "RB", "RWB": "RB",
    "DMF": "DMF", "LDMF": "DMF", "RDMF": "DMF",
    "CMF": "CMF", "LCMF": "CMF", "RCMF": "CMF",
    "AMF": "AMF", "LAMF": "AMF", "RAMF": "AMF",
    "LW": "WF", "RW": "WF", "LWF": "WF", "RWF": "WF",
    "CF": "CF", "SS": "CF",
}

# Fallback por grupo se posição não reconhecida
GROUP_FALLBACK = {
    "Goalkeeper": "GK",
    "Defender":   "CB",
    "Midfielder": "CMF",
    "Forward":    "CF",
}


def _get_position_key(position_str, position_group):
    """Determina a chave de métricas pela posição principal do jogador."""
    positions = [p.strip() for p in str(position_str).split(",")]
    # Usa a primeira posição reconhecida
    for pos in positions:
        if pos in POSITION_KEY_MAP:
            return POSITION_KEY_MAP[pos]
    # Fallback por grupo
    return GROUP_FALLBACK.get(position_group, "CMF")


def _pos_group(p):
    return {"goalkeeper": "Goalkeeper", "defender": "Defender",
            "midfielder": "Midfielder", "forward": "Forward"}.get(str(p).lower(), "Midfielder")


def _hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _compute_similarity(porto_row, scout_df, pos_key, top_n=5):
    config = POSITION_METRICS.get(pos_key, POSITION_METRICS["CMF"])
    metrics = [m for m in config["metrics"] if m in porto_row.index and m in scout_df.columns]
    if not metrics:
        return pd.DataFrame()

    # Filtrar scout pelo grupo de posição
    pos_group = config["group"]
    pool = scout_df[scout_df["position_group"].str.lower() == pos_group.lower()].copy()

    # Filtrar por posição específica
    porto_positions = [p.strip() for p in str(porto_row.get("position", "")).split(",")]

    def has_overlap(pos_str):
        return any(p in [x.strip() for x in str(pos_str).split(",")] for p in porto_positions)

    specific = pool[pool["position"].apply(has_overlap)]
    if len(specific) >= 10:
        pool = specific

    pool = pool.dropna(subset=metrics, thresh=len(metrics) // 2)
    if pool.empty:
        return pd.DataFrame()

    pv = pd.DataFrame([porto_row[metrics].values], columns=metrics)
    all_v = pd.concat([pv, pool[metrics]], ignore_index=True)
    scaled = MinMaxScaler().fit_transform(all_v.fillna(0))
    sims = cosine_similarity(scaled[0:1], scaled[1:])[0]
    pool = pool.copy()
    pool["similarity"] = sims
    pool["similarity_pct"] = (sims * 100).round(1)
    top = pool.nlargest(top_n, "similarity").reset_index(drop=True)
    top["rank"] = range(1, len(top) + 1)
    return top


def _radar_single(porto_row, sim_row, pos_key, porto_name, sim_color, scout_pool=None):
    config = POSITION_METRICS.get(pos_key, POSITION_METRICS["CMF"])
    metrics = config["metrics"]
    labels = config["labels"]
    am = [m for m in metrics if m in porto_row.index and m in sim_row.index]
    al = [labels[i] for i, m in enumerate(metrics) if m in am]
    if not am:
        return go.Figure()

    pv = [float(porto_row[m]) if not pd.isna(porto_row[m]) else 0 for m in am]
    sv = [float(sim_row[m]) if not pd.isna(sim_row[m]) else 0 for m in am]

    if scout_pool is not None and len(scout_pool) > 2:
        avail = [m for m in am if m in scout_pool.columns]
        pool_vals = scout_pool[avail].fillna(0).values
        mn = np.nanmin(pool_vals, axis=0)
        mx = np.nanmax(pool_vals, axis=0)
    else:
        all_v = np.array([pv, sv], dtype=float)
        mn = np.nanmin(all_v, axis=0)
        mx = np.nanmax(all_v, axis=0)

    rng = np.where(mx > mn, mx - mn, 1)
    norm = np.clip(np.array([(np.array(pv) - mn) / rng, (np.array(sv) - mn) / rng]), 0, 1)
    lc = al + [al[0]]
    pvc = norm[0].tolist() + [norm[0][0]]
    svc = norm[1].tolist() + [norm[1][0]]

    fig = go.Figure()
    r, g, b = _hex_rgb(sim_color)
    fig.add_trace(go.Scatterpolar(
        r=svc, theta=lc, fill="toself",
        fillcolor=f"rgba({r},{g},{b},0.15)",
        line=dict(color=sim_color, width=2.5),
        name=f"{sim_row.get('player', '?')} ({sim_row.get('team', '?')})",
        hovertemplate=[
            f"<b>{l}</b><br>{sim_row.get('player', '?')}: "
            f"{round(float(sim_row[m]), 2) if m in sim_row.index and not pd.isna(sim_row[m]) else chr(8212)}"
            f"<extra></extra>"
            for l, m in zip(lc, [*am, am[0]])
        ]
    ))
    fig.add_trace(go.Scatterpolar(
        r=pvc, theta=lc, fill="toself",
        fillcolor="rgba(0,51,160,0.25)",
        line=dict(color=PORTO, width=2.5),
        name=porto_name,
        hovertemplate=[
            f"<b>{l}</b><br>{porto_name}: "
            f"{round(float(porto_row[m]), 2) if m in porto_row.index and not pd.isna(porto_row[m]) else chr(8212)}"
            f"<extra></extra>"
            for l, m in zip(lc, [*am, am[0]])
        ]
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=CARD_BG,
            radialaxis=dict(visible=False, range=[0, 1]),
            angularaxis=dict(tickfont=dict(color="white", size=11), linecolor=GRID, gridcolor=GRID),
        ),
        paper_bgcolor=BG, height=450,
        margin=dict(l=60, r=60, t=50, b=40),
        legend=dict(font=dict(color="white", size=12), bgcolor=CARD_BG, bordercolor=GRID,
                    orientation="h", x=0.2, y=-0.1),
        title=dict(text=f"{porto_name}  vs  {sim_row.get('player', '?')}",
                   font=dict(color=GOLD, size=14), x=0.5, xanchor="center"),
    )
    return fig


def _table(porto_row, similars, pos_key, porto_name):
    config = POSITION_METRICS.get(pos_key, POSITION_METRICS["CMF"])
    metrics = config["metrics"]
    labels = config["labels"]
    am = [m for m in metrics if m in porto_row.index and m in similars.columns]
    al = [labels[i] for i, m in enumerate(metrics) if m in am]
    rows = {
        "Métrica": al,
        porto_name: [round(float(porto_row[m]), 2) if not pd.isna(porto_row[m]) else "—" for m in am],
    }
    for _, sr in similars.iterrows():
        rows[f"{sr.get('player', '?')} ({sr.get('team', '?')})"] = [
            round(float(sr[m]), 2) if not pd.isna(sr[m]) else "—" for m in am
        ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render(porto_players, scout_players):
    st.markdown(
        f'<h2 style="color:{GOLD};margin-bottom:4px;">Similarity Scout</h2>'
        f'<p style="color:{MUTED};margin-top:0;">Top 5 jogadores similares · {len(scout_players)} jogadores analisados</p>',
        unsafe_allow_html=True,
    )
    if porto_players.empty or scout_players.empty:
        st.warning("Dados insuficientes.")
        return

    ps = porto_players.copy()
    ps["_ord"] = ps["position_group"].map(
        {"Goalkeeper": 0, "Defender": 1, "Midfielder": 2, "Forward": 3}
    ).fillna(9)
    ps = ps.sort_values(["_ord", "player"]).reset_index(drop=True)
    ps_min = ps[ps["minutes"] >= 200] if "minutes" in ps.columns else ps

    col_sel, col_f1, col_f2 = st.columns([3, 2, 2])
    with col_sel:
        selected = st.selectbox("Jogador do Porto", ps_min["player"].tolist(), key="tab5s_player")
    with col_f1:
        min_min = st.slider("Minutos mínimos", 300, 1500, 500, 100, key="tab5s_min")
    with col_f2:
        age_min, age_max = int(scout_players["age"].min()), int(scout_players["age"].max())
        age_range = st.slider("Idade", age_min, age_max, (age_min, age_max), key="tab5s_age")

    porto_row = ps_min[ps_min["player"] == selected].iloc[0]
    pos_group = _pos_group(porto_row.get("position_group", "Midfielder"))
    pos_key = _get_position_key(porto_row.get("position", ""), pos_group)
    config = POSITION_METRICS.get(pos_key, POSITION_METRICS["CMF"])
    scout_f = scout_players[
        (scout_players["minutes"] >= min_min) &
        (scout_players["age"] >= age_range[0]) &
        (scout_players["age"] <= age_range[1])
    ].copy()

    st.markdown(
        f'<p style="color:{MUTED};font-size:12px;margin-bottom:16px;">'
        f'{selected} · {porto_row.get("position", "?")} · {pos_group} · '
        f'{int(porto_row.get("minutes", 0))} min · '
        f'<span style="color:{GOLD};">métricas: {config["group"]} ({pos_key})</span></p>',
        unsafe_allow_html=True,
    )

    with st.spinner("A calcular similaridade..."):
        similars = _compute_similarity(porto_row, scout_f, pos_key, top_n=10)

    if similars.empty:
        st.warning("Sem jogadores suficientes para esta posição.")
        return

    st.markdown(
        f'<div style="font-size:14px;font-weight:600;color:{GOLD};margin:16px 0 10px 0;">'
        f'Top 10 Similares — clica num jogador para comparar</div>',
        unsafe_allow_html=True,
    )

    def _render_card_row(row_similars, offset=0):
        cols = st.columns(5)
        for i, (col, (_, row)) in enumerate(zip(cols, row_similars.iterrows())):
            idx = offset + i
            color = SIM_COLORS[idx % len(SIM_COLORS)]
            with col:
                age_str = int(row.get("age", 0)) if not pd.isna(row.get("age", 0)) else "?"
                min_str = int(row.get("minutes", 0)) if not pd.isna(row.get("minutes", 0)) else "?"
                st.markdown(
                    f'<div style="background:{CARD_BG};border:1px solid {color};'
                    f'border-radius:10px;padding:14px 12px;text-align:center;">'
                    f'<div style="font-size:20px;font-weight:700;color:{color};">#{idx+1}</div>'
                    f'<div style="font-size:13px;font-weight:600;color:white;margin:4px 0;">{row.get("player", "?")}</div>'
                    f'<div style="font-size:11px;color:{MUTED};">{row.get("team", "?")}</div>'
                    f'<div style="font-size:11px;color:{MUTED};">{row.get("position", "?")} · {age_str} anos</div>'
                    f'<div style="font-size:11px;color:{MUTED};">{min_str} min</div>'
                    f'<div style="font-size:18px;font-weight:700;color:{color};margin-top:8px;">'
                    f'{row.get("similarity_pct", 0):.1f}%</div>'
                    f'<div style="font-size:10px;color:{MUTED};">similaridade</div></div>',
                    unsafe_allow_html=True,
                )
                if st.button("Ver radar", key=f"sim_btn_{idx}"):
                    st.session_state["selected_sim_idx"] = idx

    _render_card_row(similars.iloc[:5], offset=0)
    st.markdown("<div style='margin-top:10px'></div>", unsafe_allow_html=True)
    _render_card_row(similars.iloc[5:], offset=5)

    sim_idx = st.session_state.get("selected_sim_idx", 0)
    sim_row = similars.iloc[sim_idx]
    sim_color = SIM_COLORS[sim_idx % len(SIM_COLORS)]

    st.markdown(
        f'<div style="font-size:13px;color:{MUTED};margin:16px 0 8px 0;">'
        f'A comparar com: <b style="color:{sim_color};">'
        f'{sim_row.get("player", "?")} ({sim_row.get("team", "?")})</b></div>',
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    scout_pool = scout_f[scout_f["position_group"].str.lower() == config["group"].lower()]
    st.plotly_chart(
        _radar_single(porto_row, sim_row, pos_key, selected, sim_color, scout_pool=scout_pool),
        use_container_width=True,
    )

    st.markdown(
        f'<div style="font-size:14px;font-weight:600;color:{GOLD};margin:16px 0 10px 0;">'
        f'Comparação de Métricas</div>',
        unsafe_allow_html=True,
    )
    _table(porto_row, similars, pos_key, selected)

    st.markdown(
        f'<p style="color:{MUTED};font-size:11px;text-align:right;">'
        f'Data: Wyscout | Analysis: Martim Jesus</p>',
        unsafe_allow_html=True,
    )
