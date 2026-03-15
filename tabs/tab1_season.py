
import streamlit as st
import plotly.graph_objects as go

BG      = "#0d0d1a"
CARD_BG = "#12122a"
PORTO_BL= "#0033A0"
GOLD    = "#FFD700"
GRID    = "#1e1e3a"
MUTED   = "#8888aa"
WIN_C   = "#2ecc71"
DRAW_C  = "#f39c12"
LOSS_C  = "#e74c3c"
XG_FOR  = "#0033A0"
XG_AG   = "#CC0000"

COMP_LABELS = {
    "Todas":         None,
    "Liga Portugal": "Liga Portugal",
    "Europa League": "Europa League",
}

def _kpi_card(label, value, delta="", delta_positive=True):
    arrow   = "▲" if delta_positive else "▼"
    d_color = "#2ecc71" if delta_positive else "#e74c3c"
    delta_html = f'<div style="font-size:12px;color:{d_color};margin-top:4px">{arrow} {delta}</div>' if delta else ""
    return f"""
    <div style="background:{CARD_BG};border:1px solid {GRID};border-radius:12px;
                padding:20px 16px;text-align:center;">
      <div style="font-size:12px;color:{MUTED};text-transform:uppercase;
                  letter-spacing:1px;margin-bottom:8px;">{label}</div>
      <div style="font-size:32px;font-weight:700;color:white;">{value}</div>
      {delta_html}
    </div>"""

def _result_color(r):
    return {"Win": WIN_C, "Draw": DRAW_C, "Loss": LOSS_C}.get(r, MUTED)

def render(porto):
    st.markdown(
        f'<h2 style="color:{GOLD};margin-bottom:4px;">Season Overview</h2>'
        f'<p style="color:{MUTED};margin-top:0;">FC Porto · Farioli · 2025/26</p>',
        unsafe_allow_html=True,
    )

    col_filter, _ = st.columns([2, 6])
    with col_filter:
        comp_choice = st.selectbox("Competição", list(COMP_LABELS.keys()),
                                   key="tab1_comp", label_visibility="collapsed")

    comp_val = COMP_LABELS[comp_choice]
    df = porto.copy()
    if comp_val:
        df = df[df["competition_clean"] == comp_val]

    if df.empty:
        st.warning("Sem jogos para a competição seleccionada.")
        return

    n      = len(df)
    ppda   = df["ppda"].mean()
    xg     = df["xg"].mean()
    poss   = df["possession_pct"].mean()
    ga     = df["goals_against"].mean()
    wins   = (df["result"] == "Win").sum()
    draws  = (df["result"] == "Draw").sum()
    losses = (df["result"] == "Loss").sum()

    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        (c1, "PPDA",           f"{ppda:.1f}",  "pressing intenso < 8",  ppda < 8),
        (c2, "xG Criado / 90", f"{xg:.2f}",    "referência Big3: 2.00", xg >= 2.0),
        (c3, "Posse %",        f"{poss:.1f}%",  "",                      True),
        (c4, "Golos Sofridos", f"{ga:.2f}",    "por jogo",               ga < 0.6),
        (c5, "Jogos",          f"{n}",          f"V{wins} E{draws} D{losses}", wins > losses),
    ]
    for col, label, val, delta, positive in cards:
        with col:
            st.markdown(_kpi_card(label, val, delta, positive), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    df_sorted = df.sort_values("date").reset_index(drop=True)
    df_sorted["game_num"] = range(1, len(df_sorted) + 1)
    df_sorted["label"]    = df_sorted.apply(
        lambda r: f"{r['match']}<br>{r['result']} | xG {r['xg']:.1f}", axis=1)
    result_colors = [_result_color(r) for r in df_sorted["result"]]
    pts_map = {"Win": 3, "Draw": 1, "Loss": 0}
    df_sorted["pts"]     = df_sorted["result"].map(pts_map)
    df_sorted["cum_pts"] = df_sorted["pts"].cumsum()

    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(
        x=df_sorted["game_num"], y=[0.5] * len(df_sorted),
        mode="markers+text",
        marker=dict(size=22, color=result_colors,
                    line=dict(color="white", width=1.5), symbol="circle"),
        text=df_sorted["result"].str[0],
        textfont=dict(color="white", size=10, family="Arial Black"),
        textposition="middle center",
        hovertext=df_sorted["label"], hoverinfo="text",
        showlegend=False,
    ))
    fig_tl.add_trace(go.Scatter(
        x=df_sorted["game_num"], y=df_sorted["cum_pts"],
        mode="lines", line=dict(color=GOLD, width=2, dash="dot"),
        name="Pts acumulados", yaxis="y2",
        hovertemplate="%{y} pts<extra></extra>",
    ))
    fig_tl.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, height=220,
        margin=dict(l=0, r=0, t=30, b=10),
        title=dict(text="Resultados jogo a jogo", font=dict(color=GOLD, size=14), x=0),
        xaxis=dict(title="Jogo nº", tickfont=dict(color=MUTED), gridcolor=GRID, showgrid=False),
        yaxis=dict(visible=False, range=[-0.2, 1.2]),
        yaxis2=dict(overlaying="y", side="right",
                    tickfont=dict(color=GOLD, size=10), gridcolor=GRID, showgrid=False,
                    title=dict(text="Pts", font=dict(color=GOLD, size=10))),
        legend=dict(font=dict(color="white", size=10), bgcolor=CARD_BG,
                    bordercolor=GRID, x=0.01, y=0.99),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

    fig_xg = go.Figure()
    fig_xg.add_trace(go.Bar(
        x=df_sorted["game_num"], y=df_sorted["xg"],
        name="xG Criado", marker_color=XG_FOR,
        hovertemplate="Jogo %{x}<br>xG: %{y:.2f}<extra></extra>",
    ))
    fig_xg.add_trace(go.Bar(
        x=df_sorted["game_num"], y=-df_sorted["xga"],
        name="xG Sofrido", marker_color=XG_AG,
        hovertemplate="Jogo %{x}<br>xGA: %{customdata:.2f}<extra></extra>",
        customdata=df_sorted["xga"],
    ))
    fig_xg.add_hline(y=0, line_color="white", line_width=1, opacity=0.3)
    mean_xg = df_sorted["xg"].mean()
    fig_xg.add_hline(y=mean_xg, line_color=GOLD, line_width=1.5, line_dash="dot",
                     annotation_text=f"Média xG: {mean_xg:.2f}",
                     annotation_font_color=GOLD, annotation_position="top right")
    fig_xg.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, barmode="overlay", height=300,
        margin=dict(l=0, r=0, t=40, b=10),
        title=dict(text="xG Criado vs Sofrido por jogo", font=dict(color=GOLD, size=14), x=0),
        xaxis=dict(title="Jogo nº", tickfont=dict(color=MUTED), gridcolor=GRID),
        yaxis=dict(tickfont=dict(color=MUTED), gridcolor=GRID, tickformat=".1f",
                   title=dict(text="xG", font=dict(color=MUTED))),
        legend=dict(font=dict(color="white", size=11), bgcolor=CARD_BG, bordercolor=GRID),
    )
    st.plotly_chart(fig_xg, use_container_width=True)

    st.markdown(
        f'<p style="color:{MUTED};font-size:11px;text-align:right;">'
        f'Data: Wyscout | Analysis: Martim Jesus</p>',
        unsafe_allow_html=True,
    )
