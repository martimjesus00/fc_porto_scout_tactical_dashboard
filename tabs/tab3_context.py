
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

BG      = "#0d0d1a"
CARD_BG = "#12122a"
GOLD    = "#FFD700"
GRID    = "#1e1e3a"
MUTED   = "#8888aa"
WIN_C   = "#2ecc71"
DRAW_C  = "#f39c12"
LOSS_C  = "#e74c3c"
COLORS  = {"Porto": "#0033A0", "Benfica": "#CC0000", "Sporting": "#008000"}
CONTEXT_ORDER  = ["dominant", "balanced", "challenged"]
CONTEXT_LABELS = {"dominant": "Dominant\n(Liga, não-Big3)", "balanced": "Balanced\n(Clássicos)", "challenged": "Challenged\n(EL)"}
CONTEXT_COLORS = {"dominant": "#2ecc71", "balanced": "#f39c12", "challenged": "#e74c3c"}

def _section(title):
    st.markdown(f'<div style="font-size:15px;font-weight:700;color:{GOLD};margin:28px 0 12px 0;border-left:3px solid {GOLD};padding-left:10px;">{title}</div>', unsafe_allow_html=True)

def _render_wdl_by_context(porto):
    rows = []
    for ctx in CONTEXT_ORDER:
        sub = porto[porto["opponent_context"] == ctx]
        n = len(sub)
        if n == 0: continue
        w = (sub["result"] == "Win").sum()
        d = (sub["result"] == "Draw").sum()
        l = (sub["result"] == "Loss").sum()
        rows.append({"context": ctx, "label": CONTEXT_LABELS[ctx], "n": n,
                     "W_pct": w/n*100, "D_pct": d/n*100, "L_pct": l/n*100,
                     "W": w, "D": d, "L": l})
    df = pd.DataFrame(rows)
    fig = go.Figure()
    for result, color, col in [("W", WIN_C, "W_pct"), ("D", DRAW_C, "D_pct"), ("L", LOSS_C, "L_pct")]:
        fig.add_trace(go.Bar(
            x=df["label"], y=df[col],
            name={"W":"Vitória","D":"Empate","L":"Derrota"}[result],
            marker_color=color, marker_line=dict(color=BG, width=1.5),
            text=[f"{v:.0f}%\n({r}j)" for v,r in zip(df[col], df[result])],
            textposition="inside", textfont=dict(color="white", size=11, family="Arial Black"),
            hovertemplate=f"<b>%{{x}}</b><br>{result}: %{{y:.1f}}%<extra></extra>",
        ))
    for _, row in df.iterrows():
        fig.add_annotation(x=row["label"], y=103, text=f"{row['n']} jogos",
            showarrow=False, font=dict(color=MUTED, size=10), yref="y")
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, barmode="stack", height=360,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Resultados por contexto de jogo (%)", font=dict(color=GOLD, size=13), x=0),
        xaxis=dict(tickfont=dict(color="white", size=11), showgrid=False),
        yaxis=dict(tickfont=dict(color=MUTED), gridcolor=GRID, range=[0,115],
                   title=dict(text="%", font=dict(color=MUTED))),
        legend=dict(font=dict(color="white",size=11), bgcolor=CARD_BG, bordercolor=GRID,
                    orientation="h", x=0, y=-0.15),
        bargap=0.35)
    return fig

def _render_xg_by_context(porto, liga_big3):
    porto_ctx = porto.groupby("opponent_context")[["xg","goals_for","goals_against"]].mean().round(2).reindex(CONTEXT_ORDER)
    liga_avg  = liga_big3.groupby("club")[["xg","goals_for","goals_against"]].mean().round(2)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["xG Criado /90","Golos Marcados /jogo"], horizontal_spacing=0.12)
    fig.add_trace(go.Bar(
        x=[CONTEXT_LABELS[c] for c in porto_ctx.index], y=porto_ctx["xg"],
        name="Porto (por contexto)", marker_color=[CONTEXT_COLORS[c] for c in porto_ctx.index],
        marker_line=dict(color="white", width=0.8),
        text=[f"{v:.2f}" for v in porto_ctx["xg"]], textposition="outside",
        textfont=dict(color="white", size=11),
        hovertemplate="Porto %{x}<br>xG: %{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=[CONTEXT_LABELS[c] for c in porto_ctx.index], y=porto_ctx["goals_for"],
        name="Porto (golos)", marker_color=[CONTEXT_COLORS[c] for c in porto_ctx.index],
        marker_line=dict(color="white", width=0.8),
        text=[f"{v:.2f}" for v in porto_ctx["goals_for"]], textposition="outside",
        textfont=dict(color="white", size=11),
        hovertemplate="Porto %{x}<br>Golos: %{y:.2f}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)
    for club, color in COLORS.items():
        if club not in liga_avg.index: continue
        for col_key, r, c in [("xg",1,1),("goals_for",1,2)]:
            val = liga_avg.loc[club, col_key]
            fig.add_hline(y=val, line_color=color, line_width=1.5, line_dash="dot", row=r, col=c,
                annotation_text=f"{club}: {val:.2f}", annotation_font_color=color,
                annotation_position="top right", annotation_font_size=9)
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        title=dict(text="xG e Golos por contexto — Porto (linhas = médias Liga Big 3)",
                   font=dict(color=GOLD, size=13), x=0),
        legend=dict(font=dict(color="white",size=10), bgcolor=CARD_BG, bordercolor=GRID),
        bargap=0.35)
    for ax in ["xaxis","xaxis2"]:
        fig.update_layout(**{ax: dict(tickfont=dict(color="white",size=10), showgrid=False)})
    for ax in ["yaxis","yaxis2"]:
        fig.update_layout(**{ax: dict(tickfont=dict(color=MUTED), gridcolor=GRID)})
    for ann in fig.layout.annotations:
        ann.font.color = MUTED; ann.font.size = 11
    return fig

def _render_classicos(porto):
    classicos = porto[porto["opponent_context"] == "balanced"].copy()
    if classicos.empty:
        st.info("Sem dados de clássicos."); return
    classicos["opponent"] = classicos["match"].apply(
        lambda m: "Benfica" if "Benfica" in str(m) else "Sporting" if "Sporting CP" in str(m) else "Outro")
    classicos["hover"] = classicos.apply(
        lambda r: (f"<b>{r['match']}</b><br>Data: {str(r['date'])[:10]}<br>"
                   f"xG: {r['xg']:.2f} | xGA: {r['xga']:.2f}<br>"
                   f"Resultado: {r['result']} {r['score']}<br>"
                   f"PPDA: {r['ppda']:.1f} | Posse: {r['possession_pct']:.1f}%"), axis=1)
    fig = go.Figure()
    max_val = max(classicos["xg"].max(), classicos["xga"].max()) + 0.3
    fig.add_shape(type="line", x0=0, x1=max_val, y0=0, y1=max_val,
                  line=dict(color="white", width=1, dash="dot"), opacity=0.2)
    fig.add_annotation(x=max_val*0.85, y=max_val*0.92, text="Linha de equilíbrio",
                       font=dict(color=MUTED, size=9), showarrow=False)
    for opp, color in [("Benfica", COLORS["Benfica"]), ("Sporting", COLORS["Sporting"])]:
        sub = classicos[classicos["opponent"] == opp]
        if sub.empty: continue
        for result, symbol in [("Win","circle"),("Draw","diamond"),("Loss","x")]:
            rs = sub[sub["result"] == result]
            if rs.empty: continue
            fig.add_trace(go.Scatter(
                x=rs["xg"], y=rs["xga"], mode="markers",
                marker=dict(size=16, color=color, symbol=symbol,
                            line=dict(color="white", width=1.5)),
                name=f"{opp} — {result}",
                hovertext=rs["hover"], hoverinfo="text"))
        mx, my = sub["xg"].mean(), sub["xga"].mean()
        fig.add_trace(go.Scatter(x=[mx], y=[my], mode="markers",
            marker=dict(size=22, color=color, symbol="star",
                        line=dict(color=GOLD, width=2)),
            name=f"{opp} — média",
            hovertemplate=f"<b>{opp} média</b><br>xG: {mx:.2f}<br>xGA: {my:.2f}<extra></extra>"))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, height=400,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Clássicos — xG Criado vs xG Sofrido (símbolo = resultado)",
                   font=dict(color=GOLD, size=13), x=0),
        xaxis=dict(title=dict(text="xG Criado", font=dict(color=MUTED)),
                   tickfont=dict(color=MUTED), gridcolor=GRID, range=[0, max_val]),
        yaxis=dict(title=dict(text="xG Sofrido", font=dict(color=MUTED)),
                   tickfont=dict(color=MUTED), gridcolor=GRID, range=[0, max_val]),
        legend=dict(font=dict(color="white",size=10), bgcolor=CARD_BG, bordercolor=GRID))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f'<div style="font-size:13px;color:{GOLD};font-weight:600;margin:16px 0 8px 0;">Resumo dos Clássicos</div>', unsafe_allow_html=True)
    summary = (classicos[classicos["opponent"] != "Outro"].groupby("opponent").agg(
        Jogos=("result","count"),
        Vitórias=("result", lambda x: (x=="Win").sum()),
        Empates=("result", lambda x: (x=="Draw").sum()),
        Derrotas=("result", lambda x: (x=="Loss").sum()),
        xG_médio=("xg","mean"), xGA_médio=("xga","mean"),
        PPDA_médio=("ppda","mean"), Posse_média=("possession_pct","mean"),
    ).round(2).reset_index())
    st.dataframe(summary, use_container_width=True, hide_index=True)

def render(porto, liga_big3):
    st.markdown(f'<h2 style="color:{GOLD};margin-bottom:4px;">Context Analysis</h2><p style="color:{MUTED};margin-top:0;">Performance do Porto por contexto de jogo</p>', unsafe_allow_html=True)
    if "xga" not in porto.columns:
        porto = porto.copy()
        porto["xga"] = porto["goals_conceded"] if "goals_conceded" in porto.columns else porto["goals_against"]
    _section("① Resultados por Contexto")
    st.plotly_chart(_render_wdl_by_context(porto), use_container_width=True)
    _section("② xG e Golos por Contexto — Porto vs Big 3")
    st.plotly_chart(_render_xg_by_context(porto, liga_big3), use_container_width=True)
    _section("③ Clássicos Isolados — Benfica & Sporting")
    _render_classicos(porto)
    st.markdown(f'<p style="color:{MUTED};font-size:11px;text-align:right;">Data: Wyscout | Analysis: Martim Jesus</p>', unsafe_allow_html=True)
