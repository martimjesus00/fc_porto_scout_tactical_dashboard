
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

BG       = "#0d0d1a"
CARD_BG  = "#12122a"
GOLD     = "#FFD700"
GRID     = "#1e1e3a"
MUTED    = "#8888aa"

COLORS = {"Porto": "#0033A0", "Benfica": "#CC0000", "Sporting": "#008000"}
CONTEXT_COLORS = {
    "dominant":   "#2ecc71",
    "balanced":   "#f39c12",
    "challenged": "#e74c3c",
}

def _section(title):
    st.markdown(
        f'<div style="font-size:15px;font-weight:700;color:{GOLD};'
        f'margin:28px 0 12px 0;border-left:3px solid {GOLD};padding-left:10px;">'
        f'{title}</div>', unsafe_allow_html=True)

def _dot_plot(metrics, agg, agg_norm, clubs, title, invert=None):
    invert = invert or []
    fig = go.Figure()
    labels = list(metrics.keys())
    n = len(labels)

    for i, (label, col) in enumerate(metrics.items()):
        norm_vals = [agg_norm.loc[c, col] for c in clubs if c in agg_norm.index]
        fig.add_shape(type="line",
            x0=min(norm_vals), x1=max(norm_vals), y0=i, y1=i,
            line=dict(color="white", width=1.5, dash="dot"), opacity=0.15)

    for club in clubs:
        if club not in agg_norm.index:
            continue
        x_vals, y_vals, hovers = [], [], []
        for i, (label, col) in enumerate(metrics.items()):
            x = agg_norm.loc[club, col]
            raw = agg.loc[club, col]
            x_vals.append(x)
            y_vals.append(i)
            hovers.append(f"<b>{club}</b><br>{label}: {raw:.2f}")
            fig.add_annotation(x=x, y=i, text=f"{raw:.1f}", showarrow=False,
                font=dict(color=COLORS[club], size=9),
                xshift=14,
                yshift={"Porto": 12, "Benfica": -18, "Sporting": 0}.get(club, 8))

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="markers",
            marker=dict(size=18, color=COLORS[club], line=dict(color="white", width=1.5)),
            name=club, hovertext=hovers, hoverinfo="text"))

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        height=max(300, n * 75),
        margin=dict(l=10, r=40, t=40, b=10),
        title=dict(text=title, font=dict(color=GOLD, size=13), x=0),
        legend=dict(font=dict(color="white", size=11), bgcolor=CARD_BG, bordercolor=GRID),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.05, 1.35]),
        yaxis=dict(tickvals=list(range(n)), ticktext=list(metrics.keys()),
                   tickfont=dict(color="white", size=10), showgrid=False),
    )
    return fig

def render(porto, liga_big3):
    st.markdown(
        f'<h2 style="color:{GOLD};margin-bottom:4px;">Tactical Identity</h2>'
        f'<p style="color:{MUTED};margin-top:0;">Impressão digital táctica · Liga Portugal 2025/26</p>',
        unsafe_allow_html=True)

    liga = liga_big3.copy()
    if "recoveries_low_pct" not in liga.columns:
        liga["recoveries_low_pct"]  = liga["recoveries_low"]  / liga["recoveries_total"] * 100
    if "recoveries_mid_pct" not in liga.columns:
        liga["recoveries_mid_pct"]  = liga["recoveries_mid"]  / liga["recoveries_total"] * 100
    if "recoveries_high_pct" not in liga.columns:
        liga["recoveries_high_pct"] = liga["recoveries_high"] / liga["recoveries_total"] * 100

    clubs = ["Porto", "Benfica", "Sporting"]

    # ── 1. Tactical Fingerprint ───────────────────────────────
    _section("① Tactical Fingerprint — Big 3")
    fp_metrics = {
        "PPDA\n(lower = more pressing)": "ppda",
        "Possession %":                   "possession_pct",
        "Progressive\nPasses/90":        "progressive_passes",
        "xG Created /90":                 "xg",
        "Goals Conceded\n/game":         "goals_against",
        "Def. Third\nRecoveries %":      "recoveries_low_pct",
    }
    fp_agg  = liga.groupby("club")[list(fp_metrics.values())].mean().round(2)
    fp_norm = fp_agg.copy()
    for col in fp_agg.columns:
        mn, mx = fp_agg[col].min(), fp_agg[col].max()
        fp_norm[col] = (fp_agg[col] - mn) / (mx - mn) if mx > mn else 0.5
    for col in ["ppda", "goals_against"]:
        if col in fp_norm.columns:
            fp_norm[col] = 1 - fp_norm[col]
    st.plotly_chart(_dot_plot(fp_metrics, fp_agg, fp_norm, clubs,
        "Tactical Fingerprint — Liga Portugal avg"), use_container_width=True)

    # ── 2. PPDA por contexto ──────────────────────────────────
    _section("② PPDA por Contexto de Jogo (Porto)")
    context_order  = ["dominant", "balanced", "challenged"]
    context_labels = {"dominant": "Dominant (Liga, não-Big3)",
                      "balanced": "Balanced (Clássicos)",
                      "challenged": "Challenged (Europa League)"}
    ppda_ctx = (porto.groupby("opponent_context")["ppda"]
                .agg(["mean","std"]).reindex(context_order).reset_index())

    fig_ppda = go.Figure()
    for _, row in ppda_ctx.iterrows():
        ctx  = row["opponent_context"]
        mean = row["mean"]
        std  = row["std"] if not pd.isna(row["std"]) else 0
        fig_ppda.add_trace(go.Bar(
            x=[context_labels.get(ctx, ctx)], y=[mean],
            error_y=dict(type="data", array=[std], visible=True,
                         color="white", thickness=1.5, width=6),
            marker_color=CONTEXT_COLORS.get(ctx, MUTED),
            marker_line=dict(color="white", width=0.8),
            name=ctx.title(),
            text=[f"{mean:.1f}"], textposition="outside",
            textfont=dict(color="white", size=12, family="Arial Black"),
            hovertemplate=f"<b>{context_labels.get(ctx,ctx)}</b><br>PPDA: %{{y:.2f}}<extra></extra>",
        ))
    fig_ppda.add_hline(y=8, line_color=GOLD, line_width=1.5, line_dash="dot",
        annotation_text="Threshold pressing: 8.0",
        annotation_font_color=GOLD, annotation_position="top right")
    fig_ppda.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG, height=320, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="PPDA médio por contexto (menor = mais pressing)",
                   font=dict(color=GOLD, size=13), x=0),
        xaxis=dict(tickfont=dict(color="white", size=11), showgrid=False),
        yaxis=dict(tickfont=dict(color=MUTED), gridcolor=GRID,
                   title=dict(text="PPDA", font=dict(color=MUTED)),
                   range=[0, ppda_ctx["mean"].max() + 3]),
        bargap=0.4)
    st.plotly_chart(fig_ppda, use_container_width=True)

    # ── 3. Zonas de recuperação ───────────────────────────────
    _section("③ Zonas de Recuperação — Big 3")
    rec_metrics = {
        "Def. Third (baixo)": "recoveries_low_pct",
        "Mid Third (médio)":  "recoveries_mid_pct",
        "Att. Third (alto)":  "recoveries_high_pct",
    }
    rec_agg  = liga.groupby("club")[list(rec_metrics.values())].mean().round(2)
    rec_norm = rec_agg.copy()
    for col in rec_agg.columns:
        mn, mx = rec_agg[col].min(), rec_agg[col].max()
        rec_norm[col] = (rec_agg[col] - mn) / (mx - mn) if mx > mn else 0.5
    st.plotly_chart(_dot_plot(rec_metrics, rec_agg, rec_norm, clubs,
        "Distribuição das recuperações de bola por zona (%)"), use_container_width=True)

    # ── 4. Estilo de passe ────────────────────────────────────
    _section("④ Estilo de Passe — Big 3")
    pass_metrics = {
        "Passes frente /90":      "forward_passes",
        "Passes laterais /90":    "lateral_passes",
        "Passes trás /90":        "back_passes",
        "Passes longos /90":      "long_passes",
        "Passes progressivos /90":"progressive_passes",
        "Smart Passes /90":       "smart_passes",
    }
    available = {k: v for k, v in pass_metrics.items() if v in liga.columns}
    pass_agg  = liga.groupby("club")[list(available.values())].mean().round(1)
    pass_norm = pass_agg.copy()
    for col in pass_agg.columns:
        mn, mx = pass_agg[col].min(), pass_agg[col].max()
        pass_norm[col] = (pass_agg[col] - mn) / (mx - mn) if mx > mn else 0.5
    st.plotly_chart(_dot_plot(available, pass_agg, pass_norm, clubs,
        "Perfil de passe — Liga Portugal avg"), use_container_width=True)

    st.markdown(f'<p style="color:{MUTED};font-size:11px;text-align:right;">'
                f'Data: Wyscout | Analysis: Martim Jesus</p>', unsafe_allow_html=True)
