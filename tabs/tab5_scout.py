
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

BG="$0d0d1a";CARD_BG="$12122a";GOLD="$FFD700";GRID="$1e1e3a";MUTED="$8888aa";PORTO="$0033A0"
BG=BG.replace("$","#");CARD_BG=CARD_BG.replace("$","#");GOLD=GOLD.replace("$","#")
GRID=GRID.replace("$","#");MUTED=MUTED.replace("$","#");PORTO=PORTO.replace("$","#")
SIM_COLORS=["#FFD700","#c0c0c0","#cd7f32","#4a90d9","#7b68ee"]

SIM_METRICS={
    "Goalkeeper":["save_pct","goals_prevented_p90","sweeper_actions_p90","aerial_duels_won_pct","xgc_p90"],
    "Defender":["def_actions_p90","def_duels_won_pct","aerial_duels_won_pct","interceptions_p90","progressive_passes_p90","long_passes_accurate_pct","passes_accurate_pct"],
    "Midfielder":["progressive_passes_p90","key_passes_p90","passes_accurate_pct","def_actions_p90","dribbles_p90","xa_p90","fwd_passes_p90"],
    "Forward":["goals_p90","xg_p90","shots_p90","shots_on_target_pct","dribbles_p90","xa_p90","box_touches_p90"],
}
RADAR_LABELS={
    "Goalkeeper":["Save %","Goals Prev/90","Sweeper/90","Aerial Won%","xGC/90"],
    "Defender":["Def Act/90","Def Duel%","Aerial%","Interc/90","Prog Pass/90","Long Pass%","Pass Acc%"],
    "Midfielder":["Prog Pass/90","Key Pass/90","Pass Acc%","Def Act/90","Dribbles/90","xA/90","Fwd Pass/90"],
    "Forward":["Goals/90","xG/90","Shots/90","Shot Tgt%","Dribbles/90","xA/90","Box Touch/90"],
}

    def _pos(p): return {"goalkeeper":"Goalkeeper","defender":"Defender","midfielder":"Midfielder","forward":"Forward"}.get(str(p).lower(),"Midfielder")
    def _hex_rgb(h): h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))

    def _compute_similarity(porto_row, scout_df, pos_group, top_n=5):
        metrics=[m for m in SIM_METRICS.get(pos_group,[]) if m in porto_row.index and m in scout_df.columns]
        if not metrics: return pd.DataFrame()
        porto_positions = [p.strip() for p in str(porto_row.get("position","")).split(",")]

    def has_position_overlap(pos_str):
        scout_positions = [p.strip() for p in str(pos_str).split(",")]
        return any(p in scout_positions for p in porto_positions)

    pool = scout_df[scout_df["position_group"].str.lower()==pos_group.lower()].copy()
    pool = pool[pool["position"].apply(has_position_overlap)]

    if len(pool) < 10:
        pool = scout_df[scout_df["position_group"].str.lower()==pos_group.lower()].copy()

    pool = pool.dropna(subset=metrics, thresh=len(metrics)//2)
        if pool.empty: return pd.DataFrame()
        pv=pd.DataFrame([porto_row[metrics].values],columns=metrics)
        all_v=pd.concat([pv,pool[metrics]],ignore_index=True)
        scaled=MinMaxScaler().fit_transform(all_v.fillna(0))
        sims=cosine_similarity(scaled[0:1],scaled[1:])[0]
        pool=pool.copy(); pool["similarity"]=sims; pool["similarity_pct"]=(sims*100).round(1)
        top=pool.nlargest(top_n,"similarity").reset_index(drop=True)
        top["rank"]=range(1,len(top)+1)
        return top

def _radar(porto_row, similars, pos_group, porto_name):
    metrics=SIM_METRICS.get(pos_group,[]); labels=RADAR_LABELS.get(pos_group,metrics)
    am=[m for m in metrics if m in porto_row.index and m in similars.columns]
    al=[labels[i] for i,m in enumerate(metrics) if m in am]
    if not am: return go.Figure()
    all_r=np.array([porto_row[am].values.tolist()]+[similars.iloc[i][am].values.tolist() for i in range(len(similars))],dtype=float)
    mn=np.nanmin(all_r,axis=0); mx=np.nanmax(all_r,axis=0); rng=np.where(mx>mn,mx-mn,1)
    norm=(all_r-mn)/rng; lc=al+[al[0]]; fig=go.Figure()
    for i,row in similars.iterrows():
        v=norm[i+1].tolist(); vc=v+[v[0]]
        r,g,b=_hex_rgb(SIM_COLORS[i])
        fig.add_trace(go.Scatterpolar(r=vc,theta=lc,fill="toself",
            fillcolor=f"rgba({r},{g},{b},0.08)",
            line=dict(color=SIM_COLORS[i],width=1.5,dash="dot"),
            name=f"#{i+1} {row.get('player','?')}",
            hovertemplate=f"<b>{row.get('player','?')}</b> ({row.get('team','?')})<br>Sim: {row.get('similarity_pct',0):.1f}%<extra></extra>"))
    pv=norm[0].tolist(); pvc=pv+[pv[0]]
    fig.add_trace(go.Scatterpolar(r=pvc,theta=lc,fill="toself",
        fillcolor="rgba(0,51,160,0.3)",line=dict(color=PORTO,width=3),name=porto_name))
    fig.update_layout(
        polar=dict(bgcolor=CARD_BG,radialaxis=dict(visible=False,range=[0,1]),
            angularaxis=dict(tickfont=dict(color="white",size=10),linecolor=GRID,gridcolor=GRID)),
        paper_bgcolor=BG,height=450,margin=dict(l=60,r=60,t=50,b=40),
        legend=dict(font=dict(color="white",size=10),bgcolor=CARD_BG,bordercolor=GRID,orientation="h",x=0,y=-0.15),
        title=dict(text=f"Comparação — {porto_name} vs Top 5 Similares",font=dict(color=GOLD,size=13),x=0.5,xanchor="center"))
    return fig

def _cards(similars):
    cols=st.columns(len(similars))
    for i,(col,(_,row)) in enumerate(zip(cols,similars.iterrows())):
        with col:
            st.markdown(f'<div style="background:{CARD_BG};border:1px solid {SIM_COLORS[i]};border-radius:10px;padding:14px 12px;text-align:center;"><div style="font-size:20px;font-weight:700;color:{SIM_COLORS[i]};">#{i+1}</div><div style="font-size:13px;font-weight:600;color:white;margin:4px 0;">{row.get("player","?")}</div><div style="font-size:11px;color:{MUTED};">{row.get("team","?")}</div><div style="font-size:11px;color:{MUTED};">{row.get("position","?")} · {int(row.get("age",0)) if not pd.isna(row.get("age",0)) else "?"} anos</div><div style="font-size:11px;color:{MUTED};">{int(row.get("minutes",0)) if not pd.isna(row.get("minutes",0)) else "?"} min</div><div style="font-size:18px;font-weight:700;color:{SIM_COLORS[i]};margin-top:8px;">{row.get("similarity_pct",0):.1f}%</div><div style="font-size:10px;color:{MUTED};">similaridade</div></div>',unsafe_allow_html=True)

def _table(porto_row, similars, pos_group, porto_name):
    metrics=SIM_METRICS.get(pos_group,[]); labels=RADAR_LABELS.get(pos_group,metrics)
    am=[m for m in metrics if m in porto_row.index and m in similars.columns]
    al=[labels[i] for i,m in enumerate(metrics) if m in am]
    rows={"Métrica":al, porto_name:[round(float(porto_row[m]),2) if not pd.isna(porto_row[m]) else "—" for m in am]}
    for _,sr in similars.iterrows():
        rows[f"{sr.get('player','?')} ({sr.get('team','?')})" ]=[round(float(sr[m]),2) if not pd.isna(sr[m]) else "—" for m in am]
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)


def _radar_single(porto_row, sim_row, pos_group, porto_name, sim_color, scout_pool=None):
    """Radar limpo — 1 jogador Porto vs 1 similar, normalizado contra o pool."""
    metrics=SIM_METRICS.get(pos_group,[]); labels=RADAR_LABELS.get(pos_group,metrics)
    am=[m for m in metrics if m in porto_row.index and m in sim_row.index]
    al=[labels[i] for i,m in enumerate(metrics) if m in am]
    if not am: return go.Figure()
    pv=[float(porto_row[m]) if not pd.isna(porto_row[m]) else 0 for m in am]
    sv=[float(sim_row[m]) if not pd.isna(sim_row[m]) else 0 for m in am]
    # Normalizar contra o pool completo para contexto real
    if scout_pool is not None and len(scout_pool) > 2:
        pool_vals = scout_pool[am].fillna(0).values
        mn=np.nanmin(pool_vals,axis=0); mx=np.nanmax(pool_vals,axis=0)
    else:
        all_v=np.array([pv,sv],dtype=float)
        mn=np.nanmin(all_v,axis=0); mx=np.nanmax(all_v,axis=0)
    rng=np.where(mx>mn,mx-mn,1)
    norm=np.array([(np.array(pv)-mn)/rng, (np.array(sv)-mn)/rng])
    norm=np.clip(norm,0,1)
    lc=al+[al[0]]
    pvc=norm[0].tolist()+[norm[0][0]]
    svc=norm[1].tolist()+[norm[1][0]]
    fig=go.Figure()
    r,g,b=_hex_rgb(sim_color)
    fig.add_trace(go.Scatterpolar(r=svc,theta=lc,fill="toself",
        fillcolor=f"rgba({r},{g},{b},0.15)",
        line=dict(color=sim_color,width=2.5),
        name=f"{sim_row.get('player','?')} ({sim_row.get('team','?')})",
        hovertemplate=[f"<b>{l}</b><br>{sim_row.get('player','?')}: {round(float(sim_row[m]),2) if m in sim_row.index and not pd.isna(sim_row[m]) else '—'}<extra></extra>" for l,m in zip(lc,[*am,am[0]])]))
    fig.add_trace(go.Scatterpolar(r=pvc,theta=lc,fill="toself",
        fillcolor="rgba(0,51,160,0.25)",
        line=dict(color=PORTO,width=2.5),
        name=porto_name,
        hovertemplate=[f"<b>{l}</b><br>{porto_name}: {round(float(porto_row[m]),2) if m in porto_row.index and not pd.isna(porto_row[m]) else '—'}<extra></extra>" for l,m in zip(lc,[*am,am[0]])]))
    fig.update_layout(
        polar=dict(bgcolor=CARD_BG,
            radialaxis=dict(visible=False,range=[0,1]),
            angularaxis=dict(tickfont=dict(color="white",size=11),linecolor=GRID,gridcolor=GRID)),
        paper_bgcolor=BG,height=450,
        margin=dict(l=60,r=60,t=50,b=40),
        legend=dict(font=dict(color="white",size=12),bgcolor=CARD_BG,bordercolor=GRID,orientation="h",x=0.2,y=-0.1),
        title=dict(text=f"{porto_name}  vs  {sim_row.get('player','?')}",font=dict(color=GOLD,size=14),x=0.5,xanchor="center"))
    return fig

def render(porto_players, scout_players):
    st.markdown(f'<h2 style="color:{GOLD};margin-bottom:4px;">Similarity Scout</h2><p style="color:{MUTED};margin-top:0;">Top 5 jogadores similares · {len(scout_players)} jogadores analisados</p>',unsafe_allow_html=True)
    if porto_players.empty or scout_players.empty: st.warning("Dados insuficientes."); return
    ps=porto_players.copy()
    ps["_ord"]=ps["position_group"].map({"Goalkeeper":0,"Defender":1,"Midfielder":2,"Forward":3}).fillna(9)
    ps=ps.sort_values(["_ord","player"]).reset_index(drop=True)
    ps_min=ps[ps["minutes"]>=200] if "minutes" in ps.columns else ps
    col_sel,col_filter=st.columns([3,3])
    with col_sel: selected=st.selectbox("Jogador do Porto",ps_min["player"].tolist(),key="tab5s_player")
    with col_filter: min_min=st.slider("Minutos mínimos (scout)",300,1500,500,100,key="tab5s_min")
    porto_row=ps_min[ps_min["player"]==selected].iloc[0]
    pos_group=_pos(porto_row.get("position_group","Midfielder"))
    scout_f=scout_players[scout_players["minutes"]>=min_min].copy()
    st.markdown(f'<p style="color:{MUTED};font-size:12px;margin-bottom:16px;">{selected} · {porto_row.get("position","?")} · {pos_group} · {int(porto_row.get("minutes",0))} min</p>',unsafe_allow_html=True)
    with st.spinner("A calcular similaridade..."):
        similars=_compute_similarity(porto_row,scout_f,pos_group)
    if similars.empty: st.warning("Sem jogadores suficientes."); return
    st.markdown(f'<div style="font-size:14px;font-weight:600;color:{GOLD};margin:16px 0 10px 0;">Top 5 Similares — clica num jogador para comparar</div>',unsafe_allow_html=True)
    
    # Cards clicáveis com botões
    cols = st.columns(len(similars))
    selected_sim = None
    for i, (col, (_, row)) in enumerate(zip(cols, similars.iterrows())):
        with col:
            st.markdown(f'''<div style="background:{CARD_BG};border:1px solid {SIM_COLORS[i]};border-radius:10px;padding:14px 12px;text-align:center;"><div style="font-size:20px;font-weight:700;color:{SIM_COLORS[i]};">#{i+1}</div><div style="font-size:13px;font-weight:600;color:white;margin:4px 0;">{row.get("player","?")}</div><div style="font-size:11px;color:{MUTED};">{row.get("team","?")}</div><div style="font-size:11px;color:{MUTED};">{row.get("position","?")} · {int(row.get("age",0)) if not pd.isna(row.get("age",0)) else "?"} anos</div><div style="font-size:11px;color:{MUTED};">{int(row.get("minutes",0)) if not pd.isna(row.get("minutes",0)) else "?"} min</div><div style="font-size:18px;font-weight:700;color:{SIM_COLORS[i]};margin-top:8px;">{row.get("similarity_pct",0):.1f}%</div><div style="font-size:10px;color:{MUTED};">similaridade</div></div>''', unsafe_allow_html=True)
            if st.button(f"Ver radar", key=f"sim_btn_{i}"):
                st.session_state["selected_sim_idx"] = i

    # Radar do jogador seleccionado
    sim_idx = st.session_state.get("selected_sim_idx", 0)
    sim_row = similars.iloc[sim_idx]
    sim_color = SIM_COLORS[sim_idx]
    
    st.markdown(f'<div style="font-size:13px;color:{MUTED};margin:16px 0 8px 0;">A comparar com: <b style="color:{sim_color};">{sim_row.get("player","?")} ({sim_row.get("team","?")})</b></div>', unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    st.plotly_chart(_radar_single(porto_row, sim_row, pos_group, selected, sim_color, scout_pool=scout_f[scout_f['position_group'].str.lower()==pos_group.lower()]),use_container_width=True)
    st.markdown(f'<div style="font-size:14px;font-weight:600;color:{GOLD};margin:16px 0 10px 0;">Comparação de Métricas</div>',unsafe_allow_html=True)
    _table(porto_row,similars,pos_group,selected)
    st.markdown(f'<p style="color:{MUTED};font-size:11px;text-align:right;">Data: Wyscout | Analysis: Martim Jesus</p>',unsafe_allow_html=True)
