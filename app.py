"""
CNN HyperTuner — Streamlit Visual Dashboard
Animated comparison of GA · PSO · SA for CNN hyperparameter optimisation
Run: streamlit run cnn_hypertuner_app.py
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import math

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CNN HyperTuner",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #0e1117; }

  .metric-card {
    background: linear-gradient(135deg, #1a1d27 0%, #141720 100%);
    border: 1px solid #2d3142;
    border-radius: 12px;
    padding: 18px 20px;
    text-align: center;
    transition: border-color .3s;
  }
  .metric-card:hover { border-color: #4f46e5; }
  .metric-label { font-size: 11px; color: #6b7280; letter-spacing: .07em; text-transform: uppercase; margin-bottom: 6px; }
  .metric-value { font-size: 28px; font-weight: 600; line-height: 1.1; }
  .metric-sub   { font-size: 11px; color: #6b7280; margin-top: 4px; }

  .algo-card {
    background: #141720;
    border: 1px solid #2d3142;
    border-radius: 14px;
    padding: 20px;
    height: 100%;
    transition: all .25s;
  }
  .algo-card.winner { border-color: #4f46e5; box-shadow: 0 0 24px rgba(79,70,229,.15); }
  .algo-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: .05em;
    margin-bottom: 10px;
  }
  .badge-ga  { background:#451a03; color:#fb923c; }
  .badge-pso { background:#1e1b4b; color:#818cf8; }
  .badge-sa  { background:#052e16; color:#4ade80; }

  .winner-tag {
    background: #4f46e5;
    color: white;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 8px;
    vertical-align: middle;
  }

  .cfg-chip {
    display: inline-block;
    background: #1e2130;
    border: 1px solid #2d3142;
    border-radius: 8px;
    padding: 6px 12px;
    margin: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
  }
  .cfg-key   { color: #6b7280; font-size: 10px; display: block; margin-bottom: 2px; }
  .cfg-val   { color: #e2e8f0; font-weight: 500; }

  .section-title {
    font-size: 11px;
    font-weight: 600;
    color: #4b5563;
    letter-spacing: .1em;
    text-transform: uppercase;
    margin-bottom: 14px;
    padding-bottom: 6px;
    border-bottom: 1px solid #1f2333;
  }

  .stProgress > div > div { background: #4f46e5; }

  div[data-testid="stMetricValue"] { font-size: 28px !important; }

  .log-entry { font-family: 'JetBrains Mono', monospace; font-size: 11px; padding: 2px 0; }
  .log-info    { color: #60a5fa; }
  .log-success { color: #4ade80; }
  .log-warn    { color: #fb923c; }
  .log-muted   { color: #4b5563; }

  .stSidebar { background: #0e1117; border-right: 1px solid #1f2333; }

  .run-status {
    background: #0f172a;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
  }
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────────────────
COLORS   = {"ga": "#fb923c", "pso": "#818cf8", "sa": "#4ade80"}
FILLS    = {"ga": "rgba(251,146,60,.15)", "pso": "rgba(129,140,248,.15)", "sa": "rgba(74,222,128,.15)"}
BG_COLOR = "#0e1117"
GRID_COL = "#1f2333"
TICK_COL = "#4b5563"

FINAL_SCORES = {
    "cifar10": {"ga": 0.8742, "pso": 0.8813, "sa": 0.8698},
    "mnist":   {"ga": 0.9834, "pso": 0.9867, "sa": 0.9801},
}
CONV_SPEED = {
    "cifar10": {"ga": 18, "pso": 14, "sa": 22},
    "mnist":   {"ga": 9,  "pso": 7,  "sa": 12},
}
BEST_CONFIGS = {
    "ga": {
        "learning_rate": "2.3e-3", "batch_size": "64",
        "num_filters_1": "96",     "num_filters_2": "192",
        "num_filters_3": "384",    "kernel_size": "3",
        "dropout_rate":  "0.28",   "dense_units": "512",
        "optimizer":     "adamw",  "weight_decay": "4.1e-4",
        "activation":    "gelu",   "num_conv_layers": "4",
    },
    "pso": {
        "learning_rate": "1.8e-3", "batch_size": "128",
        "num_filters_1": "80",     "num_filters_2": "160",
        "num_filters_3": "320",    "kernel_size": "3",
        "dropout_rate":  "0.31",   "dense_units": "640",
        "optimizer":     "adam",   "weight_decay": "2.3e-4",
        "activation":    "leaky_relu", "num_conv_layers": "3",
    },
    "sa": {
        "learning_rate": "3.1e-3", "batch_size": "64",
        "num_filters_1": "112",    "num_filters_2": "224",
        "num_filters_3": "448",    "kernel_size": "5",
        "dropout_rate":  "0.25",   "dense_units": "448",
        "optimizer":     "adamw",  "weight_decay": "5.7e-5",
        "activation":    "gelu",   "num_conv_layers": "4",
    },
}

ALGO_LABELS  = {"ga": "Genetic Algorithm", "pso": "Particle Swarm", "sa": "Simulated Annealing"}
ALGO_DESC    = {
    "ga":  "BLX-α crossover · tournament selection · adaptive σ · elitism · stagnation restart",
    "pso": "Inertia decay · ring/global topology · velocity clamping · cognitive + social · reheat",
    "sa":  "Metropolis criterion · Cauchy/Gaussian perturbation · geometric cooling · adaptive reheat",
}

RADAR_LABELS  = ["Accuracy", "Speed", "Stability", "Diversity", "Exploration", "Convergence"]
RADAR_SCORES  = {
    "ga":  [85, 70, 80, 90, 75, 78],
    "pso": [88, 85, 82, 78, 80, 88],
    "sa":  [83, 65, 88, 65, 70, 72],
}

PARAM_IMPORTANCE = {
    "learning_rate": 31, "optimizer": 22, "num_filters": 18,
    "dropout_rate":  12, "batch_size": 8, "dense_units": 5,
    "activation":    3,  "kernel_size": 1,
}


# ── Simulation helpers ─────────────────────────────────────────────────────────
def sim_curve(algo: str, t: int, total: int, dataset: str, rng: np.random.Generator) -> float:
    final   = FINAL_SCORES[dataset][algo]
    speeds  = {"ga": 4.5, "pso": 5.8, "sa": 3.6}
    noises  = {"ga": 0.025, "pso": 0.020, "sa": 0.030}
    p       = t / total
    base    = final * (1 - math.exp(-speeds[algo] * p))
    noise   = (rng.random() - 0.5) * noises[algo] * (1 - p)
    return float(np.clip(base + noise, 0.01, 0.999))


def build_convergence_fig(curves: dict, iters_done: int) -> go.Figure:
    fig = go.Figure()
    labels = list(range(1, iters_done + 1))
    for algo in ["ga", "pso", "sa"]:
        if not curves[algo]:
            continue
        fig.add_trace(go.Scatter(
            x=labels, y=curves[algo],
            name=ALGO_LABELS[algo],
            line=dict(color=COLORS[algo], width=2.5),
            fill="tozeroy",
            fillcolor=FILLS[algo],
            mode="lines",
            hovertemplate=f"<b>{ALGO_LABELS[algo]}</b><br>Iter: %{{x}}<br>Acc: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        margin=dict(l=0, r=0, t=4, b=0), height=260,
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11, color="#9ca3af"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Iteration", color=TICK_COL, gridcolor=GRID_COL, zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(title="Val accuracy", color=TICK_COL, gridcolor=GRID_COL, zeroline=False,
                   tickformat=".0%", range=[0, 1.0], tickfont=dict(size=10)),
        hovermode="x unified",
    )
    return fig


def build_dist_fig(all_scores: dict) -> go.Figure:
    fig = go.Figure()
    for algo in ["ga", "pso", "sa"]:
        if not all_scores[algo]:
            continue
        fig.add_trace(go.Histogram(
            x=all_scores[algo], name=ALGO_LABELS[algo],
            nbinsx=12,
            marker_color=COLORS[algo], opacity=0.75,
            hovertemplate=f"{ALGO_LABELS[algo]}<br>Acc: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        margin=dict(l=0, r=0, t=4, b=0), height=220, barmode="overlay",
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10, color="#9ca3af"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Score", color=TICK_COL, gridcolor=GRID_COL, zeroline=False, tickformat=".2f", tickfont=dict(size=10)),
        yaxis=dict(title="Count", color=TICK_COL, gridcolor=GRID_COL, zeroline=False, tickfont=dict(size=10)),
    )
    return fig


def build_radar_fig() -> go.Figure:
    fig = go.Figure()
    cats = RADAR_LABELS + [RADAR_LABELS[0]]
    for algo in ["ga", "pso", "sa"]:
        vals = RADAR_SCORES[algo] + [RADAR_SCORES[algo][0]]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=cats, name=ALGO_LABELS[algo],
            line=dict(color=COLORS[algo], width=2),
            fill="toself", fillcolor=FILLS[algo],
            hovertemplate=f"<b>{ALGO_LABELS[algo]}</b><br>%{{theta}}: %{{r}}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=BG_COLOR,
        polar=dict(
            bgcolor=BG_COLOR,
            radialaxis=dict(visible=True, range=[0, 100], color=TICK_COL, gridcolor=GRID_COL, tickfont=dict(size=8)),
            angularaxis=dict(color="#9ca3af", gridcolor=GRID_COL, tickfont=dict(size=10)),
        ),
        legend=dict(orientation="h", y=-0.08, x=0.1, font=dict(size=10, color="#9ca3af"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=10, b=20), height=260,
    )
    return fig


def build_importance_fig() -> go.Figure:
    items  = sorted(PARAM_IMPORTANCE.items(), key=lambda x: x[1])
    params = [k for k, _ in items]
    vals   = [v for _, v in items]
    palette = ["#312e81","#3730a3","#4338ca","#4f46e5","#6366f1","#818cf8","#a5b4fc","#c7d2fe"]
    fig = go.Figure(go.Bar(
        x=vals, y=params, orientation="h",
        marker=dict(color=palette, cornerradius=4),
        text=[f"{v}%" for v in vals], textposition="outside",
        textfont=dict(size=10, color="#9ca3af"),
        hovertemplate="%{y}: %{x}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        margin=dict(l=0, r=40, t=4, b=0), height=240,
        xaxis=dict(color=TICK_COL, gridcolor=GRID_COL, zeroline=False,
                   ticksuffix="%", tickfont=dict(size=10), range=[0, 40]),
        yaxis=dict(color="#9ca3af", gridcolor=GRID_COL, tickfont=dict(size=10), zeroline=False),
    )
    return fig


def build_scatter_fig(all_scores: dict, iters: int) -> go.Figure:
    fig = go.Figure()
    for algo in ["ga", "pso", "sa"]:
        if not all_scores[algo]:
            continue
        n = len(all_scores[algo])
        xs = list(range(1, n + 1))
        fig.add_trace(go.Scatter(
            x=xs, y=all_scores[algo],
            name=ALGO_LABELS[algo],
            mode="markers",
            marker=dict(color=COLORS[algo], size=5, opacity=0.6),
            hovertemplate=f"{ALGO_LABELS[algo]}<br>Iter %{{x}}: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        margin=dict(l=0, r=0, t=4, b=0), height=220,
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10, color="#9ca3af"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(title="Iteration", color=TICK_COL, gridcolor=GRID_COL, zeroline=False, tickfont=dict(size=10)),
        yaxis=dict(title="Score", color=TICK_COL, gridcolor=GRID_COL, zeroline=False,
                   tickformat=".2f", tickfont=dict(size=10)),
    )
    return fig


def build_heatmap_fig(all_scores: dict, iters: int) -> go.Figure:
    algos = ["ga", "pso", "sa"]
    n = min(iters, min((len(all_scores[a]) for a in algos if all_scores[a]), default=1))
    if n == 0:
        n = 1
    matrix = np.array([[all_scores[a][i] if i < len(all_scores[a]) else 0 for i in range(n)] for a in algos])
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=list(range(1, n + 1)),
        y=[ALGO_LABELS[a] for a in algos],
        colorscale=[[0, "#0f172a"], [0.4, "#312e81"], [0.7, "#7c3aed"], [1, "#c4b5fd"]],
        showscale=True,
        colorbar=dict(tickfont=dict(size=9, color=TICK_COL), thickness=10, tickformat=".2f"),
        hovertemplate="Iter %{x}<br>%{y}: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        margin=dict(l=0, r=0, t=4, b=0), height=160,
        xaxis=dict(title="Iteration", color=TICK_COL, tickfont=dict(size=9), zeroline=False),
        yaxis=dict(color="#9ca3af", tickfont=dict(size=10), zeroline=False),
    )
    return fig


def build_violin_fig(all_scores: dict) -> go.Figure:
    fig = go.Figure()
    for algo in ["ga", "pso", "sa"]:
        if len(all_scores[algo]) < 4:
            continue
        fig.add_trace(go.Violin(
            y=all_scores[algo], name=ALGO_LABELS[algo],
            box_visible=True, meanline_visible=True,
            fillcolor=FILLS[algo], line_color=COLORS[algo],
            opacity=0.85,
            hovertemplate=f"{ALGO_LABELS[algo]}<br>Score: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        margin=dict(l=0, r=0, t=4, b=0), height=220,
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10, color="#9ca3af"), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(color=TICK_COL, gridcolor=GRID_COL, tickfont=dict(size=10)),
        yaxis=dict(title="Score", color=TICK_COL, gridcolor=GRID_COL, zeroline=False,
                   tickformat=".2f", tickfont=dict(size=10)),
    )
    return fig


def build_final_bar_fig(final_scores: dict) -> go.Figure:
    algos = ["ga", "pso", "sa"]
    names = [ALGO_LABELS[a] for a in algos]
    vals  = [final_scores[a] for a in algos]
    cols  = [COLORS[a] for a in algos]
    fig = go.Figure(go.Bar(
        x=names, y=vals,
        marker=dict(color=cols, cornerradius=6),
        text=[f"{v*100:.2f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=12, color="#e2e8f0"),
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=BG_COLOR, plot_bgcolor=BG_COLOR,
        margin=dict(l=0, r=0, t=4, b=0), height=220,
        xaxis=dict(color="#9ca3af", tickfont=dict(size=11), zeroline=False),
        yaxis=dict(color=TICK_COL, gridcolor=GRID_COL, zeroline=False,
                   tickformat=".0%", range=[0, max(vals)*1.15], tickfont=dict(size=10)),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ CNN HyperTuner")
    st.markdown("<div style='color:#4b5563;font-size:12px;margin-bottom:20px'>Soft computing · Visual dashboard</div>", unsafe_allow_html=True)

    dataset  = st.selectbox("Dataset", ["CIFAR-10", "MNIST"], index=0)
    ds_key   = dataset.lower().replace("-", "")

    n_iters  = st.slider("Iterations", 10, 100, 40, step=5)
    speed_ms = st.slider("Animation speed (ms/frame)", 20, 200, 60, step=10)

    st.markdown("---")
    st.markdown("<div class='section-title'>Algorithm toggles</div>", unsafe_allow_html=True)
    show_ga  = st.checkbox("Genetic Algorithm (GA)", value=True)
    show_pso = st.checkbox("Particle Swarm (PSO)",   value=True)
    show_sa  = st.checkbox("Simulated Annealing (SA)", value=True)
    active_algos = [a for a, v in [("ga", show_ga), ("pso", show_pso), ("sa", show_sa)] if v]

    st.markdown("---")
    run_btn = st.button("▶  Run Comparison", use_container_width=True, type="primary")
    reset_btn = st.button("↺  Reset", use_container_width=True)

    st.markdown("---")
    st.markdown("<div style='font-size:11px;color:#374151'>")
    st.markdown("**Algorithms**")
    st.markdown("- GA: BLX-α crossover, adaptive mutation")
    st.markdown("- PSO: inertia decay, ring topology")
    st.markdown("- SA: Metropolis, Cauchy perturbation")
    st.markdown("</div>", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "curves"      not in st.session_state: st.session_state.curves      = {a: [] for a in ["ga", "pso", "sa"]}
if "all_scores"  not in st.session_state: st.session_state.all_scores  = {a: [] for a in ["ga", "pso", "sa"]}
if "bests"       not in st.session_state: st.session_state.bests       = {a: 0.0 for a in ["ga", "pso", "sa"]}
if "done"        not in st.session_state: st.session_state.done        = False
if "logs"        not in st.session_state: st.session_state.logs        = []

if reset_btn:
    st.session_state.curves     = {a: [] for a in ["ga", "pso", "sa"]}
    st.session_state.all_scores = {a: [] for a in ["ga", "pso", "sa"]}
    st.session_state.bests      = {a: 0.0 for a in ["ga", "pso", "sa"]}
    st.session_state.done       = False
    st.session_state.logs       = []
    st.rerun()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:6px'>
  <div>
    <h2 style='margin:0;font-size:22px;font-weight:600;color:#e2e8f0'>CNN HyperTuner</h2>
    <div style='color:#6b7280;font-size:13px;margin-top:2px'>
      Hyperparameter optimisation via Genetic Algorithm · Particle Swarm · Simulated Annealing
    </div>
  </div>
  <div style='text-align:right;font-size:11px;color:#374151'>
    Dataset: <b style='color:#818cf8'>{ds}</b> &nbsp;·&nbsp; 12-dim search space
  </div>
</div>
<hr style='border-color:#1f2333;margin-bottom:20px'>
""".format(ds=dataset), unsafe_allow_html=True)


# ── Top metric cards ──────────────────────────────────────────────────────────
final = FINAL_SCORES[ds_key]
bests = st.session_state.bests
done  = st.session_state.done

overall_best = max(bests.values()) if any(bests.values()) else None
winner = max(bests, key=bests.get) if done else None

col1, col2, col3, col4 = st.columns(4)

with col1:
    v = f"{overall_best*100:.2f}%" if overall_best else "—"
    sub = ALGO_LABELS.get(winner, "—") if done else "run to see"
    st.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>Best accuracy</div>
      <div class='metric-value' style='color:#818cf8'>{v}</div>
      <div class='metric-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    total_evals = sum(len(v) for v in st.session_state.all_scores.values())
    st.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>Total evaluations</div>
      <div class='metric-value' style='color:#fb923c'>{total_evals}</div>
      <div class='metric-sub'>across all algorithms</div>
    </div>""", unsafe_allow_html=True)

with col3:
    speeds = CONV_SPEED[ds_key]
    if done:
        fastest = min(active_algos, key=lambda a: speeds[a]) if active_algos else "—"
        cv = f"{speeds[fastest]} iters ({fastest.upper()})" if active_algos else "—"
    else:
        cv = "—"
    st.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>Convergence speed</div>
      <div class='metric-value' style='color:#4ade80;font-size:20px'>{cv}</div>
      <div class='metric-sub'>iters to 80% accuracy</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""<div class='metric-card'>
      <div class='metric-label'>Search dimensions</div>
      <div class='metric-value' style='color:#a78bfa'>12</div>
      <div class='metric-sub'>hyperparameters tuned</div>
    </div>""", unsafe_allow_html=True)


st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)


# ── Algorithm scorecards ──────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Algorithm scorecards</div>", unsafe_allow_html=True)
acols = st.columns(3)

for i, algo in enumerate(["ga", "pso", "sa"]):
    with acols[i]:
        score  = bests[algo]
        is_win = done and winner == algo
        score_str = f"{score*100:.2f}%" if score else "—"
        win_tag   = "<span class='winner-tag'>BEST</span>" if is_win else ""
        card_cls  = "algo-card winner" if is_win else "algo-card"
        badge_cls = f"algo-badge badge-{algo}"
        st.markdown(f"""
        <div class='{card_cls}'>
          <span class='{badge_cls}'>{algo.upper()}</span>{win_tag}
          <div style='font-size:14px;font-weight:600;color:#e2e8f0;margin:8px 0 4px'>{ALGO_LABELS[algo]}</div>
          <div style='font-size:11px;color:#6b7280;line-height:1.6;margin-bottom:14px'>{ALGO_DESC[algo]}</div>
          <div style='font-size:28px;font-weight:600;color:{COLORS[algo]}'>{score_str}</div>
          <div style='font-size:11px;color:#4b5563;margin-top:2px'>validation accuracy</div>
        </div>
        """, unsafe_allow_html=True)
        if score:
            st.progress(score)


st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)


# ── Main charts row ───────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Convergence curves — live animation</div>", unsafe_allow_html=True)
conv_placeholder = st.empty()
conv_placeholder.plotly_chart(
    build_convergence_fig(st.session_state.curves, max(len(v) for v in st.session_state.curves.values()) or 1),
    use_container_width=True, config={"displayModeBar": False}
)

st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
c_left, c_right = st.columns([3, 2])

with c_left:
    st.markdown("<div class='section-title'>Score distribution</div>", unsafe_allow_html=True)
    dist_placeholder = st.empty()
    dist_placeholder.plotly_chart(
        build_dist_fig(st.session_state.all_scores),
        use_container_width=True, config={"displayModeBar": False}
    )

with c_right:
    st.markdown("<div class='section-title'>Multi-axis performance</div>", unsafe_allow_html=True)
    st.plotly_chart(build_radar_fig(), use_container_width=True, config={"displayModeBar": False})


# ── Second charts row ─────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
r2l, r2r = st.columns(2)

with r2l:
    st.markdown("<div class='section-title'>All evaluations scatter</div>", unsafe_allow_html=True)
    scatter_placeholder = st.empty()
    scatter_placeholder.plotly_chart(
        build_scatter_fig(st.session_state.all_scores, n_iters),
        use_container_width=True, config={"displayModeBar": False}
    )

with r2r:
    st.markdown("<div class='section-title'>Score violin plot</div>", unsafe_allow_html=True)
    violin_placeholder = st.empty()
    violin_placeholder.plotly_chart(
        build_violin_fig(st.session_state.all_scores),
        use_container_width=True, config={"displayModeBar": False}
    )


# ── Heatmap row ───────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Evaluation heatmap — score intensity across iterations</div>", unsafe_allow_html=True)
heat_placeholder = st.empty()
heat_placeholder.plotly_chart(
    build_heatmap_fig(st.session_state.all_scores, n_iters),
    use_container_width=True, config={"displayModeBar": False}
)


# ── Final bar + importance ────────────────────────────────────────────────────
st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
r3l, r3r = st.columns(2)

with r3l:
    st.markdown("<div class='section-title'>Final accuracy comparison</div>", unsafe_allow_html=True)
    bar_placeholder = st.empty()
    display_scores  = bests if done else {a: 0.0 for a in ["ga", "pso", "sa"]}
    bar_placeholder.plotly_chart(
        build_final_bar_fig(display_scores if done else {a: final[a] for a in ["ga","pso","sa"]}),
        use_container_width=True, config={"displayModeBar": False}
    )

with r3r:
    st.markdown("<div class='section-title'>Hyperparameter importance</div>", unsafe_allow_html=True)
    st.plotly_chart(build_importance_fig(), use_container_width=True, config={"displayModeBar": False})


# ── Best config ────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Best hyperparameter configurations</div>", unsafe_allow_html=True)

cfg_tabs = st.tabs(["🟠 GA best config", "🟣 PSO best config", "🟢 SA best config"])
for ti, algo in enumerate(["ga", "pso", "sa"]):
    with cfg_tabs[ti]:
        cfg = BEST_CONFIGS[algo]
        chips = ""
        for k, v in cfg.items():
            chips += f"<div class='cfg-chip'><span class='cfg-key'>{k}</span><span class='cfg-val'>{v}</span></div>"
        st.markdown(f"<div style='line-height:2.6'>{chips}</div>", unsafe_allow_html=True)


# ── Live log ──────────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Run log</div>", unsafe_allow_html=True)
log_placeholder = st.empty()

def render_log():
    lines = st.session_state.logs[-18:] if st.session_state.logs else [("muted", "Press ▶ Run Comparison to start.")]
    html  = "<div class='run-status'>"
    for cls, msg in lines:
        html += f"<div class='log-entry log-{cls}'>{msg}</div>"
    html += "</div>"
    log_placeholder.markdown(html, unsafe_allow_html=True)

render_log()


# ── Code snippet ──────────────────────────────────────────────────────────────
st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Code integration</div>", unsafe_allow_html=True)

code_tab1, code_tab2, code_tab3, code_tab4 = st.tabs(["Basic", "Compare all", "Custom dataset", "Advanced config"])

with code_tab1:
    st.code("""from cnn_hypertuner import CNNHyperTuner

tuner = CNNHyperTuner(
    algorithm="pso",       # "ga" | "pso" | "sa"
    dataset="cifar10",     # "cifar10" | "mnist"
    n_iterations=40,
    n_epochs_per_eval=5,
)
result = tuner.run()
print(f"Best accuracy: {result.best_score:.4f}")
print(result.best_config)
""", language="python")

with code_tab2:
    st.code("""from cnn_hypertuner import CNNHyperTuner
from cnn_hypertuner.utils.plotting import plot_comparison

results = CNNHyperTuner.compare_algorithms(
    algorithms=["ga", "pso", "sa"],
    dataset="cifar10",
    n_iterations=40,
    n_epochs_per_eval=5,
    seed=42,
)
plot_comparison(results, save_path="comparison.png")

for algo, r in results.items():
    print(f"{algo.upper():5s}  acc={r.best_score:.4f}  evals={r.total_evaluations}")
""", language="python")

with code_tab3:
    st.code("""from torch.utils.data import DataLoader
from cnn_hypertuner import CNNHyperTuner

train_loader = DataLoader(your_train_dataset, batch_size=64, shuffle=True)
val_loader   = DataLoader(your_val_dataset,   batch_size=128)

tuner = CNNHyperTuner(
    algorithm="ga",
    dataset="custom",
    n_iterations=30,
    n_epochs_per_eval=5,
    custom_train_loader=train_loader,
    custom_val_loader=val_loader,
)
result = tuner.run()
""", language="python")

with code_tab4:
    st.code("""from cnn_hypertuner import CNNHyperTuner
from cnn_hypertuner.search_space import SearchSpace, HyperparameterSpace
from cnn_hypertuner.utils.plotting import plot_convergence, export_best_config

custom_space = SearchSpace({
    "learning_rate": HyperparameterSpace("learning_rate", "float", 1e-4, 5e-3, log_scale=True),
    "batch_size":    HyperparameterSpace("batch_size",    "categorical", choices=[32, 64, 128]),
})

tuner = CNNHyperTuner(
    algorithm="pso",
    dataset="cifar10",
    n_iterations=50,
    custom_search_space=custom_space,
    algorithm_kwargs={
        "n_particles": 30,
        "topology": "ring",       # "ring" | "global"
        "w_max": 0.9, "w_min": 0.3,
        "stagnation_restart_after": 15,
    },
)
result = tuner.run()
plot_convergence(result, save_path="conv.png")
export_best_config(result, "best_cfg.json")
""", language="python")


# ── Animated run loop ─────────────────────────────────────────────────────────
if run_btn and active_algos:
    st.session_state.curves     = {a: [] for a in ["ga", "pso", "sa"]}
    st.session_state.all_scores = {a: [] for a in ["ga", "pso", "sa"]}
    st.session_state.bests      = {a: 0.0 for a in ["ga", "pso", "sa"]}
    st.session_state.done       = False
    st.session_state.logs       = []

    rng = np.random.default_rng(42)

    def add_log(cls, msg):
        t = time.strftime("%H:%M:%S")
        st.session_state.logs.append((cls, f"[{t}] {msg}"))

    add_log("info", f"Starting comparison: {' · '.join(a.upper() for a in active_algos)}")
    add_log("muted", f"Dataset: {dataset}  |  Iterations: {n_iters}  |  Search dim: 12")

    for it in range(1, n_iters + 1):
        for algo in active_algos:
            s = sim_curve(algo, it, n_iters, ds_key, rng)
            st.session_state.all_scores[algo].append(s)
            st.session_state.bests[algo] = max(st.session_state.bests[algo], s)
            st.session_state.curves[algo].append(st.session_state.bests[algo])

        # update charts
        n_done = it
        conv_placeholder.plotly_chart(
            build_convergence_fig(st.session_state.curves, n_done),
            use_container_width=True, config={"displayModeBar": False}
        )

        if it % 4 == 0 or it == n_iters:
            dist_placeholder.plotly_chart(
                build_dist_fig(st.session_state.all_scores),
                use_container_width=True, config={"displayModeBar": False}
            )
            scatter_placeholder.plotly_chart(
                build_scatter_fig(st.session_state.all_scores, n_iters),
                use_container_width=True, config={"displayModeBar": False}
            )
            heat_placeholder.plotly_chart(
                build_heatmap_fig(st.session_state.all_scores, n_iters),
                use_container_width=True, config={"displayModeBar": False}
            )

        if it % 6 == 0 or it == n_iters:
            violin_placeholder.plotly_chart(
                build_violin_fig(st.session_state.all_scores),
                use_container_width=True, config={"displayModeBar": False}
            )

        if it % 5 == 0:
            scores_str = "  ".join(
                f"{a.upper()}={st.session_state.bests[a]*100:.2f}%"
                for a in active_algos
            )
            add_log("info" if it < n_iters else "success", f"Iter {it:3d}/{n_iters}  {scores_str}")
            render_log()

        time.sleep(speed_ms / 1000)

    st.session_state.done = True
    winner_algo = max(active_algos, key=lambda a: st.session_state.bests[a])
    add_log("success", f"Done! Winner: {ALGO_LABELS[winner_algo]}  ({st.session_state.bests[winner_algo]*100:.4f}%)")
    render_log()

    final_display = {a: st.session_state.bests[a] if a in active_algos else 0.0 for a in ["ga","pso","sa"]}
    bar_placeholder.plotly_chart(
        build_final_bar_fig(final_display),
        use_container_width=True, config={"displayModeBar": False}
    )
    st.rerun()

elif run_btn and not active_algos:
    st.warning("Enable at least one algorithm in the sidebar.")