"""
DRL Railyard Visualizer
=======================
A multi-tab Dash application for presenting how a Deep Reinforcement Learning
agent learns to optimise railway scheduling operations.

Tabs
----
1. Training Progress  -- reward curves, on-time %, conflict reduction
2. Live Episode Replay -- animated railyard with Gantt schedule
3. Agent vs Random     -- side-by-side comparison (same scenario)
4. How DRL Works       -- educational diagrams

Usage:
    python drl_visualizer.py          # opens on http://127.0.0.1:8051
"""

import os, json, math
from bisect import bisect_right

import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

# ---- railyard geometry (reuse existing) --------------------------------
from railyard_app import (
    coordinates, edges_with_lengths, switches, SCALE_X,
    polyline_edges, curve_edges, mixed_edges,
    create_polyline, create_quadratic_bezier, create_mixed_edge,
    edge_paths,
)

# ---- environment -------------------------------------------------------
from railyard_env import RailyardEnv

# ---- SB3 (optional) ---------------------------------------------------
try:
    from stable_baselines3 import PPO
    _SB3 = True
except ImportError:
    _SB3 = False

# ====================== CONSTANTS ======================================
OUTPUT_DIR = "training_output"
DARK_BG   = "#1a1a2e"
PANEL_BG  = "#16213e"
BORDER    = "#2c3e50"
TEXT_COL  = "#ecf0f1"
MUTED     = "#7f8c8d"
ACCENT    = "#3498db"
CARGO_COLORS = ["#e74c3c", "#f39c12", "#3498db"]
CARGO_NAMES  = ["Iron", "Pallets", "Chalk"]
STATUS_COLORS = {
    0: "#95a5a6",  # waiting
    1: "#3498db",  # en-route in
    2: "#9b59b6",  # operating
    3: "#1abc9c",  # en-route out
    4: "#2ecc71",  # completed
}
TRACK_LABELS = {
    98:  ("Iron Loading A",    0),
    100: ("Iron Unloading",    0),
    99:  ("Pallets Loading",   1),
    101: ("Pallets Unloading", 1),
    95:  ("Chalk Loading A",   2),
    96:  ("Chalk Loading B",   2),
    97:  ("Chalk Unloading",   2),
}

# ====================== LOAD ARTEFACTS =================================
def _load_csv(name):
    p = os.path.join(OUTPUT_DIR, name)
    if os.path.isfile(p):
        return pd.read_csv(p)
    return None

def _load_model():
    if not _SB3:
        return None
    # prefer best checkpoint over final model
    best = os.path.join(OUTPUT_DIR, "drl_model_best.zip")
    final = os.path.join(OUTPUT_DIR, "drl_model.zip")
    for p in [best, final]:
        if os.path.isfile(p):
            return PPO.load(p)
    return None

# ====================== GEOMETRY HELPERS ================================
def _interpolate_route(route_nodes, progress):
    """Return (x, y) at *progress* (0-1) along a route of node-ids."""
    if not route_nodes:
        return None
    if len(route_nodes) < 2:
        return coordinates.get(route_nodes[0])
    edges = [(route_nodes[i], route_nodes[i + 1])
             for i in range(len(route_nodes) - 1)]
    elens = []
    for e in edges:
        pd_ = edge_paths.get(e) or edge_paths.get((e[1], e[0]))
        if pd_:
            elens.append(pd_["total_length"])
        else:
            p1 = coordinates.get(e[0], (0, 0))
            p2 = coordinates.get(e[1], (0, 0))
            elens.append(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    total = sum(elens)
    if total < 1e-6:
        return coordinates.get(route_nodes[0])
    target = max(0.0, min(1.0, progress)) * total
    cum = 0.0
    for idx, (e, el) in enumerate(zip(edges, elens)):
        if cum + el >= target or idx == len(edges) - 1:
            frac = (target - cum) / el if el > 0 else 0.0
            frac = max(0.0, min(1.0, frac))
            fwd = e
            rev = (e[1], e[0])
            if fwd in edge_paths:
                pd_ = edge_paths[fwd]
                d = frac * pd_["total_length"]
            elif rev in edge_paths:
                pd_ = edge_paths[rev]
                d = (1.0 - frac) * pd_["total_length"]
            else:
                p1 = coordinates.get(e[0], (0, 0))
                p2 = coordinates.get(e[1], (0, 0))
                return (p1[0] + frac * (p2[0] - p1[0]),
                        p1[1] + frac * (p2[1] - p1[1]))
            pts = pd_["points"]
            cd = pd_["cumulative_dist"]
            d = max(0, min(d, cd[-1]))
            i = max(0, min(bisect_right(cd, d) - 1, len(pts) - 2))
            seg = cd[i + 1] - cd[i]
            t = (d - cd[i]) / seg if seg > 1e-9 else 0
            return (pts[i][0] + t * (pts[i + 1][0] - pts[i][0]),
                    pts[i][1] + t * (pts[i + 1][1] - pts[i][1]))
        cum += el
    return coordinates.get(route_nodes[-1])


def _train_visual_pos(t):
    """Compute (x, y) for a single train dict."""
    s = t["status"]
    if s == 0:
        return coordinates.get(t["entry"])
    if s == 1:
        if t["total_travel_in"] > 0:
            prog = 1.0 - t["travel_remaining"] / t["total_travel_in"]
            pos = _interpolate_route(t["route_in"], prog)
            if pos:
                return pos
        return coordinates.get(t["entry"])
    if s == 2:
        node = RailyardEnv.TRACKS[t["assigned_track"]]["node"]
        return coordinates.get(node)
    if s == 3:
        if t["total_travel_out"] > 0:
            prog = 1.0 - t["return_remaining"] / t["total_travel_out"]
            pos = _interpolate_route(t["route_out"], prog)
            if pos:
                return pos
        if t["assigned_track"] >= 0:
            node = RailyardEnv.TRACKS[t["assigned_track"]]["node"]
            return coordinates.get(node)
        return coordinates.get(t["entry"])
    if s == 4:
        return coordinates.get(t["entry"])
    return None


# ====================== FIGURE BUILDERS ================================
def _base_railyard_fig():
    """Draw the railyard tracks and switches (no trains)."""
    fig = go.Figure()
    # edges
    for n1, n2, _ in edges_with_lengths:
        if n1 not in coordinates or n2 not in coordinates:
            continue
        x1, y1 = coordinates[n1]
        x2, y2 = coordinates[n2]
        if (n1, n2) in polyline_edges or (n2, n1) in polyline_edges:
            key = (n1, n2) if (n1, n2) in polyline_edges else (n2, n1)
            xs, ys = create_polyline(x1, y1, x2, y2,
                                     polyline_edges[key]["segments"],
                                     polyline_edges[key]["offset"])
            col, lw, dash_ = "#4a90d9", 2, "dash"
        elif (n1, n2) in curve_edges or (n2, n1) in curve_edges:
            key = (n1, n2) if (n1, n2) in curve_edges else (n2, n1)
            xs, ys = create_quadratic_bezier(x1, y1, x2, y2,
                                             curve_edges[key]["control_offset"])
            col, lw, dash_ = "#f5a623", 2, "dot"
        elif (n1, n2) in mixed_edges or (n2, n1) in mixed_edges:
            key = (n1, n2) if (n1, n2) in mixed_edges else (n2, n1)
            me = mixed_edges[key]
            xs, ys = create_mixed_edge(x1, y1, x2, y2,
                                       me["straight_length"],
                                       me["control_offset"],
                                       me.get("angle"),
                                       me.get("curve_first", False))
            col, lw, dash_ = "#7ed321", 2, "longdash"
        else:
            xs, ys = [x1, x2], [y1, y2]
            col, lw, dash_ = "#6c757d", 1.5, None
        line_d = dict(color=col, width=lw)
        if dash_:
            line_d["dash"] = dash_
        fig.add_trace(go.Scatter(x=list(xs), y=list(ys), mode="lines",
                                  line=line_d, hoverinfo="skip",
                                  showlegend=False))
    # nodes
    for sw in switches:
        if sw not in coordinates:
            continue
        x, y = coordinates[sw]
        is_track = sw in TRACK_LABELS
        if is_track:
            tname, cargo = TRACK_LABELS[sw]
            mc = CARGO_COLORS[cargo]
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text",
                marker=dict(size=14, color=mc, symbol="square",
                            line=dict(width=2, color="white")),
                text=[tname], textposition="top center",
                textfont=dict(color=mc, size=9, family="Arial Black"),
                hoverinfo="text", hovertext=tname, showlegend=False))
        elif sw in (11, 16):
            label = "Mainline North" if sw == 11 else "Mainline South"
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text",
                marker=dict(size=12, color="#2ecc71",
                            line=dict(width=2, color="white")),
                text=[label], textposition="bottom center",
                textfont=dict(color="#2ecc71", size=10),
                hoverinfo="text", hovertext=label, showlegend=False))
        else:
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers",
                marker=dict(size=6, color="#e8f4f8",
                            line=dict(width=1, color="#3498db")),
                hoverinfo="text", hovertext=f"Sw {sw}", showlegend=False))
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="y", scaleratio=1),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=5, r=5, t=5, b=5), showlegend=False,
        uirevision="const", height=700)
    return fig


def _add_trains_to_fig(fig, trains, decisions=None, current_step=0):
    """Overlay train markers + optional decision/conflict annotations."""
    icons = ["circle", "diamond", "square", "cross", "star",
             "triangle-up", "hexagon", "pentagon"]
    for t in trains:
        pos = _train_visual_pos(t)
        if pos is None:
            continue
        cargo = t["cargo"]
        col = CARGO_COLORS[cargo]
        status = t["status"]
        symbol = icons[t["id"] % len(icons)]
        opacity = 0.35 if status == 4 else 1.0
        size = 22 if status in (1, 3) else (18 if status == 2 else 14)
        label = f"T{t['id']} {CARGO_NAMES[cargo]}"
        fig.add_trace(go.Scatter(
            x=[pos[0]], y=[pos[1]], mode="markers+text",
            marker=dict(size=size, color=col, symbol=symbol,
                        opacity=opacity, line=dict(width=2, color="white")),
            text=[f"T{t['id']}"], textfont=dict(size=9, color="white"),
            textposition="top center",
            hoverinfo="text",
            hovertext=f"{label}<br>{RailyardEnv.STATUS_NAMES[status]}",
            showlegend=False))
    # conflict flash
    if decisions:
        recent = [d for d in decisions
                  if d["result"] == "conflict"
                  and d["step"] >= current_step - 3]
        for d in recent:
            tid = d["train"]
            tr = next((t for t in trains if t["id"] == tid), None)
            if tr:
                pos = _train_visual_pos(tr)
                if pos:
                    fig.add_trace(go.Scatter(
                        x=[pos[0]], y=[pos[1]], mode="markers",
                        marker=dict(size=40, color="red", opacity=0.25,
                                    symbol="circle"),
                        hoverinfo="skip", showlegend=False))
    return fig


def _gantt_fig(trains, current_step):
    """Horizontal Gantt-style schedule chart."""
    fig = go.Figure()
    labels = []
    for t in sorted(trains, key=lambda x: x["id"]):
        lab = f"T{t['id']} {CARGO_NAMES[t['cargo']]}"
        labels.append(lab)
        # slot window
        sw_start = t["slot_center"] - t["slot_window"]
        sw_end   = t["slot_center"] + t["slot_window"]
        fig.add_trace(go.Bar(
            y=[lab], x=[sw_end - sw_start], base=[sw_start],
            orientation="h", marker_color="rgba(46,204,113,0.18)",
            showlegend=False, hoverinfo="text",
            hovertext=f"Slot {sw_start}-{sw_end}"))
        # actual progress
        if t["status"] >= 1:
            start = t["arrival"]
            end = t["completed_step"] if t["status"] == 4 else current_step
            on_time = (t["status"] == 4 and
                       abs(t["completed_step"] - t["slot_center"]) <= t["slot_window"])
            bar_col = "#2ecc71" if on_time else (
                CARGO_COLORS[t["cargo"]] if t["status"] != 4 else "#e74c3c")
            fig.add_trace(go.Bar(
                y=[lab], x=[max(1, end - start)], base=[start],
                orientation="h", marker_color=bar_col,
                marker_opacity=0.85, showlegend=False,
                hoverinfo="text",
                hovertext=f"{'ON TIME' if on_time else 'Active'} {start}-{end}"))
    fig.update_layout(
        barmode="overlay", paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
        xaxis=dict(title="Timestep", color=TEXT_COL, gridcolor=BORDER,
                   range=[0, RailyardEnv.MAX_STEPS]),
        yaxis=dict(color=TEXT_COL, autorange="reversed"),
        margin=dict(l=120, r=10, t=10, b=35), height=220,
        font=dict(color=TEXT_COL, size=10))
    return fig


# ====================== TRAINING CHARTS ================================
def _training_charts():
    """Build the four charts for Tab 1."""
    t_df = _load_csv("training_log.csv")
    b_df = _load_csv("baseline_log.csv")
    e_df = _load_csv("eval_log.csv")

    figs = {}

    if t_df is not None and len(t_df) > 0:
        # reward curve
        window = max(1, len(t_df) // 20)
        smooth = t_df["reward"].rolling(window, min_periods=1).mean()
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatter(y=t_df["reward"], mode="lines",
                                    line=dict(color="rgba(52,152,219,0.2)", width=1),
                                    name="Raw"))
        fig_r.add_trace(go.Scatter(y=smooth, mode="lines",
                                    line=dict(color="#3498db", width=3),
                                    name="Smoothed"))
        fig_r.update_layout(title="Episode Reward During Training",
                            **_chart_layout(), yaxis_title="Reward",
                            xaxis_title="Episode")
        figs["reward"] = fig_r

        # on-time %
        sm_ot = t_df["on_time_pct"].rolling(window, min_periods=1).mean()
        fig_ot = go.Figure()
        fig_ot.add_trace(go.Scatter(y=t_df["on_time_pct"], mode="lines",
                                     line=dict(color="rgba(46,204,113,0.2)", width=1),
                                     name="Raw"))
        fig_ot.add_trace(go.Scatter(y=sm_ot, mode="lines",
                                     line=dict(color="#2ecc71", width=3),
                                     name="Smoothed"))
        fig_ot.update_layout(title="On-Time Departure %",
                            **_chart_layout(), yaxis_title="%",
                            xaxis_title="Episode")
        figs["on_time"] = fig_ot

        # conflicts
        sm_c = t_df["conflicts"].rolling(window, min_periods=1).mean()
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(y=t_df["conflicts"], mode="lines",
                                    line=dict(color="rgba(231,76,60,0.2)", width=1),
                                    name="Raw"))
        fig_c.add_trace(go.Scatter(y=sm_c, mode="lines",
                                    line=dict(color="#e74c3c", width=3),
                                    name="Smoothed"))
        fig_c.update_layout(title="Junction Conflicts per Episode",
                            **_chart_layout(), yaxis_title="Conflicts",
                            xaxis_title="Episode")
        figs["conflicts"] = fig_c
    else:
        placeholder = _placeholder_fig("Run  python train_agent.py  first")
        figs["reward"] = figs["on_time"] = figs["conflicts"] = placeholder

    # comparison bar
    fig_bar = go.Figure()
    metrics = ["reward", "on_time_pct", "conflicts", "avg_delay"]
    nice = ["Avg Reward", "On-Time %", "Conflicts", "Avg Delay"]
    b_vals, e_vals = [], []
    for m in metrics:
        b_vals.append(b_df[m].mean() if b_df is not None and m in b_df else 0)
        e_vals.append(e_df[m].mean() if e_df is not None and m in e_df else 0)
    fig_bar.add_trace(go.Bar(x=nice, y=b_vals, name="Random",
                              marker_color="#e74c3c"))
    fig_bar.add_trace(go.Bar(x=nice, y=e_vals, name="Trained Agent",
                              marker_color="#2ecc71"))
    fig_bar.update_layout(title="Trained Agent vs Random Baseline (100 ep avg)",
                          barmode="group", **_chart_layout())
    figs["comparison"] = fig_bar
    return figs


def _chart_layout():
    return dict(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                font=dict(color=TEXT_COL, size=11),
                margin=dict(l=50, r=20, t=40, b=40),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                height=320)


def _placeholder_fig(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=16, color=MUTED))
    fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                      xaxis=dict(visible=False), yaxis=dict(visible=False),
                      height=320)
    return fig


# ====================== GLOBAL STATE ===================================
class _S:
    """Mutable global state (single-user presentation tool)."""
    # Tab 2
    r_env = None
    r_model = None
    r_obs = None
    r_running = False
    # Tab 3
    c_env_a = None      # agent
    c_env_r = None      # random
    c_obs_a = None
    c_obs_r = None
    c_running = False
    c_seed = 42

S = _S()

# ====================== STYLES =========================================
_card = dict(backgroundColor=PANEL_BG, borderRadius="8px", padding="14px",
             border=f"1px solid {BORDER}")
_btn_base = dict(padding="10px 22px", border="none", borderRadius="6px",
                 cursor="pointer", fontWeight="700", fontSize="13px",
                 color="white")
_btn_green = {**_btn_base, "background": "linear-gradient(135deg,#2ecc71,#27ae60)"}
_btn_red   = {**_btn_base, "background": "linear-gradient(135deg,#e74c3c,#c0392b)"}
_btn_blue  = {**_btn_base, "background": "linear-gradient(135deg,#3498db,#2980b9)"}

# ====================== DASH APP =======================================
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Railyard DRL Optimizer"

# ---- Tab layouts -------------------------------------------------------
def _tab1_layout():
    charts = _training_charts()
    return html.Div(style={"padding": "20px"}, children=[
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                         "gap": "16px"}, children=[
            html.Div(style=_card, children=[dcc.Graph(figure=charts["reward"],
                                                 config={"displayModeBar": False})]),
            html.Div(style=_card, children=[dcc.Graph(figure=charts["on_time"],
                                                 config={"displayModeBar": False})]),
            html.Div(style=_card, children=[dcc.Graph(figure=charts["conflicts"],
                                                 config={"displayModeBar": False})]),
            html.Div(style=_card, children=[dcc.Graph(figure=charts["comparison"],
                                                 config={"displayModeBar": False})]),
        ]),
    ])


def _tab2_layout():
    return html.Div(children=[
        # controls
        html.Div(style={"padding": "12px 20px", "display": "flex",
                         "gap": "12px", "alignItems": "center",
                         "backgroundColor": PANEL_BG,
                         "borderBottom": f"1px solid {BORDER}"}, children=[
            html.Button("Start Episode", id="r-start", n_clicks=0,
                         style=_btn_green),
            html.Button("Stop", id="r-stop", n_clicks=0, style=_btn_red),
            html.Span("Speed:", style={"color": MUTED, "marginLeft": "18px"}),
            dcc.Slider(id="r-speed", min=50, max=500, step=50, value=150,
                       marks={50: "Fast", 150: "Normal", 500: "Slow"},
                       tooltip={"placement": "bottom"}),
            html.Div(id="r-status", style={"marginLeft": "auto",
                                            "color": ACCENT, "fontWeight": "600"}),
        ]),
        # body
        html.Div(style={"display": "flex", "height": "calc(100vh - 180px)"}, children=[
            # railyard
            html.Div(style={"flex": "1", "minWidth": 0}, children=[
                dcc.Graph(id="r-graph", figure=_base_railyard_fig(),
                          style={"height": "100%"},
                          config={"displayModeBar": False, "scrollZoom": True}),
            ]),
            # side panel
            html.Div(style={"width": "260px", "padding": "12px",
                             "overflowY": "auto",
                             "borderLeft": f"1px solid {BORDER}",
                             "backgroundColor": PANEL_BG}, children=[
                html.Div("Metrics", style={"fontSize": "11px",
                         "fontWeight": "700", "color": ACCENT,
                         "textTransform": "uppercase",
                         "marginBottom": "10px"}),
                html.Div(id="r-metrics"),
                html.Div("Decisions", style={"fontSize": "11px",
                         "fontWeight": "700", "color": ACCENT,
                         "textTransform": "uppercase",
                         "marginTop": "16px", "marginBottom": "10px"}),
                html.Div(id="r-decisions",
                         style={"maxHeight": "300px", "overflowY": "auto"}),
            ]),
        ]),
        # gantt
        html.Div(style={"padding": "0 20px 10px"}, children=[
            dcc.Graph(id="r-gantt", figure=_gantt_fig([], 0),
                      config={"displayModeBar": False}),
        ]),
        dcc.Interval(id="r-tick", interval=150, disabled=True),
    ])


def _tab3_layout():
    return html.Div(children=[
        html.Div(style={"padding": "16px 20px", "backgroundColor": PANEL_BG,
                         "borderBottom": f"1px solid {BORDER}"}, children=[
            html.Div([
                html.Span("Same scenario, same delays — different decisions. ", style={"color": TEXT_COL}),
                html.Span("Random policy picks a track (or wait) at random. "),
                html.Span("The trained agent uses what it learned: ", style={"color": TEXT_COL}),
                html.Span("higher total reward", style={"color": "#2ecc71", "fontWeight": "700"}),
                html.Span(" = more trains ", style={"color": TEXT_COL}),
                html.Span("on time", style={"color": "#2ecc71", "fontWeight": "700"}),
                html.Span(", fewer ", style={"color": TEXT_COL}),
                html.Span("conflicts", style={"color": "#e74c3c", "fontWeight": "700"}),
                html.Span(", and more trains ", style={"color": TEXT_COL}),
                html.Span("completed", style={"color": "#2ecc71", "fontWeight": "700"}),
                html.Span(". Compare the numbers below after a run.", style={"color": MUTED, "fontSize": "12px"}),
            ], style={"fontSize": "13px", "lineHeight": "1.5", "marginBottom": "12px"}),
            html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center"}, children=[
                html.Button("Start Comparison", id="c-start", n_clicks=0,
                             style=_btn_blue),
                html.Button("Stop", id="c-stop", n_clicks=0, style=_btn_red),
                html.Div(id="c-status", style={"marginLeft": "auto",
                                                "color": ACCENT, "fontWeight": "600"}),
            ]),
        ]),
        html.Div(style={"display": "flex", "gap": "4px"}, children=[
            html.Div(style={"flex": "1", "textAlign": "center"}, children=[
                html.Div("RANDOM POLICY", style={"padding": "6px",
                         "backgroundColor": "#4a1a1a", "color": "#e74c3c",
                         "fontWeight": "700", "fontSize": "13px",
                         "letterSpacing": "2px"}),
                dcc.Graph(id="c-graph-rand", figure=_base_railyard_fig(),
                          config={"displayModeBar": False, "scrollZoom": True},
                          style={"height": "60vh"}),
            ]),
            html.Div(style={"flex": "1", "textAlign": "center"}, children=[
                html.Div("TRAINED DRL AGENT", style={"padding": "6px",
                         "backgroundColor": "#0a2a1a", "color": "#2ecc71",
                         "fontWeight": "700", "fontSize": "13px",
                         "letterSpacing": "2px"}),
                dcc.Graph(id="c-graph-agent", figure=_base_railyard_fig(),
                          config={"displayModeBar": False, "scrollZoom": True},
                          style={"height": "60vh"}),
            ]),
        ]),
        html.Div(id="c-metrics", style={"padding": "16px 20px"}),
        dcc.Interval(id="c-tick", interval=150, disabled=True),
    ])


def _tab4_layout():
    box = lambda title, body, col=ACCENT: html.Div(
        style={**_card, "borderLeft": f"4px solid {col}",
               "marginBottom": "14px"}, children=[
            html.Div(title, style={"fontWeight": "700", "fontSize": "14px",
                                    "color": col, "marginBottom": "6px"}),
            html.Div(body, style={"color": TEXT_COL, "fontSize": "13px",
                                   "lineHeight": "1.6"}),
        ])
    return html.Div(style={"padding": "24px", "maxWidth": "960px",
                             "margin": "0 auto"}, children=[
        html.H2("How Deep Reinforcement Learning Works",
                style={"color": TEXT_COL, "textAlign": "center",
                        "marginBottom": "28px"}),
        # DRL loop
        html.Div(style={"display": "flex", "justifyContent": "center",
                         "gap": "0", "alignItems": "center",
                         "marginBottom": "30px", "flexWrap": "wrap"}, children=[
            _loop_box("STATE", "Track occupancy, train positions,\ntime, delays",
                      "#3498db"),
            _arrow_right(),
            _loop_box("AGENT", "Neural network\n(PPO policy)",
                      "#9b59b6"),
            _arrow_right(),
            _loop_box("ACTION", "Assign train to track\nor wait",
                      "#f39c12"),
            _arrow_right(),
            _loop_box("ENVIRONMENT", "Railyard simulation\n(moves trains, applies delays)",
                      "#1abc9c"),
            _arrow_right(),
            _loop_box("REWARD", "+10 on-time, -5 conflict\n-3 wrong cargo",
                      "#e74c3c"),
        ]),
        html.Div(style={"textAlign": "center", "color": MUTED,
                         "marginBottom": "28px", "fontSize": "13px"},
                 children="The agent observes the state, picks an action, "
                          "receives a reward, and repeats -- thousands of times."),

        box("State Space (what the agent sees)",
            "For each of the 7 tracks: is it occupied? what cargo? time remaining. "
            "For each of the 8 train slots: cargo type, status, departure slot, delay. "
            "Plus the current timestep. Total: 62 numbers.",
            "#3498db"),
        box("Action Space (what the agent can do)",
            "8 discrete choices: assign the first waiting train to one of 7 tracks "
            "(Iron Loading, Iron Unloading, Pallets Loading, Pallets Unloading, "
            "Chalk Loading A, Chalk Loading B, Chalk Unloading) -- or WAIT.",
            "#f39c12"),
        box("Reward Signal (how the agent learns)",
            html.Ul(style={"margin": "6px 0", "paddingLeft": "20px"}, children=[
                html.Li("+10  Train departs mainline on time"),
                html.Li("+3   Train arrives at correct track"),
                html.Li("+1.5 Valid track assignment"),
                html.Li("-5   Junction conflict between two trains"),
                html.Li("-3   Wrong cargo type assignment"),
                html.Li("-2   Track already occupied"),
                html.Li("+20  All trains completed / -penalty for timeout"),
            ]),
            "#e74c3c"),
        box("Training Process",
            "The agent plays ~250 episodes (50 000 timesteps). Early on it acts "
            "almost randomly and scores poorly. Through gradient updates (PPO), "
            "it gradually discovers: (1) match cargo types, (2) avoid occupied "
            "tracks, (3) time assignments to meet departure slots, and "
            "(4) stagger routes to prevent junction conflicts.",
            "#2ecc71"),
        box("Why the trained agent is better than random",
            html.Div([
                "In the \"Agent vs Random\" tab, both policies see the same scenario (same trains, delays, slots). "
                "The difference is how they choose: ",
                html.Strong("random", style={"color": "#e74c3c"}),
                " picks an action by chance; the ",
                html.Strong("trained agent", style={"color": "#2ecc71"}),
                " uses its learned policy. You’ll see the agent do better on: ",
                html.Ul(style={"margin": "8px 0", "paddingLeft": "20px"}, children=[
                    html.Li(html.Strong("Reward") + " — higher total (more on-time bonuses, fewer penalties)."),
                    html.Li(html.Strong("On-Time") + " — more trains leaving within their mainline slot."),
                    html.Li(html.Strong("Conflicts") + " — fewer junction conflicts (safer, less delay)."),
                    html.Li(html.Strong("Completed") + " — more trains finished before the step limit."),
                ]),
                "So \"better\" here means: the agent has learned to schedule in a way that maximizes reward in this environment.",
            ], style={"lineHeight": "1.6"}),
            "#2ecc71"),
        box("Cargo Types & Tracks",
            html.Div(style={"display": "flex", "gap": "20px",
                             "flexWrap": "wrap"}, children=[
                _cargo_badge("Iron", "#e74c3c",
                             "Loading: Track 98 | Unloading: Track 100"),
                _cargo_badge("Pallets", "#f39c12",
                             "Loading: Track 99 | Unloading: Track 101"),
                _cargo_badge("Chalk", "#3498db",
                             "Loading: Tracks 95 & 96 | Unloading: Track 97"),
            ]),
            "#9b59b6"),
    ])


def _loop_box(title, body, color):
    return html.Div(style={"backgroundColor": PANEL_BG, "border": f"2px solid {color}",
                            "borderRadius": "10px", "padding": "12px 16px",
                            "textAlign": "center", "width": "130px"}, children=[
        html.Div(title, style={"fontWeight": "800", "color": color,
                                "fontSize": "12px", "letterSpacing": "1px"}),
        html.Div(body, style={"color": MUTED, "fontSize": "10px",
                               "marginTop": "4px", "whiteSpace": "pre-line"}),
    ])


def _arrow_right():
    return html.Div(style={"fontSize": "22px", "color": MUTED,
                            "padding": "0 6px"}, children=">>>")


def _cargo_badge(name, color, desc):
    return html.Div(style={"display": "flex", "alignItems": "center",
                            "gap": "8px"}, children=[
        html.Div(style={"width": "16px", "height": "16px",
                         "borderRadius": "4px", "backgroundColor": color}),
        html.Div([html.Span(name, style={"fontWeight": "700", "color": color}),
                   html.Br(),
                   html.Span(desc, style={"fontSize": "11px", "color": MUTED})]),
    ])


# ====================== MAIN LAYOUT ====================================
app.layout = html.Div(style={"backgroundColor": DARK_BG, "minHeight": "100vh",
                               "fontFamily": "'Segoe UI', sans-serif",
                               "color": TEXT_COL}, children=[
    html.Div(style={"display": "flex", "alignItems": "center",
                     "padding": "12px 24px",
                     "borderBottom": f"1px solid {BORDER}",
                     "backgroundColor": PANEL_BG}, children=[
        html.Div(style={"fontSize": "18px", "fontWeight": "800",
                         "letterSpacing": "1px"}, children=[
            html.Span("RAILYARD ", style={"color": ACCENT}),
            html.Span("DRL OPTIMIZER", style={"color": TEXT_COL}),
        ]),
        html.Div(style={"marginLeft": "auto", "display": "flex",
                         "gap": "12px"}, children=[
            html.Div(style={"display": "flex", "alignItems": "center",
                             "gap": "5px"}, children=[
                html.Div(style={"width": "10px", "height": "10px",
                                 "borderRadius": "50%",
                                 "backgroundColor": c}),
                html.Span(n, style={"fontSize": "11px", "color": MUTED}),
            ]) for n, c in zip(CARGO_NAMES, CARGO_COLORS)
        ]),
    ]),
    dcc.Tabs(id="main-tabs", value="tab1",
             style={"backgroundColor": PANEL_BG},
             children=[
        dcc.Tab(label="Training Progress", value="tab1",
                style={"backgroundColor": PANEL_BG, "color": MUTED,
                        "border": "none", "padding": "10px 20px"},
                selected_style={"backgroundColor": DARK_BG, "color": ACCENT,
                                 "borderTop": f"2px solid {ACCENT}",
                                 "padding": "10px 20px"}),
        dcc.Tab(label="Live Episode Replay", value="tab2",
                style={"backgroundColor": PANEL_BG, "color": MUTED,
                        "border": "none", "padding": "10px 20px"},
                selected_style={"backgroundColor": DARK_BG, "color": ACCENT,
                                 "borderTop": f"2px solid {ACCENT}",
                                 "padding": "10px 20px"}),
        dcc.Tab(label="Agent vs Random", value="tab3",
                style={"backgroundColor": PANEL_BG, "color": MUTED,
                        "border": "none", "padding": "10px 20px"},
                selected_style={"backgroundColor": DARK_BG, "color": ACCENT,
                                 "borderTop": f"2px solid {ACCENT}",
                                 "padding": "10px 20px"}),
        dcc.Tab(label="How DRL Works", value="tab4",
                style={"backgroundColor": PANEL_BG, "color": MUTED,
                        "border": "none", "padding": "10px 20px"},
                selected_style={"backgroundColor": DARK_BG, "color": ACCENT,
                                 "borderTop": f"2px solid {ACCENT}",
                                 "padding": "10px 20px"}),
    ]),
    html.Div(id="tab-content"),
])


# ====================== CALLBACKS ======================================

# ---- Tab routing -------------------------------------------------------
@app.callback(Output("tab-content", "children"),
              Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "tab1":
        return _tab1_layout()
    if tab == "tab2":
        return _tab2_layout()
    if tab == "tab3":
        return _tab3_layout()
    return _tab4_layout()


# ---- Tab 2: replay controls -------------------------------------------
@app.callback(
    [Output("r-tick", "disabled"), Output("r-tick", "interval"),
     Output("r-status", "children")],
    [Input("r-start", "n_clicks"), Input("r-stop", "n_clicks")],
    [State("r-speed", "value")],
    prevent_initial_call=True,
)
def replay_control(start, stop, speed):
    tid = ctx.triggered_id
    if tid == "r-start":
        env = RailyardEnv(num_trains=6, delay_prob=0.2, log_episodes=False)
        obs, _ = env.reset(seed=np.random.randint(0, 9999))
        S.r_env = env
        S.r_obs = obs
        S.r_model = _load_model()
        S.r_running = True
        return False, speed or 150, "Running..."
    S.r_running = False
    return True, 150, "Stopped"


@app.callback(
    [Output("r-graph", "figure"), Output("r-gantt", "figure"),
     Output("r-metrics", "children"), Output("r-decisions", "children"),
     Output("r-status", "children", allow_duplicate=True)],
    Input("r-tick", "n_intervals"),
    prevent_initial_call=True,
)
def replay_tick(_):
    if not S.r_running or S.r_env is None:
        raise dash.exceptions.PreventUpdate
    env = S.r_env
    obs = S.r_obs
    if S.r_model is not None:
        action, _ = S.r_model.predict(obs, deterministic=False)
    else:
        action = env.action_space.sample()
    obs, rew, term, trunc, info = env.step(action)
    S.r_obs = obs
    if term or trunc:
        S.r_running = False

    trains = info["trains_full"]
    fig = _base_railyard_fig()
    _add_trains_to_fig(fig, trains, info["decisions"], info["step"])
    gantt = _gantt_fig(trains, info["step"])
    metrics = _metrics_panel(info)
    decisions = _decision_log(info["decisions"])
    status_txt = (f"Step {info['step']}/{RailyardEnv.MAX_STEPS}  |  "
                  f"Reward: {info['total_reward']:.1f}")
    if term:
        status_txt += "  |  EPISODE COMPLETE"
    elif trunc:
        status_txt += "  |  TIME OUT"
    return fig, gantt, metrics, decisions, status_txt


# ---- Tab 3: comparison controls ---------------------------------------
@app.callback(
    [Output("c-tick", "disabled"), Output("c-status", "children")],
    [Input("c-start", "n_clicks"), Input("c-stop", "n_clicks")],
    prevent_initial_call=True,
)
def compare_control(start, stop):
    tid = ctx.triggered_id
    if tid == "c-start":
        seed = np.random.randint(0, 9999)
        S.c_seed = seed
        env_a = RailyardEnv(num_trains=6, delay_prob=0.2, log_episodes=False)
        env_r = RailyardEnv(num_trains=6, delay_prob=0.2, log_episodes=False)
        S.c_obs_a, _ = env_a.reset(seed=seed)
        S.c_obs_r, _ = env_r.reset(seed=seed)
        S.c_env_a = env_a
        S.c_env_r = env_r
        S.c_running = True
        S.r_model = _load_model()  # reuse
        return False, "Running..."
    S.c_running = False
    return True, "Stopped"


@app.callback(
    [Output("c-graph-rand", "figure"), Output("c-graph-agent", "figure"),
     Output("c-metrics", "children"),
     Output("c-status", "children", allow_duplicate=True)],
    Input("c-tick", "n_intervals"),
    prevent_initial_call=True,
)
def compare_tick(_):
    if not S.c_running:
        raise dash.exceptions.PreventUpdate
    # random
    env_r = S.c_env_r
    act_r = env_r.action_space.sample()
    obs_r, _, t_r, tr_r, info_r = env_r.step(act_r)
    S.c_obs_r = obs_r
    # agent
    env_a = S.c_env_a
    if S.r_model is not None:
        act_a, _ = S.r_model.predict(S.c_obs_a, deterministic=False)
    else:
        act_a = env_a.action_space.sample()
    obs_a, _, t_a, tr_a, info_a = env_a.step(act_a)
    S.c_obs_a = obs_a

    done = (t_r or tr_r) and (t_a or tr_a)
    if done:
        S.c_running = False

    fig_r = _base_railyard_fig()
    _add_trains_to_fig(fig_r, info_r["trains_full"],
                       info_r["decisions"], info_r["step"])
    fig_a = _base_railyard_fig()
    _add_trains_to_fig(fig_a, info_a["trains_full"],
                       info_a["decisions"], info_a["step"])

    metrics = _compare_metrics(info_r, info_a, done=done)
    step = max(info_r["step"], info_a["step"])
    status = f"Step {step}/{RailyardEnv.MAX_STEPS}"
    if done:
        status += "  |  DONE"
    return fig_r, fig_a, metrics, status


# ====================== UI BUILDERS ====================================
def _metrics_panel(info):
    items = [
        ("Step", f"{info['step']} / {RailyardEnv.MAX_STEPS}"),
        ("Reward", f"{info['total_reward']:.1f}"),
        ("Completed", f"{info['completed']} / {info['total_trains']}"),
        ("On-Time", str(info["on_time"])),
        ("Late", str(info["late"])),
        ("Conflicts", str(info["conflicts"])),
    ]
    return html.Div([
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                         "padding": "6px 0",
                         "borderBottom": f"1px solid {BORDER}"}, children=[
            html.Span(k, style={"color": MUTED, "fontSize": "11px"}),
            html.Span(v, style={"color": TEXT_COL, "fontWeight": "700",
                                 "fontSize": "12px"}),
        ]) for k, v in items
    ] + [
        # train cards
        html.Div(style={"marginTop": "12px"}, children=[
            _train_card(t) for t in info.get("trains_full", [])
        ]),
    ])


def _train_card(t):
    cargo = t["cargo"]
    col = CARGO_COLORS[cargo]
    status = t["status"]
    return html.Div(style={"padding": "6px 8px", "marginBottom": "4px",
                            "borderRadius": "4px", "backgroundColor": DARK_BG,
                            "borderLeft": f"3px solid {col}"}, children=[
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                         "alignItems": "center"}, children=[
            html.Span(f"T{t['id']} {CARGO_NAMES[cargo]}",
                      style={"color": col, "fontSize": "11px",
                              "fontWeight": "700"}),
            html.Span(RailyardEnv.STATUS_NAMES[status],
                      style={"fontSize": "9px", "fontWeight": "700",
                              "color": "white", "padding": "1px 6px",
                              "borderRadius": "8px",
                              "backgroundColor": STATUS_COLORS.get(status, MUTED)}),
        ]),
    ])


def _decision_log(decisions):
    recent = decisions[-15:] if decisions else []
    result_colors = {"assigned": "#2ecc71", "arrived": "#3498db",
                     "departing": "#9b59b6", "on_time": "#2ecc71",
                     "late": "#e74c3c", "conflict": "#e74c3c",
                     "delay": "#f39c12", "occupied": "#e74c3c",
                     "wrong_cargo": "#e74c3c"}
    return html.Div([
        html.Div(style={"padding": "4px 6px", "marginBottom": "3px",
                         "borderRadius": "3px", "backgroundColor": DARK_BG,
                         "borderLeft": f"2px solid "
                                       f"{result_colors.get(d['result'], MUTED)}",
                         "fontSize": "10px"}, children=[
            html.Span(f"[{d['step']:>3}] ", style={"color": MUTED}),
            html.Span(f"T{d['train']} ", style={"fontWeight": "700"}),
            html.Span(d["desc"]),
        ]) for d in reversed(recent)
    ])


def _compare_metrics(info_r, info_a, done=False):
    rows = [
        ("Reward", info_r["total_reward"], info_a["total_reward"]),
        ("On-Time", info_r["on_time"], info_a["on_time"]),
        ("Late", info_r["late"], info_a["late"]),
        ("Conflicts", info_r["conflicts"], info_a["conflicts"]),
        ("Completed", info_r["completed"], info_a["completed"]),
    ]
    cards = [
        html.Div(style={**_card, "textAlign": "center", "minWidth": "130px"},
                 children=[
            html.Div(label, style={"fontSize": "10px", "color": MUTED,
                                    "textTransform": "uppercase",
                                    "letterSpacing": "1px"}),
            html.Div(style={"display": "flex", "justifyContent": "center",
                             "gap": "20px", "marginTop": "6px"}, children=[
                html.Div([
                    html.Div(f"{rv:.1f}" if isinstance(rv, float) else str(rv),
                             style={"fontSize": "20px", "fontWeight": "800",
                                     "color": "#e74c3c"}),
                    html.Div("Random", style={"fontSize": "9px", "color": MUTED}),
                ]),
                html.Div([
                    html.Div(f"{av:.1f}" if isinstance(av, float) else str(av),
                             style={"fontSize": "20px", "fontWeight": "800",
                                     "color": "#2ecc71"}),
                    html.Div("Agent", style={"fontSize": "9px", "color": MUTED}),
                ]),
            ]),
        ]) for label, rv, av in rows
    ]
    # When episode is done, show a one-line verdict so "how is agent better" is obvious
    summary = []
    if done:
        d_reward = info_a["total_reward"] - info_r["total_reward"]
        d_ontime = info_a["on_time"] - info_r["on_time"]
        d_conflicts = info_r["conflicts"] - info_a["conflicts"]  # fewer is better
        d_completed = info_a["completed"] - info_r["completed"]
        winner = "Agent" if d_reward > 0 else ("Random" if d_reward < 0 else "Tie")
        parts = []
        if abs(d_reward) >= 0.5:
            parts.append(f"{d_reward:+.1f} reward")
        if d_ontime != 0:
            parts.append(f"{d_ontime:+d} on-time")
        if d_conflicts != 0:
            parts.append(f"{d_conflicts:+d} conflicts")
        if d_completed != 0:
            parts.append(f"{d_completed:+d} completed")
        verdict = f"{winner} wins this run" + (": " + ", ".join(parts) if parts else ".")
        summary = [
            html.Div(style={**_card, "marginBottom": "16px", "textAlign": "center",
                            "border": f"2px solid {'#2ecc71' if d_reward > 0 else '#e74c3c' if d_reward < 0 else MUTED}",
                            "padding": "12px 20px"}, children=[
                html.Div("Run complete — how did the agent do?", style={"fontSize": "10px", "color": MUTED, "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "4px"}),
                html.Div(verdict, style={"fontSize": "16px", "fontWeight": "800", "color": TEXT_COL}),
            ])
        ]
    return html.Div(children=summary + [
        html.Div(style={"display": "flex", "justifyContent": "center",
                        "gap": "40px", "flexWrap": "wrap"}, children=cards)
    ])


# ====================== MAIN ===========================================
if __name__ == "__main__":
    print("\n  Railyard DRL Visualizer")
    print("  http://127.0.0.1:8051\n")
    app.run(debug=False, port=8051)
