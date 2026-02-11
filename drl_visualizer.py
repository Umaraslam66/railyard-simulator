"""
DRL Railyard Visualizer
=======================
A multi-tab Dash application for presenting how a Deep Reinforcement Learning
agent learns to optimise railway scheduling operations.

Tabs
----
1. Training Progress       -- reward curves, on-time %, conflict reduction
2. Live Episode Replay     -- animated railyard with Gantt schedule
3. Schedule & DRL Recovery  -- nominal schedule, 200 delayed variants, time-series
4. How DRL Works           -- educational diagrams

Usage:
    python drl_visualizer.py          # opens on http://127.0.0.1:8051
"""

import os, json, math
from bisect import bisect_right

import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
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
from schedule_builder import (
    schedule_to_env_format, build_delay_vectors as sb_build_delay_vectors,
    load_schedule as sb_load_schedule, TOTAL_STEPS as SB_TOTAL_STEPS,
    STEP_MINUTES as SB_STEP_MINUTES,
)

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
# For time-series: y-axis locations (order matters)
LOCATION_ORDER = [
    "Waiting", "En route (in)", "Iron Loading A", "Iron Unloading",
    "Pallets Loading", "Pallets Unloading", "Chalk Loading A", "Chalk Loading B",
    "Chalk Unloading", "En route (out)", "Completed",
]
LOCATION_TO_Y = {loc: i for i, loc in enumerate(LOCATION_ORDER)}

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


# ====================== SCHEDULE & DELAY SCENARIOS =======================
# Uses schedule_builder.py as the single source of truth for the nominal
# schedule and delay vector generation.  No duplicate functions here.


def _train_location_at_step(train, step):
    """Return location string for time-series: Waiting, track name, En route (in/out), Completed."""
    s = train["status"]
    if s == RailyardEnv.WAITING:
        return "Waiting"
    if s == RailyardEnv.EN_ROUTE_IN:
        return "En route (in)"
    if s == RailyardEnv.OPERATING:
        idx = train.get("assigned_track", -1)
        if 0 <= idx < len(RailyardEnv.TRACKS):
            return RailyardEnv.TRACKS[idx]["name"]
        return "En route (in)"
    if s == RailyardEnv.EN_ROUTE_OUT:
        return "En route (out)"
    if s == RailyardEnv.COMPLETED:
        return "Completed"
    return "Waiting"


def _run_scenario_recording(schedule, delay_vector, model, max_steps=None):
    """Run one scenario with the agent; record (step, train_id, location) for every step."""
    max_steps = max_steps or SB_TOTAL_STEPS
    env = RailyardEnv(num_trains=len(schedule), delay_prob=0.0, log_episodes=False)
    obs, _ = env.reset(options={"schedule": schedule, "delay_vector": delay_vector, "max_steps": max_steps})
    timeline = []  # list of (step, train_id, location)
    for step in range(max_steps):
        for t in env.trains:
            loc = _train_location_at_step(t, step)
            timeline.append((step, t["id"], loc))
        if model is not None:
            action, _ = model.predict(obs, deterministic=False)
        else:
            action = env.action_space.sample()
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break
    # one more step so Completed is recorded
    for t in env.trains:
        timeline.append((env.current_step, t["id"], _train_location_at_step(t, env.current_step)))
    return timeline, info


# ---- Trajectory-diagram helpers ----------------------------------------
# Y-axis numeric position for each location (bottom → top = train lifecycle)
_LOC_TO_Y = {loc: i for i, loc in enumerate(LOCATION_ORDER)}

# Distinct per-train line colors (high-contrast palette for up to 10 trains)
_TRAIN_COLORS = [
    "#e74c3c",   # red
    "#f39c12",   # orange
    "#2ecc71",   # green
    "#3498db",   # blue
    "#9b59b6",   # purple
    "#1abc9c",   # teal
    "#e67e22",   # dark orange
    "#e84393",   # pink
    "#00cec9",   # cyan
    "#6c5ce7",   # indigo
]


def _step_to_time_str(step, step_minutes):
    """Convert step index to 'HH:MM' string (0 → 00:00)."""
    total_min = int(step) * step_minutes
    h, m = divmod(total_min, 60)
    return f"{h:02d}:{m:02d}"


def _extract_train_trajectory(timeline, tid):
    """From raw timeline list of (step, train_id, location), extract the
    ordered (step, location) sequence for one train, keeping only transition
    points (first & last step of each contiguous location span) so the line
    stays clean."""
    points = sorted([(s, loc) for (s, t, loc) in timeline if t == tid])
    if not points:
        return []
    # Compress: keep first and last step of each contiguous run
    compressed = []
    run_start = points[0]
    prev = points[0]
    for s, loc in points[1:]:
        if loc == prev[1]:
            prev = (s, loc)  # extend current run
        else:
            # End of run → emit start and end of previous run
            compressed.append(run_start)
            if run_start != prev:
                compressed.append(prev)
            run_start = (s, loc)
            prev = (s, loc)
    compressed.append(run_start)
    if run_start != prev:
        compressed.append(prev)
    return compressed


def _time_series_fig(timeline=None, trains_full=None, timeline_nominal=None,
                     trajectories=None, trajectories_nominal=None,
                     cargo_colors=None, max_steps=None, step_minutes=None):
    """Time-space trajectory diagram.

    Y-axis = railyard locations (Waiting → tracks → Completed, bottom to top).
    X-axis = real time (HH:MM).
    Each train is a coloured line tracing its journey through the yard.

    Accepts EITHER raw *timeline* lists (live simulation) OR pre-extracted
    *trajectories* dicts ``{tid: [(step, loc), ...], ...}`` (pre-computed mode).
    """
    max_steps = max_steps or SB_TOTAL_STEPS
    step_min = step_minutes or SB_STEP_MINUTES
    cargo_colors = cargo_colors or CARGO_COLORS
    train_cargo = {t["id"]: t["cargo"] for t in (trains_full or [])}
    num_trains = len(trains_full or [])

    fig = go.Figure()

    # --- resolve trajectory source (raw timeline vs pre-extracted) -------
    def _get_traj(source_tl, source_precomp, tid):
        """Return [(step, loc), ...] for one train from whichever source."""
        if source_precomp is not None:
            return source_precomp.get(str(tid), source_precomp.get(tid, []))
        if source_tl is not None:
            return _extract_train_trajectory(source_tl, tid)
        return []

    # --- helper to add one train's trajectory ----------------------------
    def _add_trajectory(traj, tid, is_nominal):
        if not traj:
            return
        xs = [s for s, _ in traj]
        ys = [_LOC_TO_Y.get(loc, 0) for _, loc in traj]
        hovers = [
            f"T{tid} {CARGO_NAMES[train_cargo.get(tid, 0)]}<br>"
            f"{loc}<br>{_step_to_time_str(s, step_min)} (step {s})"
            for s, loc in traj
        ]
        col = _TRAIN_COLORS[tid % len(_TRAIN_COLORS)]
        label = f"T{tid} {CARGO_NAMES[train_cargo.get(tid, 0)]}"

        if is_nominal:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=col, width=2, dash="dot"),
                opacity=0.35,
                name=label + " (nominal)",
                legendgroup=f"t{tid}",
                legendgrouptitle_text=label if not is_nominal else None,
                showlegend=True,
                hovertext=hovers, hoverinfo="text",
            ))
        else:
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                line=dict(color=col, width=3),
                marker=dict(size=5, color=col),
                opacity=0.95,
                name=label,
                legendgroup=f"t{tid}",
                showlegend=True,
                hovertext=hovers, hoverinfo="text",
            ))

    # Draw nominal (behind) then DRL (on top)
    has_nominal = (timeline_nominal is not None or trajectories_nominal is not None)
    for tid in range(num_trains):
        if has_nominal:
            traj_n = _get_traj(timeline_nominal, trajectories_nominal, tid)
            _add_trajectory(traj_n, tid, is_nominal=True)
        traj_d = _get_traj(timeline, trajectories, tid)
        _add_trajectory(traj_d, tid, is_nominal=False)

    # --- Axis setup ------------------------------------------------------
    # X-axis: numeric steps but labelled as HH:MM
    hour_ticks = list(range(0, max_steps + 1, 12))  # every hour (12 steps × 5 min)
    hour_labels = [_step_to_time_str(s, step_min) for s in hour_ticks]

    # Y-axis: numeric positions with location labels
    y_vals = list(range(len(LOCATION_ORDER)))
    y_labels = list(LOCATION_ORDER)

    has_overlay = has_nominal
    title = ("Train trajectories through the railyard — "
             "dashed = nominal, solid = delayed + DRL recovery"
             if has_overlay else
             "Train trajectories through the railyard")

    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        xaxis=dict(
            title="Time",
            color=TEXT_COL,
            gridcolor=BORDER,
            range=[0, max_steps],
            tickmode="array",
            tickvals=hour_ticks,
            ticktext=hour_labels,
            minor=dict(tickmode="array",
                       tickvals=[s for s in range(0, max_steps + 1, 6)
                                 if s not in hour_ticks],
                       gridcolor="rgba(127,140,141,0.15)"),
        ),
        yaxis=dict(
            title="Location",
            color=TEXT_COL,
            gridcolor=BORDER,
            tickmode="array",
            tickvals=y_vals,
            ticktext=y_labels,
            range=[-0.5, len(LOCATION_ORDER) - 0.5],
        ),
        margin=dict(l=150, r=20, t=60, b=55),
        height=560,
        font=dict(color=TEXT_COL, size=12),
        title=dict(text=title, font=dict(size=14)),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(size=11),
        ),
        hovermode="closest",
    )
    return fig


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


def _gantt_fig(trains, current_step, max_steps=None):
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
                   range=[0, max_steps or SB_TOTAL_STEPS]),
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


# Schedule & delay scenarios (built on first use; 12h from nominal_schedule.json when present)
_NOMINAL_SCHEDULE = None
_DELAY_VECTORS = None


def _get_nominal_schedule():
    """Return dict with keys: schedule (env-format list), max_steps (int), step_minutes (int).

    Loads from nominal_schedule.json via schedule_builder.  If the file is
    missing, raises an error — run ``python schedule_builder.py`` first.
    """
    global _NOMINAL_SCHEDULE
    if _NOMINAL_SCHEDULE is not None:
        return _NOMINAL_SCHEDULE
    try:
        data = json.load(open("nominal_schedule.json"))
        if isinstance(data, dict) and "schedule" in data:
            env_schedule = schedule_to_env_format(data["schedule"])
            _NOMINAL_SCHEDULE = {
                "schedule": env_schedule,
                "max_steps": data.get("TOTAL_STEPS", SB_TOTAL_STEPS),
                "step_minutes": data.get("STEP_MINUTES", SB_STEP_MINUTES),
            }
            return _NOMINAL_SCHEDULE
    except FileNotFoundError:
        pass
    # Fallback: build via schedule_builder and warn
    from schedule_builder import build_nominal_schedule, save_schedule
    print("  [WARN] nominal_schedule.json not found — generating now ...")
    raw = build_nominal_schedule(num_trains=8, seed=42, check_junctions=False)
    save_schedule(raw)
    env_schedule = schedule_to_env_format(raw)
    _NOMINAL_SCHEDULE = {
        "schedule": env_schedule,
        "max_steps": SB_TOTAL_STEPS,
        "step_minutes": SB_STEP_MINUTES,
    }
    return _NOMINAL_SCHEDULE


def _get_delay_vectors():
    """200 delay scenarios from schedule_builder (same distribution used in training)."""
    global _DELAY_VECTORS
    if _DELAY_VECTORS is None:
        nominal = _get_nominal_schedule()
        sched = nominal["schedule"] if isinstance(nominal, dict) else nominal
        num_trains = len(sched)
        _DELAY_VECTORS = sb_build_delay_vectors(
            n=200, num_trains=num_trains, seed=0, big_delay_prob=0.2,
        )
    return _DELAY_VECTORS


# Pre-computed scenario results (for deployment without torch/SB3)
_PRECOMPUTED = None

def _get_precomputed():
    """Load pre-computed scenario trajectories from JSON if available."""
    global _PRECOMPUTED
    if _PRECOMPUTED is not None:
        return _PRECOMPUTED
    path = os.path.join(os.path.dirname(__file__) or ".", "precomputed_scenarios.json")
    if os.path.isfile(path):
        with open(path) as f:
            _PRECOMPUTED = json.load(f)
        print(f"  Loaded {len(_PRECOMPUTED.get('delayed', []))} pre-computed scenarios")
        return _PRECOMPUTED
    _PRECOMPUTED = {}  # empty dict = not available
    return _PRECOMPUTED


# ====================== GLOBAL STATE ===================================
class _S:
    """Mutable global state (single-user presentation tool)."""
    # Tab 2
    r_env = None
    r_model = None
    r_obs = None
    r_running = False
    r_max_steps = SB_TOTAL_STEPS

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
            html.Div(style={"width": "340px", "padding": "14px",
                             "overflowY": "auto",
                             "borderLeft": f"1px solid {BORDER}",
                             "backgroundColor": PANEL_BG}, children=[
                html.Div("Metrics", style={"fontSize": "12px",
                         "fontWeight": "700", "color": ACCENT,
                         "textTransform": "uppercase",
                         "marginBottom": "10px"}),
                html.Div(id="r-metrics"),
                html.Div("Decisions", style={"fontSize": "13px",
                         "fontWeight": "700", "color": ACCENT,
                         "textTransform": "uppercase",
                         "marginTop": "18px", "marginBottom": "10px"}),
                html.Div(id="r-decisions",
                         style={"maxHeight": "480px", "overflowY": "auto",
                                "fontSize": "14px", "lineHeight": "1.4"}),
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
    schedule = _get_nominal_schedule()
    delay_vecs = _get_delay_vectors()
    options = [{"label": "Scenario 0 (Nominal — no delays)", "value": 0}]
    for i in range(1, len(delay_vecs) + 1):
        options.append({"label": f"Scenario {i} (Delayed)", "value": i})
    return html.Div(children=[
        html.Div(style={"padding": "20px", "backgroundColor": PANEL_BG,
                         "borderBottom": f"1px solid {BORDER}"}, children=[
            html.Div([
                html.Div("Schedule & DRL recovery", style={"fontSize": "18px", "fontWeight": "800", "color": ACCENT, "marginBottom": "8px"}),
                html.P([
                    "We have a nominal railyard schedule: runtimes, loading/unloading times, and mainline slots. "
                    "We create 200 delayed variants by injecting arrival and loading delays. "
                    "Select a delayed scenario and run: you get one graph showing each train's trajectory (enter → inside railyard → leave). "
                    "Nominal run = faded; delayed scenario with DRL recovery = bold — so you see how DRL solved the delays.",
                ], style={"color": TEXT_COL, "fontSize": "14px", "lineHeight": "1.6", "marginBottom": "16px"}),
                html.Div(style={"display": "flex", "gap": "12px", "alignItems": "center", "flexWrap": "wrap"}, children=[
                    html.Div(style={"minWidth": "220px"}, children=[
                        html.Label("Scenario", style={"fontSize": "11px", "color": MUTED, "marginRight": "8px"}),
                        dcc.Dropdown(id="s-scenario", options=options, value=1, clearable=False,
                                      style={"backgroundColor": DARK_BG, "color": TEXT_COL}),
                    ]),
                    html.Button("Run DRL recovery", id="s-run", n_clicks=0, style=_btn_green),
                    html.Div(id="s-status", style={"color": MUTED, "fontSize": "13px"}),
                ]),
            ]),
        ]),
        html.Div(style={"padding": "20px"}, children=[
            html.Div(style={**_card, "marginBottom": "16px"}, children=[
                dcc.Graph(id="s-timeseries", figure=_placeholder_timeseries(),
                          config={"displayModeBar": True, "displaylogo": False},
                          style={"width": "100%"}),
            ]),
            html.Div(id="s-summary", style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
        ]),
    ])


def _placeholder_timeseries():
    fig = go.Figure()
    fig.add_annotation(text="Select a scenario and click ‘Run DRL recovery’ to see the schedule over time.",
                       x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=MUTED))
    fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
                      xaxis=dict(visible=False), yaxis=dict(visible=False), height=520)
    return fig


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
            "Assign the first waiting train to one of 7 tracks (Iron, Pallets, Chalk loading/unloading) or WAIT. "
            "For each assignment the agent also picks runtime and loading within bounds: 0.8× to 1.2× (travel time and "
            "load/unload time). That gives room to speed up or slow down to meet the mainline schedule when there are delays.",
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
        box("Schedule & DRL recovery",
            html.Div([
                "In the \"Schedule & DRL Recovery\" tab we take one nominal schedule and create 200 delayed variants "
                "(arrival and loading delays). The ",
                html.Strong("trained agent", style={"color": "#2ecc71"}),
                " recovers by reordering assignments and using wait strategies. You see: ",
                html.Ul(style={"margin": "8px 0", "paddingLeft": "20px"}, children=[
                    html.Li("A time-series diagram: time on the x-axis, tracks/locations on the y-axis."),
                    html.Li("How each train moves from Waiting to En route (in), track, En route (out), Completed."),
                    html.Li("Recovery metrics: reward, on-time count, conflicts, completed trains."),
                ]),
                "This shows how DRL optimizes against delays in a concrete schedule.",
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
        dcc.Tab(label="Schedule & DRL Recovery", value="tab3",
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
        # Use the canonical 12h schedule with a random delay vector
        nominal = _get_nominal_schedule()
        schedule = nominal["schedule"]
        max_steps = nominal["max_steps"]
        delay_vecs = _get_delay_vectors()
        delay_idx = np.random.randint(0, len(delay_vecs))
        delay_vector = delay_vecs[delay_idx]

        env = RailyardEnv(num_trains=len(schedule), delay_prob=0.0, log_episodes=False)
        obs, _ = env.reset(options={
            "schedule": schedule,
            "delay_vector": delay_vector,
            "max_steps": max_steps,
        })
        S.r_env = env
        S.r_obs = obs
        S.r_model = _load_model()
        S.r_running = True
        S.r_max_steps = max_steps
        return False, speed or 150, f"Running (delay scenario {delay_idx + 1}) ..."
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
    max_s = S.r_max_steps
    gantt = _gantt_fig(trains, info["step"], max_steps=max_s)
    metrics = _metrics_panel(info, max_steps=max_s)
    decisions = _decision_log(info["decisions"])
    status_txt = (f"Step {info['step']}/{max_s}  |  "
                  f"Reward: {info['total_reward']:.1f}")
    if term:
        status_txt += "  |  EPISODE COMPLETE"
    elif trunc:
        status_txt += "  |  TIME OUT"
    return fig, gantt, metrics, decisions, status_txt


# ---- Tab 3: Schedule & DRL Recovery ------------------------------------
@app.callback(
    [Output("s-timeseries", "figure"), Output("s-summary", "children"),
     Output("s-status", "children")],
    Input("s-run", "n_clicks"),
    State("s-scenario", "value"),
    prevent_initial_call=True,
)
def run_recovery(n_clicks, scenario_ix):
    if scenario_ix is None:
        scenario_ix = 0
    nominal = _get_nominal_schedule()
    schedule = nominal["schedule"] if isinstance(nominal, dict) else nominal
    max_steps = nominal.get("max_steps", SB_TOTAL_STEPS) if isinstance(nominal, dict) else SB_TOTAL_STEPS
    step_minutes = nominal.get("step_minutes", SB_STEP_MINUTES) if isinstance(nominal, dict) else SB_STEP_MINUTES
    num_trains = len(schedule)

    # ---- Try pre-computed results first (for Vercel / no-model deployment)
    pc = _get_precomputed()
    if pc and "delayed" in pc:
        trains_meta = pc.get("trains_meta", [])
        trains_full = [{"id": m["id"], "cargo": m["cargo"]} for m in trains_meta]
        pc_nominal = pc.get("nominal", {})
        if scenario_ix == 0:
            # Show nominal only
            fig = _time_series_fig(
                trains_full=trains_full,
                trajectories=pc_nominal.get("trajectories"),
                max_steps=max_steps, step_minutes=step_minutes,
            )
            info = pc_nominal.get("info", {})
        else:
            idx = min(scenario_ix - 1, len(pc["delayed"]) - 1)
            sc = pc["delayed"][idx]
            fig = _time_series_fig(
                trains_full=trains_full,
                trajectories=sc.get("trajectories"),
                trajectories_nominal=pc_nominal.get("trajectories"),
                max_steps=max_steps, step_minutes=step_minutes,
            )
            info = sc.get("info", {})
    else:
        # ---- Live simulation (local with trained model) ------------------
        zero_delays = [{"arrival_delay": 0, "loading_delay": 0} for _ in range(num_trains)]
        if scenario_ix == 0:
            delay_vector = zero_delays
        else:
            delay_vecs = _get_delay_vectors()
            idx = min(scenario_ix - 1, len(delay_vecs) - 1)
            delay_vector = delay_vecs[idx]
        model = _load_model()
        timeline, info = _run_scenario_recording(schedule, delay_vector, model, max_steps=max_steps)
        timeline_nominal = None
        if scenario_ix >= 1:
            timeline_nominal, _ = _run_scenario_recording(schedule, zero_delays, model, max_steps=max_steps)
        fig = _time_series_fig(timeline=timeline, trains_full=info["trains_full"],
                               timeline_nominal=timeline_nominal,
                               max_steps=max_steps, step_minutes=step_minutes)
    reward = info.get("total_reward", info.get("reward", 0))
    on_time = info.get("on_time", 0)
    conflicts = info.get("conflicts", 0)
    completed = info.get("completed", 0)
    total_tr = info.get("total_trains", num_trains)
    step_done = info.get("step", max_steps)
    summary = html.Div(style={"display": "flex", "gap": "20px", "flexWrap": "wrap"}, children=[
        html.Div(style={**_card, "padding": "12px 20px"}, children=[
            html.Span("Reward ", style={"color": MUTED}), html.Span(f"{reward:.1f}", style={"color": "#2ecc71", "fontWeight": "700"}),
        ]),
        html.Div(style={**_card, "padding": "12px 20px"}, children=[
            html.Span("On-time ", style={"color": MUTED}), html.Span(str(on_time), style={"color": "#2ecc71", "fontWeight": "700"}),
        ]),
        html.Div(style={**_card, "padding": "12px 20px"}, children=[
            html.Span("Conflicts ", style={"color": MUTED}), html.Span(str(conflicts), style={"color": "#e74c3c", "fontWeight": "700"}),
        ]),
        html.Div(style={**_card, "padding": "12px 20px"}, children=[
            html.Span("Completed ", style={"color": MUTED}), html.Span(f"{completed}/{total_tr}", style={"color": TEXT_COL, "fontWeight": "700"}),
        ]),
    ])
    status = f"Scenario {scenario_ix} — {step_done}/{max_steps} steps"
    return fig, summary, status


# ====================== UI BUILDERS ====================================
def _metrics_panel(info, max_steps=None):
    max_s = max_steps or SB_TOTAL_STEPS
    items = [
        ("Step", f"{info['step']} / {max_s}"),
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
    recent = decisions[-28:] if decisions else []
    result_colors = {"assigned": "#2ecc71", "arrived": "#3498db",
                     "departing": "#9b59b6", "on_time": "#2ecc71",
                     "late": "#e74c3c", "conflict": "#e74c3c",
                     "delay": "#f39c12", "occupied": "#e74c3c",
                     "wrong_cargo": "#e74c3c"}
    return html.Div([
        html.Div(style={"padding": "8px 10px", "marginBottom": "6px",
                         "borderRadius": "4px", "backgroundColor": DARK_BG,
                         "borderLeft": f"3px solid "
                                       f"{result_colors.get(d['result'], MUTED)}",
                         "fontSize": "14px"}, children=[
            html.Span(f"[{d['step']:>3}] ", style={"color": MUTED, "fontWeight": "600"}),
            html.Span(f"T{d['train']} ", style={"fontWeight": "700", "fontSize": "15px"}),
            html.Span(d["desc"], style={"fontSize": "14px"}),
        ]) for d in reversed(recent)
    ])


# ====================== MAIN ===========================================
if __name__ == "__main__":
    print("\n  Railyard DRL Visualizer")
    print("  http://127.0.0.1:8051\n")
    app.run(debug=False, port=8051)
