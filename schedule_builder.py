"""
12-hour conflict-free railyard schedule builder.
Uses the same infrastructure as railyard_env (graph, tracks, entries, travel times).
Output: nominal schedule + optional delay vectors for training/visualization.
"""

import json
import numpy as np
import networkx as nx
from pathlib import Path

# Reuse env constants and graph
from railyard_env import RailyardEnv

# ---------------------------------------------------------------------------
# 12-hour time model: 1 step = 5 min -> 144 steps
# ---------------------------------------------------------------------------
STEP_MINUTES = 5
TOTAL_STEPS = 144  # 12 * 60 / 5
MIN_GAP_ENTRY = 5   # min steps between two train entries (avoid queue at entry)
MIN_GAP_TRACK = 1   # min steps between train end and next start on same track
JUNCTION_GAP = 0    # min steps between trains at same junction (0 = no overlap)


def _build_graph():
    G = nx.Graph()
    for n1, n2, w in [
        (11,12,468),(16,12,468),(12,13,1132),(13,91,112),(13,52,176),
        (52,90,2713),(52,102,89),(102,112,427),(112,121,1265),(112,114,49),
        (114,116,624),(116,123,555),(123,121,44),(121,92,75),(114,125,730),
        (125,123,447),(125,116,104),(102,129,1913),(129,93,135),(129,132,375),
        (132,135,171),(135,134,64),(135,152,69),(152,94,900),(94,193,67),
        (193,191,40),(152,154,40),(154,193,900),(154,156,40),(156,160,102),
        (164,96,250),(162,97,300),(160,95,290),(156,191,900),(132,133,100),
        (133,134,9),(133,131,280),(160,164,40),(164,162,100),(131,100,530),
        (131,101,535),(134,142,125),(142,98,1000),(142,99,1000),
    ]:
        G.add_edge(n1, n2, weight=w)
    return G


def _precompute_routes_and_times(graph):
    routes = {}
    travel_times = {}
    speed = RailyardEnv.SPEED
    for entry in RailyardEnv.ENTRY_NODES:
        for idx, track in enumerate(RailyardEnv.TRACKS):
            node = track["node"]
            try:
                p = nx.shortest_path(graph, entry, node, weight="weight")
                d = nx.shortest_path_length(graph, entry, node, weight="weight")
                routes[(entry, idx)] = p
                travel_times[(entry, idx)] = max(1, int(np.ceil(d / speed)))
            except nx.NetworkXNoPath:
                pass
            try:
                p = nx.shortest_path(graph, node, entry, weight="weight")
                d = nx.shortest_path_length(graph, node, entry, weight="weight")
                routes[(idx, entry)] = p
                travel_times[(idx, entry)] = max(1, int(np.ceil(d / speed)))
            except nx.NetworkXNoPath:
                pass
    return routes, travel_times


def _segment_times_along_route(graph, route, speed):
    """Return list of (node, t_start, t_end) in steps for each node on the route (excluding first node at 0)."""
    if not route or len(route) < 2:
        return []
    out = []
    t = 0
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        w = graph.edges[u, v].get("weight", 0)
        seg_steps = max(1, int(np.ceil(w / speed)))
        out.append((v, t, t + seg_steps))
        t += seg_steps
    return out


def _junction_passage_times(routes, travel_times, graph, key_junctions):
    """For each (entry, track_idx) and (track_idx, entry), return list of (junction, t_start, t_end) in steps."""
    speed = RailyardEnv.SPEED
    result = {}
    for (entry, idx), route in routes.items():
        if (entry, idx) not in travel_times:
            continue
        segs = _segment_times_along_route(graph, route, speed)
        result[(entry, idx)] = [(n, a, b) for n, a, b in segs if n in key_junctions]
    for (idx, entry), route in routes.items():
        if (idx, entry) not in travel_times:
            continue
        segs = _segment_times_along_route(graph, route, speed)
        result[(idx, entry)] = [(n, a, b) for n, a, b in segs if n in key_junctions]
    return result


def _track_occupancy_conflict(track_occupancy, start, end, track_idx):
    """Check if [start, end] conflicts with any existing occupancy on track_idx."""
    for (s, e) in track_occupancy.get(track_idx, []):
        if not (end + MIN_GAP_TRACK <= s or e + MIN_GAP_TRACK <= start):
            return True
    return False


def _junction_conflict(junction_occupancy, passage_list, t_entry):
    """passage_list = [(junction, t_start, t_end)] relative to route start. t_entry = when train enters route."""
    for junc, a, b in passage_list:
        abs_a = t_entry + a
        abs_b = t_entry + b
        for (s, e) in junction_occupancy.get(junc, []):
            if not (abs_b + JUNCTION_GAP <= s or e + JUNCTION_GAP <= abs_a):
                return True
    return False


def _add_junction_occupancy(junction_occupancy, passage_list, t_entry):
    for junc, a, b in passage_list:
        abs_a = t_entry + a
        abs_b = t_entry + b
        junction_occupancy.setdefault(junc, []).append((abs_a, abs_b))


def _add_track_occupancy(track_occupancy, track_idx, start, end):
    track_occupancy.setdefault(track_idx, []).append((start, end))


def build_nominal_schedule(num_trains=10, seed=42, check_junctions=False):
    """
    Build one 12-hour schedule. Track-conflict-free always; if check_junctions=True also junction-conflict-free (fewer trains).
    Returns list of dicts: train_id, cargo, entry, arrival, slot_center, track_idx, start_at_track, end_at_track, ...
    """
    rng = np.random.default_rng(seed)
    graph = _build_graph()
    routes, travel_times = _precompute_routes_and_times(graph)
    junction_passage = _junction_passage_times(routes, travel_times, graph, RailyardEnv.KEY_JUNCTIONS) if check_junctions else {}
    op_times = RailyardEnv.OP_TIMES

    # Spread entry times evenly so trains have room to finish by 12h and avoid junction clashes
    max_entry = TOTAL_STEPS - 65
    step = max(10, (max_entry - 0) // max(num_trains, 1))
    entries = []
    for i in range(num_trains):
        cargo = int(rng.integers(0, 3))
        entry_node = int(RailyardEnv.ENTRY_NODES[rng.integers(0, 2)])
        t = min(i * step, max_entry)
        entries.append((i, cargo, entry_node, t))

    # Sort by entry time
    entries.sort(key=lambda x: x[3])

    schedule = []
    track_occupancy = {}
    junction_occupancy = {}

    for train_id, cargo, entry_node, entry_time in entries:
        compatible = [idx for idx, t in enumerate(RailyardEnv.TRACKS) if t["cargo"] == cargo]
        if not compatible:
            continue
        # Try tracks in random order for variety
        rng.shuffle(compatible)
        placed = False
        for track_idx in compatible:
            tt_in = travel_times.get((entry_node, track_idx), 20)
            tt_out = travel_times.get((track_idx, entry_node), 20)
            op = op_times[cargo]
            start_at_track = entry_time + tt_in
            end_at_track = start_at_track + op
            exit_time = end_at_track + tt_out
            if exit_time >= TOTAL_STEPS:
                continue
            if _track_occupancy_conflict(track_occupancy, start_at_track, end_at_track, track_idx):
                continue
            if check_junctions:
                pass_in = junction_passage.get((entry_node, track_idx), [])
                if _junction_conflict(junction_occupancy, pass_in, entry_time):
                    continue
                pass_out = junction_passage.get((track_idx, entry_node), [])
                if _junction_conflict(junction_occupancy, pass_out, end_at_track):
                    continue
            _add_track_occupancy(track_occupancy, track_idx, start_at_track, end_at_track)
            if check_junctions:
                _add_junction_occupancy(junction_occupancy, junction_passage.get((entry_node, track_idx), []), entry_time)
                _add_junction_occupancy(junction_occupancy, junction_passage.get((track_idx, entry_node), []), end_at_track)
            schedule.append({
                "train_id": train_id,
                "cargo": cargo,
                "entry": entry_node,
                "arrival": int(entry_time),
                "slot_center": int(exit_time),
                "track_idx": track_idx,
                "start_at_track": int(start_at_track),
                "end_at_track": int(end_at_track),
                "travel_in_steps": tt_in,
                "travel_out_steps": tt_out,
                "op_steps": op,
            })
            placed = True
            break
        if not placed:
            # Bump entry time and retry once with first compatible track
            entry_time_retry = entry_time + MIN_GAP_ENTRY * 2
            if entry_time_retry < max_entry:
                track_idx = compatible[0]
                tt_in = travel_times.get((entry_node, track_idx), 20)
                tt_out = travel_times.get((track_idx, entry_node), 20)
                op = op_times[cargo]
                start_at_track = entry_time_retry + tt_in
                end_at_track = start_at_track + op
                exit_time = end_at_track + tt_out
                if exit_time < TOTAL_STEPS and not _track_occupancy_conflict(track_occupancy, start_at_track, end_at_track, track_idx):
                    ok = True
                    if check_junctions:
                        pass_in = junction_passage.get((entry_node, track_idx), [])
                        pass_out = junction_passage.get((track_idx, entry_node), [])
                        ok = not _junction_conflict(junction_occupancy, pass_in, entry_time_retry) and not _junction_conflict(junction_occupancy, pass_out, end_at_track)
                    if ok:
                        _add_track_occupancy(track_occupancy, track_idx, start_at_track, end_at_track)
                        if check_junctions:
                            _add_junction_occupancy(junction_occupancy, junction_passage.get((entry_node, track_idx), []), entry_time_retry)
                            _add_junction_occupancy(junction_occupancy, junction_passage.get((track_idx, entry_node), []), end_at_track)
                        schedule.append({
                            "train_id": train_id,
                            "cargo": cargo,
                            "entry": entry_node,
                            "arrival": int(entry_time_retry),
                            "slot_center": int(exit_time),
                            "track_idx": track_idx,
                            "start_at_track": int(start_at_track),
                            "end_at_track": int(end_at_track),
                            "travel_in_steps": tt_in,
                            "travel_out_steps": tt_out,
                            "op_steps": op,
                        })
                        placed = True
    return schedule


def schedule_to_env_format(schedule):
    """Convert to list of dicts with keys expected by env reset(options={'schedule': ...}): arrival, cargo, entry, slot_center."""
    return [
        {
            "arrival": s["arrival"],
            "cargo": s["cargo"],
            "entry": s["entry"],
            "slot_center": s["slot_center"],
        }
        for s in sorted(schedule, key=lambda x: x["train_id"])
    ]


def build_delay_vectors(n=200, num_trains=None, seed=0, big_delay_prob=0.2):
    """
    Build n delay vectors. Each train gets arrival_delay and loading_delay.
    With big_delay_prob we sample a "big" delay (30-90 min in steps: 6-18 steps); else small (0-15 min: 0-3 steps).
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        vec = []
        n_t = num_trains or 10
        for _ in range(n_t):
            if rng.random() < big_delay_prob:
                arrival_delay = int(rng.integers(6, 19))   # 30-90 min
                loading_delay = int(rng.integers(2, 8))    # 10-40 min
            else:
                arrival_delay = int(rng.integers(0, 4))    # 0-15 min
                loading_delay = int(rng.integers(0, 3))    # 0-15 min
            vec.append({"arrival_delay": arrival_delay, "loading_delay": loading_delay})
        out.append(vec)
    return out


def save_schedule(schedule, path="nominal_schedule.json"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Renumber train_id to 0..n-1 and add metadata
    out = [{"train_id": i, **{k: v for k, v in s.items() if k != "train_id"}} for i, s in enumerate(sorted(schedule, key=lambda x: x["train_id"]))]
    payload = {"TOTAL_STEPS": TOTAL_STEPS, "STEP_MINUTES": STEP_MINUTES, "schedule": out}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(out)} services to {path}")


def load_schedule(path="nominal_schedule.json"):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "schedule" in data:
        return data["schedule"]
    return data


if __name__ == "__main__":
    # check_junctions=False -> track conflict-free only (more trains); True -> full conflict-free (fewer trains)
    schedule = build_nominal_schedule(num_trains=8, seed=42, check_junctions=False)
    print(f"Built schedule: {len(schedule)} trains over {TOTAL_STEPS} steps (12h, 1 step={STEP_MINUTES} min)")
    for s in schedule:
        print(f"  T{s['train_id']}: cargo={s['cargo']} entry={s['arrival']} track={s['track_idx']} "
              f"at_track=[{s['start_at_track']},{s['end_at_track']}] exit={s['slot_center']}")
    save_schedule(schedule)
    env_format = schedule_to_env_format(schedule)
    print(f"Env format: {len(env_format)} entries")
    delays = build_delay_vectors(5, num_trains=len(schedule), seed=0, big_delay_prob=0.3)
    print(f"Delay vectors: {len(delays)} samples, first vector: {delays[0]}")
