"""
Railyard DRL Environment
========================
A Gymnasium environment for training DRL agents to optimize railway scheduling.

Trains arrive at mainline entry points and must be routed to compatible
loading/unloading tracks (Iron, Pallets, Chalk), then return to meet
their departure slot windows -- all while avoiding junction conflicts
and coping with random loading delays.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx

# ---------------------------------------------------------------------------
# Shared episode log -- the training script reads this after each phase
# ---------------------------------------------------------------------------
_episode_log = []


class RailyardEnv(gym.Env):
    """Railyard scheduling environment for Deep Reinforcement Learning."""

    metadata = {"render_modes": ["human"]}

    # ---- cargo types ----
    IRON = 0
    PALLETS = 1
    CHALK = 2
    CARGO_NAMES = ["Iron", "Pallets", "Chalk"]
    CARGO_COLORS = ["#e74c3c", "#f39c12", "#3498db"]

    # ---- track definitions (node, cargo, operation, display name) ----
    TRACKS = [
        {"node": 98,  "cargo": 0, "op": "loading",   "name": "Iron Loading A"},
        {"node": 100, "cargo": 0, "op": "unloading", "name": "Iron Unloading"},
        {"node": 99,  "cargo": 1, "op": "loading",   "name": "Pallets Loading"},
        {"node": 101, "cargo": 1, "op": "unloading", "name": "Pallets Unloading"},
        {"node": 95,  "cargo": 2, "op": "loading",   "name": "Chalk Loading A"},
        {"node": 96,  "cargo": 2, "op": "loading",   "name": "Chalk Loading B"},
        {"node": 97,  "cargo": 2, "op": "unloading", "name": "Chalk Unloading"},
    ]
    NUM_TRACKS = 7

    # ---- entry / exit points (mainline connection) ----
    ENTRY_NODES = [11, 16]
    ENTRY_NAMES = {11: "North Entry", 16: "South Entry"}

    # ---- operation times per cargo type (timesteps) ----
    OP_TIMES = {0: 15, 1: 10, 2: 8}

    # ---- key junction nodes for conflict detection ----
    KEY_JUNCTIONS = {12, 13, 52, 102, 112, 121, 129, 134, 142}

    # ---- train status codes ----
    WAITING = 0
    EN_ROUTE_IN = 1
    OPERATING = 2
    EN_ROUTE_OUT = 3
    COMPLETED = 4
    STATUS_NAMES = [
        "Waiting", "En Route In", "Loading/Unloading", "En Route Out", "Completed"
    ]

    MAX_TRAINS = 8
    MAX_STEPS = 200
    SPEED = 250.0  # distance-units per timestep

    # ------------------------------------------------------------------
    def __init__(self, num_trains=6, delay_prob=0.2,
                 render_mode=None, log_episodes=True):
        super().__init__()
        self.num_trains = min(num_trains, self.MAX_TRAINS)
        self.delay_prob = delay_prob
        self.render_mode = render_mode
        self.log_episodes = log_episodes

        self.graph = self._build_graph()
        self._precompute_routes()
        self._episode_max_steps = self.MAX_STEPS  # can be overridden in reset(options={"max_steps": N})

        # action: (track_or_wait, runtime_option, loading_option)
        # track_or_wait: 0..6 = assign to track, 7 = wait
        # runtime_option: 0=0.8x, 1=1.0x, 2=1.2x travel time (within bounds)
        # loading_option: 0=0.8x, 1=1.0x, 2=1.2x loading time (within bounds)
        # Agent has room to speed up or slow down to meet mainline schedule under delays
        self.action_space = spaces.MultiDiscrete([self.NUM_TRACKS + 1, 3, 3])

        # observation: tracks(7*3) + trains(8*5) + time(1) = 62
        obs_dim = self.NUM_TRACKS * 3 + self.MAX_TRAINS * 5 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    # ---------------------- graph helpers --------------------------------
    def _build_graph(self):
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

    def _precompute_routes(self):
        self.routes = {}
        self.travel_times = {}
        for entry in self.ENTRY_NODES:
            for idx, track in enumerate(self.TRACKS):
                node = track["node"]
                # inbound
                try:
                    p = nx.shortest_path(self.graph, entry, node, weight="weight")
                    d = nx.shortest_path_length(self.graph, entry, node, weight="weight")
                    self.routes[(entry, idx)] = p
                    self.travel_times[(entry, idx)] = max(1, int(np.ceil(d / self.SPEED)))
                except nx.NetworkXNoPath:
                    pass
                # outbound (return)
                try:
                    p = nx.shortest_path(self.graph, node, entry, weight="weight")
                    d = nx.shortest_path_length(self.graph, node, entry, weight="weight")
                    self.routes[(idx, entry)] = p
                    self.travel_times[(idx, entry)] = max(1, int(np.ceil(d / self.SPEED)))
                except nx.NetworkXNoPath:
                    pass

    # ---------------------- reset / obs / info --------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        opts = options or {}
        self._episode_max_steps = int(opts["max_steps"]) if opts.get("max_steps") is not None else self.MAX_STEPS

        self.track_state = np.zeros((self.NUM_TRACKS, 3), dtype=np.float32)
        self.trains = []

        schedule = opts.get("schedule")
        delay_vector = opts.get("delay_vector", [])

        if schedule is not None and len(schedule) >= 1:
            # Fixed schedule: list of {arrival, cargo, entry, slot_center}
            for i, row in enumerate(schedule):
                if i >= self.num_trains:
                    break
                arrival = int(row["arrival"])
                cargo = int(row["cargo"])
                entry = int(row["entry"])
                slot_center = int(row["slot_center"])
                # apply arrival delay if provided
                if i < len(delay_vector) and "arrival_delay" in delay_vector[i]:
                    arrival = min(arrival + int(delay_vector[i]["arrival_delay"]), self._episode_max_steps - 1)
                loading_delay = 0
                if i < len(delay_vector) and "loading_delay" in delay_vector[i]:
                    loading_delay = int(delay_vector[i]["loading_delay"])
                self.trains.append({
                    "id": i, "cargo": cargo, "entry": entry,
                    "arrival": arrival, "slot_center": slot_center,
                    "slot_window": 8,
                    "status": self.WAITING, "assigned_track": -1,
                    "travel_remaining": 0, "op_remaining": 0,
                    "return_remaining": 0,
                    "total_travel_in": 0, "total_travel_out": 0,
                    "route_in": [], "route_out": [],
                    "delay": 0, "current_node": entry,
                    "completed_step": -1,
                    "loading_delay": loading_delay,
                    "loading_delay_applied": False,
                })
        else:
            # Random schedule (original behaviour)
            for i in range(self.num_trains):
                cargo = int(self.np_random.integers(0, 3))
                entry = self.ENTRY_NODES[int(self.np_random.integers(0, 2))]
                arrival = int(self.np_random.integers(max(0, i * 5),
                                                       min(15 + i * 10, 80)))
                compat = [j for j, t in enumerate(self.TRACKS) if t["cargo"] == cargo]
                if compat:
                    tt_in = self.travel_times.get((entry, compat[0]), 20)
                    tt_out = self.travel_times.get((compat[0], entry), 20)
                else:
                    tt_in = tt_out = 20
                op_t = self.OP_TIMES[cargo]
                slot_center = arrival + tt_in + op_t + tt_out + int(
                    self.np_random.integers(5, 20))
                slot_center = min(slot_center, self._episode_max_steps - 5)
                self.trains.append({
                    "id": i, "cargo": cargo, "entry": entry,
                    "arrival": arrival, "slot_center": slot_center,
                    "slot_window": 8,
                    "status": self.WAITING, "assigned_track": -1,
                    "travel_remaining": 0, "op_remaining": 0,
                    "return_remaining": 0,
                    "total_travel_in": 0, "total_travel_out": 0,
                    "route_in": [], "route_out": [],
                    "delay": 0, "current_node": entry,
                    "completed_step": -1,
                })

        self.current_step = 0
        self.total_reward = 0.0
        self.conflicts = 0
        self.on_time = 0
        self.late = 0
        self.decisions = []
        self._conflict_pairs = set()

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        o = []
        for i in range(self.NUM_TRACKS):
            o.append(self.track_state[i, 0])
            o.append(self.track_state[i, 1] / 3.0)
            o.append(min(self.track_state[i, 2] / self._episode_max_steps, 1.0))
        for i in range(self.MAX_TRAINS):
            if i < len(self.trains):
                t = self.trains[i]
                o.append(1.0 if t["arrival"] <= self.current_step else 0.0)
                o.append((t["cargo"] + 1) / 4.0)
                o.append(t["status"] / 4.0)
                o.append(min(t["slot_center"] / self._episode_max_steps, 1.0))
                o.append(min(t["delay"] / 50.0, 1.0))
            else:
                o.extend([0.0] * 5)
        o.append(self.current_step / self._episode_max_steps)
        return np.array(o, dtype=np.float32)

    def _get_info(self):
        completed = sum(1 for t in self.trains if t["status"] == self.COMPLETED)
        return {
            "step": self.current_step,
            "conflicts": self.conflicts,
            "on_time": self.on_time,
            "late": self.late,
            "completed": completed,
            "total_trains": self.num_trains,
            "total_reward": self.total_reward,
            "decisions": list(self.decisions),
            "trains_full": [
                {k: (list(v) if isinstance(v, list) else v)
                 for k, v in t.items()}
                for t in self.trains
            ],
        }

    def _first_waiting_train(self):
        for t in self.trains:
            if t["status"] == self.WAITING and t["arrival"] <= self.current_step:
                return t
        return None

    # ---------------------- step ----------------------------------------
    def _parse_action(self, action):
        """Support MultiDiscrete [track, runtime_opt, loading_opt] and legacy Discrete(8)."""
        a = np.asarray(action).flatten()
        if a.size == 0:
            return 7, 1, 1
        track = int(a[0]) if a.size > 0 else 7
        rt_opt = int(a[1]) if a.size > 1 else 1
        ld_opt = int(a[2]) if a.size > 2 else 1
        rt_opt = max(0, min(2, rt_opt))
        ld_opt = max(0, min(2, ld_opt))
        return track, rt_opt, ld_opt

    def step(self, action):
        track, rt_opt, ld_opt = self._parse_action(action)
        reward = 0.0
        step_dec = []
        wt = self._first_waiting_train()

        # Multipliers: 0=0.8x, 1=1.0x, 2=1.2x (within upper bounds)
        rt_mult = 0.8 + 0.4 * rt_opt
        ld_mult = 0.8 + 0.4 * ld_opt

        # ---- 1. process action ----
        if track < self.NUM_TRACKS and wt is not None:
            trk = self.TRACKS[track]
            if self.track_state[track, 0] == 1.0:
                reward -= 1.0
                step_dec.append(dict(step=self.current_step, train=wt["id"],
                    action=track, result="occupied",
                    desc=f"Track {trk['name']} occupied"))
            elif trk["cargo"] != wt["cargo"]:
                reward -= 1.5
                step_dec.append(dict(step=self.current_step, train=wt["id"],
                    action=track, result="wrong_cargo",
                    desc=f"Wrong cargo: {self.CARGO_NAMES[wt['cargo']]} -> {trk['name']}"))
            else:
                reward += 8.0
                wt["status"] = self.EN_ROUTE_IN
                wt["assigned_track"] = track
                key = (wt["entry"], track)
                tt_base = self.travel_times.get(key, 20)
                tt = max(1, int(np.ceil(tt_base * rt_mult)))
                rt = self.routes.get(key, [])
                wt["travel_remaining"] = tt
                wt["total_travel_in"] = tt
                wt["route_in"] = list(rt)
                op_base = self.OP_TIMES[wt["cargo"]]
                wt["op_remaining"] = max(1, int(np.ceil(op_base * ld_mult)))
                self.track_state[track] = [1.0, float(wt["cargo"] + 1),
                                            float(tt + wt["op_remaining"])]
                step_dec.append(dict(step=self.current_step, train=wt["id"],
                    action=track, result="assigned",
                    desc=f"{self.CARGO_NAMES[wt['cargo']]} train -> {trk['name']} (rt×{rt_mult:.1f}, ld×{ld_mult:.1f})"))
        elif track == self.NUM_TRACKS and wt is not None:
            # Waiting is ALWAYS the worst option when a free compatible track exists
            has_opt = any(self.TRACKS[j]["cargo"] == wt["cargo"]
                         and self.track_state[j, 0] == 0
                         for j in range(self.NUM_TRACKS))
            if has_opt:
                reward -= 3.0
            else:
                reward -= 0.5  # mild cost even when justified

        # ---- 2. advance trains ----
        en_route = []
        for t in self.trains:
            if t["status"] == self.EN_ROUTE_IN:
                t["travel_remaining"] -= 1
                if t["route_in"] and t["total_travel_in"] > 0:
                    prog = 1.0 - t["travel_remaining"] / t["total_travel_in"]
                    idx = min(int(prog * len(t["route_in"])),
                              len(t["route_in"]) - 1)
                    t["current_node"] = t["route_in"][idx]
                if t["travel_remaining"] <= 0:
                    t["status"] = self.OPERATING
                    t["current_node"] = self.TRACKS[t["assigned_track"]]["node"]
                    reward += 3.0
                    step_dec.append(dict(step=self.current_step, train=t["id"],
                        action=t["assigned_track"], result="arrived",
                        desc=f"Arrived at {self.TRACKS[t['assigned_track']]['name']}"))
                else:
                    en_route.append(t)

            elif t["status"] == self.OPERATING:
                # apply fixed loading delay from scenario (once per train)
                if not t.get("loading_delay_applied", True) and t.get("loading_delay", 0) > 0:
                    extra = t["loading_delay"]
                    t["op_remaining"] += extra
                    t["delay"] += extra
                    t["loading_delay_applied"] = True
                    step_dec.append(dict(step=self.current_step, train=t["id"],
                        action=t["assigned_track"], result="delay",
                        desc=f"Injected delay +{extra} steps"))
                elif self.delay_prob > 0 and self.np_random.random() < self.delay_prob * 0.15:
                    extra = int(self.np_random.integers(1, 4))
                    t["op_remaining"] += extra
                    t["delay"] += extra
                    step_dec.append(dict(step=self.current_step, train=t["id"],
                        action=t["assigned_track"], result="delay",
                        desc=f"Random delay +{extra} steps"))
                t["op_remaining"] -= 1
                if t["op_remaining"] <= 0:
                    t["status"] = self.EN_ROUTE_OUT
                    rk = (t["assigned_track"], t["entry"])
                    tt = self.travel_times.get(rk, 20)
                    rt = self.routes.get(rk, [])
                    t["return_remaining"] = tt
                    t["total_travel_out"] = tt
                    t["route_out"] = list(rt)
                    self.track_state[t["assigned_track"]] = [0, 0, 0]
                    step_dec.append(dict(step=self.current_step, train=t["id"],
                        action=t["assigned_track"], result="departing",
                        desc=f"Leaving {self.TRACKS[t['assigned_track']]['name']}"))

            elif t["status"] == self.EN_ROUTE_OUT:
                t["return_remaining"] -= 1
                if t["route_out"] and t["total_travel_out"] > 0:
                    prog = 1.0 - t["return_remaining"] / t["total_travel_out"]
                    idx = min(int(prog * len(t["route_out"])),
                              len(t["route_out"]) - 1)
                    t["current_node"] = t["route_out"][idx]
                if t["return_remaining"] <= 0:
                    t["status"] = self.COMPLETED
                    t["completed_step"] = self.current_step
                    t["current_node"] = t["entry"]
                    diff = self.current_step - t["slot_center"]
                    if abs(diff) <= t["slot_window"]:
                        reward += 15.0
                        self.on_time += 1
                        step_dec.append(dict(step=self.current_step,
                            train=t["id"], action=-1, result="on_time",
                            desc=f"ON TIME (slot {t['slot_center']}+/-{t['slot_window']})"))
                    else:
                        lateness = max(0, diff - t["slot_window"])
                        reward -= lateness * 0.5
                        self.late += 1
                        step_dec.append(dict(step=self.current_step,
                            train=t["id"], action=-1, result="late",
                            desc=f"LATE by {max(1,int(lateness))} steps"))
                else:
                    en_route.append(t)

        # ---- 3. junction conflicts ----
        for i in range(len(en_route)):
            for j in range(i + 1, len(en_route)):
                a, b = en_route[i], en_route[j]
                if (a["current_node"] == b["current_node"]
                        and a["current_node"] in self.KEY_JUNCTIONS):
                    pair = (min(a["id"], b["id"]), max(a["id"], b["id"]))
                    if pair not in self._conflict_pairs:
                        self._conflict_pairs.add(pair)
                        self.conflicts += 1
                        reward -= 3.0
                        later = b
                        if later["status"] == self.EN_ROUTE_IN:
                            later["travel_remaining"] += 3
                            later["delay"] += 3
                        elif later["status"] == self.EN_ROUTE_OUT:
                            later["return_remaining"] += 3
                            later["delay"] += 3
                        step_dec.append(dict(step=self.current_step,
                            train=later["id"], action=-1, result="conflict",
                            desc=f"Conflict at junction {a['current_node']}"))

        # ---- 4. per-step incentives ----
        for t in self.trains:
            if t["status"] in (self.EN_ROUTE_IN, self.OPERATING, self.EN_ROUTE_OUT):
                reward += 0.05  # small bonus for keeping trains active
                if self.current_step > t["slot_center"] + t["slot_window"]:
                    reward -= 0.15  # late penalty outweighs activity bonus

        # ---- bookkeeping ----
        for i in range(self.NUM_TRACKS):
            if self.track_state[i, 2] > 0:
                self.track_state[i, 2] -= 1.0

        self.decisions.extend(step_dec)
        self.current_step += 1
        self.total_reward += reward

        all_done = all(t["status"] == self.COMPLETED for t in self.trains)
        truncated = self.current_step >= self._episode_max_steps
        terminated = all_done

        if terminated:
            reward += 30.0
            self.total_reward += 30.0
        elif truncated:
            inc = sum(1 for t in self.trains if t["status"] != self.COMPLETED)
            penalty = inc * 8.0
            reward -= penalty
            self.total_reward -= penalty

        info = self._get_info()

        if (terminated or truncated) and self.log_episodes:
            _episode_log.append({
                "reward": round(self.total_reward, 2),
                "length": self.current_step,
                "on_time": self.on_time,
                "late": self.late,
                "conflicts": self.conflicts,
                "completed": info["completed"],
                "total_trains": self.num_trains,
                "on_time_pct": round(self.on_time / max(1, self.num_trains) * 100, 1),
                "avg_delay": round(float(np.mean([t["delay"] for t in self.trains])), 2),
            })

        return self._get_obs(), reward, terminated, truncated, info

    # ---- helpers exposed for the visualiser ----
    def compatible_tracks(self, cargo):
        """Return indices of tracks compatible with a cargo type."""
        return [i for i, t in enumerate(self.TRACKS) if t["cargo"] == cargo]
