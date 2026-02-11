# Plan: 12-Hour Railyard Schedule + DRL Recovery

## Implementation status

- **Schedule builder** (`schedule_builder.py`): Builds a 12h nominal schedule (144 steps, 1 step = 5 min). **Track conflict-free** by default (`check_junctions=False` → 8 trains); set `check_junctions=True` for full junction conflict-free (fewer trains). Saves to `nominal_schedule.json` with `TOTAL_STEPS`, `STEP_MINUTES`, and `schedule`. Also provides `build_delay_vectors()` with small + big delay distributions and `schedule_to_env_format()` for env consumption.
- **Env** (`railyard_env.py`): `reset(options={"schedule": ..., "delay_vector": ..., "max_steps": 144})` uses per-episode `max_steps` for truncation and observation scaling. Action space `MultiDiscrete([8, 3, 3])`. Loading delay applied once per train via `loading_delay` / `loading_delay_applied` fields.
- **Training** (`train_agent.py`): `ScheduleEnv` wrapper loads `nominal_schedule.json` and cycles through 500 delay vectors (from `schedule_builder.build_delay_vectors`). Every episode = same 8-train schedule + different delay scenario (with 10% no-delay episodes). PPO 200k steps, saves best + final model. Baseline (100 ep random) and evaluation (100 ep agent) also use the schedule wrapper.
- **Visualizer** (`drl_visualizer.py`): All four tabs use the canonical schedule. Tab 1: training charts. Tab 2: live replay with nominal schedule + random delay vector from the same 200-delay pool. Tab 3: scenario picker (0 = nominal, 1–200 = delayed) with time–space overlay (nominal faded, delayed+DRL bold). Tab 4: educational. Schedule and delay functions are imported from `schedule_builder` — no duplicate generators in the visualizer.

---

## Goal

Redo the railyard DRL system around a **single conflict-free 12-hour schedule** aligned to existing infrastructure. Add **delays (including large ones)** that require rearranging trains. Train the RL agent with the current action space. Visualize with a **time–space graph** (time on x-axis, tracks on y-axis).

---

## 1. Time model

- **Horizon**: 12 hours of operations.
- **Discretization**: Choose one:
  - **Option A**: 1 step = 1 minute → 720 steps per run (longer training, finer control).
  - **Option B**: 1 step = 5 minutes → 144 steps (faster, coarser).
  - **Option C**: 1 step = 2 minutes → 360 steps (compromise).
- **Recommendation**: Start with **Option B (144 steps)** for faster iteration; later move to Option C or A if needed.
- **Constants**: Keep existing infra (7 tracks, 2 entries, graph, travel times, op times). Convert travel/op times from “steps” to the new time unit (e.g. if 1 step = 5 min, then a 20-step travel becomes 100 min, or scale speeds).

---

## 2. Nominal schedule (conflict-free, 12 hours)

- **Definition**: A list of **train services** over the 12-hour window. Each service has:
  - **Train id**
  - **Cargo type** (Iron / Pallets / Chalk)
  - **Entry** (North 11 / South 16)
  - **Planned entry time** (minutes from 0 or step index)
  - **Track** (one of the 7 tracks, compatible with cargo)
  - **Planned start at track** (when it starts loading/unloading)
  - **Planned end at track** (when it finishes)
  - **Planned exit time** (when it returns to mainline)
- **Construction**:
  1. Use existing graph and routes to get **travel_in**, **travel_out**, and **op_time** per (entry, track) and cargo.
  2. Generate a **fixed number of trains** (e.g. 8–12) spread across the 12 hours so that:
     - No two trains occupy the **same track** at the same time (track conflict-free).
     - No two trains are at the **same junction** at the same time (junction conflict-free), using existing `KEY_JUNCTIONS` and route node sequences.
  3. Output: **nominal schedule** = ordered list of (train_id, entry_time, track, start_at_track, end_at_track, exit_time, cargo, entry_node).
- **Storage**: One canonical schedule (e.g. JSON or Python dict) used by env and visualizer. Optionally one file per 12h schedule variant.

---

## 3. Delays

- **Small delays**: +5–15 min (e.g. late arrival or extra loading). May not require reordering.
- **Big delays**: +30–90 min (e.g. breakdown, crew change). Often require **rearranging** which train uses which track and when, to avoid conflicts and meet mainline slots.
- **Implementation**: When building a “delayed scenario”:
  - Take the nominal schedule.
  - For each train, sample **arrival_delay** and **loading_delay** (e.g. from different distributions for “small” vs “big”).
  - Apply delays to entry time and/or to op end time; recompute downstream times only for display/analysis. In the **env**, delays are applied at runtime (late arrival = train appears later; extra loading = longer op_remaining when at track).
- **Scenarios**: e.g. 50–200 delayed variants (some with mostly small delays, some with 1–2 big delays). Same nominal schedule, different delay vectors.

---

## 4. Environment (align with schedule + delays)

- **Episode**: One 12-hour window (e.g. 144 steps if 1 step = 5 min). Same nominal schedule structure; delays injected per scenario.
- **Reset**: `reset(options={"schedule": nominal_schedule, "delay_vector": [...]})`. Build train list from schedule; apply delay_vector to entry times and (optionally) loading times. No random train generation; schedule is fixed.
- **State**: As now (track occupancy, train status, times, delays, etc.), possibly with more slots if we have more trains. Time in state = current step (or clock time in min).
- **Action space**: Keep current **MultiDiscrete([8, 3, 3])**: assign first waiting train to one of 7 tracks or wait; runtime and loading within bounds (0.8×–1.2×). Agent can **reorder** by choosing when to assign which train and with which runtime/loading.
- **Reward**: Emphasize on-time exit, conflict avoidance, and completing as many trains as possible within the 12-hour window. Penalize conflicts and late departures.
- **Conflicts**: Same as now: track conflict (two trains same track) and junction conflict (two trains same junction node at same time). Nominal schedule has none; delays can introduce conflicts if the agent does not reschedule well.

---

## 5. Training

- **Data**: Many episodes with the **same nominal schedule** and **different delay vectors** (sample from a delay generator). Optionally mix in “no delay” episodes.
- **Algorithm**: PPO (or current algorithm) with the same action/state as above.
- **Goal**: Agent learns to reassign and reorder under delays so that the executed schedule stays conflict-free and meets mainline slots as well as possible.

---

## 6. Time–space graph (visualization)

- **One main graph**:
  - **X-axis**: **Time** (0 to 12h or 0 to max_step), in steps or minutes.
  - **Y-axis**: **Tracks** (and optionally “Waiting” / “En route” / “Completed”). Use **all** track names from the infra (e.g. 7 tracks + Waiting + En route in + En route out + Completed, or a flatter list of “locations”).
- **Content**: For each train, show **when** it is **where**:
  - Horizontal segments: train on a given track (or state) from time A to time B. **Stopped at track** = solid/bold segment; **moving** (en route) = lighter segment or separate “En route in/out” rows.
- **Nominal vs delayed + DRL**: Overlay nominal run (faded) and delayed + DRL run (bold) on the **same** time–space graph so we see how the agent shifted trains in time and across tracks.
- **Hover**: Train id, location, start time, end time, duration.

---

## 7. Implementation order

| Phase | Task | Output |
|-------|------|--------|
| 1 | **Time model** | Choose step length (e.g. 5 min). Define `MAX_STEPS = 144`, scale speeds/times if needed. | `railyard_env.py` constants |
| 2 | **Schedule builder** | Script that generates one conflict-free 12h schedule from graph + 7 tracks + 2 entries. Check track and junction occupancy. | `schedule_builder.py` + `nominal_schedule.json` (or .py) |
| 3 | **Delay generator** | Given nominal schedule, produce delay vectors (small and big). | In `schedule_builder.py` or `drl_visualizer.py` |
| 4 | **Env adaptation** | Env loads nominal schedule; applies delay_vector; runs 12h (MAX_STEPS); state/action/reward unchanged in spirit. | `railyard_env.py` |
| 5 | **Training** | Train on many delay scenarios (same schedule, varying delays). | `train_agent.py` + `training_output/` |
| 6 | **Time–space graph** | Single graph: x=time, y=tracks; segments per train; optional nominal vs delayed overlay. | `drl_visualizer.py` |
| 7 | **App flow** | Select scenario (nominal or delayed variant) → Run recovery → Show time–space graph + metrics. | `drl_visualizer.py` tabs |

---

## 8. File changes summary

- **New**: `schedule_builder.py` — build conflict-free 12h schedule; optional delay vector generation.
- **New**: `nominal_schedule.json` (or embedded in code) — single canonical schedule.
- **Modify**: `railyard_env.py` — 12h horizon, load schedule from file/options, apply delays, keep action/state.
- **Modify**: `train_agent.py` — use schedule + delay sampling for training.
- **Modify**: `drl_visualizer.py` — time–space diagram (x=time, y=tracks), nominal vs delayed overlay, scenario selection.
- **Keep**: `railyard_app.py` — unchanged (geometry only).

---

## 9. Success criteria

- One **conflict-free** nominal 12-hour schedule that uses the existing 7 tracks and 2 entries.
- Delayed scenarios that sometimes require **rearranging** trains (big delays).
- RL agent **trained** on these scenarios; improves over baseline (e.g. random or no reschedule).
- **Time–space graph** clearly shows time on x-axis, tracks (and states) on y-axis, with train trajectories and optional nominal vs DRL overlay.

This plan keeps the current action space and infra and focuses the redo on schedule design, delay model, and the time–space visualization.
