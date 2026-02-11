"""
Pre-compute scenario results for deployment without torch/SB3.
Runs the trained model on all 201 scenarios (nominal + 200 delayed)
and saves compressed trajectory data to precomputed_scenarios.json.

Usage:
    python precompute_scenarios.py
"""

import json
import os
import sys

from schedule_builder import (
    load_schedule, schedule_to_env_format, build_delay_vectors,
    TOTAL_STEPS, STEP_MINUTES,
)
from railyard_env import RailyardEnv

# Import the trajectory extractor and scenario runner from the visualizer
from drl_visualizer import (
    _extract_train_trajectory, _run_scenario_recording, _load_model,
    CARGO_NAMES,
)


def main():
    print("=" * 60)
    print("  Pre-computing scenario results for web deployment")
    print("=" * 60)

    # Load schedule
    schedule_raw = load_schedule("nominal_schedule.json")
    env_schedule = schedule_to_env_format(schedule_raw)
    num_trains = len(env_schedule)
    print(f"\n  Schedule: {num_trains} trains, {TOTAL_STEPS} steps")

    # Load trained model
    model = _load_model()
    if model is None:
        print("  [WARN] No trained model found — using random actions")
    else:
        print("  Model loaded OK")

    # Build delay vectors (same as training/visualizer)
    delay_vectors = build_delay_vectors(
        n=200, num_trains=num_trains, seed=0, big_delay_prob=0.2,
    )
    print(f"  Delay vectors: {len(delay_vectors)} scenarios")

    # Train metadata
    trains_meta = []
    env = RailyardEnv(num_trains=num_trains, delay_prob=0.0, log_episodes=False)
    env.reset(options={"schedule": env_schedule, "max_steps": TOTAL_STEPS})
    for t in env.trains:
        trains_meta.append({"id": t["id"], "cargo": t["cargo"]})

    # --- Run nominal (no delays) -----------------------------------------
    print("\n  Running nominal scenario (0 delays) ...")
    zero_delays = [{"arrival_delay": 0, "loading_delay": 0}
                   for _ in range(num_trains)]
    tl_nom, info_nom = _run_scenario_recording(
        env_schedule, zero_delays, model, max_steps=TOTAL_STEPS,
    )
    nom_trajs = {}
    for tid in range(num_trains):
        nom_trajs[str(tid)] = _extract_train_trajectory(tl_nom, tid)

    nominal_data = {
        "trajectories": nom_trajs,
        "info": {
            "reward": round(info_nom.get("total_reward", 0), 1),
            "on_time": info_nom.get("on_time", 0),
            "conflicts": info_nom.get("conflicts", 0),
            "completed": info_nom.get("completed", 0),
            "total_trains": info_nom.get("total_trains", num_trains),
            "step": info_nom.get("step", TOTAL_STEPS),
        },
    }
    print(f"    Reward={nominal_data['info']['reward']:.1f}  "
          f"On-time={nominal_data['info']['on_time']}  "
          f"Conflicts={nominal_data['info']['conflicts']}")

    # --- Run all 200 delayed scenarios -----------------------------------
    delayed_data = []
    print(f"\n  Running {len(delay_vectors)} delayed scenarios ...")
    for i, dv in enumerate(delay_vectors):
        tl, info = _run_scenario_recording(
            env_schedule, dv, model, max_steps=TOTAL_STEPS,
        )
        trajs = {}
        for tid in range(num_trains):
            trajs[str(tid)] = _extract_train_trajectory(tl, tid)
        delayed_data.append({
            "trajectories": trajs,
            "info": {
                "reward": round(info.get("total_reward", 0), 1),
                "on_time": info.get("on_time", 0),
                "conflicts": info.get("conflicts", 0),
                "completed": info.get("completed", 0),
                "total_trains": info.get("total_trains", num_trains),
                "step": info.get("step", TOTAL_STEPS),
            },
        })
        if (i + 1) % 50 == 0:
            print(f"    ... {i + 1}/{len(delay_vectors)}")

    # --- Save -------------------------------------------------------------
    payload = {
        "trains_meta": trains_meta,
        "max_steps": TOTAL_STEPS,
        "step_minutes": STEP_MINUTES,
        "nominal": nominal_data,
        "delayed": delayed_data,
    }

    out_path = "precomputed_scenarios.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))  # compact
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\n  Saved to {out_path} ({size_kb:.0f} KB)")

    # Summary stats
    rewards = [d["info"]["reward"] for d in delayed_data]
    ontimes = [d["info"]["on_time"] for d in delayed_data]
    print(f"  Avg reward across delayed: {sum(rewards)/len(rewards):.1f}")
    print(f"  Avg on-time across delayed: {sum(ontimes)/len(ontimes):.1f}")
    print("\n" + "=" * 60)
    print("  Done! precomputed_scenarios.json is ready for deployment.")
    print("=" * 60)


if __name__ == "__main__":
    main()
