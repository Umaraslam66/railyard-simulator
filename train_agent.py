"""
DRL Agent Training Script
=========================
Trains a PPO agent on the RailyardEnv using the canonical 12-hour nominal
schedule (from nominal_schedule.json) with varied delay vectors.

Each episode uses the **same** schedule but a **different** delay vector,
so the agent learns to recover from arrival + loading delays by reordering
and reassigning trains.

Usage:
    python train_agent.py
"""

import os
import json
import time
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from railyard_env import RailyardEnv, _episode_log
from schedule_builder import (
    load_schedule, schedule_to_env_format, build_delay_vectors,
    TOTAL_STEPS, STEP_MINUTES,
)

OUTPUT_DIR = "training_output"


# ---------- schedule-aware env wrapper ------------------------------------
class ScheduleEnv(RailyardEnv):
    """Wraps RailyardEnv to automatically inject the nominal schedule and a
    rotating set of delay vectors on each reset.

    * reset() cycles through pre-built delay vectors (with a small chance of
      a no-delay episode so the agent also sees the nominal baseline).
    * Every episode runs for TOTAL_STEPS (144 = 12 h at 5-min steps).
    """

    def __init__(self, schedule, delay_vectors, no_delay_prob=0.1, **kwargs):
        num_trains = len(schedule)
        super().__init__(num_trains=num_trains, delay_prob=0.0, **kwargs)
        self._schedule = schedule
        self._delay_vectors = delay_vectors
        self._no_delay_prob = no_delay_prob
        self._delay_idx = 0

    def reset(self, seed=None, options=None):
        # Pick delay vector: cycle through list, with occasional no-delay
        rng = np.random.default_rng(seed)
        if rng.random() < self._no_delay_prob:
            delay_vec = [{"arrival_delay": 0, "loading_delay": 0}
                         for _ in self._schedule]
        else:
            delay_vec = self._delay_vectors[self._delay_idx % len(self._delay_vectors)]
            self._delay_idx += 1

        opts = {
            "schedule": self._schedule,
            "delay_vector": delay_vec,
            "max_steps": TOTAL_STEPS,
        }
        return super().reset(seed=seed, options=opts)


# ---------- callback ---------------------------------------------------
class ProgressCallback(BaseCallback):
    """Print live training stats and save the best model encountered."""

    def __init__(self, print_freq=5000, save_dir=OUTPUT_DIR, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.save_dir = save_dir
        self.t0 = None
        self.best_reward = -1e9

    def _on_training_start(self):
        self.t0 = time.time()

    def _on_step(self):
        if self.num_timesteps % self.print_freq == 0:
            eps = len(_episode_log)
            if eps:
                recent = _episode_log[-20:]
                avg_r = np.mean([e["reward"] for e in recent])
                avg_ot = np.mean([e["on_time_pct"] for e in recent])
                avg_c = np.mean([e["conflicts"] for e in recent])
            else:
                avg_r = avg_ot = avg_c = 0.0
            elapsed = time.time() - self.t0
            tag = ""
            if avg_r > self.best_reward and eps > 20:
                self.best_reward = avg_r
                self.model.save(os.path.join(self.save_dir, "drl_model_best"))
                tag = "  * best *"
            print(f"  Step {self.num_timesteps:>6} | Ep {eps:>4} | "
                  f"R(20)={avg_r:>7.1f} | OnTime={avg_ot:>5.1f}% | "
                  f"Confl={avg_c:>4.1f} | {elapsed:.0f}s{tag}")
        return True


# ---------- helper: run N episodes with a policy ----------------------
def _run_episodes(env, n, model=None, deterministic=False):
    """Run *n* episodes.  If *model* is None use random actions."""
    for ep in range(n):
        obs, _ = env.reset(seed=ep + 5000)
        done = False
        while not done:
            if model is not None:
                act, _ = model.predict(obs, deterministic=deterministic)
            else:
                act = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(act)
            done = term or trunc


# ---------- main -------------------------------------------------------
def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 62)
    print("   Railyard DRL Agent  --  12h Schedule Training Pipeline")
    print("=" * 62)

    # ---- Load canonical schedule and build delay scenarios ---------------
    schedule_raw = load_schedule("nominal_schedule.json")
    env_schedule = schedule_to_env_format(schedule_raw)
    num_trains = len(env_schedule)
    delay_vectors = build_delay_vectors(
        n=500, num_trains=num_trains, seed=0, big_delay_prob=0.2,
    )

    print(f"\n  Schedule : {num_trains} trains, {TOTAL_STEPS} steps "
          f"({TOTAL_STEPS * STEP_MINUTES // 60}h, 1 step = {STEP_MINUTES} min)")
    print(f"  Delays   : {len(delay_vectors)} scenarios "
          f"(big_delay_prob=0.2)")

    cfg = {
        "algorithm": "PPO",
        "total_timesteps": 200_000,
        "num_trains": num_trains,
        "total_steps": TOTAL_STEPS,
        "step_minutes": STEP_MINUTES,
        "delay_scenarios": len(delay_vectors),
        "big_delay_prob": 0.2,
        "no_delay_prob": 0.1,
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.02,
    }
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ---- Phase 1: Train PPO ------------------------------------------
    print("\n[1/3] Training PPO agent ...")
    _episode_log.clear()

    env = Monitor(ScheduleEnv(
        schedule=env_schedule,
        delay_vectors=delay_vectors,
        no_delay_prob=cfg["no_delay_prob"],
        log_episodes=True,
    ))
    model = PPO(
        "MlpPolicy", env,
        learning_rate=cfg["learning_rate"],
        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        ent_coef=cfg["ent_coef"],
        verbose=0,
    )
    model.learn(total_timesteps=cfg["total_timesteps"],
                callback=ProgressCallback(print_freq=5000))

    model_path = os.path.join(OUTPUT_DIR, "drl_model")
    model.save(model_path)
    print(f"\n  Final model  -> {model_path}.zip")

    # Use best model for evaluation if available
    best_path = os.path.join(OUTPUT_DIR, "drl_model_best.zip")
    if os.path.isfile(best_path):
        model = PPO.load(best_path)
        print(f"  Using best checkpoint for evaluation")

    if _episode_log:
        df = pd.DataFrame(_episode_log)
        df.insert(0, "episode", range(1, len(df) + 1))
        df.to_csv(os.path.join(OUTPUT_DIR, "training_log.csv"), index=False)
        print(f"  Training log  -> {len(df)} episodes")

    # ---- Phase 2: Random baseline ------------------------------------
    print("\n[2/3] Random-policy baseline (100 episodes) ...")
    _episode_log.clear()
    baseline_env = ScheduleEnv(
        schedule=env_schedule,
        delay_vectors=delay_vectors,
        no_delay_prob=0.0,
        log_episodes=True,
    )
    _run_episodes(baseline_env, 100, model=None)

    if _episode_log:
        df = pd.DataFrame(_episode_log)
        df.insert(0, "episode", range(1, len(df) + 1))
        df.to_csv(os.path.join(OUTPUT_DIR, "baseline_log.csv"), index=False)
        print(f"  Baseline log  -> {len(df)} episodes")

    # ---- Phase 3: Evaluate trained agent -----------------------------
    print("\n[3/3] Evaluating trained agent (100 episodes) ...")
    _episode_log.clear()
    eval_env = ScheduleEnv(
        schedule=env_schedule,
        delay_vectors=delay_vectors,
        no_delay_prob=0.0,
        log_episodes=True,
    )
    _run_episodes(eval_env, 100, model=model, deterministic=False)

    if _episode_log:
        df = pd.DataFrame(_episode_log)
        df.insert(0, "episode", range(1, len(df) + 1))
        df.to_csv(os.path.join(OUTPUT_DIR, "eval_log.csv"), index=False)

        print(f"\n  --- Evaluation (100 ep) ---")
        print(f"  Avg reward   : {df['reward'].mean():.1f}")
        print(f"  Avg on-time  : {df['on_time_pct'].mean():.1f}%")
        print(f"  Avg conflicts: {df['conflicts'].mean():.1f}")
        print(f"  Avg delay    : {df['avg_delay'].mean():.1f} steps")

    _episode_log.clear()

    print("\n" + "=" * 62)
    print("  Done!  Run  python drl_visualizer.py  to launch the dashboard.")
    print("=" * 62)


if __name__ == "__main__":
    train()
