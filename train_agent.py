"""
DRL Agent Training Script
=========================
Trains a PPO agent on the RailyardEnv, logs per-episode metrics,
runs a random-policy baseline for comparison, then evaluates the
trained agent.  All artefacts are saved to  training_output/ .

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

OUTPUT_DIR = "training_output"


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
    print("   Railyard DRL Agent  --  Training Pipeline")
    print("=" * 62)

    cfg = {
        "algorithm": "PPO",
        "total_timesteps": 100_000,
        "num_trains": 6,
        "delay_prob": 0.2,
        "learning_rate": 3e-4,
        "n_steps": 1024,
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

    env = Monitor(RailyardEnv(num_trains=cfg["num_trains"],
                               delay_prob=cfg["delay_prob"]))
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
    baseline_env = RailyardEnv(num_trains=cfg["num_trains"],
                                delay_prob=cfg["delay_prob"])
    _run_episodes(baseline_env, 100, model=None)

    if _episode_log:
        df = pd.DataFrame(_episode_log)
        df.insert(0, "episode", range(1, len(df) + 1))
        df.to_csv(os.path.join(OUTPUT_DIR, "baseline_log.csv"), index=False)
        print(f"  Baseline log  -> {len(df)} episodes")

    # ---- Phase 3: Evaluate trained agent -----------------------------
    print("\n[3/3] Evaluating trained agent (100 episodes) ...")
    _episode_log.clear()
    eval_env = RailyardEnv(num_trains=cfg["num_trains"],
                            delay_prob=cfg["delay_prob"])
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
