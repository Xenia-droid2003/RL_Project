
from __future__ import annotations

import json
import random
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


TOTAL_STEPS = 150_000          
N_EVAL      = 20               
EVAL_NOISE  = 0.1              
SEEDS       = [0, 1, 2]
OUTPUT_DIR  = Path("results")

CONFIGS = {
    "baseline":    {"train_noise": 0.0},
    "noisy_train": {"train_noise": 0.1},
}


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


class ActionNoiseWrapper(gym.ActionWrapper):

    def __init__(self, env, sigma: float = 0.1, seed: int | None = None):
        super().__init__(env)
        self.sigma = sigma
        self.low = env.action_space.low.copy()
        self.high = env.action_space.high.copy()
        self.rng = np.random.default_rng(seed)

    def action(self, action):
        noise = self.rng.normal(0.0, self.sigma, size=action.shape).astype(np.float32)
        return np.clip(action + noise, self.low, self.high).astype(np.float32)


class RewardLogger(BaseCallback):
    """Собирает награды и длины завершённых эпизодов."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.timesteps: list[int] = []
        self.rewards: list[float] = []
        self.lengths: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.timesteps.append(self.num_timesteps)
                self.rewards.append(float(info["episode"]["r"]))
                self.lengths.append(int(info["episode"]["l"]))
        return True


def make_train_env(seed: int, noise_sigma: float) -> gym.Env:
    env = gym.make("LunarLanderContinuous-v3")
    if noise_sigma > 0:
        env = ActionNoiseWrapper(env, sigma=noise_sigma, seed=seed)
    env = Monitor(env)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def train_ppo(seed: int, noise_sigma: float, total_steps: int):
    set_global_seed(seed)
    env = make_train_env(seed, noise_sigma)
    logger = RewardLogger()
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        learning_rate=3e-4,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=total_steps, callback=logger, progress_bar=False)
    env.close()
    return model, logger


def evaluate(model: PPO, n_episodes: int, eval_noise: float, seed: int) -> np.ndarray:

    env = gym.make("LunarLanderContinuous-v3")
    if eval_noise > 0:
        env = ActionNoiseWrapper(env, sigma=eval_noise, seed=seed + 999)
    rewards: list[float] = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed * 10_000 + i)
        done, total = False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total += float(r)
            done = term or trunc
        rewards.append(total)
    env.close()
    return np.asarray(rewards)


def _smooth(x: np.ndarray, k: int = 20) -> np.ndarray:
    if len(x) < k:
        return np.asarray(x)
    return np.convolve(x, np.ones(k) / k, mode="valid")


def plot_learning_curves(learn_curves: dict, total_steps: int, out_path: Path) -> None:
    def curves_to_grid(cond: str):
        grid = np.linspace(0, total_steps, 200)
        ys = []
        for _seed, (ts, rs) in learn_curves[cond].items():
            if len(rs) == 0:
                continue
            rs_s = _smooth(rs, k=20)
            ts_s = ts[len(ts) - len(rs_s):]
            if len(ts_s) < 2:
                continue
            ys.append(np.interp(grid, ts_s, rs_s))
        if not ys:
            return grid, np.zeros_like(grid), np.zeros_like(grid)
        stacked = np.vstack(ys)
        return grid, stacked.mean(0), stacked.std(0)

    plt.figure(figsize=(9, 5))
    for cond in CONFIGS:
        g, m, s = curves_to_grid(cond)
        plt.plot(g, m, label=cond)
        plt.fill_between(g, m - s, m + s, alpha=0.2)
    plt.xlabel("Environment steps")
    plt.ylabel("Episode reward (moving avg, window=20)")
    plt.title("Learning curves: baseline vs noisy_train (mean ± std over seeds)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_comparison(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, (metric, title) in enumerate(
        [("mean", "Mean reward"), ("success", "Success rate (reward >= 200)")]
    ):
        xs = np.arange(2)
        w = 0.35
        for i, cond in enumerate(CONFIGS):
            clean = df[df.condition == cond][f"clean_{metric}"]
            noisy = df[df.condition == cond][f"noisy_{metric}"]
            ax[j].bar(
                xs - w / 2 + i * w,
                [clean.mean(), noisy.mean()],
                width=w,
                yerr=[clean.std(), noisy.std()],
                capsize=4,
                label=cond,
                alpha=0.85,
            )
        ax[j].set_xticks(xs)
        ax[j].set_xticklabels(["clean env", "noisy env"])
        ax[j].set_title(title)
        ax[j].set_ylabel(title)
        ax[j].grid(axis="y", alpha=0.3)
        if metric == "success":
            ax[j].set_ylim(0, 1)
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def record_video(model: PPO, seed: int, out_dir: Path, n_episodes: int = 2) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(out_dir),
        episode_trigger=lambda k: k < n_episodes,
        name_prefix=f"best_seed{seed}",
    )
    for i in range(n_episodes):
        obs, _ = env.reset(seed=999 + i)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            done = term or trunc
    env.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    models_dir = OUTPUT_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    results: dict = {}
    learn_curves: dict = {}

    for cond, cfg in CONFIGS.items():
        results[cond], learn_curves[cond] = {}, {}
        for seed in SEEDS:
            print(f"[{cond}] seed={seed}: training {TOTAL_STEPS} steps...")
            model, logger = train_ppo(seed, cfg["train_noise"], TOTAL_STEPS)
            model.save(str(models_dir / f"{cond}_seed{seed}"))

            rewards_clean = evaluate(model, N_EVAL, eval_noise=0.0, seed=seed)
            rewards_noisy = evaluate(model, N_EVAL, eval_noise=EVAL_NOISE, seed=seed)

            results[cond][seed] = {
                "clean_mean":    float(rewards_clean.mean()),
                "clean_std":     float(rewards_clean.std()),
                "noisy_mean":    float(rewards_noisy.mean()),
                "noisy_std":     float(rewards_noisy.std()),
                "clean_success": float((rewards_clean >= 200).mean()),
                "noisy_success": float((rewards_noisy >= 200).mean()),
            }
            learn_curves[cond][seed] = (logger.timesteps, logger.rewards)
            print(
                f"  clean: {results[cond][seed]['clean_mean']:.1f} | "
                f"noisy: {results[cond][seed]['noisy_mean']:.1f}"
            )

    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    plot_learning_curves(learn_curves, TOTAL_STEPS, OUTPUT_DIR / "learning_curves.png")
    print("Saved:", OUTPUT_DIR / "learning_curves.png")

    rows = []
    for cond in CONFIGS:
        for seed in SEEDS:
            r = results[cond][seed]
            rows.append(
                {
                    "condition": cond,
                    "seed": seed,
                    "clean_mean": r["clean_mean"],
                    "noisy_mean": r["noisy_mean"],
                    "clean_success": r["clean_success"],
                    "noisy_success": r["noisy_success"],
                }
            )
    df = pd.DataFrame(rows)
    summary = df.groupby("condition").agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    print("\n--- SUMMARY ---")
    print(summary.round(2).to_string())
    summary.to_csv(OUTPUT_DIR / "summary.csv")

    plot_comparison(df, OUTPUT_DIR / "final_comparison.png")
    print("Saved:", OUTPUT_DIR / "final_comparison.png")

    best_cond = max(
        CONFIGS, key=lambda c: np.mean([results[c][s]["noisy_mean"] for s in SEEDS])
    )
    best_seed = max(SEEDS, key=lambda s: results[best_cond][s]["noisy_mean"])
    print(f"Best: {best_cond}, seed={best_seed} -> recording video")
    best_model = PPO.load(str(models_dir / f"{best_cond}_seed{best_seed}"))
    record_video(best_model, best_seed, OUTPUT_DIR / "videos", n_episodes=2)
    print("Videos saved to:", OUTPUT_DIR / "videos")


if __name__ == "__main__":
    main()