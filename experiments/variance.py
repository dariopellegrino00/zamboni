"""Variance of the four gradient estimators at fixed parameters, saved in results/variance.json.

    uv run python experiments/variance.py

The parameters are taken along one run of the estimators experiment (GPOMDP with its baseline,
seed 0). At each of them the same 4000 trajectories feed the four estimators, and their terms
g(theta; tau_i) give the variance of each estimator, parameter by parameter.
"""

import json
from pathlib import Path

import numpy as np

from zamboni.env import ZamboniBatch
from zamboni.estimators import gpomdp, reinforce
from zamboni.policy import LinearGaussianPolicy, phi_bar
from zamboni.training import Config, rollout

RESULTS = Path(__file__).resolve().parent.parent / "results"
RUN = "estimators/gpomdp_baseline_m75_C0.5_n50_lr0.01_s0.npz"
ITERATIONS = (0, 100, 500, 1999)
BATCHES = 80  # of n = 50 trajectories, as in the training: 4000 trajectories per point
ESTIMATORS = {
    "REINFORCE": (reinforce, False),
    "REINFORCE + baseline": (reinforce, True),
    "GPOMDP": (gpomdp, False),
    "GPOMDP + baseline": (gpomdp, True),
}


def main() -> None:
    cfg = Config()
    thetas = np.load(RESULTS / RUN)["theta"].astype(np.float64)
    policy = LinearGaussianPolicy()
    env_rng, noise_rng = np.random.default_rng(0), np.random.default_rng(1)
    rows = []
    for k in ITERATIONS:
        policy.theta = thetas[k]
        terms = {name: [] for name in ESTIMATORS}
        for _ in range(BATCHES):
            sim = ZamboniBatch(cfg.batch_size, cfg.mass, cfg.energy_coeff)
            batch = rollout(sim, lambda obs: policy.sample(phi_bar(obs), noise_rng), env_rng)
            scores = policy.score(phi_bar(batch.states), batch.actions)
            for name, (estimate, baseline) in ESTIMATORS.items():
                terms[name].append(estimate(scores, batch.rewards, baseline=baseline))
        variance = {name: np.concatenate(t).var(axis=0, ddof=1) for name, t in terms.items()}
        row = {"iteration": k, "sigma": policy.sigma}
        for name, var in variance.items():
            # geometric mean over the 17 parameters of the variance relative to REINFORCE
            row[name] = float(np.exp(np.mean(np.log(var / variance["REINFORCE"]))))
        rows.append(row)
        print(row)
    (RESULTS / "variance.json").write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
