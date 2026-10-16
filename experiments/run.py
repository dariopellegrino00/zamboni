"""Training runs of the experiments, in parallel, saved in results/<experiment>/.

    uv run python experiments/run.py tuning

Runs already saved are skipped, so an interrupted experiment can be resumed.
"""

import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from zamboni.training import TEST_SEED, VALIDATION_SEED, Config, evaluate, train

RESULTS = Path(__file__).resolve().parent.parent / "results"
BUDGET = 100_000  # training episodes per run, n * K


def tuning():
    for n, lr, seed in itertools.product((50, 100, 200), (0.001, 0.003, 0.01, 0.03, 0.1), range(3)):
        yield Config(batch_size=n, iterations=BUDGET // n, lr=lr, seed=seed)


def estimators():
    for estimator, baseline, seed in itertools.product(
        ("reinforce", "gpomdp"), (False, True), range(5)
    ):
        yield Config(estimator=estimator, baseline=baseline, seed=seed)


def grid():
    for mass, energy_coeff, seed in itertools.product((50, 75, 100), (0.1, 0.5, 0.9), range(5)):
        yield Config(mass=mass, energy_coeff=energy_coeff, seed=seed)


def sigma():
    for sigma0, seed in itertools.product((10.0, 20.0, 30.0), range(3)):
        yield Config(sigma0=sigma0, learn_sigma=False, seed=seed)
    for seed in range(3):
        yield Config(seed=seed)  # learned, from 30 N


EXPERIMENTS = {"tuning": tuning, "estimators": estimators, "grid": grid, "sigma": sigma}


def run_name(cfg: Config) -> str:
    estimator = cfg.estimator + ("_baseline" if cfg.baseline else "")
    if not cfg.learn_sigma:
        sigma = f"_sigma{cfg.sigma0:g}fixed"
    else:
        sigma = "" if cfg.sigma0 == Config.sigma0 else f"_sigma{cfg.sigma0:g}"
    return (
        f"{estimator}_m{cfg.mass:g}_C{cfg.energy_coeff:g}"
        f"_n{cfg.batch_size}_lr{cfg.lr:g}{sigma}_s{cfg.seed}"
    )


def run(cfg: Config, path: Path) -> str:
    policy, history = train(cfg)
    validation = evaluate(policy.act, cfg.mass, cfg.energy_coeff, seed=VALIDATION_SEED)
    test = evaluate(policy.act, cfg.mass, cfg.energy_coeff, seed=TEST_SEED)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        config=json.dumps(asdict(cfg)),
        final_theta=policy.theta,
        **{key: value.astype(np.float32) for key, value in history.items()},
        **{f"validation_{key}": value for key, value in validation.items()},
        **{f"test_{key}": value for key, value in test.items()},
    )
    tmp.replace(path)  # never a half-written file
    return path.name


def main(names: list[str]) -> None:
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        sys.exit(f"unknown experiments: {', '.join(unknown)} (available: {', '.join(EXPERIMENTS)})")
    configs, paths = [], []
    for name in names:
        folder = RESULTS / name
        folder.mkdir(parents=True, exist_ok=True)
        for cfg in EXPERIMENTS[name]():
            path = folder / f"{run_name(cfg)}.npz"
            if not path.exists():  # runs already saved are skipped
                configs.append(cfg)
                paths.append(path)
    print(f"{len(configs)} runs to do")
    if not configs:
        return
    with ProcessPoolExecutor() as pool:  # one process per CPU thread
        for saved in pool.map(run, configs, paths):
            print(saved)


if __name__ == "__main__":
    main(sys.argv[1:])
