"""Playing episodes and training the policy (pg1, Algorithm 1)."""

from dataclasses import dataclass

import numpy as np

from .env import EPISODE_STEPS, ZamboniBatch
from .estimators import gpomdp, reinforce
from .policy import LinearGaussianPolicy, phi_bar

TEST_SEED = 2026  # episodes for the reported results
VALIDATION_SEED = 2025  # episodes to choose the hyperparameters
EVAL_EPISODES = 1000


@dataclass
class Rollout:
    states: np.ndarray  # S_t, shape (T, n, 7)
    actions: np.ndarray  # u_t, shape (T, n, 2)
    rewards: np.ndarray  # r(S_t, u_t), shape (T, n)
    targets_reached: np.ndarray  # per episode, shape (n,)


def rollout(sim: ZamboniBatch, act, rng: np.random.Generator) -> Rollout:
    """Plays sim.n episodes in parallel with u_t = act(S_t); rng draws the targets."""
    obs = sim.reset(rng)
    T, n = EPISODE_STEPS, sim.n
    states = np.empty((T, n, obs.shape[1]))
    actions = np.empty((T, n, 2))
    rewards = np.empty((T, n))
    for t in range(T):
        u = act(obs)
        states[t], actions[t] = obs, u
        obs, rewards[t], _ = sim.step(u)
    return Rollout(states, actions, rewards, sim.targets_reached.astype(np.float64))


class Adam:
    """Adam (Kingma and Ba, 2015): the step to add to theta, for gradient ascent."""

    def __init__(self, dim: int, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.k = 0
        self.beta1, self.beta2, self.eps = beta1, beta2, eps

    def step(self, grad: np.ndarray, lr: float) -> np.ndarray:
        self.k += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad  # mean of the gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2  # mean of its square
        # m and v start from 0 so in the first iterations they are too small
        m_hat = self.m / (1 - self.beta1**self.k)
        v_hat = self.v / (1 - self.beta2**self.k)
        return lr * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class Config:
    mass: float = 75.0
    energy_coeff: float = 0.5
    # n, K and lr chosen by the tuning on validation episodes (experiments/run.py)
    batch_size: int = 50  # n
    iterations: int = 2000  # K
    lr: float = 0.01  # initial step size of Adam, decreasing linearly to 0
    sigma0: float = 30.0
    estimator: str = "gpomdp"  # or "reinforce"
    baseline: bool = True
    seed: int = 0


def train(cfg: Config) -> tuple[LinearGaussianPolicy, dict]:
    """Gradient ascent on J(theta) with the estimator chosen in cfg."""
    # one random stream for the targets and one for the policy noise
    env_seed, noise_seed = np.random.SeedSequence(cfg.seed).spawn(2)
    env_rng, noise_rng = np.random.default_rng(env_seed), np.random.default_rng(noise_seed)
    sim = ZamboniBatch(cfg.batch_size, cfg.mass, cfg.energy_coeff)
    policy = LinearGaussianPolicy(cfg.sigma0)
    adam = Adam(policy.theta.size)
    estimate = {"reinforce": reinforce, "gpomdp": gpomdp}[cfg.estimator]

    def explore(obs):
        return policy.sample(phi_bar(obs), noise_rng)

    history = {"return": [], "targets_reached": [], "sigma": [], "theta": []}
    for k in range(cfg.iterations):
        batch = rollout(sim, explore, env_rng)
        scores = policy.score(phi_bar(batch.states), batch.actions)
        grad = estimate(scores, batch.rewards, baseline=cfg.baseline)

        # the average return of the batch estimates J(theta_k)
        history["return"].append(batch.rewards.sum(axis=0).mean())
        history["targets_reached"].append(batch.targets_reached.mean())
        history["sigma"].append(policy.sigma)
        history["theta"].append(policy.theta)

        lr = cfg.lr * (1 - k / cfg.iterations)
        policy.theta = policy.theta + adam.step(grad, lr)

    # return: (K,), targets_reached: (K,), sigma: (K,), theta: (K, 17), all at iteration k
    return policy, {key: np.array(value) for key, value in history.items()}


def evaluate(act, mass: float, energy_coeff: float, seed: int = TEST_SEED) -> dict:
    """Average return and targets reached of act on fixed episodes, by default the test set."""
    sim = ZamboniBatch(EVAL_EPISODES, mass, energy_coeff)
    batch = rollout(sim, act, np.random.default_rng(seed))
    return {
        "return": batch.rewards.sum(axis=0).mean(),
        "targets_reached": batch.targets_reached.mean(),
    }
