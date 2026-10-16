"""Policy gradient estimates from a batch of n trajectories (pg1).

Each estimator returns the n terms g(theta; tau_i), one per trajectory: their mean is the
estimate of grad J, their spread its variance.
"""

import numpy as np


# x is the returns (n,) for reinforce or the rewards (T, n) for gpomdp
def leave_one_out_mean(x: np.ndarray) -> np.ndarray:
    """For each trajectory the mean of the other n - 1, along the last axis."""
    n = x.shape[-1]
    return (x.sum(axis=-1, keepdims=True) - x) / (n - 1)


# scores (T, n, 17), rewards (T, n) and result: (n, 17)
def reinforce(scores: np.ndarray, rewards: np.ndarray, baseline: bool = False) -> np.ndarray:
    """Per trajectory (sum_t grad log pi) (R(tau) - b)."""
    returns = rewards.sum(axis=0)
    if baseline:
        returns = returns - leave_one_out_mean(returns)
    return scores.sum(axis=0) * returns[:, None]


# scores (T, n, 17), rewards (T, n) and result: (n, 17)
def gpomdp(scores: np.ndarray, rewards: np.ndarray, baseline: bool = False) -> np.ndarray:
    """Per trajectory sum_t (r_t - b_t) sum_{h<=t} grad log pi(A_h|S_h)."""
    if baseline:
        rewards = rewards - leave_one_out_mean(rewards)
    past_scores = np.cumsum(scores, axis=0)  # at each t the scores summed from 0 to t
    return (rewards[..., None] * past_scores).sum(axis=0)
