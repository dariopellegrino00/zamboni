import numpy as np

from zamboni.estimators import gpomdp, reinforce

# T = 2 steps, n = 3 trajectories and a single parameter
SCORES = np.array([[1.0, 2.0, 0.0], [-1.0, 1.0, 3.0]])[..., None]
REWARDS = np.array([[1.0, 0.0, 2.0], [3.0, 1.0, -1.0]])


def test_reinforce_by_hand():
    # returns R = [4, 1, 1] and scores summed over time [0, 3, 3]
    terms = [0 * 4, 3 * 1, 3 * 1]
    assert np.allclose(reinforce(SCORES, REWARDS)[:, 0], terms)
    # leave-one-out baselines [1, 2.5, 2.5] so R - b = [3, -1.5, -1.5]
    terms = [0 * 3, 3 * -1.5, 3 * -1.5]
    assert np.allclose(reinforce(SCORES, REWARDS, baseline=True)[:, 0], terms)


def test_gpomdp_by_hand():
    # scores summed from 0 to t: [1, 2, 0] at t = 0 and [0, 3, 3] at t = 1
    terms = [1 * 1 + 3 * 0, 0 * 2 + 1 * 3, 2 * 0 + -1 * 3]
    assert np.allclose(gpomdp(SCORES, REWARDS)[:, 0], terms)
    # rewards minus leave-one-out b_t: [0, -1.5, 1.5] at t = 0 and [3, 0, -3] at t = 1
    terms = [0 * 1 + 3 * 0, -1.5 * 2 + 0 * 3, 1.5 * 0 + -3 * 3]
    assert np.allclose(gpomdp(SCORES, REWARDS, baseline=True)[:, 0], terms)
