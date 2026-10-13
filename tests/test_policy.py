import itertools

import numpy as np

from zamboni.policy import LinearGaussianPolicy, phi_bar

OBS = np.array([3.0, -1.5, 2.0, -4.0, 13.0, 4.0, 50.0])


def random_policy(rng):
    policy = LinearGaussianPolicy(sigma=20.0)
    policy.W = rng.normal(size=policy.W.shape)  # a mean that is not zero
    return policy


def test_density_integrates_to_one():
    policy = LinearGaussianPolicy(sigma=20.0)
    phi = phi_bar(OBS)
    step = 0.5  # side of each grid square in N
    half_width = 8 * policy.sigma  # beyond 8 sigma the density is about 0
    grid = np.arange(-half_width, half_width + step, step)
    u = policy.mean(phi) + np.array(list(itertools.product(grid, grid)))  # every pair (u_x, u_y)
    density = np.exp(policy.log_pdf(phi, u))
    assert np.isclose(density.sum() * step**2, 1.0)  # the integral by rectangles


def test_score_matches_finite_differences():
    rng = np.random.default_rng(0)
    policy = random_policy(rng)
    phi = phi_bar(OBS)
    u = policy.sample(phi, rng)
    theta, h = policy.theta, 1e-6
    numeric = np.empty_like(theta)
    for j in range(theta.size):  # nudge one parameter at a time
        step = np.zeros_like(theta)
        step[j] = h
        policy.theta = theta + step
        plus = policy.log_pdf(phi, u)
        policy.theta = theta - step
        minus = policy.log_pdf(phi, u)
        numeric[j] = (plus - minus) / (2 * h)
    policy.theta = theta
    assert np.allclose(policy.score(phi, u), numeric, rtol=1e-5, atol=1e-7)


def test_score_has_zero_mean():
    rng = np.random.default_rng(1)
    policy = random_policy(rng)
    phi = phi_bar(np.tile(OBS, (200_000, 1)))  # the same state with many sampled commands
    scores = policy.score(phi, policy.sample(phi, rng))
    standard_error = scores.std(axis=0) / np.sqrt(len(scores))
    assert np.all(np.abs(scores.mean(axis=0)) < 5 * standard_error)
