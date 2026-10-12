"""Checks on the MDP rules: bugs here would be silent during training."""

import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from zamboni.env import (
    DT,
    EPISODE_STEPS,
    MAX_FORCE,
    RINK_HALF_X,
    RINK_HALF_Y,
    TARGET_RADIUS,
    TARGET_WINDOW,
    ZamboniBatch,
    ZamboniEnv,
    make_env,
)


def make_sim(n=1, mass=75.0, energy_coeff=0.5, seed=0):
    sim = ZamboniBatch(n, mass, energy_coeff)
    sim.reset(np.random.default_rng(seed))
    return sim


def test_gymnasium_interface():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # advice on unbounded/unnormalized spaces
        check_env(ZamboniEnv(), skip_render_check=True)

    def rollout(seed):
        env = make_env()
        obs, _ = env.reset(seed=seed)
        actions = np.random.default_rng(123).normal(scale=50.0, size=(EPISODE_STEPS, 2))
        rows, flags = [obs], []
        for a in actions:
            obs, reward, terminated, truncated, _ = env.step(a)
            rows.append(np.append(obs, reward))
            flags.append((terminated, truncated))
        return np.concatenate(rows), flags

    first, flags = rollout(7)
    assert np.array_equal(first, rollout(7)[0])
    assert not np.array_equal(first, rollout(8)[0])
    # the global limit is a truncation, never a termination
    assert flags == [(False, False)] * (EPISODE_STEPS - 1) + [(False, True)]


def test_render_shows_rink_target_and_robot():
    env = ZamboniEnv(render_mode="rgb_array")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env)
    env.reset(seed=0)
    img = env.render()
    assert img.shape == (500, 800, 3) and img.dtype == np.uint8

    def rgb(color):
        return (255 * color).astype(np.uint8)

    assert np.array_equal(img[250, 400], rgb(ZamboniEnv._ROBOT))  # starts at the center
    assert np.array_equal(img[5, 5], rgb(ZamboniEnv._OUTSIDE))
    tx, ty = env.sim.target[0]
    assert np.array_equal(img[int(250 - 10 * ty), int(400 + 10 * tx)], rgb(ZamboniEnv._TARGET))
    assert ZamboniEnv().render() is None


def test_step_hand_computed():
    sim = make_sim(mass=80.0, energy_coeff=0.5)
    sim.pos[:] = [1.0, 2.0]
    sim.vel[:] = [3.0, -1.0]
    sim.target[:] = [4.0, 6.0]
    _, reward, info = sim.step([30.0, -40.0])  # ||u|| = 50: not saturated
    # cost on x_t, not x_{t+1}: 3² + 4² = 25, plus (0.5 / 100) * 50² = 12.5
    assert reward[0] == pytest.approx(-37.5)
    # explicit Euler: the position moves with the old velocity
    assert np.allclose(sim.pos, [[1.3, 1.9]])
    assert np.allclose(sim.vel, [[3.0 + DT * 30.0 / 80.0, -1.0 - DT * 40.0 / 80.0]])
    assert not info["saturated"][0]


def test_saturation_is_radial_and_priced_on_the_force():
    sim = make_sim(mass=50.0, energy_coeff=0.9)
    sim.target[:] = [3.0, 4.0]
    _, reward, info = sim.step([300.0, 400.0])  # ||u|| = 500 -> F = [60, 80]
    assert np.allclose(sim.vel, DT * np.array([[60.0, 80.0]]) / 50.0)
    assert reward[0] == pytest.approx(-(25.0 + (0.9 / 100.0) * MAX_FORCE**2))
    assert info["saturated"][0]
    assert info["force_norm"][0] == pytest.approx(MAX_FORCE)


def test_targets_spawn_on_rink_outside_success_radius():
    n = 2000
    sim = make_sim(n=n)
    sim.pos[:] = [29.0, 14.0]  # near a corner, where rejection matters most
    sim.steps_remaining = 1
    sim.step(np.zeros((n, 2)))
    for target in (make_sim(n=n).target, sim.target):  # at reset and at respawn
        assert np.all(np.abs(target[:, 0]) <= RINK_HALF_X)
        assert np.all(np.abs(target[:, 1]) <= RINK_HALF_Y)
    assert np.all(np.linalg.norm(sim.target - sim.pos, axis=1) > TARGET_RADIUS)


def test_same_seed_same_targets_whatever_the_policy():
    n = 1000
    a, b = make_sim(n=n, seed=5), make_sim(n=n, seed=5)
    b.pos[:] = [20.0, -10.0]  # b's robots stand elsewhere at the respawn
    a.steps_remaining = b.steps_remaining = 1
    a.step(np.zeros((n, 2)))
    b.step(np.zeros((n, 2)))
    # they differ only where the first proposal fell near one of the robots
    assert np.mean(np.all(a.target == b.target, axis=1)) > 0.97


def test_target_windows_last_exactly_200_steps():
    n = 50
    sim = make_sim(n=n)
    for _ in range(3):
        old_target = sim.target.copy()
        for k in range(1, TARGET_WINDOW + 1):
            obs, _, info = sim.step(np.zeros((n, 2)))
            if k < TARGET_WINDOW:
                assert np.array_equal(sim.target, old_target)
                assert np.all(obs[:, 6] == TARGET_WINDOW - k)
                assert not info["window_expired"]
        assert info["window_expired"]
        assert np.all(sim.target != old_target)
        assert np.all(obs[:, 6] == TARGET_WINDOW)
    # standing still at the center never reaches a target spawned > 2 m away
    assert np.all(sim.targets_missed == 3)
    assert np.all(sim.targets_reached == 0)


def test_success_is_a_metric_counted_once_per_window():
    sim = make_sim()
    sim.pos[:] = sim.target  # on the target, at rest
    for k in range(10):
        _, reward, info = sim.step(np.zeros(2))
        assert reward[0] == 0.0  # the assignment's cost, no bonus
        assert info["reached_now"][0] == (k == 0)
    assert sim.targets_reached[0] == 1


def test_success_on_last_step_credited_to_old_target():
    sim = make_sim()
    for _ in range(TARGET_WINDOW - 1):
        sim.step(np.zeros(2))
    sim.pos[:] = sim.target + [2.5, 0.0]
    sim.vel[:] = [-10.0, 0.0]  # x_{t+1} = target + (1.5, 0): inside
    _, _, info = sim.step(np.zeros(2))
    assert info["reached_now"][0] and info["window_expired"]
    assert sim.targets_reached[0] == 1
    assert sim.targets_missed[0] == 0
