import numpy as np

from zamboni.env import ZamboniBatch
from zamboni.training import Adam, rollout


def pull_to_target(obs):
    return obs[..., 4:6] - obs[..., 0:2]  # 1 N per m: here always below 100 N, so F = u


def test_rollout_pairs_each_state_with_its_command_and_reward():
    sim = ZamboniBatch(4, mass=75.0, energy_coeff=0.5)
    batch = rollout(sim, pull_to_target, np.random.default_rng(0))
    assert np.array_equal(batch.actions, pull_to_target(batch.states))
    distance_sq = np.sum((batch.states[..., 0:2] - batch.states[..., 4:6]) ** 2, axis=-1)
    energy = (0.5 / 100) * np.sum(batch.actions**2, axis=-1)
    assert np.allclose(batch.rewards, -(distance_sq + energy))


def test_adam_first_step_and_scale():
    grads = np.random.default_rng(0).normal(size=(5, 17))
    adam, adam_scaled = Adam(17), Adam(17)
    steps = [adam.step(g, 0.01) for g in grads]
    steps_scaled = [adam_scaled.step(1000 * g, 0.01) for g in grads]
    assert np.allclose(steps[0], 0.01 * np.sign(grads[0]))  # first step: m_hat = g, v_hat = g^2
    assert np.allclose(steps, steps_scaled)  # rewards times a constant: same steps
