import numpy as np

from zamboni.controllers import PDController, lqr_gains
from zamboni.env import DT
from zamboni.policy import LinearGaussianPolicy

OBS = np.array(
    [[3.0, -1.5, 2.0, -4.0, 13.0, 4.0, 50.0], [-20.0, 10.0, -1.0, 0.5, 5.0, -8.0, 120.0]]
)


def axis_cost(kp, kd, mass=75.0, energy_coeff=0.5, steps=3000):
    """Cost of one axis without saturation, from 10 m away and at rest."""
    e, v, cost = 10.0, 0.0, 0.0
    for _ in range(steps):
        force = -kp * e - kd * v
        cost += e**2 + (energy_coeff / 100) * force**2
        e, v = e + DT * v, v + DT * force / mass
    return cost


def test_lqr_gains_minimize_the_cost():
    kp, kd = lqr_gains(75.0, 0.5)
    best = axis_cost(kp, kd)
    for dkp, dkd in [(1.05, 1), (0.95, 1), (1, 1.05), (1, 0.95)]:
        assert axis_cost(kp * dkp, kd * dkd) > best


def test_a_pd_is_a_linear_policy():
    pd = PDController.lqr(75.0, 0.5)
    policy = LinearGaussianPolicy()
    policy.W[0, 2] = policy.W[1, 3] = pd.kp / 10  # (target - x) / 10 and (target_y - y) / 10
    policy.W[0, 4] = policy.W[1, 5] = -pd.kd / 10  # vx / 10 and vy / 10
    assert np.allclose(policy.act(OBS), pd.act(OBS))
