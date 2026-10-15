"""Hand-designed reference controllers: a critically damped PD and the LQR."""

import numpy as np

from .env import DT


def lqr_gains(mass: float, energy_coeff: float, tol: float = 1e-12) -> tuple[float, float]:
    """kp and kd of the LQR for one axis, from the Riccati recursion."""
    A = np.array([[1.0, DT], [0.0, 1.0]])  # s = [e, v] with e = x - target
    B = np.array([[0.0], [DT / mass]])
    Q = np.diag([1.0, 0.0])  # weight of the distance in the cost
    R = np.array([[energy_coeff / 100.0]])  # weight of the energy
    P = Q
    for _ in range(100_000):
        K = (B.T @ P @ A) / (R + B.T @ P @ B)  # best gains [kp, kd] given the future cost P
        P_next = Q + A.T @ P @ (A - B @ K)  # the future cost one step further back
        if np.max(np.abs(P_next - P)) <= tol * np.max(np.abs(P_next)):
            kp, kd = K.ravel()
            return float(kp), float(kd)
        P = P_next
    raise RuntimeError("the Riccati iteration did not converge")


class PDController:
    def __init__(self, kp: float, kd: float):
        self.kp = float(kp)
        self.kd = float(kd)

    @classmethod
    def critically_damped(cls, mass: float, kp: float = 10.0) -> "PDController":
        # saturates at 10 m and settles in about 10 s
        return cls(kp, 2.0 * np.sqrt(kp * mass))

    @classmethod
    def lqr(cls, mass: float, energy_coeff: float) -> "PDController":
        return cls(*lqr_gains(mass, energy_coeff))

    def act(self, obs: np.ndarray) -> np.ndarray:
        pos, vel, target = obs[..., 0:2], obs[..., 2:4], obs[..., 4:6]
        return self.kp * (target - pos) - self.kd * vel
