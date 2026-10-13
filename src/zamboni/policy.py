"""Linear Gaussian policy: pi(u|s) = N(u; 100 W phi_bar(s), sigma^2 I)."""

import numpy as np

from .env import MAX_FORCE
from .features import N_FEATURES, features

ACTION_DIM = 2  # u = [u_x, u_y]


def phi_bar(obs: np.ndarray) -> np.ndarray:
    """The features but with a single 1 appended for bias."""
    phi = features(obs)
    return np.concatenate([phi, np.ones(phi.shape[:-1] + (1,))], axis=-1)


class LinearGaussianPolicy:
    """W is 2 x 8 with one row per component of u. Both share one sigma learned as log sigma."""

    def __init__(self, sigma: float = 30.0):
        # only noise at start
        self.W = np.zeros((ACTION_DIM, N_FEATURES + 1))
        self.log_sigma = float(np.log(sigma))

    @property
    def sigma(self) -> float:
        return float(np.exp(self.log_sigma))

    @property
    def theta(self) -> np.ndarray:
        """All 17 parameters in one vector: W row by row and then log sigma."""
        return np.append(self.W.ravel(), self.log_sigma)

    @theta.setter
    def theta(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float64)
        self.W = value[:-1].reshape(self.W.shape).copy()
        self.log_sigma = float(value[-1])

    def mean(self, phi: np.ndarray) -> np.ndarray:
        return MAX_FORCE * phi @ self.W.T  # phi has one state per row

    def sample(self, phi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mu = self.mean(phi)
        return mu + self.sigma * rng.standard_normal(mu.shape)  # u = mu + sigma z

    def act(self, obs: np.ndarray) -> np.ndarray:
        """The mean action used for evaluation."""
        return self.mean(phi_bar(obs))

    def log_pdf(self, phi: np.ndarray, u: np.ndarray) -> np.ndarray:
        z = (u - self.mean(phi)) / self.sigma
        log_norm = ACTION_DIM * (self.log_sigma + 0.5 * np.log(2 * np.pi))  # 2 log sigma + log 2pi
        return -0.5 * np.sum(z**2, axis=-1) - log_norm

    def score(self, phi: np.ndarray, u: np.ndarray) -> np.ndarray:
        """grad_theta log pi(u|s) as 17 numbers per command in the same order as theta."""
        z = (u - self.mean(phi)) / self.sigma
        # for each command z is a 2 x 1 column and phi_bar a 1 x 8 row
        column, row = z[..., :, None], phi[..., None, :]
        grad_W = (MAX_FORCE / self.sigma) * column @ row  # z phi_bar^T for each command
        grad_W = grad_W.reshape(grad_W.shape[:-2] + (-1,))  # each 2 x 8 matrix as 16 numbers
        grad_log_sigma = np.sum(z**2, axis=-1) - ACTION_DIM  # ||z||^2 - 2
        return np.concatenate([grad_W, grad_log_sigma[..., None]], axis=-1)
