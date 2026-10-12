"""The Zamboni MDP: a batched simulator and a Gymnasium wrapper around it."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

DT = 0.1
RINK_HALF_X = 30.0
RINK_HALF_Y = 15.0
MAX_FORCE = 100.0
TARGET_RADIUS = 2.0
TARGET_WINDOW = 200
EPISODE_STEPS = 600

N_PROPOSALS = 8  # target proposals per episode, see _spawn_targets


class ZamboniBatch:
    """n episodes simulated together, one row per episode.

    State of an episode: [x, y, vx, vy, target_x, target_y, steps_left].
    All episodes share the clock, so the target windows end together.
    """

    def __init__(self, n: int, mass: float = 75.0, energy_coeff: float = 0.5):
        if not 50.0 <= mass <= 100.0:
            raise ValueError(f"mass must lie in [50, 100] kg, got {mass}")
        if not 0.0 < energy_coeff < 1.0:
            raise ValueError(f"C must lie in (0, 1), got {energy_coeff}")
        self.n = n
        self.mass = float(mass)
        self.energy_coeff = float(energy_coeff)
        self.rng = np.random.default_rng()

        self.pos = np.zeros((n, 2))
        self.vel = np.zeros((n, 2))
        self.target = np.zeros((n, 2))
        self.steps_remaining = TARGET_WINDOW
        self.target_reached = np.zeros(n, dtype=bool)
        self.targets_reached = np.zeros(n, dtype=np.int64)
        self.targets_missed = np.zeros(n, dtype=np.int64)

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        self.rng = rng
        self.pos = np.zeros((self.n, 2))
        self.vel = np.zeros((self.n, 2))
        self.targets_reached = np.zeros(self.n, dtype=np.int64)
        self.targets_missed = np.zeros(self.n, dtype=np.int64)
        self._spawn_targets()
        return self.obs()

    def step(self, u) -> tuple[np.ndarray, np.ndarray, dict]:
        u = np.asarray(u, dtype=np.float64).reshape(self.n, 2)
        u_norm = np.linalg.norm(u, axis=1)
        saturated = u_norm > MAX_FORCE
        # radial saturation: same direction, norm at most 100 N
        force = u * (MAX_FORCE / np.maximum(u_norm, MAX_FORCE))[:, None]

        # cost on the state before the transition, as in the assignment
        dist_sq = np.sum((self.pos - self.target) ** 2, axis=1)
        force_sq = np.sum(force**2, axis=1)
        reward = -(dist_sq + (self.energy_coeff / 100.0) * force_sq)

        self.pos = self.pos + DT * self.vel
        self.vel = self.vel + DT * (force / self.mass)

        # reaching is only a metric, checked on x_{t+1} before the target changes
        inside = np.linalg.norm(self.pos - self.target, axis=1) <= TARGET_RADIUS
        reached_now = inside & ~self.target_reached
        self.target_reached |= reached_now
        self.targets_reached += reached_now

        self.steps_remaining -= 1
        window_expired = self.steps_remaining == 0
        if window_expired:
            self.targets_missed += ~self.target_reached
            self._spawn_targets()

        info = self.metrics()
        info.update(
            reached_now=reached_now,
            window_expired=window_expired,
            saturated=saturated,
            force_norm=np.sqrt(force_sq),
            distance_cost=dist_sq,
            energy=force_sq,
        )
        return self.obs(), reward, info

    def _spawn_targets(self) -> None:
        # Uniform on the rink, at least 2 m from the robot. A fixed number of
        # proposals keeps the random stream independent of the policy, so
        # every policy faces the same targets.
        proposals = self._proposals(N_PROPOSALS)
        valid = self._valid(proposals)
        while not valid.any(axis=0).all():  # no valid proposal: practically never
            missing = ~valid.any(axis=0)
            proposals[0, missing] = self._proposals(1)[0, missing]
            valid = self._valid(proposals)
        first = valid.argmax(axis=0)
        self.target = proposals[first, np.arange(self.n)]
        self.steps_remaining = TARGET_WINDOW
        self.target_reached = np.zeros(self.n, dtype=bool)

    def _proposals(self, k: int) -> np.ndarray:
        low = np.array([-RINK_HALF_X, -RINK_HALF_Y])
        high = np.array([RINK_HALF_X, RINK_HALF_Y])
        return self.rng.uniform(low, high, size=(k, self.n, 2))

    def _valid(self, proposals: np.ndarray) -> np.ndarray:
        on_rink = (np.abs(proposals[..., 0]) <= RINK_HALF_X) & (
            np.abs(proposals[..., 1]) <= RINK_HALF_Y
        )
        away = np.linalg.norm(proposals - self.pos, axis=-1) > TARGET_RADIUS
        return on_rink & away

    def obs(self) -> np.ndarray:
        timer = np.full((self.n, 1), float(self.steps_remaining))
        return np.hstack([self.pos, self.vel, self.target, timer])

    def metrics(self) -> dict:
        return {
            "target_reached": self.target_reached.copy(),
            "targets_reached": self.targets_reached.copy(),
            "targets_missed": self.targets_missed.copy(),
            "distance": np.linalg.norm(self.pos - self.target, axis=1),
            "speed": np.linalg.norm(self.vel, axis=1),
        }


class ZamboniEnv(gym.Env):
    """Gymnasium interface to a single episode."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    _OUTSIDE = np.array([0.80, 0.84, 0.88])
    _ICE = np.array([0.96, 0.98, 1.00])
    _TARGET = np.array([0.90, 0.30, 0.25])
    _ROBOT = np.array([0.10, 0.20, 0.45])
    _ROBOT_RADIUS = 0.6  # m, only for drawing

    def __init__(
        self, mass: float = 75.0, energy_coeff: float = 0.5, render_mode: str | None = None
    ):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.sim = ZamboniBatch(1, mass, energy_coeff)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = self.sim.reset(self.np_random)
        return obs[0], _unbatch(self.sim.metrics())

    def step(self, action):
        obs, reward, info = self.sim.step(action)
        return obs[0], float(reward[0]), False, False, _unbatch(info)

    def render(self) -> np.ndarray | None:
        """The rink with a 10 m margin, the target with its 2 m disk and the robot."""
        if self.render_mode != "rgb_array":
            return None
        ppm, half_x, half_y = 10, RINK_HALF_X + 10, RINK_HALF_Y + 10  # pixels per meter, view
        x = (np.arange(2 * half_x * ppm) + 0.5) / ppm - half_x
        y = half_y - (np.arange(2 * half_y * ppm) + 0.5) / ppm  # image rows go down
        X, Y = np.meshgrid(x, y)
        img = np.tile(self._OUTSIDE, X.shape + (1,))
        img[(np.abs(X) <= RINK_HALF_X) & (np.abs(Y) <= RINK_HALF_Y)] = self._ICE
        tx, ty = self.sim.target[0]
        img[np.hypot(X - tx, Y - ty) <= TARGET_RADIUS] = self._TARGET
        rx, ry = np.clip(self.sim.pos[0], [-half_x, -half_y], [half_x, half_y])  # stay in view
        img[np.hypot(X - rx, Y - ry) <= self._ROBOT_RADIUS] = self._ROBOT
        return (255 * img).astype(np.uint8)


def _unbatch(info: dict) -> dict:
    return {k: v[0].item() if isinstance(v, np.ndarray) else v for k, v in info.items()}


def make_env(
    mass: float = 75.0,
    energy_coeff: float = 0.5,
    render_mode: str | None = None,
    max_episode_steps: int = EPISODE_STEPS,
) -> gym.Env:
    """ZamboniEnv with the 600-step limit as a truncation."""
    env = ZamboniEnv(mass, energy_coeff, render_mode)
    return TimeLimit(env, max_episode_steps)
