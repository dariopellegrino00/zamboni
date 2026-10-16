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

RELATIVE_DISTANCE = (5.0, 30.0)  # m from the robot, for "relative" targets
N_PROPOSALS = {"absolute": 8, "relative": 32}  # target proposals per episode, see _draw_target


class ZamboniBatch:
    """n episodes simulated together, one row per episode.

    State of an episode: [x, y, vx, vy, target_x, target_y, steps_left].
    All episodes share the clock, so the target windows end together.

    friction: b in the assignment's friction variant.
    target_sampling: "absolute" anywhere on the rink or "relative" 5 to 30 m from the robot.
    """

    def __init__(
        self,
        n: int,
        mass: float = 75.0,
        energy_coeff: float = 0.5,
        friction: float = 0.0,
        target_sampling: str = "absolute",
    ):
        if not 50.0 <= mass <= 100.0:
            raise ValueError(f"mass must lie in [50, 100] kg, got {mass}")
        if not 0.0 < energy_coeff < 1.0:
            raise ValueError(f"C must lie in (0, 1), got {energy_coeff}")
        if friction != 0.0 and not 0.01 <= friction <= 0.1:
            raise ValueError(f"friction b must be 0 or lie in [0.01, 0.1], got {friction}")
        if target_sampling not in N_PROPOSALS:
            raise ValueError(f"unknown target sampling {target_sampling!r}")
        self.n = n
        self.mass = float(mass)
        self.energy_coeff = float(energy_coeff)
        self.friction = float(friction)
        self.target_sampling = target_sampling
        self.rng = np.random.default_rng()

        # one row per episode
        self.pos = np.zeros((n, 2))
        self.vel = np.zeros((n, 2))
        self.target = np.zeros((n, 2))  # the current target
        self.reached_this_window = np.zeros(n, dtype=bool)
        self.targets_reached = np.zeros(n, dtype=np.int64)  # count over the episode
        self.steps_remaining = TARGET_WINDOW  # shared by all episodes

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        self.rng = rng
        self.pos = np.zeros((self.n, 2))
        self.vel = np.zeros((self.n, 2))
        self.targets_reached = np.zeros(self.n, dtype=np.int64)
        self._new_window()
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
        self.vel = self.vel + DT * (force / self.mass - self.friction * self.vel)

        # reaching is only a metric, checked on x_{t+1} before the target changes
        inside = np.linalg.norm(self.pos - self.target, axis=1) <= TARGET_RADIUS
        reached_now = inside & ~self.reached_this_window
        self.reached_this_window |= reached_now
        self.targets_reached += reached_now

        self.steps_remaining -= 1
        window_expired = self.steps_remaining == 0
        if window_expired:
            self._new_window()

        info = {
            "reached_now": reached_now,
            "window_expired": window_expired,
            "saturated": saturated,
            "distance_cost": dist_sq,
            "energy": force_sq,
            "speed": np.linalg.norm(self.vel, axis=1),
        }
        return self.obs(), reward, info

    def _new_window(self) -> None:
        """A new target for each episode and a new window of 200 steps."""
        self.target = self._draw_target()
        self.reached_this_window = np.zeros(self.n, dtype=bool)
        self.steps_remaining = TARGET_WINDOW

    def _draw_target(self) -> np.ndarray:
        """For each episode the first valid one of a fixed number of proposals."""
        # a fixed number keeps the random draws independent of the policy
        proposals = self._proposals(self.target_sampling, N_PROPOSALS[self.target_sampling])
        valid = self._valid(proposals)  # (k, n): proposal j valid for episode i
        first = valid.argmax(axis=0)  # on booleans argmax gives the first True
        target = proposals[first, np.arange(self.n)]  # proposals[first[i], i] for each episode i
        # an episode without valid proposals draws absolute targets until one is valid
        missing = ~valid.any(axis=0)
        while missing.any():
            draw = self._proposals("absolute", 1)[0]
            accepted = missing & self._valid(draw)
            target[accepted] = draw[accepted]
            missing &= ~accepted
        return target

    def _proposals(self, sampling: str, k: int) -> np.ndarray:
        if sampling == "absolute":
            low = np.array([-RINK_HALF_X, -RINK_HALF_Y])
            high = np.array([RINK_HALF_X, RINK_HALF_Y])
            return self.rng.uniform(low, high, size=(k, self.n, 2))
        angle = self.rng.uniform(0.0, 2 * np.pi, size=(k, self.n))
        distance = self.rng.uniform(*RELATIVE_DISTANCE, size=(k, self.n))
        direction = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
        return self.pos + distance[..., None] * direction

    def _valid(self, proposals: np.ndarray) -> np.ndarray:
        x, y = proposals[..., 0], proposals[..., 1]  # (k proposals, n episodes)
        on_rink = (np.abs(x) <= RINK_HALF_X) & (np.abs(y) <= RINK_HALF_Y)
        distance = np.linalg.norm(proposals - self.pos, axis=-1)  # from the robot of each episode
        return on_rink & (distance > TARGET_RADIUS)

    def obs(self) -> np.ndarray:
        timer = np.full((self.n, 1), float(self.steps_remaining))
        return np.hstack([self.pos, self.vel, self.target, timer])


class ZamboniEnv(gym.Env):
    """Gymnasium interface to a single episode."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    _OUTSIDE = np.array([0.80, 0.84, 0.88])
    _ICE = np.array([0.96, 0.98, 1.00])
    _TARGET = np.array([0.90, 0.30, 0.25])
    _ROBOT = np.array([0.10, 0.20, 0.45])
    _ROBOT_RADIUS = 0.6  # m, only for drawing

    def __init__(
        self,
        mass: float = 75.0,
        energy_coeff: float = 0.5,
        friction: float = 0.0,
        target_sampling: str = "absolute",
        render_mode: str | None = None,
    ):
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.sim = ZamboniBatch(1, mass, energy_coeff, friction, target_sampling)
        self.action_space = spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float64)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float64)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = self.sim.reset(self.np_random)
        return obs[0], {}

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
    friction: float = 0.0,
    target_sampling: str = "absolute",
    render_mode: str | None = None,
    max_episode_steps: int = EPISODE_STEPS,
) -> gym.Env:
    """ZamboniEnv with the 600-step limit as a truncation."""
    env = ZamboniEnv(mass, energy_coeff, friction, target_sampling, render_mode)
    return TimeLimit(env, max_episode_steps)
