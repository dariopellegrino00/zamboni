"""Features phi(s) of the linear policy: the state rescaled to be close to [-1, 1]"""

import numpy as np

from .env import RINK_HALF_X, RINK_HALF_Y, TARGET_WINDOW

RELATIVE_SCALE = 10.0  # order of the distance to a target in m
VELOCITY_SCALE = 10.0  # order of the robot's speeds in m/s
N_FEATURES = 7


def features(obs: np.ndarray) -> np.ndarray:
    """[x, y, vx, vy, target_x, target_y, steps_left] to phi(s) for one state or a batch."""
    x, y, vx, vy, tx, ty, steps_left = np.moveaxis(np.asarray(obs, dtype=np.float64), -1, 0)
    return np.stack(
        [
            x / RINK_HALF_X,
            y / RINK_HALF_Y,
            (tx - x) / RELATIVE_SCALE,
            (ty - y) / RELATIVE_SCALE,
            vx / VELOCITY_SCALE,
            vy / VELOCITY_SCALE,
            steps_left / TARGET_WINDOW,
        ],
        axis=-1,
    )
