# Robots on Ice (PG-1): policy-gradient control of a Zamboni

A holonomic robot on a frictionless ice rink must reach a sequence of
random targets, each within 20 seconds, paying the squared distance to the
target and the energy it spends. The project models the task as a
finite-horizon MDP with continuous states and actions, and controls the
robot with a policy learned by policy-gradient methods.

## The MDP

**Dynamics** given from the assignment the control rate at 10 Hz `dt = 0.1 s`:

    x(t+1) = x(t) + 0.1 v(t)
    v(t+1) = v(t) + 0.1 F(t) / m

**Action.** The policy outputs an unbounded command `u in R^2`; the actuator
applies `F = u` if `||u|| ≤ 100 N` and the radial saturation `F = 100 u / ||u||` otherwise. Since the limit from the assignment is on the norm of the force, the admissible forces form a disk; the radial saturation brings every command into it by shortening it, so the direction of the command is preserved. Since the saturation is part of the environment, the policy gradient uses the exact log-density of the Gaussian command, log pi(u|s): the distribution of the saturated force F is never needed.

**Reward.** The negative of the assignment's stage cost, computed on the
state before the transition:

    r(s_t, F_t) = -( ||x_t - target||^2 + (C / 100) ||F_t||^2 )

**Targets.** The rink is a 60 m × 30 m (hockey rink standard size) rectangle centered at the origin.
Each target stays active for a fixed window of 200 steps (20 s); then the
next one appears, always sampled uniformly on the rink outside the 2 m disk around
the robot. A target counts as **reached** if the robot comes within 2 m of
it during its window. Reaching is only recorded as a metric: it gives no
bonus, and the target stays until its window ends. The base problem has no walls: the rink only
defines where targets appear, and the robot may leave it.

**Episodes.** The robot starts at rest at the center. An episode lasts
600 steps (three targets); the limit is a truncation, never a termination.

**State.** `[x, y, vx, vy, target_x, target_y, steps_left]`, where
`steps_left` counts the steps left in the current window. The timer is
needed for the Markov property, since the target changes at a fixed time.

## Usage
With `uv`:
```bash
uv sync                        # install the required dependencies
uv run pytest                  # run the additional tests
uv run jupyter lab notebooks/  # open the notebooks
```

You can also use venv with `requirements.txt`, exported from `uv.lock`
(Python 3.12 or newer, from the project folder):
```bash
python3 -m venv .venv            # create the environment
source .venv/bin/activate        # activate it
pip install -r requirements.txt  # install the dependencies and this project (editable)
pytest                           # run the additional tests
jupyter lab notebooks/           # open the notebooks
```

## Layout

| Path | Content |
|---|---|
| `src/zamboni/` | Where all the Enviroment and Algorithms code for the assignment will be implemented. Now is just the MDP: batched simulator and gymnasium environment |
| `notebooks/*` | Renders of the environment and, as the project grows, of the learned policies, with plots and discussion of the experiments. For now, the environment at its initial state. |
| `tests/*` | Some additional checks on the MDP rules |
