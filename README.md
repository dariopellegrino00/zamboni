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

## The policy

A linear Gaussian policy, `pi(u|s) = N(u; 100 W phi_bar(s), sigma^2 I)`.
`phi_bar(s)` is the state rescaled to order 1 (`features.py`) with a 1
appended for the bias, and `W` is 2 × 8, one row per component of `u`.
The parameters are `W` and `log sigma`, 17 numbers. The course treats
sigma as a hyperparameter. Here it is learned as pg1 allows, and the
parameter is its log so that sigma stays positive. `W` starts at zero, so
the initial policy is pure noise, and sigma starts at 30 N. The score
`grad log pi(u|s)` is derived by hand.

## Training

Gradient ascent on the expected return. Each
iteration plays a batch of n episodes with the current policy and estimates
the gradient with GPOMDP, or with REINFORCE if chosen. Both can use a
leave-one-out baseline: each episode is compared with the average of the
other n - 1 (their returns for REINFORCE, their rewards at the same step for
GPOMDP), which does not depend on the episode itself and so adds no bias.
Adam updates the parameters with a step size that decreases linearly to
zero. There is no discount, since the horizon is finite.

The batch size n and the initial step size lr are chosen by a tuning with
the same budget of 100 000 training episodes per run: n = 50, K = 2000
iterations and lr = 0.01. The first guess (n = 100, K = 1000, lr = 0.001)
learned too slowly. Both trainings are in `notebooks/training.ipynb`.

## Reference controllers

Two PD controllers `u = kp (target - x) - kd v` are the comparison for the
learned policy, on the same test episodes, all without noise. The first is critically
damped, with kp = 10 N/m chosen by hand and `kd = 2 sqrt(kp m)`. The second
has the gains of the LQR, computed from the dynamics and the cost with the
Riccati recursion. The LQR ignores the 100 N limit and the
target changes, so it is a reference and not the optimum of the task. Both
are linear policies of the class above.

## Experiments

`experiments/run.py` runs the trainings of an experiment in parallel and
saves each run in `results/<experiment>/`, so `notebooks/experiments.ipynb`
reads them without training again:

```bash
uv run python experiments/run.py tuning  # skips the runs already saved
```

The hyperparameters are chosen on 1000 validation episodes. All the other
results are on 1000 different test episodes, so they do not benefit from
the choice.

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
| `src/zamboni/` | Where all the Enviroment and Algorithms code for the assignment will be implemented. Now: the MDP (batched simulator and gymnasium environment), the features, the policy, the gradient estimators, the training and the reference controllers |
| `notebooks/*` | Renders of the environment and, as the project grows, of the learned policies, with plots and discussion of the experiments. For now, the environment at its initial state, the training and the tuning. |
| `experiments/` | The script that runs the experiments |
| `results/` | One file per training run: configuration, learning curves, final parameters and evaluation |
| `tests/*` | Some additional checks on the MDP rules, the features, the policy's density and score, the estimators, the rollout, Adam and the LQR gains |
