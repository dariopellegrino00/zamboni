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

**Action.** The policy outputs an unbounded command `u in R^2`. The actuator
applies `F = u` if `||u|| ≤ 100 N` and the radial saturation `F = 100 u / ||u||` otherwise. Since the limit from the assignment is on the norm of the force, the admissible forces form a disk. The radial saturation brings every command into it by shortening it, so the direction of the command is preserved. Since the saturation is part of the environment, the policy gradient uses the exact log-density of the Gaussian command, log pi(u|s): the distribution of the saturated force F is never needed.

**Reward.** The negative of the assignment's stage cost, computed on the
state before the transition:

    r(s_t, F_t) = -( ||x_t - target||^2 + (C / 100) ||F_t||^2 )

**Targets.** The rink is a 60 m × 30 m (hockey rink standard size) rectangle centered at the origin.
Each target stays active for a fixed window of 200 steps (20 s), then the
next one appears, always sampled uniformly on the rink outside the 2 m disk around
the robot. A target counts as **reached** if the robot comes within 2 m of
it during its window. Reaching is only recorded as a metric: it gives no
bonus, and the target stays until its window ends. The base problem has no walls: the rink only
defines where targets appear, and the robot may leave it.

**Episodes.** The robot starts at rest at the center. An episode lasts
600 steps (three targets). The limit is a truncation, never a termination.

**State.** `[x, y, vx, vy, target_x, target_y, steps_left]`, where
`steps_left` counts the steps left in the current window. The timer is
needed for the Markov property, since the target changes at a fixed time.

**Variants.** Two variants change one rule each. With friction, the second
challenging variant of the assignment, the velocity follows
`v(t+1) = v(t) + 0.1 (F(t) / m - b v(t))` with b = 0.01 or 0.1. With
relative targets, a variant added here and not in the assignment, we sample
a direction and a distance between 5 and 30 m uniformly, then reject
targets outside the rink. If none of the 32 proposals is valid, we draw an
absolute target.
The first challenging variant, the walls of the rink, is not implemented.

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

`experiments/run.py` runs the trainings of an experiment in parallel and saves each run in `results/<experiment>/`, so the notebooks read them without training again:

```bash
uv run python experiments/run.py tuning estimators grid sigma convergence friction relative_targets
uv run python experiments/variance.py
```

The hyperparameters are chosen on 1000 validation episodes. All the other
results are on 1000 different test episodes, so they do not benefit from
the choice. Runs already saved are skipped. The experiments, with their
tables and plots in `notebooks/experiments.ipynb` and
`notebooks/analysis.ipynb`:

- **tuning** of n and lr, 3 seeds per pair.
- **estimators**: REINFORCE and GPOMDP, with and without baseline, 5 seeds. GPOMDP with its baseline is the best (−67.0 ± 0.4 thousand at m = 75 kg, C = 0.5)
  and REINFORCE without baseline the worst (−102 ± 20 thousand).
- **variance** of the four estimators at fixed parameters: GPOMDP with its baseline has 4 to 19 times less variance than REINFORCE.
- **mass and energy cost** (the comparison asked by the assignment): m in {50, 75, 100} kg and C in {0.1, 0.5, 0.9}, 5 seeds. The learned policy beats both PD controllers in every pair, while reaching fewer targets.
- **σ fixed or learned**: learning σ matches the best fixed values (20 and 30 N) and beats 10 N, without having to choose it.
- **convergence**: twice the training budget gives the same return.
- **friction**, b in {0.01, 0.1}, 5 seeds. The learned policy still beats the LQR, also computed with friction (−69.2 ± 0.4 against −86.0 thousand with b = 0.1).
- **relative targets**, 5 seeds with 100 000 and 5 with 200 000 training episodes, in the next section.

## Why the learned policy beats the LQR

`notebooks/analysis.ipynb` reads the learned policies as gains. They are
much softer than the LQR's (kp about 4 N/m against 10 to 30), weakly
damped (damping ratio about 0.2 against 0.72), and add a pull towards the
center of the rink that the LQR does not have. The LQR stops on each
target, 18 m from the center on average when the window ends. The learned
policy swings past the target and back, and is 10 m from the center. A new
target is uniform on the rink, so its expected squared distance is 377 m²
from the center against 780 m² from 20 m away: being closer to the center
makes the next window cheaper.

A direct test measures how much this explains: adding the learned pull
towards the center (1.8 N/m) to the LQR improves it from −81.6 to −75.0
thousand, about half of the gap to the learned policy (−67.0 thousand).
The rest is most likely in the gains: the LQR is computed without the
100 N limit and saturates in 13% of the steps, the learned policy, trained
with the limit, in about 4%. The price is fewer targets reached (2.19 of 3
against 3), and that count depends on the 2 m radius (0.9 targets with
0.5 m, 2.9 with 5 m), so the return is the comparison that counts.

Relative targets test this explanation. With them the expected squared distance of the next target depends much less on where the robot is along x.
Along y it still grows towards the long edges, from 266 to 341 m². The learned pull along x drops from 1.85 to 0.43 N/m, and along y it
decreases from 1.92 to 1.69 N/m. This supports the explanation but does not prove it, since relative targets also
change the distances and the states the robot visits. With relative targets the LQR is better than the learned policy (−30.0 against −34.7 thousand,
and −31.5 with twice the training). Our policy can represent the LQR controller, but these training runs give worse returns.
The LQR saturates in 6.9% of steps with relative targets and 12.6% with absolute targets,
so this comparison does not separate the two parts of the explanation.

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
| `src/zamboni/` | The MDP (batched simulator and Gymnasium environment), the features, the policy, the gradient estimators, the training and the reference controllers |
| `notebooks/*` | The environment, the training, the experiments and the analysis of the learned controller, with plots and discussion |
| `experiments/` | The scripts that run the experiments |
| `results/` | One file per training run: configuration, learning curves, final parameters and evaluation |
| `tests/*` | Some additional checks on the MDP rules, the features, the policy's density and score, the estimators, the rollout, Adam and the LQR gains |
