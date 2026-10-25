# Langmuir oscillations and density inference in JAX

A small educational example: evolve a **linear cold-electron fluid**, measure the electric field at one position, and infer the equilibrium density by differentiating through the simulation. Fixed ions neutralize the equilibrium electrons. Boundaries are periodic. This is original demonstration code, not a reproduction of a published research result.

## Reproduce

Tested on Windows, Python 3.12, JAX CPU with 64-bit arithmetic. A GPU is not required; no GPU speedup is claimed. Use a separate environment:

```sh
python -m venv .venv
# Activate it: Windows .venv\Scripts\activate; macOS/Linux source .venv/bin/activate
python -m pip install -r requirements.txt
python run_example.py --check-only
python run_example.py --output results
```

The second command repeats validation, performs clean/noisy inference, and writes plots, an animation, data, and a machine-readable summary. Compilation can make the first run slower. `explore.ipynb` provides a short notebook entry point; install Jupyter separately if desired. Do not run with `python -O`, which disables assertions.

## Equations and fixed units

Let `eta = (n_e - n0)/n0`, `q = n0/n_ref`, and `u` be the perturbed electron velocity. The linearized equations are

```text
eta_t = -u_x
u_t   = -E
phi_xx = q eta
E = -phi_x
```

The electron charge is `-e`. With `n_ref = 1e14 m^-3`, define `omega_ref = sqrt(n_ref e^2/(m_e epsilon_0))`. Length, time, velocity, and electric field units are respectively `L/(2 pi)`, `1/omega_ref`, `omega_ref L/(2 pi)`, and `m_e omega_ref^2 L/(2 pi e)`, where `L = 1 cm`. **All these scales remain fixed while fitting q.** Physical observation times therefore retain information about the unknown density.

Initial conditions are `eta(x,0) = 0.02 cos(x)`, `u(x,0) = 0`. The continuum solution is `eta = 0.02 cos(x) cos(sqrt(q)t)` and `E = -0.02 q sin(x) cos(sqrt(q)t)`. The probe is at `x = pi/2`, or `L/4` physically. Its continuum frequency is `omega_pe = sqrt(q) omega_ref`.

## Numerical method

The solver evolves all grid values, not a fitted sinusoidal formula. It uses second-order centered first differences and FFT inversion of the second-order discrete Laplacian, with mean potential fixed to zero. A safe denominator avoids a zero-mode division in automatic differentiation. Classical RK4 advances the state inside `jax.lax.scan`; `jax.jit` compiles it and `jax.value_and_grad` differentiates the inference objective.

For the initial mode, the spatial discretization gives `omega_h = sqrt(q) cos(dx/2)` and probe amplitude `0.02 q (dx/2) cot(dx/2)`, with `dx = 2 pi/N`. These formulas explain the small grid-dependent frequency shift. They are used as independent validation references, not in the forward solver or optimizer.

The reported discrete energy is the domain average of `(q u^2 + E_edge^2)/2`, where `E_edge` is the forward potential difference. This matches the discrete Poisson operator; the centered force is the average of adjacent edge fields. The small RK4 energy drift is a consistency diagnostic, not sufficient evidence of convergence by itself.

## Inference experiment

- True density: `1.35e14 m^-3`; initial guess: `1.00e14 m^-3`.
- Model: 64 grid points, 720 steps, `dt = 0.025/omega_ref`.
- 181 observations, every fourth step, over about 32 ns.
- Data come from the **independent analytical continuum solution**, not the numerical solver. The noisy case adds zero-mean independent Gaussian noise with standard deviation 3% of the true field peak; seed 20261025.
- Bounded L-BFGS-B fits log density with reverse-mode JAX gradients. Density bounds are `[0.7, 1.8] * 1e14 m^-3`. The loss is a mean squared field residual divided by the fixed initial perturbation amplitude squared; it does not use the unknown density as a fit input.
- Initial perturbation amplitude, initial phase, probe position, field calibration, time calibration, and model assumptions are treated as known. This is a single-parameter synthetic exercise, not experimental parameter identification.
- A second clean-data fit at 128 grid points checks that the small density bias decreases under refinement. The noisy result is one realization, not an uncertainty interval or general noise-robustness claim. Oscillatory inverse problems can have local minima; the specified initial guess and bounds matter.

## Validation

`--check-only` asserts:

1. Equilibrium preservation, finite states, zero mean density perturbation, and discrete Poisson consistency.
2. Small relative discrete-energy drift.
3. Approximately second-order spatial convergence against the continuum field trace (N = 16, 32, 64, 128).
4. Approximately fourth-order timestep convergence against the exact semi-discrete trace (dt = 0.4, 0.2, 0.1, 0.05), isolating temporal error.
5. Reverse-mode gradients agree with central finite differences at three density ratios and three perturbation sizes.

The full run additionally checks successful clean/noisy recovery and decreasing clean-fit bias at 128 points. Detailed values, package versions, and backend are in `results/summary.json`.

## Outputs

- `inference.png`: probe traces and optimizer histories.
- `evolution.png`: density and electric-field space-time maps.
- `validation.png`: separate spatial and temporal convergence studies.
- `oscillation.gif`, `poster.png`: a user-playable animation of approximately one numerical period.
- `traces.csv`, `observations.csv`, `*_optimization.csv`: numerical plot data.
- `fields.npz`: `time_ns`, `x_mm`, `eta`, dimensionless `u`, and `E_V_m` arrays.
- `summary.json`: settings, validation metrics, and fitted densities.

## Scope

This linear model omits electron pressure, collisions, ion motion, nonlinear advection, Landau damping, trapping, and wave breaking. Small, smooth perturbations are required. Collocated centered differences have an unresolved Nyquist mode; this is not a general-purpose plasma solver. Validation covers the documented mode-one example and parameter range, not arbitrary initial conditions. The results are CPU calculations; potential GPU compatibility is not a performance benchmark.

JAX references: [automatic differentiation and transformations](https://docs.jax.dev/en/latest/101/transformations.html), [scan](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html), [64-bit arithmetic](https://docs.jax.dev/en/latest/default_dtypes.html).
