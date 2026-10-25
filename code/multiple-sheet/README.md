# Multiple-sheet drift-space model

Yves Heri's numerical core for the short-pulse electron-beam drift-space study, packaged with a standalone example and archived current-limit comparisons.

Related article: [Y. Heri and P. Zhang, Space Charge Effects on Short-Pulse Beam Dynamics in a Drift Space](https://doi.org/10.1109/TED.2026.3670804).

## Run

Use Python 3.10 or newer, from this directory:

```sh
python -m pip install -r requirements.txt
python run_example.py --check
python run_example.py --output results
```

The default example uses 80 sheets per profile and 1,200 uniformly spaced time samples from 0 to 4. It writes two plots, sampled trajectories, an arrival table, and run metadata. `--sheets` and `--steps` adjust resolution. Outputs in the selected folder are replaced when rerunning.

## What is included

- `multiple_sheet.py`: numerical definitions extracted unchanged from the February 13, 2026 research notebook, with standalone imports.
- `run_example.py`: a new, compact driver and plotting wrapper. No local files outside this package are required.
- `data/current_limits.csv`: saved sheet-model and XOOPIC comparisons from the notebook, including its PIC error values. These are archived values, not a newly executed current-limit scan. Their error values are reproduced without reinterpretation as statistical confidence intervals.
- `results/`: the supplied demonstration plots, representative trajectory samples, all sheet arrivals, and numerical settings.
- `provenance.json`: original notebook name, SHA-256, and source cell indices.

## Model and normalization

This is a one-dimensional electrostatic, collisionless, nonrelativistic sheet model between grounded drift-space boundaries. It represents square-top, trapezoidal, and Gaussian injection profiles. The drift-space reference transit time is `T_DCL = 3*d/sqrt(8*e*V0/m_e)`. Position is normalized to the reference gap, velocity to `sqrt(2*e*V0/m_e)`, and time to `T_DCL`; consequently `dx_bar/dt_bar = 1.5*v_bar` in the archived core. The input `V0` sets the reference injection energy; it is not an applied accelerating voltage in this drift-space implementation.

The demonstration adapts the notebook's k=0.1 loading with total normalized sheet charges 0.5 times the archived critical-charge values (0.825, 0.825, 0.885 for the three profiles). Injection spans differ by profile, using the original factors 1, 1.15, and 2.1276595. Incoming normalized velocity is 1. The packaged run is a lower-resolution illustration, not an exact reproduction of the publication figures.

## Numerical limits

The archived implementation tracks injection order and does not dynamically re-sort sheets after crossing. Use it in the ordered-sheet regime; it is not a general post-crossing kinetic solver. The boundary freezes exited sheets, and velocity becomes zero on subsequent samples. Arrival CSV values use the first saved crossing of the downstream boundary; arrival times have time-step quantization. Zero-weight endpoint sheets in the trapezoidal loading are retained by the original implementation.

`--check` verifies the zero-charge ballistic limit and normalization/nonnegativity of all three pulse profiles. The driver also rejects a run if active, nonzero-charge sheets change order. The original 0.95-fraction near-limit setting failed this ordering check for the trapezoidal profile at the demonstration resolution; 0.5 is used for the shipped example. These checks do not establish convergence, arbitrary-parameter validity, or agreement with PIC. Refine sheet count and time step for quantitative work. The original notebook's unbounded arrival-search helper is preserved for provenance but is not used by the example driver.
