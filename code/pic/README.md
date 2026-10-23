# General electrostatic 1D1V PIC

The main workflow is now **input JSON → `pic_solver.py` → HDF5 → `pic_plots.py`**.
The solver has no case selector and no branches for two-stream, Landau, or Pierce.
Those names identify example input files only. You can change the population list,
distributions, charge/mass, sources, boundaries, resolution, and output cadence.

## Install, simulate, then plot

Use Python 3.10 or newer. Open PowerShell:

```powershell
python -m pip install -r requirements.txt

# Optional: validate and initialize the input without stepping or saving output.
python pic_solver.py inputs/two_stream.json --check

# Simulation only: HDF5, CSV diagnostics, and resolved input JSON.
python pic_solver.py inputs/two_stream.json

# Independent postprocessing; does not run or import the solver.
python pic_plots.py results/general_two_stream/run.h5 --animate
```

Replace `python` with `py` if that is your Python command. The solver requires
NumPy and h5py; Matplotlib and Pillow are only used by the separate plotter.

The other example inputs are `landau.json`, `pierce_single.json`,
`pierce_two_stream.json`, and `reflecting_plasma.json` (mobile electrons and ions).
For example:

```powershell
python pic_solver.py inputs/landau.json
python pic_plots.py results/general_landau/run.h5 --animate --distribution-animation
```

## Create your own input

Copy an example, change the JSON values, and run that file:

```powershell
Copy-Item inputs/two_stream.json inputs/my_input.json
notepad inputs/my_input.json
python pic_solver.py inputs/my_input.json --check
python pic_solver.py inputs/my_input.json --output results/my_run
python pic_plots.py results/my_run/run.h5 --animate --distribution-animation
```

JSON has no comments and no expressions: write `12.566370614359172`, not `4*pi`.
Unknown fields and incompatible combinations are rejected. A minimal practical
input for a warm electron population is:

```json
{
  "seed": 42,
  "domain": {
    "length": 10.0,
    "cells": 128,
    "field_boundary": "periodic",
    "particle_boundary": "periodic"
  },
  "time": {"dt": 0.01, "end": 10.0, "diagnostic_interval": 0.05},
  "background_charge_density": 1.0,
  "populations": [{
    "name": "electrons", "charge": -1.0, "mass": 1.0,
    "density": 1.0, "particles": 32768,
    "initial": {
      "position": {
        "sampling": "quiet",
        "density_mode": {"amplitude": 0.01, "mode": 1, "phase": 0.0}
      },
      "velocity": {
        "distribution": "maxwellian", "drift": 0.0,
        "thermal_speed": 1.0, "sampling": "random"
      }
    }
  }],
  "phase_space": {
    "position_bins": 128, "velocity_bins": 256,
    "velocity_min": -5.0, "velocity_max": 5.0
  },
  "output": {
    "directory": "../results/my_run", "save_interval": 0.1,
    "particle_dump": "sample", "display_particles": 10000
  }
}
```

Input file paths and `output.directory` are relative to the input JSON's folder.
`--output` overrides the destination and is relative to your terminal directory.
Reusing the same destination overwrites the data. Plotting defaults to a `plots`
subfolder beside the HDF5 file; it never changes the simulation data.

## Population and initial distribution settings

Each population has its own `charge`, `mass`, initial mean number `density`, and
`particles`. Its constant macroparticle weight is `density*length/particles`.
The solver supports unequal weights **between populations**, not within one.
Add populations to represent beams, mobile ions, or mixtures. The positive or
negative fixed background is independently specified as `background_charge_density`.

Position `sampling`:

- `quiet`: uniformly spaced positions.
- `random`: uniformly random positions.
- `lattice`: `spatial_points` positions, with a repeated velocity sample at each;
  `particles` must be divisible by `spatial_points`.

An optional `density_mode` specifies mean-density times
`1 + amplitude*cos(2*pi*mode*x/length + phase)`. `mode` is a positive integer,
phase is in radians, and `abs(amplitude)<1`. Positions are sampled by inverting
the density CDF, rather than approximating it with a displacement.

Velocity `distribution` is `cold`, `maxwellian`, or `waterbag`. Set `drift` and
`thermal_speed` (standard deviation, including for waterbag). Cold distributions
require zero thermal speed. Velocity `sampling` is `random` or `quiet` (quantiles).
For a quiet Maxwellian at every spatial point, combine `position.sampling=lattice`
with `velocity.sampling=quiet`, as in the Landau input. Quiet velocity samples
without a lattice are shuffled to avoid artificial position-velocity correlation.

An optional velocity `perturbation` adds
`amplitude*sin(2*pi*mode*x/length + phase)` to v. Amplitude is in velocity units,
not a relative fraction. Periodic domains require integer mode numbers;
`mode=0.5` produces the half-wave used in the bounded Pierce examples.

### Arbitrary initial particles from a file

You can supply any finite arrays of initial x,v values without modifying the
solver. Save them in HDF5 using a separate preparation script:

```python
import numpy as np
import h5py

N, L = 32768, 10.0
rng = np.random.default_rng(42)
x = (np.arange(N) + 0.5) * L / N
v = 0.2 * rng.standard_normal(N)  # Replace with your distribution.
with h5py.File('inputs/initial_particles.h5', 'w') as f:
    f['x'] = x
    f['v'] = v
```

In that population's input, replace its `initial` object with:

```json
"initial": {"file": "initial_particles.h5"}
```

Arrays must have the specified `particles` length, with `0 <= x < length`.
An optional `group` chooses a subgroup containing `x` and `v`. For instance,
`"group": "final_particles/p000"` reads population zero from a previous output.
This loads particle arrays as a new t=0 initial condition, **not a time-continuous
restart**: source clocks and cumulative diagnostics reset. NPZ x,v input is also
accepted for compatibility; simulation output is HDF5.

## Boundary conditions and sources

Periodic fields require periodic particles, an even cell count, and net charge
neutrality: `background_charge_density + sum(charge*density) = 0`. The mean
electric field is fixed to zero. Non-neutral periodic input is rejected.

For a bounded domain, use:

```json
"domain": {
  "length": 12.0, "cells": 384,
  "field_boundary": "dirichlet",
  "potential_left": 0.0, "potential_right": 0.0,
  "particle_boundary": {"left": "absorbing", "right": "reflecting"}
}
```

Each particle boundary can independently be `absorbing` or `reflecting`.
Electrode potentials may differ. The field boundary is fixed potential even
when the particle wall reflects; a reflecting wall is not an insulating-field
boundary condition. No periodic/Dirichlet hybrid field option is implemented.

Each population may have a list of independent constant cold sources:

```json
"sources": [{"boundary": "left", "number_flux": 1.0, "velocity": 1.0}]
```

`number_flux` is physical number flux, not macroparticles per step. The
macroparticle emission rate is `number_flux/particle_weight`. For a cold source
of density n and speed u, set `number_flux=n*abs(u)`. Left source velocity must
be positive; right source velocity must be negative. Sources require absorbing
inlet boundaries. Initial particles and continuing injection are independent.
Multiple cold sources can approximate a prescribed incoming velocity mixture.

An initially empty source population uses `particles=0`, `density=0`, and an
explicit positive `particle_weight`. This avoids tying source resolution to
an initial particle count.

## Adjustable data-saving frequency

The physics time step and output cadence are independent:

```json
"time": {"dt": 0.005, "end": 100.0, "diagnostic_interval": 0.05},
"output": {
  "directory": "../results/my_run",
  "save_interval": 0.5,
  "field_interval": 0.1,
  "distribution_interval": 0.5,
  "particle_interval": 1.0,
  "particle_dump": "sample",
  "display_particles": 10000
}
```

- `save_interval` is the common default for fields, distributions, and particles.
- `field_interval`, `distribution_interval`, and `particle_interval` optionally
  override that default independently.
- `time.diagnostic_interval` controls scalar histories such as energy and counts.
- All intervals and final time must be positive integer multiples of dt.
- Initial and final data are always saved, even when an interval does not divide
  the total duration. The last interval may therefore be shorter.
- `particle_dump`: `none`, `sample`, or `all`. `display_particles` caps the
  **total** sample size across populations and only applies to `sample`.
- Full initial and final particles are always saved. Full intermediate dumps
  can be large; choose `all` when you need to rebin intermediate distributions
  at arbitrary resolution later.

Changing output frequency does not change particle dynamics. Each data family
has its own time coordinates in the file. A plotted animation frame is one
saved state, not one integration step.

## Standard data format and f(x,v,t)

`run.h5` is a standard HDF5 file, readable with h5py, MATLAB `h5read`/`h5info`,
Julia HDF5.jl, or HDFView. The layout below is a documented project schema,
**not a claim of openPMD compliance**. No Python objects or pickles are stored.
`config.json` and `diagnostics.csv` are also written for convenient inspection.

| HDF5 path | Contents |
|---|---|
| `/configuration_json` | Resolved input including defaults |
| `/populations/name`, `charge`, `mass`, `particle_weight` | Population metadata |
| `/fields/time`, `/fields/x` | Field coordinates |
| `/fields/electric`, `potential`, `charge_density` | Shape `(time, mesh_node)` |
| `/fields/electric_energy` | Field energy at field-save times |
| `/distribution/time` | Distribution save times |
| `/distribution/x_edges`, `v_edges`, `x`, `v` | Bin edges and centers |
| `/distribution/f` | Shape **`(time, population, x_bin, v_bin)`** |
| `/distribution/number` | Total physical number per population at each distribution time |
| `/distribution/number_in_range`, `number_below_range`, `number_above_range` | Velocity coverage accounting |
| `/particles/step_000000100/p000/x`, `v` | Sampled or full intermediate particles; group attrs give time and sampling flag |
| `/initial_particles/p000/x`, `v` | Full initial particles |
| `/final_particles/p000/x`, `v` | Full final particles |
| `/diagnostics/time`, `electric`, `kinetic`, `total`, etc. | Scalar histories |

`p000`, `p001`, etc. follow input population order. `p0_balance_error` in scalar
diagnostics is `current-initial-injected+absorbed_left+absorbed_right`, in
macroparticle counts. Electric energy is `0.5*integral(E² dx)` and kinetic energy
is `sum(0.5*mass*particle_weight*v²)`. Background/electrode energy is not tracked.

Every distribution snapshot bins **every solver particle**, regardless of
`particle_dump` or `display_particles`. For one population:

`f[time,population,i,j] = particle_weight * count_in_bin / (dx_bin * dv_bin)`.

Thus `sum(f*dx_bin*dv_bin)` is physical particle number **inside the velocity
range**, not 1. Integrating f over velocity gives the represented number density
on the histogram grid. Charge/mass weighting is separate. The histogram grid
is independent of the field mesh, and histogram density need not equal CIC
mesh density point by point. No smoothing is applied.

Set the velocity range and bin counts in `phase_space`. Out-of-range physical
number is explicitly saved; it is never folded into the edge bins. Widen the
range if overflow matters. Counts in range plus both overflow counts recover
the total number. With fixed bins, output f is a coarse-grained distribution;
it cannot recover unresolved filaments. Use finer bins/more particles or full
particle dumps if needed for reconstruction and learning.

Read one snapshot without loading the complete history:

```python
import h5py
import numpy as np

with h5py.File('results/general_two_stream/run.h5', 'r') as h:
    times = h['distribution/time'][:]
    f = h['distribution/f'][10, 0, :, :]  # Time index 10, population 0
    x_edges = h['distribution/x_edges'][:]
    v_edges = h['distribution/v_edges'][:]
    density_x = (f * np.diff(v_edges)[None, :]).sum(axis=1)
    number = (density_x * np.diff(x_edges)).sum()
    print('Time:', times[10], 'represented number in velocity range:', number)
```

HDF5 stores normalization metadata and dimension scales for fields/distributions.
The root `complete` flag is set only after final data are written. Interrupted
files remain marked incomplete; the plotter rejects these by default.

## Standalone plotting

```powershell
# Static field map, RMS field, energy, f snapshots/marginals, particle snapshots.
python pic_plots.py results/my_run/run.h5

# Add the particle phase-space movie.
python pic_plots.py results/my_run/run.h5 --animate

# Add a quantitative f(x,v,t) movie, optionally for one population.
python pic_plots.py results/my_run/run.h5 --distribution-animation --population electrons
```

The plotter reads only HDF5; it does not need the original input or solver.
`distribution.png` and `distribution.gif` use all-particle histograms. Particle
movies use saved particle dumps and are colored by population. If
`particle_dump=none`, use the distribution movie instead. `--output` chooses
a plot directory. The default total distribution sums **number distributions**
over populations; choose a population when mixing species would obscure meaning.

## Model, numerical method, and limits

The normalized equations are dx/dt=v, dv/dt=(q/m)E, dE/dx=rho, and E=-dphi/dx.
Reference omega_p0=1, epsilon0=1, electron mass=1, elementary charge magnitude=1;
velocity is in v0, position in v0/omega_p0, and number density in n0. Potential
is in m_e*v0²/e. Inputs may include mobile ions with positive charge and larger mass.

The numerical core uses CIC deposition/interpolation and kick-drift-kick with
full-time synchronized positions/velocities. Periodic Poisson is spectral,
with zero mean E and the unrepresentable Nyquist field derivative omitted.
Dirichlet Poisson uses a direct finite-difference solve; endpoint deposition
uses half-width control volumes. Injected particles advance for the fractional
remaining time step. Existing exits are removed after drift; reflecting walls
reverse the drift velocity. Source emission is deterministic, using constant
macroparticle weights.

This is collisionless electrostatic 1D1V, not an electromagnetic, relativistic,
collisional, multidimensional, or arbitrary field-boundary solver. Sources are
constant cold beams; time-dependent voltages and kinetic source distributions
are not built in. Nonuniform fixed background profiles are not built in, but
arbitrary kinetic initial profiles can be supplied as particle files.

Validity of an input is not evidence of numerical accuracy. Refine dt, cell
size and particle count and inspect relevant scales; these checks are not
automatically enforced. Finite quiet velocity grids can show recurrence.
Open systems exchange particles and energy with electrodes and sources, so
stored total energy is not a closed-system conservation residual.

## Files and tests

Only `pic_solver.py`, `pic_io.py`, an input JSON, NumPy, and h5py are needed to
simulate. `pic_plots.py` and plotting dependencies are independent postprocessing.

```powershell
python -m unittest test_pic_solver.py
```

Tests cover deposition charge, manufactured Poisson solutions, charge/mass
forces, reflection, source balance, quiet Maxwellian loading, arbitrary file
inputs, time refinement, save cadences, HDF5 output, and f normalization.

## Published results

The `results` folders contain resolved inputs and diagnostic CSV histories from five completed runs. The two-stream animation and figures on the website are generated from the corresponding full-duration HDF5 output. Run the supplied input to regenerate HDF5; large raw HDF5 files are not included in this download. These are illustrative numerical results, not a convergence study.

Run commands from this folder. To reproduce the compact website animation after simulating:

```sh
python make_preview.py results/general_two_stream/run.h5 --output preview
```
