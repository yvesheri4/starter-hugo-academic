{
  "title": "Differentiable plasma modeling",
  "description": "A proposed research direction connecting GPU simulation, diagnostic measurements, and energy-efficient discharge design.",
  "period": "Proposed research direction",
  "institution": "University of Michigan",
  "date": "2026-10-25",
  "category": "Research outlook",
  "symbol": "∇",
  "links": [
    {
      "name": "PASCHEN-1D paper",
      "url": "https://doi.org/10.1016/j.cpc.2026.110404"
    },
    {
      "name": "Existing simulation codes and results",
      "url": "/codes/"
    }
  ]
}

## Research vision

How can plasma diagnostics help us infer uncertain physical parameters and design more energy-efficient discharges? My proposed research connects differentiable simulation with parameter inference and neural surrogate models, with an immediate focus on semiconductor plasma processing and longer-term relevance to preionization for fusion startup.

This page describes planned work. The solver, inference pipeline, and design results outlined below are research objectives.

## Three connected directions

### 1. Build a verified differentiable solver

Develop a one-dimensional discharge solver in JAX, coupling drift-diffusion transport, Poisson electrostatics, electrode emission, and an external circuit. PASCHEN-1D benchmarks would provide a reference. Mesh and timestep refinement, finite-difference gradient checks, and GPU runtime and memory measurements would establish numerical reliability before optimization.

### 2. Learn physical parameters from measurements

Use discharge-current and gap-voltage traces to estimate effective secondary-electron emission and an ionization-rate correction. Synthetic observations would test parameter recovery; experimental measurements would support calibration and validation. Sensitivity analysis would help identify which operating conditions distinguish parameters that produce similar signals.

### 3. Design energy-efficient plasma generation

Optimize driving-voltage waveforms and gas pressure, and compare characterized electrode surfaces, to reduce source energy while reaching a target electron density. Neural surrogates, including a Fourier neural operator developed with NVIDIA PhysicsNeMo, would screen candidate designs. The verified discharge solver would then check the selected candidates and their operating constraints.

## Foundations and planned outputs

This direction builds on my contributions to PASCHEN-1D and my studies of short-pulse electron transport and emission. Independent numerical comparisons and uncertainty analysis are central to the proposed approach.

Planned outputs include reproducible solver and gradient benchmarks, parameter-recovery studies, and verified design comparisons. My currently available [codes and results](/codes/) cover particle-in-cell and multiple-sheet examples.
