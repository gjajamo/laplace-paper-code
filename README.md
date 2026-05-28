# Laplace FOCEI AD Paper Code

This repository contains the Julia code used for the manuscript analyses comparing finite-difference and automatic-differentiation gradients for the Laplace approximation in nonlinear mixed-effects models.

The code includes two examples:

- `flipflop_single_start.jl` and `flipflop_multistart.jl`: one-compartment population PK example with absorption/elimination ambiguity.
- `warfarin_single_start.jl` and `warfarin_multistart.jl`: public warfarin PK/PD example with the effect-compartment equation solved by fixed-step fourth-order Runge--Kutta integration of the ODE.

The first-submission scripts have been removed from this release. The current code path is Julia-based and uses forward-mode automatic differentiation for the small subject-level random-effect and population-parameter dimensions used in the manuscript.

## Contents

- `Project.toml` and `Manifest.toml`: Julia environment used by the computation scripts.
- `src/flipflop_multistart_methods.jl`: flip-flop model, Laplace objective, and optimization methods.
- `src/warfarin_model.jl`: warfarin PK/PD model, including the ODE effect-compartment solver.
- `src/warfarin_multistart_methods.jl`: warfarin Laplace objective and optimization methods.
- `data/warfarin_dat.csv`: public warfarin dataset used by the warfarin scripts.

## Environment

Julia 1.10 was used for the revised analyses. From the repository root, instantiate the environment with:

```bash
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

## Single-Start Runs

The single-start wrappers run one matched start for each manuscript method and write CSV outputs under `outputs/`.

```bash
julia --project=. flipflop_single_start.jl
julia --project=. warfarin_single_start.jl
```

## Multistart Runs

The multistart wrappers use the same defaults as the revised manuscript timing comparisons unless overridden by environment variables.

```bash
julia --project=. flipflop_multistart.jl
julia --project=. warfarin_multistart.jl
```

Useful controls include:

- `FLIPFLOP_JULIA_N_STARTS` and `WARFARIN_JULIA_N_STARTS`: number of starts.
- `FLIPFLOP_JULIA_METHODS` and `WARFARIN_JULIA_METHODS`: comma-separated subset of `FULL_IMPLICIT,FULL_UNROLL,STOP,FD`.
- `FLIPFLOP_JULIA_MAXITER_OUTER` and `WARFARIN_JULIA_MAXITER_OUTER`: maximum outer iterations.
- `FLIPFLOP_JULIA_MAXITER_ETA` and `WARFARIN_JULIA_MAXITER_ETA`: maximum Newton iterations for subject-specific eta modes.
- `WARFARIN_JULIA_DT`: maximum RK4 step size in hours for the warfarin effect-compartment ODE. The manuscript uses `0.25`.
- `FLIPFLOP_JULIA_OUTDIR` and `WARFARIN_JULIA_OUTDIR`: output directories.

## Outputs

Default outputs are CSV files:

- `outputs/FlipFlopSingleStart/flipflop_julia_multistart_methods.csv`
- `outputs/WarfarinSingleStart/warfarin_julia_multistart_methods.csv`
- `outputs/FlipFlopMultistart/flipflop_julia_multistart_methods.csv`
- `outputs/WarfarinMultistart/warfarin_julia_multistart_methods.csv`

The flip-flop script also writes the simulated start bank and NONMEM-style data used for matched comparisons. The warfarin script writes the sampled start bank used in the run.
