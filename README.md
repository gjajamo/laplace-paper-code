# Laplace AD Paper Code

This repository contains the Julia code used for the revised manuscript analyses comparing finite-difference and automatic-differentiation gradients for the Laplace approximation of the marginal likelihood in nonlinear mixed-effects models.

The code includes two examples:

- `flipflop_single_start.jl` and `flipflop_multistart.jl`: one-compartment population PK example with absorption/elimination ambiguity.
- `warfarin_single_start.jl` and `warfarin_multistart.jl`: public warfarin PK/PD example with an ODE effect-compartment representation.

The current code path is Julia-based. The subject-level eta derivatives use forward-mode automatic differentiation, while the population-level methods implement the final manuscript versions of finite differences (FD), FULL-implicit, FULL-unroll, STOP, and the staged Warfarin variants STOP+FULL and FULL+STOP.

## Contents

- `Project.toml` and `Manifest.toml`: Julia environment used by the computation scripts.
- `src/flipflop_multistart_methods.jl`: flip-flop model, Laplace objective, and optimization methods.
- `src/warfarin_multistart_methods.jl`: Warfarin Laplace objective, ODE effect-compartment solver, and optimization methods.
- `src/warfarin_subset_methods.jl`: matched-start subset runner used for focused Warfarin checks.
- `data/flipflop/`: fixed simulated subjects, start bank, true etas, and cached contour slice for the flip-flop example.
- `data/warfarin/`: public Warfarin data and cached contour slices, including the final high-logit(Emax) slice used for the manuscript figure.
- `scripts/`: figure-generation scripts for the manuscript contour and runtime panels.

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

The multistart wrappers use the fixed manuscript datasets by default. They are configured for reproducible reruns, but the exact manuscript-scale runs can still take a long time, especially FD and FULL-unroll.

```bash
julia --project=. flipflop_multistart.jl
julia --project=. warfarin_multistart.jl
```

Useful controls include:

- `FLIPFLOP_JULIA_N_STARTS` and `WARFARIN_JULIA_N_STARTS`: number of starts.
- `FLIPFLOP_JULIA_METHODS`: comma-separated subset of `FULL_IMPLICIT,FULL_UNROLL,STOP,FD`.
- `WARFARIN_JULIA_METHODS`: comma-separated subset of `FULL_IMPLICIT,FULL_UNROLL,STOP,FD,STOP+FULL,FULL+STOP`.
- `FLIPFLOP_JULIA_MAXITER_OUTER` and `WARFARIN_JULIA_MAXITER_OUTER`: maximum outer iterations.
- `FLIPFLOP_JULIA_MAXITER_ETA` and `WARFARIN_JULIA_MAXITER_ETA`: maximum Newton iterations for subject-specific eta modes.
- `WARFARIN_JULIA_DT`: internal ODE integration resolution for the Warfarin effect-compartment equation. The manuscript uses `0.25` hours.
- `FLIPFLOP_JULIA_OUTDIR` and `WARFARIN_JULIA_OUTDIR`: output directories.

## Outputs

Default outputs are CSV files:

- `outputs/FlipFlopSingleStart/flipflop_julia_multistart_methods.csv`
- `outputs/WarfarinSingleStart/warfarin_julia_multistart_methods.csv`
- `outputs/FlipFlopMultistart/flipflop_julia_multistart_methods.csv`
- `outputs/WarfarinMultistart/warfarin_julia_multistart_methods.csv`

The flip-flop script also writes the start bank and NONMEM-style data used for matched comparisons. The Warfarin script writes the sampled start bank used in the run. Runtime outputs under `outputs/`, `logs/`, and `tables/` are intentionally ignored by Git.
