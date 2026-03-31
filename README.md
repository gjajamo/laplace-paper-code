# FOCEI Single-Start Examples

This folder contains two standalone Python scripts used to reproduce the single-start FOCEI comparisons from the manuscript:

- `flipflop_single_start.py`
- `warfarin_single_start.py`

The scripts are trimmed from the full manuscript analysis code so they run only the single-start comparisons and write a compact set of figures and tables. They are not intended to reproduce the full multistart analyses or the complete manuscript figure set.

## Contents

- `flipflop_single_start.py`: self-contained single-start comparison for the one-compartment subcutaneous absorption model with absorption/elimination ambiguity. It compares FD, STOP, FULL-unroll, and FULL-implicit.
- `warfarin_single_start.py`: single-start joint PK/PD fit for the public Warfarin dataset. It compares FULL-unroll, FULL-implicit, STOP, STOP+FULL, FULL+STOP, and FD.
- `data/warfarin_dat.csv`: bundled public Warfarin dataset used by the Warfarin script.
- `requirements.txt`: minimal Python dependencies.
- `LICENSE`: conservative placeholder license text that should be replaced with the institution-approved public license before release.

## Recommended Environment

Python 3.11 was used for development.

Install dependencies with:

```bash
pip install -r requirements.txt
```

## How To Run

Both scripts write results to the current directory by default. To keep outputs organized, it is usually easier to give each run its own output folder.

### FlipFlop

```bash
python flipflop_single_start.py --output-dir outputs/FlipFlopSingleStart
```

Expected outputs include:

- `figures/fig_simulated_data_summary.png`
- `figures/fig_singlefit_runtime_bar_minutes.png`
- `tables/focei_single_run_comparison.csv`
- `tables/flipflop_single_start_summary.csv`

### Warfarin

```bash
python warfarin_single_start.py --output-dir outputs/WarfarinSingleStart
```

Expected outputs include:

- `tables/warfarin_single_start_summary.csv`
- `tables/result_<METHOD>_<FIELD>.csv` files moved into `tables/`

The Warfarin script uses the bundled `data/warfarin_dat.csv`. If that file is missing, the script falls back to the original notebook behavior and attempts to locate or download the dataset.

## Notes

- These scripts are intended for manuscript reproducibility and are deliberately narrower than the full project code.
- The FlipFlop script simulates its own dataset and does not require external inputs.
- The Warfarin script can still take substantial time, especially for the finite-difference method.
- Matplotlib is configured with the non-interactive `Agg` backend so the scripts can run in headless environments.

## Suggested Repository Layout

If you publish this folder as a standalone repository, the manuscript can point directly to the repository root and, ideally, to a Zenodo DOI for a tagged release.
