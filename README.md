# Adaptive two-moment sensor fusion

Reproducible simulation code for evaluating adaptive multisensor state
estimation under nonstationary sensor reliability. The study uses a simulated
differential-drive ground vehicle with GNSS, odometry, gyroscope, and
magnetometer measurements. The estimators are evaluated in open loop and in
closed loop with the same controller.

The proposed estimator combines two innovation statistics. The normalized
innovation squared controls bounded measurement-covariance inflation. An
exponentially weighted mean of the whitened innovation detects persistent
mean shifts. The response can exclude a channel or activate bias-state process
noise.

## Repository contents

- `src/estimhyb/` contains the plant, sensors, controller, filters, adapters,
  degradation models, and batch execution code.
- `scripts/` contains validation, simulation, statistical analysis, figure,
  and table generation scripts.
- `configs/tuning.json` records the nominal process-noise calibration.
- `requirements.txt` records the tested Python dependencies.

Generated data, logs, tables, and figures are intentionally not versioned.
They are recreated by the commands below.

## Requirements

- Python 3.12 or newer
- Sufficient free disk space for generated CSV, JSON, figure, and table files

Create and activate an isolated environment in PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with
`source .venv/bin/activate`.

## Quick implementation checks

Run the checks in numerical order.

```powershell
python scripts/01_smoke_plant_sensors.py
python scripts/02_smoke_control.py
python scripts/03_smoke_ekf.py
python scripts/04_tune_nominal.py
python scripts/06_smoke_batch.py
```

The calibration script rewrites `configs/tuning.json`. Commit or record that
file if a different numerical environment produces a different calibration.

## Full reproduction workflow

Run all commands from the repository root. The default confirmatory runs use
40 paired repetitions per cell.

```powershell
python scripts/08_confirmatory.py 40
python scripts/09_negative_control.py
python scripts/10_hypotheses.py
python scripts/11_frontier_cost.py
python scripts/12_fine_frontier.py 40
python scripts/13_confirmatory_v2.py 40
python scripts/14_verification.py 20
python scripts/16_analysis_v2.py
python scripts/15_figures.py
python scripts/17_tables.py
python scripts/22_sensitivity_review.py
```

The scripts create these directories as needed.

- `results/primary/` contains simulation-level CSV and JSON outputs.
- `tables/` contains analysis tables.
- `figures/` contains working figures.
- `submission_figures/` contains seven numbered PNG and PDF figures.
- `submission_tables/` contains numbered CSV and Markdown tables.

The sensitivity script removes repeated nominal conditions from the rank
analysis and applies a five-second final-dwell requirement to recovery.

The main output files are
`results/primary/confirmatory_runs.csv` with 9600 rows and
`results/primary/confirmatory_v2_runs.csv` with 12000 rows when the default
40 repetitions are used.

## Reduced test run

For a faster functional test, use two repetitions. These outputs verify that
the pipeline runs but do not reproduce the study estimates.

```powershell
python scripts/08_confirmatory.py 2
python scripts/12_fine_frontier.py 2
python scripts/13_confirmatory_v2.py 2
```

Do not mix reduced-run outputs with the 40-repetition analysis.

## Reproducibility notes

- Random seeds are generated deterministically from scenario identifiers and
  fixed study salts.
- All estimators in a cell use the same noise realization, enabling paired
  comparisons.
- The two confirmatory runs use disjoint seed salts.
- Figures and tables are generated from the saved confirmatory CSV files.
- Computational-cost results describe the vectorized Python implementation
  and should not be interpreted as embedded execution times.

## Citation

If you use this software, cite the associated research article and the
repository.

```text
fjburgosf. Adaptive two-moment sensor fusion. GitHub repository.
https://github.com/fjburgosf/adaptive-two-moment-sensor-fusion
```

## Contact

Repository owner: [fjburgosf](https://github.com/fjburgosf)
