# Dycore Progression Validation Suite

Runner:
- `tests/validation/run_dycore_progression_suite.py`

Default output location:
- `tests/validation/results/dycore_progression_suite`

## Purpose

Provides one ordered atmospheric dynamical-core ladder with consistent per-case artifacts:

1. Shallow-water: spectral lat-lon, FV lat-lon, FV cube-sphere
2. Hydrostatic: FV cube, FV lat-lon, spectral (Held-Suarez + baroclinic)
3. Non-hydrostatic: FV cube + spectral (TC1/TC2a/TC3)

All maps are generated with:
- natural lat-lon projection by default (`platecarree`)
- no coastlines by default

## Standardized artifacts per case

- `field_snapshots.png`
- `snapshot_times.txt`
- `mean_timeseries.csv`
- `mean_timeseries.png`
- `conservation_timeseries.csv`
- `conservation_timeseries.png`
- `mean_profiles.csv`
- `mean_profiles.png`
- `standardized_manifest.json`

## Usage

Run full suite:

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python tests/validation/run_dycore_progression_suite.py
```

Prepare only (write plan/manifest without running):

```bash
.venv/bin/python tests/validation/run_dycore_progression_suite.py --skip-run
```

Optional map style:

```bash
.venv/bin/python tests/validation/run_dycore_progression_suite.py --projection robinson --coastlines
```
