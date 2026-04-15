# ML Physics Parameterization

This is the supported joint ML physics workflow for the current analytical
AMIP experiments.

## Scope

The workflow is intentionally narrow:

- analytical AMIP only
- one joint model for turbulence and convection
- physical baseline: `louis + mass_flux`
- ML runtime: same physical path plus `physics_parameterization="ml"`

There are no separate public turbulence-only, convection-only, or SBM-based ML
workflows in this path.

## Runtime Contract

The joint model predicts one flat output vector per atmospheric column:

- `Km[nlev]`
- `Kh[nlev]`
- `M_eq`

The runtime remains hybrid:

- `Km/Kh` are injected into the Louis vertical diffusion machinery
- `M_eq` is injected into the mass-flux closure
- `M_c` remains a prognostic physical state input
- `tau_adj` remains fixed in the physical mass-flux scheme

So the ML model does not predict tendencies directly. It predicts closure
variables used by the existing physical solvers.

## Workflow Script

The user-facing entrypoint is:

- `scripts/ml_physics_parameterization.py`

It performs one linear workflow:

1. run a 28-day analytical AMIP baseline with physical `louis + mass_flux`
2. sample atmospheric columns from that run
3. compute joint teacher targets `Km/Kh + M_eq`
4. train one joint model
5. run a 28-day analytical AMIP full-ML case
6. write a comparison package

## Output Layout

The default output root is:

- `results/ml_physics_parameterization`

It contains:

- `training/`
- `default_run/`
- `full_ml_run/`
- `comparison/`
- `workflow_summary.json`

Key comparison products:

- `results/ml_physics_parameterization/comparison/timeseries_compare.png`
- `results/ml_physics_parameterization/comparison/final_profiles_compare.png`
- `results/ml_physics_parameterization/comparison/maps_final.png`
- `results/ml_physics_parameterization/comparison/difference_maps_final.png`
- `results/ml_physics_parameterization/comparison/summary.json`

## Current Result

The current trained workflow used `4224` sampled columns from days:

- `0, 1, 2, 3, 4, 5, 6, 7, 14, 21, 28`

Saved training metrics are in:

- `results/ml_physics_parameterization/training/metrics.json`

Final analytical full-ML minus default differences are small:

- `dT_atm = -0.0138 K`
- `dT_low = +0.0426 K`
- `dCWV = +0.0920 kg/m²`
- `dprecip = 0.0 mm/day`
- `dmax_wind = +0.00248 m/s`

## Validation

Focused validation for the active path currently passes:

- `tests/unit/test_run_amip_cli.py`
- `tests/unit/test_config_validation.py`
- `tests/unit/test_config_roundtrip.py`
- `tests/unit/test_ml_physics_parameterization.py`
- `tests/unit/test_ml_physics_workflow.py`

One unrelated broader turbulence-suite assertion in `YSU` can fail, but it is
outside the joint ML physics path implemented here.

## Note On Precipitation

The analytical `gray + louis + mass_flux` setup is extremely dry. Precipitation
is not missing from diagnostics, but it is very small, peaks early, and decays
toward zero by the end of the 28-day run. That behavior comes from the physical
mass-flux baseline in this analytical forcing setup, not from the plotting
pipeline.
