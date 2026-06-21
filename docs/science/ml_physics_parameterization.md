# ML Physics Parameterization

This is the supported joint ML physics workflow for the current analytical
AMIP experiments.

## Scope

The workflow is intentionally narrow:

- analytical AMIP only
- one joint model for `louis + mass_flux`
- optional moist extension for `sundqvist` microphysics
- runtime stays hybrid: the ML model augments the existing physical solvers

There are no separate public turbulence-only, convection-only, or SBM-based ML
workflows in this path.

## Runtime Contract

In the default dry path, the joint model predicts one flat output vector per
atmospheric column:

- `Km[nlev]`
- `Kh[nlev]`
- `M_eq`

Those outputs are injected back into the physical Louis and mass-flux schemes:

- `Km/Kh` drive the Louis vertical diffusion closure
- `M_eq` drives the mass-flux closure
- `M_c` remains a prognostic physical state input
- `tau_adj` remains fixed in the physical mass-flux scheme

So the ML model does not replace the full tendency machinery. It predicts
closure variables that are consumed by the existing physics.

## Moist Sundqvist Extension

When `microphysics="sundqvist"` is enabled, the same column model also ingests
`q_c[nlev]` and predicts one additional scalar per column:

- `rain_survival_fraction`

This is defined from the physical Sundqvist teacher as:

- `precip / sum(P_auto * rho * dz)`

At runtime, the workflow keeps physical Sundqvist condensation and
autoconversion, and only uses the learned scalar to rescale how much generated
rain survives to the surface. The physical Sundqvist algebra then rebuilds:

- evaporation
- latent heating
- `dq_v_dt`
- `dq_c_dt`
- `dq_r_dt`
- surface precipitation

That is the current supported moist target. Earlier experiments with direct
`dq_v_dt`, condensation, `P_auto`, and evaporation targets were not kept as the
final path.

## Workflow Script

The user-facing entrypoint is:

- `scripts/run/ml_physics_parameterization.py`

It performs one linear workflow:

1. run an analytical AMIP baseline with physical `louis + mass_flux`
2. sample atmospheric columns from that run
3. compute joint teacher targets
4. train one joint model
5. run an analytical AMIP full-ML case
6. write a comparison package

When `--microphysics sundqvist` is selected, step 1 switches to:

- `rrtmgp + sundqvist clouds + sundqvist microphysics + louis + mass_flux`

and step 3 captures:

- `Km`
- `Kh`
- `M_eq`
- `rain_survival_fraction`

The script defaults still reflect the original dry workflow. The current moist
canonical run is specified explicitly through CLI flags.

## Output Layout

The canonical output root is:

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

The comparison maps now show only:

- `T_low`
- `q_v_low`
- `precip`
- `wind`

For very coarse cubed-sphere runs (`C8` and lower), those maps use native point
rendering instead of interpolated filled lat-lon fields to avoid fake
polar/seam artifacts.

## Current Canonical Result

The current canonical moist run is:

- `resolution=16`
- `nlev=8`
- `dt=180 s`
- `days=7`
- `sample_days=0,1,2,3,4,5,6,7`
- `radiation=rrtmgp`
- `cloud_scheme=sundqvist`
- `microphysics=sundqvist`
- `convection=mass_flux`
- `turbulence=louis`

Saved training metrics are in:

- `results/ml_physics_parameterization/training/metrics.json`

That run used `12288` sampled columns and trained a stable online-coupled moist
workflow. Final analytical full-ML minus default differences are small:

- `dT_atm = 0.0 K`
- `dT_low = +0.00119 K`
- `dCWV = +4.58e-05 kg/m²`
- `dprecip = -1.40e-04 mm/day`
- `dmax_wind = -5.54e-05 m/s`

Radiative differences also remain small:

- `dSW_up_TOA = -0.128 W/m²`
- `dLW_up_TOA = -0.0115 W/m²`
- `dSW_net_sfc = +0.144 W/m²`
- `dLW_net_sfc = +0.00446 W/m²`

## Validation

Focused validation for the active path currently passes:

- `tests/unit/test_ml_physics_parameterization.py`
- `tests/unit/test_ml_physics_workflow.py`

The broader AMIP round-trip issue in `tests/unit/test_config_roundtrip.py` is
still unrelated to this workflow.

## Notes

The dry `gray + louis + mass_flux` workflow is still supported and remains the
fastest baseline path for debugging.

The current moist extension is supported at the canonical `C16/L8, dt=180 s,
days=7` scale. Larger or longer moist configurations should still be treated as
new stability experiments, not assumed to work automatically.
