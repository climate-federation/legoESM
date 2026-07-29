# MPAS Single-Point T500 Variational Assimilation

This configuration bundle reproduces a six-hour MPAS single-observation
comparison among 3DVar, nonlinear strong-constraint 4DVar, and the
tangent-linear evolution of the 4DVar initial increment.

## Experiment

The observation is 500 hPa temperature at the MPAS cell nearest 45 degrees
north and 50 degrees west. The synthetic innovation is +2 K and the
observation-error standard deviation is 1 K. The 3DVar observation is applied
at the initial time, while the 4DVar observation is applied at the end of the
six-hour window.

The background is the 24-hour MPAS forecast from NMC case 009. It was
initialized from ERA5 at 00 UTC 14 January 2017 and is valid at 00 UTC
15 January 2017. The model uses an MPAS resolution-6 mesh, 40 sigma levels,
and a 60-second time step.

The static background-error covariance is fitted with GEN_BE from 80 MPAS
48-hour-minus-24-hour forecast differences valid at common times. The selected
balance configuration retains the fitted temperature balance and removes the
surface-pressure balance regression. The assimilation multiplies the fitted
horizontal length scales and background-error standard deviations by three.

## Configuration

- `nmc.yaml` controls ERA5 ingestion, NMC sample generation, and GEN_BE fitting.
- `experiment.yaml` controls the observation, assimilation, diagnostics, and
  figure layout.

Large ERA5, NMC, GEN_BE, and experiment outputs are external artifacts and are
not stored in Git. Paths may be overridden on the command line. The ERA5 RDA
root may also be set with `LEGOESM_ERA5_RDA_ROOT`.

## Commands

Run all model and JAX workloads inside a scheduler allocation.

```bash
python scripts/data/generate_mpas_nmc.py \
  --config config/4DVar_single/nmc.yaml

python scripts/data/fit_mpas_gen_be.py \
  --config config/4DVar_single/nmc.yaml

python scripts/run/mpas_4dvar_single/run_assimilation.py \
  --config config/4DVar_single/experiment.yaml

python scripts/run/mpas_4dvar_single/run_tlm.py \
  --config config/4DVar_single/experiment.yaml

python scripts/plot/plot_mpas_single_point_t500.py \
  --config config/4DVar_single/experiment.yaml
```

The final figure contains 3DVar, nonlinear 4DVar, and tangent-linear rows at
the initial time and at two-, four-, and six-hour lead times. Filled contours
show the 500 hPa temperature increment, black contours show background 500 hPa
geopotential height, and the cross identifies the observation cell.
