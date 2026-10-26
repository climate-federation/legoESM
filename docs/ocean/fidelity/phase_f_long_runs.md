# Phase F -- Climate-scale ocean run drivers

## What landed

Two cluster-bound drivers + two acceptance-template reports:

* ``scripts/run/ocean_long_runs/run_omip2.py`` -- 30-year forced ocean run
  driven by JRA55-do forcing through the Large & Yeager 2009 bulk-flux
  module. Lat-lon C-grid @ 1 deg or MPAS @ 100 km. Per-year RPE / energy
  / tracer diagnostics + restart writer.
* ``scripts/run/ocean_long_runs/run_bryan_thc.py`` -- 500-1000 yr Bryan
  1986/1987 THC spinup on an idealised hemispheric basin, CORE-II NYF
  forcing. Diagnostic cadence configurable (default every 10 model
  years).
* ``scripts/run/ocean_long_runs/__init__.py`` -- empty package marker.
* ``docs/ocean/long_runs/results_omip2_skeleton.md`` -- acceptance
  template with the Tsujino 2020 multi-model bars (AMOC 15+/-3 Sv,
  ACC 130+/-15 Sv, SST bias <1.5 K) and an empty per-year results
  table for the cluster to fill in.
* ``docs/ocean/long_runs/results_bryan_thc_skeleton.md`` -- acceptance
  template with the Bryan 1987 bars (MOC 15-20 Sv, heat drift <1 W/m^2
  after 100 yr) and an empty long-spinup results table.

## Smoke verification (local laptop)

Both drivers pass the ``--smoke`` path (one model day, synthetic
forcing fallback):

```
$ python scripts/run/ocean_long_runs/run_omip2.py --smoke --allow-synthetic --output ...
==> Building global rest-state on latlon/36x72
   RPE_0 = -7.7476e+25 J  |  KE_0 = 0.000e+00  |  vol_0 = 2.764e+18
==> Year 1/1 (48 steps)
   RPE_flux = -3.683e-02 W/m^2  |  KE = 0.000e+00  |  vol_drift = 0.000e+00
Wall time: 1.1s

$ python scripts/run/ocean_long_runs/run_bryan_thc.py --smoke --output ...
==> Building Bryan hemispheric basin @ 24x24
   RPE_0 = -3.7683e+24  |  vol_0 = 1.855e+17
==> Year 1/1
Wall time: 0.7s
```

Both write a ``summary.json`` matching the production-run schema +
per-year restart .npz files (Phase C restart harness).

## Production readiness

End-to-end wiring complete after the
``legoesm.ocean.coupler.apply_omip2_surface_fluxes`` follow-up:

* ``src/legoesm/ocean/coupler/omip2_applicator.py`` ports the
  Large & Yeager 2009 bulk-flux output (``tau_x``, ``tau_y``,
  ``shflx``, ``lhflx``) into the state's top-layer u / v / T
  fields via a forward-Euler ``rho_0 * c_p * dz_0`` rescaling
  + C-grid face-interpolation of the cell-centred stresses.
* Both ``run_omip2.py`` and ``run_bryan_thc.py`` call the
  applicator once per timestep before ``model.step``; the
  per-year diagnostics now reflect the genuine wind-driven
  + thermally-forced response.
* Smoke results after wiring (1 day, synthetic forcing):
  - OMIP-2: ``KE`` grows 0 -> 1.6e16 J, ``vol_drift`` ~ -5e-10
    (float noise), ``eta_integral`` ~ -1e5 m^3 (Ekman-pumping
    response).
  - Bryan: ``KE`` grows 0 -> 2.2e15 J on the smaller hemispheric
    basin.

Unit tests ``tests/ocean/unit/test_omip2_applicator.py`` -- **4 / 4
pass**:

* Applicator returns the same state-type with finite fields.
* Wind injects nonzero top-cell u and v from rest; deeper levels
  stay at rest after a single step (no vertical mixing in the
  applicator -- that is the dycore's job).
* Top-cell T responds to the surface heat flux without runaway.
* Raises ``NotImplementedError`` on unsupported grid types
  (cube + MPAS extension is a follow-up).

## Acceptance gate

Phase F **infrastructure** complete:

* Two production drivers land and execute end-to-end at the smoke
  level.
* Per-year diagnostics + restart cadence match the acceptance bars.
* Report skeletons enumerate the metrics and tolerances against which
  the cluster runs will be evaluated.

The **acceptance results** for the bulletproof claim (AMOC 15+/-3 Sv,
ACC 130+/-15 Sv, SST bias <1.5 K, MOC 15-20 Sv at Bryan equilibrium,
RPE drift <0.5 mW/m^2) require the cluster to fire the production
drivers + the missing bulk-flux applicator. That is the next
work item; this commit closes the infrastructure side.
