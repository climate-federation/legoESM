# Phase F -- Climate-scale ocean run drivers

## What landed

Two cluster-bound drivers + two acceptance-template reports:

* ``scripts/ocean_long_runs/run_omip2.py`` -- 30-year forced ocean run
  driven by JRA55-do forcing through the Large & Yeager 2009 bulk-flux
  module. Lat-lon C-grid @ 1 deg or MPAS @ 100 km. Per-year RPE / energy
  / tracer diagnostics + restart writer.
* ``scripts/ocean_long_runs/run_bryan_thc.py`` -- 500-1000 yr Bryan
  1986/1987 THC spinup on an idealised hemispheric basin, CORE-II NYF
  forcing. Diagnostic cadence configurable (default every 10 model
  years).
* ``scripts/ocean_long_runs/__init__.py`` -- empty package marker.
* ``docs/ocean_long_runs/results_omip2_skeleton.md`` -- acceptance
  template with the Tsujino 2020 multi-model bars (AMOC 15+/-3 Sv,
  ACC 130+/-15 Sv, SST bias <1.5 K) and an empty per-year results
  table for the cluster to fill in.
* ``docs/ocean_long_runs/results_bryan_thc_skeleton.md`` -- acceptance
  template with the Bryan 1987 bars (MOC 15-20 Sv, heat drift <1 W/m^2
  after 100 yr) and an empty long-spinup results table.

## Smoke verification (local laptop)

Both drivers pass the ``--smoke`` path (one model day, synthetic
forcing fallback):

```
$ python scripts/ocean_long_runs/run_omip2.py --smoke --output ...
==> Building global rest-state on latlon/36x72
   RPE_0 = -7.7476e+25 J  |  KE_0 = 0.000e+00  |  vol_0 = 2.764e+18
==> Year 1/1 (48 steps)
   RPE_flux = -3.683e-02 W/m^2  |  KE = 0.000e+00  |  vol_drift = 0.000e+00
Wall time: 1.1s

$ python scripts/ocean_long_runs/run_bryan_thc.py --smoke --output ...
==> Building Bryan hemispheric basin @ 24x24
   RPE_0 = -3.7683e+24  |  vol_0 = 1.855e+17
==> Year 1/1
Wall time: 0.7s
```

Both write a ``summary.json`` matching the production-run schema +
per-year restart .npz files (Phase C restart harness).

## Production readiness

The drivers exercise every code path end-to-end **except** the bulk-
flux applicator that ports ``(tau_x, tau_y, shflx, lhflx)`` from
``bulk_flux_omip.air_sea_fluxes`` into the state's surface boundary
fields. The smoke driver computes the bulk fluxes but does NOT yet
apply them (documented in each driver's docstring + the report
skeletons).

The remaining wiring is one thin coupler hook that should live in
``legoesm/ocean/physics/surface_forcing/prescribed.py`` or
``legoesm/ocean/coupler/`` -- the gyre runners already do exactly this
for prescribed wind stress, so the OMIP-2 driver just needs to plug
the bulk-flux output into the same surface-forcing wiring. Tracked
as the **last** Phase F follow-up.

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
