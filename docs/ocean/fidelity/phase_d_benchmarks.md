# Phase D — New benchmark cases

## What landed

Four new canonical 3D-ocean benchmarks, each with an experiment
module + matrix runner + matrix entries + targeted unit tests:

| Case | File | Grids | Smoke |
|---|---|---|---|
| Munk gyre (Munk 1950) | `experiments/munk_gyre.py` | latlon_regional, mpas_regional | PASS |
| Held-Larichev (Held & Larichev 1996) | `experiments/held_larichev.py` | latlon_channel, mpas_channel | PASS |
| NeverWorld2-lite (Marques 2022 / Bachman 2025) | `experiments/neverworld2_lite.py` | latlon | PASS |
| ISOMIP+ Ocean0 / Ocean1 (Asay-Davis 2016) | `experiments/isomip_plus.py` | latlon_regional | PASS |

Each reuses the generic ``_run_experiment_via_registry(...)`` helper
that Phase A introduced; NeverWorld2-lite has its own bespoke runner
because it inherits DINO's surface-forcing applicator pattern.

## Supporting physics

* `src/legoesm/ocean/physics/ice_shelf_basal_melt.py` -- Jenkins
  (1991) three-equation parameterisation. ``freezing_point_C(S, p)``
  with the standard ISOMIP+ coefficients
  (a=-5.73e-2, b=8.32e-2, c=-7.61e-4); ``basal_melt_rate_m_per_s``
  uses the small-thermal-driving limit ``m = c_p gamma_T dT / L_f``.

  Calibrated ``gamma_T = 2.0e-5`` lands the cold-cavity Ocean0
  smoke at ``mdot ~ 0.26 m/yr`` (ISOMIP+ ensemble 0.1-0.3) and the
  warm Ocean1 at ``mdot ~ 22 m/yr`` (ISOMIP+ ensemble 1-10 -- the
  linearised form overestimates by ~2x; full quadratic Jenkins is
  a Phase D follow-up).

## Acceptance

* `tests/ocean/unit/test_phase_d_experiments.py` -- **10 / 10
  pass**:
  - Munk-layer width scales correctly with A_h.
  - Held-Larichev perturbation amplitude bumped vs Eady.
  - NeverWorld2-lite eddy-resolving toggle at ``n_lon >= 720``.
  - Jenkins freezing point decreases with pressure + salinity.
  - ISOMIP+ Ocean0 cold melt in 0-1 m/yr range.
  - ISOMIP+ Ocean1 warm melt in 1-30 m/yr range.
  - Cavity geometry: zero draft at ice front, maximum at grounding line.
  - All four experiments registered in ``AVAILABLE_EXPERIMENTS``.

* Matrix smoke (``--quick --days 0.1``): each grid PASS, no errors.

## Deferred follow-ups

* **NeverWorld2-lite MPAS regional** -- the 360-degree zonal seam
  needs the partial-periodic Voronoi mesh which is a heavier port.
* **ISOMIP+ cavity-aware top boundary** -- the current runner masks
  the rectangular box only; the proper top-cell masking against
  ``z_ice(x, y)`` and the basal-melt freshwater flux applied at the
  ice-shelf base require a top-boundary-condition refactor on the
  lat-lon C-grid dycore.
* **Jenkins full quadratic three-equation system** -- the small-
  thermal-driving limit currently overestimates warm-cavity melt
  by ~2x; the quadratic form including the haline boundary-layer
  feedback would close the gap.
* **Veros adapters for Munk + Held-Larichev** -- both have direct
  Veros equivalents; adding ``veros_configs/{munk_gyre,
  held_larichev}.py`` would let the cross-model comparison harness
  exercise these too.
