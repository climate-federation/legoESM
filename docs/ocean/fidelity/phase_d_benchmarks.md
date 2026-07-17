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

* `packages/ocean/legoesm/ocean/physics/ice_shelf.py` (`three_equation_melt`; the linearised `ice_shelf_basal_melt.py` closure was deleted 2026-07-17 -- hand-tuned gamma_T + rho_fw convention shortcut) -- Jenkins
  (1991) three-equation parameterisation. ``freezing_point_C(S, p)``
  with the standard ISOMIP+ coefficients
  (a=-5.73e-2, b=8.32e-2, c=-7.61e-4); ``three_equation_melt`` solves
  the FULL Holland-Jenkins quadratic (interface salt balance included)
  at ``gamma_T = 1.0e-4`` — no hand calibration.  The smoke probe feeds
  far-field T as the boundary-layer ambient, so it reads Ocean0
  ``~1.5`` / Ocean1 ``~65 m/yr`` ICE-equivalent (documented
  overestimate vs the 0.1-0.3 / 1-10 circulating-cavity ensemble; the
  meltwater-throttling feedback needs the cavity experiment).

## Acceptance

* `tests/ocean/unit/test_phase_d_experiments.py` -- **10 / 10
  pass**:
  - Munk-layer width scales correctly with A_h.
  - Held-Larichev perturbation amplitude bumped vs Eady.
  - NeverWorld2-lite eddy-resolving toggle at ``n_lon >= 720``.
  - Jenkins freezing point decreases with pressure + salinity.
  - ISOMIP+ Ocean0 cold melt brackets the faithful closure's value.
  - ISOMIP+ Ocean1 warm melt pinned rel-1e-12 against an independent
    quadratic solve + physical orderings (warm >> cold, T_f < T_b < T_amb).
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
* ~~Jenkins full quadratic three-equation system~~ -- DONE
  (2026-07-17): the linearised closure was deleted and all callers
  use ``ice_shelf.three_equation_melt`` (full quadratic with the
  haline boundary-layer feedback).
* **Veros adapters for Munk + Held-Larichev** -- both have direct
  Veros equivalents; adding ``veros_configs/{munk_gyre,
  held_larichev}.py`` would let the cross-model comparison harness
  exercise these too.
