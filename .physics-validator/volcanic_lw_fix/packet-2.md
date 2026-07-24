# Adversarial review request — Volcanic LW aerosol fix, ROUND 2 (LegoESM)

You are an independent adversarial physics reviewer. This is round 2. Your
round-1 review is summarized below with my response + the fix for each point.
Re-examine: (a) are the round-1 findings actually resolved, (b) did the fixes
introduce new bugs, (c) any remaining sign/unit/conservation/gradient/test
defect. Cite file:line. If no substantive findings remain, say so explicitly.
Working tree is available (cd = repo root). Unified diff:
`.physics-validator/volcanic_lw_fix/fix.diff` (pasted at end).

## Round-1 findings and resolutions

1. **B2 geometric-vs-geopotential height (CONFIRMED, FIXED).** You were right:
   the file altitude is geometric but the USSA base heights are geopotential
   (uncorrected p low by 0.98%/2.36%/3.31% at 20/32/39.75 km — I reproduced
   your numbers exactly). `_ussa1976_pressure` now converts
   `h = R_e z/(R_e+z)` (R_e = `constants.R_earth`) before the table lookup
   (`external.py`, the function). The test now passes the geometric heights
   that map to the geopotential bases and recovers the published base
   pressures, and asserts the correction is applied.

2. **B2 "exact fractional-pressure-overlap" is not exact — source OD is uniform
   in height, not pressure (CONFIRMED, FIXED).** I verified: for a 0.5 km layer
   a linear-pressure split gives 0.5098 where the true (height-uniform) split
   is 0.5000; a LOG-pressure split gives 0.5000. `place_stratospheric_aod_
   profile_to_layers` now interpolates the cumulative-OD CDF in LOG-pressure
   (≈ geometric height, the coordinate the source `ext*dz` is uniform in),
   exact at layer edges, still conservative. The docstring no longer claims
   "exact fractional-pressure-overlap"; it states the log-pressure within-layer
   assumption.

3. **B3 "the correct gray equivalent" overclaim (CONFIRMED, FIXED).** Reworded
   to a Planck-weighted gray HEURISTIC — strictly better than a flat mean (which
   overweights the ~zero-flux near-IR bands) but NOT the exact gray equivalent
   (which would fold in the full spectral radiative kernel). Documented as an
   approximation.

4. **B3 temperature-sensitivity claim false (CONFIRMED, FIXED).** I reproduced
   your number: ±10 K moves the Planck-weighted gray column OD by ∓4.7% (not
   <2%). The `_VOLC_LW_PLANCK_TEMP_K` comment now states ~±4.7%.

5. **"All ops are smooth" — actually piecewise-linear / a.e.-differentiable
   (CONFIRMED, FIXED).** Docstring now says piecewise-linear, a.e.-
   differentiable with well-defined one-sided gradients (kinks at edges),
   JIT/vmap-safe. The FD test validates the a.e. gradient off-kink; the
   `delta=1e-4` is the grad-vs-FD tolerance, not the FD step (eps = 1 Pa).

6. **Missing mechanical gate test + no source→OLR regression (CONFIRMED,
   FIXED).** The `_ext_forcing` boolean is extracted to a pure module-level
   `_external_forcing_active(...)` called at both sites; `TestExternalForcing
   Gate` pins that LW-aerosol-only opens the gate and that a non-gas-radiation
   scheme closes it. `TestLwAerosolOlrSign` runs RRTMGP `solve_columns` with a
   synthetic stratospheric absorption bump and asserts OLR drops (correct
   warming sign) by a non-trivial amount — an end-to-end absorption-slot sign
   test. The stale "backend spy" module docstring is rewritten.

7. **Centroid test uses latitude-independent USSA edges (ACKNOWLEDGED, not a
   bug).** The placement uses a CLIMATOLOGICAL (lat-independent) z→p by design:
   `_precompute_external_forcing` has no per-column geopotential at forcing
   time. The centroid test validates the SOURCE profile is stratospheric (its
   purpose). Lat-dependent placement is a documented residual, not in scope for
   a background-magnitude term.

8. **SW volcanic shares the defect (ACKNOWLEDGED, filed).** SW `get_aerosol_at_
   time` sums background Kinne + volcanic into one column then distributes by
   pressure mass, and the SW slot is extinction-with-scattering (needs
   `omega_sun`/`g_sun`), so a faithful fix is not cheap. I agree the
   vertical-only fix (distribute background vs volcanic separately) is more
   contained; it is filed as fast-follow in the report, not done here (task
   said fix SW only "if same-shaped and cheap").

9. **Unrelated Bechtold diff (OUT OF SCOPE).** `model_driver.py:5795-5830` and
   `physics_pipeline.py convection_config_for` are a SEPARATE landed fix; my
   task explicitly forbade touching them. They are pre-existing uncommitted
   changes in the tree, not part of this LW-aerosol fix. Please review only the
   LW-aerosol diff below.

## Re-verified numbers (CPU, x64)

- All 16 tests in `tests/unit/test_volcanic_lw_aerosol.py` pass (incl. the new
  gate + OLR-sign + geopotential-corrected USSA tests).
- Collateral unchanged: `test_convection_config_for.py` 6/6,
  `test_bechtold_column_conservation.py` 3/3.
- Tropical-column OLR impact (1979, RRTMGP solve_columns), unchanged by the
  sub-percent placement refinements: baseline 271.308 W/m²; FIX ΔOLR
  −0.040 W/m²; OLD (extinction-by-mass) −0.029 W/m²; fix−old −0.011 W/m².
  Background year ⇒ immaterial in absolute terms; the sign and placement are
  what matter for eruption amplitudes.

## Attack surface for round 2
- Is the log-pressure CDF interp still exactly conservative (Σ_layers = total
  when the column spans the source), and correct when a model layer straddles
  several source layers?
- Does the geopotential conversion break monotonicity or the layer-boundary
  continuity of `_ussa1976_pressure` anywhere in 0–40 km?
- Any new gradient trap from `jnp.log` (p→0)? (pressures are ≥100 Pa here;
  a 1e-12 floor guards it.)
- Is `_external_forcing_active` wired identically at both call sites (scheme
  test differs: rrtmg+rrtmgp vs rrtmgp-only)?

## Full unified diff
See `.physics-validator/volcanic_lw_fix/fix.diff`.
