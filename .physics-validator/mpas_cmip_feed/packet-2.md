# Adversarial review ROUND 2: MPAS monthly-CMOR accumulator feed

You are an independent adversarial reviewer. This is round 2. Below is your
round-1 review, my disposition of each finding, and the UPDATED code + tests.
Re-review: verify each claimed fix is actually correct, check I did not
introduce NEW bugs (esp. the `_tas_2m` refactor that the cube path also uses,
the atomic-commit restructuring, and the MPI gate), and confirm the
documented-as-inherited items are genuinely inherited/out-of-scope. Cite line
numbers. If no substantive bugs remain, say so and justify.

## Round-1 findings and my disposition

1. **`tas` was a lowest-level proxy under the CMOR name.** FIXED. The driver
   now computes the proper 2 m MOST temperature (`_tas_2m`) from the
   RECONSTRUCTED cell winds + prescribed `get_sst_sic`, and passes it to the
   collector as `tas`. `_tas_2m` gained optional `u_low`/`v_low` overrides so
   MPAS can pass cell winds (the cube path calls it unchanged → identical
   behaviour). The collector falls back to the lowest level only when the
   caller supplies no `tas` (e.g. no `get_sst_sic`). Tests:
   `test_feed_uses_provided_2m_tas`, `test_driver_helper_computes_2m_tas`.

2. **Daily/monthly are instantaneous end-of-interval samples; `pr` is the
   last physics-step rate.** DOCUMENTED, not fixed. The lean MPAS loop has no
   per-interval time integrator (unlike the compiled cube path that passes
   segment means). The method docstring now states the snapshot/diurnal-alias
   caveat, that `tasmin`/`tasmax` collapse at 1 sample/day, and that `pr` is a
   rate snapshot — with the device-side interval accumulator as the named
   follow-up. This matches the task's explicit allowance ("feed what IS
   available and document `pr` as a TODO"). A device-side accumulator is a
   separate, larger change (touches the physics-step precip export).

3. **MPI silently left CMOR empty; disabled a safe 1-rank case.** FIXED. The
   gate is now `_mpas_cmip_feed_enabled` (unit-tested): it PERMITS a 1-rank
   Voronoi layout (owns whole mesh, local weights == global) and disables only
   MULTI-rank, where it emits a LOUD rank-0 warning. Test:
   `test_feed_gate_serial_mpi_singlerank`.

4. **Below-ground plev extrapolation (`_interp_to_plev19`).** DOCUMENTED as
   inherited — identical to `collect()`; fixing the shared helper would change
   cube-path output, so it is a separate PR. Noted in the docstring.

5. **Defensive `except` could commit partial state.** FIXED. The collector
   method is now ATOMIC: PHASE 1 does every regrid / plev interpolation into
   local dicts (no accumulator touched); PHASE 2 does only the cheap,
   shape-checked `add_*` calls. A mid-computation failure commits nothing. The
   driver-level `try/except` stays loud-but-nonfatal (matches the existing
   `_save_cmor_accumulator_sidecar` convention; a 100-yr run must not abort on
   a diagnostic glitch).

6. **Inaccurate "free host sync" claim.** FIXED the comment (feed adds a few
   host transfers at DAILY cadence, negligible). You confirmed no retrace /
   gradient issue. Kept the implementation (materialize-once of the np arrays
   at the top of the method).

7. **Zonal means unweighted by cell area.** DOCUMENTED as inherited
   `MonthlyAccumulator` behaviour; the SPATIAL (regridded) CMOR path — the
   primary deliverable — is unaffected. Matters only for variable-resolution
   meshes.

**Test gaps** — ADDRESSED: real `load_cmor_accumulators` resume roundtrip;
Dec→Jan + year-1 bucket test; level-varying winds (`ua850` vs surface) +
vertical `ta` structure; `psl > ps` over 2 km orography and `psl == ps` for
`phis=0`; proper-2m-`tas`-differs-from-lowest (collector + driver); gate
serial/MPI/1-rank test. 17 tests pass (was 10); adjacent suites + constants
ratchet green (3386 passed).

## Facts unchanged from round 1

MPAS state: `T (nCells,nlev)`, `p_s (nCells,)`, `phis (nCells,)`, `u (nEdges,
nlev)` edge-normal, `q_v (nCells,nlev)`, sigma ascending (index -1 = surface).
`reconstruct_cell_velocity -> (u_east, v_north) (nCells,nlev)`. Absolute
`day = START_DAY + elapsed`; `year = int(day//365)`, `doy = day%365+1`.

## UPDATED CODE

<INSERT changes_v2 HERE>

## UPDATED TESTS

<INSERT tests HERE>

## Your task

Re-review focusing on: (a) does the `_tas_2m` refactor keep the cube path
byte-identical and is the MPAS 2 m computation correct (cell winds, sst/sic
reshape, `T_ice` handling, the inner fallback)? (b) is the atomic PHASE1/PHASE2
restructuring actually atomic and behaviour-preserving vs round 1? (c) is the
1-rank MPI permit safe, and the multi-rank warning correct? (d) any NEW shape,
unit, sign, or dtype bug introduced? (e) are the new tests non-vacuous and do
they actually assert the fixed behaviour? If no substantive bugs remain, say so
explicitly.
