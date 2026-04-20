# FV3 Fortran Fidelity Review

Baselined 2026-04-14.  Older per-iteration prose has been
condensed to save tokens; only the newest Ralph-loop iterations
remain in full form below.

> **Metadata convention (iter-174)**: the earlier "updated through
> Ralph iter N" dateline in this title was removed because each
> iteration bumping it triggered a Codex-flagged inconsistency
> (the number would lag by 1-4 iterations as soon as the NEXT
> stop-hook committed anything).  The authoritative current
> iteration count is the commit count on branch plus the HEAD
> commit message's iter-N tag; there is no need to duplicate that
> metadata in the title.


## Architectural unresolved items (carried through all iterations)

1. **W2 v-wind cube-face imprint at C36** — production path
   (`FV3EdgeShallowWaterModel` → `fv3_sw_tendencies`) uses
   Arakawa-Lamb + RK3, NOT the FV3 FB chain.  iter-505 reduced
   amplitude from ~0.56 m/s to ~0.30 m/s by fixing a PPM x-axis
   bug; the residual 4-fold mode-4/8/12 signature at ±30° is
   inherent to A-L + halo-interpolation + `boundary_fix` and is
   NOT eliminable within the production path.  Fix requires
   either (a) stabilize FV3 FB chain at C36 (needs ng=3 halo
   infrastructure — partially scaffolded, see iter-496..598
   below), or (b) accept and lock the imprint magnitude.
   Iter-592 chose (b) as short-term guard.

2. **FB-path C36 instability** — `_c_sw` first-order upwind
   amplifies face-boundary halo divergence; requires ng=3
   MPI DGRID_NE halo infrastructure for `_d2a2c_vect`.  Scalar
   `pad_halo(halo=3)` + `halo_interp_offsets_h3` + padded grid
   angle + half-metrics + vector `pad_halo_vector(halo=3)` on
   non-MPI backend ALL wired (iter-496..598).  Remaining:
   `pad_halo_mpi_4d(halo=3)` + flipping `_d2a2c_vect` and
   `fv3_fb_sw_step` callers from h=2 to h=3.

## Priority resolution history (iter-96..131)

- **Priority 1 (panel-edge corner metrics)**: RESOLVED iter-96/98/99.
  Python `cdgrid.cosa_corner`/`sina_corner`/`rsin2_corner` matches
  Fortran halo-averaged formula with cross-face sign-flip rotation
  at all 24 seams to 1e-10.  iter-91..95 "convention gap" claim
  was wrong (naive halo-copy model).  Locked by
  `test_cosa_corner_panel_edge_fortran_match_all_24_seams` +
  `test_rsin2_corner_matches_fortran_at_interior`.

- **Priority 2 (d_sw3 BGRID_NE component sync)**: RESOLVED iter-102/103.
  `synchronize_bgrid_ne_corner_geo` routes through geographic frame,
  handling all 24 seams + 8 cube vertices via
  `synchronize_corner_scalar`.  Wired into `_bgrid_ke_transport`
  replacing scalar-KE fallback, matching `dyn_core.F90:968-1019`.
  Iter-140 removed dead 12-seam legacy (−233 lines).

- **Priority 3 (non-duogrid `_d2a2c_vect` cube-vertex gap)**:
  GAP ISOLATED AND LOCKED iter-107/108/128..131.  Fortran
  `sw_core.F90:3527-3545,3620-3640` writes 3 halo cells per
  corner-axis; Python's `_fill_corners_h2` can represent only 2
  per axis (architecturally bound, needs halo=3).  Reachability:
  only experimental FB chain + `fv3_csw_tendencies`; default
  production A-L path does NOT call `_d2a2c_vect`.  Runtime
  tripwires (iter-129/130) pin the call graph.  iter-569/570
  extracted exact ndsl / pyFV3 `fill_corners_3cells_mult_x`
  recipe for future h=3 port.

- **Priority 4 (FB-path diagnostic)**: RESOLVED iter-109..111
  with `test_fb_path_component_vs_scalar_sync_propagates_to_wind`
  pinning end-to-end propagation at ±2e-3 of measured diffs.

## Polar-face asymmetry investigation (iter-112..127, closed iter-510)

Pre-iter-505 baseline showed 16% N-S mass-tendency asymmetry
between faces 4 and 5 on W2 alpha=0 balanced state.  Bisection
localized to `cgrid_mass_flux_divergence` PPM upwind at polar-face
halo cells.  **Iter-505 axis fix eliminated this entirely** —
faces 4/5 now identical tendency magnitudes to 1e-4 of 1.0 and
exact N-S reflection symmetry at machine precision.  Locked by
`TestFv3SwTendenciesPolarFaceSymmetry` (iter-510).

## ITER-505 major production-path bug fix (2026-04-19)

**Root cause**: `cgrid_mass_flux_divergence` (`operators_cdgrid.py`)
and `_cgrid_fct_fluxes_2d` (iter-506) passed x-direction strips of
shape `(6, n+4, n)` directly to `_ppm_reconstruct_1d`, which
reconstructs along the LAST axis.  The halo-padded i-axis was
axis 1 but the 8-cell interior j-axis was LAST — so reconstruction
ran along the wrong axis.  **Fix**: `swapaxes(1, 2)` before
reconstruction, swap back after.

**Impact on canonical W2 alpha=0 C36 dt=300s 1 day** (matrix path):

| Quantity                        | Pre-iter-505 | Post-iter-505 | Change |
| ------------------------------- | ------------ | ------------- | ------ |
| Williamson 2 L2                 | 1.53e-3      | **2.42e-4**   | 6.3× ↓ |
| Williamson 2 Linf               | 4.07e-3      | **1.83e-3**   | 2.2× ↓ |
| Williamson 2 max\|v_ll\|        | 0.557 m/s    | **0.303 m/s** | 1.8× ↓ |
| Williamson 5 mass drift         | 1.42e-5      | **1.74e-5**   | within order |
| Cosine bell L1                  | 1.20e-1      | **1.20e-1**   | unchanged |
| Ocean rest state                | machine prec | **machine prec** | invariant |

Cosine bell uses `transport_step` directly (Lin-Rood), not
`cgrid_mass_flux_divergence`.  Iter-508/509 refactored
`_ppm_reconstruct_1d` to take mandatory `axis` kwarg; AST guard
`test_no_future_caller_passes_non_halo_last_axis_to_ppm` enforces
contract.

## Historical archive (token-trimmed)

The full prose log for `iter-1..643` was collapsed to save tokens.
Current live conclusions are preserved above in:

- `Architectural unresolved items`
- `Priority resolution history`
- `Polar-face asymmetry investigation`
- `ITER-505 major production-path bug fix`

Compressed milestones:

- `iter-1..66`: core shallow-water FV3 metric/operator port
  against the Fortran oracle; FB chain landed but remained
  C36-unstable.
- `iter-67`: baseline snapshot with duogrid-branch operator
  formulas verified against oracle conventions.
- `iter-68..131`: remaining seam/sync and non-duogrid gaps
  isolated; architectural blockers documented.
- `iter-132..174`: regression expansion, metadata cleanup, and
  fidelity-note hardening.
- `iter-505..510`: major production-path polar/PPM fix; old
  face-4 vs face-5 asymmetry closed.
- `iter-511..643`: W2/W5 artifact characterization, halo=3
  readiness, exact-formula locks, and test/doc cleanup.

Use git history if you need the full pre-iter-644 narrative.

## Latest Ralph-loop iterations (full form)

### Iter-644 — full Fortran-formula lock for `_divergence_corner_duo`

**Motivation**: `_divergence_corner_duo` in `src/legoesm/core/fv3_sw_core.py` (lines 827-922) computes the corner divergence for the duogrid nord>0 divergence-damping branch and matches `sw_core.F90:2345-2447`.  Prior to iter-644 only the face-boundary zeroing + 0.25 attenuation post-processing was locked (`test_divergence_corner_duo_face_boundary_zeroing`, iter-554).  The interior formula — uf/vf with cross-velocity correction, corner-divergence stencil, edge-index selection — had no direct numerical lock.

Drift in this helper silently distorts divergence damping at cube edges, which is exactly the region most prone to v-wind artefacts on Williamson 2.

**Tests added** (`TestDivergenceCornerDuoFortranFormula`, 2 tests):
- `_ref_divergence_corner_duo`: numpy line-by-line reproduction covering uf (with cross-velocity `-0.25*(va_below + va_above)*(cos_N + cos_S)` correction), vf (with `(ua_left + ua_right)*(cos_E + cos_W)` correction), the corner stencil `(vf[i,j-1] - vf[i,j] + uf[i-1,j] - uf[i,j]) * rarea_c`, the 4-face-boundary zeroing, and the 0.25× attenuation at face-adjacent rows/cols.
- `test_divergence_corner_duo_matches_fortran`: real duogrid CDGrid at n=8, float64 round-off match against the numpy reference at `atol=1e-12`.
- `test_divergence_corner_duo_mutation_suite_iter644`: 3 mutations:
  - M1: `cos_N ↔ cos_S` swap in uf (wrong edge index).  Signal ~2e-8 at C6 — threshold set to 1e-10 because cos_N and cos_S differ by only ~1e-2 at near-axis-aligned cube interior cells.
  - M2: 0.25 attenuation coefficient → 0.5.  Detected at `>1e-6`.
  - M3: sign flip on the vf contribution in the corner stencil.  Detected at `>1e-6`.

All 2 tests pass.  Cumulative Fortran-formula lock inventory: **21 helpers** (iter-641's 20 + `_divergence_corner_duo`).

### Iter-645 — Fortran-formula lock for the 4th-order D→A Lagrange stencil

**Motivation**: The 4th-order D→A averaging stencil in `_d2a2c_vect` (fv3_sw_core.py:464-469, non-duogrid interior override) and `_d2a2c_vect_duogrid` (fv3_sw_core.py:347-355, on the fully-haloed domain) implements FV3's 4th-order Lagrange interpolation from edge values to cell centres at `sw_core.F90:3421-3435`:

    utmp(i, j) = A2 * (u(i, j-1) + u(i, j+2)) + A1 * (u(i, j) + u(i, j+1))

with `A1 = 0.5625` (9/16) and `A2 = -0.0625` (-1/16).  These are canonical 4th-order Lagrange coefficients, cubic-exact on a uniform grid, with partition-of-unity `2*(A1+A2) = 1.0`.  Pre-iter-645 no direct lock on the coefficients or their accuracy order existed; only integration-level coverage via constant-field preservation tests.

A regression that changed `A1` to 0.5 (2nd-order 2-point avg), swapped `A1` ↔ `A2`, or mis-indexed the stencil slices would silently reduce accuracy order from 4 to 2 at interior cells — invisible on constant fields but visible on gradient transport.

**Tests added** (`TestD2A2C4thOrderStencilFortranFormula`, 3 total):
- `_ref_4th_order_1d`: numpy reproduction of the 4th-order Lagrange stencil on a generic axis.
- `test_d2a2c_vect_4th_order_coefficients_iter645`: asserts `_A1 == 9/16`, `_A2 == -1/16`, and the partition-of-unity `2*(A1+A2) == 1.0` at 15 decimal places.
- `test_d2a2c_4th_order_stencil_cubic_exactness_iter645`: feeds a cubic polynomial `q(j) = c0 + c1*j + c2*j² + c3*j³` through the 4th-order stencil and asserts the output matches the cubic evaluated at cell centres (`j + 1.5`) at `atol=1e-12`.  Locks the accuracy-order property directly (a 2nd-order stencil would fail here by large amounts).
- `test_d2a2c_4th_order_stencil_coefficients_load_bearing_iter645`: meta-test — swaps `A1` ↔ `A2` in a hand-rolled reference, applies it to the cubic, and asserts the swapped-coefficient output DIFFERS from the cubic-exact value.  Proves the cubic-exactness test has real detection power for coefficient swaps.

All 3 tests pass.  Cumulative Fortran-formula lock inventory: **22 helpers** (iter-644's 21 + the 4th-order D→A stencil).

### Iter-646 — strengthen iter-645 stencil lock to exercise production JAX code

**Codex stop-time finding on iter-645**: "the new 'Fortran-formula lock' does not actually exercise the production stencil."

**Analysis**: iter-645's three tests locked (a) the `_A1` / `_A2` module constants, (b) the numpy reference's cubic-exactness, and (c) coefficient-swap sensitivity of the numpy reference.  None of these actually called the JAX production code at `fv3_sw_core.py:464-469` (`_d2a2c_vect`) or `:347-355` (`_d2a2c_vect_duogrid`) — a regression that inlined the literal `0.5625` value into the production formula (while leaving the `_A1` constant untouched) would not have been caught.

**Fix**: added two production-consumption tests (`test_d2a2c_vect_production_stencil_consumes_A1_A2_iter646` and `test_d2a2c_vect_duogrid_production_stencil_consumes_A1_A2_iter646`) that:
1. Call the real JAX `_d2a2c_vect` / `_d2a2c_vect_duogrid` on random `(u_d, v_d)` at n=12 (where the 4th-order branch fires).
2. Then patch `fv3_sw_core._A1` to a sentinel value (`0.5`) and re-run.
3. Assert the output at AT LEAST ONE return element changes by `>1e-8`.

A regression that inlined the literal and dropped the symbol reference would produce identical output in patched and unpatched runs — the tests would then fail visibly.

All 5 tests (3 from iter-645 + 2 from iter-646) pass on the correct code.  The production stencils are now demonstrably consuming `_A1` at the sites the lock claims to cover.

### Iter-647 — direct interior-cell stencil lock via orthogonalised CDGrid

**Codex stop-time finding on iter-646**: "iter-646's tests can false-pass without proving the targeted D→A stencil uses `_A1`/`_A2`."

**Analysis**: iter-646's patch-and-detect-diff approach catches regressions that remove `_A1` entirely, but NOT regressions that inline the literal at the u4 stencil site (fv3_sw_core.py:464-469) while leaving other `_A1` usages (e.g., the v4 stencil on the next line) dynamic.  The total-output diff would still fire on v4's dynamic behavior, falsely passing the u4 test.

**Fix**: added `test_d2a2c_vect_interior_stencil_output_matches_inline_formula_iter647` which:
1. Builds a CDGrid and overrides its metrics to orthogonal: `cos_sg[..., 4] = 0` and `rsin2_cell = 1`.
2. Under that override, the cov→contra step in `_d2a2c_vect` (`ua = (utmp - vtmp*cos_sg5)*rsin2`) reduces to `ua = utmp` at interior cells.
3. Feeds a random `u_d` with `v_d = 0` and computes the explicit 4th-order Lagrange formula directly:

        expected_utmp_interior = A2*(u_d[:, :, :-3] + u_d[:, :, 3:])
                               + A1*(u_d[:, :, 1:-2] + u_d[:, :, 2:-1])

4. Asserts `ua[:, 2:-2, npt:n-npt]` (interior cells) matches that expected slice at float64 round-off (iter-649: `atol=1e-14, rtol=0`).

**Verification**: deliberately applying a mutation that inlines the literal `0.7` at the u4 line (leaving `_A1` / `_A2` and v4 untouched) causes the iter-647 test to fail with 100% mismatch and max diff 0.55 — proving the lock now targets the u4 stencil specifically, independent of whether other `_A1` usages remain dynamic.  Source restored after the verification.

This closes the false-pass gap Codex flagged.  The 4th-order D→A stencil in `_d2a2c_vect` is now locked at three strengths: constant values (iter-645), production consumption of `_A1` somewhere (iter-646), and bit-exact interior-cell output under orthogonalised metrics (iter-647).

### Iter-648 — tighten iter-647 tolerance to true float64 round-off

**Codex stop-time finding on iter-647**: "the new 'bit-for-bit' stencil lock still allows `1e-7` relative error."

**Root cause**: iter-647 used `atol=1e-10` alone.  `np.testing.assert_allclose` defaults to `rtol=1e-7`, so the EFFECTIVE tolerance on values of O(1) was `atol + rtol*|desired| ≈ 1e-7` — seven orders of magnitude above float64 round-off and indistinguishable from float32 rounding.  The claim "bit-for-bit" in iter-647's docstring was wrong.

**Fix**: two changes to `test_d2a2c_vect_interior_stencil_output_matches_inline_formula_iter647`:
1. Build the CDGrid with `dtype=jnp.float64` so the metrics don't default to float32 and promote downstream computation to float32 (which was producing the ~1e-7 round-off).  The grid metric override for `cos_sg` / `rsin2_cell` also now explicitly uses `dtype=jnp.float64`.
2. Set `rtol=0.0, atol=1e-13` on `assert_allclose` — float64 round-off match (NOT IEEE-754 bit identity; iter-649 measured 4 ULPs of float64 ε).

Added a note in the `err_msg` documenting the iter-648 tightening.

All 6 tests in `TestD2A2C4thOrderStencilFortranFormula` now pass at the tightened tolerance.  The 4th-order D→A stencil lock matches the explicit inline formula to within float64 round-off (the "bit-for-bit" wording here was corrected in iter-649).

### Iter-649 — stop overclaiming exactness; set tolerance at real float64 round-off floor

**Codex stop-time finding on iter-648**: "iter-648 still overclaims exactness; the new lock is not actually bit-for-bit."

**Root cause**: iter-647 and iter-648's docstrings used the phrase "bit-for-bit".  The actual measured max `|JAX - numpy|` diff on a random n=12 grid is `4.44e-16`, which is 4 ULPs of float64 machine epsilon (`ε = 1.11e-16`).  This arises from reordering of floating-point additions between the two pipelines and does NOT meet IEEE-754 bit identity (which would require `atol=0`).  The overclaiming language is misleading.

**Fix**:
1. Tightened `atol` from `1e-13` to `1e-14` (≈ 100 ULPs — well below any semantic change, well above the observed 4-ULP round-off floor).  `rtol=0.0` stays.
2. Replaced every "bit-for-bit" / "truly bit-for-bit" / "genuinely bit-for-bit" claim with "float64 round-off identity" / "~4 ULPs of float64 ε" / "at the float64 round-off floor" — accurate descriptions of what the test actually enforces.
3. The test docstring now includes a dedicated iter-649 paragraph documenting the measured discrepancy and why atol=1e-14 is the appropriate threshold.

All 6 stencil tests pass at the new tolerance.  The lock detection power is unchanged (the iter-647 mutation verification at 100% mismatch still triggers under the tightened threshold), but the stated strength now matches the enforced strength.

### Iter-650 — scrub remaining "bit-for-bit" overclaims in review doc

**Codex stop-time finding on iter-649**: "iter-649 still leaves contradictory 'bit-for-bit' claims in the review doc."

**Root cause**: iter-649 fixed the iter-647/648 entries but left "bit-for-bit" in 9+ other entries dating back to iter-618.  Most paired the phrase with a nonzero `atol` or `rtol < 1e-10` — the same contradiction pattern iter-649 identified.  Any reader skimming the doc would still see "bit-for-bit at atol=1e-12" and draw the wrong conclusion about what the tests actually enforce.

**Fix (iter-650, review-doc scope ONLY)**: scrubbed every "bit-for-bit at atol=X" claim in the REVIEW DOC (9 occurrences across iter-618, iter-621, iter-622, iter-623, iter-629, iter-630, iter-634, iter-636, iter-637, iter-638, iter-639, iter-641, iter-642, iter-644, iter-647).  Standard replacements:
- "bit-for-bit at `atol=X`" → "float64 round-off match at `atol=X` (not IEEE bit identity)"
- "bit-for-bit equal" → "matches ... to within float64 round-off"
- "compared bit-for-bit (rel < 1e-10)" → "compared at rtol < 1e-10 (round-off match, not IEEE bit identity)"

**Iter-650 was incomplete (Codex finding on iter-650)**: the "scrubbed every remaining" phrasing in the original iter-650 entry was wrong — it only covered the review doc, not the 18+ "bit-for-bit" occurrences that remained in test docstrings under `tests/unit/test_cdgrid_fv3_regression.py`.  That gap was closed in iter-651 below.

### Iter-651 — scrub remaining "bit-for-bit" claims from test docstrings

**Codex stop-time finding on iter-650**: "iter-650's cleanup is incomplete; its new 'scrubbed every remaining …' claim is false."

**Root cause**: iter-650 operated on `docs/fv3_fortran_fidelity_review.md` only.  `tests/unit/test_cdgrid_fv3_regression.py` still had 18 docstring/comment claims of "bit-for-bit" paired with nonzero tolerances — the same contradiction pattern iter-649 identified, now in test docstrings rather than the review doc.

**Fix**: systematic replacement of "bit-for-bit" → "at float64 round-off" (or "float64 round-off match") across 18 test docstrings, then a follow-up pass to restore the intentionally quoted occurrences inside the iter-649 correction paragraph (lines 7575-7584, 7638) which explicitly document the historical overclaiming phrase.  Three docstrings with damaged grammar (post-blanket-replace) were also rewritten to read cleanly.

**Verification**: full `test_cdgrid_fv3_regression.py` suite (118 tests) runs green after the edits.  A final repo-wide `grep "bit-for-bit"` shows occurrences only in: (a) the iter-647/648/649/650 correction narratives (intentional quotes), (b) the iter-649 correction paragraph inside `test_d2a2c_vect_interior_stencil_output_matches_inline_formula_iter647`'s docstring (intentional quote), and (c) one pre-existing legitimate use at `docs/fv3_fortran_fidelity_review.md:916` where iter-650 already annotated the qualification.

No test logic changes; documentation-only completion of the iter-650 scrub.

### Iter-652 — correct iter-651's false-precision claims for looser tolerances

**Codex stop-time finding on iter-651**: "iter-651's blanket wording change still makes false claims about what the tests enforce."

**Root cause**: iter-651 did a blanket `bit-for-bit → at float64 round-off` replacement across 18 test docstrings, but six of the resulting claims are wrong about the actual enforced tolerance:

- `test_duogrid_random_inputs_match_numpy_reference` (iter-622): enforces `diff / rms < 1e-10` — that's `rtol < 1e-10` relative, not float64 round-off (~1e-15).
- `test_random_input_matches_numpy_reference` (iter-623): same, `rel < 1e-10`.
- `test_full_field_matches_numpy_reference` (iter-624): same, `rel < 1e-10`.
- `test_non_uniform_dxa_matches_explicit_fortran_formula` (iter-617): uses `assertAlmostEqual(places=12)` → 12 decimal places (1e-12 absolute), close to round-off but better described by the places count.
- The `TestEdgeInterpolate4FortranFormula` class docstring: same `places=12` context.
- One error message in the iter-554 `test_divergence_corner_duo_face_boundary_zeroing` which uses `assertEqual(interior_diff, 0.0)` after float32 cast — that's IEEE-identity-after-cast, not "float64 round-off".

**Fix** (targeted per-test, NOT blanket):
- 3 `rel < 1e-10` tests: rewrote to "at `rel < 1e-10` relative tolerance (not IEEE bit identity)".
- 2 `places=12` tests: rewrote to "to 12 decimal places".
- 1 `assertEqual(..., 0.0)` error message: rewrote to "match production exactly (both sides cast to float32 before subtraction, assertEqual to 0.0 verifies IEEE-identity after cast)".

Remaining 12 test docstrings that say "at float64 round-off" all pair with actual `atol ∈ [1e-14, 1e-12]` — close enough to float64 round-off on O(1) values that the description is accurate (the iter-649 measurement showed 4 ULPs ≈ 4e-16 is the natural floor for the stencil in question; `atol=1e-12` is ~10^4 ULPs but well within the reordering-round-off tolerance for linear combinations with more operands).

All 118 tests still pass.  The test-docstring tolerance claims now match the actual enforced tolerances at each call site.

### Iter-653 — replace remaining "float64 round-off" claims with actual atol

**Codex stop-time finding on iter-652**: "iter-652 still makes a false 'remaining docstrings are accurate' precision claim."

**Root cause**: iter-652 defended the remaining 10+ "at float64 round-off" docstrings by arguing `atol ∈ [1e-14, 1e-12]` is "close enough to round-off".  For `atol=1e-14` on O(1) values this is reasonable (~100 ULPs above float64 ε ≈ 1e-16), but for `atol=1e-12` or `atol=1e-13` the claim overstates the strictness — those are 10^3 to 10^4 ULPs above round-off, and calling them "float64 round-off" is misleading.

**Fix**: replaced every remaining "at float64 round-off" / "float64 round-off match" docstring claim with the literal `atol` value used in that test:

- `test_pert_ppm_iv1_matches_fortran_on_branch_probes` (iter-635): → "at `atol=1e-14`"
- `test_pert_ppm_iv1_matches_fortran_on_random_grid` (iter-635): → "at `atol=1e-14`"
- `test_ke_upwind_matches_fortran_non_duogrid` (iter-636): → "`atol=1e-13` match"
- `test_vorticity_flux_matches_fortran_non_duogrid` (iter-637): → "`atol=1e-12` match"
- `test_corner_vorticity_matches_fortran_non_duogrid` (iter-638): → "`atol=1e-12` match"
- `test_xppm_flux_formula_matches_fortran` (iter-639): → "at `atol=1e-14`"
- `test_sina_u_v_from_sin_sg_matches_fortran` (iter-641): → "`atol=1e-14` match"
- `test_fill_corners_h1_matches_reference` (iter-642): → "`atol=1e-14` match"
- `test_fill_corners_h2_matches_reference` (iter-642): → "`atol=1e-14` match"
- `test_divergence_corner_duo_matches_fortran` (iter-644): → "`atol=1e-12` match"

Also changed the comment at line 2334 to describe the actual `~1e-13 round-trip noise` accurately instead of "float64 round-off".

All 118 tests still pass.  The test-docstring tolerance claims now cite the literal `atol` value enforced at each call site — no more subjective "close enough to round-off" language.

### Iter-654 — FB-chain halo=3 wiring, part 1: `_d2a2c_vect_duogrid`

**Motivation**: user directive to implement (a) FB-chain halo=3 wiring to attack the W2 v-wind artifacts (polar-cap + mid-latitude cube-face imprint).  Review doc item #2: the C36-instability blocker requires ng=3 halo throughout the FB chain.  Smallest first piece: generalise `_d2a2c_vect_duogrid` so its halo depth can be flipped from 2 to 3 when the duogrid structure supports it.

**Change** (`src/legoesm/core/fv3_sw_core.py:283-385`):
- Replaced the literal `h = 2` at line 284 with `h = 3 if (dg is not None and dg.ng >= 3) else 2`.
- Rewrote the 4th-order D→A stencil slices at lines 347-355 from hard-coded `[0:n, 1:n+1, 2:n+2, 3:n+3]` to `h`-parameterised `[h-2:h-2+n, h-1:h-1+n, h:h+n, h+1:h+1+n]` (identical at h=2, correct at h=3).
- Rewrote the 4th-order A→C stencil slices at lines 375-381 from `[:-3, 1:-2, 2:-1, 3:]` to `h`-parameterised `[h-2:h-1+n, h-1:h+n, h:h+n+1, h+1:h+2+n]`.
- Tiny-grid fallback at lines 358-359 also generalised.

**Tests**: all 118 regression tests pass.  Small-n tests exercise h=2 (backward-compatible); n≥6 tests exercise h=3.

**W2 result**: the SW test matrix at C36 is UNCHANGED (L2=2.42e-04, Linf=1.83e-03).  Reason: the production W2 path runs `FV3EdgeShallowWaterModel.tendencies` → `fv3_sw_tendencies` (A-L + RK3), NOT the FB chain.  `_d2a2c_vect_duogrid` is consumed only by `_c_sw` / `_d_sw_native` / `fv3_fb_sw_step`.

**FB-chain stability**: ran `scripts/diag_williamson2_fb.py` at C36 — FB chain still **BLOWS UP at step 26** (4.3h in).  Iter-654's change to `_d2a2c_vect_duogrid` alone is NOT sufficient to stabilise the FB chain at C36.  Further h=3 wiring is needed at other sites:
- `_c_sw` (fv3_sw_core.py:1198): uses `fv3_d2cc` / `fv3_cc2c` + `cgrid_mass_flux_divergence` which all run at halo=2.
- `_d_sw_native` (fv3_sw_core.py:1742): complex orchestration with many halo=2 call sites.
- `fv3_fb_sw_step` (fv3_sw_core.py:1874): top-level orchestrator.

**Next iter target**: identify which specific halo=2 site in `_c_sw` amplifies the cube-edge error most, and upgrade that next.  Likely candidates: `cgrid_mass_flux_divergence`, the sin_sg halo exchange for upwind selection, or `_d_sw5_corner_divergence`.

**Pending**: visual inspection of W2 v-wind after the FB-chain is stable.  Today's iter-654 is infrastructure only — no visible W2 change, no regression.

### Iter-655 — FB-chain halo=3 wiring, part 2: replace `mode='edge'` with `pad_halo` in `_d_sw5_corner_divergence` nord=0 branch

**Motivation**: continued FB-chain halo=3 wiring per user directive.  Investigated `_c_sw` and `_d_sw_native` for halo-depth gaps.  Found that `_d_sw5_corner_divergence` (called by `_d_sw_native`) uses `jnp.pad(ua, ..., mode='edge')` for ua/va halo padding at fv3_sw_core.py:993-994.  Fortran FV3 fills ua/va halos via `mpp_update_domains(DGRID_NE)` before `d_sw5` fires — `mode='edge'` is a same-face 1D extension, NOT the Fortran cross-face halo exchange.  This is a documented fidelity gap at the cube-edge region most prone to v-wind amplification.

**Change** (`src/legoesm/core/fv3_sw_core.py:993-1015`): replaced `jnp.pad(..., mode='edge')` with `pad_halo(halo=1, duogrid=dg)` for the nord=0 (del-2) branch:

```python
ua_full = pad_halo(ua, halo=1, interp_offsets=_offs, duogrid=dg)
va_full = pad_halo(va, halo=1, interp_offsets=_offs, duogrid=dg)
ua_pad = ua_full[:, :, 1:-1]  # preserves downstream (6, n+2, n) shape
va_pad = va_full[:, 1:-1, :]  # preserves downstream (6, n, n+2) shape
```

The full 2D halo=1 pad delivers cross-face neighbour data.  Slicing to 1D preserves the downstream indexing unchanged.

**Tests**: all 118 regression tests pass.  The iter-644 `TestDivergenceCornerDuoFortranFormula` is NOT affected because it tests a different function (`_divergence_corner_duo`).

**FB-chain stability**: UNCHANGED.  `scripts/diag_williamson2_fb.py` still blows up at step 26 at C36.  Reason: the default FB-chain config uses `nord=1` (del-4 damping) which routes through `_divergence_corner_duo` (fv3_sw_core.py:1032), not through the nord=0 branch I just fixed.  `_divergence_corner_duo` has the SAME `mode='edge'` gap at lines 877-878 but fixing it requires updating the iter-644 `test_divergence_corner_duo_matches_fortran` numpy reference too (the iter-644 lock currently reproduces the Python `mode='edge'` behaviour, not the Fortran `mpp_update_domains` behaviour — iter-644's "Fortran-formula lock" classification was partially wrong for the halo treatment).

**Flagged as unresolved**: `_divergence_corner_duo` still uses `mode='edge'` at fv3_sw_core.py:877-878 (nord≥1 path).  A full fix requires:
1. Replace `jnp.pad(..., mode='edge')` with `pad_halo(halo=1)` in `_divergence_corner_duo`.
2. Update iter-644's `_ref_divergence_corner_duo` numpy reference to use the same `pad_halo` pattern.
3. Re-run FB diag to check C36 stability.

Also `_d_sw5_corner_divergence` at lines 1014-1015 (the `vort_pad` / `ptc_pad` inside the nord=0 branch) still uses `mode='edge'` for D-grid face-midpoint fields — those need an edge-midpoint halo helper, which doesn't yet exist.

### Iter-656 — close the `nord=1` `_divergence_corner_duo` gap + Codex iter-655 scope fix

**Codex stop-time finding on iter-655**: "nord=0 fix leaks into the default nord=1 path."

**Root cause**: iter-655 placed the new `pad_halo` call BEFORE the `if nord == 0:` branch in `_d_sw5_corner_divergence`, so it fired unconditionally — wasted work plus a silent halo dependency that didn't previously exist on the nord>=1 default path.

**Fix A** (scope correction): moved the `pad_halo(ua)` / `pad_halo(va)` + `ua_pad` / `va_pad` slicing INSIDE the `if nord == 0:` branch.  The pad now fires only when the nord=0 del-2 branch is active.  For nord>=1 (default FB-chain config), the pad is never computed.

**Fix B** (close the nord=1 default-path gap): applied the same `mode='edge' → pad_halo` replacement inside `_divergence_corner_duo` at fv3_sw_core.py:877-879.  This is the helper called by the nord>=1 branch of `_d_sw5_corner_divergence` (the default FB-chain damping path).

**iter-644 reference update**: `_ref_divergence_corner_duo` numpy reproduction in `tests/unit/test_cdgrid_fv3_regression.py` now accepts an optional `cdgrid=` argument.  When passed, it uses `pad_halo(halo=1, duogrid=dg)` matching the production code.  When `cdgrid=None` (iter-644 mutation-suite tests that swap sin_sg tables), it falls back to the legacy `mode='edge'` — those tests aren't halo-sensitive and the sin_sg swap exercises only the coefficient-selection branches.

**Tests**: all 118 regression tests pass, including both iter-644 tests (`test_divergence_corner_duo_matches_fortran` and `test_divergence_corner_duo_mutation_suite_iter644`).

**FB-chain stability**: still blows up at step 26 at C36.  The halo-fidelity gap was real (iter-655/656 closes it), but the C36 instability is NOT bottlenecked on halo padding.  Other plausible sources:
- Divergence damping coefficient mismatch vs Fortran default (`d4_bg=0.16` may be too low for C36 stability given the FB chain coupling).
- Phase ordering between c_sw → p_grad_c → d_sw may differ in a subtle way from FV3 dyn_core.F90.
- `_d_sw1_recompute_ut_vt` adjacent-strip recomputation may have a cube-vertex gap not captured by existing tests.

Next iter: profile the step-26 blowup with more diagnostic output (max|v_d| per step, max|divg| per step) to localise the amplifier.

### Iter-657 — revert iter-656 Fix B (hot-path no-op) per Codex finding

**Codex stop-time finding on iter-656**: "Fix B is a hot-path no-op with no demonstrated behavioral gain."

**Verification**: measured `|divg_d_pad_halo - divg_d_mode=edge|` on random C36 input and confirmed `max diff = 0.0` bit-for-bit.  The upstream halo choice DOES produce different `uf` / `vf` at j=0 and j=n cells (measured ~1e6 absolute diff on random O(10) input), but those cells' contributions to `divg_d` are ZEROED at the face-boundary zeroing step (`divg_d.at[:, 0, :].set(0.0)` etc.).  The zeroing cancels the halo difference entirely.

Codex is right — Fix B was a numerical no-op in `_divergence_corner_duo`, despite appearing to improve Fortran fidelity at the code-surface level.

**Revert**:
1. Reverted `_divergence_corner_duo` at fv3_sw_core.py:877-889 back to `jnp.pad(mode='edge')` (original pre-iter-656 behaviour).
2. Reverted iter-644's `_ref_divergence_corner_duo` numpy reference to drop the `cdgrid=` kwarg and go back to unconditional `np.pad(mode='edge')`.

**Iter-655/656 Fix A (scope correction) is KEPT**: the `_d_sw5_corner_divergence` nord=0 branch still uses `pad_halo` inside the `if nord == 0:` block.  That change is also a numerical no-op (same face-boundary-zeroing logic applies) — so it too could be reverted.  But the nord=0 branch is not hot under default config (nord=1), so the wasted work is negligible, and the pad_halo call documents the Fortran-oracle intent for future readers.  Leaving it for now.

**Lesson for future halo changes**: before claiming a halo-source change improves Fortran fidelity, A/B test the end-to-end numerical output on realistic input.  Many FV3 helpers perform face-boundary zeroing or attenuation that cancels upstream halo differences.  Apparent code-level fidelity gains can be numerical no-ops.

**Regression tests**: all 118 tests pass after the revert.  W2 SW matrix metrics unchanged (still L2=2.42e-04, Linf=1.83e-03).  FB chain still blows up at step 26 — the C36 instability is NOT bottlenecked on halo padding in `_divergence_corner_duo`.

**FB-chain stability next-step candidates** (unchanged from iter-656):
- Divergence damping coefficients (`d4_bg=0.16` may be too low for C36 coupling).
- Phase ordering between `c_sw` / `p_grad_c` / `d_sw`.
- `_d_sw1_recompute_ut_vt` adjacent-strip cube-vertex gap.

### Iter-658 — FB-chain blowup diagnostic: localize to cube vertices

**Motivation**: iter-657 reverted halo changes after A/B showed they were no-ops.  To make further progress on the FB-chain C36 instability, need to pinpoint WHERE the error grows rather than guess.  Added a per-step diagnostic that traces `max|h|`, `max|u_d|`, `max|v_d|`, and `max|v_err|` with the `(face, i, j)` location of the max error.

**Script added**: `scripts/diag_fb_blowup_trace.py`.  Runs Williamson 2 at C36 for 30 steps through `FV3FBShallowWaterModel` and logs per-step diagnostics.

**Finding**: the blowup is concentrated at **cube vertices** (3-face meeting points), not face edges.  Observed trajectory:

| step | v_err (m/s) | location       | regime |
|-----:|------------:|:---------------|:-------|
|    1 |       3.14  | face 0, i=36, j=0  (SE cube vertex) | linear bias |
|    4 |      12.4   | face 3, i=0, j=0  (cube vertex)     | linear growth |
|   14 |      67     | face 1, i=0, j=0  (cube vertex)     | transition to exp |
|   24 |     898     | face 1, i=1, j=3                    | exponential  |
|   26 |     1e14    | face 0, i=0, j=0                    | catastrophic |
|   27 |     NaN     |                                     | crashed |

**Diagnosis**: the linear-growth phase (steps 1-14) indicates a **bias error** being integrated step-by-step — most likely an incorrect value at cube vertices fed into the momentum equation each step.  The bias originates at cube vertices (3-face meeting points), which is documented in review doc **Priority 3** (`_d2a2c_vect` non-duogrid cube-vertex gap — Python's `_fill_corners_h2` represents 2 halo cells per corner-axis vs Fortran's 3).

Under C36+duogrid the "Priority 3 GAP" is supposed to be RESOLVED via the Duo-Grid kinked-to-extended Lagrange remap — but `_d_sw1_recompute_ut_vt` still uses `jnp.pad(mode='edge')` at its line 75-83 for the vc/uc 4-cell average.  The `mode='edge'` at cube vertices gives the SAME-FACE boundary value, while Fortran/duogrid would give the neighbor-face cross-axis value.  At cube vertices (face boundaries of BOTH axes meeting), the same-face extension can be drastically wrong.

**Next iter target**: A/B test whether replacing the `jnp.pad(vc, [(0,0),(1,1),(0,0)], mode='edge')` at `_d_sw1_recompute_ut_vt:75` with a proper cross-face halo (halo.pad_halo_vector or an edge-midpoint halo helper) reduces the step-1 `v_err` from 3.14 to a value more consistent with machine noise for a well-balanced IC.

**Tests**: all 118 regression tests still pass.  The diagnostic script is non-invasive (script only, no source changes).

### Iter-659 — correct iter-658 cube-vertex localization overclaim

**Codex stop-time finding on iter-658**: "tracer does not actually localize the blowup to cube vertices."

**Reality check**: built an error-distribution histogram at step 1.  The cell counts and magnitudes do NOT support "localized to cube vertices":

| region                                       | count | max v_err (m/s) | mean v_err |
|:---------------------------------------------|------:|----------------:|-----------:|
| cube VERTICES (i∈{0, n} AND j∈{0, n−1})      |    24 |            3.16 |       2.40 |
| cube EDGES (XOR of i-boundary and j-boundary)|   828 |            3.00 |       1.45 |
| INTERIOR (rest)                              |  7140 |            2.15 |       1.04 |
| cells with v_err > 1.0 m/s                   |  4432 |               — |          — |

The error is **field-wide** — 55% of cells have v_err > 1.0 m/s, not a few cube-vertex cells.  Vertices are 1.5-2× the interior magnitude but the total error by cell count is dominated by interior cells.

**What iter-658 got wrong**: tracking only the MAX location gives a misleading "localization" impression.  The max happens to land at a cube vertex because vertices are 1.5× noisier, but the SIGNAL is everywhere.

**What the histogram actually supports**: the step-1 `v_err` distribution is consistent with a widespread error signal, not a few outlier corner cells.  That is all the histogram data alone can tell us — the histogram does NOT identify a cause.

**Upper bound on propagation in one step** (separate argument, not from the histogram): at C36 with dt=600s, wind speeds ~38 m/s advect by ~23 km per step; gravity-wave phase speed √(gh) ≈ 172 m/s propagates ~100 km per step.  Cell size dx ≈ 600 km.  So information travels at most ~20% of a grid cell in one step.  A bug concentrated at cube vertices CANNOT reach the grid interior in a single step via normal transport/gravity-wave dynamics — which argues against vertex-local bugs as the cause of the field-wide step-1 signature, but does not prove any specific alternative.

**Candidate causes (hypotheses, not conclusions)**:
1. Momentum balance residual in the FB chain's forward-backward coupling.
2. IC inconsistency at D-grid edge-midpoint positions (analytic W2 formula vs. d2a2c_vect + d_sw1 pipeline expectation).
3. Temporal-scheme residual at this time step.

Distinguishing these requires additional diagnostics (e.g., step-1 tendency magnitudes, balanced-IC residual norms) — not available from the current tracer.

**Iter-660 (Codex correction on iter-659)**: tightened the language above.  Iter-659 wrote "plausible causes (all field-wide, not vertex-local)" which suggested the histogram proved "not vertex-local".  The histogram alone does not prove that — only the separate propagation-speed argument does, and even that is a bound, not a proof.

**Next iter target** (unchanged): quantify step-1 tendency magnitudes to separate the hypotheses.  Specifically:
- If `max|dh/dt|`, `max|du_d/dt|`, `max|dv_d/dt|` at t=0 are O(1) instead of O(ε), hypothesis 1 or 2 is active.
- If reducing dt by 10× leaves the relative step-1 `v_err` unchanged (scaled by dt), the spatial scheme is at fault, not the temporal integrator.

**Tests**: no source-code changes in iter-658/659/660; all 118 regression tests still pass.  These three iters are diagnostic + documentation only.

### Iter-661 — FB-chain step-1 tendency diagnostic: dt-independent spatial residual

**Motivation**: iter-660 proposed measuring step-1 tendency magnitudes and dt-scaling to distinguish hypotheses (1) momentum balance residual, (2) IC inconsistency, (3) temporal-scheme residual.

**Experiment** (`scripts/diag_fb_tendency.py`): ran one FB-chain step on the balanced Williamson 2 IC at C36 for four time steps: `dt ∈ {1, 10, 100, 600}` s, measuring `max|dh|/dt`, `max|du|/dt`, `max|dv|/dt`.

**Results**:

| dt (s) | max\|dh\|/dt (m/s) | max\|du\|/dt (m/s²) | max\|dv\|/dt (m/s²) |
|-------:|-------------------:|--------------------:|--------------------:|
|      1 |           5.47e-2  |           4.86e-3   |           6.08e-3   |
|     10 |           5.48e-2  |           2.96e-3   |           5.03e-3   |
|    100 |           5.65e-2  |           2.96e-3   |           5.07e-3   |
|    600 |           6.32e-2  |           2.96e-3   |           5.23e-3   |

**Analysis**:

1. **Tendency is dt-independent** (for dt ≥ 10 s).  If the residual were a temporal-scheme artifact (leapfrog/forward-backward coupling error), it would scale with dt.  The dt-independence means the residual is **spatial** — it comes from the FB chain's spatial operators producing nonzero RHS on a state that should have exactly zero RHS.

   → **Hypothesis #3 (temporal-scheme residual) is ruled out.**

2. **The residual exactly explains the step-1 v_err**: `600 s × 5.23e-3 m/s² = 3.14 m/s` — which matches the observed step-1 `v_err = 3.14 m/s` from iter-658's trace to three significant figures.  So the entire step-1 error signal comes from this t=0 tendency.

3. The small drift at dt=1 (`max|du|/dt=4.86e-3` vs `max|du|/dt=2.96e-3` at dt≥10) is likely a purely temporal effect of measuring a single step at a very small dt; the dt≥10 results are the converged spatial residual.

**Remaining hypotheses**: the FB-chain spatial operators produce a dt-independent residual of ~5 mm/s² on the balanced W2 IC.  This is either:
1. **Momentum balance residual**: the c_sw / p_grad_c / d_sw spatial operators do not exactly produce zero RHS on a geostrophically balanced state (scheme-level fidelity gap).
2. **IC inconsistency**: the analytic `u_d`, `v_d` values at D-grid edge midpoints are not exactly what the d2a2c_vect / d_sw1 pipeline treats as "balanced" (IC is off by the same amount as the scheme residual at t=0).

Distinguishing (1) vs (2) requires a spatially-resolved residual test: feed the balanced IC through just `c_sw` (phase 1 alone) and inspect where in the cubed-sphere the residual concentrates.  If it concentrates at face boundaries, the spatial operator has a seam bug.  If it's spread through the interior, the d2a2c_vect 4th-order stencil likely has a subtle imbalance.

**Script added**: `scripts/diag_fb_tendency.py`.  Reproducible for future iters.

**Next iter target**: spatial breakdown of step-1 tendency — which operator (c_sw vs p_grad_c vs d_sw) produces the 5 mm/s² residual, and where does it live on the grid.

**Tests**: no source-code changes.  118 regression tests pass.  Diagnostic only.
