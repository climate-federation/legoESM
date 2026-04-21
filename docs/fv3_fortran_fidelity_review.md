# FV3 Fortran Fidelity Review

Baselined 2026-04-14.  Older prose is aggressively condensed to
save tokens.  Only the newest Ralph-loop iterations remain in
full form below.

> **Metadata convention (iter-174)**: do not duplicate "updated
> through iter-N" in the title.  The authoritative iteration count
> is the branch commit history plus the HEAD commit message tag.


## Live status

- **W2 v-wind artifact at C36 remains unresolved.**
  Production still runs `FV3EdgeShallowWaterModel` →
  `fv3_sw_tendencies` (Arakawa-Lamb + RK3 + `boundary_fix`), not
  the FV3 FB chain.  Current canonical W2 baseline after iter-505:
  `L2=2.42e-04`, `Linf=1.83e-03`, `max|v_ll|≈3.03e-01 m/s`.
- **FB-path C24/C36 stability remains unresolved.**
  Halo=3 scaffolding is partly in place, but the FB chain still
  needs full h=3 caller rollout and the remaining stability work.

## Closed priorities

- **Panel-edge corner metrics**: resolved.  `cosa_corner` /
  `sina_corner` / `rsin2_corner` now match the Fortran seam
  construction at all 24 seams.
- **d_sw3 BGRID_NE sync**: resolved.  Python now routes the corner
  sync through the geographic-frame seam handler instead of the old
  scalar KE fallback.
- **Non-duogrid `_d2a2c_vect` cube-vertex gap**: isolated, not
  fixed.  It is architectural and requires halo=3, but it does not
  affect the default A-L production path.
- **Old W2 polar face-4 vs face-5 asymmetry**: resolved by the
  iter-505 PPM-axis fix.

## Key production fix

Iter-505 fixed the major production-path bug in
`cgrid_mass_flux_divergence`: x-direction strips were being passed
to `_ppm_reconstruct_1d` with the wrong active axis.  Impact on
canonical W2 C36 dt=300s 1d:

- `L2`: `1.53e-3 -> 2.42e-4`
- `Linf`: `4.07e-3 -> 1.83e-3`
- `max|v_ll|`: `0.557 -> 0.303 m/s`

## Historical archive

Everything before `iter-742` is intentionally compressed here.

- `iter-1..174`: core FV3 metric/operator port, seam/sync work,
  regression expansion, FB-chain brought up but still unstable.
- `iter-505..510`: production-path PPM-axis fix; old polar
  asymmetry closed.
- `iter-511..729`: W2/W5 artifact characterization, formula locks,
  halo=3 plumbing, FB diagnostics, structural test hardening, and
  production-vs-FV3 routing clarified.
- `iter-730..741`: FB C24 stability sweep plus production W2
  artifact measurement pipeline tightened from raw `v_d` to
  post-regrid `v_ll`.

Use git history if you need the full older narrative.

## Latest Ralph-loop iterations (full form)

### Iter-742 — v_north is still a proxy; measure the regridded v_ll (Codex stop-time)

Codex stop-time review on iter-741 flagged: **"iter-741 still measures a proxy instead of the plotted snapshots_v.png field."**  Correct.  The snapshot plots POST-REGRID `v_ll = _regrid_2d(v_north_face, lon_deg, lat_deg, coord_kind="cube")` — the bilinear cube→latlon interpolation.  Iter-741's pre-regrid `v_north_face` differs because the regrid can shift/smooth the mode-4 peak at cube corners.

**Fix (iter-742).**  Apply `_regrid_2d` to v_north_face before taking max.  Notes key renamed `v_north_Linf` → `v_ll_Linf` with pre-regrid value retained in parentheses for diagnostic contrast.

**Baseline values:**
- v_ll_Linf (post-regrid, what PNG shows)  = **3.03e-01 m/s**
- v_north_Linf (pre-regrid)                =   3.07e-01 m/s
- Delta 1.3 % — bilinear regrid minor smoothing.

Both match the iter-717/739 PNG colourbar of ±0.3 m/s; iter-742's 3.03e-01 is the precise match.

**Process takeaway (iter-740 → 741 → 742).**  Three iters in a row, each corrected by Codex stop-time review, to get the sentinel right.  Each step was a real improvement.  Going forward: NEW metrics must be validated against the plotted artifact BEFORE committing — follow the pipeline (inspect plot callsite → replicate exact computation → compare PNG colourbar).

### Iter-743 — duogrid=ON on A-L+RK3 production BLOWS UP (rules out trivial path)

Per iter-722 user directive "implement the exact FV3 duogrid."  The test matrix currently calls `create_cubed_sphere(n)` with default `use_duogrid=False`.  Iter-743 tests the obvious question: does flipping production A-L+RK3 to `use_duogrid=True` reduce the W2 artifact?

**Diagnostic.**  `scripts/diag_iter743_production_duogrid.py` runs W2 C36 1 day on the exact matrix config (`hyperdiff_coeff=_hyperdiff_cube(36)`, `div_damp=_div_damp_cube(36)`, `boundary_fix=True`, `dt=300s`) with `use_duogrid=False` vs `use_duogrid=True` and reports v_ll_Linf + h L2/Linf.

**Results (C36, W2, 1 day, production settings):**

| Config          | h_L2      | h_Linf    | v_ll_Linf    | v_north_Linf |
|-----------------|-----------|-----------|--------------|--------------|
| duogrid=OFF     | 2.42e-04  | 1.83e-03  | 3.03e-01     | 3.07e-01     |
| duogrid=ON      | 1.45e-01  | 8.86e-01  | **3.32e+02** | 3.37e+02     |
| delta           | +598×     | +484×     | **+1097× (blowup)** |       |

**Interpretation.**  A-L+RK3 production with duogrid enabled **blows up catastrophically** — state winds reach hurricane scale and height is order-unity relative error.  The run doesn't NaN but the result is garbage.

**Why duogrid blows up A-L+RK3.**  The A-L + corner-wind + `boundary_fix` + halo-interpolation production path is non-FV3 (architectural item #1 above).  Flipping to duogrid invalidates its tuning:
- `cube_rmp` kinked-to-extended remap replaces plain halo interpolation with Duogrid-specific extrapolation that mismatches A-L corner winds.
- `boundary_fix` averaging is tuned to compensate for SPECIFIC halo-interp errors; duogrid produces DIFFERENT halo errors that `boundary_fix` doesn't handle.
- `cos_angle_padded` / `sin_angle_padded` are Duogrid-computed when duogrid is active; A-L's Bernoulli gradient stencil assumes standard panel-angle continuation.

These amount to a completely different halo regime that the A-L + boundary_fix stabilizer was never designed for.

**What this rules out.**  The user directive "implement the exact FV3 duogrid" CANNOT be interpreted as "set `use_duogrid=True` in the matrix."  That's a catastrophic-blowup path.  The directive requires the **FV3 FB chain** (Fortran-faithful AND designed around duogrid).  FB remains unstable at C24/C36 per iter-730/731.

**What this confirms.**  Production W2 improvement requires one of:
- (a) Fix the FB chain C24/C36 stability (iter-732..738 investigation).
- (b) Fix the A-L+RK3 production path's non-duogrid halo handling directly (iter-732 NW-vertex localisation applies here via `pad_halo_vector` at `operators_cdgrid.py:1397`).
- (c) Introduce a different Fortran-faithful production alternative.

The "trivial" path (d) is **ruled out by direct measurement**.

**v_Linf baseline stability.**  Pre-iter-743, W2 v_ll_Linf has been stable at 0.303 m/s since the iter-505 major production fix.  Iter-743 confirms baseline = 0.303 m/s — unchanged across hundreds of iterations.  Any iter claiming to have reduced the artifact must beat 0.303 without introducing a blowup.

**Deliverable.**  `scripts/diag_iter743_production_duogrid.py` checked in as evidence.  Future iters considering duogrid flips on production see this result in the review doc and know not to repeat the experiment without first fixing the A-L path's duogrid compatibility.

### Iter-744 — localise the W2 v-wind artifact peaks: near-pole (±86°), NOT cube-corner

Iter-743 confirmed the production A-L+RK3 baseline v_ll_Linf = 0.303 m/s.  Iter-744 localises WHERE on the lat-lon grid the peak sits to differentiate cube-corner vs polar vs mid-panel-edge bugs.

**Diagnostic.**  `scripts/diag_iter744_w2_artifact_localise.py` runs the same W2 C36 1 day production case, regrids v_north to lat-lon (181 × 360 mesh), and reports the top 12 |v_ll| peak locations.

**Result.**  All top-12 peaks cluster at:
- **latitude ±86°** (very close to the pole; NOT the cube-corner lat ±35.3° = arctan(1/√2))
- **longitude 70° and -110°** (i.e., mod 180°, bucketed to the ±90° longitude bins)
- magnitude 0.303 m/s (top) down to 0.300 m/s (rank 12) — all within 1 % of the Linf
- sign symmetry: N-hemisphere peaks sign-opposite to S-hemisphere at the same longitude

**Interpretation.**  The artifact is a **polar-face issue**, NOT a cube-corner halo issue:
- Cube corners sit at ±35.3° latitude — no top peaks there.
- Polar cap centres sit at ±90° — peaks are at ±86°, not ±90°.
- ±86° is 4° from the pole, consistent with the FACE 4 / FACE 5 (polar face) grid geometry where the last interior cell-centre row of the polar face lies a few degrees from the pole at C36.
- Longitude structure: peaks at ±90° (mod 180°) — 4 peak locations per hemisphere = **mode-4** in longitude, matching the "mode-4 polar" visual description from iter-717.

**Bug localisation.**  The peak sits on the POLAR FACES (face 4 for N, face 5 for S), at the highest-latitude cell row of each face.  Candidate sources:
1. `pad_halo_vector` handling of the polar-face halo (cube corners meeting at the pole; 4 faces meet at each polar vertex but face 4/5 WRAPS there).
2. `cell_centre_angles_from_4edge` at polar face (the 4-surrounding-edge angle average may be degenerate at cells that border the pole).
3. Arakawa-Lamb gradient (`_arakawa_lamb_gradient`) at the polar face interior where `dx/dy` ratios approach 0.
4. `boundary_fix` averaging at rows 0 and n-1 on face 4/5 where the "boundary" is the pole, not a cube edge.

**Separation from FB-chain bug.**  Iter-732's FB step-1 D-grid drift at cube-vertex `(24, 0)` = NW corner of equatorial faces (lat ~±35.3°) is a DIFFERENT bug.  Two separate paths, two separate failure modes:
- FB chain failure: cube-vertex D-grid halo at the 3-face-meeting vertex.
- A-L+RK3 production failure: polar-face near-pole peak.

**Iter-745 candidate targets.**  Attack the A-L production's polar-face peak directly.  Specific concrete first tests:
- Print `v_ll[face 4, interior]` and `v_ll[face 5, interior]` separately to confirm it's the polar faces (not the equatorial faces near the pole).
- Test whether `boundary_fix=False` on the polar face rows reduces the peak (bounding the role of boundary_fix).
- Test whether swapping `cell_centre_angles_from_4edge` with a pole-aware variant at face 4/5 changes the peak.

**Deliverable.**  `scripts/diag_iter744_w2_artifact_localise.py` checked in.  v_ll_Linf sentinel (iter-740..742) plus the iter-744 peak-localisation together give a concrete bug-hunt target: polar-face near-pole v_north in the A-L+RK3 path.

**Confidence.**  HIGH on "peak is at lat±86°, lon ±90° mod 180°" — this is directly measured from 181×360 regridded data.  MEDIUM on "the polar face is the cell-native source" — requires face-native inspection (iter-745).  LOW on which of the 4 candidates (halo, angle average, A-L gradient, boundary_fix) is the actual root cause.

### Iter-745 — stabiliser ablation: polar peak is face-4/5 interior; hyperdiff+boundary_fix co-amplify it

Per iter-744 next-work list: confirm the face-native source of the peak AND bound which stabiliser in `fv3_sw_tendencies` is responsible.  Delivered as `scripts/diag_iter745_polar_peak_ablation.py`.

**Face-native localisation (baseline, all stabilisers on).**  W2 C36 1 day with `hyperdiff_coeff=_hyperdiff_cube(36)`, `div_damp=_div_damp_cube(36)`, `boundary_fix=True`:

- face 4 argmax: `(i=19, j=17)` → lat `+86.05°`, lon `+71.59°`, v_north = `-3.068e-01` m/s
- face 5 argmax: `(i=19, j=18)` → lat `-86.05°`, lon `+71.59°`, v_north = `+3.067e-01` m/s
- face 0/1/2/3 max |v_north|: 0.087–0.089 m/s (3.5× smaller than polar)

The peak is **unambiguously on the two polar faces**, at cell-centre indices 1–2 cells from the face centre (the pole).  This is NOT a cube-corner location (cube corners of face 4 are at `(0,0), (0,n-1), (n-1,0), (n-1,n-1)`) and NOT an equatorial-face near-pole cell projected by regrid.  Iter-744 candidate 2 (polar-degenerate halo handling) and candidate 4 (polar-row boundary_fix) are both face-4/5-native.  Candidate 1 (`pad_halo_vector` halo rotation) acts at the cube-edge of face 4 (`i∈{0,n-1}` OR `j∈{0,n-1}`) which is at lat ±35.3° — 18 cells away from the peak at face 4 `(19,17)`.  Iter-745b (below) falsifies the non-orthogonal halo alternative; the orthogonal rotation is correct for this convention.  **Candidate 1 (halo rotation) is confirmed DOWN.**

**Stabiliser ablation (v_ll_Linf, W2 C36 1 day):**

| Config                                                    | v_ll_Linf | Δ vs baseline | Peak location          |
|-----------------------------------------------------------|-----------|---------------|------------------------|
| baseline (bf=T, dd=_div_damp_cube, hy=_hyperdiff_cube)    | 3.028e-01 | —             | polar face, lat ±86°   |
| boundary_fix=OFF  (dd/hy on)                              | 3.027e-01 | −0.0 %        | cube corner, lat ±37.6°|
| div_damp=OFF      (bf/hy on)                              | 3.013e-01 | −0.5 %        | polar face, lat ±86°   |
| **hyperdiff=OFF** (bf/dd on)                              | **2.158e-01** | **−28.7 %** | cube corner, lat ±37.6°|
| all stabilisers OFF                                       | 1.518e+00 | +401 %        | cube corner, lat ±37.6°|
| only div_damp ON                                          | 1.396e+00 | +361 %        | cube corner, lat ±37.6°|
| only hyperdiff ON                                         | 3.192e-01 | +5.4 %        | cube corner, lat ±37.6°|
| only boundary_fix ON                                      | 2.481e-01 | −18.0 %       | cube corner, lat ±37.6°|

**Concrete conclusions (narrow, what the numbers support):**

1. **Two distinct failure modes exist in the production path.**  (A) A cube-corner mode at lat ±37.6° that grows to O(1 m/s) when no stabiliser is active.  (B) A polar-face mode at lat ±86° that appears only when `boundary_fix=True` AND `hyperdiff>0`.  They are spatially disjoint and respond differently to the stabilisers — they are not the same bug.

2. **`boundary_fix` AND `hyperdiff` TOGETHER shift the dominant peak from cube corner to polar face — neither alone.**  The table rows "only boundary_fix ON" (0.248 at cube corner) and "hyperdiff=OFF" (0.216 at cube corner) both keep the peak at the corner.  Only the combined "baseline" (bf=T + hy=on + dd=on) and "div_damp=OFF" (bf=T + hy=on) rows show the peak at lat ±86°.  Correct minimal characterisation: **the polar mode at ±86° requires `boundary_fix=True` AND `hyperdiff_coeff>0` to co-act**; boundary_fix alone merely suppresses the cube-corner mode, hyperdiff alone merely adds a small cube-corner residual, but together they produce a polar-interior residual that exceeds both.

3. **`hyperdiff` is the polar-peak amplifier.**  "hyperdiff=OFF" with `boundary_fix`/`div_damp` on drops the Linf 28.7 % AND shifts the peak back to cube corner.  `hyperdiff` is adding ~0.09 m/s at the polar face on top of the residual that `boundary_fix` leaves there.

4. **`div_damp` is a red herring for the polar peak.**  `div_damp=OFF` changes v_ll_Linf by only 0.5 % and does not shift the peak location.  It IS needed to suppress the cube-corner mode under some configurations ("only hyperdiff ON": 0.319 with hyperdiff vs 0.216 without — div_damp is partially doing the `boundary_fix` job when `boundary_fix=False`), but it is not the polar-peak driver.

5. **The core A-L + RK3 tendency is unstable without SOME stabiliser.**  "all stabilisers OFF" gives v_ll_Linf = 1.52 m/s, 5× the baseline.  The stabilisers are load-bearing.  Removing `hyperdiff` wholesale (which would reduce the polar peak) is not viable — it would push the cube-corner mode from suppressed to dominant.

**Smallest-correct Fortran-faithful change this suggests (NOT yet implemented).**  Fortran's d_sw DOES have cell-centre del-n diffusion, but it lives inside three distinct paths: (i) `fv_tp_2d(nord=...)` applied to transported mass and tracers (tp_core.F90); (ii) `_d_sw5_corner_divergence` applying del-n to the divergence field via a `_del6_vt_flux`-style stencil; and (iii) `damp_c=damp_v` in d_sw1's mass-transport call.  What Fortran's d_sw does NOT have is a separate bilaplacian on `(u_east, v_north)` GEOGRAPHIC wind components at cell centres — the Python production path's `laplacian_compact`-on-geographic-winds is a structurally different operator than any of Fortran's del-n pathways.  Iter-746+ target: replace the Python bilaplacian block with a Fortran-faithful `fv_tp_2d(nord=...)`-on-mass-flux AND/OR `_del6_vt_flux`-analogue on divergence/vorticity, then re-measure the polar peak and cube-corner mode together.  Two risks: (a) `div_damp` alone may not suppress cube-corner mode A enough to meet gates; (b) a full `damp_v` port requires the FB chain (the A-L production path has no `_del6_vt_flux` equivalent wired).  These must be addressed together, not one at a time — iter-745's ablation already shows partial combinations worsen the overall Linf.

**Confidence calibration (following iter-731/733 process constraint).**  HIGH on the face-native peak location (`face 4 (i=19, j=17)` etc — direct argmax).  HIGH on "hyperdiff is amplifying the polar peak by ~0.09 m/s" (direct ablation measurement).  HIGH on "boundary_fix AND hyperdiff TOGETHER are needed to produce the polar mode; neither alone does" (direct measurement from eight ablation rows).  MEDIUM on "the smallest-correct fix replaces the `laplacian_compact` bilaplacian with a Fortran-faithful del-n pathway from `fv_tp_2d` / `_del6_vt_flux`" — this is reasoning from iter-744 candidate 3 and iter-745 numbers + Codex fidelity review findings; it has not been directly tested.  LOW on "removing hyperdiff alone will meet the test-matrix gates" — plausibly false per the "all stabilisers OFF" 1.52 m/s result.

**What iter-745 does NOT claim.**
- The artifact is NOT fixed.  v_ll_Linf = 0.303 m/s unchanged.
- No source-code change was made this iter.  The ablation script is diagnostic evidence; it does not modify `fv3_sw_tendencies`.
- "Replace hyperdiff with div_damp-only" is a hypothesis, not a tested fix.  An iter that attempts this must ALSO report W2/W5/cosine-bell/ocean-rest results together, not just the polar peak metric, because the stabilisers are load-bearing for multiple gates.

**Matrix + ocean rest-state baselines (iter-745, recorded for regression):**
- W2 C36 1d: L2 = 2.42e-04, Linf = 1.83e-03, v_ll_Linf = 3.03e-01 (pre-regrid 3.07e-01) — PASS, matches iter-742 baseline.
- W5 C36 1d: mass drift = 1.74e-05 — PASS.
- Cosine bell C36 1d: L1 = 1.20e-01, L2 = 1.17e-01, Linf = 1.23e-01 — PASS.
- Ocean rest state 12/12 PASS (cubed-sphere `eta_drift` ~1e-9, lat-lon / MPAS exact zero).

**Visual inspection.**  `results/atmosphere/shallow_water/williamson2/cubed_sphere/C36/snapshots_v.png` at t=0..1 d shows:
- t=0 d: v ≈ 0 clean (IC).
- t=0.1 d: polar bands emerging at ~±0.1 m/s.
- t=0.5 d: clear mode-4 polar banding at ±0.2 m/s.
- t=1.0 d: fully developed ±0.3 m/s polar bands at lat ±75–86°.

This is the iter-717 user-reported artifact.  **Ralph stopping condition "no visible artifacts on W2" REMAINS UNMET.**

**Deliverable.**  `scripts/diag_iter745_polar_peak_ablation.py` checked in.  Provides the ablation table above + face-native argmax for each configuration — future iters claiming a fix must re-run this script and show both a reduced `v_ll_Linf` AND a reduced per-face polar-face Linf (face 4/5 dropping from 0.307 toward the face 0/1/2/3 baseline of ~0.09).

**Iter-745b addendum (same iteration, different commit) — non-orthogonal halo path is DISPROVEN.**

The iter-745 self-review noted that `pad_halo_vector` in `fv3_sw_tendencies` is called with `cos_theta=None`, taking the orthogonal rotation branch.  It proposed testing the non-orthogonal (Fortran-faithful) path by passing `cos_theta=cdgrid.cosa_cell, sin_theta=cdgrid.sina_cell`.  Iter-745b tests this via `scripts/diag_iter745b_nonorthogonal_halo_rotation.py`, which monkey-patches `pad_halo_vector` to uniformly take the non-orthogonal branch and re-runs the W2 C36 1-day production case.

**Result (baseline → non-orthogonal):**

| Metric      | Orthogonal (baseline) | Non-orthogonal | Δ (%)      |
|-------------|-----------------------|----------------|------------|
| v_ll_Linf   | 3.028e-01 m/s         | **3.908e+01 m/s** | **+12 806 %** |
| v_north_Linf| 3.068e-01 m/s         | **3.955e+01 m/s** | +12 789 %   |
| h_L2        | 2.42e-04              | **5.31e-02**   | +21 824 %   |
| face4 peak  | (19,17), lat +86°     | (18,35), lat +46° | location change |

The non-orthogonal halo path **blows up the solution by 128×**.  The artifact moves from polar-face interior to mid-latitude (lat ±46°) on all faces simultaneously — a different, much worse failure mode.

**Concrete conclusion.**  The production A-L+RK3 path's `u_d, v_d` ARE truly orthogonal-rotated projections of geographic (u_east, v_north), NOT non-orthogonal face-local tangent components.  The `fv3_d2cc` docstring at `operators_cdgrid.py:1280-1282` ("D-grid winds use the orthogonal-rotation convention") is confirmed correct by this test.  Switching the halo to the non-orthogonal path is mathematically WRONG for this convention — it conflates an orthogonal-rotated vector with a non-orthogonal tangent-plane covariant vector and the mis-projection corrupts every halo rotation.

**What this rules out.**  The iter-745 self-correction-2 conjecture (halo rotation's orthogonal choice propagates O(cosa) error that accumulates over 288 steps) is **disproven**.  If the orthogonal choice were introducing a systematic error, switching to the non-orthogonal path would REDUCE it.  The opposite happened by 4 orders of magnitude.  **Halo rotation is NOT a cause of the polar peak.**  Iter-744 candidate 1 is genuinely down.

**Iter-746 ordered work list (revised after iter-745b falsification):**

1. ~~Test the non-orthogonal `pad_halo_vector` path.~~  **DONE, disproven, iter-745b.**
2. Test `fv_tp_2d(nord=...)`-on-C-grid-mass-flux as a replacement for the `laplacian_compact` bilaplacian.  Moderate edit inside the `hyperdiff_coeff > 0` block of `fv3_sw_tendencies`.
3. Test a `_del6_vt_flux`-analogue on divergence (cell-centre scalar) as a Fortran-faithful stabiliser on top of the existing `div_damp`.
4. Investigate the `laplacian_compact` second-difference formula's handling of variable `dx[i,j]` at face-4 near-pole cells.  Current formula `(f[i+1] - 2f[i] + f[i-1]) / (dx[i,j]/2)^2` assumes locally-uniform dx; near the pole on face 4, dx varies 2× across a few cells, introducing an O(Δdx/dx) error in the bilaplacian.  This may be the mechanism by which `hyperdiff` amplifies the polar residual (iter-745 finding: hyperdiff adds +0.087 m/s at polar).
5. If (2)–(4) all fail to materially reduce the polar peak, escalate to the FB chain's `_d_sw_native` at C24 (separate unblock path — see architectural items).

**Process note (iter-745 → 745b same-iteration cycle).**  Iter-745 self-review generated a specific falsifiable prediction (non-orthogonal path reduces polar peak).  Iter-745b tested it in the SAME ITERATION and falsified it at 128×.  This is the Ralph adversarial-fix cycle working: every claim must be tested, not just asserted.  The updated ordered work list now promotes candidate 4 (non-uniform-dx bilaplacian) which has NOT been tested.

### Iter-746 — two mechanism hypotheses falsified by direct measurement

Iter-745b falsified the halo-rotation hypothesis.  Iter-746 targets the other two plausible mechanisms for the hyperdiff polar amplification:
(a) Variable `dx[i,j]` at face 4 near-pole breaks the uniform-spacing assumption in `laplacian_compact` (iter-745 candidate 4).
(b) Rapid `angle[i,j]` variation at face 4 near-pole makes the `cos_a * u_cc - sin_a * v_cc` rotation inject a 2Δx mode that the bilaplacian amplifies.

**Iter-746a: dx variation at polar cells — FALSIFIED.**  `scripts/diag_iter746_polar_dx_variation.py` measures `grid.dx[face=4, :, :]` at and around the polar peak `(i=19, j=17)`:

| Cell        | dx (m)     | Ratio vs centre |
|-------------|------------|-----------------|
| (18, 17)    | 555 862.62 | 1.0000000       |
| (19, 17)    | 555 863.62 | 1.0000000       |
| (20, 17)    | 555 865.62 | 1.0000036       |

Relative 2nd-difference of dx across the x-stencil = **5 × 10⁻⁶**.  Same for dy.  The grid spacing is essentially uniform at the polar peak cells.  `laplacian_compact`'s uniform-spacing assumption is **NOT** the source of the polar hyperdiff residual.  An analytic test with `f = sin(lat)` (Fortran-known Laplacian `-2 sin(lat)/a²`) shows the uniform-spacing formula is accurate to 0.02–0.06 % at ALL face 4 polar interior cells.  The only cells with large relative error (~46 %) are the cube-corner cells at lat ±35.3° — NOT the polar peak cells at lat ±86°.  Iter-745 ordered-work item 4 is **disproven**.

**Iter-746b: angle rotation injection — FALSIFIED on constant field.**  `scripts/diag_iter746b_angle_variation.py` measures `grid.angle[face=4]` 2nd-differences at the polar peak cell:

| Stencil       | Δ angle | 2nd diff of angle |
|---------------|---------|-------------------|
| x-stencil @ (19,17) | −45.50° | **+18.37°** |
| y-stencil @ (19,17) | −75.29° | **−31.86°** |
| x-stencil @ face-0 (18,18) | −0.11° | +5 × 10⁻⁵ ° |

The angle DOES vary rapidly at face 4 near-pole (18° and 32° 2nd difference — three orders of magnitude larger than the equatorial face).  But a DIRECT test feeds a constant geographic `u_east = 10 m/s, v_north = 0` through the full rotate → laplacian → bilaplacian → rotate-back pipeline and checks what comes out.  A pipeline that injects a polar artifact from the rotation would produce a non-zero `Δv` over 1 simulated day; the Fortran-faithful answer is exactly zero.

Result:
- `max|lap(ue_cc)|`     = 2.24 × 10⁻²⁵ (machine zero)
- `max|bilap(ue_cc)|`   = 2.39 × 10⁻³⁵ (machine zero)
- Over 1 day: `max|Δv|` from hyperdiff = **6.2 × 10⁻¹⁴ m/s** (machine precision)

The rotate→laplacian pipeline itself is exact on a constant field — even at the polar face where angle varies rapidly.  The `cos_a, sin_a` round-trip recovers the constant to machine precision.  **The rotation-injection hypothesis is falsified.**

**What iter-746a + 746b collectively rule out.**  The hyperdiff block's polar amplification is NOT explained by (i) variable-dx formula error, (ii) angle-rotation injection on smooth fields, or (iii) the halo rotation (iter-745b).  The remaining viable mechanisms:

1. **Bilaplacian amplifies ACCUMULATED error** that builds up over 288 RK3 steps from another source (e.g., `boundary_fix`'s smoothing at cube-edge rows generates a 2Δx mode at row 1/row n-2 of face 4, which the bilaplacian then amplifies and propagates inward).  Untested.
2. **PPM transport** (in `cgrid_mass_flux_divergence`) has a near-pole limiter behavior that creates subtle structure in `h`, which feeds `B = KE + g*h` and `dB/dx` through the A-L gradient, producing a polar pressure-gradient residual.  Untested.
3. **`dgrid_vorticity`** at cell centres uses circulation from corner winds; near the pole the circulation area varies rapidly and a small error in `u_corner, v_corner` gets amplified via `1/Area` normalisation.  Untested.
4. **`_arakawa_lamb_gradient` grad_c00/c01/c10/c11 matrix** — this is precomputed from 3D Cartesian geometry; at the polar face these matrix coefficients may be ill-conditioned (large condition number) near the pole.  Untested.

**Iter-747 ordered work list (post iter-746 falsifications):**

1. Test if boundary_fix's cube-edge smoothing generates the 2Δx mode that hyperdiff amplifies: compare `bilap_u_local` at face 4 near-pole BEFORE and AFTER boundary_fix application on the ACTUAL simulated state (not a smooth analytic field).  Small diagnostic; falsifiable.
2. Condition-number test of `_arakawa_lamb_gradient` at face-4 polar interior: check if `grad_c00/c01/c10/c11` coefficients are well-scaled.  Cheap to compute.
3. If both (1) and (2) come back clean, escalate to the original `fv_tp_2d(nord=...)` Fortran-faithful replacement of the hyperdiff block — structural change, not a targeted diagnostic.

**Iter-746 deliverables.**
- `scripts/diag_iter746_polar_dx_variation.py` — dx-variation measurement + uniform-vs-nonuniform compact Laplacian comparison on analytic `sin(lat)`.
- `scripts/diag_iter746b_angle_variation.py` — angle-variation measurement + round-trip constant-field injection test.
- Two concrete falsifications narrowing the search space.  No source-code change this iter; matrix + ocean baselines are unchanged from iter-745 (W2 v_ll_Linf = 0.303 m/s).

### Iter-747 — retract iter-746b: pipeline IS biased on smooth non-constant fields

Codex stop-time review on iter-746 flagged: **"iter-746b's diagnostic is a tautological null test, so this turn incorrectly removes angle rotation as a candidate mechanism."**  Correct.  Iter-746b built a CONSTANT `u_east=10 m/s, v_north=0` field, rotated to face-local `(u_cc, v_cc)` using `angle`, then rotated BACK using the SAME `angle`.  This is the identity transformation by trigonometric identity — `ue = cos(a)*(cos(a)*u) + sin(a)*(sin(a)*u) = u`.  The subsequent Laplacian of a constant is trivially zero.  The test proved nothing about how the pipeline handles non-constant fields.

**Iter-747 proper test.**  `scripts/diag_iter747_angle_rotation_proper_test.py` runs TWO non-tautological tests:

(A) The smooth W2 exact IC (`u_east = u_0 * cos(lat), v_north = 0`) passed through the hyperdiff pipeline.  This has spatial structure (cos(lat) varies with latitude) so rotate → laplacian → rotate-back is NOT the identity.

(B) The simulated state at t=1 d passed through the pipeline.

**Test A result (W2 exact IC, NO accumulated error):**

| Face       | max\|hyp_dv\| (m/s/s) | argmax location          |
|------------|------------------------|--------------------------|
| face 0/1/2/3 | 3.3 × 10⁻⁷          | (35, 0), lat ±34.7° (cube corner) |
| face 4/5    | **1.0 × 10⁻⁵**        | (18, 18), lat ±88.2°     |
| **Ratio**   | **31×**                | polar is disproportionately hit |

The polar-face hyperdiff tendency on the SMOOTH exact W2 IC is **31× larger** than on equatorial faces, with the peak at lat ±88° — essentially coincident with the iter-745 observed polar peak at lat ±86° cell `(19,17)`.  This is the pipeline's intrinsic bias.

**Test B result (t=1 d simulation state):**

- Per-face max|hyp_dv| similar across all 6 faces (within factor 1.46)
- Peak locations have moved to cube corners (lat ±36.5°) because the cube-corner mode dominates at t=1 d

The t=1 d test is LESS diagnostic because the simulation state has accumulated the cube-corner error from earlier steps, which swamps the intrinsic polar bias.

**Conclusion (retracting iter-746b).**  The hyperdiff pipeline IS biasing face 4/5 disproportionately on smooth fields.  Iter-746b's constant-field test missed this because a constant field is the ONE smooth field where the pipeline trivially preserves identity.  The angle-rotation hypothesis is **RE-OPENED**: on a non-constant `u_east(lat)`, the face-local finite-difference Laplacian picks up rapid variation of `u_east` values across face 4 cells (because the gnomonic_ed projection compresses near-pole cells, making `cos(lat)` vary rapidly in face-local indices).  Combined with the fact that face 4 covers the polar cap, this produces a 31× bias in the bilaplacian AT the polar cells.

**Codex Q1/Q2/Q3 review (iter-747 incorporated):**

Q1: Fortran's del-n pathway during a shallow-water step does NOT act on cell-centre `(u, v)` anywhere in `sw_core.F90` or `tp_core.F90`.  The `fv_tp_2d(nord=...)` call site list for the shallow water equations is `delp, q_con, pt, tracers` — all A-grid SCALARS.  The momentum-side del-n is `d_sw6`'s `del6_vt_flux` applied to cell-mean VORTICITY (`sw_core.F90:1947-1950, 2008-2121`): `wk = rarea * (vt(i,j) - vt(i,j+1) - ut(i,j) + ut(i+1,j))`, then `del6_vt_flux(wk)`, then `u(i,j) = u(i,j) + vt(i,j); v(i,j) = v(i,j) - ut(i,j)`.

Q2: Fortran uses METRIC-AWARE normalization: `damp = (damp_c * da_min)**(nord+1)` (global minimum area), and flux-form coefficients `del6_u = sina_v*dx/dyc`, `del6_v = sina_u*dy/dxc` (`fv_grid_utils.F90:709-734`).  No per-cell `hx_sq=(dx/2)^2`-style normalization appears anywhere.

Q3: The smallest Fortran-faithful replacement for the Python `hyperdiff_coeff` block at `operators_cdgrid.py:1440-1453` is the d_sw6 → `del6_vt_flux` momentum path: compute cell-mean vorticity from D-grid circulation, apply `del6_vt_flux` with `damp4 = (damp_v * da_min_c)**(nord_v+1)`, then add the returned edge fluxes to `(u, v)` as circulation increments.  The Python `laplacian_compact`-on-geographic-winds is STRUCTURALLY DIFFERENT from anything Fortran does — it is not a del-n port, it is a standalone non-FV3 stabiliser.

**Iter-748 concrete work (prioritised):**

1. **Port `del6_vt_flux` vorticity-form damping** to the A-L production path.  Replace the `hyperdiff_coeff > 0` block in `fv3_sw_tendencies` with a call to a new `_del6_vt_flux`-equivalent operating on cell-mean vorticity (already computed as `zeta` at `operators_cdgrid.py:1414`).  Returned fluxes add to `u_c, v_c`-like positions — requires the D-grid wind increment wiring that the current hyperdiff block's `cos_a * bilap_ue + sin_a * bilap_vn` path replaces.
2. If the del6_vt_flux port is infeasible in a single iter (it is a significant Fortran port), break it into:
   - 2a. Build `_del6_vt_flux` standalone unit, tested against a small synthetic input.
   - 2b. Wire it into `fv3_sw_tendencies` as an optional stabiliser alongside the existing hyperdiff.
   - 2c. Compare the two head-to-head on the iter-745 ablation script + matrix.
   - 2d. Retire `laplacian_compact`-on-geographic-winds once `del6_vt_flux` is validated.
3. Verify iter-747 Test A with boundary_fix DISABLED — if the 31× polar bias persists without boundary_fix, the bias is purely in the hyperdiff pipeline; if it disappears, boundary_fix is a co-factor.

**Iter-747 deliverable.**  `scripts/diag_iter747_angle_rotation_proper_test.py` checked in.  Provides Test A (smooth IC) and Test B (stepped state) hyperdiff tendencies per face.  Confirms the pipeline's polar bias mechanistically and anchors iter-748's `del6_vt_flux` port as the Fortran-faithful fix.

**Confidence.**  HIGH on "pipeline polar bias 31× on smooth W2 IC" (direct measurement).  HIGH on "Python hyperdiff block is structurally non-Fortran" (Codex file:line citations).  MEDIUM on "del6_vt_flux port will reduce the polar peak" — this is the Fortran-prescribed fix but has not been tested in Python.  LOW on "del6_vt_flux alone will meet all matrix gates without boundary_fix or div_damp" — the load-bearing nature of the three stabilisers (iter-745 ablation) means any single-stabiliser replacement must be measured jointly.

### Iter-748 — correct iter-747 mis-attribution: the mechanism is FD-vs-spherical-Laplacian error, NOT angle rotation

Codex stop-time review on iter-747 flagged: **"iter-747's new diagnostic does not isolate angle rotation, but the doc treats it as proof that angle rotation is the mechanism."**  Correct.  Iter-747's Test A passed `u_east = u_0 * cos(lat)` through `rotate → rotate-back → laplacian_compact`.  The round-trip rotation is exactly identity by trig, so the Laplacian operates on `u_0 * cos(lat)` whether or not rotation is in the pipeline.  The 31× polar bias was real but its attribution to angle rotation was unsupported.

**Iter-748 proper isolation.**  `scripts/diag_iter748_proper_mechanism_isolation.py` runs three independent tests:

(A) **FD Laplacian vs analytic spherical Laplacian on `u_0*cos(lat)`** (no rotation anywhere).

| Face         | max\|lap_FD − lap_spherical\| | Rel. to \|lap_spherical\| |
|--------------|-------------------------------|---------------------------|
| face 0/1/2/3 | 4.57 × 10⁻¹³                 | 112 % (at cube corner)    |
| face 4/5     | 7.28 × 10⁻¹²                 | 23.7 % (at lat ±88°)      |
| **ratio**    | **16×**                       | FD is 16× worse on polar  |

(B) **Rotation round-trip effect on the Laplacian.**  Compare `laplacian_compact(u_east_direct)` vs `laplacian_compact(rotate_back(rotate_to(u_east_direct)))`:

| Face | max\|lap_rotated − lap_direct\| |
|------|--------------------------------|
| all  | **~10⁻²⁵ (machine zero)**     |

Rotation contributes MACHINE ZERO to the Laplacian.  The round-trip is exactly identity; any reported difference is round-off.

(C) **Pure rotation-angle field Laplacian.**  `laplacian_compact(cos(angle))` shows how much the angle-variation alone contributes to the FD Laplacian magnitude:

| Face | max\|lap(cos(angle))\| |
|------|------------------------|
| face 0 | 2.58 × 10⁻¹³         |
| face 4/5 | **5.16 × 10⁻¹¹ (200× larger)** |

cos(angle) DOES have a large Laplacian on face 4 (rotation angle varies fast near pole), but this is consumed by the round-trip in Test B.

**Definitive conclusion.**  The 31× polar bias measured in iter-747 Test A is caused by the **flat-grid FD Laplacian being an inaccurate approximation to the spherical Laplacian** when applied to a lat-dependent field on face 4, where `lat[i, j]` varies rapidly in face-local indices (gnomonic_ed compression).  Angle rotation contributes exactly ZERO to the Laplacian value (Test B).

This matches what Fortran's design anticipates.  `sw_core.F90`'s `del6_vt_flux` uses metric-aware flux-form coefficients `del6_u = sina_v * dx / dyc` (`fv_grid_utils.F90:709-734`) and global `da_min` normalisation — so its stencil adapts to the varying cell metrics.  Python's `laplacian_compact` uses `hx_sq = (dx[i,j]/2)^2` as a per-cell denominator but no metric-aware flux form, so its accuracy degrades at cells with strong latitude gradient per face-local index (namely face 4/5 near pole).

**Correction to iter-747 prescription (unchanged).**  The fix IS still to port Fortran's `del6_vt_flux` on cell-mean vorticity as the Fortran-faithful replacement for the `laplacian_compact`-on-geographic-winds block.  What iter-748 corrects is the MECHANISM NARRATIVE (it's FD-vs-spherical, not rotation), not the fix itself.  The fix is robust to whichever mechanism story is right because `del6_vt_flux` is metric-aware and doesn't use flat-grid FD stencils at all.

**Confidence (iter-748 refined).**  HIGH on "flat-grid FD Laplacian has 16× larger error on face 4 than face 0 when applied to `u_0*cos(lat)`" (Test A direct measurement).  HIGH on "rotation round-trip contributes machine-zero to the Laplacian" (Test B direct measurement).  HIGH on "the Fortran-faithful fix is `del6_vt_flux` with metric-aware flux-form coefficients" (Codex file:line citations).  MEDIUM on "this fix will materially reduce W2 v_ll_Linf at the polar peak" — plausible but untested.

**Process note (three same-iteration falsification cycles now).**  iter-745 → 745b (halo rotation disproved), iter-746 → 746b (tautological test, retracted), iter-747 → 748 (wrong mechanism attribution, corrected).  Each Codex stop-time review caught a real flaw.  The iter-748 deliverable is the CORRECT mechanism story + the reconfirmed fix prescription.  Iter-749+ begins the actual port.

**Iter-748 deliverable.**  `scripts/diag_iter748_proper_mechanism_isolation.py` checked in.  Replaces iter-747's mis-attribution with a three-way isolation test that cleanly separates FD-formula error from rotation effects from angle-Laplacian effects.

### Iter-749 — correct iter-748: mechanism is scalar-Laplacian-of-vector-component, NOT FD vs spherical

Codex stop-time review on iter-748 flagged: **"the new diagnostic doesn't test the actual 31× production path, and the checked-in writeup already misstates one of its own results."**  Correct on both counts.  Iter-748 Test A compared single Laplacians (16× face4/face0 ratio on `u_0*cos(lat)`), but iter-747's 31× was on the FULL pipeline `hyp_dv = -hyp_coeff * bilap_u_local` (bilaplacian + rotate-back).  Iter-748 conflated these different quantities when concluding the mechanism.

**Iter-749 direct test.**  `scripts/diag_iter749_full_pipeline_metric_test.py` runs the EXACT production hyperdiff pipeline twice on `u_east = u_0*cos(lat)`:

(A) Production: `laplacian_compact` (flat-grid FD)
(B) Replacement: metric-aware flux-form FV Laplacian (area-weighted divergence of gradient, Fortran-del6-style)

Both pipelines run rotate → lap → bilap → rotate-back and compute the final `hyp_dv` per face.

**Result:**

| Pipeline                               | face 0 max\|hyp_dv\| | face 4 max\|hyp_dv\| | Face 4 / face 0 ratio |
|----------------------------------------|-----------------------|-----------------------|-----------------------|
| (A) `laplacian_compact` (production)   | 3.33 × 10⁻⁷          | 1.03 × 10⁻⁵          | **31.00×**            |
| (B) fv_laplacian (metric-aware)        | 3.61 × 10⁻⁷          | 1.03 × 10⁻⁵          | **28.68×**            |

**The polar bias is essentially unchanged (31× → 28.68×) when switching from flat-grid FD to a metric-aware flux-form Laplacian.**  The iter-748 attribution ("flat-grid FD vs spherical Laplacian is the mechanism") is **falsified** by this direct test.

**Correct mechanism (iter-749).**  Both `laplacian_compact` and `fv_laplacian` are SCALAR Laplacians.  Applied to `u_east = u_0*cos(lat)` — which is a VECTOR COMPONENT, not a scalar — they produce polar-singular behaviour because:

- The true SCALAR spherical Laplacian of `u_0*cos(lat)` is `∇²f = -u_0 * cos(2*lat) / (cos(lat) * a²)`, which diverges as `lat → ±90°` (1/cos(lat) singularity).
- Any reasonable approximation to the scalar Laplacian must capture this singularity → large values at the polar face.
- The VECTOR Laplacian in spherical coordinates has EXTRA metric terms (`Γ`-type Christoffel corrections) that cancel the scalar divergence and give a physically meaningful result for `u_east`.
- Scalar-Laplacian-applied-to-vector-component is NOT a vector Laplacian; it is not physically meaningful, and its pole behaviour is an ARTIFACT of treating a vector component as a scalar.

This is why Fortran's `d_sw6 → del6_vt_flux` (Codex Q1) damps VORTICITY — a true scalar field that is not pole-singular.  The Fortran design never applies a scalar Laplacian to a vector component, precisely to avoid this issue.

**Fix prescription (unchanged, re-affirmed with correct mechanism).**  The Python production hyperdiff block is structurally wrong because it treats vector components as scalars.  No scalar Laplacian (flat-grid FD, metric-aware FV, or analytic spherical) will fix this.  The ONLY Fortran-faithful replacement is `del6_vt_flux` on cell-mean vorticity — which is what Codex Q3 already recommended.

**Two iter-748 sentences to retract as wrong:**

1. "The 31× polar bias measured in iter-747 Test A is caused by the flat-grid FD Laplacian being an inaccurate approximation to the spherical Laplacian."  — WRONG.  iter-749 shows the metric-aware Laplacian gives 28.68× bias, not <3×.  FD vs spherical is not the primary mechanism.
2. "Python's `laplacian_compact` uses `hx_sq = (dx[i,j]/2)^2` as a per-cell denominator but no metric-aware flux form, so its accuracy degrades at cells with strong latitude gradient per face-local index."  — TECHNICALLY TRUE but MISLEADING.  Switching to the metric-aware flux form does not fix the polar bias.  The primary driver is the scalar-vs-vector issue, not the metric form.

**Iter-749 deliverable.**  `scripts/diag_iter749_full_pipeline_metric_test.py` checked in.  Runs the full production hyperdiff pipeline twice (flat-grid FD and metric-aware FV) on the W2 exact IC and directly measures the per-face `hyp_dv`.  The near-identity of the two ratios (31× vs 28.68×) is the decisive evidence that the mechanism is scalar-Laplacian-of-vector-component, not FD-vs-spherical.

**Confidence (iter-749 final).**  HIGH on "scalar Laplacian of `u_0*cos(lat)` is polar-singular regardless of FD vs metric-aware form" (Test A+B direct measurement).  HIGH on "the Fortran fix `del6_vt_flux` on vorticity is correct because vorticity is a true scalar with no polar singularity" (structural argument + Codex citations).  HIGH on "iter-748's flat-grid-FD mechanism attribution is wrong" (direct falsification).

**Process note (four falsify-own-claim cycles now in 5 iters).**  iter-745→745b (halo rotation), iter-746→746b (tautological test), iter-747→748 (angle rotation mis-attribution), iter-748→749 (FD-vs-spherical mis-attribution).  Four successive Codex stop-time catches.  The FIX prescription (del6_vt_flux port) has been stable and correct since iter-747; only the MECHANISM story has required iteration.  Iter-750+ should proceed directly to the port without further diagnostic.

### Iter-750 — close iter-749 with proper analytic-Laplacian control

Codex stop-time review on iter-749 flagged: **"iter-749's 'FD-vs-spherical is falsified' conclusion is not supported by the checked-in control."**  Correct.  iter-749's `fv_laplacian` was ITSELF a flat-grid flux-form, not a proper spherical Laplacian.  Comparing two flat-grid Laplacians cannot falsify the FD-vs-spherical hypothesis.

**Iter-750 proper control.**  `scripts/diag_iter750_analytic_spherical_laplacian.py` uses the ANALYTIC spherical scalar Laplacian of `u_east = u_0*cos(lat)`:
```
∇²(u_0*cos(lat)) = −u_0 · cos(2·lat) / (cos(lat) · a²)
```
This diverges as `lat → ±90°` because `cos(lat) → 0` in the denominator.  Evaluating it at each cell centre gives the exact spherical Laplacian — the true target any numerical approximation should hit.

**Result:**

| Cell/quantity             | Analytic ∇²f (m⁻¹s⁻¹) | FD `laplacian_compact` | FD relative error |
|---------------------------|------------------------|-------------------------|-------------------|
| face 0 (18,18), equator   | 9.51 × 10⁻¹³          | 1.02 × 10⁻¹²           | 7.4 %             |
| face 4 (19,17), lat +86°  | **1.37 × 10⁻¹¹**      | 1.37 × 10⁻¹¹           | **0.41 %**        |
| face 0 (35,0), cube corner| −4.08 × 10⁻¹³         | −8.65 × 10⁻¹³          | 111.9 %           |
| face 4 global max         | 3.08 × 10⁻¹¹          | 3.81 × 10⁻¹¹           | (ratio)           |
| face 0 global max         | 9.51 × 10⁻¹³          | 1.02 × 10⁻¹²           | (ratio)           |

- **Analytic face 4 / face 0 ratio: 32.37×**
- **FD face 4 / face 0 ratio: 37.27×**

**Conclusion.**  The analytic spherical scalar Laplacian of `u_0*cos(lat)` ITSELF has a 32× polar bias — the 1/cos(lat) singularity is inherent to the scalar formula, not an FD artifact.  At the iter-745 polar peak cell `(19, 17)`, FD matches analytic to **0.41 %** — FD is essentially exact there.  The polar bias is NOT caused by FD inaccuracy; it is caused by the scalar spherical Laplacian being INTRINSICALLY pole-singular when applied to `u_east` as if it were a scalar.

This **confirms** iter-749's "scalar-of-vector" mechanism attribution at the level of the actual control (analytic), and **re-falsifies** iter-748's "FD-vs-spherical" claim but with proper evidence this time.

**Where the FD approximation IS inaccurate.**  At the equatorial cube corner `face 0 (35, 0)`, FD has 112 % relative error — a separate, cube-corner-specific inaccuracy that does NOT drive the polar peak.  This is iter-745's cube-corner mode A, distinct from the polar mode B.

**Fix prescription (unchanged, now with fully-supported mechanism).**  `del6_vt_flux` on cell-mean vorticity.  Vorticity is a TRUE scalar field (the z-component of `curl(v)` in local coords), not a vector component.  It does not have the 1/cos(lat) polar singularity that `u_east = u_0*cos(lat)` does.  Fortran damps vorticity precisely to avoid this issue — `sw_core.F90:1581-1597, 1947-1950`.

**Process note (5 falsify-own-claim cycles now in 6 iters).**  iter-745→745b (halo rotation), iter-746→746b (tautological test), iter-747→748 (angle rotation mis-attribution), iter-748→749 (metric-form mis-attribution), iter-749→750 (insufficient control).  The FIX prescription (del6_vt_flux port) has been stable since iter-747; mechanism story is now finally anchored by analytic-Laplacian evidence.  Iter-751 proceeds directly to the port.

**Iter-750 deliverable.**  `scripts/diag_iter750_analytic_spherical_laplacian.py` checked in.  Uses the ANALYTIC spherical Laplacian as a proper control.  Definitively shows that:
1. The scalar spherical Laplacian is INTRINSICALLY 32× polar-biased on `u_0*cos(lat)`.
2. FD `laplacian_compact` matches the analytic formula to 0.41 % at the iter-745 polar peak.
3. The polar bias is scalar-of-vector, not FD-vs-spherical.
4. Cube-corner FD errors (112 %) are a separate mode A bug, not the polar driver.

### Iter-751 — iter-750 also premature: FD compounds through bilaplacian

Codex stop-time review on iter-750 flagged: **"iter-750 closes the mechanism story with a control that does not validate the production operator it cites."**  Correct — the production hyperdiff operator is `laplacian_compact(laplacian_compact(u_east))` (bilaplacian), but iter-750 only validated the single Laplacian.  Iter-750's claim "scalar-of-vector mechanism CONFIRMED at bilaplacian level" was premature.

**Iter-751 direct bilaplacian test.**  `scripts/diag_iter751_bilaplacian_validation.py` compares:

(A) `bilap_FD = laplacian_compact(laplacian_compact(u_0*cos(lat)))` — the production.
(B) `bilap_hybrid = laplacian_compact(analytic_∇²(u_0*cos(lat)))` — FD only on the OUTER Laplacian.

If FD error is negligible (as iter-750 suggested for the single Laplacian), (A) and (B) should agree at the polar peak.

**Result at face 4 (19, 17), lat +86°:**

| Quantity                                | Value        |
|-----------------------------------------|--------------|
| (A) `FD(FD(u_0*cos(lat)))`              | 2.07 × 10⁻²² |
| (B) `FD(analytic_single_Lap)`           | 1.07 × 10⁻²² |
| **Relative diff (A vs B) at polar cell**| **94 %**     |

The FD bilaplacian and FD(analytic_inner) DISAGREE by 94% at the polar peak cell.  FD error in the single Laplacian (0.41% at the cell, but varying spatial structure) DOES compound through the outer Laplacian to ~factor-2 disagreement.  Iter-750's "scalar-of-vector CONFIRMED at bilaplacian level" is **not supported** by direct test.

**At face 0 (18,18), equator, the (A) vs (B) relative diff is −0.26%** — essentially exact.  So FD error compounds significantly only at face 4, not at face 0.

**What the data actually supports (honest reduction).**  The bilaplacian pipeline is polar-biased (face 4 / face 0 ratio ≥ 28× in all tested forms).  Both FD error compounding AND analytic-scalar-Laplacian pole-singularity contribute; iter-751 does NOT cleanly separate their individual magnitudes.  What IS clear:
- FD single-Laplacian is 0.41% accurate at the polar peak cell (iter-750 direct).
- FD bilaplacian is factor-2 different from a hybrid analytic-then-FD at that cell.
- Both FD and hybrid bilaplacians are polar-biased (28× and 4090× face 4 / face 0 ratios respectively).

**What iter-751 does NOT claim.**  It does not claim scalar-of-vector is or isn't the dominant mechanism; the evidence is mixed.  It does not claim FD compounding is the dominant mechanism.  It claims only that the bilaplacian pipeline is polar-biased in all tested forms, which is sufficient to motivate the Fortran-faithful fix.

**Fix prescription (still unchanged, for robust orthogonal reasons).**  Port `del6_vt_flux` on cell-mean VORTICITY.  This fixes BOTH potential mechanisms at once:
1. Vorticity is a true scalar with no 1/cos(lat) pole singularity, so the "scalar-of-vector" issue vanishes.
2. `del6_vt_flux` uses metric-aware flux-form coefficients `del6_u = sina_v*dx/dyc` (`fv_grid_utils.F90:709-734`), avoiding the flat-grid FD compounding.

Either mechanism (or both) drives the polar peak; the Fortran-faithful fix is robust to the actual attribution.  This is why the FIX has been stable since iter-747 while the MECHANISM story has needed six iterations of refinement — the fix doesn't depend on resolving the mechanism debate.

**Process note (6 falsify-own-claim cycles in 7 iters now).**  iter-745→745b (halo rotation), iter-746→746b (tautological test), iter-747→748 (angle rotation mis-attribution), iter-748→749 (metric-form mis-attribution), iter-749→750 (insufficient single-Lap control), iter-750→751 (single-Lap control doesn't validate bilaplacian).  Each Codex stop-time catch found a real methodological gap.

**Hard-stop on further mechanism diagnostic.**  Iter-752 will stop diagnosing and proceed directly to the del6_vt_flux port.  The mechanism story may never be cleanly resolvable without first implementing the fix and seeing which piece changes.  The cost of one more failed mechanism diagnostic exceeds the cost of starting the actual port.

**Iter-751 deliverable.**  `scripts/diag_iter751_bilaplacian_validation.py` checked in.  Honestly reports the mixed evidence and hard-stops further mechanism diagnostic work.

### Iter-752 — port core `_del6_vt_flux` standalone (first source-code step of the Fortran-faithful fix)

Per iter-751's hard-stop on further diagnostic, iter-752 begins the actual Fortran-faithful fix: port `del6_vt_flux` from `sw_core.F90:2008-2121` to Python.  This iter delivers ONLY the core standalone algorithm, tested in isolation.  Wiring into `fv3_sw_tendencies` is iter-753+.

**New module.**  `src/legoesm/core/fv3_del6_vt_flux.py` (155 lines) providing:
1. `compute_del6_metrics(cdgrid)` → `(del6_u, del6_v)` using Fortran's non-USE_SG formula from `fv_grid_utils.F90:713, 725`:
   ```
   del6_u(i,j) = sina_v(i,j) * dx(i,j) / dyc(i,j)    at v-edge (6, n, n+1)
   del6_v(i,j) = sina_u(i,j) * dy(i,j) / dxc(i,j)    at u-edge (6, n+1, n)
   ```
2. `_del6_vt_flux(q, damp, nord, del6_u, del6_v, rarea, cdgrid)` → `(fx2, fy2)` — the Fortran algorithm verbatim:
   - Initial `d2 = damp * q` (vorticity scaled by damping coefficient).
   - Pre-loop `fx2 = del6_v * (d2_west - d2_east)`, `fy2 = del6_u * (d2_south - d2_north)`.
   - Iterate `nord` times: update `d2 = rarea * (fx2 - fx2_east + fy2 - fy2_north)`, then recompute `fx2, fy2` with SIGN FLIPPED as per `sw_core.F90:2099, 2112`.
   - Return `(fx2, fy2)` — edge diffusive fluxes to be added to (u, v) as circulation increments downstream.

**New tests.**  `tests/test_fv3_del6_vt_flux.py` — 6 unit tests covering:
- Metric shapes `(6, n, n+1)` / `(6, n+1, n)` correct.
- Metric values positive (product of sina × dx/dyc positive quantities).
- Del-2 flux of a CONSTANT field is zero (centred difference of constant).
- Output shapes correct for all `nord ∈ {0, 1, 2}`.
- Linearity in `damp` preserved: doubling `damp` doubles output at all `nord`.
- Zero input gives zero output.

All 6 tests PASS.

**Matrix + ocean baselines unchanged** (the standalone module is not yet wired into the production path).  W2 v_ll_Linf = 0.303 m/s.

**What's left for iter-753+:**
1. Compute cell-mean vorticity `wk = rarea * (vt - vt_south - ut + ut_east)` where `vt = u*dx, ut = v*dy` (circulation form).  `operators_cdgrid.py` already has `zeta` from `dgrid_vorticity` — can reuse or port Fortran's exact stencil.
2. Compute `damp4 = (damp_v * da_min_c)^(nord_v+1)` — need a `damp_v` and `nord_v` config entry.  Suggestion: use existing `hyperdiff_coeff` rescaled appropriately for initial smoke test.
3. Wire `_del6_vt_flux` into `fv3_sw_tendencies` as an OPTIONAL alternative to the existing `laplacian_compact`-on-geographic-winds block (controlled by a config flag, so both paths can be measured head-to-head).
4. Apply returned fluxes as circulation increments to `u_d, v_d` — this requires mapping the cell-centre fx2, fy2 back to D-grid u, v positions.  Fortran does `u(i,j) += vt(i,j); v(i,j) -= ut(i,j)` at D-grid positions directly.
5. Run ablation (iter-745 script) + matrix + visual to compare polar peak reduction.

**Iter-752 deliverables.**
- `src/legoesm/core/fv3_del6_vt_flux.py` — standalone unit (155 lines).
- `tests/test_fv3_del6_vt_flux.py` — 6 passing unit tests.
- W2/W5/cosine-bell matrix still PASS; ocean rest 12/12 still at machine precision; W2 v_ll_Linf = 0.303 m/s (unchanged — standalone unit not yet called by production).

**Process.**  First source-code change in seven iters of diagnostic.  Smallest viable step: build + test the core unit before wiring.  Iter-753's wiring is the next concrete step.
