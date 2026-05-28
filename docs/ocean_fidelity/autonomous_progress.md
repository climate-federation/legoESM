# Autonomous Execution Progress Log

Spec: `docs/ocean_fidelity/autonomous_execution_spec.md`. Branch: `matching_Veros_oracle`.
One entry per task attempt. Newest at the bottom of each task block.

## Queue status
- [x] Q1 — Phase G density residual: FIXED (grid aligned to Veros; all-region L2 ~0.037)
- [ ] Q2 — Per-process tendency comparison (face-stagger interpolation)
- [ ] Q3 — Register fidelity modules (test discovery)
- [ ] Q4 — ConstantsConfig (G-C1…G-C5)
- [ ] Q5 — CI clarity guard
- [ ] Q6 — Equivariance tier expansion
- [ ] Q7 — Config regrouping + footguns
- [ ] Q8 — Decompose baroclinic_tendencies (bit-identical gate)

---

## Q1 — density residual

### 2026-05-28 · iter 1 · START (diagnosis)
Prior committed result (test_equivariance.py, green): cumsum/flip order exonerated
(Δρ≈2.5e-15); ρ-vs-depth bounded ≈0.05 kg/m³; EOS at T=0,S=0 → −27 kg/m³/cell.
Open question the diagnosis must answer BEFORE fixing: does the *interior* region mask in
`build_region_masks` include the zero-padded wall cells (so walls explain interior L2=5.1),
or is interior already wall-clean (so the residual is cause (a) time-level, not walls)?
Action this iter: read `tendency_probe.py` (build_region_masks + rho comparison) and
`veros_state_bridge.py` (wall-row padding + land_mask handling) to settle it.

### 2026-05-28 · iter 1 · DIAGNOSIS CONFIRMED + FIX APPLIED
Confirmed (CPU probe): ACC grid is 44 rows = NY(42) + 2 padding rows at lat=-41 and +45,
OUTSIDE the physical domain [-40,+44]. Both padding rows are WET in the recipe land_mask
(row0 100% via lat<-20; row-1 97% via lon>1) and land in `interior` (435+405=840 cells).
The bridge (`veros_state_bridge._pad_y_walls`, lines 201-208) zero-fills their T,S → EOS
sees T=S=0 → rho≈997 → −27 kg/m³/cell. Predicted interior L2 ≈ 5.79 kg/m³ vs observed 5.1
→ CAUSE CONFIRMED = non-physical zero-padded rows in the comparison, NOT a numerics/EOS bug.
FIX: `build_acc_land_mask` now marks rows with lat outside [Y_ORIGIN, Y_ORIGIN+NY*DYT] as
land (in-construction via land_mask_override → u/v masks stay consistent). No production
numerics touched.
GATE (a) [no wet T=S=0 cells]: GREEN — new test `test_bridged_acc_state_has_no_zero_TS_wet_cells`
+ `test_acc_land_mask_marks_out_of_domain_padding_rows_as_land`.
GATE (c) [no regression]: GREEN — 248 passed / 1 skipped across fidelity + recipe + probe +
equivariance + recipe_constants.
GATE (b) [Veros re-run: interior density L2 < 0.1, corr > 0.999]: PENDING — requires running
compare_tendencies_acc.py (Veros, slow). Next iteration verifies (b); only then is Q1 DONE.
Committing the verified increment now.

### 2026-05-28 · iter 2 · GATE (b) interior MET, but Q1 NOT DONE — deeper bug found
Ran compare_tendencies_acc.py (Veros executed, exit 0). Density: interior L2 = **0.0376
kg/m³** (was 5.1; 135x drop), interior pattern-corr = **1.0000**. Gate (b) as DECLARED is met.
BUT boundary L2 = 12, equator/ML L2 ≈ 4 — NOT waved past. Diagnosis (CPU probes
/tmp/q1_boundary.py, /tmp/q1_mask.py):
- 465 WET cells (legoESM mask) carry Veros land values T=0&S=0 (rho 997 vs Veros 1024,
  up to 26.7 kg/m³). [Indicator T==0 AND S==0 is sound: ACC ocean has S=35; only land has
  both zero. Deep ocean has T=0 but S=35 — not flagged.]
- Pattern: legoESM lon-centres = {0,2,...,58}; Veros western wall is TWO columns wide
  (ocean from col 2), legoESM analytic mask (lon>1) lands only col 0 → legoESM OVER-WETS
  col 1 (lon=2) by 31 cells at lat>=-20. Root cause = **half-cell lon grid-centre offset**
  between build_acc_grid and Veros xt. Interior density still matched (corr 1.0) only
  because T,S are ~zonally uniform; the offset bites only at the sharp wall.
- Gate-(a) test was a **FALSE GREEN**: it used the synthetic VerosResult mock (random
  non-zero T,S), which cannot reproduce Veros's land masking.
VERDICT: Q1 NOT done. Interior gate passed on a technicality while a real grid-alignment
bug remains. Per anti-gaming, do NOT close Q1.
NEXT (iter 3): read Veros `set_grid` (in /home/dbalwada/veros) for the exact xt/yt centre
convention; fix build_acc_grid to match it (so analytic kbot lands identically), re-derive
land mask, keep build_acc_z_coord/lat consistent. Strengthen the gate-(a) test to assert
legoESM wet footprint == Veros ocean footprint (no wet T=0&S=0) using a Veros-realistic
zeroed snapshot, not the random mock. Then re-run gate (b) and confirm boundary L2 also
drops to physical levels. Report regenerated this iter (not committed; Q1 still open).

### 2026-05-28 · iter 3 · Q1 DONE — build_acc_grid aligned to Veros xt/yt
Read Veros set_grid + ran a live ACCSetup: Veros xt=[-1,1,...,57], yt=[-41,...,41]
(u-centred: first centre = origin - d/2). create_regional_latlon_grid places interior
cell i at lower+(i-0.5)d with N/S walls at index 0/-1. build_acc_grid was OFF: lon=[0,..,58]
(half-cell), lat interior shifted by a row + 1-row bridge offset. Density still matched
point-wise but the wall mis-aligned (boundary L2=12) and momentum metrics f(lat) were wrong
-> would corrupt Q2.
FIX (veros_acc_recipe.build_acc_grid): lon_west=X_ORIGIN-DXT/2=-1, lon_east=59,
lat_south=Y_ORIGIN-DYT=-42, lat_north=+42 -> interior centres now coincide with Veros
xt/yt exactly; the symmetric bridge y-padding then aligns automatically (no bridge change).
build_acc_land_mask: kbot now lands Veros's 2-wide wall exactly; mark N/S wall rows (0,-1)
land. Removed the earlier (lat-value-based) in_domain hack.
GATES: (a) NEW Veros-free locks test_acc_grid_centres_match_veros_xt_yt +
test_acc_land_mask_matches_veros_ocean_footprint (wet/col == [11,11,42x28]) GREEN.
(b) Veros re-run: density L2 interior=0.0376, boundary=0.0376, equator=0.0375, ML=0.0234,
corr=1.0000, sign=1.0 — UNIFORM ~0.037 across ALL regions (no contamination anywhere). The
~0.037 is the genuine physics residual (rho-vs-geometric-depth, bounded <=0.05 earlier).
< 0.1 tol MET. (c) no-regression: 255 passed across fidelity+recipe+equivariance+probe (the
one earlier fail was my own float32-too-tight test tol, corrected). Q1 honestly COMPLETE.
NEXT: Q2 — face-stagger interpolation so du_*/dv_*/dtemp_*/dsalt_* compare per-process.
