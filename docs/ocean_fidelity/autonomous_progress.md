# Autonomous Execution Progress Log

Spec: `docs/ocean_fidelity/autonomous_execution_spec.md`. Branch: `matching_Veros_oracle`.
One entry per task attempt. Newest at the bottom of each task block.

## Queue status
- [ ] Q1 — Phase G density residual: diagnose-then-fix (wall-padding prime suspect)
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
