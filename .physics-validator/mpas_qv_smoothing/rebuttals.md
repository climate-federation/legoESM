# Round-1 finding classification (codex review-1.md)

## CONFIRMED (fix required)

- **B2 — hybrid dp <= 0 over high terrain (CRITICAL).** Numerically verified
  (`scripts/tmp/_probe_hybrid_dp.py`): default MPAS coord
  `make_hybrid_levels(40, p_top=200, stretching=2.0)` has surface-layer
  `dA=-0.027<0`, so `dp = dA*p_ref + dB*p_s` crosses zero at **p_s=662.7 hPa**;
  at Tibet (600 hPa) min(dp)=-256 Pa. `vertical_coord` DEFAULTS to `"hybrid"`
  (config.py:64). The mass-weighted smoother divides by this dp -> Inf/NaN /
  negative q_v over Tibet/Andes/Antarctica, then the dycore's next-step
  `maximum(q_v,0)` (primitive_eq_mpas.py:~910) fabricates water. Real for the
  realistic-topography AMIP use case.

- **B1 — setup geometry-only CFL cannot bound the mass-weighted operator.**
  Confirmed: dp_edge/dp_cell is unbounded (148x sea/Tibet). No setup-time guard
  can establish the DMP for the evolving pressure-weighted operator. The x2
  headroom silently assumed dp_n/dp_c<=3, false over terrain.

- **"same convention as diagnostics" claim false.** Confirmed: smoother used
  hybrid dp; the hard-sat drain uses pure-sigma pressure. (Dissolved by the fix,
  which drops pressure entirely.)

- **Test weakness: driver guard duplicated in the test (can drift).** Fix by
  extracting a shared `scalar_del2_cell_cfl_factor(mesh)` used by both.

- **Test weakness: positivity/conservation not exercised on hybrid dp.** Fix by
  a driver-path regression test on the real hybrid coord at low p_s.

## ROOT-CAUSE FIX (not a symptom patch)

FV's own qv smoother (`_apply_qv_smoothing`, compiled_segments.py:1368) is a
PLAIN scalar hyperdiffusion + `jnp.maximum(q,0)` floor; water conservation is
handled separately by `fix_moisture`. Mirror it: **plain SCVT del2 (no dp) +
positivity floor**. This makes the operator coordinate-agnostic (no dp division
-> no NaN), the geometry CFL guard EXACT (plain-del2 monotonicity factor is
geometry-only), positivity guaranteed (convex combo under the guard; floor is a
no-op insurance), and conserves the per-level `sum_c A_c q_c` integral exactly.
Drops the unachievable "exact column water MASS" guarantee — which the FV lane
does not provide either, and which is impossible robustly on a dp<=0 coordinate.

## REFUTED / not bugs (codex agreed these hold)

- Conservation for fixed positive dp; sign & constant annihilation; padding /
  scope / shapes / Field.replace / tracer-dict copy; no-op default & no retrace;
  validate_strict lane refusal + bounds; CLI round-trip. All correct.

## ACCEPTED-RISK / minor (note, no change or light change)

- **NamedTuple positional-ABI: field inserted mid-struct.** No positional
  construction in-repo (all keyword); sits with sibling `mpas_land_*` fields
  added the same way. Low risk; leave.
- **x64 forced at module import, not restored; fp32 not tested.** Production MPAS
  AMIP runs fp64 (campaign: fp64/dt240), so x64 tests match production. Common
  repo pattern. Leave (note).
- **mixed_fp64_storage policy bypassed by eager compute.** Perf/precision, not
  correctness; production is fp64. Note only.
- **del2 vs del4.** FV uses del4 (more scale-selective); MPAS smoother uses del2
  (author's choice; sufficient to damp grid-scale speckle; more diffusive at
  meso-scale). Acceptable; del4 a future refinement.

# Round-2 finding classification (codex review-2.md)

## CONFIRMED (fixed)

- **#1 (blocker): "FV handles conservation separately" is FALSE.** MPAS never
  calls fix_moisture; FV runs its fixer BEFORE the smoother. Plain del2 across a
  p_s gradient is genuinely NON-CWV-conserving (bias ~ humidity–terrain corr).
  FIX: honest docs everywhere — it is an explicitly non-conservative grid-scale
  filter conserving only the per-level sum_c A_c q_c integral; monitor the water
  budget. (Robust CWV conservation is impossible with explicit down-gradient
  diffusion when dp<=0, so this is the correct honest framing, which codex
  accepts.)
- **#2 (major): "mirrors FV" overstated (FV=del4, mine=del2).** FIX: language
  corrected to del2 (∇²) vs FV del4 (∇⁴), with the monotone/conservation
  tradeoff stated.
- **#3 (major): finiteness/fp32 claims need qualification** (maximum(NaN,0)=NaN;
  fp32 default). FIX: docs say the floor sanitises finite negatives, NOT
  pre-existing NaN; "exact" is exact-arithmetic/x64. Added an fp32 driver test.
- **#5: tests replicate the operator expression, not the driver line.** FIX:
  extracted module-scope `_mpas_qv_smooth_step` (the real driver code, mirroring
  FV's module-scope `_apply_qv_smoothing`); the driver calls it and the test
  exercises it directly (incl. hybrid-low-p_s regression + fp32).
- **#7: retained mass-weighted branch is a docstring-only footgun.** FIX:
  REMOVED the dp branch entirely — the operator is plain-only. Reverting the
  driver to `dp=` is now a TypeError (structurally stronger than any test).
- **#8: field mid-tuple shifts positional ABI.** FIX: moved to the END of
  ExperimentConfig.

## REFUTED / not bugs

- **#6 YAML adapter gap.** The `src/legoesm/config.py` YAML→canonical adapter maps
  a CORE SUBSET only (grid/dycore/output/held_suarez); it does NOT map ANY sibling
  opt-in knob (mpas_land_lapse/beta, hard-sat overrides, sponge, snow-albedo,
  conv-cloud). My field is consistent with every sibling; repo policy requires a
  CLI flag (done + round-trip tested), not YAML wiring. Adding only mine would be
  inconsistent. No change.
- **CFL/helper correctness.** Codex CONFIRMED g_c matches div(grad(q)), padded-edge
  mask correct, <=0.5 stricter than <=1. (Helper clamps dcEdge, operator does not
  — identical for any valid SCVT mesh where dcEdge>0; the clamp only guards masked
  padding gathers. Not a bug.)
