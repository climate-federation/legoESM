# DIFF REVIEW r1: Zhang-McFarlane net rain flux takes the surface route on the spectral bridge

Worktree: /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6_mem (branch wb/cam6-baseline, base a0038f279). The CLAIM behind this diff was reviewed first
(docs/wb/reviews/2026-09-24_zm_spectral_surface_route_claim.md; verdicts: codex HOLD on guard/test/measurement points,
GLM SHIP with asks). Findings addressed in this diff:
- codex 1 / GLM 5: q_v AND q_c must be present (new raise), skip covers both the q_r slot and the q_c fold.
- codex 2: tests assert zero q_r tendency with a rain tracer present, unchanged q_c, missing-q_c refusal,
  and the Tiedtke (>=0 per-layer) routes byte-identical with and without q_r.
- codex 3 / GLM 2: REAL ZM kernel run through the spectral bridge on convecting columns (20 sigma levels,
  deep to stable): pressure-weighted column closure sum dp*(dq_v+dq_c+dq_r)=0, surface rain >= 0 everywhere and
  > 0 somewhere, convective mask non-empty, booked q_v/q_c equal the kernel's outputs exactly.
- GLM 4: gated on the trait alone (no rain_to_surface knob exists on the spectral bridge; none added).
- GLM 7: total precipitation is NOT a WB training target on this lane (grep of the training script, the
  rollout and the deck loss block: no precip term); silent-zero surface precip pre-exists for every scheme there.
- Non-vacuity: with the integration.py change reverted (control worktree at a0038f279) the 4 ZM tests FAIL
  with the old refusal and the 2 Tiedtke tests pass: 4 failed, 2 passed. With the change: 6 passed.
Out of scope, stated: no conv_precip carry published on spectral (nothing reads it there); no surface-precip
channel on SpectralHydrostaticState; bechtold/tiedtke per-layer sign histogram (GLM 3) untouched behaviour.

Review the diff at docs/wb/reviews/2026-09-24_zm_spectral_r1.diff against the code. Attack: correctness of the skip (both
branches), the guard, JIT-safety (all gates static Python), differentiability (no new non-diff op), test
vacuity, sign/units, scope words. Verdict SHIP/HOLD first, then numbered findings with file:line.
