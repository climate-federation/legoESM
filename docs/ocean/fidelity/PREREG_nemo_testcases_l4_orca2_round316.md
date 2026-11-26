# PREREG — ORCA2 round 316: gradient-only NaN guards (frozen before measuring)

Base `27dc0be613c2` (round 315 landed). Operator decision: apply the
gradient-only remedies of round 315's corrected receipt; forward values must
stay bit-identical; the tridiagonal forward-mode rule stays deferred.

Instrument for forward bits: `nemo_testcase_l4_orca2_round316_forward_bits.py`
(SHA-256 of every state leaf's bytes at every kt, plus the step jaxpr hash),
base vs each group, on rung 0, OMT-1..4, rung 10, GYRE, VORTEX x4, OVERFLOW,
LOCK_EXCHANGE. Final: the seven-pair ladder rows against round 313's shipped
rows (no model change since), GYRE ten-step residual archive and day-30
snapshots, DINO month gate at landing (operator).

Groups (each a separate commit, bits re-proven after each):
- A: dry-cell denominators in the stage-1/2 tracer update (`:1933`, `:1946`,
  the `q_after` combine and its inactive alternate) -> 1 where zero.
- B: zero pivots in the literal vertical solve; zero top-face thickness in
  the literal surface-stress division -> 1 where zero.
- C: rung 10: square roots of clipped-to-zero arguments in internal-wave
  mixing and TKE (`legoesm.core.profile_metrics.safe_sqrt`), the masked
  below-bottom thickness in shortwave penetration -> 1 on masked lanes.
- FIX 2: drop the five `float()` casts in both drag paths.

Predictions:
- P1: every group is forward bit-identical on every card (falsifier: any
  leaf hash moves at any kt; that group stays out).
- P2: after A, the one-step rung-0/OMT-4 T-loss reverse gradient is finite
  and equals forward mode to <= 1e-12 relative along T_dir0 (falsifier: NaN,
  or a larger gap).
- P3: after A+B, the walker records 0 sources on rung 0 and OMT-1 (both
  losses, chained and fused).
- P4: after A+B+C, rung 10's one-step walker records 0 sources; its fused
  reverse gradient is finite. Rung-10 forward-mode NaN: no prediction of its
  location (OPEN from round 315); it may survive C.
- P5: after FIX 2 the drag coefficient traces on every card that refused.
- P6: multi-step fused `jit(grad)` on rung 0 is finite after A (falsifier:
  NaN — a source hidden behind the first).
- P7 (branch-switch reading): on OMT-2, the number of lanes whose
  limiter/where/max/min selection differs between x+eps and x-eps along
  T_dir0 falls with eps, and the FD error is nonzero only where flips occur
  (falsifier: FD error stays ~linear in eps at eps where 0 flips occur).
