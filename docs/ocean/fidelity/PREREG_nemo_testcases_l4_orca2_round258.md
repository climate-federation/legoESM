# ORCA2 round 258 preregistration — stage-1 pivot V-transport operands

Date: 2026-10-12. Frozen base: `d401dd65e`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**, with the current complete
private fold/transport unit used by rounds 256–257. This is an offline
measurement round. It changes no deck, card, selector, carried-state contract,
stabiliser, sea-ice field, or `unmeasured_features` entry.

## Existing instrument and compiled order

No new numerical method is permitted. The round extends the committed
five-operand split in
`nemo_testcase_l4_orca2_round232_v_transport_operand_gate.py` and the
current-tree stage trace in
`nemo_testcase_l4_orca2_round257_tracer_writer_gate.py`. The admitted
round-230 record is self-described and already contains the five operands.

The executing record build constructs `zvb` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:269-279`, then writes
`zFv` in the order `e1v`, live `e3v(Kmm)`, `vv(Kmm)`, `zvb`, `vmask` at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:282-285`. The
compiled compact V-fold copy is
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:684-721`; the completed
stage calls it for V with sign -1 at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802`.

## Required reconciliation

Round 247's `zFv 668 -> 0` used its private
`_resolved_candidate_stage_geometry` context and explicitly replaced the
stage transport's compact 3-D mask with the D114 resolved mask. Round 257's
`zFv` result used the current complete private unit without that held stage-mask
arm. Round 258 must print the arm, support, and record identity side by side;
it may not call the two historical rows contradictory unless those three
quantities agree.

## Frozen predictions and falsifiers

1. **R258-P1 — provenance and passivity.** Both labels run CPU fp64/libm
   production JIT from one clean committed tree, use the production D114
   geometry route, and reproduce round 257's `zFv` census and stage-1
   `0.17735225873138316 K` / `3.283688696376231 PSU` physical-half boundary.
   A dirty tree, wrong commit, missing label, non-passive trace, or failed
   reproduction invalidates the round.
2. **R258-P2 — written operand order.** Score `e1v`, live `e3v(Kmm)`,
   `vv(Kmm)`, `zvb`, and `vmask` in exactly that compiled order on the NEMO-live
   pivot support. The first value-unequal operand owns the walk. Signed-zero
   differences are reported separately and do not become a numerical owner.
3. **R258-P3 — compact stage-mask prediction.** `e1v`, live `e3v(Kmm)`,
   `vv(Kmm)`, and `zvb` are predicted bit-exact on the 668 pivot-row slots;
   `vmask` is predicted first unequal, candidate zero where NEMO's folded raw
   V mask is one. Any valued mismatch in an earlier operand, or an exact
   `vmask`, **REFUTES** this prediction and the earlier operand is named.
4. **R258-P4 — source-exact causal arm.** Replacing only the stage transport's
   compact V mask by the already-resolved D114 V mask must reduce the 668
   pivot-row `zFv` differences to zero and reduce the stage-1 physical-half T/S
   value differences to zero under both labels. If `zFv` stays non-bit, the
   five-operand replay is insufficient. If `zFv` closes but T/S does not, the
   transport is exonerated and the first later consumer remains open.
5. **R258-P5 — round-247 reconciliation.** With the resolved-mask arm enabled,
   the new five-operand table must reproduce round 247's exact `vmask` and
   `zFv` rows on the same support and record. Failure to reproduce leaves both
   measurements unresolved; it does not select the newer result by provenance.
6. **R258-P6 — disposition.** The held post-stage/in-loop associations remain
   governed by pending Decision 116. No partial operand or association lands
   this round. A source-exact mask result is retained as a private atomic-unit
   measurement unless every already-authorised predicate can be satisfied
   without selecting D116.
7. **R258-P7 — controls.** Exact worktree, both-label coverage, compiled order,
   pivot support, round-257 reproduction, candidate replay, round-247
   reconciliation, first-owner selection, mask substitution closure, and a
   one-bit operand perturbation each have a planted violation that must refuse.

## Stop conditions

If the admitted record lacks any pivot-row operand, if the offline replay does
not reproduce the candidate and NEMO endpoints, or if the two historical arms
cannot be reconstructed exactly, the round is `STOPPED_FOR_RECORD` or `HELD`
without attribution. No in-executable observer or NEMO acquisition is allowed.
