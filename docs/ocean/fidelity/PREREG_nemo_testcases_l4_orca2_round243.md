# Preregistration — ORCA2 round 243 substep-6 V-transport operands

Date: 2026-10-11. Frozen base: `92358803b`. Scope: split the first
interior-over-floor row left by round 242, OMT-4 kt=1 stage-1 external
substep 6 `metric_transport_v`, using only admitted passive records and the
current production D114 mesh route. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round243/`.

Every result is reported separately as **independent** or **given NEMO's
entry**. Sea ice, configuration selectors, carried state, stabilisers, and the
ORCA2 card's `unmeasured_features` tuple are immutable. No NEMO acquisition is
planned: the admitted 65-substep advective-mean record contains all three
operands for this statement.

## Frozen source statement and operand order

The record-producing compiled build is
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dynspg_ts.f90`. OMT-4 reuses that
binary and changes namelist switches only. The executing statement is
`dynspg_ts.f90:530-536`:

```
zhV(ji,jj) = e1v(ji,jj) * va_e(ji,jj) * zhvp2_e(ji,jj)
```

The frozen source order is therefore `e1v`, `va_e`, `zhvp2_e`. The comparison
uses the exact current-tree candidate operands consumed by
`nemo_literal_metric_transports`, the admitted NEMO operands from
`oracle_bt_advmean_operands_kt00000001.bin`, and NEMO's recorded `zhV` target.
The candidate product is replayed with the same written association as the
compiled statement. The round-242 passive trace and admission predicates must
remain green.

## Frozen predictions and falsifiers

1. **R243-P1 — record and passivity.** Both twins remain exact on defined
   payload, the production-JIT observer remains bit-passive, fp64/libm is
   printed, and both entry labels produce the same split. Any moved completed
   state, missing stream, label disagreement, or non-fp64 run
   **INVALIDATES** the measurement.
2. **R243-P2 — metric.** D114's raw ORCA2 geometry leaves `e1v` bit-exact at
   substep 6. Any unequal `e1v` cell **REFUTES** P2 and stops the attribution
   at geometry.
3. **R243-P3 — first operand.** `va_e` is the first non-bit operand in source
   order; its individually sub-floor difference is magnified by the two
   metric/depth factors into round 242's 29 interior over-floor `zhV` cells.
   An exact `va_e`, or `zhvp2_e` being first, **REFUTES** P3.
4. **R243-P4 — cumulative closure.** Substituting recorded operands in source
   order removes all 29 interior over-floor cells by the `va_e` row; after all
   three substitutions the replay is bit-exact against recorded `zhV`.
   Residual interior cells after all three operands **INVALIDATE** the replay;
   closure only after `zhvp2_e` **REFUTES** the single-operand owner.
5. **R243-P5 — landing predicate.** This round is measurement-first. A package
   change is eligible only if one cited NEMO statement is isolated, the
   changed operator is 0 cells unequal given NEMO inputs, and the full
   OMT-1..4/rung-0/rung-10/GYRE/DINO/tank gates satisfy Decision 96. Otherwise
   the result is **HELD** with the next upstream operand named.
6. **R243-P6 — controls.** Plants for label coverage, source order, one-ULP
   operand movement, replay closure, and passivity must each refuse. A plant
   that stays green invalidates its corresponding claim.

## Stop conditions

No configuration choice, NEMO source edit, acquisition, or downstream
measurement contaminated by a newly named upstream operand is authorized.
If the admitted record cannot split substep 6, write a fail-closed acquisition
request rather than inferring the operands. Failed predictions remain in the
receipt as retractions or refutations.
