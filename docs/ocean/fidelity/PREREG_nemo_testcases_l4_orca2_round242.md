# Preregistration — ORCA2 round 242 generic HPG route repair

Date: 2026-10-11. Frozen base: `560309036`. Scope: repair the round-241
shared-path regression, finish its landing gates, then continue the already
ordered downstream operand measurement. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round242/`.

Every ORCA2 number is labelled either **independent** or **given NEMO's
entry**. Sea ice, its selectors, and the ORCA2 card's `unmeasured_features`
tuple are immutable in this round.

## Frozen source statement and regression

The executing NEMO vector branch writes literal HPG into `pvv` without a
compact face mask at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427`, then applies its
stored raw `vmask` only in the depth integral at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215`.

Round 241 transcribed that ordering, but its production predicate calls
`nemo_qco_resolved_mesh_operands` for every vector-form WS-RK3 card. That
resolver correctly refuses a non-partial analytic coordinate without a raw
NEMO mesh bundle. GYRE therefore fails before stepping, although its analytic
mesh previously used the card-built operand source. The repair keeps the raw
resolver for a card carrying `nemo_een_barotropic`; otherwise it uses the
existing `nemo_qco_card_mesh_operands` with the already-computed reference
thickness and live masks. No card selector or physics option is added.

## Frozen predictions and falsifiers

1. **R242-P1 — shared-path repair.** The five identified GYRE recipe tests
   pass unchanged. A raw-bundle ORCA2 card still selects
   `nemo_qco_resolved_mesh_operands`; analytic GYRE selects the card-built
   operands. Any test edit, GYRE raw-resolver call, or ORCA2 route change
   **REFUTES** P1.
2. **R242-P2 — round-241 science preserved.** The completed V-RHS gate still
   reports `668/1,050 -> 0/1,050` unequal slots under both labels, with zero
   off-support movement. OMT-4, rung-0 and rung-10 retain zero moved rows above
   the fixed floor. Any changed endpoint **REFUTES** P2.
3. **R242-P3 — shared gates.** GYRE's certified ladder, residual NPZ and year
   snapshots are byte-identical to the pinned round-241 values; DINO, tanks,
   the focused battery, and the 128-test push gate pass. A new reproducible red
   **HOLDS** the landing.
4. **R242-P4 — downstream owner.** With alpha, exact geometry and the literal
   HPG boundary active, the source-ordered external-mode table reduces
   `vn_adv` from `919/8,589` unequal to approximately zero; the materialised
   `zFv` table then has exact `e1v`, `e3v`, `vv`, and `vmask`, with `zvb` exact
   if `vn_adv` closes. Failure to move those rows **REFUTES** this owner
   prediction and keeps the complete round-217 unit private.
5. **R242-P5 — refusal split.** If the five `zFv` operands are exact but the
   complete unit still reaches the kt=8 live-W-thickness refusal, its four
   parts are bisected at the kt=7-to-8 guard boundary. The first producing
   part and the first bad cell's ssh/depth are reported; no package change
   follows without a separately cited statement and full gate.
6. **R242-P6 — controls and documentary gates.** Route-selection, endpoint,
   citation and trajectory plants must fire. The citation gate, its rigid-
   shift plant, focused tests and independent read-only review are recorded.
   A control that stays green **INVALIDATES** its measurement.

## Stop conditions

No configuration choice, carried-state change, stabiliser, NEMO source edit,
or NEMO acquisition is authorized. A missing passive operand record yields
`ACQUISITION_NEEDED`; an unresolved deck choice yields `DECISION_NEEDED`.
Failed predictions remain in the receipt as `REFUTED`.
