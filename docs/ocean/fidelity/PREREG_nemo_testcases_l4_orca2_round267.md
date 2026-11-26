# ORCA2 round 267 preregistration — source-exact HPG accumulator carry

Date: 2026-10-13. Frozen base: `a864b99f4`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**. This round continues the
round-266 matched-program-point walk using the admitted stage-2 term and HPG
literal records. No deck, card selector, carried state, stabiliser, sea-ice
field, or `unmeasured_features` entry may change.

## Compiled statement and candidate boundary

NEMO assigns the already-computed literal HPG sums directly into the stage
accumulator at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynhpg.f90:401-402` and
`:426-427`, before `dyn_vor` and vector-form `dyn_adv` are called in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:372-419`.
Round 266 proved the candidate's live HPG term is at the campaign floor against
that literal, while its subsequent optimisation-barrier materialisation is the
first non-floor boundary. The private arm therefore removes only that extra
HPG materialisation and carries the existing production literal into the
accumulator. It does not substitute an oracle array or recompute HPG.

## Frozen predictions and falsifiers

1. **R267-P1 — reproduction.** With the arm off, both labels reproduce round
   266: HPG accumulator maximum `4.6166034627296855e-7 m/s2`, matched stage-2
   T/S no larger than `2e-10`, and passive completed states. Any mismatch is an
   instrument regression and stops the round.
2. **R267-P2 — HPG closure.** With only the source-exact HPG carry enabled,
   the HPG cumulative row falls to `<=2e-10` under both labels and its literal
   source remains at the floor. A larger residual refutes the arm; no package
   landing or downstream attribution follows.
3. **R267-P3 — next boundary.** After HPG closes, the accumulator walk resumes
   in compiled order: VOR, KEG, ZAD. The registered prediction is that KEG is
   the next non-floor self-replay boundary, near round 266's
   `4.616603462729692e-7 m/s2`. If VOR or ZAD is first instead, that measured
   boundary replaces the prediction and is retained as **REFUTED** evidence.
4. **R267-P4 — completed RHS.** If the next boundary is another candidate-only
   materialisation of an already-exact NEMO addend, a second private arm may
   carry that exact addend in NEMO's written addition order. Only closure of
   the completed stage-2 RHS under both labels permits scoring the complete
   atomic unit through OMT-4 kt=1..10 and the full Decision-96 gates. A
   non-closing boundary is **HELD**; missing operands are
   `ACQUISITION_NEEDED`.
5. **R267-P5 — controls.** Both-label coverage, clean-tree provenance, record
   identity, arm-off reproduction, passivity, source order, HPG-arm
   sensitivity, next-boundary sensitivity, completed-RHS closure and a
   one-bit control each have a planted refusal.

## Stop and landing conditions

This is a CPU-fp64/libm matched-program-point measurement. No NEMO run is
planned. The private arm is unavailable to public configuration. No model
statement lands unless its own cumulative row is at the bar, the completed RHS
closes, and the complete unit passes every required trajectory and regression
gate. No floor, clip, damping, alternate mask, carried-state change,
configuration choice, or sea-ice change is permitted.

ASKED choices: none. Unasked configuration choices made: none.
