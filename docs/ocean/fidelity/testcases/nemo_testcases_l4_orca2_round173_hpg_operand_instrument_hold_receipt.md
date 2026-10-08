# ORCA2 round 173 — kt=8 HPG operand instrument hold

**Status:** `HELD_INSTRUMENT_NOT_PASSIVE`  
**Claim label:** every attempted scientific comparison is **independent**:
hierarchy rung 0 starts from its own climatological T/S, zero velocity and zero
sea surface. No given-NEMO-entry rung-7 number is mixed into this receipt.  
**Producer:** committed CPU/fp64 tree; evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round173/`.

## Question and compiled program

Round 172 named HPG as the first non-bit and explosive kt=8 RHS accumulator,
then requested HPG's inputs and statement boundaries. The operator acquired
and admitted that rank-complete record.

The executing rung-0 program forms `rhd` with `eos` and calls HPG first
(`ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo/stp2d.f90:145-149`). The compiled
`hpg_sco` then forms the surface `zhpi/zhpj`, `zuap/zvap`, and their sums
(`ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo/dynhpg.f90:402-429`), followed by the
same cumulative boundaries at interior levels
(`ORCA2_OMIP_L4_R172HPG8/BLD/ppsrc/nemo/dynhpg.f90:433-460`). The registered
walk was therefore `rhd`, `e3w`, `gdept_z0`, metric reciprocal, `zhpi`, `zuap`,
then sum, with U before V inside paired boundaries.

## Oracle record admission

The operator's launcher finished with `STATUS PASS_R172_HPG8_ADMISSION`.
Both self-describing rank records parse, cover 148 x 180 exactly once, expose
all eleven registered fields, and retain byte-identical kt=1..10 terminal
restarts against the round-170 producer. The header, field-name,
field-dimension, truncation, swapped-rank and restart-byte plants all fired.
The admission JSON SHA-256 is
`b70fe4dd72a192e7ab6d4f2516e2c3a4e35e37bf64cd712e3cae7ce71c656711`.

The round-173 gate also replayed the recorded operands through the previously
certified literal HPG helper. All six recorded `zhpi`, `zuap` and sum rows were
bit-exact before the gate reached its candidate-side passivity refusal. Thus
R173-P1 and R173-P2 are **CONFIRMED**.

## Candidate instrument refusals

No candidate operand or statement value is admitted.

The first committed instrument reconstructed the candidate HPG inputs in a
separate compiled graph. Its resulting HPG moved 17 active U cells and 13
active V cells relative to the existing standalone component boundary. The
maximum movements were `8.470329472543003e-22` and
`1.376428539288238e-21 m s-2`. Although both paths retained the same
`1.5835360371918837e51` U and `1.6377650405462966e51 m s-2` V maxima, a
near-zero low-bit refusal is still a refusal. The log SHA-256 is
`ad1ac62d1bba31cb1dd55c63b3c35df7649ca94b9f3f183959924b051d9586e4`.

The corrected instrument exposed `rhd`, `e3w` and `gdept_z0` from the existing
standalone component graph with a host callback, then compared exposed and
unexposed HPG outputs. It reproduced exactly the same 17 U / 13 V differences
and maxima. Therefore the exposure changes the quantity it observes and is not
an instrument. The gate refused before writing a scientific JSON report. Its
log SHA-256 is
`aa2bc04c34d8f5d3d790409cefb10c6eb3aecad1f196d174d9840015879ed23e`.

The first attempt's separate-graph statement rows and the second attempt's
published operands are withheld. Neither can name `rhd`, geometry, metrics or
any compiled HPG statement. R173-P3 and R173-P4 are **UNMEASURED**, not
refuted: their classifier is downstream of the failed passivity prerequisite.
R173-P5 is **CONFIRMED**: no package physics, card, configuration, carried
state, stabiliser or sea-ice selector changed.

## Scope, review and tests

This round lands only a frozen preregistration, a fail-closed measurement gate,
its controls, and this receipt. No `packages/` file changed, so neither ORCA2
ladder, the independent month, GYRE, DINO nor the tanks can move. The held
halo/transport atomic unit remains private and unchanged.

The requested review disposition is **independent review unavailable
in-sandbox**. `codex exec --sandbox read-only` failed before reading the diff
with `failed to initialize in-process app-server client: Read-only file
system`; review log SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The focused round-173 controls report 8 passed. Citation and full fidelity
battery results are recorded below after their terminal summaries are read.

## OPEN

1. Extend the existing fixed-shape standalone operator-component bundle with
   the already-computed `nemo_hpg_rhd`, `e3w` and `gdept_z0` operands; do not
   add a callback or a second evaluation.
2. Before reading them, require the extended and unextended component calls to
   return array-identical HPG U/V and require kt=1..7 live traced versus
   untraced states and existing component arrays to remain bit-identical.
3. Only if that passivity gate is exact, repeat the frozen operand walk. A
   non-bit operand routes the next round to its producer; it does not license a
   downstream HPG statement claim.
4. The complete halo/transport unit remains HELD. No configuration decision or
   NEMO acquisition is needed.
