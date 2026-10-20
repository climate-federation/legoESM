# NEMO testcase grid ladder L4b — round 1 preregistration

Date: 2026-10-20

Starting merge: `e68604baf0b8f78c8df6ab91bbd934a9fdbe8709`, joining
the certified lane-A L4 parent `b4f70973d0d4148fe4390a3a7071568ad892f623`
and the certified lane-B L1--L3 parent
`fb02c4b29f397fbd1e8bdbcb10239ec9a1a2102f`.

Status: **PREREGISTERED BEFORE L4b IMPLEMENTATION, ORACLE ACQUISITION, OR
SCORING.** Decision 144 defines L4b as certified G5 plus certified L2
standard-isoneutral diffusion plus certified L4 variable geothermal heating,
with the stage-3 tracer calls in NEMO order. This turn may prepare a WRITE-only
oracle acquisition, but Codex may not execute NEMO or score L4b.

## Frozen merge-preservation measurement

Before L4b changes, a committed two-step identity probe will compare every
floating state leaf from the merge tree against the parent that owns the card:
G1--G9 and L4 against the lane-A parent; V1--V5 and L1--L3 against the lane-B
parent. Every compared leaf must be bit-identical. A planted one-bit mutation
must be refused. Any real difference blocks L4b implementation until the merge
selector conflict is repaired.

The four textual merge conflicts and their frozen resolutions are:

1. the historical GYRE receipt keeps lane B's re-anchored citations;
2. the shared recipe keeps the union of lane A's geothermal selector and lane
   B's V1--V5/L1--L3 selectors, including lane B's repaired L1 end-of-stage
   pivot fold;
3. the ladder scorer keeps the union of geothermal, TKE-mixing-length, and
   DDM arms and all corresponding case tables; and
4. the receipt citation map keeps the union, with every shifted pin to be
   re-anchored by the citation gate before the final commit.

## Frozen L4b card and order

The card name is `ORCA2L4b-iso-bbc`. Its deck is L2's rendered deck plus
exactly L4's `nambbc` block: `ln_trabbc=T`, `nn_geoflx=2`, `cn_dir='./'`, and
the exact variable `sn_qgh` descriptor. No EIV, BBL, damping, tracer
advection, or vertical-mixing selector is added.

The executed oracle call order is frozen from
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-740`: `tra_ldf`, then
`tra_bbc`, then the off L4b neighbors `tra_bbl` and `tra_dmp`. The production
stage-3 temperature RHS must therefore materialize the certified L2 tendency
addition first and the certified geothermal addition second. L4 and every
other existing card retain their prior step program.

A new default-false order selector may be true only with the geothermal
selector and the complete certified L2 standard-isoneutral selection. The
validator must refuse: order without geothermal, geothermal plus any lateral
diffusion without the order selector, the order selector without L2, the
written FCT Krhs tuple, EIV, BBL, or damping. Exact x64 tests use JIT and traced
inputs and must fail when either ordered add is removed or reversed. No new
division or square root is introduced; the existing geothermal and L2 kernels
retain their gradient-only guards and forward bits.

## Frozen acquisition and admission boundary

The committed entry point will live at
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l4b_orca2iso_bbc/run.sh`.
Its default action is a read-only seven-item preflight; `--run` remains
operator-only, and `--admit-existing` validates an operator-produced record.
The preflight must prove source patch context and syntax, executable/deck
identity, input manifest, exact record schema/counts, an acting smoke witness,
writer-free boundary behavior, and an absent acquisition target. The same
validators must refuse one planted violation for each item.

An operator run will execute a two-step smoke, a writer-free boundary run,
and three independently staged twins to a common endpoint no greater than ten
steps. Numeric log tokens are parsed as binary64 and compared exactly with
the rendered deck. Observers write only cells NEMO defines. Admission rechecks
logs, rendered decks, smoke, all twin payloads, raw twin identity, and restart
passivity. No `cmd | grep -q` construct is permitted under `pipefail`.

Frozen outcome: this round stops at `ACQUISITION_NEEDED`. No trajectory score,
bar verdict, or certification claim is allowed before an admitted L4b record
exists.

## ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| merge certified lane B | ASKED | immutable two-parent merge plus exact two-step preservation probe |
| add G5 + L2 + geothermal | ASKED, Decision 144 | one L4b card and NEMO-ordered stage-3 composition |
| add EIV, BBL, damping, or altered vertical mixing | forbidden | all remain off |
| execute NEMO in this sandbox | operator-only | never |
| score or certify L4b | forbidden this round | no admitted record exists |
| change any existing card's two-step values | forbidden control | exact zero movement |
