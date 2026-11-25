# ORCA2 round 179 — independent V-RHS HPG operand walk

Date: 2026-10-08. Base `0f7aa62f4`; measurement commit `e56b98427`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round179/`.
Status: **HELD**.

Every trajectory number is **independent hierarchy rung 0**. No
given-NEMO-entry result is mixed into this receipt. No production model,
configuration, carried-state rule, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Verdict

The round-178 68-face V-forcing debt is produced by the first operator in
NEMO's compiled accumulation order: HPG. NEMO calls HPG before LDF, VOR, KEG
and ZAD
(`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:141-175`). At the completed
2-D HPG boundary, all 68 northern-fold faces on row 147 differ: RMS
`1.0748834693208086e-06` and maximum `2.4423127429248864e-06 m s-2` at
`[147,135]`. LDF, VOR, KEG and ZAD do not move that result on the registered
faces.

The first non-bit operand of NEMO's literal depth average is the raw 3-D HPG
V RHS. NEMO constructs that operand by its surface and interior `zhpj+zvap`
assignments
(`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427`), before the
compiled vertical average multiplies the 3-D RHS by `e3v_3d`, `vmask`, and
`r1_hv_0`
(`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215`). On the 1,319 wet
levels contributing to the 68 faces, the raw HPG operand differs in all 1,319
cells, with maximum `1.1053405011865161e-03 m s-2`. Its one-variable arm is
non-vacuous: replacing only raw HPG while retaining NEMO's depth-average
geometry moves all 68 faces (RMS `2.458998106935011e-04`, maximum
`5.051998857187377e-04 m s-2`).

This does **not** authorize a landing. The same 68 faces also differ in all
three later depth-average operands: `e3v` differs on 27 contributing cells,
`vmask` on 1,319, and `r1_hv_0` on all 68 faces. Each one-variable arm moves
the result. In particular, the raw-HPG arm is much farther from NEMO than the
completed baseline, so the operator is inside a cancelling multi-operand
unit. Round 179 names the first statement boundary and holds it; it does not
land one half of that unit.

## Mechanical closure

The passive live-state wrapper reproduces ordinary T/S/u/v/ssh bit-for-bit.
The pure offline HPG/LDF/VOR/KEG/ZAD replay reproduces the production
completed-V debt set exactly: 68 versus 68 faces, zero symmetric difference,
and zero maximum difference at the completed after-ZAD boundary. Both rank
shards cover every admitted record exactly once.

The NEMO after-ZAD accumulator and its recorded depth average differ in 428
sign bits of numerical zero, with RMS and maximum exactly 0.0. The next
depth-average-to-completed-forcing boundary is bit-identical. This refutes the
preregistered bit-exact cross-record prediction without changing any physical
score; the gate retains the REFUTED result rather than silently weakening it.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R179-P1 record assemblies and cross-record identities close bit-exactly | **REFUTED**: 428 signed-zero bits differ; both boundaries close at exactly 0.0 magnitude. |
| R179-P2 offline replay reproduces the production completed accumulator | **CONFIRMED**: 68/68 faces, zero symmetric difference and zero maximum calibration residual. |
| R179-P3 HPG is the first operator | **CONFIRMED**. |
| R179-P4 `vmask` is the first operand and its arm reproduces candidate HPG | **REFUTED**: raw HPG is first; mask-only matches candidate magnitude but differs on 23 zero sign bits. |
| R179-P5 measurement only | **CONFIRMED**. |

## Mechanical controls, tests, and review

Rank-placement, record-bit, source-order, cross-record, target-mask,
first-arm-non-vacuity and endpoint-ULP plants all exit 2 with
`STATUS PLANT-FIRED`.

The clean focused battery passes 45/45 (round-179 gate, citation gate, RHS,
slow-forcing and static-operand parsers). The default and round-179 citation
gates pass with zero unmapped spans or audit failures; shifting the HPG
citation by two lines makes the citation gate exit 1.

The required single `tests/ocean/fidelity -n 12` invocation collected 2,828
tests and reached 99%, with 2,794 passes, 7 skips and 4 pre-existing failures,
then reproduced the known nonterminal tail and was interrupted without a
terminal pytest summary. Twenty-three tests were unfinished. The failures are
the round-35 stamp-scope ratchet, round-129 GYRE record-backed gate pin, the
worktree-stamp ratchet and the SI3 scalar-math provenance gate. No round-179
focused test failed.

The required separate read-only Codex review was attempted and returned
**independent review unavailable in-sandbox**: `failed to initialize
in-process app-server client: Read-only file system`.

Artifact:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round179/v_rhs_operator_walk.json`.

## OPEN

1. Split the raw HPG V statement in NEMO source order: `zhpj` accumulation,
   `zvap` correction, then `pvv = zhpj + zvap`, using round 42's certified
   literal-input extractor and an admitted NEMO component record. Stop at the
   first component or input that differs; do not add an in-executable legoESM
   observer.
2. Analyse the cancelling unit jointly: raw HPG plus `e3v`, `vmask`, and
   `r1_hv_0`. No half lands unless the complete unit is a Decision-96 net
   improvement under both ladders and the shared-card gates.
3. Only after the upstream unit closes, re-test the held V-transport/halo unit
   and the corrected independent month's step-96 live-thickness refusal.
   The 240-step month score remains **UNMEASURED-with-spec**.

No acquisition is needed for the completed operator/depth-average result.
An HPG component split may reuse the already-admitted round-41/42 component
record if its entry provenance matches the corrected independent rung-0
state; otherwise round 180 must write a new fail-closed acquisition before
reading components. ASKED choices are the source-ordered HPG component walk.
UNASKED choices are empty.
