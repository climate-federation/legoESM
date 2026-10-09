# ORCA2 round 196 — substep-3 AB3 midpoint-V input hold

Date: 2026-10-09. Incoming tip:
`776389945b6aa09b148f0712bacc16e4160f401e`. Preregistration:
`ac90cb82d`. Measurement instrument: `cb91cc774`; citation correction:
`21e43ac1d`.

## Result

**HELD.** Every number is **independent hierarchy rung 0**. The card starts
from its corrected climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry or rung-10 number is mixed into this result. The shipped ORCA2
card, sea ice, all six ice selectors and its `unmeasured_features` tuple are
unchanged. `git diff 776389945b6aa09b148f0712bacc16e4160f401e..HEAD --
packages` is empty.

The substep-3 AB3 coefficients are bit-exact, and both older V histories are
bit-exact. The first non-bit input is current `vn_e`: 16,506 of 26,640 record
cells differ, maximum absolute difference
`9.009781378703638e-07 m s-1`, first at `[j=0,i=50]`. Replacing only that
recorded current velocity closes the completed substep-3 midpoint `va_e`
bit-for-bit: its 15,943 unequal cells and maximum
`1.604736666251539e-06 m s-1` become 0 / 26,640 and 0.0.

The source order is exact. NEMO selects the three full-AB3 coefficients and
evaluates `va_e = za1*vn_e + za2*vb_e + za3*vbb_e` at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:486-511`, then rotates
`va_e` into the next substep's `vn_e` at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:842-844`. Thus substep 3's
current input is exactly the completed substep-2 V update; this round names
that upstream boundary but does not yet attribute its arithmetic or boundary
association.

The resolved rung-0 run takes the vector-form update
(`ORCA2_R96SPG/ocean.output:765`), whose completed raw V statement is
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:715-727`; the ordinary
seven-array boundary association follows at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:770-779`. Those two
source-ordered boundaries are the next walk. No depth, reciprocal or trajectory
census follows this operand result.

## Frozen source-ordered table

| boundary | unequal / 26,640 | maximum absolute | disposition |
|---|---:|---:|---|
| `za1` | 0 | 0.0 | exact |
| `za2` | 0 | 0.0 | exact |
| `za3` | 0 | 0.0 | exact |
| current `vn_e` | 16,506 | 9.009781378703638e-07 m/s | **first non-bit input** |
| prior `vb_e` | 0 | 0.0 | exact |
| prior-prior `vbb_e` | 0 | 0.0 | exact |
| `za1*vn_e` | 16,506 | 1.604736666251539e-06 m/s | carries current-input debt |
| completed `va_e` | 15,943 | 1.604736666251539e-06 m/s | non-bit |
| substitute current `vn_e` only | 0 | 0.0 | exact |

The admitted record supplies `j003_ext_coef`, `j002_va_new`, `j001_va_new`,
`i000_vn_e` and `j003_va_ext` from both ranks exactly once. The record-only
history replay is bit-exact. Swapping the current/prior mapping is deliberately
non-exact on 15,943 cells with maximum `0.018062577575284934 m s-1`, proving
the history-rotation control is non-vacuous. The passive traced and untraced
solver states remain array-identical in SSH, U, V, both external modes and both
transport averages.

The measurement artifact is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round196/midpoint_v_inputs.json`
(SHA256 `65778292eec8e8bf784e7c43a6760b2ba4061eea96e4a49e60591cfe51fb880d`;
log SHA256 `32f450066e403ec2a79b653dd3a8fb7dbdb03a7b684b054e99574be5d57d7b4f`).

## Prediction ledger

| prediction | verdict |
|---|---|
| R196-P1 record sufficient | **CONFIRMED**: all five streams exist and record-only replay is exact. |
| R196-P2 three coefficients exact | **CONFIRMED**: 0 / 3 unequal. |
| R196-P3 current `vn_e` first non-bit | **CONFIRMED**: 16,506 cells before both exact older histories. |
| R196-P4 one substitution closes `va_e` | **CONFIRMED**: current `vn_e` alone closes 15,943 / 15,943 output cells. |
| R196-P5 controls bind | **CONFIRMED**: registry, coefficient-bit, rotation-map and missing-stream plants refuse. |

## Validation and choices

The four known-answer plants refuse as required (transcript SHA256
`80db744ec1eb3cebc41b694d40d6b2a501a803636c9d6fd00a3ac94f772593e1`).
The pre-measurement focused battery passed 37/37 in 1.52 s.

The final focused set (rounds 146, 194, 195, 196 and the citation gate)
passed 61/61 in 4.28 s (SHA256
`bb08abd64ac34ec575a0475dac5cd6bc12b88020110bac3f3a41355e709553e4`).
The default citation audit passes 274 citations with zero failures and zero
unmapped citations (SHA256
`2afd9263411a6a64400de2043498c858dd5d42e58817fcb566b0acf2ce92f64d`).
This receipt passes five citations with the same zero counts (SHA256
`536b3866d0ff2fd25d2ae5f4f986501655bd679dc5ed171a93105330115033db`).
Shifting the AB3 span by two lines fails as required (SHA256
`0fe3501289f7fc153fefbfa95215b95d90d40dd5b957a4d5f3c9e3b1b88959fe`).

The required single `tests/ocean/fidelity -n 12` battery collected 2,938
tests and reached 88%. It displayed exactly the four registered pre-existing
reds: the SI3 scalar-math provenance gate, the GYRE spread-floor record gate,
the stamp-scope ratchet and the worktree-stamp ratchet. It then stopped
producing progress and was interrupted once; it was not rerun. This is an
incomplete battery, not a green claim. Its transcript SHA256 is
`8126b2f0b8ccc36b3db24c0b3e610be712520f2c95e7f93b0ea87f4e004a68a1`.

The mandated separate `codex exec --sandbox read-only` review exited before
loading the diff because its in-process app-server client could not initialize
on a read-only filesystem. Verdict: **independent review unavailable
in-sandbox** (transcript SHA256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`).

No configuration choice, carried-state change, stabiliser, tolerance, NEMO
source change or executable observer was introduced. ASKED choices: round
195's OPEN source-ordered AB3 input split. UNASKED choices: empty.

## OPEN

Round 197 keeps the same private complete fold/transport arm and walks the
substep-2 completed V producer in compiled order. First replay the vector-form
raw update from recorded `vn_e`, `zv_spg`, `zv_trd`, `zv_frc`, timestep and
mask; then compare the post-`lbc_lnk` V value. Name the first non-bit boundary
before revisiting the 68-cell midpoint-depth partner, the reciprocal, or any
trajectory census. The admitted round-96 record carries those operands and the
post-association target, so no acquisition is needed unless the raw
pre-association target is required and absent.
