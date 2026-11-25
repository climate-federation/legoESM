# ORCA2 round 209 preregistration — OMT-1 vector-form rung

Date: 2026-10-09. Frozen base: `9884a20d9`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round209/`.
Every trajectory number will be labelled either **independent OMT-1** or
**given NEMO's entry OMT-1**. The shipped rung-10 card, sea ice, its selectors,
and its `unmeasured_features` tuple stay unchanged.

Round 208's requested flux-form operand stream refused on
`zv_frc dimensions moved`. Decision 109 and the operator's round-208 handoff
supersede that flux-form walk: the refusal is registered as an instrument
failure, no value from the stream is read, and this round starts OMT-1.

## Frozen source order and one-module edge

OMT-1 differs from admitted OMT-0 only in `namdyn_adv`:
`ln_dynadv_OFF=.true. -> .false.` and
`ln_dynadv_vec=.false. -> .true.`. The compiled selector maps those mutually
exclusive switches to `np_LIN_dyn` and `np_VEC_c2` at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynadv.f90:162-190`; the resolved
vector program is KEG + ZAD + VOR at `:193-201`. The split-explicit solver
then takes the direct vector U/V update at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:655-681`, rather than
OMT-0's depth-weighted flux-form arm at `:681-702`.

The card applies the same single module edge by restoring the instantiated
rung-0 vector settings on top of the OMT-0 card: vector-invariant momentum,
the rung-0 KEG/ZAD/VOR and vertical-momentum selections, EEN total vorticity,
and the already-selected literal second WZV evaluation. Drag, momentum LDF,
tracer advection, and tracer LDF remain OFF exactly as in OMT-0.

The acquisition is copied from the admitted round-203/204 OMT-0 frame
protocol. Its run-control, passive per-stage writer, two-rank layout, twin
calibration, and admission predicates do not change. Only the two deck
selector values, target/build names, and OMT-1 labels change. A two-step smoke
run uses the same physical deck and run-control schema before the ten-step
calibration/twins. The separate 96-step from-rest run records either normal
completion or NEMO's own first `stp_ctl` boundary.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R209-P1 | OMT-1 is exactly one NEMO module edge from OMT-0. | The namelist diff contains only the two mutually-exclusive `namdyn_adv` assignments above; every other non-protocol line is byte-identical. Resolved output selects vector KEG + ZAD + VOR and rejects linear dynamics. | Any other live deck difference or ambiguous selector: **REFUTED**; stop before reading a trajectory. |
| R209-P2 | The copied frame instrument remains passive. | Smoke run passes; both ten-step twins contain 80 self-describing frames; twin payloads are array-identical; instrumented and uninstrumented rank-complete terminal restarts are byte-identical. | Any missing frame, twin bit, terminal byte, malformed header, or non-finite payload: **REFUTED**; no ladder number is citable. |
| R209-P3 | The explicit OMT-1 card resolves the same vector program. | Instantiated CPU/fp64/libm card prints vector-invariant momentum, rung-0 KEG/ZAD/VOR and WZV settings, while the other four OMT switches remain OFF. The deck/card census agrees field by field. | Any silent default, extra enabled module, or card/deck mismatch: **REFUTED**; stop. |
| R209-P4 | Both labelled kt=1..10 ladders complete and first leave bit identity at kt=1 stage-1 external SSH. | Forty checkpoints / 200 rows per label are finite; active entries are exact; first non-bit is kt=1 stage 1 SSH. Its maximum is expected to return to the vector-rung scale near rung 0's `0.1314585958201272 m`, because round 207 proved the flux-form fold unit moves OMT-0 but moves rung 0 zero rows. | Earlier entry debt or a different first field: **REFUTED**, keep the measured first boundary. A non-finite candidate before NEMO's boundary: stop and name it. |
| R209-P5 | Vector-form momentum advection moves the month boundary later than OMT-0's kt=11 stop. | NEMO reaches at least kt=12; record normal kt=96 completion or the exact later `stp_ctl` step and maxima. | Stop at or before kt=11: **REFUTED**; retain the actual boundary. Missing/ambiguous STOP evidence: **UNMEASURED_WITH_SPEC**. |
| R209-P6 | The gates bind. | Deck-delta, selector, header, field-name, truncation, non-finite, missing-frame, twin-ULP, terminal-byte, changed-binary, card-module, and entry-bit plants each refuse at their named predicate. | Any plant stays green: no claim is citable. |

## Landing predicate

This round lands the explicit OMT-1 card only if the record admits and both
labelled ladders complete under the frozen one-module edge. No shared model
file is expected to change; therefore GYRE/DINO/tank trajectories cannot move.
If the record is unavailable, the committed acquisition is the deliverable
and status is **STOPPED_FOR_RECORD**. The first non-bit internal vector-branch
statement becomes the next source-ordered walk; no OMT-0 flux-form operand is
carried into that claim.

ASKED choices: Decision 109's OMT-1 rung and Decision 103's hierarchy protocol.
UNASKED choices: empty.
