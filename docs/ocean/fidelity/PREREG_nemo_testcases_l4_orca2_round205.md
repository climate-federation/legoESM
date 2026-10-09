# ORCA2 round 205 preregistration — OMT-0 split-explicit statement walk

Date: 2026-10-09. Frozen base: `c46995c4c`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round205/`.
All trajectory results are labelled **independent OMT-0**. The shipped rung-10
ORCA2 card, sea ice, its six selectors and `unmeasured_features` stay unchanged.

## Source order and admitted prior evidence

The round-204 card and both kt=1..10 ladders are admitted prior evidence. Their
first non-bit checkpoint is kt=1 stage 1 SSH (`0.006243153742634906 m` RMS,
`0.13136125371061705 m` maximum). The stage association is not an internal
owner because its external `ssha` operand is already the result of the
split-explicit solve.

The executing record build copies the completed slow forcing at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/dynspg_ts.f90:286-324`, initializes
the cold barotropic histories and live entry at `:339-381`, then executes the
65-substep loop in this order: midpoint extrapolation and face depths
`:458-520`; metric transports and continuity `:524-558`; transport sum and
face SSH `:563-599`; backward SSH, pressure gradient, Coriolis and drag
`:601-653`; flux-form velocity update `:681-703`; seven-array boundary
association `:712-741`; and state rotation/averaging `:781-813`. The
instrument writes the post-association substep state and source operands at
`:742-780` and the final means at `:829-837`.

The existing round-203 frame build is already additions-only calibrated: both
instrumented kt=10 terminal restarts are byte-identical to the uninstrumented
OMT-0 calibration. This round separately admits the `oracle_bt_substeps` and
`oracle_bt_ordered_operands` streams by strict header/EOF parsing and twin
equality before reading any number. Candidate intermediates come only from an
offline call to the existing pure barotropic implementation on the admitted
completed entry state. The replay is usable only if its terminal SSH,
barotropic U/V and transport means reproduce the untraced pure call bit for
bit. No new in-executable observer is permitted.

## Instantiated identity

The pre-measurement Rule-10 print resolves fp64, `forward_euler` + `rk3_ws`,
65 explicit substeps, `nemo_ab3am4`, `nemo_literal` continuity and transport
accumulation, `nemo_ssh_avg` face depth, literal PGF, `een_metric`, flux-form
momentum with both advection arms OFF, and no barotropic drag. Any moved
selection is a refusal, not a fallback.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R205-P1 | The inherited barotropic streams are deterministic and structurally complete. | Twins A/B parse as `BTSUB_2` (65 rows) and `BTORD_2` (2 rows), reach EOF, contain finite fp64 values on every defined slot, and compare array-equal field by field. | Any header, field, finite, EOF or twin mismatch: **REFUTED**; cite no internal number and request a new acquisition only if the existing files are incomplete. |
| R205-P2 | The offline replay is a valid instrument. | Traced and untraced pure barotropic calls from the same OMT-0 state are bit-exact in terminal SSH, barotropic U/V and both transport means; a one-ULP terminal plant fires. | Any unplanted bit move: **REFUTED**, `HELD_INSTRUMENT_NOT_PASSIVE`; read no intermediate claim. |
| R205-P3 | Entry state, cold histories and completed slow forcing are exact on the recorded rank-0 owned slab. | SSH/U/V entry, six histories and SSH/U/V forcing are bit-exact before the substep loop. | The first non-bit input owns the walk; stop before interpreting its consumer and retain this prediction as **REFUTED**. |
| R205-P4 | The first internal debt is in substep 1, after the exact entry family and no later than the post-association U/V exit. | The source-ordered table first leaves the `2e-10` floor at a substep-1 midpoint/depth/continuity/pressure/Coriolis/velocity/association row; all earlier rows are printed. The leading expectation is the ORCA2 fold-sensitive U/V association, absent from flat TSUNAMI. | An exact substep 1 or an earlier entry debt is accepted as the stronger result; name the actual first row and mark this prediction **REFUTED**. |
| R205-P5 | One recorded-operand substitution localizes the first internal statement or proves a cancelling unit. | Replaying the first statement with NEMO's recorded operands closes it bit-exact; if no single operand closes it, the smallest cumulative source-ordered unit and each half are reported. | No closure means the record is insufficient: name the missing operand stream and write an acquisition rather than guessing. |
| R205-P6 | No unrelated model/card behavior moves. | A measurement-only round has no `packages/` diff. A cited fix, if any, lands only after the OMT-0, rung-0, rung-7, GYRE, DINO and tank gates satisfy Decision 96 with every moved row registered. | Any unexplained movement or exact-row loss holds the candidate. |
| R205-P7 | Controls bind to the intended claims. | Header/truncation, twin ULP, source-order, passivity and terminal-ULP plants each refuse at distinct predicates. | Any green or prematurely refused plant invalidates the instrument. |

## Landing predicate

The round may name the first internal statement only after R205-P1/P2/P7
pass. It may land a physics statement only if its oracle-input replay is
bit-exact and Decision 96's complete gate passes. A cancelling unit is scored
atomically; no partial operand lands. OMT-1 does not begin first.

ASKED choices: Decision 103's OMT-0 card and source-ordered walk.
UNASKED choices: empty.
