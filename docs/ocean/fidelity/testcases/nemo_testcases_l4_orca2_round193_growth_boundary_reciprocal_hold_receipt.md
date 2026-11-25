# ORCA2 round 193 — growth boundary and V-reciprocal owner (HELD)

Date: 2026-10-09. Incoming tip:
`f79b81c0417779307f18ae920ea5c308cfabc787`. Measurement commits:
`683ef8c340ddae80adbb1a30e5c134e7dfd80945` through
`94b7187be9b2e62866bf32bd480760ffe5da6436`; production restoration:
`72f5ec5460240d9fdd5ffe7b014f30a4df6406fc`.

## Result

**HELD.** Every number below is **independent hierarchy rung 0**: legoESM
starts from its own corrected climatological T/S, zero velocity and zero sea
surface. No given-NEMO-entry or rung-10 number is mixed into the result. The
shipped ORCA2 card, sea ice, all six ice selectors and its
`unmeasured_features` tuple are unchanged.

The corrected independent entry is bit-exact on every active T/S/u/v/ssh
cell. Under round 192's complete private unit, the first greater-than-ten
growth boundary remains kt=1 stage 1: salinity carries the largest error,
3.2847473521544472 PSU at `[j=147,i=49,k=0]`, from the fixed 2e-10 floor, a
ratio of 16423736760.772236. This respects NEMO's external-mode then RK3
stage-1/2/3 order in
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:202-233`.

Offline replay from that exact entry clears every recorded external-mode row
through substep 1 and substep-2 U accumulation. The first non-bit statement is
substep-2 V accumulation in
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:600-608`: NEMO adds
`wgtbtp2 * zhV * r1_e1v` without a compact V mask. legoESM's helper builds the
reciprocal face metric with that extra mask. The difference is confined to 68
northern-fold halo cells, with complete-record maximum 1.674329379614266 and
RMS 0.02681087428733878; the active wet-cell comparison is exact and therefore
would have hidden it.

This statement is not landed alone. The previous accumulator, completed
`zhV`, and scalar weight are independently bit-exact over the complete record.
Replacing only the reciprocal with unmasked `1/e1v` closes all 26,640
completed-accumulator values bit-for-bit. It is the next cancelling partner of
the held fold/transport unit, so Decision 96 must score the enlarged unit
atomically in round 194.

## Frozen measurement

The stage gate admitted all 80 round-90 self-describing rank shards. The pure
external replay admitted both rank-complete round-96 shards exactly once.
Detached stage output never fed the carried state; a duplicate ordinary run
was array-identical in all five fields. The pure barotropic replay's traced and
untraced results were array-identical for ssh, U/V state, U/V external state
and U/V transports.

| boundary / operand | result |
|---|---:|
| independent entry T/S/u/v/ssh unequal active cells | 0 / 0 / 0 / 0 / 0 |
| first growth boundary | kt=1 stage 1 |
| S maximum / location | 3.2847473521544472 PSU / `[147,49,0]` |
| T maximum at that boundary | 0.17740943620440253 K |
| candidate terminal boundary | kt=9 stage 3, live `e3w_int` refusal |
| first source-order replay debt | substep 2 `transport_sum_v` |
| full-record debt | 68 / 26,640 cells; max 1.674329379614266 |
| previous accumulator unequal cells | 0 / 26,640 |
| completed `zhV` unequal cells | 0 / 26,640 |
| `wgtbtp2` value / unequal | 1.0 / 0 |
| unmasked reciprocal replay unequal | 0 / 26,640 |

The stage artifact is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round193/stage_growth.json`
(SHA256 `a25b92b470ba0c00cd4d49778204c9c38ea832aa3a93df857f88ca97087a2a29`;
log `9e6467f27ca9ad0a317d81b20d1d74d99614b519eeb1d46239fa3d1da369b0f9`).
The final replay is `external_replay.json` in the same directory (SHA256
`eca25b2505522c19c73025fe9b80a4b38d7e03599c7ecfcf5253668f82b06ace`;
log `971487ad0bb1d65183cb055d07123a55057b81adad14c2f23086a74a1c72f694`).

## Instrument correction and retraction

The first replay artifact, retained as
`external_replay_refused_preassociation.json`, is not evidence. It compared a
pre-association reconstructed inverse V depth with NEMO's post-association
record. NEMO performs the seven-array boundary association at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:770-779`, then records the
associated inverse at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:791-796`. The corrected
trace selects the same post-association tuple the ordinary executable consumes;
a known-answer raw/post selector plant fires. The correction changes no
trajectory value. The intermediate `external_replay_pre_split.json` is also
superseded by the final operand-split artifact.

## Prediction ledger

| prediction | verdict | evidence |
|---|---|---|
| R193-P1 corrected entry exact | CONFIRMED | zero active unequal cells in five fields |
| R193-P2 private unit prerequisites exact | CONFIRMED | local arithmetic, dependencies and private-arm identity pass |
| R193-P3 first >10x boundary kt=1 stage 1 | CONFIRMED | ratio 1.6423736760772236e10 |
| R193-P4 first replay debt downstream of external mode | **REFUTED** | external substep-2 V accumulation is first |
| R193-P5 round-175 record sufficient | CONFIRMED BY EARLIER RECORD | the admitted round-96 external record contains the earlier required boundary; no acquisition needed |
| R193-P6 measurement-only unless the complete unit passes | CONFIRMED | final production package tree equals incoming tip |
| R193-P7 unmasked reciprocal closes accumulator | CONFIRMED | 0 / 26,640 unequal, array-equal |

P5's named round-175 record was not needed: source order stopped inside the
external mode, and the already-admitted rank-complete round-96 record is the
earlier and sufficient stream. No stage-operator claim is made.

## Controls and validation

The two gates classify frozen artifacts and carry non-vacuous plants for entry
identity, record admission/bit integrity, passivity, source order,
first-boundary selection, private-arm identity and a one-ULP perturbation.
The focused round-193 battery passed 17/17 tests in 0.60 s.

The default citation audit passed with 274 citations, zero failures, zero
unmapped citations and zero failing map entries (artifact SHA256
`b97f58bf6481aa5f213bec03a29ae861d1999afcf5d1b5b77552b9fcecca28bb`).
This receipt passed with four citations and the same zero counts (SHA256
`373872e9b66d3b4615b7cf6fd46cb4a51672e76d0d8c726ae1eb79c9133a2f80`).
Shifting the owning `dynspg_ts` citation by two lines failed on its first
endpoint as required (exit 1; SHA256
`298d947d738177f6b7ca8731ecc264e718e2b4f2f89cc3aca3f87114a998e3f8`).

The required single `tests/ocean/fidelity -n 12` battery collected 2,917 tests
and reached 98%. It exposed one known pre-existing red,
`test_nemo_testcase_l2_gyre_round129_spread_floor_gate.py::test_record_backed_gate_passes`,
then made no progress for an extended interval and was interrupted (exit 130).
It was not rerun. Both round-193 files had already completed green inside that
battery as well as in the focused run. This is an incomplete campaign-wide
battery, not a green claim and not a round-193 regression.

The mandated separate `codex exec --sandbox read-only` review exited before
loading the diff because its in-process app-server client could not initialize
on a read-only filesystem. Its recorded verdict is **independent review
unavailable in-sandbox** (log SHA256
`b363a58514134bb920e7f2c4aaf1d95b1d1558748489689e05722739452baffa`).

The final tree has no model or production-test difference from the incoming
tip. Consequently there is no retained shared implementation to score against
GYRE, DINO, tanks, rung 7 or the ORCA2 ladders in this held round. The private
unit remains an arm, not product behavior.

## OPEN

Round 194 must preregister one atomic arm: round 192's complete fold/transport
unit plus NEMO's unmasked V reciprocal metric in the external-mode
accumulation. First prove every reciprocal and accumulator exact through all
65 substeps; then rerun the corrected-entry rung-0 growth and 200-row ladder.
Only if that complete unit is a Decision-96 net improvement may it proceed to
the rung-7 ladder, independent month boundary, GYRE, DINO and tank gates. No
partial operand may land. No acquisition is needed: the admitted round-96
record contains the required accumulator operands.
