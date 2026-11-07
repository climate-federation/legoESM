# ORCA2 round 93 — rung-0 stage-1 RHS walk

**Verdict: STOPPED_FOR_RECORD.** On the independent rung-0 entry, legoESM's
production-JIT stage-1 HPG, LDF, VOR, KEG, and ZAD cumulative momentum RHS is
bit-identical to NEMO in every active U/V cell. The first non-bit statement is
therefore later than ZAD. The admitted record ends there, so this round does
not infer an owner from the already non-bit completed stage. A committed,
additions-only acquisition requests the next source-ordered boundaries. No
`packages/` file, card selector, configuration choice, stabilizer, carried
state, threshold, or sea-ice debt changed.

Base: `da33d419a`. Preregistration:
`PREREG_nemo_testcases_l4_orca2_round93.md` at `995ef7a965`. All measurements
below are labelled **independent**: the state is NEMO rung 0's from-rest entry,
not NEMO's recorded shipped-card entry.

## Admitted input

The operator-run round-92 acquisition re-admits with
`STATUS PASS_R92_RHS_ADMISSION`. Its two self-describing rank files cover the
148x180 domain exactly once, carry all ten requested arrays, and preserve all
20 terminal ocean restart shards byte for byte. Across the ten arrays, all
266,400 values in NEMO's unused `jpk` slot are exact positive zero (zero
nonzero values and zero negative zeros).

The independent bridge remains exact at the stage-0 entry:

| Field | Compared values | Unequal bits | max abs | rms |
|---|---:|---:|---:|---:|
| T | 799,200 | 0 | 0 | 0 |
| S | 799,200 | 0 | 0 | 0 |
| u | 799,200 | 0 | 0 | 0 |
| v | 799,200 | 0 | 0 | 0 |
| ssh | 26,640 | 0 | 0 | 0 |

## Source-ordered result

The compiled vector-invariant program calls HPG, LDF, VOR, KEG, and ZAD in
that order at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:141-175`. The round-93 gate
captures the existing legoESM production tendency components during the first
JIT stage, accumulates them in that order, and separately proves that the
capture leaves final T/S/u/v/ssh bit-identical to an ordinary production step.

| NEMO cumulative boundary | U active cells unequal / compared | V active cells unequal / compared | max abs | rms |
|---|---:|---:|---:|---:|
| after HPG | 0 / 411,276 | 0 / 413,856 | 0 | 0 |
| after LDF | 0 / 411,276 | 0 / 413,856 | 0 | 0 |
| after VOR | 0 / 411,276 | 0 / 413,856 | 0 | 0 |
| after KEG | 0 / 411,276 | 0 / 413,856 | 0 | 0 |
| after ZAD | 0 / 411,276 | 0 / 413,856 | 0 | 0 |

This is not a zero-against-zero comparison. NEMO HPG reaches
8.557700048658789e-5 m/s2 on U and 1.1710938588096454e-4 m/s2 on V. NEMO's
after-LDF payload happens to equal its after-HPG payload exactly at this entry;
the gate records that fact rather than requiring a fabricated movement.

The table's scope is the model's active face masks. Outside those masks, the
record's owned canonical arrays and legoESM's internal padding are not a
physical-value identity claim: after HPG they differ in 386,170 U and 385,344
V full-domain slots, and the later frames differ in 97,360 U and 95,904 V
slots. These excluded land/halo slots are reported rather than silently called
bit-exact. They cannot own the active stage-1 trajectory.

R93-P1 is **REFUTED**: HPG is exact. Continuing in compiled order finds LDF,
VOR, KEG, and ZAD exact too. R93-P2 is **CONFIRMED** because selection follows
compiled boundary order rather than residual magnitude. R93-P3 is
**CONFIRMED** by 0 / 3,223,440 trace-induced state differences. R93-P4 is
**CONFIRMED** for the admitted cumulative boundaries and stops at the first
missing rank-complete boundary.

## First unmeasured statement and acquisition

The next executed statement is the vector-invariant vertical average of the
now-exact 3-D RHS at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215`. NEMO then adds
baroclinic drag and returns its drag coefficients
(`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:227-230`), adds wind stress
(`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:232-245`), constructs the sea
surface RHS (`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:283-300`), and
calls the split-explicit solver
(`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:305-315`). The resolved rung-0
deck leaves atmospheric pressure, embedded ice load, and wave load off; their
guarded branches remain visible at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:247-277` and are not collapsed
into a guessed zero operand.

The committed round-93 acquisition records, on both ranks at kt=1, the
depth-averaged U/V RHS, post-drag U/V RHS and CdU coefficients, post-wind and
final U/V RHS, ssh RHS, post-solver ssh, and post-solver barotropic U/V. Its
self-describing header declares every name and payload length; admission
requires exactly-once rank coverage and byte identity for all 20 terminal
restart shards against the admitted round-92 parent. The patch adds calls only,
applies at fuzz zero, and compiles against the pinned build modules. Its source
preflight passes and its shifted-rank layout plant refuses. The operator must
run `nemo_testcase_l4_orca2_round93_slow_acquisition/run.sh --run`.

## Controls, tests, and review

The real measurement reports `STATUS MEASURED_R93_RHS_WALK`. Its rank-layout,
nonzero-`jpk`, and trace-passivity plants all refuse, and a one-ULP synthetic
active-cell residual is counted as exactly one unequal cell. Instrument work
also refused before measurement when a full-model trace incorrectly required
inactive TKE operands, when an unsupported model API keyword was supplied, and
when face masks were assumed three-dimensional; none produced a model number.
The final gate narrows capture to existing production tendency components and
uses the card gate's established face masks.

The focused round-93, rung-0-card, and citation-gate battery passes 25/25. The
round receipt citation gate passes all seven citations and the default gate
passes all 274 citations, both with zero failures and zero unmapped entries;
shifting the vertical-average citation by two lines makes the gate fail.

The prescribed single `tests/ocean/fidelity -n 12` run selected 2,233 tests,
displayed six failures, and reached 96%, then reproduced the established
silent-tail stall. It was stopped after three consecutive empty 30-second
waits following its last progress. Because xdist emitted no failure summary
before the interrupt, this receipt does not assign names to those six visible
failures and makes no full-suite PASS claim. The round-93 focused tests had
already passed both alone and within that battery.

The required separate read-only review failed before reading the diff:
`failed to initialize in-process app-server client: Read-only file system`.
Verdict: **independent review unavailable in-sandbox**. A local `git diff
--check` is clean; it is not represented as independent review.

## OPEN

1. Operator runs the committed slow-boundary acquisition and admits its two
   rank records plus terminal-restart identity.
2. Compare the vertical average first, then drag, wind, ssh RHS, and the
   split-explicit output in compiled order; the first active non-bit boundary
   owns the next walk.
3. Only after a numerical statement is isolated may a model change land. The
   rung-0 package card, given-entry and independent ten-step ladders, and the
   independent month remain subsequent work.

## UNVERIFIED

- The first active non-bit statement after ZAD is not yet measured.
- The exact implicit linear-drag composition remains unbuilt.
- No rung-0 card is exposed through the package dispatcher.
- Both ten-step ladders and the independent month remain unmeasured.
