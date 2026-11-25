# ORCA2 round 170 — kt=8 slow-forcing RHS boundary

Date: 2026-10-08. Base `665afb4ce`; preregistration `4d9998522`; final
measurement instrument `34fdfd35f`. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round170/`.
Verdict: **STOPPED_FOR_RECORD**. The U-face reference mesh and NEMO's literal
depth reduction are bit-exact; the first upstream U boundary is the completed
three-dimensional momentum RHS, already finite but explosive at
`1.5835360371918837e51 m s^-2`. The admitted record does not split that kt=8
RHS by HPG/LDF/VOR/KEG/ZAD, so no owning physics statement is named yet. No
physics or configuration changed.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into the result. Sea ice, all six
sea-ice selectors and the shipped card's `unmeasured_features` tuple remain
unchanged.

## Record and compiled order

The operator's round-169 acquisition admits with two self-describing rank
records, exactly-once 148 x 180 coverage, all 23 registered fields and 20
terminal restarts byte-identical to the round-166 baseline. The compiled
program accumulates HPG, LDF, VOR, KEG and ZAD in that order at
`ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:145-179`, then records the
completed three-dimensional RHS, reference face thickness and masks at
`ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:190-196`.

For the active vector-invariant branch, NEMO performs the literal reference-
thickness reduction at
`ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:215-227`. Drag and wind
follow at `ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:244-274`, and the
three possible load branches follow at
`ORCA2_OMIP_L4_R169SLOW8/BLD/ppsrc/nemo/stp2d.f90:278-306`.

The instrument needed three fail-closed corrections before it produced a
report: the admitted frame producer had to be routed separately from the new
slow record; the record's structural, non-contributing `jpk` slot had to be
appended only for comparisons; and the reference-mesh helper was already in
native-face layout. Two later arm-construction refusals established that the
private RHS hook accepts the native 148 x 180 x 30 slab directly. Every
refusal log is retained. The final run is stamped to clean commit
`34fdfd35fc6e5381688945d7269e88f2c820f567`, production JIT, CPU, fp64/x64
and libm.

## U-face result

The U-face result is source ordered:

| boundary or operand | active unequal cells | candidate non-finite | maximum absolute difference |
|---|---:|---:|---:|
| reference `e3u` | 0 | 0 | 0 m |
| completed 3-D RHS | 413,030 | 0 | 1.5835360371918837e51 m s^-2 |
| `umask` | 0 | 0 | 0 |
| reference reciprocal depth | 0 | 0 | 0 m^-1 |
| reduced slow forcing | 15,853 | 0 | 1.5216884364408584e51 m s^-2 |

Replaying NEMO's recorded `SUM(e3u*rhs_u*umask)*r1_hu_0` reproduces the
recorded active U boundary with 0 unequal cells. More importantly, replacing
only legoESM's completed 3-D U RHS by NEMO's recorded RHS and running the
production reduction also gives 0/15,853 unequal active U faces, with 0
full-domain differences. Thus the reduction is **DISCHARGED** for U and the
completed 3-D RHS is the first unresolved upstream boundary. It is not yet an
owning statement: the current record sees only the value after ZAD.

## V-face refutation retained

R170-P3 predicted that the same one-variable RHS arm would close both faces.
It is **REFUTED and retained**. V still contains the already-held north-fold
unit's operand debts:

| V operand or arm | active unequal cells | maximum absolute difference |
|---|---:|---:|
| reference `e3v` | 27 | 437.2169154512344 m |
| `vmask` (full operand domain) | 1,319 | 1 |
| reference reciprocal depth | 68 | 0.03332976059679253 m^-1 |
| recorded-RHS-only production reduction | 68 | 7.5211445529092786e-6 m s^-2 |

NEMO's own recorded V operands replay its active reduction exactly (0
unequal), so this is not a record-arithmetic ambiguity. The result does not
weaken the U attribution and does not promote the adverse complete halo unit:
that unit remains private and HELD.

R170-P1 and R170-P2 are **CONFIRMED** by the admitted record and the two exact
recorded replays. R170-P4 is **CONFIRMED**: the instantiated card has implicit
surface-stress placement but the rung-0 stress arrays are exactly zero;
NEMO's resolved atmospheric-pressure and embedded-ice switches are false,
the deck's Bernoulli-wave switch is false, and recorded drag, wind and final
boundaries are array-identical for U and V. The card still declares
`linear_implicit_bottom_drag` unmeasured and legoESM's rung-0 private card
still has `barotropic_drag_substep=False`; this round makes no drag-fidelity
claim. R170-P5 is **CONFIRMED**: no package, card, selector, forcing,
carried-state, stabiliser or sea-ice change landed.

## Acquisition and controls

The committed launcher is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round170_rhs8_acquisition/run.sh`.
It creates the new `ORCA2_OMIP_L4_R170RHS8` target and writes one
self-describing kt=8 record per rank at each already-compiled accumulator
boundary: after HPG, LDF, VOR, KEG and ZAD. It reuses the existing write-only
call sites and changes only the recorder's selected step, magic and filename.
The checker parses every field header and payload length, demands exactly-once
rank coverage, and compares all 20 terminal restarts byte-for-byte against the
admitted round-169 run. Producer identity is content-pinned from the committed
launcher, writer, checker and preregistration rather than a moving commit SHA.

The clean-tree preflight reports `SYNTAX_PROOF_PASS` and
`ORCA2_ROUND170_RHS8_PREFLIGHT_READY`. Its planted writer-layout violation
exits 69 with `STATUS PLANT-FIRED layout`. The measurement gate independently
plants rank placement, literal replay, RHS-arm coverage, source order and
zero-wind identity.

## Review and validation

Focused round-170 and citation coverage passes 26/26. The round receipt's five
citations pass with zero failures or unmapped spans; shifting the literal
vertical-reduction span by two lines makes the citation plant fail. The
cumulative receipt gate also passes with zero failures and zero unmapped
spans. The acquisition preflight and its layout plant pass as recorded above.

The one required `tests/ocean/fidelity -n 12` invocation collected 2,735
tests and reached 99%, then stopped emitting with no pytest process remaining
to poll; it was not run a second time. Its retained log contains 2,724
terminal outcomes: 2,713 passed, 7 skipped and 4 failed. The four failures are
the registered pre-existing reds from rounds 166-168: the moved GYRE spread
record, unscoped allow-dirty drivers, unstamped report emitters and SI3
scalar-math provenance. No round-170 test failed; the log contains no symbol-
materialisation error, `MemoryError`, worker crash or pytest error.

The separate `codex exec --sandbox read-only` review attempt returned
**independent review unavailable in-sandbox** before reading the diff:
`failed to initialize in-process app-server client: Read-only file system`.

Because this round changes no model or configuration file, no ORCA2 ladder,
GYRE trajectory/year, DINO or tank trajectory can move; their rerun predicates
are therefore not invoked for this measurement-only stop.

ASKED choices: continue the compiled-source independent rung-0 walk. UNASKED
choices: empty. No configuration decision is requested.

## OPEN

1. The operator runs the round-170 RHS8 launcher. Admit both rank records and
   the 20 byte-identical terminal restarts.
2. At kt=8, compare the accumulated U RHS after HPG, LDF, VOR, KEG and ZAD in
   compiled order. The first finite-scale to explosive or first non-bit
   boundary owns the next walk.
3. Keep the V halo/mask unit and the round-169 1/41 face-average residual as
   separate HELD debts. If a cited upstream statement closes the explosive U
   chain, re-test the complete unit atomically under Decision 96 and every
   shared gate before landing anything.
