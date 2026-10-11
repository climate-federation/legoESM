# NEMO SWG level-1 round 3b — non-degenerate production replay

Date: 2026-10-11. Base: `640a4ff8c3fe7f34845d96025da9845fa87c181a`.
Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round3b`.

## Preregistration

This section is frozen before running a round-3b legoESM measurement.  The
operator's independent review reproduced five blockers in round 3.  This fix
round changes only the stage program, its production dispatch, the ladder and
their direct tests and receipts.

Frozen predictions and falsifiers:

1. The admitted `kt=1, stage=1 -> stage=2` boundary will be labelled
   **NON-DISCRIMINATING**, never AT-BAR, because all 13 target fields equal the
   entry fields.  Any fidelity verdict attached to that identity boundary
   **REFUTES** the repaired instrument.
2. The same stage kernel called by normal production `step()` will implement
   all three compiled stages in source order: `Dt/3`, `Dt/2`, `Dt`; C2 then
   one-layer-zero ZAD; total ENS; pressure; continuity from the unchanged
   whole-step before SSH and the current-stage transport; QCO ratios; and the
   before-level masked velocity update.  Stage 3 additionally executes the
   shared NEMO Laplacian, centred wind stress divided by current thickness,
   and `-rn_rfr` times the whole-step before velocity, as compiled at
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:141-367` and
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf_lev.f90:130-150`.  A separate replay
   arithmetic path, an unknown stage/selector fallback, or ordinary `step()`
   rejecting `swe_rk3` **REFUTES** the production repair.
3. Given-entry rows will start at nonzero `kt=2, stage=1` and cover every
   available stage transition through `kt=10, stage=2 -> stage=3`; stage-3
   transitions will be scored through `kt=9 -> kt=10`.  Every output field is
   predicted AT-BAR at `1e-15`; any unequal wet value is DEBT, and the first
   source-order statement producing it will be named.  Missing computable rows
   or relabelling DEBT as AT-BAR **REFUTES** coverage.
4. A chained replay seeded only from admitted `kt=2, stage=1` will run stages
   1, 2 and 3 through the remaining record and score every boundary.  It may
   expose accumulated DEBT, but its first-over-bar boundary may not precede the
   given-entry boundary.  An unreported divergence or a chain that rereads the
   oracle between stages **REFUTES** the chain.
5. Separate zero and one controls will perturb a wet, dynamically consumed
   `u_k` stage-entry input by exactly one `nextafter` step before JIT.  The
   scored `u_k` output must change and the control must exit nonzero.  Touching
   a pass-through field, scoring the input copy, a no-op ULP, or either control
   staying bit-identical **REFUTES** the controls.
6. A nonzero manufactured whole-stage test will exercise C2, ENS, pressure,
   continuity, QCO and U/V masking together and compare against a separately
   written source-statement oracle.  Each named contribution must be nonzero,
   and deleting or changing any one named term must fail the test.  A
   pressure-only case or a test assembled from the production helper
   **REFUTES** the manufactured oracle.
7. The round-3 preregistration's three post-measurement citation edits will be
   reverted byte-for-byte to commit `edcc0c9c1`; their corrected anchors will
   appear only in an ERRATUM below.  Any further mutation of that frozen section
   **REFUTES** provenance.
8. No other card selects `swe_rk3`.  Existing certified rows/hashes must remain
   unchanged, and no first-over-bar boundary may move earlier.  Otherwise Rule
   12 eligibility is **REFUTED**.

No configuration or carried-state choice is introduced.  The three stages,
forcing values, viscosity, drag, gravity, masks and timestep are all already
fixed by the shipped deck and the round-1 resolved card.

## ERRATUM to the frozen round-3 preregistration

Round 3 improperly edited three citation ranges after measurement.  The
historical text is restored to `dynadv.f90:134-139`, `domqco.f90:207-234`, and
`stprk3.f90:170-176`.  The precise compiled statement anchors are respectively
`dynadv.f90:134-138`, `domqco.f90:208-234`, and `stprk3.f90:172-176`; this
erratum records the correction without rewriting the preregistration again.

## Measurement

Preregistration commit: `b6c136dce2d6663985bb1b790297a743e64b0ca6`.
Clean measured implementation: `77cbea57e0668d91962d6914dff0a74e797a126e`.

1. **CONFIRMED.**  The former kt=1 stage-1 verdict is withdrawn.  Its entry
   and target are identical on all 13 fields, so the ladder emits one
   `NON-DISCRIMINATING` boundary and no fidelity rows for it.
2. **CONFIRMED.**  One shared, JIT-compiled stage kernel implements stages 1,
   2 and 3, and ordinary `step()` calls that kernel three times.  Unknown
   stages/selectors and absent stage-2/3 whole-step-before entries raise.  A
   direct reverse-mode test crosses ordinary `step()`.  Gravity is the card's
   `0.02`, drag is the card's `1e-7`, and no SWG feature remains in
   `SWG_UNMEASURED`.
3. **REFUTED at stage 3.**  The given-entry arm scores 26 nondegenerate
   transitions and 338 field rows.  It has 306 bit-identical AT-BAR rows and
   32 DEBT rows.  Every kt=2..10 stage-1 and stage-2 row is bit-identical.
   The first unequal row is `given.kt2.s3.u_k`: 6320 unequal wet U values,
   maximum absolute difference `1.7267697688394534e-13`.  Later stage-3
   rows reach `1.9599059363287208e-09`.  Artifact
   `swg_stages_given.json`, SHA-256
   `9ab1d9722c4a090a0c7a703736ec21f8286d0a61c76e73186b61157bd5c4792b`.
4. **CONFIRMED.**  The chain reads the oracle only once, at kt=2 stage 1.  It
   scores the same 26 transitions/338 rows, with 127 bit-identical rows and
   211 propagated DEBT rows.  Its first debt is the same kt=2 stage-3 U row,
   never earlier; maximum absolute difference through the chain is
   `2.639979230778522e-07`.  Artifact `swg_stages_chained.json`, SHA-256
   `f673f6285733a9a1c73ebed7a8fc40ad03faad0b8a0570188fc89829ed6fd5db`.
5. **PARTLY REFUTED.**  Both controls perturb a wet, consumed `u_k` before
   JIT and score the production `u_k` result.  At one, one output U value
   changes by `6.938893903907228e-18`, giving DEBT and exit 1.  At exact zero,
   the `5e-324` input is flushed by CPU/XLA continuity arithmetic and the
   output remains bit-identical.  The instrument reports `REFUTED`, never
   AT-BAR.  Artifact SHA-256 values are respectively
   `d00dcd86e66f5039999fdf43db154cc1e84d9ab921a3a4aa05f65269051a757e`
   and `8e15869914940e09d08d4c23b4f604bb1d4bb0034f75fa6e1f032e1a374b7c0e`.
6. **CONFIRMED.**  The manufactured whole-stage test uses separately written
   scalar NEMO statements.  C2 U/V, ENS U/V, pressure U/V and continuity are
   each asserted nonzero; velocity, SSH and QCO-T results match, dry U/V faces
   remain zero, and reverse-mode remains finite.
7. **CONFIRMED.**  The three round-3 preregistration strings are restored
   byte-for-byte; their corrections live only in the ERRATUM above.
8. **CONFIRMED for the landed exact rows.**  No non-SWG card selects
   `swe_rk3`; shared QCO and ENS defaults retain their prior arithmetic.  The
   known certified-card actual digests remain TSUNAMI `8bb5ed6c1bc031bd`,
   VORTEX `f7489467bc156303`, VORTEX_VEC `5041b91c67bd1443`, GYRE
   `f02d2f42a570f413`, LOCK_EXCHANGE `d5dfdf83b3a1e0ab`, and OVERFLOW
   `83de0010e1170032`.

The independent normal-step ladder starts from legoESM's card, generates the
compiled analytic wind formula without reading an oracle trajectory, and
scores ten restart after-states.  kt=1 is 5/5 bit-identical.  The first debt is
`independent.kt2.un`, at the same stage-3 frontier, with maximum absolute
difference `1.7267697688394534e-13`; the ten-step arm has 24/50 bit-identical
rows and a worst difference of `3.1954825074897997e-07`.  Artifact
`swg_independent_kt10.json`, SHA-256
`504793c904791778fd6329cf641e60be183616a9b3b46a6424750c8894b74693`.

## Review findings and fixes

The independent round-3 verdict was:

> DO NOT SHIP — The AT-BAR evidence and plants are non-discriminating, and the preregistered nonzero whole-stage falsifier was not performed.

Finding 1 is fixed by withdrawing the identity-map verdict and beginning both
given and chained scoring at nonzero kt=2.  Finding 2 is fixed at ordinary one
but remains honestly REFUTED at exact zero because the smallest binary64
subnormal is flushed in production arithmetic.  Finding 3 is fixed by the
nonzero scalar-statement whole-stage test.  Finding 4 is fixed by routing
ordinary production `step()` through the same stage kernel.  Finding 5 is
fixed by restoring the frozen preregistration and recording corrected anchors
only in the ERRATUM.

## First non-bit statement

Given-entry stages 1 and 2 are bit-identical through kt=10.  The first non-bit
boundary is kt=2 stage 3.  The first newly executed source statement on that
boundary is the `dyn_ldf` call at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:317-317`, whose one-layer Laplacian
body is `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf_lev.f90:130-150`.  The current
record discriminates the stage boundary, not its internal terms, so this is
the first source-order *candidate* and not an unsupported attribution: the
same stage subsequently adds centred wind and before-velocity drag at
`SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:319-332`.

## Rule 12 disposition

The 306 given-entry AT-BAR rows do not move, and the chain's first-over-bar
boundary is identical to the given-entry boundary.  The shared live-F helper
adds explicit opt-ins for SWE's unmasked `(1+r3f)` division and source-
associated QCO evaluation; every existing caller retains the old defaults.
The normal-step dispatch is unreachable unless a card selects `swe_rk3`, and
only SWG does.

The certified-card digest test reproduces the same three already documented
stale expected pins for GYRE/LOCK_EXCHANGE/OVERFLOW while printing their
unchanged actual digests above.  ORCA2's focused card gate passes 4/4.  The
larger combined cross-card processes again hit the repository's documented
per-process compiler-resource boundary after their already completed tests;
no new substantive failure appeared.

## Gates

The focused SWG battery passes 166/166.  The ENS direct-division battery passes
12/12; split QCO tests completed eight green cases before the documented
compiler-resource termination.  The geometry gate remains 34/34
bit-identical, SHA-256
`96a48c0807237896258483c07e71b49a7d10a81b340d0a2ee5715be024c11162`;
its signed-y Coriolis, wet-cell and layer-thickness plants each exit 1.  The
given, chained and independent numerical gates exit 1 on reported DEBT.  The
ordinary-value input plant exits 1 on its own changed output; the zero plant
also exits 1 but is explicitly REFUTED rather than credited as a control.

## Independent review

Independent review run by the operator.

## OPEN

1. Isolate and remove the stage-3 source-association debt, beginning with
   `dyn_ldf` before testing the later wind/drag statements.  Stages 1 and 2 are
   the fixed bit-identical control surface.
2. Replace the exact-zero ULP control with a production control that survives
   CPU/XLA's subnormal flush without weakening the frozen claim; do not call
   the current zero arm a control.
3. Only after every given and chained stage row is bit-identical, run the
   30-day rung and then the shipped 172800-step statistics.

No acquisition or configuration/carried-state decision is requested.
