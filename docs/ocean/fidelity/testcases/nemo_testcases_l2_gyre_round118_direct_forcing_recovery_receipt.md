# NEMO-testcases L2 GYRE round 118 receipt: direct forcing recovery

Date: 2026-09-19

Incoming lane tip: `c8f5d513d`

Preregistration commit:
`54f24e428`

Authoritative measurement commit:
`8a8fe548c5848f1a6f6f917fefa82a0a2b31ae17`

Round status: **HELD — the existing Round-117 record was recovered and the
complete recorded NEMO producer chain closes BIT, but the first model
cumulative mismatch remains unowned because an isolated NEMO-order
accumulator does not close BIT to the production-JIT live total; no physics,
configuration, or trajectory changed**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round118/`

## Outcome first

The operator's 112,560-byte pre-loop record is complete and usable.  Round
117's 127,408-byte expectation had reversed the compiled extents: the record
contains six full-domain arrays and twelve owned-domain arrays.  The compiled
writer lists all 18 fields, then writes direct Coriolis and final forcing at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynspg_ts.f90:329-345`.
Round 118 retracts the old arithmetic in the Round-117 receipt and tool,
parses physical EOF, and admits the existing run without invoking `makenemo`
or `mpirun` again.

The authoritative production-JIT walk directly scores NEMO's incoming,
pre-loop Kmm Coriolis, and final slow forcing.  The first direct non-bit row is
incoming U: 580/580 wet faces at
`1.0529650291768787e-11 m s-2`; incoming V is 570/570 at
`1.076555991792179e-11 m s-2`.  Coriolis is not BIT as preregistered: U is
457/580 at `1.3234889800848443e-23`, and V is 425/570 at
`1.1166938269465874e-23`.  It is downstream and negligible.  The final
subtract is BIT given the model's own inputs, and incoming/Coriolis
cancellation is only `1.35e-12` U and `1.33e-12` V by the registered L1
metric.

The same-run NEMO input chain closes: the inherited Round-46 `after_adv`
snapshot equals the Round-117 direct `Krhs` BIT, and NEMO's vertical mean,
drag, wind, and pre-loop subtract replay BIT for both faces.  Those executing
statements are the vector-form vertical mean at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90:222-230`, drag and
wind at `GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90:246-259`,
and the copy/Coriolis/subtract sequence at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynspg_ts.f90:290-345`.
Thus the `1e-11` incoming error is inherited from the 3-D momentum RHS; it is
not born in vertical averaging, drag, wind, or the external pre-loop
subtract.

The first non-bit statement boundary in the compiled 3-D source order is the
shared accumulator immediately after `CALL dyn_ldf(...,Krhs)` and its writer
at `GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90:147-179`:
HPG is BIT, then the after-LDF row differs everywhere wet at
`2.5292467120726215e-14` U and `3.502735092670824e-14` V.  This names the
first boundary, not an internal LDF owner.  The largest incremental mismatch
is ZAD, `1.9220297482797664e-9` U and `1.966061294804274e-9` V, approximately
five orders larger than LDF.

Critically, the isolated NEMO-order accumulator still differs from the
production-JIT live RHS in 6,882/17,400 U and 6,566/17,100 V cells at
`8.470329472543003e-22`.  Therefore the Round-117 falsifier fires: an isolated
reconstruction cannot prove a production statement owner.  No LDF or ZAD
patch is eligible, no Decision-43 ladder/month arm is spent, and the round is
HELD.

## Frozen prediction ledger

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round118.md`.

| frozen prediction or falsifier | disposition | evidence |
|---|---|---|
| compiled record is exactly 112,560 bytes with header `(1,2,3,36,26,64,18,704)` and physical EOF | **CONFIRMED** | Six full plus twelve owned arrays parse; record SHA-256 is `259fcbb042ce0aa7a28f7db75b746ca56dce2ea2a62e960a6a265f58f691c176`. |
| no-build recovery verifies source, producer tools, toolchain, both executable copies, ten steps and `STOP 0` before stamping | **CONFIRMED** | Recovery ends in `ROUND117_PRELOOP_RECORDS_READY`; executable SHA-256 is `b5c750863e71af3c23e958794ecb24dfce3c36faa57ba6247026e081b7cc497e`. |
| incoming and final rows reproduce 580/570 and the prior final maxima; direct Coriolis is BIT | **REFUTED** | Counts reproduce. U incoming reproduces the prior maximum, but V incoming is `1.076555991792179e-11`, not final's `1.0765559917925099e-11`; Coriolis is 457/425 non-bit at about `1e-23`. |
| incoming is the first direct non-bit boundary and there is no material incoming/Coriolis cancellation | **CONFIRMED** | Incoming precedes Coriolis in the compiled program; registered cancellation fractions are about `1.3e-12`. |
| same-run `after_adv -> depth -> drag -> wind -> final` replay is BIT | **CONFIRMED after instrument repair** | Every U/V row is zero unequal.  The first run exposed the retained Round-28 30-vs-31-level probe defect; preserving the non-contributing `jpk` slot repairs it and has a focused regression. |
| HPG is BIT; after LDF is first; ZAD is the largest incremental mismatch; live-total closure remains 6,882/6,566 at `8.470329472543003e-22` | **CONFIRMED** | Production-JIT reproduces every frozen value exactly. |
| no source-exact candidate exists if live-total closure is non-bit | **CONFIRMED** | `candidate_eligible=false`; no production diff or after arm exists. |

## Acquisition recovery and admission

The operator's run reached NEMO `STOP 0` for ten steps and produced the record,
then the old shell size assertion exited 69.  Reading the compiled statement,
not guessing from the receipt, gives this payload:

| class | fields | extent | bytes |
|---|---:|---:|---:|
| header | magic plus eight integers | fixed | 48 |
| full | Kmm U/V, U/V mask, Coriolis U/V | `6 x 36 x 26` fp64 | 44,928 |
| owned | incoming U/V, eight ENE coefficients, final U/V | `12 x 32 x 22` fp64 | 67,584 |
| total | 18 arrays plus header | exact physical EOF | **112,560** |

The recovery path uses the existing target and executable.  It compares the
original source manifest, uses the producer commit's versions of every
repository-owned admission helper, verifies external toolchain inputs, and
requires the built and copied executable hashes to match the recorded binary
manifest.  It does not contain a recovery call to `makenemo` or `mpirun`.

Admission rows are all BIT over the full 704 owned cells:

| duplicate/replay boundary | U unequal | V unequal |
|---|---:|---:|
| slow post-wind -> pre-loop incoming | 0 | 0 |
| compiled incoming minus direct Coriolis -> final | 0 | 0 |
| pre-loop final -> admitted Round-81 frozen forcing | 0 | 0 |

The clean validation JSON has SHA-256
`aa833eec6256c876e7572e29fd9b39d4c74874c6f005e9fa272d2cfbd74bad02`.
The inherited consumed-field admission passes at 46 exact / 20 intentionally
changed / 132 admitted.

## Production-JIT direct split

The compiled program first copies the completed `Ue_rhs`/`Ve_rhs`, computes
Kmm Coriolis, and subtracts it under the face masks at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynspg_ts.f90:290-345`.
The measured rows follow that order:

| boundary | U unequal / wet; max | V unequal / wet; max |
|---|---:|---:|
| incoming versus direct NEMO | 580/580; `1.0529650291768787e-11` | 570/570; `1.076555991792179e-11` |
| pre-loop Coriolis versus direct NEMO | 457/580; `1.3234889800848443e-23` | 425/570; `1.1166938269465874e-23` |
| final versus direct NEMO | 580/580; `1.0529650291768787e-11` | 570/570; `1.0765559917925099e-11` |
| isolated subtract versus production final | 0/580; `0` | 0/570; `0` |

The Coriolis discrepancy changes the final last bits in some cells but carries
essentially none of its magnitude.  The registered L1 cancellation fractions
are `1.3472556403826275e-12` U and `1.329936161198475e-12` V; this is not a
cancelling pair.

The authoritative artifact is `producer_walk/production_jit.json`, SHA-256
`eb3d5ae2d9e9fd8836b214a1a134f6790ff157038060e96a9c38240ff37ba4e5`.
It stamps a clean worktree at the authoritative commit, fp64/x64/libm on CPU,
and production JIT.  The traced and untraced returned pytrees are BIT over 23
leaves and 198,956 cells, and the traced final pair is BIT to the actual
external-solver call.

## Complete producer chain

The compiled program calls and snapshots HPG, LDF, VOR, KEG, ZAD and ADV in
that order at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/stp2d.f90:147-179`.
Given NEMO's admitted inputs, every downstream producer statement closes:

| boundary | U unequal / wet; max | V unequal / wet; max |
|---|---:|---:|
| inherited Round-46 after-ADV -> same-run Round-117 `Krhs` | 0/17,400; `0` | 0/17,100; `0` |
| literal vertical mean -> direct depth mean | 0/580; `0` | 0/570; `0` |
| drag replay -> direct post-drag | 0/580; `0` | 0/570; `0` |
| wind replay -> direct incoming | 0/580; `0` | 0/570; `0` |
| forward subtract -> direct final | 0/580; `0` | 0/570; `0` |

The first attempted replay incorrectly passed an already trimmed 30-level
array to the legacy literal helper, which itself omits the final
non-contributing level.  That reproduced the known Round-28 probe defect and
gave a false `1.97e-9`/`1.62e-9` row.  The executable gate now preserves the
31st `jpk` slot, the focused regression distinguishes 31 input levels from 30
scored levels, and the false row is retracted rather than interpreted.

## Cumulative 3-D RHS and first statement

Production-JIT full wet-cell rows are:

| compiled boundary | U unequal / wet; max | V unequal / wet; max | incremental residual max U / V |
|---|---:|---:|---:|
| after HPG | 0/17,400; `0` | 0/17,100; `0` | `0` / `0` |
| after LDF | 17,400/17,400; `2.5292467120726215e-14` | 17,100/17,100; `3.502735092670824e-14` | same / same |
| after VOR | 17,400/17,400; `2.5292467332484452e-14` | 17,100/17,100; `3.502735092670824e-14` | `8.470329472543003e-22` / `8.205631676526035e-22` |
| after KEG | 17,400/17,400; `2.5292467332484452e-14` | 17,100/17,100; `3.502735092670824e-14` | `0` / `2.117582368135751e-22` |
| after ZAD | 17,400/17,400; `1.9220276136604423e-9` | 17,100/17,100; `1.966059508480186e-9` | `1.9220297482797664e-9` / `1.966061294804274e-9` |
| after ADV | same as after ZAD | same as after ZAD | `0` / `0` |
| isolated source-order final -> production live total | 6,882/17,400; `8.470329472543003e-22` | 6,566/17,100; `8.470329472543003e-22` | closure, not an oracle row |

The NEMO after-ADV and after-ZAD snapshots are BIT to each other, as required
by the executing vector-form branch.  The first measured non-bit statement
boundary is therefore the `dyn_ldf` accumulator write, but ownership is
**WITHHELD_PRODUCTION_ASSOCIATION**.  The gate uses actual production operands
and an isolated JIT accumulator; those are not the full production closure.
Calling the LDF operator source-exact would repeat the eager/isolated proof
error identified in rounds 91 and 102.

Magnitude ordering is explicit: ZAD's incremental discrepancy is the largest
recorded 3-D contributor, while the final slow-forcing pair carries all of the
Round-117 final-SSH maximum on the prior directed arm.  LDF is first in source
order but not first by magnitude.  Round 119 must resolve the production
association before either label can become a candidate.

## Eager control and plant

Production eager is secondary and never substitutes for production JIT.  It
finds the same first direct and cumulative boundaries and the same BIT NEMO
producer replay.  Its incoming/final U maximum is
`1.0529650291765478e-11`; V is `1.076555991792179e-11`.  Coriolis is 458/424
cells at `1.1580528575742387e-23` / `9.926167350636332e-24`.  Its live-total
closure is 6,859 U and 6,569 V cells at `8.470329472543003e-22`.  This
eager-versus-JIT difference is why the production result is authoritative.
Artifact: `producer_walk/production_eager.json`, SHA-256
`f5e0a5af2e28376893c6ab9a4a3a01b1eef5ca44ea234eb9a2bf800148defd8e`.

The production incoming-U one-ULP plant changes exactly one incoming word at
`[1,1]` by `1.6543612251060553e-24`, leaves Coriolis BIT to the unplanted arm,
and changes the same one final-U word by the same amount.  It prints
`ROUND118 INCOMING-ULP STATUS PLANT-FIRED` and exits 1.  Artifact:
`producer_walk/incoming_ulp_plant.json`, SHA-256
`cacaf956a5ed5c14497c063e3b7a9fd54d57bf354eff5c9dad0555f7928d33cb`.

## Decision-43 trajectory and cross-card scope

No production physics or configuration changed, so there is no after arm.
The current immutable Round-110 landed arm remains the trajectory authority:

| required headline | before | after | disposition |
|---|---:|---:|---|
| kt2 T | `1.4210854715202004e-14 K` | same | AT-BAR, unchanged |
| kt2 S | `2.1316282072803006e-14` | same | AT-BAR, unchanged |
| kt2 U | `2.7377110452773967e-12 m s-1` | same | first-over-bar DEBT, unchanged |
| kt2 V | `3.2849219221489645e-12 m s-1` | same | first-over-bar DEBT, unchanged |
| kt3 T | `8.600419718618468e-7 K` | same | DEBT, unchanged |
| kt3 S | `6.979441735666114e-8` | same | DEBT, unchanged |
| day-30 T RMS | `6.890484901489568e-5 K` | same | no candidate; month not rerun |

Decision 43's month decrease, first-over-bar, kt1, moved-row registry and
recipe-derived executing-card gates are not invoked by an empty production
diff.  The diagnostic changes move no trajectory row.

GYRE-zco and the generic NEMO-GYRE recipe execute this producer path.  DINO
shares the momentum operators implicated by the cumulative walk and therefore
retains explicit risk, but no DINO bits can move from this diagnostic-only
round; a future shared candidate must measure DINO before landing.
LOCK_EXCHANGE and OVERFLOW likewise cannot move.

ORCA2 remains **UNMEASURED-WITH-SPEC**: independently record and admit its
source-ordered 3-D RHS accumulator boundaries, full vertical mean operands,
drag, wind, direct pre-loop incoming/Kmm-Coriolis/final pair, complete
external step, weighted final SSH, next-step entry and local kt3 U/V/W/T/S;
run production JIT, production eager and isolated controls with matching
plants; then run the certified trajectory for any shared candidate.

No restart/checkpoint representation, carried state, selector, coefficient,
timestep, stabilizer, year harness, reconciliation gate, freshwater pair,
#1484 guard, held manifest, or NEMO source changed.

## Review, citations, and tests

The required separate adversarial review was invoked against the committed
Round-118 diff with `codex exec --sandbox read-only -C`.  Independent review
was unavailable in-sandbox: the read-only app-server failed before producing
a verdict.  This is neither approval nor a `SHIP` verdict.  The complete
terminal result is quoted verbatim:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

The process exited 1.  Its complete log is
`round118/review/codex_review.log`; no reviewer marked the diff `DO NOT SHIP`.

The clean-tree citation gate finds five Round-118 citations and ten citations
in the amended Round-117 receipt, with zero unmapped citations, zero symbol
failures, zero map-audit failures, and all built-in controls firing.  The
first diagnostic pass correctly rejected four newly added broad ranges whose
endpoint symbols were inside rather than at the range endpoints; each was
narrowed to the exact compiled statement, not weakened.  Shifting the direct
writer citation by two lines reports `SYMBOL-NOT-AT-LINE` and exits 1.
Artifacts are in `round118/citations/`.

CPU/fp64 pytest summaries are:

```text
20 passed in 0.70s
119 passed in 372.94s (0:06:12)
8 passed in 511.30s (0:08:31)
```

The 20-test focused suite covers every changed Round-51/83/117 gate and all
reader plants.  The 119-test inherited push suite covers receipt citations,
all TKE NEMO terms, NEMO recipe resolution, and real freshwater closure.  The
eight-test serial suite replays every node at which an xdist worker aborted.

The required single-piece
`-n 12 tests/ocean/fidelity tests/ocean/unit` attempt collected 8,201 tests,
then suffered eight fatal JAX worker aborts.  It reached 99%, recorded 7,524
explicit passes and 501 failure labels, and produced no pytest summary or
JUnit file.  After nine minutes with no controller output it was interrupted
once with exit 130 to stay within the round's CPU budget.  The literal terminal
state therefore has no summary line to quote; its last progress marker was:

```text
[gw19] [ 99%] PASSED tests/ocean/unit/test_no_scheme_duplication.py::test_ocean_pe_uses_eos_helper[ocean_pe_latlon_cgrid.py]
```

Diffing the 501 unique labels against the pinned 87-ID incoming baseline gives
67 in baseline, 434 outside baseline, and 20 baseline IDs not observed.  This
is not a trustworthy regression census: xdist labeled large assigned queues
failed after worker aborts.  None of the changed Round-83, Round-117, or
citation tests appears in the failed-ID set; those tests are green in both
clean focused suites.  More decisively, all eight actual crash nodes pass
serially in the third summary above.  The full log, extracted IDs, baseline
intersection/difference, and serial replay are retained in
`round118/tests/`.

## ASKED / UNASKED

ASKED and completed: frozen preregistration before parsing; compiled-writer
layout reconciliation; recovery from the existing run without a NEMO
rebuild/rerun; fail-closed source/binary/toolchain/stamp admission; direct
incoming/Coriolis/final scoring under production JIT and production eager;
incoming/Coriolis cancellation; complete same-run producer replay; explicit
30/31-level probe retraction; first non-bit boundary and largest incremental
owner; no-candidate falsifier; unchanged kt2 U/V, kt3 T/S and day-30 T; DINO
risk and ORCA2 acquisition specification; compiled citation gate and shifted
plant; independent review attempt; focused, push-gate and whole-tree test
attempts with serial crash-node discrimination.

UNASKED and not done: NEMO source was not modified; Codex did not invoke
`makenemo` or `mpirun`; no record was invented; no production physics,
configuration, default, coefficient, stabilizer, carried state, restart
schema, year harness, reconciliation gate, freshwater pair, #1484 guard, or
held patch changed; no isolated closure is called a production proof.

## OPEN for round 119

1. Stay in the existing shared Round-83 producer gate.  Capture the actual
   production accumulator at every stage-1 routine barrier under the full
   production step; do not compare only a separately jitted term sum.
2. Discriminate the 6,882/6,566-cell live-total closure: with identical
   production operands, score the model's actual HPG -> VOR -> KEG -> ZAD ->
   LDF association against NEMO's compiled HPG -> LDF -> VOR -> KEG -> ZAD
   association.  Label production JIT, production eager, and isolated JIT
   separately; a production plant must move the targeted boundary.
3. If the production association closes BIT, resume the magnitude-ranked
   source walk at ZAD, whose `~1.9e-9` incremental discrepancy dominates LDF's
   `~3.5e-14`.  If it does not close, the first non-bit production association
   owns the walk.  Do not revive the held LDF patch by label.
4. Only a statement proven source-exact through the production closure may
   spend the Decision-43 kt1..10 ladder and days 1--30 arm.  Measure DINO for
   any shared statement and register every moved row.

ACQUISITION_NEEDED: NONE

DECISION_NEEDED: NONE
