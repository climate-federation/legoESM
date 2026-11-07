# NEMO-testcases L2 GYRE round 72: tracer-stage source walk receipt

Date: 2026-09-12. Final disposition: **STOPPED FOR RECORD / FIRST-BOUNDARY
PREDICTION REFUTED; no production physics change landed**. The admitted oracle
producer is `5f9df918ca4d6d9536d920ccd82888df329ec306`; 132 consumed values were
admitted, with 45 of 65 inherited files exact and 20 changed only in admitted
regions.

## Verdict and first non-bit statement

The preregistered prediction that every stage-1 row would be exact is
**REFUTED**. The first measured non-bit output is stage-1 `zFu`: all 17,400
wet U-face cells differ, with maximum absolute difference
`0.8916110997497526` and RMS `0.15292562665127313`. Stage-1 `zFv` is also
unequal in all 17,100 wet V-face cells, with maximum `0.7485346468365606`.

This is the first non-bit statement named by the executed compiled walk:
GYRE constructs `zub`/`zvb` from the external-mode average, inverse live
depth, and Kmm barotropic velocity at
`GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:289-290`, then
writes `zFu` at `GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:295`
before `zFv` at
`GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:296`.
The record used this round begins only at `zFu`; it does not distinguish an
upstream `un_adv`, inverse-depth, `uu_b`, Kmm velocity, thickness, mask, or
arithmetic-association owner. Those operands are therefore **UNMEASURED**, not
inferred, and the walk stops for their exact kt=2 record.

The stage-2 prediction and causal arm are **UNREACHED**, not failed. Source
order forbids replacing stage-2 transports once an earlier stage-1 statement
is already non-bit. No oracle substitution or production numerical edit is
eligible.

## Record replay and measured ladder

The clean citable report is
`round72/round72_tracer_stage_final.json`, stamped to clean instrumentation
commit `c0044f86111178de5a55c5c1b62de29ec5018499`, with SHA-256
`ec6af7bd5d2a4fb4cf1a43001729756663f8c895fb52ab29491a252953551271`.
The record, live tracers, and live geometry are all float64. Its independently
constructed trace state is exact, and it reproduces the round-71 stage-3 Kmm
census exactly.

The oracle self-replay is bit-exact for all four stage/tracer rows: zero
unequal cells for stage 1 T/S and stage 2 T/S. This checks the compiled
thickness-weighted recurrence itself, not a live-model attribution. NEMO calls
advection and then the surface boundary source at
`GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:860,867` and applies
the recorded Kbb/Kmm/Krhs/r3t recurrence at
`GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:899-901`.
The source card records the transport after its producer at
`GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:843-853`, both RHS
boundaries at
`GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:864-872`, and the
update inputs/output at
`GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:907-913`.

| boundary | first quantity: unequal / max | second quantity: unequal / max |
|---|---:|---:|
| stage-1 Kbb | T: `0 / 0` | S: `0 / 0` |
| stage-1 Kmm | T: `0 / 0` | S: `0 / 0` |
| stage-1 horizontal transport | zFu: `17400 / 0.8916110997497526` | zFv: `17100 / 0.7485346468365606` |
| stage-1 post-advection RHS | T: `18000 / 2.657320367980935e-13` | S: `18000 / 4.2066170832062363e-13` |
| stage-1 Kaa | T: `10687 / 2.1881163547732285e-11` | S: `8918 / 1.4921397450962104e-12` |
| stage-2 horizontal transport | zFu: `17400 / 10.360390496698528` | zFv: `17100 / 10.600986959700094` |
| stage-2 post-advection RHS | T: `18000 / 1.1648552074514723e-10` | S: `18000 / 9.852692552935084e-12` |
| stage-2 Kaa | T: `17994 / 8.369461070856232e-7` | S: `16769 / 6.794565621248694e-8` |

The stage-1 vertical transport has 17,613 unequal values of 21,824 scored
interfaces and maximum `3.1650415621697903e-10`. It follows horizontal
transport in this instrument's array order and does not supersede the earlier
`zFu` stop. Stage-1 r3t Kbb/Kmm differs in all 600 horizontal cells but only by
`1.1031574463560756e-16`; those weights are first consumed by the tracer update
after transport and therefore do not own the first consumed-statement
boundary in this walk.

Two implementation mistakes were rejected before any citable report was
written. The first draft tried to score NEMO's 31-interface `zFw` against a
30-level cell mask; the corrected instrument scores the full interface
ladder. The second draft passed traced grid metadata through a Python boolean
inside JIT; the corrected replay closes over the static card grid. Neither
failed attempt emitted or replaced the clean artifact.

## Controls

Both preregistered scientific plants fail closed and their commands exit
nonzero.

- The Kaa-ULP plant changes one recorded wet stage-2 T value by one
  `nextafter` ULP. Oracle self-replay becomes false. Its artifact is
  `round72/plant_kaa_ulp.json`, SHA-256
  `99cc295d969d9bc547f28c12238a9db6fd1860a45deef48b2d5219f133c396fa`.

- The transport-null plant executes the otherwise source-order-skipped causal
  arm with the live transport as its own target. For both tracers,
  active movement, fourfold maximum improvement, and unequal-cell halving are
  all false. Its artifact is `round72/plant_transport_null.json`, SHA-256
  `880d45bce2cc1c32c3131477cdc0ca16384165aff4a8bf4d24d81819c32a4bec`.

Inherited dry-cell division warnings are diagnostic only; every reported row
is scored on its owned wet mask. The default-inert private stage-2 transport
hook is tested directly, and no public configuration or production physics
path was added.

## Acquisition prepared for the actual owner walk

The missing kt=2 stage-1 operand record is prepared at
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round72_stage1_transport/run.sh`.
It creates the new target `GYRE_OMIP_L2_P3_SM_R72ZFOP` from the exact R71
source card, copying EXP00 and MY_SRC file by file. The R71 compiled writer
already records metric, live face thickness, Kmm velocity, correction, mask,
`zFu/zFv`, `un_adv/vn_adv`, inverse live depth, and Kmm `uu_b/vv_b` at
`GYRE_OMIP_L2_P3_SM_R71FCTST2/BLD/ppsrc/nemo/stprk3_stg.f90:299-321`.
The additive patch changes only its WRITE predicate and filename so the same
record is emitted for kt=2 without overwriting kt=1.

The exact dry-applied R71 source passed `gfortran -fsyntax-only` with the R71
build includes. The acquisition gate parses the exact header, reconstructs
`zub/zvb` and `zFu/zFv` in compiled association, and requires bit identity.
Its stamp, header, truncation, replay-ULP, and consumed-field admission plants
must all exit nonzero. Twin admission permits only
`oracle_rkstage1_transport_operands_kt00000002.bin` and requires the final
restart and mesh mask bit-identical. Per campaign prohibition, this round did
not invoke makenemo or mpirun.

## Rule 12 card

| card | changed statement | disposition |
|---|---|---|
| GYRE | none in production; default-inert private measurement hook and WRITE-only acquisition only | **STOPPED FOR RECORD** before an eligible edit. The kt1--10 ladder and days 1--30 are **UNREACHED**. No registered row or AT-BAR row moved; first-over-bar cannot move. |
| LOCK_EXCHANGE | none | No candidate production statement exists to execute or gate; tank trajectory is **UNREACHED** and unchanged. |
| OVERFLOW | none | No candidate production statement exists to execute or gate; tank trajectory is **UNREACHED** and unchanged. |
| DINO | none | **UNREACHED**. Its shared ocean stage/transport implementation makes any future change at this boundary an explicit cancellation risk requiring DINO's own row gate. |
| ORCA2 | none | **UNMEASURED WITH SPEC**: resolve its compiled card; record stage-1/2 transport primitives, compound corrections, `zF`, RHS and Kaa for kt1--10; replay each statement in compiled order; register every moved row, preserve every AT-BAR row, and forbid an earlier first-over-bar boundary. |

No configuration/default, carried state, stabilizer, NEMO source/build/run,
year harness, reconciliation gate, freshwater pair, #1484 guard, or held
manifest changed.

## Review and focused checks

The required independent pass was invoked with `codex exec --sandbox
read-only` against the complete committed round diff and an adversarial prompt
covering the first boundary, oracle replay, plants, acquisition, and Rule 12.
It exited 1 before reviewing. Its terminal verdict, quoted verbatim, is:
**“Error: failed to initialize in-process app-server client: Read-only file
system (os error 30)”**. There is no SHIP/DO NOT SHIP verdict; the review
requirement remains **UNMET/BLOCKED**, and absence of a verdict is not approval.
The full log is `round72/codex_round72_review.log`, SHA-256
`92015cb4c0f52be782d229ae8a686707c0122077dd13c515153d648e2a22da67`.
No production numerical diff is being landed under that blocked review.

The round-72 replay/hook and acquisition-gate tests report four passed. The
combined focused run reports 64 passed in 1,639.15 seconds; its JUnit artifact
is `round72/focused_tests.xml`, SHA-256
`9d4514880fd474267e6961703fcab629ec00f76c295887f881cbed944c51333c`.
Ruff on the new round-72 Python files, shell parsing, Python compilation,
`git diff --check`, and the dry Fortran syntax proof pass. The entire
receipt-citation test file reports 16 passed after the line shifts caused by
the private hook were re-anchored in their own commit. The citation gate audits
this receipt against the R71 compiled branch; its shifted-citation plant
targets the stage-1 `zFu` line and exits nonzero with SYMBOL-NOT-AT-LINE.

## ASKED / UNASKED and OPEN

| state | item | disposition |
|---|---|---|
| ASKED | configuration choice | none encountered |
| UNASKED | configuration, carried state, stabilizer, NEMO, or harness change | none performed |

OPEN for round 73: first run and admit the prepared R72 record, then read and
cite the new target's compiled writer. Walk stage-1 U transport in compiled
order: `un_adv`, live inverse depth, their product, Kmm `uu_b`, `zub`, metric,
live face thickness, Kmm velocity, mask, the corrected velocity, and the two
final multiplication associations producing `zFu`. Stop at the first non-bit
input or operation; only then walk the V analogue and `zFw`. Require exact
NEMO self-replay and non-vacuous plants before comparing live legoESM operands.
Do not infer from the old kt=1 record or the round-46 stage record, and do not
resume the held round-70 output-pair patch. If the named owner becomes an
independently computable shared legoESM statement, preregister its magnitude
and full Rule-12 causal gates before editing production.
