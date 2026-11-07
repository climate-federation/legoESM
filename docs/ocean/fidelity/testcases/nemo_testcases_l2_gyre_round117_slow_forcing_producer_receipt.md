# NEMO-testcases L2 GYRE round 117 receipt: slow-forcing producer

Date: 2026-09-19

Incoming lane tip: `475ca7396`

Preregistration commit: `c82b903f3cf2623c1ab2634880554863611b0477`

Authoritative measurement commit:
`1d1828ca81b4660118fe7d6ee75fc42a6b4da034`

Acquisition-card commit: `84d858b33a07917013da41e604dd43607bc0faeb`

Round status: **STOPPED_FOR_RECORD — no physics or configuration changed; the
direct final slow-forcing pair owns the complete kt2 external-SSH error, but
the two NEMO fields that enter its pre-loop subtract were not recorded, so the
first source-owning statement cannot yet be named without a new passive
record**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round117/`

## Round 118 acquisition correction — RETRACTION

Round 117's `127,408`-byte pre-loop contract and its description as fourteen
full-domain plus four owned-domain arrays are **RETRACTED**.  Reading the
record build's compiled writer shows six full `36x26` arrays (`puu_b`,
`pvv_b`, the two masks, and `cor_u`/`cor_v`) and twelve owned `32x22` arrays
(incoming U/V, eight ENE coefficients, and final U/V), not the reverse.  The
write and the subsequently executed Coriolis/subtract statements are at
`GYRE_OMIP_L2_P3_SM_R117PRELOOP/BLD/ppsrc/nemo/dynspg_ts.f90:329-345`.
Including the 48-byte header, the physical record is therefore exactly
`112,560` bytes.  All 18 intended fields are present; this was a reader and
shell-arithmetic defect, not a short NEMO write.

The operator's existing record was recovered without rebuilding or rerunning
NEMO.  The no-build admission path verified the original source manifest,
producer-time repository tools, external toolchain inputs, built and copied
executable SHA-256 (`b5c750863e71af3c23e958794ecb24dfce3c36faa57ba6247026e081b7cc497e`),
ten-step `STOP 0`, and producer commit
`c8f5d513df453f3f12a3d5eb05b05be8c8d3a2fd`; only then did it write the
missing record stamps and admission report.  The admitted pre-loop record is
`112,560` bytes with SHA-256
`259fcbb042ce0aa7a28f7db75b746ca56dce2ea2a62e960a6a265f58f691c176`,
header `(1,2,3,36,26,64,18,704)`, and physical EOF.  All six duplicate and
subtract rows are BIT, and the inherited consumed-field census passes with 46
exact, 20 intentionally changed, and 132 admitted fields.  Stamp, truncation,
header, layout, input-ULP, reference-ULP, byte-size, compiled-layout and
consumed-field plants each print a named failure and exit nonzero.

## Outcome first

The complete production-JIT walk reaches the first directly recorded non-bit
boundary at the completed pre-loop slow forcing: U is 580/580 wet faces at
`1.0529650291768787e-11 m s-2`; V is 570/570 at
`1.0765559917925099e-11 m s-2`.  The compiled program first copies
`Ue_rhs`/`Ve_rhs` into `zu_frc`/`zv_frc` at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:286-291`, then
evaluates the Kmm `dyn_cor_2D` call and subtracts its masked result at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:298-321`.

Neither direct input to that subtract is present in the admitted record.  The
Round-81 `cor_u`/`cor_v` fields are written after the *substep-1* call on
`ua_e`/`va_e`, not after the pre-loop Kmm call; the write and subsequent
frozen-forcing write are in
`GYRE_OMIP_L2_P3_SM_R81BTSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:670-713`.
Round 117 therefore retracts the earlier pre-loop label in the executable gate
and calls those comparisons `PROXY`.  The isolated subtract from the model's
own live incoming and Coriolis fields to its final field is BIT for U and V,
which exonerates the transcription *given its own inputs* but does not prove
which input NEMO would disagree with.

This is the required first-non-bit statement disposition: the first direct
non-bit boundary is the output of the compiled subtract assignments above;
ownership among the preceding copy, Kmm Coriolis call, and subtract is
**WITHHELD_NO_DIRECT_RECORD**.  Calling either reconstructed proxy a NEMO
input would violate the preregistration and source order.

Magnitude is large at the external output.  Replacing only the direct final
slow-U/slow-V pair with the admitted NEMO pair makes weighted final SSH BIT in
all 600 wet cells, from an ordinary maximum error of
`7.072560112143626e-7 m` to zero.  It slightly improves local kt3 T from
`8.600420500215478e-7` to `8.600379715062445e-7 K` and local kt3 S from
`6.979443156751586e-8` to `6.979384181704518e-8`.  This one-variable
oracle-directed arm establishes carried magnitude, not a source-exact
candidate.

The current cumulative 3-D RHS walk reproduces HPG as BIT and first differs
after LDF, but its final live-total closure is not BIT.  Consequently LDF is
the first measured cumulative boundary, not yet an operator owner.  No
physics lands, and no Decision-43 ladder or month arm is spent.

## Frozen prediction ledger

The frozen preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round117.md`.

| frozen prediction or falsifier | disposition | evidence |
|---|---|---|
| production-JIT final U/V reproduces 580/570 cells and `1.0529650291768787e-11` / `1.0765559917925099e-11` | **CONFIRMED** | Counts and maxima reproduce exactly. |
| Round-81 `cor_u`/`cor_v` is the direct pre-loop Kmm result and therefore BIT | **REFUTED** | Compiled source places that record at the substep call; the gate retracts the label. |
| direct incoming U/V differs by `[1.0e-11,1.2e-11]` | **REFUTED / UNMEASURED** | The direct field is absent.  The available proxy is `3.4219257004139595e-10` / `2.80846148198001e-10` and is explicitly not substituted for the missing record. |
| isolated subtract from live inputs closes BIT to the production final | **CONFIRMED** | U and V are 0 unequal with maximum zero. |
| HPG remains BIT; LDF is the first cumulative non-bit boundary in the registered range | **CONFIRMED** | LDF maxima are `2.5292467120726215e-14` U and `3.502735092670824e-14` V. |
| the largest incremental cumulative residual is ZAD | **CONFIRMED** | U/V incremental maxima are `1.9220297482797664e-9` / `1.966061294804274e-9`. |
| the NEMO `after_adv` snapshot equals `after_zad` BIT | **CONFIRMED** | Both faces are zero unequal. |
| current cumulative `after_adv` closes BIT to the production live total | **REFUTED** | U is 6,882/17,400 and V 6,566/17,100, both at `8.470329472543003e-22`.  Operator ownership is withheld. |
| directed SSH, T and S move but remain non-bit; SSH and T improve | **PARTLY REFUTED** | All move and SSH/T improve, but SSH becomes exactly BIT, triggering the frozen falsifier.  S also improves. |
| an incoming-U one-ULP plant moves incoming and final, leaves Coriolis BIT, prints `STATUS PLANT-FIRED`, and exits nonzero | **CONFIRMED** | Exactly one incoming and final word move by `1.6543612251060553e-24`; Coriolis stays BIT. |
| absent direct inputs forbid a source-exact candidate | **CONFIRMED** | The Round-64/Round-81 forward join is non-bit and `direct_input_certified=false`. |

## Record and execution discipline

The authoritative artifact is
`producer_walk/production_jit_v2.json`, SHA-256
`c99da908bbf44f3f8f5ce3c20467868a80e570821e86a489b3c12bad62f3f80b`.
It stamps the clean authoritative commit and labels its regime
`production-jit-cpu-fp64-x64-libm`.  A second ordinary production step and the
instrumented step are BIT over 23 leaves and 198,956 cells.  The trace's final
pair is BIT to the actual external-solver call.  Thus the observer does not
change the production result.

Production eager is secondary.  Its artifact is
`producer_walk/production_eager_v2.json`, SHA-256
`479607225018357c680326b3454e10b6780db0697f56023f1866010e04819a67`.
It finds the same scientific boundaries; last-bit differences from production
JIT include final maxima `1.0529650291765478e-11` U and
`1.076555991792179e-11` V and LDF U
`2.5292467226605334e-14`.  It is not used to certify the production owner.

The separately labelled isolated-closure JIT scores only the final subtract.
It is BIT for both faces but is not called a production-step proof.  This
preserves the Round-102 full-closure JIT discipline.

The admitted compiled cumulative program calls HPG, LDF, VOR, KEG and ZAD and
snapshots the shared accumulator in exactly that order at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:141-176`.
It then evaluates the vector-form vertical mean at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:202-207`, drag at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:225-227`, and wind at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:229-235`.

## Production-JIT measured rows

The direct/proxy split is:

| boundary | U unequal / wet; max | V unequal / wet; max | authority |
|---|---:|---:|---|
| incoming versus substep preimage | 580/580; `3.4219257004139595e-10` | 570/570; `2.80846148198001e-10` | proxy only |
| pre-loop live Coriolis versus substep-1 `cor` | 580/580; `3.421803558457394e-10` | 570/570; `2.8003826188779935e-10` | proxy only |
| isolated live subtract versus production final | 0/580; `0` | 0/570; `0` | isolated arithmetic closure |
| production final versus direct NEMO record | 580/580; `1.0529650291768787e-11` | 570/570; `1.0765559917925099e-11` | direct |

The Round-64 producer replay joined to Round-81 final is not exact: U is
580/580 at `2.105283055809266e-9`; V is 570/570 at
`1.7281484610543052e-9`.  The gate therefore withholds the kt2 wind-stress
identity rather than treating a cross-run reconstruction as an input record.

The cumulative rows are full 3-D wet-cell censuses:

| boundary | U unequal / wet; max | V unequal / wet; max |
|---|---:|---:|
| after HPG | 0/17,400; `0` | 0/17,100; `0` |
| after LDF | 17,400/17,400; `2.5292467120726215e-14` | 17,100/17,100; `3.502735092670824e-14` |
| after VOR | 17,400/17,400; `2.5292467332484452e-14` | 17,100/17,100; `3.502735092670824e-14` |
| after KEG | 17,400/17,400; `2.5292467332484452e-14` | 17,100/17,100; `3.502735092670824e-14` |
| after ZAD | 17,400/17,400; `1.9220276136604423e-9` | 17,100/17,100; `1.966059508480186e-9` |
| after ADV | 17,400/17,400; `1.9220276136604423e-9` | 17,100/17,100; `1.966059508480186e-9` |
| live-total closure | 6,882/17,400; `8.470329472543003e-22` | 6,566/17,100; `8.470329472543003e-22` |

The last-bit closure failure matters: “LDF first” does not establish that
LDF's statement alone owns the live producer mismatch.  No held LDF patch is
revived in this round.

## External-SSH and kt3 magnitude

The diagnostic arm changes exactly one registered input pair at the real
external-solver boundary.  Its ordinary wrapper is BIT to the unwrapped
production step, and incoming/Coriolis fields remain BIT between ordinary and
directed arms.  Results are against the independent NEMO next-step/state
records:

| output | ordinary | final-slow-pair directed | directed minus ordinary |
|---|---:|---:|---:|
| weighted final SSH | 600/600; `7.072560112143626e-7 m` | 0/600; `0` | 600/600; `7.072560112143626e-7 m` |
| local kt3 T | 17,999/18,000; `8.600420500215478e-7 K` | 17,997/18,000; `8.600379715062445e-7 K` | 17,197/18,000; `8.618172842034255e-11 K` |
| local kt3 S | 17,265/18,000; `6.979443156751586e-8` | 17,007/18,000; `6.979384181704518e-8` | 15,251/18,000; `1.333262389380252e-10` |

The pair carries 100% of the measured final-SSH maximum on this diagnostic
arm.  It carries only a very small beneficial change at local kt3 T/S.  The
intervention replaces a completed NEMO boundary rather than one attributed
source statement, so it is not a landing candidate.

## Plants and non-vacuity

The authoritative plant artifact is
`producer_walk/incoming_ulp_plant_v2.json`, SHA-256
`f79db7135ef6a48d852e8012228fa56f4a84e58d9cf2a7ca2d4af68b5275e37d`.
It advances incoming U at `[1,1]` by one ULP, changes exactly that incoming and
final word by `1.6543612251060553e-24`, leaves the Coriolis word BIT, prints
`ROUND117 INCOMING-ULP STATUS PLANT-FIRED`, and exits 1.  Observer-identity
unit controls cover all returned leaves.

The acquisition gate adds independent stamp, truncation, header,
incoming-ULP and Round-81-reference-ULP plants.  Synthetic tests prove each
prints its own `STATUS PLANT-FIRED` and exits nonzero.  The operator-run card
will execute those same plants on the acquired records.

## Acquisition required

The missing direct fields block the next exact statement walk, so the round
ships one operator-run source card:

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round117_preloop_forcing/run.sh`

It clones the exact `GYRE_OMIP_L2_P3_SM_R111FCTW` source card file by file
into the new target `GYRE_OMIP_L2_P3_SM_R117PRELOOP`, changes no namelist, and
applies two additive zero-fuzz patches.  The kt2 slow record observes the
vertical mean, drag and wind program already compiled at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stp2d.f90:202-207` and
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/stp2d.f90:225-239`.
The pre-loop record observes the direct incoming pair, Kmm velocity, masks,
eight ENE coefficients, direct Kmm Coriolis result, and completed final pair
around the compiled copy/call/subtract sequence at
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:286-291` and
`:298-321`.

The corrected mixed byte layout is explicit from the compiled writer (the
original preflight arithmetic below is retained as retracted evidence):

| record | exact bytes | payload contract |
|---|---:|---|
| `oracle_slow_forcing_kt00000002.bin` | 1,486,548 | admitted `NEMO_L2_SLOW_2` schema |
| `oracle_preloop_forcing_kt00000002.bin` | 112,560 | six full 36x26 arrays plus twelve owned 32x22 arrays; header also writes `SIZE(zu_frc)=704` |

The new gate parses physical EOF, requires both exact sizes and headers,
requires kt2 slow post-wind to equal pre-loop incoming BIT, replays the written
subtract BIT, and requires final U/V to equal Round-81 substep-1 frozen forcing
BIT.  The existing admission gate also requires every inherited oracle record,
the final restart and mesh to remain consumed-field/byte exact and fires its
owned-cell plant.

Codex ran only `--preflight-only`; it did **not** run `makenemo` or `mpirun`.
The retained preflight at commit `84d858b33a07917013da41e604dd43607bc0faeb`
reports:

```text
ROUND117_SYNTAX_PASS stp2d.f90 dynspg_ts.f90
ROUND117_LAYOUT slow=1486548 preloop=127408 full2=936 owned2=704
ROUND117_PREFLIGHT_READY commit=84d858b33a07917013da41e604dd43607bc0faeb target=GYRE_OMIP_L2_P3_SM_R117PRELOOP
```

The `preloop=127408` line in that historical preflight is the now-retracted
arithmetic defect.  The recovered committed tool prints
`ROUND117_LAYOUT slow=1486548 preloop=112560 full2=936 owned2=704` and ends in
`ROUND117_PRELOOP_RECORDS_READY`.

The log SHA-256 is
`49cb604fe43096f1e07ae366a0874cf3ea736337ef739a0b1e12a4d6744d3300`.

## Decision-43 trajectory and cross-card scope

No production statement changed, so there is no after arm and every certified
trajectory headline remains the incoming value:

| required headline | before | after | disposition |
|---|---:|---:|---|
| kt2 T | `1.4210854715202004e-14 K` | same | AT-BAR, unchanged |
| kt2 S | `2.1316282072803006e-14` | same | AT-BAR, unchanged |
| kt2 U | `2.7377110452773967e-12 m s-1` | same | first-over-bar DEBT, unchanged |
| kt2 V | `3.2849219221489645e-12 m s-1` | same | first-over-bar DEBT, unchanged |
| kt3 T | `8.600419718618468e-7 K` | same | DEBT, unchanged |
| kt3 S | `6.979441735666114e-8` | same | DEBT, unchanged |
| day-30 T RMS | `6.890484901489568e-5 K` | same | no candidate; month not rerun |

Decision 43's month-decrease, first-over-bar, kt1, moved-row registry and
recipe-derived executing-card criteria are not invoked by an empty production
diff.  The diagnostic arm is not called an after arm.

GYRE-zco and the generic NEMO-GYRE recipe execute the measured shared path.
DINO shares downstream momentum machinery and retains its known cancellation
risk, but no production bits changed and no DINO number is claimed.  A future
shared candidate must measure DINO before landing.  LOCK_EXCHANGE and OVERFLOW
also cannot move from this round's diagnostic/test-only changes.

ORCA2 remains **UNMEASURED-WITH-SPEC**: independently record and admit its
kt2 3-D cumulative RHS boundaries, depth average, drag, wind, direct pre-loop
incoming/Kmm-Coriolis/final forcing, complete external step, weighted final
SSH, next-step entry and local kt3 U/V/W/T/S; run production JIT, production
eager and isolated controls with the same plants; then run its certified
trajectory for any shared candidate.

No restart/checkpoint representation, configuration, selector, coefficient,
timestep, stabilizer, carried state, year harness, reconciliation gate,
freshwater pair, #1484 guard, held manifest, or NEMO source changed.

## Review, citations, and tests

The required separate adversarial review was invoked against the committed
receipt and full Round-117 diff with `codex exec --sandbox read-only -C`.
Independent review was unavailable in-sandbox: the read-only app-server
initialization failed before a reviewer verdict existed.  This is not an
approval or a `SHIP` verdict.  The terminal result is quoted verbatim:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
CODEX_REVIEW_EXIT=1
```

The complete review attempt is `round117/review/codex_review.log`.

The clean-tree citation gate found all nine citations, with zero failures,
zero unmapped citations, zero failing map-audit entries, and every built-in
control fired.  The correctly targeted shifted-line plant moved
`GYRE_OMIP_L2_P3_SM_R111FCTW/BLD/ppsrc/nemo/dynspg_ts.f90:298-321` by two
lines, reported `SYMBOL-NOT-AT-LINE`, and exited 1.  An earlier invocation
mistakenly supplied the resolved absolute path instead of the extracted
citation key; that control did not fire and correctly exited 2.  It was
replaced, not reinterpreted, by the firing control.  Reports and logs are in
`round117/citations/`.

CPU/fp64 test summaries already complete are:

```text
7 passed in 0.49s
17 passed in 0.71s
119 passed in 348.61s (0:05:48)
```

The first is the acquisition reader/gate before bundling; the second is every
Round-117 changed focused control (Round-51 live operands, extended Round-83
walk, acquisition gate); the third is the inherited four-file push gate:
receipt citations, TKE NEMO terms, NEMO recipe, and real freshwater closure.

Two intermediate citation diagnostics are retained rather than hidden.  The
first, before re-anchoring the model file, reported:

```text
3 failed, 13 passed in 1.90s
```

It identified 31 shifted map entries and nine used citations.  After rigid
re-anchoring but before commit, the symbol audit was clean and only the
intentionally fail-closed dirty-worktree stamp tests remained:

```text
2 failed, 14 passed in 1.93s
```

The final committed four-file run above includes all 16 citation tests green.

The required one-piece `-n 12 tests/ocean/fidelity tests/ocean/unit` run was
attempted once and terminated at 83% after repeated JAX compiler-worker
aborts and an xdist `MemoryError`.  Its exact terminal summary was:

```text
186 failed, 6465 passed, 140 skipped, 2 xfailed, 56 warnings, 44 errors in 1072.29s (0:17:52)
```

It collected 8,198 cases but emitted a 6,839-case partial JUnit file.  The
partial file contains 230 unique failure/error node IDs: 65 are in the pinned
87-ID incoming baseline, 165 are not, and 22 pinned IDs were not reached as
bad before the internal abort.  None of the 230 belongs to a Round-117 changed
test.  This is a resource-contaminated partial observation, not a failing-set
verdict.

Five worker-crash IDs were recoverable from JUnit and the sixth from xdist's
terminal crash item.  Their clean serial replay reported:

```text
1 failed, 5 passed in 236.12s (0:03:56)
```

The sole serial failure is the pinned incoming
`test_model_rollout_grads_finite_f32[ppm_fct]` custom-VJP/JVP red.  The EKE,
multigrid, shared-QCO-face, RK3-WS and tripole-partial-cell crash nodes all pass
serially.  Complete logs, partial JUnit, baseline diff and serial replay are in
`round117/tests/`.

## ASKED / UNASKED

ASKED and completed: preregistration before measurement; extension of the
admitted producer gate rather than a second science harness; complete
production-JIT/eager/isolated labels; observer identity; direct/proxy
retraction in executable code; source-order cumulative walk; final SSH and
local kt3 T/S magnitude; nonzero plant; first directly recorded non-bit
boundary; explicit withheld statement ownership; exact baseline kt2 U/V, kt3
T/S and day-30 T; DINO risk; ORCA2 acquisition specification; additive NEMO
source card; exact mixed-layout arithmetic; gfortran syntax proof; focused,
push and full-tree test attempts.

UNASKED and not done: NEMO source was not modified; `makenemo` and `mpirun`
were not run by Codex; no oracle record was invented or acquired; no
production physics, configuration, default, coefficient, stabilizer, carried
state, restart schema, year harness, reconciliation gate, freshwater pair,
#1484 guard, or held patch changed; no proxy or isolated closure is called a
direct NEMO production proof.

## OPEN for round 118

1. The operator runs the exact committed acquisition card and requires
   `ROUND117_PRELOOP_RECORDS_READY`; do not rebuild or rerun an unchanged
   binary if post-run admission alone needs correction.
2. Start with the acquired gate and its five nonzero plants.  Require every
   inherited record, final restart and mesh to pass admission before citing a
   new value.
3. In the existing Round-83 producer gate, replace the proxies with the direct
   kt2 incoming and direct pre-loop Kmm Coriolis fields.  Score copy, Coriolis
   and subtract separately under production JIT, with eager and isolated rows
   secondary.  Quantify any incoming/Coriolis cancellation.
4. If incoming is first, join the same-run kt2 slow record to the inherited
   admitted kt2 Round-46 cumulative snapshots and rerun the complete HPG ->
   LDF -> VOR -> KEG -> ZAD -> ADV -> depth -> drag -> wind chain.  A
   candidate requires BIT direct inputs and a closed live total; otherwise
   walk the first unclosed input rather than patching LDF by label.
5. If Kmm Coriolis is first, walk its direct Kmm velocity, eight ENE
   coefficients, mask and written association.  Do not use the substep-1
   `cor_u`/`cor_v` proxy.
6. Only after one source-owning statement is exact may Decision 43 spend the
   kt1..10 ladder and days 1--30 month arm.  Register every moved row and
   measure DINO if the statement is shared.

ACQUISITION_NEEDED:
`/tmp/autopilot-work-114221346/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round117_preloop_forcing/run.sh`

DECISION_NEEDED: NONE
