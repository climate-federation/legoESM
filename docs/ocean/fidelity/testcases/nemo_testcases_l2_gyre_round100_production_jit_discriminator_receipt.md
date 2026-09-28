# NEMO testcase L2 GYRE phase 3 — Round 100 production-JIT discriminator receipt

Date: 2026-09-16

## Outcome

**HELD; no production physics landed.** Round 100 completed the exact OPEN
work from Round 99. The held Round-89 source-materialized RK3 assignment was
bit-identical in all three registered modes: isolated-closure eager,
isolated-closure JIT, and scalar replay from operands captured inside the full
production step. The production-fusion hypothesis is therefore **REFUTED for
this member**, and the proposed general production-JIT rule is **not adopted**
from this negative result. This does not retract or generalize the separate
TKE K_H observation.

The admitted Round-99 same-call record confirmed the W-walk diagnosis exactly:
the stored Kbb ratio equals `ssh(Kbb)*r1_ht_0`, the algebraic
`ssh(Kaa)*r1_ht_0` association misses NEMO's Kaa ratio in 201/600 wet cells at
`2.6469779601696886e-23`, and independent interpolation of the separately
formed full-step ratio is BIT. Reapplying the held Round-99
ratio/clock/full-RHS manifest through the production step reproduced every one
of the 17 kt=1 stage-1 output fields BIT.

That full production proof does not overturn Rule 12. The exact same manifest
already moved 85/954 registered ladder rows and failed 56 rows / 131,713 cells
without moving any headline maximum or class. The measurement found no fused
operand or wrong time level behind those movements: the local stage proof
survives production, while the corrected U/V/W/zFw bits propagate through a
model chain whose closure state is already non-bit. The immutable trajectory
veto remains binding, both held patches were removed, and the three affected
production files are byte-identical to preregistration parent `b87ac0edbc9b`.

A post-hoc cross-round attribution sharpens that result without changing the
acceptance statistic: Round 97's full-RHS-only arm already produced the same
aggregate 85/954 moved rows and 56 violating rows, with no class change and
the same kt2 U/V first-over-bar. Thus the full-RHS correction is sufficient
for the row-count veto; no W fusion or time-level defect is needed to explain
it, and adding the locally exact W/ratio/clock members did not cure it.

The first owned stage on restored production therefore remains kt=1 stage 1,
at W. The first named non-bit statement remains legoESM's algebraic stage-ratio
association in place of NEMO's independently interpolated HYB ratio. Nothing
advances to the external step or stage 2.

## Frozen registration and record admission

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round100.md`, committed as
`b87ac0edbc9b` before any Round-99 scientific field was parsed or either held
member was applied. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round100/`.

The operator's existing acquisition was admitted without rebuilding or
rerunning NEMO. Its producer commit is `f3289962e6c4`, record SHA-256 is
`eb8264eb41d6b01c709d8dd4d62ccf14433480677f65b7240ceb1c9cd75a758c`,
and executable SHA-256 is
`5911e1386ee683d0627850da22496bbf88e8f509cc488b3819212d05f6438ce2`.
The operator admission reports PASS with 197 admitted differences and zero
violations. The consolidated gate independently verifies the record digest,
producer stamp, exact header and physical EOF before exposing a field.

Candidate and restoration commits were:

- Round-89 assignment measurement `4b408c3753e4`; restored `e133bc352184`.
- Round-99 stage candidate measurement `0c524edfc20b`; restored
  `393e390bd677`.

## Round-89 execution discriminator

The canonical artifact is
`round89_assignment_production_discriminator.json`, SHA-256
`cf0838d6aaded935741dbbf3b8592001a2dc801c234890d8a150c368a1492cd0`.
It contains one row for each kt=1/kt=2, stage-1/2/3 and U/V combination.

| execution label | rows BIT / total | unequal cells | maximum |
|---|---:|---:|---:|
| isolated-closure eager | 12 / 12 | 0 | 0 |
| isolated-closure JIT | 12 / 12 | 0 | 0 |
| production step, scalar replay from captured live operands | 12 / 12 | 0 | 0 |
| production step versus NEMO output/transcription | 0 / 12 BIT; 2 AT-BAR, 10 DEBT | 180,305 | `2.3065088442784468e-4` |

The last row group deliberately includes upstream live-operand error and is not
a fusion test. Stages 1 and 2 use NEMO's direct post-assignment outputs;
stage 3 is explicitly a compiled-statement transcription because no NEMO dump
exists between its explicit assignment and implicit solve. For example,
kt=2 stage-1 production versus NEMO reaches
`1.1405100346439196e-4` U and `2.3065088442784468e-4` V, consistent with the
existing Round-92 finding that Krhs is already non-bit. The discriminating
condition—isolated BIT but captured-operand production transcription
non-BIT—never occurs.

The production-output one-ULP plant changes the named kt=1 stage-1 U row from
zero to exactly one unequal cell at index `[1,1,0]`, maximum
`6.776263578034403e-21`, and exits 1. Its JSON SHA-256 is
`cba6181c26c27615fdc1f0660e7b7daedec1620bb1fca5849ac2803157b234f1`.
The shifted Round-99 producer stamp and commit stamp controls also exit 1 with
named `producer commit mismatch` and `commit stamp mismatch` failures.

## Same-call ratio record

The four registered rows in the same stage-twin artifact are:

| boundary | classification | unequal / wet | maximum | verdict |
|---|---|---:|---:|---|
| recorded Kbb versus independent stage record | BIT | 0 / 600 | 0 | record identity |
| `ssh(Kbb)*r1_ht_0` versus recorded Kbb ratio | BIT | 0 / 600 | 0 | source product confirmed |
| `ssh(Kaa)*r1_ht_0` versus recorded Kaa ratio | AT-BAR | 201 / 600 | `2.6469779601696886e-23` | algebraic association refuted |
| source-ordered HYB ratio versus recorded Kaa ratio | BIT | 0 / 600 | 0 | independent interpolation confirmed |

All frozen same-call predictions are confirmed. The acquisition field list is
exactly six `36*26` binary64 arrays after its 52-byte header, matching the
44,980-byte admitted record.

## Production stage-twin tables

`round99_candidate_production_stage_twin.json`, SHA-256
`f1453e0cec69cfa8e663d7effca6c44329286cb5aef499a66ef9550966ed7cec`,
is the required production-step reproof. The JSON retains the canonical
row-per-output-field tables. The compact cells below are
`BIT / AT-BAR-not-BIT / DEBT` row counts; stage rows contain 17 fields and
external rows contain 11.

Given NEMO's recorded stage entry:

| kt | stage 1 | external | stage 2 | stage 3 |
|---:|---:|---:|---:|---:|
| 1 | **17 / 0 / 0** | 11 / 0 / 0 | 14 / 2 / 1 | 11 / 1 / 5 |
| 2 | 11 / 3 / 3 | 0 / 0 / 11 | 12 / 2 / 3 | 11 / 1 / 5 |

The candidate's first owned non-bit output moves to kt=1 stage-2 U:
4,787 unequal cells at `1.0842021724855044e-19` (AT-BAR). At kt=1 stage 1,
T/S/U/V, SSH, W/zFw, zFu/zFv, all three Kmm thicknesses and all five closure
fields are each BIT.

Model-chained entry:

| kt | stage 1 | external | stage 2 | stage 3 |
|---:|---:|---:|---:|---:|
| 1 | 12 / 0 / 5 | 10 / 1 / 0 | 9 / 2 / 6 | 3 / 5 / 9 |
| 2 | 3 / 0 / 14 | 0 / 0 / 11 | 0 / 0 / 17 | 0 / 0 / 17 |

At chained kt=1 stage 1 the candidate's U, V, W and zFw rows are all BIT;
the five DEBT rows are inherited TKE closure state (en 914 cells, avm/avt
5,714 each, dissl 17,400, surface 571). Thus the production closure preserves
the candidate's target statements; no eager-only proof was promoted. The
restored implementation returns the campaign to kt=1 stage-1 W.

## Rule 12 and magnitude

No fresh ladder or year member was run because no newly eligible candidate
emerged. The reapplied production candidate is byte-for-byte the manifest
already adjudicated in Round 99, and this round reproduced its prerequisite
stage proof rather than changing it. The canonical Rule-12 artifact remains
`round99/ladder_rule12_b01e550a.json`, SHA-256
`78ef102e2d4f584225e238fef66cf64ada837f9c3cd0a206193a3254d259a1e0`;
it compares all 954 rows, registers all 85 moved rows, finds 56 violating rows
and 131,713 violating cells, preserves every class, and keeps first-over-bar at
kt2 U/V.

For attribution only, the independently preregistered Round-97 full-RHS arm's
canonical `round97/ladder_rule12.json` (SHA-256
`1d2bdd640ffcdf5e12b056913ad8d9a972910376736dfc419de28ab8e66c45d8`)
already reports 85 moved rows, 56 violating rows, no class change and the same
first-over-bar. This is post-hoc evidence that full RHS alone is sufficient
for the row-count failure, not a replacement for the Round-99 gate.

| headline | immutable before | held candidate | disposition |
|---|---:|---:|---|
| kt2 T | `1.4210854715202004e-14` | `1.4210854715202004e-14` | AT-BAR retained |
| kt2 S | `2.1316282072803006e-14` | `2.1316282072803006e-14` | AT-BAR retained |
| kt2 U | `2.7377110452773967e-12` | `2.7377110452773967e-12` | first-over-bar unchanged |
| kt2 V | `3.284922138989399e-12` | `3.284922138989399e-12` | first-over-bar unchanged |
| kt3 T | `1.627497246303733e-4` | `1.627497246303733e-4` | magnitude target unchanged |
| kt3 S | `6.327735185607253e-6` | `6.327735185607253e-6` | unchanged |
| day-30 T RMS | `1.2397011295506804e-2 K` | `1.2397011291846179e-2 K` | improves only `3.66062562207059e-12 K`; still held |

The day score remains the fresh Round-99 artifact
`round99/day_gap_b01e550a.json`, SHA-256
`81896fb17b24e04b5a5dcb742ae770f1623837cfbc3cb6ad7f4da33678218f17`.
The post-hoc attribution above is not promoted to an acceptance claim.

## Testcase dispositions

| lane | disposition |
|---|---|
| GYRE stage twin | Round-89 discriminator refutes a production-fusion miss for that member; Round-99 candidate makes all 17 kt1-stage1 fields BIT through production; restored production keeps stage-1 W owned |
| GYRE kt=1--10 | **FAIL / HELD:** immutable 954-row gate has 85 moved rows, 56 violating rows and 131,713 violating cells; no class change; first-over-bar stays kt2 U/V |
| GYRE days 1--30 | Existing fresh candidate member retained; day-30 improvement is only `3.66062562207059e-12 K` and cannot override Rule 12 |
| LOCK_EXCHANGE-zco | Shared statement not landed; retained constructibility/shared-path coverage, no new tank-fidelity claim |
| OVERFLOW-zps | Shared statement not landed; partial-cell construction retained, no new tank-fidelity claim |
| DINO | **SHARED-STATEMENT RISK:** shared W and WS-RK3 arithmetic execute; no neutrality claim, and the 96--98% regional-cancellation warning remains explicit |
| ORCA2 | **UNMEASURED-WITH-SPEC:** first resolve its integrator, then record native production-step stage entries, raw assignments, W operands/carries, outputs, histories and closure state with red plants |

## Compiled-source basis

The exact record-producing compiled branch first sets stage 1 to `rn_Dt/3`
and saves the full external SSH at
`GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/stprk3_stg.f90:140-150`.
It interpolates the stage SSH separately, calls `dom_qco_r3c_RK3` with the
saved full-step SSH, and then interpolates `r3t/r3u/r3v(Kaa)` at
`GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/stprk3_stg.f90:150-192`.
The ratio kernel's t-cell source statement is the product with stored
`r1_ht_0` at
`GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/domqco.f90:256-258`.

At the actual W call, the additive recorder writes `ssh(Kaa)`, `r1_ht_0`,
`r3t(Kaa)`, `ssh(Kbb)`, `r3t(Kbb)` and `ht_0` at
`GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/sshwzv.f90:283-291`.
The caller invokes `wzv` with `Kbb/Kmm/Kaa` and turns its W into `pFw` at
`GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/traadv.f90:266-280`.
The executing QCO recurrence consumes the Kaa-minus-Kbb ratio with the
stage-local reciprocal clock at
`GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/sshwzv.f90:305-310`.

For stages 1 and 2, NEMO's executing vector assignment is the source-ordered
Kbb plus `rDt*Krhs`, followed by the face mask, at
`GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/stprk3_stg.f90:666-674`.
Stage 3 executes the same expression before its implicit solve at
`GYRE_OMIP_L2_P3_SM_R99R3OP/BLD/ppsrc/nemo/dynzdf.f90:166-169`.
These are the statements replayed in all three discriminator modes; the
compiled statements, not an isolated JAX expression, define the proof target.

## Review and verification

The required separate read-only Codex review was invoked against the clean
draft receipt, the branch diff, both manifests, both full stage-twin artifacts
and the immutable Rule-12 table. A reviewer model did not start. Its verbatim
terminal result is:

> Error: failed to initialize in-process app-server client: Read-only file
> system (os error 30)

Therefore **independent review unavailable in-sandbox**; unavailability is not
approval. The artifact SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
The mechanical Rule-12 verdict already requires HOLD, so no reviewed physics
diff is being shipped.

The focused final-tree suite passed **61 tests in 14.15 s**, covering the
consolidated stage twin and its fail-closed reader, the one shared RK3 stage
helper, the receipt citation machinery and source-rounding primitive. Its log
SHA-256 is
`33d18281be2d7051cdea52444f41248de77382ad30e72e36103dea691b6ecd28`.
The clean-tree receipt citation gate maps all 8/8 compiled-source citations,
finds no failure or unmapped citation, audits every map entry and passes every
self-control. Its JSON SHA-256 is
`d3eec2be1e9e233ee956cad57a057d45f0d1d817bec927dd40764a2f555e405d`.
Shifting the stage-1 clock citation by two lines exits 1 with
`SYMBOL-NOT-AT-LINE`; the plant JSON SHA-256 is
`67781a23c4e954a52a713feba5acb67ce87ad551c2fa37720dfd8f06eca320b3`.

No configuration, coefficient, timestep, carried state, stabilizer, year
harness, reconciliation gate, freshwater pair, #1484 guard, NEMO source or
NEMO executable changed. GitHub issue #1455 cannot be accessed from this
clone's local-only remote, so no issue-update claim is made.

## OPEN — next round

1. Start from restored production at the kt=1 stage-1 W boundary. Do not
   advance to the external step or stage 2: the locally exact Round-99 stage
   candidate remains trajectory-vetoed.
2. Treat the production-fusion hypothesis as refuted only for the measured
   Round-89 source-rounded assignment. Do not promote this negative result to
   TKE K_H or to a general rule; retain the three execution labels in every
   future discriminator.
3. Do not repeat the generic Round-99 split: the independent Round-97 arm
   already proves that full RHS alone is sufficient for the 85-row/56-row
   Rule-12 veto. Compare the Round-97 and Round-99 per-row movement tables,
   then walk only same-stage statements that could compensate the full-RHS
   cellwise damage while preserving all 17 BIT stage outputs. Keep this
   comparison explicitly post-hoc until a new candidate is preregistered.
4. Preserve the admitted same-call record and its field/stamp plants. No new
   NEMO acquisition is needed unless the next split requires an operand not
   present in the Round-46/64/75/98/99 records.
5. Keep the magnitude targets explicit: kt3 T
   `1.627497246303733e-4 K` and day-30 T RMS
   `1.2397011295506804e-2 K`. A candidate lands only if its production-stage
   proof closes and the complete 954-row Rule-12 gate passes.

No configuration or carried-state decision is needed.
