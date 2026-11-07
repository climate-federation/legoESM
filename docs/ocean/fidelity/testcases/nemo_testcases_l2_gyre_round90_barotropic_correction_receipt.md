# NEMO testcase L2 GYRE phase 3 — round 90 barotropic-correction receipt

## Outcome

**STOPPED_FOR_RECORD.** The frozen target-owner prediction is **REFUTED**, and
no production physics changed.  The inherited Round-46 kt2 stage-one record
does not contain the target consumed by the RK3 barotropic correction: its
`uu_b_Kaa` and `vv_b_Kaa` payloads have zero nonzero wet values, and it has no
direct `zub/zvb` payload.  The record opens before the external solve writes
Kaa; the correction executes later, after stage-one interpolation of that
solved Kaa target.  Therefore neither the preregistered given-input proof nor a
target-only substitution can be made from the inherited record.

The invalid live-trace experiment was retracted from the tool and from the
final tree.  A new additive, write-only NEMO acquisition card is committed and
passes preprocessing plus `gfortran -fsyntax-only`.  The operator must run it
before the first non-bit correction statement can be named.  The production
tree at the end of this round is identical to the incoming tree; only probes,
tests, citation metadata, acquisition files, preregistration, and this receipt
remain.

## Frozen registration and provenance

The preregistration is
`docs/ocean/fidelity/testcases/PREREG_nemo_testcases_l2_gyre_round90_barotropic_correction.md`,
committed as `90bde89f951eb17daf109e9bfa18acb30a9cfadb` before any Round-90
measurement.  Its immutable before arm remains the landed Round-85 production
state: kt2 T/S/U/V maxima `1.4210854715202004e-14`,
`2.1316282072803006e-14`, `2.7377110452773967e-12`, and
`3.284922138989399e-12`; kt3 T/S `1.627497246303733e-4` and
`6.327735185607253e-6`; day-30 T `1.2397011295506804e-2 K`.

The corrected record audit and acquisition package were committed as
`e27a8c4927b32d8db1f0e2bb6ee9fdcf118a9e5c`.  The clean-tree audit artifact is
`record_audit.json`, SHA-256
`4217c173379b455f0f6db6c883fb0f1d36aae14383d22445abeb02aa2b83e253`.
It reports `STOPPED_FOR_RECORD`, header kt=2/stage=1/Kaa=1/fp64, zero nonzero
wet entry targets on both faces, and all eight missing correction-site fields.
Its one-ULP zero-target control is `record_audit_plant.json`, SHA-256
`96b07de07ecbabcb909837d9f1ad4d50b38357c9ca4b8739b16b7c8dee3b721f`;
it prints `PLANT_FIRED` and exits 1.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round90/`.  The admitted
record producer remains `3b3b045bd9e03b60330204e7590e4c4470b7a0ca`.
No NEMO source or executable was modified or run in this sandbox.

## Refutations and retractions

The following failed statements are preserved rather than repaired post hoc.

1. The preregistration said the inherited stage-one record directly carried
   raw Kaa, the consumed target, `zub/zvb`, and final U/V.  **REFUTED:** it
   carries raw/final state, but the recorded entry target is an all-zero
   pre-solve slot and `zub/zvb` are absent.
2. “The target/raw gap is at least four on both faces and target substitution
   closes at least 75%” is **REFUTED/UNMEASURABLE WITH THIS RECORD**.  The first
   attempted table produced 7.213097394424344 U and 3.137884429864073 V, while
   substitution closure was -89.12553783250293 and -76.56249972104727.  Those
   values compare against the wrong all-zero entry slot and are invalid for
   physics ownership; they are retained only as evidence that the instrument
   failed.
3. The preregistered given-input correction replay is **REFUTED as an
   admissible proof**.  Its large residual was caused by the stale target, not
   by a proven correction error.
4. The initial one-ULP target control exited nonzero before `PLANT_FIRED`
   because it found no nonzero wet target.  It therefore did not prove the
   scientific gate.  The replacement structural control plants one ULP into
   the measured zero-target invariant, prints `PLANT_FIRED`, and exits 1.
5. The first probe also used `expose_live_stage_operands`, whose established
   contract warns that returning the diagnostic tuple changes XLA optimization
   boundaries.  Its internal stage fields were not an independent production
   boundary.  All Round-90 extensions to that trace were removed.  The stale
   artifact `baro_split.json`, SHA-256
   `8b01ec7ccd14b1171bf8c4edd0b12c0365b990dd1e0b2470e3f57baa5a02b9ae`,
   is explicitly **INVALID / NOT CITABLE**.  The current tool can no longer
   print those ownership rows; it emits only the fail-closed record audit.

## Compiled-source finding

The exact admitted compiled branch is
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo`.

The stage-one recorder is called at the beginning of the `stp_2D` momentum
walk, before HPG/LDF/VOR/KEG/ZAD have completed
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:141-176`).  Only later
does `stp_2D` call `dyn_spg_ts` to compute SSH and `uu_b/vv_b` at Kaa
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stp2d.f90:301-311`).  Thus the
Round-46 `r46_begin` payload is not the downstream target.

At RK3 stage one the compiled program saves the solved Kaa external fields and,
under the configured barotropic update mode, writes the stage-specific Kaa
`uu_b/vv_b` values
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:140-170`).  The
later correction subtracts the reference-thickness mean from exactly those
current Kaa targets and then adds `zub/zvb` to every wet level
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:728-759`).
Those are the direct fields the new record will capture at the consuming site.

The inherited first non-bit compiled momentum statement remains LDF's
iso-level Laplacian addition from Round 84/85.  Round 90 neither retracts it nor
claims a new correction owner.  The first unresolved statement in this walk is
withheld until its actual inputs and output exist in one aligned record.

## Acquisition package

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round90_baro_correction/run.sh`
creates the new target `GYRE_OMIP_L2_P3_SM_R90BARO` from the exact Round-64
source card, copies EXP00 and MY_SRC file by file, applies only additive calls,
and refuses any namelist difference.  The writer records, only at kt2 stage
one, raw Kaa U/V, current Kaa targets, direct `zub/zvb`, reference thicknesses,
stored reciprocals, masks, and final U/V.  It produces
`oracle_baro_correction_kt00000002_s1.bin`, stamps it with the clean repository
commit, runs a bit-exact final-add gate plus a nonzero final-ULP plant, and then
runs the inherited twin admission plus its consumed-field plant.  It never
changes canonical NEMO source.

The exact source card and patched `stprk3_stg.F90` both preprocess under the
Round-64 keys and pass `gfortran -fsyntax-only`; `gfortran_syntax.log`, SHA-256
`d85c551b8b059e0a4badbfc40507ef643a8d44d07be83440fe6f995a30a89309`,
contains `SYNTAX_PASS writer_and_patched_stprk3`.  The acquisition itself was
not run because `makenemo` and `mpirun` are operator-only.

## Rule-12 table

| Lane | Registered disposition |
|---|---|
| GYRE kt=1..10 | **PRESERVED, NOT RERUN.** No production physics or state changed.  The Round-85 ladder remains the before arm; first-over-bar remains kt2 U/V. |
| Every moved GYRE row | **NONE.** The final tree has no model diff from the incoming commit. |
| GYRE days 1..30 | **PRESERVED, NOT RERUN.** No locally exact candidate exists; the Round-85 day-gap artifact remains immutable. |
| LOCK_EXCHANGE-zco | **PRESERVED, NOT RERUN.** No executing production statement changed.  The acquisition writer is restricted to the GYRE kt2/stage-one record. |
| OVERFLOW-zps | **PRESERVED, NOT RERUN.** No executing production statement changed. |
| DINO | **NO CURRENT PRODUCTION CHANGE / FUTURE SHARED-STATEMENT RISK.** DINO uses MLF rather than this RK3 stage, but a later shared correction-helper or bundled W/ZAD change must be scored; its regional cancellation forbids inferred neutrality. |
| ORCA2 | **UNMEASURED WITH SPEC.** Prove the selected integrator, then independently align raw Kaa, current external target, reference weights/reciprocals, correction, final add, restart state, and kt1..10 T/S/U/V/SSH on native masks.  Require elementwise fp64 equality and normalized L-infinity at `1e-15`; reject any AT-BAR loss, earlier first-over-bar, or wet-point operand/history mismatch. |

No candidate reached the ladder, so running the ladder or days 1--30 would not
adjudicate a changed statement.  No configuration, coefficient, timestep,
stabilizer, carried state, restart contract, year harness, reconciliation gate,
freshwater pair, #1484 guard, or held manifest changed.

## Review, citation gate, and focused verification

The required separate read-only review command was attempted against the
retraction, acquisition card, and Rule-12 disposition.  It failed before a
reviewer model started, so there is no `SHIP` or `DO NOT SHIP` verdict.  Its
terminal verdict is quoted verbatim:

> Error: failed to initialize in-process app-server client: Read-only file system (os error 30)

Per the operator instruction, **independent review unavailable in-sandbox**;
work continued.  `codex_review.txt` has SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The focused CPU/fp64 suite passes 13 tests: the new synthetic record reader,
final-add replay and live plant, the inherited Round-46 reader/control suite,
and provenance controls.  `focused_tests.log` has SHA-256
`2786b120206b5cbe330cdd83313b50bd26d3e9e266646797152e7112b500b8c4`.
The citation gate maps every compiled-source citation in this receipt and
passes with no unmapped citation; `citation_gate.json` has SHA-256
`4ac90af8b646cf011cf292b0927e858ca6ec044be5308bb1887bcb92aac363f3`.
Its shifted external-solve citation plant changes the status to FAIL and
exits 1; `citation_gate_plant.json` has SHA-256
`1083dbba73a30969c442616b8c6f747b74bbf49579251e18e05c218dc74753a0`.
Both Python tools compile, the run script passes `bash -n`, and
`git diff --check` passes.  The known unrelated
`test_rk3_ws_differs_from_rk3_and_is_finite` failure was not encountered.

The required #1455 post could not be made.  The clone's only remote is the
local read-only source clone and `gh auth status` reports the configured token
invalid.  No external issue state was mutated; the next authenticated round
must post this stop, retraction, and acquisition request.

## Choices and uncertainty

Choices made: none.  Decision 37, Decision 38, and Decision 39 remain in force.
This round adds no configuration or carried-state choice.  The uncertainty is
strictly evidentiary: the current record predates the consumed external target,
so the correction's first non-bit operand and any cancelling pair cannot be
identified without the new correction-site record.

## OPEN — exact handoff to round 91

1. The operator must run the absolute `ACQUISITION_NEEDED` script below.  Do
   not reuse `round46/oracle_kt2_stage/uu_b_Kaa` as the stage-one correction
   target.
2. Admit `round90/oracle_baro_correction` only if its `round90_baro_record.json`
   is `READY`, both plants exit nonzero, and `round90_admission.json` is PASS.
   Cite the new target's compiled `stprk3_stg.f90` call sites before using a
   payload.
3. Preregister and score direct raw Kaa, current target, direct `zub/zvb`,
   reference products/mean/reciprocal, and final add in source order.  Require
   the given-NEMO-input shared helper to reproduce direct `zub/zvb` and final
   U/V bit-for-bit before proposing a change.
4. If a correction statement becomes locally exact, pair it with a held member
   only under Decision 38, using the Round-85 immutable ladder/day before arms.
   Otherwise keep the patch held.  Preserve LDF as the first known non-bit debt
   and rank any candidate by kt3 T and day-30 T movement.

ACQUISITION_NEEDED: /tmp/autopilot-work-lIuCcg/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round90_baro_correction/run.sh

DECISION_NEEDED: NONE
