# SI3 round 29 (`si3_r29`) — O17 design and record audit receipt

Date: 2026-10-31. Base: `44ca9044917b13ac14e074223f436385a78eb058`.
No NEMO build, launch, or integration was run.

## Outcome

**ACQUISITION_NEEDED; the unified O17 producer is now scorable.** Resume 5 below is the
authoritative closure of the `6e02036e3f` review. The admitted O8--O16 master has
the correct unmodified production deck, 100-step window, dynamics disposition,
thermodynamic stage frames, exchange outputs, and restart. It does not have a
complete ice-step-entry state or raw O7 meteorology at every step. The O1--O7
records do not fill those holes over the same 100-step production trajectory.

The audit in this opening section predates the Resume 1--3 implementation. The
typed composition entry and one-trajectory acquisition/scoring tooling now
exist; the operator-produced unified O17 record and its through-entry numerical
verdict still do not.

There is also no existing legoESM production entry point that performs the O17
program. The current dispatcher accepts a typed finalization forcing and calls
O15, O16, and the stress tail, while its earlier O2 forcing path raises
`NotImplementedError` (`packages/ice/legoesm/ice/sea_ice.py:642-725`). O4--O16
therefore do **not** already score through one common full-step entry point.
The exact missing piece is a typed `si3_orca2_o17` production composition entry
that orders the already admitted O4 Prather, O5 H79+aEVP, O6 redistribution and
post-dynamics correction, live O7 surface step, and O8--O16 thermodynamics and
finalization. This is composition wiring and a forcing/response contract, not a
new physical operator.

The round instruction says to stop and name the missing piece if composition
needs one. Consequently this round did not invent that entry point, did not
write an acquisition driver whose read-set would precede the missing production
contract, and did not run the requested one-step gradient program. An O17
acquisition remains required after that entry point fixes the authoritative
step-entry and live-forcing pytrees.

## Measure-first record audit

The target oracle row is the shipped `ORCA2_SAS_ICE` deck: steps 1--100 and
`rn_Dt=10800` (`cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg:21-28`),
`nn_fsbc=1` (`cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg:59-61`), and CPP keys
`key_si3 key_linssh key_vco_1d3d key_xios key_RK3`
(`cfgs/ORCA2_SAS_ICE/cpp_ORCA2_SAS_ICE.fcm:1`). The O8--O16 acquisition was
preregistered as the unmodified production ice namelist, production dynamics
on but outside stage projections, O7-completed surface products as its forcing
boundary, and 100 steps (`docs/ice/fidelity/si3_r18_o8_o16_master_preregister.md:3-24`).
Those identities agree with O17; the forcing boundary does not.

| Evidence | Deck / window | What it proves | O17 missing data |
|---|---|---|---|
| O4 admitted record | Prather-only diagnostic overlay, 2 steps | Five-category Prather is 0 ULP for both sweep orders; its manifest explicitly disables all other dynamics and thermodynamics (`docs/ice/fidelity/si3_r25_o4_score_receipt.md:3-6,19-29`). | Not the production deck or 100-step carry; no live O7 forcing. |
| O5 admitted record | Dynamics diagnostic overlay, 3 steps | H79+aEVP is 0 ULP on its registered three-step production-JIT comparison; thermo and redistribution are disabled (`docs/ice/fidelity/si3_r12_o5_acquisition_receipt.md:3-14,26-34`). | Not the production deck or 100-step carry. |
| O6 receipt | Local model admission only | The receipt explicitly leaves ORCA2 NEMO-record bit exactness plausible because no ORCA2 record exists (`docs/ice/fidelity/si3_r14_o6_receipt.md:252-255`). | No O6 production record. |
| O7 admitted record | Unmodified production deck, 2 steps | All 91 scored leaves at each of `kt=1,2` are 0 ULP, including composed live surface stages (`docs/ice/fidelity/si3_r13_o7_admission_receipt.md:3-11,64-90`). | Raw meteorology and computed products stop after step 2. |
| O8--O16 master | Unmodified production deck, 100 steps | Every thermodynamic boundary and the O16 exchange/tau boundary is registered; restart pairs are byte-identical (`docs/ice/fidelity/si3_r18_o8_o16_master_preregister.md:15-24,40-58`). | No full pre-dynamics ice-step-entry carry, Prather moments, dynamic stress carry, or raw O7 meteorology. |

The master schema makes the gap mechanical rather than inferential. Kind 21 is
only the thermodynamic `STATE` tuple immediately before category work; kinds
18/15/16/19/20 cover pre-aggregation, flux, and stress boundaries. Its complete
projection registry contains no step-entry or raw-meteorology kind
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_gate.py:93-100,136-238`).
O8--O12, O13, O14, and O15/O16 nevertheless remain individually admitted at
0 ULP over all 100 steps (`docs/ice/fidelity/si3_r21_o8_o12_score_receipt.md:3-13`,
`docs/ice/fidelity/si3_r22_o13_receipt.md:3-8`,
`docs/ice/fidelity/si3_r23_o14_score_receipt.md:3-15`, and
`docs/ice/fidelity/si3_r24_o15_o16_receipt.md:3-19`). This audit retracts none
of those operator verdicts; it says only that their separate exact-input views
cannot be relabelled as O17 composition evidence.

## Frozen O17 score contract

The design row now freezes the full O17 selector card. In particular,
`core2_precipitation_closure=False` is explicit for the NEMO-comparison path;
the library default is `True` and its source says comparison configurations may
disable it (`packages/ice/legoesm/ice/config.py:524-529`). Dynamics and
thermodynamics are explicitly live in the O17 `SI3ORCA2StateConfig`; the current
state carrier otherwise defaults both flags off for the O1 overlay
(`packages/ice/legoesm/ice/si3_orca2_state.py:92-131`).

One production program must produce two non-interchangeable reports:

1. `INDEPENDENT`: initialize once from the O17 initial state, advance all 100
   steps with live O7 products, and compare every recorded field at every step.
2. `GIVEN-NEMO-ENTRY`: for each step, replace the complete step-entry state and
   live-forcing inputs with that step's NEMO record, execute exactly one same
   production step, and compare every recorded field. No output of this label
   may seed `INDEPENDENT`.

The local bar is exact bits, including signed zero. The trajectory bar is the
already frozen pointwise normalized `1e-15`. The first normal-range non-bit row
is `DEBT` and cites its first admitted owner. The window is all 100 steps; the
scored set is every recorded field at every step; the acquisition subset is
exactly complete step-entry state, live O7 inputs and products, named call-order
boundaries, exchange outputs, and restart. None may be sampled or relaxed.

## O17/C4 differentiability disposition

The design now requires CPU binary64 `jax.value_and_grad` for the exact one-step
production entry, with an unweighted sum of all floating prognostic and exchange
output leaves as the scalar reachability objective. It publishes shape, dtype,
finite, NaN, Inf, and nonzero counts for every floating input leaf; integer and
Boolean scheduling/routing leaves are reported as literal non-differentiable
inputs. Category participation, CFL/sweep selection, limiter and donor routing,
`min`/`max`/`where` switches, fixed iteration counts, convergence masks, category
search, and restart/write schedules are the literal non-differentiable
statements. The primal output must retain every forward bit unchanged. GPU
memory and multi-step tape budgeting are explicitly later work.

No count table is published as a measurement: there is no O17 callable to
compile. `step-entry floating leaves: UNMEASURED`, `live-forcing floating
leaves: UNMEASURED`, and `finite/NaN/Inf/nonzero counts: UNMEASURED`. Running
`value_and_grad` on only O15/O16 would be a different program and is refused.

## Decisions, controls, and review

- Window: fixed at 100 steps; no decision.
- Bars: exact local and fixed `1e-15` trajectory; no decision.
- Scored set and record subset: fixed above; no decision.
- AD scalarization: fixed as a mechanical unweighted output-leaf sum; it is not
  a scientific objective or a new bar.
- Acquisition tool: not authored because the explicit composition-entry stop
  fired first. After the entry exists, the acquisition tool must import the
  O5/O7/O8--O16 machinery and shared extent/read-set gates; it may not copy
  their logic.
- NEMO: not built or run.

`CONFIRMED`: the master/record mismatch and missing production entry are direct
schema/source findings. `PLAUSIBLE`: none promoted to an O17 physics verdict.

## Gates

- The required receipt-citation test file passed 45/45. The round receipt gate
  found 17 citations with zero unmapped citations, failures, or full-map audit
  failures; all 12 built-in controls fired. The cumulative default gate found
  274 citations with the same zero-failure result.
- The documented landing battery passed. The combined process reached the
  known per-process JAX compiler limit in the two-case stage-face-mask file, so
  that file and the following barotropic-state file were completed in fresh,
  serial processes as required by the repository instructions: both mask cases
  passed and the barotropic-state file passed 8/8. No test ran with `pytest -n`.
- The nine CI ratchet files reported 11 failed, 6,936 passed, and 2 skipped.
  Those are the established lane failures in hardcoded constants, inline
  coefficients, and parameter specifications; every named failure is in an
  untouched model, acquisition-fixture, or unit-test file. This round changes
  only the ladder design, this receipt, and the citation registry.
- All commands used CPU fp64 with `CUDA_VISIBLE_DEVICES=""`,
  `JAX_PLATFORMS=cpu`, and `JAX_ENABLE_X64=1`. No NEMO or GPU command ran.

independent review: operator (Claude)

## Resume 6: the operator --run refusal

Review d was LAND; RUN from both reviewers. User DECISION 201 keeps O17
scoring eager. The operator's --run at `2b1921abc4` then refused before
NEMO started. Master preflight steps 1-10 ran, and then step 10's stub
driver stopped with "REFUSE: stub run needs the real scorer report
O17_STUB_SCORE_JSON". Only the wrapper's --preflight-only path exported the
two real-scorer reports. The --run and --admit-existing branches called the
master, whose own full preflight runs that stub driver, without them. The
--run control flow had never been run end to end (checklist item 15).

The check that refused is unchanged. The changes:

- --run and --admit-existing first build the 2-step writer-shaped ORCA2 record
  from the model and score it with the real scorer: identical passes; the
  1-ULP `t_si` and signed-zero plants fail. They export those two reports,
  then call the master. --run goes through one shared entry, the same one
  the operator uses.
- The master refuses a missing report with the same message at the start of
  its preflight, instead of after steps 1-9.
- `O8O16_STOP_BEFORE_BUILD=1` stops the master's --run after its full preflight
  and its RAM and disk gates, and before any evidence directory, build or
  NEMO. The stub driver's own children never inherit it.
- --preflight-only now runs the wrapper's own --run branch: same environment,
  same branch, same master --run entry, a clean committed tree, and the stop
  above. Its evidence path is under the preflight's scratch. It also runs a
  plant that repeats `2b1921abc4`: the shared entry without the reports. The
  plant must be refused with the stub-score message. Measured, it is refused
  in seconds with that message.
- A static test pins the supply on both branches, the plant, the dry stop's
  position after the gates and before any build, and the doubled refusal.
  Run at `2b1921abc4`, it fails at its first assertion: the --run branch had
  no score supply.

Verification from a clean clone of `d28bccc16b`, TMPDIR under the round's
work directory: `run.sh --preflight-only` exit 0 in 3525 s. Its last lines
were `PASS_100_STEP_STUB_DRIVER_RETENTION`,
`PASS_EXISTING_CFG_DIGESTS_UNCHANGED`, `PREFLIGHT_OK: no real NEMO integration
was run; worst-case master 140965556460 bytes (131.28 GiB)`,
`DRY_RUN_STOPPED_BEFORE_BUILD_AND_NEMO: <scratch>/si3_rounds/round0/oracle_o17_dry`
and `PASS_O17_UNIFIED_MASTER_PREFLIGHT_AND_NUMERICAL_PLANTS`. No
`ORCA2_SAS_ICE_O17_REF` or `_REC` build exists, and nothing was written under
the real `si3_rounds/`. Other results at that commit: the master acquisition
file had 28 passed; the O17 static tests, 9 passed. The O17 unit file was not
re-run, because nothing it imports changed. No NEMO command ran.

## Resume 5 review-HOLD closure

This section is authoritative for the review of `6e02036e3f`. No NEMO
command was run; only the stub-binary preflight. All numbers are CPU
binary64 (`CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`),
and every reviewer item was reproduced on the parent before any change.

### First correction: the ORCA2 rank grid is 184 x 152, owned 180 x 148

The parent's scorer self-test, byte budget and resource probe all used
186 x 153 with a 182 x 149 interior. The admitted deck log gives
`Ni0glo = 180`, `Nj0glo = 148` and `nn_hls = 2`
(`scripts/validate/ocean_fidelity/testcases/nemo_si3_o7_orca2_sas_acquisition/ocean.output.excerpt:40-46`),
so `jpi x jpj = 184 x 152` and `A2D(0) = 180 x 148`. The round-18 master
records carry exactly those headers (kind 20 `utau` (184,152), `taum`
(180,148)). The reviewers' `(182,149,5)` against `(186,153,5)` is the same
defect measured at the parent's wrong extent. Every Resume-5 number uses the
real grid.

### Items 1-10

1. **Scorer extents.** Reproduced on the parent with a writer-extent kind-22
   record at 184 x 152: `_entry_state` raised `TypeError: mul got incompatible
   shapes for broadcasting: (180, 148, 5), (184, 152, 5)`, and its
   `_align_owned` scored an owned `t_si` record against the identical
   full-halo model array as `(different, max_relative) = (139840, inf)`.
   Every record field now drives its own extent. Owned (`A2D(0)`) entries are
   padded with zero, the lane's `t_si` halo convention, and fed to the
   full-halo state; owned kernel arguments take the owned slice of any
   full-halo record; a comparison with one owned operand restricts both to
   the owned interior, otherwise it compares the full arrays
   (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:172-191`
   and `:342-367`).
   The same identical-value comparison now gives `(0, 0.0)`. Padding cannot
   reach an owned output: the unit test re-runs step 1 with every halo `t_si`
   set to 123.0 instead of the padded 0.0 and the GIVEN-NEMO-ENTRY score of
   the owned records still passes bit for bit. Two parts of the finding are
   refuted by the source. `snwice_fmass` is allocated `(jpi,jpj)`
   (`src/OCE/SBC/sbc_ice.F90:125`), not on `A2D(0)`; and `snwice_mass`,
   `snwice_mass_b`, `snwice_fmass` are updated on `DO_2D(nn_hls, nn_hls,
   nn_hls, nn_hls)` (`src/ICE/iceupdate.F90:189-194`), so they stay full-grid
   comparisons. The rewritten reader also exposed a second parent defect that
   the TypeError had hidden: it passed `ssh_m` as the aEVP `ssh_t`, which the
   ORCA2 solver refuses because it builds that field itself from
   `ice_var_sshdyn(ssh_m, ...)` (`src/ICE/icedyn_rhg_evp.F90:268`); the
   reader now passes `None`.
2. **Boundary 27: refuted.** In the real patched and preprocessed source,
   `CALL o17_state(27)` is two lines after `CALL ice_thd( kt )`
   (`icestp.f90` 264 and 266), and `ice_thd` itself calls
   `ice_cor( kt , 2 )`, ages `oa_i` and links the halos before it returns
   (`src/ICE/icethd.F90:183-201`). Kind 27 is therefore the state after
   `ice_cor(kt,2)` and the tail, which is `trace.finalize.correction.state`.
   `trace.thermodynamics` is the category-stage tuple from before remapping,
   open-water growth and `ice_cor(kt,2)`; it has no `t_si` or snow-mass
   member. The mapping is unchanged; a plant that points kind 27 at another
   boundary fails the identical record.
3. **Halo thermodynamics.** Parent, 7 x 7 mixed card: 7 halo columns were
   gathered in each of the five categories, they saw `t_bo = 0 K`, and 35 halo
   `t_si` values were overwritten. The gather predicate is now
   `a_i > epsi10` on owned columns only, NEMO's `DO_2D(0,0,0,0)`
   (`src/ICE/icethd.F90:128-135`,
   `packages/ice/legoesm/ice/si3_o17.py:395-398` and `:484-488`); after it,
   0 halo `t_si` values change.
   NEMO's later halo refresh is the existing tail link
   (`src/ICE/icethd.F90:188-190,201`), which the test checks as a fixed point
   of `lbc_lnk` on `a_i`, `v_i`, `oa_i`, `t_su`, `e_i`, `u_ice` and `v_ice`.
4. **One source-derived byte budget.** The parent budgeted the private tau
   record as three full fields at 186 x 153 (683,244 bytes) and 964,823,300
   bytes per step. The layout is now derived from source: each writer `put`
   resolves through the shared allocation catalog over the real ORCA2 ppsrc,
   and each writer dummy is followed to the observer patch or writer call that
   passes it and then to the caller's NEMO declaration. That gives `tmod_io`
   (184,152) with `utau_oce` and `vtau_oce` (180,148)
   (`src/ICE/iceupdate.F90:55`), raw `emp` full and raw `qsr` owned, and
   `t_si` owned (`src/ICE/ice.F90:451`). Per kind, the tau record is 650,236
   bytes; one step is 948,127,092 bytes; the exact 100-step maximum is
   94,812,709,200 bytes; with the 140,965,556,460-byte O8-O16 master that is
   235,778,265,660 bytes and one 4% disk budget of 239,462,302 KiB. The
   wrapper and the master read the calculator's `--shell` line and the gate
   imports the same function.
5. **Streaming scorer.** Parent, on the 3-step writer-shaped ORCA2 record
   below: `_record_index` alone peaked at 1,140,624, 2,067,036 and
   2,988,912 KiB for 1, 2 and 3 steps, so it grows about 0.92 GB per step,
   or about 92 GB for 100 steps before any model evaluation. The scorer now
   indexes file names only; it loads one step's 13 records, scores them and
   drops them. The unit test proves this by counting one load per step. A
   first 3-step ORCA2 run still grew, with resident set 12,357,240,
   15,686,332 and 18,205,892 KiB after steps 1-3: each step's 1.29 GB trace
   (measured with `jax.eval_shape`) stayed referenced into the next step. The
   loop now keeps only the 0.27 GB INDEPENDENT state across steps and samples
   the resident set after `gc.collect()` and `malloc_trim(0)`. Re-measured at
   `cd5fd47d69`, the live resident set still grew. Scoring the same record
   twice in one process gave 3,841,044, 5,568,464 and 6,917,784 KiB on the
   first pass, then 7,580,020, 8,674,820 and 9,739,944 KiB on the second.
   Over the same six steps, JAX's eager per-primitive executable cache held
   497, 1140, 1782, 1886, 2528 and 3170 entries, every one a miss: the
   repeated steps 2 and 3 added 642 new entries again. The growth is that
   cache, which JAX bounds only by a 4096-entry LRU (`jax/_src/util.py` 263,
   `jax/_src/dispatch.py` 100). It is not data the scorer keeps. Which
   primitive rebuilds its cache key on every call is not identified. The
   scorer now clears JAX's caches after each eagerly scored step; the final
   per-step resident set is in the verification section below. The first run
   scored the identical 3-step record PASS under both labels, with 0
   differing values. INDEPENDENT entries at steps 2 and 3 were not bitwise
   equal to the record entries (the record carries owned `t_si` only), so
   they ran as separate evaluations and still matched every scored owned
   value bit for bit. Measured record bytes per step were 946,784,256 at step
   1 (no delayed CFL, no icedyn masks yet) and 948,127,092 at steps 2-3,
   which is the calculator's per-step maximum.
6. **Compiled is not bitwise equal to eager; the scorer stays eager.** One
   production-card step (100 aEVP subcycles) on the 184 x 152 grid with 3000
   ice-covered, 20,700 ice-free and 2940 near-empty owned columns: eager
   519.8 s, peak RSS 7,544,464 KiB; compiled 583.5 s compile plus 74.5 s
   execute, peak 27,284,040 KiB. 62 of 1209 leaves differ. Most differences
   are signed zeros from two XLA rewrites, each reproduced in isolation:
   `x * convert(mask)` becomes a select and stores +0 where NEMO's
   multiplication by its REAL 0/1 mask stores -0
   (`src/ICE/icedyn_rhg_evp.F90:189-192,459-461,822-824`), and an add into a
   compile-time zero accumulator is folded away, storing -0 where NEMO's
   runtime +0 accumulator stores +0. But 47 `delta` values and the ridging
   fluxes they feed (32-36 values in `wfx_dyn`, `sfx_dyn`, `hfx_dyn`,
   `wfx_snw_dyn`, then `wfx_snw`) differ in value. PLAUSIBLE: fused or
   re-associated arithmetic in the deformation kernel, which has no
   source-rounding barrier. Eager scoring costs 519.8 s per step: 100 steps
   take 14.4 h when INDEPENDENT stays bitwise on the NEMO entry, because a
   bitwise-identical entry reuses the GIVEN-NEMO-ENTRY result, and up to
   199 steps, 28.7 h, after it diverges. The scorer's own 3-step runs took
   4,224-4,476 s for 5 evaluations including record I/O, alongside other jobs,
   which is about 880 s per evaluation. At that rate the bound is 24-49 h. All
   of this runs after the acquisition clock stops.
7. **Preflight runs the real scorer.** The scorer self-test that compared zero
   arrays is deleted, and so is the stub driver's literal PASS report. The O17
   preflight now builds a 2-step, writer-shaped record at 184 x 152 from the
   model, with owned extents where NEMO has them, and runs the real scorer on
   it: the identical record passes; a 1-ULP change in step-2 exit `t_si`
   fails, and so does a +0 to -0 flip in step-2 exit `wfx_bog`, each failing
   at its planted field. The stub driver hands that real report to the
   driver, and the preflight now also runs the stub driver with the real FAIL
   report; it refuses, and it refuses for that reason. The driver parses the
   score as JSON: the old `grep '"status": "PASS"'` matched a nested label
   status, so a FAIL report whose INDEPENDENT label passed would have been
   admitted. The preflight's final lines are quoted in the verification
   section below.
8. **Code-revert plants.** Every plant now changes code, not results:
   monkeypatched production functions or a skipped call. The plants are
   `t_bo` (the freezing-point call dropped), `u_ice_b` (the store_fields
   velocities not passed to aEVP), `a_i_b` (the second store_fields skipped),
   the thermodynamic tail call skipped, the tail's halo links skipped, the
   full-grid gather (item 3), the scorer without padding or without owned
   alignment (item 1), and kind 27 pointed at the wrong boundary (item 2).
   Each fails the same production assertions the unplanted step passes;
   the failing line of each is quoted in the verification section below. The
   checks are pinned digests of the
   eager surface boundary and full-step scorer projection, a reference
   evaluation of the `t_bo` statement, the `src/ICE/icethd.F90:185` ageing, the halo
   fixed points, and halo `t_si`.
9. **Signed-zero canonicalisation removed.** NEMO assigns `zmsk` as a REAL 0
   or 1 (`src/ICE/icedyn_rhg_evp.F90:189-192`). It multiplies the stress
   update by it (`:459-461`), stores the linked stresses unchanged
   (`:838-843`), and forms `pdivu_i` as the product with `zmsk`
   (`:822-824`). None of these canonicalises a zero, so a negative bracket
   stores -0. Its only literal +0 there is the `ELSE` of `pdelta_i`
   (`:830-831`), and the deformation kernel's `SQRT(...)*mask` cannot produce
   -0. On the 7 x 7 card, eager raw values have 31 -0 in `stress2` and 2 in
   `divu`; NEMO's arithmetic stores them. Both canonicalisations are removed.
   The same class at the end of the step, which cited the step-entry
   `diag_set0` (`src/ICE/icestp.F90:438-464`), is also removed: eager raw
   final fluxes had no -0 on the card, so that removal changes no eager bit
   there. The canonicalisation had been hiding item 6.
10. **Non-finite values fail.** The parent's Python `max` returned 0.0 for
    `[0.0, nan]`, so its INDEPENDENT test passed. Every compared field now
    counts non-finite values, and any count fails the step under both labels.
    A planted NaN in step-2 exit `qns` fails INDEPENDENT.

### Choices

- Scorer execution stays eager, as item 6 instructs when the compiled step is
  not bitwise. ASKED (item 6).
- The scorer clears JAX's caches after every eagerly scored step to bound
  its memory. ASKED (item 5).
- The halo pad value is zero, the lane's existing `t_si` convention, and the
  test shows it cannot reach an owned output. ASKED (item 1: "the way the
  lane's state convention requires").
- The end-of-step flux canonicalisation was removed with the item-9 pair. It
  is the same unmeasured class with no NEMO counterpart, and it changes no
  eager bit on the card. UNASKED; offered for revert.
- The synthetic preflight writes +0 into writer fields the scorer never
  reads, to keep the writer's exact size and order. UNASKED; offered for
  revert.

Finding, not fixed: the preregistration says every recorded field is scored
at every step, but the scorer compares only a named subset at kinds 23-31,
plus the complete exit and tau records.

`CONFIRMED`: items 1-5 and 7-10 are measured as stated. Item 6: the
compiled/eager mismatch census and both signed-zero mechanisms are CONFIRMED;
the cause of the 47 value differences is PLAUSIBLE.

### Decision needed

1. Make the compiled O17 step bitwise equal to eager, which is IEEE per
   operation like NEMO's unfused gfortran build. That means a barrier on the
   0/1 mask operands in the shared aEVP and deformation kernels, runtime
   (not compile-time) zeros for the `diag_set0` accumulators, and source
   rounding in the deformation arithmetic. Then re-measure O5, lane 3 and O6,
   and switch scoring to the compiled step (about 75 s per step after three
   compiles of about 580 s). **Pick.** It changes shared certified kernels,
   so it is the user's decision.
2. Keep eager scoring (14-49 h after NEMO stops) and leave production JIT
   as it is. The real record can be acquired and scored now under either
   option.

## Resume 5 final verification

Run from a fresh clone under the round scratch directory: the full battery at
`cd5fd47d69`, then the O17 unit file and the preflight again at `7f8e3f99f7`
(the only later code change is the scorer's per-step cache clear), on
CPU binary64 with `CUDA_VISIBLE_DEVICES=""`, `JAX_PLATFORMS=cpu` and
`JAX_ENABLE_X64=1`. Every pytest command named its file, ran serially in a
foreground child bounded by `timeout`, and used no xdist worker. No NEMO
command ran.

- O17 unit file at `7f8e3f99f7`: `15 passed, 1 xfailed in 3312.26s (0:55:12)`;
  at `cd5fd47d69`: `15 passed, 1 xfailed in 3263.16s (0:54:23)`. The xfail is
  the strict compiled-equals-eager test (item 6). The code-revert plants
  failed at these lines. `t_bo`: `Arrays are not equal | t_bo
  icestp.F90:136-138`. `u_ice_b` and `a_i_b`: `surface-boundary digest ...
  != pinned 7e6ff54d...`. Thermodynamic tail: `icethd.F90:185 ageing after
  ice_cor(kt,2)`. Tail halo links: `a_i halo not refreshed
  (icethd.F90:188-190)`. Full-grid gather: `halo columns ran thermodynamics
  (icethd.F90:130 DO_2D(0,0,0,0))`. The scorer plants (no padding, no owned
  alignment, wrong kind-27 boundary, 1 ULP, signed zero, NaN) all fail
  inside the passing scorer test. After the run, `git status --porcelain` in
  the clone was empty.
- `tests/ocean/fidelity/test_nemo_si3_*.py`, 17 files, one process each: 206
  passed. The O4 Prather file passed its 8 forward cases with only its
  full-step AD case deselected (the Decision 194 boundary). The scalarmath-v2
  file had 4 passed and the pre-existing external-state failure `A MY_SRC is
  not verbatim`.
- `tests/ice/unit/test_si3_*.py`, the other 15 files: 397 passed, 2
  deselected by the repository markers (O5 and O6 files), 1 xfailed. The only
  failure is the pre-existing
  `test_production_path_plants_fire[sum_first_category]` (`test_si3_surface.py`
  line 531, `assert [] != []`).
- O8-O12 master scorer: `30 passed in 724.39s`. O8-O16/O17 master
  acquisition: `28 passed in 498.28s`. O5: `17 passed`. O7: `19 passed`.
- Landing battery, one process per file: receipt citations 45, private
  imports 6, TKE terms 71, NEMO recipe 25, freshwater closure 16, stage-face
  masks 2, prognostic barotropic state 8; all passed.
- Ten ratchet files: `11 failed, 13183 passed, 4 skipped`. The 11 are the
  established set, none in a Resume-5 file. Three are hardcoded-constant rows
  (`test_si3_o13_itd_remap.py`, `test_si3_thd_do.py`, the O4 stub writer).
  Five are inline-coefficient rows (`bitz_lipscomb.py`, `si3_orca2_state.py`,
  `geothermal.py`, `mle.py`, `mle_latlon_cgrid.py`). Three are parameter-spec
  rows (`si3_orca2_state.py`, `mle.py`, `shortwave_penetration.py`).
- Item 5, the scorer's own memory, at `7f8e3f99f7`: the 3-step record scored
  twice in one process (six steps) left a live resident set of 1,567,192,
  1,611,868, 1,620,208, 1,529,428, 1,671,252 and 1,628,668 KiB after the
  successive steps. That is flat, with JAX's eager cache emptied every step.
  Peak RSS was 15,657,944 and 16,754,884 KiB for the two passes, which took
  4,416 and 4,565 s; both passes scored PASS. Before the cache clear the
  same two passes took 4,396 and 4,475 s, so clearing costs no measured
  time.
- Receipt citation gates at `0cada202c8` (clean clone): per-receipt from
  "Resume 5 review-HOLD closure", 62 citations, and default cumulative, 274
  citations; both PASS with 0 unmapped citations, 0 failures and 0 map
  entries failing the full audit; 12 of 12 built-in controls fired.
- Memory-rule disclosure: the O17 unit file peaks at about 25 GB because of
  the 7 x 7 full-step compile behind its compiled-versus-eager census. It ran
  alongside the preflight and the RSS probe, about 60 GB combined with over
  900 GB available. This breaks the standing note's "over 20 GB runs alone"
  rule; no command approached 100 GB.
- Read-only acquisition preflight, `run.sh --preflight-only`: exit 0 in 3591 s
  at `7f8e3f99f7` and 3762 s at `cd5fd47d69`. In the `cd5fd47d69` run the
  synthetic part ran 2 eager steps at 184 x 152 in 2939 s with
  peak RSS 15,525,592 KiB. The identical record passed under both labels with
  0 differing values. The 1-ULP plant (step-2 exit `t_si` at owned index
  (90,0,0)) failed GIVEN-NEMO-ENTRY at `step_exit.t_si` with 1 differing
  value. The signed-zero plant (step-2 exit `wfx_bog` at (0,1)) failed at
  `step_exit.wfx_bog`. Final lines: `PASS_100_STEP_STUB_DRIVER_RETENTION`,
  `PASS_EXISTING_CFG_DIGESTS_UNCHANGED`, `PREFLIGHT_OK: no real NEMO
  integration was run; worst-case master 140965556460 bytes (131.28 GiB)`,
  `PASS_O17_UNIFIED_MASTER_PREFLIGHT_AND_NUMERICAL_PLANTS`.

The acquisition tool writes the unified record within the derived
239,462,302 KiB disk budget and the 21,600 s NEMO wall clock, then scores it
with the real eager scorer.

## Resume 4 final verification

All numerical verification used CPU binary64 with `CUDA_VISIBLE_DEVICES=""`,
`JAX_PLATFORMS=cpu`, and `JAX_ENABLE_X64=1`. Every pytest invocation named a
test file, ran serially, used no xdist worker, and was bounded by `timeout`. No
NEMO command ran.

The complete O17 unit file passed 8/8 in 978.57 s. The complete
`tests/ocean/fidelity/test_nemo_si3_*.py` sweep produced 206 passing tests. Its
two already-disclosed non-green boundaries were unchanged: the O4 file passed
all eight forward cases before the full-step AD compile reached the 2400 s file
cap without a compiler diagnostic, and scalarmath-v2 returned four passes plus
the external-state failure `A MY_SRC is not verbatim`. The O8--O12 master
scorer passed 30/30 in 833.82 s; the extended O8--O16/O17 acquisition suite
passed 28/28 in 707.08 s.

The complete `tests/ice/unit/test_si3_*.py` sweep produced 404 passes, two
deselections, one skip, and one expected xfail. Its sole failure is the
pre-existing `test_production_path_plants_fire[sum_first_category]` at
`test_si3_surface.py` line 531 (`assert [] != []`); that file otherwise had 22
passes. O12 passed 81/81, O6 passed 28 selected tests with one deselected, and
the new O17 file passed 8/8 including the exact eager/JIT full step.

The landing files returned: receipt citations 45/45 after retaining historical
citation spellings and registering 44 Resume-4 anchors; private imports 6/6;
TKE terms 71/71; NEMO recipe 25/25; freshwater closure 16/16; stage-face masks
2/2; and prognostic barotropic state 8/8. The Resume-4 per-receipt gate and the
default cumulative gate both report no unmapped citations, citation failures,
or full-map audit failures; every built-in control fires.

The nine repository ratchets collected 6,958 cases and returned 11 failed,
6,945 passed, and 2 skipped. All eleven are pre-existing and outside this
round's changed model/tool paths: hardcoded-constant rows in
`test_si3_o13_itd_remap.py`, `test_si3_thd_do.py`, and the O4 stub writer;
inline-coefficient rows in `bitz_lipscomb.py`, `si3_orca2_state.py`,
`geothermal.py`, `mle.py`, and `mle_latlon_cgrid.py`; and parameter-spec rows
in `si3_orca2_state.py`, `mle.py`, and `shortwave_penetration.py`.

The acquisition wrapper is ready to write the unified record and run the real
composition scorer within its derived disk and acquisition-clock budgets. A
successful score remains operator evidence and has not been claimed here.

## Resume 4 review-HOLD closure

This section is authoritative for the review of `999827602052`. No NEMO
command was run. Every reviewer item below was treated as a hypothesis and was
measured on the reviewed parent before its correction was accepted.

### Measured defects and literal corrections

1. **Ice-surface temperature scatter.** On the parent, the full step changed an
   ice-free column from the `diag_set0` value 273.15 K to
   266.89337065264044 K. NEMO initializes every category to `rt0` but scatters
   the diffusion result only at gathered ice points
   (`src/ICE/icestp.F90:485-498`, `src/ICE/icethd.F90:121-140,518`). O17 now
   retains the entry `t_si` wherever the category gather predicate is false
   (`packages/ice/legoesm/ice/si3_o17.py:933-940`). The corrected mixed-regime
   measurement is exactly 273.15 K in every inactive category.

2. **`ice_thd_1d2d` gather and units.** Before correction, the two-column
   complete-step probe had six non-finite final flux leaves and three
   non-finite final output leaves. The parent passed extensive J/m2 `e_i/e_s`
   directly into the J/m3 category kernels and executed them at `a_i=0`.
   NEMO's active predicate and conversion are
   `src/ICE/icethd.F90:121-141,418-435`, and its selected scatter is
   `src/ICE/icethd.F90:438-526`. The production entry now applies that same
   predicate, source-rounded layer scaling, safe inactive dummies, and
   selected scatter (`packages/ice/legoesm/ice/si3_o17.py:395-473`,
   `packages/ice/legoesm/ice/si3_thd_pnd.py:865-874,955-1006`). Repeating the
   complete step produced zero non-finite trace, flux, or output leaves.

3. **Production dynamics selector.** The shipped deck says
   `ln_dynALL=.true.` (`cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_ref:49-54`). The
   parent nevertheless selected the `np_dynADV2D` helper path, which executes
   `ice_var_zapsmall` immediately after advection. NEMO's selected path orders
   rheology, advection, ridging, then `ice_cor(kt,1)`
   (`src/ICE/icedyn.F90:128-135`), with the zap inside that correction
   (`src/ICE/icecor.F90:111-116`). `si3_dyn_orca2` now has an explicit path
   selector: its historical default preserves every admitted O4 caller, while
   O17 selects `np_dynALL` and returns the unzapped advection state for the
   existing ridging/correction calls
   (`packages/ice/legoesm/ice/si3_orca2_state.py:3380-3427,3583-3663`,
   `packages/ice/legoesm/ice/si3_o17.py:660-683`). The near-empty plant at
   `tests/ice/unit/test_si3_o17.py:301-305` fails if that category is zapped at
   the advection boundary.

4. **Snow/ice mass carry.** The parent complete step measured entry and exit
   `SI3State.snwice_mass` both as 1211.504 while O16 had computed
   1212.47614344. NEMO carries all three updated mass fields at
   `src/ICE/iceupdate.F90:187-194`. The returned step state now takes
   `snwice_mass`, `snwice_mass_b`, and `snwice_fmass` from O16
   (`packages/ice/legoesm/ice/si3_o17.py:1115-1120`), so the independent scorer
   advances them rather than reusing step 1. The full-step test proves the state
   and O16 output are bit-identical and differ from the entry
   (`tests/ice/unit/test_si3_o17.py:278-285`); the stale-carry plant fails at
   lines 306-310.

5. **Fold extent.** The parent hard-coded `ldfull=True` at every thermodynamic
   tail link. The parsed one-rank ORCA2 card has
   `fold.nfd_full_when_absent=True`, while the planted absent-full-fold card is
   `False` and changes the U-link bits. The tail now uses that card value at
   each link (`packages/ice/legoesm/ice/si3_o17.py:501-543`), corresponding to
   NEMO's omitted optional `ldfull` at `src/ICE/icethd.F90:188-201`. The wrong
   fold plant is rejected at `tests/ice/unit/test_si3_o17.py:215-224`.

6. **Compiled arithmetic.** The parent eager/compiled comparison differed in
   six `oa_i` entries by one ULP, with maximum absolute difference
   2.2737367544323206e-13. The `src/ICE/icethd.F90:185` multiply and add now use the
   lane's `nemo_source_round` at both NEMO statement boundaries
   (`packages/ice/legoesm/ice/si3_o17.py:489-499`). Every former
   `atol=1e-12` comparison was removed. The fast composition test and the full
   scorer-visible projection now compare binary64 views exactly, including
   signed zero (`tests/ice/unit/test_si3_o17.py:29-48,157-158,200-208,264`).
   The focused fast gate passed in 122.57 s and the complete mixed-regime gate
   passed in 1107.04 s.

7. **Complete-step coverage and plants.** The new 7x7 fixture has a 3x3 owned
   tile with separated active-ice, open-water, and nonzero-at-or-below-`epsi10`
   columns, plus nonzero ice and ocean velocities
   (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_ad_probe.py:55-90`).
   One test executes the entire step eagerly and compiled, requires exact bits
   on every scorer-visible state/boundary/output leaf, and checks every floating
   trace leaf is finite (`tests/ice/unit/test_si3_o17.py:230-285`). The named
   assertion line rejected each plant: `t_bo` line 172, `u_ice_b` lines
   176-179, `a_i_b` line 174, thermodynamic-tail omission lines 210-214,
   wrong `ldfull` lines 219-224, inactive `t_si` scatter lines 292-297,
   gather/scaling NaN lines 298-300, early zap lines 303-305, and stale mass
   lines 306-310. The pre-existing swap-order, missing-fold, and forcing plants
   remain numerical red controls at lines 181-198.

8. **Scorer extents and delayed scalar.** Before correction, comparing the
   model's `(182,149)` owned field with the writer's `(186,153)` full-halo field
   produced `(different,max_relative)=(28458,inf)`. The scorer now takes the
   owned slice only when those exact extent signatures differ
   (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:248-278,287-350`).
   The writer omits `cflice_delayed` while the delayed buffer is unallocated,
   matching `src/OCE/LBC/mpp_allreduce_generic.h90:147-152`, and the reader
   supplies `None` when it is absent
   (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_acquisition/o17_master_writer.F90:133-154`,
   `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:170-177`).
   At full ORCA2 shape the synthetic identical-record control passes; changing
   one owned zero to its next binary64 value fails with exactly one differing
   element and maximum relative error 2.220446049250313e-16.

9. **One derived acquisition budget and resource measurement.** The writer
   schema calculation uses `jpi*jpj=28,458`, A2D(0)=27,118, every field header,
   and every state/Prather/raw/tau record. It gives 964,823,300 bytes per step
   and an exact 100-step maximum of **96,482,330,000 O17 bytes**. Combined with
   the O8--O16 worst-case 140,965,556,460 bytes, one 4% budget is
   **241,158,010 KiB**; the extended wall budget is 21,600 s
   (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_record_budget.py:13-67`).
   The wrapper, manifest, measured-size check, and gate all derive those values
   from that one calculator. With `O17_EXTENDED=0`, the round-18 contract stays
   exactly 150,000,000 KiB and 14,400 s
   (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_acquisition/run.sh:24-43`,
   `scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o16_master_gate.py:677-692`).
   Scoring starts after the acquisition stop timestamp, so its wall time is not
   silently charged to the NEMO integration clock, while disk/RUN_COMPLETE
   admission still includes its report.

   One ORCA2-size, production-card step with 100 aEVP subcycles measured:
   eager execution 554.434314 s and peak RSS 7,747,388 KiB; compiled execution
   555.355544 s compile plus 80.127607 s execute (635.483151 s total), peak RSS
   22,514,800 KiB. The production scorer uses eager execution and publishes its
   own wall time and peak RSS in the score JSON
   (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:351-400`).
   This is below the operator's one-process 100 GB ceiling; no concurrent heavy
   process was started.

10. **Read-only operator preflight.** The final stub-only preflight completed in
    760.573 s. Its final lines were `PASS_EXISTING_CFG_DIGESTS_UNCHANGED`,
    `PREFLIGHT_OK: no real NEMO integration was run; worst-case master
    140965556460 bytes (131.28 GiB)`, and
    `PASS_O17_UNIFIED_MASTER_PREFLIGHT_AND_NUMERICAL_PLANTS`. It exercised the
    complete generated driver, retention/refusal/admit-existing flow, exact
    budget manifest, scorer controls, and numerical plants without launching
    NEMO.

Decision 194 remains the next round. No JAX-loop conversion, checkpointing, or
AD graph restructuring was implemented here.

## Resume 2 measurement and disposition

The probe now keeps every scheduler value outside the differentiated pytree.
The literal static inputs are `kt=1`, `jpl=5`, `nlay_i=10`, `nlay_s=5`,
`nn_hls=2`, `n_subcycles=100`, `category_count=5`, `ln_icedyn=True`, and
`ln_icethd=True`.  `jax.eval_shape` completes without converting a traced
integer to Boolean.  The floating pytree is an 8 by 7 full tile with a 4 by 3
owned region: eight owned columns contain ice in five categories, snow, and
ponds; four owned columns are open water; ice/ocean velocities, wind stress,
ocean stress, heat, freshwater, and precipitation forcings are nonzero.

The exact compiled CPU-fp64 `value_and_grad` program was run twice under the
required `timeout 3600`.  Both runs returned exit 124 without a compiler
diagnostic.  The resource-accounted current-source run measured 3607.36 s wall
including timeout reap and 18,061,120 KiB peak RSS (about 17.2 GiB).  The same
exact transform under `jax.disable_jit()` also returned exit 124 after 3604.83 s
and measured 17,710,772 KiB peak RSS.  Thus neither execution produced a
numerical gradient or a primal-from-gradient result; forward-bit equality for
that transformed program is unmeasured, not failed.

Trace-only accounting completed in 43 seconds. Argon's corrected recursive
count for the reviewed graph is 576,364 equations. `_thermo_category_loop`
owns 342,178 (the five O9--O11 category paths contribute about 191k and the O12
pond path about 151k), `si3_ice_thd_do` owns 53,490, and `si3_dyn_orca2` owns
48,325. The remaining named and inlined program is 132,371 equations. The
earlier attribution of 179,770 equations to a catch-all bucket is retracted:
those equations were nested beneath the thermodynamic owner. The reproducible
recursive/source-owner counter is
`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_jaxpr_count.py`.
The same counter on the Resume 3 source reports 577,982 total: 342,170 in the
thermodynamic loop, 53,489 in open-water thermodynamics, 48,325 in Prather
dynamics, and 133,998 remaining. It retains the reviewed `ea3e570` snapshot
above separately, so the Decision 194 measurement is not silently rewritten by
the Resume 3 carry/tail changes and their surrounding composition wiring.

The complete 206-leaf publication is
`docs/ice/fidelity/si3_r29_o17_ad_leaf_table.md`.  Shape and dtype are measured
for every floating entry/forcing leaf.  Finite, NaN, Inf, and nonzero gradient
counts are `UNMEASURED_TIMEOUT` on every row.  Consequently no leaf is called
zero and no zero reason is asserted.  The probe contains reason classification
for a completed run (overwritten live-state fields, masks/routing, disabled
landfast drag, inactive bulk-salinity Prather moments, and exact local zeros),
but publishing any of those as observed zero leaves would fabricate evidence.

The production dispatcher and frozen O17 card are wired, and the structural
preflight proves the admitted call order.  Its swapped-rheology/advection and
dropped-redistribution plants both fire.  The O4 forward/JIT exact cases passed
before only its final AD test reached its 1800-second timeout; O5 passed 17/17,
O7 passed 19/19, the O8--O16 master scorer passed 30/30, O6 passed 28 selected
tests (one deselected), O12 passed 80/80, finalization passed 8/8, and the new
O17 unit file passed 6/6.  These retain every certified component's forward
bits.

The nine repository ratchets report the established baseline: 11 failed,
6,936 passed, and 2 skipped.  The failures are the three pre-existing
hardcoded-constant rows (`test_si3_o13_itd_remap.py`, `test_si3_thd_do.py`, and
the O4 stub writer), five pre-existing inline-coefficient rows
(`bitz_lipscomb.py`, `si3_orca2_state.py`, `geothermal.py`, `mle.py`, and
`mle_latlon_cgrid.py`), and three pre-existing parameter-spec rows
(`si3_orca2_state.py`, `mle.py`, and `shortwave_penetration.py`).  O17 adds no
ratchet failure; its frozen deck literals carry explicit provenance comments.

They are not, however, a through-entry composition proof.  The retained O4,
O5, O7, and O8--O16 records are different projections/runs and do not contain
one complete O17 step-entry plus live forcing.  The composition scorer now
reports both `INDEPENDENT` and `GIVEN-NEMO-ENTRY` as `ACQUISITION_NEEDED`
instead of relabelling separate component scores as an O17 result.  A true
eager/JIT through-entry proof, and meaningful swap/drop numerical red scores,
require the unified O17 record named by the acquisition tool.

### Decision needed

1. Convert the O9--O12 category/layer/Picard Python loops to bit-preserving JAX
   control flow and place checkpoint boundaries between already admitted rung
   calls.  This changes traced operator structure but not intended forward
   arithmetic; every O4--O16 exactness test must remain 0 ULP. **Pick.**
2. Add stagewise custom VJPs around the admitted calls, preserving the current
   forward graph but taking on hand-maintained reverse rules.
3. Replace the full-step objective with separate per-stage gradients.  This is
   a different program and is not recommended.

No option was applied in Resume 2 because the operator explicitly prohibited
operator-structure changes.  Until option 1 or 2 is approved, the C4 numerical
reachability table remains measured as timed out rather than passing.

## Resume 3 review closure

This section is authoritative for the review of `ea3e570732`. No NEMO command
was run. The O2 production boundary again raises the original
`NotImplementedError("ice_update_flx" ...)` class/message contract; both
certified O2 gate cases pass unchanged. O17 still refuses the incomplete O2
forcing as a production full-step input, so restoring the exception contract
does not invent any forcing or weaken either gate.

The production step now includes every cited tail operation rather than taking
it as forcing: `t_bo` is formed at `src/ICE/icestp.F90:136-138`;
`u_ice_b/v_ice_b` at the first `store_fields` boundary
(`src/ICE/icestp.F90:151`, `src/ICE/icestp.F90:421-423`); `a_i_b/at_i_b` at the
second `store_fields` boundary (`src/ICE/icestp.F90:184`,
`src/ICE/icestp.F90:399-418`); `diag_set0` resets owned `t_si` at
`src/ICE/icestp.F90:488`; category unpack retains `t_si` at
`src/ICE/icethd.F90:163,518`; and the post-`ice_cor(kt,2)` ageing, T/U/V halo
links, category sum, and melted-edge velocity zeroing are transcribed from
`src/ICE/icethd.F90:183-201`. C-grid deformation now occurs between aEVP and
Prather advection, matching `src/ICE/icedyn.F90:132-134` and
`src/ICE/icedyn_rhg_evp.F90:791-827`.

The admitted live O9--O12 category loop now returns its carried module flux
state. Salinity's `sfx_res/sfx_bri` are retained from
`src/ICE/icethd_sal.F90:231,239,489,596,624`; diffusion's
`hfx_err_dif/hfx_dif` are carried from
`src/ICE/icethd_zdf_bl99.F90:828-831`; the already admitted category
aggregation is used directly. The five-snapshot replay arm is unchanged. The
live-arm unit test numerically proves all four carries across all five
categories.

The O17 unit gate executes one mixed open/ice, five-category, snow-and-pond
entry through the real `si3_o17_step` to the O7 surface boundary in eager and
compiled modes. It uses the explicit redistribution plant because O6 already
has its admitted eager/JIT numerical suite, and it stops before the 342k
thermodynamic category equations whose compilation Decision 194 reserves for
the next round. The separate live O9--O12 test executes all five category
carries, and the post-thermodynamic ageing/halo/velocity tail runs in eager and
compiled modes. Red controls numerically change the velocity pair presented to
Prather when rheology/advection are swapped, ORCA2 deformation when the fold is
dropped, and O7 heat flux when `frq_m` is halved; `ln_ice_embd=True` refuses.
The production dispatcher test calls the live composed entry, and the
record-reader test proves self-described extents and truncation refusal.
Removing any of those live arms therefore fails a numerical assertion rather
than an AST-list or `callable()` check.

The former four-component O17 driver is removed. The operator tool now invokes
the extended O8--O16 master exactly once. That master retains its passive
reference twin, then runs one instrumented, unmodified 100-step
`ORCA2_SAS_ICE` production trajectory. Per-step/per-rank O17 kinds 22--34 cover
entry, rheology, advection, redistribution, dynamic correction, thermodynamics,
exit, Prather entry/exit, flux-other, both raw bulk boundaries, and the private
stress exit. Records are self-describing and include full prognostic state,
all 55 Prather arrays, dynamic stress, category bounds, delayed CFL, raw O7
meteorology, precipitation, ocean/grid/mask forcing, all carried fluxes, O16
exchange fields, and all stress outputs. The manifest names exactly those
kinds. Production-flag FP-census and same-program gates cover every observer.

The scorer reads that one trajectory and calls `si3_o17_step` for every step
and rank. `GIVEN-NEMO-ENTRY` enforces bitwise equality (including signed zero)
at the named boundaries and exit; `INDEPENDENT` seeds only step 1 from NEMO and
enforces the fixed `1e-15` trajectory bar thereafter. A non-PASS score prevents
the top-level `RUN_COMPLETE` marker and exit zero. The read-only preflight uses
stub NEMO only for driver retention/control flow and separately executes live
numeric swap/fold plants. RAM, disk, and one global wall-clock gates remain.
The operator entry is:

`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_acquisition/run.sh`

Decision 194 selected option 1: bit-preserving JAX loop conversion with
checkpoints. It is deliberately not implemented in this round; it is the next
round's work. The corrected Argon ownership numbers above are the frozen
measurement motivating that decision.

### Resume 3 final verification

All verification used CPU binary64 with `CUDA_VISIBLE_DEVICES=""`,
`JAX_PLATFORMS=cpu`, and `JAX_ENABLE_X64=1`; every pytest command named its
test file, ran serially, and used no xdist worker. No NEMO command ran.

The complete `tests/ocean/fidelity/test_nemo_si3_*.py` sweep returned 206
passing assertions. The O8--O12 master scorer passed 30/30 in 703.01 s and the
extended O8--O16/O17 acquisition suite passed 28/28 in 493.97 s. The O4
Prather file completed its eight forward cases before the deliberately deferred
full-step AD compile reached the 1800-second cap, which is the Decision 194
boundary rather than a numerical red. The scalarmath-v2 file returned four
passes and the pre-existing external-state failure `A MY_SRC is not verbatim`;
neither that test nor its gate changed from `ea3e570732`.

The complete `tests/ice/unit/test_si3_*.py` sweep returned 404 passing
assertions, two deselections, and one expected xfail. O12 passes 81/81 after the
exact-input five-snapshot replay arm was kept compatible with its historical
mock entries while the live arm continues to require and carry
`hfx_err_dif/hfx_dif`; the old composer test and the new five-category live
carry test both pass. O17 passes 7/7, including the eager/JIT numerical
composition test. The only remaining red is the pre-existing
`test_production_path_plants_fire[sum_first_category]` in `test_si3_surface.py`
(23 passed, 1 failed, 1 xfailed); neither the test nor `si3_surface.py` changed
from `ea3e570732`, and the earlier O12 receipt already records the same inert
plant after `e96c1875f8`.

The documented landing files pass after the mechanical O12 citation reanchor:
receipt citations 45/45, private imports 6/6, TKE terms 71/71, NEMO recipe
25/25, freshwater closure 16/16, stage-face masks 2/2, and prognostic
barotropic state 8/8. The nine repository ratchets report the unchanged failure
set: 11 failed, 6,944 passed, and 2 skipped. The three hardcoded-constant rows
are `test_si3_o13_itd_remap.py`, `test_si3_thd_do.py`, and the O4 stub writer;
the five inline-coefficient rows are `bitz_lipscomb.py`, `si3_orca2_state.py`,
`geothermal.py`, `mle.py`, and `mle_latlon_cgrid.py`; the three parameter-spec
rows are `si3_orca2_state.py`, `mle.py`, and `shortwave_penetration.py`.

The unified acquisition's final read-only preflight exited zero after proving
the real preprocessed-source placement, patched bounds fixture, writer layout,
observer-free same program, 100-step stub-driver retention, unchanged cfg
digests, and independent admission. Its final line is
`PREFLIGHT_OK: no real NEMO integration was run; worst-case master 140965556460
bytes (131.28 GiB)`. A top-level `RUN_COMPLETE` remains impossible until the
real record has been scored successfully.

## Resume 1 implementation (operator ruling applied)

The earlier stop above is historical.  The operator confirmed that O17's one
new piece is the typed full-deck composition entry, not a new physical
operator.  Resume 1 therefore implemented that composition while retaining the
stop on any previously unadmitted physical kernel.

The dispatcher now accepts `SI3O17StepState` plus
`SI3O17LiveForcing`.  The explicit `SI3O17Card` writes the live O4--O16 deck,
including `ln_icedyn=True`, `ln_icethd=True`, 100 aEVP subcycles, five
categories, 10+5 layers, 10800 s, `nn_hls=2`, all surface selectors, and
`core2_precipitation_closure=False`.  The production call graph is:

1. O7 `blk_ice_1` and stress exchange;
2. O5 H79 strength and aEVP;
3. O5 C-grid deformation at the rheology exit;
4. O4 Prather advection, then O6 redistribution/rafting and `ice_cor(kt,1)`;
5. the remainder of O7 (`ice_alb`, `blk_ice_2`, `ice_flx_other`);
6. O8 frazil, O9--O12 category/pond work, O13 remap, and O14 open water;
7. O15 correction, O16 aggregation/flux update, and tau update.

No new physical formula was added.  The only supporting changes are production
wiring: the admitted deformation routine can select the admitted ORCA2 fold
exchange, and the admitted O9--O12 loop can carry its flux tuple live between
categories while preserving its old five-snapshot exact-record replay arm.

The read-only O17 preflight imports the O5, O7, O8--O16, read-set, and shared
extent tools.  It mechanically verifies the call graph above, the 100 x 10800 s
contract, both score labels, and the required complete entry/raw
meteorology/products/boundaries/exchanges/restart record set.  Both a swapped
O5/O4-order plant and a dropped-O6 plant fail.  The acquisition driver defaults
to this read-only preflight, writes its source/resource/retention/admission
manifest before an explicit operator run, and routes output only below
`si3_rounds/round<k>/`.

Resume 1 initially left the differentiability probe with an ice-free synthetic
state.  Resume 2 above replaces it with the active mixed-regime state and
records the bounded compiled/eager measurements and per-leaf disposition.

independent review: operator (Claude)

## DECISION 194 — traced O9–O12 loops

DECISION 194 (option 1 of the Resume 2 decision) converted the O9–O12
category, layer and iteration Python loops to JAX control flow with static trip
counts.  Every converted loop keeps its statements, their order and their
source-rounding barriers; only the iteration mechanism changes, and only when
the call is traced (`jax.jit`, `jax.grad`): an eager call still runs the Python
loop.  No default, configuration field, scheme selection or physics changed,
no knob was added, and the O17 composition module was not edited.

What was converted (read off the code):

- the O12 composer's `DO jl = 1, jpl` category loop (O9 + O10 + O11 +
  `ice_thd_1d2d(jl, 2)`) and the `pnd_LEV` category loop: one shared helper
  runs them as a `lax.scan` over the stacked category inputs with the body
  under `jax.checkpoint`, so reverse mode keeps only the category carry;
- the `ice_thd_dh` surface melt/sublimation and basal melt layer passes:
  static-trip `fori_loop`s over the stacked layer carriers;
- the shared `ice_var_vremap`/`snw_ent` old-segment loop: a `fori_loop`
  updating every new interface elementwise in the same old-segment order
  (O14 open-water growth uses the same helper);
- the salinity-4 harmonic-mean permeability: one `fori_loop` over the inner
  layer with all start layers carried together.

One barrier was added.  With the category body scanned, XLA contracted the
BL99 bottom conductive flux into fused multiply-adds in its two consumers (the
`ice_thd_dh` available bottom heat and the `hfx_err_dif` budget): 1–12 ULP in
the basal-growth fluxes and `hfx_err_dif` on the five-category fixture
(measured).  NEMO's object code does not fuse there (DECISION 158), so the
product is now materialised like its neighbours; the fixture is then 0 ULP.

Left as Python loops (small; inside an already-traced body or a few
statements each): the `ice_thd_dh` snow loops, the salinity-4 per-layer and
RJW velocity loops, the `ice_perm_eff` layer loop, the `ice_thd_pnd`
vanished-pond loop, the vremap cumulative sums, and the BL99 matrix rows
inside the existing Picard `fori_loop`.  No checkpoint was placed between the
O17 rung calls (that module is being edited by another round, and a measured
checkpoint variant compiled slower; see below).

### Measurements (CPU fp64, libm arm, probe tile unless stated)

Recursive JAXPR equations of the O17 objective, by owner (the committed
counter, parent 9998276020 → this round):

| owner | parent | DECISION 194 |
|---|---:|---:|
| O9–O12 thermodynamic owner | 342,170 | 35,858 |
| O14 open water (shares vremap) | 53,489 | 9,659 |
| O4 Prather dynamics | 48,325 | 48,325 |
| everything else | 133,998 | 133,998 |
| total | 577,982 | 227,840 |

Forward bits:

- five-category O12 composition fixture (6 × 6, live flux carry, 499 output
  leaves): eager and `jax.jit` both bit-identical to the parent; the committed
  test pins that digest (`e74fcfa0…73bf`);
- full O17 step, eager: all 679 published leaves bit-identical to the parent;
- full O17 step, `jax.jit`: no finite non-zero value changed; 588 elements in
  51 leaves differ only in the sign of zero (572) or of a NaN (16).  581 of the
  588 now equal the parent's eager value (the parent's compiled program had
  folded `0 + (-0)` against compile-time-zero flux accumulators, which the
  scanned carry no longer exposes — mechanism plausible, not instrumented); the
  other 7 are NaN signs in the known open-water NaN columns.  Parent eager vs
  parent jit differed in 607 elements; final jit vs parent eager differs in 33.

Compile cost (wall, peak RSS):

| program | parent | DECISION 194 |
|---|---:|---:|
| O12 composition fixture, forward jit (compile + run) | 254 s | 16 s |
| full O17 step, forward jit | 568 s, 22.4 GB | 230 s, 7.2 GB |
| O12 composition fixture, reverse mode | not measured | 141 s, 7.3 GB |
| one O9 category, reverse mode | not measured | 27.5 s |
| full O17 step, the AD probe's `value_and_grad`, `timeout 3600` | exit 124 | exit 124 |
| same program, `timeout 10800` | — | exit 124 (≥ 62 GB observed) |
| finite-masked full-step `value_and_grad`, `timeout 10800` | — | exit 124 (≥ 173 GB observed) |

Before the `ice_thd_dh` layer passes were traced (category scan, vremap and
permeability already traced) the O12 reverse mode took 729 s and 43.9 GB and
one O9 category 216 s: the straight-line layer passes dominated the reverse
compile.  The parent's own O9 reverse compile aborted before finishing, so no
parent number is quoted for those rows.

The full step still does not compile.  Isolated per-rung reverse-mode
compiles on recorded probe inputs: `ice_update_tau` 0.8 s, O12 composition
119 s, O13 remap 11 s, O14 open water 12 s, O16 46 s, O15 96 s, aEVP 19 s,
O6 `ice_cor` 396 s, O4 Prather 878 s (50 GB), and O6 redistribution did
not finish within about 2.4 h (at least 148 GB observed).  The pre-thermodynamics half of the step alone (O7 → O5 →
O4 → O6 → O7, no O8–O16) needs 82 s to lower and 1131 s to compile (55 GB);
with `jax.checkpoint` around the aEVP, Prather, redistribution and
`ice_cor` calls it needed 1292 s (60 GB).  The remaining compile cost is
therefore outside the O9–O12 owner, and the whole program compiles far
slower than the sum of its rungs.

Gradient checks: the AD probe produced no per-leaf counts (both runs timed
out), so the per-leaf table stays unmeasured.  At the O12 composition level
the reverse pass is finite on all 171 floating input leaves (157 carry a
non-zero gradient).  Central differences of the same compiled forward
(relative step 1e-4, objective = output sum weighted by 1/(1+max|leaf|)):

| leaf | step | AD | FD | rel. error |
|---|---:|---:|---:|---:|
| category 3 `h_i`, cell (3,2) | 2.34e-4 | 4.184320186 | 4.184320175 | 2.6e-9 |
| category 1 `a_i`, cell (3,2) | 2.48e-5 | 16.23678458 | 16.23678420 | 2.4e-8 |
| category 3 `e_i`, cell (3,2), layer 4 | 3.31e+4 | −3.317754415e-7 | −3.317754416e-7 | 2.6e-10 |
| category 2 `h_s`, cell (2,3) | 8.65e-6 | 104.0114698 | 104.0114690 | 7.1e-9 |
| pond `a_ip`, cell (2,3), category 1 | 6.53e-6 | 2.648828360 | 2.648828386 | 9.9e-9 |
| category 1 `qns_ice` forcing, cell (3,2) | 6.94e-3 | 0.06519869983 | 0.06519869923 | 9.1e-9 |

A full-step finite-difference check was not possible: it needs the full-step
reverse pass, which did not compile within 10800 s.  (A trial with every rung
under its own `jax.jit` and no outer `jit` aborted after 19 min with LLVM
"Cannot allocate memory"; not pursued.)

Plants (the committed test's forward pin): reversing the category scan, reversing
the `ice_thd_dh` layer order, skipping the first vremap segment, joining a
permeability start layer one step late, and removing the new bottom-flux
barrier each change the digest; stopping the gradient on the scanned category
inputs zeroes the checked thickness gradient.

Exactness battery, run serially file by file (default `-m 'not slow'`, CPU
fp64), parent 9998276020 against 17caeb46bf: every one of the 33 SI3 unit and
fidelity files, the receipt-citation gate, the four lane-3 `si3thd` gates and
the two shared-kernel unit files reports the same summary line before and
after (the new DECISION 194 file: 2 passed).  The three failures are present in
both runs and unrelated: one `test_si3_surface` production-path plant
(`sum_first_category`), the `scalarmath_v2` gate's "MY_SRC is not verbatim"
host check, and an atmosphere-config tail assertion in `test_pr_f_fixes`.  The
O4 file needed 2 h 58 min before and 3 h 58 min after on the shared host (its
reverse-mode test; the per-file limit was raised from 3 h to 4 h for the after
run only).  The O8–O16 master scorer over all 100 steps: before and after,
`PASS_ALL_O8_O13_0_ULP`, `PASS_O14_0_ULP` and `PASS_ALL_O15_O16_0_ULP`.

### Decision needed

1. The full-step reverse pass still does not compile, and the remaining cost
   is outside O9–O12: O6 redistribution alone did not compile in about 2.4 h,
   O4 Prather needs 878 s and O6 `ice_cor` 396 s, and the whole step compiles
   far slower than the sum of its rungs.
   1. Apply the same bit-pinned loop tracing to the O6 redistribution, O6
      `ice_cor`, O4 Prather, O15 and O16 rungs, one rung per round, largest
      first.  **Pick.**
   2. Keep each admitted rung in its own XLA compile unit and let JAX chain
      the true full-step gradient through them (no hand-written VJP); this
      needs O17 glue changes, and an eager-glue trial aborted with an LLVM
      memory error.
   3. Stop here and use rung-level gradients.
2. The compiled full-step forward changes only zero and NaN signs (581 of 588
   now match the eager program).  Accept as is (**pick**), or require the
   parent's compiled signed zeros.

independent review a: Gemini Argon HOLD, Gemini flash HOLD (both confirmed
the loop orders, 0 ULP against the parent eager and jit on three fixtures,
the plants and the master scorer; the five items below were raised).

### Resume 1 (review a)

1. **Eager path pinned.**  The forward test now runs the composition under
   `jax.disable_jit()` (concrete inputs, so the Python-loop path) and under
   `jax.jit` (the scan/`fori_loop` path) and requires both digests to equal the
   parent digest.  Reversing only the eager category loop changes the eager
   digest (`77c261c4…6770`) while the compiled one still matches.
2. **FMA audit.**  The reviewers' `acc + e_old[old - 1] * weight` is the
   vremap interpolation (icevar.F90 lines 1020–1022), whose two products are
   already materialised.  Counting x86 FMA instructions in the dumped XLA CPU
   kernels and mapping each back to its multiply: a jitted vremap (20,000
   random columns) had one FMA, the new-layer thickness `zhnew`
   (icevar.F90 line 1007) fused into the first interface addition (line 1012),
   value-identical there (adding +0) and 0 bit differences against eager on
   200,000 outputs; the parent's O12 program had 12 such vremap FMAs.  The
   same audit of the O12 composition found a value-changing one inherited from
   the parent: the SSL snow transmission (icethd_zdf_bl99.F90 line 209) fused
   into the absorption difference (line 211) — 7,754 of 20,000 `ab_s`
   elements differed from eager on a random input, although the composition
   fixtures and the master records happened not to expose it.  Both products
   are now materialised.  Afterwards the vremap and SSL kernels have no FMA and
   match eager bit for bit; the O12 composition keeps only FMAs on the exact
   halving `0.5 * (t_i(jk) + t_i(jk+1))`, which cannot change a value; in the
   full O17 forward program the only O9–O12 FMAs left are those exact halvings;
   the others are outside O9–O12 (O17 composition and tail, O4, O5).  A new unit check pins the SSL statement
   (removing its barrier gives 7,754 mismatches); no input was found on which
   the `zhnew` barrier changes bits, so it rests on the audit, not on a test.
   The full O17 forward (`jit`) stays bit-identical to 2428dd0472 on all 679
   published leaves, and eager to the parent on all 1,209 trace leaves.
3. **Citations.**  The 101 parent-blob pins are gone.  87 citations into the
   two edited files (81 pure moves from DECISION 194 plus 6 from the new SSL
   barrier) are re-lined on the live HEAD files with the existing re-line
   helper and keep their anchor text; 21 cite statements DECISION 194 replaced
   and now cite the replacement statements, anchored on HEAD's text.  The seven
   receipts that cite replaced code carry a DECISION 194 note.  The gate's
   checking code is unchanged.  No citation of si3_thd_pnd.py lines 862-970 exists in
   this receipt at 2428dd0472 (searched), so none was changed.
4. **Finite differences.**  Each of the six cases is judged by its own
   relative error (no absolute floor); its rounding-noise bound is at most
   1.2e-7 of its gradient.  Stopping the gradient on the scanned inputs fails
   all six (relative error 1.0 each); scaling the scanned inputs' tangent by
   1 + 1e-5 also fails all six (1.0e-5, 9.9e-6, 1.0e-5 for the 3.3e-7
   enthalpy gradient, 1.0e-5, 1.0e-5, 1.0e-5).
5. **Scored O17 fields, jit against eager, at the final source** (report
   only): 21 elements — `state.oa_i` and the after-thermodynamics `oa_i`,
   9 elements each, differ in value by up to 1.5e-16 relative (one ULP), and
   one NaN sign each in `wfx_snw_sub`, `wfx_spr` and `wfx_sub`.  The parent
   has the same 18 `oa_i` differences (from an FMA in the O17 thermodynamic
   tail, outside O9–O12) plus 12 sign-only differences in other fluxes and
   `emp`; it matched eager on the `wfx_spr` NaN sign, which is new here.
   Final jit against parent jit in scored fields: 11 elements, all zero or NaN
   signs.  User decision 2 (accept the sign changes) is still pending.

Final checks from a clean checkout of 594d6720f1, serial, CPU fp64, largest
resident set 7.9 GB (sampled every 20 s): the DECISION 194 unit file
`3 passed`; the O12 file `81 passed`; the citation gate `45 passed`; the O8–O16
master-scorer tests `30 passed`; and the O9 `15 passed`, O10 `44 passed`, O14
`10 passed`, lane-3 `si3thd` gates `22`, `8`, `13`, `17 passed` and shared BL99
unit file `12 passed`, each the same count as at the parent.  Master scorer,
100 steps: `PASS_ALL_O8_O13_0_ULP`, `PASS_O14_0_ULP`,
`PASS_ALL_O15_O16_0_ULP`.  The full-step `value_and_grad`, the AD probe, the O6
reverse compiles and the O4 file were not run in this resume.

independent review: pending (author: Claude)

## DECISION 198 round 1 — traced O6 redistribution

Review b of b17f14b73c: Gemini Argon LAND, Gemini flash LAND.  DECISION 200
accepts the compiled zero/NaN sign changes; the one-ULP `oa_i` difference from
the O17 thermodynamic tail stays out of scope.  This round extends the
DECISION 194 method to the O6 redistribution only; O6 `ice_cor`, O4 Prather,
O15 and O16 are untouched.

What changed (read off the code): the `rdgrft_shift` donor-category loop
(`DO jl1 = 1, jpl`, icedyn_rdgrft.f90 lines 722–939) became one body that an
eager call runs as the Python loop and a traced call runs as a static-trip
`fori_loop` with the category state as carry; the receiver loop
(`DO jl2`, lines 846–937) stays unrolled inside the body, and the ridging-pass
loop was already a `fori_loop`.  No statement, statement order, barrier or
MIN/MAX operand order changed: the module's barrier sweep and MIN/MAX audit
tests read the same source text.  An object-code audit of the compiled
forward redistribution (10 ice and 5 snow layers) finds no FMA instruction,
before or after.

Measurements (CPU fp64, libm arm; every command that could pass 20 GB ran
under a watchdog that kills only its own child above 120 GiB, after checking
at least 250 GiB available and no process above 100 GiB):

- **Forward bits.**  The O6 unit fixture at 3/2 and at 10/5 ice/snow layers:
  eager and `jax.jit` both bit-identical to b17f14b73c (22 result leaves each,
  19 ridging cells, 5 passes).  The full O17 step: eager and `jax.jit` both
  bit-identical to b17f14b73c on all 1,209 trace leaves, signed zeros and NaN
  bits included, so no compiled field moved.
- **Equations** (the committed counter): 228,050 at b17f14b73c → 198,323;
  the redistribution owner 43,529 → 13,802; the O9–O12 owner (35,918), open
  water (9,809) and Prather (48,325) unchanged.  The counter had not been
  re-pinned after the resume-1 barriers (it reported 228,050 against a pinned
  227,840); both snapshots are now recorded.
- **Compile.**  Isolated O6 reverse pass (`value_and_grad` of a weighted sum of
  all redistribution outputs, 10/5 layers): 12.5 s to lower, 107.6 s to
  compile, 5.0 GiB peak, all 21 input gradients finite and non-zero.  The
  parent's was not re-run here; on the probe's recorded inputs it had not
  finished after about 2.4 h at 148 GB (DECISION 194 measurement).  Forward
  jit of the full step: 154 s, 7.3 GiB → 114 s, 5.7 GiB.  The O6 unit file:
  `28 passed, 1 deselected in 219.82s`, 3.4 GiB peak (1,769 s at the parent).
- **Full-step `value_and_grad`** (the AD probe's program; allowed because the
  isolated O6 reverse compile stayed under 60 GB; run once): stopped by its
  10,800 s limit while still compiling, peak 79.8 GiB, so the per-leaf table
  stays unmeasured.  The remaining cost is outside O6 redistribution (O4
  Prather 878 s, O6 `ice_cor` 396 s, O15 96 s, O16 46 s isolated).
- **FD** (relative step 1e-4, objective = outputs weighted by
  1/(1+max|output|), each case by its own relative error; rounding noise at
  most 9.1e-8 of the gradient):

| leaf (O6 fixture) | AD | FD | rel. error |
|---|---:|---:|---:|
| `a_i` (6,4) cat 5 | 0.3667349272 | 0.3667349267 | 1.4e-9 |
| `v_i` (6,4) cat 5 | 0.07407449431 | 0.07407449425 | 7.5e-10 |
| `v_s` (6,4) cat 5 | 0.3946576485 | 0.3946576486 | 1.3e-10 |
| `e_i` (5,3) layer 1, cat 3 | 2.201112618e-9 | 2.201112617e-9 | 2.1e-10 |
| `v_ip` (6,4) cat 5 | 0.9765167278 | 0.9765167447 | 1.7e-8 |
| `oa_i` (5,3) cat 3 | 3.610880287e-7 | 3.610880287e-7 | 1.3e-10 |

- **Plants** (new test, each restored afterwards): reversing the traced donor
  order changes the compiled digest to `6c77559b…` (eager unchanged); removing
  the barrier between the receiver product and its sum changes the compiled
  digest to `943b21f4…` (removing the one on the ridged-volume sum did not
  change any bit, so it is not used as the plant); stopping the gradient on
  the donor-loop carry gives a zero gradient in all six FD cases; scaling the
  carry tangent by 1 + 1e-5 per donor step gives relative errors of
  2.5e-4 in all six (compounded over 5 donors × 5 passes).
- **Citations.**  The SI3 receipts' four citations into the redistribution
  module are pinned to historical content (DECISION 177 blobs and commits) and
  still resolve; the full map audit and every SI3 receipt pass, so nothing
  needed re-anchoring.  The gate is unchanged.

Final checks from a clean checkout of b7c4412990, serial, CPU fp64: the
DECISION 194 file `3 passed`; the DECISION 198 file `2 passed`; the O12 file
`81 passed`; the citation gate `45 passed`; the O8–O16 master-scorer tests
`30 passed`; master scorer over 100 steps `PASS_ALL_O8_O13_0_ULP`,
`PASS_O14_0_ULP`, `PASS_ALL_O15_O16_0_ULP`.  The O4 Prather file was not run.

independent review: pending (author: Claude)

## DECISION 198 round 2 — traced O4 Prather subcycles

Round 1 (743146b590) is under review; this round works on top of it and
touches O4 Prather advection only (O6 `ice_cor`, O15 and O16 untouched).
DECISION 200 accepts compiled sign-of-zero and NaN-sign changes; none occurred
here.

What changed (read off the code): under a JAX transform the subcycle loop
(`DO jt = 1, icycle`, icedyn_adv_pra.f90 lines 156–498) runs as one
three-step `lax.scan` whose body is checkpointed; each step uses the same
halo width, sweep parity and `icycle >= jt` selection as the three unrolled
calls it replaces.  The alternating sweep order (lines 267–367) becomes a
`lax.cond` on the traced parity; both branches hold the same two sweep calls
in the same order as before.  Eager calls keep the three unrolled calls and a
Python branch.  No statement, barrier or MIN/MAX operand order changed.  An
object-code audit of the compiled forward finds no FMA instruction, before or
after (probe tile through the O4 step, and the two-dimensional fixture below).

Measurements (CPU fp64, libm arm; every command that could pass 20 GB ran
under a watchdog that kills only its own child above 120 GiB, after checking
at least 250 GiB available and no process above 100 GiB):

- **Forward bits.**  The O4 step on the probe tile at one, two and three
  subcycles (both sweep parities): eager and `jax.jit` bit-identical to
  743146b590 (134 leaves each).  A 12 × 10 all-ocean fixture with ice in every
  cell, two-dimensional velocities and the O17 card's configuration (5
  categories, 10/5 layers, LEV ponds, salinity option 4) at two and three
  subcycles: eager and jit digests equal the parent's.  The full O17 step:
  eager and jit bit-identical to 743146b590 on all 1,209 trace leaves, signed
  zeros and NaN bits included.
- **Equations** (the committed counter): 198,323 at 743146b590 → 178,146;
  the Prather owner 48,325 → 28,148 (the six traced sweep substeps become
  four: two per `lax.cond` branch); the other owners unchanged.  The round-1
  snapshot is kept.
- **Compile.**  Isolated O4 reverse pass on the probe's recorded inputs (the
  per-rung instrument that gave 878 s): 743146b590 13.7 s to lower and 846.9 s
  to compile, 39.0 GiB peak; now 19.4 s and 223.1 s, 16.2 GiB.  (The earlier
  50 GB came from a process that had already compiled eight other rungs.)
  Forward jit of the O4 step on the probe tile, compile and run: about 55 s →
  12 s.
- **Full-step `value_and_grad`** (the AD probe's program; allowed because the
  isolated O4 reverse compile stayed under 30 GB; run once): stopped by its
  10,800 s limit while still compiling, peak 50.3 GiB (79.8 GiB in round 1),
  so the per-leaf table stays unmeasured.  O6 `ice_cor` (396 s), O15 (96 s)
  and O16 (46 s) are the rungs not yet traced.
- **FD** on the two-dimensional fixture at three subcycles (relative step
  1e-4, objective = outputs weighted by 1/(1+max|output|), each case by its
  own relative error; rounding noise at most 5.1e-9 of the gradient):

| input | AD | FD | rel. error |
|---|---:|---:|---:|
| `v_i` (6,5) cat 3 | 205.3812379 | 205.3812381 | 9.7e-10 |
| `a_i` (6,5) cat 3 | 8.589927548 | 8.589927551 | 3.0e-10 |
| `v_s` (4,6) cat 5 | 153.8986185 | 153.8986184 | 6.3e-10 |
| `e_i` (6,5) layer 3, cat 3 | 6.268041510e-9 | 6.268041516e-9 | 9.8e-10 |
| `v_ip` (4,6) cat 5 | 787.3811739 | 787.3811754 | 1.9e-9 |
| `u_ice` (4,6) | −19.09890745 | −19.09890748 | 1.5e-9 |
| `v_ice` (4,6) | −17.90208879 | −17.90208892 | 7.5e-9 |

  Reverse-mode values equal the parent's to 1.6e-15 relative on all 33
  scanned cases.  Two scanned cases disagree with central differences in
  both programs: `v_i` and `a_i` at cell (5,4), category 1 (AD 5.864 vs FD
  4.775; 3.667 vs 6.731).  Measured one-sided slopes differ there at steps
  from 1e-3 to 1e-7 (8.13e-8 and 0 for one area output against `v_i`), so the
  input sits on a branch boundary; reverse and forward mode both return one
  of the two one-sided slopes (the right one for that output against `v_i`,
  the left one for an area output against `a_i`).  PLAUSIBLE, read off the
  code: the
  thickness-capping test in `Hbig_pra` compares h after the sweep with the
  pre-advection maximum, which is equal at a local-maximum cell that only
  exports.  These cases are left out of the test.
- **Plants** (new test, each restored afterwards): reversing the traced
  subcycle order changes the compiled digest to `79ddec22…` (on the probe
  tile, whose meridional velocity is zero, it changed nothing, which is why
  the test uses the two-dimensional fixture); removing the barrier around one
  product of the pre-advection mass diagnostic changes the compiled digest to
  `c702ec2b…`; stopping the gradient on the scan carry fails all seven FD
  cases (zero gradient for the five state inputs); scaling the carry tangent
  by 1 + 1e-5 per subcycle gives relative errors of 3.0e-5 for the five
  state inputs and 1.1e-5 and 1.7e-5 for the two velocities, which enter
  outside the carry (all seven fail).
- **Citations.**  The traced subcycles moved every later line of the state
  module; 24 citations into it are re-lined live through the DECISION 194
  mechanism (two re-pin a closing-parenthesis occurrence number) and the
  grown dispatcher anchors on HEAD's text, with a note in the r8 O4 receipt.
  The gate logic is unchanged and passes (274 citations).
- **O4 file** (run once, alone, under the watchdog): `9 passed in 1897.02s`,
  13.5 GiB peak (2 h 58 min before DECISION 194, 60–148 GB at the parent per
  the round brief).

Final checks from a clean checkout of 67137462f8 (the round's later commits
change only this receipt, the leaf table and comments in the new test),
serial, CPU fp64: the DECISION 194 file
`3 passed`; the DECISION 198 round-1 file `2 passed`; the new O4 file
`2 passed`; the O12 file `81 passed`; the O6 file `28 passed, 1 deselected`;
the citation gate `45 passed` and its default run `PASS`; the O8–O16
master-scorer tests `30 passed`; master scorer over 100 steps
`PASS_ALL_O8_O13_0_ULP`, `PASS_O14_0_ULP`, `PASS_ALL_O15_O16_0_ULP`.

independent review: pending (author: Claude)

## DECISION 198 round 3 — traced O6 ice_cor loops

Round 2 (56b768300a) is under review; this round works on top of it and
touches O6 `ice_cor` only (O15, O16 and the round-2 `Hbig_pra` tie untouched).

Today's order (isolated reverse compile on the probe's recorded inputs, each
rung in its own process at 56b768300a, lower + compile, peak):

| rung | lower s | compile s | peak GiB |
|---|---:|---:|---:|
| O6 `ice_cor` | 7.1 | 449.8 | 20.3 |
| O4 Prather (traced in round 2) | 19.3 | 227.3 | 16.1 |
| O12 composition | 29.1 | 117.2 | 9.0 |
| O6 redistribution (traced in round 1) | 14.7 | 116.3 | 7.1 |
| O15 | 7.7 | 100.5 | 8.2 |
| O16 | 4.0 | 48.1 | 4.6 |
| aEVP | 6.6 | 17.1 | 3.7 |
| O14 open water | 4.3 | 11.1 | 3.8 |
| O13 remap | 3.8 | 9.3 | 3.7 |
| `ice_update_tau` | 0.3 | 0.5 | 3.2 |

`ice_cor` was still the largest.

What changed (read off the code): under a JAX transform the `ice_itd_reb`
passes (iceitd.f90 `DO jl = 1, jpl-1` at line 628 and `DO jl = jpl-1, 1, -1`
at line 664, each calling `itd_shiftice`), the salinity bounds
(icecor.f90 lines 114–122, `DO jl` / `DO jk`) and the `ice_var_zapsmall` flux
sums (icevar.f90 lines 633–645 salt, 664–690 heat, 695–722 water) run the same
bodies as static-trip `fori_loop`s through one helper; eager calls keep the
Python loops.  No statement, barrier or MIN/MAX operand order changed: the O6
file's barrier sweep and MIN/MAX audit read the edited source and pass.  An
object-code audit of the compiled forward `ice_cor` (3/2 and 10/5 layers)
finds no FMA instruction, before or after.

Measurements (CPU fp64, libm arm; every command that could pass 20 GB ran
under a watchdog that kills only its own child above 120 GiB, after checking
at least 250 GiB available and no process above 100 GiB):

- **Forward bits.**  The O6 `ice_cor` fixture at 3/2 and 10/5 ice/snow layers
  and the probe's recorded `ice_cor` inputs: eager and `jax.jit` bit-identical
  to 56b768300a (24 result leaves each; the 3/2 fixture rebins cells in all
  eight passes).  The full O17 step: eager and jit bit-identical to 56b768300a on
  all 1,209 trace leaves, signed zeros and NaN bits included.
- **Equations** (the committed counter): 178,146 at 56b768300a → 162,751;
  inside `ice_cor` 20,327 → 4,932; the named owners unchanged.  The round-2
  snapshot is kept.
- **Compile.**  Isolated `ice_cor` reverse pass, same instrument: 7.1 s to
  lower and 449.8 s to compile, 20.3 GiB → 3.3 s and 7.1 s, 3.4 GiB.
- **Full-step `value_and_grad`** (the AD probe's program; allowed because the
  isolated `ice_cor` reverse compile stayed under 30 GB; run once): stopped by
  its 10,800 s limit while still compiling, peak 29.2 GiB (50.3 GiB in round 2,
  79.8 GiB in round 1), so the per-leaf table stays unmeasured.
- **FD** on the O6 fixture (3/2 layers; relative step 1e-4; objective =
  outputs weighted by 1/(1+max|output|); the quotient is taken element by
  element so the objective's large unchanged part does not swamp it; each
  case by its own relative error):

| input | AD | FD | rel. error |
|---|---:|---:|---:|
| `v_i` (5,2) cat 2 | 0.7245951532 | 0.7245951537 | 7.9e-10 |
| `a_i` (5,2) cat 2 | 0.6663113636 | 0.6663113637 | 6.5e-11 |
| `v_i` (5,5) cat 1 | 11.95596085 | 11.95596085 | 3.2e-12 |
| `a_i` (5,5) cat 1 | −0.01023501231 | −0.01023501241 | 1.0e-8 |
| `szv_i` (5,2) layer 2, cat 2 | 0.02216291618 | 0.02216291618 | 9.7e-13 |
| `v_s` (5,5) cat 1 | 0.3796893385 | 0.3796893385 | 4.6e-13 |

  A scan of all 1,057 non-zero interior entries of seven inputs: reverse-mode
  values and quotients identical to the parent's (largest relative difference
  0.0).  1,017 agree to 1e-6.  The other 40 are excluded, with the same
  reverse-mode values in the parent.  39 have inputs of 1e-11 or less, all in
  categories whose input ice volume is below 1e-10; there the step falls
  below the outputs' rounding: 37 quotients
  move by more than 1e-3 between steps 1e-4 and 1e-6, one (`v_i` (3,3)
  category 1) gives a zero quotient at both steps, and one (`szv_i` (2,4)
  layer 2, category 1) splits one-sided at step 1e-6.  The 40th, `a_i` (6,4)
  category 4, is curvature (1.7e-6 at step 1e-4, 6.3e-9 at 1e-6).  No entry
  shows a stable one-sided split, the mark of a branch boundary.
- **Plants** (new test, each restored afterwards): reversing every traced
  loop changes the compiled digest to `59ebad1b…`; removing one barrier in
  the salinity-bound flux update (icecor.f90 lines 114–122) changes it to
  `ce6d1e8a…`; stopping the gradient on the loop carry gives a zero gradient
  in all six FD cases; scaling the carry tangent by 1 + 1e-5 per iteration
  gives relative errors from 4.2e-5 to 3.7e-4 in all six.
- **Citations.**  The full-map audit finds every citation at its anchor
  without re-lining, so the gate script is unchanged; it passes.
- **O6 file** (alone, under the watchdog): `28 passed, 1 deselected in
  210.22s`, 3.4 GiB peak.

Final checks from a clean checkout of a3e6e1f5e3 (the round's later commit
changes only this receipt and the leaf table), serial, CPU fp64: the DECISION
194 file `3 passed`; the DECISION 198 round-1 and round-2 files `2 passed`
each; the new ice_cor file `2 passed`; the O12 file `81 passed`; the O6 file
`28 passed, 1 deselected`; the citation gate `45 passed` and its default run
`PASS`; the O8–O16 master-scorer tests `30 passed`; master scorer over 100
steps `PASS_ALL_O8_O13_0_ULP`, `PASS_O14_0_ULP`, `PASS_ALL_O15_O16_0_ULP`.

independent review: pending (author: Claude)

## DECISION 198 round 4 — full-step reverse-compile and runtime diagnosis

DECISION 209 (2026-11-03) scopes DECISION 198 round 4 as a pure diagnosis at
`22af29b743a8` (rounds D194 and D198 r1–r3 landed) with **no model code changes**
(`git diff --stat 22af29b743a8..HEAD -- packages` is empty; forward bits are
unchanged by construction).

### 1. Question and candidate hypotheses

After DECISION 194 and DECISION 198 rounds 1–3, every isolated rung's reverse
compile (`lower + compile`) finishes in minutes (`o4_prather` 238.4 s compile,
`o12_thermo` 111.2 s, `o6_rdgrft` 110.2 s, `o15_cor` 75.9 s, `o16_update_flx`
43.6 s, `o5_aevp` 20.1 s, `o14_open_water` 13.0 s, `o13_remap` 10.4 s,
`o6_ice_cor_dyn` 7.0 s, rest ≤ 5.3 s; sum across all 14 rungs = 643.5 s
`.compile()` and 88.7 s `make_jaxpr + lower`), yet running the full-step AD
probe (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_ad_probe.py`)
still did not finish within `10,800 s` (peak `29.2 GiB` in Round 3).

Four candidate explanations were pre-registered for discrimination:

- **(a) JAX tracing/lowering vs XLA compile (and runtime execution inside the
  probe)** — separated by timing `jax.make_jaxpr(fwd)`,
  `jax.make_jaxpr(jax.value_and_grad(fwd))`, `.lower()`, `.compile()`, and
  `compiled(inputs)` independently on the full 14-rung step (`--prefixes 14
  --compile-only`) and on growing prefixes (`k = 1..14`).
- **(b) One or a few XLA passes scaling super-linearly with module size** —
  separated by capturing `TF_CPP_VMODULE=hlo_pass_pipeline=1,cpu_compiler=2,buffer_assignment=1,ir_compiler=2`
  into scratch logs (`--xla-pass-log`) and comparing per-pass wall times, buffer
  counts, and pre/post-optimization HLO instruction counts between isolated
  rungs (`k-k`), growing prefixes (`1..k`), and the full step (`1..14`).
- **(c) Composition / specific rung jumps** — separated by measuring both
  `.compile()` and `compiled(inputs)` across the 14 isolated rungs (`1-1..14-14`),
  growing prefixes (`k = 1..8, 12, 14`), and suffixes (`8-14`, `11-14`), then
  isolating any rung whose addition causes a jump in compile or execution time
  down to the individual function and field.
- **(d) Reverse pass of uncounted glue (state packing, 69-field post-dynamics
  halo exchange, `glo2eqv`, `agg(kn=1)`, surface/frazil/tail glue, outer step)**
  — separated by `--glue-only` (which mocks the 10 major D198 rungs with identity
  projections on their recorded outputs while differentiating through all
  inter-rung glue and the 4 previously uncounted stages `k = 1, 6, 7, 12` in the
  real `si3_o17_step`) and by measuring `k = 1, 6, 7, 12` in isolation.

### 2. Instrument and direct unit test

`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_ad_probe.py` was
extended (without duplicating scripts) to provide:

1. Canonical 14-rung execution-order registry `RUNG_SPECS` (`k = 1..14`)
   covering every top-level stage of `si3_o17_step`:
   `1: o7_blk1_strength`, `2: o5_aevp`, `3: o4_prather`, `4: o6_rdgrft`,
   `5: o6_ice_cor_dyn`, `6: postdyn_halos_agg`, `7: o7_surface_o8`,
   `8: o12_thermo`, `9: o13_remap`, `10: o14_open_water`, `11: o15_cor`,
   `12: thermo_tail`, `13: o16_update_flx`, `14: o16_update_tau`.
2. Separate stage timers (`fwd_make_jaxpr_seconds`, `vg_make_jaxpr_seconds`,
   `lower_seconds`, `lower_total_seconds`, `compile_seconds`,
   `execution_seconds`), peak RSS (`ru_maxrss` in KiB/GiB), and pre/post-opt HLO
   instruction and computation counts (`_count_hlo_text`).
3. CLI flags `--prefixes`, `--slices`, `--isolated-funcs`, `--glue-only`,
   `--checkpoint-rungs`, `--compile-only`, `--xla-pass-log`, and `--vmodule`,
   with provenance (`git_sha`, dirty status, `argv`, env vars) stamped in every
   JSON artifact and `/tmp` paths rejected.
4. Direct unit test `tests/ice/unit/test_si3_d198_r4_compile_diagnosis.py`
   (`2 passed in 42.01s`), which runs `diagnose_slices` on prefixes `k = 1, 2`
   and verifies planted violations on `/tmp` log paths, invalid slice bounds,
   planted HLO instructions, and planted XLA `fusion` pass durations.

### 3. Measurements (CPU fp64, `22af29b743a8`)

All runs used `CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1` on
`/home/dbalwada/legoESM/.venv/bin/python` with `TMPDIR` and all logs/outputs
under `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r4_scratch`
(after verifying `free -g` ≥ 930 GiB available and no process > 16 GiB RSS).

#### Table 1 — All 14 isolated rungs (`k-k`) and `--glue-only` (`MEASURED`)

*(Sources: `diag_p1_3_glue.json`, `diag_iso2_7.json`, `diag_iso8_14.json`,
`diag_exec_8_14.json`, `diag_exec_9_10_12_13_14.json`)*

| k | Rung label | Fwd eqns (top / rec) | VG eqns (top / rec) | `make_jaxpr(vg)` s | `lower()` s | `compile()` s | `execute()` s | Pre-opt HLO | Post-opt HLO | HLO passes s | Buffer assign s (buffers) | Backend JIT s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `o7_blk1_strength` | 670 / 1,345 | 1,753 / 3,646 | 0.28 | 0.17 | 0.87 | 0.0041 | 2,276 | 8,267 | 0.29 | 0.01 (8,049) | 0.47 |
| 2 | `o5_aevp` | 1,616 / 10,194 | 5,145 / 29,805 | 3.91 | 2.50 | 20.10 | 0.0927 | 34,883 | 130,813 | 11.54 | 1.54 (115,119) | 6.73 |
| 3 | `o4_prather` | 2,344 / 28,487 | 6,527 / 107,555 | 18.35 | 4.07 | 238.39 | 0.1145 | 66,532 | 905,032 | 151.78 | 4.77 (882,128) | 80.56 |
| 4 | `o6_rdgrft` | 1,329 / 13,934 | 3,289 / 41,298 | 14.59 | 7.30 | 110.19 | 0.0604 | 89,416 | 406,889 | 53.76 | 14.08 (341,390) | 50.55 |
| 5 | `o6_ice_cor_dyn` | 564 / 5,216 | 2,546 / 12,091 | 1.39 | 1.76 | 7.03 | 0.0094 | 20,189 | 57,657 | 3.71 | 0.42 (47,623) | 2.44 |
| 6 | `postdyn_halos_agg` | 5,044 / 7,405 | 11,760 / 17,737 | 2.50 | 0.99 | 5.28 | 0.0053 | 12,809 | 33,531 | 2.61 | 0.09 (30,739) | 1.78 |
| 7 | `o7_surface_o8` | 2,588 / 5,518 | 6,499 / 14,980 | 1.27 | 0.54 | 5.01 | 0.0143 | 8,808 | 49,201 | 2.35 | 0.11 (48,388) | 2.14 |
| 8 | `o12_thermo` | 3,900 / 38,088 | 11,241 / 148,711 | 25.11 | 7.03 | 111.16 | 1.0457 | 105,679 | 555,014 | 70.24 | 3.21 (521,098) | 35.54 |
| 9 | `o13_remap` | 5,383 / 11,687 | 14,401 / 32,121 | 2.72 | 1.18 | 10.42 | 0.0130 | 17,360 | 101,938 | 5.62 | 0.31 (97,527) | 3.58 |
| 10 | `o14_open_water` | 5,356 / 10,634 | 14,517 / 29,038 | 3.65 | 1.50 | 13.03 | 0.0116 | 23,215 | 109,512 | 6.86 | 0.45 (100,361) | 4.67 |
| 11 | `o15_cor` | 10,589 / 22,265 | 27,890 / 62,327 | 5.72 | 2.25 | 75.91 | >150 (40.8 in `1..12`) | 35,637 | 552,597 | 38.70 | 1.81 (538,549) | 33.76 |
| 12 | `thermo_tail` | 1,594 / 1,798 | 4,025 / 4,536 | 0.72 | 0.38 | 1.32 | 0.0119 | 3,353 | 7,300 | 0.37 | 0.07 (6,759) | 0.60 |
| 13 | `o16_update_flx` | 5,232 / 12,637 | 14,649 / 35,276 | 2.70 | 1.25 | 43.62 | **>10,800 (hang)** | 18,190 | 212,055 | 11.66 | 0.56 (208,217) | 30.56 |
| 14 | `o16_update_tau` | 1,175 / 1,751 | 3,467 / 5,061 | 1.08 | 0.26 | 1.02 | 0.0044 | 2,492 | 8,130 | 0.25 | 0.08 (8,125) | 0.49 |
| **Sum** | **14 isolated rungs** | **47,384 / 170,959** | **127,709 / 544,183** | **84.01** | **31.18** | **643.52** | — | **440,839** | **3,137,936** | **359.74** | **27.51 (2,954,072)** | **253.87** |
| — | `--glue-only` (10 D198 rungs mocked) | 11,146 / 17,772 | 25,356 / 43,745 | 4.96 | 1.95 | 19.03 | 0.0233 | 30,550 | 120,208 | 10.39 | 0.62 (115,447) | 6.41 |

#### Table 2 — Growing prefixes `1..k` and multi-rung suffixes (`MEASURED`)

*(Sources: `diag_p1_3_glue.json`, `diag_p4_5.json`, `diag_p6_7.json`,
`diag_p8.json`, `diag_p12.json`, `diag_p14.json`, `xla_s8_14.log`,
`xla_s8_8_11_14_slice_11_14_o15_cor_to_o16_update_tau.log`. Remaining
intermediate prefixes `k = 9, 10, 11, 13` are scripted in
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r4_scratch/operator_run.sh`.)*

| Slice / Prefix | Fwd eqns (top / rec) | VG eqns (top / rec) | `make_jaxpr(vg)` s | `lower()` s | `compile()` s | `execute()` s | Peak RSS GiB | Pre-opt HLO | Post-opt HLO | `fusion` s | `copy-insertion` s | `buffer_assignment` s (buffers) | `llvm_ir` + `codegen` wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1..1` (`o7_blk1_strength`) | 670 / 1,345 | 1,753 / 3,646 | 0.28 | 0.17 | 0.87 | 0.0041 | 3.16 | 2,276 | 8,267 | 0.09 | 0.04 | 0.01 (8,049) | 0.21 + 0.22 |
| `1..2` (`+ o5_aevp`) | 2,178 / 11,431 | 6,569 / 33,122 | 4.34 | 2.14 | 19.11 | 0.0919 | 3.98 | 36,780 | 141,185 | 2.09 | 3.22 | 1.28 (125,282) | 1.98 + 1.34 |
| `1..3` (`+ o4_prather`) | 4,201 / 39,597 | 12,538 / 140,119 | 23.01 | 6.05 | 257.69 | 0.1810 | 17.18 | 102,351 | 1,044,467 | 104.23 | 20.66 | 8.45 (1,005,657) | 31.09 + 33.05 |
| `1..4` (`+ o6_rdgrft`) | 5,420 / 53,421 | 15,385 / 180,975 | 36.79 | 13.48 | 392.35 | 0.2894 | 18.50 | 191,314 | 1,450,657 | 124.08 | 38.46 | 33.04 (1,346,333) | 39.78 + 50.30 |
| `1..5` (`+ o6_ice_cor_dyn`) | 5,739 / 58,392 | 17,359 / 192,494 | 40.51 | 15.75 | 435.12 | 0.2683 | 20.50 | 210,728 | 1,507,314 | 130.48 | 42.34 | 41.04 (1,392,968) | 42.31 + 53.90 |
| `1..6` (`+ postdyn_halos_agg`) | 10,645 / 65,659 | 28,727 / 209,839 | 40.80 | 17.07 | 432.77 | 0.2698 | 19.29 | 222,670 | 1,539,143 | 126.50 | 42.11 | 41.73 (1,422,028) | 43.92 + 56.30 |
| `1..7` (`+ o7_surface_o8`) | 12,851 / 70,699 | 34,347 / 223,834 | 43.36 | 17.83 | 455.24 | 0.2966 | 21.75 | 230,477 | 1,591,552 | 130.91 | 43.57 | 43.49 (1,473,466) | 43.14 + 57.55 |
| `1..8` (`+ o12_thermo`) | 16,288 / 108,228 | 44,452 / 371,303 | 71.48 | 23.18 | 639.69 | 1.3189 | 24.34 | 334,853 | 2,135,526 | 163.95 | 67.85 | 65.99 (1,983,458) | 55.01 + 67.13 |
| `1..12` (`+ o13,o14,o15,tail`) | 36,984 / 152,002 | 98,015 / 491,497 | 81.95 | 29.86 | 791.90 | 42.0981 | 31.08 | 408,734 | 2,901,500 | 193.92 | 81.63 | 81.93 (2,721,187) | 72.38 + 96.75 |
| **`1..14` (full O17 step)** | **39,930 / 162,737** | **103,347 / 503,371** | **79.79** | **27.67** | **803.95** | **>10,800 (hang)** | **32.12** | **402,843** | **3,052,370** | **200.29** | **78.48** | **76.72 (2,869,611)** | **74.10 + 109.75** |
| `8..14` (`o12` to `update_tau`) | 29,638 / 94,552 | 76,650 / 302,163 | 47.00 | 13.79 | 274.20 | — | — | 194,207 | 1,622,550 | 73.17 | 31.02 | 22.84 (1,563,689) | 50.85 + 31.83 |
| `11..14` (`o15` to `update_tau`) | 16,134 / 35,895 | 43,727 / 98,948 | 10.69 | 3.43 | 122.40 | — | — | 52,053 | 786,876 | 30.89 | 12.23 | 4.86 (771,846) | 27.66 + 26.86 |

#### Table 3 — Full-step (`k = 14`, `prefix_1_14_o16_update_tau`) compile time breakdown (`MEASURED`)

*(Source: `diag_p14.json` and `xla_p14.log`)*

| Phase / XLA Pass | Calls | Wall s | Max single call s | Share of `.compile()` (`803.952 s`) | Share of total (`911.405 s`) |
|---|---:|---:|---:|---:|---:|
| **`jax.make_jaxpr(fwd)`** | 1 | 6.455 | 6.455 | — | — |
| **`jax.make_jaxpr(value_and_grad(fwd))`** | 1 | 79.785 | 79.785 | — | 8.75% |
| **JAX `.lower()` (MLIR / StableHLO emission)** | 1 | 27.668 | 27.668 | — | 3.04% |
| **Subtotal: `lower_total_seconds` (`make_jaxpr(vg) + lower`)** | — | **107.453** | — | — | **11.79%** |
| **XLA HLO pass pipeline (`hlo_passes_wall_s`)** | — | **465.004** | — | **57.84%** | **51.02%** |
| — `fusion` (CPU `InstructionFusion`) | 1 | 200.293 | 200.293 | 24.91% | 21.98% |
| — `copy-insertion` | 1 | 78.476 | 78.476 | 9.76% | 8.61% |
| — `simplify-while-loops` | 50 | 27.615 | 1.636 | 3.43% | 3.03% |
| — `algsimp` | 53 | 19.578 | 1.612 | 2.44% | 2.15% |
| — `tree_reduction_rewriter` | 100 | 14.426 | 0.166 | 1.79% | 1.58% |
| — `layout-assignment` | 1 | 12.112 | 12.112 | 1.51% | 1.33% |
| — `while-loop-invariant-code-motion` | 50 | 10.429 | 1.245 | 1.30% | 1.14% |
| — `tuple-simplifier` | 52 | 9.322 | 2.445 | 1.16% | 1.02% |
| — `constant_folding` | 50 | 7.419 | 1.175 | 0.92% | 0.81% |
| — `cpu-parallel-task-assigner` | 1 | 7.395 | 7.395 | 0.92% | 0.81% |
| — all other HLO passes | — | 77.939 | — | 9.69% | 8.55% |
| **XLA CPU backend JIT (`backend_jit_total_s`)** | 1 | **306.660** | 306.660 | **38.14%** | **33.65%** |
| — `buffer_assignment` (`2,869,611` buffers, `51.65 MiB` alloc) | 1 | 76.724 | 76.724 | 9.54% | 8.42% |
| — `llvm_ir_emit` (HLO → LLVM IR across `10,009` kernels) | 1 | 74.099 | 74.099 | 9.22% | 8.13% |
| — `llvm_codegen` (`10,009` modules; `965.43 s` thread-sum) | 10,009 | 109.752 | 43.567 | 13.65% | 12.04% |
| — backend scheduling / thunk emission / linking | — | 46.085 | — | 5.73% | 5.06% |
| **HLO module setup / pre-pass & post-JIT wrapper overhead** | — | **32.288** | — | **4.02%** | **3.54%** |
| **Subtotal: `.compile()` (`compile_seconds`)** | 1 | **803.952** | 803.952 | **100.00%** | **88.21%** |
| **Total compile (`lower_total_seconds + compile_seconds`)** | — | **911.405** | — | — | **100.00%** |

#### Table 4 — Isolation of the `> 10,800 s` probe timeout to `si3_var_agg(..., kn=2).tm_i` inside `si3_o16_step` (`MEASURED`)

*(Sources: sub-function and per-field isolation runs on the recorded `si3_o16_step`
inputs at `22af29b743a8`)*

| Unit under `jax.value_and_grad` | `lower()` s | `compile()` s | `execute()` s | Post-opt HLO | Notes |
|---|---:|---:|---:|---:|---|
| `si3_o16_step` (`k = 13`, baseline `22af29b743a8`) | 1.25 | 43.62 | **>10,800 (hang)** | 212,055 | Calls `si3_var_glo2eqv(kn=2)`, `si3_var_agg(kn=2)`, `si3_ice_update_flx` |
| — `si3_var_glo2eqv(..., kn=2)` alone | 0.26 | 2.69 | **0.0046** | — | Executes in 4.6 ms |
| — `si3_ice_update_flx(...)` alone | 0.35 | 6.92 | **0.0033** | — | Executes in 3.3 ms |
| — `si3_var_agg(..., kn=1)` (`k = 6`, post-dynamics) | 0.99 | 5.28 | **0.0053** | 33,531 | `kn <= 1` returns before `tm_i` / `tm_s` loops (`si3_orca2_state.py` line 1141) |
| — `si3_var_agg(..., kn=2)` alone (all 21 fields) | 0.67 | 28.12 | **>10,800 (hang)** | 131,806 | Enters `kn > 1` block (`si3_orca2_state.py` lines 1143–1210) |
| — `si3_var_agg(kn=2)` fields `hm_i, hm_s` | 0.08 | 0.86 | **0.0007** | — | Elementwise `jnp.where` |
| — `si3_var_agg(kn=2)` field `tm_si` | 0.14 | 1.66 | **0.0014** | — | 5-step `jpl` loop |
| — `si3_var_agg(kn=2)` fields `om_i, sm_i` | 0.18 | 2.09 | **0.0014** | — | 5-step `jpl` loop |
| — `si3_var_agg(kn=2)` field `tm_s` | 0.30 | 3.51 | **0.0023** | — | 25-step `jpl * nlay_s` loop (`5 * 5 = 25`) |
| — `si3_var_agg(kn=2)` fields `hm_ip, hm_il` | 0.21 | 2.36 | **0.0015** | — | 5-step `jpl` loop |
| — **`si3_var_agg(kn=2)` field `tm_i` alone** | **0.28** | **20.23** | **>10,800 (hang)** | — | **50-step unrolled `jpl * nlay_i` loop (`5 * 10 = 50`, `si3_orca2_state.py` lines 1171–1178)** |
| **In-memory test: `si3_var_agg(..., kn=2)` with `lax.scan` on `tm_i, tm_s`** | **0.35** | **2.60** | **0.0021** | — | **0.000e+00 max abs/ULP forward diff on all 21 fields** |
| **In-memory test: `si3_o16_step` with `lax.scan` on `tm_i, tm_s`** | **0.88** | **6.64** | **0.0098** | **60,095** | **Compile `43.62 s -> 6.64 s` (6.6x); execute `>10,800 s -> 0.0098 s`** |

### 4. Candidate verdicts (labeled `MEASURED` or `PLAUSIBLE`)

1. **Candidate (a) — JAX tracing/lowering vs XLA compile vs runtime execution (`MEASURED`)**:
   - **MEASURED**: On the full 14-rung O17 step (`prefix_1_14_o16_update_tau`),
     `jax.make_jaxpr(fwd)` takes `6.455 s`, `jax.make_jaxpr(value_and_grad(fwd))`
     takes `79.785 s`, and `.lower()` takes `27.668 s` (`lower_total_seconds =
     107.453 s`, 3.04 GiB RSS, `402,843` pre-opt HLO instructions). XLA
     `.compile()` takes **`803.952 s` (`13.40 min`)** at **`32.120 GiB` peak
     RSS**, producing `3,052,370` post-opt HLO instructions. Thus **full-step
     compilation (`lower + compile`) completes in `911.405 s` (`15.19 min`)**.
   - **MEASURED**: In Round 3's log (`work/si3_d194_scratch/probe_d198r3.log`),
     XLA's `SlowOperationAlarm` (`[Compiling module jit_objective for CPU]`,
     which wraps `CpuCompiler`'s backend JIT phase) printed completion
     (`5m12.90s`) 15 minutes after job start — matching today's `5m06.66s`
     backend JIT and `15.19 min` total compile. Immediately after `.compile()`
     returned, `nemo_si3_o17_ad_probe.py` executed `compiled(inputs)` before
     printing any output, whereas Round 3's isolated-rung script (`ad_rungs.py`)
     only called `.lower(diff).compile()` and never called `compiled(diff)`.
     Therefore **`911.4 s` (`8.4%`) of the `10,800 s` timeout was compile time
     and `> 9,880 s` (`> 91.6%`) was a runtime execution hang in
     `compiled(inputs)`**.

2. **Candidate (b) — XLA passes scaling with module size (`MEASURED`)**:
   - **MEASURED**: Full-step `.compile()` (`803.952 s`) is **1.25x** the sum of
     the 14 isolated rungs' `.compile()` (`643.521 s`), and post-opt HLO
     (`3,052,370` instructions) is **7.58x** pre-opt HLO (`402,843`
     instructions).
   - **MEASURED**: Five XLA phases account for **`71.1%` (`571.64 s` /
     `803.95 s`)** of full-step `.compile()`:
     1. HLO `fusion` (`200.293 s`, 1 call, `24.9%`);
     2. Backend `llvm_codegen` (`109.752 s` wall / `965.433 s` thread-sum across
        `10,009` kernel modules, `13.7%`);
     3. HLO `copy-insertion` (`78.476 s`, 1 call, `9.8%`);
     4. Backend `buffer_assignment` (`76.724 s` over `2,869,611` buffers,
        `9.5%`, which scales **2.79x** relative to the sum of isolated rungs'
        `27.51 s` because whole-step live ranges coexist in one module);
     5. Backend `llvm_ir_emit` (`74.099 s`, `9.2%`).

3. **Candidate (c) — Composition / specific rung jumps (`MEASURED` + `PLAUSIBLE`)**:
   - **MEASURED (compile jumps)**: Four rungs account for **`83.2%` (`535.64 s`
     / `643.52 s`)** of isolated `.compile()` and drive the prefix `.compile()`
     jumps: `k = 3` (`o4_prather`, `+238.58 s` in prefix `1..3`), `k = 4`
     (`o6_rdgrft`, `+134.67 s` in prefix `1..4`), `k = 8` (`o12_thermo`,
     `+184.45 s` in prefix `1..8`), and `k = 11` (`o15_cor`, `75.91 s` isolated)
     + `k = 13` (`o16_update_flx`, `43.62 s` isolated; `+152.20 s` across
     `1..8 -> 1..12`). In particular, `si3_o15_step` (`si3_ice_cor` in
     `packages/ice/legoesm/ice/si3_finalize.py` lines 553–730) still unrolls the
     `itd_shiftice`, salinity-bound, and `zapsmall` loops (`27,890` top-level VG
     eqns, `75.91 s` `.compile()`), whereas `si3_ice_cor_dyn` (traced in Round 3,
     `k = 5`) has `2,546` top-level VG eqns and compiles in `7.03 s`.
   - **MEASURED (runtime execution jumps)**: Prefixes `1..1` through `1..8`
     execute `compiled(inputs)` in `0.0041 s – 1.3189 s`. Prefix `1..12`
     executes in `42.0981 s` (`+40.78 s` from `k = 11` `si3_o15_step`'s unrolled
     `ice_cor` loops). Adding `k = 13` (`si3_o16_step`) causes `compiled(inputs)`
     to hang for **`> 10,800 s`**, isolated specifically to the 50-step unrolled
     `tm_i_acc` loop in `si3_var_agg(..., kn=2)`
     (`packages/ice/legoesm/ice/si3_orca2_state.py` lines 1171–1178).
   - **MEASURED (HLO inspection) + PLAUSIBLE (XLA CPU emitter mechanism)**:
     In `si3_var_agg(..., kn=2)`, `tm_i` unrolls `50` iterations (`jpl=5 *
     nlay_i=10`) of
     `tm_i_acc = jnp.where(vt_i > _EPSI20, sr(tm_i_acc + sr(sr(t_i[:, :, jk, jl] * v_i[:, :, jl]) * r1_nlay_i)), rt0)`,
     where `sr` (`nemo_source_round`,
     `packages/core/legoesm/core/source_rounding.py` lines 14–27) is
     `jnp.where(jnp.isfinite(value), value, jnp.copysign(jnp.abs(value), value))`.
     MEASURED in the post-opt HLO dump of `si3_var_agg(..., kn=2)`: XLA fuses all
     50 backward steps and their `slice`/`pad`/`dynamic-update-slice` scatters
     into two 2,859-line `kLoop` fusions (`%fused_computation.1` and
     `%fused_computation.2`, `backend_config={"outer_dimension_partitions":["3"]}`,
     matching the 3 `tf_XLAEigen` worker threads observed spinning at 100% CPU).
     PLAUSIBLE from XLA CPU's `FusedIrEmitter`: because the per-step cotangent
     feeds multiple branch/select operands (`isfinite`, `copysign`, `abs`,
     `vt_i > _EPSI20`) and non-elementwise `slice`/`pad` index expressions inside
     a single `kLoop` fusion, `FusedIrEmitter` re-emits upstream value
     evaluations at each use instead of caching per-step temporaries, turning a
     50-step diamond recurrence into `O(2^50)` runtime work.
   - **MEASURED (in-memory verification without touching `packages/`)**:
     Replacing the unrolled `tm_i_acc` and `tm_s_acc` loops in
     `si3_var_agg(..., kn=2)` with `jax.lax.scan` preserves `0.000e+00` ULP
     forward agreement across all 21 fields of `SI3AggregatedVars`, cuts
     `si3_var_agg(..., kn=2)` reverse compile from `28.12 s` to `2.60 s` and
     execution from `> 10,800 s` to `0.0021 s`, and cuts `si3_o16_step` reverse
     compile from `43.62 s` to `6.64 s` and execution from `> 10,800 s` to
     `0.0098 s`.

4. **Candidate (d) — Uncounted glue, state packing, 69-field halo exchange,
   outer step (`MEASURED` — REFUTED)**:
   - **MEASURED**: `--glue-only` (all 10 major D198 rungs mocked; all inter-rung
     glue, 69-field `si3_apply_state_halos_orca2`, `si3_var_glo2eqv`,
     `si3_var_agg(kn=1)`, `o7_blk1_strength`, `o7_surface_o8`, `_thermo_entries`,
     `_thermodynamic_tail`, `SI3UpdateFluxForcing`, and `SI3O17Trace` packing
     live) lowers in `1.952 s`, compiles in `19.034 s` (`120,208` post-opt HLO),
     and executes in `0.0233 s`. The four previously uncounted rungs in isolation
     (`k = 1, 6, 7, 12`) sum to `2.08 s` `lower()`, `12.48 s` `.compile()`, and
     `0.036 s` `execute()`.

### 5. Next changes pointed to by this diagnosis (no model changes made this round)

1. **Primary change (eliminates the `> 10,800 s` runtime hang in `compiled(inputs)`
   and cuts `si3_o16_step` reverse compile from `43.62 s` to `6.64 s`)**:
   In `si3_var_agg` (`packages/ice/legoesm/ice/si3_orca2_state.py` lines 1171–1187),
   when `kn > 1` and tracing under a JAX transform, replace the unrolled 50-step
   `tm_i_acc` (`jpl * nlay_i`) and 25-step `tm_s_acc` (`jpl * nlay_s`) Python
   loops with static-trip `jax.lax.fori_loop` / `jax.lax.scan` carries in exact
   `(jl, jk)` order (already verified in memory to be `0.000e+00` ULP
   bit-identical on all 21 `SI3AggregatedVars` fields).
2. **Secondary change (cuts `si3_o15_step` isolated reverse compile from
   `75.91 s` to `~7 s` and removes its `~40.8 s` execution cost in `1..12`)**:
   Route `si3_o15_step` (`si3_ice_cor` in
   `packages/ice/legoesm/ice/si3_finalize.py` lines 553–730) through the traced
   `fori_loop` helpers (`itd_shiftice`, salinity bounds, `ice_var_zapsmall`)
   already landed for `si3_ice_cor_dyn` in Round 3.

independent review: operator

## O17 thermodynamics composition fixes D1, D2

Date: 2026-11-05. Base: `23d39a9bc7eaf260be49f60aed42142953da2047`
(clean SI3 lane merge). No NEMO executable was run and the admitted record remained
read-only.

Two measured composition defects were corrected without adding a selector or changing
a default:

- **D1, DH -> DA flux carry.** `si3_ice_thd_o12_stages` now gives
  `si3_ice_thd_da` the DH-updated `dh.fluxes.hfx_thd`, matching the executed NEMO
  order in `icethd.F90` and `icethd_da.F90`. The direct `6 x 6` traced-tile test
  observes the actual DA call, proves the DH value differs bitwise from the stale
  entry flux, proves DA receives the DH value bit for bit, and proves substituting
  the stale value changes the DA result. Fixed: `1 passed in 289.77s`. In isolated
  scratch copy `d1_revert.sMABeq`, reverting only D1 makes that same test fail on
  all `36 / 36` values (`1 failed in 301.35s`).
- **D2, before-remap thickness.** The O17 composition computes the saved
  before-remap thickness directly as `v_i_b / a_i_b`, before the remap, instead
  of reusing the reciprocal-multiplied equivalent thickness. The small mixed
  production tile proves the two formulations differ by at least one binary64
  ULP and observes the actual remap input. Fixed: `1 passed in 305.84s`. In
  isolated scratch copy `d2_revert.ShPELa`, reverting only D2 makes the same test
  fail on `5 / 45` values, each by exactly one bit in the `uint64` view
  (`1 failed in 294.03s`).

All commands below used this prefix (and every command was additionally bounded by
`timeout 1500`):

```bash
env TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_thdfix_scratch \
  CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src \
  /home/dbalwada/legoESM/.venv/bin/python
```

### MEASURED fidelity and regression results

The merged production composition scorer, eager, scored the read-only real ORCA2
record at GIVEN-NEMO-ENTRY with **zero differing values, zero non-finite values,
and `max_relative = 0.0`** at each requested step:

| Step | Verdict | Wall time (s) | Report |
|---:|---|---:|---|
| 1 | `PASS`, 0 differences | 1175.4967863 | `o17_step1_eager.json` |
| 2 | `PASS`, 0 differences | 1257.8526021 | `o17_step2_eager.json` |
| 3 | `PASS`, 0 differences | 1280.0754074 | `o17_step3_eager.json` |
| 50 | `PASS`, 0 differences | 1220.4446083 | `o17_step50_eager.json` |

The command was run once for each `N` in `1 2 3 50`:

```bash
scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py \
  --evidence /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step \
  --labels GIVEN-NEMO-ENTRY --execution eager --expect-steps 100 --steps N..N \
  --out /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_thdfix_scratch/o17_stepN_eager.json
```

The admitted O8--O16 master record was rescored, unchanged, for steps `1..2`:

```bash
scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py --rung O8-O13 --steps 1..2 --output-json .../o8_o13_steps1_2.json
scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py --rung O13 --steps 1..2 --output-json .../o13_steps1_2.json
scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py --rung O14 --steps 1..2 --output-json .../o14_steps1_2.json
scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py --rung O15-O16 --steps 1..2 --output-json .../o15_o16_steps1_2.json
```

The respective final statuses were `PASS_ALL_O8_O13_0_ULP`, `PASS_O13_0_ULP`,
`PASS_O14_0_ULP`, and `PASS_ALL_O15_O16_0_ULP`.

The real-grid step-1 JIT command was identical to the eager command except for
`--execution jit --steps 1..1 --out .../o17_step1_jit.json`. Because eager is
bitwise identical to the NEMO record, the JIT-to-record comparison is also the
JIT-to-eager comparison. It is **not accepted by DECISION 200**: `FAIL`, `1513`
finite numeric differences in `44` fields, `0` signed-zero-only differences,
`0` non-finite/NaN differences, and `max_relative = 1.9021217570415877e-13`.
The non-accepted fields were:

```text
after_rdgrft: a_i, szv_i, wfx_dyn, sfx_dyn, hfx_dyn
after_dyn_cor: a_i, szv_i
after_thermo: v_i, v_s, sv_i, a_i, t_su, oa_i, a_ip, v_ip, e_s, e_i,
              szv_i, t_si
step_exit: v_i, v_s, sv_i, a_i, t_su, oa_i, a_ip, v_ip, e_s, e_i, szv_i,
           t_si, wfx_snw, wfx_pnd, wfx_dyn, sfx_bom, sfx_bri, sfx_dyn,
           hfx_dif, hfx_err_dif, hfx_sub, hfx_dyn, qns, emp, sfx
```

Focused test commands and measured results:

```text
pytest tests/ice/unit/test_si3_d194_traced_loops.py -q
pytest tests/ice/unit/test_si3_d198_icecor_traced.py -q        -> 2 passed in 21.57s
pytest tests/ice/unit/test_si3_d198_prather_traced.py -q       -> 2 passed in 299.49s
pytest tests/ice/unit/test_si3_d198_rdgrft_traced.py -q        -> 2 passed in 150.06s
pytest tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py -q -> 4 passed in 48.88s
pytest tests/ice/unit/test_si3_o17.py -k test_o17_inactive_category_open_water_snowfall_nan_fix_and_plant -q
  -> 1 passed, 22 deselected in 105.01s
pytest tests/ice/unit/test_si3_o17.py -k test_o17_newly_guarded_outputs_reverse_mode_matches_central_differences_and_plants -q
  -> 1 passed, 22 deselected in 152.65s
```

The final full-file D194 rerun passed (`4 passed in 828.69s`). The refreshed O17
production-invariant digest is
`46317520de43e28ead25d27239c5d0dc05fe7dde92087f6594f59c317a74c16d`, and its
direct node rerun passed (`1 passed in 318.76s`). After live-source re-anchoring,
`LEGOESM_GATE_ALLOW_DIRTY=1 pytest tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py -q`
passed (`45 passed in 9.44s`); the scoped dirty stamp is required because the
operator, rather than this agent, commits the completed worktree.

## DECISION 198 round 5 — traced `si3_var_agg(kn=2)` and `si3_o15_step` loops

### Summary of changes (`3dd560f7f05f` -> Round 5)

1. **Primary (`si3_var_agg(kn=2)` layer-sum loops)**:
   In `packages/ice/legoesm/ice/si3_orca2_state.py` lines 1176–1211, when `kn > 1`
   and `_si3_traced(a_i, v_i, v_s, t_i, t_s)` is True (traced under a JAX
   transform), replaced the unrolled 50-step `tm_i_acc` (`jpl * nlay_i = 5 * 10`)
   and 25-step `tm_s_acc` (`jpl * nlay_s = 5 * 5`) Python loops with static-trip
   `jax.lax.scan` carries (`_scan_layer_temp`) over the `(jl, jk)` index sequence
   (`jnp.moveaxis(t_arr, (3, 2), (0, 1)).reshape(jpl * nlay, jpi, jpj)`), citing
   NEMO 5.0.2 `src/ICE/icevar.F90` lines 181–191 (`tm_i`) and lines 192–200
   (`tm_s`). The eager path (`not _si3_traced(...)`) is preserved untouched.

2. **Secondary (`si3_o15_step` redistribution and `ice_cor` loops)**:
   - Extracted `si3_ice_cor_salinity_bounds` (`src/ICE/icecor.f90` lines 114–122,
     50-step `sz_i` clipping loop with static-trip `jax.lax.fori_loop` when traced)
     in `packages/ice/legoesm/ice/itd.py` lines 1344–1404, and threaded optional
     `traced: bool | None = None` through `si3_ice_itd_reb`,
     `si3_ice_cor_salinity_bounds`, and `si3_var_zapsmall`.
   - Routed `si3_o15_step`'s `_ice_itd_reb` (`packages/ice/legoesm/ice/si3_finalize.py`
     lines 440–594) and `si3_ice_cor` (`packages/ice/legoesm/ice/si3_finalize.py`
     lines 597–880) through `si3_ice_itd_reb`, `si3_ice_cor_salinity_bounds`, and
     `si3_var_zapsmall` when `_si3_traced(...)` is True and
     `not _plant_strip_transfer_barriers`. The eager path and explicit plant path
     remain unchanged.

3. **Codex candidate audit (isolated reverse tracing, lowering, compilation, and execution)**:
   Measured now by running:
   - Command A (isolated units on recorded inputs):
     `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_ad_probe.py --reproduce-d198r5 --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch/reproduce_d198r5.json`
   - Command B (step slices `11..11`, `13..13`, `11..14`):
     `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_ad_probe.py --slices 11-11,13-13,11-14 --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch/slices_11_13_14.json`

   with tracing (`jax.make_jaxpr(jax.value_and_grad(fn))`), MLIR lowering
   (`.lower()`), XLA compilation (`.compile()`), and execution timed separately:

   | Candidate region | Mode | Tracing (s) | Lowering (s) | Compilation (s) | Lower+Compile (s) | Trace+Lower+Compile (s) | Execute (s) | VG JAXPR (top / rec) | Post-opt HLO | Disposition |
   | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
   | `si3_var_agg(kn=2)` (`si3_orca2_state.py` lines 1176–1211, 50+25 steps) | unrolled (compile-only) | `1.407` | `0.654` | `28.743` | `29.397` | `30.804` | — (hangs `>10,800 s` in r4) | `8,419 / 22,472` | `140,389` | **Converted (Primary)** |
   | `si3_var_agg(kn=2)` (`si3_orca2_state.py` lines 1176–1211, 50+25 steps) | scan | `0.350` | `0.324` | `2.169` | `2.493` | `2.843` | `0.0019` | `2,291 / 7,091` | `21,491` | **Converted (`11.8x` lower+compile speedup; executes in `1.9 ms`)** |
   | `si3_o16_step` (`si3_finalize.py` lines 1018–1182, calls `si3_var_agg(kn=2)`) | unrolled (compile-only) | `2.375` | `1.185` | `46.358` | `47.544` | `49.918` | — (hangs `>10,800 s` in r4) | `12,740 / 33,151` | `205,510` | **Converted via Primary** |
   | `si3_o16_step` (`si3_finalize.py` lines 1018–1182, calls `si3_var_agg(kn=2)`) | scan | `1.203` | `0.675` | `5.821` | `6.496` | `7.698` | `0.0067` | `6,612 / 17,770` | `69,020` | **Converted (`7.3x` lower+compile speedup; executes in `6.7 ms`)** |
   | `si3_o15_step` (`si3_finalize.py` lines 440–906: `_ice_itd_reb` + `si3_ice_cor` 50-step `sz_i` + `zapsmall`) | unrolled (compile-only) | `4.691` | `1.851` | `72.734` | `74.586` | `79.277` | — | `26,360 / 60,691` | `548,378` | **Converted (Secondary)** |
   | `si3_o15_step` (`si3_finalize.py` lines 440–906: `_ice_itd_reb` + `si3_ice_cor` 50-step `sz_i` + `zapsmall`) | traced `fori_loop` | `1.764` | `1.380` | `7.339` | `8.719` | `10.483` | `0.0090` | `2,654 / 12,592` | `59,118` | **Converted (`8.6x` lower+compile speedup; executes in `9.0 ms`)** |
   | O15 `_ice_itd_reb` (`si3_finalize.py` lines 440–594) | unrolled | `2.948` | `1.032` | `8.873` | `9.905` | `12.853` | `0.0049` | `13,546 / 29,362` | `82,473` | **Converted as part of Secondary** |
   | O15 `_ice_itd_reb` (`si3_finalize.py` lines 440–594) | traced `fori_loop` | `1.028` | `1.504` | `5.249` | `6.752` | `7.780` | `0.0061` | `1,405 / 8,409` | `45,440` | **Converted (`1.47x` lower+compile speedup)** |
   | O15 50-step salinity bounds (`itd.py` lines 1344–1404) | unrolled | `1.739` | `0.730` | `43.233` | `43.963` | `45.702` | `0.0043` | `9,667 / 24,937` | `275,966` | **Converted as part of Secondary** |
   | O15 50-step salinity bounds (`itd.py` lines 1344–1404) | traced `fori_loop` | `0.062` | `0.125` | `0.480` | `0.605` | `0.667` | `0.0015` | `50 / 538` | `2,882` | **Converted (`72.7x` lower+compile speedup)** |
   | `rung_11_o15_cor` slice `11..11` in `si3_o17_step` | traced `fori_loop` | `2.102` | `1.788` | `8.095` | `9.883` | `11.985` | `0.0184` | `4,184 / 14,228` | `61,576` | **Verified via `--slices 11-11` (was `78.16 s` lower+compile in r4)** |
   | `rung_13_o16_update_flx` slice `13..13` in `si3_o17_step` | scan | `1.723` | `0.794` | `6.495` | `7.289` | `9.012` | `0.0145` | `8,521 / 19,895` | `75,565` | **Verified via `--slices 13-13` (was `44.87 s` lower+compile, `>10,800 s` exec in r4)** |
   | `slice_11_14_o15_cor_to_o16_update_tau` tail slice `11..14` in `si3_o17_step` | traced `fori_loop` + scan | `4.226` | `2.249` | `16.680` | `18.930` | `23.155` | `0.0232` | `12,902 / 35,537` | `140,158` | **Verified via `--slices 11-14` (was `125.83 s` lower+compile in r4)** |
   | O12 10-step pond permeability loop (`si3_thd_pnd.py` lines 247–260) | unrolled | `0.114` | `0.099` | `0.464` | `0.563` | `0.677` | `0.0006` | `1,265 / 2,256` | `5,150` | **Not converted (`0.563 s` lower+compile, `0.0006 s` exec)** |
   | `fortran_sum_categories` 10-step sum (`si3_orca2_state.py` lines 768–777) | unrolled | `0.036` | `0.058` | `0.339` | `0.397` | `0.434` | `0.0004` | `299 / 931` | `2,827` | **Not converted (`0.397 s` lower+compile, `0.0004 s` exec)** |
   | `si3_var_glo2eqv` 10-step `s_i` sum (`si3_orca2_state.py` lines 997–998) | unrolled | `0.048` | `0.046` | `0.392` | `0.437` | `0.486` | `0.0004` | `387 / 1,017` | `764` | **Not converted (`0.437 s` lower+compile, `0.0004 s` exec)** |

### Proofs

1. **Proof 1 — Forward bits of O4–O16 and full O17 step (`1,209` leaves, `1,187` float64 leaves) unchanged vs `3dd560f7f05f` (eager and `jax.jit`, signed zeros and NaN bits included) + negative-zero fixture**:
   - **Full O17 step (`1,209` leaves, `1,187` float64 leaves)**:
     Measured now by running:
     `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 /home/dbalwada/legoESM/.venv/bin/python -c 'import hashlib, json, time; import jax, numpy as np; from legoesm.ice import si3_finalize as finalize_mod, si3_orca2_state as state_mod; from legoesm.ice.si3_o17 import SI3O17LiveForcing, si3_o17_step; from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_ad_probe import DEFAULT_MASTER, _case, _is_float_leaf; entry, forcing, card = _case(DEFAULT_MASTER); kt, dyn = int(forcing.kt), forcing[1:]; run_step = lambda e, d: si3_o17_step(e, SI3O17LiveForcing(kt, *d), card); ...'`
     - Eager (`si3_o17_step`): `0` raw `uint64` bit diffs across all `1,209`
       leaves (`1,187` float64 leaves) vs `3dd560f7f05f` eager
       (`sha256 = 2855020ad789d59b64374976c29adccb4a6ebf776471cecba08f62974017ae7b`,
       preserving all `17,734` `-0.0` entries and `44` NaNs bit-for-bit; `73.521 s`).
     - Compiled (`jax.jit(si3_o17_step)`): `0` raw `uint64` bit diffs across
       all `1,209` leaves (`1,187` float64 leaves) between Round 5 `jax.jit` and
       `3dd560f7f05f` unrolled `jax.jit`
       (`sha256 = 980bc967d48b103c2ee13733cf165e5ca7071c322ebf0696cd64ebe0a285e4fe`,
       preserving all `17,734` `-0.0` entries and `44` NaNs bit-for-bit; `70.907 s`).
   - **Recorded subphase parity (`si3_var_agg(kn=2)`, `si3_o15_step`, `si3_o16_step`) and `build_negative_zero_var_agg_fixture` in `nemo_si3_o17_ad_probe.py --reproduce-d198r5`**:
     Measured now by running Command A above (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_ad_probe.py --reproduce-d198r5`):
     - `var_agg_recorded_eager_vs_scan_jit` and `var_agg_recorded_unrolled_jit_vs_scan_jit`:
       `21` float64 output leaves, `0` raw `uint64` bit mismatches (`569` `+0.0`,
       `0` `-0.0`, `0` NaNs).
     - `si3_o15_step_recorded_eager_vs_traced_jit`:
       `155` float64 output leaves, `0` raw `uint64` bit mismatches (`26,686` `+0.0`,
       `4,422` `-0.0`, `5` NaNs).
     - `si3_o16_step_recorded_eager_vs_scan_jit`:
       `169` float64 output leaves, `0` raw `uint64` bit mismatches (`25,352` `+0.0`,
       `4,416` `-0.0`, `9` NaNs).
     - `build_negative_zero_var_agg_fixture` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_ad_probe.py`
       lines 1453–1530): injects IEEE-754 `-0.0` (`0x8000000000000000`) into all
       `16` input arrays of `si3_var_agg(kn=2)` (`318` total `-0.0` input entries at
       `(nlay_i, nlay_s) = (10, 5)`: `a_i: 7`, `a_ip: 7`, `a_ip_eff: 4`, `e_i: 49`,
       `e_s: 24`, `oa_i: 8`, `sv_i: 5`, `szv_i: 98`, `t_i: 49`, `t_s: 24`,
       `t_si: 8`, `t_su: 8`, `v_i: 7`, `v_il: 8`, `v_ip: 8`, `v_s: 4`, plus
       interior cell `(2, 2)` where `-1.0e-300 / 1.0e30` underflows to `-0.0`).
       In both `var_agg_neg_zero_eager_vs_scan_jit` and
       `var_agg_neg_zero_unrolled_jit_vs_scan_jit`, all `18` surviving `-0.0`
       output entries across `12` fields (`at_ip_eff: 1`, `et_i: 1`, `et_s: 1`,
       `hm_il: 2`, `hm_ip: 2`, `om_i: 2`, `sm_i: 2`, `st_i: 1`, `tm_i: 1`,
       `tm_s: 1`, `tm_si: 2`, `tm_su: 2`) survive identically with `0` raw `uint64`
       bit mismatches across all `21` float64 output leaves (`540` `+0.0`,
       `18` `-0.0`, `0` NaNs).
   - **Committed unit-test forward digests and negative-zero checks (`tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py`)**:
     Measured now by running:
     `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 taskset -c 0-7 /home/dbalwada/legoESM/.venv/bin/python -m pytest tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py -v`
     (`4 passed in 47.06s`):
     - `test_traced_var_agg_kn2_keeps_parent_forward_bits_and_negative_zeros`
       (`tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py` lines 213–245):
       verifies across `(nlay_i, nlay_s) = (3, 2)` (`148` negative-zero input
       entries across all `16` input arrays) and `(10, 5)` (`318` negative-zero
       input entries across all `16` input arrays) that all `18` surviving `-0.0`
       output entries across the `12` fields in `SURVIVING_NEG_ZERO_FIELDS`
       (`at_ip_eff`, `et_i`, `et_s`, `hm_il`, `hm_ip`, `om_i`, `sm_i`, `st_i`,
       `tm_i`, `tm_s`, `tm_si`, `tm_su`) survive identically in eager, `jax.jit`
       (scan), and unrolled `jax.jit`, with exact digest
       `PARENT_VAR_AGG_DIGEST = "587141e538eafe408a2cee07111a5ad3c6c0b96ac28ff571b60a4ff9527ee2ad"`.
     - `test_traced_o15_step_keeps_parent_forward_bits`
       (`tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py` lines 248–263):
       verifies across `(nlay_i, nlay_s) = (3, 2)` and `(10, 5)` that eager,
       `jax.jit` (traced `fori_loop`), and unrolled `jax.jit` match
       `PARENT_O15_DIGEST = "193e003a7fe31bbf87ab7495e4406f19a032f80925470bf1a5705ced94c6894b"`
       with `0` raw `uint64` bit diffs.

2. **Proof 2 — Planted loop-order swap and planted barrier removal fail `tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py`**:
   Measured now by running `pytest tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py` under each plant and restoring `packages/`:
   - **Plant 1 (loop-order swap `(jl, jk)` -> `(jk, jl)` in `_scan_layer_temp`)**:
     In `packages/ice/legoesm/ice/si3_orca2_state.py` lines 1185–1188, swapping
     `t_seq` and `v_seq` from `(jl, jk)` order:
     `t_seq = jnp.moveaxis(t_arr, (3, 2), (0, 1)).reshape(jpl * nlay, jpi, jpj)`
     `v_seq = jnp.broadcast_to(jnp.moveaxis(v_arr, 2, 0)[:, None, :, :], (jpl, nlay, jpi, jpj)).reshape(jpl * nlay, jpi, jpj)`
     to `(jk, jl)` order:
     `t_seq = jnp.moveaxis(t_arr, (2, 3), (0, 1)).reshape(jpl * nlay, jpi, jpj)`
     `v_seq = jnp.broadcast_to(jnp.moveaxis(v_arr, 2, 0)[None, :, :, :], (nlay, jpl, jpi, jpj)).reshape(jpl * nlay, jpi, jpj)`
     with command:
     `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 taskset -c 0-7 /home/dbalwada/legoESM/.venv/bin/python -m pytest tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py -k test_traced_var_agg_kn2_keeps_parent_forward_bits_and_negative_zeros -v`
     fails `test_traced_var_agg_kn2_keeps_parent_forward_bits_and_negative_zeros` (`exit=1`):
     `FAILED tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py::test_traced_var_agg_kn2_keeps_parent_forward_bits_and_negative_zeros`
     `E AssertionError: assert ('587141e538eafe408a2cee07111a5ad3c6c0b96ac28ff571b60a4ff9527ee2ad', '143944c1febff4c8c0b2db7e654fcc54e29f7376832ea8283611cc507ce8b10e') == ('587141e538eafe408a2cee07111a5ad3c6c0b96ac28ff571b60a4ff9527ee2ad', '587141e538eafe408a2cee07111a5ad3c6c0b96ac28ff571b60a4ff9527ee2ad')`.
     Restored and verified `git status --porcelain` clean on `packages/`.
   - **Plant 2a (barrier removal in `si3_var_agg`)**:
     In `packages/ice/legoesm/ice/si3_orca2_state.py` line 1119, replacing
     `tm_su_num = fortran_sum_categories(sr(t_su * a_i), axis=-1)` with
     `tm_su_num = fortran_sum_categories(t_su * a_i, axis=-1)` under the same
     `pytest -k test_traced_var_agg_kn2_keeps_parent_forward_bits_and_negative_zeros`
     command fails `test_traced_var_agg_kn2_keeps_parent_forward_bits_and_negative_zeros` (`exit=1`):
     `FAILED tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py::test_traced_var_agg_kn2_keeps_parent_forward_bits_and_negative_zeros`
     `E AssertionError: assert ('587141e538eafe408a2cee07111a5ad3c6c0b96ac28ff571b60a4ff9527ee2ad', '5d3f3d5a36a39bdfdd01777fa30901e337f2e051391cf8d2958ac8810f1502ee') == ('587141e538eafe408a2cee07111a5ad3c6c0b96ac28ff571b60a4ff9527ee2ad', '587141e538eafe408a2cee07111a5ad3c6c0b96ac28ff571b60a4ff9527ee2ad')`.
     Restored and verified `git status --porcelain` clean on `packages/`.
   - **Plant 2b (barrier removal in `si3_ice_cor_salinity_bounds` used by `si3_o15_step`)**:
     In `packages/ice/legoesm/ice/itd.py` line 1375, replacing
     `dsz = sr(szv_jk - szv_old)` with `dsz = szv_jk - szv_old` under:
     `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 taskset -c 0-7 /home/dbalwada/legoESM/.venv/bin/python -m pytest tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py -k test_traced_o15_step_keeps_parent_forward_bits -v`
     fails `test_traced_o15_step_keeps_parent_forward_bits` (`exit=1`):
     `FAILED tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py::test_traced_o15_step_keeps_parent_forward_bits`
     `E AssertionError: assert ('193e003a7fe31bbf87ab7495e4406f19a032f80925470bf1a5705ced94c6894b', 'b51c9336794dc4e0a413f86d38dacebb86c946b4c0c27f0f0ff02312639a0afb') == ('193e003a7fe31bbf87ab7495e4406f19a032f80925470bf1a5705ced94c6894b', '193e003a7fe31bbf87ab7495e4406f19a032f80925470bf1a5705ced94c6894b')`.
     Restored and verified `git status --porcelain` clean on `packages/`.

3. **Proof 3 — Per-entry-relative central FD checks on `si3_var_agg(kn=2)` and `si3_o15_step` outputs (`relative <= 1.0e-6`, `g != 0.0`, no absolute floor) and parent-exclusion equivalence at `3dd560f7f05f`**:
   Measured now by running:
   `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 taskset -c 0-7 /home/dbalwada/legoESM/.venv/bin/python -m pytest tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py -v`
   (`4 passed in 47.06s`), and by measuring both `HEAD` (`96bb44f0ae`) and parent `3dd560f7f05f` (via `git show 3dd560f7f0:packages/ice/legoesm/ice/{itd.py,si3_orca2_state.py,si3_finalize.py}` copies in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch/{parent_itd.py,parent_state.py,parent_finalize.py}`, verified by `sha256sum`) with:
   `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 taskset -c 0-7 /home/dbalwada/legoESM/.venv/bin/python -c 'import importlib.util, json, sys; import jax, jax.numpy as jnp, numpy as np; from legoesm.core.precision import PrecisionPolicy, set_policy; import legoesm.ice as ice_pkg; import tests.ice.unit.test_si3_d198_r5_var_agg_o15_traced as tmod; ...'`
   (`SUMMARY: {'var_agg_ad_bit_diff': 0, 'var_agg_fd_bit_diff': 0, 'o15_ad_bit_diff': 0, 'o15_fd_bit_diff': 0, 'o15_zero_diff': 0}`, saved to `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch/parent_exclusions_measured.json`):
   - **`test_traced_var_agg_kn2_reverse_mode_matches_central_differences` (`tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py` lines 315–451)**:
     Evaluates reverse-mode AD cotangents `g` and central finite differences
     `fd = (f(x + h e_k) - f(x - h e_k)) / (2 h)` with `h = 1.0e-4 * abs(x0)`
     and `relative = abs(g - fd) / max(abs(g), abs(fd)) <= 1.0e-6` (`g != 0.0`,
     no absolute floor) separately on each of the `21` `SI3AggregatedVars`
     output fields (`tm_i`, `tm_s`, and all `19` other aggregated fields) with
     respect to every non-zero entry across all `16` input arrays in both
     interior cells `(3, 2)` and `(2, 2)`, as well as on the weighted objective:
     - **Interior cell `(3, 2)`**: **0 exclusions**. All `210` non-zero
       `(out_field, in_name, index)` entries pass `relative <= 1.0e-6` (`20`
       active entries on `tm_i` wrt `v_i` and `t_i`, max rel err `1.294e-9`;
       `15` active entries on `tm_s` wrt `v_s` and `t_s`, max rel err
       `1.355e-9`; `175` active entries across the other `19` aggregated fields,
       max rel err `7.464e-9`). On the weighted objective across all `120`
       entries of cell `(3, 2)`, all `115` active entries pass `relative <= 1.0e-6`
       (max rel err `1.448e-9`) and the `5` entries of `sv_i[3, 2, 0..4]` have
       exact structural zero (`AD == FD == 0.0`) because `si3_var_agg` reads
       `szv_i`, not `sv_i`.
     - **Interior cell `(2, 2)` (the intentional `-1.0e-300 / 1.0e30 -> -0.0` underflow cell)**:
       All `65` non-excluded active `(out_field, in_name, index)` entries pass
       `relative <= 1.0e-6` (max rel err `4.236e-9`). The `19` excluded entries are
       committed in `VAR_AGG_CELL22_EXCLUDED_PARENT_TABLE` (and `VAR_AGG_CELL22_EXCLUDED`,
       `tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py` lines 276–299) and
       checked by the test at lines 376–390 for bitwise `uint64` AD equality and
       exact FD equality (`16e` and `uint64`) against parent `3dd560f7f05f`: `13`
       entries where `a_i[2,2,0] = v_i[2,2,0] = a_ip[2,2,0] = 1.0` (or
       `a_ip_eff[2,2,0] = 0.17556776050637313` or `t_i[2,2,0..2,0]`) is absorbed into
       the category-1 `1.0e30` sum/denominator in forward float64 (`FD == 0.0`), and
       `6` entries (`t_su[2,2,0]`, `t_si[2,2,0]`, `oa_i[2,2,0]`, `v_ip[2,2,0]`,
       `v_il[2,2,0]`, `szv_i[2,2,0,0]`, all `-1.0e-300`) where division by `1.0e30`
       underflows to `-0.0` in forward float64 (`FD == 0.0`):

       | # | `(out_field, in_name, index)` | `x0` | `AD_head` (`96bb44f0ae`) | `AD_parent` (`3dd560f7f05f`) | `AD_uint64` | `FD_head` | `FD_parent` | `FD_uint64` |
       | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
       | `1` | `('at_i', 'a_i', (2, 2, 0))` | `1.0` | `1.0` | `1.0` | `4607182418800017408` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `2` | `('at_ip', 'a_ip', (2, 2, 0))` | `1.0` | `1.0` | `1.0` | `4607182418800017408` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `3` | `('at_ip_eff', 'a_i', (2, 2, 0))` | `1.0` | `0.17556776050637313` | `0.17556776050637313` | `4595493515496893620` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `4` | `('at_ip_eff', 'a_ip_eff', (2, 2, 0))` | `0.17556776050637313` | `1.0` | `1.0` | `4607182418800017408` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `5` | `('ato_i', 'a_i', (2, 2, 0))` | `1.0` | `-1.0` | `-1.0` | `13830554455654793216` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `6` | `('hm_i', 'a_i', (2, 2, 0))` | `1.0` | `-9.999999999999999e-31` | `-9.999999999999999e-31` | `13381399884061196959` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `7` | `('hm_i', 'v_i', (2, 2, 0))` | `1.0` | `9.999999999999999e-31` | `9.999999999999999e-31` | `4158027847206421151` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `8` | `('hm_il', 'v_il', (2, 2, 0))` | `-1e-300` | `9.999999999999999e-31` | `9.999999999999999e-31` | `4158027847206421151` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `9` | `('hm_ip', 'v_ip', (2, 2, 0))` | `-1e-300` | `9.999999999999999e-31` | `9.999999999999999e-31` | `4158027847206421151` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `10` | `('hm_s', 'a_i', (2, 2, 0))` | `1.0` | `-8.941731195714678e-62` | `-8.941731195714678e-62` | `12916997040353963472` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `11` | `('om_i', 'oa_i', (2, 2, 0))` | `-1e-300` | `9.999999999999999e-31` | `9.999999999999999e-31` | `4158027847206421151` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `12` | `('sm_i', 'szv_i', (2, 2, 0, 0))` | `-1e-300` | `9.999999999999999e-31` | `9.999999999999999e-31` | `4158027847206421151` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `13` | `('tm_i', 't_i', (2, 2, 0, 0))` | `265.85944180689484` | `3.333333333333333e-31` | `3.333333333333333e-31` | `4150923644875288105` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `14` | `('tm_i', 't_i', (2, 2, 1, 0))` | `260.71001171568673` | `3.333333333333333e-31` | `3.333333333333333e-31` | `4150923644875288105` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `15` | `('tm_i', 't_i', (2, 2, 2, 0))` | `262.18678397813267` | `3.333333333333333e-31` | `3.333333333333333e-31` | `4150923644875288105` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `16` | `('tm_i', 'v_i', (2, 2, 0))` | `1.0` | `-2.7319640688144486e-30` | `-2.7319640688144486e-30` | `13387992871745285632` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `17` | `('tm_si', 't_si', (2, 2, 0))` | `-1e-300` | `9.999999999999999e-31` | `9.999999999999999e-31` | `4158027847206421151` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `18` | `('tm_su', 't_su', (2, 2, 0))` | `-1e-300` | `9.999999999999999e-31` | `9.999999999999999e-31` | `4158027847206421151` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |
       | `19` | `('vt_i', 'v_i', (2, 2, 0))` | `1.0` | `1.0` | `1.0` | `4607182418800017408` | `0.0000000000000000e+00` | `0.0000000000000000e+00` | `0` |

   - **`test_traced_o15_step_reverse_mode_matches_central_differences` (`tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py` lines 565–764)**:
     Scans all `1,057` non-zero interior entries across the `7` active input
     fields `("a_i", "v_i", "v_s", "oa_i", "v_ip", "e_i", "szv_i")` on
     `_o15_fixture(monkeypatch, 3, 2)` and also checks `25` individual
     `(in_name, in_idx, out_name, out_idx)` output-entry pairs
     (`O15_OUTPUT_ENTRY_FD_CASES`, `tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py`
     lines 466–492) where each single output scalar `out_field[out_idx]` is
     differentiated and compared against its own central difference with no
     cross-field or cross-category reduction:
     - All `1,007` non-excluded active input entries (and all `25` individual
       output-entry pairs) pass `relative <= 1.0e-6` (`max_rel = 1.145e-7`,
       and `AD_head` vs `AD_parent` at `3dd560f7f05f` matches to `max_rel = 2.619e-15`).
     - `11` entries in `O15_EXACT_ZERO_ENTRIES` (`tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py`
       lines 504–518: `a_i[(2,2,0),(2,2,1),(2,3,0),(3,4,0)]` and
       `oa_i[(2,2,0),(2,2,1),(2,3,0),(2,4,0),(3,2,0),(3,3,0),(3,4,0)]`) have
       exact structural zero (`AD == FD == 0.0`, `uint64 == 0` in both HEAD and
       `3dd560f7f05f`) because those categories are zapsmalled (`v_i <= 1.0e-11 < 1.0e-10`)
       and `ice_var_zapsmall` zeroes `a_i` and `oa_i` without routing them into
       water/heat/salt conservation fluxes where `a_ip == 0`.
     - `39` entries in `O15_ZAPSMALL_EXCLUDED_PARENT_TABLE` (and `O15_ZAPSMALL_EXCLUDED_ENTRIES`,
       `tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py` lines 521–562) are in
       categories zapsmalled by `si3_var_zapsmall` (`v_i <= 1.0e-11 < 1.0e-10`) where
       the input magnitude is `abs(x0) <= 1.0e-11`, so the relative step
       `h = 1.0e-4 * abs(x0) <= 1.0e-15` changes `wfx_res`/`hfx_res`/`sfx_res` near
       the float64 rounding floor; every one of the `39` entries is checked by the test
       at lines 706–720 for bitwise `uint64` AD equality and exact FD equality (`16e`
       and `uint64`) against parent `3dd560f7f05f`:

       | # | `(in_name, index)` | `x0` | `AD_head` (`96bb44f0ae`) | `AD_parent` (`3dd560f7f05f`) | `AD_uint64` | `FD_head` (`96bb44f0ae`) | `FD_parent` (`3dd560f7f05f`) | `FD_uint64` |
       | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
       | `1` | `('a_i', (2, 4, 0))` | `1e-12` | `-1.5969134833187906` | `-1.5969134833187906` | `13833242714995840146` | `-1.1928445391805340e+00` | `-1.1928445391805340e+00` | `13831422950249587104` |
       | `2` | `('a_i', (3, 2, 0))` | `1e-12` | `-1.3601818920843725` | `-1.3601818920843725` | `13832176570689769996` | `-1.1841214586599635e+00` | `-1.1841214586599635e+00` | `13831383664987405140` |
       | `3` | `('a_i', (3, 3, 0))` | `1.3999999999999999e-12` | `-1.6111403518142584` | `-1.6111403518142584` | `13833306787115494984` | `-1.3440754603351188e+00` | `-1.3440754603351188e+00` | `13832104033769745789` |
       | `4` | `('szv_i', (2, 2, 0, 0))` | `9.203541154613764e-13` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.7920272466355937e-02` | `1.7920272466355937e-02` | `4580822388282142871` |
       | `5` | `('szv_i', (2, 2, 1, 0))` | `1.7509847978974406e-12` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6954683879567663e-02` | `1.6954683879567663e-02` | `4580544076320565087` |
       | `6` | `('szv_i', (2, 2, 2, 0))` | `3.2365565812017814e-13` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.0191693610668028e-02` | `1.0191693610668028e-02` | `4577028733147105126` |
       | `7` | `('szv_i', (2, 3, 0, 0))` | `5.700386391579139e-14` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6274880161048874e-02` | `1.6274880161048874e-02` | `4580348136239067084` |
       | `8` | `('szv_i', (2, 3, 1, 0))` | `1.168778903581845e-13` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6095708171847693e-02` | `1.6095708171847693e-02` | `4580296493429223777` |
       | `9` | `('szv_i', (2, 3, 2, 0))` | `6.650118822705144e-14` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6275692587254195e-02` | `1.6275692587254195e-02` | `4580348370404977839` |
       | `10` | `('szv_i', (2, 4, 0, 0))` | `3.157103682105899e-12` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6145195673378777e-02` | `1.6145195673378777e-02` | `4580310757230404890` |
       | `11` | `('szv_i', (2, 4, 1, 0))` | `1.048120611074301e-12` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6144542613306272e-02` | `1.6144542613306272e-02` | `4580310568998654542` |
       | `12` | `('szv_i', (3, 2, 0, 0))` | `5.271233943396759e-13` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6133226905047007e-02` | `1.6133226905047007e-02` | `4580307307467806551` |
       | `13` | `('szv_i', (3, 2, 1, 0))` | `1.6723821968604005e-12` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6148989054869552e-02` | `1.6148989054869552e-02` | `4580311850598178863` |
       | `14` | `('szv_i', (3, 2, 2, 0))` | `2.1124912319114183e-12` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6145393112562589e-02` | `1.6145393112562589e-02` | `4580310814138375107` |
       | `15` | `('szv_i', (3, 3, 0, 0))` | `2.747763155990917e-13` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6131276830154022e-02` | `1.6131276830154022e-02` | `4580306745396986622` |
       | `16` | `('szv_i', (3, 3, 1, 0))` | `3.5155317499204345e-13` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6126913238721348e-02` | `1.6126913238721348e-02` | `4580305487677386610` |
       | `17` | `('szv_i', (3, 3, 2, 0))` | `1.0510877736488021e-12` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6157197119392463e-02` | `1.6157197119392463e-02` | `4580314216411703779` |
       | `18` | `('szv_i', (3, 4, 0, 0))` | `4.211957085090081e-13` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6152493780992420e-02` | `1.6152493780992420e-02` | `4580312860766707566` |
       | `19` | `('szv_i', (3, 4, 1, 0))` | `1.1893969339273605e-13` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6033358314652225e-02` | `1.6033358314652225e-02` | `4580278522306431322` |
       | `20` | `('szv_i', (3, 4, 2, 0))` | `2.8118972342495787e-13` | `0.016145254201342288` | `0.016145254201342288` | `4580310774099941828` | `1.6129942397149492e-02` | `1.6129942397149492e-02` | `4580306360772859777` |
       | `21` | `('v_i', (2, 2, 0))` | `7.497709038862792e-13` | `0.08486547647313808` | `0.08486547647313808` | `4590779622714701636` | `8.5274498820328198e-02` | `8.5274498820328198e-02` | `4590809095880947902` |
       | `22` | `('v_i', (2, 2, 1))` | `1e-11` | `0.08486547647313808` | `0.08486547647313808` | `4590779622714701636` | `8.5180966201768432e-02` | `8.5180966201768432e-02` | `4590802356145490418` |
       | `23` | `('v_i', (2, 3, 0))` | `5.915295163755829e-14` | `0.08486547647313808` | `0.08486547647313808` | `4590779622714701636` | `8.4728801607131976e-02` | `8.4728801607131976e-02` | `4590769774252691780` |
       | `24` | `('v_i', (2, 4, 0))` | `1.2249264837930447e-12` | `0.2605453760682258` | `0.2605453760682258` | `4598365188152201790` | `-1.4344312192269018e-02` | `-1.4344312192269018e-02` | `13802794591633459816` |
       | `25` | `('v_i', (3, 2, 0))` | `9.732754483907965e-13` | `0.23492007647377722` | `0.23492007647377722` | `4597631908041488636` | `3.4002435265856412e-01` | `3.4002435265856412e-01` | `4599796954109625934` |
       | `26` | `('v_i', (3, 3, 0))` | `6.530684666460284e-13` | `0.23502659082625255` | `0.23502659082625255` | `4597635745625473576` | `1.9470509720113369e+00` | `1.9470509720113369e+00` | `4611447557204668531` |
       | `27` | `('v_i', (3, 4, 0))` | `1.637735903829609e-13` | `0.08486547647313808` | `0.08486547647313808` | `4590779622714701636` | `8.4778505254865633e-02` | `8.4778505254865633e-02` | `4590773355777962376` |
       | `28` | `('v_ip', (2, 2, 0))` | `6.865098269936168e-15` | `0.09254686638292048` | `0.09254686638292048` | `4591333125190467772` | `1.5785160010130883e-01` | `1.5785160010130883e-01` | `4594855223549704152` |
       | `29` | `('v_ip', (2, 3, 0))` | `3.161963887628452e-14` | `0.09254686638292048` | `0.09254686638292048` | `4591333125190467772` | `9.2105871079622481e-02` | `9.2105871079622481e-02` | `4591301348129930092` |
       | `30` | `('v_ip', (2, 4, 0))` | `6.186910170094338e-15` | `0.09254686638292048` | `0.09254686638292048` | `4591333125190467772` | `9.8524550752353252e-02` | `9.8524550752353252e-02` | `4591763862744047226` |
       | `31` | `('v_ip', (3, 2, 0))` | `5.728345009025055e-16` | `0.09254686638292048` | `0.09254686638292048` | `4591333125190467772` | `1.1823514743949007e-01` | `1.1823514743949007e-01` | `4593184160918374258` |
       | `32` | `('v_ip', (3, 3, 0))` | `8.80780742621719e-14` | `0.09254686638292048` | `0.09254686638292048` | `4591333125190467772` | `9.2660577049623633e-02` | `9.2660577049623633e-02` | `4591341318907526850` |
       | `33` | `('v_ip', (3, 4, 0))` | `1.380442936455338e-14` | `0.09254686638292048` | `0.09254686638292048` | `4591333125190467772` | `9.3220388021572426e-02` | `9.3220388021572426e-02` | `4591381657539281514` |
       | `34` | `('v_s', (2, 2, 0))` | `1.94732500889082e-13` | `0.030540465906363758` | `0.030540465906363758` | `4584459911384463690` | `3.0606945835850448e-02` | `3.0606945835850448e-02` | `4584479072919546178` |
       | `35` | `('v_s', (2, 3, 0))` | `7.154584972108825e-15` | `0.030540465906363758` | `0.030540465906363758` | `4584459911384463690` | `2.8399622872605775e-02` | `2.8399622872605775e-02` | `4583842855391561855` |
       | `36` | `('v_s', (2, 4, 0))` | `1.0216707750938762e-13` | `0.030540465906363758` | `0.030540465906363758` | `4584459911384463690` | `3.0494577831639683e-02` | `3.0494577831639683e-02` | `4584446685047425092` |
       | `37` | `('v_s', (3, 2, 0))` | `2.5604556586049365e-13` | `0.030540465906363758` | `0.030540465906363758` | `4584459911384463690` | `3.0552059364509693e-02` | `3.0552059364509693e-02` | `4584463252971265992` |
       | `38` | `('v_s', (3, 3, 0))` | `6.411126668060193e-14` | `0.030540465906363758` | `0.030540465906363758` | `4584459911384463690` | `3.0636518044350049e-02` | `3.0636518044350049e-02` | `4584487596528325655` |
       | `39` | `('v_s', (3, 4, 0))` | `1.8207948515121027e-14` | `0.030540465906363758` | `0.030540465906363758` | `4584459911384463690` | `2.9758068182755625e-02` | `2.9758068182755625e-02` | `4584234400594287875` |

   - **Planted AD failures (`stop_gradient`, `1 + 1e-5` tangent scaling, and parent-exclusion table edits)**:
     Measured now by running `pytest tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py`
     under each plant and restoring `packages/` (or scratch copies):
     - **Plant 3a (`jax.lax.stop_gradient` on `_scan_layer_temp` carry)**:
       In `packages/ice/legoesm/ice/si3_orca2_state.py` line 1193, replacing
       `return jnp.where(has_vol, acc_next, rt0), None` with
       `return jax.lax.stop_gradient(jnp.where(has_vol, acc_next, rt0)), None`
       under:
       `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 taskset -c 0-7 /home/dbalwada/legoESM/.venv/bin/python -m pytest tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py -k test_traced_var_agg_kn2_reverse_mode_matches_central_differences -v`
       fails `test_traced_var_agg_kn2_reverse_mode_matches_central_differences` (`exit=1`):
       `FAILED tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py::test_traced_var_agg_kn2_reverse_mode_matches_central_differences`
       `E AssertionError: assert [('tm_i', 'v_...973, 1.0), ...] == []`
       `E Left contains 54 more items, first extra item: ('tm_i', 'v_i', (3, 2, 0), 0.0, 1.3893957293976973, 1.0)`.
       Restored and verified `git status --porcelain` clean on `packages/`.
     - **Plant 3b (`1 + 1e-5` tangent scaling on `_scan_layer_temp` carry)**:
       In `packages/ice/legoesm/ice/si3_orca2_state.py` line 1193, scaling the
       step carry tangent by `1.0 + 1.0e-5` via
       `_y = jnp.where(has_vol, acc_next, rt0); return jax.lax.stop_gradient(_y) + (1.0 + 1.0e-5) * (_y - jax.lax.stop_gradient(_y)), None`
       under the same `pytest -k test_traced_var_agg_kn2_reverse_mode_matches_central_differences`
       command fails `test_traced_var_agg_kn2_reverse_mode_matches_central_differences` (`exit=1`):
       `FAILED tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py::test_traced_var_agg_kn2_reverse_mode_matches_central_differences`
       `E AssertionError: assert [('tm_i', 'v_...346489), ...] == []`
       `E Left contains 54 more items, first extra item: ('tm_i', 'v_i', (3, 2, 0), 1.3997521885923163, 1.3893957293976973, 0.007398780388346489)`.
       Restored and verified `git status --porcelain` clean on `packages/`.
     - **Plant 3c (editing one excluded entry's parent value in `VAR_AGG_CELL22_EXCLUDED_PARENT_TABLE` in scratch)**:
       Copying `tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py` to
       `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch/test_plant_var_agg_parent.py`
       and editing `("at_ip_eff", "a_i", (2, 2, 0)): (0.17556776050637313, 0.0)`
       by 1 ULP to `(0.17556776050637316, 0.0)` under:
       `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 taskset -c 0-7 /home/dbalwada/legoESM/.venv/bin/python -m pytest /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch/test_plant_var_agg_parent.py -k test_traced_var_agg_kn2_reverse_mode_matches_central_differences -v`
       fails `test_traced_var_agg_kn2_reverse_mode_matches_central_differences` (`exit=1`):
       `FAILED ../../../work/si3_d198r5b_scratch/test_plant_var_agg_parent.py::test_traced_var_agg_kn2_reverse_mode_matches_central_differences`
       `E AssertionError: assert [('at_ip_eff'...0637316, ...)] == []`
       `E Left contains one more item: ('at_ip_eff', 'a_i', (2, 2, 0), 0.17556776050637313, 0.0, 0.17556776050637316, ...)`.
       Removed scratch file and verified `git status --porcelain` clean.
     - **Plant 4a (`jax.lax.stop_gradient` on `_si3_loop` carry used by `si3_o15_step`)**:
       In `packages/ice/legoesm/ice/itd.py` line 1165, wrapping the `fori_loop`
       body output with `jax.tree_util.tree_map(jax.lax.stop_gradient, body(idx, val))`
       under:
       `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 taskset -c 0-7 /home/dbalwada/legoESM/.venv/bin/python -m pytest tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py -k test_traced_o15_step_reverse_mode_matches_central_differences -v`
       fails `test_traced_o15_step_reverse_mode_matches_central_differences` (`exit=1`):
       `FAILED tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py::test_traced_o15_step_reverse_mode_matches_central_differences`
       `E AssertionError: assert [('a_i', (2, ...86, 1.0), ...] == []`
       `E Left contains 1038 more items, first extra item: ('a_i', (2, 2, 2), 0.0, 0.6173060059866636, 1.0)`.
       Restored and verified `git status --porcelain` clean on `packages/`.
     - **Plant 4b (`1 + 1e-5` tangent scaling on `_si3_loop` carry used by `si3_o15_step`)**:
       In `packages/ice/legoesm/ice/itd.py` line 1165, scaling floating carry
       tangents by `1.0 + 1.0e-5` via
       `jax.lax.stop_gradient(x) + (1.0 + 1.0e-5) * (x - jax.lax.stop_gradient(x))`
       under the same `pytest -k test_traced_o15_step_reverse_mode_matches_central_differences`
       command fails `test_traced_o15_step_reverse_mode_matches_central_differences` (`exit=1`):
       `FAILED tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py::test_traced_o15_step_reverse_mode_matches_central_differences`
       `E AssertionError: assert [('a_i', (2, ...156e-05), ...] == []`
       `E Left contains 1038 more items, first extra item: ('a_i', (2, 2, 2), 0.6173553964270941, 0.6173060059866636, 8.000325374387586e-05)`.
       Restored and verified `git status --porcelain` clean on `packages/`.
     - **Plant 4c (editing one excluded entry's parent value in `O15_ZAPSMALL_EXCLUDED_PARENT_TABLE` in scratch)**:
       Copying `tests/ice/unit/test_si3_d198_r5_var_agg_o15_traced.py` to
       `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch/test_plant_o15_parent.py`
       and editing `("a_i", (2, 4, 0)): (-1.5969134833187906, -1.192844539180534)`
       to `(-1.5969134833187906, -1.192844539180535)` under:
       `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 taskset -c 0-7 /home/dbalwada/legoESM/.venv/bin/python -m pytest /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch/test_plant_o15_parent.py -k test_traced_o15_step_reverse_mode_matches_central_differences -v`
       fails `test_traced_o15_step_reverse_mode_matches_central_differences` (`exit=1`):
       `FAILED ../../../work/si3_d198r5b_scratch/test_plant_o15_parent.py::test_traced_o15_step_reverse_mode_matches_central_differences`
       `E AssertionError: assert [('a_i', (2, ...844539180535)] == []`
       `E Left contains one more item: ('a_i', (2, 4, 0), -1.5969134833187906, -1.192844539180534, -1.5969134833187906, -1.192844539180535)`.
       Removed scratch file and verified `git status --porcelain` clean.

4. **Proof 4 — Equation counter updated with historical snapshots kept**:
   Measured now by running:
   `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 taskset -c 0-7 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_jaxpr_count.py --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch/jaxpr_count.json`
   (`status: "PASS"`, exit code `0`):
   `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_jaxpr_count.py`
   preserves `REVIEWED_EA3E570` (`576,364`), `PRE_D194_9998276020` (`577,982`),
   `D194_48526F950C` (`227,840`), `D194_RESUME1_B17F14B73C` (`228,050`),
   `D198_R1_743146B590` (`198,323`), `D198_R2_56B768300A` (`178,146`), and
   `D198_R3_22AF29B743` (`162,751` total recursive equations, `88,876`
   `remaining_named_and_inlined`), and updates `EXPECTED` to Round 5's measured
   counts:
   - `recursive_total`: `162,751` -> `141,340` (`-21,411` equations, `-13.16%`
     vs Round 3 `162,751`; `-436,642` / `-75.55%` vs `PRE_D194_9998276020`
     `577,982`).
   - `remaining_named_and_inlined`: `88,876` -> `67,465` (`-21,411`, well below
     the `110,000` ceiling).
   - `_thermo_category_loop`: `35,918` (unchanged).
   - `si3_ice_thd_do`: `9,809` (unchanged).
   - `si3_dyn_orca2`: `28,148` (unchanged).

5. **Proof 5 — Citation gate re-anchored and passing**:
   Re-anchored the `29` shifted citations in `packages/ice/legoesm/ice/si3_orca2_state.py`
   and `2` shifted citations in `packages/ice/legoesm/ice/si3_finalize.py` in
   `scripts/validate/ocean_fidelity/testcases/nemo_testcase_receipt_citation_gate.py`.
   Measured now by running:
   `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_d198r5b_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 taskset -c 0-7 /home/dbalwada/legoESM/.venv/bin/python -m pytest tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py -v`
   (`45 passed`).

independent review: operator

## O17 inactive-category NaN fix

### 1. Bug diagnosis and NEMO gather/scatter semantics (`MEASURED`)

At `96bb44f0aeee`, `si3_o17_step` (`packages/ice/legoesm/ice/si3_o17.py` composing O12
thermodynamics via `packages/ice/legoesm/ice/si3_thd_pnd.py` lines 874–990 and
`packages/ice/legoesm/ice/bitz_lipscomb.py`) executed category thermodynamics on
categories that NEMO 5.0.2 excludes:

- **NEMO gather/scatter (`src/ICE/icethd.F90` lines 117–165 and 336–526, `src/ICE/icetab.F90` lines 191–216, `src/ICE/icestp.F90` lines 438–501)**:
  Inside `DO jl = 1, jpl` (`src/ICE/icethd.F90` line 127), NEMO loops over owned
  interior points `DO_2D( 0, 0, 0, 0 )` (`src/ICE/icethd.F90` line 130) and gathers
  only `a_i(ji,jj,jl) > epsi10` (`1.0e-10`) into `nptidx(1:npti)`
  (`src/ICE/icethd.F90` lines 131–135). The category sequence
  `ice_thd_1d2d(jl, 1) -> ice_thd_zdf -> ice_thd_dh -> ice_thd_temp -> ice_thd_sal -> ice_thd_temp -> ice_thd_da -> ice_thd_1d2d(jl, 2)`
  executes only under `IF( npti > 0 )` (`src/ICE/icethd.F90` lines 138–165), and
  `ice_thd_1d2d(jl, 2)` scatters only `1:npti` via `tab_1d_2d`
  (`src/ICE/icethd.F90` lines 456–526, `src/ICE/icetab.F90` lines 191–216).
  Categories with `a_i(ji,jj,jl) <= epsi10` (or outside `DO_2D(0,0,0,0)`):
  - leave pre-thermo state (`a_i`, `h_i`, `h_s`, `t_su`, `s_i`, `o_i`, `v_i`, `v_s`,
    `sv_i`, `oa_i`, `t_i`, `t_s`, `e_i`, `e_s`, `sz_i`, `szv_i`) untouched
    (`src/ICE/icethd.F90` lines 456–471 and 523–526);
  - add `0._wp` to cumulative 2D fluxes (`wfx_snw_sni`, `wfx_snw_sum`, `wfx_sub`,
    `wfx_snw_sub`, `wfx_ice_sub`, `wfx_err_sub`, `wfx_bog`, `wfx_bom`, `wfx_sum`,
    `wfx_sni`, `wfx_spr`, `sfx_sni`, `sfx_bri`, `sfx_sub`, `sfx_bog`, `sfx_bom`,
    `sfx_sum`, `hfx_thd`, `hfx_spr`, `hfx_snw`, `hfx_sub`, `hfx_bog`, `hfx_bom`,
    `hfx_sum`, `hfx_sni`, `hfx_dif`, `hfx_err_dif`, `wfx_lam`, `hfx_lam`, `sfx_lam`
    at `src/ICE/icethd.F90` lines 161 and 473–507, initialized to `0._wp` in
    `diag_set0` at `src/ICE/icestp.F90` lines 438–465);
  - keep `qcn_ice_bot(:,:,jl) = 0._wp`, `qcn_ice_top(:,:,jl) = 0._wp`,
    `cnd_ice(:,:,jl) = 0._wp`, `qtr_ice_bot(:,:,jl) = 0._wp`,
    `qml_ice(:,:,jl) = 0._wp`, `dh_i_sum_2d(:,:,jl) = 0._wp`, and
    `dh_s_sum_2d(:,:,jl) = 0._wp` (`src/ICE/icethd.F90` lines 510–521, initialized
    in `diag_set0` at `src/ICE/icestp.F90` lines 489–495);
  - keep `t_si(:,:,jl) = rt0` (`273.15`, initialized in `diag_set0` on `A2D(0)` at
    `src/ICE/icestp.F90` line 488 and updated only for `1:npti` at
    `src/ICE/icethd.F90` line 518).
- **Unguarded failure modes at `96bb44f0aeee`**:
  1. Forward `0/0 -> NaN` in snow thickness update at
     `packages/ice/legoesm/ice/bitz_lipscomb.py` line 1156 when `sprecip > 0`,
     `zsnw = 0`, and `at_i = 0`, where `jnp.where(at_i > 0.0, sprecip * dt * (a_i / at_i) / rho_s, 0.0)`
     evaluates `0.0 / 0.0` in the unselected branch before `nemo_source_round` (`sr`)
     propagates `copysign(abs(NaN), NaN) = NaN` through `wfx_spr`, `hfx_spr`,
     `wfx_snw_sub`, `wfx_sub`, `wfx_err_sub`, `hfx_sub`, `emp`, `fwfice`, and `qns`.
     On REAL ORCA2 step 1 (`kt = 1`, `rank = 0`), this produced `1,813` NaNs at
     open-water snowfall points in those 9 fields where NEMO's `kind = 28` record is
     finite, plus `36,305` mismatches in inactive-category `t_si`.
  2. Reverse-mode AD `0/0 -> NaN` on the probe tile (`objective = NaN` and `31` of
     `37` input gradients containing `304` `NaN` entries) from open-water snowfall at
     owned cell `(0, 0)` (`full (2, 2)`) and hidden `0/0` divisions inside unguarded
     inactive columns at `packages/ice/legoesm/ice/bitz_lipscomb.py` lines 633–636
     (`k_si = 0.0 / 0.0` when `h_i = h_s = 0`) and lines 745–748 (snow-free Thomas
     surface row `0.0 / 0.0` when `dqns_ice = k_i_half = 0`).

### 2. Model changes (exact NEMO gather/scatter; no knob, no default change)

1. **`packages/ice/legoesm/ice/bitz_lipscomb.py` lines 2495–2508**:
   Extended `guard_inactive_columns(value, active_columns, dtype, inactive_fill=None)`
   to accept an optional constant `inactive_fill` (defaulting to `value` when `None`)
   placed under `jax.lax.stop_gradient` on `~active_columns`, reusing the existing
   `guard_inactive_columns` mechanism rather than introducing a new one.
2. **`packages/ice/legoesm/ice/si3_o17.py` lines 382–456, 824, 899–905, 1008–1030**:
   - Added `_owned_mask(shape, h)` (`src/ICE/icethd.F90` line 130 `DO_2D(0,0,0,0)`)
     and set `active = owned_mask & (state.a_i[..., jl] > epsi10)`
     (`src/ICE/icethd.F90` lines 130–135) in `_thermo_entries`, passing pre-thermo
     2D state (`e_i_2d`, `e_s_2d`, `v_i`, `v_s`, `sv_i`, `oa_i`, `szv_i`, `t_si`).
   - Masked `thermo_state.t_si` with `entry["active"]` so inactive categories retain
     `state.t_si[..., jl]` (`rt0 = 273.15` on `owned`, `src/ICE/icethd.F90` line 518,
     `src/ICE/icestp.F90` line 488).
   - Masked `qtr_bot`, `qcn_top`, `qcn_bot`, and `qml_ice` with `thermo_active_owned`
     to `0.0` (`src/ICE/icestp.F90` lines 489–493, `src/ICE/icethd.F90` lines 510–521)
     so inactive-category values cannot leak into
     `packages/ice/legoesm/ice/si3_finalize.py` lines 937–988.
3. **`packages/ice/legoesm/ice/si3_thd_pnd.py` lines 883–1245**:
   - In `_thermo_category_loop`, resolved `active = raw_entry["active"]` (or
     `raw_entry["a_i"] > constants.epsi10` when `"active"` is omitted), guarded all
     category inputs and carried fluxes on `~active` via
     `guard_inactive_columns(..., inactive_fill=...)` before calling
     `si3_ice_thd_o10_category_stages` and `si3_ice_thd_da` so inactive lanes cannot
     produce non-finite values or cotangents, and selected/scattered category outputs
     (`SI3O12CategoryOutput`, `o10.o9.zdf`, `o10.o9.dh`, `o10.sal`, `da_out`) and
     carried fluxes only on `active` (`_where_active`).

### 3. Proofs

1. **Proof 1 — Every ACTIVE-category forward bit unchanged vs `96bb44f0aeee` (`0` ULP, signed zeros and NaN bits included, eager AND `jax.jit`) on O12 fixtures, O8–O16 master scorer, and probe tile**:
   - **O12 fixtures and probe tile (`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_nanfix_scratch/proof1_active_bits.json`)**:
     Measured by running `scripts/validate/ocean_fidelity/testcases/si3_nanfix/verify_proof1_active_bits.py` across `--case o12_5cat`,
     `--case d194`, and `--case probe_tile`:
     - `o12_5cat_fixture` (`eager_vs_parent_96bb44f0aeee`, `jit_vs_parent_96bb44f0aeee`, `eager_vs_jit`):
       `{"compared": 32250, "differing": 0, "equal": 32250, "max_ulp": 0}`
     - `d194_fixture` (`eager_vs_parent_96bb44f0aeee`, `jit_vs_parent_96bb44f0aeee`, `eager_vs_jit`):
       `{"compared": 38700, "differing": 0, "equal": 38700, "max_ulp": 0}`
     - `probe_tile` (`eager_vs_parent_96bb44f0aeee`, `jit_vs_parent_96bb44f0aeee`, `eager_vs_jit`):
       `{"compared": 1075, "differing": 0, "equal": 1075, "max_ulp": 0}`
     - Final line: `"status": "PASS_0_ULP_ACTIVE_CATEGORIES"`
   - **O8–O16 master scorer (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o8_o12_master_scorer.py`) per rung group (each under 20 min; full `1..100` in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_nanfix_scratch/operator_run.sh`)**:
     - `--rung O8-O13 --steps 1,2,100`: `"status": "PASS_ALL_O8_O13_0_ULP"`
     - `--rung O8-O13 --steps 1..20`: `"status": "PASS_ALL_O8_O13_0_ULP"`
     - `--rung O13 --steps 1..100`: `"status": "PASS_O13_0_ULP"`
     - `--rung O14 --steps 1..100`: `"status": "PASS_O14_0_ULP"`
     - `--rung O15-O16 --steps 1..100`: `"status": "PASS_ALL_O15_O16_0_ULP"`

2. **Proof 2 — On REAL ORCA2 step 1 (`kt = 1`, `rank = 0`), eager `si3_o17_step` has `0` NaN in owned outputs and fluxes, and the 9 fields at the `1,813` open-water snowfall points equal NEMO's record bit for bit (`0` ULP)**:
   Measured by running `scripts/validate/ocean_fidelity/testcases/si3_nanfix/verify_proof2_orca2_step1.py`
   (`elapsed_seconds: 535.72`, output `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_nanfix_scratch/proof2_orca2_step1.json`):
   - `total_owned_nans`: `0` across all `18` owned state fields, `43` owned flux fields,
     `14` owned output fields, and `5` owned stress fields.
   - `open_water_snowfall_points`: `1813` (`at_i_b <= 1.0e-10` and `sprecip > 0.0`).
   - Per-field comparison at the `1,813` open-water snowfall points against NEMO's
     `kind = 28` record:

     | Field | Compared | Equal | Differing | Max ULP |
     |---|---:|---:|---:|---:|
     | `hfx_spr` | 1,813 | 1,813 | 0 | 0 |
     | `hfx_sub` | 1,813 | 1,813 | 0 | 0 |
     | `wfx_err_sub` | 1,813 | 1,813 | 0 | 0 |
     | `wfx_snw_sub` | 1,813 | 1,813 | 0 | 0 |
     | `wfx_spr` | 1,813 | 1,813 | 0 | 0 |
     | `wfx_sub` | 1,813 | 1,813 | 0 | 0 |
     | `emp` | 1,813 | 1,813 | 0 | 0 |
     | `fwfice` | 1,813 | 1,813 | 0 | 0 |
     | `qns` | 1,813 | 1,813 | 0 | 0 |

   - Inactive-category `t_si` (`a_i <= 1.0e-10`) across the full ORCA2 domain:
     - `t_si_inactive_categories_kind27`: `{"compared": 114415, "differing": 0, "equal": 114415, "max_ulp": 0}`
     - `t_si_inactive_categories_kind28`: `{"compared": 114415, "differing": 0, "equal": 114415, "max_ulp": 0}`
   - Final line: `"status": "PASS"`

3. **Proof 3 — Unit test failing under the `96bb44f0aeee` unguarded plant and passing now on a small tile with an open-water snowfall column**:
   `tests/ice/unit/test_si3_o17.py::test_o17_inactive_category_open_water_snowfall_nan_fix_and_plant`
   (`tests/ice/unit/test_si3_o17.py` lines 174–271) runs `si3_o17_step` on the `6x5`
   (`2x1` owned) tile where owned cell `(0, 0)` (`full (2, 2)`) is open water with
   snowfall (`at_i = 0.0`, `sprecip = 4.0e-8 > 0.0`) and owned cell `(1, 0)`
   (`full (3, 2)`) is active 5-category sea ice:
   - Passes now with `0` NaNs in all owned state, flux, output, and stress fields,
     `0.0` at `(0, 0)` for `hfx_spr`, `hfx_sub`, `wfx_err_sub`, `wfx_snw_sub`,
     `wfx_spr`, `wfx_sub`, and `fwfice`, finite `emp` (`2.0e-8`) and `qns`, and
     `t_si[2, 2, :] == 273.15`.
   - Under the planted `96bb44f0aeee` unguarded path (`active = True` for all
     categories in `_thermo_entries`), produces `NaN` at `(0, 0)` in all 9 fields
     (`hfx_spr`, `hfx_sub`, `wfx_err_sub`, `wfx_snw_sub`, `wfx_spr`, `wfx_sub`,
     `emp`, `fwfice`, `qns`) and corrupts `t_si[2, 2, :]` (`266.89337065`), while
     preserving `0` ULP (`uint64` bit-identical) agreement on all active-category
     outputs at `(1, 0)` (`full (3, 2)`).

4. **Proof 4 — AD probe on the small tile (prefix through O12, `--prefixes 8`, under 20 min) gives a finite objective and `0` NaN gradients**:
   Measured by running:
   `TMPDIR=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_nanfix_scratch CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src timeout 1200 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_ad_probe.py --prefixes 8 --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_nanfix_scratch/proof4_ad_probe_prefix8.json`
   - `prefix_1_8_o12_thermo`:
     - `"objective": -7.963900479420319e+18`, `"objective_is_finite": true`
     - `"gradient_nan_entries": 0`, `"gradient_inf_entries": 0`, `"gradient_finite_entries": 38425`, `"gradient_nonzero_entries": 8957`
     - `"trace_plus_lower_plus_compile_seconds": 651.201`, `"execution_seconds": 1.3162`, `"peak_rss_gib": 24.707`
   - Final line: `"status": "PASS"`

5. **Proof 5 — Per-entry-relative central FD on the newly guarded outputs (`relative <= 1.0e-6`, `ad != 0.0`, no absolute floor), failing under `stop_gradient` and `1 + 1e-5` tangent plants**:
   `tests/ice/unit/test_si3_o17.py::test_o17_newly_guarded_outputs_reverse_mode_matches_central_differences_and_plants`
   (`tests/ice/unit/test_si3_o17.py` lines 274–662) verifies `0` NaNs across all
   forward outputs and all `15` reverse-mode input cotangent arrays on the mixed
   open-water snowfall / active sea-ice tile, checks per-entry-relative central
   finite differences (`relative = abs(ad - fd) / max(abs(ad), abs(fd)) <= 1.0e-6`,
   `ad != 0.0`) across all `21` output-entry pairs covering `t_si` (active and
   inactive passthrough), `qtr_bot`, `qcn_top`, `qcn_bot`, `qml_ice`, `t_su`, `a_i`,
   `v_i`, `v_s`, `e_i`, `e_s`, `szv_i`, `sfx_bri`, `hfx_thd`, `hfx_dif`, `hfx_sub`,
   `wfx_snw_sub`, `wfx_ice_sub`, `hfx_spr`, and `wfx_spr` (`max_rel = 6.723e-8`),
   and proves that both the `jax.lax.stop_gradient` plant and the `1 + 1e-5`
   tangent-scaling plant fail the check.

6. **Equation counter (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_jaxpr_count.py`)**:
   Preserves `D198_R5_96BB44F0AE` (`recursive_total: 141,340`) and updates `EXPECTED`
   to the measured counts with inactive-category guarding (`recursive_total: 141,935`,
   `_thermo_category_loop: 36,355`, `si3_ice_thd_do: 9,809`, `si3_dyn_orca2: 28,148`,
   `remaining_named_and_inlined: 67,623`; `"status": "PASS"`).

independent review: operator


## Round `si3_o17_free` (DECISION 210, 2026-11-03): parallel step ranges and carried-state free run

Date: 2026-11-04 (Resume 4). Base: `94a4069d61ee` (`si3_r29`). No NEMO build or integration was run, and `packages/` is untouched (`git diff --stat 94a4069d61ee..HEAD -- packages` is empty).

### (A) Parallel per-step ranges (`--steps START..END`) and exact report merge (`--merge`)

The composition scorer (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:450-877` and `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:886-949`) accepts `--steps START..END` (`parse_step_range`, 1-based inclusive) and `--merge <range1.json> ...` (`merge_reports`).

- **Retraction of split free-run step ranges (Resume 2 Finding 1):** The Round-1 `exit_state` helper has been removed. In free-run mode (`free_run`, or `score` when `"INDEPENDENT"` is in `labels`), restarting a range `START > 1` from NEMO's recorded state would reset an early trajectory divergence back to NEMO rather than carrying our own trajectory. The free run is inherently serial: `free_run` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:758-877`) refuses any `step_range` other than one contiguous range `1..target_steps` starting at step `1`, `score` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:675-755`) refuses any `step_range` with `start != 1` when `"INDEPENDENT"` is in `labels`, and `merge_reports` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:552-672`) refuses `FREE_RUN` reports and multi-range reports containing `INDEPENDENT`. Step ranges (`--steps START..END`) and `--merge` are restricted to per-step scoring (`PER_STEP_LABELS = ("GIVEN-NEMO-ENTRY",)`, `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:450-456` and `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:886-949`).
- `merge_reports` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:552-672`) verifies that every step in `1..expect_steps` is present once across all ranks, recomputes per-field, per-step, per-label, and top-level `PASS`/`FAIL` verdicts with the exact serial rules, and refuses any missing step (`O17 merge requires steps 1..N; missing=[...]`) or duplicated step (`duplicate O17 step(s) in merge: [...]`).
- Verified on the model-built 2-step synthetic record (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_synthetic_record.py:157-202` and `tests/ice/unit/test_si3_o17.py:587-771`):
  - **Parent `94a4069d61` serial byte identity (Resume 2 Finding 2):** Loading the parent scorer from `94a4069d61:scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py` (`_load_parent_scorer_94a4069d61`, `tests/ice/unit/test_si3_o17.py:587-655`) and running both `parent_scorer.score` and the new scorer's default serial `scorer.score` on the same 2-step synthetic record produces byte-for-byte identical JSON reports (`138,424` bytes, `sha256 = 0998de68d2b0869cbb72915767d9e7b73cbc251f62050fda7bfa1ed972fd59b3`) with the three named timing fields (`scorer_wall_seconds`, `scorer_rss_kib_after_each_step`, `scorer_peak_rss_kib`) excepted (`strip_timing`), and byte-for-byte identical full JSON reports when `time.monotonic` and `_rss_kib` are monkeypatched identically on both modules.
  - Scoring per-step ranges (`PER_STEP_LABELS = ("GIVEN-NEMO-ENTRY",)`) on `--steps 1..1` and `--steps 2..2` separately and merging via `merge_reports` equals the serial per-step report (`strip_timing(merged) == strip_timing(serial_per_step)`, `split_merge_equals_serial = true`) for both the passing identical record and a planted 1-ULP `FAIL` record.
  - Split free-run ranges (`free_run(..., step_range=(2, 2))`, `free_run(..., step_range=(1, 1))` on a 2-step run, `score(..., VALID_LABELS, step_range=(2, 2))`, and `merge_reports` on multi-range `INDEPENDENT` reports) are refused with `ValueError` (`split_free_run_refused = true`). On the early-divergence plant (`state + 2.0` vs `expected = float(kt)`), `parent_scorer.score` (`94a4069d61`), serial `scorer.score` (`INDEPENDENT`), and serial `scorer.free_run` all produce the exact same step-2 `max_relative = 1.0` (`tests/ice/unit/test_si3_o17.py:669-739`).
  - Merging with step 1 or step 2 omitted raises `ValueError` (`missing_step_refused = true`), and merging with step 1 or step 2 duplicated raises `ValueError` (`duplicated_step_refused = true`).

### (B) Carried-state free run (`--free-run`) and `CARRIED` vs `EXTERNAL` input classification

`free_run` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:758-877`) initializes step `kt = 1` from NEMO's recorded state before step 1 (`entry_state`, kinds `22` and `29`). For every subsequent step `kt = 2..N`, the step-entry state `entry: SI3O17StepState` comes strictly from our previous step's output `out` via `carry_step_state` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:489-496`); only external inputs (`forcing: SI3O17LiveForcing`) come from the step's record via `external_forcing` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:475-486`), which strips all `CARRIED_KIND22_FIELDS` from kind `22` before constructing `SI3O17LiveForcing`. After every step, `free_run` compares every scored output field across kinds `23, 24, 25, 26, 27, 28, 30, 31, 34` with NEMO bit for bit (0 ULP, signed zeros and NaN bits included via `compare`), records `differing_values_per_field` per step and `first_difference` (`step, rank, step_key, field, different, nonfinite, max_relative`), and continues through all `N` steps without stopping at the first difference.

Read-only inspection of steps 1 and 2 of the real NEMO record `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step` confirmed that NEMO's step 1 exit (`kinds 28, 30, 23`) equals step 2 entry (`kinds 22, 29`) with 0 differing bits across all carried prognostic, Prather, and aEVP stress fields, and that step 2's recorded `cflice_delayed` (`0.08826992350223003`) matches step 1's `MAX(MAXVAL(|u_ice|*rDt_ice*r1_e1u), MAXVAL(|v_ice|*rDt_ice*r1_e2v))` (`src/ICE/icedyn_adv_pra.F90:119-127`) to 0 ULP.

#### Complete classification of every `si3_o17_step` input (`CARRIED`, `EXTERNAL`, `RESET AT STEP ENTRY`, `RECOMPUTED IN-STEP`)

Every input leaf of `si3_o17_step(entry: SI3O17StepState, forcing: SI3O17LiveForcing, card: SI3O17Card)` (`packages/ice/legoesm/ice/si3_o17.py:559-568` and `packages/ice/legoesm/ice/si3_o17.py:1138-1152`) has been re-checked against NEMO 5.0.2 source (Resume 2 Finding 3; `DECISION_NEEDED: none`).

| Input path in `si3_o17_step` | Class | Record kind (`kt=1` / `kt>=2`) | NEMO 5.0.2 source and `si3_o17_step` behavior |
|---|---|---|---|
| `entry.ice.v_i` `(jpi,jpj,jpl)` | `CARRIED` | Kind 22 at `kt=1`; `prev_out.ice.v_i` at `kt>=2` (scored vs kind 28) | Updated in SI3 by Prather advection (`src/ICE/icedyn_adv_pra.F90:219-247`), ridging/rafting (`src/ICE/icedyn_rdgrft.F90:247-263`), corrections (`src/ICE/icecor.F90:111-116`), thermodynamics (`src/ICE/icethd.F90:128-135`, `src/ICE/icethd.F90:183-201`), ITD remapping (`src/ICE/iceitd.F90:550-562`), and `ice_var_eqv2glo` (`src/ICE/icevar.F90:454-534`). |
| `entry.ice.v_s` `(jpi,jpj,jpl)` | `CARRIED` | Kind 22 at `kt=1`; `prev_out.ice.v_s` at `kt>=2` (scored vs kind 28) | Updated in SI3 by `src/ICE/icedyn_adv_pra.F90:219-247`, `src/ICE/icedyn_rdgrft.F90:247-263`, `src/ICE/icecor.F90:111-116`, `src/ICE/icethd.F90:128-135`, `src/ICE/iceitd.F90:550-562`, and `src/ICE/icevar.F90:454-534`. |
| `entry.ice.sv_i` `(jpi,jpj,jpl)` | `CARRIED` | Kind 22 at `kt=1`; `prev_out.ice.sv_i` at `kt>=2` (scored vs kind 28) | Updated in SI3 by `src/ICE/icedyn_adv_pra.F90:219-247`, `src/ICE/icedyn_rdgrft.F90:247-263`, `src/ICE/icecor.F90:111-116`, `src/ICE/icethd.F90:128-135`, `src/ICE/iceitd.F90:550-562`, and `src/ICE/icevar.F90:454-534`. |
| `entry.ice.a_i` `(jpi,jpj,jpl)` | `CARRIED` | Kind 22 at `kt=1`; `prev_out.ice.a_i` at `kt>=2` (scored vs kind 28) | Updated in SI3 by `src/ICE/icedyn_adv_pra.F90:219-247`, `src/ICE/icedyn_rdgrft.F90:247-263`, `src/ICE/icecor.F90:111-116`, `src/ICE/icethd.F90:128-135`, `src/ICE/icethd.F90:183-201`, and `src/ICE/iceitd.F90:550-562`. |
| `entry.ice.t_su` `(jpi,jpj,jpl)` | `CARRIED` | Kind 22 at `kt=1`; `prev_out.ice.t_su` at `kt>=2` (scored vs kind 28) | Updated in SI3 by vertical heat diffusion (`src/ICE/icethd.F90:438-526`), ITD remapping (`src/ICE/iceitd.F90:543-559`), and the `ice_thd` halo tail (`src/ICE/icethd.F90:188-201`). |
| `entry.ice.u_ice`, `entry.ice.v_ice` `(jpi,jpj)` | `CARRIED` | Kind 22 at `kt=1`; `prev_out.ice.(u_ice,v_ice)` at `kt>=2` (scored vs kind 28) | Updated in SI3 by aEVP rheology (`src/ICE/icedyn_rhg_evp.F90:530-560`) and masked/linked in the `ice_thd` tail (`src/ICE/icethd.F90:188-201`). |
| `entry.ice.oa_i` `(jpi,jpj,jpl)` | `CARRIED` | Kind 22 at `kt=1`; `prev_out.ice.oa_i` at `kt>=2` (scored vs kind 28) | Updated in SI3 by `src/ICE/icedyn_adv_pra.F90:219-247`, `src/ICE/icedyn_rdgrft.F90:247-263`, `src/ICE/icecor.F90:111-116`, `src/ICE/iceitd.F90:550-562`, and aged in `ice_thd` (`src/ICE/icethd.F90:185`). |
| `entry.ice.a_ip`, `entry.ice.v_ip`, `entry.ice.v_il` `(jpi,jpj,jpl)` | `CARRIED` | Kind 22 at `kt=1`; `prev_out.ice.(a_ip,v_ip,v_il)` at `kt>=2` (scored vs kind 28) | Updated in SI3 by `src/ICE/icedyn_adv_pra.F90:219-247`, `src/ICE/icedyn_rdgrft.F90:247-263`, `src/ICE/icecor.F90:111-116`, melt-pond thermodynamics (`src/ICE/icethd.F90:438-526`), `src/ICE/iceitd.F90:550-562`, and `src/ICE/icevar.F90:454-534`. |
| `entry.ice.e_s` `(jpi,jpj,nlay_s,jpl)`, `entry.ice.e_i` `(jpi,jpj,nlay_i,jpl)`, `entry.ice.szv_i` `(jpi,jpj,nlay_i,jpl)` | `CARRIED` | Kind 22 at `kt=1`; `prev_out.ice.(e_s,e_i,szv_i)` at `kt>=2` (scored vs kind 28) | Updated in SI3 by `src/ICE/icedyn_adv_pra.F90:219-247`, `src/ICE/icedyn_rdgrft.F90:247-263`, `src/ICE/icecor.F90:111-116`, `src/ICE/icethd.F90:438-526`, `src/ICE/iceitd.F90:550-562`, and `src/ICE/icevar.F90:454-534`. |
| `entry.ice.t_si` `(jpi,jpj,jpl)` | `RESET AT STEP ENTRY (NOT CARRIED)` | Kind 22 at `kt=1`; `prev_out.ice.t_si` passed at `kt>=2` and immediately reset on `owned` to `rt0` | Allocated on `A2D(0)` (`src/ICE/ice.F90:451`) and reset to `rt0` across all `A2D(0)` points at the start of every step in `diag_set0` (`src/ICE/icestp.F90:164` and `src/ICE/icestp.F90:485-493`), then written on active ice points in `ice_thd_zdf_BL99` (`src/ICE/icethd_zdf_bl99.F90:860-869`) and scattered in `src/ICE/icethd.F90:163,518`. In `si3_o17_step`, `state.t_si.at[owned].set(card.constants.T0)` (`packages/ice/legoesm/ice/si3_o17.py:587-589`) resets `t_si[owned]` to `rt0` at step entry before any kernel reads `t_si` (halo cells remain `0.0`, are never read, and are sliced out by `compare`), so `free_run` is unaffected. |
| `entry.ice.cnd_ice` `(jpi,jpj,jpl)` | `RESET AT STEP ENTRY / UNUSED IN SAS (NOT CARRIED)` | `0.0` at `kt=1`; `prev_out.ice.cnd_ice` (`0.0`) at `kt>=2` | Allocated on `A2D(0)` (`src/ICE/ice.F90:414`), reset to `0._wp` at the start of every step in `diag_set0` (`src/ICE/icestp.F90:164` and `src/ICE/icestp.F90:485-493`), written in `src/ICE/icethd_zdf_bl99.F90:842-850` and `src/ICE/icethd.F90:438-526`, and only read by `sbc_cpl_snd` (`src/OCE/SBC/sbccpl.F90:2485-2498`, `ln_cpl=T`). Never read or written in `si3_o17_step`, so `free_run` is unaffected. |
| `entry.ice.t1_ice` `(jpi,jpj,jpl)` | `COUPLED-ONLY DIAGNOSTIC / UNUSED IN SAS (NOT CARRIED)` | `0.0` at `kt=1`; `prev_out.ice.t1_ice` (`0.0`) at `kt>=2` | Allocated on `A2D(0)` (`src/ICE/ice.F90:198-199` and `src/ICE/ice.F90:414`), written on active ice points in `src/ICE/icethd_zdf_bl99.F90:842-850` and `src/ICE/icethd.F90:438-526`, and only read by `sbc_cpl_snd` (`src/OCE/SBC/sbccpl.F90:2302-2309`, `ln_cpl=T`). Never read or written in `si3_o17_step`, so `free_run` is unaffected. |
| `entry.ice.snwice_mass`, `entry.ice.snwice_mass_b`, `entry.ice.snwice_fmass` `(jpi,jpj)` | `CARRIED` | Kind 22 at `kt=1`; `prev_out.ice.(snwice_mass,snwice_mass_b,snwice_fmass)` at `kt>=2` (scored vs kinds 24, 27, 28) | `SAVE` module arrays in `src/OCE/SBC/sbc_ice.F90:125` and `src/OCE/SBC/sbc_ice.F90:217-219`, updated at step exit by `ice_update_flx` (`src/ICE/iceupdate.F90:189-194`, `packages/ice/legoesm/ice/si3_o17.py:1125-1130`); `snwice_mass` and `snwice_mass_b` are read in `ice_var_sshdyn` (`src/ICE/icedyn_rhg_evp.F90:268`) and `ice_update_flx`, while `snwice_fmass` persists in module memory across kinds `24` (`after_advection`) and `27` (`after_thermo`) until `ice_update_flx` updates it at kind `28`. |
| `entry.ice.prather.*` (55 moment arrays across 11 families: `ice, snw, a, s, age, ap, vp, vl, c0, e, si`) | `CARRIED` | Kind 29 at `kt=1`; `prev_out.ice.prather` at `kt>=2` (scored vs kind 30) | Updated in SI3 by Prather advection (`src/ICE/icedyn_adv_pra.F90:219-247` and `src/ICE/icedyn_adv_pra.F90:438-482`). |
| `entry.ice.bounds.(hi_max, hi_mean)` `(jpl+1,), (jpl,)` | `CARRIED` | Kind 22 at `kt=1`; `prev_out.ice.bounds` at `kt>=2` | Initialized in `ice_itd_init` (`src/ICE/iceitd.F90:715-769`) and carried unchanged on `SI3State.bounds`. |
| `entry.ice.equivalent.*` (`h_i, h_s, o_i, h_ip, h_il, a_ip_frac, a_ip_eff, s_i, sz_i, t_i, t_s, a_i, v_s`) | `RECOMPUTED IN-STEP (NOT CARRIED)` | Reconstructed via `ice_var_glo2eqv(2)` at `kt=1`; `prev_out.ice.equivalent` passed at `kt>=2` and overwritten before use | Recomputed in-step after dynamics by `CALL ice_var_glo2eqv(2)` (`src/ICE/icestp.F90:182`, `src/ICE/icevar.F90:255-428`). In `si3_o17_step`, dynamics (`si3_dyn_orca2`, `si3_ice_cor_dyn`) recomputes pre-advection thicknesses from extensive volumes/areas and `si3_var_glo2eqv(kn=2)` overwrites `state.equivalent` at `packages/ice/legoesm/ice/si3_o17.py:744-760` before surface fluxes and thermodynamics, so `free_run` is unaffected. |
| `entry.ice.aggregated.(tm_su, vt_i, vt_s, vt_ip, vt_il, ato_i)` | `CARRIED` | Reconstructed via `ice_var_agg(2)` at `kt=1`; `prev_out.ice.aggregated.(tm_su,vt_i,vt_s,vt_ip,vt_il,ato_i)` at `kt>=2` | Computed at the end of step `kt - 1` by `CALL ice_var_agg(2)` (`src/ICE/icestp.F90:211`, `src/ICE/icevar.F90:105-252`, `packages/ice/legoesm/ice/si3_o17.py:1117-1124`) and read at the start of step `kt` by `blk_ice_1` (`src/ICE/icestp.F90:159`, `src/ICE/icesbc.F90:85-87`), `ice_dyn_rhg_evp` (`src/ICE/icedyn_rhg_evp.F90:283-285`), and `ice_dyn_adv_pra` (`src/ICE/icedyn_adv_pra.F90:219-247`) before `CALL ice_var_agg(1)` runs at `src/ICE/icestp.F90:183`. |
| `entry.ice.aggregated.(at_i, at_ip, et_s, et_i, at_ip_eff, st_i, hm_i, hm_s, tm_si, om_i, sm_i, tm_i, tm_s, hm_ip, hm_il)` | `RECOMPUTED IN-STEP (NOT CARRIED)` | Reconstructed via `ice_var_agg(2)` at `kt=1`; `prev_out.ice.aggregated` passed at `kt>=2` and overwritten before use | Recomputed in-step by `CALL ice_var_agg(1)` (`src/ICE/icestp.F90:183`, `packages/ice/legoesm/ice/si3_o17.py:761-781`) and `CALL ice_var_agg(2)` (`src/ICE/icestp.F90:211`, `packages/ice/legoesm/ice/si3_o17.py:1117-1124`); none of these 15 fields on `entry.ice.aggregated` is read by any scored output before `si3_var_agg(kn=1)` overwrites `state.aggregated` at `packages/ice/legoesm/ice/si3_o17.py:761-781`, so `free_run` is unaffected. |
| `entry.ice.cflice_delayed` `()` | `CARRIED` | `None` at `kt=1`; `prev_out.ice.cflice_delayed` at `kt>=2` (scored vs kind 22 `cflice_delayed`) | Updated in SI3 by `CALL mpp_max('icedyn_adv_pra', zcfl, cdelay='cflice')` (`src/ICE/icedyn_adv_pra.F90:119-127`). |
| `entry.aevp.(stress1_t, stress2_t, stress12_f)` `(jpi,jpj)` | `CARRIED` | Kind 22 at `kt=1`; `prev_out.aevp.(stress1_t,stress2_t,stress12_f)` at `kt>=2` (scored vs kind 23) | Updated in SI3 by `ice_dyn_rhg_evp` (`src/ICE/icedyn_rhg_evp.F90:459-461` and `src/ICE/icedyn_rhg_evp.F90:838-843`). |
| `entry.aevp.(u_ice_u, v_ice_v)` `(jpi,jpj)` | `OVERWRITTEN AT STEP ENTRY (NOT CARRIED SEPARATELY)` | Kind 22 `(u_ice,v_ice)` at `kt=1`; `prev_out.aevp.(u_ice_u,v_ice_v)` passed at `kt>=2` and overwritten at step entry | Overwritten at step entry with `entry.ice.(u_ice, v_ice)` (`packages/ice/legoesm/ice/si3_o17.py:610-619`), matching `src/ICE/icestp.F90:151`, `src/ICE/icestp.F90:167`, and `src/ICE/icedyn.F90:128-135`, so `free_run` is unaffected. |
| `forcing.kt` `int` | `EXTERNAL` | Record header `meta["kt"]` at every step | Stepped by the NEMO driver and passed into `sbc` (`src/OCE/SBC/sbcmod.F90:397-403`) and `ice_stp` (`src/ICE/icestp.F90:96-238`). |
| `forcing.(sss_m, sst_m, ssh_m)` and `forcing.surface.(sst_m, sss_m, u_oce, v_oce, e3t_m, frq_m)` `(jpi,jpj)` | `EXTERNAL` | Kind 22 (`sss_m, sst_m, ssh_m, ssu_m, ssv_m, e3t_m, frq_m`) at every step | Prescribed in SAS by `sbc_ssm` (`src/SAS/sbcssm.F90:87-121` and `src/SAS/sbcssm.F90:122-126`); read by SI3 in `src/ICE/icestp.F90:130-138`, `src/ICE/icesbc.F90:358-406`, `src/ICE/icedyn_rhg_evp.F90:268`, and `src/ICE/icevar.F90:390-395`. |
| `forcing.rn_amax_2d` `(jpi,jpj)` | `EXTERNAL` | Kind 22 `rn_amax_2d` at every step | Set once from namelist `rn_amax_n, rn_amax_s` and latitude `gphit` in `ice_itd_init` (`src/ICE/iceitd.F90:757-759`). |
| `forcing.advection_grid.(e1e2t, r1_e1e2t, e2u, e1v)` `(jpi,jpj)` | `EXTERNAL` | Kind 22 at every step | Initialized from domain horizontal metrics (`domhgr.F90:222-227`) and read by `ice_dyn_adv_pra` (`src/ICE/icedyn_adv_pra.F90:119-136` and `src/ICE/icedyn_adv_pra.F90:438-482`). |
| `forcing.aevp_forcing.(u_oce_u, v_oce_v, ssh_m, ff_t, ff_f, fast_tmask, fast_umask, fast_vmask, iceberg_tmask, iceberg_umask, iceberg_vmask, utau_ice, vtau_ice, ssh_t)` | `EXTERNAL` | Kind 22 (`ssu_m, ssv_m, ssh_m, ff_t, ff_f`) and kind 23 (masks) at every step | Ocean velocity/SSH from `src/SAS/sbcssm.F90:87-121`; Coriolis from `tools/DOMAINcfg/src/domhgr.F90:383-384`; rheology masks read in `src/ICE/icedyn_rhg_evp.F90:530-560`; `utau_ice, vtau_ice` overwritten inside `si3_o17_step` (`packages/ice/legoesm/ice/si3_o17.py:610-619`) by live `ice_sbc_tau` (`src/ICE/icestp.F90:159`, `src/ICE/icesbc.F90:85-87`, `src/ICE/icesbc.F90:108`); `ssh_t=None` computed in-kernel (`src/ICE/icedyn_rhg_evp.F90:268`). |
| `forcing.aevp_metrics.*` (`e1t, e2t, e1u, e2u, e1v, e2v, e1f, e2f, r1_e1t, r1_e2t, r1_e1u, r1_e2v, r1_e1f, r1_e2f, e1e2t, r1_e1e2t, e2_e1u, e1_e2v, e1e1t, e2e2t, e1e1f, e2e2f, tmask, umask, vmask, fmask, bdy_umask, bdy_vmask`) | `EXTERNAL` | Kind 22 at every step | Domain metrics and land/sea masks from `domhgr.F90:222-227` and `src/OCE/DOM/dommsk.F90:203-205`; read in `src/ICE/icedyn_rhg_evp.F90:189-192`, `src/ICE/icedyn_rhg_evp.F90:459-461`, `src/ICE/icedyn_rhg_evp.F90:791-827`, and `src/ICE/icestp.F90:421-423`. |
| `forcing.surface.(wndi, wndj, theta_air_zt, q_air_zt, slp, rhoa, smask0, qlw, precip, snow, qsr, cloud_fra, qsr_oce, qns_oce, emp_oce, utau_oce_in, vtau_oce_in, taum_oce_in)` | `EXTERNAL` | Kinds 32, 33, and 22 at every step | Set in `sbc_blk` before `ice_stp` (`src/OCE/SBC/sbcmod.F90:474-477` and `src/OCE/SBC/sbcmod.F90:570`); read by `ice_sbc_tau` (`src/ICE/icesbc.F90:85-87`), `ice_sbc_flx` (`src/ICE/icesbc.F90:150-165` and `src/ICE/icesbc.F90:195`), `ice_flx_other` (`src/ICE/icesbc.F90:358-406`), and `ice_update_flx` (`src/ICE/iceupdate.F90:187-194`). Optional precomputed flux overrides are `None` in O17 live forcing. |
| `forcing.stress.(utau_oce, vtau_oce, taum_oce, u_oce, v_oce, umask, vmask, tmask, u_ice, v_ice)` | `EXTERNAL` | Kind 22 at every step (`u_ice, v_ice` zeroed in `external_forcing`) | Pre-ice ocean stress/velocity and masks read in `ice_update_tau` (`src/ICE/iceupdate.F90:438-484`); `u_ice, v_ice` on `forcing.stress` are overwritten inside `si3_o17_step` (`packages/ice/legoesm/ice/si3_o17.py:1131-1136`) with post-thermodynamic `final_state.(u_ice, v_ice)`. |
| `card.state: SI3ORCA2StateConfig` (32 fields: `jpl, nlay_i, nlay_s, nn_icesal, rn_icesal, rn_sal_gd, rn_sal_fl, rn_simax, rn_simin, rn_sinew, rn_himean, rn_himax, opt_hips, ln_pnd, ln_pnd_LEV, ln_pnd_TOPO, ln_pnd_lids, ln_pnd_alb, rn_apnd_min, rn_apnd_max, rn_pnd_hl_min, rn_pnd_hl_max, nn_snwfra, ln_icedyn, ln_icethd, ln_dynadv_pra, ln_mppdelay, nn_fsbc, rDt_ice, nn_hls, ln_cpl, ice_constants`) | `EXTERNAL` | `output.namelist.ice` and `output.namelist.dyn` (`card_for`, `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:444-447`) | Time-invariant state/domain/category/pond/salinity config initialized in `par_init` (`src/ICE/icestp.F90:328-385`) and `ice_init` (`src/ICE/icestp.F90:260-325`); constructed in `build_orca2_sas_ice_o17_card` (`packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.aevp: SI3CGridAEVPConfig` (22 fields: `scheme, staggering, dt_s, n_subcycles, eccentricity, creep_limit_s_inv, strength_parameter_pa, strength_decay, rho_snow, rho_ice, rho_water, rho_ocean, gravity, rn_ishlat, halo_width, category_count, landfast, landfast_depth_fraction, landfast_basal_friction_n_m3, landfast_relaxation_s_inv, landfast_tensile_fraction, convergence_check`) | `EXTERNAL` | `output.namelist.ice` and `output.namelist.dyn` | Time-invariant C-grid aEVP rheology and subcycling parameters (`nammpp`, `namdyn_neots`, `namdyn_rhg`, `packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.strength: SI3ORCA2StrengthConfig` (4 fields: `ln_str_h79, rn_pstar, rn_crhg, ln_str_smooth`) | `EXTERNAL` | `output.namelist.ice` (`namdyn_rhg`) | Time-invariant Hibler (1979) ice strength parameters (`packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.fold: SI3ORCA2FoldProgram` (2 fields: `nn_hls, nfd_full_when_absent`) | `EXTERNAL` | `output.namelist.dyn` (`nammpp`) | Time-invariant ORCA2 north-fold exchange configuration (`packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.redistribution: SI3RdgrftConfig` (25 fields: `rn_csrdg, ln_partf_exp, rn_astar, ln_distf_lin, ln_distf_exp, rn_murdg, ln_ridging, rn_hstar, rn_porordg, rn_fsnwrdg, rn_fpndrdg, ln_rafting, rn_hraft, rn_craft, rn_fsnwrft, rn_fpndrft, ln_rhg_evp_or_vp, ln_pnd, ln_pnd_lids, nn_icesal, rdt_ice, rhoi, rhos, rhow, rcp`) | `EXTERNAL` | `output.namelist.ice` (`namdyn_rdgrft`) | Time-invariant ridging and rafting parameters (`src/ICE/icedyn_rdgrft.F90:247-263`, `packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.correction: SI3IceCorConfig` (11 fields: `rn_himin, rn_simin, rn_sinew, nn_icesal, ln_pnd, ln_pnd_lids, rdt_ice, rhoi, rhos, rhow, rt0`) | `EXTERNAL` | `output.namelist.ice` (`namitd`, `namthd_sal`, `namthd_pnd`) | Time-invariant `ice_cor` parameters (`src/ICE/icecor.F90:111-116`, `packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.surface: SI3SurfaceConfig` (32 fields: `rn_Cd_ia, rn_Ce_ia, rn_Ch_ia, rn_pfac, rn_efac, rn_snwblow, nn_snwfra, rn_alb_sdry, rn_alb_smlt, rn_alb_idry, rn_alb_imlt, rn_alb_dpnd, rn_alb_lpnd, rn_alb_hpiv, ld_pnd_alb, nn_qtrice, ln_icedyn, rn_Cd_io, rDt_ice, nn_hls, ln_blk, ln_mixcpl, ln_Cx_ice_cst, ln_Cx_ice_frm, ln_Cx_ice_AN05, ln_Cx_ice_LU12, ln_Cx_ice_LG15, nn_flxdist, ln_cndflx, ln_leadhfx, ln_icedO, ln_icedH`) | `EXTERNAL` | `output.namelist.ice` and `output.namelist.dyn` (`namsbc`, `namsbc_blk`, `namalb`, `namthd`) | Time-invariant surface bulk/albedo/transmission/drag config (`src/ICE/icesbc.F90:85-87`, `src/ICE/icesbc.F90:150-165`, `packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.thermo_namelist: dict[str, Any]` (60 keys: `ln_drgice_imp, ln_virtual_itd, rn_himin, ln_icedH, ln_icedA, ln_icedO, ln_leadhfx, ln_zdf_BL99, ln_cndi_U64, ln_cndi_P07, rn_cnd_s, rn_kappa_i, rn_kappa_s, ln_zdf_chkcvg, rn_beta, rn_dmin, rn_hinew, ln_frazil, rn_maxfraz, rn_vfraz, rn_Cfraz, nn_icesal, ln_flushing, ln_drainage, rn_sinew, rn_simin, rn_icesal, rn_sal_himin, nn_liquidus, nn_drainage, nn_flushing, rn_flushrate, rn_vbrc, nn_sal_scheme, rn_alpha_RJW, rn_Rc_RJW, rn_alpha_GN, rn_Rc_GN, rn_alpha_CW, ln_sal_chk, ln_pnd, ln_pnd_LEV, rn_apnd_min, rn_apnd_max, rn_pnd_flush, ln_pnd_CST, rn_apnd, rn_hpnd, ln_pnd_TOPO, ln_pnd_lids, ln_pnd_rain, ln_pnd_alb, rn_pnd_hl_min, rn_pnd_hl_max, nn_snwfra, rn_snwblow, ln_cndflx, ln_cndemulate, nn_qtrice, ln_icediachk`) | `EXTERNAL` | `output.namelist.ice` (`namthd`, `namthd_zdf`, `namthd_da`, `namthd_do`, `namthd_sal`, `namthd_pnd`, `namitd`, `nampar`) | Time-invariant thermodynamic namelist dictionary passed to O9--O12 (`packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.constants: IceConstantsConfig` (15 fields: `T0, rho_ice, rho_snow, rho_ocean, c_ice, c_ocean, latent_fusion, latent_sublimation, k_ice, k_snow, liquidus_slope, S_ocean_ref, lead_albedo, epsi10, epsi20`) | `EXTERNAL` | Physical constants (`ICE_CONSTANTS_SI3`) | Time-invariant SI3 physical constants (`packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.rn_himin: float` (`0.1`) | `EXTERNAL` | `output.namelist.ice` (`namitd%rn_himin`) | Minimum ice thickness selector (`0.1 m`, `packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.rn_hinew: float` (`0.1`) | `EXTERNAL` | `output.namelist.ice` (`namthd_do%rn_hinew`) | New-ice collection thickness (`0.1 m`, `packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.ln_frazil: bool` (`False`) | `EXTERNAL` | `output.namelist.ice` (`namthd_do%ln_frazil`) | Open-water frazil collection selector (`False`, `packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.ln_drgice_imp: bool` (`False`) | `EXTERNAL` | `output.namelist.dyn` (`namdrg_top%ln_drgice_imp`) | Implicit top/ice drag selector (`False`, `packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.ln_ice_embd: bool` (`False`) | `EXTERNAL` | `output.namelist.dyn` (`namsbc%nn_ice_embd != 0`) | Embedded sea-ice draft selector (`False`, `packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.differentiable: bool` (`True`) | `EXTERNAL` | `build_orca2_sas_ice_o17_card` selector (`True`) | Differentiable execution selector (`True`, `packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |
| `card.core2_precipitation_closure: bool` (`False`) | `EXTERNAL` | `build_orca2_sas_ice_o17_card` selector (`False`) | CORE2 precipitation-closure selector (`False` for live O7 `ice_flx_other`, `packages/ice/legoesm/ice/si3_o17.py:79-97` and `packages/ice/legoesm/ice/si3_o17.py:164-291`). |

The 16-field `SI3O17Card` inventory above was regenerated directly from `SI3O17Card` (`packages/ice/legoesm/ice/si3_o17.py:79-97`) and `card_for` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:444-447`) by running:

```bash
timeout 1200 env CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  PYTHONPATH="$PWD:$PWD/packages/core:$PWD/packages/ocean:$PWD/packages/ice:$PWD/packages/atmosphere:$PWD/packages/land:$PWD/packages/coupler:$PWD/packages/driver" \
  /home/dbalwada/legoESM/.venv/bin/python -c '
from pathlib import Path
from legoesm.ice.si3_o17 import SI3O17Card
from scripts.validate.ocean_fidelity.testcases.nemo_si3_o17_composition_scorer import card_for
card = card_for(Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step/record_100step"))
assert isinstance(card, SI3O17Card)
sep = ", "
print(f"SI3O17Card._fields ({len(SI3O17Card._fields)}): {SI3O17Card._fields}")
for idx, name in enumerate(SI3O17Card._fields, 1):
    val = getattr(card, name)
    tname = type(val).__name__
    if hasattr(val, "_fields"):
        detail = f"{len(val._fields)} fields: ({sep.join(val._fields)})"
    elif isinstance(val, dict):
        detail = f"{len(val)} keys: ({sep.join(val.keys())})"
    else:
        detail = f"value={val!r}"
    print(f"{idx:2d}. card.{name}: {tname} | class=EXTERNAL | {detail}")
'
```

#### Synthetic-record proofs for (A) and (B) and operator script

- `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_synthetic_record.py:157-250` runs both proofs on the 2-step writer-shaped synthetic record (`7 x 7` full grid, `3 x 3` owned interior, `eager` binary64):
  - `status = "PASS_O17_SYNTHETIC_RECORD_SCORER"` in `720.01 s`, peak RSS `4,056,148 KiB`.
  - `(A) parallel_ranges`: `split_merge_equals_serial = true`, `split_free_run_refused = true`, `missing_step_refused = true`, `duplicated_step_refused = true`, `merged_status = "PASS"`.
  - `(B) free_run`: `identical_status = "PASS"`, `identical_different = 0`, `identical_first_difference = null`. When a 1-ULP change is applied to our step 1 output `out.ice.v_i[2, 2, 0]` (`1.1794439265913363e-16` relative) while a plant takes `v_i` from step 2's record instead of from our previous step, `free_run` detects the broken carry chain at step 2 (`carried_plant_status = "FAIL"`, `carried_plant_detected = true`, `carried_plant_first_difference = {"step": 2, "rank": 0, "step_key": "2:0", "field": "carried_entry.ice.v_i", "different": 1, "nonfinite": 0, "max_relative": 1.1794439265913363e-16}`).
- Direct unit tests `test_o17_scorer_parallel_step_ranges_and_merge`, `test_o17_scorer_free_run_and_carried_state_plants`, and `test_o17_operator_script_refuses_scratch_evidence_tree_before_mkdir` in `tests/ice/unit/test_si3_o17.py:577-1075` passed, proving byte-for-byte identity against the parent `94a4069d61` scorer on the default serial path, refusal of split free-run step ranges, identical step-2 relative error (`1.0`) on the early-divergence plant across `parent_scorer.score`, serial `scorer.score`, and `scorer.free_run`, zero bit difference when all step-entry-reset / in-step-recomputed / SAS-unused fields (`t_si[owned]`, `cnd_ice`, `t1_ice`, `equivalent.*`, the 15 non-carried `aggregated.*` fields, and `aevp.(u_ice_u, v_ice_v)`) are perturbed to `-999.0` on step 1's output, detection of the 1-ULP `v_i` carry plant, continuation past step-1 differences, and pre-creation refusal of `--out` inside a scratch evidence tree without ever targeting the real read-only evidence directory.
- The committed operator script `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_operator_run.sh:1-180` hardcodes `readonly DEFAULT_EVIDENCE="/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step"` and allows `NEMO_SI3_O17_EXTRA_PROTECTED_EVIDENCE` only to append an additional protected root (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_operator_run.sh:13-14`), resolves `ABS_OUT`, `ABS_EVIDENCE`, and `ABS_DEFAULT_EVIDENCE` via `realpath -m` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_operator_run.sh:84-106`), and refuses any output path equal to or under the canonical real evidence tree, `--evidence`, or `NEMO_SI3_O17_EXTRA_PROTECTED_EVIDENCE` BEFORE `mkdir -p "$ABS_OUT"` creates anything:
  - **(a) Canonical real evidence tree always refused even when `NEMO_SI3_O17_EXTRA_PROTECTED_EVIDENCE` is set to a scratch root:** With `NEMO_SI3_O17_EXTRA_PROTECTED_EVIDENCE` set to `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_o17_free_scratch/operator_refusal_guard/si3_rounds/round29/oracle_o17_full_step`, invoking `nemo_si3_o17_operator_run.sh` with `--out /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step/refused_probe_do_not_create` (both without `--evidence` and with `--evidence <scratch_root>`) exits with code `2` and prints `REFUSE: --out resolves inside evidence tree (/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step/refused_probe_do_not_create inside read-only /data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step)` before `mkdir -p`; `refused_probe_do_not_create` does not exist before (`REAL_NONEXISTENT_BEFORE=absent`) or after (`REAL_NONEXISTENT_AFTER=absent`).
  - **(b) Scratch root refusal test:** `test_o17_operator_script_refuses_scratch_evidence_tree_before_mkdir` (`tests/ice/unit/test_si3_o17.py:967-1075`) builds `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_o17_free_scratch/operator_refusal_guard/si3_rounds/round29/oracle_o17_full_step/record_100step/o17_master_records` and an alternate scratch root `alt_oracle_o17_full_step`, sets `NEMO_SI3_O17_EXTRA_PROTECTED_EVIDENCE` to the first scratch root, and verifies that `--out` inside either scratch root is refused (`returncode == 2`, `"REFUSE: --out resolves inside evidence tree"`, `not target.exists()`).
  - **(c) Planted moved-refusal control in scratch (quoted and restored):** Planting a copy of `nemo_si3_o17_operator_run.sh` in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_o17_free_scratch/planted_operator_run.sh` with the `for protected_root in "${protected_roots[@]}"` refusal loop moved immediately after `mkdir -p "$ABS_OUT"` and running `NEMO_SI3_O17_OPERATOR_SCRIPT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_o17_free_scratch/planted_operator_run.sh pytest tests/ice/unit/test_si3_o17.py -k test_o17_operator_script_refuses_scratch_evidence_tree_before_mkdir -v` fails as required (`FAILED tests/ice/unit/test_si3_o17.py::test_o17_operator_script_refuses_scratch_evidence_tree_before_mkdir - AssertionError: assert not True` on `assert not target.exists()`, `1 failed, 19 deselected in 0.47s`), after which the scratch plant was removed and the unplanted test passed (`1 passed, 19 deselected in 0.47s`).
  - **Real read-only evidence tree untouched before and after:** Listing `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step` before and after running (a), (b), and (c) shows the exact same four entries (`input`, `manifest`, `record_100step`, `reference_100step`) and `2026-11-02` modification times:
    ```text
    drwxr-xr-x. 6 dbalwada dbalwada  102 2026-11-02 11:41:30.264872348 -0500 .
    drwxr-xr-x. 3 dbalwada dbalwada   42 2026-11-02 11:38:40.007474617 -0500 ..
    drwxr-xr-x. 2 dbalwada dbalwada 4096 2026-11-02 11:38:54.052013596 -0500 input
    drwxr-xr-x. 7 dbalwada dbalwada 4096 2026-11-02 12:05:56.739443679 -0500 manifest
    drwxr-xr-x. 4 dbalwada dbalwada 4096 2026-11-02 11:46:53.676722160 -0500 record_100step
    drwxr-xr-x. 2 dbalwada dbalwada 4096 2026-11-02 11:41:30.262872411 -0500 reference_100step
    ```
  - After the path guard, `nemo_si3_o17_operator_run.sh` checks `free -g` (`>= 250 GiB` available and no process above `100 GiB` RSS), runs (A) `K` parallel per-step range processes (`--labels GIVEN-NEMO-ENTRY --steps START..END`) followed by `--merge` on `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step`, and runs (B) `--free-run --n-steps 100`, writing all outputs under the operator-supplied output directory.

independent review: operator

## Committed NaN-fix proof scripts (2026-11-06)

The two proof scripts cited above were committed byte-identical to their
scratch sources; sha256 of source and committed copy, measured with
`sha256sum`:

- `scripts/validate/ocean_fidelity/testcases/si3_nanfix/verify_proof1_active_bits.py`:
  `cdab33c8f83557c1df686d7e7db9ca51738f5eaf01107a4747cb2da4cae368b6`
- `scripts/validate/ocean_fidelity/testcases/si3_nanfix/verify_proof2_orca2_step1.py`:
  `4852e5f7177cb1a2de4f46552f86bacf407f6431af7870f3a0da1024da03d857`
