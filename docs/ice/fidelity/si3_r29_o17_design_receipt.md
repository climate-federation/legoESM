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
   `packages/ice/legoesm/ice/si3_thd_pnd.py:865-874,955-1003`). Repeating the
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

## Round `si3_o17_free` (DECISION 210, 2026-11-03): parallel step ranges and carried-state free run

Date: 2026-11-04 (Resume 3). Base: `94a4069d61ee` (`si3_r29`). No NEMO build or integration was run, and `packages/` is untouched (`git diff --stat 94a4069d61ee..HEAD -- packages` is empty).

### (A) Parallel per-step ranges (`--steps START..END`) and exact report merge (`--merge`)

The composition scorer (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:450-877` and `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:886-949`) accepts `--steps START..END` (`parse_step_range`, 1-based inclusive) and `--merge <range1.json> ...` (`merge_reports`).

- **Retraction of split free-run step ranges (Resume 2 Finding 1):** The Round-1 `exit_state` helper has been removed. In free-run mode (`free_run`, or `score` when `"INDEPENDENT"` is in `labels`), restarting a range `START > 1` from NEMO's recorded state would reset an early trajectory divergence back to NEMO rather than carrying our own trajectory. The free run is inherently serial: `free_run` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:758-877`) refuses any `step_range` other than one contiguous range `1..target_steps` starting at step `1`, `score` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:675-755`) refuses any `step_range` with `start != 1` when `"INDEPENDENT"` is in `labels`, and `merge_reports` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:552-672`) refuses `FREE_RUN` reports and multi-range reports containing `INDEPENDENT`. Step ranges (`--steps START..END`) and `--merge` are restricted to per-step scoring (`PER_STEP_LABELS = ("GIVEN-NEMO-ENTRY",)`, `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:450-456` and `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:886-949`).
- `merge_reports` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:552-672`) verifies that every step in `1..expect_steps` is present once across all ranks, recomputes per-field, per-step, per-label, and top-level `PASS`/`FAIL` verdicts with the exact serial rules, and refuses any missing step (`O17 merge requires steps 1..N; missing=[...]`) or duplicated step (`duplicate O17 step(s) in merge: [...]`).
- Verified on the model-built 2-step synthetic record (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_synthetic_record.py:157-202` and `tests/ice/unit/test_si3_o17.py:545-729`):
  - **Parent `94a4069d61` serial byte identity (Resume 2 Finding 2):** Loading the parent scorer from `94a4069d61:scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py` (`_load_parent_scorer_94a4069d61`, `tests/ice/unit/test_si3_o17.py:545-613`) and running both `parent_scorer.score` and the new scorer's default serial `scorer.score` on the same 2-step synthetic record produces byte-for-byte identical JSON reports (`138,424` bytes, `sha256 = 0998de68d2b0869cbb72915767d9e7b73cbc251f62050fda7bfa1ed972fd59b3`) with the three named timing fields (`scorer_wall_seconds`, `scorer_rss_kib_after_each_step`, `scorer_peak_rss_kib`) excepted (`strip_timing`), and byte-for-byte identical full JSON reports when `time.monotonic` and `_rss_kib` are monkeypatched identically on both modules.
  - Scoring per-step ranges (`PER_STEP_LABELS = ("GIVEN-NEMO-ENTRY",)`) on `--steps 1..1` and `--steps 2..2` separately and merging via `merge_reports` equals the serial per-step report (`strip_timing(merged) == strip_timing(serial_per_step)`, `split_merge_equals_serial = true`) for both the passing identical record and a planted 1-ULP `FAIL` record.
  - Split free-run ranges (`free_run(..., step_range=(2, 2))`, `free_run(..., step_range=(1, 1))` on a 2-step run, `score(..., VALID_LABELS, step_range=(2, 2))`, and `merge_reports` on multi-range `INDEPENDENT` reports) are refused with `ValueError` (`split_free_run_refused = true`). On the early-divergence plant (`state + 2.0` vs `expected = float(kt)`), `parent_scorer.score` (`94a4069d61`), serial `scorer.score` (`INDEPENDENT`), and serial `scorer.free_run` all produce the exact same step-2 `max_relative = 1.0` (`tests/ice/unit/test_si3_o17.py:627-697`).
  - Merging with step 1 or step 2 omitted raises `ValueError` (`missing_step_refused = true`), and merging with step 1 or step 2 duplicated raises `ValueError` (`duplicated_step_refused = true`).

### (B) Carried-state free run (`--free-run`) and `CARRIED` vs `EXTERNAL` input classification

`free_run` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:758-877`) initializes step `kt = 1` from NEMO's recorded state before step 1 (`entry_state`, kinds `22` and `29`). For every subsequent step `kt = 2..N`, the step-entry state `entry: SI3O17StepState` comes strictly from our previous step's output `out` via `carry_step_state` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:489-496`); only external inputs (`forcing: SI3O17LiveForcing`) come from the step's record via `external_forcing` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_composition_scorer.py:475-486`), which strips all `CARRIED_KIND22_FIELDS` from kind `22` before constructing `SI3O17LiveForcing`. After every step, `free_run` compares every scored output field across kinds `23, 24, 25, 26, 27, 28, 30, 31, 34` with NEMO bit for bit (0 ULP, signed zeros and NaN bits included via `compare`), records `differing_values_per_field` per step and `first_difference` (`step, rank, step_key, field, different, nonfinite, max_relative`), and continues through all `N` steps without stopping at the first difference.

Read-only inspection of steps 1 and 2 of the real NEMO record `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step` confirmed that NEMO's step 1 exit (`kinds 28, 30, 23`) equals step 2 entry (`kinds 22, 29`) with 0 differing bits across all carried prognostic, Prather, and aEVP stress fields, and that step 2's recorded `cflice_delayed` (`0.08826992350223003`) matches step 1's `MAX(MAXVAL(|u_ice|*rDt_ice*r1_e1u), MAXVAL(|v_ice|*rDt_ice*r1_e2v))` (`src/ICE/icedyn_adv_pra.F90:119-127`) to 0 ULP.

#### Complete classification of every `si3_o17_step` input (`CARRIED`, `EXTERNAL`, `RESET AT STEP ENTRY`, `RECOMPUTED IN-STEP`)

Every input leaf of `si3_o17_step(entry: SI3O17StepState, forcing: SI3O17LiveForcing, card: SI3O17Card)` (`packages/ice/legoesm/ice/si3_o17.py:559-568` and `packages/ice/legoesm/ice/si3_o17.py:1131-1145`) has been re-checked against NEMO 5.0.2 source (Resume 2 Finding 3; `DECISION_NEEDED: none`).

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
| `entry.ice.snwice_mass`, `entry.ice.snwice_mass_b`, `entry.ice.snwice_fmass` `(jpi,jpj)` | `CARRIED` | Kind 22 at `kt=1`; `prev_out.ice.(snwice_mass,snwice_mass_b,snwice_fmass)` at `kt>=2` (scored vs kinds 24, 27, 28) | `SAVE` module arrays in `src/OCE/SBC/sbc_ice.F90:125` and `src/OCE/SBC/sbc_ice.F90:217-219`, updated at step exit by `ice_update_flx` (`src/ICE/iceupdate.F90:189-194`, `packages/ice/legoesm/ice/si3_o17.py:1118-1123`); `snwice_mass` and `snwice_mass_b` are read in `ice_var_sshdyn` (`src/ICE/icedyn_rhg_evp.F90:268`) and `ice_update_flx`, while `snwice_fmass` persists in module memory across kinds `24` (`after_advection`) and `27` (`after_thermo`) until `ice_update_flx` updates it at kind `28`. |
| `entry.ice.prather.*` (55 moment arrays across 11 families: `ice, snw, a, s, age, ap, vp, vl, c0, e, si`) | `CARRIED` | Kind 29 at `kt=1`; `prev_out.ice.prather` at `kt>=2` (scored vs kind 30) | Updated in SI3 by Prather advection (`src/ICE/icedyn_adv_pra.F90:219-247` and `src/ICE/icedyn_adv_pra.F90:438-482`). |
| `entry.ice.bounds.(hi_max, hi_mean)` `(jpl+1,), (jpl,)` | `CARRIED` | Kind 22 at `kt=1`; `prev_out.ice.bounds` at `kt>=2` | Initialized in `ice_itd_init` (`src/ICE/iceitd.F90:715-769`) and carried unchanged on `SI3State.bounds`. |
| `entry.ice.equivalent.*` (`h_i, h_s, o_i, h_ip, h_il, a_ip_frac, a_ip_eff, s_i, sz_i, t_i, t_s, a_i, v_s`) | `RECOMPUTED IN-STEP (NOT CARRIED)` | Reconstructed via `ice_var_glo2eqv(2)` at `kt=1`; `prev_out.ice.equivalent` passed at `kt>=2` and overwritten before use | Recomputed in-step after dynamics by `CALL ice_var_glo2eqv(2)` (`src/ICE/icestp.F90:182`, `src/ICE/icevar.F90:255-428`). In `si3_o17_step`, dynamics (`si3_dyn_orca2`, `si3_ice_cor_dyn`) recomputes pre-advection thicknesses from extensive volumes/areas and `si3_var_glo2eqv(kn=2)` overwrites `state.equivalent` at `packages/ice/legoesm/ice/si3_o17.py:744-760` before surface fluxes and thermodynamics, so `free_run` is unaffected. |
| `entry.ice.aggregated.(tm_su, vt_i, vt_s, vt_ip, vt_il, ato_i)` | `CARRIED` | Reconstructed via `ice_var_agg(2)` at `kt=1`; `prev_out.ice.aggregated.(tm_su,vt_i,vt_s,vt_ip,vt_il,ato_i)` at `kt>=2` | Computed at the end of step `kt - 1` by `CALL ice_var_agg(2)` (`src/ICE/icestp.F90:211`, `src/ICE/icevar.F90:105-252`, `packages/ice/legoesm/ice/si3_o17.py:1110-1117`) and read at the start of step `kt` by `blk_ice_1` (`src/ICE/icestp.F90:159`, `src/ICE/icesbc.F90:85-87`), `ice_dyn_rhg_evp` (`src/ICE/icedyn_rhg_evp.F90:283-285`), and `ice_dyn_adv_pra` (`src/ICE/icedyn_adv_pra.F90:219-247`) before `CALL ice_var_agg(1)` runs at `src/ICE/icestp.F90:183`. |
| `entry.ice.aggregated.(at_i, at_ip, et_s, et_i, at_ip_eff, st_i, hm_i, hm_s, tm_si, om_i, sm_i, tm_i, tm_s, hm_ip, hm_il)` | `RECOMPUTED IN-STEP (NOT CARRIED)` | Reconstructed via `ice_var_agg(2)` at `kt=1`; `prev_out.ice.aggregated` passed at `kt>=2` and overwritten before use | Recomputed in-step by `CALL ice_var_agg(1)` (`src/ICE/icestp.F90:183`, `packages/ice/legoesm/ice/si3_o17.py:761-781`) and `CALL ice_var_agg(2)` (`src/ICE/icestp.F90:211`, `packages/ice/legoesm/ice/si3_o17.py:1110-1117`); none of these 15 fields on `entry.ice.aggregated` is read by any scored output before `si3_var_agg(kn=1)` overwrites `state.aggregated` at `packages/ice/legoesm/ice/si3_o17.py:761-781`, so `free_run` is unaffected. |
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
| `forcing.stress.(utau_oce, vtau_oce, taum_oce, u_oce, v_oce, umask, vmask, tmask, u_ice, v_ice)` | `EXTERNAL` | Kind 22 at every step (`u_ice, v_ice` zeroed in `external_forcing`) | Pre-ice ocean stress/velocity and masks read in `ice_update_tau` (`src/ICE/iceupdate.F90:438-484`); `u_ice, v_ice` on `forcing.stress` are overwritten inside `si3_o17_step` (`packages/ice/legoesm/ice/si3_o17.py:1124-1129`) with post-thermodynamic `final_state.(u_ice, v_ice)`. |
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
- Direct unit tests `test_o17_scorer_parallel_step_ranges_and_merge`, `test_o17_scorer_free_run_and_carried_state_plants`, and `test_o17_operator_script_refuses_scratch_evidence_tree_before_mkdir` in `tests/ice/unit/test_si3_o17.py:535-1010` passed, proving byte-for-byte identity against the parent `94a4069d61` scorer on the default serial path, refusal of split free-run step ranges, identical step-2 relative error (`1.0`) on the early-divergence plant across `parent_scorer.score`, serial `scorer.score`, and `scorer.free_run`, zero bit difference when all step-entry-reset / in-step-recomputed / SAS-unused fields (`t_si[owned]`, `cnd_ice`, `t1_ice`, `equivalent.*`, the 15 non-carried `aggregated.*` fields, and `aevp.(u_ice_u, v_ice_v)`) are perturbed to `-999.0` on step 1's output, detection of the 1-ULP `v_i` carry plant, continuation past step-1 differences, and pre-creation refusal of `--out` inside a scratch evidence tree without ever targeting the real read-only evidence directory.
- The committed operator script `scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_operator_run.sh:1-173` sets `readonly DEFAULT_EVIDENCE="${NEMO_SI3_O17_DEFAULT_EVIDENCE:-/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step}"` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_operator_run.sh:13`), resolves `ABS_OUT`, `ABS_EVIDENCE`, and `ABS_DEFAULT_EVIDENCE` via `realpath -m` (`scripts/validate/ocean_fidelity/testcases/nemo_si3_o17_operator_run.sh:83-99`), and refuses any output path equal to or under the evidence directory BEFORE `mkdir -p "$ABS_OUT"` creates anything:
  - **Scratch-tree refusal test (Resume 3 Finding 1):** `test_o17_operator_script_refuses_scratch_evidence_tree_before_mkdir` (`tests/ice/unit/test_si3_o17.py:925-1010`) builds `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_o17_free_scratch/operator_refusal_guard/si3_rounds/round29/oracle_o17_full_step/record_100step/o17_master_records` and points both `NEMO_SI3_O17_DEFAULT_EVIDENCE` and `--evidence` at that scratch tree while passing `--out <scratch_evidence>/refused_probe_do_not_create`, asserting `returncode == 2`, `"REFUSE: --out resolves inside evidence tree"`, and `not refused_target.exists()`.
  - **Planted moved-refusal control in scratch (quoted and restored):** Planting a copy of `nemo_si3_o17_operator_run.sh` in `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_o17_free_scratch/planted_operator_run.sh` with the refusal check moved immediately after `mkdir -p "$ABS_OUT"` and running `NEMO_SI3_O17_OPERATOR_SCRIPT=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/si3_o17_free_scratch/planted_operator_run.sh pytest tests/ice/unit/test_si3_o17.py -k test_o17_operator_script_refuses_scratch_evidence_tree_before_mkdir -v` fails as required (`FAILED tests/ice/unit/test_si3_o17.py::test_o17_operator_script_refuses_scratch_evidence_tree_before_mkdir - AssertionError: assert not True` on `assert not refused_target.exists()`, `1 failed, 19 deselected in 0.47s`), after which the scratch plant was removed and the unplanted test passed (`1 passed, 19 deselected in 0.48s`).
  - **Real read-only evidence tree untouched before and after:** Listing `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round29/oracle_o17_full_step` before and after running the unplanted and planted tests shows the exact same four entries (`input`, `manifest`, `record_100step`, `reference_100step`) and `2026-11-02` modification times:
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
