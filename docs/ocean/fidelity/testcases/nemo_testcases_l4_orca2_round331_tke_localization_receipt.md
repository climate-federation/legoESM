# ORCA2 round 331 — first-step shear diagnosis and stored-velocity TKE boundary

Date: 2026-10-24

Base: `484b60f5b7da`; preregistration: `a69fa0005d23`; acquisition tool:
`ba88fcbdb48e`.

Initial disposition: **STOP_PROPOSAL; ACQUISITION_NEEDED**.  The resume-3
disposition at the end supersedes this after admitting the retained record;
no configuration value changes.

## Answer

**CONFIRMED.**  The landed round-330 card reproduces the admitted residual on
the CPU fp64 production-JIT path: returned TKE energy has 901 unequal wet
interfaces, post-TKE viscosity has 931, and the viscosity consumed after DDM
and IWM has 925.  The repaired deepest level remains exact in every score.
The remaining returned-energy family is 279 surface plus 622 interior plus
zero deepest interfaces.

**CONFIRMED.**  No literal surface-boundary transcription difference is
visible.  The record build sets the held surface value and its identity row at
`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:273-276`; the card
selects `nemo_z0` plus its literal matrix and solver
(`nemo_testcase_recipe.py:416,421-422`) and constructs the held virtual row in
`tke.py:1845-1857`.  That source read does not exclude a differing input or a
later source-order statement from the surface family.

**CONFIRMED.**  The old records cannot distinguish the live shear output from
the boundary, RHS/matrix, solve, or post-solve penetration statements.  No
single cited statement owns all or part of the residual, so Decision 122 is
not opened and no physics change is eligible this round.

## Production-JIT census

The committed round-328 replay was run unchanged against the landed round-330
card on CPU with fp64 and production JIT.  Its exact per-level census is:

| zero-based `k` | returned `en` | post-TKE `avm` | consumed `avm` |
|---:|---:|---:|---:|
| 0 | 279 | 284 | 282 |
| 1 | 111 | 116 | 116 |
| 2 | 67 | 67 | 67 |
| 3 | 55 | 58 | 58 |
| 4 | 53 | 56 | 56 |
| 5 | 43 | 46 | 46 |
| 6 | 23 | 28 | 28 |
| 7 | 15 | 18 | 18 |
| 8 | 15 | 18 | 18 |
| 9 | 11 | 13 | 13 |
| 10 | 13 | 12 | 10 |
| 11 | 13 | 13 | 13 |
| 12 | 17 | 17 | 16 |
| 13 | 12 | 12 | 12 |
| 14 | 17 | 16 | 16 |
| 15 | 15 | 15 | 15 |
| 16 | 15 | 15 | 15 |
| 17 | 3 | 3 | 3 |
| 18 | 5 | 5 | 5 |
| 19 | 48 | 48 | 48 |
| 20 | 11 | 11 | 11 |
| 21 | 5 | 5 | 5 |
| 22 | 26 | 26 | 25 |
| 23 | 4 | 4 | 4 |
| 24 | 4 | 4 | 4 |
| 25 | 6 | 6 | 6 |
| 26 | 9 | 9 | 9 |
| 27 | 6 | 6 | 6 |
| 28 | 0 | 0 | 0 |
| **total** | **901** | **931** | **925** |

The family totals are respectively surface/interior/deepest = `279/622/0`,
`284/647/0`, and `282/643/0`.  Returned `en` first differs at global
`[j,i,k]=[1,119,19]`; its maximum absolute difference is
`5.153991021727283e-9` at `[117,55,3]`.  The entry `rn2` operand remains exact
at all 414,119 wet interfaces.  This confirms R331-P1 and exactly reconciles
the round-330 totals; nothing is retracted.

Evidence:
`orca2_rounds/round331/replay/production_jit.json`, SHA-256
`fddba8b2921325e2f2237dac83b4f2896841173dfc2b4448461e820147c0333c`.

## Source-order discriminator

**CONFIRMED.**  `zdf_phy` computes shear, calls bottom then top drag, computes
mixed-layer outputs, and only then calls TKE
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:316-336`).

- `zdf_sh2` is live.  The executing no-Stokes arm forms U and V shear and
  horizontally averages them into `p_sh2`
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfsh2.f90:98-112`).  TKE
  consumes that value in both the Richardson-number calculation and the RHS;
  the RHS/matrix record boundary is therefore needed
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:411-428`).
- The deck uses linear bottom drag.  The bottom call therefore leaves the
  initialized coefficient unchanged, while the top call writes implicit ice
  drag because there is no cavity
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfdrg.f90:165-187`).  The
  coefficients are still recorded to verify their handoff and to localize any
  shallow bottom-boundary cells; source reading alone is not promoted to a
  numerical exclusion.
- `zdf_mxl` does compute `nmln` and `hmlp`
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfmxl.f90:109-123`), but
  the resolved `nn_etau=1` arm applies penetration over every interior level.
  The only adjacent arm that reads `nmln` is inactive `nn_etau=2`
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:500-507`).  Thus
  the mixed-layer outputs are **CONFIRMED inactive** for this TKE update, while
  their record remains a control on the source-order acquisition.
- The compiled matrix/RHS and Thomas solve remain the candidate boundaries
  after the input handoff
  (`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:411-428` and
  `:459-477`).  Existing post-TKE records skip all of those boundaries.

R331-P2 and P3 are confirmed.  R331-P4 is also confirmed: the old records do
not select the first unequal source-order boundary.

## Operator acquisition

The round-328 acquisition harness now has a `tke331` mode selected by the thin
launcher at
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round331_tke_acquisition/run.sh`.
It imports the reviewed round-328 helpers and preserves the record build's
existing round-326 `zdftke` observer.  Additions-only observers capture both
ranks at kt 1:

1. `zdf_sh2`, bottom/top `zdf_drg`, and `zdf_mxl` outputs, followed by a
   second copy at TKE entry to prove exact handoff, plus entry `avm_k` and
   `dissl` operands;
2. `en` after the ordinary top/bottom boundary conditions;
3. RHS `en` and all three tridiagonal coefficients after matrix construction
   and any wave-boundary override; the active Langmuir term lies between the
   boundary and matrix records and therefore falls inside this after-matrix
   segment;
4. `en` immediately after the Thomas solve and floor; and
5. returned `en`, `avm_k`, `avt_k`, and `wmask` for exact reconciliation with
   the admitted round-328 post-TKE record.

Admission requires the inherited `dyn_zdf` records to remain bit-identical,
the new drag fields to equal those admitted inputs, exact output-to-input
handoffs, exact surface continuity across each internal boundary, and exact
wet-cell returned fields against the admitted post-TKE record.  Eleven planted
violations cover header, order, truncation, stamp, shear, drag, mixing length,
boundary, matrix, solve, and final output; the semantic plants each move one
binary64 word by one ULP.

The clean read-only preflight passed at `ba88fcbdb48e`: 45 tests passed, all 11
preflight plants fired, all 140 inherited deterministic files matched, all six
writer/patched Fortran units passed syntax, and the same-program census found
410 identical preprocessed units and 422 identical normalized objects.  Its
non-observer source plant refused.  The first preflight attempt correctly
refused because it would have replaced the record build's already instrumented
`MY_SRC/zdftke.F90` with pristine source; the tool now extends that exact
record unit and the same-program proof passes.

Operator command:

```text
/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/orca2_r331-322489136/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round331_tke_acquisition/run.sh --run
```

Expected evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round331/acquisition/orca1ice_tke331_kt1_np2`.
No NEMO executable was run by Codex.

## Gates and disposition

- Preregistered predictions R331-P1 through P4: all **CONFIRMED**.
- Focused compiled-layout, admission, control, and same-program battery:
  20 passed; the imported clean preflight battery is 45 passed.
- Per-receipt citation gate: PASS, 10 citations, 0 unmapped, 0 failures, and
  0 map-audit failures; the required citation-gate pytest is 17 passed.
- Default ORCA2 landing battery: 148 passed in 943.01 s, including the
  six-test private-import ratchet.
- No configuration value changed: **ASKED**.
- No model change, seven-pair Decision-122 score, cross-card measurement, or
  gradient claim is made because ownership is still unmeasured: **ASKED**.
- Stop for the source-order record: **ASKED conditional** and now satisfied.

independent review: operator (Claude)

The next admissible action is the operator-only `--run`.  Its first unequal
boundary will either identify the live input family (`sh2`/drag), the surface
and bottom BC boundary, the matrix/RHS, the solve, or the post-solve
penetration.  Until that record exists, every such owner is **PLAUSIBLE**, not
confirmed.

## Resume-3 localization preregistration

Frozen at `75687076d7e8`, after admission/localization and before changing the
model statement.

**CONFIRMED input.**  Every wet shear value is zero at kt 1, but the writer
proves both the producer and consumer executed.  The live production trace is
bit-identical through the boundary, all three matrix arrays, and the complete
RHS on wet cells.  Substituting only NEMO's recorded RHS into the unchanged
literal solver changes the solve census from 926 unequal wet values to zero;
substituting all three recorded matrix arrays changes none of the 926.

**Single-statement hypothesis.**  NEMO's bottom TKE boundary reads the stored
`Kbb` velocity (`zdftke.f90:291-293`).  At this from-rest step `Kbb=Kmm=1` and
that field is zero.  legoESM already receives the corresponding entry-level
`u_now/v_now`, but the bottom-boundary call reads the post-explicit state.  The
candidate will change only that call's velocity operand to the supplied entry
pair; no configuration value will change.

**Predictions.**  The candidate should make all 430,552 wet post-solve values,
all 414,119 returned TKE values, and the returned `avm/avt` values bit-exact on
the admitted record.  It is eligible only if Decision 122 passes for all seven
required card/label pairs with zero exact-row loss and no first-over-bar move
away, and every other card that executes the changed line is measured.  V1 is
predicted unchanged because its deck disables this bottom boundary.  DINO is
an explicit cross-card measurement obligation; GPU execution remains
operator-only.

## Resume-3 diagnosis: the refusal was a checker-premise defect

**CONFIRMED — what the refused check tested.**  At producer commit
`4e20a8af573e5484d0e331200d20284a8b3c8338`, `_verify_source_order` first
required the recorded `sh2_output` to equal the value handed to TKE, then
separately required at least one `sh2_output != 0.0`.  The latter was the
`REFUSE: shear channel is inactive` premise.  It did not test whether the
producer and consumer executed; it inferred execution from numerical
nonzeroness.  The repaired gate keeps the bitwise producer-to-consumer check
and explicitly admits a zero field at a from-rest first step
(`nemo_testcase_l4_orca2_round331_tke_gate.py:124-146`).

The observer did not silently preserve its zero initialization.  Its begin
routine zeros storage and clears `lsh2`, while the `r331_sh2` call copies the
computed array and sets `lsh2=.TRUE.`
(`tke_round331_writer.F90:51-78`).  Finish refuses unless that flag is true and
unless every TKE input, boundary, matrix and solve row was captured
(`tke_round331_writer.F90:153-161`).  Thus the retained record proves both the
producer call and the downstream TKE capture ran.

**CONFIRMED — what the retained record holds.**  Rank 0 has 224,547 wet shear
cells and rank 1 has 189,572, for 414,119 total.  Every wet binary64 value is
exactly `+0.0`: min `0.0`, max `0.0`, nonzero count `0` on each rank.  Each
rank's complete `sh2_output` is bit-identical to `tke_sh2_input`.  These are
real observed zeros, not a missing or wrong array.

**CONFIRMED — NEMO computes rather than skips the channel.**  The record build
selects TKE (`ln_zdftke`) and initializes it
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:266`).  That makes
`l_zdfsh2=.TRUE.` under its resolved switches
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:275-278`), and the
time-step routine calls `zdf_sh2` before `zdf_tke`
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfphy.f90:316-336`).
`zdf_sh2` executes the `2:jpkm1` shear loop, horizontally averages it, and
only then explicitly zeroes the surface and bottom output levels
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdfsh2.f90:83-119`).

At `nit000`, this is a start from rest: NEMO sets `uu/vv(...,Kbb)=0` and copies
that stored field to `Kmm`
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/istate.f90:105-140`).  The
shear formula therefore legitimately evaluates to zero everywhere even though
it executes.  TKE consumes that live zero in its matrix/RHS
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:411-428`).

Diagnosis: the checker's nonzero premise was wrong; the observer captured the
right `p_sh2` array.  A focused plant establishes non-vacuity: the untouched
zero record passes, while moving one handoff word by one ULP refuses.  This is
**CONFIRMED**, not merely plausible.

## Retained-record readmission

The refused directory was re-admitted with `--admit-existing`; NEMO was not
rebuilt or rerun.  Admission is pinned to producer commit
`4e20a8af573e5484d0e331200d20284a8b3c8338`, executable SHA-256
`ad665a96be3b73ebbe83793c09e7996aba69fd191fb6510288be40d82161b0b3`, and
rank-record SHA-256 values `920326db...a9a42d` and
`168076b0...35fc89`.  The canonical retained directory is
`orca2_rounds/round331/acquisition/orca1ice_tke331_kt1_np2`; its admission
status is `PASS_R331_TKE_ADMISSION`.  All inherited same-program,
determinism, reference and semantic plants pass/fail as specified.

## First owning statement and single-statement change

The CPU fp64 production-JIT replay is exact on all 430,552 wet loop cells
through the recorded boundary, upper/lower/diagonal matrix arrays, and the
complete RHS.  The baseline first differs after the solve/floor on 926 wet
values, then on returned energy 901, `avm` 931 and `avt` 929.  Replacing only
the recorded matrix leaves the solve census at 926; replacing only the
recorded RHS closes it to zero.  Replaying the literal NEMO recurrence from
the recorded arrays also reproduces `after_solve` bit-for-bit.  This names the
RHS boundary, not the Thomas algorithm, as the first owning family.

Inside that boundary, NEMO's bottom TKE condition reads stored `Kbb` velocity
when it forms `zebot`
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/zdftke.f90:287-295`).  At this
first step `Kbb=Kmm=1` and the stored pair is zero.  legoESM instead formed the
boundary from the already updated solver state.  The single landed production
statement now passes the existing entry `u_now/v_now` pair into the bottom
boundary call (`ocean_model_latlon_cgrid.py:13883-13885`).  Its fallback is
unchanged for direct callers, and an incomplete pair refuses.  No selector,
constant, deck, recipe or configuration value changed.

The candidate replay confirms the owner.  Boundary, full matrix/RHS,
solve/floor and returned energy are bit-identical on every wet and every loop
cell.  Returned `avm` and `avt` improve from 931/929 unequal wet values to
7/7, with maxima `5.551115123125783e-17` and
`3.469446951953614e-18`; those seven are downstream source-order rounding,
not a regression.  Evidence:
`orca2_rounds/round331/localization/tke_statement_candidate.json`, SHA-256
`b53a1e14dd43c71c3a48b42010b4f50f3e13c30d6d7ec62741198167da7a78e7`.

## Decision-122/139, cross-card and gradient gates

The pinned baseline is preregistration commit `0f463433e04e`; the candidate is
`0034a5f2a469`.  All fourteen CPU fp64 production-JIT measurements passed the
source-ladder gate.  The aggregate result is
`PASS_R278_D122_DECISION`:

| card / label | touched rows toward / away / equal | exact rows lost | first over-bar |
|---|---:|---:|---|
| rung 0 independent | 0 / 0 / 0 | 0 | equal |
| OMT1--4 independent, each | 0 / 0 / 0 | 0 | equal |
| rung 10 independent | 183 / 0 / 0 | 0 | equal |
| rung 10 given NEMO entry | 183 / 0 / 0 | 0 | equal |

For each rung-10 label, 43 moved rows are at kt 1--3 and 140 at kt 4--10;
there is no away row at any step.  The kt-10 stage-3 sea-level maximum drops
from `6.930562311546762e-8` to `4.20363743813823e-12` m in both labels.  The
exact-loss plant fires on all seven pairs.  Aggregate evidence SHA-256:
`a07f48e7fa44a509af75497e3e5464026cc79e12336b08af7984b102f2fc5fb1`.

V1 does not execute this bottom boundary because its resolved card disables
it (`nemo_testcase_v1_orca2tke/namelist_cfg:193`).  The NEMO-faithful DINO
production card does execute it (`dino.py:1287`): three CPU steps are
bit-identical on all 66 carried leaf/step outputs between the pinned trees.
The reconstructed default-grid variant refuses before TKE because it lacks
the now-required raw NEMO depth operands, so it does not execute the changed
line and is not represented as a passing comparison.  DINO evidence SHA-256:
`0a5806c70970d5af0d9fedee60a899dcc1ec238c2bb0a67b568f09e193477db9`.
No GPU was used; the operator retains the month/GPU push gate.

The preregistered rung-10 N=1 candidate gradient check is complete.  Forward
parity has zero non-bit leaves; both reverse rows have zero non-finite initial
temperature entries and empty non-finite cotangent-leaf maps.  Reverse versus
forward relative errors are `1.819369557262068e-14` for the T direction,
`1.3868976507379975e-15` for the SSH direction, and
`1.501929835492298e-15` for `rn_ud`'s T loss (`rn_ud` reverse
`-4.51170448705634e-5`).  The construction rebuild has zero non-bit traced
leaves.  Peak RSS was 73.40 GiB.  Evidence SHA-256:
`9c1f6452a3dbd3760ae593b4504c851a99683a2dd854ee730f2d3e547a9a903c`.

## Resume-3 final gate ledger and disposition

- Gate-premise repair plus plant: pass; focused round-331 tests pass.
- Retained-output admission: `PASS_R331_TKE_ADMISSION`; no NEMO rerun.
- Localization: post-solve/returned energy exact; `avm/avt` strictly better.
- Decision 122/139: seven pairs, both labels, zero exact loss, zero away.
- Other executing card: faithful DINO 66/66 leaf-steps bit-identical; GPU
  remains operator-only.
- Rung-10 N=1 gradient: complete, finite, forward parity bit-identical, and
  reverse/forward consistent.
- Focused resume-3 functional gates: 28 passed; citation gate: 22 citations,
  0 unmapped, 0 failures, 0 map-audit failures; shifted-citation plant fired;
  citation pytest: 17 passed.
- Default ORCA2 landing battery: 149 passed in 1464.81 s, including the
  six-test private-import ratchet.  Log SHA-256
  `9ed91b2fdadb61da54454e2c91a1242d856ec8c4cfb4f021263fda6a69ff5896`.
- No configuration change: **ASKED**.
- Independent review: operator (Claude).

Final disposition: **LANDED-pending operator review, push gate and DINO month
gate**.  The only production statement landed is the stored-`Kbb` velocity
operand at the existing TKE bottom boundary.  The next open TKE debt is the
seven wet `avm/avt` source-order roundings after exact returned energy.

## Resume-4 independent-review closure

Both independent reviews returned **LAND** (Claude and Argon).  This resume
changes no NEMO source, model arithmetic, selector, recipe, deck, default or
configuration value.

**Named debt — `TKE_BOTTOM_KBB_LEAPFROG_TIME_LEVEL`.**  The stored-velocity
statement is NEMO-literal only on the certified RK3 path: the record build
calls `zdf_phy` with `Kbb=Kmm=Nbb`
(`ORCA2_ORCA1ICE_OMIP_L4_R326HMI/BLD/ppsrc/nemo/stprk3.f90:183`).  DINO's
leapfrog build instead passes `(Nbb,Nnn)`
(`DINO/BLD/ppsrc/nemo/stpmlf.f90:187-193`), so its bottom boundary reads the
BEFORE-level velocity, while legoESM's two leapfrog production calls still
pass the now level (`ocean_model_latlon_cgrid.py:16856` and
`ocean_model_latlon_cgrid.py:17232`).  Round 331 therefore certifies the
stored entry velocity only for RK3; leapfrog time-level parity remains open.

The debt has no discriminating effect on today's DINO check.  With
`rn_emin=1e-6`, `Cd0=1e-3` and `ke0=2.5e-3`, the bottom value stays on its
floor until bottom speed is about `0.5 m/s`; DINO's measured maximum was
`0.059 m/s`.  Thus the reported 66/66 DINO identity is retained but cannot
distinguish BEFORE from now velocity.

The new regression test enters the production vertical-mixing method and
captures the operands at its real bottom-boundary call.  It passes with the
landed call site.  Removing only `u_now=u_now, v_now=v_now` makes it fail with
`None is entry_u` (exit 1); restoring the call site gives 74/74 TKE-term tests
passing.  This closes the direct-helper-test review gap without changing model
behaviour.
