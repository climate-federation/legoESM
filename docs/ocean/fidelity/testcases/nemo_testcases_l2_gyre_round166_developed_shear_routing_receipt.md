# Round 166 receipt — developed shear routing

**Status: HELD.**  No production physics, configuration, carried state, or
restart schema changed.  The developed-state walk names the first non-bit
shear value as an inherited live face thickness: legoESM supplies the RK3
stage-3 half-step free surface where NEMO's shear call supplies the step-entry
slot in both formal positions.  The one-variable routing split makes all four
face metrics and `p_sh2` bit-exact through the production JIT, but it is held
because day-240 T RMS worsens by `5.925159807240732e-10` K and ORCA2-zps also
executes the route without a native measurement.  Production was restored at
`5740b7264`; its model source is byte-identical to incoming round-166 tip
`edbb74902`.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round166/`.  The frozen
preregistration is commit `f309ee26b`; the measurement extension is
`7998836a8`; the candidate is `81a743bb95034f3255eaec506b64e27d2f61a735`.
The current-tip-applicable held patch is
`scripts/validate/ocean_fidelity/testcases/manifests/nemo_testcase_l2_gyre_round166_held_shear_routing_split.patch`.

## Compiled-source statement

The record-producing program has a commented `Nbb,Nnn` call immediately
followed by the executed `Nbb,Nbb` call at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/stprk3.f90:167-168`.
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfphy.f90:317-320`
passes those two slots to `zdf_sh2`.  The no-Stokes U statement and divisor
are `GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90:97-103`; its V
twin is `GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90:104-108`.
The T-point collapse is
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90:111-114`, followed
by the surface/bottom assignments at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdfsh2.f90:116-119`.

The masks consumed there are constructed at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/dommsk.f90:234-242`, and the
live U/V ratios are written from the supplied sea surface at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/domqco.f90:211-218`.
Therefore NEMO's own executed call requires step-entry free-surface ratios in
both shear divisor factors.  This is a time-level routing statement, not a
new selector or stabiliser.

## Production-JIT walk

The existing developed TKE walk was extended; no second state bridge was
created.  It loads NEMO's admitted step-1080 restart and executes legoESM's
production-jitted step 1081.  The unobserved and observed returned states
have zero unequal bytes.  Rebuilding NEMO's `zsh2u`, `zsh2v`, and `p_sh2`
from its own recorded operands is BIT, so the reference replay is calibrated.

| source-order row | baseline unequal / scored | baseline max abs | candidate unequal / scored | candidate max abs |
|---|---:|---:|---:|---:|
| U/V face velocities | `0 / 21,780`, `0 / 22,080` | `0` | same | `0` |
| U/V vertical differences | `0 / 21,054`, `0 / 21,344` | `0` | same | `0` |
| carried `avm` and U/V face sums | all BIT | `0` | same | `0` |
| step-entry `r3u/r3v` | `0 / 726`, `0 / 736` | `0` | same | `0` |
| `e3u` NOW / BEFORE | each `16,820 / 21,054` | `4.265924758328765e-06` m | each `0 / 21,054` | `0` |
| `e3v` NOW / BEFORE | each `16,530 / 21,344` | `4.269350483809831e-06` m | each `0 / 21,344` | `0` |
| U/V reconstructed divisor | `16,820`, `16,530` non-bit | `2.5629308947827667e-03`, `2.5649776071077213e-03` | both BIT | `0` |
| masks and coast factors | all BIT | `0` | same | `0` |
| production `p_sh2` | `17,400 / 17,400` | `2.1204821986655657e-15` s-2 | `0 / 17,400` | `0` |

The first baseline non-bit statement is therefore the divisor at the cited
U/V assignments, caused by the wrong free-surface input rather than wrong
arithmetic inside the divide.  The candidate closes that statement under the
production JIT.  It reduces the following TKE RHS from 14,649/18,000 at
`3.053494349730679e-11` to 9,699/18,000 at
`1.3877787807814457e-17`; the remaining first source-order operand is `rn2`
at 16,394/17,400 and `1.8431436932253575e-18`.  That RHS is NEMO's compiled
sum at
`GYRE_OMIP_L2_P3_SM_R164TKEDEV/BLD/ppsrc/nemo/zdftke.f90:438-441`.
The `rn2` remainder is the production-rounding floor already localized in
rounds 106--108, not a new magnitude candidate.

The production control advances consumed U velocity `[1,11,1]` by one ULP.
It moves two vertical-difference cells and four `p_sh2` cells, prints
`STATUS PLANT-FIRED`, and exits 1.  Thus neither the trace nor its final row
is vacuous.

## Certified trajectory and month

The canonical comparison uses the same 70-row oracle-relative comparator as
the live ladder.  The first attempt compared the trajectory-only candidate
to a full report and was refused for schema mismatch; it is retained but is
not scientific evidence.  The corrected offline comparison carries the
candidate report's complete clean-worktree stamp and is canonical.

| property | result |
|---|---:|
| rows compared | `70` |
| moved rows | `55`, all in the committed TSV registry |
| status changes | `0` |
| kt1 AT-BAR losses | `0` |
| first-over-bar | kt3 T/S/U/V/SSH -> kt3 T/S/U/V/SSH |
| legacy cellwise comparison | FAIL, maximum worsening `7,513,154` ULP |

| headline | before | candidate | disposition |
|---|---:|---:|---|
| kt2 T | `1.4210854715202004e-14` | same | AT-BAR |
| kt2 S | `2.1316282072803006e-14` | same | AT-BAR |
| kt2 U | `8.326672684688674e-17` | same | AT-BAR |
| kt2 V | `9.71445146547012e-17` | same | AT-BAR |
| kt3 T | `4.940071072212504e-07` | same | DEBT |
| kt3 S | `4.0086298724872904e-08` | same | DEBT |
| kt3 U | `4.7515474156903555e-06` | `4.751486826853023e-06` | improves |
| kt3 V | `5.008617717412733e-06` | `5.008341423334839e-06` | improves |
| day-30 T RMS | `6.572574374770603e-05` | `6.572571060658968e-05` | improves `3.314111634e-11` K |

Decision 43 permits kt>=2 movement, so the legacy cellwise FAIL is reported
but is not this candidate's veto.  All 55 moved rows are registered exactly;
the gate reports no missing or unexpected row.

## Mandatory year

The candidate ran independently from rest for 360 days under the same fp64
harness and flags as the immutable round-163 arm.

| day | before T RMS (K) | candidate T RMS (K) | candidate - before | row |
|---:|---:|---:|---:|---|
| 30 | `6.57257437477060260e-05` | `6.57257106065896828e-05` | `-3.31411163431976491e-11` | PASS |
| 60 | `2.08080200035203869e-04` | `2.08080234333835458e-04` | `+3.42986315889602655e-11` | WORSE |
| 90 | `1.86430187737560626e-03` | `1.86430187250401767e-03` | `-4.87158859259484434e-12` | PASS |
| 120 | `1.04847381054101003e-03` | `1.04847377317762139e-03` | `-3.73633886417007455e-11` | PASS |
| 180 | `3.58388546864078263e-03` | `3.58388592651926212e-03` | `+4.57878479489387535e-10` | WORSE |
| 240 | `1.64483607011786798e-02` | `1.64483612936946605e-02` | `+5.92515980724073188e-10` | **VETO** |
| 300 | `1.36029900410691660e-02` | `1.36029902473891861e-02` | `+2.06320020096351087e-10` | WORSE |
| 360 | `1.12256600185513065e-02` | `1.12256598013908172e-02` | `-2.17160489243695132e-10` | PASS |

The measured run-to-run floor is about `2e-10` K at day 240.  The candidate's
day-240 movement is `3.60e-08` relative and about three floor units.  More
importantly, no certified trajectory row changes DEBT to AT-BAR, so Decision
55's less-than-`1e-3` relative year exception does not apply.  Decision 45
therefore vetoes the candidate mechanically.  The day-240-worse plant prints
`STATUS PLANT-FIRED` and exits 1.

## Recipe-derived blast radius

The Decision-43 gate resolves every real recipe rather than using a
hard-coded card list.

| card | executes this exact route | reason / disposition |
|---|---|---|
| GYRE-zco | yes | measured above |
| ORCA2-zps | yes | prognostic TKE, step-entry native-Nbb2 shear, live QCO faces; **UNMEASURED-WITH-SPEC** |
| LOCK_EXCHANGE-zco | no | no vertical-mixing physics |
| OVERFLOW-zps | no | no vertical-mixing physics |
| generic NEMO-GYRE recipe | no | implicit-solve, centered shear path |
| DINO KAMM | no | Euler momentum program and centered shear |
| DINO KAMM-MLF | no | Euler momentum program and NOW x BEFORE shear |

The exact changed call argument exists only in the WS stage program, so DINO
does not execute it and no DINO before/after measurement is required.  ORCA2
does execute it; the gate reports `unmeasured_executing_cards=[ORCA2-zps]`.
That is a second independent landing failure.  The spec is one native ORCA2
production step and ten-step ladder with step-entry versus half-step shear
metrics captured under its own deck; pending Decision 58 is not answered or
changed here.

## Gate verdict, tests, and review

The complete Decision-43/45 report is FAIL.  It reports: month improvement
PASS; month/year day-30 identity PASS; all eight year rows registered PASS;
first-over-bar retention PASS; no kt1 loss PASS; all 55 movements registered
PASS; DINO non-execution PASS; day-240 non-worsening FAIL; all executing cards
measured FAIL because of ORCA2.

Focused tests report `82 passed, 2 failed in 43.37s`.  The two failing nodes
are the campaign-state pre-existing debts
`test_step_entry_n2_bundle_matches_live_geometry_construction` and
`test_step_entry_n2_bundle_fails_closed_without_raw_w_mesh`; the new shear,
card-census, offline-provenance, and movement-registry tests pass.

The required combined `tests/ocean/fidelity` plus `tests/ocean/unit` run was
started once with 12 workers.  It reached 92%, eight workers terminated
improperly in JAX/native compilation, replacement workers stopped making
progress, and the controller emitted no terminal pytest summary.  After the
workers were gone and the log was unchanged for more than three minutes, the
orphaned controller was interrupted (exit 130).  This is reported as **NO
TERMINAL SUMMARY**, not as a pass and not as a count of scientific failures.

The required separate command was run with `codex exec --sandbox read-only`
against the current clone.  It emitted no scientific verdict.  Its exact
disposition is:

> independent review unavailable in-sandbox

The terminal diagnostic was `Error: failed to initialize in-process
app-server client: Read-only file system (os error 30)`.  There is no hidden
SHIP or DO NOT SHIP verdict.

## Canonical evidence hashes

| artifact | SHA-256 |
|---|---|
| `developed_shear_walk.json` | `462b400974d06689245f7eddaf5591f9b12a65ad75d67006526f7ac26a438b15` |
| `developed_shear_walk_candidate.json` | `35bfd6a94757b850578db5c769d7b9e705eb49c58a9302e0ab1e84f917272698` |
| `developed_shear_walk_plant.json` | `d77bc50c9567a79e456b3114f32617fefbd93d8895b8f20c654d914cd6596054` |
| `ladder_candidate.json` | `89d6d70a4624436e38a7d42b1053f6b0ba2deae0e6fa9d67e64923e8275e2186` |
| `ladder_comparison_offline.json` | `6b5f7f735d3a5ebff2c743bc2eba364707a9f07783974535fbb0dbf626c74f4b` |
| `day_gap_candidate.json` | `dd52f711930d1523cc8204a2d282f4eb7624aef79bd48b310bf08f72670e1155` |
| `year_gap_candidate.json` | `76410e9681955ad9622fc5e6423157f22c78c79a270a142d72810c9514e0e7d7` |
| `decision43_45.json` | `c3567a906f5b55029801f45f6d77ef92200fbc1f5b609bce5b1f864a53fa4d9e` |

## OPEN — round 167

1. Do not re-land or retune the shear routing split; its local source proof is
   closed and its year/ORCA2 vetoes are registered.  The immutable before arm
   remains round 163.
2. Do not walk `rn2` or its depth weight again: rounds 106--108 and this round
   bound it at the production-rounding floor.  Return to magnitude ranking
   inside the developed vertical-diffusion owner and select the largest
   remaining non-floor boundary after the restored production `p_sh2`, using
   day-240 sensitivity rather than local cell count.
3. The first eligible next measurement is a one-variable developed-state
   ranking of the post-sweep TKE/closure outputs that feed `avt` versus the
   implicit tracer solve.  Name a compiled statement only if that ranking
   carries day 240; otherwise exonerate the TKE closure and move to the
   tracer-solve operands.  No configuration or carried-state change is
   authorized.
4. ORCA2's shear measurement is deferred until this candidate is otherwise
   eligible; measuring it cannot cure the independent day-240 veto.

No acquisition and no user decision are requested.
