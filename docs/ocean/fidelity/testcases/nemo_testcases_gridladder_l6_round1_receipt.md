# Grid ladder rung L6 — round 1 receipt (gridladder_r55, Codex)

Date: 2026-10-21

Starting commit: `43ac6db0c8ff6a36f5174c48f14c730634618079`

Preregistration commit: `3e11d8c805`

Implementation and acquisition-tool commits: `11c7147545`, `3583affb12`,
`64bcc0c9e6`

Receipt and citation re-anchor commits: `a4f0eb3224`, `2e72445e10`

Status: **STOPPED — L6 BUILT; ACQUISITION_NEEDED.**  L6 is the certified L4b
card plus ORCA2 interior temperature/salinity damping in NEMO's stage-3
position.  Its exact local controls and read-only acquisition preflight pass.
Codex invoked `--run`; the operator guard stopped before building or executing
NEMO.  There is no L6 record, score, bar verdict, or certification.

- codex review: operator
- independent review: operator (Claude)
- NEMO execution: operator-only; not run here
- scoring: not run; no L6 record exists

## 1. Survey and frozen inputs

The production ORCA2 deck enables the T/S data path, uses monthly
`votemper`/`vosaline`, enables time interpolation, marks both inputs as cyclic
climatology, and names their existing input files
(`cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:44-55`).  Its damping block sets
`ln_tradmp=.true.` and `nn_zdmp=0`
(`cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:326-329`); the shared reference supplies
`cn_resto='resto.nc'` (`cfgs/SHARED/namelist_ref:1012-1018`).  Thus this is
full-water-column damping, not mixed-layer-selective damping.

The running branch is stage 3 (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:700`).
Within it NEMO calls `tra_ldf`, `tra_bbc`, `tra_bbl`, and then `tra_dmp`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-740`).  BBL is disabled on
L6, so the active order is certified isoneutral diffusion, geothermal heating,
then damping.  The receipt does not claim a BBL transcription on this rung.

`tra_dmp` obtains the target through `dta_tsd`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:181`).  For `nn_zdmp=0`, every real
level and both tracers receive the written update
`Krhs = Krhs + resto * (target - Kbb)`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:188-194`).  Initialization reads the
resolved damping namelist (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:266-270`),
initializes the target-data path if needed, opens `cn_resto`, and reads variable
`resto` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:284-307`).

The climatology path calls `fld_read`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dtatsd.f90:212`), enters ORCA_R2's damping-only
hand-adjustment arm (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dtatsd.f90:219`), applies
the Alboran/Red-Sea alterations
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dtatsd.f90:229-253`), copies the interpolated
temperature and salinity (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dtatsd.f90:261-262`),
and then masks the z/zps fields
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dtatsd.f90:308-309`).  `fld_read` derives its
half-step time from the model clock
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/fldread.f90:200-205`) and writes the two-record
linear interpolation in before-then-after order
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/fldread.f90:235-246`).

`resto.nc` is absent from the ladder's inherited 16-file set but already exists
both in the on-disk `ORCA2_ICE_v5.0.0.tar.gz` and its existing extraction.  No
file was fetched or vendored.  The extracted file and streamed tar member are
byte-identical with SHA-256
`f4b6aec4695f984eab3a3ad9f2efdcf80d19180cf5f55aa4a93e3f45639b8634`;
the archive SHA-256 is
`5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4`.
It has one binary64 `resto(z=31,y=148,x=180)` field.  L6 consumes its 30 real
levels in `(y,x,z)` order; their minimum is `0.0 s^-1` and maximum is exactly
`1.1574074074074073e-05 s^-1`.  The dummy level is excluded.

The implementation extends the existing ORCA2 climatology loader in
`nemo_testcase_recipe.py` and the existing shared tracer-relaxation kernel in
`ocean_tendency_common.py`.  It does not introduce a second climatology reader
or damping formula.

## 2. L6 card and exact source association

The new card is `ORCA2L6-dmp`.  Its rendered deck differs from L4b only by
`namtra_dmp.ln_tradmp: F -> T` and the explicit `nn_zdmp=0`; EIV, tracer
advection, changed vertical mixing, and BBL stay off.  Only L6 selects the new
default-false `rk3_stage3_damping_after_geothermal_nemo_order` flag.  The model
validator requires the certified L4b composition and rejects either BBL arm.

The ordinary stage-10c tracer-sponge route is withheld for L6.  At stage 3,
production first constructs the certified L4b T/S source, then calls the shared
relaxation kernel with full-rank `resto` and the current-step T/S targets.  The
kernel materializes NEMO's final Krhs assignment with `nemo_source_round`.
Trace fields expose the live pre-damping source and damping rate without
changing the production state.

The one-step production test runs L4b and L6 through the same live WS-RK3
observer.  For T and S it proves bitwise that the L6 pre-damping source is the
L4b composed source and that the consumed result is exactly
`nemo_source_round(L4b + resto * (target - Kbb))`.  The standalone JIT test
uses traced binary64 state, targets, and rate.  Its rounding discriminator is
`(1e16 + -1e16) + 1 == 1`, while moving damping before geothermal gives
`(1e16 + 1) + -1e16 == 0`; therefore a moved call fails.  All checked gradients
are finite.  This feature adds no division or square root and no new
zero-operand gradient site.

The card and loader plants reject missing certified L4b, BBL, incompatible
forcing rank, a non-exact deck delta, a changed `resto.nc`, or failure to vary
the interpolated target across steps.  The focused L6 file passed 7 / 7 tests
in 449.84 seconds.  The selected L4b regression and merge-probe unit tests
passed 4 / 4 in 9.16 seconds; the shared sponge tests passed 10 / 10.

## 3. Inherited-card preservation

The committed merge-identity probe ran two production steps for all 19
inherited cards: G1--G9, L4, L4b, V1--V5, and L1--L3.  Across 736 state leaves,
all 19 reports have `different_count=0` and `bit_identical=true`.  L4b was
compared directly with starting commit `43ac6db0c8ff`; the other frozen owner
snapshots retain their recorded source commits.  Candidate snapshots stamp
`64bcc0c9e6dccb6353b235c4d360eacf82da5edc`.  A planted L2 mutation was
refused with `IDENTITY_PLANT_REFUSED`.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round55/merge_identity`.

## 4. Acquisition boundary

The committed entry point defaults to a read-only seven-item preflight and has
separate `--run` and `--admit-existing` modes.  It stages L4b's deck plus only
the damping block and adds the SHA-pinned existing `resto.nc`.  It defines a
two-step acting smoke, a writer-free boundary, and three separately staged
twins to one even endpoint no greater than ten steps.  Observers canonicalize
and write only NEMO-defined wet cells.

The preflight executes whitespace-tolerant numeric/logical checks on both a
committed excerpt of a real L2 `ocean.output` and a STAMPED rendered L6 deck.
It executes `check_acting` only on the smoke and as its own statement, proves
the writer-free boundary has no observer records, contains no `cmd | grep -q`
under `pipefail`, and removes collision-safe scratch with an `EXIT` trap.  All
seven planted violations fire.  From clean commit `64bcc0c9e6`, default
preflight ended:

```text
L2_ISO_ACTING ldf_wet_nonzero=860708 ldf_dry_nonzero=0 completed_tracer_unequal=861013 entry_T_unequal=430464 entry_S_unequal=430549 entry_u_unequal=0 entry_v_unequal=0 entry_ssh_unequal=0 barotropic_checked=False barotropic_unequal=0 pass=True
L6_PREFLIGHT_7_OF_7_PASS plants=7/7 checker_functions=3/3 evidence_parent_absent=true target_absent=true
L6_PREFLIGHT_OK commit=64bcc0c9e6dccb6353b235c4d360eacf82da5edc reference=ORCA2G5_OMIP_L4_R31REF_FIX1 record_target=ORCA2L6_DMP_R55
WOULD_RUN smoke, writer-free boundary, twins A/B/C into /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round55/oracle_l6_r55
```

Codex then invoked `run.sh --run`.  It reran the complete preflight and stopped
with exit status 66 before `build_target`:

```text
REFUSE: L6 acquisition: ACQUISITION_NEEDED: --run is operator-only (set NEMO_OPERATOR_ACK=L6_R55_OPERATOR)
```

The target and evidence directory remain absent.  NEMO was not built or run.

## 5. Gates and ASKED / UNASKED

The explicit gate found 16 citations in this receipt and 17 in the
preregistration, with zero unmapped citations, zero failures, zero failing
map-audit entries, and all nine self-test plants firing in each run.  The
required citation-gate file passed 17 / 17.  The exact default ORCA2 landing
suite from `claude_rounds/land.sh` passed 145 / 145 in 1019.78 seconds with
x64 and the landing environment; that total includes the private-import
ratchet at 6 / 6 and both ORCA2 parallel-receipt checks.  No trajectory scorer
or fidelity-bar gate was invoked because no L6 record exists.

ASKED and completed: survey the running deck and compiled Oracle; verify and
SHA-pin the existing on-disk `resto.nc`; extend existing shared mechanisms;
add L4b-plus-damping in NEMO's selected order; keep BBL off; add exact JIT,
traced-input, x64, gradient, deck, validation, and call-position controls;
preserve every inherited two-step card; commit the ladder acquisition tool;
run its seven-item preflight; invoke its operator-only `--run` boundary; and
stop without scoring.

UNASKED and not done: fetch or vendor inputs; execute, build, or alter NEMO;
create, admit, or score an L6 record; claim an L6 bar verdict or certification;
enable BBL; relax any fidelity bar; push; post to GitHub; or perform an author
review.  Independent review remains with the operator.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r55-269571393
ACQUISITION_NEEDED: operator must run the committed L6 acquisition entry point with NEMO_OPERATOR_ACK=L6_R55_OPERATOR
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED — L6 built and preflighted; operator acquisition needed; not scored
