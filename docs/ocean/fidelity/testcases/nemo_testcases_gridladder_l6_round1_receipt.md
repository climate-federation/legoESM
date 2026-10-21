# Grid ladder rung L6 — round 1 receipt (gridladder_r55, Codex)

Date: 2026-10-21

Starting commit: `43ac6db0c8ff6a36f5174c48f14c730634618079`

Preregistration commit: `3e11d8c805`

Implementation and acquisition-tool commits: `11c7147545`, `3583affb12`,
`64bcc0c9e6`

Receipt and citation re-anchor commits: `a4f0eb3224`, `2e72445e10`

Review-remediation commit: `945f885f94`

Remediation receipt and citation commits: `73bb58474f`, `31d3095311`,
`f742ed19df`

Resume scoring and order-control commit: `b8149cd7a1`

Status: **LANDED — L6 CERTIFIED.**  L6 is the certified L4b card plus ORCA2
interior temperature/salinity damping in NEMO's stage-3 position.  The
operator's admitted kt 1--10 record, both trajectory labels, every available
full-depth row, the kt-1 stage records, and both selected trend records are all
exact or at the preregistered bar.

- codex review: operator
- independent review: operator (Claude HOLD reproduced and remediated)
- NEMO execution: operator-only; no NEMO execution in this resume
- scoring: complete from a clean tree

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
kernel materializes both binary64 roundings in NEMO's final Krhs assignment:
the damping product is rounded before the final addition is rounded.
Trace fields expose the live pre-damping source and damping rate without
changing the production state.

The independent review falsified the original exactness claim with the test
trace disabled: XLA fused the damping multiply and final add on this CPU.  The
supplied random-array probe found 229,255 temperature and 226,838 salinity
words unequal out of 808,080 under JIT, while eager execution had zero.  With
the product-rounding guard, both JIT counts are zero.  The standalone JIT test
now uses traced binary64 non-unit rates and non-power-of-two values and proves
the two-rounding result bit for bit; all checked gradients remain finite.

The original “moved call fails” discriminator was only standalone NumPy
arithmetic, not a model run; that claim is retracted.  The replacement uses
the ordinary compiled L4b and L6 model traces on the same random, per-cell
INEXACT T/S state.  It requires L6's live pre-damping source to equal L4b's
live composed source bit for bit, and requires the final live source to equal
`nemo_source_round(pre + nemo_source_round(resto * (target - Kbb)))` bit for
bit.  Its runtime plant leaves the production call text present but defers
geothermal until after damping; the pre-source assertion fails and the final
composed source changes.  Thus execution, not source-text order, pins damping
after geothermal.  A separate ordinary one-step plant removes the inner
product-rounding guard and changes the returned T/S state.  This feature adds
no division or square root and no new zero-operand gradient site.

The card and loader plants reject missing certified L4b, BBL, incompatible
forcing rank, a non-exact deck delta, a changed `resto.nc`, or failure to vary
the interpolated target across steps.  Each target now carries its NEMO step;
the L6 model requires the caller's positive step and rejects a missing or stale
stamp before compilation.  Thus a scorer cannot silently reuse the stored
kt=1 target at kt=2.  The corrected focused and shared-sponge battery passed
29 / 29 tests in 188.91 seconds.

The admitted-record gate runs the shared damping implementation through an
actual JIT with all seven operands traced as binary64 arrays.  It fails if the
JIT call is removed, if any recorded operand is closed over instead of traced,
if x64 is removed, or if the L6 damping wire is removed.  The kt-2 wire-removal
plant moves both temperature and salinity rows.

## 3. Inherited-card preservation

The committed merge-identity probe ran two production steps for all 19
inherited cards: G1--G9, L4, L4b, V1--V5, and L1--L3.  Across 736 state leaves,
all 19 reports have `different_count=0` and `bit_identical=true`.  L4b was
compared directly with starting commit `43ac6db0c8ff`; the other frozen owner
snapshots retain their recorded source commits.  Candidate snapshots stamp
`64bcc0c9e6dccb6353b235c4d360eacf82da5edc`.  A planted L2 mutation was
refused with `IDENTITY_PLANT_REFUSED`.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round55/merge_identity`.
The clean scoring commit retains the same packages tree
`bff737b887e4205cfe8efbf4504c1adc436933e9`; the resume changes only the scorer,
tests, and this receipt, so all 19 zero-difference reports remain applicable.

## 4. Acquisition boundary

The committed entry point defaults to a read-only seven-item preflight and has
separate `--run` and `--admit-existing` modes.  It stages L4b's deck plus only
the damping block and adds the SHA-pinned existing `resto.nc`.  It defines a
two-step acting smoke, a writer-free boundary, and three separately staged
twins to one even endpoint no greater than ten steps.  Observers canonicalize
and write only NEMO-defined wet cells.

The preflight executes whitespace-tolerant numeric/logical checks on both a
committed excerpt of a real L2 `ocean.output` and a STAMPED rendered L6 deck.
For L6 it now requires the active-mask damping banner and the successful
record-1 `resto.nc` read, not the generic initialization text also emitted with
damping disabled.  A SHA-pinned round-54 damping-off log is refused.
It executes `check_acting` only on the smoke and as its own statement, proves
the writer-free boundary has no observer records, contains no `cmd | grep -q`
under `pipefail`, and removes collision-safe scratch with an `EXIT` trap.  All
seven checklist plants fire.  The synthetic writer now emits an inert kt=1
payload, which passes, followed by an acting kt=2 payload; a separate inert
kt=2 plant is refused.

The first operator acquisition reached the two-step smoke and was refused
because the old checker incorrectly required kt=1 to act.  The retained smoke
record proves all 430,552 wet cells have target equal to before and no RHS
changes at kt=1, while kt=2 changes all 5,871 cells with positive `resto` for
both tracers.  The failed evidence is under
`gridladder_rounds/round55/oracle_l6_r55_failed1`; its build was moved to
`cfgs/ORCA2L6_DMP_R55_failed1`, leaving the acquisition target free.  No failed
payload was admitted as an L6 record.

From clean remediation commit `945f885f94`, default preflight ended:

```text
L2_ISO_ACTING ldf_wet_nonzero=860708 ldf_dry_nonzero=0 completed_tracer_unequal=861013 entry_T_unequal=430464 entry_S_unequal=430549 entry_u_unequal=0 entry_v_unequal=0 entry_ssh_unequal=0 barotropic_checked=False barotropic_unequal=0 pass=True
L6_PREFLIGHT_7_OF_7_PASS plants=7/7 l6_specific_plants=3/3 checker_functions=3/3 evidence_parent_absent=true target_absent=true
L6_PREFLIGHT_OK commit=945f885f9407479f32b59bc549705158357c0a95 reference=ORCA2G5_OMIP_L4_R31REF_FIX1 record_target=ORCA2L6_DMP_R55
WOULD_RUN smoke, writer-free boundary, twins A/B/C into /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round55/oracle_l6_r55
```

Codex then invoked `run.sh --run` without the operator acknowledgement.  It
reran the complete preflight and stopped
with exit status 66 before `build_target`:

```text
REFUSE: L6 acquisition: ACQUISITION_NEEDED: --run is operator-only (set NEMO_OPERATOR_ACK=L6_R55_OPERATOR)
```

The operator then ran the corrected tool from commit `3c0c35711d`.  It admitted
three identical kt 1--10 damping and geothermal inventories, writer-free
boundary output, and restarts covering every scored kt 2--10.  Damping is
correctly inert at kt 1 because target equals Kbb, then acts on all 5,871
positive-restoring wet cells for both tracers at every kt 2--10.  The accepted
record is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round55/oracle_l6_r55`.
The acquisition log SHA-256 is
`0c2eb6aa6876bdbb9f56261612313e6ceea9a098f94ecffeb651737a388b6a56`.
Its final line is:

```text
L6_ACQUISITION_PASS steps=1..10 twins=3 evidence=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round55/oracle_l6_r55
```

NEMO was not built or run in this resume.

## 5. Clean-tree L6 score

All five score files stamp clean commit
`b8149cd7a14c970146a6d046b8cb606548a1dc85` and packages tree
`bff737b887e4205cfe8efbf4504c1adc436933e9`.  CPU, x64, production JIT, the
admitted restart roots, and step-specific damping targets were used.

| label / record | coverage | exact / at bar | unequal words |
|---|---:|---:|---:|
| INDEPENDENT surface | kt 1--10 | 70 / 70 exact | 0 |
| INDEPENDENT full/partial/dry depth | restarts kt 2--10 | 34 exact, 36 / 36 at bar | 3 |
| GIVEN-NEMO-ENTRY surface | direct + two-step outputs kt 1--10 | 70 / 70 exact | 0 |
| GIVEN-NEMO-ENTRY full/partial/dry depth | restarts kt 2--10 | 34 exact, 36 / 36 at bar | 3 |
| stage surface | kt 1, stages 1--3 | 21 / 21 exact | 0 |
| stage full/partial/dry depth | kt 1, stages 1--3 | 12 / 12 exact | 0 |
| geothermal wet-bottom trend | kt 1--10 | 164,330 / 164,330 cells exact | 0 |
| damping wet-cell T/S statements | kt 1--10 | 8,611,040 / 8,611,040 cells exact | 0 |

The three nonexact full-depth words in each trajectory label are temperature:
two full-cell words at kt 9 and one at kt 10.  Their largest normalized error
is `5.803808481195575e-20`, well below the fixed `1e-15` bar.  Every partial
cell and dry slot is exact.  The geothermal record has 16,433 wet columns per
step, including five zero-flux wet columns.  The damping record has 430,552 wet
cells per tracer per step and canonical zero dry slots.

Score SHA-256 hashes:

```text
86ddf2f4e1369b133ab4f1b9fd225140a1dbdd76b3dca41a0eccdd64c42249e2  l6_independent_kt1_10.json
4ad8b149fde4f8dd9bdbacf1fc88e00ca7d056b02c6d232c66c37b0c8ebbb135  l6_given_entry_kt1_10.json
81d8842ff320a1c62959c6eb5f615340fa584999c90b5828abcc62ca44a7ced2  l6_stages_kt1.json
88ae64815ef48f5a7a9625c9507d491aadf74e283f06f245d56987e626c33082  l6_geothermal_kt1_10.json
9f5e40ecb9275a8f63afe1393999453aa2df8353a1e22d13f11b7fb59dde66af  l6_damping_kt1_10.json
```

L6 is therefore **CERTIFIED: damping after geothermal in NEMO's order**.

## 6. Gates and ASKED / UNASKED

The explicit citation gate found 16 citations in this corrected receipt with
zero unmapped citations, zero failures, zero failing map-audit entries, and all
nine self-test plants firing.  Its shifted `tradmp` citation plant exited
nonzero.  The exact ORCA2 default landing suite from `claude_rounds/land.sh`
passed 145 / 145 in 1058.33 seconds under CPU, x64, and the landing package
path; this includes the private-import ratchet at 6 / 6 and both ORCA2
parallel-receipt checks.  All 10 focused L6 tests passed across the split
foreground runs: the trace/order plant in 546.43 seconds, the ordinary
guard-removal plant in 177.81 seconds, and the remaining eight L6 controls plus
both merge-identity tests in 11.61 seconds.  The admitted-record JIT/traced/x64
and damping-wire test was also isolated and passed in 7.74 seconds.  The
default citation-map audit was re-anchored after the L6 scorer extension and
is green from clean commit `369e35725c`.

ASKED and completed: survey the running deck and compiled Oracle; verify and
SHA-pin the existing on-disk `resto.nc`; extend existing shared mechanisms;
add L4b-plus-damping in NEMO's selected order; keep BBL off; add exact JIT,
traced-input, x64, gradient, deck, validation, and call-position controls;
preserve every inherited two-step card; commit the ladder acquisition tool;
run its seven-item preflight; admit the operator record; restore trace-based
runtime order assertions with a moved-call plant; score both labels through kt
10 at every available depth; score the kt-1 stages and both trend records; and
certify only because every row is exact or at the fixed bar.

UNASKED and not done: fetch or vendor inputs; execute, build, or alter NEMO;
enable BBL; relax any fidelity bar; push; post to GitHub; or perform an author
review.  Independent review remains with the operator.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/gridladder_r55/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — L6 CERTIFIED: damping after geothermal in NEMO's order; both labels and every recorded depth are exact or at bar
