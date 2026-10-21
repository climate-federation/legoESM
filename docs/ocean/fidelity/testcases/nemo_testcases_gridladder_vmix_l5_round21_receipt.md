# GRIDLADDER-VMIX L5 round 21 receipt — ORCA2 mixed-layer eddies

Status: **L5 CERTIFIED**.  This round surveyed mixed-layer eddies,
river-mouth mixing, and (for the record only) interior tracer damping.  It
selected L5 as mixed-layer eddies on certified G9.  Independent build review
held the first acquisition before a record existed; its hypotheses were
reproduced and corrected here.  The operator subsequently acquired the
preregistered record; Codex did not build or run NEMO.

The frozen preregistration is commit `18498bec4a`.  The first L5
implementation is commit `acda81beb9`; this continuation supersedes its
observer and numerical-association claims.

## Survey

### Mixed-layer eddies — selected L5

ORCA2's experiment deck changes only `ln_mle=T` at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:312`.  Its resolved reference group is
`cfgs/SHARED/namelist_ref:950-960`: `rn_ce=0.06`, `nn_mle=1`,
`rn_lf=5000 m`, `rn_time=172800 s`, `rn_lat=20 degrees`, `nn_mld_uv=0`,
`nn_conv=0`, and `rn_rho_c_mle=0.01 kg/m3`.  The L5 deck is therefore exactly
G9 with that one flag changed; every other card remains flag-off.

The running DINO/ORCA2 build contains `tra_mle_trp_RK3_t` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:144`.  It diagnoses mixed-layer
depth and mean buoyancy from the stage's potential density and live layer
thickness at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:234-277`, takes the
`nn_mle=1` streamfunction arm, applies its vertical structure, and adds the
horizontal transport at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:299-347`.  The complete namelist is
read reference-first and experiment-second at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:643-648`; the selected constant-f
coefficient is formed at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:697`.  In particular, the divisor is
evaluated left to right using NEMO's card-bound `omega`; the former receipt's
two-line citation did not establish that expression.

`tra_adv_trp` calls EIV and then MLE only at stage 3, evaluating the EOS at
the Kmm tracer state before MLE, at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:254-263`.  It then reconstructs the
vertical transport from the augmented horizontal pair at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:296`.  MLE therefore enters through
tracer advection and needs certified G9, not G5 or L3.  It consumes live
tracers, density, geometry, metrics, and masks; it has no external input file.

The round-20 L3 smoke log prints the complete `tra_mle_init` group and its
flag-off verdict, and the compiled source above contains the called RK3
routine.  This establishes availability in the inherited build, not that the
round-21 observer tool was safe or ready.  legoESM already had
the Fox-Kemper diagnosis and C-grid transport construction.  L5 extends that
operator with a `through_fct` route, suppresses its former standalone centred
tendency on that route, adds its existing U/V transports to the existing
stage-3 transport tuple, and reuses L3's WZV reconstruction.  No second MLE
operator was introduced.

### River-mouth mixing — proposed L6

The deck selects `ln_rnf_mouth=T`, `rn_hrnf=15 m`, and
`rn_avt_rnf=0.001 m2/s`, with both runoff and its `socoefr0` mouth mask in
`runoff_core_monthly`, at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:159-170`.  Initialization derives the number
of affected W levels and calls the mouth-mask reader at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:504-526`; that reader obtains the mask
from the configured runoff file and fixes its vertical profile at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:569-588`.

The once-per-step vertical-physics call is
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:179`.  After copying the selected
turbulent closure, `zdf_phy` adds twice the configured coefficient on the
mouth mask at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:353-355`; the resulting
diffusivity is then used by the tracer stages.  This mechanism acts without
tracer advection or eddy velocity, so its base is G5.

`runoff_core_monthly.nc` is already in the ladder's immutable 16-file ORCA2
input set and in the on-disk `ORCA2_ICE_v5.0.0.tar.gz` extraction.  No fetch or
vendoring is required.  legoESM already transcribes runoff mass and tracer
content, but not this mouth-mask augmentation of vertical diffusivity; the
next rung should extend the existing vertical-mixing composition rather than
duplicate runoff forcing.

### Interior T/S damping — record only, deferred

The deck selects full-column tracer damping with `ln_tradmp=T` and
`nn_zdmp=0` at `ORCA2_OMIP_L4/EXP00/namelist_cfg:326-329`; the reference names
`resto.nc` at `cfgs/SHARED/namelist_ref:1012-1018`.  The compiled routine is
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:142`.  It reads/interpolates target
T/S and adds `resto * (target - before)` throughout the water column at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tradmp.f90:181-194`.  The RK3 program calls it
after lateral diffusion, geothermal heating, and BBL at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:734-740`.

`resto.nc` is not in the current 16-file ladder set, but it is already present
both in the on-disk `ORCA2_ICE_v5.0.0.tar.gz` and its extracted directory, so
no download or vendoring decision is needed.  This lane does not start the
damping rung because Decision 141 permits lane A to own it.  legoESM has
generic sponge/restoring machinery, but no ORCA2 card selects this exact
full-column climatology program yet.

The proposed order is therefore: L5 MLE on G9; L6 river-mouth mixing on G5;
interior damping remains unassigned pending lane-A coordination.

## Build-review continuation

The independent review evidence was copied before use to
`gridladder_vmix_rounds/round21/review_reproduced_codex_r21/`.  The review's
first hypothesis reproduced on the real source: the BEFORE snapshot was
nested under an `iom_use` branch, while the non-XIOS `iom_use` implementation
returns false at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/iom.f90:1551-1553`; the AFTER
block could consequently write and deallocate arrays which were never
allocated.  The instrument now inserts BEFORE ahead of that branch, validates
that placement, and refuses a nested-BEFORE plant.  Its preflight transforms
the real `src/OCE/TRA/tramle.F90` input and compiles the result with
preprocessing enabled.

The remaining hypotheses also reproduced on the captured real operands:
using legoESM's rotation rate gave `rc_f=0.24057633572`, versus NEMO's
`0.24057253898`; substituting only NEMO's rate was still one ulp away because
the association differed.  The former compact V mask erased all 16 nonzero
fold-row deltas.  Before the statement-order correction, 5,409 of 8,502 U
words, 5,786 of 8,748 interior V words, and 5,653 of 16,433 wet-column mean
buoyancies differed.  The committed NumPy statement reference now evaluates
the captured NEMO operands in source order and pins complete SHA-256 values for
U, V, MLD, and mean buoyancy.  The production path is bit-identical to it,
including 16 nonzero fold-row V transports.

ORCA2 round 316's group-A change (`fae1fe1a42b9`) was imported first: only
discarded dry `q_after` and zero dry `e3t_kmm` divisors are replaced by one.
Reproduction falsified round 20's “complete three-source sequence” inference:
group A alone reduced, but did not eliminate, the nonfinite gradient.  JAX's
de-optimized NaN trace then stopped in the below-blue-level shortwave scan,
and the round-316 zero-pivot unit witness independently returned four
nonfinite entries for each differentiated vertical-solver operand.  The
matching later round-316 forward-neutral guards were therefore also imported:
dry pivots in the literal vertical solve, zero top-face surface-stress
thickness, and the shortwave divisor.  The first shortwave import covered only
levels below the blue cutoff; the equation walker then reproduced round 316
C2's 12,518 zero-thickness dry levels above that cutoff, so the complete C2
predicate also requires `ze3t != 0`.  No TKE or internal-wave path is selected
by G9.

With those actual G9 dependencies guarded, the round-20 stamped one-step T/S
loss with respect to initial T is finite on both G9 and G9+L2.  The forward
state remains pinned to the pre-import G9 and L3 two-step hashes; the guards
change no selected wet operand or forward value.

## L5 implementation and controls

The L5 card binds every resolved `namtra_mle` value, requires
`through_fct=True`, refuses each single-field wiring plant, and requires the
WS-RK3 FCT program.  The existing standalone MLE configuration remains the
default for every non-L5 caller.  The new horizontal-transport return exits
before the standalone tracer-flux divergence, so the selected path does not
compute and discard a second tendency.

The exact x64 traced transport test pins 18 nonzero U transport words, zero V
transport words in its zonal-front fixture, and an exact zero source-ordered
U sum.  The production one-step control removes only L5 MLE and pins the
resulting unequal-bit counts as 24,584 T cells, 21,878 S cells, and zero u, v,
or SSH cells.  Thus deleting the stage-3 wiring fails while the tracer-only
scope remains non-vacuous.  L5 alone selects the new NEMO-association option;
the ORCA1 and MPAS defaults and coefficient word remain unchanged.

The selector now binds the 10 m reference depth and 111 km scale cap as well
as every namelist field.  The staged route refuses
`no_mle_in_convection=True`, because it does not receive the required N2
operand, and the unused bolus-transport return was removed.  A committed
control censuses G5-G9, V1-V5, L1-L3, L5, and every non-ladder testcase
dispatcher, proves L5 is the sole MLE selector, and pins 29 non-L5 resolved
model-configuration pickles to hashes measured at `9039513ff729`.  Those
hashes are configuration pickles, not compiled step programs.  Every RK3 card
runs the guarded vertical solver; its top-stress guard is forward-neutral on
the masked faces.  G9 and L3 additionally reach the earlier group-A tracer
guards and pin their complete two-step states to that baseline.  Direct guard
tests pin live derivatives and forward equality; only L5 can enter the MLE
NEMO-association route.

## Acquisition preflight

The committed entry point defaults to read-only preflight and reserves
`--run` for the operator.  The earlier `bedbcf650e` “tool ready” claim is
withdrawn: that preflight transformed preprocessed source and did not detect
the nested allocation bug.  The corrected preflight uses the real source and
adds the nested-BEFORE refusal before reporting success.

The seven checks cover: exact `tramle` insertion plus compiler syntax; a real
`makenemo` reference; every rendered clock; strict six-frame layout and
zero-first dry/halo canonicalization; numeric Fortran-log comparison with a
neighbouring-float plant; absent-parent creation plus full completed-run audit
plants; and absence of both target configurations and the evidence root.
Admission rechecks the binary, decks, logs, immutable inputs, smoke completion,
the writer-free boundary arm, and every twin.  The operator workflow is a
two-step smoke, a writer-free run to NEMO's own stop or 100 steps, then three
twins to a common even endpoint no greater than 10 steps.  No score is allowed
until that record is admitted.

## Landing gates

All landing gates used `JAX_ENABLE_X64=1`, the exact `land.sh` `PYTHONPATH`,
and CPU JAX.  The focused L5/review suite passed 26 tests, including the real
captured-operand statement reference, north-fold row, every L5 selector plant,
observer placement, and the complete two-step G9/L3 value controls.  The
unchanged legacy ORCA1/MPAS MLE suites passed 67 tests; the shortwave and
literal-implicit guard batch passed 30 tests.

The exact historical one-step loss witnesses printed
`G9 T-loss finite=True dT_nonfinite=0 of=799200 wet_nonfinite=0
dT_nonzero=430537` and the corresponding G9+L2 result with
`dT_nonzero=430550`.  Their complete logs and the before witnesses are pinned
by `gridladder_vmix_rounds/round21/review_reproduced_codex_r21/MANIFEST.sha256`.

The heading-scoped citation gate passed all 20 citations with zero failures,
zero map-audit failures, and every planted control firing.  The complete
default landing gate then passed 145 tests, and the private-import ratchet
passed independently, 6 tests.  The writer-free entry point printed
`L5_PREFLIGHT_OK checklist=7 plants=all_refused`; it transformed and
syntax-checked the real source while leaving both targets and the acquisition
evidence root absent.  No NEMO command was run and no record was scored.

independent review: operator (Claude)

codex review: operator

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r21/repo
ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r21/repo/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l5_orca2mle/run.sh --run
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — review fixes and preflight complete; acquisition needed; record not acquired or scored

## Continuation (Codex, 2026-10-20) — admitted-record score and L5 certification

The operator ran the committed preflight and acquisition at
`9bd35acb5c41a2a0676d578d0db61a8ac80b512e`.  The preflight reported
`L5_PREFLIGHT_OK checklist=7 plants=all_refused`; the admitted acquisition
reported `L5_ACQUISITION_PASS twins=3 steps=1..10 reference_completed=21` and
placed its evidence in
`gridladder_vmix_rounds/round21/oracle_l5_r21`.  Codex invoked no NEMO build or
run command.

The final package score was made from a clean tree at
`43071fbf2f8e3629ecd82580b78ee2d8aa4f85cb`, package tree
`30c8006304061932c84eaacb6a31d33fd1cee13f`; both score records stamp an empty
dirty string.  They are
`gridladder_vmix_rounds/round21/codex_clean_score_43071fbf2f/independent.json`
and `gridladder_vmix_rounds/round21/codex_clean_score_43071fbf2f/given.json`.

Both preregistered labels score every held step, kt 1 through 10.  Each label
has 241 full-depth stage/restart rows; the independent label additionally has
five exact kt-1 entry-admission rows.  The kt-1 comparisons use the stage
records and kt 2 onward use the acquired restarts.  Every ocean value and bit
is exact, and every dry value is exact.  The only dry bit differences are
signed zero: 207,218 for the independent label and 213,822 for the given
label.  Both labels have zero ordinary value differences, zero D127 flush
classes, zero nonfinite classes, and maximum normalized error zero.  This
covers full-depth T, S, u, and v plus the two-dimensional SSH, barotropic, and
current-history fields.  Every row is therefore exact or at the declared bar:
**L5 CERTIFIED**.

The three acquired MLE witnesses have the identical SHA-256
`d306fb9e76d6a1e5a80557f19d139a8e28c2e5ff72711095ad0a774f745aa3f8`.
Against the admitted mesh, the transport changes 8,464 wet U cells and 8,764
wet V cells while every dry transport word is positive zero.  The committed
record test compares the complete before/delta/after U and V frames at full
depth, then proves that applying that exact transport produces the acquired
stage-3 T and S.  Removing MLE or its stage-3 FCT route changes 24,584 T and
21,878 S words; removing the NEMO numerical association changes 16,964 T and
13,524 S words.  The predicted mixed-layer eddy transport is thus active in
the selected wet cells and causes the acquired tracer change.

That record test executes the production step under an outer `jax.jit`, passes
the prognostic state and forcing as traced arguments, requires x64, and runs
all three wiring removals through the same compiled path.  At that
continuation, the complete L5 file passed 23 tests.  Its standing controls
retain the frozen G5-G9, V1-V5,
and L1-L3 selections and the frozen GYRE, DINO, full ORCA2, SMT, VORTEX, OMT,
and rung-0 resolved model configurations; the group-A G9 and L3 routes also
retain their complete two-step values.

## Post-certification gradient re-check

The closed south V-row metric ratio now substitutes one only for its zero
denominator before division; padding still overwrites that row, so this is a
gradient-only change.  A live-stretch-gradient test covers both the real
ORCA2 row and a synthetic closed row and fails with the guard reverted.  The
already forward-neutral top-stress divisor guard now has its own closed-face
reverse-gradient test, including an unguarded non-finite control.

The L5 file passed all 38 tests in isolated compiler workers.  Its committed
two-step value validator replays G5-G9, V1-V5, and L1-L3 plus L5 from their
own admitted forcing records.  All 14 complete `T`, `S`, `u`, `v`, and `eta`
state digests remain bit-identical to the pre-change `353b24265796` baseline.

independent review: operator (Claude)

codex review: operator

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/claude_rounds/vmix_r21/repo
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: L5 CERTIFIED — both labels exact/at bar through kt 10 at full depth; MLE transport and caused T/S change exact
