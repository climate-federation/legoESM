# GRIDLADDER-VMIX L7 round 22 receipt — ORCA2 river-mouth mixing

Status: **STOPPED FOR RECORD; NOT SCORED OR CERTIFIED**.  Decision 141 assigns
river-mouth mixing to lane B, and Decision 146 leaves interior damping with
lane A.  This round preregistered L7 at `1544e06eac`, built it on certified V5
at `a932422d60`, ran only the committed read-only acquisition preflight, and
stopped.  Codex did not invoke `makenemo`, NEMO, MPI, or a scoring command.

## Re-verified selection and oracle program

The ORCA2 experiment selects `ln_rnf_mouth=T`, `rn_hrnf=15 m`,
`rn_avt_rnf=1e-3 m2/s`, and `sn_cnf=runoff_core_monthly/socoefr0` with
frequency zero at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:159-171`.  The admitted input is already in
the ladder set with SHA-256
`b92ff46e4c406096a1e1fe3815dfdcce12b234e84541a3b6dffb1037e777d4eb`;
no file was fetched or vendored.  Its selected variable has dimensions
`(y=148,x=180)`, no time dimension, and zero nonzero values.  NEMO's admitted
log identifies it as `socoefr0 (rec: 1)` and reports `nkrnf=2`.

The compiled initializer starts `nkrnf` at two, advances it against the
one-dimensional W-depth ladder, and calls the mask reader at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:504-526`.  That routine opens the
configured field, reads it once, and exchanges its T-grid boundary at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcrnf.f90:569-588`.  The experiment also sets
`ln_closea=F` at `ORCA2_OMIP_L4/EXP00/namelist_cfg:36`, so the closed-sea
augmentation does not turn the zero file into an acting mask.  This corrects
round 21's provisional statement that the admitted field contains active
river mouths.

The ORCA2 compiled step calls `zdf_phy` on the BEFORE state at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:179`.  The compiled coefficient
program copies the selected closure, adds
`2 * rn_avt_rnf * rnfmsk * wmask` to heat diffusivity only, then calls EVD,
double diffusion, and internal-wave mixing in that order at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfphy.f90:347-372`.  L7 therefore uses V5 as
its base; the bare G5 composition was not assumed equivalent.

## Build and exact controls

The new L7 selector attaches the static mask and three resolved deck choices
to the partial-cell vertical coordinate.  It does not add a field to the
shared model configuration, so every existing configuration pickle remains
unchanged.  The production coefficient composer applies the source-associated
heat-only expression between the closure and EVD.  Its depth helper transcribes
the oracle `gdepw_1d` loop.  No new division, square root, or NaN-gradient
source was added.

The x64 controls use traced arguments and JIT.  A nonzero synthetic mask pins
the exact left-to-right multiplication/addition, the affected W-level range,
the heat-only channel, finite gradients, a deletion plant, and a reordering
plant across EVD.  The admitted zero mask gives an exact V5/L7 identity in a
full production one-step run: all `T`, `S`, `u`, `v`, and SSH words match.
That identity is recorded only as the expected dormant-card result, never as
an acting claim.

The merge-identity controls passed the complete non-L7 configuration census
and all 14 pre-existing certified two-step state digests.  Pytest reported
15 passed in 2627.44 seconds for that census-plus-digest batch.  Thus every
other card is flag-off, every resolved configuration remains on its prior
hash, and every certified card retains its complete `T/S/u/v/eta` value hash.

## Acquisition boundary

The committed entry point is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l7_orca2rnf/run.sh`.
Default mode is read-only.  `--run` and `--admit-existing` require the exact
operator acknowledgement `L7_OPERATOR_ACK=I_ACKNOWLEDGE_NEMO_BUILD_AND_RUN`.
The target deck is V5's resolved deck with exactly the river-mouth selector
enabled; preflight compares that one-line delta and pins the stamped rendered
deck SHA-256 as
`840372df87518b7ae3e7fd6a83318ddb4a833139f72fa5021fdbcd8ae05408c3`.

The observer brackets four canonical coefficient boundaries: closure, river,
post-EVD/DDM, and post-IWM.  It zero-initializes each view and copies only wet
interior W cells, sits outside XIOS-dependent branches, and has an exact
synthetic writer/reader layout test.  The workflow is a two-step smoke, a
writer-free boundary run to NEMO's own stop or 100 steps, and three twins at a
common even endpoint no greater than 10 steps.  The nonvacuity assertion is a
separate `check_acting` statement on the smoke record only.  Because the
admitted mask is zero, the preregistered prediction is that this statement
refuses; it is intentionally not weakened to a switch or log check.

The committed preflight compiled the transformed real Fortran source and
reported:

`L7_PREFLIGHT_OK checklist=7 plants=all_refused commit=a932422d60cbffbb793f38952e17d617809d9900`

Its seven items covered the transform and continuation/order plants; a real
`makenemo -r` reference; every smoke/boundary/twin clock; exact record layout;
whitespace-tolerant numeric/logical parsing on the committed real log excerpt
(SHA-256
`efe1e074fda531cb3f3d6d5baf13fc42eb8b893bbd55a3d063ea456f918f9447`)
and the stamped rendered deck; absent-parent/output admission; and target/run
collision refusal.  Scratch cleanup is an EXIT trap, and the script contains
no `cmd | grep -q` pipeline.  The target configuration and evidence root
remain absent.  No acquisition record exists and no score was produced.

## Landing gates

All tests use `JAX_ENABLE_X64=1`, CPU JAX, and the exact `land.sh`
`PYTHONPATH`.  The focused L7 file passed seven tests, including the full
one-step control and all acquisition plants.  The exhaustive merge-identity
batch passed 15 tests as stated above.  The per-receipt citation gate, its
mandatory pytest wrapper, the default landing gate, and the private-import
ratchet are recorded at the final committed tip after citation re-anchoring.

independent review: operator (Claude)

codex review: operator

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/vmix_r22-18606393
ACQUISITION_NEEDED: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/vmix_r22-18606393/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l7_orca2rnf/run.sh --run
DECISION_NEEDED: NONE
ROUND_STATUS: STOPPED_FOR_RECORD — L7 built and preflighted; acquisition needed; record not acquired or scored
