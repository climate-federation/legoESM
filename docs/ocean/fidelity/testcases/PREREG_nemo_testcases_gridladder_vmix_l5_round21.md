# GRIDLADDER-VMIX lane — round 21 (vmix_r21): L5 mixed-layer eddies

Status: **PREREGISTERED; ACQUISITION NOT YET RUN**.

L5 is one switch on certified G9: ORCA2's Fox-Kemper mixed-layer-eddy
transport.  It is not composed with L2 or L3.  NEMO adds it to the stage-3
tracer transport before FCT, so G9 is the smallest certified base that can run
the statement (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:254-265`).

## Frozen resolved card

The ORCA2 deck selects `ln_mle=T` at
`ORCA2_OMIP_L4/EXP00/namelist_cfg:310-313`.  The other eight members come from
the reference group at `cfgs/SHARED/namelist_ref:950-961`.  L5 therefore binds
exactly:

- `ln_mle=T`, `nn_mle=1`, `rn_ce=0.06`, `rn_lf=5000 m`;
- `rn_time=172800 s`, `rn_lat=20 degrees`, `nn_mld_uv=0`;
- `nn_conv=0`, `rn_rho_c_mle=0.01 kg m-3`.

Any missing, additional, mistyped, or changed member is a refusal.  All other
cards retain `ln_mle=F` and their normalized one-step programs must be
identical to the round-20 tip.

## Frozen predictions and falsifiers

P1. L5 reuses the existing shared MLE mixed-layer diagnosis, streamfunction,
and transport construction.  The ladder extension only exports that existing
transport to the certified G9 stage-3 FCT route.  A second MLE operator, a
standalone centered tendency on L5, or any stage-1/2 MLE transport refutes P1.

P2. NEMO's `nn_mle=1` path diagnoses the mixed layer and mean buoyancy at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:234-278`, constructs the two face
streamfunctions at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:299-306`, forms
their vertical structure at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:317-343`,
and adds their vertical differences to U/V transport at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:344-348`.  The L5 value tests will be
exact, jitted, use traced inputs in fp64, and fail when the corresponding
wiring is removed.

P3. `nn_conv=0` means the convection suppression branch at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/tramle.f90:309-314` is not selected.  L5 must
therefore configure the existing MLE core with that gate off.  Leaving the
core's production default on refutes P3.

P4. The L5 acting witness is the six canonical transport frames immediately
around NEMO's `tra_mle_trp` call: before/delta/after U and V at kt 1 stage 3.
Three independent twins must be byte-identical, the delta must be nonzero on
at least one defined wet face, and canonical dry/halo cells must be +0.  A raw
undefined cell, a zero acting delta, or any twin disagreement refuses
admission.

P5. Acquisition is smoke first, then a writer-free boundary run to NEMO's own
stop or 100 steps, then three twins to the common even endpoint no greater
than 10 steps.  Admission rechecks the binary, immutable inputs, all rendered
decks, numeric float log values, smoke, boundary arm, every twin, exit status,
endpoint, frames, and canonical domain.  Each check has a planted violation.

P6. This round stops after the committed read-only preflight.  No NEMO build,
run, trajectory score, bar verdict, or certification is permitted before the
operator produces and admits the record.

## Survey disposition and order

- **L5 mixed-layer eddies:** no external input; compiled and initialized from
  `tra_adv_init` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:675`); base G9.
- **L6 river-mouth mixing:** ORCA2 selects `ln_rnf_mouth=T`, `rn_hrnf=15 m`,
  and `rn_avt_rnf=1e-3 m2/s` at
  `ORCA2_OMIP_L4/EXP00/namelist_cfg:159-164`.  It reads `socoefr0` from the
  already-admitted ladder `runoff_core_monthly.nc` and adds heat diffusivity
  after the closure in `zdf_phy`; base G5.  It is next because it needs no new
  acquisition input.
- **Interior T/S damping:** recorded only, not started.  ORCA2 selects
  `ln_tradmp=T`, `nn_zdmp=0` at
  `ORCA2_OMIP_L4/EXP00/namelist_cfg:326-330`; the reference selects
  `cn_resto='resto.nc'` at `cfgs/SHARED/namelist_ref:1012-1019`.  `resto.nc`
  is absent from the ladder's sixteen-file set but present in the already
  downloaded and extracted `ORCA2_ICE_v5.0.0` input bundle, SHA-256
  `f4b6aec4695f984eab3a3ad9f2efdcf80d19180cf5f55aa4a93e3f45639b8634`.
  Decision 141 permits lane A to take it, so this lane does not assign it a
  rung yet.

## Planned evidence

The committed acquisition entry point will be
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l5_orca2mle/run.sh`.
Its default action is read-only preflight; `--run` is operator-only and
`--admit-existing` revalidates an already-produced record.  Evidence will live
outside git under `gridladder_vmix_rounds/round21/oracle_l5_r21/`.
