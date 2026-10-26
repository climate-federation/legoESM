# Standalone ORCA2 surface-forcing design

Date: 2026-10-25.  Decision basis: Decisions 168, 171, and 175.  This document designs the
route from the comparison card's recorded surface exchange to an ORCA2 ocean
which reads CORE normal-year atmosphere, computes its own fluxes, and uses
NEMO's `nn_ice = 1` ice-interface model.  It changes no configuration.

The rule for every rung is **exact input, separate NEMO record**.  A rung is
eligible only when its input frame is bitwise the frame consumed by both
implementations; its output frame is scored cellwise in binary64; every
defined wet word must be bit-exact (`0 / n` unequal).  The existing O1
cell-score plant is retained.  A one-ULP mutation of every new output group
must also reach that same scorer and fail.  Aggregate tolerances do not certify
this ladder.

The executing order matters.  The record build first forms surface means at
`sbcmod.f90:428`, calls `sbc_blk` at `sbcmod.f90:440-449`, calls the selected
ice model at `sbcmod.f90:499-503`, then runoff, surface restoring, and the
freshwater-budget adjustment at `sbcmod.f90:517-521`.  It fills halos and
rotates T-point stress to U/V points at `sbcmod.f90:552-570`.  The numbered
rungs below are delivery increments, not permission to reorder those calls.

## S0 — time-varying input and two-step surface means

S0 has two source-ordered boundaries.  **S0-F** owns `fld_read` record
selection and temporal interpolation for every downstream time-varying field,
including ice cover, runoff, and chlorophyll.  The existing O1 result certifies
only the spatial CORE-to-ORCA2 map and rotation for the selected first record
with `ln_tint=.false.`.  It does **not** certify `fld_update` before/after
record selection, record-centre construction, `rec_shft`, or the temporal
blend.  S0-F therefore records the before/after arrays, their record times,
the exact model time, interpolation weights, and `fnow`, and scores all of
them at every forcing update.  This is required for `sn_ice`, `sn_rnf`, and
`sn_chl`, whose resolved namelist entries enable time interpolation.

**S0-M** owns the call to `sbc_ssm` on every ocean step.  With the comparison
deck's `nn_fsbc=2`, this is a stateful two-step mean, not an instantaneous
read: it initializes or restarts the accumulators, resets them on the source
schedule, accumulates `sst_m,sss_m,ssu_m,ssv_m,ssh_m,e3t_m,frq_m` every step,
and divides only at the forcing boundary.  Exact input is both time levels of
the surface ocean state plus the prior mean/restart state; exact output is the
complete post-call mean state.  Score kt 1--10 on both ranks so the alternating
reset/accumulate/divide behavior is exercised before S1 consumes the means.

## S1 — `sbc_blk` ocean side

Extend, do not copy, `packages/ocean/legoesm/ocean/bulk_flux_omip.py`, its
shared `legoesm.core.bulk_flux` kernels, and the shared transcendental policy.
Transcribe the executing NCAR `blk_oce_1`, `blk_oce_2`, `turb_ncar`, and
`sbc_phy` routines from the round-5 record build.  Source rounding barriers
separate Fortran assignments; `exp`, `log`, `log10`, `pow`, and `atan` use the
fp64 glibc arm.  MIN/MAX and SIGN retain the record object's signed-zero tie
operand.

Exact input is a frame immediately after all nine `fld_read` fields have been
updated/rotated and `sst_m`, `ssu_m`, and `ssv_m` have been formed, immediately
before the `blk_oce_1` call.  It contains `wndi,wndj,tair,humi,qsr_down,
qlw_down,precip_raw,snow_raw,slp,sst_m,ssu_m,ssv_m,smask`.  Record the resolved
deck scalars `rn_zt,rn_zu,rn_pfac,rn_efac,rn_vfac,rn_albo,ln_tair_pot,
ln_humi_dpt,ln_humi_rlh,ln_crt_fbk,ln_skin_cs` beside the frame.

Outputs are the post-`blk_oce_1` fields `theta_air,q_air,precip,tsk,ssq,
Cd,Ch,Ce,rhoa,sensible,latent,evap,utau,vtau,taum,wndm,qsr`, followed by the
post-`blk_oce_2` fields `qlwn,qns,emp`.  The bar is `0 / n` unequal binary64
words on each output's defined wet T domain, separately per rank and step.
The existing phase-2h O1 record already contains all external inputs and 18
defined public outputs at kt 1 rank 0; it is the offline first-rung record.
The fuller acquisition below is preregistered for the standalone deck rather
than inferred after this offline score.

## S2 — remaining executed surface terms

Build on the existing `sss_restoring.py`, runoff/river-source code, and stress
rotation in the comparison card; grep those production paths before adding
any helper.  S0-F supplies the certified **temporally interpolated** runoff
and chlorophyll fields.  Exact input is S1's public
output plus the pre-call ocean means/state and the deck's independently read
runoff, runoff salinity/temperature/depth, and SSS-restoring fields.

Because `sbc_ice_if` executes before runoff/restoring in the real program, the
S2 record also contains the **post-ice-interface boundary as an explicit
oracle-supplied operand**.  S2 certifies only the source-ordered operations
after it: `sbc_rnf`, `sbc_ssr`, `sbc_fwb`, halo exchange, T-to-U/V stress
rotation, runoff mass plus tracer content, freshwater/non-solar merge, salt
flux, and the `qsr` hand-off to penetrative radiation.  Its outputs are
`rnf,rnf_tsc,qns,emp,sfx,utauU,vtauV,qsr` and the penetrative-radiation entry
field.  Each field's bar is independently `0 / n` on its native owned wet
staggering for both ranks and every recorded step.  This rung does not claim
the substituted ice boundary.

## S3 — `sbc_ice_if` (`nn_ice = 1`)

Replace S2's oracle substitution with a transcription of the 147-line
`sbcice_if` unit.  It reads its cover with `fld_read` at
`sbcice_if.f90:132-150`, obtains freezing temperature and clamps the surface
state at `sbcice_if.f90:152-167`, then computes model ice fraction and modifies
`qsr`/`qns` at `sbcice_if.f90:169-196`.  Exact input is the S1 post-bulk
boundary plus `ice_cover`, `sss_m`, `sst_m`, `ts(Kbb)`, `ff_f`, masks, and the
freezing-point output.  Outputs are the post-call `ts(Kmm),fr_i,qsr,qns`; the
source states that stress, `taum`, `wndm`, `emp`, and `sfx` remain unchanged.
Score changed and unchanged channels independently at `0 / n` on both ranks
and all recorded steps, with signed zero included.

The temperature clamp mutates `ts(:,:,:,Kmm)` over `A2D(nn_hls)`, including
the halo, before the owned-interior flux loop; the S3 record and scorer must
therefore carry and compare that halo mutation, not only owned T cells.

The reference namelist requests a **yearly mean** `ice_cover_clim.nc` record:
`sn_ice%freqh=-12`, `ln_tint=.true.`, and `clftyp='yearly'`.  In `fld_read`,
`-12` selects one yearly record, not twelve monthly records.  The file is
absent from the ORCA2 inputs.  Decision 171 supplies it as a pinned native-grid
yearly mean from a fixed full-SI3 ORCA2 run.  S0-F owns whatever yearly-record
interpolation the admitted file and calendar actually execute.

## S4 — composed standalone comparison card

Add a new NEMO configuration and build named
`ORCA2_STANDALONE_IIF_OMIP_L4`: clone the current round-5 ORCA2 deck, keep
`key_si3`, select `nn_ice=1`, resolve `ln_drgice_imp=F`, retain
`ln_mxl0=T`, select `nn_mxlice=0`, resolve `ln_icebergs=F`, and point
`namsbc_iif` at the Decision-171 pinned climatology.  This is K0, selected by
Decision 175.  Keep the CORE normal-year forcing/weights and every other
ORCA2 ocean option fixed.  The
matching legoESM card reads those same files, enters S1 from its own
S0-F/S0-M output, executes S1 -> S3 -> S2 in NEMO source order, and never
reads a NEMO exchange record.

**Decision 172's premise is refuted.**  Keeping `key_si3` with `nn_ice=1`
does not preserve the certified ocean path.  `sbcmod` calls neither
`sbc_ice_alloc` nor `ice_init`; `zdfdrg` forcibly resolves
`ln_drgice_imp=F`; and TKE/GLS reject `nn_mxlice>1` before the first step.
The current `ln_icebergs=T` is independently unsafe: the key-SI3 iceberg path
reads `snwice_mass`, `at_i`, `vt_i`, `u_ice`, and `v_ice`, none of which was
allocated or initialized.  This is a boot/undefined-read defect, not merely a
different scientific choice.

An exhaustive import-and-dereference grep over `src/OCE` gives this safety
envelope for the selected ice-if deck.

| SI3-allocated field/read cone | Required resolved state at `nn_ice=1` |
|---|---|
| `rCdU_ice` in `zdfdrg` | `ln_drgice_imp=F` (NEMO also coerces it false) |
| `hm_i,h_i` in TKE/GLS | with `ln_mxl0=T`, use `nn_mxlice=0` or `1`; alternatively disable `ln_mxl0` and recertify the global surface anchor |
| `snwice_mass*` in `stp2d`/`dynspg` | retain `ln_ice_embd=F` |
| `snwice_mass*` in freshwater budget | `nn_fwb=2` may remain, but NEMO resolves `nn_fwb_voltype=2` when `nn_ice/=2` |
| `snwice_mass*`, `at_i`, `vt_i`, `u_ice`, `v_ice` in icebergs | with `key_si3`, require `ln_icebergs=F`; without the key, the compiled fallback supplies zero ice velocity/thickness |
| `a_i` in `sbcice_if` | retain standalone `ln_cpl=F`; the assignment is coupling-only |
| SI3 fields in diagnostics/assimilation/BDY | retain `ln_diadct=F`, `ln_diaobs=F`, `ln_asmdin=F`, `ln_asmiau=F`, `ln_bdy=F`, and do not request `sshice` |
| bulk, ABL, coupler, restart, and ice-writer SI3 fields | already runtime-dead under resolved `nn_ice=1`, `ln_abl=F`, `ln_cpl=F`; restart/ice-writer calls are guarded by `nn_ice==2` |

Dropping `key_si3` is still not an equivalent simplification.  It changes
compiled TKE/GLS, top-drag, iceberg, bulk/ABL, coupler, allocation, restart,
and diagnostics branches.  Conversely, retaining the key requires disabling
all live consumers listed above.  Decision 175 therefore selects the
key-SI3 K0 safety envelope above; no shipped configuration or model code is
changed by this design round.

K0's drag-off and mixing-under-ice-off branches are **not NEMO-certified**.
The `0 / 1,360` figures in rounds 324 and 326 compare legoESM's flag-off arm
with earlier legoESM rows.  They never compare against a NEMO run with
`ln_drgice_imp=F` or `nn_mxlice=0`; every existing ORCA2 NEMO record ran with
drag on and `nn_mxlice=2`.  The accepted Phase-2d icebergs-off evidence also
does not certify the new K0 composition.  K0 and its changed branches receive
their first NEMO comparison inside S4; none is grandfathered from those
controls.

The first S4 action is an **operator-run boot plus ten-step smoke run**.  Its
checklist is fail-closed:

1. The resolved namelist and compiled-key echo must show exactly K0, including
   `key_si3` present and `key_xios` absent, with the Decision-171 input and its
   digest.
2. Both ranks must boot, reach and complete step 10, exit zero, and contain no
   NEMO error or stop banner; retain `ocean.output`, resolved namelists,
   executable digest, and restart/output inventory.
3. The acquisition builds delete `key_xios`, so the ice-axis calls in
   `iom.F90:241-245` are compiled out by the outer `key_xios` guard at
   `iom.F90:116-123` and the enabled ice output group at
   `ORCA2_OMIP_L4/EXP00/file_def_nemo-ice.xml:13-14` is not consumed.  If a
   future smoke unexpectedly retains `key_xios`, refuse it before stepping:
   at `nn_ice=1`, `sbcmod.F90:331-340` calls neither allocation branch nor
   `ice_init`, while `ice_init` is what calls `par_init`
   (`icestp.F90:241-263`) to read `jpl` and `nlay_i`
   (`icestp.F90:341-346`).  The reported zero-length axes must then be resolved
   or the enabled ice output disabled and separately approved; a successful
   no-XIOS smoke is not evidence for an XIOS-enabled build.

After the smoke passes, S4 acquires a reproducibility twin and the exact entry,
S0/S1/S3/S2 boundaries plus the existing ten-step ladder checkpoints from the
same K0 NEMO executable.  It scores every registered word and each binding
plant at the exact bar.  The drag-off and `nn_mxlice=0` branches are certified
only if that K0 NEMO comparison passes; a boot alone is not certification.
Only after the exact S4 comparison passes may the 2,920-step statistical arms
run.

The exact-input NEMO record contains entry state, every S0/S1/S3/S2 boundary,
and the existing ten-step ladder checkpoints for steps 1--10 on ranks 0--1.
The composed bar first requires every surface boundary above to be `0 / n`,
then applies the existing per-row rung-10 bar without relaxing any constant.
Only after a ten-step independent run passes does the one-year (2,920-step)
statistical comparison run.  The round-5 two-rank record spent 11.22145 s in
ten steps and 13.077387 s total; linear step-only extrapolation is about 54.6
minutes for 2,920 steps.  Build, I/O, and instrumentation cost are unmeasured,
so reserve 90 minutes plus build time per NEMO arm and run a reproducibility
twin before the candidate.

## Choice ledger

| Choice | Status | Disposition |
|---|---|---|
| standalone surface flux + `nn_ice=1` target | ASKED | five-rung S0--S4 design above |
| `fld_read` temporal interpolation | ASKED | S0-F; prior O1 certifies spatial mapping only |
| two-step `sbc_ssm` means for `nn_fsbc=2` | ASKED | S0-M, every step and restart state |
| literal record-build NCAR arithmetic | ASKED | S1, including barriers/libm/ties |
| reuse existing bulk, restoring, runoff, rotation, and fld-read code | ASKED | extend after grep; no parallel copy |
| one exact-input NEMO record per rung | ASKED | separate immutable record labels |
| S1 model preparation and offline phase-2h score | ASKED | this round |
| preregister, but do not build, S1 acquisition | ASKED | companion preregistration |
| K0 reference configuration | ASKED | Decision 175; first applied by the operator in S4, not by this design edit |
| configure/build/run NEMO | ASKED | future S4 operator work; boot plus 10-step smoke is first |
| change the shipped comparison or training card | UNASKED | no configuration change |
| relax any bar or hide signed-zero differences | UNASKED | prohibited |
| select or manufacture an ice-cover climatology | ASKED | DECISION 171: pinned native-grid yearly mean from a fixed full-SI3 ORCA2 run |
| retain `key_si3` build identity | ASKED | Decision 175 |
| remove existing record-replay comparison modes | UNASKED | retain as diagnostics |

## DECISION 171 — ice-cover provenance (TAKEN)

Use one pinned native-grid **yearly mean** of `fr_i` from a fixed full-SI3
NEMO ORCA2 run.  Record the producer commit, run and averaging years, calendar,
mask, schema, and file digest.  No monthly record set or external regridding is
part of this reference.  An independently observed product may be added later
only as a separately named sensitivity experiment.

## DECISION 175 — NEMO reference configuration (TAKEN)

K0 is selected: `key_si3`, `nn_ice=1`, `ln_drgice_imp=F`, `ln_mxl0=T`,
`nn_mxlice=0`, and `ln_icebergs=F`, with the complete off-consumer safety
envelope above.  This supersedes the refuted Decision 172.  It selects the
reference but does not claim prior certification: the operator-run S4 smoke is
first, and the ensuing exact S4 acquisition is the first NEMO comparison and
certification opportunity for K0's drag-off and `nn_mxlice=0` branches.
