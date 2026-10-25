# Standalone ORCA2 surface-forcing design

Date: 2026-10-25.  Decision basis: Decision 168.  This document designs the
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
absent from the ORCA2 inputs, and its provenance is deliberately unresolved;
see DECISION_NEEDED below.  S0-F owns whatever yearly-record interpolation
the admitted file and calendar actually execute.

## S4 — composed standalone comparison card

Add a new NEMO configuration and build named
`ORCA2_STANDALONE_IIF_OMIP_L4`: clone the current round-5 ORCA2 deck, select
`nn_ice=1`, and point `namsbc_iif` at the user-selected pinned climatology.
Whether `key_si3` remains compiled is an explicit build decision below; it is
not silently removed.  Keep the CORE normal-year forcing/weights and every
other ORCA2 ocean option fixed.  The
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
envelope for an ice-if deck.  It is part of every viable option below.

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
all live consumers listed above.  Each option therefore has an explicit
certification cost in the decision table below; no configuration is changed
by this design round.

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
| configure/build/run NEMO | UNASKED | design only; operator work after decision |
| change the shipped comparison or training card | UNASKED | no configuration change |
| relax any bar or hide signed-zero differences | UNASKED | prohibited |
| select or manufacture an ice-cover climatology | UNASKED | DECISION_NEEDED |
| change `key_si3` build identity | UNASKED | DECISION_NEEDED; no silent choice |
| remove existing record-replay comparison modes | UNASKED | retain as diagnostics |

## DECISION_NEEDED — ice-cover provenance

1. **Monthly `fr_i` climatology from the pinned NEMO ORCA2_ICE reference
   (pick).**  Average the same calendar months on the native ORCA2 grid, pin
   run/years/digest, and name it honestly as reference-derived cover.  This is
   the lowest-confound fidelity experiment, but it is not independent
   observational evidence.
2. **Pinned external observed climatology.**  The user supplies or approves a
   named product/version; preregister its mask, calendar, units, regridding,
   missing-data rule, and digest.  This best matches the scientific phrase
   “observed cover” but introduces a new data/regridding owner.
3. **User-supplied existing `ice_cover_clim.nc`.**  Admit its provenance,
   schema, calendar, values, and digest before use; reject an opaque file.

No option is selected by this document.  The pick is option 1 for a controlled
NEMO-vs-legoESM fidelity reference; option 2 should be a later observational
sensitivity experiment.

## DECISION_NEEDED — viable NEMO reference configurations

All rows set `nn_ice=1`, explicitly resolve `ln_drgice_imp=F`, and retain the
diagnostic/coupling safety envelope above.  These are the complete viable
families while retaining the deck's TKE scheme; `nn_mxlice=2/3` with
`ln_mxl0=T` and key-SI3 plus active icebergs are not viable.

| ID | Configuration | Earlier certification | Incremental cost |
|---|---|---|---|
| K0 **(pick)** | keep `key_si3`; `ln_mxl0=T`; `nn_mxlice=0`; `ln_icebergs=F` | each changed branch is already certified separately: drag-off is the round-324 pre-D150/shipped control (0/1,360 rows); mxlice-0 is the round-326 pre-D154/round-325 control (0/1,360); icebergs-off is the accepted Phase-2d VARIANT with the Phase-2c 10-step and 30-day controls | no rebuild; new composition still needs boot, reproducibility twin, and full ten-step S0--S4 acquisition; then reserve 90 minutes for each yearly arm |
| K1 | K0 but `nn_mxlice=1` with a pinned `rn_mxlice` | **not certified** in this campaign; prior rungs certify 0 and record-fed 2 only | K0 cost plus a new TKE surface-mixing branch record, plants, gradients, and D122 comparison |
| K2 | keep `key_si3` and `nn_mxlice=2` inert by setting `ln_mxl0=F`; `ln_icebergs=F` | **not certified**; every ORCA2 rung used `ln_mxl0=T` | K0 cost plus recertification of the global stress-derived surface mixing-length anchor; highest-confound key-SI3 option |
| N0 | remove `key_si3`; set `nn_mxlice=0`; `ln_icebergs=F` | **not certified**; no earlier rung certifies the no-SI3 compiled ocean | one new build, zero-step state census, full ten-step ladder/acquisition and twin, then the yearly arm |
| N1 | N0 but retain `ln_icebergs=T`, selecting the compiled zero-ice-velocity/thickness fallback | **not certified**; existing iceberg runs use SI3 fields | N0 cost plus iceberg entry/restart/trajectory census and at least the existing 30-day iceberg control before the yearly arm |

K0 minimizes new physics and reuses three independently certified branch
states, but their `nn_ice=1` composition remains unexecuted and is not
grandfathered.  No option is selected by this document; the operator must
decide.  The engineering pick is K0.  Carrying both K0 and N0 as references
is also admissible, at roughly double build/acquisition/twin/yearly cost.
