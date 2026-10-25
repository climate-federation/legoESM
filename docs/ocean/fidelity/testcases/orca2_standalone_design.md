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

Source inspection confirms that `sbcmod` permits `nn_ice=1` in a
`key_si3` build: only case 2 calls `ice_stp`.  It does not prove the new deck
boot-safe.  With `key_si3`, bulk initialization still allocates and validates
SI3 air-ice fields; `sbcice_if` binds `a_i` from the SI3 module, while
`sbcmod` calls neither `ice_init` nor the no-SI3 allocator for `nn_ice=1`.
The candidate must therefore pass an initialization/ten-step smoke gate with
coupling disabled before any scientific comparison.

Dropping `key_si3` is not an equivalent simplification.  It selects different
compiled ocean branches in `zdftke`/`zdfgls` surface mixing-length scaling and
removes the SI3 implicit top-drag contribution in `zdfdrg`; it also changes
bulk/ABL/coupler allocation and blending branches.  Most SI3-only routines
would be runtime-dead at `nn_ice=1`, but these shared ocean paths are not
certified equivalent merely because `ice_stp` is skipped.

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

## DECISION_NEEDED — NEMO reference build identity

1. **Keep `key_si3` compiled and set `nn_ice=1` (pick).**  This preserves the
   comparison card's compiled ocean mixing/drag/bulk branches and changes only
   the runtime ice selector.  Cost: the executable still links SI3, retains
   SI3 initialization/namelist obligations, and needs the explicit boot/ten-
   step smoke gate above because `nn_ice=1` skips `ice_init`.
2. **Remove `key_si3` and recertify a new ocean reference.**  This produces a
   smaller, conceptually cleaner ice-if executable, but changes TKE/GLS surface
   mixing, top drag, allocation, ABL, bulk, and coupler code at compile time.
   Cost: a new zero-step state comparison plus the full ten-step ocean ladder;
   it cannot inherit the current comparison card's certification.
3. **Carry two NEMO references.**  Use option 1 as the fidelity control and a
   no-SI3 build as a sensitivity.  This resolves attribution best but doubles
   build, acquisition, twin-reproducibility, and long-run cost.

No option is selected.  The pick is option 1 because it preserves the ocean
code paths the existing card actually certified; option 2 is admissible only
as a separately named and fully recertified reference.
