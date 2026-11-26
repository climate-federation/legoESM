# ORCA2 round 337 preregistration — standalone S0/S2

Frozen 2026-10-26 before the first round-337 model score.  Base:
`aae42be88d67`.  This preregisters an operator-run acquisition but does not
build its tool, configure NEMO, execute NEMO, or acquire any data.

## Record identity

The future deck/build is `ORCA2_STANDALONE_IIF_OMIP_L4`, with Decision 175's
K0 identity: `key_si3`, `nn_ice=1`, `ln_drgice_imp=F`, `ln_mxl0=T`,
`nn_mxlice=0`, and `ln_icebergs=F`.  Decision 171's pinned native-grid yearly
ice cover is part of that future record's input manifest; this round neither
selects nor manufactures it.

Run two ranks (`jpni=2,jpnj=1`) for ocean steps kt 1--10 from one immutable
from-rest state.  Produce independent A/B twins.  Each rank writes its own
file and records owned bounds and global indices, binary64 dtype, field names
and order, kt, rank, calendar state, deck/build/source hashes, compiler flags,
resolved namelist digest, executable digest, and every input-file digest.

## Frozen call points and arrays

1. **S0-F entry**, on every call of `fld_read` for every field used by S0--S3,
   before `fld_update`: `kt,kn_fsbc,nsec_year,nsec1jan000,rn_Dt,zt_offset`,
   `freqh,clftyp,ln_tint,ln_clim,rec_shft,nreclast,nrecsec,nbb,naa,nrec`, and
   the complete before/after spatially interpolated arrays `fdta(:,:,:,nbb)`
   and `fdta(:,:,:,naa)`.  Record the resolved file, variable, scale, offset,
   and weight-file identity for atmosphere, ice cover, runoff, SSS target,
   and chlorophyll separately.
2. **S0-F exit**, immediately after `fld_update`, vector rotation, and the
   temporal blend: updated `nbb,naa,nrec`, exact integer `isecsbc`, binary64
   `ztinta,ztintb`, and `fnow` for every field.  Record at every invocation;
   mark whether `MOD(kt-1,kn_fsbc)==0` caused an update.
3. **S0-M entry**, immediately before the every-step `sbc_ssm` call at
   `sbcmod.f90:428`: `kt,nit000,nn_fsbc,l_ssm_mean,ln_apr_dyn,l_useCT`, the
   complete prior seven-field mean/restart state, `uu(:,:,1,Kbb)`,
   `vv(:,:,1,Kbb)`, `ts(:,:,1,:,Kmm)`, `ssh(:,:,Kmm)`, `ssh_ib`,
   `e3t_3d(:,:,1)`, `r3t(:,:,Kmm)`, `tmask(:,:,1)`, and `fraqsr_1lev`.
4. **S0-M exit**, at return from `sbc_ssm`: `ssu_m,ssv_m,sst_m,sss_m,ssh_m,
   e3t_m,frq_m`, including full local arrays and owned bounds.  Record every
   step, not only forcing-update steps, so reset, accumulate, and divide arms
   are all observed.
5. **S2 entry**, after S1 and the S3 `sbcice_if` call and immediately before
   `sbc_rnf`: the oracle-supplied post-ice `utau,vtau,taum,wndm,qsr,qns,emp,
   sfx,fr_i,ts(Kmm)` boundary; `sst_m,sss_m,smask0,tmask,umask,vmask`; the
   interpolated runoff and SSS-target fields; `rnfmsk`; prior `rnf,rnf_tsc`;
   `emp_corr`; and all resolved `sbc_rnf`, `sbc_ssr`, and `sbc_fwb` scalars.
   The S3 boundary is an input substitution, not an S2 claim.
6. **S2 sub-boundaries**, immediately after each return from `sbc_rnf`,
   `sbc_ssr`, and `sbc_fwb`: respectively `rnf,rnf_tsc`; then `erp,qrp,qns,
   emp,sfx`; then `qns,emp,emp_corr,a_fwb,a_fwb_b`.  These boundaries retain
   source order and local halos.
7. **S2 exit**, after `lbc_lnk` and T-to-U/V stress rotation, immediately
   before the previous-level save: `rnf,rnf_tsc,qns,emp,sfx,utau,vtau,utauU,
   vtauV,taum,qsr` and the exact `qsr` field handed to penetrative radiation.

Every array is raw native binary64 in one declared order.  S0-F is scored on
its declared full interpolation domain; S0-M and S2 are scored per rank on
each native owned wet staggering.  No assembled-domain score may replace a
rank score.

## Frozen offline predictions and falsifiers

- **R337-P1, existing records.**  The immutable round-335 S1/O1 replay remains
  `0 / 158,292` unequal bulk words and `0 / 119,880` unequal spatial-map
  words.  The round-5 surface record can independently score only fields for
  which it contains both operands and results: runoff tracer content and
  T-to-U/V stress rotation.  S0 means, generic `fld_read` temporal selection,
  SSS restoring, and the complete S2 merge are **UNMEASURED_NEEDS_ACQUISITION**
  when their entry operands are absent; substituting oracle outputs for those
  operands is not a certification.
- **R337-P2, literal improvement.**  For every independently scorable group,
  record parent and current unequal counts before inspecting residuals.  No
  group may worsen; a model claim lands only at `0 / n` unequal binary64
  words.  The first source-ordered mismatch owns the next work.
- **R337-P3, S0 cadence.**  From-rest `nn_fsbc=2` makes kt 1 equal the
  instantaneous state, kt 2 a reset-plus-single accumulation without divide,
  kt 3 the two-sample mean, and the alternating pattern repeats.  A different
  branch sequence refutes the transcription.  Restart means are never
  silently replaced with from-rest initialization.
- **R337-P4, controls.**  A one-ULP mutation in each independently scored
  output group must make the scorer nonzero.  Literal scalar references,
  whole-call eager execution, CPU fp64 JIT, finite-difference gradients,
  signed-zero/tie cases, and production-path plants must pass.  Every
  pre-existing non-NEMO caller of changed shared code is compared against the
  immutable parent; comparison against itself is invalid.

## Admission and stop rule

Admission first checks exact deck/build/calendar/compiler/ranks/steps,
source/object/writer/input hashes, complete schema, and A/B twin identity.
It then replays source order under CPU binary64 JIT with M1 barriers and the
record-build transcendental/tie policy.  A missing input operand or entry
mismatch stops at that boundary.  The kill rule applies at three strikes for
one cause or once for a structural contradiction; no aggregate or injected
oracle output may be used to conceal the first failed boundary.

## Choice ledger

| Choice | Status | Disposition |
|---|---|---|
| S0 `fld_read` temporal selection/blend | ASKED | transcribe; reuse phase-2i-certified spatial maps |
| S0 every-step seven-field `sbc_ssm` state | ASKED | `nn_fsbc=2`, from-rest and restart arms |
| S2 post-ice entry as explicit operand | ASKED | S3 remains separately owned |
| runoff mass, tracer content, and deck surface-depth arm | ASKED | extend existing runoff interfaces |
| SSS restoring and FWB correction | ASKED | deck-resolved branches only |
| T-to-U/V stress rotation and `qsr` hand-off | ASKED | source order; native staggering |
| K0 `nn_ice=1` zero/inert ice-related S2 channels | ASKED | only S2 inputs/outputs; no `sbcice_if` code |
| phase-2i spatial `fld_read` map | ASKED | reuse; do not duplicate or recertify as temporal |
| build an acquisition tool or run NEMO | UNASKED | prohibited in round 337 |
| edit `sbcice_if` or ice-cover plan | UNASKED | round 338 / Decisions 171 and 175 |
| edit the vertical solve | UNASKED | round 336 |
| alter OMIP/coupler defaults or non-NEMO behavior | UNASKED | parent/current exact control |
| change any configuration, card, bar, or gate | UNASKED | prohibited |

