# PREREG — lane B rung L2 (preregistered round 17, vmix_r17)

Frozen before any L2 code or measurement. No L2 record exists and this round
does not score one. Decision 138 and the round-16 L1 certificate remain in
force.

**L2 = certified G5 + ORCA2's standard isoneutral tracer diffusion, exactly
as its resolved `&namtra_ldf` selects it. Eddy-induced velocity remains off.**
The six configuration overrides are `ln_traldf_lap = .true.`,
`ln_traldf_iso = .true.`, `ln_traldf_msc = .true.`,
`nn_aht_ijk_t = 20`, `rn_Ud = 0.018`, and `rn_Ld = 200.e3`
(`ORCA2_OMIP_L4/EXP00/namelist_cfg:298-308`). The remaining resolved keys come
from the reference group: `ln_traldf_OFF`, `ln_traldf_blp`,
`ln_traldf_lev`, `ln_traldf_hor`, `ln_traldf_triad`, `ln_triad_iso`, and
`ln_botmix_triad` are false, `rn_slpmax = 0.01`, and `rn_sw_triad = 1`
(`ORCA2_OMIP_L4/EXP00/namelist_ref:915-948`). `rn_Ld`, `ln_triad_iso`,
`rn_sw_triad`, and `ln_botmix_triad` are resolved but inert under mode 20 and
the standard operator. `&namtra_eiv` remains G5's `ln_ldfeiv = .false.`.

## Survey of the statements that run

All compiled-source citations are from
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/`.

- **Slopes — `SUBROUTINE ldf_slp`** (`ldfslp.f90:99`). The routine consumes
  locally referenced density and N2, makes the U/V gradients
  (`ldfslp.f90:185-213`), applies the 0.01 and 7-km stability caps and the
  mixed-layer ramp (`ldfslp.f90:222-256`), Shapiro-filters U/V
  (`ldfslp.f90:260-275`), builds/caps/ramps the W slopes
  (`ldfslp.f90:284-317`), Shapiro-filters W (`ldfslp.f90:319-334`), and
  exchanges their lateral boundaries (`ldfslp.f90:338`). `stp_RK3` computes
  EOS/N2 and calls it once before the stages when `l_ldfslp` is true
  (`stprk3.f90:170-193`).
- **Coefficient — `SUBROUTINE ldf_tra_init`** (`ldftra.f90:181`). It reads the
  reference and configuration groups (`ldftra.f90:231-233`), resolves the
  standard isoneutral Laplacian (`ldftra.f90:278-289`), sets the slope flag
  (`ldftra.f90:312-313`), forms `zUfac = 0.5*rn_Ud`
  (`ldftra.f90:353-358`), and sends mode 20 to `ldf_c2d`
  (`ldftra.f90:386-390`). `SUBROUTINE ldf_c2d` (`ldfc1d_c2d.f90:125`)
  independently assigns U and V coefficients as
  `pUfac*MAX(e1,e2)**knn` (`ldfc1d_c2d.f90:160-164`) and copies them down the
  column (`ldfc1d_c2d.f90:168-172`); `ldf_tra_init` masks them once
  (`ldftra.f90:427-434`). Mode 20 is static, so the time-varying `ldf_tra`
  call at `stprk3.f90:193` is not taken.
- **Explicit tensor and MSC — `SUBROUTINE tra_ldf`** (`traldf.f90:69`) sends
  the resolved operator to `traldf_iso_lap` (`traldf.f90:105-110`) and adds
  its tendency to the RHS (`traldf.f90:124-128`).
  `SUBROUTINE traldf_iso_lap` (`traldf_iso.f90:73`) calls
  `traldf_iso_a33` (`traldf_iso.f90:154`), constructs the horizontal and
  off-diagonal vertical fluxes, and with MSC keeps only
  `ah_wslp2-akz` explicit (`traldf_iso.f90:257-291`).
  `SUBROUTINE traldf_iso_a33` (`traldf_iso.f90:746`) makes full A33
  (`traldf_iso.f90:780-795`) and the nonnegative MSC stabiliser
  (`traldf_iso.f90:811-830`). The stage-3 caller places `tra_ldf` before the
  vertical solve (`stprk3_stg.f90:734-749`).
- **Implicit K33 — `SUBROUTINE tra_zdf_imp`** (`trazdf.f90:134`). With DDM
  off it constructs one active-tracer matrix (`trazdf.f90:171-176`), adds
  `akz` to heat's `avt` under MSC (`trazdf.f90:180-191`), and uses the same
  isoneutral term on salt (`trazdf.f90:198-208`). The latter is the literal
  salt-matrix rule landed in ORCA2 round 313.

The G5 writer-free and record builds contain ORCA2-identical `.text` for
`ldfslp.o`, `ldftra.o`, `ldfc1d_c2d.o`, `traldf.o`, `traldf_iso.o`, and
`trazdf.o`; this is a run-time namelist selection, so **no binary is built**.
The record build's `stprk3_stg.o` is also ORCA2-identical. The writer-free
stage objects differ because of instrumentation history, not physics; their
six listed operator objects are identical. No isoneutral input file is read:
the live grid metrics, T/S, thickness, masks, EOS/N2, and namelist scalars are
the complete inputs.

## Existing model path and every known difference

Extend the existing `gm_redi` path; do not duplicate it. The VORTEX SMT-3
card already selects ORCA2's active block: native neutral slopes with NEMO's
caps, mixed-layer ramp, carried step-entry N2, reciprocal metrics, live QCO
face/depth geometry and Shapiro filter; independently evaluated mode-20 U/V
coefficients from `rn_Ud`; the standard rotated-Laplacian tensor; MSC; and
implicit K33. The shared operator now also contains the landed closed deepest
W mask and live stage-3 Kmm divisor. ORCA2 round 313 already gives salt the
same K33 contribution while building a separate salt matrix only for DDM.

Known differences from NEMO's executed statements, all retained as measured
debt rather than hidden:

1. The shared explicit tensor defaults to JAX/vectorised association for its
   horizontal flux, area reciprocal, divergence, and return tendency; NEMO's
   source fixes pairwise parentheses and adds directly to the existing Krhs.
   VORTEX SMT rounds 14-24 measured this arithmetic boundary; source-order
   candidates that failed their cellwise ratchet remain held.
2. legoESM returns the lateral tendency to the WS-RK3 composition instead of
   mutating NEMO's `ts(...,Krhs)` array in `tra_ldf`. Its stage placement is
   the same, but this composition is not claimed bit-for-bit before an L2
   record exists.
3. legoESM supplies the computed MSC `akz` to its implicit tracer solver;
   NEMO adds it to `avt` while constructing `tra_zdf_imp`. The heat/salt
   matrix ownership is now the same (round 313), but arithmetic association
   in the complete ORCA2 stage is unmeasured.
4. `rn_Ld = 200000`, `ln_triad_iso = F`, `rn_sw_triad = 1`, and
   `ln_botmix_triad = F` have no state fields because the executed mode-20,
   standard-operator branches never read them. The deck reader and selector
   must nevertheless pin them as resolved values.

These differences require neither new model state nor a new data file, so the
round proceeds.

## Card change and purity claim

Add `ORCA2L2-iso` as a G5-derived SI3 case. It selects a shared helper also
used by VORTEX SMT-3, with `kappa_GM = 0`, `kappa_Redi` only as the stated
equatorial mode-20 scalar, and the literal metric/Ud path active. BBL,
geothermal heating, MLE, damping, tracer advection, momentum diffusion, TKE,
EVD, DDM, IWM, and every other G5-off card remain off. In particular this L2
does not inherit L1's BBL.

One-step jaxprs of G1-G5, V1-V5, L1, full ORCA2, DINO, and VORTEX SMT-3 must
be byte-identical to base `fab19df07cd3`; L2 must differ from G5. L2's initial
state must equal G5's field by field.

## Acquisition (defined exactly; operator runs it)

- **Binaries:** reuse G5's pinned writer-free
  `ORCA2G5_OMIP_L4_R31REF_FIX1` and record
  `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1`; build nothing. Preflight pins their
  hashes and the object/source identity described above.
- **Namelist diff against G5:** only the complete `&namtra_ldf` resolved group
  changes from OFF/F/F/F/F/F plus inactive reference coefficients to the
  values at the top of this preregistration. `&namtra_eiv` remains
  `ln_ldfeiv = .false.`. The ice deck is identical.
- **Inputs:** G5's sixteen SHA-pinned files. No new input.
- **Records:** G5's restarts 2..N over twins A/B/C, surface-input records and
  step-entry records every kt, plus the existing kt-1 stage-3 tracer stream.
  That stream records Krhs immediately before and after `tra_ldf`, then T/S
  after `tra_zdf` (`stprk3_stg.f90:731-760`).
- **What proves the isoneutral flux acts:** in the kt-1 stage-3 stream,
  post-`tra_ldf` minus pre-`tra_ldf` must be nonzero in at least one wet T or
  S cell and zero in every dry cell. The completed post-`tra_zdf` T or S must
  differ from G5's admitted stage-3 record. The kt-2 step-entry T or S must
  differ from G5, while u, v, sea level, and both barotropic velocity fields
  remain bit-identical because L2 changes only tracer diffusion. A self-file
  control must be refused as inert.
- **Logs:** require the complete resolved `&namtra_ldf` values, the standard
  isoneutral initialisation line, MSC true, mode 20, `rn_Ud = 0.018`,
  `rn_Ld = 200000`, and `ln_ldfeiv = F`; refuse BBL and geothermal heating.
- **Steps:** preflight -> 2-step record-binary smoke and acting check ->
  100-step writer-free boundary -> twins A/B/C through
  `min(boundary, 10)`, rounded down to the surface-forcing cadence -> admission.

## Predictions

- **P1 (record):** the exact group resolves, the smoke acts, at least ten
  steps complete, and twins agree. Falsifier: acquisition refuses.
- **P2 (placement):** the stage stream and kt-2 entry meet the acting checks
  above. Falsifier: no wet tracer move, any dry move, or any momentum/SSH move.
- **P3 (score, PLAUSIBLE):** do not evaluate this round. On a later admitted
  record, L2 is either at the bar or its first debt is inside the stage-3
  `tra_ldf`/`tra_zdf` pair named above; no earlier-stage owner is inferred.
- **P4 (purity):** all listed flag-off jaxprs are identical and the L2 jaxpr
  alone moves. Any other move blocks landing.

