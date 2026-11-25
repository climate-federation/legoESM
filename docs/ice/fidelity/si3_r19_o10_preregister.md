# SI3 Ladder Rung O10 Preregistration — Round 19 (`si3_r19`): salinity-4 drainage and flushing

- Date: 2026-10-26. Lane `si3`, round 19, rung O10
  (`docs/ice/fidelity/si3_ladder_design.md:230`).
- Model parent: `9abb6be028cb` (through O9 and O11 model preparation).
- Status: committed before any O10 model comparison. No NEMO array has been
  read for O10 and no acquisition tool is built. DECISION 173 supplies O10
  from the one 100-step production `ORCA2_SAS_ICE` master record
  (`docs/ice/fidelity/si3_ladder_design.md:205-212`); this file registers only
  O10's projection.
- Reference build: `ORCA2_SAS_ICE_O7_REF`. Its production object was compiled
  with `-fdefault-real-8 -O3 -funroll-all-loops -fno-tree-vectorize`
  (`ORCA2_SAS_ICE_O7_REF/BLD/arch_nemo.fcm:27`). The cited source is
  `ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd_sal.f90`.

## 1. Executing production identity

The resolved deck has `jpl=5`, `nlay_i=10`, `nn_icesal=4`, both drainage and
flushing enabled, VC19 liquidus (`nn_liquidus=2`), RJW14 drainage
(`nn_sal_scheme=1`), drainage/flushing limits 50/5, `rn_simin=0.1`,
`rn_sinew=0.75`, `rn_sal_himin=0.1`, `rn_flushrate=0.3`, `rn_vbrc=0.05`,
`rn_alpha_RJW=0.03`, `rn_Rc_RJW=8`, and salinity diagnostics off
(`tests/ice/unit/data/orca2_sas_ice_expref/namelist_ice_ref:197-237`).

O10 consumes O9's first-temperature exit. In the compiled production call
order, `ice_thd_sal` follows the first `ice_thd_temp` and is followed by a
second `ice_thd_temp` (`ORCA2_SAS_ICE_O7_REF/BLD/ppsrc/nemo/icethd.f90:201-205`).
The executing salinity-4 source is initialization and drainage at
`icethd_sal.f90:311-560`, flushing at `:562-669`, and the final layer bounds
and residual salt flux at `:671-683`. The option-2, GN13, CW88, alternate
liquidus, `np_perm_eff=1`, and `ln_sal_chk` arms do not execute and are refused
rather than transcribed.

## 2. Registered master-record projection (GIVEN-NEMO-ENTRY only)

All payloads are binary64. Every category frame carries `kt`, category `jl`,
`npti`, and `nptidx(1:npti)`. The projection is every master-record step
`kt=1..100` and all five categories, including zero-length gathered categories:

- F1, immediately after `icethd.f90:201` and before `:203`, O9 TEMP1 exit / O10
  entry: `a_i h_i s_i sss t_su t_bo dh_i_sum dh_s_sum sfx_bri sfx_res`, plus
  layered `t_i e_i sz_i`.
- F2, immediately after `icethd.f90:203`, salinity exit: `a_i h_i s_i sss
  sfx_bri sfx_res`, plus layered `t_i e_i sz_i`.
- F3, immediately after `icethd.f90:205`, second-temperature exit / O11 entry:
  the F1 category state and diagnostics, with the updated `t_i` and `sz_i`.

Run scalars in the master manifest must include the resolved selectors and
coefficients named in section 1, `rDt_ice`, `r1_Dt_ice`, `r1_nlay_i`, `rhoi`,
`rhos`, `rhow`, `grav`, and `epsi06/epsi10`. O10 does not request inner-loop
observers: bounded subcycle counts and intermediate drainage/flushing states
are verified by the independent scalar transcription, so this round adds no
observer to the shared acquisition.

## 3. Frozen predictions (CPU fp64, production `jax.jit`)

- P1: `si3_ice_thd_sal(F1, resolved deck)` reproduces every F2 `sz_i`,
  `sfx_bri`, and `sfx_res` bit, on every gathered cell, category and step.
- P2: `si3_ice_thd_temp(F2)` reproduces every F3 `t_i` bit.
- P3: the selectable O10 category-stage composition consumes the public O9
  stage result and reproduces F2-F3 without a second implementation.
- P4: the independent scalar transcription, sharing no model code, reproduces
  every model output bit and both bounded subcycle counts under real eager and
  production JIT execution.
- Falsifier for P1-P4: any differing normal binary64 bit. DECISION 158's CPU
  subnormal floor is counted separately at the producing statement and is not
  called bit-identical. The first non-bit statement, if any, is named with its
  compiled-source line before another change.

## 4. Required controls

The model tests must contain: independent scalar replay; output digests and
inexact deck-value pins; drainage-only, flushing-only, inactive and bounded
subcycle branch coverage; a production-composition spy and call-removal plant;
real `jax.disable_jit()` versus JIT equality; finite gradients and one central
finite-difference check; required physical inputs with no defaults; and the D5
C1D year plus every other touched shared caller measured parent versus current.

Every executing source `MIN`, `MAX`, and `SIGN` site is inventoried against
`ORCA2_SAS_ICE_O7_REF/BLD/obj/icethd_sal.o`. Each site gets an object-code pin
and either a bit-changing survivor test or a checked proof that the survivor
cannot change a scored bit. Integer subcycle `MIN/MAX` sites are included.
Every multiply-add intermediate, constant division, and root/reciprocal-root
candidate is materialized in NEMO order. This production arm executes no
`EXP`, `LOG`, `POW`, `SQRT`, or `SIGN`; the glibc selector is therefore inert.

## 5. CHOICES (ASKED/UNASKED)

- Shared 100-step production master record: ASKED (DECISION 173).
- F1-F3 at every step and category: ASKED by the O10 design row and DECISION
  159's every-outer-stage/every-step record density.
- No O10 inner-loop observer: UNASKED, mechanical consequence of the existing
  17-kind master schema; offered for revert.
- Production selectors and values: ASKED by the unmodified production-deck
  identity; no selector, default, production config, or deck changes.
- Exact-input label only; no independent-trajectory claim: ASKED by the ladder
  design's operator-admission contract.

