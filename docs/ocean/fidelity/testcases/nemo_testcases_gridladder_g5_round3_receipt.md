# NEMO test cases: grid ladder G5 round 3 receipt — ice/ocean top drag

**Date:** 2026-10-16  
**Rung:** G5 = certified G4 + ORCA2 SI3 only  
**Verdict:** requested top-drag path landed; **G5 remains HELD, not certified**

The preregistration was committed at `784857c14` before candidate
measurement.  The first candidate made every kt-1 stage AT-BAR but retained
123 U and 180 V last-place differences after stage 3.  Preserving NEMO's
unaveraged face sum through `dyn_zdf` removed those differences.  The final
kt-1 stage handoffs are exact.  The first remaining over-bar row is kt-2 V,
so the preregistered certification prediction is falsified.

## NEMO deck and source

ORCA2 resolves `ln_drgimp=.true.` and `ln_drgice_imp=.true.` from
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/EXP00/namelist_ref:817-818`; the general
top-drag defaults are `rn_Cd0=1.e-3` and `rn_Uc0=0.4` at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/EXP00/namelist_ref:823-824`, and SI3's
ice/ocean coefficient is `rn_Cd_io=5.e-3` at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/EXP00/namelist_ice_ref:136`.  For this
no-cavity deck, `zdf_drg` copies `rCdU_ice` to `rCdU_top` at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/zdfdrg.f90:178-190`;
SI3 forms the recorded, non-positive value at
`ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/iceupdate.f90:403`.

The four executing consumers, opened at their enclosing routines, are:

| consumer | G5 compiled source |
|---|---|
| U/V frozen external coefficient | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynspg_ts.f90:1428-1433` (`dyn_drg_init`, :1404) |
| top baroclinic residual, forward/NOW branch | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynspg_ts.f90:1466-1489` (`dyn_drg_init`) |
| non-wetting external-mode drag (the deck's executing branch) | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynspg_ts.f90:633-640` (`dyn_spg_ts`, :136) |
| `dyn_zdf` top barotropic RHS and U/V implicit diagonal | `ORCA2G5_OMIP_L4_R31FULLENTRY_FIX1/BLD/ppsrc/nemo/dynzdf.f90:158-180`, `:303-314`, `:470-482` (`dyn_zdf`, :83) |

The apparent naming mismatch in the third row is NEMO's own source: this
deck has `ll_wd=.false.`, so the feature called implicit ice drag executes
the explicit external-substep statements at :635-639; the 3-D solve remains
implicit.

## Landing

The record already contains `rCdU_ice`; no acquisition was needed.  D123's
G5 forcing route now supplies it through `OceanSurfaceForcing`.  One shared
helper preserves its negative sign, U/V staggering, north fold, signed zero,
and NEMO's pair-sum/add-zero/half association
(`packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:4077-4103`).
The same result feeds the external coefficient
(`packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:3315-3329`),
the top baroclinic residual
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:6714-6727`),
the substep loop (`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:7587-7590`),
and the `dyn_zdf` top RHS/diagonal
(`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:13511-13544`).

`implicit_ice_ocean_top_drag` is appended with default `False`
(`packages/ocean/legoesm/ocean/state.py:3257-3266`) and selected only by the
G5 card (`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:3828-3832`).
Missing `rCdU_ice`, non-implicit vertical mixing, a non-baroclinic-only solve,
bottom-drag composition, or the wide-halo external solver raises.  G1 through
G4 retain the old path bit-for-bit.

## Input and coefficient controls

The kt 1..10 forcing identity has 16,433 wet T cells and zero unequal input
rows.  At kt 1, all 3,630 ice-covered wet cells have nonzero `rCdU_ice`, range
`[-1.5504269195258426e-3, -4.816384579465245e-10]` m/s.  The coefficient
replay scores all 15,853 wet U faces and 15,875 wet V faces: zero unequal and
zero max absolute error.  NEMO has 3,519 nonzero U faces and 3,593 nonzero V
faces; the V score includes 68 northern-fold faces.  The planted input change
refuses with exit 1.

## kt 1 staged score

Bar is normalized max absolute error <= `1e-15`; `A` means AT-BAR.

| exposure | u | v | T | S | ssh |
|---|---|---|---|---|---|
| stage 1 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 |
| stage 2 | A 0/0 | A 0/0 | A 1.110e-16/1 | A 7.105e-15/2 | A 0/0 |
| stage 3 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 |

Cell format is `status max_abs/unequal`.  The nonzero stage-2 tracer rows are
below their field-relative bar and return to exact at stage 3.

## Both labels, kt 1..10

### INDEPENDENT

| kt | ssh | uu_b | vv_b | u | v | T | S |
|---|---|---|---|---|---|---|---|
| 1 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 |
| 2 | A 1.110e-15/3550 | A 1.787e-16/4076 | A 1.735e-16/4326 | A 3.046e-15/3267 | D 2.984e-15/3256 | A 4.441e-16/5 | A 7.105e-15/7 |
| 3 | D 5.951e-14/6950 | D 9.850e-15/7337 | D 7.362e-15/7435 | D 4.063e-14/5876 | D 4.144e-14/5853 | A 1.776e-15/12 | A 7.105e-15/11 |
| 4 | D 7.483e-14/9776 | D 9.168e-15/10043 | D 6.453e-15/10039 | D 1.241e-13/8052 | D 1.235e-13/8024 | A 1.776e-15/22 | A 1.421e-14/24 |
| 5 | D 3.797e-14/13119 | D 1.215e-14/13408 | D 9.159e-15/13466 | D 1.062e-13/9811 | D 1.090e-13/10335 | A 3.553e-15/33 | A 1.421e-14/38 |
| 6 | D 4.619e-14/14425 | D 1.598e-14/14642 | D 1.124e-14/14769 | D 7.538e-14/11177 | D 7.519e-14/11786 | A 3.553e-15/38 | A 2.132e-14/43 |
| 7 | D 3.997e-14/14675 | D 1.844e-14/15007 | D 1.285e-14/15001 | D 6.197e-14/12065 | D 6.317e-14/12473 | A 7.105e-15/55 | A 2.132e-14/53 |
| 8 | D 6.284e-14/15011 | D 1.762e-14/15053 | D 1.525e-14/15125 | D 8.091e-14/12713 | D 8.294e-14/13123 | A 7.105e-15/89 | A 2.132e-14/92 |
| 9 | D 5.818e-14/15425 | D 1.844e-14/15354 | D 1.698e-14/15391 | D 7.438e-14/13289 | D 6.922e-14/13565 | A 1.066e-14/103 | A 2.842e-14/120 |
| 10 | D 6.017e-14/15547 | D 2.105e-14/15416 | D 1.608e-14/15508 | D 8.476e-14/13631 | D 8.592e-14/13838 | — | — |

There are 68 surface rows: 41 DEBT and 7 bit-identical.  The all-level
restart score has 36 rows and 17 DEBT.  Its first DEBT is kt-2 V: full-cell
relative `1.489e-15`, partial-cell relative `4.571e-16`; worst later relative
errors are `5.232e-14` for partial U and `1.334e-14` for full V.

### GIVEN-NEMO-ENTRY (odd entries plus their two-step continuations)

| kt | ssh | uu_b | vv_b | u | v | T | S |
|---|---|---|---|---|---|---|---|
| 1 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 | A 0/0 |
| 2 | A 1.110e-15/3550 | A 1.787e-16/4076 | A 1.735e-16/4326 | A 3.046e-15/3267 | D 2.984e-15/3256 | A 4.441e-16/5 | A 7.105e-15/7 |
| 3 | A 1.443e-15/3760 | A 1.527e-16/4331 | A 2.498e-16/4433 | D 5.845e-14/3425 | D 5.945e-14/3353 | A 8.882e-16/5 | A 7.105e-15/3 |
| 4 | D 5.784e-14/7305 | A 3.886e-16/7703 | A 6.809e-16/7781 | D 5.205e-14/6246 | D 5.240e-14/5927 | A 1.776e-15/10 | A 7.105e-15/9 |
| 5 | A 1.110e-15/3544 | A 2.359e-16/4131 | A 1.943e-16/4189 | A 4.011e-15/3424 | A 4.149e-15/3417 | A 8.882e-16/4 | A 7.105e-15/5 |
| 6 | D 2.998e-15/6775 | A 5.759e-16/7226 | A 3.608e-16/7388 | D 3.302e-14/5673 | D 3.245e-14/5845 | A 4.441e-16/12 | A 1.066e-14/12 |
| 7 | A 1.332e-15/3756 | A 1.804e-16/4535 | A 1.388e-16/4530 | A 2.859e-15/3589 | A 3.400e-15/3579 | A 8.882e-16/3 | A 7.105e-15/3 |
| 8 | A 2.442e-15/7277 | A 8.778e-16/7920 | A 9.021e-16/7928 | D 4.577e-14/6188 | D 4.646e-14/6196 | A 1.776e-15/11 | A 7.105e-15/12 |
| 9 | A 1.554e-15/3749 | A 3.331e-16/4447 | A 2.776e-16/4426 | A 1.693e-15/3692 | A 2.290e-15/3709 | A 4.441e-16/8 | A 7.105e-15/11 |
| 10 | D 5.174e-14/7092 | A 1.414e-15/7545 | A 1.339e-15/7615 | D 1.404e-14/6108 | D 1.370e-14/6076 | — | — |

There are 68 surface rows: 14 DEBT and 7 bit-identical.  The 20 all-level
rows contain 9 DEBT.  Direct odd-step entries 5, 7 and 9 are entirely AT-BAR;
the two-step continuation is not.  Thus the remaining problem is not the
record route and G5 is not certified.

## Compiled-order walk after kt 1

The record has no restart after kt 1, so a pure given-entry kt-2 stage score
is unavailable.  The committed post-hoc arm saved the exact kt-1 card state
and compared kt-2's ordinary external-mode exit with NEMO's existing frame.
It leaves before the 3-D stage: `uu_b` has 4,076 unequal faces, max
`1.7867651802561113e-16`; `vv_b` has 4,326, max
`1.734723475976807e-16`.  Both are AT-BAR.  A one-cell plant makes the U row
DEBT with max/normalized error `1.0`.  The subsequent stage walk first becomes
DEBT at stage 2 (U `6.176e-15`, V `6.106e-15`) and exits stage 3 with V
`2.984e-15` on 3,256 cells.  The next owner is therefore registered as
`kt2_barotropic_continuation_with_ice_drag`; this round makes no unsupported
source-statement attribution.

## Decision 122: ORCA2_OMIP_L4

The same enclosing subroutines and identical line numbers execute in the full
ORCA2 build:

| statement | ORCA2_OMIP_L4 compiled source |
|---|---|
| `rCdU_top=rCdU_ice` | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:178-190` (`zdf_drg`, :150) |
| SI3 formation | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceupdate.f90:403` (`ice_update_tau`, :332) |
| external coefficient | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1428-1433` (`dyn_drg_init`, :1404) |
| top baroclinic residual | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:1466-1489` (`dyn_drg_init`) |
| external substep drag | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:633-640` (`dyn_spg_ts`, :136) |
| 3-D top RHS and U/V diagonal | `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynzdf.f90:158-180`, `:303-314`, `:470-482` (`dyn_zdf`, :83) |

ORCA2 resolves the same drag flags/defaults at
`ORCA2_OMIP_L4/EXP00/namelist_ref:817-824` and the same SI3 coefficient at
`ORCA2_OMIP_L4/EXP00/namelist_ice_ref:136`.  This receipt supplies the D122
source pointer; it does not claim a full-ORCA2 trajectory measurement.

## Evidence, tests, and review

Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round33`.

| artifact | SHA-256 |
|---|---|
| `g5_forcing_identity.json` | `451e47fb3951243725e1b8dda9eefb7177f3321b893f5c6767d20fa065b24d46` |
| `g5_top_drag.json` | `88020e0da9347db3499c201bf99dd6fea48fb0169d8530ee365b9a4f8da603b2` |
| `g5_top_drag_plant.log` | `a8af02d1b3890973723026f057928c5d51e8d5348008adc7d85f9490b73786b6` |
| `g5_stages_kt1_assoc.json` | `4e8d75af76396d45b6dc4328e163f522173d291252363dbd1e8a9c3a1d69b836` |
| `g5_stages_kt2_independent_assoc.json` | `4fcc06479fd6a0789ea367e77ad348fa10c752a7d941e4ae2002512521a8f313` |
| `g5_independent_assoc_kt1_10.json` | `01e8b8ba75d1ad4829874fc6acfe6d5ebc5cf89e9d1af7caec84aedeb87b1cba` |
| `g5_given_assoc_kt1_10.json` | `412b1eed535b496d3cbae75c9f465f5e11e0bb8695f67b8c9dcdc3dfbcfaaf6c` |
| `g5_barotropic_exit_kt2.json` | `722ed140d8fdc2a7cc9cfc740001d82ae454c3ca45bb4ccef7eb9016cedd0636` |
| `g5_barotropic_exit_kt2_plant.json` | `8503913c15ca93a3124ee81431a91cf5c624ad55d13d40c5ef367b586af7e38d` |

- Focused G5 + G2 ladder, CPU fp64: 22 passed, 7 slow deselected.
- G2/G3/G4 fast regression, CPU fp64: 56 passed, 11 slow deselected.
- Top-drag coefficient plant: exit 1.  Barotropic-exit plant: U row DEBT.
- Citation gate after the mechanical 87-line reanchor: 17 passed.
- independent review: operator (Claude)

ASKED: the recorded input; all four source-ordered consumers; G5-only default
off selection and missing-input refusal; kt-1 staged score; both labels through
kt 10; compiled-order continuation; Decision-122 source pointers; focused
tests; receipt.  UNASKED: no NEMO run, no full-ORCA2 measurement.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r33-1163310947
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: HELD — top drag lands and kt 1 is exact, but kt 2 V exceeds the bar inside the continuing external mode.
