# SI3 round 11 receipt — O7 model preparation: the ice-atmosphere surface step

## Own errors, first

1. My first transcription let XLA turn `zta/rtt0` into a multiply by the
   reciprocal: 93 one-ULP rows in the latent-heat chain against the
   independent oracle. The divisor is now materialized
   (`packages/ice/legoesm/ice/si3_orca2_surface.py:290`).
2. My first Fortran `SUM` seed `0 + x` was folded away by XLA, so an
   open-ocean `emp_ice` kept a `-0`. The seed is now materialized
   (`packages/ice/legoesm/ice/si3_orca2_surface.py:272`).
3. My first subnormal-snow guard read the bit pattern; it works eagerly but
   XLA `jit` still flushes the operand. Deleted; the case is under the D5
   platform floor. The preregistration's sentence claiming it is retracted in
   its addendum.

## Outcome

O7's four routines are in the one shared ice package as a selectable SI3
path. Bit-identical to an independent per-cell glibc transcription under
production `jit`. No NEMO array was read for O7; the acquisition is
preregistered (`docs/ice/fidelity/si3_r11_o7_preregister.md`), not built.

## Build cited

`ORCA2_SAS_ICE_O2_REF`. Its `sbcblk.f90`, `sbc_phy.f90`, `icealb.f90`,
`icesbc.f90` and `icevar.f90` are byte-identical (`cmp`) to `_O1_REF`,
`_O1_REC` and `_O2_REC`. Flags, `ORCA2_SAS_ICE_O2_REF/BLD/arch_nemo.fcm:27`:
`%FCFLAGS             -fdefault-real-8 -O3 -funroll-all-loops -fcray-pointer -ffree-line-length-none -fallow-argument-mismatch -fno-tree-vectorize`.
So every bare literal (`0.05`, `0.18`, `0.35`, `0.02`) is binary64, and no
FMA is emitted.

## Deck values, from the resolved namelist

The admitted O2 run's `output.namelist.dyn`, committed byte-for-byte
(SHA-256 `27a4ebeb6196546e1dae94fc09317ef38cf2712b1e4c3ddf31f4892cc5c4b16f`),
`tests/ice/unit/data/si3_o2_ref_output.namelist.dyn:243-245`: first line
`RN_CD_IA=  1.4000000000000000E-003,`, last line
`RN_CH_IA=  1.4000000000000000E-003,`. No coefficient is hard-coded; the
config has no defaults and is built only by
`packages/ice/legoesm/ice/si3_orca2_surface.py:107`
(`class SI3O7SurfaceConfig(NamedTuple):`). It applies NEMO's post-echo pond
rule, `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icethd_pnd.f90:1481`:
`IF( ln_pnd_alb  ) THEN ; ln_pnd_alb  = .FALSE. ; CALL ctl_warn( 'ln_pnd_alb=false when no ponds' )  ; ENDIF`.

## Search before build (RULE 4)

- `git grep "blk_ice\|_nemo_si3_\|ice_flx_other\|snwfra"`: lane 3's private
  C1D identity `_nemo_si3_ice_albedo`, `_nemo_si3_blk_ice_2`,
  `_nemo_si3_ice_flx_other` (`packages/ice/legoesm/ice/sea_ice.py:664`,
  `:726`, `:877`), `nemo_si3_constant_fluxes` and
  `nemo_si3_saturation_over_ice`. One-column, no pond branch, ORCA1
  constants, no source-order barriers, native transcendentals. Not
  extendable without changing C1D bits; not touched (finding below).
- Reused: `si3_var_snwfra_3d`, `si3_lbc_lnk_orca2`, `SI3ORCA2FoldProgram`,
  `NEMO_SI3_CONSTANTS_CONFIG`, `nemo_source_round`, the shared
  transcendentals module.
- Added to the shared transcendentals module: a glibc `pow` arm
  (`packages/core/legoesm/core/transcendentals.py:151`, `def pow(base, exponent) -> jax.Array:  # noqa: A001 - mirrors the C name`).
  Nothing else there changed.

## What was transcribed

`blk_ice_1` (constant arm) and the stress exchange:
- `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1082`: `wndm_ice(ji,jj) = SQRT( pwndi(ji,jj) * pwndi(ji,jj) + pwndj(ji,jj) * pwndj(ji,jj) )`
- `:1093`: `Cd_ice(:,:) = rn_Cd_ia` (and `Ch_ice`, `Ce_ice`, `theta_zu_i`, `q_zu_i` on the following lines)
- `:1137`: `zztmp         = rhoa(ji,jj) * Cd_ice(ji,jj) * wndm_ice(ji,jj)`; `:1138`: `putaui(ji,jj) = zztmp * pwndi(ji,jj)`
- `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:167`: `CALL lbc_lnk( 'icesbc', utau_ice, 'T', -1.0_wp, vtau_ice, 'T', -1.0_wp )`

`ice_alb`, pond branch:
- `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icealb.f90:184`: `z1_c1 = 1._wp / ( LOG(rn_alb_hpiv) - LOG(0.05_wp) )`
- `:201`: `zafrac_pnd = MIN( pafrac_pnd(ji,jj,jl), 1._wp - zafrac_snw ) ! make sure (a_ip_eff + a_s_fra) <= 1`
- `:218`: `IF( 0.05 < ph_ice(ji,jj,jl) .AND. ph_ice(ji,jj,jl) <= rn_alb_hpiv ) THEN      ! 5cm < hi < 100cm`
- `:231`: `IF( ph_lid(ji,jj,jl) <= 0.02 ) THEN ! do not change pond albedo if the lid is < 2cm`
- `:244`: `palb_ice(ji,jj,jl) = ( 1._wp - pcloud_fra(ji,jj) ) * zalb_cs + pcloud_fra(ji,jj) * zalb_os`
- Snow fraction: `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1597`: `pa_s_fra(ji,jj,jl) = ph_s(ji,jj,jl) / ( ph_s(ji,jj,jl) + 0.02_wp )`

`blk_ice_2` and its saturation functions:
- `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbcblk.f90:1231`: `qsr_ice(ji,jj,jl) = zztmp * ( 1. - palb(ji,jj,jl) ) * qsr(ji,jj)`
- `:1255`: `qla_ice (ji,jj,jl) = zztmp1 * (zsq - q_zu_i(ji,jj))`
- `:1290`: `emp_ice(ji,jj) = SUM( a_i_b(ji,jj,:) * evap_ice(ji,jj,:) ) - sprecip(ji,jj) * zsnw(ji,jj)`
- `:1320`: `ztri(:,:) = 0.18 * ( 1.0 - cloud_fra(:,:) ) + 0.35 * cloud_fra(:,:)  ! surface transmission when hi>10cm`
- `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1636`: `pout = ( 1._wp - ( pin )**rn_snwblow )`
- `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/sbc_phy.f90:692`: `zle  = rAg_i*(ztmp - 1._wp) + rBg_i*LOG10(ztmp) + rCg_i*(1._wp - zta/rtt0) + rDg_i`
- `:694`: `e_sat_ice_sclr = 100._wp * 10._wp**zle`
- `:724`: `zde = -(rAg_i*rtt0)/(zta*zta) - rBg_i/(zta*LOG(10._wp)) - rCg_i/rtt0`
- `:764`: `q_sat_sclr = reps0*ze_s/(ppa - (1._wp - reps0)*ze_s)`
- `:803`: `dq_sat_dt_ice_sclr = reps0*ppa*zde_s_dt / ( ztmp*ztmp )`
- `:350`: `theta_exner_sclr = pta * ( rpref / ppa ) ** rgamma_dry`
- Compile-time folds `LOG(0.05_wp)`, `LOG(10._wp)` and `:90` `&      rDg_i = LOG10(6.1071_wp)` are frozen as hex constants, checked equal to correctly rounded values (mpmath) and to glibc.

`ice_flx_other`, both dynamics arms:
- `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icesbc.f90:396`: `zfric(ji,jj) = zdrag * ( 0.5_wp * (  ( zu_io*zu_io + zu_iom1*zu_iom1 )   &   ! add () for NP repro`
- `:403`: `zfric(ji,jj) = r1_rho0 * SQRT( utau(ji,jj)*utau(ji,jj) + vtau(ji,jj)*vtau(ji,jj) ) * smask0(ji,jj)`
- `:436`: `qsb_ice_bot(ji,jj) = zswitch * MIN( qsb_ice_bot(ji,jj), - zqfr_neg * r1_Dt_ice / MAX( at_i(ji,jj), epsi10 ) )`
- `:450`: `IF( ( zqld - zqfr ) < 0._wp ) THEN`
- `:472`: `IF( zvel(ji,jj) <= 5.e-03_wp .AND. at_i(ji,jj) >= (rn_amax_2d(ji,jj)-0.001_wp) .AND. vt_i(ji,jj) >= 3._wp )   qlead(ji,jj) = 0._wp`
- Time step: `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icestp.f90:418`: `rDt_ice   = REAL(nn_fsbc) * rn_Dt          !--- sea-ice timestep and its inverse`

Refused, not transcribed: non-constant coefficients, form drag, `ln_mixcpl`,
`nn_flxdist /= -1`, `ln_cndflx`, `ln_pnd_alb = F`, `nn_qtrice = 1`,
`ln_leadhfx/ln_icedO/ln_icedH = F`. The constant arm's unused `zsipt` and
the `iom_put` diagnostics are not transcribed.

## Exact-input structure

`si3_o7_blk_ice_1`, `si3_o7_ice_sbc_tau_lbc`, `si3_o7_ice_alb`,
`si3_o7_blk_ice_2`, `si3_o7_ice_flx_other`, each callable alone with NEMO's
arrays at its call, and `si3_o7_surface_step`
(`packages/ice/legoesm/ice/si3_orca2_surface.py:591`), which chains them in
NEMO's order. Arrays are NEMO's interior, except the four velocity fields
`ice_flx_other` reads at west/south neighbours. `None` raises.

## Tests (`tests/ice/unit/test_si3_o7_surface.py`)

Result: `14 passed, 1 xfailed in 74.33s` (`si3_rounds/round11/o7_unit.log`).
- Bit-for-bit against an independent per-cell transcription written from NEMO's literals with glibc `math` (`tests/ice/unit/test_si3_o7_surface.py:330`), with forced branch coverage: land, thin/break/mid/pivot/thick ice, melting/freezing, lid threshold, open-ocean `-0` sum, supercooling, warm and cooling leads, landfast and full-cover stops shown to bind.
- Inexact-value pins checked against the oracle (`:346`).
- Plants that fire (`:364`): `Cd_ia` +1 ULP, `rtt0` +1 ULP, `LOG(10)` +1 ULP, `SUM` seeded from the first category, `nn_snwfra` swapped. Reverting each barrier fix makes the oracle test fail (`plant_divisor_reciprocal.log`: 93 rows; `plant_sum_seed_fold.log`: `emp_ice` at the open-ocean cell).
- Eager under `jax.disable_jit` equals `jit` on all 33 outputs (`:379`).
- No-dynamics arm, cyclic seam, closed south, fold sign binding (`:391`).
- Gradients finite everywhere; central difference on one `t_su` cell within 1e-5 (`:405`).
- No defaults and refused arms (`:423`).
- Platform floor pinned as strict xfail (`:441`).
- The new `pow`: equal to glibc, finite zero subgradient at 0 (`:320`); existing transcendentals suite `18 passed in 2.19s`.

## Controls for every other caller (parent `f76638f1a7`)

- D5 C1D year under CPU `jit`: all 237,111 non-provenance values identical parent vs current; shipped-arm hash `b7a7329bed74c17918fa3cbc11c7613acb2074851e2357a08869fcd4661e12d2` (`d5_parent_cpu.json`, `d5_current_cpu.json`).
- Lane-3 bulk-flux gate, O3 fit and landfast suites (transcendentals consumers): `27 passed` on current; parent result in `shared_callers_parent_vs_current.txt`.
- ORCA2 ocean card: not touched. Lane-3 cards: no touched file on their path except the additive `pow`.

## Gates

- Citation gate, all SI3 receipts, full map: see `citation_gate_all_si3.log`.
- Ratchets: the same 11 pre-existing reds as round 9 recorded, none in a touched file; the new module and test pass the inline-coefficient, hard-coded-constant and private-import ratchets (`ratchets_current.txt`).
- Review: `codex review: operator`.
- GPU spot check: not run (no certification this round).

## Findings (not acted on)

- Lane 3's private C1D bulk/albedo functions are a second, un-barriered
  implementation of the same statements. Moving C1D onto the O7 body would
  change C1D bits; that is a user decision.
- The shared `fortran_sum_categories` starts from the first term, not `+0`;
  it differs from gfortran only in the sign of an all-`-0` sum.

## CONFIRMED vs PLAUSIBLE

- CONFIRMED: bit-identity to the independent glibc transcription under `jit`; plants fire; eager == `jit`; gradients finite and FD-consistent; D5 C1D year unchanged.
- PLAUSIBLE: bit-exactness against NEMO (no record; P1-P6); gfortran's `MIN`/`MAX` keep the first argument on a tie (sign of zero on land cells, P5).

## CHOICES

- O7 acquisition deck and window. ASKED (DECISION_NEEDED).
- Reference build `ORCA2_SAS_ICE_O2_REF` (byte-identical alternatives). UNASKED, mechanical.
- Upstream fields (`theta_air_zt`, `q_air_zt`, `rhoa`, `precip`, `emp`, `qsr`, `qsr_oce`, `qns_oce`, `cloud_fra`, `rn_amax_2d`, `t_bo`) enter as recorded arrays. UNASKED; follows the round's exact-input instruction.
- Both `ice_flx_other` dynamics arms transcribed. UNASKED; offered for revert.
- Lane 3's C1D functions left untouched (no migration). UNASKED; offered as a decision.
- New glibc `pow` in the shared transcendentals module (DECISION 162 names `pow`). UNASKED, additive.
- Fixture: the admitted O2 run's resolved namelists committed under `tests/ice/unit/data/`. UNASKED; offered for revert.
- No configuration default, production config or other caller changed.
