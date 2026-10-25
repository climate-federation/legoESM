# SI3 Ladder Rung O7 Preregistration — Round 11 (`si3_r11`): the ice-atmosphere surface step

- Date: 2026-10-25. Lane `si3`, round 11, rung O7 (`docs/ice/fidelity/si3_ladder_design.md:227`).
- Model parent: `f76638f1a7ff` (O1, O2, O3, O5-prep, D5 landed).
- Status: committed BEFORE any O7 measurement. No NEMO array has been read for O7. The acquisition tool is NOT built this round.
- Reference build cited: `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/` (unpatched O2 reference). Its `sbcblk.f90`, `sbc_phy.f90`, `icealb.f90`, `icesbc.f90`, `icevar.f90` are byte-identical (`cmp`) to `ORCA2_SAS_ICE_O1_REF`, `_O1_REC` and `_O2_REC`. Build flags `-fdefault-real-8 -O3 -fno-tree-vectorize`, no `-march` (so every bare literal such as `0.05` is binary64 and no FMA is emitted).

## 1. Deck — PENDING USER DECISION (section 7)

Pick: the **unmodified production** `ORCA2_SAS_ICE` deck (`ln_icedyn = T`,
`ln_icethd = T`), one MPI rank (`jpni = 1`) as O1/O2/O5, `nn_it000 = 1`,
`nn_itend = 2`. Recording happens at the call points of section 2; every
rung input is the exact NEMO array at its call, so dynamics/thermodynamics
being on does not compose into the score. Alternative: the O2 diagnostic
overlay (`ln_icedyn = F`, `ln_icethd = F`), which executes the no-dynamics
arm of `ice_flx_other` (`icesbc.f90:401-406`) instead of the production arm
(`icesbc.f90:385-400`). Both arms are transcribed.

Every scalar is read from that run's resolved `output.namelist.dyn` /
`output.namelist.ice` (never the deck text): `rn_Cd_ia = rn_Ce_ia = rn_Ch_ia`
(expected `1.4000000000000000E-003`), `rn_pfac`, `rn_efac`, `rn_snwblow`,
`nn_snwfra`, the seven `rn_alb_*`, `ln_pnd`, `ln_pnd_alb`, `nn_qtrice`,
`nn_flxdist`, `ln_cndflx`, `ln_Cx_ice_*`, `ln_blk`, `ln_mixcpl`,
`ln_icedyn`, `ln_leadhfx`, `ln_icedO`, `ln_icedH`, `rn_Cd_io`, `nn_fsbc`,
`rn_Dt`, `nn_hls`.

## 2. Arrays and call points (labels: GIVEN-NEMO-ENTRY only)

Binary64, every array as NEMO holds it at the call (`A2D(0)` interior or
full halo allocation, recorded with its bounds).

- S1 `blk_ice_1` entry, `icesbc.f90:144`: `sf(jp_wndi)%fnow`, `sf(jp_wndj)%fnow`, `theta_air_zt`, `q_air_zt`, `sf(jp_slp)%fnow`, `tm_su`, `rhoa`.
- S2 `blk_ice_1` exit, before `icesbc.f90:167`: `utau_ice`, `vtau_ice` (interior), `wndm_ice`, `Cd_ice`, `Ch_ice`, `Ce_ice`, `theta_zu_i`, `q_zu_i`.
- S3 after `icesbc.f90:167` (`lbc_lnk 'T' -1`): `utau_ice`, `vtau_ice` full halo.
- S4 `ice_alb` entry `icesbc.f90:209`: `t_su`, `h_i`, `h_s`, `a_ip_eff`, `h_ip`, `h_il`, `cloud_fra`, `smask0`; exit: `alb_ice`.
- S5 `blk_ice_2` entry `icesbc.f90:221`: `t_su`, `h_s`, `h_i`, `alb_ice`, `theta_air_zt`, `q_air_zt`, `sf(jp_slp)%fnow`, `sf(jp_qlw)%fnow`, `precip`, `sf(jp_snow)%fnow`, `qsr`, `rhoa`, `wndm_ice`, `Ch_ice`, `Ce_ice`, `theta_zu_i`, `q_zu_i`, `sst_m`, `emp`, `at_i_b`, `a_i_b`, `qns_oce`, `qsr_oce`; exit after `:224`: `qsr_ice`, `qns_ice`, `dqns_ice`, `qla_ice`, `dqla_ice`, `evap_ice`, `devap_ice`, `tprecip`, `sprecip`, `emp_oce`, `emp_ice`, `emp_tot`, `qemp_oce`, `qemp_ice`, `qns_tot`, `qsr_tot`, `qprec_ice`, `qevap_ice`, `qtr_ice_top`.
- S6 `ice_flx_other` entry `icesbc.f90:254`: `u_ice`, `v_ice`, `ssu_m`, `ssv_m` (full halo), `utau`, `vtau`, `at_i`, `at_i_b`, `vt_i`, `qsr_oce`, `qns_oce`, `qemp_oce`, `frq_m`, `e3t_m`, `sst_m`, `t_bo`, `rn_amax_2d`; exit: `qsb_ice_bot`, `fhld`, `qlead`, `frq_m`.
- S7 run scalars: `kt`, `nit000`, `jpni`, `rDt_ice`, `r1_Dt_ice`, `rho0`, `rcp`.

Both steps `kt = 1, 2`. Estimated size well under 1 GiB (the design row's
10–20 GiB is an upper bound).

## 3. Frozen predictions (CPU fp64, production `jax.jit`, `transcendentals="libm"`)

- P1: S2 replays from S1 exactly (0 ULP, every interior cell) through `si3_o7_blk_ice_1`; `Cd_ice = Ce_ice = Ch_ice` equal the bit pattern of the resolved `rn_Cd_ia` everywhere.
- P2: S3 replays from S2 exactly through `si3_o7_ice_sbc_tau_lbc` with fold program `nfd_full_when_absent = True` (one rank).
- P3: S4 `alb_ice` replays exactly through `si3_o7_ice_alb`.
- P4: every S5 exit array replays exactly through `si3_o7_blk_ice_2` from S5 entry.
- P5: every S6 exit array replays exactly through `si3_o7_ice_flx_other`, including the sign of zero on land cells (`MIN`/`MAX` written as gfortran's keep-first-argument-on-tie select; PLAUSIBLE until this record).
- P6: the composed `si3_o7_surface_step` from S1 + S4 + S5/S6 raw inputs reproduces S2-S6 exits exactly (internal products feed forward without replay).
- Falsifier for each: any nonzero-ULP interior cell. A binary64 subnormal input or intermediate that XLA CPU flushes is counted under the D5 platform floor (DECISION 158), never called bit-identical; the `h_s <= 0` transmission predicate is evaluated on the IEEE bit pattern.
- The first non-bit statement, if any, is named with its `ppsrc` line before any change lands.

## 4. Controls the gate must carry

Swapped-field and one-bit payload plants on the reader; plants that must
turn the score red: `Cd_ia` off by one ULP; wrong stress fold sign; pond
branch off; `nn_snwfra` swapped; native `exp` instead of glibc; SUM started
from the first category instead of `+0`.

## 5. Not exercised by the picked deck (stated, not claimed)

The non-production arms are refused, not transcribed: `ln_Cx_ice_cst = F`,
form drag, `ln_abl`, `ln_mixcpl`, `nn_flxdist /= -1`, `ln_cndflx`,
`ln_pnd_alb = F`, `nn_qtrice = 1`, `ln_leadhfx/ln_icedO/ln_icedH = F`.
`blk_ice_1` also computes `zsipt = theta_exner(ptsui, pslp)`
(`sbcblk.f90:1086`), which the constant-coefficient arm never reads; it is
not transcribed. The `iom_put`/`prt_ctl` diagnostics are not transcribed.

## 6. CHOICES (ASKED/UNASKED)

- Deck, rank count and step window: section 1. ASKED; answered by DECISION 165 (Addendum 3).
- Reference build `ORCA2_SAS_ICE_O2_REF` (byte-identical alternatives). UNASKED, mechanical.
- `theta_air_zt`, `q_air_zt`, `rhoa`, `precip`, `emp`, `qsr`, `qsr_oce`, `qns_oce`, `cloud_fra`, `rn_amax_2d`, `t_bo` enter as recorded arrays (they are produced by `sbc_blk`/`blk_oce_1/2`/`ice_stp`, outside the four O7 routines). UNASKED; follows the round's exact-input instruction.
- Both `ice_flx_other` dynamics arms transcribed so either deck answer is covered. UNASKED; offered for revert.

## 7. Pending decision

O7 acquisition deck: (1) unmodified production `ORCA2_SAS_ICE`, 1 rank,
`kt = 1..2` (pick: it runs every production arm the model now carries);
(2) O2 diagnostic overlay `ln_icedyn = ln_icethd = F`, same window.

## Addendum (after the unit tests; predictions P1-P6 unchanged)

- Retracted: section 3's "the `h_s <= 0` transmission predicate is evaluated
  on the IEEE bit pattern". Under XLA CPU `jit` the subnormal operand is
  flushed even through a bitcast, so that case is under the D5 platform floor
  (pinned by a strict xfail), not claimed.
- Section 4's "native `exp` instead of glibc" plant did not fire on the
  synthetic unit fixture (native and glibc agreed there); it stays a required
  plant on the real record, where it must be shown to fire or be reported
  vacuous.

## Addendum 2 (after the codex HOLD; predictions P1-P6 unchanged)

- **Deck: only the deck the user chooses will be acquired.** Section 1's
  "Pick" is a recommendation awaiting the user, not a decision. Both
  `ice_flx_other` arms exist in the model; that does not choose the deck.
  The O7 acquisition tool is not written until the user answers section 7.
- The functions named in section 3 are renamed, same statements:
  `si3_o7_blk_ice_1` -> `si3_blk_ice_1`, `si3_o7_ice_sbc_tau_lbc` ->
  `si3_ice_sbc_tau_lbc`, `si3_o7_ice_alb` -> `si3_ice_alb`,
  `si3_o7_blk_ice_2` -> `si3_blk_ice_2`, `si3_o7_ice_flx_other` ->
  `si3_ice_flx_other`, `si3_o7_surface_step` -> `si3_surface_step`
  (module `legoesm.ice.si3_surface`). C1D and O7 now call the same code.
- Retracted: section 5's "`zsipt` ... is not transcribed". It is now
  transcribed. It is a local of `blk_ice_1`, so the record cannot score it;
  the model test pins it against the oracle and shows nothing on this arm
  reads it.
- The production configuration comes from the deck's shipped namelist
  files (`cfgs/SHARED/namelist_ref`, `namelist_ice_ref`,
  `cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg`, `namelist_ice_cfg`), merged as
  NEMO reads them. If the production deck is chosen, the run's own
  `output.namelist.dyn/.ice` must equal it field for field before scoring.

## Addendum 3 (user decision; predictions P1-P6 unchanged)

- **DECISION 165 (2026-10-25), ASKED:** the O7 acquisition uses the
  unmodified production `ORCA2_SAS_ICE` deck, one MPI rank, `kt = 1..2`
  (section 7 option 1). The O2 diagnostic overlay is not acquired.
