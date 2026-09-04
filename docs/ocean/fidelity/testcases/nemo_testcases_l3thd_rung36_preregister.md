# SI3 lane 3b rung 3.6 preregistration — coupled one-layer column

Date: 2026-09-04  
Tracker: `climate-federation/legoESM#1699`  
State: **DESIGN ONLY — UNMEASURED; independent review required before code**

This document fixes the oracle, interfaces, time levels, coverage register,
gate, hypotheses, and controls for rung 3.6.  It does not implement the rung
and makes no fidelity or trajectory claim.  `CONFIRMED` below means confirmed
from executing-path source/configuration, not confirmed numerical agreement.

## Oracle choice

**Choice: a config-local, one-wet-layer OCE+ICE variant of
`C1D_OMIP_L3`, provisionally named `C1D_OMIP_L3_COUPLED`.**  It replaces the
SAS component with the prognostic ocean, keeps the official C1D ERA5 member,
keeps the phase-1 ORCA1 SI3 thermodynamic deck, runs the ocean at 3600 s, and
runs SI3 every fourth ocean step (`nn_fsbc=4`, hence `rDt_ice=14400 s`).  The
planned component list is `OCE ICE`; the planned preprocessor set is
`key_si3 key_vco_1d3d key_RK3` with no `key_linssh`, so both the RK3 tracer
surface boundary and prognostic SSH are observable.  The work configuration
will be copied; the shipped NEMO tree will remain read-only.

The one-layer geometry is a config-local copy of C1D's `usrdef_nam.F90` with
only its vertical-size assignment changed from `kpk=75` to `kpk=2` (one wet
level plus NEMO's terminal level).  The source currently chooses 75 for OCE and
2 for SAS at `cfgs/C1D/MY_SRC/usrdef_nam.F90:73-80`; this one-line change is a
declared geometry composition, not an SI3 physics overlay.  Initial `sst`,
`sss`, `ssu`, `ssv`, and `ssh` will be extracted without numerical alteration
from the first record of the same official ERA5 member that C1D SAS reads at
`cfgs/C1D/EXP_SASICE/namelist_cfg:153-157`; the derived input and extraction
command must be hashed in the future receipt.

Why this oracle, rather than ICE_ADV2D purpose 2:

- `tests/README.rst:178-186` documents ICE_ADV2D as an ice-advection test.
  Its shipped ice deck sets `ln_icethd=.false.` and selects prescribed 2-D
  advection (`tests/ICE_ADV2D/EXPREF/namelist_ice_cfg:24-37`).
- Its surface deck is `ln_usr=.true., nn_fsbc=1`
  (`tests/ICE_ADV2D/EXPREF/namelist_cfg:72-78`), while its
  `usrdef_sbc.F90` supplies a fixed ice stress and zero heat/mass inputs.  The
  proposed purpose-2 case would therefore replace the defining surface forcing,
  ice cadence, and enabled ice physics.
- The C1D choice retains the measured real-forcing bulk/thermodynamic column and
  changes only the coupling partner and slab geometry.  It therefore exposes
  the first uncertified boundary without introducing analytic forcing.

This choice is **CONFIRMED as the selected design**, but all behavior of the
new composition is **UNMEASURED** until the reviewed implementation rung.

## Resolved configuration and provenance contract

Every future overlay is one row and names its source.  No independently valid
selector may be mixed with it.

| row | selected value | sole source / reason |
|---|---:|---|
| components | `OCE ICE` | this design: replace SAS by the prognostic ocean |
| `rn_Dt` | `3600 s` | C1D EXP_SASICE `namelist_cfg:38`; ORCA1 `namelist_cfg:58` agrees |
| `nn_fsbc` | `4` | ORCA1 `EXPREF/namelist_cfg:106`; means four hourly ocean samples and calls SI3 at `MOD(kt-1,4)=0` |
| `nn_ice` | `2` | ORCA1 `namelist_cfg:111` |
| `ln_ice_embd` | `.false.` | ORCA1 `namelist_cfg:115`; levitating identity |
| `ln_blk` / ocean bulk | `.true.` / NCAR | ORCA1 `namelist_cfg:109,132` |
| ice-air transfer | constant; `Cd=Ce=Ch=1e-3` | ORCA1 `namelist_cfg:139-142` |
| `nn_fwb` | `1` | ORCA1 `namelist_cfg:122-125` |
| `nn_fwb_voltype` | `1` | shipped `cfgs/SHARED/namelist_ref:666`; ice+ocean volume |
| EOS | TEOS-10 | ORCA1 `namelist_cfg:307-308` |
| drag | `ln_drgimp=T`, `ln_drgice_imp=T` | ORCA1 `namelist_cfg:258-261` |
| SI3 thermodynamics | single-category HFN, BL99 3+3/P07, option-2 salinity with `rn_sinew=0.75`, `ln_pnd=F`, `ln_icedA=F` | byte-for-byte phase-1 resolved ice deck; its per-row ORCA1 citations remain authoritative |
| ice dynamics | `ln_icedyn=T`, dynamically skipped | ORCA1 ice deck; C1D's `ln_c1d=T` makes the skip explicit at `src/ICE/icestp.F90:167-168` |

The official archive remains
`/data/abyssal/dbalwada/nemo-inputs/C1D_v5.0.0/C1D_v5.0.0.tar.gz`
(MD5 `9456e6a0a84d40630ad1804fd4061caf`, SHA-256
`54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382`).
The atmospheric member is
`ERA5_NorthGreenland_surface_84N_-36E_1h_y2018.nc`; its existing phase-1
per-file SHA-256 must be rechecked, not copied by assertion, in the rung-3.6
receipt.  No synthetic forcing is permitted.

## Source-defined call graph

The following facts are **CONFIRMED** from NEMO 5.0.2 source.

1. `sbc()` calls `sbc_ssm` before the bulk formulation
   (`src/OCE/SBC/sbcmod.F90:395-424`), calls SI3 through `ice_stp` at
   `:474-480`, and calls `sbc_fwb` after ice/runoff/restoring at `:494-498`.
2. With `nn_fsbc=4`, `sbc_ssm` initializes the first accumulator from three
   copies of the first instantaneous field (`src/OCE/SBC/sbcssm.F90:99-126`),
   accumulates the fourth copy and every later hourly sample at `:139-161`, and
   divides at `MOD(kt-1,4)=0` at `:164-173`.  The means are `ssu_m`, `ssv_m`,
   `sst_m`, `sss_m`, `ssh_m`, `e3t_m`, and `frq_m`.
3. `ice_stp` executes only at `MOD(kt-1,nn_fsbc)=0`, copies mean current and
   layer thickness, and computes the zero-pressure TEOS-10 freezing point from
   mean SSS (`src/ICE/icestp.F90:126-138`).  `eos_fzp` evaluates the Horner
   polynomial at `src/OCE/TRA/eosbn2.F90:1672-1683`; the omitted `pdep` arm at
   `:1685-1689` is inactive here.
4. After the already scoped thermodynamic chain, `ice_update_flx` is called at
   `src/ICE/icestp.F90:201-213`.  It constructs transmitted solar heat
   (`src/ICE/iceupdate.F90:128-158`), residual non-solar heat at `:160-161`,
   freshwater at `:167-178`, salt at `:180-183`, and ice/snow mass history at
   `:187-194`.
5. `ice_update_tau` is outside the ice-step guard and therefore executes every
   ocean step (`src/ICE/icestp.F90:228-234`).  It refreshes `tmod_io` only on
   ice steps (`src/ICE/iceupdate.F90:361-385`), then with
   `ln_drgice_imp=T` writes negative `rCdU_ice` at `:387-396` and blends the
   explicit surface stress at `:398-407` using instantaneous `uu/vv(:,:,1,Kbb)`.
6. RK3 calls `tra_sbc_RK3` at every stage
   (`src/OCE/stprk3_stg.F90:508-522`).  It writes mass-associated T/S terms on
   stages 1 and 2 and the heat/salt boundary terms on stage 3
   (`src/OCE/TRA/trasbc.F90:278-315`).  The nonlinear-SSH stage-3 additions are
   `qns/(rho0*rcp*e3t)` and `sfx/(rho0*e3t)` at `:308-313`; penetrative `qsr`
   remains owned by `tra_qsr` (`src/OCE/stprk3_stg.F90:585`).
7. Positive NEMO `emp` removes ocean mass: the RK3 SSH forcing is
   `emp/rho0` (`src/OCE/DYN/dynspg_ts.F90:405-422`), and the direct SSH update
   subtracts it (`src/OCE/DYN/sshwzv.F90:117-129`).
8. With `nn_fwb=1`, `sbc_fwb` corrects only on the surface cadence.  Under
   `nn_fwb_voltype=1` it subtracts `snwice_fmass` from the budget before the
   area mean (`src/OCE/SBC/sbcfwb.F90:224-239`).  The selector's meaning is
   printed at `:143-153`.

## legoESM reuse and exact exchange card

The future implementation must extend the **existing**
`packages/coupler/legoesm/coupler/ocean_forcing.py`; it must not add a second
exchange module.  Its current contract says:

- ice melt/freeze freshwater is positive into the ocean
  (`ocean_forcing.py:30-33`);
- ice basal heat extraction is positive out of the ocean, so existing
  `q_net` negates it (`:34-36`, executing assignment `:88`);
- its stored stress uses the atmosphere convention and is negated before the
  ocean consumer (`:37-41`, assignments `:89-90`);
- the common blend applies masks/weights at `:213-249` and freshwater assembly
  at `:251-286`.

Rung 3.6 will add a selectable `nemo_si3` card inside that machinery.  It will
accept NEMO-equivalent, already grid-cell-aggregated SI3 exchange fields and
will not apply a second concentration weight.  It will use the SI3-generated
`qsr` rather than the existing generic under-ice shortwave constant.  All
coefficients must come from `legoesm.constants` through the selectable NEMO
`ConstantsConfig`; no inline constants or monkey-patching are allowed.  The
entire card and ocean are fp64 via `set_policy(PrecisionPolicy.fp64())`, and the
harness must print every participating dtype and the CPU backend.

### Sign and unit map

| NEMO field | NEMO meaning and consumer | legoESM card value |
|---|---|---|
| `qsr`, `qns` | W m-2, positive into ocean; `qns` enters `tra_sbc_RK3`, `qsr` enters `tra_qsr` | `sw_down=qsr`; `q_net=qsr+qns` only at the combined-forcing interface (no second SW addition) |
| `emp` | kg m-2 s-1, positive ocean mass loss; positive lowers SSH | physical freshwater `F_fw=-emp`, positive into ocean |
| `sfx` | PSS kg m-2 s-1, positive salinity source in `tra_sbc_RK3` | real salt mass `salt_flux=1e-3*sfx` kg m-2 s-1; the existing ocean consumer converts kg salt back to salinity tendency |
| `utau`, `vtau` | Pa, positive force on the ocean in NEMO momentum | legoESM stored atmosphere-convention `tau_x=-utau`, `tau_y=-vtau`; its ocean sign flip restores the NEMO force |
| `rCdU_ice` | m s-1-like implicit top-drag coefficient, non-positive by `-tmod_io*at_i/rho0` | dedicated implicit top-drag input with the same sign; never folded into explicit `tau` |
| `snwice_fmass` | kg m-2 s-1, positive increase in ice+snow mass | budget-only term; subtract from NEMO-style `emp` before the global/one-cell FWB correction |

The separate `qsr` and `qns` paths prevent double application.  Likewise,
`F_fw` updates real mass/SSH once; a virtual-salt or KPP diagnostic must not
apply it a second time.  These are preregistered interface requirements, not
measured agreements.

## Frame and time-level registry (Rule 1d)

The config-local NEMO writer will be WRITE-only and will serialize interior
values explicitly; it may not mutate model arrays.  Every frame stores `kt`,
`Kbb`, `Kmm`, and, where applicable, `Krhs`, `Kaa`, and RK3 stage.  Runtime
indices, rather than inferred labels such as “now,” are authoritative.

| frame | cadence | registered time level and fields |
|---|---|---|
| `PRE_SSM` | every ocean step | `uu,vv(:,:,1,Kbb)`; `ts(:,:,1,jp_tem/jp_sal,Kmm)`; `ssh(:,:,Kmm)`; `e3t(:,:,1,Kmm)`; `fraqsr_1lev`; carried seven accumulators before `sbc_ssm` |
| `POST_SSM` | every ocean step | seven accumulators after add/divide; division flag `MOD(kt-1,4)==0` |
| `POST_FZP` | each ice step | `sss_m` at completed four-sample mean; `t_bo` after zero-pressure TEOS-10 conversion and Kelvin offset |
| `PRE_UPDATE_FLX` | each ice step | before-level `at_i_b,a_i_b`; current `at_i`; all heat/mass/salt operands enumerated below |
| `POST_UPDATE_FLX` | each ice step | `qsr,qns,emp,sfx,qt_atm_oi,qt_oce_ai,fwfice,snwice_mass_b,snwice_mass,snwice_fmass,fr_i,tn_ice,alb_ice` |
| `POST_UPDATE_TAU` | every ocean step | instantaneous `pu_oce,pv_oce=uu,vv(:,:,1,Kbb)`; carried `tmod_io` and ice-step refresh flag; `at_i,u_ice,v_ice,drag_io,utau_oce,vtau_oce,rCdU_ice,utau,vtau,taum` |
| `POST_FWB` | every ocean step | incoming `emp`, `snwice_fmass`, area/mask, `emp_ext`, `emp_corr`, outgoing corrected `emp`; active-cadence flag |
| `POST_TRA_SBC_RK3` | every RK3 stage | `Kbb,Kmm,Krhs,kstg`; before/after top-level T/S RHS; `emp,qns,sfx,e3t,rho0,rcp`, and top-level T/S at `Kbb` |
| `POST_TRA_QSR` | every RK3 stage where executed | `qsr`, penetration fraction/profile, and before/after top-level temperature RHS |
| `POST_SSH` | every ocean step | `Kbb,Kaa`; input SSH, `emp`, horizontal divergence, mask, output SSH, and isolated surface-mass increment |

The first `POST_SSM` mean is deliberately tested against NEMO's three seeded
copies plus the ordinary fourth accumulation; it is not assumed to be a
one-sample mean.  Restarts must cover the seven mean accumulators, the last
ice-step `tmod_io`, `utau_oce/vtau_oce`, `rCdU_ice`, `snwice_mass(_b)`, FWB
state (`emp_corr` and any active accumulator), and the ocean/ice prognostics.

## Coverage register seed

The future machine-readable gate must contain exactly one `VERIFIED` or
`WAIVED` disposition with a source-backed reason for every item below; missing
and duplicate names are fatal.

1. **Mean-ocean inputs:** `ssu_m`, `ssv_m`, `sst_m`, `sss_m`, `ssh_m`,
   `e3t_m`, `frq_m`; plus their instantaneous inputs and restart carries.
2. **All 36 established exchange-stream fields:** `alb_ice`, `devap_ice`,
   `dqla_ice`, `dqns_ice`, `emp`, `emp_ice`, `emp_oce`, `evap_ice`, `fr_i`,
   `qcn_ice`, `qemp_ice`, `qemp_oce`, `qevap_ice`, `qla_ice`, `qml_ice`,
   `qns`, `qns_ice`, `qns_oce`, `qprec_ice`, `qsr`, `qsr_ice`, `qsr_oce`,
   `qtr_ice_top`, `rCdU_ice`, `sfx`, `snwice_fmass`, `snwice_mass`,
   `snwice_mass_b`, `sstfrz`, `taum`, `tn_ice`, `utau`, `utau_ice`, `vtau`,
   `vtau_ice`, `wndm_ice`.
3. **`ice_update_flx` owner operands absent from that stream:** `at_i_b`,
   `a_i_b`, `qsr_tot`, `qns_tot`, `qtr_ice_bot`, `fhld`, every `hfx_*` in
   `iceupdate.F90:139-142`, every `wfx_*` in `:165-178`, and every `sfx_*` in
   `:182-183`.
4. **Stress owner fields:** `u_oce`, `v_oce`, instantaneous `pu_oce/pv_oce`,
   `u_ice`, `v_ice`, `drag_io`, `tmod_io`, `utau_oce`, `vtau_oce`, and
   `rCdU_ice`.
5. **Ocean consumers:** isolated increments and total before/after values for
   top-level T, S, SSH, U, and V; `qsr`, `qns`, `emp`, `sfx`, `utau`, `vtau`,
   and `rCdU_ice` at the exact consuming call.
6. **FWB:** `snwice_mass`, `snwice_mass_b`, `snwice_fmass`, input/output
   `emp`, `emp_ext`, `emp_corr`, wet area/mask, and `nn_fwb_voltype`.
7. **State/restart:** every prognostic in Appendix A of the lane-3 dossier,
   the seven `sbc_ssm` carries, stress carries, FWB carries, and the one-layer
   ocean T/S/SSH/U/V state.  Inactive pond, option-4 salinity, and dynamics
   moments may be WAIVED only with their resolved selector and call-site skip.

Interior pointwise values are the scientific gate.  Halos are separately
registered and either verified after their defining `lbc_lnk` or waived with a
consumer/time-level reason; padding bytes are never compared as science.

## Preregistered gates

All gates use fp64 on CPU.  The pointwise bar is

`abs(legoESM - NEMO) / max(abs(NEMO), 1) <= 1e-15`.

The oracle must run the full 2018 forcing year: 8,760 ocean steps and 2,190
ice calls.  The primary gate is an **exact-entry operator sweep**: each legoESM
boundary/operator consumes the corresponding NEMO entry frame, preventing
upstream trajectory error from being credited to a downstream owner.  The
first over-bar frame is reported by step, RK3 stage if applicable, sub-call,
field, absolute error, denominator, and normalized error.  A continuous
one-layer run is secondary and cannot certify an operator that fails the
exact-entry gate.

Mandatory sub-gates are:

- four-sample `sbc_ssm` recurrence and restart split/rejoin;
- surface `eos_fzp` against the TEOS-10 source order;
- `ice_update_flx` heat, freshwater, salt, and mass ledger, component rows and
  totals;
- every-step `ice_update_tau`, including carried-between-ice-step behavior and
  the implicit/explicit split;
- `sbc_fwb` with ice+ocean volume and its restart split/rejoin;
- all three `tra_sbc_RK3` stages, penetrative shortwave, and isolated T/S
  increments;
- isolated SSH and momentum increments before continuous composition;
- complete coverage and exact input hashes.

### Preregistered ownership hypotheses and private arms

These are **PLAUSIBLE and UNMEASURED**, not findings.

| ID | hypothesis | private one-variable arm | CONFIRM | REFUTE |
|---|---|---|---|---|
| H1 | existing legoESM freshwater sign is opposite NEMO `emp` | change only `F_fw=-emp` to `F_fw=emp` | exact-entry freshwater/SSH residual changes sign and the selected arm alone moves it toward the oracle | residual is unchanged or another earlier field owns it |
| H2 | existing stored-stress convention needs exactly one negation from NEMO `utau/vtau` | remove only `tau=-utau/-vtau` | explicit momentum residual changes sign while `rCdU_ice` and other channels do not | no isolated stress response or earlier operand mismatch |
| H3 | NEMO's FWB volume-1 snow/ice term is absent from the generic legoESM freshwater closure | omit only `snwice_fmass` from the correction | error equals the registered snow/ice mass term and scales with it | mismatch precedes FWB or does not track the term |
| H4 | generic under-ice SW partition double-applies attenuation if used with SI3 `qsr` | route only SI3 `qsr` through the generic attenuation instead of the exact card | solar/top-temperature residual appears at the card boundary and scales with `qsr` | no boundary change or an earlier `ice_update_flx` mismatch owns it |

For each candidate owner, run a scaling probe before naming ownership: inject
the registered nonzero operand at factors 0.5, 1, and 2 while holding every
other exact-entry value fixed.  A first-order owner must show the predicted
sign and approximately linear magnitude (within two fp64 ulps of the replay
where the formula is linear).  Only then may its private arm be interpreted.
No private arm becomes a public selector.

## Planted violations

Every plant must alter a nonzero registered value and the command must exit
nonzero:

1. omit one of the four `sbc_ssm` samples;
2. perturb the active TEOS-10 Horner operand;
3. perturb one nonzero `ice_update_flx` heat-ledger bucket;
4. reverse only the freshwater sign;
5. omit only the `1e-3` PSS-to-kg salt conversion;
6. reverse only the stored-stress sign;
7. replace instantaneous `pu_oce/pv_oce` with the four-step mean on a
   nonzero-current step;
8. zero one nonzero `rCdU_ice` and separately disable its ocean consumer;
9. omit `snwice_fmass` from a nonzero FWB step;
10. skip the boundary term at exactly one RK3 stage;
11. reverse the isolated SSH mass-flux increment;
12. alter one stream byte/hash and remove one coverage-register name.

Self-comparisons, perturbations of zero, and tautological bounds are forbidden.

## Acceptance and stopping rule

The rung may be called AT-BAR only if every non-waived exact-entry row is at or
below `1e-15`, every coverage item is disposed, all plants are red, restart
split/rejoin is at bar, and all code/config/input/output hashes are recorded.
Otherwise the receipt reports the first owner and DEBT/UNMEASURED rows without
relaxing the bar.  Any physics repair must transcribe an active NEMO identity
with `file:line`, be preregistered, and have a one-variable private ablation.

This dispatch stops at this design.  No oracle work configuration, NEMO
instrument, legoESM card, gate, or test is created here.

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| close rung 3.5 as mixed debt | ASKED | recorded in the phase-6 receipt before this design |
| diagnose/stabilize exchange stream before later rungs | ASKED | completed before bulk certification |
| select and justify one rung-3.6 oracle | ASKED | one-layer C1D OCE+ICE selected above |
| cover `sbc_ssm`, `eos_fzp`, `ice_update_flx`, RK3/SSH, every-step stress, and FWB volume type 1 | ASKED | source and future frames/gates specified |
| write signs, coverage, private arms, and failing controls | ASKED | specified above |
| preserve ORCA1-resolved identity, real forcing, CPU/fp64, copied NEMO sources | ASKED | binding design constraints |
| implement or run rung 3.6 in this dispatch | UNASKED | deliberately not done; independent review comes first |
| certify ICE_ADV2D, alternate thermodynamics/dynamics, ponds, or option-4 salinity | UNASKED | outside ORCA1 scope |
| alter a production default or add a second exchange implementation | UNASKED | forbidden by design |
| modify/delete shipped NEMO or push the branch | UNASKED | not done |

## Flagged for future deletion

Nothing is deleted.  If the reviewed implementation supersedes a provisional
run or instrument, it must be listed in its receipt rather than removed.
