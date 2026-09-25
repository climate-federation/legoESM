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
`key_si3 key_vco_1d3d key_RK3 key_qco` with no `key_linssh`, matching ORCA1's
`cpp_ORCA1.fcm:1`, so both the RK3 tracer surface boundary and prognostic SSH
are observable.  `key_qco` is mandatory here: RK3 is compiled only with
`key_qco` or `key_linssh` (`src/OCE/stprk3.F90:10-16`), and NEMO stops unless
exactly one is selected (`src/OCE/DOM/domzgr.F90:88-93`).  The work
configuration will be copied; the shipped NEMO tree will remain read-only.

The ocean namelist base is the shipped, prognostic-OCE
`cfgs/C1D/EXP_PAPA/namelist_cfg`; the surface/domain/clock rows are then
replaced by the explicitly enumerated EXP_SASICE and ORCA1 rows below.  This
base choice supplies a complete runnable C1D ocean deck without silently
borrowing an independently tuned ORCA1 ocean.  Its retained one-column
scaffolding is outside the scientific claim and is not exposed as another
selectable physics identity.

The one-layer geometry is a config-local copy of C1D's `usrdef_nam.F90` with
only its vertical-size assignment changed from `kpk=75` to `kpk=2` (one wet
level plus NEMO's terminal level).  The source currently chooses 75 for OCE and
2 for SAS at `cfgs/C1D/MY_SRC/usrdef_nam.F90:73-80`; this one-line change is a
declared geometry composition, not an SI3 physics overlay.

Initialization is deterministic and consumes, rather than merely documents,
the first record of the same official ERA5 member that C1D SAS reads at
`cfgs/C1D/EXP_SASICE/namelist_cfg:153-157`.  A config-local extraction creates
`C1D_OMIP_L3_COUPLED_init.nc` containing a depth dimension of exactly `jpk=2`
for `votemper`, `vosaline`, `u_current`, and `v_current`.  Level 1 is bit-copied
from `sst`, `sss`, `ssu`, and `ssv`; the masked terminal level deterministically
repeats those same source bits and is registered as mask-inactive rather than
left missing or uninitialized.  This two-level schema is mandatory because
`dta_tsd` allocates/copies all `jpk` levels
(`src/OCE/DOM/dtatsd.F90:105-108,196-199`) and C1D U/V likewise copies the complete field
(`src/OCE/C1D/c1d.F90:146-149,195-198`).  The corresponding `namtsd` uses
`ln_tsd_init=T` and `sn_tem/sn_sal`
(`EXP_PAPA/namelist_cfg:48-58`), while `namc1d` uses `ln_uvd_init=T` and
`sn_ucur/sn_vcur` (`:65-79`); NEMO actually reads those arrays at
`src/OCE/DOM/istate.F90:114-123`.  Because non-restart user configurations
initialize SSH through `usr_def_istate_ssh` (`src/OCE/IOM/restart.F90:452-462`),
a config-local copy of that routine will read a single `rn_ssh_init` namelist
value emitted from the same ERA5 `ssh` record and assign it to the masked
column.  The future receipt must hash the source, namelist, derived file, and
extraction command and print the five ingested values and dtypes.  The current
source record is float32 (`sst=-1.690032958984375 degC`, `sss=34`, `ssu=0`,
`ssv=0`, `ssh=0`); conversion to NEMO `wp` is registered, not described as
added precision.  Ice initialization remains the phase-1 resolved
C1D_OMIP_L3 state and must be hash-identical at the first ice-thermodynamics
entry.  Initial SSH is registered at all three initialized time slots: NEMO
copies `Kbb` into `Kaa` and `Kmm` at `restart.F90:466,481`.

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
| namelist base | copied C1D `EXP_PAPA` prognostic-OCE deck | shipped-case runnable scaffold; every retained active selector is listed below |
| components | `OCE ICE` | this design: replace SAS by the prognostic ocean |
| run/calendar | `nn_it000=1`, `nn_itend=8760`, `nn_date0=20180101`, `nn_leapy=1`, no restart, final restart/output | C1D EXP_SASICE `namelist_cfg:25-33`; documented forcing year |
| `rn_Dt` | `3600 s` | C1D EXP_SASICE `namelist_cfg:38`; ORCA1 `namelist_cfg:58` agrees |
| RK3 averaging | `ln_shuman=T` | ORCA1 `namelist_cfg:60`; active RK3 convention |
| one-layer domain | `ln_c1d=T`, `rn_bathy=1 m`, `rn_lat1d=84`, `rn_lon1d=324`, `jpk=2` | phase-1 C1D_OMIP_L3 case value `scripts/validate/ocean_fidelity/testcases/configs/c1d_omip_l3_namelist_cfg:19`, EXP_SASICE `:40,49-50`, and declared copied-source `jpk` edit above |
| initial T/S/U/V | `ln_tsd_init=T`, `ln_uvd_init=T`, no dynamical restoring; exact four-variable derived file | C1D EXP_PAPA ingestion schema `:48-79`; first EXP_SASICE ERA5 record |
| initial SSH | config-local `usr_def_istate_ssh`, `rn_ssh_init` from exact first ERA5 record | active user-domain call `restart.F90:452-462`; no implicit zero fallback |
| `nn_fsbc` | `4` | ORCA1 `EXPREF/namelist_cfg:106`; means four hourly ocean samples and the cadence-guarded SI3 thermodynamic/update body executes at `MOD(kt-1,4)=0` |
| `nn_ice` | `2` | ORCA1 `namelist_cfg:111` |
| `ln_ice_embd` | `.false.` | ORCA1 `namelist_cfg:115`; levitating identity |
| atmosphere schema | EXP_SASICE ERA5 variables and `ln_humi_sph=F`, `ln_humi_dpt=T`, `ln_humi_rlh=F`, `ln_tair_pot=F` | C1D EXP_SASICE `namelist_cfg:96-111,126-134`; `d2m` is dew-point temperature, so ORCA1's inherited specific-humidity default is intentionally not selected |
| `ln_blk` / ocean bulk | `.true.` / NCAR | ORCA1 `namelist_cfg:109,132`; the phase-1 C1D ECMWF bulk arm is replaced, not combined |
| ice-air transfer | constant; `Cd=Ce=Ch=1e-3` | ORCA1 `namelist_cfg:139-142` |
| current feedback | `ln_crt_fbk=F` | ORCA1 `namelist_cfg:134`; no hidden wind/current feedback arm |
| unused surface inputs | `ln_dm2dc=F`, `ln_ssr=F`, `ln_rnf=F` | C1D-resolved shipped defaults, shared ref `:203-210`; hourly ERA5 needs no daily disaggregation and no real restoring/runoff input is available. These are deliberate deviations from ORCA1 `:119,121,126`, outside the exchange claim |
| `nn_fwb` | `1` | ORCA1 `namelist_cfg:122-125` |
| `nn_fwb_voltype` | `1` | shipped `cfgs/SHARED/namelist_ref:666`; ice+ocean volume |
| EOS | TEOS-10 | ORCA1 `namelist_cfg:307-308` |
| solar penetration | `ln_traqsr=T`, RGB, data chlorophyll | ORCA1 `namelist_cfg:118,160-170` (`ln_qsr_rgb=T`, `nn_chldta=1`) |
| RGB profile/constants | `nn_chlprfl=1`, `rn_abs=0.58`, `rn_si0=0.35` | shipped `cfgs/SHARED/namelist_ref:429-432`, inherited by ORCA1 |
| drag | `ln_drgimp=T`, `ln_drgice_imp=T` | ORCA1 `namelist_cfg:258-261` |
| bottom drag | `ln_non_lin=T`, `rn_Cd0=1e-3`, `rn_Cdmax=0.1`, `rn_ke0=2.5e-3`, `rn_z0=3e-3` | ORCA1 `namelist_cfg:258,264-269`; no coefficient is inherited from the PAPA deck by omission |
| vertical coordinate | `ln_vvl_zstar=T` | ORCA1 `namelist_cfg:368`; required prognostic-SSH coordinate identity |
| external mode | `ln_dynspg_ts=T`, `ln_bt_auto=T`, `rn_bt_cmax=0.8`; inherited `ln_bt_fw=T`, `nn_bt_flt=1` | ORCA1 `namelist_cfg:388-399`; shared ref `:1091-1101` for forward integration and the active boxcar filter |
| tracer horizontal operators | `ln_traadv_OFF=T`, `ln_traldf_OFF=T` | retained C1D EXP_PAPA scaffold `namelist_cfg:286-295`; a 1x1 column has no resolved horizontal-transport claim |
| momentum horizontal operators | `ln_dynadv_OFF=T`, `ln_dynvor_ene=T`, `ln_hpg_sco=T`, `ln_dynldf_OFF=T` | retained C1D EXP_PAPA scaffold `namelist_cfg:326-349`; only executed surface-stress/drag increments are claimed |
| vertical closure | `ln_zdfgls=T`, all competing ZDF closures false | retained C1D EXP_PAPA scaffold `namelist_cfg:367-370`; GLS is required for a runnable prognostic column but is outside this exchange rung |
| lateral boundary | `rn_shlat=2`; C1D periodic 1x1 layout | ORCA1 `namelist_cfg:243`; config-local C1D geometry remains the declared oracle |
| passive tracers | `ln_top=F` | C1D EXP_SASICE `namelist_cfg:33`; no TOP model or unregistered tracer state |
| SI3 thermodynamics | single-category HFN, BL99 3+3/P07, option-2 salinity with `rn_sinew=0.75`, `ln_pnd=F`, `ln_icedA=F` | byte-for-byte phase-1 resolved ice deck; its per-row ORCA1 citations remain authoritative |
| ice dynamics | `ln_icedyn=T`, dynamically skipped | ORCA1 ice deck; C1D's `ln_c1d=T` makes the skip explicit at `src/ICE/icestp.F90:167-168` |

Resolution is mechanical: copy the complete EXP_PAPA deck, replace only the
rows above (one line per deviation with its source in an adjacent comment),
resolve it over the shipped reference namelist, and archive both the resolved
files and `ocean.output`.  A pre-run gate must stop unless exactly one bulk,
light, EOS, tracer-advection, tracer-LDF, momentum-advection, vorticity, HPG,
momentum-LDF, ZDF, vertical-coordinate, and surface-pressure selector is true.
It must also stop if any active input-dependent ORCA1 ocean arm not listed here
(restoring, runoff, geothermal/BLL, internal-wave mixing, MLE/EIV, or
space-varying lateral diffusion) leaks in from another deck.  Those global
ORCA1 arms are not silently approximated: they are intentionally absent from
this isolated exchange oracle and are **UNMEASURED** by rung 3.6.

The official archive remains
`/data/abyssal/dbalwada/nemo-inputs/C1D_v5.0.0/C1D_v5.0.0.tar.gz`
(MD5 `9456e6a0a84d40630ad1804fd4061caf`, SHA-256
`54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382`).
The atmospheric member is
`ERA5_NorthGreenland_surface_84N_-36E_1h_y2018.nc`; its existing phase-1
per-file SHA-256 is
`e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe` and
must be rechecked, not copied by assertion, in the rung-3.6 receipt.  ORCA1's
RGB arm additionally names
`merged_ESACCI_BIOMER4V1R1_CHL_REG05` and
`weights_reg05_bilinear.nc` (`namelist_cfg:166-170`).  They are not in the
official C1D archive.  The implementation dispatch must fetch and hash the
documented real files (or receive them out of sandbox) and must stop if they
cannot be obtained; substituting a constant or the unrelated BATS chlorophyll
file is forbidden.  No synthetic forcing is permitted.

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
3. `sbcmod` calls `ice_stp` every ocean step; only its thermodynamic/update
   body is cadence-guarded by `MOD(kt-1,nn_fsbc)=0`.  On an ice step it copies
   mean current and layer thickness and computes the zero-pressure TEOS-10
   freezing point from mean SSS (`src/ICE/icestp.F90:126-138`).  `eos_fzp`
   evaluates the Horner polynomial at `src/OCE/TRA/eosbn2.F90:1672-1683`; the
   omitted `pdep` arm at `:1685-1689` is inactive here.
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
7. Positive NEMO `emp` removes ocean mass.  The active RK3 path calls
   `stp_2D` before its three tracer stages (`src/OCE/stprk3.F90:183-207`),
   builds `sshe_rhs=emp/rho0` (`src/OCE/stp2d.F90:243-251`), and passes it to
   the selected split-explicit solver at `:273-281`.  That solver assigns
   `ssh_frc=sshe_rhs` (`src/OCE/DYN/dynspg_ts.F90:267-285`), subtracts
   `rDt_e*(ssh_frc+divergence)` each external substep at `:623-630`, and forms
   the weighted final SSH at `:818-847`.  The superficially simpler
   `sshwzv.F90` update belongs to MLF and is not evidence for this oracle.
8. With `nn_fwb=1`, `sbc_fwb` corrects only on the surface cadence.  Under
   `nn_fwb_voltype=1` it subtracts `snwice_fmass` from the budget before the
   area mean (`src/OCE/SBC/sbcfwb.F90:224-239`), then applies both
   `emp += emp_corr` and `qns -= emp_corr*rcp*sst_m` at `:292-295`.  The latter
   is the heat content of the freshwater correction, not an optional
   diagnostic.  The selector's meaning is printed at `:143-153`.
9. Before dynamics, `sbcmod` fills the T-grid stress halos and converts them
   to U/V points with the mask-aware neighbor average
   (`src/OCE/SBC/sbcmod.F90:533-547`).  `stp_2D` consumes `utauU/vtauV`, not
   the unstaggered arrays, in `Ue_rhs/Ve_rhs` at `src/OCE/stp2d.F90:195-202`.
10. The ice coefficient is copied into the active top drag as
    `rCdU_top=rCdU_ice` when there is no cavity
    (`src/OCE/ZDF/zdfdrg.F90:116-129`).  The 3-D implicit vertical solve adds
    top drag at `src/OCE/DYN/dynzdf.F90:298-303,472-477`.  The split-explicit
    operator separately combines top and bottom coefficients
    (`src/OCE/DYN/dynspg_ts.F90:1608-1618`) and adds the top baroclinic RHS at
    `:1646-1669`.  In a one-wet-layer column the top and bottom indices refer
    to the same cell, but the two coefficients remain distinct operands and
    neither may overwrite the other.
11. RK3 deposits `utauU/vtauV` again as the surface boundary of the 3-D
    implicit vertical solve, after the first Thomas recurrence and before the
    remaining forward/back substitutions
    (`src/OCE/DYN/dynzdf.F90:322-345,496-519`; the actual insertions are
    `:328-330,501-504`).  This is distinct
    from the same stress's split-explicit depth-mean consumer in `stp_2D`; both
    must execute, while a separate explicit top-cell kick must not.

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

The pre-implementation search also found an existing lat-lon RGB consumer:
when both `sw_down` and `chl` are present it subtracts solar from total heat
and applies the shared RGB penetration kernel
(`packages/ocean/legoesm/ocean/dynamics/ocean_pe_latlon_cgrid.py:3977-4003`).
That is the path to reuse.  Supplying `sw_down` without `chl` would instead
select the generic 0.94/Jerlov path at `:4004-4018` and is forbidden for this
ORCA1 identity.  No existing `rCdU_ice`/implicit **top**-drag path was found;
the existing matrix plumbing is for bottom `rCdU_bot` (for example
`packages/ocean/legoesm/ocean/state.py:2678-2697`).  The implementation must
extend the shared vertical/momentum solve for a top coefficient rather than
copying a solver or relabeling bottom drag.  If the C1D coordinate does not
satisfy the existing partial-cell-only bottom helper, that helper is generalized
in place for the registered flat one-layer geometry; it is not copied.  This is
a preregistered **new extension of the existing shared solve**, not claimed
reuse of an already present top-drag feature.

The ocean card must explicitly set
`LatLonCGridOceanConfig.freshwater_closure="real_freshwater"` and
`normalize_freshwater=False`.  The class otherwise defaults to
`"virtual_salt_flux"` (`packages/ocean/legoesm/ocean/state.py:1879-1886`).
The selected real-volume arm preserves salt content during z-star dilution and
skips the extra virtual-salt source
(`ocean_model_latlon_cgrid.py:6002-6018`).  Normalization is off because NEMO's
registered `sbc_fwb` output already includes `emp_corr`; applying legoESM's
generic mean removal would be a second freshwater-budget correction.  The
construction gate and continuous-run gate must reject either other value.

The future legoESM column card must resolve the following existing execution
knobs explicitly; defaults are not accepted as evidence:

| legoESM row | selected value | status / source |
|---|---|---|
| free surface | `barotropic_solver="explicit_substep"` | existing solver, `state.py:1397-1403` |
| NEMO filter | `barotropic_time_filter="nemo_boxcar1_ab3"` | existing forward `nn_bt_flt=1` identity, `state.py:1082-1088` |
| substeps | exact NEMO `icycle` and `rDt_e` | source-replayed for exact-entry gates; continuous card must reproduce `ln_bt_auto=T,rn_bt_cmax=0.8` and hard-stop if its per-step values differ |
| coupled time step | `tracer_time_integrator="rk3_ws"`, `momentum_time_integrator="rk3_ws"`, `outer_integrator="forward_euler"` | existing coupled validator, `state.py:2071-2077,2332`; `ocean_model_latlon_cgrid.py:3175-3205` |
| stage mean | `nemo_stage_mean_imposition=True` | existing NEMO RK3 reconciliation, `state.py:1256-1265` |
| vertical solve | `implicit_vertical_mixing=True` | existing shared solve, `state.py:2158`; one wet layer has no interior interface, but the registered surface stress and top/bottom drag still alter its RHS/diagonal |
| stress placement | `surface_stress_implicit=True` | existing NEMO `dynzdf` surface-BC path; withholds the ordinary explicit kick, adds its depth mean to split-explicit forcing, and inserts the top-cell implicit RHS (`state.py:2290-2300`; executing `ocean_pe_latlon_cgrid.py:3937-3960`, `ocean_model_latlon_cgrid.py:4041-4065,7842-7863`) |
| forcing time level | `barotropic_forcing_centred=False` | NEMO inherits `ln_bt_fw=T`; existing selector and source map at `state.py:2759-2785` |
| freshwater | `freshwater_closure="real_freshwater"`, `normalize_freshwater=False`, `fix_eta_drift=True`, `use_conservation_fixer=False` | existing real-volume path and safety checks, `ocean_model_latlon_cgrid.py:2191-2268,5896-5912` |
| implicit bottom drag | `bottom_drag_scheme="nemo_quadratic"`, `zdf_drag_in_matrix=True`, `zdf_baroclinic_only=True`, `barotropic_drag_substep=True` | existing shared NEMO composition, `state.py:2671-2758` |
| implicit top drag | raw `rCdU_top` plus a new top-coefficient extension of those same shared paths | **NEW prerequisite**, source identity registered above; no private or duplicate solver |

There is one explicit construction gap.  The existing `rk3_ws` validator
requires its previously certified FCT2/flux-form horizontal companions, while
this oracle deliberately retains C1D's `ln_traadv_OFF` and `ln_dynadv_OFF` on a
1x1 periodic column.  Those are both structural-zero operators but they are not
interchangeable selectors.  Before any run, the existing RK3-WS implementation
must be extended to admit the exact `off` operator pair only for the registered
1x1 geometry, with a hard geometry guard and planted off-manifold failure; the
implementation may not silently select FCT2/upwind3 or claim ORCA1 horizontal
transport coverage.  Until that prerequisite and the top-drag extension are
reviewed, rung 3.6 remains **UNMEASURED and not constructible**.

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
| `qsr`, `qns` | W m-2, positive into ocean; `qns` enters `tra_sbc_RK3`, `qsr` enters `tra_qsr` | `sw_down=qsr`, `chl=the registered ORCA1 input`, and `q_net=qsr+qns`; the existing RGB consumer recovers `qns=q_net-sw_down` before depositing solar, with no second SW addition |
| `emp` | kg m-2 s-1, positive ocean mass loss; positive lowers SSH | physical freshwater `F_fw=-emp`, positive into ocean |
| `sfx` | PSS kg m-2 s-1, positive salinity source in `tra_sbc_RK3` | real salt mass `salt_flux=1e-3*sfx` kg m-2 s-1; the existing ocean consumer converts kg salt back to salinity tendency |
| `utau`, `vtau` | Pa, positive force on the ocean in NEMO momentum | legoESM stored atmosphere-convention `tau_x=-utau`, `tau_y=-vtau`; its ocean sign flip restores the NEMO force |
| `rCdU_ice` | m s-1-like implicit top-drag coefficient, non-positive by `-tmod_io*at_i/rho0` | retain the signed raw operand through the card, then form `r_eff_top=-rCdU_top >= 0` exactly once at the shared positive-rate solver boundary; never route the raw sign into the matrix or fold it into explicit `tau` |
| `snwice_fmass` | kg m-2 s-1, positive increase in ice+snow mass | budget-only term; subtract from NEMO-style `emp` before the global/one-cell FWB correction |

The separate `qsr` and `qns` paths prevent double application.  Likewise,
`F_fw` updates real mass/SSH once; a virtual-salt or KPP diagnostic must not
apply it a second time.  These are preregistered interface requirements, not
measured agreements.

The sign conversion mirrors the existing bottom-drag convention: NEMO's raw
`rCdU_bot<=0` becomes legoESM's positive `r_eff=-rCdU_bot` before the diagonal
gain (`state.py:2678-2686`; executing use
`ocean_model_latlon_cgrid.py:8119-8130,8075-8098`).  The new top extension must
call the same conversion/assembly owner.  A separate implementation or direct
use of the negative raw value would create anti-drag and is a construction
error.

## Frame and time-level registry (Rule 1d)

The config-local NEMO writer will be WRITE-only and will serialize interior
values explicitly; it may not mutate model arrays.  Every frame stores `kt`,
`Kbb`, `Kmm`, and, where applicable, `Krhs`, `Kaa`, and RK3 stage.  Runtime
indices, rather than inferred labels such as “now,” are authoritative.

| frame | cadence | registered time level and fields |
|---|---|---|
| `INITIAL_STATE` | once, before `nit000` | source-file float32 values and post-ingestion `wp` T/S/U/V at `Kbb/Kmm`, including deterministic masked terminal level; `rn_ssh_init` and SSH at `Kbb/Kaa/Kmm`; exact phase-1 ice state; `e3t/u/v`, masks, top/bottom indices, and all input/source hashes |
| `PRE_SSM` | every ocean step | `uu,vv(:,:,1,Kbb)`; `ts(:,:,1,jp_tem/jp_sal,Kmm)`; `ssh(:,:,Kmm)`; `e3t(:,:,1,Kmm)`; `fraqsr_1lev`; carried seven accumulators before `sbc_ssm` |
| `POST_SSM` | every ocean step | seven accumulators after add/divide; division flag `MOD(kt-1,4)==0` |
| `POST_FZP` | each ice step | `sss_m` at completed four-sample mean; `t_bo` after zero-pressure TEOS-10 conversion and Kelvin offset |
| `PRE_UPDATE_FLX` | each ice step | before-level `at_i_b,a_i_b`; current `at_i`; all heat/mass/salt operands enumerated below |
| `POST_UPDATE_FLX` | each ice step | `qsr,qns,emp,sfx,qt_atm_oi,qt_oce_ai,fwfice,snwice_mass_b,snwice_mass,snwice_fmass,fr_i,tn_ice,alb_ice` |
| `POST_UPDATE_TAU` | every ocean step | instantaneous `pu_oce,pv_oce=uu,vv(:,:,1,Kbb)`; carried `tmod_io` and ice-step refresh flag; `at_i,u_ice,v_ice,drag_io,utau_oce,vtau_oce,rCdU_ice,utau,vtau,taum` |
| `POST_FWB` | every ocean step | incoming and outgoing `emp` **and `qns`**, `snwice_fmass`, area/mask, `emp_ext`, `emp_corr`, `rcp`, `sst_m`; active-cadence flag |
| `POST_SBC_STAGGER` | every ocean step | post-`lbc_lnk` T-point `utau/vtau` including halos; neighbor indices and `umask/vmask/tmask`; resulting `utauU/vtauV` consumed by `stp_2D` |
| `POST_ZDF_DRG_COEFF` | every ocean step, after `zdf_phy` and before `stp_2D` | `Kbb`; signed raw `rCdU_ice`, resulting raw `rCdU_top`, independent raw `rCdU_bot`, their positive legoESM `r_eff_top/r_eff_bot` conversions, masks, and `miku/mikv/mbku/mbkv`; this is coefficient assembly only (`stprk3.F90:163-186`) |
| `POST_TRA_SBC_RK3` | every RK3 stage | `Kbb,Kmm,Krhs,kstg`; before/after top-level T/S RHS; `emp,qns,sfx,e3t,rho0,rcp`, and top-level T/S at `Kbb` |
| `POST_TRA_QSR` | every ocean step, RK3 stage 3 only (8,760 frames) | `qsr`, exact interpolated `chl`, RGB selectors/lookup and per-level attenuation operands, `fraqsr_1lev`, and before/after temperature RHS at every wet level; the call is inside `SELECT CASE(kstg)`, `CASE(3)` at `stprk3_stg.F90:540-585` |
| `PRE_DYN_SPG_TS` | every ocean step, after complete `stp_2D` RHS assembly | `Kbb,Kaa`; `emp`, `rnf/isf` inactive flags, complete `sshe_rhs`, `Ue_rhs`, `Ve_rhs`; separate top/bottom face coefficients and baroclinic RHS from `dyn_drg_init`; external-mode selectors, `icycle`, `rDt_e`, and initial external state (`stp2d.F90:112,195-202,243-281`) |
| `SSH_SUBSTEP` | every `jn=1..icycle` | `jn/icycle`, `rDt_e`, `ssh_frc`, extrapolation coefficients, separate top/bottom `pCdU_u/v` and drag tendencies, `sshn_e/sshb_e/sshbb_e`, `zsshp2_e`, `zhU/zhV`, `zhdiv`, `ssha_e`, `wgtbtp1/2`, and running SSH/U/V sums |
| `POST_STP2D` | every ocean step | final weight sums; divided `ssh(:,:,Kaa)`, `uu_b/vv_b(:,:,Kaa)`, `un_adv/vn_adv`; isolated surface-mass contribution replayed with every other RHS operand held fixed |
| `PRE_DYN_ZDF_SOLVE` | every ocean step, RK3 stage 3 only | `Kbb,Kmm,Krhs,Kaa,kstg=3`; top/bottom indices and coefficients, pre-drag diagonal and RHS, separate top/bottom diagonal additions; `utauU/vtauV`, `rho0`, `e3u/e3v(:,:,1,Kaa)`, masks, and top-cell RHS immediately before/after the stress insertion; complete tridiagonal matrix/RHS before the shared solve (`stprk3_stg.F90:430`; `dynzdf.F90:293-342,466-516`) |

The first `POST_SSM` mean is deliberately tested against NEMO's three seeded
copies plus the ordinary fourth accumulation; it is not assumed to be a
one-sample mean.  Restarts must cover the seven mean accumulators, the last
ice-step `tmod_io`, `utau_oce/vtau_oce`, `rCdU_ice`, `snwice_mass(_b)`, FWB
state (`emp_corr` and any active accumulator), active solar carry
`fraqsr_1lev`, and the ocean/ice prognostics.  `fraqsr_1lev` is written to the
ocean restart at `traqsr.F90:238-244` and read or explicitly defaulted at
`:1433-1437`; the restart split/rejoin gate must compare it before its next
`sbc_ssm` use.

## Coverage register seed

The future machine-readable gate must contain exactly one `VERIFIED` or
`WAIVED` disposition with a source-backed reason for every item below; missing
and duplicate names are fatal.

1. **Mean-ocean inputs:** `ssu_m`, `ssv_m`, `sst_m`, `sss_m`, `ssh_m`,
   `e3t_m`, `frq_m`; plus their instantaneous inputs, RGB chlorophyll, and
   restart carries.
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
   `rCdU_ice`; post-halo T stresses, every neighbor/mask operand, and consumed
   `utauU/vtauV` are separate required rows.  The 1x1 periodic identity stencil
   is VERIFIED if it executes at bar; general off-column interpolation geometry
   is WAIVED as outside this oracle, never inferred from that identity.
5. **Solar consumer:** RGB `chl`, `ln_qsr_rgb`, `nn_chldta`, `nn_chlprfl`,
   `rn_abs`, `rn_si0`, lookup/attenuation operands, and the resulting per-level
   solar tendency and `fraqsr_1lev`; none is silently waived as deep-ocean
   detail because `fraqsr_1lev` returns to `ice_update_flx`.
6. **Ocean consumers:** isolated increments and total before/after values for
   top-level T, S, SSH, U, and V; `qsr`, `qns`, `emp`, `sfx`, `utau`, `vtau`,
   and `rCdU_ice` at the exact consuming call.  SSH includes `sshe_rhs`,
   `ssh_frc`, `icycle/rDt_e`, every substep state/divergence/weight/running sum,
   and final division—not only `Kbb/Kaa` endpoints.  Implicit momentum includes
   separate `rCdU_top` and `rCdU_bot` face coefficients, coincident top/bottom
   level indices, each matrix/RHS contribution, and their combined operator.
   The same registered `utauU/vtauV` must be VERIFIED independently in the
   split-explicit RHS and in the stage-3 implicit top-cell RHS, including its
   `rho0`, `e3u/e3v(Kaa)`, mask, and before/after operands.
7. **FWB:** `snwice_mass`, `snwice_mass_b`, `snwice_fmass`, input/output
   `emp` and `qns`, `emp_ext`, `emp_corr`, `rcp`, `sst_m`, wet area/mask, and
   `nn_fwb_voltype`.
8. **State/restart:** all `INITIAL_STATE` source and ingested values; every
   prognostic in Appendix A of the lane-3 dossier,
   the seven `sbc_ssm` carries, stress carries, FWB carries, and the one-layer
   ocean T/S/SSH/U/V state; `fraqsr_1lev` is a mandatory VERIFIED solar carry.
   Inactive pond, option-4 salinity, and dynamics moments may be WAIVED only
   with their resolved selector and call-site skip.

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

- exact geometry and T/S/U/V/SSH/ice initialization, including the derived
  input hash and post-conversion `wp` values;
- four-sample `sbc_ssm` recurrence and restart split/rejoin, including the
  restored `fraqsr_1lev` input;
- surface `eos_fzp` against the TEOS-10 source order;
- `ice_update_flx` heat, freshwater, salt, and mass ledger, component rows and
  totals;
- every-step `ice_update_tau`, including carried-between-ice-step behavior and
  the implicit/explicit split, T-to-U/V stress staggering, and separate
  top/bottom drag contributions in both vertical and split-explicit consumers;
- `sbc_fwb` with ice+ocean volume and its restart split/rejoin;
- all three `tra_sbc_RK3` stages, penetrative shortwave, and isolated T/S
  increments;
- every split-explicit SSH substep and final weighted SSH/momentum result,
  including an isolated `sshe_rhs` surface-mass replay, before continuous
  composition;
- complete coverage and exact input hashes.

Both exact-entry and continuous gates must print and assert
`freshwater_closure="real_freshwater"` and `normalize_freshwater=False` before
integration; either another closure or a second normalization is a hard
configuration failure, not an alternate measured arm.

### Preregistered ownership hypotheses and private arms

These are **PLAUSIBLE and UNMEASURED**, not findings.

| ID | hypothesis | private one-variable arm | CONFIRM | REFUTE |
|---|---|---|---|---|
| H1 | existing legoESM freshwater sign is opposite NEMO `emp` | change only `F_fw=-emp` to `F_fw=emp` | exact-entry freshwater/SSH residual changes sign and the selected arm alone moves it toward the oracle | residual is unchanged or another earlier field owns it |
| H2 | existing stored-stress convention needs exactly one negation from NEMO `utau/vtau` | remove only `tau=-utau/-vtau` | explicit momentum residual changes sign while `rCdU_ice` and other channels do not | no isolated stress response or earlier operand mismatch |
| H3 | NEMO's FWB volume-1 snow/ice term is absent from the generic legoESM freshwater closure | omit only `snwice_fmass` from the correction | error equals the registered snow/ice mass term and scales with it | mismatch precedes FWB or does not track the term |
| H4 | generic under-ice SW partition double-applies attenuation if used with SI3 `qsr` | route only SI3 `qsr` through the generic attenuation instead of the exact card | solar/top-temperature residual appears at the card boundary and scales with `qsr` | no boundary change or an earlier `ice_update_flx` mismatch owns it |
| H5 | the absent legoESM top-drag channel, rather than explicit stress, owns an implicit momentum residual | zero only registered `rCdU_top` while retaining `rCdU_bot` | isolated implicit residual changes by the source-replayed top contribution at both consumers | residual precedes drag assembly or scales with another operand |

For each candidate owner, run a scaling probe before naming ownership: inject
the registered nonzero operand at factors 0.5, 1, and 2 while holding every
other exact-entry value fixed.  A first-order owner must show the predicted
sign and approximately linear magnitude (within two fp64 ulps of the replay
where the formula is linear).  Only then may its private arm be interpreted.
No private arm becomes a public selector.

## Planted violations

Every plant must alter a nonzero registered value and the command must exit
nonzero.  For structurally zero initial U/V/SSH, the plant replaces the source
record with a finite nonzero sentinel so the ingestion check remains
non-vacuous:

1. alter one nonzero initial T/S value and inject the finite sentinel through
   one of U/V/SSH while holding the generated file manifest stale;
2. omit one of the four `sbc_ssm` samples;
3. perturb the active TEOS-10 Horner operand;
4. omit the registered chlorophyll so the generic 0.94/Jerlov arm is selected;
5. perturb one nonzero `ice_update_flx` heat-ledger bucket;
6. reverse only the freshwater sign;
7. omit only the `1e-3` PSS-to-kg salt conversion;
8. reverse only the stored-stress sign;
9. replace instantaneous `pu_oce/pv_oce` with the four-step mean on a
   nonzero-current step;
10. zero one nonzero `rCdU_ice` and separately disable its ocean consumer; a
   third plant drops only top drag while retaining coincident one-layer bottom
   drag, proving that the gate distinguishes them;
11. omit `snwice_fmass` from a nonzero FWB step;
12. omit only FWB's nonzero `qns -= emp_corr*rcp*sst_m` correction;
13. skip the boundary term at exactly one RK3 stage;
14. reverse the isolated `sshe_rhs` mass-flux increment in one external
    substep;
15. perturb one nonzero `utauU/vtauV` after staggering while leaving its
    T-point input unchanged;
16. perturb the nonzero `fraqsr_1lev` restart carry before split/rejoin;
17. remove the required terminal level from the derived initialization file;
18. select `virtual_salt_flux` or enable a second freshwater normalization;
19. feed one negative raw `rCdU_top` directly to the positive-rate matrix
    interface;
20. omit or reposition only the stage-3 implicit top-cell stress insertion
    while retaining the identical stress in `stp_2D`;
21. alter one stream byte/hash and remove one coverage-register name.

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
