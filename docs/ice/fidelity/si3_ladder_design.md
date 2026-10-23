# ORCA2 SI3 fidelity ladder design

Status: **SURVEY AND DESIGN ONLY**; no model code, NEMO build, run, or acquisition.
ORCA2 inherits the complete SI3 reference ice deck: 5 categories, 10 ice and 5 snow layers.
Its ice step is every 2 ocean steps: 21,600 s from `rn_Dt=10800` and `nn_fsbc=2`.
The active stack is H79+aEVP(100), Prather, ridge+raft, BL99/P07, salinity-4, and level ponds.
Lane 3 already built narrow one-category Prather, C-grid aEVP, ridge/raft, and 3+3 BL99 cards.
Those cards are reusable, but no complete lane-3 rung is bit-identical and four ownership debts remain.
The production ORCA2 card is a fold-aware `LatLonCGridGeometry`; current ice dynamics rejects that type.
Current ORCA2 ocean fidelity replays NEMO's post-surface ice fields, so legoESM SI3 is not exercised.
Recommendation: close lane-3 debts first, then add one ORCA2 selector or coupling boundary per rung.
Every proposed oracle record is write-only, versioned, binary64, source-stamped, and scored offline.
The forward bar is normalized pointwise `1e-15`; exact equality is reported separately and never implied.
The design does not include ICE_AGRIF: no selected ORCA2 key or mechanism requires AGRIF.
Implementation is blocked on the numbered decisions in the final table.

## 1. Executed ORCA2 SI3 identity

### Resolution and cadence

Both ORCA2_ICE_PISCES and ORCA2_OMIP_L4 contain the same empty ice override
(`cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_ice_cfg:22-84` and
`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_cfg:22-84`), so all ice selectors below
come from the executable build's copied reference namelist. The OMIP build keys
are `key_si3 key_qco key_vco_1d3d key_RK3`
(`cfgs/ORCA2_OMIP_L4/cpp_ORCA2_OMIP_L4.fcm:1`); ICE_PISCES adds `key_top` and
`key_xios` but keeps those four ice/ocean-time keys
(`cfgs/ORCA2_ICE_PISCES/cpp_ORCA2_ICE_PISCES.fcm:1`). The OMIP ocean deck sets
`rn_Dt=10800` (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:28`) and `nn_fsbc=2`
(`cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:80`); ICE_PISCES selects the same values
(`cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg:28`,
`cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg:80`). Thus both use `rDt_ice=21600 s`.

| SI3 block | Resolved executable choice |
|---|---|
| State / ITD | `jpl=5`, `nlay_i=10`, `nlay_s=5`; virtual ITD off; dynamics and thermodynamics on; `amax_n=amax_s=.997`. HFN categories on (`rn_himean=2`), user bounds off, `himin=.1`, `himax=99` (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:24-32`, `cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:41-46`). |
| Dynamics | `dynALL` on; `dynRHGADV`, `dynADV1D`, and `dynADV2D` off; no-slip (`rn_ishlat=2`); landfast L16 off. Prescribed ADV2D velocities and inactive landfast/file parameters remain reference defaults but do not execute (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:51-58`). |
| Strength / redistribution | H79 on (`pstar=2e4`, `crhg=20`), R75/constant off, strength smoothing on; linear redistribution on (`murdg=3` inert), exponential participation on (`astar=.03`, linear off, `gstar=.15` inert); `csrdg=.5`; ridging on (`hstar=25`, porosity `.3`, snow survival `.5`, pond survival `1`); rafting on (`hraft=.75`, `craft=5`, snow survival `.5`, pond survival `1`) (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:76-103`). |
| Rheology / transport | EVP and adaptive EVP on; EAP/VP off; creep limit `2e-9`, eccentricity 2, 100 subcycles, elastic ratio `.333`, convergence check 0. Prather on; UMx off (`nn_UMx=5` inert) (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:108-131`). |
| Ice surface | `Cd_io=.005`; snow fraction method 2; snow blow `.66`; category flux distribution `-1`; conduction flux and its emulator off; transmitted-solar method 0 (GM77) (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:136-150`). |
| Thermodynamics | Thickness growth/melt, lateral melt, open-water growth, and lead-heat melt all on. BL99 and P07 conductivity on, U64 off; snow conductivity `.31`; attenuation `kappa_i=1`, `kappa_s=10`, melting/dry snow `7/10`; convergence diagnostic off (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:157-174`). |
| Lateral/open water | `beta=1`, `dmin=8 m`; new ice `.1 m`; frazil collection off (inactive defaults `maxfraz=1`, `vfraz=.417`, `Cfraz=5`) (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:179-194`). |
| Salinity | Scheme 4; flushing and drainage on; `simin=.1`, new-ice fraction `.75`, minimum drainage thickness `.1`; VC19 liquidus 2; maximum 50 drainage and 5 flushing subcycles; flush rate `.3`, critical brine volume `.05`; RJW scheme 1 (`alpha=.03`, `Rc=8`); sanity check off. Scheme-1/2 and GN/CW parameters are read but inactive (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:199-236`). |
| Ponds | Ponds on; contribution bounds `.15/.80`; topographic and constant schemes off; level scheme on with flush efficiency `.1`, rain and frozen lids on; pond albedo on with lid thresholds `.005/.015` (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:241-257`). |
| Initial state / diagnostics | Ice initialization on from SST (`nn_iceini_file=0`, threshold 2 K); the reference initial values are 2/1 m ice, .2 m snow, .9 concentration, 6.3 g/kg, 270 K, and pond area/depth/lid `.2/.05/0` (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:262-286`). Albedos are dry/melt snow `.85/.75`, dry/melt ice `.64/.53`, dark/lidded pond `.18/.30`, pivotal thickness 1 m (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:305-311`). Online conservation and point debug are off (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:316-320`). |

The copied `namelist_ice_cfg` is intentionally cited by absence: it contains
only section comments and terminator (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_cfg:22-84`).

### One executed ice step, in source order

The cadence gate is `MOD(kt-1,nn_fsbc)==0`
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:186`).  The active call tree is:

| Order | Executed routine(s) and state boundary |
|---:|---|
| 1 | Copy mean surface current; call `eos_fzp`; then `store_fields`, `ice_sbc_tau`, `diag_set0`, and `ice_rst_opn` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:188-214`). `ice_frm` is not selected (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ref:263`). |
| 2 | `ice_dyn`: `ice_dyn_rhg` -> `ice_dyn_rhg_evp`, then `ice_dyn_adv` -> `ice_dyn_adv_pra`, then `ice_dyn_rdgrft`, then `ice_cor(kt,1)` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn.f90:187-194`). Prather transports ice/snow volume, area, age, all 15 enthalpy layers, all 10 salt layers, and three pond/lid fields; odd/even ice steps reverse x/y sweep order (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_adv_pra.f90:324-366`). |
| 3 | `diag_trends(1)`, then `ice_var_glo2eqv(2)`, `ice_var_agg(1)`, and `store_fields` again (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:220-233`). Global open-boundary `bdy_ice` is inactive (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ref:724`); the dynamics routine has already applied its halo/fold links. Aggregation defines total area and mean ice thickness (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90:181-187`). |
| 4 | `ice_sbc_flx`; then `ice_thd`: `ice_thd_frazil`, category packing via `ice_thd_1d2d`, `ice_thd_zdf` (BL99), `ice_thd_dh`, `ice_thd_temp`, `ice_thd_sal`, a second `ice_thd_temp`, `ice_thd_da`, unpack, `ice_thd_pnd`, `ice_itd_rem`, `ice_thd_do`, and `ice_cor(kt,2)` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd.f90:182-245`). Virtual-ITD/`ice_thd_mono` is the sole inactive thermodynamic child. |
| 5 | `diag_trends(2)`, `ice_var_glo2eqv(2)`, `ice_var_agg(2)`, and `ice_update_flx`; then `ice_dia` and `ice_wri`. `ice_rst_write` is schedule-conditional; `ice_drift_wri` and `ice_ctl` are disabled by this deck (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:250-275`). |
| 6 | Outside the cadence gate, every ocean step calls `ice_update_tau` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:282`). It blends ice/open-water stress and forms implicit `rCdU_ice` from relative ocean/ice speed (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceupdate.f90:373-418`). |

`ice_update_flx` returns the ocean heat, freshwater, salt and cover boundary
fields `qsr`, `qns`, `emp`, `sfx`, `fr_i`, `tn_ice`, and snow/ice mass
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceupdate.f90:145-216`).  The ocean also consumes
ice/ocean stress and `rCdU_ice`; TKE option `nn_mxlice=2` uses `fr_i` and the
mean `hm_i` to set the under-ice mixing length
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:606-629`).

## 2. Gap map from lane 3 to ORCA2

Status meanings: **EXISTS-SIMILAR** means a production abstraction exists but
not this SI3 identity; **PARTIAL** means a narrow SI3 source-ordered arm exists;
**MISSING** means no selectable implementation of the executed mechanism.
Lane-3 verdicts below are reproduced in its merge receipt
(`docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_merge_si3_2026-09-20_receipt.md:236-251`)
and the ORCA2 selector stop names categories, layers, salinity, ponds, and
lateral melt (`docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_merge_si3_2026-09-20_receipt.md:264-270`).

| Executed piece | legoESM status and proof | Delta to ORCA2 / lane-3 status |
|---|---|---|
| 5-category, 10+5-layer state | **PARTIAL.** Generic dynamic state permits category axes but stores only bulk T/salinity and one snow/pond depth; the SI3 column state is explicitly single-category 3+3 (`packages/ice/legoesm/ice/state.py:33-75`). | Missing category-layer enthalpy, 10 layer salt contents, age volume, pond/lid volumes, Prather moments for every extensive. Lane 3 had jpl1 cards only; ICE_RHEO carried 10+5 arrays but thermo was off. |
| HFN category bounds / ITD remap | **EXISTS-SIMILAR.** Generic `simple` and Lipscomb remaps exist (`packages/ice/legoesm/ice/itd.py:1-18`). | Not source-ordered SI3 HFN bounds/remap/corrections; never scored with five categories. |
| C-grid H79+aEVP | **PARTIAL.** A narrow same-index SI3 C-grid config/state/forcing and solver exist (`packages/ice/legoesm/ice/dynamics.py:153-209`). | One-category periodic/closed cards only; fold-aware curvilinear metrics, five-category mass, strength smoothing and halo order remain. Lane 3 rung 3.3 was numeric AT-BAR but trajectory DEBT; rung 3.4 first diverged at completed step 2. |
| Prather transport | **PARTIAL.** Five restart moments and source-ordered pack/unpack exist (`packages/ice/legoesm/ice/transport.py:69-104`). | Extend carry from narrow cards to the full ORCA2 field set and tripole seams/fold. ADV1D prognostics were at bar but restart moments had 5/50 failures; ADV2D was at bar only through step 15 with one unresolved before-level contract. |
| Ridging and rafting | **PARTIAL.** Generic Lipscomb multi-category ridging exists and a selectable SI3 jpl1 arm carries the ORCA1 deck constants (`packages/ice/legoesm/ice/ridging.py:84-100`). | SI3 five-category linear redistribution, exponential participation, rafting, pond/snow/salt/layer-energy transfers, repeated correction loop, and strength smoothing are absent. Lane-3 jpl1 composition first diverged at step 2. |
| BL99/P07 vertical thermo and snow/growth/melt | **PARTIAL.** Selectable SI3 config is hard-validated to 3+3, jpl1, salinity-2, ponds/lateral off (`packages/ice/legoesm/ice/config.py:291-338`). | Generalize to five categories, 10+5, salinity-4, ponds and lateral melt without changing the lane-3 arm. C1D's 8,760-step record retained 35,282 over-bar rows after phase 6 (`docs/ocean/fidelity/testcases/nemo_testcases_l3thd_phase6_receipt.md:180-208`). |
| Lateral melt | **MISSING** for SI3. | Implement `ice_thd_da` at `beta=1`, `dmin=8`; lane 3 deliberately selected it off. |
| Salinity-4 drainage/flushing | **MISSING.** Generic bulk brine is not the 10-layer SI3 salt-content program. | Implement VC19 liquidus, RJW drainage, flushing and their bounded subcycles. Lane 3 only built salinity-2. |
| Level ponds with lids/albedo | **EXISTS-SIMILAR.** Generic package ponds are CESM-style; current state has area/depth but no lid volume. | Implement SI3 level ponds, rain, lids, category pond/lid volume, advection/ridging/remap and albedo coupling. Lane 3 selected ponds off. |
| `ice_cor`, equivalents, aggregation | **PARTIAL.** Narrow ADV2D/RHEO cards contain source corrections; generic production path uses different clamps. | Promote source ordering across 5 categories and both post-dynamics/post-thermo stages, including layer salt/energy/pond conservation. |
| SI3 atmosphere/bulk surface flux | **PARTIAL.** Lane-3 constant-coefficient bulk/albedo gate was AT-BAR, not bit-identical: 16,512/227,760 binary64 rows differed (`docs/ocean/fidelity/testcases/nemo_testcases_l3thd_scalarmath_v2_receipt.md:30-36`). | ORCA2 uses its live bulk forcing and pond/category albedo; preserve the prior card and extend selectors. |
| Ocean exchange and implicit drag | **PARTIAL.** Generic coupler exchange exists, but the ORCA2 fidelity gate currently replays 35 NEMO post-surface inputs including heat, water, salt, `fr_i`, snow/ice mass and `rCdU_ice` (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2b_exchange_gate.py:27-52`). | Build source-convention `ice_update_flx/tau`; replace replay in two rungs so flux and drag ownership do not move together. |
| Under-ice TKE | **EXISTS-SIMILAR** in the certified ocean V1 implementation, but not driven by live ice. | Supply live category-summed `fr_i` and `hm_i`, then select `nn_mxlice=2`; score TKE at its first consumer. |
| Restart / diagnostics | **PARTIAL.** Each lane-3 card has a versioned restart carry. | One unified full-state schema plus category-layer moments and exact before-level/cadence state is missing. |

### Grid verdict: transport yes, dynamics no

The ocean card reads the actual domain and constructs `create_tripole_grid`
with the ORCA fold convention (`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1622-1629`,
`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1676-1689`); the card explicitly names the unmeasured five-category layered
Prather state (`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2091-2107`).
The resulting type is `LatLonCGridGeometry`.  `grid_supports_ice_transport`
admits that type, while `grid_supports_ice_dynamics` admits only cubed-sphere,
regular lat-lon and Voronoi (`packages/ice/legoesm/ice/sea_ice.py:89-121`).
The step guard consequently rejects EVP/mEVP or strain ridging on the ORCA2
grid (`packages/ice/legoesm/ice/sea_ice.py:329-362`).  Therefore the package
**cannot run ORCA2 SI3 dynamics today**; its existing tripole transport is
donor-cell rather than the required Prather program.

## 3. Proposed ladder

### Common record and bar contract

Each rung reuses the ocean ladder's acquisition shape: a write-only Fortran
instrument emits a magic/version header, dimensions, field table, stage and
completed-step number, binary64 payload and source/build/deck hashes; a
separate offline reader admits exact schema/dtype/count/hash and a separate
gate scores it.  The existing ORCA2 surface record demonstrates fixed header
and ordered field registry (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2b_exchange_gate.py:27-52`).
Record entry, every named source stage, post-correction, completed step, and
restart state including all five moments.  Never inject a stage dump as a
continuous carry.  Add a planted swapped-field or one-bit payload control.

For every rung: local pure kernels replaying exact NEMO inputs must be
bit-exact; continuous trajectory is **AT-BAR** only when every registered row
has `abs(L-N)/max(abs(N),1) <= 1e-15`; exact-unequal counts remain published.
Any first failure is DEBT, not a relaxed bar. Size estimates below are
uncompressed planning ranges, not acquisition commitments.

| Rung | NEMO deck and keys | Exactly one new piece / legoESM work | Required record, bar, estimated size |
|---:|---|---|---|
| D1 | `tests/ICE_ADV1D`, shipped SAS keys `key_si3 key_xios key_linssh key_vco_1d`; retain the lane-3 Prather overlay, 40 x 2 s. | Close **restart-moment ownership** in the existing `ICEAdv1DCard`, no new algorithm. | Reuse entry/step/restart record; add moment pre/post-correction provenance. Exact local moment replay and `1e-15` full trajectory; about 20 MiB. |
| D2 | `tests/ICE_ADV2D`, same keys; Prather overlay, 485 x 1200 s. | Close the **before-level / `snwice_mass` input contract**, then run the existing 2-D Prather card unchanged. | All extensive fields, five moments per field, odd/even sweep boundary and restart. Exact local replay, `1e-15` trajectory; 4–8 GiB. |
| D3 | ICE_ADV2D_RHG work case derived from shipped ICE_ADV2D, same keys, lane-3 aEVP/H79 overlay. | Close **aEVP stress-carry trajectory**; retain existing C-grid solver and record. | Outer-step entry, strength, selected aEVP subcycles, U/V/stress exit, Prather exit. Exact local operations, `1e-15` trajectory; 2–5 GiB with sparse subcycle frames. |
| D4 | `tests/ICE_RHEO`, same SAS keys; retain lane-3 H79+aEVP+Prather+ridge/raft overlay, 720 x 30 s. | Close the **step-2 full-dynamics divergence** in the existing RHEO card. | Rheology, advection, ridge/raft iteration, correction and completed-step/restart boundaries. Exact pure stages, `1e-15` trajectory; 10–25 GiB. |
| D5 | `cfgs/C1D/EXP_SASICE`, `key_si3 key_RK3 key_vco_1d3d key_linssh`, lane-3 3+3/salinity-2/pond-off overlay, 8,760 x 3600 s. | Reopen and close the **retained BL99 source-arithmetic debt** only if Decision 1 selects debt-first. | Reuse the 621,960-row year record; retain every ZDF/DH/TEMP/SAL/DO stage. Exact statement probes, `1e-15` trajectory; existing record plus under 1 GiB. |
| O1 | `cfgs/ORCA2_SAS_ICE`, `key_si3 key_linssh key_vco_1d3d key_xios key_RK3`; one-step diagnostic overlay with dynamics/thermo off. | Add **full ORCA2 SI3 state/restart schema**: jpl5, 10+5 energy/salt, age, pond/lid, moments, HFN bounds. | Initial/read/aggregate/restart state and seam/fold halos; exact bridge and restart, about 1 GiB. |
| O2 | ORCA2_SAS_ICE with only Prather active. | Add **Prather across cyclic seam and north fold** for the O1 field registry. | Both sweep orders, halo exchanges and all moments each step; exact local seam/fold replay, `1e-15` 100-step trajectory; 10–20 GiB. |
| O3 | ORCA2_SAS_ICE with H79+aEVP+Prather, ridge/thermo off. | Add **curvilinear tripole aEVP dynamics** and no-slip fold/halo ordering. | Metrics, masks, strength, selected 100-subcycle frames, velocity/stress and transport stages; exact local, `1e-15` trajectory; 15–30 GiB. |
| O4 | ORCA2_SAS_ICE dynamics on, thermo off. | Add **five-category SI3 ridging/rafting** including linear redistribution. | Participation, donor/receiver transfers, each repeat, all extensive conservation channels, post-dynamics `ice_cor`; exact local, `1e-15` trajectory; 15–30 GiB. |
| O5 | ORCA2_SAS_ICE with salinity-1, ponds/lateral off diagnostic overlay. | Generalize **BL99/P07 vertical thermodynamics to jpl5 and 10+5 layers**. | Per-category ZDF/DH/TEMP/open-water stages and tridiagonal/Picard state; exact local, `1e-15` trajectory; 20–40 GiB. |
| O6 | ORCA2_SAS_ICE with salinity-4 enabled, ponds/lateral still off. | Add **salinity-4 VC19/RJW drainage and flushing**. | Layer salt/brine, each bounded subcycle, salt flux and temperature reconversion; exact local, `1e-15` trajectory; 10–20 GiB. |
| O7 | ORCA2_SAS_ICE plus default lateral melt, ponds still off. | Add **lateral melt**. | `ice_thd_da` entry/exit and heat/freshwater diagnostics; bit-exact pure replay, `1e-15` trajectory; 2–5 GiB. |
| O8 | ORCA2_SAS_ICE plus default ITD path, ponds still off. | Add **SI3 HFN ITD remap and post-thermo correction**. | Category boundary crossings, remap fluxes, all conserved extents, `ice_cor(kt,2)`; exact local, `1e-15`; 10–20 GiB. |
| O9 | Unmodified ORCA2_SAS_ICE reference ice deck. | Add **level ponds with rain, lids and pond albedo**. | Pond/lid growth, drainage/flush/freeze, advection/ridging/remap channels and albedo; exact local, `1e-15`; 10–20 GiB. |
| O10 | Unmodified ORCA2_SAS_ICE, 100 x 10800 s, `nn_fsbc=1`. | Add **full-deck composition/cadence**; no new physics implementation. | Complete call-order frames, exchange outputs and restart for all 100 steps; `1e-15`, exact count separate; 40–80 GiB. |
| C1 | Coupled certified ORCA2 ocean card, production `nn_fsbc=2`; keep NEMO `rCdU_ice` replay. | Replace **replayed ice heat/freshwater/salt/cover fields** with live O10 `ice_update_flx`. | Pre-surface, post-flux, ocean first-consumer and 21,600-s ice carry; existing ocean bar plus SI3 `1e-15`; 20–40 GiB for the short admission window. |
| C2 | Same coupled card; Decision 150 drag replay disabled. | Replace **ice stress and `tmod_io/rCdU_ice` replay** with live `ice_update_tau`. | Relative velocity, stress blend, drag and ocean top-stress first consumer every ocean step; exact local, ocean registered bars unchanged; 10–20 GiB. |
| C3 | Same coupled card with `nn_mxlice=2` consuming live ice. | Enable **live under-ice TKE length coupling** using `fr_i` and `hm_i`. | Aggregation -> TKE first-consumer frame and short coupled trajectory; exact local owner and inherited ocean/ice bars; 10–20 GiB. |

Rungs D1–D5 reuse the existing lane-3 cards and immutable records rather than
reacquire them unless admission proves the record lacks the discriminating
operand. ICE_AGRIF is excluded because none of the above decks or cpp keys
selects AGRIF and no mechanism in the target call graph depends on it.

## 4. Recipe shape

The ocean catalog makes recipes named pure scheme bundles, returns copies, and
assembles recipe + setup + explicit overrides (`packages/ocean/legoesm/ocean/recipes.py:272-377`).
Its NEMO fidelity recipe is explicitly config-only—no solver, bridge, grid,
forcing or state translation (`packages/ocean/legoesm/ocean/fidelity/nemo_recipe.py:1-7`).
SI3 should copy that separation:

| Layer | Proposed shape |
|---|---|
| `packages/ice/legoesm/ice/recipes.py` | `ICE_RECIPES`, `list_recipes`, `get_recipe`, `assemble_ice_config`; initial name `nemo_si3_orca2_v1`. Unknown names/selectors fail. |
| Pure fidelity config | `fidelity/nemo_recipe.py` with frozen `NEMOSI3RecipeConfig` and builder. Fields: category/layer counts and HFN bounds; `si3_cgrid_aevp`; H79/strength smoothing; `si3_prather`; five-category ridge/raft/participation/redistribution; BL99/P07; lateral/open-water/snow choices; salinity-4 controls; level-pond controls; `ice_cor`; `nemo_si3_exchange`; cadence. No grid, forcing, state bridge, record reader or test-case arithmetic. |
| Setup/card | ORCA2_SAS_ICE and coupled-ORCA2 setup constructors own mesh, masks, initial/restart data, forcing, cadence schedule and record bridges. Existing ADV1D/ADV2D/RHG/RHEO fidelity cards remain narrow admission fixtures, not recipe alternatives. |
| YAML adapter | Extend the existing `SeaIceExperimentConfig`, which is currently setup-only and accepts only cubed-sphere/column matrix grids (`packages/ice/legoesm/ice/experiment_config.py:1-14`, `packages/ice/legoesm/ice/experiment_config.py:34-46`). Add a required named `recipe:` beside `setup:` for SI3 runs, route through the common assembler, and fail on unused overrides. Preserve setup-only legacy templates. |

This avoids a second sea-ice model: the recipe selects shared-package pieces;
the setup selects ORCA2 geometry/forcing; the fidelity card only translates an
oracle record.

## 5. Differentiability register

The literal forward program must remain oracle-exact. Guards below therefore
belong only in custom JVP/VJP rules or safe derivative expressions; smoothing
the primal is forbidden.

| Executed literal | AD issue and required treatment |
|---|---|
| Ice-presence packing and category participation | Discrete `a_i > epsi10` compaction in thermo and ridge/raft changes shape/membership (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd.f90:187-195`; `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_rdgrft.f90:268-274`). Keep static full arrays with masked primal; zero/one-sided derivative at the threshold. |
| aEVP | `sqrt` of relative speed/deformation, `MAX` creep floor, masks, and a fixed 100-iteration solve have cusps. Keep fixed iteration count; guard only zero denominators and choose documented subgradients. Existing SI3 `_si3_sqrt` does exactly this without changing the primal (`packages/ice/legoesm/ice/dynamics.py:239-264`). |
| Prather | CFL cycle-count selection, alternating sweep, sign/upwind switches and moment limiters are piecewise/non-differentiable. Treat integer cycle/order as static config; differentiate the selected branch with boundary subgradients. |
| Ridging/rafting | `ABS/MIN/MAX`, participation tests, donor/category routing, and convergence-controlled `DO WHILE` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_rdgrft.f90:291-312`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_rdgrft.f90:341-384`). Implement a fixed maximum masked loop under JAX; no data-dependent Python loop; zero derivative at routing boundaries. |
| BL99 heat solve | Ice-point packing, phase switches, caps, tridiagonal solve and Picard convergence/iteration cap are piecewise. Use fixed masked iterations and implicit/custom linear-solve transpose; do not perturb convergence thresholds in the primal. |
| Salinity-4 | Permeability thresholds, `MINVAL`, early `EXIT`, Rayleigh critical switches and 50/5 bounded subcycles (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd_sal.f90:357-414`). Replace early exits with fixed masked scans and define boundary subgradients. |
| Ponds/lids | Ice/pond removal tests, level-scheme caps, freeze/flush switches and square roots (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd_pnd.f90:203-237`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd_pnd.f90:329-381`). Mask on full static arrays; custom zero-safe derivatives at square-root/cap cusps. |
| `ice_cor`, remap, open-water growth, exchange | Concentration/thickness caps, zapping, category search, `min/max/where`, ice/no-ice stress and drag-speed square roots are piecewise. Preserve literal forward state and publish the chosen derivative convention per switch. |

Existing component coverage is real but not end-to-end: ADV1D and ADV2D cards
have JIT plus finite nonzero reverse-mode checks
(`tests/ice/fidelity/test_nemo_si3_phase2_gate.py:58-71`,
`tests/ice/fidelity/test_nemo_si3_phase2_adv2d_gate.py:64-74`); C-grid aEVP has
the same (`tests/ice/fidelity/test_nemo_si3_phase2_rung33_kernel.py:233-238`);
the SI3 jpl1 ridge arm has finite JIT/gradient coverage
(`tests/ice/fidelity/test_si3_jpl1_ridging.py:221-250`); and the landfast depth
path has a finite gradient test (`tests/ice/unit/test_si3_landfast_l16.py:191-207`).
There is **no** full five-category 10+5, salinity-4, pond,
tripole, coupled-ORCA2 gradient test because that path does not yet exist.
Every rung therefore adds `jit`, `grad`, finite, and nonzero controls for its
new differentiable path, plus a planted `stop_gradient` violation.

## 6. Decisions before rung 1

| Decision | Numbered options | Pick |
|---:|---|---|
| 1. First implementation track | **1)** D1–D5 debt-first; **2)** begin O1 immediately and carry lane-3 debts as declared dependencies. | **1.** The ORCA2 deck composes all four debt-bearing operators; moving directly to ORCA2 would destroy first-owner attribution. This reopens C1D only by explicit approval. |
| 2. Shipped-case identity | **1)** retain lane-3 overlays that select the ORCA2-family Prather/aEVP arms; **2)** certify shipped defaults first (UMx/EAP), then add overlay rungs. | **1.** Reuse the established cards and avoid unrelated algorithms; record each overlay as a deck deviation. |
| 3. Bars | **1)** exact local replay plus fixed `1e-15` normalized trajectory bar; **2)** choose new mechanism-specific bars now. | **1.** It preserves lane-3 comparability. Any later statistical bar requires preregistered evidence, never retroactive relaxation. |
| 4. ORCA2_SAS staging | **1)** diagnostic overlays O1–O9, one selector each; **2)** jump to the unmodified full deck. | **1.** It enforces one new owner per rung. |
| 5. Record density / storage | **1)** every outer stage and step, sparse selected aEVP/salinity iterations, full detail at first divergence; **2)** dump every inner iteration. | **1.** It retains attribution while avoiding multi-terabyte records. |
| 6. Coupling order | **1)** fluxes C1, drag C2, under-ice TKE C3; **2)** replace all replay together. | **1.** Decision 150 drag and TKE then remain independently falsifiable. |
| 7. Forward/gradient policy | **1)** literal primal with gradient-only guards; **2)** smooth forward switches. | **1.** Forward fidelity is the target and JAX purity is retained with explicit custom derivatives. |
| 8. AGRIF | **1)** exclude; **2)** add ICE_AGRIF. | **1.** No target selector or key needs it. |

## Validation and handoff

This round changed documentation and its citation registry only. No oracle
artifact was produced or requested. Independent review: operator (Claude).
The implementation round must not start until Decision 1 is answered.
