# ORCA2 SI3 fidelity ladder design

Status: **SURVEY AND DESIGN ONLY**; no model code, NEMO build, run, or acquisition.
ORCA2 inherits the complete SI3 reference ice deck: 5 categories, 10 ice and 5 snow layers.
Coupled ORCA2 steps ice every 21,600 s; ORCA2_SAS_ICE instead steps it every 10,800 s.
The active stack is H79+aEVP(100), Prather, ridge+raft, BL99/P07, salinity-4, and level ponds.
Lane 3 built reusable narrow one-category Prather, C-grid aEVP, ridge/raft, and 3+3 BL99 cards, but no rung is bit-identical.
The SI3 arms themselves hard-code one category and doubly periodic halos; that blocks the tripole grid.
Current ORCA2 ocean fidelity replays NEMO's post-surface ice fields, so legoESM SI3 is not exercised.
Recommendation: classify the C1D snow branch per platform, then start O1; a literal close is not required on an FTZ backend.
Every proposed oracle record is write-only, versioned, binary64, source-stamped, and scored offline.
The forward bar is normalized pointwise `1e-15`; exact equality is reported separately and never implied.
O8-O16 are stage views of one instrumented production-deck run, not nine acquisitions.
ORCA2_SAS acquisition also needs a user-approved, hash-pinned `SAS_v5.0.0.tar.gz` source.
ICE_AGRIF stays excluded: no selected ORCA2 key or mechanism requires AGRIF.
Implementation is blocked on Decision 1; later acquisition is blocked on the data-source decision.

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
| Ocean/air coupling convention | Ice is **levitating**, not embedded (`ln_ice_embd=.false.`), so SI3 does not add local ice pressure/mass/salt loading to the ocean (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ref:199`). Ice-air drag, heat and moisture use the constant arm with `Cd_ia=Ce_ia=Ch_ia=1.4e-3` (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ref:259-262`). O2 owns the initialization arm, O5 owns its per-ice-step aEVP surface-slope arm, C0 records its per-ocean-step coupled load disposition, and O7 owns the constant-coefficient surface step. |
| Thermodynamics | Thickness growth/melt, lateral melt, open-water growth, and lead-heat melt all on. BL99 and P07 conductivity on, U64 off; snow conductivity `.31`; attenuation `kappa_i=1`, `kappa_s=10`, melting/dry snow `7/10`; convergence diagnostic off (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:157-174`). |
| Lateral/open water | `beta=1`, `dmin=8 m`; new ice `.1 m`; frazil collection off (inactive defaults `maxfraz=1`, `vfraz=.417`, `Cfraz=5`) (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:179-194`). |
| Salinity | Scheme 4; flushing and drainage on; `simin=.1`, new-ice fraction `.75`, minimum drainage thickness `.1`; VC19 liquidus 2; maximum 50 drainage and 5 flushing subcycles; flush rate `.3`, critical brine volume `.05`; RJW scheme 1 (`alpha=.03`, `Rc=8`); sanity check off. Scheme-1/2 and GN/CW parameters are read but inactive (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:199-236`). |
| Ponds | Ponds on; contribution bounds `.15/.80`; topographic and constant schemes off; level scheme on with flush efficiency `.1`, rain and frozen lids on; pond albedo on with lid thresholds `.005/.015` (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:241-257`). |
| Initial state / diagnostics | Ice initialization on from SST (`nn_iceini_file=0`, threshold 2 K); the reference initial values are 2/1 m ice, .2 m snow, .9 concentration, 6.3 g/kg, 270 K, and pond area/depth/lid `.2/.05/0` (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:262-286`). Albedos are dry/melt snow `.85/.75`, dry/melt ice `.64/.53`, dark/lidded pond `.18/.30`, pivotal thickness 1 m (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:305-311`). Online conservation and point debug are off (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_ref:316-320`). |

The copied `namelist_ice_cfg` is intentionally cited by absence: it contains
only section comments and terminator (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ice_cfg:22-84`).

### ORCA2_SAS staging data and cadence

The shipped SETTE manifest requires both `ORCA2_ICE_v5.0.0.tar.gz` and
`SAS_v5.0.0.tar.gz` (`sette/input_ORCA2_SAS_ICE.cfg:1-2`).  A filesystem census
found the former archive but no SAS archive or extracted `sas_grid_*` files;
the ORCA2_ICE archive also contains no `sas_grid_*` member.  Those fields are
the prescribed ocean state read by the SAS deck
(`cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ref:398-404`).  The official SETTE fetcher
constructs the versioned filename and downloads it from the NEMO JASMIN store
(`sette/sette_fetch_inputs.sh:39-46`).

**DATA-SOURCE DECISION (not taken):** before any O-rung acquisition, approve
fetching exactly `SAS_v5.0.0.tar.gz` from
`https://gws-access.jasmin.ac.uk/public/nemo/sette_inputs/r5.0.0/`, then pin the
URL, byte size, raw-archive SHA-256 and extracted-member SHA-256 manifest in the
receipt.  The pick is the official archive, not generating substitute SAS
fields or assuming they are in `ORCA2_ICE_v5.0.0.tar.gz`.

The SAS deck explicitly sets `rn_Dt=10800` and `nn_fsbc=1`, hence a 10,800 s
ice step (`cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg:28`,
`cfgs/ORCA2_SAS_ICE/EXPREF/namelist_cfg:61`).  Coupled ORCA2 uses `nn_fsbc=2`
and a 21,600 s ice step.  C0 below owns that cadence change; C1 no longer
bundles it with the live-flux swap.

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

The review hypothesis that `ln_ice_embd` is tested inside `iceupdate` is
**REFUTED by the compiled source**: that module has no reference to the switch.
The repeated false arm instead runs in aEVP, where `ice_var_sshdyn` returns the
unloaded ocean SSH for levitating ice
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_rhg_evp.f90:269-283`;
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90:1075-1090`), and in the coupled ocean
pressure-gradient step, where the snow/ice load block is skipped
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:247-258`).  O5 and C0 therefore own
those executed dispositions.  O16 records `snwice_mass(_b)` because it is an
exchange output, but does not claim a nonexistent `iceupdate` branch.

Two other ocean modules mention ice cover but do not add SI3 ladder owners.
Surface salinity restoring is active (`nn_sssr=2`), but `nn_sssr_ice=1` makes
its `fr_i`-dependent branch false, so the restoring coefficient stays one
everywhere (`cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:151-156`,
`cfgs/ORCA2_OMIP_L4/EXP00/namelist_ref:446-458`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/sbcssr.f90:166-174`).  It is therefore an
**inert ice-cover consumer** for this identity.  Icebergs would interpolate
`fr_i` at each berg (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icbutl.f90:168-173`), and
the shipped deck turns icebergs on
(`cfgs/ORCA2_OMIP_L4/EXP00/namelist_cfg:204-206`), but the accepted ORCA2 card
and record are the one-variable `ln_icebergs=.false.` variant selected by user
Decision 7
(`docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_phase2_manifest.md:136-143`).
Icebergs are therefore **excluded**, not silently inherited from the shipped
deck.

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
| SI3 atmosphere/bulk surface flux | **PARTIAL.** Lane-3 constant-coefficient bulk/albedo gate was AT-BAR, not bit-identical: 16,512/227,760 binary64 rows differed (`docs/ocean/fidelity/testcases/nemo_testcases_l3thd_scalarmath_v2_receipt.md:30-36`). | O7 must own the whole executed ice-atmosphere surface step: `blk_ice_1`, `ice_alb`, `blk_ice_2`, and `ice_flx_other`, including the resolved 1.4e-3 coefficients. The calls and thermodynamic inputs/outputs are source-ordered at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:203-250`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icesbc.f90:139-166`, and `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icesbc.f90:201-264`; `ice_flx_other` then forms bottom/lead heat terms (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icesbc.f90:440-570`). |
| Ocean exchange and implicit drag | **PARTIAL.** Generic coupler exchange exists, but the ORCA2 fidelity gate currently replays 35 NEMO post-surface inputs including heat, water, salt, `fr_i`, snow/ice mass and `rCdU_ice` (`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2b_exchange_gate.py:27-52`). | Build source-convention `ice_update_flx/tau`; replace replay in two rungs so flux and drag ownership do not move together. |
| Under-ice TKE | **EXISTS-SIMILAR** in the certified ocean V1 implementation, but not driven by live ice. | Supply live category-summed `fr_i` and `hm_i`, then select `nn_mxlice=2`; score TKE at its first consumer. |
| Restart / diagnostics | **PARTIAL.** Each lane-3 card has a versioned restart carry. | One unified full-state schema plus category-layer moments and exact before-level/cadence state is missing. |

### Grid verdict: the narrow SI3 arms are periodic, not tripolar

The ocean card reads the actual domain and constructs `create_tripole_grid`
with the ORCA fold convention (`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1694-1701`,
`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1748-1761`); the card explicitly names the unmeasured five-category layered
Prather state (`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2134-2150`).
The resulting type is `LatLonCGridGeometry`.  The generic production path has
a type guard for strain dynamics (`packages/ice/legoesm/ice/sea_ice.py:88-120`,
`packages/ice/legoesm/ice/sea_ice.py:329-362`), but that guard is **not the
binding blocker for the fidelity arms**, which are called directly by their
cards.  The aEVP arm hard-codes a two-cell halo and exactly one category
(`packages/ice/legoesm/ice/dynamics.py:101-103`), then rebuilds both horizontal
halos with `mode="wrap"`
(`packages/ice/legoesm/ice/dynamics.py:809-822`).  The SI3 Prather arm likewise
implements an ICE_ADV2D bi-periodic halo
(`packages/ice/legoesm/ice/transport.py:1829-1834`).  Thus neither arm can
represent ORCA2's cyclic east-west seam plus north fold today.  O4/O5 must
replace those arm-local periodic assumptions with the existing fold-aware
geometry operations and extend the carried state to five categories; changing
only the generic guard would still run the wrong halo program.

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

For every rung, local pure kernels replaying exact NEMO inputs must be
bit-exact except for a separately proved and platform-stamped intrinsic or
denormal floor defined in section 5. Continuous trajectory is **AT-BAR** only
when every registered row has `abs(L-N)/max(abs(N),1) <= 1e-15`;
exact-unequal and floor counts remain published and are never called
bit-identical. Any first normal-range failure is DEBT, not a relaxed bar. Size
estimates below are uncompressed projection ranges, not acquisition
commitments.

The O rungs separate **operator admission** from **composition**. O1-O16 replay
the exact NEMO input at each named source boundary, so upstream drift cannot
hide a first owner; only O17 advances the complete live deck. O1-O6 replay
NEMO's completed ice-surface stress/flux products. O7 alone replays raw
meteorology and computes the air-ice surface step. O8-O16 replay O7's exact
outputs while admitting their own stage. O17 and all C rungs compute the O7
surface products live. Dynamics are off for O7-O16, so their thermodynamic
admission does not compose D1-D4. This forcing disposition is part of every
record header and a mismatch fails admission.

O8-O16 are admitted from **one** instrumented 100-step ORCA2_SAS_ICE run with
the unmodified production ice namelist. That single master record contains
every named stage boundary; each rung reads only its registered entry/exit
projection. The size in an O8-O16 row is therefore a retained subset of that
one record, not the cost of a different deck or another NEMO run. In
particular, O9 does not turn salinity-4 or ponds off: it replays the production
ZDF/DH/TEMP entry and stops its score at the temperature-stage exit, before
the later production salinity, lateral-melt and pond calls.

| Rung | NEMO deck and keys | Exactly one new piece / legoESM work | Required record, bar, estimated size |
|---:|---|---|---|
| D1 | `tests/ICE_ADV1D`, shipped SAS keys `key_si3 key_xios key_linssh key_vco_1d`; retain the lane-3 Prather overlay, 40 x 2 s. | Close **restart-moment ownership** in the existing `ICEAdv1DCard`, no new algorithm. | Reuse entry/step/restart record; add moment pre/post-correction provenance. Exact local moment replay and `1e-15` full trajectory; about 20 MiB. |
| D2 | `tests/ICE_ADV2D`, same keys; Prather overlay, 485 x 1200 s. | Close the **before-level / `snwice_mass` input contract**, then run the existing 2-D Prather card unchanged. | All extensive fields, five moments per field, odd/even sweep boundary and restart. Exact local replay, `1e-15` trajectory; 4–8 GiB. |
| D3 | ICE_ADV2D_RHG plus `tests/ICE_RHEO`, retaining their lane-3 H79+aEVP overlays. | Diagnose the **legacy carried-stress trajectory debt** only if O5 reuses that one-category periodic arm. RHEO's completed-step-2 first owner is `stress1_i` (`docs/ocean/fidelity/testcases/nemo_testcases_l3dyn_phase2_receipt.md:856-866`), while exact-input aEVP is byte-exact and the long growth is amplification of an fp64-scale carried seed (`docs/ocean/fidelity/testcases/nemo_testcases_l3dyn_phase2_receipt.md:1629-1649`). O5 instead rewrites the arm for five categories and the fold, so this old-arm debt becomes **MOOT**, not a prerequisite to close. | Historical cross-card register only if the old arm is retained; otherwise record the supersession by O5's exact-input admission. |
| D4 | `tests/ICE_RHEO`, after D3's stress carry is controlled. | Diagnose the legacy **redistribution/restart-moment debt** only if the old one-category periodic carry is retained. Its active-window owner is `sxxe_l01` at `3.492459543785742e-9` (`docs/ocean/fidelity/testcases/nemo_testcases_l3dyn_phase2_receipt.md:872-884`). Once O4/O5 replace that Prather/aEVP carry for five categories and the fold, this debt is **MOOT**, not something to close first; O6 then admits the new five-category redistribution separately. | Historical ridge/raft register only if the old arm is retained; otherwise record supersession by O4/O5 and O6's exact-input admission. |
| D5 | `cfgs/C1D/EXP_SASICE`, `key_si3 key_RK3 key_vco_1d3d key_linssh`, lane-3 3+3/salinity-2/pond-off overlay, 8,760 x 3600 s. | Decide the **platform question** first under recommended Decision 1: does the compiled target preserve NEMO's smallest positive binary64 snow value, or does FTZ/DAZ turn it into zero before the snow-present predicate? NEMO carries `5e-324 m`; the largest downstream row is `0.02510099530281747` (`docs/ocean/fidelity/testcases/nemo_testcases_l3thd_phase6_receipt.md:194-209`). | On every target CPU/GPU, record the input/output bit patterns of a compiled smallest-subnormal identity probe, the ZDF `h_s > 0` predicate, and exact ZDF outputs using the recorded operand. If the platform preserves the operand, close the branch literally. If it flushes and literal closure is impossible, declare `D5 PLATFORM_FLOOR`, report the downstream normal rows as unresolved rather than matched, and start O1. |
| O1 | ORCA2_SAS_ICE, dynamics/thermo off, exact NEMO state injected. | Add the **full ORCA2 SI3 state/restart schema** only: jpl5, 10+5 energy/salt, age, pond/lid, five moments and HFN bounds. No SST initialization or thickness fit. | Read/aggregate/restart state and seam/fold halos; exact bridge/restart; about 1 GiB. |
| O2 | Same diagnostic deck, initialized from the recorded SST/SSS and namelist scalars. | Add the **SST/freezing-point and hemisphere-dependent scalar initial state**, including the levitating-ice initial SSH branch. NEMO masks and assigns the scalar state before category fitting (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:309-338`) and adjusts SSH differently for levitating ice (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:440-460`). | SST/SSS, freeze point, mask, scalar ice/snow/pond state and SSH adjustment; about 1–2 GiB. |
| O3 | Same diagnostic deck, O2 scalar state as exact input. | Add the **initial HFN thickness-distribution fit** (`ice_var_itd`) separately from state construction (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:354-364`). | Category fit inputs/outputs, layer initialization and conservation; about 1–2 GiB. |
| O4 | ORCA2_SAS_ICE with Prather only; dynamics/thermo otherwise off. | Add **five-category Prather across the cyclic seam and north fold**, replacing the periodic arm-local halo. D1 and D2 gate composition, but exact-input seam/fold admission can proceed. | Both sweep orders, fold/seam exchanges and every moment; 10–20 GiB. |
| O5 | ORCA2_SAS_ICE diagnostic overlay: `ln_dynALL=T`, `ln_dynRHGADV=F`, `ln_ridging=F`, `ln_rafting=F`, `ln_icethd=F`. | Add **curvilinear tripole H79+aEVP** and no-slip fold/halo ordering. This keeps the production `dynALL` statements; it does not select `dynRHGADV`, whose source also calls non-production `Hpiling` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn.f90:187-201`). Apply the exact-input rule inside the rung: first replay category area/volume into five-category H79 plus smoothing and record strength entry/exit (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icedyn_rdgrft.f90:1077-1114`); then replay NEMO's post-strength field into aEVP and record its entry/exit, including the levitating `ice_var_sshdyn` arm. Both redistribution flags are false, and the record proves no O6 redistribution. D3 is superseded because this is a new five-category fold-aware arm. | Metrics, masks, category area/volume, raw/smoothed strength, levitating SSH operand, selected 100-subcycle frames, velocity/stress and post-advection boundary; 15–30 GiB. |
| O6 | ORCA2_SAS_ICE `dynALL`, thermodynamics off, production ridging and rafting restored. | Add **five-category ridging/rafting plus post-dynamics correction**, including linear redistribution. It consumes O4/O5's new exact-input-admitted carry; D4 is moot unless the old arm is retained. | Participation, transfers, repeats, conservation channels and `ice_cor(kt,1)`; 15–30 GiB. |
| O7 | ORCA2_SAS_ICE bulk surface step; dynamics/thermo state held at exact input. | Add the complete **ice-atmosphere surface step**: `blk_ice_1`, pond/category `ice_alb`, `blk_ice_2`, and `ice_flx_other`. This rung owns computed raw atmospheric forcing and `Cd_ia=Ce_ia=Ch_ia=1.4e-3`; prior O rungs replay its products. | Raw meteorology, coefficient fields, stress, albedo, heat/mass fluxes and lead/bottom terms; 10–20 GiB. |
| O8 | Exact-input frazil-stage view of the one production-deck record. | Add **frazil collection bookkeeping** from the always-called `ice_thd_frazil`, even though collection is disabled. | Frazil inputs, `ht_i_new`, `fraz_frac`; 1–2 GiB retained projection. |
| O9 | Exact-input category-stage view of the same production salinity-4/pond-on record. | Generalize **BL99/P07 ZDF, thickness growth/melt and temperature updates** to jpl5 and 10+5 layers. It does not claim a full thermodynamic trajectory and uses no salinity-1/pond-off overlay. | ZDF/DH/TEMP entry/exits and Picard/tridiagonal state; 20–40 GiB retained projection. |
| O10 | Exact-input production salinity-4 stage from the same record. | Add **VC19/RJW drainage and flushing**. | Layer salt/brine, bounded subcycles, salt flux and temperature reconversion; 10–20 GiB retained projection. |
| O11 | Exact-input `ice_thd_da` replay with default lateral melt. | Add **lateral melt**. | Routine entry/exit and heat/freshwater diagnostics; 2–5 GiB. |
| O12 | Exact-input pond stage replay with production pond selectors. | Add **level ponds, rain, lids and pond albedo state changes**. | Pond/lid growth, drainage/flush/freeze and conservation; 10–20 GiB. |
| O13 | Exact-input post-category replay. | Add **HFN `ice_itd_rem`** with all energy/salt/pond extents. | Boundary crossings, remap fluxes and conservation; 10–20 GiB. |
| O14 | Exact-input post-remap replay. | Add **open-water growth and lead-heat melt** in `ice_thd_do`. | New-ice category choice, heat/freshwater/salt changes; 5–10 GiB. |
| O15 | Exact-input post-open-water replay. | Add **post-thermodynamic `ice_cor(kt,2)`** as its own owner. | All corrected state, zapping/caps and conservation; 5–10 GiB. |
| O16 | Exact-input post-aggregation view of the same production-deck record. | Add **`ice_update_flx`** and its ocean heat, freshwater, salt, cover, temperature and mass outputs. It records `snwice_mass(_b)` but, as verified above, does not claim an `ln_ice_embd` branch absent from this routine. C1 consumes this owned implementation rather than treating it as “no new physics.” | Pre/post aggregation, every returned exchange field and first ocean-format boundary; 5–10 GiB retained projection. |
| O17 | Unmodified ORCA2_SAS_ICE, 100 x 10800 s, `nn_fsbc=1`. | Add **full-deck composition at SAS cadence**; no new operator implementation. This is the first rung that calls the full `ice_thd`, so its unconditional frazil, jpl5 remap, open-water growth and correction have already been admitted at O8/O13/O14/O15 (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd.f90:182-245`). | Complete call-order frames, live O7 forcing, exchange outputs and restart; 40–80 GiB. |
| C0 | Coupled certified ORCA2 card, still replaying NEMO's post-surface exchange fields and drag. | Change **cadence only**, from SAS `nn_fsbc=1` (10,800 s) to coupled `nn_fsbc=2` (21,600 s). The producer also changes: SAS reads prescribed `sst_m/sss_m/ssu_m/ssv_m/ssh_m/e3t_m` (`src/SAS/sbcssm.F90:87-121`), whereas coupled NEMO accumulates and divides the two ocean steps (`src/OCE/SBC/sbcssm.F90:128-173`). Apply the exact-input rule: record the coupled NEMO mean fields and replay those exact fields into both ice implementations; do not compare SAS prescribed fields with live coupled means. The false `ln_ice_embd` ocean-load disposition is recorded at its first coupled consumer. | Coupled mean-field producer output, ice carry on both ocean steps, cadence gate, levitating-load flag, before/after levels and restart; 10–20 GiB. |
| C1 | Same coupled `nn_fsbc=2` card; keep NEMO `rCdU_ice` replay. | Replace **replayed ice heat/freshwater/salt/cover fields** with live O16 `ice_update_flx`; cadence is already owned by C0. | Pre-surface, post-flux and ocean first-consumer; 20–40 GiB. |
| C2 | Same coupled card; Decision 150 drag replay disabled. | Replace **ice stress and `tmod_io/rCdU_ice` replay** with live `ice_update_tau`. | Relative velocity, stress blend, drag and ocean top-stress first consumer every ocean step; exact local, ocean registered bars unchanged; 10–20 GiB. |
| C3 | Same coupled card with `nn_mxlice=2` consuming live ice. | Enable **live under-ice TKE length coupling** using `fr_i` and `hm_i`. | Aggregation -> TKE first-consumer frame and short coupled trajectory; exact local owner and inherited ocean/ice bars; 10–20 GiB. |
| C4 | Same coupled card after C0-C3 admission. | Add **full coupled composition** by removing only the C0 exact-NEMO mean-field replay; the already certified ocean card produces the two-step means live. No new operator is introduced. | Mean producer -> ice first-consumer, complete ice/ocean call order, exchange outputs and restart; inherited bars; 30–60 GiB. |

Rungs D1-D5 reuse the existing lane-3 cards and immutable records rather than
reacquire them unless admission proves the record lacks the discriminating
operand. D3/D4 are historical diagnostics, not mandatory closure work, when
O5/O6 replace their one-category periodic arms. ICE_AGRIF is excluded because
none of the above decks or cpp keys selects AGRIF and no mechanism in the
target call graph depends on it.

O9 cannot be a full-thermodynamics rung followed later by O13: with `jpl=5`
and thermodynamics on, SI3 always executes frazil, `ice_itd_rem`, `ice_thd_do`
and `ice_cor(kt,2)` (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icethd.f90:182-245`).
The stage-replay definition above is deliberate; O17 is the first composition
of those already admitted stages.

C0 similarly does not claim the SAS-to-coupled mean-field producer swap. Its
recorded NEMO `sbc_ssm` output is an exact input to the ice program throughout
C0-C3, so those rungs isolate cadence, flux, drag and TKE respectively. C4 is
the first rung that composes the live, already certified ocean mean producer.

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

### Training risks and reverse-mode memory plan

**Arithmetic floor.** The local-kernel bar is literal bit identity only for
the same registered operands, statement order, and platform arithmetic. Lane
3 measured 16,512/227,760 exact-unequal bulk rows while every row remained at
the `1e-15` bar: 18 were JAX `exp` rows and 16,494 were ordinary binary64
operation-order rows (`docs/ocean/fidelity/testcases/nemo_testcases_l3thd_scalarmath_v2_receipt.md:30-43`).
That receipt supports a libm candidate and written-operation-order debt; it
does **not** establish FMA as the owner, so FMA remains a hypothesis requiring
an isolated compiled-expression discriminator.
Consequently, a candidate is called `BIT_EXACT` only with zero unequal rows.
An exception may be called `PLATFORM_INTRINSIC_FLOOR` only after an isolated
same-input, same-written-expression probe attributes it to the target
libm/compiler fusion or reassociation; the receipt reports backend, device,
JAX/jaxlib/compiler versions, affected intrinsic or expression, exact-unequal
count, maximum ulps and normalized error. It remains AT-BAR rather than
bit-exact, and changing runtime invalidates the classification.

**Fixed-loop tape size.** The 5-category 10+5-layer state must not be retained
at every inner iteration. Implement the literal counts as static `lax.scan`
programs with masked inactive iterations, and wrap scan bodies in
`jax.checkpoint`/`jax.remat` using a nothing-saveable policy so recomputation
changes only the reverse pass, never primal order. Use a nested 10 x 10 scan
for the 100 aEVP subcycles and retain only ten chunk-boundary carries; use
nested rematerialised scans for the 50 drainage x 5 flushing bounds; and use a
fixed maximum masked scan for repeated ridging/rafting, retaining only its
outer carry. BL99's tridiagonal solve uses a custom/implicit transpose rather
than taping every Picard temporary. Before training admission, compile
`value_and_grad` for one O17/C4 ice step and a representative multi-step
segment on the target accelerator, publish peak device memory and recompute
factor, and compare them with a preregistered device budget. The acceptance
plant disables rematerialisation and must increase the measured peak.

**Denormals are platform-specific.** D5's CPU result cannot be inherited by a
GPU training run. Every admitted CPU and accelerator tuple runs the smallest-
subnormal identity/predicate probe specified in D5 and stamps FTZ/DAZ behavior
in its receipt. A preserving platform must close the literal snow-present
branch. A flushing platform reports `D5 PLATFORM_FLOOR`, keeps all downstream
normal differences visible, and proceeds to O1; no per-statement subnormal
workaround is allowed.

## 6. Decisions before rung 1

| Decision | Numbered options | Pick |
|---:|---|---|
| 1. First implementation track — **DECISION NEEDED** | **1)** classify D5 on each target platform, close it literally where possible or declare the stated FTZ/DAZ floor, then start O1; D1-D2 gate reused Prather work and D3-D4 are moot if O5/O6 replace the old arms; **2)** close every D1-D5 legacy debt first; **3)** start O1 now without the D5 platform discriminator. | **1.** The 0.025 C1D row is a real downstream branch mismatch, but the deciding operand may be unrepresentable in XLA on the target platform. The explicit D5 measurement now distinguishes a code debt from a platform floor. Per-stage exact-input replay still assigns later first owners; O7-O16 keep dynamics off; and exact-input aEVP is bit-exact with its large endpoint measured as amplification of an fp64-scale seed (`docs/ocean/fidelity/testcases/nemo_testcases_l3dyn_phase2_receipt.md:1629-1649`). Closing obsolete one-category trajectory debts is not required before five-category fold-aware replacements. The pick remains option 1 and is still a recommendation, not a decision for the user. |
| 2. Shipped-case identity | **1)** reuse the lane-3 Prather/aEVP arithmetic but replace its jpl1/periodic wrappers at O4/O5, using O5's explicit production-`dynALL` overlay; **2)** certify shipped UMx/EAP defaults first. | **1.** Extend the established shared implementation to five categories and the fold without selecting the `dynRHGADV`/`Hpiling` call graph that production ORCA2 does not execute. |
| 3. Bars | **1)** bit-exact local replay except a proved, platform-stamped intrinsic/denormal floor, plus the fixed `1e-15` normalized trajectory bar; **2)** choose new mechanism-specific bars now. | **1.** It preserves lane-3 comparability without misreporting the measured 16,512 exact-unequal rows as bit identity. Any floor or later statistical bar requires preregistered evidence, never retroactive relaxation. |
| 4. ORCA2_SAS staging | **1)** take O8-O16 as stage projections from one instrumented production-deck run, then full composition O17; **2)** acquire a separate modified deck per rung; **3)** jump to the unmodified full deck without stage admission. | **1.** One production identity gives every unconditional thermodynamic child and `ice_update_flx` a first owner without multiplying acquisition cost or introducing O9's former salinity-1/pond-off deck. |
| 5. Record density / storage | **1)** every outer stage and step, sparse selected aEVP/salinity iterations, full detail at first divergence; **2)** dump every inner iteration. | **1.** It retains attribution while avoiding multi-terabyte records. |
| 6. Coupling order | **1)** cadence C0 under exact NEMO coupled-mean replay, fluxes C1, drag C2, under-ice TKE C3, then remove mean replay at composition C4; **2)** change cadence, mean producer and all exchange replay together. | **1.** The 10,800 -> 21,600 s cadence, SAS-to-coupled mean producer, Decision 150 drag and TKE remain independently falsifiable. |
| 7. Forward/gradient policy | **1)** literal primal with gradient-only guards; **2)** smooth forward switches. | **1.** Forward fidelity is the target and JAX purity is retained with explicit custom derivatives. |
| 8. AGRIF | **1)** exclude; **2)** add ICE_AGRIF. | **1.** No target selector or key needs it. |
| 9. ORCA2_SAS data source — **DATA-SOURCE DECISION, deferred until acquisition** | **1)** fetch official `SAS_v5.0.0.tar.gz` from the versioned NEMO JASMIN SETTE endpoint and pin raw/member hashes; **2)** generate replacement `sas_grid_*` fields and pin that producer. | **1.** It is the exact shipped-case dependency named by SETTE. No fetch is authorized by this design round. |
| 10. Atmospheric forcing boundary | **1)** replay completed surface products through O6, compute raw meteorology only at O7, replay O7 outputs for O8-O16, then compose live at O17/C; **2)** compute live on every O rung. | **1.** It gives the surface producer one owner while keeping the other exact-input admissions attributable. |

## Validation and handoff

This round changed documentation and its citation registry only. No oracle
artifact was produced or requested. The independent Fable concept review was
largely SOUND; each proposed gap was verified and fixed or rebutted above.
Codex review: operator.
The implementation round must not start until Decision 1 is answered.
