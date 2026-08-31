# ORCA1-OMIP coverage matrix

**Audit date:** 2026-08-31
**Target:** Alexis Barge ORCA1 configuration, NEMO 5.0.1, paper
[10.1029/2024MS004824](https://doi.org/10.1029/2024MS004824)
**Target commit:** `5babc1e0de584f34e35a686182d0270b50f5643b`
**legoESM commit audited:** `de1426ea315d8ecb9f43115eaa05a5c6b7bc6e55`
**Machine-readable authority:**
[`omip_orca1_coverage_matrix.json`](omip_orca1_coverage_matrix.json)

## Verdict

The checked-in production deck is **not an extension of the DINO-certified
configuration**. It changes the outer time-integration family from MLF to
`key_RK3`, activates SI3 and ISF, changes S-EOS to TEOS-10, introduces a real
tripolar partial-cell domain, and replaces analytic forcing with climatological
CORE-II file forcing. Those are composition boundaries: a DINO-certified kernel
does not certify its ORCA1 execution order, geometry, or forcing.

Of 48 coverage rows, **2 are CERTIFIED**, **32 PRESENT-UNCERTIFIED**, **12 ABSENT**,
and **2 N/A-INFRA**. The blocking structural gaps are:

1. **Whole-step `key_RK3` (XL).** legoESM has NEMO-like Wicker-Skamarock
   momentum stages and a separate generic tracer RK3, but not the single NEMO
   stage program that composes QCO, tracers, momentum, barotropic updates,
   LDF/ZDF, surface forcing, SI3, and ISF. DINO ran MLF. The GYRE audit certifies
   momentum `rk3_ws` while explicitly leaving tracer stepping unmatched
   ([GYRE wiring, lines 32–40](nemo_gyre_wiring_comparison.md#2-time-stepping)).
2. **SI3 (XXL).** legoESM has a real prognostic ice subsystem, not an SI3
   implementation. Exact selected gaps include Prather transport, integrated
   three-ice/three-snow-layer BL99 thermodynamics, SI3 varying salinity,
   Lemieux-2016 landfast ice, and the SI3 ocean-exchange recipe.
3. **DOMAINcfg partial cells and global fold (L).** eORCA loading, partial cells,
   and north-fold-aware operators exist, but DINO's idealized full-step grid
   certified none of them. The campaign already records the decisive known
   debt: DINO's reconciliation uses an average at a face where DOMAINcfg partial
   cells require the shallower-neighbour minimum
   ([outstanding debt, D2.5](dino_outstanding_fidelity_debt.md)).
4. **Climatological forcing (L).** The OMIP driver contains CORE-II sampling,
   NCAR bulk fluxes, runoff, SSS restoring, RGB/CHLA, and diurnal shortwave
   components. DINO certified analytic forcing only; its `tra_sbc` application
   and two-band `tra_qsr` redistribution remain explicitly unmeasured
   ([round 94, lines 29–37](dino_full_step_coverage_round94_result.md#residual-active-and-waived-table)).
5. **`key_isf` composition (L).** The exact selected `spe` prescribed-melt
   loader/deposition path is present, so this is not a blanket absence. It has
   no oracle receipt, its RK3 placement is untested, and general overhanging
   cavity geometry remains deferred.

This is an audit only. No model, GPU, or MPI run was performed and no physics
was changed.

## Status semantics

| Status | Exact meaning |
|---|---|
| `CERTIFIED` | legoESM has the selected requirement and an applicable DINO campaign receipt. The row's scope is part of the verdict. |
| `PRESENT-UNCERTIFIED` | Code and/or production wiring exists, but no applicable DINO oracle receipt certifies the selected ORCA1 path. |
| `ABSENT` | The selected executable recipe is absent. Similar components do not count. |
| `N/A-INFRA` | XIOS, output, restart, diagnostics, timing, or MPP rather than physics. |

“Present” is intentionally weaker than “faithful.” Comments saying “faithful
port,” unit tests, earlier ORCA climate runs, and the separate GYRE comparison
are presence/readiness evidence only. DINO's bar remains exact correlation 1.0
and ratio 1.0; a near match is not promoted. Round 94's `COVERED` also means a
row was measured and dispositioned, not automatically bit-exact
([round 94, lines 14–27](dino_full_step_coverage_round94_result.md#gate-provenance-and-verdict)).

## Pinned configuration and resolution limits

The five inputs used to resolve the deck are pinned in the JSON by SHA-256.
The effective order is `namelist_cfg` over `namelist_ref`, and
`namelist_ice_cfg` over `namelist_ice_ref`. The compiled keys are exactly:

```text
key_si3 key_xios key_qco key_vco_1d3d key_RK3 key_isf
```

Source: `cpp_ORCA1.fcm:1` at the target commit.

Two limits are observable in the source clone and must not be hidden:

- The clone contains no `INPUTS/domain_cfg.nc` or forcing payloads. `deploy.sh`
  downloads a Zenodo archive plus CORE normal-year forcing and links those
  files into the run directory (`deploy.sh:11-34,86-91`). Therefore the audit
  resolves every selector and declared input record exactly, but cannot claim
  byte identity for absent NetCDF payloads. In particular, `ln_isfcav` is
  carried by `domain_cfg.nc`, so cavity-open activation is **UNRESOLVED**;
  `key_isf` and the `namisf` overlay do not resolve that domain flag.
- `EXPREF/namelist_cfg:456` is a bare line of hyphens without a Fortran `!`
  comment marker. It lies between `namzdf_tke` and `namzdf_iwm`. This audit
  records the byte-level anomaly but makes no runnability claim because the
  user prohibited a run.

The reference source available locally is NEMO 5.0.2. It was used only to
understand control flow where needed; the target configuration and selections
come exclusively from the pinned NEMO 5.0.1 configuration repository.

## Resolved active ocean deck

The JSON contains typed effective values for all 34 blocks present in the
ocean cfg. This table is the compact scheme view; “ref” identifies selectors
inherited because the cfg is silent.

| Block | Effective active selection/value |
|---|---|
| `namrun` | ORCA1; steps 1–350400; date 20000101; cold start; output every 720; no periodic restart write |
| `namdom` | `rn_Dt=3600`; `ln_shuman=T`; no mesh-mask output |
| `namcfg` | `ln_read_cfg=T`; `cn_domcfg='domain_cfg'` |
| `namtsd` | WOCE monthly T/S climatology initializes; T/S damping input off |
| `namsbc` | every 4 steps; bulk on and user/flux/ABL/coupled modes off; `nn_ice=2`; levitating ice; qsr, dm2dc, SSS restoring, fwb=1, runoff on; pressure/waves off; input land mask option 0 |
| `namsbc_blk` | NCAR only; zTq=zU=10 m; 5 iterations; skin corrections/current feedback off; pfac=efac=1; specific humidity, absolute temperature, precipitation in m; constant ice-air Cd/Ce/Ch=1e-3; alternate ice drag laws off; all nine CORE-II FLD_N records typed in JSON |
| `namtra_qsr` | RGB; `nn_chldta=1`; monthly ESACCI `CHLA` |
| `namsbc_ssr` | `nn_sssr=2`; −220 mm/day piston, bounded at 4 mm/day; no restoring under ice; monthly SSS |
| `namsbc_rnf` | monthly river runoff; depth initialized from field; max .05, cap 150 m; mouth/direct depth/T/S and iceberg (`ln_rnf_icb=F`) branches off; coastal mask static |
| `namisf` | on; parametrized `spe` melt on; cavity melt/coupling off; uniform load if the unresolved domain cavity flag is true; zmin/zmax/fwf records at `freqh=-12` (one per 12 months) |
| `namlbc` | `rn_shlat=2`, no-slip |
| `namdrg` | nonlinear and implicit; implicit ice drag; OFF/linear/log-layer false (ref) |
| `namdrg_bot` | Cd0=1e-3, Cdmax=0.1, ke0=2.5e-3, z0=3e-3; regional boost false (ref) |
| `nambbc` | geothermal on; option 2 variable field |
| `nambbl` | BBL on; diffusive off; advective option 2; Aht=1000 m²/s; gamma=20 s |
| `nameos` | TEOS-10 on; EOS-80 and S-EOS off (ref) |
| `namtra_adv` | FCT, horizontal=2, vertical=2, implicit treatment 1; OFF/CEN/MUSCL/UBS/QUICKEST false (ref) |
| `namtra_ldf` | laplacian standard isoneutral + MSC; other directions/operators off; slope cap .01; coefficient option 21; Ud=.018 m/s, Ld=100 km |
| `namtra_mle` | on; inherited `nn_mle=1`, ce=0.06, lat=20°, MLD U/V=min, convection gate off, rho criterion 0.01 |
| `namtra_eiv` | on; option 21 Treguier; Ue=0.018 m/s, Le=100 km; GEOMETRIC EKE off (ref) |
| `namtra_dmp` | off |
| `nam_vvl` | z-star; z-tilde/layer off; face interpolation option 2 (ref) |
| `namdyn_adv` | vector form; Hollingsworth KE gradient (`nn_dynkeg=1`) |
| `namdyn_vor` | EEN; `nn_e3f_typ=0`; vorticity masking off (ref) |
| `namdyn_hpg` | standard s-coordinate Jacobian (`hpg_sco`); `hpg_isf` off (ref) |
| `namdyn_spg` | split explicit; auto substeps, cmax=0.8; inherited `ln_bt_fw=T`, `nn_bt_flt=1`, alpha=0 |
| `namdyn_ldf` | div-rot (`nn_dynldf_typ=0`) level laplacian; coefficient option −30 from `eddy_viscosity_3D.nc` |
| `namzdf` | adaptive implicit vertical advection; TKE; EVD tracer-only (`nn_evdm=0`, 100 m²/s); IWM; DDM off; backgrounds 1.2e-4/1.2e-5 |
| `namzdf_tke` | `nn_mxl=2`; surface anchor with minimum .04 m; under-ice option 2; NIW option 1, fraction .08 and decay type 1; Langmuir .25; ice attenuation option 3; `rn_bshear=1e-20`; wave-height scaling off; remaining constants typed in JSON |
| `namzdf_iwm` | constant efficiency; no differential T/S; six climatological fields at `freqh=-12` (one per 12 months) |
| `namhsb` | online heat/salt budget diagnostic on |
| `namnc4` | chunks 1×1×75; compression on |
| `nammpp` | inherited `jpni=0`, `jpnj=0` (automatic); no-gather/delayed communication on; halo 2; comm option 1 |
| `namctl` | timing on; CFL diagnostics off |

Sources: `EXPREF/namelist_cfg:34-527` and the inherited blocks in
`EXPREF/namelist_ref:810-841,878-1262,1339-1354,1510-1534`. The commented
`nn_bt_flt=3` and `rn_bt_alpha=.07` in the cfg are not active; the active
reference values are filter 1 and alpha 0.

## Resolved active SI3 deck

The JSON contains effective values for all 16 ice cfg blocks, including
inherited choices. The active recipe is:

| Block | Effective active selection/value |
|---|---|
| `nampar` | 1 category; 3 ice layers; 3 snow layers; dynamics and thermodynamics on; virtual ITD off |
| `namitd` | HFN categories; expected mean 2 m; min thickness .05 m; max 99 m (ref) |
| `namdyn` | full dynamics; no RHGADV-only mode; no-slip ice LBC (ref); Lemieux-2016 landfast on |
| `namdyn_rdgrft` | H79 strength, pstar=20000, C=20; no smoothing; exponential redistribution (`murdg=3`) and participation (`astar=.03`); ridging+rafting on (`hstar=25`, `hraft=.75`, `craft=5`); snow/pond survival .5; ridge porosity 0 |
| `namdyn_rhg` | EVP; EAP/VP off; 100 EVP subcycles and relaxation .333 (ref) |
| `namdyn_adv` | Prather on; Ultimate-Macho off |
| ice `namsbc` | Cd_io=.005; CICE snow fraction option 2; no heat-flux redistribution; no conduction-flux BC; GM77 transmitted solar (all ref) |
| `namthd` | thickness growth/melt and open-water growth on (ref); lateral melt off; lead heat melts ice first (ref) |
| `namthd_zdf` | BL99; P07 conductivity; 3-layer solve; snow conductivity .5; convergence check off |
| `namthd_da` | inactive because lateral melt is off |
| `namthd_do` | new ice thickness .05 m; frazil off (ref) |
| `namthd_sal` | `nn_icesal=2`, varying salinity; flushing/drainage on; salinity and ice-thickness minima .1; new-ice fraction .75; option-2 restoration constants active. Option-4-only subcycle/formulation controls are not selected. |
| `namthd_pnd` | melt ponds off |
| `namini` | initialization on; file option 1; `at_i`, `ht_i`, `ht_s`, `sm_i`, `tmsu` climatology at `freqh=-12` (one record per 12 months); `tm_i`/`tm_s` records say `NOT USED` |
| `namalb` | inherited dry/melt snow .85/.75; dry/melt ice .64/.53; pond .18/.30; pivotal thickness 1 m |
| `namdia` | online ice budget check and debug control off |

Sources: `EXPREF/namelist_ice_cfg:22-199` over
`EXPREF/namelist_ice_ref:22-324`.

## Coverage matrix

The “receipt/scope” column is deliberately terse; full code and receipt arrays
are in the JSON. A certified row never silently certifies the ORCA1 composition
named as an exclusion.

| ID | ORCA1 requirement | Disposition | Receipt / precise boundary |
|---|---|---|---|
| CPP-SI3 | SI3 top-level orchestration | **ABSENT** | DINO compiled without ice; generic ice is not SI3 |
| CPP-XIOS | XIOS | **N/A-INFRA** | output plumbing |
| CPP-QCO | QCO z-star/VVL core | **PRESENT-UNCERTIFIED** | narrow formula match, but production DINO r3t/r3u/r3v remains DEBT; ORCA adds partial/RK3/ISF composition |
| CPP-VCO-1D3D | 1-D/3-D vertical-coordinate dispatch | **PRESENT-UNCERTIFIED** | DINO compiled `key_vco_3d`, not `key_vco_1d3d` |
| CPP-RK3 | whole NEMO RK3 stage program | **ABSENT** | DINO was MLF; separate momentum/tracer integrators are not the whole step |
| CPP-ISF | key_isf geometry and step integration | **PRESENT-UNCERTIFIED** | `spe` exists; general cavity geometry/RK3 composition open |
| DOM-DOMAINCFG | real eORCA DOMAINcfg ingestion | **PRESENT-UNCERTIFIED** | reader exists; input payload absent; DINO geometry not applicable |
| DOM-PARTIAL | partial cells and face min rules | **PRESENT-UNCERTIFIED** | explicit D2.5 average-versus-min debt |
| DOM-TRIPOLE | multi-basin/north-fold global grid | **PRESENT-UNCERTIFIED** | tripole implementation/status document; no DINO oracle |
| DOM-LBC | no-slip `rn_shlat=2` | **PRESENT-UNCERTIFIED** | DINO certified free-slip `rn_shlat=0` |
| DOM-TS-INIT | WOCE monthly T/S initialization | **PRESENT-UNCERTIFIED** | loaders exist; different DINO IC |
| SBC-FLDREAD | climatology sampling/interpolation/regridding/rotation | **PRESENT-UNCERTIFIED** | production path exists, not a general certified fldread clone |
| SBC-NCAR | NCAR/Large-Yeager bulk fluxes | **PRESENT-UNCERTIFIED** | implementation/unit oracle exists; DINO analytic forcing only |
| SBC-ICE | `nn_ice=2` levitating SI3 exchange | **ABSENT** | generic tile coupling lacks SI3 state/order/budgets |
| SBC-QSR-RGB | RGB + monthly CHLA | **PRESENT-UNCERTIFIED** | port exists; DINO two-band redistribution still unmeasured |
| SBC-DM2DC | diurnal reconstruction | **PRESENT-UNCERTIFIED** | wired for tripole/lat-lon; DINO off |
| SBC-SSS | bounded SSS restoring under ice | **PRESENT-UNCERTIFIED** | implementation exists; DINO off |
| SBC-FWB | per-step global E-P-R correction | **PRESENT-UNCERTIFIED** | global/ice/runoff composition untested |
| SBC-RUNOFF | river runoff and 150 m depth rule | **PRESENT-UNCERTIFIED** | loader/deposition exists; iceberg record inactive (`ln_rnf_icb=F`); DINO off |
| SBC-ISF-SPE | prescribed ISF `spe` deposition | **PRESENT-UNCERTIFIED** | exact selected path exists; no oracle receipt |
| BOT-DRAG | implicit nonlinear bottom drag | **CERTIFIED** | bit-exact coefficient/index and negligible assembled difference; excludes ice/RK3 composition |
| BOT-GEOTHERMAL | variable geothermal heat flux | **PRESENT-UNCERTIFIED** | implementation exists; DINO off |
| BOT-BBL | advective BBL option 2 | **PRESENT-UNCERTIFIED** | implementation exists; DINO off; OVERFLOW receipt needed |
| EOS-TEOS10 | TEOS-10 | **PRESENT-UNCERTIFIED** | DINO selected S-EOS |
| TRA-FCT2 | FCT2 tracer advection | **PRESENT-UNCERTIFIED** | DINO full 3-D corr 0.9945, vertical mass-flux clipping open; RK3 stage use differs |
| TRA-REDI-MSC | standard isoneutral Redi + MSC core | **PRESENT-UNCERTIFIED** | round-93 ladder retained S `zfw` Rule-1b; coefficient/RK3 separate |
| TRA-AHT21 | time-varying Redi diffusivity option 21 | **ABSENT** | option-21 machinery found is GM/EIV, not the selected Redi coefficient path |
| TRA-MLE | MLE option 1 | **PRESENT-UNCERTIFIED** | port exists; no applicable DINO receipt located |
| TRA-EIV | Treguier EIV option 21 | **PRESENT-UNCERTIFIED** | DINO meridional EIV transport remains DEBT; option/fold/RK3 differ |
| DYN-ADV | Hollingsworth KE-gradient kernel | **CERTIFIED** | current gate: corr=1, ratio=1, byte-exact, per-element error 3.578e-19; other terms/fold/RK3 excluded |
| DYN-EEN-E3F0 | EEN with `nn_e3f_typ=0` | **PRESENT-UNCERTIFIED** | DINO used type 1 and fresh EEN score remained just off bar |
| DYN-HPG-SCO | standard-Jacobian HPG | **PRESENT-UNCERTIFIED** | current DINO meridional HPG score remains DEBT; partial/ISF/RK3 differ |
| DYN-SPG | split-explicit, forward, filter 1 | **PRESENT-UNCERTIFIED** | DINO used filter 2; old inventory explicitly lacked substep oracle |
| DYN-LDF | file-based option −30 3-D momentum viscosity | **ABSENT** | current loader collapses `ahmf_3d` to a surface zonal-median profile |
| ZDF-AIMP | adaptive-implicit vertical momentum advection | **PRESENT-UNCERTIFIED** | DINO off |
| ZDF-TKE | full ORCA TKE card | **PRESENT-UNCERTIFIED** | shared DINO pieces, but nn_mxl/ice/TEOS/IWM/RK3 composition differs |
| ZDF-EVD | tracer-only EVD | **PRESENT-UNCERTIFIED** | DINO used tracer+momentum EVD |
| ZDF-IWM | climatological internal-wave mixing | **PRESENT-UNCERTIFIED** | kernel/forcing present; DINO off |
| ICE-ITD | SI3 single-category HFN/remap | **ABSENT** | lego ITD is explicitly simplified, not the exact SI3 recipe |
| ICE-LANDFAST | Lemieux-2016 landfast | **ABSENT** | no implementation found |
| ICE-RIDGING | exact SI3 H79/exponential ridging+rafting | **ABSENT** | ridging analogue exists, but no rafting or exact combined recipe |
| ICE-EVP | SI3 EVP/100 subcycles on eORCA tripole | **ABSENT** | generic EVP exists, but target-tripole dynamics degrades to free drift |
| ICE-PRATHER | Prather ice transport | **ABSENT** | active lego transport is not Prather |
| ICE-THERMO | SI3 3+3-layer BL99/P07 | **ABSENT** | BL99 module is parked under `_future`, not integrated |
| ICE-SALINITY | SI3 option-2 layered varying salinity | **ABSENT** | lego tracks bulk category salinity |
| ICE-INIT | SI3 aggregate file initialization | **PRESENT-UNCERTIFIED** | loader exists; evolution semantics differ |
| ICE-ALBEDO-SBC | SI3 snow/albedo/leads/exchange | **PRESENT-UNCERTIFIED** | generic analogues exist; exact formulas/order unmatched |
| INFRA-HSB-NC4-MPP | diagnostics/output/restart/MPP | **N/A-INFRA** | no physics claim |

Key DINO evidence used for dispositions:

- Full-step registry: 34/37 active calls measured, with `ldf_dyn`, `tra_sbc`,
  and `tra_qsr` still unmeasured
  ([round 94](dino_full_step_coverage_round94_result.md)).
- HPG's current meridional exact-bar debt from the authoritative
  [`fidelity_bar_gate.py`](../../../scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py),
  with selector resolution cross-checked against the
  [step-chain inventory](dino_step_chain_coverage.md).
- Redi/MSC ladder closure with its retained Rule-1b qualification
  ([round 93](dino_split_explicit_momentum_chain_round93_result.md)).
- Bottom-drag coefficient/index/application evidence and vector/EEN context
  ([campaign synthesis, §2.4](dino_campaign_synthesis.md)).

## Ranked roadmap, constrained by the NEMO testcase sequence

Priority means “must be closed before claiming this target configuration,” not
an authorization to implement it in this audit.

| Order | Required gate | Coverage retired | Effort | Exit condition before proceeding |
|---:|---|---|---|---|
| 1 | **OVERFLOW + LOCK_EXCHANGE** | partial-cell min rule, TEOS-10 density/BN2, FCT monotonicity/RPE, adaptive vertical advection, BBL | L | Exact geometry/transport/tendency receipts and conservation controls with planted failures |
| 2 | **GYRE** | whole `key_RK3`, tracer stages, barotropic filter/stage composition, TKE placement | XL | One integrated RK3 step program matches NEMO stage dumps; momentum-only success is insufficient |
| 3 | **SI3 cases** | HFN/ITD, BL99 thermo, layered salinity, Prather, EVP, ridging/rafting, landfast, file initialization, albedo/surface fluxes, ocean exchange | XXL | Ascending SI3 testcase receipts plus heat/freshwater/salt conservation, then coupled ocean-ice step |
| 4 | **ORCA2** | DOMAINcfg, tripole/fold, multi-basin LBC, fldread/NCAR, runoff/SSS/RGB/IWM/geothermal/BBL/ISF | XL | Lower-cost global integration has term and budget receipts with exact forcing/calendar provenance |
| 5 | **ORCA1** | final configuration composition | XXL | Run the exact pinned keys, overlays and payload hashes; issue per-term, conservation and climate receipts |

Within that forced ordering, OMIP criticality ranks the work as:

1. full `key_RK3` orchestration — XL, blocking;
2. SI3 and SI3-ocean exchange — XXL, blocking;
3. real partial-cell rules — L, blocking;
4. tripole/north-fold/multi-basin composition — L, blocking;
5. fldread + CORE-II/NCAR forcing — L, blocking;
6. `key_isf`/`spe` composition — L, high;
7. FCT vertical flux and RK3 staging — L, high;
8. TEOS-10/rab/bn2 — M, high;
9. ORCA TKE card composition — L, high;
10. RGB/CHLA, runoff, SSS and freshwater correction — M each, medium-to-high.

The sequence prevents a tempting but invalid shortcut: an ORCA1 climate run
before the testcase receipts would combine every structural gap at once and
could only demonstrate stability or aggregate similarity, not coverage.

## Audit non-claims

- No input NetCDF was downloaded or hashed.
- No NEMO or legoESM executable was run.
- No GPU or MPI command was used.
- No configuration choice, tolerance, or fidelity bar was changed.
- No GYRE or older ORCA climate result was promoted to `CERTIFIED`; only the
  DINO receipts named by a row can do that under this audit's requested rule.
- Presence of an implementation is not a claim of exact scheme identity,
  coupled ordering, conservation under this deck, or statistical equivalence.
