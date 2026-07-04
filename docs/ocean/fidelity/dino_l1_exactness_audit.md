# DINO Level-1 exactness audit — legoESM NEMO recipe vs the real NEMO DINO_R1 run

Campaign directive (2026-07-02): DINO (Kamm et al.) is the fidelity testbed.
**Level 1**: the legoESM NEMO recipe must work EXACTLY like the real NEMO run —
exact NEMO numerics, no divergence to approximate choices; verified against a
NEMO run we execute ourselves. **Level 2**: the MITgcm / Oceananigans / Veros
recipes run the same DINO setup; divergence mapped; broad large-scale metrics
expected to agree.

## The oracle

* Source/config: `nemo_orca1/nemo_5.0.1/tests/DINO_R1` (MY_SRC identical to
  `tests/DINO`); the RUN's own `EXP00/namelist_cfg` is authoritative.
* Reference run: 4 years complete (job `nemo_dino_8690287`, 12 ranks + XIOS,
  finished 2026-06-30): `EXP00/DINO_1m_grid_{T,U,V,W}.nc` (48 monthly records:
  toce/soce/e3t/rhop + velocity grids), `DINO_10d_grid_T.nc`, restarts every
  year, `domain_cfg_out_*.nc`.
* Build keys: `key_xios key_qco key_vco_3d` — **NO `key_RK3`** ⇒ the oracle
  integrates with **MLF (leapfrog + Robert–Asselin)** and the quasi-Eulerian
  z\* (`key_qco`).

## Oracle configuration (DINO_R1/EXP00/namelist_cfg + MY_SRC)

### Domain / grid / calendar
| item | value |
|---|---|
| grid | 1° Mercator, lat −70..70, lon 0..50, i-periodic channel −65..−45 (slope 1.5°), free slip (`rn_shlat=0`) |
| vertical | 36 levels, **pure z** (`ln_zco`, no partial steps), stretched tanh: `dzmin=10 m, kth=35, hco=1000 m, acr=10.5`, H≈4000 m |
| bathymetry | bowl-cosh (`nn_botcase=1`, `hborder=2000 m`, `distLam=3°`), **Drake sill ON** (2500 m deep, 4° wide), mid-ridge OFF |
| calendar | **360-day year** (`nn_leapy=30`), `rn_Dt=2700 s`, run = 46080 steps = 4 y, cold start from rest |
| time stepping | **MLF leapfrog + Asselin** (no key_RK3), `key_qco` z\* |

### Numerics
| block | oracle |
|---|---|
| EOS | **S-EOS**: `a0=0.165, b0=0.76554, λ1=0.06, λ2=0, μ1=1.497e-4, μ2=0, ν=0` |
| tracer advection | **FCT 2nd/2nd** (`ln_traadv_fct`, `nn_fct_h=2, nn_fct_v=2`) |
| tracer lateral diff | laplacian **iso-neutral (standard)** + **MSC** (`ln_traldf_msc`), `rn_slpmax=0.01`, coefficient form `nn_aht_ijk_t=20` with `rn_Ud=0.027 m/s` (aht = Ud·Δ/2, grid-size scaling) |
| eddy-induced velocity | **OFF** (`ln_ldfeiv=.false.`) — no GM at R1 |
| momentum advection | **vector form** (`ln_dynadv_vec`) + **Hollingsworth** KEG (`nn_dynkeg=1`) |
| vorticity | **EEN** (`ln_dynvor_een`, `nn_e3f_typ=1`) |
| pressure gradient | `ln_hpg_sco` (standard s-coordinate jacobian, on the stretched-z grid) |
| free surface | **split-explicit**, `ln_bt_fw=.false.` (**CENTRED** barotropic integration), `nn_bt_flt=1`, `ln_bt_auto` with `rn_bt_cmax=0.8`, `rn_bt_alpha=0` |
| momentum lateral diff | laplacian **iso-level**, `nn_ahm_ijk_t=20`, `rn_Uv=0.27 m/s` (ahm = Uv·Δ/2) |
| vertical physics | **TKE** (`nn_etau=1`; all other TKE params = namelist_ref defaults: `rn_ediff=0.1, rn_ediss=0.7, rn_ebb=67.83, nn_pdl=1, nn_mxl=3 (ln_mxl0=T, rn_mxl0=0.04), ln_lc=T rn_lc=0.15, rn_efr=0.05, nn_htau=1`) + **EVD** (`nn_evdm=1`, `rn_evd=100`); backgrounds `avm0=1.2e-4, avt0=1.2e-5` |
| bottom drag | `ln_non_lin` (namdrg_bot ref defaults: `Cd0=1e-3, ke0=2.5e-3`) — the P4 `nemo_quadratic` scheme |
| solar penetration | `ln_qsr_2bd` (2-band), constant Chl |

### Analytical forcing (`MY_SRC/usrdef_sbc.F90`, `nn_forcingtype=4`, annual cycle ON, diurnal OFF, `rn_emp_prop=0` ⇒ **emp = 0**)
* Seasonal phases (360-day year): `cos_sais1 = cos(π(t−21 Jun)/(half year))`
  (solar), `cos_sais2` same with **21 July** phase (T\*, one-month lag).
* Wind (zonal only): smoothstep interpolation
  `v(s) = v_s + (v_n−v_s)(3−2s)s²` between nodes
  `φ = [−70, −45, −15, 0, 15, 45, 70]`,
  `τ = [0, 0.2, −0.1, −0.02, −0.1, 0.1, 0]`; `taum = |utau|`, **×1.3 where
  `utau>0`** (westerly boost, feeds TKE only).
* T\* (Munday): south `T*_s + (27−T*_s)·sin(π(φ+70)/140)` /
  north analog; seasonal `T*_s = −0.5 − 0.5·c2`, `T*_n = 5.0 + 3.0·c2`;
  `qtot = −40·(SST − T*)`.
* Solar: `qsr_dayMean = max(230·cos(π(φ − 23.5·c1)/180), 0)`;
  `qns = qtot − qsr_dayMean`.
* Salt: **real salt flux** `sfx = −3.858e-3·(SSS − S*)`, S\* Munday cosine
  (`35/35.1/37.25` s/n/eq) + equatorial dip `−1.25·exp(−φ²/7.5²)`; emp = 0.
* IC (`nn_initcase=4`): analytic tanh T/S profiles (case-1) blended linearly
  in latitude toward uniform bottom values (usrdef_istate.F90:129-175); rest.

## legoESM-side status (gap table, 2026-07-02 survey)

| # | oracle item | legoESM status | Level-1 action |
|---|---|---|---|
| 1 | **MLF leapfrog + Asselin** time stepping | **MISSING** — lat-lon integrator dispatch = {euler, rk3} only | **The structural long pole**: implement `momentum_time_integrator="mlf"` (3-time-level carry, Robert–Asselin filter, NEMO operator placement, qco-consistent thicknesses) |
| 2 | Seasonal cycles (`ln_ann_cyc=T`): T\* asymmetric (−0.5·c2 south / +3.0·c2 north, 21-Jul phase), qsr declination 23.5·c1 (21-Jun phase) | **MISSING** — dino.py forcing is ANNUAL-MEAN only (`dino_T_star_annual_mean`, `dino_Q_sr_annual_mean`, trapezoid-averaged) | Add the time-dependent forcing path (360-day calendar phases, exact usrdef_sbc formulas); keep annual-mean as the historical option |
| 3 | FCT **2nd/2nd** tracer advection | **DONE (b540eff19)** — `tracer_advection="fct2"` = centred-2 high flux (h+v, no pre-clamp) + sign-split Zalesak, verified vs traadv_fct.F90:183-263; r1_exact preset selects it. REMAINING: MLF time-levels (upwind at Kbb over 2·dt) — ladder step 4 | — |
| 4 | `hpg_sco` standard jacobian PGF | **DONE (2 commits, codex r3+r4)** — DINO_R1 is ln_zco (FLAT masked z-levels, NOT terrain-following): `vertical_coordinate="masked_zco"` builds the exact mi96 ladder (jpk convention: 35 wet cells; ANALYTIC t-depths, verified vs reference-run deptht to 1.5e-4 m) with full-cell snap per zgr_msk_top_bot; on flat wet levels the sco slope-correction term is identically zero, so NEMO's PGF = plain gradient of the TRAPEZOID p' — `pgf_quadrature="nemo_trapezoid"` (e3w from the t-ladder per depth_to_e3, e3w(1)=2·gdept(1)). No separate sco-jacobian kernel needed. REMAINING: qco live-e3w in the baroclinic term (η/H-order; step-4 integrator work) | — |
| 5 | Split-explicit **centred** (`ln_bt_fw=F`), `nn_bt_flt=1`, auto substeps cmax=0.8 | **DONE (codex r9 P2 fixed, r10 CLEAN)** — `barotropic_time_filter="nemo_boxcar_centred"` (ts_wgt CASE(1) boxcar width n centred at t+Δt, ~1.5n substeps, tail-sum transport weights, continuity-preserving; weights locked vs F90 transliteration) + `nemo_auto_substeps` (ln_bt_auto formula; DINO auto n_e=25 at 1°); preset KEEPS implicit_cn — the forward-frame centred window is UNSTABLE without the MLF before-state start (720d A/B job 8826132: energy grows from d30, NaN d180; NEMO's own DINO namelist: \"model crashes if ln_bt_fw=T\"). Blocks selectable + F90-locked; the preset flips with step-4 MLF | — |
| 6 | S-EOS with DINO coefficients | **EXACT** — `nemo_seos` (Roquet 2015) with a0/b0/λ1/μ1 defaults | Select in the exact recipe (recipe currently picks `wright`!) |
| 7 | TKE (ref defaults incl. `ln_mxl0`, `nn_mxl=3`, pdl(Ri), lc, etau=1/htau=1) + EVD (evdm=1, 100 m²/s) | EXACT blocks exist (P1/P2 campaign) — parameter-by-parameter check vs TKEConfig defaults pending | Diff every TKE field against namelist_ref; wire `vmix_scheme="tke"` in the exact recipe (DINO default currently kpp for stability — the known SW-corner TKE instability must be re-examined under MLF) |
| 8 | Iso-neutral laplacian + **MSC**, slpmax=0.01, aht=Ud·Δ/2 | **DONE (codex r6 CLEAN)** — `lateral_tracer_mixing="isoneutral"`: Redi-only (kappa_GM=0, EIV separate), kappa=K_h_base·cosφ row-scaled (`kappa_redi_lat_scaling`), neutral slopes, S_max=0.01, K_h=0, `implicit_K33=True` = MSC (vertical diagonal backward-Euler). Slope-cap SHIPPED (codex r7 fixed + r8 CLEAN): `slope_limit="nemo_cap"` — slope capped at ±S_max, unit tapers, all triad consumers; preset selects it (A/B vs taper = job 8820394). REMAINING deviation: ldfslp mixed-layer linear slope ramp toward the surface | — |
| 9 | ahm=Uv·Δ/2 iso-level laplacian, NO floor/boost | Form EXACT (0.5·U_M·Δ), but DINO config adds `A_h_floor=1000`, `A_h_eq_boost=3` stabilizers | Exact recipe must run floor/boost OFF (or document as stability deviation if it blows up) |
| 10 | EIV **OFF** | DINO config defaults `use_gm_redi=True` (Visbeck/Treguier) | OFF in the exact recipe |
| 11 | 2-band qsr | EXACT (Paulson–Simpson 2-band) | Coefficient check vs NEMO's `ln_qsr_2bd` constants |
| 12 | `ln_non_lin` drag (Cd0=1e-3, ke0=2.5e-3) | **EXACT** — P4 `bottom_drag_scheme="nemo_quadratic"` (#738), wired into DINOConfig | Select in exact recipe |
| 13 | Real salt flux `sfx = srp·(SSS−S*)`, emp=0; `taum ×1.3` westerly boost (TKE input) | Formula check pending on the apply path (virtual-salt vs real-salt channel; boost presence) | Verify/add both |
| 14 | Grid/bathy/IC/calendar (Mercator 1°, tanh-z, bowl+Drake sill, initcase 4, 360-day) | **EXACT** (Lévy stretching, `_exp_bathy`+`_gauss_ring`, case-4 IC, 30-day months in forcing) | — |
| 15 | Free slip, free surface z\* (key_qco) | free-slip ✓; z\* ✓ | — |

Recipe surface: `ocean/recipes.py` already registers `nemo_dino_v1` (currently:
tvd + adcroft + implicit_cn + wright + euler — several selections differ from
the oracle) plus `mitgcm_v1`, `oceananigans_v1`, `veros_faithful_v1` and a
`fidelity/veros_configs/dino.py` — the Level-2 surface exists; Level-2 work =
point those recipes at the DINO setup and run the comparison battery.

### Implementation order (Level 1)
1. `dino_r1_exact_v1` recipe skeleton: flip everything already EXACT
   (S-EOS, nemo_quadratic drag, TKE+EVD, EIV off, floors/boosts off,
   slpmax, backgrounds 1.2e-4/1.2e-5) — cheap, immediate fidelity gain.
2. Seasonal forcing path (#2) + salt-flux/taum checks (#13).
3. NEMO FCT2/2 (#3), sco-jacobian check (#4), centred split-explicit (#5),
   MSC (#8) — each codex-reviewed.
4. **MLF integrator** (#1) — the big one, last because everything else is
   testable under rk3/euler first and MLF touches the whole step.
5. Exact-match harness runs (protocol above) after each stage; final 4-y
   comparison vs the NEMO reference.

## Verification protocol (Level 1)

1. Rebuild/refresh NEMO reference as needed (container job
   `_run_dino_nemo.sbatch`); reference diagnostics: ACC transport, MOC,
   basin-mean T/S drift profiles, MLD (monthly), KE.
2. legoESM DINO NEMO-recipe run, same 4-y window, same diagnostics.
3. Exactness ladder: (a) static fields (grid depths, bathymetry, masks,
   IC, forcing snapshots) bit/1e-12-compared; (b) single-step tendency
   comparison per term where extractable; (c) trajectory metrics with
   documented divergence-growth expectations (chaotic separation bounds via
   twin-perturbation run).
4. Truth tiers (conservation/equivariance) still outrank oracle matching
   (`ocean/fidelity/precedence.py`).
