# NEMO GYRE_BARE ↔ legoESM `build_nemo_gyre_recipe` — Wiring Comparison

Nut-and-bolt comparison of every scheme and parameter, built by tracing both
codebases (NEMO 5.0.2 `cfgs/GYRE_BARE/WORK/*.F90` + resolved namelists; legoESM
`build_nemo_gyre_recipe` + `LatLonCGridOceanModel.step`). Purpose: match NEMO's
GYRE exactly, differentiability being the only intended difference.

Status: ✓ matched · ✗ mismatch · ≈ equivalent-but-different-form.

## 0. Headline (re-audited 2026-07-16, post-fix state)
Every row below is now ✓/≈ except tracer stepping (Euler vs RK3 — retested
inert on the OLD config only). Balance decomposition at yr 5: barotropic mode
**1.03×** NEMO; lego's flow is in **perfect thermal-wind balance with its own
density** (actual/geo = 0.98 vs NEMO's 1.13 ageostrophic WBC enhancement) — the
momentum side is exonerated. The ONE remaining gap: winter erosion of the
PERMANENT thermocline doming (lego winter retention ~0.1 vs NEMO ~0.4) caps
lego's geostrophic shear at 0.81× → actual baroclinic flow 0.71×. Connects to
the known residual: deep TKE in near-neutral winter columns still 5–50×
NEMO's rn_emin floor even after the alpha_tke fix.

## 1. Grid & coordinate
| Item | NEMO | legoESM | |
|---|---|---|---|
| Horizontal | beta-plane C-grid, dx=dy=106 km, un-rotated | same | ✓ |
| f0 / beta | 3.775e-5 / 2.002e-11 (φ0=15°) | same | ✓ |
| Interior cells | 32×22 (30×20 wet) | same | ✓ |
| Wet levels / thicknesses | 30, MI96 ladder e3t 10→301 m | same dz_ref | ✓ |
| Vertical coordinate | pure z, full step, `key_linssh` (FIXED thicknesses) | z-star with `linear_free_surface=True` (frozen J, no σ-redistribution, top-cell flux) | ✓ (fixed 128feaffb) |
| Bathymetry | flat, H=4300.71 m, no partial cells | same | ✓ |
| Initial T(z)/S(z) | analytic tanh (usrdef_istate) | identical formula | ✓ |

## 2. Time stepping
| Item | NEMO | legoESM | |
|---|---|---|---|
| Momentum RK3 | Wicker-Skamarock (stage dt/3, dt/2, dt; RHS asym: LDF stages 1&3, ZDF stage 3) | `momentum_time_integrator="rk3_ws"` (same stages + skip_lateral_viscosity gating) | ✓ (fixed e2b4b03a0) |
| **Tracer stepping** | **RK3** (FCT stage 3, centred stages 1-2) | **forward Euler** | ✗ |
| dt | 14400 s | same | ✓ |
| Barotropic substeps | 50 (auto, Δt=288 s) | 50 | ✓ |
| Barotropic filter | nn_bt_flt=3 Demange (temporal, rn_bt_alpha=0.07, no spatial diffusion) | `nemo_ab3am4` (AB3+AM4 α=0.07, final-value output, alpha=0, cross-window `bt_hist`) | ✓ (fixed 483b6477a + b56590a1a) |

## 3. Momentum
| Item | NEMO | legoESM | |
|---|---|---|---|
| Advection form | vector-invariant | same | ✓ |
| Vorticity | ENE Sadourny, f+ζ combined, vertex-f (ff_f) | `vorticity_scheme="ene_total"` (combined vertex-f) | ✓ (fixed c189fde38) |
| KE gradient | C2 mean-of-squares (nn_dynkeg=0) | c2 (same formula) | ✓ |
| Vertical mom. advection | 2nd-order centred (dynzad) | upwind_perturbation | ≈ (retested 2026-07-16 on the fixed config: centred changes baroclinic u' 0.71→0.72 — INERT; flow is in perfect thermal-wind balance either way) |
| Lateral viscosity | Laplacian div-rot, A_m=1e5 | vector_laplacian, A_h=1e5 (bit-exact) | ✓ |
| Vertical viscosity | implicit, TKE avm | same | ✓ |
| Bottom drag | non-linear implicit, Cd0=1e-3, ke0=2.5e-3 | nemo_quadratic (same) | ✓ |
| PGF | z-coord (ln_hpg_zco), e3w-weighted integral | nemo_trapezoid quadrature; ladder certified bit-exact vs `rhd_stg` (5.3e-23 full-field); the old "3.4% deficit" was a stage-3 time-level comparison artifact (see plan §C) | ✅ |
| Lateral BC | free-slip (rn_shlat=0) | free_slip | ✓ |

## 4. Tracer
| Item | NEMO | legoESM | |
|---|---|---|---|
| Advection | FCT2 (2-substep upstream low-order under RK3) | fct2 (Euler) | ≈ |
| Iso-neutral diffusion | Laplacian, A_ht=1000, rn_slpmax=0.01 | kappa_Redi=1000, S_max=0.01 | ✓ (fixed bc2f5eb1d) |
| GM eddy | OFF (ln_ldfeiv=F) | kappa_GM=0 | ✓ |
| Vertical diffusion | implicit, TKE avt | same | ✓ |
| Convection (EVD) | avt=avm=100 where min(rn2,rn2b)≤-1e-12 on rn2=ADIABATIC N² | enhanced_diffusion K=nu=100, hard threshold, n2_mode="adiabatic" | ✓ (fixed ccbdf2020 — the in-situ trigger fired on spurious compressibility negatives in STABLE spring columns, blocking the thermocline rebuild; min(rn2,rn2b) 2-level + -1e-12 threshold remain minor deltas) |

## 5. Vertical mixing (TKE)
| Item | NEMO | legoESM | |
|---|---|---|---|
| Scheme | zdftke prognostic | tke prognostic | ✓ |
| **TKE vertical self-diffusion** | **avm × 1** (zdftke:130, zfact1=-0.5·rn_Dt) | was alpha_tke=30 (Veros/CATKE) — 30× too fast, THE thermocline self-lock driver; now alpha_tke=1 | ✓ (fixed 6f3e4f5d6) |
| c_k / c_eps | 0.10 / 0.70 | same | ✓ |
| Background avm/avt | 1.2e-4 / 1.2e-5 (nn_avb=0 const) | kappaM_min/kappaH_min same, no BL profile | ✓ |
| Prandtl (nn_pdl=1) | Ri-dependent, 1/ri_cri=4.5 | prandtl_ri_coeff=4.5 | ✓ |
| Mixing length | nn_mxl=3 + ln_mxl0 anchor | tke_mxl_choice=3 + anchor | ✓ (fixed aeb9b7df4 — choice 2 was 7x high at 10 m in the near-neutral ML; certified 3-4 sig figs vs avt_k) |
| **Surface BC (wind-driven)** | **Dirichlet en(1)=max(rn_emin0, rn_ebb·\|τ\|/ρ0), rn_ebb=67.83** | same formula in tke.py, ACTIVE once the step-level wind forcing supplies τ (audit 2026-07-16: was inert — the body-force wind bypassed TKE → surface TKE sat at the 1e-4 floor, ~50× low) | ✓ (rewired) |
| Langmuir (ln_lc=T, rn_lc=0.15) | active with wind (W_lc from taum) | lc=True on the post-mixing path | ✓ (fixed faa4cb20a) |
| N² for TKE | rn2/bn2 (alpha·dT/dz−beta·dS/dz, locally-referenced = adiabatic-equivalent) | adiabatic | ✓ (equivalent forms) |

## 6. EOS
| Item | NEMO | legoESM | |
|---|---|---|---|
| Scheme | EOS-80 (Roquet-55 poly) | nemo_eos80 (certified 4.5e-13) | ✓ |
| rho_0 | 1026 | constants.rho_ocean_nemo=1026 | ✓ (fixed c874633b6) |
| g | 9.80665 | same | ✓ |

## 7. Surface forcing
| Item | NEMO | legoESM | |
|---|---|---|---|
| **WIND STRESS** | **double-gyre, ~0.074 Pa, seasonal** (usrdef_sbc:167) | `nemo_gyre_wind_forcing` via the step-level `surface_forcing=` (canonical route: stage-10b' momentum + TKE Dirichlet BC; sign converted atm↔ocean; seasonal) | ✓ (rewired 2026-07-16) |
| Haney heat restoring | −40 W/m²/K | 40, via tau conversion | ✓ |
| Solar qsr | 230·cos, 2-band Jerlov type-I | same | ✓ |
| E-P freshwater | emp sin-split, seasonal, net-zero | same | ✓ |
| Calendar | 360-day (nn_leapy=30) | same | ✓ |

## 8. Everything else
| Item | NEMO | legoESM | |
|---|---|---|---|
| GM / MLE / geothermal / tidal / sponge / BBL | all OFF | off | ✓ |

## Remaining-gap summary (re-audited 2026-07-16; the historical ranked list is superseded)
1. **Winter erosion of the permanent thermocline** (retention 0.1 vs 0.4) — the
   single driver of the baroclinic 0.71×; suspects: residual deep TKE in
   near-neutral winter columns (5–50× floor), EVD min(rn2,rn2b) 2-level +
   −1e-12 threshold minor deltas.
2. NEMO's ageostrophic WBC enhancement (actual/geo 1.13 vs lego 0.98) — likely
   follows #1 (inertial recirculation strengthens with the gyre).
3. Tracer stepping Euler vs NEMO RK3 (only row not retested post-fix).
4. SST max −0.8 °C (follows #1: winter convection slightly deeper).

## Verified progress (yr-5 metrics vs NEMO)
| config | surf rms(u) | baroclinic u′ | surf/deep | summer thermocline |
|---|---|---|---|---|
| pre-campaign | 0.58× | — | 0.19–0.38 | eroded (0) |
| all fixes 2026-07-16 (bothfix card) | 0.72× | 0.71× (balance 0.98) | 6.2 (full-metric) | **3.46 vs NEMO 3.17 ✓** |
| NEMO target | 1.00 | 1.00 | 8.2 | 3.17 |

Fix ledger (this campaign): Coriolis-RK3 (`70ab903c9`), namelist knobs
(`ffc98144d`), A_v/K_v (`bfc3126a4`), EVD operator (`456922278`), wind rewire
(`673d56ee1`), Dirichlet TKE (`76b239532`), linssh (`128feaffb`), ab3am4
(`483b6477a`), rho_0 (`c874633b6`), WS-RK3 (`e2b4b03a0`), pgf trapezoid, mxl3
(`aeb9b7df4`), l_eps, bt_hist (`b56590a1a`), S_max (`bc2f5eb1d`),
**alpha_tke 30→1 (`6f3e4f5d6`)**, **EVD adiabatic N² (`ccbdf2020`)**.
