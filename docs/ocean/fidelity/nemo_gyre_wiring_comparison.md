# NEMO GYRE_BARE ↔ legoESM `build_nemo_gyre_recipe` — Wiring Comparison

Nut-and-bolt comparison of every scheme and parameter, built by tracing both
codebases (NEMO 5.0.2 `cfgs/GYRE_BARE/WORK/*.F90` + resolved namelists; legoESM
`build_nemo_gyre_recipe` + `LatLonCGridOceanModel.step`). Purpose: match NEMO's
GYRE exactly, differentiability being the only intended difference.

Status: ✓ matched · ✗ mismatch · ≈ equivalent-but-different-form.

## 0. Headline
The dominant mismatch is a **missing primary forcing**: NEMO GYRE_BARE is a
**wind-driven** gyre (`usrdef_sbc.F90:167-176`, double-gyre stress ~0.074 Pa);
legoESM's recipe is thermal-only. Without wind legoESM has only a weak thermal
circulation, which — combined with excessive downward momentum spreading and an
abyssal density drift — produces a bottom-intensified, weak-surface flow (surf/deep
rms(u) 0.19–0.38 vs NEMO 8.2) at ~0.58–0.66× NEMO's RMS.

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

## Mismatch summary (ranked by likely impact on the strength gap)
1. **Missing wind stress** — the primary gyre driver. legoESM thermal-only. (biggest)
2. **Momentum spreads down** — coupled with the deep-convection momentum-EVD + the abyssal density drift; the injected wind doesn't form a surface Ekman layer (surf/deep 0.19–0.38 vs 8.2).
3. **Abyssal density drift** — deep horizontal density gradient (uniform-with-depth) that NEMO lacks → deep geostrophic flow + deep convection. Deeper root; candidates below.
4. Vertical momentum advection: upwind_perturbation vs 2nd-order centred (dissipative).
5. Redi slope clip: S_max=0.005 vs rn_slpmax=0.01.
6. Barotropic: 120 substeps/cosine+spatial-diffusion vs 50/Demange nn_bt_flt=3 (no spatial diffusion).
7. RK3 variant (Shu-Osher vs Wicker-Skamarock) + tracer stepping (Euler vs RK3).
8. z-star vs key_linssh; rho_0 1025 vs 1026; TKE N² adiabatic vs in-situ; nn_mxl 2 vs 3 (deconfounded).

## Verified progress (5-yr wet-masked RMS ratio vs NEMO)
| config | ratio | surf/deep rms(u) |
|---|---|---|
| matched knobs, no convection | 0.61 | — |
| + faithful EVD (well-mixed ML) | 0.58 | 0.31 |
| + wind | 0.58 | 0.19 (wind mixed down) |
| + wind, no momentum-EVD | **0.66** | **0.38** |
| NEMO target | 1.00 | 8.2 |

Committed fidelity fixes this line of work: Coriolis-RK3 blow-up (`70ab903c9`),
faithful namelist knobs (`ffc98144d`), A_v/K_v double-count (`bfc3126a4`), EVD
operator (`456922278`). Wind + the deep-structure mismatches remain to wire.
