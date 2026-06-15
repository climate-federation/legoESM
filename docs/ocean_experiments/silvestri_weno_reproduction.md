# Silvestri et al. 2024 (WENO vector-invariant momentum advection) — legoESM reproduction

Reproduction of **Silvestri, Wagner, Campin, Constantinou, Hill, Souza, Ferrari (2024)**,
*"A New WENO-Based Momentum Advection Scheme for Simulations of Ocean Mesoscale Turbulence,"*
JAMES 16(7), e2023MS004130. Build plan + iteration log:
`docs/planning/silvestri_weno_reproduction_ralph.md`.

## What was built (Phases 1-3)
All five paper schemes are selectable legoESM config (each adversarially reviewed):
| Paper | legoESM | block |
|---|---|---|
| **W9V** | `momentum_advection="weno9"` + `weno_smoothness="split"` | B1+B2 |
| **W9D** | `weno9` + `weno_smoothness="standard"` | B2 |
| **UP3** | `flux_form` + `momentum_flux_scheme="upwind3"` | B3 |
| **SM2** | `lateral_friction_scheme="om4p25"` | B4 |
| **QG2** | `lateral_friction_scheme="qg_leith"` (BAROTROPIC approx — stretching omitted, see B5b) | B5 |

Diagnostics (`ocean/diagnostics.py`): relative vorticity, KE/enstrophy integrals, eddy
decomposition + EKE/TKE/eddy-APE, w′b′ flux, Parseval-correct zonal/isotropic spectra,
zonal-mean, deformation radius. Experiments + drivers: `silvestri_turbulence_2d.py` (§4
harness) + `run_silvestri_turbulence_2d.py`; `silvestri_baroclinic_jet.py` (§5 recipe) +
`run_silvestri_baroclinic_jet.py`; `plot_silvestri_comparison.py`.

## §4 — 2D decaying turbulence — REPRODUCED ✅
A standalone Cartesian doubly-periodic 2D NS vorticity solver driving the canonical WENO
vorticity-flux kernels (§4 is a Cartesian box, not the lat-lon channel). Ishiko IC,
Re=3.3e4, t=6 (~18 eddy turnovers, T_e≈0.33). Run at N=96 (CPU; the paper's 4096² DNS +
1024² coarse sweep need GPU — pending):

| Scheme | KE retained (t=6) | Enstrophy retained |
|---|---|---|
| **W9V** | **0.967** | 0.424 |
| W9D | 0.933 | 0.381 |
| DNS (2nd-order + 1/Re) | 0.943 | 0.789 |
| W5V | 0.869 | 0.322 |
| W5D | 0.700 | 0.237 |
| Leith1 (C=1) | 0.574 | 0.246 |
| Leith2 (C=2) | 0.144 | 0.048 |

**All three of the paper's §4 findings reproduced:**
1. **WENO conserves energy; Leith over-damps it.** W9V/W9D retain 93-97% of KE; Leith2 (C=2)
   collapses to 14% — the paper's "too low kinetic energy" result for the explicit closure.
2. **The headline `{ζ;u}` (W*V) > `{ζ;ζ}` (W*D) smoothness distinction.** W9V (0.967) > W9D
   (0.933) AND W5V (0.869) > W5D (0.700): the velocity-based smoothness measure retains more
   energy at fixed order — the paper's central novelty. (Confirms the point→cell-average
   conversion is active; without it W9V would degenerate to W5V.)
3. **Higher WENO order keeps more energy.** W9V > W5V, W9D > W5D.
   WENO selectively dissipates ENSTROPHY (ens 0.32-0.42 vs DNS 0.79) while preserving energy —
   "dissipates enstrophy, but not energy, at small scales" (paper §4).

Figures: `fig4_turb2d_timeseries.png` (KE/enstrophy(t)), `fig5_turb2d_spectra.png` (isotropic
spectra @ t=3.6), `fig3_turb2d_vorticity.png` (vorticity fields). (Runtime artifacts, gitignored.)

## §5 — baroclinic jet — PIPELINE READY, matrix PENDING (GPU)
The §5 recipe (front Eqs 52-53 + thermal wind + τ=50d zonal-mean restoring, uniform 20m/50lev,
L_d≈5.7km), the per-scheme driver (1000-day scan + Fig-7/8/9/10 metrics), and the plotter are
built + smoke-tested (16×16/2d stable, L_d correct). The full matrix — {1/8°,1/16°,1/32°} ×
{UP3,W9V,W9D,SM2,QG2} × 1000 days (15 runs) — needs GPU (no CUDA in the build session); CPU is
impractical at 1000 days × production resolution.

## Caveats / remaining work
- **B5b (before the QG2 §5 matrix case):** the QG-Leith is the BAROTROPIC approximation — the
  baroclinic stretching term ∂_z(f/N²∇b) is omitted (needs buoyancy threaded into the viscosity
  stage). Until then the §5 QG2 case must be labeled "QG-Leith (barotropic)", not "QG2".
- **w′b′ cospectrum (Fig 9 right):** deferred — needs the diagnosed vertical velocity (not in
  the prognostic state).
- **§4 DNS reference + resolution sweep (64→1024, 4096 DNS):** run on GPU for the full
  convergence figure; the N=96 single-resolution run already establishes the scheme ranking.
- **§5 1000-day matrix:** GPU. Pin one GPU per (scheme,res) job; re-launch on kill.
