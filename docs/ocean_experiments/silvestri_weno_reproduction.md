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

### §4 effective-resolution convergence — REPRODUCED ✅ (resolution sweep N=64/128/256)
KE retained at t=6 vs resolution (the paper's "effective resolution" Fig-4 claim):

| Scheme | N=64 | N=128 | N=256 |
|---|---|---|---|
| DNS (ref) | 0.898 | 0.963 | 0.957 |
| **W9V** | **0.975** | 0.963 | 0.956 |
| W9D | 0.867 | 0.950 | 0.954 |
| W5V | 0.790 | 0.908 | 0.941 |
| W5D | 0.544 | 0.775 | 0.879 |
| Leith2 (C=2) | 0.054 | 0.282 | 0.670 |

**W9V is converged (≈DNS, KE≈0.96) already at N=64**, while W5V needs N=256 and W5D/Leith2 still
lag — a ~4× effective-resolution advantage for W9V, and the convergence ordering W9V > W9D >
W5V > W5D > Leith is exactly the paper's §4 result ("the W9V approach comes closer to achieving
the LES goal of resolution independence"). The absolute KE also converges with N (DNS 1.28→1.96,
W9V 1.39→1.96 — W9V tracks DNS at every grid).

Figures: `fig4_turb2d_timeseries.png` (KE/enstrophy(t)), `fig4b_turb2d_convergence.png`
(effective-resolution convergence), `fig5_turb2d_spectra.png` (isotropic spectra @ t=3.6),
`fig3_turb2d_vorticity.png` (vorticity fields). (Runtime artifacts, gitignored.)

## §5 — baroclinic jet — PIPELINE BUILT; production runs UNSTABLE (open problem)
The §5 recipe (front Eqs 52-53 + thermal wind + τ=50d zonal-mean restoring, uniform 20m/50lev,
L_d≈5.7km), the per-scheme driver (1000-day scan + Fig-7/8/9/10 metrics), and the plotter are
built. QG2 uses the FAITHFUL full QG-Leith (B5b stretching), no longer the barotropic approximation.

**CPU sanity validation (48×32, 40 days, W9V vs faithful QG2):**
- The pipeline integrates stably over a long run; the restoring holds the mean jet; L_d≈5.4–5.5 km.
- Baroclinic instability DEVELOPS (EKE grows from the white-noise seed in both schemes).
- The schemes are qualitatively distinguishable as the paper predicts: **W9V** (no explicit closure)
  is far more energetic (EKE→1.1e14, max|u|→3.9) and grid-scale-noisier (gridscale_frac 0.083);
  **QG2** is more damped (EKE→1.8e11, max|u|→0.46, gridscale_frac 0.026) — the explicit QG-Leith
  viscosity controlling the grid scale.
- **The faithful-QG2 B5b stretching path is stable over 40 days** (not just the 1-step smoke test).
- Caveat: at 48×32 the deformation radius (~5.4 km) is under-resolved ~7×, so WENO accumulates
  grid-scale energy (expected — the paper's effective-resolution point); a clean eddy field needs
  the paper's 7-km grid.

**Production-resolution status (1/8° = 160×128×50) — UNSTABLE, OPEN PROBLEM.** On GPU (V100S, ~8.5
s/day) all 5 schemes were run to 1000 days but **blow up early during instability onset**: W9V day
11, QG2 day 15, SM2 day 24 (max|u|→nan). Halving dt to 450 s did NOT help (W9V still blows at day
11), and even the explicit-closure schemes (SM2/QG2) blow up — so it is not a simple CFL fix nor a
pure lack-of-dissipation issue. (An earlier 3-day probe wrongly read as "viable" — it never reached
the day-11 failure; corrected here.)
- **1/16° (320×256×50) blows up EVEN FASTER — day 4 vs 1/8° day 11.** Finer-grid-blows-faster is
  the signature of a GRID-SCALE NUMERICAL INSTABILITY, **not** under-resolution (under-resolution
  would improve with finer grids). The WENO schemes have NO explicit dissipation (paper-faithful),
  and legoESM's WENO vector-invariant evidently does not supply enough IMPLICIT dissipation to
  control the C-grid grid-scale mode at eddy-resolving resolution — the SAME conclusion the Eady
  rebuild reached (it needed a tuned A_h≈1000 + C_smag≈0.1 backstop). (Also: fp64 320×256×50 ≈ 24 GB
  → near the V100S 32 GB limit; 1/16°/1/32° likely need float32 or memory-managed blocks.)
- **Open work to get §5 producing results** (the real reproduction blocker — a numerics problem,
  not the pipeline): the WENO vector-invariant momentum needs enough implicit dissipation to control
  the C-grid grid-scale mode without an explicit closure (as Oceananigans' WENOVectorInvariant does).
  Options: (a) confirm-the-diagnosis — add the Eady min-dissipation backstop (A_h≈1000+C_smag≈0.1) to
  the WENO §5 schemes; if it stabilizes, it pins the cause to insufficient implicit dissipation (but
  it's then NOT a faithful no-closure W9V). (b) Investigate legoESM's weno9 implicit-dissipation gap
  vs Oceananigans (the D-term divergence damping, the KE-gradient form, the barotropic-baroclinic
  coupling, the smc03 PGF on the tilted front). (c) The dispersive schemes (SM2/QG2) blow too despite
  their closure → also check the setup (thermal-wind IC projection onto a grid mode? the AB2 outer +
  implicit-CN barotropic at fine res?). This is the same hard problem the Eady eddy-resolving work
  worked through; see `docs/planning/eady_eddy_resolving_ralph.md`.

Launch once stabilized: `run_silvestri_baroclinic_jet.py --scheme S --resolution RxC --days 1000`,
one job per pinned GPU; then `plot_silvestri_comparison.py --case jet` + fill the §5 scoreboard.

## Caveats / remaining work
- **B5b DONE:** QG2 now uses the full QG-Leith with the baroclinic stretching ∂_z(f/N²∇b) + the
  Bachman Bu/Ro bound — faithful. (Caveats: L_d is a fixed config constant 6.75 km, velocity_scale=1
  for Ro — fine for the near-uniform-stratification Silvestri channel; the stretching is zeroed in
  statically-unstable columns.)
- **w′b′ cospectrum (Fig 9 right):** deferred — needs the diagnosed vertical velocity (not in
  the prognostic state).
- **§4 DNS reference + resolution sweep (64→1024, 4096 DNS):** run on GPU for the full
  convergence figure; the N=96 single-resolution run already establishes the scheme ranking.
- **§5 1000-day matrix:** GPU. Pin one GPU per (scheme,res) job; re-launch on kill.
