# Moist true-LES on the pseudo-spectral incompressible core (BOMEX → DYCOMS-II → RICO)

Goal: run the GCSS warm boundary-layer cloud cases as **true LES** (the
non-dissipative pseudo-spectral incompressible core `spectral_les_plane.py`,
which *sustains* turbulence — the compressible CRM relaminarises), coupled to
**swappable** microphysics (Morrison/M2005 default). NOT the compressible CRM.

User decisions (2026-06-11):
- True LES (spectral core), not CRM, for the shallow-cloud cases.
- BOMEX first, validate, codex-review, THEN replicate to DYCOMS-II RF01 + RICO.
- Microphysics must stay **swappable** (legoESM philosophy) — couple via the
  generic scheme dispatch, never hardwire Morrison.
- Reuse what already exists (SAM/legoESM), don't re-derive.
- Run `/codex:adversarial-review` systematically after each increment.
- Every LES run outputs x-y field PNGs + profile PNGs.

## What is REUSED (no re-derivation)
- gSAM case decks `CASES/{BOMEX,DYCOMS_RF01,RICO}` (snd/lsf/sfc) + the readers
  `sam_case_forcing.read_sam_{snd,lsf,sfc}` (already parse these).
- The SAM-faithful M2005 microphysics (`physics/microphysics/morrison.py`) +
  the **swappable dispatch** `microphysics/integration._get_microphysics_fn(cfg)`
  → `(scheme_name, micro_fn, scheme_cfg)`; `micro_fn(T, q_v, hydrometeors,
  p_full, p_half, rho, dz, dt, cfg)` is the generic per-scheme tendency call →
  Morrison/Kessler/Thompson/P3/SDM/ML all swap by `MicrophysicsConfig.scheme`.
  Tracer-slot layout from `_PLANE_MIN_TRACER_SLOTS` (0=q_v,1=q_c,2=q_r,3=q_i,
  6=N_c,7=N_r,8=N_i).
- Thermo: `legoesm.thermo` saturation; `legoesm.constants` (no literals).
- Transport/SGS: the core's existing `scalar_rhs` (advection + K_h=ν_t/Pr +
  surface flux BC) — call it per scalar.
- Diagnostics/plots: `scripts/run/les_record.py` + `scripts/plot/plot_les_diagnostics.py`
  (x-y cross-sections + profile evolution + 3D box), extended with q_c/LWP.

## What is BUILT in the spectral core (the moist framework)
1. **Anelastic reference state** on the grid: `rho_ref(z)`, `p_ref(z)`,
   `exner_ref(z)`, hydrostatic from `p_sfc` + a base θ(z). (Boussinesq momentum
   kept; thermodynamics use the reference profiles.)
2. **Prognostic scalars** added to `SpectralLESState`: θ_l (use θ slot as θ_l) +
   a `tracers` array (q_v, q_c, q_r, N_c, N_r; slot layout = micro dispatch's).
   Each advected + SGS-diffused by `scalar_rhs` inside `rhs()`; the shared SSP-RK
   integrator's pytree axpy advances them with the same scheme. Dry path stays
   byte-unchanged (tracers=None ⇒ skipped).
3. **Moist buoyancy**: θ_v = θ_l·(1 + 0.61 q_v − q_c − q_r) (after condensation);
   `buoyancy_w` uses θ_v anomaly instead of θ anomaly.
4. **Microphysics coupling** (applied OUTSIDE `step()`, like CRM radiation/sfc —
   host-side, swappable, no AD through it): each step reshape (ny,nx,nz)→(ncol,nz),
   call `micro_fn`, apply tendencies to θ_l (latent heating via exner) + tracers,
   incl. sedimentation (the scheme's fall-speed flux).
5. **Moist surface fluxes**: prescribed sensible (θ) + latent (q_v) into the
   lowest level (BOMEX/RICO/DYCOMS `SFC_FLX_FXD`).
6. **Large-scale forcing**: subsidence (−w_ls ∂φ/∂z) + tls (θ) + qls (q_v),
   from the lsf deck, on the spectral z-grid. DYCOMS adds the Stevens (2005)
   simple-LW radiation parameterization.

## Drivers
`scripts/run/run_{bomex,dycoms,rico}_les.py` — IC from snd, forcing from lsf,
fixed sfc fluxes; record frames; plot x-y + profiles each run.

## Validation targets (literature)
- BOMEX (Siebesma et al. 2003, JAS 60): cloud cover ~10–15%, cloud-base mass
  flux ~0.02–0.04 m/s, LWP ~5–10 g/m², well-mixed subcloud to ~500 m, trade
  inversion ~1500 m, negligible precip.
- DYCOMS-II RF01 (Stevens et al. 2005, MWR 133): LWP ~50–80 g/m², near-100%
  cloud cover, zi ~840 m, entrainment rate ~0.4 cm/s, well-mixed θ_l/q_t.
- RICO (van Zanten et al. 2011, JAMES 3): cloud cover ~10–20%, surface precip
  ~0.3 mm/day, deeper cumulus to ~2–3 km, LWP ~?.

## Status / increments (codex-review each)
- [ ] 1. Reference state + moist config/state scaffolding (dry path unchanged).
- [ ] 2. Multi-scalar transport + moist θ_v buoyancy.
- [ ] 3. Swappable microphysics coupling + sedimentation.
- [ ] 4. Moist surface fluxes + LSF (subsidence/tls/qls) on the spectral grid.
- [ ] 5. BOMEX driver + x-y/profile PNGs; validate vs Siebesma'03.
- [ ] 6. Replicate DYCOMS-II RF01 (+ Stevens LW) + RICO (precip).

## Cross-scheme audit + SGS buoyancy (2026-07-16)
Re-ran all 3 cases × {static-Smag, static-Vreman, LASD-dynamic} × {kessler,
sundqvist, seifert_beheng, morrison, thompson, p3, sdm, fast_sbm} (54 combos):
**zero** stability / dispatch bugs — all finite, water non-negative, schemes
diverge correctly (kessler==morrison below the autoconversion threshold is
expected, not a dispatch bug).

Two corrections made:
1. **`sundqvist` rejected in the LES microphysics adapter** (`make_les_microphysics_fn`).
   It is a LARGE-SCALE GCM diagnostic condensation scheme (RH>RH_crit partial
   cloud fraction for coarse grid boxes); on the resolved LES grid it condenses
   in every cell above RH_crit → spurious ~100 % cloud cover + inflated LWP
   (measured cc ~0.9–1.0, LWP 20–43 vs 0.03–0.24 / a few g/m² for the
   resolved-cloud schemes). Use kessler/seifert_beheng/morrison/thompson/p3/
   sdm/fast_sbm.
2. **Lilly (1962) SGS stratification suppression** (`sgs_buoyancy` flag,
   default OFF ⇒ neutral byte-identical; ON in the cloudy drivers). Multiplies
   the strain-only ν_t by `√(max(0, 1 − Ri/Pr_t))`, `Ri = N²/|S|²` from the
   resolved θ_v gradient — the SGS previously had NO stratification dependence
   (documented follow-up) and over-mixed the stable inversion. Reuses the factor
   already in the single-column `turbulence/smagorinsky.py` (extracted to shared
   `_shared.lilly_buoyancy_factor`). Codex-reviewed (0 HIGH; 1 MED = build the
   Ri denominator `2 S_ij S_ij` from the strain components, not `Smag**2`, so the
   adjoint is self-contained AD-safe at zero strain). 6 unit tests + 104-case LES
   regression green; column faithful-pin bit-identical.

### DYCOMS low-LWP: the SETTLED cloud is robustly thin in every regime
DYCOMS-II RF01 SETTLED (2–6 h quasi-steady) cloud sits ~5–10× below the 50–80
g/m² benchmark (cc ~0.2–0.4 vs ~1.0) while **z_i ≈ 820 m, mixed-layer θ and q_t
are correct** and the cloud is present (not zero). The saturated IC (LWP ~62)
COLLAPSES to a thin quasi-steady in EVERY controlled config tried — this is the
well-known LES difficulty of RF01 (cf. the Stevens 2005 intercomparison spread),
NOT a single bug. Settled 2nd-half-mean LWP measured here:

| config (64², morrison) | settled LWP | cc | note |
|---|---|---|---|
| f32 van_leer nz192, sgs_buoyancy OFF | 4–8 | 0.2–0.3 | baseline, keeps decaying 8→4 over 3→6 h |
| f32 van_leer nz192, sgs_buoyancy **ON** | **7–10** | 0.33–0.40 | **best durable; +83 % vs OFF at 6 h** |
| f32 van_leer nz128 / 256 | ~7 | ~0.28 | resolution 11.7→5.9 m: no material change |
| kessler / thompson nz128 | 6.7 / 6.7 | 0.25–0.29 | scheme: no material change |
| f64 weno5 + θ-hyperdiff 2e4 nz120 | ~4–6 (2.8 h) | 0.2–0.3 | collapses too; clip_q explodes (see below) |

**Corrected from a 33-day-old note** that claimed `f64 + weno5 + θ-hyperdiff →
LWP ~48`: that does NOT reproduce as a SETTLED value — a controlled current run
collapses to ~4–6 by 2.8 h. The "48" was a TRANSIENT early-time reading (the
cloud passes through ~35 g/m² at 0.3 h on its way down from the LWP-62 IC).
Lesson: quote the 2nd-half time-mean, not the peak (the drivers now print both).

The Lilly `sgs_buoyancy` fix is the **strongest durable lever found**: on the
van_leer core it nearly DOUBLES the settled LWP (nz192 6 h: 7.7 vs 4.2; peak LWP
16.5 vs 8.1, cc 0.67 vs 0.54 at t≈1.25 h) by suppressing cloud-top
over-entrainment — real, correct missing physics. It does NOT fully cure the
collapse; the residual to 50–80 g/m² is the open RF01 hard-case problem
(radiative-turbulent cloud maintenance vs entrainment at LES resolution; protocol
dz = 5 m needs nz ~300, blocked by the projection's nz≲200 compile cliff). On the
weno5 f64 core the SGS effect is swamped by a **clip_q explosion** (~7, vs ~0.5
for van_leer) — weno5 is non-positivity-preserving on the water tracers here (the
CRM guard `weno5→van_leer` from #966 is NOT ported to this moist-LES core; SEPARATE
follow-up). van_leer stays the moist-LES default. BOMEX/RICO (thin transient
cumulus) sit near the low end of their bands, more defensible for those types.

Repro (controlled A/B, change only the flag): `scripts/run/run_dycoms_les.py
--nx 64 --ny 64 --nz 192 --hours 6 --f32 --microphysics morrison
[--sgs-buoyancy | --no-sgs-buoyancy]`. All 3 cloudy drivers now report a trailing
2nd-half time-mean of cc/LWP (GCSS reports time-means; instantaneous values swing
frame-to-frame). Figure: `results/dycoms_lilly_sgs_ab_2026-07-16.png`.
