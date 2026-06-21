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
