# Ocean faithfulness vs NEMO — all-grid tracker

**Goal:** every legoESM ocean grid (tripole/eORCA, latlon, cubed_sphere, mpas, spectral)
gives a faithful comparison to NEMO ORCA1 (CORE-II NYF). Codex-review every change.
**Shrunk at iter-28** (was 338). Older detail: `OMIP_faithful.md`, git history. Memories:
[[omip-faithful-project]], [[omip-rk3-coldstart-solve]], [[omip-pipeline-coordinate-bugs]],
[[omip-smag-cap-stabilizer]], [[omip-postmerge-packages-layout]].

**Branch:** `omip-faithful-nemo-comparison`. Completion promise DONE only when ALL grids
genuinely match — currently **2 of 5** (tripole, latlon). NOT done.

## Faithful pipeline (built + proven)
- **Runner = `scripts/run/run_omip_core2.py`** = THE faithful path: applies CORE-II OMIP-2
  bulk forcing (`omip2_applicator.apply_omip2_surface_forcing` / `compute_omip2_surface_forcing`)
  inside `model.step`. `scripts/run/run_omip.py` is a DIFFERENT runner using SST/SSS RESTORING
  (NOT faithful — don't use for the NEMO match). `run_omip._create_setup` is the shared
  grid/model builder reused by run_omip_core2 for all grids.
- NEMO ref: ORCA1 5-yr (morays container). Annual `ORCA1_1y_*` AND monthly `ORCA1_1m_*`
  grid_T in `.../EXP00/RUN_REF/`. Forcing `~/.cache/legoesm/ocean_fidelity/forcing/core2_nyf/`.
- **Scorer `scripts/validate/compare_omip_nemo.py`**: cKDTree-IDW regrid (grid-agnostic —
  flattens any source incl. cube (6,n,n)), SST/SSS RMSE/corr + `_band_breakdown`.
  **`--nemo-month M`** = climatological calendar-month scoring (CF-decoded) — CRITICAL: the
  model runs perpetual-NYF from a WOA *annual* IC, so a day-D snapshot ≈ calendar day-D; it
  MUST be scored vs the same-month NEMO climatology, not the annual mean (else a fake
  NH-cold/SH-warm seasonal dipole appears). Day→month (365d): 30→Jan,45→Feb,60/90→Mar.

## SSS RUNOFF RESULT (validated)
latlon+runoff (8418473) day-90 vs NEMO Mar: SST RMSE **1.11** corr 0.995 (runoff barely
touches SST). SSS **improved by runoff**: RMSE 1.89→**1.79**, corr 0.692→**0.730**, bias
−0.06→−0.10 (rivers freshen coasts). Net positive + stable (runoff doesn't destabilise).
SSS still > good-tol (<1.0) → needs coastal-SPREADING + area-CONSERVATION of the runoff
(current k=4 IDW from discharge cells is too concentrated; codex conservation flag) to
improve further. Runoff feature: wired+codex+tested+validated (commits 7ecf0b88,5c1f21bd,b9e44703).

## Per-grid status
| grid | cold-start | vs NEMO (seasonal) |
|---|---|---|
| **tripole/eORCA025 ¼°** | STABLE (corrected-IC + RK3 + stack) | **day-90 SST RMSE 1.15, corr 0.99 — EXCELLENT** |
| **latlon 1°** | STABLE (mask-aware polar filter) | **day-90 SST RMSE 1.12, corr 0.99 — EXCELLENT** (converges 1.23→1.12) |
| cubed_sphere | **PIVOT to C-D grid** (atmosphere-matching FV3; FC A-grid DEPRECATED). Partial cells now on BOTH backends (FC + cd-grid). | cd-grid cold-start smoke → smc03 PGF on C-D AL corner gradient → switch default to cd-grid → ~C256 ¼° |
| mpas | untested w/ CORE-II; ico3 ~900 km RESOLUTION-LIMITED | #160 full-PV refactor (HARD, see below) + core2 builder + ¼° |
| spectral | applicator CANNOT force it; T21 ~5.6° RESOLUTION-LIMITED | grid-space forcing path (largest gap) + core2 builder + higher res |

### mpas #160 roadmap (mapped iter — implementation is HARD, deferred)
Three sites handle planetary f (all `mesh.fEdge`/`fVertex`): (1) PV flux `ocean_pe_mpas.py:466`
uses `zero_f` → q=ζ/h (relative only), paired with full u_3d; (2) `_forward_backward_coriolis_mpas_3d`
(`ocean_model_mpas.py:56`, called ~549) applies f to the PERTURBATION u'=u−ū (Matsuno);
(3) `barotropic_mpas.py:273` applies online f·v_t(ū) to the DEPTH-MEAN each substep. `F_slow_u`
EXCLUDES depth-mean f. **#103** (commit 3e12e4ec): full PV froze the depth-mean f across 30
barotropic substeps → near-inertial (τ~1/f) instability → the current split was the fix.
**#160 (full PV q=(f+ζ)/h, drop the Matsuno) is HARD because in the energy-conserving TRiSK
flux the planetary-f and relative-ζ are ENTANGLED** — you cannot cleanly subtract the
depth-mean Coriolis from F_slow to keep the barotropic online-f without double-count or
re-freezing (the MOM6-style slow-forcing refactor). Config gate `full_pv_coriolis` would go in
`mpas_config.py` after `pv_alpha`. CORRECTNESS hinges on a #103 near-inertial GPU regression
test (currently blocked) on a resolution-limited grid → NOT safe to implement blind. Defer
until GPU + the ¼° mpas builder exist.

## DONE grids (1 & 2) — winning configs
**Both** use run_omip_core2 + `--woa-init --partial-cell --pgf-scheme smc03
--adaptive-implicit-vertadv --momentum-rk3 --freeze-floor`.
- **tripole**: `--mesh eORCA025_mesh_mask.nc --dt 75 --balanced-init --min-levels 2
  --C-smag-lap 3.0 --smag-cfl-safety 0.125`. RK3 is the key (solved the corrected-WOA cold-start
  forward-Euler+Matsuno couldn't); implicit_cn barotropic (NOT explicit_substep).
- **latlon**: `--grid latlon_bathy --latlon-res 180x360 --dt 300 --min-levels 2 --C-smag-lap 3.0
  --smag-cfl-safety 0.125 --polar-filter --polar-filter-cutoff-lat 60`.

### Key shipped features (all tested + codex-clean)
- **Seasonally-matched scoring** `--nemo-month` (31b2539f): exposed the "equilibration
  degradation" (day-90 RMSE 2.92) as a metric artifact — properly scored, NHmid −3.68→−0.33,
  SHmid +2.30→−0.07. ALL prior "POOR at equilibration" verdicts were this metric bug.
- **Freeze-floor** `config.freeze_floor` (7a81bc55): surface `jnp.maximum(T, −1.8°C)` sea-ice
  surrogate; closes the Arctic super-cool gap (−4.9→−0.1). `LatLonCGridOceanModel._apply_freeze_floor`,
  applied in step() AFTER the polar filter, gated static bool.
- **Mask-aware Fourier polar filter** `config.use_polar_filter` (acde1c6d):
  `LatLonCGridOceanModel._apply_polar_filter` reuses `grids.polar_filter`; fixes the latlon
  N-pole CFL blowup (~day 0.25). Land filled with per-lat ocean zonal mean before FFT, restored
  after; per-lat wet-cell zonal mean restored (eta volume / zonal-mean u exact; tracer content
  exact for uniform thickness, approx under partial cells). `compute_v_face_coords` promoted
  public. 7 unit tests; codex 2 HIGH fixed → CLEAN. latlon Arctic caveat did NOT manifest.
- Enabling fixes (committed/merged): smag-cfl viscosity ceiling (`smag_cfl_safety`),
  adaptive-implicit vertadv (NEMO ln_zad_Aimp), partial-cell smc03 PGF, RK3; pipeline
  coordinate/unit bug fixes (scorer double-rad2deg, WOA lat/lon mismatch, NEMO mask, _idx_t +3h).

## GRID-3 cubed_sphere — C-D GRID PIVOT (2026-06-06, user directive)
**Directive:** "for the atmosphere we aimed to stay on C or C-D grid; change carefully,
atmosphere as reference." CONFIRMED by code: the legoESM atmosphere cube is a UNIFIED
**FV3-style C-D grid** (Lin 2004 — D-grid corner winds prognostic, C-grid edge velocities
for transport, vorticity-from-circulation to avoid the Hollingsworth-Kallberg instability
that plagues COLLOCATED A-grid solvers; shared `operators_cdgrid` for atmos+ocean).
**The ocean cube FC-Gram A-grid backend is DEPRECATED** — `OceanModel` maps `fc_gram`,
`fc_gram_cgrid`, `centered`, `finite_volume`, `fv` → `cdgrid` with a DeprecationWarning;
"all cubed-sphere ocean discretizations map to the C-D grid implementation" (ocean_model.py
OCEAN_DISCRETIZATIONS=["cdgrid"]). The FC path only runs when `fc_config is not None`,
documented as a WORKAROUND for "the cd-grid A-L face-edge halo instability under horizontal
density gradients". **So the faithful cube path = the C-D grid (cd-grid backend); the real
job = FIX that A-L face-edge instability (smc03 PGF + partial cells), NOT keep the A-grid
workaround.** All prior cube work (partial cells, smc03 plan) was on the deprecated FC A-grid.

**Shipped this iter (C-D pivot, codex-review pending):**
- REVERTED the half-built FC A-grid smc03 (wrong grid).
- Promoted `extrapolate_below_seafloor` FC-private → public `legoesm.ocean.vertical`
  (grid-neutral; shared by FC + cd-grid; no duplication).
- **Partial-cell substrate on the cd-grid (C-D) backend** (`ocean_pe_cdgrid.py`):
  `is_partial` branch gates below-seafloor u/v at source + extrapolates T/S into rock;
  passes `h_actual=h_k` + `is_active_3d` to the (already partial-ready) shared
  `iterate_eos_and_pressure_anomaly`; gates tendencies `*active_3d` at land-mask step.
  z* path BIT-EXACT (h_actual=None). Removed the `build_cubed_sphere` cd-grid-partial guard.
  5 leaf tests (flat bit-exact, below-seafloor zero + Σh=H, sloped differs, T/S-poison
  isolation, AD-finite).
**A-GRID REMOVED (user directive "if A grid is not used anymore, remove it; codex-validate"):**
deleted `ocean_pe_fc.py` + the FC operator toolkit `operators_fc.py`/`operators_fc_3d.py`/
`fc_gram.py` (used ONLY by the A-grid backend + its FC balanced-init — nothing in
atmos/coupler/barotropic/RCE/spectral uses them) + 6 FC-only test files. `OceanModel` no
longer takes `fc_config`; `_compute_tendencies` always → `ocean_baroclinic_tendencies_cdgrid`;
`fc_gram`/`fc_gram_cgrid` dropped from the legacy-name map (now raise). Runners
(`run_omip_core2`/`run_omip`/`run_ocean_test_matrix`) build the cube on cd-grid only;
`build_cubed_sphere` no longer has `use_fc`; cube `--balanced-init` (FC-gradient) removed →
raises (C-D cube balanced-init is future work). `halo.py` mock-patch list, deck doc,
inline-import audit, coupler/fv3edge tests updated. **The cube ocean is now exclusively C-D
grid** — the "never A-grid / FV3-faithfulness" directive is now enforced by deletion, not just
deprecation. Codex adversarial review of the deletion: CLEAN (no remaining live FC imports/
calls; dispatch correct; `fc_gram` raises). Import-health job confirmed all 4 modules gone +
`fc_gram` raises + cd-grid is sole backend.
**Barotropic also de-A-gridded (codex follow-on):** the cube OMIP `OceanConfig` defaulted to the
forbidden `a_grid` barotropic (the ~40%-non-zonal-eta computational mode). Set
`barotropic_staggering="fv3sw"` (FV3-faithful C-D barotropic) on the cube paths: `run_omip.
_create_setup` (→ also `build_cubed_sphere`), modular `scripts/ocean_test_matrix/setup.py`,
and coupler `component_factory` FULL_3D — each with the SW-core provider import so fv3sw
registers. **Remaining a_grid follow-ups (deferred):** `cs_regional` single-panel (fv3sw on a
panel unverified), the global `OceanConfig.barotropic_staggering="a_grid"` default (broad blast
radius — non-cube grids), stale FC doc refs (SPECIFICATION.md/scaling.md/HTML/
cubed_sphere_pgf_stability.md), the strict C-face wet/rock mass-flux closure (codex HIGH), and a
pre-existing inline-import-budget drift (latlon/mpas/reductions, unrelated to this work).
**NEXT:** cd-grid `--partial-cell` cold-start smoke (does C-D + partial cells delay/fix the
A-L face-edge blowup like FC's step-11→130?) → strict C-face wet/rock mass-flux closure
(codex HIGH; coastline + seafloor faces, cross-seam is_active halo) → smc03 PGF on the C-D
Arakawa-Lamb corner gradient (reuse grid-neutral `pgf_smc03.py`; gradient at D-grid corners
from the 4 cell-centre pressures evaluated at a corner-common reference depth = min of the 4
centroids).

## (DEPRECATED-BACKEND HISTORY, FC A-grid) cubed_sphere — harness COMPLETE; cold-start is the gate
**BREAKTHROUGH (2026-06-06, partial-cell substrate):** the cube backends had **NO
partial-cell support** — `ocean_pe_fc.py`/`ocean_pe_cdgrid.py` hardcoded `z_coord.dz_ref`
and the builder never built a partial coord, so the cube ran **full z\* (sigma-like uniform
stretch) on real NEMO bathy** — the seafloor entered ONLY via the 2D land mask. ALL prior
"lever exhausted" cube verdicts below were on that crippled substrate. FIX: added the
partial-cell substrate to the FC backend (`ocean_pe_fc.py`: `h_actual=h_k`, `dz_actual=h_k`,
3D `is_active` below-seafloor gating of velocities+tendencies, partial-aware single-cell
bottom drag at `bottom_level`) + wired `--partial-cell` into `build_cubed_sphere` via the
canonical `make_partial_cell` (bathy now built before the model so the coord folds it in).
z\* path **bit-exact** (10/10 existing cube-FC tests pass; 4/4 new leaf tests:
flat-bottom bit-exact, below-seafloor zero, sloped-bottom changes result, AD-finite).
**EMPIRICAL (C32 cold-start, dt=30, woa-init, job 8419062):** z\* NaN @ **step 11**;
partial cells **finite to step ~121, smooth ramp** (max|u| 0.27→3.6 m/s over 120 steps),
blowup @ step ~132→NaN 143. **~12× delay + smooth physical spin-up** — partial cells are a
MAJOR cold-start stabilizer ON THEIR OWN (fixed the deep-bottom staircase PGF). Residual
seed moved to a SURFACE coastal front (43.6N/295E lev2, NW-Atlantic shelf), NOT the deep
bottom → that is the smc03-PGF target. NEXT (iter 2): smc03 density-Jacobian PGF on the cube
A-grid (`pgf_smc03_cube.py`, reuse grid-neutral `pgf_smc03.py` kernels) + FC dispatch +
`OceanConfig.pgf_scheme`. The cd-grid (`--cube-no-fc`) substrate is still TODO (guarded).
**Built (d4e6c90b, 26fe0d54, 233c9cf0, 0852f97c):** `build_cubed_sphere` in run_omip_core2
(`_create_setup` cube FC-Gram backend + fv3sw barotropic; NEMO eORCA1 bathy→cube cells via new
`_regrid_curv_to_points` IDW; WOA IC via grid-agnostic `compute_woa_3d` with N-D flood-fill;
external surface-forcing physics so CORE-II tau/q_net are applied — `SurfaceForcingConfig(
scheme="external")`, `shortwave_penetration=None` to avoid sw double-count, drag via model
config). `--grid cubed_sphere --cube-n`; cube-safe `_diag` (4-D umax-location) /
`_grid_lat2d_deg`; `compute_omip2_surface_forcing` cube branch. Runs E2E (step-0 SST 17.6
finite). 3 regrid unit tests; codex CLEAN.
**GATE = WOA-cold-start PGF instability** (NOT face-edge — FC is active; NOT bathy — flat-bottom
also blows; NOT forcing — unforced also blows, just slower). Step-by-step: smooth spin-up to
~0.3 m/s then single-cell 35× jump in ~6 steps. Ignites in MARGINAL SEAS (Med 34.8N/9.8E lev4,
Persian Gulf) and, post-IC-smoothing, the EQUATOR (f→0). INSENSITIVE to A_h(10×)/hyperdiff/
div-damp (bit-identical); WORSE with WOA-smoothing (corrupts IC). **RK3 ported to the cube
(config.baroclinic_rk3, 1c402437, tested + codex-CLEAN) but made it WORSE** (3× tendency
amplifies the unstable mode — unlike the tripole, RK3 is NOT the cube fix). The cube blowup
is a FAST single-cell SPATIAL PGF spike at poorly-resolved marginal-sea cube cells (Med:
the Gibraltar density contrast is sub-grid at C32 ~2.8° → huge 1-cell PGF), NOT a
time-integration-order issue. **LEVER SPACE EXHAUSTED** (RK3 worse, viscosity/hyperdiff/
div-damp no-effect, smoothing worse, flat no-effect) — same dead-end the 1° tripole hit.
**Marginal-sea masking TRIED (a07b8e20) — DEAD END:** masking the Med delayed the blowup
(step 6→10) but it re-ignited at the next semi-enclosed basin (Gulf of Mexico 21N/268.6E).
Whack-a-mole across ALL sharp-gradient semi-enclosed basins (Med, Gulf, Japan, Okhotsk,
Bering...) → masking removes too much ocean, not faithful. Kept as opt-in
`--cube-mask-marginal-seas` (delays; useful only WITH the PGF fix).
**iter-33 — REAL BUG FOUND + FIXED (viscosity gap):** ocean_pe_fc gates ALL core viscosity
(A_h/A_v/hyperdiff on u,v) behind `if physics_fn is None`. The cube's external-forcing
physics (physics_fn set) ran with NEAR-ZERO momentum viscosity (default harmonic A_h=1e4,
enforce_cfl=False) — that's why earlier OceanConfig.A_h overrides were bit-identical (wrong,
skipped field). FIX (2c253db9): build_cubed_sphere configures the PHYSICS harmonic with a
STRONG CFL-CAPPED A_h (1e9, enforce_cfl=True, cfl_dt_estimate=dt) = cube smag-cfl-cap analogue.
Spike 11.6→8.4 m/s (~28%) but INSUFFICIENT alone; dt=10 delays; balanced-init+viscosity also
blows (the cube `_balanced_init_cube` is low-quality: max|u_g| clip-saturates 2.5 m/s
everywhere, deep/equatorial level-of-no-motion reference wrong). NEXT: fix the cube
balanced-init quality (correct LNM at depth + equator) THEN combine with the now-working
viscosity; add partial cells + smc03 PGF. The viscosity gap fix is the first real stack piece.

**DEFINITIVE (iter-32): cube cold-start needs the FULL conditioning stack ported to the
A-grid.** cd-grid FD PGF (8418136) ALSO blows at the marginal seas (Red Sea/Persian Gulf,
step 6-12) → the FC-Gibbs lead below is WRONG; BOTH gradient ops fail. C64 higher-res
(8418081) ALSO blows (step 9) → resolution alone insufficient. The marginal-sea blowup is
the real sharp-front violent geostrophic adjustment, identical mechanism whatever the
gradient/resolution. KEY: lat-lon 1° survives the SAME Med because it has partial cells +
smc03 PGF + balanced-init + RK3 + smag-cfl-cap + polar filter; the cube A-grid `OceanModel`
has NONE of these. **Cube faithful path = port the entire conditioning stack to the cube
A-grid (partial cells, smc03 PGF, working balanced-init, smag-cfl viscosity cap) — a major
multi-iteration dycore project** (the C-grid stack took ~20 iters to build). Individually
each cube lever failed (RK3 worse, balanced-init clip-saturates, masking cascades); they
are needed TOGETHER + likely ~C90 resolution. This is the cube's true scope.

**(WRONG lead, kept for record) FC GIBBS AT SHARP FRONTS:** cube
geostrophic balanced-init (`_balanced_init_cube`, c7d99add) FAILED — `max|u_g|`
saturated the 2.5 m/s clip everywhere, i.e. the geostrophic velocity from the SAME 1°
Med front is ~100× the latlon's. Cause: the **FC (Fourier-continuation) SPECTRAL
gradient rings (Gibbs) at the sharp 1-cell marginal-sea density front** → spurious huge
∇p → both the rest-state PGF spike AND the balanced-init velocity blow up. The latlon
uses a FINITE-DIFFERENCE gradient (no Gibbs) → no blowup. This reframes the cube fix:
**not SMC03/resolution but a NON-SPECTRAL / flux-limited gradient for the PGF at sharp
fronts** (e.g. the cd-grid FD path, or a hybrid FD-at-fronts). NEXT TEST (GPU-queued,
8418081 C64 + a cd-grid-path `fc_config=None` run): does the FD cd-grid path survive
the marginal seas where FC's Gibbs does not? GPU queue currently jammed (~11h).

**(superseded lead) cube needs the SMC03 (Shchepetkin-McWilliams 2003) density-Jacobian PGF on the
cube FC/cd-grid backend** (`ocean_pe_fc.py` / `ocean_pe_cdgrid.py` `_arakawa_lamb_gradient`)
— compute the horizontal PGF as a Jacobian of (in-situ density, depth) instead of a direct
∇p, which removes the sharp-gradient/topography PGF error that seeds the single-cell spike.
The lat-lon path's `pgf_scheme="smc03"` is the reference. Large multi-iteration dycore task.
ALL pragmatic levers exhausted (RK3 worse, viscosity/hyperdiff/div-damp no-effect, smoothing
worse, flat no-effect, marginal-mask cascades). This is the cube's sole remaining gate.

## CUBE C256 (¼°) VERDICT — resolution alone INSUFFICIENT (definitive)
C256 (~0.35°, ¼°-class like eORCA025) + viscosity + woa-init (no balanced): blew up EARLIER
(step 15-30) than C128 (step 37), at the Aegean/Med marginal sea — sub-grid even at ¼° on a
cube. **The resolution trend BROKE → resolution alone does NOT solve the cube.** The eORCA025
tripole works at ¼° ONLY because it has the FULL stack (partial cells + smc03 PGF +
balanced-init + RK3 + smag-cfl-cap); the cube A-grid has ONLY the viscosity cap. C256-woa-only
blew faster than C128+balanced → the stack pieces matter. **DEFINITIVE: faithful cube = ¼° AND
the full conditioning stack ported to the A-grid** (partial cells, smc03 PGF, working no-clip
balanced-init, RK3) — a major multi-piece dycore port + expensive ¼° GPU runs. Not achievable
quickly. Viscosity-gap fix + balanced-init are 2 of ~5 stack pieces done.

## CUBE C128 VERDICT (resolution-convergence confirmed)
C128 (~0.7°, finer than latlon 1°) + strong CFL-viscosity + balanced-init: NaN step 37
(vs C32 step 6, C64 step 9) — **blowup delays ~2× per resolution doubling**. Resolution
is the cube lever (consistent w/ the universal ¼° finding); ≤0.7° insufficient. Path =
**C256 (¼°)** single expensive run (393k cells, dt~3s) — GPU-blocked (jn2808 holds nodes).
Balanced-init still clip-saturates at sharp fronts (max|u_g| hits clip) — needs either ¼°
resolution (fronts carriable) or a no-clip balanced IC. Trend strongly suggests C256 works.

## UNIFYING INSIGHT (iter-34): faithful free CORE-II cold-start needs ~¼° + full stack
Proven across grids: 1° tripole FAILED (config-exhausted) → eORCA025 ¼° SUCCEEDED;
latlon 1° succeeded ONLY with the full conditioning stack (partial cells + smc03 PGF +
balanced-init + RK3 + smag-cfl-cap + polar filter). The remaining 3 grids run at COARSE
defaults — cube C32 (~2.8°), **mpas ico3 (~900 km, even coarser)**, **spectral T21** — and
the WOA density fronts there imply geostrophic jets the coarse grid can't carry → cold-start
blowup, regardless of integrator/viscosity/IC. **So cube/mpas/spectral each need BOTH ¼°-ish
resolution AND their dynamical core's full conditioning stack ported** — a major, expensive,
per-grid undertaking comparable to the original tripole ¼° effort. This is the true remaining
scope; not a quick fix. 2/5 grids (tripole, latlon) are genuinely faithful TODAY.

## Open work toward DONE
1. **cubed_sphere cold-start** — the gate above (port conditioning stack / marginal-sea+equator handling).
2. **mpas** — #160 full-PV TRiSK refactor (remove the relative-only-PV + separate-Matsuno split
   without re-double-counting the barotropic Coriolis) + a run_omip_core2 mpas builder + bathy.
3. **spectral** — add a grid↔spectral CORE-II forcing path (SpectralOceanState lacks grid-space
   state); largest infra gap.
4. **SSS runoff ungate — WIRED (iter, 7ecf0b88)**: `--runoff` loads NEMO's OWN Dai-Trenberth
   file (river+isf+iceberg, 12 monthly) → IDW-regrid → per-step `apply_runoff_step`. Real file
   = host target of the container symlink. Validation run `legoesm_latlon_runoff` (3mo+runoff)
   queued → score SSS vs NEMO when done (was runoff=0-gated).
5. **Transports** (ACC@Drake, AMOC@26N): grid metrics in scorer + NEMO grid_U/V.

## Infra (iter-19, tested + codex-reviewed)
#353 tripolar MPI halo (`parallel/latlon_mpi.py`); grid-agnostic convection
(`ocean/physics/column.py`, opt-in `--convection`); #354 lax.scan forcing
(`compute_omip2_surface_forcing_jax`, opt-in `--scan-block`).
