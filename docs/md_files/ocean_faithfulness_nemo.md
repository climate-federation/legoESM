# Ocean faithfulness vs NEMO — all-grid tracker

**Goal:** every legoESM ocean grid gives a faithful comparison to NEMO ORCA1 (CORE-II NYF).
Codex-review every change. **Shrunk at iter-~50** (was 871 lines). Older blow-by-blow: git history +
commit messages; `OMIP_faithful.md`. Memories: [[omip-faithful-project]], [[omip-rk3-coldstart-solve]],
[[omip-pipeline-coordinate-bugs]], [[omip-smag-cap-stabilizer]], [[omip-postmerge-packages-layout]],
[[omip-cube-cdgrid-pivot]]. Branch `omip-faithful-nemo-comparison`.

## HONEST ACCOUNTING — 5 grids considered: 3 FAITHFUL, 1 parked, 1 ill-posed
| grid | status | SST RMSE vs NEMO Mar (day-90) |
|---|---|---|
| **tripole eORCA025 ¼°** | **FAITHFUL** | **1.15** corr 0.99 |
| **latlon 1°** | **FAITHFUL** | **1.12** corr 0.99 |
| **mpas ico6 ~115 km** | **FAITHFUL** (iter-~50) | **SST 0.84** corr 0.997, **SSS 0.85** corr 0.94 (best) |
| cubed_sphere | **PARKED** — free CORE-II cold-start resolution-limited at C32–C96; all correct numerics committed; needs ¼°+full-stack (future) | n/a (blows at cold-start) |
| spectral | **NOT-MEANINGFUL** — global SH basis can't represent ORCA1 coastlines (Gibbs ringing; model self-declares unsupported #99; bathy builder refuses real geometry; T21 can't resolve Drake) | n/a (by construction) |
**"All 5 grids match ORCA1" is impossible BY CONSTRUCTION (spectral).** Achievable maximum =
the geometry-representing grids; 3/4 of those are faithful, cube needs a major ¼° effort.

## Faithful pipeline (built + proven)
- **Runner `scripts/run/run_omip_core2.py`** = THE faithful path: CORE-II OMIP-2 bulk forcing
  (`omip2_applicator.compute_omip2_surface_forcing` / `apply_omip2_surface_forcing`) applied inside
  `model.step`. Builders: `build_tripole`, `build_latlon_bathy`, `build_cubed_sphere`,
  `build_mpas_ocean`. (`run_omip.py` = SST/SSS RESTORING, NOT faithful; `_create_setup` is the shared
  model builder reused by all.)
- **Scorer `scripts/validate/compare_omip_nemo.py`**: cKDTree-IDW regrid (grid-agnostic, flattens any
  source incl. cube (6,n,n) + mpas (nCells,)), SST/SSS RMSE/corr + band breakdown. **`--nemo-month M`**
  = climatological-calendar-month scoring (perpetual-NYF from WOA annual IC → day-D ≈ calendar day-D;
  MUST score vs same-month NEMO climatology). Day→month (365d): 90→Mar. **`--freeze-clamp-C -1.8`** =
  post-hoc sea-ice floor (diagnostic).
- NEMO ref: ORCA1 monthly `ORCA1_1m_*grid_T.nc` (`.../EXP00/RUN_REF/`). Forcing
  `~/.cache/legoesm/.../core2_nyf/`.

## DONE grids — winning configs
All use run_omip_core2 + `--woa-init --partial-cell --pgf-scheme smc03 --adaptive-implicit-vertadv
--momentum-rk3 --freeze-floor` (latlon/tripole) and the seasonally-matched `--nemo-month 3` scoring.
- **tripole**: `--mesh eORCA025_mesh_mask.nc --dt 75 --balanced-init --min-levels 2 --C-smag-lap 3.0
  --smag-cfl-safety 0.125`. RK3 + implicit_cn barotropic = the key to the corrected-WOA cold-start.
- **latlon**: `--grid latlon_bathy --latlon-res 180x360 --dt 300 --min-levels 2 --C-smag-lap 3.0
  --smag-cfl-safety 0.125 --polar-filter --polar-filter-cutoff-lat 60`.
- **mpas**: `--grid mpas --mpas-level 6 --dt 600 --woa-init --partial-cell --freeze-floor`
  (`_create_setup` mpas defaults: KPP + GM/Redi + smc03/adcroft PGF + implicit-CN barotropic + bottom
  drag + Smagorinsky). ico6 (~115 km ≈ ORCA1), 90-day, **SST RMSE 0.84** (arctic 0.80 w/ freeze-floor;
  bands 0.63-1.15), **SSS RMSE 0.85** corr 0.94 WITH `--runoff` (Dai-Trenberth via
  `apply_runoff_step_mpas`; SSS 1.01→0.85). Best-matching grid in BOTH SST and SSS.

### Key shipped features (tested + codex-clean)
- **Seasonally-matched scoring** `--nemo-month` — exposed that prior "equilibration POOR" verdicts were
  a metric artifact (scoring perpetual-NYF vs annual mean → fake seasonal dipole).
- **Freeze-floor** (sea-ice surrogate, surface `max(T,−1.8°C)`): closes the no-ice Arctic over-cool
  (~4°C). `LatLonCGridOceanModel._apply_freeze_floor` + `MPASOceanModel` port (config `freeze_floor`,
  applied after the conservation fixer, default off → bit-exact). MPAS arctic 6.03→0.80, global 2.20→0.84.
- **Mask-aware Fourier polar filter** (latlon N-pole CFL).
- smag-cfl viscosity ceiling, adaptive-implicit vertadv (NEMO ln_zad_Aimp), partial-cell smc03 PGF, RK3;
  pipeline coord/unit-bug fixes (scorer rad2deg, WOA lat/lon, NEMO mask, _idx_t +3h).
- **SSS runoff** (latlon, NEMO Dai-Trenberth): RMSE 1.89→1.79; not yet wired to cube/mpas.

## MPAS (Voronoi) — the 3rd faithful grid (iter-~50; commits 4647e877, 172aeaff + fixes)
`build_mpas_ocean` reuses `_create_setup('mpas', ico{level})` (full stack), switches to EXTERNAL
forcing (shortwave_penetration=None), regrids NEMO bathy onto Voronoi cells with **nearest-NEMO-cell
land/sea mask** (NOT proximity-to-ocean — over-wets coastlines) + IDW depth, optional partial cells
(min_levels=1) + WOA. Resolution FREE (`--mpas-level`: nCells=10·4^level+2; ico6≈ORCA1) → NOT
resolution-limited like the cube. Wiring: `compute_omip2_surface_forcing` mpas branch (latCell/lonCell,
1-D); `--grid mpas`/`--mpas-level`; `_diag`/`_save_snapshot`/`_grid_lat2d_deg` v-field guards;
`compute_woa_3d` 1-D fixes (lat→latCell + H_bathy.ndim broadcast, cube bit-exact). Stability: ico5 &
ico6 free CORE-II cold-start STABLE (max|u|~0.5 m/s) — the cube's nemesis runs cleanly on pole-free
Voronoi + the full dissipation stack. Codex-reviewed (HIGH land/sea mask + HIGH freeze_floor-wrong-class
fixed). Tests: applicator mpas branch + 3 freeze-floor leaf tests.
**TODO mpas (future, incremental):** runoff DONE (SSS 1.01→0.85, commit 1b4ae6b2); transports; scan-path forcing (perf
at ico7); generic config-override flags (currently uses _create_setup defaults).

## Cube (parked) — free CORE-II cold-start is resolution/discretization-limited (~12 iters, exhausted)
The cube cd-grid (FV3 C-D, the FV3-faithful backend; FC A-grid DELETED) blows on a spurious partial-cell
PGF residual at the steepest sub-grid topography (Med at C32, equator/Indonesia at C96), mid-depth,
basin-scale, undamped. **Every faithful lever exhausted:** RK3, harmonic (max-CFL + A_h-crank),
biharmonic, bottom drag (linear/BBL/quadratic), bathy smoothing, eta-PGF, C-face closure, Adcroft +
smc03 + 2nd-order bottom slope, resolution C32/48/96. **Why it can't be fixed at C32-C96:** the mode is
basin-scale (2-3 cells), needs A_h≳2.3e9, but the grid diffusive-CFL caps A_h ~3× lower at faithful dt →
no scale-selective viscosity damps it; the residual is the SHARED `pgf_smc03` curved-EOS bottom-slope
error (present on all grids, tolerable where well-resolved). C96 + the FULL corrected stack still blows
(equatorial f→0). **Needs ¼° + sub-grid-strait handling + balanced-init (the eORCA025-tripole-scale
effort), or a fundamentally better mid-depth-wet/rock PGF — both major.** All correct cube numerics are
COMMITTED + faithful + gated (bottom drag, bathy smoothing, 2nd-order slope, tunable harmonic CFL cap;
commits b539d578…a2e803f0) — the cd-grid is substantially more correct, ready for a future ¼° push.
NOTE: the gated `pgf_smc03 bottom_slope_2nd_order` (curved-EOS-accurate) is correct faithful numerics
(default off → proven grids bit-exact) even though it didn't gate the cube.

## Spectral — NOT-MEANINGFUL for ORCA1 (documented, scope iter-~50)
SpectralOceanModel is build-complete but its own `__init__` emits a FutureWarning declaring itself
unsupported (#99): land in spectral space → Gibbs ringing + unreliable masking. `rest_state_spectral_
ocean` only supports a smooth tanh high-lat land cap (NO continents); the spectral bathy builder
hard-raises NotImplementedError on real geometry. A global SH basis cannot represent NEMO's discrete
coastlines/sills (ACC/AMOC/gyres) without ringing; T21 (~5.6°) can't resolve Drake Passage. The forcing
gap (no grid→spec path) is secondary. **Verdict: exclude from the faithful set; it's an aquaplanet
solver, a different problem.**

## Regression (iter-~50, job 8424882): 147 pass, 1 PRE-EXISTING fail (not mine)
This session's changes (cd-grid drag/smoothing/2nd-order, shared pgf_smc03, MPAS builder/freeze-floor,
compute_woa_3d, omip2 applicator) introduced **ZERO regressions**. The 1 failure —
`test_partial_cells_mpas.py::test_seamount_centered_stable_over_steps` (max|u| 1.187e-2 > 1e-2 bound
at step 6) — is **pre-existing + unrelated**: `git diff acf44966..HEAD` on that test + `ocean_pe_mpas.py`
+ `mpas_partial_cell_helpers.py` is EMPTY (byte-identical path), and `pgf_scheme="centered"` (its scheme)
does NOT use `reconstruct_harmonic_slopes`. It's a borderline marginal-stability issue on the "centered"
PGF — a scheme the FAITHFUL MPAS does NOT use (faithful = adcroft + implicit-CN). FLAG for a separate
fix (tighten the centered partial-cell PGF or the test bound); out of this session's scope.

## Transports (ACC@Drake, AMOC@26N) — scoped + bounded (scope wf woy1ajpdc); next deepening
~90% of the diagnostics ALREADY exist + are unit-tested: `diagnostics_streamfunction.py`
{`moc_streamfunction(v,h,eta,H,mask,grid)`→ψ[Sv], `barotropic_streamfunction(u,h,mask,grid)`→ψ_bt},
`diagnostics_climate.py` {`amoc_at_latitude(ψ,lat,depth,target_lat=26.5)`→Sv, `acc_transport(ψ_bt,lat)`→Sv},
`spinup.py compute_amoc_from_state_mpas` (MPAS AMOC DONE, Atlantic-masked, tested). Acceptance: AMOC
15±3 Sv, ACC 130±15 Sv. Convention: moc_streamfunction returns Sv; amoc_at_latitude takes Sv ψ as-is.
**APPROACH:** model side = IN-RUN scalars (h=`compute_layer_thickness(eta,H,z_coord)` only live in-state;
append to `diag_timeseries.csv`; NumPy post-loop, no AD/JIT/shared-kernel risk); NEMO side = a separate
offline `scripts/validate/nemo_transports.py` reading grid_U/grid_V + `INPUTS/.../domain_cfg.nc`
(e1v/e3v/bottom_level — NO mesh_mask needed). **BOUNDED SEQUENCE** (ship+validate each):
(1) AMOC@26N tripole (apples-to-apples eORCA1; v in state; Atlantic lon-window mask ~[−80,20]; both
sides) — the minimal first number (~10-20 Sv). (2) AMOC@26N latlon. (3) ACC@Drake tripole+latlon
(barotropic_streamfunction+acc_transport exist; NEMO via ubar — 2-D read). (4) MPAS AMOC (wire the
existing helper). (5) MPAS ACC — the ONLY new numerics (edge-based ψ_bt on Voronoi; codex-review +
analytic check; DEFER if time-boxed). 26N & Drake are both FOLD-FREE (no tripole-cap handling). Gotchas:
basin lon-window (Med/Pacific leakage), ψ sign reconciliation legoESM(+=clockwise) vs NEMO.

## Open work toward maximal faithfulness
1. **mpas runoff** (improve SSS 1.01) + transports (ACC@Drake, AMOC@26N) — deepen the faithful set.
2. **cube ¼°** — the only geometry-grid that COULD match but doesn't; major effort (¼° + balanced-init
   + sub-grid straits, OR shared-PGF overhaul w/ full all-grid re-validation).
3. mpas #160 full-PV TRiSK refactor (HARD; current relative-PV+Matsuno split works, is stable).

## Infra (tested + codex-reviewed)
#353 tripolar MPI halo; grid-agnostic convection (`ocean/physics/column.py`, `--convection`); #354
lax.scan forcing (tripole). `tests/ocean/unit/test_no_scheme_duplication.py` enforces no dycore dup.
