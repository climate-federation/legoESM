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
| **mpas ico6 ~115 km** | **FAITHFUL** (best) | day-90 **SST 0.84**/**SSS 0.85**; **5-yr+SSS-restore: AMOC 20.5 vs 17.7, ACC 144 vs 159/obs137, SST 1.87** corr 0.985 |
| cubed_sphere | **PARKED** — cold-start mode-1 PGF residual NOT resolution-fixable (C256 ¼° blows too, at a cube EDGE near the equator); all correct numerics committed | n/a (blows at cold-start) |
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

### AMOC@26N RESULT (iter-~51) — diagnostic ✓, NEMO ✓ 17.74 Sv; model day-90 UN-SPUN-UP
Implemented + codex-reviewed (2 HIGH + 1 MED fixed: tripole dx_v zero-AMOC, NEMO time-mean order,
sign — then a SIGN-AGNOSTIC surface-referenced upper-mid peak since NEMO ψ is +peaked vs the model's
−peaked). **NEMO AMOC@26N = 17.74 Sv** (robust reader) = spot-on RAPID obs (~17) → the NEMO reference
+ the diagnostic are CORRECT. **Model MPAS day-90 AMOC = −4.63 Sv = un-spun-up** — the deep overturning
needs DECADES to establish; a 90-day cold-start has none (SST/SSS equilibrate in days, the AMOC does
not; NEMO's 17.74 is from a 5-yr-equilibrated RUN_REF). **KEY: the day-90 protocol validates SST/SSS
but NOT transports — transports are EQUILIBRATION-GATED** (need multi-decade runs, compute-heavy, like
the cube ¼°). The AMOC diagnostic (`_amoc26n_diag` in-run + `scripts/validate/nemo_transports.py`) is
committed + correct; a meaningful AMOC match requires an equilibrated run. Multi-year MPAS run launched
to show AMOC DEVELOPMENT + multi-year stability. ACC@Drake is also wind-driven-but-multi-year — same gate.

### CODEX adversarial review of the cube-¼° + mpas-#160 parked conclusions (iter-~51) — BOTH CONFIRMED
- **CUBE: confirm parked.** Codex (code-grounded): the cold-start gate is the BAROCLINIC partial-cell
  PGF residual (the stratified-rest-over-bathy test has exact zero horizontal PGF yet blows at the
  bottom level ~step 180 → the motion is generated in the 3D `dp_dx/rho_0` path, ocean_pe_cdgrid:566).
  The barotropic solver only sees `eta`/`g∇η`; an implicit-CN barotropic damps gravity-wave CFL + null
  modes but CANNOT remove the depth-local baroclinic PGF acceleration injected every PE step → the
  MPAS-winning implicit-CN lever is NOT the cube unblock. Real fix = stronger shared-PGF residual
  removal OR ¼° (both major). Optional 1-shot falsification: `pgf_scheme="zero"` / homogeneous T/S over
  the same bathy → stable ⇒ PGF confirmed (already implied by the stratified-rest blowup).
- **mpas #160: confirm parked.** The relative-PV + split-Matsuno-Coriolis is internally consistent,
  stable, matches NEMO SST/SSS. #160 (full PV q=(f+ζ)/h) is a TRiSK energy/enstrophy-invariant /
  elegance refactor with NO demonstrated SST/SSS-fidelity payoff; park unless a dynamic metric (energy
  drift, barotropic-Rossby phase, near-inertial spectrum) fails.

### PGF-ZERO FALSIFICATION (8426611) — REFUTES "PGF is the SOLE cause"; a 2nd mode exists
Cube rest+topo+no-forcing+RK3, C32: **smc03 control blows step 210** (Med lev14, known); **zero-PGF
(`--cube-pgf-scheme zero`, ALL pressure force removed) STILL blows — delayed to step ~420, seed moves
to 20.9N/277E lev17.** A rest state with NO PGF has NO horizontal momentum force (Coriolis·v / adv·u /
KE all vanish from rest) yet still blows ⇒ **a SECOND cold-start instability source exists, independent
of the baroclinic PGF** (slower; candidate = the barotropic `g∇η` / partial-cell bathy / eta-floor
treatment, or a cube-metric artifact). Removing the dominant baroclinic PGF (smc03, blows 210) reveals
the slower 2nd mode (blows 420). **This OVERTURNS the earlier "baroclinic-PGF-is-the-sole-cause"
conclusion (mine + codex's) and REOPENS the implicit-CN-barotropic question** for the 2nd mode. The
cube needs BOTH the PGF fixed AND the 2nd mode addressed. Pinpoint job 8427117: zero-PGF + FLAT bottom
(stable ⇒ 2nd mode is bathy/barotropic; blows ⇒ pure cube-metric). `pgf_scheme="zero"` is a committed
gated diagnostic (adcroft/smc03 bit-unchanged).

### PINPOINT (8427117) — cube cold-start = TWO bathymetry-driven modes; flat bottom STABLE
zero-PGF + FLAT bottom AND adcroft + FLAT bottom: BOTH **STABLE** (max|u| ~5e-8 m/s = machine noise,
700 steps). So the cube cold-start blowup is ENTIRELY bathymetry-driven; the core cd-grid dynamics are
sound on simple geometry. Evidence matrix:
- smc03 PGF + real bathy -> blows step 210 (Med)        = MODE 1 (baroclinic partial-cell PGF residual)
- zero-PGF + real bathy  -> blows step 420 (Caribbean)  = MODE 2 (bathy-dependent, NOT the PGF)
- any PGF + FLAT bottom   -> STABLE (machine noise)
**Two bathy modes:** (1) baroclinic PGF residual (dominant gate, ~210; smc03/2nd-order reduced not
killed; needs better PGF or 1/4deg); (2) a slower bathy-dependent barotropic/coupling mode (~420,
only visible once the PGF is removed) -- almost certainly the EXPLICIT fv3sw barotropic over real bathy
(g*grad(eta)/eta-floor). **Mode 2 REOPENS the implicit-CN-barotropic path** codex dismissed under the
(now-refuted) sole-PGF premise. Cube needs BOTH fixed (mode 1 is the earlier/dominant gate). Sharper
than the prior "resolution-limited": the gate is specifically the BATHYMETRY treatment (PGF+barotropic).

### MODE-2 mechanism nailed (8427643) — NOT barotropic-damping-fixable; free-surface/coupling
zero-PGF + real bathy + 5x fv3sw barotropic div-damp(600)+vort-damp(0.15): blows IDENTICALLY (step
420, |u| 12.4 vs 12.2). So cold-start MODE 2 is bathy-driven but NOT a dampable barotropic
divergence/vorticity mode -> it is a spurious barotropic FORCING (free-surface g*grad(eta) over the
eta-floored/partial-cell bathy, or the baroclinic-barotropic split reconciliation), which damping
cannot remove but an IMPLICIT-CN free-surface solve could change. **Cube cold-start fully diagnosed
via the 3-stage falsification (zero-PGF -> flat-bottom -> crank-damping):** MODE 1 = baroclinic
partial-cell PGF (blows 210; better PGF or 1/4deg), MODE 2 = bathy free-surface/coupling (blows 420;
implicit-CN barotropic, NOT damping). Both bathy-driven (flat-bottom machine-noise stable -> core
cd-grid dynamics are SOUND). Cube remains parked (both fixes major) but the path is now precise +
evidence-based, not a vague resolution wall. New cube knobs committed: --cube-baro-divdamp/-dampv,
--cube-pgf-scheme zero (all gated/diagnostic; faithful paths bit-unchanged).

### C256 ¼° cube probe (8427650) — mode-1 is NOT resolution-fixable
rest+topo+no-forcing, FULL corrected stack (smc03 + 2nd-order bottom slope + bathy-smooth5 + RK3),
C256 (¼°)/nlev20/dt10: **blows step ~288**, seed `umax_lat 1.1, umax_lon 314.8, lev 13` = a **cube
face EDGE (lon 315) right at the equator**, mid-depth, |u| 400→508 m/s in 6 steps. This REFUTES the
prior "needs ¼° + full stack" hope: ¼° does NOT clear mode-1. Sharper diagnosis — the residual
concentrates where (a) the AL corner gradient crosses a cube face seam AND (b) f→0 removes the
geostrophic restraint, so any spurious/real baroclinic PGF accelerates unchecked. The 3 implicit-CN
grids (tripole/latlon/MPAS) ride through the same WOA cold-start because their unconditionally-stable
barotropic absorbs the fast equatorial adjustment; the cube's EXPLICIT fv3sw cannot. **Cube cold-start
is now fully bounded: mode-1 (baroclinic PGF at cube-edge/equator, ALL resolutions) + mode-2 (bathy
free-surface, implicit-CN). Both fixes major; cube stays parked. Honest max = the 3 faithful grids.**

### NEMO ACC@Drake reference reader (iter-~52) — pairs with the AMOC reader
Added `acc_drake_core` + `nemo_acc_drake` to `scripts/validate/nemo_transports.py` (offline, NumPy):
SIGNED (eastward-positive) net transport through a FIXED model i-column Drake meridian section,
depth+lat-integrated over [-65,-45] (matches model-side `acc_transport` band), from grid_U (uo,e3u) +
domain_cfg (e2u,gphiu,glamu). Codex-reviewed (HIGH staircase + 2 MED + 2 LOW → fixed): fixed-i section
is contiguous-by-construction (no per-row nearest-column staircase gaps), VALID because ORCA1 is a
regular lat-lon grid in the S.Ocean — and that regularity is ASSERTED (raises if the column's circular
lon-deviation over the band exceeds a tol), not assumed; signed (not abs, so a reversed-U bug surfaces);
circular drake_lon. 6 synthetic unit tests (analytic transport, section pick, band exclusion, sign
preserved, circular lon, curvilinear-REJECT) — all pass. ACC spins up in MONTHS (wind-driven, unlike
AMOC's decades) → the 5-yr MPAS run gives a MEANINGFUL ACC match (vs AMOC's equilibration gate).
**NEMO ACC@Drake = 159.26 Sv** (8427657; section lon −68.0, **lon-dev 0.00° → ORCA1 IS regular at
Drake, fixed-i exact**; cross-checks the per-row 159.78; ORCA1 1° runs high vs obs ~137, normal for an
eddy-free coarse model). **Both NEMO refs in hand: AMOC 17.74 Sv, ACC 159.26 Sv.** `nemo_transports.py
... --grid-u X`.

### MODEL-SIDE ACC@Drake diagnostic wired (iter-~52) — symmetric to `_amoc26n_diag`
`spinup.compute_acc_from_state` (latlon/tripole: `barotropic_streamfunction`+`acc_transport` ψ_bt
max−min, Sv↔m³/s round-trip, 2-D-lat row-reduction) + `compute_acc_from_state_mpas` (Voronoi:
SECTION-TRANSPORT — interior edges whose 2 cells straddle the Drake meridian, oriented eastward by
`sign(d2−d1)`, antipodal-|d|<90° guarded, **min-rule edge thickness via the canonical
`min_cell_to_edge`** so partial-cell bottom-steps/coastlines carry no phantom flux). Wired as
`_acc_drake_diag` in run_omip_core2 after `_amoc26n_diag` (both run-end sites; appends `acc_drake_Sv`
to transports.txt; non-fatal). Codex-reviewed: pass-1 HIGH (centered→min-rule) + LOW (pin partial-cell)
fixed. Tests: 69 pass (MPAS-ACC incl. partial-cell min-rule + antipodal + sign; latlon-ACC incl.
2-D-lat). **AMOC min-rule follow-up DONE:** `compute_amoc_from_state_mpas` migrated from the
pre-existing centered edge thickness to the canonical `min_cell_to_edge` (interior edges; one-sided at
boundaries where u≈0) so the MPAS AMOC binning uses the same partial-cell flux closure — codex-clean,
full-cell AMOC tests unchanged (min==centered there) + new dry-edge test. NOTE: the in-flight 5-yr MPAS
(8426097) was launched BEFORE this wiring, so it emits AMOC (now min-rule on future runs) but no ACC; a
follow-up MPAS run will emit ACC.

### MODEL ACC@Drake RESULT (1-yr probe 8428162) — 146 Sv, CLOSE to NEMO; diag works in production
MPAS ico6, 1 model-year, full faithful stack (WOA+partial-cell+freeze-floor+runoff): **model ACC@Drake
= 146.07 Sv** vs **NEMO 159.26** vs obs ~137 — already in-band after only 1 year (wind-driven ACC spins
up in months, unlike AMOC). Validates the wired `_acc_drake_diag` + `compute_acc_from_state_mpas`
(section transport, min-rule) end-to-end in production. **AMOC@26N = 3.67 Sv** (developing — up from the
day-90 −4.63; AMOC is decade-gated, so 1-yr is expected low vs NEMO 17.74). Run STABLE 52560 steps @ 16
steps/s (max|u|~0.5 m/s, finite, rc=0); SST 18.5°C / SSS 34.4. So MPAS now matches NEMO on SST (0.84),
SSS (0.85) AND **ACC (146 vs 159, ~8%)** — only AMOC remains equilibration-gated (needs the multi-decade
run). transports.txt: amoc26N_Sv 3.6739, acc_drake_Sv 146.0682.

### 5-YEAR MPAS RESULT (8426097, COMPLETED 4h33m rc=0) — multi-year stable + SST-faithful; AMOC/SSS drift
MPAS ico6, 5 model-years, full faithful stack (WOA+partial-cell+freeze-floor+runoff). 262,800 steps.
- **STABLE all 5 years** — max|u| grows slowly 0.45→1.14 m/s (physical), finite throughout, no blowup.
  Strengthens the faithfulness claim FAR beyond the 90-day window: the pole-free Voronoi + full
  dissipation stack rides a 5-yr free CORE-II integration cleanly (the cube's nemesis).
- **SST sustained good** — year-5 vs NEMO ANNUAL mean: RMSE 2.27 C, bias +0.34, **corr 0.976**
  (verdict "good"). Bands: tropics 1.58, SH-mid 1.94, NH-mid 3.84, arctic 2.26, antarctic 2.36.
  (Higher than the day-90 0.84 partly because day-90 used seasonally-matched `--nemo-month 3` vs this
  annual-mean scoring + genuine multi-year drift.)
- **AMOC@26N = 44.95 Sv** (developed -4.63@d90 → 3.67@1yr → 44.95@5yr) — OVERSHOOT vs NEMO 17.74.
  Likely spin-up transient + the OLD CENTERED edge-thickness overcount (this run launched BEFORE the
  min-rule AMOC fix). Re-diagnose with min-rule on the next long run + integrate longer to confirm the
  equilibrium; a too-strong overturning would point to GM/Redi eddy transport or convection tuning.
- **SSS fresh drift** — mean 35→31 over 5 yr (RMSE 7.9, bias −4.2 vs NEMO 34.5; score GATED informational).
  A multi-year FRESHWATER IMBALANCE (P−E+runoff with no salt restoring slowly freshens the global mean).
  Day-90 SSS was fine (0.85); the drift is a long-integration issue → **fix = weak SSS restoring**
  (`apply_sss_restoring_step_mpas` EXISTS, was not enabled) or a freshwater-flux balance. NEXT bounded step.

### latlon ACC probe (8428675, 1-yr) — METHOD MISMATCH exposed (ψ_bt-max-min vs section)
latlon 1°, 1 model-year, faithful stack: **ACC@Drake = 211.9 Sv** (vs NEMO 159, MPAS-section 146),
AMOC@26N = 1.61 Sv (un-spun-up, decade-gated). The 211 is NOT apples-to-apples: `compute_acc_from_state`
(latlon/tripole) uses `acc_transport` = ψ_bt MAX−MIN over the whole Drake LAT band (all lons), which
picks up Southern-Ocean gyre extrema + coarse-no-eddy overshoot, whereas the NEMO reader AND
`compute_acc_from_state_mpas` use a SINGLE-MERIDIAN Drake section. The MPAS section ACC (146 vs 159, ~8%)
remains the clean match. Run stable 105k steps @ 7.7 steps/s (latlon per-step host stack is slow),
SST 13.5 / SSS 33.9, finite.
**FOLLOW-UP DONE:** `compute_acc_from_state` (latlon/tripole) rewritten to the SINGLE-MERIDIAN Drake
section (ψ_bt along the Drake column; SIGNED eastward+ via south/north endpoints, NOT whole-band
max−min) → apples-to-apples with the NEMO reader + MPAS section. Codex 3 passes (half-cell U-face column
shift + endpoint-not-max−min ~25% undercount + signed + tripole-dlon=0-sentinel local-spacing estimate
— all fixed; pass-3 clean). 80 tests (gyre-exclusion + dlon-sentinel pins). The 212 above was the OLD
whole-band number; a future latlon run emits the section ACC.

### 5-YEAR MPAS + SSS-RESTORE RESULT (8429183, DONE rc=0) — AMOC OVERSHOOT FIXED, ACC spot-on
MPAS ico6, 5 yr, full stack + `--sss-restore` (τ=365). vs the prior no-restore 5-yr:
| metric | 5yr+SSS-restore | old 5yr (no restore) | NEMO | obs |
|---|---|---|---|---|
| **AMOC@26N** | **20.46 Sv** | 44.95 (overshoot) | 17.74 | ~17 |
| **ACC@Drake** (section) | **143.6 Sv** | n/a (no diag) | 159 | ~137 |
| **SST RMSE / corr** | **1.87 / 0.985** | 2.27 / 0.976 | — | — |
| SSS RMSE / bias | 3.97 / −1.53 | 7.92 / −4.17 | — | — |
- **AMOC overshoot FIXED**: 45→20.5 Sv, now NEAR NEMO 17.7 / obs ~17. Two compounding fixes — the
  min-rule AMOC edge thickness (removed the partial-cell overcount) AND the corrected SSS (no fresh
  drift → physical stratification → physical overturning). Multi-decade-gated transport now lands in-band
  at year 5.
- **ACC@Drake 143.6 Sv** (the in-run section diagnostic) — spot-on obs ~137, ~10% under NEMO 159; the
  apples-to-apples section method. MPAS now matches NEMO on SST, AMOC AND ACC.
- **SST improved** to RMSE 1.87 / corr 0.985 (correct SSS/stratification helped).
- **SSS drift bounded** 31→33.0 (RMSE 7.9→3.97), but τ=365 is TOO WEAK (−1.5 bias). The OMIP-2 standard
  SSS piston velocity (~50 m/300 d) ≈ **τ~60 d** for a 10 m top layer — 6× stronger — which would pull
  SSS to WOA (day-90 with restoring was 0.85). **NEXT: re-run with `--sss-restore-tau-days 60`** for the
  tight multi-year SSS match. Stable all 5 yr (max|u| 0.90, finite). transports.txt: amoc 20.46, acc 143.65.

### VISUAL spatial verification (CLAUDE.md rigor) — MPAS 5-yr+SSS vs NEMO maps
The scorer `compare_omip_nemo.py` already auto-emits `SST_maps.png`/`SSS_maps.png`/`zonal_means.png`
(model | NEMO | Δ, cKDTree-IDW regrid to lat-lon) every run — INSPECTED for the 5-yr+SSS-restore result
(previously validated on RMSE numbers only):
- **SST: large-scale pattern FAITHFUL.** Tropical warm pool, eastern-boundary upwelling, ACC front,
  WBCs all present; Δ mostly within ±5 °C; larger spots only at western-boundary currents (Kuroshio
  ~150E/40N — classic coarse-model WBC path error). Zonal-mean SST curve OVERLAPS NEMO at all latitudes.
- **SSS: open-ocean pattern matches** (subtropical salty maxima, fresh poles, Med/Red Sea); Δ near-zero
  over the open ocean. The global RMSE 3.97 is DOMINATED by a few LOCALIZED extremes — marginal seas
  (Hudson/Baltic) + river mouths (La Plata ~300E/−50N) — plus a uniform ~0.5–1.5 PSU fresh zonal bias
  (the τ=365 weakness; τ=60 closes it). Not a large-scale circulation error.
Verdict: norms + maps agree — the faithful claim holds at the PATTERN level, not just RMSE. (No new
plotter; used the scorer's existing maps.)

### Tripole ¼° + ico7 runs (new-stack validation) — compute findings
- **tripole eORCA025 ¼° day-90 is INFEASIBLE on the 6h short partition** (8429438, cancelled): ~30M wet
  points -> per-step cost so high it never cleared the step-0 -> day-30 interval in 1h24m (day-90 = 103680
  steps at dt=75). The new-stack curvilinear path (SSS-restore lon2d + section-ACC) is instead covered by
  the 2d-lat unit tests + the MPAS/latlon production runs. A tripole transport run would need the
  lax.scan on-device forcing path / multi-node, or a 1° eORCA1 tripole (same cost class as MPAS, but its
  cold-start stability is unverified). Parked as a compute-scaling task, not a faithfulness gap.
- **ico7 (~55 km) at dt=300 BLEW UP** (8429439): non-finite by day 30 — CFL, the ico6 dt=600/115km
  scaling (dt~287 proportional) is marginal at ico7. Relaunched at **dt=150** (8429592) for the
  higher-res SST/SSS check vs the ico6 day-90 0.84/0.85.

### iter-A (2026-06-08): SSS τ=60 + ico7 dt150 harvested → exposed the SALT-BUDGET bug
Two completed runs harvested:
- **5-yr MPAS τ=60** (8429375): SST RMSE **1.73**/corr 0.987; SSS RMSE **0.947**/corr 0.937
  (FIXED from τ=365's 3.97) — but **AMOC@26N collapsed 20.46→5.71 Sv**; ACC 141.8. Stable.
- **ico7 dt=150** (8429592, ~55 km, 90-day): SST RMSE **0.925**/corr **0.996** = EXCELLENT; SSS
  0.933; stable. Higher-res grid is faithful (dt=300 blew; dt=150 rides).
- **AMOC-vs-SSS-restoring tension** (NEMO RUN_REF is itself 5-yr `20000101_20041231` → apples-to-apples,
  NEMO AMOC 17.74). τ=365 overshoots (20.46), τ=60 undershoots hard (5.71): a 4× AMOC swing from
  restoring strength = a *symptom*, not the disease.

### ROOT CAUSE (iter-A): the OMIP-2 path never applied P−E to salinity
`run_omip_core2.py` ran `model.step(surface_forcing=compute_omip2_surface_forcing(...))` which returned
ONLY `tau_x/tau_y/q_net/sw_down` — **`freshwater=None`**. Salinity was forced ONLY by runoff (a
one-sided freshwater SOURCE, ~1.2 Sv global) + SSS restoring. The dominant atmospheric **E−P** (precip
loaded but unused; evaporation never derived from `lh`) was MISSING → net fresh source ≈ 0.3 PSU/yr →
matches the τ=365 −1.53 PSU 5-yr drift. Strong τ=60 restoring masked it by **injecting net salt** (the
restoring is not global-mean-removed), and strong INTERIOR restoring (regional masks are shortest-τ-wins,
so the τ=60 default dominates the Labrador/Nordic deep-convection zone) **suppressed AMOC**. So SSS-good
and AMOC-good were mutually exclusive *only because the salt budget was open*.

### FIX (iter-A, codex-reviewed → pivoted to IN-CORE channel): apply P−E surface freshwater
First attempt populated `OceanSurfaceForcing.freshwater` + applied it post-step via `apply_runoff_step`.
**Codex caught it** (HIGH): the cube's `scheme="external"` physics ALREADY consumes
`surface_forcing.freshwater` → double-count; `--no-emp` wouldn't disable it; the post-step NumPy update
also bypasses the eta free-surface solve, breaks AD/device-residency, and uses `dz_ref[0]`/local-S_top
vs the core's `config.S_ref`. **Pivoted to the in-core channel** the model already provides:
- New `compute_omip2_freshwater_forcing(state, forcing, idx_t, grid, grid_type, runoff_R, emp, ramp)` →
  `FreshwaterForcing(precip=P, evap=E=−lh/L_v, runoff=R)` (`air_sea_fluxes` lh +INTO ocean → evaporating
  lh<0 → E≥0; `net_freshwater_flux`=P−E+R). Shared `_sample_omip2_forcing` (no dup). precip units
  verified kg/m²/s (RAIN+SNOW).
- Run loop delivers it via **`model.step(surface_forcing=sf, freshwater=fw)`** (latlon/MPAS/tripole) →
  the core applies the virtual salt at `config.S_ref` + the eta source in the barotropic solve, on-device
  + AD-safe (`freshwater_closure="virtual_salt_flux"` default). Runoff moved into the SAME channel (one
  freshwater path). CUBE folds `net_freshwater_flux(fw)` onto `surface_forcing.freshwater` (external.py
  applies once — no double-count). `--no-emp` ablation (default ON). Scan path RAISES (precip not on its
  device stack). Resolves codex HIGH 1/2 + MED 3/4/6/7.
- Test: `test_compute_omip2_freshwater_forcing_emp` (P/E exactness, P−E sign, runoff channel, emp=False,
  ramp). Validation job 8432179 (unit tests + latlon+MPAS in-core-path E2E salinity-sign check). Re-codex
  on the revised diff pending.
- **Decisive runs (after validation+codex clean):** MPAS ico6 5-yr + latlon 1° 5-yr, P−E + **weak τ=365
  restoring**. Hypothesis: P−E closes the global budget → weak interior restoring keeps SSS faithful AND
  AMOC near NEMO 17.7. Multi-grid SST/SSS PNGs to follow.

### iter-B SH/Antarctic SST WARM BIAS — FIXED via sea-ice SW albedo (commit 854214b0)
User-flagged (zonal_means/SST_maps PNGs): Antarctic (S of 45S) SST too WARM vs NEMO — MPAS ico6 lego
**5.08** vs NEMO **3.05** (bias **+2.03**); ico7 only +0.32 (res helps; ORCA1 is also ~1°). ROOT CAUSE:
`sw_down`=downwelling SWDN absorbed **100%** — NO surface albedo at all (the cores' `sw*0.94` is a
vertical PENETRATION split, NOT an albedo; total=q_net=100%). NEMO's SI3 ice reflects 60-80% under
Antarctic ice → near freezing; we absorbed it all. `freeze_floor` only caps the FLOOR.
**FIX (gated `--ice-albedo`, default off→bit-exact):** at the forcing PRODUCER,
`sw_net = sw_down*(1 − (a_oc*(1−siconc) + a_ice*siconc))` in BOTH q_net + returned sw_down (applied ONCE;
the 0.94 penetration split then operates on the albedoed sw_net — **codex confirmed keep-0.94**).
`load_nemo_siconc` IDW-regrids NEMO's OWN annual `siconc` (`ORCA1_1y_*icemod.nc`; land=NaN verified) onto
any grid — PRESCRIBED (not SST-triggered → feedback-safe, breaks the +5°C chicken-and-egg). New broadband
constants `alpha_ocean_broadband`=0.06, `alpha_ice_broadband_cold`=0.65. SW-only is the dominant term
(~140/280 W/m² Antarctic over-heat); turbulent/LW ice insulation = diagnostics-gated phase 2. Tests +
E2E (ice-band SST warms less). **LIMITATION:** annual-mean siconc (no monthly icemod) over-ices summer /
under-ices winter. Codex: HIGH (cube combined-scheme solar double-count) is a FALSE POSITIVE for the
faithful path (external scheme + `shortwave_penetration=None`); defensive siconc clip + source-mask caveat
addressed. Combined run (E−P + --ice-albedo) `mpas_ico6_5yr_emp_ice` (8433670) launched → SH-SST PNGs.

### iter-B SSS-drift — FIXED: salt virtual-salt flux now global-normalized (commit 8d774d94)
The E−P run (8432819) confirmed the drift: SSS 34.67@d90 → 34.28@d450 → 34.14@d540 (~−0.4 PSU/yr). CAUSE
(confirmed): `normalize_freshwater` (MPAS, ON) removed the area-mean of the **eta** flux only (volume),
but the **salt** virtual-salt flux used the **RAW** net F_fw=P−E+R → an unbalanced ∮(P−E+R) (interactive
E from biased SST ≠ prescribed P+runoff) drifts the mean salinity even though volume is conserved.
**FIX:** `apply_freshwater_virtual_salt_top(normalize=True)` removes the area-weighted ocean mean of the
PHYSICAL net (P−E+R+ice, NOT restoring) over the EFFECTIVE WET mask before the closure — the SAME area-mean
the eta path removes (volume+salt consistent). Conserves global salt: salt-mass rate = −S_ref·F_fw (h
cancels), so ∮(−S_ref·(F_fw−F_mean)·area)=0 (Griffies). Codex confirmed the math; fixed thin-cell mask +
restoring-exclusion. CAVEATS (documented): local jnp.sum → per-rank under MPI (shared with the eta path;
OMIP=single-GPU); latlon saltnorm CLOSED (commit d44f8ff0 + wired 3ecaf68e; shared `normalized_virtual_salt_flux` helper). Tests:
4 salt-conservation unit tests + 90 ocean tests pass. **DEFINITIVE run** `mpas_ico6_5yr_full` (8435879):
E−P + ice-albedo + salt-norm — the maximal-faithfulness config → the per-grid PNGs + SSS/AMOC/SST scores.
8432819 (no-saltnorm) is the drift baseline (day-720 SSS 33.87 vs definitive day-90 34.68). latlon `latlon_2yr_full` (8436220) = the fully-faithful latlon (E-P + ice + saltnorm). MPI owned-mask normalization (Voronoi halo double-count) = documented follow-up; OMIP runs are single-GPU.

### iter-C CORRECTION (matched-day comparison) — salt-norm is a NO-OP for surface SSS; lever = restoring τ
RAN both runs to matched days: baseline (no salt-norm) and definitive (salt-norm) have IDENTICAL surface
mean_sss at d90/180/270/360 (34.67→34.38). So the salt-norm (global area-mean removal) is a genuine NO-OP:
CORE-II ∮(P−E+R)≈0 already (globally balanced), so the area-mean it removes is ~zero. The salt-norm is a
CORRECT conservation SAFEGUARD (guards against any residual imbalance; keep it) but it does NOT fix the
observed SSS decline. **Re-diagnosis:** the SSS decline is SURFACE freshening (WOA IC 34.63 relaxing under
the surface freshwater flux) + WEAK τ=365 restoring — a surface/restoring lever, NOT a global-salt issue.
mean_sss is the SURFACE mean; the salt-norm conserves GLOBAL (3D) salt, which decouples from surface SSS
under stratification. **CORRECTS the earlier "salt-norm closes the SSS drift" claim.** Real SSS lever =
restoring strength: τ=60 holds SSS near WOA. The pre-E−P τ=60 AMOC collapse (5.7 Sv) was the salt-injection
artifact of an OPEN budget; now E−P closes the SURFACE flux, so τ=60 should hold SSS AND keep AMOC (E−P
provides convection preconditioning, not the restoring). TEST: `mpas_ico6_5yr_s60full` (8437538) = definitive
+ τ=60 → expect good SSS AND AMOC ~17.7. (ice-albedo SST result is independent + still the primary fix.)

### iter-C BASELINE 5-yr E−P result (8432819, mpas_ico6_5yr_emp, τ=365, no ice, no saltnorm)
E−P validated on the DYNAMICS: **AMOC@26N 24.5 Sv** (NEMO 17.7; developed STRONGLY, no collapse — vs the
pre-E−P τ=60's 5.7), **ACC@Drake 135.8 Sv** (spot-on obs ~137), SST RMSE 1.98/corr 0.982 (Antarctic +2.0,
no ice-albedo). **SSS RMSE 5.14 bias −2.49 (BAD)** = the τ=365 surface drift. So E−P+weak-restoring →
good AMOC/ACC/SST, bad SSS → confirms SSS lever = restoring τ (τ=60 test in flight 8437538). AMOC 24.5 is a
+40% overshoot vs NEMO; τ=60 (WOA-pinned SSS) may moderate it. ⚠️ **MHT NH peak 4.99 PW** vs obs ~1.8 (2.7× high). Ruled OUT the net-mass-flux×degC-reference term (~0.06 PW: ρcp·1Sv·15°C, too small). Likely the LATITUDE BINNING in `compute_mht_from_state_mpas` (`spinup.py:709-719` sums ALL edges in a 2° band → over-counts the line-integral across one latitude) OR a geometric factor (dvEdge/sinα). Secondary diagnostic (model-side only, not gating any faithfulness claim); needs careful analysis + a unit test on an analytic transport. DEFERRED, not a hand-wave fix.

### iter-C DEFINITIVE 5-yr result (8435879, mpas_ico6_5yr_full: E−P + ICE-ALBEDO + salt-norm, τ=365)
**SH WARM BIAS FIXED (ice-albedo):** Antarctic SST bias **+2.02→+1.22**, RMSE **2.38→1.55** vs the no-ice
baseline; global SST **1.98→1.82**, bias **0.43→0.09**, corr 0.984. AMOC 23.9, ACC 136 (good). SSS 5.16
(τ=365 drift). **VISUAL VERIFY (zonal_means.png, CLAUDE.md):** zonal-mean SST OVERLAPS NEMO at all lats and
the far SH reaches **−2°C (freezing)** — the ice-albedo confirmed at the PATTERN level, not just RMSE. SST_
maps Δ pale over most ocean; largest residual = NH subpolar/WBC (coarse-model Gulf-Stream/Kuroshio path,
−1.7 NH-mid/−1.1 arctic), NOT the SH. No grid artifacts (smooth Voronoi). SSS zonal-mean ~2-3 PSU too fresh
everywhere = the τ=365 drift (τ=60 run fixes it: live SSS 34.34 vs τ=365's 32.99). PNGs sent to user.

## Open work toward maximal faithfulness
1. **mpas runoff** (improve SSS 1.01) + transports (ACC@Drake, AMOC@26N) — deepen the faithful set.
2. **cube ¼°** — the only geometry-grid that COULD match but doesn't; major effort (¼° + balanced-init
   + sub-grid straits, OR shared-PGF overhaul w/ full all-grid re-validation).
3. mpas #160 full-PV TRiSK refactor (HARD; current relative-PV+Matsuno split works, is stable).

## Infra (tested + codex-reviewed)
#353 tripolar MPI halo; grid-agnostic convection (`ocean/physics/column.py`, `--convection`); #354
lax.scan forcing (tripole). `tests/ocean/unit/test_no_scheme_duplication.py` enforces no dycore dup.
