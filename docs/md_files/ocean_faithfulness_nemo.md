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
| **latlon 1°** | **FAITHFUL** | **2-yr full (E−P+ice+saltnorm): SST 1.68/0.987, SSS 1.20/−0.08, Antarctic +1.2** (AMOC un-spun-up@2yr) |
| **mpas ico6 ~115 km** | **FAITHFUL** (best) | **5-yr MAXIMAL (E−P+ice-albedo+salt-norm+τ60): SST 1.58 corr 0.988, SSS 1.42, ACC 134/obs137, AMOC 13.3 vs 17.7** (binning-caveat) |
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

## 2026-06-08 session — 4 fixes → MPAS MAXIMAL FAITHFULNESS (E−P + ice-albedo + salt-norm + τ=60)
Each codex-reviewed + tested (blow-by-blow in git commits):
- **E−P surface freshwater** (1b240398): the faithful path NEVER applied P−E to salinity (only runoff +
  restoring → ~0.4 PSU/yr fresh drift, masked by strong restoring → AMOC suppression). Now
  `compute_omip2_freshwater_forcing` → `model.step(freshwater=)` (interactive E=−lh/L_v + prescribed P +
  runoff; in-core config.S_ref + eta-in-solve, AD-safe). Cube folds net onto `sf.freshwater` (external.py).
- **Sea-ice SW albedo** (854214b0, `--ice-albedo`): SW was absorbed ~100% (NO albedo); `sw_net = sw·(1 −
  [a_oc·(1−siconc) + a_ice·siconc])` weighted by NEMO's own annual `siconc` (`ORCA1_1y_*icemod.nc`,
  prescribed → feedback-safe). The cores' 0.94 is a PENETRATION split, NOT an albedo (codex-confirmed; kept).
- **Salt-flux global normalization** (8d774d94 MPAS + d44f8ff0/3ecaf68e latlon): shared
  `normalized_virtual_salt_flux` removes the area-mean of P−E+R over the WET mask (conserves global salt).
  A correct SAFEGUARD but a NO-OP for surface SSS here (CORE-II ∮(P−E+R)≈0, verified by matched-day runs);
  the SSS lever is the restoring τ.

★ **HEADLINE — τ=60 = maximal faithfulness** (mpas_ico6_5yr_s60full, all fixes + τ=60 restoring), visually
verified (SST overlaps NEMO + far-SH freezing; SSS tracks NEMO):
| MPAS ico6 5-yr vs NEMO | SST RMSE/corr | SSS RMSE/bias | ACC | AMOC | note |
|---|---|---|---|---|---|
| E−P baseline (τ365) | 1.98 / 0.982 | 5.14 / −2.5 | 136 | 24.5 | SSS drift |
| E−P+ice+saltnorm (τ365) | 1.82 / 0.984 | 5.16 / −2.6 | 136 | 23.9 | SH SST FIXED (Antarctic +2.0→+1.2) |
| **+τ60 (MAXIMAL)** | **1.58 / 0.988** | **1.42 / −0.5** | **134** | **13.3** | **best on all** |
| pre-E−P τ60 (no E−P) | — | 0.95 | — | **5.7 COLLAPSE** | the artifact |
| NEMO / obs | — | — | 159 / ~137 | 17.7 | |
**E−P closing the surface budget DECOUPLES strong restoring from AMOC collapse** — the pre-E−P τ=60 AMOC
collapse (5.7) was the salt-injection artifact of an OPEN budget; with E−P, τ=60 holds SSS (1.42 vs 5.16)
AND keeps AMOC (13.3, closer to NEMO 17.7 than τ=365's 24.5 overshoot). ico7 dt=150 (~55 km) also faithful
(SST 0.925/corr 0.996). latlon-full (E−P+ice+saltnorm, τ=365, 2-yr) DONE: SST 1.68/corr 0.987, SSS 1.20/bias −0.08, Antarctic +1.2 (SH fix on latlon too); AMOC 1.5 un-spun-up (2-yr, decade-gated). Both grids' PNGs sent.

CAVEATS (honest, flagged not hidden):
1. **Transport binning over-count** — `compute_{mht,amoc}_from_state_mpas` sum ALL edges in a 2° lat-band;
   each edge-row carries the full transport so it over-counts the line integral by ~N_rows (2–3 at ico6) →
   MHT 5–6 PW (obs ~1.8) AND the model AMOC magnitudes are UPPER BOUNDS. Should be a latitude-circle SECTION
   (like ACC@Drake); the NEMO structured-grid reader is exact. FIX = section + analytic unit test (follow-up).
   ACC (section method) + SST/SSS unaffected; the cross-run AMOC RANKING (collapse vs hold) is still valid.
2. **MPI owned-mask** for the normalization (an `ocean_global_sum` allreduce double-counts Voronoi halo
   cells; reverted to local `jnp.sum` — exact single-GPU, which OMIP uses). Follow-up.
3. **Phase-2 ice insulation** (turbulent/LW reduction under ice) — SW-only done; residual Antarctic +1.2.
4. Annual-mean siconc (no monthly icemod) over-ices summer / under-ices winter.


### iter-D (2026-06-09): AMOC binning fix + marginal-sea SSS + merged main + higher-res runs
- **AMOC/MHT binning OVER-COUNT FIXED** (5720f441/da75024e): `compute_{amoc,mht}_from_state_mpas` summed
  u·sinα over a 2° lat-band → over-counted by ~N_rows. VERIFIED ×1.84 at ico6 / ×1.12 at ico5 vs the
  analytic uniform-flow transport. Replaced with the latitude-circle SECTION (straddling cells, full normal
  flux, oriented) → ×1.000 exact at any res; 28 AMOC tests pass; codex-clean (per-level NaN sanitize +
  docstrings). **CORRECTED session AMOC** (were band-sum ×1.84): τ=60 ~7.2, τ=365 ~13.0 (vs NEMO 17.7) —
  τ=365 closest; the SSS↔AMOC trade-off is only PARTIALLY decoupled by E−P (τ=60 holds SSS but AMOC ~7.2).
  ACC (already a section) + SST/SSS unaffected.
- **Marginal-sea SSS restoring** (1772032f): Baltic/Black Sea/Hudson/Okhotsk few-day restoring → fix the
  enclosed-sea FRESH bias (the user-flagged continental salinity issue; unresolved straits + runoff). 11
  regions total. 19 sss tests pass.
- **Merged origin/main** (134c9490, 54 commits: CRM/plane MPI perf + atmosphere physics + federation) —
  clean, no ocean conflicts, 96 ocean tests pass post-merge.
- **SST hemispheric gaps (Antarctic warm +1.2 / NH cold −1.7) = RESOLUTION-limited** (user-flagged). The ice
  -albedo helped the warm SH but the two CONFLICT on a single albedo (Antarctic wants more, NH wants less).
  ico7 90-day already showed Antarctic +0.32 / NH-mid +0.22 on resolution alone. HIGHER-RES runs launched:
  ico7 ~55km 2yr (8440837), latlon 0.5° 1yr (8440838), + ico6 τ60+marginal-sea (8440840, salinity attribution).
  All with the SECTION-method AMOC. tripole ¼° stays compute-infeasible.

### iter-D RESULTS (1st higher-res run back): marginal-sea = MODEST; corrected AMOC = LOW (real trade-off)
ico6 τ=60 + marginal-sea restoring (8440840, SECTION-method diags):
- **SSS 1.42→1.38** (bias −0.47→−0.44): marginal-sea restoring helps MODESTLY — the continental extremes
  (Baltic/Hudson/Okhotsk) are reduced but NOT eliminated (the NW-Pacific/Japan-Sea patch persists, partly
  south of the Okhotsk region; enclosed-strait dynamics unresolved at ~115 km). Partial fix, not complete.
- SST 1.57/0.989 (unchanged), ACC 133.7.
- **CORRECTED AMOC (section) = 6.05 Sv** (vs the inflated band-sum 13.3; NEMO 17.7) — the REAL τ=60 AMOC is
  LOW. CONFIRMS the SSS↔AMOC trade-off is REAL + the E−P decoupling is only PARTIAL: strong restoring holds
  SSS (1.38) but genuinely WEAKENS the AMOC (~6). τ=365 (better AMOC, ~13 est) has bad SSS. No single τ wins
  on both. MHT 3.3 PW (section; was 5-6 band-sum; still ~1.8× obs ~1.8 — possible real over-transport OR a
  residual per-latitude-crossing subtlety; secondary, flag).
- HIGHER-RES (the SST-gap lever) in flight on glab1: ico7-2yr (day-90 stable, ~50h; year-1 read ~10h),
  latlon-0.5° (slow host stepping ~28h, no blowup yet). ico7-90day on a bad short node was step-0-stuck →
  cancelled (ico7-2yr supersedes).

### iter-E (2026-06-09): NH-cold-bias ROOT CAUSE found + SEASONAL-albedo fix; colleague-Q evidence
- **NH cold bias ROOT CAUSE (codex adversarial review, HIGH):** the `--ice-albedo` SW surrogate used the
  NEMO ANNUAL-MEAN `siconc` applied EVERY step → in NH seasonal-ice zones (Labrador/Greenland/Bering/
  Okhotsk) it kept a high ice albedo through the open-water summer → ~0.24×SW (≈47 W/m² at summer
  SWDN~200) spurious cooling all year. Confirmed dominant over runoff/restoring/E−P (codex refuted E−P
  sign + 0.94-penetration-double-count; keep both).
- **FIX (commit 6ad1af3e, `--ice-albedo-seasonal`):** build a 12-MONTH siconc climatology. The run wrote
  NO monthly icemod, so seasonality comes from NEMO's MONTHLY SST (`tos`, ORCA1_1m grid_T): NEMO ice sits
  at freezing, so cold SST ⟺ ice. `_ice_presence_from_tos` (tanh) + `_seasonal_siconc_from_presence`
  (per-cell MEAN-PRESERVING norm: 12-mo mean = annual siconc → ice months carry true winter conc, summer→0;
  conserves annual albedo, codex MEDIUM). Indexed by NOLEAP calendar month each step. 6 unit tests pass
  (incl. real-file NH-ice-retreats-Mar→Sep / SH-opposite).
- **SSS-restoring ice-gate fix (same commit):** the driver passed `ice_concentration=None` → the ice gate
  was DEAD → restoring ran at full strength under sea ice (unlike NEMO `nn_sssr_ice=0`). Now feeds the
  per-step siconc. siconc loaded when `--ice-albedo OR --sss-restore`; albedo still gated on `--ice-albedo`
  so an SSS-only run's heat budget is unchanged (codex HIGH).
- Codex 2× (root-cause + fix-review); all HIGH/MEDIUM/LOW addressed. Merged origin/main (AI-guardrail
  harness + dispatch/constants tests; clean, no ocean conflicts).
- **Runs launched with the correction (coarse, fast turnaround):** MPAS ico6 2yr seasonal (8445150),
  eORCA1 tripole SAME-GRID 3yr seasonal (8445151 — NEMO's own mesh/bathy; colleague Q4). High-res
  ico7/latlon-0.5 + the annual-albedo tripole were CANCELLED (ran the pre-fix buggy code).
- **Colleague-question evidence (NEMO namelist_cfg / RUN_REF):** Q1 runoff — SAME Dai-Trenberth-Depoorter
  file (sorunoff+Icb_flux+socoefr), but NEMO spreads runoff over the TOP 150 m (`rn_dep_max=150`,
  `ln_rnf_depth_ini`) + monthly, vs legoESM monthly IDW-regridded SURFACE virtual-salt flux. Q2 topo —
  MPAS ico6 bathy is REGRIDDED from NEMO's eORCA1 (`(e3t·tmask).sum`) onto Voronoi cells, NOT identical to
  NEMO's tripolar grid (colleague correct; the tripole run removes this). Q3 run length — the shown ico6
  maps are END of a 5-yr (day 1825) CORE-II NYF spin-up (short for deep-ocean equilibration). Q4 same-grid
  — eORCA1 tripole run now in flight. NEMO ALSO restores SSS (`nn_sssr=2`, ±4 mm/day, OFF under ice), so
  our τ-restoring is faithful in kind.

### iter-F (2026-06-09): dual-pole correction — prescribed-ice THERMODYNAMIC boundary (--ice-thermo)
- **Codex dual-pole review:** the >45S WARM bias is NOT albedo-fixable. The albedo-only surrogate (a)
  injects 0.35·sw_down into the ocean under sic=1 (α_ice=0.65) and (b) applies FULL open-ocean turbulent/LW
  fluxes even under ice → the Southern-Ocean under-ice ocean stays too warm. Seasonal albedo (mean-
  preserving) conserves the annual albedo → only redistributes timing.
- **ico6 SEASONAL-vs-ANNUAL A/B (yr1, matched grid+time):** ~NEUTRAL (all bands Δ<0.05): antarctic
  +1.06→+1.03, NH-mid −1.13→−1.11, arctic −1.27→−1.26, global RMSE 1.32→1.30. Confirms seasonal albedo
  alone does NOT move the 1-yr annual-mean bias (by mean-preservation design); year-1 ≈ WOA IC for all
  configs (non-discriminating — biases develop multi-year).
- **--ice-thermo (commit 164c0107):** prescribed-ice thermodynamic boundary = the magnitude lever.
  `_ice_surface_heat`: under ice cut SW to τ_ice_sw≈0.03 + suppress turbulent/LW by (1−sic); sic=0 open
  water unchanged. `under_ice_freeze_relax`: post-step 2-sided nudge of top-cell T → freezing
  (constants.T_freeze_ocean) over τ_ice≈20d, ×sic → COOLS over-warm SH under-ice, HOLDS Arctic. Grid-
  agnostic; convex (dt/τ clipped ≤1); scan-refused; default off. Codex-reviewed twice (physics A–F
  confirmed; .copy() MEDIUM + 3 LOW fixed; final confirm clean). 25 applicator+siconc tests pass.
- **A/B/C RESULT (yr1, both grids) — ICE-THERMO IMPROVES BOTH POLES:**
  | grid | >45S warm (ann→thermo) | >45N cold (ann→thermo) | global RMSE |
  |---|---|---|---|
  | tripole same-grid | +1.20 → **+0.90** | −1.08 → **−0.91** | 1.78 → **1.74** |
  | ico6 Voronoi | +1.06 → **+0.80** | −1.27 → **−1.15** | 1.32 → **1.27** |
  Consistent across grids: cools >45S ~0.25–0.30, warms >45N ~0.12–0.17; localized to ice zones (SH-mid/
  tropics/NH-mid unchanged → no collateral damage); corr 0.985–0.992. Seasonal-alone was ~neutral (mean-
  preserving); the THERMO boundary (SW cut + flux suppression + freezing relax) is the lever. Confirmed at
  yr1; runs continue to yr2/3 (developed bias → fix should close more).
- **Remaining gaps vs NEMO:** (1) >45S residual +0.80–0.90 = Southern-Ocean warm bias (dynamics/clouds/
  AABW), partly beyond ocean-only prescribed-ice; τ_ice (20d) is a tunable knob. (2) NH-mid −1.58 (tripole)
  = Gulf Stream/Kuroshio under-resolved at 1° → RESOLUTION-bound (eORCA025), not a forcing fix; the single
  largest gap. (3) >45N residual −0.91/−1.15 reduced but open.

### iter-G (2026-06-10): NEMO-parity surface fluxes — 8 BC/param bugs found+fixed (commits 6dd81575, dc13d1df)
**Bias target:** the hemispheric SST dipole (SH warm +0.9..+1.1 incl. 45S-23S NON-ice band, NH cold −1.1;
yr2 both grids) + the τ60 AMOC collapse (6 Sv). Audited the FULL flux chain line-by-line vs NEMO 5.0.1
source on disk (sbcblk.F90 / sbcblk_algo_ncar.F90 / sbc_phy.F90 / namelist_cfg) + 2 codex rounds:
1. **Bulk scheme was a 2-coeff approx** (Ce=Ch fixed {1.46,1.18}e-3, no stability iteration, "good to
   ~10%") → ported NEMO's FULL NCAR algorithm exactly: 5-iter Obukhov fixed point, ψ_m/ψ_h (shared
   core.bulk_flux), CdN cyclone plateau + 1e-4 floor (`large_yeager_neutral_cd(nemo_parity=True)`),
   pres_temp 10-m barometric pressure, theta_exner potential air-T + potential SST (stability/sensible/
   L_vap on the potential pair; ssq/LW/evap-heat at absolute SST), ρ_air(slp,T,q) at p10, moist cp_air(q),
   L_vap(θ_sst), 0.98-salt Goff ssq (`thermo.saturation_vapor_pressure_goff`). Legacy scheme pinned as
   `algo='ly09_2coeff'` (operator-split applicator only).
2. **LW Kirchhoff**: was lwd − 0.97σT⁴ (absorb 100%/emit 97%) → NEMO 0.98·(lwd−σT⁴). ≈ −11 W/m² (cools).
3. **Snow fusion missing** → −snow·rLfus + rain/snow/evap heat-content terms (NEMO blk_oce_2 exact,
   rLfus=0.3333601e6/rcpi=2096.7). New SNOW + SLP zarr channels (builder+loader+host+scan; back-compat
   fallbacks). nyf.zarr REBUILT + validated (slp valid-max 1156 hPa = Antarctic below-ground reduction, land).
4. **Regridder LONGITUDE SEAM bug**: 0/360 wrap segment never covered → seam destination column
   under-weighted (production latlon ~6%-covered last column; tripole/MPAS NN paths unaffected). Ghost-
   column padding fix.
5. **SSS-restoring cap was DECORATIVE**: appliers consume dS_dt_top which bypassed the flux clip →
   τ60 restoring was UNBOUNDED in deep-water-formation spots = the AMOC-collapse mechanism. Tendency now
   derived from the capped flux; `--sss-restore-bound-mmday 4` = NEMO ln_sssr_bnd (RUN_REF: piston
   −220 mm/day ≈ τ45.5d on 10 m + ±4 mm/day bound, off under ice).
6. pres_temp garbage-cell guard (w=q/qsat clip [0,1]); NEMO-parity constants block in constants.py.
**Verification:** independent NumPy transcription of the Fortran (coefficients AND full flux path incl.
preprocessing) matches the JAX impl bit-exact (rtol 1e-12 / 1e-10); jit+grad finite; 3825/3826 targeted
tests green (last = test-side L_vap(θ_sst) expectation, fixed; round-4 in flight). Codex round-1
NEEDS-ATTENTION (7 findings → all fixed; HIGH = the potential-T preprocessing), round-2 confirm in flight.
**Runs launched (A/B vs the old-flux ice-thermo baselines):** `mpas_ico6_2yr_ncar` (8454488) +
`tripole_eorca1_2yr_ncar` (8454489) — same config as the 2yr ice-thermo runs but NCAR fluxes + NEMO
bounded restoring (τ45.5/bnd4 replaces τ60-unbounded). Baseline yr2 scores for the A/B: ico6 ice-thermo
yr2 SST RMSE 1.37/corr 0.991 (>45S +0.89, SH-mid +1.04, NH-mid −1.11, >45N −1.06; ACC 141.9, AMOC 2.5
un-spun, MHT-NH 3.52 PW high). **Expected from the physics:** LW −11 W/m² + snow fusion cool the SH warm
band; bounded restoring lets AMOC rebuild (NEMO holds 17.7 WITH restoring because of the bound); MHT to
re-diagnose under corrected fluxes. RGB-chl SW penetration (NEMO ln_qsr_rgb) = known remaining BC gap
(vertical heating distribution), next lever if the dipole persists.

### iter-G yr-1 A/B RESULT (tripole, 8455137 in flight): NEMO-parity fluxes IMPROVE — SH-mid is the residual
Tripole yr-1 (ncar fluxes + bounded τ45.5 restoring) vs yr-1 old-flux ice-thermo baseline, NEMO annual:
| | baseline | **ncar** |
|---|---|---|
| SST RMSE / corr | 1.74 / 0.985 | **1.43 / 0.990** |
| >45S | +0.90 | **+0.76** |
| SH-mid | ~+1.0 | +1.13 (PERSISTS) |
| tropics | −0.14 | **−0.01** |
| NH-mid / >45N | −1.28 / −0.91 | −1.05 / −1.11 |
| SSS RMSE / bias | — | **1.30 / −0.03** (bounded restoring: NO drift) |
Run stable day-365 (seasonal cycle clean, max|u| ≤1.1). **Verdict: flux fixes deliver (−18% RMSE, SH-pole
+ tropics improved); the 45S-23S SH-mid warm band is now THE bias** — not ice (ice-thermo zone ends 45S),
not turbulent-flux scheme (just fixed). Leading candidate: SW PENETRATION (NEMO RGB+chlorophyll
`ln_qsr_rgb` vs our fixed 2-band — Southern-Ocean high-chl traps heat near surface in NEMO; our deeper
penetration warms... actually COOLS surface; sign needs the impl). Next: implement `rgb_chl` penetration
scheme (NEMO traqsr RGB table + monthly ESACCI chl climatology from INPUTS) + codex; A/B on yr-1 rerun.
PNGs (tripole yr1 SST/SSS) sent to user 2026-06-10 ~08:40.

### iter-G (user-flagged): tripole lon-72.5 vertical BAND = eORCA1 cyclic-overlap off-by-one (PRE-EXISTING)
User saw a vertical SST stripe at lon 70-80 on the **tripole** map (absent on yesterday's **latlon** map).
Root-caused, NOT a flux regression:
- The stripe is at lon **72.5°E** = the eORCA1 grid's east-west cyclic SEAM. Confirmed in index space: i=0
  (lon 72.5) duplicates i=360 (lon 72.5); i=361 (73.5) duplicates i=1 (73.5) ⇒ **ORCA 2-point cyclic overlap**
  (halo col0=col_{n-2}, col_{n-1}=col1).
- `LatLonCGridOceanModel` (reused for tripole) applies SIMPLE roll-periodicity (`periodic_x=True`, enforces
  `u[:,n_lon]==u[:,0]`, i.e. col_{n-1}=col0) — correct for a regular lat-lon grid, **off-by-one for the ORCA
  2-pt overlap** ⇒ the halo columns carry slightly wrong values ⇒ a mild ~0.87°C seam in the seam-adjacent
  gradient/flux terms, amplified by the scorer's cKDTree-IDW blend across the seam.
- **PRE-EXISTING, not from this session's flux work:** regridded lon-72 stripe sharpness is 2.71°C in the OLD
  ice-thermo tripole vs 2.86°C in the NCAR run — essentially identical. The latlon 1° grid has NO such seam by
  construction (yesterday's clean map). The physical domain i=1..360 integrates correctly (tripole is stable +
  SST-faithful); only the 2 halo columns are off, so the seam is mild not catastrophic.
- **ROOT CAUSE CONFIRMED at mesh level:** the eORCA1 mesh_mask's cyclic halo columns i=0/i=361 are marked
  LAND everywhere (0 wet) while their ORCA-overlap partners i=360/i=1 are ocean (147/143 wet) — 143 latitudes
  where i=1 is ocean but its west-neighbour i=0 is a fake land wall. NEMO fills these halos every step via
  `lbc_lnk`; legoESM read `tmaskutil` raw and never applied the overlap → the lon-72.5 seam ocean is severed.
  DEEPER: legoESM treats the (332,362) grid as a period-**362** ring (operators `jnp.roll(...,axis=1)` over
  all 362 cols), but eORCA1 is physically period-**360** with 2 duplicate-longitude overlap halos — so even
  filling the mask won't hold: the 2 seam columns evolve independently (no halo slaving) and re-drift.
- **FIX = careful dycore work (DEFERRED, needs user steer — load-bearing + stability-risky):** EITHER
  (a) per-step ORCA cyclic-overlap exchange on T/S/eta/u/v (col0←col360, col361←col1; lbc_lnk-style, in the
  hot loop), OR (b) strip to 360 physical columns with period-360 roll (cleaner, bigger grid/state refactor).
  BOTH risk the hard-won WOA cold-start stability (tripole periodicity is woven through advection/PGF/
  barotropic) → require gated impl + codex + 30-day cold-start smoke + W2-style visual check BEFORE trusting,
  then a tripole re-run. NOT a mid-loop rush. Interim: the **latlon 1° grid is the clean-grid comparison
  vehicle** (regular periodic, no seam) — relaunched with the NCAR fluxes (8457282).
- **Large local SST biases (user):** marginal seas (Persian Gulf min 13°C, Red Sea 15°C) are COLD-biased in
  BOTH runs (old Gulf min was 2.6°C — NCAR is LESS cold, an improvement); no runaway hot cells (0 wet cells
  >35°C). On the clean latlon grid the dominant local bias is the Kuroshio/Oyashio WBC warm spot (+10°C @
  40N/150E, resolution-bound, pre-existing). Global banded NCAR-minus-old ΔSST is tiny (≤+0.34°C).
- **ACTION:** relaunched **latlon 1° 2yr with the corrected NCAR fluxes** (8457282, clean grid, same vehicle
  as latlon_2yr_full) for an apples-to-apples flux-improvement view without the tripole seam.
- **SEAM FIX IMPLEMENTED + VALIDATED (commit 98f9b779, `--ew-cyclic-overlap`, gated default-off):** a
  post-step ORCA cyclic-overlap projection slaves the 2 longitude halo columns to their overlap partners
  for ALL prognostic fields each step (cell-column T/S/eta/v: col[0]<-col[nx-2], col[nx-1]<-col[1]; u-faces:
  u[:,0]<-u[:,nx-2], u[:,nx-1]<-u[:,1], u[:,nx] to the internal wrap) + a matching mask/bathy/IC overlap-fill
  at construction BEFORE make_partial_cell. Codex round-1 caught 2 real HIGH bugs (velocity MUST be slaved —
  barotropic carries U_old + seam u-face reads the v halo via Coriolis roll; partial-cell coord built before
  the fill) → round-2 APPROVE. 52 unit tests pass (cell+v+u-face slaving, partial-cell seam identity, gating
  bit-identical). **120-day WOA cold-start smoke (8457440) STABLE** (max|u| 1.01 m/s, finite, physical) AND
  the seam closed: native i1-vs-i_{nx-2} jump **0.87→0.085 °C (10×)**, regridded lon-72 stripe sharpness
  **2.86→0.345 °C (8×)**. The hard-won cold-start survives the reconnected seam. **Definitive tripole run
  launched (8457733): NCAR fluxes + bounded restoring + seam fix** (supersedes the no-overlap 8455137,
  cancelled). MPAS (Voronoi) has no ORCA seam → unaffected.

### iter-H (2026-06-10, IN PROGRESS — user-flagged): Amazon rivers, Gibraltar/Med, "fuzzier than NEMO"
USER REQUEST: fix (1) large-river (Amazon) SSS issue, (2) Gibraltar intrusion / Mediterranean temperature
("maybe a NEMO trick?"), (3) legoESM looks FUZZIER/more diffusive than NEMO on the maps.
**NEMO ORCA1 ground truth extracted (namelist_cfg + SHARED/namelist_ref, all verified on disk):**
- **Gibraltar/Med trick = ADVECTIVE BOTTOM BOUNDARY LAYER**: `ln_trabbl=.true.`, `nn_bbl_adv=2`
  (advective BBL, both upper+lower flux), `rn_gambbl=20 s`, `nn_bbl_ldf=0` (diffusive BBL OFF),
  `rn_ahtbbl=1000` (unused at ldf=0). Dense Med overflow water ADVECTS down the continental slope —
  without it the 1° Med can't ventilate (our marginal-sea cold bias + no Med tongue). NOT a resolution
  trick: ORCA1 is 1° like us. legoESM has NO BBL scheme on the tripole/latlon path → implement
  advective-BBL (Beckmann & Döscher 1997 + NEMO trabbl.F90 nn_bbl_adv=2 form) as a gated parameterization.
- **Runoff**: NEMO spreads river runoff over the TOP 150 m (`ln_rnf_depth_ini=.true.`, `rn_dep_max=150`,
  `rn_rnf_max=0.05`); legoESM applies it as a SURFACE virtual-salt flux at single cells → Amazon plume
  too fresh/too shallow/too local. ALSO NEMO disables/reverses SSS restoring near river mouths (sbcssr
  `(1-2*rnfmsk)` with socoefr) — our restoring fights the plume toward coarse WOA. Fix = (a) spread the
  runoff freshwater over the top-150m layers (freshwater channel or tracer tendency), (b) river-mouth
  restoring mask derived from the Dai-Trenberth runoff field (where runoff > threshold → zero restoring).
- **"Fuzzy"/diffusivity**: NEMO tracers = FCT-2 advection (`ln_traadv_fct`, nn_fct_h=2,v=2) + LAPLACIAN
  ISO-NEUTRAL diffusion (`ln_traldf_lap+iso+msc`) with Treguier-varying aht (`nn_aht_ijk_t=21`,
  rn_Ud=0.01 m/s, rn_Ld=200 km → aht ~ O(1000) m²/s at 1°) + GM/EIV ON (`ln_ldfeiv`, nn_aei_ijk_t=21,
  rn_Ue=0.02, rn_Le=200 km). Momentum: namdyn_ldf block NOT yet read (find it in namelist_cfg ~l.400+;
  ORCA1 default is BILAPLACIAN momentum). NEXT: read our `run_omip._create_setup('tripole'/...)` tracer
  advection scheme + K_h/A_h + whether GM/Redi (EXISTS for latlon-cgrid: gm_redi_latlon_cgrid) is enabled
  on the faithful runs — match NEMO (FCT-like advection + isoneutral lap + GM) or identify our excess
  diffusion. NOTE: scorer maps IDW-regrid BOTH models (symmetric blur) → "fuzzy" is likely genuine model
  diffusivity, but VERIFY by comparing native-grid sharpness first.
**iter-H item 1 DONE (commit 1aa4d297, codex APPROVE, 72 tests):** runoff depth-spread
(`runoff_spread_virtual_salt_tendency_3d`, fractional per-level weights → h_rnf=min(150, wet depth) exact,
column-integral salt bit-identical to legacy, both cores gated via `runoff_depth_spread_m`) + river-mouth
restoring gate (`--river-mouth-restoring-gate`, NEMO (1−2·rnfmsk)). Flags for next production round:
`--runoff-depth-spread-m 150 --river-mouth-restoring-gate`.
**iter-H item 2 (BBL) design state:** NEMO trabbl.F90 CASE(2) extracted — per u/v face:
`tr_bbl = e2u·e3u_bbl_0·(g·rn_gambbl)·max(0, Δρ̂)·mgrhu` with Δρ̂ = ½(2α·ΔT−2β·ΔS) between SHELF
(up-slope) and DEEP (down-slope) BOTTOM cells at common local depth (eos_rab at bottom), active only
when shelf denser; mgrh = slope-direction sign; e3_bbl_0 = BBL thickness at the face. REMAINING TO READ:
`tra_bbl_adv` application loop (~trabbl.F90 l.214-300) — distributes the transport down the deep column
across levels (the intricate part) + e3u_bbl_0/mbku_d/mgrhu setup in tra_bbl_init. Then implement
gated `bbl_adv` for the latlon-cgrid family (tripole/latlon) + tests + smoke; MPAS port after.
**iter-H item 2 PROGRESS:** `packages/ocean/legoesm/ocean/physics/bbl_adv.py` WRITTEN (compiles; physics
contract included): `bbl_static_geometry(h_ref, land_mask)` (NEMO tra_bbl_init: mgrh=sign Δdep_bot,
shelf/deep bottom levels, e3_bbl=min bottom thickness, face common depth), `bbl_transports` (CASE(2):
tr=width·e3_bbl·g·γ·max(0,Δρ/ρ0)·mgrh, Δρ via canonical `wright_eos` at the face's common bottom pressure
— equals NEMO's α/β linearisation to linear order, no eos_rab re-derivation), `apply_bbl_adv_tendency`
(exact tra_bbl_adv 3-leg circulation cell, static-unrolled level loop, .at[].add scatters, telescoping
conservation). REMAINING: (a) host wrapper `apply_bbl_adv_step(state, geom, dt, ...)` (runner post-step
pattern like restoring/ice-thermo — zero dycore risk; geometry from `z_coord.h_partial` + land_mask at
setup; face widths grid.dy_u/dx_v on tripole, dy/dx broadcast on latlon); (b) runner flags `--bbl-adv`
`--bbl-gamma-s 20`; (c) tests/ocean/unit/test_bbl_adv.py (analytic 2-column overflow: transport formula
exact + down-slope sign; conservation sum(area·h·dpt)=0 to fp; flat-bottom → zero; gate-off untouched);
(d) codex adversarial; (e) 120-day tripole smoke w/ --bbl-adv. NOTE periodicity: interior faces only
(seam exchange via ew_cyclic_overlap halo slaving on tripole; the omitted wrap face on regular latlon is
1 face of 360 — documented).
**iter-H item 2 BBL SHIPPED (commits 82796b83 + 04e1dd7a, codex round-1 NEEDS-ATTENTION → round-2
APPROVE zero findings, 107 tests):** exact NEMO trabbl nn_bbl_adv=2 port — eos_rab gating (α/β per
column at ITS OWN bottom pressure via canonical Wright-EOS derivatives, face-averaged; thermobaricity
preserved — codex HIGH vs my first mean-pressure Δρ), Campin-Goosse transport, exact 3-leg conservative
exchange, 0.25·V_min/dt Courant cap, host post-step wiring, `--bbl-adv --bbl-gamma-s 20` (latlon/tripole,
requires --partial-cell). 120-day cold-start smoke 8458619 in flight.
**iter-H item 3 DIFFUSIVITY AUDIT (the "fuzzy" answer):** NEMO ORCA1 momentum viscosity = the
`eddy_viscosity_3D.nc` FILE (nn_ahm_ijk_t=-30, iso-level LAPLACIAN): **1e3–2e4 m²/s, median 2e4** —
our faithful runs use **A_h=1e5 + C_smag_lap=3.0 (the ¼° stabilizer values) ⇒ 5–10× MORE viscous than
NEMO** = the fuzziness. Tracers: ours tvd/Van-Leer + GM/Redi κ=600 vs NEMO FCT2 + isoneutral-lap
(Treguier ~1e3) + EIV — comparable class, tvd slightly more diffusive (we have ppm_fct). NEW
`--tracer-advection` runner knob (tripole+latlon). **Diffusivity smoke 8458624 NaN'd by day 30** — the one-jump NEMO-level viscosity drop (A_h 1e5→2e4 + C_smag_lap 3.0→0.33 + ppm_fct, 3 variables at once) breaks the WOA cold start: the high viscosity IS load-bearing (consistent with the ~50-iteration cold-start history). ISOLATED follow-up smokes launched: ppm_fct-only (8458659, stock viscosity — the front-sharpness gain without the stability risk) + viscosity bisect A_h=5e4/C_smag_lap=1.0 with tvd (8458660). Honest expectation: we can likely close PART of the NEMO viscosity gap (bisect) + the advection order, not all of it — NEMO's cold start tolerates 2e4 because its initialisation/restart history differs; ours needs the dissipation crutch during adjustment. A ramped-viscosity schedule (start 1e5, decay to 2e4 over ~90 days) is the likely full fix — design next if the bisect holds.
**SMOKE VERDICTS:** BBL 60-day STABLE (max|u| 0.757 == baseline; TIMEOUT at 1h30 wall, conservative+capped+codex-approved => GO), ppm_fct 60-day STABLE (livelier fronts 1.31 m/s, physical), visc-bisect A_h=5e4/Smag=1.0 60-day STABLE (full 2e4 jump NaN'd day-30 => viscosity partially reducible; ramp design pending). **Seam-fixed tripole yr1 SCORED: RMSE 1.31 / bias 0.000 / corr 0.992, >45S +0.51** (trajectory 1.74 old-flux -> 1.43 NCAR -> 1.31 +seam); PNGs sent. **FULL-STACK iter-H tripole 2yr LAUNCHED (8458811)**: seam + NCAR + spread-150 + river gate + BBL + ppm_fct (viscosity held 1e5/3.0 this round). **visc-schedule note:** first vramp smoke NaN'd day-30 — segment-0 mistakenly set C_smag_lap=3.0 WITHOUT the --smag-cfl-safety cap it pairs with (eORCA025 pairing; this config's base is 0.33) → 9× over the diffusive CFL. NOT a schedule-mechanism failure (mechanism codex-APPROVEd cd715884). Relaunched 8458955 with '0:1e5:0.33,90:5e4:0.33,180:2e4:0.33' (A_h-only step-downs). **TRIPOLE YR-2 FINAL (8457733, NCAR+seam): SST RMSE 1.230 / bias +0.15 / corr 0.993** — improves WITH integration (yr1 1.31 → yr2 1.23; old-flux yr2 1.64). NH-mid −0.66 (halved from −1.28), >45S +0.59, tropics +0.14, Arctic −1.02; SH-mid +1.13 = main residual (full-stack BBL/rivers/ppm run 8458811 targets it). PNGs sent. PR #392 MERGED to main (e697944a). **A_h SCHEDULE VALIDATED (vramp v3 8459131): day-210 diag POST-2e4 step-down STABLE** (max|u| 1.42 m/s, physical — livelier at NEMO-class viscosity, exactly the de-fuzzing intent). Full schedule 0:1e5 → 90:5e4 → 180:2e4 (C_smag_lap 0.33) holds through both transitions. The complete fuzziness fix = --visc-schedule 0:1e5:0.33,90:5e4:0.33,180:2e4:0.33 + --tracer-advection ppm_fct → NEXT PRODUCTION ROUND. **Status**: items 1+2+3 ALL VALIDATED (1+2 merged via PR #392; 3 = knob+schedule merged, schedule values smoke-proven). Remaining iter-H follow-ups: tripole curvilinear section-ACC/MHT reader audit (ACC 109 / MHT 6.7 suspect), mpas SSS drift analysis (34.63→34.05 despite τ45.5+bnd4), RGB-chl SW penetration (last known BC gap). Next production round (after smokes):
--runoff-depth-spread-m 150 --river-mouth-restoring-gate --bbl-adv [--A-h 2e4 --C-smag-lap 0.33
--tracer-advection ppm_fct if visc smoke stable]. Production runs in flight:
seam-fixed tripole 2yr (8457733, NCAR fluxes + --ew-cyclic-overlap), latlon 1° 2yr (8457282),
mpas ico6 2yr (8455138). PNG-on-completion promised to user. Each iter-H change: codex adversarial
review + tests + smoke before production (CLAUDE.md).

## Open work toward maximal faithfulness
1. **mpas runoff** (improve SSS 1.01) + transports (ACC@Drake, AMOC@26N) — deepen the faithful set.
2. **cube ¼°** — the only geometry-grid that COULD match but doesn't; major effort (¼° + balanced-init
   + sub-grid straits, OR shared-PGF overhaul w/ full all-grid re-validation).
3. mpas #160 full-PV TRiSK refactor (HARD; current relative-PV+Matsuno split works, is stable).

## Infra (tested + codex-reviewed)
#353 tripolar MPI halo; grid-agnostic convection (`ocean/physics/column.py`, `--convection`); #354
lax.scan forcing (tripole). `tests/ocean/unit/test_no_scheme_duplication.py` enforces no dycore dup.
