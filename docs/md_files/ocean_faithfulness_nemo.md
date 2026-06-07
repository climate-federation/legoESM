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

### CODEX REVIEW of cd-grid partial-cell (f5a09943) — 3 findings, #1 FIXED this iter
1. **HIGH FIXED** `ocean_pe_cdgrid.py:183`: passed eta-STRETCHED `h_k` as `h_actual` to
   `iterate_eos_and_pressure_anomaly`, violating the J=1/eta=0 reference-pressure contract
   (#109) — reintroduced SSH into the baroclinic pressure → double-counted the barotropic
   `-g∇η`. This is itself a spurious cold-start PGF error (the class that seeds cube blowup).
   FIX: pass eta-INDEPENDENT `z_coord.h_partial` (Σ_k=H_bathy), matching the PROVEN latlon
   C-grid backend (`ocean_pe_latlon_cgrid.py:870`). z* path stays bit-exact (is_partial→None).
2. **HIGH OPEN** `:435` deta_dt sums flux_div over the unclosed wet/rock C-face → small mass
   leak across seafloor steps = the documented "strict C-face wet/rock closure" next upgrade
   (face active iff BOTH adjacent A-cells active + cross-seam is_active halo). Multi-step.
3. **MEDIUM DEFERRED** `vertical.py:756` `vertical_advection_ocean` uses `dz_half_ref*J` —
   wrong at partial bottom. SHARED routine (latlon/tripole use it too, both match NEMO well)
   → fixing risks perturbing validated grids for small bottom-cell impact; logged debt.
**Experiments running:** 8421080 = baseline cd-grid cold-start (OLD h_k code, z* vs partial);
8421082 = cd-grid partial leaf tests + cold-start WITH the #1 PGF fix (does removing the
spurious eta-PGF delay/fix the marginal-sea blowup?).

### C-D CUBE COLD-START — empirical (C32, dt30, woa-init) + ROOT-CAUSE FOUND
**Baseline (8421080):** cd-grid **z\*** NaN step ~12 (surface marginal-sea spike, Persian
Gulf lev4 65 m/s — same as deprecated FC z* step 11). cd-grid **partial cells** = smooth
physical ramp to step ~52 (max|u|~2.5 m/s) then EXPONENTIAL growth (doubling ~8 steps)
**seeding at lev 19 = the BOTTOM level** (46N/352E N.Atlantic), NaN step 140. ~12× delay vs
z*, mirrors FC-partial (~132). The bottom-level seed pointed straight at the partial-cell
bottom treatment.
**#1 eta-PGF fix (8421082): BIT-IDENTICAL cold-start to baseline** (step132 654.11 both) —
eta≈0 at cold start so h_k≈h_partial; #1 is correct for long (eta≠0) runs, NOT the gate.
**#2 C-face wet/rock closure (8421084): cold-start nearly unchanged** (step132 636 vs 654) —
the seafloor/coastline mass leak is real but SMALL; correct, not the gate.
**ROOT CAUSE (found by backend cross-check) = vertical-velocity DOUBLE-THICKNESS bug.**
`cgrid_mass_flux_divergence(h_k,...)` returns the THICKNESS-WEIGHTED `∇·(h u)` [m/s], but the
cd-grid called `diagnose_w_from_flux_div(flux_div_k, z_coord)` WITHOUT `thickness_weighted=
True` → it multiplied by `dz_ref` a SECOND time → w inflated by ~layer-thickness (10–200 m) →
huge spurious vertical advection, worst where the bottom-up cumsum is largest = the deepest
level (lev 19) = exactly the observed seed. The PROVEN latlon (`ocean_pe_latlon_cgrid.py:911`)
and mpas (`ocean_pe_mpas.py:266`) BOTH pass `thickness_weighted=True`; only the cd-grid
omitted it — which is why the cube alone was unstable while latlon/tripole match NEMO.
FIX applied at `ocean_pe_cdgrid.py:222`. (z* path also fixed; w-unit tests call the function
directly with default args → unaffected. Leaf tests 5/5.)

### Fixes shipped this iter (codex-reviewed)
- **#1** `ocean_pe_cdgrid.py` PGF h_actual = eta-independent `z_coord.h_partial` (was eta-
  stretched h_k). codex: OK (matches latlon, z* bit-exact).
- **#2** NEW `cgrid_wet_face_masks` (`operators_cdgrid.py`) + cd-grid partial wiring: strict
  wet/rock C-face closure (face wet iff both adjacent A-cells wet, halo-correct). codex: all
  HARD checks OK (slicing/halo/bit-exact/no-double-mask); 2 MED follow-ups: (a) tracer-FCT
  duogrid seam flux sync [pre-existing], (b) zeta/KE use unmasked u_d/v_d while div_v masked
  [O(coastline-err), the D-grid is already source-zeroed below seafloor].
- **w-fix** `ocean_pe_cdgrid.py:222` add `thickness_weighted=True` (THE cold-start root cause).
**COMMITTED d45234c7** (all 3 fixes; validation: leaf 5/5, diagnose_w unit 15, ocean
regression 262 pass, cd-grid differentiability pass [AD intact]; bundled-suite SIGABRT was
env OOM in test_differentiability_ocean — passes isolated 824s).
**w-fix RESULT (8421125):** partial cold-start blowup delayed **140→182 (~30%)** but NOT
solved; z* unaffected (step 14, surface horizontal spike). #1 bit-identical (eta≈0). #2 tiny
(636 vs 654). Each fix real+correct, but the **definitive remaining cube gate** is below.

### THE CUBE GATE (next): partial-cell horizontal PGF correction on the cd-grid AL corner grad
The persistent lev-19 bottom exponential mode (partial-only; z* never reaches it) = the
Adcroft/SMC03 partial-cell PGF correction that the proven latlon applies
(`ocean_pe_latlon_cgrid.py:1097-1144`, default `pgf_scheme="adcroft"`, smc03 option) but the
cd-grid LACKS — the cube's plain `_arakawa_lamb_gradient(p_prime)` differences 4 cell pressures
at the same level index k, but on partial topography those cells sit at different centroid
depths → residual PGF error → spurious bottom flow.
**VERIFIED DESIGN (Adcroft via linear-operator decomposition):** `_arakawa_lamb_gradient` is
LINEAR in its input and z_ref is constant across a corner's 4 cells, so the depth-shifted
`p_eff = p' − g·rho'·(centroid − z_ref)` corner gradient decomposes EXACTLY into existing
operator calls:
  `corr = −AL_grad(g·rho'·centroid) + z_ref_corner · AL_grad(g·rho')`
added to (dp_dx, dp_dy_perp) at `ocean_pe_cdgrid.py:273`. centroid = `compute_centroid_depth(
eta=0, H_bathy, z_coord)` (eta=0 = J=1 reference, matches latlon line 1137). z_ref_corner =
4-cell MIN of centroid. For z* (uniform centroid per level) the two terms cancel → **bit-exact**.
**OPEN IMPL DETAIL (read first):** confirm the EXACT corner→cell-centre staggering + metric of
`_arakawa_lamb_gradient` (output is (6,n,n,nlev) like KE) so `z_ref_corner` is reduced at the
SAME locations the operator differences — getting this wrong = plausible-but-wrong PGF.
**Validation gates:** rest-state machine-zero (uniform T/S → corr=0), z* bit-exact leaf,
partial-sloped nonzero, AD-finite, then C32 cold-start (does the lev-19 mode clear?) + visual.
Then smc03 option (`pgf_smc03.py` density-Jacobian kernels) if Adcroft alone insufficient.

### Adcroft PGF correction IMPLEMENTED (cd-grid AL corners) — codex-clean, but C32 still blows
Shipped: public `cgrid_corner_min` (operators_cdgrid; 4-cell corner MIN = z_ref, unit-tested
incl. decomposition-identity over all corners/seams) + inline correction at
`ocean_pe_cdgrid.py:275` (`corr = −AL_grad(g·rho'·centroid) + z_ref·AL_grad(g·rho')`, eta=0
ref). Codex review: **NO HIGH**; 2 LOW fixed (z*-bit-exact-via-gate wording; decomposition
regression test added). corner-min 3/3 pass.
**C32 cold-start RESULT (8421374): still NaN step 182** — same as w-fix-only; Adcroft slowed
growth (step-84 max|u| 24→12) but did NOT clear the lev-19 N.Atlantic (46N/352E) bottom mode.
**The mode is robust to all 4 cd-grid fixes** (w-fix delayed, C-face tiny, eta-PGF none,
Adcroft slowed) → consistent with the established **UNIFYING INSIGHT: cube cold-start is
RESOLUTION-limited** (C32 ~2.8°; WOA density fronts imply geostrophic jets the coarse grid
can't carry). Note cube physics path has NO bottom drag (scheme="none") + only KPP vertical
mixing (weak interior A_v) → an undamped bottom mode; deep momentum dissipation is a separate
candidate. **RESOLUTION SWEEP (8421377) — REFUTES the "¼°-limited" hypothesis for the corrected cd-grid.**
C32 NaN@182, C48 NaN@176, C96 NaN@200 — blowup is **resolution-INDEPENDENT** (~step 180-200
regardless), location MOVES (C32 46N N.Atlantic → C48/C96 EQUATOR 0±5°/Indonesia/S.Ocean,
levels 10-18 not just bottom), growth doubles every ~8 steps (e-fold ~350 model-s ≈ the
gravity-wave timescale at these dx). Unlike the OLD A-grid (where resolution DID delay: C32→6,
C64→9, C128→37), the corrected cd-grid blows at a fixed STEP regardless of resolution → NOT
topography/PGF/resolution. **The cube gate is a deeper undamped FAST mode** (gravity-wave /
momentum-adjustment at cold start), seeding at the equator (f→0). The cube physics path has NO
interior/bottom momentum dissipation (bottom_drag="none"; KPP momentum is surface-only) and the
smoke used forward-Euler momentum. **NEXT levers (faithful, in order):** (1) `--cube-rk3`
(SSP-RK3 momentum — the NEMO-faithful cold-start fix that SOLVED the tripole; on the OLD A-grid
RK3 was worse, but that grid had different pathology — retest on corrected cd-grid); (2) deep
momentum dissipation (vertical viscosity / bottom drag — plumb into build_cubed_sphere);
(3) barotropic-baroclinic split coupling under fv3sw at cold start. Validation 8421407 (leaf
z*-bit-exact reformulated + decomposition test) gates the PGF-correction commit.

### The 4 cd-grid fixes this session (all real, codex-clean, COMMITTED)
w double-thickness (d45234c7) · C-face wet/rock closure (d45234c7) · partial PGF ref-thickness
(d45234c7) · Adcroft partial-cell PGF corner correction (0bdfc496, 9/9 tests). These make the
cd-grid ocean substantially more correct; the cold-start gate is the fast-mode dissipation, not
these.

### CUBE COLD-START — lever results (corrected cd-grid stack, C32, dt30, woa, partial)
- forward-Euler: NaN step 182 · RK3 (8421408): NaN step **210** (delayed ~15%, NOT solved →
  not an integration-order problem) · resolution C48/C96 (8421377): **resolution-independent**
  (~180-200) · Adcroft PGF (8421374): slowed growth, NaN 182. **All levers fail** → undamped
  GRID-SCALE fast mode (doubling ~8 steps ≈ gravity-wave timescale), equator-seeding.
- Hypotheses: (i) barotropic fast external-gravity-wave mode under fv3sw split (harmonic
  baroclinic A_h can't damp the barotropic); (ii) missing scale-selective dissipation
  (biharmonic/Smagorinsky) — the cube external-physics path applies ONLY a fixed CFL-capped
  harmonic A_h (~1.6e8 at C32), bypassing the dycore biharmonic/smag that stabilises the proven
  tripole/latlon (`--C-smag-lap 3.0`). NOTE: estimated harmonic damping timescale at grid scale
  ~60s < growth ~350s, so harmonic SHOULD damp a baroclinic grid mode → suggests the mode is
  BAROTROPIC (not reached by baroclinic A_h).
- **Mechanism discriminator (8421411): DECISIVE.** ARM A rest-state IC (uniform T/S, NO fronts)
  → blows within ~30 steps (FASTER than WOA's 182) ⇒ the fast mode is **INTRINSIC** (barotropic/
  numerical over topography+wind forcing), NOT the WOA-front baroclinic adjustment (rules out
  PGF/front cause; caveat: confirm rest_state_ocean partial-cell setup isn't itself the artifact).
  ARM B WOA + **velocity-cap 3 m/s → STABLE to step 308+** (max|u| PEGGED at 3.0, SST/SSS evolve)
  = first cube cold-start that completes. Band-aid only (clips momentum at the unstable cells →
  NOT faithful), but proves the integrator runs once the spike is bounded.
- **CONCLUSION:** cube cold-start = an intrinsic undamped GRID-SCALE fast mode; faithful fix =
  proper SCALE-SELECTIVE dissipation, not a clip. `LateralMixingConfig` supports harmonic/
  **biharmonic** (B_h_momentum, CFL-capped)/gm_redi; the cube external-physics path uses harmonic
  ONLY. **NEXT LEVER (faithful, the tripole smag-cap analogue): enable biharmonic momentum on the
  cd-grid** in `build_cubed_sphere` (verify `biharmonic_lateral_mixing` supports the cdgrid grid;
  add a leaf test) + retest cold-start. If the mode survives biharmonic too → it is BAROTROPIC
  (fv3sw split): target the barotropic div-damp/time-filter / the wind-stress→cd-grid-momentum
  projection. Velocity-cap remains an opt-in fallback to obtain a (non-faithful) completable run.

### CUBE COLD-START iter (2026-06-07) — ROADMAP CORRECTED + seed-isolation job
Code-map of the cd-grid dissipation surface (5-agent workflow) overturns two doc assumptions:
- **Biharmonic momentum is ALREADY wired** on the cd-grid — `--cube-hyperdiff B` →
  `config.hyperdiff_coeff` → `hyperdiffusion_3d` (`ocean_pe_cdgrid.py:494`), applied to the
  FULL 3D cell-centre velocity (barotropic component INCLUDED), NOT gated by physics_fn
  (the "if A_h>0 or hyperdiff_coeff>0" branch at :459 always runs). The "next lever = enable
  biharmonic" was already implemented; only the OLD FC A-grid tested it (bit-identical). NOT
  yet tested on the corrected cd-grid. No CFL cap on this path → must pick B below the explicit
  ∇⁴ CFL (~6e18 at C32/dt30) by hand.
- **fv3sw barotropic ALREADY has full FV3 dissipation** — div-damp (`barotropic_sw_div_damp_factor=120`,
  ocean-tuned vs atmos 8×), SSP-RK3 time integrator, adaptive Smagorinsky (`dddmp`), vorticity
  damp (`barotropic_sw_damp_v=0.030`); `barotropic_substeps_fv3sw` (`barotropic_cgrid.py:314`) →
  FV3 SW core (`shallow_water_fv3_cdgrid.py`). So "barotropic missing div-damp" is FALSE; the
  barotropic solver is well-conditioned. (`div_damp_2/4` OceanConfig fields are NOT consumed on
  the cube — dead knobs there; only latlon/mpas barotropic use `barotropic_div_damp`.)
- **Wind-stress→cube projection = HIGH checkerboard risk** (`external.py:94-127`,
  `omip2_applicator.py:363`): CORE-II tau sampled NEAREST-NEIGHBOUR to cube cell centres, applied
  AT centres with NO spatial smoothing, top-layer only; the cd-grid then projects this cell-centre
  forcing to D-grid corners. ARM A (rest+partial-topo+wind) blew <step 30 — FAR faster than WOA's
  182 → wind injection is the prime intrinsic-seed suspect.
**Job 8421758 — DECISIVE.** Seed isolation (`--forcing-ramp-days 100000`⇒forcing≈0):
| ARM | setup | result |
|---|---|---|
| 1 | rest + FLAT + no-forcing | **STABLE** — max\|u\| 0.005→0.022 m/s over 700 steps (clean) |
| 2 | rest + FLAT + wind | **STABLE** — max\|u\|~0.02 m/s bounded |
| 3 | rest + partial-TOPO + no-forcing | **BLOWS** NaN~182, seed lev19(bottom) 46N/352E→lev14 Gibraltar |
| 4/5 | WOA + biharmonic 1e18/1e17 | **BLOW** ~145 (same seed) |
**CONCLUSION (locks the cube gate):** the seed is **TOPOGRAPHY at rest**, NOT wind (flat+wind
stable), NOT pure numerics (flat stable), NOT baroclinic fronts (ARM3 has no WOA). Biharmonic
momentum does NOT damp it (can't fix a PGF error). `rest_state_ocean` = horizontally-UNIFORM but
vertically-STRATIFIED column (exp T 2→20°C) → exact horizontal PGF MUST be 0 (∇_h ρ=0 at constant
z); ARM3 blows ⇒ the **cd-grid partial-cell horizontal PGF error** over real bathy is the gate,
bottom-intensified, ≫ tripole's smc03 rest-test (1e-6 m/s²). The section-10b Adcroft-Campin
correction is the LINEAR depth shift → leaves a 2nd-order residual for stratified columns. KEY:
**both proven grids use `--pgf-scheme smc03`, NOT adcroft** (line 60); the cd-grid ONLY has
Adcroft. **FIX = port the smc03 (Shchepetkin-McWilliams 2003 density-Jacobian) PGF to the cd-grid
AL corners** (reuse grid-neutral `pgf_smc03.py`; latlon ref `ocean_pe_latlon_cgrid.py:1097-1144`).
This RETIRES the "intrinsic fast mode / needs full stack" framing — the cube gate is a single,
classic partial-cell PGF problem. Validation gate = stratified-rest step-1 PGF→~0 (ARM3 becomes
stable). All prior cube levers (RK3, resolution, viscosity, biharmonic, velocity-cap, wind) were
chasing a PGF error.

**smc03 PGF PORTED to the cd-grid (this iter) — IMPLEMENTED, validation pending (job 8421811).**
`ocean_pe_cdgrid.py` §10b now branches on `config.pgf_scheme`: `"smc03"` REPLACES the corner PGF
with the S&M03 density-Jacobian; `"adcroft"` (default) keeps the linear correction. The smc03
in-cell-k pressure reconstruction `P_k(z)=P_top+g(z−z_top)[ρ'+0.5σ(z+z_top−2z_c)]` is QUADRATIC in
z; in the centroid anomaly ẑ=z−cref it is `a+bẑ+cẑ²`, so the AL-corner gradient decomposes EXACTLY
(AL_grad linear, z_ref=`cgrid_corner_min(cent_anom)` corner-constant) into
`AL_grad(a)+z_ref·AL_grad(b)+z_ref²·AL_grad(c)` — the proven Adcroft linear decomposition extended
by one order (reuses `cgrid_corner_min`+batched `_arakawa_lamb_gradient`+`reconstruct_harmonic_slopes`).
`a=P_top+g(cref−z_top)(ρ'+0.5σ(cref+z_top−2z_c))`, `b=g(ρ'+σ(cref−z_c))`, `c=0.5gσ`. Rest cancels:
horizontally-uniform stratification → a,b uniform among active cells → AL_grad=0. Gated `is_partial`
(z* bit-exact). New leaf tests: linear-ρ rest PGF→~0 ≫ adcroft residual (canonical), z* gated-off,
AD-finite. CLI `--cube-pgf-scheme smc03`; config default "adcroft" (preserves flat-bit-exact test).

**Codex round-1 (8421811): algebra/decomposition/units/sign/AD/gating ALL confirmed correct.**
2 findings fixed: (HIGH) `cgrid_corner_min` included below-seafloor cells → z_ref could be pinned by
a rock cell → made z_ref **wet-aware** (bounded sentinel `max|cent_anom|+1`; corner-min over active
only). (LOW) unknown `pgf_scheme` → `raise ValueError`. **AD fixed** (8421864 passes
`test_smc03_partial_path_differentiable`). **smc03 cancels the partial-bottom-centroid PGF** — new
ALL-WET-partial canonical test (`..._rest_pgf_vanishes`) machine-zero, ≫ adcroft residual.
**EMPIRICAL GATE (8421864): smc03 alone is NOT enough.** rest+topo+no-forcing still NaN @182 (seed
back at **lev19 bottom 46N/352.6E N.Atlantic** = steep topo = seafloor STEPS), and
`test_smc03_beats_adcroft_on_stepped_bathy` FAILED (smc03 4.40e-7 > adcroft 2.51e-7) — the real gate
is the **seafloor-STEP PGF** (active/rock corners), not the partial-bottom centroid. The earlier
"wet/rock closure added" claim was DOC-ahead-of-code: the code only filled the 2-D coastline mask.

**WET/ROCK CLOSURE NOW IMPLEMENTED (this iter, validation 8421971).** New module helper
`_fill_inactive_per_level(field, wet_3d, grid)` = per-level `jax.vmap` of the 2-D `fill_land_cells`
over the level axis with the 3-D wet mask `wet_cc_3d`(=`mask·active_3d`, reused from the §5 C-face
closure): fills EVERY inactive cell (coastline AND below-seafloor rock) at each level from active
same-level neighbours, so the smc03 a,b,c the AL corner gradient differences carry no wet/rock step
jump (cd-grid analogue of the latlon wet/rock FACE mask). Coefficients filled (NOT a large sentinel —
that overflowed via z_ref²). AD-safe (`fill_land_cells` safe-divides via `maximum(count,1)`). z*
untouched (whole branch gated `is_partial`). **Codex round-2 (this iter): CLEAN** except 1 MEDIUM —
the fill pads via `grid.halo_interp_offsets` while the AL gradient pads via `_pad_halo_auto(cdgrid)`
(prefers `cdgrid.base.duogrid`). MOOT for the current cube path: `base.duogrid is None` (the T/S
duogrid-halo upgrade is tracked future work, run_omip:456) so both fall back to the SAME
`halo_interp_offsets`; AND it is the EXISTING cd-grid convention (base `p_prime` :251 + adcroft both
`fill_land_cells(...,grid)`). FOLLOW-UP: when the T/S-duogrid upgrade lands, make the fill
duogrid-aware too. PERF debt: per-level fill issues nlev halos (single-GPU fine, MPI 4-D follow-up).

**RESULT (8421971): closure cuts the rest PGF ~30× — seafloor-step PGF SOLVED globally; gate
localizes to the MEDITERRANEAN.** Leaf **9/9 pass** (incl. `test_smc03_beats_adcroft_on_stepped_bathy`
now smc03<adcroft). rest+topo+no-forcing: step-14 max|u| 0.12→**0.0095**, step-84 13.5→**0.37 m/s**
(broad bottom N.Atlantic mode GONE) — but a residual ignites ~step 140 at **lev14 mid-depth
40.7N/4.2E = NW Med** → NaN ~220. WOA: same, seed lev11 46.2N/7.4E (Ligurian/Med) → NaN ~210.
**Both rest & WOA now seed at the single under-resolved Med basin** (C32 ~2.8°, ~2-3 cells, Gibraltar
sill + deep basin walls), not the global bottom. The closure resolved the GLOBAL seafloor-step PGF;
the last holdout is the steepest marginal sea — the classic sub-grid Med problem (extensively
documented in the FC-A-grid history below). **NEXT LEVER (faithful): lateral dissipation.** The cube
external-physics path runs ONLY a CFL-capped harmonic A_h=1e9 (`--cube-Ah`); it LACKS the
scale-selective biharmonic/Smagorinsky that stabilises the proven tripole/latlon (`--C-smag-lap 3.0`).
Biharmonic is wired on the cd-grid (`--cube-hyperdiff`→`hyperdiffusion_3d`, full 3-D velocity, no CFL
cap → pick B<∇⁴CFL~6e18 at C32/dt30 by hand). Earlier biharmonic (8421758 arms 4/5: 1e18/1e17) blew
~145 — but that was with the 30×-LARGER PGF error; retest on the now-much-smaller residual. Sweep
job: rest+topo+no-forcing+smc03(closure)+`--cube-hyperdiff` {1e17,1e18,5e18} (+RK3) → does any keep
the Med mode bounded past ~300?

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
