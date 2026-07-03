# Cubed-sphere C-grid ocean — implementation plan

Status: planning, post-review v3 — APPROVED WITH MINOR CHANGES (2026-04-29)
Owner: Dhruv
Reviewers: ocean-model-expert, dycore-expert (2026-04-29, two passes)
Related: issue #214 (ocean grid-agnostic refactoring, Phase 1+2 landed),
`cubed_sphere_edge_artifacts.md`, `ocean_grid_staggering.md`.

## 1. Goal

Build a pure Arakawa C-grid ocean dynamical core on the cubed-sphere grid,
replacing the current C-D grid cubed-sphere ocean (`ocean_pe_cdgrid.py`)
for production work. Follow MITgcm LLC precedent on staggering, compact
PGF, and HK management — but adopt MITgcm's *actual* numerical choices
(vertex-PV Coriolis, biharmonic-Leith viscosity, Shchepetkin-McWilliams
PGF on steep slopes), not idealized substitutes.

The existing C-D grid ocean stays in-tree as a fallback during bring-up.

## 2. Why a separate effort from #214

Issue #214 refactors duplicated scheme logic into grid-agnostic helpers
(`ocean_tendency_common.py`, `barotropic_common.py`). Phase 1+2 landed
in commit `7c51d2d`. Phase 3 (callable-operator API for lateral mixing)
is still open.

This plan is a *new dycore on a new grid staggering*, not a refactor.
It benefits from #214 Phase 1+2 (helpers exist; we plug into them) and
shares Phase 3's GM/Redi blocker, but is otherwise independent.

## 3. Why C-grid, not C-D grid

The current C-D grid cubed-sphere ocean has three known structural
issues (see `cubed_sphere_edge_artifacts.md` and
`project_cubesphere_ocean_instability.md`):

1. Float64 required for the baroclinic PGF — the Arakawa-Lamb 4-point
   stencil amplifies halo-interpolation error at face boundaries.
2. Geostrophic adjustment blows up at face boundaries by day 5–7
   (e-folding ≈ 0.8 days).
3. The A-grid barotropic gradient is inconsistent with the C-D grid
   baroclinic gradient at face boundaries.

No production ocean model uses a C-D grid on a cubed sphere. MITgcm's
LLC mesh proves a pure C-grid cubed-sphere ocean works at scale (ECCO,
LLC4320). We adopt MITgcm's specific choices wherever applicable.

## 4. What already exists (do not duplicate)

Audit done 2026-04-29 against `main` at `1a777a5e`.

### 4.1 Cubed-sphere C-grid scalar operators — present
`src/legoesm/core/operators_cdgrid.py` provides cell-face stencils:
`cgrid_divergence`, `cgrid_gradient_2d`, `cgrid_mass_flux_divergence`,
`cgrid_scalar_advection`, `cgrid_tracer_advection_fct`, plus C↔D bridges
(`fv3_d2cc`, `fv3_cc2c`, `dgrid_to_cgrid`, `cgrid_to_dgrid`).

**Caveats discovered in review** (must be addressed in Phase 1; see §6):
- `cgrid_divergence` assumes 4 edges per cell — invalid at the 8 cube
  corners where only 3 edges meet.
- `fv3_cc2c` applies the non-orthogonality (`cosa`) correction to the
  u-branch but not the v-branch (asymmetric, see §6.5).
- `_cgrid_fct_fluxes_2d` is dimension-by-dimension Zalesak; monotonicity
  across face seams is unverified for ocean tracers with sharp fronts.

### 4.2 Cubed-sphere C-grid barotropic solver — present, partial coverage
`ocean/dynamics/barotropic_cgrid.py` (299 LOC) is a forward-backward
C-grid solver on `(u_c : (6, n+1, n), v_c : (6, n, n+1))`. It uses
`fv3_cc2c` for Coriolis projection and is reachable today via
`OceanConfig.barotropic_staggering = "c_grid"` (`ocean/state.py:127`).

**Important**: this exercises the seam at the **gravity-wave** level only
(halo-1 stencils, no baroclinic feedback). It is *not* evidence that the
seam survives baroclinic dynamics. The plan's Phase 0 must explicitly
exercise baroclinic feedback (see §7 Phase 0).

### 4.3 Lat-lon C-grid PE — structural blueprint
`ocean_pe_latlon_cgrid.py` (1400 LOC) is the structural template. The new
file mirrors its sequencing, but operators, halos, and metric calls
differ.

### 4.4 Refactored helpers (#214 Phase 1+2) — present
`ocean_tendency_common.py` (`iterate_eos_and_pressure_anomaly`,
`apply_sponge_tracer_relaxation`, `apply_freshwater_virtual_salt_top`,
`implicit_bottom_drag_factor`) and `barotropic_common.py`
(`compute_filter_weights`, `bebt_blend`, `maxvel_clip`). Enforced by
`tests/ocean/unit/test_no_scheme_duplication.py`.

### 4.5 C-D grid CS PE — fallback only
`ocean_pe_cdgrid.py` (401 LOC). Stays during bring-up; flagged for
removal once C-grid CS reaches scientific parity (§8 acceptance gate).

## 5. Gaps to close

1. No `CubedSphereCGrid` metric struct — decision §6.1: reuse
   `CubedSphereCDGrid` with a Phase 1 metric-continuity test.
2. **No corner-cell topology design** — must be specified, not deferred
   (§6.2).
3. **No tracer-thickness consistency design** — must be specified before
   any tracer code lands (§6.3).
4. **No edge-parity velocity halo** — needed for tracer FCT and
   biharmonic (§6.4).
5. **No vertex-PV / Sadourny Coriolis** — §6.5.
6. **No topographic-PGF strategy** — §6.7.
7. No CS C-grid initialization path (§7 Phase 1).
8. No CS C-grid 3D PE (§7 Phase 2).
9. No driver / `ocean_model.py` dispatch (§7 Phase 3).
10. No GM/Redi triad operator on cubed sphere — shared with MPAS,
    deferred to a later effort (§7 Phase 5).

## 6. Locked design decisions

These are decisions to make once and not relitigate during
implementation. Each was raised by the post-review and is binding.

### 6.1 State layout and metric struct
- `T, S, eta` on cell centres `(nlev, 6, n, n)` (eta has no `nlev`).
- `u` on west/east edges `(nlev, 6, n+1, n)`, `v` on south/north edges
  `(nlev, 6, n, n+1)`. Edge masks shaped to match.
- PyTree leaf names match the lat-lon C-grid `OceanState` so loss
  functions, `ml/channel_packing.py`, and ML I/O carry over unchanged.
- Reuse `CubedSphereCDGrid` (`grids/cubed_sphere_cdgrid.py`) as the
  metric struct. **Phase 1 gate**: a unit test asserting bit-precision
  continuity of `dx_edge_x`, `dy_edge_y`, and edge areas across all 12
  cube edges after halo rotation. Without this test, `cgrid_divergence`
  may be silently lossy at seams.

### 6.2 Corner-cell topology (BLOCKER from review)
At the 8 cube vertices three faces meet → corner cells have 3 edges,
not 4. `cgrid_divergence` and `cgrid_gradient_2d` currently assume 4.

**Decision**: mask the 8 corner cells as inactive (land) for the entire
PE. Set their cell-centre tracers to a sentinel, set the three adjacent
edges to `u_mask = v_mask = 0`. Mass flux through corner cells is then
identically zero by construction.

**Stencil consumption rule**: the divergence and tracer-flux operators
must consume `(u_mask * u, v_mask * v)` (i.e., the masked velocity), not
bare `(u, v)`. This makes corner-adjacent cells see a closed wall on the
masked side and is the mechanism by which corner cells stay exactly
zero. State explicitly in `cgrid_divergence` and the mass-flux
divergence wrapper.

**Corner-halo sentinel**: halo cells whose physical position falls on a
cube vertex have ambiguous edge-parity (three faces meet, no single
"neighbour" face). Fill these halo cells with a zero sentinel
consistent with the corner-cell mask; the edge-parity halo of §6.4 is
short-circuited there.

**Rationale**: an active corner-cell treatment would require a
non-Cartesian divergence stencil (3 edges weighted by their respective
edge lengths) and matching Coriolis at a 3-way junction — neither is
worth the implementation cost at the resolutions we plan to run
(n ≤ 192). MITgcm LLC sidesteps this with the polar cap; we sidestep
it by masking.

**Phase 0 gates** (corner-cell):
1. Stationary-tracer-blob test centred on a corner cell, run for 30
   days. Tracer drift ≤ round-off.
2. Masked corner cells must remain *exactly* zero in `η, T, S, h*T`
   over 30 days under arbitrary forcing. This catches accidental
   leakage through the divergence stencil if the `mask*flux` rule is
   violated.

### 6.3 Tracer-thickness consistency (BLOCKER from review)
The barotropic substep updates `eta` (and therefore layer thickness `h`
in z*) on a fast timescale; tracer advection runs on the slow
baroclinic timestep with `(u, v)`. Without explicit reconciliation,
`d(h*T)/dt = -∇·(h*T*u)` uses stale `h` and conservation breaks under
any freshwater forcing or non-trivial η-tendency.

**Decision** (Griffies et al. 2001, Adcroft et al. 2019 MOM6 pattern):
- During the barotropic subcycle, **accumulate** `U_bar*H` and `V_bar*H`
  integrated over all substeps. Carry the time-integrated edge mass
  flux `(M_x, M_y)` in `SegmentCarry`.
- Build the baroclinic edge mass flux as `M_x_bc = M_x/Δt_bc -
  ⟨u'⟩H_u` (and similarly for y), where the bracketed term is the
  baroclinic perturbation flux.
- Use `(M_x_bc, M_y_bc)` for **both** the layer-thickness update *and*
  the tracer advection — they share the same flux. Tracer-thickness
  consistency holds by construction.
- No additive uniform fixers. Do not paper over conservation drift; it
  indicates a bug in the flux reconciliation.

**Phase 1 cross-cutting**: `SegmentCarry` must gain `M_x_accum`,
`M_y_accum` fields, zeroed at the start of each baroclinic step. All
constructors enumerated in CLAUDE.md "SegmentCarry discipline" must be
updated in the same PR.

**Phase 4 diagnostic**: RPE drift (Ilicak 2012), not just Σ h·T.
Resting passive-tracer drift ≤ round-off; RPE drift comparable to
lat-lon C-grid baseline at matched resolution.

### 6.4 Edge-parity velocity halo (MAJOR from review)
At a cube seam, an edge that is u-aligned on face A is v-aligned on
the neighbouring face after the 90° rotation. PPM tracer FCT
(`cgrid_mass_flux_divergence`) and biharmonic viscosity both need
halo-2 velocity values across seams. The current `barotropic_cgrid`
only requires halo-1, so this issue is unexposed today.

**Decision**: implement `pad_halo_cgrid_velocity(u_c, v_c, cdgrid)` in
`grids/halo.py` (or alongside `fv3_cc2c` in `operators_cdgrid.py`) that:
- Fills u-edge halo across an x-aligned seam from the neighbour's u-edge
  values.
- Fills u-edge halo across a y-aligned seam from the neighbour's v-edge
  values, with the appropriate sign flip and non-orthogonality
  correction.
- Symmetrically for v-edge halo.

**Phase 1 gate**: solid-body rotation through a face seam. Discrete
divergence of the rotated velocity must be ≤ round-off everywhere
including the seam. Without this test, biharmonic and FCT will both
silently corrupt the seam.

### 6.5 Coriolis force (MAJOR from review — design change)
`fv3_cc2c` projection of cell-centre Coriolis to edges (a) is
asymmetric between u- and v-branches (cosa correction on u only) and
(b) conserves neither energy nor enstrophy. MITgcm LLC uses a
Sadourny-style C-grid Coriolis with vorticity at vertices.

**Decision**: implement vertex-PV Coriolis in the Arakawa-Lamb 1981
**EEN** (energy-and-enstrophy) form, matching MITgcm LLC production
practice (the prior v2 specified pure Sadourny EN; the ocean reviewer
flagged that LLC actually runs EEN — Arakawa-Lamb 1981 9-point
q-stencil — which conserves both KE and potential enstrophy in the
non-divergent inviscid limit, modulo a small dispersive error).
- Compute relative vorticity `ζ` on cell **vertices** (corners) from
  edge `u, v` via the standard 4-point curl.
- Form `q = (f + ζ) / h_vertex` on vertices, where `h_vertex` is
  4-point average of cell-centre `h`.
- Edge Coriolis flux uses the AL81 9-point q-stencil rather than a
  single nearest-vertex value; this is what gives EEN its joint
  conservation property.
- Document explicitly: EEN conserves global KE *and* potential
  enstrophy in the non-divergent inviscid limit. The 9-point stencil
  costs more halo than EN but is the production standard.

**Corner-vertex mitigation**: at each of the 8 cube vertices the
surrounding cells are masked (§6.2), so `h_vertex = 0` and `q = (f+ζ)/0`
is formally undefined. Set `q := 0` at the 8 cube-vertex points (or
equivalently, zero the Coriolis flux on the 12 edges immediately
adjacent to a cube vertex via the edge mask). This drops a single
vertex contribution per corner from the AL81 9-point stencil; the
truncation error is local and bounded.

**Phase 0 gate**: discrete energy budget over 30 days at constant ρ.
KE drift ≤ explicit dissipation contribution (i.e., no extra energy
created or destroyed by the Coriolis discretization).

**Why not TRiSK** (Ringler et al. 2010): TRiSK is the natural choice
on Voronoi meshes (already used by MPAS in this codebase); on
quadrilateral C-grids the AL81 EEN form is the production standard
(MITgcm, MOM6, NEMO). Stay with the production precedent.

### 6.6 Pressure-gradient force, flat / smooth bottom
Compact 2-cell stencil on hydrostatic pressure anomaly built by
`iterate_eos_and_pressure_anomaly`:
```
PGF_x[i, j] = -(p[i, j] - p[i-1, j]) / dx_c[i, j]
PGF_y[i, j] = -(p[i, j] - p[i, j-1]) / dy_c[i, j]
```
applied via `cgrid_gradient_2d`. No Arakawa-Lamb 4-point stencil.

### 6.7 Pressure-gradient force, steep topography (MAJOR from review)
The compact 2-cell PGF in z* over realistic bathymetry exhibits the
classic Shchepetkin-McWilliams 2003 cancellation error: the horizontal
pressure gradient at depth z is the difference of two large numbers
(integrated weight of overlying water column + the depth-varying
density), and the discretization must make those cancel exactly in a
resting stratified ocean.

**Decision**:
- For Phase 0–4 (idealized bathymetry, smooth or flat), use the §6.6
  compact stencil.
- Add a **rest-state acceptance test** with realistic bathymetry
  (synthetic seamount / continental slope) before declaring scientific
  parity in §8: `max|u|` after 30 days of unforced spin-up must be
  < 1e-6 m/s. If it is not, ship Shchepetkin-McWilliams density
  Jacobian PGF (separate PR; see §10).
- Document explicitly that the Phase 0–4 plan is flat / smooth-bottom
  only and that production runs over realistic topography are gated on
  the SM-PGF follow-on.

### 6.8 Tracer advection
- Production: `cgrid_tracer_advection_fct` with cross-seam monotonicity
  unit test (Phase 1 gate; see §6.4).
- No upwind fallback path. Upwind on a CS C-grid produces cube-imprint
  diffusion that masks the very seam artifacts we test for.
- Cross-term Lin-Rood requirement (Putman & Lin 2007) for 2D CS tracer
  transport: verify the existing FCT either honours it or document the
  truncation error.

### 6.9 Lateral viscosity (MAJOR from review — design change)
"Plain biharmonic" is rejected. Adopt biharmonic-Leith
(Fox-Kemper & Menemenlis 2008) — the same family MITgcm LLC uses in
production:
```
A_4 = (C_L * dx)^5 * |∇ζ|
```
with `C_L ≈ 1.0–2.0`.

**Implementation note**: vector biharmonic on a manifold must be
written in vector-invariant form `∇(∇·u) - ∇×(∇×u)`, with halo
exchange on the intermediate `∇²u` (which lives on edges). The
intermediate halo is the edge-parity halo of §6.4 — same machinery,
applied twice.

**Corner-sponge rule**: the intermediate `∇²u` near corner-adjacent
edges sees a sharp 0-step where a corner cell is masked. Applying the
second Laplacian across that step amplifies the discontinuity. Mitigate
by zeroing the biharmonic *tendency* on the 1-cell ring of edges
immediately adjacent to a corner cell (a localised 1-edge sponge). This
is documented in the Munk-layer-on-seam Phase 4 gate so the boundary
layer is not measured across a corner.

**Phase 4 gate**: Munk-layer test with the western boundary positioned
on a face seam, *not* on a face corner. Boundary-layer width must agree
with the analytic `(A_4 / β)^(1/5)` to within 10%.

### 6.10 Time integration
**Decision**: AB3 (Adams-Bashforth third-order) for baroclinic
momentum and tracers, **with trapezoidal-implicit Coriolis treatment**
(MOM6 pattern). No leapfrog / Robert-Asselin.

**Rationale**:
- AB3 matches the lat-lon C-grid PE; sequencing transfers, including
  the AB3↔barotropic subcycle interface (slow-forcing evaluated at
  AB3-extrapolated time levels consistent with the barotropic
  predictor). Reuse that pattern; do not invent a new one.
- Forward-backward (used by the barotropic) does not generalize to
  baroclinic with advection.
- Leapfrog + RA + Coriolis compounds dissipation; AB3 is cleaner.
- RK3-SSP is a credible alternative (no startup, no tendency-history
  carry, pairs cleanly with `jax.checkpoint`) but is rejected for this
  PR to keep the time-stepping interface identical to the lat-lon
  C-grid PE. Reconsider if Phase 4 KE-drift gates fail.

**Coriolis treatment under AB3 (mandatory)**: pure AB3 is
unconditionally unstable for pure imaginary eigenvalues — the inertial
oscillation `du/dt = -fv, dv/dt = +fu` grows under pure-explicit AB3
regardless of timestep. Production ocean models that use AB3 for
momentum apply a **two-step trapezoidal correction for Coriolis**: the
non-Coriolis terms use AB3 extrapolation, then Coriolis is applied
implicitly via the 2×2 rotation-matrix solve (energy-conserving). This
is mandatory; leaving Coriolis explicit fails Phase 0 KE-budget gate by
construction. The lat-lon C-grid PE already implements this pattern;
reuse the helper.

**CFL**: include the cubed-sphere min-edge length (≈ 0.7× equator
value at corners) in the timestep selection. Document in the user-facing
config.

### 6.11 Momentum form
Vector-invariant momentum equations (KE gradient + Coriolis-vorticity
flux), not flux form. Required because the seam halo for momentum
components carries the rotation (sign flip and non-orthogonality
correction), and vector-invariant naturally separates the rotational
(Coriolis + ζ) from the kinetic-energy gradient.

### 6.12 Edge bathymetry
`H_u, H_v` on edges from cell-centre `H_bathy` via simple averaging
across the seam, **with the edge masked to land where either flanking
cell is land**. Adequate for Phase 0–4. Steep-bathymetry refinement
(weighted partial-cell averaging, Adcroft et al. 1997) gates on the
SM-PGF work (§6.7).

### 6.13 Precision
Float64 throughout for momentum, tracers, and PGF. Float32 is **not**
a deliverable; the float32 vs float64 comparison is dropped from the
acceptance gate. Atmosphere CS shallow-water tests already required
float64 for cross-face diagnostics; the same holds here.

### 6.14 Filter / damping
Biharmonic-Leith only (§6.9). No divergence damping (would amplify
face-boundary gradient errors per CLAUDE.md "Diffusion coefficient
sensitivity").

## 7. Phased work

Estimates assume one engineer working day-by-day; multiply by 1.5–2×
if interleaving with other work. Estimates revised upward post-review.

### Phase 0 — Decisions and falsifying POC (3–5 days)
- §6 decisions are locked. Revisit only if Phase 1 reveals a blocker.
- **Falsifying POC** (replaces the original constant-ρ POC): a 2-layer
  stratified geostrophic-adjustment run with EOS + compact PGF +
  vertex-PV Coriolis active, n=48, 30 days, single rank, float64.
  Acceptance gates:
  1. No NaNs.
  2. Face-boundary η growth: σ(η at seam) / σ(η interior) drift over
     30 days < 5%.
  3. KE drift bounded by explicit viscous dissipation only (no spurious
     production / destruction).
  4. Stationary tracer blob centred on a corner cell drifts ≤ round-off
     over 30 days (§6.2 gate).
  5. Discrete divergence of solid-body rotation through a seam ≤
     round-off (§6.4 gate).

If any gate fails, root-cause before writing Phase 1 code. n=24 is too
coarse to expose seam errors at production-relevant scales (Rossby
radius unresolved); the POC must run at n=48 minimum.

### Phase 1 — Metric tests, edge-parity halo, init (2–3 days)
- Lift `_derive_edge_masks` from `barotropic_cgrid.py:44` to a shared
  helper (e.g., `ocean/dynamics/edge_mask.py`).
- New `pad_halo_cgrid_velocity` (§6.4) with the seam-parity test as the
  acceptance gate.
- Metric continuity unit test (§6.1) for `CubedSphereCDGrid`.
- New `init_cubed_sphere_cgrid.py`: rest-state builder, edge masks,
  edge bathymetry, atomic `replace_land_mask`, corner-cell masking
  (§6.2). Mirror the lat-lon C-grid `_assert_runtime_invariants`.
- Cross-cutting `SegmentCarry` extension: add `M_x_accum`, `M_y_accum`
  (§6.3) and the new edge-staggered velocities. Update **every**
  `SegmentCarry` constructor — `pack_carry`, `unpack_carry` docstring,
  `test_compiled_segments.py`, `test_scale_tpu_compat.py`,
  `test_scale_jit_health.py`, validation tests. One dedicated PR.

### Phase 2 — Baroclinic 3D PE (5–7 days, revised up)
- New `ocean/dynamics/ocean_pe_cubedsphere_cgrid.py`, structurally
  mirroring `ocean_pe_latlon_cgrid.py`.
- Vertex-PV Coriolis (§6.5).
- Vector-invariant momentum (§6.11).
- Compact PGF (§6.6); bathymetry partial-cell deferred (§6.7).
- AB3 time integrator (§6.10).
- EOS / hydrostatic via `iterate_eos_and_pressure_anomaly`.
- Tracer-thickness-consistent FCT advection (§6.3, §6.8) using
  `M_x_accum`, `M_y_accum` from `SegmentCarry`.
- Biharmonic-Leith viscosity (§6.9) using edge-parity halo.
- Sponge / freshwater / bottom drag through `ocean_tendency_common.py`.
- AD: `build_segment_fn(...).raw` for training (no buffer donation);
  `jax.checkpoint` on the barotropic substep scan body
  (n_baro ≈ 60–120 substeps would otherwise OOM under
  `eqx.filter_value_and_grad`).

### Phase 3 — Driver wiring + smoke test (1–2 days)
- `ocean_model.py` dispatch on `(grid_type=cubed_sphere,
  staggering=c_grid)` → new `ocean_model_cubedsphere_cgrid.py`.
- `OceanConfig` literal, factory case in `integration.py`,
  `supported_matrix.py` entry, public-config dispatch test
  (CLAUDE.md "Code Hygiene Rules").
- Smoke test: 1-day spinup at n=24 (smoke only — coarser grid is
  fine here because seam tests already passed in Phase 0).

### Phase 4 — Validation (5–7 days, revised up)
Acceptance gate before declaring scientific parity with the lat-lon
C-grid.

| Test | Pass criterion |
|---|---|
| Geostrophic adjustment, n=48, 30 days | No face-boundary growth; eta error norm flat |
| Lock exchange (Ilicak 2012), 2D vertical | RPE drift comparable to lat-lon C-grid baseline |
| Stommel/Munk gyre, western boundary on face seam | Boundary-layer width within 10% of analytic `(A_4/β)^(1/5)` |
| DOME overflow (z* + idealized topography) | Plume descent rate within 20% of community baseline |
| Seamount rest-state, realistic bathymetry | `max|u|` < 1e-6 m/s after 30 days (§6.7 gate) |
| Inertia-gravity wave isotropy across seam | Phase-speed error in seam-crossing direction ≤ 5% of interior |
| Baroclinic Rossby wave on β-plane analog | Phase speed within 5% of analytic |
| Cross-seam tracer-front advection (sharp S front) | Zero new extrema to round-off (FCT monotonicity) |
| Mass conservation, 30 days | Drift ≤ round-off |
| Passive-tracer (h·T) conservation, 30 days | Drift ≤ round-off |
| RPE drift (Ilicak diagnostic), 30 days | Within factor 2 of lat-lon C-grid baseline |
| KE / enstrophy budget | KE drift bounded by explicit viscous dissipation |
| Visual inspection at face boundaries (n=48, n=96) | No cube imprint, no grid-scale noise |
| Single-rank vs MPI 4-rank, n=48, float64 | RMS difference ≤ 1e-12 |
| AD smoke test: `eqx.filter_value_and_grad` of segment loss | Finite gradient; matches finite-difference at 1e-5 |

CLAUDE.md "Validation Rules" mandate visual inspection of snapshots —
error norms alone are not sufficient when grid metrics or halo
exchange change.

### Phase 5 — Physics (deferred, separate PRs)
- KPP, convection: drop-in (column-local, grid-agnostic).
- GM/Redi: requires a triad operator on cubed sphere. **Same blocker
  as MPAS Phase 3 in #214.** Document the gap; do not attempt inside
  this plan.
- HK monitoring: keep biharmonic-Leith on; add CFL/HK diagnostic to
  `SegmentCarry` (zeroed each segment per CLAUDE.md "SegmentCarry
  discipline").

## 8. Validation matrix and CI

- Add a `cubed_sphere_cgrid` row to `scripts/matrix/run_ocean_test_matrix.py`
  and `scripts/run_ocean_all_grids_matrix.py`.
- Unit tests under `tests/ocean/unit/` for: edge-mask helper, metric
  continuity (§6.1), edge-parity velocity halo (§6.4), corner-cell
  masking (§6.2).
- Validation tests under `tests/ocean/validation/` for each Phase 4
  acceptance row.
- MPI 2-rank test under `tests/ocean/distributed/` once Phase 3
  smoke passes.

Scientific parity declared only when **all** Phase 4 rows pass. Until
then, the new dycore is experimental and the C-D grid CS ocean remains
the default.

## 9. Risks and tradeoffs

- **Highest risk**: cube-corner topology and edge-parity halo. Both
  were missing from the v1 plan; both are addressed in §6.2 / §6.4
  but the implementation has no existing precedent in this repo. The
  Phase 0 falsifying POC is designed specifically to detect failures
  in these two areas.
- **Tracer-thickness consistency** (§6.3) is the most invasive design
  decision — it changes `SegmentCarry`. Land it in one dedicated PR;
  do not interleave with PGF or Coriolis tuning.
- **Vertex-PV Coriolis** (§6.5) is new code, not a reuse of
  `fv3_cc2c`. Conservation properties must be verified discretely
  (Phase 0 KE-budget gate).
- **Topographic PGF** (§6.7): deferred to a follow-on PR. Realistic
  bathymetry runs are gated on Shchepetkin-McWilliams; do not promise
  global-realistic-bathymetry runs in this plan's deliverables.
- **GM/Redi gap** is real and shared with MPAS. Acknowledge in
  user-facing docs that eddy-permitting science remains gated on
  #214 Phase 3.
- **AD memory**: barotropic subcycle scan with 60–120 substeps under
  `filter_value_and_grad` will OOM without `jax.checkpoint`. Build
  this in from Phase 2; do not retrofit.

## 10. Out of scope

- GM/Redi on cubed sphere (waits on #214 Phase 3).
- Removal of `ocean_pe_cdgrid.py` (separate PR after parity).
- Vertical coordinate changes (stays z*).
- Coupled-mode runs with full atmosphere (bring-up is ocean-only).
- Shchepetkin-McWilliams density Jacobian PGF (separate PR; gated by
  the §6.7 seamount rest-state test).
- Partial bottom cells (separate PR; gated together with SM-PGF).
- Float32 PGF investigation (dropped; §6.13).

## 11. First concrete step

Implement the Phase 0 falsifying POC: 2-layer stratified geostrophic
adjustment with EOS + vertex-PV Coriolis + compact PGF + AB3, n=48,
30 days, single rank, float64. Five acceptance gates listed in §7
Phase 0. Estimate 3–5 days.

If any gate fails, root-cause before writing any of Phase 1. The POC
is deliberately designed to detect the failure modes the C-D grid
exhibited (baroclinic feedback at the seam) and the failure modes
specific to the cubed sphere C-grid (corner cells, edge-parity halo).
A passing POC is strong evidence the architecture is sound; a failing
POC saves weeks of Phase 1–2 work.
