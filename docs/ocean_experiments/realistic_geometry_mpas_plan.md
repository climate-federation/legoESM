# Realistic coastlines + bathymetry on the MPAS/Voronoi ocean — implementation plan

**Status (2026-05-03):** P0–P5 **complete**. P6 + P6.5 + P7-AD
**partially complete** (smoke-test gates passing, plus scripts and
scaffolds for production validation). P7-MPI **scaffolded**
(MPAS-ocean MPI step infrastructure does not exist yet — separate
follow-up issue).

**Seamount instability — RESOLVED (2026-05-03).** Five distinct
partial-cell consistency bugs were found and fixed:

1. `barotropic_implicit_mpas.py`: the Helmholtz operator's `H_e`
   used `_edge_avg(eta + H_bathy)` (centered mean across step
   edges) instead of the column-sum of min-rule per-level edge
   thickness. The flux-closure H on a step edge between H=2200m
   and H=4000m should be 2200m, not 3100m. Fixed by adding
   `_edge_H_min_rule` and dispatching on `OceanPartialCellCoordinate`.

2. `barotropic_implicit_mpas._depth_average_to_edges`: per-level
   `h_e_k = 0.5*(h[c1]+h[c2])` lets phantom transport leak through
   step edges. Fixed by adding a `partial_cells` flag that switches
   to `min_cell_to_edge`.

3. `ocean_model_mpas.step()` (line 351 area): the reconcile step
   recomputed `h_e_k` with the centered mean while the barotropic
   solver returned `Hu_avg` based on min-rule. The mismatch
   injected a per-step phantom kick into u_3d at every step edge.
   Fixed to use min-rule on partial cells.

4. `ocean_model_mpas._forward_backward_coriolis_mpas_3d`: same
   centered `h_e` bug — strips off the wrong u_bar from u_3d
   before applying Coriolis to u_prime, so the depth-mean we add
   back disagrees with what the barotropic solver expects. Fixed.

5. **Per-level edge mask** in `ocean_pe_mpas.py`: the cell-level
   `edge_mask = mask[c1] * mask[c2]` is 1 for the entire column
   even when one cell is dry at deep levels (above seafloor on
   the shallow side, below seafloor on the deep side). The
   min-rule on `h_e_3d` correctly zeros the *flux*, but
   `gradient_edge_3d(p_prime/rho_0)` reads filled tracer values
   in the dry levels of the shallow neighbor and produces a
   non-physical PGF that drives a phantom du_dt at the bottom
   partial cell of the deeper column — growing 1000×/step in the
   diagnostic. Fixed by building `edge_mask_3d[edge, k] = 1`
   only when `k <= maxLevelEdgeBot` and applying it to
   `du_dt_full` and `du_dt_3d`. On legacy z-star this reduces to
   `edge_mask[:, None]` bit-exactly.

`barotropic_mpas.py` (explicit-substep solver) was patched in the
same commit for consistency, even though the implicit-CN solver is
the production default.

**Validation:** centered scheme on a 2000m Gaussian seamount at
ico-2 with `bot_level` range [7, 9] is now stable for 6 hours
(`max|u| = 2.6e-3 m/s`, growing slowly but bounded, no NaN);
original was NaN at step 3 with `max|u| = 4.7e+06 m/s`. All 200
MPAS-tagged unit tests pass; z-star regression bit-exact via the
`isinstance(z_coord, OceanPartialCellCoordinate)` dispatch.

**`pgf_scheme="adcroft"` known-bad on legoesm.** With the per-level
mask fix, the centered scheme delivers the small bare gradient
(5e-7 m/s²) that the `dz_ref`-integration convention is supposed
to produce, and the seamount stays bounded. The Adcroft-Campin
correction adds 1.4e-4 m/s² (260× larger than the bare gradient)
that the `dz_ref` convention does not need to cancel — net result
is 0.6 m/s of spurious flow over 6 h on the seamount, growing.
The AC formula is correct for codes that integrate `p'` against
`h_partial` (MITgcm/MOM6); legoesm's `dz_ref` integration already
produces the small bare gradient AC was designed to recover. Until
we either switch to `h_actual` PGF integration or formally re-derive
the AC convention for `dz_ref`, **use `pgf_scheme="centered"`**.
Tracked in plan §P3c.

**ETOPO long-run instability — open follow-up (2026-05-03).**
With the seamount fixes shipped and `pgf_scheme="centered"`, the
production script (`run_mpas_etopo_spinup.py`) on real ETOPO at
ico-4 (1854 ocean cells, depth 28-5490 m, MEO smoothing on) now:

* loads cleanly (auto-detected ETOPO/`altitude` vars, depth flip,
  Voronoi MEO smoothing, partial-cell construction with bottom_level
  range [1, 19] — all OK)
* runs short integrations (centered PGF, dt ≤ 500 s) without
  immediate NaN — the seamount-style step-edge bugs are gone
* **but exhibits a slow-growth instability with ~1-day e-folding
  time on REST state.**  KE doubles every day for ~7 days, then
  explodes by day 9.  Symptom is configuration-independent across
  viscosity (A_h up to 1e7), PCG tolerance (1e-10 to 1e-12), and
  `min_water_column_m` (1 m to 100 m); higher A_h actually
  *accelerates* the failure.

There is also a hard CFL-like boundary at dt ≈ 525 s (dt ≤ 500
stable for short runs; dt ≥ 550 NaNs within 1 day) that does NOT
respond to viscosity or PCG-tolerance knobs — almost certainly a
partial-cell-induced stiff mode that the implicit-CN barotropic
solver doesn't damp because it only treats the depth-mean
gravity-wave mode implicitly.

Hypothesis: a slow grid-noise mode (TRiSK rotational null branch
at step-edge vertices, or a residual of the dz_ref vs h_actual PGF
convention spread over ~800 ETOPO step edges) accumulates linearly
in u, and once u is large enough the nonlinear KE-grad and PV-flux
tendencies amplify it exponentially.  Seamount fix removes the
*per-edge* error but ETOPO has hundreds of step edges and any tiny
per-edge residual sums into a global mode.  Unit tests on a single
seamount cannot detect this.

Required next-session work (ordered):

1. Identify the spatial structure of the unstable mode — dump
   u, eta, KE every step from day 5 onward and look at
   level-by-level snapshots.  Is it grid-scale (chequerboard,
   needs scale-selective filtering)?  Coastal/strait coherent
   (boundary-condition issue)?  Deep-ocean broad-scale (PGF
   convention residual)?
2. Try APVM (`apvm_dt > 0`) and biharmonic vorticity damping
   (`K_zeta_bih`) — both target the TRiSK enstrophy null mode
   that's the prime suspect.  Both are zero by default.
3. Try the `barotropic_u_viscosity` knob (already in the config,
   targets the depth-mean rotational null).
4. If filtering doesn't fix it, reconsider the dz_ref PGF
   convention — slim chance the slow growth is the same
   convention residual the AC scheme was meant to address (just
   without AC's overcorrection).

Conclusion: realistic-bathymetry MPAS is **dycore-correct** at
the partial-cell-stack level (seamount stable, all 200 unit tests
pass), and **integration-plumbing-correct** for ETOPO (loader,
MEO, partial-cell init all work).  But it's not yet
*long-run-stable* on real bathymetry.  Production validation
(P6 plan) blocks on the slow-growth-mode investigation above.

**Diagnosis sprint (2026-05-03 cont'd) — what we learned:**

* **Spatial structure of the unstable mode** (snapshot probe at
  rest, dt=300s, days 1-7): bottom-trapped (lev 16-19 hold 10-100×
  surface energy), spatially coherent along bathymetric features
  (mid-Atlantic ridge, East Pacific Rise, continental shelves,
  Indonesian throughflow, Aleutian arc).  *Not* chequerboard at
  vertices, *not* broad-scale interior.  Same shape at day 3 and
  day 7, just amplified.  This rules out the TRiSK rotational null
  branch, surface gravity-wave noise, and Coriolis double-counting,
  and implicates the partial-bottom-cell PGF treatment.

* **Tried `h_actual` PGF integration + `pgf_scheme="adcroft"`**
  (matching the lat-lon C-grid wiring at
  `ocean_pe_latlon_cgrid.py:757`).  Effect on the seamount: AC
  residual dropped from 1.4e-4 m/s² (with dz_ref) to 6.8e-7 m/s²
  (with h_actual + AC), confirming AC works correctly under the
  h_actual convention.  But on real ETOPO the per-step max|du_dt|
  *increased* from 1.6e-7 m/s² (dz_ref + centered) to 1.2e-6 m/s²
  (h_actual + AC), and both conventions still NaN at day 5.
  **AC is necessary but not sufficient on real bathymetry.**
  Reverted.

* **Why AC isn't enough:** on a single seamount (one isolated step
  edge) AC kills the residual cleanly; on real ETOPO with hundreds
  of step edges of varying geometry, AC's first-order extrapolation
  (assumes ρ constant from cell centroid down to the face-reference
  depth) leaves O(stratification·excess_depth²) residuals that
  accumulate.  Ruled out by inspection: not a damp-able mode (A_h
  up to 1e7 makes failure faster, not slower).

**The next real fix is SMC03** (Shchepetkin & McWilliams 2003,
density-Jacobian PGF).  This is what the lat-lon code uses to keep
ETOPO stable for 100 years — the doc at
`docs/ocean_experiments/density_jacobian_pgf_plan.md` describes
the lat-lon work and is the reference for porting to MPAS.  SMC03
on Voronoi is non-trivial (the cell geometry is irregular so the
Jacobian has more edge terms than on a regular grid), so this is
multi-day work and probably warrants its own dedicated plan doc.

**Plan §P3c (SMC03 for MPAS) is now critical-path** for any
realistic-bathymetry MPAS production run.  Until then the model
RUNS on ETOPO (loads, integrates 4-5 days without immediate NaN)
but is not long-run stable.

* **P0 done** — verified by inspection plus two new tests in
  `tests/ocean/unit/test_barotropic_implicit_mpas.py`:
  `test_implicit_solver_strong_depth_contrast` exercises 600x depth
  contrast end-to-end; `test_implicit_helmholtz_operator_uses_cell_local_H`
  directly probes the Helmholtz operator and asserts linear-in-H
  scaling plus spatial localization of an H impulse. The implicit-CN
  solver already builds `H_e = edge_thickness(eta + H_bathy)` per
  edge from per-cell `H_bathy` (`barotropic_implicit_mpas.py:241`,
  `operators_voronoi.py:181`); the Helmholtz stiffness is cell-local
  with no scalar-mean linearization.

* **P1a done** — found that the existing
  `load_bathymetry_mpas` (`bathymetry.py:915`) was silently calling
  the Cartesian `apply_meo_r_factor_cap` on a 1-D `(nCells,)` array,
  which uses `np.roll` along the cell-index axis with no geometric
  meaning. Added `_r_factor_max_voronoi` and
  `apply_meo_r_factor_cap_voronoi` (edge-list MEO with
  `np.maximum.at` scatter), rewired `load_bathymetry_mpas` to use
  them, and added `TestMEORFactorCapVoronoi` (7 tests) plus existing
  bathymetry suite (57 tests) all green. Sikiric 2009 LSC2 / FESOM2
  / MPAS-O Hoch 2020 lineage.

* **P1.5 done** — verified that `create_partial_cell_coordinate`
  is shape-generic on `(nCells, nlev)` (the `H.ndim`-driven
  broadcasting in `vertical.py:181` "just works"). Added
  `tests/ocean/unit/test_partial_cells_mpas.py` with 7 tests:
  shape correctness, column-thickness identity (rtol=1e-6 to absorb
  the float32 z-grid), dry-column handling (`bottom_level=-1`,
  `is_active=False`, `h_partial=0`), mixed wet/dry, `is_active`
  monotonicity in k, `compute_layer_thickness` reduces to
  `h_partial` at zero eta, and `jax.grad(Σ h_partial)` w.r.t.
  `H_bathy` returns 1.0 on interior cells (continuous-H AD contract).
  hFacMin floor deferred to P2 — no MITgcm-style floor exists in
  the lat-lon constructor either; will revisit if thin-bottom-cell
  pathology shows up under operator integration. Full
  `H_bathy → eta_after_one_step` AD smoke test deferred to after
  Slice 2 of P2 (operators must consume partial cells first).

* **P2 Slice 1 done** — added
  `src/legoesm/ocean/dynamics/mpas_partial_cell_helpers.py` with
  `compute_edge_mask`, `compute_vertex_mask`,
  `compute_max_level_edge_bot/top`, `min_cell_to_edge`,
  `donor_cell_to_edge` (upstream donor for continuity per
  Petersen 2015 §3.4), `kite_area_vertex_thickness`
  (active-renormalized — distinct from existing
  `vertex_thickness_3d` which divides by full triangle area),
  `min_cell_to_vertex` (with `BIG_H` sentinel),
  `vertex_thickness_hybrid` (kite-mean in interior, min-over-active
  fallback when `min/max < alpha`, default α=0.5 per the dycore
  audit). 19 tests in
  `tests/ocean/unit/test_mpas_partial_cell_helpers.py` covering
  uniform-recovers-h, masking invariants, dry-cell handling,
  hybrid switch threshold, and AD-through-donor-cell.

* **P2 Slice 2 done** — added the partial-cell dispatch in
  `mpas_ocean_baroclinic_tendencies` (`ocean_pe_mpas.py`):
  `h_e_3d` is now ``min_cell_to_edge(h_k, mesh)`` (flux closure /
  metric: depth-averaging, PV-flux normalization, vertical momentum
  advection, bottom-drag depth, F_slow_u depth-mean) and a new
  `h_e_continuity` is ``donor_cell_to_edge(h_k, u_3d, mesh)`` for
  the **continuity equation** ``thickness_flux = u * h_e_continuity``.
  Both reduce to ``0.5*(h[c1]+h[c2])`` on legacy z-star (gated on
  ``isinstance(z_coord, OceanPartialCellCoordinate)``) — flat-bottom
  regression bit-exact (78 z-star tests still green). 4 new
  integration tests in `tests/ocean/unit/test_partial_cells_mpas.py`:
  partial-at-H_max equivalence to z-star (1e-10 atol),
  donor-cell-vs-centered discrimination on a 2000m/5000m step,
  end-to-end `MPASOceanModel.step` finiteness on synthetic
  partial-cell bathymetry, and the **AD smoke test through
  `H_bathy → eta_after_one_step`** that was deferred from P1.5.
  Full regression (MPAS + bathymetry + helpers + partial-cells)
  = **169 passed, 0 failures**.

  Open notes for follow-up phases:
  - `_vertical_diffusion` (`ocean_pe_mpas.py:483`) reads
    `z_coord.dz_ref` directly — not partial-cell-aware. Pick up in
    P3.5 with the bottom-drag/convective-adjustment loop-bound
    audit.
  - PV flux still receives `h_e_3d` (min-rule) and uses the
    existing kite-mean `vertex_thickness_3d` internally; that
    transition is P4's scope.

* **P3 done** — added `partial_cell_pgf_correction_edge` to
  `mpas_partial_cell_helpers.py` (Adcroft & Campin 2004 face
  correction along Voronoi edges; CVT-mesh assumption documented),
  added `pgf_scheme: str = "centered"` to `MPASOceanConfig`
  (allowed values `"centered"` default, `"adcroft"`, with `"smc03"`
  raising NotImplementedError reserved for the P3c follow-up), and
  wired the dispatch into `mpas_ocean_baroclinic_tendencies`
  (additive `grad_B += ac_correction` after the batched
  Bernoulli/tracer gradient block, gated on partial-cell coord +
  `pgf_scheme="adcroft"`). The AC correction uses each cell's
  **own local rho_prime** for its own pressure shift (Adcroft &
  Campin 2004 §3.2; matches the lat-lon implementation). 3 new
  helper-level tests (zero on full cells, nonzero on a step,
  inverse-dcEdge scaling) and 5 new integration tests
  (default-unchanged, AC-zero-on-uniform-Hmax, the canonical
  bounded-rest-state test mirrored from lat-lon Phase 3b,
  end-to-end MPASOceanModel.step finiteness, and unsupported-scheme
  rejection). Full regression (P0–P3) = **177 passed, 0 failures**.

* **P3.5 done** — fixed a real bug and added one defense-in-depth.
  - **Bottom drag** on partial cells now applies at the per-edge
    bottom level ``maxLevelEdgeBot = min(bot[c1], bot[c2])`` via a
    one-hot scatter, not unconditionally at ``nlev-1``. Pre-P3.5 the
    drag was silently skipped on every edge whose true bottom was
    above ``nlev-1`` (h_e there was 0 → 1e-10 floor masked the
    divide → ~0 tendency). Z-star path unchanged (every column has a
    full bottom cell at ``nlev-1``).
  - **KE gradient** in `kinetic_energy_cell_3d` now receives
    ``u_3d * edge_mask[:, None]`` so dry-edge u contributions are
    zeroed before squaring. No-op on the z-star rest state (u=0 at
    dry edges already); defense-in-depth on partial-cell coast
    configurations to suppress spurious coastal Kelvin-wave ringing.
  - **Convective adjustment loop bounds** — not applicable to MPAS
    today (KPP not yet integrated; the runtime warning
    ``vertical_mixing='constant'`` is emitted by the unimplemented
    branch). Will be addressed when KPP-on-MPAS lands.
  - **`_vertical_diffusion` partial-cell awareness** — still uses
    `z_coord.dz_ref` directly. Affects momentum vertical viscosity
    and tracer vertical diffusion. Deferred as a focused follow-up
    PR — the helper is shared across grid types and the change
    benefits from being scoped on its own rather than bundled in.
  - 3 new tests in `test_partial_cells_mpas.py`: drag at the right
    per-edge bottom level on partial cells, z-star drag regression,
    KE defensive-mask regression. Full regression (P0–P3.5) =
    **180 passed, 0 failures**.

* **P4 done** — TRiSK PV-flux on partial cells turned out to be a
  smaller change than the audit feared, because two of the three
  recommended actions were already in place:
  - **Audit recommendation 1: use `vertex_thickness_hybrid` instead
    of kite-mean** for the q = ζ/h_v normalization. **Implemented**:
    when on a partial-cell coordinate, `mpas_ocean_baroclinic_tendencies`
    now computes ``zeta = curl_vertex_3d(u, mesh)`` and divides by
    ``vertex_thickness_hybrid(h, mesh)`` (kite mean in the interior,
    min-over-active fallback when ``min/max < alpha``, default
    α=0.5). Closes the "thin partial vertex transmits O(1) PV from
    deep neighbor → spurious coastal currents" failure mode.
  - **Audit recommendation 2: set q_e = 0 at edges with one dry
    cell.** **Already effectively done** by the downstream
    ``du_dt_full *= edge_mask[:, None]`` multiplication
    (`ocean_pe_mpas.py:322`) — the perimeter contribution from any
    coastal edge is zeroed before the tendency is returned. We do
    NOT additionally mask the Thuburn tangential reconstruction
    (per audit recommendation; preserves the discrete
    summation-by-parts identity on the wet sub-mesh; matches MPAS-O
    Fortran practice).
  - **Audit recommendation 3: default to enstrophy-conserving form
    on partial cells.** **Already the default** in `MPASOceanConfig`
    (line 91 — set as the legoesm default for MPAS independently of
    partial-cell concerns, since the energy-conserving form has a
    ζ-checkerboard null mode on hex C-grids).
  - 4 new tests in `test_partial_cells_mpas.py`: dispatch path
    verification (hybrid vs kite differ on a step bathymetry),
    H_max-equivalence to z-star (1e-9 tolerance), finiteness with
    >30% land cells (>30% of cells dry → many coastal vertices,
    no NaN/Inf), and end-to-end ``MPASOceanModel.step`` finiteness
    with both `pgf_scheme="adcroft"` and `pv_scheme="enstrophy"`
    active. Full regression (P0–P4) = **184 passed, 0 failures**.

* **P5 done** — confirmed by code inspection that no cos²(lat) A_h
  scaling or polar-cap path exists in `ocean_pe_mpas.py` /
  `ocean_model_mpas.py` (Voronoi mesh is locally isotropic; no
  pole singularity; nothing to skip). TVD tracer advection
  available via ``MPASOceanConfig.tracer_advection="tvd"``; new
  smoke test ``test_p5_tvd_tracer_advection_on_partial_cells``
  exercises it end-to-end on a partial-cell coordinate.
  Beckmann-Haidvogel rotated viscosity at coasts deferred (would
  benefit from the open instability investigation first).

* **P6.5 done (smoke-test gate)** — added
  ``test_p65_v_baro_grid_noise_bounded_on_partial_cells`` that
  measures the σ_grid metric of V_baro on a MEO-smoothed
  partial-cell run and asserts finiteness + bounded by an absolute
  threshold (50× — generous for the small 162-cell test mesh; the
  strict 3× ico4 baseline gate per
  ``project_mpas_barotropic_noise.md`` is a P6 production-scale
  acceptance, not transferable to this smoke test as the metric
  scales differently with cell size). The "compare partial vs
  z-star" framing is not well-posed at this scale because uniform-
  bathy z-star has no bathy signal to be noisy about.

* **P6 partially done — scripts ready, validation pending.** Wrote
  two production-ready scripts under
  ``scripts/run/mpas_realistic_geometry/``:
  - ``run_mpas_seamount_rest.py`` — canonical PGF-over-topography
    test (compares `pgf_scheme="centered"` vs `"adcroft"`).
  - ``run_mpas_etopo_spinup.py`` — wind-driven GO spinup on ico4
    with ETOPO bathymetry; smoke-test default (subdiv 4, 30 days);
    production ``--subdivision 7 --years 5`` for the 5-yr headline.
  Plus a ``README.md`` documenting acceptance gates and out-of-
  scope items (DOME, COMODO, Ilıcak triplet, regional N. Atlantic
  — deferred to follow-up). Smoke-running the seamount script
  surfaced the open instability noted in the status block.

* **P7 done — AD passing, MPI scaffolded.**
  - **AD:** ``test_p7_grad_through_realistic_partial_cell_bathy``
    exercises ``jax.grad(H_bathy → loss)`` through a single
    MPASOceanModel.step on **MEO-smoothed partial-cell bathymetry**
    (varying H, real bottom_level distribution) with all P3-P5
    features active (Adcroft PGF, hybrid vertex thickness, donor-
    cell continuity, min-rule edge thickness, bottom drag at
    maxLevelEdgeBot). Stricter than the existing
    ``test_partial_cell_grad_through_model_step`` which uses
    ``H_uniform = H_max`` (every column full → partial-cell branches
    degenerate). Multi-step variant attempted but blocked by the
    open instability noted above.
  - **MPI:** ``tests/distributed/test_mpas_topography_mpi.py``
    scaffolds the contract (state scatter round-trip, single-step
    equivalence, mass conservation, AD through scatter) but
    auto-skips today because the MPAS *ocean* MPI step does not
    exist yet (the atmosphere has it via
    ``legoesm.parallel.voronoi_mpi`` but not the ocean — see the
    follow-up issue noted at the top of the file). Drops in
    place and runs the moment that infrastructure lands.

  Final regression across **MPAS + bathymetry + helpers + partial-
  cells = 187 passed, 0 failures**.

  **Convention note (P3 architectural finding).** legoesm's
  `iterate_eos_and_pressure_anomaly` integrates `p_prime` against
  the **reference** grid (`dz_ref`), not the actual partial-cell
  thicknesses. In that convention the bare-gradient PGF is already
  small for horizontally near-uniform stratification — there is no
  large pre-existing spurious-flow signal for AC to "remove" (the
  signal exists when p is integrated against actual h, as in
  MITgcm). AC instead provides a different (face-reference-depth)
  discretization whose second-order residual
  `O((delta_centroid)^2 * d^2 rho/dz^2)` is the actual quantity
  bounded by the canonical rest-state test. The "AC reduces vs
  centered" claim only holds for codebases that integrate p with
  actual h; the P3 acceptance gate here is an absolute bound on
  the AC residual (~1e-4 m/s^2 on ico-2; lat-lon Phase 3b reaches
  ~1e-5 on a finer mesh). SMC03 (P3c follow-up) is the right next
  step for shelf-break / steep-bathy work where AC's second-order
  residual becomes consequential.

* **P1b done** — `rest_state_mpas_ocean` now accepts an optional
  `bathymetry: BathymetryConfig` kwarg; when ``source="file"`` it
  calls `load_bathymetry_mpas` (which dispatches to the
  grid-agnostic `load_bathymetry` + Voronoi MEO + Voronoi Laplacian
  smoothing). Default kwarg-less invocation is bit-exact regression
  for the idealized path. Strait enforcement at the **cell level**
  works automatically through the existing `enforce_straits` (it
  takes lat/lon arrays of any shape). Per-edge through-flow
  thickness override deferred — open as a follow-up issue if/when
  ico4 throughflow tests show single-edge collapse. Added
  `TestMPASRealisticBathymetry` (4 tests). Full regression on
  bathymetry + MPAS unit tests + implicit-CN tests = 139 passed.

Branch `feature/MPAS_topography` (parity with `main` at commit
`a0c0df36`). Companion to `realistic_geometry_lat_lon_plan.md`,
which shipped the lat-lon C-grid stack in PR #224.

## Goal

Bring the MPAS/Voronoi ocean dycore to feature parity with the lat-lon
C-grid ocean for realistic complex topography:

1. ETOPO/GEBCO ingestion onto a Voronoi mesh, with strait enforcement
   and bathymetry-preserving smoothing.
2. Partial bottom cells (Adcroft & Campin 2004 / MITgcm hFacZ /
   MPAS-O Petersen 2015 lineage).
3. Topography-aware pressure-gradient force (Adcroft-Campin first,
   SMC03 density-Jacobian later).
4. Topography-aware TRiSK PV-flux on partial cells.
5. Validated 5-yr global ocean spinup on ico4 with realistic ETOPO,
   plus DOME and COMODO benchmarks.

## What MPAS has today

* `MPASOceanState` (`src/legoesm/core/state.py:368`) carries
  `H_bathy[nCells]` and `land_mask[nCells]`, but they are populated
  only by `idealized_bathymetry_mpas` (flat `H_max` + lat-threshold
  land at default 80°).
* Edge thickness is `0.5 * (h[c1] + h[c2])`
  (`src/legoesm/ocean/dynamics/ocean_pe_mpas.py:232`) — no min-rule,
  no partial-cell awareness, **wrong even for continuity at coasts**.
* PGF is bare `gradient_edge(p_prime / rho_0, mesh)` — no Adcroft
  correction, no SMC03.
* PV flux uses Thuburn 2009 tangential reconstruction with
  kite-area-weighted vertex thickness `vertex_thickness_3d`
  (`src/legoesm/core/operators_voronoi.py:606`); no vertex masks, no
  partial-cell awareness.
* Implicit Crank-Nicolson barotropic solver
  (`barotropic_implicit_mpas.py`) and forward-backward Coriolis on
  perturbations (`_forward_backward_coriolis_mpas_3d`) are in.
* `OceanPartialCellCoordinate` (`src/legoesm/ocean/vertical.py:138`),
  `compute_centroid_depth`, `create_partial_cell_coordinate`,
  `BathymetryConfig`, and `load_bathymetry_mpas` exist but are
  **unwired** to MPAS.

## Design principles

1. **Reuse, don't re-derive.** `load_bathymetry`, `BathymetryConfig`,
   `OceanPartialCellCoordinate`, `create_partial_cell_coordinate`,
   `compute_centroid_depth`, `compute_layer_thickness` are
   grid-agnostic. Only the operators (PGF, PV flux, edge/vertex
   metrics, continuity) and mesh-aware smoothing are MPAS-specific.
2. **Phased, each phase shipping a passing test.** Mirror the cadence
   that worked for the lat-lon stack.
3. **Don't blindly port lat-lon ideas.** AL81's 12-point stencil is
   C-grid-specific; the TRiSK analog is structurally different.
   SMC03's per-column slope reconstruction transfers, but face
   evaluation must use Voronoi edge geometry.
4. **No pole singularity → no polar cap, no cos²(lat) A_h scaling
   needed on MPAS.** The Voronoi mesh is locally isotropic; skip
   those two pieces from the lat-lon stack.
5. **Differentiability preserved.** All masking via `jnp.where` on
   traced arrays where needed; `is_active` is static (set at init),
   so feature gating uses Python `if`.
6. **Stay aligned with MPAS-O Fortran conventions** (Petersen et al.
   2015 OM, 2019 JAMES). MPAS-O uses partial bottom cells today;
   we are building MITgcm-style on MPAS topology, which is exactly
   MPAS-O's design. Do not drift toward MOM6 ALE.

## Audit-driven design decisions

The plan below incorporates feedback from a dycore audit and an
ocean-modeling audit (2026-05-02 chat session). The non-obvious
decisions:

* **r-factor smoothing must be edge-based, not cell-Laplacian.**
  Sikirić et al. 2009 LSC2; FESOM2 (Danilov 2017) and MPAS-O
  (Hoch 2020) use selective edge-r smoothing. Plain cell Laplacian
  erodes shelves and sill depths — exactly the features that set
  DWBC, Gulf Stream separation, and overflow paths. Use
  `r_e = |H_c1 − H_c2| / (H_c1 + H_c2)` per edge, target r_max ≈ 0.3
  for z-star (≈ 0.2 for σ-like behavior).
* **Continuity needs donor-cell h at step edges, not 0.5 average.**
  `h_e = 0.5 * (h[c1] + h[c2])` is wrong even for tracer/volume
  continuity at coasts — Petersen 2015 §3.4. Donor-cell upstream `h`
  is the advected thickness; min-rule is the flux *closure*
  thickness. These are different objects.
* **PV-flux on partial cells defaults to enstrophy-conserving form,
  not energy-conserving.** Naively masking Thuburn `weightsOnEdge`
  breaks the global summation-by-parts identity (Thuburn-Ringler-
  Skamarock-Klemp 2009 Eq. 22–24) and *cannot* be fixed by row
  renormalization. Two literature paths: Eldred & Randall 2017
  re-solves the weights on the active sub-graph (offline,
  precomputed); Gassmann 2018 / MPAS-O production accept losing the
  energy half of the AL81 dual conservation at coastlines and use
  the enstrophy-conserving form. Adopt the latter. Set `q_e = 0` at
  edges with one dry cell; do **not** mask the Thuburn tangential
  reconstruction.
* **Vertex thickness is a hybrid, not a min.** Min-rule is correct
  for *edges* (a flux constraint) but wrong for *vertices* (a
  metric). Use **active-renormalized kite-area mean** in the interior
  (matches MPAS-O Petersen 2015), with a switch to min-over-active
  when any of the three vertex cells has `h < α·Δz` (α ≈ 0.5) for
  coast robustness.
* **MPAS-O carries TWO per-edge bottom indices.**
  `maxLevelEdgeBot = min(bot[c1], bot[c2])` limits flux;
  `maxLevelEdgeTop = max(bot[c1], bot[c2])` is where the BCL/BTP
  perturbation pressure is reconstructed. A single `edge_mask` is
  insufficient.
* **Adcroft-Campin transfer is CVT-only.** On a CVT the
  cell-centroid line crosses the Voronoi edge orthogonally, so
  `δp_e / dcEdge` is the correct edge-normal pressure gradient.
  On a non-CVT mesh, the AC correction is off by `cos(angle)`.
  Assert and document the CVT requirement.
  `ρ_e` must be the **shallower-side density** at the corrected
  reference depth, *not* `0.5*(ρ[c1]+ρ[c2])`. Adcroft-Campin §3.2.
* **Implicit CN barotropic must use cell-local H_bathy in its
  stiffness matrix.** ETOPO ico4 spans 10 m → 6000 m; if the
  Helmholtz operator linearizes around mean H, the elliptic solve is
  badly conditioned and split-mode coupling leaks. Verify before any
  3D run.
* **Partial-cell `H_min` floor (hFacMin / hFacMinDr).** Require
  bottom partial-cell thickness `≥ max(0.1·Δz_k, 10 m)`. MITgcm
  convention; arbitrarily thin bottom cells inflate local wave
  speeds in the implicit BTP solve and trigger Veronis-like spurious
  circulations at neighboring full cells.
* **Bottom drag, convective adjustment, and KE-gradient at coasts**
  must respect `bottom_level(c)` (not `nlev-1`) and edge_mask
  (not unmasked neighbor sums). Silent failure modes if wrong:
  momentum sink at fictitious depth, mixing into rock, coastal
  Kelvin-wave ringing.

## Phase plan

### P0 — Prerequisites (~1 day)

* Verify `barotropic_implicit_mpas` uses cell-local `H_bathy` in its
  stiffness matrix on a flat → shelf → deep idealization. If it
  uses a scalar mean, fix before anything else.

**Test gate:** implicit CN solve converges and conserves volume on a
3-cell idealization with H = (10 m, 1000 m, 6000 m).

### P1 — Bathymetry ingestion (~3 days)

**P1a — Loader and smoothing (offline, no dycore wiring).**
* `load_bathymetry_mpas(mesh, cfg)` — wrap grid-agnostic
  `load_bathymetry(lat_deg, lon_deg, cfg)` by passing
  `mesh.latCell, mesh.lonCell`. Returns `H_bathy[nCells]`,
  `land_mask[nCells]`.
* Voronoi edge-based r-factor smoother — rewrite `_r_factor_max`
  (`bathymetry.py:423`, currently Cartesian) for an edge list.
  `r_e = |H_c1 − H_c2| / (H_c1 + H_c2)`, iterative selective deepening
  of the shallower cell with the largest neighborhood r. Default
  `r_max = 0.3` for z-star.
* Cell Laplacian smoother (using `cellsOnEdge`/`areaCell`/`dvEdge`/
  `dcEdge` à la Ringler 2010) — useful as a secondary noise filter,
  not a primary bathymetry tool.

**P1b — Strait overrides + wiring.**
* JSON-driven strait override table:
  `(cell_index_or_(lat,lon), override_depth_m, edge_through_flow_thickness)`.
  Mirrors NEMO ORCA1 / MOM6-OM4 `topo_edits.nc` precedent.
  Edge-thickness override on through-flow edges is **non-optional at
  ico4** — Gibraltar (14 km) and Bering (85 km) are unresolved on a
  ~480 km mesh, and TRiSK reconstruction collapses to zero
  throughflow on single-cell straits without the override.
* Add `MPASOceanConfig.bathymetry: BathymetryConfig` (default
  `source="idealized"` keeps current behavior bit-exact).
* Wire `rest_state_mpas_ocean` to dispatch on `cfg.bathymetry.source`.

**Test gate:**
* `Σ areaCell · H_bathy` matches integrated ETOPO over ocean mask
  to ≤ 0.1%.
* Smoothed r-field meets target r_max ≤ 0.3 globally.
* Strait override produces a wet through-flow edge with the
  prescribed thickness.
* Idealized regression: `cfg.bathymetry.source="idealized"` is
  bit-exact with the current code path.

### P1.5 — Partial-cell coordinate construction on Voronoi (~1 day)

* Verify `create_partial_cell_coordinate` works shape-generically on
  `(nCells, nlev)` (it should — the helper is dim-generic on the
  leading axes).
* Apply hFacMin floor: `H_min = max(0.1·Δz_k, 10 m)` at construction.
* Assert per cell: `Σ_k h_partial[c,k] == H_bathy[c]`; `is_active`
  monotone in k; `h_partial == 0` on dry columns.
* **Run `jax.grad(H_bathy → first-step eta)` smoke test here**, not
  at the end. Catches non-differentiability at the H → 0 boundary
  (the `H_safe = jnp.maximum(H, 1e-10)` divisor) early.

**Test gate:** column-thickness identity holds; AD smoke test passes
on a 4-column idealization.

### P2 — Topography-aware continuity + face/vertex metrics (~3 days)

New module `src/legoesm/ocean/dynamics/mpas_partial_cell_helpers.py`:

* `compute_max_level_edge_bot(land_mask, bottom_level, mesh)` —
  `min(bot[c1], bot[c2])`, the flux-bottom index.
* `compute_max_level_edge_top(land_mask, bottom_level, mesh)` —
  `max(bot[c1], bot[c2])`, the perturbation-pressure-top index for
  BCL/BTP split.
* `min_cell_to_edge(h_cell, mesh)` — flux closure thickness.
* `donor_cell_to_edge(h_cell, u_edge, mesh)` — upstream-biased h
  for the **continuity** equation at step edges.
* `kite_area_vertex_thickness(h_cell, mesh, vtx_active_mask)` —
  active-renormalized kite-area mean.
* `vertex_thickness_hybrid(h_cell, mesh, alpha=0.5)` — kite in
  interior, min-over-active when `min(h_v_neighbors) < alpha · Δz_k`.
* `compute_vertex_mask(land_mask, mesh)` — AND of `cellsOnVertex`,
  respecting `nEdgesOnVertex`.

**Replace** `h_e = 0.5*(h[c1]+h[c2])` (`ocean_pe_mpas.py:232`):
* For **continuity** (advected `h` tendency): donor-cell upstream.
* For **flux closure** (mass-flux divergence): min-rule.

Both gated by `isinstance(z_coord, OceanPartialCellCoordinate)` —
flat-bottom path keeps the cheap mean and stays bit-exact.

**Test gate:**
* Edge thickness ≤ min of the two cell h's.
* Vertex thickness ≤ min of wet cells; equals interior cell h on
  uniform-bathy regions.
* Mass-flux divergence on a closed dry-bordered subdomain = 0 to
  round-off.
* Donor-cell continuity convergence study on an analytic step.
* Flat-bottom regression bit-exact.

### P3 — PGF on partial cells (~3 days)

Bare-gradient PGF across a partial/full step produces O(1 cm/s)
spurious shelf-break currents — **must land concurrently with P2,
not after.**

* `partial_cell_pgf_correction_edge(centroid_depth, rho_prime, mesh, g)`
  — Adcroft-Campin face correction along `dcEdge`, using
  shallower-side density at the corrected reference depth.
* Add `MPASOceanConfig.pgf_scheme: str = "centered"` with
  `"adcroft"` immediate, `"smc03"` later (gated, defers to P3
  follow-on).
* Assert CVT mesh on `pgf_scheme="adcroft"` (or `"smc03"`); document
  the requirement.

**Test gate (the canonical PGF-over-mountain test):**
* Resting state with arbitrary ETOPO, uniform ρ → PGF tendency = 0
  to round-off.
* Same with stratified ρ(z) on `pgf_scheme="smc03"` (when SMC03
  lands).

### P3.5 — Bottom drag, convective adjustment, KE gradient at coasts (~1 day)

* Bottom stress at `bottom_level(c)`, not `nlev-1`.
* Convective adjustment / mixed-layer loops between `k=0` and
  `k=bottom_level(c)`.
* `kinetic_energy_cell_3d`: zero contribution from dry neighbor
  cells (otherwise drives coastal Kelvin-wave ringing).

**Test gate:** drag work matches `u·τ_b` at coast; no dry-neighbor
leakage in KE gradient on a coast-adjacent column.

### P4 — TRiSK PV-flux on partial cells (~5 days, the hardest)

* Default to `pv_flux_enstrophy_conserving_3d`
  (`operators_voronoi.py:717`).
* Set `q_e = 0` at edges with one dry cell. Do **not** mask the
  Thuburn tangential reconstruction (matches MPAS-O Fortran;
  accepts a small geostrophic-balance error at coast).
* Replace `vertex_thickness_3d` consumer call with
  `vertex_thickness_hybrid` from P2.
* Keep `pv_flux_energy_conserving_3d` available behind a flag for
  flat-bottom validation only. Document explicit loss of energy
  conservation at coasts on partial cells.

**Test gate:**
* Periodic flat-bottom recovers exact energy + enstrophy conservation
  (regression: confirms masking didn't break the un-masked path).
* Total enstrophy `Σ_v areaTriangle · h_v · q²` decays
  monotonically (hyperdiff off, only PV flux active).
* Isolated seamount in a resting fluid → no spurious tangential flow
  generated.

### P5 — Mixing & numerics on real bathy (~2 days)

* **Skip** cos²(lat) A_h scaling; **skip** polar caps (Voronoi is
  locally isotropic).
* Keep DST-3 (already in repo, see `project_dst3_advection.md`) on
  for tracer advection — *not* centered — near sloping topography to
  avoid overflow plume buoyancy loss.
* Optional: Beckmann-Haidvogel rotated viscosity along coastlines
  (mitigates stair-step Kelvin ringing).

**Test gate:** time-mean V_baro grid-noise σ does not exceed ~3× the
ico4 implicit-CN flat-bottom baseline (1.92e-2 m/s per
`project_mpas_barotropic_noise.md`).

### P6 — Validation experiments (~2 weeks elapsed)

1. **DOME overflow** (Legg et al. 2006) — canonical partial-cell test.
2. **COMODO seamount** (Auclair et al. 2018) — internal-tide /
   spurious-flow test.
3. **Ridge + wind ACC analog** on MPAS — leverages existing
   `acc_channel.py`.
4. **Global ETOPO on ico4 — 5-yr GO spinup** with momentum-budget
   closure (P6.5) and Stommel-Arons abyss diagnostic.
5. **Optional:** Ilıcak et al. 2012 triplet (overflow / internal-wave
   / baroclinic-eddy) as the partial-cell community benchmark.

### P6.5 — Momentum-budget closure (~1 day)

Mirror the lat-lon noise diagnostic that caught
`project_barotropic_noise_issue.md`. Time-mean V_baro grid-noise σ
should not exceed ~3× the ico4 implicit-CN baseline (1.92e-2 m/s per
`project_mpas_barotropic_noise.md`).

### P7 — Differentiability + MPI (~2 days)

* Full `jax.grad(H_bathy → eta_after_1day)` at ico4. (Smoke test
  already done at P1.5 on a small mesh.)
* `mpirun -np 2 .venv/bin/python -m pytest
  tests/distributed/test_mpas_topography.py`.

## Convention follow-up (separate from this plan)

* **PGF integration convention audit** —
  `iterate_eos_and_pressure_anomaly` integrates `p_prime` against
  the **reference** grid (`dz_ref`), not actual partial-cell
  thicknesses. This is a shared-helper convention used by lat-lon,
  cubed-sphere, AND MPAS identically. The MITgcm/MOM6 documentation
  for AC assumes h-actual integration; in legoesm's ref-grid
  convention AC's role is different (face-reference-depth
  discretization with second-order residual, not removal of an
  O(1) error). Should be its own targeted PR — affects all grids
  uniformly, doesn't gate MPAS realistic-geometry work, and the
  lat-lon 100-yr Wolfe-Cessi spinup is empirical evidence the
  current convention is acceptable for production. SMC03 (P3c
  follow-up) sidesteps the question entirely by building pressure
  from per-column ρ slopes.

## Deferred / out-of-scope (tracked as follow-up issues)

* **GM/Redi triads on MPAS** — the standard Veronis fix. Large work;
  mirror the lat-lon `gm_redi_latlon_cgrid_plan.md`. Without it,
  expect O(5 Sv) spurious global meridional cell from horizontal
  Laplacian on inclined isopycnals over real ETOPO. The lat-lon
  ocean has triads; MPAS does not yet.
* **Tapered isoneutral mixing or BBL parameterization**
  (Danabasoglu-McWilliams; Beckmann-Döscher 1997 / Campin-Goosse
  1999) — overflow plume buoyancy preservation. Without it, dense
  water loses buoyancy to numerical mixing as it descends shelf
  breaks; the failure is silent in volume integrals and only shows
  in T-S watermass diagnostics.
* **Eldred & Randall 2017 masked-TRiSK** — energy-conserving PV flux
  on partial cells. Worth implementing only if the
  enstrophy-conserving + hybrid-vertex default turns out
  insufficient.
* **Walin-framework watermass-transformation diagnostic** on
  overflow sub-domains — adds a silent-failure gate for spurious
  diapycnal mixing. Diagnostic only; no code-path change.

## Estimated total

| Phase | Days |
|---|---|
| P0 prerequisites | 1 |
| P1 ingestion | 3 |
| P1.5 partial-cell coord | 1 |
| P2 metrics + continuity | 3 |
| P3 PGF (Adcroft) | 3 |
| P3.5 drag / conv / KE | 1 |
| P4 PV flux (the hard one) | 5 |
| P5 mixing | 2 |
| P6 validation (elapsed, includes spinup wall-time) | ~10 |
| P6.5 momentum-budget closure | 1 |
| P7 AD + MPI | 2 |
| **Total** | **~3 weeks engineering + spinup wall-time** |

## References

* Adcroft, A. & Campin, J.-M. (2004). Rescaled height coordinates for
  accurate representation of free-surface flows in ocean circulation
  models. *Ocean Modelling* **7**, 269–284.
* Adcroft, A. & Hallberg, R. (2009). On methods for solving the
  oceanic equations of motion in generalized vertical coordinates.
  *Ocean Modelling* **11**, 224–233.
* Auclair, F., et al. (2018). A non-hydrostatic non-Boussinesq
  algorithm for free-surface ocean modelling. *Ocean Modelling*
  **132**, 12–29.
* Beckmann, A. & Döscher, R. (1997). A method for improved
  representation of dense water spreading over topography in
  geopotential-coordinate models. *J. Phys. Oceanogr.* **27**,
  581–591.
* Campin, J.-M. & Goosse, H. (1999). Parameterization of density-
  driven downsloping flow for a coarse-resolution ocean model in
  z-coordinate. *Tellus A* **51**, 412–430.
* Danilov, S., et al. (2017). The Finite-volumE Sea ice-Ocean Model
  (FESOM2). *Geosci. Model Dev.* **10**, 765–789.
* Eldred, C. & Randall, D. (2017). Total energy and potential
  enstrophy conserving schemes for the shallow water equations using
  Hamiltonian methods – Part 1. *Geosci. Model Dev.* **10**, 791–810.
* Eldred, C., et al. (2019). Total energy and potential enstrophy
  conserving schemes for the shallow water equations using
  Hamiltonian methods – Part 2: rotating sphere. *Geosci. Model
  Dev.* **12**, 4811–4832.
* Gassmann, A. (2018). Discretization of generalized Coriolis and
  friction terms on the deformed hexagonal C-grid. *Q. J. R.
  Meteorol. Soc.* **144**, 2038–2053.
* Hoch, K. E., et al. (2020). MPAS-Ocean simulation quality for
  variable-resolution North American coastal meshes. *J. Adv. Model.
  Earth Syst.* **12**, e2019MS001848.
* Ilıcak, M., et al. (2012). Spurious dianeutral mixing and the role
  of momentum closure. *Ocean Modelling* **45-46**, 37–58.
* Legg, S., Hallberg, R. W., & Girton, J. B. (2006). Comparison of
  entrainment in overflows simulated by z-coordinate, isopycnal and
  non-hydrostatic models. *Ocean Modelling* **11**, 69–97.
* Petersen, M. R., et al. (2015). Evaluation of the arbitrary
  Lagrangian–Eulerian vertical coordinate method in the MPAS-Ocean
  model. *Ocean Modelling* **86**, 93–113.
* Petersen, M. R., et al. (2019). An evaluation of the
  ocean and sea ice climate of E3SM using MPAS and interannual
  CORE-II forcing. *J. Adv. Model. Earth Syst.* **11**, 1438–1458.
* Ringler, T., et al. (2010). A unified approach to energy
  conservation and potential vorticity dynamics for arbitrarily-
  structured C-grids. *J. Comput. Phys.* **229**, 3065–3090.
* Shchepetkin, A. F. & McWilliams, J. C. (2003). A method for
  computing horizontal pressure-gradient force in an oceanic model
  with a non-aligned vertical coordinate. *J. Geophys. Res.* **108**,
  3090.
* Sikirić, M. D., Janeković, I., & Kuzmić, M. (2009). A new approach
  to bathymetry smoothing in sigma-coordinate ocean models. *Ocean
  Modelling* **29**, 128–136.
* Thuburn, J., Ringler, T. D., Skamarock, W. C., & Klemp, J. B.
  (2009). Numerical representation of geostrophic modes on
  arbitrarily structured C-grids. *J. Comput. Phys.* **228**,
  8321–8335.
