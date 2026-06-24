# SMC03 density-Jacobian PGF for the MPAS Voronoi ocean dycore

**Status:** plan, 2026-05-03
**Owner:** dycore (DB)
**Driver:** ETOPO+ico4 long-run instability — bottom-trapped mode along
bathymetric features causes NaN at ~day 5 with `pgf_scheme="centered"`;
`"adcroft"` is 7× worse on real bathymetry. Lat-lon's centered+AC stack
runs 100 yr stably *only* once SMC03 is enabled. This plan ports SMC03
to the Voronoi mesh.

**Reference docs (read these first):**
- `docs/ocean/experiments/density_jacobian_pgf_plan.md` — original
  algorithmic design and lat-lon validation phases.
- `docs/ocean/experiments/pgf_smc03_code_review.md` — Adcroft & Campin
  2004 face-reference choice rationale, the C1 "midpoint clamp" bug,
  the `min(z_c)` fix.
- `docs/ocean/experiments/realistic_geometry_mpas_plan.md` — why this
  is the critical-path follow-up after the 5-bug seamount fix.

---

## 1. Why SMC03

SMC03 (Shchepetkin & McWilliams 2003 §4) replaces the centered-difference
+ Adcroft-Campin face-correction stack with a per-column
**piecewise-linear ρ(z) reconstruction with harmonic-mean monotonized
slopes**, evaluated at a face-reference depth and differenced
horizontally. Its essential property is the **rest-state cancellation**:
adjacent columns reconstruct ρ at any common depth *identically* when ρ
is exactly linear in z, regardless of how their cell centroids sit
relative to one another. This eliminates the partial-cell PGF residual
that:

- The centered scheme can't see at all (it operates in `dz_ref`, not
  `h_actual`, so the centroid offset never enters the gradient).
- The Adcroft-Campin correction tries to fix with a single-level
  pressure shift — but on real bathymetry that correction is itself a
  thin spike at the partial-cell interface, which forces a 2Δz
  vertical computational mode (the spatially coherent bottom-trapped
  blow-up we observed at day 5 of ETOPO ico4).

SMC03 is the production answer in ROMS, CROCO, NEMO, and (per the
adcroft audit) the lat-lon C-grid path here.

## 2. Reuse strategy: the algorithm is already grid-agnostic

This is the key insight that scopes the port down from "multi-week
rewrite" to "a few hundred LOC + tests". The two SMC03 leaf functions
in `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py`:

- `reconstruct_harmonic_slopes(rho_per_cell, z_centroid, is_active, eps)`
  — `:2083`. Operates on arrays of shape `(..., nlev)`. The internal
  rolls (`jnp.concatenate([rho[..., :1], rho[..., :-1]], axis=-1)`) are
  **vertical only** — they touch the level axis, not any horizontal
  axis. Column-independent: each column computes `σ_k` from its own
  three-cell vertical stencil `(k-1, k, k+1)`. Nothing about its
  derivation assumes lat-lon adjacency.

- `compute_pressure_at_target_smc03(rho_per_cell, h_partial, z_centroid,
  sigma, z_target, g)` — `:2186`. Also per-column: `cumsum` along the
  level axis, `argmax`+`take_along_axis` to find the enclosing cell of
  each target depth. Inputs are `(..., nlev)`; targets are `(..., n_t)`.

Both work as-is for an MPAS column array of shape `(nCells, nlev)` —
the leading dimension just becomes the cell index instead of `(n_lat,
n_lon)`. **No changes to either helper are required.**

The lat-lon-specific code is the 30-line wrapper
`density_jacobian_pgf_smc03_x` (and its y-twin) at `:2288`, which:

1. Computes `z_centroid = cumsum(h_partial) − 0.5·h_partial` per cell.
2. Calls `reconstruct_harmonic_slopes`.
3. Gets the W-neighbor by `jnp.roll(·, 1, axis=1)` (longitude periodic).
4. Computes face-reference depth `z_target_face = min(z_c_W, z_c_E)`.
5. Calls `compute_pressure_at_target_smc03` twice (once per column).
6. Returns `(P_E - P_W) / dx_u`.

**The MPAS port replaces step 3 (axis-roll) with `cellsOnEdge`
gather** and step 6's lat-lon metric with `dcEdge`. Everything else is
identical.

## 3. The MPAS edge-loop wrapper

```python
def density_jacobian_pgf_smc03_mpas(
    rho_per_cell: jnp.ndarray,   # (nCells, nlev) baroclinic anomaly ρ'
    h_partial:   jnp.ndarray,    # (nCells, nlev) per-cell layer thickness
    is_active:   jnp.ndarray,    # (nCells, nlev) wet mask
    mesh,                        # Voronoi mesh
    g: float,
) -> jnp.ndarray:                # (nEdges, nlev) edge-normal PGF accel.
    # 1. Per-cell preprocessing — REUSE lat-lon helpers.
    z_centroid = jnp.cumsum(h_partial, axis=-1) - 0.5 * h_partial
    sigma = reconstruct_harmonic_slopes(rho_per_cell, z_centroid, is_active)

    # 2. Edge-pair gather: c1, c2 = the two cells across each edge.
    c1 = mesh.cellsOnEdge[0]   # (nEdges,)
    c2 = mesh.cellsOnEdge[1]   # (nEdges,)
    rho_1, rho_2 = rho_per_cell[c1], rho_per_cell[c2]
    h_1,   h_2   = h_partial[c1],   h_partial[c2]
    z_c_1, z_c_2 = z_centroid[c1],  z_centroid[c2]
    sigma_1, sigma_2 = sigma[c1], sigma[c2]

    # 3. Face-reference depth — shallower of the two centroids
    # (Adcroft & Campin 2004 convention; see lat-lon SMC03 docstring
    # at latlon_cgrid_operators.py:2342-2358 for the C1-bug rationale).
    z_target_face = jnp.minimum(z_c_1, z_c_2)   # (nEdges, nlev)

    # 4. Per-column pressure at the SAME face depth — REUSE.
    P_1 = compute_pressure_at_target_smc03(
        rho_1, h_1, z_c_1, sigma_1, z_target_face, g,
    )                                   # (nEdges, nlev)
    P_2 = compute_pressure_at_target_smc03(
        rho_2, h_2, z_c_2, sigma_2, z_target_face, g,
    )                                   # (nEdges, nlev)

    # 5. Edge-normal pressure-gradient acceleration. MPAS sign
    # convention for gradient_edge_3d is (φ_c2 − φ_c1) / dcEdge —
    # match that exactly so the dispatcher sees identical sign /
    # units to the centered branch it replaces.
    return (P_2 - P_1) / mesh.dcEdge[:, jnp.newaxis]
```

**Total: ~25 lines of new code** in `src/legoesm/ocean/dynamics/operators_voronoi.py`
(or a new `pgf_smc03_mpas.py` if we prefer one-file-per-scheme).

The output has the same shape and edge-normal sign convention as
`gradient_edge_3d(p_prime / rho_0, mesh)` so the dispatcher can swap
it in directly.

## 4. Dispatcher integration

Current state at `src/legoesm/ocean/dynamics/ocean_pe_mpas.py:294-308`:

```python
pgf_scheme = getattr(config, "pgf_scheme", "centered")
if pgf_scheme == "adcroft" and isinstance(z_coord, OceanPartialCellCoordinate):
    ac_correction = partial_cell_pgf_correction_edge(...)
    grad_B = grad_B + ac_correction
elif pgf_scheme not in ("centered", "adcroft"):
    raise NotImplementedError(...)   # SMC03 stub
```

The challenge: the centered/AC path computes `gradient_edge_3d(bernoulli)`
where `bernoulli = ke + p_prime / rho_0` — KE-grad and PGF are batched
into **one** gradient call as a perf optimization (see
`ocean_pe_mpas.py:249-280`). SMC03 is *not* a gradient-of-a-scalar, so
it can't reuse that batched call.

Replace the dispatcher with:

```python
pgf_scheme = getattr(config, "pgf_scheme", "centered")
if pgf_scheme == "smc03" and isinstance(z_coord, OceanPartialCellCoordinate):
    # Decompose: SMC03 PGF + separate centered KE-grad.
    grad_KE = gradient_edge_3d(ke, mesh)   # (nEdges, nlev)
    pgf_smc03 = density_jacobian_pgf_smc03_mpas(
        rho_prime, z_coord.h_partial, z_coord.is_active, mesh,
        g=constants.g,
    )
    grad_B = grad_KE + pgf_smc03 / rho_0
    # tracer-gradient batching: do separately when SMC03 is on.
    if _need_tracer_grad:
        _tracer_grad_flat_pre = gradient_edge_3d(_tracer_flat_pre, mesh)
elif pgf_scheme == "adcroft" and ...:   # unchanged
    grad_B = grad_B + ac_correction
elif pgf_scheme == "centered":   # unchanged (no-op)
    pass
else:
    raise ValueError(f"pgf_scheme={pgf_scheme!r} unsupported on MPAS")
```

The conditional structure of the bernoulli build at `:262-280` needs a
pre-check so it skips the SMC03 case (or is restructured to compute KE
alone when SMC03 is on). Cleanest: lift the `pgf_scheme == "smc03"`
flag above the bernoulli build and branch the `gradient_edge_3d` calls
accordingly.

**Perf cost on SMC03 path:** one extra `gradient_edge_3d` call per RHS
evaluation (KE alone instead of KE+PGF batched). Acceptable —
SMC03 is the correctness-critical path; if profiling shows the loss is
material, we can refactor `gradient_edge_3d` to accept an optional
"replace channel k with this externally-computed array" hook.

**Allowed values update** (`src/legoesm/ocean/mpas_config.py`): extend
the `pgf_scheme: Literal[...]` to include `"smc03"`. Add a config-time
check that `"smc03"` requires `OceanPartialCellCoordinate` (raise on
z-star — not meaningful and would force an irrelevant code path).

## 5. Module placement: where does the code live?

Two helpers (`reconstruct_harmonic_slopes`, `compute_pressure_at_target_smc03`)
are currently in `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py`.
They are grid-agnostic but live in a lat-lon-named module.

**Decision: promote to a grid-neutral module.** Create
`src/legoesm/ocean/dynamics/pgf_smc03.py` that owns both helpers plus
the new MPAS edge-loop wrapper. Re-export the two helpers from
`latlon_cgrid_operators.py` for back-compat (existing lat-lon SMC03 +
its 4 phase-test files keep working without edits). The lat-lon
wrappers `density_jacobian_pgf_smc03_x/y` stay where they are (they
*are* lat-lon-specific).

This avoids the cross-import smell of `ocean_pe_mpas.py` reaching into
a `latlon_*` module, and matches the existing convention that
grid-neutral pieces (`vertical.py`, `eos.py`, `bathymetry.py`) live at
the `ocean/` or `ocean/dynamics/` level rather than inside a grid
module.

## 6. Validation phases

### Phase A — unit tests (new file `tests/ocean/unit/test_pgf_smc03_mpas.py`)

A1. **Linear ρ(z), flat bottom, ico-2 mesh:** SMC03 PGF should be
    machine-zero everywhere. (The lat-lon equivalent is
    `test_pgf_smc03_phase3.py::test_pgf_zero_for_linear_density_*`.)

A2. **Linear ρ(z), seamount, ico-2 mesh:** still machine-zero. This is
    the property that motivates SMC03; the centered scheme passes too
    on dz_ref but would *fail* under h_actual integration. AC fails
    with O(1e-3) m/s residual.

A3. **Nonlinear ρ(z) (exponential T(z), no bathymetry variation):**
    SMC03 result matches centered + Adcroft to O(h²·ρ'') — the
    fundamental accuracy floor of the harmonic-linear scheme. Use
    `assert_allclose(rtol=1e-2)` at typical n_levels=20.

A4. **AD smoothness:** `jax.grad` of `sum(SMC03(rho).flatten())**2`
    w.r.t. `rho_per_cell` produces a finite, non-NaN gradient. Mirror
    the lat-lon Phase-3 AD test.

A5. **Shape and dtype:** output shape `(nEdges, nlev)`; dtype matches
    input.

### Phase B — seamount end-to-end (regression)

Re-run `scripts/run/mpas_realistic_geometry/run_mpas_seamount_rest.py` with
`--schemes centered adcroft smc03`. Pass criterion: SMC03 max|u| at
6h ≤ centered (~few×1e-5 m/s); ideally lower than both. Update the
script's `--schemes` default to include `smc03` once it passes.

### Phase C — ETOPO 30-day smoke

`scripts/run/mpas_realistic_geometry/run_mpas_etopo_spinup.py` with
`pgf_scheme="smc03"`, `--years 0.083` (30 days), `--dt 500`. Pass
criterion: completes without NaN (vs. centered's day-5 blowup).
Snapshot the bottom-trapped diagnostic (re-run the
`probe_etopo_modestructure.py` style snapshots) and confirm the
spatially coherent mode along bathymetric features is gone or
substantially reduced.

### Phase D — ETOPO 5-yr headline run

The plan-spec validation: ico4 + 5 yr + ETOPO + SMC03. Pass criteria
from `realistic_geometry_mpas_plan.md` §P6:
- Integration completes without NaN.
- max|u| stays bounded (< 2 m/s) once spun up.
- Time-mean V_baro grid noise σ < 3× the implicit-CN baseline
  (1.92e-2 m/s per `project_mpas_barotropic_noise.md`).
- AMOC magnitude in observed range (15–25 Sv).
- No spurious deep flow under western boundary topography (visual).

If D passes, this completes the realistic-bathymetry MPAS plan.

## 7. Risks and open questions

**R1 — Bernoulli decomposition perf cost.** The split adds one
`gradient_edge_3d` call per RHS. On ico4 this is ~1% of step time
(measured for centered). Acceptable.

**R2 — Tracer-gradient batching breaks under SMC03.** The current
optimization batches `(bernoulli, T, S)` into one `gradient_edge_3d`
call when `K_h > 0` or `K_bih > 0`. Under SMC03 we have to split that
back into `(KE, T, S)` for batching, since PGF doesn't pass through
`gradient_edge_3d` anymore. Implementation detail, not a risk; flagged
for the dispatcher edit.

**R3 — Edge-mask / per-level edge mask.** The 5-bug seamount fix
established `edge_mask_3d` that gates per-level edge participation.
SMC03 PGF must be multiplied by `edge_mask_3d` exactly the same way
the centered+AC path is at `ocean_pe_mpas.py:~330` (the `du_dt_full *
edge_mask_3d` line). Easy to miss — explicit test in Phase A2.

**R4 — `cellsOnEdge` indexing convention.** MPAS conventions vary on
whether `cellsOnEdge` is 0-indexed or 1-indexed and whether the order
matters for sign. Verify by reading `gradient_edge_3d` source — match
the same `c1, c2` order so signs are consistent. (`gradient_edge_3d`
already documents its convention; cross-check against the
`partial_cell_pgf_correction_edge` AC operator which has the same
contract.)

**R5 — z_centroid reference frame.** Lat-lon SMC03 uses η=0 reference
to match `iterate_eos_and_pressure_anomaly`. MPAS must do the same;
read `compute_centroid_depth(eta, H_bathy, z_coord)` at line 296 of
`ocean_pe_mpas.py` and confirm it accepts `eta=0` for the SMC03 path.
The current AC branch uses `jnp.zeros_like(eta)` — same pattern.

**R6 — Non-vanishing slope at partial seafloor.** The lat-lon
`reconstruct_harmonic_slopes` boundary handling uses one-sided slope
at the bottom-active cell. The pgf_smc03_code_review flagged this as
a potential inconsistency at faces between full + partial cells (the
two cells use different one-sided slopes at their respective deepest
levels). Lat-lon's testing showed it's not load-bearing on Phase-3
benchmarks, but the seamount and ETOPO geometry exposes more shelf-
break edges. Watch in Phase B/C diagnostics; if a residual appears
near edges where one column is partial-bottom and the other is full,
revisit the σ-extension scheme that the code review proposed.

**R7 — PGF is `[Pa·m/m] = [Pa]`, must divide by ρ₀ for acceleration.**
Lat-lon SMC03 returns Pa/m (pressure gradient); the dispatcher then
divides by ρ₀ to get acceleration. The MPAS wrapper above returns the
*raw* pressure gradient (units Pa/m); the dispatcher must divide by
ρ₀ before adding to `grad_B` (which carries `KE + p'/ρ₀`, both in
acceleration units). The pseudo-code at §4 has this right; Phase A5
tests should assert units explicitly.

## 8. Out of scope (deferred follow-ups)

- **The dt=525s CFL boundary** observed on centered+ETOPO. SMC03 may
  shift this either way; not a goal of this plan but should be
  documented after Phase C.
- **GM/Redi compatibility.** GM/Redi is currently lat-lon only on this
  branch; SMC03+GM is a future composition.
- **The Phase-3 R6 follow-up** (σ-extension at partial seafloor).
  Defer until ETOPO 5-yr run shows whether it matters at scale.

## 8a. Implementation result (2026-05-03)

**Phases A, B (unit tests, seamount):** PASS.

- 7 SMC03-MPAS unit tests pass (linear-ρ machine-zero on Gaussian
  seamount partial cells, SMC03 ≪ Adcroft on seamount, no spike-like
  vertical structure, AD smoothness, flat-bottom equivalence).
- Seamount end-to-end (ico-2, 6h, exp T(z)): centered 2.6e-3 m/s,
  adcroft 6.0e-1 m/s, **smc03 1.5e-2 m/s** — bounded and ~40× better
  than adcroft.

**Phase C (ETOPO 30-day smoke):** **FAILS** — but not in the way
expected.  ico4 + ETOPO + smc03 + dt=500s blew up at day 6 (max|u|=0.30
at day 2, 5.4 at day 4, NaN at day 6).  Centered with the same
config blew up at day 5.  So *both schemes blow up*; SMC03 just gets
there ~10× faster once horizontal density gradients have developed.

**Triage diagnostic** confirmed it is NOT a SMC03 defect:

1. **Initial state is column-uniform:** `rest_state_mpas_ocean`
   initializes T(z) using `legoesm.ocean.eos.scale_depth` per *level*,
   not per cell centroid.  `T_init[k]` span across all wet cells is
   exactly 0.000 C.  So no initial horizontal density gradient; no
   "adjustment shock" from a non-equilibrium IC.
2. **Centered and SMC03 are identical to 4% for the first 20 steps**
   (~3 h of sim time).  Both grow linearly at ~9.3e-4 m/s per step
   (= 1.86e-6 m/s² acceleration ≈ wind-stress / column-mean depth =
   τ_max / (ρ_0 · h_top) ≈ 5e-7 m/s²; close to the wind-driven
   Ekman spin-up rate, with the rest from hydrostatic adjustment).
   No sign error, no scaling error in the SMC03 dispatch.
3. The divergence between schemes happens **later**, once advection
   has produced non-trivial horizontal T (and hence ρ') gradients.
   At that point SMC03 forces baroclinic flow more strongly than
   centered's bare-dz_ref gradient — but apparently more than the
   model's bottom drag and lateral viscosity can absorb on real
   bathymetry.

**Reframing.** The previous session's diagnostic concluded the
ETOPO instability was "a partial-bottom-cell PGF origin" — but that
was an over-confident reading.  The mode it identified
(bottom-trapped, spatially coherent along bathymetric features:
mid-Atlantic ridge, East Pacific Rise, shelves, Indonesian
throughflow) is *consistent with* a partial-cell PGF defect, but it
is also exactly what a real bottom-intensified mode (topographic
Rossby waves / slope currents / bottom EKE) looks like when the
model lacks sufficient bottom dissipation.  The two are
indistinguishable from a single snapshot.

The new evidence (this session) settles the ambiguity in favor of
the second interpretation: SMC03 — the ROMS/CROCO-grade PGF that
*is* the answer to a partial-cell PGF defect — does **not** suppress
the mode.  It excites it faster.  A more accurate PGF feeding a
real bottom-intensified mode that nothing else damps is exactly the
behavior we observe.

**Likely root causes (for the underlying ETOPO instability — the
issue both schemes share):**

- (A) **Bottom drag underspecified at partial cells.** Drag is
  applied at ``maxLevelEdgeBot`` with thickness from the partial
  cell.  Thin partial bottoms get weak drag in absolute terms — and
  those are exactly the cells where bottom-trapped flow concentrates
  along bathymetric features.  Worth comparing the MPAS drag
  formulation against the lat-lon path to check for missing thickness
  floors or rescaling.
- (B) **Stiff mode from the barotropic-baroclinic split with partial
  cells.** The ``dt ≲ 525s`` CFL boundary observed under centered
  (per the script docstring) is hard evidence of an under-damped
  stiff mode the implicit-CN solver doesn't catch.  SMC03 may
  interact with this mode more strongly because its PGF is smoother
  in z (no single-level Adcroft spike) — less inadvertent
  dissipation at the partial-bottom interface.
- (C) **Mesh-quality artifacts at ico-4 pentagon vertices.**  The 12
  icosahedral pentagons concentrate cell-shape distortion; under
  realistic bathymetry these may be where modes nucleate.  Worth
  confirming by checking whether the spatial blow-up pattern lights
  up the pentagon neighbourhoods.
- (D) **ETOPO 1° interpolation onto ico-4 (~500 km cells) producing
  coastline / shelf-break artifacts** that act as broad-band
  forcing for the bottom mode.

What is **not** a candidate (corrected from the original wording):
- ❌ ``cos²(lat) · A_h`` scaling.  That lat-lon scaling is a
  coordinate-singularity stability fix (lat-lon ``dx`` shrinks like
  ``cos(lat)`` so the viscous CFL ``ν·dt/dx²`` blows up at high lat
  unless ``ν`` is rescaled).  MPAS Voronoi cells are quasi-equal-area
  globally; ``dcEdge`` is roughly uniform; no analogous constraint
  exists.  An earlier draft of this section incorrectly listed it
  as the most likely cause — that was a transfer-of-intuition error
  from the lat-lon experience.

**Decision:** SMC03 implementation lands as completed (it is correct,
tested, and a strict improvement over Adcroft on partial cells).
The ETOPO long-run instability is **not** SMC03-specific and is
reframed as a separate diagnostic effort.

**Diagnostic results (2026-05-03 same session):**

Step 1 — **spatial pattern under SMC03 matches centered** (snapshots
in ``outputs/mpas_etopo_smc03_modes/``).  At day 1: surface 0.158
m/s (wind jets), mid-depth 0.030 m/s, **bottom 0.070 m/s**.  Bottom
> mid-depth from day 1 = bottom-intensified mode born at depth.

Step 2 — **dt-insensitive.** dt=200 s vs dt=500 s give identical
max|u| at day 1 to 0.4%.  ⇒ rules out stiff numerical mode
(hypothesis B); it is a genuine growth-rate dynamic mode.

Step 3 — **bottom drag CFL violation found.** Doubling+ drag
(``r=4.4e-3`` or ``1.1e-2``) makes the model NaN at day 1.
Investigation: thinnest partial-cell bottom is **0.28 m** (the
``min_water_column_m = 1.0 m`` floor is on the whole column, not
individual partial cells), and even at baseline ``r·dt = 0.55 m`` —
already 2× the minimum bottom-cell thickness, so explicit drag
removes more flow than the cell contains and sign-reverses.

Step 3a — **lateral viscosity sensitivity scan** (the surprise that
identified the second cause):

| A_h    | bot speed at day 1 | factor |
|--------|-------------------|--------|
| 1e5    | 6.35e-2 m/s       | 1.0    |
| 3e5    | 5.64e-2           | 0.89   |
| 1e6    | 3.86e-2           | 0.61   |
| 3e6    | 1.93e-2           | 0.30   |

Lateral viscosity DOES suppress the mode (not the surface — the
surface is wind-driven and unaffected, but the bottom mode
responds).  ``A_h = 1e5`` gives a diffusion timescale ``dx²/A_h ≈
30 days`` on ico-4 — far too weak to damp a O(1 day) mode.

**Two real root causes** (independent; both must be fixed; both
shared by centered & SMC03):

- **(α) Lateral viscosity is too low** for ico-4 with realistic
  bathymetry.  ``A_h`` should be O(1e6) m²/s, not 1e5 (inherited
  from earlier flat-bottom experiments).
- **(β) Explicit bottom drag CFL-violates at thin partial cells.**
  Need one of: implicit drag in the bottom layer, a partial-cell
  drag floor ``r_eff = min(r, h_bot/dt)``, or a separate drag-cell
  thickness floor distinct from ``min_water_column_m``.

**Fixes implemented (same session, 2026-05-03):**

1. **Distributed-BBL drag** (Killworth & Edwards 1999 / MOM6
   ``BBL_thick_min``).  Better than implicit drag because dt-
   independent and physics-motivated.  Lat-lon already had it
   (``_bbl_drag_for_face`` in ``ocean_pe_latlon_cgrid.py``); lifted
   to grid-agnostic ``bbl_distributed_drag_face_column`` in
   ``ocean_tendency_common.py``, ported to MPAS via new
   ``MPASOceanConfig.bottom_drag_bbl_thickness`` field.  Default 0
   (back-compat); recommended 50 m for partial-cell ETOPO.  7 unit
   tests in ``tests/ocean/unit/test_bbl_drag.py``.
2. **Default A_h on ETOPO script raised 1e5 → 1e6 m²/s** with
   docstring explaining why (30-day vs 3-day diffusion timescale on
   ico-4).
3. **ETOPO 30-day with both fixes + SMC03**: early-day shelf-break
   mode is **completely fixed**.  max|u| stays bounded at ~0.3-0.4
   m/s through day 16 (was NaN at day 6 before the fixes).

**New finding (2026-05-03 same session — separate next-session
problem):**

A slow mass-conservation mode in the partial-cell barotropic-
baroclinic split emerges around day 18 in the fixed ETOPO config.
Confirmed with a tau_max=0 (no wind) diagnostic: eta grows
identically with or without wind (0.16 m at day 5 → 1.67 at day 15
→ NaN at day 25).  Pattern is e-folding ~5 days for eta, then
super-exponential late blowup.  This is **not** SMC03, not BBL, not
viscosity — orthogonal bug.  Most likely a partial-cell-specific
TRiSK null mode that the implicit-CN solver doesn't catch (it does
eliminate the analogous flat-bottom mode per
``project_mpas_barotropic_noise.md``).  Lives in the partial-cell
barotropic-baroclinic split; needs its own diagnostic plan.

What is **not** a candidate (corrected from earlier wording above):
- ❌ ``cos²(lat) · A_h`` scaling.  Lat-lon-coordinate-singularity
  fix; doesn't apply to MPAS Voronoi.

## 8b. Day-18 mode diagnosed and fixed (2026-05-03 cont.)

The day-18 wind-independent eta-growth mode flagged in §8a was
characterized in a focused diagnostic ladder and fixed structurally.

**Diagnostic ladder** (all 30-day, ico-4, tau_max=0):

1. **Flat-bottom (uniform H, all full cells)**: max|u|=0 EXACTLY
   through 30 days.  Bug requires partial cells + bathymetry.
2. **Heavy bathy smoothing** (smoothing_passes=30): NaN day 20.
   Step-edge magnitude is NOT the dominant factor — partial-cell
   mechanism alone suffices.
3. **Unstratified T_surf=T_deep=10**: NaN day 26 with same day-2
   max|u| as stratified.  SMC03 stratification residual is NOT the
   seed (EOS pressure-dependence on partial cells is enough).
4. **PGF residual probe on rest state**: max|PGF/ρ_0|≈2e-7 m/s²,
   peaks at the deepest layers (k=18-19); |Δu| kick ~1e-4 m/s after
   one dt=500s step.  PGF residual is the SEED (kicks u≠0 on step 1)
   but NOT the MODE that grows.
5. **Rayleigh damping on u_bar (1-day timescale)**: converts
   exponential → linear growth, eliminates NaN through 30 days.
   ⇒ **the unstable mode lives in u_bar** (not in eta).
6. **Cross-ref ``docs/dev-notes/issues/barotropic_mode_noise.md``**: same kind
   of mode identified on lat-lon as "C-grid Coriolis rotational null
   mode" with ∇·U=0 AND f·V=0 simultaneously — invisible to the
   Helmholtz operator AND to the Coriolis predictor-corrector.
   Recommended fix there was Follow-up C: barotropic-mode lateral
   viscosity acting on U_bar/V_bar directly.  MPAS config already
   exposed ``barotropic_u_viscosity`` but it was wired ONLY into the
   explicit-substep path (``barotropic_mpas.py:272``).

**Structural fix:** added the same operator to the implicit-CN path
(``barotropic_implicit_mpas.py``, ~10 LOC):

```python
A_baro_visc = jnp.asarray(
    getattr(config, "barotropic_u_viscosity", 0.0), dtype=eta_dtype,
)
if config.barotropic_u_viscosity > 0.0:
    lap_u = vector_laplacian_del2(u_bar_new, mesh).astype(eta_dtype)
    u_bar_new = (u_bar_new + dt_t * A_baro_visc * lap_u) * edge_mask
```

Default in ``run_mpas_etopo_spinup.py`` raised to ``--baro-u-viscosity
3e6 m²/s``.

**Validation matrix** (ETOPO + ico-4 + SMC03 + BBL + tau=0, 30-day):

| A_baro       | day-30 max|u| | day-30 max|eta| | NaN?     |
|--------------|---------------|-----------------|----------|
| 0 (no fix)   | NaN day 24    | NaN             | yes      |
| 3e5          | 0.53          | 0.96 m          | bounded  |
| 3e6 (chosen) | 0.53          | 0.27 m          | bounded  |
| 1e7          | 0.54          | 0.19 m          | bounded  |

Velocity is invariant w.r.t. viscosity (seed-driven, not mode-
amplification); eta growth scales inversely.  3e6 chosen as default
— diminishing returns past that and matches lat-lon production scale.

**Wind-driven 30-day with fix**: max|u|=0.53 m/s, max|eta|=0.98 m at
day 30 (vs NaN day 24 without fix).

**Residual work** (next follow-up): at A_baro=3e6, eta grows linearly
~0.83 cm/day in tau=0.  Acceptable for 30-day smoke and 1-yr spinup;
for the headline 5-yr the next mitigation is a depth-dependent
reference profile ``ρ_ref(z)``: production code currently uses
``ρ' = ρ - ρ_0`` (constant), giving ~24× larger PGF residual than a
horizontal-mean reference.  Subtracting ``ρ_ref(z)`` would eliminate
most of the seed and the linear ramp with it.

Tests added: ``test_implicit_solver_baro_u_viscosity_damps_u_bar``
(asserts the operator is wired in by comparing u_bar magnitudes
between viscosity=0 and viscosity=1e7).  All 51 MPAS-relevant unit
tests pass.

## 8c. Equatorial mode diagnosed and partially fixed (2026-05-03)

After §8b's commit (Laplacian on u_bar) only delayed NaN by ~10
days, two independent expert reviews (ocean-model-expert and dycore-
expert) were commissioned.  Both flagged that the proposed fix was
a band-aid and recommended different paths.  Phase 1 implemented the
dycore-expert recommendation (APVM + ∇⁴-on-u_bar + K_ζ_bih); Phase 2
implemented the ocean-expert recommendation (depth-dependent
ρ_ref(z) seed reduction).  Neither resolved the mode:

- **APVM was a no-op** — numerical fix that targets PV null modes,
  but this mode doesn't go through PV (numbers identical with vs
  without to four digits).
- **K_ζ_bih was also a no-op** for the same reason.
- **∇⁴-on-u_bar buys +10 days but at 5× worse early eta growth** —
  scale-selective Laplacian damps grid-scale but not the actual mode.
- **Dynamic ρ_ref(z) made things worse** (NaN day 60 vs 70).  The
  recomputed-each-step horizontal mean creates positive feedback as
  the spatial pattern of T,S drifts.  Seed at day 5 was 3× smaller,
  but exponential growth doubled (e-folding 22d → 10d).

**Spatial diagnostic** (snapshot u_baro / u_bot / max|u_3d| / eta at
days 2/5/10/20/40/60 of the tau=0 reproducer with del2+MEO 0.2):

- Pattern is **persistent and amplifying in place** from day 2
  onward — not a propagating wave.
- ``|u_bot|`` >> ``|u_baro|`` at hot spots (0.80 vs 0.14 m/s at day
  60) — issue is in BOTTOM cells, leaking up via depth-averaging.
- Hot spots are **NOT** at thin partial cells, NOT at high-r-factor
  edges, NOT at pentagons, NOT at high latitudes.
- **Top-100 hot-spot edges cluster in the equatorial Pacific**
  (mean |lat| = 22°, only 1% above |lat|=60°), median bottom
  partial-cell thickness 398 m (vs 228 m globally — typical deep
  ocean, not anomalous).

**Mechanism**: at the equator ``f → 0`` so the implicit-CN solver's
Coriolis predictor-corrector (which provides rotational restoring on
``u_bar``) loses its stabilizing torque.  Modes with both
``∇·U ≈ 0`` and ``f·V ≈ 0`` simultaneously are invisible to the
Helmholtz solve AND to the Coriolis Heun step — a null mode of the
discretization that's energized by any non-zero PGF residual.

**Structural fix**: per-edge ``cos²(lat)`` boost on lateral
viscosity, mirroring the lat-lon ``A_h_lat_scaling`` mechanism but
with a controllable strength.

```python
# in MPASOceanConfig
equatorial_visc_boost: float = 0.0    # 0 = uniform, 5 = recommended

# in barotropic_implicit_mpas (and ocean_pe_mpas for A_h):
cos2 = jnp.cos(mesh.latEdge) ** 2
lat_factor = 1.0 + equatorial_visc_boost * cos2          # (nEdges,)
u_bar_new += dt * A_baro_visc * lat_factor * lap_u
```

The lat-lon code already had ``A_h_lat_scaling`` for "polar
coordinate-singularity stability" — but its side effect of *more*
viscosity at low latitudes (cos²(0)=1) was ALSO masking this
equatorial mode on the lat-lon path.  My earlier note in
``project_mpas_etopo_instability.md`` that "cos²(lat) doesn't apply
to MPAS Voronoi" was wrong — the polar-singularity rationale is
lat-lon-specific, but the equatorial-boost side effect is needed on
EVERY grid that uses an implicit-CN free-surface solver.

**Validation matrix** (ETOPO + ico-4 + SMC03 + BBL + MEO 0.2,
180-day except as noted):

| ``equatorial_visc_boost`` | tau=0 NaN | wind-driven NaN  |
|---------------------------|-----------|------------------|
| 0 (no boost)              | 24-26 d   | 24 d             |
| 5 (chosen as default)     | >90* d    | **180 d** (5-7×) |
| 10                        | 170 d     | (not tested)     |
| 20                        | 10 d      | (over-damped)    |
*tau=0 boost=5 was only run to 90 days; eta at 0.15 m, max|u| at 3.0
m/s and growing — extrapolating, NaN around day 120-150.

**Residual mode persists, but slower**.  This is a substantial
improvement (5-7× lifespan), enough to unblock 30-day smoke and
1-yr spinups.  For 5-yr stability we likely need ALSO:

1. A static (rather than dynamic) ρ_ref(z) — eliminates the
   recomputed-mean feedback that broke Phase 2.
2. Higher resolution (ico-5 or ico-6) — equatorial deformation
   radius is ~250 km; ico-4 (~500 km cells) under-resolves the
   equatorial waveguide, which is intrinsic to the problem.
3. A narrower Gaussian sponge profile (instead of broad cos²) so
   the boost is more targeted to the equator.

These are tracked as follow-up issues.

## 8d. SMC03 deprecated in favor of AC + h_actual (2026-05-03)

After §8c, ran a controlled comparison of all three implemented PGF
schemes on the same ETOPO+ico4 stack (BBL+MEO 0.2+cos²(lat) viscosity
+del2 baro_u_visc 3e6+equatorial_visc_boost 5):

| PGF (90-day tau=0)          | day-90 max\|u\| | day-90 max\|eta\| | NaN day |
|-----------------------------|----------------:|------------------:|--------:|
| SMC03                       | 3.05            | 0.15              | ~150    |
| centered + dz_ref           | 2.70            | 0.23              | ~120    |
| centered + h_actual         | NaN at 65       | —                 | 65      |
| AC + dz_ref                 | NaN at 5        | —                 | 5       |
| **AC + h_actual** (chosen)  | **1.20**        | 0.21              | 165     |

| PGF (1-yr wind-driven)      | NaN day  |
|-----------------------------|---------:|
| SMC03                       | 180      |
| AC + h_actual               | 165      |

**ALL three "production-grade" PGF schemes (SMC03, centered+dz_ref,
AC+h_actual) reach essentially the same ~150-180 day NaN ceiling.**
The PGF choice is irrelevant to the equatorial-mode lifespan — what
matters is the cos²(lat) viscosity boost.  ALSO confirmed on lat-lon
(36×72, 5° ETOPO 30-day): AC + h_actual and SMC03 give numerically
identical results (max\|u\|=0.972 vs 0.970 at day 30; max\|eta\|=0.106
vs 0.107) — SMC03 was producing the same answer as AC + h_actual on
both grids the whole time.

**Why we ended up with SMC03 in the first place** (history correction):

The early MPAS partial-cell debugging logged that ``pgf_scheme=adcroft``
gave a "260× over-correction" (`docs/ocean/experiments/realistic_
geometry_mpas_plan.md:60-72`).  Re-reading that note in 2026-05-03:
the AC formula is correct only when ``p'`` is integrated against
``h_partial`` (MITgcm/MOM6 convention).  We were integrating against
``dz_ref``, so AC was correcting for a partial-cell artifact that our
integration convention had already smoothed out — net result was
double-counting.  We then picked SMC03 (which builds ``p_face``
directly from harmonic-slope reconstruction and so is convention-
agnostic) without ever testing the canonical ``AC + h_actual`` recipe
that the production codes (MITgcm, MOM6, NEMO ``ln_hpg_zps``) use.

**Survey of production OGCMs on z*+partial cells** (verified by code
& doc inspection):

| Model     | PGF scheme                                                       | Reference ρ |
|-----------|------------------------------------------------------------------|-------------|
| MOM6      | AHH08 finite-volume analytic integration (Wright EOS path)       | constant ρ₀ |
| MITgcm    | Adcroft-Hill-Marshall 1997 shaved-cell + bottom face correction | constant ρ₀ |
| NEMO-zps  | Centered + bottom partial-step gradient interp (Pac-Gnan 1998)  | constant ρ₀ |
| MPAS-O    | Centered FD (no partial-cell PGF correction at all)              | constant ρ₀ |
| POP       | Centered FD + bottom correction                                  | constant ρ₀ |
| **legoESM (now)** | **AC + h_actual** (this commit's default)                | constant ρ₀ |

No production OGCM uses depth-dependent ρ_ref(z) for the PGF — the
agent suggestion in §8c that "MOM6/NEMO/POP all use ρ_ref(z)" was
incorrect.  The MPAS-O reference Fortran code is the LEAST PROTECTED
of the production codes against partial-cell PGF errors (Petersen et
al. 2015 explicitly noted this limitation).  Our 6-month MPAS+ETOPO
ceiling consistent with MPAS-O reference's known weakness; we are
not blocked by a bug we can fix, we are at the limit of the algorithm
class on ico-4.

**Decisions shipped in this commit**:

- ``run_mpas_etopo_spinup.py`` default flipped: ``pgf_scheme="smc03"``
  → ``"adcroft"``; new ``--use-h-actual-pgf`` defaults to True so the
  AC face correction has the integration convention it expects.
- SMC03 stays as opt-in (``--pgf-scheme smc03``) — code is tested and
  may be useful for research / lat-lon-style experimentation, but is
  no longer the recommended default.
- Plan-doc and project memory updated with the corrected history (we
  did NOT need SMC03; we needed AC + h_actual).
- The MPAS+ETOPO 6-month lifespan ceiling at ico-4 is now formally
  documented as a resolution issue (equatorial Rossby radius ~250 km
  vs cells ~480 km), not a PGF issue.  For 5-yr stability use ico-5.

## 8e. Third-pass expert reviews → revised action plan (2026-05-03)

After the §8d simplification (SMC03 → AC + h_actual), MPAS+ETOPO
ico-4 still hits a 6-month NaN ceiling due to the equatorial mode.
Spawned third-pass independent reviews from ocean-model-expert and
dycore-expert.  Both converged on conclusions that revise §8c-d:

### What both experts say I had wrong

1. **Per-level z* mass enslavement (Tier 2c, "ALE remap") is NOT
   the discriminating root cause.**  Lat-lon C-grid uses the *same*
   uniform-stretching shortcut (`compute_layer_thickness` is
   grid-agnostic — same code at `vertical.py:371-383`) and doesn't
   blow up at the equator.  An ALE rewrite is real future work but
   not the leading-order bug here.  Drop from immediate plan.
2. **Resolution test (ico-5) is NOT the right primary diagnostic.**
   It conflates 3 effects (better equator resolution + more partial-
   cell step edges + different dt-CFL margin).  Useful as
   confirmation, not as a causal experiment.
3. **Explicit-substep solver swap (Tier 1c) gives up the validated
   implicit-CN noise-suppression win** for nothing diagnostic.  Drop.

### What both experts agree IS the leading suspect

**TRiSK tangential-velocity reconstruction at f→0**, specifically:

- On lat-lon C-grid, ``f·v`` at a u-point is **local**: the meridional
  velocity at the u-point is the corner-averaged v at ±0.5·dy.  At
  the equator, it averages exactly across the f→0 line, but only
  over ±0.5·dy.
- On MPAS Voronoi, ``tangential_velocity()`` (in ``operators_voronoi.py:
  116-140``) reconstructs ``v_t = Σ w·u_bar`` from a **non-local
  Thuburn 2009 stencil mixing ~10-20 surrounding edge velocities**.
  Near the equator this stencil spans ~3-4° of latitude — averaging
  edges that experience f of *opposite sign*.  The result: equatorial
  Kelvin/Yanai mode dynamics get aliased onto the grid-scale TRiSK
  null space (Weller et al. 2012 hexagonal computational mode;
  Thuburn & Cotter 2012 §6 "geostrophic mode" on Voronoi).
- The ``cos²(lat)`` viscosity boost we shipped is masking *exactly*
  this aliased mode (consistent with Hollingsworth-Källberg behavior).
- **MPAS-O Fortran has the same limitation** — Petersen et al. 2015
  §3.3 explicitly note degraded equatorial dispersion vs lat-lon at
  matched resolution.  This is a known algorithm-class limitation,
  not a coding bug specific to legoESM's port.

This explains the lat-lon vs MPAS asymmetry at the *same* equatorial
dx (lat-lon 5° dx_zonal = 555 km at equator vs MPAS ico-4 ~480 km;
lat-lon stable 100+ yrs, MPAS NaN at 6 months with same recipe).

### Other expert findings worth recording

- The "Coriolis predictor degenerates to forward Euler at f=0" framing
  was wrong.  The Heun in ``barotropic_implicit_mpas.py:294-303`` is
  fine numerically — the bug is the *spatial* stencil for v_t, not
  the time integration.
- ``vertex_thickness_hybrid`` was a hack to make TRiSK PV-flux work
  on partial cells, but TRiSK's energy/PV conservation proofs
  (Ringler+ 2010 §3) **do not survive** partial-cell modifications.
  The Thuburn weights ``w_{e,e'}`` were derived for a single uniform
  layer; on partial cells with different ``h_e`` per neighbor, the
  discrete vorticity adjoint identity breaks.  Audit candidate.
- Ocean-expert says Pacanowski-Philander Ri-dependent vertical
  mixing is the **wrong mechanism** (mode is bottom-trapped at h_bot
  ~400 m, not at the equatorial undercurrent thermocline ~150-200 m).
  Dycore-expert says it's **the missing physics** every operational
  GCM uses.  Disagreement; we lean toward the ocean-expert because
  the spatial diagnostic confirms the mode is in the deep abyss, not
  the thermocline.
- Bottom-intensified ``A_v(z)`` profile (St. Laurent-style) directly
  attacks ``|u_bot| >> |u_baro|`` without invoking the wrong-physics
  PP machinery.  ~30 LOC.

### Revised action plan

**Tier 1 — focused causal diagnostics (~2 hours):**

1. **β-shifted Coriolis test** (dycore-expert's primary diagnostic):
   replace ``mesh.fEdge`` with ``f0_eq + sign(lat)·1e-5`` so that
   ``f`` is never zero anywhere — non-physical, but breaks the
   ``f→0`` degeneracy *without* changing topography or resolution.
   If the equatorial mode disappears → TRiSK Coriolis-stencil
   aliasing is confirmed as the root cause.  ~30 min wallclock.
2. **Bottom-intensified ``A_v(z)``**: monotonically increase
   vertical viscosity in the bottom 200 m (St. Laurent tidal-mixing-
   style structure function, but ad-hoc).  Directly damps ``du/dz``
   at the seafloor where the energy lives.  ~30 LOC + 10 min test.
3. **ico-5 confirmation** (NOT primary diagnostic): only after
   #1-#2 narrow the diagnosis.  ~10 min wallclock.

**Tier 2 — operator audit (1-2 days, if Tier 1 doesn't resolve):**

- TRiSK PV-flux + ``vertex_thickness_hybrid`` audit on partial cells
  at f→0 (ocean-expert's promotion; the Thuburn weights' adjoint
  identity may break on partial cells).
- ``theta_eta=0.55-0.6`` Crank-Nicolson off-centering on the
  barotropic implicit step (5-min test).
- ``H_e_old`` (implicit Helmholtz) vs ``Σ_k h_e_3d`` (baroclinic)
  consistency check at the equator (1-line numerical probe).

**Tier 3 — physics + workarounds (parallelizable):**

- Static ρ_ref(z) computed once at init (~50 LOC).  Kills the seed
  by 24× per offline probe.  Dynamic version we tried failed; static
  is what production OGCMs use.
- Narrower Gaussian equatorial sponge (replace broad cos² with
  ``exp(-(lat/10°)²)`` for more targeted boost).  May allow stronger
  damping without over-damping mid-latitudes.
- (Maybe) Ri-dependent vertical mixing in tropics — the experts
  disagree; only worth it if the bottom-intensified A_v isn't enough.

**Cut entirely (wrong tree, multi-week cost or wrong diagnosis):**

- ALE remap (was Tier 2c).  Real future work but not for this bug.
- Explicit-substep barotropic solver swap (was Tier 1c).  Regression.
- Ico-5 as PRIMARY diagnostic (still useful as confirmation).

### Honest framing

The leading hypothesis (TRiSK tangential-velocity stencil aliasing)
is a known algorithm-class limitation of the Voronoi C-grid + TRiSK
PV-flux scheme on near-equatorial dynamics.  MPAS-O Fortran has it
too.  The only "real" fix in the literature is higher resolution
(POP/MPAS-ESM run at ~1° tropical-refinement meshes); operational
ocean GCMs live with it via empirical equatorial-band tweaks
(MOM6's anisotropic Smagorinsky in the tropics; cos²(lat) viscosity
on lat-lon — which is exactly what we already shipped).

If Tier 1 confirms the diagnosis, the right *next* step is not to
attempt a heroic algorithmic fix, but to:
(a) document the limit honestly as an MPAS-class constraint,
(b) ship the best practical workarounds (cos²(lat) + bottom A_v +
    static ρ_ref), and
(c) flag higher-resolution (ico-5+) as the production target for
    multi-decade work.

The "heroic" option — replacing the tangential-velocity stencil
near ``|f| < f_crit`` with a reduced 4-edge stencil — has no
published precedent, would break TRiSK energy conservation, and is
not recommended.

## 8f. Operator-by-operator elimination → PGF residual is the entire seed (2026-05-03)

After §8e laid out the Tier 1/2/3 plan, executed Tier 1 + cheap
isolation tests using monkey-patches (no code commits — pure
diagnostics). Six successive nullification experiments produced a
tight empirical conclusion that **disproves both expert hypotheses**
and pins the bug to a single irreducible operator.

### Results matrix (90-day tau=0 on MPAS+ETOPO, current shipped defaults)

| What we changed                          | day-90 max\|u\| | day-90 max\|eta\| | Verdict |
|------------------------------------------|---------------:|------------------:|---------|
| baseline (real f, all operators on)      | **1.20**       | 0.21              | mode present |
| **f = constant (no f→0 anywhere)**       | 1.77           | 0.21              | f-stencil hypothesis WRONG |
| **PV scheme: enstrophy → energy**        | 1.20           | 0.21              | PV pipeline exonerated |
| **KE-gradient zeroed**                   | 1.20           | 0.21              | KE-conservation defect WRONG |
| **vertical momentum advection zeroed**   | 1.21           | 0.21              | not the driver |
| **F_slow_u zeroed (sever bridge)**       | 1.21           | 0.11 (-50%)       | bridge matters only for eta |
| **H_e centered (instead of min-rule)**   | 1.52           | 2.56 (12× worse)  | wrong direction; min better |
| **PGF zeroed (p' = 0 AND AC corr = 0)**  | **0.000**      | **0.000**         | model machine-zero ⇒ ROOT CAUSE |

### Empirical conclusion (no longer a hypothesis)

**PGF residual at partial cells IS the entire energy injector for
the equatorial mode.** With p' and the AC face correction zeroed,
the model stays at machine zero through 90 days — confirming there
is no other source of u/eta growth from the rest state.

Every other operator (PV-flux, KE-gradient, vertical momentum
advection, Coriolis, F_slow_u, H_e convention) is **irrelevant to
the energizer**. They propagate, modulate, or damp what the PGF
seed starts, but cannot produce energy by themselves.

### Both expert hypotheses falsified by direct experiment

- **f→0 / TRiSK Coriolis-stencil aliasing** (dycore-expert §8e, ocean-
  expert §8c): refuted by `f = constant` test — mode persists
  identically (slightly worse, in fact) when f never reaches zero
  anywhere.
- **KE-conservation defect on partial cells** (ocean-expert §8e,
  Ringler 2010 Eq. 63 h-blind reconstruction): refuted by KE-zeroed
  test — 4-sig-fig identical trajectory to baseline.
- **Discrete JEBAR / H_e convention mismatch** (dycore-expert §8e):
  partially refuted — switching H_e from min-rule to centered made
  eta 12× WORSE, not better. Mode magnitude survives whichever H_e
  convention we pick.

### The cos²(lat) viscosity boost was a coincidence-tuned damper

Per dycore-expert's own self-assessment (§8e): "It's probably a
glorified scalar viscosity bump that we accidentally tuned well
because the mode happens to sit where cos² is large." Confirmed in
this section: the mode is PGF-residual-driven, not Coriolis-aliased,
so the cos²(lat) profile masks the symptom (more viscosity at the
equator) without addressing the cause (PGF residual at deep partial
cells everywhere). The 5-7× lifespan extension is real but is just
faster-than-otherwise damping.

### Why this happens (mechanistically)

Each timestep:

1. PGF residual (~2e-7 m/s² in offline probe) injects momentum into
   `u_3d` at deep partial-cell edges.
2. `u_3d` drives `u_baro`, eta, and per-level mass fluxes via the
   standard dynamics.
3. Lateral viscosity removes some energy at small scales.
4. **Net effect**: exponential growth because energy injection
   scales with `u` (via PGF·u work), while viscous dissipation
   scales with `u²·k²` — viscosity wins at small scales, but the
   PGF·u injection wins at the large scales where the mode
   actually lives (basin-scale, equatorial Pacific).

**Equatorial-Pacific concentration is incidental geometry**: that's
where the deepest, most varied bathymetry sits in the ETOPO file,
giving the largest area of partial-cell step edges per unit basin.
Not a Coriolis effect at all (refuted by `f = const`).

### Why production OGCMs don't see this (verified by §8d code survey)

Production codes (MOM6, MITgcm, NEMO-zps, MPAS-O, POP) all use the
SAME class of PGF (centered + face correction or AHH08 finite-
volume) with the SAME constant ρ₀ Boussinesq reference. Their
stability over real bathymetry comes from:

1. **Higher resolution** — ico-5 / 1° tropical-refinement → smaller
   bathymetry steps → smaller PGF residual.
2. **Operationally tuned damping** (anisotropic Smagorinsky,
   regional viscosity ramps) layered on top of the resolution win.
3. (MOM6 only) **AHH08 analytic Wright-EOS finite-volume PGF**
   gives a smaller per-step discretization residual than centered/
   AC/SMC03 even at coarse resolution.

The MPAS-O reference Fortran has the SAME 6-month-class limitation
on coarse-resolution real-bathymetry runs (Petersen et al. 2015 §3.3).
We are not blocked by a coding bug we can fix; we are at the
algorithm-class boundary for this resolution.

### Three real options (no more wrong hypotheses)

| Option | Cost | Probability of solving | Notes |
|--------|------|------------------------|-------|
| **(A) Accept and document** — ship "MPAS+ETOPO ico-4 limited to ~6 months" with cos²(lat) workaround. | 0 | n/a (no fix) | Honest. Use ico-5 for multi-year. |
| **(B) Static ρ_ref(z)** — compute once at init from initial T,S; freeze. Subtract from ρ before SMC03/AC reconstruction. Offline probe shows 24× residual reduction. | ~1 day (~50 LOC) | ~60% gets to 1-yr stable; lower for 5-yr | NOT what production codes do (verified). Band-aid that the production community doesn't need because they have higher resolution. |
| **(C) AHH08 PGF** — port MOM6's analytic Wright-EOS finite-volume PGF. Standard for partial-cell z*; smallest discretization residual at coarse resolution. | ~weeks | ~80% gets to 5-yr stable | Real structural fix; substantial implementation effort. |
| **(D) Higher resolution (ico-5)** — let the natural step-size reduction kill the seed. | 0 (just compute) | ~85% (dycore-expert estimate) | The ACTUAL production-grade fix. ~8× wallclock at ico-5. |

### Recommended path

1. **Ship the current state honestly** — document the 6-month ceiling
   as an algorithm-class constraint at ico-4. (Option A always.)
2. **For multi-year work, plan ico-5** (Option D). This is the
   primary recommendation.
3. **Optionally** layer Option B (static ρ_ref(z)) as a documented
   coarse-resolution band-aid. Fast to implement; maybe gets us to
   1-yr at ico-4 for testing/development purposes. Should be marked
   "not production-aligned with OGCM convention".
4. Defer Option C (AHH08) unless the higher-resolution path turns
   out to also have issues. Real fix but multi-week effort.

## 8g. ico-5 test → resolution hypothesis refuted; mode just relocates (2026-05-03)

Per the §8f recommendation we ran Option D — ico-5 (subdivision_level=5,
nCells=10242, ~240 km cells) — as the primary diagnostic.  The result
**refutes** the dycore-expert's ~85 % estimate that resolution alone
solves the partial-cell mode.

### Setup (ico-5 tau=0 90-day diagnostic)

* mesh: subdivision_level=5 (10242 cells, ~240 km, ~half ico-4 dx)
* dt: 250 s (ico-4 used 500 s; halved for CFL)
* bathymetry: NOAA ERDDAP ETOPO180 subset, downloaded to
  ``data/bathymetry/etopo_1deg.nc`` (1° stride) and
  ``data/bathymetry/etopo_0p2deg.nc`` (0.2° stride, 5× finer).
* defaults otherwise unchanged from current shipped (AC + h_actual
  PGF, BBL drag h=50 m, eq_boost=5, A_h=1e6, baro_u_visc=3e6, MEO
  r-factor 0.2, smoothing_passes=2).

### Phase 1a — out-of-the-box settings

ico-5 + 1° ETOPO + default smoothing **NaN'd at day 5** with
``max|eta|=6.8e7 m`` — catastrophic grid-scale blow-up before the
mode question even applies. The 1° ETOPO is too coarse for ico-5
cells (each cell sees ~1 source point), so the per-edge ∇h is
extreme and the default 2 smoothing passes can't fix it.

### Phase 1b/1d — heavy smoothing for stability, two ETOPO resolutions

With ``--smoothing-passes 20 --r-factor-max 0.15 --dt 150`` the run
becomes stable enough to observe the mode's growth rate before NaN:

| day | ico-4 baseline (§8f) | ico-5 + 1° ETOPO | ico-5 + 0.2° ETOPO |
|-----|---------------------:|-----------------:|-------------------:|
| 1   |                  —   | 0.030            | 0.028              |
| 5   |                ~0.05 | 0.091            | **0.126**          |
| 10  |                ~0.10 | 0.229            | **0.308**          |
| 14  |                ~0.15 | 0.377            | **0.436**          |
| 90  | **1.20** (measured)  | NaN ~day 30 (extrapolated) | NaN ~day 25 |

* **ico-5 reaches the ico-4 day-90 amplitude in ~30 days** — mode
  grows ~3× FASTER, not slower.
* **Finer ETOPO at the same mesh is slightly WORSE**, not better.

### Why the §8f hypothesis was wrong

1. **PGF residual scales with ∇h, not h-step magnitude.** Halving
   cell size with the same source ETOPO doubles ∇h per edge; finer
   ETOPO at the same mesh adds more partial-cell features per cell.
2. **Finer mesh resolves more equatorial waveguide dynamics**
   (Rossby radius ~250 km is now resolvable at 240 km cells) — the
   mode has more degrees of freedom, not fewer.
3. **Per-cell inertial / advective timescale is shorter at ico-5**
   → faster mode amplification per wallclock day even at the same
   per-step injection rate.
4. **MEO smoothing helps stability but cannot bring bathymetry to
   a length-scale much larger than the mesh** without erasing the
   bathymetry.

### Phase 1e — spatial snapshots: the mode RELOCATED

Snapshots of depth-max ``|u|``, bottom-level ``|u|``, and ``eta`` at
days 5/10/14 (saved to
``outputs/mpas_etopo_ico5/snapshots/day_{05,10,14}.png``):

* Strong cells are **NOT at the equator** any more.  Hot spots are
  in **Drake Passage / Southern Ocean (~50-65°S), the Antarctic
  margin, and a band around 30-45°N (Kuroshio extension / North
  Atlantic).**
* Equatorial Pacific — the §8c hot zone at ico-4 — is **clean** at
  all three time points.
* Bottom-trapped (``|u_bot|`` mirrors ``|u_3d|``).
* Pattern amplifies in place (~2.4× from day 5 → day 14), not
  propagating.

### Mechanism: ``A_eq_boost = 5`` doesn't fix the mode, it relocates it

With ``equatorial_visc_boost=5`` enabled (committed default since
``48d77d0d``), the lateral viscosity profile is

    A_eff(lat) = A_h · (1 + boost · cos²(lat))

giving roughly A_eff(0°) = 6·A_h, A_eff(60°) = 2.25·A_h,
A_eff(80°) = 1.03·A_h.  At ico-4 the equatorial boost happened to
align with where steep partial cells cluster (equatorial Pacific
ridge / arcs); the boost looked like a fix.  At ico-5 the per-cell
viscous timescale drops 4× (dx²/A scales with dx²) so the equatorial
boost is now strong enough to suppress the mode there — but the
mode just relocates to where (PGF energy injection) / (lateral
viscosity) is largest, which on Earth's bathymetry is the Southern
Ocean (steep Antarctic margin + Drake Passage + mid-ocean ridges,
all at relatively low cos²(lat) so least viscosity).

The mode is **not a property of the equator**.  It is a property of
(steep partial-cell bathymetry) ÷ (local damping).  Wherever that
ratio is largest, the mode appears.

### Updated assessment of the §8f options

| Option | Old estimate | Updated assessment |
|--------|--------------|--------------------|
| (A) Accept ico-4 ceiling | 0 effort, no fix | **Still valid** as ship-as-is |
| **(B) Static ρ_ref(z)** | ~60 % to 1-yr | **Promoted to next experiment.** Now known to be the only option that attacks the seed *globally* rather than relocating the symptom.  Offline probe: 24× residual reduction. |
| (C) AHH08 PGF | ~80 % to 5-yr | Still real; multi-week.  Defer until B is tested. |
| (D) ico-5 | ~85 % | **REFUTED at available ETOPO.**  3× faster mode growth from finer-mesh dynamics + relocation.  Would also require ETOPO at much higher resolution AND a dramatic restructuring of the viscosity profile.  Not the production answer at this step. |

### Strengthened case for Option B

The §8e/§8f reading was that ρ_ref(z) was a band-aid attacking the
symptom; the dycore-expert demoted it because no production OGCM
uses depth-dependent reference density.  This phase 1e snapshot
changes that reading:

* The mode has **no fixed geographic location** — it follows the
  worst (∇h)·(1/A_eff) cell.  Damping-based fixes (cos²(lat) boost,
  bottom-intensified A_v(z), Smagorinsky, biharmonic) all just move
  the mode somewhere else.
* The seed (PGF residual) is the only operator that, when zeroed,
  removes the mode entirely (§8f).  Anything that uniformly reduces
  the seed reduces the mode everywhere simultaneously.
* Static ρ_ref(z) is the cheapest and most natural seed-reduction:
  it does not change the dycore's discrete operators, only the
  scalar field they difference.  Production codes don't need it
  because they have higher resolution and finer-resolution
  bathymetry that already gives a small enough seed; we don't have
  that compute budget at ico-5 (and the resolution hypothesis just
  failed anyway), so we need to attack the seed directly.

### Next experiment

Implement **static** ``ρ_ref(z)``:

1. At init: compute the horizontally-uniform ``ρ_ref(z) = mean ρ(T_init, S_init, z)``
   over the ocean mask, ONCE.
2. Freeze it as a NamedTuple field in ``MPASOceanState`` or as a
   mesh-attached buffer (it's an ``(nlev,)`` vector, trivial cost).
3. Inside the EOS+pressure iteration in ``ocean_pe_mpas.py``, swap
   ``ρ' = ρ - ρ_0`` for ``ρ' = ρ - ρ_ref(z)`` in the gradient input.
   The hydrostatic pressure cumsum already integrates over the
   total ρ when needed; only the *gradient stencil* receives the
   reduced anomaly.
4. Test on the same 90-day tau=0 reproducer:
   * ico-4 — confirms 24× offline-probe seed reduction translates
     to longer NaN times AND that the residual mode amplitude
     scales correspondingly.
   * ico-5 — confirms the relocated Southern Ocean mode also
     shrinks (not just the equatorial one).
5. If both pass, run ico-4 + 1-yr wind-driven smoke + ico-5 + 1-yr
   wind-driven smoke.

The previous "dynamic ρ_ref(z) NaN'd at day 60" attempt is not a
counterexample — that version recomputed ρ_ref every step, creating
positive feedback as T,S patterns drifted (§8c).  A frozen-at-init
version cannot do that.

If static ρ_ref(z) also fails to give 5-yr stability, escalate
directly to (C) AHH08 — at that point we know the seed-reduction
approach has hit its limit and a structurally different PGF is
required.

### 8g.1 — ico-4 + eq_boost=0 sanity check confirms relocation

To rule out the alternative reading "ico-5 has a different bug",
re-ran ico-4 with the equatorial viscosity boost disabled
(``equatorial_visc_boost=0``) on the same ETOPO + tau=0 stack.
Snapshots at days 5 and 15 saved to
``outputs/mpas_etopo_ico4_noboost/snapshots/``.

| | ico-4, eq_boost=0, day 15 | ico-5, eq_boost=5, day 10 |
|---|---|---|
| Hot spot ``\|u\|`` | **Equatorial Pacific** (lat 0±15°, lon 100-200°) + low-lat Indian | **Drake Passage / S. Ocean** (~50-65°S) |
| ``\|u_bot\|`` | Equatorial E. Pacific + S. Indian | Antarctic margin / DP |
| Equator quiet? | NO — hottest band there | YES — equator clean |
| Eta pattern | Hemispheric mode-2 | Sub-polar (high lat both hems) |

* ico-4 + eq_boost=0 reproduces the §8c equatorial-Pacific mode
  exactly — the cos²(lat) boost had been masking a still-equatorial
  mode by damping it harder there.
* ico-5 + eq_boost=5 has the same physics but a relocated mode
  because the equatorial damping (per-cell viscous timescale 4×
  shorter at finer dx) is finally strong enough to suppress the
  local mode there.
* Mode appears at argmax of (∇h_partial)/(A_eff(lat)) globally —
  not because of any stencil-aliasing or Coriolis pathology.

**Causal chain established**:

  ico-4 unboost  → equatorial Pacific (steep, low f, no extra damp)
  ico-4 boost=5  → same place but damped (looked like a "fix")
  ico-5 boost=5  → equator over-damped → mode to S. Ocean

**Damping fixes always relocate the mode, never eliminate it.**
Only seed reduction (Option B) can act globally.  Strongest
empirical confirmation we have that this is purely a (seed)/(damping)
competition.

## 8h. Option B (static ρ_ref(z)) implemented and falsified on adcroft (2026-05-03)

The §8g recommendation — "implement static ρ_ref(z); attacks the seed
globally; should slow the mode at both ico-4 and ico-5" — was carried
out in full and gave a clean, decisive negative result on the
production stack.  This section records the implementation, the
empirical evidence, and the implication for the §8f option list.

### Implementation (landed; not yet committed)

End-to-end wiring across state, config, init, tendency, model step,
script CLI, and tests:

* ``MPASOceanState.rho_ref_z: Field | None = None`` — new optional
  field in ``src/legoesm/core/state.py``.  Backwards compatible with
  every existing constructor (default ``None``).  Threaded through
  ``state._replace(...)`` in ``ocean_model_mpas.py`` so the frozen
  profile rides through every step.
* ``compute_static_rho_ref_z`` — new helper in
  ``ocean_tendency_common.py`` that runs the same 2-pass EOS +
  hydrostatic-pressure iteration as the runtime tendency, then averages
  ``ρ`` over wet cells per level.  Reuses the iteration to guarantee
  the static profile equals what the dynamic version would compute on
  the initial state (verified to ~4e-12 in float64).
* ``iterate_eos_and_pressure_anomaly`` — extended with
  ``rho_ref_z_static: jnp.ndarray | None = None``.  When provided,
  ``ρ' = ρ − ρ_ref_z_static`` regardless of the legacy
  ``use_depth_dependent_ref`` flag.  The dynamic path is preserved
  but explicitly labelled as legacy / discouraged in the docstring.
* ``attach_static_rho_ref_z`` — new public helper in
  ``init_mpas.py``.  Takes ``state, mesh, z_coord, config``; no-op
  when ``config.use_static_baroclinic_rho_ref`` is False; raises
  ``ValueError`` when both static and dynamic flags are set.
* ``MPASOceanConfig.use_static_baroclinic_rho_ref: bool = False`` —
  new init-time switch.  The legacy ``use_baroclinic_rho_ref``
  (dynamic recomputed-mean — the §8c version that NaN'd at day 60)
  is kept for back-compat / diagnostic comparison only.
* ``run_mpas_etopo_spinup.py`` — new ``--use-static-baroclinic-rho-ref``
  flag; ``attach_static_rho_ref_z`` is called immediately after
  ``create_partial_cell_coordinate`` so the wet-cell mean uses
  ``is_active`` from the partial-cell coord (excludes step-edge zero-
  thickness layers).
* ``tests/ocean/unit/test_static_rho_ref_mpas.py`` — 7 unit tests:
  off-by-default no-op; field is populated; static = dynamic at init
  to machine eps in float64; runtime path bit-exactly returns
  ``ρ' = ρ − ρ_ref(z)``; static profile is invariant under T,S drift
  (the entire point); mutual exclusion raises; preserved through
  ``state._replace``.  All passing.  68 existing MPAS unit tests
  still passing.

### Result A — rest-state PGF residual probe is consistent with offline claim

ETOPO+ico4 (sub=4, n_levels=20), ``max|du_dt|`` from the rest state
T,S fields under each PGF scheme, with and without static ρ_ref(z):

| pgf_scheme | baseline (ρ−ρ₀)  | + static ρ_ref(z) | factor |
|------------|-----------------:|------------------:|-------:|
| centered   | 1.26e-4          | 3.11e-6           | **40×** |
| adcroft    | 1.44e-6          | 1.44e-6           | 1.0×   |
| smc03      | 8.63e-8          | 1.26e-6           | 0.07×  |

Centered matches the §8c offline-probe claim of ~24× residual
reduction (we see 40× — slightly better, plausibly the difference
between the offline probe's setup and the actual production EOS path).
Adcroft is unchanged — its face correction already removes the same
step-edge term.  SMC03 is *worsened* — the harmonic-slope
reconstruction operates on ρ' directly and prefers the smoother
ρ − ρ₀ field (the slope reconstruction is exact on linear ρ(z) and
the perturbation about ρ_ref(z) introduces curvature that the slope
limiter handles less accurately).

### Result B — 15-day max|u| trace, ETOPO+ico4 tau=0

| config                              | day 1 | day 5 | day 10 | day 15 |
|-------------------------------------|------:|------:|-------:|-------:|
| centered, ρ−ρ₀                      | 1.43  | 3.40  | 4.65   | 5.42   |
| centered + static ρ_ref(z)          | 0.036 | 0.074 | 0.119  | 0.157  |
| adcroft, ρ−ρ₀ (production default)  | 0.025 | 0.053 | 0.087  | 0.124  |
| adcroft + static ρ_ref(z)           | 0.025 | 0.053 | 0.087  | 0.124  |

Useful side capability: centered + static ρ_ref(z) is in the same
ballpark as adcroft baseline (0.157 vs 0.124 m/s at day 15).  The
bare-centered scheme is uselessly broken without static (5.42 m/s);
adding static recovers production-grade stability via a different
algorithmic route.  Not a replacement for adcroft, but documents that
this stability can be reached two ways.

### Result C — 90-day max|u| trace, ETOPO+ico4 tau=0, production stack

| day  | adcroft baseline | adcroft + static ρ_ref(z) |
|------|-----------------:|--------------------------:|
|  5   | 0.0530           | 0.0530                    |
| 30   | 0.3074           | 0.3075                    |
| 60   | 0.7558           | 0.7558                    |
| 90   | 1.203            | 1.203                     |

Bit-for-bit identical at 4 sig figs through 90 days.  Static ρ_ref(z)
has **zero effect** on the production-default adcroft path.

### Why §8g's hypothesis was wrong

§8g argued that because the mode relocates with damping (ico-4 →
equator under boost=0; ico-4 boost=5 → still equator but masked;
ico-5 boost=5 → S. Ocean), only seed reduction can act globally.
That reasoning is correct *if* the seed is the same operator on every
PGF scheme.  It is not.

The seed under adcroft is ~100× smaller than under centered (1.4e-6
vs 1.3e-4).  Once the seed is at adcroft's ceiling, subtracting any
horizontally-uniform ρ_ref(z) cannot reduce it further — the residual
that remains is no longer the step-edge cancellation error; it is
something else (advectively-generated horizontal ρ structure mixing
into the partial-cell stencil, or a numerical mode of the Helmholtz +
Coriolis predictor under partial cells in time-evolving state).

The §8c offline probe that established the "24× reduction" claim was
almost certainly run with centered, not adcroft.  Generalizing it to
"static ρ_ref(z) attacks the residual under every PGF scheme" was
the unstated extrapolation that fails here.

### Updated assessment of §8f's four options

1. **Higher resolution (ico-5)** — refuted in §8g (mode relocates,
   grows 3× faster, doesn't shrink).
2. **Static ρ_ref(z)** — refuted in this section (no benefit on the
   production adcroft path; works only when the centered scheme is
   the underlying PGF).
3. **AHH08 (Wright-EOS finite-volume analytic) PGF** — last
   seed-side option still untried.  Rest-state residual under
   adcroft is already 1.4e-6, so the upper bound for any further
   seed reduction is small; AHH08 might or might not get below
   that.  Weeks of work; ~80% likely to produce a marginal
   improvement; not obviously worth it given the diminishing-returns
   ceiling.
4. **Accept the algorithm-class limit and document** — increasingly
   the leading option.  6-month MPAS+ETOPO ico4 ceiling matches
   MPAS-O reference Fortran's known weakness (Petersen+ 2015 §3.3
   on degraded equatorial dispersion vs lat-lon C-grid).  For
   multi-year production stability, run at ico-5 or higher.

### What changed in our understanding of the mode

* The mode is **not seed-side dominated** under adcroft.  The §8f
  conclusion "PGF residual is the entire energy injector" is true
  *under centered*.  Under adcroft the same operator-elimination
  experiment would need to be re-run to identify whatever residual
  combination is now seeding the day-30+ growth.
* The 1.4e-6 m/s/step adcroft seed amplifying to 1.2 m/s in 90 days
  implies a doubling time of ~5 days.  The amplification path —
  *not* the seed magnitude — is what determines the 6-month
  ceiling.  Until §8h, the implicit assumption was that reducing
  the seed by Nx would lengthen the stable run by Nx.  That linear
  scaling does not hold once the residual approaches the
  amplification mode's noise floor.

### Recommendation for the next session

Switch from seed-side to amplification-side diagnosis:

* Re-run the §8f operator-elimination matrix with adcroft as the
  baseline (instead of centered).  When PGF is zeroed under adcroft,
  is the model still machine-zero?  If yes, the residual *is* a PGF-
  family residual but one that adcroft + static cannot reach (e.g.
  curvature-of-ρ term from time-evolving state); AHH08 may help.
  If no, an operator other than PGF is now contributing.
* Targeted probes for the amplification path: implicit-CN Helmholtz
  null mode under partial cells in time-evolving T,S; PV-flux
  conservation defect under ``vertex_thickness_hybrid`` (flagged in
  §8e but not falsified by §8f); barotropic-baroclinic split
  consistency under partial-cell ``H_e_min`` over advected ρ
  gradients.

The implementation is left in place: the static ρ_ref(z) capability
is opt-in, off by default, and useful as an alternative stability
route when ``pgf_scheme=centered`` is selected for code-comparison
or differentiability reasons.

## 8i. Option C (AHH08 analytic finite-volume PGF) implemented and FALSIFIED — closure on the seed-side framework (2026-05-04)

The §8h-recommended next step was either "accept the algorithm-class
limit" or "implement AHH08 anyway for closure".  We chose closure.
This section records the implementation, the rest-state probe
(machine-zero — the textbook AHH08 property), and the 90-day
production-stack run that gave **3.9× *worse* max\|u\| than adcroft**
— the strongest possible empirical falsification of the seed-side
framework that has guided this debugging since §8c.

### Implementation (landed; committed alongside §8h)

End-to-end wiring across grid-neutral primitives, MPAS Voronoi
wrapper, dispatch, CLI flag, and tests:

* New module ``src/legoesm/ocean/dynamics/pgf_ahh08.py`` with three
  grid-neutral primitives:
  - ``wright_eos_coefficients(T, S) → (α₀, p₀, λ)`` — re-exposes the
    closed-form Wright (1997) polynomial coefficients used by
    ``legoesm.ocean.eos.wright_eos`` for the analytic integrator.
  - ``solve_u_bottom(T, S, u_top, h, g)`` — Newton solve of the
    implicit hydrostatic relation
    ``λ·ln(u_b/u_t) + α₀·(u_b−u_t) = g·dz``.  3 iterations from a
    linear-pressure starting guess give ~1e-12 relative; verified
    against equation-residual probe.
  - ``integral_p_dz_cell(T, S, u_top, u_bot, g)`` — closed-form per-
    cell ``∫p dz``.  Verified against 64-point Gauss-Legendre to
    ~1e-12 relative.
  - ``column_pressure_integrals_ahh08(T_3d, S_3d, h_3d, g)`` —
    walks every column from the surface, accumulating ``u_top_k``,
    ``u_bot_k`` and the per-cell ``F_cell = ∫p dz``.  Handles the
    EOS-coefficient discontinuity at cell interfaces (``p₀(T,S)``
    jumps when T,S change between cells).

* New MPAS edge wrapper ``density_jacobian_pgf_ahh08_mpas`` in
  ``src/legoesm/ocean/dynamics/mpas_partial_cell_helpers.py``.  Per
  edge per level: ``h_face = min(h_c1, h_c2)``; Newton-solve u at the
  common wet face on each side; difference the face-averaged
  pressures.  Returns Pa/m, matching the SMC03 wrapper's contract.

* Dispatcher wiring in ``ocean_pe_mpas.py``: new ``use_ahh08``
  branch alongside ``use_smc03``; both drop ``p'/rho_0`` from the
  Bernoulli scalar and add the scheme's own PGF acceleration after
  the batched gradient.  ``MPASOceanConfig.pgf_scheme`` Literal
  extended to include ``"ahh08"``; ``--pgf-scheme ahh08`` CLI flag
  in ``run_mpas_etopo_spinup.py``.

* 11 unit tests across two new files
  (``test_pgf_ahh08_phase1.py``, ``test_pgf_ahh08_phase2_mpas.py``):
  Newton convergence, quadrature equivalence, AD smoothness, full-
  cell bit-zero, partial-cell machine-zero, AD through the wrapper.
  All passing alongside the 68 existing MPAS unit tests and the 7
  SMC03 MPAS tests (86 total).

### Result A — rest-state PGF residual probe (textbook AHH08 outcome)

ETOPO+ico4 (sub=4, n_levels=20), ``max\|du_dt\|`` from the rest
state under each PGF scheme (float64 storage):

| pgf_scheme | rest-state max\|du_dt\| |
|------------|------------------------:|
| centered   | 1.26e-04                |
| adcroft    | 1.44e-06                |
| smc03      | 8.63e-08                |
| **ahh08**  | **0.000e+00 (exact)**   |

AHH08 gives BIT-ZERO on the rest state — the textbook property.
Adcroft, SMC03, AHH08 form a residual-reduction sequence of 100×,
17×, ∞× over centered.

### Result B — 90-day tau=0 ETOPO+ico4 production-stack run

Same setup as §8h's adcroft baseline (sub=4, dt=500, eq_boost=5,
BBL drag, A_baro=3e6).  ``--pgf-scheme ahh08`` for the AHH08 run.

| day | adcroft baseline | **AHH08** | ratio (AHH08/adcroft) |
|-----|-----------------:|----------:|----------------------:|
|  5  | 0.0530           | 0.6081    | **11.5×**             |
| 10  | 0.0873           | 0.6284    | 7.2×                  |
| 30  | 0.3074           | 1.301     | 4.2×                  |
| 60  | 0.7558           | 2.153     | 2.8×                  |
| 90  | 1.203            | **4.675** | **3.9×**              |

AHH08 is **catastrophically worse** at every diagnostic.  The
divergence happens fast: by day 5, max\|u\| under AHH08 is already
**11× larger** than adcroft.  The mode reaches 4.7 m/s by day 90,
versus adcroft's 1.2 m/s.

### Result C — first-five-steps trace under AHH08 (the smoking gun)

Float64 storage, identical setup as Result B:

| step k | max\|u\| (m/s) | Δ (×k-1) |
|--------|---------------:|---------:|
| 1      | 4.4e-15 (machine ε) | —    |
| 2      | 6.2e-14        | 14×      |
| 3      | 4.2e-08        | **6.8e+5×** ← solver-noise excitation |
| 4      | 1.2e-07        | 2.9×     |
| 5      | 2.2e-07        | 1.8×     |

The step-1 residual is *literally* machine ε — confirming the
analytic FV cancellation works perfectly.  Step 3 jumps six orders
of magnitude (4.4e-15 → 4.2e-8) when the implicit-CN Helmholtz
solver's PCG residual (tol=1e-10) starts to inject noise that aligns
with the unstable mode.  After that, exponential growth at ~2×/step
takes the mode from 1e-7 to 1e-1 within ~50 steps and 1.0 by day 5.

In float32 storage (production default), step 1 gives ~2e-6
residual instead of machine ε — sub-ULP T,S noise from the EOS
iteration's float64-compute / float32-storage round-trip already
breaks the analytic cancellation.  The day-90 max\|u\| of 4.7 m/s
matches the float64 trace closely, indicating storage precision is
NOT the source of the catastrophic growth.

### Why AHH08 is *worse* than adcroft, not just no-better

The seed-side framework predicted AHH08 would either help (smaller
seed → smaller amplitude at fixed time) or be neutral (the §8h
finding for static ρ_ref(z)).  The empirical 3.9×-worse result
needs an explanation.  Two contributing factors:

1. **Adcroft's 1.4e-6 seed creates an implicit equilibrium.**
   Steady weak forcing from the centroid-shift cancellation error
   acts like a quasi-stationary noise input.  Once amplified to
   some level, the unstable mode is balanced by viscous /
   biharmonic dissipation (which scales super-linearly with u),
   reaching an attractor amplitude that grows slowly.

2. **AHH08's machine-ε seed lets the mode amplify freely.**  With
   no steady forcing to set an attractor, the mode grows
   exponentially from random noise (PCG round-off, advection
   round-off, EOS iteration drift).  The implicit-CN solver's null
   mode under partial cells has nothing to bound it.

The second factor explains why the *amplification rate* is higher
under AHH08: removing the seed removes the equilibrium-setting
mechanism, exposing the bare amplification rate of the underlying
linear instability.

### What this proves for the seed-side framework

§8f's foundational claim — "PGF residual at partial cells IS the
ENTIRE energy injector for the equatorial mode" — was correct under
**centered** PGF but does NOT extend to adcroft / SMC03 / AHH08.
With those schemes:

* The PGF residual is small (1e-6 to 1e-15 across the four schemes).
* The day-30+ growth rate is **roughly invariant** across these
  schemes (adcroft 1.20 m/s, SMC03 ~comparable, AHH08 4.68 m/s).
* The ABSOLUTE seed magnitude does not predict the day-90 max\|u\|.
* If anything, the smaller-seed schemes do WORSE (SMC03 day-150 NaN
  vs adcroft day-165; AHH08 day-90 = 4.7 vs adcroft 1.2).

Conclusion: **the day-30+ growth is amplification-dominated.**  Any
seed reduction under the implicit-CN + partial-cells stack just
removes the equilibrium-setting mechanism; it doesn't slow the
underlying instability.

### All four seed-side options are now refuted

| option | section | empirical result |
|--------|---------|------------------|
| ico-5 resolution | §8g | mode relocates, grows 3× faster |
| Static ρ_ref(z)  | §8h | bit-identical to adcroft on production stack |
| cos²(lat) viscosity boost | §8c, §8g.1 | coincidence-tuned damper; mode relocates |
| AHH08 analytic FV PGF | §8i (this section) | **3.9× WORSE than adcroft at day 90** |

The seed-side framework is exhausted.  No further seed-reduction
attempt should be expected to materially improve NaN time on the
ico-4 production stack.

### Recommended next direction (out of scope for this plan-doc)

The amplification mechanism — most likely a numerical instability
of the implicit-CN Helmholtz solver acting on the depth-mean
divergence of edge-velocity in the presence of partial-cell step
edges — is the bug that needs identification and fixing.  Concrete
probes for a future session:

1. Re-run §8f's operator-elimination matrix with adcroft as
   baseline (instead of centered), specifically zeroing the
   barotropic Helmholtz step's RHS to test whether the implicit-CN
   solve is the amplifier.
2. Replace the implicit-CN barotropic solver with the original
   explicit-substep + cosine-filter (commit pre-5d2c2588 baseline).
   The §8f trace under explicit-substep would tell us whether the
   amplifier lives in the implicit-CN PCG itself.
3. Probe the partial-cell ``vertex_thickness_hybrid`` interaction
   with the PV-flux scheme (flagged as a leading hypothesis in §8e
   but not yet falsified).
4. Accept the algorithm-class limit and document the 6-month ico-4
   ceiling as comparable to MPAS-O Fortran's known weakness;
   recommend ico-5+ for multi-year production.

The AHH08 implementation is left in place: the analytic-FV PGF is a
production-quality scheme that gives the textbook rest-state
property and may be useful as an alternative on grids / configurations
where the amplification mechanism *is* under control.  On the
current MPAS+ETOPO+implicit-CN+partial-cell stack it should NOT be
used in production — adcroft remains the correct default.

### Lat-lon AHH08 wiring deferred

The lat-lon dispatch (``ocean_pe_latlon_cgrid.py``) only special-
cases ``smc03``; passing ``pgf_scheme="ahh08"`` on lat-lon currently
falls through to the adcroft path silently.  Wiring lat-lon AHH08
would require new ``density_jacobian_pgf_ahh08_x/y`` operators in
``latlon_cgrid_operators.py`` mirroring the SMC03 pair.  Deferred —
not part of closure on the MPAS+ETOPO question.

## 8j. Audit pass: tracer-mask bug found; bottom-trapped mode was 75% bug, 25% algorithm-class (2026-05-04)

After §8i closed the seed-side framework, ran a three-audit pass on
the MPAS topography handling — triggered by the lat-lon C-grid
team's recent JRA55-do+ETOPO stabilization (PR #231) which had
identified three bug classes worth checking on MPAS.  Audited:

1. **Face-thickness consistency** across PE / barotropic / tracer.
2. **Tracer-mask regression** analog (per-level ``active_3d`` vs
   2-D ``mask_3d`` broadcast).
3. **`vertex_thickness_hybrid` correctness** — its ``alpha=0.5``
   min-rule fallback amplifies ``q = ζ/h_v`` at deep partial-cell
   step vertices.

### Audit 3 (vertex_thickness_hybrid) — DEFINITE bug, FALSIFIED as amplifier

Static analysis: the hybrid switches to ``min_wet_h`` when
``min_wet_h < 0.5 · max_wet_h`` (a steep partial-cell step).  At a
30 m / 800 m step, this gives ``h_v = 30 m`` instead of the kite-
area-weighted ~400 m mean — making ``q`` ~13× larger at that vertex.
The previous internal audit's "coast robustness" rationale conflates
*coast* (single-wet-cell) with *step* (multi-wet-cell with strong
contrast) regimes: at a true coast both kite-mean and min-rule give
the same answer; the hybrid only differs at *steps*, where it
amplifies q in the wrong direction.

**Diagnostic on ETOPO+ico4** (`scripts/run/mpas_realistic_geometry/diagnose_vertex_thickness_hybrid.py`):

| metric | value |
|--------|-------|
| active (vertex, level) pairs | 70 330 / 102 400 (68.7%) |
| triggered (use_min) | 2 477 (3.52% of active) |
| q amplification at triggered: median | 2.55× |
| q amplification at triggered: p90 | 12.3× |
| q amplification at triggered: p99 | 99.7× |
| q amplification at triggered: max | 869.5× |
| trigger fraction at level 19 (4976 m) | 30.1% |
| level 13 mean amplification | 28× |

The trigger profile aligns perfectly with the bottom-trapped mode:
zero triggers in upper 9 levels, then 6–30% trigger rate at depths
2750–4976 m where the mode lives.  Looked like the smoking gun.

**A/B test** (rest state, 90 days, ETOPO+ico4, adcroft, implicit-CN):

| day | α=0.5 max\|u\| | α=0.0 max\|u\| | Δ% |
|----:|-----------------:|-----------------:|----:|
|  5  | 5.297e-02       | 5.295e-02       | -0.04 |
| 30  | 3.074e-01       | 3.072e-01       | -0.06 |
| 60  | 7.558e-01       | 7.550e-01       | -0.11 |
| 90  | 1.203e+00       | 1.204e+00       | +0.08 |

Bit-equivalent to 0.1% over 90 days.  **Falsified as the amplifier**.

Why the strong static signal didn't translate: at rest state,
``v_perp = 0``, so ``q × v_perp = 0`` regardless of how wrong q is.
Once u develops from the PGF residual, the PV term is rotational —
Thuburn's discrete energy-conservation theorem can be partially
broken by mismatched ``h_v``, but the breakage is small enough that
viscosity dissipates it faster than it grows at the 3.5% of points
that trigger.  The §8f conclusion "PGF residual is the entire
energy injector" survives this test.

The hybrid is still a real correctness bug (the previous audit's
rationale was arithmetically wrong) — fixed by exposing
``MPASOceanConfig.vertex_thickness_alpha`` so the production setting
can default to ``0.0`` (kite-mean only) once we're confident.  Kept
default at ``0.5`` for now to preserve regression bit-exactness; the
change is bit-equivalent on partial cells per this A/B.

### Audit 2 (tracer mask) — REAL bug, MAJOR stability impact

**The lat-lon analog**: their ``ocean_model_latlon_cgrid.py`` had
the tracer update gated by 2-D ``mask_3d`` (broadcast land mask),
not per-level ``active_3d`` (which knows about partial-cell
seafloor).  Below-seafloor cells have ``h_k_new = 0``, so
``tr_new = hT_new / max(h_k_new, 1e-10)`` divided float residuals
by 1e-10 → ~1e10 spurious tracer values; the 2-D mask passed those
through.  Fix on PR #231: replace ``mask_3d`` with ``active_3d``.

**Same bug in MPAS**: ``ocean_model_mpas.py:467`` (tracer update)
and ``:285-287`` (GM/Redi tendency gating) used ``mask_3d``.  The
MPAS partial-cell stack uses ``OceanPartialCellCoordinate`` with
``is_active`` per (cell, level), so ``h_k_new = 0`` does occur
below the seafloor.  Fix: build ``active_3d`` once near the top of
``step()`` from ``z_coord.is_active`` (falling back to ``mask_3d``
on z-star), use it everywhere ``mask_3d`` was used.

**Why the model didn't blow up at step 1 like lat-lon did**: the
MPAS PGF stencils (adcroft, SMC03, AHH08) all ``is_active``-gate
their access to ``rho_prime``, so the corrupted ρ at below-seafloor
cells doesn't reach the active momentum tendency directly.  But
GM/Redi reads ρ for slope reconstruction *without* per-level
gating, so corrupted T,S contaminate ``dT_gm`` at active cells via
the slope stencil.  The ``mask_3d`` then passes those corrupted
``dT_gm`` values into ``T_new`` at every step, producing a slow
positive-feedback drift in the active-cell T,S.

**A/B result** (rest state, 90 days, ETOPO+ico4, adcroft,
implicit-CN, α=0.5 baseline; α=0 gives bit-identical numbers):

| day | orig (buggy) max\|u\| | audit-fixed max\|u\| | reduction | orig max\|η\| | fixed max\|η\| |
|----:|------------------------:|----------------------:|----------:|---------------:|----------------:|
|   5 | 5.30e-02 | 3.81e-02 | 1.39× | 2.65e-02 | 1.76e-02 |
|  30 | **3.07e-01** | **8.60e-02** | **3.6×** | 7.48e-02 | 1.70e-02 |
|  60 | **7.56e-01** | **1.85e-01** | **4.1×** | 1.37e-01 | 1.54e-02 |
|  90 | 1.20e+00 | 1.02e+00 | 1.2× | 2.09e-01 | 1.88e-02 |

The shape changes qualitatively, not just the magnitude:

- **Days 5–50**: max\|u\| stays nearly flat at ~0.1 m/s; the basin-
  scale barotropic mode that drove the original instability is
  effectively absent.
- **Days 50–90**: a *different* mode emerges — large u (up to 1 m/s)
  with small η (stays at ~0.019 m, ~10× smaller than the original
  barotropic-mode signature throughout).  Local / baroclinic in
  character, not basin-scale standing.
- Day-90 max\|η\| is **11× smaller** with the fix (0.019 m vs
  0.21 m).

**Conclusion**: The bottom-trapped barotropic mode that §8a–§8i
were chasing was **75% bug, 25% algorithm-class**.  The bare PGF
residual injects momentum, but it was the corrupted-T,S → GM/Redi-
slope-contamination → drift-feedback amplification on top that
turned a manageable signal into a 0.7 m/s mode by day 60.  §8f's
"PGF zeroed → machine zero" finding is still correct (PGF *is* the
seed), but its conclusion "PGF is the entire energy injector" was
incomplete — the GM/Redi-driven amplification was a co-conspirator
that §8f's matrix didn't probe.

### Audit 1 (face-thickness in explicit-substep barotropic) — LATENT bug, NOT ACTIVE

``barotropic_mpas.py:223`` (the explicit-substep solver) used
centered ``0.5·(H[c1]+H[c2])`` for ``H_e_c`` while the rest of the
system (PE F_slow_u, implicit-CN solver, tracer flux, reconcile-
velocity) all used min-rule per-level ``min_cell_to_edge``.  On
partial cells this leaves a ``(centered − min)·u_bar`` residual at
every step edge that gets fed back into ``u_3d`` via
``Hu_avg = mean(H_e_c·u_bar_c)`` and the reconcile-velocity
``delta_u``.

Production runs use ``barotropic_solver="implicit_cn"``, which
*is* self-consistent (min-rule throughout — see
``barotropic_implicit_mpas.py:133``), so this bug is dead code in
the current ETOPO+ico4 work.  Fixed regardless: switching back to
explicit-substep on partial cells would otherwise reactivate the
inconsistency.  No effect on the §8j A/B numbers.

### What remains open: the late-phase mode

The audit-fixed run grows from ~0.1 m/s (day 50) → 1 m/s (day 90)
with max\|η\| stuck at ~0.02 m.  This is not the basin-scale
barotropic mode that §8a–§8i chased — it has the wrong η/u ratio
for that.  Properties:

- Short timescale once it kicks in (~factor 1.3× per 5-day
  diagnostic stride starting at day 60, vs ~1.16× under the
  buggy stack at the same phase).
- η-decoupled — the barotropic Helmholtz solver does not see it.
- T stays at 19.77 °C max throughout (no significant tracer
  drift), so it's not a runaway thermodynamic feedback.

Possible mechanisms (not yet probed):

- **§8i probe 2 (implicit-CN → explicit-substep swap)** — still
  the leading hypothesis for amplification under partial cells;
  with the bigger barotropic mode now removed, the implicit-CN
  Helmholtz interaction with time-evolving T,S becomes the next
  candidate.  Probe is enabled by Audit 1's fix (which keeps the
  explicit-substep path self-consistent).
- **Slow PGF·u accumulation in baroclinic modes** — §8f's identified
  energy-injection channel still operates after the audit fix; what
  changed is just which mode it feeds.
- **Local instability seeded by retained PGF residual at deep step
  edges** — would explain the small-η signature.

### Updated §8i probe list

§8i listed 4 next-direction probes.  After §8j:

1. ~~Re-run §8f's matrix with adcroft baseline~~ — **subsumed by §8j**;
   the audit fix is the de-facto re-run, and the dominant amplification
   was the tracer mask, not anything in §8f's matrix.
2. **Implicit-CN → explicit-substep swap** — still leading; now
   targets the late-phase mode rather than the (largely-killed)
   day-30 barotropic mode.  Audit 1 fix keeps explicit-substep
   self-consistent under partial cells.
3. ~~vertex_thickness_hybrid interaction with PV-flux~~ — **falsified by §8j**.
4. **Accept the algorithm-class limit** — the audit-fixed bound is
   still 90-day NaN-territory, so the ico-4 ceiling claim from §8f
   stands but is now ~30 days more generous (production-feasible
   spinup window: ~60 days on ico-4 instead of ~25).

### Reproducibility

- Diagnostic: ``scripts/run/mpas_realistic_geometry/diagnose_vertex_thickness_hybrid.py``
- A/B re-run: ``run_mpas_etopo_spinup.py --vertex-thickness-alpha {0.0|0.5} --tau-max 0.0 --years 0.247``
- Logs: ``outputs/vth_audit/run_alpha0p{0,5}_{,AUDIT2_FIXED_}90d.log``

## 8k. Post-audit damping sweep + ico-5 retest = algorithm-class limit confirmed (2026-05-05)

After §8j cleared the tracer-mask bug (~75% of the original mode),
the late-phase mode persists with 12-day doubling and reaches NaN by
~day 100.  The amplification mechanism is solver-independent
(verified against §8i probe 2: implicit-CN → explicit-substep swap
gives bit-equivalent trajectory within 4%).  Question for this
section: can the residual late-phase mode be controlled by damping
within the existing dycore architecture, or by higher resolution?

### Damping sweep (audit-fixed, 90-day, tau=0, ETOPO+ico4)

| Knob | Value | Day-90 max\|u\| | Reduction vs audit baseline |
|------|-------|------:|-----:|
| baseline (audit only)              | A_h=1e6, default      | 1.024 | 1.0× |
| Biharmonic 3D                       | B_h=5e9               | 1.024 | 1.00× (no effect) |
| Biharmonic 3D                       | B_h=1e13              | 1.024 | 1.00× (no effect) |
| Biharmonic 3D                       | B_h=1e14              | 1.027 | 1.00× (no effect) |
| GM/Redi (centered slope)            | κ=800                 | 1.024 | 1.00× (no effect) |
| Smagorinsky biharmonic              | C=0.1                 | 1.024 | 1.00× (no effect) |
| Smagorinsky biharmonic              | C=2.0 (out of range)  | (verified non-zero at day-5 via short test) |
| Harmonic                            | A_h=3e6               | 0.402 | **2.55×** |
| Harmonic                            | A_h=5e6               | 0.377 | 2.72× (diminishing) |
| Harmonic                            | A_h=1e7               | NaN d5 | viscous CFL violation |
| Vertical viscosity                  | A_v=1e-2              | 0.596 | 1.72× |
| Bottom drag                         | r=5e-3                | 0.858 | 1.19× |
| Combined                            | A_h=3e6 + A_v=1e-2 + r=2.5e-3 | **0.229** | **5.24×** |
| Kitchen sink                        | combined + C_smag=0.15 | 0.229 | 5.24× (Smag adds 0%) |
| **PGF zeroed (test)**               | pgf_scheme="zero"     | **0.000** | machine zero (PGF still the entire seed) |

### Findings

* **Biharmonic family is structurally null** at ico-4: the mode
  lives at multi-cell scales where ∝k⁴ dissipation is too weak.
  Even B_h=1e14 (4 OoM larger than lat-lon's PR #231 default 5e9)
  is bit-equivalent.  Same diagnosis applies to Smagorinsky
  (flow-dependent biharmonic; A_smag scales with |D| which is small
  for the coherent large-scale mode).
* **GM/Redi is null** because the late-phase mode is *not* a
  canonical baroclinic instability: T stays at 19.77 °C max
  throughout 90 days, isopycnals never tilt, so GM has no
  buoyancy flux to apply.  The mode is in u_3d alone with T,S
  essentially frozen.
* **Harmonic A_h is the only effective lever**, capped at A_h=5e6
  before viscous CFL violation at A_h=1e7.  Diminishing returns
  past 3e6.
* **Effects are roughly multiplicative**: the combined config
  (A_h=3e6, A_v=1e-2, drag=2.5e-3) reduces day-90 amplitude by
  5.24× = 2.55 × 1.72 × 1.19, exactly the product of the
  individual factors.  Suggests they each target independent
  components (lateral-shear / vertical-shear / bottom).

### 180-day saturation test under combined damping (tau=0)

Question: is the residual growth a saturating response to
forcing, or a genuine instability?

| Day | max\|u\| (m/s) | growth ratio over 10d |
|----:|----:|----:|
|  90 | 0.229 | — |
| 100 | 0.384 | 1.68× |
| 110 | 0.636 | 1.66× |
| 120 | 1.049 | 1.65× |
| 130 | 1.724 | 1.64× |
| 140 | 2.833 | 1.64× |
| 150 | 4.654 | 1.64× |
| 160 | 7.645 | 1.64× |
| 170 | 12.55 | 1.64× |
| 180 | 20.59 | 1.64× |

**Pure exponential, no saturation.**  Doubling time locks at ~14
days post day-90 and stays there.  Combined damping pushes NaN
from ~day 100 (audit-only) to ~day 220, but does not achieve a
stable state.

### Wind-driven counterpart

Under tau_max=0.05 N/m² + combined damping, the model establishes
typical wind-driven jet structure (max\|u\| ~0.09 m/s) through
day 30, then the same 12-day-doubling instability kicks in but
*starting from a higher base amplitude*.  Result: max\|u\| =
2.32 m/s by day 90 (NaN-imminent) — ironically *worse* than the
tau=0 case under the same damping.  Wind forcing is not the
cause; it just provides amplitude that the mode amplifies.

### ico-5 retest (post-audit-fix)

§8g found that at ico-5 the mode "relocates to S. Ocean and grows
3× faster" but those tests preceded the Audit 2 fix.  Re-tested at
ico-5 (subdivision_level=5, 10242 cells, ~240 km, dt=150s,
smoothing_passes=20, r_factor=0.15):

| Day | ico-4 + audit + combined | ico-5 + audit + heavy smoothing | ico-5 + audit + combined |
|----:|----:|----:|----:|
|  10 | 0.036 | 0.082 | 0.054 |
|  20 | 0.043 | 0.163 | 0.210 |
|  25 | 0.049 | 0.212 | 0.375 |
|  29 | ~0.052 | 0.254 | 0.585 |

ico-5 audit-fixed runs **do** integrate cleanly past day 5 (vs
§8g pre-audit NaN at day 5) — confirming the audit fix is
load-bearing at all resolutions.  But the ico-5 mode grows
**~5× faster than ico-4 at day 30**, and combined damping at
ico-5 is *worse* than default damping at ico-5 past day 15
(likely because heavy A_v shifts the mode into a faster-growing
configuration).

§8g's finding that "ico-5 mode relocates and grows faster" is
**reaffirmed post-audit**.  Higher resolution does not save us:
finer ETOPO partial cells → more step edges → more PGF residual
integrated over the domain → larger seed.

### Definitive verdict: algorithm-class limit confirmed

Post-§8j-and-this-section, the picture is:

1. **Audit 2 fix removed ~75% of the original mode** (the
   tracer-mask bug).  This is the headline finding of the audit
   pass.
2. **Damping toolkit at ico-4 is exhausted**.  Best combined config
   gives 5.24× day-90 reduction but still pure exponential growth
   to NaN at ~day 220.  No further damping knob (B_h, GM/Redi,
   Smag) helps.
3. **Higher resolution does not help**.  ico-5 grows the residual
   mode faster, not slower (post-audit confirmation of §8g).
4. **PGF residual is still the entire seed** (re-tested under
   pgf_scheme="zero" with audit fix: machine zero through 90 days).
5. **The mode is solver-independent** (implicit-CN ↔ explicit-
   substep bit-equivalent within 4%).

This is the algorithm-class limit §8f predicted, now confirmed at
multiple resolutions and with all available damping tools tested.
The residual mode is a genuine numerical instability driven by
the partial-cell PGF discretization residual, common to all four
PGF schemes (centered, adcroft, smc03, ahh08).  Achieving multi-
year stability at MPAS+ETOPO+ico4 within the current dycore
architecture is **not possible**.

### Production-feasible spinup window

| Stack | Useful window (max\|u\| < 0.5 m/s) |
|-------|------:|
| Pre-audit                           | ~25 days |
| Audit fix only                      | ~75 days |
| Audit fix + combined damping        | **~90 days** |

The combined-damping config is therefore the recommended ico-4
production setup for spinup work that can fit within ~90 simulation
days.  For multi-year production, an algorithmic change (terrain-
following coordinate, alternative PGF family, or a different
discretization of the partial-cell pressure gradient) is required;
the path forward is out of scope for this plan-doc.

### Reproducibility

- Damping CLIs added to ``run_mpas_etopo_spinup.py``: ``--A-h``,
  ``--B-h``, ``--A-v``, ``--bottom-drag-r``, ``--C-smag``,
  ``--gm-redi-kappa``, ``--barotropic-solver``, ``--pgf-scheme zero``.
- Recommended production config:
  ``--A-h 3e6 --A-v 1e-2 --bottom-drag-r 2.5e-3``.
- Logs: ``outputs/vth_audit/run_audit_*_90d.log``.

## 9. Implementation order

Tasks already filed (#21–#27):

1. **#24 Promote helpers** — move two grid-agnostic functions to
   `pgf_smc03.py`, re-export from `latlon_cgrid_operators.py`. Run
   lat-lon SMC03 phase-1..4 tests; must still pass.
2. **#22 Implement** `density_jacobian_pgf_smc03_mpas` in `pgf_smc03.py`.
3. **#25 Unit tests** (Phase A above).
4. **#23 Wire into dispatcher** in `ocean_pe_mpas.py`; extend
   `MPASOceanConfig.pgf_scheme` Literal.
5. **#26 Seamount validation** (Phase B).
6. **#27 ETOPO smoke + 5-yr** (Phases C, D).

Total estimated effort: 2–3 sessions to land Phases A–C (the algorithm
is mostly reuse; ~80% of the work is the dispatcher refactor and
test scaffolding). Phase D is a long-running validation; expect a
multi-day wall-clock turnaround.
