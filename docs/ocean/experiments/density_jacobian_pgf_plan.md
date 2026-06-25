# Density-Jacobian PGF Implementation Plan

**Goal**: Close the Beckmann-Haidvogel 5 mm/s gap on partial-cell
seamount tests by replacing the leading-order Adcroft & Campin (2004)
single-level face PGF correction with a Shchepetkin & McWilliams 2003
("S&M03") density-Jacobian formulation that produces a smooth-in-z
pressure gradient.

**Branch**: start a fresh branch off `ocean-partial-cells` head
(commit `8499505c`).  Suggested name: `ocean-pgf-smc03`.

**Reference commit** for context-loading: read
`docs/ocean/experiments/partial_cells_results.md` first — it
documents the empirical work that ruled out simpler alternatives.

---

## 1. Background — what we know empirically

After the existing Phase 7 + P0/P1 + #9/#10 fixes (commits `acac0622`
through `7c03ff91`), the BH seamount stress test (smoothing=5,
r_max=0.54, 30-day integration with implicit-CN barotropic and
linear bottom drag r=1e-3 m/s) gives:

| coord    | |u|max (mm/s) at day 30 |
|----------|------------------------|
| z*       | 35                     |
| partial  | 99                     |
| target   | < 5                    |

The partial-cells residual is a **2Δz vertical computational mode**
in u(z) at columns over the seamount slope, with amplitude growing
toward the partial seafloor:

```
lev 14: u =   62 mm/s
lev 15: u =    2 mm/s
lev 16: u =  −14 mm/s
lev 17: u =    2 mm/s
lev 18: u =  −99 mm/s   ← |u|max
lev 19: u =    0 mm/s   (drag sink, partial bottom)
```

Z* on the same setup gives a smooth surface-trapped profile (no
oscillation).  The mode is **forced** by the Adcroft correction's
single-level z-spike at the partial-cell level: the per-face PGF
correction is non-zero only at the level where the partial-vs-full
mismatch lives, and the resulting acceleration at that one level
sets up an equilibrium 2Δz oscillation as vertical viscosity and
barotropic continuity compete to redistribute it.

**Empirically ruled out** as fixes (see `partial_cells_results.md`
for full data):

1. **Advection scheme** (vector-invariant, WENO5, upwind, TVD): mode
   is independent of advection (<4% variation).
2. **Vertical Laplacian viscosity** A_v: 100× increase only drops
   |u|max by ~18%.  Mode is actively forced faster than Laplacian
   damps.
3. **Vertical biharmonic viscosity** B_v (∂⁴u/∂z⁴): CFL violated at
   the partial seafloor for any meaningful B_v in the
   forward-Euler path.  Could be salvaged with implicit
   biharmonic solver — separate work, doesn't fix the root cause.
4. **Piecewise-constant or piecewise-linear simplified
   density-Jacobian**: rest-state PGF residual is **88×
   WORSE** than Adcroft because the centered-FD ``dρ/dz`` estimate
   uses each column's local neighbours, which differ in z due to
   centroid mismatch → adjacent columns disagree on ρ at
   intermediate depths → spurious PGF.  Reverted; framework was a
   good start but needs proper higher-order ρ(z) reconstruction.
5. **Strong bottom drag**: r=1e−2 m/s plateaus at 27 mm/s; r ≥
   5e−2 m/s NaNs (CFL).  The plateau confirms the 2Δz mode at
   adjacent levels (where drag doesn't act) holds the residual.
6. **z-smoothing the Adcroft correction**: makes things 25–200×
   *worse* — the per-face correction *should* have step-structure
   because the geometric pressure mismatch physically lives at the
   partial-cell level; smoothing redistributes the correction to
   neighbouring levels where the mismatch is zero.

---

## 2. Mathematical formulation

The residual problem with all simpler approaches is that adjacent
columns' ``ρ(z)`` reconstructions don't agree on ρ at intermediate
depths.  For PGF at a partial-cell face to be zero in a stratified
rest state, both columns' reconstructed ρ(z) must give the same
value at any z where both are evaluated.

This requires a reconstruction whose ``dρ/dz`` is **locally
determined by the smoothness of ρ across multiple cells in z**, not
by a 2-cell finite difference using the column's own centroid
spacing.

### 2.1 Reconstruction: harmonic-limited piecewise-linear (S&M03 §4)

Within each cell ``k`` of a column, reconstruct ``ρ(z)`` as a linear
function:

  ``ρ(z) = ρ_k + σ_k · (z − z_centroid_k)``

where ``σ_k`` is the **harmonic-mean of one-sided slopes**:

```
  Δρ_top_k = (ρ_{k-1} − ρ_k) / (z_centroid_{k-1} − z_centroid_k)
  Δρ_bot_k = (ρ_k − ρ_{k+1}) / (z_centroid_k − z_centroid_{k+1})

  σ_k = 2 · Δρ_top_k · Δρ_bot_k / (Δρ_top_k + Δρ_bot_k + ε)   (harmonic)
       if sign(Δρ_top) == sign(Δρ_bot), else 0                (monotonized)
```

The harmonic mean has two key properties:

1. For **linear ρ(z)**, ``Δρ_top = Δρ_bot = a`` (the slope) →
   ``σ_k = a`` exactly in every column, regardless of where the
   centroids are.  This is what makes adjacent columns agree on
   ρ at intermediate depths.
2. For **monotone non-linear ρ(z)** (e.g. exponential T(z)),
   harmonic mean is bounded by the smaller of the two one-sided
   slopes — preserves monotonicity, no overshoots.
3. At local extrema (slopes change sign): ``σ_k = 0`` — flat-top
   reconstruction, prevents Gibbs ringing.

**Boundary handling**:
- **Top cell (k=0)**: no cell above, use ``Δρ_bot_0`` only (one-sided).
- **Bottom active cell (k=bottom_level)**: no cell below the partial
  seafloor.  Use ``Δρ_top_{bot}`` only.  Inactive cells (h=0) below
  this contribute nothing to the integral, so their slope is moot.
- **Cells adjacent to land**: when neighbour is land, use one-sided
  from the wet side.

### 2.2 Per-column pressure profile

Pressure at any depth z within cell k:

```
  P(z) = P_top_k + g · ∫_{z_top_k}^z ρ(z') dz'
       = P_top_k + g · [ρ_k · (z_top_k − z) + 0.5 · σ_k · ((z_top_k − z_centroid)² − (z − z_centroid)²)]
       = P_top_k + g · (z_top_k − z) · [ρ_k + 0.5 · σ_k · (z_top_k + z − 2 · z_centroid_k)]
       = P_top_k + g · (z_top_k − z) · ρ_at_midpoint(z_top_k, z)
```

where ``P_top_k`` is the pressure at the top interface of cell k,
computed by accumulation:

```
  P_top_0 = 0
  P_top_{k+1} = P_top_k + g · h_partial_k · ρ_k    (cell-mean integral)
```

(Note: cell-mean integral of piecewise-linear ρ over the full cell
equals ``ρ_k · h_k`` because the linear deviation integrates to
zero by definition.)

### 2.3 Face-reference depth

At face level ``k_f`` between adjacent columns W and E, evaluate
``P_W(z_face_k_f)`` and ``P_E(z_face_k_f)``.  Choice of
``z_face_k_f``:

**Option A**: ``z_face_k_f = z_full_ref[k_f]`` (smooth in k_f, same
across all faces at that level).  Backwards-compat: for full cells,
``z_full_ref[k_f] = z_centroid_k_f``, ``σ_k_f`` reduces to the
standard slope, and the integral equals the standard cumsum p_prime
at the centroid.  Bit-exact backwards-compat preserved.

**Option B**: ``z_face_k_f = 0.5 · (z_centroid_E_k_f + z_centroid_W_k_f)``
(per-face mean, varies across columns at the same level).  Adcroft-
like smoothness within each face but column-pair-dependent.

**Recommendation**: Start with Option A.  It is simpler, naturally
smooth in k_f, and the harmonic-mean slope already provides the
needed cross-column consistency for the PGF to vanish at rest.
Only fall back to Option B if Option A leaves residual at
``r_max ≥ 0.5``.

### 2.4 Horizontal Jacobian

```
  ∂P/∂x|_face_kf = (P_E(z_face_k_f) − P_W(z_face_k_f)) / dx_u_face
```

For v-faces: same formula with N/S columns and dy_v.

### 2.5 Why this fixes the 2Δz mode

The PGF correction at face level k_f depends only on:
- ``z_face_k_f`` (smooth in k_f)
- ``ρ_at_midpoint(z_top_k, z_face_k_f)`` in each adjacent column

Both vary smoothly with k_f.  No single-level spike → no 2Δz mode
forcing.  The horizontal-Jacobian residual scales with the
**second derivative** of ρ across cells (the "leftover" curvature
beyond piecewise-linear), not with the centroid offset.

For BH-test stratification (linear T(z), thus linear ρ(z) under
linear EOS), ``σ_k = a`` exactly → reconstruction is exact →
**rest-state PGF is machine-zero** at all faces.

---

## 3. Implementation plan — phased

Each phase produces a self-contained commit with passing tests
before the next phase begins.  Total estimated effort: **2-3 days**
of focused work.

### Phase 1: Harmonic slope reconstruction operator

**Deliverable**: ``reconstruct_harmonic_slopes`` in
``src/legoesm/ocean/dynamics/latlon_cgrid_operators.py``.

Signature:
```python
def reconstruct_harmonic_slopes(
    rho_per_cell: jnp.ndarray,         # (..., nlev)
    z_centroid: jnp.ndarray,           # (..., nlev)
    is_active: jnp.ndarray,            # (..., nlev)  — 1=wet, 0=below seafloor
    eps: float = 1e-30,
) -> jnp.ndarray:                       # (..., nlev) σ_k
```

Returns: per-cell harmonic-mean monotonized density slopes ``σ_k``.

**Algorithmic details**:
- Compute one-sided slopes ``Δρ_top``, ``Δρ_bot`` via
  ``jnp.roll`` on cell axis with fill values (top: roll +1, fill
  ρ_0 and z_centroid_0; bottom: roll −1, mask using is_active).
- Harmonic mean: ``σ = 2 * Δρ_top * Δρ_bot / (Δρ_top + Δρ_bot + eps)``
  with ``jnp.where(sign(Δρ_top) == sign(Δρ_bot), σ_harm, 0)`` for
  monotonization.
- One-sided at top: ``σ_0 = Δρ_bot_0``.
- One-sided at bottom_active: ``σ_{bot_active} = Δρ_top_{bot_active}``.
- Inactive cells (``is_active == 0``): set ``σ = 0`` (no contribution
  to integral anyway).

**Tests** (`tests/ocean/unit/test_pgf_smc03_phase1.py`):

1. **Linear ρ(z)** in a uniform-cell column: ``σ_k`` exact for all
   interior cells, ``Δρ_bot_0`` for top, ``Δρ_top_{nlev-1}`` for
   bottom.  Test on multiple slopes including zero (constant ρ).
2. **Linear ρ(z)** in a partial-cell column with one column having
   ``bottom_level < nlev−1``: σ at the partial bottom is one-sided
   (``Δρ_top``), others exact slope.  Test that two adjacent columns
   with different ``bottom_level`` give the same ``σ`` at the cell
   above the partial bottom (where both columns have full cells).
3. **Sign-changing ρ(z)** (synthetic): at the local extremum cell,
   ``σ = 0`` (monotonization).
4. **AD smoothness**: ``jax.grad`` of ``σ_k`` w.r.t. ``ρ_per_cell``
   gives finite values, no NaN through the harmonic-mean formula.

### Phase 2: Per-column pressure profile from harmonic reconstruction

**Deliverable**: ``compute_pressure_at_target_smc03`` in same
operators file.

Signature:
```python
def compute_pressure_at_target_smc03(
    rho_per_cell: jnp.ndarray,         # (..., nlev)
    h_partial: jnp.ndarray,            # (..., nlev)
    z_centroid: jnp.ndarray,           # (..., nlev)
    sigma: jnp.ndarray,                # (..., nlev) — from Phase 1
    z_target: jnp.ndarray,             # (..., n_targets)
    g: float,
) -> jnp.ndarray:                       # (..., n_targets)
```

Returns: ``P(z_target)`` per column.

**Algorithm** (vectorized over targets):

1. Compute ``P_top`` at each cell-top interface:
   ``P_top_0 = 0``,
   ``P_top_{k+1} = P_top_k + g · h_partial_k · rho_per_cell_k``.
2. Compute ``z_top``, ``z_bot`` per cell from ``h_partial``.
3. For each target ``z_t``: find the cell ``k_t`` containing ``z_t``
   (i.e. ``z_top_k_t >= z_t >= z_bot_k_t``).  Use ``jnp.searchsorted``
   on ``-z_top`` (ascending) for ``-z_t``.
4. Linear contribution:
   ``P(z_t) = P_top_{k_t} + g · (z_top_{k_t} − z_t) ·
              [ρ_k_t + 0.5 · σ_k_t · (z_top_{k_t} + z_t − 2 · z_centroid_{k_t})]``.
5. For ``z_t`` below the column's seafloor: saturate to
   ``P_top_{nlev}`` (or whatever is appropriate; the face will be
   masked downstream).

**Tests**:

1. **Constant ρ_const column**: ``P(z) = −g · ρ_const · z``.  Verify
   for multiple z in different cells.
2. **Linear ρ(z) = a·z + b** in a column with full cells: ``P(z) =
   −g · (a · z²/2 + b · z)``.  Verify exact match.
3. **Continuity across cell interfaces**: ``P(z_top_k) ==
   P(z_bot_{k-1})`` to machine precision.
4. **Cross-column consistency at a common z**: build two columns
   with same analytical ρ(z) but different cell discretizations
   (one full cells, one partial-bottom).  ``P_W(z) − P_E(z) = 0``
   to machine precision for any z within both columns' active range.

### Phase 3: Density-Jacobian PGF operator

**Deliverable**: ``density_jacobian_pgf_smc03_x/y`` in same operators
file.  Replaces the simpler ``density_jacobian_pgf_x/y`` framework
sketched in the prior session (which was reverted).

Signature:
```python
def density_jacobian_pgf_smc03_x(
    rho_per_cell, h_partial, is_active, z_full_ref, grid, g,
) -> jnp.ndarray:    # (n_lat, n_lon+1, nlev)
```

**Algorithm**:
1. Compute ``z_centroid`` from ``h_partial``.
2. Compute ``σ`` via Phase 1.
3. Set ``z_target = z_full_ref`` per face level (Option A from §2.3).
4. Compute ``P_at_target`` per column via Phase 2.
5. Horizontal Jacobian:
   ``∂P/∂x = (P_E(z_target_kf) − P_W(z_target_kf)) / dx_u``,
   per face level ``k_f``, with periodic wrap in longitude.

**Tests** (`tests/ocean/unit/test_pgf_smc03_phase3.py`):

1. **Flat-bottom z* equivalence**: configure with ``isinstance(z_coord,
   OceanZStarCoordinate) and bottom_level == nlev−1``.  Compare to
   the existing centered-diff PGF — must be bit-exact (since
   harmonic-mean reduces to standard slope, and analytical integral
   at centroid reduces to ``cumsum(g·ρ·h)`` at centroid).
2. **Stepped bathymetry with centroid-aware T**: ``|du/dt|max`` at
   rest must be **lower than Adcroft** (target: ≤ 1e−9 m/s²
   vs Adcroft's 6.8e−7 m/s²).
3. **Smoothness in k**: at a partial-cell face, the per-level
   ``|du/dt|`` must be a continuous function of ``k_f`` (no spike).
   Quantitative check: ``|du/dt[k+1] − du/dt[k]|`` at adjacent face
   levels < some threshold (~3× the typical magnitude).
4. **AD smoothness**: ``jax.grad`` of ``sum(du/dt²)`` w.r.t. T finite
   everywhere.

### Phase 4: Integrate into ocean_pe pipeline

**Deliverable**: dispatch on ``config.pgf_scheme`` in
``ocean_pe_latlon_cgrid.py``:

- ``"adcroft"`` (default): existing centered-diff p_prime + Adcroft
  correction.  Bit-exact preserved.
- ``"smc03"``: skip the centered-diff p_prime and Adcroft correction,
  use Phase 3 operator directly.

```python
if isinstance(z_coord, OceanPartialCellCoordinate):
    if config.pgf_scheme == "smc03":
        dp_dx = density_jacobian_pgf_smc03_x(
            rho_per_cell, z_coord.h_partial, z_coord.is_active,
            z_coord.z_full_ref, grid, g_val,
        )
        dp_dy = density_jacobian_pgf_smc03_y(...)
    else:
        # existing adcroft path
        dp_dx = dp_dx + partial_cell_pgf_correction_x(...)
        dp_dy = dp_dy + partial_cell_pgf_correction_y(...)
```

**Important**: ``rho_per_cell`` for SMC03 should be the **same**
``rho_prime`` field used by the Adcroft path — i.e. the output of
``iterate_eos_and_pressure_anomaly`` at the J=1, η=0 reference
(``h_actual = h_partial``).  This preserves AD pytree consistency
with the existing pipeline.

**Tests**: re-run all of the existing partial-cell test suite
(`test_partial_cells_phase{0..7}.py`) with ``pgf_scheme="smc03"``.
Must preserve all invariants:
- Phase 6 forward + AD bit-exact regression on flat bottom.
- Phase 7 H&A column-sum identity.
- Phase 7 tracer mass conservation.
- Phase 7 w=0 at the partial seafloor.

The first three should be bit-exact preserved because SMC03 acts
only on the PGF computation; nothing else changes.

### Phase 5: BH stress-test validation

Run `scripts/validate/realistic_geometry/run_phase3a_seamount.py`
with `--pgf-scheme smc03` (add the script flag) at smoothing 0, 2, 5,
10, with bottom_drag_r ∈ {0, 1e-3, 5e-3}.

**Pass criterion**: at smoothing=5, r_max=0.54, drag=1e-3 → |u|max
≤ 5 mm/s after 30 days.

**If it passes**: proceed to Phase 6.  Document the result in
`partial_cells_results.md`.

**If it doesn't pass**:
- Diagnose: is |u|max still a 2Δz mode, or now a smoother flow?
- If still 2Δz: increase reconstruction order (cubic spline instead
  of harmonic-linear).
- If smooth bottom-trapped: try Option B for ``z_face`` (per-face
  mean of centroids) or look at residual horizontal Jacobian
  curvature terms.

### Phase 6: Real-ETOPO 30-day stress test

The headline experiment.  Use real-bathymetry script (currently
exists on `realistic-geometry-full` branch — port to current
branch).  Configure:
- 1° lat-lon grid (180 × 360)
- 30 vertical levels
- ETOPO bathymetry with smoothing_passes=5
- Implicit-CN barotropic
- bottom_drag_r=1e-3, distributed BBL with thickness=100m
- pgf_scheme="smc03"
- 30-day rest-state integration

**Pass criterion**:
- Model integrates 30 days without NaN.
- |u|max < 50 mm/s (much looser than BH; real ETOPO has many
  partial cells but we don't expect zero spurious flow).
- Visual check: spurious flow should not concentrate at any
  particular bathymetric feature.

---

## 4. Risks and unknowns

1. **Cubic spline vs harmonic-linear**: this plan starts with
   harmonic-limited piecewise-linear (S&M03 §4 simplified form).
   The full S&M03 Section 4 also describes a quadratic-monotonized
   variant.  Linear-with-harmonic-slope is the simplest version
   that should kill the 2Δz mode for linear stratification.  If
   real-ETOPO with non-linear stratification still shows residuals,
   may need to upgrade to cubic.

2. **`searchsorted` AD compatibility**: ``jnp.searchsorted`` is
   piecewise-constant in its arguments, so ``jax.grad`` through it
   is zero — *not* an issue for our use because we differentiate
   w.r.t. ρ and η, not w.r.t. cell indices.  But verify with a smoke
   test in Phase 2.

3. **Cost**: Phase 2's vectorization over targets gives
   ``(n_lat, n_lon, nlev_cell, nlev_face)`` intermediate.  For
   1° × 30 levels = 180 · 360 · 30 · 30 = 5.8M ≈ 47 MB at fp64.
   Fine.  At 0.25° × 60 levels: 720 · 1440 · 60 · 60 = 3.7B ≈ 30 GB.
   Too much for GPU.  Need to chunk over ``n_lon`` or ``n_lat`` for
   high-resolution runs — or replace with explicit per-cell
   contribution accumulation that doesn't materialize the full
   ``(nlev_cell, nlev_face)`` tensor.  Address in Phase 5 if
   high-res tests fail.

4. **Boundary handling at the partial seafloor**: the ``σ`` slope
   estimate at the bottom-active cell uses one-sided ``Δρ_top`` only.
   For a column where the partial cell sits at a level k where
   ``ρ_{k-1} − ρ_k`` reflects the smooth profile, this is fine.
   But if the partial cell's own ``ρ_k`` is itself an outlier (e.g.
   from EOS quirks at deep low-T water), one-sided slope can be
   wrong.  Test on real-ETOPO will catch this.

5. **Interaction with non-zero η**: the existing
   ``iterate_eos_and_pressure_anomaly`` uses ``h_partial`` (zero-η
   reference) for the cumsum.  SMC03 must use the same convention to
   stay consistent with ``rho_prime``.  The plan does so (Phase 4
   ``rho_per_cell = rho_prime``).  At η ≠ 0, both schemes pretend
   η = 0 — this is a known shared limitation, separate from the
   density-Jacobian work.

---

## 5. Backward-compatibility contracts

1. **``pgf_scheme`` default = ``"adcroft"``**: existing tests, runs,
   and downstream code see no change.  Adding the new
   path is purely additive.
2. **Bit-exact for pure z* on flat bottom**: under
   ``pgf_scheme="smc03"``, the harmonic-linear reconstruction
   reduces to the analytical linear ρ(z) for any column with
   uniform ``dz``, and the integral evaluated at ``z_full_ref =
   centroid`` reduces to the standard ``cumsum p_prime`` at the
   centroid.  Phase 3 test 1 enforces this.
3. **Existing partial-cell tests**: under
   ``pgf_scheme="smc03"``, all of `test_partial_cells_phase{0..7}.py`
   pass.  Phase 4 enforces this.

---

## 6. References

1. **Shchepetkin, A. F., & McWilliams, J. C.** (2003). A method for
   computing horizontal pressure-gradient force in an oceanic model
   with a non-aligned vertical coordinate. *Journal of Geophysical
   Research: Oceans*, 108(C9), 3090.  https://doi.org/10.1029/2001JC001047

   The canonical reference for the density-Jacobian PGF.  Their
   §4 has the discrete algorithm; §5 has stress tests.  Their
   "Standard Jacobian" version (§4.1) is essentially what this
   plan implements; their "Modified Jacobian" (§4.2) adds higher
   order corrections that we don't need for the harmonic-linear
   form.

2. **Adcroft, A., & Campin, J.-M.** (2004). Rescaled height
   coordinates for accurate representation of free-surface flows
   in ocean circulation models. *Ocean Modelling*, 7(3-4), 269–284.
   The leading-order face-PGF correction the existing code uses.
   Provides the baseline against which SMC03 is an improvement.

3. **Beckmann, A., & Haidvogel, D. B.** (1993). Numerical Simulation
   of Flow around a Tall Isolated Seamount.  Part I: Problem
   Formulation and Model Accuracy.  *Journal of Physical
   Oceanography*, 23(8), 1736–1753.  The canonical seamount stress
   test (§5.1 of this plan).

4. **MOM6 source**: ``MOM_PressureForce_Mont.F90`` and
   ``MOM_PressureForce_blocked_AFV.F90`` in the GFDL MOM6 repo
   implement S&M03-style density-Jacobian PGF in Fortran.  Useful
   to cross-check our Python translation.

---

## 7. Checklist for the new session to follow

Open this file at the start of the new session.  Work through
phases in order, committing after each one passes its tests.

- [ ] **Phase 1**: harmonic slope reconstruction
  - [ ] Implement ``reconstruct_harmonic_slopes``
  - [ ] Add 4 unit tests in ``test_pgf_smc03_phase1.py``
  - [ ] All 4 tests pass
  - [ ] Commit
- [ ] **Phase 2**: pressure profile
  - [ ] Implement ``compute_pressure_at_target_smc03``
  - [ ] Add 4 unit tests in ``test_pgf_smc03_phase2.py``
  - [ ] All 4 tests pass
  - [ ] Commit
- [ ] **Phase 3**: density-Jacobian PGF operator
  - [ ] Implement ``density_jacobian_pgf_smc03_x/y``
  - [ ] Add 4 unit tests in ``test_pgf_smc03_phase3.py``
  - [ ] Verify rest-state PGF on stepped bathymetry < 1e-9 m/s²
  - [ ] Commit
- [ ] **Phase 4**: PE pipeline integration
  - [ ] Wire ``pgf_scheme="smc03"`` dispatch in ``ocean_pe_latlon_cgrid.py``
  - [ ] Re-run full ``test_partial_cells_phase{0..7}.py`` with
        ``pgf_scheme="smc03"`` — all pass
  - [ ] Re-run with ``pgf_scheme="adcroft"`` (default) — all pass
        (no regression)
  - [ ] Commit
- [ ] **Phase 5**: BH validation
  - [ ] Add ``--pgf-scheme`` flag to BH script
  - [ ] Run sweep at smoothing 0, 2, 5, 10
  - [ ] Confirm |u|max < 5 mm/s at smoothing=5 (target)
  - [ ] Update ``partial_cells_results.md`` with the result
  - [ ] Commit
- [ ] **Phase 6**: ETOPO stress test
  - [ ] Port real-ETOPO script from ``realistic-geometry-full``
  - [ ] Run 30-day integration with ``pgf_scheme="smc03"``
  - [ ] Document |u|max, NaN-free, spatial pattern
  - [ ] Commit
- [ ] Open PR with the 6 commits, ``docs/ocean/experiments/partial_cells_results.md``
      updated to reflect the BH gap closure
