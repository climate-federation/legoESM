# Plan: Mercator Lat-Lon Grid for legoESM Ocean

## Goal

Add a Mercator latitude-longitude grid generator for the ocean lat-lon C-grid in legoESM, and make the existing C-grid operators work on a non-uniform-dlat grid. Land as a PR-ready branch on `main`.

Branch: `feature/mercator-grid` (already created off `origin/main`).
Working directory: `/Users/dhruvbalwada/Work/Projects2026/legoESM-mercator`.

## Motivation

This is a prerequisite for the DINO ocean experiment replication (Kamm et al. 2025, GMD), which is being worked on in parallel on `feature/dino-experiment`. DINO uses a Mercator grid so that horizontal grid spacing decreases poleward in proportion to `cos(φ)` — matches the latitudinal scaling of the first baroclinic deformation radius (Hallberg 2013).

You do **not** need to read the DINO plan to do this task. Treat Mercator as a self-contained piece of grid infrastructure that any future lat-lon ocean experiment might use.

## Structural blocker (the real reason this is a separate PR)

`LatLonGrid.dy` is currently `float` (a Python scalar). The C-grid operators in `src/legoesm/ocean/dynamics/latlon_cgrid_operators.py` and elsewhere compute `dy = R * dlat` as a constant scalar everywhere. This assumes uniformly-spaced latitudes.

Mercator places latitudes via `φ(j) = (180/π) · arcsin(tanh(Δλ · π/180 · j))`, so cell heights `dy(j)` vary with latitude (decreasing poleward). The scalar `dy` assumption must be lifted.

**Decision (already made)**: `dy` becomes a 1D `jax.Array` of shape `(n_lat,)`. This is sufficient for any orthogonal lat-lon grid (each latitude band has the same `dy` for every longitude). **Do not make `dy` 2D** — that adds complexity needed only for tripolar / displaced-pole grids, which are out of scope.

## Hard constraints

- Read `CLAUDE.md` before editing anything. Audit-enforced rules apply (no hardcoded constants, etc.).
- Do **not** change MPAS code paths.
- Do **not** add new physical constants — use `legoesm.constants` (e.g., `constants.R_earth`, `constants.Omega`). Tests must follow the same rule.
- Do **not** add backwards-compat shims like `_dy_scalar` fallback fields. Direct API change.
- Do **not** modify the DINO plan (`docs/ocean_experiments/dino_replication_plan.md`) or any DINO experiment code.
- If you discover an operator that needs more than a mechanical `R*dlat → grid.dy` substitution (e.g., a numerical scheme that genuinely assumed uniform `dlat`), **stop and write up the issue** rather than guessing. Report and ask before proceeding.

## Phased plan

### Pass 1 — Audit

Before editing anything:

1. Read `src/legoesm/grids/latlon.py` end-to-end. It defines `LatLonGrid` (NamedTuple), `create_latlon_grid` (global), `create_regional_latlon_grid` (channel / closed basin).
2. Grep every use of `grid.dy` and `grid.dlat` across `src/legoesm/` and `tests/`. Note the file paths and line numbers.
3. Identify which operators do `dy = R * dlat` as a scalar — these are the refactor surface.

**Deliverable for Pass 1**: a short audit report (text) listing files that must change, before any edits. The audit informs Pass 2 scope.

### Pass 2 — Refactor `LatLonGrid.dy` to 1D array

Change `LatLonGrid.dy` annotation from `float` to `jax.Array` of shape `(n_lat,)`.

Update `create_latlon_grid()` and `create_regional_latlon_grid()`:
- Build a 1D `dy` array with the existing constant value (`R * dlat`) broadcast to `(n_lat,)`.
- Pass it into the `LatLonGrid` constructor.

Update every operator + physics module identified in Pass 1:
- Replace `dy = R * dlat` (scalar) with `dy = grid.dy[:, None]` where the broadcasting target is a 2D `(n_lat, n_lon, ...)` field.
- For face-located fields (between cells), use the appropriate face-dy (you may need to introduce `grid.dy_face` or compute on the fly — keep this minimal and document choice).

After Pass 2, every existing test must still pass. Verify:

```bash
JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/ocean/ -x
```

If any test fails, fix before moving on. Don't proceed to Pass 3 with regressions.

### Pass 3 — Add `create_mercator_grid()`

Add to `src/legoesm/grids/latlon.py`:

```python
def create_mercator_grid(
    n_lon: int,                  # number of zonal cells
    lat_max_deg: float,          # truncation latitude (symmetric N/S)
    lon_west_deg: float = 0.0,
    lon_east_deg: float = 360.0,
    radius: float = constants.R_earth,
    omega: float = constants.Omega,
    dtype=None,
) -> LatLonGrid:
    """Mercator (isotropic) latitude-longitude grid.

    Latitudes are placed via φ(j) = (180/π) · arcsin(tanh(Δλ · π/180 · j))
    where Δλ = (lon_east_deg - lon_west_deg) / n_lon, j symmetric around 0.
    This yields dx(j) ≈ dy(j) at every latitude (isotropic cells) and
    Δy decreases poleward in proportion to cos(φ).

    n_lat is determined automatically from lat_max_deg by inverting the
    placement formula.
    """
```

Latitude placement details:
- `Δλ_deg = (lon_east_deg - lon_west_deg) / n_lon`
- `j_max = arctanh(sin(lat_max_deg · π/180)) / (Δλ_deg · π/180)`
- `j ∈ {-floor(j_max), ..., -1, 0, +1, ..., +floor(j_max)}` for symmetric cell centers, or shifted by ½ if you want centers (not face) placement — be explicit about which.
- Cell-center latitudes: `φ_c(j) = (180/π) · arcsin(tanh(Δλ_deg · π/180 · j))` evaluated at half-integer j (so centers don't coincide with the equator face).
- Cell-face latitudes: `φ_f(j+½)` evaluated at integer j.
- `dy(j) = R · (φ_f(j+½) - φ_f(j-½))` in radians × R → meters.
- `dx(j) = R · cos(φ_c(j)) · Δλ_rad`.
- Verify: `dx(j) ≈ dy(j)` at every latitude (within numerical accuracy of the placement formula).

`n_lat` = number of cell centers between ±lat_max_deg (symmetric).

Include a short ASCII sketch in the docstring showing the relationship.

### Pass 4 — Tests

Two new test files. Both must follow CLAUDE.md rules: `from legoesm import constants`, no hardcoded `6.371e6`, `9.80616`, `7.292e-5`, etc.

**`tests/grids/test_mercator.py`**:
- Lat placement formula matches reference (compute manually for a few j values, compare).
- `dx(j) ≈ dy(j)` to within 1% at every latitude (isotropy check) for a few sample resolutions (n_lon ∈ {180, 360, 720}).
- Total area integrates to the spherical-cap area `2πR²·(1 - sin(lat_max))·2` (both hemispheres).
- Edge cases:
  - Very low n_lon (e.g., 36 = 10° resolution).
  - Near-equator-only domain (`lat_max_deg=10`).
  - Near-pole truncation (`lat_max_deg=85`).
- Round-trip through `jax.jit` — the grid is a pytree, should be JIT-traceable.

**`tests/ocean/unit/test_operators_on_mercator.py`**:
- Gradient of `sin(lat)·cos(lon)` (analytic): the C-grid `gradient_cgrid` should recover this to < 1% at 1° Mercator. Test in interior, avoid pole boundary effects.
- Divergence-of-curl is ≤ 1e-12 (numerical identity on a rest field).
- Rest-state advection: a constant tracer field should remain constant after one step of tracer advection (with zero velocity).
- Optional but valuable: PGF rest state — initialize hydrostatic density profile on Mercator and confirm horizontal PGF acceleration is small (compare with equirectangular for context).

### Pass 5 — Full ocean test matrix + slopbuster + PR

1. Run the full ocean test matrix:
   ```bash
   JAX_ENABLE_X64=1 .venv/bin/python scripts/matrix/run_ocean_test_matrix.py
   ```
   This is slow — run in background, check periodically. Confirm zero regressions in equirectangular-grid paths (DST-3, Eady, ACC channel, etc.).

2. Run slopbuster:
   ```
   /slopbuster review
   ```
   Address every finding before committing. Pay special attention to:
   - Hardcoded constants snuck in (`6.371e6`, `7.292e-5`, etc.)
   - Unused imports
   - NamedTuple field defaults that should reference `constants.X`
   - Test files using literal constants

3. Commit with a clear message. Example structure:

   ```
   Mercator grid + variable-dy refactor for lat-lon ocean

   - LatLonGrid.dy: float → jax.Array of shape (n_lat,)
   - Update C-grid operators in latlon_cgrid_operators.py, lateral
     mixing modules, [list others]
   - Add create_mercator_grid() with isotropic dx≈dy
   - Tests: grid placement, isotropy, area integration, operators on
     Mercator (gradient, divergence-of-curl, rest state)
   - Full ocean test matrix: no regressions

   Prerequisite for DINO experiment replication (feature/dino-experiment).
   ```

4. Push the branch and open the PR against `main`:

   ```bash
   git push -u origin feature/mercator-grid
   gh pr create --base main --title "..." --body "..."
   ```

   PR body should reference the motivation (DINO replication) and call out the API change (`dy` type) so reviewers know to look for downstream breakage.

## Report mode

Terse. State what you found, what you changed, and what broke. No prose, no progress reports written as essays. After each pass, summarize in a few bullets and pause for any go/no-go input.

## Done criteria

- [ ] Pass 1 audit report posted
- [ ] Pass 2 — `LatLonGrid.dy` is 1D array; `pytest tests/ocean/ -x` green
- [ ] Pass 3 — `create_mercator_grid()` exists with isotropy verified
- [ ] Pass 4 — both new test files green
- [ ] Pass 5 — full ocean test matrix green, slopbuster clean, PR open
