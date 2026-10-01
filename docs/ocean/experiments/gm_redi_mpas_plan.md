# GM/Redi on MPAS — implementation plan

**Status (2026-04-28):** Scoping only. No implementation in tree.
API skeleton at `src/legoesm/ocean/physics/lateral_mixing/gm_redi_mpas.py`
raises `NotImplementedError` against this plan.

## Motivation

The lat-lon C-grid global overturning baseline relies on GM/Redi triads
+ Visbeck adaptive coefficient (see `run_global_overturning_50yr_gmredi.py`)
to set deep stratification and the meridional overturning circulation.
The MPAS port of the same experiment
(`run_global_overturning_mpas_baseline.py`, ico4 ~4°) currently runs
without parameterised eddy transfer, so:

- Deep stratification is set by spurious numerical mixing at the
  ico4 cell scale rather than by the parameterised eddy buoyancy flux.
- The Drake-band ACC structure has no GM-driven Eulerian-mean overturning
  cancellation, so the residual circulation seen in the MPAS run is not
  comparable to the lat-lon `50yr_implicit` MOC reference.
- The 1-yr shakedown shows a concentrated equatorial jet at ~1 m/s; part
  of that is plausibly an under-damped Kelvin/Yanai response which GM
  partially mitigates by extracting buoyancy variance.

GM/Redi on MPAS is therefore the largest open feature gap on the path to
quantitative MPAS-vs-lat-lon comparison of the global overturning
experiment.

## Scientific contract

The MPAS port must reproduce, on the equivalent test cases, the
behaviours that the lat-lon implementation already validates:

1. **Redi cancellation** — when `q = f(rho)` (linear EOS, T-only), the
   off-diagonal Redi flux must cancel the horizontal Redi flux to
   numerical precision (this is what triads buy and centred lacks).
2. **GM streamfunction** — eddy buoyancy flux integrated zonally
   produces a clean clockwise residual cell in the ACC channel test
   (matches `validate_triad_redi_120day.py` baseline).
3. **Visbeck adaptive κ** — coefficient grows with `⟨N · |S|⟩_z` and
   tapers at high latitudes via Coriolis.
4. **Slope tapering at S_max** — Danabasoglu-McWilliams 1995 tanh
   taper applied to slopes whose magnitude exceeds `S_max ≈ 5e-3`.
5. **Conservation** — total tracer mass invariant to round-off across
   any closed sub-domain (MPAS has no land-leak path; verify via
   `tests/ocean/unit/test_mpas_ocean.py`-style closure check).
6. **Land/face mask respect** — fluxes through edges adjacent to a
   land cell are zero.

## API surface (already drafted)

The public entry point mirrors the lat-lon counterpart so the
`MPASOceanModel.step` hook is mechanical:

```python
def gm_redi_tracer_tendency_mpas(
    T, S, eta, H_bathy,
    mesh,            # VoronoiMesh (replaces grid)
    z_coord, cfg,
    *, eos="wright", eos_linear=None,
    mask=None,           # (nCells,)
    edge_mask=None,      # (nEdges,)  — replaces u_mask + v_mask
    f_coriolis=None,     # (nCells,)
) -> tuple[dT_dt, dS_dt]:
```

Internally, the function is composed of the same five pieces that exist
on lat-lon:

1. `compute_ocean_rho` (already grid-agnostic — operates on `state.T.data`)
2. `compute_isopycnal_slopes_mpas` — **new, this is most of the work**
3. Optional Visbeck `kappa_GM` (mostly grid-agnostic, just needs `f_coriolis`)
4. `gm_redi_tracer_tendency_centered_mpas` and/or
   `gm_redi_tracer_tendency_triads_mpas` (the actual flux+divergence)
5. Optional taper

## Implementation phasing

### Phase 1 — slope computation on Voronoi  (≈ 1–2 days)

The lat-lon centred slopes use:
- Horizontal `dρ/dx` on u-faces, `dρ/dy` on v-faces (free).
- Vertical `dρ/dz` at cell-centred interfaces.
- `S_x = -(dρ/dx) / (dρ/dz)`, `S_y = -(dρ/dy) / (dρ/dz)` after
  averaging horizontal gradients to interfaces.

On TRiSK MPAS the natural primitive is the **edge-normal density
gradient** `(ρ[c2] − ρ[c1]) / dcEdge`. This gives a *scalar* gradient
along each edge's own normal, indexed by `mesh.angleEdge`.

Two viable representations:

- **(A) Edge-normal slope `S_n[edge, lev_int]`.** Slope magnitude along
  the edge's normal. Then horizontal Redi flux at an edge becomes
  `F_n = κ_R · dq/dn + (κ_R − κ_GM) · S_n · dq/dz_at_edge`. Vertical
  flux becomes the cell-averaged sum over incident edges. This stays
  fully native to TRiSK and avoids cell-centred (S_x, S_y)
  reconstruction. **Recommended.**

- **(B) Cell-centred (S_x, S_y).** Reconstruct cell-centred horizontal
  ρ-gradients via Perot reconstruction
  (`reconstruct_cell_velocity`-style), build `(S_x, S_y)`, then average
  back to edges. More expensive and introduces extra reconstruction
  error. Avoid unless triads force it.

Subproblems for Phase 1:
- Vertical density-gradient at edge interfaces: simplest is to average
  `dρ/dz` from the two neighbouring cells, as the lat-lon code does.
- Slope tapering: `_triad_taper(S, S_max)` is grid-agnostic, reuse
  directly.
- Neumann fill on Voronoi: the lat-lon `_neumann_fill_cgrid` extends ρ
  one cell into land before differencing. The MPAS analogue extends ρ
  to land cells via averaging over ocean neighbours; this needs care
  because Voronoi cells have variable degree. Likely cheapest: skip the
  fill and rely on edge masking — gradients across a land/ocean edge
  are zeroed before flux computation.

Tests:
- Slope of a constant ρ field is zero everywhere (machine precision).
- Slope of `ρ = ρ₀ + α·z` is zero (no horizontal gradient).
- Slope of `ρ = ρ₀ + β·lat + α·z` reproduces the analytic
  `S = −β/(α·R_earth)`.
- Slope respects land mask (no slope across land edges).

### Phase 2 — centred GM/Redi tracer tendency  (≈ 1 day)

With (A) above, the centred tendency on MPAS is:

```
F_n[edge, lev_int] = κ_R · dq/dn + (κ_R − κ_GM) · S_n · dq/dz_edge
F_z[cell, lev_int] = (κ_R + κ_GM) · sum_edges_of_cell(S_n · dq/dn) · w_e
                   + κ_R · S_n²_cell · dq/dz_cell
```

where `w_e = (dvEdge · dcEdge) / (2 · areaCell)` is the standard
Perot weight. Divergence at cells is the standard TRiSK
`Σ_e signOnCell · F_n · dvEdge / areaCell`.

This is conceptually simple but the bookkeeping for averaging
horizontal gradients to interfaces (mirroring the lat-lon
"interface-only flux" trick that gives algebraic cancellation) needs
to be done carefully on the irregular Voronoi.

Tests:
- `q = f(ρ)` ⇒ off-diagonal flux cancels diagonal flux to ~1e-10
  (centred — exact cancellation only with triads).
- Reproduces lat-lon Eady GM-only steady state to within
  expected resolution-dependent error.
- Conservation: integrated ∫dT/dt · dV = 0 over the domain.

### Phase 3 — Visbeck adaptive κ_GM  (≈ 0.5 day)

`compute_visbeck_kappa_gm(rho, S_x, S_y, z_coord, jacobian, f, cfg)`
already exists in `_gm_redi_common.py` and is grid-agnostic at the
math level. The only Voronoi-specific work: provide `f_coriolis` shaped
`(nCells,)` and adapt the per-column reduction (already operates on
trailing-axis arrays).

Tests:
- κ_GM grows with prescribed `(S_x, S_y)` magnitude.
- Capped between `kappa_min` and `kappa_max`.
- κ shaped `(nCells,)`.

### Phase 4 — dycore hook  (≈ 0.25 day)

In `ocean_model_mpas.py:step`, after physics tendencies and before
tracer advection, mirror the lat-lon block at
`ocean_model_latlon_cgrid.py:590`:

```python
if self.config.gm_redi is not None:
    from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
        gm_redi_tracer_tendency_mpas,
    )
    dT_gm, dS_gm = gm_redi_tracer_tendency_mpas(
        T_mid, S_mid, state_new.eta.data, state_new.H_bathy.data,
        self.mesh, self.z_coord, self.config.gm_redi,
        eos=self.config.eos, eos_linear=self.config.eos_linear,
        mask=state.land_mask.data,
        f_coriolis=self.mesh.fEdge,    # or fCell
    )
    T_mid = T_mid + dt * dT_gm * mask_3d
    S_mid = S_mid + dt * dS_gm * mask_3d
```

Add `gm_redi: object = None` to `MPASOceanConfig`.

### Phase 5 — triad scheme  (≈ 2–3 days)

The triad scheme on Voronoi is the major open question. The lat-lon
implementation forms 4 triads per face (2 vertical × 2 horizontal
neighbours). On Voronoi each edge has 2 incident cells × 2 vertical
faces = 4 candidate triads per (edge, level) — geometrically a near-
direct analogue. The challenge is:

- The "horizontal pair" is unambiguous — the two cells `c1, c2` either
  side of the edge.
- The "vertical pair" must be consistent with the cell on which the
  triad's tracer values live (e.g. triad T1 uses cell `c1` at level
  `k+1`, triad T3 uses cell `c2` at level `k+1`).
- The tracer-gradient cancellation property requires that each triad
  uses *its own* `(dρ/dx, dρ/dz)` pair, not averaged values. This
  bookkeeping is identical to lat-lon — implementable with `vmap` over
  edge × triad index.

This phase blocks on Phase 1–4 being correct. Defer until centred is
working and the dycore hook is verified.

## Scientific validation plan

1. **Phase 1–4 smoke**: run `run_global_overturning_mpas_baseline.py`
   with `gm_redi=GMRediConfig(slope_scheme="centered", kappa_GM=1000,
   kappa_Redi=1000)` for 1 sim-yr. Verify state stays finite and the
   restart NPZ matches the lat-lon equivalent qualitatively
   (deep T cooler, |u|max smaller because of GM cancellation).

2. **Eady channel parity**: replicate the
   `tests/ocean/unit/test_gm_redi_eady_physics.py` lat-lon tests on an
   MPAS channel mesh. The PV/buoyancy signatures should match within
   resolution-dependent error.

3. **Triad cancellation** (Phase 5 only): isolated unit test confirming
   `dT/dt → 0` for `T = T(ρ)` to within 1e-10.

4. **50-yr global overturning re-run with GM/Redi** (final): launch the
   real 50-yr equivalent of `run_global_overturning_50yr_gmredi.py`
   on MPAS, compare MOC and Drake transport to lat-lon.

## Why not just "do it now"

`legoesm` codebase rule (CLAUDE.md): *"never imply completion you have
not delivered ... If the task is genuinely too large for one pass, say
so explicitly, list every piece that remains, and quantify the
residual risk."*

Phases 1–4 alone are 3–4 days of careful work. Phase 5 (triads on
Voronoi) is the open scientific subproblem that motivated the lat-lon
triad commit (806 LOC). A rushed centred-only port of GM/Redi here
would either silently produce inverted slope conventions, fail to
cancel diagonal Redi flux (the bug that motivated triads on lat-lon
in the first place), or introduce conservation errors that only
manifest at multi-year integration timescales.

The skeleton at `gm_redi_mpas.py` makes the API contract concrete so
the actual port is mechanical, and the dispatch test guards against
silent fallback to a no-op (the failure mode that bit us when
`scheme='combined'` silently dropped under MPAS surface forcing).
