# Fox-Kemper MLE — MPAS (Voronoi) port plan

**Goal:** bring the Fox-Kemper mixed-layer-eddy (MLE) restratification parameterization
(NEMO `ln_mle=.true.`, `nn_mle=1`) to the MPAS icosahedral grid, so the *high-res faithful
grid* (ico6/ico7) gets submesoscale restratification. Today MLE is **C-grid only**
(tripole/latlon, commit 97855a07); `run_omip_core2.py:2196` raises `ValueError` for
`--grid mpas`. This is the top remaining missing parameterization (see
`docs/md_files/ocean_faithfulness_nemo.md` iter-J): it is the single mechanism most plausibly
fixing BOTH the too-deep NH winter MLD (ico6 March +47..51 m) AND the weak MLD pattern
(ico7 corr 0.57) / tropical upper-ocean stratification.

## Why this is now SMALL, not from-scratch
The doc's old "gm_redi_mpas is an unfinished skeleton" note is **stale**. The MPAS GM/Redi
*centered* scheme is functional and **in production** (ico7 ran `gm_redi.kappa_GM=600`,
`slope_scheme="centered"`). So the hard Voronoi infrastructure MLE needs **already exists**:
- `gm_redi_mpas.compute_isopycnal_slopes_mpas` — edge buoyancy/density gradients on the dual mesh.
- `gm_redi_mpas.gm_redi_tracer_tendency_centered_mpas` — the **conservative cell tracer-tendency
  from a bolus/eddy streamfunction** (edge→cell Perot/divergence, `_voronoi_neumann_fill`,
  partial-cell `min_cell_to_edge` thickness). MLE is just a *different, near-surface, MLD-confined*
  overturning streamfunction fed through the same conservative divergence.
- `mle.py` is **already grid-agnostic** (`MLEConfig`, `mle_coefficient` rc_f, `mle_vertical_structure`
  μ(z), `mle_mld_and_buoyancy`, `face_mld`) — reused verbatim; nothing C-grid in it.

So the port = **(a)** an MPAS-native buoyancy-gradient + Ψ_MLE assembly on edges, **(b)** route it
through the existing GM bolus-divergence, **(c)** config/CLI/guard removal, **(d)** tests. The
genuinely new numerics is only the edge-MLD + edge-Ψ assembly on Voronoi (a Perot-consistent analog
of `mle_latlon_cgrid.py`'s face-Ψ).

## NEMO oracle (unchanged from the C-grid port)
`ln_mle=T, nn_mle=1, rn_ce=0.06, rn_lat=20, rn_rho_c_mle=0.01`. Ψ_MLE structure
(`tramle.F90`): `Ψ = C_e · (Δb_ml · H_ml² / |f|_lat-floored) · μ(z) · (1/Lf)` with the MLD from a
0.01-in-situ-ρ criterion, ML-mean buoyancy, μ(z) the FK vertical shape (`mle_vertical_structure`),
and a convection gate (suppress where the ML column is statically unstable, min-N² < 0).

## Phased implementation (each phase: codex-review + test before next)

### Phase 1 — `mle_mpas.py` (the only new numerics)
New `packages/ocean/legoesm/ocean/physics/lateral_mixing/mle_mpas.py`,
`def mle_tracer_tendency_mpas(T, S, rho_insitu, N2, mesh, z_coord, cfg, mask, h_k=None, eos=...) -> (dT_dt, dS_dt)`
— signature mirrors `gm_redi_tracer_tendency_centered_mpas` (cell arrays in, cell tendencies out).
Steps:
1. **Cell MLD + ML-mean buoyancy:** call `mle.mle_mld_and_buoyancy` per cell column (already
   grid-agnostic; takes T,S,rho,z, returns zmld [stop_grad], b_ml, in_ml mask). Use live `h_k`
   from `compute_layer_thickness` on partial cells (same as GM-MPAS) for gdepw/volume consistency.
2. **Edge MLD** via `mle.face_mld(h_c1, h_c2, mode)` on `mesh.cellsOnEdge` (NEMO `nn_mld_uv`,
   default min).
3. **Edge horizontal buoyancy gradient** `∂b/∂n`: reuse the SAME edge-gradient primitive
   `compute_isopycnal_slopes_mpas` uses (`(b[c2]-b[c1])/dcEdge`), ML-mean b at the two cells.
4. **Edge Ψ_MLE(z):** `Ψ_e = C_e · Δb_e · H_ml,e² / max(|f_e|, f_floor) · μ(z; gdepw/H_ml,e)`,
   f at the edge (`mesh.fEdge` or interpolated `fCell`), `f_floor = 2Ω sin(rn_lat=20°)` from
   `mle_coefficient` (the equatorial-singularity guard — CRITICAL, see Risk 1). Project onto the
   edge normal (Ψ is a transport, already normal at edges).
5. **Bolus tracer tendency:** the MLE overturning advects T,S. Build the w-interface bolus flux
   `dk[Ψ_e]·tracer_face` and take its conservative cell divergence — **reuse the GM-MPAS
   divergence path** (`_perot_inner_product_cell` / the centered flux-divergence helper) rather
   than re-deriving. Convection gate (`in_ml` + min-N²<0) zeroes Ψ where the column overturns.
6. Conserve exactly: Σ dT·vol ≈ 0, Σ dS·vol ≈ 0 (the divergence form guarantees it; assert in test).

### Phase 2 — wire into the model step
`ocean_model_mpas.py` ~L458: right AFTER the GM/Redi block
(`S_new = S_new + dt*dS_gm*active_3d`), add the symmetric MLE block:
```python
if config.mle is not None:
    dT_mle, dS_mle = mle_tracer_tendency_mpas(T_new, S_new, ..., mesh, z_coord, config.mle, mask=mask)
    T_new = T_new + dt * dT_mle * active_3d
    S_new = S_new + dt * dS_mle * active_3d
```
Additive, forward-Euler, before advection — bit-identical pattern to GM and to the C-grid MLE.
`config.mle` already exists on `OceanPhysicsConfig` (used by C-grid); MPAS model reads the same field.

### Phase 3 — config / CLI / guard
- Remove the `args.grid in ("mpas","cubed_sphere")` rejection in `run_omip_core2.py:2196`
  (keep cube rejected — parked grid). Build `MLEConfig(ce=args.mle_ce)` and attach to the MPAS
  physics config (the `_create_setup('mpas', ...)` path), same as latlon/tripole.
- No new CLI flags (`--mle`, `--mle-ce` already exist and are grid-neutral).

### Phase 4 — tests (`tests/ocean/unit/test_mle_mpas.py`)
1. **Exact tracer conservation** Σ dT·vol≈0 / Σ dS·vol≈0 on a small ico mesh (the must-pass).
2. **Restratify sign:** an imposed ML horizontal buoyancy front → Ψ tendency *shoals* the MLD /
   tilts isopycnals toward horizontal (de-densifies the light side at depth). Compare sign to the
   C-grid `test_mle_latlon_cgrid` analog.
3. **μ(z) / MLD reuse:** assert `mle_mpas` calls the shared `mle.py` core (no re-derived μ or rc_f).
4. **Equatorial guard:** Ψ finite at the equator (f→0) — the `rn_lat=20°` floor active (Risk 1).
5. **Convection gate:** Ψ=0 in a statically-unstable ML column.
6. **Physics contract** + a leaf import test (every new `.py` gets a direct unit test — CLAUDE.md).

### Phase 5 — validation run + score
ico6 (~115 km, cheap) `--mle` 2-yr vs the no-MLE baseline: does MLE **shoal the NH-subtrop /
subpolar winter MLD** (the +47..51 m over-deepening) and **raise the MLD pattern corr** (0.57→?)?
Score with `compare_omip_nemo.py --sss-faithful` (MLD block already emitted). Then ico7 (~57 km).

## Risks (carried from the C-grid port + MPAS-specific)
1. **Equatorial robustness (BLOCKER on the C-grid too).** `tripole_2yr_mle` (1°, dt600) blew up at
   **day-240 near the equator** (max|u| 1.8→4.6→nan, umax_lat −4.8..8.5) under the weakest-viscosity
   window. rc_f has |f| in the denominator; the `rn_lat=20°` floor bounds it, so the blowup is likely
   a *secondary* interaction (MLE bolus + weak late-run viscosity + tropical-instability waves), not
   the raw singularity — but it MUST be reproduced/closed before trusting MLE-MPAS at the equator.
   Mitigation to test: keep the 20° f-floor, and/or an equatorial Ψ taper, and/or do not drop the
   background viscosity as far in the MLE config.
2. **Perot/divergence consistency.** The MLE Ψ MUST go through the *same* conservative
   edge→cell divergence as GM (no parallel hand-rolled divergence) or it breaks tracer conservation
   on the dual mesh. This is the reuse that keeps the port small AND correct.
3. **Partial cells.** Use `min_cell_to_edge` edge thickness + live `h_k` (OMIP runs `--partial-cell`),
   exactly as GM-MPAS and the C-grid MLE do — no `dz_ref`.
4. **Cost at ico7.** MLE adds one more per-step lateral pass; ico7 is already ~2.4 steps/s. Acceptable;
   verify no recompile (stable shapes, `config.mle` static gate à la `fix_mass`).

## Out of scope
Cube MLE (parked grid). MLE bolus on *momentum* (NEMO applies MLE to tracers only). Triad GM
(`gm_redi_tracer_tendency_triads_mpas` still raises NotImplementedError — independent follow-up).
