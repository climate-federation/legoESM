# Adversarial review packet — MPAS q_v smoothing (round 3, post round-2 fixes)

Independent adversarial reviewer. Rounds 1-2 raised BLOCKERS (hybrid dp<=0 NaN;
false "FV handles conservation"; overstated "mirrors FV"; finiteness/fp32; tests
replicate the op not the driver; footgun dp branch; ABI). All are addressed
below. Re-review: confirm each is resolved, and find any REMAINING or NEW bug.
Cite line numbers. If clean, say so per item and explain why.

## What changed since round 2

1. **Operator is now UNWEIGHTED-only** — the mass-weighted `dp_cell_3d` branch is
   DELETED (was a docstring-only-protected footgun). `scalar_del2_cell_3d(q,
   mesh)` returns `divergence_cell_3d(gradient_edge_3d(q, mesh), mesh)`. Reverting
   the driver to a dp-weighted call is now a TypeError, not a silent NaN.

2. **Real driver code extracted to a module-scope, directly-tested helper**
   (mirrors FV's module-scope `_apply_qv_smoothing`):

```python
def _mpas_qv_smooth_step(q_v, mesh, nu, dt):
    """UNWEIGHTED SCVT del2 + q>=0 floor.  Plain (no dp) is deliberate: the
    default hybrid coordinate's surface dp can be <= 0 over high terrain.
    Monotone under nu*dt*scalar_del2_cell_cfl_factor(mesh) <= 1 (driver
    enforces <= 0.5), so the floor is a no-op and per-level sum_c A_c q_c is
    conserved; the floor sanitises finite negatives, NOT pre-existing NaN."""
    from legoesm.core.operators_voronoi import scalar_del2_cell_3d
    lap = scalar_del2_cell_3d(q_v, mesh)
    return jnp.maximum(q_v + dt * nu * lap.astype(q_v.dtype), 0.0)
```

   The driver apply block now just calls it:

```python
if _qv_smooth_nu > 0.0:
    _trc_sm = self.state.tracers
    _qv_new_sm = _mpas_qv_smooth_step(
        _trc_sm["q_v"].data, self.grid, _qv_smooth_nu, DT)
    _new_trc_sm = dict(_trc_sm)
    _new_trc_sm["q_v"] = _trc_sm["q_v"].replace(data=_qv_new_sm)
    self.state = self.state._replace(tracers=_new_trc_sm)
```

   Setup guard unchanged except it now imports only
   `scalar_del2_cell_cfl_factor` (the CFL guard `nu*dt*g_max <= 0.5`, the
   distributed + q_v-tracer refusals, logging are all as round 2).

3. **Honest documentation everywhere** — dropped the false "FV handles it
   separately" and "conserves column water mass" claims. The config field note,
   driver comments, operator docstring, and CLI help now state: an UNWEIGHTED
   del2 (∇²) — NOT FV's scale-selective del4 (∇⁴) — that conserves the per-level
   mixing-ratio integral `sum_c A_c q_c` but is NOT column-water-vapour
   conserving (across a p_s gradient it redistributes a little water mass, bias
   set by the humidity–terrain correlation): an explicitly NON-conservative
   grid-scale filter; monitor the water budget in long runs. del2 is chosen
   (over del4) because it is monotone under the explicit guard, so the floor is
   a no-op and the per-level integral is exact.

4. **Config field moved to the END of `ExperimentConfig`** (after
   `gravity_wave_drag_override`) to preserve positional NamedTuple ABI.

5. **Tests (18 pass, `JAX_ENABLE_X64=1`)** — mass-weighted operator tests
   removed; dead `_rand_dp` helper removed. Added, exercising the REAL driver
   helper `_mpas_qv_smooth_step`:
   - `test_driver_helper_finite_and_positive_on_hybrid_low_ps`: builds the real
     `make_hybrid_levels(8,...)`, p_s 350..1013 hPa so `min(dp) < 0` (asserted,
     non-vacuous); asserts the helper output is finite and `>= 0`, AND equals
     `max(q + dt*nu*scalar_del2_cell_3d(q, mesh), 0)` bit-for-bit.
   - `test_driver_helper_fp32_safe`: q in float32 → output float32, finite, >=0.
   - Kept: constant annihilation, per-level plain conservation, sign, plain-step
     monotone+positive at the CFL bound, per-level step conservation (rel 1e-13),
     checkerboard variance decay, autodiff, `scalar_del2_cell_cfl_factor` ==
     manual recompute; validate_strict accept/refuse/bounds.

## Rebuttal to round-2 #6 (YAML gap)

`src/legoesm/config.py`'s YAML→canonical adapter maps a CORE SUBSET only
(grid/dycore/output/held_suarez_forcing). It maps NONE of the sibling opt-in
ExperimentConfig knobs — not `mpas_land_lapse_K_per_km`, `mpas_land_beta`, the
hard-sat overrides, `sponge_coeff_per_day`, snow-albedo, conv-cloud, etc. The
advertised boundary for these is the CLI (repo rule: "new user-tunable field →
same-PR CLI flag", satisfied + round-trip tested). Adding only this field to the
subset adapter would be inconsistent with all its siblings. Do you agree this is
not a change required by this PR?

## Explicit ask

1. Is the hybrid dp<=0 NaN class fully closed now that the operator has NO dp
   division and the driver's exact code is the tested helper?
2. Are the conservation/finiteness/precision claims now correctly qualified (no
   remaining overclaim)? Is the non-conservative-filter framing acceptable?
3. `_mpas_qv_smooth_step` correctness: dtype (`lap.astype(q_v.dtype)`), the floor,
   shape, purity, autodiff-safety; any scope/import problem (function-local
   import of `scalar_del2_cell_3d`)?
4. Does moving the field to the tuple end interact with anything (validate_strict
   still references it; `_EXPERIMENT_DEFAULTS`; CLI builder)?
5. Any NEW bug introduced by the refactor or the deletions? Any remaining
   substantive finding? If none, say "no substantive findings" and justify.
