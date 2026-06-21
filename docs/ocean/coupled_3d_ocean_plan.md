# Plan: wire the 3D `OceanModel` into the coupled driver

**Status (2026-06-15):** NOT started. The coupled driver (`CoupledESMDriver`,
driven by `scripts/run/run_coupled.py`) runs a **thermodynamic slab ocean
only**. The `--ocean two_layer` default (mixed + deep layer, bulk vertical
mixing + deep restoring) is the most ocean physics the coupled *slab* supports.
Real ocean physics (KPP/Richardson vertical mixing, convective adjustment, EOS,
baroclinic + barotropic dynamics, prognostic salinity + currents, isopycnal/GM,
tidal mixing) lives in the **standalone** `OceanModel`
(`packages/ocean/legoesm/ocean/dynamics/ocean_model.py`) and is **never stepped
by the coupler**. This doc plans that integration (the deferred coupled-gap #1).

## Current state (anchors)

- `CoupledESMDriver._init_ocean` (`coupled_esm_driver.py:106-142`)
  unconditionally calls `make_ocean(cfg.ocean_config)` + `init_slab_state(...)`
  — slab only. No branch to `OceanModel`.
- `CoupledESMDriver._step_ocean` (`coupled_esm_driver.py:509-538`) steps the
  slab; the docstring (510-531) explicitly **drops the ice→ocean
  freshwater/salt/stress channels** and returns `u_sfc = v_sfc = 0` (no
  prognostic currents, no salinity).
- `CoupledConfig.ocean_mode` (`coupled_config.py:57`) advertises a `"dynamic"`
  (3D) value in its docstring (`coupled_config.py:26-29`) but the field is
  **never dispatched** — it is only a decorative log label (`_OCEAN_MODE_LABEL`,
  `coupled_config.py:116-119`). `preset_complexity` raises `ValueError` if the
  ocean rung is `OceanComplexity.FULL_3D` (`coupled_config.py:155-162`) because
  `CoupledConfig` can only hold a `SimpleOceanConfig`.
- A standalone 3D-ocean factory **does** exist:
  `component_factory.create_ocean_component` (`component_factory.py:605-702`,
  `OceanConfig → OceanModel`) and `resolve_model_complexity` `'full' →
  OceanConfig` (`component_factory.py:870-918`) — but no coupled driver calls
  `OceanModel.step`.

## Hard parts / risks

1. **Config plumbing.** `CoupledConfig` holds a `SimpleOceanConfig`. Need a
   union (`SimpleOceanConfig | OceanConfig`) + an `_init_ocean` branch that
   builds an `OceanModel` (via `create_ocean_component`) when an `OceanConfig`
   / `ocean_mode="dynamic"` is supplied. Wire `ocean_mode="dynamic"` into the
   `_OCEAN_MODE_LABEL` map and stop `preset_complexity` raising on `FULL_3D`.
2. **Grid coupling.** The atmosphere is cubed-sphere; the production ocean
   (OMIP work) lives on lat-lon / eORCA tripole / cube C-D grids. The coupler
   `grid_remap` (`coupler/grid_remap.py`, gaps #1/#2) already handles
   same-grid identity + lat-lon↔lat-lon + same-family conservative remap, but
   **cross-family cube-atm ↔ tripole-ocean remap and differentiable vector
   (u,v) basis rotation are still deferred** — both are prerequisites for a
   cube-atm + tripole-ocean CMIP config.
3. **Ice→ocean channels.** `_step_ocean` currently drops freshwater/salt/stress.
   The 3D ocean needs them (surface buoyancy + momentum forcing) for a closed,
   conservative coupling. Thread `AtmToSurface` + ice fluxes through.
4. **Time-stepping cadence.** The OMIP ocean uses split-explicit barotropic +
   **implicit** vertical mixing at `dt≈3600 s` (see `[[omip_nemo_recipe]]`,
   `[[omip_latlon_75lev_solved]]`); the atm runs `dt≤450 s`. Need an
   ocean-substep / accumulate-atm-flux coupling loop, not lockstep.
5. **State + checkpoint size.** 3D ocean state (T,S,u,v × nlev × ncol) is large;
   the coupled checkpoint (`_CKPT_VERSION`, ocean-grid-shape validation already
   added) must persist + validate it. Differentiability (`jax.grad`) must flow
   through `OceanModel.step` as it already does standalone.
6. **OMIP-session overlap (coordination).** The OMIP session owns the ocean
   dycore + its uncommitted files (`barotropic_implicit_mpas.py`,
   `mpas_config.py`). Do the coupler-side wiring on the shared branch with
   explicit pathspecs and coordinate so the ocean-internal edits do not collide.

## Phased approach (each phase independently validatable)

- **Phase 1 — same-grid, no remap.** Step `OceanModel` inside
  `CoupledESMDriver` on the SAME grid as the atmosphere (cubed-sphere C-D
  ocean), aquaplanet, no ice. Validates the `_init_ocean`/`_step_ocean` branch,
  the flux hand-off (`AtmToSurface`), the ocean substep loop, and `jax.grad`
  end-to-end without any regridding. Acceptance: stable multi-day coupled run,
  SST evolves from ocean dynamics (not a slab), conservation diagnostics close.
- **Phase 2 — cross-grid remap.** Add cube-atm ↔ lat-lon/tripole-ocean
  conservative remap + differentiable vector rotation in `grid_remap.py`;
  couple a cube atm to a lat-lon ocean. Acceptance: flux area-integral conserved
  to ~1e-10, equivariance tests pass, `jax.grad` flows across the remap.
- **Phase 3 — ice→ocean + conservation.** Thread freshwater/salt/stress; close
  the coupled water + salt + energy budgets with the sea-ice model.
- **Phase 4 — CMIP realism.** Cube atm (full physics + rrtmgp, this session) +
  eORCA tripole ocean (OMIP recipe), historical forcing; compare to CMIP6.

## Quick interim already shipped

`--ocean two_layer` (default) gives a deep reservoir that damps SST drift —
real-er than the pure 50 m slab, zero coupler changes (state already carries
`T_deep`, `make_ocean` dispatches `two_layer`). Use `--ocean slab` / `--ocean
fixed` for the cheaper modes. See `[[cmip6_coupled_infra_session]]`.
