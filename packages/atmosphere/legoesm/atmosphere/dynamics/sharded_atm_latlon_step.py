"""Single-process lat-band SPMD step for the lat-lon C-grid hydrostatic atm.

The atmosphere analogue of ``ocean.dynamics.sharded_ocean_step``: wrap
``CGridLatLonPrimitiveEquationModel`` in a ``jax.shard_map`` over a 1-D ``"lat"``
device mesh so the dycore runs multi-GPU/TPU with pure ``ppermute``/``psum``
collectives (no mpi4jax). It REUSES the shared grid-agnostic primitives
(``latlon_spmd`` band perms / halo body / pole masks, ``reconstruct_vface_lower``
/ ``to_vface_lower`` for the staggered v, ``batch_psum_spmd`` /
``_spmd_lat_psum_or_none`` for global reductions, ``latlon_mpi`` band-geometry
slicers) and writes NEW only the atm-specific state layout + geometry-band glue.

Built in stages (each independently sbatch-validated on CPU host devices):
  * Stage 2 (THIS): ``shard_state_atm_latlon`` / ``gather_state_atm_latlon`` —
    the 6-field C-grid state layout with the staggered-v ``v_lower`` round-trip.
  * Stage 3-4: SPMD-aware Coriolis + v-face interp operators; the un-jitted
    band body with ``grid=`` threading.
  * Stage 5: ``make_sharded_atm_latlon_step`` + the serial-vs-SPMD equivalence
    gate.

The staggered meridional velocity ``v`` has leading dim ``n_lat+1`` (coprime
with ``n_lat`` for ``N>1``), so it is carried sharded as ``v_lower = v[:n_lat]``
and reconstructed to the full faces inside the shard_map (see
:func:`legoesm.parallel.latlon_spmd.reconstruct_vface_lower`). All other leaves
have leading dim ``n_lat`` and shard ``P("lat")`` directly; longitude is kept
local (periodic). No land/u/v masks (the atm domain is global).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from legoesm.atmosphere.dynamics.primitive_eq_latlon_cgrid import (
    CGridLatLonHydrostaticState,
)


def _lat_spec(arr) -> P:
    """``P("lat", None, ...)`` for an array sharded on its leading (lat) axis."""
    return P("lat", *((None,) * (arr.ndim - 1)))


def shard_state_atm_latlon(
    state: CGridLatLonHydrostaticState, mesh,
) -> CGridLatLonHydrostaticState:
    """Lay out a C-grid hydrostatic atm state for the lat-band shard_map.

    Cell/u leaves (``u, T, p_s, phis`` + every tracer; all leading-dim
    ``n_lat``) shard ``P("lat", None, ...)``. The staggered ``v`` (leading dim
    ``n_lat+1``) drops its top pole-wall face -> ``v_lower = v[:n_lat]``
    (``n_lat`` rows, divisible by the device count) sharded the same way; the
    dropped face is the north pole wall (``v[n_lat] == 0`` after any step) and
    is reconstructed inside the body. Mirrors ``ocean.shard_state_latlon`` but
    walks the 6-field atm pytree (bare arrays + a tracers dict, no masks).
    """
    def _put(arr):
        return jax.device_put(arr, NamedSharding(mesh, _lat_spec(arr)))

    n_lat = state.T.shape[0]
    v_lower = state.v[:n_lat]
    return state._replace(
        u=_put(state.u),
        v=_put(v_lower),
        T=_put(state.T),
        p_s=_put(state.p_s),
        phis=_put(state.phis),
        tracers={k: _put(val) for k, val in state.tracers.items()},
    )


def gather_state_atm_latlon(
    state: CGridLatLonHydrostaticState, mesh,
) -> CGridLatLonHydrostaticState:
    """Inverse of :func:`shard_state_atm_latlon`: replicate every leaf and
    rebuild the full ``(n_lat+1, ...)`` ``v`` by re-appending the zero north
    pole-wall face. Bit-comparable to the single-device state (whose top v-face
    is the pole wall == 0)."""
    rep = NamedSharding(mesh, P())

    def _get(arr):
        return jax.device_put(arr, rep)

    v_lower = _get(state.v)
    v_full = jnp.concatenate([v_lower, jnp.zeros_like(v_lower[:1])], axis=0)
    return state._replace(
        u=_get(state.u),
        v=v_full,
        T=_get(state.T),
        p_s=_get(state.p_s),
        phis=_get(state.phis),
        tracers={k: _get(val) for k, val in state.tracers.items()},
    )


# LatLonGrid scalar fields that stay STATIC per band (uniform bands -> one
# shard_map program); every OTHER LatLonGrid field is a jax.Array and is
# stacked over the band axis + indexed by axis_index in the body.
_ATM_GRID_STATIC_FIELDS = frozenset({"n_lat", "n_lon", "radius", "dlon", "dlat"})


def _atm_grid_array_field_names(grid) -> list[str]:
    """Order-stable list of the ``jax.Array`` fields of a ``LatLonGrid`` (all
    except the static scalars). NamedTuple field order, so the host stack and
    the in-body ``[r]`` index agree."""
    import numpy as np
    names = []
    for name in grid._fields:
        if name in _ATM_GRID_STATIC_FIELDS:
            continue
        val = getattr(grid, name)
        if isinstance(val, (jax.Array, np.ndarray)):
            names.append(name)
    return names


def _build_band_grids_atm(grid, n_devices: int):
    """Build the ``n_devices`` UNIFORM lat-band ``LatLonGrid`` geometries via the
    tested MPI slicer (no bespoke metric re-derivation).

    ``skip_total_area_reduce=True`` keeps ``total_area`` the GLOBAL full-sphere
    sum on EVERY band (the mass-fixer denominator must stay global; the band
    slicer would otherwise ``global_sum_mpi`` a band-local area — wrong / errors
    without mpi4py). Requires ``n_lat % n_devices == 0`` so all bands share the
    leading shape (one shard_map program). rank 0 = south band, N-1 = north.
    """
    from legoesm.parallel.latlon_mpi import (
        make_latlon_band_layout, slice_latlon_grid_to_band)
    n_lat = int(grid.n_lat)
    if n_devices < 1:
        raise ValueError(f"n_devices must be >= 1, got {n_devices}")
    if n_lat % n_devices != 0:
        raise ValueError(
            f"atm lat-band SPMD requires n_lat ({n_lat}) divisible by "
            f"n_devices ({n_devices}) so every band is uniform (one shard_map "
            f"program). Pick n_devices among the divisors of {n_lat}.")
    fold = getattr(grid, "fold", None)
    return [
        slice_latlon_grid_to_band(
            grid,
            make_latlon_band_layout(r, n_devices, n_lat, int(grid.n_lon), fold),
            skip_total_area_reduce=True)
        for r in range(n_devices)
    ]


def make_sharded_atm_latlon_step(model, mesh):
    """Return ``step(c_state, dt) -> c_state`` running the C-grid hydrostatic atm
    step lat-band-SPMD over the 1-D ``"lat"`` mesh.

    ``c_state`` is a ``CGridLatLonHydrostaticState`` laid out with
    :func:`shard_state_atm_latlon` (``v`` carried as the ``n_lat``-row
    ``v_lower``). The body reconstructs each band's ``nl+1`` v-faces, runs the
    UN-jitted ``model._step_cgrid_impl`` on the band geometry + band polar masks
    + per-band pole masks, and converts the result ``v`` back to ``v_lower``.
    REUSES the shared primitives (band perms, the v-face round-trip, the band
    halo via the swapped backend, ``spmd_pole_end_masks``, the band slicer);
    atm-NEW is only the 6-field state/geometry walk. ``check_vma=False`` (the
    band halo reads neighbour-rank data). Mirrors ``make_sharded_ocean_step``.
    """
    from legoesm.parallel.latlon_spmd import (
        latlon_band_perms, reconstruct_vface_lower, to_vface_lower,
        spmd_pole_end_masks, activate_latlon_spmd_halo)
    from legoesm.parallel.shard_map_compat import shard_map

    if mesh is None:                       # single-device: plain C-grid step
        return lambda c_state, dt: model._step_cgrid(c_state, dt)[0]

    n_dev = mesh.devices.size
    axis = mesh.axis_names[0]
    grid = model.grid

    # Dispatch-hardening: only a non-fold lat-lon grid with an SPMD-safe mass
    # path is supported. Fail LOUD rather than silently mis-fold / band-local-sum.
    fold = getattr(grid, "fold", None)
    if fold is not None and bool(getattr(fold, "is_active", False)):
        raise NotImplementedError(
            "atm lat-band SPMD: tripole north-fold is a follow-up.")
    if getattr(model.config, "anchor_mass_to_initial", False):
        raise NotImplementedError(
            "atm lat-band SPMD: anchor_mass_to_initial uses a band-local "
            "jnp.sum(p_s*area) target that is not yet SPMD-routed; disable it "
            "or use fix_mass with the pre-state (psum'd) path.")

    band_grids = _build_band_grids_atm(grid, n_dev)
    template = band_grids[0]
    array_field_names = _atm_grid_array_field_names(template)
    rep = NamedSharding(mesh, P())
    stacks = {
        name: jax.device_put(
            jnp.stack([jnp.asarray(getattr(g, name)) for g in band_grids],
                      axis=0), rep)
        for name in array_field_names
    }
    # Per-band polar-filter masks (only when the filter is on): slice the global
    # masks [s:e] (cell) / [s:e+1] (v-face stagger) and stack. Absent otherwise
    # -> the body passes None -> _step_cgrid_impl resolves to self._polar_mask
    # (also None), no filter.
    nl = int(grid.n_lat) // n_dev
    if model._polar_mask is not None:
        stacks["__polar_mask"] = jax.device_put(jnp.stack(
            [model._polar_mask[r * nl:(r + 1) * nl] for r in range(n_dev)],
            axis=0), rep)
        stacks["__polar_mask_v"] = jax.device_put(jnp.stack(
            [model._polar_mask_v[r * nl:r * nl + nl + 1] for r in range(n_dev)],
            axis=0), rep)

    perm_north, _perm_south = latlon_band_perms(n_dev)
    dt_closure = [None]

    def _body(state_local, stacks_local):
        r = jax.lax.axis_index(axis)
        band_geom = template._replace(
            **{name: stacks_local[name][r] for name in array_field_names})
        pmask = (stacks_local["__polar_mask"][r]
                 if "__polar_mask" in stacks_local else None)
        pmaskv = (stacks_local["__polar_mask_v"][r]
                  if "__polar_mask_v" in stacks_local else None)
        # Reconstruct the band's nl+1 v-faces from v_lower (shared interface row
        # via ppermute), run the un-jitted band step, convert v back to v_lower.
        v_full = reconstruct_vface_lower(state_local.v, axis, perm_north)
        state_band = state_local._replace(v=v_full)
        out, _ = model._step_cgrid_impl(
            state_band, dt_closure[0],
            grid=band_geom, sigma_coord=model.sigma_coord,
            polar_mask=pmask, polar_mask_v=pmaskv,
            pole_v_bc_masks=spmd_pole_end_masks(),
        )
        return out._replace(v=to_vface_lower(out.v))

    def sharded_step(c_state, dt):
        dt_closure[0] = dt
        in_spec = jax.tree.map(_lat_spec, c_state)
        stacks_spec = jax.tree.map(lambda _x: P(), stacks)  # all replicated
        fn = shard_map(
            _body, mesh=mesh, in_specs=(in_spec, stacks_spec),
            out_specs=in_spec, check_vma=False)
        # Arm the SPMD band halo around the call ONLY; save+restore the FULL
        # backend state (a later serial/full-domain call must not take SPMD-only
        # branches outside a shard_map). Per-call re-trace bakes the band halo.
        from legoesm.grids.halo import (
            get_halo_backend, get_mpi_topology, get_spmd_mesh,
            set_halo_backend, set_spmd_mesh)
        _prev_backend = get_halo_backend()
        _prev_topo = get_mpi_topology()
        _prev_mesh = get_spmd_mesh()
        activate_latlon_spmd_halo(mesh)
        try:
            return fn(c_state, stacks)
        finally:
            set_spmd_mesh(_prev_mesh)
            set_halo_backend(_prev_backend, _prev_topo)

    return sharded_step
