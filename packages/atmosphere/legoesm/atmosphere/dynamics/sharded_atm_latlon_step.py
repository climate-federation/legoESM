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
