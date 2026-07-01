"""Tally a batch of terminated photons into flux/absorption fields.

Because the walk is analog (each photon ends in exactly one bin), tallying is a
single ``jax.ops.segment_sum`` per bin type — no in-loop scatter-add. Out-of-range
segment ids (``-1`` for photons that did not terminate in that bin) are ignored
by ``segment_sum``, so the masking is free.
"""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import (
    PlaneRTGeometry,
    STATUS_MAXITER,
    STATUS_SFC_ABS,
    STATUS_TOD_UP,
    STATUS_VOL_ABS,
    PhotonState,
)

Array: TypeAlias = jax.Array

_INT32_MAX = 2**31 - 1


def assert_count_capacity(n_total: int, *, x64_enabled: bool) -> None:
  """Refuse loudly if the integer photon tally would overflow.

  Counts are accumulated as integers for exact conservation. Under x64 the
  accumulator is a true int64 (safe to 2^63). With x64 OFF, ``jnp.int64``
  silently downcasts to int32, so a domain with > 2^31-1 photons would wrap
  silently and return wrong fluxes -- raise instead (no silent coerce).
  """
  if (not x64_enabled) and n_total > _INT32_MAX:
    raise ValueError(
        f"Total photons N={n_total} exceeds int32 capacity ({_INT32_MAX}) and "
        "JAX x64 is disabled, so the integer count tally would overflow. Enable "
        "JAX_ENABLE_X64=1, or reduce nx*ny*photons_per_pixel."
    )


class MC3DResult(NamedTuple):
  """Monochromatic MC tally, normalised by photons-per-column ``P``.

  All ``*_frac`` fields are dimensionless fractions of the launched photons:
  multiply by the incident flux (``F0 * mu0`` for SW) to get W/m^2.
  """

  abs_frac: Array        # (nx,ny,nz) volume-absorbed fraction (/P)
  sfc_abs_frac: Array    # (nx,ny) surface-absorbed fraction (/P)
  tod_up_frac: Array     # scalar TOD-escape fraction incl. max-iter (/N_total)
  vol_abs_total: Array   # scalar domain volume-absorbed fraction (/N_total)
  sfc_abs_total: Array   # scalar domain surface-absorbed fraction (/N_total)
  maxiter_frac: Array    # scalar diagnostic: subset of tod_up_frac (/N_total)
  # Exact budget: vol_abs_total + sfc_abs_total + tod_up_frac == 1.


def tally_batch(
    states: PhotonState,
    geom: PlaneRTGeometry,
    photons_per_column: int,
) -> MC3DResult:
  """Reduce a flat batch of terminal photon states to an ``MC3DResult``."""
  status = states.status
  n_total = status.shape[0]
  ncells = geom.nx * geom.ny * geom.nz
  ncols = geom.nx * geom.ny
  # Integer counts -> exact conservation regardless of photon count.
  assert_count_capacity(n_total, x64_enabled=jax.config.jax_enable_x64)
  count_dt = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
  out_dt = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
  ones = jnp.ones((n_total,), dtype=count_dt)

  vol_ids = jnp.where(status == STATUS_VOL_ABS, states.vol_idx, -1)
  vol_counts = jax.ops.segment_sum(ones, vol_ids, num_segments=ncells)
  vol_counts = vol_counts.reshape(geom.nx, geom.ny, geom.nz)

  sfc_ids = jnp.where(status == STATUS_SFC_ABS, states.sfc_idx, -1)
  sfc_counts = jax.ops.segment_sum(ones, sfc_ids, num_segments=ncols)
  sfc_counts = sfc_counts.reshape(geom.nx, geom.ny)

  tod_up = jnp.sum(status == STATUS_TOD_UP).astype(count_dt)
  maxiter = jnp.sum(status == STATUS_MAXITER).astype(count_dt)
  # Fold max-iter photons into escaped so the physical budget closes exactly;
  # maxiter_frac stays a diagnostic subset of tod_up_frac.
  escaped = tod_up + maxiter

  inv_p = 1.0 / photons_per_column
  inv_n = 1.0 / n_total
  return MC3DResult(
      abs_frac=vol_counts.astype(out_dt) * inv_p,
      sfc_abs_frac=sfc_counts.astype(out_dt) * inv_p,
      tod_up_frac=escaped.astype(out_dt) * inv_n,
      vol_abs_total=jnp.sum(vol_counts).astype(out_dt) * inv_n,
      sfc_abs_total=jnp.sum(sfc_counts).astype(out_dt) * inv_n,
      maxiter_frac=maxiter.astype(out_dt) * inv_n,
  )
