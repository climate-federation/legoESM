"""Photon-sharded parallelism for the 3D Monte-Carlo ray tracer.

The tracer parallelises by SHARDING PHOTONS, not the domain (see docs/specs):
each rank/device replicates the (cheap, per-g-point) optical field and traces an
independent photon shard with a distinct RNG offset; the per-cell tallies are
then combined by an AVERAGE of the per-shard fractions — equivalently an
``allreduce-SUM`` of the counts divided by the photon total. SUM is the one
AD-safe reduction (matches ``global_sum_mpi``), so this combine survives the
Phase-B differentiable path. This avoids cross-domain photon migration entirely
and scales near-linearly; variance ~ 1/sqrt(N_total).

Three entry points:
- ``solve_sw_sharded`` — serial simulation of P photon shards in one process
  (also a memory lever: K shards of P/K photons). Combine = average.
- ``pmean_result`` — combine across a ``pmap``/``shard_map`` device axis via
  ``jax.lax.pmean`` (AD-safe). Each device runs the per-shard solve, then calls
  this on its local result.
- MPI deployment: each rank runs the local solve and combines its ``MC3DResult``
  fields with ``legoesm`` ``global_sum_mpi`` (allreduce SUM) divided by the rank
  count — identical math to ``pmean``.
"""

from __future__ import annotations

from typing import TypeAlias

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics.radiation.mc3d import raytracer_sw
from legoesm.atmosphere.physics.radiation.mc3d.config import MC3DRadiationConfig
from legoesm.atmosphere.physics.radiation.mc3d.photon_walk import PlaneRTGeometry
from legoesm.atmosphere.physics.radiation.mc3d.tally import MC3DResult

Array: TypeAlias = jax.Array


def shard_keys(base_key: Array, n_shards: int) -> Array:
  """Distinct per-shard PRNG keys (one per rank/device) from a base key."""
  return jax.random.split(base_key, n_shards)


def solve_sw_sharded(
    k_ext: Array,
    ssa: Array,
    g: Array,
    geom: PlaneRTGeometry,
    *,
    mu0,
    azimuth: float,
    albedo,
    config: MC3DRadiationConfig,
    key: Array,
    n_shards: int,
) -> MC3DResult:
  """Trace ``n_shards`` independent photon shards and combine by average.

  Serial (one-process) simulation of photon-sharded parallelism: ``n_shards``
  shards each launch ``config.photons_per_pixel`` photons/column with a distinct
  key, so the combined estimate uses ``n_shards * photons_per_pixel`` total —
  same variance as one big run, lower peak memory. The distributed version
  replaces this Python loop with one local solve per rank + ``pmean_result`` /
  ``global_sum_mpi``.
  """
  if n_shards < 1:
    raise ValueError(f"n_shards must be >= 1, got {n_shards}.")
  keys = shard_keys(key, n_shards)

  def _shard(i):
    return raytracer_sw.solve_sw_monochromatic(
        k_ext, ssa, g, geom, mu0=mu0, azimuth=azimuth, albedo=albedo,
        config=config, key=keys[i])

  # Fold-accumulate the running sum (NOT a list of all K results + a stacked
  # copy): peak memory is one result + one accumulator, independent of n_shards.
  # Each shard's WALK only materializes its own (smaller) photon state, so this
  # is a genuine memory lever, not just variance reduction.
  acc = _shard(0)
  for i in range(1, n_shards):
    acc = jax.tree.map(jnp.add, acc, _shard(i))
  inv = 1.0 / n_shards
  return jax.tree.map(lambda x: x * inv, acc)


def pmean_result(result: MC3DResult, axis_name: str) -> MC3DResult:
  """Combine per-device photon shards across a ``pmap``/``shard_map`` axis.

  Call on each device's local ``MC3DResult``; ``jax.lax.pmean`` (= psum / n) is
  AD-safe, so gradients flow through the combine. Equivalent to the MPI
  ``global_sum_mpi(...) / n_ranks`` path.
  """
  return jax.tree.map(lambda x: jax.lax.pmean(x, axis_name), result)
