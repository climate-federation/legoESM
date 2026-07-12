"""Deterministic NCCL comm-init warmup for the closed-loop tiled cube (#921).

The closed-loop blocked step fuses, in ONE executable, the halo
collective-permute cliques AND the mass-fixer ``psum`` all-reduce clique over
the SAME ``(face, tile_i, tile_j)`` axes.  On multi-process GPU the two clique
KINDS can be scheduled for NCCL comm-init in a different relative order per rank
and deadlock; :func:`warmup_tiled_cube_comms` breaks that by priming each clique
kind in isolation, in a fixed order, before the first real step.

This gate cannot reproduce the cross-PROCESS NCCL race (single process; the CPU
backend has no NCCL).  It proves, on the 24 virtual-device mesh, that (1) the
warmup self-skips single-process runs, (2) it executes under ``force=True`` on
the real tiled mesh, (3) the mesh-shape guard fires, and (4) the two warmup
executables cleanly split into the permute-clique and reduce-clique kinds the
step fuses — the isolation the fix depends on.

Run with ``XLA_FLAGS=--xla_force_host_platform_device_count=24``.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=24")

from functools import partial

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from legoesm.parallel.shard_map_compat import shard_map
from legoesm.parallel.tiled_production_cdgrid import warmup_tiled_cube_comms
from legoesm.parallel.cubesphere_exchange import (
    _get_tiled_tables, _tiled_diag_perms, _tiled_guard_perms,
)

KT = 2
AXES = ("face", "tile_i", "tile_j")


def _mesh(kt=KT):
    if len(jax.devices()) < 6 * kt * kt:
        pytest.skip(f"needs {6 * kt * kt} devices "
                    f"(XLA_FLAGS=--xla_force_host_platform_device_count).")
    dev = np.array(jax.devices()[:6 * kt * kt]).reshape(6, kt, kt)
    return Mesh(dev, axis_names=AXES)


def test_warmup_self_skips_single_process():
    # process_count() == 1 in the test harness -> no cross-process rendezvous,
    # so the warmup is a no-op (single-device / CPU-virtual runs unaffected).
    assert warmup_tiled_cube_comms(_mesh(), KT) is False


def test_warmup_runs_under_force():
    # force=True exercises the real warmup executables on the tiled mesh: if any
    # perm table or axis name were wrong the ppermute/psum would raise here.
    assert warmup_tiled_cube_comms(_mesh(), KT, force=True) is True


def test_warmup_rejects_wrong_mesh_shape():
    bad = Mesh(np.array(jax.devices()[:6]).reshape(6, 1, 1),
               axis_names=AXES)
    with pytest.raises(ValueError, match=r"\(6, kt, kt\)"):
        warmup_tiled_cube_comms(bad, KT, force=True)


def test_warmup_executables_split_the_two_clique_kinds():
    """The halo executable issues ONLY collective-permutes; the reduce
    executable issues ONLY an all-reduce.  Each NCCL comm therefore inits in
    isolation (the whole point) — replicated here with the SAME perms/axes the
    warmup uses."""
    mesh = _mesh()
    perms = (list(_get_tiled_tables(KT).perms)
             + list(_tiled_guard_perms(KT)) + list(_tiled_diag_perms(KT)))
    dummy = jax.device_put(np.zeros((6, KT, KT), np.float32),
                           NamedSharding(mesh, P(*AXES)))

    @partial(shard_map, mesh=mesh, in_specs=P(*AXES), out_specs=P(*AXES),
             check_vma=False)
    def _halo_warm(x):
        v = x.reshape((1,))
        acc = v
        for perm in perms:
            acc = acc + jax.lax.ppermute(v, AXES, perm)
        return acc.reshape((1, 1, 1))

    @partial(shard_map, mesh=mesh, in_specs=P(*AXES), out_specs=P(*AXES),
             check_vma=False)
    def _reduce_warm(x):
        v = x.reshape((1,))
        return (v + jax.lax.psum(v, axis_name=AXES)).reshape((1, 1, 1))

    halo = jax.jit(_halo_warm).lower(dummy).compile().as_text()
    red = jax.jit(_reduce_warm).lower(dummy).compile().as_text()

    def _cp(h):
        return sum(1 for ln in h.splitlines()
                   if "collective-permute" in ln and "done" not in ln)

    assert _cp(halo) > 0 and "all-reduce" not in halo
    assert "all-reduce" in red and _cp(red) == 0
