"""#1286: the WB-scale ERA5 loader shards at build (fix B) and is
host-resident (fix A) instead of parking the whole training set on the GPU.

These are the invariants that make the fix correct WITHOUT changing training
semantics; the actual OOM relief is an integration property (fewer device
arrays alive), exercised on the cluster.
"""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.training import scale_build
from legoesm.training.data_parallel import shard_samples


# ---------------------------------------------------------------------------
# Fix B: shard-at-build partition == the old shard_samples(build_all()) split
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 6, 7, 100, 101])
@pytest.mark.parametrize("nproc", [1, 2, 3, 4, 8])
def test_sharded_indices_matches_shard_samples(monkeypatch, n, nproc):
    """``_sharded_indices`` must hand each rank BYTE-IDENTICALLY the slice the
    old ``shard_samples(build_all(), rank, nproc)`` produced — else the fix
    would silently change which samples a rank trains on."""
    full = [(1979, i, i + 4) for i in range(n)]
    monkeypatch.setattr(scale_build, "_training_sample_indices",
                        lambda *a, **k: iter(full))

    # Union over ranks + per-rank equality with the reference splitter.
    seen = []
    for rank in range(nproc):
        got = scale_build._sharded_indices(
            None, None, None, None, None, rank, nproc)
        ref = shard_samples(full, rank, nproc)
        assert got == ref, (n, nproc, rank)
        seen.extend(got)

    # drop_remainder: the union is the first nproc*per of the global list,
    # disjoint, order-preserved (the data-parallel contract).
    per = (len(full) // nproc) if nproc > 1 else len(full)
    assert seen == full[: (per * nproc if nproc > 1 else len(full))]


def test_single_process_returns_full_list(monkeypatch):
    full = [(1979, i, i + 4) for i in range(10)]
    monkeypatch.setattr(scale_build, "_training_sample_indices",
                        lambda *a, **k: iter(full))
    assert scale_build._sharded_indices(
        None, None, None, None, None, 0, 1) == full


def test_shards_are_disjoint_and_balanced(monkeypatch):
    full = [(1979, i, i + 4) for i in range(23)]
    monkeypatch.setattr(scale_build, "_training_sample_indices",
                        lambda *a, **k: iter(full))
    nproc = 4
    shards = [scale_build._sharded_indices(None, None, None, None, None, r, nproc)
              for r in range(nproc)]
    lens = {len(s) for s in shards}
    assert lens == {23 // nproc}                      # balanced (drop_remainder)
    flat = [x for s in shards for x in s]
    assert len(flat) == len(set(flat))                # disjoint


# ---------------------------------------------------------------------------
# Fix A: _sample_to_host yields host numpy that round-trips through device_put
# ---------------------------------------------------------------------------

def test_sample_to_host_is_numpy_and_roundtrips():
    import jax
    import jax.numpy as jnp

    # A (ic-like, target-like, forcing-dict) pytree of device arrays.
    sample = (
        {"a": jnp.arange(6.0), "b": jnp.ones((2, 3))},
        jnp.asarray([1.0, 2.0, 3.0]),
        {"T_sfc": jnp.zeros(4), "day_of_year": jnp.asarray(5.0)},
    )
    host = scale_build._sample_to_host(sample)

    # Every array leaf is now a HOST numpy array (nothing left on device).
    leaves = jax.tree_util.tree_leaves(host)
    assert leaves, "no leaves"
    for leaf in leaves:
        assert isinstance(leaf, np.ndarray), type(leaf)
        assert not isinstance(leaf, jax.Array)

    # device_put (what the training loop does per batch) restores the values.
    back = jax.device_put(host)
    for a, b in zip(jax.tree_util.tree_leaves(sample),
                    jax.tree_util.tree_leaves(back)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_loop_device_puts_host_samples():
    """The training loop must accept host-numpy samples (device_put per batch)
    and produce finite losses — a smoke of the fix-A consume path."""
    import jax
    import jax.numpy as jnp
    import optax

    from legoesm.training.data_parallel import mpi_data_parallel_training_loop

    # Trivial 1-param model; samples are HOST numpy (as the loader now returns).
    params = jnp.asarray(0.0)
    host_samples = [(np.asarray(1.0, dtype=np.float64),
                     np.asarray(2.0, dtype=np.float64)) for _ in range(3)]

    def loss_fn(p, sample):
        x, y = sample                      # device arrays after the loop's put
        # assert the loop moved them onto the device before calling us
        assert isinstance(x, jax.Array)
        return (p * x - y) ** 2

    opt = optax.sgd(0.01)
    opt_state = opt.init(params)
    params, opt_state, hist = mpi_data_parallel_training_loop(
        loss_fn, params, opt_state, opt, host_samples, n_epochs=2,
        num_processes=1)
    assert len(hist) == 2
    assert all(np.isfinite(h) for h in hist)
