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


@pytest.mark.parametrize("n, nproc", [(24, 32), (24, 5), (7, 2)])
def test_validation_windows_that_do_not_divide_raise(monkeypatch, n, nproc):
    """User decision 2026-11-09: the balanced split would drop held-out
    windows (24 over 32 ranks scored NONE), so it must refuse, naming both
    counts. Training samples keep drop_remainder (tests above)."""
    full = [(2014, i, i + 4) for i in range(n)]
    monkeypatch.setattr(scale_build, "_training_sample_indices",
                        lambda *a, **k: iter(full))
    yml = {scale_build._VAL_INITS_KEY: [[2014, 0]]}
    for rank in (0, nproc - 1):
        with pytest.raises(ValueError, match=rf"{n} validation windows .* "
                                             rf"over {nproc} ranks"):
            scale_build._sharded_indices(None, yml, None, None, None,
                                         rank, nproc)


@pytest.mark.parametrize("n, nproc", [(24, 8), (24, 24), (24, 1)])
def test_validation_windows_that_divide_are_all_scored(monkeypatch, n, nproc):
    full = [(2014, i, i + 4) for i in range(n)]
    monkeypatch.setattr(scale_build, "_training_sample_indices",
                        lambda *a, **k: iter(full))
    yml = {scale_build._VAL_INITS_KEY: [[2014, 0]]}
    seen = [x for r in range(nproc) for x in scale_build._sharded_indices(
        None, yml, None, None, None, r, nproc)]
    assert seen == full


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


def _val_yml(**val):
    return {scale_build._VAL_INITS_KEY: [[2014, 0]],
            "validation": {"years": [2014], "n_inits": 24,
                           "init_stride_hours": 360, **val}}


def test_scoring_ranks_split_every_window_exactly_once(monkeypatch):
    """User decision 2026-11-09 (v9: 24 windows, 32-rank job, 8 scoring
    ranks): ranks 0-7 hold 3 windows each, ranks 8-31 none, and the union is
    the full list once, in order."""
    full = [(2014, i, i + 4) for i in range(24)]
    monkeypatch.setattr(scale_build, "_training_sample_indices",
                        lambda *a, **k: iter(full))
    yml = _val_yml(scoring_ranks=8)
    shards = [scale_build._sharded_indices(None, yml, None, None, None, r, 32)
              for r in range(32)]
    assert [len(x) for x in shards] == [3] * 8 + [0] * 24
    assert [x for sh in shards for x in sh] == full


def test_the_v9_validation_deck_keeps_its_scoring_ranks(monkeypatch):
    """The key must survive into the derived validation deck the loader sees
    (wb_validation_yaml), or the 32-rank job would pass the preflight and die
    at load."""
    import yaml

    deck = yaml.safe_load(open("config/wb/campaign/spectral_t63_ngcm_v9.yaml"))
    val = scale_build.wb_validation_yaml(deck, 24)
    full = [(2014, i, i + 4) for i in range(24)]
    monkeypatch.setattr(scale_build, "_training_sample_indices",
                        lambda *a, **k: iter(full))
    lens = [len(scale_build._sharded_indices(None, val, None, None, None, r, 32))
            for r in range(32)]
    assert lens == [3] * 8 + [0] * 24


def test_column_sharded_processes_each_score_every_window(monkeypatch):
    """Column mode is ONE logical trainer: its processes pass their physical
    rank with nproc=1 and must all hold the full window list (codex P1)."""
    full = [(2014, i, i + 4) for i in range(24)]
    monkeypatch.setattr(scale_build, "_training_sample_indices",
                        lambda *a, **k: iter(full))
    for rank in (0, 1, 3):
        assert scale_build._sharded_indices(
            None, _val_yml(), None, None, None, rank, 1) == full


@pytest.mark.parametrize("key, nproc, msg", [
    (8, 4, r"scoring_ranks 8 exceeds the job's 4 ranks"),
    (8, 1, r"scoring_ranks 8 exceeds the job's 1 ranks"),
    (5, 32, r"24 validation windows do not divide evenly over "
            r"validation.scoring_ranks 5"),
    (0, 32, r"positive integer"),
    (True, 32, r"positive integer"),
])
def test_scoring_ranks_contract_is_enforced(key, nproc, msg):
    """Voted rule: the key must divide the windows and fit the job; a smaller
    job is an error, never a silent fallback."""
    with pytest.raises(ValueError, match=msg):
        scale_build.validation_scoring_ranks(_val_yml(scoring_ranks=key),
                                             24, nproc)


def test_without_the_key_every_rank_scores():
    assert scale_build.validation_scoring_ranks(_val_yml(), 24, 8) == 8
    assert scale_build.validation_scoring_ranks(_val_yml(), 24, 1) == 1
    with pytest.raises(ValueError, match=r"24 validation windows .* over 32 ranks"):
        scale_build.validation_scoring_ranks(_val_yml(), 24, 32)
