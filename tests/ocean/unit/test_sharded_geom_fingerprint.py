"""Unit tests for the band-geometry cross-process fingerprint gate.

The gate decides whether ``make_sharded_ocean_step`` accepts per-process
band-geometry stacks without the (removed, nd-linear-cost) process-0
broadcast — see the 2026-08-03 fix note at the sharded put. These tests
pin the gate's discrimination properties single-process (the
multicontroller allgather wiring is exercised by the distributed suite).
"""
import numpy as np
import pytest

from legoesm.parallel.geometry_consistency import (
    band_fingerprint as geom_band_fingerprint,
    band_fingerprints_agree,
)

N_BANDS = 4
SHAPE = (N_BANDS, 6, 8)


def _gather(*hosts):
    """Simulate process_allgather over per-process fingerprints."""
    fps = [geom_band_fingerprint(h, N_BANDS) for h in hosts]
    exacts = {fp[2] for fp in fps}
    assert len(exacts) == 1
    g_struct = np.stack([fp[0] for fp in fps])
    g_vals = np.stack([fp[1] for fp in fps])
    return g_struct, g_vals, fps[0][2]


def test_identical_float_stacks_agree():
    rng = np.random.default_rng(0)
    a = rng.normal(size=SHAPE).astype(np.float32)
    assert band_fingerprints_agree(*_gather(a, a.copy()))


def test_ulp_scale_drift_agrees():
    rng = np.random.default_rng(1)
    a = rng.normal(size=SHAPE).astype(np.float64) + 10.0
    b = np.nextafter(a, np.inf)  # a TRUE 1-ULP elementwise drift
    assert band_fingerprints_agree(*_gather(a, b))


def test_band_local_drift_refused_where_global_gate_passed():
    # THE r14 discrimination case: a band-local drift SMALL enough that the
    # old whole-array gate (sum/sumsq/absmax at rtol 1e-5) accepts it, with
    # the global absmax held by an UNAFFECTED band — the per-band gate must
    # still refuse.
    rng = np.random.default_rng(2)
    a = rng.normal(size=SHAPE).astype(np.float64) + 10.0
    a[0, 0, 0] = 500.0          # absmax lives in band 0
    b = a.copy()
    b[2] *= 1.0 + 3e-5          # one band drifts; global sums move ~7e-6 rel

    def _global_moments(x):
        f = x.ravel().astype(np.float64)
        return np.array([f.sum(), (f * f).sum(), np.abs(f).max()])

    # the OLD global gate would have ACCEPTED this pair...
    assert np.allclose(_global_moments(a), _global_moments(b),
                       rtol=1e-5, atol=0.0)
    # ...the per-band gate refuses it.
    assert not band_fingerprints_agree(*_gather(a, b))


def test_exact_dtype_permutation_refused():
    # Moment fingerprints are blind to permutations; the positional
    # per-band byte digest must not be.
    a = np.zeros(SHAPE, dtype=np.int32)
    a[1, 2, 3] = 1
    b = np.zeros_like(a)
    b[1, 3, 2] = 1  # same count, different position, same band
    assert not band_fingerprints_agree(*_gather(a, b))


def test_bool_mask_two_cell_flip_refused():
    a = np.zeros(SHAPE, dtype=bool)
    a[0, 0, 0] = True
    b = a.copy()
    b[0, 0, 0] = False
    b[0, 5, 7] = True  # true-count preserved
    assert not band_fingerprints_agree(*_gather(a, b))


def test_nonfinite_count_mismatch_refused():
    a = np.ones(SHAPE, dtype=np.float32)
    b = a.copy()
    b[3, 0, 0] = np.nan  # struct carries per-band non-finite counts
    assert not band_fingerprints_agree(*_gather(a, b))


def test_shape_mismatch_refused():
    a = np.ones(SHAPE, dtype=np.float32)
    b = np.ones((N_BANDS, 6, 9), dtype=np.float32)
    fa = geom_band_fingerprint(a, N_BANDS)
    fb = geom_band_fingerprint(b, N_BANDS)
    # Different shapes -> different struct lengths; the agree helper is
    # only called on stackable gathers, so assert the structs differ.
    assert fa[0].shape != fb[0].shape or not np.array_equal(fa[0], fb[0])


def test_wrong_leading_axis_raises():
    with pytest.raises(ValueError):
        geom_band_fingerprint(np.ones((3, 2)), N_BANDS)


def test_scalar_type_divergence_distinguished():
    # codex r21: python 0 and 0.0 share raw bytes on 64-bit hosts; the
    # dtype-aware leaf digest must still tell them apart.
    from legoesm.parallel.geometry_consistency import leaf_digest48

    assert leaf_digest48(0) != leaf_digest48(0.0)
    assert leaf_digest48(1.5) == leaf_digest48(1.5)
    assert leaf_digest48(np.float32(1.5)) != leaf_digest48(np.float64(1.5))


def test_replicate_pytree_single_process_unchanged():
    # codex r23: replicate_pytree's multi-process branch must not perturb
    # the single-process path (replicated_sharding=None returns the tree
    # untouched; a real single-device sharding takes the historical
    # device_put branch, byte-identical values).
    import jax.numpy as jnp
    from legoesm.parallel.mesh import DeviceConfig, replicate_pytree

    cfg = DeviceConfig(
        mesh=None, face_sharding=None, replicated_sharding=None,
        n_devices=1, backend="cpu", is_distributed=False, tiling=(1, 1),
        grid_type="voronoi", voronoi_dims=None)
    tree = {"a": jnp.arange(4.0), "b": 3}
    out = replicate_pytree(tree, cfg)
    assert np.allclose(np.asarray(out["a"]), np.arange(4.0))
    assert out["b"] == 3
