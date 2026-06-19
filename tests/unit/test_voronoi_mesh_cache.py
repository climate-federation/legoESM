"""Tests for the Voronoi (MPAS) mesh disk cache.

The SCVT build (Lloyd relaxation) is deterministic in
``(subdivision_level, radius, lloyd_iterations, omega)`` when ``density_fn`` is
None, but slow (~10^2 s at production resolution).  Under MPI every rank rebuilds
the same mesh, which at high rank-count contends for cores and stalls the job.
``create_voronoi_mesh`` therefore caches the built mesh to a ``.npz`` keyed on
those args and loads it on a hit.  These tests exercise: round-trip fidelity, the
on-disk hit path, the env switches, schema-drift invalidation, corruption
resilience, x64 keying, and the pre-warm helper.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from legoesm.grids.voronoi import (  # white-box: test reaches into module internals
    VoronoiMesh,
    _load_voronoi_cache,
    _save_voronoi_cache,
    _voronoi_cache_disabled,
    _voronoi_cache_path,
    create_voronoi_mesh,
    prewarm_voronoi_cache,
)

# Small + few Lloyd iters => fast build, still a real full mesh.
_LEVEL = 2
_LLOYD = 2


def _mesh_fields_equal(a: VoronoiMesh, b: VoronoiMesh) -> None:
    """Assert two meshes are identical field-by-field (exact for the cache round-trip)."""
    assert a._fields == b._fields
    for name in a._fields:
        va, vb = getattr(a, name), getattr(b, name)
        if hasattr(va, "shape") and getattr(va, "shape", ()) != ():
            np.testing.assert_array_equal(
                np.asarray(va), np.asarray(vb),
                err_msg=f"field {name!r} differs")
        else:
            assert va == vb, f"scalar field {name!r}: {va!r} != {vb!r}"


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    """Point the mesh cache at an isolated temp dir and ensure it is enabled."""
    d = tmp_path / "voronoi_cache"
    monkeypatch.setenv("LEGOESM_MESH_CACHE_DIR", str(d))
    monkeypatch.delenv("LEGOESM_MESH_CACHE_DISABLE", raising=False)
    return d


def test_build_then_hit_round_trips(cache_dir):
    """First call builds + writes the .npz; second call loads an identical mesh."""
    path = _voronoi_cache_path(_LEVEL, 1.0, _LLOYD, 1e-4)
    assert not os.path.exists(path)

    m1 = create_voronoi_mesh(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
    assert os.path.exists(path), "build did not write the cache file"

    m2 = create_voronoi_mesh(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
    _mesh_fields_equal(m1, m2)


def test_loaded_mesh_matches_uncached(cache_dir, monkeypatch):
    """A cache-loaded mesh equals the same mesh built with the cache disabled."""
    cached = create_voronoi_mesh(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)

    monkeypatch.setenv("LEGOESM_MESH_CACHE_DISABLE", "1")
    assert _voronoi_cache_disabled()
    fresh = create_voronoi_mesh(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
    _mesh_fields_equal(cached, fresh)


def test_disable_env_skips_write(cache_dir, monkeypatch):
    """With the cache disabled, no file is written and nothing is loaded."""
    monkeypatch.setenv("LEGOESM_MESH_CACHE_DISABLE", "1")
    path = _voronoi_cache_path(_LEVEL, 1.0, _LLOYD, 1e-4)
    create_voronoi_mesh(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
    assert not os.path.exists(path), "cache file written despite DISABLE=1"


def test_key_separates_params(cache_dir):
    """Distinct mesh args map to distinct cache files (no collisions)."""
    paths = {
        _voronoi_cache_path(_LEVEL, 1.0, _LLOYD, 1e-4),
        _voronoi_cache_path(_LEVEL + 1, 1.0, _LLOYD, 1e-4),
        _voronoi_cache_path(_LEVEL, 2.0, _LLOYD, 1e-4),
        _voronoi_cache_path(_LEVEL, 1.0, _LLOYD + 1, 1e-4),
        _voronoi_cache_path(_LEVEL, 1.0, _LLOYD, 2e-4),
    }
    assert len(paths) == 5


def test_schema_drift_invalidates(cache_dir):
    """A cache whose stored field set != VoronoiMesh._fields is ignored (returns None)."""
    path = _voronoi_cache_path(_LEVEL, 1.0, _LLOYD, 1e-4)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save an .npz with the wrong field set.
    with open(path, "wb") as f:
        np.savez(f, nCells=np.asarray(42), bogus=np.zeros(3))
    assert _load_voronoi_cache(path) is None


def test_corrupt_cache_returns_none(cache_dir):
    """A truncated/garbage file warns and returns None rather than raising."""
    path = _voronoi_cache_path(_LEVEL, 1.0, _LLOYD, 1e-4)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"not a real npz file")
    with pytest.warns(RuntimeWarning):
        assert _load_voronoi_cache(path) is None


def test_missing_cache_returns_none(cache_dir):
    """A non-existent path is a clean miss (None), not an error."""
    assert _load_voronoi_cache(str(cache_dir / "does_not_exist.npz")) is None


def test_save_is_atomic_no_tmp_left(cache_dir):
    """After a successful save, only the final file remains (no .tmp.<pid> debris)."""
    m = create_voronoi_mesh(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
    path = _voronoi_cache_path(_LEVEL, 1.0, _LLOYD, 1e-4)
    # Re-save explicitly and confirm no temp files linger in the dir.
    _save_voronoi_cache(path, m)
    leftovers = [p for p in os.listdir(os.path.dirname(path)) if ".tmp." in p]
    assert leftovers == [], f"temp files left behind: {leftovers}"


def test_x64_keyed_separately(cache_dir):
    """The cache key encodes the active x64 flag so precisions never cross-pollute."""
    import jax

    x64 = bool(jax.config.read("jax_enable_x64"))
    path = _voronoi_cache_path(_LEVEL, 1.0, _LLOYD, 1e-4)
    assert f"_x64{int(x64)}.npz" in os.path.basename(path)


def test_prewarm_creates_cache(cache_dir):
    """prewarm_voronoi_cache builds + caches and returns the existing path."""
    path = prewarm_voronoi_cache(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
    assert os.path.exists(path)
    assert path == _voronoi_cache_path(_LEVEL, 1.0, _LLOYD, 1e-4)


def test_prewarm_refuses_when_disabled(cache_dir, monkeypatch):
    """Pre-warming a disabled cache is a hard error (no silent no-op)."""
    monkeypatch.setenv("LEGOESM_MESH_CACHE_DISABLE", "1")
    with pytest.raises(RuntimeError, match="disabled cache"):
        prewarm_voronoi_cache(_LEVEL, radius=1.0, lloyd_iterations=_LLOYD, omega=1e-4)
