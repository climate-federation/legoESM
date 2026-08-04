"""Big-mesh cache-or-prewarm policy (voronoi.py, codex round-19).

The six cases codex specified: cache-hit bypass, miss rejection, opt-in
prewarm build, single-builder concurrency, stale-lock takeover, and the
hard cap. All run at tiny subdivision levels by lowering the policy
thresholds — the policy code path is identical; only the mesh is small.
"""
import os
import threading

import pytest

from legoesm.grids import voronoi as V


@pytest.fixture()
def big_at_2(monkeypatch, tmp_path):
    """Make subdiv-3 a 'big' mesh with a tmp cache dir (fast tests)."""
    monkeypatch.setattr(V, "_BIG_MESH_ROUTINE_LEVEL", 2)
    monkeypatch.setattr(V, "_BIG_MESH_MAX_LEVEL", 4)
    monkeypatch.setattr(V, "_BIG_MESH_LOCK_STALE_S", 3.0)
    monkeypatch.setenv(V._MESH_CACHE_ENV, str(tmp_path))
    monkeypatch.delenv(V._MESH_CACHE_DISABLE_ENV, raising=False)
    monkeypatch.delenv(V._BIG_MESH_BUILD_ENV, raising=False)
    return tmp_path


def test_hard_cap_refused_regardless_of_optin(big_at_2, monkeypatch):
    monkeypatch.setenv(V._BIG_MESH_BUILD_ENV, "1")
    with pytest.raises(ValueError, match="unsupported"):
        V.create_voronoi_mesh(5, lloyd_iterations=0)


def test_miss_rejected_without_optin(big_at_2):
    with pytest.raises(ValueError, match="prewarm"):
        V.create_voronoi_mesh(3, lloyd_iterations=0)


def test_optin_builds_then_hit_bypasses_policy(big_at_2, monkeypatch):
    monkeypatch.setenv(V._BIG_MESH_BUILD_ENV, "1")
    m1 = V.create_voronoi_mesh(3, lloyd_iterations=0)
    # hit path: opt-in removed, still loads (cache hit is always admissible)
    monkeypatch.delenv(V._BIG_MESH_BUILD_ENV)
    m2 = V.create_voronoi_mesh(3, lloyd_iterations=0)
    assert m1.nCells == m2.nCells


def test_disabled_cache_refused_for_big(big_at_2, monkeypatch):
    monkeypatch.setenv(V._MESH_CACHE_DISABLE_ENV, "1")
    monkeypatch.setenv(V._BIG_MESH_BUILD_ENV, "1")
    with pytest.raises(ValueError, match="cache"):
        V.create_voronoi_mesh(3, lloyd_iterations=0)


def test_single_builder_concurrency(big_at_2, monkeypatch):
    monkeypatch.setenv(V._BIG_MESH_BUILD_ENV, "1")
    results, errors = [], []

    def worker():
        try:
            results.append(V.create_voronoi_mesh(3, lloyd_iterations=0))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not errors and len(results) == 3
    assert len({m.nCells for m in results}) == 1
    # the lock must be gone afterwards
    locks = [p for p in os.listdir(big_at_2) if p.endswith(".lock")]
    assert not locks


def test_stale_lock_taken_over(big_at_2, monkeypatch):
    monkeypatch.setenv(V._BIG_MESH_BUILD_ENV, "1")
    # plant a dead builder's lock next to the would-be cache file
    cache_path = V._voronoi_cache_path(3, V.constants.R_earth, 0,
                                       V.constants.Omega)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(f"{cache_path}.lock", "w") as f:
        f.write("999999")
    with pytest.warns(UserWarning, match="stale"):
        m = V.create_voronoi_mesh(3, lloyd_iterations=0)
    assert m.nCells == 10 * 4 ** 3 + 2
