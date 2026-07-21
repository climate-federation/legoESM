"""download_clm_surfdata must probe local/shared-FS candidates before any
network I/O, and a failed download must raise an actionable error — the bare
UCAR-SVN fetch (SSL hostname-mismatch as of 2026-07) killed whole SLURM jobs at
startup when /tmp was empty on a fresh node (#869/#847 probe jobs)."""
from __future__ import annotations

import pytest

from legoesm.land import clm_surface_map as clm


def test_cache_hit_short_circuits(tmp_path):
    cache = tmp_path / "clm_surfdata.nc"
    cache.write_bytes(b"x")
    assert clm.download_clm_surfdata(str(cache)) == str(cache)


def test_env_override_wins(tmp_path, monkeypatch):
    f = tmp_path / "my_surfdata.nc"
    f.write_bytes(b"x")
    monkeypatch.setenv("LEGOESM_CLM_SURFDATA", str(f))
    # cache path does NOT exist -> env candidate is returned, no network
    assert clm.download_clm_surfdata(str(tmp_path / "missing.nc")) == str(f)


def _isolate_repo_root(tmp_path, monkeypatch):
    """Point the module's repo-root probe at an empty fake repo (hermetic: the
    real checkout may legitimately have data/clm/surfdata_*.nc).  Mirrors the
    real depth: <root>/packages/land/legoesm/land/clm_surface_map.py.
    Returns the fake repo root."""
    root = tmp_path / "fake_repo"
    fake = root / "packages" / "land" / "legoesm" / "land" / "clm_surface_map.py"
    fake.parent.mkdir(parents=True)
    monkeypatch.setattr(clm, "__file__", str(fake))
    monkeypatch.chdir(tmp_path)  # cwd (= tmp_path, NOT root) glob finds nothing
    return root


def test_candidates_order_and_filtering(tmp_path, monkeypatch):
    _isolate_repo_root(tmp_path, monkeypatch)
    monkeypatch.setenv("LEGOESM_CLM_SURFDATA", "/some/override.nc")
    cands = clm._surfdata_candidates("/tmp/c.nc")
    assert cands == ["/tmp/c.nc", "/some/override.nc"]  # order; empties/globs gone


def test_repo_data_clm_probed(tmp_path, monkeypatch):
    # cwd != fake repo root, so ONLY the __file__-derived probe can find it.
    root = _isolate_repo_root(tmp_path, monkeypatch)
    monkeypatch.delenv("LEGOESM_CLM_SURFDATA", raising=False)
    f = root / "data" / "clm" / "surfdata_test.nc"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    assert clm.download_clm_surfdata(str(tmp_path / "missing.nc")) == str(f)


def test_download_failure_raises_actionable(tmp_path, monkeypatch):
    import urllib.request

    def _boom(url, dst):
        raise OSError("SSL: CERTIFICATE_VERIFY_FAILED")

    _isolate_repo_root(tmp_path, monkeypatch)
    monkeypatch.delenv("LEGOESM_CLM_SURFDATA", raising=False)
    monkeypatch.setattr(urllib.request, "urlretrieve", _boom)
    with pytest.raises(RuntimeError) as ei:
        clm.download_clm_surfdata(str(tmp_path / "missing.nc"))
    msg = str(ei.value)
    assert "LEGOESM_CLM_SURFDATA" in msg and "--clm-surfdata-path" in msg