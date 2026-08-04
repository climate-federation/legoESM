"""Direct test for scripts/data/prewarm_voronoi_mesh.py (CLAUDE.md rule)."""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "data" / "prewarm_voronoi_mesh.py"


def _env(tmp_path, extra=None):
    env = dict(os.environ)
    env["LEGOESM_MESH_CACHE_DIR"] = str(tmp_path)
    env["JAX_PLATFORMS"] = "cpu"
    env.pop("LEGOESM_MESH_CACHE_DISABLE", None)
    if extra:
        env.update(extra)
    return env


def test_refuses_without_cache_dir(tmp_path):
    env = _env(tmp_path)
    del env["LEGOESM_MESH_CACHE_DIR"]
    r = subprocess.run([sys.executable, str(SCRIPT), "--level", "3"],
                       env=env, capture_output=True, text=True)
    assert r.returncode != 0
    assert "LEGOESM_MESH_CACHE_DIR" in r.stderr


def test_prewarms_small_level(tmp_path):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--level", "3", "--lloyd", "0"],
        env=_env(tmp_path), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "prewarmed level=3" in r.stdout
    assert any(p.suffix == ".npz" for p in tmp_path.rglob("*.npz"))
