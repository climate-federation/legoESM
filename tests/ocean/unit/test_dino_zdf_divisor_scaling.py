"""Direct unit tests for the DINO zdf-divisor scaling instrument
(``scripts/validate/ocean_fidelity/dino_1226/dino_zdf_divisor_scaling.py``).

Synthetic only -- no NEMO artifacts, no model build, CPU-fast. Each test has a
synthetic-violation arm so it cannot pass vacuously.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"


@pytest.fixture(scope="module")
def mod():
    sys.path.insert(0, str(SCRIPTS_DIR / "validate" / "ocean_fidelity"
                          / "dino_1226"))
    sys.path.insert(0, str(SCRIPTS_DIR))
    import importlib
    m = importlib.import_module("dino_zdf_divisor_scaling")
    return m


def _write_mesh(path, e3w_profile, spread=0.0):
    nc = pytest.importorskip("netCDF4")
    nz, ny, nx = len(e3w_profile), 3, 4
    d = nc.Dataset(path, "w")
    d.createDimension("t", 1)
    d.createDimension("z", nz)
    d.createDimension("y", ny)
    d.createDimension("x", nx)
    v = d.createVariable("e3w_0", "f8", ("t", "z", "y", "x"))
    a = np.broadcast_to(np.asarray(e3w_profile)[:, None, None],
                        (nz, ny, nx)).copy()
    a[:, -1, -1] += spread          # the synthetic-violation knob
    v[0] = a
    d.close()


def test_nemo_e3w_ref_alignment(mod, tmp_path):
    """Entry k of the returned profile is NEMO's e3w_0(k+1): the divisor
    between cells k and k+1, dropping the surface e3w(1) the solve never uses.
    """
    prof = [1.0, 2.0, 3.0, 4.0, 5.0]
    p = tmp_path / "mesh_mask.nc"
    _write_mesh(p, prof)
    mod._E3W_CACHE.clear()
    out = mod.nemo_e3w_ref(str(p), 5)
    assert out.shape == (4,)
    np.testing.assert_allclose(out, [2.0, 3.0, 4.0, 5.0], rtol=0, atol=0)
    # a model with fewer levels takes the leading slice, still offset by one
    mod._E3W_CACHE.clear()
    np.testing.assert_allclose(mod.nemo_e3w_ref(str(p), 3), [2.0, 3.0])


def test_nemo_e3w_ref_rejects_non_uniform_mesh(mod, tmp_path):
    """The 1-D reduction is only valid on a full-step ln_zco mesh; a mesh with
    horizontal structure must be REFUSED, not silently averaged away."""
    p = tmp_path / "mesh_mask.nc"
    _write_mesh(p, [1.0, 2.0, 3.0], spread=0.5)
    mod._E3W_CACHE.clear()
    with pytest.raises(SystemExit, match="not horizontally uniform"):
        mod.nemo_e3w_ref(str(p), 3)
    # and the same mesh WITHOUT the planted spread is accepted (non-vacuity)
    p2 = tmp_path / "mesh_uniform.nc"
    _write_mesh(p2, [1.0, 2.0, 3.0], spread=0.0)
    mod._E3W_CACHE.clear()
    assert mod.nemo_e3w_ref(str(p2), 3).shape == (2,)


def test_nemo_e3w_ref_rejects_short_mesh(mod, tmp_path):
    p = tmp_path / "mesh_mask.nc"
    _write_mesh(p, [1.0, 2.0])
    mod._E3W_CACHE.clear()
    with pytest.raises(SystemExit, match="levels, model has"):
        mod.nemo_e3w_ref(str(p), 5)


def test_stats_reports_signed_mean_and_abs_extremes(mod):
    rel = np.array([[-0.02, 0.01, 0.03]])
    mask = np.array([[True, True, False]])
    s = mod._stats(rel, mask, "t")
    assert s["n"] == 2
    assert s["max_abs"] == pytest.approx(0.02)      # abs, not signed max
    assert s["mean"] == pytest.approx(-0.005)       # signed, keeps the bias
