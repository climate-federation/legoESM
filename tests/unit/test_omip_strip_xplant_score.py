import importlib.util
import pathlib

import numpy as np

_p = pathlib.Path(__file__).resolve().parents[2] / 'scripts/validate/ocean_fidelity/omip_strip_xplant_score.py'
_s = importlib.util.spec_from_file_location('xs', _p)
xs = importlib.util.module_from_spec(_s)
_s.loader.exec_module(xs)


def test_interior_drops_buffer_cells():
    sj, si = np.meshgrid(np.arange(13), np.arange(31), indexing='ij')
    m = xs.interior(sj.ravel(), si.ravel(), 3)
    assert m.sum() == 7 * 25 and not m[0] and m[3 * 31 + 3]


def test_share_and_rel_err():
    Tc = np.zeros((1, 12)); Tc[0, :3] = 1.0          # one 1 K step between levels 2 and 3
    assert xs.share(Tc) == 1.0
    Tc[0] = np.linspace(2.0, 0.0, 12)                  # uniform gradient: 10 equal jumps
    np.testing.assert_allclose(xs.share(Tc), 0.1)
    o = np.array([1.0, 2.0, 3.0]); n = np.array([1.0, 2.2, np.nan])
    np.testing.assert_allclose(xs.rel_err(o, n), 0.2 / 2.2)


def test_n2_sign():
    zt = np.array([0.5, 1.5, 2.5])
    T = np.array([[20.0, 19.0, 18.0]]); S = np.full((1, 3), 35.0)
    assert xs.n2(T, S, 0, zt)[0] > 0
