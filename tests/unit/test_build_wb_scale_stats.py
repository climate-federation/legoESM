"""Direct tests for scripts/data/build_wb_scale_stats.py."""
import importlib.util
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_P = Path(__file__).resolve().parents[2] / "scripts/data/build_wb_scale_stats.py"
_spec = importlib.util.spec_from_file_location("wb_scale_stats_under_test", _P)
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)

C = namedtuple("C", "T u v q_v p_s phis")
NL, NO, NZ = 3, 4, 5


def _grid():
    return SimpleNamespace(weights=np.array([0.2, 0.5, 0.3]), n_lon=NO,
                           lat=np.array([-0.8, 0.0, 0.8]),
                           lon=np.arange(NO) * 2 * np.pi / NO)


def _rec(i):
    r = np.random.default_rng(i)
    c = C(*(r.normal(size=(NL, NO, NZ)) for _ in range(4)),
          r.normal(size=(NL, NO)), r.normal(size=(NL, NO)))
    s = {k: r.normal(size=(NL, NO)) for k in ("t2m", "u10", "v10", "mslp")}
    land = np.array([[0.0, 0.4, 0.5, 1.0]] * NL)
    f = {"land_frac": land.ravel(), "soil_moisture": r.normal(size=NL * NO),
         "seconds_of_day": 21600.0 * (i % 4)}
    return c, s, f


def _wstd(xs, w):
    x = np.stack(xs)
    w = np.broadcast_to(w, x.shape)
    m = np.average(x, weights=w, axis=tuple(range(3)))
    return np.sqrt(np.average((x - m) ** 2, weights=w, axis=tuple(range(3))))


def test_accumulate_matches_a_direct_weighted_std():
    g = _grid()
    pairs = [(0, 1), (1, 2), (2, None), (5, 6)]
    diag = lambda c: {"t2m": c.T[..., -1], "u10": c.u[..., -1], "v10": c.v[..., -1]}  # noqa: E731
    out = st.accumulate(_rec, pairs, g, diag)
    a = st._area(g)
    d = [(_rec(j)[0].T - _rec(i)[0].T) for i, j in pairs if j is not None]
    np.testing.assert_allclose(out["loss"]["T_resid_scale_lev"],
                               _wstd(d, a[:, :, None]), rtol=1e-12)
    dm = [_rec(j)[1]["mslp"] - _rec(i)[1]["mslp"] for i, j in pairs if j is not None]
    np.testing.assert_allclose(out["loss"]["mslp_resid_scale"], _wstd(dm, a), rtol=1e-12)
    ins = [_rec(i)[0].q_v[..., -3:] for i, _ in pairs]
    np.testing.assert_allclose(out["head_inputs"]["q_v"]["std"],
                               _wstd(ins, a[:, :, None]), rtol=1e-12)
    land = (_rec(0)[2]["land_frac"].reshape(NL, NO) >= 0.5)
    sm = [_rec(i)[2]["soil_moisture"].reshape(NL, NO) for i, _ in pairs]
    np.testing.assert_allclose(out["head_inputs"]["soil_moisture"]["std"],
                               _wstd(sm, a * land), rtol=1e-12)
    gap = [_rec(i)[1]["u10"] - _rec(i)[0].u[..., -1] for i, _ in pairs]
    np.testing.assert_allclose(out["head_outputs"]["u10_gap_std"], _wstd(gap, a), rtol=1e-12)
    assert len(out["loss"]["q_resid_scale_lev"]) == NZ


def test_pairs_stay_in_the_years_and_need_exact_6h_successors():
    times = (np.datetime64("2014-12-31T12") + np.arange(12) * np.timedelta64(6, "h")
             ).astype("datetime64[ns]")
    times[8] += np.timedelta64(1, "h")                 # a non-6 h step
    store = [0, 1, 2, 3, 4, 6, 7, 8]                   # 5 missing
    got = st.pairs_in_years(store, times, years=(2015, 2015))
    # 0, 1 are 2014; 4 has no cached successor; 7 -> 8 is 7 h.
    assert got == [(2, 3), (3, 4), (4, None), (6, 7), (7, None), (8, None)]
