"""Direct tests of the F4 spread harness modules other than the token rewriter:
scripts/validate/oracle/spread_f4/{families,f4_spy,analyze}.py."""
from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

D = Path(__file__).resolve().parents[2] / "scripts/validate/oracle/spread_f4"
SPREAD = Path("/burg/glab/users/pg2328/oracle_audit/spread/perturb.py")


def _load(name, env=None, monkeypatch=None):
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    sys.path.insert(0, str(D))
    try:
        spec = importlib.util.spec_from_file_location(f"f4_{name}", D / f"{name}.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    finally:
        sys.path.remove(str(D))
    return m


def test_families_registry_is_complete():
    F = _load("families").FAMILIES
    assert {"zm_detrain", "rad_boundary", "louis", "kuo_mahfouf", "kain_fritsch", "mynn25",
            "mg2_prog_nc", "mg2_prog_nc_act", "mg2_prog_nc_sed"} <= set(F)
    for name, f in F.items():
        assert {"build", "generator", "argv", "exes", "fixture", "test", "fix_attr",
                "metrics", "select"} <= set(f), name
        assert any("{OUT}" in a for a in f["argv"]), name
        assert f["exes"] and f["metrics"], name
        assert f["generator"].endswith("make_fixture.py"), name
    assert F["mg2_prog_nc_sed"]["metrics"] == ["_rel", "_rel_flux"]
    assert F["louis"]["exclude_src"] == "ri_pin"


def test_spy_passes_through_records_and_neutralises(tmp_path, monkeypatch):
    env = dict(F4_FIX_ATTR="FIXTURE", F4_FIX_PATH=str(tmp_path / "f.npz"),
               F4_METRICS="_rel", F4_OUT=str(tmp_path / "r.pkl"))
    S = _load("f4_spy", env, monkeypatch)

    def rel(a, b, scale=1.0):
        return np.abs(np.asarray(a) - np.asarray(b)) * scale
    w = S._spy(rel, "_rel")
    S.STATE.update(node="t::x", i=0)
    out = w(np.array([1.0, 2.0]), np.array([1.0, 1.5]), 2.0)
    np.testing.assert_array_equal(out, [0.0, 1.0])          # base run: unchanged value
    r = S.REC[-1]
    assert (r["node"], r["i"], r["metric"], r["args"], r["value"]) == ("t::x", 0, "_rel", [2.0], 1.0)
    np.testing.assert_array_equal(r["oracle"], [1.0, 1.5])
    S.NEUTRAL = True                                          # seed run: array-shaped zeros
    z = w(np.array([1.0, 2.0]), np.array([1.0, 1.5]))
    assert z.shape == (2,) and not z.any() and S.REC[-1]["value"] == 0.5
    assert w(1.0, 3.0) == 0.0 and S.REC[-1]["value"] == 2.0
    S.pytest_sessionfinish(None, 0)
    d = pickle.load(open(tmp_path / "r.pkl", "rb"))
    assert d["exitstatus"] == 0 and len(d["records"]) == 3


@pytest.mark.skipif(not SPREAD.exists(), reason="shared perturb.py lives on Ginsburg")
def test_analyze_npz_equal_and_recs(tmp_path):
    A = _load("analyze")
    a, b, c = tmp_path / "a.npz", tmp_path / "b.npz", tmp_path / "c.npz"
    np.savez(a, x=np.array([1.0, 2.0]), meta=np.array("m"))
    np.savez(b, x=np.array([1.0, 2.0]), meta=np.array("m"))
    np.savez(c, x=np.array([1.0, np.nextafter(2.0, 3.0)]), y=np.zeros(1))
    assert A.npz_equal(a, b) == {}
    bad = A.npz_equal(a, c)
    assert bad["_keys"] == ["meta", "y"] and bad["x"] == np.spacing(2.0)
    pickle.dump(dict(records=[dict(node="n", i=0, line=7, value=1.0)], exitstatus=1),
                open(tmp_path / "r.pkl", "wb"))
    st, r = A.recs(tmp_path / "r.pkl")
    assert st == 1 and list(r) == [("n", 0, 7)]
