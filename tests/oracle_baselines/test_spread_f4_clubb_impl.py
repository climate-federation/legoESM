"""Direct tests of the F4 CLUBB implemented-arm harness helpers:
scripts/validate/oracle/spread_f4/clubb_impl/{build_gen,analyze_clubb_impl}.py.

build_gen.py is a module-level script with absolute source paths; it is run
unchanged via runpy with ``open`` redirected to synthetic anchor texts."""
from __future__ import annotations

import ast
import builtins
import importlib.util
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

D = Path(__file__).resolve().parents[2] / "scripts/validate/oracle/spread_f4/clubb_impl"
R2 = "/burg/glab/users/pg2328/oracle_audit/spread/clubb_core/gen_spread.py"
GEN = "/burg/glab/users/pg2328/wt_f4/scripts/validate/oracle/clubb_core/gen_clubb_core_fixture.py"
SPREAD = Path("/burg/glab/users/pg2328/oracle_audit/spread/perturb.py")

SYN_R2 = '''X = 1
# ---- SPREAD adaptation
SPREAD_MODE = None


def main():
    import clubb_f2py
    _spy = 1
    clubb_f2py.f2py_advance_clubb_core = _spy

    def spread_step(name, st, n):
        d = st
        A._advance_clubb_core_api(d)
        res[f"{name}/seed{sd}/s{n}/{k}"] = 1

    pass
'''

OLD_STEP = '''        if IMPLEMENTED:
            # _advance_clubb_core_api hard-codes l_implemented=False (the SCM
            # driver); CAM passes .true.
            orig = clubb_api.advance_clubb_core
            clubb_api.advance_clubb_core = (
                lambda *a, **k: orig(*a, **dict(k, l_implemented=True)))
            try:
                A._advance_clubb_core_api(st)
            finally:
                clubb_api.advance_clubb_core = orig
        else:
            A._advance_clubb_core_api(st)
'''

SYN_G = ('''def run():
    from m import (
        init_clubb_case, read_namelist)

    def step(st):
''' + OLD_STEP + '''
    def trajectory(name, st):
        for n in range(1, N_STEPS + 1):
            step(st)
            record(name, st, n)


def main():
    pass
''')


def _run_build_gen(tmp_path, monkeypatch, r2, g):
    texts = {R2: r2, GEN: g}
    real_open = builtins.open

    def fake_open(p, *a, **k):
        if str(p) in texts:
            q = tmp_path / Path(str(p)).name
            q.write_text(texts[str(p)])
            return real_open(q, *a, **k)
        return real_open(p, *a, **k)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(builtins, "open", fake_open)
    runpy.run_path(str(D / "build_gen.py"), run_name="__main__")
    monkeypatch.setattr(builtins, "open", real_open)
    return (tmp_path / "gen_spread_impl.py").read_text()


def test_build_gen_splices_spread_into_generator(tmp_path, monkeypatch):
    out = _run_build_gen(tmp_path, monkeypatch, SYN_R2, SYN_G)
    ast.parse(out)
    assert out.index("# ---- SPREAD adaptation") < out.index("\ndef main(")
    assert out.count("    def _advance(st):") == 1 and out.count("A._advance_clubb_core_api(st)") == 2
    assert "        _advance(st)\n" in out and "_advance(d)" in out
    assert "A._advance_clubb_core_api(d)" not in out
    assert out.index("    def spread_step(") < out.index("    def trajectory(")
    assert 'if SPREAD_MODE == "pert":\n                spread_step(name, st, n)\n            step(st)' in out
    assert out.index("clubb_f2py.f2py_advance_clubb_core = _spy") > out.index("read_namelist)")


def test_build_gen_refuses_missing_anchor(tmp_path, monkeypatch):
    with pytest.raises(AssertionError):
        _run_build_gen(tmp_path, monkeypatch, SYN_R2, SYN_G.replace("record(name, st, n)", "rec()"))
    with pytest.raises(AssertionError):
        _run_build_gen(tmp_path, monkeypatch, SYN_R2, SYN_G.replace("            try:\n", "", 1))


@pytest.mark.skipif(not (Path(R2).exists() and Path(GEN).exists()),
                    reason="real sources live on Ginsburg")
def test_build_gen_reproduces_committed_generator(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(D / "build_gen.py"), run_name="__main__")
    assert (tmp_path / "gen_spread_impl.py").read_bytes() == (D / "gen_spread_impl.py").read_bytes()


@pytest.mark.skipif(not SPREAD.exists(), reason="shared perturb.py lives on Ginsburg")
def test_analyze_bitwise_diff_and_load(tmp_path):
    spec = importlib.util.spec_from_file_location("f4_analyze_clubb", D / "analyze_clubb_impl.py")
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    x = np.array([1.0, 2.0])
    a = dict(x=x, y=np.zeros(2), s=np.array(["a"]))
    assert A.bitwise_diff(a, dict(a), ["x", "y", "s"]) == {}
    b = dict(x=np.array([1.0, np.nextafter(2.0, 3.0)]), y=np.zeros(3),
             s=np.array(["b"]), z=np.array([0.0]))
    bad = A.bitwise_diff(a, b, ["x", "y", "s"])
    assert bad["x"] == np.spacing(2.0) and isinstance(bad["y"], str) and isinstance(bad["s"], str)
    neg = dict(y=np.array([-0.0, 0.0]))     # bitwise, not value, equality
    assert A.bitwise_diff(dict(y=np.zeros(2)), neg, ["y"]) == {"y": 0.0}
    (tmp_path / "m.py").write_text("V = 7\n")
    assert A.load("m_test", str(tmp_path / "m.py")).V == 7
