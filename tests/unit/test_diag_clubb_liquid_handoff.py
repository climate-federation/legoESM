"""Direct test for scripts/validate/diag_clubb_liquid_handoff.py.

The probe measured how much of the CLUBB closure's liquid never reaches the
cloud optics, and how much of the baseline's reflectivity the diagnostic
condensate floor supplies.  Its argument surface is load-bearing: a renamed
flag would make a re-run measure a different state or a different deck.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "diag_clubb_liquid_handoff.py")


def _load():
    spec = importlib.util.spec_from_file_location("diag_clubb_handoff", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_required_flags_and_defaults():
    a = _load().build_arg_parser().parse_args(
        ["--config", "deck.yaml", "--restart", "ckpt.npz"])
    assert (a.config, a.restart) == ("deck.yaml", "ckpt.npz")
    assert a.label == "arm"
    assert a.extra == []


def test_restart_is_required():
    """Without a checkpoint there is no state to measure, so it cannot default."""
    with pytest.raises(SystemExit):
        _load().build_arg_parser().parse_args(["--config", "deck.yaml"])


def test_passthrough_flags_survive():
    a = _load().build_arg_parser().parse_args(
        ["--config", "deck.yaml", "--restart", "ckpt.npz", "--label", "CAM6",
         "--extra", "--solar-file", "/s.nc", "--resolution", "6"])
    assert a.label == "CAM6"
    assert a.extra == ["--solar-file", "/s.nc", "--resolution", "6"]


def test_closure_gets_the_host_liquid_only_when_the_partition_is_on():
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    f = _load().closure_liquid
    q_c = object()
    assert f(TurbulenceConfig(liquid_partition=True), q_c) is q_c
    assert f(TurbulenceConfig(), q_c) is None
    with pytest.raises(SystemExit, match="q_c"):
        f(TurbulenceConfig(liquid_partition=True), None)


def test_host_liquid_inventory_reads_the_host_values():
    """Section 1 integrates the liquid and cloud fraction the host receives.
    Hand-computed on two columns: layers with cf = 0 drop out, negative liquid
    clips to 0, and H counts only liquid above the host's previous q_c."""
    import numpy as np
    from legoesm import constants
    mod = _load()
    q_before = np.array([[1.0e-4, 0.0, 2.0e-4], [0.0, 3.0e-4, 0.0]])
    q_host = np.array([[3.0e-4, 5.0e-4, -1.0e-5], [2.0e-4, 1.0e-4, 4.0e-4]])
    cf = np.array([[0.5, 0.0, 0.2], [1.0, 0.3, 0.0]])
    dp = np.full((2, 3), 1.0e4)
    A = np.array([0.25, 0.75])
    out = mod.host_liquid_inventory(q_before, q_host, cf, dp, A)
    g = constants.g
    # cloudy layers: col0 k0, k2 (k2 liquid clipped to 0); col1 k0, k1
    L = (0.25 * (3.0e-4 + 0.0) + 0.75 * (2.0e-4 + 1.0e-4)) * 1.0e4 / g
    Lqc = (0.25 * (1.0e-4 + 2.0e-4) + 0.75 * (0.0 + 3.0e-4)) * 1.0e4 / g
    num = 0.25 * 2.0e-4 + 0.75 * 2.0e-4
    den = 0.25 * 3.0e-4 + 0.75 * 3.0e-4
    assert np.isclose(out["L_rcm"], L, rtol=1e-14, atol=0.0)
    assert np.isclose(out["L_qc"], Lqc, rtol=1e-14, atol=0.0)
    assert np.isclose(out["H"], num / den, rtol=1e-14, atol=0.0)
    np.testing.assert_array_equal(out["mask"], cf > 0.0)
    np.testing.assert_array_equal(out["liquid"], np.maximum(q_host, 0.0))


def test_host_after_clubb_takes_the_lane_entrys_outputs():
    """Section 1 reads the lane entry's host outputs, not clubb_step's
    post-advance diagnostics: a fake entry whose dq_c_dt / cloud_fraction are
    unrelated to any closure must come back unchanged."""
    import types
    import numpy as np
    mod = _load()
    q_c = np.array([[1.0e-4, 2.0e-4]])
    dq = np.array([[3.0e-7, -1.0e-7]])
    cf = np.array([[0.37, 0.11]])
    seen = {}

    def fake_entry(*args, q_c=None, **kw):
        seen.update(dt=args[12], q_c=q_c, kw=kw)
        return types.SimpleNamespace(dq_c_dt=dq, cloud_fraction=cf), None

    z = np.zeros((1, 2))
    ql, cfo = mod.host_after_clubb(fake_entry, q_c, 600.0, z, z, z, z, z, z, z,
                                   z, z, z[:, 0], z[:, 0], z, "cfg", lat=z[:, 0])
    np.testing.assert_array_equal(ql, q_c + 600.0 * dq)
    np.testing.assert_array_equal(cfo, cf)
    assert seen["dt"] == 600.0 and seen["q_c"] is q_c and "lat" in seen["kw"]


def test_section1_wires_the_lane_entry_into_the_inventory():
    """``main`` (the function the CLI runs) feeds section 1 from the lane entry
    through host_after_clubb into host_liquid_inventory, and no longer reads the
    core's post-advance ``rcm_grid`` diagnostic."""
    import ast
    tree = ast.parse(_SCRIPT.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)]
    hac = [c for c in calls if c.func.id == "host_after_clubb"]
    assert hac and isinstance(hac[0].args[0], ast.Name)
    assert hac[0].args[0].id == "clubb_turbulence_prognostic"
    assert any(c.func.id == "host_liquid_inventory" for c in calls)
    consts = {n.value for n in ast.walk(main) if isinstance(n, ast.Constant)}
    assert "rcm_grid" not in consts
