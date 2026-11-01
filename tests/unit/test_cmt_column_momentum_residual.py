"""Direct test for scripts/validate/cmt_column_momentum_residual.py (#2266).

The gate must hold on a conserving tendency, FLAG a column whose momentum is
deliberately broken, refuse a vacuous (all-zero) on-arm tendency, and, through
the real MPAS Emanuel bridge, see a nonzero conserving FU/FV with CMT on and
exact zeros with it off.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "cmt_column_momentum_residual.py")


def _load():
    spec = importlib.util.spec_from_file_location("cmt_resid", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _conserving(ncol=5, nlev=12, seed=0):
    rng = np.random.default_rng(seed)
    dp = 500.0 + 3000.0 * rng.random((ncol, nlev))
    f = rng.standard_normal((ncol, nlev))
    f -= (f * dp).sum(1, keepdims=True) / dp.sum(1, keepdims=True)
    return f, dp


def test_flags():
    m = _load()
    a = m.build_arg_parser().parse_args(["--config", "d.yaml", "--restart", "c.npz", "--expect", "on"])
    assert a.expect == "on" and a.json is None
    with pytest.raises(SystemExit):
        m.build_arg_parser().parse_args(["--config", "d.yaml", "--restart", "c.npz"])


def test_conserving_tendency_passes_and_broken_column_is_flagged():
    m = _load()
    f, dp = _conserving()
    assert m.violations(f, dp).size == 0
    bad = f.copy()
    bad[3, 4] *= 1.001                       # one level of one column broken
    assert list(m.violations(bad, dp)) == [3]
    ctrl, i = m.broken_control(f, dp)
    assert list(m.violations(ctrl, dp)) == [i]


def test_zero_column_passes_and_is_not_active():
    m = _load()
    f, dp = _conserving()
    f[1] = 0.0
    assert m.violations(f, dp).size == 0
    assert m.summarize(f, dp)["n_active"] == f.shape[0] - 1


def test_gate_on_requires_a_nonzero_tendency():
    m = _load()
    f, dp = _conserving()
    ok, rep = m.gate(f, -f, dp, "on")
    assert ok and rep["FU"]["control_flagged"] and rep["FV"]["n_violations"] == 0
    ok, rep = m.gate(np.zeros_like(f), np.zeros_like(f), dp, "on")
    assert not ok and rep["FU"]["n_active"] == 0
    ok, rep = m.gate(f, np.zeros_like(f), dp, "on")          # zonal-only response is legal
    assert ok and rep["FU"]["control_flagged"] and "control_flagged" not in rep["FV"]
    bad = f.copy(); bad[0] += 1e-3
    assert not m.gate(bad, f, dp, "on")[0]
    assert not m.gate(None, None, dp, "on")[0]
    nan = np.full_like(f, np.nan)
    assert not m.gate(f, nan, dp, "on")[0]                    # NaN component cannot hide


def test_gate_off_requires_exact_zero_edges_and_no_scheme_tendency():
    m = _load()
    f, dp = _conserving()
    assert m.gate(None, None, dp, "off", edge_du=np.zeros((7, 3)))[0]
    assert not m.gate(None, None, dp, "off", edge_du=np.full((7, 3), 1e-300))[0]
    assert not m.gate(f, f, dp, "off", edge_du=np.zeros((7, 3)))[0]
    assert not m.gate(None, None, dp, "off", edge_du=None)[0]


@pytest.mark.parametrize("cmt", [True, False])
def test_real_mpas_emanuel_bridge(cmt):
    """The spy captures what the production MPAS bridge hands on, and the
    shipped scheme passes the gate on the convecting Voronoi fixture."""
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig, EmanuelConfig
    from legoesm.grids.voronoi import create_voronoi_mesh
    from tests.unit.test_mpas_cmt_wiring import _state
    m = _load()
    mesh = create_voronoi_mesh(3, lloyd_iterations=3)
    state, sigma = _state(mesh)
    cfg = ConvectionConfig(scheme="emanuel", mpas_cmt=cmt, emanuel=EmanuelConfig(enable_cmt=cmt))
    prog = None
    for _ in range(8):   # spin the cloud-base mass flux up, as the wiring test does
        ps = None if prog is None else types.SimpleNamespace(
            conv_prog_profile=jnp.asarray(prog), conv_stoch_state=jnp.zeros((mesh.nCells,)))
        fu, fv, p_half, edge_du, out, prog_in = m.capture_bridge(state, mesh, sigma, cfg, 600.0, ps)
        if prog is not None:   # the scheme saw the carry we fed, bit for bit
            np.testing.assert_array_equal(prog_in, np.asarray(prog))
        prog = out["conv_prog_profile"]
    from legoesm.atmosphere.physics.convection import integration as integ
    assert integ.emanuel_convection.__name__ == "emanuel_convection"   # spy removed
    dp = p_half[:, 1:] - p_half[:, :-1]
    ok, rep = m.gate(fu, fv, dp, "on" if cmt else "off", edge_du=edge_du)
    print(rep)
    assert ok, rep
    if cmt:
        assert rep["FU"]["n_active"] > 0 and np.abs(edge_du).max() > 0.0
        # the same capture with one column broken must fail the gate
        bad, _ = m.broken_control(fu, dp)
        assert not m.gate(bad, fv, dp, "on")[0]
