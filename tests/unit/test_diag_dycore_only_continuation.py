"""Direct test for scripts/validate/diag_dycore_only_continuation.py.

The probe attributed the CAM6 arm's warming to the dynamical core, so its
argument surface is load-bearing: a silently-renamed flag would make a future
re-run measure a different thing.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "diag_dycore_only_continuation.py")


def _load():
    spec = importlib.util.spec_from_file_location("diag_dyncore_only", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_defaults_and_flags():
    a = _load().build_arg_parser().parse_args(["--config", "deck.yaml"])
    assert a.config == "deck.yaml"
    # No --restart means a cold start from the deck's own initial condition.
    assert a.restart is None
    assert a.thermo_terms is False
    assert (a.steps, a.stride, a.hot_level, a.hot_threshold) == (192, 16, 3, 300.0)
    assert a.extra == []


def test_restart_and_passthrough_flags():
    a = _load().build_arg_parser().parse_args(
        ["--config", "deck.yaml", "--restart", "ckpt.npz", "--steps", "64",
         "--stride", "8", "--hot-level", "5", "--hot-threshold", "200",
         "--thermo-terms", "--extra", "--ic-path", "/x.zarr", "--solar-file", "/s.nc"])
    assert a.restart == "ckpt.npz"
    assert a.thermo_terms is True
    assert (a.steps, a.stride, a.hot_level, a.hot_threshold) == (64, 8, 5, 200.0)
    # --extra forwards run_amip's machine-path flags verbatim.
    assert a.extra == ["--ic-path", "/x.zarr", "--solar-file", "/s.nc"]


def test_config_is_required():
    with pytest.raises(SystemExit):
        _load().build_arg_parser().parse_args([])


def test_global_masses_known_answer():
    """Mass weights: q = 1 in every layer gives the air mass of the layers;
    a vapour scaled by (1+e) scales its own mass by exactly (1+e); numbers
    are not counted as water."""
    from types import SimpleNamespace

    import numpy as np

    from legoesm import constants

    w = np.array([0.1, 0.3, 0.6])                   # layer fractions, sum 1
    sc = SimpleNamespace(layer_thickness_dp=lambda ps: np.asarray(ps)[:, None] * w)
    area = np.array([1.0e10, 2.0e10])
    ps = np.array([1.0e5, 9.0e4])
    one = np.ones((2, 3))
    qv = np.array([[1e-3, 2e-3, 3e-3], [4e-3, 5e-3, 6e-3]])
    st = SimpleNamespace(p_s=SimpleNamespace(data=ps),
                         tracers={"q_v": qv, "q_c": 0.5 * qv, "N_c": 1e8 * one,
                                  "ones": one})
    m = _load().global_masses(st, area, sc)
    air = float(np.sum(area * ps) / constants.g)
    assert m["air"] == pytest.approx(air, rel=1e-15)
    assert m["ones"] == pytest.approx(air, rel=1e-15)
    expect_qv = float(np.sum(area[:, None] * qv * ps[:, None] * w) / constants.g)
    assert m["q_v"] == pytest.approx(expect_qv, rel=1e-15)
    assert m["water"] == pytest.approx(1.5 * expect_qv, rel=1e-15)
    st.tracers["q_v"] = qv * (1 + 1e-6)
    assert global_ratio(_load().global_masses(st, area, sc)["q_v"], expect_qv) == pytest.approx(1e-6, rel=1e-9)


def global_ratio(a, b):
    return a / b - 1.0


def test_water_terms_on_a_small_mesh():
    """The per-term split on a real dycore state: horizontal transport and the
    p_s filter each change global water by rounding only (the filter's air is
    carried as dry), and the terms add up to the total."""
    from types import SimpleNamespace

    import jax.numpy as jnp
    import numpy as np

    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig)
    from legoesm.atmosphere.idealized.held_suarez_topo import held_suarez_topo_init_mpas
    from legoesm.core.field import Field
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.grids.vertical import standard_hybrid_levels
    from legoesm.grids.voronoi import create_voronoi_mesh

    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        mesh = create_voronoi_mesh(3)
        coord = standard_hybrid_levels(10)
        s = held_suarez_topo_init_mpas(mesh, coord)
        rng = np.random.default_rng(0)
        q = jnp.asarray(rng.uniform(1e-6, 1e-2, size=s.T.data.shape))
        s = s._replace(tracers={"q_v": Field(data=q, name="q_v", dims=("nCells", "nlev"))},
                       u=s.u.replace(data=jnp.asarray(rng.normal(0, 10, s.u.data.shape))))
        model = SimpleNamespace(mesh=mesh, sigma_coord=coord, config=MPASPrimitiveEquationConfig(
            fix_mass=False, nu_del4_ps=1.0e16))
        w = _load().water_tendency_terms(s, model)
    finally:
        set_policy(saved)
    assert w["total"] == pytest.approx(w["horizontal"] + w["ps_del4"] + w["rest"], abs=1e-15)
    assert abs(w["horizontal"]) < 1e-10
    assert abs(w["ps_del4"]) < 1e-10
