"""Pressure-surface T-damping options in the ML-training builders + run_aimip.

``latlon_ah_pressure_surfaces`` / ``spectral_t_damping_pressure_surfaces``
(DycoreConfig / run_amip names; user decision 2026-11-24) reach:

* the functional spectral training rollouts (``spectral_rollout`` /
  ``spectral_amip_rollout``): hyperdiffusion via ``spectral_pe_tendencies`` and
  the post-step filter via the model class's shared correction;
* the WB scale trainer (``scale_build``: spectral ``_spectral_pe_config``,
  lat-lon ``build_latlon_config``), ``run_aimip`` (spectral suite key),
  ``run_aimip_latlon`` (CLI) and ``train_neural_gcm_spectral`` (CLI).

Default OFF everywhere, each refused on the core that does not honour it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest



@pytest.fixture(autouse=True, scope="module")
def _x64():
    """x64 for the spectral transforms, scoped to this module's tests."""
    prev = bool(jax.config.jax_enable_x64)
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)

_ROOT = Path(__file__).resolve().parents[2]
_LL, _SP = "latlon_ah_pressure_surfaces", "spectral_t_damping_pressure_surfaces"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# shared deck parser
# --------------------------------------------------------------------------

def test_deck_flags_default_off_and_refusals():
    from legoesm.training.campaign_driver import pressure_surface_flags
    assert pressure_surface_flags({}, "spectral", where="t") == {_LL: False, _SP: False}
    with pytest.raises(ValueError, match="must be a bool"):   # explicit null
        pressure_surface_flags({_SP: None}, "latlon", where="t")
    assert pressure_surface_flags({_SP: True}, "spectral", where="t")[_SP] is True
    assert pressure_surface_flags({_LL: True}, "latlon", where="t")[_LL] is True
    with pytest.raises(ValueError, match=_LL):
        pressure_surface_flags({_LL: True}, "spectral", where="t")
    with pytest.raises(ValueError, match=_SP):
        pressure_surface_flags({_SP: True}, "latlon", where="t")
    with pytest.raises(ValueError, match="must be a bool"):
        pressure_surface_flags({_SP: "false"}, "spectral", where="t")


# --------------------------------------------------------------------------
# functional spectral rollouts
# --------------------------------------------------------------------------

_N_MAX, _N_LEV, _DT = 10, 4, 900.0


@pytest.fixture(scope="module")
def spec_setup():
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        isothermal_rest_state_spectral,
    )
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    grid = create_gaussian_grid(_N_MAX, dealiasing="quadratic")
    sigma = create_sigma_coordinate(_N_LEV)
    rest = isothermal_rest_state_spectral(grid, sigma, T_init=280.0, p_s_init=1.0e5)
    filt = jnp.exp(-1e-2 * (jnp.asarray(grid.ls, jnp.float64) / _N_MAX) ** 8)
    return grid, sigma, rest, filt


def _with_terrain(state, grid, amp=0.02):
    """lnps with non-mean structure at degree 6 (where the filter acts:
    terrain-like p_s), a stratified T."""
    idx = int(np.argmax(np.asarray(grid.ls) == 6))
    lnps = state.lnps_hat.data.at[idx].add(amp)
    T = state.T_hat.data.at[0].add(jnp.linspace(-40.0, 0.0, _N_LEV) * np.sqrt(4 * np.pi))
    return state._replace(lnps_hat=state.lnps_hat.replace(data=lnps),
                          T_hat=state.T_hat.replace(data=T))


def _zero_physics(state, grid_, sigma_, **_kw):
    return jax.tree.map(jnp.zeros_like, state)


def _pe(**kw):
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    return SpectralPEConfig(hyperdiff_coeff=1e15, time_integrator="ssp_rk3", **kw)


def test_rollout_filter_matches_the_model_class(spec_setup):
    """The functional rollout's post-step filter correction is the model
    class's (one implementation): same state, same filter -> same T."""
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPrimitiveEquationModel,
    )
    from legoesm.training.neural_gcm_spectral import _make_state_filter
    grid, sigma, rest, _ = spec_setup
    state = _with_terrain(rest, grid)
    cfg = _pe(T_damping_pressure_surfaces=True, spectral_filter_strength=0.01,
              spectral_filter_order=8)
    model = SpectralPrimitiveEquationModel(grid=grid, sigma_coord=sigma, config=cfg)
    ref = model._apply_spectral_filter(state)
    got = _make_state_filter(cfg, grid, sigma, model._spectral_filter)(state)
    np.testing.assert_array_equal(np.asarray(got.T_hat.data), np.asarray(ref.T_hat.data))
    off = _make_state_filter(cfg._replace(T_damping_pressure_surfaces=False),
                             grid, sigma, model._spectral_filter)(state)
    assert not np.array_equal(np.asarray(off.T_hat.data), np.asarray(got.T_hat.data))
    # flat p_s: L(pi) = 0, so the correction vanishes (to round-off)
    flat = _with_terrain(rest, grid, amp=0.0)
    f_on = _make_state_filter(cfg, grid, sigma, model._spectral_filter)(flat)
    f_off = _make_state_filter(cfg._replace(T_damping_pressure_surfaces=False),
                               grid, sigma, model._spectral_filter)(flat)
    np.testing.assert_allclose(np.asarray(f_on.T_hat.data),
                               np.asarray(f_off.T_hat.data), rtol=0, atol=1e-9)


@pytest.mark.parametrize("gated", [False, True])
def test_rollout_on_differs_over_terrain(spec_setup, gated):
    from legoesm.training.neural_gcm_spectral import spectral_rollout
    grid, sigma, rest, filt = spec_setup
    kw = {"rad_physics_fn": _zero_physics, "rad_update_interval": 2} if gated else {}

    def run(state, on):
        return spectral_rollout(state, _zero_physics, grid, sigma,
                                _pe(T_damping_pressure_surfaces=on), _DT, 3,
                                None, filt, **kw)
    hill = _with_terrain(rest, grid)
    d = np.abs(np.asarray(run(hill, True).T_hat.data - run(hill, False).T_hat.data))
    assert d.max() > 1e-6


def _phys_state_fn():
    """Zero physics carrying the phys-state-thread markers (third body of
    ``spectral_rollout``); same marker contract as
    test_spectral_phys_state_thread."""
    def fn(state, grid_, sigma_, **_kw):
        return jax.tree.map(jnp.zeros_like, state)

    def fn_with_state(state, grid_, sigma_, phys_state, forcing=None):
        return jax.tree.map(jnp.zeros_like, state), phys_state
    fn.with_phys_state = fn_with_state
    fn.init_phys_state = lambda ncol, nlev, dtype=None: jnp.asarray(0.0)
    return fn


@pytest.mark.parametrize("body", ["ungated", "gated", "phys_state", "amip"])
def test_every_rollout_body_corrects_its_filter(spec_setup, body):
    """Each post-step filter call site applies the correction on its own:
    with a negligible (but > 0) hyperdiffusion the tendency-side correction
    is ~0, so ON - OFF is the filter correction alone."""
    from legoesm.training.neural_gcm_spectral import (
        spectral_amip_rollout, spectral_rollout,
    )
    grid, sigma, rest, filt = spec_setup
    hill = _with_terrain(rest, grid)

    class _PhysStub:
        sfclay = None

        def _replace(self, **_kw):
            return self

    def run(on):
        pe = _pe(T_damping_pressure_surfaces=on)._replace(hyperdiff_coeff=1e-30)
        if body == "amip":
            return spectral_amip_rollout(
                hill, _zero_physics, _zero_physics, grid, sigma, pe, _DT, 3,
                sst_col=jnp.zeros(grid.n_lat * grid.n_lon),
                sizing_phys_state=_PhysStub(), rad_update_interval=2,
                spectral_filter=filt)
        phys = _phys_state_fn() if body == "phys_state" else _zero_physics
        kw = ({"rad_physics_fn": _zero_physics, "rad_update_interval": 2}
              if body in ("gated", "phys_state") else {})
        return spectral_rollout(hill, phys, grid, sigma, pe, _DT, 3, None,
                                filt, **kw)
    on, off = run(True), run(False)
    assert np.all(np.isfinite(np.asarray(on.T_hat.data)))
    assert np.abs(np.asarray(on.T_hat.data - off.T_hat.data)).max() > 1e-6


def test_amip_rollout_honours_the_flag(spec_setup):
    from legoesm.training.neural_gcm_spectral import spectral_amip_rollout
    grid, sigma, rest, filt = spec_setup
    hill = _with_terrain(rest, grid)

    class _PhysStub:   # the rollout reads .sfclay and _replace's one field
        sfclay = None

        def _replace(self, **_kw):
            return self

    def run(on):
        return spectral_amip_rollout(
            hill, _zero_physics, _zero_physics, grid, sigma,
            _pe(T_damping_pressure_surfaces=on), _DT, 3,
            sst_col=jnp.zeros(grid.n_lat * grid.n_lon),
            sizing_phys_state=_PhysStub(), rad_update_interval=2,
            spectral_filter=filt)
    d = np.abs(np.asarray(run(True).T_hat.data - run(False).T_hat.data))
    assert d.max() > 1e-6 and np.all(np.isfinite(np.asarray(run(True).T_hat.data)))


def test_rollout_refuses_inert_or_two_level(spec_setup):
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.neural_gcm_spectral import _make_spectral_integrator
    grid, sigma, _, _ = spec_setup
    with pytest.raises(ValueError, match="hyperdiff_coeff <= 0"):
        _make_spectral_integrator(
            _pe(T_damping_pressure_surfaces=True)._replace(hyperdiff_coeff=0.0),
            grid, sigma, _DT, "ssp_rk3")
    with pytest.raises(ValueError, match="nlev >= 3"):
        _make_spectral_integrator(_pe(T_damping_pressure_surfaces=True), grid,
                                  create_sigma_coordinate(2), _DT, "ssp_rk3")
    with pytest.raises(ValueError, match="implicit_hyperdiff"):
        _make_spectral_integrator(
            _pe(T_damping_pressure_surfaces=True, implicit_hyperdiff=True),
            grid, sigma, _DT, "ssp_rk3")
    with pytest.raises(ValueError, match="hyperdiff"):   # -coeff x -boost > 0
        _make_spectral_integrator(
            _pe(T_damping_pressure_surfaces=True, semi_implicit=True,
                si_hyperdiff_boost=-1.0)._replace(hyperdiff_coeff=-1e15),
            grid, sigma, _DT, "ssp_rk3")
    with pytest.raises(ValueError, match="si_hyperdiff_boost"):
        _make_spectral_integrator(
            _pe(T_damping_pressure_surfaces=True, semi_implicit=True,
                si_hyperdiff_boost=0.0), grid, sigma, _DT, "ssp_rk3")
    with pytest.raises(ValueError, match="must be a bool"):
        _make_spectral_integrator(_pe(T_damping_pressure_surfaces="false"),
                                  grid, sigma, _DT, "ssp_rk3")


def test_gradient_through_the_corrected_rollout(spec_setup):
    """jax.grad through the ON path is finite, and the CORRECTION's own
    contribution (ON minus OFF, so the uncorrected dynamics cancel) matches a
    central finite difference.  A strong filter makes that part O(1e-4) of
    the total instead of round-off."""
    from legoesm.training.neural_gcm_spectral import spectral_rollout
    grid, sigma, rest, _ = spec_setup
    filt = jnp.exp(-1.0 * (jnp.asarray(grid.ls, jnp.float64) / _N_MAX) ** 8)
    base = _with_terrain(rest, grid, amp=0.0)
    idx = int(np.argmax(np.asarray(grid.ls) == 6))
    w = jnp.linspace(1.0, 2.0, _N_LEV)

    def f(a, on):
        lnps = base.lnps_hat.data.at[idx].add(a)
        s = base._replace(lnps_hat=base.lnps_hat.replace(data=lnps))
        out = spectral_rollout(s, _zero_physics, grid, sigma,
                               _pe(T_damping_pressure_surfaces=on), _DT, 3,
                               None, filt)
        return jnp.sum(jnp.real(out.T_hat.data[idx]) * w)

    def corr(a):
        return f(a, True) - f(a, False)

    a0, h = 0.02, 1e-4
    g_on = float(jax.grad(lambda a: f(a, True))(a0))
    g = float(jax.grad(corr)(a0))
    fd = float((corr(a0 + h) - corr(a0 - h)) / (2 * h))
    assert np.isfinite(g_on) and np.isfinite(g)
    assert abs(g) > 1e-6 * abs(g_on) and abs(g) > 0
    assert abs(g - fd) <= 1e-4 * abs(fd)


# --------------------------------------------------------------------------
# WB scale trainer (scale_build)
# --------------------------------------------------------------------------

def test_wb_spectral_core_key(spec_setup):
    from legoesm.training.scale_build import (
        _spectral_pe_config, validate_wb_campaign_yaml,
    )
    off, on = _spectral_pe_config({}), _spectral_pe_config({_SP: True})
    assert off.T_damping_pressure_surfaces is False
    assert on.T_damping_pressure_surfaces is True
    assert on._replace(T_damping_pressure_surfaces=False) == off
    assert _spectral_pe_config({_SP: False}) == off
    with pytest.raises(ValueError, match=_LL):
        _spectral_pe_config({_LL: True})
    validate_wb_campaign_yaml({_SP: True, _LL: False})  # keys are consumed


def _wb_cfg():
    try:
        from scripts.run.train_weatherbench_scale import build_scale_config_from_args
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"train_weatherbench_scale import ({exc})")
    return build_scale_config_from_args(["--mode", "physics", "--smoke"])


def test_wb_latlon_core_key():
    from legoesm.training.scale_build import build_latlon_config
    cfg = _wb_cfg()
    base = {"n_lat": 32, "n_lon": 64, "nlev": 8, "dt": 300.0, "radiation": "gray"}
    off = build_latlon_config(cfg, dict(base))
    on = build_latlon_config(cfg, dict(base, **{_LL: True}))
    assert off.dycore.latlon_ah_pressure_surfaces is False
    assert on.dycore.latlon_ah_pressure_surfaces is True
    assert on._replace(dycore=on.dycore._replace(**{_LL: False})) == off
    assert build_latlon_config(cfg, dict(base, **{_LL: False})) == off
    with pytest.raises(ValueError, match=_SP):
        build_latlon_config(cfg, dict(base, **{_SP: True}))
    from legoesm.driver.restart import resolved_config_record
    assert resolved_config_record(on)["dycore"][_LL] is True


def test_wb_fingerprint_records_the_flag_only_when_set():
    tws = _load("_tws_ps_probe", "scripts/run/train_weatherbench_scale.py")
    cfg = _wb_cfg()._replace(n_epochs=1, lr=1e-3, optimizer="muon")
    fp = lambda y: tws._run_fingerprint(cfg, y, 0, 1, 1, 1)  # noqa: E731
    assert _SP not in fp({}) and _LL not in fp({_LL: False})
    assert _SP not in fp({_SP: False})
    assert fp({_SP: True})[_SP] is True


# --------------------------------------------------------------------------
# run_aimip (spectral suite)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def run_aimip():
    return _load("run_aimip_ps_probe", "scripts/run/run_aimip.py")


def _aimip_pe(mod, overrides, variant="classical"):
    base = mod._load_yaml(_ROOT / "config" / "aimip" / "aimip_era5.yaml")
    merged = mod._merge(base, overrides)
    merged["aimip_variant"] = variant
    return mod._build_spectral_config(merged).pe_config


def test_run_aimip_suite_key(run_aimip):
    off = _aimip_pe(run_aimip, {})
    on = _aimip_pe(run_aimip, {_SP: True})
    assert off.T_damping_pressure_surfaces is False
    assert on.T_damping_pressure_surfaces is True
    assert on._replace(T_damping_pressure_surfaces=False) == off
    assert _aimip_pe(run_aimip, {_SP: False}) == off
    with pytest.raises(ValueError, match=_LL):
        _aimip_pe(run_aimip, {_LL: True})
    with pytest.raises(ValueError, match="must be a bool"):
        _aimip_pe(run_aimip, {_SP: "true"})
    with pytest.raises(SystemExit, match="sfno_full"):
        _aimip_pe(run_aimip, {_SP: True}, variant="sfno_full")
    assert _aimip_pe(run_aimip, {}, variant="sfno_full") == off


@pytest.mark.parametrize("script", ["scripts/run/run_aimip_amip_finetune.py",
                                    "scripts/run/run_aimip_amip_inference.py"])
def test_aimip_amip_consumers_honour_the_suite_key(script):
    """Fine-tune / inference read the same suite key as run_aimip, so a
    checkpoint trained with the option keeps it."""
    mod = _load("_amip_ps_" + Path(script).stem, script)
    cfg = {"n_max": 21, "nlev": 8, "dt": 1800.0}

    def pe(extra):
        args = (dict(cfg, **extra),) + ((1979,) if "finetune" in script else ())
        return mod._build_spec_cfg(*args).pe_config
    off, on = pe({}), pe({_SP: True})
    assert off.T_damping_pressure_surfaces is False
    assert on._replace(T_damping_pressure_surfaces=False) == off
    with pytest.raises(ValueError, match=_LL):
        pe({_LL: True})


# --------------------------------------------------------------------------
# run_aimip_latlon + train_neural_gcm_spectral (CLI)
# --------------------------------------------------------------------------

def test_run_aimip_latlon_cli():
    mod = _load("_aimip_latlon_ps_probe", "scripts/run/run_aimip_latlon.py")
    p = mod.build_parser()
    assert p.parse_args([]).latlon_ah_pressure_surfaces is False
    off = mod.build_latlon_config(p.parse_args([]))
    on = mod.build_latlon_config(p.parse_args(["--latlon-ah-pressure-surfaces"]))
    assert off.dycore.latlon_ah_pressure_surfaces is False
    assert on.dycore.latlon_ah_pressure_surfaces is True
    assert on._replace(dycore=on.dycore._replace(**{_LL: False})) == off
    assert mod.build_latlon_config(
        p.parse_args(["--no-latlon-ah-pressure-surfaces"])) == off
    import argparse
    with pytest.raises(ValueError, match="must be a bool"):
        mod.build_latlon_config(argparse.Namespace(
            **{**vars(p.parse_args([])), _LL: "false"}))
    with pytest.raises(SystemExit):   # spectral option: no such flag here
        mod.build_parser().parse_args(["--spectral-t-damping-pressure-surfaces"])


@pytest.mark.parametrize("argv, expect", [
    ([], False),
    (["--spectral-t-damping-pressure-surfaces"], True),
    (["--no-spectral-t-damping-pressure-surfaces"], False),
])
def test_train_neural_gcm_spectral_cli(monkeypatch, tmp_path, argv, expect):
    import legoesm.training.neural_gcm_spectral as ngs
    mod = _load("_tngs_ps_probe", "scripts/run/train_neural_gcm_spectral.py")
    seen = {}

    def _fake_train(config, cache_dir, seed):
        seen["cfg"] = config
        return None, [0.0]
    monkeypatch.setattr(ngs, "train_neural_gcm_spectral", _fake_train)
    monkeypatch.setattr(sys, "argv", ["prog", *argv, "--checkpoint-dir",
                                      str(tmp_path / "ck")])
    mod.main()
    import json
    rec = json.loads((tmp_path / "ck" / "last_launch_dycore_options.json").read_text())
    assert rec == {_SP: expect}
    pe = seen["cfg"].pe_config
    assert pe.T_damping_pressure_surfaces is expect
    from legoesm.atmosphere.dynamics.gcm.spectral_pe import SpectralPEConfig
    assert pe._replace(T_damping_pressure_surfaces=False) == SpectralPEConfig(
        hyperdiff_coeff=2.5e15, hyperdiff_order=2, time_integrator="ssp_rk3",
        spectral_filter_strength=0.01, spectral_filter_order=8)
    # a refused invocation leaves an existing record untouched
    monkeypatch.setattr(sys, "argv", [
        "prog", "--spectral-t-damping-pressure-surfaces", "--hyperdiff", "0",
        "--checkpoint-dir", str(tmp_path / "ck")])
    with pytest.raises(ValueError, match="hyperdiff"):
        mod.main()
    assert json.loads((tmp_path / "ck" / "last_launch_dycore_options.json").read_text()) == rec
    monkeypatch.setattr(sys, "argv", [   # valid flag, CFL-violating dt
        "prog", "--spectral-t-damping-pressure-surfaces", "--dt", "7200",
        "--checkpoint-dir", str(tmp_path / "ck")])
    with pytest.raises(ValueError):
        mod.main()
    assert json.loads((tmp_path / "ck" / "last_launch_dycore_options.json").read_text()) == rec
    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, "argv", ["prog", "--latlon-ah-pressure-surfaces"])
        mod.main()


def test_latlon_gradient_through_the_corrected_tendency():
    """AD through the lat-lon A_h pressure-surface correction (the operator the
    lat-lon training models step with when the flag is on): d/d(terrain
    amplitude of p_s) of the corrected-minus-eta T tendency matches a central
    finite difference."""
    import math
    from legoesm import constants
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_generic import (
        cgrid_latlon_pe_tendencies)
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonPrimitiveEquationConfig, hydrostatic_to_cgrid)
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from tests.test_cases.dcmip2012.rest_state_topography import (
        lapse_rate_state_topography_init_latlon)
    prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        grid, coord = create_latlon_grid(18, 36), create_sigma_coordinate(8)
        st = hydrostatic_to_cgrid(
            lapse_rate_state_topography_init_latlon(grid, coord), grid)
        A_h = (0.1 * math.sqrt(constants.R_d * 300.0) * math.pi
               * constants.R_earth / 18)
        w = jnp.linspace(1.0, 2.0, 8)

        def f(a, on):
            s = st._replace(p_s=st.p_s * (1.0 + a * (st.p_s / jnp.mean(st.p_s) - 1.0)))
            cfg = CGridLatLonPrimitiveEquationConfig(A_h=A_h, A_h_pressure_surfaces=on)
            dT = cgrid_latlon_pe_tendencies(s, grid, coord, cfg,
                                            vertical_form="legacy")[2]
            return jnp.sum(dT * w)

        def corr(a):
            return f(a, True) - f(a, False)
        g = float(jax.grad(corr)(1.0))
        fd = float((corr(1.0 + 1e-4) - corr(1.0 - 1e-4)) / 2e-4)
        assert np.isfinite(g) and abs(g) > 0
        assert abs(g - fd) <= 1e-6 * abs(fd)
    finally:
        set_policy(prev)
