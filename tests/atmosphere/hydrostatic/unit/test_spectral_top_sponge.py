"""CAM-SE top Laplacian sponge on the spectral PE dycore.

nu_top x (4, 2, 1) del2 on vor, div and T of the top three levels, applied as
the implicit post-step factor exp(-nu_k n(n+1)/a^2 dt) (CAM-SE dyn_comp.F90
nu_scale_top, ptop > 300 Pa).  0 = off and bit-identical to the old step.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.gaussian import create_gaussian_grid
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
    SpectralPEConfig,
    SpectralPrimitiveEquationModel,
    compute_sponge_factor,
    isothermal_rest_state_spectral,
)

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="spectral tests need JAX_ENABLE_X64=1",
)

NLEV = 6
DT = 600.0
NU_TOP = 2.0e6


def _model(nu_top=0.0, tau=0.0, integrator="ssp_rk3"):
    grid = create_gaussian_grid(21)
    cfg = SpectralPEConfig(time_integrator=integrator, spectral_filter_strength=0.0,
                           sponge_tau=tau, sponge_del2_nu_top_m2s=nu_top)
    return SpectralPrimitiveEquationModel(
        grid=grid, sigma_coord=create_sigma_coordinate(NLEV), config=cfg)


def _seeded_state(model):
    state = isothermal_rest_state_spectral(
        model.grid, model.sigma_coord, perturbation_amplitude=0.0)
    ls = np.asarray(model.grid.ls)
    rng = np.random.default_rng(0)
    keep = (ls > 0) & (ls <= int(0.667 * model.grid.n_max))
    noise = rng.standard_normal(state.vor_hat.data.shape) * keep[:, None] * 1e-6
    return state._replace(
        vor_hat=state.vor_hat.replace(data=state.vor_hat.data + noise),
        div_hat=state.div_hat.replace(data=state.div_hat.data + 0.3 * noise),
        T_hat=state.T_hat.replace(data=state.T_hat.data + 1e4 * noise),
    )


def test_factor_is_the_cam_se_del2_profile():
    m = _model(NU_TOP)
    m._ensure_sponge_factor(DT)
    ls = np.asarray(m.grid.ls)
    nn1 = ls * (ls + 1.0)
    a2 = m.grid.radius ** 2
    scale = np.array([4.0, 2.0, 1.0] + [0.0] * (NLEV - 3))
    want_T = np.exp(-NU_TOP * scale[None, :] * (nn1 / a2)[:, None] * DT)
    want_v = np.exp(-NU_TOP * scale[None, :] * (np.maximum(nn1 - 2.0, 0.0) / a2)[:, None] * DT)
    np.testing.assert_allclose(np.asarray(m._sponge_factor), want_v, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(m._sponge_factor_T), want_T, rtol=1e-14)
    assert np.all(np.asarray(m._sponge_factor)[:, 3:] == 1.0)
    assert np.all(np.asarray(m._sponge_factor_T)[ls == 0] == 1.0)
    # vector Laplacian: rigid rotation (n = 1 vor/div) is not damped
    assert np.all(np.asarray(m._sponge_factor)[ls <= 1] == 1.0)
    assert np.all(np.asarray(m._sponge_factor)[ls == 2, :3] < 1.0)


def test_off_keeps_the_old_factors():
    off = _model(0.0)
    off._ensure_sponge_factor(DT)
    assert off._sponge_factor is None
    ray = _model(0.0, tau=86400.0)
    ray._ensure_sponge_factor(DT)
    sig = ray.sigma_coord.sigma_full
    assert bool(jnp.array_equal(
        ray._sponge_factor, compute_sponge_factor(sig, 0.1, 86400.0, DT)))


def test_rayleigh_and_del2_compose():
    both = _model(NU_TOP, tau=86400.0)
    both._ensure_sponge_factor(DT)
    ray = _model(0.0, tau=86400.0)
    ray._ensure_sponge_factor(DT)
    d2 = _model(NU_TOP)
    d2._ensure_sponge_factor(DT)
    np.testing.assert_allclose(np.asarray(both._sponge_factor),
                               np.asarray(ray._sponge_factor)[None, :]
                               * np.asarray(d2._sponge_factor), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(both._sponge_factor_T),
                               np.asarray(ray._sponge_factor_T)
                               * np.asarray(d2._sponge_factor_T), rtol=1e-14)


def test_step_damps_only_the_top_three_levels():
    on, off = _model(NU_TOP), _model(0.0)
    s = _seeded_state(on)
    a, b = on.step(s, DT), off.step(s, DT)
    for f in ("vor_hat", "div_hat", "T_hat"):
        x, y = np.asarray(getattr(a, f).data), np.asarray(getattr(b, f).data)
        assert np.array_equal(x[:, 3:], y[:, 3:]), f
        for k in range(3):
            assert np.sum(np.abs(x[:, k]) ** 2) < np.sum(np.abs(y[:, k]) ** 2), (f, k)
    assert np.array_equal(np.asarray(a.lnps_hat.data), np.asarray(b.lnps_hat.data))


def test_column_lane_step_applies_the_sponge():
    from legoesm.atmosphere.dynamics.gcm.spectral_column import SpectralColumnModel
    on, off = _model(NU_TOP, integrator="ssp_rk54"), _model(0.0, integrator="ssp_rk54")
    s = _seeded_state(on)
    a = SpectralColumnModel(on).step(SpectralColumnModel(on).from_bundle(s), DT).native
    b = SpectralColumnModel(off).step(SpectralColumnModel(off).from_bundle(s), DT).native
    for f in ("vor_hat", "div_hat", "T_hat"):
        x, y = np.asarray(getattr(a, f).data), np.asarray(getattr(b, f).data)
        assert np.array_equal(x[:, 3:], y[:, 3:]), f
        assert not np.array_equal(x[:, 0], y[:, 0]), f


@pytest.mark.parametrize("integrator", ["ssp_rk3", "ssp_rk54"])
def test_applied_decay_per_mode_is_the_factor(integrator):
    """End to end: the operators after the sponge are per-mode diagonal, so
    step(on)/step(off) at a top level is exactly the sponge factor per mode."""
    on, off = _model(NU_TOP, integrator=integrator), _model(0.0, integrator=integrator)
    s = _seeded_state(on)
    a, b = on.step(s, DT), off.step(s, DT)
    on._ensure_sponge_factor(DT)
    fac = {"vor_hat": on._sponge_factor, "div_hat": on._sponge_factor,
           "T_hat": on._sponge_factor_T}
    for f, fk in fac.items():
        x, y = np.asarray(getattr(a, f).data), np.asarray(getattr(b, f).data)
        for k in range(3):
            live = np.abs(y[:, k]) > 1e-12 * np.abs(y[:, k]).max()
            np.testing.assert_allclose(x[live, k] / y[live, k],
                                       np.asarray(fk)[live, k], rtol=1e-9, err_msg=f)


def test_refuses_too_few_levels():
    grid = create_gaussian_grid(21)
    m = SpectralPrimitiveEquationModel(
        grid=grid, sigma_coord=create_sigma_coordinate(2),
        config=SpectralPEConfig(sponge_del2_nu_top_m2s=NU_TOP))
    with pytest.raises(ValueError, match="levels"):
        m._ensure_sponge_factor(DT)


def test_factory_forwards_and_config_validates():
    from legoesm.driver.component_factory import create_atmosphere_dycore
    from legoesm.driver.config import DycoreConfig, ExperimentConfig, GridConfig
    from legoesm.grids.factory import create_grid
    grid = create_grid("gaussian", 21)
    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="gaussian", resolution=21, nlev=NLEV),
        dycore=DycoreConfig(energy_fixer=False, model_type="hydrostatic",
                            discretization="spectral", dt=300.0,
                            spectral_sponge_nu_top_m2s=NU_TOP))
    model = create_atmosphere_dycore(cfg, grid, create_sigma_coordinate(NLEV))
    assert model.config.sponge_del2_nu_top_m2s == NU_TOP
    assert DycoreConfig().spectral_sponge_nu_top_m2s == 0.0
    grid_cfg = GridConfig(grid_type="gaussian", resolution=21, nlev=NLEV)
    for bad in (dict(spectral_sponge_nu_top_m2s=-1.0),
                dict(spectral_sponge_nu_top_m2s=NU_TOP, discretization="mpas")):
        with pytest.raises(ValueError, match="spectral_sponge_nu_top_m2s"):
            ExperimentConfig(grid=grid_cfg, dycore=DycoreConfig(
                model_type="hydrostatic",
                **{"discretization": "spectral", **bad})).validate_strict()


def test_cli_round_trip():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    *[os.pardir] * 4, "scripts", "run"))
    from run_amip import build_arg_parser, build_config_from_args
    args = build_arg_parser().parse_args(["--spectral-sponge-nu-top-m2s", "1.25e5"])
    assert build_config_from_args(args).dycore.spectral_sponge_nu_top_m2s == 1.25e5
    args = build_arg_parser().parse_args([])
    assert build_config_from_args(args).dycore.spectral_sponge_nu_top_m2s == 0.0


def test_production_deck_resolves_8h_and_the_sponge():
    import os
    import sys
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 4)
    sys.path.insert(0, os.path.join(root, "scripts", "run"))
    from run_amip import build_arg_parser, build_config_from_args
    deck = os.path.join(root, "config", "amip", "amip_production_spectral_t31.yaml")
    from legoesm.driver.run_config_yaml import load_yaml_config
    parser = build_arg_parser()
    parser.set_defaults(**load_yaml_config(deck, parser, example_keys=""))
    d = build_config_from_args(parser.parse_args([])).dycore
    assert d.spectral_sponge_nu_top_m2s == 1.16e6
    assert d.hyperdiff_scale == 0.0625
    assert d.discretization == "spectral"
