"""Gap 5 selection, corrected thin snow and incident-to-ocean SW contracts."""
from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.coupler.ocean_forcing import blend_ice_ocean_forcing
from legoesm.grids.tripole import create_synthetic_tripole
from legoesm.ice import SeaIceConfig, step_sea_ice, uses_new_physics
from legoesm.ice.shortwave import compute_ice_sw, delta_eddington_albedo
from legoesm.ocean.state import OceanSurfaceForcing
import scripts.run.run_omip_core2 as runner
from tests.ice.unit.test_omip_bulk_snow import (
    _main_config, _state_and_forcing, _fixed_bulk, _column,
)
from tests.unit.test_run_omip_core2_ice_categories import _min_forcing

SCHEMES = ("constant", "maykut_untersteiner", "delta_eddington")
# Synthetic test inputs, not new production coefficients.
INCIDENT = 240.0
SKIN_T = constants.T_freeze - 10.0
SNOW_DEPTHS = (0.0, 0.001, 0.01, 0.02, 0.025, 0.05, 0.1)


def test_optics_default_and_constant_override():
    assert runner._build_arg_parser().parse_args([]).ice_shortwave is None
    cfg = _main_config([])
    assert cfg.shortwave_scheme == SeaIceConfig().shortwave_scheme == "constant"
    assert cfg.albedo_ice == SeaIceConfig().albedo_ice
    assert cfg.sw_transmittance_const == 0.03
    cfg = _main_config(["--prognostic-sea-ice", "--ice-shortwave", "constant",
                        "--ice-thermo-sw-trans", "0.17"])
    assert cfg.sw_transmittance_const == 0.17


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("snow", ["off", "bulk"])
def test_actual_main_forwards_optics_and_snow(scheme, snow):
    cfg = _main_config(["--prognostic-sea-ice", "--ice-shortwave", scheme,
                        "--ice-snow", snow])
    assert cfg.shortwave_scheme == scheme
    assert cfg.snow.enabled == (snow == "bulk")
    assert cfg.sw_transmittance_const == (0.03 if scheme == "constant" else 0.0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_actual_main_refuses_selection_without_ice(scheme):
    with pytest.raises(ValueError, match="requires --prognostic-sea-ice"):
        _main_config(["--ice-shortwave", scheme])


@pytest.mark.parametrize("scheme", SCHEMES[1:])
@pytest.mark.parametrize("argv_tail", [["--ice-thermo-sw-trans", "0.03"],
                                       ["--ice-thermo-sw-trans=0"]])
def test_actual_main_refuses_inactive_transmission(scheme, argv_tail):
    with pytest.raises(ValueError, match="owns its transmission"):
        _main_config(["--prognostic-sea-ice", "--ice-shortwave", scheme] + argv_tail)


def test_unknown_optics_and_unwired_fesom_raise():
    with pytest.raises(ValueError, match="Unknown ice shortwave"):
        runner._resolve_ice_shortwave("typo", True, sw_transmittance=0.0)
    parser = runner._build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--ice-shortwave", "typo"])
    args = parser.parse_args(["--grid", "fesom", "--fesom-mesh-dir", "/unused",
                              "--ice-shortwave", "delta_eddington"])
    with pytest.raises(SystemExit, match="ice-shortwave"):
        runner.validate_fesom_stage(args, parser)
    # The explicit-flag detector is safe because the real parser forbids abbreviations.
    with pytest.raises(SystemExit):
        parser.parse_args(["--ice-thermo-sw-tra", "0.03"])


@pytest.mark.parametrize("temperature", [SKIN_T, constants.T_freeze])
@pytest.mark.parametrize("thickness", [0.1, 1.0])
def test_thin_snow_brightens_instead_of_masking_with_black(temperature, thickness):
    hs = jnp.asarray(SNOW_DEPTHS)
    z = jnp.zeros_like(hs)
    alpha, transmission = delta_eddington_albedo(z + temperature, z + thickness, hs, z, z)
    assert np.all(np.diff(alpha) >= 0)
    assert float(alpha[1]) > float(alpha[0])
    assert np.all(np.diff(transmission) <= 0)
    assert float(transmission[0]) > 0 and float(transmission[-1]) == 0


@pytest.mark.parametrize("scheme", SCHEMES)
def test_column_absorption_and_shortwave_budget(monkeypatch, scheme):
    _fixed_bulk(monkeypatch)
    cfg = _main_config(["--prognostic-sea-ice", "--ice-shortwave", scheme])
    cfg = cfg._replace(ocean_heat_transfer_coeff=0.0)
    h, hs, dt, conc = 1.0, 0.01, 60.0, 0.7
    forcing = _min_forcing((1,))._replace(
        sw_down=jnp.full((1,), INCIDENT),
        lw_down=jnp.full((1,), constants.sigma_sb * SKIN_T**4))
    out = _column(cfg, forcing, h=h, snow=hs, T=SKIN_T, conc=conc, dt=dt)
    # Independent skin storage + conductive boundary reconstruction, not
    # absorbed := incident - reflected - transmitted from the SW kernel.
    capacity = cfg.rho_ice * cfg.c_ice * h / 2 + cfg.snow.rho_snow * cfg.snow.c_snow * hs
    conductance = 1 / (h / cfg.k_ice + hs / cfg.snow.k_snow)
    absorbed = capacity * (out["T"] - SKIN_T) / dt - conductance * (cfg.T_freeze_ocean - out["T"])
    transmitted = out["sw_penetrated_to_ocean"] / conc
    reflected = out["alpha"] * INCIDENT
    assert np.all(absorbed > 0)
    assert np.all(transmitted >= 0)
    np.testing.assert_allclose(absorbed + reflected + transmitted, INCIDENT, atol=1e-8, rtol=0)
    if scheme == "constant":
        np.testing.assert_allclose(transmitted, cfg.sw_transmittance_const * INCIDENT)
    elif scheme == "maykut_untersteiner":
        np.testing.assert_array_equal(transmitted, 0)
    else:
        assert np.all(transmitted > 0)


def _actual_main_blend(sf, response, cfg):
    """Execute the real host blend call with real mappers, not a retyped call."""
    tree = ast.parse(inspect.getsource(runner.main))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "blend_ice_ocean_forcing"]
    assert len(calls) == 1
    shape = response.ocean_heat_extraction.shape
    env = dict(sf=sf, fw=None, ice_resp=response, ice_config=cfg,
               _ice_conc_pre=response.ice_concentration_thermo,
               state=SimpleNamespace(land_mask=SimpleNamespace(data=jnp.ones(shape))),
               _ice_const=constants, uses_new_physics=uses_new_physics,
               blend_ice_ocean_forcing=blend_ice_ocean_forcing)
    return eval(compile(ast.Expression(calls[0]), inspect.getsourcefile(runner), "eval"), env)[1]


@pytest.mark.parametrize("ncat", [1, 5])
@pytest.mark.parametrize("scheme", SCHEMES)
def test_production_step_transmission_reaches_actual_ocean_blend(monkeypatch, ncat, scheme):
    _fixed_bulk(monkeypatch)
    cfg = _main_config(["--prognostic-sea-ice", "--ice-shortwave", scheme,
                        "--ice-categories", str(ncat)])
    cfg = cfg._replace(ocean_heat_transfer_coeff=0.0)
    st, forcing = _state_and_forcing(ncat)
    shape = (8, 12)
    if ncat == 5:
        st = st._replace(
            h_ice=st.h_ice.replace(data=jnp.broadcast_to(jnp.array([0.1, 0.4, 1., 2., 4.]), shape + (5,))),
            concentration=st.concentration.replace(data=jnp.broadcast_to(jnp.array([0.05, 0.1, 0.2, 0.3, 0.15]), shape + (5,))))
    st = st._replace(h_snow=st.h_snow.replace(data=jnp.zeros_like(st.h_snow.data)))
    forcing = forcing._replace(sw_down=jnp.full(shape, INCIDENT))
    z = jnp.zeros(shape)
    grid = create_synthetic_tripole(*shape)
    new, resp = step_sea_ice(st, forcing, z + constants.T_freeze, z, z, cfg,
                             U_min=0.0, dt=60.0, grid=grid)
    if scheme == "delta_eddington":
        compiled = jax.jit(lambda state: step_sea_ice(
            state, forcing, z + constants.T_freeze, z, z, cfg,
            U_min=0.0, dt=60.0, grid=grid))(st)
        for field in ("h_ice", "h_snow", "T_ice", "S_ice"):
            np.testing.assert_allclose(
                getattr(compiled[0], field).data * compiled[0].concentration.data,
                getattr(new, field).data * new.concentration.data, atol=1e-9, rtol=1e-9)
        for a, b in zip(jax.tree.leaves(compiled[1]), jax.tree.leaves(resp)):
            np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-9)
    sw = forcing.sw_down[..., None] if ncat > 1 else forcing.sw_down
    optical = compute_ice_sw(sw, st.T_ice.data, st.h_ice.data, st.h_snow.data,
                             jnp.zeros_like(st.h_ice.data), jnp.zeros_like(st.h_ice.data),
                             scheme=scheme, albedo_const=cfg.albedo_ice,
                             sw_transmittance_const=cfg.sw_transmittance_const)
    expected = optical.sw_penetrated * st.concentration.data
    if ncat > 1:
        expected = expected.sum(axis=-1)
    if scheme != "maykut_untersteiner":
        assert np.all(expected > 0)
    np.testing.assert_allclose(-resp.ocean_heat_extraction, expected, atol=1e-10, rtol=1e-12)
    open_sf = OceanSurfaceForcing(sw_down=z + INCIDENT, q_net=z + INCIDENT,
                                  tau_x=z, tau_y=z)
    sf = _actual_main_blend(open_sf, resp, cfg)
    open_absorbed = INCIDENT * (1 - constants.alpha_ocean_broadband) * (1 - resp.ice_concentration_thermo)
    np.testing.assert_allclose(sf.q_net - open_absorbed, expected, atol=1e-10, rtol=1e-12)
    # The response albedo describes the POST-step state, unlike the absorbed
    # flux above. Check per-category weighting on that state independently.
    post_sw = compute_ice_sw(sw, new.T_ice.data, new.h_ice.data, new.h_snow.data,
                             new.pond_area.data, new.pond_depth.data,
                             scheme=scheme, albedo_const=cfg.albedo_ice,
                             sw_transmittance_const=cfg.sw_transmittance_const)
    alpha = post_sw.albedo_eff
    if ncat > 1:
        alpha = (alpha * new.concentration.data).sum(-1) / new.concentration.data.sum(-1)
    np.testing.assert_allclose(resp.albedo, alpha, atol=1e-12)


@pytest.mark.parametrize("snow_mode", ["off", "bulk"])
def test_optics_reads_carried_snow_and_bulk_accumulation(monkeypatch, snow_mode):
    _fixed_bulk(monkeypatch)
    cfg = _main_config(["--prognostic-sea-ice", "--ice-shortwave", "delta_eddington",
                        "--ice-snow", snow_mode])
    cfg = cfg._replace(ocean_heat_transfer_coeff=0.0)
    rate, hs, dt = 0.001, 0.005, 60.0
    forcing = _min_forcing((1,))._replace(sw_down=jnp.full((1,), INCIDENT),
        precip_snow=jnp.full((1,), rate), precip_total=jnp.full((1,), rate),
        has_precipitation=jnp.asarray(1.0))
    out = _column(cfg, forcing, snow=hs, dt=dt)
    after_snow = hs + (rate * dt / cfg.snow.rho_snow if snow_mode == "bulk" else 0)
    a = lambda x: jnp.full((1,), x)
    expected = compute_ice_sw(a(INCIDENT), a(260.), a(1.5), a(after_snow), a(0.), a(0.),
                              scheme=cfg.shortwave_scheme, albedo_const=cfg.albedo_ice)
    np.testing.assert_allclose(out["alpha"], expected.albedo_eff, atol=1e-13)
    np.testing.assert_allclose(out["sw_penetrated_to_ocean"], expected.sw_penetrated * 0.8, atol=1e-13)
    bare = _column(cfg, forcing, snow=0, dt=dt)
    assert float(out["alpha"][0]) > float(bare["alpha"][0])


def test_optics_jit_and_nonzero_state_gradients():
    # Both a partially covered snow surface and nonsaturated thin ice.
    def outputs(x):
        return jnp.stack(delta_eddington_albedo(SKIN_T, x[0], x[1], 0., 0.))
    x = jnp.array([0.2, 0.01])
    np.testing.assert_allclose(jax.jit(outputs)(x), outputs(x), atol=1e-13)
    jac = jax.jit(jax.jacfwd(outputs))(x)
    assert np.all(np.isfinite(jac)) and np.all(np.abs(jac) > 0)
    eps = 1e-6
    fd = jnp.stack([(outputs(x + eps * e) - outputs(x - eps * e)) / (2 * eps)
                    for e in jnp.eye(2)], axis=-1)
    np.testing.assert_allclose(jac, fd, atol=1e-8, rtol=1e-7)
