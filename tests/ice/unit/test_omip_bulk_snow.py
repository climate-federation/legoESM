"""Gap 4 bulk-snow wiring and its existing tripole physics contracts.

CPU, x64 for budget cancellation; no NEMO climate or multilayer equivalence.
"""
from __future__ import annotations

import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import latent_heat_sublimation
from legoesm.grids.latlon import create_latlon_geometry
from legoesm.ice import (
    SeaIceConfig, distribute_dynamic_state_to_categories,
    init_dynamic_ice_state, step_sea_ice,
)
from legoesm.ice.config import BrineConfig, RidgingConfig, SnowConfig
import legoesm.ice.sea_ice as ice
import scripts.run.run_omip_core2 as runner
from tests.unit.test_run_omip_core2_ice_categories import _min_forcing


def _main_config(argv):
    """Execute the production validation/config statements, before ocean I/O.

    Extract by assigned name, not presence of the new keyword: deleting the
    forwarding line must exercise the default and fail the value assertions.
    """
    args = runner._build_arg_parser().parse_args(argv)
    tree = ast.parse(inspect.getsource(runner.main))
    statements = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        names = {n.id for target in node.targets for n in ast.walk(target)
                 if isinstance(n, ast.Name)}
        if names & {"_snow", "_n_cat", "_ice_sw_scheme"}:
            statements.append(node)
        elif ("ice_config" in names and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Name)
              and node.value.func.id == "SeaIceConfig"):
            statements.append(node)
    assert len(statements) == 4
    env = dict(vars(runner), args=args, _supports_dyn=False,
               _cli_flags_given=lambda: runner._cli_flags_given(argv),
               _supports_transport=True, grid=create_latlon_geometry(8, 12),
               _ice_dyn="free_drift", _transport="advect",
               _brine=BrineConfig(enabled=True), SeaIceConfig=SeaIceConfig,
               RidgingConfig=RidgingConfig)
    exec(compile(ast.Module(body=sorted(statements, key=lambda n: n.lineno),
                            type_ignores=[]), inspect.getsourcefile(runner), "exec"), env)
    return env["ice_config"]


def test_defaults_preserved():
    args = runner._build_arg_parser().parse_args([])
    assert (args.ice_snow, args.ice_snow_k, args.ice_snow_flooding) == ("off", None, None)
    assert runner._resolve_ice_snow("off", False) == SnowConfig()
    assert _main_config([]).snow == SnowConfig()
    expected = SeaIceConfig(dynamics="free_drift", transport="advect",
                            brine=BrineConfig(enabled=True),
                            sw_transmittance_const=args.ice_thermo_sw_trans)
    assert _main_config([]) == expected


@pytest.mark.parametrize("flooding", [None, "on", "off"])
@pytest.mark.parametrize("conductivity", [None, 0.5, 0.8])
def test_actual_main_forwards_all_snow_fields(conductivity, flooding):
    argv = ["--prognostic-sea-ice", "--ice-snow", "bulk"]
    if conductivity is not None:
        argv += ["--ice-snow-k", str(conductivity)]
    if flooding is not None:
        argv += ["--ice-snow-flooding", flooding]
    cfg = _main_config(argv)
    expected = SnowConfig(enabled=True)
    if conductivity is not None:
        expected = expected._replace(k_snow=conductivity)
    if flooding is not None:
        expected = expected._replace(flooding=flooding == "on")
    assert cfg.snow == expected


@pytest.mark.parametrize("flag", ["--ice-snow", "--ice-snow-flooding"])
def test_parser_rejects_unknown(flag):
    with pytest.raises(SystemExit):
        runner._build_arg_parser().parse_args([flag, "typo"])


@pytest.mark.parametrize("scheme,prognostic,kw,match", [
    ("typo", True, {}, "Unknown ice snow scheme"),
    ("off", True, {"flooding": "typo"}, "Unknown ice snow flooding"),
    ("bulk", False, {}, "requires --prognostic-sea-ice"),
    ("off", True, {"k_snow": 0.5}, "require --ice-snow bulk"),
    ("off", True, {"flooding": "off"}, "require --ice-snow bulk"),
])
def test_resolver_refuses_unknown_or_inert(scheme, prognostic, kw, match):
    with pytest.raises(ValueError, match=match):
        runner._resolve_ice_snow(scheme, prognostic, **kw)


@pytest.mark.parametrize("value", [0, -0.5, float("nan"), float("inf"), -float("inf")])
def test_conductivity_must_be_positive_finite(value):
    with pytest.raises(ValueError, match="finite and positive"):
        _main_config(["--prognostic-sea-ice", "--ice-snow", "bulk",
                      "--ice-snow-k=" + str(value)])


def test_actual_main_refuses_snow_without_prognostic_ice():
    with pytest.raises(ValueError, match="requires --prognostic-sea-ice"):
        _main_config(["--ice-snow", "bulk"])


@pytest.mark.parametrize("flag,value", [("--ice-snow", "bulk"),
                                        ("--ice-snow-k", "0.5"),
                                        ("--ice-snow-flooding", "off")])
def test_fesom_refuses_each_unwired_option_independently(flag, value):
    parser = runner._build_arg_parser()
    args = parser.parse_args(["--grid", "fesom", "--fesom-mesh-dir", "/unused", flag, value])
    with pytest.raises(SystemExit, match=flag[2:]):
        runner.validate_fesom_stage(args, parser)


def _state_and_forcing(ncat=1):
    shape = (8, 12)
    st = init_dynamic_ice_state(shape, S_ice_init=4.0)
    st = st._replace(
        h_ice=st.h_ice.replace(data=jnp.full(shape, 1.5)),
        concentration=st.concentration.replace(data=jnp.full(shape, 0.8)),
        T_ice=st.T_ice.replace(data=jnp.full(shape, 260.0)),
        h_snow=st.h_snow.replace(data=jnp.full(shape, 0.02)))
    if ncat > 1:
        st = distribute_dynamic_state_to_categories(st, ncat)
    forcing = _min_forcing(shape)
    return st, forcing


def _fixed_bulk(monkeypatch, sensible=0.0, latent=0.0):
    """Isolate the real thermal operators with prescribed upward fluxes."""
    def bulk(T, forcing, config, U_min):
        z = jnp.zeros_like(T)
        return z, z, z + sensible, z + latent
    monkeypatch.setattr(ice, "_bulk_flux_dispatch", bulk)


@pytest.mark.parametrize("ncat", [1, 5])
def test_tripole_snowfall_accumulates_and_jit_matches(monkeypatch, ncat):
    _fixed_bulk(monkeypatch)
    st, forc = _state_and_forcing(ncat)
    shape = (8, 12)
    rate, dt = 2e-4, 600.0
    forc = forc._replace(precip_snow=jnp.full(shape, rate),
                         precip_total=jnp.full(shape, rate),
                         has_precipitation=jnp.asarray(1.0))
    cfg = _main_config(["--prognostic-sea-ice", "--ice-snow", "bulk",
                        "--ice-snow-k", "0.5", "--ice-categories", str(ncat)])
    grid = create_latlon_geometry(*shape)
    zero = jnp.zeros(shape)
    # Above freezing blocks new lead ice; the zero heat coefficient isolates
    # snow capture, without changing the production defaults.
    cfg = cfg._replace(ocean_heat_transfer_coeff=0.0)
    def step(state, config):
        return step_sea_ice(state, forc, jnp.full(shape, constants.T_freeze),
                            zero, zero, config, U_min=0.0, dt=dt, grid=grid)
    out, _ = step(st, cfg)
    def snow_mass(state):
        mass = state.h_snow.data * state.concentration.data * cfg.snow.rho_snow
        return mass.sum(axis=-1) if ncat > 1 else mass
    np.testing.assert_allclose(snow_mass(out) - snow_mass(st), rate * dt * 0.8,
                               rtol=1e-9, atol=1e-10)
    off, _ = step(st, cfg._replace(snow=cfg.snow._replace(enabled=False)))
    np.testing.assert_allclose(snow_mass(off), snow_mass(st), rtol=1e-9)
    jit_out = jax.jit(lambda state: step(state, cfg))(st)
    eager_out = step(st, cfg)
    # Remapping creates arbitrarily small category areas. Compare the carried
    # extensive quantities there: dividing their roundoff by tiny area makes
    # a per-category depth/temperature comparison ill-conditioned.
    for name in ("h_ice", "h_snow", "S_ice", "T_ice"):
        a = getattr(jit_out[0], name).data * jit_out[0].concentration.data
        b = getattr(eager_out[0], name).data * eager_out[0].concentration.data
        if name in ("S_ice", "T_ice"):
            a = a * jit_out[0].h_ice.data
            b = b * eager_out[0].h_ice.data
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(jit_out[0].concentration.data,
                               eager_out[0].concentration.data, rtol=1e-9, atol=1e-10)
    for a, b in zip(jax.tree.leaves(jit_out[1]), jax.tree.leaves(eager_out[1])):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)


def _column(cfg, forc, *, h=1.5, snow=0.02, T=260.0, conc=0.8, dt=600.0):
    a = lambda x: jnp.full((1,), x)
    return ice._thermo_v2(a(h), a(T), a(conc), a(snow), a(4.0), a(0.0), a(0.0),
                         forc, a(constants.T_freeze), cfg, 0.0, dt,
                         enable_lead_freeze=False)


@pytest.mark.parametrize("q_surface", [-80.0, 300.0, 200000.0])
def test_surface_energy_closes_with_snow_and_meltout(monkeypatch, q_surface):
    # Prescribe sensible heat UPWARD (15 W/m2). Latent zero isolates phase
    # melt from sublimation. Both signs are checked in the thermal solve.
    sensible = 15.0
    _fixed_bulk(monkeypatch, sensible=sensible)
    cfg = _main_config(["--prognostic-sea-ice", "--ice-snow", "bulk",
                        "--ice-snow-k", "0.5", "--ice-snow-flooding", "off"])
    cfg = cfg._replace(ocean_heat_transfer_coeff=0.0)
    h, hs, conc, dt = 0.2, 0.02, 0.8, 600.0
    T = cfg.T_melt_surface - 0.1
    forc = _min_forcing((1,))
    # Downwelling LW supplies the desired net surface forcing after
    # reflection/emission and upward sensible heat. No solar in this test.
    lw = constants.sigma_sb * T**4 + (q_surface + sensible) / cfg.emissivity_ice
    forc = forc._replace(sw_down=jnp.zeros(1), lw_down=jnp.full((1,), lw))
    out = _column(cfg, forc, h=h, snow=hs, T=T, conc=conc, dt=dt)
    new_T = out["T"]
    capacity = cfg.rho_ice * cfg.c_ice * h / 2 + cfg.snow.rho_snow * cfg.snow.c_snow * hs
    # Positive UP from base into skin. Surface budget uses post-solve T.
    conductance = 1 / (h / cfg.k_ice + hs / cfg.snow.k_snow)
    conduction = conductance * (cfg.T_freeze_ocean - new_T)
    storage = capacity * (new_T - T) / dt
    available = q_surface + conduction - storage
    snow_latent = cfg.snow.rho_snow * hs * cfg.L_f
    ice_latent = cfg.rho_ice * h * cfg.L_f
    if q_surface < 0:
        assert float(new_T[0]) < T
        melt = jnp.zeros(1)
        np.testing.assert_allclose(out["ocean_heat_extraction"], 0, atol=1e-12)
    elif q_surface < 1000:
        # Only snow melted, so no retreat/dilution; direct storage decrease.
        assert 0 < float(out["h_snow"][0]) < hs
        melt = cfg.snow.rho_snow * (hs - out["h_snow"]) * cfg.L_f / dt
        np.testing.assert_allclose(out["ocean_heat_extraction"], 0, atol=1e-12)
    else:
        assert float(out["h_snow"][0]) == 0 and float(out["h"][0]) == 0
        # Surplus arrives in ocean as NEGATIVE extraction, on input ice area.
        assert float(out["ocean_heat_extraction"][0]) < 0
        melt = (snow_latent + ice_latent) / dt - out["ocean_heat_extraction"] / conc
    np.testing.assert_allclose(available, melt, rtol=1e-10, atol=1e-8)


def test_conductivity_changes_temperature_and_gradient(monkeypatch):
    _fixed_bulk(monkeypatch)
    cfg = _main_config(["--prognostic-sea-ice", "--ice-snow", "bulk",
                        "--ice-snow-k", "0.5", "--ice-snow-flooding", "off"])
    forc = _min_forcing((1,))
    def temperature(k):
        return _column(cfg._replace(snow=cfg.snow._replace(k_snow=k)), forc)["T"].sum()
    assert float(temperature(0.8)) > float(temperature(0.5))
    k, delta = jnp.asarray(0.5), 1e-4
    deriv = jax.jit(jax.grad(temperature))(k)
    fd = (temperature(k + delta) - temperature(k - delta)) / (2 * delta)
    assert float(deriv) > 1e-4
    np.testing.assert_allclose(deriv, fd, rtol=1e-6)
    np.testing.assert_allclose(jax.jit(temperature)(k), temperature(k), rtol=1e-12)


def test_flooding_switch_and_ocean_water_sign(monkeypatch):
    _fixed_bulk(monkeypatch)
    cfg = _main_config(["--prognostic-sea-ice", "--ice-snow", "bulk",
                        "--ice-snow-flooding", "on"])
    forc = _min_forcing((1,))
    off = _column(cfg._replace(snow=cfg.snow._replace(flooding=False)), forc, h=0.2, snow=0.3)
    on = _column(cfg, forc, h=0.2, snow=0.3)
    d = on["h"] - off["h"]
    assert float(d[0]) > 0
    np.testing.assert_allclose(off["h_snow"] - on["h_snow"], d, rtol=1e-10)
    # White ice adds more mass than the snow it replaces; the difference is
    # seawater withdrawn from the ocean, hence NEGATIVE freshwater-to-ocean.
    draw = d * (cfg.rho_ice - cfg.snow.rho_snow) * on["conc"] / 600.0
    delta_fw = on["freshwater_to_ocean"] - off["freshwater_to_ocean"]
    assert float(delta_fw[0]) < 0
    np.testing.assert_allclose(delta_fw, -draw, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("latent", [-30.0, 30.0])
def test_sublimation_deposition_mass_and_surface_energy_sign(monkeypatch, latent):
    _fixed_bulk(monkeypatch, latent=latent)
    cfg = _main_config(["--prognostic-sea-ice", "--ice-snow", "bulk",
                        "--ice-snow-flooding", "off"])
    forc = _min_forcing((1,))
    T, hs, h, dt, conc = 260.0, 0.02, 1.5, 600.0, 0.8
    out = _column(cfg, forc, T=T, snow=hs, h=h, dt=dt, conc=conc)
    # Positive latent is upward sublimation; negative latent is deposition.
    np.testing.assert_allclose((hs - out["h_snow"]) * cfg.snow.rho_snow,
                               latent / float(latent_heat_sublimation(T)) * dt, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(out["sublim_mass_to_atmos"],
                               latent / float(latent_heat_sublimation(T)) * conc, rtol=1e-10)
    q = (forc.sw_down * (1 - cfg.albedo_ice - min(cfg.sw_transmittance_const, 1 - cfg.albedo_ice))
         + cfg.emissivity_ice * (forc.lw_down - constants.sigma_sb * T**4) - latent)
    k = 1 / (h / cfg.k_ice + hs / cfg.snow.k_snow)
    cap = cfg.rho_ice * cfg.c_ice * h / 2 + cfg.snow.rho_snow * cfg.snow.c_snow * hs
    np.testing.assert_allclose(q + k * (cfg.T_freeze_ocean - out["T"]),
                               cap * (out["T"] - T) / dt, rtol=1e-9, atol=1e-9)


def test_core2_snow_forcing_reaches_real_tripole_step():
    st, _ = _state_and_forcing()
    shape = (8, 12)
    full = lambda x: np.full(shape, x)
    fields = dict(u10=full(5.0), v10=full(0.0), T_air=full(250.0),
                  q_air=full(1e-3), sw_down=full(50.0), lw_down=full(200.0),
                  precip=full(3e-4), snow=full(2e-4))
    ramp = 0.7
    forcing = runner._build_atm_to_surface_core2(fields, ramp=ramp)
    np.testing.assert_allclose(forcing.precip_snow, ramp * fields["snow"])
    assert float(forcing.has_precipitation) == 1
    cfg = _main_config(["--prognostic-sea-ice", "--ice-snow", "bulk",
                        "--ice-snow-k", "0.5"])
    zero = jnp.zeros(shape)
    grid = create_latlon_geometry(*shape)
    def step(forcing):
        return step_sea_ice(st, forcing, jnp.full(shape, constants.T_freeze),
                            zero, zero, cfg, U_min=0.0, dt=600.0, grid=grid)
    wet, resp = step(forcing)
    dry, _ = step(forcing._replace(has_precipitation=jnp.asarray(0.0)))
    assert np.all(np.asarray(wet.h_snow.data) > np.asarray(dry.h_snow.data))
    for leaf in jax.tree.leaves((wet, resp)):
        assert np.all(np.isfinite(leaf))
    # Same precipitation mask must remain traced under jit, not Python bool.
    compiled = jax.jit(step)(forcing)
    for a, b in zip(jax.tree.leaves(compiled), jax.tree.leaves((wet, resp))):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-10)


def test_upward_conduction_freezes_the_ice_base(monkeypatch):
    _fixed_bulk(monkeypatch)
    cfg = _main_config(["--prognostic-sea-ice", "--ice-snow", "bulk",
                        "--ice-snow-k", "0.5", "--ice-snow-flooding", "off"])
    cfg = cfg._replace(ocean_heat_transfer_coeff=0.0)
    forc = _min_forcing((1,))
    h, hs, conc, dt = 1.5, 0.02, 0.8, 600.0
    out = _column(cfg, forc, h=h, snow=hs, conc=conc, dt=dt)
    upward = (cfg.T_freeze_ocean - out["T"]) / (h / cfg.k_ice + hs / cfg.snow.k_snow)
    assert float(upward[0]) > 0
    latent_growth = (out["h"] * out["conc"] - h * conc) * cfg.rho_ice * cfg.L_f / dt
    assert float(latent_growth[0]) > 0
    np.testing.assert_allclose(latent_growth, conc * upward, rtol=1e-9, atol=1e-8)
