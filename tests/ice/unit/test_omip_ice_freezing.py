"""Gap 6: live surface liquidus, salt/heat ledgers, and real driver forwarding."""
from __future__ import annotations

import ast
import inspect
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.tripole import create_synthetic_tripole
from legoesm.ice import SeaIceConfig, step_sea_ice
from legoesm.ice.config import BrineConfig
from legoesm.ocean.eos import nemo_eos_fzp
from legoesm.ocean.freshwater import salt_flux_salinity_tendency
from legoesm.ocean.state import OceanSurfaceForcing
import scripts.run.run_omip_core2 as runner
from tests.ice.unit.test_omip_bulk_snow import _fixed_bulk, _state_and_forcing
from tests.ice.unit.test_omip_ice_optics import _actual_main_blend

SCHEMES = ("constant", "linear_S", "unesco", "nemo_teos10")
# Independent NEMO 5.0.1 eosbn2.F90:1674-1701 reference coefficients.
FZP_COEFF = (-5.87701e-2, 2.07679e-2, -3.12775e-2,
             2.28348e-2, -9.64972e-3, 1.46873e-3)
SA0 = 35.16504
LINEAR = -0.0575
UNESCO = (LINEAR, 1.710523e-3, -2.154996e-4)
SHAPE = (8, 12)
DT = 60.0
DZ = 2.0
SAMPLES = (5.0, 10.0, 35.0, 40.0)


@pytest.fixture(autouse=True)
def fp64():
    old = get_policy()
    old_x64 = jax.config.jax_enable_x64
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(old)
    jax.config.update("jax_enable_x64", old_x64)


def _reference(s, scheme):
    s = np.asarray(s)
    if scheme == "constant":
        return np.full_like(s, constants.T_freeze_ocean)
    if scheme == "linear_S":
        depression = LINEAR * np.maximum(s, 0)
    elif scheme == "unesco":
        s = np.maximum(s, 0)
        depression = sum(c * s**p for c, p in zip(UNESCO, (1, 1.5, 2)))
    else:
        x = np.sqrt(np.abs(s) / SA0)
        depression = s * sum(c * x**k for k, c in enumerate(FZP_COEFF))
    return constants.T_freeze + depression


def _main_scheme(argv):
    args = runner._build_arg_parser().parse_args(argv)
    tree = ast.parse(inspect.getsource(runner.main))
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_ice_freeze_scheme"
                     for t in n.targets)]
    assert len(nodes) == 1
    env = dict(vars(runner), args=args)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), inspect.getsourcefile(runner), "exec"), env)
    return env["_ice_freeze_scheme"]


def _sf(q=0.0):
    z = jnp.zeros(SHAPE)
    return OceanSurfaceForcing(tau_x=z, tau_y=z, q_net=z + q, sw_down=z)


def _main_step(st, forcing, sst, salinity, cfg, scheme, sf=None):
    """Execute the actual main host step call; a missing keyword must fail."""
    tree = ast.parse(inspect.getsource(runner.main))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "step_sea_ice"]
    assert len(calls) == 1
    # A poisoned deep level ensures main forwards local SURFACE salinity.
    s3d = jnp.stack([salinity, salinity + 20], axis=-1)
    env = dict(vars(runner), step_sea_ice=step_sea_ice,
               ice_state=st, atm_ice=forcing, sst_K=sst,
               ocn_u=jnp.zeros(SHAPE), ocn_v=jnp.zeros(SHAPE),
               ice_config=cfg, dt=DT, grid=create_synthetic_tripole(*SHAPE),
               state=SimpleNamespace(S=SimpleNamespace(data=s3d)),
               sf=_sf() if sf is None else sf,
               z_coord=SimpleNamespace(dz_ref=np.array([DZ])),
               _ice_freeze_scheme=scheme)
    return eval(compile(ast.Expression(calls[0]), inspect.getsourcefile(runner), "eval"), env)


def _case(ncat, *, empty=False, temperature=None):
    st, forcing = _state_and_forcing(ncat)
    dims = SHAPE if ncat == 1 else SHAPE + (ncat,)
    h = jnp.ones(dims) if ncat == 1 else jnp.broadcast_to(jnp.array([0.1, 0.4, 1., 2., 4.]), dims)
    a = jnp.ones(dims) / ncat
    temp = constants.T_freeze - 8 if temperature is None else temperature
    st = st._replace(
        h_ice=st.h_ice.replace(data=h * (not empty)),
        concentration=st.concentration.replace(data=a * (not empty)),
        T_ice=st.T_ice.replace(data=jnp.full(dims, temp)),
        h_snow=st.h_snow.replace(data=jnp.zeros(dims)),
        S_ice=st.S_ice.replace(data=jnp.full(dims, 8.0)))
    forcing = forcing._replace(
        sw_down=jnp.zeros(SHAPE),
        lw_down=jnp.full(SHAPE, constants.sigma_sb * temp**4))
    cfg = SeaIceConfig(brine=BrineConfig(enabled=True), n_categories=ncat,
                       itd_remap="lipscomb2001" if ncat > 1 else "simple",
                       lead_freeze_source="nemo_qlead")
    s = jnp.broadcast_to(jnp.tile(jnp.array(SAMPLES), 3), SHAPE)
    return st, forcing, cfg, s


def _sumcat(x, ncat):
    return jnp.sum(x, axis=-1) if ncat > 1 else x


def _ice_mass(st, cfg, salt=False):
    x = st.h_ice.data * st.concentration.data * cfg.rho_ice
    if salt:
        x = x * st.S_ice.data / 1000
    return _sumcat(x, cfg.n_categories)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_actual_main_selection_and_inactive_refusal(scheme):
    assert _main_scheme(["--prognostic-sea-ice", "--ice-freeze-scheme", scheme]) == scheme
    with pytest.raises(ValueError, match="requires --prognostic-sea-ice"):
        _main_scheme(["--ice-freeze-scheme", scheme])


def test_defaults_unknown_and_unwired_fesom():
    parser = runner._build_arg_parser()
    assert parser.parse_args([]).ice_freeze_scheme is None
    assert _main_scheme([]) == "constant"
    assert runner._ice_freezing_temperature_K(None, "constant") is None
    with pytest.raises(SystemExit):
        parser.parse_args(["--ice-freeze-scheme", "typo"])
    with pytest.raises(ValueError, match="Unknown"):
        runner._resolve_ice_freezing("typo", True)
    with pytest.raises(ValueError, match="Unknown"):
        runner._ice_freezing_temperature_K(jnp.array(SAMPLES), "typo")
    for scheme in SCHEMES:
        args = parser.parse_args(["--grid", "fesom", "--fesom-mesh-dir", "/unused",
                                  "--ice-freeze-scheme", scheme])
        with pytest.raises(SystemExit, match="ice-freeze-scheme"):
            runner.validate_fesom_stage(args, parser)


@pytest.mark.parametrize("scheme", SCHEMES[1:])
def test_eos_reference_and_kelvin_units(scheme):
    s = jnp.array((0.0,) + SAMPLES)
    tf = runner._ice_freezing_temperature_K(s, scheme)
    np.testing.assert_allclose(tf, _reference(s, scheme), atol=1e-12, rtol=0)
    assert float(tf[0]) == constants.T_freeze
    assert np.all(np.diff(tf) < 0)


@pytest.mark.parametrize("ncat", [1, 5])
@pytest.mark.parametrize("scheme", SCHEMES)
def test_actual_main_lead_salt_heat_budget(monkeypatch, ncat, scheme):
    _fixed_bulk(monkeypatch)
    st, forcing, cfg, s = _case(ncat, empty=True)
    sst = jnp.full(SHAPE, constants.T_freeze - 1)
    sf = _sf(-100.0)
    new, resp = _main_step(st, forcing, sst, s, cfg, scheme, sf)
    tf = _reference(s, scheme)
    capacity = constants.rho_ocean * constants.c_p_seawater * DZ
    latent = np.maximum(capacity * (tf - sst) - sf.q_net * DT, 0)
    mass = latent / cfg.L_f
    np.testing.assert_allclose(_ice_mass(new, cfg), mass, atol=1e-10, rtol=1e-11)
    np.testing.assert_allclose(resp.freshwater_flux, -mass / DT, atol=1e-11)
    np.testing.assert_allclose(resp.salt_flux, -mass * cfg.brine.S_ice_new / (1000 * DT), atol=1e-12)
    np.testing.assert_allclose(resp.ocean_heat_extraction, -latent / DT, atol=1e-8)
    mixed = _actual_main_blend(sf, resp, cfg)
    np.testing.assert_allclose(mixed.salt_flux * DT + _ice_mass(new, cfg, salt=True), 0, atol=1e-10)
    ocean_final = sst + mixed.q_net * DT / capacity
    np.testing.assert_allclose(ocean_final, np.maximum(sst + sf.q_net * DT / capacity, tf), atol=1e-12)
    if scheme != "constant":
        assert np.any(mass > 0) and np.any(mass == 0)
        assert np.any(mixed.salt_flux < 0)


@pytest.mark.parametrize("ncat", [1, 5])
@pytest.mark.parametrize("warming", [0.1, 10.0])
def test_basal_growth_melt_and_ocean_salt_signs(monkeypatch, ncat, warming):
    _fixed_bulk(monkeypatch)
    st, forcing, cfg, s = _case(ncat)
    tf = _reference(s, "nemo_teos10")
    sst = jnp.asarray(tf + warming)
    new, resp = _main_step(st, forcing, sst, s, cfg, "nemo_teos10")
    h, a, t = map(np.asarray, (st.h_ice.data, st.concentration.data, st.T_ice.data))
    base = tf[..., None] if ncat > 1 else tf
    cap_dt = cfg.rho_ice * cfg.c_ice * h / (2 * DT)
    # Existing combined_conductance retains its numerical snow floor even
    # for snow-free columns; no conductivity/default change in this gap.
    conductance = 1 / (h / cfg.k_ice + cfg.snow.h_snow_min / cfg.snow.k_snow)
    skin = (cap_dt * t + conductance * base) / (cap_dt + conductance)
    cond = conductance * (base - skin)
    ocean_heat = cfg.ocean_heat_transfer_coeff * warming
    dm = DT * (cond - ocean_heat) / cfg.L_f * a
    expected_mass = _ice_mass(st, cfg) + _sumcat(dm, ncat)
    np.testing.assert_allclose(_ice_mass(new, cfg), expected_mass, atol=1e-10, rtol=0)
    dsalt = _sumcat(dm * np.where(dm > 0, cfg.brine.S_ice_new, st.S_ice.data) / 1000, ncat)
    mixed = _actual_main_blend(_sf(), resp, cfg)
    np.testing.assert_allclose(mixed.salt_flux, -dsalt / DT, atol=1e-12)
    np.testing.assert_allclose(_ice_mass(new, cfg, True) - _ice_mass(st, cfg, True) + mixed.salt_flux * DT, 0, atol=1e-10)
    np.testing.assert_allclose(resp.freshwater_flux, -_sumcat(dm, ncat) / DT, atol=1e-11)
    np.testing.assert_allclose(mixed.q_net, -ocean_heat, atol=1e-8)
    if warming == 10.0:
        assert np.all(mixed.salt_flux > 0) and np.all(resp.freshwater_flux > 0)
    else:
        assert np.all(mixed.salt_flux < 0) and np.all(resp.freshwater_flux < 0)
    # Joint water/salt exchange: independently reconstruct mixed-layer salt
    # mass. This distinguishes salt conservation from mere salinity change.
    ocean_mass = constants.rho_ocean * DZ
    # Execute the real ocean salt-channel consumer as well as its mapper.
    dS_real = salt_flux_salinity_tendency(mixed.salt_flux, jnp.full(SHAPE, DZ), constants.rho_ocean)
    np.testing.assert_allclose(ocean_mass * dS_real * DT / 1000 + dsalt, 0, atol=1e-10)
    ocean_mass_new = ocean_mass + resp.freshwater_flux * DT
    ocean_salt_new = ocean_mass * s / 1000 + mixed.salt_flux * DT
    sal_new = 1000 * ocean_salt_new / ocean_mass_new
    np.testing.assert_allclose(ocean_mass_new * sal_new / 1000 + _ice_mass(new, cfg, True),
                               ocean_mass * s / 1000 + _ice_mass(st, cfg, True), atol=1e-10)
    if warming == 0.1:
        assert np.all(np.asarray(sal_new - s) > 0)
    else:
        # Melt need not freshen: these 8 PSU ice columns melt into both
        # 5 PSU and oceanic water. The salt flux stays positive in both.
        np.testing.assert_array_equal(np.sign(sal_new - s), np.sign(8.0 - s))


@pytest.mark.parametrize("ncat", [1, 5])
def test_jit_and_nonzero_surface_salinity_gradient(monkeypatch, ncat):
    _fixed_bulk(monkeypatch)
    st, forcing, cfg, s = _case(ncat)
    z = jnp.zeros(SHAPE)
    sst = z + constants.T_freeze + 1
    def run(sal):
        return step_sea_ice(st, forcing, sst, z, z, cfg, 0., DT,
                           grid=create_synthetic_tripole(*SHAPE),
                           q_open_top=z, ocean_dz_top_m=DZ,
                           ocean_freezing_temperature_K=runner._ice_freezing_temperature_K(sal, "nemo_teos10"))
    eager = run(s)
    compiled = jax.jit(run)(s)
    for a, b in zip(jax.tree_util.tree_leaves(eager), jax.tree_util.tree_leaves(compiled)):
        np.testing.assert_allclose(a, b, atol=1e-9, rtol=1e-10)
    objective = lambda sal: jnp.sum(run(sal)[1].ocean_heat_extraction)
    gradient = jax.jit(jax.grad(objective))(s)
    assert np.all(np.isfinite(gradient)) and np.all(gradient > 0)
    eps = 1e-4
    fd = (objective(s + eps) - objective(s - eps)) / (2 * eps)
    np.testing.assert_allclose(jnp.sum(gradient), fd, rtol=2e-4)


def test_teos_zero_salinity_adjoint_and_nonzero_forward():
    grad0 = jax.jit(jax.grad(nemo_eos_fzp))(0.0)
    np.testing.assert_allclose(grad0, FZP_COEFF[0], atol=1e-14)
    np.testing.assert_allclose(nemo_eos_fzp(jnp.array(SAMPLES)),
                               _reference(SAMPLES, "nemo_teos10") - constants.T_freeze, atol=1e-13)


def test_shape_guard_and_default_config_unchanged(monkeypatch):
    _fixed_bulk(monkeypatch)
    st, forcing, cfg, s = _case(1)
    z = jnp.zeros(SHAPE)
    with pytest.raises(ValueError, match="SST shape"):
        step_sea_ice(st, forcing, z + constants.T_freeze, z, z, cfg, 0., DT,
                     ocean_freezing_temperature_K=jnp.ones(SHAPE + (1,)))
    custom = cfg._replace(T_freeze_ocean=constants.T_freeze_ocean + 0.2)
    default = _main_step(st, forcing, z + constants.T_freeze, s, custom, "constant")
    direct = step_sea_ice(st, forcing, z + constants.T_freeze, z, z, custom, 0., DT,
                         grid=create_synthetic_tripole(*SHAPE), q_open_top=z, ocean_dz_top_m=DZ)
    for a, b in zip(jax.tree_util.tree_leaves(default), jax.tree_util.tree_leaves(direct)):
        np.testing.assert_array_equal(a, b)
    _main_step(st, forcing, z + constants.T_freeze, s, cfg, "nemo_teos10")
    assert cfg.T_freeze_ocean == constants.T_freeze_ocean
    assert cfg.T_melt_surface == constants.T_freeze


@pytest.mark.parametrize("brine_enabled,dynamics", [(True, "none"), (False, "none"), (False, "free_drift")])
def test_legacy_lead_gate_and_ice_free_skin_use_boundary(monkeypatch, brine_enabled, dynamics):
    _fixed_bulk(monkeypatch, sensible=100.0)
    st, forcing, cfg, s = _case(1, empty=True)
    cfg = cfg._replace(lead_freeze_source="ice_skin", dynamics=dynamics,
                       brine=BrineConfig(enabled=brine_enabled))
    sst = jnp.full(SHAPE, constants.T_freeze - 1)
    new, _ = _main_step(st, forcing, sst, s, cfg, "nemo_teos10")
    tf = _reference(s, "nemo_teos10")
    np.testing.assert_array_equal(np.asarray(_ice_mass(new, cfg)) > 0, sst <= tf)
    np.testing.assert_allclose(new.T_ice.data, tf, atol=1e-12)


def test_surface_melt_point_stays_freshwater(monkeypatch):
    _fixed_bulk(monkeypatch)
    st, forcing, cfg, s = _case(1, temperature=constants.T_freeze)
    forcing = forcing._replace(sw_down=jnp.full(SHAPE, 300.0))
    sst = jnp.asarray(_reference(s, "nemo_teos10") + 0.1)
    new, _ = _main_step(st, forcing, sst, s, cfg, "nemo_teos10")
    assert np.all(_ice_mass(new, cfg) < _ice_mass(st, cfg))
    np.testing.assert_allclose(new.T_ice.data, constants.T_freeze, atol=1e-12)
