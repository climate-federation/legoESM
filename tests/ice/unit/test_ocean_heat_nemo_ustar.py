"""NEMO SI3 ocean-to-ice sensible heat (``ocean_heat_flux_scheme="nemo_ustar"``).

Oracle: NEMO 5.0.1 src/ICE/icesbc.F90 ice_flx_other, lines 327-386:
  zfric = rn_Cd_io*|u_ice-u_oce|^2 (dynamics on) or |tau|/rho0 (off)
  qsb   = zswitch*rho0*rcp*zch*sqrt(zfric)*(sst - t_bo)             (:371)
  qsb   = zswitch*MIN(qsb, -MIN(zqfr,0)/Dt/MAX(at_i,epsi10))         (:377)
  qsb   = 0 where supercooled and vt_i >= 20 m                       (:381)
Every expected value below is computed by hand in NumPy, not by the kernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice import SeaIceConfig, step_sea_ice
from legoesm.ice.config import BrineConfig
from legoesm.ice.sea_ice import _ocean_to_ice_heat_flux
from tests.ice.unit.test_omip_bulk_snow import _state_and_forcing

RHO_CP = constants.rho_ocean * constants.c_p_seawater
ZCH = 0.0057          # icesbc.F90:321
CD_IO = 5.0e-3        # namelist_ice_ref:136 rn_Cd_io (ORCA1 does not override)
DT = 3600.0


@pytest.fixture(autouse=True)
def fp64():
    old = get_policy()
    old_x64 = jax.config.jax_enable_x64
    set_policy(PrecisionPolicy.fp64())
    jax.config.update("jax_enable_x64", True)
    yield
    set_policy(old)
    jax.config.update("jax_enable_x64", old_x64)


def _cfg(dynamics="free_drift", **kw):
    return SeaIceConfig(dynamics=dynamics, drag_ocean=CD_IO,
                        ocean_heat_flux_scheme="nemo_ustar", **kw)


def _flux(cfg, dT, *, du=0.3, dv=0.4, at=0.8, vt=1.0, dz=10.0, tau=(0.0, 0.0)):
    a = lambda x: jnp.asarray(np.atleast_1d(np.asarray(x, dtype=np.float64)))
    return np.asarray(_ocean_to_ice_heat_flux(
        a(cfg.T_freeze_ocean + dT), cfg,
        du_ice_ocean=a(du), dv_ice_ocean=a(dv),
        tau_x=a(tau[0]), tau_y=a(tau[1]),
        ice_area_total=a(at), ice_volume_total=a(vt),
        ocean_dz_top_m=dz, dt=DT))[0]


def test_formula_matches_hand_computation():
    cfg = _cfg()
    ustar = np.sqrt(CD_IO * (0.3**2 + 0.4**2))
    expected = RHO_CP * ZCH * ustar * 0.5
    np.testing.assert_allclose(_flux(cfg, 0.5), expected, rtol=1e-12)
    assert expected > 0.0     # positive = ocean -> ice (cools the ocean)


def test_no_dynamics_uses_stress_branch():
    cfg = _cfg(dynamics="none")
    ustar = np.sqrt(np.hypot(0.06, 0.08) / constants.rho_ocean)
    np.testing.assert_allclose(_flux(cfg, 0.5, tau=(0.06, 0.08)),
                               RHO_CP * ZCH * ustar * 0.5, rtol=1e-12)


def test_freezing_cap_binds_and_closes_to_freezing():
    cfg = _cfg()
    dz, dT, at = 0.01, 0.5, 0.5
    cap = RHO_CP * dz * dT / DT / at
    assert RHO_CP * ZCH * np.sqrt(CD_IO * 0.25) * dT > cap   # would bind
    q = _flux(cfg, dT, at=at, dz=dz)
    np.testing.assert_allclose(q, cap, rtol=1e-12)
    # One ice step removes exactly the heat above freezing: SST -> Tf.
    np.testing.assert_allclose(q * at * DT, RHO_CP * dz * dT, rtol=1e-12)


def test_zswitch_below_epsi10():
    cfg = _cfg()
    assert _flux(cfg, 0.5, at=1e-11) == 0.0
    assert _flux(cfg, 0.5, at=1e-9) > 0.0


def test_supercooled_negative_then_zero_under_thick_pack():
    cfg = _cfg()
    ustar = np.sqrt(CD_IO * 0.25)
    np.testing.assert_allclose(_flux(cfg, -0.1, vt=1.0),
                               RHO_CP * ZCH * ustar * -0.1, rtol=1e-12)
    assert _flux(cfg, -0.1, vt=20.0) == 0.0


def test_constant_scheme_unchanged_and_unknown_raises():
    cfg = SeaIceConfig()
    assert cfg.ocean_heat_flux_scheme == "constant"
    sst = jnp.asarray([cfg.T_freeze_ocean + 0.5, cfg.T_freeze_ocean - 0.5])
    np.testing.assert_array_equal(
        np.asarray(_ocean_to_ice_heat_flux(sst, cfg)),
        np.asarray(cfg.ocean_heat_transfer_coeff * jnp.maximum(
            sst - cfg.T_freeze_ocean, 0.0)))
    with pytest.raises(ValueError, match="Unknown SeaIceConfig.ocean_heat_flux"):
        _ocean_to_ice_heat_flux(sst, cfg._replace(ocean_heat_flux_scheme="ustar"))


def test_legacy_path_refuses_nemo_ustar():
    st, forc = _state_and_forcing(1)
    z = jnp.zeros((8, 12))
    with pytest.raises(ValueError, match="v2"):
        step_sea_ice(st, forc, z + 272.0, z, z,
                     SeaIceConfig(dynamics="free_drift",
                                  ocean_heat_flux_scheme="nemo_ustar"),
                     U_min=0.0, dt=DT)


def _step(cfg, ocean_u, sst):
    st, forc = _state_and_forcing(1)
    z = jnp.zeros((8, 12))
    return step_sea_ice(st, forc, sst, ocean_u, z, cfg, U_min=0.0, dt=DT,
                        ocean_dz_top_m=10.0)


def test_step_wires_ice_ocean_relative_velocity():
    """The full step charges the ocean conc * qsb(u_ice_new - u_ocean)."""
    shape = (8, 12)
    base = SeaIceConfig(dynamics="free_drift", brine=BrineConfig(enabled=True),
                        drag_ocean=CD_IO)
    sst = jnp.full(shape, base.T_freeze_ocean + 0.3)
    ocean_u = jnp.full(shape, 0.05)
    ref_state, ref = _step(base._replace(ocean_heat_transfer_coeff=0.0),
                           ocean_u, sst)
    new_state, out = _step(base._replace(ocean_heat_flux_scheme="nemo_ustar"),
                           ocean_u, sst)
    du = np.asarray(new_state.u_ice.data) - 0.05
    dv = np.asarray(new_state.v_ice.data)
    assert np.all(np.hypot(du, dv) > 1e-3)     # non-degenerate relative drift
    expected = RHO_CP * ZCH * np.sqrt(CD_IO * (du**2 + dv**2)) * 0.3 * 0.8
    got = np.asarray(out.ocean_heat_extraction - ref.ocean_heat_extraction)
    np.testing.assert_allclose(got, expected, rtol=1e-10)
    assert np.all(got > 0.0)


def test_jit_matches_eager_and_grad_finite():
    shape = (8, 12)
    cfg = SeaIceConfig(dynamics="free_drift", brine=BrineConfig(enabled=True),
                       ocean_heat_flux_scheme="nemo_ustar")
    sst = jnp.full(shape, cfg.T_freeze_ocean + 0.3)
    ocean_u = jnp.full(shape, 0.05)

    def heat(u, s):
        return _step(cfg, u, s)[1].ocean_heat_extraction

    eager = heat(ocean_u, sst)
    jitted = jax.jit(heat)(ocean_u, sst)
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager),
                               rtol=1e-12, atol=1e-12)
    g_u, g_s = jax.grad(lambda u, s: jnp.sum(heat(u, s)), argnums=(0, 1))(
        ocean_u, sst)
    assert np.all(np.isfinite(np.asarray(g_u)))
    assert np.all(np.isfinite(np.asarray(g_s)))
    assert np.any(np.asarray(g_s) != 0.0)
    # Zero relative velocity: the AD-safe sqrt keeps the adjoint finite.
    g0 = jax.grad(lambda d: _ocean_to_ice_heat_flux(
        jnp.asarray(cfg.T_freeze_ocean + 0.3), cfg, du_ice_ocean=d,
        dv_ice_ocean=jnp.asarray(0.0), ice_area_total=jnp.asarray(0.8),
        ice_volume_total=jnp.asarray(1.0), ocean_dz_top_m=10.0, dt=DT))(
        jnp.asarray(0.0))
    assert np.isfinite(float(g0))


def test_cli_flag_roundtrip():
    import scripts.run.run_omip_core2 as core2
    p = core2._build_arg_parser()
    assert p.parse_args(["--grid", "tripole"]).ice_ocean_heat_scheme == "constant"
    a = p.parse_args(["--grid", "tripole", "--prognostic-sea-ice",
                      "--ice-ocean-heat-scheme", "nemo_ustar"])
    assert a.ice_ocean_heat_scheme == "nemo_ustar"
    with pytest.raises(SystemExit):
        p.parse_args(["--grid", "tripole", "--ice-ocean-heat-scheme", "bogus"])
    assert "ice_ocean_heat_scheme" in core2._FESOM_WIRED_DESTS
