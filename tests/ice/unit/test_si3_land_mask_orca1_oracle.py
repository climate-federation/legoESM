"""ORCA1 oracle: the SI3 surface ports apply NEMO's land mask ``smask0``.

NEMO 5.0.1 multiplies ``smask0`` into the ice albedo (icealb.F90:179-182),
blk_ice_2's net longwave, total/solid precipitation and rain/snow/SST heat
contents (sbcblk.F90), and ice_flx_other's friction velocity, ice switch and
lead/freezing energies (icesbc.F90).  The ports carried none of them, so on
land cells they returned non-NEMO albedo, qns_ice, precipitation, emp/qemp and
qlead (tier-4 surface finding S1; rel up to 2.0, land only).

Fixture ``fixtures/orca1_t4/si3_land_mask.npz`` (+ ``.json`` provenance):
seeded land / wet-ice / wet-open cells of the tier-4 NEMO 5.0.1 ORCA1 records
(kt 17281, day-30 restart, 32 ranks; tier4/blkice, albedo, flxother).  Wet
flx_other cells exclude the 5.0.1-vs-5.0.2 landfast cells (separate defect).
Tolerance: rel 1e-12 with atol 1e-12 * max|field| (vote V11); the NEMO 20-seed
spreads of these fields are >= that on wet cells.
"""
from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

FIX = pathlib.Path(__file__).parent / "fixtures" / "orca1_t4" / "si3_land_mask.npz"
TOL = 1e-12
BLK_OUT = ("qla_ice", "dqla_ice", "qns_ice", "dqns_ice", "evap_ice", "devap_ice", "tprecip", "sprecip",
           "emp_oce", "emp_ice", "emp_tot", "qemp_oce", "qemp_ice", "qns_tot", "qprec_ice", "qevap_ice")
FLX_OUT = ("qsb_ice_bot", "fhld", "qlead")


@pytest.fixture(autouse=True)
def _fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


@pytest.fixture(scope="module")
def fx():
    z = np.load(FIX)
    return {k: z[k] for k in z.files}


def _close(port, nemo, name):
    port = np.asarray(port, np.float64)
    assert np.all(np.isfinite(port)), name
    np.testing.assert_allclose(port, nemo, rtol=TOL, atol=TOL * float(np.max(np.abs(nemo))), err_msg=name)


def _sec(fx, prefix):
    return {k.split(":", 1)[1]: v for k, v in fx.items() if k.startswith(prefix + ":")}


def _blk(fx, mask=None):
    from legoesm import constants
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
    from legoesm.ice.sea_ice import _nemo_si3_blk_ice_2
    i = {k: jnp.asarray(v) for k, v in _sec(fx, "blk2_in").items()}
    return _nemo_si3_blk_ice_2(
        T_surface=i["t_su"], h_ice=i["h_i"], h_snow=i["h_s"], cloud_fraction=i["cloud_fra"],
        theta_air=i["theta_air_zt"], q_air=i["q_air_zt"], p_surface=i["slp"], lw_down=i["qlw"],
        precip=i["precip"], snow=i["snow"], qsr=i["qsr"], rho_air=i["rhoa"], wind=i["wndm_ice"],
        Ch=i["Ch_ice"], Ce=i["Ce_ice"], ice_fraction_before=i["at_i_b"], category_fraction_before=i["a_i_b"],
        sst_celsius=i["sst_m"], qns_ocean=i["qns_oce"], qsr_ocean=i["qsr_oce"], emp_ocean_raw=i["emp"],
        precip_factor=constants.precip_factor_orca1, evap_factor=constants.evap_factor_orca1,
        snow_blow_exponent=constants.snow_blow_exponent_orca1,
        land_mask=i["smask0"] if mask is None else mask, ice_constants=NEMO_SI3_CONSTANTS_CONFIG,
        switch_width=0.0)


def _alb(fx, mask=None, **arrays):
    from legoesm.ice.sea_ice import _nemo_si3_ice_albedo
    i = {k: jnp.asarray(v) for k, v in _sec(fx, "alb_in").items()}
    i.update(arrays)
    return _nemo_si3_ice_albedo(i["t_su"], i["h_i"], i["h_s"], i["cloud_fra"],
                                land_mask=i["smask0"] if mask is None else mask, switch_width=0.0)


def _flx(fx, mask=None, **arrays):
    from legoesm.ice.sea_ice import _nemo_si3_ice_flx_other
    i = {k: jnp.asarray(v) for k, v in _sec(fx, "flx_in").items()}
    i.update(arrays)
    s = {k: float(v) for k, v in _sec(fx, "flx_scalar").items()}
    return _nemo_si3_ice_flx_other(
        ice_fraction=i["at_i"], ice_fraction_before=i["at_i_b"], ice_volume=i["vt_i"], u_ice=i["u_ice"],
        u_ice_west=i["u_ice_w"], v_ice=i["v_ice"], v_ice_south=i["v_ice_s"], u_ocean=i["ssu_m"],
        u_ocean_west=i["ssu_m_w"], v_ocean=i["ssv_m"], v_ocean_south=i["ssv_m_s"], drag_io=s["rn_Cd_io"],
        frq=i["frq_m"], qsr_ocean=i["qsr_oce"], qns_ocean=i["qns_oce"], qemp_ocean=i["qemp_oce"],
        ocean_layer_thickness=i["e3t_m"], sst_celsius=i["sst_m"], T_bottom=i["t_bo"], dt=s["rDt_ice"],
        rho_ocean=s["rho0"], c_ocean=s["rcp"], T0=s["rt0"], ice_epsilon=1.0e-10,
        max_ice_fraction=i["rn_amax_2d"], lead_cut_version="5.0.1", switch_width=0.0,
        land_mask=i["smask0"] if mask is None else mask)


def test_fixture_has_land_and_wet_cells(fx):
    for sec in ("blk2", "alb", "flx"):
        g = fx[f"{sec}_group"]
        assert {"land", "wet_ice", "wet_open"} <= set(g.tolist()), sec
        assert np.all(fx[f"{sec}_in:smask0"][g == "land"] == 0.0)
        assert np.all(fx[f"{sec}_in:smask0"][g != "land"] == 1.0)


def test_blk_ice_2_matches_nemo_on_every_cell(fx):
    out = _blk(fx)
    for k in BLK_OUT:
        _close(out[k], fx[f"blk2_nemo:{k}"], k)


def test_albedo_matches_nemo_on_every_cell(fx):
    _close(_alb(fx), fx["alb_nemo:alb_ice"], "alb_ice")


def test_flx_other_matches_nemo_on_every_cell(fx):
    out = _flx(fx)
    for k in FLX_OUT:
        _close(out[k], fx[f"flx_nemo:{k}"], k)


def test_land_cells_need_the_mask(fx):
    """Control: with the mask forced to 1 the land cells no longer match NEMO."""
    land = fx["blk2_group"] == "land"
    out = _blk(fx, mask=jnp.ones_like(jnp.asarray(fx["blk2_in:smask0"])))
    assert not np.allclose(np.asarray(out["qns_ice"])[land], fx["blk2_nemo:qns_ice"][land], rtol=TOL, atol=0)
    land = fx["alb_group"] == "land"
    alb = _alb(fx, mask=jnp.ones_like(jnp.asarray(fx["alb_in:smask0"])))
    assert np.max(np.abs(np.asarray(alb)[land] - fx["alb_nemo:alb_ice"][land])) > 0.01
    land = fx["flx_group"] == "land"
    q = _flx(fx, mask=jnp.ones_like(jnp.asarray(fx["flx_in:smask0"])))["qlead"]
    assert np.max(np.abs(np.asarray(q)[land] - fx["flx_nemo:qlead"][land])) > 0.0


def _fd_ratio(f, x, rel=1e-6):
    g = float(jax.grad(f)(x))
    h = rel * max(abs(float(x)), 1.0)
    fd = (float(f(x + h)) - float(f(x - h))) / (2.0 * h)
    return g, fd


def test_masked_ports_are_differentiable(fx):
    """jax.grad vs central FD (rel step 1e-6) at three wet fixture cells (vote V12)."""
    from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG  # noqa: F401
    wet = np.flatnonzero(fx["blk2_group"] == "wet_ice")[:3]
    for c in wet:
        def qns(t, c=c):
            i = {k: jnp.asarray(v[c]) for k, v in _sec(fx, "blk2_in").items()}
            sub = {f"blk2_in:{k}": (t if k == "t_su" else v) for k, v in i.items()}
            return _blk(sub)["qns_tot"]
        g, fd = _fd_ratio(qns, jnp.asarray(fx["blk2_in:t_su"][c]))
        assert np.isfinite(g) and g != 0.0 and abs(g / fd - 1.0) < 1e-4, (c, g, fd)
    wet = np.flatnonzero(fx["alb_group"] == "wet_ice")[:3]
    for c in wet:
        def alb(h, c=c):
            i = {k: v[c] for k, v in _sec(fx, "alb_in").items()}
            return _alb({f"alb_in:{k}": v for k, v in i.items()}, h_i=h)
        g, fd = _fd_ratio(alb, jnp.asarray(fx["alb_in:h_i"][c]))
        assert np.isfinite(g) and abs(g - fd) <= 1e-4 * max(abs(fd), 1e-12), (c, g, fd)
    wet = np.flatnonzero(fx["flx_group"] == "wet_ice")[:3]
    for c in wet:
        def qsb(s, c=c):
            i = {k: v[c] for k, v in _sec(fx, "flx_in").items()}
            sub = {f"flx_in:{k}": v for k, v in i.items()}
            sub.update({k: v for k, v in fx.items() if k.startswith("flx_scalar:")})
            return _flx(sub, sst_m=s)["qsb_ice_bot"]
        g, fd = _fd_ratio(qsb, jnp.asarray(fx["flx_in:sst_m"][c]))
        assert np.isfinite(g) and abs(g - fd) <= 1e-4 * max(abs(fd), 1e-12), (c, g, fd)


def test_flx_other_gradient_is_finite_on_land(fx):
    """Masked friction is 0 on land: the friction velocity's gradient must stay finite there (vote V15)."""
    land = np.flatnonzero(fx["flx_group"] == "land")[:3]
    for c in land:
        sub = {k: (v[c] if v.ndim else v) for k, v in fx.items() if k.startswith(("flx_in:", "flx_scalar:"))}
        for name in ("u_ice", "sst_m"):
            g = jax.grad(lambda x, name=name: sum(jnp.sum(v) for v in _flx(sub, **{name: x}).values()))(
                jnp.asarray(sub[f"flx_in:{name}"]))
            assert np.isfinite(float(g)), (c, name)
