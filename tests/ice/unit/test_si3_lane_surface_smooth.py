"""Lane SI3 surface port (``si3_surface``): smooth default, NEMO hard limit, gradients (RULE 5).

* Hard limit: at ``SI3_HARD_SWITCHES`` the port matches the tier-4 ORCA1 records
  (``fixtures/orca1_t4/si3_land_mask.npz``; no-pond arm, no landfast cells) and reproduces, bit for
  bit, ``si3_lane_hard_pin.npz``.  That pin was written on the parent commit by ``SI3_RECORD_PIN=1``;
  it holds the pond arm and the 5.0.2 lead cuts, which have no record on this machine.
* Default: smooth minus hard per fixture cell and per in-band case, stored in
  ``si3_lane_smooth_gaps.npz`` (1e-6 relative, 1e-12 absolute).  A change to a ramp or width
  re-records it with ``SI3_RECORD_GAPS=1`` and states why.
* Gradients: inside each ramp jax.grad matches a central finite difference.
"""
from __future__ import annotations

import os
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.ice import config as ice_config  # noqa: E402
from legoesm.ice import si3_surface  # noqa: E402

FIXDIR = pathlib.Path(__file__).parent / "fixtures" / "orca1_t4"
DATA = pathlib.Path(__file__).parent / "data"
GAP_FILE = FIXDIR / "si3_lane_smooth_gaps.npz"
PIN_FILE = FIXDIR / "si3_lane_hard_pin.npz"
RECORD_GAPS = bool(os.environ.get("SI3_RECORD_GAPS"))
RECORD_PIN = bool(os.environ.get("SI3_RECORD_PIN"))
TOL = 1e-12
BLK_OUT = ("qla_ice", "dqla_ice", "qns_ice", "dqns_ice", "evap_ice", "devap_ice", "tprecip", "sprecip",
           "emp_oce", "emp_ice", "emp_tot", "qemp_oce", "qemp_ice", "qns_tot", "qprec_ice", "qevap_ice")
FLX_OUT = ("qsb_ice_bot", "fhld", "qlead")
_recorded = {}


@pytest.fixture(scope="module", autouse=True)
def _write_recorded():
    yield
    if RECORD_GAPS:
        np.savez_compressed(GAP_FILE, **_recorded)


@pytest.fixture(autouse=True)
def _fp64_policy():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    saved = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(saved)


def _assert_gap(name, gap):
    gap = np.asarray(gap, dtype=np.float64)
    if RECORD_GAPS:
        _recorded[name] = gap
        return
    np.testing.assert_allclose(gap, np.load(GAP_FILE)[name], rtol=1e-6, atol=1e-12, err_msg=name)


def _hard(cfg):
    # The parent commit has no switches field: it is what SI3_RECORD_PIN runs on.
    if "switches" not in cfg._fields:
        return cfg
    return cfg._replace(switches=ice_config.SI3_HARD_SWITCHES)


def _deck():
    """The committed 5.0.2 C1D deck, as NEMO resolved it."""
    return si3_surface.SI3SurfaceConfig.from_resolved_namelists(
        DATA / "c1d_omip_l3_v2_output.namelist.dyn", DATA / "c1d_omip_l3_v2_output.namelist.ice")


def _orca1(fx):
    """The deck with the ORCA1 record's values."""
    return _deck()._replace(
        rn_alb_sdry=constants.albedo_snow_dry_orca1, rn_alb_smlt=constants.albedo_snow_melt_orca1,
        rn_alb_idry=constants.albedo_ice_dry_orca1, rn_alb_imlt=constants.albedo_ice_melt_orca1,
        rn_alb_hpiv=constants.albedo_ice_pivot_orca1, rn_pfac=1.0, rn_efac=1.0, rn_snwblow=0.66,
        nn_snwfra=2, ld_pnd_alb=False, ln_icedyn=True,
        rn_Cd_io=float(fx["flx_scalar:rn_Cd_io"]), rDt_ice=float(fx["flx_scalar:rDt_ice"]),
    )


def _load():
    z = np.load(FIXDIR / "si3_land_mask.npz")
    return {k: z[k] for k in z.files}


def _sec(fx, prefix):
    return {k.split(":", 1)[1]: jnp.asarray(v) for k, v in fx.items() if k.startswith(prefix + ":")}


def _alb(i, cfg, **over):
    # si3_ice_alb needs a spatial axis, so a single column travels as (1,).
    i = {k: jnp.atleast_1d(v) for k, v in {**i, **over}.items()}
    z = jnp.zeros_like(i["h_i"])
    return si3_surface.si3_ice_alb(
        t_su=i["t_su"][..., None], h_i=i["h_i"][..., None], h_s=i["h_s"][..., None],
        a_ip_eff=i.get("a_ip_eff", z)[..., None], h_ip=i.get("h_ip", z)[..., None],
        h_il=i.get("h_il", z)[..., None], cloud_fra=i["cloud_fra"], smask0=i["smask0"], config=cfg,
    )[..., 0]


def _blk(i, cfg, **over):
    i = {**i, **over}
    return si3_surface.si3_blk_ice_2(
        t_su=i["t_su"][..., None], h_s=i["h_s"][..., None], h_i=i["h_i"][..., None],
        alb_ice=_alb(i, cfg)[..., None], theta_air_zt=i["theta_air_zt"], q_air_zt=i["q_air_zt"],
        slp=i["slp"], qlw=i["qlw"], precip=i["precip"], snow=i["snow"], qsr=i["qsr"], rhoa=i["rhoa"],
        wndm_ice=i["wndm_ice"], Ch_ice=i["Ch_ice"], Ce_ice=i["Ce_ice"], theta_zu_i=i["theta_air_zt"],
        q_zu_i=i["q_air_zt"], smask0=i["smask0"], sst_m=i["sst_m"], emp=i["emp"], at_i_b=i["at_i_b"],
        a_i_b=i["a_i_b"][..., None], qns_oce=i["qns_oce"], qsr_oce=i["qsr_oce"],
        cloud_fra=i["cloud_fra"], config=cfg,
    )


def _flx(i, cfg, **over):
    i = {**i, **over}
    return si3_surface.si3_ice_flx_other(
        u_ice=i["u_ice"], u_ice_w=i["u_ice_w"], v_ice=i["v_ice"], v_ice_s=i["v_ice_s"],
        ssu_m=i["ssu_m"], ssu_m_w=i["ssu_m_w"], ssv_m=i["ssv_m"], ssv_m_s=i["ssv_m_s"],
        utau=None, vtau=None, smask0=i["smask0"], at_i=i["at_i"], at_i_b=i["at_i_b"], vt_i=i["vt_i"],
        qsr_oce=i["qsr_oce"], qns_oce=i["qns_oce"], qemp_oce=i["qemp_oce"], frq_m=i["frq_m"],
        e3t_m=i["e3t_m"], sst_m=i["sst_m"], t_bo=i["t_bo"], rn_amax_2d=i["rn_amax_2d"], config=cfg,
    )


def _close(got, want, name):
    got, want = np.asarray(got).reshape(-1), np.asarray(want).reshape(-1)
    np.testing.assert_allclose(got, want, rtol=TOL, atol=TOL * np.max(np.abs(want)), err_msg=name)


def test_hard_limit_matches_the_orca1_record():
    fx = _load()
    cfg = _hard(_orca1(fx))
    _close(_alb(_sec(fx, "alb_in"), cfg), fx["alb_nemo:alb_ice"], "alb_ice")
    blk = _blk(_sec(fx, "blk2_in"), cfg)
    for k in BLK_OUT:
        _close(getattr(blk, k), fx[f"blk2_nemo:{k}"], k)
    flx = _flx(_sec(fx, "flx_in"), cfg)
    for k in FLX_OUT:
        _close(getattr(flx, k), fx[f"flx_nemo:{k}"], k)


def _pin_cases():
    """Seeded columns that reach the pond arm and every 5.0.2 lead cut."""
    rng = np.random.default_rng(2388)
    n = 96
    u = lambda a, b: jnp.asarray(rng.uniform(a, b, n))  # noqa: E731
    h_s = jnp.where(jnp.arange(n) % 3 == 0, 0.0, u(0.0, 0.2))
    alb = dict(t_su=constants.T_freeze + u(-3.0, 0.5), h_i=jnp.exp(u(-4.5, 1.2)), h_s=h_s, a_ip_eff=u(0.0, 0.6),
               h_ip=u(0.0, 0.3), h_il=u(0.0, 0.05), cloud_fra=u(0.0, 1.0),
               smask0=jnp.where(jnp.arange(n) % 11 == 0, 0.0, 1.0))
    packed = jnp.arange(n) % 2 == 0
    slow = u(0.0, 1.0) * jnp.where(jnp.arange(n) % 4 < 2, 1.0e-3, 1.0)
    flx = dict(u_ice=slow, u_ice_w=slow, v_ice=0.5 * slow, v_ice_s=0.5 * slow, ssu_m=u(-0.1, 0.1), ssu_m_w=u(-0.1, 0.1),
               ssv_m=u(-0.1, 0.1), ssv_m_s=u(-0.1, 0.1), smask0=alb["smask0"],
               at_i=jnp.where(packed, u(0.9985, 1.0), u(0.0, 0.99)), at_i_b=u(0.0, 1.0),
               vt_i=jnp.where(jnp.arange(n) % 5 == 0, u(19.0, 25.0), u(0.0, 6.0)),
               qsr_oce=u(0.0, 200.0), qns_oce=u(-300.0, 100.0), qemp_oce=u(-5.0, 5.0), frq_m=u(0.5, 1.0),
               e3t_m=u(1.0, 10.0), sst_m=u(-2.2, 1.0), t_bo=constants.T_freeze + u(-2.0, -1.7),
               rn_amax_2d=jnp.full((n,), 0.997))
    return alb, flx


def _pin_values():
    alb, flx = _pin_cases()
    deck = _hard(_deck())
    out = {
        "alb_nopond": _alb(alb, deck._replace(ld_pnd_alb=False)),
        "alb_pond": _alb(alb, deck._replace(ld_pnd_alb=True)),
    }
    other = _flx(flx, deck._replace(ln_icedyn=True))
    out.update({f"flx_{k}": getattr(other, k) for k in FLX_OUT})
    return {k: np.asarray(v) for k, v in out.items()}


def test_hard_limit_is_bitwise_the_parent_commit():
    got = _pin_values()
    if RECORD_PIN:
        np.savez_compressed(PIN_FILE, **got)
        return
    pin = np.load(PIN_FILE)
    assert set(pin.files) == set(got)
    for k, v in got.items():
        np.testing.assert_array_equal(v.view(np.int64), pin[k].view(np.int64), err_msg=k)
    cut = np.asarray(_pin_cases()[1]["at_i"]) > 0.996
    assert np.any(pin["flx_qlead"][cut] == 0.0) and np.any(pin["flx_qlead"] != 0.0)


def test_default_is_smooth_and_the_snow_switch_is_straight_through():
    assert _deck().switches == ice_config.SI3SwitchConfig()
    assert _deck().switches.relative_width > 0.0
    assert _deck().switches.snow_ramp_width_m == 2.0e-4


def test_snow_surrogate_leaves_the_lane_forward_bitwise():
    """Forward value of the snow-free test is NEMO's hard test: the surrogate width changes no output."""
    fx = _load()
    cfg = _orca1(fx)
    off = cfg._replace(switches=cfg.switches._replace(snow_ramp_width_m=0.0))
    for blk, alb in ((_blk, _alb), (jax.jit(_blk, static_argnums=1), jax.jit(_alb, static_argnums=1))):
        i = _sec(fx, "blk2_in")
        a, b = blk(i, cfg), blk(i, off)
        for k in a._fields:
            np.testing.assert_array_equal(np.asarray(getattr(a, k)).view(np.int64),
                                          np.asarray(getattr(b, k)).view(np.int64), err_msg=k)
        i = _sec(fx, "alb_in")
        np.testing.assert_array_equal(np.asarray(alb(i, cfg)).view(np.int64), np.asarray(alb(i, off)).view(np.int64))


def test_smooth_default_gap_per_fixture_cell():
    fx = _load()
    soft, hard = _orca1(fx), _hard(_orca1(fx))
    i = _sec(fx, "alb_in")
    _assert_gap("alb:alb_ice", _alb(i, soft) - _alb(i, hard))
    i = _sec(fx, "blk2_in")
    a, b = _blk(i, soft), _blk(i, hard)
    for k in ("qns_tot", "qprec_ice", "qemp_oce", "qemp_ice", "qtr_ice_top", "qsr_ice"):
        assert np.all(np.isfinite(getattr(a, k))), k
        _assert_gap(f"blk2:{k}", getattr(a, k) - getattr(b, k))
    i = _sec(fx, "flx_in")
    a, b = _flx(i, soft), _flx(i, hard)
    for k in FLX_OUT:
        assert np.all(np.isfinite(getattr(a, k))), k
        _assert_gap(f"flx:{k}", getattr(a, k) - getattr(b, k))


def test_smooth_default_gap_on_the_pond_and_cut_columns():
    alb, flx = _pin_cases()
    soft, hard = _deck(), _hard(_deck())
    for pond in (False, True):
        s, h = soft._replace(ld_pnd_alb=pond), hard._replace(ld_pnd_alb=pond)
        _assert_gap(f"pin:alb_pond={pond}", _alb(alb, s) - _alb(alb, h))
    s, h = soft._replace(ln_icedyn=True), hard._replace(ln_icedyn=True)
    a, b = _flx(flx, s), _flx(flx, h)
    for k in FLX_OUT:
        assert np.all(np.isfinite(getattr(a, k))), k
        _assert_gap(f"pin:flx_{k}", getattr(a, k) - getattr(b, k))


def _switch(name, f, cfg, x0, h, tol=1e-4):
    """One in-band case of ``f(x, cfg)``: smooth differs from hard by the stored gap, and grad == FD."""
    x = jnp.asarray(x0)
    v = lambda t, c: jnp.sum(f(t, c))  # noqa: E731
    soft, hard = float(v(x, cfg)), float(v(x, _hard(cfg)))
    assert soft != hard, (name, soft, hard)
    _assert_gap(name, soft - hard)
    g = float(jax.grad(lambda t: v(t, cfg))(x))
    fd = (float(v(x + h, cfg)) - float(v(x - h, cfg))) / (2.0 * h)
    assert np.isfinite(g) and g != 0.0, (name, g, fd)
    assert abs(g - fd) <= tol * abs(fd), (name, g, fd)
    check_grads(lambda t: v(t, cfg), (x,), order=1, modes=("fwd", "rev"), eps=h, rtol=tol, atol=tol * abs(fd))


def _col(fx, section, pick):
    idx = np.flatnonzero((fx[f"{section}_group"] == "wet_ice") & pick)
    c = int(idx[0])
    return {k.split(":", 1)[1]: jnp.asarray(v[c]) for k, v in fx.items()
            if k.startswith(f"{section}_in:") and np.ndim(v)}


def test_gradients_inside_the_albedo_ramps():
    fx = _load()
    cfg = _orca1(fx)
    tf = constants.T_freeze
    bare = _col(fx, "alb", fx["alb_in:h_s"] == 0.0)
    bare["h_s"] = jnp.asarray(0.0)
    bare["h_i"] = jnp.asarray(2.0)
    _switch("alb:melt", lambda T, c: _alb(bare, c, t_su=T), cfg, tf - 0.005, 1e-5)
    cold = {**bare, "t_su": jnp.asarray(tf - 1.0)}
    for name, h0 in (("thin", 0.05), ("pivot", cfg.rn_alb_hpiv)):
        _switch(f"alb:{name}", lambda h, c: _alb(cold, c, h_i=h), cfg, h0 * 1.005, h0 * 1e-6)
    snowy = {**bare, "h_s": jnp.asarray(0.02)}
    _switch("alb:snow_melt", lambda T, c: _alb(snowy, c, t_su=T), cfg, tf - 0.005, 1e-5)
    pond = cfg._replace(ld_pnd_alb=True, rn_alb_dpnd=0.27, rn_alb_lpnd=0.07)
    wet = {**cold, "a_ip_eff": jnp.asarray(0.3), "h_ip": jnp.asarray(0.1)}
    _switch("alb:pond_lid", lambda h, c: _alb(wet, c, h_il=h), pond, 0.02 * 1.005, 2e-8)
    # the pond cap at 1 - snow fraction, band 1e-5 wide
    fra = 0.02 / (0.02 + constants.snow_cover_depth_nemo)
    _switch("alb:pond_cap", lambda a, c: _alb({**wet, "h_s": jnp.asarray(0.02)}, c, a_ip_eff=a), pond,
            1.0 - fra - 2.0e-6, 1e-8)


def test_nn_snwfra_zero_reads_the_snow_width():
    fx = _load()
    i = _col(fx, "alb", fx["alb_in:h_s"] == 0.0)
    cfg = _orca1(fx)._replace(nn_snwfra=0)
    off = cfg._replace(switches=cfg.switches._replace(snow_ramp_width_m=0.0))
    f = lambda h, c: _alb(i, c, h_s=h)  # noqa: E731
    h = jnp.asarray(0.5e-4)
    v = lambda t, c: jnp.sum(f(t, c))  # noqa: E731
    # forward: NEMO's h_s > 0 in every mode (straight-through)
    assert float(v(h, cfg)) == float(v(h, _hard(cfg))) == float(v(h, off))
    g, g_off = (float(jax.grad(lambda t: v(t, c))(h)) for c in (cfg, off))
    assert np.isfinite(g) and g != g_off                   # the surrogate carries a gradient
    assert np.isfinite(float(jax.grad(lambda t: v(t, cfg))(jnp.asarray(0.0))))   # snow-free cell


def test_gradients_inside_the_blk_ice_2_ramps():
    fx = _load()
    cfg = _orca1(fx)
    tf = constants.T_freeze
    pick = (fx["blk2_in:h_s"] == 0.0) & (fx["blk2_in:qsr"] > 10.0)
    b = _col(fx, "blk2", pick)
    b["snow"] = jnp.maximum(b["snow"], 1e-5)
    _switch("blk2:theta_cap", lambda t, c: _blk(b, c, theta_air_zt=t).qprec_ice, cfg, tf - 0.005, 1e-5)
    _switch("blk2:thin_sw", lambda h, c: _blk(b, c, h_i=h).qtr_ice_top, cfg, 0.1 * 0.995, 1e-8)
    floor = constants.T_goff_floor_nemo
    _switch("blk2:goff", lambda T, c: _blk(b, c, t_su=T).qla_ice, cfg, floor + 0.005, 1e-5, tol=1e-3)


def test_gradients_inside_the_flx_other_ramps():
    fx = _load()
    cfg = _orca1(fx)
    warm = fx["flx_in:sst_m"] + constants.T_freeze - fx["flx_in:t_bo"] > 0.01
    i = _col(fx, "flx", warm)
    eps = 1.0e-10
    _switch("flx:ice_switch", lambda a, c: _flx(i, c, at_i=a).qsb_ice_bot, cfg, eps * (1.0 - 0.005), 1e-16)
    zero = jnp.asarray(0.0)
    rest = dict(u_ice=zero, u_ice_w=zero, v_ice=zero, v_ice_s=zero)
    grown = {**i, **rest, "vt_i": jnp.asarray(5.0), "qsr_oce": jnp.asarray(0.0),
             "qns_oce": jnp.asarray(-300.0), "qemp_oce": jnp.asarray(0.0), "at_i_b": jnp.asarray(0.5),
             "smask0": jnp.asarray(1.0), "sst_m": i["t_bo"] - constants.T_freeze}
    edge = float(grown["rn_amax_2d"]) - 0.001
    _switch("flx:packed", lambda a, c: _flx(grown, c, at_i=a).qlead, cfg, edge - 5.0e-6, 1e-9)
    open_ice = {**grown, "at_i": jnp.asarray(0.5)}
    _switch("flx:thick_cut", lambda v, c: _flx({**grown, "at_i": jnp.asarray(edge + 1e-3)}, c, vt_i=v).qlead,
            cfg, 3.0 * (1.0 - 0.005), 1e-7)
    q = lambda s, c: _flx(open_ice, c, u_ice=s, u_ice_w=s, at_i=jnp.asarray(edge + 1e-3)).qlead  # noqa: E731
    _switch("flx:slow", q, cfg, 5.0e-3 * (1.0 + 0.002), 1e-9)
    assert np.isfinite(float(jax.grad(lambda s: jnp.sum(q(s, cfg)))(zero)))
    # full cover, ice moving: a 1e-12 wide ramp, close to the fp64 spacing of at_i
    moving = {**open_ice, "u_ice": jnp.asarray(1.0), "u_ice_w": jnp.asarray(1.0)}
    _switch("flx:full_cover", lambda a, c: _flx(moving, c, at_i=a).qlead, cfg, 1.0 - eps - 0.5e-12, 1e-13, tol=0.1)
    # supercooled water under thick ice
    cold = {**open_ice, "at_i": jnp.asarray(0.5), "vt_i": jnp.asarray(25.0), "qns_oce": jnp.asarray(200.0)}
    _switch("flx:supercool", lambda t, c: _flx(cold, c, sst_m=t).fhld, cfg,
            float(cold["t_bo"]) - constants.T_freeze - 0.005, 1e-5)
    _switch("flx:supercool_volume", lambda v, c: _flx({**cold, "sst_m": cold["sst_m"] - 1.0}, c, vt_i=v).qlead, cfg,
            20.0 * (1.0 - 0.005), 1e-5)
    # lead floor and cap, near zero; at_i_b = 0.5 and at_i = 0.5 make both equal qns_oce / 2 or qns_oce
    calm = {**cold, "vt_i": jnp.asarray(1.0)}
    _switch("flx:fhld_floor", lambda n, c: _flx(calm, c, qns_oce=n).fhld, cfg, 0.005, 1e-6)
    _switch("flx:qlead_cap", lambda n, c: _flx(calm, c, qns_oce=n).qlead, cfg, -0.01, 1e-6)
    # qsb cap: zqfr scales with e3t, so pick e3t that puts the cap 0.005 W/m2 above the uncapped flux
    warm_i = {**calm, "sst_m": jnp.asarray(float(calm["sst_m"]) + 0.5)}
    uncapped = float(jnp.sum(_flx(warm_i, _hard(cfg), e3t_m=jnp.asarray(1.0e6)).qsb_ice_bot))
    per_metre = float(jnp.sum(_flx(warm_i, _hard(cfg), e3t_m=jnp.asarray(1.0e-6)).qsb_ice_bot)) / 1.0e-6
    _switch("flx:qsb_cap", lambda e, c: _flx(warm_i, c, e3t_m=e).qsb_ice_bot, cfg,
            (uncapped + 0.005) / per_metre, 1e-5 / per_metre)


def test_flx_other_smooth_gradient_is_finite_for_ice_at_rest():
    fx = _load()
    cfg = _orca1(fx)
    i = _col(fx, "flx", fx["flx_in:at_i"] > 0.5)
    z = jnp.asarray(0.0)
    f = lambda s: _flx(i, cfg, u_ice=s, u_ice_w=s, v_ice=z, v_ice_s=z, ssu_m=z, ssu_m_w=z,  # noqa: E731
                       ssv_m=z, ssv_m_s=z).qsb_ice_bot
    assert np.isfinite(float(jax.grad(lambda s: jnp.sum(f(s)))(z)))
