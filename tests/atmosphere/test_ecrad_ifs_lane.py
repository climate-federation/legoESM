"""ecRad 49r1 lane step (radiation/ecrad_ifs/lane.py) on the IFS-interface
oracle fixture's eight columns.

The three inputs pending the user (TSI, gases, ozone; votes L12-L14) are
injected from the fixture.  Checks: carry pack/unpack, the radintg.F90
half-level temperature rule, the traced McICA seed against the host form,
the lane step against the composed port, column energy closure (vote L24c),
a held step on the carry, eager vs jit (J1a, V51) and gradients (W1/W2)."""
from __future__ import annotations

import json
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.radiation.ecrad_ifs import ecrad  # noqa: E402
from legoesm.atmosphere.physics.radiation.ecrad_ifs import ifs_interface as ifs  # noqa: E402
from legoesm.atmosphere.physics.radiation.ecrad_ifs import lane  # noqa: E402
from legoesm.atmosphere.physics.radiation.ecrad_ifs.config import (  # noqa: E402
    ECRAD_IFS_GASES, EcradIFSConfig, EcradIFSSources)

FIXTURE = os.path.join(os.path.dirname(__file__), "oracle_fixtures",
                       "ecrad_ifs49_interface.npz")
SPREAD_ARTIFACT = os.path.join(os.path.dirname(__file__), "oracle_fixtures",
                               "spread_ecrad_ifs49_interface.json")
JIT_RTOL = 1.0e-12                 # vote V51
ORACLE_TOL = 1.0e-12               # the port's oracle bar (AGENT_BRIEF_COMMON)
FD_STEPS, FD_RTOL = (1.0e-5, 1.0e-6), 1.0e-6     # votes W1, W2
ENERGY_REL, ENERGY_ABS = 1.0e-12, 1.0e-9         # vote L24c: x max|net|, W/m^2


@pytest.fixture(scope="module")
def fx():
    with np.load(FIXTURE) as z:
        return {k: z[k] for k in z.files}


def _d(fx, n):
    return jnp.asarray(fx["dump:" + n])


def _inputs(fx):
    p_half = jnp.asarray(fx["in:pressure_hl"])
    dp = p_half[:, 1:] - p_half[:, :-1]
    ncol, nlev = fx["in:q"].shape
    src = EcradIFSSources(
        solar_irradiance=float(fx["in:solar_irradiance"]),
        gas_mmr={k: _d(fx, f"drv_in_mmr_{k}") for k in ECRAD_IFS_GASES},
        o3_mmr=_d(fx, "drv_in_o3_dp") / dp)
    kw = dict(
        T=_d(fx, "drv_in_temperature_fl"), p_full=_d(fx, "drv_in_pressure_fl"),
        p_half=p_half, T_skin=jnp.asarray(fx["in:skin_temperature"]),
        lat=jnp.arcsin(_d(fx, "drv_in_gemu")), lon=_d(fx, "drv_in_gelam"),
        q=jnp.asarray(fx["in:q"]), q_liquid=jnp.asarray(fx["in:q_liquid"]),
        q_ice=jnp.asarray(fx["in:q_ice"]), q_rain=jnp.zeros((ncol, nlev)),
        q_snow=jnp.zeros((ncol, nlev)),
        cloud_fraction=jnp.asarray(fx["in:cloud_fraction"]),
        land_fraction=_d(fx, "drv_in_lsm"),
        mu0=jnp.asarray(fx["in:cos_solar_zenith_angle"]),
        albedo_dif=jnp.asarray(fx["in:sw_albedo"][:, 0]),
        albedo_dir=jnp.asarray(fx["in:sw_albedo_direct"][:, 0]),
        emissivity=jnp.asarray(fx["in:lw_emissivity"][:, 0]),
        iday=1, elapsed_s=3600.0, g=constants.g, c_pd=constants.c_pd)
    return src, kw


def _full(fx, cfg=EcradIFSConfig(), **over):
    src, kw = _inputs(fx)
    kw.update(over)
    return lane.step(cfg, src, full_call=True, packed_carry=None, **kw)


def test_carry_pack_roundtrip_is_bitwise(fx):
    _, _, packed = _full(fx)
    nlev = fx["in:q"].shape[1]
    assert packed.shape == (fx["in:q"].shape[0], lane.carry_width(nlev))
    again = lane.pack_carry(lane.unpack_carry(packed, nlev))
    assert np.array_equal(np.asarray(again), np.asarray(packed))
    with pytest.raises(ValueError, match="width"):
        lane.unpack_carry(packed[:, :-1], nlev)


def test_half_level_temperature_is_the_radintg_rule(fx):
    _, kw = _inputs(fx)
    T, pf, ph, ts = (np.asarray(kw[k]) for k in ("T", "p_full", "p_half", "T_skin"))
    ncol, nlev = T.shape
    ref = np.empty((ncol, nlev + 1))
    for jk in range(2, nlev + 1):                  # radintg.F90:800-806, 1-based
        a, b = jk - 2, jk - 1                      # PT(JK-1), PT(JK)
        ref[:, jk - 1] = ((T[:, a] * pf[:, a] * (pf[:, b] - ph[:, jk - 1])
                           + T[:, b] * pf[:, b] * (ph[:, jk - 1] - pf[:, a]))
                          * (1.0 / (ph[:, jk - 1] * (pf[:, b] - pf[:, a]))))
    ref[:, 0] = T[:, 0] - pf[:, 0] * (T[:, 0] - ref[:, 1]) / (pf[:, 0] - ph[:, 1])
    ref[:, nlev] = ts
    got = np.asarray(lane.half_level_temperature(*(kw[k] for k in ("T", "p_full", "p_half", "T_skin"))))
    np.testing.assert_allclose(got, ref, rtol=ORACLE_TOL, atol=0.0)


@pytest.mark.parametrize("nindat,nstep,tstep", [(20200115, 4, 900.0), (19790101, 0, 112.5),
                                                (19790101, 7, 450.0), (19790131, 3, 10.0)])
def test_traced_seed_equals_host_seed(fx, nindat, nstep, tstep):
    gelam, gemu = _d(fx, "drv_in_gelam"), _d(fx, "drv_in_gemu")
    host = ifs.mcica_seed(nindat, nstep, tstep, gelam, gemu)
    traced = jax.jit(ifs.mcica_seed_traced)(nindat % 100, nstep * tstep, gelam, gemu)
    assert np.array_equal(np.asarray(host), np.asarray(traced))


def test_lane_step_is_the_composed_port(fx):
    """The lane step = radiation_scheme + radheatn on the same inputs
    (plumbing: bands, units, seed, half levels), heating with lane g/c_pd."""
    src, kw = _inputs(fx)
    dT, fl, packed = _full(fx)
    ncol, nlev = kw["T"].shape
    cfg = EcradIFSConfig()
    seed = ifs.mcica_seed_traced(1, 3600.0, kw["lon"], jnp.sin(kw["lat"]))
    u_lw, u_sw = ecrad.draw_uniforms(seed, nlev)
    dp = kw["p_half"][:, 1:] - kw["p_half"][:, :-1]
    _, carry = ifs.radiation_scheme(
        mu0=kw["mu0"], uniforms_lw=u_lw, uniforms_sw=u_sw, sharpness=cfg.sharpness,
        solar_irradiance=src.solar_irradiance, skin_temperature=kw["T_skin"],
        albedo_dif=jnp.repeat(kw["albedo_dif"][:, None], 6, 1),
        albedo_dir=jnp.repeat(kw["albedo_dir"][:, None], 6, 1),
        spectral_emiss=jnp.repeat(kw["emissivity"][:, None], 2, 1),
        ccn_land=jnp.full((ncol,), 900.0), ccn_sea=jnp.full((ncol,), 50.0),
        gelam=kw["lon"], gemu=jnp.sin(kw["lat"]), land_sea_mask=kw["land_fraction"],
        pressure_fl=kw["p_full"], temperature_fl=kw["T"], pressure_hl=kw["p_half"],
        temperature_hl=lane.half_level_temperature(kw["T"], kw["p_full"], kw["p_half"], kw["T_skin"]),
        q=kw["q"], o3_dp=src.o3_mmr * dp, cloud_fraction=kw["cloud_fraction"],
        q_liquid=kw["q_liquid"], q_ice=kw["q_ice"], q_rain=kw["q_rain"], q_snow=kw["q_snow"],
        aerosol_old=jnp.zeros((ncol, 6, nlev)), nindat=0, nstep=0, tstep=0.0,
        **{k: src.gas_mmr[k] for k in ECRAD_IFS_GASES})
    assert np.array_equal(np.asarray(lane.pack_carry(carry)), np.asarray(packed))
    rh = ifs.radheatn(carry, kw["p_half"], kw["q"], kw["T_skin"], kw["mu0"], src.solar_irradiance)
    net = rh["frso"] + rh["frth"]
    assert np.array_equal(np.asarray(fl["net_flux"]), np.asarray(net))
    ref = -(constants.g / constants.c_pd) * (net[:, 1:] - net[:, :-1]) / dp
    assert np.array_equal(np.asarray(dT), np.asarray(ref))


def _closure(kw, dT, fl):
    dp = kw["p_half"][:, 1:] - kw["p_half"][:, :-1]
    col = np.asarray(jnp.sum(constants.c_pd / constants.g * dT * dp, axis=1))
    toa = np.asarray(fl["sw_down_toa"] - fl["sw_up_toa"] - fl["lw_up_toa"])
    sfc = np.asarray(fl["sw_net_sfc"] + fl["lw_net_sfc"])
    tol = np.maximum(ENERGY_REL * np.max(np.abs(np.asarray(fl["net_flux"])), axis=1), ENERGY_ABS)
    assert np.all(np.abs(col - (toa - sfc)) <= tol), np.abs(col - (toa - sfc))
    return col


def test_column_energy_closure_on_a_held_step_with_the_sun_moved(fx):
    """Same closure on a RADHEATN-only step at a different mu0, so the
    Manners SW update of the whole profile is inside the check."""
    src, kw = _inputs(fx)
    _, fl0, packed = _full(fx)
    kw2 = dict(kw, mu0=kw["mu0"] * 0.8, T_skin=kw["T_skin"] + 1.0)
    dT, fl, _ = lane.step(EcradIFSConfig(), src, full_call=False, packed_carry=packed, **kw2)
    _closure(kw2, dT, fl)
    assert np.max(np.abs(np.asarray(fl["sw_net_sfc"] - fl0["sw_net_sfc"]))) > 1.0


def test_column_energy_closure(fx):
    """sum(c_pd/g * dT * dp) = net(TOA) - net(sfc) = (SW_in - SW_up - OLR)
    - (SW_net_sfc + LW_net_sfc), per column (vote L24c)."""
    _, kw = _inputs(fx)
    dT, fl, _ = _full(fx)
    col = _closure(kw, dT, fl)
    assert np.all(np.isfinite(np.asarray(dT)))
    # non-vacuous: the column really heats/cools by W/m^2-scale amounts
    assert np.max(np.abs(col)) > 1.0


def test_held_step_on_the_carry(fx):
    """A held step at the call's mu0 reproduces the full step bitwise; at a
    later mu0 it changes the SW only (RADHEATN Manners update)."""
    src, kw = _inputs(fx)
    dT, fl, packed = _full(fx)
    dT2, fl2, packed2 = lane.step(EcradIFSConfig(), src, full_call=False,
                                  packed_carry=packed, **kw)
    assert np.array_equal(np.asarray(dT2), np.asarray(dT))
    assert np.array_equal(np.asarray(packed2), np.asarray(packed))
    kw2 = dict(kw, mu0=kw["mu0"] * 0.8)
    dT3, fl3, _ = lane.step(EcradIFSConfig(), src, full_call=False, packed_carry=packed, **kw2)
    assert np.max(np.abs(np.asarray(fl3["sw_down_toa"] - fl["sw_down_toa"]))) > 1.0
    np.testing.assert_array_equal(np.asarray(fl3["lw_up_toa"]), np.asarray(fl["lw_up_toa"]))


def test_missing_gas_raises(fx):
    src, kw = _inputs(fx)
    bad = src._replace(gas_mmr={k: v for k, v in src.gas_mmr.items() if k != "ccl4"})
    with pytest.raises(ValueError, match="ccl4"):
        lane.step(EcradIFSConfig(), bad, full_call=True, packed_carry=None, **kw)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    den = np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1), 0)))


def test_eager_vs_jit(fx):
    src, kw = _inputs(fx)
    cfg = EcradIFSConfig()
    f = lambda T, el: lane.step(cfg, src, full_call=True, packed_carry=None,  # noqa: E731
                                **dict(kw, T=T, elapsed_s=el))
    dTa, fa, ca = f(kw["T"], 3600.0)
    dTb, fb, cb = jax.jit(f)(kw["T"], jnp.asarray(3600.0))
    dp = kw["p_half"][:, 1:] - kw["p_half"][:, :-1]
    flux_units = np.max(np.abs(np.asarray((dTa - dTb) * dp * constants.c_pd / constants.g)))
    print(f"eager-vs-jit: net flux rel {_rel(fa['net_flux'], fb['net_flux']):.3e}, "
          f"carry rel {_rel(ca, cb):.3e}, heating rel {_rel(dTa, dTb):.3e}, "
          f"heating in flux units {flux_units:.3e} W/m^2, "
          f"max|net| {float(np.max(np.abs(np.asarray(fa['net_flux'])))):.3e}")
    # vote J1a: the SW part of a full call carries round-off amplified by the
    # radintg albedo-update cancellation (measured 2026-11-03: SW carry
    # 1.6e-9, net flux 2.8e-8, heating 4.3e-6 rel); fluxes and heating are
    # bounded by the IFS-interface oracle's own 20-seed 1-ulp spread
    # (committed artifact, controls PASS), the LW-only carry fields by V51
    with open(SPREAD_ARTIFACT) as fh:
        art = json.load(fh)
    assert all(art["controls"][c] == "PASS" for c in ("C1", "C2", "C3", "C4", "C5"))
    sp = art["spread"]
    flux_bound = max(sp[f"rh_{k}_s0"] for k in ("frso", "frth"))
    assert _rel(fa["net_flux"], fb["net_flux"]) <= flux_bound
    assert _rel(dTa, dTb) <= sp["rh_te_s0"]
    # every exported flux against the spread of the RADHEATN output it is
    # read from (sw_up_toa = incsr - frso[0], lw_up_toa = -frth[0], ...)
    src_of = {"sw_net_sfc": "frso", "lw_net_sfc": "frth", "sw_down_toa": "incsr",
              "sw_up_toa": "frso", "lw_up_toa": "frth", "sw_down_sfc": "frsod",
              "lw_down_sfc": "frthd", "sw_up_toa_clr": "frsoc", "lw_up_toa_clr": "frthc"}
    assert set(src_of) == set(fa) - {"net_flux"}
    for k, r in src_of.items():
        bound = max(JIT_RTOL, sp[f"rh_{r}_s0"])
        assert _rel(fa[k], fb[k]) <= bound, (k, _rel(fa[k], fb[k]), bound)
    nlev = kw["T"].shape[1]
    ua, ub = lane.unpack_carry(ca, nlev), lane.unpack_carry(cb, nlev)
    for k in ("emtd", "emtc", "frted", "emtdc", "emit", "lwderivative", "mu0_rad"):
        assert _rel(getattr(ua, k), getattr(ub, k)) <= JIT_RTOL, k


def test_gradients(fx):
    """Full call: d(column heating)/dT finite and nonzero.  Held step: AD vs
    central FD on the skin temperature at both voted steps."""
    src, kw = _inputs(fx)
    cfg = EcradIFSConfig()
    full = lambda T: jnp.sum(lane.step(cfg, src, full_call=True, packed_carry=None,  # noqa: E731
                                       **dict(kw, T=T))[0] ** 2)
    g = np.asarray(jax.grad(full)(kw["T"]))
    assert np.all(np.isfinite(g)) and np.max(np.abs(g)) > 0.0
    _, _, packed = _full(fx)
    held = lambda ts: jnp.sum(lane.step(cfg, src, full_call=False, packed_carry=packed,  # noqa: E731
                                        **dict(kw, T_skin=ts))[1]["lw_net_sfc"])
    ts = kw["T_skin"]
    # directional derivative along ts: AD vs central FD with relative step h
    ad_dir = float(jnp.vdot(jax.grad(held)(ts), ts))
    assert ad_dir != 0.0
    for h in FD_STEPS:
        fd = (float(held(ts + h * ts)) - float(held(ts - h * ts))) / (2.0 * h)
        assert abs(fd - ad_dir) <= FD_RTOL * abs(ad_dir), (h, fd, ad_dir)


@pytest.mark.parametrize("scale", [0.0, 1.0e-3])
def test_held_step_after_sunset(fx, scale):
    """Sunlit carry, current sun at/just above the horizon (GLM r3): the
    RADHEATN gates on the CURRENT mu0 keep fluxes and gradients finite."""
    src, kw = _inputs(fx)
    _, _, packed = _full(fx)
    kw2 = dict(kw, mu0=kw["mu0"] * scale)
    dT, fl, _ = lane.step(EcradIFSConfig(), src, full_call=False, packed_carry=packed, **kw2)
    for k, v in fl.items():
        assert np.all(np.isfinite(np.asarray(v))), k
    _closure(kw2, dT, fl)
    held = lambda ts: jnp.sum(lane.step(EcradIFSConfig(), src, full_call=False,  # noqa: E731
                                        packed_carry=packed, **dict(kw2, T_skin=ts))[0])
    assert np.all(np.isfinite(np.asarray(jax.grad(held)(kw2["T_skin"]))))
