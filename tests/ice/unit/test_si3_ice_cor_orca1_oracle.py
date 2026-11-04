"""NEMO 5.0.1 SI3 ``ice_cor(kn=1)`` oracle test at ORCA1 settings.

Fixture: ``fixtures/orca1_t4/si3_ice_cor.{npz,json}`` -- 100 cells of the tier-4
cor-channel record (NEMO 5.0.1 ORCA1_T4DYN, kt 17281, jpl 1, nn_icesal 2,
ponds off, rn_amax 0.99999, rn_himin 0.05): thin-ice rescale, concentration cap,
zapped, salinity-clamp, untouched-ice and open-water cells.  Provenance in the JSON.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm import constants
from legoesm.ice.si3_ice_cor import (
    SI3_ICE_COR_DEFAULT_LIMIT_SHARPNESS_SCALE,
    SI3IceCorConfig,
    SI3IceCorFluxes,
    SI3IceCorState,
    apply_si3_jpl1_ice_cor,
)

_FIX = Path(__file__).parent / "fixtures" / "orca1_t4" / "si3_ice_cor.npz"
_REL_BAR = 1.0e-13  # vote V10 (codex/GLM/author); NEMO record is fp64
_CONS_REL = 1.0e-12
_SMOOTH_SCALE = 100.0
_LN2 = math.log(2.0)


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    jax.config.update("jax_enable_x64", True)
    old = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(old)


@pytest.fixture(scope="module")
def fx():
    z = np.load(_FIX)
    return {k: z[k] for k in z.files}


def _config(fx, scale=math.inf):
    return SI3IceCorConfig(
        rn_himin_m=float(fx["scalar_rn_himin"]),
        rn_simin_g_kg=float(fx["scalar_rn_simin"]),
        rn_sinew=float(fx["scalar_rn_sinew"]),
        nn_icesal=2,
        ponds=False,
        rho_ice_kg_m3=float(fx["scalar_rhoi"]),
        rho_snow_kg_m3=float(fx["scalar_rhos"]),
        rho_water_kg_m3=float(fx["scalar_rhow"]),
        rt0_k=float(fx["scalar_rt0"]),
        dt_s=1.0 / float(fx["scalar_r1_Dt_ice"]),
        limit_sharpness_scale=scale,
    )


def _state(fx, **over):
    j = lambda k: jnp.asarray(over.get(k, fx[k]))  # noqa: E731
    return SI3IceCorState(
        ice_area=j("a_i"), ice_volume=j("v_i"), snow_volume=j("v_s"), salt_content=j("sv_i"),
        age_content=j("oa_i"), pond_area=j("a_ip"), pond_volume=j("v_ip"), pond_lid_volume=j("v_il"),
        snow_enthalpy=j("e_s"), ice_enthalpy=j("e_i"), surface_temperature=j("t_su"),
        open_water_area=j("ato_i"), snow_thickness=j("h_s"), snow_temperature=j("t_s"),
        ice_temperature=j("t_i"),
    )


def _fluxes(fx):
    return SI3IceCorFluxes(*(jnp.asarray(fx[k]) for k in ("sfx_res", "wfx_res", "hfx_res")))


def _run(fx, scale=math.inf, state=None):
    return apply_si3_jpl1_ice_cor(
        _state(fx) if state is None else state, _fluxes(fx), jnp.asarray(fx["rn_amax_2d"]),
        jnp.asarray(fx["sss_m"]), jnp.asarray(fx["sst_m"]), _config(fx, scale))


def _pairs(out):
    s, f = out.state, out.fluxes
    return {
        "a_i_out": s.ice_area, "v_i_out": s.ice_volume, "v_s_out": s.snow_volume, "sv_i_out": s.salt_content,
        "oa_i_out": s.age_content, "a_ip_out": s.pond_area, "v_ip_out": s.pond_volume,
        "v_il_out": s.pond_lid_volume, "h_i_out": out.ice_thickness, "h_s_out": s.snow_thickness,
        "t_su_out": s.surface_temperature, "e_s_out": s.snow_enthalpy, "e_i_out": s.ice_enthalpy,
        "t_s_out": s.snow_temperature, "t_i_out": s.ice_temperature, "at_i_out": out.total_ice_area,
        "vt_i_out": out.total_ice_volume, "ato_i_out": s.open_water_area, "sfx_res_out": f.sfx_res,
        "wfx_res_out": f.wfx_res, "hfx_res_out": f.hfx_res,
    }


def test_fixture_exercises_every_branch(fx):
    g = fx["group"]
    a, ao = fx["a_i"], fx["a_i_out"]
    assert (ao[g == "zapped"] == 0).all() and (a[g == "zapped"] > 0).all()
    assert (ao[g == "thin_himin"] < a[g == "thin_himin"]).all()
    assert (ao[g == "over_amax"] < a[g == "over_amax"]).all() and (g == "over_amax").sum() == 25
    assert (fx["sv_i_out"][g == "salt_clip"] != fx["sv_i"][g == "salt_clip"]).all()
    assert (fx["sfx_res_out"][g == "zapped"] != fx["sfx_res"][g == "zapped"]).any()
    assert (fx["hfx_res_out"][g == "zapped"] != fx["hfx_res"][g == "zapped"]).any()
    assert (fx["wfx_res_out"][g == "zapped"] != fx["wfx_res"][g == "zapped"]).all()


@pytest.mark.parametrize("group", ["thin_himin", "over_amax", "zapped", "salt_clip", "untouched_ice", "open_water"])
def test_hard_limits_reproduce_nemo(fx, group):
    out = _pairs(_run(fx))
    m = fx["group"] == group
    for key, port in out.items():
        nemo = fx[key][m]
        port = np.asarray(port)[m]
        assert np.isfinite(port).all(), key
        err = float(np.max(np.abs(port - nemo))) / max(float(np.max(np.abs(nemo))), 1e-300)
        assert err <= _REL_BAR, f"{group} {key}: rel {err:.3e}"


_RAMP_MAX = 4.0 / 27.0  # positive_part_c1 shortfall <= 4/27 x width


def _bands(fx, scale):
    """Cells inside any one-sided ramp band (width = limit scale / sharpness scale)."""
    himin, simin, sinew = (float(fx[f"scalar_{k}"]) for k in ("rn_himin", "rn_simin", "rn_sinew"))
    a, v, amax = fx["a_i"], fx["v_i"], fx["rn_amax_2d"]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(a >= 1e-20, v / a, 0.0)
        spec = np.where(v > 0, fx["sv_i"] / v, 0.0)
    smax = sinew * fx["sss_m"]
    wc = np.maximum(1 - amax, 1e-10) / scale
    return ((h > himin * (1 - 1 / scale)) & (h < himin)) | ((a > amax) & (a < amax + wc)) | (
        (v > 0) & (spec < simin) & (spec > simin * (1 - 1 / scale))) | (
        (v > 0) & (spec > smax) & (spec < smax * (1 + 1 / scale)))


def test_smooth_default_equals_nemo_outside_ramp_bands(fx):
    """DEFAULT (one-sided C1 ramps, scale 100) vs the NEMO record: bitwise-to-1e-13 wherever no
    limit sits inside its ramp band (the ramp is exact on both sides of the band); inside a band
    the area / salt gap is bounded by 4/27 x width."""
    scale = SI3_ICE_COR_DEFAULT_LIMIT_SHARPNESS_SCALE
    smooth = _run(fx, scale)
    out = _pairs(smooth)
    zf = np.asarray(smooth.zap_fraction)
    inside = _bands(fx, scale) | ((zf > 0) & (zf < 1))
    for key, port in out.items():
        port = np.asarray(port)[~inside]
        nemo = fx[key][~inside]
        err = float(np.max(np.abs(port - nemo))) / max(float(np.max(np.abs(nemo))), 1e-300)
        assert err <= _REL_BAR, f"{key}: {err:.3e}"
    hard = _run(fx)
    gap_a = np.abs(np.asarray(smooth.state.ice_area) - np.asarray(hard.state.ice_area))[inside]
    himin = float(fx["scalar_rn_himin"])
    bound = _RAMP_MAX * (fx["a_i"] / scale + np.maximum(1 - fx["rn_amax_2d"], 1e-10) / scale)[inside]
    assert (gap_a <= bound * (1 + 1e-9)).all()
    del himin


@pytest.mark.parametrize("scale", [math.inf, _SMOOTH_SCALE])
def test_conservation_with_booked_fluxes(fx, scale):
    out = _run(fx, scale)
    s, f = out.state, out.fluxes
    dt = 1.0 / float(fx["scalar_r1_Dt_ice"])
    rhoi, rhos, rhow = (float(fx[f"scalar_{k}"]) for k in ("rhoi", "rhos", "rhow"))
    water_in = fx["v_i"] * rhoi + fx["v_s"] * rhos + (fx["v_ip"] + fx["v_il"]) * rhow
    water_out = np.asarray(s.ice_volume) * rhoi + np.asarray(s.snow_volume) * rhos + (
        np.asarray(s.pond_volume) + np.asarray(s.pond_lid_volume)) * rhow
    dw = (np.asarray(f.wfx_res) - fx["wfx_res"]) * dt
    np.testing.assert_allclose(water_in - water_out, dw, rtol=_CONS_REL, atol=1e-12 * water_in.max())
    ds = (np.asarray(f.sfx_res) - fx["sfx_res"]) * dt
    np.testing.assert_allclose((fx["sv_i"] - np.asarray(s.salt_content)) * rhoi, ds,
                               rtol=_CONS_REL, atol=1e-12 * (fx["sv_i"] * rhoi).max())
    heat_in = fx["e_i"].sum(-1) + fx["e_s"].sum(-1)
    heat_out = np.asarray(s.ice_enthalpy).sum(-1) + np.asarray(s.snow_enthalpy).sum(-1)
    dh = (np.asarray(f.hfx_res) - fx["hfx_res"]) * dt
    np.testing.assert_allclose(heat_out - heat_in, dh, rtol=_CONS_REL, atol=1e-12 * np.abs(heat_in).max())


def _sub(fx, cells):
    n = fx["group"].size
    return {k: v[cells] if np.ndim(v) and v.shape[0] == n else v for k, v in fx.items()}


def _smooth_fn(sub):
    def f(a, v, sv):
        out = _run(sub, _SMOOTH_SCALE, _state(sub, a_i=a, v_i=v, sv_i=sv))
        return out.state.ice_area + out.state.salt_content + out.fluxes.sfx_res * 1e-3
    return f


def test_gradients_finite_through_smooth_limits(fx):
    # thin-ice (width 5e-4 m) and salinity (width 1e-3 g/kg) limits: check_grads order 2;
    # cells with a_i > 0.05 and away from the cap so the FD step resolves each transition.
    g, a = fx["group"], fx["a_i"]
    ok = (a > 0.05) & (a < 0.9)
    cells = np.concatenate([np.flatnonzero((g == k) & ok)[:2] for k in ("thin_himin", "salt_clip", "untouched_ice")])
    assert cells.size >= 4
    sub = _sub(fx, cells)
    f = _smooth_fn(sub)
    args = tuple(jnp.asarray(sub[k]) for k in ("a_i", "v_i", "sv_i"))
    check_grads(f, args, order=2, modes=("fwd", "rev"), eps=1e-8, atol=1e-6, rtol=1e-3)
    jac = np.asarray(jax.jacfwd(lambda x: f(x, args[1], args[2]))(args[0]))
    assert np.isfinite(jac).all()


def test_gradient_fd_ratio_at_concentration_cap(fx):
    # cap width 0.01*(1-rn_amax) = 1e-7: concentrations straddling the cap (thickness ~1 m,
    # far from rn_himin), central FD with step 1e-11 against AD, ratio ~ 1.
    cells = np.flatnonzero(fx["group"] == "over_amax")[:5]
    sub = _sub(fx, cells)
    amax = sub["rn_amax_2d"]
    width = (1.0 - amax) / _SMOOTH_SCALE
    a0 = jnp.asarray(amax + width * np.array([-3.0, -1.0, 0.25, 0.5, 0.75])[: cells.size])
    v = jnp.ones_like(a0)

    def g(a):
        return _run(sub, _SMOOTH_SCALE, _state(sub, a_i=a, v_i=v)).state.ice_area

    ad = np.diag(np.asarray(jax.jacrev(g)(a0)))
    eps = 1e-11
    fd = (np.asarray(g(a0 + eps)) - np.asarray(g(a0 - eps))) / (2 * eps)
    # d/da [a - ramp(a - amax)] = 1 - ramp' in [-1/3, 1] (positive_part_c1 slope peaks at 4/3)
    assert np.isfinite(ad).all() and (ad >= -1 / 3 - 1e-9).all() and (ad <= 1 + 1e-9).all() and (ad < 1 - 1e-3).any()
    np.testing.assert_allclose(ad, fd, rtol=1e-3)


_RHOI, _RHOS, _RHOW, _RT0 = constants.rho_ice, constants.rho_snow, constants.rho_water, constants.T_freeze
_DT = 3600.0
_SSS, _SST = 34.0, -1.5


def _synthetic_config(**kw):
    base = dict(rn_himin_m=0.05, rn_simin_g_kg=0.1, rn_sinew=0.75, nn_icesal=4, ponds=True,
                rho_ice_kg_m3=_RHOI, rho_snow_kg_m3=_RHOS, rho_water_kg_m3=_RHOW, rt0_k=_RT0, dt_s=_DT,
                limit_sharpness_scale=math.inf)
    base.update(kw)
    return SI3IceCorConfig(**base)


def _synthetic_state(a, v, salt, e_i, e_s, **kw):
    a = jnp.asarray(a)
    z = jnp.zeros_like(a)
    fields = dict(ice_area=a, ice_volume=jnp.asarray(v), snow_volume=z + 1e-12, salt_content=jnp.asarray(salt),
                  age_content=z + 1.0, pond_area=z + 0.1, pond_volume=z + 1e-13, pond_lid_volume=z + 2e-13,
                  snow_enthalpy=jnp.asarray(e_s), ice_enthalpy=jnp.asarray(e_i), surface_temperature=z + 260.0,
                  open_water_area=1.0 - a, ice_temperature=jnp.full(e_i.shape, 265.0),
                  ice_layer_salinity=jnp.full(e_i.shape, 5.0))
    fields.update(kw)
    return SI3IceCorState(**fields)


def test_layered_salt_and_ponds_known_answer():
    """nn_icesal=4 (2 layers) + ponds on: thin-ice/pond rescale, layer clamp, cap, layered zap."""
    a = np.array([0.5, 1.0, 1e-11])
    v = np.array([0.01, 2.0, 1e-12])
    salt = np.array([[1e-4, 0.2], [0.5, 0.5], [1e-13, 2e-13]])
    e_i = np.array([[-1e5, -2e5], [-1e8, -1e8], [-1e3, -2e3]])
    e_s = np.array([[-5e4], [-1e6], [-7e2]])
    st = _synthetic_state(a, v, salt, e_i, e_s)
    z = jnp.zeros(3)
    out = apply_si3_jpl1_ice_cor(st, SI3IceCorFluxes(z, z, z), jnp.asarray([0.99999, 0.99, 0.99999]),
                                 jnp.full(3, _SSS), jnp.full(3, _SST), _synthetic_config())
    s, f = out.state, out.fluxes
    r1 = 1.0 / _DT
    # cell 0: h = 0.02 < himin -> a = 0.5*0.02/0.05, pond area scaled by 0.4; layer clamp [5e-4, 0.1275]
    # cell 1: cap 1.0 -> 0.99, salinity in bounds; cell 2: zapped
    exp_a = np.array([(0.5 * 0.02) / 0.05, 0.99, 0.0])
    exp_ap = np.array([(0.1 * 0.02) / 0.05, 0.1, 0.0])
    lo0, hi0 = (0.1 * 0.01) * 0.5, ((0.75 * _SSS) * 0.01) * 0.5
    exp_salt = np.array([[lo0, hi0], [0.5, 0.5], [0.0, 0.0]])
    sfx0 = (0.0 - ((lo0 - 1e-4) * _RHOI) * r1) - ((hi0 - 0.2) * _RHOI) * r1
    sfx2 = (0.0 + (1e-13 * _RHOI) * r1) + (2e-13 * _RHOI) * r1
    hfx2 = (((0.0 - (-1e3) * r1) - (-2e3) * r1) - (-7e2) * r1)
    wfx2 = ((0.0 + (1e-12 * _RHOI) * r1) + (1e-12 * _RHOS) * r1) + ((1e-13 + 2e-13) * _RHOW) * r1
    np.testing.assert_allclose(np.asarray(s.ice_area), exp_a, rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.asarray(s.pond_area), exp_ap, rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.asarray(s.salt_content), exp_salt, rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.asarray(f.sfx_res), [sfx0, 0.0, sfx2], rtol=1e-13, atol=0)
    np.testing.assert_allclose(np.asarray(f.hfx_res), [0.0, 0.0, hfx2], rtol=1e-14, atol=0)
    np.testing.assert_allclose(np.asarray(f.wfx_res), [0.0, 0.0, wfx2], rtol=1e-14, atol=0)
    assert np.asarray(out.zap_fraction).tolist() == [0.0, 0.0, 1.0]
    np.testing.assert_array_equal(np.asarray(s.ice_layer_salinity)[2], [0.1, 0.1])
    np.testing.assert_array_equal(np.asarray(s.ice_temperature)[2], [_RT0, _RT0])
    np.testing.assert_allclose(float(s.surface_temperature[2]), _SST + _RT0, rtol=1e-15)
    assert float(s.open_water_area[2]) == 1.0 and float(s.pond_area[2]) == 0.0
    np.testing.assert_array_equal(np.asarray(s.ice_layer_salinity)[:2], 5.0)
    # ponds off: pond area untouched by the thin-ice rescale
    off = apply_si3_jpl1_ice_cor(st, SI3IceCorFluxes(z, z, z), 0.99999, jnp.full(3, _SSS), jnp.full(3, _SST),
                                 _synthetic_config(ponds=False))
    assert float(off.state.pond_area[0]) == 0.1


def test_one_sided_ramps_known_answer_in_each_band():
    """Mid-band (x = threshold -/+ w/2) the ramp keeps 0.375 w where NEMO keeps 0.5 w; at the
    threshold itself and outside the band the result is NEMO's exactly."""
    himin, amax, simin, sinew = 0.05, 0.99999, 0.1, 0.75
    smax = sinew * _SSS
    w_h, w_c, w_lo, w_hi = himin / _SMOOTH_SCALE, (1 - amax) / _SMOOTH_SCALE, simin / _SMOOTH_SCALE, smax / _SMOOTH_SCALE
    a = np.array([0.5, amax + w_c / 2, 0.5, 0.5, 0.5, 0.5, amax])
    h = np.array([himin - w_h / 2, 1.0, 1.0, 1.0, himin, himin - 2 * w_h, 1.0])
    v = a * h
    spec = np.array([5.0, 5.0, simin - w_lo / 2, smax + w_hi / 2, 5.0, 5.0, 5.0])
    st = _synthetic_state(a, v, v * spec, np.full((7, 1), -1e7), np.full((7, 1), -1e5), pond_area=jnp.zeros(7))
    z = jnp.zeros(7)
    cfg = _synthetic_config(nn_icesal=2, ponds=False, limit_sharpness_scale=_SMOOTH_SCALE)
    out = apply_si3_jpl1_ice_cor(st, SI3IceCorFluxes(z, z, z), amax, jnp.full(7, _SSS), jnp.full(7, _SST), cfg)
    hard = apply_si3_jpl1_ice_cor(st, SI3IceCorFluxes(z, z, z), amax, jnp.full(7, _SSS), jnp.full(7, _SST),
                                  cfg._replace(limit_sharpness_scale=math.inf))
    ga, ha = np.asarray(out.state.ice_area), np.asarray(hard.state.ice_area)
    gs, hs = np.asarray(out.state.salt_content), np.asarray(hard.state.salt_content)
    np.testing.assert_allclose(ga[0], a[0] * (himin - 0.375 * w_h) / himin, rtol=1e-12)
    np.testing.assert_allclose(ha[0], a[0] * h[0] / himin, rtol=1e-12)
    np.testing.assert_allclose(ga[1], a[1] - 0.375 * w_c, rtol=1e-14)
    np.testing.assert_allclose(gs[2] / v[2], spec[2] + 0.375 * w_lo, rtol=1e-12)
    np.testing.assert_allclose(gs[3] / v[3], spec[3] - 0.375 * w_hi, rtol=1e-12)
    # at the threshold (h = himin, a = amax) and below the band (h = himin - 2w): NEMO exactly
    np.testing.assert_allclose(ga[4:], ha[4:], rtol=1e-15, atol=0)
    # amax = 1 (NEMO caps rn_amax <= 1 - epsi10) stays finite with finite gradients
    g = jax.grad(lambda x: apply_si3_jpl1_ice_cor(st._replace(ice_area=x), SI3IceCorFluxes(z, z, z), 1.0,
                                                  jnp.full(7, _SSS), jnp.full(7, _SST), cfg).state.ice_area.sum())
    assert np.isfinite(np.asarray(g(jnp.asarray(a)))).all()
    # check_grads (order 2) inside the thin-ice and salinity bands, where the ramp is a cubic
    idx = np.array([0, 2, 3])
    sub_st = jax.tree_util.tree_map(lambda x: x[idx] if x is not None and np.ndim(x) else x, st)
    z3 = jnp.zeros(3)

    def f(av, vv, sv):
        o = apply_si3_jpl1_ice_cor(sub_st._replace(ice_area=av, ice_volume=vv, salt_content=sv),
                                   SI3IceCorFluxes(z3, z3, z3), amax, jnp.full(3, _SSS), jnp.full(3, _SST), cfg)
        return o.state.ice_area + o.state.salt_content

    check_grads(f, (jnp.asarray(a[idx]), jnp.asarray(v[idx]), jnp.asarray((v * spec)[idx])), order=2,
                modes=("fwd", "rev"), eps=1e-9, atol=1e-6, rtol=1e-3)


def test_dispatch_and_layout_guards(fx):
    with pytest.raises(ValueError, match="nn_icesal"):
        apply_si3_jpl1_ice_cor(_state(fx), _fluxes(fx), 0.99, fx["sss_m"], fx["sst_m"],
                               _config(fx)._replace(nn_icesal=1))
    with pytest.raises(ValueError, match="layer axis"):
        apply_si3_jpl1_ice_cor(_state(fx), _fluxes(fx), 0.99, fx["sss_m"], fx["sst_m"],
                               _config(fx)._replace(nn_icesal=4))
    with pytest.raises(ValueError, match="sharpness"):
        apply_si3_jpl1_ice_cor(_state(fx), _fluxes(fx), 0.99, fx["sss_m"], fx["sst_m"],
                               _config(fx)._replace(limit_sharpness_scale=0.0))


def test_fixture_provenance(fx):
    prov = json.loads(_FIX.with_suffix(".json").read_text())
    assert prov["nemo_build_job"] == 11022552 and "ORCA1_T4DYN" in prov["cfg"]
    assert prov["group_sizes"]["over_amax"] == 25
