"""IFS 49r1 CLOUDSC standard-contract adapter and the NCLDTOP rule.

* the adapter is CLOUDSC with the integration inputs (votes VC4a/VC5a/VC21a:
  PTENDENCY_CML = PVFL = PVFI = PEIS = 0) and the slot map L, I, R, S ->
  q_c, q_i, q_r, q_s (VC18a), surface precipitation = FPLSL + FPLSN at the
  surface half level, frozen = FPLSN there;
* through the adapter's own outputs, column water and liquid-water enthalpy
  close against the surface precipitation and the detrainment CLOUDSC takes
  in (the kernel's budget form, tests/atmosphere/microphysics/unit/
  test_ifs_cloudsc.py);
* NCLDTOP = the sucldp.F90:631-634 rule: 15 on the oracle's L137, 2 on CAM L32.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import legoesm.atmosphere.physics.microphysics.ifs_cloudsc as M
from legoesm import constants
from legoesm.atmosphere.physics._ifs_thermo import lego_thermo
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState

ROOT = Path(__file__).resolve().parents[4]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load("t_cloudsc_oracle_a", ROOT / "tests/atmosphere/test_ifs_cloudsc_oracle.py")


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(T._FIX))


def _adapter_inputs(x):
    """The oracle column as the host would hand it: species slots, the
    convective hand-off, land fraction; everything the adapter zeroes is
    read off ``x`` only to build the reference call."""
    c = x.clv
    hyd = HydrometeorState(q_c=c[..., 0], q_i=c[..., 1], q_r=c[..., 2], q_s=c[..., 3],
                           q_g=jnp.zeros_like(x.t), N_c=jnp.zeros_like(x.t),
                           N_r=jnp.zeros_like(x.t), N_i=jnp.zeros_like(x.t))
    conv = M.CloudscConvection(plu=x.plu, plude=x.plude, pludeli=x.pludeli,
                               psnde=x.psnde, pmfu=x.pmfu, pmfd=x.pmfd,
                               ldcum=x.ldcum, ktype=x.ktype)
    return hyd, conv


def _run(cfg, x, dt):
    hyd, conv = _adapter_inputs(x)
    return M.ifs_cloudsc_microphysics(
        x.t, x.q, hyd, x.pap, x.paph, None, None, dt, cfg,
        cloud_fraction=x.a, land_frac=x.plsm, conv=conv)


def _reference(cfg, x, dt):
    z = jnp.zeros_like(x.t)
    xr = x._replace(tend_t=z, tend_q=z, tend_a=z, tend_cld=jnp.zeros_like(x.tend_cld),
                    pvfl=z, pvfi=z, peis=jnp.zeros_like(x.peis))
    return xr, M.cloudsc(cfg, lego_thermo(), xr, dt, hard=False)


@pytest.mark.parametrize("arm", ("base", "detr"))
def test_adapter_is_cloudsc_with_the_integration_inputs_and_slot_map(fix, arm):
    cfg, _, x, dt = T.setup(fix, arm)
    out = _run(cfg, x, dt)
    _, ref = _reference(cfg, x, dt)
    eq = np.testing.assert_array_equal
    eq(out.dT_dt, ref.tend_t)
    eq(out.dq_v_dt, ref.tend_q)
    eq(out.dq_c_dt, ref.tend_cld[..., M.NCLDQL])
    eq(out.dq_i_dt, ref.tend_cld[..., M.NCLDQI])
    eq(out.dq_r_dt, ref.tend_cld[..., M.NCLDQR])
    eq(out.dq_s_dt, ref.tend_cld[..., M.NCLDQS])
    for f in ("dq_g_dt", "dN_c_dt", "dN_r_dt", "dN_i_dt"):
        assert np.all(np.asarray(getattr(out, f)) == 0.0), f
    eq(out.precipitation, ref.fplsl[:, -1] + ref.fplsn[:, -1])
    eq(out.precip_frozen, ref.fplsn[:, -1])
    eq(out.cloud_fraction_new, x.a + dt * ref.tend_a)
    # the four species really differ on these columns, so a swapped slot fails
    tc = np.asarray(ref.tend_cld[..., :4])
    for i in range(4):
        for j in range(i + 1, 4):
            assert not np.array_equal(tc[..., i], tc[..., j]), (i, j)


@pytest.mark.parametrize("arm", ("base", "detr"))
def test_adapter_conserves_water_and_enthalpy_with_surface_precip(fix, arm):
    cfg, _, x, dt = T.setup(fix, arm)
    th = lego_thermo()
    out = _run(cfg, x, dt)
    xr, ref = _reference(cfg, x, dt)
    g = th.RG
    dp = np.asarray(x.paph[:, 1:] - x.paph[:, :-1])
    k0 = cfg.ncldtop - 1
    # detrainment CLOUDSC took in (kernel budget form): PLUDE_out / (PLUDE_in dt g/dp)
    pin = np.asarray(xr.plude) * dt * g / dp
    w = np.where(pin > 0, np.asarray(ref.plude) / np.where(pin > 0, pin, 1.0), 0.0)
    w[:, :k0] = 0.0
    w[:, -1] = 0.0
    dl = np.sum(w * np.asarray(xr.pludeli[..., 0]), 1)
    di = np.sum(w * np.asarray(xr.pludeli[..., 1]), 1)
    assert np.all(np.asarray(xr.psnde) == 0)
    p_tot = np.asarray(out.precipitation)
    p_ice = np.asarray(out.precip_frozen)
    p_liq = p_tot - p_ice
    lv, ls = th.RALVDCP, th.RALSDCP
    dql = np.asarray(out.dq_c_dt) + np.asarray(out.dq_r_dt)
    dqi = np.asarray(out.dq_i_dt) + np.asarray(out.dq_s_dt)
    water = (np.asarray(out.dq_v_dt) + dql + dqi) * dp / g
    h = (np.asarray(out.dT_dt) - lv * dql - ls * dqi) * dp / g
    res_w = water.sum(1) + p_tot - dl - di - water[:, :k0].sum(1)
    res_h = (h.sum(1) - lv * p_liq - ls * p_ice + lv * dl + ls * di
             - h[:, :k0].sum(1))
    sw = np.sum(np.abs(water), 1) + np.abs(p_tot) + dl + di + 1e-30
    sh = np.sum(np.abs(h), 1) + ls * (np.abs(p_tot) + dl + di) + 1e-30
    assert np.max(np.abs(res_w) / sw) < 1e-12
    assert np.max(np.abs(res_h) / sh) < 1e-12
    assert np.max(p_tot) > 0.0          # the surface term is exercised
    if arm == "detr":
        assert np.max(dl + di) > 0.0    # and so is the detrainment


def test_adapter_refuses_a_missing_convective_handoff(fix):
    cfg, _, x, dt = T.setup(fix, "base")
    hyd, _ = _adapter_inputs(x)
    with pytest.raises(TypeError, match="CloudscConvection"):
        M.ifs_cloudsc_microphysics(x.t, x.q, hyd, x.pap, x.paph, None, None, dt, cfg,
                                   cloud_fraction=x.a, land_frac=x.plsm, conv=None)


def _l137_reference_full_levels(fix):
    """L137 half levels at p_s = 101325 Pa from two oracle columns (A + B p_s
    solved exactly per half level), full level = half-level mean."""
    ph = np.asarray(fix["base_in_PAPH"], np.float64)
    i, j = 0, int(np.argmax(np.abs(ph[:, -1] - ph[0, -1])))
    ps_i, ps_j = ph[i, -1], ph[j, -1]
    assert abs(ps_i - ps_j) > 10.0
    b = (ph[i] - ph[j]) / (ps_i - ps_j)
    a = ph[i] - b * ps_i
    half = a + b * constants.p_atm_std
    return 0.5 * (half[:-1] + half[1:]), half


def test_ncldtop_rule_gives_15_on_the_oracle_l137(fix):
    full, half = _l137_reference_full_levels(fix)
    assert full.size == 137
    assert M.ncldtop_from_reference_pressure(full) == 15
    # the log-pressure full level gives the same level (rule not sensitive here)
    logfull = np.sqrt(np.maximum(half[:-1], 1e-30) * half[1:])
    assert M.ncldtop_from_reference_pressure(logfull) == 15
    # and the oracle ran CLOUDSC with NCLDTOP = 15
    assert int(fix["cfg_ncldtop"]) == 15


def test_ncldtop_rule_on_cam_l32_and_edge_cases():
    from legoesm.grids.vertical import make_cam6_l32_levels
    sig = make_cam6_l32_levels()
    full = np.asarray(sig.pressure_at_full(jnp.asarray([constants.p_atm_std])))
    assert full.size == 32 and full.reshape(-1)[0] > M._RCLDTOPP
    assert M.ncldtop_from_reference_pressure(full) == 2   # sucldp floor: level 1 never
    # rule mechanics: topmost level >= 2 above the threshold; none -> 2
    assert M.ncldtop_from_reference_pressure([1.0, 50.0, 99.0, 101.0, 500.0]) == 4
    assert M.ncldtop_from_reference_pressure([1.0, 50.0, 99.0]) == 2
