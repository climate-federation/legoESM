"""Oracle match: OpenIFS 49r1 WMS GWD, Cariolle ozone chemistry and METHOX.

Fixture ``oracle_fixtures/ifs_wms_ozone.npz``: the tier-2 audit's 24 columns
(16 L32, 5 L137, 3 stress) with latitude, cell size, standard pressures and
synthetic Cariolle coefficients/ozone added (votes V3-V6,
oracle_audit/ifs_port/wms_ozone/VOTES.md), and the outputs of RUNNING OpenIFS
8f6f722 phys_ec/{sugwwms,gwdrag_wms,o3chem,sumethox,methox}.F90 (gfortran 8.5,
fp64) in two arms: ``L`` legoESM constants, ``N`` the 49r1 constants
(scripts/validate/oracle/ifs_wms_ozone/{build.sh,make_fixture.py}).
Time steps 450 s and 3600 s; RQLIM = SUMETHOX default and 5e-6.

Error metric (as test_bechtold_ifs_oracle.py): per field,
max|ours - oracle| / max|oracle| over all columns and levels; the ports run
their hard (width 0) forms.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.gravity_wave_drag.ifs_wms import (
    IFSWMSConfig, wms_call_interval, wms_drag, wms_heating, wms_launch_level,
)
from legoesm.atmosphere.physics.methane_oxidation import (
    MethoxConfig, methox_q_lim, methox_tendency,
)
from legoesm.atmosphere.physics.radiation.ozone_cariolle import (
    OzoneCariolleConfig, cariolle_tendency,
)

_FIX = Path(__file__).parent / "oracle_fixtures" / "ifs_wms_ozone.npz"
_GAP_FILE = Path(__file__).parent / "oracle_fixtures" / "ifs_wms_ozone_default_gap.npz"
_TOL = 1.0e-12   # user-approved bar: rel 1e-12 fp64 (or the oracle's 20-seed spread)
_GROUPS = ("L32", "L137", "STRESS")
_DTS = (450, 3600)
_HARD_WMS = IFSWMSConfig(n2_floor_rel_width=0.0, speed_width_m_s=0.0,
                         saturation_rel_width=0.0, deposit_rel_width=0.0)
_HARD_O3 = OzoneCariolleConfig(t_clamp_width_K=0.0, floor_rel_width=0.0,
                               seed_overhead_column=False)
# RULE 5 gap = smoothing only; the model-top seed (vote Y1a) is its own test.
_SMOOTH_O3 = OzoneCariolleConfig(seed_overhead_column=False)


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(_FIX))


def _consts(fix, arm):
    """(g, R_d, c_pd, R_earth) the oracle ran with, from its CONST_USED line."""
    log = str(fix[f"{arm}_const_line"])
    vals = [float(x) for x in re.search(r"CONST_USED RG RD RCPD RA RPI RDAYI(.*)", log).group(1).split()]
    if arm == "L":
        assert vals[:4] == [constants.g, constants.R_d, constants.c_pd, constants.R_earth]
    assert vals[4] == math.pi and vals[5] == 86400.0
    return vals[:4]


def _err(v, ref):
    v = np.asarray(v, np.float64)
    if not (np.all(np.isfinite(v)) and np.all(np.isfinite(ref))):
        return np.inf
    return float(np.max(np.abs(v - ref)) / max(np.max(np.abs(ref)), 1e-300))


def _wms(fix, arm, g, dt, cfg=_HARD_WMS):
    a = lambda k: jnp.asarray(fix[f"{g}_{k}"])
    gg, rd, cp, ra = _consts(fix, arm)
    lev = wms_launch_level(fix[f"{g}_stpre"])
    return wms_drag(a("T"), a("u"), a("v"), a("p_full"), a("p_half"), a("geo_full"), a("lat"),
                    jnp.asarray(fix[f"{arm}_{g}_pgaw"]), launch_level=lev,
                    dt_gw=wms_call_interval(float(dt)), cfg=cfg, g=gg, R_d=rd, c_pd=cp, radius=ra)


@pytest.mark.parametrize("arm", ["L", "N"])
@pytest.mark.parametrize("dt", _DTS)
@pytest.mark.parametrize("g", _GROUPS)
def test_wms_matches_openifs(fix, arm, g, dt):
    t = f"{arm}_{g}_wms_dt{dt}"
    assert wms_launch_level(fix[f"{g}_stpre"]) == int(fix[f"{t}_NLAUNCH"].item())
    assert wms_call_interval(float(dt)) == float(fix[f"{t}_GTPHYGWWMS"].item())
    out = _wms(fix, arm, g, dt)
    assert np.max(np.abs(fix[f"{t}_PTENU"])) > 0.0 and np.max(np.abs(fix[f"{t}_PFLUXV"])) > 0.0
    errs = {k: _err(o, fix[f"{t}_{k}"]) for k, o in zip(("PTENU", "PTENV", "PFLUXU", "PFLUXV"), out)}
    print(t, {k: f"{v:.2e}" for k, v in errs.items()})
    bad = {k: v for k, v in errs.items() if not v <= _TOL}
    assert not bad, f"{t}: fields above {_TOL}: {bad}"


@pytest.mark.parametrize("arm", ["L", "N"])
@pytest.mark.parametrize("dt", _DTS)
@pytest.mark.parametrize("g", _GROUPS)
def test_cariolle_matches_openifs(fix, arm, g, dt):
    a = lambda k: jnp.asarray(fix[f"{g}_{k}"])
    gg = _consts(fix, arm)[0]
    ten = cariolle_tendency(a("PO3"), a("T"), a("p_full"), a("p_half"), a("PKOZO"), float(dt),
                            cfg=_HARD_O3, g=gg)
    e = _err(ten, fix[f"{arm}_{g}_o3_dt{dt}_PTENO3"])
    print(arm, g, dt, f"PTENO3 {e:.2e}")
    assert e <= _TOL


@pytest.mark.parametrize("arm", ["L", "N"])
@pytest.mark.parametrize("i", [0, 1])
@pytest.mark.parametrize("g", _GROUPS)
def test_methox_matches_openifs(fix, arm, g, i):
    t = f"{arm}_{g}_methox_rq{i}"
    rq = float(fix[f"{t}_RQLIM"].item())
    if i == 0:
        assert rq == float(fix[f"{t}_RQLIM_SUMETHOX"].item()) == MethoxConfig().q_lim_kg_kg
    ten = methox_tendency(jnp.asarray(fix[f"{g}_q_methox"]), jnp.asarray(fix[f"{g}_p_full"]),
                          MethoxConfig(q_lim_kg_kg=rq, qlim_gate_rel_width=0.0))
    e = _err(ten, fix[f"{t}_PTENQ"])
    print(t, f"PTENQ {e:.2e}")
    assert e <= _TOL


def test_harness_io_reproduces_independent_methox(fix):
    """Instrument self-check: a plain-numpy transcription of methox.F90 reproduces
    what the harness read back from the oracle (validates the text I/O)."""
    a1 = 19.0 * math.log(10.0) / math.log(20.0) ** 4
    a2 = math.log(1.0 / 3.0 + 0.01)
    a3 = 0.5 * (math.log(100.0) + a2)
    lpph = math.log(0.1 / 20.0)
    for g in _GROUPS:
        q, p = fix[f"{g}_q_methox"], fix[f"{g}_p_full"]
        ref = np.zeros_like(q)
        rq = 4.81e-6
        for idx in np.ndindex(q.shape):
            pp, qq, t = p[idx], q[idx], 0.0
            if pp < 1.0e4 and qq < rq:
                d = 100.0 if pp <= 50.0 else 100.0 * (1 + a1 * math.log(pp / 50.0) ** 4 / math.log(1e4 / pp))
                t += (rq - qq) / (86400.0 * d)
            if pp < 20.0:
                d = 3.0 if pp <= 0.1 else 1.0 / (math.exp(a2 - a3 * (1 + math.cos(math.pi * math.log(pp / 20.0) / lpph))) - 0.01)
                t -= qq / (86400.0 * d)
            ref[idx] = t
        assert _err(ref, fix[f"L_{g}_methox_rq0_PTENQ"]) <= 1e-12


def test_fixture_exercises_branches(fix):
    """The O3 temperature clamp binds on both sides, the 1e-9 floor binds,
    METHOX sees q on both sides of RQLIM above 100 hPa and photolysis levels,
    WMS sees resolution factors < 1 and critical-level filtering."""
    hit_lo = hit_hi = floor = False
    for g in _GROUPS:
        dT = fix[f"{g}_T"] - fix[f"{g}_PKOZO"][..., 3]
        hit_lo |= bool(np.any(dT < -15.0))
        hit_hi |= bool(np.any(dT > 15.0))
        r_new = fix[f"{g}_PO3"] + 450.0 * fix[f"L_{g}_o3_dt450_PTENO3"]
        floor |= bool(np.any(np.isclose(r_new, 1e-9, rtol=1e-9, atol=0.0)
                             & (fix[f"{g}_PO3"] != 1e-9)))
    assert hit_lo and hit_hi and floor
    q, p = fix["STRESS_q_methox"], fix["STRESS_p_full"]
    strat = p < 1e4
    assert np.any(q[strat] < 4.81e-6) and np.any(q[strat] > 4.81e-6)
    assert any(np.any(fix[f"{g}_p_full"] < 20.0) for g in _GROUPS)
    dx = np.sqrt(4.0 * math.pi * constants.R_earth ** 2 * fix["L_L32_pgaw"])
    assert np.any(dx < 29.0e3) and np.any(dx < 3.5e3)


@pytest.mark.parametrize("g", _GROUPS)
def test_wms_conserves_column_momentum(fix, g):
    """sum(dudt dp/g) = 0: the drag redistributes the launched flux (both forms)."""
    for cfg in (_HARD_WMS, IFSWMSConfig()):
        dudt, dvdt, _, _ = _wms(fix, "L", g, 450, cfg)
        dp = np.diff(fix[f"{g}_p_half"], axis=1)
        for d in (dudt, dvdt):
            col = np.sum(np.asarray(d) * dp, axis=1) / constants.g
            scale = np.max(np.abs(np.asarray(d) * dp), axis=1) / constants.g
            assert np.all(np.abs(col) <= 1e-12 * np.maximum(scale, 1e-300))


def test_smooth_forms_differentiable_and_close(fix):
    """RULE 5: default widths give finite, nonzero gradients and stay close to
    the hard forms; finite-difference check on one wind and one temperature."""
    g = "L137"
    a = lambda k: jnp.asarray(fix[f"{g}_{k}"])
    lev = wms_launch_level(fix[f"{g}_stpre"])
    pg = jnp.asarray(fix[f"L_{g}_pgaw"])

    def loss_wms(u):
        d, _, _, _ = wms_drag(a("T"), u, a("v"), a("p_full"), a("p_half"), a("geo_full"),
                              a("lat"), pg, launch_level=lev, dt_gw=3600.0)
        return jnp.sum(d ** 2) * 1e10

    u0 = a("u")
    gr = jax.grad(loss_wms)(u0)
    assert np.all(np.isfinite(gr)) and np.any(np.asarray(gr) != 0.0)
    i, k = 2, lev - 10
    e = jnp.zeros_like(u0).at[i, k].set(1e-4)
    fd = (loss_wms(u0 + e) - loss_wms(u0 - e)) / 2e-4
    assert abs(float(fd) - float(gr[i, k])) <= 1e-4 * max(abs(float(fd)), 1e-12)

    def loss_o3(T):
        return jnp.sum(cariolle_tendency(a("PO3"), T, a("p_full"), a("p_half"), a("PKOZO"),
                                         450.0) ** 2) * 1e20

    T0 = a("T")
    go = jax.grad(loss_o3)(T0)
    assert np.all(np.isfinite(go)) and np.any(np.asarray(go) != 0.0)

    def loss_mx(q):
        return jnp.sum(methox_tendency(q, a("p_full")) ** 2) * 1e20

    gm = jax.grad(loss_mx)(a("q_methox"))
    assert np.all(np.isfinite(gm))

    for part in (dict(speed_width_m_s=0.0), dict(speed_width_m_s=0),
                 dict(saturation_rel_width=0.0, deposit_rel_width=0.0)):
        assert np.all(np.isfinite(np.asarray(_wms(fix, "L", g, 450, IFSWMSConfig(**part))[0])))
    soft = _wms(fix, "L", g, 450, IFSWMSConfig())[0]
    hard = _wms(fix, "L", g, 450)[0]
    assert _err(soft, np.asarray(hard)) < 0.2


def test_methox_q_lim_formula():
    """updrgas.F90:147-153: at the 2010 reference CH4 the limit is RQLIM_REFYR;
    each extra CH4 molecule adds two water molecules."""
    ref_vmr = 1.0e-6 * constants.M_dry / constants.M_ch4
    assert abs(methox_q_lim(ref_vmr) - 4.60e-6) <= 1e-18
    d = 1.0e-7
    dq = methox_q_lim(ref_vmr + d) - methox_q_lim(ref_vmr)
    assert abs(dq - 2.0 * d * constants.M_h2o / constants.M_dry) <= 1e-18


def test_wms_heating_closes_kinetic_energy():
    """Vote V11b: c_pd dT dt = -(KE after - KE before) for the applied step; at
    dt -> 0 it reduces to the 49r1 form -(u du + v dv)/c_pd (callpar.F90:1357)."""
    u, v = np.array([[10.0, -5.0]]), np.array([[2.0, 3.0]])
    du, dv = np.array([[1e-3, -2e-3]]), np.array([[3e-4, 4e-4]])
    dt = 450.0
    heat = np.asarray(wms_heating(u, v, du, dv, dt))
    dke = 0.5 * ((u + dt * du) ** 2 + (v + dt * dv) ** 2 - u ** 2 - v ** 2)
    np.testing.assert_allclose(constants.c_pd * heat * dt, -dke, rtol=1e-12)
    ref49 = -(u * du + v * dv) * (1.0 / constants.c_pd)
    np.testing.assert_array_equal(np.asarray(wms_heating(u, v, du, dv, 0.0)), ref49)


def _default_gaps(fix):
    """Signed per-point gap (default RULE 5 widths minus the oracle), L arm, dt 450 s."""
    out = {}
    for g in _GROUPS:
        a = lambda k: jnp.asarray(fix[f"{g}_{k}"])
        dudt, dvdt, _, _ = _wms(fix, "L", g, 450, IFSWMSConfig())
        out[f"{g}_wms_dudt"] = np.asarray(dudt) - fix[f"L_{g}_wms_dt450_PTENU"]
        out[f"{g}_wms_dvdt"] = np.asarray(dvdt) - fix[f"L_{g}_wms_dt450_PTENV"]
        o3 = cariolle_tendency(a("PO3"), a("T"), a("p_full"), a("p_half"), a("PKOZO"), 450.0,
                               cfg=_SMOOTH_O3)
        out[f"{g}_o3"] = np.asarray(o3) - fix[f"L_{g}_o3_dt450_PTENO3"]
        mx = methox_tendency(a("q_methox"), a("p_full"))
        out[f"{g}_methox"] = np.asarray(mx) - fix[f"L_{g}_methox_rq0_PTENQ"]
    return out


def test_default_smoothing_gap_per_point(fix):
    """RULE 5: the default-width gap to the oracle equals the stored measurement
    (signed, per point; regenerate with ``python -m tests.atmosphere.test_ifs_wms_ozone_oracle``)."""
    stored = dict(np.load(_GAP_FILE))
    gaps = _default_gaps(fix)
    assert set(gaps) == set(stored)
    for k, v in gaps.items():
        scale = max(np.max(np.abs(stored[k])), 1e-300)
        np.testing.assert_allclose(v, stored[k], rtol=1e-6, atol=1e-6 * scale, err_msg=k)
        print(k, f"max|gap| {np.max(np.abs(v)):.3e}")


def test_wms_fp32_gradient_finite(fix):
    """Smooth path in fp32 (default model precision): finite forward and gradient,
    including the zero-launch-flux (1 km) columns."""
    g = "L32"
    a = lambda k: jnp.asarray(fix[f"{g}_{k}"], jnp.float32)
    lev = wms_launch_level(fix[f"{g}_stpre"])

    def loss(u):
        d, _, _, _ = wms_drag(a("T"), u, a("v"), a("p_full"), a("p_half"), a("geo_full"), a("lat"),
                              jnp.asarray(fix[f"L_{g}_pgaw"], jnp.float32), launch_level=lev,
                              dt_gw=3600.0)
        return jnp.sum(d * 1e4)

    val, gr = jax.value_and_grad(loss)(a("u"))
    assert np.isfinite(float(val)) and np.all(np.isfinite(np.asarray(gr)))


def test_methox_jits_with_traced_q_lim(fix):
    """q_lim from time-varying CH4 is traced: the scheme must jit."""
    q, p = jnp.asarray(fix["L32_q_methox"]), jnp.asarray(fix["L32_p_full"])
    f = jax.jit(lambda ch4: methox_tendency(q, p, MethoxConfig(q_lim_kg_kg=methox_q_lim(ch4))))
    assert np.all(np.isfinite(np.asarray(f(jnp.asarray(1.9e-6)))))


if __name__ == "__main__":
    np.savez_compressed(_GAP_FILE, **_default_gaps(dict(np.load(_FIX))))
    print("wrote", _GAP_FILE)


def test_wired_wms_kernel_matches_oracle_and_retains(fix):
    """Wired path (vote G2a): the GWD dispatch's ifs_wms kernel reproduces the
    oracle (hard widths) on its first call, re-applies the retained tendency
    until GTPHYGWWMS has elapsed, then recomputes; heating every step; the
    composite with mcfarlane threads the same carry."""
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        get_gwd_fn, gwd_carry_seed, gwd_carry_shape,
    )
    g = "L137"
    a = lambda k: jnp.asarray(fix[f"{g}_{k}"])
    lev = wms_launch_level(fix[f"{g}_stpre"])
    gcfg = GravityWaveDragConfig(scheme="ifs_wms",
                                 ifs_wms=_HARD_WMS._replace(launch_level=lev))
    name, fn, sub = get_gwd_fn(gcfg)
    assert name == "ifs_wms"
    ncol, nlev = fix[f"{g}_T"].shape
    carry = gwd_carry_seed(gcfg, ncol, nlev)
    assert carry.shape == gwd_carry_shape(gcfg, ncol, nlev)
    area = jnp.asarray(fix[f"L_{g}_pgaw"]) * 4.0 * math.pi * constants.R_earth ** 2
    z = a("geo_full") / constants.g
    dt = 450.0
    args = (a("u"), a("v"), a("T"), a("p_full"), a("p_half"), z, z, None, a("lat"), dt, sub)
    out1, carry = fn(*args, carry, cell_area=area)
    assert _err(out1.du_dt, fix[f"L_{g}_wms_dt450_PTENU"]) <= _TOL
    assert _err(out1.dv_dt, fix[f"L_{g}_wms_dt450_PTENV"]) <= _TOL
    np.testing.assert_array_equal(np.asarray(out1.dT_dt),
                                  np.asarray(wms_heating(a("u"), a("v"), out1.du_dt, out1.dv_dt, dt)))
    u2 = a("u") * 1.2          # sheared state (WMS is Galilean-invariant to a uniform shift)
    n_keep = int(wms_call_interval(dt) / dt) - 1
    for _ in range(n_keep):
        out, carry = fn(u2, *args[1:], carry, cell_area=area)
        np.testing.assert_array_equal(np.asarray(out.du_dt), np.asarray(out1.du_dt))
    out, carry = fn(u2, *args[1:], carry, cell_area=area)   # interval reached
    assert _err(out.du_dt, np.asarray(out1.du_dt)) > 1e-6
    # composite threads the same carry and adds the orographic member
    ccfg = gcfg._replace(scheme="mcfarlane+ifs_wms")
    _, cfn, csub = get_gwd_fn(ccfg)
    assert gwd_carry_shape(ccfg, ncol, nlev) == carry.shape
    with pytest.raises(ValueError, match="cell area"):
        fn(*args, gwd_carry_seed(gcfg, ncol, nlev))


def test_wired_wms_refused_on_factory_lanes():
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import make_gwd_physics
    with pytest.raises(ValueError, match="MPAS column lane"):
        make_gwd_physics(GravityWaveDragConfig(scheme="mcfarlane+ifs_wms"), "hydrostatic", 300.0)


def test_overhead_column_seed_at_model_top(fix):
    """Vote Y1a: the top level sees the climatological column above the top
    half-level; its tendency moves by exactly k7 * seed / rapp vs the seed-0
    oracle form (hard switches, so no smoothing enters the difference)."""
    assert OzoneCariolleConfig().seed_overhead_column
    g = _GROUPS[0]
    a = lambda k: jnp.asarray(fix[f"{g}_{k}"])
    k, pf, ph, dt = a("PKOZO"), a("p_full"), a("p_half"), 450.0
    on = cariolle_tendency(a("PO3"), a("T"), pf, ph, k, dt,
                           cfg=_HARD_O3._replace(seed_overhead_column=True))
    off = cariolle_tendency(a("PO3"), a("T"), pf, ph, k, dt, cfg=_HARD_O3)
    gg = constants.g
    seed = k[:, 0, 5] - k[:, 0, 0] * (pf[:, 0] - ph[:, 0]) / gg
    rapp = 1.0 - dt * (k[:, 0, 2] + k[:, 0, 6] * (pf[:, 0] - ph[:, 0]) / gg)
    np.testing.assert_allclose(np.asarray(on[:, 0] - off[:, 0]),
                               np.asarray(k[:, 0, 6] * seed / rapp),
                               rtol=1e-9, atol=1e-30)

