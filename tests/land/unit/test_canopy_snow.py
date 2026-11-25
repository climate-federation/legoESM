"""CLM5.0 intercepted canopy snow (snowveg_flag ON_RAD) on the two-leaf land.

Oracle: ``oracle_fixtures/clm5_canopy_snow.npz``, produced by compiling the
verbatim CTSM release-clm5.0.34 CanopyHydrologyMod.F90 (294-304, 309-392,
417-443, 717-742) and CanopyFluxesMod.F90 (1257-1272) line ranges with gfortran
(generator: scripts/validate/oracle/clm5_canopy_snow/).

RULE 5, per smooth switch (fall-off only at or below freezing, the 270.15 K
unloading ramp, the leaf-phase split, the caps): (1) the hard limit (widths 0)
matches the oracle to 1e-12 relative; (2) the default widths' per-case gap is
the stored measurement ``oracle_fixtures/clm5_canopy_snow_gaps.json``
(1e-6 relative, 1e-12 absolute floor; regenerate with
``LEGOESM_REGEN_CANOPY_SNOW_GAPS=1`` and record why); (3) ``check_grads``
through each switch at the default.  RULE 0: the store and the land column
close their water budget and the canopy sublimation's latent heat is L_s.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.land.canopy.interception import (
    CLM5_CANOPY_WATER, CanopySnowConfig, canopy_cold_weight, canopy_snow_fraction,
    canopy_snow_step, split_wet_evap, wetted_fraction)

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"), reason="float64 oracle; JAX_ENABLE_X64=1")

_DIR = Path(__file__).parent / "oracle_fixtures"
_FIX = _DIR / "clm5_canopy_snow.npz"
_GAPS = _DIR / "clm5_canopy_snow_gaps.json"
_HARD = CanopySnowConfig(temperature_width_K=0.0, relative_width=0.0, fcansno_ramp=0.0)
# Fixture cases whose evaporation exceeds both stores (counted 2026-11-22 on
# the generator's grid); a change here means the fixture or the filter moved.
_N_OVERDRAW = 144
_N_DRY_SHED = 8
_FIELDS = ("snocan_hyd", "grnd_snow", "fwet", "fcansno", "snocan_evap", "liqcan_evap")


def _ours(fx, cfg):
    """Our port on the fixture inputs; same order as the oracle driver."""
    g = lambda k: jnp.asarray(fx[f"in_{k}"])
    dt = float(fx["in_dtime"][0])
    ic = CLM5_CANOPY_WATER._replace(dewmx=float(fx["in_dewmx"][0]))
    vai = g("elai") + g("esai")
    s_hyd, grnd = canopy_snow_step(g("snocan"), g("forc_snow"), g("forc_t"),
                                   g("forc_wind"), vai, dt, ic, cfg)
    liq = g("liqcan")
    fwet = wetted_fraction(s_hyd + liq, vai, ic)
    fcs = canopy_snow_fraction(s_hyd, vai, ic, cfg)
    dem = g("evap") * dt
    cold = canopy_cold_weight(g("t_veg"), constants.T_freeze, cfg.temperature_width_K)
    fs, fl = split_wet_evap(cold * dem, (1.0 - cold) * dem, s_hyd, liq, cfg.relative_width)
    return dict(snocan_hyd=s_hyd, grnd_snow=grnd, fwet=fwet, fcansno=fcs,
                snocan_evap=s_hyd - fs, liqcan_evap=liq - fl)


def _dry_shed(fx):
    """Dry steps below freezing with the store above capacity: CLM5 sheds
    only when it precipitates, the port sheds any such step (recorded)."""
    cap = 60.0 * fx["in_dewmx"] * (fx["in_elai"] + fx["in_esai"])
    return ((fx["in_forc_snow"] + fx["in_forc_rain"] == 0.0)
            & (fx["in_snocan"] > cap) & (fx["in_forc_t"] <= constants.T_freeze))


def _valid(fx):
    """Cases inside CLM's own contract: the evaporation the stores can supply
    (CLM caps it upstream by h2ocan/dtime; an over-draw drives one store
    negative, which the port refuses by construction); minus the dry sheds."""
    dem = fx["in_evap"] * fx["in_dtime"]
    return (dem <= fx["out_snocan_hyd"] + fx["in_liqcan"]) & ~_dry_shed(fx)


def _gap(ours, ref):
    return np.abs(np.asarray(ours) - ref) / np.maximum(np.abs(ref), 1e-12)


def test_hard_limit_matches_clm5_oracle():
    fx = np.load(_FIX)
    ok = _valid(fx)
    ours = _ours(fx, _HARD)
    for k in _FIELDS:
        np.testing.assert_allclose(np.asarray(ours[k])[ok], fx[f"out_{k}"][ok],
                                   rtol=1e-12, atol=1e-15, err_msg=k)
    # Over-draw cases (demand above both stores, after unloading shrank the
    # snow store): the hydrology fields still match exactly; the port empties
    # both stores where CLM5 drives one of them negative (recorded difference).
    dry = _dry_shed(fx)
    assert int(dry.sum()) == _N_DRY_SHED
    cap = 60.0 * fx["in_dewmx"] * (fx["in_elai"] + fx["in_esai"])
    # same unloading fraction applied to the shed store (port) vs the full one (CLM)
    np.testing.assert_allclose(np.asarray(ours["snocan_hyd"])[dry],
                               fx["out_snocan_hyd"][dry] * cap[dry] / fx["in_snocan"][dry],
                               rtol=1e-12)
    bad = ~ok & ~dry
    assert int(bad.sum()) == _N_OVERDRAW
    for k in ("snocan_hyd", "grnd_snow", "fwet", "fcansno"):
        np.testing.assert_allclose(np.asarray(ours[k])[bad], fx[f"out_{k}"][bad],
                                   rtol=1e-12, atol=1e-15, err_msg=k)
    np.testing.assert_allclose(np.asarray(ours["snocan_evap"])[bad], 0.0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(ours["liqcan_evap"])[bad], 0.0, atol=1e-15)
    assert np.all((fx["out_liqcan_evap"][bad] < 0.0) | (fx["out_snocan_evap"][bad] < 0.0))


def test_masking_passes_the_canopy_snow_fraction_to_the_two_stream():
    """Delta form with canopy snow: alb_sf + TS(elai, esai; snowy, fcansno)
    - TS(tlai, tsai; snow-free, 0), the two-stream itself being oracle-tested
    with fcansno in test_canopy_snow_albedo."""
    from legoesm.land.canopy import radiative_transfer as rt
    a = dict(band=0, LAI=np.array([2.0]), SAI=np.array([0.6]), htop=np.array([17.0]),
             hbot=np.array([8.5]), pft_index=np.array([2.0]), f_snow=np.array([1.0]),
             snow_depth=np.array([0.2]), cosz=np.array([0.4]), f_diffuse=np.array([0.3]))
    m0 = rt.canopy_masked_snow_albedo(np.array([0.05]), np.array([0.8]), **a)
    m1 = rt.canopy_masked_snow_albedo(np.array([0.05]), np.array([0.8]), **a, fcansno=0.7)
    xl, rl, tl, rs, ts = (np.asarray(t) for t in rt._clm5_pft_optics())
    def ts_(g, f):
        d, i = rt.clm5_two_stream_albedo(jnp.asarray(2.6), jnp.asarray(2.0 / 2.6), rl[2, 0],
                                         tl[2, 0], rs[2, 0], ts[2, 0], xl[2], jnp.asarray(0.4),
                                         jnp.asarray(g), fcansno=f, omega_snow=0.8)
        return 0.3 * i + 0.7 * d
    np.testing.assert_allclose(float(m1[0] - m0[0]), float(ts_(0.8, 0.7) - ts_(0.8, 0.0)),
                               rtol=1e-12)
    assert float(m1[0]) > float(m0[0]) + 0.05


def test_default_widths_gap_is_the_recorded_one_per_case():
    fx = np.load(_FIX)
    ok = _valid(fx)
    ours = _ours(fx, CanopySnowConfig())
    gaps = {k: _gap(ours[k], fx[f"out_{k}"])[ok].tolist() for k in _FIELDS}
    if os.environ.get("LEGOESM_REGEN_CANOPY_SNOW_GAPS") == "1":
        _GAPS.write_text(json.dumps(gaps))
    stored = json.loads(_GAPS.read_text())
    for k in _FIELDS:
        np.testing.assert_allclose(gaps[k], stored[k], rtol=1e-6, atol=1e-12, err_msg=k)
    # The smooth default is a real, bounded departure, not a silent no-op.
    assert max(max(v) for v in gaps.values()) > 0.0


@pytest.mark.parametrize("T_air,s0,wind", [
    (constants.T_freeze + 0.25, 30.0, 3.0),   # inside the fall-off cold band
    (270.15 + 0.2, 6.0, 3.0),                 # inside the unloading ramp
    (250.0, 23.6, 3.0),                       # store just above capacity (cap band)
    (275.0, 6.0, 1.0e5),                      # unloading capped at the store
])
def test_canopy_snow_step_gradient_through_each_switch(T_air, s0, wind):
    cfg = CanopySnowConfig()

    def f(s, snow, T, U, vai):
        a, b = canopy_snow_step(s, snow, T, U, vai, 1800.0, CLM5_CANOPY_WATER, cfg)
        return a + 1000.0 * b

    args = tuple(jnp.asarray(v) for v in (s0, 3.0e-4, T_air, wind, 3.0))
    check_grads(f, args, order=1, modes=("fwd", "rev"), eps=1e-6)
    g = jax.grad(f, argnums=(0, 2))(*args)
    assert all(np.isfinite(float(x)) for x in g)
    assert float(jnp.abs(g[0])) > 0.0


def test_fcansno_and_phase_split_gradients():
    cfg = CanopySnowConfig()
    check_grads(lambda s: canopy_snow_fraction(s, jnp.asarray(2.6), CLM5_CANOPY_WATER, cfg),
                (jnp.asarray(15.0),), order=1, modes=("fwd", "rev"), eps=1e-6)
    # at and near an empty store (C1 ramp) the derivative is finite and positive
    for s0 in (0.0, 1e-5, 0.01):
        g0 = jax.grad(lambda s: canopy_snow_fraction(s, jnp.asarray(2.6), CLM5_CANOPY_WATER, cfg))(
            jnp.asarray(s0))
        assert np.isfinite(float(g0)) and float(g0) > 0.0
    # at the empty store it is the ramp's analytic slope a / snocanmx
    p, x0, cap = 0.15, cfg.fcansno_ramp, 60.0 * 0.1 * 2.6
    g0 = jax.grad(lambda s: canopy_snow_fraction(s, jnp.asarray(2.6), CLM5_CANOPY_WATER, cfg))(
        jnp.asarray(0.0))
    np.testing.assert_allclose(float(g0), (2.0 - p) * x0 ** (p - 1.0) / cap, rtol=1e-12)
    check_grads(lambda s: canopy_snow_fraction(s, jnp.asarray(2.6), CLM5_CANOPY_WATER, cfg),
                (jnp.asarray(0.01),), order=1, modes=("fwd", "rev"), eps=1e-6)
    # near fcansno -> 1 (cap band) the derivative is finite and non-negative
    gs = jax.grad(lambda s: canopy_snow_fraction(s, jnp.asarray(2.6), CLM5_CANOPY_WATER, cfg))(
        jnp.asarray(15.4))
    assert np.isfinite(float(gs)) and float(gs) >= 0.0

    def split(T, dem, sn, lq):
        c = canopy_cold_weight(T, constants.T_freeze, cfg.temperature_width_K)
        a, b = split_wet_evap(c * dem, (1.0 - c) * dem, sn, lq, cfg.relative_width)
        return a - 2.0 * b

    args = tuple(jnp.asarray(v) for v in (constants.T_freeze + 0.2, 0.1, 0.2, 0.05))
    check_grads(split, args, order=1, modes=("fwd", "rev"), eps=1e-6)
    assert float(jnp.abs(jax.grad(split)(*args))) > 0.0


@pytest.mark.parametrize("cfg", [CanopySnowConfig(), _HARD], ids=["default", "hard"])
def test_canopy_snow_step_mass_closes(cfg):
    rng = np.random.default_rng(3)
    n, dt = 400, 1800.0
    s0 = jnp.asarray(rng.uniform(0.0, 40.0, n))
    snow = jnp.asarray(rng.uniform(0.0, 1e-3, n))
    T = jnp.asarray(rng.uniform(250.0, 280.0, n))
    U = jnp.asarray(rng.uniform(0.0, 20.0, n))
    vai = jnp.asarray(rng.uniform(0.0, 6.0, n))
    s1, grnd = canopy_snow_step(s0, snow, T, U, vai, dt, CLM5_CANOPY_WATER, cfg)
    res = snow * dt - grnd * dt - (s1 - s0)
    assert float(jnp.max(jnp.abs(res))) < 1e-12 * float(jnp.max(s0))
    assert float(jnp.min(s1)) >= 0.0 and float(jnp.min(grnd)) >= 0.0


# ---- the land step: column water and energy with canopy snow ----------------
from legoesm.core.coupling_fields import AtmToSurface  # noqa: E402
from legoesm.land.boundary_data.gap_fill import bare_canopy_params  # noqa: E402
from legoesm.land.multilayer_land import (  # noqa: E402
    MultiLayerLandConfig, init_multilayer_land_state,
    step_multilayer_land_with_diagnostics)
from legoesm.land.soil_grid import make_soil_grid  # noqa: E402
from legoesm.land.surface_scheme import TwoLeafCanopyConfig  # noqa: E402
from legoesm.thermo import latent_heat_sublimation, latent_heat_vaporization  # noqa: E402

_N = 3   # spruce (cold, snowing) / spruce (warm, unloading) / bare ground


def _forcing(T_air):
    o = jnp.ones(_N)
    return AtmToSurface(
        sw_down=200.0 * o, lw_down=220.0 * o, precip_total=4.0e-4 * o,
        precip_snow=4.0e-4 * o, T_lowest=jnp.asarray(T_air), q_lowest=0.0006 * o,
        u_lowest=6.0 * o, v_lowest=1.0 * o, p_lowest=95000.0 * o,
        p_surface=1.0e5 * o,
        rho_lowest=1.0e5 / (constants.R_d * jnp.asarray(T_air)),
        cos_zenith=0.4 * o, co2_ppmv=400.0 * o, has_radiation=o,
        has_precipitation=o)


def _cfg(on=True):
    return MultiLayerLandConfig(
        snow_albedo_feedback=True, canopy_snow_masking=True,
        surface_scheme=TwoLeafCanopyConfig(implicit_ground_flux=False),
        canopy_snow=CanopySnowConfig() if on else None)


def _params():
    return bare_canopy_params(_N, canopy_structure=True)._replace(
        LAI=jnp.asarray([2.0, 2.0, 0.0]), SAI_dom=jnp.asarray([0.6, 0.6, 0.0]),
        hc=jnp.full(_N, 17.0), hbot_dom=jnp.full(_N, 8.5),
        pft_dom=jnp.asarray([2.0, 2.0, 0.0]),
        ALB_VIS=jnp.full(_N, 0.05), ALB_NIR=jnp.full(_N, 0.22))


def _run(n_steps, T_air=(258.0, 274.0, 258.0), on=True, s_can0=(5.0, 12.0, 0.0)):
    cfg = _cfg(on)
    st = init_multilayer_land_state(_N, cfg, T_init=262.0)
    st = st._replace(snow_depth=jnp.full(_N, 60.0))
    if on:
        st = st._replace(W_canopy_snow=jnp.asarray(s_can0))
    lp, f, dt = _params(), _forcing(T_air), 1800.0
    dz = jnp.asarray(make_soil_grid(cfg.soil_grid).dz)
    step = jax.jit(lambda s: step_multilayer_land_with_diagnostics(
        s, f, cfg, 1.0, dt, lat=jnp.full(_N, 1.1), land_params=lp))

    def water(s):
        wc = 0.0 if s.W_canopy_snow is None else s.W_canopy_snow
        sw = 0.0 if s.surface_water is None else s.surface_water
        return (jnp.sum(dz * s.theta_soil, -1) + sw) * constants.rho_water + s.snow_depth + wc

    out = []
    for _ in range(n_steps):
        s2, r, _, so = step(st)
        out.append((st, s2, r, so))
        st = s2
    return f, dt, water, out


def test_land_column_water_closes_with_canopy_snow():
    f, dt, water, out = _run(6)
    for s, s2, r, so in out:
        assert int(np.asarray(so.n_held).sum()) == 0
        ro = (s2.runoff_surface + s2.runoff_subsurface) * dt
        res = water(s2) - water(s) - (f.precip_total - r.surface_mass_flux) * dt + ro
        np.testing.assert_allclose(np.asarray(res), 0.0, atol=1e-9)
    s_end = out[-1][1].W_canopy_snow
    assert float(s_end[0]) > 5.0          # cold snowing spruce accumulates
    assert float(s_end[1]) < 12.0         # warm spruce unloads
    assert float(s_end[2]) == 0.0         # no plant area: nothing intercepted


def test_canopy_sublimation_pays_ls_and_leaves_sh_plus_le():
    """No ground snow, no precipitation, calm air, so the only ice leaving is
    the canopy store's.  (a) The reported latent heat is the vapour mass at L_v plus
    (L_s - L_v) E_canopy_snow; (b) the sensible heat pays that difference, so
    SH + LE equals the surface solve's demanded total."""
    cfg = _cfg(True)
    st = init_multilayer_land_state(_N, cfg, T_init=262.0)
    st = st._replace(W_canopy_snow=jnp.asarray([10.0, 10.0, 0.0]))
    # calm and below 270.15 K: nothing unloads, so no ground snow appears
    f = _forcing((258.0,) * _N)._replace(precip_total=jnp.zeros(_N),
                                         precip_snow=jnp.zeros(_N),
                                         u_lowest=jnp.zeros(_N), v_lowest=jnp.zeros(_N))
    s2, r, _, so = step_multilayer_land_with_diagnostics(
        st, f, cfg, 1.0, 1800.0, lat=jnp.full(_N, 1.1), land_params=_params())
    E = np.asarray(so.canopy_snow_sublimation)
    assert E[0] > 0.0 and E[2] == 0.0
    T_s = st.T_soil[:, 0]
    L_v = np.asarray(latent_heat_vaporization(T_s))
    dL = np.asarray(latent_heat_sublimation(T_s)) - L_v
    np.testing.assert_allclose(np.asarray(r.lhflx) - np.asarray(r.surface_mass_flux) * L_v,
                               dL * E, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(r.shflx + r.lhflx),
                               np.asarray(so.shflx + so.lhflx), rtol=1e-12, atol=1e-9)


def test_empty_store_without_snow_is_bit_identical_to_switch_off():
    """Canopy snow ON with an empty store, no snowfall and calm air (nothing
    to intercept or unload) must reproduce the switch-off step bit for bit:
    the port changes nothing it has no snow for.  Off carries no store."""
    f = _forcing((258.0, 274.0, 258.0))._replace(
        precip_total=jnp.zeros(_N), precip_snow=jnp.zeros(_N),
        u_lowest=jnp.zeros(_N), v_lowest=jnp.zeros(_N))
    res = {}
    for on in (False, True):
        cfg = _cfg(on)
        st = init_multilayer_land_state(_N, cfg, T_init=262.0)._replace(
            snow_depth=jnp.full(_N, 60.0))
        res[on] = step_multilayer_land_with_diagnostics(
            st, f, cfg, 1.0, 1800.0, lat=jnp.full(_N, 1.1), land_params=_params())
    assert res[False][0].W_canopy_snow is None
    assert np.all(np.asarray(res[True][0].W_canopy_snow) == 0.0)
    for name in res[False][1]._fields:
        a, b = getattr(res[False][1], name), getattr(res[True][1], name)
        if a is not None:
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=name)
    for name in ("T_soil", "theta_soil", "snow_depth", "snow_age"):
        np.testing.assert_array_equal(np.asarray(getattr(res[False][0], name)),
                                      np.asarray(getattr(res[True][0], name)), err_msg=name)


def test_canopy_snow_refuses_without_masking():
    cfg = _cfg(True)._replace(canopy_snow_masking=False)
    st = init_multilayer_land_state(_N, cfg, T_init=262.0)
    with pytest.raises(ValueError, match="canopy_snow needs canopy_snow_masking"):
        step_multilayer_land_with_diagnostics(
            st, _forcing((258.0,) * _N), cfg, 1.0, 1800.0, lat=jnp.full(_N, 1.1),
            land_params=_params())


def test_canopy_snow_whitens_the_exported_albedo():
    _, _, _, on = _run(1, s_can0=(20.0, 20.0, 0.0))
    _, _, _, off = _run(1, on=False)
    a_on, a_off = np.asarray(on[0][2].albedo), np.asarray(off[0][2].albedo)
    assert a_on[0] > a_off[0] + 0.02          # snow on the spruce brightens it
    assert a_on[2] == a_off[2]                # bare ground: no canopy, unchanged


def test_restart_round_trips_the_canopy_snow_store_and_old_files_start_empty(tmp_path):
    """The store is real mass: it round-trips through the land restart, and a
    restart written without it (feature off) merges as an empty store."""
    from legoesm.land.restart import (
        load_land_restart, merge_land_restart_into_template, save_land_restart)
    cfg = _cfg(True)
    st = init_multilayer_land_state(_N, cfg, T_init=262.0)._replace(
        W_canopy_snow=jnp.asarray([3.0, 7.5, 0.0]))
    kw = dict(land_mode="multilayer", t_end_s=0.0, n_steps_completed=0,
              soil_grid=cfg.soil_grid)
    save_land_restart(tmp_path / "on.npz", st, **kw)
    ld, _ = load_land_restart(tmp_path / "on.npz", expected_land_mode="multilayer",
                              expected_ncol=_N)
    m = merge_land_restart_into_template(ld, st._replace(W_canopy_snow=jnp.zeros(_N)))
    np.testing.assert_array_equal(np.asarray(m.W_canopy_snow), [3.0, 7.5, 0.0])
    off = init_multilayer_land_state(_N, _cfg(False), T_init=262.0)
    save_land_restart(tmp_path / "off.npz", off, **kw)
    ld, _ = load_land_restart(tmp_path / "off.npz", expected_land_mode="multilayer",
                              expected_ncol=_N)
    m = merge_land_restart_into_template(ld, st._replace(W_canopy_snow=jnp.zeros(_N)))
    np.testing.assert_array_equal(np.asarray(m.W_canopy_snow), np.zeros(_N))


def test_restart_refuses_to_discard_canopy_snow(tmp_path):
    from legoesm.land.restart import (
        load_land_restart, merge_land_restart_into_template, save_land_restart)
    cfg = _cfg(True)
    st = init_multilayer_land_state(_N, cfg, T_init=262.0)._replace(
        W_canopy_snow=jnp.asarray([3.0, 0.0, 0.0]))
    save_land_restart(tmp_path / "on.npz", st, land_mode="multilayer", t_end_s=0.0,
                      n_steps_completed=0, soil_grid=cfg.soil_grid)
    ld, _ = load_land_restart(tmp_path / "on.npz", expected_land_mode="multilayer",
                              expected_ncol=_N)
    off = init_multilayer_land_state(_N, _cfg(False), T_init=262.0)
    with pytest.raises(ValueError, match="refusing to discard that water"):
        merge_land_restart_into_template(ld, off)
    # an EMPTY store carries no mass and merges into the switch-off run
    m = merge_land_restart_into_template(
        ld._replace(W_canopy_snow=jnp.zeros(_N)), off)
    assert m.W_canopy_snow is None


def test_canopy_sublimation_not_split_onto_dry_bare_soil():
    """Snow-free ground over a dry root zone: the canopy snow store funds the
    whole wet-leaf demand it can supply (reserved before the bare-soil / root
    partition), not the share that partition would give the canopy."""
    cfg = _cfg(True)
    st = init_multilayer_land_state(_N, cfg, T_init=262.0, theta_init=0.03)
    st = st._replace(W_canopy_snow=jnp.asarray([10.0, 10.0, 0.0]))
    f = _forcing((258.0,) * _N)._replace(precip_total=jnp.zeros(_N),
                                         precip_snow=jnp.zeros(_N),
                                         u_lowest=jnp.zeros(_N), v_lowest=jnp.zeros(_N))
    _, r, _, so = step_multilayer_land_with_diagnostics(
        st, f, cfg, 1.0, 1800.0, lat=jnp.full(_N, 1.1), land_params=_params())
    le_wet = np.asarray(so.LE_wet_canopy)
    E = np.asarray(so.canopy_snow_sublimation)
    L_v = np.asarray(latent_heat_vaporization(st.T_soil[:, 0]))
    assert le_wet[0] > 1.0
    np.testing.assert_allclose(E[0] * L_v[0], le_wet[0], rtol=0.05)


def test_held_column_reports_no_canopy_sublimation():
    """A column the land step holds (non-finite forcing) is reverted; its
    canopy-snow sublimation diagnostic must be zero, not the discarded value."""
    cfg = _cfg(True)
    st = init_multilayer_land_state(_N, cfg, T_init=262.0)._replace(
        W_canopy_snow=jnp.asarray([10.0, 10.0, 0.0]))
    f = _forcing((258.0,) * _N)
    f = f._replace(lw_down=f.lw_down.at[0].set(jnp.nan))
    s2, _, _, so = step_multilayer_land_with_diagnostics(
        st, f, cfg, 1.0, 1800.0, lat=jnp.full(_N, 1.1), land_params=_params())
    assert bool(np.asarray(so.held)[0])
    assert float(so.canopy_snow_sublimation[0]) == 0.0
    assert float(s2.W_canopy_snow[0]) == 10.0


@pytest.mark.parametrize("rw", [0.05, 1.0])
def test_split_draws_stay_within_each_store(rw):
    """The chained smooth mins never exceed their smaller operand, so every
    draw is in [0, its store] and the total never exceeds demand or stores,
    including inside the smoothing bands (random demands near the stores)."""
    rng = np.random.default_rng(7)
    n = 5000
    sn = jnp.asarray(rng.uniform(0.0, 1.0, n))
    lq = jnp.asarray(rng.uniform(0.0, 1.0, n))
    dem = jnp.asarray(rng.uniform(0.0, 2.5, n)) * (sn + lq) / 2.0
    c = jnp.asarray(rng.uniform(0.0, 1.0, n))
    fs, fl = split_wet_evap(c * dem, (1.0 - c) * dem, sn, lq, rw)
    fs, fl = np.asarray(fs), np.asarray(fl)
    assert fs.min() >= 0.0 and fl.min() >= 0.0
    assert np.all(fs <= np.asarray(sn) + 1e-15) and np.all(fl <= np.asarray(lq) + 1e-15)
    assert np.all(fs + fl <= np.asarray(dem) + 1e-15)


def test_land_step_overdraw_empties_but_never_overdraws_the_store():
    """A nearly empty store under a wet-leaf demand far above it: the column
    sublimates at most the store, which ends non-negative."""
    cfg = _cfg(True)
    st = init_multilayer_land_state(_N, cfg, T_init=262.0)._replace(
        W_canopy_snow=jnp.asarray([1e-4, 1e-4, 0.0]))
    f = _forcing((258.0,) * _N)._replace(precip_total=jnp.zeros(_N),
                                         precip_snow=jnp.zeros(_N),
                                         u_lowest=jnp.zeros(_N), v_lowest=jnp.zeros(_N))
    s2, _, _, so = step_multilayer_land_with_diagnostics(
        st, f, cfg, 1.0, 1800.0, lat=jnp.full(_N, 1.1), land_params=_params())
    E = np.asarray(so.canopy_snow_sublimation) * 1800.0
    assert np.all(np.asarray(s2.W_canopy_snow) >= 0.0)
    assert np.all(E <= np.asarray(st.W_canopy_snow) + 1e-15)
    np.testing.assert_allclose(np.asarray(s2.W_canopy_snow) + E,
                               np.asarray(st.W_canopy_snow), atol=1e-15)


def test_warm_leaves_draw_liquid_first_then_the_reserved_snow():
    """Liquid interception on, warm leaves, a small liquid store, plenty of
    canopy snow and a dry root zone: CLM5's warm order takes the liquid first
    and the REST of the wet-leaf demand from the snow store; the snow share
    reserved at the soil cap must reach the snow store (not be re-split onto
    the liquid)."""
    from legoesm.land.canopy.interception import InterceptionConfig
    cfg = _cfg(True)._replace(interception=InterceptionConfig())
    st = init_multilayer_land_state(_N, cfg, T_init=275.0, theta_init=0.03)
    st = st._replace(W_canopy_snow=jnp.asarray([10.0, 10.0, 0.0]),
                     W_canopy=jnp.asarray([0.003, 0.003, 0.0]))
    f = _forcing((276.0,) * _N)._replace(precip_total=jnp.zeros(_N),
                                         precip_snow=jnp.zeros(_N),
                                         u_lowest=jnp.zeros(_N), v_lowest=jnp.zeros(_N))
    s2, _, _, so = step_multilayer_land_with_diagnostics(
        st, f, cfg, 1.0, 1800.0, lat=jnp.full(_N, 1.1), land_params=_params())
    dt = 1800.0
    L_v = np.asarray(latent_heat_vaporization(st.T_soil[:, 0]))
    dem = np.asarray(so.LE_wet_canopy) / L_v * dt
    liq = np.asarray(st.W_canopy - s2.W_canopy)
    snow_e = np.asarray(so.canopy_snow_sublimation) * dt
    assert dem[0] > 0.006                         # demand well above the liquid store
    np.testing.assert_allclose(liq[0], 0.003, rtol=0.05)  # liquid first, emptied
    np.testing.assert_allclose(liq[0] + snow_e[0], dem[0], rtol=0.05)


def test_reserved_snow_is_not_counted_in_the_root_partition(monkeypatch):
    """The reserved canopy sublimation is not transpiration: the bare-soil /
    root partition of the remaining soil stream must see the canopy latent
    heat WITHOUT it (else ground evaporation is routed to the roots)."""
    import legoesm.land.multilayer_land as ml
    seen = {}
    orig = ml._partition_latent_root_top

    def spy(soil_evap, has_snow, f_veg, le_canopy, le_soil):
        jax.debug.callback(lambda v: seen.__setitem__("le_canopy", np.asarray(v)),
                           le_canopy)
        return orig(soil_evap, has_snow, f_veg, le_canopy, le_soil)

    monkeypatch.setattr(ml, "_partition_latent_root_top", spy)
    cfg = _cfg(True)
    st = init_multilayer_land_state(_N, cfg, T_init=274.0, theta_init=0.3)
    st = st._replace(W_canopy_snow=jnp.asarray([10.0, 10.0, 0.0]))
    f = _forcing((271.0,) * _N)._replace(
        precip_total=jnp.zeros(_N), precip_snow=jnp.zeros(_N),
        u_lowest=jnp.zeros(_N), v_lowest=jnp.zeros(_N))
    _, _, _, so = step_multilayer_land_with_diagnostics(
        st, f, cfg, 1.0, 1800.0, lat=jnp.full(_N, 1.1), land_params=_params())
    jax.effects_barrier()
    E = np.asarray(so.canopy_snow_sublimation)
    assert E[0] > 0.0
    L_v = np.asarray(latent_heat_vaporization(st.T_soil[:, 0]))
    np.testing.assert_allclose(np.asarray(seen["le_canopy"]),
                               np.asarray(so.LE_canopy) - E * L_v, rtol=1e-12, atol=1e-9)


def test_fully_intercepted_snowfall_does_not_make_the_ground_snowy():
    """A dense canopy (exposed plant area 40: fpisnow = 1 - e^-20) catches all
    but ~1e-9 kg m-2 of the step's snowfall: the bare ground must NOT be
    treated as snow-covered (its latent heat stays L_v), because the gate
    reads the snow that actually reaches it."""
    cfg = _cfg(True)
    st = init_multilayer_land_state(_N, cfg, T_init=262.0)
    lp = _params()._replace(LAI=jnp.asarray([40.0, 2.0, 0.0]),
                            SAI_dom=jnp.asarray([0.0, 0.6, 0.0]))
    f = _forcing((258.0,) * _N)._replace(u_lowest=jnp.zeros(_N), v_lowest=jnp.zeros(_N))
    _, _, _, so = step_multilayer_land_with_diagnostics(
        st, f, cfg, 1.0, 1800.0, lat=jnp.full(_N, 1.1), land_params=lp)
    L_v = np.asarray(latent_heat_vaporization(st.T_soil[:, 0]))
    L_s = np.asarray(latent_heat_sublimation(st.T_soil[:, 0]))
    np.testing.assert_allclose(float(so.L_soil[0]), L_v[0], rtol=1e-12)   # no ground snow
    np.testing.assert_allclose(float(so.L_soil[2]), L_s[2], rtol=1e-12)   # bare: all reaches it


def test_buried_canopy_with_snow_has_finite_gradients():
    """Zero exposed plant area (capacity 0) while the store still holds snow:
    the relative caps see a zero scale; values and gradients stay finite."""
    cfg = CanopySnowConfig()

    def f(s, T, vai):
        a, b = canopy_snow_step(s, 3.0e-4, T, 4.0, vai, 1800.0, CLM5_CANOPY_WATER, cfg)
        return a + 1000.0 * b + canopy_snow_fraction(a, vai, CLM5_CANOPY_WATER, cfg)

    for T in (255.0, constants.T_freeze + 0.25, 275.0):
        g = jax.grad(f, argnums=(0, 1, 2))(jnp.asarray(5.0), jnp.asarray(T), jnp.asarray(0.0))
        assert all(np.isfinite(float(x)) for x in g), (T, g)


def test_fcansno_gradient_is_continuous_at_the_ramp_join():
    """At snocan/snocanmx == fcansno_ramp both sides give p x0**(p-1)/cap."""
    cfg = CanopySnowConfig()
    vai = jnp.asarray(2.6)
    cap = 60.0 * CLM5_CANOPY_WATER.dewmx * 2.6
    x0, p = cfg.fcansno_ramp, 0.15
    fn = lambda s: canopy_snow_fraction(s, vai, CLM5_CANOPY_WATER, cfg)
    want = p * x0 ** (p - 1.0) / cap
    for s0 in (x0 * cap, x0 * cap * (1 - 1e-9), x0 * cap * (1 + 1e-9)):
        np.testing.assert_allclose(float(jax.grad(fn)(jnp.asarray(s0))), want, rtol=1e-6)
