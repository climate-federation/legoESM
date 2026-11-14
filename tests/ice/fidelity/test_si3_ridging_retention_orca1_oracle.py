"""ORCA1 oracle for SI3 jpl=1 ridging snow/pond retention (tier-4 dyn U11, defect D4).

Fixture ``fixtures/orca1_t4/si3_ridging_retention.{npz,json}``: 48 cells of NEMO
5.0.1 SI3 ``ice_dyn_rdgrft`` inputs/outputs recorded in the ORCA1_T4DYN run
(provenance in the JSON).  ORCA1 runs ``ln_icethd=T`` with
``rn_fsnwrdg = rn_fsnwrft = rn_fpndrdg = rn_fpndrft = 0.5``; the port's
validator used to refuse anything but 1.0 (NEMO's ``ln_icethd=F`` override).
Open water / ice area are deliberately not scored here (vote V6: the open-water
iteration is a separate defect).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ice.ridging import (
    SI3JPL1RidgingConfig,
    SI3JPL1RidgingState,
    apply_si3_jpl1_ridging,
)

_FIX = Path(__file__).parent / "fixtures" / "orca1_t4" / "si3_ridging_retention.npz"
_RTOL = 1.0e-14  # vote V5b: NEMO 1-ulp spread 8e-16 on these fields
_N_ACTIVE = 40  # first 40 fixture cells ridge with snow loss, last 8 untouched (V4a)


@pytest.fixture(autouse=True)
def _fp64_policy():
    old = get_policy()
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    yield
    set_policy(old)


def _load():
    z = dict(np.load(_FIX))
    meta = json.loads(_FIX.with_suffix(".json").read_text())
    return z, meta["scalars"]


def _state(z) -> SI3JPL1RidgingState:
    j = lambda k: jnp.asarray(z[k], jnp.float64)  # noqa: E731
    return SI3JPL1RidgingState(
        ice_area=j("a_i"), open_water_area=j("ato_i"), ice_volume=j("v_i"),
        snow_volume=j("v_s"), age_content=j("oa_i"), pond_area=j("a_ip"),
        pond_volume=j("v_ip"), pond_lid_volume=j("v_il"), snow_enthalpy=j("e_s"),
        ice_enthalpy=j("e_i"), ice_salt_content=j("szv_i"))


def _orca1(sc) -> SI3JPL1RidgingConfig:
    return SI3JPL1RidgingConfig(
        snow_ridge_retention=sc["rn_fsnwrdg"], snow_raft_retention=sc["rn_fsnwrft"],
        pond_ridge_retention=sc["rn_fpndrdg"], pond_raft_retention=sc["rn_fpndrft"],
        open_water_limit_sharpness=math.inf)  # NEMO hard open-water limits (oracle mode)


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    den = np.maximum(np.abs(a), np.abs(b))
    return np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0)


def _assert_close(port, nemo, what) -> None:
    """Vote T1a: rel <= 1e-14 OR abs <= 1e-14 * max|NEMO field over the fixture| (sub-ulp cells)."""
    port, nemo = np.asarray(port, np.float64), np.asarray(nemo, np.float64)
    ok = (_rel(port, nemo) <= _RTOL) | (np.abs(port - nemo) <= _RTOL * np.abs(nemo).max())
    assert ok.all(), (what, np.flatnonzero(~ok), float(_rel(port, nemo).max()))


def _run(z, sc, config):
    return apply_si3_jpl1_ridging(_state(z), jnp.asarray(z["divu_i"]), jnp.asarray(z["delta_i"]),
                                  sc["rDt_ice"], config=config)


def test_fixture_is_the_orca1_retention_regime() -> None:
    z, sc = _load()
    assert {sc[k] for k in ("rn_fsnwrdg", "rn_fsnwrft", "rn_fpndrdg", "rn_fpndrft")} == {0.5}
    assert sc["rn_porordg"] == 0.0
    assert (z["v_s"][:_N_ACTIVE] > z["v_s_out"][:_N_ACTIVE]).all()
    assert (z["v_s"][_N_ACTIVE:] == z["v_s_out"][_N_ACTIVE:]).all()


def test_orca1_retention_matches_nemo_snow_and_ice_inventories() -> None:
    z, sc = _load()
    out, _ = _run(z, sc, _orca1(sc))
    pairs = {"snow_volume": "v_s_out", "snow_enthalpy": "e_s_out", "ice_volume": "v_i_out",
             "ice_enthalpy": "e_i_out", "ice_salt_content": "szv_i_out", "age_content": "oa_i_out"}
    for port_key, nemo_key in pairs.items():
        _assert_close(getattr(out, port_key), z[nemo_key], port_key)
    # untouched cells stay bitwise
    np.testing.assert_array_equal(np.asarray(out.snow_volume)[_N_ACTIVE:], z["v_s_out"][_N_ACTIVE:])


def test_orca1_losses_reproduce_nemo_ocean_flux_increments() -> None:
    """Lost snow -> ocean: wfx_snw_dyn += rhos*v/dt, hfx_dyn += -e/dt (icedyn_rdgrft.F90:735-739)."""
    z, sc = _load()
    _, loss = _run(z, sc, _orca1(sc))
    dt = sc["rDt_ice"]
    wfx_port = sc["rhos"] * np.asarray(loss.snow_volume) / dt
    hfx_port = -np.asarray(loss.snow_enthalpy).sum(-1) / dt
    d_wfx = z["wfx_snw_dyn_out"] - z["wfx_snw_dyn"]
    d_hfx = z["hfx_dyn_out"] - z["hfx_dyn"]
    _assert_close(wfx_port, d_wfx, "wfx_snw_dyn")
    _assert_close(hfx_port, d_hfx, "hfx_dyn")
    assert (d_wfx[:_N_ACTIVE] > 0).all() and (d_hfx[:_N_ACTIVE] <= 0).all()
    # "untouched" cells: snow unchanged to the last bit, flux at most sub-ulp of v_s
    assert (np.abs(d_wfx[_N_ACTIVE:]) * sc["rDt_ice"] / sc["rhos"] <= 1.0e-16 * z["v_s"][_N_ACTIVE:]).all()
    # no porosity / ponds in this regime: the other ledgers do not move
    np.testing.assert_array_equal(z["wfx_dyn_out"], z["wfx_dyn"])
    np.testing.assert_array_equal(z["wfx_pnd_out"], z["wfx_pnd"])
    np.testing.assert_array_equal(np.asarray(loss.pond_volume), 0.0)


def test_snow_mass_and_heat_close_with_the_ocean_flux() -> None:
    z, sc = _load()
    out, loss = _run(z, sc, _orca1(sc))
    dt = sc["rDt_ice"]
    # ice side: storage change == material handed to the ocean
    dv = z["v_s"] - np.asarray(out.snow_volume)
    de = (z["e_s"] - np.asarray(out.snow_enthalpy)).sum(-1)
    scale_v, scale_e = np.abs(z["v_s"]).max(), np.abs(z["e_s"]).sum(-1).max()
    assert np.abs(dv - np.asarray(loss.snow_volume)).max() <= _RTOL * scale_v
    assert np.abs(de - np.asarray(loss.snow_enthalpy).sum(-1)).max() <= _RTOL * scale_e
    # ocean side: NEMO's recorded flux increments close against the port's storage change
    # (storage differences cancel to ~1 ulp of the inventory, so the floor is the inventory scale)
    w_nemo = z["wfx_snw_dyn_out"] - z["wfx_snw_dyn"]
    h_nemo = z["hfx_dyn_out"] - z["hfx_dyn"]
    assert np.abs(sc["rhos"] * dv / dt - w_nemo).max() <= _RTOL * sc["rhos"] * scale_v / dt
    assert np.abs(-de / dt - h_nemo).max() <= _RTOL * scale_e / dt


def test_ln_icethd_false_default_is_wrong_for_orca1() -> None:
    """Planted violation: the 1.0 default (ln_icethd=F override) misses NEMO by O(10%)."""
    z, sc = _load()
    out, loss = _run(z, sc, SI3JPL1RidgingConfig())
    assert _rel(out.snow_volume, z["v_s_out"])[:_N_ACTIVE].max() > 1.0e-2
    np.testing.assert_array_equal(np.asarray(loss.snow_volume), 0.0)


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan"), float("inf")])
def test_retention_outside_unit_interval_is_rejected(bad) -> None:
    z, sc = _load()
    with pytest.raises(ValueError, match="retained fraction"):
        _run(z, sc, SI3JPL1RidgingConfig(snow_raft_retention=bad))


def test_other_selector_fields_stay_pinned() -> None:
    z, sc = _load()
    with pytest.raises(ValueError, match="no Frankenstein fallback"):
        _run(z, sc, _orca1(sc)._replace(hstar_m=50.0))


def test_gradient_finite_and_matches_fd_at_orca1_retention() -> None:
    z, sc = _load()
    state, cfg = _state(z), _orca1(sc)
    div, dlt = jnp.asarray(z["divu_i"]), jnp.asarray(z["delta_i"])

    def objective(v_s, e_s):
        out, loss = apply_si3_jpl1_ridging(state._replace(snow_volume=v_s, snow_enthalpy=e_s), div, dlt,
                                           sc["rDt_ice"], config=cfg)
        return jnp.sum(out.snow_volume), jnp.sum(out.snow_enthalpy)

    g_v = jax.grad(lambda v: objective(v, state.snow_enthalpy)[0])(state.snow_volume)
    g_e = jax.grad(lambda e: objective(state.snow_volume, e)[1])(state.snow_enthalpy)
    assert np.isfinite(np.asarray(g_v)).all() and np.isfinite(np.asarray(g_e)).all()
    k = int(np.argmax(z["v_s"][:_N_ACTIVE] - z["v_s_out"][:_N_ACTIVE]))
    eps = 1.0e-6 * float(z["v_s"][k])
    bump = jnp.zeros_like(state.snow_volume).at[k].set(eps)
    fd = (objective(state.snow_volume + bump, state.snow_enthalpy)[0]
          - objective(state.snow_volume - bump, state.snow_enthalpy)[0]) / (2 * eps)
    assert 0.0 < float(g_v[k]) < 1.0  # snow partly lost
    assert abs(float(fd) / float(g_v[k]) - 1.0) < 1.0e-6
    eps_e = 1.0e-6 * float(z["e_s"][k, 0])
    bump_e = jnp.zeros_like(state.snow_enthalpy).at[k, 0].set(eps_e)
    fd_e = (objective(state.snow_volume, state.snow_enthalpy + bump_e)[1]
            - objective(state.snow_volume, state.snow_enthalpy - bump_e)[1]) / (2 * eps_e)
    assert 0.0 < float(g_e[k, 0]) < 1.0
    assert abs(float(fd_e) / float(g_e[k, 0]) - 1.0) < 1.0e-6
    # retention is a static namelist fraction (vote T2a); more retention never loses more snow
    outs = [np.asarray(apply_si3_jpl1_ridging(
        state, div, dlt, sc["rDt_ice"],
        config=cfg._replace(snow_ridge_retention=f, snow_raft_retention=f))[0].snow_volume) for f in (0.0, 0.5, 1.0)]
    assert (outs[0] <= outs[1]).all() and (outs[1] <= outs[2]).all()
    resolved = (z["v_s"] - z["v_s_out"]) > 1.0e-12 * z["v_s"]  # loss above roundoff of the inventory
    assert resolved.sum() >= 30
    assert (outs[0][resolved] < outs[2][resolved]).all()


def test_pond_retention_fields_are_wired_ridge_and_raft_separately() -> None:
    """Synthetic control (ORCA1 runs ln_pnd=F, so the record has no ponds): ponds injected on the
    fixture's single-shift cells.  Each pond factor, varied ALONE (snow kept at 1.0), must remove the
    same ridged/rafted share that the matching snow factor removes from snow (icedyn_rdgrft.F90:
    735-747,858-863); pond area transfer is linear in the factor; pond volume/lid close."""
    z, sc = _load()
    state = _state(z)
    state = state._replace(pond_area=0.1 * state.ice_area, pond_volume=0.05 * state.ice_volume,
                           pond_lid_volume=0.01 * state.ice_volume)
    div, dlt = jnp.asarray(z["divu_i"]), jnp.asarray(z["delta_i"])
    run = lambda **kw: apply_si3_jpl1_ridging(state, div, dlt, sc["rDt_ice"],  # noqa: E731
                                              config=SI3JPL1RidgingConfig(**kw))
    shares = {}
    for which in ("ridge", "raft"):
        snow_out, snow_loss = run(**{f"snow_{which}_retention": 0.3})
        pond = {f: run(**{f"pond_{which}_retention": f}) for f in (0.0, 0.3, 1.0)}
        out, loss = pond[0.3]
        one = (np.asarray(loss.iterations) == 1) & (z["v_s"] > 1e-6)
        assert one.sum() >= 10
        np.testing.assert_array_equal(np.asarray(loss.snow_volume), 0.0)          # snow untouched
        np.testing.assert_array_equal(np.asarray(snow_loss.pond_volume), 0.0)     # ponds untouched
        snow_share = np.asarray(snow_loss.snow_volume)[one] / z["v_s"][one]
        for inv in ("pond_volume", "pond_lid_volume"):
            share = np.asarray(getattr(loss, inv))[one] / np.asarray(getattr(state, inv))[one]
            np.testing.assert_allclose(share, snow_share, rtol=1e-12, atol=0)
            d = np.asarray(getattr(state, inv)) - np.asarray(getattr(out, inv))
            assert np.abs(d - np.asarray(getattr(loss, inv))).max() <= _RTOL * np.abs(np.asarray(getattr(state, inv))).max()
        a0, a3, a1 = (np.asarray(pond[f][0].pond_area)[one] for f in (0.0, 0.3, 1.0))
        res = snow_share > 1.0e-6  # cells where this mode (ridge or raft) moves a resolvable share
        assert res.sum() >= 5
        assert (a1 - a0 > 0)[res].all()
        np.testing.assert_allclose(((a1 - a3) / np.where(res, a1 - a0, 1.0))[res], 0.7, rtol=1e-9)
        shares[which] = snow_share
    assert not np.allclose(shares["ridge"], shares["raft"])
