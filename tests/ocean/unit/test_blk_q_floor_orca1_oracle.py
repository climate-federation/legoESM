"""ORCA1 oracle: the legacy NCAR bulk path floors humidity like NEMO and leaves
the 10-m pressure's q/q_sat unclipped.

NEMO 5.0.1 ``turb_ncar`` sets ``t_zu = MAX(t_zt, 180)``, ``q_zu = MAX(q_zt,
1e-6)`` after the first guess (sbcblk_algo_ncar.F90) and feeds them to 1/L,
dq and BULK_FORMULA (evaporation, cp_air); ``pres_temp`` uses q/q_sat
unclipped (sbc_phy.F90:275).  CORE-II humidity is negative on 1,201 ORCA1 wet
cells (tier-3 BLK D2) and 1,793 wet cells have q/q_sat outside [0, 2] at the
10-m pressure (D3); legoESM used the raw q and clipped the ratio.

Fixture: 200 D2 + 200 D3-only + 64 ordinary wet cells of the tier-3 BLK record
(provenance in ``fixtures/orca1_t3/blk_q_floor.json``).  Other open defects are
isolated per campaign vote C6: NEMO's ``wndm`` feeds the coefficients (D8),
NEMO's ``rhoa`` is injected into the fluxes (D4), and latent heat is scored as
evaporation (D5 latent-heat slope).  Tolerance = max(1e-12, NEMO 20-seed wet
spread), vote C2.  The floors are smooth in production (C1 blend over 1 % of
the floor); the oracle runs the hard NEMO form (vote B3).
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean import bulk_flux_omip as bf

jax.config.update("jax_enable_x64", True)

_FIX = Path(__file__).parent / "fixtures" / "orca1_t3" / "blk_q_floor"


@pytest.fixture(scope="module")
def fx():
    set_policy(PrecisionPolicy.fp64())
    data = dict(np.load(str(_FIX) + ".npz"))
    data["spread"] = json.loads(Path(str(_FIX) + ".json").read_text())["spread_wet"]
    return data


def _rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), "non-finite operand"
    den = np.maximum(np.abs(a), np.abs(b))
    return np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0)


def _port(fx, **kw):
    T_sfc = jnp.asarray(fx["sst_m"] + fx["rt0"])
    q = jnp.asarray(fx["humi"])
    slp = jnp.asarray(fx["slp"])
    ssq = bf.seawater_q_sat(T_sfc, slp)
    theta_air, p10 = bf.potential_air_temperature_10m(
        jnp.asarray(fx["tair"]), q, slp)
    Cd, Ch, Ce, _ = bf.ncar_transfer_coefficients(
        theta_air, q, bf.exner_potential_temperature(T_sfc, slp), ssq,
        jnp.asarray(fx["wndm"]), **kw)
    _, _, sh, _, ev = bf.air_sea_fluxes(
        jnp.asarray(fx["wndi"]), jnp.asarray(fx["wndj"]),
        jnp.asarray(fx["tair"]), q, T_sfc, rho_air=jnp.asarray(fx["rhoa"]),
        slp_Pa=slp, **kw)
    return dict(Cd=Cd, Ch=Ch, Ce=Ce, psen=sh, pevp=ev,
                theta_air_zt=theta_air, zpre_zt=p10)


@pytest.mark.parametrize("width", [0.0, None])
def test_bulk_matches_nemo_on_negative_and_supersaturated_humidity(
        fx, width):
    assert int(fx["d2"].sum()) == 200 and int((fx["d3"] & ~fx["d2"]).sum()) == 200
    # width 0.0 = NEMO's hard MAX (vote B3); None = production default.
    got = _port(fx, **({} if width is None else {"zu_floor_rel_width": width}))
    groups = (("d2", fx["d2"]), ("d3_only", fx["d3"] & ~fx["d2"]),
              ("ordinary", ~fx["d2"] & ~fx["d3"]))
    for name, val in got.items():
        tol = max(1e-12, fx["spread"][name])
        err = _rel(val, fx[name])
        for label, m in groups:
            assert err[m].max() <= tol, (name, label, float(err[m].max()), tol)


@pytest.mark.parametrize("which", [1, 0], ids=["q", "theta"])
def test_zu_floors_are_c1_and_exact_outside_band(which):
    """Both floors (q 1e-6, theta 180 K): exact outside [f, f + w], C1 blend
    inside whose shortfall below MAX is at most 4w/27, finite gradients that
    match finite differences inside the band.  No ORCA1 cell falls inside
    either band, so this synthetic case is what covers the blend."""
    floor = bf._Q_ZU_FLOOR_KG_KG if which == 1 else bf._T_ZU_FLOOR_K
    w = bf._ZU_FLOOR_REL_WIDTH * floor
    xs = jnp.asarray([floor - 10 * w, floor - 0.5 * w, floor + w / 3,
                      floor + 0.5 * w, floor + w, floor + 1e3 * w])

    def f(x):
        args = (x, 1e-3) if which == 0 else (280.0, x)
        return bf.nemo_zu_air_state(*args)[which]

    y = np.asarray(jax.vmap(f)(xs))
    hard = np.maximum(np.asarray(xs), floor)
    assert y[0] == floor and y[1] == floor
    assert y[4] == float(xs[4]) and y[5] == float(xs[5])
    gap = hard - y
    assert np.all(gap >= 0.0) and gap.max() <= 4.0 * w / 27.0 * (1 + 1e-12)
    g = np.asarray(jax.vmap(jax.grad(f))(xs))
    assert np.all(np.isfinite(g))
    assert g[0] == 0.0 and g[5] == 1.0 and g[2] > 0.0 and g[3] > 0.0
    for k in (2, 3):
        x0, h = float(xs[k]), 1e-4 * w
        fd = (float(f(jnp.asarray(x0 + h))) - float(f(jnp.asarray(x0 - h)))) / (2 * h)
        assert abs(g[k] / fd - 1.0) < 1e-4


def test_flux_gradient_finite_through_negative_humidity(fx):
    q = jnp.asarray(fx["humi"][fx["d2"]][:8])

    def evap_sum(qq):
        n = qq.shape[0]
        return jnp.sum(bf.air_sea_fluxes(
            jnp.asarray(fx["wndi"][fx["d2"]][:n]),
            jnp.asarray(fx["wndj"][fx["d2"]][:n]),
            jnp.asarray(fx["tair"][fx["d2"]][:n]), qq,
            jnp.asarray(fx["sst_m"][fx["d2"]][:n] + fx["rt0"]),
            slp_Pa=jnp.asarray(fx["slp"][fx["d2"]][:n]))[4])

    assert np.all(np.isfinite(np.asarray(jax.grad(evap_sum)(q))))


def test_rel_rejects_non_finite():
    with pytest.raises(AssertionError):
        _rel([np.nan], [1.0])
