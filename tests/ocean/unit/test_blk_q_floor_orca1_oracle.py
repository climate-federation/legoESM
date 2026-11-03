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
    den = np.maximum(np.abs(a), np.abs(b))
    return np.where(den > 0, np.abs(a - b) / np.where(den > 0, den, 1.0), 0.0)


def _port(fx):
    T_sfc = jnp.asarray(fx["sst_m"] + fx["rt0"])
    q = jnp.asarray(fx["humi"])
    slp = jnp.asarray(fx["slp"])
    ssq = bf.seawater_q_sat(T_sfc, slp)
    theta_air, p10 = bf.potential_air_temperature_10m(
        jnp.asarray(fx["tair"]), q, slp)
    Cd, Ch, Ce, _ = bf.ncar_transfer_coefficients(
        theta_air, q, bf.exner_potential_temperature(T_sfc, slp), ssq,
        jnp.asarray(fx["wndm"]))
    _, _, sh, _, ev = bf.air_sea_fluxes(
        jnp.asarray(fx["wndi"]), jnp.asarray(fx["wndj"]),
        jnp.asarray(fx["tair"]), q, T_sfc, rho_air=jnp.asarray(fx["rhoa"]),
        slp_Pa=slp)
    return dict(Cd=Cd, Ch=Ch, Ce=Ce, psen=sh, pevp=ev,
                theta_air_zt=theta_air, zpre_zt=p10)


@pytest.mark.parametrize("width", [0.0, None])
def test_bulk_matches_nemo_on_negative_and_supersaturated_humidity(
        fx, monkeypatch, width):
    assert int(fx["d2"].sum()) == 200 and int((fx["d3"] & ~fx["d2"]).sum()) == 200
    if width is not None:          # hard NEMO MAX (vote B3)
        monkeypatch.setattr(bf, "_ZU_FLOOR_REL_WIDTH", width)
    got = _port(fx)
    groups = (("d2", fx["d2"]), ("d3_only", fx["d3"] & ~fx["d2"]),
              ("ordinary", ~fx["d2"] & ~fx["d3"]))
    for name, val in got.items():
        tol = max(1e-12, fx["spread"][name])
        err = _rel(val, fx[name])
        for label, m in groups:
            assert err[m].max() <= tol, (name, label, float(err[m].max()), tol)


def test_zu_floor_is_c1_and_exact_outside_band():
    f = bf._Q_ZU_FLOOR_KG_KG
    w = bf._ZU_FLOOR_REL_WIDTH * f
    qs = jnp.asarray([-1e-5, f - 0.5 * w, f + 0.5 * w, f + w, 3e-3])
    _, q_zu = bf.nemo_zu_air_state(jnp.full_like(qs, 280.0), qs)
    q_zu = np.asarray(q_zu)
    assert q_zu[0] == f and q_zu[1] == f
    assert f < q_zu[2] < f + 0.5 * w
    assert q_zu[3] == float(qs[3]) and q_zu[4] == float(qs[4])
    g = jax.vmap(jax.grad(lambda q: bf.nemo_zu_air_state(280.0, q)[1]))(qs)
    g = np.asarray(g)
    assert np.all(np.isfinite(g))
    assert g[0] == 0.0 and g[4] == 1.0 and g[2] > 0.0   # C1 blend slope peaks at 4/3
    q0, h = f + 0.5 * w, 1e-3 * w
    fd = (float(bf.nemo_zu_air_state(280.0, q0 + h)[1])
          - float(bf.nemo_zu_air_state(280.0, q0 - h)[1])) / (2 * h)
    assert abs(g[2] / fd - 1.0) < 1e-4


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
