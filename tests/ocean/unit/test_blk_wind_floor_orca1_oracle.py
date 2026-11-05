"""ORCA1 oracle: the NEMO-parity neutral drag keeps NEMO's 0.25 m/s neutral-wind floor.

NEMO 5.0.1 ``turb_ncar`` floors the bulk wind at 0.5 m/s for the first guess
(sbcblk_algo_ncar.F90:115) and the iterated neutral 10-m wind at 0.25 m/s
(:192); ``cd_n10_ncar`` itself has no floor (:242-255).  legoESM's
``large_yeager_neutral_cd(nemo_parity=True)`` re-floored its argument at
0.5 m/s, overriding the 0.25 floor on calm cells (tier-3 BLK D1: Cd 86 % off).

Fixture: the 67 wet cells of the tier-3 BLK record whose NEMO iterated neutral
wind drops below 0.5 m/s, plus 64 seeded ordinary wet cells (provenance in
``fixtures/orca1_t3/blk_wind_floor.json``).  The wind speed fed is NEMO's own
``wndm``: legoESM adds 1e-12 under the wind square root (finding D8, not this
defect), which moves calm-cell Cd by 1.8e-11 (campaign vote C6).  Tolerance =
max(1e-12, NEMO's 20-seed wet spread), campaign vote C2.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.bulk_flux import large_yeager_neutral_cd
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean import bulk_flux_omip as bf

jax.config.update("jax_enable_x64", True)

_FIX = Path(__file__).parent / "fixtures" / "orca1_t3" / "blk_wind_floor"


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


def _coefficients(fx):
    """Production preprocessing of ``air_sea_fluxes`` (NCAR branch) + the
    transfer-coefficient solve, on NEMO's recorded inputs."""
    T_sfc = jnp.asarray(fx["sst_m"] + fx["rt0"])
    q = jnp.asarray(fx["humi"])
    slp = jnp.asarray(fx["slp"])
    ssq = bf.seawater_q_sat(T_sfc, slp)
    theta_air, _ = bf.potential_air_temperature_10m(
        jnp.asarray(fx["tair"]), q, slp)
    theta_sst = bf.exner_potential_temperature(T_sfc, slp)
    return bf.ncar_transfer_coefficients(
        theta_air, q, theta_sst, ssq, jnp.asarray(fx["wndm"]))


@pytest.mark.parametrize("name, idx", [("Cd", 0), ("Ch", 1), ("Ce", 2)])
def test_transfer_coefficients_match_nemo(fx, name, idx):
    assert int(fx["calm"].sum()) == 67
    got = np.asarray(_coefficients(fx)[idx])
    tol = max(1e-12, fx["spread"][name])
    err = _rel(got, fx[name])
    for label, m in (("calm", fx["calm"]), ("ordinary", ~fx["calm"])):
        assert err[m].max() <= tol, (name, label, float(err[m].max()), tol)


def test_parity_cd_has_no_internal_floor():
    # NEMO cd_n10_ncar at 0.3 m/s: the bare polynomial (no floor), 9.18e-3.
    U = 0.3
    want = 1e-3 * (2.7 / U + 0.142 + U / 13.09 - 3.14807e-10 * U ** 6)
    got = float(large_yeager_neutral_cd(jnp.asarray(U), nemo_parity=True))
    assert abs(got - want) <= 1e-15 * want
    # The default (MOST-solver) convention is unchanged: floored at 0.5 m/s,
    # then clipped to [0.5e-3, 3e-3].
    assert float(large_yeager_neutral_cd(jnp.asarray(U))) == 3.0e-3


def test_parity_cd_gradient_below_half_metre_per_second():
    f = lambda u: large_yeager_neutral_cd(u, nemo_parity=True)  # noqa: E731
    U, h = 0.3, 1e-6
    g = float(jax.grad(f)(jnp.asarray(U)))
    fd = (float(f(jnp.asarray(U + h))) - float(f(jnp.asarray(U - h)))) / (2 * h)
    assert np.isfinite(g) and g < 0.0
    assert abs(g / fd - 1.0) < 1e-6


def test_rel_rejects_non_finite():
    with pytest.raises(AssertionError):
        _rel([np.nan], [1.0])
