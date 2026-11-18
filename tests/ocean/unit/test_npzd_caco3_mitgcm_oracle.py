"""Implicit CaCO3 against COMPILED MITgcm pkg/dic (c3f71ac8).

Fixture ``oracle_fixtures/mitgcm_c3f71ac8_caco3.npz`` banks ``CAR_FLUX``
(``car_flux.F`` compiled from source) and the ``dic_init_fixed.F`` production
level count ``nlev`` on eight columns: full depth, sea floor at level 9, sea
floor inside the production levels, and a partial bottom cell
(``hFacC = 0.5``), each for two (``zca``, ``zcrit``) pairs.  Driver and votes:
``oracle_audit/shelf_npzd_bs_oracle``.

KNOWN DIFFERENCE (pinned per case): the port runs on the 1-D reference
thickness, so a partial bottom cell dissolves its share over the full cell
thickness; MITgcm divides by ``drF*hFacC``.  The rate there is ``hFacC`` times
MITgcm's (0.5 here).
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.biogeochemistry import npzd

jax.config.update("jax_enable_x64", True)

FIX = np.load(Path(__file__).parent / "oracle_fixtures" / "mitgcm_c3f71ac8_caco3.npz")
TOL, FLOOR = 1e-12, 1e-30
PIN_REL, PIN_ABS = 1e-6, 1e-12
CASES = range(len(FIX["names"]))


def _ours(j):
    zca, zcrit = FIX["zca_zcrit"][j]
    drf = jnp.asarray(FIX["drf"])
    prod = jnp.asarray(FIX["car_s"]) * npzd._production_levels(drf, zcrit)
    wet = jnp.asarray((FIX["hfac"][j] > 0).astype(float))
    return np.asarray(npzd.implicit_caco3_dissolution(prod, drf, wet, zca))


@pytest.mark.parametrize("j", CASES)
def test_production_levels_match_mitgcm_nlev(j):
    _, zcrit = FIX["zca_zcrit"][j]
    lev = np.asarray(npzd._production_levels(jnp.asarray(FIX["drf"]), zcrit))
    assert np.array_equal(lev, (np.arange(lev.size) < FIX["nlev"][j]).astype(float))


@pytest.mark.parametrize("j", [j for j in CASES if np.all(np.isin(FIX["hfac"][j], (0.0, 1.0)))])
def test_dissolution_matches_car_flux(j):
    ref = FIX["cflux"][j]
    got = _ours(j)
    assert np.array_equal(got != 0, ref != 0)
    assert (np.abs(got - ref) / np.maximum(np.abs(ref), FLOOR)).max() <= TOL


@pytest.mark.parametrize("j", CASES)
def test_dissolution_gap_pinned(j):
    ref = FIX["cflux"][j]
    got = _ours(j)
    gap = np.where(ref != 0, got / np.where(ref != 0, ref, 1.0) - 1.0, got)
    np.testing.assert_allclose(gap, FIX["pin_rel"][j], rtol=PIN_REL, atol=PIN_ABS)
