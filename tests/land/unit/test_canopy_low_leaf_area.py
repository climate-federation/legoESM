"""The canopy must stay well posed as the leaf area goes to zero.

144 of the 2562 columns in a coupled AMIP run carry a winter leaf area below
0.05 while the global parameter builder declares them fully vegetated. Two
things made that regime unsolvable, and both are pinned here.

**Ground-reflected radiation must vanish with the leaves.** The shortwave
transfer returns part of the ground-intercepted radiation upward into the
canopy, where leaves absorb it. That term was scaled by a per-column
``FNonVeg`` field which two builders filled incompatibly — the flux-tower
builder with the canopy gap fraction, the global builder with a BINARY
dominant-plant-type flag. At leaf area 0.019 those disagree by 0.99, and the
binary value kept an order-one heat source on a canopy with no leaves to hold
it. The radiation now derives the cover from the leaf area and clumping it
already receives, so the two cannot disagree.

The sunlit-leaf anchor is deliberately left triggering on the sunlit FRACTION:
retargeting it to the sunlit leaf area was tried and reverted, because it
replaces the residual of the leaf that still has area with equality to the empty
one. The test below pins that a normal canopy keeps its two-leaf split.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land.canopy.config import CanopyConfig
from legoesm.land.canopy.radiative_transfer import canopy_cover, canopy_shortwave_rt
from legoesm.land.canopy.solver import CanopyForcingBundle, solve_canopy_closure


# --------------------------------------------------------------------------- #
# canopy cover                                                                 #
# --------------------------------------------------------------------------- #

def test_cover_vanishes_with_the_leaf_area():
    assert float(canopy_cover(jnp.array(0.0), jnp.array(0.75))) == 0.0
    # Small-leaf-area limit is linear: cover -> 0.5 * CI * LAI.
    for lai in (1e-4, 1e-3, 1e-2):
        got = float(canopy_cover(jnp.array(lai), jnp.array(0.75)))
        assert got == pytest.approx(0.5 * 0.75 * lai, rel=1e-2)


def test_cover_saturates_for_a_dense_canopy():
    dense = float(canopy_cover(jnp.array(8.0), jnp.array(1.0)))
    assert 0.95 < dense < 1.0


def test_cover_is_the_complement_of_the_gap_fraction():
    # The flux-tower builder's gap fraction and this cover are one quantity.
    for lai, ci in ((0.0194, 0.75), (1.5, 0.75), (5.0, 0.9)):
        gap = float(np.exp(-0.5 * ci * lai))
        # float32 default precision: 1e-12 was my error, not the code's.
        assert float(canopy_cover(jnp.array(lai), jnp.array(ci))) == \
            pytest.approx(1.0 - gap, rel=1e-6)


def test_ground_reflected_radiation_vanishes_on_a_leafless_column():
    """REGRESSION. With the binary cover this stayed order one at zero leaves."""
    def absorbed(lai):
        out = canopy_shortwave_rt(
            PAR_dir=jnp.array([200.0]), PAR_diff=jnp.array([60.0]),
            NIR_dir=jnp.array([200.0]), NIR_diff=jnp.array([60.0]),
            UV=jnp.array([10.0]), SZA=jnp.array([30.0]),
            LAI=jnp.array([lai]), CI=jnp.array([0.75]),
            ALB_VIS=jnp.array([0.12]), ALB_NIR=jnp.array([0.21]),
            Vcmax25_C3_leaf=jnp.array([78.2]), Vcmax25_C4_leaf=jnp.array([0.0]),
            kn=jnp.array([0.3]))
        return float(out.ASW_Sun[0] + out.ASW_Sh[0])

    leafless, leafy = absorbed(1e-6), absorbed(3.0)
    assert leafless < 1.0, (
        f"a canopy with no leaves absorbed {leafless:.1f} W/m2 of shortwave")
    assert leafy > 50.0, "a real canopy should absorb a lot"


# --------------------------------------------------------------------------- #
# the sunlit-leaf anchor                                                       #
# --------------------------------------------------------------------------- #

def _bundle(LAI, fSun):
    base = dict(
        LAI=LAI, SZA=30.0, La=340.0, epsf=0.97, epss=0.96, fSun=fSun,
        APAR_Sun=200.0, APAR_Sh=50.0, Vcmax25_Sun=40.0, Vcmax25_Sh=20.0,
        Vcmax25_C4Sun=0.0, Vcmax25_C4Sh=0.0, ASW_Sun=200.0, ASW_Sh=60.0,
        ASW_Soil=400.0, Ts_bc=290.0, Ca=400.0, Ps=95000.0, Ta=288.0,
        lam=2.45e6, Cp=1005.0, rhoa=1.15, Tv_atm=289.0, q_atm=0.006,
        m=9.0, b0=0.01, alf=0.3, TgC=20.0, fC4=0.0, fStress_soil=0.5,
        ur=2.0, CI=0.75, z0m=0.02, displa=0.01, z0=10.0, cv=0.0135,
        d_leaf=0.025, r_soil_surface=100.0, fwet=0.0)
    return CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in base.items()})


_X0 = jnp.array([288.0, 288.0, 280.0, 280.0, 288.0, 0.006])


def test_a_normal_canopy_keeps_its_two_leaf_split():
    """The anchor must stay OFF where the two-leaf partition is meaningful."""
    x, _n, converged = solve_canopy_closure(
        _X0, _bundle(LAI=3.0, fSun=0.5), CanopyConfig())
    assert bool(converged)
    assert abs(float(x[0]) - float(x[1])) > 1e-3, (
        "sunlit and shaded leaves collapsed on a normal canopy")
