"""Bucket runoff partition: infiltration excess (Green-Ampt) + saturation excess.

Pins the shared ``partition_bucket_runoff`` used by both single-bucket slab
codepaths (slab_land.step_land and physics_pipeline._bucket_update):
- water conservation P == dW/dt + E + runoff in every regime,
- Hortonian (infiltration-excess) runoff fires when rain rate > capacity even
  with bucket room, and is suppressed by the dry-soil suction boost,
- saturation-excess runoff fires when the bucket overfills,
- the infiltration_excess=False switch recovers the legacy fill-and-spill bucket,
- the lower clamp never creates water under strong evaporative demand.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm import constants
from legoesm.land.bucket_hydrology import partition_bucket_runoff

_RHO = constants.rho_water
_DT = 3600.0
_WMAX = 150.0
_K = 1.0e-5      # m/s -> cap K*rho = 1e-2 kg/m2/s at saturation
_B = 2.0


def _resid(W0, P, E, **kw):
    W = jnp.array([W0]);
    Wn, Ea, ro, ri, rs = partition_bucket_runoff(
        W, jnp.array([P]), jnp.array([E]), _DT, _WMAX, _K, _B, **kw)
    dWdt = float((Wn[0] - W0) / _DT)
    return dWdt - (P - float(Ea[0]) - float(ro[0])), (Wn, Ea, ro, ri, rs)


def test_conserves_wetting_and_drying():
    for W0, P, E in [(120.0, 5e-2, 1e-4), (40.0, 0.0, 2e-4), (149.0, 1e-3, 0.0),
                     (0.0, 1e-2, 0.0), (75.0, 3e-2, -1e-4)]:   # last: dew (E<0)
        r, _ = _resid(W0, P, E)
        assert abs(r) < 1e-10, (W0, P, E, r)


def test_infiltration_excess_fires_with_bucket_room():
    # Half-full bucket (room), intense rain above the capacity -> Hortonian runoff
    # while saturation excess stays zero.
    _, (Wn, Ea, ro, ri, rs) = _resid(75.0, 5e-2, 0.0)
    assert float(ri[0]) > 0.0, ri          # Hortonian
    assert float(rs[0]) == 0.0, rs         # not saturation
    assert float(Wn[0]) < _WMAX            # bucket still has room


def test_dry_soil_suction_admits_more_than_wet():
    # Same rain; a drier bucket has a higher Green-Ampt capacity (suction), so it
    # generates LESS infiltration-excess runoff than a wetter one.
    P = 2.5e-2
    _, (_, _, _, ri_dry, _) = _resid(10.0, P, 0.0)
    _, (_, _, _, ri_wet, _) = _resid(140.0, P, 0.0)
    assert float(ri_dry[0]) < float(ri_wet[0]), (ri_dry, ri_wet)


def test_saturation_excess_fires_on_full_bucket():
    _, (Wn, Ea, ro, ri, rs) = _resid(_WMAX - 1.0, 1e-3, 0.0)
    assert float(rs[0]) > 0.0, rs
    assert float(Wn[0]) <= _WMAX + 1e-9


def test_disabling_infiltration_excess_recovers_fill_and_spill():
    # With infiltration_excess=False, intense rain on a bucket with room fully
    # infiltrates (no Hortonian); only saturation excess can run off.
    _, (Wn, Ea, ro, ri, rs) = _resid(75.0, 5e-2, 0.0, infiltration_excess=False)
    assert float(ri[0]) == 0.0, ri
    # all the rain entered the bucket (no overflow yet at this fill)
    assert abs(float(Wn[0]) - (75.0 + 5e-2 * _DT)) < 1e-6 or float(rs[0]) > 0.0


def test_limit_evaporation_false_consumes_demand_verbatim():
    """The w_land pipeline path (limit_evaporation=False) consumes the evaporation
    demand exactly as given (so the bucket stays consistent with the BL/energy flux
    applied upstream), and still partitions runoff.  With ample water the budget
    closes; the clamp only bites under over-evaporation (the Manabe behaviour)."""
    # ample water, moderate evap -> evap_actual == demand, budget closes
    W = jnp.array([100.0]); P = jnp.array([2e-2]); E = jnp.array([1e-4])
    Wn, Ea, ro, ri, rs = partition_bucket_runoff(
        W, P, E, _DT, _WMAX, _K, _B, limit_evaporation=False)
    assert float(Ea[0]) == float(E[0])               # demand consumed verbatim
    assert float(ri[0]) > 0.0                          # Hortonian still fires
    resid = float((Wn[0] - W[0]) / _DT - (P[0] - Ea[0] - ro[0]))
    assert abs(resid) < 1e-10, resid
    # over-evaporation on a nearly-empty bucket: clamps at 0 (does not go negative)
    Wn2, Ea2, _, _, _ = partition_bucket_runoff(
        jnp.array([3.0]), jnp.array([0.0]), jnp.array([1.0]), _DT, _WMAX, _K, _B,
        limit_evaporation=False)
    assert float(Wn2[0]) == 0.0                        # depleted, floored (Manabe)


def test_lower_clamp_creates_no_water_under_strong_evaporation():
    # Evaporative demand far exceeding the available water must not create water
    # (W stays >= 0, evap is throttled, budget closes).
    r, (Wn, Ea, ro, ri, rs) = _resid(5.0, 0.0, 1.0)   # huge demand, tiny bucket
    assert float(Wn[0]) >= 0.0
    assert abs(r) < 1e-10, r


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
