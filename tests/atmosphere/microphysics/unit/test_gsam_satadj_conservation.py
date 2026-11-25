"""Water and energy budget of the gSAM liquid saturation adjustment.

Budgets touched: total water q_v + q_c (one exchange amount moves between the
two with opposite signs) and the latent heat that goes with it
(dT = L_v/c_pd * dq_c, i.e. c_pd*T + L_v*q_v is unchanged).
The fp32 closure of the Morrison tendencies built on this exchange is tested
in tests/unit/test_plane_mixed_precision.py.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics._warm_rain import saturation_adjustment_gsam
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.thermo import saturation_specific_humidity

pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64,
                                reason="needs JAX_ENABLE_X64=1")

GSAM_DEFAULT = dict(
    excess_sharpness=MorrisonConfig._field_defaults["gsam_satadj_excess_sharpness"],
    qc_floor_sharpness=MorrisonConfig._field_defaults["gsam_satadj_qc_floor_sharpness"],
    p_min_width_pa=MorrisonConfig._field_defaults["gsam_satadj_p_min_width_pa"])
GSAM_HARD = dict(excess_sharpness=math.inf, qc_floor_sharpness=math.inf,
                 p_min_width_pa=0.0)


def _states(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    T = rng.uniform(200.0, 310.0, n)
    p = rng.uniform(5.0e3, 1.0e5, n)
    qs = np.asarray(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p)))
    q_v = qs * rng.uniform(0.5, 1.3, n)
    q_c = np.where(rng.uniform(size=n) < 0.5, 0.0, rng.uniform(0.0, 1e-3, n))
    return [jnp.asarray(x) for x in (T, q_v, q_c, p)]


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("kw", [GSAM_DEFAULT, GSAM_HARD], ids=["smooth", "hard"])
def test_satadj_moves_water_and_heat_as_one_amount(kw, dtype):
    """Per cell, the water budget closes to 2 eps of the VAPOUR (measured
    1.9e-16 fp64 / 1.1e-7 fp32). Rebuilding q_v from q_v + q_c instead rounds
    at the total-water scale: 3.1e-14 fp64 / 9.0e-6 fp32 of the vapour."""
    T, q_v, q_c, p = (jnp.asarray(np.asarray(x), dtype) for x in _states())
    Tn, qvn, qcn = saturation_adjustment_gsam(T, q_v, q_c, p, **kw)
    assert qvn.dtype == qcn.dtype == Tn.dtype == dtype
    f = lambda a: np.asarray(a, np.float64)
    dqv, dqc, dT = f(qvn) - f(q_v), f(qcn) - f(q_c), f(Tn) - f(T)
    eps = np.finfo(dtype).eps
    # water: in - out - d(storage) = 0 per cell, to round-off of the vapour
    np.testing.assert_array_less(np.abs(dqv + dqc), 2 * eps * f(q_v))
    # energy: the heating is the latent heat of the moved water
    fac = constants.L_v / constants.c_pd
    np.testing.assert_array_less(np.abs(dT - fac * dqc), 8 * eps * f(T))
    assert np.abs(dqc).max() > 1e-6        # the adjustment actually acted
