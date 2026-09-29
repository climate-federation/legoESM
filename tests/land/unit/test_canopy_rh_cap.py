"""Smooth cap on canopy-air relative humidity (CanopyConfig.rh_cap_smoothing_width).

The hard ``clip(e/e_sat, 0, 1)`` put a kink on the Ball-Berry gs -> Ci rows of the
two-leaf canopy Newton solve wherever the canopy air saturates; the captured
production column below stalls with the (near-)hard cap and converges with the
smooth one.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.land.canopy.config import CanopyConfig
from legoesm.land.canopy.energy_balance import (
    canopy_met_variables,
    saturation_specific_humidity,
)
from legoesm.land.canopy.solver import CanopyForcingBundle, solve_canopy_closure

jax.config.update("jax_enable_x64", True)

_W = CanopyConfig().rh_cap_smoothing_width


def _rh(r, w=_W):
    Ps, Tc = jnp.asarray(1.0e5), jnp.asarray(290.0)
    e_target = r * canopy_met_variables(Ps, Tc, jnp.asarray(0.0), w)[1]
    q = constants.epsilon * e_target / (Ps - (1.0 - constants.epsilon) * e_target)
    e, es, _, rh, _, _, _ = canopy_met_variables(Ps, Tc, q, w)
    return float(e / es), float(rh)


def test_cap_is_smooth_bounded_and_close_below_saturation():
    bias = lambda r: _W * np.log1p(np.exp((r - 1.0) / _W))
    for r in (0.5, 0.9, 0.97, 1.0, 1.1):
        r_act, rh = _rh(r)
        assert abs(r_act - r) < 1e-9
        assert abs(rh - (r - bias(r))) < 1e-9
    assert 1.0 - 2 * _W * np.exp(-0.1 / _W) < _rh(1.1)[1] <= 1.0
    # derivative finite and continuous through saturation (no kink)
    Ps, Tc = jnp.asarray(1.0e5), jnp.asarray(290.0)
    q_sat = saturation_specific_humidity(Tc, Ps)
    d = jax.grad(lambda q: canopy_met_variables(Ps, Tc, q, _W)[3])
    lo, hi = float(d(q_sat * (1 - 1e-7))), float(d(q_sat * (1 + 1e-7)))
    assert np.isfinite(lo) and abs(lo - hi) < 1e-3 * abs(lo)


def test_validate_rejects_nonpositive_width():
    with pytest.raises(ValueError, match="rh_cap_smoothing_width"):
        CanopyConfig(rh_cap_smoothing_width=0.0).validate()


# Captured production AMIP column (2026-09-28, 40962-cell mesh), saturated
# canopy air near sunset; stalled at the 60-iteration cap under the hard cap.
_X0 = [286.7324534221029, 285.91839047684766, 349.6037339465827,
       363.5548400063025, 285.92326622780405, 0.00932154719930254]
_B = {'LAI': 0.40749812420560283, 'SZA': 76.43014401553582, 'La': 310.84686279296875,
      'epsf': 0.97, 'epss': 0.96, 'fSun': 0.1699933261968924,
      'APAR_Sun': 52.868335476925516, 'APAR_Sh': 11.19826890377048,
      'Vcmax25_Sun': 13.722414005372064, 'Vcmax25_Sh': 10.875746811610094,
      'Vcmax25_C4Sun': 0.0, 'Vcmax25_C4Sh': 0.0, 'ASW_Sun': 16.92302116909803,
      'ASW_Sh': 3.5918737810162598, 'ASW_Soil': 86.99476011602367,
      'Ts_bc': 285.96219736403395, 'Ca': 415.0, 'Ps': 99054.95187514076,
      'Ta': 286.3225997932251, 'lam': constants.L_v, 'Cp': constants.c_pd,
      'rhoa': 1.2077837452382452, 'Tv_atm': 287.9434580821958,
      'q_atm': 0.00931431884733052, 'm': 7.758096983671923, 'b0': 0.00862010775963547,
      'alf': 0.3, 'TgC': 25.0, 'fC4': 0.0, 'fStress_soil': 0.9989968562130214,
      'ur': 8.760683855774609, 'CI': 0.75, 'z0m': 0.02000784803858193,
      'displa': 0.13160440110760768, 'z0': 62.489969056588215, 'cv': 0.0135,
      'd_leaf': 0.025, 'r_soil_surface': 469.3715799953223, 'fwet': 0.0}


def test_saturated_canopy_column_converges_only_with_the_smooth_cap():
    b = CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in _B.items()})
    x0 = jnp.asarray(_X0)
    _, n, conv = solve_canopy_closure(x0, b, CanopyConfig())
    assert bool(conv) and int(n) < CanopyConfig().max_iters
    _, _, conv_hard = solve_canopy_closure(
        x0, b, CanopyConfig(rh_cap_smoothing_width=1e-7))
    assert not bool(conv_hard)


def test_width_reaches_both_the_solve_and_the_reported_fluxes():
    from legoesm.land.canopy.solver import canopy_forward
    b = CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in _B.items()})
    x0 = jnp.asarray(_X0)
    c1, c5 = CanopyConfig(), CanopyConfig(rh_cap_smoothing_width=0.05)
    x1, _, ok1 = solve_canopy_closure(x0, b, c1)
    x5, _, ok5 = solve_canopy_closure(x0, b, c5)
    assert bool(ok1) and bool(ok5)
    assert float(jnp.max(jnp.abs(x1 - x5))) > 1e-6
    fwd = lambda x, c: canopy_forward(x, b, c.LE_module, c.stomatal_model, c.le_cap_mode,
                                      c.use_ta_for_photosynthesis, c.rh_cap_smoothing_width, c.zeta_cap_smoothing_width)
    assert float(fwd(x1, c1)["LE_Sun"]) != float(fwd(x1, c5)["LE_Sun"])


def test_solver_refuses_zero_width_without_validate():
    b = CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in _B.items()})
    with pytest.raises(ValueError, match="rh_cap_smoothing_width"):
        solve_canopy_closure(jnp.asarray(_X0), b, CanopyConfig(rh_cap_smoothing_width=0.0))


def test_solver_refuses_width_above_the_validated_range():
    b = CanopyForcingBundle(**{k: jnp.asarray(v) for k, v in _B.items()})
    with pytest.raises(ValueError, match="rh_cap_smoothing_width"):
        solve_canopy_closure(jnp.asarray(_X0), b, CanopyConfig(rh_cap_smoothing_width=1.0))
