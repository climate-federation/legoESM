"""Sundqvist: GFS grid-area blend of psautco (R11) and the liquid/ice split of
precpd's condensate with the heat of fusion of the re-partition (R18)."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.atmosphere.physics.microphysics.sundqvist import (
    gfs_area_weight,
    liquid_fraction,
    split_condensate,
    sundqvist_microphysics,
)

jax.config.update("jax_enable_x64", True)

_TEM = 2.0 * np.pi ** 2 * constants.R_earth ** 2
_FINE = _TEM / (5000 * 2000)        # GFS's finest grid: wk1 -> 1
_COARSE = _TEM / (192 * 94)         # GFS's coarsest grid: wk1 -> 0
_K = SundqvistConfig().liquid_fraction_clamp_sharpness   # default clamp sharpness


def _cold_columns(ncol=2, nlev=10, q_i=0.0):
    T = jnp.linspace(215.0, 285.0, nlev)[None, :].repeat(ncol, axis=0)
    p_half = jnp.linspace(1e4, 1e5, nlev + 1)[None, :].repeat(ncol, axis=0)
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    from legoesm.thermo import saturation_specific_humidity
    q_v = 0.95 * saturation_specific_humidity(T, p_full)
    z = jnp.zeros((ncol, nlev))
    q_c = z.at[:, 2:8].set(1e-3)
    hyd = HydrometeorState(q_c=q_c, q_r=z, q_i=z.at[:, 2:5].set(q_i), q_s=z, q_g=z,
                           N_c=z, N_r=z, N_i=z)
    rho = p_full / (constants.R_d * T)
    dz = constants.R_d * T * (p_half[:, 1:] - p_half[:, :-1]) / (constants.g * p_full)
    return T, q_v, hyd, p_full, p_half, rho, dz


def test_area_weight_is_the_gfs_clip_and_differentiable_inside():
    """wk1 is GFS's hard clip (grid geometry, not traced state): exactly 1 at
    and finer than the finest grid, 0 at and coarser than the coarsest."""
    w = gfs_area_weight(jnp.array([_FINE / 2, _COARSE * 2, np.sqrt(_FINE * _COARSE)]))
    assert float(w[0]) == 1.0 and float(w[1]) == 0.0
    assert float(w[2]) == pytest.approx(0.5, abs=1e-12)
    for a in (1.0e10, 4.0e10, 1.0e9):
        x = jnp.log(jnp.asarray(a))
        f = lambda la: gfs_area_weight(jnp.exp(la))   # noqa: E731
        check_grads(f, (x,), order=1, modes=("rev",))
        g = float(jax.grad(f)(x))
        assert np.isfinite(g) and g != 0.0, (a, g)


def test_liquid_fraction_ramp_is_differentiable_inside_and_at_the_ends():
    """RULE 5 gradient test at the default sharpness, incl. the clamp ends."""
    T0 = constants.T_freeze
    lf = lambda T: liquid_fraction(T, _K)   # noqa: E731
    for T in (T0 - 15.0, T0 - 7.5, T0, T0 - 3.0):
        x = jnp.asarray(T)
        check_grads(lf, (x,), order=1, modes=("rev",))
        g = float(jax.grad(lf)(x))
        assert np.isfinite(g) and g > 0.0, (T, g)
    assert float(lf(jnp.asarray(T0 - 7.5))) == pytest.approx(0.5, abs=1e-12)
    assert float(lf(jnp.asarray(T0 - 40.0))) < 1e-12
    assert float(lf(jnp.asarray(T0 + 20.0))) > 1.0 - 1e-12


# RULE 5 cases: temperatures [C] across and beyond the -15..0 C ramp.
_LF_CASES_C = (-20.0, -15.0, -14.9, -7.5, -0.1, 0.0, 5.0)
# |smooth - hard| per case at the default sharpness 100, measured 2026-10-29
# (#2251); a change to the smooth form or its default re-measures these.
_LF_DEFAULT_GAP = [3.330669073875464e-17, 0.006931471805599453, 0.004143700868521745, 0.0, 0.004143700868521671, 0.006931471805599498, 0.0]


def _hard_ramp(T_c):
    return np.clip((np.asarray(T_c) + 15.0) / 15.0, 0.0, 1.0)


def test_liquid_fraction_hard_limit_is_the_clipped_ramp():
    """RULE 5: sharpness = inf reproduces the user's hard-clipped linear ramp."""
    T = constants.T_freeze + jnp.asarray(_LF_CASES_C)
    got = np.asarray(liquid_fraction(T, math.inf))
    np.testing.assert_allclose(got, _hard_ramp(_LF_CASES_C), rtol=0.0, atol=1e-14)


def test_liquid_fraction_default_gap_per_case():
    """RULE 5: the default smooth clamp departs from the hard ramp by the
    recorded per-case gap."""
    T = constants.T_freeze + jnp.asarray(_LF_CASES_C)
    gap = np.abs(np.asarray(liquid_fraction(T, _K)) - _hard_ramp(_LF_CASES_C))
    np.testing.assert_allclose(gap, np.asarray(_LF_DEFAULT_GAP), rtol=1e-6, atol=1e-12)


def test_sundqvist_refuses_a_missing_cell_area():
    with pytest.raises(ValueError, match="cell area"):
        sundqvist_microphysics(*_cold_columns(), 300.0)


def test_area_moves_ice_autoconversion_on_cold_columns():
    args = _cold_columns()
    fine = sundqvist_microphysics(*args, 300.0, cell_area=_FINE)
    coarse = sundqvist_microphysics(*args, 300.0, cell_area=_COARSE)
    same = sundqvist_microphysics(*args, 300.0, SundqvistConfig(
        psautco_fine=5.0e-4, psautco_coarse=5.0e-4), cell_area=_FINE)
    assert float(jnp.max(jnp.abs(fine.dq_c_dt - coarse.dq_c_dt))) > 0.0
    np.testing.assert_allclose(np.asarray(coarse.dq_c_dt), np.asarray(same.dq_c_dt),
                               rtol=1e-2, atol=1e-12)


def _H(T, q_v, q_i, dp):
    """Column moist enthalpy c_pd T + L_v q_v - L_f q_i [J/m^2]."""
    return jnp.sum((constants.c_pd * T + constants.L_v * q_v
                    - constants.L_f * q_i) * dp, axis=1) / constants.g


def test_split_closes_moist_enthalpy_and_fails_without_the_heating():
    T, q_v, hyd, p_full, p_half, rho, dz = _cold_columns()
    dt = 300.0
    dp = p_half[:, 1:] - p_half[:, :-1]
    args = (T, q_v, hyd, p_full, p_half, rho, dz, dt)
    split = sundqvist_microphysics(*args, cell_area=1.0e10, carries_q_i=True)
    whole = sundqvist_microphysics(*args, cell_area=1.0e10, carries_q_i=False)
    q_i_split = hyd.q_i + split.dq_i_dt * dt
    assert float(jnp.max(q_i_split)) > 1e-5          # the split made ice
    # same precpd (q_i = 0 in): the split alone must not change H
    H_split = _H(T + split.dT_dt * dt, q_v + split.dq_v_dt * dt, q_i_split, dp)
    H_whole = _H(T + whole.dT_dt * dt, q_v + whole.dq_v_dt * dt, hyd.q_i, dp)
    scale = float(jnp.max(jnp.abs(H_whole)))
    assert float(jnp.max(jnp.abs(H_split - H_whole))) <= 1e-12 * scale
    # without the fusion heating the same comparison misses by L_f * ice
    H_noheat = _H(T + whole.dT_dt * dt, q_v + split.dq_v_dt * dt, q_i_split, dp)
    assert float(jnp.max(jnp.abs(H_noheat - H_whole))) > 1e3 * 1e-12 * scale


def test_split_condensate_conserves_mass_and_enthalpy_with_an_input_ice_share():
    T = jnp.linspace(250.0, 280.0, 7)
    cwm = jnp.full((7,), 2e-3)
    share = jnp.linspace(0.0, 1.0, 7)
    q_c, q_i, dT = split_condensate(T, cwm, share, _K)
    np.testing.assert_allclose(np.asarray(q_c + q_i), np.asarray(cwm), rtol=1e-15)
    resid = constants.c_pd * dT - constants.L_f * (q_i - share * cwm)
    assert float(jnp.max(jnp.abs(resid))) <= 1e-12 * constants.L_f * 2e-3


def test_lanes_without_q_i_keep_all_condensate_in_q_c():
    args = _cold_columns(q_i=5e-4)
    out = sundqvist_microphysics(*args, 300.0, cell_area=1.0e10, carries_q_i=False)
    assert float(jnp.max(jnp.abs(out.dq_i_dt))) == 0.0


def test_lane_kwargs_carry_the_grid_area_per_column():
    from legoesm.atmosphere.physics.microphysics.integration import sundqvist_lane_kwargs

    class G:
        grid_area = jnp.arange(1.0, 7.0).reshape(2, 3)
    kw = sundqvist_lane_kwargs("sundqvist", G(), 6, True)
    np.testing.assert_array_equal(np.asarray(kw["cell_area"]), np.arange(1.0, 7.0))
    assert kw["carries_q_i"] is True
    assert sundqvist_lane_kwargs("kessler", G(), 6, True) == {}
    assert sundqvist_lane_kwargs("sundqvist", object(), 6, False)["cell_area"] is None


def test_split_reads_the_end_of_step_temperature(monkeypatch):
    """The liquid/ice ramp reads the temperature after condensation heating and
    precipitation (user 2026-10-28), not the pre-step or precpd-only T."""
    import legoesm.atmosphere.physics.microphysics.sundqvist as sq
    seen = {}

    def spy(T_in, cwm, share, sharpness):
        out = split_condensate(T_in, cwm, share, sharpness)
        seen["T"], seen["dT_fus"] = T_in, out[2]
        return out

    monkeypatch.setattr(sq, "split_condensate", spy)
    T, q_v, hyd, p_full, p_half, rho, dz = _cold_columns()
    q_v = q_v * (1.3 / 0.95)       # supersaturated: condensation heats the step
    dt = 300.0
    args = (T, q_v, hyd, p_full, p_half, rho, dz, dt)
    cond = sq.diagnose_sundqvist_process_rates(*args, cell_area=1.0e10).condensation
    assert float(jnp.max(cond)) > 0.0   # end-of-step T differs from precpd's T
    out = sq.sundqvist_microphysics(*args, cell_area=1.0e10, carries_q_i=True)
    end_T = T + out.dT_dt * dt - seen["dT_fus"]
    np.testing.assert_allclose(np.asarray(seen["T"]), np.asarray(end_T),
                               rtol=1e-13, atol=0.0)
