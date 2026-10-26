"""Tests for the shared netCDF SCM-forcing IO primitives."""
from __future__ import annotations

import numpy as np
import pytest
from legoesm.atmosphere.forcing.scm.scm_forcing_io import (
    forcing_cadence,
    interp_profile_to_pressure,
    omega_to_w,
    profile_time_fn,
    scalar_time_fn,
)

from legoesm import constants


def test_interp_profile_sorts_masks_and_clamps():
    # Unsorted source with a NaN sample; np.interp clamps outside the range.
    p_source = np.array([90000.0, np.nan, 50000.0, 70000.0])
    values = np.array([5.0, 999.0, 1.0, 3.0])
    p_target = np.array([40000.0, 60000.0, 70000.0, 95000.0])
    out = interp_profile_to_pressure(p_target, p_source, values)
    # 60000 sits halfway between 50000->1 and 70000->3 => 2.0; ends clamp.
    assert out[1] == pytest.approx(2.0)
    assert out[2] == pytest.approx(3.0)
    assert out[0] == pytest.approx(1.0)  # below range -> clamp to min-pressure value
    assert out[3] == pytest.approx(5.0)  # above range -> clamp to max-pressure value


def test_interp_profile_requires_two_finite_samples():
    with pytest.raises(ValueError, match="two finite"):
        interp_profile_to_pressure(
            np.array([1.0, 2.0]), np.array([1.0, np.nan]), np.array([1.0, 2.0])
        )


def test_omega_to_w_sign_and_magnitude():
    # omega > 0 (subsidence, positive DOWN) => w < 0 (SCM w positive UP).
    T = np.full((1, 3), 300.0)
    p = np.array([100000.0, 90000.0, 80000.0])
    omega = np.full((1, 3), 1.0)  # Pa/s downward
    w = omega_to_w(omega, T, p, q_v=np.zeros((1, 3)))
    assert w.shape == (1, 3)
    assert np.all(w < 0.0)
    # rho = p/(R_d T) at the surface level (dry): w = -omega/(rho g).
    rho0 = 100000.0 / (constants.R_d * 300.0)
    assert w[0, 0] == pytest.approx(-1.0 / (rho0 * constants.g), rel=1e-3)


def test_omega_to_w_broadcasts_1d_profile_over_time():
    T = np.array([300.0, 290.0])          # (nz,) broadcast across time
    p = np.array([100000.0, 90000.0])
    omega = np.array([[1.0, 1.0], [2.0, 2.0]])  # (nt, nz)
    w = omega_to_w(omega, T, p)
    assert w.shape == (2, 2)
    # Second time has 2x omega => 2x downward speed.
    assert w[1, 0] == pytest.approx(2.0 * w[0, 0], rel=1e-6)


def test_forcing_cadence_picks_smallest_positive_step():
    assert forcing_cadence(np.array([0.0, 300.0, 900.0])) == pytest.approx(300.0)
    assert forcing_cadence(np.array([5.0])) == pytest.approx(300.0)  # fallback


def test_profile_and_scalar_time_fns_interpolate_linearly():
    time = np.array([0.0, 100.0])
    prof = np.array([[1.0, 10.0], [3.0, 30.0]])  # (nt, nz)
    fn = profile_time_fn(time, prof, np.float64)
    mid = np.asarray(fn(50.0))
    assert mid == pytest.approx([2.0, 20.0])

    sfn = scalar_time_fn(time, np.array([2.0, 4.0]))
    assert float(sfn(25.0)) == pytest.approx(2.5)
