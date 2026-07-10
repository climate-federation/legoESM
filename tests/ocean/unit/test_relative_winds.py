"""NEMO ln_crt_dwn relative-wind / current feedback (rn_vfac).

Exercises the REAL functions end to end:

* ``ocean.bulk_flux_omip.air_sea_fluxes`` -- the frame-agnostic kernel: with
  ``vfac > 0`` it subtracts ``vfac * (u_oce, v_oce)`` from the wind BEFORE the
  speed + stress bulk, and the stress uses the RELATIVE vector (not
  |rel| * wind_direction).  ``vfac = 0.0`` (default) is BYTE-IDENTICAL to the
  absolute-wind behaviour.
* the ``run_omip_core2`` CLI resolver ``_resolve_wind_vfac`` (``--relative-winds``
  / ``--wind-vfac``) + the geographic surface-current helper
  ``_surface_currents_geographic`` (frame consistency: identity on a regular
  lat-lon grid; cube rejected loudly).

Acceptance (spec HIGH-2): 5 m/s eastward wind over a 1 m/s eastward current with
vfac=1 -> stress computed from the 4 m/s RELATIVE wind (byte-identical to a
current-free 4 m/s run); vfac=0 -> 5 m/s (byte-identical to current-free 5 m/s);
zero current -> byte-identical for any vfac.
"""

from __future__ import annotations

import types

import jax

jax.config.update("jax_enable_x64", True)  # OMIP bulk runs float64

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm.ocean.bulk_flux_omip import air_sea_fluxes  # noqa: E402
from scripts.run.run_omip_core2 import (  # noqa: E402
    _resolve_wind_vfac,
    _surface_currents_geographic,
    _WIND_VFAC_RANGE,
)


# Fixed thermodynamic background (unstable: SST warmer than air) so the only
# thing that differs between paired calls is the effective wind VECTOR.
_T_AIR_K = jnp.asarray([286.0, 288.0, 290.0])
_Q_AIR = jnp.asarray([0.004, 0.006, 0.008])
_T_SFC_K = jnp.asarray([288.0, 290.5, 292.0])


def _flux(u10, v10, **kw):
    """Call the real NCAR air_sea_fluxes with the fixed thermo background."""
    u10 = jnp.broadcast_to(jnp.asarray(u10, dtype=jnp.float64), _T_AIR_K.shape)
    v10 = jnp.broadcast_to(jnp.asarray(v10, dtype=jnp.float64), _T_AIR_K.shape)
    out = air_sea_fluxes(
        u10=u10, v10=v10, T_air_K=_T_AIR_K, q_air=_Q_AIR, T_sfc_K=_T_SFC_K, **kw
    )
    return tuple(np.asarray(o) for o in out)   # (tau_x, tau_y, shflx, lhflx, evap)


def _assert_byte_identical(a, b):
    for name, (ai, bi) in zip(("tau_x", "tau_y", "shflx", "lhflx", "evap"),
                              zip(a, b)):
        np.testing.assert_array_equal(ai, bi, err_msg=f"{name} differs")


# --------------------------------------------------------------------------
# (a) 5 m/s eastward wind over a 1 m/s eastward current
# --------------------------------------------------------------------------
def test_vfac1_over_current_is_relative_4ms_byte_identical():
    """vfac=1: 5 m/s wind - 1 m/s current -> stress from the 4 m/s RELATIVE wind,
    byte-identical to a current-free 4 m/s run (the whole bulk sees 4, not 5)."""
    with_feedback = _flux(5.0, 0.0, u_oce=1.0, v_oce=0.0, vfac=1.0)
    absolute_4 = _flux(4.0, 0.0)
    _assert_byte_identical(with_feedback, absolute_4)
    # ... and it is NOT the current-free 5 m/s answer (the feedback did something).
    absolute_5 = _flux(5.0, 0.0)
    assert not np.allclose(with_feedback[0], absolute_5[0])
    # Stress magnitude is REDUCED by the co-flowing current (|tau(4)| < |tau(5)|).
    assert np.all(np.abs(with_feedback[0]) < np.abs(absolute_5[0]))


def test_vfac0_byte_identical_to_no_current():
    """vfac=0 with a current present -> absolute 5 m/s, byte-identical to the
    current-free call (the guarded default path, even with u_oce supplied)."""
    off = _flux(5.0, 0.0, u_oce=1.0, v_oce=0.0, vfac=0.0)
    absolute_5 = _flux(5.0, 0.0)
    _assert_byte_identical(off, absolute_5)


# --------------------------------------------------------------------------
# (b) zero current -> byte-identical for ANY vfac (subtraction is identity)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("vfac", [0.0, 0.3, 0.7, 1.0])
def test_zero_current_byte_identical_any_vfac(vfac):
    zero_cur = _flux(5.0, 2.0, u_oce=0.0, v_oce=0.0, vfac=vfac)
    absolute = _flux(5.0, 2.0)
    _assert_byte_identical(zero_cur, absolute)


# --------------------------------------------------------------------------
# Stress uses the RELATIVE VECTOR, not |rel| * absolute_wind_direction
# --------------------------------------------------------------------------
def test_stress_follows_relative_vector_not_wind_direction():
    """Wind purely eastward (5, 0), current purely northward (0, 3): the RELATIVE
    vector is (5, -3), so the stress MUST gain a southward (tau_y != 0) component
    matching a current-free (5, -3) run.  |rel| * wind_dir would leave tau_y = 0
    -- the classic slip this guards against."""
    cross = _flux(5.0, 0.0, u_oce=0.0, v_oce=3.0, vfac=1.0)
    relative_vec = _flux(5.0, -3.0)
    _assert_byte_identical(cross, relative_vec)
    assert np.all(np.abs(cross[1]) > 0.0)   # tau_y non-zero from the -3 relative v


def test_legacy_algo_also_relative():
    """The current feedback applies to the legacy ly09_2coeff stress too."""
    kw = dict(algo="ly09_2coeff", q_sfc=jnp.asarray([0.01, 0.012, 0.014]))
    fb = _flux(5.0, 0.0, u_oce=1.0, v_oce=0.0, vfac=1.0, **kw)
    ref = _flux(4.0, 0.0, **kw)
    _assert_byte_identical(fb, ref)


# --------------------------------------------------------------------------
# (c) CLI resolution / round-trip
# --------------------------------------------------------------------------
def test_resolve_default_is_absolute_wind():
    assert _resolve_wind_vfac(False, None) == 0.0


def test_resolve_relative_winds_shorthand():
    assert _resolve_wind_vfac(True, None) == 1.0


def test_resolve_explicit_vfac():
    assert _resolve_wind_vfac(False, 0.5) == 0.5


def test_resolve_argparse_roundtrip():
    """--wind-vfac / --relative-winds parse and resolve to the right fraction
    (mirrors the real arg types: store_true + float)."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--relative-winds", action="store_true")
    p.add_argument("--wind-vfac", type=float, default=None)
    for argv, expect in (
        (["--wind-vfac", "0.5"], 0.5),
        (["--relative-winds"], 1.0),
        (["--wind-vfac", "1.0", "--relative-winds"], 1.0),  # agreeing pair
        ([], 0.0),
    ):
        a = p.parse_args(argv)
        assert _resolve_wind_vfac(a.relative_winds, a.wind_vfac) == expect


@pytest.mark.parametrize("bad", [-0.1, 1.1, _WIND_VFAC_RANGE[1] + 0.5,
                                 float("nan"), float("inf")])
def test_resolve_rejects_out_of_range(bad):
    with pytest.raises(SystemExit):
        _resolve_wind_vfac(False, bad)


def test_resolve_rejects_conflicting_flags():
    with pytest.raises(SystemExit):
        _resolve_wind_vfac(True, 0.5)      # relative-winds says 1.0, vfac says 0.5


def test_resolve_accepts_range_endpoints():
    lo, hi = _WIND_VFAC_RANGE
    assert _resolve_wind_vfac(False, lo) == lo
    assert _resolve_wind_vfac(False, hi) == hi


# --------------------------------------------------------------------------
# Frame consistency: geographic surface-current helper
# --------------------------------------------------------------------------
def _fake_cgrid_state(u_face, v_face):
    return types.SimpleNamespace(
        u=types.SimpleNamespace(data=jnp.asarray(u_face)),
        v=types.SimpleNamespace(data=jnp.asarray(v_face)),
    )


def test_surface_currents_geographic_latlon_identity_and_centering():
    """Regular lat-lon grid (no cos_alpha_u): grid-i == east, grid-j == north,
    so the helper returns the T-centre face averages unrotated."""
    # n_lat=2, n_lon=3, n_z=1.  u on EW faces (2, 4, 1); v on NS faces (3, 3, 1).
    u_face = np.arange(2 * 4 * 1, dtype=np.float64).reshape(2, 4, 1)
    v_face = np.arange(3 * 3 * 1, dtype=np.float64).reshape(3, 3, 1) * 0.1
    state = _fake_cgrid_state(u_face, v_face)
    grid = types.SimpleNamespace()   # no cos_alpha_u -> identity
    u_c, v_c = _surface_currents_geographic(state, grid, "latlon")
    exp_u = 0.5 * (u_face[:, :-1, 0] + u_face[:, 1:, 0])
    exp_v = 0.5 * (v_face[:-1, :, 0] + v_face[1:, :, 0])
    np.testing.assert_array_equal(np.asarray(u_c), exp_u)
    np.testing.assert_array_equal(np.asarray(v_c), exp_v)
    assert u_c.shape == (2, 3) and v_c.shape == (2, 3)


def test_surface_currents_geographic_rejects_cube():
    """cubed_sphere is parked -> fail loud (never a silent absolute-wind
    fall-through)."""
    with pytest.raises(NotImplementedError):
        _surface_currents_geographic(None, None, "cubed_sphere")
