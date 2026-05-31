"""SAM MLS ozone profile tests (iter-19 RAD-4 O3).

The bundled ``mls_ozone_vmr`` reproduces gSAM's MLS standard O3 profile
(from ``rrtmg_lw.nc``), interpolated log-log to the model levels — the same
ozone the oracle uses. It replaces legoESM's built-in skewed-Gaussian, which
over-estimates lower-stratospheric O3 by ~3x.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.radiation.ozone_mls import (
    mls_ozone_vmr,
    _MLS_P_HPA,
    _MLS_O3_VMR,
)
from legoesm.atmosphere.physics.radiation.integration import _compute_ozone_vmr
from legoesm.atmosphere.physics.radiation.config import OzoneProfileConfig


jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("p_hPa,o3", [
    (1053.63, 1.7351e-08),   # surface
    (52.457, 1.4865e-06),    # lower stratosphere
    (8.671, 9.9645e-06),     # near peak
    (7.099, 9.8541e-06),
    (0.961, 2.7350e-06),     # upper stratosphere
])
def test_reproduces_mls_table_points(p_hPa, o3):
    """At the tabulated pressures the interpolant returns the MLS VMR."""
    got = float(mls_ozone_vmr(jnp.asarray(p_hPa * 100.0)))
    assert got == pytest.approx(o3, rel=1e-3)


def test_lower_stratosphere_not_overestimated():
    """The crux of the fix: at 52 hPa MLS O3 ≈ 1.49 ppm, NOT the built-in
    Gaussian's ~4.9 ppm (a ~3x over-estimate)."""
    o3_52 = float(mls_ozone_vmr(jnp.asarray(52.457 * 100.0)))
    assert o3_52 == pytest.approx(1.4865e-06, rel=1e-3)
    assert o3_52 < 2.0e-6                        # well below the Gaussian


def test_peak_magnitude_and_location():
    """O3 peaks near 8-9 hPa at ~10 ppm (MLS), not the Gaussian's 10 hPa."""
    p = jnp.asarray(_MLS_P_HPA) * 100.0
    o3 = mls_ozone_vmr(p)
    assert float(jnp.max(o3)) == pytest.approx(9.9645e-06, rel=1e-3)
    p_at_peak = float(jnp.asarray(_MLS_P_HPA)[int(jnp.argmax(o3))])
    assert 6.0 < p_at_peak < 12.0                # hPa


def test_interpolated_midpoint_between_neighbours():
    """A pressure between two table levels yields an O3 strictly between the
    neighbouring tabulated values (monotone log-log interp)."""
    # 58 hPa is between 64.072 (8.64e-7) and 52.457 (1.49e-6).
    o3 = float(mls_ozone_vmr(jnp.asarray(58.0 * 100.0)))
    assert 8.6365e-07 < o3 < 1.4865e-06


def test_clamps_outside_table():
    """Below the surface table pressure / above the top, clamp to endpoints
    (no extrapolation blow-up)."""
    deep = float(mls_ozone_vmr(jnp.asarray(1.1e5)))      # 1100 hPa > table max
    high = float(mls_ozone_vmr(jnp.asarray(0.05)))       # 5e-4 hPa < table min
    assert deep == pytest.approx(_MLS_O3_VMR[0], rel=1e-6)
    assert high == pytest.approx(_MLS_O3_VMR[-1], rel=1e-6)


def test_ad_finite():
    """jax.grad of the column O3 wrt pressure is finite (log-clamped p +
    piecewise-linear interp)."""
    p = jnp.asarray([5.0e3, 5.0e4, 1.0e5])
    g = jax.grad(lambda pp: jnp.sum(mls_ozone_vmr(pp)))(p)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_dispatch_through_compute_ozone_vmr():
    """source='mls' routes through the radiation ozone dispatcher and returns
    the (ncol, nlev) MLS field; an unknown source still raises."""
    p_full = jnp.asarray([[5.0e4, 2.0e4, 5.0e3]])     # (1, 3)
    lat = jnp.asarray([0.0])
    out = _compute_ozone_vmr(p_full, lat, OzoneProfileConfig(source="mls"))
    assert out is not None and out.shape == (1, 3)
    assert bool(jnp.all(out > 0.0))
    with pytest.raises(ValueError, match="Unknown OzoneProfileConfig.source"):
        _compute_ozone_vmr(p_full, lat, OzoneProfileConfig(source="typo"))
