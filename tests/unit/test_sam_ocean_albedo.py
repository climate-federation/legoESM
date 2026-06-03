"""SAM Briegleb ocean surface albedo tests (iter-17 RAD-3).

gSAM ``cam_rad_parameterizations.f90:albedo`` (ocean branch, ts>271):
    a_dir = 0.026/(μ^1.7 + 0.065) + 0.15·(μ−0.1)·(μ−0.5)·(μ−1.0)
paired with a fixed diffuse adif=0.07 (RCEMIP, AAW 2017). legoESM's RRTMGP
uses a single surface albedo, so the fixed-zenith DIRECT-beam RCE takes the
direct value (~0.033 at μ=0.7425) — replacing the flat 0.06 default.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.atmosphere.physics.radiation.integration import sam_ocean_albedo


jax.config.update("jax_enable_x64", True)


def _briegleb(mu):
    return 0.026 / (mu ** 1.7 + 0.065) + 0.15 * (mu - 0.1) * (mu - 0.5) * (mu - 1.0)


@pytest.mark.parametrize("mu", [1.0, 0.7425, 0.62, 0.5, 0.2])
def test_matches_briegleb_formula_over_ocean(mu):
    """Ice-free ocean (T_sfc>271): exact Briegleb direct-beam albedo."""
    assert float(sam_ocean_albedo(mu, 300.0)) == pytest.approx(
        _briegleb(mu), abs=1e-12)


def test_rcemip_zenith_value():
    """At the RCEMIP zenith (42.05°, μ=0.7425) the albedo is ~0.033 — much
    lower than the previous flat 0.06 default (≈11 W/m² more absorbed SW)."""
    a = float(sam_ocean_albedo(0.7425, 300.0))
    assert a == pytest.approx(0.0329, abs=1e-3)
    assert a < 0.06                        # lower than the old default


def test_sea_ice_branch():
    """T_sfc ≤ 271 K ⇒ sea-ice/snow direct albedo 0.75 (SAM albedo())."""
    assert float(sam_ocean_albedo(0.7425, 270.0)) == pytest.approx(0.75)
    # Just above the threshold stays ocean.
    assert float(sam_ocean_albedo(0.7425, 271.5)) < 0.1


def test_night_zero_albedo():
    """μ ≤ 0 (no sun) ⇒ zero albedo (matches SAM's coszrs≤0 branch)."""
    assert float(sam_ocean_albedo(-0.2, 300.0)) == 0.0
    assert float(sam_ocean_albedo(0.0, 300.0)) == 0.0


def test_albedo_rises_toward_grazing_sun():
    """Ocean albedo increases as the sun approaches the horizon (small μ) —
    the physical specular-reflection behaviour the flat 0.06 missed."""
    high_sun = float(sam_ocean_albedo(1.0, 300.0))
    low_sun = float(sam_ocean_albedo(0.2, 300.0))
    assert low_sun > high_sun
    assert high_sun == pytest.approx(0.0244, abs=1e-3)


def test_ad_safe_at_zero_zenith():
    """grad d(albedo)/dμ is finite at μ=0 (exponent 1.7>0 ⇒ no safe_pow)."""
    g = jax.grad(lambda m: sam_ocean_albedo(m, 300.0))(jnp.asarray(0.0))
    assert bool(jnp.isfinite(g))
    # and at the RCE point
    g2 = jax.grad(lambda m: sam_ocean_albedo(m, 300.0))(jnp.asarray(0.7425))
    assert bool(jnp.isfinite(g2))


def test_array_input_vectorized():
    """Vectorises over a per-column μ array (for future varying-zenith GATE)."""
    mu = jnp.asarray([0.2, 0.5, 0.7425, 1.0])
    out = sam_ocean_albedo(mu, 300.0)
    assert out.shape == (4,)
    assert jnp.all(jnp.isfinite(out))
