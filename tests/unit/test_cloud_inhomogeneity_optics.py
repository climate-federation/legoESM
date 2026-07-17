"""Sub-grid cloud-optics inhomogeneity factor (two-region / Tripleclouds).

Pins the closed-form ``chi_eff = 1 - fsd^2 tau/(gamma0+tau)`` properties -- a
TAU-DEPENDENT reduction (a thick cloud reduced more than a thin one, unlike a
constant Cahalan scalar), bounded by the asymptote ``1 - fsd^2``:

* homogeneous limits (``fsd -> 0`` and ``tau -> 0``) give ``chi_eff = 1``;
* ``chi_eff < 1`` for any real cloud with ``fsd > 0``;
* ``chi_eff`` DECREASES monotonically as ``tau`` grows -- a thick layer is
  reduced MORE (toward the ``1 - fsd^2`` floor);
* ``chi_eff`` decreases as ``fsd`` grows (more inhomogeneity => more reduction);
* the closed form matches the two-stream reflectance-inversion hand value;
* the factor is ``jax.grad``-safe in both ``tau`` and ``fsd`` (no clip / zero
  guard / division-by-near-zero -- the closed form is analytically bounded).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    _two_region_inhomogeneity_factor as chi_eff,
)


def test_homogeneous_limits_give_unity():
    tau = jnp.array([1.0, 10.0, 100.0, 500.0])
    # fsd = 0 => a single homogeneous region => no reduction.
    assert bool(jnp.allclose(chi_eff(tau, 0.0, 0.85), 1.0, atol=1e-6))
    # tau -> 0 => chi_eff -> 1 (thin homogeneous cloud), any fsd.  The closed
    # form approaches 1 smoothly (1 - O(tau)); it is not bit-exactly 1 at a
    # tiny-but-nonzero tau, which is correct (no zero-guard/where).
    assert abs(float(chi_eff(jnp.array([1e-10]), 0.75, 0.85)[0]) - 1.0) < 1e-8
    assert abs(float(chi_eff(jnp.array([1e-3]), 0.75, 0.85)[0]) - 1.0) < 1e-2


def test_reduction_below_one_for_real_cloud():
    val = float(chi_eff(jnp.array([50.0]), 0.75, 0.85)[0])
    assert 0.0 < val < 1.0


def test_chi_eff_decreases_with_tau_breaking_saturation():
    """THE property: a thicker cloud is reduced MORE (constant chi cannot)."""
    tau = jnp.array([1.0, 5.0, 20.0, 50.0, 100.0, 300.0])
    vals = chi_eff(tau, 0.75, 0.85)
    d = jnp.diff(vals)
    assert bool(jnp.all(d < 0.0)), f"chi_eff not strictly decreasing in tau: {vals}"


def test_chi_eff_decreases_with_fsd():
    tau = jnp.array([100.0])
    fsds = jnp.array([0.2, 0.5, 0.75, 0.95])
    vals = jnp.array([float(chi_eff(tau, float(f), 0.85)[0]) for f in fsds])
    assert bool(jnp.all(jnp.diff(vals) < 0.0)), f"not decreasing in fsd: {vals}"


def test_matches_two_stream_inversion_hand_value():
    # tau=100, fsd=0.75, g=0.85 => gamma0=13.333, tau_thin=25, tau_thick=175,
    # R_bar=0.5(0.6522+0.9292)=0.7907, tau_eff=13.333*0.7907/0.2093=50.37,
    # chi_eff=0.5037.
    val = float(chi_eff(jnp.array([100.0]), 0.75, 0.85)[0])
    assert abs(val - 0.5037) < 2e-3, val


def test_differentiable_in_tau_and_fsd():
    g_tau = jax.grad(lambda t: chi_eff(jnp.array([t]), 0.75, 0.85)[0])(100.0)
    g_fsd = jax.grad(lambda f: chi_eff(jnp.array([100.0]), f, 0.85)[0])(0.75)
    assert bool(jnp.isfinite(g_tau)) and bool(jnp.isfinite(g_fsd))
    # thicker => smaller chi_eff => d(chi)/d(tau) < 0; more fsd => smaller => <0.
    assert float(g_tau) < 0.0 and float(g_fsd) < 0.0


def test_bounded_and_fp32_safe_at_large_tau():
    """Closed form is analytically in [1-fsd^2, 1]; a huge tau must NOT
    overflow/NaN (the earlier inversion clipped a spurious >1 down to 1)."""
    for fsd in (0.5, 0.75, 0.95):
        big = chi_eff(jnp.array([1.0e10]), fsd, 0.85)[0]
        assert bool(jnp.isfinite(big))
        assert abs(float(big) - (1.0 - fsd * fsd)) < 1e-3   # -> asymptote 1-fsd^2
    tau = jnp.array([0.0, 1e-6, 1.0, 1e3, 1e10])
    v = chi_eff(tau, 0.75, 0.85)
    assert bool(jnp.all(jnp.isfinite(v)))
    assert bool(jnp.all(v <= 1.0 + 1e-9))                   # never brightens
    assert bool(jnp.all(v >= (1.0 - 0.75 * 0.75) - 1e-9))   # floor 1-fsd^2


# --- end-to-end wiring through compute_cloud_properties ---------------------

import numpy as np                                           # noqa: E402
import pytest                                                # noqa: E402
from legoesm.atmosphere.physics.clouds.cloud_fraction import (  # noqa: E402
    compute_cloud_properties,
)
from legoesm.atmosphere.physics.clouds.config import CloudConfig  # noqa: E402


def _liquid_cloud_column(q_c_val, nlev=20):
    T = jnp.linspace(240.0, 295.0, nlev)[None, :]            # warm (liquid) BL
    p_full = jnp.linspace(1.0e4, 1.0e5, nlev)[None, :]
    p_half = jnp.linspace(9.0e3, 1.013e5, nlev + 1)[None, :]
    q_v = jnp.full((1, nlev), 5.0e-3)
    dp = p_half[:, 1:] - p_half[:, :-1]
    q_c = jnp.zeros((1, nlev)).at[0, 15:18].set(q_c_val)     # thick low cloud
    q_i = jnp.zeros((1, nlev))
    return T, p_full, q_v, dp, q_c, q_i


def _lwp(scheme, q_c_val, **kw):
    T, p_full, q_v, dp, q_c, q_i = _liquid_cloud_column(q_c_val)
    cp = compute_cloud_properties(
        T=T, p_full=p_full, q_v=q_v, dp=dp,
        config=CloudConfig(scheme="resolved",
                           cloud_optics_inhomogeneity=scheme, **kw),
        q_cloud=q_c, q_ice=q_i,
    )
    return np.asarray(cp.lwp)


def test_constant_scheme_scales_linearly_byte_identical():
    """scheme='constant' reproduces the legacy chi*grid-mean path exactly."""
    lwp1 = _lwp("constant", 4.0e-4, cloud_inhomogeneity_factor=1.0)
    lwp_half = _lwp("constant", 4.0e-4, cloud_inhomogeneity_factor=0.5)
    assert np.allclose(lwp_half, 0.5 * lwp1, rtol=1e-6)     # exact scalar scaling


def test_two_region_thins_relative_to_homogeneous():
    lwp_homog = _lwp("constant", 4.0e-4, cloud_inhomogeneity_factor=1.0)
    lwp_tr = _lwp("two_region", 4.0e-4, cloud_fsd=0.75)
    cloudy = lwp_homog > 0
    assert np.all(lwp_tr[cloudy] < lwp_homog[cloudy])       # thinned
    assert np.all(lwp_tr[cloudy] > 0)                       # but not zero


def test_two_region_thins_thick_cloud_more_than_thin():
    """The tau-saturation break: the reduction factor is SMALLER (more
    reduction) for a thicker cloud."""
    def ratio(q_c_val):
        homog = _lwp("constant", q_c_val, cloud_inhomogeneity_factor=1.0)
        tr = _lwp("two_region", q_c_val, cloud_fsd=0.75)
        m = homog > 0
        return float(np.mean(tr[m] / homog[m]))
    chi_thin = ratio(5.0e-5)     # optically thinner cloud
    chi_thick = ratio(8.0e-4)    # optically thicker cloud
    assert chi_thick < chi_thin, (chi_thick, chi_thin)


def test_unknown_inhomogeneity_scheme_raises():
    T, p_full, q_v, dp, q_c, q_i = _liquid_cloud_column(4.0e-4)
    with pytest.raises(ValueError, match="cloud_optics_inhomogeneity"):
        compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp,
            config=CloudConfig(scheme="resolved",
                               cloud_optics_inhomogeneity="bogus"),
            q_cloud=q_c, q_ice=q_i,
        )
