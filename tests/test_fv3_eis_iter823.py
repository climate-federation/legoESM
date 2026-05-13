"""FV3_3D iter 823: eis_fv3 (Wood-Bretherton 2006 EIS).

EIS = LTS − Γ_m · (z_700 − LCL).
LTS = θ_700 − θ_surf.

Tests
-----

1. ``test_eis_subtropical_sc``: cold SST, warm 700 → EIS > 8.
2. ``test_eis_equals_lts_when_lcl_eq_z700``: LCL = z_700 → EIS = LTS.
3. ``test_eis_monotonic_lts``: ↑LTS → ↑EIS.
4. ``test_eis_monotonic_lcl``: ↑LCL → ↑EIS (less subtraction).
5. ``test_eis_composes_iter806``: compose with iter-806 Γ_m.
6. ``test_eis_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import eis_fv3, lapse_rate_moist_fv3


def test_eis_subtropical_sc():
    """Subtropical Sc deck: θ_sfc=288, θ_700=300, LCL=400 m, Γ_m=5e-3 K/m.

    LTS = 300 - 288 = 12 K
    EIS = 12 - 5e-3·(3000 - 400) = 12 - 13 = -1 K
    Hmm, that gives small EIS. Use stronger inversion:
    θ_sfc=288, θ_700=305, Γ_m=4e-3 K/m, LCL=600 m:
    LTS = 17 K
    EIS = 17 - 4e-3·(3000-600) = 17 - 9.6 = 7.4 K — transitional
    """
    theta_700 = jnp.array([305.0])
    theta_sfc = jnp.array([288.0])
    lcl = jnp.array([600.0])
    gamma = jnp.array([4.0e-3])
    eis = eis_fv3(theta_700, theta_sfc, lcl, gamma)
    np.testing.assert_allclose(np.asarray(eis), [7.4], rtol=1e-12)


def test_eis_equals_lts_when_lcl_eq_z700():
    """LCL = z_700 → second term = 0 → EIS = LTS."""
    theta_700 = jnp.array([305.0])
    theta_sfc = jnp.array([288.0])
    lcl = jnp.array([3000.0])  # = z_700 default
    gamma = jnp.array([4.0e-3])
    eis = eis_fv3(theta_700, theta_sfc, lcl, gamma)
    lts = 305.0 - 288.0
    np.testing.assert_allclose(np.asarray(eis), [lts], rtol=1e-12)


def test_eis_monotonic_lts():
    """↑LTS → ↑EIS at fixed (LCL, Γ_m)."""
    theta_sfc = jnp.array([288.0, 288.0, 288.0])
    theta_700 = jnp.array([295.0, 305.0, 310.0])  # increasing LTS
    lcl = jnp.array([600.0, 600.0, 600.0])
    gamma = jnp.array([4.0e-3, 4.0e-3, 4.0e-3])
    eis = eis_fv3(theta_700, theta_sfc, lcl, gamma)
    assert jnp.all(jnp.diff(eis) > 0.0)


def test_eis_monotonic_lcl():
    """↑LCL → ↑EIS (less Γ_m·(z_700-LCL) subtracted)."""
    theta_700 = jnp.array([305.0, 305.0])
    theta_sfc = jnp.array([288.0, 288.0])
    lcl_lo = jnp.array([400.0])
    lcl_hi = jnp.array([1500.0])
    gamma = jnp.array([4.0e-3])
    eis_lo = eis_fv3(theta_700[:1], theta_sfc[:1], lcl_lo, gamma)
    eis_hi = eis_fv3(theta_700[:1], theta_sfc[:1], lcl_hi, gamma)
    assert float(eis_hi[0]) > float(eis_lo[0])


def test_eis_composes_iter806():
    """Compose with iter-806 Γ_m from (T_850, q_sat_850)."""
    t_850 = jnp.array([285.0])
    q_sat_850 = jnp.array([0.008])
    gamma = lapse_rate_moist_fv3(t_850, q_sat_850)
    theta_700 = jnp.array([305.0])
    theta_sfc = jnp.array([288.0])
    lcl = jnp.array([600.0])
    eis = eis_fv3(theta_700, theta_sfc, lcl, gamma)
    assert jnp.all(jnp.isfinite(eis))
    # EIS in physical range for moist Sc regime
    assert -5.0 < float(eis[0]) < 20.0


def test_eis_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=823)
    n_x, n_y = 6, 8
    theta_700 = jnp.asarray(rng.uniform(295.0, 315.0, size=(n_x, n_y)))
    theta_sfc = jnp.asarray(rng.uniform(280.0, 295.0, size=(n_x, n_y)))
    lcl = jnp.asarray(rng.uniform(200.0, 2500.0, size=(n_x, n_y)))
    gamma = jnp.asarray(rng.uniform(3e-3, 6e-3, size=(n_x, n_y)))
    eis = eis_fv3(theta_700, theta_sfc, lcl, gamma)
    assert eis.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(eis))
