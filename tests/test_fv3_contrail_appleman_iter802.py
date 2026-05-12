"""FV3_3D iter 802: contrail_appleman_fv3 (simplified Schmidt-Appleman).

Boolean: (T < T_SA) AND (RH_ice ≥ rh_ice_thresh).

Composes iter-800 ``relative_humidity_ice_fv3``.

Tests
-----

1. ``test_warm_dry_no_contrail``: T=250K + dry → False.
2. ``test_cold_supersat_contrail``: T=220K + supersat ice → True.
3. ``test_cold_dry_no_contrail``: T=220K + dry → False.
4. ``test_warm_moist_no_contrail``: T=250K + saturated → False.
5. ``test_custom_T_SA``: 240K threshold catches warmer cases.
6. ``test_shapes_3d_bool``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import thermo
from legoesm.grids.cubed_sphere import contrail_appleman_fv3


def test_warm_dry_no_contrail():
    """T=250K + dry → no contrail."""
    t = jnp.array([250.0])
    p = jnp.array([30_000.0])
    q = jnp.array([1e-6])
    assert bool(contrail_appleman_fv3(t, p, q)[0]) is False


def test_cold_supersat_contrail():
    """T=220K + 1.5·q_sat_ice (RH_ice=150%) → contrail."""
    t = jnp.array([220.0])
    p = jnp.array([25_000.0])
    q = 1.5 * thermo.saturation_mixing_ratio_ice(t, p)
    assert bool(contrail_appleman_fv3(t, p, q)[0]) is True


def test_cold_dry_no_contrail():
    """T=220K + dry → no contrail (RH_ice ≪ 100%)."""
    t = jnp.array([220.0])
    p = jnp.array([25_000.0])
    q = jnp.array([1e-7])
    assert bool(contrail_appleman_fv3(t, p, q)[0]) is False


def test_warm_moist_no_contrail():
    """T=250K + saturated → no contrail (T > T_SA)."""
    t = jnp.array([250.0])
    p = jnp.array([50_000.0])
    q = 1.2 * thermo.saturation_mixing_ratio_ice(t, p)
    assert bool(contrail_appleman_fv3(t, p, q)[0]) is False


def test_custom_T_SA():
    """T=235K + saturated: default T_SA=233.15 rejects, T_SA=240 catches."""
    t = jnp.array([235.0])
    p = jnp.array([30_000.0])
    q = 1.1 * thermo.saturation_mixing_ratio_ice(t, p)
    assert bool(contrail_appleman_fv3(t, p, q)[0]) is False
    assert bool(contrail_appleman_fv3(t, p, q, T_SA=240.0)[0]) is True


def test_shapes_3d_bool():
    """3-D shapes preserved, dtype bool."""
    rng = np.random.default_rng(seed=802)
    n_x, n_y, km = 4, 5, 20
    t = jnp.asarray(rng.uniform(200.0, 270.0, size=(n_x, n_y, km)))
    p = jnp.asarray(rng.uniform(10_000.0, 50_000.0, size=(n_x, n_y, km)))
    q = jnp.asarray(rng.uniform(0.0, 0.005, size=(n_x, n_y, km)))
    mask = contrail_appleman_fv3(t, p, q)
    assert mask.shape == (n_x, n_y, km)
    assert mask.dtype == jnp.bool_
