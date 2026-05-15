"""FV3_3D iter 651: get_pt_on_great_circle port.

Faithful JAX port of FV3 ``get_pt_on_great_circle``
(tools/test_cases.F90:4805-4826).

Tests
-----

1. ``test_get_pt_zero_dist_returns_start``.
2. ``test_get_pt_north_heading_increases_lat``.
3. ``test_get_pt_distance_matches_gc_dist``.
4. ``test_get_pt_east_heading_at_equator_only_longitude``.
5. ``test_get_pt_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    get_pt_on_great_circle,
    great_circle_distance,
)


def test_get_pt_zero_dist_returns_start():
    """dist=0 → target = start."""
    lon1, lat1 = 0.5, 0.3
    lon3, lat3 = get_pt_on_great_circle(
        jnp.asarray(lon1), jnp.asarray(lat1),
        jnp.asarray(0.0), jnp.asarray(jnp.pi / 4),
        radius=1.0,
    )
    assert abs(float(lat3) - lat1) < 1e-12
    # lon may differ by 2π wrap; check mod
    lon3_mod = float(lon3) % (2 * jnp.pi)
    lon1_mod = lon1 % (2 * jnp.pi)
    diff = (lon3_mod - lon1_mod + jnp.pi) % (2 * jnp.pi) - jnp.pi
    assert abs(float(diff)) < 1e-12


def test_get_pt_north_heading_increases_lat():
    """At equator, heading=0 (north) → lat increases."""
    lon1, lat1 = 0.0, 0.0
    lon3, lat3 = get_pt_on_great_circle(
        jnp.asarray(lon1), jnp.asarray(lat1),
        jnp.asarray(0.1), jnp.asarray(0.0),
        radius=1.0,
    )
    assert float(lat3) > lat1
    # Lat3 should be ≈ 0.1 (small angle approx)
    assert abs(float(lat3) - 0.1) < 1e-10


def test_get_pt_distance_matches_gc_dist():
    """great_circle_distance(start, get_pt(start, dist, ...)) ≈ dist."""
    rng = np.random.default_rng(seed=651)
    for _ in range(5):
        lon1 = float(rng.uniform(0.1, 2 * jnp.pi - 0.1))
        lat1 = float(rng.uniform(-1.0, 1.0))
        dist = float(rng.uniform(0.01, 1.5))
        heading = float(rng.uniform(0.0, 2 * jnp.pi))
        lon3, lat3 = get_pt_on_great_circle(
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(dist), jnp.asarray(heading),
            radius=1.0,
        )
        d_check = float(great_circle_distance(
            jnp.asarray(lon1), jnp.asarray(lat1),
            lon3, lat3, radius=1.0,
        ))
        assert abs(d_check - dist) / max(dist, 1e-10) < 1e-10, (
            f"dist={dist}, check={d_check}"
        )


def test_get_pt_east_heading_at_equator_only_longitude():
    """At equator, heading=π/2 (east) along small dist → lat stays ~0."""
    lon1, lat1 = 1.0, 0.0
    lon3, lat3 = get_pt_on_great_circle(
        jnp.asarray(lon1), jnp.asarray(lat1),
        jnp.asarray(0.05), jnp.asarray(jnp.pi / 2),
        radius=1.0,
    )
    # Lat should remain ~0 (great circle from equator with east heading
    # is equator, so lat stays 0 throughout)
    assert abs(float(lat3)) < 1e-12


def test_get_pt_finite():
    """No NaN/Inf for various inputs."""
    rng = np.random.default_rng(seed=652)
    for _ in range(5):
        lon1 = float(rng.uniform(-jnp.pi, jnp.pi))
        lat1 = float(rng.uniform(-1.0, 1.0))
        dist = float(rng.uniform(0.0, 1.0))
        heading = float(rng.uniform(0.0, 2 * jnp.pi))
        lon3, lat3 = get_pt_on_great_circle(
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(dist), jnp.asarray(heading),
            radius=1.0,
        )
        assert jnp.isfinite(lon3)
        assert jnp.isfinite(lat3)
