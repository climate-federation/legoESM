"""FV3_3D iter 413: gradient-flow test through iter-336/337
dynamic Exner Π_total.

Verifies jax.grad finite + non-zero through the dynamic Exner
path (Π_total = Π_ref + π') with respect to ρ' and θ'
perturbations.  Confirms iter-336/337 wiring doesn't introduce
gradient breakage.

Tests
-----

1. ``test_grad_through_pi_total_wrt_rho_prime``
2. ``test_grad_through_pi_total_wrt_theta_prime``
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    compute_exner_perturbation,
)
from legoesm.grids.vertical import create_height_coordinate


def test_grad_through_pi_total_wrt_rho_prime():
    nlev = 5
    z_top = 30000.0
    hc = create_height_coordinate(nlev, z_top)
    theta_prime = jnp.zeros((6, 8, 8, nlev))

    def loss(rho_prime_amp):
        rho_prime = rho_prime_amp * jnp.ones((6, 8, 8, nlev))
        pi_prime = compute_exner_perturbation(
            rho_prime, theta_prime, hc,
        )
        pi_total = hc.exner_ref[None, None, None, :] + pi_prime
        return jnp.mean(pi_total ** 2)

    g = jax.grad(loss)(0.01)
    assert jnp.isfinite(g)
    assert abs(g) > 1e-15, "Gradient must be non-zero"


def test_grad_through_pi_total_wrt_theta_prime():
    nlev = 5
    z_top = 30000.0
    hc = create_height_coordinate(nlev, z_top)
    rho_prime = jnp.zeros((6, 8, 8, nlev))

    def loss(theta_prime_amp):
        theta_prime = theta_prime_amp * jnp.ones((6, 8, 8, nlev))
        pi_prime = compute_exner_perturbation(
            rho_prime, theta_prime, hc,
        )
        pi_total = hc.exner_ref[None, None, None, :] + pi_prime
        return jnp.mean(pi_total ** 2)

    g = jax.grad(loss)(0.5)
    assert jnp.isfinite(g)
    assert abs(g) > 1e-15
