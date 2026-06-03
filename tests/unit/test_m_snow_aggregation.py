"""SAM M2005 snow self-aggregation NSAGG tests (iter-30).

gSAM ``MICRO_M2005`` (module_mp_graupel.f90:2894) lets falling snow flakes
collide and merge (Passarelli 1978 / Reisner 1998)::

    CONS15 = −1108·EII·π^((1−BS)/3)·ρ_sn^((−2−BS)/3)/(4·720)          (EII=0.1)
    NSAGG  = CONS15·ASN·ρ^((2+BS)/3)·q_s^((2+BS)/3)·(N_s·ρ)^((4−BS)/3)/ρ

It is a pure SINK of snow NUMBER (CONS15<0): mass q_s is CONSERVED (no q_s
tendency, no latent heat), only N_s decreases as flakes merge into fewer,
larger ones. Requires double-moment snow (N_s).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    snow_self_aggregation_nsagg,
)
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState


jax.config.update("jax_enable_x64", True)
_CFG = MorrisonConfig()


def _na(q_s=1.0e-3, N_s=1.0e4, rho=1.0, dt=None):
    return float(snow_self_aggregation_nsagg(
        jnp.asarray(q_s), jnp.asarray(N_s), jnp.asarray(rho), _CFG, dt=dt))


def _hand(q_s=1.0e-3, N_s=1.0e4, rho=1.0):
    bs = _CFG.fall_b_s
    asn = _CFG.fall_a_s * (_CFG.rho_su / rho) ** 0.54
    cons15 = (-1108.0 * _CFG.snow_aggregation_eii
              * math.pi ** ((1.0 - bs) / 3.0)
              * _CFG.rho_snow ** ((-2.0 - bs) / 3.0) / (4.0 * 720.0))
    nsagg = (cons15 * asn * rho ** ((2.0 + bs) / 3.0)
             * q_s ** ((2.0 + bs) / 3.0)
             * (N_s * rho) ** ((4.0 - bs) / 3.0) / rho)
    return -nsagg  # positive loss (caller subtracts)


def test_nsagg_matches_sam_formula():
    assert _na() == pytest.approx(_hand(), rel=1e-6)
    assert _na(q_s=2.0e-3, N_s=5.0e4) == pytest.approx(
        _hand(q_s=2.0e-3, N_s=5.0e4), rel=1e-6)


def test_nsagg_is_a_number_sink():
    """CONS15<0 ⇒ the returned positive loss is subtracted from N_s."""
    assert _na() > 0.0


def test_nsagg_grows_with_number_and_mass():
    """More/denser snow ⇒ more collisions ⇒ faster aggregation."""
    assert _na(N_s=1.0e5) > _na(N_s=1.0e4) > 0.0
    assert _na(q_s=5.0e-3) > _na(q_s=1.0e-3) > 0.0


def test_nsagg_gate_tiny_snow():
    assert _na(q_s=1.0e-9) == 0.0


def test_nsagg_donor_clamped():
    assert _na(q_s=5.0e-3, N_s=1.0e7, dt=20.0) <= 1.0e7 / 20.0 + 1e-6


def test_nsagg_timescale_physical():
    """Snow self-aggregation is a slow process: τ of order tens of minutes."""
    loss = _na(q_s=1.0e-3, N_s=1.0e4)
    tau = 1.0e4 / loss
    assert 300.0 < tau < 7200.0  # 5 min .. 2 h


def test_nsagg_conserves_snow_mass():
    """Aggregation must NOT change q_s — only N_s. Toggle do_snow_aggregation
    and confirm dq_s_dt is identical while dN_s_dt differs."""
    T = 263.0
    rho = 8.0e4 / (constants.R_d * T)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=z, q_i=z, q_s=jnp.full((1, 1), 2.0e-3), q_g=z,
        N_c=z, N_r=z, N_i=z, N_s=jnp.full((1, 1), 1.0e5),
    )
    args = (jnp.full((1, 1), T), jnp.full((1, 1), 1.0e-3), hm,
            jnp.full((1, 1), 8.0e4), jnp.full((1, 2), 8.0e4),
            jnp.full((1, 1), rho), jnp.full((1, 1), 1.0e9), 20.0)
    on = morrison_microphysics(*args, MorrisonConfig(do_snow_aggregation=True))
    off = morrison_microphysics(*args, MorrisonConfig(do_snow_aggregation=False))
    # Mass tendency unchanged by aggregation.
    assert float(on.dq_s_dt[0, 0]) == pytest.approx(
        float(off.dq_s_dt[0, 0]), abs=1e-12)
    # Number tendency MORE negative (extra sink) with aggregation on.
    assert float(on.dN_s_dt[0, 0]) < float(off.dN_s_dt[0, 0])


def test_nsagg_ad_safe():
    def loss(q_s0):
        return jnp.sum(snow_self_aggregation_nsagg(
            q_s0.reshape(1, 1), jnp.full((1, 1), 1.0e4),
            jnp.full((1, 1), 1.0), _CFG, dt=20.0))

    for q0 in (0.0, 1.0e-3):
        g = jax.grad(loss)(jnp.asarray(q0))
        assert bool(jnp.isfinite(g))
