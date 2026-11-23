"""McSnow Hall+Beheng collision kernel and Beard (1976) terminal velocity.

Reference values were computed with the McSnow-faithful implementation that
generated the esm-co-scientist PySDM box-model data
(esm-co-scientist ``scripts/datagen/hall_kernel.py``: ``hall_efficiency`` on
McSnow's ``colleff_HallBeheng.dat``, and ``beard1976`` ported from McSnow's
``vterm_beard``).  On 2026-11-23 the legoESM port matched that implementation
exactly on 200k random pairs (0.5 um - 3 mm, plus every table node and r=0
edges) and the velocity to 3e-14 relative.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.microphysics.sdm.config import SDMConfig
from legoesm.atmosphere.physics.microphysics.sdm.kernels import (
    collision_kernel,
    hall_beheng_kernel,
    terminal_velocity,
    terminal_velocity_beard1976,
)

jax.config.update("jax_enable_x64", True)

# (collector radius R [m], small radius r [m], E(R, r/R)) -- covers the floor
# column R < 10 um, interior bilinear cells, a node, R > 300 um clamping.
EFFICIENCY_REFERENCE = [
    (5e-06, 2e-06, 0.014249999999999997),
    (1.5e-05, 7.5e-06, 0.0505),
    (2.5e-05, 2.25e-05, 0.08425),
    (3e-05, 3e-05, 0.027),
    (4e-05, 4e-06, 0.053999999999999986),
    (6.5e-05, 2e-05, 0.9001538461538463),
    (0.00012, 6e-05, 1.0),
    (0.0003, 1.5e-05, 0.96),
    (0.0008, 0.0004, 1.0),
]

# (radius [m], v_t [m/s]) -- all three Beard regimes and the 0.7 cm diameter cap.
VELOCITY_REFERENCE = [
    (5e-06, 0.0030428390473778515),
    (9.5e-06, 0.010877795766585919),
    (5e-05, 0.2492461225933794),
    (0.0002, 1.5787323571528036),
    (0.000535, 4.220845903911309),
    (0.001, 6.468631436085741),
    (0.0035, 9.049292480259759),
    (0.005, 9.049292480259759),
]


@pytest.mark.parametrize("R, r, E_ref", EFFICIENCY_REFERENCE)
def test_hall_beheng_efficiency_matches_mcsnow(R, r, E_ref):
    dv = 1.0  # K / (pi (R+r)^2 |dv|) is the efficiency
    K = hall_beheng_kernel(jnp.asarray(R), jnp.asarray(r), jnp.asarray(dv))
    E = float(K) / (np.pi * (R + r) ** 2)
    assert E == pytest.approx(E_ref, rel=1e-12)
    # symmetric in the pair
    K_swap = hall_beheng_kernel(jnp.asarray(r), jnp.asarray(R), jnp.asarray(dv))
    assert float(K_swap) == float(K)


@pytest.mark.parametrize("r, v_ref", VELOCITY_REFERENCE)
def test_beard1976_matches_mcsnow(r, v_ref):
    assert float(terminal_velocity_beard1976(jnp.asarray(r))) == pytest.approx(v_ref, rel=1e-12)


def test_dispatch_and_zero_radius():
    cfg = SDMConfig(collision_kernel="hall_beheng", terminal_velocity="beard1976")
    r = jnp.asarray([2.0e-5, 0.0])
    v = terminal_velocity(r, 1.2, 1.0e5, 290.0, cfg)
    np.testing.assert_allclose(np.asarray(v[:1]), np.asarray(terminal_velocity_beard1976(r[:1])))
    K = collision_kernel(jnp.asarray([2.0e-5, 0.0]), jnp.asarray([1.0e-5, 0.0]),
                         jnp.asarray([0.01, 0.0]), cfg)
    assert float(K[1]) == 0.0
    assert np.all(np.isfinite(np.asarray(jax.grad(
        lambda a: hall_beheng_kernel(a, jnp.asarray(1.0e-5), jnp.asarray(0.01)))(jnp.asarray(2.0e-5)))))
