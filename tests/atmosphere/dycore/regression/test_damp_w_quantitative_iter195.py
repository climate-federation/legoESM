"""FV3_3D iter 195: quantitative correctness test for iter-193 damp_w.

iter-193 added FV3-faithful post-step ``damp_w + nord_w`` damping
of vertical velocity ``w`` to the NH 3D path.  Its unit tests
(``test_damp_w_nh_iter193.py``) verify the wiring CHANGES the state
but not the *direction* of the change.  A sign-flipped damping
(e.g., ``w_new = w - dw`` where it should be ``+= dw``) would still
pass "changes-the-state" but would AMPLIFY w instead of damping it
— the worst-case silent failure for a damping mechanism.

This iter mirrors the iter-174 (div_damp) / iter-175 (damp_v)
quantitative pattern: initialise with a sinusoidal w perturbation
and verify post-step ``max|w|`` is LESS than the no-damping
baseline.

Design
------

* Use a sinusoidal w pattern on each face so ``max|w|`` reflects
  the smooth large-scale signal that del-(2*(nord_w+1)) damping
  attenuates.
* Run a few steps so the damping compounds.
* Assert ``max|w_active| < max|w_baseline|`` with a 1 % minimum
  reduction floor — catches sign errors while being insensitive to
  ULP noise.

Tests
-----

1. ``test_damp_w_reduces_w_amplitude`` — ``damp_w=0.030`` with
   ``nord_w=1`` reduces ``max|w|`` after 5 steps vs no-damp baseline.
2. ``test_damp_w_no_amplification_for_any_nord[0]`` /
   ``[1]`` / ``[2]`` — del-2 / del-4 / del-6 paths NEVER amplify
   w (max|w_active| <= max|w_baseline| × 1.001 floor for ULP).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
    CDGridCompressibleEulerConfig,
    CDGridCompressibleEulerModel,
)
from legoesm.core.field import Field
from legoesm.core.state import NonHydrostaticState
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.vertical import (
    create_height_coordinate, compute_terrain_metric,
)


@pytest.fixture(scope="module")
def w_perturbed_nh_state():
    """NH state with a sinusoidal w perturbation on each face.
    Pattern: ``w(face, i, j, k) = W0 * sin(4π i / n) * cos(4π j / n)``
    so ``max|w|`` is well-defined (= W0) and del-n damping has a
    clear amplitude to attenuate."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    nlev_half = nlev + 1
    i_idx = jnp.arange(n)
    j_idx = jnp.arange(n)
    k_idx = jnp.arange(nlev_half)
    pattern = (
        jnp.sin(4 * jnp.pi * i_idx[None, :, None, None] / n)
        * jnp.cos(4 * jnp.pi * j_idx[None, None, :, None] / n)
        * jnp.ones_like(k_idx[None, None, None, :], dtype=jnp.float64)
    )
    pattern = jnp.broadcast_to(pattern, (6, n, n, nlev_half))
    W0 = 0.5
    w_perturb = jnp.asarray(W0 * pattern, dtype=jnp.float64)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=jnp.zeros((6, n, n, nlev)), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=w_perturb, name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(data=jnp.zeros((6, n, n, nlev)),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(data=jnp.zeros((6, n, n)), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(data=jnp.zeros((6, n, n, nlev, 0)), name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )
    return grid, height_coord, terrain_metric, state


def _step5(model, state, dt=10.0):
    s = state
    for _ in range(5):
        s = model.step(s, dt)
    return s


def test_damp_w_reduces_w_amplitude(w_perturbed_nh_state):
    """``damp_w=0.030`` reduces ``max|w|`` vs no-damp baseline.
    Catches a sign error in the iter-193 wiring (which would
    AMPLIFY w instead of damping)."""
    grid, height_coord, terrain_metric, state = w_perturbed_nh_state

    cfg_baseline = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,                # disable other damping
        damp_w=0.0,
    )
    cfg_active = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        damp_w=0.030, nord_w=1,
    )

    m_base = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_baseline,
    )
    m_active = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_active,
    )

    s_base = _step5(m_base, state)
    s_active = _step5(m_active, state)

    max_w_base = float(jnp.max(jnp.abs(s_base.w.data)))
    max_w_active = float(jnp.max(jnp.abs(s_active.w.data)))

    assert max_w_active < max_w_base, (
        f"damp_w must REDUCE max|w|: "
        f"baseline={max_w_base:.4e}, active={max_w_active:.4e} "
        f"(damping AMPLIFIED w — sign error)"
    )
    rel_reduction = (max_w_base - max_w_active) / max(max_w_base, 1e-30)
    assert rel_reduction > 0.01, (
        f"damp_w must reduce max|w| by at least 1 % "
        f"(observed {100*rel_reduction:.3f} %)."
    )


@pytest.mark.parametrize("nord_w", [0, 1, 2])
def test_damp_w_no_amplification_for_any_nord(w_perturbed_nh_state, nord_w):
    """Across nord_w in {0, 1, 2} (del-2 / del-4 / del-6), damping
    should never AMPLIFY w.  Catches a sign error or scale issue
    that would only surface at a specific nord_w."""
    grid, height_coord, terrain_metric, state = w_perturbed_nh_state

    # Tiny damp_w so the test is sensitive to direction but the
    # effect is small (catches AMPLIFICATION reliably while
    # remaining bounded).
    cfg_baseline = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0, damp_w=0.0,
    )
    cfg_active = CDGridCompressibleEulerConfig(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        damp_w=0.005, nord_w=nord_w,
    )

    m_base = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_baseline,
    )
    m_active = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_active,
    )

    s_base = _step5(m_base, state)
    s_active = _step5(m_active, state)

    max_w_base = float(jnp.max(jnp.abs(s_base.w.data)))
    max_w_active = float(jnp.max(jnp.abs(s_active.w.data)))

    # Allow 0.1 % cushion for float roundoff.  The damping must not
    # amplify w (its sign is to remove energy).
    assert max_w_active <= max_w_base * 1.001, (
        f"nord_w={nord_w}: damp_w must not AMPLIFY w: "
        f"baseline={max_w_base:.4e}, active={max_w_active:.4e}.  "
        f"Sign error in the iter-193 wiring at this nord_w."
    )
