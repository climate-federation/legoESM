"""FV3_3D iter 199: quantitative direction test for the iter-180
Smagorinsky-adaptive A_h on the NH 3D path.

iter-180 ported the PE iter-57/58 Smagorinsky-A_h augmentation to
the NH path.  Existing tests in
``test_smagorinsky_ah_nh_iter180.py`` verify:

* default ``smagorinsky_cs=0.0`` is bit-for-bit baseline,
* ``smagorinsky_cs > 0`` CHANGES winds vs the baseline,
* ``smagorinsky_cs > 0`` with ``A_h == 0`` is bit-for-bit baseline
  (gated inside the A_h > 0 block).

But NO test verifies the *direction* of the Smagorinsky effect.  A
sign-flipped Smagorinsky (subtracting ``c_s * dx² * |D|`` instead
of adding) would still pass "changes winds" but would AMPLIFY
winds at high-strain regions instead of damping them — the
worst-case silent failure for a viscous closure.

This iter mirrors iter-174 (div_damp) / iter-175 (damp_v) /
iter-198 (nord >= 1) quantitative pattern: a high-strain IC,
3 configs (no damping, static A_h only, static A_h + smag),
verify Smagorinsky adds MORE damping (smaller max|u|) than
static A_h alone.

Tests
-----

1. ``test_smag_reduces_max_wind_more_than_static_ah`` — high-
   strain IC; smag_cs > 0 + A_h > 0 produces SMALLER max|u| after
   5 steps than A_h-only.  Catches sign error in Smagorinsky.
2. ``test_smag_no_amplification_for_typical_cs_range`` — sweep
   smag_cs in {0.1, 0.2, 0.4} (PE-tested range from iter-60); none
   should AMPLIFY winds beyond the A_h-only baseline.
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
def high_strain_nh_state():
    """NH state with a high-strain shear pattern in u that engages
    Smagorinsky.  Pattern: ``u(face, i, j, k) = U0 * (j - n/2)``
    so ``∂u/∂y`` is large and uniform — pure shear, max strain
    rate ≈ U0 / dx."""
    n = 8
    nlev = 5
    z_top = 30000.0
    grid = create_cubed_sphere(n)
    height_coord = create_height_coordinate(nlev, z_top)
    terrain = jnp.zeros((6, n, n))
    terrain_metric = compute_terrain_metric(terrain, height_coord)

    j_idx = jnp.arange(n)
    shear = (j_idx - n / 2)[None, None, :, None]
    pattern = jnp.broadcast_to(shear, (6, n, n, nlev)).astype(jnp.float64)
    U0 = 5.0
    u_perturb = jnp.asarray(U0 * pattern, dtype=jnp.float64)

    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")

    state = NonHydrostaticState(
        u=Field(data=u_perturb, name="u", dims=dims_3d, units="m/s"),
        v=Field(data=jnp.zeros((6, n, n, nlev)), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(data=jnp.zeros((6, n, n, nlev + 1)), name="w",
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


def test_smag_reduces_max_wind_more_than_static_ah(high_strain_nh_state):
    """smagorinsky_cs > 0 + A_h > 0 produces SMALLER max|u| after
    5 steps than A_h-only baseline on a high-strain IC.  Catches a
    sign error or scale issue in the Smagorinsky augmentation."""
    grid, height_coord, terrain_metric, state = high_strain_nh_state

    common = dict(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        A_h=1e6,
    )
    cfg_static_only = CDGridCompressibleEulerConfig(
        **common,
        smagorinsky_cs=0.0,
    )
    cfg_static_plus_smag = CDGridCompressibleEulerConfig(
        **common,
        smagorinsky_cs=0.20,
    )

    m_static = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_static_only,
    )
    m_smag = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_static_plus_smag,
    )

    s_static = _step5(m_static, state)
    s_smag = _step5(m_smag, state)

    max_u_static = float(jnp.max(jnp.abs(s_static.u.data)))
    max_u_smag = float(jnp.max(jnp.abs(s_smag.u.data)))

    assert max_u_smag < max_u_static, (
        f"smagorinsky_cs=0.20 must REDUCE max|u| relative to "
        f"static A_h alone: static={max_u_static:.4e}, "
        f"smag={max_u_smag:.4e} (Smagorinsky AMPLIFIED winds — "
        f"sign error in compute_smagorinsky_ah_3d)."
    )


@pytest.mark.parametrize("smag_cs", [0.1, 0.2, 0.4])
def test_smag_no_amplification_for_typical_cs_range(
    high_strain_nh_state, smag_cs,
):
    """Across smag_cs in {0.1, 0.2, 0.4} (PE iter-60-tested range)
    none should AMPLIFY winds beyond the A_h-only baseline."""
    grid, height_coord, terrain_metric, state = high_strain_nh_state

    common = dict(
        hyperdiff_coeff=0.0, n_acoustic_substeps=4,
        sponge_coeff=0.0,
        A_h=1e6,
    )
    cfg_static_only = CDGridCompressibleEulerConfig(
        **common,
        smagorinsky_cs=0.0,
    )
    cfg_smag = CDGridCompressibleEulerConfig(
        **common,
        smagorinsky_cs=smag_cs,
    )

    m_static = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_static_only,
    )
    m_smag = CDGridCompressibleEulerModel(
        grid, height_coord, terrain_metric, cfg_smag,
    )

    s_static = _step5(m_static, state)
    s_smag = _step5(m_smag, state)

    max_u_static = float(jnp.max(jnp.abs(s_static.u.data)))
    max_u_smag = float(jnp.max(jnp.abs(s_smag.u.data)))

    assert max_u_smag <= max_u_static * 1.001, (
        f"smag_cs={smag_cs}: must not AMPLIFY winds: "
        f"static={max_u_static:.4e}, smag={max_u_smag:.4e}.  "
        f"Sign error in the Smagorinsky augmentation at this c_s."
    )
