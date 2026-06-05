"""FV3_3D iter 308: ``fv3_del6_vorticity_damping`` linearity
in the ``damp`` coefficient.

The damp_v post-step helper computes velocity updates as::

    damp4 = (damp_v * da_min_c)^(nord+1)
    (du, dv) = damp4 * F(u, v, nord, cdgrid)

where F is a fixed-weight finite-difference operator chain
(circulation → vorticity → del^n flux → velocity update).
The ``damp4`` enters as a scalar prefactor, so::

    damp = α * d  →  (du, dv) = α * (du@d, dv@d)
    damp = 0      →  (du, dv) = (0, 0)

Pins the linear-in-damp prefactor structure that's the FV3-
faithful contract for the iter-12 (PE) / iter-169 (NH)
post-step damp_v site.

Tests
-----

1. ``test_del6_damp_v_zero_damp`` — damp=0 → (du, dv) = 0
   exactly.
2. ``test_del6_damp_v_linear_in_damp`` — damp=α*d output
   equals α times damp=d output bit-for-bit.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.fv3_del6_vt_flux import fv3_del6_vorticity_damping
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid


def _setup(seed=308):
    n = 8
    grid = create_cubed_sphere(n)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    rng = np.random.default_rng(seed=seed)
    # FV3 D-grid layout: u at v-edges (n, n+1), v at u-edges (n+1, n)
    u = jnp.asarray(rng.uniform(-3.0, 3.0,
                                size=(6, n, n + 1)))
    v = jnp.asarray(rng.uniform(-3.0, 3.0,
                                size=(6, n + 1, n)))
    return cdgrid, u, v


@pytest.mark.parametrize("nord", [0, 1, 2])
def test_del6_damp_v_zero_damp(nord):
    """damp=0 → (du, dv) = 0 exactly across nord ∈ {0, 1, 2}."""
    cdgrid, u, v = _setup()
    du, dv = fv3_del6_vorticity_damping(
        u, v, damp=0.0, nord=nord, cdgrid=cdgrid,
    )
    assert jnp.all(du == 0.0), (
        f"damp=0 must yield du=0 exactly at nord={nord}."
    )
    assert jnp.all(dv == 0.0), (
        f"damp=0 must yield dv=0 exactly at nord={nord}."
    )


@pytest.mark.parametrize("nord", [0, 1, 2])
@pytest.mark.parametrize("alpha", [-2.5, 0.5, 3.0, 1e-6])
def test_del6_damp_v_linear_in_damp(nord, alpha):
    """damp = α*d → (du, dv) = α * (du@d, dv@d) bit-for-bit.

    The damp coefficient is a scalar prefactor on the flux
    output, so the damping output is exactly linear in damp.
    """
    cdgrid, u, v = _setup()
    base_damp = 1e-12 if nord == 0 else 1e-30 if nord == 2 else 1e-21

    du1, dv1 = fv3_del6_vorticity_damping(
        u, v, damp=base_damp, nord=nord, cdgrid=cdgrid,
    )
    du2, dv2 = fv3_del6_vorticity_damping(
        u, v, damp=alpha * base_damp, nord=nord, cdgrid=cdgrid,
    )

    np.testing.assert_allclose(
        np.asarray(du2), alpha * np.asarray(du1),
        rtol=1e-13, atol=1e-30,
        err_msg=(
            f"du-component not linear in damp (alpha={alpha}, "
            f"nord={nord}, base_damp={base_damp})"
        ),
    )
    np.testing.assert_allclose(
        np.asarray(dv2), alpha * np.asarray(dv1),
        rtol=1e-13, atol=1e-30,
    )
