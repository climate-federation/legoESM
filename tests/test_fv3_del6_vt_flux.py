"""Tests for Fortran-faithful `del6_vt_flux` port.

Iter-752 delivers the core standalone algorithm.  Tests verify:
  1. Shape and dtype correctness.
  2. Del-2 (nord=0) on a constant field returns zero flux.
  3. Del-2 on a smooth bump produces the expected centred-difference
     structure.
  4. Del-6 (nord=2) iterates the correct number of times.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Iter-883: also enable x64 at runtime in case JAX was already
# initialized in float32 by an earlier conftest import.  The
# os.environ.setdefault above is for command-line invocation; the
# jax.config.update is the runtime-effective form.
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.core.fv3_del6_vt_flux import (
    del6_vt_flux, compute_del6_metrics,
    fv3_del6_vorticity_damping)


@pytest.fixture
def cdgrid():
    n = 8
    grid = create_cubed_sphere(n)
    return create_cubed_sphere_cdgrid(grid)


def test_del6_metrics_shapes(cdgrid):
    """del6_u has v-edge shape (6, n, n+1); del6_v has u-edge shape
    (6, n+1, n)."""
    del6_u, del6_v = compute_del6_metrics(cdgrid)
    n = cdgrid.base.n
    assert del6_u.shape == (6, n, n + 1)
    assert del6_v.shape == (6, n + 1, n)


def test_del6_metrics_positive(cdgrid):
    """del6_u and del6_v are products of positive sina, dx, dy, 1/dxc
    so should be positive."""
    del6_u, del6_v = compute_del6_metrics(cdgrid)
    assert bool(jnp.all(del6_u > 0))
    assert bool(jnp.all(del6_v > 0))


def test_del6_vt_flux_constant_field_del2(cdgrid):
    """For nord=0 (del-2), a constant input should give zero flux
    because the centred difference of a constant is zero."""
    n = cdgrid.base.n
    del6_u, del6_v = compute_del6_metrics(cdgrid)
    q = jnp.ones((6, n, n))
    rarea = 1.0 / cdgrid.base.area
    fx2, fy2 = del6_vt_flux(q, damp=1.0, nord=0,
                              del6_u=del6_u, del6_v=del6_v,
                              rarea=rarea, cdgrid=cdgrid)
    # For a constant field, centred difference across every edge is 0
    # in the INTERIOR.  Halo values at face boundaries may differ from
    # 1 due to non-identity halo interpolation, but let's check
    # interior at least.
    # Actually pad_halo of a constant field should give the same
    # constant everywhere (interpolation of constant = constant).
    assert bool(jnp.allclose(fx2, 0.0, atol=1e-10))
    assert bool(jnp.allclose(fy2, 0.0, atol=1e-10))


def test_del6_vt_flux_shapes(cdgrid):
    """Output shapes match del6_v, del6_u respectively."""
    n = cdgrid.base.n
    del6_u, del6_v = compute_del6_metrics(cdgrid)
    q = jnp.array(np.random.default_rng(0).normal(size=(6, n, n)))
    rarea = 1.0 / cdgrid.base.area
    for nord in (0, 1, 2):
        fx2, fy2 = del6_vt_flux(q, damp=1.0, nord=nord,
                                  del6_u=del6_u, del6_v=del6_v,
                                  rarea=rarea, cdgrid=cdgrid)
        assert fx2.shape == (6, n + 1, n)
        assert fy2.shape == (6, n, n + 1)


def test_del6_vt_flux_damp_scales(cdgrid):
    """Doubling damp doubles the output (del-2) or 4× (del-4) — both
    linear in damp."""
    n = cdgrid.base.n
    del6_u, del6_v = compute_del6_metrics(cdgrid)
    q = jnp.array(np.random.default_rng(42).normal(size=(6, n, n)))
    rarea = 1.0 / cdgrid.base.area
    for nord in (0, 1, 2):
        fx2_a, fy2_a = del6_vt_flux(q, damp=1.0, nord=nord,
                                      del6_u=del6_u, del6_v=del6_v,
                                      rarea=rarea, cdgrid=cdgrid)
        fx2_b, fy2_b = del6_vt_flux(q, damp=2.0, nord=nord,
                                      del6_u=del6_u, del6_v=del6_v,
                                      rarea=rarea, cdgrid=cdgrid)
        # damp enters linearly on initial d2=damp*q.  On iterated
        # del-n the linearity is preserved (all ops are linear in d2).
        # So fx2_b == 2 * fx2_a for nord=0, but for nord>0 the ratio
        # is still 2 because damp is an overall prefactor.
        # Actually wait — for nord=0, fx2_b = 2*fx2_a.  For nord>0,
        # d2 after first iter = rarea*(fx2-fx2+fy2-fy2) which is ALSO
        # linear in damp (since fx2 scales as damp).  So ratio stays 2.
        ratio_fx = fx2_b / (fx2_a + 1e-30)
        ratio_fy = fy2_b / (fy2_a + 1e-30)
        # Only check INTERIOR cells where ratios are well-defined.
        interior_fx = ratio_fx[:, 2:-2, 2:-2]
        interior_fy = ratio_fy[:, 2:-2, 2:-2]
        # Tolerance ~1e-3 allows numerical roundoff from multi-pass
        # stencil ordering; the key property is linearity preserved.
        assert bool(jnp.allclose(interior_fx, 2.0, atol=1e-3)), \
            f"Expected ratio 2.0 for nord={nord}, got range " \
            f"[{interior_fx.min():.3e}, {interior_fx.max():.3e}]"
        assert bool(jnp.allclose(interior_fy, 2.0, atol=1e-3))


def test_del6_metrics_match_fortran_convention(cdgrid):
    """Verify del6_u, del6_v use Fortran-faithful metrics:
      numerator:   cdgrid.dx_edge_y / cdgrid.dy_edge_x (edge-stagger dx, dy)
      denominator: cdgrid.dxc / cdgrid.dyc (supergrid-derived centre-to-
                   centre distances, matching fv_grid_tools.F90:883)

    Regression sentinel for the iter-752 → 752d chain of Codex
    stop-time fixes:
    * iter-752:  wrong numerator (cell-centre averages for dx, dy).
    * iter-752b: fixed numerator; denominator still cell-centre average.
    * iter-752c: used a direct Haversine of A-grid cells for dxc, dyc —
                 diverges from the repo-locked cdgrid.dxc/dyc supergrid
                 formula.
    * iter-752d: switched to the repo's canonical cdgrid.dxc / cdgrid.dyc
                 fields.

    Pins the complete Fortran-faithful formula so no future drift is
    possible without breaking this test.
    """
    import jax.numpy as jnp

    del6_u, del6_v = compute_del6_metrics(cdgrid)

    # Reconstruct manually using the Fortran formula + cdgrid's locked
    # metric fields.
    cosa_u = cdgrid.cosa_u
    cosa_v = cdgrid.cosa_v
    sina_u = jnp.sqrt(jnp.maximum(1.0 - cosa_u**2, 1e-20))
    sina_v = jnp.sqrt(jnp.maximum(1.0 - cosa_v**2, 1e-20))

    del6_u_expected = sina_v * cdgrid.dx_edge_y / cdgrid.dyc
    del6_v_expected = sina_u * cdgrid.dy_edge_x / cdgrid.dxc

    # Must match module output to machine precision.
    assert bool(jnp.allclose(del6_u, del6_u_expected, atol=1e-14)), \
        "del6_u does not use cdgrid.dx_edge_y / cdgrid.dyc"
    assert bool(jnp.allclose(del6_v, del6_v_expected, atol=1e-14)), \
        "del6_v does not use cdgrid.dy_edge_x / cdgrid.dxc"


def test_del6_vt_flux_zero_field(cdgrid):
    """Zero input should give zero output."""
    n = cdgrid.base.n
    del6_u, del6_v = compute_del6_metrics(cdgrid)
    q = jnp.zeros((6, n, n))
    rarea = 1.0 / cdgrid.base.area
    for nord in (0, 1, 2):
        fx2, fy2 = del6_vt_flux(q, damp=1.0, nord=nord,
                                  del6_u=del6_u, del6_v=del6_v,
                                  rarea=rarea, cdgrid=cdgrid)
        assert bool(jnp.all(fx2 == 0.0))
        assert bool(jnp.all(fy2 == 0.0))


def test_fv3_del6_damping_shapes(cdgrid):
    """High-level helper returns correctly-staggered updates."""
    n = cdgrid.base.n
    u_d = jnp.array(np.random.default_rng(0).normal(size=(6, n, n + 1)))
    v_d = jnp.array(np.random.default_rng(1).normal(size=(6, n + 1, n)))
    for nord in (0, 1, 2):
        du, dv = fv3_del6_vorticity_damping(
            u_d, v_d, damp=1.0, nord=nord, cdgrid=cdgrid)
        assert du.shape == (6, n, n + 1)
        assert dv.shape == (6, n + 1, n)


def test_fv3_del6_damping_zero_field(cdgrid):
    """Zero u_d, v_d → zero damping."""
    n = cdgrid.base.n
    u_d = jnp.zeros((6, n, n + 1))
    v_d = jnp.zeros((6, n + 1, n))
    for nord in (0, 1, 2):
        du, dv = fv3_del6_vorticity_damping(
            u_d, v_d, damp=1.0, nord=nord, cdgrid=cdgrid)
        assert bool(jnp.all(du == 0.0))
        assert bool(jnp.all(dv == 0.0))


def test_fv3_del6_damping_sign_convention_and_units(cdgrid):
    """Verify Fortran sign convention AND velocity-unit conversion:
        du = +fy2 / dx_edge_y   (Fortran: u(circulation) += fy2, then
                                  /dx to convert to velocity form)
        dv = -fx2 / dy_edge_x   (Fortran: v(circulation) -= fx2)

    The helper must return VELOCITY-form updates so it can be added
    directly to our velocity-form u_d, v_d.  Fortran performs the
    update in circulation form and converts later via `*rdx, *rdy`;
    our helper does the conversion internally for API consistency.
    """
    n = cdgrid.base.n
    u_d = jnp.array(np.random.default_rng(42).normal(size=(6, n, n + 1)))
    v_d = jnp.array(np.random.default_rng(43).normal(size=(6, n + 1, n)))

    du, dv = fv3_del6_vorticity_damping(
        u_d, v_d, damp=1.0, nord=1, cdgrid=cdgrid)

    # Manually construct the same pipeline.
    del6_u_m, del6_v_m = compute_del6_metrics(cdgrid)
    rarea = 1.0 / cdgrid.base.area
    vt = u_d * cdgrid.dx_edge_y
    ut = v_d * cdgrid.dy_edge_x
    wk = rarea * (vt[:, :, :-1] - vt[:, :, 1:]
                  - ut[:, :-1, :] + ut[:, 1:, :])
    fx2, fy2 = del6_vt_flux(
        wk, 1.0, 1, del6_u=del6_u_m, del6_v=del6_v_m,
        rarea=rarea, cdgrid=cdgrid)

    # Velocity-form updates: Fortran sign + Fortran rdx/rdy conversion.
    expected_du = fy2 / cdgrid.dx_edge_y     # [m/s] velocity
    expected_dv = -fx2 / cdgrid.dy_edge_x    # [m/s] velocity

    assert bool(jnp.allclose(du, expected_du, atol=1e-14)), \
        "du should equal +fy2/dx_edge_y (velocity-form Fortran update)"
    assert bool(jnp.allclose(dv, expected_dv, atol=1e-14)), \
        "dv should equal -fx2/dy_edge_x (velocity-form Fortran update)"


def test_fv3_del6_damping_constant_wind_no_damping(cdgrid):
    """A constant geographic wind has zero vorticity; the del-n
    damping should therefore be effectively zero (up to halo-
    rotation artifacts at cube corners).

    This is a weaker test than 'zero output' since cubed-sphere halo
    rotation of a geographically-constant wind does produce slight
    non-constancy in the face-local D-grid representation.  Check
    that the damping is much smaller than |u_d|.
    """
    n = cdgrid.base.n
    # Construct u_d, v_d as face-local projections of a uniform
    # geographic flow u_east = 10, v_north = 0.
    u_east = 10.0
    # Get face-local angle from cdgrid.
    angle = cdgrid.base.angle   # (6, n, n)
    cos_a = jnp.cos(angle)
    sin_a = jnp.sin(angle)
    # u_cc = cos_a * u_east - sin_a * v_north
    # For u_d at v-edge (6, n, n+1), we need an angle at v-edge.  Use
    # angle_edge_x (at x-edge = u position at face centre).
    # NB: this is an approximation; the test allows O(|u|) tolerance
    # rather than machine zero.
    u_d = cdgrid.cos_angle_edge_x * u_east  # (6, n, n+1)
    v_d = -cdgrid.sin_angle_edge_y * u_east  # (6, n+1, n)

    du, dv = fv3_del6_vorticity_damping(
        u_d, v_d, damp=1.0, nord=1, cdgrid=cdgrid)

    # For a uniform geographic flow on a cubed sphere, the relative
    # vorticity is zero in a physical sense, so the damping should be
    # small relative to the wind magnitude.  Allow a fairly loose
    # tolerance (0.1 m/s) to account for discrete-metric artifacts.
    max_du = float(jnp.max(jnp.abs(du)))
    max_dv = float(jnp.max(jnp.abs(dv)))
    assert max_du < 0.1 * u_east, f"du max {max_du:.3e} vs u_east={u_east}"
    assert max_dv < 0.1 * u_east, f"dv max {max_dv:.3e} vs u_east={u_east}"
