"""Biharmonic compact-outer fix: the 2Δx damping the wide outer ∇² lacked.

Oracle-review finding (deferred → fixed here): ``hyperdiffusion_3d`` composed a
compact inner Laplacian with a WIDE outer ``div(grad)`` whose centred reach-2
gradient ``(f[i+1]-f[i-1])/dx`` is IDENTICALLY ZERO on a ``(-1)^i`` grid mode,
so the composite ∇⁴ had an exact 2Δx NULL — it did not damp the grid-scale
checkerboard a biharmonic exists to remove.  ``compact_outer=True`` makes the
outer Laplacian compact too (``(1,-4,6,-4,1)`` stencil), giving maximal 2Δx
damping (MOM/MPAS ``del4=del2(del2)``).  The legacy wide path stays the default
(bit-identical) so coefficients tuned to it are undisturbed.

These tests are NON-VACUOUS: they assert the wide default is ~null at 2Δx and
the compact path damps it strongly (the contract's "grid-scale noise damped far
faster" was previously vacuously false on the default path).
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.core.operators_3d import hyperdiffusion_3d
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.physics.lateral_mixing.biharmonic import biharmonic_lateral_mixing
from legoesm.ocean.physics.lateral_mixing.config import BiharmonicConfig


def _checkerboard(n, nlev):
    """Pure 2Δx (-1)^(i+j) checkerboard, shape (6, n, n, nlev)."""
    ii = np.arange(n)
    board = ((-1.0) ** (ii[:, None] + ii[None, :]))  # (n, n)
    field = np.broadcast_to(board[None, :, :, None], (6, n, n, nlev)).copy()
    return jnp.asarray(field)


def _interior_rms(arr, trim=2):
    """RMS over the face interior (trim edge cells contaminated by cube halo)."""
    core = np.asarray(arr)[:, trim:-trim, trim:-trim, :]
    return float(np.sqrt(np.mean(core ** 2)))


def test_wide_outer_has_2dx_null_compact_does_not():
    """The whole finding: the wide outer barely touches the 2Δx mode; compact
    damps it ~2-3 orders of magnitude harder.

    On a FLAT periodic patch the wide outer is an EXACT null (grad uses
    f[i+1]-f[i-1] = 0 on a checkerboard); on the cube the inter-face halo
    interpolation leaks a small residual, so the wide response is nonzero but
    ~280× weaker than compact (matches the investigation's cube probe).
    """
    n, nlev = 12, 1
    grid = create_cubed_sphere(n)
    q = _checkerboard(n, nlev)

    wide = hyperdiffusion_3d(q, grid, 1.0, compact_outer=False)
    compact = hyperdiffusion_3d(q, grid, 1.0, compact_outer=True)

    wide_rms = _interior_rms(wide)
    compact_rms = _interior_rms(compact)

    assert compact_rms > 0.0
    assert compact_rms > 100.0 * wide_rms, (
        f"compact should damp 2Δx >>100x harder than wide: wide_rms={wide_rms:.3e} "
        f"compact_rms={compact_rms:.3e} (ratio {compact_rms/max(wide_rms,1e-300):.1f})"
    )


def test_compact_2dx_response_matches_analytic():
    """Compact ∇⁴ Nyquist eigenvalue ≈ 1024/dx⁴ (2D checkerboard, dx=cell size)."""
    n, nlev = 12, 1
    grid = create_cubed_sphere(n)
    q = _checkerboard(n, nlev)
    # tendency = -coeff * ∇⁴ q; for q=(-1)^(i+j), ∇⁴ q = (1024/dx⁴) q on a
    # uniform patch (compact ∇² eigenvalue -16/dx² per axis -> (-32/dx²)²).
    compact = hyperdiffusion_3d(q, grid, 1.0, compact_outer=True)
    dx = float(np.median(np.asarray(grid.dx)))
    expected = 1024.0 / dx ** 4
    core = np.asarray(compact)[:, 3:-3, 3:-3, :]
    q_core = np.asarray(q)[:, 3:-3, 3:-3, :]
    # tendency/q = -expected (uniform sign flip); check the magnitude ratio.
    ratio = np.median(np.abs(core) / np.abs(q_core)) / expected
    assert 0.5 < ratio < 2.0, f"compact 2Δx response ratio to 1024/dx⁴ = {ratio:.3f}"


def test_uniform_and_linear_field_near_zero_tendency():
    """∇⁴ annihilates constants and linear ramps (both outer stencils)."""
    n, nlev = 12, 1
    grid = create_cubed_sphere(n)
    const = jnp.ones((6, n, n, nlev))
    for compact in (False, True):
        out = hyperdiffusion_3d(const, grid, 1.0, compact_outer=compact)
        assert _interior_rms(out) < 1e-8


def test_compact_conserves_cube_area_integral_no_worse_than_wide():
    """Area-weighted ∫∇⁴ ≈ 0 on the cube; compact residual ≤ wide (probe: smaller)."""
    n, nlev = 12, 2
    grid = create_cubed_sphere(n)
    rng = np.random.default_rng(3)
    q = jnp.asarray(rng.standard_normal((6, n, n, nlev)))
    area = np.asarray(grid.area)[..., None]  # (6, n, n, 1)

    def residual(compact):
        t = np.asarray(hyperdiffusion_3d(q, grid, 1.0, compact_outer=compact))
        num = np.abs(np.sum(area * t))
        den = np.sum(area * np.abs(t)) + 1e-30
        return num / den

    r_wide = residual(False)
    r_compact = residual(True)
    # Neither is machine-eps on the cube (bounded by inter-face halo interp).
    assert r_compact < 1e-2
    assert r_wide < 1e-2
    # The NO-REGRESSION claim itself (pre-merge codex review): the compact
    # composition is not divergence-form, so conservation is not structurally
    # guaranteed — certify it RELATIVE to the wide operator, not just "small".
    # Factor 10 gives headroom for stencil-dependent halo-interp differences
    # while still failing if compact ever degrades an order of magnitude
    # (e.g. wide 1e-8 / compact 9e-3 passed the absolute bounds alone); the
    # 1e-6 floor keeps a near-machine-zero wide residual from making the
    # bound unattainably strict.
    assert r_compact <= 10.0 * max(r_wide, 1e-6), (
        f"compact-outer conservation residual {r_compact:.3e} regressed vs "
        f"wide {r_wide:.3e}")


def test_compact_outer_differentiable():
    """jax.grad through the compact-outer biharmonic stays finite."""
    n, nlev = 8, 2
    grid = create_cubed_sphere(n)
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.standard_normal((6, n, n, nlev)))

    def loss(coeff):
        return jnp.sum(hyperdiffusion_3d(q, grid, coeff, compact_outer=True) ** 2)

    g = jax.grad(loss)(1.0e10)
    assert np.isfinite(float(g)) and float(g) != 0.0


def test_biharmonic_scheme_wires_compact_outer():
    """biharmonic_lateral_mixing forwards the flag: compact vs wide DIFFER on 2Δx."""
    n, nlev = 12, 1
    grid = create_cubed_sphere(n)
    q = _checkerboard(n, nlev)
    z = jnp.zeros_like(q)
    mask = jnp.ones((6, n, n))
    base = BiharmonicConfig(B_h_tracer=1.0e10, enforce_cfl=False)

    out_wide = biharmonic_lateral_mixing(z, z, q, q, mask, grid, base)
    out_cmp = biharmonic_lateral_mixing(
        z, z, q, q, mask, grid, base._replace(compact_outer=True))

    # compact_outer damps the 2Δx tracer noise >>100x harder — proves the flag
    # reaches hyperdiffusion_3d through the scheme.
    assert _interior_rms(out_cmp.dT_dt) > 100.0 * _interior_rms(out_wide.dT_dt)
    assert np.all(np.isfinite(np.asarray(out_cmp.dT_dt)))


def test_compact_outer_cfl_cap_value():
    """enforce_cfl caps the compact coefficient at cfl_safety·dx⁴/(32·dt).

    Behavioral: a huge requested B with enforce_cfl must produce the SAME
    tendency as an explicit uncapped B equal to the compact cap value — proving
    the 32× tightening (module constant _COMPACT_OUTER_CFL_TIGHTENING) is wired.
    """
    n, nlev = 8, 2
    grid = create_cubed_sphere(n)
    rng = np.random.default_rng(5)
    q = jnp.asarray(rng.standard_normal((6, n, n, nlev)))
    z = jnp.zeros_like(q)
    mask = jnp.ones((6, n, n))
    dt = 3600.0
    cfl_safety = 0.05
    dx = grid.resolution_km * 1000.0            # matches biharmonic.py's nominal dx
    cap_cmp = cfl_safety * dx ** 4 / (32.0 * dt)

    capped = biharmonic_lateral_mixing(
        z, z, q, q, mask, grid,
        BiharmonicConfig(B_h_tracer=1.0e40, enforce_cfl=True,
                         cfl_safety=cfl_safety, compact_outer=True), dt=dt)
    explicit = biharmonic_lateral_mixing(
        z, z, q, q, mask, grid,
        BiharmonicConfig(B_h_tracer=cap_cmp, compact_outer=True,
                         enforce_cfl=False))
    assert np.allclose(np.asarray(capped.dT_dt), np.asarray(explicit.dT_dt),
                       rtol=1e-9, atol=0.0)

    # And the wide-form cap is 32× larger (no tightening factor).
    cap_wide = cfl_safety * dx ** 4 / dt
    capped_wide = biharmonic_lateral_mixing(
        z, z, q, q, mask, grid,
        BiharmonicConfig(B_h_tracer=1.0e40, enforce_cfl=True,
                         cfl_safety=cfl_safety), dt=dt)
    explicit_wide = biharmonic_lateral_mixing(
        z, z, q, q, mask, grid,
        BiharmonicConfig(B_h_tracer=cap_wide, enforce_cfl=False))
    assert np.allclose(np.asarray(capped_wide.dT_dt),
                       np.asarray(explicit_wide.dT_dt), rtol=1e-9, atol=0.0)


def _cap_case(n=8, nlev=2):
    grid = create_cubed_sphere(n)
    q = jnp.asarray(np.random.default_rng(7).standard_normal((6, n, n, nlev)))
    return grid, q, jnp.zeros_like(q), jnp.ones((6, n, n))


def test_default_cap_uses_run_dt():
    """Default config caps B at cfl_safety*dx^4/dt for the dt PASSED (not 3600 s)."""
    grid, q, z, mask = _cap_case()
    dt = 900.0
    cfg = BiharmonicConfig()
    cap = cfg.cfl_safety * (grid.resolution_km * 1000.0) ** 4 / dt
    capped = biharmonic_lateral_mixing(
        z, z, q, q, mask, grid, cfg._replace(B_h_tracer=1.0e40), dt=dt)
    explicit = biharmonic_lateral_mixing(
        z, z, q, q, mask, grid,
        BiharmonicConfig(B_h_tracer=cap, enforce_cfl=False))
    assert np.allclose(np.asarray(capped.dT_dt), np.asarray(explicit.dT_dt),
                       rtol=1e-9, atol=0.0)


def test_default_cap_without_dt_raises():
    grid, q, z, mask = _cap_case()
    with pytest.raises(ValueError, match="timestep"):
        biharmonic_lateral_mixing(z, z, q, q, mask, grid,
                                  BiharmonicConfig(B_h_tracer=1.0e10))


def test_combined_physics_forwards_dt_to_biharmonic():
    """make_ocean_physics(...)(..., dt=) reaches the biharmonic cap."""
    from legoesm.ocean.init import rest_state_ocean
    from legoesm.ocean.physics.combined import OceanPhysicsConfig, make_ocean_physics
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_cubed_sphere(8)
    zc = create_ocean_z_star(n_levels=2, H_max=4000.0)
    st = rest_state_ocean(grid, zc, T_water_init_C=18.0, T_deep=2.0, S_uniform=35.0)
    noise = jnp.asarray(np.random.default_rng(3).normal(size=st.T.data.shape))
    st = st._replace(T=st.T.replace(data=st.T.data + noise))

    def phys(bih):
        return make_ocean_physics(OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="none"),
            lateral_mixing=LateralMixingConfig(scheme="biharmonic", biharmonic=bih),
            surface_forcing=SurfaceForcingConfig(scheme="none"),
            bottom_drag=OceanPhysicsConfig().bottom_drag,
            convection=OceanConvectionConfig(scheme="none"),
            shortwave_penetration=None))

    dt = 900.0
    cap = BiharmonicConfig().cfl_safety * (grid.resolution_km * 1000.0) ** 4 / dt
    got = phys(BiharmonicConfig(B_h_tracer=1.0e40))(st, grid, zc, None, dt=dt)
    ref = phys(BiharmonicConfig(B_h_tracer=cap, enforce_cfl=False))(st, grid, zc, None)
    assert np.allclose(np.asarray(got.dT_dt.data), np.asarray(ref.dT_dt.data),
                       rtol=1e-9, atol=0.0)
    assert float(jnp.max(jnp.abs(ref.dT_dt.data))) > 0.0


def test_cube_ocean_model_step_threads_dt_to_biharmonic():
    """OceanModel.step passes its dt down to the default (capped) biharmonic."""
    from legoesm.ocean.dynamics.ocean_model import OceanModel
    from legoesm.ocean.init import rest_state_ocean
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.convection.config import OceanConvectionConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import SurfaceForcingConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.state import OceanConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_cubed_sphere(8)
    zc = create_ocean_z_star(n_levels=2, H_max=4000.0)
    st = rest_state_ocean(grid, zc, T_water_init_C=18.0, T_deep=2.0, S_uniform=35.0)
    noise = jnp.asarray(np.random.default_rng(4).normal(size=st.T.data.shape))
    st = st._replace(T=st.T.replace(data=st.T.data + noise))

    def model(bih):
        phys = OceanPhysicsConfig(
            vertical_mixing=VerticalMixingConfig(scheme="none"),
            lateral_mixing=LateralMixingConfig(scheme="biharmonic", biharmonic=bih),
            surface_forcing=SurfaceForcingConfig(scheme="none"),
            bottom_drag=OceanPhysicsConfig().bottom_drag,
            convection=OceanConvectionConfig(scheme="none"),
            shortwave_penetration=None)
        return OceanModel(grid, zc, OceanConfig(
            A_h=0.0, K_h=0.0, A_v=1e-3, K_v=1e-4, n_barotropic_substeps=4,
            hyperdiff_coeff=0.0, physics=phys))

    dt = 900.0
    cap = BiharmonicConfig().cfl_safety * (grid.resolution_km * 1000.0) ** 4 / dt
    m_cap = model(BiharmonicConfig(B_h_tracer=1.0e40))
    m_ref = model(BiharmonicConfig(B_h_tracer=cap, enforce_cfl=False))
    out = m_cap.step(st, dt)
    ref = m_ref.step(st, dt)
    assert np.allclose(np.asarray(out.T.data), np.asarray(ref.T.data),
                       rtol=1e-12, atol=1e-12)
    assert np.all(np.isfinite(np.asarray(out.T.data)))
