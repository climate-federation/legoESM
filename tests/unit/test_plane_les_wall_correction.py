"""LES boundary-layer wall correction tests for plane Smag.

Pins the Mason 1989 / Pope 2000 §10.4 mixing-length cap
``l_m = min(c_s · Δ, κ · z)``. Without this cap, ``K_m`` at the
first cell from the wall is far too large; with it, ``K_m → 0`` at
the surface as ``κ · z → 0``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    _compute_smagorinsky_K_m_plane,
)
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import (
    create_height_coordinate, create_stretched_height_coordinate,
)

jax.config.update("jax_enable_x64", True)


def _grid(nx=4, ny=4, nlev=10):
    return create_plane_grid(
        nx=nx, ny=ny, nlev=nlev, dx=2_000.0, dy=2_000.0,
        dtype=jnp.float64,
    )


def test_K_m_surface_layer_capped_by_wall():
    """Linear vertical shear ``u(z) = a · z`` with ``a = 0.01 1/s``;
    K_m at the lowest level should equal ``(κ · z_sfc)² · |a|`` (the
    wall-corrected mixing length), NOT the Smag form ``(c_s · Δ)² ·
    |a|`` (which would be ~100× larger on this resolution)."""
    grid = _grid()
    hc = create_stretched_height_coordinate(
        n_levels=grid.nlev, H=2_000.0, dz_sfc=50.0,
    )
    # u(z) = a · z (a in 1/s); v = w = 0.
    a = 0.01
    u = a * jnp.broadcast_to(
        hc.z_full[None, None, :], (grid.ny, grid.nx, grid.nlev),
    )
    v = jnp.zeros_like(u)
    w = jnp.zeros((grid.ny, grid.nx, grid.nlev + 1))
    c_s = 0.2
    K_m = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, c_s=c_s)

    # Lowest centre = z_full[-1] ≈ 0.5 · dz_sfc = 25 m.
    z_sfc = float(hc.z_full[-1])
    # Strain magnitude: pure ∂u/∂z = a → |S|² = 2 · 2 · S_13²
    # = 2·(2·(0.5·a)²) = a². |S| = a.
    expected_l_wall = 0.4 * z_sfc   # κ · z
    # The Smag mixing length is (c_s · (dx · dy · dz_sfc)^(1/3))
    # = 0.2 · (2000 · 2000 · 50)^(1/3) ≈ 0.2 · 271.4 ≈ 54.3 m;
    # wall is 0.4 · 25 = 10 m — wall caps it.
    delta = (grid.dx * grid.dy * float(hc.dz[-1])) ** (1.0 / 3.0)
    l_smag = c_s * delta
    assert expected_l_wall < l_smag, (
        f"test setup broken: wall length {expected_l_wall:.2f} "
        f"≥ smag length {l_smag:.2f}; wall cap is inactive here"
    )
    expected_K_sfc = expected_l_wall ** 2 * a
    K_sfc = float(jnp.max(K_m[..., -1]))
    np.testing.assert_allclose(
        K_sfc, expected_K_sfc, rtol=5.0e-2, atol=1.0e-8,
    )


def test_K_m_aloft_uses_smag_length_not_wall():
    """At z far above the surface, κ · z >> c_s · Δ so the Smag length
    governs — K_m matches the un-capped Smag value."""
    grid = _grid()
    H = 10_000.0
    hc = create_height_coordinate(grid.nlev, H=H)
    a = 0.01
    u = a * jnp.broadcast_to(
        hc.z_full[None, None, :], (grid.ny, grid.nx, grid.nlev),
    )
    v = jnp.zeros_like(u)
    w = jnp.zeros((grid.ny, grid.nx, grid.nlev + 1))
    c_s = 0.2
    K_m = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, c_s=c_s)

    # Top centre: z ~ H - dz/2 ≈ 9.5 km. κ · z ≈ 3800 m. Δ uses
    # dz = H/nlev = 1000 m → c_s · Δ = 0.2 · (2000·2000·1000)^(1/3)
    # ≈ 0.2 · 1587 ≈ 317 m. Smag length much smaller than wall →
    # Smag form dominates.
    delta_top = (
        grid.dx * grid.dy * float(hc.dz[0])
    ) ** (1.0 / 3.0)
    expected_K_top = (c_s * delta_top) ** 2 * a
    K_top = float(jnp.max(K_m[..., 0]))
    np.testing.assert_allclose(
        K_top, expected_K_top, rtol=5.0e-2, atol=1.0e-8,
    )


def test_K_m_zero_at_surface_in_limit_dz_sfc_to_zero():
    """As dz_sfc → 0, z_full[-1] = 0.5 · dz_sfc → 0, so the wall
    length κ · z → 0 and K_m at the lowest level → 0 regardless of
    strain magnitude."""
    grid = _grid()
    nlev = grid.nlev
    H = 2_000.0
    K_at_sfc = []
    for dz_sfc in (10.0, 1.0, 0.1):
        hc = create_stretched_height_coordinate(
            n_levels=nlev, H=H, dz_sfc=dz_sfc,
        )
        u = 1.0 * jnp.broadcast_to(
            hc.z_full[None, None, :],
            (grid.ny, grid.nx, grid.nlev),
        )
        v = jnp.zeros_like(u)
        w = jnp.zeros((grid.ny, grid.nx, grid.nlev + 1))
        K_m = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, c_s=0.2)
        K_at_sfc.append(float(jnp.max(K_m[..., -1])))
    # Should be monotonically decreasing (each factor-10 dz_sfc → ~100×
    # smaller K_m at the surface since K ∝ z²).
    assert K_at_sfc[0] > K_at_sfc[1] > K_at_sfc[2]
    # Ratio approximately 100 between consecutive (10× smaller z → 100×
    # smaller K). Allow generous tolerance because strain magnitude
    # is computed from one-sided diff at the boundary so |S| also shifts.
    ratio_1 = K_at_sfc[0] / K_at_sfc[1]
    ratio_2 = K_at_sfc[1] / K_at_sfc[2]
    assert 50.0 < ratio_1 < 200.0, (
        f"K_m surface scaling not z² as dz_sfc shrinks; ratio_1={ratio_1:.1f}"
    )
    assert 50.0 < ratio_2 < 200.0


def test_K_m_no_wall_cap_dosmagor_uses_full_smag_length():
    """SAM-faithful CRM path (iter-46): ``wall_damping=False`` ⇒ ``smix=grd``
    (no von-Kármán cap). The lowest cell then uses the FULL ``c_s·Δ`` length
    (NOT ``κz``) ⇒ K_sfc equals the un-capped Smag value, strictly LARGER than
    the wall-capped default. Matches SAM ``dosmagor`` (``tke_full.f90:286``)."""
    grid = _grid()
    hc = create_stretched_height_coordinate(
        n_levels=grid.nlev, H=2_000.0, dz_sfc=50.0,
    )
    a = 0.01
    u = a * jnp.broadcast_to(
        hc.z_full[None, None, :], (grid.ny, grid.nx, grid.nlev),
    )
    v = jnp.zeros_like(u)
    w = jnp.zeros((grid.ny, grid.nx, grid.nlev + 1))
    c_s = 0.19
    K_capped = _compute_smagorinsky_K_m_plane(
        u, v, w, grid, hc, c_s=c_s, wall_damping=True)
    K_nocap = _compute_smagorinsky_K_m_plane(
        u, v, w, grid, hc, c_s=c_s, wall_damping=False)
    delta_sfc = (grid.dx * grid.dy * float(hc.dz[-1])) ** (1.0 / 3.0)
    expected_K_nocap = (c_s * delta_sfc) ** 2 * a
    np.testing.assert_allclose(
        float(jnp.max(K_nocap[..., -1])), expected_K_nocap,
        rtol=5.0e-2, atol=1.0e-8)
    # cap was active at the surface ⇒ no-cap K strictly larger there
    assert float(jnp.max(K_nocap[..., -1])) > float(jnp.max(K_capped[..., -1]))
    # ALOFT the cap was inactive ⇒ both identical (gate only touches the surface)
    np.testing.assert_allclose(
        np.asarray(K_nocap[..., 0]), np.asarray(K_capped[..., 0]),
        rtol=0.0, atol=1.0e-12)


def test_crm_run_scripts_set_dosmagor_no_wall_cap():
    """codex iter-46 G: guard that all three SAM-faithful CRM drivers pass
    ``smagorinsky_wall_damping=False`` to the kernel (SAM dosmagor smix=grd).
    A regression that drops the setting (reverting to the von-Kármán cap) would
    silently make the near-surface SGS mixing un-SAM-faithful — fail loudly
    here instead."""
    import re
    from pathlib import Path
    # namespace-safe repo root (legoesm is a PEP-420 namespace pkg, no __file__)
    repo = Path(__file__).resolve().parents[2]
    for script in ("run_gate_plane.py", "run_lba_plane.py",
                   "run_rcemip_plane.py"):
        src = (repo / "scripts" / "run" / script).read_text()
        # gate/lba hardcode the no-cap setting; run_rcemip_plane.py exposes it
        # through ``--smag-wall-damping`` whose default is the SAM-faithful
        # no-cap value (wired via ``smagorinsky_wall_damping=args.smag_wall_damping``).
        hardcoded = "smagorinsky_wall_damping=False" in src
        arg_block = re.search(
            r'add_argument\(\s*"--smag-wall-damping".*?\)', src, re.DOTALL)
        arg_default_off = (
            "smagorinsky_wall_damping=args.smag_wall_damping" in src
            and arg_block is not None
            and "default=False" in arg_block.group(0))
        assert hardcoded or arg_default_off, (
            f"{script} must reach the kernel with the SAM dosmagor no-wall-cap "
            f"setting (smagorinsky_wall_damping=False directly, or via "
            f"--smag-wall-damping default=False)")


def test_rest_state_K_m_zero_with_wall_correction():
    """Wall correction must not break the bit-exact zero on rest
    state (no strain → K_m = 0 regardless of length scale)."""
    grid = _grid()
    hc = create_stretched_height_coordinate(
        n_levels=grid.nlev, H=2_000.0, dz_sfc=50.0,
    )
    u = jnp.zeros((grid.ny, grid.nx, grid.nlev))
    v = jnp.zeros((grid.ny, grid.nx, grid.nlev))
    w = jnp.zeros((grid.ny, grid.nx, grid.nlev + 1))
    K_m = _compute_smagorinsky_K_m_plane(u, v, w, grid, hc, c_s=0.2)
    assert float(jnp.max(jnp.abs(K_m))) == 0.0
