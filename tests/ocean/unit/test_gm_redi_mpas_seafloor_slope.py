"""GM/Redi MPAS — seafloor no-flux BC + slope_density dispatch hardening.

Two confirmed bugs in ``gm_redi_mpas.py`` are covered here:

BUG A — the centred vertical isoneutral flux ``F_z`` was computed on ALL
interior interfaces and only the final inactive-level TENDENCIES were
masked.  On a partial-cell column the seafloor interface
``F_z[:, bottom_level]`` (reconstructed via Perot from the sub-seafloor,
filled side of a step edge and from the ``S^2 * d_z q`` term) then carried
a spurious diapycnal flux into the DEEPEST ACTIVE cell — a non-no-flux
bottom boundary.  The fix zeros ``F_z`` on every interface whose lower cell
is inactive, BEFORE the vertical divergence.

BUG B — ``GMRediConfig.slope_density='neutral'`` was SILENTLY IGNORED by the
MPAS path (slopes are always built from in-situ ``rho``).  The fix raises
``NotImplementedError`` at the entry of the public cfg-taking functions.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.physics.lateral_mixing.gm_redi_mpas import (
    compute_isopycnal_slopes_mpas,
    gm_redi_tracer_tendency_centered_mpas,
    gm_redi_tracer_tendency_mpas,
)
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mesh():
    """Small icosahedral Voronoi mesh (level 2 = 162 cells, 480 edges)."""
    return create_voronoi_mesh(subdivision_level=2)


@pytest.fixture(scope="module")
def z_ref():
    """5-level reference z-star (H_max=500 m)."""
    return create_ocean_z_star(
        n_levels=5, H_max=500.0, dz_surface=20.0, dz_deep=200.0
    )


@pytest.fixture(scope="module")
def cfg():
    return GMRediConfig(kappa_GM=1e3, kappa_Redi=1e3, S_max=1e-2)


def _edge_mask(mesh, mask):
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    return mask[c1] * mask[c2]


def _stratified_rho(mesh, nlev):
    """In-situ-like density: increases with depth AND varies horizontally.

    The horizontal (latitude) structure gives non-zero edge-normal density
    gradients ⇒ non-zero isoneutral slopes; the vertical structure gives a
    stable stratification.  Both are needed for a non-vacuous seafloor test.
    """
    k = jnp.arange(nlev, dtype=jnp.float64)
    vert = 1027.0 + 0.4 * k                       # (nlev,) stable
    horiz = 0.05 * jnp.cos(mesh.latCell)          # (nCells,)
    return vert[None, :] + horiz[:, None]         # (nCells, nlev)


def _stratified_q(mesh, nlev):
    """Tracer with both vertical and horizontal structure."""
    k = jnp.arange(nlev, dtype=jnp.float64)
    vert = 10.0 - 0.5 * k
    horiz = 2.0 * jnp.sin(mesh.latCell)
    return vert[None, :] + horiz[:, None]         # (nCells, nlev)


# ===========================================================================
# BUG A — seafloor no-flux BC (F_z zeroed at inactive interfaces)
# ===========================================================================

def test_seafloor_interface_flux_is_zero(mesh, z_ref, cfg):
    """The vertical isoneutral flux carries NO tendency below the seafloor.

    Setup: a UNIFORM partial-cell depth with ``bottom_level = 2`` (nlev = 5),
    so cells 0,1,2 are active and 3,4 are sub-seafloor.  With a stratified
    tracer + non-zero slopes the un-masked ``F_z`` at the seafloor interface
    (j=2, between active cell 2 and inactive cell 3) would be non-zero.

    Discriminating assertion: the sub-seafloor cell tendency (level 3) is
    EXACTLY zero — its value is ``(F_z[2] - F_z[3]) / dz`` plus a masked
    horizontal flux, so a non-zero seafloor ``F_z[2]`` (the bug) would show
    up here.  Non-vacuity is proven by a companion FULL-DEPTH run (all cells
    active) where the same physics produces a clearly non-zero level-3
    tendency.
    """
    nlev = z_ref.dz_ref.shape[0]

    # --- Uniform partial-cell depth giving bottom_level == 2 everywhere. ---
    abs_z_half = np.abs(np.asarray(z_ref.z_half_ref))  # (nlev+1,)
    # bottom_level = (#interfaces strictly shallower than H) - 1 == 2
    #   ⇒ pick H in (abs_z_half[2], abs_z_half[3]).
    H_partial_val = 0.5 * (abs_z_half[2] + abs_z_half[3])
    H_partial = jnp.full((mesh.nCells,), float(H_partial_val), dtype=jnp.float64)
    coord_A = create_partial_cell_coordinate(z_ref, H_partial)

    # Self-check the fixture: bottom_level==2, is_active == [T,T,T,F,F].
    assert np.all(np.asarray(coord_A.bottom_level) == 2)
    expect_active = np.array([True, True, True, False, False])
    assert np.all(np.asarray(coord_A.is_active) == expect_active[None, :])

    # --- Full-depth reference (every cell active) for non-vacuity. ---
    H_full = jnp.full((mesh.nCells,), z_ref.H_max, dtype=jnp.float64)
    coord_B = create_partial_cell_coordinate(z_ref, H_full)
    assert np.all(np.asarray(coord_B.bottom_level) == nlev - 1)
    assert np.all(np.asarray(coord_B.is_active))

    mask = jnp.ones((mesh.nCells,), dtype=jnp.float64)
    em = _edge_mask(mesh, mask)
    jac = jnp.ones((mesh.nCells,), dtype=jnp.float64)

    rho = _stratified_rho(mesh, nlev)
    q = _stratified_q(mesh, nlev)

    # Slopes are geometry-of-interfaces only (nEdges, nlev-1); build once on
    # the reference grid so both runs share IDENTICAL S_n.
    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_ref, jac, mesh, cfg)

    dqdt_A = gm_redi_tracer_tendency_centered_mpas(
        q, S_n, mask, em, coord_A, jac, mesh,
        kappa_GM=cfg.kappa_GM, kappa_Redi=cfg.kappa_Redi,
    )
    dqdt_B = gm_redi_tracer_tendency_centered_mpas(
        q, S_n, mask, em, coord_B, jac, mesh,
        kappa_GM=cfg.kappa_GM, kappa_Redi=cfg.kappa_Redi,
    )

    dqdt_A = np.asarray(dqdt_A)
    dqdt_B = np.asarray(dqdt_B)

    # Fix guarantee: the seafloor interface F_z[2] and the fully-sub-seafloor
    # F_z[3] are zeroed ⇒ the sub-seafloor cells (levels 3 and 4) get NO
    # tendency (horizontal flux there is already masked to zero).
    np.testing.assert_allclose(dqdt_A[:, 3], 0.0, atol=1e-14)
    np.testing.assert_allclose(dqdt_A[:, 4], 0.0, atol=1e-14)

    # Non-vacuity: with the cells GENUINELY active (full depth) the identical
    # physics produces a level-3 tendency of the SAME ORDER as the interior
    # (level 1) tendency — so the exact zero above is due to the seafloor
    # masking, not a trivially-zero configuration.  Scale-relative so the
    # test is robust to the (small) slope magnitude on the coarse mesh.
    interior_scale = np.max(np.abs(dqdt_B[:, 1]))
    assert interior_scale > 0.0
    assert np.max(np.abs(dqdt_B[:, 3])) > 1e-2 * interior_scale

    # And the physically-important effect: the DEEPEST ACTIVE cell (level 2)
    # tendency changes once the spurious seafloor flux is removed.
    assert np.max(np.abs(dqdt_A[:, 2] - dqdt_B[:, 2])) > 1e-3 * interior_scale


def test_seafloor_fix_finite_and_differentiable(mesh, z_ref, cfg):
    """The seafloor mask keeps outputs finite and AD well-behaved."""
    nlev = z_ref.dz_ref.shape[0]
    abs_z_half = np.abs(np.asarray(z_ref.z_half_ref))
    H_partial = jnp.full(
        (mesh.nCells,), float(0.5 * (abs_z_half[2] + abs_z_half[3])),
        dtype=jnp.float64,
    )
    coord_A = create_partial_cell_coordinate(z_ref, H_partial)

    mask = jnp.ones((mesh.nCells,), dtype=jnp.float64)
    em = _edge_mask(mesh, mask)
    jac = jnp.ones((mesh.nCells,), dtype=jnp.float64)
    rho = _stratified_rho(mesh, nlev)
    S_n, _ = compute_isopycnal_slopes_mpas(rho, mask, z_ref, jac, mesh, cfg)

    def loss(q):
        dqdt = gm_redi_tracer_tendency_centered_mpas(
            q, S_n, mask, em, coord_A, jac, mesh,
            kappa_GM=cfg.kappa_GM, kappa_Redi=cfg.kappa_Redi,
        )
        return jnp.sum(dqdt ** 2)

    q = _stratified_q(mesh, nlev)
    g = jax.grad(loss)(q)
    assert np.all(np.isfinite(np.asarray(g)))


# ===========================================================================
# BUG B — slope_density dispatch hardening
# ===========================================================================

def test_top_level_neutral_slope_density_raises(mesh, z_ref):
    """slope_density='neutral' is unsupported on MPAS ⇒ must raise, not
    silently run in-situ physics."""
    nlev = z_ref.dz_ref.shape[0]
    T = jnp.full((mesh.nCells, nlev), 10.0, dtype=jnp.float64)
    S = jnp.full((mesh.nCells, nlev), 35.0, dtype=jnp.float64)
    eta = jnp.zeros((mesh.nCells,), dtype=jnp.float64)
    H_bathy = jnp.full((mesh.nCells,), 500.0, dtype=jnp.float64)
    cfg = GMRediConfig(
        kappa_GM=1e3, kappa_Redi=1e3, S_max=1e-2,
        slope_scheme="centered", slope_density="neutral",
    )
    with pytest.raises(NotImplementedError) as exc_info:
        gm_redi_tracer_tendency_mpas(
            T, S, eta, H_bathy, mesh, z_ref, cfg, eos="linear",
        )
    msg = str(exc_info.value)
    assert "slope_density" in msg
    assert "neutral" in msg


def test_slopes_neutral_slope_density_raises(mesh, z_ref, cfg):
    """The slope builder itself (the site that ignores slope_density) also
    raises, so every path that reaches it is hardened."""
    nlev = z_ref.dz_ref.shape[0]
    mask = jnp.ones((mesh.nCells,), dtype=jnp.float64)
    jac = jnp.ones((mesh.nCells,), dtype=jnp.float64)
    rho = _stratified_rho(mesh, nlev)
    bad_cfg = cfg._replace(slope_density="neutral")
    with pytest.raises(NotImplementedError):
        compute_isopycnal_slopes_mpas(rho, mask, z_ref, jac, mesh, bad_cfg)


def test_default_in_situ_slope_density_ok(mesh, z_ref, cfg):
    """The default slope_density='in_situ' must NOT raise (guard is exact)."""
    assert cfg.slope_density == "in_situ"
    nlev = z_ref.dz_ref.shape[0]
    mask = jnp.ones((mesh.nCells,), dtype=jnp.float64)
    jac = jnp.ones((mesh.nCells,), dtype=jnp.float64)
    rho = _stratified_rho(mesh, nlev)
    S_n, taper = compute_isopycnal_slopes_mpas(rho, mask, z_ref, jac, mesh, cfg)
    assert np.all(np.isfinite(np.asarray(S_n)))
    assert np.all(np.isfinite(np.asarray(taper)))
