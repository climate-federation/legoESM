"""Cross-grid NH regression test for iter-7..9 anchored mass fixers.

Parallel to ``test_mass_conservation_anchored.py`` (hydrostatic PE) and
``test_sw_mass_conservation_anchored.py`` (SW): runs a short DCMIP-2025
TC1 integration on every non-hydrostatic dycore and asserts the global
dry-mass integral drifts at the fp64 floor (≤ 1e-10).

A regression here means one of the NH fixers (iter-7 cube
``compressible_euler_cdgrid``, iter-8 MPAS
``compressible_euler_mpas`` + new helpers in ``core/conservation.py``,
iter-9 spectral ``compressible_euler_spectral``) has been undone.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

DRIFT_TOL = 1e-12
N_STEPS = 20


def _rel_drift(m0: float, m1: float) -> float:
    return abs(m1 - m0) / max(abs(m0), 1.0)


def test_nh_mass_conservation_cubed_sphere():
    """Cubed-sphere NH (iter-7): anchored dry-mass via
    ``CompressibleEulerConfig.fix_mass=True, anchor_mass_to_initial=True``.
    """
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerModel,
        CDGridCompressibleEulerConfig,
    )
    from tests.test_cases.dcmip2025 import dcmip25_tc1_init
    from legoesm.core.conservation import compute_nh_dry_mass

    grid = create_cubed_sphere(12)
    state, hcoord, tmetric = dcmip25_tc1_init(grid, n_levels=10)
    cfg = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=10, semi_implicit_acoustic=True,
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = CDGridCompressibleEulerModel(grid, hcoord, tmetric, cfg)

    def _mass(s):
        return float(compute_nh_dry_mass(
            s.rho_prime.data, hcoord, tmetric, grid,
        ))

    m0 = _mass(state)
    for _ in range(N_STEPS):
        state = model.step(state, 5.0)
    assert _rel_drift(m0, _mass(state)) < DRIFT_TOL


def test_nh_mass_conservation_mpas():
    """MPAS NH (iter-8): new ``compute_nh_dry_mass_mpas`` +
    ``fix_mass_nonhydrostatic_mpas`` in ``core/conservation.py``.
    """
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas import (
        MPASCompressibleEulerModel, MPASCompressibleEulerConfig,
    )
    from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_1_mpas import (
        dcmip25_tc1_init_mpas,
    )
    from legoesm.core.conservation import compute_nh_dry_mass_mpas

    mesh = create_voronoi_mesh(4)
    state, hcoord, tmetric = dcmip25_tc1_init_mpas(mesh, n_levels=10)
    cfg = MPASCompressibleEulerConfig(
        n_acoustic_substeps=10,
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = MPASCompressibleEulerModel(mesh, hcoord, tmetric, cfg)

    def _mass(s):
        return float(compute_nh_dry_mass_mpas(
            s.rho_prime.data, hcoord, tmetric, mesh,
        ))

    m0 = _mass(state)
    for _ in range(N_STEPS):
        state = model.step(state, 5.0)
    assert _rel_drift(m0, _mass(state)) < DRIFT_TOL


def test_nh_mass_conservation_spectral():
    """Spectral NH (iter-9): ``rho_prime_hat[0,:] += Δρ·sqrt(4π)``."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
        SpectralCompressibleEulerModel, SpectralNHConfig,
        dcmip25_tc1_init_spectral,
    )

    grid = create_gaussian_grid(21)
    state, hcoord, tmetric = dcmip25_tc1_init_spectral(grid, n_levels=10)
    cfg = SpectralNHConfig(
        n_acoustic_substeps=10, semi_implicit_acoustic=True,
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = SpectralCompressibleEulerModel(
        grid, hcoord, tmetric, cfg, allow_unsupported_backend=True,
    )

    m0 = float(model.compute_dry_mass(state))
    for _ in range(N_STEPS):
        state = model.step(state, 5.0)
    m_final = float(model.compute_dry_mass(state))
    assert _rel_drift(m0, m_final) < DRIFT_TOL


# ---------------------------------------------------------------------------
# iter-34: long-run drift check (parallel to iter-33 hydro PE / iter-34 SW).
# ---------------------------------------------------------------------------

def test_long_run_nh_mass_conservation_cubed_sphere():
    """100-step cube NH: anchored dry mass must NOT random-walk."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerModel, CDGridCompressibleEulerConfig,
    )
    from tests.test_cases.dcmip2025 import dcmip25_tc1_init
    from legoesm.core.conservation import compute_nh_dry_mass

    grid = create_cubed_sphere(12)
    state, hcoord, tmetric = dcmip25_tc1_init(grid, n_levels=10)
    cfg = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=10, semi_implicit_acoustic=True,
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = CDGridCompressibleEulerModel(grid, hcoord, tmetric, cfg)

    def _mass(s):
        return float(compute_nh_dry_mass(
            s.rho_prime.data, hcoord, tmetric, grid,
        ))

    m0 = _mass(state)
    for _ in range(100):
        state = model.step(state, 5.0)
    drift = _rel_drift(m0, _mass(state))
    assert drift < 1e-12, (
        f"cube NH 100-step dry-mass drift {drift:.2e} exceeds 1e-12"
    )


def test_long_run_nh_mass_conservation_mpas():
    """iter-60: 100-step MPAS NH parallel to iter-34 cube NH long-run.

    Covers the MPAS code path on a longer integration than the iter-16
    20-step short-run gate.  Direct measurement: drift = 3.89e-16
    over 100 steps on level-4 Voronoi mesh + 10 levels.
    """
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas import (
        MPASCompressibleEulerModel, MPASCompressibleEulerConfig,
    )
    from tests.atmosphere.nonhydrostatic.test_cases.dcmip2025.test_case_1_mpas import (
        dcmip25_tc1_init_mpas,
    )
    from legoesm.core.conservation import compute_nh_dry_mass_mpas

    mesh = create_voronoi_mesh(4)
    state, hcoord, tmetric = dcmip25_tc1_init_mpas(mesh, n_levels=10)
    cfg = MPASCompressibleEulerConfig(
        n_acoustic_substeps=10,
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = MPASCompressibleEulerModel(mesh, hcoord, tmetric, cfg)

    def _mass(s):
        return float(compute_nh_dry_mass_mpas(
            s.rho_prime.data, hcoord, tmetric, mesh,
        ))

    m0 = _mass(state)
    for _ in range(100):
        state = model.step(state, 5.0)
    drift = _rel_drift(m0, _mass(state))
    assert drift < 1e-12, (
        f"MPAS NH 100-step dry-mass drift {drift:.2e} exceeds 1e-12"
    )


def test_long_run_nh_mass_conservation_spectral():
    """iter-61: 100-step spectral NH long-run guard.

    Parallel to iter-33/34/60 long-run guards on cube PE / cube SW /
    cube NH / MPAS NH.  Validates iter-9 spectral NH anchored fixer
    (`rho_prime_hat[0,:] += Δρ·sqrt(4π)`).  Direct measurement: drift
    = 1.94e-16 on T21 spectral NH with DCMIP-2025 TC1 init.
    """
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.atmosphere.dynamics.gcm.spectral_nh import (
        SpectralCompressibleEulerModel, SpectralNHConfig,
        dcmip25_tc1_init_spectral,
    )

    grid = create_gaussian_grid(21)
    state, hcoord, tmetric = dcmip25_tc1_init_spectral(grid, n_levels=10)
    cfg = SpectralNHConfig(
        n_acoustic_substeps=10, semi_implicit_acoustic=True,
        fix_mass=True, anchor_mass_to_initial=True,
    )
    model = SpectralCompressibleEulerModel(
        grid, hcoord, tmetric, cfg, allow_unsupported_backend=True,
    )

    m0 = float(model.compute_dry_mass(state))
    for _ in range(100):
        state = model.step(state, 5.0)
    m_final = float(model.compute_dry_mass(state))
    drift = _rel_drift(m0, m_final)
    assert drift < 1e-12, (
        f"spectral NH 100-step dry-mass drift {drift:.2e} exceeds 1e-12"
    )
