"""Cube (FC-Gram) partial-cell substrate tests.

Exercises the ``OceanPartialCellCoordinate`` path through the cubed-sphere
FC ocean PE (``ocean_baroclinic_tendencies_fc``):

1. Flat-bottom (H_bathy == H_max) is BIT-EXACT vs the pure z* coord — the
   partial-cell branch must not perturb the legacy result.
2. Below-seafloor cells (``is_active == False``) carry exactly-zero
   momentum/tracer tendencies, and the partial-cell column thickness sums to
   the local bathymetry.
3. On sloped bathymetry the partial-cell result DIFFERS from z* (the
   substrate is not a silent no-op).
4. The partial-cell path stays differentiable (``jax.grad`` finite).

Run under ``JAX_ENABLE_X64=1`` (the FC backend builds a float64 config).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_layer_thickness,
)
from legoesm.ocean.init import rest_state_ocean
from legoesm.ocean.state import OceanConfig
from legoesm.ocean.dynamics.ocean_pe_fc import ocean_baroclinic_tendencies_fc
from legoesm.core.operators_fc import build_fc_config

N = 8
NLEV = 5
HMAX = 4000.0
_FIELDS_3D = ("du_dt", "dv_dt", "dT_dt", "dS_dt")


def _cfg():
    # All horizontal/vertical diffusion off → tendency = advection + PGF only,
    # so the partial-cell PGF/thickness change is isolated.
    return OceanConfig(A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0)


def _fc():
    return build_fc_config(d=2, C=4, degree=5)


def _state(grid, z_coord, H_bathy_2d, t_pert_amp=2.0):
    """Rest state on an all-ocean cube with a prescribed bathymetry and a
    horizontal temperature perturbation (so the baroclinic PGF is nonzero —
    ``rest_state_ocean`` alone is horizontally uniform => PGF == 0)."""
    st = rest_state_ocean(grid, z_coord, H_max=HMAX)
    land = jnp.ones((6, N, N), dtype=st.land_mask.data.dtype)
    lon = np.asarray(grid.lon)[..., np.newaxis]            # (6, N, N, 1)
    T_pert = np.asarray(st.T.data) + t_pert_amp * np.sin(lon)
    return st._replace(
        land_mask=st.land_mask.replace(data=land),
        H_bathy=st.H_bathy.replace(data=jnp.asarray(H_bathy_2d)),
        T=st.T.replace(data=jnp.asarray(T_pert, dtype=st.T.data.dtype)),
    )


def _sloped_bathy(grid):
    lat = np.asarray(grid.lat)                              # (6, N, N) radians
    return (1000.0 + (HMAX - 1000.0) * 0.5 * (1.0 + np.sin(2.0 * lat))).astype(
        np.float64
    )


def test_flat_bottom_bit_exact():
    """H_bathy == H_max everywhere: partial-cell coord reproduces z* exactly."""
    grid = create_cubed_sphere(N)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = jnp.full((6, N, N), HMAX, dtype=jnp.float64)
    pc = create_partial_cell_coordinate(zc, H)
    st = _state(grid, zc, H)
    fc, cfg = _fc(), _cfg()

    t_z = ocean_baroclinic_tendencies_fc(st, grid, zc, fc, cfg)
    t_p = ocean_baroclinic_tendencies_fc(st, grid, pc, fc, cfg)

    for name in _FIELDS_3D + ("deta_dt",):
        a = np.asarray(getattr(t_z, name).data)
        b = np.asarray(getattr(t_p, name).data)
        np.testing.assert_array_equal(
            a, b, err_msg=f"{name}: flat-bottom partial cells must match z* BIT-EXACT"
        )


def test_below_seafloor_masked_and_thickness_sums_to_bathy():
    """Below-seafloor tendencies are exactly zero; Σ h_k == H_bathy."""
    grid = create_cubed_sphere(N)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _state(grid, zc, jnp.asarray(H))

    inactive = ~np.asarray(pc.is_active)
    assert inactive.any(), "sloped bathy should leave some below-seafloor cells"

    t_p = ocean_baroclinic_tendencies_fc(st, grid, pc, _fc(), _cfg())
    for name in _FIELDS_3D:
        vals = np.asarray(getattr(t_p, name).data)[inactive]
        assert np.max(np.abs(vals)) == 0.0, (
            f"{name}: below-seafloor tendency must be exactly zero, "
            f"got max|.|={np.max(np.abs(vals)):.3e}"
        )

    h_k = compute_layer_thickness(jnp.zeros((6, N, N)), jnp.asarray(H), pc)
    col_sum = np.asarray(h_k).sum(axis=-1)
    assert np.allclose(col_sum, H, rtol=1e-4, atol=1e-2), (
        "partial-cell column thickness must sum to the local bathymetry"
    )


def test_partial_cells_change_result_on_sloped_bottom():
    """On real (sloped) bathymetry the partial-cell PGF differs from z*."""
    grid = create_cubed_sphere(N)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _state(grid, zc, jnp.asarray(H))
    fc, cfg = _fc(), _cfg()

    t_z = ocean_baroclinic_tendencies_fc(st, grid, zc, fc, cfg)
    t_p = ocean_baroclinic_tendencies_fc(st, grid, pc, fc, cfg)
    assert not jnp.allclose(t_z.du_dt.data, t_p.du_dt.data, atol=1e-8), (
        "partial cells must change the momentum tendency on sloped bathymetry"
    )


def test_inactive_TS_poison_does_not_reach_active_tendencies():
    """Huge finite T/S poison in below-seafloor cells must NOT change any
    active-cell tendency — the rock extrapolation-fill isolates active cells
    (codex CRITICAL #1: FC stencils could otherwise import rock values)."""
    grid = create_cubed_sphere(N)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _state(grid, zc, jnp.asarray(H))
    fc, cfg = _fc(), _cfg()

    inactive = ~np.asarray(pc.is_active)
    assert inactive.any()
    t_clean = ocean_baroclinic_tendencies_fc(st, grid, pc, fc, cfg)

    T_pois = np.asarray(st.T.data).copy()
    S_pois = np.asarray(st.S.data).copy()
    T_pois[inactive] += 1.0e6
    S_pois[inactive] += 1.0e6
    st_pois = st._replace(
        T=st.T.replace(data=jnp.asarray(T_pois, dtype=st.T.data.dtype)),
        S=st.S.replace(data=jnp.asarray(S_pois, dtype=st.S.data.dtype)),
    )
    t_pois = ocean_baroclinic_tendencies_fc(st_pois, grid, pc, fc, cfg)

    active = np.asarray(pc.is_active)
    for name in _FIELDS_3D:
        a = np.asarray(getattr(t_clean, name).data)[active]
        b = np.asarray(getattr(t_pois, name).data)[active]
        assert np.allclose(a, b, atol=1e-9, rtol=0.0), (
            f"{name}: below-seafloor T/S poison leaked into active-cell tendency"
        )


def test_partial_cell_path_differentiable():
    """jax.grad through the partial-cell FC tendency is finite."""
    grid = create_cubed_sphere(N)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _state(grid, zc, jnp.asarray(H))
    fc, cfg = _fc(), _cfg()

    def loss(T_field):
        s2 = st._replace(T=st.T.replace(data=T_field))
        tend = ocean_baroclinic_tendencies_fc(s2, grid, pc, fc, cfg)
        return jnp.sum(tend.du_dt.data ** 2 + tend.dT_dt.data ** 2)

    g = jax.grad(loss)(st.T.data)
    assert jnp.all(jnp.isfinite(g)), "gradient through partial-cell path is non-finite"
