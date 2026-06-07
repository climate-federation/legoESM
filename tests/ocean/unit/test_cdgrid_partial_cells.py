"""C-D grid (FV3-style, atmosphere-matching) partial-cell substrate tests.

Exercises the ``OceanPartialCellCoordinate`` path through the cubed-sphere
C-D grid ocean PE (``ocean_baroclinic_tendencies_cdgrid``) — the faithful
cube backend (the FC A-grid backend is deprecated; all cubed-sphere ocean
discretizations map to the C-D grid).  Mirrors ``test_cube_partial_cells.py``
(which covers the deprecated FC backend) so the partial-cell semantics are
validated on the grid the atmosphere actually uses:

1. Flat-bottom (H_bathy == H_max) is BIT-EXACT vs the pure z* coord — the
   partial-cell branch must not perturb the legacy result.
2. Below-seafloor cells (``is_active == False``) carry exactly-zero
   momentum/tracer tendencies, and the partial-cell column thickness sums to
   the local bathymetry.
3. On sloped bathymetry the partial-cell result DIFFERS from z* (the
   substrate is not a silent no-op).
4. Below-seafloor T/S poison does NOT reach active-cell tendencies (the rock
   extrapolation-fill isolates active cells).
5. The partial-cell path stays differentiable (``jax.grad`` finite).

Run under ``JAX_ENABLE_X64=1`` (the cd-grid PGF cumsum is float64).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
    compute_layer_thickness,
)
from legoesm.ocean.init import rest_state_ocean
from legoesm.ocean.state import OceanConfig
from legoesm.ocean.dynamics.ocean_pe_cdgrid import ocean_baroclinic_tendencies_cdgrid

N = 8
NLEV = 5
HMAX = 4000.0
_FIELDS_3D = ("du_dt", "dv_dt", "dT_dt", "dS_dt")


def _cfg():
    # All horizontal/vertical diffusion off → tendency = advection + PGF only,
    # so the partial-cell PGF/thickness change is isolated.
    return OceanConfig(A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0)


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
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = jnp.full((6, N, N), HMAX, dtype=jnp.float64)
    pc = create_partial_cell_coordinate(zc, H)
    st = _state(grid, zc, H)
    cfg = _cfg()

    t_z = ocean_baroclinic_tendencies_cdgrid(st, grid, zc, cdgrid, cfg)
    t_p = ocean_baroclinic_tendencies_cdgrid(st, grid, pc, cdgrid, cfg)

    for name in _FIELDS_3D + ("deta_dt",):
        a = np.asarray(getattr(t_z, name).data)
        b = np.asarray(getattr(t_p, name).data)
        np.testing.assert_array_equal(
            a, b, err_msg=f"{name}: flat-bottom partial cells must match z* BIT-EXACT"
        )


def test_below_seafloor_masked_and_thickness_sums_to_bathy():
    """Below-seafloor tendencies are exactly zero; Σ h_k == H_bathy."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _state(grid, zc, jnp.asarray(H))

    inactive = ~np.asarray(pc.is_active)
    assert inactive.any(), "sloped bathy should leave some below-seafloor cells"

    t_p = ocean_baroclinic_tendencies_cdgrid(st, grid, pc, cdgrid, _cfg())
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
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _state(grid, zc, jnp.asarray(H))
    cfg = _cfg()

    t_z = ocean_baroclinic_tendencies_cdgrid(st, grid, zc, cdgrid, cfg)
    t_p = ocean_baroclinic_tendencies_cdgrid(st, grid, pc, cdgrid, cfg)
    assert not jnp.allclose(t_z.du_dt.data, t_p.du_dt.data, atol=1e-8), (
        "partial cells must change the momentum tendency on sloped bathymetry"
    )


def test_inactive_TS_poison_does_not_reach_active_tendencies():
    """Huge finite T/S poison in below-seafloor cells must NOT change any
    active-cell tendency — the rock extrapolation-fill isolates active cells."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _state(grid, zc, jnp.asarray(H))
    cfg = _cfg()

    inactive = ~np.asarray(pc.is_active)
    assert inactive.any()
    t_clean = ocean_baroclinic_tendencies_cdgrid(st, grid, pc, cdgrid, cfg)

    T_pois = np.asarray(st.T.data).copy()
    S_pois = np.asarray(st.S.data).copy()
    T_pois[inactive] += 1.0e6
    S_pois[inactive] += 1.0e6
    st_pois = st._replace(
        T=st.T.replace(data=jnp.asarray(T_pois, dtype=st.T.data.dtype)),
        S=st.S.replace(data=jnp.asarray(S_pois, dtype=st.S.data.dtype)),
    )
    t_pois = ocean_baroclinic_tendencies_cdgrid(st_pois, grid, pc, cdgrid, cfg)

    active = np.asarray(pc.is_active)
    for name in _FIELDS_3D:
        a = np.asarray(getattr(t_clean, name).data)[active]
        b = np.asarray(getattr(t_pois, name).data)[active]
        assert np.allclose(a, b, atol=1e-9, rtol=0.0), (
            f"{name}: below-seafloor T/S poison leaked into active-cell tendency"
        )


def _linear_rho_cfg(scheme):
    """OceanConfig with a LINEAR EOS (rho pressure-independent) so a linear
    T(z) profile gives an exactly-linear rho(z) — the regime where smc03's
    harmonic-slope reconstruction is exact and the rest PGF must vanish."""
    from legoesm.ocean.eos import LinearEOSConfig
    return OceanConfig(A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0,
                       eos="linear", eos_linear=LinearEOSConfig(),
                       pgf_scheme=scheme)


def _linear_strat_rest_state(grid, zc, pc, H, dTdz=-2.0e-3):
    """Horizontally-uniform, vertically-LINEAR density rest column on an
    all-ocean cube with sloped bathy.  T is set linear in each cell's PARTIAL
    centroid depth so the in-situ density field is a single linear function of
    physical depth (rho sampled at the column-specific centroids).  The exact
    horizontal PGF is therefore ZERO at every depth (∇_h rho|_z = 0)."""
    st = rest_state_ocean(grid, zc, H_max=HMAX)
    land = jnp.ones((6, N, N), dtype=st.land_mask.data.dtype)
    h_part = np.asarray(pc.h_partial)
    z_c = np.cumsum(h_part, axis=-1) - 0.5 * h_part        # (6,N,N,NLEV), +down
    T_lin = 15.0 + dTdz * z_c                               # linear in depth
    return st._replace(
        land_mask=st.land_mask.replace(data=land),
        H_bathy=st.H_bathy.replace(data=jnp.asarray(H)),
        T=st.T.replace(data=jnp.asarray(T_lin, dtype=st.T.data.dtype)),
        S=st.S.replace(data=jnp.full((6, N, N, NLEV), 35.0,
                                     dtype=st.S.data.dtype)),
    )


def _allwet_partial_bathy(grid):
    """Varying bottom-cell thickness with NO column losing a level (every column
    keeps all NLEV active), so there are no active/inactive corners — isolates
    the partial-bottom-centroid PGF (the smc03 cancellation) from the separate
    wet/rock seafloor-step PGF."""
    lat = np.asarray(grid.lat)
    dz = HMAX / NLEV
    # H in ((NLEV-1)·dz, HMAX]  ->  bottom partial-cell thickness in (0, dz].
    return ((NLEV - 1) * dz + 0.5 * dz
            + 0.49 * dz * np.sin(2.0 * lat)).astype(np.float64)


def _maxacc_active(t, active):
    return max(np.max(np.abs(np.asarray(t.du_dt.data)[active])),
               np.max(np.abs(np.asarray(t.dv_dt.data)[active])))


def test_smc03_linear_rho_rest_pgf_vanishes_and_beats_adcroft():
    """THE canonical Adcroft-Campin partial-cell test: a vertically-linear-
    density rest column over an ALL-WET partial bathy (varying bottom-cell
    thickness, no level loss).  smc03 is EXACT for linear rho(z) (harmonic-slope
    reconstruction) so the spurious rest PGF acceleration is ~machine-zero; the
    linear Adcroft correction leaves a 2nd-order residual ∝ slope·(z_ref−z_c)²
    — the bottom-trapped spurious PGF that seeds the cube cold-start blowup.
    Validates --cube-pgf-scheme smc03."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _allwet_partial_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    assert np.asarray(pc.is_active).all(), (
        "all-wet bathy must keep every cell active (no seafloor steps)")
    st = _linear_strat_rest_state(grid, zc, pc, H)

    active = np.asarray(pc.is_active)
    t_ad = ocean_baroclinic_tendencies_cdgrid(
        st, grid, pc, cdgrid, _linear_rho_cfg("adcroft"))
    t_smc = ocean_baroclinic_tendencies_cdgrid(
        st, grid, pc, cdgrid, _linear_rho_cfg("smc03"))
    ad, smc = _maxacc_active(t_ad, active), _maxacc_active(t_smc, active)
    assert smc < 1.0e-9, (
        f"smc03 linear-rho rest PGF acceleration must ~vanish, got {smc:.3e} m/s^2")
    assert ad > 20.0 * smc, (
        f"adcroft residual ({ad:.3e}) should dwarf smc03 ({smc:.3e}) — "
        "the spurious partial-cell PGF smc03 removes")


def test_smc03_beats_adcroft_on_stepped_bathy():
    """On steep bathy WITH seafloor steps (active/inactive corners) the smc03
    PGF + wet/rock horizontal closure gives a smaller spurious rest PGF than the
    linear Adcroft correction.  Linear rho(z); exact horizontal PGF is 0."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    assert (~np.asarray(pc.is_active)).any(), "stepped bathy needs inactive cells"
    st = _linear_strat_rest_state(grid, zc, pc, H)

    active = np.asarray(pc.is_active)
    t_ad = ocean_baroclinic_tendencies_cdgrid(
        st, grid, pc, cdgrid, _linear_rho_cfg("adcroft"))
    t_smc = ocean_baroclinic_tendencies_cdgrid(
        st, grid, pc, cdgrid, _linear_rho_cfg("smc03"))
    ad, smc = _maxacc_active(t_ad, active), _maxacc_active(t_smc, active)
    assert smc < ad, (
        f"smc03 stepped-bathy rest PGF ({smc:.3e}) must beat adcroft ({ad:.3e})")


def test_smc03_gated_off_on_pure_zstar():
    """With a pure z* coord (is_partial False) the smc03 branch is skipped, so
    pgf_scheme is inert — smc03 and adcroft give the IDENTICAL z* tendency."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = jnp.full((6, N, N), HMAX, dtype=jnp.float64)
    st = _state(grid, zc, H)                                # nonzero PGF (T pert)
    base = _cfg()
    t_ad = ocean_baroclinic_tendencies_cdgrid(
        st, grid, zc, cdgrid, base._replace(pgf_scheme="adcroft"))
    t_smc = ocean_baroclinic_tendencies_cdgrid(
        st, grid, zc, cdgrid, base._replace(pgf_scheme="smc03"))
    for name in _FIELDS_3D + ("deta_dt",):
        np.testing.assert_array_equal(
            np.asarray(getattr(t_ad, name).data),
            np.asarray(getattr(t_smc, name).data),
            err_msg=f"{name}: pgf_scheme must be inert on a pure z* coord",
        )


def test_smc03_partial_path_differentiable():
    """jax.grad through the cd-grid smc03 partial-cell PGF is finite."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _state(grid, zc, jnp.asarray(H))
    cfg = _cfg()._replace(pgf_scheme="smc03")

    def loss(T_field):
        s2 = st._replace(T=st.T.replace(data=T_field))
        tend = ocean_baroclinic_tendencies_cdgrid(s2, grid, pc, cdgrid, cfg)
        return jnp.sum(tend.du_dt.data ** 2 + tend.dT_dt.data ** 2)

    g = jax.grad(loss)(st.T.data)
    assert jnp.all(jnp.isfinite(g)), "gradient through smc03 partial path non-finite"


def test_partial_cell_path_differentiable():
    """jax.grad through the C-D grid partial-cell tendency is finite."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _state(grid, zc, jnp.asarray(H))
    cfg = _cfg()

    def loss(T_field):
        s2 = st._replace(T=st.T.replace(data=T_field))
        tend = ocean_baroclinic_tendencies_cdgrid(s2, grid, pc, cdgrid, cfg)
        return jnp.sum(tend.du_dt.data ** 2 + tend.dT_dt.data ** 2)

    g = jax.grad(loss)(st.T.data)
    assert jnp.all(jnp.isfinite(g)), "gradient through partial-cell path is non-finite"


# ---------------------------------------------------------------------------
# Bottom drag (cd-grid cell-centre) — the proven dissipation-stack piece ported
# to the cube cold-start path.  Drag is isolated as the difference between a run
# WITH ``bottom_drag_r > 0`` and the same run with it off (default 0.0).
# ---------------------------------------------------------------------------

_DRAG_R = 1.0e-3   # [m/s] linear bottom-drag coefficient for the tests


def _cfg_drag(r=_DRAG_R, bbl=0.0, bg=0.0):
    return OceanConfig(A_h=0.0, K_h=0.0, A_v=0.0, K_v=0.0, hyperdiff_coeff=0.0,
                       bottom_drag_r=r, bottom_drag_bbl_thickness=bbl,
                       bottom_drag_bg_velocity=bg)


def _with_uniform_flow(st, u0=0.1, v0=-0.05):
    return st._replace(
        u=st.u.replace(data=jnp.full_like(st.u.data, u0)),
        v=st.v.replace(data=jnp.full_like(st.v.data, v0)),
    )


def _drag_only(st, grid, zc_or_pc, cdgrid, cfg_drag):
    """Isolate the drag tendency = (with drag) − (without drag)."""
    t0 = ocean_baroclinic_tendencies_cdgrid(st, grid, zc_or_pc, cdgrid, _cfg())
    t1 = ocean_baroclinic_tendencies_cdgrid(st, grid, zc_or_pc, cdgrid, cfg_drag)
    du = np.asarray(t1.du_dt.data) - np.asarray(t0.du_dt.data)
    dv = np.asarray(t1.dv_dt.data) - np.asarray(t0.dv_dt.data)
    return du, dv


def test_bottom_drag_zero_at_rest():
    """Drag ∝ u, so at u=v=0 it adds NOTHING (no spurious rest tendency)."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _state(grid, zc, jnp.asarray(H))  # rest_state_ocean → u=v=0
    du, dv = _drag_only(st, grid, pc, cdgrid, _cfg_drag())
    assert np.max(np.abs(du)) == 0.0 and np.max(np.abs(dv)) == 0.0, (
        "bottom drag must vanish at rest (u=0)"
    )


def test_bottom_drag_off_bit_exact():
    """bottom_drag_r=0 (default) leaves the cd-grid tendency BIT-EXACT — the gate
    adds literally nothing — on both z* and partial-cell coords."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _with_uniform_flow(_state(grid, zc, jnp.asarray(H)))
    for coord in (zc, pc):
        t_a = ocean_baroclinic_tendencies_cdgrid(st, grid, coord, cdgrid, _cfg())
        t_b = ocean_baroclinic_tendencies_cdgrid(
            st, grid, coord, cdgrid, _cfg_drag(r=0.0))
        for name in _FIELDS_3D + ("deta_dt",):
            np.testing.assert_array_equal(
                np.asarray(getattr(t_a, name).data),
                np.asarray(getattr(t_b, name).data),
                err_msg=f"{name}: bottom_drag_r=0 must be bit-exact",
            )


def test_bottom_drag_applied_at_partial_bottom_level_only():
    """On partial cells, drag is nonzero ONLY at each column's own seafloor
    (``bottom_level``), opposes the flow, and is zero at every other level."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _with_uniform_flow(_state(grid, zc, jnp.asarray(H)), u0=0.1, v0=-0.05)
    du, dv = _drag_only(st, grid, pc, cdgrid, _cfg_drag())

    lvl = np.arange(NLEV)
    botlev = np.asarray(pc.bottom_level)                      # (6, N, N)
    is_bot = lvl[None, None, None, :] == botlev[..., None]    # (6, N, N, NLEV)
    assert np.max(np.abs(du[~is_bot])) < 1e-12, "drag leaked off the bottom level"
    assert np.max(np.abs(dv[~is_bot])) < 1e-12, "drag leaked off the bottom level"
    assert np.min(np.abs(du[is_bot])) > 0.0, "drag missing at the bottom level"
    # Opposes the flow: u0>0 → drag_u<0; v0<0 → drag_v>0.
    assert np.all(du[is_bot] < 0.0) and np.all(dv[is_bot] > 0.0), (
        "bottom drag must oppose the velocity"
    )


def test_bottom_drag_zstar_at_deepest_level():
    """z* (full columns): drag acts at the deepest reference level only."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = jnp.full((6, N, N), HMAX, dtype=jnp.float64)
    st = _with_uniform_flow(_state(grid, zc, H))
    du, _ = _drag_only(st, grid, zc, cdgrid, _cfg_drag())
    above = np.max(np.abs(du[..., :-1]))
    bottom = np.min(np.abs(du[..., -1]))
    assert above < 1e-12, f"z* drag leaked above the deepest level (max={above:.2e})"
    assert bottom > 0.0, "z* drag missing at the deepest level"


def test_bottom_drag_dissipative():
    """Drag removes kinetic energy: Σ (u·drag_u + v·drag_v) ≤ 0."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _with_uniform_flow(_state(grid, zc, jnp.asarray(H)))
    du, dv = _drag_only(st, grid, pc, cdgrid, _cfg_drag())
    u = np.asarray(st.u.data)
    v = np.asarray(st.v.data)
    ke_rate = np.sum(u * du + v * dv)
    assert ke_rate <= 0.0, f"bottom drag is not dissipative: ΣKE rate={ke_rate:.3e}"


def test_bottom_drag_bbl_spreads_above_bottom():
    """With a BBL thickness, drag is distributed over the near-seafloor band
    (more wet cells than the single bottom cell) — the thin-cell blowup fix."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _with_uniform_flow(_state(grid, zc, jnp.asarray(H)))
    active = np.asarray(pc.is_active)
    du_single, _ = _drag_only(st, grid, pc, cdgrid, _cfg_drag())
    du_bbl, _ = _drag_only(st, grid, pc, cdgrid, _cfg_drag(bbl=1500.0))
    n_single = int(np.sum((np.abs(du_single) > 1e-14) & active))
    n_bbl = int(np.sum((np.abs(du_bbl) > 1e-14) & active))
    assert n_bbl > n_single, (
        f"BBL must spread drag over more cells (single={n_single}, bbl={n_bbl})"
    )


def test_bottom_drag_differentiable():
    """jax.grad through the cd-grid bottom-drag path is finite (safe-divide)."""
    grid = create_cubed_sphere(N)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    zc = create_ocean_z_star(n_levels=NLEV, H_max=HMAX)
    H = _sloped_bathy(grid)
    pc = create_partial_cell_coordinate(zc, jnp.asarray(H))
    st = _with_uniform_flow(_state(grid, zc, jnp.asarray(H)))
    cfg = _cfg_drag(bbl=1500.0, bg=0.05)  # exercise BBL + quadratic-floor too

    def loss(u_field):
        s2 = st._replace(u=st.u.replace(data=u_field))
        tend = ocean_baroclinic_tendencies_cdgrid(s2, grid, pc, cdgrid, cfg)
        return jnp.sum(tend.du_dt.data ** 2 + tend.dv_dt.data ** 2)

    g = jax.grad(loss)(st.u.data)
    assert jnp.all(jnp.isfinite(g)), "gradient through bottom-drag path non-finite"
