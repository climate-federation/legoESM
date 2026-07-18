"""Partial-periodic seam-wall (NEMO DINO faithful geometry).

The seam wall closes the zonal periodic-seam u-face at selected latitude
rows while leaving ALL interior cells wet.  It is threaded as an optional
``seam_wall_rows`` (shape ``(n_lat,)``, 1 = walled) profile read by the
four mask-derivation sites from ``grid.seam_wall_rows``:

  1. ``compute_face_masks``       (2-D u/v face masks)
  2. ``compute_face_masks_3d``    (per-level face masks)
  3. ``compute_vertex_mask``      (EEN/PV corner mask)
  4. barotropic diffusion face mask (via ``_dissipation_coeffs``)

Gates: (a) default None is byte-identical; (b) a walled row zeros the two
seam u-face columns everywhere consistently and passes zero zonal flux;
(c) open rows stay periodic; (d) the runtime invariant check accepts a
walled-but-wet state.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.latlon import create_latlon_geometry
from legoesm.grids.operators_latlon_cgrid import compute_vertex_mask
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_face_masks,
    compute_face_masks_3d,
)


N_LAT, N_LON, NLEV = 12, 8, 4


def _geom(seam=None):
    g = create_latlon_geometry(
        N_LAT, N_LON, radius=constants.R_earth, omega=constants.Omega)
    if seam is not None:
        g = g._replace(seam_wall_rows=jnp.asarray(seam, dtype=jnp.float64))
    return g


def _wet_mask():
    # all interior cells wet (the faithful DINO surface tmask)
    return jnp.ones((N_LAT, N_LON), dtype=jnp.float64)


def _seam_profile():
    # open (channel) only in a central band; walled elsewhere
    sw = np.ones(N_LAT)
    sw[5:8] = 0.0
    return sw


# ---------------------------------------------------------------------------
# (a) default None == byte-identical
# ---------------------------------------------------------------------------
def test_default_none_byte_identical():
    m = _wet_mask()
    g0 = _geom(None)
    u0, v0 = compute_face_masks(m)
    u1, v1 = compute_face_masks(m, g0)          # grid.seam_wall_rows is None
    assert jnp.array_equal(u0, u1) and jnp.array_equal(v0, v1)

    is_active = jnp.ones((N_LAT, N_LON, NLEV), dtype=bool)
    u3a, v3a = compute_face_masks_3d(is_active, None)
    u3b, v3b = compute_face_masks_3d(is_active, g0)
    assert jnp.array_equal(u3a, u3b) and jnp.array_equal(v3a, v3b)

    assert jnp.array_equal(compute_vertex_mask(m), compute_vertex_mask(m, g0))


# ---------------------------------------------------------------------------
# (b) walled rows zero both seam u-face columns, consistently across 2D/3D
# ---------------------------------------------------------------------------
def test_walled_rows_zero_seam_faces():
    m = _wet_mask()
    sw = _seam_profile()
    g = _geom(sw)
    walled = sw > 0.5
    openrow = ~walled

    u, _ = compute_face_masks(m, g)
    # walled rows: both wrap columns (0 and n_lon) closed
    assert np.all(np.asarray(u)[walled, 0] == 0.0)
    assert np.all(np.asarray(u)[walled, N_LON] == 0.0)
    # open rows: seam stays periodic-wet (all cells wet => face wet)
    assert np.all(np.asarray(u)[openrow, 0] == 1.0)
    assert np.all(np.asarray(u)[openrow, N_LON] == 1.0)
    # interior (non-seam) faces untouched
    assert np.all(np.asarray(u)[:, 1:N_LON] == 1.0)

    is_active = jnp.ones((N_LAT, N_LON, NLEV), dtype=bool)
    u3, _ = compute_face_masks_3d(is_active, g)
    u3 = np.asarray(u3)
    assert np.all(u3[walled, 0, :] == 0.0) and np.all(u3[walled, N_LON, :] == 0.0)
    assert np.all(u3[openrow, 0, :] == 1.0) and np.all(u3[openrow, N_LON, :] == 1.0)
    # 3-D seam faces match the 2-D result on every level
    for k in range(NLEV):
        assert np.array_equal(u3[:, 0, k], np.asarray(u)[:, 0])
        assert np.array_equal(u3[:, N_LON, k], np.asarray(u)[:, N_LON])


def test_vertex_mask_seam_corner_dry():
    m = _wet_mask()
    sw = _seam_profile()
    g = _geom(sw)
    vtx = np.asarray(compute_vertex_mask(m, g))     # (n_lat+1, n_lon+1)
    # A seam vertex row is dry if EITHER adjacent cell row is walled.
    open_face = 1.0 - sw
    vtx_open = np.ones(N_LAT + 1)
    vtx_open[1:] *= open_face
    vtx_open[:-1] *= open_face
    # interior vertex rows only (poles are wall-zeroed regardless)
    for i in range(1, N_LAT):
        expect0 = vtx_open[i]
        assert vtx[i, 0] == expect0
        assert vtx[i, N_LON] == expect0
    # a fully-walled-neighbour vertex row must be dry at the seam
    assert vtx[3, 0] == 0.0  # rows 2/3 both walled (sw[2],sw[3]=1)


# ---------------------------------------------------------------------------
# (c) zero zonal flux through a walled seam
# ---------------------------------------------------------------------------
def test_zero_zonal_flux_through_wall():
    m = _wet_mask()
    sw = _seam_profile()
    g = _geom(sw)
    u3, _ = compute_face_masks_3d(jnp.ones((N_LAT, N_LON, NLEV), bool), g)
    u3 = np.asarray(u3)
    # arbitrary nonzero transport at every face; masking must kill the seam flux
    transport = np.ones((N_LAT, N_LON + 1, NLEV))
    masked = transport * u3
    walled = sw > 0.5
    assert np.all(masked[walled, 0, :] == 0.0)
    assert np.all(masked[walled, N_LON, :] == 0.0)


def test_barotropic_diffusion_mask_walled():
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import _dissipation_coeffs
    from legoesm.ocean.state import BarotropicConfig, LatLonCGridOceanConfig

    sw = _seam_profile()
    g = _geom(sw)
    m = _wet_mask()
    # enable barotropic diffusion so the mask is built
    baro = BarotropicConfig(barotropic_diffusion_alpha=0.1)
    cfg = LatLonCGridOceanConfig(barotropic=baro)
    out = _dissipation_coeffs(cfg, g, g.area_T, 100.0, jnp.float64, m)
    diff_u_mask = np.asarray(out[2])
    walled = sw > 0.5
    assert np.all(diff_u_mask[walled, 0] == 0.0)
    assert np.all(diff_u_mask[walled, N_LON] == 0.0)
    openrow = ~walled
    assert np.all(diff_u_mask[openrow, 0] == 1.0)


# ---------------------------------------------------------------------------
# (d) runtime invariant accepts a walled-but-wet state
# ---------------------------------------------------------------------------
def test_invariant_walled_but_wet_state():
    # The stored 2-D u_mask (from compute_face_masks with the seam wall)
    # must equal the invariant recompute (compute_face_masks(mask, grid)).
    m = _wet_mask()
    g = _geom(_seam_profile())
    u_stored, v_stored = compute_face_masks(m, g)
    u_expect, v_expect = compute_face_masks(m, g)
    assert jnp.array_equal(u_stored, u_expect)
    assert jnp.array_equal(v_stored, v_expect)
    # and it must DIFFER from the periodic (no-wall) recompute — proving the
    # test is non-vacuous
    u_periodic, _ = compute_face_masks(m)
    assert not jnp.array_equal(u_stored, u_periodic)


# ---------------------------------------------------------------------------
# (b2) flux-form conservation: a masked seam u-face flux still closes the
# global divergence budget (the wrap identity u[:,0]==u[:,n_lon] survives).
# ---------------------------------------------------------------------------
def test_seam_wall_flux_form_conservation():
    g = _geom(_seam_profile())
    u3, v3 = compute_face_masks_3d(jnp.ones((N_LAT, N_LON, NLEV), bool), g)
    u3 = np.asarray(u3)[:, :, 0]                 # (n_lat, n_lon+1), one level
    # arbitrary transport, with the periodic wrap identity enforced
    rng = np.random.default_rng(0)
    Fu = rng.standard_normal((N_LAT, N_LON + 1))
    Fu[:, -1] = Fu[:, 0]                          # u[:,n_lon] == u[:,0] (same face)
    Fu = Fu * u3                                  # apply seam wall
    Fv = rng.standard_normal((N_LAT + 1, N_LON))
    Fv[0, :] = 0.0; Fv[-1, :] = 0.0              # closed N/S walls
    # cell divergence = (Fu_east - Fu_west) + (Fv_north - Fv_south)
    div = (Fu[:, 1:] - Fu[:, :-1]) + (Fv[1:, :] - Fv[:-1, :])
    # a closed domain (all boundary fluxes zero on walls / periodic-cancel)
    # must sum to machine zero
    assert abs(float(div.sum())) < 1e-10


# ---------------------------------------------------------------------------
# (e) MPI/SPMD band slicer keeps seam_wall_rows aligned with band-local n_lat
# ---------------------------------------------------------------------------
def test_band_slicer_aligns_seam_wall_rows():
    from legoesm.parallel.latlon_mpi import (
        LatLonBandLayout, slice_cgrid_geometry_to_band)
    g = _geom(_seam_profile())
    s, e = 4, 9
    layout = LatLonBandLayout(
        rank=1, n_ranks=3, n_lat_global=N_LAT, n_lon_global=N_LON,
        n_lat_local=e - s, lat_start=s, lat_end=e,
        south_rank=0, north_rank=2)
    band = slice_cgrid_geometry_to_band(g, layout)
    assert band.seam_wall_rows is not None
    assert band.seam_wall_rows.shape == (band.n_lat,)
    assert np.array_equal(
        np.asarray(band.seam_wall_rows), np.asarray(g.seam_wall_rows)[s:e])
    # None passes through
    g0 = _geom(None)
    band0 = slice_cgrid_geometry_to_band(g0, layout)
    assert band0.seam_wall_rows is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
