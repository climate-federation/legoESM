"""General immersed-boundary (GridFittedBoundary) geometry — arbitrary solid
cells + thin-wall face barriers on the lat-lon C-grid.

Correctness gate: IMPERMEABILITY — no flux-form transport crosses a masked /
barrier face for ANY velocity — and the telescoping mass conservation that
follows, plus the surface-connectivity guard that fails loudly on the
deferred cavity/overhang geometry.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.dynamics.immersed_boundary import (
    combine_solid_mask,
    assert_columns_surface_connected,
    add_face_barriers,
    immersed_face_masks,
    mask_immersed_field,
    immersed_partial_cell_coordinate,
    immersed_column_depth,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d


def _all_active(nlat, nlon, nlev):
    return jnp.ones((nlat, nlon, nlev), dtype=bool)


def test_combine_solid_mask():
    a = _all_active(3, 4, 2)
    solid = jnp.zeros_like(a).at[1, 2, :].set(True)
    out = combine_solid_mask(a, solid)
    assert not bool(out[1, 2, 0]) and not bool(out[1, 2, 1])
    assert bool(out[0, 0, 0])
    assert int(jnp.sum(out)) == int(jnp.sum(a)) - 2


def test_surface_connectivity_guard():
    """A top-anchored (bathymetry-like) mask passes; an overhang raises."""
    # Bathymetry-like: active from the surface down to a per-column bottom.
    a = jnp.stack([
        jnp.array([True, True, True, True]),   # level 0
        jnp.array([True, True, False, True]),  # level 1 (col 2 seafloor)
        jnp.array([True, False, False, True]), # level 2
    ], axis=-1)[None]  # (1, 4, 3)
    assert_columns_surface_connected(a)  # no raise

    # Overhang: col 0 active at level 2 but INACTIVE at level 1 (cavity).
    overhang = a.at[0, 0, 1].set(False)   # active below inactive at (0,0)
    with pytest.raises(ValueError, match="surface-connected"):
        assert_columns_surface_connected(overhang)


def test_impermeable_faces_zero_flux_for_any_velocity():
    """Every face touching a solid cell is masked, so the flux-form transport
    F = u * u_mask is EXACTLY zero there regardless of the velocity."""
    nlat, nlon, nlev = 5, 6, 3
    a = _all_active(nlat, nlon, nlev)
    solid = jnp.zeros_like(a).at[2, 3, :].set(True)      # an interior obstacle
    active = combine_solid_mask(a, solid)
    u_mask, v_mask = compute_face_masks_3d(active)

    rng = np.random.default_rng(1)
    u = jnp.asarray(rng.normal(size=u_mask.shape))
    v = jnp.asarray(rng.normal(size=v_mask.shape))
    Fu, Fv = u * u_mask, v * v_mask

    # The obstacle's four faces (east/west u-faces, north/south v-faces) must
    # carry zero flux.
    assert float(Fu[2, 3, 0]) == 0.0 and float(Fu[2, 4, 0]) == 0.0
    assert float(Fv[2, 3, 0]) == 0.0 and float(Fv[3, 3, 0]) == 0.0
    # And generally: flux vanishes wherever the mask is zero.
    assert bool(jnp.all(jnp.where(u_mask == 0.0, Fu == 0.0, True)))
    assert bool(jnp.all(jnp.where(v_mask == 0.0, Fv == 0.0, True)))


def test_mass_conservation_telescopes_to_zero():
    """With walls at the meridional boundaries and periodic longitude, the
    discrete flux divergence of a masked flux field sums to zero over the whole
    domain — the flux-form guarantee that an impermeable geometry conserves
    mass (no spurious sources/sinks at solid faces)."""
    nlat, nlon, nlev = 4, 5, 1
    a = _all_active(nlat, nlon, nlev)
    solid = jnp.zeros_like(a).at[1, 2, :].set(True).at[2, 1, :].set(True)
    active = combine_solid_mask(a, solid)
    u_mask, v_mask = compute_face_masks_3d(active)   # v walls at N/S by default

    rng = np.random.default_rng(2)
    u = jnp.asarray(rng.normal(size=u_mask.shape))
    # The last u-face is the periodic wrap of face 0 (same PHYSICAL face), so
    # its velocity must equal face 0's; then Fu telescopes cleanly in longitude.
    u = u.at[:, -1, :].set(u[:, 0, :])
    Fu = u * u_mask                                            # (nlat, nlon+1, k)
    Fv = jnp.asarray(rng.normal(size=v_mask.shape)) * v_mask   # (nlat+1, nlon, k)

    # Cell divergence (uniform area): (Fu_east - Fu_west) + (Fv_north - Fv_south).
    div = (Fu[:, 1:, :] - Fu[:, :-1, :]) + (Fv[1:, :, :] - Fv[:-1, :, :])
    # Sum over ALL cells = net domain-boundary flux. Longitude telescopes to the
    # (equal) periodic wrap faces and the meridional boundary v-faces are walls
    # (v_mask=0), so the net must vanish to round-off — no spurious source/sink
    # at the solid faces (flux-form impermeability => mass conservation).
    total = float(jnp.sum(div))
    assert abs(total) < 1e-10


def test_thin_wall_barrier_blocks_without_masking_cells():
    """A thin-wall v-barrier zeroes only the target faces (no cell masked), so
    both adjacent columns stay fully active — the safe interior-wall path."""
    nlat, nlon, nlev = 4, 4, 2
    a = _all_active(nlat, nlon, nlev)
    u_mask, v_mask = compute_face_masks_3d(a)
    # Wall along the v-face between rows 1 and 2, spanning all longitudes.
    v_barrier = jnp.zeros((nlat + 1, nlon), dtype=bool).at[2, :].set(True)
    um2, vm2 = add_face_barriers(u_mask, v_mask, v_barrier=v_barrier)

    assert bool(jnp.all(vm2[2, :, :] == 0.0))        # barrier faces closed
    assert bool(jnp.all(um2 == u_mask))              # u-faces untouched
    # No cell was masked (connectivity preserved -> vertical logic untouched).
    assert_columns_surface_connected(a)              # still valid
    # A barrier is stricter than the bare face mask.
    assert float(jnp.sum(vm2)) < float(jnp.sum(v_mask))


def test_immersed_face_masks_combines_solid_and_barrier():
    a = _all_active(4, 4, 1)
    solid = jnp.zeros_like(a).at[1, 1, :].set(True)
    active = combine_solid_mask(a, solid)
    u_barrier = jnp.zeros((4, 5), dtype=bool).at[0, 2].set(True)
    um, vm = immersed_face_masks(active, u_barrier=u_barrier)
    assert float(um[0, 2, 0]) == 0.0                 # barrier face closed
    assert float(um[1, 1, 0]) == 0.0 and float(um[1, 2, 0]) == 0.0  # solid faces


def test_mask_immersed_field():
    a = _all_active(3, 3, 2)
    solid = jnp.zeros_like(a).at[1, 1, 0].set(True)
    active = combine_solid_mask(a, solid)
    field = jnp.ones((3, 3, 2))
    out = mask_immersed_field(field, active, value=-9.0)
    assert float(out[1, 1, 0]) == -9.0
    assert float(out[1, 1, 1]) == 1.0 and float(out[0, 0, 0]) == 1.0


def test_immersed_partial_cell_coordinate():
    """Carving a bottom-anchored solid column into a real partial-cell coord
    zeroes its thickness/activity and stays surface-connected; an overhang
    solid mask raises."""
    from legoesm.ocean.vertical import (
        create_z_star_from_thicknesses, create_partial_cell_coordinate)
    nlat, nlon = 3, 4
    zc = create_z_star_from_thicknesses(jnp.array([10.0, 20.0, 40.0]))
    H = jnp.full((nlat, nlon), 70.0)     # full-depth ocean everywhere
    coord = create_partial_cell_coordinate(zc, H)

    # Bottom-anchored solid: mask the DEEPEST level of one column (shallower
    # seafloor) -> surface-connected, valid.
    solid = jnp.zeros((nlat, nlon, 3), dtype=bool).at[1, 2, 2].set(True)
    masked = immersed_partial_cell_coordinate(coord, solid)
    assert not bool(masked.is_active[1, 2, 2])
    assert float(masked.h_partial[1, 2, 2]) == 0.0
    assert bool(masked.is_active[1, 2, 1])           # cells above still active
    # bottom_level MUST move up to the new deepest active level (codex BLOCKER:
    # else bottom drag / BBL target the carved-out level).
    assert int(masked.bottom_level[1, 2]) == 1
    assert int(masked.bottom_level[0, 0]) == 2       # uncarved column unchanged
    # H_bathy = sum(h_partial) shrinks for the carved column and stays the depth
    # invariant callers must use for layer thickness.
    depth = immersed_column_depth(masked)
    assert float(depth[1, 2]) == pytest.approx(30.0, rel=1e-9)   # 10+20, level 2 gone
    assert float(depth[0, 0]) == pytest.approx(70.0, rel=1e-9)
    np.testing.assert_allclose(
        np.asarray(depth), np.asarray(jnp.sum(masked.h_partial, axis=-1)))

    # Interior overhang solid (mask a MIDDLE level, water below) -> raises.
    bad = jnp.zeros((nlat, nlon, 3), dtype=bool).at[0, 0, 1].set(True)
    with pytest.raises(ValueError, match="surface-connected"):
        immersed_partial_cell_coordinate(coord, bad)
