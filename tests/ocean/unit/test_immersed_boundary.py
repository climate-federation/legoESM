"""General immersed-boundary (GridFittedBoundary) geometry — arbitrary solid
cells + thin-wall face barriers on the lat-lon C-grid.

The module is a validated-but-NOT-YET-INTEGRATED core under
``ocean.dynamics._future`` (no production experiment carves an interior obstacle
yet — all use smooth bathymetry via GridFittedBottom); these tests exercise the
core so the impermeability / conservation / atomicity guarantees are locked in
before it is wired.

Correctness gate: IMPERMEABILITY — no flux-form transport crosses a masked /
barrier face for ANY velocity — the mass conservation that follows, the
surface-connectivity guard that fails loudly on the deferred cavity/overhang
geometry, and the ATOMIC state carve that keeps H_bathy / land_mask / face masks
consistent with the carved coordinate.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.dynamics._future.immersed_boundary import (
    combine_solid_mask,
    assert_columns_surface_connected,
    add_face_barriers,
    immersed_face_masks,
    mask_immersed_field,
    immersed_partial_cell_coordinate,
    immersed_column_depth,
    carve_immersed_state,
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


def test_solid_cell_has_exactly_zero_net_flux():
    """The STRICT impermeability statement: the net flux-divergence into EACH
    solid cell is exactly zero for any velocity — a solid cell neither gains
    nor loses mass (no spurious source/sink at an immersed boundary).  Stronger
    than a whole-domain telescoping sum (which vanishes even with no obstacle).
    """
    nlat, nlon, nlev = 4, 5, 1
    a = _all_active(nlat, nlon, nlev)
    solid = jnp.zeros_like(a).at[1, 2, :].set(True).at[2, 1, :].set(True)
    active = combine_solid_mask(a, solid)
    u_mask, v_mask = compute_face_masks_3d(active)

    rng = np.random.default_rng(3)
    Fu = jnp.asarray(rng.normal(size=u_mask.shape)) * u_mask   # (nlat, nlon+1, k)
    Fv = jnp.asarray(rng.normal(size=v_mask.shape)) * v_mask   # (nlat+1, nlon, k)

    # Per-cell flux divergence (uniform area, arbitrary velocities).
    div = (Fu[:, 1:, :] - Fu[:, :-1, :]) + (Fv[1:, :, :] - Fv[:-1, :, :])
    solid_2d = np.asarray(~np.asarray(active))[..., 0]         # (nlat, nlon) bool
    # EVERY solid cell has all four faces masked, so its net flux is exactly 0
    # regardless of the (random) velocities.
    assert float(jnp.max(jnp.abs(div[solid_2d]))) == 0.0


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
    # Sum over the WATER cells only (active) = net domain-boundary flux. Longitude
    # telescopes to the (equal) periodic wrap faces and the meridional boundary
    # v-faces are walls (v_mask=0), so the net must vanish to round-off — no
    # spurious source/sink at the solid faces (flux-form impermeability => mass
    # conservation).  Summing over active cells (not all cells) is the honest
    # water-budget statement; solid cells are separately exactly-zero above.
    water_2d = np.asarray(np.asarray(active))[..., 0]
    total = float(jnp.sum(jnp.where(jnp.asarray(water_2d)[..., None], div, 0.0)))
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


def test_add_face_barriers_rejects_mismatched_shape():
    """A transposed / mis-sized barrier fails LOUDLY (static geometry check),
    not by silently broadcasting a wall onto the wrong faces OR levels."""
    nlat, nlon, nlev = 4, 4, 2
    u_mask, v_mask = compute_face_masks_3d(_all_active(nlat, nlon, nlev))
    # v_barrier must be (nlat+1, nlon[, nlev]); give it a wrong first dim.
    bad_v = jnp.zeros((nlat, nlon), dtype=bool)
    with pytest.raises(ValueError, match="v_barrier"):
        add_face_barriers(u_mask, v_mask, v_barrier=bad_v)
    # A wrong rank.
    bad_u = jnp.zeros((nlat,), dtype=bool)
    with pytest.raises(ValueError, match="u_barrier"):
        add_face_barriers(u_mask, v_mask, u_barrier=bad_u)
    # A per-level barrier with a SINGLETON level dim (would silently broadcast
    # one level onto the whole column) must be rejected, not broadcast.
    bad_lvl1 = jnp.zeros((nlat, nlon + 1, 1), dtype=bool)
    with pytest.raises(ValueError, match="u_barrier"):
        add_face_barriers(u_mask, v_mask, u_barrier=bad_lvl1)
    # A per-level barrier with the WRONG number of levels must be rejected too.
    bad_lvlN = jnp.zeros((nlat, nlon + 1, nlev + 1), dtype=bool)
    with pytest.raises(ValueError, match="u_barrier"):
        add_face_barriers(u_mask, v_mask, u_barrier=bad_lvlN)
    # A CORRECT per-level (3-D) barrier passes.
    good_lvl = jnp.zeros((nlat, nlon + 1, nlev), dtype=bool).at[0, 2, 1].set(True)
    um, _ = add_face_barriers(u_mask, v_mask, u_barrier=good_lvl)
    assert float(um[0, 2, 1]) == 0.0 and float(um[0, 2, 0]) == float(u_mask[0, 2, 0])


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


def _minimal_cgrid_state(land_mask_2d, H_bathy_2d):
    """A bare LatLonCGridOceanState from 2-D masks (no grid needed): carve_
    immersed_state only reads land_mask/H_bathy and recomputes the face masks."""
    from legoesm.core.field import Field
    from legoesm.ocean.state import LatLonCGridOceanState
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
    nlat, nlon = land_mask_2d.shape
    um, vm = compute_face_masks(land_mask_2d)
    F = lambda d, n, dims, u="": Field(data=d, name=n, dims=dims, units=u)
    return LatLonCGridOceanState(
        u=F(jnp.zeros((nlat, nlon + 1)), "u", ("y", "x"), "m/s"),
        v=F(jnp.zeros((nlat + 1, nlon)), "v", ("y", "x"), "m/s"),
        T=F(jnp.zeros((nlat, nlon)), "T", ("y", "x"), "degC"),
        S=F(jnp.zeros((nlat, nlon)), "S", ("y", "x"), "PSU"),
        eta=F(jnp.zeros((nlat, nlon)), "eta", ("y", "x"), "m"),
        H_bathy=F(H_bathy_2d, "H_bathy", ("y", "x"), "m"),
        land_mask=F(land_mask_2d, "land_mask", ("y", "x")),
        u_mask=F(um, "u_mask", ("y", "x")),
        v_mask=F(vm, "v_mask", ("y", "x")),
        w=F(jnp.zeros((nlat, nlon)), "w", ("y", "x"), "m/s"),
    )


def test_carve_immersed_state_atomic_H_bathy_and_land_mask():
    """carve_immersed_state keeps state.H_bathy == Σ h_partial AND drops a
    fully-solid column to land (with consistent face masks) — the atomic entry
    that prevents the H_bathy/land_mask desync footgun."""
    from legoesm.ocean.vertical import (
        create_z_star_from_thicknesses, create_partial_cell_coordinate)
    nlat, nlon = 3, 4
    zc = create_z_star_from_thicknesses(jnp.array([10.0, 20.0, 40.0]))
    H = jnp.full((nlat, nlon), 70.0)
    coord = create_partial_cell_coordinate(zc, H)
    state = _minimal_cgrid_state(jnp.ones((nlat, nlon)), jnp.full((nlat, nlon), 70.0))

    # Partial carve at (1,2): deepest level only -> shallower, still wet.
    # Full carve at (0,3): ALL levels solid -> becomes land.
    solid = (jnp.zeros((nlat, nlon, 3), dtype=bool)
             .at[1, 2, 2].set(True)
             .at[0, 3, :].set(True))
    new_state, new_coord = carve_immersed_state(state, coord, solid)

    # ATOMIC invariant: state.H_bathy tracks the carved coordinate exactly.
    np.testing.assert_allclose(
        np.asarray(new_state.H_bathy.data),
        np.asarray(immersed_column_depth(new_coord)))
    # Partial-carve column: shallower (30 m) but STILL OCEAN.
    assert float(new_state.H_bathy.data[1, 2]) == pytest.approx(30.0, rel=1e-9)
    assert float(new_state.land_mask.data[1, 2]) == 1.0
    # Full-carve column: land_mask -> 0 and depth -> 0.
    assert float(new_state.land_mask.data[0, 3]) == 0.0
    assert float(new_state.H_bathy.data[0, 3]) == 0.0
    # Face masks recomputed from the new land_mask (the fully-solid column walls
    # off): both u-faces bordering (0,3) are dry.
    assert float(new_state.u_mask.data[0, 3]) == 0.0
    assert float(new_state.u_mask.data[0, 4]) == 0.0
    # Untouched column unchanged.
    assert float(new_state.H_bathy.data[2, 0]) == pytest.approx(70.0, rel=1e-9)
    assert float(new_state.land_mask.data[2, 0]) == 1.0
