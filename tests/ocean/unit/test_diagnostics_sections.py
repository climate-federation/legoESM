"""Direct tests for ``legoesm.ocean.diagnostics_sections``.

These are NON-VACUOUS by construction: each asserts a value that a WRONG
orientation, a WRONG face selection, or a double-counted periodic seam would
change.  Several tests explicitly demonstrate the failure mode they guard by
computing the wrong-sign / wrong-selection answer and asserting it differs.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from legoesm.ocean.diagnostics_sections import (  # noqa: E402
    ARCTIC_GATEWAYS,
    accumulate_gateways,
    arctic_region_mask,
    gateway_face_masks,
    new_gateway_accumulator,
    prepare_gateway_stack,
    region_boundary_faces,
    section_transport,
    upwind_face_values,
)

N_LAT, N_LON, NLEV = 6, 8, 3


def _uniform_grid(dy=2000.0, dx=3000.0):
    dy_u = jnp.full((N_LAT, N_LON + 1), dy)
    dx_v = jnp.full((N_LAT + 1, N_LON), dx)
    return dy_u, dx_v


def _band_region(j0: int) -> jnp.ndarray:
    """Region = all cells with j >= j0 (a zonal band touching the north wall)."""
    r = np.zeros((N_LAT, N_LON), dtype=bool)
    r[j0:, :] = True
    return jnp.asarray(r)


# ---------------------------------------------------------------- selection


def test_zonal_band_selects_only_the_one_straddling_v_row():
    """A zonal band's boundary is exactly one v-row and NO u-faces."""
    faces = region_boundary_faces(_band_region(3))
    assert int(jnp.sum(faces.u_sel)) == 0, "a zonal band has no u-boundary"
    assert int(jnp.sum(faces.v_sel)) == N_LON
    # the straddling row is j0 = 3 (between cell 2 outside and cell 3 inside)
    assert bool(jnp.all(faces.v_sel[3, :]))
    # orientation: the NORTH cell is inside => +1
    assert float(jnp.min(faces.v_sign[3, :])) == 1.0


def test_periodic_seam_face_is_not_double_counted():
    """The i = n_lon face mirrors i = 0 and must never be selected."""
    r = np.zeros((N_LAT, N_LON), dtype=bool)
    r[:, 2:5] = True                      # a meridional block, two u-boundaries
    faces = region_boundary_faces(jnp.asarray(r))
    assert int(jnp.sum(faces.u_sel[:, N_LON])) == 0, (
        "the periodic image face was selected -> the seam is double counted")
    assert int(jnp.sum(faces.u_sel)) == 2 * N_LAT       # west + east edges


def test_region_touching_the_seam_is_still_counted_once():
    """A region straddling i=0 has its seam face selected exactly once."""
    r = np.zeros((N_LAT, N_LON), dtype=bool)
    r[:, 0] = True                        # single column at the seam
    faces = region_boundary_faces(jnp.asarray(r))
    # boundaries at i=0 (with cell n_lon-1 outside) and i=1
    assert int(jnp.sum(faces.u_sel)) == 2 * N_LAT
    assert int(jnp.sum(faces.u_sel[:, N_LON])) == 0


def test_non_periodic_wall_is_never_a_boundary_face():
    r = np.zeros((N_LAT, N_LON), dtype=bool)
    r[:, 0] = True
    f_per = region_boundary_faces(jnp.asarray(r), periodic_x=True)
    f_wall = region_boundary_faces(jnp.asarray(r), periodic_x=False)
    assert int(jnp.sum(f_per.u_sel)) == 2 * N_LAT
    assert int(jnp.sum(f_wall.u_sel)) == N_LAT      # only the i=1 face


# ------------------------------------------------------------- the analytic


def test_uniform_inflow_through_a_straight_section_is_exact():
    """UNIFORM v through a zonal section: transport = v * dx * h * n_lon * nlev.

    This is the analytic case.  It pins the metric (face length x thickness)
    AND the sign, and it fails if either the face selection or the orientation
    is wrong.
    """
    dy_u, dx_v = _uniform_grid()
    v0, h0, dx = 0.25, 10.0, 3000.0
    mfu = jnp.zeros((N_LAT, N_LON + 1, NLEV))
    mfv = jnp.full((N_LAT + 1, N_LON, NLEV), v0 * h0)   # thickness-weighted
    faces = region_boundary_faces(_band_region(3))
    got = section_transport(mfu, mfv, dy_u, dx_v, faces)
    expect = v0 * h0 * dx * N_LON * NLEV
    assert got.volume == pytest.approx(expect, rel=1e-12)
    assert got.volume > 0.0, "northward flow into a northern band must be +"


def test_orientation_flips_sign_for_the_complementary_region():
    """The SAME flow is an outflow for the southern complement.

    A sign error in ``region_boundary_faces`` would make these equal instead of
    opposite -- this test fails loudly in that case.
    """
    dy_u, dx_v = _uniform_grid()
    mfu = jnp.zeros((N_LAT, N_LON + 1, NLEV))
    mfv = jnp.full((N_LAT + 1, N_LON, NLEV), 2.5)
    north = region_boundary_faces(_band_region(3))
    south_mask = jnp.asarray(~np.asarray(_band_region(3)))
    south = region_boundary_faces(south_mask)
    tn = section_transport(mfu, mfv, dy_u, dx_v, north).volume
    ts = section_transport(mfu, mfv, dy_u, dx_v, south).volume
    assert tn == pytest.approx(-ts, rel=1e-12)
    assert tn > 0.0 > ts


def test_flow_entirely_outside_the_section_contributes_nothing():
    """Velocity on non-boundary faces must NOT leak into the section sum."""
    dy_u, dx_v = _uniform_grid()
    mfv = np.zeros((N_LAT + 1, N_LON, NLEV))
    mfv[5, :, :] = 99.0            # a v-row well inside the region
    faces = region_boundary_faces(_band_region(3))
    got = section_transport(jnp.zeros((N_LAT, N_LON + 1, NLEV)),
                            jnp.asarray(mfv), dy_u, dx_v, faces)
    assert got.volume == pytest.approx(0.0, abs=1e-12)


def test_tracer_transport_equals_volume_times_uniform_tracer():
    """With a spatially uniform tracer S0, tracer transport = S0 * volume."""
    dy_u, dx_v = _uniform_grid()
    mfu = jnp.zeros((N_LAT, N_LON + 1, NLEV))
    mfv = jnp.full((N_LAT + 1, N_LON, NLEV), 1.5)
    faces = region_boundary_faces(_band_region(3))
    S0 = 34.5
    tr = jnp.full((N_LAT, N_LON, NLEV), S0)
    tr_u, tr_v = upwind_face_values(tr, mfu, mfv)
    got = section_transport(mfu, mfv, dy_u, dx_v, faces, tr_u, tr_v)
    assert got.tracer == pytest.approx(S0 * got.volume, rel=1e-12)


def test_upwind_picks_the_donor_cell_and_would_differ_from_the_wrong_one():
    """Upwind must take the UPSTREAM cell; the downstream choice differs."""
    tr = np.zeros((N_LAT, N_LON, NLEV))
    tr[2, :, :] = 10.0          # south cell
    tr[3, :, :] = 20.0          # north cell
    mfv_pos = jnp.full((N_LAT + 1, N_LON, NLEV), +1.0)   # northward
    mfv_neg = jnp.full((N_LAT + 1, N_LON, NLEV), -1.0)   # southward
    mfu = jnp.zeros((N_LAT, N_LON + 1, NLEV))
    _, v_pos = upwind_face_values(jnp.asarray(tr), mfu, mfv_pos)
    _, v_neg = upwind_face_values(jnp.asarray(tr), mfu, mfv_neg)
    # face j=3 sits between cell 2 (south) and cell 3 (north)
    assert float(v_pos[3, 0, 0]) == 10.0, "northward flow must carry the SOUTH cell"
    assert float(v_neg[3, 0, 0]) == 20.0, "southward flow must carry the NORTH cell"
    assert float(v_pos[3, 0, 0]) != float(v_neg[3, 0, 0])


# ---------------------------------------------------------------- gateways


def test_gateway_bins_partition_the_boundary_exactly():
    """Sum over gateways must reproduce the whole-boundary transport."""
    dy_u, dx_v = _uniform_grid()
    rng = np.random.default_rng(0)
    mfu = jnp.asarray(rng.normal(size=(N_LAT, N_LON + 1, NLEV)))
    mfv = jnp.asarray(rng.normal(size=(N_LAT + 1, N_LON, NLEV)))
    lat = jnp.asarray(np.linspace(60.0, 80.0, N_LAT)[:, None]
                      * np.ones((1, N_LON)))
    lon = jnp.asarray(np.linspace(-180.0, 170.0, N_LON)[None, :]
                      * np.ones((N_LAT, 1)))
    region = arctic_region_mask(lat, jnp.ones((N_LAT, N_LON)))
    faces = region_boundary_faces(region)
    total = section_transport(mfu, mfv, dy_u, dx_v, faces).volume
    gates = gateway_face_masks(faces, lon)
    parts = sum(float(section_transport(mfu, mfv, dy_u, dx_v, g).volume)
                for g in gates.values())
    assert parts == pytest.approx(float(total), rel=1e-12), (
        "gateway bins do not partition the boundary")
    # and every gateway name is present
    assert set(gates) == {nm for nm, _ in ARCTIC_GATEWAYS}


def test_accumulator_returns_the_time_mean_not_the_sum():
    """Two identical steps must give the SAME mean as one step."""
    dy_u, dx_v = _uniform_grid()
    mfu = jnp.zeros((N_LAT, N_LON + 1, NLEV))
    mfv = jnp.full((N_LAT + 1, N_LON, NLEV), 1.0)
    lat = jnp.asarray(np.linspace(60.0, 80.0, N_LAT)[:, None]
                      * np.ones((1, N_LON)))
    lon = jnp.asarray(np.linspace(-180.0, 170.0, N_LON)[None, :]
                      * np.ones((N_LAT, 1)))
    region = arctic_region_mask(lat, jnp.ones((N_LAT, N_LON)))
    gates = gateway_face_masks(region_boundary_faces(region), lon)
    tr = jnp.full((N_LAT, N_LON, NLEV), 30.0)
    tr_u, tr_v = upwind_face_values(tr, mfu, mfv)
    names = tuple(gates)
    acc1 = accumulate_gateways(new_gateway_accumulator(names),
                               prepare_gateway_stack(gates),
                               mfu, mfv, dy_u, dx_v, tr_u, tr_v)
    acc2 = accumulate_gateways(acc1, prepare_gateway_stack(gates),
                               mfu, mfv, dy_u, dx_v, tr_u, tr_v)
    assert acc1.n == 1 and acc2.n == 2
    np.testing.assert_allclose(np.asarray(acc1.volume_sv),
                               np.asarray(acc2.volume_sv), rtol=1e-12)
    # the SUM did grow -- proving the mean is a real division, not a no-op
    assert float(jnp.sum(acc2.volume)) == pytest.approx(
        2.0 * float(jnp.sum(acc1.volume)), rel=1e-12)


def test_accumulator_time_mean_of_two_different_steps():
    """mean(a, b) == (a + b)/2 -- a real average, not the last sample."""
    dy_u, dx_v = _uniform_grid()
    lat = jnp.asarray(np.linspace(60.0, 80.0, N_LAT)[:, None]
                      * np.ones((1, N_LON)))
    lon = jnp.asarray(np.linspace(-180.0, 170.0, N_LON)[None, :]
                      * np.ones((N_LAT, 1)))
    region = arctic_region_mask(lat, jnp.ones((N_LAT, N_LON)))
    faces = region_boundary_faces(region)
    gates = gateway_face_masks(faces, lon)
    names = tuple(gates)
    mfu = jnp.zeros((N_LAT, N_LON + 1, NLEV))
    tr = jnp.full((N_LAT, N_LON, NLEV), 1.0)
    acc = new_gateway_accumulator(names)
    per_step = []
    for amp in (1.0, 3.0):
        mfv = jnp.full((N_LAT + 1, N_LON, NLEV), amp)
        tr_u, tr_v = upwind_face_values(tr, mfu, mfv)
        acc = accumulate_gateways(acc, prepare_gateway_stack(gates),
                                  mfu, mfv, dy_u, dx_v, tr_u, tr_v)
        per_step.append(
            np.array([float(section_transport(mfu, mfv, dy_u, dx_v,
                                              gates[nm]).volume)
                      for nm in names]))
    expect_sv = 0.5 * (per_step[0] + per_step[1]) / 1.0e6
    np.testing.assert_allclose(np.asarray(acc.volume_sv), expect_sv, rtol=1e-12)


def test_accumulator_is_jit_safe_and_does_not_retrace():
    """The kernel must be traced once for a fixed set of shapes."""
    dy_u, dx_v = _uniform_grid()
    lat = jnp.asarray(np.linspace(60.0, 80.0, N_LAT)[:, None]
                      * np.ones((1, N_LON)))
    lon = jnp.asarray(np.linspace(-180.0, 170.0, N_LON)[None, :]
                      * np.ones((N_LAT, 1)))
    gates = gateway_face_masks(
        region_boundary_faces(arctic_region_mask(
            lat, jnp.ones((N_LAT, N_LON)))), lon)
    names = tuple(gates)
    acc = new_gateway_accumulator(names)
    _stk = prepare_gateway_stack(gates)
    mfu = jnp.zeros((N_LAT, N_LON + 1, NLEV))
    tr = jnp.full((N_LAT, N_LON, NLEV), 34.0)
    from legoesm.ocean import diagnostics_sections as ds
    before = ds._accumulate_jit._cache_size()
    for _ in range(4):
        mfv = jnp.full((N_LAT + 1, N_LON, NLEV), 2.0)
        tr_u, tr_v = upwind_face_values(tr, mfu, mfv)
        acc = accumulate_gateways(acc, prepare_gateway_stack(gates),
                                  mfu, mfv, dy_u, dx_v, tr_u, tr_v)
    after = ds._accumulate_jit._cache_size()
    assert after - before <= 1, f"retraced {after - before} times"
    assert acc.n == 4


def test_as_dict_shape_and_keys():
    names = tuple(nm for nm, _ in ARCTIC_GATEWAYS)
    acc = new_gateway_accumulator(names)
    d = acc.as_dict()
    assert set(d) == set(names)
    assert all(len(v) == 2 for v in d.values())


# ------------------------------------------------- the bin-tiling invariant


def test_gateway_bins_tile_the_circle_exactly():
    """No gap, no overlap, full coverage of [-180, 180).

    Guards the real defect this test suite already caught once: an earlier bin
    set left [-155, -100) uncovered, which drops boundary faces SILENTLY while
    every per-gateway total still looks self-consistent.
    """
    edges = sorted(b for _n, bins in ARCTIC_GATEWAYS for b in bins)
    cursor = -180.0
    for lo, hi in edges:
        assert lo == cursor, f"gap or overlap at {cursor}: next bin starts {lo}"
        assert hi > lo
        cursor = hi
    assert cursor == 180.0, f"bins stop at {cursor}, not 180"


def test_tiling_guard_rejects_a_gapped_bin_set():
    """The guard is NON-VACUOUS: it must reject the exact defect it guards."""
    from legoesm.ocean.diagnostics_sections import _assert_bins_tile
    gapped = (("a", ((-180.0, -155.0),)), ("b", ((-100.0, 180.0),)))
    with pytest.raises(ValueError, match="do not tile"):
        _assert_bins_tile(gapped)
    overlapping = (("a", ((-180.0, 0.0),)), ("b", ((-10.0, 180.0),)))
    with pytest.raises(ValueError):
        _assert_bins_tile(overlapping)
    short = (("a", ((-180.0, 100.0),)),)
    with pytest.raises(ValueError, match="not 180"):
        _assert_bins_tile(short)


# ------------------------------------------------ U-FACE analytics (codex #9)
# Every earlier analytic case set mfu = 0, so a u-face off-by-one or a U-ONLY
# sign inversion passed all of them.  These exercise u-faces with NON-ZERO flux
# and pin the u index convention against the canonical
# ``upwind_cell_to_uface`` ("face i is between cell i-1 and cell i").


def _meridional_block(i0: int, i1: int) -> jnp.ndarray:
    """Region = columns [i0, i1) -- boundary is exactly two u-columns."""
    r = np.zeros((N_LAT, N_LON), dtype=bool)
    r[:, i0:i1] = True
    return jnp.asarray(r)


def test_uniform_zonal_inflow_through_u_faces_is_exact_and_cancels():
    """Uniform eastward flow ENTERS the west face and LEAVES the east face.

    Net must be exactly zero, and each face's own contribution must equal
    +/- u*h*dy*n_lat*nlev.  A u-face off-by-one shifts WHICH column is the
    boundary; a u-only sign flip breaks the +/- pairing.  Both fail here.
    """
    dy_u, dx_v = _uniform_grid()
    u0, h0, dy = 0.5, 8.0, 2000.0
    mfu = jnp.full((N_LAT, N_LON + 1, NLEV), u0 * h0)
    mfv = jnp.zeros((N_LAT + 1, N_LON, NLEV))
    faces = region_boundary_faces(_meridional_block(2, 5))
    net = section_transport(mfu, mfv, dy_u, dx_v, faces).volume
    assert net == pytest.approx(0.0, abs=1e-9), "through-flow must cancel"

    # west face only (i = 2): the EAST cell is inside => sign +1 => inflow
    west = region_boundary_faces(_meridional_block(2, 5))
    only_west = west._replace(
        u_sel=west.u_sel & (jnp.arange(N_LON + 1)[None, :] == 2))
    got_w = section_transport(mfu, mfv, dy_u, dx_v, only_west).volume
    expect = u0 * h0 * dy * N_LAT * NLEV
    assert got_w == pytest.approx(expect, rel=1e-12)
    assert got_w > 0.0, "eastward flow into the block's WEST face must be +"

    # east face only (i = 5): the WEST cell is inside => sign -1 => outflow
    only_east = west._replace(
        u_sel=west.u_sel & (jnp.arange(N_LON + 1)[None, :] == 5))
    got_e = section_transport(mfu, mfv, dy_u, dx_v, only_east).volume
    assert got_e == pytest.approx(-expect, rel=1e-12)


def test_u_face_index_convention_matches_the_canonical_upwind():
    """Face i must lie between cell i-1 and cell i, as the canonical helper says.

    Directly pins the off-by-one: put the region on ONE column and check WHICH
    face indices are selected.
    """
    faces = region_boundary_faces(_meridional_block(3, 4))
    sel = np.asarray(faces.u_sel)
    cols = sorted(set(np.nonzero(sel.any(axis=0))[0].tolist()))
    assert cols == [3, 4], (
        f"u-boundary of column 3 must be faces 3 and 4, got {cols} -- the "
        "face<->cell index convention is off by one")
    assert float(faces.u_sign[0, 3]) == 1.0    # east cell (3) inside
    assert float(faces.u_sign[0, 4]) == -1.0   # west cell (3) inside


def test_u_only_sign_inversion_would_change_a_mixed_flow_answer():
    """A u-ONLY sign flip must change the result of a mixed u+v flow."""
    dy_u, dx_v = _uniform_grid()
    # A UNIFORM mfu cancels west-inflow against east-outflow, which makes a
    # u-only sign flip invisible -- the first draft of this test was vacuous
    # for exactly that reason and said so when it ran.  Use an i-VARYING mfu so
    # the two u boundaries carry different magnitudes and cannot cancel.
    i_ramp = jnp.arange(N_LON + 1, dtype=jnp.float64)[None, :, None]
    mfu = jnp.broadcast_to(1.0 + i_ramp, (N_LAT, N_LON + 1, NLEV))
    mfv = jnp.full((N_LAT + 1, N_LON, NLEV), 1.0)
    r = np.zeros((N_LAT, N_LON), dtype=bool)
    r[3:, 2:5] = True                      # a corner region: u AND v boundary
    faces = region_boundary_faces(jnp.asarray(r))
    good = section_transport(mfu, mfv, dy_u, dx_v, faces).volume
    flipped = section_transport(
        mfu, mfv, dy_u, dx_v, faces._replace(u_sign=-faces.u_sign)).volume
    assert good != pytest.approx(flipped, abs=1e-9), (
        "a u-only sign inversion left the answer unchanged -- the test cannot "
        "detect it")
    # and the corner region really does have BOTH face families
    assert int(jnp.sum(faces.u_sel)) > 0 and int(jnp.sum(faces.v_sel)) > 0


def test_upwind_delegates_to_the_canonical_helper():
    """Our wrapper must return exactly what the canonical helpers return."""
    from legoesm.grids.operators_latlon_cgrid import (
        upwind_cell_to_uface, upwind_cell_to_vface,
    )
    rng = np.random.default_rng(3)
    tr = jnp.asarray(rng.normal(size=(N_LAT, N_LON, NLEV)))
    mfu = jnp.asarray(rng.normal(size=(N_LAT, N_LON + 1, NLEV)))
    mfv = jnp.asarray(rng.normal(size=(N_LAT + 1, N_LON, NLEV)))
    u_ours, v_ours = upwind_face_values(tr, mfu, mfv)
    np.testing.assert_array_equal(np.asarray(u_ours),
                                  np.asarray(upwind_cell_to_uface(tr, mfu)))
    np.testing.assert_array_equal(np.asarray(v_ours),
                                  np.asarray(upwind_cell_to_vface(tr, mfv, None)))


def test_stack_name_mismatch_is_rejected():
    """A stack built from different gateways must not silently mis-assign."""
    dy_u, dx_v = _uniform_grid()
    lat = jnp.asarray(np.linspace(60.0, 80.0, N_LAT)[:, None] * np.ones((1, N_LON)))
    lon = jnp.asarray(np.linspace(-180.0, 170.0, N_LON)[None, :] * np.ones((N_LAT, 1)))
    gates = gateway_face_masks(
        region_boundary_faces(arctic_region_mask(lat, jnp.ones((N_LAT, N_LON)))), lon)
    stack = prepare_gateway_stack(gates)
    bad = new_gateway_accumulator(("only_one",))
    mfu = jnp.zeros((N_LAT, N_LON + 1, NLEV))
    mfv = jnp.zeros((N_LAT + 1, N_LON, NLEV))
    with pytest.raises(ValueError, match="do not match"):
        accumulate_gateways(bad, stack, mfu, mfv, dy_u, dx_v, mfu, mfv)


# ------------------------------------------- codex r2: M9 / L11 / L8 gaps


def test_seam_face_carries_the_right_SIGNED_transport_not_just_selection():
    """codex M9: the seam tests only checked SELECTION, so an implementation
    that flipped only ``u_sign[:, 0]`` passed everything.  Pin the SIGNED
    transport across the periodic seam."""
    dy_u, dx_v = _uniform_grid()
    r = np.zeros((N_LAT, N_LON), dtype=bool)
    r[:, 0] = True                       # single column AT the seam
    faces = region_boundary_faces(jnp.asarray(r))
    u0, dy = 1.0, 2000.0
    mfu = jnp.full((N_LAT, N_LON + 1, NLEV), u0)
    mfv = jnp.zeros((N_LAT + 1, N_LON, NLEV))
    # seam face i=0 only: EAST cell (0) is inside => eastward flow ENTERS => +
    only_seam = faces._replace(
        u_sel=faces.u_sel & (jnp.arange(N_LON + 1)[None, :] == 0))
    got = section_transport(mfu, mfv, dy_u, dx_v, only_seam).volume
    assert got == pytest.approx(u0 * dy * N_LAT * NLEV, rel=1e-12)
    assert got > 0.0, "eastward flow through the seam into the column must be +"
    assert float(faces.u_sign[0, 0]) == 1.0
    # flipping ONLY the seam sign must change the answer (it did not before)
    flipped = only_seam._replace(u_sign=only_seam.u_sign.at[:, 0].multiply(-1.0))
    assert float(section_transport(mfu, mfv, dy_u, dx_v, flipped).volume) \
        == pytest.approx(-float(got), rel=1e-12)


def test_upwind_forwards_the_grid_to_the_canonical_v_helper():
    """codex L11: the delegation test only used grid=None, so DROPPING the
    forwarded grid would still pass.  Assert the grid argument is threaded."""
    import legoesm.ocean.diagnostics_sections as ds
    seen = {}
    import legoesm.grids.operators_latlon_cgrid as ops
    real = ops.upwind_cell_to_vface

    def _spy(f, flux_v, grid=None):
        seen["grid"] = grid
        return real(f, flux_v, grid)

    ops.upwind_cell_to_vface = _spy
    try:
        tr = jnp.zeros((N_LAT, N_LON, NLEV))
        mfu = jnp.zeros((N_LAT, N_LON + 1, NLEV))
        mfv = jnp.zeros((N_LAT + 1, N_LON, NLEV))
        sentinel = object()
        ds.upwind_face_values(tr, mfu, mfv, sentinel)
    finally:
        ops.upwind_cell_to_vface = real
    assert seen["grid"] is sentinel, (
        "the grid argument was not forwarded to upwind_cell_to_vface -- on a "
        "tripole grid that changes the north-fold donor values")


def test_stack_with_mismatched_selector_shape_is_rejected():
    """codex L8: matching NAMES alone must not be accepted as proof."""
    from legoesm.ocean.diagnostics_sections import GatewayStack
    names = ("a", "b")
    acc = new_gateway_accumulator(names)
    bad = GatewayStack(names,
                       jnp.zeros((1, N_LAT, N_LON + 1), dtype=bool),
                       jnp.zeros((1, N_LAT + 1, N_LON), dtype=bool),
                       jnp.zeros((N_LAT, N_LON + 1)),
                       jnp.zeros((N_LAT + 1, N_LON)))
    dy_u, dx_v = _uniform_grid()
    mfu = jnp.zeros((N_LAT, N_LON + 1, NLEV))
    mfv = jnp.zeros((N_LAT + 1, N_LON, NLEV))
    with pytest.raises(ValueError, match="selector rows"):
        accumulate_gateways(bad, bad, mfu, mfv, dy_u, dx_v, mfu, mfv) \
            if False else accumulate_gateways(acc, bad, mfu, mfv, dy_u, dx_v,
                                              mfu, mfv)
