"""Direct test for scripts/validate/ocean_grid_consistency.py.

The decisive one is ``test_land_fill_value_does_not_move_the_metric``:
arms disagree about what a LAND cell holds (lat-lon pins 0 degC, MPAS
Neumann-fills ~20 degC), so if the metric moves when the land fill
changes, it is measuring the land convention rather than the dycores --
which is exactly what an earlier revision did.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]


def _load():
    p = _REPO / "scripts" / "validate" / "ocean_grid_consistency.py"
    spec = importlib.util.spec_from_file_location("_ocn_cons", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_ocn_cons"] = mod
    spec.loader.exec_module(mod)
    return mod


def _arm(root: Path, case: str, grid: str, res: str, field_vals,
         land_fill: float, nlat=10, nlon=12, field_vals_t0=None):
    """One arm whose LAND cells hold *land_fill*.

    ``field_vals_t0`` defaults to ``field_vals`` (a static arm); pass it to
    make the initial and final states differ.
    """
    d = root / case / case / grid / res
    d.mkdir(parents=True)
    mask = np.ones((nlat, nlon))
    mask[:2, :] = 0.0                      # polar land, as in the suite
    mask[-2:, :] = 0.0
    sst = np.where(mask > 0.5, field_vals, land_fill)
    sst0 = np.where(mask > 0.5,
                    field_vals if field_vals_t0 is None else field_vals_t0,
                    land_fill)
    np.savez(d / "snapshots_latlon.npz",
             times_days=np.array([0.0, 1.0]),
             lat=np.linspace(-85, 85, nlat),
             lon=np.linspace(0, 360 - 360 / nlon, nlon),
             SST=np.stack([sst0, sst]), eta=np.zeros((2, nlat, nlon)),
             u_sfc=np.zeros((2, nlat, nlon)),
             land_mask=np.stack([mask, mask]))
    (d / "results.txt").write_text("status: PASS\n")


def test_erode_does_not_wrap_across_the_poles():
    """Latitude is NOT periodic: eroding must not pull the south pole's
    neighbour into the north pole's stencil."""
    mod = _load()
    m = np.ones((5, 6), dtype=bool)
    m[0, :] = False                     # land along the SOUTH edge only
    out = mod._erode(m)
    assert not out[1].any(), "row adjacent to south land must be eroded"
    assert out[3].all(), "north rows must be untouched by south land"
    # Longitude DOES wrap: land at column 0 must erode the last column.
    m2 = np.ones((5, 6), dtype=bool)
    m2[:, 0] = False
    out2 = mod._erode(m2)
    assert not out2[2, -1], "longitude must wrap when eroding"


def test_land_fill_value_does_not_move_the_metric(tmp_path):
    rng = np.random.default_rng(0)
    ocean = 10.0 + rng.random((10, 12))
    mod = _load()
    results = []
    for fill_b in (0.0, 20.5):          # lat-lon convention vs MPAS-like
        root = tmp_path / f"fill{fill_b}"
        _arm(root, "geostrophic_adjustment", "latlon", "36x72", ocean, 0.0)
        _arm(root, "geostrophic_adjustment", "mpas", "ico4", ocean, fill_b)
        pairs, _ref = mod.cross_grid_rms(
            root, "geostrophic_adjustment", ["latlon", "mpas"])
        results.append(pairs["latlon|mpas"]["rms_abs"])
    assert results[0] == pytest.approx(results[1], abs=1e-12), (
        f"metric moved when only the LAND fill changed: {results} -- it is "
        f"measuring the land convention, not the dycores")
    assert results[0] < 1e-12, "identical ocean fields must compare equal"


def test_stale_artifact_is_refused(tmp_path, capsys):
    """An artifact at a resolution the matrix no longer registers must be
    ignored loudly, not silently mixed into the comparison."""
    mod = _load()
    ocean = np.full((10, 12), 12.0)
    _arm(tmp_path, "geostrophic_adjustment", "mpas", "ico3", ocean, 0.0)
    assert mod._find(tmp_path, "geostrophic_adjustment", "mpas") is None
    assert "stale artifact ignored" in capsys.readouterr().out


def test_registered_resolution_honours_per_case_overrides():
    """barotropic_wave overrides the per-grid default for latlon."""
    mod = _load()
    assert mod._registered_resolution("barotropic_wave", "latlon") == "48x72"
    assert mod._registered_resolution("geostrophic_adjustment",
                                      "latlon") == "36x72"


# ---------------------------------------------------------------------------
# t=0 control
# ---------------------------------------------------------------------------

def test_initial_difference_is_reported_separately(tmp_path):
    """The pair's t=0 difference must be measured at t=0, not inherited.

    Two arms identical at the final time but DIFFERENT initially must show
    rms_abs = 0 with rms_initial > 0. Without the control it is impossible
    to tell a dycore difference from discretisation of the shared IC --
    which is what the lock-exchange row turned out to be.
    """
    mod = _load()
    same = np.full((10, 12), 12.0)
    _arm(tmp_path, "geostrophic_adjustment", "latlon", "36x72", same, 0.0)
    _arm(tmp_path, "geostrophic_adjustment", "mpas", "ico4", same, 0.0,
         field_vals_t0=np.full((10, 12), 15.0))
    pairs, _ref = mod.cross_grid_rms(tmp_path, "geostrophic_adjustment",
                                     ["latlon", "mpas"])
    p = pairs["latlon|mpas"]
    assert p["rms_abs"] < 1e-12
    assert p["rms_initial"] == pytest.approx(3.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Self-error plumbing
# ---------------------------------------------------------------------------

def test_refine_doubles_every_grid_family_and_refuses_mesh_files():
    mod = _load()
    assert mod._refine("cubed_sphere", "C24") == "C48"
    assert mod._refine("latlon", "36x72") == "72x144"
    # One more icosahedral SUBDIVISION is the 2x step, not 2x the index.
    assert mod._refine("mpas", "ico4") == "ico5"
    # Mesh-file-backed arms ship exactly one mesh: no sibling, say so.
    assert mod._refine("fesom", "pi") is None
    assert mod._refine("tripole", "eorca1") is None


def test_tolerance_prefers_the_measured_self_error(monkeypatch):
    """A case with a measured reference-arm self-error uses it; a case
    without one falls back AND says the number is another case's."""
    mod = _load()
    tol, why = mod._tolerance("geostrophic_adjustment")
    assert tol == mod.SELF_ERROR[(mod.TOLERANCE_ARM, "geostrophic_adjustment")]
    assert "measured" in why
    monkeypatch.delitem(mod.SELF_ERROR,
                        (mod.TOLERANCE_ARM, "geostrophic_adjustment"))
    tol2, why2 = mod._tolerance("geostrophic_adjustment")
    assert tol2 == mod.AGREE_TOL["SST"] and "FALLBACK" in why2


# ---------------------------------------------------------------------------
# Front-position metric (the lock exchange is a discontinuity)
# ---------------------------------------------------------------------------

def test_crossings_finds_the_wrap_around_front():
    """The lock exchange's front sits AT longitude 0, i.e. between the last
    column and the first. A scan over interior pairs only misses it."""
    mod = _load()
    lon = np.arange(0.0, 360.0, 30.0)              # 0, 30, ..., 330
    # The suite's IC: cold WEST of longitude 0, warm east. lon=0 is warm and
    # lon=330 is cold, so one of the two fronts lies on the 330|0 pair.
    dlon = ((lon + 180.0) % 360.0) - 180.0
    row = np.where(dlon < 0.0, 5.0, 30.0)
    xs, n = mod._crossings(lon, row, 17.5)
    assert n == 2, f"expected the two fronts, got {xs}"
    assert any(x == pytest.approx(345.0) for x in xs), (
        f"the wrap-around front (between the last column and the first) was "
        f"missed: {xs}")
    assert any(x == pytest.approx(165.0) for x in xs)


def test_crossings_does_not_double_count_an_exact_plateau():
    """A run of cells sitting EXACTLY at the level is ONE front, not one
    front per cell, AND it sits at the plateau centre.

    The longitude matters: interpolating along the SHORTEST arc instead of
    the forward one puts this crossing at 315 (on top of the other front)
    rather than 135 -- codex round 2, after round 1 fixed only the count.
    """
    mod = _load()
    lon = np.arange(0.0, 360.0, 90.0)
    xs, n = mod._crossings(lon, np.array([-1.0, 0.0, 0.0, 1.0]), 0.0)
    assert n == 2, f"a zero plateau plus the wrap crossing is 2, got {xs}"
    assert sorted(xs) == [pytest.approx(135.0), pytest.approx(315.0)], xs


def test_crossings_degenerate_rows():
    """All-level, one-sample and empty rows produce nothing, not a crash
    and not a spurious front."""
    mod = _load()
    lon = np.arange(0.0, 360.0, 90.0)
    assert mod._crossings(lon, np.zeros(4), 0.0) == ([], 0)
    # Exactly one non-zero sample: there is no pair to cross between.
    assert mod._crossings(lon, np.array([1.0, 0.0, 0.0, 0.0]), 0.0) == ([], 0)


def test_crossings_ignores_a_tangential_touch():
    """Touching the level and returning the same way is NOT a front."""
    mod = _load()
    lon = np.arange(0.0, 360.0, 90.0)
    xs, n = mod._crossings(lon, np.array([-1.0, 0.0, -1.0, 1.0]), 0.0)
    assert n == 2, f"the zero touch at index 1 is not a crossing: {xs}"


def test_crossings_does_not_invent_one_across_land():
    """A NaN (land) between two samples breaks the pair."""
    mod = _load()
    lon = np.arange(0.0, 360.0, 90.0)
    xs, n = mod._crossings(lon, np.array([-1.0, np.nan, 1.0, 1.0]), 0.0)
    # Only the +1 -> -1 wrap pair survives; the -1 | nan | +1 pair does not.
    assert n == 1 and xs[0] == pytest.approx(315.0)


def test_crossings_on_a_row_entirely_one_side_is_empty():
    mod = _load()
    lon = np.arange(0.0, 360.0, 90.0)
    assert mod._crossings(lon, np.full(4, 1.0), 0.0) == ([], 0)
    assert mod._crossings(lon, np.full(4, np.nan), 0.0) == ([], 0)


def test_a_missing_front_is_not_assigned_to_the_other_reference(tmp_path):
    """If land hides the 180E front, the surviving 0E crossing must NOT be
    reported as the 180E front sitting ~180 deg away (codex 2026-08-10)."""
    mod = _load()
    nlat, nlon = 10, 72
    lon = np.linspace(0.0, 360.0 - 360.0 / nlon, nlon)
    dlon = ((lon + 180.0) % 360.0) - 180.0
    fld = np.tile(np.where(dlon < 0.0, 5.0, 30.0), (nlat, 1))
    d = tmp_path / "arm"
    d.mkdir()
    mask = np.ones((nlat, nlon))
    mask[:, (lon > 90.0) & (lon < 270.0)] = 0.0   # land over the 180E front
    np.savez(d / "s.npz", times_days=np.array([0.0, 1.0]),
             lat=np.linspace(-85, 85, nlat), lon=lon,
             SST=np.stack([fld, fld]), land_mask=np.stack([mask, mask]))
    off, _n = mod._front_offsets(d / "s.npz", 17.5, (0.0, 180.0))
    assert np.isfinite(off[0.0]).any(), "the 0E front is still resolved"
    assert not np.isfinite(off[180.0]).any(), (
        "the hidden 180E front must be NaN, not the 0E crossing relabelled")


def test_front_curve_is_not_interpolated_across_a_land_band(tmp_path):
    """np.interp would draw a straight line through a missing latitude
    band; the segment guard must blank it instead."""
    mod = _load()
    nlat, nlon = 21, 72
    lat = np.linspace(-80.0, 80.0, nlat)
    lon = np.linspace(0.0, 360.0 - 360.0 / nlon, nlon)
    dlon = ((lon + 180.0) % 360.0) - 180.0
    fld = np.tile(np.where(dlon < 0.0, 5.0, 30.0), (nlat, 1))
    mask = np.ones((nlat, nlon))
    band = (lat > -20.0) & (lat < 20.0)
    mask[band, :] = 0.0                      # an equatorial land band
    d = tmp_path / "arm"
    d.mkdir()
    np.savez(d / "s.npz", times_days=np.array([0.0, 1.0]), lat=lat, lon=lon,
             SST=np.stack([fld, fld]), land_mask=np.stack([mask, mask]))
    off, _n = mod._front_offsets(d / "s.npz", 17.5, (0.0, 180.0))
    lat_t = np.linspace(-89.0, 89.0, 91)
    inside = np.abs(lat_t) < 15.0
    assert not np.isfinite(off[0.0][inside]).any(), (
        "the front was invented across the land band")
    assert np.isfinite(off[0.0][np.abs(lat_t - 40.0) < 5.0]).any(), (
        "latitudes with real data must survive the guard")


def test_a_single_missing_latitude_row_is_also_not_bridged(tmp_path):
    """ONE missing row, the case a nearest-source distance test cannot
    catch: the 2h gap it leaves has a midpoint only h from a source point
    (codex round 2)."""
    mod = _load()
    nlat, nlon = 21, 72
    lat = np.linspace(-80.0, 80.0, nlat)
    lon = np.linspace(0.0, 360.0 - 360.0 / nlon, nlon)
    dlon = ((lon + 180.0) % 360.0) - 180.0
    fld = np.tile(np.where(dlon < 0.0, 5.0, 30.0), (nlat, 1))
    mask = np.ones((nlat, nlon))
    mask[10, :] = 0.0                        # exactly one dry row (lat 0)
    d = tmp_path / "arm"
    d.mkdir()
    np.savez(d / "s.npz", times_days=np.array([0.0, 1.0]), lat=lat, lon=lon,
             SST=np.stack([fld, fld]), land_mask=np.stack([mask, mask]))
    off, _n = mod._front_offsets(d / "s.npz", 17.5, (0.0, 180.0))
    lat_t = np.linspace(-89.0, 89.0, 91)
    gap = np.abs(lat_t) < 4.0                # inside the single missing row
    assert not np.isfinite(off[0.0][gap]).any(), (
        "one missing row was bridged; the segment guard is not doing its job")


def test_crossings_counts_oscillations():
    """An arm that rings around the mid-temperature produces EXTRA
    crossings; the count is what exposes it."""
    mod = _load()
    lon = np.arange(0.0, 360.0, 30.0)
    row = np.array([5.0, 30.0, 5.0, 30.0, 5.0, 30.0,
                    5.0, 30.0, 5.0, 30.0, 5.0, 30.0])
    _xs, n = mod._crossings(lon, row, 17.5)
    assert n == 12, "a 2-cell oscillation must not read as one clean front"


def test_front_metric_separates_position_from_sharpness(tmp_path):
    """The metric the case needs: a SMEARED front in the SAME place must
    score near-zero separation while the field RMS is large.

    A field RMS cannot make that distinction -- it is the reason this
    metric exists -- so the test asserts both halves.
    """
    mod = _load()
    nlat, nlon = 10, 72
    lon = np.linspace(0.0, 360.0 - 360.0 / nlon, nlon)
    dlon = ((lon + 180.0) % 360.0) - 180.0
    sharp = np.tile(np.where(dlon < 0.0, 5.0, 30.0), (nlat, 1))
    # A cell-centred step crosses the mid-temperature at the CELL EDGE, half
    # a cell west of longitude 0. Centre the tanh there so the two fronts are
    # genuinely co-located and only the WIDTH differs -- otherwise the test
    # would be measuring a half-cell offset and calling it agreement.
    edge = -0.5 * (360.0 / nlon)
    smeared = np.tile(17.5 + 12.5 * np.tanh((dlon - edge) / 15.0), (nlat, 1))
    _arm(tmp_path, "lock_exchange", "latlon", "36x72", sharp, 0.0,
         nlat=nlat, nlon=nlon)
    _arm(tmp_path, "lock_exchange", "mpas", "ico4", smeared, 0.0,
         nlat=nlat, nlon=nlon)
    fp = mod.front_position(tmp_path, ["latlon", "mpas"])
    r0, r1 = fp["refs_deg"]
    pair = fp["pairs"]["latlon|mpas"]
    # Both arms are static, so displacement separation is exactly zero and
    # the mesh-phase floor carries any residual position difference.
    assert pair[f"front_{r0:.0f}E_sep_km"] == pytest.approx(0.0, abs=1e-9)
    assert pair[f"front_{r0:.0f}E_init_sep_km"] < 50.0, (
        "the two fronts are at the same longitude; only the width differs")
    # ... while the field RMS calls them wildly different.
    pairs, _ref = mod.cross_grid_rms(tmp_path, "lock_exchange",
                                     ["latlon", "mpas"])
    assert pairs["latlon|mpas"]["rms_abs"] > 2.0, (
        "control: the field RMS DOES see the smearing (2.2 degC here, the "
        "same size as the suite's real lock-exchange rows) while the fronts "
        "coincide -- which is why the RMS cannot be read as a statement "
        "about front position")


def test_displacement_differencing_removes_mesh_phase(tmp_path):
    """The point of differencing displacements, proved non-vacuously.

    The two arms start with the front 10 deg apart (mesh phase) and BOTH
    move it 5 deg east. The displacement separation must be ~0 while the
    reported mesh-phase floor stays ~10 deg worth of km -- i.e. the raw
    final-position comparison would have called them 1000 km apart
    (codex 2026-08-10: the earlier tests gave both arms the same initial
    front, so they could not tell the two metrics apart).
    """
    mod = _load()
    nlat, nlon = 10, 72
    lon = np.linspace(0.0, 360.0 - 360.0 / nlon, nlon)

    def _step(centre):
        d = ((lon - centre + 180.0) % 360.0) - 180.0
        return np.tile(np.where(d < 0.0, 5.0, 30.0), (nlat, 1))

    _arm(tmp_path, "lock_exchange", "latlon", "36x72", _step(5.0), 0.0,
         nlat=nlat, nlon=nlon, field_vals_t0=_step(0.0))
    _arm(tmp_path, "lock_exchange", "mpas", "ico4", _step(15.0), 0.0,
         nlat=nlat, nlon=nlon, field_vals_t0=_step(10.0))
    fp = mod.front_position(tmp_path, ["latlon", "mpas"])
    r0 = fp["refs_deg"][0]
    pair = fp["pairs"]["latlon|mpas"]
    assert pair[f"front_{r0:.0f}E_sep_km"] == pytest.approx(0.0, abs=1.0), (
        "both arms moved the front by the same 5 deg; the displacement "
        "separation must be ~0")
    assert pair[f"front_{r0:.0f}E_init_sep_km"] > 500.0, (
        "the 10-deg mesh phase must still be reported as the floor")


def test_front_metric_sees_a_displaced_front(tmp_path):
    """Non-vacuity: move one arm's front and the separation must grow."""
    mod = _load()
    nlat, nlon = 10, 72
    lon = np.linspace(0.0, 360.0 - 360.0 / nlon, nlon)
    dlon = ((lon + 180.0) % 360.0) - 180.0
    base = np.tile(np.where(dlon < 0.0, 5.0, 30.0), (nlat, 1))
    shifted = np.tile(np.where(((lon - 20.0 + 180.0) % 360.0) - 180.0 < 0.0,
                               5.0, 30.0), (nlat, 1))
    _arm(tmp_path, "lock_exchange", "latlon", "36x72", base, 0.0,
         nlat=nlat, nlon=nlon)
    # Same IC as latlon, final state displaced 20 deg east.
    _arm(tmp_path, "lock_exchange", "mpas", "ico4", shifted, 0.0,
         nlat=nlat, nlon=nlon, field_vals_t0=base)
    fp = mod.front_position(tmp_path, ["latlon", "mpas"])
    r0 = fp["refs_deg"][0]
    # 20 deg at the equator is ~2200 km; the cos-lat weighting over a
    # +-70 deg basin brings the weighted RMS down but not near zero.
    assert fp["pairs"]["latlon|mpas"][f"front_{r0:.0f}E_sep_km"] > 500.0


# ---------------------------------------------------------------------------
# The unstructured -> lat-lon regrid (run_ocean_test_matrix)
# ---------------------------------------------------------------------------

def _matrix():
    import importlib.util as _iu
    p = _REPO / "scripts" / "matrix" / "run_ocean_test_matrix.py"
    if "_rm_regrid" in sys.modules:
        return sys.modules["_rm_regrid"]
    spec = _iu.spec_from_file_location("_rm_regrid", p)
    mod = _iu.module_from_spec(spec)
    sys.modules["_rm_regrid"] = mod
    sys.path.insert(0, str(p.parent))
    spec.loader.exec_module(mod)
    return mod


def _fib_sphere(n):
    """Quasi-uniform points on the sphere (Fibonacci spiral).

    A real mesh is quasi-uniform; POISSON-random points are not, and their
    sparse patches trip a nearest-source cutoff that a mesh never would.
    Using random points here made the cutoff test fail for the fixture's
    reason rather than the code's (measured: 19% of interior targets
    dropped).
    """
    i = np.arange(n) + 0.5
    lat = np.degrees(np.arcsin(1.0 - 2.0 * i / n))
    lon = np.mod(i * 180.0 * (1.0 + 5.0 ** 0.5), 360.0)
    return lon, lat


def _cap_mesh(lat_max=60.0, n=6000):
    """A quasi-uniform point set that COVERS ONLY |lat| <= lat_max.

    Stands in for the FESOM pi mesh, which reaches -78.5 deg in the south
    while the 181x360 target runs to -90.
    """
    lon, lat = _fib_sphere(n)
    keep = np.abs(lat) <= lat_max
    return lon[keep], lat[keep]


def test_regrid_refuses_to_extrapolate_past_the_source_mesh():
    """Targets beyond the mesh must be NaN, not the nearest node's value.

    Without a target-side cutoff every row poleward of the mesh takes the
    nearest wet point however far away and the weights renormalise to 1,
    so extrapolation is returned looking like data (codex 2026-08-10).
    """
    M = _matrix()
    lon, lat = _cap_mesh(60.0)
    vals = 5.0 * np.cos(np.radians(lat))
    idxs, w = M._build_latlon_weights(lon, lat, 181, 360)
    out = M._apply_weights(vals, idxs, w, 181, 360, limit=True)
    lat_t = np.linspace(-90.0, 90.0, 181)
    assert np.isnan(out[np.abs(lat_t) > 75.0]).all(), (
        "targets far outside the source mesh must be NaN")
    assert np.isfinite(out[np.abs(lat_t) < 50.0]).all(), (
        "targets well inside the mesh must survive the cutoff")


def test_land_mask_regrid_uses_the_same_cutoff():
    """Otherwise the mask calls the uncovered cap ocean while the field
    there is NaN -- which is how one arm ended up with 5% more ocean than
    the arms it was being compared against, cell by cell."""
    M = _matrix()
    lon, lat = _cap_mesh(60.0)
    wet = np.ones_like(lat)                       # every SOURCE point wet
    out = M._regrid_land_mask(wet, lon, lat, "fesom")
    lat_t = np.linspace(-90.0, 90.0, 181)
    assert (out[np.abs(lat_t) > 75.0] == 0.0).all(), (
        "the mask must not extend the ocean past the mesh")
    assert (out[np.abs(lat_t) < 50.0] == 1.0).all()


def test_linear_weights_are_exact_for_a_linear_field():
    """The property inverse-distance weighting lacks.

    A field linear in 3-D position is reproduced exactly by the plane fit;
    IDW leaves an error that does not vanish with more neighbours.
    """
    M = _matrix()
    lon, lat = _fib_sphere(3000)
    d2r = np.pi / 180.0
    xyz = np.column_stack([np.cos(lat * d2r) * np.cos(lon * d2r),
                           np.cos(lat * d2r) * np.sin(lon * d2r),
                           np.sin(lat * d2r)])
    coef = np.array([0.3, -0.7, 1.1])
    vals = xyz @ coef
    truth = None
    for linear in (True, False):
        idxs, w = M._build_latlon_weights(lon, lat, 91, 180, k=8,
                                          linear=linear)
        out = M._apply_weights(vals, idxs, w, 91, 180, limit=linear)
        if truth is None:
            lat_t = np.linspace(-90.0, 90.0, 91)
            lon_t = np.linspace(0.0, 360.0, 180)
            lo, la = np.meshgrid(lon_t, lat_t)
            truth = (np.column_stack([
                np.cos(la.ravel() * d2r) * np.cos(lo.ravel() * d2r),
                np.cos(la.ravel() * d2r) * np.sin(lo.ravel() * d2r),
                np.sin(la.ravel() * d2r)]) @ coef).reshape(91, 180)
        err = np.nanmax(np.abs(out - truth))
        if linear:
            assert err < 5e-3, f"linear fit should be near-exact, got {err:.2e}"
        else:
            assert err > 1e-2, (
                "control: IDW must NOT be exact here, or the test above "
                "proves nothing")


def test_linear_weights_cannot_overshoot_a_step():
    """Signed weights can leave the data range at a discontinuity; the
    limiter is what keeps the lock exchange's 25 K step bounded."""
    M = _matrix()
    lon, lat = _fib_sphere(4000)
    dlon = ((lon + 180.0) % 360.0) - 180.0
    vals = np.where(dlon < 0.0, 5.0, 30.0)
    idxs, w = M._build_latlon_weights(lon, lat, 91, 180, k=8)
    lim = M._apply_weights(vals, idxs, w, 91, 180, limit=True)
    raw = M._apply_weights(vals, idxs, w, 91, 180, limit=False)
    assert np.nanmin(lim) >= 5.0 - 1e-9 and np.nanmax(lim) <= 30.0 + 1e-9, (
        f"limited output left the data range: "
        f"[{np.nanmin(lim):.3f}, {np.nanmax(lim):.3f}]")
    assert np.nanmax(raw) > 30.0 + 1e-6 or np.nanmin(raw) < 5.0 - 1e-6, (
        "control: the unlimited fit MUST overshoot here, or the limiter "
        "test proves nothing")


# ---------------------------------------------------------------------------
# Fields the two arms do not reduce the same way
# ---------------------------------------------------------------------------
#
# The by-field decomposition compares each saved field across arms. For
# surface speed the two arms build the field with DIFFERENT operators --
# lat-lon takes a 2-point mean of the staggered components, MPAS does a
# Perot reconstruction over ~6 edges -- so the row contains the difference
# between the reductions as well as any difference in the flow. That has to
# be stated where the number is printed, or the row reads as a cross-grid
# result (2026-08-13).

def test_surface_speed_is_flagged_as_not_comparable():
    M = _load()
    assert "speed_sfc" in M._NOT_LIKE_FOR_LIKE
    why = M._NOT_LIKE_FOR_LIKE["speed_sfc"]
    # The reason must name BOTH reductions, not just assert a caveat.
    assert "Perot" in why and "2-point" in why


def test_every_flagged_field_is_one_the_decomposition_actually_reports():
    """A caveat for a field that is never printed protects nothing."""
    M = _load()
    assert set(M._NOT_LIKE_FOR_LIKE) <= set(M._DECOMPOSE_FIELDS)


def test_the_fields_that_are_comparable_are_not_flagged():
    """NON-VACUITY: the flag must distinguish, not blanket everything."""
    M = _load()
    for fld in ("eta", "SST"):
        assert fld not in M._NOT_LIKE_FOR_LIKE, (
            f"{fld} is reduced identically on both arms and must stay "
            f"readable as a cross-grid number")
