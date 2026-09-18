"""load_subgrid_orography + grid.subgrid_topo_stddev wiring (offline).

The orographic GWD launch reads a per-column ``h_topo`` stddev from the
canonical grid attribute ``subgrid_topo_stddev`` (see
``gravity_wave_drag/integration._extract_subgrid_topo_stddev``); the driver
sets it via ``grid._replace(subgrid_topo_stddev=load_subgrid_orography(...))``
from ``ExperimentConfig.subgrid_orography_path``. These tests lock the loader
output layout and the grid-field plumbing that wiring depends on.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from legoesm.grids.cubed_sphere import CubedSphereGrid, create_cubed_sphere
from legoesm.grids.gaussian import GaussianGrid
from legoesm.grids.topography import load_subgrid_orography


@pytest.fixture(scope="module")
def sso_file(tmp_path_factory):
    """Synthetic 2-deg SSO_STDH: 800 m in a NH mountain box, 0 elsewhere."""
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(1.0, 360.0, 2.0)
    box = ((lat[:, None] >= 25) & (lat[:, None] <= 45)
           & (lon[None, :] >= 70) & (lon[None, :] <= 100))
    sso = np.where(box, 800.0, 0.0)
    ds = xr.Dataset({"SSO_STDH": (("lat", "lon"), sso)},
                    coords={"lat": lat, "lon": lon})
    path = tmp_path_factory.mktemp("sso") / "sso_stdh_2deg.nc"
    ds.to_netcdf(path)
    return str(path)


def test_loader_regrids_to_cube_layout_and_range(sso_file):
    grid = create_cubed_sphere(8)
    sso = load_subgrid_orography(grid, sso_file)
    assert sso.shape == (6, 8, 8)
    assert float(sso.min()) >= 0.0
    # peak preserved to within bilinear smoothing of the box edge
    assert 400.0 < float(sso.max()) <= 800.0 + 1e-6
    # mountains cover ~ (20/180)*(30/360) of the sphere -> most columns zero
    assert float((np.asarray(sso) < 1.0).mean()) > 0.8


def test_grid_field_replace_flows_to_per_column_view(sso_file):
    grid = create_cubed_sphere(8)
    assert grid.subgrid_topo_stddev is None          # default: scalar fallback
    sso = load_subgrid_orography(grid, sso_file)
    grid2 = grid._replace(subgrid_topo_stddev=sso)
    # the physics integration reads getattr(grid, 'subgrid_topo_stddev') and
    # flattens to (ncol,); lock that contract here
    raw = getattr(grid2, "subgrid_topo_stddev", None)
    assert raw is not None
    ncol = 6 * 8 * 8
    col = np.asarray(raw).reshape(-1)[:ncol]
    assert col.shape == (ncol,)
    assert np.array_equal(col, np.asarray(sso).reshape(-1))


def test_both_grid_classes_declare_the_field():
    assert "subgrid_topo_stddev" in CubedSphereGrid._fields
    assert "subgrid_topo_stddev" in GaussianGrid._fields
    # field is defaulted so existing full-arity constructions stay valid
    assert CubedSphereGrid._field_defaults["subgrid_topo_stddev"] is None
    assert GaussianGrid._field_defaults["subgrid_topo_stddev"] is None


def test_loader_missing_variable_raises(tmp_path):
    ds = xr.Dataset({"junk": (("lat", "lon"), np.zeros((4, 8)))},
                    coords={"lat": np.linspace(-60, 60, 4),
                            "lon": np.linspace(0, 315, 8)})
    path = tmp_path / "bad.nc"
    ds.to_netcdf(path)
    grid = create_cubed_sphere(4)
    with pytest.raises(KeyError, match="subgrid orography variable"):
        load_subgrid_orography(grid, str(path))


# ---------------------------------------------------------------------------
# #1712: the file's scale decomposition has to belong to THIS grid
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_sso_scale_reports():
    """The guard reports once per (file, grid) pair per process; tests need a
    clean slate or the second test to use a path would see silence."""
    from legoesm.grids import topography as _topo
    _topo._SSO_SCALE_REPORTED.clear()
    yield
    _topo._SSO_SCALE_REPORTED.clear()


def _sso_ds(block_deg=None, cutoff_deg=None, history=None):
    """A tiny SSO file with a chosen construction record."""
    lat = np.arange(-88.0, 90.0, 4.0)
    lon = np.arange(2.0, 360.0, 4.0)
    ds = xr.Dataset(
        {"SSO_STDH": (("lat", "lon"),
                      np.full((lat.size, lon.size), 100.0))},
        coords={"lat": lat, "lon": lon},
    )
    if block_deg is not None:
        ds.attrs["block_deg"] = float(block_deg)
    if cutoff_deg is not None:
        ds.attrs["resolved_cutoff_deg"] = float(cutoff_deg)
    if history is not None:
        ds.attrs["history"] = history
    return ds


def _write(tmp_path, name, ds):
    path = tmp_path / name
    ds.to_netcdf(path)
    return str(path)


def test_a_file_coarser_than_the_grid_is_reported(tmp_path, caplog):
    """A 2-deg block field on a fine grid double-counts, and now says so.

    `sgh` is the variance the model does NOT resolve. A file whose variance
    runs up to 2 deg, read on a grid whose effective resolution is finer than
    that, hands the gravity-wave scheme orography the model already carries in
    its own topography — the drag is launched from it twice. Before #1712 the
    loader interpolated and clipped, ignoring both the file's construction and
    the grid.
    """
    import logging
    grid = create_cubed_sphere(48)          # ~1.9 deg cells
    path = _write(tmp_path, "coarse.nc", _sso_ds(block_deg=2.0))
    with caplog.at_level(logging.WARNING):
        out = load_subgrid_orography(grid, path)
    assert out.shape == (6, 48, 48)         # still loads; this is a report
    assert any("SECOND time" in r.message for r in caplog.records), caplog.text
    # the message must carry BOTH readings of where the cutoff belongs, so the
    # ambiguity #1712 records does not have to be rediscovered
    assert any("effective resolution" in r.message for r in caplog.records)

    # ... and 'error' refuses outright, naming both scales.
    with pytest.raises(ValueError, match="block_deg"):
        load_subgrid_orography(grid, path, scale_check="error")

    # ... and 'off' is silent, for a caller who accepts the double count.
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        load_subgrid_orography(grid, path, scale_check="off")
    assert not caplog.records


def test_a_file_matched_to_the_grid_is_silent(tmp_path, caplog):
    """Non-vacuity: the check must not fire on a file that IS appropriate."""
    import logging
    grid = create_cubed_sphere(12)          # ~7.5 deg cells
    # block 2 deg is finer than one cell, so nothing is double-counted
    path = _write(tmp_path, "fine_enough.nc", _sso_ds(block_deg=2.0))
    with caplog.at_level(logging.WARNING):
        load_subgrid_orography(grid, path)
    assert not caplog.records, caplog.text

    # an explicit residual cutoff at the grid's own EFFECTIVE resolution --
    # the construction the guard's own message recommends, so rejecting it
    # would make the guard contradict itself (codex review)
    path2 = _write(tmp_path, "residual.nc", _sso_ds(block_deg=2.0,
                                                    cutoff_deg=26.0))
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        load_subgrid_orography(grid, path2)
    assert not caplog.records, caplog.text


def test_the_construction_is_read_from_history_when_unstamped(tmp_path, caplog):
    """Files already on disk predate the attributes; parse their history."""
    import logging
    grid = create_cubed_sphere(48)
    path = _write(tmp_path, "legacy.nc", _sso_ds(history=(
        "prep_subgrid_orography.py --fine-res-deg 1.0 --block-deg 2.0")))
    with caplog.at_level(logging.WARNING):
        load_subgrid_orography(grid, path)
    assert any("2.000 deg" in r.message for r in caplog.records), caplog.text


def test_a_file_with_no_record_at_all_is_reported_as_unknown(tmp_path, caplog):
    """Silence about the decomposition is itself the finding."""
    import logging
    grid = create_cubed_sphere(48)
    path = _write(tmp_path, "bare.nc", _sso_ds())
    with caplog.at_level(logging.WARNING):
        load_subgrid_orography(grid, path)
    assert any("no scale" in r.message for r in caplog.records), caplog.text


def test_unknown_scale_check_mode_raises(tmp_path):
    grid = create_cubed_sphere(12)
    path = _write(tmp_path, "m.nc", _sso_ds(block_deg=2.0))
    with pytest.raises(ValueError, match="scale_check"):
        load_subgrid_orography(grid, path, scale_check="yes")


def test_an_explicit_cutoff_coarser_than_the_effective_resolution_is_reported(
        tmp_path, caplog):
    """An explicit decomposition is judged, just against the looser reading.

    Non-vacuity partner of the acceptance case above: a cutoff BEYOND the
    effective resolution is coarser than anything this grid could call
    subgrid, and must still be reported.
    """
    import logging
    grid = create_cubed_sphere(12)              # ~7.5 deg cells, ~26 effective
    path = _write(tmp_path, "too_coarse.nc", _sso_ds(cutoff_deg=60.0))
    with caplog.at_level(logging.WARNING):
        load_subgrid_orography(grid, path)
    assert any("explicit resolved cutoff" in r.message
               for r in caplog.records), caplog.text


def test_a_partly_stamped_file_still_gets_checked(tmp_path, caplog):
    """One attribute must not suppress the history fallback for the others.

    Attributes used to win WHOLESALE, so a file stamping only
    ``fine_res_deg`` left the guard with no scale to check and it passed in
    silence -- the failure mode the guard exists to remove (codex review).
    """
    import logging
    grid = create_cubed_sphere(48)
    ds = _sso_ds(history=("prep_subgrid_orography.py --fine-res-deg 1.0 "
                          "--block-deg 2.0"))
    ds.attrs["fine_res_deg"] = 1.0              # one attribute, not the scale
    path = _write(tmp_path, "partial.nc", ds)
    with caplog.at_level(logging.WARNING):
        load_subgrid_orography(grid, path)
    assert any("2.000 deg" in r.message for r in caplog.records), caplog.text


def test_history_is_parsed_in_both_flag_spellings(tmp_path, caplog):
    """``--block-deg 2.0`` and ``--block-deg=2.0`` are the same file."""
    import logging
    grid = create_cubed_sphere(48)
    path = _write(tmp_path, "equals.nc", _sso_ds(history=(
        "prep_subgrid_orography.py --fine-res-deg=1.0 --block-deg=2.0")))
    with caplog.at_level(logging.WARNING):
        load_subgrid_orography(grid, path)
    assert any("2.000 deg" in r.message for r in caplog.records), caplog.text


def test_the_generator_stamps_what_the_loader_reads(tmp_path):
    """Generator and loader agree on the record, end to end.

    The construction used to live only in a free-text ``history`` string that
    nothing parsed; if the stamps and the reader ever drift apart the guard
    goes quiet, which is exactly the state #1712 found.
    """
    import sys
    sys.path.insert(0, "scripts/data")
    from legoesm.grids.topography import _sso_file_construction

    elev = xr.Dataset(
        {"elevation": (("lat", "lon"),
                       np.zeros((180, 360), dtype=np.float64))},
        coords={"lat": np.arange(-89.5, 90.0, 1.0),
                "lon": np.arange(0.5, 360.0, 1.0)},
    )
    src = tmp_path / "elev.nc"
    elev.to_netcdf(src)
    out = tmp_path / "sso.nc"

    import prep_subgrid_orography as prep
    prep.main(["--input", str(src), "--out", str(out),
               "--elev-var", "elevation",
               "--fine-res-deg", "1.0", "--block-deg", "4.0"])
    with xr.open_dataset(out) as ds:
        built = _sso_file_construction(ds)
    assert built["block_deg"] == 4.0
    assert built["fine_res_deg"] == 1.0
    assert "resolved_cutoff_deg" not in built


def test_the_report_fires_once_per_file_and_grid(tmp_path, caplog):
    """A guard that shouts on every load is alarm fatigue, not a guard.

    The condition is a property of the (file, grid) PAIR, so it is reported
    once per process -- and the count is asserted, because "warn every time"
    and "warn once" are indistinguishable from a single call (GLM review).
    """
    import logging
    grid = create_cubed_sphere(48)              # ~1.875 deg cells
    # 4 deg against ~1.9 deg cells: a mismatch the measured block sweep can
    # actually price.  (At 2 deg the two scales land in the same sweep row and
    # the hint correctly says nothing rather than quoting a bogus ratio.)
    path = _write(tmp_path, "repeat.nc", _sso_ds(block_deg=4.0))
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            load_subgrid_orography(grid, path)
    hits = [r for r in caplog.records if "SECOND time" in r.message]
    assert len(hits) == 1, f"expected one report, got {len(hits)}"
    # and the message carries the SIZE of the mismatch, not only its existence
    assert "launch stress" in hits[0].message, hits[0].message

    # the marginal case says nothing about size, on purpose
    caplog.clear()
    path2 = _write(tmp_path, "marginal.nc", _sso_ds(block_deg=2.0))
    with caplog.at_level(logging.WARNING):
        load_subgrid_orography(grid, path2)
    marginal = [r for r in caplog.records if "SECOND time" in r.message]
    assert len(marginal) == 1
    assert "launch stress" not in marginal[0].message
