"""Unit tests for ``scripts/validate/ocean_fidelity/compare_three_way_nemo.py``.

``main`` does live I/O (three multi-hundred-MB snapshots plus a NEMO grid_T)
and is not exercised here, but every pure helper that decides WHICH cells get
scored is — because that is where this instrument can produce a confident wrong
number.  Each test below pins a specific finding from the adversarial review of
the first version:

* ``_coastal_mask`` padded latitude with land, so every ocean cell in the top
  and bottom rows was "coastal" even with no land anywhere near it;
* ``_nn_wet_mask`` exists because the regridder's coverage flag is a
  distance-to-wet-data flag, not a land/sea classification, and admitted land
  cells within 2.5 deg of ocean into the near-land sub-domain it was meant to
  measure;
* ``_tail`` is what separates "every column too deep" from "a few columns
  convecting to the sea floor", the distinction a bias and an RMSE cannot make.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Normally the scorer in this repo.  C3W_MODULE points the whole module at a
# MUTATED COPY instead, which is how the non-vacuity stage of
# scripts/cluster/omip_nemo/_oracle_record_gate_tests.sbatch proves that
# test_the_scorer_actually_calls_the_gate really does fail when the call it
# claims to pin is deleted.
_VALIDATE = Path(__file__).resolve().parents[2] / "scripts" / "validate"
_MOD = (Path(os.environ["C3W_MODULE"]) if os.environ.get("C3W_MODULE")
        else _VALIDATE / "ocean_fidelity" / "compare_three_way_nemo.py")


def _load():
    # The module inserts scripts/validate on sys.path at import time so it can
    # reuse the NEMO scorecard's regridder; load it by path for the same reason.
    # That path is always the REPO's, even when the module itself is a copy.
    sys.path.insert(0, str(_VALIDATE))
    spec = importlib.util.spec_from_file_location("compare_three_way_nemo", _MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load()


# --------------------------------------------------------------------------
# _coastal_mask
# --------------------------------------------------------------------------
def test_coastal_mask_all_ocean_has_no_coast():
    """REGRESSION: latitude used to be padded with land, so the polar rows of an
    entirely ocean domain were reported as near-land."""
    ocean = np.ones((10, 20), dtype=bool)
    assert m._coastal_mask(ocean, 1).sum() == 0
    assert m._coastal_mask(ocean, 2).sum() == 0


def test_coastal_mask_marks_the_ring_around_an_island():
    ocean = np.ones((11, 21), dtype=bool)
    ocean[5, 10] = False                       # one land cell, far from any edge
    coast = m._coastal_mask(ocean, 1)
    # 8-neighbour dilation of a single land cell -> the 8 surrounding cells,
    # land itself excluded (the mask is intersected with `ocean`).
    assert coast.sum() == 8
    assert coast[4, 9] and coast[6, 11]        # diagonals included
    assert not coast[5, 10]                    # the land cell itself is not ocean
    assert not coast[3, 10]                    # two cells away is not in a 1-halo


def test_coastal_mask_halo_width_is_chebyshev():
    ocean = np.ones((21, 41), dtype=bool)
    ocean[10, 20] = False
    c1 = m._coastal_mask(ocean, 1)
    c2 = m._coastal_mask(ocean, 2)
    assert c1.sum() == 3 * 3 - 1               # 3x3 block minus the land cell
    assert c2.sum() == 5 * 5 - 1               # 5x5 block: "2 cells" is a square halo
    assert c2[8, 18]                           # the far corner of the 5x5 block


def test_coastal_mask_wraps_in_longitude_only():
    ocean = np.ones((7, 9), dtype=bool)
    ocean[3, 0] = False                        # land on the western edge column
    coast = m._coastal_mask(ocean, 1)
    assert coast[3, -1]                        # x is periodic: the far column is adjacent
    # y is not periodic: land in the top row must not create coast in the bottom row
    ocean2 = np.ones((7, 9), dtype=bool)
    ocean2[0, 4] = False
    coast2 = m._coastal_mask(ocean2, 1)
    assert not coast2[-1].any()


def test_coastal_mask_rejects_zero_width():
    with pytest.raises(ValueError, match="coast-cells"):
        m._coastal_mask(np.ones((4, 4), dtype=bool), 0)


# --------------------------------------------------------------------------
# _nn_wet_mask
# --------------------------------------------------------------------------
def test_nn_wet_mask_classifies_by_nearest_source_not_by_distance_to_water():
    """A target cell whose nearest SOURCE cell is land is land, however close
    some other wet cell happens to be — the property the coverage flag lacks."""
    # Source: a 1-degree band along the equator, wet for lon < 180, dry above.
    src_lon = np.arange(0.5, 360.0, 1.0)
    src_lat = np.zeros_like(src_lon)
    src_wet = (src_lon < 180.0).astype(float)
    tgt_lat = np.array([0.0])
    tgt_lon = np.array([90.0, 270.0, 179.6, 180.4])
    wet = m._nn_wet_mask(src_lat, src_lon, src_wet, tgt_lat, tgt_lon)[0]
    assert wet[0]                # deep inside the wet half
    assert not wet[1]            # deep inside the dry half
    assert wet[2]                # just wet-side of the boundary
    assert not wet[3]            # just dry-side, though wet water is 1 deg away


def test_nn_wet_mask_handles_the_dateline_seam():
    """Longitudes are compared on the sphere, so 359.9 and 0.1 are neighbours."""
    src_lon = np.array([0.1, 180.0])
    src_lat = np.array([0.0, 0.0])
    src_wet = np.array([1.0, 0.0])
    wet = m._nn_wet_mask(src_lat, src_lon, src_wet,
                         np.array([0.0]), np.array([359.9]))[0]
    assert wet[0]


# --------------------------------------------------------------------------
# _tail
# --------------------------------------------------------------------------
def test_tail_separates_a_broad_shift_from_a_few_runaway_columns():
    mask = np.ones(100, dtype=bool)
    nemo = np.full(100, 50.0)
    broad = nemo + 20.0                              # every column 20 m deeper
    spiky = nemo.copy(); spiky[:2] = 1050.0          # two columns hit the floor
    tb = m._tail(broad, nemo, mask, 500.0)
    ts = m._tail(spiky, nemo, mask, 500.0)
    # The two cases are constructed to have the SAME mean depth (70 m), which
    # is exactly why a bias cannot tell them apart -- and why the campaign
    # needed the median and the deep fraction to read an Arctic MLD RMSE of
    # 372 m with a bias of +88 m.
    assert np.mean(broad) == pytest.approx(np.mean(spiky))
    assert tb["median_diff"] == pytest.approx(20.0)
    assert ts["median_diff"] == pytest.approx(0.0)   # the median is untouched
    assert tb["frac_a_deeper_than"] == 0.0
    assert ts["frac_a_deeper_than"] == pytest.approx(0.02)
    assert ts["frac_b_deeper_than"] == 0.0


def test_tail_returns_none_on_an_empty_or_all_nan_domain():
    a = np.array([1.0, 2.0]); b = np.array([1.0, 2.0])
    assert m._tail(a, b, np.zeros(2, dtype=bool), 10.0) is None
    assert m._tail(np.array([np.nan, np.nan]), b,
                   np.ones(2, dtype=bool), 10.0) is None


# --------------------------------------------------------------------------
# build_ocean_mask — the decision the helper tests above CANNOT pin
# --------------------------------------------------------------------------
def _one_degree_source(wet_lon_max):
    """A 1-degree equatorial source band, wet west of ``wet_lon_max``."""
    lon = np.arange(0.5, 360.0, 1.0)
    return {"lat": np.zeros_like(lon), "lon": lon,
            "mask": (lon < wet_lon_max).astype(float)}


def test_build_ocean_mask_nearest_excludes_land_that_coverage_admits():
    """The R1-RED-1 regression at the level that MATTERS.

    The per-helper tests would all still pass if ``main`` reverted to
    ``ocean = coverage`` and left ``_nn_wet_mask`` unused, so this pins the
    selection itself: the two modes must disagree, and 'nearest' must be the
    one that drops the land cells.
    """
    tgt_lat = np.array([0.0])
    tgt_lon = np.array([90.0, 179.6, 180.4, 270.0])
    sources = [_one_degree_source(180.0) for _ in range(3)]
    coverage = np.ones((1, 4), dtype=bool)     # regridder says "data is near"
    cov = m.build_ocean_mask(coverage, sources, tgt_lat, tgt_lon, "coverage")
    near = m.build_ocean_mask(coverage, sources, tgt_lat, tgt_lon, "nearest")
    assert cov.all()                            # coverage keeps the land cells
    assert near.tolist() == [[True, True, False, False]]
    assert near.sum() < cov.sum()               # the modes MUST differ


def test_build_ocean_mask_keeps_coverage_in_the_conjunction():
    """Coverage is not redundant: nearest-wet has no distance limit, so a cell
    far outside the regridder's validity radius must still be refused."""
    tgt_lat = np.array([0.0])
    tgt_lon = np.array([90.0, 100.0])
    sources = [_one_degree_source(180.0) for _ in range(3)]
    coverage = np.array([[True, False]])        # second cell out of range
    near = m.build_ocean_mask(coverage, sources, tgt_lat, tgt_lon, "nearest")
    assert near.tolist() == [[True, False]]


def test_build_ocean_mask_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown mask mode"):
        m.build_ocean_mask(np.ones((1, 1), dtype=bool), [_one_degree_source(180.0)],
                           np.array([0.0]), np.array([90.0]), "nearset")


def test_build_ocean_mask_requires_every_source_to_be_wet():
    """One source calling a cell land is enough to drop it — the fields are
    only comparable where all three actually have ocean."""
    tgt_lat = np.array([0.0])
    tgt_lon = np.array([90.0])
    coverage = np.ones((1, 1), dtype=bool)
    sources = [_one_degree_source(180.0), _one_degree_source(180.0),
               _one_degree_source(50.0)]        # third is land at lon 90
    assert not m.build_ocean_mask(coverage, sources, tgt_lat, tgt_lon, "nearest").any()


# --------------------------------------------------------------------------
# _smooth_common_footprint — the control for the regrid-asymmetry concern
# --------------------------------------------------------------------------
def test_smoothing_is_a_mean_over_the_radius_and_respects_validity():
    lat = np.array([0.0])
    lon = np.array([0.0, 1.0, 2.0, 180.0])
    field = np.array([[0.0, 10.0, 20.0, 999.0]])
    valid = np.array([[True, True, True, False]])
    out = m._smooth_common_footprint(field, valid, lat, lon, radius_deg=1.5)
    # cell 1 (lon 1) averages lon 0,1,2 -> 10; the far cell is excluded entirely
    assert out[0, 1] == pytest.approx(10.0)
    assert out[0, 0] == pytest.approx(5.0)      # averages lon 0,1
    assert np.isnan(out[0, 3])


def test_smoothing_removes_short_scale_structure():
    """The point of the control: it must damp the fine structure that one
    source retains and the other has already lost.

    It does NOT claim to preserve the area-weighted mean.  The filter gives every
    target CENTRE equal weight, which on a lat-lon grid is coordinate-area and
    not spherical-area averaging, so a multi-latitude field's cos(lat)-weighted
    mean does move.  Asserting mean preservation would be pinning a property the
    filter does not have."""
    lat = np.array([0.0])
    lon = np.arange(0.5, 360.0, 1.0)
    valid = np.ones((1, lon.size), dtype=bool)
    smooth = np.sin(np.deg2rad(lon))                       # resolved everywhere
    noisy = smooth + 0.5 * (-1.0) ** np.arange(lon.size)   # 2-cell wiggle
    o_s = m._smooth_common_footprint(smooth[None, :], valid, lat, lon, 3.0)
    o_n = m._smooth_common_footprint(noisy[None, :], valid, lat, lon, 3.0)
    assert np.nanstd(o_n - o_s) < 0.5 * np.std(noisy - smooth)
    # Single equatorial row, so every cell has the same area and the unweighted
    # mean IS meaningful here; this is the only geometry in which it is.
    assert abs(np.nanmean(o_n) - np.mean(noisy)) < 1e-3


# --------------------------------------------------------------------------
# _json_safe
# --------------------------------------------------------------------------
def test_json_safe_nulls_non_finite_so_the_report_is_valid_json():
    import json
    r = {"corr": float("nan"), "deep": [1.0, float("inf")], "n": 3, "s": "x"}
    safe = m._json_safe(r)
    assert safe["corr"] is None and safe["deep"] == [1.0, None]
    assert safe["n"] == 3 and safe["s"] == "x"
    json.dumps(safe, allow_nan=False)           # would raise on a bare NaN


# --------------------------------------------------------------------------
# Gaps codex round 3 named: radius bounds, and the zonal support gate
# --------------------------------------------------------------------------
def _args(**kw):
    """A Namespace with valid defaults, overridden per test."""
    import argparse
    base = dict(res_deg=1.0, smooth_radius_deg=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_smoothing_radius_bounds_are_enforced_functionally():
    """A radius above 180 deg is not a wider filter -- 2*sin(r/2) turns back on
    itself, so 270 deg silently becomes a 90 deg chord and the REPORTED width
    would be a lie.  0 must be rejected too: the caller who typed 0 believed
    they requested a filter; 'unsmoothed' is spelled by omitting the flag.
    These CALL the validator (codex r4: a source-text assertion passed while
    the 0 case slipped through a truthiness check)."""
    import pytest
    m._validate_args(_args())                        # None = unsmoothed, OK
    m._validate_args(_args(smooth_radius_deg=4.0))   # in range, OK
    m._validate_args(_args(smooth_radius_deg=180.0)) # boundary, OK
    for bad in (0.0, -5.0, 270.0, float("nan"), float("inf")):
        with pytest.raises(SystemExit):
            m._validate_args(_args(smooth_radius_deg=bad))


def test_res_deg_bounds_are_enforced_functionally():
    import pytest
    m._validate_args(_args(res_deg=2.0))
    for bad in (0.0, -1.0, 7.0, float("nan")):
        with pytest.raises(SystemExit):
            m._validate_args(_args(res_deg=bad))


def test_main_validates_args_before_touching_any_input():
    """main must call _validate_args right after parsing; reverting that call
    (or the build_ocean_mask selection below) must fail this suite."""
    import inspect
    src = inspect.getsource(m.main)
    assert "_validate_args(a)" in src
    assert src.index("_validate_args(a)") < src.index("mkdir")


def test_main_selects_the_ocean_mask_through_build_ocean_mask():
    """REGRESSION TRIPWIRE (codex r3/r4): reverting main to `ocean = coverage`
    passed every helper test.  This string check fails on that exact revert;
    it names the symbol that runs (main) and the call it must make."""
    import inspect
    src = inspect.getsource(m.main)
    assert "ocean = build_ocean_mask(coverage" in src


def test_mld_tails_are_suppressed_when_smoothing():
    """Averaging a deep column with its neighbours before counting columns
    deeper than a threshold does not measure that fraction.  main must not ask
    for tails on a smoothed field."""
    import inspect
    src = inspect.getsource(m.main)
    assert 'name == "MLD" and a.smooth_radius_deg is None' in src


def test_zonal_gate_uses_pre_dropout_availability_not_the_scored_mask():
    """REGRESSION: deriving row availability from the already-dropped-out mask
    made cnt == avail, so every non-empty row passed its own support test and
    the gate was a tautology."""
    import inspect
    src = inspect.getsource(m._plot3)
    assert "avail = avail_row" in src
    assert "_MIN_ZONAL_CELLS_FLOOR" in src
    assert "avail_row" in inspect.signature(m._plot3).parameters


# --------------------------------------------------------------------------
# --also-mask: three grids scored on ONE cell set
# --------------------------------------------------------------------------
def test_build_ocean_mask_takes_any_number_of_sources():
    """A fourth (mask-only) source that is land at a cell must drop it, exactly
    as a scored source would -- the generic reduction --also-mask relies on."""
    tgt_lat = np.array([0.0])
    tgt_lon = np.array([90.0])
    coverage = np.ones((1, 1), dtype=bool)
    three_wet = [_one_degree_source(180.0) for _ in range(3)]
    assert m.build_ocean_mask(coverage, three_wet, tgt_lat, tgt_lon, "nearest").any()
    four = three_wet + [_one_degree_source(50.0)]      # extra source: land at 90E
    assert not m.build_ocean_mask(coverage, four, tgt_lat, tgt_lon, "nearest").any()


def test_main_intersects_also_mask_sources_into_both_footprints():
    """REGRESSION TRIPWIRE: --also-mask must reach BOTH the SST/SSS common mask
    (coverage AND the nearest-wet classification) and the MLD footprint;
    a snapshot that only narrowed one of them would leave the pair runs on
    different cell sets for the other field."""
    import inspect
    src = inspect.getsource(m.main)
    assert "build_ocean_mask(coverage, (T, M, N, *X)" in src
    assert "for _, ocX in sstX:\n        coverage &= ocX > 0.5" in src
    assert "for x, mx in zip(X, mldX_raw):" in src
    assert "mld_ok &= ocXm > 0.5" in src
    # and a mask-only source without MLD geometry is refused, not skipped
    # the message names the field the selected mode needs (window means vs snapshot)
    assert "FATAL: an --also-mask snapshot lacks " in src
    assert "mld_mean (run it with --mld-accumulate)" in src
    assert 'else "z_center_ref/H_bathy")\n                         + ", so the MLD footprint cannot be shared"' in src


# --------------------------------------------------------------------------
# node_cloud_extrapolation — exposure of a land-free source at the coast
# --------------------------------------------------------------------------
def test_node_cloud_check_is_none_for_a_source_that_carries_land():
    src = _one_degree_source(180.0)             # half its cells are land
    scored = np.ones((1, 4), dtype=bool)
    assert m.node_cloud_extrapolation(src, scored, np.array([0.0]),
                                      np.array([10.0, 20.0, 30.0, 40.0])) is None


def test_node_cloud_check_measures_distance_to_the_nearest_node():
    """Nodes every 1 deg along the equator from 0 to 90E (spacing ~111 km).
    A scored cell at 95E is 5 deg (~556 km) from the last node: farther than
    2x the spacing, so it must be counted as 'far'; a cell at 10.5E is not."""
    lon = np.arange(0.0, 90.5, 1.0)
    src = {"lat": np.zeros_like(lon), "lon": lon, "mask": np.ones_like(lon)}
    tgt_lat = np.array([0.0])
    tgt_lon = np.array([10.5, 95.0])
    chk = m.node_cloud_extrapolation(src, np.ones((1, 2), dtype=bool), tgt_lat, tgt_lon)
    assert abs(chk["median_node_spacing_km"] - 111.2) < 1.0
    assert abs(chk["nearest_node_max_km"] - 5 * 111.2) < 2.0
    assert chk["n_far"] == 1 and chk["n_scored"] == 2


def test_main_runs_the_node_cloud_check_on_both_scored_slots():
    import inspect
    src = inspect.getsource(m.main)
    assert "for lab, src in ((lab_a, T), (lab_b, M)):" in src
    assert "node_cloud_extrapolation(src, ocean, tgt_lat, tgt_lon)" in src


# --------------------------------------------------------------------------
# snapshot_day — the day every snapshot must agree on
# --------------------------------------------------------------------------
def test_snapshot_day_prefers_the_time_stamp_over_the_file_name(tmp_path):
    p = tmp_path / "snapshot_day0025.npz"
    np.savez(p, time_days=np.asarray(30.0), T=np.zeros(1))
    assert m.snapshot_day(p) == 30.0                    # stamp wins
    q = tmp_path / "snapshot_day0025_nostamp.npz"
    np.savez(q, T=np.zeros(1))
    assert m.snapshot_day(q) is None                    # name pattern broken -> unknown
    np.savez(tmp_path / "snapshot_day0015.npz", T=np.zeros(1))
    assert m.snapshot_day(tmp_path / "snapshot_day0015.npz") == 15.0
    assert m.snapshot_day(tmp_path / "snapshot_final.npz") is None   # absent file, no day


def test_main_rejects_mixed_days_and_honours_expect_day():
    import inspect
    src = inspect.getsource(m.main)
    assert "snap_days = {str(pth): snapshot_day(pth) for pth in (a.tripole, a.mpas, *a.also_mask)}" in src
    assert "snapshots are from different days" in src
    assert "not the expected day" in src


# --------------------------------------------------------------------------
# per-field finiteness must include the mask-only sources (codex, HIGH)
# --------------------------------------------------------------------------
def test_finite_gate_and_digest_include_mask_only_sources():
    """Behavioural: the exact reduction main applies, on planted NaNs.  A NaN
    carried by a mask-only source on a covered cell must drop that cell from
    the scored set, and the digest must change with it -- otherwise two pair
    runs could 'share' a mask while scoring different cells."""
    import hashlib
    Tg = np.ones((2, 3)); Mg = np.ones((2, 3)); Ng = np.ones((2, 3))
    ar = np.ones((2, 3))
    xf = np.ones((2, 3)); xf[1, 2] = np.nan            # mask-only source NaN
    finite3 = np.isfinite(Tg) & np.isfinite(Mg) & np.isfinite(Ng)
    for f in [xf]:
        finite3 &= np.isfinite(f)
    ar2 = np.where(finite3, ar, 0.0)
    assert int((ar2 > 0).sum()) == 5
    d_with = hashlib.sha1(np.ascontiguousarray(ar2 > 0).tobytes()).hexdigest()
    d_without = hashlib.sha1(np.ascontiguousarray(ar > 0).tobytes()).hexdigest()
    assert d_with != d_without
    import inspect
    src = inspect.getsource(m.main)
    assert "for xf in xs:\n            finite3 &= np.isfinite(xf)" in src
    assert 'report.setdefault("scored_mask_sha1", {})[name]' in src


# --------------------------------------------------------------------------
# _manifest_summary — the caveat must travel for BOTH manifest layouts
# --------------------------------------------------------------------------
def _write_manifest(dirpath, payload):
    import json
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "run_manifest.json").write_text(json.dumps(payload))
    snap = dirpath / "snapshot_day0030.npz"
    np.savez(snap, T=np.zeros(1))
    return snap


def test_manifest_summary_reads_the_nested_layout(tmp_path):
    snap = _write_manifest(tmp_path / "trp", {
        "run": {"command_line": "scripts/run/run_omip_core2.py --grid tripole",
                "creation_time": "2026-09-06T16:21:00"},
        "reproducibility": {"git_dirty": False, "legoesm_version": "0.1",
                            "git_sha": "abc123"}})
    s = m._manifest_summary(snap)
    assert s["manifest_layout"] == "nested"
    assert s["command_line"].endswith("--grid tripole")
    assert s["git_sha"] == "abc123" and s["git_dirty"] is False
    assert "note" not in s


def test_manifest_summary_reads_the_flat_fesom_layout(tmp_path):
    """REGRESSION: the FESOM lane writes argv + top-level git_sha and no
    'run' block, so reading only run.command_line returned all-None for every
    FESOM arm -- the provenance vanished for the grid it matters most for."""
    snap = _write_manifest(tmp_path / "fesom", {
        "lane": "fesom", "git_sha": "def456",
        "argv": ["scripts/run/run_omip_core2.py", "--grid", "fesom",
                 "--fesom-vmix", "legoesm_tke"]})
    s = m._manifest_summary(snap)
    assert s["manifest_layout"] == "flat"
    assert s["command_line"] == ("scripts/run/run_omip_core2.py --grid fesom "
                                 "--fesom-vmix legoesm_tke")
    assert s["git_sha"] == "def456"
    assert "note" not in s


def test_manifest_summary_says_so_when_no_command_line_is_recorded(tmp_path):
    snap = _write_manifest(tmp_path / "bare", {"lane": "mystery"})
    s = m._manifest_summary(snap)
    assert s["command_line"] is None
    assert "NOT recorded" in s["note"]


# --- the oracle record must be the one ending on the day we say we scored ---
#
# The snapshot check in ``main`` pins OUR day and has since 2026-09-15.  The
# ORACLE's day was unpinned: ``--nemo-time-idx`` defaults to -1, the LAST
# record, so a card that simply forgot the flag scored its day-30 state against
# NEMO's day 90 and the difference came back labelled a regression.  These pin
# the gate the user authorised in its place.


def test_matching_record_passes_silently():
    m.check_oracle_record(expect_day=20, nemo_time_idx=3, n_time=18)


def test_the_default_last_record_is_refused_on_a_long_oracle_file():
    # THE DEFECT ITSELF: 18 records is 90 days, -1 is record 17, and the run
    # being scored is at day 20.  Before this gate the run proceeded.
    with pytest.raises(SystemExit) as e:
        m.check_oracle_record(expect_day=20, nemo_time_idx=-1, n_time=18)
    msg = str(e.value)
    assert "record 3" in msg and "record 17" in msg
    assert "--nemo-time-idx 3" in msg          # says what to pass, not just no
    assert "days 15 to 20" in msg and "days 85 to 90" in msg


def test_a_negative_index_that_lands_on_the_right_record_passes():
    # -1 is not wrong in itself: on a file holding exactly the days scored it
    # IS the right record, and refusing it would be a false alarm.
    m.check_oracle_record(expect_day=20, nemo_time_idx=-1, n_time=4)


def test_no_expectation_means_no_check():
    m.check_oracle_record(expect_day=None, nemo_time_idx=-1, n_time=18)


def test_the_monthly_path_is_exempt():
    # --nemo-month selects by month from a monthly file; the 5-day record
    # arithmetic does not apply to it and must not fire.
    m.check_oracle_record(expect_day=20, nemo_time_idx=-1, n_time=18,
                          nemo_month=1)


def test_the_scorer_actually_calls_the_gate(tmp_path, monkeypatch):
    """NON-VACUITY, and the only test here that proves ENFORCEMENT.

    Codex, reviewing the first version: every test above passes with the call
    in ``main`` deleted, because they exercise the helper and nothing asserts
    the wiring.  A gate nobody calls is the same defect it was written to stop.

    So drive ``main`` itself.  The three loaders are replaced with stubs -- the
    real ones read hundreds of megabytes -- and everything between argument
    parsing and the gate is the module's own code.  Deleting the call makes
    this test fail on the stub state instead of on the record mismatch.
    """
    snaps = []
    for nm in ("tri", "mpas"):
        p = tmp_path / f"{nm}_snapshot_day0020.npz"
        np.savez(p, time_days=np.asarray(20.0))
        snaps.append(p)
    gridt = tmp_path / "oracle_grid_T.nc"
    gridt.write_bytes(b"")

    def _stub_lego(path):
        raise AssertionError(f"reached the loader for {path}: the gate did "
                             "not fire, so main no longer calls it")

    monkeypatch.setattr(m, "_load_legoesm", lambda p, use_mean=False: {"sst": np.zeros((2, 2))})
    monkeypatch.setattr(m, "_load_nemo",
                        lambda p, idx, month=None: {"n_time": 18,
                                                    "sst": np.zeros((2, 2))})
    monkeypatch.setattr(m, "_scored", _stub_lego, raising=False)
    monkeypatch.setattr(sys, "argv", [
        "compare_three_way_nemo.py",
        "--tripole", str(snaps[0]), "--mpas", str(snaps[1]),
        "--nemo-gridt", str(gridt), "--out-dir", str(tmp_path / "out"),
        "--expect-day", "20"])          # and NO --nemo-time-idx: the default -1

    with pytest.raises(SystemExit) as e:
        m.main()
    assert "record 3" in str(e.value) and "record 17" in str(e.value)


def test_a_day_that_is_not_a_record_boundary_is_refused():
    # GLM, reviewing the gate above: rounding a non-multiple to the nearest
    # record turns a config typo into a plausible wrong score.  Day 27 rounds
    # to record 4, whose window ENDS on day 25 -- a number that would have been
    # reported as if it were day 27.
    with pytest.raises(SystemExit) as e:
        m.check_oracle_record(expect_day=27, nemo_time_idx=4, n_time=18)
    msg = str(e.value)
    assert "not a multiple" in msg
    assert "days 25 and 30" in msg


def test_the_non_boundary_refusal_fires_before_the_index_check():
    # Even a caller who passes the index the old rounding would have chosen
    # is refused: the day itself is unscoreable, so no index can be right.
    with pytest.raises(SystemExit) as e:
        m.check_oracle_record(expect_day=27, nemo_time_idx=-1, n_time=5)
    assert "not a multiple" in str(e.value)


def test_a_float_day_that_is_a_boundary_still_passes():
    # 20.0 is a boundary written as a float; refusing it would be a false alarm
    # from the tolerance, not from the convention.
    m.check_oracle_record(expect_day=20.0, nemo_time_idx=3, n_time=18)
