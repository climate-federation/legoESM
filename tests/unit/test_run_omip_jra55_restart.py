"""Tests for restart-save and progress plotting on the JRA55-do path.

Covers:

- ``run_omip._save_restart`` writes an npz with all state fields the
  global-overturning plotter expects.
- ``_run_omip_loop(checkpoint_days=...)`` actually triggers restart
  saves at the right cadence.
- ``scripts/plot/plot_jra55_tropical_progress.py`` reads those restarts
  and produces the four progress PNGs without crashing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

# Reuse the Day-2/3 test fixtures.
_DAY23 = (
    Path(__file__).resolve().parent / "test_run_omip_jra55_dispatch.py"
)
_spec = importlib.util.spec_from_file_location("_day23_for_restart", _DAY23)
_day23 = importlib.util.module_from_spec(_spec)
sys.modules["_day23_for_restart"] = _day23
_spec.loader.exec_module(_day23)

run_omip = _day23.run_omip
_make_synthetic_cache = _day23._make_synthetic_cache
_make_tiny_latlon_setup = _day23._make_tiny_latlon_setup
_make_woa_like_targets = _day23._make_woa_like_targets
_argparse_namespace = _day23._argparse_namespace


# ============================================================================
# _save_restart
# ============================================================================

def test_save_restart_writes_expected_npz(tmp_path):
    grid, z_coord, _, _, _ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    out = tmp_path / "out"
    fname = run_omip._save_restart(state, day=10.0, step=2880, output_dir=out)
    assert fname.exists()
    assert fname.name == "restart_day000010.npz"

    data = np.load(fname, allow_pickle=False)
    # Required scalar metadata
    assert int(data["step"]) == 2880
    assert float(data["time_days"]) == 10.0
    assert str(data["grid_type"]) == "latlon"
    # Required state fields the progress plotter consumes
    for key in ("T", "S", "eta", "u", "v", "land_mask"):
        assert key in data.files, f"missing {key} in restart"


def test_save_restart_filename_zero_pads_to_six_digits(tmp_path):
    grid, z_coord, _, _, _ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    fname = run_omip._save_restart(state, day=5.0, step=1, output_dir=tmp_path)
    assert fname.name == "restart_day000005.npz"


def test_save_restart_labels_non_latlon_grid_type(tmp_path):
    """The npz grid_type label matches the run's grid selection (S10).

    Historically _save_restart hardcoded 'latlon', mislabeling MPAS/tripole
    checkpoints.  The label is provenance-only (nothing reads it on load),
    so we exercise the label selection directly on a cheap state: the field
    dump is grid-agnostic and only the label depends on grid_type.
    """
    grid, z_coord, _, _, _ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    for gt in ("tripole", "mpas"):
        out = tmp_path / gt
        fname = run_omip._save_restart(state, day=1.0, step=1, output_dir=out,
                                       grid_type=gt)
        data = np.load(fname, allow_pickle=False)
        assert str(data["grid_type"]) == gt


def test_save_restart_rejects_unknown_grid_type(tmp_path):
    grid, z_coord, _, _, _ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    with pytest.raises(ValueError, match="grid_type"):
        run_omip._save_restart(state, day=1.0, step=1, output_dir=tmp_path,
                               grid_type="ico")


# ============================================================================
# _run_omip_loop checkpointing
# ============================================================================

def test_run_omip_loop_requires_checkpoint_dir(tmp_path):
    """checkpoint_days without checkpoint_dir is a programmer error."""
    grid, z_coord, _, model, _ = _make_tiny_latlon_setup(n_lat=4, n_lon=8)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    with pytest.raises(ValueError, match="checkpoint_dir"):
        run_omip._run_omip_loop(
            model, state, "latlon", grid, z_coord,
            dt=300.0, n_steps=1, diag_every=1,
            checkpoint_days=1.0, checkpoint_dir=None,
        )


def test_run_omip_loop_writes_restarts_at_cadence(tmp_path, monkeypatch):
    """Cadence ⇒ N restart files at the right simulation days."""
    n_lat, n_lon = 4, 8
    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon,
                                   n_records=200)
    grid, z_coord, _, model, _ = _make_tiny_latlon_setup(
        n_lat=n_lat, n_lon=n_lon,
    )
    T_woa, S_woa = _make_woa_like_targets(grid, nlev=4)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _argparse_namespace(jra55_cache=str(cache))
    js = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon",
        z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
    )

    # S10 threading spy: the loop must pass its grid_type explicitly to
    # _save_restart (latlon equals the legacy default, so asserting the
    # npz label alone would not prove the loop threads it).
    seen_grid_types = []
    real_save = run_omip._save_restart

    def _spy_save(state, day, step, output_dir, **kwargs):
        seen_grid_types.append(kwargs.get("grid_type"))
        return real_save(state, day, step, output_dir, **kwargs)

    monkeypatch.setattr(run_omip, "_save_restart", _spy_save)

    out = tmp_path / "run_out"
    # 6-hour runs at dt=10800 s = 3 h: 2 steps = 6 hours = 0.25 day.
    # checkpoint every 0.125 day → expect restarts at step 1 and step 2,
    # i.e. days 0, 0.125, 0.25.
    state, diag, wall, ok, _ = run_omip._run_omip_loop(  # +blowup_info (5-tuple)
        model, state, "latlon", grid, z_coord,
        dt=10800.0, n_steps=2, diag_every=1,
        jra55_state=js,
        checkpoint_days=0.125,
        checkpoint_dir=out,
    )
    files = sorted(out.glob("restart_day*.npz"))
    assert len(files) >= 1, f"no restarts written; got {files}"
    # Final step should always trigger a save.
    days_written = sorted(int(p.stem.removeprefix("restart_day")) for p in files)
    # At least one restart for the final day (day 0 = step 2 × 3h = 0.25 d ≈ 0)
    assert ok, "loop reported failure"
    # S10: every save call received the loop's grid_type explicitly.
    assert seen_grid_types, "spy never saw a _save_restart call"
    assert all(g == "latlon" for g in seen_grid_types), seen_grid_types
    data = np.load(files[-1], allow_pickle=False)
    assert str(data["grid_type"]) == "latlon"


def test_run_omip_loop_no_checkpoint_when_disabled(tmp_path):
    """checkpoint_days=None ⇒ no restart files written."""
    n_lat, n_lon = 4, 8
    cache = _make_synthetic_cache(tmp_path, n_lat=n_lat, n_lon=n_lon,
                                   n_records=8)
    grid, z_coord, _, model, _ = _make_tiny_latlon_setup(
        n_lat=n_lat, n_lon=n_lon,
    )
    T_woa, S_woa = _make_woa_like_targets(grid, nlev=4)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)
    args = _argparse_namespace(jra55_cache=str(cache))
    js = run_omip._setup_jra55_forcing_state(
        args, grid, "latlon",
        z_coord=z_coord, T_woa=T_woa, S_woa=S_woa,
    )

    out = tmp_path / "no_cp"
    out.mkdir()
    run_omip._run_omip_loop(
        model, state, "latlon", grid, z_coord,
        dt=300.0, n_steps=1, diag_every=1,
        jra55_state=js,
        checkpoint_days=None,
        checkpoint_dir=None,
    )
    assert list(out.glob("restart_day*.npz")) == []


# ============================================================================
# Progress plotter — end-to-end on a tiny synthetic run
# ============================================================================

def _load_progress_plotter():
    """Import the plot script as a module."""
    plot_path = (
        Path(__file__).resolve().parents[2]
        / "scripts" / "plot" / "plot_jra55_tropical_progress.py"
    )
    spec = importlib.util.spec_from_file_location(
        "plot_jra55_tropical_progress", plot_path,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plot_jra55_tropical_progress"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_progress_plotter_runs_on_synthetic_restarts(tmp_path):
    """The plotter consumes restart npzs and produces 4 PNGs without
    raising — even on a tiny grid with only a handful of restarts."""
    n_lat, n_lon = 4, 8
    grid, z_coord, _, _, _ = _make_tiny_latlon_setup(n_lat=n_lat, n_lon=n_lon)
    state = run_omip._init_rest_state("latlon", grid, z_coord, H_max=1000.0)

    # Hand-write three restarts so the plotter has multiple snapshots.
    run_dir = tmp_path / "restart_run"
    for day in (0, 5, 10):
        run_omip._save_restart(state, day=float(day), step=day,
                                output_dir=run_dir)

    plot_mod = _load_progress_plotter()
    rc = plot_mod.main(["--run-dir", str(run_dir)])
    assert rc == 0
    # Four PNGs landed alongside the npzs.
    expected = [
        "timeseries_progress.png",
        "snapshots_progress.png",
        "moc_progress.png",
        "barotropic_streamfunction_progress.png",
    ]
    for name in expected:
        assert (run_dir / name).exists(), f"missing {name}"


def test_progress_plotter_handles_missing_run_dir(tmp_path):
    plot_mod = _load_progress_plotter()
    missing = tmp_path / "no_such_dir"
    rc = plot_mod.main(["--run-dir", str(missing)])
    assert rc == 1


def test_progress_plotter_handles_empty_run_dir(tmp_path):
    plot_mod = _load_progress_plotter()
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = plot_mod.main(["--run-dir", str(empty)])
    assert rc == 1
