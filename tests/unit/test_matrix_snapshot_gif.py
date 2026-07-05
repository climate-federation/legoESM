"""Direct test for ``_save_snapshot_gifs`` (matrix animated snapshots, #521)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_matrix_module():
    script = (Path(__file__).resolve().parents[2]
              / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")
    name = "_modons_matrix_gif_unit"
    spec = importlib.util.spec_from_file_location(name, script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def test_gif_written_with_one_frame_per_snapshot(tmp_path):
    from PIL import Image

    M = _load_matrix_module()
    n_lat, n_lon = 37, 72
    lat = np.linspace(-90.0, 90.0, n_lat)
    lon = np.linspace(0.0, 360.0 - 360.0 / n_lon, n_lon)
    rng = np.random.default_rng(0)
    snapshots = {
        s: {"height": 5000.0 + rng.standard_normal((n_lat, n_lon))}
        for s in (0, 144, 288)
    }
    M._save_snapshot_gifs(
        tmp_path, "unit_case", snapshots, dt=300.0,
        field_specs=[("height", "h [m]", "viridis")],
        coord_kind="latlon", lon_deg=lon, lat_deg=lat)

    gif = tmp_path / "animation_height.gif"
    assert gif.exists()
    with Image.open(gif) as im:
        assert getattr(im, "n_frames", 1) == 3
        assert im.size[0] > 100 and im.size[1] > 100


def test_gif_skips_missing_or_single_frame_fields(tmp_path):
    M = _load_matrix_module()
    lat = np.linspace(-90.0, 90.0, 19)
    lon = np.linspace(0.0, 350.0, 36)
    snapshots = {0: {"height": np.ones((19, 36))}}   # single frame only
    M._save_snapshot_gifs(
        tmp_path, "unit_case", snapshots, dt=300.0,
        field_specs=[("height", "h", "viridis"), ("absent", "x", "viridis")],
        coord_kind="latlon", lon_deg=lon, lat_deg=lat)
    assert not (tmp_path / "animation_height.gif").exists()
    assert not (tmp_path / "animation_absent.gif").exists()
