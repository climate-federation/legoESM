"""Direct two-frame test for the coupled C4 ORCA2+SI3 movie."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts/plot/c4_orca2_si3_daily_movie.py"
)
_SPEC = importlib.util.spec_from_file_location("c4_orca2_si3_daily_movie", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
movie = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(movie)


def _synthetic_geometry() -> dict[str, np.ndarray]:
    n_lat, n_lon = 6, 8
    base_lon = np.arange(n_lon, dtype=np.float32) * (360.0 / n_lon)
    glamt = np.broadcast_to(base_lon, (n_lat, n_lon)).copy()
    gphit = np.broadcast_to(
        np.linspace(-88.0, 70.0, n_lat, dtype=np.float32)[:, None],
        (n_lat, n_lon),
    ).copy()
    # A curved bipolar cap with a 180-degree longitude seam in the owned row.
    glamt[-1] = np.array(
        [0.0, 35.0, 70.0, 105.0, 255.0, 290.0, 325.0, 360.0], dtype=np.float32
    )
    gphit[-1] = np.array(
        [72.0, 77.0, 82.0, 87.0, 87.0, 82.0, 77.0, 72.0], dtype=np.float32
    )
    wet = np.ones((n_lat, n_lon), dtype=bool)
    wet[2, 3] = False
    area_t = np.ones((n_lat, n_lon), dtype=np.float32)
    area_t[2, 3] = 0.0
    return {
        "gphit": gphit,
        "glamt": glamt,
        "area_t": area_t,
        "wet": wet,
    }


def _write_month(path: Path, day: int, geometry: dict[str, np.ndarray]) -> None:
    lat = np.deg2rad(geometry["gphit"])
    lon = np.deg2rad(geometry["glamt"])
    smooth = (np.sin(lat) + 0.25 * np.cos(2.0 * lon)).astype(np.float32)
    ice_area = (0.55 + 0.15 * np.sin(lat)).astype(np.float32)
    nemo_sst = (12.0 + smooth + np.float32(day) * 0.01).astype(np.float32)
    candidate_sst = (nemo_sst + 0.2).astype(np.float32)
    ice_area[0, 0] = 0.0
    nemo_volume = (ice_area * (1.2 + 0.1 * smooth)).astype(np.float32)
    candidate_area = (ice_area + 0.03).astype(np.float32)
    candidate_area[0, 0] = 0.0
    candidate_volume = (candidate_area * (1.3 + 0.1 * smooth)).astype(np.float32)
    zeros = np.zeros_like(ice_area)
    np.savez(
        path,
        day=np.asarray([day], dtype=np.int32),
        **geometry,
        nemo_sst=nemo_sst[None],
        candidate_sst=candidate_sst[None],
        nemo_ice_area=ice_area[None],
        candidate_ice_area=candidate_area[None],
        nemo_ice_volume=nemo_volume[None],
        candidate_ice_volume=candidate_volume[None],
        nemo_snow_volume=zeros[None],
        candidate_snow_volume=zeros[None],
    )


def test_two_month_files_render_two_frames_without_tripolar_connectors(tmp_path):
    assert movie._ROWS[0][1] == "Sea surface temperature (°C)"
    geometry = _synthetic_geometry()
    _write_month(tmp_path / "c4_daily_maps_month01.npz", 1, geometry)
    _write_month(tmp_path / "c4_daily_maps_month02.npz", 2, geometry)
    output = tmp_path / "movie.mp4"
    assert movie.write_movie(
        tmp_path, output, fps=2, days=2,
        math_mode="test-math", commit="0123456789abcdef",
    ) == 2
    frame_count, _seconds = imageio_ffmpeg.count_frames_and_secs(output)
    assert frame_count == 2

    with np.load(tmp_path / "c4_daily_maps_month01.npz") as archive:
        fields = {
            key: np.asarray(archive[key][0]) for key in movie._VALUE_KEYS
        }
    rows = movie._row_fields(fields, geometry)
    sst_nemo, sst_candidate, sst_difference, _ = rows["sst_global"]
    np.testing.assert_allclose(
        sst_difference[geometry["wet"]],
        (sst_candidate - sst_nemo)[geometry["wet"]],
        rtol=0.0,
        atol=0.0,
    )
    nemo_thickness, candidate_thickness, thickness_difference, _ = rows[
        "ice_thickness_south"
    ]
    assert np.isnan(nemo_thickness[0, 0])
    assert np.isnan(candidate_thickness[0, 0])
    assert np.isnan(thickness_difference[0, 0])

    # The same smooth field crosses both the longitude seam and curved north
    # fold.  One PathCollection marker is reused at every owned T point: there
    # are no mesh edges or polygons that could bridge either discontinuity.
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    smooth = np.sin(np.deg2rad(geometry["gphit"])) + np.cos(
        np.deg2rad(geometry["glamt"])
    )
    figure = plt.figure()
    axis = figure.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    artist = movie._draw_native_points(
        axis, geometry, smooth, geometry["wet"], cmap="viridis",
        limits=(float(smooth.min()), float(smooth.max())),
    )
    figure.canvas.draw()
    assert artist.get_offsets().shape[0] == int(geometry["wet"].sum())
    assert len(artist.get_paths()) == 1
    np.testing.assert_allclose(artist.get_array(), smooth[geometry["wet"]])
    plt.close(figure)


def test_real_orca2_unique_geometry_excludes_fold_duplicates_and_renders(
    tmp_path: Path,
) -> None:
    from scripts.validate.ocean_fidelity.testcases.nemo_coupled_c_ladder_acquisition import (
        c4_year_runner as runner,
    )

    ref_c4 = Path(
        "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/si3_rounds/round31/"
        "oracle_coupled_c_ladder/REF_C4"
    )
    geo = runner.year_metrics.pilot_metrics.static_geometry(ref_c4)
    raw_wet = np.asarray(geo["tmask"])[..., 0] > 0
    unique_wet = raw_wet & (np.asarray(geo["area_t"]) > 0.0)
    assert int(raw_wet.sum() - unique_wet.sum()) == 35
    real_geometry = {
        "gphit": np.asarray(geo["gphit"], dtype=np.float32),
        "glamt": np.asarray(geo["glamt"], dtype=np.float32),
        "area_t": np.asarray(geo["area_t"], dtype=np.float32),
        "wet": unique_wet,
    }
    path = tmp_path / "c4_daily_maps_month01.npz"
    _write_month(path, 1, real_geometry)
    geometry = movie._geometry(path)
    assert int((geometry["wet"] & (geometry["area_t"] == 0.0)).sum()) == 0
    with np.load(path, allow_pickle=False) as archive:
        fields = {key: np.asarray(archive[key][0]) for key in movie._VALUE_KEYS}
    limits, count = movie._fixed_limits([path], geometry)
    assert count == 1
    frame = movie.render_frame(1, fields, geometry, limits, "real REF_C4 geometry")
    assert frame.shape == (movie.FRAME_SIZE[1], movie.FRAME_SIZE[0], 3)
