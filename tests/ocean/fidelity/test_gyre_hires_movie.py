import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "scripts/validate/ocean_fidelity/testcases"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT = SCRIPT_DIR / "nemo_testcase_l2_gyre_hires_movie.py"
SPEC = importlib.util.spec_from_file_location("gyre_hires_movie", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MOVIE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOVIE)


def test_common_scales_are_shared_and_signed_fields_are_symmetric(monkeypatch):
    frames = [MOVIE.DailyFrame(1, np.ones((2, 2)), np.ones((2, 2)),
                               np.ones((2, 2)), np.ones((2, 2)))]
    providers = (
        type("P", (), {"iterate": lambda self: iter(frames)})(),
        type("P", (), {"iterate": lambda self: iter(frames)})(),
    )
    values = {
        "relative vorticity / f": np.array([[-2.0, 1.0]]),
        "SSH (m)": np.array([[-1.0, 3.0]]),
        "surface speed (m/s)": np.array([[1.0, 4.0]]),
        "SST (C)": np.array([[8.0, 12.0]]),
    }
    monkeypatch.setattr(MOVIE, "movie_fields", lambda *args: values)
    scales = MOVIE.common_scales(
        providers, {"wet": np.ones((1, 2), dtype=bool)}, object())
    assert scales["relative vorticity / f"] == (-2.0, 2.0)
    assert scales["SSH (m)"] == (-3.0, 3.0)
    assert scales["surface speed (m/s)"] == (0.0, 4.0)


def test_common_scales_refuses_a_derived_nan_on_a_wet_pixel(monkeypatch):
    frames = [MOVIE.DailyFrame(1, *(np.ones((1, 2)) for _ in range(4)))]
    provider = type("P", (), {"iterate": lambda self: iter(frames)})()
    values = {
        "relative vorticity / f": np.array([[0.0, np.nan]]),
        "SSH (m)": np.ones((1, 2)),
        "surface speed (m/s)": np.ones((1, 2)),
        "SST (C)": np.ones((1, 2)),
    }
    monkeypatch.setattr(MOVIE, "movie_fields", lambda *args: values)
    with np.testing.assert_raises_regex(Exception, "non-finite wet movie pixel"):
        MOVIE.common_scales(
            (provider,), {"wet": np.ones((1, 2), dtype=bool)}, object())


def test_render_frame_has_encoder_shape(monkeypatch):
    frame = MOVIE.DailyFrame(7, *(np.ones((2, 2)) for _ in range(4)))
    values = {
        "relative vorticity / f": np.ones((2, 2)),
        "SSH (m)": np.ones((2, 2)),
        "surface speed (m/s)": np.ones((2, 2)),
        "SST (C)": np.ones((2, 2)),
    }
    monkeypatch.setattr(MOVIE, "movie_fields", lambda *args: values)
    scales = {name: (0.0, 2.0) for name in values}
    image = MOVIE.render_frame((frame, frame), ("NEMO", "legoESM"),
                               scales, {}, object())
    assert image.shape == (640, 1280, 3)
    assert image.dtype == np.uint8


def test_movie_fields_run_on_the_registered_grid():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        GYRE_R4_SMAG_NAMELIST,
        build_gyre_zco_card,
    )

    geom = MOVIE.geometry()
    grid = build_gyre_zco_card(GYRE_R4_SMAG_NAMELIST).recipe.grid
    shape = geom["wet"].shape
    zeros = np.zeros(shape, dtype=np.float64)
    frame = MOVIE.DailyFrame(1, zeros, zeros, zeros, zeros)
    fields = MOVIE.movie_fields(frame, geom, grid)
    assert tuple(fields) == (
        "relative vorticity / f", "SSH (m)", "surface speed (m/s)", "SST (C)",
    )
    for value in fields.values():
        assert value.shape == shape
        assert np.all(np.isfinite(value[geom["wet"]]))
        assert np.all(np.isnan(value[~geom["wet"]]))
