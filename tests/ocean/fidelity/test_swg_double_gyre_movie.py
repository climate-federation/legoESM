"""Direct tiny-grid tests for the SWG_DG side-by-side movie renderer."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_PATH = (Path(__file__).resolve().parents[3]
         / "scripts/plot/swg_double_gyre_movie.py")
_SPEC = importlib.util.spec_from_file_location("swg_double_gyre_movie", _PATH)
movie = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(movie)


def test_tiny_derived_fields_and_frame_are_finite_on_wet_cells():
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_MADDISON_NAMELIST,
        build_swg_card,
    )

    climate = movie._load_climate()
    tiny = SWG_MADDISON_NAMELIST._replace(
        nn_SWG=1, nn_itend=2, nn_write=1, nn_stock=1)
    card = build_swg_card(tiny)
    shape = card.recipe.land_mask.shape
    jj, ii = np.indices(shape, dtype=np.float64)
    frame = climate.DailyFrame(
        1, 1.0e-3 * (ii - jj), 1.0e-4 * jj, 2.0e-4 * ii)
    relative, ssh, speed = movie.derive_fields(frame, card)
    wet = np.asarray(card.recipe.land_mask, dtype=bool)
    assert relative.shape == ssh.shape == speed.shape == shape
    assert np.all(np.isfinite(relative[wet]))
    assert np.all(speed[wet] >= 0.0)
    rgb = movie.render_frame(frame, frame, card, absolute_day=1)
    assert rgb.shape == (720, 1280, 3)
    assert rgb.dtype == np.dtype("uint8")
    limits = movie._shared_limits(
        (relative, ssh, speed), (2.0 * relative, 2.0 * ssh, 2.0 * speed))
    assert limits[0][1] == 2.0 * np.nanmax(np.abs(relative))
    assert limits[1][0] == -2.0 * np.nanmax(np.abs(ssh))
    assert limits[2] == (0.0, 2.0 * np.nanmax(speed))


def test_movie_starts_encoder_before_provider_can_start_jax(tmp_path, monkeypatch):
    events = []

    def write_frames(path, size, **_kwargs):
        assert size == movie.FRAME_SIZE
        Path(path).write_bytes(b"fake-mp4")

        def encoder():
            events.append("ffmpeg-started")
            while True:
                frame = yield
                events.append(("frame", frame.shape))

        return encoder()

    def provider(*_args, **_kwargs):
        assert events[0] == "ffmpeg-started"
        events.append("provider-validated")
        return SimpleNamespace(
            sampling="daily_mean", iterate=lambda: iter((object(),)))

    fake_climate = SimpleNamespace(
        nemo_ioipsl_provider=provider, legoesm_npz_provider=provider)
    monkeypatch.setitem(
        sys.modules, "imageio_ffmpeg",
        SimpleNamespace(write_frames=write_frames))
    monkeypatch.setattr(movie, "_load_climate", lambda: fake_climate)
    monkeypatch.setattr(
        movie, "render_frame",
        lambda *_args, **_kwargs: np.zeros((720, 1280, 3), dtype=np.uint8))

    output = tmp_path / "movie.mp4"
    movie.write_movie(
        tmp_path, tmp_path, output, start_day=1, days=1, fps=12)
    assert output.read_bytes() == b"fake-mp4"
    assert events[:3] == [
        "ffmpeg-started", "provider-validated", "provider-validated"]
    assert events[-1] == ("frame", (720, 1280, 3))
    assert not (tmp_path / ".movie.partial.mp4").exists()
