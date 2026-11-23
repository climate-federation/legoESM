#!/usr/bin/env python3
"""Render the coupled C4 ORCA2+SI3 daily NEMO/legoESM comparison movie."""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np

FRAME_SIZE = (1440, 1280)
_GRID_KEYS = ("gphit", "glamt", "area_t", "wet")
_FIELDS = ("sst", "ice_area", "ice_volume", "snow_volume")
_VALUE_KEYS = tuple(
    f"{model}_{field}" for model in ("nemo", "candidate") for field in _FIELDS
)
_REQUIRED_KEYS = {"day", *_GRID_KEYS, *_VALUE_KEYS}
_POLAR_EDGE_DEG = 45.0
_POINT_AREA_PT2 = 4.0

_ROWS = (
    ("sst_global", "Sea surface temperature (K)", "global", "RdYlBu_r"),
    ("ice_area_north", "Arctic ice concentration", "north", "Blues"),
    ("ice_thickness_north", "Arctic ice thickness (m)", "north", "viridis"),
    ("ice_area_south", "Antarctic ice concentration", "south", "Blues"),
    ("ice_thickness_south", "Antarctic ice thickness (m)", "south", "viridis"),
)


class MovieInputError(RuntimeError):
    """Raised when the daily-map evidence cannot support the movie."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MovieInputError(message)


def _month_files(maps_dir: Path) -> list[Path]:
    paths = sorted(maps_dir.glob("c4_daily_maps_month??.npz"))
    _require(bool(paths), f"no c4_daily_maps_month??.npz files in {maps_dir}")
    return paths


def _geometry(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        missing = _REQUIRED_KEYS.difference(archive.files)
        _require(not missing, f"{path.name}: missing keys {sorted(missing)}")
        geometry = {key: np.asarray(archive[key]) for key in _GRID_KEYS}
    shape = geometry["wet"].shape
    _require(len(shape) == 2, f"{path.name}: wet must be two-dimensional")
    for key in _GRID_KEYS:
        _require(
            geometry[key].shape == shape,
            f"{path.name}: {key} shape {geometry[key].shape} != {shape}",
        )
    geometry["wet"] = geometry["wet"].astype(bool, copy=False)
    wet = geometry["wet"]
    _require(bool(wet.any()), f"{path.name}: wet mask is empty")
    for key in ("gphit", "glamt", "area_t"):
        _require(
            bool(np.isfinite(geometry[key][wet]).all()),
            f"{path.name}: {key} is non-finite on a wet T point",
        )
    _require(
        bool((geometry["area_t"][wet] > 0.0).all()),
        f"{path.name}: area_t is not positive on every wet T point",
    )
    return geometry


def _iter_days(
    paths: list[Path], geometry: dict[str, np.ndarray]
) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    shape = geometry["wet"].shape
    previous_day = 0
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            missing = _REQUIRED_KEYS.difference(archive.files)
            _require(not missing, f"{path.name}: missing keys {sorted(missing)}")
            for key in _GRID_KEYS:
                observed = np.asarray(archive[key])
                expected = geometry[key]
                if key == "wet":
                    observed = observed.astype(bool, copy=False)
                _require(
                    np.array_equal(observed, expected),
                    f"{path.name}: {key} differs from the first month",
                )
            raw_days = np.asarray(archive["day"])
            _require(raw_days.ndim == 1, f"{path.name}: day must be one-dimensional")
            _require(bool(np.isfinite(raw_days).all()), f"{path.name}: non-finite day")
            days = raw_days.astype(np.int64)
            _require(
                np.array_equal(raw_days, days),
                f"{path.name}: day contains a non-integer value",
            )
            values = {}
            for key in _VALUE_KEYS:
                value = np.asarray(archive[key])
                _require(
                    value.dtype == np.dtype(np.float32),
                    f"{path.name}: {key} dtype {value.dtype} != float32",
                )
                _require(
                    value.shape == (days.size, *shape),
                    f"{path.name}: {key} shape {value.shape} != {(days.size, *shape)}",
                )
                values[key] = value
            for index, day_value in enumerate(days):
                day = int(day_value)
                _require(day > previous_day, f"daily maps repeat or reverse at day {day}")
                previous_day = day
                fields = {key: value[index] for key, value in values.items()}
                wet = geometry["wet"]
                for key, value in fields.items():
                    _require(
                        bool(np.isfinite(value[wet]).all()),
                        f"{path.name}: {key} is non-finite on a wet T point at day {day}",
                    )
                yield day, fields


def _thickness(volume: np.ndarray, area: np.ndarray, wet: np.ndarray) -> np.ndarray:
    small = np.finfo(area.dtype).tiny
    valid = wet & (area > 0.0)
    output = np.full(area.shape, np.nan, dtype=np.float32)
    np.divide(volume, np.maximum(area, small), out=output, where=valid)
    return output


def _row_fields(
    fields: dict[str, np.ndarray], geometry: dict[str, np.ndarray]
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    wet = geometry["wet"]
    latitude = geometry["gphit"]
    north = wet & (latitude >= _POLAR_EDGE_DEG)
    south = wet & (latitude <= -_POLAR_EDGE_DEG)
    masks = {"global": wet, "north": north, "south": south}

    nemo_thickness = _thickness(fields["nemo_ice_volume"], fields["nemo_ice_area"], wet)
    candidate_thickness = _thickness(
        fields["candidate_ice_volume"], fields["candidate_ice_area"], wet
    )
    source = {
        "sst_global": (fields["nemo_sst"], fields["candidate_sst"], "global"),
        "ice_area_north": (
            fields["nemo_ice_area"], fields["candidate_ice_area"], "north"
        ),
        "ice_thickness_north": (nemo_thickness, candidate_thickness, "north"),
        "ice_area_south": (
            fields["nemo_ice_area"], fields["candidate_ice_area"], "south"
        ),
        "ice_thickness_south": (nemo_thickness, candidate_thickness, "south"),
    }
    output = {}
    for key, (nemo, candidate, region) in source.items():
        region_mask = masks[region]
        left = np.where(region_mask, nemo, np.nan)
        middle = np.where(region_mask, candidate, np.nan)
        difference = np.where(
            region_mask & np.isfinite(left) & np.isfinite(middle),
            middle - left,
            np.nan,
        )
        output[key] = (left, middle, difference, region_mask)
    return output


def _fixed_limits(
    paths: list[Path], geometry: dict[str, np.ndarray]
) -> tuple[dict[str, tuple[tuple[float, float], tuple[float, float]]], int]:
    extrema = {
        key: {"low": np.inf, "high": -np.inf, "difference": 0.0}
        for key, *_rest in _ROWS
    }
    count = 0
    for _day, fields in _iter_days(paths, geometry):
        count += 1
        rows = _row_fields(fields, geometry)
        for key, (nemo, candidate, difference, _mask) in rows.items():
            absolute = np.concatenate(
                (nemo[np.isfinite(nemo)], candidate[np.isfinite(candidate)])
            )
            delta = difference[np.isfinite(difference)]
            _require(bool(absolute.size), f"{key}: no finite values in the year")
            _require(bool(delta.size), f"{key}: no finite differences in the year")
            extrema[key]["low"] = min(extrema[key]["low"], float(absolute.min()))
            extrema[key]["high"] = max(extrema[key]["high"], float(absolute.max()))
            extrema[key]["difference"] = max(
                extrema[key]["difference"], float(np.max(np.abs(delta)))
            )
    _require(count > 0, "daily-map files contain no frames")
    limits = {}
    for key, *_rest in _ROWS:
        row = extrema[key]
        if key.startswith("ice_area"):
            absolute_limits = (0.0, 1.0)
        elif key.startswith("ice_thickness"):
            absolute_limits = (0.0, row["high"])
        else:
            absolute_limits = (row["low"], row["high"])
        bound = row["difference"]
        limits[key] = (absolute_limits, (-bound, bound))
    return limits, count


def _metadata_value(metadata: dict, names: tuple[str, ...]) -> str | None:
    for name in names:
        value = metadata.get(name)
        if value is not None:
            return str(value)
    provenance = metadata.get("provenance")
    if isinstance(provenance, dict):
        return _metadata_value(provenance, names)
    return None


def _caption(
    maps_dir: Path,
    month_paths: list[Path],
    *,
    math_fallback: str | None,
    commit_fallback: str | None,
) -> str:
    metadata: list[dict] = []
    candidates = [*month_paths, *sorted(maps_dir.glob("c4_month??_step*.npz"))]
    for path in candidates:
        with np.load(path, allow_pickle=False) as archive:
            if "__metadata_json__" not in archive.files:
                continue
            try:
                value = json.loads(str(archive["__metadata_json__"].item()))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise MovieInputError(f"{path.name}: invalid __metadata_json__") from exc
            _require(isinstance(value, dict), f"{path.name}: metadata is not an object")
            metadata.append(value)
    math_mode = next(
        (value for item in metadata if (value := _metadata_value(item, ("math", "math_mode")))),
        math_fallback,
    )
    commit = next(
        (
            value
            for item in metadata
            if (value := _metadata_value(
                item, ("commit", "git_commit", "git_sha", "repository_commit")
            ))
        ),
        commit_fallback,
    )
    _require(math_mode is not None, "no checkpoint math metadata; pass --math")
    _require(commit is not None, "no checkpoint commit metadata; pass --commit")
    return f"C4 coupled ORCA2+SI3 · math: {math_mode} · commit: {commit[:12]}"


def _day_label(day: int) -> str:
    if 1 <= day <= 365:
        date = dt.date(2001, 1, 1) + dt.timedelta(days=day - 1)
        return f"{calendar.month_abbr[date.month]} {date.day:02d} · day {day:03d}"
    return f"day {day}"


def _draw_native_points(
    axis,
    geometry: dict[str, np.ndarray],
    field: np.ndarray,
    mask: np.ndarray,
    *,
    cmap: str,
    limits: tuple[float, float],
):
    """Draw independent owned T points; no polygon crosses the seam or fold."""
    import cartopy.crs as ccrs

    longitude = geometry["glamt"]
    latitude = geometry["gphit"]
    axis.set_facecolor("#d9d9d9")
    axis.scatter(
        longitude[mask], latitude[mask], c="#f7fbff", marker="s",
        s=_POINT_AREA_PT2, linewidths=0.0, transform=ccrs.PlateCarree(),
        rasterized=True, zorder=1,
    )
    valid = mask & np.isfinite(field)
    artist = axis.scatter(
        longitude[valid], latitude[valid], c=field[valid], cmap=cmap,
        vmin=limits[0], vmax=limits[1], marker="s", s=_POINT_AREA_PT2,
        linewidths=0.0, transform=ccrs.PlateCarree(), rasterized=True, zorder=2,
    )
    return artist


def render_frame(
    day: int,
    fields: dict[str, np.ndarray],
    geometry: dict[str, np.ndarray],
    limits: dict[str, tuple[tuple[float, float], tuple[float, float]]],
    caption: str,
) -> np.ndarray:
    """Render one fixed-scale RGB movie frame."""
    import matplotlib

    matplotlib.use("Agg")
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    rows = _row_fields(fields, geometry)
    figure = plt.figure(figsize=(14.4, 12.8), dpi=100)
    grid = figure.add_gridspec(len(_ROWS), 3)
    axes = np.empty((len(_ROWS), 3), dtype=object)
    for row_index, (key, label, view, cmap) in enumerate(_ROWS):
        if view == "global":
            projection = ccrs.Robinson(central_longitude=0.0)
        elif view == "north":
            projection = ccrs.NorthPolarStereo(central_longitude=0.0)
        else:
            projection = ccrs.SouthPolarStereo(central_longitude=0.0)
        artists = []
        nemo, candidate, difference, region_mask = rows[key]
        for column, field in enumerate((nemo, candidate, difference)):
            axis = figure.add_subplot(grid[row_index, column], projection=projection)
            axes[row_index, column] = axis
            if view == "global":
                axis.set_global()
            elif view == "north":
                axis.set_extent(
                    [-180.0, 180.0, _POLAR_EDGE_DEG, 90.0], crs=ccrs.PlateCarree()
                )
            else:
                axis.set_extent(
                    [-180.0, 180.0, -90.0, -_POLAR_EDGE_DEG],
                    crs=ccrs.PlateCarree(),
                )
            scale = limits[key][1 if column == 2 else 0]
            artists.append(
                _draw_native_points(
                    axis, geometry, field, region_mask,
                    cmap="RdBu_r" if column == 2 else cmap, limits=scale,
                )
            )
            axis.set_title(("NEMO", "legoESM", "legoESM minus NEMO")[column], fontsize=9)
        axes[row_index, 0].text(
            -0.08, 0.5, label, rotation=90, va="center", ha="center",
            transform=axes[row_index, 0].transAxes, fontsize=9,
        )
        figure.colorbar(
            artists[1], ax=[axes[row_index, 0], axes[row_index, 1]],
            orientation="horizontal", fraction=0.045, pad=0.025, aspect=35,
        )
        figure.colorbar(
            artists[2], ax=axes[row_index, 2], orientation="horizontal",
            fraction=0.045, pad=0.025, aspect=18,
        )
    figure.suptitle(f"Daily NEMO and legoESM fields · {_day_label(day)}", fontsize=14)
    figure.text(0.5, 0.008, caption, ha="center", va="bottom", fontsize=8)
    figure.subplots_adjust(
        left=0.055, right=0.99, bottom=0.035, top=0.95, hspace=0.25, wspace=0.04
    )
    figure.canvas.draw()
    rgb = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    plt.close(figure)
    _require(
        rgb.shape[:2][::-1] == FRAME_SIZE,
        f"renderer returned {rgb.shape[:2][::-1]}, expected {FRAME_SIZE}",
    )
    return rgb


def write_movie(
    maps_dir: Path,
    output: Path,
    *,
    fps: int = 12,
    days: int | None = None,
    math_mode: str | None = None,
    commit: str | None = None,
) -> int:
    """Scan annual limits, then stream the requested daily frames to MP4."""
    import imageio_ffmpeg

    maps_dir = maps_dir.resolve()
    output = output.resolve()
    _require(maps_dir.is_dir(), f"maps directory does not exist: {maps_dir}")
    _require(output.suffix.lower() == ".mp4", "movie output must end in .mp4")
    _require(not output.exists(), f"refusing to overwrite {output}")
    _require(type(fps) is int and fps > 0, "fps must be a positive integer")
    _require(days is None or (type(days) is int and days > 0), "days must be positive")
    paths = _month_files(maps_dir)
    geometry = _geometry(paths[0])
    limits, available = _fixed_limits(paths, geometry)
    requested = available if days is None else days
    _require(requested <= available, f"requested {requested} days, only {available} available")
    run_caption = _caption(
        maps_dir, paths, math_fallback=math_mode, commit_fallback=commit
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial.mp4")
    _require(not temporary.exists(), f"stale partial movie exists: {temporary}")
    writer = imageio_ffmpeg.write_frames(
        temporary, FRAME_SIZE, fps=fps, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", quality=7,
    )
    writer.send(None)
    count = 0
    try:
        for day, fields in _iter_days(paths, geometry):
            if count == requested:
                break
            writer.send(render_frame(day, fields, geometry, limits, run_caption))
            count += 1
        _require(count == requested, f"rendered {count} frames, expected {requested}")
    except BaseException:
        writer.close()
        if temporary.exists():
            temporary.unlink()
        raise
    writer.close()
    temporary.replace(output)
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--days", type=int, help="render only the first N available days")
    parser.add_argument("--math", help="caption fallback when checkpoints lack math metadata")
    parser.add_argument("--commit", help="caption fallback when checkpoints lack commit metadata")
    args = parser.parse_args(argv)
    try:
        count = write_movie(
            args.maps_dir, args.output, fps=args.fps, days=args.days,
            math_mode=args.math, commit=args.commit,
        )
    except (MovieInputError, OSError, ValueError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(f"C4_DAILY_MOVIE COMPLETE frames={count} output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
