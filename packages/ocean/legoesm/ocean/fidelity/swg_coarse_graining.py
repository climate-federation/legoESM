"""Mask-normalized Gaussian coarse-graining for the SWG native C-grid."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

FILTER_TRUNCATE = 6.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _weighted_mean(field: np.ndarray, mask: np.ndarray,
                   area: np.ndarray) -> float:
    weights = np.where(mask, area, 0.0)
    return float(np.sum(field * weights, dtype=np.float64)
                 / np.sum(weights, dtype=np.float64))


def _restore_mean(field: np.ndarray, mask: np.ndarray, area: np.ndarray,
                  target_mean: float) -> np.ndarray:
    out = np.where(mask, field, 0.0)
    out[mask] += target_mean - _weighted_mean(out, mask, area)
    # Put the final reduction-roundoff residual in one wet cell.  This makes
    # basin-integral preservation explicit instead of relying on summation
    # order after adding a uniform correction.
    weights = np.where(mask, area, 0.0)
    target_sum = target_mean * np.sum(weights, dtype=np.float64)
    residual = target_sum - np.sum(out * weights, dtype=np.float64)
    at = tuple(np.argwhere(mask)[0])
    out[at] += residual / area[at]
    return out


def masked_gaussian_filter(field: np.ndarray, mask: np.ndarray, *,
                           sigma_cells: float, area: np.ndarray | None = None,
                           truncate: float = FILTER_TRUNCATE) -> np.ndarray:
    """Filter a wet field, normalize at walls, and preserve its wet mean."""
    field = np.asarray(field, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if field.ndim != 2 or field.shape != mask.shape:
        raise ValueError("field and mask must be equal-shape 2-D arrays")
    if not np.all(np.isfinite(field)) or not np.any(mask):
        raise ValueError("field must be finite and mask non-empty")
    if not np.isfinite(sigma_cells) or sigma_cells <= 0.0:
        raise ValueError("sigma_cells must be positive finite")
    if area is None:
        area = np.ones(field.shape, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    if area.shape != field.shape or not np.all(np.isfinite(area)) \
            or np.any(area <= 0.0):
        raise ValueError("area must be positive finite with field shape")
    first = field[mask][0]
    if np.all(field[mask] == first):
        return np.where(mask, first, 0.0)
    boundary_mode = "wrap" if np.all(mask) else "constant"
    numerator = gaussian_filter(
        np.where(mask, field, 0.0), sigma_cells, mode=boundary_mode, cval=0.0,
        truncate=truncate)
    denominator = gaussian_filter(
        mask.astype(np.float64), sigma_cells, mode=boundary_mode, cval=0.0,
        truncate=truncate)
    filtered = np.zeros_like(field)
    filtered[mask] = numerator[mask] / denominator[mask]
    return _restore_mean(
        filtered, mask, area, _weighted_mean(field, mask, area))


def overlap_weights(source_centres: np.ndarray, target_centres: np.ndarray,
                    source_spacing_m: float,
                    target_spacing_m: float) -> np.ndarray:
    """Return exact one-dimensional cell-overlap lengths."""
    source = np.asarray(source_centres, dtype=np.float64)
    target = np.asarray(target_centres, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("overlap coordinates must be one-dimensional")
    if source.size < 2 or target.size < 2:
        raise ValueError("overlap coordinates need at least two points")
    if not np.all(np.diff(source) > 0.0) or not np.all(np.diff(target) > 0.0):
        raise ValueError("overlap coordinates must increase strictly")
    source_lo = source - 0.5 * source_spacing_m
    source_hi = source + 0.5 * source_spacing_m
    target_lo = target - 0.5 * target_spacing_m
    target_hi = target + 0.5 * target_spacing_m
    return np.maximum(
        0.0,
        np.minimum(target_hi[:, None], source_hi[None, :])
        - np.maximum(target_lo[:, None], source_lo[None, :]),
    )


def area_average(field: np.ndarray, source_mask: np.ndarray,
                 target_mask: np.ndarray, *, source_x_m: np.ndarray,
                 source_y_m: np.ndarray, target_x_m: np.ndarray,
                 target_y_m: np.ndarray, source_spacing_m: float,
                 target_spacing_m: float) -> np.ndarray:
    """Area-average a native stagger from one uniform C-grid to another."""
    field = np.asarray(field, dtype=np.float64)
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    if field.shape != source_mask.shape:
        raise ValueError("source field/mask shapes differ")
    if np.asarray(source_x_m).shape != field.shape \
            or np.asarray(source_y_m).shape != field.shape:
        raise ValueError("source coordinate shapes differ")
    if np.asarray(target_x_m).shape != target_mask.shape \
            or np.asarray(target_y_m).shape != target_mask.shape:
        raise ValueError("target coordinate shapes differ")
    wx = overlap_weights(source_x_m[0], target_x_m[0],
                         source_spacing_m, target_spacing_m)
    wy = overlap_weights(source_y_m[:, 0], target_y_m[:, 0],
                         source_spacing_m, target_spacing_m)
    weights = wy @ source_mask.astype(np.float64) @ wx.T
    numerator = wy @ np.where(source_mask, field, 0.0) @ wx.T
    if np.any(target_mask & (weights <= 0.0)):
        raise ValueError("a wet target cell has no wet source overlap")
    first = field[source_mask][0]
    if np.all(field[source_mask] == first):
        return np.where(target_mask, first, 0.0)
    out = np.zeros(target_mask.shape, dtype=np.float64)
    out[target_mask] = numerator[target_mask] / weights[target_mask]
    source_area = np.full(field.shape, source_spacing_m ** 2)
    target_area = np.full(target_mask.shape, target_spacing_m ** 2)
    return _restore_mean(
        out, target_mask, target_area,
        _weighted_mean(field, source_mask, source_area))


def cgrid_geometry(namelist) -> dict[str, dict[str, np.ndarray] | float]:
    """Instantiate native T/U/V masks and coordinates from a resolved deck."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_swg_card,
        swg_horizontal_coordinates,
    )

    card = build_swg_card(namelist)
    coordinates = swg_horizontal_coordinates(namelist)
    u_full, v_full = compute_face_masks(card.recipe.land_mask, card.recipe.grid)
    masks = {
        "T": np.asarray(card.recipe.land_mask, dtype=bool),
        "U": np.asarray(u_full[:, 1:], dtype=bool),
        "V": np.asarray(v_full[1:, :], dtype=bool),
    }
    return {
        "spacing_m": float(coordinates["spacing_m"]),
        "mask": masks,
        "x": {"T": np.asarray(coordinates["glamt"]),
              "U": np.asarray(coordinates["glamu"]),
              "V": np.asarray(coordinates["glamv"])},
        "y": {"T": np.asarray(coordinates["gphit"]),
              "U": np.asarray(coordinates["gphiu"]),
              "V": np.asarray(coordinates["gphiv"])},
    }


def coarse_grain_cgrid(fields: dict[str, np.ndarray], source: dict,
                       target: dict, *, gaussian_length_m: float) -> dict[str, np.ndarray]:
    """Filter and area-average SSH/U/V on their native C-grid staggers."""
    expected = {"T", "U", "V"}
    if set(fields) != expected:
        raise ValueError(f"fields must be exactly {sorted(expected)}")
    sigma_cells = gaussian_length_m / float(source["spacing_m"])
    output = {}
    for kind in ("T", "U", "V"):
        filtered = masked_gaussian_filter(
            fields[kind], source["mask"][kind], sigma_cells=sigma_cells)
        output[kind] = area_average(
            filtered, source["mask"][kind], target["mask"][kind],
            source_x_m=source["x"][kind], source_y_m=source["y"][kind],
            target_x_m=target["x"][kind], target_y_m=target["y"][kind],
            source_spacing_m=float(source["spacing_m"]),
            target_spacing_m=float(target["spacing_m"]))
    return output


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def filter_truth_directory(input_root: Path, output_root: Path, *,
                           commit: str) -> dict:
    """Write coarse daily snapshots and means from a complete fine v2 run."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        SWG_MADDISON_NAMELIST,
    )

    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ValueError("commit must be a forty-character lowercase Git SHA")
    manifest_path = input_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" \
            or manifest.get("sampling") != "daily_mean" \
            or manifest.get("completed_days") != manifest.get("days"):
        raise ValueError("input must be a complete daily-mean fine run")
    expected_source = json.loads(json.dumps(SWG_MADDISON_NAMELIST._asdict()))
    source_namelist = manifest.get("namelist", {})
    for field in expected_source.keys() - {"nn_itend", "nn_stock"}:
        if source_namelist.get(field) != expected_source[field]:
            raise ValueError(f"input fine deck differs at {field}")
    source = cgrid_geometry(SWG_MADDISON_NAMELIST)
    target = cgrid_geometry(SWG_DG50_NAMELIST)
    gaussian_length_m = float(target["spacing_m"])
    output_root.mkdir(parents=True)
    roots = {name: output_root / name for name in ("snapshots", "daily_means")}
    for root in roots.values():
        root.mkdir()
    source_hashes = manifest.get("daily_sha256", {})
    output_hashes = {name: {} for name in roots}
    days = int(manifest["days"])
    base = {
        "schema": "legoesm-swg-filtered-truth-v1",
        "status": "pending",
        "completed_days": 0,
        "days": days,
        "legoesm_git_sha": commit,
        "source_root": str(input_root.resolve()),
        "source_manifest_sha256": sha256(manifest_path),
        "source_daily_sha256": source_hashes,
        "source_namelist": source_namelist,
        "target_namelist": SWG_DG50_NAMELIST._asdict(),
        "filter": {
            "name": "mask-normalized Gaussian then overlap-area average",
            "gaussian_length_m": gaussian_length_m,
            "sigma_fine_cells": gaussian_length_m / float(source["spacing_m"]),
            "truncate_sigma": FILTER_TRUNCATE,
            "mean_conservation": "float64 wet-stagger integral correction",
            "staggering": {"ssh_m": "T", "u_m_s": "U", "v_m_s": "V"},
        },
    }
    for day in range(1, days + 1):
        name = f"day{day:04d}.npz"
        path = input_root / name
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if source_hashes.get(name) != digest:
            raise ValueError(f"input digest mismatch: {name}")
        with np.load(io.BytesIO(payload), allow_pickle=False) as saved:
            snapshot = coarse_grain_cgrid(
                {"T": saved["ssh_m"], "U": saved["u_m_s"],
                 "V": saved["v_m_s"]}, source, target,
                gaussian_length_m=gaussian_length_m)
            daily_mean = coarse_grain_cgrid(
                {"T": saved["ssh_daily_mean_m"],
                 "U": saved["u_daily_mean_m_s"],
                 "V": saved["v_daily_mean_m_s"]}, source, target,
                gaussian_length_m=gaussian_length_m)
            metadata = {
                "schema": np.str_("legoesm-swg-filtered-truth-v1"),
                "source_day": np.int64(saved["day"]),
                "source_step": np.int64(saved["step"]),
                "source_dt_s": np.float64(saved["dt_s"]),
                "day": np.int64(day),
            }
        records = {
            "snapshots": metadata | {"ssh_m": snapshot["T"],
                                      "u_m_s": snapshot["U"],
                                      "v_m_s": snapshot["V"]},
            "daily_means": metadata | {
                "daily_mean_sampling": np.str_(
                    "filtered fine arithmetic mean of every post-step state"),
                "daily_mean_sample_count": np.int64(1_728),
                "ssh_daily_mean_m": daily_mean["T"],
                "u_daily_mean_m_s": daily_mean["U"],
                "v_daily_mean_m_s": daily_mean["V"]},
        }
        for sampling, arrays in records.items():
            temporary = roots[sampling] / f".{name}.tmp"
            with temporary.open("wb") as stream:
                np.savez(stream, **arrays)
            temporary.replace(roots[sampling] / name)
            output_hashes[sampling][name] = sha256(roots[sampling] / name)
        for sampling, root in roots.items():
            _write_json_atomic(root / "manifest.pending.json", base | {
                "sampling": ("daily_snapshot" if sampling == "snapshots"
                             else "daily_mean"),
                "completed_days": day,
                "output_daily_sha256": output_hashes[sampling],
            })
    manifests = {}
    for sampling, root in roots.items():
        pending = root / "manifest.pending.json"
        complete = json.loads(pending.read_text()) | {"status": "complete"}
        _write_json_atomic(pending, complete)
        pending.replace(root / "manifest.json")
        manifests[sampling] = complete
    return manifests


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        from legoesm.ocean.fidelity.provenance import git_sha

        manifests = filter_truth_directory(
            args.input, args.output, commit=git_sha())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSE: {exc}")
        return 2
    print(f"COMPLETED days={manifests['snapshots']['days']} output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
