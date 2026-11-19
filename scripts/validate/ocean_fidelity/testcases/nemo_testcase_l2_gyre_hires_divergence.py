#!/usr/bin/env python3
"""Measure GYRE twin divergence and yearly eddy fractions on one protocol.

The divergence definition is the operator's round-8 definition: RMS of the
difference divided by the RMS spatial anomaly of the certified vector-math
NEMO field, plus pattern correlation, on daily-mean SSH, SST, native surface
U/V, and depth-mean U/V.  Round 11 also attributes the day-30..180 excess
over the scalar/vector NEMO twin to wet-span longitude bands and compares
exponential with algebraic growth in log-error space.
The eddy fraction is the climate scorer's 31-day nearest-boundary high-pass
EKE divided by surface KE, evaluated from the full five-year series.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from nemo_testcase_l2_gyre_hires_climate import (  # noqa: E402
    FILTER_DAYS,
    GLOBAL_SHAPE,
    DailyProvider,
    assemble_domain_tiles,
    geometry,
    legoesm_npz_provider,
    nemo_ioipsl_provider,
    sha256,
)
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

TOTAL_DAYS = 1825
CURVE_DAYS = 365
CHECKPOINT_DAYS = (1, 10, 30, 60, 90, 120, 180, 240, 300, 365)
# Independent operator probe, printed to two digits after the decimal in
# scientific notation.  String equality preserves that probe's real precision
# instead of inventing a tighter numeric tolerance after the fact.
OPERATOR_TWIN_SSH_BENCHMARK = {
    30: "1.04e-09",
    60: "1.17e-09",
    90: "1.19e-08",
    120: "1.21e-06",
    180: "2.95e-04",
    365: "3.89e-03",
}


def normalized_rms_and_correlation(
    reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray,
) -> dict[str, float]:
    """Score two same-stagger daily fields on a common active mask."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(reference.shape == candidate.shape == mask.shape,
            "divergence field/mask shapes differ")
    require(np.any(mask), "divergence mask is empty")
    left, right = reference[mask], candidate[mask]
    require(np.all(np.isfinite(left)) and np.all(np.isfinite(right)),
            "non-finite active divergence field")
    left_anomaly = left - np.mean(left, dtype=np.float64)
    right_anomaly = right - np.mean(right, dtype=np.float64)
    reference_rms = float(np.sqrt(np.mean(
        left_anomaly * left_anomaly, dtype=np.float64)))
    require(reference_rms > 0.0, "reference spatial anomaly is zero")
    denominator = np.sqrt(
        np.sum(left_anomaly * left_anomaly, dtype=np.float64)
        * np.sum(right_anomaly * right_anomaly, dtype=np.float64)
    )
    require(denominator > 0.0, "pattern-correlation denominator is zero")
    absolute_rms = float(np.sqrt(np.mean(
        (left - right) ** 2, dtype=np.float64)))
    return {
        "absolute_rms_difference": absolute_rms,
        "reference_spatial_anomaly_rms": reference_rms,
        "normalized_rms_difference": absolute_rms / reference_rms,
        "pattern_correlation": float(np.sum(
            left_anomaly * right_anomaly, dtype=np.float64) / denominator),
    }


def divergence_curve(
    reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply the registered pointwise score to every matched day."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    require(reference.shape == candidate.shape and reference.ndim == 3,
            "divergence time series shapes differ")
    rows = [
        normalized_rms_and_correlation(left, right, mask)
        for left, right in zip(reference, candidate, strict=True)
    ]
    return {
        "absolute_rms_difference": np.asarray([
            row["absolute_rms_difference"] for row in rows]),
        "reference_spatial_anomaly_rms": np.asarray([
            row["reference_spatial_anomaly_rms"] for row in rows]),
        "normalized_rms_difference": np.asarray([
            row["normalized_rms_difference"] for row in rows]),
        "pattern_correlation": np.asarray([
            row["pattern_correlation"] for row in rows]),
    }


def yearly_eddy_statistics(
    u: np.ndarray, v: np.ndarray, wet: np.ndarray,
) -> list[dict[str, float | int]]:
    """Return full-series yearly KE/EKE using the registered 31-day filter."""
    from scipy.ndimage import uniform_filter1d

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    wet = np.asarray(wet, dtype=bool)
    require(u.shape == v.shape and u.ndim == 3 and u.shape[1:] == wet.shape,
            "eddy series/mask shapes differ")
    require(u.shape[0] % 365 == 0 and np.any(wet),
            "eddy series is not an integer number of non-empty years")
    require(np.all(np.isfinite(u[:, wet])) and np.all(np.isfinite(v[:, wet])),
            "non-finite active eddy series")
    u_hp = u - uniform_filter1d(u, FILTER_DAYS, axis=0, mode="nearest")
    v_hp = v - uniform_filter1d(v, FILTER_DAYS, axis=0, mode="nearest")
    ke = 0.5 * (u * u + v * v)
    eke = 0.5 * (u_hp * u_hp + v_hp * v_hp)
    result = []
    for year in range(u.shape[0] // 365):
        window = np.s_[365 * year:365 * (year + 1)]
        mean_ke = float(np.mean(ke[window][:, wet], dtype=np.float64))
        mean_eke = float(np.mean(eke[window][:, wet], dtype=np.float64))
        require(mean_ke > 0.0, "yearly surface KE is zero")
        result.append({
            "year": year + 1,
            "surface_ke_m2_s2": mean_ke,
            "highpass_eke_m2_s2": mean_eke,
            "eddy_fraction": mean_eke / mean_ke,
        })
    return result


def nemo_surface_masks(root: Path) -> dict[str, np.ndarray]:
    """Stitch NEMO surface T/U/V masks from their own DOMAIN metadata."""
    import netCDF4

    paths = sorted(root.glob("mesh_mask_????.nc"))
    require(len(paths) == 8, "NEMO reference requires eight mesh-mask tiles")
    tiles = {"wet": [], "u": [], "v": []}
    for path in paths:
        with netCDF4.Dataset(path) as dataset:
            require(np.array_equal(dataset.DOMAIN_size_global, [122, 82]),
                    "mesh-mask DOMAIN_size_global is not 122x82")
            first = np.asarray(dataset.DOMAIN_position_first)
            tiles["wet"].append((first, np.asarray(dataset["tmask"][0, 0])))
            tiles["u"].append((first, np.asarray(dataset["umask"][0, 0])))
            tiles["v"].append((first, np.asarray(dataset["vmask"][0, 0])))
    return {
        name: assemble_domain_tiles(value) > 0.5
        for name, value in tiles.items()
    }


def load_surface_series(
    provider: DailyProvider,
    *,
    total_days: int = TOTAL_DAYS,
    curve_days: int = CURVE_DAYS,
) -> dict:
    """Stream one provider once; retain only fields used by this instrument."""
    require(0 < curve_days <= total_days, "invalid divergence window")
    ssh = np.empty((curve_days, *GLOBAL_SHAPE), dtype=np.float64)
    sst = np.empty_like(ssh)
    u = np.empty((total_days, *GLOBAL_SHAPE), dtype=np.float64)
    v = np.empty_like(u)
    count = 0
    for count, frame in enumerate(provider.iterate(), start=1):
        require(count <= total_days and frame.day == count,
                f"{provider.name}: daily sequence differs")
        if count <= curve_days:
            ssh[count - 1] = frame.ssh
            sst[count - 1] = frame.sst
        u[count - 1] = frame.u
        v[count - 1] = frame.v
    require(count == total_days, f"{provider.name}: expected {total_days} days")
    return {
        "ssh": ssh, "sst": sst, "u": u, "v": v,
        "provenance": provider.provenance,
    }


def wet_span_bins(mask: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Assign active cells to equal fractional-distance bins within each row."""
    mask = np.asarray(mask, dtype=bool)
    require(mask.ndim == 2 and n_bins > 1, "invalid wet-span bin request")
    result = np.full(mask.shape, -1, dtype=np.int8)
    for j in range(mask.shape[0]):
        active = np.flatnonzero(mask[j])
        if active.size:
            labels = np.minimum(
                (np.arange(active.size, dtype=np.int64) * n_bins)
                // active.size,
                n_bins - 1,
            )
            result[j, active] = labels.astype(np.int8)
    return result


def regional_excess(
    reference: np.ndarray,
    candidate: np.ndarray,
    twin: np.ndarray,
    mask: np.ndarray,
    *,
    start_day: int = 30,
    stop_day: int = 180,
) -> list[dict]:
    """Signed candidate-minus-twin SSE in five per-row wet-span bands."""
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    twin = np.asarray(twin, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(reference.shape == candidate.shape == twin.shape,
            "regional series shapes differ")
    require(reference.ndim == 3 and reference.shape[1:] == mask.shape,
            "regional mask shape differs")
    require(1 <= start_day <= stop_day <= reference.shape[0],
            "invalid regional day window")
    require(np.all(np.isfinite(reference[:, mask]))
            and np.all(np.isfinite(candidate[:, mask]))
            and np.all(np.isfinite(twin[:, mask])),
            "non-finite active regional series")
    window = slice(start_day - 1, stop_day)
    candidate_error = (candidate[window] - reference[window]) ** 2
    twin_error = (twin[window] - reference[window]) ** 2
    bins = wet_span_bins(mask)
    names = ("western_boundary", "western_interior", "central_interior",
             "eastern_interior", "eastern_boundary")
    result = []
    for index, name in enumerate(names):
        active = bins == index
        require(np.any(active), f"empty regional bin {name}")
        candidate_sse = float(np.sum(
            candidate_error[:, active], dtype=np.float64))
        twin_sse = float(np.sum(twin_error[:, active], dtype=np.float64))
        result.append({
            "region": name,
            "active_cells": int(np.count_nonzero(active)),
            "candidate_sse": candidate_sse,
            "twin_sse": twin_sse,
            "signed_excess_sse": candidate_sse - twin_sse,
        })
    total_excess = float(np.sum(
        [row["signed_excess_sse"] for row in result], dtype=np.float64))
    require(np.isfinite(total_excess) and total_excess > 0.0,
            "regional candidate has no positive signed SSE excess")
    for row in result:
        row["signed_excess_share"] = row["signed_excess_sse"] / total_excess
    return result


def first_factor_excess(
    candidate: dict[str, dict[str, np.ndarray]],
    twin: dict[str, dict[str, np.ndarray]],
    *, factor: float = 4.0,
    start_day: int = 30,
    stop_day: int = 180,
) -> dict:
    """Name the earliest field/day whose normalised error clears a factor."""
    require(set(candidate) == set(twin), "curve field inventories differ")
    hits = []
    for field in sorted(candidate):
        left = candidate[field]["normalized_rms_difference"]
        floor = twin[field]["normalized_rms_difference"]
        require(left.shape == floor.shape, f"{field}: curve shapes differ")
        days = np.arange(start_day, stop_day + 1)
        selected = left[start_day - 1:stop_day] > factor * floor[
            start_day - 1:stop_day]
        if np.any(selected):
            day = int(days[np.flatnonzero(selected)[0]])
            hits.append((day, field, float(left[day - 1]), float(floor[day - 1])))
    require(bool(hits), "candidate never clears registered twin factor")
    first_day = min(row[0] for row in hits)
    first = [row for row in hits if row[0] == first_day]
    return {
        "factor": factor,
        "day": first_day,
        "fields": [row[1] for row in first],
        "rows": [
            {"field": field, "candidate": value, "twin": floor,
             "ratio": value / floor if floor > 0.0 else None,
             "twin_zero": floor == 0.0}
            for _, field, value, floor in first
        ],
    }


def fit_growth_laws(days: np.ndarray, error: np.ndarray) -> dict:
    """Compare exponential and algebraic fits using log-space RMS residual."""
    days = np.asarray(days, dtype=np.float64)
    error = np.asarray(error, dtype=np.float64)
    require(days.ndim == error.ndim == 1 and days.shape == error.shape,
            "growth arrays differ")
    require(days.size >= 3 and np.all(days > 0.0)
            and np.all(np.isfinite(error)) and np.all(error > 0.0),
            "growth fit requires finite positive samples")
    target = np.log(error)

    def fit(x):
        design = np.column_stack((np.ones_like(x), x))
        coefficient, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficient
        return coefficient, float(np.sqrt(np.mean(residual * residual)))

    exponential, exp_rms = fit(days)
    algebraic, alg_rms = fit(np.log(days))
    require(exponential[1] > 0.0 and algebraic[1] > 0.0,
            "registered growth fits require positive fitted growth")
    return {
        "classification": "exponential" if exp_rms < alg_rms else "algebraic",
        "exponential": {
            "log_intercept": float(exponential[0]),
            "per_day_rate": float(exponential[1]),
            "log_rms_residual": exp_rms,
        },
        "algebraic": {
            "log_intercept": float(algebraic[0]),
            "power": float(algebraic[1]),
            "log_rms_residual": alg_rms,
        },
    }


def _assemble_time_tiles(
    tiles: list[tuple[np.ndarray, np.ndarray]], shape=GLOBAL_SHAPE,
) -> np.ndarray:
    """Assemble time-by-y-by-x tiles using NEMO's DOMAIN positions."""
    require(bool(tiles), "no time tiles")
    days = np.asarray(tiles[0][1]).shape[0]
    assembled = np.empty((days, *shape), dtype=np.float64)
    filled = np.zeros(shape, dtype=bool)
    for first, raw in tiles:
        first = np.asarray(first, dtype=np.int64)
        values = np.asarray(raw, dtype=np.float64)
        require(first.shape == (2,) and values.ndim == 3
                and values.shape[0] == days, "invalid time tile")
        i0, j0 = int(first[0] - 1), int(first[1] - 1)
        target = np.s_[j0:j0 + values.shape[1], i0:i0 + values.shape[2]]
        require(assembled[:, target[0], target[1]].shape == values.shape,
                "time tile exceeds global shape")
        require(not np.any(filled[target]), "overlapping time tiles")
        require(np.all(np.isfinite(values)), "non-finite time tile")
        assembled[:, target[0], target[1]] = values
        filled[target] = True
    require(np.all(filled), "time assembly missed cells")
    return assembled


def thickness_weighted_depth_mean(
    velocity: np.ndarray, weights: np.ndarray,
) -> np.ndarray:
    """Form daily z-coordinate column means with validated layer weights."""
    velocity = np.asarray(velocity, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    require(velocity.ndim == 4 and weights.ndim == 3
            and velocity.shape[1:] == weights.shape,
            "depth-mean velocity/weight shapes differ")
    require(np.all(np.isfinite(velocity)) and np.all(np.isfinite(weights)),
            "non-finite depth-mean input")
    require(np.all(weights >= 0.0), "negative depth-mean layer weight")
    column = np.sum(weights, axis=0, dtype=np.float64)
    require(np.any(column > 0.0), "depth-mean weights have no active column")
    numerator = np.sum(
        velocity * weights[None, ...], axis=1, dtype=np.float64)
    return np.divide(
        numerator, column[None, ...], out=np.zeros_like(numerator),
        where=column[None, ...] > 0.0,
    )


def load_nemo_depth_mean(root: Path, days: int) -> dict[str, np.ndarray]:
    """Read z-coordinate thickness-weighted daily U/V column means."""
    import netCDF4

    require(days > 0, "depth-mean window must be positive")
    meshes = sorted(root.glob("mesh_mask_????.nc"))
    paths = {
        "u": sorted(root.glob("*grid_U_????.nc")),
        "v": sorted(root.glob("*grid_V_????.nc")),
    }
    require(len(meshes) == 8 and all(len(value) == 8 for value in paths.values()),
            "depth-mean reader requires eight NEMO mesh/U/V tiles")
    result = {}
    with ExitStack() as stack:
        mesh_sets = [stack.enter_context(netCDF4.Dataset(path)) for path in meshes]
        for field, variable, metric, mask_name in (
            ("depth_mean_u", "vozocrtx", "e3u_0", "umask"),
            ("depth_mean_v", "vomecrty", "e3v_0", "vmask"),
        ):
            dsets = [
                stack.enter_context(netCDF4.Dataset(path))
                for path in paths["u" if field.endswith("u") else "v"]
            ]
            tiles = []
            for mesh, dataset in zip(mesh_sets, dsets, strict=True):
                mesh.set_auto_mask(False)
                dataset.set_auto_mask(False)
                require(np.array_equal(mesh.DOMAIN_position_first,
                                       dataset.DOMAIN_position_first),
                        f"{field}: mesh/output tile positions differ")
                weights = (np.asarray(mesh[metric][0], dtype=np.float64)
                           * np.asarray(mesh[mask_name][0], dtype=np.float64))
                velocity = np.asarray(dataset[variable][:days], dtype=np.float64)
                require(velocity.shape[0] == days,
                        f"{field}: too few daily records")
                mean = thickness_weighted_depth_mean(velocity, weights)
                tiles.append((dataset.DOMAIN_position_first, mean))
            result[field] = _assemble_time_tiles(tiles)
    return result


def load_legoesm_depth_mean(provider: DailyProvider, days: int) -> dict[str, np.ndarray]:
    """Load the registered external-mode daily means stored by the driver."""
    registration = provider.provenance.get("registration", {})
    root = Path(registration.get("run_directory", ""))
    require(root.is_absolute() and days > 0,
            "missing registered legoESM depth-mean root")
    result = {
        "depth_mean_u": np.empty((days, *GLOBAL_SHAPE), dtype=np.float64),
        "depth_mean_v": np.empty((days, *GLOBAL_SHAPE), dtype=np.float64),
    }
    names = {
        "depth_mean_u": "barotropic_u_m_s_daily_mean",
        "depth_mean_v": "barotropic_v_m_s_daily_mean",
    }
    for day in range(1, days + 1):
        path = root / f"day{day:04d}.npz"
        with np.load(path, allow_pickle=False) as data:
            require(int(data["day"]) == day, f"{path.name}: day differs")
            for field, name in names.items():
                require(name in data, f"{path.name}: missing {name}")
                value = np.asarray(data[name], dtype=np.float64)
                require(value.shape == GLOBAL_SHAPE and np.all(np.isfinite(value)),
                        f"{path.name}: invalid {name}")
                result[field][day - 1] = value
    return result


def _checkpoint_rows(curves: dict[str, dict[str, dict]]) -> list[dict]:
    rows = []
    for day in CHECKPOINT_DAYS:
        row = {"day": day}
        for comparison, fields in curves.items():
            for field, values in fields.items():
                row[f"{comparison}.{field}.normalized_rms_difference"] = float(
                    values["normalized_rms_difference"][day - 1])
                row[f"{comparison}.{field}.pattern_correlation"] = float(
                    values["pattern_correlation"][day - 1])
        rows.append(row)
    return rows


def _validate_twin(curve: dict[str, dict[str, np.ndarray]]) -> dict:
    require("ssh" in curve and ("surface_u" in curve or "u" in curve),
            "NEMO twin curve is missing registered fields")
    measured = {
        day: f"{curve['ssh']['normalized_rms_difference'][day - 1]:.2e}"
        for day in OPERATOR_TWIN_SSH_BENCHMARK
    }
    require(measured == OPERATOR_TWIN_SSH_BENCHMARK,
            "NEMO twin curve does not reproduce the independent benchmark")
    return {
        "status": "PASS",
        "quantity": "SSH normalized RMS difference",
        "precision": "two digits after decimal in scientific notation",
        "expected": OPERATOR_TWIN_SSH_BENCHMARK,
        "measured": measured,
    }


def main(argv=None) -> int:
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-nemo-root", type=Path, required=True)
    parser.add_argument("--scalar-nemo-root", type=Path, required=True)
    parser.add_argument("--scalar-nemo-manifest", type=Path, required=True)
    parser.add_argument("--scalar-nemo-label", required=True)
    parser.add_argument("--baseline-legoesm-run-label", required=True)
    parser.add_argument("--candidate-legoesm-run-label", required=True)
    parser.add_argument(
        "--run-registry", type=Path,
        default=HERE / "gyre_hires_registered_runs.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        vector_provider = nemo_ioipsl_provider(
            args.vector_nemo_root, TOTAL_DAYS, start_day=1)
        scalar_provider = nemo_ioipsl_provider(
            args.scalar_nemo_root,
            TOTAL_DAYS,
            start_day=1,
            alternative_manifest=args.scalar_nemo_manifest,
            alternative_label=args.scalar_nemo_label,
        )
        baseline_provider = legoesm_npz_provider(
            args.baseline_legoesm_run_label,
            TOTAL_DAYS,
            start_day=1,
            registry_path=args.run_registry,
        )
        candidate_provider = legoesm_npz_provider(
            args.candidate_legoesm_run_label,
            TOTAL_DAYS,
            start_day=1,
            registry_path=args.run_registry,
        )

        masks = nemo_surface_masks(args.vector_nemo_root)
        card_geom = geometry()
        require(np.array_equal(masks["wet"], card_geom["wet"]),
                "NEMO and legoESM T masks differ")
        require(np.array_equal(masks["u"], card_geom["u_mask"]),
                "NEMO and legoESM U masks differ")
        require(np.array_equal(masks["v"], card_geom["v_mask"]),
                "NEMO and legoESM V masks differ")

        vector = load_surface_series(vector_provider)
        vector_years = yearly_eddy_statistics(
            vector["u"], vector["v"], masks["wet"])
        vector_curve = {
            "ssh": vector["ssh"],
            "sst": vector["sst"],
            "surface_u": np.array(vector["u"][:CURVE_DAYS], copy=True),
            "surface_v": np.array(vector["v"][:CURVE_DAYS], copy=True),
            **load_nemo_depth_mean(args.vector_nemo_root, CURVE_DAYS),
        }

        curves: dict[str, dict[str, dict]] = {}
        yearly = {"nemo_vector": vector_years}
        provenance = {"nemo_vector": vector["provenance"]}
        raw_curves: dict[str, dict[str, np.ndarray]] = {}
        del vector
        providers = (
            ("nemo_scalar", scalar_provider, args.scalar_nemo_root),
            ("legoesm_baseline", baseline_provider, None),
            ("legoesm_candidate", candidate_provider, None),
        )
        field_masks = {
            "ssh": masks["wet"], "sst": masks["wet"],
            "surface_u": masks["u"], "surface_v": masks["v"],
            "depth_mean_u": masks["u"], "depth_mean_v": masks["v"],
        }
        for label, provider, nemo_root in providers:
            series = load_surface_series(provider)
            yearly[label] = yearly_eddy_statistics(
                series["u"], series["v"], masks["wet"])
            fields = {
                "ssh": series["ssh"],
                "sst": series["sst"],
                "surface_u": np.array(series["u"][:CURVE_DAYS], copy=True),
                "surface_v": np.array(series["v"][:CURVE_DAYS], copy=True),
                **(load_nemo_depth_mean(nemo_root, CURVE_DAYS)
                   if nemo_root is not None else
                   load_legoesm_depth_mean(provider, CURVE_DAYS)),
            }
            curves[label] = {
                field: divergence_curve(vector_curve[field], fields[field], mask)
                for field, mask in field_masks.items()
            }
            provenance[label] = series["provenance"]
            if label in ("nemo_scalar", "legoesm_candidate"):
                raw_curves[label] = fields
            del series

        twin_validation = _validate_twin(curves["nemo_scalar"])
        field_controls = {}
        for field, mask in field_masks.items():
            reference = vector_curve[field][0]
            exact = normalized_rms_and_correlation(reference, reference, mask)
            planted = np.array(reference, copy=True)
            active_indices = np.argwhere(mask)
            plant_index = tuple(active_indices[
                int(np.argmax(np.abs(reference[mask])))])
            planted[plant_index] = np.nextafter(
                planted[plant_index], np.inf)
            plant = normalized_rms_and_correlation(reference, planted, mask)
            require(exact["normalized_rms_difference"] == 0.0
                    and plant["normalized_rms_difference"] > 0.0,
                    f"{field}: exact/one-ULP controls did not separate")
            field_controls[field] = {
                "exact": exact,
                "one_ulp": {
                    **plant,
                    "index": [int(value) for value in plant_index],
                    "status": "FIRED",
                },
            }

        first_excess = first_factor_excess(
            curves["legoesm_candidate"], curves["nemo_scalar"])
        regional = {
            field: regional_excess(
                vector_curve[field], raw_curves["legoesm_candidate"][field],
                raw_curves["nemo_scalar"][field], mask,
            )
            for field, mask in field_masks.items()
        }
        region_names = (
            "western_boundary", "western_interior", "central_interior",
            "eastern_interior", "eastern_boundary",
        )
        fit_days = np.arange(30, 181, dtype=np.float64)
        leading_regions = {}
        growth = {}
        for field in first_excess["fields"]:
            leading_region = max(
                regional[field], key=lambda row: row["signed_excess_sse"])
            leading_bin = region_names.index(leading_region["region"])
            leading_mask = wet_span_bins(field_masks[field]) == leading_bin
            leading_curve = divergence_curve(
                vector_curve[field], raw_curves["legoesm_candidate"][field],
                leading_mask)
            basin_growth = fit_growth_laws(
                fit_days, curves["legoesm_candidate"][field][
                    "absolute_rms_difference"][29:180])
            region_growth = fit_growth_laws(
                fit_days, leading_curve["absolute_rms_difference"][29:180])
            leading_regions[field] = leading_region
            growth[field] = {
                "classification": (
                    basin_growth["classification"]
                    if basin_growth["classification"]
                    == region_growth["classification"] else "mixed"
                ),
                "basin": basin_growth,
                "leading_region": region_growth,
            }

        arrays_path = args.output / "gyre_hires_divergence_arrays.npz"
        np.savez_compressed(arrays_path, **{
            f"{comparison}.{field}.{metric}": value
            for comparison, fields in curves.items()
            for field, metrics in fields.items()
            for metric, value in metrics.items()
        })
        report = {
            "schema": "gyre-r4-smag-divergence-envelope-v3",
            "instrument": {
                "worktree": worktree_stamp(),
                "argv": list(sys.argv if argv is None else [__file__, *argv]),
            },
            "definition": {
                "divergence": (
                    "RMS(candidate-reference) / RMS(reference spatial anomaly)"
                ),
                "correlation": "same-day spatial anomaly pattern correlation",
                "reference": "certified vector-math NEMO daily mean",
                "fields": [
                    "SSH on T mask", "SST on T mask",
                    "native surface U on U mask", "native surface V on V mask",
                    "depth-mean U on U mask", "depth-mean V on V mask",
                ],
                "depth_mean": (
                    "NEMO daily 3-D velocity weighted by z-coordinate e3u_0/"
                    "e3v_0 and masks; legoESM stored external-mode daily mean"
                ),
                "curve_days": [1, CURVE_DAYS],
                "regional_window_days": [30, 180],
                "regional_bins": (
                    "five equal fractional-distance bins within each wet row"
                ),
                "bottom_velocity": (
                    "UNMEASURED: the registered legoESM run stores surface and "
                    "external-mode velocities, not bottom velocity"
                ),
                "eddy_fraction": (
                    "31-day nearest-boundary high-pass native U/V EKE / KE"
                ),
                "eddy_years": [1, TOTAL_DAYS // 365],
            },
            "provenance": provenance,
            "mesh_mask_artifacts": {
                "nemo_vector": {
                    path.name: sha256(path)
                    for path in sorted(args.vector_nemo_root.glob(
                        "mesh_mask_????.nc"))
                },
                "nemo_scalar": {
                    path.name: sha256(path)
                    for path in sorted(args.scalar_nemo_root.glob(
                        "mesh_mask_????.nc"))
                },
            },
            "mask_identity": {
                "T": "NEMO mesh mask equals legoESM card mask",
                "U": "NEMO mesh mask equals legoESM card mask",
                "V": "NEMO mesh mask equals legoESM card mask",
            },
            "instrument_validation": {
                "nemo_twin_benchmark": twin_validation,
                "per_field_exact_and_one_ulp": field_controls,
            },
            "checkpoints": _checkpoint_rows(curves),
            "first_four_times_twin": first_excess,
            "regional_signed_sse_excess": regional,
            "leading_region_by_tied_field": leading_regions,
            "growth_days_30_180_by_tied_field": {
                "fields": growth,
                "metric": "absolute RMS error; fit residual in log-error space",
            },
            "yearly_statistics": yearly,
            "artifacts": {arrays_path.name: sha256(arrays_path)},
        }
        report_path = args.output / "gyre_hires_divergence_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed instrument boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(f"PASS -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
