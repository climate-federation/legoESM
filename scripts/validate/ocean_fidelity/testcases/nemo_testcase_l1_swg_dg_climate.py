#!/usr/bin/env python3
"""Compare matched-window NEMO and legoESM SWG_DG daily climate fields."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import NamedTuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

NEMO_FIELDS = {"T": "sossheig", "U": "vozocrtx", "V": "vomecrty"}
SECONDS_PER_DAY = 86_400.0


class DailyFrame(NamedTuple):
    day: int
    ssh: np.ndarray
    u: np.ndarray
    v: np.ndarray


class DailyProvider(NamedTuple):
    name: str
    sampling: str
    provenance: dict
    iterate: Callable[[], Iterator[DailyFrame]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def assemble_coordinate_tiles(
        tiles: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
        expected_lon: np.ndarray, expected_lat: np.ndarray,
) -> np.ndarray:
    """Assemble overlapping rank tiles by coordinates, refusing conflicts."""
    expected_lon = np.asarray(expected_lon)
    expected_lat = np.asarray(expected_lat)
    require(expected_lon.shape == expected_lat.shape,
            "expected coordinate shapes differ")
    keys = {
        (float(lon), float(lat)): (j, i)
        for (j, i), lon, lat in zip(
            np.ndindex(expected_lon.shape), expected_lon.flat,
            expected_lat.flat, strict=True)
    }
    require(len(keys) == expected_lon.size,
            "expected coordinates are not unique")
    assembled = np.empty(expected_lon.shape, dtype=np.float64)
    filled = np.zeros(expected_lon.shape, dtype=bool)
    for lon, lat, values in tiles:
        lon, lat, values = np.asarray(lon), np.asarray(lat), np.asarray(values)
        require(lon.shape == lat.shape == values.shape,
                "rank tile coordinate/value shapes differ")
        require(np.all(np.isfinite(values)), "rank tile has non-finite values")
        for key, value in zip(
                zip(lon.flat, lat.flat, strict=True), values.flat, strict=True):
            at = keys.get((float(key[0]), float(key[1])))
            if at is None:  # NEMO outer or inter-rank halo outside card storage.
                continue
            if filled[at]:
                require(assembled[at] == value,
                        f"conflicting rank overlap at coordinate {key}")
            else:
                assembled[at] = value
                filled[at] = True
    require(np.all(filled),
            f"coordinate assembly missed {np.count_nonzero(~filled)} points")
    return assembled


def _expected_coordinates() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_MADDISON_NAMELIST,
        swg_horizontal_coordinates,
    )

    src = swg_horizontal_coordinates(SWG_MADDISON_NAMELIST)
    return {
        "T": (src["glamt"], src["gphit"]),
        "U": (src["glamu"], src["gphiu"]),
        "V": (src["glamv"], src["gphiv"]),
    }


def nemo_ioipsl_provider(root: Path, days: int) -> DailyProvider:
    """Build a repeatable provider over per-rank NEMO IOIPSL daily means."""
    import netCDF4

    coordinates = _expected_coordinates()
    paths = {
        kind: sorted(root.glob(f"SWG_*_grid_{kind}_*.nc"))
        for kind in NEMO_FIELDS
    }
    counts = {kind: len(found) for kind, found in paths.items()}
    require(len(set(counts.values())) == 1 and next(iter(counts.values())) > 0,
            f"NEMO rank-file counts differ: {counts}")
    expected_time = (np.arange(days, dtype=np.float64) + 0.5) * SECONDS_PER_DAY
    for kind, found in paths.items():
        variable = NEMO_FIELDS[kind]
        for path in found:
            with netCDF4.Dataset(path) as ds:
                ds.set_auto_mask(False)
                require(variable in ds.variables,
                        f"{path.name}: missing {variable}")
                require(ds[variable].dtype == np.dtype("float32"),
                        f"{path.name}.{variable}: not fp32")
                clock = np.asarray(ds["time_counter"][:], dtype=np.float64)
                require(clock.size >= days
                        and np.array_equal(clock[:days], expected_time),
                        f"{path.name}: daily clock/window differs")
    file_digests = {
        path.name: sha256(path) for found in paths.values() for path in found
    }

    def iterate() -> Iterator[DailyFrame]:
        with ExitStack() as stack:
            datasets = {
                kind: [stack.enter_context(netCDF4.Dataset(path))
                       for path in found]
                for kind, found in paths.items()
            }
            for group in datasets.values():
                for ds in group:
                    ds.set_auto_mask(False)
            for index in range(days):
                fields = {}
                for kind, variable in NEMO_FIELDS.items():
                    tiles = []
                    for ds in datasets[kind]:
                        values = np.asarray(ds[variable][index])
                        if values.ndim == 3:
                            require(values.shape[0] == 2
                                    and np.all(values[1] == 0.0),
                                    f"{variable}: invalid dummy layer")
                            values = values[0]
                        require(values.ndim == 2,
                                f"{variable}: rank field is not 2-D")
                        tiles.append((np.asarray(ds["nav_lon"][:]),
                                      np.asarray(ds["nav_lat"][:]), values))
                    fields[kind] = assemble_coordinate_tiles(
                        tiles, *coordinates[kind])
                yield DailyFrame(index + 1, fields["T"], fields["U"], fields["V"])

    return DailyProvider(
        "NEMO", "daily_mean",
        {"root": str(root.resolve()), "days": days,
         "rank_files_per_field": next(iter(counts.values())),
         "files_sha256": file_digests},
        iterate)


def legoesm_npz_provider(root: Path, days: int) -> DailyProvider:
    """Build a repeatable provider over content-pinned legoESM daily states."""
    manifest_path = root / "manifest.json"
    require(manifest_path.is_file(), f"missing complete manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get("status") == "complete"
            and manifest.get("completed_days") == days
            and manifest.get("days") == days,
            "legoESM manifest does not contain the complete requested window")
    require(manifest.get("schema") == "legoesm-swg-double-gyre-daily-v1",
            "unknown legoESM daily schema")
    expected = [root / f"day{day:04d}.npz" for day in range(1, days + 1)]
    require(sorted(root.glob("day????.npz")) == expected,
            "legoESM daily file sequence differs")
    pins = manifest.get("daily_sha256", {})
    require(set(pins) == {path.name for path in expected},
            "legoESM manifest does not pin every daily file")
    for path in expected:
        require(sha256(path) == pins[path.name],
                f"legoESM daily digest mismatch: {path.name}")
    sampling = manifest.get("sampling", "daily_snapshot")

    def iterate() -> Iterator[DailyFrame]:
        for day, path in enumerate(expected, 1):
            with np.load(path, allow_pickle=False) as data:
                require(int(data["day"]) == day,
                        f"{path.name}: embedded day differs")
                require(float(data["dt_s"]) == 50.0,
                        f"{path.name}: timestep differs")
                fields = [np.asarray(data[name], dtype=np.float64)
                          for name in ("ssh_m", "u_m_s", "v_m_s")]
            require(all(field.shape == (644, 644) for field in fields),
                    f"{path.name}: field shape differs")
            require(all(np.all(np.isfinite(field)) for field in fields),
                    f"{path.name}: non-finite field")
            yield DailyFrame(day, *fields)

    return DailyProvider(
        "legoESM", sampling,
        {"root": str(root.resolve()), "days": days,
         "manifest_sha256": sha256(manifest_path),
         "legoesm_git_sha": manifest.get("legoesm_git_sha"),
         "mode": manifest.get("mode"),
         "certified_policy": manifest.get("certified_policy")},
        iterate)


def _cell_centred(frame: DailyFrame, masks) -> DailyFrame:
    import jax.numpy as jnp
    from legoesm.ocean.diagnostics import velocity_to_cell_centre

    u_native = np.where(masks["U"], frame.u, 0.0)
    v_native = np.where(masks["V"], frame.v, 0.0)
    u = np.concatenate([np.zeros_like(u_native[:, :1]), u_native], axis=1)
    v = np.concatenate([np.zeros_like(v_native[:1, :]), v_native], axis=0)
    u_c, v_c = velocity_to_cell_centre(jnp.asarray(u), jnp.asarray(v))
    return DailyFrame(
        frame.day, np.where(masks["T"], frame.ssh, 0.0),
        np.asarray(u_c), np.asarray(v_c))


def diagnose(provider: DailyProvider, *, days: int, card=None,
             geometry: dict | None = None) -> dict[str, np.ndarray]:
    """Compute the frozen Round-4 metrics in two streaming passes."""
    import jax.numpy as jnp
    from legoesm.ocean.diagnostics import isotropic_energy_spectrum
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_MADDISON_NAMELIST,
        build_swg_card,
        swg_horizontal_coordinates,
    )

    if geometry is None:
        if card is None:
            card = build_swg_card(SWG_MADDISON_NAMELIST)
        from nemo_testcase_l1_swg_ladder import _card_masks
        source = swg_horizontal_coordinates(SWG_MADDISON_NAMELIST)
        geometry = {
            "masks": _card_masks(card),
            "area": np.asarray(card.recipe.grid.area),
            "x": np.asarray(source["glamt"]),
            "y": np.asarray(source["gphit"]),
            "spacing_m": float(source["spacing_m"]),
        }
    masks = geometry["masks"]
    wet = masks["T"]
    area = np.where(wet, np.asarray(geometry["area"]), 0.0)
    area_sum = np.sum(area, dtype=np.float64)
    require(area_sum > 0.0, "empty wet area")
    wet_indices = np.argwhere(wet)
    j0, i0 = wet_indices.min(axis=0)
    j1, i1 = wet_indices.max(axis=0) + 1
    wet_box = wet[j0:j1, i0:i1]
    require(np.all(wet_box), "spectral wet basin is not rectangular")
    spacing = float(geometry["spacing_m"])

    sum_ssh = np.zeros(wet.shape, dtype=np.float64)
    sum_u = np.zeros(wet.shape, dtype=np.float64)
    sum_v = np.zeros(wet.shape, dtype=np.float64)
    sum_speed = np.zeros(wet.shape, dtype=np.float64)
    ke_series = np.empty(days, dtype=np.float64)
    spectrum_sum = None
    wavenumber = None
    for index, raw in enumerate(provider.iterate()):
        require(index < days and raw.day == index + 1,
                f"{provider.name}: daily sequence differs")
        frame = _cell_centred(raw, masks)
        speed2 = frame.u * frame.u + frame.v * frame.v
        sum_ssh += frame.ssh
        sum_u += frame.u
        sum_v += frame.v
        sum_speed += np.sqrt(speed2)
        ke_series[index] = np.sum(0.5 * speed2 * area) / area_sum
        k, spectrum = isotropic_energy_spectrum(
            jnp.asarray(frame.u[j0:j1, i0:i1]),
            jnp.asarray(frame.v[j0:j1, i0:i1]), spacing)
        k, spectrum = np.asarray(k), np.asarray(spectrum)
        if spectrum_sum is None:
            wavenumber, spectrum_sum = k, np.zeros_like(spectrum)
        else:
            require(np.array_equal(k, wavenumber),
                    "spectrum wavenumbers changed across days")
        spectrum_sum += spectrum
    require(index + 1 == days, f"{provider.name}: daily sequence ended early")
    mean_ssh = sum_ssh / days
    mean_u = sum_u / days
    mean_v = sum_v / days
    mean_speed = sum_speed / days

    eke_map_sum = np.zeros(wet.shape, dtype=np.float64)
    eke_series = np.empty(days, dtype=np.float64)
    for index, raw in enumerate(provider.iterate()):
        frame = _cell_centred(raw, masks)
        anomaly2 = ((frame.u - mean_u) ** 2 + (frame.v - mean_v) ** 2)
        eke_map_sum += 0.5 * anomaly2
        eke_series[index] = np.sum(0.5 * anomaly2 * area) / area_sum

    y = np.asarray(geometry["y"])
    x = np.asarray(geometry["x"])
    jet_y = np.full(wet.shape[1], np.nan, dtype=np.float64)
    jet_speed = np.full(wet.shape[1], np.nan, dtype=np.float64)
    jet_x = np.full(wet.shape[1], np.nan, dtype=np.float64)
    for i in range(wet.shape[1]):
        js = np.flatnonzero(wet[:, i])
        if js.size:
            j = js[np.argmax(mean_u[js, i])]
            jet_x[i] = x[j, i]
            jet_y[i] = y[j, i]
            jet_speed[i] = mean_u[j, i]
    return {
        "mean_ssh_m": mean_ssh,
        "mean_u_m_s": mean_u,
        "mean_v_m_s": mean_v,
        "mean_speed_m_s": mean_speed,
        "eke_map_m2_s2": eke_map_sum / days,
        "basin_mean_ke_m2_s2": ke_series,
        "basin_mean_eke_m2_s2": eke_series,
        "jet_x_m": jet_x,
        "jet_y_m": jet_y,
        "jet_eastward_speed_m_s": jet_speed,
        "spectrum_wavenumber_m-1": wavenumber,
        "mean_kinetic_energy_spectrum_m3_s2": spectrum_sum / days,
    }


def compare(providers: tuple[DailyProvider, DailyProvider], *, days: int) -> dict:
    """Require controlled sampling and diagnose both models independently."""
    left, right = providers
    require(left.sampling == right.sampling,
            f"sampling mismatch: {left.name}={left.sampling}, "
            f"{right.name}={right.sampling}")
    return {left.name: diagnose(left, days=days),
            right.name: diagnose(right, days=days)}


def _summary(arrays: dict[str, np.ndarray]) -> dict:
    out = {}
    for name, value in arrays.items():
        finite = np.asarray(value)[np.isfinite(value)]
        out[name] = {
            "shape": list(np.asarray(value).shape),
            "sha256": array_digest(np.asarray(value)),
            "min": float(np.min(finite)), "max": float(np.max(finite)),
            "mean": float(np.mean(finite, dtype=np.float64)),
        }
    return out


def write_outputs(
        output: Path, diagnosed: dict[str, dict[str, np.ndarray]],
        providers: tuple[DailyProvider, DailyProvider], *, days: int,
) -> dict:
    """Write the JSON, compressed arrays, and common-scale comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    flat = {f"{model}.{name}": value
            for model, arrays in diagnosed.items() for name, value in arrays.items()}
    arrays_path = output / "swg_dg_climate_arrays.npz"
    np.savez_compressed(arrays_path, **flat)
    names = [provider.name for provider in providers]
    left, right = (diagnosed[name] for name in names)
    fig, axes = plt.subplots(4, 4, figsize=(16, 14), constrained_layout=True)
    maps = ("mean_ssh_m", "mean_u_m_s", "mean_v_m_s", "mean_speed_m_s",
            "eke_map_m2_s2")
    for index, field in enumerate(maps):
        row, col = divmod(index, 2)
        values = np.concatenate([left[field].ravel(), right[field].ravel()])
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        for offset, model in enumerate(names):
            ax = axes[row, 2 * col + offset]
            image = ax.imshow(diagnosed[model][field], origin="lower",
                              vmin=vmin, vmax=vmax)
            ax.set_title(f"{model}: {field}")
            fig.colorbar(image, ax=ax, shrink=0.75)
    axes[3, 0].plot(left["basin_mean_ke_m2_s2"], label=f"{names[0]} KE")
    axes[3, 0].plot(right["basin_mean_ke_m2_s2"], label=f"{names[1]} KE")
    axes[3, 0].plot(left["basin_mean_eke_m2_s2"], "--", label=f"{names[0]} EKE")
    axes[3, 0].plot(right["basin_mean_eke_m2_s2"], "--", label=f"{names[1]} EKE")
    axes[3, 0].legend(fontsize=7)
    axes[3, 0].set_title("Basin mean energy")
    axes[3, 1].plot(left["jet_x_m"], left["jet_eastward_speed_m_s"], label=names[0])
    axes[3, 1].plot(right["jet_x_m"], right["jet_eastward_speed_m_s"], label=names[1])
    axes[3, 1].legend()
    axes[3, 1].set_title("Jet eastward-extension profile")
    axes[3, 2].plot(left["jet_x_m"], left["jet_y_m"], label=names[0])
    axes[3, 2].plot(right["jet_x_m"], right["jet_y_m"], label=names[1])
    axes[3, 2].legend()
    axes[3, 2].set_title("Mean eastward jet core")
    for model, arrays in diagnosed.items():
        axes[3, 3].loglog(arrays["spectrum_wavenumber_m-1"][1:],
                          arrays["mean_kinetic_energy_spectrum_m3_s2"][1:],
                          label=model)
    axes[3, 3].legend()
    axes[3, 3].set_title("Mean KE spectrum")
    figure_path = output / "swg_dg_climate_side_by_side.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    report = {
        "schema": "swg-dg-climate-comparison-v1",
        "label": "year 1, spin-up" if days == 365 else f"days 1-{days}",
        "window_days": [1, days], "sampling": providers[0].sampling,
        "provenance": {provider.name: provider.provenance for provider in providers},
        "metrics": {model: _summary(arrays) for model, arrays in diagnosed.items()},
        "artifacts": {
            arrays_path.name: sha256(arrays_path),
            figure_path.name: sha256(figure_path),
        },
    }
    report_path = output / "swg_dg_climate_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-root", type=Path, required=True)
    parser.add_argument("--legoesm-root", type=Path, required=True)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(args.days > 0, "days must be positive")
        providers = (nemo_ioipsl_provider(args.nemo_root, args.days),
                     legoesm_npz_provider(args.legoesm_root, args.days))
        diagnosed = compare(providers, days=args.days)
        report = write_outputs(args.output, diagnosed, providers, days=args.days)
    except Exception as exc:  # noqa: BLE001 - fail-closed instrument boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(f"PASS {report['label']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
