#!/usr/bin/env python3
"""Compare matched-window NEMO and legoESM SWG_DG daily climate fields."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
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
NEMO_DG_EXECUTABLE_SHA256 = (
    "0be0d4b0846c34b3540105a54eaa6de225e03a9d8927cf5dcc8d85198b88bf09")
NEMO_DG_NAMELIST_CFG_SHA256 = (
    "0f6fc303af68549c0c78dc1f22f9e29ed5a6feb3b4fafc88b6ee4281bb42e309")
NEMO_DG_NAMELIST_REF_SHA256 = (
    "b04f2ce4d12aa247ae3b707483d25cf2d38d0b3cf8817d71d33e66918fd81cfd")


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


def _require_legoesm_deck(namelist: dict, expected_namelist=None) -> None:
    """Bind a legoESM manifest to one explicitly selected resolved deck."""
    if expected_namelist is None:
        from legoesm.ocean.fidelity.nemo_testcase_recipe import (
            SWG_MADDISON_NAMELIST,
        )
        expected_namelist = SWG_MADDISON_NAMELIST

    # Normalize tuple-valued records through the same JSON representation the
    # driver manifest uses before exact field comparison.
    expected = json.loads(json.dumps(expected_namelist._asdict()))
    # Run length is the only permitted difference between year 1 and the
    # registered three-year window.  Everything physical -- including
    # rn_shlat/free-slip and all dispatch switches -- must match the deck.
    for field in expected.keys() - {"nn_itend", "nn_stock"}:
        require(field in namelist and namelist[field] == expected[field],
                f"legoESM manifest deck differs at {field}")


def _require_nemo_deck(root: Path) -> dict:
    """Bind an IOIPSL directory to the operator's resolved SWG_DG deck."""
    executable = root / "nemo"
    require(executable.is_file(), f"missing NEMO executable {executable}")
    require(sha256(executable) == NEMO_DG_EXECUTABLE_SHA256,
            "NEMO executable is not the registered SWG_DG build")
    pins = {
        "nemo": NEMO_DG_EXECUTABLE_SHA256,
        "namelist_cfg": NEMO_DG_NAMELIST_CFG_SHA256,
        "namelist_ref": NEMO_DG_NAMELIST_REF_SHA256,
    }
    manifest_path = root / "manifest.sha256"
    require(manifest_path.is_file(), f"missing NEMO manifest {manifest_path}")
    manifest = {}
    for line in manifest_path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        manifest[name] = digest
    require(manifest == pins, "NEMO manifest does not describe the registered deck")
    for name, digest in pins.items():
        require((root / name).is_file() and sha256(root / name) == digest,
                f"NEMO deck digest mismatch: {name}")
    resolved_path = root / "resolved_namelist_lines.txt"
    require(resolved_path.is_file(),
            f"missing resolved namelist inventory {resolved_path}")
    resolved = {}
    for line in (root / "namelist_cfg").read_text().splitlines():
        match = re.match(
            r"^\s*([A-Za-z0-9_]+)\s*=\s*([^,!\s/]+)", line)
        if match:
            resolved[match.group(1)] = match.group(2).replace(
                "D", "E").replace("d", "e")
    for line in resolved_path.read_text().splitlines():
        match = re.match(
            r"^\d+:\s*([A-Za-z0-9_]+)\s*=\s*([^!\s]+)", line)
        if match:
            resolved[match.group(1)] = match.group(2).replace("D", "E").replace(
                "d", "e")
    expected = {
        "nn_SWG": 16, "rn_domsiz": 4_000_000.0, "rn_tau": 0.10,
        "rn_modified_grav": 9.80665, "rn_rfr": 1.0e-7,
        "rn_beta": 2.0e-11, "rn_f0": 0.7e-4, "rn_Uv": 2.0,
        "rn_Lv": 10.0, "rn_Dt": 50.0, "nn_write": 1_728,
        "rn_dx": 100_000.0, "jpkglo": 2, "rn_shlat": 0.0,
    }
    for field, value in expected.items():
        require(field in resolved, f"resolved NEMO deck is missing {field}")
        actual = float(resolved[field])
        require(actual == value, f"resolved NEMO deck differs at {field}")
    return {
        "executable_sha256": NEMO_DG_EXECUTABLE_SHA256,
        "manifest_sha256": sha256(manifest_path),
        "namelist_cfg_sha256": NEMO_DG_NAMELIST_CFG_SHA256,
        "namelist_ref_sha256": NEMO_DG_NAMELIST_REF_SHA256,
        "resolved_namelist_sha256": sha256(resolved_path),
        "resolved_values": {field: resolved[field] for field in expected},
    }


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


def nemo_ioipsl_provider(
        root: Path, days: int, *, start_day: int = 1,
) -> DailyProvider:
    """Build a repeatable provider over per-rank NEMO IOIPSL daily means."""
    import netCDF4

    require(start_day > 0 and days > 0, "NEMO window must be positive")
    deck_provenance = _require_nemo_deck(root)
    coordinates = _expected_coordinates()
    paths = {
        kind: sorted(root.glob(f"SWG_*_grid_{kind}_*.nc"))
        for kind in NEMO_FIELDS
    }
    counts = {kind: len(found) for kind, found in paths.items()}
    require(len(set(counts.values())) == 1 and next(iter(counts.values())) > 0,
            f"NEMO rank-file counts differ: {counts}")
    stop_day = start_day + days - 1
    expected_time = (
        np.arange(start_day - 1, stop_day, dtype=np.float64) + 0.5
    ) * SECONDS_PER_DAY
    for kind, found in paths.items():
        variable = NEMO_FIELDS[kind]
        for path in found:
            with netCDF4.Dataset(path) as ds:
                ds.set_auto_mask(False)
                require(variable in ds.variables,
                        f"{path.name}: missing {variable}")
                require(ds[variable].dtype == np.dtype("float32"),
                        f"{path.name}.{variable}: not fp32")
                require(getattr(ds[variable], "online_operation", None) == "ave(x)",
                        f"{path.name}.{variable}: not a NEMO daily mean")
                require(float(getattr(ds[variable], "interval_operation", -1.0))
                        == 50.0,
                        f"{path.name}.{variable}: averaging sample interval differs")
                require(float(getattr(ds[variable], "interval_write", -1.0))
                        == SECONDS_PER_DAY,
                        f"{path.name}.{variable}: averaging window differs")
                clock = np.asarray(ds["time_counter"][:], dtype=np.float64)
                require(clock.size >= stop_day and np.array_equal(
                    clock[start_day - 1:stop_day], expected_time),
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
            for local_index, index in enumerate(
                    range(start_day - 1, stop_day), 1):
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
                yield DailyFrame(
                    local_index, fields["T"], fields["U"], fields["V"])

    return DailyProvider(
        "NEMO", "daily_mean",
        {"root": str(root.resolve()), "days": days,
         "window_days": [start_day, stop_day],
         "rank_files_per_field": next(iter(counts.values())),
         "files_sha256": file_digests, "deck": deck_provenance},
        iterate)


def legoesm_npz_provider(
        root: Path, days: int, *, start_day: int = 1,
        name: str = "legoESM", expected_namelist=None,
) -> DailyProvider:
    """Build a repeatable provider over content-pinned legoESM daily states."""
    require(start_day > 0 and days > 0, "legoESM window must be positive")
    manifest_path = root / "manifest.json"
    require(manifest_path.is_file(), f"missing complete manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    total_days = manifest.get("days")
    stop_day = start_day + days - 1
    require(isinstance(total_days, int) and manifest.get("status") == "complete"
            and manifest.get("completed_days") == total_days
            and total_days >= stop_day,
            "legoESM manifest does not contain the complete requested window")
    require(manifest.get("schema") in {
                "legoesm-swg-double-gyre-daily-v1",
                "legoesm-swg-double-gyre-daily-v2"},
            "unknown legoESM daily schema")
    if expected_namelist is None:
        from legoesm.ocean.fidelity.nemo_testcase_recipe import (
            SWG_MADDISON_NAMELIST,
        )
        expected_namelist = SWG_MADDISON_NAMELIST
    _require_legoesm_deck(manifest.get("namelist", {}), expected_namelist)
    complete = [root / f"day{day:04d}.npz"
                for day in range(1, total_days + 1)]
    require(sorted(root.glob("day????.npz")) == complete,
            "legoESM daily file sequence differs")
    expected = complete[start_day - 1:stop_day]
    pins = manifest.get("daily_sha256", {})
    require(set(pins) == {path.name for path in complete},
            "legoESM manifest does not pin every daily file")
    for path in expected:
        require(sha256(path) == pins[path.name],
                f"legoESM daily digest mismatch: {path.name}")
    sampling = manifest.get("sampling", "daily_snapshot")
    fields = (
        ("ssh_daily_mean_m", "u_daily_mean_m_s", "v_daily_mean_m_s")
        if sampling == "daily_mean" else ("ssh_m", "u_m_s", "v_m_s"))
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card
    expected_shape = tuple(
        build_swg_card(expected_namelist).recipe.land_mask.shape)
    if sampling == "daily_mean":
        require(manifest.get("schema") == "legoesm-swg-double-gyre-daily-v2",
                "daily-mean sampling requires the v2 legoESM schema")

    def iterate() -> Iterator[DailyFrame]:
        for local_day, path in enumerate(expected, 1):
            absolute_day = start_day + local_day - 1
            with np.load(path, allow_pickle=False) as data:
                require(int(data["day"]) == absolute_day,
                        f"{path.name}: embedded day differs")
                require(float(data["dt_s"]) == expected_namelist.rn_Dt_s,
                        f"{path.name}: timestep differs")
                if sampling == "daily_mean":
                    require(data["daily_mean_sampling"].item() ==
                            "arithmetic mean of every post-step state in model day",
                            f"{path.name}: daily-mean sampling label differs")
                    require(int(data["daily_mean_sample_count"]) == 1_728,
                            f"{path.name}: daily-mean sample count differs")
                values = [np.asarray(data[name], dtype=np.float64)
                          for name in fields]
            require(all(field.shape == expected_shape for field in values),
                    f"{path.name}: field shape differs")
            require(all(np.all(np.isfinite(field)) for field in values),
                    f"{path.name}: non-finite field")
            yield DailyFrame(local_day, *values)

    return DailyProvider(
        name, sampling,
        {"root": str(root.resolve()), "days": days,
         "window_days": [start_day, stop_day],
         "manifest_sha256": sha256(manifest_path),
         "legoesm_git_sha": manifest.get("legoesm_git_sha"),
         "mode": manifest.get("mode"),
         "certified_policy": manifest.get("certified_policy")},
        iterate)


def filtered_npz_provider(
        root: Path, days: int, *, start_day: int = 1,
        name: str = "filtered truth",
) -> DailyProvider:
    """Build a provider over content-pinned dg50-filtered fine truth."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_DG50_NAMELIST,
        build_swg_card,
    )

    require(start_day > 0 and days > 0, "filtered window must be positive")
    manifest_path = root / "manifest.json"
    require(manifest_path.is_file(), f"missing complete manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    total_days = manifest.get("days")
    stop_day = start_day + days - 1
    require(manifest.get("schema") in {
                "legoesm-swg-filtered-truth-v1",
                "legoesm-swg-conservative-truth-v2"}
            and manifest.get("status") == "complete"
            and manifest.get("completed_days") == total_days
            and isinstance(total_days, int) and total_days >= stop_day,
            "filtered manifest does not contain the complete requested window")
    _require_legoesm_deck(
        manifest.get("target_namelist", {}), SWG_DG50_NAMELIST)
    sampling = manifest.get("sampling")
    require(sampling in ("daily_snapshot", "daily_mean"),
            "filtered sampling is unsupported")
    pins = manifest.get("output_daily_sha256", {})
    complete = [root / f"day{day:04d}.npz" for day in range(1, total_days + 1)]
    require(sorted(root.glob("day????.npz")) == complete,
            "filtered daily file sequence differs")
    require(set(pins) == {path.name for path in complete},
            "filtered manifest does not pin every output")
    expected = complete[start_day - 1:stop_day]
    shape = tuple(build_swg_card(SWG_DG50_NAMELIST).recipe.land_mask.shape)
    fields = (("ssh_daily_mean_m", "u_daily_mean_m_s", "v_daily_mean_m_s")
              if sampling == "daily_mean" else ("ssh_m", "u_m_s", "v_m_s"))

    def iterate() -> Iterator[DailyFrame]:
        for local_day, path in enumerate(expected, 1):
            require(sha256(path) == pins[path.name],
                    f"filtered output digest mismatch: {path.name}")
            with np.load(path, allow_pickle=False) as data:
                require(int(data["day"]) == start_day + local_day - 1,
                        f"{path.name}: embedded day differs")
                values = [np.asarray(data[field], dtype=np.float64)
                          for field in fields]
            require(all(value.shape == shape for value in values),
                    f"{path.name}: filtered field shape differs")
            require(all(np.all(np.isfinite(value)) for value in values),
                    f"{path.name}: filtered field is non-finite")
            yield DailyFrame(local_day, *values)

    return DailyProvider(
        name, sampling,
        {"root": str(root.resolve()), "days": days,
         "window_days": [start_day, stop_day],
         "manifest_sha256": sha256(manifest_path),
         "legoesm_git_sha": manifest.get("legoesm_git_sha"),
         "filter": manifest.get("filter")},
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


def checkerboard_index(u: np.ndarray, v: np.ndarray) -> float:
    """Normalized native velocity projection onto the two-cell grid mode."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError("checkerboard fields must be conformable 2-D arrays")
    jj, ii = np.indices(u.shape)
    pattern = np.where((ii + jj) % 2 == 0, 1.0, -1.0)
    projection2 = np.mean(u * pattern) ** 2 + np.mean(v * pattern) ** 2
    energy2 = np.mean(u * u + v * v, dtype=np.float64)
    return float(np.sqrt(projection2 / energy2)) if energy2 > 0.0 else 0.0


def diagnose(provider: DailyProvider, *, days: int, card=None, namelist=None,
             geometry: dict | None = None) -> dict[str, np.ndarray]:
    """Compute the frozen Round-4 metrics in two streaming passes."""
    import jax.numpy as jnp
    from legoesm.ocean.diagnostics import (
        isotropic_energy_spectrum,
        isotropic_enstrophy_spectrum,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_MADDISON_NAMELIST,
        build_swg_card,
        swg_horizontal_coordinates,
    )

    if geometry is None:
        if card is None:
            card = build_swg_card(SWG_MADDISON_NAMELIST)
            namelist = SWG_MADDISON_NAMELIST
        require(namelist is not None,
                "a supplied card requires its resolved namelist")
        from nemo_testcase_l1_swg_ladder import _card_masks
        source = swg_horizontal_coordinates(namelist)
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
    enstrophy_spectrum_sum = None
    wavenumber = None
    checkerboard = np.empty(days, dtype=np.float64)
    min_thickness = np.empty(days, dtype=np.float64) if card is not None else None
    max_u_courant = np.empty(days, dtype=np.float64) if namelist is not None else None
    max_v_courant = np.empty(days, dtype=np.float64) if namelist is not None else None
    bathymetry = (np.asarray(card.recipe.initial_state.H_bathy.data)
                  if card is not None else None)
    for index, raw in enumerate(provider.iterate()):
        require(index < days and raw.day == index + 1,
                f"{provider.name}: daily sequence differs")
        frame = _cell_centred(raw, masks)
        if card is not None:
            min_thickness[index] = np.min(
                (bathymetry + raw.ssh)[wet])
        if namelist is not None:
            max_u_courant[index] = (
                np.max(np.abs(raw.u[masks["U"]])) * namelist.rn_Dt_s / spacing)
            max_v_courant[index] = (
                np.max(np.abs(raw.v[masks["V"]])) * namelist.rn_Dt_s / spacing)
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
        u_box = frame.u[j0:j1, i0:i1]
        v_box = frame.v[j0:j1, i0:i1]
        zeta = (np.gradient(v_box, spacing, axis=1)
                - np.gradient(u_box, spacing, axis=0))
        k_zeta, enstrophy_spectrum = isotropic_enstrophy_spectrum(
            jnp.asarray(zeta), spacing)
        k_zeta = np.asarray(k_zeta)
        enstrophy_spectrum = np.asarray(enstrophy_spectrum)
        require(np.array_equal(k_zeta, wavenumber),
                "KE and enstrophy wavenumbers differ")
        if enstrophy_spectrum_sum is None:
            enstrophy_spectrum_sum = np.zeros_like(enstrophy_spectrum)
        enstrophy_spectrum_sum += enstrophy_spectrum
        checkerboard[index] = checkerboard_index(u_box, v_box)
    require(index + 1 == days, f"{provider.name}: daily sequence ended early")
    mean_ssh = sum_ssh / days
    mean_u = sum_u / days
    mean_v = sum_v / days
    mean_speed = sum_speed / days
    mean_ke_spectrum = spectrum_sum / days
    mean_enstrophy_spectrum = enstrophy_spectrum_sum / days
    top_octave = wavenumber >= 0.5 * np.max(wavenumber)
    resolved = wavenumber > 0.0
    ke_spectral_total = np.sum(mean_ke_spectrum[resolved], dtype=np.float64)
    enstrophy_spectral_total = np.sum(
        mean_enstrophy_spectrum[resolved], dtype=np.float64)
    require(ke_spectral_total > 0.0 and enstrophy_spectral_total > 0.0,
            "spectral health denominator is zero")

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
    result = {
        "mean_ssh_m": np.where(wet, mean_ssh, np.nan),
        "mean_u_m_s": np.where(wet, mean_u, np.nan),
        "mean_v_m_s": np.where(wet, mean_v, np.nan),
        "mean_speed_m_s": np.where(wet, mean_speed, np.nan),
        "eke_map_m2_s2": np.where(wet, eke_map_sum / days, np.nan),
        "basin_mean_ke_m2_s2": ke_series,
        "basin_mean_eke_m2_s2": eke_series,
        "jet_x_m": jet_x,
        "jet_y_m": jet_y,
        "jet_eastward_speed_m_s": jet_speed,
        "spectrum_wavenumber_m-1": wavenumber,
        "mean_kinetic_energy_spectrum_m3_s2": mean_ke_spectrum,
        "mean_enstrophy_spectrum_m_s2": mean_enstrophy_spectrum,
        "top_octave_ke_fraction": np.asarray([
            np.sum(mean_ke_spectrum[top_octave], dtype=np.float64)
            / ke_spectral_total]),
        "top_octave_enstrophy_fraction": np.asarray([
            np.sum(mean_enstrophy_spectrum[top_octave], dtype=np.float64)
            / enstrophy_spectral_total]),
        "native_checkerboard_index": checkerboard,
    }
    if card is not None:
        result.update({
            "minimum_total_layer_thickness_m": min_thickness,
            "maximum_u_courant": max_u_courant,
            "maximum_v_courant": max_v_courant,
        })
    return result


def compare(providers: tuple[DailyProvider, DailyProvider], *, days: int,
            card=None, namelist=None) -> dict:
    """Require controlled sampling and diagnose both models independently."""
    left, right = providers
    require(left.sampling == right.sampling,
            f"sampling mismatch: {left.name}={left.sampling}, "
            f"{right.name}={right.sampling}")
    return {left.name: diagnose(left, days=days, card=card, namelist=namelist),
            right.name: diagnose(right, days=days, card=card, namelist=namelist)}


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


def _comparison_summary(
        left: dict[str, np.ndarray], right: dict[str, np.ndarray],
) -> dict:
    """Return explicit right-minus-left scalar comparisons for every metric."""
    out = {}
    require(left.keys() == right.keys(), "model metric sets differ")
    for name in left:
        a, b = np.asarray(left[name]), np.asarray(right[name])
        require(a.shape == b.shape, f"comparison shape differs for {name}")
        paired = np.isfinite(a) & np.isfinite(b)
        require(np.any(paired), f"comparison has no finite pairs for {name}")
        delta = b[paired] - a[paired]
        out[name] = {
            "orientation": "right_minus_left",
            "paired_values": int(delta.size),
            "bias": float(np.mean(delta, dtype=np.float64)),
            "mean_absolute_difference": float(
                np.mean(np.abs(delta), dtype=np.float64)),
            "root_mean_square_difference": float(np.sqrt(
                np.mean(delta * delta, dtype=np.float64))),
            "max_absolute_difference": float(np.max(np.abs(delta))),
        }
    return out


def write_outputs(
        output: Path, diagnosed: dict[str, dict[str, np.ndarray]],
        providers: tuple[DailyProvider, DailyProvider], *, days: int,
        start_day: int = 1,
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
        "label": ("year 1, spin-up" if start_day == 1 and days == 365
                  else "years 2-3" if start_day == 366 and days == 730
                  else f"days {start_day}-{start_day + days - 1}"),
        "window_days": [start_day, start_day + days - 1],
        "sampling": providers[0].sampling,
        "provenance": {provider.name: provider.provenance for provider in providers},
        "metrics": {model: _summary(arrays) for model, arrays in diagnosed.items()},
        "comparison": {
            "left": names[0], "right": names[1],
            "metrics": _comparison_summary(left, right),
        },
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
    parser.add_argument("--nemo-root", type=Path)
    parser.add_argument("--legoesm-root", type=Path)
    parser.add_argument("--filtered-root", type=Path)
    parser.add_argument("--coarse-root", type=Path)
    parser.add_argument("--start-day", type=int, default=1)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(args.start_day > 0 and args.days > 0,
                "window must be positive")
        oracle_pair = args.nemo_root is not None or args.legoesm_root is not None
        online_pair = args.filtered_root is not None or args.coarse_root is not None
        require(oracle_pair != online_pair,
                "select exactly one complete NEMO/legoESM or filtered/coarse pair")
        if oracle_pair:
            require(args.nemo_root is not None and args.legoesm_root is not None,
                    "NEMO/legoESM pair is incomplete")
            providers = (nemo_ioipsl_provider(
                             args.nemo_root, args.days, start_day=args.start_day),
                         legoesm_npz_provider(
                             args.legoesm_root, args.days,
                             start_day=args.start_day))
            card = None
            namelist = None
        else:
            require(args.filtered_root is not None and args.coarse_root is not None,
                    "filtered/coarse pair is incomplete")
            from legoesm.ocean.fidelity.nemo_testcase_recipe import (
                SWG_DG50_NAMELIST,
                build_swg_card,
            )
            providers = (filtered_npz_provider(
                             args.filtered_root, args.days,
                             start_day=args.start_day),
                         legoesm_npz_provider(
                             args.coarse_root, args.days,
                             start_day=args.start_day, name="dg50 no closure",
                             expected_namelist=SWG_DG50_NAMELIST))
            card = build_swg_card(SWG_DG50_NAMELIST)
            namelist = SWG_DG50_NAMELIST
        diagnosed = compare(
            providers, days=args.days, card=card, namelist=namelist)
        report = write_outputs(
            args.output, diagnosed, providers, days=args.days,
            start_day=args.start_day)
    except Exception as exc:  # noqa: BLE001 - fail-closed instrument boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    print(f"PASS {report['label']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
