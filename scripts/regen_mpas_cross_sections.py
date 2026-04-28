"""Regenerate T / u / speed latitude-vertical and longitude-vertical cross-section
plots for MPAS Eady runs from their saved `snapshots_native.nc`, using the fixed
`_bin_cross_section` in `diagnostic_io.py`.

The cross-section PNGs on disk from MPAS runs completed before the fix contain
a lat-compression artefact: with a global 181-row target grid and a regional
point cloud spanning only ~18° in latitude, the bin_centers were linearly
mapped onto all 181 target rows, compressing the actual gradient zone into a
narrow visual band. The fix (in `scripts/ocean_test_matrix/diagnostic_io.py`)
switches to a regional target lat-lon grid + max_dist KDTree cap when the
source cloud covers less than 80% of the global range.

Usage
-----
    .venv/bin/python scripts/regen_mpas_cross_sections.py \
        <run_dir_1> [<run_dir_2> ...]

Each directory is expected to contain `snapshots_native.nc` and is overwritten
with new T_{latitude,longitude}_vertical_cross_sections.png and the analogous
u and speed plots.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))
from ocean_test_matrix.diagnostic_io import _save_cross_sections  # noqa: E402


def _snapshots_from_nc(ds: xr.Dataset) -> dict[int, dict[str, np.ndarray]]:
    """Rebuild the per-step {field: array} snapshot dict expected by
    `_save_cross_sections`."""
    n_time = ds.sizes["time"]
    step_ids = list(range(n_time))  # synthetic step ids; dt is carried separately
    snaps: dict[int, dict[str, np.ndarray]] = {s: {} for s in step_ids}
    for name, da in ds.data_vars.items():
        arr = np.asarray(da.values)
        for i, s in enumerate(step_ids):
            snaps[s][name] = arr[i]
    return snaps


def regen_run(run_dir: Path) -> None:
    nc_path = run_dir / "snapshots_native.nc"
    npz_path = run_dir / "snapshots_native.npz"
    if not nc_path.exists() or not npz_path.exists():
        print(f"[skip] {run_dir}: snapshots_native.{{nc,npz}} missing")
        return
    ds = xr.open_dataset(nc_path)
    coord_kind = ds.attrs.get("coord_kind", "mpas")
    if coord_kind != "mpas":
        print(f"[skip] {run_dir}: coord_kind={coord_kind} (not mpas)")
        ds.close()
        return

    lon_deg = np.asarray(ds.lonCell.values, dtype=np.float64)
    lat_deg = np.asarray(ds.latCell.values, dtype=np.float64)
    ds.close()

    # The npz has clean 3D shapes (time, nCells, nlev) without the
    # interleaved-depth padding used by the nc writer. Use it to drive the
    # cross-section binning.
    npz = np.load(npz_path, allow_pickle=True)
    steps = np.asarray(npz["steps"], dtype=np.int64)
    times_days = np.asarray(npz["times_days"], dtype=np.float64)
    if len(times_days) >= 2 and len(steps) >= 2:
        dt_interval = (float(times_days[1] - times_days[0]) * 86400.0
                       / max(int(steps[1] - steps[0]), 1))
    else:
        dt_interval = 1.0

    # z_full depths for the 20 T-levels: the `depth` coordinate in the nc
    # file is the interleaved w+T stack. Reopen briefly to pull the last nlev
    # entries, which correspond to full-level T depths (positive downward).
    with xr.open_dataset(nc_path) as ds2:
        depth_all = np.asarray(ds2.depth.values, dtype=np.float64)
    nlev = int(npz["T_3d"].shape[-1])
    levels = depth_all[-nlev:]

    # Reconstruct per-step snapshot dict.
    snaps: dict[int, dict[str, np.ndarray]] = {}
    for i, s in enumerate(steps):
        snaps[int(s)] = {}
        for name in npz.files:
            if name in ("steps", "times_days"):
                continue
            arr = np.asarray(npz[name])
            if arr.ndim >= 2 and arr.shape[0] == len(steps):
                snaps[int(s)][name] = np.asarray(arr[i], dtype=np.float64)

    case_name = run_dir.name

    for field_3d_key, cmap in (
        ("T_3d", "RdYlBu_r"),
        ("u_3d", "RdBu_r"),
        ("speed_3d", "RdBu_r"),
    ):
        if field_3d_key not in npz.files:
            continue
        _save_cross_sections(
            run_dir, case_name, snaps, dt_interval, field_3d_key,
            coord_kind, lon_deg, lat_deg, levels, "Depth (m)",
            cmap=cmap)
        print(f"[ok]   {run_dir}: regenerated {field_3d_key} cross-sections")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for run_dir_str in sys.argv[1:]:
        run_dir = Path(run_dir_str).resolve()
        regen_run(run_dir)


if __name__ == "__main__":
    main()
