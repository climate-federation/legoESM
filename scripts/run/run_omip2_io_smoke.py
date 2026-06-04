#!/usr/bin/env python
"""End-to-end OMIP-2 I/O smoke test.

Exercises the OMIP-2 protocol-compliance additions:
    * WOA bilinear + monthly + bathymetry-aware IC loader.
    * Expanded CMOR Omon (thetao, so, uo, vo, wo, sos, zos,
      tos, mlotst, hfds, wfo, tauuo, tauvo, rhopoto, sic) +
      SImon (siconc, sithick, siu, siv, sitemptop) tables.

Pipeline:
    1.  Synthesize a small WOA-style NetCDF dataset in a temp dir.
    2.  Build a 5°-resolution lat-lon ocean grid and a z* coordinate.
    3.  Initialize T, S from WOA via bilinear horizontal interp with a
        bathymetry mask.  No integration — sidesteps the pre-existing
        cubed-sphere dycore blow-up that is unrelated to the IO work.
    4.  Write a minimal Omon + SImon snapshot bundle through
        CFWriter.write_field.
    5.  Re-open every written file and validate basic CF/CMIP6
        attributes are present and consistent with the spec.

Run:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
        .venv/bin/python scripts/run_omip2_io_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the project importable when invoked from a checkout
_PROJ = Path(__file__).resolve().parents[2]
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

import numpy as np
import xarray as xr


def _write_synthetic_woa(tmp: Path, n_time: int = 12):
    """Write a length-``n_time`` WOA-style monthly T/S NetCDF."""
    lat = np.linspace(-87.5, 87.5, 36)        # 5° in lat
    lon = np.linspace(2.5, 357.5, 72)         # 5° in lon
    depth = np.array(
        [0, 10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000],
        dtype=np.float64,
    )
    T = np.zeros((n_time, depth.size, lat.size, lon.size))
    S = np.zeros((n_time, depth.size, lat.size, lon.size))
    for t in range(n_time):
        for k, z in enumerate(depth):
            T[t, k, :, :] = (
                25.0 * np.cos(np.radians(lat))[:, None]
                * np.exp(-z / 1500.0)
                + 0.5 * np.sin(2 * np.pi * t / 12.0)
                * np.cos(np.radians(lat))[:, None]
            )
            S[t, k, :, :] = 34.5 + 0.4 * np.sin(np.radians(lat))[:, None]
    T_path = tmp / "woa18_T.nc"
    S_path = tmp / "woa18_S.nc"
    xr.Dataset(
        {"t_an": (("time", "depth", "lat", "lon"), T)},
        coords={"time": np.arange(n_time), "depth": depth,
                "lat": lat, "lon": lon},
    ).to_netcdf(T_path)
    xr.Dataset(
        {"s_an": (("time", "depth", "lat", "lon"), S)},
        coords={"time": np.arange(n_time), "depth": depth,
                "lat": lat, "lon": lon},
    ).to_netcdf(S_path)
    return T_path, S_path


def _build_grid_and_coord():
    from legoesm.ocean.vertical import create_ocean_z_star
    z_coord = create_ocean_z_star(n_levels=15, H_max=5000.0)

    # 5° lat-lon, lat-major.  Duck-typed grid with ``lat``/``lon``
    # attributes in radians, matching init_ocean_from_woa's branch.
    nlat, nlon = 36, 72
    lat_deg = np.linspace(-87.5, 87.5, nlat)
    lon_deg = np.linspace(2.5, 357.5, nlon)
    LAT, LON = np.meshgrid(lat_deg, lon_deg, indexing="ij")

    class _Grid:
        lat = np.radians(LAT)
        lon = np.radians(LON)
    grid = _Grid()
    return grid, z_coord, lat_deg, lon_deg


def _omip2_ic(grid, z_coord, T_path, S_path):
    """Initialise T/S via the new WOA path (bilinear, January, with a
    smooth bathymetry).  ``bathymetry_depth`` is positive metres and
    must be strictly positive — set polar/shallow cells to 200 m."""
    from legoesm.ocean.init_woa import init_ocean_from_woa
    # Smooth shallow→deep mask
    lat_deg = np.degrees(grid.lat)
    bath = 200.0 + 4500.0 * np.cos(np.radians(lat_deg)) ** 2
    T, S = init_ocean_from_woa(
        grid, z_coord, T_path, S_path,
        month=1,                     # January monthly climatology
        interp="bilinear",
        bathymetry_depth=bath,
        monthly_layout="concatenated",
    )
    return np.asarray(T), np.asarray(S), bath


def _write_omip2_outputs(tmp: Path, T, S, lat, lon, depth):
    """Write a representative Omon + SImon snapshot bundle."""
    from legoesm.io.cmor_output import CFWriter
    writer = CFWriter(
        output_dir=str(tmp / "cmor"),
        experiment_id="omip2",
        model_id="legoESM-1-0",
        freq="mon",
        calendar="noleap",
        ref_date="0001-01-01",
    )
    written = []

    # --- 3D ocean: thetao + so (callers must pre-convert to CMIP6
    # units; thetao expects degC, our init_woa already returns degC,
    # so no conversion needed here).
    # CMOR Omon dim order is (time, depth, lat, lon) — the writer
    # prepends a leading time axis, so callers pass (depth, lat, lon).
    # init_ocean_from_woa returns (lat, lon, depth); transpose.
    T_dll = np.moveaxis(T, -1, 0)
    S_dll = np.moveaxis(S, -1, 0)
    written.append(writer.write_field(
        "thetao", T_dll, time=15.0, time_bounds=(0.0, 30.0),
        lat=lat, lon=lon, depth=depth,
    ))
    written.append(writer.write_field(
        "so", S_dll, time=15.0, time_bounds=(0.0, 30.0),
        lat=lat, lon=lon, depth=depth,
    ))

    # --- 2D surface ocean: tos (K), sos, zos, mlotst, hfds, tauuo,
    # tauvo — synthesize realistic 2D fields from the IC.
    tos = T[..., 0] + 273.15            # surface T → K for CMIP6 tos
    sos = S[..., 0]                     # surface S
    zos = 0.1 * np.sin(np.radians(np.degrees(np.linspace(-90, 90, T.shape[0]))))[:, None]
    zos = np.broadcast_to(zos, (T.shape[0], T.shape[1]))
    mlotst = 50.0 + 100.0 * np.cos(np.radians(np.degrees(np.linspace(-90, 90, T.shape[0]))))[:, None]
    mlotst = np.broadcast_to(mlotst, (T.shape[0], T.shape[1]))
    hfds = -5.0 + np.zeros_like(tos)
    tauuo = 0.05 * np.ones_like(tos)
    tauvo = np.zeros_like(tos)

    for name, data in [
        ("tos", tos), ("sos", sos), ("zos", zos), ("mlotst", mlotst),
        ("hfds", hfds), ("tauuo", tauuo), ("tauvo", tauvo),
    ]:
        written.append(writer.write_field(
            name, data, time=15.0, time_bounds=(0.0, 30.0),
            lat=lat, lon=lon,
        ))

    # --- SImon: a synthetic Arctic-cap ice field
    lat2 = np.broadcast_to(lat[:, None], (lat.size, lon.size))
    siconc = np.clip(20.0 * (lat2 - 60.0) / 30.0, 0.0, 100.0)
    sithick = np.where(siconc > 1.0, 1.5, 0.0)
    siu = np.zeros_like(siconc)
    siv = np.zeros_like(siconc)
    sitemptop = np.where(siconc > 1.0, 263.0, 273.15)
    for name, data in [
        ("siconc", siconc), ("sithick", sithick),
        ("siu", siu), ("siv", siv), ("sitemptop", sitemptop),
    ]:
        written.append(writer.write_field(
            name, data, time=15.0, time_bounds=(0.0, 30.0),
            lat=lat, lon=lon,
        ))

    writer.close()
    return written


def _validate(written):
    """Re-open each file and check CF-1.8 / CMIP6 essentials."""
    from legoesm.io.cmor_output import lookup_cmor_entry
    fail = []
    for path in written:
        ds = xr.open_dataset(path)
        try:
            # Each file contains one CMOR variable
            data_vars = [v for v in ds.data_vars
                          if v not in ("time_bnds", "lat_bnds",
                                       "lon_bnds")]
            assert len(data_vars) == 1, (
                f"{path.name}: expected 1 CMOR var, got {data_vars}"
            )
            v = data_vars[0]
            tbl, entry = lookup_cmor_entry(v)
            var = ds[v]
            # Required attrs
            for k in ("standard_name", "long_name", "units",
                      "cell_methods"):
                assert var.attrs.get(k) == entry[k], (
                    f"{path.name}: {k} mismatch — file={var.attrs.get(k)!r}, "
                    f"cmor={entry[k]!r}"
                )
            # Lat/lon present
            assert "lat" in ds.coords, f"{path.name}: missing lat"
            assert "lon" in ds.coords, f"{path.name}: missing lon"
            # Depth-bearing variables carry an ocean-flavored depth coord
            if "depth" in entry["dimensions"]:
                assert "depth" in ds.coords, f"{path.name}: missing depth"
                d = ds["depth"]
                assert d.attrs["positive"] == "down", (
                    f"{path.name}: depth positive != down"
                )
                if tbl == "Omon":
                    assert "Ocean Depth" in d.attrs["long_name"], (
                        f"{path.name}: ocean long_name missing"
                    )
            # Realm
            assert ds.attrs.get("realm") in ("ocean", "seaIce"), (
                f"{path.name}: bad realm {ds.attrs.get('realm')!r}"
            )
            # Finite data — physical realism
            assert np.all(np.isfinite(var.values)), (
                f"{path.name}: non-finite values"
            )
            print(f"  OK  {path.name:60s} "
                  f"[{v} | range {float(var.min()):.3g}..{float(var.max()):.3g}]")
        except AssertionError as e:
            fail.append(str(e))
            print(f"  FAIL  {path.name}: {e}")
        finally:
            ds.close()
    return fail


def main() -> int:
    print("=" * 78)
    print("OMIP-2 I/O Smoke Test")
    print("=" * 78)
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)

        print("\n[1/5] Synthesize WOA-style monthly NetCDF dataset")
        T_path, S_path = _write_synthetic_woa(tmp)
        print(f"  → {T_path.name}, {S_path.name}")

        print("\n[2/5] Build 5° lat-lon grid + 15-level z* coord")
        grid, z_coord, lat, lon = _build_grid_and_coord()
        depth = np.abs(np.asarray(z_coord.z_full_ref))
        print(f"  → grid {lat.size}x{lon.size}, {z_coord.n_levels} levels, "
              f"depth {depth.min():.1f}–{depth.max():.1f} m")

        print("\n[3/5] Init T, S from WOA (bilinear + January + bathymetry)")
        T, S, bath = _omip2_ic(grid, z_coord, T_path, S_path)
        print(f"  → T shape {T.shape}, range {T.min():.2f}..{T.max():.2f} degC")
        print(f"  → S shape {S.shape}, range {S.min():.2f}..{S.max():.2f} PSU")
        print(f"  → bathymetry range {bath.min():.0f}..{bath.max():.0f} m")
        # Realism check
        assert T.min() > -3.0 and T.max() < 35.0, "T out of physical range"
        assert S.min() > 30.0 and S.max() < 40.0, "S out of physical range"

        print("\n[4/5] Write Omon + SImon CMOR snapshot bundle")
        written = _write_omip2_outputs(tmp, T, S, lat, lon, depth)
        print(f"  → {len(written)} files written under {tmp/'cmor'}")

        print("\n[5/5] Validate each file (CF/CMIP6 essentials)")
        failures = _validate(written)

    print("\n" + "=" * 78)
    if failures:
        print(f"FAIL: {len(failures)} validation errors")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"PASS: {len(written)} files / {len(written)} CMOR variables valid")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
