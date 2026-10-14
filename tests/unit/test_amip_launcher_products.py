"""The Levante and Ginsburg AMIP launchers feed the production deck the SAME
input products (review 2026-10-10 F33, majority vote).

They used to differ in radiative aerosol (fine only vs fine+coarse), the
droplet-number aerosol, the land-sea mask, the sub-grid orography (pre/post
seam fix) and the ERA5 initial state, so one deck gave a different model on
each machine and nothing recorded it.  Directories are machine-specific; the
file each flag names must be the same product (same basename).
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
LAUNCHERS = ("config/amip/amip_production.sh",
             "config/amip/amip_production.ginsburg.sh")
SAME_PRODUCT = ("--aerosol-file", "--aerosol-ccn-file", "--land-mask-file",
                "--subgrid-orography-file", "--ic-path")


def _path_flags(launcher: str, array: str = "AMIP_PATH_FLAGS",
                basename: bool = True) -> dict[str, str]:
    """flag -> path (basename by default) the sourced launcher passes in ``array``."""
    out = subprocess.run(
        ["bash", "-c", f'source "{REPO / launcher}" >/dev/null 2>&1; '
                       f'printf "%s\\n" "${{{array}[@]}}"'],
        env={"PATH": "/usr/bin:/bin"},      # no inherited overrides
        capture_output=True, text=True, check=True, cwd=REPO).stdout.split("\n")
    return {tok: Path(nxt).name if basename else nxt for tok, nxt in zip(out, out[1:])
            if tok.startswith("--") and nxt and not nxt.startswith("--")}


@pytest.mark.parametrize("flag", SAME_PRODUCT)
def test_both_launchers_name_the_same_product(flag):
    levante, ginsburg = (_path_flags(name) for name in LAUNCHERS)
    assert flag in levante and flag in ginsburg, (flag, levante, ginsburg)
    assert levante[flag] == ginsburg[flag], (flag, levante[flag], ginsburg[flag])


# Review 2026-10-10 F37: the coupled decks start in 1850, and a multi-year file
# that does not cover the run's years raises at the first forcing update.  Each
# dated forcing product the Ginsburg launcher hands the coupled lane must cover
# the decks' run years: read from the file's CF time axis where it is staged
# (Ginsburg), else from the year span in its basename.


def _file_years(path: str) -> tuple[int, int]:
    netCDF4 = pytest.importorskip("netCDF4")
    cftime = pytest.importorskip("cftime")
    with netCDF4.Dataset(path) as nc:
        t = nc["time"]
        v, units = [float(t[0]), float(t[-1])], t.units
        if units.startswith("year as"):                 # GHG: fractional year
            return int(v[0]), int(v[1])
        if units.startswith("months since"):            # ozone: standard calendar
            y0, m0 = (int(x) for x in units.split()[2].split("-")[:2])
            return tuple(y0 + (m0 - 1 + int(x)) // 12 for x in v)
        ends = cftime.num2date(v, units, getattr(t, "calendar", "standard"))
    return ends[0].year, ends[1].year


@pytest.mark.parametrize("flag", (
    "--ozone-file", "--solar-file", "--ghg-file", "--volcanic-aerosol-file",
    "--aerosol-file", "--aerosol-ccn-file"))
def test_coupled_forcing_products_cover_the_decks_run_years(flag):
    path = _path_flags("config/amip/amip_production.ginsburg.sh",
                       "CMIP_PATH_FLAGS", basename=False)[flag]
    if Path(path).exists():
        lo, hi = _file_years(path)
    else:
        span = re.search(r"_(\d{4})(?:\d{2})?-(\d{4})(?:\d{2})?[_.]", Path(path).name)
        if span is None:
            pytest.skip(f"{path} is not staged here and its name carries no years")
        lo, hi = int(span[1]), int(span[2])
    for deck in ("cmip_ocean_slab.yaml", "cmip_ocean_3D.yaml"):
        cfg = yaml.safe_load((REPO / "config" / "cmip" / deck).read_text())
        first, last = cfg["start_year"], cfg["start_year"] + (cfg["days"] - 1) // 365
        assert lo <= first and last <= hi, (flag, path, (lo, hi), deck, (first, last))


def test_levante_topography_is_finer_than_the_production_mesh():
    """The terrain product bins the topography file onto model cells and
    refuses any cell that receives no source point; the 1 deg file left 2193
    MPAS res-6 cells empty, so every production run crashed at setup.  The
    Levante launcher's file must be at least 4x finer than the res-6 cell
    spacing (~120 km, ~1.08 deg)."""
    netCDF4 = pytest.importorskip("netCDF4")
    path = _path_flags("config/amip/amip_production.sh",
                       basename=False)["--topography"]
    if not Path(path).exists():
        pytest.skip(f"{path} is not staged on this machine")
    with netCDF4.Dataset(path) as nc:
        lat = [nc[k][:] for k in nc.variables if k.startswith("lat")][0]
        lon = [nc[k][:] for k in nc.variables if k.startswith("lon")][0]
    dlat, dlon = float(abs(lat[1] - lat[0])), float(abs(lon[1] - lon[0]))
    assert max(dlat, dlon) <= 0.27, (path, dlat, dlon)
