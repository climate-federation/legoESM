"""The Levante and Ginsburg AMIP launchers feed the production deck the SAME
input products (review 2026-10-10 F33, majority vote).

They used to differ in radiative aerosol (fine only vs fine+coarse), the
droplet-number aerosol, the land-sea mask, the sub-grid orography (pre/post
seam fix) and the ERA5 initial state, so one deck gave a different model on
each machine and nothing recorded it.  Directories are machine-specific; the
file each flag names must be the same product (same basename).
"""
from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
LAUNCHERS = ("config/amip/amip_production.sh",
             "config/amip/amip_production.ginsburg.sh")
SAME_PRODUCT = ("--aerosol-file", "--aerosol-ccn-file", "--land-mask-file",
                "--subgrid-orography-file", "--ic-path", "--volcanic-aerosol-file")
# Forcing channels every Levante entry point must pass as the Ginsburg twin does.
FORCING_PRODUCTS = ("--aerosol-file", "--aerosol-ccn-file", "--volcanic-aerosol-file")


def _tokens(launcher: str, array: str = "AMIP_PATH_FLAGS") -> list[str]:
    """The tokens of ``array`` in the sourced launcher."""
    return subprocess.run(
        ["bash", "-c", f'source "{REPO / launcher}" >/dev/null 2>&1; '
                       f'printf "%s\\n" "${{{array}[@]}}"'],
        env={"PATH": "/usr/bin:/bin"},      # no inherited overrides
        capture_output=True, text=True, check=True, cwd=REPO).stdout.split("\n")


def _flag_values(out: list[str], basename: bool = True) -> dict[str, str]:
    return {tok: Path(nxt).name if basename else nxt for tok, nxt in zip(out, out[1:])
            if tok.startswith("--") and nxt and not nxt.startswith("--")}


def _path_flags(launcher: str, array: str = "AMIP_PATH_FLAGS",
                basename: bool = True) -> dict[str, str]:
    """flag -> path (basename by default) the sourced launcher passes in ``array``."""
    return _flag_values(_tokens(launcher, array), basename)


@pytest.mark.parametrize("flag", SAME_PRODUCT)
def test_both_launchers_name_the_same_product(flag):
    levante, ginsburg = (_path_flags(name) for name in LAUNCHERS)
    assert flag in levante and flag in ginsburg, (flag, levante, ginsburg)
    assert levante[flag] == ginsburg[flag], (flag, levante[flag], ginsburg[flag])


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_launcher_passes_volcanic_longwave(launcher):
    assert "--volcanic-aerosol-lw" in _tokens(launcher), launcher


def test_levante_python_launcher_names_the_production_products(monkeypatch, capsys):
    """scripts/run/run_amip_levante.py used to pass a one-year Kinne fine-mode
    file, no droplet-number (CCN) file and a one-year volcanic file without
    the longwave channel (review 2026-10-14, Levante stream)."""
    spec = importlib.util.spec_from_file_location(
        "run_amip_levante", REPO / "scripts" / "run" / "run_amip_levante.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_check_files", lambda files: [])   # Levante paths
    assert mod.main(["--year", "1979", "--days", "730", "--dry-run"]) == 0
    printed = capsys.readouterr().out.split("[levante] Command:")[1]
    cmd = [tok for tok in printed.split() if tok != "\\"]       # line continuations
    assert Path(cmd[1]).is_file(), cmd[1]                # the driver it launches exists
    got = _flag_values(cmd, basename=False)              # same machine: full paths
    want = _path_flags("config/amip/amip_production.sh", basename=False)
    for flag in FORCING_PRODUCTS:
        assert got.get(flag) == want[flag], (flag, got.get(flag), want[flag])
    assert "--volcanic-aerosol-lw" in cmd
    # Production deck sets no reference AOD, so it runs run_amip.py's
    # default 0.03; this launcher passed 0.05 (user decision 3a, 2026-10-14).
    assert cmd[cmd.index("--aerosol-reference-aod") + 1] == "0.03"


def test_levante_python_launcher_forwards_distributed(monkeypatch, capsys):
    """run_amip federates the SPMD lane only when --distributed is on ITS
    command line (user decision 2026-10-28), so the wrapper's --distributed
    must reach it."""
    spec = importlib.util.spec_from_file_location(
        "run_amip_levante", REPO / "scripts" / "run" / "run_amip_levante.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_check_files", lambda files: [])   # Levante paths
    assert mod.main(["--year", "1979", "--days", "1", "--distributed",
                     "--n-ranks", "2", "--dry-run",
                     "--extra", "--distributed-mode", "spmd"]) == 0
    printed = capsys.readouterr().out.split("[levante] Command:")[1]
    cmd = [tok for tok in printed.split() if tok != "\\"]
    assert cmd[0] == "srun"
    assert "--distributed" in cmd[cmd.index(next(t for t in cmd if t.endswith("run_amip.py"))):]


def test_levante_chain_does_not_swap_one_forcing_channel():
    """The GPU chain (every Levante submit_*.sh) must not re-point one of the
    aerosol / CCN / volcanic products the sourced launcher names (its
    CENTURY_DECK branch used to swap volcanic alone for one-year pool files),
    and must launch with the path flags."""
    text = (REPO / "scripts/cluster/levante/amip_mpas_gpu_chain.sbatch").read_text()
    code = "\n".join(line.split("#")[0] for line in text.splitlines())
    assert not re.search(r"\b(?:AEROSOL|AEROSOL_CCN|VOLCANIC)\s*:?=", code)
    assert '"${AMIP_PATH_FLAGS[@]}"' in code, "the launch no longer passes the path flags"


# Every sbatch that sources amip_production.sh to launch run_amip.py must call
# the shared refusal between the source line and the launch (review 2026-10-14).
CHECKED_LAUNCHERS = ("scripts/cluster/levante/amip_mpas_gpu_chain.sbatch",
                     "scripts/cluster/mpas_energy_1354/energy_budget_15d.sbatch",
                     "scripts/cluster/mpas_energy_1354/energy_budget_smoke.sbatch")


@pytest.mark.parametrize("launcher", CHECKED_LAUNCHERS)
def test_launcher_refuses_before_launch_on_a_missing_forcing_file(launcher):
    lines = [line.split("#")[0].strip()
             for line in (REPO / launcher).read_text().splitlines()]
    src = next(i for i, l in enumerate(lines)
               if l.startswith("source") and "amip_production.sh" in l)
    launch = next(i for i, l in enumerate(lines) if '"${AMIP_PATH_FLAGS[@]}"' in l)
    assert "amip_require_forcing_files || exit 3" in lines[src + 1:launch], launcher


@pytest.mark.parametrize("missing", ("AEROSOL", "AEROSOL_CCN", "VOLCANIC", None))
def test_forcing_file_check_names_each_missing_product(tmp_path, missing):
    env = {"PATH": "/usr/bin:/bin"}
    for var in ("AEROSOL", "AEROSOL_CCN", "VOLCANIC"):
        env[var] = str(tmp_path / f"{var}.nc")
        if var != missing:
            (tmp_path / f"{var}.nc").touch()
    p = subprocess.run(
        ["bash", "-c", f'source "{REPO / LAUNCHERS[0]}" >/dev/null; amip_require_forcing_files'],
        env=env, capture_output=True, text=True, cwd=REPO)
    if missing is None:
        assert p.returncode == 0, p.stderr
    else:
        assert p.returncode == 3 and f"{missing}.nc" in p.stderr, (p.returncode, p.stderr)


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
