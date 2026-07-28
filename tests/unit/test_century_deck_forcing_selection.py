"""CENTURY_DECK forcing selection in the AMIP MPAS chain launcher.

The block picks era-correct OZONE + VOLCANIC + SST/SIC for the link's
simulated year.  It originally switched ONLY ozone and volcanic, leaving
``amip_production.sh``'s default ``bc_sst_1979_2016.nc`` in place — so a
``CENTURY_DECK=1`` run simulating 1923 got 1923 volcanic and 1923 ozone
bolted onto 1979-2016 SST.  Nothing in the run announced the mismatch; it
surfaced only because that ICON file also stores KELVIN under a
``units='degC'`` attribute and the SST unit validator rejected the offset.
Had the file been Celsius, the run would have completed with a silently
wrong ocean boundary condition.

These execute the real block (extracted from the .sbatch, run under bash)
rather than grepping it, so a regression in the SELECTION LOGIC — not just
the text — goes red.
"""

import re
import subprocess
from pathlib import Path

import pytest

_SBATCH = (Path(__file__).resolve().parents[2]
           / "scripts/cluster/levante/amip_mpas_gpu_chain.sbatch")


def _block() -> str:
    """The `if [[ "${CENTURY_DECK:-0}" == "1" ]]; then ... fi` block."""
    text = _SBATCH.read_text()
    start = text.index('if [[ "${CENTURY_DECK:-0}" == "1" ]]')
    end = text.index("\nfi\n", start) + len("\nfi\n")
    return text[start:end]


def _run(sim_day: int, start_year: int, outdir: Path, **env):
    """Execute the block for a given checkpoint day; return its env."""
    if sim_day:
        (outdir / f"checkpoint_day_{sim_day:04d}.npz").touch()
    script = "\n".join([
        "set -uo pipefail",
        f'OUTDIR="{outdir}"',
        f'START_YEAR={start_year}',
        "CENTURY_DECK=1",
        *(f'{k}="{v}"' for k, v in env.items()),
        _block(),
        'echo "RESULT VOLCANIC=${VOLCANIC:-} OZONE=${OZONE:-} '
        'SST=${SST:-} SIC=${SIC:-}"',
    ])
    p = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    return p


def _parsed(p):
    m = re.search(r"RESULT VOLCANIC=(\S*) OZONE=(\S*) SST=(\S*) SIC=(\S*)",
                  p.stdout)
    assert m, f"block did not reach the echo:\nout={p.stdout}\nerr={p.stderr}"
    return dict(zip(("volcanic", "ozone", "sst", "sic"), m.groups()))


pytestmark = pytest.mark.skipif(
    not _SBATCH.is_file(), reason="chain launcher not present")


class TestSSTSelection:
    """THE REGRESSION: era-correct ozone/volcanic must not be paired with the
    default 1979-2016 SST."""

    def test_sst_and_sic_are_set(self, tmp_path):
        got = _parsed(_run(0, 1923, tmp_path))
        assert got["sst"], "CENTURY_DECK left SST unset -> inherits the ICON default"
        assert got["sic"], "CENTURY_DECK left SIC unset"

    def test_sst_is_not_the_1979_default(self, tmp_path):
        got = _parsed(_run(0, 1923, tmp_path))
        assert "1979_2016" not in got["sst"], (
            "century deck paired with the 1979-2016 ICON SST — that file does "
            "not cover the simulated era AND stores Kelvin under units=degC")

    def test_sst_span_covers_the_start_year(self, tmp_path):
        got = _parsed(_run(0, 1923, tmp_path))
        m = re.search(r"_(\d{4})\d{2}-(\d{4})\d{2}\.nc$", got["sst"])
        assert m, f"cannot read the time span from {got['sst']}"
        assert int(m.group(1)) <= 1923 <= int(m.group(2))

    def test_env_override_still_wins(self, tmp_path):
        """Every other path here is env-overridable; SST must stay so.  The
        override must point at a REAL file — the block's existence guard is
        deliberately applied to the overridden value too."""
        mine = tmp_path / "mine.nc"
        mine.touch()
        got = _parsed(_run(0, 1923, tmp_path, SST=str(mine)))
        assert got["sst"] == str(mine)


class TestEraSelection:
    @pytest.mark.parametrize("day,year,expect", [
        (0, 1923, "190001-194912"),
        (365 * 30, 1923, "195001-199912"),   # 1953
        (365 * 80, 1923, "200001-201412"),   # 2003
        (0, 1880, "185001-189912"),
    ])
    def test_ozone_chunk_matches_the_simulated_year(self, day, year, expect,
                                                    tmp_path):
        got = _parsed(_run(day, year, tmp_path))
        assert expect in got["ozone"], got["ozone"]

    @pytest.mark.parametrize("day,year,expect", [
        (0, 1923, "_1923.nc"),
        (365 * 40, 1923, "_1963.nc"),        # Agung
        (365 * 68, 1923, "_1991.nc"),        # Pinatubo
    ])
    def test_volcanic_file_matches_the_simulated_year(self, day, year, expect,
                                                      tmp_path):
        got = _parsed(_run(day, year, tmp_path))
        assert got["volcanic"].endswith(expect), got["volcanic"]

    def test_volcanic_clamps_above_the_available_range(self, tmp_path):
        """The CMIP6 volcanic deck stops at 2014; later years hold 2014."""
        got = _parsed(_run(365 * 95, 1923, tmp_path))   # 2018
        assert got["volcanic"].endswith("_2014.nc"), got["volcanic"]

    def test_day_comes_from_the_newest_checkpoint(self, tmp_path):
        """Multiple checkpoints: the LATEST one sets the simulated year, so a
        link resuming mid-century does not reload the 1923 deck."""
        for d in (100, 20000, 3650):
            (tmp_path / f"checkpoint_day_{d:05d}.npz").touch()
        got = _parsed(_run(0, 1923, tmp_path))           # 1923 + 20000/365
        assert got["volcanic"].endswith("_1977.nc"), got["volcanic"]


def test_missing_forcing_file_fails_loudly(tmp_path):
    """A path that does not exist must abort the link, never run with a
    silently absent forcing channel."""
    p = _run(0, 1923, tmp_path, SST="/nonexistent/sst.nc")
    assert p.returncode == 3, (p.returncode, p.stdout, p.stderr)
    assert "missing SST file" in p.stderr
