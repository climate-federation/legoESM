"""CENTURY_DECK forcing selection in the AMIP MPAS chain launcher.

The block picks era-correct OZONE + SST/SIC for the link's simulated year.
It originally switched ONLY ozone and volcanic, leaving
``amip_production.sh``'s default ``bc_sst_1979_2016.nc`` in place — so a
``CENTURY_DECK=1`` run simulating 1923 got 1923 volcanic and 1923 ozone
bolted onto 1979-2016 SST.  Nothing in the run announced the mismatch; it
surfaced only because that ICON file also stores KELVIN under a
``units='degC'`` attribute and the SST unit validator rejected the offset.
Had the file been Celsius, the run would have completed with a silently
wrong ocean boundary condition.

Volcanic (review 2026-10-14, Levante stream, vote V2): the block no longer
swaps volcanic for one-year pool files; every link reads the continuous
1850-2014 product ``amip_production.sh`` names, as on Ginsburg.  Past 2014
the loader raises unless --forcing-hold-outside-file-years is set (tested on
the loader in test_external_forcing.py::...::test_inside_covered_outside_raises).

These execute the real chain (the block plus the ``source`` of
``amip_production.sh``, extracted from the .sbatch, run under bash) rather
than grepping it, so a regression in the SELECTION LOGIC — not just the
text — goes red.  The block's Levante pool roots are redirected to a temp
tree holding every file it can select, so its existence guards run here too.
"""

import re
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SBATCH = _REPO / "scripts/cluster/levante/amip_mpas_gpu_chain.sbatch"
_SOURCE = 'source "${REPO}/config/amip/amip_production.sh"\n'


def _chain() -> str:
    """The CENTURY_DECK block through the ``source`` of amip_production.sh."""
    text = _SBATCH.read_text()
    start = text.index('if [[ "${CENTURY_DECK:-0}" == "1" ]]')
    return text[start:text.index(_SOURCE, start) + len(_SOURCE)]


def _fake_pool(root: Path) -> str:
    """The chain with its pool roots under ``root``, every selectable file present."""
    text = _chain().replace("=/pool/data/", f"={root}/pool/data/")
    iconp = re.search(r"_ICONP=(\S+)", text)[1]
    ozdir = iconp + re.search(r'_OZDIR="\$\{_ICONP\}([^"]+)"', text)[1]
    pcmdi = re.search(r"_PCMDI=(\S+)", text)[1]
    ver = re.search(r"_V=(\S+)", text)[1]
    files = [f"{ozdir}/{rel}" for rel in re.findall(r'_OZF="(historical/[^"]+)"', text)]
    files += [pcmdi + rel.replace("${_V}", ver)
              for rel in re.findall(r"\$\{_PCMDI\}(\S+?\.nc)", text)]
    assert len(files) == 6, files            # 4 ozone chunks + SST + SIC
    for f in map(Path, files):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
    return text


def _run(sim_day: int, start_year: int, outdir: Path, **env):
    """Execute the chain for a given checkpoint day; return the process."""
    if sim_day:
        (outdir / f"checkpoint_day_{sim_day:04d}.npz").touch()
    script = "\n".join([
        "set -uo pipefail",
        f'REPO="{_REPO}"',
        f'OUTDIR="{outdir}"',
        f'START_YEAR={start_year}',
        "CENTURY_DECK=1",
        *(f'{k}="{v}"' for k, v in env.items()),
        _fake_pool(outdir / "fakeroot"),
        'echo "RESULT VOLCANIC=${VOLCANIC:-} OZONE=${OZONE:-} '
        'SST=${SST:-} SIC=${SIC:-}"',
        '(IFS=,; echo "FLAGS=${AMIP_PATH_FLAGS[*]}")',
    ])
    return subprocess.run(["bash", "-c", script], env={"PATH": "/usr/bin:/bin"},
                          capture_output=True, text=True)


def _parsed(p):
    m = re.search(r"RESULT VOLCANIC=(\S*) OZONE=(\S*) SST=(\S*) SIC=(\S*)\nFLAGS=(\S*)",
                  p.stdout)
    assert m, f"chain did not reach the echo:\nout={p.stdout}\nerr={p.stderr}"
    return dict(zip(("volcanic", "ozone", "sst", "sic", "flags"), m.groups()))


def _production_volcanic() -> str:
    """VOLCANIC as amip_production.sh resolves it with no CENTURY_DECK."""
    return subprocess.run(
        ["bash", "-c", f'source "{_REPO}/config/amip/amip_production.sh" >/dev/null; '
                       'echo "${VOLCANIC}"'],
        env={"PATH": "/usr/bin:/bin"}, capture_output=True, text=True,
        check=True).stdout.strip()


pytestmark = pytest.mark.skipif(
    not _SBATCH.is_file(), reason="chain launcher not present")


class TestSSTSelection:
    """THE REGRESSION: era-correct ozone must not be paired with the
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

    @pytest.mark.parametrize("day,year", [
        (0, 1923),
        (365 * 40, 1923),        # 1963, Agung
        (365 * 68, 1923),        # 1991, Pinatubo
    ])
    def test_volcanic_is_the_continuous_production_file(self, day, year, tmp_path):
        """Every link reads the one 1850-2014 product amip_production.sh
        names, never a per-year pool file (the old per-link swap)."""
        got = _parsed(_run(day, year, tmp_path))
        assert got["volcanic"] == _production_volcanic(), got["volcanic"]
        span = re.search(r"_(\d{4})-(\d{4})_", Path(got["volcanic"]).name)
        assert span, got["volcanic"]
        assert int(span[1]) <= year + day // 365 <= int(span[2]), got["volcanic"]

    def test_past_2014_volcanic_is_not_clamped_or_held(self, tmp_path):
        """After 2014 the chain still names the 1850-2014 file (no clamp to a
        2014 pool file) and neither the sourced flags nor the chain's launch
        opt in to holding its end record, so the loader raises instead of
        silently holding 2014 (EXTRA remains the user's explicit opt-in)."""
        got = _parsed(_run(365 * 95, 1923, tmp_path))   # 2018
        assert got["volcanic"] == _production_volcanic(), got["volcanic"]
        span = re.search(r"_(\d{4})-(\d{4})_", Path(got["volcanic"]).name)
        assert span and int(span[1]) <= 1923 and int(span[2]) < 2018, got["volcanic"]
        flags = got["flags"].split(",")
        assert "--volcanic-aerosol-file" in flags, flags      # the echo saw the array
        assert "--forcing-hold-outside-file-years" not in flags
        code = "\n".join(line.split("#")[0] for line in _SBATCH.read_text().splitlines())
        assert "--forcing-hold-outside-file-years" not in code

    def test_day_comes_from_the_newest_checkpoint(self, tmp_path):
        """Multiple checkpoints: the LATEST one sets the simulated year, so a
        link resuming mid-century does not reload the 1923 deck.  Names use
        the driver's {:04d}, so widths mix and a plain lexical sort picks 3650."""
        for d in (100, 20000, 3650):
            (tmp_path / f"checkpoint_day_{d:04d}.npz").touch()
        got = _parsed(_run(0, 1923, tmp_path))           # 1923 + 20000/365 = 1977
        assert "195001-199912" in got["ozone"], got["ozone"]


def test_missing_forcing_file_fails_loudly(tmp_path):
    """A path that does not exist must abort the link, never run with a
    silently absent forcing channel."""
    p = _run(0, 1923, tmp_path, SST="/nonexistent/sst.nc")
    assert p.returncode == 3, (p.returncode, p.stdout, p.stderr)
    assert "missing SST file" in p.stderr
