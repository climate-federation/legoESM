"""The Levante and Ginsburg AMIP launchers feed the production deck the SAME
input products (review 2026-10-10 F33, majority vote).

They used to differ in radiative aerosol (fine only vs fine+coarse), the
droplet-number aerosol, the land-sea mask, the sub-grid orography (pre/post
seam fix) and the ERA5 initial state, so one deck gave a different model on
each machine and nothing recorded it.  Directories are machine-specific; the
file each flag names must be the same product (same basename).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
LAUNCHERS = ("config/amip/amip_production.sh",
             "config/amip/amip_production.ginsburg.sh")
SAME_PRODUCT = ("--aerosol-file", "--aerosol-ccn-file", "--land-mask-file",
                "--subgrid-orography-file", "--ic-path")


def _path_flags(launcher: str) -> dict[str, str]:
    """flag -> basename of the path the sourced launcher passes."""
    out = subprocess.run(
        ["bash", "-c", f'source "{REPO / launcher}" >/dev/null 2>&1; '
                       'printf "%s\\n" "${AMIP_PATH_FLAGS[@]}"'],
        env={"PATH": "/usr/bin:/bin"},      # no inherited overrides
        capture_output=True, text=True, check=True, cwd=REPO).stdout.split("\n")
    return {tok: Path(nxt).name for tok, nxt in zip(out, out[1:])
            if tok.startswith("--") and nxt and not nxt.startswith("--")}


@pytest.mark.parametrize("flag", SAME_PRODUCT)
def test_both_launchers_name_the_same_product(flag):
    levante, ginsburg = (_path_flags(name) for name in LAUNCHERS)
    assert flag in levante and flag in ginsburg, (flag, levante, ginsburg)
    assert levante[flag] == ginsburg[flag], (flag, levante[flag], ginsburg[flag])
