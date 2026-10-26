"""Every committed AMIP launcher must stage the canopy parameter file.

The default land surface scheme is the two-leaf canopy, and the driver refuses
to run it without the harmonized surface-data file rather than fall back to
generic per-plant-type constants.  That file is NOT the CLM surface data the
launchers already stage: the CLM one carries soil texture and plant-type cover,
the harmonized one carries canopy structure (canopy height, roughness ratio,
photosynthetic capacity, band albedos).  The two names differ by one word.

When the default was flipped to the canopy, one of the two committed launchers
was updated and the other was not, so a production run on that machine would
have stopped at setup.  These tests source each launcher exactly as a run does
and check the resolved flags, so the next launcher that forgets goes red.
"""
from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_LAUNCHERS = sorted((_REPO / "config" / "amip").glob("amip_production*.sh"))


def _resolved_path_flags(launcher: pathlib.Path) -> list[str]:
    script = (
        f'source "{launcher}" >/dev/null 2>&1; '
        'printf "%s\\0" "${AMIP_PATH_FLAGS[@]}"'
    )
    out = subprocess.run(
        ["bash", "-c", script], capture_output=True, check=True,
        env={"PATH": os.defpath, "REPO": str(_REPO)},
    )
    return [f for f in out.stdout.decode().split("\0") if f]


def test_there_are_launchers_to_check() -> None:
    # A glob that matches nothing would make every test below vacuous.
    assert _LAUNCHERS, "no committed AMIP launcher found to check"


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=lambda p: p.name)
def test_launcher_stages_the_canopy_surfdata(launcher: pathlib.Path) -> None:
    flags = _resolved_path_flags(launcher)
    assert "--surfdata" in flags, (
        f"{launcher.name} does not pass --surfdata. The default land surface "
        f"scheme is the two-leaf canopy, which refuses to start without the "
        f"harmonized surface data, so every run from this launcher stops at "
        f"setup. Staging the CLM surface data is not the same thing."
    )
    value = flags[flags.index("--surfdata") + 1]
    assert value.endswith(".nc"), (
        f"{launcher.name} passes --surfdata {value!r}, which is not a NetCDF "
        f"path")
    assert "clm" not in pathlib.Path(value).name.lower(), (
        f"{launcher.name} points --surfdata at {pathlib.Path(value).name}, "
        f"which looks like the CLM surface data. That file carries soil "
        f"texture and plant-type cover, not canopy structure; the canopy "
        f"parameters would be missing and every tuned per-plant-type value "
        f"inert.")


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=lambda p: p.name)
def test_launcher_still_stages_the_clm_surfdata(launcher: pathlib.Path) -> None:
    """Both files are needed; adding one must not have replaced the other."""
    flags = _resolved_path_flags(launcher)
    assert "--clm-surfdata-path" in flags, (
        f"{launcher.name} no longer passes --clm-surfdata-path: the soil "
        f"texture and plant-type cover the multilayer land needs would be "
        f"downloaded at run time, which fails on a compute node.")
