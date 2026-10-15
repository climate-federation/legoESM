"""amip_sota at C48 on Levante: the deck's vertical coordinate covers C48
terrain, and its paths script supplies the start file the shared Levante
paths lack (user decisions 2026-10-15)."""
from __future__ import annotations

import os
import pathlib
import subprocess

from legoesm.driver.run_config_yaml import load_yaml_config

_REPO = pathlib.Path(__file__).resolve().parents[2]
_AMIP = _REPO / "config" / "amip"


def _flags(launcher: pathlib.Path) -> list[str]:
    out = subprocess.run(
        ["bash", "-c", f'source "{launcher}" >/dev/null 2>&1; '
                       'printf "%s\\0" "${AMIP_PATH_FLAGS[@]}"'],
        capture_output=True, check=True, env={"PATH": os.defpath})
    return [f for f in out.stdout.decode().split("\0") if f]


def _value(flags, name):
    return flags[flags.index(name) + 1]


def test_deck_uses_transition_exponent_2():
    from scripts.run import run_amip
    parser = run_amip.build_arg_parser()
    assert load_yaml_config(str(_AMIP / "amip_sota.yaml"),
                            parser)["transition_exponent"] == 2


def test_sota_paths_pick_the_37_level_start_file_and_shared_terrain():
    sota, shared = _flags(_AMIP / "amip_sota.sh"), _flags(_AMIP / "amip_production.sh")
    assert _value(sota, "--ic-path") == "/scratch/b/b309178/era5_ic_1979-01-01.zarr"
    assert _value(shared, "--ic-path") != _value(sota, "--ic-path")
    assert (_value(sota, "--topography") == _value(shared, "--topography")
            and _value(sota, "--topography").endswith("etopo2022_60s_0p25deg.nc"))
