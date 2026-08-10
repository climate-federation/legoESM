"""The production AMIP launcher must wire a real subgrid-orography file (#1514).

``config/amip/amip_production.yaml`` runs ``gravity_wave_drag: mcfarlane+hines``.
With ``subgrid_orography_path`` empty, the orographic member launches
``tau_0 ~ h_topo^2`` from the scalar fallback ``h_topo = 500 m`` on EVERY
column — a fictional 500-m mountain over the open ocean.  Measured on the
production mesh (issue #1514): -0.29 Pa column-integrated zonal drag over
40-60S (2x the entire observed surface stress), which removed the eddy-driven
westerly belt within a week of the ERA5 initial condition and suppressed the
storm tracks in both hemispheres.

Paths are deliberately machine-specific, so the wiring lives in the launcher
``config/amip/amip_production.sh`` (``AMIP_SSO`` -> ``--subgrid-orography-file``),
not the machine-independent YAML.  These tests source the launcher exactly as a
run would and drive the merged #1514 guard predicate with the RESOLVED
production configuration: drop the wiring and they go red.
"""
from __future__ import annotations

import os
import pathlib
import subprocess

import pytest
import yaml

_REPO = pathlib.Path(__file__).resolve().parents[2]
_LAUNCHER = _REPO / "config" / "amip" / "amip_production.sh"
_CONFIG = _REPO / "config" / "amip" / "amip_production.yaml"


def _resolved_path_flags() -> list[str]:
    """Source the launcher in a clean env and return AMIP_PATH_FLAGS.

    Clean env (no AMIP_* overrides inherited from the test session) so the
    committed defaults — the configuration a fresh operator runs — are what is
    tested, exactly as ``run_amip.py`` would receive them.
    """
    script = (
        f'source "{_LAUNCHER}" >/dev/null 2>&1; '
        'printf "%s\\0" "${AMIP_PATH_FLAGS[@]}"'
    )
    out = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        check=True,
        env={"PATH": os.defpath},
    )
    return [f for f in out.stdout.decode().split("\0") if f]


def _sso_flag_value(flags: list[str]) -> str | None:
    if "--subgrid-orography-file" not in flags:
        return None
    return flags[flags.index("--subgrid-orography-file") + 1]


def test_launcher_wires_subgrid_orography_file() -> None:
    flags = _resolved_path_flags()
    sso = _sso_flag_value(flags)
    assert sso, (
        "amip_production.sh no longer passes --subgrid-orography-file: the "
        "orographic GWD member (mcfarlane) will launch from the scalar "
        "h_topo=500 m fallback over every ocean column (#1514)."
    )
    assert sso.endswith(".nc")


def test_production_config_does_not_trip_the_1514_guard() -> None:
    """The resolved production (scheme, sso_path) must satisfy the #1514 guard.

    This is the functional link, not a read-back: the SAME predicate the driver
    logs from (PR #1540) is evaluated on the SAME values the launcher resolves.
    Reverting the launcher wiring (empty path) makes the guard return its loud
    warning and this test fail.
    """
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        orographic_scalar_fallback_warning,
    )

    gwd = str(yaml.safe_load(_CONFIG.read_text())["gravity_wave_drag"])
    sso = _sso_flag_value(_resolved_path_flags()) or ""
    msg = orographic_scalar_fallback_warning(gwd, sso, True)
    assert msg is None, (
        f"production config would run the #1514 pseudo-mountain fallback: {msg}"
    )
    # Non-vacuity control: the SAME scheme with the wiring reverted (empty
    # path) MUST trip the guard — otherwise this test could never fail.
    assert orographic_scalar_fallback_warning(gwd, "", True) is not None


def test_staged_sso_file_exists_and_is_sane() -> None:
    """On the deployment machine, the default SSO file must exist + be valid.

    Skipped off-machine (the launcher defaults are Levante paths; CI checkouts
    do not carry the 140 kB .nc — data/ is gitignored like every forcing file).
    """
    sso = _sso_flag_value(_resolved_path_flags())
    assert sso, "--subgrid-orography-file missing from AMIP_PATH_FLAGS"
    p = pathlib.Path(sso)
    if not p.parent.exists():
        pytest.skip(f"deployment data dir {p.parent} not present on this host")
    assert p.exists(), (
        f"launcher default SSO file {p} is wired but NOT staged — the run "
        "would die at setup (or worse, be launched with the flag removed)."
    )
    nc = pytest.importorskip("netCDF4")
    with nc.Dataset(p) as ds:
        assert "SSO_STDH" in ds.variables, list(ds.variables)
