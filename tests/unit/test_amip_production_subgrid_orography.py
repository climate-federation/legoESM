"""The production AMIP launcher must wire a real subgrid-orography file (#1514).

``config/amip/amip_production.yaml`` runs an OROGRAPHIC gravity-wave member
(``mcfarlane``, alone since the CAM6 suite became production on 2026-09-23;
the tests below read the shipped value rather than assuming one).
With ``subgrid_orography_path`` empty, the orographic member launches
``tau_0 ~ h_topo^2`` from the scalar fallback ``h_topo = 500 m`` on EVERY
column — a fictional 500-m mountain over the open ocean.  Measured on the
production mesh (issue #1514): -0.29 Pa column-integrated zonal drag over
40-60S (2x the entire observed surface stress), which removed the eddy-driven
westerly belt within a week of the ERA5 initial condition and suppressed the
storm tracks in both hemispheres.

Paths are deliberately machine-specific, so the wiring lives in the launchers
(``AMIP_SSO`` -> ``--subgrid-orography-file``), not the machine-independent
YAML.  These tests source each launcher exactly as a run would and drive the
merged #1514 guard predicate with the RESOLVED production configuration: drop
the wiring and they go red.

EVERY committed AMIP launcher is covered, not just one.  The gate originally
checked only the Levante script while the Ginsburg twin silently omitted the
flag, so every AMIP run started on that machine reproduced the pseudo-mountain
climate the guard exists to prevent -- a machine-specific hole in a
machine-independent defect.  New launcher scripts are picked up automatically
by the glob below; there is nothing to remember to add.
"""
from __future__ import annotations

import os
import pathlib
import subprocess

import pytest
import yaml

_REPO = pathlib.Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "config" / "amip" / "amip_production.yaml"
# Every committed launcher for the production AMIP config.  Globbed, not
# listed, so a launcher added later cannot quietly escape the gate.
_LAUNCHERS = sorted((_REPO / "config" / "amip").glob("amip_production*.sh"))

# The subgrid-orography field is only defined once you fix the SCALE
# DECOMPOSITION: `block_deg` sets the upper cutoff of the variance it captures
# and `fine_res_deg` the lower.  Measured 2026-09-04 on one fixed source, the
# block size swings the Southern-Ocean launch stress ~33x across plausible
# choices (0.5/1/2/4 deg -> 0.06/0.31/1.00/1.99 relative drag; tau_0 ~ sgh^2)
# and the source choice ~2.3x -- so two machines running different
# constructions are running materially different drag, which is exactly what
# happened until this was pinned.  A file with no `history` cannot be checked
# at all, so that is a failure too.
_EXPECTED_CONSTRUCTION = "--fine-res-deg 1.0 --block-deg 2.0"


def _launcher_ids() -> list[str]:
    return [p.name for p in _LAUNCHERS]


def test_every_production_launcher_is_covered() -> None:
    """Non-vacuity for the glob: at least the two known launchers are found."""
    names = _launcher_ids()
    assert "amip_production.sh" in names, names
    assert "amip_production.ginsburg.sh" in names, names


def _resolved_path_flags(_LAUNCHER) -> list[str]:
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


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=_launcher_ids())
def test_launcher_wires_subgrid_orography_file(launcher) -> None:
    flags = _resolved_path_flags(launcher)
    sso = _sso_flag_value(flags)
    assert sso, (
        f"{launcher.name} does not pass --subgrid-orography-file: the "
        "orographic GWD member (mcfarlane) will launch from the scalar "
        "h_topo=500 m fallback over every ocean column (#1514)."
    )
    assert sso.endswith(".nc")


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=_launcher_ids())
def test_production_config_does_not_trip_the_1514_guard(launcher) -> None:
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
    sso = _sso_flag_value(_resolved_path_flags(launcher)) or ""
    msg = orographic_scalar_fallback_warning(gwd, sso, True)
    assert msg is None, (
        f"production config would run the #1514 pseudo-mountain fallback: {msg}"
    )
    # Non-vacuity control: the SAME scheme with the wiring reverted (empty
    # path) MUST trip the guard — otherwise this test could never fail.
    assert orographic_scalar_fallback_warning(gwd, "", True) is not None


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=_launcher_ids())
def test_staged_sso_file_records_the_expected_construction(launcher) -> None:
    """Every machine must build its SSO field the SAME way (#1514).

    The gate used to pin only that SOME `.nc` carrying `SSO_STDH` was wired,
    which both launchers satisfied while building the field from different
    source resolutions -- a 2.3x drag difference between machines, silently.
    Pin the construction itself, from the file's own `history` attribute.
    """
    sso = _sso_flag_value(_resolved_path_flags(launcher))
    assert sso, "--subgrid-orography-file missing from AMIP_PATH_FLAGS"
    p = pathlib.Path(sso)
    if not p.parent.exists():
        pytest.skip(f"deployment data dir {p.parent} not present on this host")
    assert p.exists(), f"{p} is wired but NOT staged"
    nc = pytest.importorskip("netCDF4")
    with nc.Dataset(p) as ds:
        history = str(getattr(ds, "history", ""))
    assert history, (
        f"{p} records no `history` attribute, so its scale decomposition "
        "cannot be checked at all — regenerate it with "
        "scripts/data/prep_subgrid_orography.py, which stamps one.")
    assert _EXPECTED_CONSTRUCTION in history, (
        f"{launcher.name} wires an SSO field built as {history!r}, not "
        f"{_EXPECTED_CONSTRUCTION!r}. Block size and source resolution ARE the "
        "scale decomposition (block size alone swings the drag ~33x), so a "
        "machine building it differently is running different physics.")


@pytest.mark.parametrize("launcher", _LAUNCHERS, ids=_launcher_ids())
def test_staged_sso_file_exists_and_is_sane(launcher) -> None:
    """On the deployment machine, the default SSO file must exist + be valid.

    Skipped off-machine (each launcher's default is a path on ITS OWN machine;
    CI checkouts do not carry the 140 kB .nc — data/ is gitignored like every
    forcing file).  So on any given host exactly one launcher's file is
    normally checkable, and the other skips.
    """
    sso = _sso_flag_value(_resolved_path_flags(launcher))
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
