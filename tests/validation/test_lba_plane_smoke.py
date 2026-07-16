"""Smoke test for the LBA land-diurnal plane run (``scripts/run/run_lba_plane.py``).

Assembles SAM's LBA case (sounding + the diurnal H/LE surface-flux series) on the
plane CRM and integrates a few steps with the prescribed surface fluxes active
(near local noon), checking the run stays finite + mass-conserving. Skipped when
the gSAM CASES tree is absent.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "run"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import run_lba_plane  # noqa: E402

from legoesm.atmosphere.forcing.sam_case_forcing import resolve_sam_case_dir  # noqa: E402

# Repo-local cache (scripts/data/fetch_les_forcing.py) or external
# LEGOESM_GSAM_ROOT; skip if neither has the deck.
_GSAM_LBA = resolve_sam_case_dir("LBA")


@pytest.mark.skipif(not os.path.isdir(_GSAM_LBA),
                    reason="LBA deck not present "
                           "(run scripts/data/fetch_les_forcing.py)")
def test_lba_runs_stable_and_conserving():
    # nlev=64 + dt=2s for the SAM-faithful stretched grid (dz_sfc=50 m, VGRID).
    out = run_lba_plane.run_lba(
        _GSAM_LBA, nx=4, ny=4, nlev=64, H=18000.0, dx=1000.0,
        dt=2.0, steps=5, day0=0.2, microphysics="morrison",
        radiation="none",   # LBA: PRESCRIBED radiative cooling (doradforcing)
        verbose=False)
    assert out["finite"]
    assert out["rel_mass"] < 1e-10
    setup = out["setup"]
    # LBA: land case, southern-hemisphere, prescribed diurnal fluxes (no lsf)
    assert setup.latitude < 0.0
    assert float(setup.sfc.lhf.max()) > 100.0      # the diurnal latent peak
    # prescribed radiative cooling profile (negative dT/dt in the troposphere)
    assert float(setup.rad.dTdt_rad.min()) < 0.0
    assert out["state"].tracers.data.shape[-1] == 11
