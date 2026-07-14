"""Smoke test for the GATE_IDEAL plane run (``scripts/run/run_gate_plane.py``).

Assembles SAM's GATE_IDEAL case (sounding + large-scale forcing + SST) on the
plane CRM and integrates a few steps through the full physics stack (oceflx +
gray radiation + double-moment Morrison + Smagorinsky + SAM forcing), checking
the run stays finite and mass-conserving. Skipped when the gSAM CASES tree is
not present (the case data is the oracle).
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

import run_gate_plane  # noqa: E402

from legoesm.atmosphere.forcing.sam_case_forcing import resolve_sam_case_dir  # noqa: E402

# Repo-local cache (scripts/data/fetch_les_forcing.py) or external
# LEGOESM_GSAM_ROOT; skip if neither has the deck.
_GSAM_GATE = resolve_sam_case_dir("GATE_IDEAL")


@pytest.mark.skipif(not os.path.isdir(_GSAM_GATE),
                    reason="GATE_IDEAL deck not present "
                           "(run scripts/data/fetch_les_forcing.py)")
def test_gate_ideal_runs_stable_and_conserving():
    # nlev=64 + dt=2s: the SAM-faithful STRETCHED grid (dz_sfc=50 m near the
    # surface, VGRID) needs the finer dt for CFL and enough levels for a gentle
    # stretch (the old uniform-667 m grid ran at nlev=20/dt=5 but couldn't
    # resolve the boundary layer SAM does).
    out = run_gate_plane.run_gate_ideal(
        _GSAM_GATE, nx=4, ny=4, nlev=64, H=18000.0, dx=1000.0,
        dt=2.0, steps=5, microphysics="morrison", radiation="gray",
        verbose=False)
    assert out["finite"]                      # no NaN/Inf through the stack
    assert out["rel_mass"] < 1e-10            # dry mass conserved
    # the SAM GATE sounding + forcing produce a sensible moist sheared column
    setup = out["setup"]
    assert setup.sst == pytest.approx(299.88)
    assert setup.coriolis is False
    # double-moment graupel (11 tracers) carried through
    assert out["state"].tracers.data.shape[-1] == 11
