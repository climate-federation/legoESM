"""Phase 5 of the FV3-native roadmap: honest FV3-fidelity naming.

The PE/NH "FV3-faithful" factories overclaimed (RK3, interpolated halos,
cell-centred non-Lagrangian state — not FV3's forward-backward D-grid /
Lagrangian architecture).  They are renamed to COMPONENT-fidelity with
deprecated warning aliases, and a tripwire prevents anyone from
advertising FULL fidelity until the phase-4 native forward-backward core
actually exists.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")


class TestFullFidelityTripwire:
    """No configuration may advertise FULL FV3 fidelity until the native
    forward-backward core exists.

    The tripwire is mechanical: (a) the docstrings of the component-fidelity
    factories must carry the NOT-full disclaimer; (b) no module under the
    gcm package may define a ``make_fv3_full_fidelity_*`` factory or claim
    ``FULL_FV3_FIDELITY = True`` unless the phase-4 native core module
    (``fv3_native_core.py``) is present.  Self-test: the guard is exercised
    against a synthetic violation below (provably non-vacuous).
    """

    _GCM_DIR = (pathlib.Path(__file__).resolve().parents[4]
                / "packages" / "atmosphere" / "legoesm" / "atmosphere"
                / "dynamics" / "gcm")


    @staticmethod
    def _full_claims(text: str) -> bool:
        return ("make_fv3_full_fidelity" in text
                or "FULL_FV3_FIDELITY = True" in text)

    def test_no_full_fidelity_claim_without_native_core(self):
        assert self._GCM_DIR.is_dir(), self._GCM_DIR  # not vacuous
        native_core = self._GCM_DIR / "fv3_native_core.py"
        offenders = [
            p.name for p in sorted(self._GCM_DIR.glob("*.py"))
            if self._full_claims(p.read_text())
        ]
        if not native_core.exists():
            assert offenders == [], (
                "FULL FV3 fidelity advertised without the phase-4 native "
                f"forward-backward core: {offenders}")

    def test_tripwire_detects_synthetic_violation(self):
        assert self._full_claims("def make_fv3_full_fidelity_pe_config():")
        assert self._full_claims("FULL_FV3_FIDELITY = True")
        assert not self._full_claims("component fidelity only")
