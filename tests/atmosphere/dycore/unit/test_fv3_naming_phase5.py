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

from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (  # noqa: E402
    make_fv3_component_fidelity_nh_config,
    make_fv3_faithful_nh_config,
)
from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (  # noqa: E402
    make_fv3_component_fidelity_pe_config,
    make_fv3_faithful_pe_config,
)


class TestDeprecatedAliases:
    def test_pe_alias_warns_and_matches(self):
        with pytest.warns(FutureWarning, match="COMPONENT-fidelity"):
            old = make_fv3_faithful_pe_config()
        new = make_fv3_component_fidelity_pe_config()
        assert old == new

    def test_nh_alias_warns_and_matches(self):
        with pytest.warns(FutureWarning, match="COMPONENT-fidelity"):
            old = make_fv3_faithful_nh_config()
        new = make_fv3_component_fidelity_nh_config()
        assert old == new

    def test_overrides_flow_through_alias(self):
        with pytest.warns(FutureWarning):
            old = make_fv3_faithful_pe_config(nord_v=2)
        assert old.nord_v == 2


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

    def test_component_factories_carry_disclaimer(self):
        for fn in (make_fv3_component_fidelity_pe_config,
                   make_fv3_component_fidelity_nh_config):
            doc = " ".join((fn.__doc__ or "").split())
            assert "NOT a full FV3" in doc, fn.__name__

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
