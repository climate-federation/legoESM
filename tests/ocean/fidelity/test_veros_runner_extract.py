"""Direct test of veros_runner._extract_state's inactive-variable handling.

Veros RAISES ``RuntimeError`` when accessing an inactive variable (a disabled
diagnostic), which a plain ``getattr(obj, name, default)`` does NOT shield (it
only catches ``AttributeError``) -- so requesting one inactive capture var would
abort the whole, already-integrated (expensive) run. This is a pure-mock test
(no Veros install needed): ``veros_runner`` imports ``veros`` lazily, so
importing ``_extract_state`` is cheap and ungated.
"""
from __future__ import annotations

import types

import numpy as np


def test_extract_state_skips_inactive_runtime_error_vars():
    from legoesm.ocean.fidelity.veros_runner import _extract_state

    class _Vars:
        def __init__(self):
            self.active = np.ones((2, 3))

        def __getattr__(self, name):
            # __getattr__ only fires for attributes NOT in __dict__; "active"
            # (set in __init__) is resolved normally and never routed here.
            if name == "inactive":
                raise RuntimeError(f"variable {name!r} is not active")
            raise AttributeError(name)

    setup = types.SimpleNamespace(
        state=types.SimpleNamespace(variables=_Vars()))

    out = _extract_state(
        setup, capture_vars=("active", "inactive", "missing"))

    # "inactive" (RuntimeError) and "missing" (AttributeError -> getattr None)
    # are both skipped; "active" is returned. Crucially, no exception escapes.
    assert set(out) == {"active"}
    np.testing.assert_array_equal(out["active"], np.ones((2, 3)))
