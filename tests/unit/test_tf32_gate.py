"""TF32-on-Turing gate (scaling review lever #4).

``_nvidia_tf32_supported`` must enable ``tensorfloat32`` matmul precision
ONLY on Ampere (sm_80) and later — Turing (RTX 8000, sm_75) has no TF32
hardware, so enabling it there is a silent no-op and a mixed-fleet
correctness foot-gun.  Verifies the compute-capability major-version parse
across the jaxlib string-format variants (``"8.0"`` / ``"80"`` /
``"sm_80"``) and fails CLOSED on anything unrecognised.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from legoesm.runtime.backend import _nvidia_tf32_supported


class _Dev:
    """Minimal fake device; omits compute_capability when cc is None."""

    def __init__(self, cc=None):
        if cc is not None:
            self.compute_capability = cc


@pytest.mark.parametrize("cc", ["8.0", "80", "sm_80", "8.6", "86", "9.0",
                                "90", "sm_90", "10.0", "100"])
def test_tf32_enabled_on_ampere_and_later(cc):
    with patch("jax.devices", return_value=[_Dev(cc)]):
        assert _nvidia_tf32_supported() is True, f"cc={cc!r} should support TF32"


@pytest.mark.parametrize("cc", ["7.5", "75", "sm_75", "7.0", "70", "6.1",
                                "61", "5.2", "52"])
def test_tf32_disabled_on_turing_and_earlier(cc):
    with patch("jax.devices", return_value=[_Dev(cc)]):
        assert _nvidia_tf32_supported() is False, (
            f"cc={cc!r} (pre-Ampere) must NOT enable TF32")


def test_tf32_disabled_when_no_devices():
    with patch("jax.devices", return_value=[]):
        assert _nvidia_tf32_supported() is False


def test_tf32_disabled_when_no_compute_capability():
    # CPU / Metal devices expose no compute_capability attribute.
    with patch("jax.devices", return_value=[_Dev(None)]):
        assert _nvidia_tf32_supported() is False


@pytest.mark.parametrize("cc", ["", "foo", "sm_", ".", "8x", "sm_8x", "8",
                                "8.x", "x.0", "8.0.1"])
def test_tf32_fails_closed_on_unparseable(cc):
    # "8x"/"sm_8x" are the dangerous ones: a naive s[:-1] parse would read
    # major 8 and (wrongly) enable TF32.  A lone "8" is ambiguous → reject.
    with patch("jax.devices", return_value=[_Dev(cc)]):
        assert _nvidia_tf32_supported() is False, (
            f"unparseable cc={cc!r} must fail closed (no TF32)")


@pytest.mark.parametrize("ccs,expected", [
    (["sm_80", "sm_80"], True),     # homogeneous Ampere
    (["8.0", "8.6"], True),         # homogeneous Ampere/Ada
    (["sm_80", "sm_75"], False),    # MIXED Ampere+Turing → process-global off
    (["sm_75", "sm_80"], False),    # order independence
    (["8.0", "7.5", "9.0"], False),  # any Turing present → off
    (["sm_75", "sm_70"], False),    # all pre-Ampere
])
def test_tf32_mixed_fleet(ccs, expected):
    """The matmul-precision config is process-global, so a heterogeneous
    CUDA fleet must gate on the MINIMUM compute capability — never enable
    TF32 if any device is pre-Ampere (codex 2026-06-13 MAJOR)."""
    with patch("jax.devices", return_value=[_Dev(c) for c in ccs]):
        assert _nvidia_tf32_supported() is expected, (
            f"fleet {ccs} should be {expected}")


@pytest.mark.parametrize("devs", [
    [_Dev("8.0"), _Dev("8x")],        # Ampere + unparseable string
    [_Dev("8.0"), _Dev(None)],        # Ampere + device with no CC attr
    [_Dev("8.0"), _Dev("foo")],       # Ampere + garbage
    [_Dev("sm_80"), _Dev("sm_8x")],
])
def test_tf32_fails_closed_on_partial_unknown(devs):
    """An Ampere device sitting next to a device whose CC is missing or
    unparseable must FAIL CLOSED — skipping the unknown would enable the
    process-global TF32 setting for an unidentified CUDA device (codex
    2026-06-13 round-2 partial-unknown fail-open)."""
    with patch("jax.devices", return_value=devs):
        assert _nvidia_tf32_supported() is False
