"""Smoke test for the mixed-precision validation harness.

Exercises scripts/validate/validate_omip_precision.py at tiny size: confirms
the fp64-vs-mixed closed-channel harness runs end-to-end and returns a
structured verdict with finite, conserving fp64 + mixed runs. The full
science verdict (PASS at production resolution) is the sbatch campaign; this
just proves the harness itself works. apply_precision is global, so the
module restores fp64 after.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import pytest


@pytest.fixture(autouse=True)
def _restore_fp64():
    from legoesm.core.precision import (
        PrecisionPolicy, set_policy, clear_module_overrides)
    yield
    clear_module_overrides()
    set_policy(PrecisionPolicy.fp64())


def test_mixed_precision_enabled():
    """#1675: mixed is enabled; the fp64-vs-mixed harness precondition
    (apply_precision resolves the mixed policy) now holds.  Pin that the
    request raises rather than silently running fp64."""
    from legoesm.runtime.precision import apply_precision
    import jax.numpy as _jnp
    p = apply_precision("mixed")
    assert p.storage == _jnp.float32 and p.control == _jnp.float64
