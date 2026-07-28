"""Tests for the oracle-fidelity fp64 precision contract (#1226, skill Rule 1c).

The gate must be NON-VACUOUS: it has to fire on the exact real-world shape that
motivated it — an f32 depth ladder sitting underneath an f64 state, which is
what silently produced the ``eos_rab``/``bn2``/``zdf_mxl`` residuals.
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.ocean.fidelity.precision_gate import (
    describe_float_leaves,
    require_fp64,
)


@pytest.fixture(autouse=True)
def _restore_policy():
    saved = get_policy()
    yield
    set_policy(saved)


def test_passes_under_fp64_policy_with_f64_leaves():
    set_policy(PrecisionPolicy.fp64())
    require_fp64(jnp.ones(3, dtype=jnp.float64), context="unit")


def test_rejects_fp32_policy():
    set_policy(PrecisionPolicy.fp32())
    with pytest.raises(ValueError, match="FLOAT64 REQUIRED"):
        require_fp64(context="unit")


def test_error_names_the_context_and_the_x64_trap():
    set_policy(PrecisionPolicy.fp32())
    with pytest.raises(ValueError) as ei:
        require_fp64(context="zdf_mxl nmln compare")
    msg = str(ei.value)
    assert "zdf_mxl nmln compare" in msg
    # the whole point: x64 is not the same thing as the policy
    assert "JAX_ENABLE_X64=1 does NOT do this" in msg


def test_fires_on_f32_geometry_under_an_f64_state():
    """THE regression: correct-looking state, single-precision ladder.

    This is the shape that cost #1226 weeks — T/S were f64 while the depth
    ladder was f32, so every state-level check passed.
    """
    set_policy(PrecisionPolicy.fp64())
    state_t = jnp.ones((2, 2, 4), dtype=jnp.float64)      # looks fine
    ladder = jnp.ones(4, dtype=jnp.float32)                # the actual bug
    require_fp64(state_t, context="state only")            # state alone passes
    with pytest.raises(ValueError, match="float32"):
        require_fp64(state_t, ladder, context="state + geometry")


def test_skips_integer_and_bool_leaves():
    """Level indices / wet masks carry no precision and must not trip it."""
    set_policy(PrecisionPolicy.fp64())
    require_fp64(
        {"bottom_level": jnp.arange(4, dtype=jnp.int32),
         "is_active": jnp.ones(4, dtype=bool),
         "gdept": jnp.ones(4, dtype=jnp.float64)},
        context="mixed pytree",
    )


def test_walks_nested_pytrees_and_reports_the_path():
    set_policy(PrecisionPolicy.fp64())
    tree = {"z": {"dz_ref": jnp.ones(3, dtype=jnp.float32)}}
    with pytest.raises(ValueError) as ei:
        require_fp64(tree, context="nested")
    assert "dz_ref" in str(ei.value)


def test_describe_float_leaves_reports_dtypes_without_raising():
    leaves = dict(describe_float_leaves(
        {"a": jnp.ones(2, dtype=jnp.float32),
         "n": jnp.arange(2, dtype=jnp.int64)}))
    assert any(v == "float32" for v in leaves.values())
    assert not any("n" in k for k in leaves)   # int leaf skipped


def test_oracle_bridge_warns_once_under_fp32():
    """RUNNING in fp32 is legitimate; COMPARING to an fp64 oracle is not.

    The bridge therefore warns rather than raising (a 5-year model run in fp32
    is a valid performance choice), while comparison probes use the hard
    require_fp64 gate. Regression for the f32 depth ladder that silently
    corrupted every #1226 measurement.
    """
    import warnings

    from legoesm.ocean.fidelity import nemo_state_bridge as bridge

    set_policy(PrecisionPolicy.fp32())
    bridge._FP32_BRIDGE_WARNED = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bridge._warn_if_not_fp64()
        bridge._warn_if_not_fp64()          # one-shot, must not repeat
    assert len(caught) == 1
    assert "JAX_ENABLE_X64=1 does NOT change this" in str(caught[0].message)

    set_policy(PrecisionPolicy.fp64())
    bridge._FP32_BRIDGE_WARNED = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bridge._warn_if_not_fp64()
    assert not caught
