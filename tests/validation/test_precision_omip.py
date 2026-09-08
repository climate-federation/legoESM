"""Cheap precision-policy validation for the OMIP ``--precision`` flag.

These are the TRIPWIRE gate, NOT the production campaign. They assert that
``--precision mixed`` installs the precision-sensitive ocean fp64 overrides
through the canonical ``apply_precision`` bridge — so the barotropic elliptic
solve / EOS / pressure-gradient / coriolis stay fp64 even when storage+compute
are fp32 (hydrostatic & elliptic cancellation are f32-unsafe) — and that the
fp32 / fp64 modes resolve the expected dtypes.

The FULL production-precision validation (century-scale fp64-vs-mixed global
SST/SSS drift, AMOC, heat/salt content) is a separate long sbatch campaign in
the OMIP-faithful pipeline, not a unit test; this gate only proves the policy
is wired correctly, cheaply and deterministically (no model run, no data).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from legoesm.core.precision import (
    PrecisionPolicy,
    clear_module_overrides,
    get_module_overrides,
    resolve_dtype,
    set_policy,
)
from legoesm.runtime.precision import apply_precision

# The ocean kernels mixed-mode must keep in fp64 (see precision._OCEAN_OVERRIDES).
_OCEAN_SENSITIVE = (
    "barotropic_solver",
    "pressure_gradient",
    "equation_of_state",
    "coriolis",
)


@pytest.fixture(autouse=True)
def _restore_precision():
    # OMIP baseline is fp64; clear overrides + reset policy BEFORE and AFTER
    # every test so the global precision state never leaks across tests.
    clear_module_overrides()
    set_policy(PrecisionPolicy.fp64())
    yield
    clear_module_overrides()
    set_policy(PrecisionPolicy.fp64())


def test_mixed_is_refused():
    # #1665 interim: mixed is refused loudly until it is consistent.
    import pytest
    with pytest.raises(NotImplementedError, match="disabled"):
        apply_precision("mixed")


def test_mixed_storage_refused():
    # #1665 interim: mixed refused (was: fp32 storage assertion).
    import pytest
    with pytest.raises(NotImplementedError, match="disabled"):
        apply_precision("mixed")


def test_fp64_mode_is_all_fp64():
    apply_precision("fp64")
    for mod in (*_OCEAN_SENSITIVE, "tracer_advection"):
        assert resolve_dtype(mod, "compute") == jnp.dtype("float64"), mod
        assert resolve_dtype(mod, "storage") == jnp.dtype("float64"), mod


def test_fp32_mode_resolves_fp32():
    apply_precision("fp32")
    assert resolve_dtype("tracer_advection", "compute") == jnp.dtype("float32")
    assert resolve_dtype("tracer_advection", "storage") == jnp.dtype("float32")


def test_unknown_precision_mode_raises():
    """apply_precision dispatch-hardening: unknown mode raises, no silent
    fp64 fallback (mirrors the argparse choices guard on --precision)."""
    with pytest.raises(ValueError, match="Unknown precision mode"):
        apply_precision("bf16")


def test_mode_switch_and_mixed_refused():
    """#1665 interim: mixed (which pinned ocean fp64 overrides) is refused;
    fp32<->fp64 switching still resolves correctly (apply_precision clears
    overrides on every switch)."""
    import pytest
    with pytest.raises(NotImplementedError, match="disabled"):
        apply_precision("mixed")
    apply_precision("fp32")
    assert resolve_dtype("barotropic_solver", "compute") == jnp.dtype("float32")
    apply_precision("fp64")
    assert resolve_dtype("barotropic_solver", "compute") == jnp.dtype("float64")

