"""Precision policy — single source of truth.

Re-exports the full precision API from ``core.precision`` (which remains
the implementation home for casting helpers, module overrides, and the
``PrecisionPolicy`` NamedTuple).  This module adds only the thin glue
that was previously in ``core.hardware`` (the 3-component legacy
policy) and consolidates it into the canonical ``PrecisionPolicy``.

Design
------
* ``core.precision`` keeps all implementation.
* ``runtime.precision`` is the **public** import path.
* The legacy 3-component dict (dynamics/ml/conservation) in
  ``core.hardware`` is no longer authoritative — we keep backward-compat
  helpers that delegate to ``PrecisionPolicy``.
"""

from __future__ import annotations

# Re-export the full public API from core.precision.
from legoesm.core.precision import (        # noqa: F401
    PrecisionPolicy,
    set_policy,
    get_policy,
    set_module_override,
    clear_module_overrides,
    get_module_overrides,
    cast,
    const,
    cast_pytree,
    finalize_to_storage,
    ACCUMULATE_ROLE_LEAVES,
    global_sum,
    global_max,
    norm,
    weighted_mean,
    compensated_sum,
    with_precision,
    set_recommended_overrides,
)


# ---------------------------------------------------------------------------
# Bridge: resolve a precision *mode string* into a PrecisionPolicy
# ---------------------------------------------------------------------------

_MODE_FACTORIES = {
    "fp32": PrecisionPolicy.fp32,
    "float32": PrecisionPolicy.fp32,
    "fp64": PrecisionPolicy.fp64,
    "float64": PrecisionPolicy.fp64,
    "mixed": PrecisionPolicy.mixed,
    "mixed_fp64_storage": PrecisionPolicy.mixed_fp64_storage,
}

#: Precision modes whose STORAGE is float64 (so they require an fp64-capable
#: backend — e.g. not Apple Metal).  ``"mixed"`` keeps fp32 storage, so it is OK
#: on fp64-less backends.
_FP64_STORAGE_MODES = frozenset({"fp64", "float64", "mixed_fp64_storage"})


def available_precision_modes() -> tuple[str, ...]:
    """Sorted names of the KNOWN precision modes.

    All four are activatable.  ``'mixed'`` / ``'mixed_fp64_storage'`` were
    refused for the duration of the #1665 interim and were re-enabled by
    #1675; they require JAX x64 (``apply_precision`` says so loudly if it is
    off), because their accumulate and control roles are float64.
    """
    return tuple(sorted(_MODE_FACTORIES))


def precision_requires_fp64(mode: str) -> bool:
    """True if *mode* stores state in float64 (needs an fp64-capable backend)."""
    return mode.strip().lower() in _FP64_STORAGE_MODES


def resolve_precision(mode: str = "fp32") -> PrecisionPolicy:
    """Return the ``PrecisionPolicy`` for *mode*.

    Parameters
    ----------
    mode : str
        ``"fp32"``, ``"fp64"``, or ``"mixed"``.

    Returns
    -------
    PrecisionPolicy
    """
    key = mode.strip().lower()
    factory = _MODE_FACTORIES.get(key)
    if factory is None:
        raise ValueError(
            f"Unknown precision mode {mode!r}. "
            f"Use one of {sorted(_MODE_FACTORIES.keys())}."
        )
    return factory()


def apply_precision(mode: str = "fp32") -> PrecisionPolicy:
    """Resolve *mode*, activate the policy globally, and enable x64 if needed.

    This is the **one-shot** entry point used by :func:`runtime.bootstrap`
    AND by the driver CLIs directly, so gating a mode here covers every caller.
    """
    policy = resolve_precision(mode)

    # #1675: the #1665 interim refusal of 'mixed' is LIFTED — the eager dycore
    # step now re-casts its bulk state to the storage dtype at the step
    # boundary (``core.precision.finalize_to_storage``), which is what the
    # refusal was waiting for.  What is NOT optional is x64: 'mixed' means
    # fp32 storage/compute with float64 accumulate/control, and with x64 off
    # JAX silently turns every float64 role back into float32 — i.e. the mode
    # would claim a precision it does not run.  Refuse that combination
    # explicitly rather than degrade in silence.  fp32/fp64 are untouched.
    if mode.strip().lower() in ("mixed", "mixed_fp64_storage"):
        from legoesm.runtime.backend import (
            get_backend, is_x64_enabled, supports_float64,
        )

        if not is_x64_enabled():
            raise RuntimeError(
                f"precision={mode!r} needs JAX x64 enabled: its accumulate and "
                "control roles are float64, and with x64 off JAX demotes them "
                "to float32, so the run would not be mixed precision at all. "
                "Go through legoesm.runtime.configure_runtime/bootstrap (which "
                "enables x64 for this mode), or set JAX_ENABLE_X64=1."
            )
        # x64 alone is not enough: on a backend with no native float64 (Apple
        # Metal) every role silently resolves to float32, so the mode would
        # again claim a precision it does not run.  Adversarial review found
        # this by mocking the backend away (#1675).
        if not supports_float64():
            raise RuntimeError(
                f"precision={mode!r} needs a backend with native float64; "
                f"{get_backend()!r} has none, so its float64 accumulate and "
                "control roles would resolve to float32 and the mode would be "
                "indistinguishable from 'fp32'. Use precision='fp32' on this "
                "backend."
            )

    # Activate globally.
    set_policy(policy)

    # Clear any per-module overrides left by a PRIOR apply_precision so a
    # mode switch is clean: without this, ``apply_precision("mixed")`` then
    # ``apply_precision("fp32")`` would leave barotropic_solver / EOS / PGF /
    # coriolis pinned fp64, i.e. fp32 mode would not actually be fp32 in a
    # long-lived process (codex 2026-06-21).
    clear_module_overrides()

    # Apply recommended per-module overrides when in mixed mode.
    if mode.strip().lower() in ("mixed", "mixed_fp64_storage"):
        set_recommended_overrides("mixed")

    return policy
