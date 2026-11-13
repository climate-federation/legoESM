"""Run-wide numerics mode for the NEMO-literal code paths.

``"nemo_exact"`` keeps every rounding guard and the card's transcendentals
(bit-for-bit with NEMO).  ``"nemo_fast"`` makes ``nemo_source_round`` the
identity and the eight ``transcendentals`` entries native XLA, whatever the
precision policy asks for.  Outside every scope the mode is ``None`` and the
code runs exactly as it did before this module existed.

The mode is a ``jax.make_user_context`` slot, which is part of the jit cache
key: one jitted function called under both modes compiles two programs.  JAX
takes that key when the outermost jitted call is made, so open the scope at the
outermost Python level of a run, never inside a traced body.  The mode is
thread-local: other threads see ``None``.

Known hole: a ``custom_vjp`` backward is traced when the cotangent is pulled,
so ``jax.vjp`` under a scope with the cotangent pulled after it closes traces
the backward under the mode current at that time.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import jax

NUMERICS_MODES = ("nemo_exact", "nemo_fast")

_MODE = jax.make_user_context(None)


def current_numerics() -> str | None:
    """The active numerics mode, or ``None`` outside every scope.

    Valid at trace time: a guard reads it while JAX traces its caller.
    """
    mode: str | None = _MODE.value
    return mode


@contextlib.contextmanager
def numerics_scope(mode: str) -> Iterator[None]:
    """Run the enclosed code under ``mode``.

    Re-opening the active mode is allowed; opening a different one inside it
    raises, because the outer caller's compiled programs already carry the
    outer mode.
    """
    if mode not in NUMERICS_MODES:
        raise ValueError(
            f"numerics mode must be one of {NUMERICS_MODES}, got {mode!r}")
    active = current_numerics()
    if active is not None and active != mode:
        raise RuntimeError(
            f"cannot open numerics mode {mode!r} inside {active!r}")
    with _MODE(mode):
        yield


def require_numerics(expected: str | None) -> None:
    """Raise unless the active mode is ``expected`` (``None`` means no scope)."""
    active = current_numerics()
    if active != expected:
        raise RuntimeError(
            f"this program was built for numerics mode {expected!r}, "
            f"but the active mode is {active!r}")


class _ModeBoundAOT:
    """An ahead-of-time program that refuses to run under another mode.

    It exposes only calling (and ``vjp`` for an Exported): passing other
    attributes through would let ``serialize`` hand the program on unstamped.
    """

    def __init__(self, program, mode: str | None) -> None:
        self._program = program
        self._mode = mode

    def __call__(self, *args, **kwargs):
        require_numerics(self._mode)
        return self._program(*args, **kwargs)

    def call(self, *args, **kwargs):
        require_numerics(self._mode)
        return self._program.call(*args, **kwargs)

    def vjp(self):
        # Exported.vjp() exports the backward program now, under the active
        # mode, so it must match the mode the primal was built under.
        require_numerics(self._mode)
        return _ModeBoundAOT(self._program.vjp(), self._mode)


def bind_aot(program):
    """Bind a compiled or exported program to the active numerics mode.

    ``Compiled`` and ``Exported`` objects run the program baked when they
    were made, under any later mode; the wrapper raises instead.  Build the
    program in the argument of this call, so it is stamped with the mode it
    was made under.  ``Traced`` and ``Lowered`` are refused: they can still
    be compiled later, under another mode.
    """
    if not isinstance(program, (jax.stages.Compiled, jax.export.Exported)):
        raise TypeError(
            "bind_aot takes a jax.stages.Compiled or jax.export.Exported, "
            f"got {type(program).__name__}")
    return _ModeBoundAOT(program, current_numerics())


__all__ = (
    "NUMERICS_MODES",
    "bind_aot",
    "current_numerics",
    "numerics_scope",
    "require_numerics",
)
