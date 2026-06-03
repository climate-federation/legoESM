"""Per-case Veros setup factories registered with the Veros runner.

Each module exposes a single ``make_setup(**kwargs) -> veros.VerosSetup``
factory. The factories are kept thin: ``acc_channel`` and
``global_overturning`` adapt Veros's built-in ``ACCSetup`` and
``GlobalFourDegreeSetup`` with light overrides; the remaining cases
(``lock_exchange``, ``overflow``, ``eady_uniform``, ``dino``) ship custom
``VerosSetup`` subclasses that mirror the corresponding legoESM
experiments.

``AVAILABLE_CASES`` is the dispatch table that :mod:`legoesm.ocean.fidelity
.veros_runner` consults to map a case name to its factory.
"""

from __future__ import annotations

from typing import Any, Callable

from . import (
    acc_channel,
    dino,
    eady_uniform,
    global_overturning,
    lock_exchange,
    overflow,
)

AVAILABLE_CASES: dict[str, Callable[..., Any]] = {
    "acc_channel": acc_channel.make_setup,
    "global_overturning": global_overturning.make_setup,
    "lock_exchange": lock_exchange.make_setup,
    "overflow": overflow.make_setup,
    "eady_uniform": eady_uniform.make_setup,
    "dino": dino.make_setup,
}

__all__ = (
    "AVAILABLE_CASES",
    "acc_channel",
    "dino",
    "eady_uniform",
    "global_overturning",
    "lock_exchange",
    "overflow",
)
