"""Per-case Veros setup factories registered with the Veros runner.

Each module exposes a single ``make_setup(**kwargs) -> veros.VerosSetup``
factory. The factories are kept thin: ``acc_channel`` and
``global_overturning`` adapt Veros's built-in ``ACCSetup`` and
``GlobalFourDegreeSetup`` with light overrides; the remaining cases
(lock_exchange, overflow, eady_uniform, dino) need custom ``VerosSetup``
subclasses and are tracked separately — see ``src/legoesm/ocean/fidelity/
veros_configs/_not_yet_implemented.md`` for status.

``AVAILABLE_CASES`` is the dispatch table that :mod:`legoesm.ocean.fidelity
.veros_runner` consults to map a case name to its factory.
"""

from __future__ import annotations

from typing import Any, Callable

from . import acc_channel, global_overturning, lock_exchange, overflow

AVAILABLE_CASES: dict[str, Callable[..., Any]] = {
    "acc_channel": acc_channel.make_setup,
    "global_overturning": global_overturning.make_setup,
    "lock_exchange": lock_exchange.make_setup,
    "overflow": overflow.make_setup,
}

__all__ = (
    "AVAILABLE_CASES",
    "acc_channel",
    "global_overturning",
    "lock_exchange",
    "overflow",
)
