"""Ocean-model fidelity-assessment harness (lat-lon C-grid).

Three layers:

1. **Layer A** — CI pytest fixtures producing scalar metrics checked against
   committed JSON tolerances. Tiers 0-4 every PR; Tier 5 nightly.
2. **Layer B** — Markdown / HTML release report (``scripts/ocean_fidelity/
   build_fidelity_report.py``). Tiers 0-7.
3. **Layer C** — Per-tier analysis notebooks under ``notebooks/ocean_fidelity/``.
   Tiers 5-8.

The fidelity layer never re-integrates the model: it consumes artifacts
emitted by ``scripts/run_ocean_test_matrix.py``.

Submodules are loaded lazily so ``import legoesm.ocean.fidelity`` stays cheap
and so that heavy optional dependencies (Veros, copernicusmarine) are only
imported when the relevant submodule is touched.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__all__ = (
    "artifacts",
    "cache",
    "compare",
    "diff",
    "metrics",
    "references",
    "registry",
    "regrid_veros",
    "report_grid",
    "tendency_probe",
    "tolerances",
    "veros_acc_recipe",
    "veros_runner",
    "veros_state_bridge",
)

if TYPE_CHECKING:  # pragma: no cover - import-time-only type stubs
    from . import (  # noqa: F401
        artifacts,
        cache,
        compare,
        diff,
        metrics,
        references,
        registry,
        regrid_veros,
        report_grid,
        tendency_probe,
        tolerances,
        veros_acc_recipe,
        veros_runner,
        veros_state_bridge,
    )


def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
