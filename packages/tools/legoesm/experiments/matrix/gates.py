"""Conservation PASS/FAIL gates for the test matrix — thin wrappers on the
already-centralized drift primitives.

Every gate delegates to
:mod:`legoesm.diagnostics.conservation_drift`
(``compute_relative_drift`` / ``apply_drift_tolerance`` / ``apply_value_threshold``),
so there is exactly **one** floor/NaN/tolerance convention across the whole
codebase (the iter-78/80/83/87/152 history lives in that module).  This module
adds only (a) the per-component default tolerances — the single source of
``CONS_THRESH``, migrated from ``scripts/validate_matrix_report.py`` — and (b)
the ``(ok, notes)`` plumbing that lets a runner chain several gates on one case.

The gates take **numeric series** (already-computed conservation diagnostics:
mass, total energy, column water vapor, angular momentum), NOT model states, so
this module imports no component package — it is safe to import from
``legoesm-tools`` without pulling in atmosphere/ocean/land/ice.  The per-component
runner computes the series (it owns the state) using
``diagnostics.compute_total_energy_*`` / ``column_water_vapor`` and feeds the
scalar timeseries here.
"""
from __future__ import annotations

from typing import Sequence

from legoesm.diagnostics.conservation_drift import (
    apply_drift_tolerance,
    apply_value_threshold,
    compute_relative_drift,
)

#: Per-component default relative-drift tolerances.  Single source of truth —
#: ``scripts/validate_matrix_report.py`` imports this rather than redefining it.
#: Values are dimensionless relative drift (production baselines) or absolute
#: drift in natural units when the baseline magnitude is below
#: ``DEFAULT_MIN_BASELINE`` (rest-state runs) — see ``conservation_drift``.
CONS_THRESH: dict[str, dict[str, float]] = {
    "atmosphere": {"mass_rel_drift": 1e-3, "energy_rel_drift": 5e-2,
                   "moisture_rel_drift": 5e-2, "aam_rel_drift": 5e-2},
    "ocean": {"volume_rel_drift": 1e-4, "heat_rel_drift": 1e-2,
              "salt_rel_drift": 1e-2},
    "sea_ice": {"volume_rel_drift": 5e-2, "mass_rel_drift": 1e-2},
    "land": {"water_rel_drift": 1e-2, "energy_rel_drift": 5e-2},
    "coupled": {"mass_rel_drift": 1e-3, "energy_rel_drift": 5e-2},
}


def default_tol(component: str, key: str) -> float:
    """Return the configured tolerance for ``(component, key)``.

    Raises ``KeyError`` (loud, fail-fast) on an unknown component or diagnostic
    key — a typo in a gate call must not silently fall back to a permissive
    default that lets a broken run PASS.
    """
    comp = CONS_THRESH[component]
    if key not in comp:
        raise KeyError(
            f"no tolerance for {key!r} under component {component!r}; "
            f"known keys: {sorted(comp)}"
        )
    return comp[key]


def drift_gate(
    ok: bool,
    notes: str,
    series: Sequence[float],
    *,
    label: str,
    component: str | None = None,
    key: str | None = None,
    tol: float | None = None,
) -> tuple[bool, str]:
    """Gate a conservation timeseries on its first→last relative drift.

    Computes ``compute_relative_drift(series)`` and applies
    ``apply_drift_tolerance`` against ``tol`` (explicit) or
    ``CONS_THRESH[component][key]``.  Exactly one of ``tol`` or
    ``(component, key)`` must be given.  Non-finite samples anywhere in the
    series → drift is NaN → gate FAILs (the ``conservation_drift`` contract).

    Returns the updated ``(ok, notes)`` so gates chain:

        ok, notes = drift_gate(ok, notes, mass, label="mass",
                               component="atmosphere", key="mass_rel_drift")
        ok, notes = drift_gate(ok, notes, te, label="energy",
                               component="atmosphere", key="energy_rel_drift")
    """
    if tol is None:
        if component is None or key is None:
            raise ValueError(
                "drift_gate requires either tol= or (component=, key=)"
            )
        tol = default_tol(component, key)
    drift = compute_relative_drift(series)
    return apply_drift_tolerance(
        ok, notes, drift, tol, label=label, n_samples=len(series)
    )


def mass_gate(ok, notes, series, *, component="atmosphere", tol=None):
    """Mass / volume conservation drift gate (key ``mass_rel_drift``)."""
    key = "volume_rel_drift" if component in ("ocean", "sea_ice") else "mass_rel_drift"
    return drift_gate(ok, notes, series, label="mass", component=component,
                      key=key, tol=tol)


def energy_gate(ok, notes, series, *, component="atmosphere", tol=None):
    """Total-energy / heat-content conservation drift gate.

    Component-agnostic by alias (codex review MEDIUM-1): the ocean's energy
    diagnostic is heat content (``heat_rel_drift``), the atmosphere/land/coupled
    use total energy (``energy_rel_drift``).  This routes ``component="ocean"``
    to the ocean heat key so the advertised shared gate works for every
    component that has an energy tolerance — feed the
    ``compute_total_energy_nh``/``_pe`` series (atmosphere) or the ocean
    heat-content series accordingly.  ``sea_ice`` has no energy key (it gates on
    volume/mass) and raises a clear ``KeyError`` if passed here.
    """
    key = "heat_rel_drift" if component == "ocean" else "energy_rel_drift"
    return drift_gate(ok, notes, series, label="energy", component=component,
                      key=key, tol=tol)


def heat_gate(ok, notes, series, *, component="ocean", tol=None):
    """Ocean heat-content conservation drift gate (key ``heat_rel_drift``).

    Explicit alias of :func:`energy_gate` for ocean callers that prefer the
    domain-native name; identical routing.
    """
    return drift_gate(ok, notes, series, label="heat", component=component,
                      key="heat_rel_drift", tol=tol)


def salt_gate(ok, notes, series, *, component="ocean", tol=None):
    """Ocean salt conservation drift gate (key ``salt_rel_drift``)."""
    return drift_gate(ok, notes, series, label="salt", component=component,
                      key="salt_rel_drift", tol=tol)


def benchmark_error_gate(ok, notes, error, tol, *, label, units=""):
    """Gate an analytic-benchmark error scalar (e.g. Williamson L2/Linf).

    Thin wrapper on ``apply_value_threshold`` with ``op="le"`` — fail if
    ``error > tol`` or non-finite.  Used by tier-1 idealized cases that compare
    against a closed-form solution rather than a conservation drift.
    """
    return apply_value_threshold(ok, notes, error, tol, label=label,
                                 op="le", units=units)


def finite_gate(ok, notes, arrays, *, label="state"):
    """Gate on all-finite snapshot arrays — the cheapest crash detector.

    ``arrays`` is a mapping name→array (or any iterable of arrays).  Fails the
    case if any value contains NaN/Inf.  No tolerance: this is a hard invariant.
    """
    import numpy as np

    if not ok:
        return ok, notes
    items = arrays.values() if hasattr(arrays, "values") else arrays
    for arr in items:
        if not bool(np.all(np.isfinite(np.asarray(arr)))):
            return False, notes + f" [FAIL: {label} contains NaN/Inf]"
    return ok, notes
