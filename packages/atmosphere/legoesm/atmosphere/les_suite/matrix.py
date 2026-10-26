"""Matrix-runner wiring for the LES-truth suite (mirrors ``_atm_matrix_spec``).

Turns the :mod:`~legoesm.atmosphere.les_suite.registry` catalog into an enumerable
test matrix and provides the shared-selector spec so the case×grid selection is
never re-implemented (CLAUDE.md recipe×setup rule). The matrix "grid" axis is the
LES **resolution label** (``LESGrid.label`` = ``"96x96x96"``) — a real experimental
axis (the D7/D8 convergence runs live at a different resolution). The LES core is
locked to ``"spectral"`` (D1), so it is not a selection axis.

Selection contract (dispatch hardening, CLAUDE.md):
:func:`select_cases` with an EXACT case name (or a grid) matching nothing raises
:class:`SystemExit`, never a silent empty no-op. ``--only =<name>`` requests exact
match; a bare ``--only <substr>`` is a substring filter.
"""
from __future__ import annotations

from dataclasses import dataclass

from .registry import LES_CASE_REGISTRY, LESCase, list_cases


def _ensure_catalog() -> None:
    """Populate the process-global registry with the default catalog once."""
    if not LES_CASE_REGISTRY:
        from .catalog import register_default_catalog

        register_default_catalog()


def valid_grid_labels() -> tuple[str, ...]:
    """Distinct resolution labels present in the registry (the ``--grid`` set)."""
    _ensure_catalog()
    return tuple(sorted({c.grid.label for c in list_cases()}))


def _les_suite_matrix_spec():
    """:class:`MatrixRunnerSpec` for a LES-suite ``setup:`` template.

    The LES matrix has no ``--levels``/``--dt``/``--days``/``--resolution`` flags
    (a case IS a fixed resolution+dt+duration); selection is case-name + grid
    label only. Deferred import keeps the federation DAG clean (atmosphere→core).
    """
    from legoesm.core.setup_selector import MatrixRunnerSpec

    return MatrixRunnerSpec(
        runner_path="scripts/matrix/run_les_suite_matrix.py",
        valid_grids=valid_grid_labels(),
        case_flag="--only",
        exact_prefix="=",
        levels_flag=None,
        dt_flag=None,
        days_flag=None,
        resolution_flag=None,
        output_flag="--output",
        quick_flag=None,
    )


@dataclass(frozen=True)
class MatrixCase:
    """One enumerable (case, grid) pair — the matrix-test unit.

    Field names (``case``, ``grid_type``) match the other components' matrix
    entries so the shared setup-template test harness can consume them uniformly.
    """

    case: str
    grid_type: str


def build_test_matrix() -> list[MatrixCase]:
    """Every registered LESCase as a ``(name, grid-label)`` matrix entry."""
    _ensure_catalog()
    return [MatrixCase(case=c.name, grid_type=c.grid.label) for c in list_cases()]


def select_cases(
    only: str | None = None, grid: str | None = None
) -> list[LESCase]:
    """Registry cases filtered by ``--only`` and ``--grid`` (dispatch-hardened).

    * ``only`` ``"=<name>"`` → exact case-name match; a bare ``"<substr>"`` →
      substring match. ``None`` → all cases.
    * ``grid`` → keep only cases whose ``grid.label`` equals it; must be a known
      label.

    Raises :class:`SystemExit` if a selection would match nothing — an exact name
    that does not exist, a grid label that is not present, or a combined filter
    with an empty intersection — so a typo fails loudly rather than silently
    running zero cases (CLAUDE.md dispatch hardening).
    """
    _ensure_catalog()
    cases = list_cases()
    exact = only is not None and only.startswith("=")

    if grid is not None:
        known = valid_grid_labels()
        if grid not in known:
            raise SystemExit(
                f"--grid {grid!r} is not a known LES grid label; "
                f"choose one of {list(known)}"
            )
        cases = [c for c in cases if c.grid.label == grid]

    if only is not None:
        if exact:
            target = only[1:]
            cases = [c for c in cases if c.name == target]
            if not cases:
                raise SystemExit(
                    f"--only ={target!r} matched no LES case"
                    + (f" at --grid {grid}" if grid is not None else "")
                    + f"; known cases: {[c.name for c in list_cases()]}"
                )
        else:
            cases = [c for c in cases if only in c.name]
            if not cases:
                raise SystemExit(
                    f"--only {only!r} (substring) matched no LES case"
                    + (f" at --grid {grid}" if grid is not None else "")
                )
    elif grid is not None and not cases:
        # grid was valid but (combined with nothing else) still empty — defensive.
        raise SystemExit(f"--grid {grid!r} selected no LES case")

    return cases
