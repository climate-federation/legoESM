"""Shared, component-agnostic data types for the legoESM test matrix.

The four per-component matrix runners (atmosphere / ocean / sea-ice / SCM)
historically each carried their own ``TestCase`` dataclass, module-global
``ALL_RESULTS`` list, and ``record()`` printer — byte-near-identical code that
drifted independently.  This module is the single source of those types so a
runner is reduced to *declaring its cases* and *running one case*; the tier
taxonomy, status taxonomy, result accumulation, and PASS/FAIL printing all live
here.

**Component-agnostic by contract.**  This module imports only the standard
library — never a component package (``legoesm.atmosphere``/``.ocean``/``.land``/
``.ice``).  Per-component case registries (which *do* import their component)
live in ``scripts/matrix/``; the framework stays importable from
``legoesm-tools`` without violating the federation dependency DAG (tools depends
on core/atmosphere/ocean/coupler, NOT land/ice).

The :class:`Tier` ladder is the cross-component contract — it mirrors the
existing physical-complexity axis
(:class:`legoesm.components.complexity.ModelComplexity`) and grid extent axis
(:data:`legoesm.grids.capability.EXTENTS`):

=====  ============  ===========================  ==================================
tier   name          complexity / extent          conservation gate
=====  ============  ===========================  ==================================
0      unit          n/a (kernels/operators)      numerical invariants only
1      research      idealized; column / SW       mass + energy drift, AAM, bench err
2      intermediate  intermediate; hydrostatic    mass + energy + moisture drift
3      operational   full; real forcing           full budget closure + reproducibility
=====  ============  ===========================  ==================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any


class Tier(IntEnum):
    """Complexity rung of a matrix case — the research→operational ladder.

    Integer-valued so callers can filter with ``case.tier <= Tier.RESEARCH``.
    The names map onto :class:`legoesm.components.complexity.ModelComplexity`
    (research↔idealized, intermediate↔intermediate, operational↔full); tier 0
    is the sub-complexity unit/operator rung that has no model-complexity peer.
    """

    UNIT = 0
    RESEARCH = 1
    INTERMEDIATE = 2
    OPERATIONAL = 3

    @property
    def label(self) -> str:
        return _TIER_LABELS[self]

    @classmethod
    def parse(cls, value: "int | str | Tier") -> "Tier":
        """Resolve an int, a member, or a name (``"research"``/``"tier1"``/``"1"``).

        Raises ``ValueError`` on an unknown spelling so a typo in a CLI flag or a
        template ``tier:`` field fails loudly rather than silently defaulting.
        """
        if isinstance(value, Tier):
            return value
        if isinstance(value, int):
            return cls(value)
        key = str(value).strip().lower()
        if key.startswith("tier"):
            key = key[4:]
        if key.isdigit():
            return cls(int(key))
        for member in cls:
            if member.name.lower() == key:
                return member
        raise ValueError(
            f"unknown tier {value!r}; expected one of "
            f"{[m.name.lower() for m in cls]} or 0..{max(cls).value}"
        )


_TIER_LABELS: dict[Tier, str] = {
    Tier.UNIT: "unit",
    Tier.RESEARCH: "research",
    Tier.INTERMEDIATE: "intermediate",
    Tier.OPERATIONAL: "operational",
}


class RunStatus(StrEnum):
    """Outcome of running one matrix case."""

    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIP = "SKIP"

    @property
    def icon(self) -> str:
        return _STATUS_ICONS[self]


_STATUS_ICONS: dict[RunStatus, str] = {
    RunStatus.PASS: "  ",
    RunStatus.FAIL: "**",
    RunStatus.ERROR: "!!",
    RunStatus.SKIP: "--",
}


class RunMaturity(StrEnum):
    """How far a case/template has been validated — the lesommer run-status taxonomy.

    Mirrors ``project_status.md`` in legoESM-ocean-runners: a template is
    ``run_tested`` once an integration has completed cleanly, ``init_only`` when
    it constructs but has not been run end-to-end, and ``structurally_validated``
    when its config passes ``validate_strict`` / ``capability.instantiate`` but
    no integration has been attempted.
    """

    RUN_TESTED = "run_tested"
    INIT_ONLY = "init_only"
    STRUCTURALLY_VALIDATED = "structurally_validated"


@dataclass
class MatrixCase:
    """One case in the matrix — a (component, case, grid, complexity) point.

    Generalizes the per-runner ``TestCase``: ``equation_set`` becomes the typed
    ``complexity`` rung, and the case gains the cross-component axes
    (``component``, ``extent``, ``tier``, ``maturity``).  The field names match
    the existing atmosphere ``TestCase`` where they overlap so the migration in
    Phase 4 is mechanical.
    """

    component: str            # atmosphere, ocean, sea_ice, land, coupled
    case: str                 # williamson2, held_suarez, rest_state, ...
    grid_type: str            # cubed_sphere, latlon, icosahedral, spectral, mpas, column
    tier: Tier
    complexity: str = ""      # ModelComplexity / *Complexity rung (e.g. shallow_water)
    extent: str = "global"    # grids.capability extent (global/regional/.../column)
    resolution: str = ""      # C36, 72x144, ico5, T21, column
    vertical_coord: str = "none"
    duration_days: float = 0.0
    quick_days: float = 0.0
    maturity: RunMaturity = RunMaturity.RUN_TESTED
    run_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Accept loose tier spellings on construction (templates pass strings).
        self.tier = Tier.parse(self.tier)

    @property
    def output_path(self) -> str:
        """Artifact subtree: ``<component>/<complexity>/<case>/<grid>/<res>[/<vcoord>]``.

        Backward-compatible with the atmosphere layout
        (``<equation_set>/<case>/<grid>/<res>[/<vcoord>]``) when ``component`` is
        folded in by the caller; the leading ``component`` segment keeps the
        per-domain result trees disjoint under one ``results/`` root.
        """
        parts = [self.component, self.complexity or self.case, self.case,
                 self.grid_type, self.resolution]
        parts = [p for p in parts if p]
        if self.vertical_coord and self.vertical_coord != "none":
            parts.append(self.vertical_coord)
        return "/".join(parts)

    @property
    def label(self) -> str:
        return f"{self.component}/{self.complexity or self.case}/{self.grid_type}"


@dataclass
class CaseResult:
    """The recorded outcome of one :class:`MatrixCase`."""

    case: MatrixCase
    status: RunStatus
    wall_time: float
    notes: str = ""
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flat JSON record — superset of the legacy atmosphere ``record()`` schema."""
        return {
            "component": self.case.component,
            "test": self.case.case,
            "grid": self.case.grid_type,
            "complexity": self.case.complexity,
            "equation_set": self.case.complexity,  # legacy alias (atmosphere)
            "extent": self.case.extent,
            "tier": int(self.case.tier),
            "resolution": self.case.resolution,
            "vertical_coord": self.case.vertical_coord,
            "maturity": str(self.case.maturity),
            "status": str(self.status),
            "wall_time": self.wall_time,
            "notes": self.notes,
            "metrics": self.metrics,
        }


class ResultRecorder:
    """Accumulates :class:`CaseResult`s and prints the live PASS/FAIL line.

    Replaces the module-global ``ALL_RESULTS`` + ``record()`` in each runner with
    an instance (no shared mutable global → safe under pytest / multiple runs).
    """

    def __init__(self, *, verbose: bool = True) -> None:
        self.results: list[CaseResult] = []
        self._verbose = verbose

    def record(
        self,
        case: MatrixCase,
        status: "RunStatus | str",
        wall_time: float,
        notes: str = "",
        metrics: dict[str, float] | None = None,
    ) -> CaseResult:
        status = RunStatus(status)
        result = CaseResult(case, status, wall_time, notes, metrics or {})
        self.results.append(result)
        if self._verbose:
            print(
                f"  {status.icon} {status.value:5s} | {case.label:<55s} | "
                f"{wall_time:7.1f}s | {notes}"
            )
        return result

    def counts(self) -> dict[str, int]:
        c = {s.value: 0 for s in RunStatus}
        for r in self.results:
            c[r.status.value] += 1
        return c

    @property
    def n_pass(self) -> int:
        return self.counts()[RunStatus.PASS.value]

    @property
    def n_fail(self) -> int:
        return self.counts()[RunStatus.FAIL.value]

    @property
    def ok(self) -> bool:
        """True iff no case FAILed or ERRORed (SKIP is allowed)."""
        c = self.counts()
        return c[RunStatus.FAIL.value] == 0 and c[RunStatus.ERROR.value] == 0
