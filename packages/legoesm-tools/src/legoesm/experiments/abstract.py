"""The ``Experiment`` contract (Stage A2 / design Phase 0).

An ``Experiment`` is the executable unit of the §6 verification matrices: it
``run()``s to produce named metrics, then ``evaluate()``s each metric against a
:class:`MetricCheck` (a reference value + tolerance + comparison kind).  The
result is a single pass/fail verdict with a per-metric report, so a regression
localises to a metric, which localises to a brick.

A ``status`` field implements the matrix-cell lifecycle: ``"active"`` cells gate
CI; ``"proposed"`` cells run but do not gate (they have no agreed reference yet).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import NamedTuple


class MetricCheck(NamedTuple):
    """A reference + tolerance for one metric.

    ``kind``:
      * ``"below"`` — pass if ``value <= reference + atol`` (error/residual stays
        under a ceiling; the common dycore/conservation case);
      * ``"above"`` — pass if ``value >= reference - atol``;
      * ``"close"`` — pass if ``value`` is within ``atol``/``rtol`` of ``reference``
        (``math.isclose`` semantics).
    """

    reference: float
    kind: str = "below"
    atol: float = 0.0
    rtol: float = 0.0

    def evaluate(self, value: float) -> tuple[bool, str]:
        """Return ``(passed, human-readable message)`` for *value*."""
        v, ref = float(value), float(self.reference)
        if self.kind == "below":
            passed = v <= ref + self.atol
            rel = f"{v:.3e} <= {ref:.3e}" + (f"+{self.atol:.1e}" if self.atol else "")
        elif self.kind == "above":
            passed = v >= ref - self.atol
            rel = f"{v:.3e} >= {ref:.3e}" + (f"-{self.atol:.1e}" if self.atol else "")
        elif self.kind == "close":
            passed = math.isclose(v, ref, rel_tol=self.rtol, abs_tol=self.atol)
            rel = f"{v:.3e} ≈ {ref:.3e} (rtol={self.rtol:g}, atol={self.atol:g})"
        else:
            raise ValueError(
                f"MetricCheck.kind must be 'below'/'above'/'close', got {self.kind!r}"
            )
        return passed, ("PASS " if passed else "FAIL ") + rel


class ExperimentResult(NamedTuple):
    """Outcome of :meth:`Experiment.evaluate`."""

    name: str
    passed: bool
    metrics: dict[str, float]
    report: dict[str, str]  # metric name -> PASS/FAIL message

    def summary(self) -> str:
        head = f"{'PASS' if self.passed else 'FAIL'} {self.name}"
        lines = [f"  {k}: {self.report[k]}" for k in sorted(self.report)]
        return "\n".join([head, *lines])


class Experiment(ABC):
    """A reproducible, self-describing characterization case.

    Subclasses implement :meth:`run` (compute named metrics) and :attr:`checks`
    (the reference + tolerance for each gated metric).  ``evaluate()`` ties them
    together.  Every metric named in :attr:`checks` must be produced by
    :meth:`run`; extra metrics from ``run`` are recorded but not gated.
    """

    #: Short identifier (matrix-cell name).
    name: str = "experiment"
    #: One-line human description.
    description: str = ""
    #: ``"active"`` (gates CI) or ``"proposed"`` (runs, no gate — no reference yet).
    status: str = "active"

    @abstractmethod
    def run(self) -> dict[str, float]:
        """Run the case and return ``{metric_name: value}``."""
        ...

    @property
    @abstractmethod
    def checks(self) -> dict[str, MetricCheck]:
        """``{metric_name: MetricCheck}`` for the metrics this case gates on."""
        ...

    def evaluate(self) -> ExperimentResult:
        """Run, then check every gated metric against its reference."""
        if self.status not in ("active", "proposed"):
            raise ValueError(
                f"Experiment.status must be 'active' or 'proposed', "
                f"got {self.status!r}"
            )
        metrics = self.run()
        checks = self.checks
        # Fail CLOSED: an ACTIVE (gating) cell that defines no checks would
        # otherwise pass while gating nothing — a green cell that guards nothing.
        if self.status == "active" and not checks:
            return ExperimentResult(
                name=self.name,
                passed=False,
                metrics=dict(metrics),
                report={
                    "<checks>": "FAIL active experiment defines no MetricChecks "
                    "(it would gate no metric)"
                },
            )
        report: dict[str, str] = {}
        all_passed = True
        for metric_name, check in checks.items():
            if metric_name not in metrics:
                report[metric_name] = f"FAIL run() did not produce metric {metric_name!r}"
                all_passed = False
                continue
            passed, msg = check.evaluate(metrics[metric_name])
            report[metric_name] = msg
            all_passed = all_passed and passed
        return ExperimentResult(
            name=self.name,
            passed=all_passed,
            metrics=dict(metrics),
            report=report,
        )
