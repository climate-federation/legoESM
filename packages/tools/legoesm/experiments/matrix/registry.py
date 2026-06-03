"""``MatrixRunner`` — the base a per-component matrix script subclasses.

A runner script in ``scripts/matrix/`` becomes a thin file: it declares its
:class:`~legoesm.experiments.matrix.core.MatrixCase` list (importing its own
component) and implements :meth:`run_case`.  Everything else — tier/grid/only
filtering, ``--list``, the run loop with ERROR capture, ``ResultRecorder``
accumulation, summary writing, regression detection, and the process exit code —
is inherited from here.

This base imports no component package; the subclass does.  That keeps the
framework on the ``legoesm-tools`` side of the federation DAG while the
component-importing registries live in ``scripts/matrix/``.
"""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Sequence

from legoesm.experiments.matrix.core import (
    MatrixCase,
    ResultRecorder,
    RunStatus,
    Tier,
)
from legoesm.experiments.matrix.report import detect_regressions, write_summary


class MatrixRunner:
    """Subclass per component: set :attr:`component`, override :meth:`build_cases`
    and :meth:`run_case`.

    Minimal subclass::

        class OceanMatrix(MatrixRunner):
            component = "ocean"

            def build_cases(self):
                return [MatrixCase("ocean", "rest_state", "cubed_sphere",
                                   tier=1, complexity="full_3d", ...), ...]

            def run_case(self, case, *, quick, output_dir):
                # integrate; compute conservation series; chain gates
                ok, notes = True, ""
                ok, notes = mass_gate(ok, notes, volume_series, component="ocean")
                return (RunStatus.PASS if ok else RunStatus.FAIL), notes, {}

        if __name__ == "__main__":
            raise SystemExit(OceanMatrix().main())
    """

    component: str = ""
    title: str = "legoESM Test Matrix"
    #: Default artifact root; ``--output`` overrides.
    default_output: str = "results"

    def __init__(self) -> None:
        if not self.component:
            raise ValueError(f"{type(self).__name__} must set .component")
        self._cases: list[MatrixCase] | None = None

    # --- subclass hooks ----------------------------------------------------

    def build_cases(self) -> list[MatrixCase]:
        """Return the full case list.  Override in the subclass."""
        raise NotImplementedError

    def run_case(
        self,
        case: MatrixCase,
        *,
        quick: bool,
        output_dir: Path,
    ) -> tuple[RunStatus, str, dict[str, float]]:
        """Run one case; return ``(status, notes, metrics)``.

        The subclass integrates the model, computes conservation series via
        ``legoesm.diagnostics`` and chains the
        :mod:`legoesm.experiments.matrix.gates`.  Raising propagates as ERROR
        (captured by :meth:`main`) — let real bugs surface rather than swallow.
        """
        raise NotImplementedError

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Hook for component-specific CLI flags (e.g. ``--radiation``)."""

    # --- shared machinery --------------------------------------------------

    @property
    def cases(self) -> list[MatrixCase]:
        if self._cases is None:
            self._cases = self.build_cases()
        return self._cases

    def filter(
        self,
        cases: Sequence[MatrixCase],
        *,
        tier: Tier | None = None,
        max_tier: Tier | None = None,
        grid: str | None = None,
        only: str | None = None,
    ) -> list[MatrixCase]:
        out = list(cases)
        if tier is not None:
            out = [c for c in out if c.tier == tier]
        if max_tier is not None:
            out = [c for c in out if c.tier <= max_tier]
        if grid is not None:
            out = [c for c in out if c.grid_type == grid]
        if only is not None:
            out = [c for c in out if only in (c.case, c.complexity, c.label)
                   or only in c.case]
        return out

    def build_parser(self) -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description=f"{self.title} ({self.component})")
        p.add_argument("--output", default=self.default_output,
                       help="artifact root directory")
        p.add_argument("--quick", action="store_true",
                       help="short durations (quick_days) for a smoke pass")
        p.add_argument("--only", default=None,
                       help="run only cases matching this case/complexity name")
        p.add_argument("--grid", default=None, help="restrict to one grid type")
        p.add_argument("--tier", default=None,
                       help="run only this tier (0..3 / unit/research/...)")
        p.add_argument("--max-tier", default=None,
                       help="run tiers up to and including this one")
        p.add_argument("--list", action="store_true",
                       help="list selected cases and exit (no integration)")
        p.add_argument("--allow-empty", action="store_true",
                       help="permit a zero-case selection instead of failing "
                            "(default: an empty selection is an error so a typo "
                            "in --grid/--only/--tier cannot silently pass CI)")
        self.add_arguments(p)
        return p

    def main(self, argv: Sequence[str] | None = None) -> int:
        """Parse args, run the selected cases, write the summary; return exit code.

        Exit code is non-zero iff any case FAILed/ERRORed or a regression was
        detected — so CI gates on the process status directly.
        """
        args = self.build_parser().parse_args(argv)
        tier = Tier.parse(args.tier) if args.tier is not None else None
        max_tier = Tier.parse(args.max_tier) if args.max_tier is not None else None
        selected = self.filter(self.cases, tier=tier, max_tier=max_tier,
                               grid=args.grid, only=args.only)

        if args.list:
            for c in selected:
                print(f"  tier{int(c.tier)} {c.label:<55s} "
                      f"[{c.maturity}]  ({c.resolution})")
            print(f"\n  {len(selected)} case(s) selected of {len(self.cases)}.")
            return 0

        # Fail-fast on an empty selection (codex review HIGH-1): a typo in
        # --grid/--only/--tier would otherwise write a zero-test summary and
        # return 0 (recorder.ok is vacuously True with no FAIL/ERROR), silently
        # passing CI with no coverage.  --allow-empty opts out for the rare
        # legitimately-empty filter.
        if not selected and not args.allow_empty:
            print(
                f"  ERROR: 0 of {len(self.cases)} cases selected for "
                f"'{self.component}' "
                f"(tier={args.tier}, max_tier={args.max_tier}, "
                f"grid={args.grid!r}, only={args.only!r}). "
                f"Check the filters, or pass --allow-empty if intended."
            )
            return 2

        output_base = Path(args.output) / self.component
        recorder = ResultRecorder()
        t0 = time.time()
        for case in selected:
            case_dir = output_base / case.output_path
            cstart = time.time()
            try:
                status, notes, metrics = self.run_case(
                    case, quick=args.quick, output_dir=case_dir
                )
            except Exception as exc:  # noqa: BLE001 — capture as ERROR, keep going
                status, notes, metrics = (
                    RunStatus.ERROR,
                    f"{type(exc).__name__}: {exc}",
                    {},
                )
                traceback.print_exc()
            recorder.record(case, status, time.time() - cstart, notes, metrics)

        total_wall = time.time() - t0
        regressions = detect_regressions(recorder, output_base / "summary.json")
        json_path = write_summary(
            output_base, recorder, total_wall=total_wall,
            title=f"{self.title} — {self.component}",
            meta={"quick_mode": args.quick},
        )
        c = recorder.counts()
        print(f"\n  Total: {len(recorder.results)} | PASS: {c['PASS']} | "
              f"FAIL: {c['FAIL']} | SKIP: {c['SKIP']} | ERROR: {c['ERROR']} | "
              f"wall {total_wall / 60:.1f} min")
        print(f"  Summary: {json_path}")
        for line in regressions:
            print(f"  {line}")
        return 0 if (recorder.ok and not regressions) else 1
