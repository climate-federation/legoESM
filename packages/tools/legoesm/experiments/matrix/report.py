"""Summary emitters + regression detection for the test matrix.

Absorbs ``scripts/summarize_matrix_results.py`` and the report half of
``scripts/validate_matrix_report.py`` into one component-agnostic module: given a
:class:`~legoesm.experiments.matrix.core.ResultRecorder`, it writes
``summary.json`` / ``summary.txt`` / ``summary.md`` in the legacy schema (so
existing readers keep working) and diffs against a prior ``summary.json`` to flag
PASS→FAIL regressions.

Pure stdlib — no numpy, no component imports.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from legoesm.experiments.matrix.core import ResultRecorder, RunStatus


def _summary_dict(recorder: ResultRecorder, *, total_wall: float,
                  meta: dict[str, Any] | None) -> dict[str, Any]:
    c = recorder.counts()
    out: dict[str, Any] = {
        "results": [r.to_dict() for r in recorder.results],
        "total_wall_time": total_wall,
        "n_pass": c[RunStatus.PASS.value],
        "n_fail": c[RunStatus.FAIL.value],
        "n_skip": c[RunStatus.SKIP.value],
        "n_error": c[RunStatus.ERROR.value],
    }
    if meta:
        out.update(meta)
    return out


def write_summary(
    output_dir: "str | Path",
    recorder: ResultRecorder,
    *,
    total_wall: float,
    title: str = "legoESM Test Matrix",
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write ``summary.{json,txt,md}`` under ``output_dir``; return the JSON path.

    ``meta`` (e.g. ``{"quick_mode": True, "radiation": "gray"}``) is merged into
    the JSON top level — matches the per-domain extras the old runners stored.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _summary_dict(recorder, total_wall=total_wall, meta=meta)

    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    header = (
        f"Total: {len(recorder.results)} tests | "
        f"PASS: {summary['n_pass']} | FAIL: {summary['n_fail']} | "
        f"SKIP: {summary['n_skip']} | ERROR: {summary['n_error']}"
    )

    with open(output_dir / "summary.txt", "w") as f:
        f.write(f"{title} Summary\n")
        f.write("=" * 60 + "\n")
        f.write(header + "\n")
        f.write(f"Wall time: {total_wall:.1f}s ({total_wall / 60:.1f} min)\n\n")
        for r in recorder.results:
            d = r.to_dict()
            f.write(f"{d['status']:5}  {d['grid']:<14}  "
                    f"{d['complexity']:<16}  {d['test']:<24}  "
                    f"{d['wall_time']:7.1f}s  {d['notes']}\n")

    with open(output_dir / "summary.md", "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"**{header}** — wall {total_wall / 60:.1f} min\n\n")
        f.write("| Status | Tier | Component | Complexity | Case | Grid | Time | Notes |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in recorder.results:
            d = r.to_dict()
            note = d["notes"].replace("|", "\\|")
            f.write(f"| {d['status']} | {d['tier']} | {d['component']} | "
                    f"{d['complexity']} | {d['test']} | {d['grid']} | "
                    f"{d['wall_time']:.1f}s | {note} |\n")

    return json_path


def _result_key(rec: dict[str, Any]) -> tuple[str, ...]:
    """Stable case identity for regression matching.

    Must include EVERY dimension that distinguishes a :class:`MatrixCase`'s
    output path (codex review HIGH): keying on only (component, test, grid,
    resolution) collapses sibling cases that differ by ``complexity`` or
    ``vertical_coord`` (e.g. the same case/grid/res run hydrostatic vs
    nonhydrostatic, or sigma vs hybrid), so one sibling's status would clobber
    the other's in the prior-status map and a real PASS->FAIL could go
    unreported.
    """
    return (
        rec.get("component", ""),
        rec.get("complexity", ""),
        rec.get("test", ""),
        rec.get("grid", ""),
        rec.get("resolution", ""),
        rec.get("vertical_coord", ""),
    )


def detect_regressions(
    current: ResultRecorder,
    previous_json: "str | Path",
) -> list[str]:
    """Return human-readable PASS→FAIL/ERROR regression lines vs a prior summary.

    A case that PASSed (or was absent) in ``previous_json`` and now FAILs/ERRORs
    is a regression.  Missing prior file → empty list (nothing to compare).
    """
    previous_json = Path(previous_json)
    if not previous_json.exists():
        return []
    prev = json.loads(previous_json.read_text())
    prev_status = {_result_key(r): r.get("status") for r in prev.get("results", [])}

    regressions: list[str] = []
    for r in current.results:
        d = r.to_dict()
        was = prev_status.get(_result_key(d))
        if r.status in (RunStatus.FAIL, RunStatus.ERROR) and was == "PASS":
            # Include complexity + vcoord so sibling cases sharing the partial
            # key are distinguishable in the report (matches _result_key).
            cx = f"/{d['complexity']}" if d.get("complexity") else ""
            vc = (f"/{d['vertical_coord']}"
                  if d.get("vertical_coord") not in (None, "", "none") else "")
            regressions.append(
                f"REGRESSION {d['component']}{cx}/{d['test']}/{d['grid']}"
                f"{vc}: was PASS, now {r.status.value} — {d['notes']}"
            )
    return regressions
