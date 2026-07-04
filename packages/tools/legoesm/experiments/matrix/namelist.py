"""Per-test-case namelist parameter file (issue #682).

Writes a human-readable ``namelist.txt`` into each test-case output subfolder
recording the resolved case configuration — the ``TestCase`` that defines the
case (equation set, case name, grid, resolution, vertical coordinate, duration,
per-case ``run_kwargs``) plus any resolved run settings the caller passes as
``extra`` (radiation scheme, the days actually run, dt, ...).  A complete,
diffable record so a run can be inspected, compared, and reproduced.

Component-agnostic (stdlib only) so it stays on the ``legoesm-tools`` side of
the federation DAG, matching the rest of ``legoesm.experiments.matrix``.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

_NAMELIST_FILENAME = "namelist.txt"


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    """Flatten nested dict/dataclass values into dotted ``key: scalar`` pairs.

    ``run_kwargs={'test_num': 8}`` -> ``run_kwargs.test_num: 8``; a nested
    dataclass (e.g. a config NamedTuple/dataclass) expands the same way so the
    file lists every leaf parameter with its value.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    # NamedTuple -> dict (has _asdict); plain namedtuple support for configs.
    elif hasattr(value, "_asdict") and callable(value._asdict):
        value = value._asdict()
    if isinstance(value, dict):
        if not value:
            out[prefix] = "{}"
            return
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten(key, v, out)
    else:
        out[prefix] = value


def build_namelist(testcase: Any, *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the flat ``{parameter: value}`` namelist for a test case.

    ``testcase`` may be a dataclass (the per-runner ``TestCase``), a namedtuple
    config, or a plain dict.  ``extra`` (resolved run settings) is merged and
    takes precedence over inferred fields.
    """
    flat: dict[str, Any] = {}
    _flatten("", testcase, flat)
    if extra:
        for k, v in extra.items():
            _flatten(str(k), v, flat)
    return flat


def write_case_namelist(
    output_dir: str | Path,
    testcase: Any,
    *,
    title: str = "",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``namelist.txt`` into ``output_dir`` and return its path.

    Parameters
    ----------
    output_dir
        The case output subfolder (created if missing) — beside ``results.txt``.
    testcase
        The case descriptor (dataclass ``TestCase`` / namedtuple config / dict).
    title
        Optional header line (e.g. ``"atmosphere test-case namelist"``).
    extra
        Optional resolved run settings to record alongside the case fields.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = build_namelist(testcase, extra=extra)
    path = output_dir / _NAMELIST_FILENAME
    with open(path, "w") as f:
        if title:
            f.write(f"# {title}\n")
        f.write("# namelist parameter file (issue #682): resolved case configuration\n")
        for key in sorted(params):
            f.write(f"{key}: {params[key]}\n")
    return path
