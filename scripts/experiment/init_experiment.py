#!/usr/bin/env python
"""Materialize a runnable experiment directory from a named template — the
lesommer ``init_experiment`` ergonomics, layered on legoESM's ``legoesm run``.

Resolves a versioned template (``config/templates/<category>/<name>.yaml``) plus
``-o dot.notation=value`` overrides plus a machine profile into a self-contained
output directory:

    config.yaml   resolved, runnable config (fed to `legoesm run`)
    run.yaml      intent record: template, overrides, machine, created-at
    run.sh        launcher (sets JAX_PLATFORMS, activates venv, `legoesm run config.yaml`)

The resolved config is strict-validated (``to_experiment_config().validate_strict()``)
BEFORE anything is written, so a bad override fails fast with no half-built dir.

Reproducibility is NOT reimplemented here: the run itself writes
``run_manifest.json`` (resolved config + state_digest + git_hash) under its output
dir, and ``legoesm reproduce <manifest> --check`` does a bit-identical re-run —
a superset of the lesommer ``experiment.tag``.

Examples::

    python scripts/experiment/init_experiment.py 2d/williamson2_sw \\
        --name w2 --output-dir ./runs/w2
    python scripts/experiment/init_experiment.py 2d/williamson2_sw \\
        --name w2hi --output-dir ./runs/w2hi \\
        -o grid.resolution=96 -o time.duration_hours=240
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATES_DIR = _REPO_ROOT / "config" / "templates"

# Same-dir sibling helper (scripts/experiment is on sys.path[0] when run directly).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lego_detect_machine import resolve_machine  # noqa: E402


def _coerce(value: str) -> Any:
    """Coerce a CLI override string to bool/int/float/None/str (in that order)."""
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def _parse_overrides(pairs: list[str]) -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"ERROR: override {p!r} must be key=value (dot notation)")
        key, raw = p.split("=", 1)
        out.append((key.strip(), _coerce(raw)))
    return out


def _template_path(template: str) -> Path:
    rel = template[:-5] if template.endswith(".yaml") else template
    path = _TEMPLATES_DIR / f"{rel}.yaml"
    if not path.is_file():
        avail = sorted(
            str(p.relative_to(_TEMPLATES_DIR))[:-5]
            for p in _TEMPLATES_DIR.rglob("*.yaml") if not p.name.startswith("_")
        )
        raise SystemExit(
            f"ERROR: template {template!r} not found at {path}\n"
            f"  available: {', '.join(avail) or '(none)'}"
        )
    return path


def _render_run_sh(machine: dict[str, Any], config_rel: str) -> str:
    plat = machine.get("jax_platforms", "cpu")
    venv = machine.get("venv_activate", "") or ""
    sched = machine.get("scheduler", "none")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    if sched == "slurm":
        s = machine.get("slurm", {}) or {}
        lines.append(f"#SBATCH --job-name=legoesm")
        if s.get("partition"):
            lines.append(f"#SBATCH --partition={s['partition']}")
        if s.get("nodes"):
            lines.append(f"#SBATCH --nodes={s['nodes']}")
        if s.get("gpus"):
            lines.append(f"#SBATCH --gpus={s['gpus']}")
        if s.get("time"):
            lines.append(f"#SBATCH --time={s['time']}")
        if s.get("account"):
            lines.append(f"#SBATCH --account={s['account']}")
        lines.append("")
    lines.append('cd "$(dirname "$0")"')
    if venv:
        lines.append(venv)
    lines.append(f"export JAX_PLATFORMS={plat}")
    lines.append(f"legoesm run {config_rel}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("template", help="template id, e.g. 2d/williamson2_sw")
    p.add_argument("--name", required=True, help="experiment nickname")
    p.add_argument("--output-dir", required=True, help="directory to create")
    p.add_argument("-o", "--override", action="append", default=[],
                   metavar="dot.key=value", help="override a config field (repeatable)")
    p.add_argument("--machine", default=None,
                   help="machine profile name (default: auto-detect by hostname)")
    args = p.parse_args(argv)

    from legoesm.config import Config

    template_path = _template_path(args.template)
    overrides = _parse_overrides(args.override)

    # Machine profile (explicit > auto-detected).
    if args.machine:
        from lego_detect_machine import load_machine
        machine = load_machine(args.machine)
    else:
        machine = resolve_machine()

    cfg = Config.from_yaml(str(template_path))

    # Machine-derived default precision, only if the user did not override it.
    if not any(k == "hardware.precision.dynamics" for k, _ in overrides):
        if machine.get("precision"):
            cfg.set("hardware.precision.dynamics", machine["precision"])

    for key, value in overrides:
        cfg.set(key, value)

    # Strict-validate BEFORE writing anything (fail fast, no half-built dir).
    try:
        cfg.to_experiment_config().validate_strict()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ERROR: resolved config is invalid: {type(exc).__name__}: {exc}")

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"ERROR: --output-dir {out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)

    cfg.to_yaml(str(out / "config.yaml"))
    (out / "run.sh").write_text(_render_run_sh(machine, "config.yaml"))
    (out / "run.sh").chmod(0o755)

    meta = cfg.get("experiment") or {}
    run_yaml = {
        "experiment": {
            "name": args.name,
            "template": args.template,
            "tier": meta.get("tier"),
            "complexity": meta.get("complexity"),
            "extent": meta.get("extent"),
            "created": datetime.now(timezone.utc).isoformat(),
        },
        "machine": machine.get("name", "default"),
        "overrides": {k: v for k, v in overrides},
        "data_required": meta.get("data", []),
    }
    (out / "run.yaml").write_text(yaml.safe_dump(run_yaml, sort_keys=False))

    print(f"  Initialized experiment '{args.name}' in {out}")
    print(f"    template: {args.template}  machine: {machine.get('name')}")
    if overrides:
        print(f"    overrides: {json.dumps(dict(overrides))}")
    if meta.get("data"):
        print(f"    data required: {meta['data']} "
              f"(run scripts/experiment/fetch_data.py to stage)")
    print(f"  Run it:        cd {out} && bash run.sh")
    print(f"  Reproduce it:  legoesm reproduce {out}/<output>/run_manifest.json --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
