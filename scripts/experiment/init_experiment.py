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
import shlex
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
        from legoesm.experiment_registry import retired_template_message
        retired = retired_template_message(template)
        if retired is not None:
            raise SystemExit(f"ERROR: {retired}")
        avail = sorted(
            str(p.relative_to(_TEMPLATES_DIR))[:-5]
            for p in _TEMPLATES_DIR.rglob("*.yaml") if not p.name.startswith("_")
        )
        raise SystemExit(
            f"ERROR: template {template!r} not found at {path}\n"
            f"  available: {', '.join(avail) or '(none)'}"
        )
    return path


def render_run_sh(
    machine: dict[str, Any],
    *,
    run_cmd: str,
    enable_x64: bool = False,
    workdir: str | None = None,
) -> str:
    """Render a ``run.sh`` launcher that sets the JAX env then runs ``run_cmd``.

    Single source of truth for the launcher shell, shared by ``init_experiment``
    (``run_cmd='legoesm run config.yaml'``) and the interactive ``wizard`` (which
    routes to ``run_coupled``, the ocean/SCM matrices, the plane-LES scripts, or
    training — none of which are ``legoesm run``).  ``enable_x64`` adds
    ``export JAX_ENABLE_X64=1`` for 64-bit runs.  ``workdir`` overrides the launch
    directory (default: the bundle dir, ``$(dirname "$0")``) — needed when the
    command references repo-root-relative paths (e.g. the AIMIP suite manifest).
    Honors the machine profile's ``jax_platforms`` / ``venv_activate`` / SLURM
    directives just as before.
    """
    plat = machine.get("jax_platforms", "cpu")
    venv = machine.get("venv_activate", "") or ""
    sched = machine.get("scheduler", "none")
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    if sched == "slurm":
        s = machine.get("slurm", {}) or {}
        lines.append("#SBATCH --job-name=legoesm")
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
    lines.append(f"cd {shlex.quote(workdir)}" if workdir else 'cd "$(dirname "$0")"')
    if venv:
        lines.append(venv)
    lines.append(f"export JAX_PLATFORMS={plat}")
    if enable_x64:
        lines.append("export JAX_ENABLE_X64=1")
    lines.append(run_cmd)
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

    from legoesm import experiment_registry

    template_path = _template_path(args.template)
    overrides = _parse_overrides(args.override)

    # Machine profile (explicit > auto-detected).
    if args.machine:
        from lego_detect_machine import load_machine
        machine = load_machine(args.machine)
    else:
        machine = resolve_machine()

    # Mode-aware adapter dispatch (atmosphere Config / ocean OceanExperimentConfig
    # / future land+ice) via the registry — the meta-package's single extensible
    # entry point. The adapter implements the uniform protocol used below.
    mode, cfg = experiment_registry.load_adapter(str(template_path))
    is_atm = experiment_registry.is_atmosphere_mode(mode)

    # Machine-derived default precision, only if the user did not override it.
    # Atmosphere-only knob (ocean runs are x64 throughout, set on the runner).
    # NOT for a ``setup:`` template (#388): it routes to a matrix runner that
    # does not consume config.yaml, and injecting hardware would trip the
    # setup-template "ignored section" guard.
    if (is_atm and cfg.get("setup") is None
            and not any(k == "hardware.precision.dynamics" for k, _ in overrides)):
        if machine.get("precision"):
            cfg.set("hardware.precision.dynamics", machine["precision"])

    # Apply overrides one at a time, asserting each actually changes the
    # RESOLVED config (codex review HIGH): a typo or non-runtime dot-path
    # (e.g. -o grid.resoluton=96) is preserved verbatim in config.yaml but
    # ignored by the runtime mapping, so it would silently no-op while being
    # recorded as applied. Comparing the resolved-config signature before/after
    # each set catches that.
    no_ops: list[str] = []
    for key, value in overrides:
        before = cfg.signature()
        cfg.set(key, value)
        if cfg.signature() == before:
            no_ops.append(f"{key}={value!r}")
    if no_ops:
        raise SystemExit(
            "ERROR: override(s) had no effect on the resolved config — likely a "
            "misspelled or non-runtime dot-path (or set to the existing value): "
            + ", ".join(no_ops)
        )

    # Strict-validate BEFORE writing anything (fail fast, no half-built dir).
    try:
        cfg.validate_strict()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ERROR: resolved config is invalid: {type(exc).__name__}: {exc}")

    # Also confirm the bundle has a runnable launcher BEFORE writing anything:
    # an ocean grid with no wired runner backend (cube/spectral) must fail here,
    # not after a half-built dir with a run.sh that dies at runtime.
    try:
        cfg.run_command("config.yaml")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"ERROR: no runnable launcher for this experiment: "
                         f"{type(exc).__name__}: {exc}")

    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"ERROR: --output-dir {out} exists and is not empty")
    out.mkdir(parents=True, exist_ok=True)

    # Ocean bundles launch from the repo root (the runner is repo-relative), so a
    # relative output.path would write under the repo root and collide across
    # bundles. Resolve a relative output.path UNDER the bundle (preserving the
    # template's subdir, e.g. output/omip_latlon/ -> <bundle>/output/omip_latlon)
    # and leave an absolute path untouched. ``out_subdir`` is where the run
    # manifest lands, used for the reproduce hint below.
    # A ``setup:`` template (#388) routes to a matrix runner (a repo-root-
    # relative script) rather than the atmosphere ``legoesm run`` CLI — so it
    # launches from the repo root just like the ocean recipe path, and its
    # relative ``output.path`` is resolved to an absolute bundle path so results
    # land in the bundle.
    is_setup = bool(cfg.get("setup"))
    out_subdir = out / "output"
    if not is_atm or is_setup:
        op = cfg.get("output.path")
        if isinstance(op, str) and Path(op).is_absolute():
            out_subdir = Path(op)
        else:
            rel = op if (isinstance(op, str) and op) else "output"
            out_subdir = (out / rel).resolve()
            cfg.set("output.path", str(out_subdir))

    cfg.to_yaml(str(out / "config.yaml"))
    # Ocean runs need 64-bit (omip is x64 throughout); atmosphere keeps its
    # template-/machine-driven precision (x32 default).  The atmosphere ``legoesm
    # run`` CLI is cwd-independent (cd into the bundle, relative config.yaml);
    # the ocean runner AND every matrix-runner setup template are repo-root-
    # relative scripts, so launch them from the repo root.
    if is_atm and not is_setup:
        run_cmd = cfg.run_command("config.yaml")
        workdir = None
    else:
        run_cmd = cfg.run_command(str((out / "config.yaml").resolve()))
        workdir = str(_REPO_ROOT)
    (out / "run.sh").write_text(
        render_run_sh(machine, run_cmd=run_cmd, enable_x64=not is_atm,
                      workdir=workdir)
    )
    (out / "run.sh").chmod(0o755)

    meta = cfg.get_meta()
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
    if is_setup:
        # Matrix-runner setup template: no run_manifest.json / reproduce path —
        # it runs an idealized matrix case and writes summary.json + per-case
        # artifacts under the bundle's output dir.
        print(f"  (setup template -> matrix runner)")
        print(f"    {run_cmd}")
        print(f"  Results:       {out_subdir} (summary.json + per-case artifacts)")
    elif is_atm:
        print(f"  Reproduce it:  legoesm reproduce {out}/<output>/run_manifest.json --check")
    else:
        # No `--check`: ocean reproduce-rerun is not wired through the atmosphere
        # driver, so --check would not perform a digest comparison. `reproduce`
        # (no --check) validates the manifest; re-run via the ocean runner to
        # compare result.state_digest.
        print(f"  Validate it:   legoesm reproduce {out_subdir}/run_manifest.json")
        print(f"  Reproduce it:  re-run via run_omip_core2.py --config "
              f"{out / 'config.yaml'} and compare result.state_digest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
