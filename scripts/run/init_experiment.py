#!/usr/bin/env python
"""Instantiate an LMIP experiment from a template.

Reads ``templates/land/<template>.yaml``, applies ``-o key.path=value``
overrides, validates the merged config, and writes a self-contained
experiment directory::

    <output-dir>/
        run.yaml           — declaration: which template + overrides
        config.yaml        — resolved config (what run_lmip_biophys consumes)
        experiment.tag     — INI-style provenance (legoESM SHA, Py/JAX versions)
        run.sh             — one-shot bash wrapper that runs the driver

Example::

    python scripts/run/init_experiment.py biophysics/spinup_5year \\
        --name derecho_2026-07-01_5yr_spinup \\
        --output-dir $SCRATCH/lmip/derecho_2026-07-01_5yr_spinup \\
        -o forcing.year_start=1980 -o forcing.year_end=1984 \\
        -o physics.bulk_scheme=constant
"""

from __future__ import annotations

import argparse
import configparser
import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from legoesm.land.lmip_config import apply_overrides, validate_config  # noqa: E402


def _templates_dir() -> Path:
    return _ROOT / "templates" / "land"


def _machines_dir() -> Path:
    return _ROOT / "configs" / "machines"


def _load_machine(name: str) -> dict:
    """Load a machine profile YAML — e.g. configs/machines/derecho.yaml."""
    import yaml
    path = _machines_dir() / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in _machines_dir().glob("*.yaml"))
        raise SystemExit(
            f"machine {name!r} not found at {path}\n"
            f"available machines: {available or '(none configured yet)'}"
        )
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_template(name: str) -> dict:
    import yaml
    path = _templates_dir() / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.relative_to(_templates_dir()).with_suffix("")
                           for p in _templates_dir().rglob("*.yaml"))
        raise SystemExit(
            f"template {name!r} not found at {path}\n"
            "available templates:\n  " + "\n  ".join(str(p) for p in available)
        )
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(_ROOT), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


def _jax_version() -> str:
    try:
        import jax
        return jax.__version__
    except Exception:                                    # noqa: BLE001
        return "unknown"


def _config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()[:16]


def _write_experiment_tag(path: Path, *, name: str, template: str,
                          cfg: dict, machine_name: str = "") -> None:
    tag = configparser.ConfigParser()
    tag["experiment"] = {
        "name": name,
        "template": template,
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    tag["legoESM"] = {
        "ref": _git_branch(),
        "commit": _git_sha(),
    }
    tag["model"] = {
        "land_mode": cfg["physics"]["land_mode"],
        "surface_scheme": cfg["physics"]["surface_scheme"],
        "bulk_scheme": cfg["physics"]["bulk_scheme"],
        "grid_type": cfg["grid"]["type"],
        "resolution": str(cfg["grid"]["resolution"]),
    }
    tag["forcing"] = {
        "source": cfg["forcing"]["source"],
        "year_start": str(cfg["forcing"]["year_start"]),
        "year_end": str(cfg["forcing"]["year_end"]),
    }
    tag["run"] = {
        "duration_steps": str(cfg["time"]["n_steps"]),
        "dt_seconds": str(cfg["time"]["dt"]),
    }
    tag["machine"] = {
        "hostname": platform.node(),
        "profile": machine_name or "none",
    }
    tag["reproducibility"] = {
        "config_hash": _config_hash(cfg),
        "python_version": platform.python_version(),
        "jax_version": _jax_version(),
    }
    with open(path, "w") as f:
        tag.write(f)


def _pbs_headers(machine: dict) -> str:
    """Emit PBS directive block from a machine profile.  NEVER emits `#PBS -A`
    (project account) — the user supplies that on the qsub line so run.sh stays
    portable across projects/users."""
    # Dispatch hardening: only an explicit 'local' (or absent) scheduler means
    # "no directives".  An unrecognized scheduler (typo, 'slurm', 'PBS') must
    # RAISE — otherwise it silently emits a header-less run.sh that the user
    # submits to run inline on the login node.
    sched = machine.get("scheduler", "local")
    if sched in ("local", "", None):
        return ""
    if sched != "pbs":
        raise ValueError(
            f"unknown machine scheduler {sched!r} (expected 'pbs' or 'local')")
    p = machine.get("pbs", {})
    lines = ["# --- PBS directives (submit with `qsub -A <ACCT> run.sh`) ---"]
    for key, flag in (("job_name", "-N"), ("queue", "-q"), ("join", "-j")):
        if key in p:
            lines.append(f"#PBS {flag} {p[key]}")
    if "select" in p:
        lines.append(f"#PBS -l select={p['select']}")
    if "walltime" in p:
        lines.append(f"#PBS -l walltime={p['walltime']}")
    return "\n".join(lines) + "\n\n"


def _write_run_sh(path: Path, config_path: Path, output_dir: Path,
                  machine: dict | None = None) -> None:
    driver = _ROOT / "scripts" / "run" / "run_lmip_biophys.py"
    # sys.executable was the interpreter used to init.  On a machine profile
    # (--machine), env_setup overrides LEGOESM_PYTHON via `which python` after
    # `conda activate`, so re-init on the target machine isn't strictly needed
    # (though CWD absolutisation of surfdata/data paths still matters).
    pbs = _pbs_headers(machine or {})
    env_setup = (machine or {}).get("env_setup", "")
    if env_setup and not env_setup.endswith("\n"):
        env_setup += "\n"
    env_block = (
        f"# --- Machine env setup (from configs/machines/) ---\n{env_setup}\n"
        if env_setup else "")
    body = (
        "#!/bin/bash\n"
        f"{pbs}"
        "# Auto-generated by init_experiment.py.  Reproduce this experiment:\n"
        "#   git checkout $(sed -n 's/^commit = //p' experiment.tag)\n"
        "#   qsub -A <ACCT> run.sh    # PBS  (--machine derecho generated)\n"
        "#   bash run.sh              # local\n\n"
        f"{env_block}"
        "set -euo pipefail\n"
        "# PBS copies the batch script to a spool dir before running, so\n"
        "# `dirname $0` points to the spool — not the submit dir.  Use\n"
        "# PBS_O_WORKDIR (set by PBS to the qsub directory) when available;\n"
        "# fall back to script's own dir for a plain `bash run.sh`.\n"
        'cd "${PBS_O_WORKDIR:-$(dirname "$0")}"\n'
        "export JAX_ENABLE_X64=1\n"
        f'PYTHON_BIN="${{LEGOESM_PYTHON:-{sys.executable}}}"\n'
        f'"$PYTHON_BIN" "{driver}" \\\n'
        f'    --config "{config_path.name}" \\\n'
        f'    --output-dir "{output_dir}"\n'
    )
    path.write_text(body)
    path.chmod(0o755)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("template", help="template name (e.g. biophysics/spinup_5year)")
    ap.add_argument("--name", required=True, help="experiment nickname")
    ap.add_argument("--output-dir", required=True, help="experiment directory to create")
    ap.add_argument("-o", "--override", action="append", default=[],
                    metavar="key.path=value",
                    help="override a config field (repeatable; YAML-parsed values)")
    ap.add_argument("--machine", default="",
                    help="machine profile from configs/machines/<name>.yaml "
                         "(e.g. 'derecho'); adds scheduler headers + env setup "
                         "to run.sh.  Project account is NEVER emitted — pass "
                         "on qsub: `qsub -A <ACCT> run.sh`.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved config to stdout, write nothing")
    args = ap.parse_args()

    import yaml

    base = _load_template(args.template)
    merged = apply_overrides(base, args.override)
    cfg = validate_config(merged).raw                    # raw = the dict we just validated

    # Absolutize path fields against the CWD at init time so config.yaml is a
    # self-contained snapshot — the generated run.sh cd's into the experiment
    # dir, and relative paths would break there.
    cwd = Path.cwd()
    for section, field in (("surfdata", "path"),
                            ("forcing", "data_dir"),
                            ("restart", "from"),
                            (None, "land_mask_file")):
        d = cfg if section is None else cfg.setdefault(section, {})
        val = d.get(field, "")
        if val and not Path(val).is_absolute():
            d[field] = str((cwd / val).resolve())

    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. run.yaml — declaration only (template + overrides)
    (out_dir / "run.yaml").write_text(yaml.safe_dump({
        "template": args.template,
        "name": args.name,
        "overrides": list(args.override),
    }, sort_keys=False))

    # 2. config.yaml — resolved config (what the driver consumes)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # 3. experiment.tag — provenance
    _write_experiment_tag(out_dir / "experiment.tag",
                          name=args.name, template=args.template, cfg=cfg,
                          machine_name=args.machine)

    # 4. run.sh — one-shot wrapper (PBS-headered if --machine sets a scheduler)
    machine = _load_machine(args.machine) if args.machine else None
    _write_run_sh(out_dir / "run.sh", out_dir / "config.yaml", out_dir,
                  machine=machine)

    print(f"initialised experiment at {out_dir}")
    for f in ("run.yaml", "config.yaml", "experiment.tag", "run.sh"):
        print(f"  {f}")
    if machine and machine.get("scheduler") == "pbs":
        print(f"\nnext step:  cd {out_dir} && qsub -A <YOUR_ACCOUNT> run.sh")
    else:
        print(f"\nnext step:  cd {out_dir} && bash run.sh")


if __name__ == "__main__":
    main()
