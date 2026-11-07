#!/usr/bin/env python3
"""Prove Decision-12's ORCA1 CORE2 driver edit leaves its resolved card fixed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
DRIVER = Path("scripts/run/run_omip_core2.py")
DECISION_COMMIT = "bb1c60beba4d"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=REPO)


def audit(plant: bool) -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        before_path = Path(tmp) / "run_omip_core2_before.py"
        before_path.write_bytes(_git("show", f"{DECISION_COMMIT}^:{DRIVER}"))
        before = _load(before_path, "orca2_phase2x_driver_before")
        after = _load(REPO / DRIVER, "orca2_phase2x_driver_after")
        before_cfg = before.build_tripole_vmix_config("tke")
        after_cfg = after.build_tripole_vmix_config("tke")
        if plant:
            after_cfg = after_cfg._replace(tke=after_cfg.tke._replace(eice=2))

    before_repr = repr(before_cfg)
    after_repr = repr(after_cfg)
    exact = before_repr == after_repr
    result = {
        "decision_commit": DECISION_COMMIT,
        "driver": str(DRIVER),
        "before_eice": int(before_cfg.tke.eice),
        "after_eice": int(after_cfg.tke.eice),
        "resolved_config_exact": exact,
        "before_repr_sha256": hashlib.sha256(before_repr.encode()).hexdigest(),
        "after_repr_sha256": hashlib.sha256(after_repr.encode()).hexdigest(),
        "decision_diff": _git("diff", f"{DECISION_COMMIT}^", DECISION_COMMIT,
                              "--", str(DRIVER)).decode().splitlines(),
    }
    if not exact:
        raise ValueError(json.dumps(result, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = audit(args.plant)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json:
        args.json.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
