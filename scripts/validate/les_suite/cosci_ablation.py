"""Round-2 ablation probe: score one candidate with parameter overrides and/or
asserted source edits (applied to a temp copy; the candidate file is unchanged) on
CloudBench cases, one rollout per case through cosci_eval's Arm and _scored_entry
(E2 + conservation gate), so the scores are those of the round-1 protocol.

    python scripts/validate/les_suite/cosci_ablation.py --candidate C.py --label L \
        --set a_N=0 --edit 'old|||new' --artifacts ... --out out.json \
        --cloudbench-forcing DIR --cloudbench-stats FILE
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import shutil
import tempfile
from pathlib import Path

import jax

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "scripts" / "validate" / "les_suite"))
import cosci_eval as ce  # noqa: E402
from legoesm.atmosphere.les_suite.candidate import load_candidate  # noqa: E402

#: the round-1 CloudBench protocol (PI 2026-10-27 .. 11-11)
BASELINE, HEAT_FORM, KE_HEATING, NLEV, DT_S = "ysu", "enthalpy", True, 32, 600.0


def ablated_source(src: str, edits: list[str]) -> str:
    """Apply each ``old|||new`` edit; each ``old`` must occur exactly once."""
    for e in edits:
        old, sep, new = e.partition("|||")
        if not sep:
            raise SystemExit(f"edit needs old|||new, got {e!r}")
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"edit target found {n} times, need exactly 1: {old!r}")
        src = src.replace(old, new)
    return src


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--set", action="append", default=[], help="param=value override")
    p.add_argument("--edit", action="append", default=[], help="old|||new source edit")
    p.add_argument("--artifacts", nargs="+", type=Path, required=True)
    p.add_argument("--cloudbench-forcing", type=Path, required=True)
    p.add_argument("--cloudbench-stats", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if not jax.config.read("jax_enable_x64"):
        raise SystemExit("run with JAX_ENABLE_X64=1")
    src = a.candidate.read_text()
    ran = src
    sib_sha = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(a.candidate.parent.glob("*.py"))}
    with tempfile.TemporaryDirectory() as tmpd:   # load AND score inside its scope
        path = a.candidate
        if a.edit:
            # siblings come along, so a candidate that loads its base beside __file__
            # still finds it
            for sib in a.candidate.parent.glob("*.py"):
                shutil.copy(sib, Path(tmpd) / sib.name)
            ran = ablated_source(src, a.edit)
            path = Path(tmpd) / a.candidate.name
            path.write_text(ran)
        cand = load_candidate(path)
        params = cand.default_params()
        for kv in a.set:
            k, _, v = kv.partition("=")
            try:
                val = float(v)
            except ValueError:
                raise SystemExit(f"bad override {kv!r}") from None
            if k not in params:
                raise SystemExit(f"unknown parameter {k!r} (parameters: {sorted(params)})")
            params[k] = jax.numpy.asarray(val)
        arts = [(str(x), ce.load_artifact(x)) for x in a.artifacts]
        cb_prov = ce.configure_cloudbench(arts, a.cloudbench_forcing, a.cloudbench_stats)
        arm = ce.Arm(ce._roster_base(BASELINE)[0], cand, HEAT_FORM, KE_HEATING)
        scores = ce._case_scores(arm, arts, NLEV, DT_S, params=params)
    split = {Path(pth).name: {"held_out": ce._held_out(art.case_name),
                              "regime": ce._regime(art.case_name)} for pth, art in arts}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "label": a.label, "scores": scores, "split": split,
        "summary": ce.failure_summary(scores, split),
        "provenance": {"git_sha": ce._git_sha(), "code_id": ce._code_id(), "argv": list(sys.argv[1:] if argv is None else argv),
                       "candidate_sha256": hashlib.sha256(src.encode()).hexdigest(),
                       "ran_sha256": hashlib.sha256(ran.encode()).hexdigest(),
                       "sibling_sha256": sib_sha,
                       "params": {k: float(v) for k, v in params.items()},
                       "baseline": BASELINE, "heat_form": HEAT_FORM, "ke_heating": KE_HEATING,
                       "nlev": NLEV, "dt": DT_S, "cloudbench": cb_prov,
                       "artifacts": {pth: ce._sha256(pth) for pth, _ in arts}}},
        indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
