"""Probe: does a candidate's E2-loss gradient agree with central finite differences?

Per case (not regime-averaged), prints the loss, the AD gradient and the central
difference (L(p+h) - L(p-h)) / 2h with h = REL_STEP * |p| for each named parameter,
using the same jitted loss the E0 gradient check and E2 tuning use.

usage: JAX_ENABLE_X64=1 python scripts/tmp/cosci_grad_fd_probe.py CANDIDATE.py BASELINE ART.npz [ART.npz ...] [--params a,b]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
from legoesm.atmosphere.les_suite.bridge import load_artifact
from legoesm.atmosphere.les_suite.candidate import load_candidate

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "validate" / "les_suite"))
from cosci_eval import SCM_DT_S, SCM_NLEV, Arm, _roster_base, candidate_loss  # noqa: E402

REL_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)
ABS_SCALE = 1e-3  # step scale for a parameter whose default is 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("baseline")
    ap.add_argument("artifacts", nargs="+")
    ap.add_argument("--params", default="")
    a = ap.parse_args()
    if not jax.config.read("jax_enable_x64"):
        raise SystemExit("run with JAX_ENABLE_X64=1")
    cand = load_candidate(a.candidate)
    base, _ = _roster_base(a.baseline)
    arm = Arm(base, cand, "enthalpy", True)
    p0 = cand.default_params()
    names = [s for s in a.params.split(",") if s] or list(p0)
    for path in a.artifacts:
        art = load_artifact(path)
        loss = candidate_loss(arm, [(path, art)], nlev=SCM_NLEV, dt=SCM_DT_S)
        l0 = float(loss(p0))
        g = jax.grad(loss)(p0)
        print(f"\n### {path}  candidate={a.candidate} baseline={a.baseline} "
              f"heat_form=enthalpy ke_heating=on loss={l0:.10f}")
        for k in names:
            print(f"{k}: value={float(p0[k]):.6g} AD={float(g[k]):+.6e}")
            print("    h            L(p+h)-L0       L0-L(p-h)      right-slope     left-slope      central")
            for r in REL_STEPS:
                h = r * max(abs(float(p0[k])), ABS_SCALE)
                lp = float(loss({**p0, k: p0[k] + h}))
                lm = float(loss({**p0, k: p0[k] - h}))
                flag = "" if all(map(math.isfinite, (lp, lm))) else "  NONFINITE"
                print(f"    {h:.3e}  {lp - l0:+.6e}  {l0 - lm:+.6e}  {(lp - l0) / h:+.6e}  "
                      f"{(l0 - lm) / h:+.6e}  {(lp - lm) / (2 * h):+.6e}{flag}")


if __name__ == "__main__":
    main()
