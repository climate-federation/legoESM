"""Sensitivity of the tuned-YSU E2 score to where the SCM's free-troposphere
nudging starts, and the size of the recorded LES nudging term over all artifacts.

Arms (same column, same everything, only the artifact's ``nudge_rate`` replaced):
``recorded`` (swirl-lm defaults, cosine ramp 3.0-3.5 km, tau 1 day) and shifted
ramps ``(z_i, z_r)`` given below; each run is scored with E2 and the conservation
gate. Survey: per artifact, the day-5 band mean of the recorded nudging term
rate * (x0 - x) between 3 and 4 km, from the LES state (theta_l K/day, q_t g/kg/day).
Numbers only. usage: cloudbench_nudge_sensitivity.py OUT_JSON CASE [CASE ...]
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cosci_eval as ce  # noqa: E402
from cloudbench_jump_audit import ART, DT, FORCING, NLEV, STATS, YSU_TUNED  # noqa: E402
from legoesm.atmosphere.les_suite.candidate_eval import e2_nsteps, e2_score  # noqa: E402
from legoesm.atmosphere.les_suite.conservation import check_conservation  # noqa: E402

TAU_S = 86400.0
RAMPS = {"recorded": None, "ramp_4.0_4.5km": (4000.0, 4500.0), "ramp_4.5_5.0km": (4500.0, 5000.0)}


def ramp_rate(z, z_i, z_r):
    r = 0.5 * (1.0 - np.cos(np.pi * (z - z_i) / (z_r - z_i)))
    return np.where(z < z_i, 0.0, np.where(z <= z_r, r, 1.0)) / TAU_S


def survey():
    rows = {}
    for p in sorted(ART.glob("cloudbench_*_amip.npz")):
        d = np.load(p, allow_pickle=True)
        z, rate = d["heights_m"], d["nudge_rate"]
        m = (z >= 3000.0) & (z < 4000.0)
        th = (rate * (d["theta"][0] - d["theta"][1:])).mean(0)
        qt = (rate * (d["qt"][0] - d["qt"][1:])).mean(0)
        rows[p.name] = (float(86400 * th[m].mean()), float(86400e3 * qt[m].mean()))
    a = np.array(list(rows.values()))
    s = {"n": len(rows)}
    for j, (k, u) in enumerate((("theta_l", "K/day"), ("q_t", "g/kg/day"))):
        x = np.abs(a[:, j])
        s[k] = {"units": u, "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
                "max": float(x.max()),
                "n_over": {str(t): int((x > t).sum()) for t in ((0.5, 1.0, 2.0) if j == 0 else (0.1, 0.3, 1.0))}}
    return s, rows


def main() -> int:
    if not jax.config.read("jax_enable_x64"):
        raise SystemExit("run with JAX_ENABLE_X64=1")
    s, rows = survey()
    print(json.dumps(s), flush=True)
    out = {"git_sha": ce._code_id(), "survey": s, "survey_cases": rows, "runs": {}}
    arts = [(str(ART / f"cloudbench_{c}_amip.npz"), ce.load_artifact(ART / f"cloudbench_{c}_amip.npz"))
            for c in sys.argv[2:]]
    ce.configure_cloudbench(arts, FORCING, STATS)
    base, key = ce._roster_base("ysu")
    fields = [m.field for m in ce._scheme_tunable_metas(key, (1,))]
    turb = ce.apply_overrides_to_base(base, {f: YSU_TUNED[f] for f in fields})
    yarm = ce.Arm(base)
    for path, art in arts:
        n = e2_nsteps(art, DT)
        z = np.asarray(art.heights_m)
        for arm, rz in RAMPS.items():
            a = art if rz is None else dataclasses.replace(art, nudge_rate=ramp_rate(z, *rz))
            prep = yarm.prepare(a, nlev=NLEV, dt=DT)
            rec = yarm.run_with_turb(prep, a, n, turb)[0]
            sc = e2_score(a, rec, DT)
            gate = check_conservation(rec, DT)
            out["runs"].setdefault(Path(path).name, {})[arm] = {
                "combined": float(sc.combined),
                "per_var": {k: float(v) for k, v in sc.per_var.items()},
                "gate_passed": bool(gate.passed)}
            print(Path(path).name, arm, json.dumps(out["runs"][Path(path).name][arm]), flush=True)
            jax.clear_caches()
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
