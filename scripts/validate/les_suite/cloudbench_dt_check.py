"""Do CloudBench E2 scores depend on the replay's timestep (600 s vs a shorter one)?

The replay calls its interactive radiation once per step (the LES: every 240 s);
a shorter step also changes the turbulence and microphysics integration, so this
measures the whole timestep sensitivity, not radiation alone. Per case and step,
the E2 score (cosci_eval's ``_case_scores``: conservation gate + E2) of:

* the tuned YSU benchmark: YSU with the parameters of the CloudBench DF tuning
  (``TUNED`` file, ``ysu`` ``inc``), the arm the tuning scored (``Arm(base)``);
* a candidate at its default parameters, on the round-2 protocol arm (YSU base,
  enthalpy heat form, KE heating), as ``cosci_ablation.py`` scores it.

Agreement rule (voted 2026-11-19 before any number: author, codex, GLM all (a)):
for every case and both arms |E2(dt) - E2(600)| / E2(600) <= 2 %, and the sign of
E2(candidate) - E2(YSU) unchanged in every case. Numbers and the rule's outcome only.

The two arms differ in heat form by design (the benchmark as tuned; the candidate on
the round-2 protocol); the rule compares each arm with itself across steps.

usage: cloudbench_dt_check.py OUT_JSON DT_S CANDIDATE.py ART.npz [ART.npz ...]
       cloudbench_dt_check.py compare REF_600.json OTHER.json [...]   (pairs by case)
Run with JAX_ENABLE_X64=1 on a compute node.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cosci_eval as ce  # noqa: E402
from legoesm.atmosphere.les_suite.candidate import load_candidate  # noqa: E402

TUNED = Path("/glade/derecho/scratch/pg2328/cosci/round1/tuned_benchmarks_cloudbench.json")
FORCING = Path("/glade/derecho/scratch/pg2328/les_suite/external/cloudbench/forcing")
STATS = Path("/glade/derecho/scratch/pg2328/les_suite/external/cloudbench/cloudbench_statistics.nc")
NLEV, HEAT_FORM, KE_HEATING = 32, "enthalpy", True
REL_TOL = 0.02                                     # the voted per-case agreement bound


def _e2(entry):
    """A scored entry's E2, or None for a failed or non-finite one."""
    v = entry.get("combined") if "failed" not in entry else None
    return v if v is not None and v == v and abs(v) != float("inf") else None


def agreement(ref: dict, other: dict) -> dict:
    """The voted rule on two outputs of this script (``ref`` at 600 s): per case and
    arm the relative E2 change, and the sign of E2(candidate) - E2(YSU) at each step.
    A failed or non-finite score anywhere is a disagreement."""
    rows, ok = {}, True
    for case in ref["ysu_tuned"]:
        r = {}
        e = {(arm, k): _e2(res[arm][case]) for arm in ("ysu_tuned", "candidate_scores")
             for k, res in (("ref", ref), ("other", other))}
        if any(v is None for v in e.values()):
            rows[case] = {"agree": False, "reason": "failed or non-finite score"}
            ok = False
            continue
        for arm in ("ysu_tuned", "candidate_scores"):
            r[f"{arm}_rel_change"] = abs(e[(arm, "other")] - e[(arm, "ref")]) / e[(arm, "ref")]
        g_ref = e[("candidate_scores", "ref")] - e[("ysu_tuned", "ref")]
        g_oth = e[("candidate_scores", "other")] - e[("ysu_tuned", "other")]
        r["gap_ref"], r["gap_other"] = g_ref, g_oth
        r["agree"] = (max(r["ysu_tuned_rel_change"], r["candidate_scores_rel_change"]) <= REL_TOL
                      and (g_ref > 0) == (g_oth > 0))
        ok &= r["agree"]
        rows[case] = r
    return {"agree": ok, "rel_tol": REL_TOL, "cases": rows}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        ref, *others = (json.loads(Path(p).read_text()) for p in sys.argv[2:])
        if ref["dt"] != 600.0:
            raise SystemExit("the first file must be the 600 s run")
        print(json.dumps({o["dt"]: agreement(ref, o) for o in others}, indent=1))
        return 0
    if not jax.config.read("jax_enable_x64"):
        raise SystemExit("run with JAX_ENABLE_X64=1")
    out, dt, cand_path = Path(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3])
    arts = [(str(p), ce.load_artifact(Path(p))) for p in sys.argv[4:]]
    prov = ce.configure_cloudbench(arts, FORCING, STATS)
    base, _ = ce._roster_base("ysu")
    tuned = json.loads(TUNED.read_text())["ysu"]["inc"]
    turb = ce.apply_overrides_to_base(base, tuned)
    if turb == base:
        raise SystemExit("the tuned YSU parameters change nothing")
    cand = load_candidate(cand_path)
    res = {"dt": dt, "tuned_ysu": tuned, "candidate": cand_path.name,
           "split": {Path(p).name: ce._regime(a.case_name) for p, a in arts},
           "nsteps": {Path(p).name: ce.e2_nsteps(a, dt) for p, a in arts},
           "ysu_tuned": ce._case_scores(ce.Arm(base), arts, NLEV, dt, turb=turb),
           "candidate_scores": ce._case_scores(
               ce.Arm(base, cand, HEAT_FORM, KE_HEATING), arts, NLEV, dt,
               params=cand.default_params()),
           "provenance": {"git_sha": ce._git_sha(), "code_id": ce._code_id(),
                          "cloudbench": prov}}
    out.write_text(json.dumps(res, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
