"""Profile figures for the co-scientist status page: the SCM profiles the E2 score
compares with the LES, at the last scored LES frame, for a few closures on a few
runs (PI 2026-10-31: one run per regime plus one held-out Ayotte run; LES, the best
benchmark at defaults, YSU tuned, wave kernel tuned, third moment at defaults,
pressure-turning tuned; the scored variables).

Each closure's parameters and its stored per-case score are read from the 27-run
E2 outputs (``ad_recorded`` = parameters tuned on the original 14 runs). Every
rollout is re-scored and its absolute difference from the stored score printed, so
the drawn profiles can be checked against the scored ones. A closure whose rollout is not finite on a run
is left out of that run's figure, and the omission is printed.

usage: cosci_profile_figures.py E2_DIR OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validate" / "les_suite"))
from cosci_eval import (  # noqa: E402
    SCM_NLEV,
    Arm,
    _roster_base,
    _sha256,
    apply_overrides_to_base,
)
from legoesm.atmosphere.les_suite.bridge import load_artifact  # noqa: E402
from legoesm.atmosphere.les_suite.candidate import load_candidate  # noqa: E402
from legoesm.atmosphere.les_suite.candidate_eval import (  # noqa: E402
    e2_nsteps,
    e2_profiles,
    e2_score,
)

DT = 600.0
ART = "/glade/derecho/scratch/pg2328/les_suite/artifacts/"
RUNS = [("cbl_nieuwstadt__lasd.npz", "Nieuwstadt convective layer"),
        ("dry_grid_b0s1__lasd__ug5.npz", "Stable dry grid run, geostrophic wind 5 m/s"),
        ("bomex_cu.npz", "BOMEX shallow cumulus"),
        ("astex_sc.npz", "ASTEX stratocumulus"),
        ("ayotte96_05sc__lasd.npz", "Ayotte 05sc (held out)")]
CAND = "scripts/validate/les_suite/cosci_candidates/round1/"
# (label, colour, E2 file, arm key, variant, candidate file or None)
ARMS = [
    ("Holtslag-Boville default", "tab:blue", "E2_baseline_holtslag_boville.json",
     "baseline:holtslag_boville", "default", None),
    ("YSU tuned", "tab:cyan", "E2_baseline_ysu.json", "baseline:ysu", "ad_recorded", None),
    ("Wave kernel tuned", "tab:red", "E2_r1_c1_wave_kernel.json",
     "candidate:r1_c1_wave_kernel", "ad_recorded", "r1_c1_wave_kernel.py"),
    ("Third moment default", "tab:orange", "E2_r1_c2_third_moment_noastex.json",
     "candidate:r1_c2_third_moment", "default", "r1_c2_third_moment.py"),
    ("Pressure-turning tuned", "tab:green", "E2_r1_codex_pressure_turning_trans.json",
     "candidate:r1_codex_pressure_turning_trans", "ad_recorded",
     "r1_codex_pressure_turning_trans.py"),
]
PANELS = [("theta", "theta_l [K]", 1.0), ("qt", "q_t [g/kg]", 1e3), ("u", "u [m/s]", 1.0),
          ("v", "v [m/s]", 1.0), ("wtheta", "heat flux w'theta' [K m/s]", 1.0),
          ("wqt", "moisture flux w'q_t' [g/kg m/s]", 1e3), ("uw", "u'w' [m2/s2]", 1.0),
          ("vw", "v'w' [m2/s2]", 1.0)]


def arm_and_params(e2_dir, fname, key, variant, cand_file):
    rec = json.loads((e2_dir / fname).read_text())
    prov = rec["provenance"]
    entry = rec["result"]["arms"][key]
    stored = entry[variant]["scores"] if variant == "ad_recorded" else entry[variant]
    params = entry[variant]["params"] if variant == "ad_recorded" else None
    if cand_file is None:
        base = _roster_base(key.split(":", 1)[1])[0]
        turb = None if params is None else apply_overrides_to_base(base, params)
        return Arm(base), {"turb": turb}, stored
    src = CAND + cand_file
    if _sha256(src) != prov["candidate"]["sha256"]:
        raise SystemExit(f"{fname}: candidate source {src} differs from the scored one")
    cand = load_candidate(src)
    ke = prov["ke_heating"]
    arm = Arm(_roster_base(prov["baseline"])[0], cand, prov["heat_form"],
              None if ke is None else ke == "on")
    return arm, {"params": cand.default_params() if params is None else params}, stored


def main(argv=None) -> int:
    e2_dir, out = (Path(a) for a in (argv or sys.argv[1:]))
    if not jax.config.read("jax_enable_x64"):
        raise SystemExit("run with JAX_ENABLE_X64=1")
    out.mkdir(parents=True, exist_ok=True)
    arms = [(lab, col, *arm_and_params(e2_dir, f, k, v, c)) for lab, col, f, k, v, c in ARMS]
    for fname, title in RUNS:
        art = load_artifact(ART + fname)
        n = e2_nsteps(art, DT)
        drawn, prof0 = [], None
        for lab, col, arm, kw, stored in arms:
            if fname not in stored:
                print(f"{fname}: {lab} has no stored score (not scored on this run); left out")
                continue
            rec, _ = arm.run(art, n, nlev=SCM_NLEV, dt=DT, **kw)
            col_z = np.asarray(rec.at(n).z_full)
            if not (np.all(np.isfinite(col_z)) and np.all(np.diff(col_z) > 0)):
                print(f"{fname}: {lab} rollout not finite; left out")
                continue
            got = float(e2_score(art, rec, DT).combined)
            want = stored[fname]["combined"]
            print(f"{fname}: {lab} re-score {got!r} stored {want!r} "
                  f"abs diff {abs(got - want):.3e}")
            prof = e2_profiles(art, rec, DT)
            bad = [k for k in prof.truth
                   if not np.all(np.isfinite(np.asarray(prof.model[k][-1])[
                       np.isfinite(np.asarray(prof.truth[k][-1]))]))]
            if bad:
                print(f"{fname}: {lab} non-finite {bad} at the last frame; left out")
                continue
            prof0 = prof
            drawn.append((lab, col, prof, got))
            jax.clear_caches()
        if prof0 is None:
            print(f"{fname}: nothing drawn")
            continue
        z = np.asarray(prof0.heights)
        keys = [k for k, _, _ in PANELS if k in prof0.truth]
        fig, axes = plt.subplots(1, len(keys), figsize=(2.6 * len(keys), 4.2), sharey=True)
        t_min = prof0.steps[-1] * DT / 60.0
        for ax, key in zip(axes, keys):
            label, scale = next((lb, sc) for k, lb, sc in PANELS if k == key)
            truth = np.asarray(prof0.truth[key][-1]) * scale
            ok = np.isfinite(truth)
            ax.plot(truth[ok], z[ok], color="k", lw=2.2, label="LES")
            for lab, col, prof, score in drawn:
                ax.plot(np.asarray(prof.model[key][-1])[ok] * scale, z[ok], color=col,
                        lw=1.4, label=f"{lab} ({score:.2f})")
            ax.set_xlabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("height [m]")
        axes[-1].legend(fontsize=7, loc="best")
        fig.suptitle(f"{title}: t = {t_min:.0f} min (last scored frame); "
                     "legend: closure (score on this run)", fontsize=9)
        fig.tight_layout()
        png = out / (Path(fname).stem + "_profiles.png")
        fig.savefig(png, dpi=110)
        plt.close(fig)
        print(f"{fname}: wrote {png} ({len(drawn)} closures)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
