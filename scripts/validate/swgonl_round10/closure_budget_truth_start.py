"""Round-10 measurement 2: closure energy budget in the truth-start regime.

Reuses energy_budget.py (step 4) unchanged: ``setup`` loads the card, the two
stage-final networks and the steady wind; ``make_terms`` gives the closure,
wind, viscous and drag work per unit mass [m2 s-3] and the KE [m2 s-2] at a
stored state (model's own stage-3 program differenced on/off at the entry
state; domain mean over wet U/V faces).

Samples: end-of-day snapshots every 10 days (days 10, 20, ..., 1090) of the
three day-4265 truth-start runs (w01, w02 with their own networks; control
with the closure off), and days 10..300 of the two from-rest pilot runs.
Also one sample at the first training-window start of the dataset contract
(2-day window, epoch 0, first window of the first batch) so measurement 3 can
quote a lambda=1 control for that state too.

Per run and year (year k = days 365(k-1)+1..365k): medians of each work term,
of the per-sample ratios closure/wind and closure/|drag|, and of KE.  Fits:
log(closure work) = a + b log(KE), least squares over all samples with
positive closure work, separately for the truth-start and the from-rest
samples, with the Pearson correlation of the logs.
Writes round10_diagnosis/step4b_energy_budget_day4265.json.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("energy_budget", HERE / "energy_budget.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)

OUT = eb.DIAG / "step4b_energy_budget_day4265.json"
STRIDE = 10
DAYS_PER_YEAR = 365
RUNS = {
    # name: (root, network index or None, last day)
    "w01_from_day4265": (eb.DIAG / "free_runs_day4265/w01", 1, 1095),
    "w02_from_day4265": (eb.DIAG / "free_runs_day4265/w02", 2, 1095),
    "control_nu10_from_day4265": (eb.DIAG / "free_runs_day4265/swgonl_control_nu10_gpu_day4265", None, 1095),
    "pilot_w01_from_rest": (eb.RUN / "free_runs/w01", 1, 300),
    "pilot_w02_from_rest": (eb.RUN / "free_runs/w02", 2, 300),
}
TERMS = ("closure_work", "wind_work", "viscous_work", "drag_work", "ke")


def sample_days(last):
    return list(range(STRIDE, last + 1, STRIDE))


def year_of(day):
    return (day - 1) // DAYS_PER_YEAR + 1


def summarise(rows):
    """Per-year medians of the terms and of the per-sample ratios."""
    out = {}
    for year in sorted({year_of(r["day"]) for r in rows}):
        sub = [r for r in rows if year_of(r["day"]) == year]
        med = lambda key: float(np.median([r[key] for r in sub]))
        ratio = lambda num, den, f=lambda x: x: float(np.median([r[num] / f(r[den]) for r in sub]))
        out[f"year{year}"] = {
            "n_samples": len(sub), "days": [sub[0]["day"], sub[-1]["day"]],
            **{key: med(key) for key in TERMS},
            "closure_over_wind": ratio("closure_work", "wind_work"),
            "closure_over_abs_drag": ratio("closure_work", "drag_work", abs),
            "n_negative_wind_samples": int(sum(r["wind_work"] < 0 for r in sub)),
            "n_nonpositive_closure_samples": int(sum(r["closure_work"] <= 0 for r in sub)),
        }
    return out


def fit(rows):
    """log(closure) = a + b log(KE) over samples with positive closure work."""
    pos = [r for r in rows if r["closure_work"] > 0]
    x = np.log([r["ke"] for r in pos])
    y = np.log([r["closure_work"] for r in pos])
    b, a = np.polyfit(x, y, 1)
    return {"a": float(a), "b": float(b), "pearson_r_of_logs": float(np.corrcoef(x, y)[0, 1]),
            "n": len(pos), "n_excluded_nonpositive_closure": len(rows) - len(pos),
            "ke_range": [float(np.exp(x.min())), float(np.exp(x.max()))]}


def main():
    driver, config, card, geometry, nets, forcing, key = eb.setup()
    out = {"definition": __doc__, "reused": "energy_budget.py setup/make_terms (same folder)",
           "units": {"work": "m2 s-3 (per unit mass, domain mean over wet faces)", "ke": "m2 s-2"},
           "sampling": f"end-of-day snapshots every {STRIDE} days starting at day {STRIDE}",
           "runs": {}}
    programs = {}
    for name, (root, net, last) in RUNS.items():
        if net not in programs:
            programs[net] = jax.jit(eb.make_terms(card, geometry, None if net is None else nets[net], forcing, key))
        fn = programs[net]
        manifest = json.loads((root / "manifest.json").read_text())
        last = min(last, manifest["completed_days"])
        rows = []
        for day in sample_days(last):
            with np.load(root / f"day{day:04d}.npz") as f:
                values = fn(jnp.asarray(f["ssh_m"]), jnp.asarray(f["u_m_s"]), jnp.asarray(f["v_m_s"]))
            rows.append({"day": day, **{k: float(v) for k, v in values.items()}})
        out["runs"][name] = {"root": str(root), "network": None if net is None else f"w{net:02d}_e05.eqx",
                             "completed_days": manifest["completed_days"],
                             "per_year": summarise(rows), "samples": rows}
        print(name, "samples", len(rows), json.dumps(
            {y: {k: f"{s[k]:.3e}" for k in ("closure_work", "wind_work", "drag_work", "ke")}
             | {"clo/wind": f"{s['closure_over_wind']:.3f}", "clo/|drag|": f"{s['closure_over_abs_drag']:.3f}"}
             for y, s in out["runs"][name]["per_year"].items()}, indent=None))
        sys.stdout.flush()

    # the first training-window start, as the spans probe built it (measurement-3 control)
    contract = eb.SWGDatasetContract.from_json(eb.RUN / "dataset_contract.json", card=card)
    batch = next(contract.batches(split="train", window_days=2, epoch=0, seed=eb.SEED))
    ssh, u, v = (np.asarray(value[0]) for value in batch.initial)
    out["training_window_start"] = {
        "start_absolute_hour": int(batch.start_absolute_hour[0]),
        "start_day": int(batch.start_absolute_hour[0]) / 24.0,
        "how": "SWGDatasetContract.batches(split='train', window_days=2, epoch=0, seed=SEED), first batch, window 0",
        **{f"w{w:02d}": {k: float(val) for k, val in programs[w](jnp.asarray(ssh), jnp.asarray(u), jnp.asarray(v)).items()}
           for w in (1, 2)}}

    truth = [r for n in ("w01_from_day4265", "w02_from_day4265") for r in out["runs"][n]["samples"]]
    rest = [r for n in ("pilot_w01_from_rest", "pilot_w02_from_rest") for r in out["runs"][n]["samples"]]
    out["fits_log_closure_vs_log_ke"] = {
        "form": "log(closure_work) = a + b*log(ke), natural logs, least squares on samples with closure_work > 0",
        "truth_start_w01_w02_days10_1090": fit(truth),
        "from_rest_w01_w02_days10_300": fit(rest),
        "per_run": {n: fit(out["runs"][n]["samples"]) for n in out["runs"] if "control" not in n},
    }
    print(json.dumps(out["fits_log_closure_vs_log_ke"], indent=1))
    OUT.write_text(json.dumps(out, indent=1))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
