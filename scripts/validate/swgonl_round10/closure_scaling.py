"""Round-10 measurement 3: how the closure's work scales with the flow amplitude.

Five stored states spanning the regimes; for each, the anomalies (sea level
about its wet mean on T points, u, v) are multiplied by lambda in LAMBDAS and
the work terms of energy_budget.make_terms (closure on/off, wind, viscous,
drag; the stage-3 program differenced at the same entry state) are evaluated
on the scaled flow with each stage-final network (w01_e05, w02_e05) and with
a zero-potential closure (the real network's output multiplied by zero,
pushed through the closure path: control, must give exactly 0 work).

Reported per state and network: work(lambda)/work(1), the least-squares
exponent of log|work| vs log(lambda) over the six lambdas (sign recorded),
the same for wind and drag, the smallest lambda at which closure work exceeds
|drag work|, and whether the closure exponent exceeds the drag exponent.
Drag is linear in u (swe_rk3.py: drag_u = r * before_entry.u_k), so its work
scales as lambda^2 exactly; the wind source is tau/(2 rho_0 e3u), state
independent except through the live layer thickness e3u, so ~lambda^1.
Controls: lambda=1 work vs the measurement-2 number for the same stored state
(step4b JSON, same sample); zero-potential closure work at every lambda.
Writes round10_diagnosis/step5_closure_scaling.json.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("energy_budget", HERE / "energy_budget.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)

M2 = eb.DIAG / "step4b_energy_budget_day4265.json"
OUT = eb.DIAG / "step5_closure_scaling.json"
LAMBDAS = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
STATES = {
    # name: (measurement-2 run name or None, root, day)
    "control_from_day4265_day1000": ("control_nu10_from_day4265",
                                     eb.DIAG / "free_runs_day4265/swgonl_control_nu10_gpu_day4265", 1000),
    "w02_from_rest_day300": ("pilot_w02_from_rest", eb.RUN / "free_runs/w02", 300),
    "w02_from_rest_day900": (None, eb.RUN / "free_runs/w02", 900),
    "training_window_start_day734": ("training_window_start", None, None),
    "w01_from_day4265_day1000": ("w01_from_day4265", eb.DIAG / "free_runs_day4265/w01", 1000),
}
WORKS = ("closure_work", "wind_work", "viscous_work", "drag_work")


def load_state(card, root, day):
    if root is None:
        contract = eb.SWGDatasetContract.from_json(eb.RUN / "dataset_contract.json", card=card)
        batch = next(contract.batches(split="train", window_days=2, epoch=0, seed=eb.SEED))
        return tuple(np.asarray(value[0], dtype=np.float64) for value in batch.initial)
    with np.load(root / f"day{day:04d}.npz") as f:
        return f["ssh_m"].astype(np.float64), f["u_m_s"].astype(np.float64), f["v_m_s"].astype(np.float64)


def scaled(wet, ssh, u, v, lam):
    mean = ssh[wet].mean()
    return np.where(wet, mean + lam * (ssh - mean), ssh), lam * u, lam * v


def exponent(values):
    """Slope of log|w| vs log(lambda); sign flips across lambdas are reported."""
    w = np.asarray(values)
    signs = set(np.sign(w).tolist())
    if 0.0 in signs or len(signs) > 1:
        return {"exponent": None, "note": "sign not constant across lambdas", "signs": sorted(signs)}
    b, _ = np.polyfit(np.log(LAMBDAS), np.log(np.abs(w)), 1)
    return {"exponent": float(b), "sign": float(next(iter(signs)))}


def main():
    driver, config, card, geometry, nets, forcing, key = eb.setup()
    wet = np.asarray(geometry.tmask, dtype=bool)
    m2 = json.loads(M2.read_text())

    def zero_potential(entry, geometry, **kw):
        return jnp.zeros_like(nets[1](entry, geometry, **kw))

    programs = {"w01": jax.jit(eb.make_terms(card, geometry, nets[1], forcing, key)),
                "w02": jax.jit(eb.make_terms(card, geometry, nets[2], forcing, key)),
                "zero_potential": jax.jit(eb.make_terms(card, geometry, zero_potential, forcing, key))}
    out = {"definition": __doc__, "lambdas": list(LAMBDAS),
           "reused": "energy_budget.py setup/make_terms (same folder)",
           "drag_form": "linear: drag_u = bottom_drag_r * u (swe_rk3.py stage 3), work ~ lambda^2",
           "states": {}}
    for name, (m2_name, root, day) in STATES.items():
        ssh, u, v = load_state(card, root, day)
        entry = {"root": None if root is None else str(root), "day": day,
                 "ssh_wet_mean_m": float(ssh[wet].mean()), "networks": {}}
        for net, fn in programs.items():
            rows = []
            for lam in LAMBDAS:
                values = fn(*(jnp.asarray(a) for a in scaled(wet, ssh, u, v, lam)))
                rows.append({"lambda": lam, **{k: float(x) for k, x in values.items()}})
            at1 = rows[LAMBDAS.index(1.0)]
            block = {"samples": rows,
                     "ratio_to_lambda1": {k: [r[k] / at1[k] if at1[k] != 0.0 else None for r in rows] for k in WORKS},
                     "exponent": {k: exponent([r[k] for r in rows]) for k in WORKS}}
            exceed = [r["lambda"] for r in rows if r["closure_work"] > abs(r["drag_work"])]
            block["closure_exceeds_abs_drag_from_lambda"] = exceed[0] if exceed else None
            e_c, e_d = block["exponent"]["closure_work"]["exponent"], block["exponent"]["drag_work"]["exponent"]
            block["closure_grows_faster_than_drag"] = None if e_c is None else bool(e_c > e_d)
            if net == "zero_potential":
                block["control_zero_closure_work_at_every_lambda"] = all(r["closure_work"] == 0.0 for r in rows)
            elif m2_name is not None:
                if m2_name == "training_window_start":
                    ref = m2["training_window_start"][net]
                else:
                    ref = next(r for r in m2["runs"][m2_name]["samples"] if r["day"] == day)
                    ref = ref if m2["runs"][m2_name]["network"] == f"{net}_e05.eqx" or m2["runs"][m2_name]["network"] is None else None
                if ref is not None:
                    block["control_lambda1_vs_measurement2"] = {
                        k: {"scaling_lambda1": at1[k], "measurement2": ref[k],
                            "rel_diff": (at1[k] - ref[k]) / ref[k] if ref[k] != 0.0 else at1[k] - ref[k]}
                        for k in WORKS + ("ke",)}
            entry["networks"][net] = block
            print(name, net, "clo ratios", [None if x is None else round(x, 3) for x in block["ratio_to_lambda1"]["closure_work"]],
                  "exp clo/wind/drag", [block["exponent"][k].get("exponent") for k in ("closure_work", "wind_work", "drag_work")],
                  "clo>|drag| from", block["closure_exceeds_abs_drag_from_lambda"])
        out["states"][name] = entry
    OUT.write_text(json.dumps(out, indent=1))
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
