"""Reference: KE of the filtered coarse truth states themselves (same KE definition).

The training/validation targets are the stored filtered truth; its KE is the
level a closure is asked to reach.  Samples the contract's rebalanced start
states at a few absolute days in the train and validation splits and writes
round10_diagnosis/step4b_truth_ke_reference.json.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("energy_budget", HERE / "energy_budget.py")
eb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eb)

DAYS = (740, 900, 1100, 1300, 1450, 4265, 4400, 4600, 4800, 5000, 5100)


def main():
    driver, config, card, geometry, nets, forcing, key = eb.setup()
    umask = np.asarray(geometry.umask, dtype=bool)
    vmask = np.asarray(geometry.vmask, dtype=bool)
    contract = eb.SWGDatasetContract.from_json(eb.RUN / "dataset_contract.json", card=card)
    rows = []
    for day in DAYS:
        (ssh, u, v), _ = contract.start_state(24 * day)
        ke = 0.5 * (np.sum(u**2 * umask) / umask.sum() + np.sum(v**2 * vmask) / vmask.sum())
        rows.append({"absolute_day": day, "ke": float(ke), "max_speed": float(max(np.abs(u).max(), np.abs(v).max()))})
        print(rows[-1])
    out = {"definition": __doc__, "ke_units": "m2 s-2 (0.5*(mean u^2 over wet U faces + mean v^2 over wet V faces))",
           "samples": rows, "median_ke_train_days": float(np.median([r["ke"] for r in rows if r["absolute_day"] < 1461])),
           "median_ke_validation_days": float(np.median([r["ke"] for r in rows if r["absolute_day"] >= 4265]))}
    (eb.DIAG / "step4b_truth_ke_reference.json").write_text(json.dumps(out, indent=1))
    print(out["median_ke_train_days"], out["median_ke_validation_days"])


if __name__ == "__main__":
    main()
