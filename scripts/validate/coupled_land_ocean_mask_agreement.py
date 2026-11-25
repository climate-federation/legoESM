"""Does the atmosphere's land fraction agree with the 3-D ocean's wet cells?

Builds two coupled drivers (build-only, no step): one whose atmosphere land
fraction comes from the land-mask file (the owned-surface recipe) and one whose
3-D ocean wet mask comes from WOA.  Both must sit on the same lat-lon grid.

Counts, per cell:
  wet_and_full_land  ocean wet (mask 1) while f_land == 1
  dry_and_some_water ocean dry (mask 0) while f_land < 1
Writes a JSON with the counts and a PNG map of both classes.

Usage: python coupled_land_ocean_mask_agreement.py --out DIR \
           --land-argv "<run_coupled args>" --ocean-argv "<run_coupled args>"
"""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import numpy as np


def classify(f_land: np.ndarray, ocean_wet: np.ndarray) -> dict:
    if f_land.shape != ocean_wet.shape:
        raise ValueError(f"grid mismatch: f_land {f_land.shape} vs ocean {ocean_wet.shape}")
    wet = ocean_wet > 0.5
    a = wet & (f_land >= 1.0)
    b = (~wet) & (f_land < 1.0)
    return {"n_cells": int(f_land.size), "n_ocean_wet": int(wet.sum()),
            "n_full_land": int((f_land >= 1.0).sum()),
            "n_partial_land": int(((f_land > 0) & (f_land < 1)).sum()),
            "wet_and_full_land": int(a.sum()), "dry_and_some_water": int(b.sum()),
            "_a": a, "_b": b}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--land-argv", required=True)
    p.add_argument("--ocean-argv", required=True)
    a = p.parse_args(argv)
    from scripts.run import run_coupled
    land = run_coupled.main(shlex.split(a.land_argv), build_only=True)
    f_land = np.asarray(land._atm._f_land, dtype=float)
    ocean = run_coupled.main(shlex.split(a.ocean_argv), build_only=True)
    wet = np.asarray(ocean._ocean_land_mask, dtype=float)
    f_land = f_land.reshape(wet.shape)
    r = classify(f_land, wet)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "masks.npz", f_land=f_land, ocean_wet=wet)
    (out / "mask_agreement.json").write_text(json.dumps(
        {k: v for k, v in r.items() if not k.startswith("_")}, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cls = np.where(r["_a"], 1, np.where(r["_b"], 2, 0))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(np.where(wet > 0.5, 0.3, 0.0) + 0 * cls, origin="lower", cmap="Blues", vmin=0, vmax=1)
    ys, xs = np.nonzero(cls == 1)
    ax.scatter(xs, ys, c="red", s=12, label=f"ocean wet, full land ({r['wet_and_full_land']})")
    ys, xs = np.nonzero(cls == 2)
    ax.scatter(xs, ys, c="orange", s=12, label=f"ocean dry, some water ({r['dry_and_some_water']})")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title("atm land mask vs 3-D ocean wet mask (blue = ocean wet)")
    fig.savefig(out / "mask_agreement.png", dpi=120, bbox_inches="tight")
    print(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}))


if __name__ == "__main__":
    main()
