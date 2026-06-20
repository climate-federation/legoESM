#!/usr/bin/env python
"""One-at-a-time (OAT) physics-combination sweep for AIMIP lat-lon.

Finds the best classical physics scheme COMBINATION by varying one
parameterization axis at a time off a baseline, each combo trained +
evaluated by ``run_aimip_latlon.py`` (classical variant) with the
radiation-flux loss on.  Mirrors the spectral
``run_aimip_classical_sweep_stage1.py`` protocol but on the faster
lat-lon C-grid carry stack.

Axes swept: convection, turbulence, gravity-wave drag — the
differentiable levers with >1 option.  Microphysics is now threaded
through the training segment (baseline uses the orchestrator default,
``kessler``), but is not an OAT axis here (one warm-rain scheme is enough
to "have microphysics"; add an axis if scheme choice needs comparing).
Radiation is the orchestrator default ``rrtmgp`` (band model — required
for AMIP-like flux generalization; gray is debug-only).

Usage:
    python run_aimip_latlon_sweep.py --list           # n combos
    python run_aimip_latlon_sweep.py --index 0 --output-root results/aimip_latlon_sweep
    # SLURM array: --array=0-(N-1) one combo per task.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Baseline classical stack (one known-good point in the space).  Full AMIP
# physics: convection + turbulence + active (non-orographic) GWD; microphysics
# (kessler) and radiation (rrtmgp) come from the orchestrator defaults.
BASELINE = {
    "convection": "sbm",
    "turbulence": "louis",
    "gravity_wave_drag": "hines",
}

# OAT alternatives per axis (baseline value excluded — it is the
# ``combo_baseline`` run, shared across axes).
SWEEP_SPACE = {
    "convection": [
        "dca", "kuo", "mass_flux", "edmf", "zhang_mcfarlane",
        "kain_fritsch", "emanuel", "bechtold", "tiedtke",
    ],
    # Stateful-TKE schemes (tke, mynn25) need a seeded prognostic carry
    # (issue #413) the orchestrator doesn't thread yet -> excluded; the
    # diagnostic closures are swept.
    "turbulence": [
        "smagorinsky", "holtslag_boville", "ysu", "none",
    ],
    # hines is the baseline; "none" probes the no-GWD effect.  mcfarlane and
    # lindzen are orographic — with no per-column subgrid-orography input they
    # fall back to the scalar config.h_topo (~500 m), so they ARE active
    # uniform-orographic drag (not an inert control), a legitimate alternative.
    "gravity_wave_drag": [
        "none", "mcfarlane", "lindzen", "rayleigh",
    ],
}

_AXIS_TAG = {"convection": "conv", "turbulence": "turb",
             "gravity_wave_drag": "gwd"}


def build_combos():
    """Return the ordered list of combos (baseline first, then OAT)."""
    combos = [{"name": "combo_baseline", **BASELINE}]
    for axis, alts in SWEEP_SPACE.items():
        for alt in alts:
            if alt == BASELINE[axis]:
                continue  # baseline value is the shared combo_baseline run
            combo = dict(BASELINE)
            combo[axis] = alt
            combo["name"] = f"combo_{_AXIS_TAG[axis]}_{alt}"
            combos.append(combo)
    return combos


def _load_orchestrator():
    spec = importlib.util.spec_from_file_location(
        "_aimip_latlon_mod", REPO / "scripts" / "run" / "run_aimip_latlon.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_combo(index, output_root, extra_argv):
    combos = build_combos()
    if not 0 <= index < len(combos):
        raise SystemExit(
            f"combo index {index} out of range [0, {len(combos)}); "
            f"--list reports {len(combos)} combos."
        )
    combo = combos[index]
    mod = _load_orchestrator()
    argv = [
        "--variants", "classical",
        "--convection", combo["convection"],
        "--turbulence", combo["turbulence"],
        "--gravity-wave-drag", combo["gravity_wave_drag"],
        "--output-dir", str(Path(output_root) / combo["name"]),
        *extra_argv,
    ]
    print(f"[sweep] combo {index}/{len(combos)-1}: {combo['name']} -> {argv}")
    return mod.main(argv)


def main(argv=None):
    p = argparse.ArgumentParser(description="AIMIP lat-lon OAT physics sweep")
    p.add_argument("--list", action="store_true", help="print combo count + names")
    p.add_argument("--index", type=int, default=None, help="run combo i")
    p.add_argument("--output-root", default="results/aimip_latlon_sweep")
    args, extra = p.parse_known_args(argv)

    combos = build_combos()
    if args.list or args.index is None:
        print(f"{len(combos)} combos:")
        for i, c in enumerate(combos):
            print(f"  {i:2d}  {c['name']:32s} "
                  f"conv={c['convection']} turb={c['turbulence']} "
                  f"gwd={c['gravity_wave_drag']}")
        return 0
    return run_combo(args.index, args.output_root, extra)


if __name__ == "__main__":
    raise SystemExit(main())
