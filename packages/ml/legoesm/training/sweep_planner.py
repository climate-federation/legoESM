"""Shared OAT (one-at-a-time) classical-physics sweep planner (D6).

One implementation behind ``scripts/run/run_aimip_classical_sweep_stage1.py``
and ``scripts/run/run_wb_sweep_stage1.py`` (previously clones): combo
enumeration, per-combo suite/overlay YAML writing, manifest emission, and
the classical-sweep radiation gate. The entry scripts keep only their
campaign-specific tables (baseline, sweep space, paths) and their runner
sbatch template (worktree/env/walltime genuinely differ per campaign).

Import-light and JAX-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

# Short per-dimension tokens for combo dirnames (ASCII, sortable).
DIM_SHORT_NAMES = {
    "aimip_convection": "conv",
    "aimip_turbulence": "turb",
    "aimip_gwd": "gwd",
    "aimip_microphysics": "micro",
    "aimip_cloud": "cloud",
}


def combo_name(dim: str, scheme: str) -> str:
    """Per-combo dirname, e.g. ``combo_conv_bechtold``."""
    if dim not in DIM_SHORT_NAMES:
        raise ValueError(
            f"Unknown sweep dimension {dim!r}; valid = "
            f"{sorted(DIM_SHORT_NAMES)}."
        )
    return f"combo_{DIM_SHORT_NAMES[dim]}_{scheme}"


def build_oat_combos(baseline: dict, sweep_space: dict) -> list[dict]:
    """Enumerate the OAT sweep: baseline once + one combo per alternative.

    Each combo dict carries ``name`` / ``dim`` / ``scheme`` / ``overrides``
    (the full baseline with exactly one dimension swapped). Unknown sweep
    dimensions raise (via :func:`combo_name`).
    """
    combos: list[dict] = [{
        "name": "combo_baseline",
        "dim": "baseline",
        "scheme": "+".join(str(v) for v in baseline.values()),
        "overrides": dict(baseline),
    }]
    for dim, alts in sweep_space.items():
        for alt in alts:
            overrides = dict(baseline)
            overrides[dim] = alt
            combos.append({
                "name": combo_name(dim, alt),
                "dim": dim,
                "scheme": alt,
                "overrides": overrides,
            })
    return combos


def validate_sweep_baseline(repo_root: Path, baseline_rel: Path) -> dict:
    """Load the sweep's baseline YAML and enforce the classical-sweep gates.

    The baseline file must exist and must pin ``aimip_radiation: rrtmgp``
    (the swap campaign holds radiation fixed — mirrors
    ``campaign_driver.validate_classical_radiation``). Returns the loaded
    YAML dict.
    """
    baseline_path = repo_root / baseline_rel
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline: {baseline_path}")
    with baseline_path.open() as fh:
        base_yaml = yaml.safe_load(fh) or {}
    radiation = str(base_yaml.get("aimip_radiation", "rrtmgp"))
    if radiation != "rrtmgp":
        raise ValueError(
            f"Invalid aimip_radiation {radiation!r} in sweep baseline "
            f"{baseline_rel}: must be 'rrtmgp' — the classical scheme-swap "
            "campaign holds the radiation backend fixed."
        )
    return base_yaml


def write_sweep_plan(
    repo_root: Path,
    *,
    campaign: str,
    combos: list[dict],
    sweep_dir_rel: Path,
    results_dir_rel: Path,
    baseline: dict,
    baseline_rel: Path,
) -> Path:
    """Write per-combo configs + manifest.json; returns the manifest path.

    Idempotent, submits no jobs. Also pre-creates the sweep's
    ``slurm_logs`` dir (SLURM opens ``#SBATCH --output`` before the job
    body runs, so it must exist at submission time). Mutates each combo
    dict to add its repo-relative ``suite`` path (recorded in the
    manifest; the array runner indexes it).
    """
    for combo in combos:
        combo_dir = repo_root / sweep_dir_rel / combo["name"]
        combo_dir.mkdir(parents=True, exist_ok=True)
        suite_yaml = {
            "base": str(baseline_rel),
            "variants": ["classical"],
            "output_dir": str(results_dir_rel / combo["name"]),
        }
        with (combo_dir / "suite.yaml").open("w") as fh:
            yaml.safe_dump(suite_yaml, fh, sort_keys=False)
        with (combo_dir / "variant_classical.yaml").open("w") as fh:
            yaml.safe_dump(dict(combo["overrides"]), fh, sort_keys=False)
        combo["suite"] = str(sweep_dir_rel / combo["name"] / "suite.yaml")

    manifest = {
        "stage": 1,
        "campaign": campaign,
        "baseline": dict(baseline),
        "baseline_yaml": str(baseline_rel),
        "results_dir": str(results_dir_rel),
        "n_combos": len(combos),
        "combos": combos,
    }
    manifest_path = repo_root / sweep_dir_rel / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (repo_root / results_dir_rel / "slurm_logs").mkdir(
        parents=True, exist_ok=True
    )
    return manifest_path
