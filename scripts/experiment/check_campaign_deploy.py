"""Deploy-check PREFLIGHT for a finished compare-reanalysis campaign.

After a (multi-day) campaign writes its corrected-coefficient JSON, the operator
DEPLOYS it by injecting it at RUNTIME into a production AMIP/CMIP ``ExperimentConfig``
(``turbulence_override`` is intentionally NOT serialized -- iter 205-207 -- so there
is no "deployed config file"; the correction rides the config pytree at run time).
This tool confirms, in SECONDS and WITHOUT running the model, that a given campaign
output actually deploys onto a given base config + production grid:

  1. ``load_base_config_and_grid`` -> the base ``ExperimentConfig`` + its grid/sigma
     (built the EXACT way the run builds them).
  2. ``corrected_turbulence_override(output, grid=...)`` -> the per-column
     ``TurbulenceConfig`` override, GRID-VERIFIED (a field learned on a different grid
     is refused -- iter 58 -- unless ``--allow-unverified-grid``).
  3. ``base._replace(turbulence_override=...).validate_strict()`` -> the full deployed
     config is schema-sound (e.g. the base ``turbulence`` is ``clubb_lite``).

It then reports WHICH coefficients were corrected (C_K / Pr_t / C_eps), the column
count, and each field's value range -- so the operator sees what the deploy will
change before committing the expensive production run.  A green check means the
runtime injection (``corrected_turbulence_override`` in the production driver) will
succeed; this is the deploy analog of ``smoke_compare_reanalysis.py`` (which
preflights the campaign INPUT) for the campaign OUTPUT.

See ``docs/COMPARE_REANALYSIS.md`` for the full deploy + held-out-verify workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the sibling ``scripts.*`` entry points importable when run as a standalone CLI.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CLUBB_FIELDS = ("C_K", "Pr_t", "C_eps")


def build_deployed_config(
    base_config_path: str,
    campaign_output_path: str,
    *,
    allow_unverified_grid: bool = False,
):
    """Load the base config + campaign output -> a DEPLOYED ``ExperimentConfig`` + grid.

    Returns ``(deployed_config, grid)``: the base config with ``turbulence_override``
    set to the GRID-VERIFIED per-column correction, ``validate_strict``-checked and
    ready to feed a production driver.  Because the override is a RUNTIME pytree leaf
    (it does NOT serialize -- iters 205-207), this is the canonical way to USE a
    campaign output in a fresh run::

        cfg, grid = build_deployed_config(base_json, campaign_out_json)
        build_driver, _ = make_base_driver_builder("amip")   # or "cmip"
        driver = build_driver(cfg); driver.run(...)          # the LES-corrected run

    Raises on a grid mismatch (``corrected_turbulence_override``), a non-``clubb_lite``
    base, or any ``validate_strict`` failure.
    """
    from legoesm.training.deploy_correction import corrected_turbulence_override

    from scripts.run.run_correction_campaign import load_base_config_and_grid

    base_cfg, grid, _sigma = load_base_config_and_grid(base_config_path)
    override = corrected_turbulence_override(
        campaign_output_path, grid=grid, allow_unverified_grid=allow_unverified_grid)
    deployed = base_cfg._replace(turbulence_override=override)
    deployed.validate_strict()   # full deployed config schema-sound (base is clubb_lite, …)
    return deployed, grid


def check_deploy(
    base_config_path: str,
    campaign_output_path: str,
    *,
    allow_unverified_grid: bool = False,
) -> dict:
    """Validate that ``campaign_output_path`` deploys onto ``base_config_path``.

    Returns a stats dict ``{"n_columns", "corrected": {field: {min,max}}, "grid_shape"}``.
    Raises (``ValueError``/``SystemExit``) on a grid mismatch, a non-``clubb_lite`` base,
    or any ``validate_strict`` failure -- exactly the failures that would otherwise
    abort the production run AFTER it started.
    """
    import numpy as np

    deployed, grid = build_deployed_config(
        base_config_path, campaign_output_path,
        allow_unverified_grid=allow_unverified_grid)

    clubb = deployed.turbulence_override.clubb_lite
    corrected: dict = {}
    n_columns = 0
    for field in _CLUBB_FIELDS:
        arr = np.asarray(getattr(clubb, field))
        if arr.ndim >= 1:                                   # a per-column field => corrected
            n_columns = int(arr.reshape(-1).shape[0])
            corrected[field] = {
                "min": float(arr.min()), "max": float(arr.max())}
    return {
        "n_columns": n_columns,
        "corrected": corrected,
        "grid_shape": tuple(int(d) for d in getattr(grid, "grid_shape_2d", ())),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", required=True,
                   help="base production ExperimentConfig JSON (turbulence=clubb_lite)")
    p.add_argument("--campaign-output", required=True,
                   help="the campaign's corrected-coefficient JSON (its --out)")
    p.add_argument("--allow-unverified-grid", action="store_true",
                   help="deploy on the array-length check alone when the output predates "
                        "grid provenance (UNSAFE if the grids differ)")
    args = p.parse_args(argv)

    stats = check_deploy(
        args.base_config, args.campaign_output,
        allow_unverified_grid=args.allow_unverified_grid)
    fields = stats["corrected"]
    if not fields:
        print("[deploy-check] WARNING: the campaign output corrected NO coefficient "
              "(all scalar defaults) — the deploy would be a no-op. Check the campaign "
              "health verdict.")
        return 1
    print(f"[deploy-check] OK: {stats['n_columns']} columns, grid {stats['grid_shape']}, "
          f"corrected {sorted(fields)}. The deploy is grid-compatible and "
          "validate_strict-sound; inject it at runtime via corrected_turbulence_override.")
    for field, rng in sorted(fields.items()):
        print(f"    {field}: [{rng['min']:.4g}, {rng['max']:.4g}]")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
