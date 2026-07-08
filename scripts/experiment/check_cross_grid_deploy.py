"""Cross-GRID deploy-check PREFLIGHT for a compare-reanalysis env-kernel.

The companion ``check_campaign_deploy.py`` preflights a SAME-grid deploy (the campaign's
per-column field onto a base config on the SAME grid).  This is the CROSS-grid analog
(iters 69/70/475): a campaign run with ``--feedback-strategy environment`` also writes a
``<out>.env_kernel.json`` — the RAW env->coefficient regression — that deploys on a
DIFFERENT (e.g. finer) production grid by ENVIRONMENTAL similarity, so a cheap low-res
campaign can inform an expensive high-res run.

Unlike the same-grid field, the cross-grid override needs the TARGET grid's ENVIRONMENT,
so the operator runs the target model ONCE and passes its restart here.  This tool, in
SECONDS and without re-running the model, confirms the kernel deploys onto the target:

  1. ``load_base_config_and_grid(--target-base-config)`` -> the target grid/sigma (built
     the EXACT way the run builds them; the grid/nlev/vertical-coord come from the config).
  2. ``load_model_from_restart(--target-restart, ...)`` -> the target ColumnState (with the
     prescribed/coupled SST from ``--sst-npz``, the env-kernel's DOMINANT predictor).
  3. ``build_env_kernel_deployed_override(...)`` -> evaluates the kernel on the target
     environment and ``validate_strict``-checks the deployed config, returning a coverage
     diagnostic (``fraction_covered`` / ``fraction_in_hull`` / ``sst_from_model``).

It then reports the per-target-column override range + the trust diagnostic, and FAILS
LOUD (exit 1) on a near-no-op (out-of-hull) deploy (``--min-fraction-covered``) or an
approximate-SST environment (default; ``--allow-approximate-sst`` to proceed knowingly).
``turbulence_override`` is intentionally NOT serialized (iter 205-207), so — like the
same-grid path — there is no "deployed config file": inject the override at RUNTIME via
``apply_env_kernel_override(kernel, target_env)`` in the production driver.  Validate the
transfer FIRST in a twin with ``run_perfect_model_osse.py --fine-resolution``.

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


def _build_arg_parser() -> argparse.ArgumentParser:
    """The CLI parser as a factory so the flag contract is unit-testable."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", required=True,
                   help="the campaign's env-kernel JSON (<out>.env_kernel.json from "
                        "--feedback-strategy environment)")
    p.add_argument("--target-base-config", required=True,
                   help="the TARGET production ExperimentConfig JSON (its grid/nlev/"
                        "vertical-coord define the deploy grid; turbulence=clubb_lite)")
    p.add_argument("--target-restart", required=True,
                   help="a restart from ONE run of the target model (supplies the target "
                        "grid's environment for the kernel evaluation)")
    p.add_argument("--sst-npz", default="",
                   help="optional .npz with a grid-shaped 'sst_K' array — the target run's "
                        "prescribed/coupled SST (the env-kernel's DOMINANT predictor); "
                        "REQUIRED unless --allow-approximate-sst")
    p.add_argument("--min-fraction-covered", type=float, default=None,
                   help="fail loud (exit 1) when fewer than this fraction of target columns "
                        "find an environmentally-similar diagnosis (a near-no-op deploy)")
    p.add_argument("--allow-approximate-sst", action="store_true",
                   help="proceed even when no --sst-npz is given: the kernel's SST predictor "
                        "is fabricated from the lowest-level air temperature (untrustworthy "
                        "for a production deploy; default fails loud)")
    return p


def main(argv: list[str] | None = None) -> int:
    """Validate a cross-grid env-kernel deploy onto a target run; report the trust diagnostic."""
    args = _build_arg_parser().parse_args(argv)

    # Fail fast on a typo'd path BEFORE the (heavier) grid/restart machinery loads.
    for label, path in (("--kernel", args.kernel),
                        ("--target-base-config", args.target_base_config),
                        ("--target-restart", args.target_restart)):
        if not Path(path).is_file():
            raise SystemExit(f"[xgrid-deploy] {label} does not exist: {path!r}")
    if args.sst_npz and not Path(args.sst_npz).is_file():
        raise SystemExit(f"[xgrid-deploy] --sst-npz does not exist: {args.sst_npz!r}")

    import jax.numpy as jnp
    import numpy as np

    from scripts.experiment.check_campaign_deploy import (
        build_env_kernel_deployed_override,
    )
    from scripts.run.run_correction_campaign import load_base_config_and_grid
    from scripts.validate.compare_amip_era5 import load_model_from_restart

    base_cfg, grid, sigma = load_base_config_and_grid(args.target_base_config)
    gc = base_cfg.grid
    sst_K = (jnp.asarray(np.load(args.sst_npz)["sst_K"]) if args.sst_npz else None)
    # The grid/nlev/vertical-coord come from the SAME base config the deploy targets, so the
    # restart's state lands on the EXACT levels the deployed config will run on (a mismatch
    # against the restart's own recorded coordinate fails loud inside load_model_from_restart).
    target_state = load_model_from_restart(
        args.target_restart, grid, sigma, gc.nlev, sst_K=sst_K,
        expected_vertical_coord=gc.vertical_coord, expected_p_top_Pa=gc.p_top_Pa)

    deployed, dgrid, coverage = build_env_kernel_deployed_override(
        args.kernel, args.target_base_config, target_state,
        min_fraction_covered=args.min_fraction_covered,
        require_sst=not args.allow_approximate_sst)

    clubb = deployed.turbulence_override.clubb_lite
    field = next(f for f in ("C_K", "Pr_t", "C_eps")
                 if np.asarray(getattr(clubb, f)).ndim >= 1)
    arr = np.asarray(getattr(clubb, field)).reshape(-1)
    print(f"[xgrid-deploy] OK: env-kernel deploys onto the target grid "
          f"{tuple(int(d) for d in getattr(dgrid, 'grid_shape_2d', ()))} "
          f"({coverage['n_columns']} columns); {field} in "
          f"[{float(arr.min()):.4g}, {float(arr.max()):.4g}].")
    print(f"    coverage: fraction_covered={coverage['fraction_covered']:.3g}, "
          f"fraction_in_hull={coverage['fraction_in_hull']:.3g}, "
          f"sst_from_model={coverage['sst_from_model']}")
    print("    inject at RUNTIME via apply_env_kernel_override(kernel, target_env) in the "
          "production driver (turbulence_override is not serialized). Validate the transfer "
          "first with run_perfect_model_osse.py --fine-resolution.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
