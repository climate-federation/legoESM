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


def build_env_kernel_deployed_override(
    kernel_path: str,
    target_base_config_path: str,
    target_model_state,
    *,
    min_fraction_covered: float | None = None,
):
    """CROSS-GRID analog of :func:`build_deployed_config` (iter 475): deploy a campaign's RAW
    env-kernel (``<out>.env_kernel.json`` from ``--feedback-strategy environment``) onto a
    DIFFERENT (e.g. finer) production grid by environmental similarity — the iter-69/70
    grid-agnostic deploy.

    ``build_deployed_config`` needs the per-column field on the SAME grid; this needs the TARGET
    grid's ENVIRONMENT, which requires the target model's state — so the operator runs the target
    model ONCE and passes its :class:`~legoesm.training.compare_reanalysis.ColumnState` as
    ``target_model_state``.  Loads the kernel, builds the target grid/sigma from
    ``target_base_config_path``, evaluates the kernel on the target environment (the SAME
    ``column_environment_grid`` the campaign used — hybrid-coordinate-correct via the true layer
    pressures), and returns ``(deployed_config, grid, coverage)``.

    The ``coverage`` diagnostic (``fraction_covered`` / ``fraction_in_hull``) reports how
    trustworthy the transfer is; pass ``min_fraction_covered`` to FAIL LOUD on a near-no-op
    (out-of-hull) deploy.  Validate the transfer first with
    ``run_perfect_model_osse.py --fine-resolution`` (the twin go/no-go for this exact path)."""
    import json

    from legoesm.training.deploy_correction import (
        apply_env_kernel_override,
        env_kernel_from_dict,
    )
    from legoesm.training.feedback_assembly import column_environment_grid

    from scripts.run.run_correction_campaign import load_base_config_and_grid

    with open(kernel_path) as f:
        kernel = env_kernel_from_dict(json.load(f))
    base_cfg, grid, sigma = load_base_config_and_grid(target_base_config_path)
    p_s = target_model_state.p_s
    grid_env, _ = column_environment_grid(
        target_model_state, sigma,
        p_full=sigma.pressure_at_full(p_s), p_half=sigma.pressure_at_half(p_s))
    override, coverage = apply_env_kernel_override(
        kernel, grid_env, min_fraction_covered=min_fraction_covered)
    deployed = base_cfg._replace(turbulence_override=override)
    deployed.validate_strict()
    return deployed, grid, coverage


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

    from legoesm.training.feedback import param_field_bounds

    clubb = deployed.turbulence_override.clubb_lite
    corrected: dict = {}
    n_columns = 0
    for field in _CLUBB_FIELDS:
        arr = np.asarray(getattr(clubb, field))
        if arr.ndim >= 1:                                   # a per-column field => corrected
            n_columns = int(arr.reshape(-1).shape[0])
            entry = {"min": float(arr.min()), "max": float(arr.max())}
            # Flag a deployed coefficient outside its calibratable (param-spec) range. The
            # campaign CLIPS to bounds by default, so an out-of-range deploy means either
            # --allow-unphysical-coeff was used OR the output was hand-edited/corrupted — and
            # the production physics uses the value RAW (no deploy-time clip), so surface it.
            bounds = param_field_bounds(clubb, field)
            if bounds is not None:
                lo, hi = bounds
                entry["bounds"] = [lo, hi]
                entry["in_bounds"] = bool(entry["min"] >= lo and entry["max"] <= hi)
            corrected[field] = entry
    # The averaging-window provenance (iter 267): the DEPLOYER (possibly a different
    # person, weeks after the campaign) confirms the SOURCE climate before deploying a
    # saved correction — a snapshot-trained correction is a different quantity than a
    # climatology-trained one. Re-read the raw output (build_deployed_config consumed it
    # for the field; this picks up the metadata it does not need).
    import json

    with open(campaign_output_path) as f:
        raw = json.load(f)
    averaging = raw.get("averaging")
    # The campaign HEALTH verdict (iter 287/312): the output JSON is written even on a
    # non-zero campaign exit (only the exit status gates), so a deployer reading a saved
    # output — possibly without having seen the campaign's exit code — must know whether
    # it actually IMPROVED the bias vs stalled / produced no valid LES diagnoses.
    health = raw.get("health")
    # Cross-grid env-kernel availability (iter 474): a campaign run with
    # --feedback-strategy environment ALSO writes a `<output>.env_kernel.json` — the RAW
    # env->coefficient regression that deploys on ANY grid by environmental similarity
    # (apply_env_kernel_override). The same-grid deploy above uses the per-column field; surface
    # that the cross-grid option exists so a deployer (possibly different from the campaign
    # operator) knows it can also deploy on a DIFFERENT grid, not just this one.
    import os
    env_kernel = None
    _kernel_path = f"{campaign_output_path}.env_kernel.json"
    if os.path.exists(_kernel_path):
        try:
            with open(_kernel_path) as f:
                _k = json.load(f)
            # Only report a genuine env-kernel artifact (env_kernel_to_dict stamps "artifact":
            # "raw_environment_kernel") — never mistake another JSON sharing the name.
            if _k.get("artifact") == "raw_environment_kernel":
                # The TRAINING-env hull (iter 475/476): the [lo, hi] range of each predictor the
                # kernel was trained on, so the operator can judge cross-grid transferability —
                # a target grid whose environments fall OUTSIDE this is out-of-hull (low coverage,
                # unreliable). Both REQUIRED in the env-kernel JSON (env_kernel_from_dict).
                _names = _k.get("predictor_names", ["sst_K", "cape_J_kg", "bulk_shear_m_s"])
                _lo, _hi = _k.get("env_lo"), _k.get("env_hi")
                hull = (None if _lo is None or _hi is None else
                        {n: [float(lo), float(hi)] for n, lo, hi in zip(_names, _lo, _hi)})
                env_kernel = {"path": _kernel_path, "field": _k.get("field"),
                              "n_samples": len(_k.get("sample_env", [])), "hull": hull}
        except (ValueError, OSError):
            env_kernel = None
    return {
        "n_columns": n_columns,
        "corrected": corrected,
        "grid_shape": tuple(int(d) for d in getattr(grid, "grid_shape_2d", ())),
        "averaging": averaging,
        "health": health,
        "env_kernel": env_kernel,
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
        line = f"    {field}: [{rng['min']:.4g}, {rng['max']:.4g}]"
        if rng.get("bounds") is not None:
            lo, hi = rng["bounds"]
            line += f" (calibratable range [{lo:.4g}, {hi:.4g}])"
            if rng.get("in_bounds") is False:
                line += "  ** OUTSIDE the calibratable range **"
        print(line)
    oob = sorted(f for f, rng in fields.items() if rng.get("in_bounds") is False)
    if oob:                                                 # not a hard fail: see message
        print(f"[deploy-check] WARNING: {oob} deploy coefficient(s) lie OUTSIDE their "
              "calibratable range — expected ONLY if the campaign used "
              "--allow-unphysical-coeff; otherwise suspect a hand-edited / corrupted output. "
              "The production physics uses the value RAW (no deploy-time clip), so an "
              "unphysical coefficient may DESTABILIZE the run.")
    health = stats.get("health")
    if health and health.get("status"):                    # campaign outcome provenance (312)
        status = health["status"]
        print(f"    campaign health: {status} — {health.get('message', '')}")
        if status != "improved":
            print(f"[deploy-check] WARNING: the campaign verdict was '{status}', NOT "
                  "'improved' — this correction did NOT lower the bias in the campaign "
                  "(the monotonic gate keeps the best-so-far field, which for a stalled / "
                  "no_valid_diagnoses run may be the unchanged background). Confirm the "
                  "campaign outcome before deploying.")
    av = stats.get("averaging")
    if av:                                                  # source-climate provenance (267)
        n = int(av.get("era5_n_times", 1))
        idx = av.get("era5_time_idx")
        kind = "a SINGLE ERA5 snapshot" if n == 1 else f"an {n}-time ERA5 climatology"
        note = ("  (a snapshot-trained correction — confirm this matches your deploy "
                "intent)") if n == 1 else ""
        print(f"    source: corrected against {kind} @ idx {idx}{note}")
        md = av.get("model_days")                          # the MODEL-side window (iter 313)
        if md is not None:
            print(f"    model climatology window: {int(md)} days "
                  f"({av.get('model_n_samples')} samples @ "
                  f"{av.get('model_diag_days')}-day cadence) — confirm it is comparable to "
                  "the ERA5 window above (both time-means, not a snapshot).")
    ek = stats.get("env_kernel")
    if ek:                                                 # cross-grid kernel available (474)
        print(f"    cross-grid: an env-kernel ({ek.get('field')}, {ek.get('n_samples')} "
              f"samples) is ALSO available at {ek['path']} — the same-grid deploy above uses "
              "the per-column field; deploy on a DIFFERENT grid via "
              "check_campaign_deploy.build_env_kernel_deployed_override (validate the transfer "
              "first with run_perfect_model_osse.py --fine-resolution).")
        if ek.get("hull"):                                 # the training-env hull (476)
            hull_str = ", ".join(f"{n}∈[{lo:.4g},{hi:.4g}]"
                                 for n, (lo, hi) in ek["hull"].items())
            print(f"      trained-env hull: {hull_str} — a target grid OUTSIDE this is "
                  "out-of-hull (low coverage; the transfer falls back to the background).")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
