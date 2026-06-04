"""Phase G.0c — tier-2 tendency comparison: legoESM-Veros mode vs Veros ACC.

End-to-end driver that runs Veros's canonical ACC setup (``acc_channel``)
for one short step, extracts the per-process tendency arrays from the
Veros snapshot, builds the matching legoESM lat-lon C-grid state and
recipe (with all Veros constants pinned via config — see
``build_acc_model_config`` / ``ConstantsConfig`` / ``VEROS_CONSTANTS_CONFIG``),
runs the legoESM tendency probe, and writes a per-region comparison report.

Usage
-----

::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \\
        scripts/ocean_fidelity/compare_tendencies_acc.py \\
        --runlen-s 4800 \\
        --write-report docs/ocean_fidelity/veros_acc_tendency_comparison.md

Requires Veros installed in the legoESM venv (``pip install -e
/home/dbalwada/veros``). If Veros is missing the script emits a clear
install message and runs the legoESM-side smoke test (recipe build,
tendency probe on a rest state) so the harness can be reviewed even
without Veros available.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.ocean.fidelity.tendency_probe import (
    build_region_masks,
    compare_momentum_at_centres,
    compare_probe_results,
    per_region_metrics,
    probe_latlon_cgrid,
)
from legoesm.ocean.fidelity.veros_acc_recipe import (
    DT_MOM_S, DT_TRACER_S, build_acc_recipe,
)


_VEROS_INSTALL_HINT = (
    "Veros is not installed in this legoESM venv. To install:\n\n"
    "    .venv/bin/pip install -e /home/dbalwada/veros\n\n"
    "(or wherever you have the team-ocean/veros checkout). The cloned "
    "source is already at /home/dbalwada/veros if you ran the Phase G "
    "session's `git clone` step."
)


def _try_run_veros(runlen_s: float) -> tuple[object | None, str | None]:
    """Attempt to run Veros ACC for ``runlen_s`` seconds. Returns
    ``(result, None)`` on success; ``(None, error_message)`` on
    ImportError / VerosRunError."""
    try:
        from legoesm.ocean.fidelity.veros_runner import (
            VerosRunError, run_veros,
        )
        from legoesm.ocean.fidelity.veros_state_bridge import (
            VEROS_TENDENCY_CAPTURE_VARS,
        )
    except ImportError as exc:
        return None, f"ImportError: {exc}\n\n{_VEROS_INSTALL_HINT}"

    try:
        result = run_veros(
            "acc_channel",
            runlen_s=runlen_s,
            capture_vars=VEROS_TENDENCY_CAPTURE_VARS,
            force_recompute=False,
        )
        return result, None
    except (ImportError, VerosRunError) as exc:
        return None, f"{type(exc).__name__}: {exc}\n\n{_VEROS_INSTALL_HINT}"


def _run_legoesm_probe(recipe, base_state):
    """Compute legoESM per-process tendencies on the supplied state."""
    probe = probe_latlon_cgrid(
        base_state,
        recipe.grid, recipe.z_coord, recipe.model_config,
        dt=DT_MOM_S,
        dt_tracer=DT_TRACER_S,
    )
    return probe


def _write_report(
    output_path: Path,
    legoesm_probe,
    veros_tendencies: dict | None,
    masks,
    veros_status: str | None,
    tracer_metrics: dict | None = None,
) -> None:
    """Write a Markdown report with per-region metrics for each
    tendency field that has a matching Veros counterpart."""
    lines = ["# Phase G.0c — legoESM-Veros ACC tier-2 tendency comparison\n"]
    lines.append("Source recipe: ``legoesm.ocean.fidelity.veros_acc_recipe.build_acc_recipe``.")
    lines.append("Veros snapshot: ``veros.setups.acc.acc.ACCSetup`` via")
    lines.append("``legoesm.ocean.fidelity.veros_runner.run_veros``.\n")
    lines.append("Constants: pinned via config (``ConstantsConfig`` / ``VEROS_CONSTANTS_CONFIG``).\n")

    if veros_status is not None:
        lines.append("## Veros snapshot unavailable\n")
        lines.append("```\n" + veros_status + "\n```\n")
        lines.append(
            "The legoESM tendency probe still runs on a rest-state to "
            "exercise the recipe wiring; per-process comparison values "
            "below are absent until Veros is installed.\n"
        )

    lines.append("## legoESM probe summary\n")
    for name in legoesm_probe._fields:
        arr = np.asarray(getattr(legoesm_probe, name))
        lines.append(
            f"- ``{name}``: shape {arr.shape}, "
            f"min={float(arr.min()):.3e}, "
            f"max={float(arr.max()):.3e}, "
            f"L1={float(np.mean(np.abs(arr))):.3e}"
        )
    lines.append("")

    if veros_tendencies is not None:
        lines.append("## Per-region per-process metrics (legoESM vs Veros)\n")
        lines.append("| Veros field | mapped to legoESM | interior L2 | "
                     "boundary L2 | equator L2 | ML L2 | abyssal L2 | "
                     "pattern corr (interior) | sign-match (interior) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for veros_name, lego_arr in veros_tendencies.items():
            if veros_name in ("rho",):
                lego_field = "rho"
            elif veros_name == "coriolis_u":
                lego_field = "coriolis_u"
            elif veros_name == "coriolis_v":
                lego_field = "coriolis_v"
            elif veros_name.startswith("veros_d") and veros_name.endswith("_mix"):
                lego_field = (
                    "av_vert_u" if "u" in veros_name and veros_name.endswith("u_mix")
                    else "av_vert_v" if veros_name.endswith("v_mix")
                    else "(aggregate)"
                )
            else:
                lego_field = "(aggregate)"

            if lego_field == "rho":
                lego_data = getattr(legoesm_probe, "rho")
            elif lego_field in legoesm_probe._fields:
                lego_data = getattr(legoesm_probe, lego_field)
            else:
                lego_data = None

            if lego_data is None or lego_data.shape != lego_arr.shape:
                lines.append(
                    f"| {veros_name} | {lego_field} (shape mismatch / "
                    f"aggregate; deferred) | — | — | — | — | — | — | — |"
                )
                continue

            metrics = per_region_metrics(
                jnp.asarray(lego_data), jnp.asarray(lego_arr), masks,
            )
            row = (
                f"| {veros_name} | {lego_field} | "
                f"{metrics['interior']['L2']:.3e} | "
                f"{metrics['boundary']['L2']:.3e} | "
                f"{metrics['equator']['L2']:.3e} | "
                f"{metrics['mixed_layer']['L2']:.3e} | "
                f"{metrics['abyssal']['L2']:.3e} | "
                f"{metrics['interior']['pattern_corr']:.4f} | "
                f"{metrics['interior']['sign_match']:.4f} |"
            )
            lines.append(row)
        lines.append("")

        # Q2: per-process MOMENTUM comparison at cell centres (face->centre
        # interpolation + legoESM->Veros process aggregation). Replaces the
        # "shape mismatch / deferred" momentum rows above.
        lines.append("## Per-process MOMENTUM comparison at cell centres (Q2)\n")
        lines.append(
            "legoESM components aggregated to Veros groupings (du_adv=vortcor+vertadv, "
            "du_mix=av_vert+botdrag, du_cor 1:1), both interpolated to cell centres. "
            "The recipe now uses flux-form momentum advection (adopted, matching Veros). "
            "Observed (developed flow, 864000 s): du_cor/dv_cor corr ~0.96/0.98 (strong); "
            "du_adv/dv_adv corr ~0.65/0.71 (MODERATE, R²≈0.43/0.51). The advection residual "
            "is a per-process MAPPING floor — legoESM's vector-invariant decomposition "
            "(vortcor+vertadv) does not align cleanly with Veros's flux-form ∇·(uu) split, "
            "so the NET momentum tendency agrees better than any single per-process row; a "
            "total-tendency / energy-budget check (roadmap T2-3) side-steps this. NOT a bug, "
            "but NOT a clean per-process match either — see the strategy doc §8 ledger. "
            "`wsign` = sign-match over cells with `|veros| > 0.1·max` (dynamically "
            "significant; the plain sign-match is near-zero-cell noise for these processes).\n"
        )
        lines.append(
            "| process | interior L2 | interior corr | interior sign | interior wsign |"
        )
        lines.append("|---|---|---|---|---|")
        mom = compare_momentum_at_centres(legoesm_probe, veros_tendencies, masks)
        for proc, m in mom.items():
            it = m["interior"]
            lines.append(
                f"| {proc} | {it['L2']:.3e} | {it['pattern_corr']:.4f} | "
                f"{it['sign_match']:.4f} | "
                f"{it.get('weighted_sign_match', float('nan')):.4f} |"
            )
        lines.append("")

    if tracer_metrics:
        # Q2: per-process TRACER comparison at cell centres (no interpolation;
        # legoESM per-process isolated by differencing probe runs).
        lines.append("## Per-process TRACER comparison at cell centres (Q2)\n")
        lines.append(
            "legoESM per-process tracer tendencies isolated by differencing probe runs "
            "(full - scheme-off); tracers are cell-centred so no interpolation is needed. "
            "iso = GM/Redi isoneutral, vmix = vertical mixing. vmix is IMPLICIT in the ACC "
            "recipe, so legoESM produces no explicit vmix tendency (documented delta — see "
            "the strategy doc §8 ledger).\n"
        )
        lines.append("| process | interior L2 | interior corr | interior sign |")
        lines.append("|---|---|---|---|")
        for label, m in tracer_metrics.items():
            it = m["interior"]
            lines.append(
                f"| {label} | {it['L2']:.3e} | {it['pattern_corr']:.4f} | "
                f"{it['sign_match']:.4f} |"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Report written to {output_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runlen-s", type=float, default=float(DT_MOM_S),
                   help=f"Veros run length [s]; default {DT_MOM_S} (one dt_mom).")
    p.add_argument("--write-report", type=Path,
                   default=Path("docs/ocean_fidelity/veros_acc_tendency_comparison.md"),
                   help="Path for the Markdown comparison report.")
    args = p.parse_args()

    # Veros side (may fail with clear message if Veros isn't installed).
    print("==> Running Veros ACC reference...")
    veros_result, veros_status = _try_run_veros(args.runlen_s)
    if veros_status is not None:
        print(veros_status, file=sys.stderr)

    # legoESM side — runs regardless of Veros availability so the harness
    # is exercised end-to-end.
    # The recipe pins all Veros constants via config (G-C4: build_acc_grid
    # radius/omega + build_acc_model_config g/rho_0/constants/A_h), so no
    # override_constants context is needed.
    print("==> Building legoESM-Veros ACC recipe...")
    recipe = build_acc_recipe()
    if veros_result is not None:
        print("==> Bridging Veros snapshot to legoESM state...")
        from legoesm.ocean.fidelity.veros_state_bridge import (
            extract_veros_tendencies,
            veros_snapshot_to_legoesm_state,
        )
        bridged = veros_snapshot_to_legoesm_state(
            veros_result, recipe.initial_state,
        )
        print(f"    Veros snapshot shapes: {bridged.info!r}")
        probe_state = bridged.state
        veros_tendencies = extract_veros_tendencies(veros_result)
    else:
        probe_state = recipe.initial_state
        veros_tendencies = None

    print("==> Running legoESM tendency probe...")
    legoesm_probe = _run_legoesm_probe(recipe, probe_state)
    masks = build_region_masks(recipe.grid, recipe.z_coord, probe_state)

    tracer_metrics = None
    if veros_result is not None:
        # iso: legoESM GM/Redi tracer tendency (computed by the probe from the
        # top-level config.gm_redi) vs Veros dtemp_iso/dsalt_iso. (vmix is
        # implicit in legoESM -> no explicit tendency to compare; documented
        # delta in the §8 ledger, omitted.)
        tracer_metrics = {}
        for label, probe_field, vk in (
            ("T_iso", legoesm_probe.dT_gm_redi, "veros_dT_iso"),
            ("S_iso", legoesm_probe.dS_gm_redi, "veros_dS_iso"),
        ):
            if vk in veros_tendencies:
                tracer_metrics[label] = per_region_metrics(
                    jnp.asarray(probe_field), jnp.asarray(veros_tendencies[vk]),
                    masks,
                )

    print("==> Writing comparison report...")
    _write_report(
        args.write_report,
        legoesm_probe=legoesm_probe,
        veros_tendencies=veros_tendencies,
        masks=masks,
        veros_status=veros_status,
        tracer_metrics=tracer_metrics,
    )

    return 0 if veros_status is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
