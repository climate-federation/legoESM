"""CLI: spin off a forced column LES per worst column + write the coefficients.

Thin driver around :mod:`legoesm.atmosphere.dynamics.les.column_les` (the importable
library): load the worst-column manifest + an AMIP restart, run a forced plane
LES for each flagged column, and write the diagnosed closure coefficients for the
feedback field.  The library lives in the package so production code + tests
import it cleanly (no ``scripts``-importing-``scripts``); this script is only the
manifest-looping CLI.
"""

from __future__ import annotations

import argparse


def _realism_verdict(breakdown) -> str:
    """One-line operator verdict from a ``LESRealismBreakdown`` (iter 508/511): ``REALISTIC``,
    or ``REJECTED(<failing criteria>)`` naming WHY the spin-off LES is untrustworthy — so the
    per-column debug run says whether a diagnosed coefficient can be believed and, if not,
    which mode fired (no turbulence / blow-up / θ-drift / moisture runaway / supersaturation).
    """
    if bool(breakdown.overall):
        return "REALISTIC"
    reasons = [name for name, ok in (
        ("not_turbulent", breakdown.turbulent),
        ("not_finite", breakdown.finite),
        ("thermo_drift", breakdown.thermo_consistent),
        ("moisture_runaway", breakdown.moisture_physical),
        ("supersaturated", breakdown.rh_ok),
    ) if not bool(ok)]
    return "REJECTED(" + ",".join(reasons) + ")"


def _valid_levels_note(diagnosis) -> str:
    """`' — N/M valid diagnosis levels'` for a per-level (profile) diagnosis (iter 524): when
    the realism verdict is REALISTIC yet N=0, the rejection is the DIAGNOSIS validity
    (insufficient resolved shear/variance for a down-gradient closure, too few valid levels, or
    the top-sponge exclusion) — NOT the realism gate, so the operator looks at the LES
    resolution/forcing, not turbulence development (iter 507's actual '0 valid' cause). Empty
    for a SCALAR diagnosis (entrainment ``w_e`` — a single inversion level, not a column
    average)."""
    valid = getattr(diagnosis, "valid", None)
    if valid is None or getattr(valid, "ndim", 0) == 0:
        return ""
    return f" — {int(valid.sum())}/{int(valid.size)} valid diagnosis levels"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Column-LES spin-off + coefficient.")
    p.add_argument("--manifest", required=True, help="worst-column JSON manifest")
    p.add_argument("--restart", required=True, help="AMIP restart for the columns")
    p.add_argument("--resolution", type=int, required=True, help="GCM resolution")
    p.add_argument("--nlev", type=int, required=True, help="GCM nlev")
    p.add_argument("--dt", type=float, default=0.5,
                   help="LES timestep [s]. Default 0.5 keeps the acoustic Courant < 1 "
                        "at the shallow-regime dx=50 m; a larger dt is rejected by "
                        "run_forced_les' acoustic-CFL pre-flight.")
    p.add_argument("--hours", type=float, default=2.0, help="LES duration [h]")
    p.add_argument("--method", default="eddy_diffusivity")
    p.add_argument("--out", default="column_coefficients.npz")
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI: load the manifest + AMIP restart (lat-lon), run a forced LES per
    worst column, and write the diagnosed coefficients for the feedback field.

    The library symbols are imported HERE (not at module scope) so this script's
    public namespace is the CLI only — it is not a re-export shim for the package
    library (CLAUDE.md).
    """
    import numpy as np
    from legoesm.atmosphere.dynamics.les.column_les import (
        ColumnLESConfig,
        ColumnLESSetup,
        coefficient_value,
        process_column,
        run_forced_les,
    )
    from legoesm.atmosphere.dynamics.les.column_les_diagnosis import (
        column_les_realism_breakdown,
    )
    from legoesm.driver.restart import load_restart
    from legoesm.grids.factory import create_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.training.column_manifest import read_manifest

    args = _build_arg_parser().parse_args(argv)
    grid = create_grid("latlon", resolution=args.resolution)
    sigma = create_sigma_coordinate(args.nlev)
    loaded = load_restart(args.restart, grid, sigma, strict=True)
    state, q_v = loaded[0], loaded[1]
    records = read_manifest(args.manifest)
    config = ColumnLESConfig(diagnosis_method=args.method)
    n_steps = int(args.hours * 3600.0 / args.dt)

    # Capture the finished LES state (the run_les_fn return) so the CLI can report the
    # per-column realism breakdown — the SAME default-threshold gate process_column applied
    # — without threading it through the library (iter 511). The loop is host-side (one
    # process_column per column), so a concrete capture is safe.
    captured: dict = {}

    def _run(setup: ColumnLESSetup):
        final_state = run_forced_les(setup, dt_s=args.dt, n_steps=n_steps)
        captured["state"], captured["hc"] = final_state, setup.height_coord
        return final_state

    results = {}
    for rec in records:
        diag = process_column(
            rec, T=state.T, q_v=q_v, u=state.u, v=state.v, p_s=state.p_s,
            grid=grid, sigma=sigma, config=config, run_les_fn=_run,
        )
        results[str(tuple(rec.grid_index))] = np.asarray(
            coefficient_value(diag, config.diagnosis_method)
        )
        # The realism verdict reflects the finished LES state, so it is shown only when the
        # LES actually ran (``_run`` was invoked → ``captured`` populated); a real run always
        # populates it.
        realism = ""
        if "state" in captured:
            realism = " — realism: " + _realism_verdict(
                column_les_realism_breakdown(captured["state"], captured["hc"]))
        levels = _valid_levels_note(diag)   # REALISTIC + 0 valid levels ⇒ a DIAGNOSIS rejection
        print(f"[column-LES] {rec.grid_index} "
              f"({rec.regime if hasattr(rec, 'regime') else ''}) diagnosed{realism}{levels}.")
    np.savez(args.out, **results)
    print(f"[column-LES] wrote {len(results)} column coefficients to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
