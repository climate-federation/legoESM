"""CLI: spin off a forced column LES per worst column + write the coefficients.

Thin driver around :mod:`legoesm.atmosphere.dynamics.column_les` (the importable
library): load the worst-column manifest + an AMIP restart, run a forced plane
LES for each flagged column, and write the diagnosed closure coefficients for the
feedback field.  The library lives in the package so production code + tests
import it cleanly (no ``scripts``-importing-``scripts``); this script is only the
manifest-looping CLI.
"""

from __future__ import annotations

import argparse


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
    from legoesm.atmosphere.dynamics.column_les import (
        ColumnLESConfig,
        ColumnLESSetup,
        coefficient_value,
        process_column,
        run_forced_les,
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

    def _run(setup: ColumnLESSetup):
        return run_forced_les(setup, dt_s=args.dt, n_steps=n_steps)

    results = {}
    for rec in records:
        diag = process_column(
            rec, T=state.T, q_v=q_v, u=state.u, v=state.v, p_s=state.p_s,
            grid=grid, sigma=sigma, config=config, run_les_fn=_run,
        )
        results[str(tuple(rec.grid_index))] = np.asarray(
            coefficient_value(diag, config.diagnosis_method)
        )
        print(f"[column-LES] {rec.grid_index} "
              f"({rec.regime if hasattr(rec, 'regime') else ''}) diagnosed.")
    np.savez(args.out, **results)
    print(f"[column-LES] wrote {len(results)} column coefficients to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
