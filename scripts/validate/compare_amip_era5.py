"""Compare a saved AMIP model snapshot to ERA5 and write a worst-column manifest.

Stage-2 driver of ``docs/COMPARE_REANALYSIS.md`` (non-matrix validator).  It
loads an AMIP restart checkpoint (the full 3-D model state) and an ERA5 slice,
regrids ERA5 → the model grid + sigma levels (the
:mod:`legoesm.training.era5_to_state` direction), then calls
:func:`legoesm.training.compare_reanalysis.compare_state_to_reference` to score
every model column and emit the ``top-N`` worst columns (with SST/CAPE/shear
environment tags) as a JSON manifest for the LES stage.

The data-agnostic comparison core lives in
:mod:`legoesm.training.compare_reanalysis`; this module is the thin glue that
(a) selects the right ERA5→grid regrid, (b) extracts grid-shaped lat/lon in
degrees, and (c) builds a :class:`ColumnState` from a restart checkpoint.  Those
helpers are importable and unit-tested; heavy I/O imports (ERA5, restart, grid
factory) are function-scoped so importing this module stays cheap.

Run::

    python scripts/validate/compare_amip_era5.py \
        --restart run/amip_chkpt.npz --grid-type cubed_sphere \
        --resolution 48 --nlev 40 \
        --era5-zarr gs://weatherbench2/.../era5.zarr --era5-time-idx 0 \
        --n-worst 20 --out worst_columns.json
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from legoesm.training.column_manifest import write_manifest
from legoesm.training.compare_reanalysis import (
    ColumnComparison,
    ColumnState,
    column_state_from_carry,
    compare_state_to_reference,
)

# Canonical grid-type tokens → aliases.  Anything else is a hard error.
_GRID_TYPE_ALIASES = {
    "spectral": "spectral",
    "gaussian": "spectral",
    "cubed_sphere": "cubed_sphere",
    "cubedsphere": "cubed_sphere",
    "cs": "cubed_sphere",
    "latlon": "latlon",
    "lat_lon": "latlon",
    "mpas": "mpas",
    "voronoi": "mpas",
    "icosahedral": "mpas",
}
_KNOWN_GRID_TYPES = sorted(set(_GRID_TYPE_ALIASES.values()))
# Canonical token → the token understood by ``legoesm.grids.factory.create_grid``
# (the factory spells the spectral/Gaussian grid "gaussian").
_FACTORY_GRID_TOKEN = {
    "spectral": "gaussian",
    "cubed_sphere": "cubed_sphere",
    "latlon": "latlon",
    "mpas": "mpas",
}


def canonical_grid_type(grid_type: str) -> str:
    """Map a grid-type token to its canonical form, raising on unknown.

    Dispatch hardening (CLAUDE.md): an unrecognized grid selects nothing, so we
    raise rather than silently defaulting to one regrid path.
    """
    key = str(grid_type).strip().lower()
    if key not in _GRID_TYPE_ALIASES:
        raise ValueError(
            f"Unknown grid_type {grid_type!r}; expected one of "
            f"{_KNOWN_GRID_TYPES} (or an alias)."
        )
    return _GRID_TYPE_ALIASES[key]


def select_era5_regrid(grid_type: str) -> Callable:
    """Return the ``era5_to_*_carry`` regrid for ``grid_type`` (raises on unknown).

    The returned callable has signature ``(era5_slice, grid, sigma) ->
    SegmentCarry`` (ERA5 → model grid + sigma).
    """
    canon = canonical_grid_type(grid_type)
    from legoesm.training import era5_to_state

    table = {
        "spectral": era5_to_state.era5_to_spectral_carry,
        "cubed_sphere": era5_to_state.era5_to_cubedsphere_carry,
        "latlon": era5_to_state.era5_to_latlon_carry,
        "mpas": era5_to_state.era5_to_mpas_carry,
    }
    return table[canon]


def sigma_levels(sigma: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract ``(sigma_full, sigma_half)`` arrays from a vertical coordinate."""
    return jnp.asarray(sigma.sigma_full), jnp.asarray(sigma.sigma_half)


def grid_lat_lon_deg(grid: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return grid-shaped lat/lon in **degrees** via the uniform grid accessors.

    Every legoESM grid exposes ``grid_lat`` / ``grid_lon`` (radians) at the
    horizontal cell centres with the grid's 2-D / cubed-sphere shape, so this
    works for cubed-sphere ``(6, n, n)`` and lat-lon / Gaussian ``(n_lat,
    n_lon)`` alike; the manifest builder consumes the grid-shaped coordinates
    directly.
    """
    rad2deg = 180.0 / jnp.pi
    return jnp.asarray(grid.grid_lat) * rad2deg, jnp.asarray(grid.grid_lon) * rad2deg


def model_state_from_restart(
    state: Any,
    q_v: jnp.ndarray,
    *,
    sst_K: jnp.ndarray | None = None,
    precip_mm_day: jnp.ndarray | None = None,
    mesh: Any = None,
    sigma: Any = None,
) -> ColumnState:
    """Build a model :class:`ColumnState` from a loaded restart ``state`` + ``q_v``.

    The restart ``state`` carries ``u``/``v``/``T``/``p_s`` (as ``Field``
    wrappers); ``q_v`` is loaded separately by
    :func:`legoesm.driver.restart.load_restart`.  Delegates to
    :func:`legoesm.training.compare_reanalysis.column_state_from_hydrostatic`,
    which unwraps the ``Field``\\ s.  SST (prescribed AMIP forcing) and precip are
    threaded in by the caller when available; when omitted the SST environment
    tag falls back to surface air temperature and the precip term is dropped.
    A SPECTRAL restart state is first synthesized to grid winds (``mesh`` = the
    model GaussianGrid, ``sigma`` the vertical coord); MPAS reconstructs the cell
    wind from the edge velocity via ``mesh``; lat-lon/cubed pass through.
    """
    from legoesm.training.compare_reanalysis import (
        column_state_from_hydrostatic,
        grid_winds_from_spectral,
    )

    state = grid_winds_from_spectral(state, mesh, sigma)
    return column_state_from_hydrostatic(
        state, q_v, sst_K=sst_K, precip_mm_day=precip_mm_day, mesh=mesh
    )


def compare_and_write(
    *,
    model: ColumnState,
    reference: ColumnState,
    sigma: Any,
    grid: Any,
    time_index: int,
    n_worst: int,
    out_path: str | None = None,
    valid_mask: jnp.ndarray | None = None,
) -> ColumnComparison:
    """Run the comparison and (optionally) write the manifest to ``out_path``."""
    sigma_full, sigma_half = sigma_levels(sigma)
    lat_deg, lon_deg = grid_lat_lon_deg(grid)
    result = compare_state_to_reference(
        model=model, reference=reference,
        sigma_full=sigma_full, sigma_half=sigma_half,
        lat_deg=lat_deg, lon_deg=lon_deg,
        time_index=time_index, n_worst=n_worst, valid_mask=valid_mask,
        # Weight the bias by the model's TRUE layer pressures (correct for a HYBRID
        # coordinate; iter 340) — byte-identical for pure-sigma (pressure_at_full == σ·p_s).
        p_full=sigma.pressure_at_full(model.p_s),
        p_half=sigma.pressure_at_half(model.p_s),
    )
    if out_path is not None:
        write_manifest(result.manifest, out_path)
    return result


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare AMIP snapshot to ERA5.")
    p.add_argument("--restart", required=True,
                   help="AMIP restart checkpoint (.npz/.zarr) — the run to score "
                        "(e.g. the LES-corrected deploy)")
    p.add_argument("--baseline-restart", default="",
                   help="Optional SECOND restart (the UNcorrected baseline) on the "
                        "same grid: when given, the per-variable global bias of "
                        "baseline-vs-corrected is reported (the clause-5 'did it "
                        "improve the biases' before/after check)")
    p.add_argument("--grid-type", required=True,
                   choices=sorted(_GRID_TYPE_ALIASES))
    p.add_argument("--resolution", type=int, required=True,
                   help="Grid resolution matching the checkpoint")
    p.add_argument("--nlev", type=int, required=True,
                   help="Number of sigma levels matching the checkpoint")
    p.add_argument("--vertical-coord", choices=("sigma", "hybrid"), default="hybrid",
                   help="The model's vertical coordinate (MUST match the checkpoint's run; "
                        "the GridConfig default is 'hybrid'). A mismatch puts the model state "
                        "and the regridded ERA5 reference on different pressure levels.")
    p.add_argument("--p-top-Pa", type=float, default=100.0,
                   help="Model-top pressure [Pa] for --vertical-coord hybrid (match the run)")
    p.add_argument("--era5-zarr", required=True,
                   help="ERA5 Zarr store (GCS or local)")
    p.add_argument("--era5-time-idx", type=int, default=0,
                   help="Time index into the ERA5 dataset")
    p.add_argument("--era5-cache", default="",
                   help="Optional local ERA5 cache dir")
    p.add_argument("--sst-npz", default="",
                   help="Optional .npz with a grid-shaped 'sst_K' array "
                        "(prescribed AMIP SST) for the environment tag; "
                        "without it the SST tag uses a surface-air proxy")
    p.add_argument("--n-worst", type=int, default=20)
    p.add_argument("--time-index", type=int, default=0,
                   help="Time index stamped into the manifest records")
    p.add_argument("--out", default="worst_columns.json")
    return p


def load_model_from_restart(restart_path, grid, sigma, nlev, *, sst_K=None,
                            expected_vertical_coord=None, expected_p_top_Pa=None):
    """Load a restart checkpoint into a comparison :class:`ColumnState`.

    Loads the restart, synthesizes grid winds (a no-op for a grid state; the MPAS
    branch reconstructs cell winds via ``mesh=grid``), shape/level-checks against
    the grid, and builds the model state.  Shared by the scored ``--restart`` and
    the optional ``--baseline-restart`` so both are loaded IDENTICALLY (same grid,
    sigma, SST source) — the before/after bias is then comparable.

    ``expected_vertical_coord`` (the operator's ``--vertical-coord``): when the
    restart RECORDS the run's coordinate (``loaded_config.grid.vertical_coord``),
    a mismatch is FAILED LOUDLY — placing the state on the wrong pressure levels
    (the warning the flag's help only described) would silently bias the compare.
    Defensive: skipped when the recorded coordinate is unavailable (legacy / no
    config), so it only raises on a CONFIRMED mismatch.
    """
    from legoesm.driver.restart import load_restart
    from legoesm.training.compare_reanalysis import grid_winds_from_spectral

    loaded = load_restart(restart_path, grid, sigma, strict=True)
    state, q_v = loaded[0], loaded[1]
    if expected_vertical_coord is not None and len(loaded) > 4:
        run_grid = getattr(loaded[4], "grid", None)
        run_vcoord = getattr(run_grid, "vertical_coord", None)
        if run_vcoord is not None and str(run_vcoord) != str(expected_vertical_coord):
            raise ValueError(
                f"--vertical-coord {str(expected_vertical_coord)!r} does not match the "
                f"restart's recorded run coordinate {str(run_vcoord)!r}; the model state "
                "would be placed on the WRONG pressure levels (a silently-biased compare). "
                f"Re-run with --vertical-coord {str(run_vcoord)}.")
        # For a HYBRID run, p_top also determines the A/B level coefficients
        # (``make_hybrid_levels``), so a p_top mismatch is the same class of wrong-levels
        # error (concentrated near the model top).  Checked only when both are hybrid and a
        # p_top expectation is given; defensive + relative-tolerant against float repr.
        if (run_vcoord == "hybrid" and str(expected_vertical_coord) == "hybrid"
                and expected_p_top_Pa is not None):
            run_p_top = getattr(run_grid, "p_top_Pa", None)
            if run_p_top is not None and abs(float(run_p_top) - float(expected_p_top_Pa)) > \
                    1e-6 * max(abs(float(run_p_top)), 1.0):
                raise ValueError(
                    f"--p-top-Pa {float(expected_p_top_Pa)} does not match the restart's "
                    f"recorded hybrid model top {float(run_p_top)} Pa; the hybrid levels "
                    f"would differ near the top. Re-run with --p-top-Pa {float(run_p_top)}.")
    state = grid_winds_from_spectral(state, grid, sigma)
    expected_cols = tuple(grid.grid_shape_2d)
    if tuple(state.T.shape[:-1]) != expected_cols:
        raise ValueError(
            f"restart column shape {tuple(state.T.shape[:-1])} != grid "
            f"{expected_cols}; --grid-type/--resolution must match the run.")
    if int(state.T.shape[-1]) != int(nlev):
        raise ValueError(f"restart nlev {state.T.shape[-1]} != --nlev {nlev}.")
    return model_state_from_restart(state, q_v, sst_K=sst_K, mesh=grid)


def main(argv: list[str] | None = None) -> int:
    """CLI: load AMIP snapshot + ERA5, regrid, compare, write manifest."""
    args = _build_arg_parser().parse_args(argv)
    canon = canonical_grid_type(args.grid_type)

    # Deferred heavy imports (kept out of module import so the unit-tested
    # helpers above load without the grid factory / ERA5 / restart machinery).
    from legoesm.grids.factory import create_grid
    from legoesm.grids.vertical import create_sigma_coordinate, make_hybrid_levels
    from legoesm.training.era5_to_state import (
        TrainingERA5Config,
        load_era5_slice,
    )

    grid = create_grid(_FACTORY_GRID_TOKEN[canon], resolution=args.resolution)
    # Build the SAME vertical coordinate the run used (default hybrid) so the model state,
    # the regridded ERA5 reference, and the bias mass weights are all on the model's TRUE
    # levels (iter 339/340).  Both coordinate types expose pressure_at_full/half.
    if args.vertical_coord == "hybrid":
        sigma = make_hybrid_levels(args.nlev, p_top_Pa=args.p_top_Pa)
    else:
        sigma = create_sigma_coordinate(args.nlev)

    # strict=True keeps the restart's reproducibility checks on (x64 / shape /
    # config-hash); a metadata mismatch surfaces instead of silently loading a
    # wrong checkpoint.  We then assert the loaded state matches the grid/sigma
    # we built, so a resolution/nlev mismatch fails loudly here, not as a
    # mis-shaped comparison downstream.
    # This one-shot CLI accepts BOTH grid- and spectral-format restarts: `load_restart`
    # reconstructs a spectral run's checkpoint (vor_hat/div_hat/... + spectral_layout)
    # into a SpectralHydrostaticState (iter 92, validating the coeff shapes vs this
    # grid/sigma under strict), and `grid_winds_from_spectral` BELOW synthesizes its
    # grid winds (a no-op for a grid state) so the compare is grid-general.
    if args.sst_npz:
        import numpy as np

        sst_K = jnp.asarray(np.load(args.sst_npz)["sst_K"])
    else:
        sst_K = None
        warnings.warn(
            "compare_amip_era5: no prescribed-SST source (--sst-npz) provided; "
            "the SST environment tag falls back to surface air temperature "
            "(proxy). CAPE/shear tags and the worst-column ranking are "
            "unaffected. Threading the run's prescribed SST is a documented "
            "follow-up (docs/COMPARE_REANALYSIS.md).",
            stacklevel=2,
        )
    model = load_model_from_restart(args.restart, grid, sigma, args.nlev, sst_K=sst_K,
                                    expected_vertical_coord=args.vertical_coord,
                                    expected_p_top_Pa=args.p_top_Pa)

    era5_cfg = TrainingERA5Config(
        zarr_store=args.era5_zarr,
        local_cache_dir=args.era5_cache,
    )
    era5_slice = load_era5_slice(era5_cfg, args.era5_time_idx)
    regrid = select_era5_regrid(canon)
    reference = column_state_from_carry(regrid(era5_slice, grid, sigma))

    result = compare_and_write(
        model=model, reference=reference, sigma=sigma, grid=grid,
        time_index=args.time_index, n_worst=args.n_worst, out_path=args.out,
    )
    print(
        f"[compare_amip_era5] wrote {len(result.manifest)} worst columns to "
        f"{args.out} (grid={canon}, N={args.resolution}, nlev={args.nlev})."
    )
    # Per-variable GLOBAL bias (physical units) — the interpretable companion to
    # the dimensionless worst-column score: run this on a baseline AND a corrected
    # snapshot to see WHICH variables a correction improves vs trades off. Weighted
    # by the grid's true cell areas (the public GridProtocol `grid_area`).
    from legoesm.training.bias_metrics import aggregate_per_variable_bias

    have_precip = (getattr(model, "precip_mm_day", None) is not None
                   and getattr(reference, "precip_mm_day", None) is not None)
    pvb = aggregate_per_variable_bias(
        result.error_fields, jnp.asarray(grid.grid_area), have_precip=have_precip)
    precip_str = (f"{float(pvb.global_precip_err_mm_day):.4g} mm/day"
                  if have_precip else "N/A (precip not compared)")
    print(
        "[compare_amip_era5] global area-weighted bias: "
        f"T_rmse={float(pvb.global_T_rmse_K):.4g} K, "
        f"qv_rmse={float(pvb.global_qv_rmse_kg_kg):.4g} kg/kg, "
        f"wind_rmse={float(pvb.global_wind_rmse_m_s):.4g} m/s, "
        f"precip_err={precip_str}"
    )

    # Before/after deploy verification: with a baseline restart, report the
    # per-variable global-bias CHANGE (the clause-5 "improve the biases" check —
    # exposes a correction that lowers the combined score by trading variables off).
    if args.baseline_restart:
        from legoesm.training.bias_metrics import (
            bias_improvement,
            per_variable_bias_improvement,
        )

        baseline_model = load_model_from_restart(
            args.baseline_restart, grid, sigma, args.nlev, sst_K=sst_K,
            expected_vertical_coord=args.vertical_coord,
            expected_p_top_Pa=args.p_top_Pa)
        # out_path=None ⇒ compare only (no manifest write) against the SAME reference.
        baseline_cmp = compare_and_write(
            model=baseline_model, reference=reference, sigma=sigma, grid=grid,
            time_index=args.time_index, n_worst=args.n_worst, out_path=None)
        pvi = per_variable_bias_improvement(
            baseline_cmp.error_fields, result.error_fields,
            jnp.asarray(grid.grid_area), have_precip=have_precip)

        def _line(name, unit, base_v, upd_v, improved):
            mark = "improved" if bool(improved) else "WORSE/same"
            return (f"  {name}: {float(base_v):.4g} -> {float(upd_v):.4g} {unit} "
                    f"({mark})")

        print("[compare_amip_era5] per-variable bias baseline -> corrected:")
        print(_line("T_rmse", "K", pvi.baseline.global_T_rmse_K,
                    pvi.updated.global_T_rmse_K, pvi.T_improved))
        print(_line("qv_rmse", "kg/kg", pvi.baseline.global_qv_rmse_kg_kg,
                    pvi.updated.global_qv_rmse_kg_kg, pvi.qv_improved))
        print(_line("wind_rmse", "m/s", pvi.baseline.global_wind_rmse_m_s,
                    pvi.updated.global_wind_rmse_m_s, pvi.wind_improved))
        if have_precip:
            print(_line("precip_err", "mm/day", pvi.baseline.global_precip_err_mm_day,
                        pvi.updated.global_precip_err_mm_day, pvi.precip_improved))
        # The COMBINED area-weighted bias is the campaign's objective + the done-criterion
        # "improve the biases" verdict. Gate the exit code on it (iter 288, mirrors the
        # campaign/OSSE go/no-go): 0 only when the correction LOWERS the held-out combined
        # bias, 1 otherwise — so an automated `deploy && verify` workflow detects a
        # correction that did NOT generalize to the held-out window (overfit the training
        # period) instead of silently reporting success.
        combined = bias_improvement(
            baseline_cmp.error_fields.combined_score, result.error_fields.combined_score,
            jnp.asarray(grid.grid_area))
        print(_line("COMBINED bias", "", combined.baseline_bias,
                    combined.updated_bias, combined.improved))
        return 0 if bool(combined.improved) else 1
    return 0       # no --baseline-restart: informational bias report only, no verdict


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
