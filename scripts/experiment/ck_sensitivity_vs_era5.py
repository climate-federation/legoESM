"""Clause-6 FEASIBILITY diagnostic: is the model-vs-ERA5 bias SENSITIVE to C_K?

The done-criterion's final clause — "updating the LES-informed parameters IMPROVES the
biases" — can only be demonstrated against real ERA5 if the model's bias is partly
ATTRIBUTABLE to the closure being tuned.  For a too-idealized model (e.g. a 1-day
aquaplanet with gray radiation) the bias is dominated by the model's IDEALIZATION
(radiation, prescribed-SST, spin-up), and tuning the CLUBB-lite ``C_K`` barely moves it —
so the monotonic correction loop would correctly find NO improvement and reject every
round.  This diagnostic runs the SAME model at two ``C_K`` values, compares each to real
ERA5, and reports how much the bias actually CHANGES with ``C_K`` (the fraction of the
total bias that is C_K-controllable) — the cheap go/no-go an operator runs BEFORE
committing HPC to a real-ERA5 correction campaign.

Verified iter 412 (a 1-day clubb_lite aquaplanet vs NCAR-RDA Sept-2017 ERA5): C_K∈{0.4,
1.0} changed the combined bias by only ~0.01% of the 11.2 total (C_K-INSENSITIVE) — i.e.
that idealized setup CANNOT demonstrate clause 6; a realistic multi-day run is required.
``bias_ck_sensitivity`` is the pure (unit-tested) metric; ``main`` runs the model + the
real-ERA5 compare (data-dependent: needs a local ERA5 archive + a model integration).
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Below this C_K-controllable FRACTION of the total bias, a real-ERA5 correction loop
#: would be dominated by model-idealization error and is unlikely to show a reduction —
#: a heuristic go/no-go, NOT a hard physical threshold (a realistic model needs less).
_FEASIBLE_FRACTION = 0.05

#: Sigma above which a (surface-last) level is classified as the boundary layer, for the
#: per-level "C_K controls the BL it acts in" interpretive note — a coarse near-surface
#: cutoff, NOT a physical BL-top diagnosis.
_BL_SIGMA_THRESHOLD = 0.8


def _most_controllable_level(frac: Any) -> int | None:
    """Index of the most C_K-controllable level (max fraction), or ``None`` if undefined.

    Returns ``None`` for an empty or all-NaN fraction vector — a degenerate run where every
    level perfectly matches (or fails to compare against) ERA5 has no "most-controllable"
    level, and ``np.nanargmax`` would otherwise raise on the all-NaN slice.
    """
    f = np.asarray(frac, dtype=float).reshape(-1)
    if f.size == 0 or bool(np.all(np.isnan(f))):
        return None
    return int(np.nanargmax(f))


def bias_ck_sensitivity(score_low_ck: Any, score_high_ck: Any) -> dict:
    """How much the per-column combined bias CHANGES between two C_K runs.

    ``score_*_ck`` are the ``compare_state_to_reference`` per-column combined-bias
    fields (any shape; NaN cells — e.g. SST-over-land env tags — are ignored).  Returns
    ``{mean_bias, mean_abs_delta, max_abs_delta, controllable_fraction, c_k_feasible}``:
    ``controllable_fraction = mean|Δbias| / mean_bias`` is the share of the bias that the
    closure tuning actually moves, and ``c_k_feasible`` flags it above
    :data:`_FEASIBLE_FRACTION` — the cheap signal that tuning C_K against real ERA5 could
    lower the bias (vs an idealization-dominated bias that it cannot).
    """
    lo = np.asarray(score_low_ck, dtype=float).reshape(-1)
    hi = np.asarray(score_high_ck, dtype=float).reshape(-1)
    if lo.shape != hi.shape:
        raise ValueError(
            f"bias_ck_sensitivity: the two score fields must share shape; got "
            f"{lo.shape} vs {hi.shape}.")
    mean_bias = float(np.nanmean(0.5 * (lo + hi)))
    abs_delta = np.abs(hi - lo)
    mean_abs_delta = float(np.nanmean(abs_delta))
    # mean_bias is a positive RMSE-like score; floor it so an (near-)zero-bias degenerate
    # case reports a finite fraction rather than dividing by ~0.
    frac = mean_abs_delta / max(mean_bias, 1e-12)
    return {
        "mean_bias": mean_bias,
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": float(np.nanmax(abs_delta)),
        "controllable_fraction": frac,
        "c_k_feasible": bool(frac >= _FEASIBLE_FRACTION),
    }


def per_level_ck_bias_sensitivity(T_low_ck, T_high_ck, T_ref, area_weights) -> dict:  # noqa: N803
    """Per-LEVEL C_K sensitivity of the T-bias — shows WHERE C_K controls the bias.

    ``T_*`` are model/reference temperature fields broadcastable to ``(ncol, nlev)``;
    ``area_weights`` is ``(ncol,)``.  Returns per-level area-weighted RMS T-bias and the
    C_K-controllable FRACTION at each level, so an operator can see that C_K controls the
    BOUNDARY LAYER (the near-surface levels where it acts) even when the full-COLUMN bias
    is dominated by a free-tropospheric error (e.g. a radiation bias) that C_K cannot fix
    — i.e. the LES→C_K correction is EFFECTIVE where it should be, and the full-column
    insensitivity (``bias_ck_sensitivity``) is the MODEL's free-trop error, not the
    correction approach (codex-review iter 416)."""
    lo = np.asarray(T_low_ck, dtype=float)
    hi = np.asarray(T_high_ck, dtype=float)
    ref = np.asarray(T_ref, dtype=float)
    if not (lo.shape == hi.shape == ref.shape):
        raise ValueError(
            f"per_level_ck_bias_sensitivity: T fields must share shape; got "
            f"{lo.shape}, {hi.shape}, {ref.shape}.")
    nlev = lo.shape[-1]
    w = np.asarray(area_weights, dtype=float).reshape(-1, 1)

    def _level_rms(field):
        d2 = (field.reshape(-1, nlev) - ref.reshape(-1, nlev)) ** 2
        return np.sqrt(np.nansum(w * d2, axis=0) / np.nansum(w))   # (nlev,)

    b_lo, b_hi = _level_rms(lo), _level_rms(hi)
    mean_b = 0.5 * (b_lo + b_hi)
    return {
        "bias_per_level": mean_b,
        "controllable_fraction_per_level": np.abs(b_hi - b_lo) / np.maximum(mean_b, 1e-12),
    }


def format_per_level_report(per_level: dict, sigma_full: Any) -> list[str]:
    """Operator-readable per-level C_K-sensitivity lines, flagging the most-controllable
    level.

    ``per_level`` is :func:`per_level_ck_bias_sensitivity`'s output; ``sigma_full`` is the
    ``[nlev]`` surface-last sigma.  All three are ``[nlev]`` and must share that length.
    Surfacing the iter-416 insight: C_K control concentrates near the surface (sigma → 1,
    the boundary layer where the closure acts), so a free-tropospherically-dominated
    full-column insensitivity is the MODEL's free-trop (radiation) error — NOT a flaw in
    the LES→C_K correction, which is tuning the right place.
    """
    bias = np.asarray(per_level["bias_per_level"], dtype=float).reshape(-1)
    frac = np.asarray(
        per_level["controllable_fraction_per_level"], dtype=float).reshape(-1)
    sig = np.asarray(sigma_full, dtype=float).reshape(-1)
    if not (bias.shape == frac.shape == sig.shape):
        raise ValueError(
            f"format_per_level_report: bias/frac/sigma must share length; got "
            f"{bias.shape}, {frac.shape}, {sig.shape}.")
    k_most = _most_controllable_level(frac)   # None on a degenerate (all-NaN) fraction
    lines = []
    for k in range(bias.shape[0]):
        tag = "   <- most C_K-controllable" if k == k_most else ""
        lines.append(
            f"  sigma={sig[k]:.3f}: T-bias {bias[k]:.3g} K, "
            f"C_K-controllable {frac[k]:.2%}{tag}")
    return lines


def _run_model_state(c_k: float, *, resolution: int, nlev: int, days: int):
    """A tiny coupled (CMIP) clubb_lite run at the given C_K → (ColumnState, grid, sigma)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig, TurbulenceConfig
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.training.compare_reanalysis import column_state_from_hydrostatic

    atm = ExperimentConfig(
        grid=GridConfig(grid_type="latlon", resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=max(days, 1)), radiation="gray", days=days,
        turbulence="clubb_lite",
        turbulence_override=TurbulenceConfig(
            scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=float(c_k))))
    driver = CoupledESMDriver(
        atm, PRESETS["aquaplanet"](),
        ocean_grid=create_latlon_grid(n_lat=resolution, n_lon=2 * resolution))
    driver.setup()
    driver.run()
    a = driver._atm
    model = column_state_from_hydrostatic(
        a.state, a.q_v, sst_K=driver._ocean_state.T_sfc.data)
    return model, a.grid, a.sigma


def main(argv: list[str] | None = None) -> int:
    """Run the C_K-sensitivity feasibility check against a local ERA5 archive."""
    import argparse

    import jax.numpy as jnp
    from legoesm.training.compare_reanalysis import (
        column_state_from_carry,
        compare_state_to_reference,
    )
    from legoesm.training.era5_to_state import (
        WB2_PRESSURE_LEVELS,
        TrainingERA5Config,
        era5_to_latlon_carry,
        load_era5_time_mean,
    )

    from scripts.data.load_local_era5 import open_local_era5_dataset

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-era5-dir", required=True)
    p.add_argument("--local-era5-date", required=True, help="YYYYMMDD")
    p.add_argument("--c-k", type=float, nargs=2, default=(0.4, 1.0),
                   help="the two C_K values to contrast")
    p.add_argument("--resolution", type=int, default=8)
    p.add_argument("--nlev", type=int, default=5)
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--era5-n-times", type=int, default=4)
    args = p.parse_args(argv)

    ck_lo, ck_hi = args.c_k
    m_lo, grid, sigma = _run_model_state(
        ck_lo, resolution=args.resolution, nlev=args.nlev, days=args.days)
    m_hi, _, _ = _run_model_state(
        ck_hi, resolution=args.resolution, nlev=args.nlev, days=args.days)
    ds = open_local_era5_dataset(args.local_era5_dir, args.local_era5_date)
    era5 = load_era5_time_mean(
        TrainingERA5Config(levels=WB2_PRESSURE_LEVELS,
                           surface_variables=("surface_pressure", "skin_temperature")),
        range(args.era5_n_times), ds=ds)
    ref = column_state_from_carry(era5_to_latlon_carry(era5, grid, sigma))

    def _score(model):
        comp = compare_state_to_reference(
            model=model, reference=ref,
            sigma_full=jnp.asarray(sigma.sigma_full),
            sigma_half=jnp.asarray(sigma.sigma_half),
            lat_deg=jnp.asarray(np.rad2deg(np.asarray(grid.grid_lat))),
            lon_deg=jnp.asarray(np.rad2deg(np.asarray(grid.grid_lon))),
            time_index=0, n_worst=1)
        return np.asarray(comp.error_fields.combined_score)

    s = bias_ck_sensitivity(_score(m_lo), _score(m_hi))
    print(f"[ck-sensitivity] C_K {ck_lo} vs {ck_hi} against real ERA5 "
          f"({args.local_era5_date}, res={args.resolution}, days={args.days}):")
    print(f"  mean bias {s['mean_bias']:.4g}  |  C_K moves it by "
          f"{s['mean_abs_delta']:.4g} (max {s['max_abs_delta']:.4g})  |  "
          f"controllable fraction {s['controllable_fraction']:.3%}")
    print(f"  c_k_feasible={s['c_k_feasible']} "
          f"(>= {_FEASIBLE_FRACTION:.0%} of the bias is C_K-controllable) — "
          + ("a real-ERA5 correction loop COULD lower the bias."
             if s["c_k_feasible"] else
             "bias is idealization-dominated; a correction loop would find NO "
             "improvement here — use a more realistic (multi-day, real-radiation) run."))

    # Per-level breakdown (iter 417): WHERE in the column is the bias C_K-controllable?
    # Area-weight per column with cos(lat) — grid.grid_lat shares the model's column grid
    # (same object _score passes to compare_state_to_reference), so reshape(-1) aligns with
    # model.T.reshape(-1, nlev).  This shows that even when the full-column bias is
    # C_K-insensitive, the closure still controls the boundary layer it acts on.
    mt = np.asarray(m_lo.T)
    ncol = int(np.prod(mt.shape[:-1]))               # leading dims = columns ([..., nlev])
    w = np.cos(np.asarray(grid.grid_lat)).reshape(-1)
    if w.size != ncol:                               # fail loud, never silently broadcast
        raise ValueError(
            f"ck-sensitivity per-level: cos(lat) area weights have {w.size} cells but "
            f"model.T has {ncol} columns — grid/state column-grid mismatch (the area "
            "weighting would be wrong).")
    pl = per_level_ck_bias_sensitivity(mt, np.asarray(m_hi.T), np.asarray(ref.T), w)
    print("  per-level C_K sensitivity (T-bias vs ERA5, surface-last sigma):")
    for line in format_per_level_report(pl, sigma.sigma_full):
        print(line)
    k_most = _most_controllable_level(pl["controllable_fraction_per_level"])
    sig_full = np.asarray(sigma.sigma_full, dtype=float).reshape(-1)
    if (k_most is not None and not s["c_k_feasible"]
            and sig_full[k_most] >= _BL_SIGMA_THRESHOLD):
        print(f"  => the most C_K-sensitive level is in the boundary layer "
              f"(sigma {float(sig_full[k_most]):.2f}); the full-column insensitivity is the "
              "model's free-trop (radiation) error, NOT the LES->C_K correction — it tunes "
              "the right place, but needs a more realistic free-troposphere for tuning to "
              "move the TOTAL bias.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
