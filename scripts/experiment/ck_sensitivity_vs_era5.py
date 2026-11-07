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

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make the sibling ``scripts.*`` entry points (``scripts.data.load_local_era5``) importable
# when this file is run as a standalone CLI; under pytest the repo root is already on the
# path, so the guard makes this a no-op there.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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


def bias_ck_sensitivity(score_low_ck: Any, score_high_ck: Any, *,
                        valid_mask: Any = None) -> dict:
    """How much the per-column combined bias CHANGES between two C_K runs.

    ``score_*_ck`` are the ``compare_state_to_reference`` per-column combined-bias
    fields (any shape; NaN cells — e.g. SST-over-land env tags — are ignored).  Returns
    ``{mean_bias, mean_abs_delta, max_abs_delta, controllable_fraction, c_k_feasible}``:
    ``controllable_fraction = mean|Δbias| / mean_bias`` is the share of the bias that the
    closure tuning actually moves, and ``c_k_feasible`` flags it above
    :data:`_FEASIBLE_FRACTION` — the cheap signal that tuning C_K against real ERA5 could
    lower the bias (vs an idealization-dominated bias that it cannot).

    ``valid_mask`` (optional, same flat column shape as the scores after a ROW-MAJOR
    flatten — e.g. :func:`legoesm.training.compare_reanalysis.ocean_valid_mask`) restricts
    the sensitivity to the SAME columns the realistic campaign ranks/corrects: non-selected
    columns are masked to NaN (ignored by the nan-aware means), so the pre-flight measures
    the OCEAN-only C_K sensitivity the ocean-only campaign actually exploits — without it a
    land-model-driven bias over continents (the wrong lever for the closure) dilutes the
    fraction and can FALSELY gate an ocean-only campaign as NO-GO (the iter-466 silent-
    feature-drop class, here in the pre-flight rather than the campaign).
    """
    lo = np.asarray(score_low_ck, dtype=float).reshape(-1)
    hi = np.asarray(score_high_ck, dtype=float).reshape(-1)
    if lo.shape != hi.shape:
        raise ValueError(
            f"bias_ck_sensitivity: the two score fields must share shape; got "
            f"{lo.shape} vs {hi.shape}.")
    if valid_mask is not None:
        m = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if m.shape != lo.shape:
            raise ValueError(
                f"bias_ck_sensitivity: valid_mask has {m.shape} cells but the score fields "
                f"have {lo.shape} — they must share the (row-major flattened) column grid.")
        if not bool(m.any()):
            raise ValueError(
                "bias_ck_sensitivity: valid_mask selects no columns — an ocean-only "
                "restriction left nothing to score (check --max-land-fraction / the land "
                "mask).")
        lo = np.where(m, lo, np.nan)
        hi = np.where(m, hi, np.nan)
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


def ck_sensitivity_trend(days: Any, bl_fractions: Any) -> dict:
    """Trend of the boundary-layer C_K-controllable fraction with run length (iter 418).

    ``days`` and ``bl_fractions`` are matching 1-D sequences (>= 2 points): the run lengths
    and the BL C_K-controllable fraction at each (sorted internally, so input order is
    free).  Returns the least-squares ``slope_per_day``, whether the fraction is strictly
    ``monotonic_increasing``, and a CRUDE linear extrapolation ``days_to_feasible_floor_
    linear`` of how many days that trend would take to reach :data:`_FEASIBLE_FRACTION`
    (``None`` if already at/above the floor, or not rising).

    A climbing fraction ⇒ EQUILIBRATION-limited (a longer, more-equilibrated run is the
    lever, because C_K's slow BL mixing needs time to accumulate a difference); a flat one
    ⇒ IDEALIZATION-limited (realism — real radiation/SST — is needed, not run length).
    iter-418 (a 1/3/6-day aquaplanet vs real ERA5) measured a climbing-BUT-shallow trend
    (~0.03 %/day, ~130 days to the floor by linear extrapolation) ⇒ run length helps but is
    insufficient alone; realism is ALSO required — the precise HPC characterization.
    """
    d = np.asarray(days, dtype=float).reshape(-1)
    f = np.asarray(bl_fractions, dtype=float).reshape(-1)
    if d.shape != f.shape:
        raise ValueError(
            f"ck_sensitivity_trend: days and bl_fractions must share shape; got "
            f"{d.shape} vs {f.shape}.")
    if d.size < 2:
        raise ValueError(
            "ck_sensitivity_trend: need >= 2 (days, fraction) points to fit a trend.")
    if bool(np.any(np.isnan(d))) or bool(np.any(np.isnan(f))):
        raise ValueError(
            "ck_sensitivity_trend: days/bl_fractions contain NaN — a NaN would silently "
            "poison the fitted slope.")
    order = np.argsort(d)                  # robust to unsorted input
    d, f = d[order], f[order]
    if bool(np.any(np.diff(d) == 0)):
        # Two fraction measurements at the SAME run length are noise, not a time trend —
        # they would falsely read as monotonic (codex-review iter 418).  main() dedups via
        # sorted(set(...)), but the public function must reject it directly.
        raise ValueError(
            f"ck_sensitivity_trend: duplicate day values are not allowed; got sorted days "
            f"{d.tolist()} — each run length must appear exactly once.")
    slope = float(np.polyfit(d, f, 1)[0])
    rising = bool(np.all(np.diff(f) > 0))
    last_d, last_f = float(d[-1]), float(f[-1])
    if rising and slope > 1e-12 and last_f < _FEASIBLE_FRACTION:
        days_to_floor: float | None = last_d + (_FEASIBLE_FRACTION - last_f) / slope
    else:
        days_to_floor = None
    return {
        "slope_per_day": slope,
        "monotonic_increasing": rising,
        "days_to_feasible_floor_linear": days_to_floor,
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


def _build_ck_atm_config(c_k: float, *, resolution: int, nlev: int, days: int,
                         radiation: str, land_mask_path: str, grid_type: str):
    """The clubb_lite ``ExperimentConfig`` the C_K-sensitivity twin runs at ``c_k`` — pure
    (no driver/run), so the GRID-TYPE threading is unit-testable without a model run.

    ``grid_type`` (``latlon`` or ``cubed_sphere``) MUST match the campaign's grid so the
    go/no-go measures the C_K sensitivity on the SAME grid the campaign ranks over (iter 493).
    ``gaussian`` is rejected: clubb_lite's prognostic physics state is not threaded by the
    spectral loop gaussian requires (issue #405 / iter 494) — it would fail at the model
    build, so fail loud here."""
    if grid_type not in ("latlon", "cubed_sphere"):
        raise ValueError(
            f"grid_type {grid_type!r} is unsupported for the clubb_lite C_K-sensitivity twin: "
            "clubb_lite carries prognostic physics state the spectral run loop (gaussian) does "
            "not thread (issue #405). Use 'latlon' or 'cubed_sphere'.")
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig, TurbulenceConfig
    from legoesm.driver.config import (
        DycoreConfig,
        ExperimentConfig,
        GridConfig,
        OutputConfig,
    )

    return ExperimentConfig(
        grid=GridConfig(grid_type=str(grid_type), resolution=resolution, nlev=nlev),
        dycore=DycoreConfig(dt=600.0, model_type="hydrostatic",
                            discretization="finite_volume"),
        output=OutputConfig(diag_days=max(days, 1)), radiation=radiation, days=days,
        turbulence="clubb_lite", land_mask_path=land_mask_path,
        turbulence_override=TurbulenceConfig(
            scheme="clubb_lite", clubb_lite=CLUBBLiteConfig(C_K=float(c_k))))


def _run_model_state(c_k: float, *, resolution: int, nlev: int, days: int,
                     radiation: str = "gray", land_mask_path: str = "",
                     grid_type: str = "latlon"):
    """A tiny coupled (CMIP) clubb_lite run at the given C_K → (ColumnState, grid, sigma,
    land_fraction).

    ``radiation`` defaults to ``"gray"`` (the idealized config the iter-412 NO-GO was
    measured at); pass ``"rrtmgp"`` to measure the C_K sensitivity at the REALISTIC config
    the HPC run uses — the pre-flight then answers whether the loop can move the bias THERE,
    not just under idealized radiation.

    ``land_mask_path`` (default ``""`` = flat/aquaplanet) gives the model the SAME land/sea
    distribution as the realistic campaign so the returned static ``land_fraction`` can
    build the ocean-only mask the campaign ranks over; an empty path keeps the flat model
    (all-ocean land fraction → an ocean-only restriction is a safe no-op).  ``grid_type``
    matches the campaign's grid family (iter 493)."""
    import jax
    jax.config.update("jax_enable_x64", True)
    from legoesm.driver.coupled_config import PRESETS
    from legoesm.driver.coupled_esm_driver import CoupledESMDriver
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.training.compare_reanalysis import column_state_from_hydrostatic

    atm = _build_ck_atm_config(
        c_k, resolution=resolution, nlev=nlev, days=days, radiation=radiation,
        land_mask_path=land_mask_path, grid_type=grid_type)
    # latlon: the explicit (resolution, 2*resolution) ocean grid the iter-445 real-run
    # validation used. Non-latlon: ocean_grid=None ⇒ the coupled driver uses its OWN atm grid
    # (same-grid identity coupling), so the aquaplanet SST source works on ANY grid family.
    ocean_grid = (create_latlon_grid(n_lat=resolution, n_lon=2 * resolution)
                  if grid_type == "latlon" else None)
    driver = CoupledESMDriver(atm, PRESETS["aquaplanet"](), ocean_grid=ocean_grid)
    driver.setup()
    driver.run()
    a = driver._atm
    model = column_state_from_hydrostatic(
        a.state, a.q_v, sst_K=driver._ocean_state.T_sfc.data)
    return model, a.grid, a.sigma, np.asarray(a.static_land_fraction())


def _preflight_exit_code(go: bool) -> int:
    """GO/NO-GO pre-flight exit status (mirrors the campaign's ``_campaign_exit_code``):
    ``0`` when the C_K loop CAN lower the real-ERA5 bias (so an HPC launch should PROCEED),
    ``1`` when it cannot (idealization-dominated — realism needed, not the loop).  Lets an
    operator gate ``ck_sensitivity_vs_era5.py … && sbatch run_correction_campaign.sbatch``
    so a multi-day HPC run is not spent on a bias the C_K closure cannot move."""
    return 0 if go else 1


def _build_arg_parser():
    """The CLI parser as a factory so the cluster sbatch ↔ CLI flag contract is testable
    (mirrors ``run_correction_campaign._build_arg_parser``)."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--local-era5-dir", required=True)
    p.add_argument("--local-era5-date", required=True, help="YYYYMMDD")
    p.add_argument("--c-k", type=float, nargs=2, default=(0.4, 1.0),
                   help="the two C_K values to contrast")
    p.add_argument("--resolution", type=int, default=8)
    p.add_argument("--nlev", type=int, default=5)
    p.add_argument("--grid-type", default="latlon",
                   choices=("latlon", "cubed_sphere"),
                   help="model grid family — MATCH the campaign's --grid-type so the go/no-go "
                        "measures the C_K sensitivity on the SAME grid (default latlon; gaussian "
                        "is unsupported for clubb_lite, issue #405)")
    p.add_argument("--days", type=int, default=1)
    p.add_argument("--radiation", default="gray",
                   help="radiation scheme for the sensitivity runs (default gray; pass "
                        "rrtmgp to measure the C_K sensitivity at the REALISTIC HPC config)")
    p.add_argument("--days-sweep", type=int, nargs="+", default=None,
                   help="run lengths [days] to sweep — reports the BL C_K-controllable "
                        "fraction vs run length (equilibration-limited vs idealization-"
                        "limited) instead of the single-run report")
    p.add_argument("--era5-n-times", type=int, default=4)
    p.add_argument("--land-mask-path", default="",
                   help="land/sea mask file giving the sensitivity runs the SAME land "
                        "distribution as the realistic campaign (default flat/aquaplanet); "
                        "required for --ocean-only to select a real ocean subset")
    p.add_argument("--ocean-only", action="store_true",
                   help="restrict the C_K sensitivity to OCEAN columns (land_fraction <= "
                        "--max-land-fraction), matching an ocean-only campaign so a land-"
                        "model-driven bias does not falsely gate it NO-GO")
    p.add_argument("--max-land-fraction", type=float, default=0.5,
                   help="ocean threshold for --ocean-only: a column is ocean where "
                        "land_fraction <= this (default 0.5 = majority ocean)")
    return p


def main(argv: list[str] | None = None) -> int:
    """Run the C_K-sensitivity feasibility check against a local ERA5 archive."""
    import jax.numpy as jnp
    from legoesm.training.compare_reanalysis import (
        column_state_from_carry,
        compare_state_to_reference,
        ocean_valid_mask,
    )
    from legoesm.training.era5_to_state import (
        WB2_PRESSURE_LEVELS,
        TrainingERA5Config,
        era5_to_latlon_carry,
        load_era5_time_mean,
    era5_terrain_product,
)

    from scripts.data.load_local_era5 import open_local_era5_dataset

    args = _build_arg_parser().parse_args(argv)

    # Fail-fast on a non-existent land mask BEFORE the (slow) model runs — without this a
    # typo'd --land-mask-path surfaces only deep in the model build (xarray open_dataset),
    # after the setup time is spent; worse, with --ocean-only it could read as a config the
    # operator never has (mirrors the iter-454 config-writer fail-fast).
    if args.land_mask_path and not Path(args.land_mask_path).is_file():
        raise SystemExit(
            f"[ck-sensitivity] --land-mask-path does not exist: {args.land_mask_path!r} — "
            "pass the campaign's land/sea mask (or drop it for a flat/aquaplanet preflight).")

    if args.radiation != "gray":
        # Fail-fast heads-up BEFORE the (slow) runs: rrtmgp's radiation graph is
        # compute-heavy (a res-4/nlev-20/2-day realistic run did NOT finish in ~16 min on a
        # login node — the gray default with the SAME ingest + coupled driver completes
        # interactively). Run the realistic pre-flight on a COMPUTE node (a short srun/sbatch).
        print(f"[ck-sensitivity] NOTE: --radiation {args.radiation} is COMPUTE-HEAVY "
              "(the radiation graph) — run this on a compute node, not the login node.",
              flush=True)

    ck_lo, ck_hi = args.c_k

    def _ocean_mask(land_fraction):
        """The ocean-only column mask (or ``None`` when ``--ocean-only`` is off) from a
        run's static land fraction, matching the campaign's ``ocean_valid_mask`` so the
        pre-flight scores the SAME columns; warns when ``--ocean-only`` is set on a flat
        model (no ``--land-mask-path``) so the would-be-silent no-op is visible."""
        if not args.ocean_only:
            return None
        m = np.asarray(ocean_valid_mask(
            land_fraction, max_land_fraction=args.max_land_fraction))
        if bool(m.all()):
            print("[ck-sensitivity] NOTE: --ocean-only selected ALL columns — the model has "
                  "no land (pass --land-mask-path to give it the campaign's land mask), so "
                  "the ocean-only restriction is a no-op here.", flush=True)
        return m

    def _build_reference(grid, sigma):
        ds = open_local_era5_dataset(args.local_era5_dir, args.local_era5_date)
        era5 = load_era5_time_mean(
            TrainingERA5Config(levels=WB2_PRESSURE_LEVELS,
                               surface_variables=("surface_pressure", "skin_temperature")),
            range(args.era5_n_times), ds=ds)
        return column_state_from_carry(era5_to_latlon_carry(
            era5, grid, sigma, target_phis=era5_terrain_product(era5, grid)))

    def _bl_per_level(m_lo_s, m_hi_s, grid, ref, ocean_mask=None):
        """(sigma, T-bias, C_K-controllable fraction) at the boundary-layer (max-sigma)
        level — shared per-level computation with the single-run report below.

        ``ocean_mask`` (optional, flat per-column bool) zero-weights LAND columns so the
        per-level sensitivity matches an ocean-only campaign (the combined-score restriction
        the GO/NO-GO gate uses), keeping the two reports consistent."""
        mt = np.asarray(m_lo_s.T)
        ncol = int(np.prod(mt.shape[:-1]))
        w = np.cos(np.asarray(grid.grid_lat)).reshape(-1)
        if w.size != ncol:                           # fail loud, never silently broadcast
            raise ValueError(
                f"ck-sensitivity per-level: cos(lat) area weights have {w.size} cells but "
                f"model.T has {ncol} columns — grid/state column-grid mismatch.")
        if ocean_mask is not None:
            om = np.asarray(ocean_mask, dtype=bool).reshape(-1)
            if om.size != ncol:
                raise ValueError(
                    f"ck-sensitivity per-level: ocean mask has {om.size} cells but model.T "
                    f"has {ncol} columns — grid/mask column-grid mismatch.")
            w = w * om                               # land columns contribute zero weight
        return per_level_ck_bias_sensitivity(mt, np.asarray(m_hi_s.T), np.asarray(ref.T), w)

    if args.days_sweep:
        days_list = sorted(set(args.days_sweep))
        ref_box: dict = {}
        fracs: list[float] = []
        print(f"[ck-sensitivity sweep] C_K {ck_lo} vs {ck_hi} vs real ERA5 "
              f"({args.local_era5_date}, res={args.resolution}); BL C_K-controllable vs "
              "run length:")
        for days in days_list:
            m_lo, grid, sigma, f_land = _run_model_state(
                ck_lo, resolution=args.resolution, nlev=args.nlev, days=days,
                radiation=args.radiation, land_mask_path=args.land_mask_path,
                grid_type=args.grid_type)
            m_hi, _, _, _ = _run_model_state(
                ck_hi, resolution=args.resolution, nlev=args.nlev, days=days,
                radiation=args.radiation, land_mask_path=args.land_mask_path,
                grid_type=args.grid_type)
            if "ref" not in ref_box:                 # grid/sigma depend only on res/nlev
                ref_box["ref"] = _build_reference(grid, sigma)
            pl = _bl_per_level(m_lo, m_hi, grid, ref_box["ref"],
                               ocean_mask=_ocean_mask(f_land))
            sig = np.asarray(sigma.sigma_full, dtype=float).reshape(-1)
            bl = int(np.argmax(sig))                 # surface-last: the BL is max sigma
            sbl = float(sig[bl])
            bbl = float(np.asarray(pl["bias_per_level"])[bl])
            fbl = float(np.asarray(pl["controllable_fraction_per_level"])[bl])
            fracs.append(fbl)
            print(f"  days={days:>3}: BL sigma {sbl:.3f}  T-bias {bbl:.3g} K  "
                  f"C_K-controllable {fbl:.3%}")
        trend = ck_sensitivity_trend(days_list, fracs)
        print(f"  trend: slope {trend['slope_per_day']:.3%}/day, "
              f"monotonic_increasing={trend['monotonic_increasing']}")
        dtf = trend["days_to_feasible_floor_linear"]
        if not trend["monotonic_increasing"]:
            print("  => FLAT: run length is NOT the lever — the bias is idealization-"
                  "dominated (needs real radiation/SST), not equilibration-limited.")
        elif dtf is None:
            print("  => already at/above the feasibility floor.")
        else:
            print(f"  => EQUILIBRATION is a lever (the fraction climbs with run length); a "
                  f"LINEAR extrapolation needs ~{dtf:.0f} days to reach the "
                  f"{_FEASIBLE_FRACTION:.0%} floor — if impractically long, realism (real "
                  "radiation/SST) is ALSO needed, not run length alone.")
        # GO/NO-GO gate: run length is a lever iff the trend rises toward feasibility; a FLAT
        # trend is idealization-dominated (the loop cannot reach the floor with run length).
        go = bool(trend["monotonic_increasing"])
        print(f"  preflight verdict: {'GO' if go else 'NO-GO'} (exit {_preflight_exit_code(go)}) "
              "— gate an HPC launch on this exit code.")
        return _preflight_exit_code(go)

    m_lo, grid, sigma, f_land = _run_model_state(
        ck_lo, resolution=args.resolution, nlev=args.nlev, days=args.days,
        radiation=args.radiation, land_mask_path=args.land_mask_path,
        grid_type=args.grid_type)
    m_hi, _, _, _ = _run_model_state(
        ck_hi, resolution=args.resolution, nlev=args.nlev, days=args.days,
        radiation=args.radiation, land_mask_path=args.land_mask_path,
        grid_type=args.grid_type)
    ref = _build_reference(grid, sigma)
    ocean_mask = _ocean_mask(f_land)

    def _score(model):
        comp = compare_state_to_reference(
            model=model, reference=ref,
            sigma_full=jnp.asarray(sigma.sigma_full),
            sigma_half=jnp.asarray(sigma.sigma_half),
            lat_deg=jnp.asarray(np.rad2deg(np.asarray(grid.grid_lat))),
            lon_deg=jnp.asarray(np.rad2deg(np.asarray(grid.grid_lon))),
            time_index=0, n_worst=1)
        return np.asarray(comp.error_fields.combined_score)

    s = bias_ck_sensitivity(_score(m_lo), _score(m_hi), valid_mask=ocean_mask)
    _scope = "ocean-only" if ocean_mask is not None else "all columns"
    print(f"[ck-sensitivity] C_K {ck_lo} vs {ck_hi} against real ERA5 "
          f"({args.local_era5_date}, res={args.resolution}, days={args.days}, {_scope}):")
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
    # _bl_per_level area-weights per column with cos(lat) (grid.grid_lat shares the model's
    # column grid, the same object _score passes to compare_state_to_reference), showing
    # that even when the full-column bias is C_K-insensitive the closure still controls the
    # boundary layer it acts on.
    pl = _bl_per_level(m_lo, m_hi, grid, ref, ocean_mask=ocean_mask)
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
    # GO/NO-GO gate: the C_K loop can lower the TOTAL bias iff it is C_K-feasible at this
    # config; if not, try --days-sweep (run length) or add realism before committing HPC hours.
    go = bool(s["c_k_feasible"])
    print(f"  preflight verdict: {'GO' if go else 'NO-GO'} (exit {_preflight_exit_code(go)}) "
          "— gate an HPC launch on this exit code (NO-GO: try --days-sweep or add realism).")
    return _preflight_exit_code(go)


if __name__ == "__main__":
    raise SystemExit(main())
