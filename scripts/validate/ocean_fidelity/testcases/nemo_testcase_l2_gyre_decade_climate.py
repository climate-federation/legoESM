#!/usr/bin/env python3
"""The GYRE DECADE: legoESM against NEMO 5.0.2 as CLIMATE, not as a trajectory.

Preregistration: ``docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_decade_climate.md``.
Read it first; every constant below is fixed there and none of them is a
judgement call made at scoring time.

This is the campaign's SECOND tier.  The first tier -- bit-exactness, then the
from-rest year -- stops being informative once the two trajectories separate at
the rounding floor and the flow amplifies the separation.  The FESOM2-JAX paper
(arXiv:2608.01546, section 4) answered the same wall by comparing STATISTICS:
climatological mean state, drift, seasonal cycle, with the JAX-minus-Fortran
RMS required to sit two orders of magnitude below the field's own scale.  That
is what this scores, over ten 360-day years from rest, at a monthly cadence.

NOTHING HERE PRINTS A VERDICT.  It prints numbers, each labelled with the file
it came from, and the receipt interprets them against the preregistered bar.

PHASES
  --score       the JSON: difference time series, climatology, drift,
                energetics, seasonal cycle
  --figures     four PNGs in the style of the year figures
  --self-check  the arithmetic, with synthetic violations that must fail

READERS ARE REUSED, NOT REWRITTEN.  ``_load_lego``, ``_load_nemo``,
``_geometry`` and ``_rms`` come from
``nemo_testcase_l2_gyre_year_fromrest.py``: the npz layout, the restart axis
contract, the kt check, the wet mask and the RMS formula are that module's and
are not re-derived here.  No physical constant is needed -- the geometry comes
from the recipe card and the two thresholds below are diagnostic conventions,
not physics -- so nothing is imported from ``legoesm.constants``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_YEAR_PATH = _HERE / "nemo_testcase_l2_gyre_year_fromrest.py"


def _year_module():
    spec = importlib.util.spec_from_file_location(
        "nemo_testcase_l2_gyre_year_fromrest", _YEAR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


YEAR = _year_module()
require = YEAR.require
GateError = YEAR.GateError
sha256 = YEAR.sha256
_load_lego = YEAR._load_lego
_load_nemo = YEAR._load_nemo
_geometry = YEAR._geometry
_rms = YEAR._rms
CASE = YEAR.CASE
DT_S = YEAR.DT_S
STEPS_PER_DAY = YEAR.STEPS_PER_DAY
YEAR_DAYS = YEAR.YEAR_DAYS
DEFAULT_NEMO_MESH = YEAR.DEFAULT_NEMO_MESH

# --------------------------------------------------------------- constants ---
MONTH_DAYS = 30                                   # nn_stock = 180 steps
MONTH_STEPS = MONTH_DAYS * STEPS_PER_DAY          # 180
MONTHS_PER_YEAR = YEAR_DAYS // MONTH_DAYS         # 12
DECADE_YEARS = 10
DECADE_MONTHS = DECADE_YEARS * MONTHS_PER_YEAR    # 120
DECADE_STEPS = DECADE_MONTHS * MONTH_STEPS        # 21600
# Years 2-10.  Year 1 is the from-rest spin-up transient and is excluded from
# every climatology; the preregistration says why.
CLIM_FIRST_MONTH = MONTHS_PER_YEAR + 1            # 13
# Temperature-threshold mixed layer.  A convention, computed identically on
# both models, chosen because both sides carry T and neither carries density.
MLD_THRESHOLD_K = 0.2

# The preregistered bar.  Reported next to every number; never applied as a
# verdict by this file.
BAR_RATIO = 1.0e-2                 # FESOM2-JAX: two orders of magnitude
BAR_REL_TREND_PER_MONTH = 0.01     # bounded difference
BAR_SPIKE = 10.0                   # max / median over the climatology window
BAR_DRIFT_K = 1.0e-2               # volume-mean temperature agreement
# Year 1 separately: a volume mean of a difference cannot exceed that
# difference's RMS, which round 183 measured at 2.670992e-03 K on day 360.
BAR_DRIFT_YEAR1_K = 3.0e-3

DEFAULT_LEGO_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/decade/lego")
DEFAULT_NEMO_DIR = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/decade/nemo")
DEFAULT_OUT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/decade")

# The five arrays every snapshot carries.  _load_nemo also returns its own
# path and sha256, which are provenance, not fields, and must not be swept
# into an array check.
FIELD_NAMES = ("T", "S", "u", "v", "ssh")

# Synthetic violations.  Each must fail, and each must fail BECAUSE of the
# line it plants.  "month-shift" is not here: it pairs legoESM month m with
# NEMO month m+1 inside score() itself, so it needs the real snapshots and is
# exercised by the data-backed test rather than by this arithmetic self-check.
PLANTS = ("ratio-denominator-zero", "mld-unsorted", "trend-short")
SCORE_PLANTS = ("month-shift",)


# ------------------------------------------------------------------ helpers --
def _weighted_mean(values, weights, mask) -> float:
    """Area- or volume-weighted mean over the wet cells.

    A GLOBAL STATISTIC ON A NON-UNIFORM GRID NEEDS AREA WEIGHTS: the GYRE box
    is a Mercator grid, so its cells are not the same size and ``np.mean``
    would silently weight the northern rows like the southern ones.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    require(values.shape == weights.shape == mask.shape,
            f"weighted mean shapes disagree: {values.shape} {weights.shape} "
            f"{mask.shape}")
    total = float(np.sum(weights[mask]))
    require(total > 0.0, "weighted mean has zero total weight")
    return float(np.sum(values[mask] * weights[mask]) / total)


def _weighted_rms(values, weights, mask) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    require(values.shape == weights.shape == mask.shape,
            f"weighted rms shapes disagree: {values.shape} {weights.shape} "
            f"{mask.shape}")
    total = float(np.sum(weights[mask]))
    require(total > 0.0, "weighted rms has zero total weight")
    return float(np.sqrt(np.sum(weights[mask] * values[mask] ** 2) / total))


def _spatial_scale(field, weights, mask) -> float:
    """The field's own spatial variability: RMS about its weighted mean.

    This is the denominator of the FESOM2-JAX ratio.  GYRE has no observations,
    so the model-minus-observation bias that paper divided by does not exist;
    the substitute, fixed in the preregistration, is how much structure the
    field actually has.
    """
    field = np.asarray(field, dtype=np.float64)
    return _weighted_rms(field - _weighted_mean(field, weights, mask),
                         weights, mask)


def _ratio(difference, reference, weights, mask, *, plant=None) -> dict:
    """The bar quantity: RMS(difference) / RMS(reference about its own mean)."""
    scale = _spatial_scale(reference, weights, mask)
    if plant == "ratio-denominator-zero":
        scale = 0.0
    require(scale > 0.0,
            "the reference field has zero spatial variability, so the "
            "two-orders-of-magnitude ratio is undefined; a constant field "
            "cannot be the denominator of this bar")
    gap = _weighted_rms(np.asarray(difference, dtype=np.float64),
                        weights, mask)
    return {"rms_difference": gap, "spatial_scale": scale,
            "ratio": gap / scale, "bar": BAR_RATIO}


def _mixed_layer_depth(temperature, depth3, wet3, *, plant=None):
    """Depth at which T first falls MLD_THRESHOLD_K below the surface cell.

    Linear interpolation between the bracketing cell centres; a column that
    never crosses the threshold is given its deepest wet cell's depth, which is
    the physically right answer for a fully mixed column and is reported as
    such rather than as a fill value.
    """
    temperature = np.asarray(temperature, dtype=np.float64)
    depth3 = np.asarray(depth3, dtype=np.float64)
    if plant == "mld-unsorted":
        depth3 = depth3[..., ::-1]
    # The depth axis is an API: a column that is not monotonically increasing
    # would make the bracketing search below return a plausible wrong number.
    require(bool(np.all(np.diff(depth3, axis=-1) > 0.0)),
            "the depth axis is not strictly increasing; the mixed-layer "
            "bracketing search would silently return the wrong cell")
    ny, nx, nz = temperature.shape
    out = np.full((ny, nx), np.nan)
    for j in range(ny):
        for i in range(nx):
            column = np.flatnonzero(wet3[j, i])
            if column.size == 0:
                continue
            k_wet = column[-1]
            surface = temperature[j, i, column[0]]
            target = surface - MLD_THRESHOLD_K
            out[j, i] = depth3[j, i, k_wet]
            for k in column[1:]:
                if temperature[j, i, k] <= target:
                    above = temperature[j, i, k - 1]
                    below = temperature[j, i, k]
                    span = above - below
                    frac = 0.0 if span == 0.0 else (above - target) / span
                    out[j, i] = (depth3[j, i, k - 1]
                                 + frac * (depth3[j, i, k] - depth3[j, i, k - 1]))
                    break
    return out


def _zonal_mean(field3, wet3):
    """Mean over the box's x extent, wet cells only -> (y, z)."""
    field3 = np.asarray(field3, dtype=np.float64)
    counts = np.sum(wet3, axis=1)
    totals = np.sum(np.where(wet3, field3, 0.0), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


def _relative_trend(series) -> dict:
    """Least-squares slope per month, divided by the series' own mean.

    The bounded-difference row.  A straight line through 108 monthly RMS values
    is a coarse instrument, so the spike ratio is reported beside it: a series
    that is flat on average but excursions by a factor of 30 is not bounded in
    any useful sense, and the slope alone would not say so.
    """
    series = np.asarray(series, dtype=np.float64)
    require(series.size >= 3,
            f"a trend over {series.size} points is not a trend; the "
            "climatology window must hold at least three months")
    index = np.arange(series.size, dtype=np.float64)
    slope, _ = np.polyfit(index, series, 1)
    mean = float(np.mean(series))
    median = float(np.median(series))
    require(mean > 0.0, "the difference series is identically zero")
    return {"slope_per_month": float(slope),
            "mean": mean,
            "relative_trend_per_month": float(slope) / mean,
            "spike_ratio": float(np.max(series)) / median,
            "bar_relative_trend_per_month": BAR_REL_TREND_PER_MONTH,
            "bar_spike_ratio": BAR_SPIKE}


# ------------------------------------------------------------------ scoring --
def score(lego_root: Path, nemo_dir: Path, *, months: int = DECADE_MONTHS,
          clim_first_month: int = CLIM_FIRST_MONTH, seed: int = 0,
          mesh_path: Path = DEFAULT_NEMO_MESH, plant: str | None = None,
          allow_dirty: bool = False) -> dict:
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card(CASE)
    require(card.dt_s == DT_S, f"card dt {card.dt_s} != {DT_S}")
    require(months >= 2, "a climate comparison needs at least two months")
    require(2 <= clim_first_month <= months,
            f"climatology window starts at month {clim_first_month} but only "
            f"{months} months were scored")
    mesh, wet3, wet2, dz, dy, area, _bands = _geometry(card, mesh_path)
    nlev = card.recipe.z_coord.n_levels
    depth3 = np.asarray(card.recipe.z_coord.nemo_gdept_0,
                        dtype=np.float64)[..., :nlev]
    # Weights.  2-D fields are weighted by cell area; 3-D fields by area x dz.
    area2 = np.asarray(area, dtype=np.float64)
    volume3 = area2[..., None] * np.asarray(dz, dtype=np.float64)[None, None, :]
    started = time.time()

    clim_months = list(range(clim_first_month, months + 1))
    series: dict[str, list] = {key: [] for key in (
        "month", "day", "T3D_rms", "S3D_rms",
        "lego_T3D_monthly_change", "nemo_T3D_monthly_change",
        "lego_S3D_monthly_change", "nemo_S3D_monthly_change",
        "lego_T_volmean", "nemo_T_volmean", "lego_S_volmean", "nemo_S_volmean",
        "lego_KE", "nemo_KE")}
    sources = []
    accum = {side: {name: None for name in ("sst", "sss", "ssh", "Tzm", "Szm")}
             for side in ("lego", "nemo")}
    velocity = {"lego": [], "nemo": []}
    seasonal = {side: {"sst": [[] for _ in range(MONTHS_PER_YEAR)],
                       "mld": [[] for _ in range(MONTHS_PER_YEAR)]}
                for side in ("lego", "nemo")}
    previous = {"lego": None, "nemo": None}

    for month in range(1, months + 1):
        day = month * MONTH_DAYS
        nemo_day = day + (MONTH_DAYS if plant == "month-shift" else 0)
        states = {
            "lego": _load_lego(lego_root, seed, day),
            "nemo": _load_nemo(lego_root, seed, nemo_day, nlev,
                               directory=nemo_dir),
        }
        lego_path = (Path(lego_root) / f"lego_seed{seed}"
                     / f"day{day:03d}.npz")
        sources.append({"month": month, "day": day,
                        "step": day * STEPS_PER_DAY,
                        "lego": str(lego_path),
                        "lego_sha256": sha256(lego_path),
                        "nemo": states["nemo"]["path"],
                        "nemo_sha256": states["nemo"]["sha256"]})
        for side, state in states.items():
            require(state["T"].shape == wet3.shape,
                    f"{side} month {month}: T has shape {state['T'].shape}, "
                    f"the mesh says {wet3.shape}")
            require(all(np.all(np.isfinite(state[name]))
                        for name in FIELD_NAMES),
                    f"{side} month {month}: non-finite field")

        series["month"].append(month)
        series["day"].append(day)
        series["T3D_rms"].append(_rms(states["lego"]["T"] - states["nemo"]["T"],
                                      wet3))
        series["S3D_rms"].append(_rms(states["lego"]["S"] - states["nemo"]["S"],
                                      wet3))
        for side in ("lego", "nemo"):
            state = states[side]
            for name, field in (("T", "T"), ("S", "S")):
                series[f"{side}_{name}_volmean"].append(
                    _weighted_mean(state[field], volume3, wet3))
                change = (None if previous[side] is None
                          else _rms(state[field] - previous[side][field], wet3))
                series[f"{side}_{name}3D_monthly_change"].append(change)
            # Specific kinetic energy [m2/s2], each model's own C-grid u and v
            # at the same indices on both sides.  Identical formula, so the
            # comparison is fair; it is not a certified energy budget.
            series[f"{side}_KE"].append(_weighted_mean(
                0.5 * (state["u"] ** 2 + state["v"] ** 2), volume3, wet3))
            previous[side] = state

        for side, state in states.items():
            calendar = (month - 1) % MONTHS_PER_YEAR
            if month in clim_months:
                mld = _mixed_layer_depth(state["T"], depth3, wet3, plant=plant)
                seasonal[side]["sst"][calendar].append(
                    _weighted_mean(state["T"][..., 0], area2, wet2))
                seasonal[side]["mld"][calendar].append(
                    _weighted_mean(mld, area2, wet2))
                fields = {"sst": state["T"][..., 0], "sss": state["S"][..., 0],
                          "ssh": state["ssh"],
                          "Tzm": _zonal_mean(state["T"], wet3),
                          "Szm": _zonal_mean(state["S"], wet3)}
                for name, field in fields.items():
                    if accum[side][name] is None:
                        accum[side][name] = np.zeros_like(field,
                                                          dtype=np.float64)
                    accum[side][name] += field
                velocity[side].append((np.asarray(state["u"], np.float64),
                                       np.asarray(state["v"], np.float64)))

    count = float(len(clim_months))
    clim = {side: {name: accum[side][name] / count for name in accum[side]}
            for side in accum}

    # The zonal-mean sections carry NaN on rows with no wet cell at that depth;
    # the section mask is those two masks' intersection, so the ratio is taken
    # over the cells both models actually have.
    zm_mask = np.isfinite(clim["lego"]["Tzm"]) & np.isfinite(clim["nemo"]["Tzm"])
    require(bool(zm_mask.any()), "the zonal-mean section is empty")
    # The section is weighted by LAYER THICKNESS.  Unit weights would count a
    # 10 m surface cell and a 300 m abyssal cell equally, which on this card is
    # a factor of thirty, and the ratio would then be a surface statistic
    # wearing a section's name.
    zm_weights = np.broadcast_to(np.asarray(dz, dtype=np.float64),
                                 clim["lego"]["Tzm"].shape)

    climatology = {}
    for name, weights, mask in (("SST", area2, wet2), ("SSS", area2, wet2),
                                ("SSH", area2, wet2),
                                ("T_zonal_mean", zm_weights, zm_mask),
                                ("S_zonal_mean", zm_weights, zm_mask)):
        key = {"SST": "sst", "SSS": "sss", "SSH": "ssh",
               "T_zonal_mean": "Tzm", "S_zonal_mean": "Szm"}[name]
        left, right = clim["lego"][key], clim["nemo"][key]
        row = _ratio(np.where(mask, left - right, 0.0), np.where(mask, right, 0.0),
                     weights, mask)
        row["lego_spatial_scale"] = _spatial_scale(
            np.where(mask, left, 0.0), weights, mask)
        climatology[name] = row

    # The kinetic energy of the departure from the climatological flow.  The
    # round brief called this EKE; it is NOT an eddy kinetic energy on this
    # card -- 106 km cells over a flat bottom resolve no mesoscale -- so it is
    # reported under the name of what it measures, the velocity variance about
    # the record mean.  Same formula on both models.
    eke = {}
    for side in ("lego", "nemo"):
        u_clim = np.mean([pair[0] for pair in velocity[side]], axis=0)
        v_clim = np.mean([pair[1] for pair in velocity[side]], axis=0)
        eke[side] = float(np.mean([
            _weighted_mean(0.5 * ((u - u_clim) ** 2 + (v - v_clim) ** 2),
                           volume3, wet3)
            for u, v in velocity[side]]))

    window = slice(clim_first_month - 1, months)
    drift = {
        "lego_T_volmean_first_last": [series["lego_T_volmean"][0],
                                      series["lego_T_volmean"][-1]],
        "nemo_T_volmean_first_last": [series["nemo_T_volmean"][0],
                                      series["nemo_T_volmean"][-1]],
        "max_abs_T_volmean_difference": float(np.max(np.abs(
            np.asarray(series["lego_T_volmean"])
            - np.asarray(series["nemo_T_volmean"])))),
        "max_abs_S_volmean_difference": float(np.max(np.abs(
            np.asarray(series["lego_S_volmean"])
            - np.asarray(series["nemo_S_volmean"])))),
        "bar_abs_T_volmean_difference": BAR_DRIFT_K,
    }
    year_one = min(MONTHS_PER_YEAR, months)
    drift["max_abs_T_volmean_difference_year1"] = float(np.max(np.abs(
        np.asarray(series["lego_T_volmean"][:year_one])
        - np.asarray(series["nemo_T_volmean"][:year_one]))))
    drift["bar_abs_T_volmean_difference_year1"] = BAR_DRIFT_YEAR1_K

    report = {
        "format": "nemo-testcase-l2-gyre-decade-climate-v1",
        "case": CASE, "seed": seed, "dt_s": DT_S,
        "steps_per_day": STEPS_PER_DAY, "month_days": MONTH_DAYS,
        "month_steps": MONTH_STEPS, "months": months,
        "climatology_window_months": [clim_first_month, months],
        "climatology_month_count": int(count),
        "mld_threshold_K": MLD_THRESHOLD_K,
        "lego_root": str(lego_root), "nemo_dir": str(nemo_dir),
        "mesh": str(mesh_path), "mesh_sha256": sha256(Path(mesh_path)),
        "sources": sources,
        "difference_series": {key: series[key] for key in series},
        "difference_trend_T3D": _relative_trend(
            np.asarray(series["T3D_rms"])[window]
            if plant != "trend-short"
            else np.asarray(series["T3D_rms"])[:2]),
        "difference_trend_S3D": _relative_trend(
            np.asarray(series["S3D_rms"])[window]),
        "climatology": climatology,
        "drift": drift,
        "energetics": {
            "lego_mean_KE_m2_s2": float(np.mean(series["lego_KE"][window])),
            "nemo_mean_KE_m2_s2": float(np.mean(series["nemo_KE"][window])),
            "lego_velocity_variance_about_climatology_m2_s2": eke["lego"],
            "nemo_velocity_variance_about_climatology_m2_s2": eke["nemo"],
        },
        # A calendar slot with no sample is reported as null, not as a NaN a
        # reader would plot as zero.  Every slot has nine samples over the
        # preregistered years 2-10; only a short validation window leaves one
        # empty.
        "seasonal_cycle": {
            side: {name: [float(np.mean(values)) if values else None
                          for values in rows]
                   for name, rows in seasonal[side].items()}
            for side in seasonal
        },
        "plant": plant,
        "wall_seconds": time.time() - started,
        # A real scoring run refuses to stamp a dirty tree, because a report
        # whose commit does not identify its code is not evidence.  Only the
        # unit test, whose report is thrown away, passes allow_dirty.
        "worktree": worktree_stamp(allow_dirty=allow_dirty),
    }
    return report


# ------------------------------------------------------------------ figures --
def figures(report: dict, out: Path, lego_root: Path, nemo_dir: Path, *,
            seed: int = 0, mesh_path: Path = DEFAULT_NEMO_MESH) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card(CASE)
    mesh, wet3, wet2, dz, dy, area, _ = _geometry(card, mesh_path)
    nlev = card.recipe.z_coord.n_levels
    first, last = report["climatology_window_months"]
    months = list(range(first, last + 1))
    area2 = np.asarray(area, dtype=np.float64)

    accum = {side: {name: None for name in ("sst", "sss", "ssh", "Tzm")}
             for side in ("lego", "nemo")}
    for month in months:
        day = month * MONTH_DAYS
        states = {"lego": _load_lego(lego_root, seed, day),
                  "nemo": _load_nemo(lego_root, seed, day, nlev,
                                     directory=nemo_dir)}
        for side, state in states.items():
            fields = {"sst": state["T"][..., 0], "sss": state["S"][..., 0],
                      "ssh": state["ssh"], "Tzm": _zonal_mean(state["T"], wet3)}
            for name, field in fields.items():
                if accum[side][name] is None:
                    accum[side][name] = np.zeros_like(field, dtype=np.float64)
                accum[side][name] += field
    clim = {side: {name: accum[side][name] / len(months) for name in accum[side]}
            for side in accum}

    lat = np.asarray(card.recipe.grid.native_lat_T_deg, dtype=np.float64)
    depth1d = np.asarray(card.recipe.z_coord.nemo_gdept_0,
                         dtype=np.float64)[0, 0, :nlev]
    written = []

    # fig1 -- the climatology: three fields, legoESM / NEMO / difference.
    figure, axes = plt.subplots(3, 3, figsize=(12.0, 9.5))
    rows = (("SST", "sst", "degC"), ("SSS", "sss", "g/kg"), ("SSH", "ssh", "m"))
    for r, (label, key, unit) in enumerate(rows):
        left = np.where(wet2, clim["lego"][key], np.nan)
        right = np.where(wet2, clim["nemo"][key], np.nan)
        lo = float(np.nanmin([left, right]))
        hi = float(np.nanmax([left, right]))
        difference = left - right
        span = float(np.nanmax(np.abs(difference))) or 1.0
        for c, (panel, data, kwargs) in enumerate((
                ("legoESM", left, dict(vmin=lo, vmax=hi, cmap="viridis")),
                ("NEMO", right, dict(vmin=lo, vmax=hi, cmap="viridis")),
                ("legoESM - NEMO", difference,
                 dict(vmin=-span, vmax=span, cmap="RdBu_r")))):
            image = axes[r, c].pcolormesh(data.T, **kwargs)
            axes[r, c].set_title(f"{label} {panel}" + (
                f"\nmax|diff| = {span:.3e} {unit}" if c == 2 else ""),
                fontsize=9)
            figure.colorbar(image, ax=axes[r, c], fraction=0.046)
    figure.suptitle(
        f"GYRE years {first // MONTHS_PER_YEAR + 1}-{last // MONTHS_PER_YEAR} "
        f"climatology ({len(months)} monthly snapshots)")
    figure.tight_layout()
    path = out / "fig1_decade_climatology_maps.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    written.append(path)

    # fig2 -- the difference time series against each model's own variability.
    series = report["difference_series"]
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    for ax, name, unit in ((axes[0], "T3D", "K"), (axes[1], "S3D", "g/kg")):
        ax.semilogy(series["month"], series[f"{name}_rms"], "o-", ms=3,
                    label="RMS(legoESM - NEMO)")
        for side, style in (("lego", "--"), ("nemo", ":")):
            ax.semilogy(series["month"],
                        series[f"{side}_{name}_monthly_change"], style,
                        label=f"{side} own month-to-month change")
        ax.axvline(first - 0.5, color="grey", lw=0.8)
        ax.set_xlabel("month")
        ax.set_ylabel(f"{name} [{unit}]")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    trend = report["difference_trend_T3D"]
    figure.suptitle("difference vs each model's own variability; "
                    f"T relative trend over the window = "
                    f"{trend['relative_trend_per_month']:+.2e} /month")
    figure.tight_layout()
    path = out / "fig2_decade_difference_series.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    written.append(path)

    # fig3 -- drift and energetics.
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    for ax, key, label in ((axes[0], "T_volmean", "volume-mean T [degC]"),
                           (axes[1], "S_volmean", "volume-mean S [g/kg]")):
        ax.plot(series["month"], series[f"lego_{key}"], label="legoESM")
        ax.plot(series["month"], series[f"nemo_{key}"], "--", label="NEMO")
        ax.set_xlabel("month")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[2].semilogy(series["month"], series["lego_KE"], label="legoESM")
    axes[2].semilogy(series["month"], series["nemo_KE"], "--", label="NEMO")
    axes[2].set_xlabel("month")
    axes[2].set_ylabel("volume-mean KE [m2/s2]")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)
    figure.suptitle("drift and energetics")
    figure.tight_layout()
    path = out / "fig3_decade_drift_energetics.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    written.append(path)

    # fig4 -- seasonal cycle and the zonal-mean temperature section.
    figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))
    calendar = np.arange(1, MONTHS_PER_YEAR + 1)
    seasonal = report["seasonal_cycle"]
    for ax, key, label in ((axes[0], "sst", "basin-mean SST [degC]"),
                           (axes[1], "mld", "basin-mean MLD [m]")):
        ax.plot(calendar, seasonal["lego"][key], "o-", ms=3, label="legoESM")
        ax.plot(calendar, seasonal["nemo"][key], "s--", ms=3, label="NEMO")
        ax.set_xlabel("calendar month")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[1].invert_yaxis()
    section = clim["lego"]["Tzm"] - clim["nemo"]["Tzm"]
    span = float(np.nanmax(np.abs(section))) or 1.0
    image = axes[2].pcolormesh(lat[:, 0], depth1d, section.T,
                               vmin=-span, vmax=span, cmap="RdBu_r")
    axes[2].invert_yaxis()
    axes[2].set_xlabel("latitude [deg]")
    axes[2].set_ylabel("depth [m]")
    axes[2].set_title(f"zonal-mean T, legoESM - NEMO\nmax|diff| = {span:.3e} K",
                      fontsize=9)
    figure.colorbar(image, ax=axes[2], fraction=0.046)
    figure.tight_layout()
    path = out / "fig4_decade_seasonal_section.png"
    figure.savefig(path, dpi=110)
    plt.close(figure)
    written.append(path)
    return written


# --------------------------------------------------------------- self-check --
def self_check(plant: str | None = None) -> int:
    """The arithmetic, on synthetic fields whose answers are known by hand.

    Every assertion is paired with a synthetic violation that must fail, so
    none of them can pass vacuously.
    """
    if plant is not None:
        return _plant(plant)

    mask = np.ones((2, 3), dtype=bool)
    weights = np.array([[1.0, 1.0, 1.0], [3.0, 3.0, 3.0]])
    values = np.array([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]])
    # weighted mean = (3*0 + 9*4)/12 = 3
    assert abs(_weighted_mean(values, weights, mask) - 3.0) < 1e-12
    # An unweighted mean would be 2; the weights are load-bearing.
    assert abs(float(np.mean(values)) - 2.0) < 1e-12
    # scale = sqrt((3*9 + 9*1)/12) = sqrt(3)
    assert abs(_spatial_scale(values, weights, mask) - np.sqrt(3.0)) < 1e-12
    row = _ratio(np.full((2, 3), 0.06), values, weights, mask)
    assert abs(row["ratio"] - 0.06 / np.sqrt(3.0)) < 1e-12
    assert row["bar"] == BAR_RATIO

    # Mixed layer: a two-cell column with a 1 K jump crosses 0.2 K one fifth of
    # the way between 5 m and 15 m, i.e. at 7 m.
    depth = np.broadcast_to(np.array([5.0, 15.0]), (1, 1, 2)).copy()
    temperature = np.array([[[20.0, 19.0]]])
    wet = np.ones((1, 1, 2), dtype=bool)
    mld = _mixed_layer_depth(temperature, depth, wet)
    assert abs(float(mld[0, 0]) - 7.0) < 1e-12, mld
    # A fully mixed column reports its deepest wet cell, not a fill value.
    mld = _mixed_layer_depth(np.array([[[20.0, 20.0]]]), depth, wet)
    assert abs(float(mld[0, 0]) - 15.0) < 1e-12, mld

    zonal = _zonal_mean(np.array([[[1.0], [3.0]], [[5.0], [9.0]]]),
                        np.ones((2, 2, 1), dtype=bool))
    assert np.allclose(zonal, np.array([[2.0], [7.0]])), zonal

    trend = _relative_trend(np.array([1.0, 2.0, 3.0, 4.0]))
    assert abs(trend["slope_per_month"] - 1.0) < 1e-12
    assert abs(trend["relative_trend_per_month"] - 1.0 / 2.5) < 1e-12
    assert abs(trend["spike_ratio"] - 4.0 / 2.5) < 1e-12

    for name in PLANTS:
        status = _plant(name)
        assert status != 0, f"plant {name} did not fail"
    print("SELF-CHECK OK")
    return 0


def _plant(name: str) -> int:
    """Each plant must raise; the harness returns non-zero when it does."""
    require(name in PLANTS, f"unknown plant {name}")
    try:
        if name == "ratio-denominator-zero":
            # The reference here HAS variability, so this plant fires only
            # because the planted line zeroes the denominator.  An earlier
            # version passed a constant reference, whose scale is zero
            # anyway -- the plant "failed" with the planted line deleted,
            # which is the definition of a guard that proves nothing.
            mask = np.ones((2, 2), dtype=bool)
            reference = np.array([[-1.0, -1.0], [1.0, 1.0]])
            assert _spatial_scale(reference, np.ones((2, 2)), mask) > 0.0
            _ratio(np.ones((2, 2)), reference, np.ones((2, 2)), mask,
                   plant=name)
        elif name == "mld-unsorted":
            depth = np.broadcast_to(np.array([5.0, 15.0]), (1, 1, 2)).copy()
            _mixed_layer_depth(np.array([[[20.0, 19.0]]]), depth,
                               np.ones((1, 1, 2), dtype=bool), plant=name)
        elif name == "trend-short":
            _relative_trend(np.array([1.0, 2.0]))
    except (YEAR.GateError, SystemExit, AssertionError,
            ValueError) as error:
        print(f"plant {name} failed as required: {error}")
        return 1
    print(f"PLANT {name} DID NOT FAIL")
    return 0


# ------------------------------------------------------------------- main ----
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--plant", default=None,
                        choices=list(PLANTS) + list(SCORE_PLANTS))
    parser.add_argument("--months", type=int, default=DECADE_MONTHS)
    parser.add_argument("--clim-first-month", type=int,
                        default=CLIM_FIRST_MONTH,
                        help="first month of the climatology window; the "
                             "preregistered decade uses 13 (years 2-10)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lego-root", type=Path, default=DEFAULT_LEGO_ROOT)
    parser.add_argument("--nemo-dir", type=Path, default=DEFAULT_NEMO_DIR)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_NEMO_MESH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.self_check or (args.plant is not None
                           and args.plant not in SCORE_PLANTS):
        return self_check(args.plant)
    if not (args.score or args.figures):
        parser.error("choose --score, --figures or --self-check")

    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.json or (args.out / "decade_climate.json")
    if args.score:
        report = score(args.lego_root, args.nemo_dir, months=args.months,
                       clim_first_month=args.clim_first_month, seed=args.seed,
                       mesh_path=args.mesh, plant=args.plant)
        destination.write_text(json.dumps(report, indent=2))
        print(json.dumps({
            "json": str(destination),
            "months": report["months"],
            "T3D_rms_first_last": [report["difference_series"]["T3D_rms"][0],
                                   report["difference_series"]["T3D_rms"][-1]],
            "climatology_ratios": {name: row["ratio"] for name, row
                                   in report["climatology"].items()},
            "T3D_relative_trend_per_month":
                report["difference_trend_T3D"]["relative_trend_per_month"],
            "max_abs_T_volmean_difference":
                report["drift"]["max_abs_T_volmean_difference"],
        }, indent=2))
    if args.figures:
        report = json.loads(destination.read_text())
        for path in figures(report, args.out, args.lego_root, args.nemo_dir,
                            seed=args.seed, mesh_path=args.mesh):
            print(f"figure {path} {path.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
