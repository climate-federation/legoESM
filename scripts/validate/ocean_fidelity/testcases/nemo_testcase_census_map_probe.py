#!/usr/bin/env python3
"""OVERFLOW-zps ``final_water_mass_census``: anatomy of the one OUTSIDE row.

The statistic, read from the scorer rather than from its name
(``nemo_testcase_full_statistics.py:826-845`` and ``:1133-1143``):

* REGION -- wet cells whose column bathymetry satisfies ``500 m < H < 2000 m``
  (``arm_metrics``: ``slope = (bathy > 500.0) & (bathy < 2000.0)``), i.e. the
  continental SLOPE only.  Not the shelf, not the abyssal plain.
* TIME -- the FINAL state alone (``mapped[int(times[-1])]``), 6120 steps =
  61200 s.  The two earlier registered times enter no other census row.
* QUANTITY -- volume fractions of that region in three temperature classes,
  ``cold [10,12)``, ``mixed [12,18)``, ``ambient [18,20]``, weighted by the
  live partial-cell volume ``area * h * mask``, normalised to sum 1.
* REDUCTION -- ``_curve_distance``, i.e. ``max_i |left_i - right_i|`` over the
  three classes (NOT total variation; the sibling histogram row uses TV).
* Candidate ``L64`` vs ``N2``; precision floor ``L32`` vs ``L64``; scheme
  spread ``N4`` vs ``N2`` (NEMO FCT4 vs NEMO FCT2).

This tool answers WHERE, WHEN and WHAT, and nothing else: it prints no
verdict.  Its instrument control is that it must reproduce the committed
scorer's candidate/floor/spread for this row to the last digit before any of
its own maps are written (``--expect-scorer``).

fp64 only.  The scorer's own loaders, geometry, snapping and precision guard
are imported and reused, so the protocol is byte-identical to the scored run
(Rule 7); nothing here re-derives a census.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
CASE = "OVERFLOW-zps"
DEFAULT_OUT = Path("/data/abyssal/dbalwada/nemo-testcases-l1/census_map")
COMMITTED_SCORE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/phantom_velocity/after/stats/"
    "overflow_statistics.json"
)
DEFAULT_LEGO_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/phantom_velocity/after/stats/legoesm"
)


class ProbeError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ProbeError(message)


def _load(name: str, relative: str):
    path = REPO_ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STATS = _load(
    "nemo_l1_full_statistics",
    "scripts/validate/ocean_fidelity/testcases/nemo_testcase_full_statistics.py",
)
CLASS_EDGES = ((10.0, 12.0), (12.0, 18.0), (18.0, 20.0))
CLASS_NAMES = ("cold_10_12", "mixed_12_18", "ambient_18_20")


def git_sha() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _classify(values: np.ndarray) -> np.ndarray:
    """Class index per cell, matching the scorer's half-open bins exactly."""
    out = np.full(values.shape, -1, dtype=np.int64)
    out[(values >= 10.0) & (values < 12.0)] = 0
    out[(values >= 12.0) & (values < 18.0)] = 1
    out[(values >= 18.0) & (values <= 20.0)] = 2
    return out


def arm_state(source: str, fields: dict) -> dict:
    """Slope-region census plus the volumes/temperatures behind it, fp64."""
    card, active, volume, centres = STATS._geometry(CASE, fields["ssh"])
    bathy = np.asarray(card.recipe.initial_state.H_bathy.data, dtype=np.float64)
    slope = (bathy > 500.0) & (bathy < 2000.0)
    select = active & slope[..., None]
    temperature = np.asarray(fields["T"])
    require(temperature.dtype == np.float64, f"{source}: census map is fp64 only")
    guard, evidence = STATS.precision_floor_guard(CASE, source, temperature.dtype)
    snapped, receipt = STATS._snap_temperature(
        temperature[select], CASE, temperature.dtype,
        gross_guard_relative=guard, guard_evidence=evidence,
    )
    weights = volume[select]
    klass = _classify(snapped)
    require(np.all(klass >= 0), f"{source}: cell outside every census class")
    absolute = np.asarray(
        [float(np.sum(weights[klass == index])) for index in range(3)], dtype=np.float64
    )
    total = float(np.sum(absolute))
    # whole-domain (unrestricted) cold/mixed/ambient volume: the water-mass
    # question without the slope mask, so a redistribution INTO or OUT OF the
    # slope band is separable from a change in how much cold water exists.
    all_T = np.asarray(fields["T"])[active]
    all_w = volume[active]
    all_klass = _classify(np.clip(all_T, 10.0, 20.0))
    domain = np.asarray(
        [float(np.sum(all_w[all_klass == index])) for index in range(3)], dtype=np.float64
    )
    return {
        "source": source,
        "census": (absolute / total).tolist(),
        "class_volume_m3": absolute.tolist(),
        "slope_volume_m3": total,
        "slope_cells": int(np.count_nonzero(select)),
        "slope_mean_T_C": float(np.sum(weights * snapped) / total),
        "slope_heat_K_m3": float(np.sum(weights * snapped)),
        "domain_class_volume_m3": domain.tolist(),
        "domain_volume_m3": float(np.sum(all_w)),
        "domain_mean_T_C": float(np.sum(all_w * all_T) / np.sum(all_w)),
        "domain_min_T_C": float(np.min(all_T)),
        "domain_max_T_C": float(np.max(all_T)),
        "salinity_range": [float(np.min(fields["S"][active])), float(np.max(fields["S"][active]))],
        "temperature_range_receipt": receipt,
        "_select": select,
        "_volume": volume,
        "_snapped_full": np.clip(temperature, 10.0, 20.0),
        "_centres": centres,
        "_active": active,
        "_bathy": bathy,
        "_card": card,
    }


def difference_map(left: dict, right: dict, label: str) -> dict:
    """Where the census fractions differ: by column, by level, by class."""
    require(np.array_equal(left["_select"], right["_select"]), f"{label}: slope masks differ")
    select = left["_select"]
    ny, nx, nz = select.shape
    lv, rv = left["_volume"], right["_volume"]
    lt, rt = left["_snapped_full"], right["_snapped_full"]
    lk, rk = _classify(lt), _classify(rt)
    ltot, rtot = left["slope_volume_m3"], right["slope_volume_m3"]

    # per-cell signed contribution to each class's FRACTION difference
    contrib = np.zeros((3, ny, nx, nz), dtype=np.float64)
    for index in range(3):
        contrib[index] = (
            np.where(select & (lk == index), lv, 0.0) / ltot
            - np.where(select & (rk == index), rv, 0.0) / rtot
        )
    fraction_delta = contrib.reshape(3, -1).sum(axis=1)
    measured = np.asarray(left["census"]) - np.asarray(right["census"])
    require(
        np.max(np.abs(fraction_delta - measured)) < 1e-12,
        f"{label}: contribution decomposition does not close ({fraction_delta} vs {measured})",
    )

    # the two mechanisms, separated: cells that CHANGE CLASS, and the volume
    # (thickness/ssh) change of cells that do not.
    both = select & (lk >= 0) & (rk >= 0)
    moved = both & (lk != rk)
    stayed = both & (lk == rk)
    moved_volume = float(np.sum(np.where(moved, 0.5 * (lv + rv), 0.0)))
    dT = np.where(both, lt - rt, 0.0)
    moved_dT = np.abs(dT[moved])
    # distance of each reclassified cell's temperature to the nearest bin edge
    edges = np.asarray([12.0, 18.0])
    mid = 0.5 * (lt + rt)
    edge_gap = np.min(np.abs(mid[..., None] - edges), axis=-1)
    volume_change_only = float(
        np.sum(np.abs(np.where(stayed, lv / ltot - rv / rtot, 0.0)))
    )

    def by_axis(axis_name: str) -> dict:
        axes = {"x": (0, 3), "z": (0, 1), "y": (1, 2)}[axis_name]
        return {
            name: np.sum(contrib[index], axis=axes).tolist()
            for index, name in enumerate(CLASS_NAMES)
        }

    x_km = STATS._x_centres_km(left["_card"], STATS._section_row(left["_active"]))
    dT_slope = np.where(select, np.abs(dT), 0.0)
    peak = np.unravel_index(int(np.argmax(dT_slope)), dT_slope.shape)
    return {
        "label": label,
        "census_delta": measured.tolist(),
        "census_max_abs_delta": float(np.max(np.abs(measured))),
        "slope_volume_delta_m3": ltot - rtot,
        "slope_volume_relative_delta": (ltot - rtot) / rtot,
        "slope_mean_T_delta_K": left["slope_mean_T_C"] - right["slope_mean_T_C"],
        "domain_class_volume_delta_m3": (
            np.asarray(left["domain_class_volume_m3"])
            - np.asarray(right["domain_class_volume_m3"])
        ).tolist(),
        "domain_mean_T_delta_K": left["domain_mean_T_C"] - right["domain_mean_T_C"],
        "reclassified_volume_m3": moved_volume,
        "reclassified_volume_fraction": moved_volume / (0.5 * (ltot + rtot)),
        "reclassified_cells": int(np.count_nonzero(moved)),
        "reclassified_abs_dT_K": {
            "median": float(np.median(moved_dT)) if moved_dT.size else 0.0,
            "p90": float(np.percentile(moved_dT, 90)) if moved_dT.size else 0.0,
            "max": float(np.max(moved_dT)) if moved_dT.size else 0.0,
        },
        "reclassified_edge_gap_K": {
            "median": float(np.median(edge_gap[moved])) if moved_dT.size else 0.0,
            "p90": float(np.percentile(edge_gap[moved], 90)) if moved_dT.size else 0.0,
        },
        "same_class_volume_reshuffle_fraction": volume_change_only,
        "slope_dT_stats_K": {
            "linf": float(np.max(dT_slope)),
            "rms": float(np.sqrt(np.mean(dT[select] ** 2))),
            "mean": float(np.mean(dT[select])),
            "peak_index_yxz": [int(v) for v in peak],
            "peak_x_km": float(x_km[peak[1]]),
            "peak_depth_m": float(left["_centres"][peak]),
            "peak_bathy_m": float(left["_bathy"][peak[0], peak[1]]),
        },
        "contribution_by_x": by_axis("x"),
        "contribution_by_z": by_axis("z"),
        "reclassified_volume_by_x": np.sum(
            np.where(moved, 0.5 * (lv + rv), 0.0), axis=(0, 2)
        ).tolist(),
        "x_centres_km": x_km.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lego-root", type=Path, default=DEFAULT_LEGO_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT / "census_map.json")
    parser.add_argument("--expect-scorer", type=Path, default=COMMITTED_SCORE)
    args = parser.parse_args()

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    set_policy(PrecisionPolicy.fp64())
    require(get_policy().control is not None, "precision policy not set")

    spec = STATS.CASES[CASE]
    arms_raw = {
        "N2": STATS.load_nemo_states(CASE, Path(spec["baseline"]), baseline=True),
        "N4": STATS.load_nemo_states(CASE, Path(spec["alternative"]), baseline=False),
        "L64": STATS.load_legoesm_states(CASE, "fp64", args.lego_root)[0],
        "L32": STATS.load_legoesm_states(CASE, "fp32", args.lego_root)[0],
    }
    times = sorted(arms_raw["N2"])
    for name, states in arms_raw.items():
        require(sorted(states) == times, f"{name}: registered times differ")
    final = int(times[-1])

    per_time = {}
    for name, states in arms_raw.items():
        per_time[name] = {}
        for time_s in times:
            fields = STATS.mapped_fields(states[int(time_s)], name)
            fields = {k: np.asarray(v, dtype=np.float64) for k, v in fields.items()}
            per_time[name][int(time_s)] = arm_state(name, fields)

    # INSTRUMENT CONTROL, run before any map is reported: the probe's own
    # final-time census must reproduce the committed scorer's three numbers.
    scored = json.loads(args.expect_scorer.read_text())
    row = next(r for r in scored["rows"] if r["name"] == "final_water_mass_census")
    def dist(a: str, b: str) -> float:
        return float(np.max(np.abs(
            np.asarray(per_time[a][final]["census"]) - np.asarray(per_time[b][final]["census"])
        )))
    control = {
        "candidate": [dist("L64", "N2"), row["candidate_distance"]],
        "precision_floor": [dist("L32", "L64"), row["precision_floor"]],
        "scheme_spread": [dist("N4", "N2"), row["scheme_spread"]],
    }
    for key, (mine, theirs) in control.items():
        require(
            abs(mine - theirs) <= 1e-15 * max(1.0, abs(theirs)),
            f"instrument control FAILED on {key}: probe {mine!r} vs scorer {theirs!r}",
        )

    maps = {
        "L64_minus_N2": difference_map(
            per_time["L64"][final], per_time["N2"][final], "L64-N2"),
        "N4_minus_N2": difference_map(
            per_time["N4"][final], per_time["N2"][final], "N4-N2"),
        "L32_minus_L64": difference_map(
            per_time["L32"][final], per_time["L64"][final], "L32-L64"),
    }
    mid = int(times[1])
    maps["L64_minus_N2_midpoint"] = difference_map(
        per_time["L64"][mid], per_time["N2"][mid], "L64-N2 @midpoint")
    maps["N4_minus_N2_midpoint"] = difference_map(
        per_time["N4"][mid], per_time["N2"][mid], "N4-N2 @midpoint")

    payload = {
        "case": CASE,
        "git_sha": git_sha(),
        "precision": "fp64",
        "class_names": list(CLASS_NAMES),
        "class_edges_C": [list(e) for e in CLASS_EDGES],
        "region": "wet cells with 500 m < H_bathy < 2000 m (the slope)",
        "scored_time_s": final,
        "registered_times_s": [int(t) for t in times],
        "frames": (
            "t=0 and t=midpoint are NEMO Nbb step entries vs legoESM pre-step "
            "prognostic; t=final is the NEMO tn/un restart vs legoESM after the "
            "full duration -- the scorer's own deterministic bridge"
        ),
        "instrument_control": control,
        "lego_root": str(args.lego_root),
        "arms": {
            name: {
                str(t): {k: v for k, v in state.items() if not k.startswith("_")}
                for t, state in by_time.items()
            }
            for name, by_time in per_time.items()
        },
        "maps": maps,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {args.out}")
    print("instrument control (probe vs committed scorer):")
    for key, (mine, theirs) in control.items():
        print(f"  {key:16s} {mine!r} vs {theirs!r}")
    for name in ("N2", "N4", "L64", "L32"):
        state = per_time[name][final]
        print(f"{name} census {np.asarray(state['census'])}  "
              f"slope_vol {state['slope_volume_m3']:.6e}  meanT {state['slope_mean_T_C']:.9f}")
