#!/usr/bin/env python
"""#1455 characterize and attribute DINO transport gaps at rows 13 and 49.

Pre-registration: ``PREREG_row13_edge_attribution.md`` at ``ea416520b``.
Adversarial-review correction protocol: the same file at ``59646d40d``,
committed before corrected statistics were computed.  This probe is offline:
it loads the saved verdict360 states and never steps either model.

Usage:
  JAX_ENABLE_X64=1 row13_edge_attribution.py --self-test
  JAX_ENABLE_X64=1 row13_edge_attribution.py --out-dir /tmp/dino_row13
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import legoesm.ocean
import numpy as np

import legoesm
from legoesm import constants as legoesm_constants

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parents[3]
sys.path.insert(0, str(_DIR))
import channel_rescore as C  # noqa: E402,N812 -- pinned sibling instrument
import regional_audit as R  # noqa: E402,N812 -- pinned upstream instrument

HORIZONS = C.HORIZONS
N_MEM = R.N_MEM
TARGETS = {
    13: {
        "basin_rows": slice(0, 13),
        "basin_neighbor": 12,
        "channel_neighbor": 14,
        "basin_name": "P1 south of band without row 13",
    },
    49: {
        "basin_rows": slice(50, 79),
        "basin_neighbor": 50,
        "channel_neighbor": 48,
        "basin_name": "P3 southern subtropics without row 49",
    },
}
TARGET_ROWS = tuple(TARGETS)

R_HIGH = 0.70
R_LOW = 0.30
R_NEIGHBOR = 0.50
R_MARGIN = 0.20
SYMMETRY_RATIO = (0.75, 1.25)
ASYMMETRY_RATIO_MAX = 0.50
ROW_BAR = R.V.K_PREREG
EFFECTIVE_INDEPENDENT_MEMBERS = 1
CORRECTION_COMMIT = "59646d40dc80302d1d21e78f1898c538f8bb203f"

UPSTREAM_REGIONAL_COMMIT = "102ef501a"
UPSTREAM_REGIONAL_SHA256 = "2f2bbc03afed792ea0148002029a6ad6481c93bd2ee37679b4bf615aab392fe7"
UPSTREAM_CHANNEL_COMMIT = "7af72bc33"
UPSTREAM_CHANNEL_SHA256 = "a3b51de6292df22b90fa9bd385dcb22ab47a3049c737f2e2abf1a28eb5a18be9"


def _f64(a):
    return np.asarray(a, dtype=np.float64)


def _git_state():
    sha = subprocess.run(
        ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(_ROOT), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return sha, dirty


def _path_inside(path, root=_ROOT):
    try:
        Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def verify_import_locations():
    """Refuse a Python environment resolving model code outside this tree."""
    locations = {
        "legoesm_namespace_paths": [str(Path(p).resolve()) for p in legoesm.__path__],
        "legoesm_constants": str(Path(legoesm_constants.__file__).resolve()),
        "legoesm_ocean": str(Path(legoesm.ocean.__file__).resolve()),
        "regional_audit": str(Path(R.__file__).resolve()),
        "channel_rescore": str(Path(C.__file__).resolve()),
        "probe": str(Path(__file__).resolve()),
        "legoesm.ocean_paths": [str(Path(p).resolve()) for p in legoesm.ocean.__path__],
    }
    direct = [
        locations[k]
        for k in (
            "legoesm_constants",
            "legoesm_ocean",
            "regional_audit",
            "channel_rescore",
            "probe",
        )
    ]
    if not all(_path_inside(p) for p in direct):
        raise SystemExit(f"FATAL: imports escaped this worktree: {locations}")
    if not locations["legoesm_namespace_paths"] or not _path_inside(
        locations["legoesm_namespace_paths"][0]
    ):
        raise SystemExit(f"FATAL: legoesm namespace does not prefer this worktree: {locations}")
    if not locations["legoesm.ocean_paths"] or not all(
        _path_inside(p) for p in locations["legoesm.ocean_paths"]
    ):
        raise SystemExit(f"FATAL: legoesm.ocean escaped this worktree: {locations}")
    return locations


def verify_upstream_hashes():
    got = {
        "regional_audit": hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest(),
        "channel_rescore": hashlib.sha256(Path(C.__file__).read_bytes()).hexdigest(),
    }
    expected = {
        "regional_audit": UPSTREAM_REGIONAL_SHA256,
        "channel_rescore": UPSTREAM_CHANNEL_SHA256,
    }
    if got != expected:
        raise SystemExit(f"FATAL: reducer SHA-256 mismatch: got {got}, expected {expected}")
    return got


def _assert_close(a, b, tol=1e-11):
    if not abs(float(a) - float(b)) <= tol:
        raise AssertionError(f"closure failed: {a} != {b} within {tol}")


def _plant_fires(label, fn):
    try:
        fn()
    except (AssertionError, SystemExit, ValueError):
        print(f"  PLANT FIRED: {label}")
        return
    raise AssertionError(f"planted violation did not fire: {label}")


def one_row_bt_bc(u, row):
    """Audit reducer's exact one-row bottom-reference/shear split [Sv]."""
    bt, bc = R.D.section_bt_bc(u, R.A.umask, rows=slice(row, row + 1))
    return float(R.D._avg(bt)), float(R.D._avg(bc))


def depth_contributions(u, row):
    """Thickness-weighted contribution of every level to one row [Sv]."""
    us = _f64(u)[row, 2:-2, :]
    wet = np.asarray(R.A.umask, dtype=bool)[row, 2:-2, :]
    e3 = _f64(R.A.e3t1d)
    e2 = float(_f64(R.A.e2u_col)[row])
    return np.mean(np.where(wet, us, 0.0), axis=0) * e3 * e2 / 1e6


def zonal_contributions(u, row):
    """Contribution of every longitude to the scored row mean [Sv]."""
    us = _f64(u)[row]
    wet = np.asarray(R.A.umask, dtype=bool)[row]
    per_lon = (
        np.sum(np.where(wet, us, 0.0) * _f64(R.A.e3t1d)[None, :], axis=1)
        * float(_f64(R.A.e2u_col)[row])
        / 1e6
    )
    out = np.zeros(R.A.NX, dtype=np.float64)
    out[2:-2] = per_lon[2:-2] / (R.A.NX - 4)
    return out


def _cyclic_distance_one(indices, n):
    out = set()
    for i in indices:
        out.add((int(i) - 1) % n)
        out.add((int(i) + 1) % n)
    return out


def zonal_classes(row):
    """Fixed blocked/near-blocked/interior partition of scored U columns."""
    open_col = np.asarray(R.A.umask, dtype=bool)[row].any(axis=1)
    blocked = set(np.flatnonzero(~open_col).tolist())
    near = _cyclic_distance_one(blocked, R.A.NX)
    scored = set(range(2, R.A.NX - 2))
    classes = {
        "blocked": np.asarray(sorted(scored & blocked), dtype=int),
        "near_blocked": np.asarray(sorted((scored & near) - blocked), dtype=int),
        "open_interior": np.asarray(sorted(scored - blocked - near), dtype=int),
    }
    joined = np.concatenate(list(classes.values()))
    if len(joined) != R.A.NX - 4 or len(np.unique(joined)) != len(joined):
        raise AssertionError("zonal classes do not partition scored longitudes")
    return classes, np.flatnonzero(~open_col), open_col


def open_segments(open_col):
    """Contiguous open-column segments on the cyclic longitude ring."""
    a = np.asarray(open_col, dtype=bool)
    if a.all():
        return [list(range(len(a)))]
    if not a.any():
        return []
    start = int(np.flatnonzero(~a)[0])
    ordered = [int((start + 1 + k) % len(a)) for k in range(len(a))]
    segments, current = [], []
    for i in ordered:
        if a[i]:
            current.append(i)
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def depth_summary(gap):
    a = _f64(gap)
    gross = float(np.sum(np.abs(a)))
    classes = {}
    for name, levels in R.X.DEPTH_CLASSES:
        signed = float(np.sum(a[levels]))
        absolute = float(np.sum(np.abs(a[levels])))
        classes[name] = {
            "signed_sv": signed,
            "gross_absolute_sv": absolute,
            "gross_fraction": absolute / gross if gross > 0.0 else None,
        }
    centroid = None
    if gross > 0.0:
        centroid = float(np.sum(np.abs(a) * _f64(R.A.gdept1d)) / gross)
    return {"classes": classes, "gross_absolute_sv": gross, "gross_depth_centroid_m": centroid}


def zonal_summary(gap, classes):
    a = _f64(gap)
    gross = float(np.sum(np.abs(a)))
    out = {}
    for name, idx in classes.items():
        signed = float(np.sum(a[idx]))
        absolute = float(np.sum(np.abs(a[idx])))
        out[name] = {
            "columns": idx.tolist(),
            "signed_sv": signed,
            "gross_absolute_sv": absolute,
            "gross_fraction": absolute / gross if gross > 0.0 else None,
        }
    return {"classes": out, "gross_absolute_sv": gross}


def component_values(states):
    """Compute target component/depth/zonal structures for loaded states."""
    out = {}
    for side, by_day in states.items():
        out[side] = {}
        for day, members in by_day.items():
            out[side][day] = {}
            for m, st in enumerate(members):
                out[side][day][m] = {}
                for row in TARGET_ROWS:
                    total = float(R.row_transports(st["u"], R.A.umask)[row])
                    bt, bc = one_row_bt_bc(st["u"], row)
                    depth = depth_contributions(st["u"], row)
                    zonal = zonal_contributions(st["u"], row)
                    _assert_close(bt + bc, total, 1e-9)
                    _assert_close(np.sum(depth), total, 1e-9)
                    _assert_close(np.sum(zonal), total, 1e-9)
                    out[side][day][m][row] = {
                        "total_sv": total,
                        "bottom_reference_sv": bt,
                        "shear_sv": bc,
                        "depth_sv": depth,
                        "zonal_sv": zonal,
                    }
    return out


def relationship_status(
    basin_profile,
    basin_neighbor,
    channel_profile,
    channel_neighbor,
):
    """Apply post-review descriptive routing, including opposite-side vetoes."""
    basin_veto = basin_neighbor < channel_neighbor
    channel_veto = channel_neighbor < basin_neighbor
    basin_ok = (
        basin_profile >= R_HIGH
        and basin_neighbor >= R_NEIGHBOR
        and basin_profile - channel_profile >= R_MARGIN
        and not basin_veto
    )
    channel_ok = (
        channel_profile >= R_HIGH
        and channel_neighbor >= R_NEIGHBOR
        and channel_profile - basin_profile >= R_MARGIN
        and not channel_veto
    )
    neither = (
        basin_profile <= R_LOW
        and channel_profile <= R_LOW
        and basin_neighbor <= R_LOW
        and channel_neighbor <= R_LOW
    )
    if basin_ok:
        status = "DESCRIPTIVE_BASIN_EDGE_MATCH_LOW_N"
    elif channel_ok:
        status = "DESCRIPTIVE_CHANNEL_EDGE_MATCH_LOW_N"
    elif neither:
        status = "DESCRIPTIVE_OWN_OBJECT_NEITHER_LOW_N"
    else:
        status = "UNRESOLVED_LOW_N"
    return status, basin_veto, channel_veto


def symmetry_status(amplitude_ratio, trajectory_abs):
    """Amplitude-aware descriptive cell; correlations do not provide replication."""
    if (
        SYMMETRY_RATIO[0] <= amplitude_ratio <= SYMMETRY_RATIO[1]
        and trajectory_abs >= R_HIGH
    ):
        return "DESCRIPTIVE_TOPOLOGY_SYMMETRIC_LOW_N"
    if amplitude_ratio <= ASYMMETRY_RATIO_MAX:
        return "DESCRIPTIVE_ROW13_AMPLITUDE_DOMINANCE_UNSATURATED"
    return "UNRESOLVED_LOW_N"


def exact_n4_null_p(r):
    """Exact two-sided Pearson tail under the n=4 independent Gaussian null."""
    return 1.0 - abs(float(r))


def _loo_profile_amplitudes(paired_rows, rows):
    """Horizon-held-out projection onto an RMS-normalized regional template."""
    indices = np.arange(R.A.NY)[rows]
    n_rows = int(len(indices))
    amp = np.empty((N_MEM, len(HORIZONS)), dtype=np.float64)
    folds = []
    for ti, day in enumerate(HORIZONS):
        training = [h for h in HORIZONS if h != day]
        template = np.mean(
            np.concatenate([_f64(paired_rows[h][:, rows]) for h in training], axis=0),
            axis=0,
        )
        squared_norm = float(np.dot(template, template))
        rms = float(np.sqrt(np.mean(template * template)))
        if not rms > 0.0:
            raise ValueError("profile projection is UNMEASURABLE on a zero-RMS template")
        divisor = float(n_rows * rms)
        amp[:, ti] = _f64(paired_rows[day][:, rows]) @ template / divisor
        folds.append(
            {
                "held_out_day": int(day),
                "training_days": training,
                "n_rows": n_rows,
                "template_mean_gap_sv": template.tolist(),
                "template_squared_norm_sv2": squared_norm,
                "template_rms_sv": rms,
                "projection_divisor_row_sv": divisor,
            }
        )
    return amp, folds


def _correlation_record(target, reference):
    signed = np.asarray(
        [C.pearson(target[m], reference[m])[0] for m in range(N_MEM)], dtype=np.float64
    )
    absolute = np.abs(signed)
    return {
        "primary_member": 0,
        "primary_signed_r": float(signed[0]),
        "primary_abs_r": float(absolute[0]),
        "primary_exact_n4_null_two_sided_p": exact_n4_null_p(signed[0]),
        "all_member_signed_r": signed.tolist(),
        "all_member_exact_n4_null_two_sided_p": [exact_n4_null_p(r) for r in signed],
        "nudge_sensitivity_abs_r_range_members_1_3": [
            float(np.min(absolute[1:])),
            float(np.max(absolute[1:])),
        ],
        "descriptive_all_member_median_abs_r": float(np.median(absolute)),
        "effective_independent_member_tests_approx": EFFECTIVE_INDEPENDENT_MEMBERS,
    }


def _relationship_from_trajectories(
    target, basin_amp, channel_amp, basin_neighbor, channel_neighbor
):
    records = {
        "basin_profile": _correlation_record(target, basin_amp),
        "channel_profile": _correlation_record(target, channel_amp),
        "basin_neighbor": _correlation_record(target, basin_neighbor),
        "channel_neighbor": _correlation_record(target, channel_neighbor),
    }
    primary = {name: rec["primary_abs_r"] for name, rec in records.items()}
    status, basin_veto, channel_veto = relationship_status(
        primary["basin_profile"],
        primary["basin_neighbor"],
        primary["channel_profile"],
        primary["channel_neighbor"],
    )
    return status, records, primary, basin_veto, channel_veto


def attribution(paired_rows):
    """Review-corrected low-n attribution for both topology-matched rows."""
    channel_amp, channel_folds = _loo_profile_amplitudes(paired_rows, C.CHANNEL_ROWS)
    result = {}
    for row, spec in TARGETS.items():
        basin_amp, basin_folds = _loo_profile_amplitudes(paired_rows, spec["basin_rows"])
        target = np.asarray(
            [[paired_rows[d][m, row] for d in HORIZONS] for m in range(N_MEM)], dtype=np.float64
        )
        basin_neighbor = np.asarray(
            [[paired_rows[d][m, spec["basin_neighbor"]] for d in HORIZONS] for m in range(N_MEM)],
            dtype=np.float64,
        )
        channel_neighbor = np.asarray(
            [[paired_rows[d][m, spec["channel_neighbor"]] for d in HORIZONS] for m in range(N_MEM)],
            dtype=np.float64,
        )

        status, records, primary, basin_veto, channel_veto = _relationship_from_trajectories(
            target, basin_amp, channel_amp, basin_neighbor, channel_neighbor
        )
        result[row] = {
            "status": status,
            "basin_name": spec["basin_name"],
            "basin_rows": list(range(R.A.NY))[spec["basin_rows"]],
            "basin_neighbor": spec["basin_neighbor"],
            "channel_neighbor": spec["channel_neighbor"],
            "target_trajectories_sv": target.tolist(),
            "basin_loo_rms_normalized_projection_sv_per_row": basin_amp.tolist(),
            "channel_loo_rms_normalized_projection_sv_per_row": channel_amp.tolist(),
            "correlation_receipts": records,
            "primary_control_member_abs_r": primary,
            "primary_profile_separation_abs_r": float(
                primary["basin_profile"] - primary["channel_profile"]
            ),
            "opposite_neighbor_veto": {
                "basin_route_vetoed": bool(basin_veto),
                "channel_route_vetoed": bool(channel_veto),
            },
            "basin_leave_one_horizon_out_folds": basin_folds,
            "channel_leave_one_horizon_out_folds": channel_folds,
            "qualification": (
                "exploratory low-n trajectory relationship only; n=4 exact null "
                "p=1-|r|; nudge members are sensitivity traces, not replication"
            ),
        }
    return result


def _spread_record(row_values, row):
    history = {}
    by_day = {}
    for day in HORIZONS:
        lv = row_values["lego"][day][:, row]
        nv = row_values["nemo"][day][:, row]
        floor, ls, ns = R.two_sided_floor(lv, nv)
        history[("lego", day)] = ls
        history[("nemo", day)] = ns
        by_day[day] = {"floor_sv": floor, "lego_spread_sv": ls, "nemo_spread_sv": ns}
    saturated, reason = R.saturated(history)
    for day in HORIZONS:
        gap = float(row_values["lego"][day][0, row] - row_values["nemo"][day][0, row])
        floor = by_day[day]["floor_sv"]
        verdict, ratio, flags = R.classify(
            gap,
            floor,
            by_day[day]["lego_spread_sv"],
            by_day[day]["nemo_spread_sv"],
            None,
            saturated if day == 360 else False,
        )
        by_day[day].update(
            {
                "control_gap_sv": gap,
                "gap_over_floor": ratio,
                "audit_verdict": verdict,
                "audit_flags": flags,
                "floor_saturation": (
                    {"saturated": saturated, "reason": reason}
                    if day == 360
                    else {
                        "saturated": None,
                        "reason": "NOT_SCORED: registered quarters end at day 360",
                    }
                ),
                "member_gaps_sv": (
                    row_values["lego"][day][:, row] - row_values["nemo"][day][:, row]
                ).tolist(),
            }
        )
        if day == 360 and ratio is not None:
            material, x2yes, growth = R.u_is_material(ratio, history, day)
            by_day[day]["unsaturated_materiality"] = {
                "could_overturn": bool(material),
                "floor_growth_needed": float(x2yes),
                "max_last_quarter_side_growth": float(growth),
            }
    return by_day, history


def _component_floor(components, key, row, day):
    lv = np.asarray([components["lego"][day][m][row][key] for m in range(N_MEM)])
    nv = np.asarray([components["nemo"][day][m][row][key] for m in range(N_MEM)])
    floor, ls, ns = R.two_sided_floor(lv, nv)
    return {"floor_sv": floor, "lego_spread_sv": ls, "nemo_spread_sv": ns}


def _full_characterization(row_values, components):
    result, histories = {}, {}
    for row in TARGET_ROWS:
        trajectory, histories[row] = _spread_record(row_values, row)
        classes, blocked, open_col = zonal_classes(row)
        structures = {}
        for day in HORIZONS:
            structures[day] = {}
            for m in range(N_MEM):
                lego_component = components["lego"][day][m][row]
                nemo_component = components["nemo"][day][m][row]
                depth_gap = lego_component["depth_sv"] - nemo_component["depth_sv"]
                zonal_gap = lego_component["zonal_sv"] - nemo_component["zonal_sv"]
                structures[day][m] = {
                    "total_gap_sv": (lego_component["total_sv"] - nemo_component["total_sv"]),
                    "bottom_reference_gap_sv": (
                        lego_component["bottom_reference_sv"]
                        - nemo_component["bottom_reference_sv"]
                    ),
                    "shear_gap_sv": (lego_component["shear_sv"] - nemo_component["shear_sv"]),
                    "lego": {
                        "total_sv": lego_component["total_sv"],
                        "bottom_reference_sv": lego_component["bottom_reference_sv"],
                        "shear_sv": lego_component["shear_sv"],
                        "depth_sv": lego_component["depth_sv"].tolist(),
                        "zonal_sv": lego_component["zonal_sv"].tolist(),
                    },
                    "nemo": {
                        "total_sv": nemo_component["total_sv"],
                        "bottom_reference_sv": nemo_component["bottom_reference_sv"],
                        "shear_sv": nemo_component["shear_sv"],
                        "depth_sv": nemo_component["depth_sv"].tolist(),
                        "zonal_sv": nemo_component["zonal_sv"].tolist(),
                    },
                    "depth_gap_sv": depth_gap.tolist(),
                    "depth_summary": depth_summary(depth_gap),
                    "zonal_gap_sv": zonal_gap.tolist(),
                    "zonal_summary": zonal_summary(zonal_gap, classes),
                }
        component_floors = {
            str(day): {
                key: _component_floor(components, key, row, day)
                for key in ("bottom_reference_sv", "shear_sv")
            }
            for day in HORIZONS
        }
        result[row] = {
            "trajectory": trajectory,
            "component_floors": component_floors,
            "structures": structures,
            "topology": {
                "wet_t_columns": int(np.asarray(R.A.tmask, dtype=bool)[row].any(axis=1).sum()),
                "wet_u_columns": int(open_col.sum()),
                "blocked_u_columns": blocked.tolist(),
                "wet_levels_per_u_column": np.asarray(R.A.umask, dtype=bool)[row]
                .sum(axis=1)
                .tolist(),
                "open_segments": open_segments(open_col),
                "scored_zonal_classes": {name: idx.tolist() for name, idx in classes.items()},
            },
        }
    return result, histories


def _symmetry(paired_rows, characterization):
    amplitude_cells = {}
    for day in HORIZONS:
        amp13 = float(np.median(np.abs(paired_rows[day][:, 13])))
        amp49 = float(np.median(np.abs(paired_rows[day][:, 49])))
        ratio = amp49 / amp13 if amp13 > 0.0 else float("inf")
        if SYMMETRY_RATIO[0] <= ratio <= SYMMETRY_RATIO[1]:
            amplitude_status = "DESCRIPTIVE_AMPLITUDE_MATCH"
        elif ratio <= ASYMMETRY_RATIO_MAX:
            amplitude_status = "DESCRIPTIVE_ROW13_AMPLITUDE_DOMINANCE"
        else:
            amplitude_status = "DESCRIPTIVE_OTHER_AMPLITUDE_RATIO"
        amplitude_cells[day] = {
            "row13_median_abs_gap_sv": amp13,
            "row49_median_abs_gap_sv": amp49,
            "row49_over_row13_ratio": ratio,
            "absolute_amplitude_difference_sv": abs(amp13 - amp49),
            "status": amplitude_status,
            "qualification": "saved-state amplitudes; row floors are unsaturated",
        }
    ratio = amplitude_cells[360]["row49_over_row13_ratio"]
    signed = np.asarray(
        [
            C.pearson(
                [paired_rows[d][m, 13] for d in HORIZONS],
                [paired_rows[d][m, 49] for d in HORIZONS],
            )[0]
            for m in range(N_MEM)
        ]
    )
    primary_abs = float(abs(signed[0]))
    status = symmetry_status(ratio, primary_abs)
    return {
        "status": status,
        "amplitude_aware_cells": amplitude_cells,
        "member_signed_trajectory_r": signed.tolist(),
        "primary_control_member_abs_trajectory_r": primary_abs,
        "primary_exact_n4_null_two_sided_p": exact_n4_null_p(signed[0]),
        "nudge_sensitivity_abs_r_range_members_1_3": [
            float(np.min(np.abs(signed[1:]))),
            float(np.max(np.abs(signed[1:]))),
        ],
        "descriptive_all_member_median_abs_trajectory_r": float(np.median(np.abs(signed))),
        "effective_independent_member_tests_approx": EFFECTIVE_INDEPENDENT_MEMBERS,
        "qualification": (
            "matched-topology low-n control; amplitudes are unsaturated upper bounds; "
            "P1 and P3 dynamics are not assumed symmetric"
        ),
    }


def _synthetic_pipeline_cases():
    """Exercise real LOO projection, Pearson, veto, and routing on trajectories."""
    rng = np.random.default_rng(1349)
    for _ in range(1000):
        paired = {
            day: rng.normal(size=(N_MEM, R.A.NY)).astype(np.float64) for day in HORIZONS
        }
        basin_amp, _ = _loo_profile_amplitudes(paired, TARGETS[13]["basin_rows"])
        channel_amp, _ = _loo_profile_amplitudes(paired, C.CHANNEL_ROWS)
        if abs(C.pearson(basin_amp[0], channel_amp[0])[0]) <= 0.15:
            break
    else:
        raise AssertionError("could not construct separated synthetic profile arms")

    def members(values):
        return np.tile(_f64(values), (N_MEM, 1))

    basin_target = members(basin_amp[0])
    channel_target = members(channel_amp[0])
    quiet = None
    for _ in range(1000):
        candidate = rng.normal(size=len(HORIZONS))
        if (
            abs(C.pearson(candidate, basin_amp[0])[0]) <= 0.25
            and abs(C.pearson(candidate, channel_amp[0])[0]) <= 0.25
        ):
            quiet = members(candidate)
            break
    if quiet is None:
        raise AssertionError("could not construct synthetic neither trajectory")

    mixed = members(basin_amp[0] + channel_amp[0])
    cases = {}
    for name, target, bn, cn in (
        ("basin", basin_target, basin_target, quiet),
        ("channel", channel_target, quiet, channel_target),
        ("neither", quiet, basin_target, channel_target),
        ("mixed", mixed, mixed, mixed),
    ):
        cases[name] = _relationship_from_trajectories(
            target, basin_amp, channel_amp, bn, cn
        )[0]
    return cases


def _load_full_states(provenance):
    states = {"lego": {}, "nemo": {}}
    for day in HORIZONS:
        print(f"loading full states day {day}: 4 legoESM + 4 NEMO", flush=True)
        states["lego"][day] = [R.load_lego(m, day) for m in range(N_MEM)]
        states["nemo"][day] = [R.load_nemo(m, day) for m in range(N_MEM)]
        for side in states:
            R.control_finite(
                f"{side} full target states day {day}",
                np.concatenate([st["u"][R.A.umask] for st in states[side][day]]),
            )
    for member in provenance["lego_members"]:
        with np.load(member["path"]) as artifact:
            for key in ("vertical_ladder_sha256", "twin_start_mode", "producer_git_sha"):
                member[key] = str(artifact[key]) if key in artifact.files else None
            member["available_stamp_keys"] = sorted(
                key
                for key in artifact.files
                if key
                in {
                    "control_dtype",
                    "nemo_ladder_mode",
                    "vertical_ladder_sha256",
                    "seasonal_t0_seconds",
                    "twin_start_mode",
                    "producer_git_sha",
                }
            )
    return states


def self_test(run_upstream=True):
    """Direct controls; every new guard is demonstrated on a violation."""
    print("ROW 13/49 EDGE ATTRIBUTION SELF-TEST")
    if run_upstream:
        R.self_test()
        C.self_test(run_upstream=False)

    wrong = UPSTREAM_CHANNEL_SHA256
    try:
        globals()["UPSTREAM_CHANNEL_SHA256"] = "0" * 64
        _plant_fires("reducer SHA mismatch", verify_upstream_hashes)
    finally:
        globals()["UPSTREAM_CHANNEL_SHA256"] = wrong

    rng = np.random.default_rng(13)
    u = rng.normal(size=(R.A.NY, R.A.NX, R.A.NZ))
    for row in TARGET_ROWS:
        total = float(R.row_transports(u, R.A.umask)[row])
        depth = depth_contributions(u, row)
        zonal = zonal_contributions(u, row)
        bt, bc = one_row_bt_bc(u, row)
        _assert_close(np.sum(depth), total, 1e-9)
        _assert_close(np.sum(zonal), total, 1e-9)
        _assert_close(bt + bc, total, 1e-9)
        _plant_fires(
            "depth closure catches planted level",
            lambda d=depth.copy(), t=total: _assert_close(np.sum(d) + 1e-3, t, 1e-9),
        )
        _plant_fires(
            "zonal closure catches planted column",
            lambda z=zonal.copy(), t=total: _assert_close(np.sum(z) - 1e-3, t, 1e-9),
        )
        _plant_fires(
            "split closure catches planted component",
            lambda b=bt, c=bc, t=total: _assert_close(b + c + 1e-3, t, 1e-9),
        )

    two_level = np.zeros_like(u)
    two_level[:, :, 0] = 1.0
    two_level[:, :, 1] = 1.0
    official = depth_contributions(two_level, 13)
    equal_layer = np.mean(np.where(R.A.umask[13], two_level[13], 0.0), axis=0)
    _plant_fires(
        "equal-layer average differs from thickness-weighted depth profile",
        lambda: np.testing.assert_allclose(
            official / np.sum(np.abs(official)),
            equal_layer / np.sum(np.abs(equal_layer)),
            rtol=1e-12,
            atol=1e-12,
        ),
    )

    for row in TARGET_ROWS:
        _, blocked, _ = zonal_classes(row)
        official = zonal_contributions(u, row)
        if not np.array_equal(official[blocked], np.zeros(blocked.size)):
            raise AssertionError("official blocked columns are nonzero")
        poisoned = official.copy()
        poisoned[blocked[0]] = 1e-3
        _plant_fires(
            "blocked-column poison",
            lambda p=poisoned, b=blocked: np.testing.assert_array_equal(p[b], np.zeros(len(b))),
        )

    gaps = np.asarray([1.0, 5.0])
    floors = np.asarray([0.1, 10.0])
    local = np.abs(gaps) > 5.0 * floors
    globalized = np.abs(gaps) > 5.0 * np.mean(floors)
    _plant_fires(
        "global floor cannot replace row floors",
        lambda: np.testing.assert_array_equal(globalized, local),
    )

    cases = _synthetic_pipeline_cases()
    expected = {
        "basin": "DESCRIPTIVE_BASIN_EDGE_MATCH_LOW_N",
        "channel": "DESCRIPTIVE_CHANNEL_EDGE_MATCH_LOW_N",
        "neither": "DESCRIPTIVE_OWN_OBJECT_NEITHER_LOW_N",
        "mixed": "UNRESOLVED_LOW_N",
    }
    if cases != expected:
        raise AssertionError((cases, expected))
    _plant_fires(
        "attribution outcomes cannot collapse to one verdict",
        lambda: np.testing.assert_array_equal(list(cases.values()), [cases["basin"]] * 4),
    )

    status = symmetry_status(1.0, 0.85)
    if status != "DESCRIPTIVE_TOPOLOGY_SYMMETRIC_LOW_N":
        raise AssertionError(status)
    dominance = symmetry_status(0.10, 0.20)
    if dominance != "DESCRIPTIVE_ROW13_AMPLITUDE_DOMINANCE_UNSATURATED":
        raise AssertionError(dominance)
    _plant_fires(
        "out-of-bar amplitude cannot pass symmetric control",
        lambda: (
            (_ for _ in ()).throw(AssertionError())
            if symmetry_status(1.5, 0.85)
            != "DESCRIPTIVE_TOPOLOGY_SYMMETRIC_LOW_N"
            else None
        ),
    )
    _plant_fires(
        "local-floor scaling cannot suppress amplitude-aware asymmetry",
        lambda: (
            (_ for _ in ()).throw(AssertionError())
            if symmetry_status(0.10, 0.20)
            == "DESCRIPTIVE_ROW13_AMPLITUDE_DOMINANCE_UNSATURATED"
            else None
        ),
    )
    print("ROW 13/49 SELF-TEST PASSED -- all planted violations fired")


def run(out_dir):
    sha_start, dirty_start = _git_state()
    if dirty_start:
        raise SystemExit("FATAL: producer tree is dirty; commit the probe first")
    imports = verify_import_locations()
    hashes = verify_upstream_hashes()
    self_test()

    row_lego, row_nemo, provenance, clock, clock_compat = C.load_rows()
    row_values = {"lego": row_lego, "nemo": row_nemo}
    states = _load_full_states(provenance)
    components = component_values(states)
    characterization, histories = _full_characterization(row_values, components)
    paired_rows = {day: row_lego[day] - row_nemo[day] for day in HORIZONS}
    relationships = attribution(paired_rows)
    symmetry = _symmetry(paired_rows, characterization)

    artifact = {
        "pre_registration": {
            "path": "scripts/validate/ocean_fidelity/dino_1226/PREREG_row13_edge_attribution.md",
            "commit": "ea416520b",
            "post_review_correction_commit": CORRECTION_COMMIT,
            "correction_is_post_registration": True,
        },
        "retractions": [
            (
                "RETRACTED: original day-360-template attribution; the scored horizon "
                "helped define the template and dominated n=4."
            ),
            (
                "RETRACTED: row 49 CONFIRMED P3 basin-edge; that claim depended on the "
                "circular statistic and ignored the stronger channel-side neighbour."
            ),
        ],
        "verdict_bars": {
            "R_HIGH": R_HIGH,
            "R_HIGH_exact_n4_null_two_sided_p": exact_n4_null_p(R_HIGH),
            "R_LOW": R_LOW,
            "R_LOW_exact_n4_null_two_sided_p": exact_n4_null_p(R_LOW),
            "R_NEIGHBOR": R_NEIGHBOR,
            "R_NEIGHBOR_exact_n4_null_two_sided_p": exact_n4_null_p(R_NEIGHBOR),
            "R_MARGIN": R_MARGIN,
            "R_MARGIN_qualification": (
                "descriptive routing only; not a significance threshold at n=4"
            ),
            "SYMMETRY_RATIO": list(SYMMETRY_RATIO),
            "ASYMMETRY_RATIO_MAX": ASYMMETRY_RATIO_MAX,
            "ROW_BAR": ROW_BAR,
            "correlation_bar_qualification": (
                "n=4 exact null p=1-|r|; unadjusted; four nudge members have "
                "effective independent tests approximately one"
            ),
        },
        "targets": list(TARGET_ROWS),
        "horizons": list(HORIZONS),
        "n_members_per_side": N_MEM,
        "characterization": characterization,
        "relationships": relationships,
        "symmetry_control": symmetry,
        "provenance": {
            **provenance,
            "clock": clock,
            "clock_helper_compatibility": clock_compat,
            "producer_sha": sha_start,
            "producer_dirty_start": dirty_start,
            "upstream_commits": {
                "regional_audit": UPSTREAM_REGIONAL_COMMIT,
                "channel_rescore": UPSTREAM_CHANNEL_COMMIT,
            },
            "upstream_sha256": hashes,
            "import_locations": imports,
            "python_executable": sys.executable,
            "pythonpath": os.environ.get("PYTHONPATH"),
            "command": sys.argv,
            "data_paths": {
                "lego_dir": R.X.LEGO_DIR,
                "lego_member_npz": [R.X.lego_npz(i) for i in range(N_MEM)],
                "nemo_member_directories": [R.X.nemo_dir(i) for i in range(N_MEM)],
            },
        },
        "floor_histories_by_side_day": {
            str(row): {f"{side}_day{day}": value for (side, day), value in history.items()}
            for row, history in histories.items()
        },
        "conventions": {
            "gap_sign": "legoESM - NEMO; positive eastward",
            "thickness": "e3t_1d; no layer averages",
            "longitude_reducer": "mean over columns 2:-2",
            "bottom_reference": ("u_bottom * H; not a true depth mean or independent ownership"),
            "relationship_scope": ("trajectory relationship, not causal or separably seasonal"),
            "profile_projection": (
                "leave-one-horizon-out template; RMS-normalized spatial projection"
            ),
            "member_inference": (
                "member 0 primary; 1e-14 nudge members are sensitivity traces, not replication"
            ),
        },
    }

    sha_end, dirty_end = _git_state()
    if sha_end != sha_start or dirty_end:
        raise SystemExit(
            f"FATAL: producer moved during run: {sha_start}/{dirty_start} -> {sha_end}/{dirty_end}"
        )
    artifact["provenance"]["producer_sha_end"] = sha_end
    artifact["provenance"]["producer_dirty_end"] = dirty_end
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "row13_edge_attribution.json"
    payload = json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False)
    path.write_text(payload + "\n")
    print(artifact["retractions"][0])
    print(artifact["retractions"][1])
    print(
        "CORRECTED METHOD: horizon-held-out RMS-normalized templates; member 0 primary; "
        "effective independent member tests ~= 1"
    )
    print(
        f"CORRELATION BARS: HIGH {R_HIGH:.2f} (exact n=4 null p={exact_n4_null_p(R_HIGH):.2f}); "
        f"NEIGHBOR {R_NEIGHBOR:.2f} (p={exact_n4_null_p(R_NEIGHBOR):.2f}); "
        f"LOW {R_LOW:.2f} (p={exact_n4_null_p(R_LOW):.2f}); descriptive only"
    )
    print(f"artifact: {path}")
    print(f"artifact sha256: {hashlib.sha256((payload + chr(10)).encode()).hexdigest()}")
    for row in TARGET_ROWS:
        d360 = characterization[row]["trajectory"][360]
        rec = relationships[row]["correlation_receipts"]
        print(
            f"row {row}: gap {d360['control_gap_sv']:+.9f} Sv, "
            f"floor {d360['floor_sv']:.9f} Sv, "
            f"{d360['gap_over_floor']:.3f}x, "
            f"{relationships[row]['status']}"
        )
        for name in ("basin_profile", "channel_profile", "basin_neighbor", "channel_neighbor"):
            receipt = rec[name]
            print(
                f"  {name}: control r={receipt['primary_signed_r']:+.4f}, "
                f"|r|={receipt['primary_abs_r']:.2f}, "
                f"exact n=4 null p={receipt['primary_exact_n4_null_two_sided_p']:.2f}"
            )
        print(
            "  basin-channel profile separation "
            f"{relationships[row]['primary_profile_separation_abs_r']:+.2f}; "
            "descriptive and inside n=4 sampling uncertainty"
        )
    print(f"symmetry: {symmetry['status']}")
    day360_cell = symmetry["amplitude_aware_cells"][360]
    print(
        "  amplitude-aware day360: "
        f"row49/row13={day360_cell['row49_over_row13_ratio']:.3f}, "
        f"difference={day360_cell['absolute_amplitude_difference_sv']:.6f} Sv, "
        f"{day360_cell['status']}"
    )
    print(
        "  trajectory control |r|="
        f"{symmetry['primary_control_member_abs_trajectory_r']:.2f}, "
        f"exact n=4 null p={symmetry['primary_exact_n4_null_two_sided_p']:.2f}"
    )
    return artifact


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out-dir", default="/tmp/dino_row13_edge_attribution")
    args = parser.parse_args(argv)
    if args.self_test:
        verify_import_locations()
        verify_upstream_hashes()
        self_test()
        return 0
    run(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
