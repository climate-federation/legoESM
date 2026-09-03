#!/usr/bin/env python3
"""OVERFLOW-zps ``final_water_mass_census``: anatomy of the one OUTSIDE row.

The statistic, read from the scorer rather than from its name
(``nemo_testcase_full_statistics.py:816-846`` and ``:1133-1143``):

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    # The scorer classifies at the ARM's OWN dtype (only the L32 precision-floor
    # arm is fp32) and accumulates the volume weights in fp64.  Reproduce that
    # exactly rather than up-casting, or the floor row cannot be reproduced.
    require(
        temperature.dtype == (np.float32 if source == "L32" else np.float64),
        f"{source}: unexpected temperature dtype {temperature.dtype}",
    )
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
    # Volume-weighted tracer VARIANCE is the standard measure of a scheme's
    # spurious diapycnal mixing: pure advection preserves it, every numerical
    # scheme destroys some.  The domain mean is identical across arms to
    # roundoff (heat is conserved), so the variance is directly comparable.
    domain_mean = float(np.sum(all_w * all_T) / np.sum(all_w))
    domain_variance = float(np.sum(all_w * (all_T - domain_mean) ** 2) / np.sum(all_w))
    # The cold ANOMALY's first and second moments about the ambient 20 C.
    # The first (the heat deficit) is conserved by advection and must agree
    # across arms; at fixed deficit the second says how CONCENTRATED the
    # anomaly is, and D**2 / M2 is the volume it effectively occupies.  This
    # is the same information as the variance -- the mean is common -- but in
    # units a plume argument can be made in.
    deficit = 20.0 - all_T
    anomaly_deficit = float(np.sum(all_w * deficit))
    anomaly_second_moment = float(np.sum(all_w * deficit * deficit))
    fine_bins = np.linspace(10.0, 20.0, 201)
    fine = np.histogram(np.clip(all_T, 10.0, 20.0), bins=fine_bins, weights=all_w)[0]
    row = STATS._section_row(active)
    bottom = STATS._bottom_values(np.asarray(fields["T"], dtype=np.float64), active, row)
    return {
        "source": source,
        "census": (absolute / total).tolist(),
        "domain_T_variance_K2": domain_variance,
        "anomaly_deficit_K_m3": anomaly_deficit,
        "anomaly_second_moment_K2_m3": anomaly_second_moment,
        "anomaly_effective_volume_m3": (
            anomaly_deficit ** 2 / anomaly_second_moment
            if anomaly_second_moment > 0.0 else 0.0),
        "anomaly_mean_amplitude_K": (
            anomaly_second_moment / anomaly_deficit
            if anomaly_deficit > 0.0 else 0.0),
        "domain_T_fine_histogram_m3": fine.tolist(),
        "domain_T_fine_bins_C": fine_bins.tolist(),
        "bottom_temperature_C": bottom.tolist(),
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
        "_klass_full": _classify(np.clip(temperature, 10.0, 20.0)),
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
    lt = np.asarray(left["_snapped_full"], dtype=np.float64)
    rt = np.asarray(right["_snapped_full"], dtype=np.float64)
    lk, rk = left["_klass_full"], right["_klass_full"]  # classified at each arm's own dtype
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
        axes = {"x": (0, 2), "z": (0, 1), "y": (1, 2)}[axis_name]
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
        "domain_T_variance_delta_K2": (
            left["domain_T_variance_K2"] - right["domain_T_variance_K2"]),
        "domain_T_variance_ratio": (
            left["domain_T_variance_K2"] / right["domain_T_variance_K2"]),
        "anomaly_deficit_relative_delta": (
            left["anomaly_deficit_K_m3"] / right["anomaly_deficit_K_m3"] - 1.0),
        "anomaly_effective_volume_ratio": (
            left["anomaly_effective_volume_m3"]
            / right["anomaly_effective_volume_m3"]),
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




# ---------------------------------------------------------------- faces ----
KT60_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l1/phase3/overflow_kt1_60")


def _mesh(ds, name: str, nlev: int) -> np.ndarray:
    """mesh_mask.nc is written on the INTERIOR frame already (1, nz, ny, nx);
    reorder to the gate's (ny, nx, nz) and drop NEMO's unused bottom level."""
    return np.asarray(ds.variables[name][0], dtype=np.float64).transpose(
        1, 2, 0)[..., :nlev]


def command_faces(args) -> None:
    """Enumerate what differs between the k=24 and k=25 u-face families.

    NEMO's rules, quoted (Rule 0):

      stprk3_stg.F90:440  zub = uu_b(Kaa) - SUM(e3u_0(:)*uu(:,Kaa)) * r1_hu_0
      stprk3_stg.F90:444  uu(jk,Kaa) = uu(jk,Kaa) + zub*umask(ji,jj,jk)
      domain.F90:145      hu_0 = SUM( e3u_0 * umask )
      dommsk / zps        umask(ji,jj,jk) = tmask(ji,jj,jk)*tmask(ji+1,jj,jk)

    legoESM's stage depth mean uses the LIVE face thickness and column depth
    (``_replace_stage_mean``: ``sum(u*h_u_pre)/H_u_pre``) where NEMO uses the
    REFERENCE ``e3u_0``/``r1_hu_0`` -- the phantom round's open item 3, never
    measured at this site.  Under z*/qco every level of a column scales by the
    same ``1 + r3u``, so the two weightings are algebraically identical; this
    command MEASURES that rather than asserting it.
    """
    import netCDF4

    from legoesm.core.precision import PrecisionPolicy, set_policy

    set_policy(PrecisionPolicy.fp64())
    card = STATS.build_nemo_testcase_card(CASE)
    masks = STATS.expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    h_ref = np.asarray(card.recipe.z_coord.h_partial, dtype=np.float64)

    with netCDF4.Dataset(Path(args.mesh_mask)) as ds:
        umask = _mesh(ds, "umask", nlev)
        tmask = _mesh(ds, "tmask", nlev)
        e3u_0 = _mesh(ds, "e3u_0", nlev)
        e3t_0 = _mesh(ds, "e3t_0", nlev)

    # CONTROL: NEMO's tmask must equal legoESM's active T mask, or every face
    # row below is comparing different columns.
    lego_t = masks["T"]
    require(
        np.array_equal(tmask > 0.5, lego_t),
        f"tmask mismatch: {int(np.count_nonzero((tmask > 0.5) != lego_t))} cells",
    )
    lego_u = masks["u"]                      # gate frame, (3, 202, nlev)
    require(
        np.array_equal(umask > 0.5, lego_u),
        f"umask mismatch: {int(np.count_nonzero((umask > 0.5) != lego_u))} faces",
    )
    hu_0_nemo = np.sum(e3u_0 * (umask > 0.5), axis=-1)
    # legoESM's reference face thickness, built the way the stage program does
    h_u_lego = np.zeros_like(h_ref)
    h_u_lego[:, :-1, :] = np.minimum(h_ref[:, :-1, :], h_ref[:, 1:, :])
    h_u_lego *= lego_u
    H_u_lego = np.sum(h_u_lego, axis=-1)

    row = STATS._section_row(lego_t)
    x_km = STATS._x_centres_km(card, row)
    bathy = np.asarray(card.recipe.initial_state.H_bathy.data, dtype=np.float64)
    bottom = np.where(lego_t.any(axis=-1), lego_t.shape[-1] - 1
                      - np.argmax(lego_t[:, :, ::-1], axis=-1), -1)

    faces = []
    for face in range(args.face_lo, args.face_hi + 1):     # gate face index
        entry = {
            "gate_face": face,
            "model_u_face": face + 1,
            "x_km_left_T": float(x_km[face]),
            "left_T_bottom_k": int(bottom[row, face]),
            "right_T_bottom_k": int(bottom[row, face + 1]),
            "left_bathy_m": float(bathy[row, face]),
            "right_bathy_m": float(bathy[row, face + 1]),
            "hu_0_nemo_m": float(hu_0_nemo[row, face]),
            "H_u_lego_m": float(H_u_lego[row, face]),
            "hu_0_minus_H_u": float(hu_0_nemo[row, face] - H_u_lego[row, face]),
            "levels": {},
        }
        for k in range(args.k_lo, args.k_hi + 1):
            entry["levels"][str(k)] = {
                "nemo_umask": float(umask[row, face, k]),
                "lego_u_active": bool(lego_u[row, face, k]),
                "lego_2d_u_mask": bool(lego_u[row, face].any()),
                "nemo_e3u_0_m": float(e3u_0[row, face, k]),
                "lego_h_u_ref_m": float(h_u_lego[row, face, k]),
                "e3u_0_minus_h_u": float(e3u_0[row, face, k] - h_u_lego[row, face, k]),
                "nemo_e3t_0_left_m": float(e3t_0[row, face, k]),
                "nemo_e3t_0_right_m": float(e3t_0[row, face + 1, k]),
                # the UP3 k-slab stencil neighbours dynadv_up3.F90:142-176 reads
                "up3_neighbour_umask": [
                    float(umask[row, max(face - 1, 0), k]),
                    float(umask[row, face, k]),
                    float(umask[row, min(face + 1, umask.shape[1] - 1), k]),
                ],
            }
        faces.append(entry)

    payload = {
        "case": CASE,
        "git_sha": git_sha(),
        "precision": "fp64",
        "section_row": row,
        "mesh_mask": str(args.mesh_mask),
        "mesh_mask_sha256": sha256(Path(args.mesh_mask)),
        "controls": {
            "tmask_equals_lego_active_T": True,
            "umask_equals_lego_active_u": True,
            "max_abs_e3u_0_minus_h_u_over_wet_faces": float(
                np.max(np.abs((e3u_0 - h_u_lego)[lego_u]))),
            "max_abs_hu_0_minus_H_u": float(np.max(np.abs(hu_0_nemo - H_u_lego))),
            "max_abs_weight_ratio_difference": float(np.max(np.abs(
                np.divide(e3u_0, hu_0_nemo[..., None],
                          out=np.zeros_like(e3u_0), where=hu_0_nemo[..., None] > 0)
                - np.divide(h_u_lego, H_u_lego[..., None],
                            out=np.zeros_like(h_u_lego), where=H_u_lego[..., None] > 0)
            )[lego_u])),
        },
        "faces": faces,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {args.out}")
    print("controls:", json.dumps(payload["controls"], indent=1))
    for entry in faces:
        marks = "".join(
            "W" if entry["levels"][str(k)]["lego_u_active"] else "."
            for k in range(args.k_lo, args.k_hi + 1))
        print(f"  gate face {entry['gate_face']:3d} (model {entry['model_u_face']:3d}) "
              f"x={entry['x_km_left_T']:6.1f} km  bottom_k L/R "
              f"{entry['left_T_bottom_k']:3d}/{entry['right_T_bottom_k']:3d}  "
              f"bathy {entry['left_bathy_m']:7.1f}/{entry['right_bathy_m']:7.1f}  "
              f"hu_0 {entry['hu_0_nemo_m']:9.3f}  k{args.k_lo}..{args.k_hi}=[{marks}]")


# ------------------------------------------------------------- variance ----
def command_variance(args) -> None:
    """Volume-weighted domain tracer variance, legoESM vs NEMO, kt=1..N.

    The census gap is a statistic of the FINAL state; this command asks
    whether the underlying mixing-rate difference is already resolvable in the
    first minute of model time, where the two trajectories are still
    bit-close, which discriminates a per-step operator gap from a slowly
    accumulated statistical difference.
    """
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    set_policy(PrecisionPolicy.fp64())
    card = STATS.build_nemo_testcase_card(CASE)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks())
    nlev = card.recipe.z_coord.n_levels
    state = card.recipe.initial_state
    rows = []
    for kt in range(1, args.max_step + 1):
        path = Path(args.oracle_dir) / f"oracle_step_entry_kt{kt:08d}.bin"
        require(path.is_file(), f"missing {path}")
        oracle = STATS.read_entry(path, CASE)
        require(oracle["step"] == kt, f"{path}: step mismatch")
        pair = {}
        for name, fields in (
            ("NEMO", {"T": np.asarray(oracle["T"])[..., :nlev],
                      "S": np.asarray(oracle["S"])[..., :nlev],
                      "ssh": np.asarray(oracle["ssh"])}),
            ("legoESM", {"T": np.asarray(state.T.data),
                         "S": np.asarray(state.S.data),
                         "ssh": np.asarray(state.eta.data)}),
        ):
            _, active, volume, _ = STATS._geometry(CASE, fields["ssh"])
            values = fields["T"][active]
            weights = volume[active]
            total = float(np.sum(weights))
            mean = float(np.sum(weights * values) / total)
            pair[name] = {
                "variance_K2": float(np.sum(weights * (values - mean) ** 2) / total),
                "mean_C": mean,
                "volume_m3": total,
            }
        rows.append({
            "kt": kt,
            "nemo_variance_K2": pair["NEMO"]["variance_K2"],
            "lego_variance_K2": pair["legoESM"]["variance_K2"],
            "variance_delta_K2": pair["legoESM"]["variance_K2"] - pair["NEMO"]["variance_K2"],
            "variance_ratio": pair["legoESM"]["variance_K2"] / pair["NEMO"]["variance_K2"],
            "nemo_mean_C": pair["NEMO"]["mean_C"],
            "lego_mean_C": pair["legoESM"]["mean_C"],
        })
        if kt < args.max_step:
            state = model.step(state, dt=card.dt_s)

    payload = {
        "case": CASE,
        "git_sha": git_sha(),
        "precision": "fp64",
        "oracle_dir": str(args.oracle_dir),
        "reduction": (
            "volume-weighted variance of T over every wet cell, each arm on "
            "its OWN ssh-derived live partial-cell volume; NEMO frame is the "
            "Nbb step entry at kt, legoESM the pre-step prognostic at kt"),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {args.out}")
    for entry in rows:
        if entry["kt"] in (1, 2, 3, 5, 10, 20, 30, 40, 50, 60) or entry["kt"] == args.max_step:
            print(f"  kt={entry['kt']:3d}  NEMO {entry['nemo_variance_K2']:.15f}  "
                  f"legoESM {entry['lego_variance_K2']:.15f}  "
                  f"delta {entry['variance_delta_K2']:+.6e}  "
                  f"ratio {entry['variance_ratio']:.12f}")


def command_map(args) -> None:
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_map = sub.add_parser("map", help="census anatomy: where, when, what")
    p_map.add_argument("--lego-root", type=Path, default=DEFAULT_LEGO_ROOT)
    p_map.add_argument("--out", type=Path, default=DEFAULT_OUT / "census_map.json")
    p_map.add_argument("--expect-scorer", type=Path, default=COMMITTED_SCORE)
    p_map.set_defaults(func=command_map)

    p_faces = sub.add_parser("faces", help="k=24 vs k=25 face-family enumeration")
    p_faces.add_argument(
        "--mesh-mask", type=Path,
        default=Path("/data/abyssal/dbalwada/nemo-testcases-l1/overflow_zps/mesh_mask.nc"))
    p_faces.add_argument("--face-lo", type=int, default=16)
    p_faces.add_argument("--face-hi", type=int, default=30)
    p_faces.add_argument("--k-lo", type=int, default=22)
    p_faces.add_argument("--k-hi", type=int, default=27)
    p_faces.add_argument("--out", type=Path, default=DEFAULT_OUT / "faces.json")
    p_faces.set_defaults(func=command_faces)

    p_var = sub.add_parser("variance", help="tracer-variance trajectory kt=1..N")
    p_var.add_argument("--oracle-dir", type=Path, default=KT60_ROOT)
    p_var.add_argument("--max-step", type=int, default=60)
    p_var.add_argument("--out", type=Path, default=DEFAULT_OUT / "variance.json")
    p_var.set_defaults(func=command_variance)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
