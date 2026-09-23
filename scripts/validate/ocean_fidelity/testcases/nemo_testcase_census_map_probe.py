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
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
CASE = "OVERFLOW-zps"
DEFAULT_OUT = Path("/data/abyssal/dbalwada/nemo-testcases-l1/census_map")
COMMITTED_SCORE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/stage_mean_weights/stats/"
    "overflow_statistics.json"
)
DEFAULT_LEGO_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l1/stage_mean_weights/stats/legoesm"
)
OWNER_PREREG_COMMIT = "11d98a704"
W_METRIC_PREREG_COMMIT = "1b72ea5660"


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
    """HEAD, with a loud ``-dirty+<n>`` suffix when the tree is not clean.

    A stamp that names a commit the artifact cannot be reproduced from is
    worse than no stamp: an earlier run of this probe carried four fields that
    did not exist at the commit it named.
    """
    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return head if not dirty else f"{head}-dirty+{len(dirty) // 2}"


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


# ------------------------------------------------------- census owner ----
def _public_state(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _state_fields(state) -> dict:
    return {
        "T": np.asarray(state.T.data),
        "S": np.asarray(state.S.data),
        "u": np.asarray(state.u.data),
        "v": np.asarray(state.v.data),
        "ssh": np.asarray(state.eta.data),
    }


def _distance(left: dict, right: dict) -> float:
    return float(np.max(np.abs(
        np.asarray(left["census"]) - np.asarray(right["census"]))))


def _bbl_geometries(card):
    from legoesm.ocean.physics.bbl_adv import (
        bbl_static_geometry,
        nemo_bbl_static_geometry,
    )

    z = card.recipe.z_coord
    legacy = bbl_static_geometry(z.h_partial, card.recipe.land_mask)
    require(z.nemo_gdept_0 is not None, "card lacks NEMO gdept_0")
    require(z.nemo_bbl_e3u_0 is not None, "card lacks NEMO BBL e3u_0")
    require(z.nemo_bbl_e3v_0 is not None, "card lacks NEMO BBL e3v_0")
    reference = nemo_bbl_static_geometry(
        z.h_partial, card.recipe.land_mask, z.nemo_gdept_0,
        z.nemo_bbl_e3u_0, z.nemo_bbl_e3v_0)
    return legacy, reference


def _bbl_scaling_at_state(card, fields: dict, label: str) -> dict:
    from legoesm.ocean.physics.bbl_adv import (
        apply_bbl_adv_tendency,
        bbl_transports,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    legacy, reference = _bbl_geometries(card)
    grid = card.recipe.grid
    T = jnp.asarray(fields["T"])
    S = jnp.asarray(fields["S"])
    ssh = jnp.asarray(fields["ssh"])
    h = compute_layer_thickness(
        ssh, card.recipe.initial_state.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m)
    legacy_depth_3d = jnp.cumsum(h, axis=-1) - 0.5 * h
    legacy_depth = jnp.take_along_axis(
        legacy_depth_3d, legacy.bot_k[..., None], axis=-1)[..., 0]
    stretch = jnp.sum(h, axis=-1) / jnp.maximum(
        jnp.sum(reference.h_ref, axis=-1), 1.0e-10)
    reference_depth = reference.dep_bot * stretch
    dy_u = jnp.asarray(grid.dy_u)[:, 1:-1]
    dx_v = jnp.asarray(grid.dx_v)[1:-1, :]
    kwargs = dict(gamma_s=card.bbl_gamma_s, rho_0=card.recipe.model_config.rho_0)
    lu, lv = bbl_transports(
        T, S, legacy, dy_u, dx_v, bottom_depth_m=legacy_depth, **kwargs)
    nu, nv = bbl_transports(
        T, S, reference, dy_u, dx_v, bottom_depth_m=reference_depth, **kwargs)
    zero = jnp.zeros_like(T)
    ldT, _ = apply_bbl_adv_tendency(
        zero, jnp.zeros_like(S), T, S, h, jnp.asarray(grid.area_T), legacy,
        lu, lv, nlev=T.shape[-1])
    ndT, _ = apply_bbl_adv_tendency(
        zero, jnp.zeros_like(S), T, S, h, jnp.asarray(grid.area_T), reference,
        nu, nv, nlev=T.shape[-1])
    active_l = np.asarray(legacy.u_active) > 0.5
    active_n = np.asarray(reference.u_active) > 0.5
    shared = active_l & active_n
    lu_np, nu_np = np.asarray(lu), np.asarray(nu)
    correlation = None
    if np.count_nonzero(shared) > 1:
        correlation = float(np.corrcoef(lu_np[shared], nu_np[shared])[0, 1])
    dT = np.asarray(ndT - ldT)
    active = np.asarray(card.recipe.z_coord.is_active)
    return {
        "label": label,
        "legacy_active_u_faces": int(np.count_nonzero(active_l)),
        "nemo_active_u_faces": int(np.count_nonzero(active_n)),
        "legacy_only_u_faces": int(np.count_nonzero(active_l & ~active_n)),
        "shared_u_faces": int(np.count_nonzero(shared)),
        "legacy_u_transport_linf_m3_s": float(np.max(np.abs(lu_np))),
        "nemo_u_transport_linf_m3_s": float(np.max(np.abs(nu_np))),
        "u_transport_difference_linf_m3_s": float(np.max(np.abs(nu_np - lu_np))),
        "shared_u_transport_correlation": correlation,
        "legacy_abs_u_transport_sum_m3_s": float(np.sum(np.abs(lu_np))),
        "nemo_abs_u_transport_sum_m3_s": float(np.sum(np.abs(nu_np))),
        "T_tendency_difference_linf_K_s": float(np.max(np.abs(dT[active]))),
        "T_tendency_difference_rms_K_s": float(np.sqrt(np.mean(dT[active] ** 2))),
        "legacy_bottom_depth_range_m": [
            float(np.min(np.asarray(legacy_depth)[active.any(axis=-1)])),
            float(np.max(np.asarray(legacy_depth)[active.any(axis=-1)])),
        ],
        "nemo_bottom_depth_range_m": [
            float(np.min(np.asarray(reference_depth)[active.any(axis=-1)])),
            float(np.max(np.asarray(reference_depth)[active.any(axis=-1)])),
        ],
    }


def command_bbl_scaling(args) -> None:
    """Source-operand scaling on already-certified midpoint/final states."""
    from legoesm.core.precision import PrecisionPolicy, set_policy

    set_policy(PrecisionPolicy.fp64())
    card = STATS.build_nemo_testcase_card(CASE)
    states = STATS.load_legoesm_states(CASE, "fp64", args.lego_root)[0]
    rows = []
    for time_s, raw in states.items():
        fields = STATS.mapped_fields(raw, "L64")
        rows.append(_bbl_scaling_at_state(card, fields, f"completed_time_{time_s}s"))
    legacy, reference = _bbl_geometries(card)
    bottom = np.asarray(card.recipe.z_coord.bottom_level)
    same_bottom = bottom[:, :-1] == bottom[:, 1:]
    payload = {
        "case": CASE,
        "git_sha": git_sha(),
        "preregistration_commit": OWNER_PREREG_COMMIT,
        "precision": "fp64",
        "reference": "NEMO 5.0.2 trabbl.F90:507-533,342-353",
        "geometry": {
            "legacy_active_u_faces": int(np.count_nonzero(legacy.u_active)),
            "nemo_active_u_faces": int(np.count_nonzero(reference.u_active)),
            "legacy_active_on_nemo_same_bottom_level": int(np.count_nonzero(
                (np.asarray(legacy.u_active) > 0.5) & same_bottom)),
            "max_abs_e3u_bbl_difference_m": float(np.max(np.abs(
                np.asarray(legacy.e3u_bbl) - np.asarray(reference.e3u_bbl)))),
        },
        "states_npz": str(args.lego_root / "overflow_zps/fp64/states.npz"),
        "states_sha256": sha256(args.lego_root / "overflow_zps/fp64/states.npz"),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def command_aimp_scaling(args) -> None:
    """Scale Arm 2 on saved live states before any full-duration run."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _nemo_ws_qco_stage_faces,
        _nemo_ws_stage_transport,
    )
    from legoesm.ocean.physics.vertical_mixing import nemo_e3w_kmm
    from legoesm.ocean.vertical import compute_layer_thickness

    set_policy(PrecisionPolicy.fp64())
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = STATS.build_nemo_testcase_card(CASE)
    z = card.recipe.z_coord
    require(z.nemo_gdept_0 is not None,
            "OVERFLOW card lacks the pinned NEMO gdept_0 ladder")
    states = STATS.load_legoesm_states(CASE, "fp64", args.lego_root)[0]
    grid = card.recipe.grid
    u_mask, v_mask = compute_face_masks_3d(z.is_active, grid)
    u_mask = u_mask.astype(jnp.float64)
    v_mask = v_mask.astype(jnp.float64)
    rows = []
    for time_s, raw in states.items():
        # The saved legoESM state retains its redundant periodic U face; the
        # scorer's mapped frame intentionally removes it.  This replay calls
        # the executing transport helper, so it must use the native state.
        eta = jnp.asarray(raw["ssh"])
        h = compute_layer_thickness(
            eta, card.recipe.initial_state.H_bathy.data, z,
            min_water_column_m=card.recipe.model_config.min_water_column_m)
        h_ref = compute_layer_thickness(
            jnp.zeros_like(eta), card.recipe.initial_state.H_bathy.data, z,
            min_water_column_m=card.recipe.model_config.min_water_column_m)
        hu, hv, _, _ = _nemo_ws_qco_stage_faces(
            eta, h_ref, u_mask, v_mask, grid)
        u = jnp.asarray(raw["u"])
        v = jnp.asarray(raw["v"])
        Hu = jnp.sum(hu * u * u_mask, axis=-1)
        Hv = jnp.sum(hv * v * v_mask, axis=-1)
        common = dict(
            eta_stage=eta, h_ref=h_ref, Hu_avg=Hu, Hv_avg=Hv,
            u_mask_3d=u_mask, v_mask_3d=v_mask, grid=grid, z_coord=z,
            config=card.recipe.model_config, dt=card.dt_s)
        legacy = _nemo_ws_stage_transport(
            (u, v), h, 2, legacy_aimp_midpoint_w_metric=True, **common)
        reference = _nemo_ws_stage_transport(
            (u, v), h, 2, legacy_aimp_midpoint_w_metric=False, **common)
        stretch = 1.0 + eta / jnp.maximum(jnp.sum(h_ref, axis=-1), 1.0e-10)
        ref_int = nemo_e3w_kmm(z, h, stretch)
        midpoint_int = 0.5 * (h[..., :-1] + h[..., 1:])
        wet_int = np.asarray(z.is_active[..., :-1] & z.is_active[..., 1:])
        old_wi = np.asarray(legacy[6])
        new_wi = np.asarray(reference[6])
        total_w = np.asarray(reference[2] + reference[6])
        fraction = np.divide(
            new_wi, total_w, out=np.zeros_like(new_wi), where=total_w != 0.0)
        rows.append({
            "physical_time_s": int(time_s),
            "wet_interface_e3w_difference_linf_m": float(np.max(np.abs(
                np.asarray(ref_int)[wet_int] - np.asarray(midpoint_int)[wet_int]))),
            "wet_interface_reference_to_midpoint_ratio_range": [
                float(np.min(np.asarray(ref_int)[wet_int] /
                             np.asarray(midpoint_int)[wet_int])),
                float(np.max(np.asarray(ref_int)[wet_int] /
                             np.asarray(midpoint_int)[wet_int])),
            ],
            "legacy_implicit_w_linf_m_s": float(np.max(np.abs(old_wi))),
            "nemo_metric_implicit_w_linf_m_s": float(np.max(np.abs(new_wi))),
            "implicit_w_difference_linf_m_s": float(np.max(np.abs(new_wi - old_wi))),
            "nemo_metric_implicit_fraction_linf": float(np.max(np.abs(fraction))),
            "implicit_w_changed_points": int(np.count_nonzero(new_wi != old_wi)),
        })
    payload = {
        "format": "nemo-testcase-census-aimp-scaling-v1",
        "case": CASE,
        "git_sha": git_sha(),
        "preregistration_commit": W_METRIC_PREREG_COMMIT,
        "precision": "fp64",
        "backend": jax.default_backend(),
        "reference": (
            "usrdef_zgr.F90:157-168; domzgr_substitute.h90:131; "
            "sshwzv.F90:812-843"),
        "states_npz": str(args.lego_root / "overflow_zps/fp64/states.npz"),
        "states_sha256": sha256(args.lego_root / "overflow_zps/fp64/states.npz"),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _model_state_from_saved(card, raw: dict):
    """Restore one native legoESM snapshot without changing its grid frame."""
    state = card.recipe.initial_state
    return state._replace(
        T=state.T.replace(data=jnp.asarray(raw["T"])),
        S=state.S.replace(data=jnp.asarray(raw["S"])),
        u=state.u.replace(data=jnp.asarray(raw["u"])),
        v=state.v.replace(data=jnp.asarray(raw["v"])),
        eta=state.eta.replace(data=jnp.asarray(raw["ssh"])),
    )


def command_zdf_scaling(args) -> None:
    """One-step source-exact ZDF W-divisor movement on saved live states."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.physics.vertical_mixing import build_dz_half, nemo_e3w_kmm
    from legoesm.ocean.vertical import compute_layer_thickness

    set_policy(PrecisionPolicy.fp64())
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = STATS.build_nemo_testcase_card(CASE)
    states = STATS.load_legoesm_states(CASE, "fp64", args.lego_root)[0]
    baseline = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            legacy_zdf_midpoint_w_metric=True))
    arm = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    active = np.asarray(card.recipe.z_coord.is_active)
    slope = np.asarray(card.recipe.initial_state.H_bathy.data)
    slope = active & ((slope > 500.0) & (slope < 2000.0))[..., None]
    rows = []
    for time_s, raw in states.items():
        entering = _model_state_from_saved(card, raw)
        base_after = baseline.step(entering, dt=card.dt_s)
        arm_after = arm.step(entering, dt=card.dt_s)
        delta = {
            name: np.asarray(getattr(arm_after, attr).data)
            - np.asarray(getattr(base_after, attr).data)
            for name, attr in (("T", "T"), ("S", "S"), ("u", "u"),
                               ("v", "v"), ("ssh", "eta"))
        }
        eta = jnp.asarray(raw["ssh"])
        h = compute_layer_thickness(
            eta, entering.H_bathy.data, card.recipe.z_coord,
            min_water_column_m=card.recipe.model_config.min_water_column_m)
        stretch = 1.0 + eta / jnp.maximum(
            jnp.sum(card.recipe.z_coord.h_partial, axis=-1), 1.0e-10)
        legacy_e3w = build_dz_half(h)
        reference_e3w = nemo_e3w_kmm(card.recipe.z_coord, h, stretch)
        wet_int = np.asarray(active[..., :-1] & active[..., 1:])
        rows.append({
            "physical_time_s": int(time_s),
            "wet_cell_e3w_difference_linf_m": float(np.max(np.abs(
                np.asarray(reference_e3w)[wet_int]
                - np.asarray(legacy_e3w)[wet_int]))),
            "T_after_difference_linf_K": float(np.max(np.abs(delta["T"][active]))),
            "T_slope_difference_linf_K": float(np.max(np.abs(delta["T"][slope]))),
            "u_after_difference_linf_m_s": float(np.max(np.abs(delta["u"]))),
            "v_after_difference_linf_m_s": float(np.max(np.abs(delta["v"]))),
            "ssh_after_difference_linf_m": float(np.max(np.abs(delta["ssh"]))),
        })
    payload = {
        "format": "nemo-testcase-census-zdf-scaling-v1",
        "case": CASE,
        "git_sha": git_sha(),
        "preregistration_commit": W_METRIC_PREREG_COMMIT,
        "precision": "fp64",
        "backend": jax.default_backend(),
        "reference": (
            "dynzdf.F90:180-195; domzgr_substitute.h90:131-133; "
            "usrdef_zgr.F90:157-168"),
        "one_variable": "legacy_zdf_midpoint_w_metric=False",
        "states_npz": str(args.lego_root / "overflow_zps/fp64/states.npz"),
        "states_sha256": sha256(args.lego_root / "overflow_zps/fp64/states.npz"),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def _effect_row(card, entering, full_after, alternate_after, term: str) -> dict:
    full = arm_state("L64", _state_fields(full_after))
    T_delta = np.asarray(full_after.T.data) - np.asarray(alternate_after.T.data)
    active = full["_active"]
    select = full["_select"]
    volume = full["_volume"]
    census_distance = None
    census_status = "MEASURED"
    census_reason = None
    try:
        alternate = arm_state("L64", _state_fields(alternate_after))
        census_distance = _distance(full, alternate)
    except STATS.StatisticalError as error:
        # A private diagnostic ablation is allowed to reveal why it is not a
        # valid model arm.  Do not clip it into the scorer's temperature
        # envelope and manufacture a census: retain its raw T/heat leverage,
        # make the classification unavailable, and quote the hard-guard error.
        census_status = "GROSS-EXCURSION"
        census_reason = str(error)
    return {
        "term": term,
        "census_status": census_status,
        "census_reason": census_reason,
        "census_distance_full_vs_ablation": census_distance,
        "T_linf_K": float(np.max(np.abs(T_delta[active]))),
        "T_rms_K": float(np.sqrt(np.mean(T_delta[active] ** 2))),
        "slope_heat_difference_K_m3": float(np.sum(T_delta[select] * volume[select])),
        "slope_abs_heat_difference_K_m3": float(np.sum(
            np.abs(T_delta[select]) * volume[select])),
    }


def command_budget(args) -> None:
    """Cadenced census trajectory plus same-entry one-step term ablations."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = STATS.build_nemo_testcase_card(CASE)
    term_hooks = {
        "bbl": _NEMOWSRK3TestHooks(disable_bbl=True),
        "vertical_transport": _NEMOWSRK3TestHooks(
            disable_tracer_vertical_transport=True),
        "fct_two_step_predictor": _NEMOWSRK3TestHooks(
            two_step_fct_predictor=False),
        "legacy_bbl_partial_geometry": _NEMOWSRK3TestHooks(
            legacy_bbl_partial_geometry=True),
    }
    registered = set(term_hooks)
    measured = set(term_hooks)
    if args.plant_unregistered:
        measured.add("PLANTED_UNREGISTERED_TERM")
    require(measured == registered, f"unregistered budget terms: {sorted(measured-registered)}")
    faithful = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks())
    models = {
        name: LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hooks)
        for name, hooks in term_hooks.items()
    }
    state = card.recipe.initial_state
    if args.plant_census:
        planted = _state_fields(state)
        base = arm_state("L64", planted)
        T = planted["T"].copy()
        index = tuple(np.argwhere(base["_select"] & (T >= 18.0))[0])
        T[index] = 17.0
        planted["T"] = T
        changed = arm_state("L64", planted)
        require(
            _distance(base, changed) <= 1.0e-14,
            f"PLANTED census excursion fired at {index}: "
            f"distance={_distance(base, changed):.6e}")

    n_steps = int(STATS.CASES[CASE]["n_steps"])
    samples = set(range(0, n_steps + 1, args.cadence)) | {3059, n_steps}
    nemo = STATS.load_nemo_states(
        CASE, Path(STATS.CASES[CASE]["baseline"]), baseline=True)
    nemo_by_completed = {
        int(time_s // card.dt_s): arm_state(
            "N2", STATS.mapped_fields(fields, "N2"))
        for time_s, fields in nemo.items()
    }
    rows = []
    started = time.perf_counter()
    for completed in range(n_steps + 1):
        sampled = completed in samples
        if sampled:
            current = arm_state("L64", _state_fields(state))
            row = {
                "completed_step": completed,
                "physical_time_s": completed * card.dt_s,
                "legoesm": _public_state(current),
                "nemo": None,
                "legoesm_minus_nemo_census": None,
                "one_step_effects": [],
            }
            if completed in nemo_by_completed:
                oracle = nemo_by_completed[completed]
                row["nemo"] = _public_state(oracle)
                row["legoesm_minus_nemo_census"] = _distance(current, oracle)
        if completed == n_steps:
            if sampled:
                rows.append(row)
            break
        next_state = faithful.step(state, dt=card.dt_s)
        if sampled:
            for term, model in models.items():
                alternate = model.step(state, dt=card.dt_s)
                row["one_step_effects"].append(
                    _effect_row(card, state, next_state, alternate, term))
            rows.append(row)
        state = next_state
    payload = {
        "format": "nemo-testcase-l1-census-budget-v1",
        "case": CASE,
        "git_sha": git_sha(),
        "preregistration_commit": OWNER_PREREG_COMMIT,
        "precision": "fp64",
        "backend": jax.default_backend(),
        "cadence_steps": args.cadence,
        "frame": (
            "legoESM pre-step prognostic at completed_step; NEMO Nbb entry "
            "at completed 0/3059 and tn restart at completed 6120; same-entry "
            "one-step full-minus-private-ablation on scorer wet-mask/slope frame"),
        "term_registry": sorted(registered),
        "controls": {
            "plant_census": args.plant_census,
            "plant_unregistered": args.plant_unregistered,
        },
        "oracle_artifacts": {
            str(path): sha256(path)
            for path in (
                Path(STATS.CASES[CASE]["baseline"]) / "oracle_step_entry_kt00000001.bin",
                Path(STATS.CASES[CASE]["baseline"]) / "oracle_step_entry_kt00003060.bin",
                Path(STATS.CASES[CASE]["baseline"]) / str(STATS.CASES[CASE]["restart"]),
                Path(STATS.CASES[CASE]["baseline"]) / "namelist_cfg",
            )
        },
        "wall_time_s": time.perf_counter() - started,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    for row in rows:
        if row["completed_step"] in (0, 600, 1200, 2400, 3059, 3600, 4800, 6000, 6120):
            print(
                f"  kt={row['completed_step']:4d} census="
                f"{row['legoesm']['census']} N-distance="
                f"{row['legoesm_minus_nemo_census']}")


def command_run_arm(args) -> None:
    """Full-duration private owner arm in scorer-compatible format."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    policy = PrecisionPolicy.fp64() if args.precision == "fp64" else PrecisionPolicy.fp32()
    set_policy(policy)
    require(get_policy() == policy, f"failed to set {args.precision}")
    require(bool(jax.config.jax_enable_x64) == (args.precision == "fp64"),
            "JAX x64/precision mismatch")
    card = STATS.build_nemo_testcase_card(CASE)
    hooks = {
        "bbl-reference": _NEMOWSRK3TestHooks(),
        "aimp-e3w": _NEMOWSRK3TestHooks(
            legacy_zdf_midpoint_w_metric=True),
        "zdf-e3w": _NEMOWSRK3TestHooks(
            legacy_aimp_midpoint_w_metric=True),
        "combined-e3w": _NEMOWSRK3TestHooks(),
    }[args.arm]
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks)
    state = card.recipe.initial_state
    samples = set(STATS.sample_completed_steps(CASE))
    captured = {}
    started = time.perf_counter()
    for completed in range(int(STATS.CASES[CASE]["n_steps"]) + 1):
        if completed in samples:
            captured[completed] = {
                name: np.array(value, copy=True)
                for name, value in _state_fields(state).items()
            }
        if completed == int(STATS.CASES[CASE]["n_steps"]):
            break
        state = model.step(state, dt=card.dt_s)
        require(all(np.all(np.isfinite(v)) for v in _state_fields(state).values()),
                f"nonfinite arm state at completed step {completed+1}")
        if (completed + 1) % 612 == 0:
            print(f"  {args.precision}: completed {completed+1}/6120", flush=True)
    arrays = {
        f"step_{completed}_{name}": value
        for completed, fields in captured.items()
        for name, value in fields.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "states.npz"
    np.savez_compressed(state_path, **arrays)
    expected_dtype = "float64" if args.precision == "fp64" else "float32"
    state_dtypes = {name: str(value.dtype) for name, value in _state_fields(state).items()}
    geometry_dtypes = {
        name: str(np.asarray(getattr(card.recipe.z_coord, name)).dtype)
        for name in ("t_depth_ref", "dz_ref", "z_full_ref", "z_half_ref", "h_partial")
    }
    require(set(state_dtypes.values()) == {expected_dtype}, f"state dtypes {state_dtypes}")
    require(set(geometry_dtypes.values()) == {expected_dtype},
            f"geometry dtypes {geometry_dtypes}")
    metadata = {
        "format": "nemo-testcase-l1-full-state-v1",
        "preregistration_commit": (
            OWNER_PREREG_COMMIT if args.arm == "bbl-reference"
            else W_METRIC_PREREG_COMMIT),
        "git_sha": git_sha(),
        "case": CASE,
        "precision": args.precision,
        "precision_policy": repr(policy),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "arm": args.arm,
        "completed_steps": sorted(samples),
        "physical_times_s": [value * card.dt_s for value in sorted(samples)],
        "wall_time_s": time.perf_counter() - started,
        "state_dtypes": state_dtypes,
        "geometry_dtypes": geometry_dtypes,
        "states_sha256": sha256(state_path),
        "diagnostic_check_every_step": True,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))




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

    # The ACTUAL open item 3: NEMO's stage depth mean is REFERENCE-weighted
    # (stprk3_stg.F90:440, e3u_0 / r1_hu_0) while legoESM's is LIVE-weighted
    # (_replace_stage_mean, h_u_pre / H_u_pre with
    # h_u_pre = min_cell_to_uface(h_k_pre)).  Because that min is taken over
    # two columns whose free-surface Jacobians differ, the live weights are NOT
    # a uniform rescale of the reference ones and the difference is not
    # algebraically zero -- measure it on real states rather than assert it.
    live = _live_stage_mean_weights(card, lego_u, args.live_steps)
    # mesh_mask stores e3u_0 at DRY levels too, so mask before forming the
    # weight -- an unmasked reference weight makes every dry level read as a
    # 1/nlev difference and swamps the number this measures.
    reference_weight = np.divide(
        e3u_0 * lego_u, hu_0_nemo[..., None],
        out=np.zeros_like(e3u_0), where=hu_0_nemo[..., None] > 0)

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
            "max_abs_reference_weight_ratio_difference": float(np.max(np.abs(
                np.divide(e3u_0, hu_0_nemo[..., None],
                          out=np.zeros_like(e3u_0), where=hu_0_nemo[..., None] > 0)
                - np.divide(h_u_lego, H_u_lego[..., None],
                            out=np.zeros_like(h_u_lego), where=H_u_lego[..., None] > 0)
            )[lego_u])),
        },
        "live_vs_reference_stage_mean_weights": [
            {
                "kt": entry["kt"],
                "max_abs_weight_difference": float(np.max(
                    np.abs(entry["weight"] - reference_weight)[lego_u])),
                "wet_levels_differing_above_1e_12": int(np.count_nonzero(
                    (np.abs(entry["weight"] - reference_weight) > 1.0e-12) & lego_u)),
                "wet_levels_differing_at_all": int(np.count_nonzero(
                    (np.abs(entry["weight"] - reference_weight) > 0.0) & lego_u)),
                "max_abs_at_injection_faces_20_21_22": float(np.max([
                    np.max(np.abs(entry["weight"] - reference_weight)[row, f][
                        lego_u[row, f]]) for f in (20, 21, 22)])),
            }
            for entry in live
        ],
        "faces": faces,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"wrote {args.out}")
    print("controls:", json.dumps(payload["controls"], indent=1))
    print("live vs reference stage-mean weights (the real open item 3):")
    for entry in payload["live_vs_reference_stage_mean_weights"]:
        print(f"  kt={entry['kt']:3d}  max |live-ref| {entry['max_abs_weight_difference']:.6e}"
              f"  wet levels differing >1e-12 "
              f"{entry['wet_levels_differing_above_1e_12']:5d}"
              f"  at injection faces 20/21/22 "
              f"{entry['max_abs_at_injection_faces_20_21_22']:.6e}")
    for entry in faces:
        marks = "".join(
            "W" if entry["levels"][str(k)]["lego_u_active"] else "."
            for k in range(args.k_lo, args.k_hi + 1))
        print(f"  gate face {entry['gate_face']:3d} (model {entry['model_u_face']:3d}) "
              f"x={entry['x_km_left_T']:6.1f} km  bottom_k L/R "
              f"{entry['left_T_bottom_k']:3d}/{entry['right_T_bottom_k']:3d}  "
              f"bathy {entry['left_bathy_m']:7.1f}/{entry['right_bathy_m']:7.1f}  "
              f"hu_0 {entry['hu_0_nemo_m']:9.3f}  k{args.k_lo}..{args.k_hi}=[{marks}]")


def _live_stage_mean_weights(card, lego_u: np.ndarray, steps: int) -> list[dict]:
    """legoESM's per-level stage depth-mean weight ``h_u_pre / H_u_pre``.

    Built exactly as ``ocean_model_latlon_cgrid.py:4255-4271`` does: the live
    layer thickness from the step-entry ssh, ``min_cell_to_uface``, and the
    column sum of the same array.  Returned in the gate's u frame.
    """
    from legoesm.ocean.dynamics.latlon_cgrid_operators import min_cell_to_uface
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.vertical import compute_layer_thickness

    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks())
    state = card.recipe.initial_state
    out = []
    for kt in range(1, steps + 1):
        h_k = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, card.recipe.z_coord,
            min_water_column_m=card.recipe.model_config.min_water_column_m)
        h_u = np.asarray(min_cell_to_uface(h_k), dtype=np.float64)[:, 1:, :]
        h_u = h_u * lego_u
        H_u = np.sum(h_u, axis=-1)
        weight = np.divide(
            h_u, H_u[..., None], out=np.zeros_like(h_u), where=H_u[..., None] > 0)
        out.append({"kt": kt, "weight": weight})
        state = model.step(state, dt=card.dt_s)
    return out


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


# ------------------------------------------------------------ chaos null ----
def command_chaos_null(args) -> None:
    """Seed one wet face at kt=0 and score the full run against the committed run.

    The precision arm is not a null for the census row: it saturates at a
    normalized temperature L-infinity of 0.054 while every operator pair in the
    scorer sits at 0.32-0.36, and its perturbation is RANDOM.  This runs the
    certified card unchanged except for one number -- the initial ``u`` at the
    ``k=24`` injection face itself -- so the seed carries the same field, cell
    and provenance as the injection whose ownership is in question and differs
    from it only in amplitude.  Scored against the COMMITTED unperturbed fp64
    states with the first round's reductions plus the scorer's own normalized
    L-infinity (scaled by ``N2``, exactly as ``_field_linf`` does), so every
    number is directly comparable to a scorer row.
    """
    import jax

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    set_policy(PrecisionPolicy.fp64())
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    card = STATS.build_nemo_testcase_card(CASE)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks())
    base = card.recipe.initial_state
    masks = STATS.expected_masks(card)
    row, face, level = args.row, args.face, args.level

    # CONTROLS, before the run: the seeded face must be WET at that level and
    # the seed must be the ONLY difference from the certified initial state.
    require(bool(masks["u"][row, face - 1, level]), "seeded face is dry at that level")
    u0 = np.asarray(base.u.data, dtype=np.float64)
    seeded = u0.copy()
    seeded[row, face, level] += args.amplitude
    difference = seeded - u0
    require(int(np.count_nonzero(difference)) == 1, "seed touched more than one face")
    require(
        float(np.max(np.abs(difference))) == float(args.amplitude),
        "seed amplitude did not land",
    )
    state = base._replace(u=base.u.replace(data=jnp.asarray(seeded)))

    samples = set(STATS.sample_completed_steps(CASE))
    n_steps = int(STATS.CASES[CASE]["n_steps"])
    captured: dict[int, dict] = {}
    started = time.perf_counter()
    for completed in range(n_steps + 1):
        if completed in samples:
            captured[completed] = {
                name: np.array(values, copy=True)
                for name, values in STATS._state_arrays(state).items()
            }
        if completed == n_steps:
            break
        state = model.step(state, dt=card.dt_s)
    require(set(captured) == samples, "milestone capture incomplete")
    wall_s = time.perf_counter() - started

    dt_s = float(STATS.CASES[CASE]["dt_s"])
    arm = {int(completed * dt_s): fields for completed, fields in captured.items()}
    reference = STATS.load_legoesm_states(CASE, "fp64", args.lego_root)[0]
    nemo = STATS.load_nemo_states(CASE, Path(STATS.CASES[CASE]["baseline"]), baseline=True)
    require(sorted(arm) == sorted(reference) == sorted(nemo), "registered times differ")
    times = sorted(arm)

    states = {}
    for name, source in (("seed", arm), ("L64", reference)):
        states[name] = {}
        for time_s in times:
            fields = STATS.mapped_fields(source[int(time_s)], "L64")
            states[name][int(time_s)] = arm_state("L64", fields)

    final = int(times[-1])
    census = float(np.max(np.abs(
        np.asarray(states["seed"][final]["census"])
        - np.asarray(states["L64"][final]["census"]))))
    # the scorer's normalized L-infinity, N2 as the common scale, last two times
    linf = {}
    for field in ("T", "u"):
        mask = masks[field]
        values = []
        for time_s in times[1:]:
            left = STATS.mapped_fields(arm[int(time_s)], "L64")[field]
            right = STATS.mapped_fields(reference[int(time_s)], "L64")[field]
            scale = max(float(np.max(np.abs(
                STATS.mapped_fields(nemo[int(time_s)], "N2")[field][mask]))), 1.0)
            values.append(float(np.max(np.abs(left[mask] - right[mask]))) / scale)
        linf[field] = max(values)

    payload = {
        "case": CASE,
        "git_sha": git_sha(),
        "precision": "fp64",
        "backend": jax.default_backend(),
        "seed": {
            "field": "u", "row": row, "model_face": face, "level": level,
            "amplitude_m_s": float(args.amplitude),
            "seeded_faces": 1,
        },
        "wall_time_s": wall_s,
        "reference": str(args.lego_root),
        "census_max_abs_distance": census,
        "temperature_linf_normalized": linf["T"],
        "u_linf_normalized": linf["u"],
        "variance_ratio_seed_over_L64": (
            states["seed"][final]["domain_T_variance_K2"]
            / states["L64"][final]["domain_T_variance_K2"]),
        "anomaly_effective_volume_ratio": (
            states["seed"][final]["anomaly_effective_volume_m3"]
            / states["L64"][final]["anomaly_effective_volume_m3"]),
        "anomaly_deficit_relative_delta": (
            states["seed"][final]["anomaly_deficit_K_m3"]
            / states["L64"][final]["anomaly_deficit_K_m3"] - 1.0),
        "census_by_time": {
            str(t): float(np.max(np.abs(
                np.asarray(states["seed"][t]["census"])
                - np.asarray(states["L64"][t]["census"])))) for t in times},
        "seed_census": states["seed"][final]["census"],
        "reference_census": states["L64"][final]["census"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    np.savez_compressed(
        args.out.with_suffix(".states.npz"),
        **{f"step_{completed}_{name}": values
           for completed, fields in captured.items() for name, values in fields.items()})
    print(f"wrote {args.out}")
    for key in (
        "census_max_abs_distance", "temperature_linf_normalized", "u_linf_normalized",
        "variance_ratio_seed_over_L64", "anomaly_effective_volume_ratio",
        "anomaly_deficit_relative_delta",
    ):
        print(f"  {key:38s} {payload[key]!r}")
    print(f"  census_by_time                          {payload['census_by_time']}")


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
    p_faces.add_argument("--live-steps", type=int, default=10)
    p_faces.add_argument("--out", type=Path, default=DEFAULT_OUT / "faces.json")
    p_faces.set_defaults(func=command_faces)

    p_var = sub.add_parser("variance", help="tracer-variance trajectory kt=1..N")
    p_var.add_argument("--oracle-dir", type=Path, default=KT60_ROOT)
    p_var.add_argument("--max-step", type=int, default=60)
    p_var.add_argument("--out", type=Path, default=DEFAULT_OUT / "variance.json")
    p_var.set_defaults(func=command_variance)

    p_null = sub.add_parser("chaos-null", help="seed one face at kt=0, full duration")
    p_null.add_argument("--amplitude", type=float, required=True)
    p_null.add_argument("--row", type=int, default=1)
    p_null.add_argument("--face", type=int, default=21, help="MODEL u-face index")
    p_null.add_argument("--level", type=int, default=24)
    p_null.add_argument("--lego-root", type=Path, default=DEFAULT_LEGO_ROOT)
    p_null.add_argument("--out", type=Path, required=True)
    p_null.set_defaults(func=command_chaos_null)

    p_scale = sub.add_parser(
        "bbl-scaling", help="NEMO-reference versus partial-centroid BBL operands")
    p_scale.add_argument("--lego-root", type=Path, default=DEFAULT_LEGO_ROOT)
    p_scale.add_argument("--out", type=Path, default=DEFAULT_OUT / "bbl_scaling.json")
    p_scale.set_defaults(func=command_bbl_scaling)

    p_aimp_scale = sub.add_parser(
        "aimp-scaling", help="NEMO e3w_0 versus midpoint adaptive-split metric")
    p_aimp_scale.add_argument("--lego-root", type=Path, default=DEFAULT_LEGO_ROOT)
    p_aimp_scale.add_argument(
        "--out", type=Path, default=DEFAULT_OUT / "aimp_scaling.json")
    p_aimp_scale.set_defaults(func=command_aimp_scaling)

    p_zdf_scale = sub.add_parser(
        "zdf-scaling", help="literal versus midpoint ZDF W divisor")
    p_zdf_scale.add_argument("--lego-root", type=Path, default=DEFAULT_LEGO_ROOT)
    p_zdf_scale.add_argument(
        "--out", type=Path, default=DEFAULT_OUT / "zdf_scaling.json")
    p_zdf_scale.set_defaults(func=command_zdf_scaling)

    p_budget = sub.add_parser(
        "budget", help="cadenced slope census plus local term ablations")
    p_budget.add_argument("--cadence", type=int, default=60)
    p_budget.add_argument("--out", type=Path, default=DEFAULT_OUT / "term_budget.json")
    p_budget.add_argument("--plant-census", action="store_true")
    p_budget.add_argument("--plant-unregistered", action="store_true")
    p_budget.set_defaults(func=command_budget)

    p_arm = sub.add_parser(
        "run-arm", help="run the private NEMO-reference BBL geometry arm")
    p_arm.add_argument("--precision", choices=("fp64", "fp32"), required=True)
    p_arm.add_argument(
        "--arm", choices=(
            "bbl-reference", "aimp-e3w", "zdf-e3w", "combined-e3w"),
        default="bbl-reference")
    p_arm.add_argument("--output-dir", type=Path, required=True)
    p_arm.set_defaults(func=command_run_arm)

    args = parser.parse_args()
    if getattr(args, "cadence", 1) <= 0:
        parser.error("--cadence must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
