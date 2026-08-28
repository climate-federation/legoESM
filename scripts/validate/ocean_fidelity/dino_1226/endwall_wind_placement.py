#!/usr/bin/env python
"""Measure DINO/NEMO wind source operands at the two placement entry points.

Pre-registration: ``PREREG_endwall_wind_placement.md``.  The oracle term is
not ``utrd_tau`` (that slot is allocated and emitted but never populated).
It is wired here from NEMO's own dyn_zdf bracket:

    tau_increment = zdf_dump_*1_poststress - zdf_dump_*1_prestress

The legoESM vertical-route operand is reconstructed from the production
``surface_stress_faces`` output as ``rDt*tau/(rho0*dz0)``.  The same A/B
captures the exact ``F_slow`` argument entering the production barotropic
loop, closing the fourth structural premise without re-deriving its
face-depth convention.  The post-barotropic B1 state is diagnostic only.

This probe does not run the alternative implicit placement through the
tridiagonal solve.  It therefore classifies source-operand agreement only,
not the placement response or sea-surface ownership; the registered free-run
arm is required for those questions.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_ROOT = _DIR.parents[3]
for _p in (str(_DIR), str(_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DINO = os.environ.get(
    "DINO_ORACLE_ROOT",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO",
)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")
os.environ.setdefault("DINO_1226_LANE", "d180")
os.environ.setdefault("DINO_1226_IC_STEP", "5760")
os.environ.setdefault("DINO_NEMO_RUN_TWIN_STEP1",
                      os.path.join(_DINO, "RUN_D180_STEP1"))

import dump_lane  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import post_tendency_stage_birth as ptsb  # noqa: E402

KT = 5761
DT = 2700.0
RDT = 2.0 * DT
SOUTH_ROW = 1
PREREG_COMMIT = "25a5afef6dd3f4e50c8b9d7febdbaf69b3b87095"


def _git(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", "-C", str(_ROOT), *args], text=True).strip()


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stats(lego: np.ndarray, nemo: np.ndarray, mask: np.ndarray) -> dict:
    if lego.shape != nemo.shape or mask.shape != lego.shape:
        raise SystemExit(
            f"shape mismatch lego={lego.shape} nemo={nemo.shape} mask={mask.shape}")
    if not (np.isfinite(lego).all() and np.isfinite(nemo).all()):
        raise SystemExit("non-finite operand -- fatal")
    x, y = lego[mask], nemo[mask]
    if x.size < 2:
        raise SystemExit("statistic mask contains fewer than two faces")
    rn = float(np.sqrt(np.mean(y * y)))
    rl = float(np.sqrt(np.mean(x * x)))
    rd = float(np.sqrt(np.mean((x - y) ** 2)))
    corr = (float(np.corrcoef(x, y)[0, 1])
            if x.std() > 0.0 and y.std() > 0.0 else float("nan"))
    return {
        "n": int(x.size),
        "nemo_rms": rn,
        "lego_rms": rl,
        "delta_rms": rd,
        "err_norm": rd / rn if rn > 0.0 else float("nan"),
        "corr": corr,
        "rms_ratio": rl / rn if rn > 0.0 else float("nan"),
        "max_abs_delta": float(np.max(np.abs(x - y))),
    }


def _inverse_participation(delta: np.ndarray, mask: np.ndarray) -> float:
    """Conventional inverse participation of squared residual amplitude."""
    d2 = np.square(np.asarray(delta, dtype=np.float64)[mask])
    denom = float(np.sum(d2 * d2))
    return float(np.sum(d2) ** 2 / denom) if denom > 0.0 else 0.0


def _peak_equivalent_faces(delta: np.ndarray, mask: np.ndarray) -> float:
    """Reviewer metric: residual energy in units of its largest face."""
    d2 = np.square(np.asarray(delta, dtype=np.float64)[mask])
    peak = float(np.max(d2)) if d2.size else 0.0
    return float(np.sum(d2) / peak) if peak > 0.0 else 0.0


def _one_percent_outliers(
        lego: np.ndarray, nemo: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    valid = mask & np.isfinite(lego) & np.isfinite(nemo) & (nemo != 0.0)
    out[valid] = np.abs(lego[valid] / nemo[valid] - 1.0) > 0.01
    return out


def _one_cell_label(participation: float, count: int,
                    row_scaled_max: float) -> str:
    if participation <= 1.01 and count == 1 and row_scaled_max >= 0.20:
        return "CONFIRMED_ONE_CELL_SIGNATURE"
    if participation >= 2.0 or count >= 2 or row_scaled_max <= 0.01:
        return "REFUTED_ONE_CELL_SIGNATURE"
    return "UNRESOLVED_ONE_CELL_SIGNATURE"


def _localize(
        name: str, lego: np.ndarray, nemo: np.ndarray, wet: np.ndarray,
        coastal: np.ndarray, coastal_multiplier: np.ndarray,
        tmask_w: np.ndarray, tmask_e: np.ndarray) -> dict:
    """Registered wet one-cell localization; coastal context is descriptive."""
    delta = lego - nemo
    row_mask = wet & (np.arange(wet.shape[0])[:, None] == SOUTH_ROW)
    if not row_mask.any():
        raise SystemExit(f"{name}: no wet face on registered south row")
    row_abs = np.where(row_mask, np.abs(delta), -np.inf)
    argmax = tuple(int(v) for v in np.unravel_index(np.argmax(row_abs), delta.shape))
    row_nemo_mean = float(np.mean(nemo[row_mask]))
    row_scale = abs(row_nemo_mean)
    if row_scale == 0.0:
        raise SystemExit(f"{name}: zero NEMO row mean")
    row_scaled_max = float(abs(delta[argmax]) / row_scale)
    outliers = _one_percent_outliers(lego, nemo, wet)
    row_outliers = outliers & row_mask
    p_row = _peak_equivalent_faces(delta, row_mask)
    p_domain = _peak_equivalent_faces(delta, wet)
    inverse_p_row = _inverse_participation(delta, row_mask)
    inverse_p_domain = _inverse_participation(delta, wet)

    peak_delta = float(np.max(np.abs(delta)))
    material_bar = 0.01 * peak_delta
    full_support = np.abs(delta) >= material_bar
    support_count = int(full_support.sum())
    coastal_count = int((full_support & coastal).sum())
    coastal_fraction = (float(coastal_count / support_count)
                        if support_count else 0.0)
    argmax_coastal = bool(coastal[argmax])
    one_cell = _one_cell_label(p_row, int(row_outliers.sum()), row_scaled_max)

    coords = np.argwhere(outliers)
    coord_records = [
        {
            "j": int(j), "i": int(i),
            "delta": float(delta[j, i]),
            "ratio_minus_one": float(lego[j, i] / nemo[j, i] - 1.0),
            "umask": int(wet[j, i]),
            "tmask_w": int(tmask_w[j, i]),
            "tmask_e": int(tmask_e[j, i]),
            "coastal_multiplier": float(coastal_multiplier[j, i]),
            "coastal_unmask": bool(coastal[j, i]),
        }
        for j, i in coords
    ]

    # Planted controls exercise the registered decision mechanics.
    candidates = np.argwhere(row_mask & ~row_outliers)
    if candidates.size == 0:
        raise SystemExit(f"{name}: no row face available for planted outlier")
    pj, pi = (int(v) for v in candidates[0])
    planted_delta = delta.copy()
    planted_delta[pj, pi] = delta[argmax]
    planted_lego = nemo + planted_delta
    planted_outliers = _one_percent_outliers(planted_lego, nemo, wet) & row_mask
    planted_p = _peak_equivalent_faces(planted_delta, row_mask)
    planted_label = _one_cell_label(
        planted_p, int(planted_outliers.sum()), row_scaled_max)
    second_face_fires = planted_label == "REFUTED_ONE_CELL_SIGNATURE"
    print(f"CONTROL {name} localization: second_face_fires={second_face_fires} "
          f"planted_P={planted_p:.9g} planted_label={planted_label}")
    if not second_face_fires:
        raise SystemExit(f"{name}: localization planted control failed")

    result = {
        "argmax": {
            "j": argmax[0], "i": argmax[1],
            "delta": float(delta[argmax]),
            "nemo": float(nemo[argmax]),
            "lego": float(lego[argmax]),
            "coastal_unmask": argmax_coastal,
        },
        "row_nemo_mean": row_nemo_mean,
        "row_scaled_max_abs": row_scaled_max,
        "peak_equivalent_faces_j1": p_row,
        "peak_equivalent_faces_domain": p_domain,
        "inverse_participation_j1_descriptive": inverse_p_row,
        "inverse_participation_domain_descriptive": inverse_p_domain,
        "wet_j1_count": int(row_mask.sum()),
        "wet_domain_count": int(wet.sum()),
        "one_percent_j1_count": int(row_outliers.sum()),
        "one_percent_domain_count": int(outliers.sum()),
        "one_percent_outliers": coord_records,
        "full_plane_material_bar": material_bar,
        "full_plane_support_count": support_count,
        "full_plane_support_coastal_count": coastal_count,
        "receiving_support_coastal_fraction_descriptive": coastal_fraction,
        "one_cell_label": one_cell,
        "controls": {
            "second_face_fires": second_face_fires,
            "planted_participation": planted_p,
            "planted_label": planted_label,
        },
    }
    print(f"LOCALIZATION {name} " + json.dumps(result, sort_keys=True))
    return {"result": result, "outliers": outliers}


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int((a | b).sum())
    return float((a & b).sum() / union) if union else 1.0


def _support_label(value: float) -> str:
    if value >= 0.90:
        return "CONFIRMED_COMMON_UPSTREAM_SUPPORT"
    if value <= 0.10:
        return "REFUTED_COMMON_UPSTREAM_SUPPORT"
    return "UNRESOLVED_COMMON_UPSTREAM_SUPPORT"


def _support_overlap(support: np.ndarray, source_mask: np.ndarray) -> float:
    count = int(support.sum())
    return float((support & source_mask).sum() / count) if count else 0.0


def _source_overlap_label(value: float) -> str:
    if value >= 0.90:
        return "CONFIRMED_DONOR_SOURCE_OVERLAP"
    if value <= 0.10:
        return "REFUTED_DONOR_SOURCE_OVERLAP"
    return "UNRESOLVED_DONOR_SOURCE_OVERLAP"


def _donor_trace(
        state, sf, arms: dict, *, direct_delta: np.ndarray,
        fslow_delta: np.ndarray, wet: np.ndarray, coastal: np.ndarray,
        coastal_multiplier: np.ndarray, tmask_w: np.ndarray,
        tmask_e: np.ndarray, rho0: float, dz0_u: np.ndarray) -> dict:
    """Trace the bridged U-as-T carry through its second U interpolation."""
    from legoesm.grids.operators_latlon_cgrid import interp_cell_to_uface

    row_mask = wet & (np.arange(wet.shape[0])[:, None] == SOUTH_ROW)
    receiver = tuple(int(v) for v in np.unravel_index(
        np.argmax(np.where(row_mask, np.abs(direct_delta), -np.inf)),
        direct_delta.shape))
    donor = (receiver[0], (receiver[1] + 1) % wet.shape[1])
    shifted_donor = (receiver[0], (donor[1] + 1) % wet.shape[1])

    # The bridge stores NEMO restart utau_b (already U-point, atmosphere
    # sign) in tau_x_prev. Production then treats it as T-point. Work in the
    # ocean-reaction sign used by surface_stress_faces.
    prior_u = -np.asarray(state.tau_x_prev, dtype=np.float64)
    current_t = -np.asarray(sf.tau_x, dtype=np.float64)
    current_u = np.asarray(interp_cell_to_uface(current_t),
                           dtype=np.float64)[:, 1:]
    production_u = np.asarray(arms[1.0]["tau_i_u"],
                              dtype=np.float64)[:, 1:]
    stagger_correct_u = 0.5 * (prior_u + current_u)
    anomaly_tau = production_u - stagger_correct_u
    predicted_delta = np.zeros_like(anomaly_tau)
    np.divide(RDT * anomaly_tau, rho0 * dz0_u,
              out=predicted_delta, where=dz0_u > 0.0)

    observed = float(direct_delta[receiver])
    predicted = float(predicted_delta[receiver])
    prediction_relative_error = (
        abs(predicted - observed) / abs(observed)
        if observed != 0.0 else float("inf"))

    def material_support(a: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(a[wet])))
        return wet & (np.abs(a) >= 0.01 * peak) if peak > 0.0 else np.zeros_like(wet)

    predicted_support = material_support(predicted_delta)
    direct_support = material_support(direct_delta)
    fslow_support = material_support(fslow_delta)
    predicted_direct_jaccard = _jaccard(predicted_support, direct_support)
    predicted_fslow_jaccard = _jaccard(predicted_support, fslow_support)

    donor_ratio = float(prior_u[donor] / prior_u[receiver])
    donor_is_coastal = bool(coastal[donor])
    receiver_ok = receiver == (1, 49)
    ratio_ok = abs(donor_ratio - 2.0) <= 1.0e-8
    prediction_ok = prediction_relative_error <= 1.0e-6
    support_ok = (predicted_direct_jaccard >= 0.90
                  and predicted_fslow_jaccard >= 0.90)
    if receiver_ok and donor_is_coastal and ratio_ok and prediction_ok and support_ok:
        verdict = "CONFIRMED_COASTAL_DONOR_DOUBLE_INTERPOLATION"
    elif (not donor_is_coastal or abs(donor_ratio - 2.0) >= 0.10
          or prediction_relative_error >= 0.10
          or predicted_direct_jaccard <= 0.10
          or predicted_fslow_jaccard <= 0.10):
        verdict = "REFUTED_COASTAL_DONOR_DOUBLE_INTERPOLATION"
    else:
        verdict = "UNRESOLVED_COASTAL_DONOR_DOUBLE_INTERPOLATION"

    # (a) A one-index donor shift must leave the registered coastal source.
    shifted_donor_fires = not bool(coastal[shifted_donor])
    # (b) Re-run the exact interpolation on a copied field after replacing
    # the east donor. The original anomaly must be nonzero and the replaced
    # donor must annihilate it at the receiver.
    bad_prior_u = np.asarray(interp_cell_to_uface(prior_u),
                             dtype=np.float64)[:, 1:]
    original_prior_anomaly = float(
        0.5 * (bad_prior_u[receiver] - prior_u[receiver]))
    equalized_prior = prior_u.copy()
    equalized_prior[donor] = equalized_prior[receiver]
    equalized_bad_prior = np.asarray(
        interp_cell_to_uface(equalized_prior), dtype=np.float64)[:, 1:]
    equalized_prior_anomaly = float(
        0.5 * (equalized_bad_prior[receiver] - equalized_prior[receiver]))
    equalized_donor_fires = bool(
        original_prior_anomaly != 0.0 and equalized_prior_anomaly == 0.0)
    # (c) Exercise the actual overlap calculation and both decision branches.
    donor_coastal_at_receiver = np.roll(coastal, -1, axis=1)
    confirm_points = np.argwhere(wet & donor_coastal_at_receiver)
    refute_points = np.argwhere(wet & ~donor_coastal_at_receiver)
    if not len(confirm_points) or not len(refute_points):
        raise SystemExit("donor overlap planted-control populations absent")
    synthetic_confirm = np.zeros_like(wet)
    synthetic_refute = np.zeros_like(wet)
    synthetic_confirm[tuple(confirm_points[0])] = True
    synthetic_refute[tuple(refute_points[0])] = True
    confirm_overlap = _support_overlap(
        synthetic_confirm, donor_coastal_at_receiver)
    refute_overlap = _support_overlap(
        synthetic_refute, donor_coastal_at_receiver)
    overlap_gate_fires = bool(
        _source_overlap_label(confirm_overlap)
        == "CONFIRMED_DONOR_SOURCE_OVERLAP"
        and _source_overlap_label(refute_overlap)
        == "REFUTED_DONOR_SOURCE_OVERLAP")
    print("CONTROL donor trace: "
          f"shifted_donor={shifted_donor} shifted_donor_fires={shifted_donor_fires} "
          f"original_prior_anomaly={original_prior_anomaly:.17e} "
          f"equalized_prior_anomaly={equalized_prior_anomaly:.17e} "
          f"equalized_donor_fires={equalized_donor_fires} "
          f"confirm_overlap={confirm_overlap:.9g} "
          f"refute_overlap={refute_overlap:.9g} "
          f"overlap_gate_fires={overlap_gate_fires}")
    if not (shifted_donor_fires and equalized_donor_fires
            and overlap_gate_fires):
        raise SystemExit("donor trace planted control failed")

    def point_record(index: tuple[int, int]) -> dict:
        return {
            "j": index[0], "i": index[1],
            "prior_ocean_sign_stress": float(prior_u[index]),
            "umask": int(wet[index]),
            "tmask_w": int(tmask_w[index]),
            "tmask_e": int(tmask_e[index]),
            "coastal_multiplier": float(coastal_multiplier[index]),
            "coastal_unmask": bool(coastal[index]),
        }

    result = {
        "receiver": point_record(receiver),
        "east_donor": point_record(donor),
        "shifted_donor": point_record(shifted_donor),
        "donor_to_receiver_stress_ratio": donor_ratio,
        "tau_production_receiver": float(production_u[receiver]),
        "tau_stagger_correct_receiver": float(stagger_correct_u[receiver]),
        "measured_direct_delta_receiver": observed,
        "predicted_direct_delta_receiver": predicted,
        "prediction_relative_error": prediction_relative_error,
        "predicted_peak_equivalent_faces_domain": _peak_equivalent_faces(
            predicted_delta, wet),
        "predicted_direct_support_jaccard": predicted_direct_jaccard,
        "predicted_fslow_support_jaccard": predicted_fslow_jaccard,
        "predicted_support_count": int(predicted_support.sum()),
        "direct_support_count": int(direct_support.sum()),
        "fslow_support_count": int(fslow_support.sum()),
        "verdict": verdict,
        "controls": {
            "shifted_donor_fires": shifted_donor_fires,
            "original_prior_anomaly": original_prior_anomaly,
            "equalized_prior_anomaly": equalized_prior_anomaly,
            "equalized_donor_fires": equalized_donor_fires,
            "overlap_gate_fires": overlap_gate_fires,
        },
    }
    print("DONOR_TRACE " + json.dumps(result, sort_keys=True))
    return result


def _scaled_surface_forcing(sf, scale: float):
    return sf._replace(
        tau_x=jnp.asarray(sf.tau_x) * scale,
        tau_y=jnp.asarray(sf.tau_y) * scale,
    )


def _entry_identity_control(state, sf) -> dict:
    """Prove the 1x arm is identical and 0x/2x change stress fields only."""
    def exact(a, b) -> bool:
        aa = jax.tree_util.tree_leaves(a)
        bb = jax.tree_util.tree_leaves(b)
        return len(aa) == len(bb) and all(
            np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(aa, bb)
        )

    one_state = state._replace(
        tau_x_prev=jnp.asarray(state.tau_x_prev) * 1.0,
        tau_y_prev=jnp.asarray(state.tau_y_prev) * 1.0,
    )
    one_sf = _scaled_surface_forcing(sf, 1.0)
    one_exact = exact(one_state, state) and exact(one_sf, sf)
    nonstress_state_exact = True
    for scale in (0.0, 2.0):
        arm_state = state._replace(
            tau_x_prev=jnp.asarray(state.tau_x_prev) * scale,
            tau_y_prev=jnp.asarray(state.tau_y_prev) * scale,
        )
        arm_sf = _scaled_surface_forcing(sf, scale)
        for name in state._fields:
            if name not in ("tau_x_prev", "tau_y_prev"):
                nonstress_state_exact &= exact(
                    getattr(arm_state, name), getattr(state, name))
        for name in sf._fields:
            if name not in ("tau_x", "tau_y"):
                nonstress_state_exact &= exact(getattr(arm_sf, name), getattr(sf, name))
    print("CONTROL entry identity: "
          f"one_x_exact={one_exact} "
          f"nonstress_fields_exact={nonstress_state_exact}")
    if not (one_exact and nonstress_state_exact):
        raise SystemExit("entry-state identity control failed")
    return {"one_x_exact": bool(one_exact),
            "nonstress_fields_exact": bool(nonstress_state_exact)}


def _run_arm(model, state, sf, *, external_rate, scale: float) -> dict:
    """Capture the actual pre-vmix state and F_slow once for one stress scale."""
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocmod
    import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as pemod
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    cap: dict[str, object] = {
        "baro_calls": 0, "vmix_calls": 0, "stress_calls": 0, "events": [],
    }
    real_baro = ocmod.barotropic_substeps_latlon_cgrid
    real_vmix = LatLonCGridOceanModel._apply_implicit_vertical_mixing
    real_stress = pemod.surface_stress_faces

    def spy_stress(*args, **kwargs):
        out = real_stress(*args, **kwargs)
        cap["stress_calls"] = int(cap["stress_calls"]) + 1
        cap["events"].append("stress")
        if out is not None and "tau_i_u" not in cap:
            cap["tau_i_u"] = np.asarray(out[0], dtype=np.float64)
            cap["tau_j_v"] = np.asarray(out[1], dtype=np.float64)
            cap["dz0_u"] = np.asarray(out[2], dtype=np.float64)
            cap["dz0_v"] = np.asarray(out[3], dtype=np.float64)
        return out

    def spy_baro(*args, **kwargs):
        if kwargs.get("eta_init") is not None:
            cap["baro_calls"] = int(cap["baro_calls"]) + 1
            cap["events"].append("baro")
            if "F_slow_u" not in cap:
                cap["F_slow_u"] = np.asarray(kwargs["F_slow_u"], dtype=np.float64)
                cap["F_slow_v"] = np.asarray(kwargs["F_slow_v"], dtype=np.float64)
        return real_baro(*args, **kwargs)

    def spy_vmix(self, state_in, *args, **kwargs):
        if kwargs.get("do_momentum", True):
            cap["vmix_calls"] = int(cap["vmix_calls"]) + 1
            cap["events"].append("vmix")
            if "B1_u" not in cap:
                cap["B1_u"] = np.asarray(state_in.u.data, dtype=np.float64)
                cap["B1_v"] = np.asarray(state_in.v.data, dtype=np.float64)
        return real_vmix(self, state_in, *args, **kwargs)

    ocmod.barotropic_substeps_latlon_cgrid = spy_baro
    LatLonCGridOceanModel._apply_implicit_vertical_mixing = spy_vmix
    pemod.surface_stress_faces = spy_stress
    if state.tau_x_prev is None or state.tau_y_prev is None:
        raise SystemExit("centred wind arm requires the bridged previous stress")
    state_scaled = state._replace(
        tau_x_prev=jnp.asarray(state.tau_x_prev) * scale,
        tau_y_prev=jnp.asarray(state.tau_y_prev) * scale,
    )
    try:
        with jax.disable_jit():
            model.step(
                state_scaled, DT,
                surface_forcing=_scaled_surface_forcing(sf, scale),
                external_tracer_rate=external_rate,
            )
    finally:
        ocmod.barotropic_substeps_latlon_cgrid = real_baro
        LatLonCGridOceanModel._apply_implicit_vertical_mixing = real_vmix
        pemod.surface_stress_faces = real_stress
    required = {"F_slow_u", "F_slow_v", "B1_u", "B1_v",
                "tau_i_u", "tau_j_v", "dz0_u", "dz0_v"}
    if not required <= cap.keys():
        raise SystemExit(f"arm {scale} missed hooks: {required - cap.keys()}")
    expected_events = ("stress", "baro", "stress", "vmix")
    observed_events = tuple(cap["events"])
    planted_events = list(observed_events)
    planted_events[0], planted_events[1] = planted_events[1], planted_events[0]
    event_plant_fires = tuple(planted_events) != expected_events
    print(f"CONTROL arm {scale} hook order: observed={observed_events} "
          f"expected={expected_events} planted_swap_fires={event_plant_fires}")
    if (cap["baro_calls"] != 1 or cap["vmix_calls"] != 1
            or cap["stress_calls"] != 2
            or observed_events != expected_events or not event_plant_fires):
        raise SystemExit(
            f"arm {scale}: expected baro/vmix/stress=1/1/2 in order "
            f"{expected_events}, got {cap['baro_calls']}/{cap['vmix_calls']}/"
            f"{cap['stress_calls']} in {observed_events}")
    return cap


def _whole_step_linearity_diagnostic(
        name: str, d1: np.ndarray, d2: np.ndarray) -> float:
    err = float(np.max(np.abs(d2 - 2.0 * d1)))
    print(f"DIAGNOSTIC {name}: whole-step max|2x-2*1x|={err:.6e} "
          "(not gated; downstream recurrence is state-dependent)")
    return err


def _stress_helper_control(arms: dict) -> dict:
    """Exact 0x/1x/2x check on the genuinely linear single-owner helper."""
    out = {}
    for key in ("tau_i_u", "tau_j_v"):
        z = np.asarray(arms[0.0][key])
        one = np.asarray(arms[1.0][key])
        two = np.asarray(arms[2.0][key])
        zero_exact = bool(np.array_equal(z, np.zeros_like(z)))
        double_exact = bool(np.array_equal(two, 2.0 * one))
        # Prove the equality gate can fail by changing one finite value by one
        # representable step and sending it through the identical comparison.
        planted = two.copy()
        idx = np.unravel_index(int(np.argmax(np.abs(one))), one.shape)
        planted[idx] = np.nextafter(planted[idx], np.inf)
        plant_fires = not np.array_equal(planted, 2.0 * one)
        print(f"CONTROL {key}: zero_exact={zero_exact} "
              f"double_exact={double_exact} planted_nextafter_fires="
              f"{plant_fires}")
        if not (zero_exact and double_exact and plant_fires):
            raise SystemExit(f"surface-stress helper control failed for {key}")
        out[key] = {"zero_exact": zero_exact,
                    "double_exact": double_exact,
                    "planted_nextafter_fires": plant_fires}
    return out


def _copy_applied_term_to_slot_shape(
        bracket_u: np.ndarray, bracket_v: np.ndarray) -> tuple:
    """Copy the applied dynzdf term into the dormant slot's 3-D shape."""
    import netCDF4

    restart = Path(dump_lane.RUN_DIR) / "DINO_00005764_restart.nc"
    with netCDF4.Dataset(restart) as d:
        emitted_u = np.asarray(d.variables["utrd_tau"][0], dtype=np.float64)
        emitted_v = np.asarray(d.variables["vtrd_tau"][0], dtype=np.float64)
    expected_shape = (36,) + bracket_u.shape
    if emitted_u.shape != expected_shape or emitted_v.shape != expected_shape:
        raise SystemExit(
            f"tau slot shape mismatch: u={emitted_u.shape} v={emitted_v.shape} "
            f"expected={expected_shape}")
    emitted_zero = (np.array_equal(emitted_u, np.zeros_like(emitted_u))
                    and np.array_equal(emitted_v, np.zeros_like(emitted_v)))
    planted_emitted = emitted_u.copy()
    planted_emitted.flat[0] = np.nextafter(0.0, 1.0)
    emitted_plant_fires = not np.array_equal(
        planted_emitted, np.zeros_like(planted_emitted))

    applied_u, applied_v = emitted_u.copy(), emitted_v.copy()
    applied_u[0] = bracket_u / RDT
    applied_v[0] = bracket_v / RDT
    lower_zero = (np.array_equal(applied_u[1:], np.zeros_like(applied_u[1:]))
                  and np.array_equal(applied_v[1:], np.zeros_like(applied_v[1:])))
    planted_lower = applied_u.copy()
    planted_lower[1].flat[0] = np.nextafter(0.0, 1.0)
    lower_plant_fires = not np.array_equal(
        planted_lower[1:], np.zeros_like(planted_lower[1:]))

    recon_u, recon_v = RDT * applied_u[0], RDT * applied_v[0]
    recon_err = max(float(np.max(np.abs(recon_u - bracket_u))),
                    float(np.max(np.abs(recon_v - bracket_v))))
    scale = max(float(np.max(np.abs(bracket_u))),
                float(np.max(np.abs(bracket_v))), 1.0e-300)
    recon_bar = 8.0 * np.finfo(np.float64).eps * scale
    planted_top = applied_u.copy()
    idx = np.unravel_index(int(np.argmax(np.abs(bracket_u))), bracket_u.shape)
    planted_top[(0,) + idx] += 1.0e-6 * scale / RDT
    top_plant_fires = float(np.max(
        np.abs(RDT * planted_top[0] - bracket_u))) > recon_bar
    print("RETRACTION=offline_dynzdf_copy_is_not_NEMO_named_jpdyn_tau")
    print("CONTROL offline applied-term slot-shape copy: "
          f"emitted_zero={emitted_zero} emitted_plant_fires={emitted_plant_fires} "
          f"lower_zero={lower_zero} lower_plant_fires={lower_plant_fires} "
          f"recon_err={recon_err:.6e} recon_bar={recon_bar:.6e} "
          f"top_plant_fires={top_plant_fires}")
    if not (emitted_zero and emitted_plant_fires and lower_zero
            and lower_plant_fires and recon_err <= recon_bar
            and top_plant_fires):
        raise SystemExit("offline applied-term slot-shape control failed")
    return applied_u, applied_v, {
        "emitted_zero": bool(emitted_zero),
        "emitted_plant_fires": bool(emitted_plant_fires),
        "lower_zero": bool(lower_zero),
        "lower_plant_fires": bool(lower_plant_fires),
        "reconstruction_max_abs": recon_err,
        "reconstruction_bar": recon_bar,
        "top_plant_fires": bool(top_plant_fires),
        "restart": str(restart),
    }


def main() -> int:
    import multistep_replay as mr
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing,
        dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays,
        dino_step_surface_forcing,
    )
    from legoesm.ocean.fidelity.precision_gate import require_fp64

    set_policy(PrecisionPolicy.fp64())
    sha = _git(["rev-parse", "HEAD"])
    dirty = _git(["status", "--porcelain", "--untracked-files=no"])
    expected_lane = os.path.join(_DINO, "RUN_SEQDUMP_D180_1R")
    expected_step1 = os.path.join(_DINO, "RUN_D180_STEP1")
    lane_inputs = {
        "DINO_1226_LANE": os.environ.get("DINO_1226_LANE"),
        "DINO_1226_IC_STEP": os.environ.get("DINO_1226_IC_STEP"),
        "DINO_NEMO_RUN_TWIN_STEP1": os.environ.get("DINO_NEMO_RUN_TWIN_STEP1"),
        "DINO_ORACLE_ROOT": _DINO,
    }
    if lane_inputs["DINO_1226_LANE"] != "d180":
        raise SystemExit(f"expected d180 lane, got {lane_inputs}")
    if lane_inputs["DINO_1226_IC_STEP"] != "5760":
        raise SystemExit(f"expected IC step 5760, got {lane_inputs}")
    step1_actual = os.path.realpath(
        str(lane_inputs["DINO_NEMO_RUN_TWIN_STEP1"]))
    if step1_actual != os.path.realpath(expected_step1):
        raise SystemExit(f"unexpected step-1 donor: {lane_inputs}")
    if os.path.realpath(dump_lane.RUN_DIR) != os.path.realpath(expected_lane):
        raise SystemExit(
            f"unexpected oracle lane {dump_lane.RUN_DIR}; expected {expected_lane}")
    nemo_root = Path(_DINO).parents[1]
    source_paths = [
        Path(_DINO) / "MY_SRC" / name for name in
        ("dynspg_ts.F90", "dynzdf.F90", "stpmlf.F90", "usrdef_sbc.F90",
         "trddump.F90")
    ] + [
        nemo_root / "src/OCE/SBC/sbcmod.F90",
        nemo_root / "src/OCE/TRD/trddyn.F90",
        nemo_root / "src/OCE/TRD/trd_oce.F90",
    ]
    dump_names = (
        "zdf_dump_u1_prestress.bin", "zdf_dump_u1_poststress.bin",
        "zdf_dump_v1_prestress.bin", "zdf_dump_v1_poststress.bin",
        "wnd_dump_zu_frc_inc.bin", "wnd_dump_zv_frc_inc.bin",
    )
    content_sha256 = {
        str(p): _sha256(p) for p in source_paths
    }
    content_sha256.update({
        dump_lane.dump_path(name): _sha256(dump_lane.dump_path(name))
        for name in dump_names
    })
    tau_restart = Path(dump_lane.RUN_DIR) / "DINO_00005764_restart.nc"
    content_sha256[str(tau_restart)] = _sha256(tau_restart)
    content_sha256[str(Path(__file__).resolve())] = _sha256(Path(__file__).resolve())
    content_sha256[str(_DIR / "PREREG_endwall_wind_placement.md")] = _sha256(
        _DIR / "PREREG_endwall_wind_placement.md")
    content_sha256[str(
        _DIR / "PREREG_endwall_onecell_round2_correction.md")] = _sha256(
            _DIR / "PREREG_endwall_onecell_round2_correction.md")
    content_sha256[str(
        _DIR / "PREREG_endwall_round2_review_corrections.md")] = _sha256(
            _DIR / "PREREG_endwall_round2_review_corrections.md")
    content_sha256[str(
        _DIR / "PREREG_endwall_coastal_donor_source.md")] = _sha256(
            _DIR / "PREREG_endwall_coastal_donor_source.md")
    print("RETRACTION=CONFIRMED_NAMED_AND_APPLIED_DIFFER_2.0966766111")
    print("RETRACTION_REASON=zDt_2_equals_rDt_over_2_and_union_mask_admitted_"
          "324_dry_coastal_u_faces_gate_had_no_reachable_REFUTE")
    print("RETRACTION=south_j1_Pearson_correlation_gate_unreachable_for_"
          "effectively_constant_NEMO_row")
    print("RETRACTION=round2_domain_participation_name_used_inverse_"
          "participation_instead_of_review_peak_equivalent_faces")
    print("RETRACTION=coastal_gate_required_wet_argmax_to_have_umask_zero_"
          "and_was_unreachable")
    print("RETRACTION=receiving_support_coastal_overlap_did_not_test_the_"
          "upstream_donor_and_is_descriptive_only")
    print(json.dumps({
        "provenance": {
            "git_sha": sha,
            "git_dirty": bool(dirty),
            "prereg_commit": PREREG_COMMIT,
            "oracle_root": _DINO,
            "oracle_lane": dump_lane.RUN_DIR,
            "kt": KT,
            "dt_s": DT,
            "lane_inputs": lane_inputs,
            "content_sha256": content_sha256,
            "flags": {
                "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
                "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
                "LEGOESM_NEMO_E3T": os.environ.get("LEGOESM_NEMO_E3T"),
            },
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    }, indent=2, sort_keys=True))
    ptsb.dump_lane.banner()

    g, br, cfg, state = mr.build_replay_ic()
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    if mc.outer_integrator != "leapfrog":
        raise SystemExit(f"active path changed: outer_integrator={mc.outer_integrator!r}")
    if mc.surface_stress_implicit:
        raise SystemExit("baseline no longer uses explicit surface stress")
    require_fp64(br.geometry, br.z_coord, state,
                 context="endwall_wind_placement bridged state")
    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)
    if sf is None or sf.tau_x is None or sf.tau_y is None:
        raise SystemExit("wind-through-step forcing absent")

    placement = getattr(cfg, "surface_tendency_placement", "applied_now")
    external_rate = None
    if placement == "leapfrog_rhs":
        state, external_rate = apply_dino_lat_lon_surface_forcing(
            state, forcing, br.z_coord, cfg, DT,
            t_seconds=KT * DT, return_rate=True)
    else:
        state = apply_dino_lat_lon_surface_forcing(
            state, forcing, br.z_coord, cfg, DT, t_seconds=KT * DT)
    print(f"ACTIVE card outer_integrator={mc.outer_integrator!r} "
          f"surface_stress_implicit={mc.surface_stress_implicit!r} "
          f"surface_tendency_placement={placement!r}")

    entry_control = _entry_identity_control(state, sf)
    arms = {s: _run_arm(model, state, sf, external_rate=external_rate, scale=s)
            for s in (0.0, 1.0, 2.0)}
    b1u = arms[1.0]["B1_u"] - arms[0.0]["B1_u"]
    b1v = arms[1.0]["B1_v"] - arms[0.0]["B1_v"]
    b2u = arms[2.0]["B1_u"] - arms[0.0]["B1_u"]
    b2v = arms[2.0]["B1_v"] - arms[0.0]["B1_v"]
    f1u = arms[1.0]["F_slow_u"] - arms[0.0]["F_slow_u"]
    f1v = arms[1.0]["F_slow_v"] - arms[0.0]["F_slow_v"]
    f2u = arms[2.0]["F_slow_u"] - arms[0.0]["F_slow_u"]
    f2v = arms[2.0]["F_slow_v"] - arms[0.0]["F_slow_v"]
    controls = _stress_helper_control(arms)
    whole_step_diagnostics = {
        "b1_u": _whole_step_linearity_diagnostic("B1_u", b1u, b2u),
        "b1_v": _whole_step_linearity_diagnostic("B1_v", b1v, b2v),
        "fslow_u": _whole_step_linearity_diagnostic("F_slow_u", f1u, f2u),
        "fslow_v": _whole_step_linearity_diagnostic("F_slow_v", f1v, f2v),
    }

    # Reuse the campaign's registered loader and known u-face offset.
    n_pre_u = ptsb._load("zdf_dump_u1_prestress.bin")
    n_post_u = ptsb._load("zdf_dump_u1_poststress.bin")
    n_pre_v = ptsb._load("zdf_dump_v1_prestress.bin")
    n_post_v = ptsb._load("zdf_dump_v1_poststress.bin")
    bracket_u = n_post_u - n_pre_u
    bracket_v = n_post_v - n_pre_v
    applied_tau_u, applied_tau_v, copy_control = (
        _copy_applied_term_to_slot_shape(
            bracket_u, bracket_v)
    )
    # All scoring reads the applied dynzdf term through a controlled copy that
    # reuses the dormant slot's 3-D shape. This is not NEMO's named tau
    # diagnostic, which active trddyn defines without the MLF half factor and
    # with Kmm thickness.
    n_ws_u = RDT * applied_tau_u[0]
    n_ws_v = RDT * applied_tau_v[0]
    n_fu = ptsb._load("wnd_dump_zu_frc_inc.bin")
    n_fv = ptsb._load("wnd_dump_zv_frc_inc.bin")
    um = np.asarray(g.umask, dtype=bool)[..., 0]
    vm = np.asarray(g.vmask, dtype=bool)[..., 0]
    tmask_w = np.asarray(g.tmask, dtype=bool)[..., 0]
    tmask_e = np.roll(tmask_w, -1, axis=1)
    coastal_multiplier = ((2.0 - um.astype(np.float64))
                          * np.maximum(tmask_w, tmask_e))
    coastal_unmask = (~um) & (coastal_multiplier == 2.0)

    # Direct vertical-route deposit, matching _bc_external_surface_forcing's
    # tau/(rho0*dz0) source integrated over the leapfrog rDt.  Do not use B1:
    # that state already includes the independent rotating barotropic route.
    rho0 = float(mc.constants.rho_0)
    dz0u = np.asarray(arms[1.0]["dz0_u"])
    dz0v = np.asarray(arms[1.0]["dz0_v"])
    direct_u = np.zeros_like(dz0u)
    direct_v = np.zeros_like(dz0v)
    np.divide(RDT * np.asarray(arms[1.0]["tau_i_u"]), rho0 * dz0u,
              out=direct_u, where=dz0u > 0.0)
    np.divide(RDT * np.asarray(arms[1.0]["tau_j_v"]), rho0 * dz0v,
              out=direct_v, where=dz0v > 0.0)
    # NEMO u(i) maps to lego u(i+1); v(j) maps directly on this bridge.
    lu = direct_u[:, 1:1 + n_ws_u.shape[1]]
    lv = direct_v[:n_ws_v.shape[0], :n_ws_v.shape[1]]
    lfu = np.asarray(f1u)[:, 1:1 + n_fu.shape[1]]
    lfv = np.asarray(f1v)[:n_fv.shape[0], :n_fv.shape[1]]
    masks = {
        "domain_u": um,
        "south_j1_u": um & (np.arange(um.shape[0])[:, None] == SOUTH_ROW),
        "domain_v": vm,
        "south_j1_v": vm & (np.arange(vm.shape[0])[:, None] == SOUTH_ROW),
    }
    scores = {
        "domain_u_zdf": _stats(lu, n_ws_u, masks["domain_u"]),
        "south_j1_u_zdf": _stats(lu, n_ws_u, masks["south_j1_u"]),
        "domain_u_fslow": _stats(lfu, n_fu, masks["domain_u"]),
        "south_j1_u_fslow": _stats(lfu, n_fu, masks["south_j1_u"]),
    }
    localization_zdf = _localize(
        "zdf", lu, n_ws_u, um, coastal_unmask, coastal_multiplier,
        tmask_w, tmask_e)
    localization_fslow = _localize(
        "fslow", lfu, n_fu, um, coastal_unmask, coastal_multiplier,
        tmask_w, tmask_e)
    z_support = localization_zdf["outliers"]
    f_support = localization_fslow["outliers"]
    support_jaccard = _jaccard(z_support, f_support)
    support_label = _support_label(support_jaccard)
    synthetic_a = np.zeros_like(z_support)
    synthetic_b = np.zeros_like(z_support)
    synthetic_c = np.zeros_like(z_support)
    synthetic_a.flat[0] = True
    synthetic_b.flat[0] = True
    synthetic_c.flat[1] = True
    identical_jaccard = _jaccard(synthetic_a, synthetic_b)
    disjoint_jaccard = _jaccard(synthetic_a, synthetic_c)
    jaccard_plant_fires = bool(
        _support_label(identical_jaccard)
        == "CONFIRMED_COMMON_UPSTREAM_SUPPORT"
        and _support_label(disjoint_jaccard)
        == "REFUTED_COMMON_UPSTREAM_SUPPORT")
    print("CONTROL common support: "
          f"jaccard={support_jaccard:.9g} "
          f"identical={identical_jaccard:.9g} disjoint={disjoint_jaccard:.9g} "
          f"plant_fires={jaccard_plant_fires}")
    if not jaccard_plant_fires:
        raise SystemExit("common-support planted control failed")

    donor_trace = _donor_trace(
        state, sf, arms, direct_delta=lu - n_ws_u,
        fslow_delta=lfu - n_fu, wet=um, coastal=coastal_unmask,
        coastal_multiplier=coastal_multiplier, tmask_w=tmask_w,
        tmask_e=tmask_e, rho0=rho0, dz0_u=dz0u[:, 1:])

    # Structural-zero v control: use max norms, since correlation/normalised
    # error are undefined against an exact zero.
    v_zero = {
        "nemo_zdf_max": float(np.max(np.abs(n_ws_v))),
        "nemo_fslow_max": float(np.max(np.abs(n_fv))),
        "lego_zdf_max": float(np.max(np.abs(lv[vm]))),
        "lego_fslow_max": float(np.max(np.abs(lfv[vm]))),
    }
    vbar = 1.0e-12
    vpass = all(x <= vbar for x in v_zero.values())
    print(f"CONTROL meridional structural zero: {v_zero} bar={vbar:.1e} "
          f"{'PASS' if vpass else 'FAIL'}")
    if not vpass:
        raise SystemExit("meridional structural-zero control failed")

    domain_close = (
        scores["domain_u_zdf"]["err_norm"] <= 0.05
        and scores["domain_u_zdf"]["corr"] >= 0.999
        and scores["domain_u_fslow"]["err_norm"] <= 0.05)
    localized_diff = any(
        localization["result"]["one_cell_label"]
        == "CONFIRMED_ONE_CELL_SIGNATURE"
        for localization in (localization_zdf, localization_fslow))
    row_close = all(
        localization["result"]["row_scaled_max_abs"] <= 0.01
        for localization in (localization_zdf, localization_fslow))
    label = ("CONFIRMED_LOCALIZED_SOURCE_DIFF" if localized_diff else
             "CONFIRMED_ALGEBRAIC_EQUIVALENCE" if domain_close and row_close else
             "PLAUSIBLE_UNRESOLVED")
    out = {
        "provenance": {"git_sha": sha, "prereg_commit": PREREG_COMMIT,
                       "oracle_lane": dump_lane.RUN_DIR, "kt": KT},
        "controls": {"entry_identity": entry_control,
                     "stress_helper": controls,
                     "offline_applied_term_copy": copy_control,
                     "hook_order": {
                         str(scale): {
                             "events": arms[scale]["events"],
                             "stress_calls": arms[scale]["stress_calls"],
                             "baro_calls": arms[scale]["baro_calls"],
                             "vmix_calls": arms[scale]["vmix_calls"],
                         }
                         for scale in (0.0, 1.0, 2.0)
                     }},
        "whole_step_linearity_diagnostics": whole_step_diagnostics,
        "b1_downstream_max": {
            "u": float(np.max(np.abs(b1u))),
            "v": float(np.max(np.abs(b1v))),
        },
        "v_structural_zero": v_zero,
        "scores": scores,
        "localization": {
            "zdf": localization_zdf["result"],
            "fslow": localization_fslow["result"],
            "one_percent_support_jaccard": support_jaccard,
            "support_label": support_label,
            "jaccard_plant_fires": bool(jaccard_plant_fires),
        },
        "coastal_donor_trace": donor_trace,
        "term_label": label,
        "ownership_label": "UNRESOLVED_REQUIRES_REGISTERED_FREE_RUN",
        "retractions": [
            "first run invalid: whole B1 is not an exactly linear stress control",
            "second run invalid: 0x/2x failed to scale the previous centred-stress carry",
            "third run invalid: an already halo-free oracle mask was stripped twice",
            "fourth run invalid: B1 includes the rotating barotropic response "
            "and is not the direct dyn-zdf deposit",
            "fifth run invalid: active leapfrog has two ordered stress-helper "
            "calls, not one",
            "utrd_tau is not a physical zero: its dump slot is emitted but never populated",
            "NEMO wind is not implicit-only: dynspg_ts adds it independently to zu_frc",
            "the offline probe reconstructs source operands; it does not fill the oracle slot",
            "this source-operand score measures neither the implicit-solve "
            "response nor eta ownership",
            "CONFIRMED_NAMED_AND_APPLIED_DIFFER and 2.0966766111: zDt_2 is "
            "rDt/2 and a union mask admitted 324 dry coastal u faces; the "
            "former source-distinction gate had no reachable REFUTE state",
            "south-j1 Pearson correlation as an equivalence gate: NEMO's row "
            "is effectively constant, so row max-absolute error over its mean "
            "is used instead",
            "receiving-face and receiving-support coastal verdicts: a wet "
            "receiver cannot itself be the dry coastal-unmasked donor",
        ],
    }
    print("RESULT " + json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
