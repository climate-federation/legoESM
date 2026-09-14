#!/usr/bin/env python3
"""Walk the GYRE kt=2 stage-1 WZV operands in compiled source order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import nemo_testcase_l2_gyre_phase3_gate as gate  # noqa: E402
import nemo_testcase_l2_gyre_round41_dynadv_split as round41  # noqa: E402
import nemo_testcase_l2_gyre_round46_kt2_stage_gate as round46  # noqa: E402
import nemo_testcase_l2_gyre_round72_tracer_stage as round72  # noqa: E402
import nemo_testcase_l2_gyre_round83_slow_forcing_walk as round83  # noqa: E402
import nemo_testcase_l2_gyre_round84_rhs_walk as round84  # noqa: E402
import nemo_testcase_l2_gyre_round86_zad_operands as round86  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.core.source_rounding import nemo_source_round  # noqa: E402
from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe  # noqa: E402
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d  # noqa: E402

ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3")
ORDER = (
    "velocity_u", "velocity_v", "metric_u", "metric_v", "thickness_u",
    "thickness_v", "metric_thickness_u", "metric_thickness_v", "flux_u",
    "flux_v", "zonal_difference", "meridional_difference", "numerator",
    "reciprocal_area_t", "thickness_t", "hdiv", "e3div", "r3_kbb",
    "r1_dt", "eta_kaa", "r3_kaa", "r3_delta", "stretch", "bracket",
    "incoming_carry", "outgoing_carry", "ww",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _row(candidate, oracle, active, *, name="unnamed") -> dict[str, object]:
    shapes = (np.asarray(candidate).shape, np.asarray(oracle).shape,
              np.asarray(active).shape)
    require(shapes[0] == shapes[1] == shapes[2],
            f"{name} comparison shape mismatch: {shapes}")
    row = round84._row(np.asarray(candidate), np.asarray(oracle), np.asarray(active))
    row["dtype"] = str(np.asarray(candidate).dtype)
    return row


def _source_trace(
    velocity_u, velocity_v, metric_u, metric_v, thickness_u, thickness_v,
    reciprocal_area_t, thickness_t, e3t_0, r3_kbb, r3_kaa, r1_dt, tmask,
):
    """Literal array-valued trace of divhor:123-154 + sshwzv:293-300."""
    sr = nemo_source_round
    metric_thickness_u = sr(metric_u[..., None] * thickness_u)
    metric_thickness_v = sr(metric_v[..., None] * thickness_v)
    flux_u = sr(metric_thickness_u * velocity_u)
    flux_v = sr(metric_thickness_v * velocity_v)
    west = jnp.roll(flux_u, 1, axis=1)
    south = jnp.concatenate([jnp.zeros_like(flux_v[:1]), flux_v[:-1]], axis=0)
    zonal = sr(flux_u - west)
    meridional = sr(flux_v - south)
    numerator = sr(zonal + meridional)
    scaled = sr(numerator * reciprocal_area_t[..., None])
    hdiv = sr(scaled / jnp.where(tmask > 0.5, thickness_t, 1.0)) * tmask
    e3div = sr(hdiv * thickness_t) * tmask
    r3_delta = sr(r3_kaa - r3_kbb)
    stretch = sr((r1_dt * e3t_0) * r3_delta[..., None])
    incoming = []
    outgoing = [None] * velocity_u.shape[-1]
    brackets = [None] * velocity_u.shape[-1]
    carry = jnp.zeros_like(r3_kbb)
    for jk in range(velocity_u.shape[-1] - 1, -1, -1):
        incoming.append(carry)
        bracket = sr(e3div[..., jk] + stretch[..., jk])
        carry = sr(carry - sr(bracket * tmask[..., jk]))
        brackets[jk] = bracket
        outgoing[jk] = carry
    incoming = list(reversed(incoming))
    ww = jnp.stack(outgoing + [jnp.zeros_like(carry)], axis=-1)
    return {
        "velocity_u": velocity_u, "velocity_v": velocity_v,
        "metric_u": metric_u, "metric_v": metric_v,
        "thickness_u": thickness_u, "thickness_v": thickness_v,
        "metric_thickness_u": metric_thickness_u,
        "metric_thickness_v": metric_thickness_v,
        "flux_u": flux_u, "flux_v": flux_v,
        "zonal_difference": zonal, "meridional_difference": meridional,
        "numerator": numerator, "reciprocal_area_t": reciprocal_area_t,
        "thickness_t": thickness_t, "hdiv": hdiv, "e3div": e3div,
        "r3_kbb": r3_kbb, "r1_dt": r1_dt,
        "r3_kaa": r3_kaa, "r3_delta": r3_delta, "stretch": stretch,
        "bracket": jnp.stack(brackets, axis=-1),
        "incoming_carry": jnp.stack(incoming, axis=-1),
        "outgoing_carry": jnp.stack(outgoing, axis=-1), "ww": ww,
    }


def _live_trace(card, seeded, trace):
    eta = seeded.eta.data
    eta_before_field = getattr(seeded, "eta_before", None)
    eta_before = eta if eta_before_field is None else eta_before_field.data
    u, v, *_ = trace.stage_states[0]
    nlev = u.shape[-1]
    tmask = card.recipe.z_coord.is_active.astype(eta.dtype)
    umask, vmask = compute_face_masks_3d(card.recipe.z_coord.is_active, card.recipe.grid)
    ops = pe.nemo_qco_resolved_mesh_operands(
        card.recipe.z_coord, card.recipe.grid, umask, vmask, eta.dtype, nlev)
    live_u, live_v = pe.nemo_qco_live_face_geometry_from_operands(
        eta, ops.e3u_0, ops.e3v_0, ops.umask3, ops.vmask3, ops.hu_0,
        ops.hv_0, ops.area_t, ops.area_u, ops.area_v)[:2]
    h0 = jnp.zeros_like(eta)
    for jk in range(nlev):
        h0 = jax.lax.optimization_barrier(h0 + ops.e3t_0[..., jk] * tmask[..., jk])
    r1_h0 = jax.lax.optimization_barrier(1.0 / jnp.where(h0 > 0.0, h0, 1.0))
    r3_kbb = jax.lax.optimization_barrier(eta * r1_h0)
    thickness_t = ops.e3t_0 * (1.0 + r3_kbb[..., None] * tmask) * tmask
    r1_area = jax.lax.optimization_barrier(1.0 / ops.area_t)

    # Preserve the exact production continuity forecast used when no Kaa
    # override is supplied to nemo_qco_wzv_operands.
    provisional = _source_trace(
        u[:, 1:, :], v[1:, :, :], ops.e2u, ops.e1v, live_u, live_v,
        r1_area, thickness_t, ops.e3t_0, r3_kbb, r3_kbb,
        jax.lax.optimization_barrier(jnp.asarray(1.0, eta.dtype) / card.dt_s),
        tmask,
    )
    barotropic_div = jnp.zeros_like(eta)
    for jk in range(nlev):
        barotropic_div = jax.lax.optimization_barrier(
            barotropic_div + provisional["e3div"][..., jk])
    eta_kaa = jax.lax.optimization_barrier(
        eta_before - jax.lax.optimization_barrier(card.dt_s * barotropic_div))
    eta_kaa = eta_kaa * tmask[..., 0]
    r3_kaa = jax.lax.optimization_barrier(eta_kaa * r1_h0)
    result = _source_trace(
        u[:, 1:, :], v[1:, :, :], ops.e2u, ops.e1v, live_u, live_v,
        r1_area, thickness_t, ops.e3t_0, r3_kbb, r3_kaa,
        jax.lax.optimization_barrier(jnp.asarray(1.0, eta.dtype) / card.dt_s),
        tmask,
    )
    result["eta_kaa"] = eta_kaa
    result["e3t_0"] = ops.e3t_0
    result["tmask"] = tmask
    return result


def _oracle_trace(arrays):
    owned = np.s_[2:-2, 2:-2]
    wet = np.asarray(arrays["tmask"])[owned][..., :30]
    # The Round-46 record is taken before stp2d writes r3t(Kaa); replay that
    # executing assignment from the stored ssh slot and recovered r1_ht_0.
    ssh_kbb = np.asarray(arrays["ssh_Kbb"])[owned]
    r3_kbb = np.asarray(arrays["r3t_Kbb"])[owned]
    r1_h0 = np.divide(r3_kbb, ssh_kbb, out=np.zeros_like(r3_kbb), where=ssh_kbb != 0.0)
    eta_kaa = np.asarray(arrays["ssh_Kaa"])[owned]
    r3_kaa = eta_kaa * r1_h0
    operands = (
        jnp.asarray(arrays["u_Kmm"])[owned][..., :30],
        jnp.asarray(arrays["v_Kmm"])[owned][..., :30],
        jnp.asarray(arrays["e2u"])[owned], jnp.asarray(arrays["e1v"])[owned],
        jnp.asarray(arrays["e3u_Kmm"])[owned][..., :30],
        jnp.asarray(arrays["e3v_Kmm"])[owned][..., :30],
        jnp.asarray(arrays["r1_e1e2t"])[owned],
        jnp.asarray(arrays["e3t_Kmm"])[owned][..., :30],
        jnp.asarray(arrays["e3t_0"])[owned][..., :30], jnp.asarray(r3_kbb),
        jnp.asarray(r3_kaa), jnp.asarray(arrays["r1_Dt"]), jnp.asarray(wet),
    )
    # Both arms must cross the same JIT boundary.  The first instrument
    # version ran this reference trace eagerly and manufactured a 1.69e-21
    # e3div association difference even though both operands were exact.
    values = jax.jit(lambda: _source_trace(*operands))()
    values = jax.device_get(values)
    values["eta_kaa"] = eta_kaa
    values["e3t_0"] = np.asarray(arrays["e3t_0"])[owned][..., :30]
    values["tmask"] = wet
    return values


def _oracle_scalar_boundaries(arrays):
    """Scalar NumPy replay whose final W is calibrated bitwise to NEMO."""
    shape = (26, 36, 30)
    hdiv = np.zeros(shape, dtype=np.float64)
    e3div = np.zeros(shape, dtype=np.float64)
    stretch = np.zeros(shape, dtype=np.float64)
    bracket = np.zeros(shape, dtype=np.float64)
    incoming = np.zeros(shape, dtype=np.float64)
    outgoing = np.zeros(shape, dtype=np.float64)
    ssh_kbb = np.asarray(arrays["ssh_Kbb"])
    r3_kbb = np.asarray(arrays["r3t_Kbb"])
    r1_h0 = np.divide(r3_kbb, ssh_kbb, out=np.zeros_like(r3_kbb),
                      where=ssh_kbb != 0.0)
    r3_kaa = np.asarray(arrays["ssh_Kaa"]) * r1_h0
    for k in range(30):
        for j in range(1, 25):
            for i in range(1, 35):
                zu = np.float64(arrays["e2u"][j, i] * arrays["e3u_Kmm"][j, i, k])
                zu = np.float64(zu * arrays["u_Kmm"][j, i, k])
                zuw = np.float64(arrays["e2u"][j, i - 1] * arrays["e3u_Kmm"][j, i - 1, k])
                zuw = np.float64(zuw * arrays["u_Kmm"][j, i - 1, k])
                zv = np.float64(arrays["e1v"][j, i] * arrays["e3v_Kmm"][j, i, k])
                zv = np.float64(zv * arrays["v_Kmm"][j, i, k])
                zvs = np.float64(arrays["e1v"][j - 1, i] * arrays["e3v_Kmm"][j - 1, i, k])
                zvs = np.float64(zvs * arrays["v_Kmm"][j - 1, i, k])
                total = np.float64(np.float64(zu - zuw) + np.float64(zv - zvs))
                value = np.float64(total * arrays["r1_e1e2t"][j, i])
                value = np.float64(value / arrays["e3t_Kmm"][j, i, k])
                hdiv[j, i, k] = value
                e3div[j, i, k] = np.float64(value * arrays["e3t_Kmm"][j, i, k])
    carry = np.zeros((26, 36), dtype=np.float64)
    for k in range(29, -1, -1):
        for j in range(1, 25):
            for i in range(1, 35):
                incoming[j, i, k] = carry[j, i]
                delta = np.float64(r3_kaa[j, i] - r3_kbb[j, i])
                z = np.float64(arrays["r1_Dt"] * arrays["e3t_0"][j, i, k])
                z = np.float64(z * delta)
                stretch[j, i, k] = z
                total = np.float64(e3div[j, i, k] + z)
                bracket[j, i, k] = total
                carry[j, i] = np.float64(carry[j, i] - total) * arrays["tmask"][j, i, k]
                outgoing[j, i, k] = carry[j, i]
    owned = np.s_[2:-2, 2:-2]
    ww = np.concatenate(
        [outgoing[owned], np.zeros((22, 32, 1), dtype=np.float64)], axis=-1)
    require(np.array_equal(ww, np.asarray(arrays["ww"])[owned]),
            "scalar intermediate trace does not reproduce NEMO W")
    return {name: value[owned] for name, value in {
        "hdiv": hdiv, "e3div": e3div, "stretch": stretch,
        "bracket": bracket, "incoming_carry": incoming,
        "outgoing_carry": outgoing,
    }.items()} | {"ww": ww}


def _direct_production_w(card, seeded, trace, eta_after_override):
    """Call the same shared WZV helper that the production tendency uses."""
    eta = seeded.eta.data
    eta_before_field = getattr(seeded, "eta_before", None)
    eta_before = eta if eta_before_field is None else eta_before_field.data
    u, v, *_ = trace.stage_states[0]
    tmask = card.recipe.z_coord.is_active.astype(eta.dtype)
    umask, vmask = compute_face_masks_3d(card.recipe.z_coord.is_active, card.recipe.grid)
    return pe.nemo_qco_wzv_operands(
        eta, eta_before, u, v, card.recipe.grid, card.recipe.z_coord,
        umask, vmask, tmask, card.dt_s,
        eta_after_override=eta_after_override,
    )[0]


def _active(name, masks):
    if name == "metric_u":
        return masks["u"][..., 0]
    if name == "metric_v":
        return masks["v"][..., 0]
    if name.endswith("_u") and name not in ("zonal_difference",):
        return masks["u"]
    if name.endswith("_v"):
        return masks["v"]
    if name in ("metric_u", "metric_v", "reciprocal_area_t", "r3_kbb",
                "r1_dt", "eta_kaa", "r3_kaa", "r3_delta"):
        if name == "r1_dt":
            return np.asarray(True)
        return masks["t"][..., 0]
    if name == "ww":
        return np.concatenate([masks["t"], np.zeros((*masks["t"].shape[:2], 1), bool)], axis=-1)
    return masks["t"]


def _recur(e3div, e3t0, r1dt, r3delta, tmask):
    carry = np.zeros(e3div.shape[:2], dtype=np.float64)
    out = [None] * e3div.shape[-1]
    for jk in range(e3div.shape[-1] - 1, -1, -1):
        stretch = np.float64(r1dt) * e3t0[..., jk]
        stretch = stretch * r3delta
        total = e3div[..., jk] + stretch
        carry = carry - total * tmask[..., jk]
        out[jk] = carry.copy()
    return np.stack(out + [np.zeros_like(carry)], axis=-1)


def measure(args) -> dict[str, object]:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "precision changed")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")
    stamp = worktree_stamp()
    require(stamp["clean"], "Round-87 measurement worktree is dirty")
    require(stamp["commit"].lower() == args.expect_commit.lower(), "Round-87 commit stamp mismatch")

    stage, _ = round83._admit_round64(args)
    arrays = stage["arrays"]
    exact_replay = round46._wzv_replay({
        **arrays,
        "r3t_Kaa": np.asarray(arrays["ssh_Kaa"]) * np.divide(
            np.asarray(arrays["r3t_Kbb"]), np.asarray(arrays["ssh_Kbb"]),
            out=np.zeros_like(arrays["r3t_Kbb"]), where=np.asarray(arrays["ssh_Kbb"]) != 0.0),
    })
    literal_unequal = int(np.count_nonzero(
        exact_replay[1:25, 1:35, :] != np.asarray(arrays["ww"])[1:25, 1:35, :]))
    require(literal_unequal == 0, "NEMO-given-input WZV replay moved")

    base, context = round72._capture_seeded_context(SimpleNamespace(
        expect_commit=args.expect_commit, expect_krhs_commit=args.expect_krhs_commit,
        output=args.prerequisite_output))
    require(base["status"] == "MEASURED", "kt=2 seeded context changed")
    card, seeded, _, _, live_step = context
    live = jax.device_get(jax.jit(lambda: _live_trace(card, seeded, live_step))())
    oracle = _oracle_trace(arrays)
    oracle.update(_oracle_scalar_boundaries(arrays))
    masks = gate.expected_masks(card)
    active = {"u": np.asarray(masks["u"], bool), "v": np.asarray(masks["v"], bool),
              "t": np.asarray(masks["T"], bool)}
    captured_w = np.asarray(live_step.operator_operands[0]["operand_zad_w"])
    direct_current_w = jax.device_get(jax.jit(
        lambda: _direct_production_w(card, seeded, live_step, None))())
    direct_kaa_w = jax.device_get(jax.jit(
        lambda: _direct_production_w(
            card, seeded, live_step, jnp.asarray(oracle["eta_kaa"])))())
    captured_consistency = _row(
        direct_current_w, captured_w, _active("ww", active), name="captured_consistency")
    actual_stage_w = np.asarray(live_step.stage_geometry[0][2])
    actual_postsolve = _row(actual_stage_w, oracle["ww"], _active("ww", active))

    ordinary = {name: _row(live[name], oracle[name], _active(name, active), name=name)
                for name in ORDER}
    rows = {name: dict(value) for name, value in ordinary.items()}
    plant_detail = None
    if args.plant:
        target_name = {"flux-ulp": "flux_u", "kaa-ulp": "r3_kaa",
                       "carry-ulp": "outgoing_carry"}[args.plant]
        changed, at = round84._away_one_ulp(
            np.asarray(oracle[target_name]), np.asarray(live[target_name]),
            _active(target_name, active))
        rows[target_name] = _row(live[target_name], changed, _active(target_name, active))
        plant_detail = {"field": target_name, "location": list(at)}
        fired = bool(rows[target_name] != ordinary[target_name])
        require(fired, f"{args.plant} differential control was invisible")
    else:
        fired = None

    first = next((name for name in ORDER if not rows[name]["bit_exact"]), None)
    kaa_w = np.asarray(direct_kaa_w)
    kaa_w_row = _row(kaa_w, oracle["ww"], _active("ww", active), name="kaa_w")
    oracle_kaa_twice_kbb = _row(
        np.asarray(oracle["r3_kbb"]) + np.asarray(oracle["r3_kbb"]),
        oracle["r3_kaa"], active["t"][..., 0], name="oracle_kaa_twice_kbb")

    split = round46._split_view(arrays)
    oracle_after = {face: np.asarray(arrays[f"after_zad_{face}"]) for face in ("u", "v")}
    zad = {}
    for label, replacement_w in (("before", np.asarray(direct_current_w)), ("kaa", kaa_w),
                                 ("actual_postsolve", actual_stage_w)):
        replay_u, replay_v = round86._replay_with(
            split, {"ww": round86._inject_owned(split["ww"], replacement_w[..., :30], nlev=30)})
        zad[label] = {
            "u": _row(round83.owned3(replay_u), round83.owned3(oracle_after["u"]), active["u"]),
            "v": _row(round83.owned3(replay_v), round83.owned3(oracle_after["v"]), active["v"]),
        }
    removed = {
        face: (1.0 - zad["kaa"][face]["absolute_max"] / zad["before"][face]["absolute_max"]
               if zad["before"][face]["absolute_max"] else 0.0)
        for face in ("u", "v")
    }
    earlier = ORDER[:ORDER.index("eta_kaa")]
    confirmed = bool(
        args.plant is None and captured_consistency["bit_exact"]
        and all(rows[name]["bit_exact"] for name in earlier)
        and first == "eta_kaa" and kaa_w_row["bit_exact"]
        and all(value >= 0.9 for value in removed.values()))
    return {
        "format": "nemo-testcase-l2-gyre-round87-wzv-inputs-v1",
        "status": "PLANT_FIRED" if args.plant else ("CONFIRMED" if confirmed else "REFUTED"),
        "worktree": stamp, "literal_given_input_unequal": literal_unequal,
        "operand_order": list(ORDER), "operand_rows": rows,
        "first_nonbit_operand": first, "captured_preexternal_consistency": captured_consistency,
        "actual_postsolve_transport_w": actual_postsolve,
        "kaa_substitution_w": kaa_w_row, "zad_replays": zad,
        "posthoc_oracle_r3_kaa_equals_kbb_plus_kbb": oracle_kaa_twice_kbb,
        "kaa_zad_fraction_removed": removed, "prediction_confirmed": confirmed,
        "plant": args.plant, "plant_detail": plant_detail, "plant_fired": fired,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-krhs-commit", required=True)
    parser.add_argument("--round64-root", type=Path, default=ROOT / "round64/oracle_krhs_split")
    parser.add_argument("--round46-root", type=Path, default=ROOT / "round46/oracle_kt2_stage")
    parser.add_argument("--round64-admission", type=Path,
                        default=ROOT / "round64/oracle_krhs_split/round64_admission.json")
    parser.add_argument("--prerequisite-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", choices=("flux-ulp", "kaa-ulp", "carry-ulp"))
    args = parser.parse_args(argv)
    report = measure(args)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("ROUND87_WZV_INPUTS", report["status"])
    return 1 if args.plant or report["status"] != "CONFIRMED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
