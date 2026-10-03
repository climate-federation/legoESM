#!/usr/bin/env python3
"""Split rung-0 EEN accumulation at NEMO's live-thickness mask boundary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round98_coriolis_residual as residual,
)


PLANTS = ("none", "current-bit", "candidate-bit")
EXPECTED_CURRENT = {
    "acc_u_ne": (3940, 426),
    "acc_u_nw": (3953, 438),
    "acc_u_se": (4009, 438),
    "acc_u_sw": (4013, 441),
    "acc_v_ne": (3990, 486),
    "acc_v_nw": (3987, 483),
    "acc_v_se": (3933, 348),
    "acc_v_sw": (3920, 336),
}


class GateError(RuntimeError):
    """The registered operand walk cannot support its claim."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _score(candidate: np.ndarray, reference: np.ndarray) -> dict:
    candidate = np.ascontiguousarray(candidate, dtype=np.float64)
    reference = np.ascontiguousarray(reference, dtype=np.float64)
    require(candidate.shape == reference.shape, "operand score shape moved")
    bits = candidate.view(np.uint64) != reference.view(np.uint64)
    magnitude = candidate != reference
    signed_zero = bits & ~magnitude
    locations = np.argwhere(bits)
    return {
        "bit_unequal": int(np.count_nonzero(bits)),
        "magnitude_unequal": int(np.count_nonzero(magnitude)),
        "signed_zero_only": int(np.count_nonzero(signed_zero)),
        "nonfold_magnitude_unequal": int(np.count_nonzero(magnitude[:-1])),
        "fold_magnitude_unequal": int(np.count_nonzero(magnitude[-1])),
        "first_bit_unequal_j_i": (
            None if locations.size == 0 else list(map(int, locations[0]))
        ),
    }


def literal_accumulators(eta, z_coord, dtype, *, grid,
                         source_face_thickness: bool,
                         literal_bottom_loop: bool = False,
                         return_fraction_operands: bool = False,
                         south_ff_copy: bool = False,
                         south_e3f0_fill=None):
    """Replay compiled dynspg_ts.f90:1231-1280 with one mask choice."""

    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_een_south_e3f0,
        _nemo_south_copy_fill,
    )
    from legoesm.ocean.vertical import nemo_e3f_0vor_from_tmask

    raw = z_coord.nemo_een_barotropic
    require(raw is not None, "rung-0 card has no carried EEN operands")
    b = nemo_source_round
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    quarter = jnp.asarray(0.25, dtype=dtype)
    eta = jnp.asarray(eta, dtype=dtype)
    ff = jnp.asarray(raw.ff_f, dtype=dtype)
    umask = jnp.asarray(raw.umask, dtype=dtype)
    vmask = jnp.asarray(raw.vmask, dtype=dtype)
    fmask = jnp.asarray(raw.fe3mask, dtype=dtype)
    e3u0 = jnp.asarray(raw.e3u_0, dtype=dtype)
    e3v0 = jnp.asarray(raw.e3v_0, dtype=dtype)
    e3f0 = b(nemo_e3f_0vor_from_tmask(
        jnp.asarray(z_coord.nemo_e3t_0, dtype=dtype), z_coord.is_active,
        jnp.asarray(raw.e3f_0, dtype=dtype), nn_e3f_typ=0, grid=grid,
    ))

    def recip(depth, wet):
        return b(wet / b(depth + one - wet))

    hu0 = jnp.asarray(raw.hu_0, dtype=dtype)
    hv0 = jnp.asarray(raw.hv_0, dtype=dtype)
    hf0 = jnp.asarray(raw.hf_0, dtype=dtype)
    r1_hu0 = recip(hu0, (hu0 > 0.0).astype(dtype))
    r1_hv0 = recip(hv0, (hv0 > 0.0).astype(dtype))
    r1_hf0 = recip(hf0, (hf0 > 0.0).astype(dtype))
    e1t = jnp.asarray(raw.e1t, dtype=dtype)
    e2t = jnp.asarray(raw.e2t, dtype=dtype)
    e1u = jnp.asarray(raw.e1u, dtype=dtype)
    e2u = jnp.asarray(raw.e2u, dtype=dtype)
    e1v = jnp.asarray(raw.e1v, dtype=dtype)
    e2v = jnp.asarray(raw.e2v, dtype=dtype)
    e1f = jnp.asarray(raw.e1f, dtype=dtype)
    e2f = jnp.asarray(raw.e2f, dtype=dtype)
    area_eta = b(b(e1t * e2t) * eta)
    east = jnp.roll(area_eta, -1, axis=1)
    north = jnp.roll(area_eta, -1, axis=0)
    northeast = jnp.roll(north, -1, axis=1)
    r3u = b(b(half * b(area_eta + east)) * r1_hu0 / b(e1u * e2u))
    r3v = b(b(half * b(area_eta + north)) * r1_hv0 / b(e1v * e2v))
    quad = b(b(area_eta + east) + b(north + northeast))
    r3f = b(b(quarter * quad) * r1_hf0 / b(e1f * e2f))
    source_e3u = b(e3u0 * b(one + r3u[..., None] * umask))
    source_e3v = b(e3v0 * b(one + r3v[..., None] * vmask))
    current_e3u = b(source_e3u * umask)
    current_e3v = b(source_e3v * vmask)
    e3u = source_e3u if source_face_thickness else current_e3u
    e3v = source_e3v if source_face_thickness else current_e3v
    e3f = b(e3f0 * b(one + r3f[..., None] * fmask))
    q = b(ff[..., None] / e3f)
    levels = jnp.arange(1, umask.shape[-1] + 1, dtype=jnp.int32)
    mbku = jnp.maximum(jnp.max(jnp.where(umask > 0.0, levels, 0), axis=-1), 1)
    mbkv = jnp.maximum(jnp.max(jnp.where(vmask > 0.0, levels, 0), axis=-1), 1)

    def shift(value, di=0, dj=0):
        out = jnp.roll(value, di, axis=1) if di else value
        return jnp.roll(out, dj, axis=0) if dj else out

    q_south = shift(q, 0, 1)
    if south_ff_copy:
        ff_south = _nemo_south_copy_fill(ff)
        south_e3f = shift(e3f, 0, 1)
        if south_e3f0_fill is not None:
            e3f0_south = _nemo_een_south_e3f0(
                e3f0, jnp.asarray(south_e3f0_fill, dtype=dtype))
            south_e3f = b(e3f0_south * b(
                one + shift(r3f, 0, 1)[..., None] * shift(fmask, 0, 1)))
        q_south = b(ff_south[..., None] / south_e3f)

    def triad(a, c, d):
        return b(b(a + c) + d)

    uq = {
        "nw": triad(shift(q, 1, 0), q, q_south),
        "ne": triad(q_south, q, shift(q, -1, 0)),
        "sw": triad(q, q_south, shift(q_south, 1, 0)),
        "se": triad(shift(q_south, -1, 0), q_south, q),
    }
    vq = {
        "se": triad(shift(q, 1, 0), q, q_south),
        "sw": triad(shift(q_south, 1, 0), shift(q, 1, 0), q),
        "ne": triad(shift(q, 0, -1), q, shift(q, 1, 0)),
        "nw": triad(q, shift(q, 1, 0), shift(q, 1, -1)),
    }
    un = {
        "nw": (e3v, vmask),
        "ne": (shift(e3v, -1, 0), shift(vmask, -1, 0)),
        "sw": (shift(e3v, 0, 1), shift(vmask, 0, 1)),
        "se": (shift(e3v, -1, 1), shift(vmask, -1, 1)),
    }
    vn = {
        "nw": (shift(e3u, 1, -1), shift(umask, 1, -1)),
        "ne": (shift(e3u, 0, -1), shift(umask, 0, -1)),
        "sw": (shift(e3u, 1, 0), shift(umask, 1, 0)),
        "se": (e3u, umask),
    }

    def accumulate(face, neighbor, neighbor_mask, q_factor, bottom):
        term = b(b(b(face * neighbor) * neighbor_mask) * q_factor)
        acc = jnp.zeros_like(eta)
        for jk in range(term.shape[-1]):
            updated = b(acc + term[..., jk])
            acc = jnp.where(jk < bottom, updated, acc) if literal_bottom_loop else updated
        return acc, term

    accumulators = {}
    terms = {}
    for corner in ("nw", "ne", "sw", "se"):
        acc, term = accumulate(
            e3u, un[corner][0], un[corner][1], uq[corner], mbku)
        accumulators[f"acc_u_{corner}"] = acc
        terms[f"term_u_{corner}"] = term
        acc, term = accumulate(
            e3v, vn[corner][0], vn[corner][1], vq[corner], mbkv)
        accumulators[f"acc_v_{corner}"] = acc
        terms[f"term_v_{corner}"] = term
    parts = {
        "source_e3u": source_e3u,
        "source_e3v": source_e3v,
        "current_e3u": current_e3u,
        "current_e3v": current_e3v,
        "zpvo_u_nw": uq["nw"],
        "mbku": mbku,
        "mbkv": mbkv,
        **terms,
    }
    if return_fraction_operands:
        parts.update({
            "een_ff": ff,
            "een_e3f0": e3f0,
            "een_r3f": r3f,
            "een_fmask": fmask,
            "een_denom": e3f,
            "een_q": q,
        })
    return accumulators, parts


def measure(deck_root: Path, frame_root: Path, accumulator_root: Path,
            expect_commit: str, plant: str) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.vertical import compute_layer_thickness, nemo_dynvor_e3f_0vor

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-107 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-107 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-107 walk requires production JIT on CPU")

    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = rung0.bridge_entry(card, rung0.assemble_frame(frame_root, 1, 0))
    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "literal EEN path has no carried operands")
    e3t0 = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data, card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    )
    source_divisor = nemo_dynvor_e3f_0vor(
        e3t0, card.recipe.z_coord.is_active, grid=card.recipe.grid,
        dtype=jnp.float64, substitute_e3f=raw.e3f_0,
    )
    source_z = card.recipe.z_coord._replace(
        nemo_een_barotropic=raw._replace(e3f_0=source_divisor),
    )
    eta = jnp.asarray(state.eta.data, dtype=jnp.float64)
    current, current_parts = jax.device_get(jax.jit(
        lambda value: literal_accumulators(
            value, source_z, jnp.float64, grid=card.recipe.grid,
            source_face_thickness=False),
    )(eta))
    candidate, candidate_parts = jax.device_get(jax.jit(
        lambda value: literal_accumulators(
            value, source_z, jnp.float64, grid=card.recipe.grid,
            source_face_thickness=True),
    )(eta))
    literal_loop, literal_loop_parts = jax.device_get(jax.jit(
        lambda value: literal_accumulators(
            value, source_z, jnp.float64, grid=card.recipe.grid,
            source_face_thickness=True, literal_bottom_loop=True),
    )(eta))
    oracle, census = residual.assemble_oracle_accumulators(accumulator_root)

    if plant == "current-bit":
        planted = np.array(current["acc_u_nw"], copy=True)
        planted[1, 49] = np.nextafter(planted[1, 49], np.float64(np.inf))
        current = dict(current, acc_u_nw=planted)
    current_score = {name: _score(value, oracle[name]) for name, value in current.items()}
    observed = {
        name: (row["bit_unequal"], row["magnitude_unequal"])
        for name, row in current_score.items()
    }
    require(observed == EXPECTED_CURRENT,
            "current replay does not reproduce round-106 accumulator census: "
            + json.dumps(observed, sort_keys=True))

    unplanted_candidate = literal_loop
    if plant == "candidate-bit":
        planted = np.array(literal_loop["acc_u_nw"], copy=True)
        planted[1, 49] = np.nextafter(planted[1, 49], np.float64(np.inf))
        literal_loop = dict(literal_loop, acc_u_nw=planted)
    candidate_score = {
        name: _score(value, oracle[name]) for name, value in candidate.items()
    }
    literal_loop_score = {
        name: _score(value, oracle[name]) for name, value in literal_loop.items()
    }
    if plant == "candidate-bit":
        require(literal_loop_score["acc_u_nw"] != _score(
            unplanted_candidate["acc_u_nw"], oracle["acc_u_nw"]),
            "candidate-bit plant rounded away")
        raise GateError("candidate-bit plant fired")

    face_changes = {
        "e3u": _score(current_parts["current_e3u"], candidate_parts["source_e3u"]),
        "e3v": _score(current_parts["current_e3v"], candidate_parts["source_e3v"]),
        "zpvo_u_nw_self_identity": _score(
            current_parts["zpvo_u_nw"], candidate_parts["zpvo_u_nw"]),
        "term_u_nw": _score(
            current_parts["term_u_nw"], candidate_parts["term_u_nw"]),
        "literal_loop_term_u_nw_identity": _score(
            candidate_parts["term_u_nw"], literal_loop_parts["term_u_nw"]),
    }
    return {
        "status": "MEASURED_R107_EEN_U_OPERAND_WALK",
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_census": census,
        "current_accumulators": current_score,
        "source_face_thickness_accumulators": candidate_score,
        "literal_bottom_loop_accumulators": literal_loop_score,
        "operand_movement": face_changes,
        "r107_p1_zpvo_expression_unchanged": (
            face_changes["zpvo_u_nw_self_identity"]["bit_unequal"] == 0),
        "r107_p2_ffu_nw_magnitude_exact": (
            candidate_score["acc_u_nw"]["magnitude_unequal"] == 0),
        "r107_p3_ffu_nw_bit_exact": (
            candidate_score["acc_u_nw"]["bit_unequal"] == 0),
        "r107_p6_ffu_nw_magnitude_exact": (
            literal_loop_score["acc_u_nw"]["magnitude_unequal"] == 0),
        "r107_p7_ffu_nw_bit_exact": (
            literal_loop_score["acc_u_nw"]["bit_unequal"] == 0),
        "r107_p8_all_u_nonfold_magnitude_exact": all(
            literal_loop_score[f"acc_u_{corner}"]["nonfold_magnitude_unequal"] == 0
            for corner in ("nw", "ne", "sw", "se")
        ),
        "worktree": stamp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--accumulator-root", type=Path, required=True)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.deck_root, args.frame_root, args.accumulator_root,
                         args.expect_commit, args.plant)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, GateError) as error:
        if args.plant != "none":
            print(f"STATUS PLANT-FIRED {args.plant}: {error}")
        else:
            print(f"STATUS REFUSE: {error}")
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    print(f"STATUS {result['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
