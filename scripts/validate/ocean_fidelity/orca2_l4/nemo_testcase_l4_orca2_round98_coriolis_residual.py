#!/usr/bin/env python3
"""Walk the independent rung-0 literal Coriolis residual after e3f_0vor."""

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
    nemo_testcase_l4_orca2_round93_rhs_walk as rhs_walk,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (
    nemo_testcase_l4_orca2_round97_spgts_walk as r97,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round98_een_coeff_acquisition import (
    check_record as coefficient_record,
)
from scripts.validate.ocean_fidelity.orca2_l4.nemo_testcase_l4_orca2_round104_een_accum_acquisition import (
    check_record as accumulator_record,
)
from scripts.validate.ocean_fidelity.testcases import (
    nemo_testcase_l2_gyre_phase3_gate as phase3_gate,
)


PLANTS = ("none", "coefficient-bit", "application-bit", "scale-bit",
          "accumulator-bit")
COEFFICIENTS = (
    "ffu_nw", "ffu_ne", "ffu_sw", "ffu_se",
    "ffv_sw", "ffv_se", "ffv_nw", "ffv_ne",
)


class GateError(RuntimeError):
    """The admitted record cannot distinguish the registered statement."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def strict_application(u_bar, v_bar, coefficients):
    """Apply compiled dyn_cor_2D association with materialized NumPy ops."""

    ua = np.asarray(u_bar)[:, 1:]
    va = np.asarray(v_bar)[1:, :]
    east_v = np.roll(va, -1, axis=1)
    south_v = np.concatenate([np.zeros_like(va[:1]), va[:-1]], axis=0)
    southeast_v = np.roll(south_v, -1, axis=1)
    north_u = np.concatenate([ua[1:], np.zeros_like(ua[:1])], axis=0)
    west_u = np.roll(ua, 1, axis=1)
    northwest_u = np.roll(north_u, 1, axis=1)

    def product(name, value):
        return np.asarray(coefficients[name]) * value

    u_nw = product("ffu_nw", va)
    u_ne = product("ffu_ne", east_v)
    u_sw = product("ffu_sw", south_v)
    u_se = product("ffu_se", southeast_v)
    v_sw = product("ffv_sw", west_u)
    v_se = product("ffv_se", ua)
    v_nw = product("ffv_nw", northwest_u)
    v_ne = product("ffv_ne", north_u)
    u_north = u_nw + u_ne
    u_south = u_sw + u_se
    v_south = v_sw + v_se
    v_north = v_nw + v_ne
    return u_north + u_south, -(v_south + v_north), {
        "u_nw": u_nw, "u_ne": u_ne, "u_sw": u_sw, "u_se": u_se,
        "v_sw": v_sw, "v_se": v_se, "v_nw": v_nw, "v_ne": v_ne,
        "u_north_pair": u_north, "u_south_pair": u_south,
        "v_south_pair": v_south, "v_north_pair": v_north,
    }


def score_application(trace, coefficients, oracle, active, *, plant="none"):
    rows = []
    product_signs = {}
    for index in range(2):
        prefix = f"j{index + 1:03d}"
        u, v, products = strict_application(
            trace["u_mid"][index], trace["v_mid"][index], coefficients)
        if plant == "application-bit" and index == 0:
            u = np.array(u, copy=True)
            u[1, 49] = np.nextafter(u[1, 49], np.float64(np.inf))
        u_row = rhs_walk.score(u, oracle[f"{prefix}_cor_u"], active["u"])
        v_row = rhs_walk.score(v, oracle[f"{prefix}_cor_v"], active["v"])
        rows.extend(({"substep": index + 1, "face": "u", **u_row},
                     {"substep": index + 1, "face": "v", **v_row}))
        product_signs[prefix] = {
            name: {
                "positive_zero": int(np.count_nonzero(
                    np.asarray(value).view(np.uint64) == np.uint64(0))),
                "negative_zero": int(np.count_nonzero(
                    np.asarray(value).view(np.uint64)
                    == np.uint64(0x8000000000000000))),
            }
            for name, value in products.items()
        }
    return {"rows": rows, "product_zero_signs": product_signs}


def coefficient_movement(base, candidate):
    rows = {}
    for name in COEFFICIENTS:
        a = np.asarray(base[name])
        b = np.asarray(candidate[name])
        moved = a != b
        rows[name] = {
            "differing_cells": int(np.count_nonzero(moved)),
            "non_fold_differing_cells": int(np.count_nonzero(moved[:-1])),
            "fold_differing_cells": int(np.count_nonzero(moved[-1])),
            "maximum_absolute": float(np.max(np.abs(a - b))),
        }
    return rows


def coefficient_difference_census(candidate, oracle):
    """Separate signed-zero differences from nonzero and fold differences."""

    candidate = np.asarray(candidate, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    require(candidate.shape == oracle.shape,
            "coefficient census shape mismatch")
    bit_unequal = (
        np.ascontiguousarray(candidate).view(np.uint64)
        != np.ascontiguousarray(oracle).view(np.uint64)
    )
    signed_zero_only = bit_unequal & (candidate == 0.0) & (oracle == 0.0)
    magnitude_unequal = candidate != oracle

    def first(mask):
        locations = np.argwhere(mask)
        return None if locations.size == 0 else list(map(int, locations[0]))

    return {
        "bit_unequal": int(np.count_nonzero(bit_unequal)),
        "signed_zero_only": int(np.count_nonzero(signed_zero_only)),
        "magnitude_unequal": int(np.count_nonzero(magnitude_unequal)),
        "fold_bit_unequal": int(np.count_nonzero(bit_unequal[-1])),
        "fold_magnitude_unequal": int(np.count_nonzero(magnitude_unequal[-1])),
        "nonfold_magnitude_unequal": int(np.count_nonzero(magnitude_unequal[:-1])),
        "first_bit_unequal_j_i": first(bit_unequal),
        "first_magnitude_unequal_j_i": first(magnitude_unequal),
        "candidate_negative_zero_at_unequal": int(np.count_nonzero(
            signed_zero_only
            & (np.ascontiguousarray(candidate).view(np.uint64)
               == np.uint64(0x8000000000000000)))),
        "oracle_negative_zero_at_unequal": int(np.count_nonzero(
            signed_zero_only
            & (np.ascontiguousarray(oracle).view(np.uint64)
               == np.uint64(0x8000000000000000)))),
    }


def one_ulp_sensitive_coefficient(trace, coefficients, active):
    """Return a one-cell/one-ULP coefficient plant visible at Coriolis output."""

    baseline_u, baseline_v, _ = strict_application(
        trace["u_mid"][0], trace["v_mid"][0], coefficients)
    for name in COEFFICIENTS:
        trial = dict(coefficients)
        values = np.asarray(coefficients[name])
        all_next = np.nextafter(values, np.float64(np.inf))
        trial[name] = all_next
        trial_u, trial_v, _ = strict_application(
            trace["u_mid"][0], trace["v_mid"][0], trial)
        face = "u" if name.startswith("ffu") else "v"
        before = baseline_u if face == "u" else baseline_v
        after = trial_u if face == "u" else trial_v
        moved = (before.view(np.uint64) != after.view(np.uint64)) & active[face]
        if not np.any(moved):
            continue
        j, i = np.argwhere(moved)[0]
        planted = dict(coefficients)
        array = np.array(values, copy=True)
        array[j, i] = np.nextafter(array[j, i], np.float64(np.inf))
        planted[name] = array
        check_u, check_v, _ = strict_application(
            trace["u_mid"][0], trace["v_mid"][0], planted)
        check = check_u if face == "u" else check_v
        require(check[j, i].view(np.uint64) != before[j, i].view(np.uint64),
                "selected one-ULP coefficient plant rounded away")
        return planted, {"name": name, "j": int(j), "i": int(i)}
    raise GateError("no active one-ULP coefficient mutation reaches Coriolis")


def assemble_oracle_coefficients(root: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Assemble validated owned-only coefficient groups from both ranks."""

    assembled = {
        name: np.empty((148, 180), dtype=np.float64) for name in COEFFICIENTS
    }
    coverage = np.zeros((148, 180), dtype=np.int8)
    records = []
    for expected_rank in (0, 1):
        path = root / (
            f"oracle_r98_een_coeff_rank{expected_rank:04d}_kt00000001.bin"
        )
        metadata = coefficient_record.read_record(path)
        require(metadata["rank"] == expected_rank, f"{path.name}: rank moved")
        raw = path.read_bytes()
        offset = 16 + coefficient_record.HEADER.size
        values = {}
        for _ in range(len(COEFFICIENTS)):
            name = raw[offset:offset + 16].decode("ascii").rstrip(" \x00")
            offset += 16
            ndim, n1, n2, n3 = coefficient_record.GROUP.unpack_from(raw, offset)
            offset += coefficient_record.GROUP.size
            require((ndim, n3) == (2, 1), f"{path.name}: {name} rank moved")
            count = n1 * n2
            values[name] = np.frombuffer(
                raw, dtype="=f8", count=count, offset=offset,
            ).copy().reshape((n1, n2), order="F")
            offset += 8 * count
        require(offset == len(raw), f"{path.name}: coefficient extraction missed EOF")
        require(set(values) == set(COEFFICIENTS),
                f"{path.name}: coefficient payload registry moved")

        nimpp, njmpp = metadata["origin"]
        ntsi, ntsj, ntei, ntej = metadata["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: owned placement moved")
        coverage[j0:j1, i0:i1] += 1
        for name, value in values.items():
            assembled[name][j0:j1, i0:i1] = r97._owned_block(
                value, ntsi, ntsj, ntei, ntej)
        records.append({key: metadata[key] for key in (
            "rank", "sha256", "bytes", "origin", "owned")})
    require(bool(np.all(coverage == 1)),
            "rank-owned coefficient slabs do not cover the domain exactly once")
    return assembled, {"coverage": "exactly-once", "records": records}


def assemble_oracle_accumulators(root: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Assemble admitted pre-scale accumulators and scales over both ranks."""

    assembled = {
        name: np.empty((148, 180), dtype=np.float64)
        for name in accumulator_record.FIELDS
    }
    coverage = np.zeros((148, 180), dtype=np.int8)
    records = []
    for expected_rank in (0, 1):
        path = root / (
            f"oracle_r104_een_accum_rank{expected_rank:04d}_kt00000001.bin"
        )
        row = accumulator_record.read_operand(path)
        require(row["rank"] == expected_rank, f"{path.name}: rank moved")
        nimpp, njmpp = row["origin"]
        ntsi, ntsj, ntei, ntej = row["owned"]
        i0, j0 = nimpp + ntsi - 4, njmpp + ntsj - 4
        i1, j1 = i0 + ntei - ntsi + 1, j0 + ntej - ntsj + 1
        require((j0, j1) == (0, 148) and 0 <= i0 < i1 <= 180,
                f"{path.name}: owned placement moved")
        coverage[j0:j1, i0:i1] += 1
        for name, value in row["groups"].items():
            assembled[name][j0:j1, i0:i1] = r97._owned_block(
                value, ntsi, ntsj, ntei, ntej)
        records.append({key: row[key] for key in (
            "rank", "sha256", "bytes", "origin", "owned")})
    require(bool(np.all(coverage == 1)),
            "rank-owned accumulator slabs do not cover the domain exactly once")
    return assembled, {"coverage": "exactly-once", "records": records}


def literal_een_scales(eta, z_coord, dtype, *, grid=None,
                       fold_metric: bool = False):
    """Evaluate only dynspg_ts' eight final EEN scale statements."""

    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round

    raw = z_coord.nemo_een_barotropic
    require(raw is not None, "literal EEN scale path has no carried operands")
    b = nemo_source_round
    one = jnp.asarray(1.0, dtype=dtype)
    half = jnp.asarray(0.5, dtype=dtype)
    leading = jnp.asarray(1.0 / 12.0, dtype=dtype)
    eta = jnp.asarray(eta, dtype=dtype)
    hu0 = jnp.asarray(raw.hu_0, dtype=dtype)
    hv0 = jnp.asarray(raw.hv_0, dtype=dtype)
    wet_u = (hu0 > 0.0).astype(dtype)
    wet_v = (hv0 > 0.0).astype(dtype)
    r1_hu0 = b(wet_u / b(hu0 + one - wet_u))
    r1_hv0 = b(wet_v / b(hv0 + one - wet_v))
    e1t = jnp.asarray(raw.e1t, dtype=dtype)
    e2t = jnp.asarray(raw.e2t, dtype=dtype)
    e1u = jnp.asarray(raw.e1u, dtype=dtype)
    e2u = jnp.asarray(raw.e2u, dtype=dtype)
    e1v = jnp.asarray(raw.e1v, dtype=dtype)
    e2v = jnp.asarray(raw.e2v, dtype=dtype)
    area_eta = b(b(e1t * e2t) * eta)
    east = jnp.roll(area_eta, -1, axis=1)
    north = jnp.roll(area_eta, -1, axis=0)
    r3u = b(b(half * b(area_eta + east)) * r1_hu0 / b(e1u * e2u))
    r3v = b(b(half * b(area_eta + north)) * r1_hv0 / b(e1v * e2v))
    r1_hu = b(r1_hu0 / b(one + r3u))
    r1_hv = b(r1_hv0 / b(one + r3v))

    def shift(value, di=0, dj=0):
        out = jnp.roll(value, di, axis=1) if di else value
        return jnp.roll(out, dj, axis=0) if dj else out

    def scale(local_metric, reciprocal_depth, neighbor_metric):
        return b(b(b(leading * b(one / local_metric)) * reciprocal_depth)
                 * neighbor_metric)

    north_e2u = shift(e2u, 0, -1)
    if fold_metric:
        from legoesm.grids.operators_latlon_cgrid import (
            fold_ghost_source_T,
            fold_perm_u,
        )
        from legoesm.ocean.dynamics.latlon_cgrid_operators import fold_row

        fold = getattr(grid, "fold", None)
        require(fold is not None and fold.is_active,
                "fold-scale arm requires an active tripolar fold")
        north = fold_row(
            fold_ghost_source_T(e2u, fold), fold_perm_u(fold), 1.0,
            fold.perm_T.shape[0])
        north_e2u = north_e2u.at[-1].set(north[0])

    return {
        "scl_u_nw": scale(e1u, r1_hu, e1v),
        "scl_u_ne": scale(e1u, r1_hu, shift(e1v, -1, 0)),
        "scl_u_sw": scale(e1u, r1_hu, shift(e1v, 0, 1)),
        "scl_u_se": scale(e1u, r1_hu, shift(e1v, -1, 1)),
        "scl_v_nw": scale(e2v, r1_hv, shift(north_e2u, 1, 0)),
        "scl_v_ne": scale(e2v, r1_hv, north_e2u),
        "scl_v_sw": scale(e2v, r1_hv, shift(e2u, 1, 0)),
        "scl_v_se": scale(e2v, r1_hv, e2u),
    }


def measure(deck_root: Path, frame_root: Path, spg_root: Path,
            expect_commit: str, *, plant: str,
            coefficient_root: Path | None = None,
            accumulator_root: Path | None = None) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_een_coefficients,
        barotropic_substeps_latlon_cgrid,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_dynvor_e3f_0vor,
        nemo_qco_live_vorticity_e3f_cgrid,
    )

    require(plant in PLANTS, f"unknown plant {plant}")
    stamp = worktree_stamp()
    require(stamp["clean"], "round-98 measurement worktree is dirty")
    require(stamp["commit"].lower() == expect_commit.lower(),
            "round-98 commit stamp mismatch")
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "fp64/libm policy is not active")
    require(jax.default_backend() == "cpu" and not jax.config.jax_disable_jit,
            "round-98 walk requires production JIT on CPU")

    oracle, census = r97.assemble_record(spg_root)
    card = rung0.build_rung0_card(deck_root)
    rung0.validate_rung0_card(card)
    state = rung0.bridge_entry(card, rung0.assemble_frame(frame_root, 1, 0))
    freshwater, surface = rhs_walk._forcing(state.eta.data.shape)
    masks = phase3_gate.expected_masks(card)
    active = {
        "u": np.asarray(masks["u"][..., 0], dtype=bool),
        "v": np.asarray(masks["v"][..., 0], dtype=bool),
    }

    trace_model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(expose_barotropic_substeps=True),
    )
    passive = jax.device_get(trace_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    slow = (r97._to_model_u(oracle["i000_zu_frc"]),
            r97._to_model_v(oracle["i000_zv_frc"]))
    raw_history = (
        r97._to_model_u(oracle["i000_ub_e"]),
        r97._to_model_u(oracle["i000_ubb_e"]),
        r97._to_model_v(oracle["i000_vb_e"]),
        r97._to_model_v(oracle["i000_vbb_e"]),
        oracle["i000_sshb_e"], oracle["i000_sshbb_e"],
    )

    raw = card.recipe.z_coord.nemo_een_barotropic
    require(raw is not None, "literal EEN path has no carried operands")
    e3t_0 = compute_layer_thickness(
        jnp.zeros_like(state.eta.data), state.H_bathy.data,
        card.recipe.z_coord,
        min_water_column_m=card.recipe.model_config.min_water_column_m,
    )
    source_divisor = nemo_dynvor_e3f_0vor(
        e3t_0, card.recipe.z_coord.is_active, grid=card.recipe.grid,
        dtype=jnp.float64, substitute_e3f=raw.e3f_0)
    source_z = card.recipe.z_coord._replace(
        nemo_een_barotropic=raw._replace(e3f_0=source_divisor))
    source_model = LatLonCGridOceanModel(
        card.recipe.grid, source_z, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            barotropic_slow_forcing_override=(slow[0], slow[1]),
            barotropic_raw_history_override=raw_history,
        ),
    )
    source_observed = jax.device_get(source_model.step(
        state, card.dt_s, freshwater=freshwater, surface_forcing=surface))
    source_trace = source_observed.substeps
    source_score = r97._coriolis_arm(source_trace, oracle, active)

    # The literal builder freezes at Kmm. On rung 0 this is the bridged state
    # eta; prove that by equality to all eight production-traced coefficients.
    coefficient_eta = jnp.asarray(state.eta.data, dtype=jnp.float64)
    source_coeff = jax.device_get(jax.jit(
        lambda eta: _nemo_literal_een_coefficients(
            eta, source_z, jnp.float64, scheme="een",
            grid=card.recipe.grid),
    )(coefficient_eta))
    coefficient_seed = {
        name: rhs_walk.score(np.asarray(source_coeff[name]),
                             np.asarray(source_trace[name][0]),
                             np.ones_like(np.asarray(source_coeff[name]), dtype=bool))
        for name in COEFFICIENTS
    }
    require(all(row["bit_exact"] for row in coefficient_seed.values()),
            "external literal coefficient seed is not the production Kmm seed: "
            + json.dumps(coefficient_seed, sort_keys=True))

    traced_coeff, current_parts = jax.device_get(jax.jit(
        lambda eta: _nemo_literal_een_coefficients(
            eta, source_z, jnp.float64, scheme="een",
            grid=card.recipe.grid, _return_een_parts=True),
    )(coefficient_eta))
    traced_identity = {
        name: rhs_walk.score(
            np.asarray(traced_coeff[name]), np.asarray(source_coeff[name]),
            np.ones_like(np.asarray(source_coeff[name]), dtype=bool))
        for name in COEFFICIENTS
    }
    require(all(row["bit_exact"] for row in traced_identity.values()),
            "accumulator trace changed a default coefficient")

    operand_discrimination = None
    accumulator_census = None
    fold_scale_arm = None
    direct_accumulator_trace = None
    if accumulator_root is not None:
        require(coefficient_root is not None,
                "accumulator discrimination requires final coefficients")
        oracle_operands, accumulator_census = assemble_oracle_accumulators(
            accumulator_root)
        current_scales = jax.device_get(jax.jit(
            lambda value: literal_een_scales(
                value, source_z, jnp.float64, grid=card.recipe.grid),
        )(coefficient_eta))
        if plant == "scale-bit":
            planted = np.array(current_scales["scl_u_nw"], copy=True)
            planted[1, 49] = np.nextafter(
                planted[1, 49], np.float64(np.inf))
            current_scales = dict(current_scales, scl_u_nw=planted)

        oracle_coeff, _ = assemble_oracle_coefficients(coefficient_root)
        if plant == "accumulator-bit":
            planted = np.array(current_parts["acc_u_nw"], copy=True)
            planted[1, 49] = np.nextafter(
                planted[1, 49], np.float64(np.inf))
            current_parts = dict(current_parts, acc_u_nw=planted)
        candidate_scales = jax.device_get(jax.jit(
            lambda value: literal_een_scales(
                value, source_z, jnp.float64, grid=card.recipe.grid,
                fold_metric=True),
        )(coefficient_eta))
        candidate_coeff = {}
        scale_rows = {}
        operand_discrimination = {}
        direct_accumulator_trace = {}
        for acc_name, scl_name, final_name in zip(
                accumulator_record.ACC, accumulator_record.SCL,
                accumulator_record.FINAL):
            scale = np.asarray(current_scales[scl_name])
            oracle_scale = oracle_operands[scl_name]
            oracle_acc = oracle_operands[acc_name]
            current_final = np.asarray(source_coeff[final_name])
            final_reference = oracle_coeff[final_name]
            final_bits = current_final.view(np.uint64) != final_reference.view(np.uint64)
            nonfold_final_bits = final_bits.copy()
            nonfold_final_bits[-1] = False
            scale_bits = scale.view(np.uint64) != oracle_scale.view(np.uint64)
            rebuilt = scale * oracle_acc
            rebuilt_bits = rebuilt.view(np.uint64) != final_reference.view(np.uint64)
            candidate_scale = np.asarray(candidate_scales[scl_name])
            candidate_scale_bits = (
                candidate_scale.view(np.uint64)
                != oracle_scale.view(np.uint64))
            candidate_coeff[final_name] = candidate_scale * oracle_acc
            candidate_coeff_bits = (
                candidate_coeff[final_name].view(np.uint64)
                != final_reference.view(np.uint64))
            current_acc = np.asarray(current_parts[acc_name])
            current_part_scale = np.asarray(current_parts[scl_name])
            acc_bits = current_acc.view(np.uint64) != oracle_acc.view(np.uint64)
            acc_magnitude = current_acc != oracle_acc
            scale_only = oracle_scale * current_acc
            acc_only = current_part_scale * oracle_acc
            direct_accumulator_trace[final_name] = {
                "accumulator_bit_unequal": int(np.count_nonzero(acc_bits)),
                "accumulator_magnitude_unequal": int(
                    np.count_nonzero(acc_magnitude)),
                "accumulator_nonfold_magnitude_unequal": int(
                    np.count_nonzero(acc_magnitude[:-1])),
                "accumulator_fold_magnitude_unequal": int(
                    np.count_nonzero(acc_magnitude[-1])),
                "scale_only_final_magnitude_unequal": int(
                    np.count_nonzero(scale_only != final_reference)),
                "accumulator_only_final_magnitude_unequal": int(
                    np.count_nonzero(acc_only != final_reference)),
                "both_nemo_final_bit_unequal": int(np.count_nonzero(
                    (oracle_scale * oracle_acc).view(np.uint64)
                    != final_reference.view(np.uint64))),
            }
            scale_rows[scl_name] = {
                "bit_unequal": int(np.count_nonzero(candidate_scale_bits)),
                "nonfold_bit_unequal": int(
                    np.count_nonzero(candidate_scale_bits[:-1])),
            }
            operand_discrimination[final_name] = {
                "current_scale_bit_unequal": int(np.count_nonzero(scale_bits)),
                "current_scale_nonfold_bit_unequal": int(
                    np.count_nonzero(scale_bits[:-1])),
                "scale_bit_unequal_at_nonfold_final_mismatch": int(
                    np.count_nonzero(scale_bits & nonfold_final_bits)),
                "nonfold_final_bit_unequal": int(
                    np.count_nonzero(nonfold_final_bits)),
                "rebuilt_with_nemo_acc_bit_unequal": int(
                    np.count_nonzero(rebuilt_bits)),
                "rebuilt_with_nemo_acc_nonfold_bit_unequal": int(
                    np.count_nonzero(rebuilt_bits[:-1])),
                "nemo_acc_negative_zero_at_nonfold_final_mismatch": int(
                    np.count_nonzero(
                        nonfold_final_bits
                        & (oracle_acc.view(np.uint64)
                           == np.uint64(0x8000000000000000)))),
                "fold_arm_coefficient_bit_unequal": int(
                    np.count_nonzero(candidate_coeff_bits)),
                "fold_arm_coefficient_magnitude_unequal": int(
                    np.count_nonzero(candidate_coeff[final_name]
                                     != final_reference)),
            }
        fold_scale_arm = {
            "scale_rows": scale_rows,
            "application": score_application(
                source_trace, candidate_coeff, oracle, active),
        }
        if plant == "scale-bit":
            require(any(row["current_scale_bit_unequal"] > 0
                        for row in operand_discrimination.values()),
                    "scale-bit plant stayed green")
            raise GateError("scale-bit plant fired")
        if plant == "accumulator-bit":
            require(any(row["accumulator_bit_unequal"] > 0
                        for row in direct_accumulator_trace.values()),
                    "accumulator-bit plant stayed green")
            raise GateError("accumulator-bit plant fired")

    oracle_coefficient_census = None
    oracle_coefficient_identity = None
    oracle_coefficient_difference_census = None
    oracle_coefficient_application = None
    signed_zero_coefficient_application = None
    fold_coefficient_application = None
    discriminator = "UNMEASURED_NO_RANK_COMPLETE_ORACLE_COEFFICIENTS"
    if coefficient_root is not None:
        oracle_coeff, oracle_coefficient_census = assemble_oracle_coefficients(
            coefficient_root)
        oracle_coefficient_identity = {
            name: rhs_walk.score(
                np.asarray(source_coeff[name]), oracle_coeff[name],
                np.ones_like(oracle_coeff[name], dtype=bool),
            )
            for name in COEFFICIENTS
        }
        oracle_coefficient_difference_census = {
            name: coefficient_difference_census(
                np.asarray(source_coeff[name]), oracle_coeff[name])
            for name in COEFFICIENTS
        }
        oracle_coefficient_application = score_application(
            source_trace, oracle_coeff, oracle, active)
        signed_zero_coeff = {}
        fold_coeff = {}
        for name in COEFFICIENTS:
            source = np.asarray(source_coeff[name])
            reference = oracle_coeff[name]
            signed_zero_coeff[name] = np.where(
                (source == 0.0) & (reference == 0.0), reference, source)
            fold_coeff[name] = np.array(source, copy=True)
            fold_coeff[name][-1] = reference[-1]
        signed_zero_coefficient_application = score_application(
            source_trace, signed_zero_coeff, oracle, active)
        fold_coefficient_application = score_application(
            source_trace, fold_coeff, oracle, active)
        discriminator = (
            "APPLICATION_AFTER_BIT_EXACT_COEFFICIENTS"
            if all(row["bit_exact"] for row in oracle_coefficient_identity.values())
            else "COEFFICIENT_CONSTRUCTION"
        )

    source_application = score_application(
        source_trace, source_coeff, oracle, active,
        plant="application-bit" if plant == "application-bit" else "none")
    if plant == "application-bit":
        require(not source_application["rows"][0]["bit_exact"],
                "application-bit plant stayed green")
        raise GateError("application-bit plant fired")

    live_source = nemo_qco_live_vorticity_e3f_cgrid(
        coefficient_eta, source_z, jnp.float64, grid=card.recipe.grid,
        e3t_0=card.recipe.z_coord.nemo_e3t_0,
        tmask=card.recipe.z_coord.is_active,
        reference_e3f=source_divisor,
    )[1:, 1:, :]
    require(live_source.shape == raw.e3f_0.shape,
            "fold-aware live source divisor does not have native A2D shape")

    # Recreate only the current literal builder's live-F divisor so the arm
    # can replace its northern fold row and leave every non-fold value fixed.
    b = jax.lax.optimization_barrier
    one = jnp.asarray(1.0, dtype=jnp.float64)
    half = jnp.asarray(0.5, dtype=jnp.float64)
    quarter = jnp.asarray(0.25, dtype=jnp.float64)
    eta = jnp.asarray(coefficient_eta, dtype=jnp.float64)
    area_eta = b(b(jnp.asarray(raw.e1t) * jnp.asarray(raw.e2t)) * eta)
    east = jnp.roll(area_eta, -1, axis=1)
    north = jnp.roll(area_eta, -1, axis=0)
    northeast = jnp.roll(north, -1, axis=1)
    wet_f = (jnp.asarray(raw.hf_0) > 0.0).astype(jnp.float64)
    r1_hf0 = b(wet_f / b(jnp.asarray(raw.hf_0) + one - wet_f))
    quad = b(b(area_eta + east) + b(north + northeast))
    r3f = b(b(quarter * quad) * r1_hf0
            / b(jnp.asarray(raw.e1f) * jnp.asarray(raw.e2f)))
    current_live = b(source_divisor * b(
        one + r3f[..., None] * jnp.asarray(raw.fmask)))
    fold_only = current_live.at[-1].set(live_source[-1])
    divisor_movement = {
        "full_source_vs_current": rhs_walk.score(
            np.asarray(live_source), np.asarray(current_live),
            np.ones_like(np.asarray(current_live), dtype=bool)),
        "fold_only_vs_current": rhs_walk.score(
            np.asarray(fold_only), np.asarray(current_live),
            np.ones_like(np.asarray(current_live), dtype=bool)),
        "full_source_nonfold_unequal": int(np.count_nonzero(
            np.asarray(live_source[:-1]) != np.asarray(current_live[:-1]))),
    }

    # The registered fold replacement is a no-op: every live divisor value,
    # including the northern row, is already bit-identical.  Reuse the source
    # coefficients rather than adding a test hook to the production model.
    require(divisor_movement["fold_only_vs_current"]["bit_exact"],
            "fold-only divisor unexpectedly moved")
    fold_coeff = source_coeff
    if plant == "coefficient-bit":
        fold_coeff, coefficient_plant = one_ulp_sensitive_coefficient(
            source_trace, fold_coeff, active)
        raise GateError(
            "coefficient-bit plant fired at "
            f"{coefficient_plant['name']}[{coefficient_plant['j']},"
            f"{coefficient_plant['i']}]")

    pre = {
        "coefficient_evaluation": "nemo_literal",
        "literal_coefficients": fold_coeff,
        "coefficient_eta": coefficient_eta,
        "scheme": "een",
    }
    substep_dt = float(oracle["i000_entry_sc"][0])
    substep_count = int(oracle["i000_entry_sc"][2])

    def candidate_run(seed, f_eta, f_u, f_v):
        return barotropic_substeps_latlon_cgrid(
            seed, substep_dt, substep_count,
            card.recipe.grid, source_z, card.recipe.model_config,
            F_slow_eta=f_eta, F_slow_u=f_u, F_slow_v=f_v,
            add_barotropic_coriolis=True,
            u_now=seed.u.data, v_now=seed.v.data,
            een_pre_override=pre,
            _nemo_raw_history_test_override=raw_history,
            _nemo_substep_trace_test_hook=True,
        )

    _, _, fold_trace = jax.device_get(jax.jit(candidate_run)(
        state, passive.slow_forcing[0], slow[0], slow[1]))
    fold_score = r97._coriolis_arm(fold_trace, oracle, active)

    movement = coefficient_movement(source_coeff, fold_coeff)
    return {
        "status": (
            "MEASURED_R99_EEN_COEFFICIENT_DISCRIMINATOR"
            if coefficient_root is not None
            else "MEASURED_R98_CORIOLIS_RESIDUAL"
        ),
        "claim_label": "independent",
        "execution": "production-jit-cpu-fp64-x64-libm",
        "record_census": census,
        "resolved_config": {
            "coefficient_evaluation": (
                card.recipe.model_config.barotropic
                .barotropic_een_coefficient_evaluation),
            "e3f_scheme": card.recipe.model_config.een_e3f_scheme,
            "barotropic_coriolis": (
                card.recipe.model_config.barotropic.barotropic_coriolis),
            "fold_active": bool(card.recipe.grid.fold.is_active),
        },
        "coefficient_seed_identity": coefficient_seed,
        "accumulator_census": accumulator_census,
        "operand_discrimination": operand_discrimination,
        "fold_scale_arm": fold_scale_arm,
        "direct_accumulator_trace": direct_accumulator_trace,
        "traced_coefficient_identity": traced_identity,
        "oracle_coefficient_census": oracle_coefficient_census,
        "oracle_coefficient_identity": oracle_coefficient_identity,
        "oracle_coefficient_difference_census": (
            oracle_coefficient_difference_census),
        "oracle_coefficient_application": oracle_coefficient_application,
        "signed_zero_coefficient_application": (
            signed_zero_coefficient_application),
        "fold_coefficient_application": fold_coefficient_application,
        "discriminator": discriminator,
        "source_divisor_score": source_score,
        "source_associated_application": source_application,
        "divisor_movement": divisor_movement,
        "fold_only_coefficient_movement": movement,
        "fold_only_score": fold_score,
        "worktree": stamp,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--frame-root", type=Path, required=True)
    parser.add_argument("--spg-root", type=Path, required=True)
    parser.add_argument("--coefficient-root", type=Path)
    parser.add_argument("--accumulator-root", type=Path)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", choices=PLANTS, default="none")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.deck_root, args.frame_root, args.spg_root,
                         args.expect_commit, plant=args.plant,
                         coefficient_root=args.coefficient_root,
                         accumulator_root=args.accumulator_root)
        require(args.plant == "none", f"{args.plant} plant stayed green")
    except (OSError, ValueError, GateError, rhs_walk.GateError) as error:
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
