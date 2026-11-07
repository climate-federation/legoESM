#!/usr/bin/env python3
"""Source-ordered GYRE stage-1 tracer walk (production JIT, fp64)."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

from nemo_testcase_l2_gyre_phase3_gate import (
    CASE,
    DIMS,
    _surface_forcings,
    expected_masks,
    lego_fields,
    read_entry,
    require,
    sha256,
)
from nemo_testcase_state_ulp_probe import ulp_distance
from legoesm.ocean.fidelity.provenance import worktree_stamp


ORACLE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round13_oracle_tracer_v1"
)
CONTROL_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round12_oracle_eos_v2"
)


def _xyz(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def _xy(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx, ny), order="F")[2:-2, 2:-2].T


def read_stage1_tracer(path: Path) -> dict:
    """Read the WRITE-only stage-1 Krhs/update stream in source order."""
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    require(time_level_for_dump(path.name) == "now", f"{path}: wrong registry level")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_RKTRA_1", f"{path}: bad magic {magic!r}")
    require(
        (version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits)
        == (1, 1, 1, 1, 1, 3, 3, *DIMS, 64),
        f"{path}: bad header {header}",
    )
    n3 = nx * ny * nz
    n2 = nx * ny
    require(values.size == 15 * n3 + 3 * n2, f"{path}: bad payload")
    names = (
        "zero_T", "zero_S", "zFu", "zFv", "zFw",
        "after_advection_T", "after_advection_S",
        "after_sbc_T", "after_sbc_S",
        "Kbb_T", "Kbb_S", "Kmm_T", "Kmm_S", "Kaa_T", "Kaa_S",
    )
    result = {
        name: _xyz(values[index * n3:(index + 1) * n3], nx, ny, nz)
        for index, name in enumerate(names)
    }
    offset = 15 * n3
    for index, name in enumerate(("r3t_Kbb", "r3t_Kmm", "r3t_Kaa")):
        begin = offset + index * n2
        result[name] = _xy(values[begin:begin + n2], nx, ny)
    for name, field in result.items():
        if name != "header":
            require(np.all(np.isfinite(field)),
                    f"{path}: non-finite scored interior in {name}")
    result["discarded_nonfinite_halo_values"] = int(
        np.count_nonzero(~np.isfinite(values)))
    result["header"] = {
        "version": version, "kt": kt, "stage": stage, "Kbb": kbb,
        "Kmm": kmm, "Krhs": krhs, "Kaa": kaa, "bits": bits,
        "registry_level": "now",
    }
    return result


def read_stage1_transport_operands(path: Path) -> dict:
    """Read the WRITE-only stage-1 horizontal-transport operands."""
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    require(time_level_for_dump(path.name) == "now", f"{path}: wrong registry level")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kmm, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_TRPOP_2", f"{path}: bad magic {magic!r}")
    require(
        (version, kt, stage, kmm, nx, ny, nz, bits)
        == (2, 1, 1, 1, *DIMS, 64),
        f"{path}: bad header {header}",
    )
    n2, n3 = nx * ny, nx * ny * nz
    require(values.size == 10 * n2 + 8 * n3, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    result = {}
    offset = 0
    for name, size, transform in (
        ("e2u", n2, lambda x: _xy(x, nx, ny)),
        ("e3u", n3, lambda x: _xyz(x, nx, ny, nz)),
        ("uu", n3, lambda x: _xyz(x, nx, ny, nz)),
        ("zub", n2, lambda x: _xy(x, nx, ny)),
        ("umask", n3, lambda x: _xyz(x, nx, ny, nz)),
        ("zFu", n3, lambda x: _xyz(x, nx, ny, nz)),
        ("e1v", n2, lambda x: _xy(x, nx, ny)),
        ("e3v", n3, lambda x: _xyz(x, nx, ny, nz)),
        ("vv", n3, lambda x: _xyz(x, nx, ny, nz)),
        ("zvb", n2, lambda x: _xy(x, nx, ny)),
        ("vmask", n3, lambda x: _xyz(x, nx, ny, nz)),
        ("zFv", n3, lambda x: _xyz(x, nx, ny, nz)),
        ("un_adv", n2, lambda x: _xy(x, nx, ny)),
        ("r1_hu", n2, lambda x: _xy(x, nx, ny)),
        ("uu_b", n2, lambda x: _xy(x, nx, ny)),
        ("vn_adv", n2, lambda x: _xy(x, nx, ny)),
        ("r1_hv", n2, lambda x: _xy(x, nx, ny)),
        ("vv_b", n2, lambda x: _xy(x, nx, ny)),
    ):
        result[name] = transform(values[offset:offset + size])
        offset += size
    result["header"] = {
        "version": version, "kt": kt, "stage": stage, "Kmm": kmm,
        "bits": bits, "registry_level": "now",
    }
    return result


def _comparison(
    candidate: np.ndarray, oracle: np.ndarray, active: np.ndarray | None = None,
) -> dict:
    candidate, oracle = np.asarray(candidate), np.asarray(oracle)
    if active is not None:
        candidate, oracle = candidate[active], oracle[active]
    delta = candidate - oracle
    return {
        "bit_exact": bool(np.array_equal(candidate, oracle)),
        "absolute_max": float(np.max(np.abs(delta), initial=0.0)),
        "ulp_max": int(np.max(ulp_distance(candidate, oracle), initial=0)),
        "differing_cells": int(np.count_nonzero(candidate != oracle)),
    }


def run_transport_operands(oracle_root: Path, *, plant: bool = False) -> dict:
    """Discriminate the literal zFu/zFv product association from its inputs."""
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.core.source_rounding import nemo_source_round

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    path = oracle_root / "oracle_rkstage1_transport_operands_kt00000001.bin"
    require(path.is_file(), f"missing {path}")
    oracle = read_stage1_transport_operands(path)

    def _literal(e_metric, e3, vel, mean, mask):
        b = nemo_source_round
        corrected = b(vel + b(mean[..., None] * mask))
        metric_thickness = b(e_metric[..., None] * e3)
        return b(metric_thickness * corrected)

    literal_jit = jax.jit(_literal)
    rows = {}
    for face, names in {
        "u": ("e2u", "e3u", "uu", "zub", "umask", "zFu"),
        "v": ("e1v", "e3v", "vv", "zvb", "vmask", "zFv"),
    }.items():
        metric, e3, vel, mean, mask, result = (oracle[name] for name in names)
        # NEMO's jpk is the all-zero extra level and the DO_3D statement writes
        # only jpkm1.  Score the live wet faces, which are the operands consumed
        # by tra_adv_cen; retain the full arrays in the artifact hash/header.
        e3, vel, mask, result = (
            value[..., :-1] for value in (e3, vel, mask, result))
        active = mask != 0.0
        numpy_corrected = vel + mean[..., None] * mask
        numpy_literal = (metric[..., None] * e3) * numpy_corrected
        numpy_reassociated = metric[..., None] * (e3 * numpy_corrected)
        jax_literal = np.asarray(literal_jit(
            jnp.asarray(metric), jnp.asarray(e3), jnp.asarray(vel),
            jnp.asarray(mean), jnp.asarray(mask)))
        if plant and face == "u":
            jax_literal = jax_literal.copy()
            index = tuple(np.argwhere(mask != 0.0)[0])
            jax_literal[index] = np.nextafter(jax_literal[index], np.inf)
        rows[face] = {
            "numpy_fortran_association_vs_oracle": _comparison(
                numpy_literal, result, active),
            "numpy_reassociated_vs_oracle": _comparison(
                numpy_reassociated, result, active),
            "jax_jit_fortran_association_vs_oracle": _comparison(
                jax_literal, result, active),
            "jax_jit_vs_numpy_fortran_association": _comparison(
                jax_literal, numpy_literal, active),
        }
    exact = all(
        row["jax_jit_fortran_association_vs_oracle"]["bit_exact"]
        for row in rows.values())
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round13-transport-operands-v1",
        "status": "AT-BAR" if exact else "DEBT",
        "regime": "production-jit-cpu-fp64-x64",
        "oracle_root": str(oracle_root),
        "oracle_dump_sha256": sha256(path),
        "oracle_header": oracle["header"],
        "rows": rows,
        "owner_label": (
            "CONFIRMED_ZF_PRODUCT_ASSOCIATION" if exact
            else "UNMEASURED_WITHIN_ZF_PRODUCT"),
        "plant": plant,
    }


def run_transport_candidate(oracle_root: Path, *, plant: bool = False) -> dict:
    """Score the production-JIT stage-1 transport after operand construction."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    path = oracle_root / "oracle_rkstage1_transport_operands_kt00000001.bin"
    oracle = read_stage1_transport_operands(path)
    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    freshwater, surface = _surface_forcings(card, card.recipe.initial_state, 1)
    def _step(hooks):
        model = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            _nemo_ws_test_hooks=hooks)
        return model.step(
            card.recipe.initial_state, dt=card.dt_s,
            freshwater=freshwater, surface_forcing=surface)

    state = _step(_NEMOWSRK3TestHooks(expose_tracer_transport_stage=1))
    legacy_state = _step(_NEMOWSRK3TestHooks(
        expose_tracer_transport_stage=1,
        legacy_reduced_stage_transport_mean_arm=True))
    thickness_state = _step(_NEMOWSRK3TestHooks(
        expose_stage1_transport_operand="thickness"))
    corrected_state = _step(_NEMOWSRK3TestHooks(
        expose_stage1_transport_operand="corrected_velocity"))
    legacy_corrected_state = _step(_NEMOWSRK3TestHooks(
        expose_stage1_transport_operand="corrected_velocity",
        legacy_reduced_stage_transport_mean_arm=True))
    average_state = _step(_NEMOWSRK3TestHooks(
        expose_stage1_transport_operand="transport_average"))
    nlev = card.recipe.z_coord.n_levels
    candidate = {
        "u": np.asarray(state.u.data)[:, 1:, :nlev],
        "v": np.asarray(state.v.data)[1:, :, :nlev],
    }
    legacy_candidate = {
        "u": np.asarray(legacy_state.u.data)[:, 1:, :nlev],
        "v": np.asarray(legacy_state.v.data)[1:, :, :nlev],
    }
    native = {
        "u": np.asarray(card.recipe.initial_state.u.data)[:, 1:, :nlev],
        "v": np.asarray(card.recipe.initial_state.v.data)[1:, :, :nlev],
    }
    thickness = {
        "u": np.asarray(thickness_state.u.data)[:, 1:, :nlev],
        "v": np.asarray(thickness_state.v.data)[1:, :, :nlev],
    }
    corrected = {
        "u": np.asarray(corrected_state.u.data)[:, 1:, :nlev],
        "v": np.asarray(corrected_state.v.data)[1:, :, :nlev],
    }
    legacy_corrected = {
        "u": np.asarray(legacy_corrected_state.u.data)[:, 1:, :nlev],
        "v": np.asarray(legacy_corrected_state.v.data)[1:, :, :nlev],
    }
    average = {
        "u": np.asarray(average_state.u.data)[:, 1:, 0],
        "v": np.asarray(average_state.v.data)[1:, :, 0],
    }
    masks = expected_masks(card)
    metric = {
        "u": np.asarray(card.recipe.grid.dy_u)[:, 1:],
        "v": np.asarray(card.recipe.grid.dx_v)[1:, :],
    }
    rows = {}
    for face, name in (("u", "zFu"), ("v", "zFv")):
        oracle_mask = oracle[f"{face}mask"][..., :nlev]
        active = oracle_mask != 0.0
        if plant and face == "u":
            candidate[face] = candidate[face].copy()
            index = tuple(np.argwhere(active)[0])
            candidate[face][index] = np.nextafter(candidate[face][index], np.inf)
        oracle_corrected = (
            oracle[f"{face}{face}"][..., :nlev]
            + oracle[f"z{face}b"][..., None] * oracle_mask)
        active2 = np.any(active, axis=-1)
        rows[face] = {
            "metric": _comparison(
                metric[face], oracle["e2u" if face == "u" else "e1v"],
                active2),
            "mask": _comparison(
                masks[face].astype(np.float64), oracle_mask, active),
            "Kmm_velocity": _comparison(
                native[face], oracle[f"{face}{face}"][..., :nlev], active),
            "Kmm_face_thickness": _comparison(
                thickness[face], oracle[f"e3{face}"][..., :nlev], active),
            "corrected_velocity": _comparison(
                corrected[face], oracle_corrected, active),
            "legacy_reduced_corrected_velocity_arm": _comparison(
                legacy_corrected[face], oracle_corrected, active),
            "transport_average": _comparison(
                average[face],
                oracle["un_adv" if face == "u" else "vn_adv"], active2),
            "zF": _comparison(
                candidate[face], oracle[name][..., :nlev], active),
            "legacy_reduced_zF_arm": _comparison(
                legacy_candidate[face], oracle[name][..., :nlev], active),
        }
    exact = all(row["zF"]["bit_exact"] for row in rows.values())
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round13-transport-candidate-v1",
        "status": "AT-BAR" if exact else "DEBT",
        "regime": "production-jit-cpu-fp64-x64",
        "oracle_root": str(oracle_root),
        "oracle_dump_sha256": sha256(path),
        "rows": rows,
        "plant": plant,
    }


def _bits(value: np.float64) -> str:
    return f"0x{int(np.asarray(value, dtype=np.float64).view(np.uint64)):016x}"


def _cell_value(oracle, candidate, index) -> dict:
    ov = np.float64(oracle[index])
    cv = np.float64(candidate[index])
    return {
        "oracle": float(ov),
        "candidate": float(cv),
        "oracle_bits": _bits(ov),
        "candidate_bits": _bits(cv),
        "absolute_delta": float(cv - ov),
        "ulp": int(ulp_distance(np.asarray([cv]), np.asarray([ov]))[0]),
        "bit_exact": bool(cv == ov),
    }


def _run_boundary(
    card, cfg, boundary: str, transport_override=None,
    *, legacy_reduced_mean: bool = False,
):
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )

    hook_args = {
        "stage1_tracer_transport_override": transport_override,
        "legacy_reduced_stage_transport_mean_arm": legacy_reduced_mean,
    }
    if boundary == "after_update":
        hook_args["expose_tracer_stage"] = 1
    else:
        hook_args["expose_tracer_stage1_boundary"] = boundary
    hooks = _NEMOWSRK3TestHooks(**hook_args)
    freshwater, surface = _surface_forcings(card, card.recipe.initial_state, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=hooks,
    )
    state = model.step(
        card.recipe.initial_state, dt=card.dt_s,
        freshwater=freshwater, surface_forcing=surface,
    )
    host = jax.tree_util.tree_map(
        lambda x: np.asarray(x) if isinstance(x, (jax.Array, np.ndarray)) else x,
        state,
    )
    jax.clear_caches()
    fields = lego_fields(host)
    return fields["T"], fields["S"]


def run(oracle_root: Path, control_root: Path, *, plant: bool = False) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")

    tracer_path = oracle_root / "oracle_rktracer_operands_kt00000001_s1.bin"
    require(tracer_path.is_file(), f"missing {tracer_path}")
    oracle = read_stage1_tracer(tracer_path)
    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    masks = expected_masks(card)
    active = masks["T"]
    nlev = card.recipe.z_coord.n_levels

    candidate = {
        "after_advection": _run_boundary(card, cfg, "after_advection"),
        "after_sbc": _run_boundary(card, cfg, "after_sbc"),
        "after_update": _run_boundary(card, cfg, "after_update"),
    }
    candidate = {
        boundary: (np.asarray(pair[0])[..., :nlev], np.asarray(pair[1])[..., :nlev])
        for boundary, pair in candidate.items()
    }
    oracle_transport = tuple(
        oracle[name][..., :nlev] if name != "zFw" else oracle[name]
        for name in ("zFu", "zFv", "zFw")
    )
    transport_arm = {
        boundary: _run_boundary(
            card, cfg, boundary, transport_override=oracle_transport)
        for boundary in ("after_advection", "after_update")
    }
    transport_arm = {
        boundary: (
            np.asarray(pair[0])[..., :nlev],
            np.asarray(pair[1])[..., :nlev],
        )
        for boundary, pair in transport_arm.items()
    }
    legacy_mean_pair = _run_boundary(
        card, cfg, "after_update", legacy_reduced_mean=True)
    legacy_mean_arm = (
        np.asarray(legacy_mean_pair[0])[..., :nlev],
        np.asarray(legacy_mean_pair[1])[..., :nlev],
    )
    if plant:
        planted = candidate["after_update"][0].copy()
        oracle_update = oracle["Kaa_T"][..., :nlev]
        equal_wet = active & (planted == oracle_update)
        require(bool(equal_wet.any()), "plant has no bit-identical wet target")
        index = tuple(np.argwhere(equal_wet)[0])
        planted[index] = np.nextafter(planted[index], np.inf)
        candidate["after_update"] = (planted, candidate["after_update"][1])

    # The newly instrumented oracle must not perturb ordinary prognostics.
    bit_identity = {}
    for name in (
        "oracle_stage_kt00000001_s1.bin",
        "oracle_stage_kt00000001_s2.bin",
        "oracle_stage_kt00000001_s3.bin",
        "GYRE_OMIP_L2_P3_00000010_restart.nc",
    ):
        measured = oracle_root / name
        control = control_root / name
        require(measured.is_file() and control.is_file(), f"missing identity control {name}")
        measured_hash, control_hash = sha256(measured), sha256(control)
        require(measured_hash == control_hash, f"instrument changed {name}")
        bit_identity[name] = measured_hash

    entry = read_entry(oracle_root / "oracle_step_entry_kt00000001.bin")
    initial = {
        "T": np.asarray(card.recipe.initial_state.T.data)[..., :nlev],
        "S": np.asarray(card.recipe.initial_state.S.data)[..., :nlev],
    }
    initial_rows = {}
    for field in ("T", "S"):
        same = np.asarray(entry[field][..., :nlev]) == initial[field]
        initial_rows[field] = {
            "bit_exact": bool(np.all(same[active])),
            "differing_wet_cells": int(np.count_nonzero(~same & active)),
        }
        require(initial_rows[field]["bit_exact"], f"initial {field} is not bit-exact")

    field_index = {"T": 0, "S": 1}
    differing = []
    for field in ("T", "S"):
        oi = oracle[f"Kaa_{field}"][..., :nlev]
        ci = candidate["after_update"][field_index[field]]
        for j, i, k in np.argwhere((oi != ci) & active):
            differing.append((field, int(j), int(i), int(k)))
    distinct_locations = sorted({(j, i, k) for _, j, i, k in differing})
    field_counts = {
        field: sum(1 for found, *_ in differing if found == field)
        for field in ("T", "S")
    }
    bottom_k = np.sum(active, axis=-1) - 1
    wet2 = np.any(active, axis=-1)
    cells = []
    boundaries = (
        ("zero_rhs", "zero"),
        ("after_advection", "after_advection"),
        ("after_sbc", "after_sbc"),
        ("after_update", "Kaa"),
    )
    for j, i, k in distinct_locations:
        idx = (j, i, k)
        field_values = {}
        for field in ("T", "S"):
            fi = field_index[field]
            if (field, j, i, k) not in differing:
                continue
            values = {}
            first = None
            for boundary, oracle_prefix in boundaries:
                if boundary == "zero_rhs":
                    ov = oracle[f"zero_{field}"][..., :nlev]
                    cv = np.zeros_like(ov)
                elif boundary == "after_update":
                    ov = oracle[f"Kaa_{field}"][..., :nlev]
                    cv = candidate[boundary][fi]
                else:
                    ov = oracle[f"{oracle_prefix}_{field}"][..., :nlev]
                    cv = candidate[boundary][fi]
                values[boundary] = _cell_value(ov, cv, idx)
                if first is None and not values[boundary]["bit_exact"]:
                    first = boundary
            field_values[field] = {
                "first_differing_boundary": first,
                "values": values,
            }
        coast = any(
            not wet2[jj, ii]
            for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1))
            if 0 <= jj < wet2.shape[0] and 0 <= ii < wet2.shape[1]
        )
        cells.append({
            "nemo_ijk_1based": [i + 3, j + 3, k + 1],
            "legoesm_jik_0based": [j, i, k],
            "level": k + 1,
            "surface": k == 0,
            "bottom": k == int(bottom_k[j, i]),
            "coast_adjacent_at_T_level": coast,
            "differs_at_kt1_entry": False,
            "fields": field_values,
        })

    aggregate = {}
    for boundary, oracle_prefix in boundaries[1:]:
        aggregate[boundary] = {}
        for field in ("T", "S"):
            fi = field_index[field]
            op = "Kaa" if boundary == "after_update" else oracle_prefix
            ov = oracle[f"{op}_{field}"][..., :nlev]
            cv = candidate[boundary][fi]
            use_o, use_c = ov[active], cv[active]
            aggregate[boundary][field] = {
                "absolute_max": float(np.max(np.abs(use_c - use_o))),
                "ulp_max": int(ulp_distance(use_c, use_o).max(initial=0)),
                "differing_wet_cells": int(np.count_nonzero(use_c != use_o)),
            }

    scaling = {}
    for boundary, oracle_prefix in (
            ("after_advection", "after_advection"), ("after_update", "Kaa")):
        scaling[boundary] = {}
        for field in ("T", "S"):
            fi = field_index[field]
            ov = oracle[f"{oracle_prefix}_{field}"][..., :nlev][active]
            faithful = candidate[boundary][fi][active]
            arm = transport_arm[boundary][fi][active]
            residual = float(np.max(np.abs(faithful - ov)))
            movement = float(np.max(np.abs(arm - faithful)))
            arm_residual = float(np.max(np.abs(arm - ov)))
            scaling[boundary][field] = {
                "faithful_residual": residual,
                "oracle_transport_arm_movement": movement,
                "oracle_transport_arm_residual": arm_residual,
                "movement_over_faithful_residual": (
                    movement / residual if residual else None),
                "arm_differing_wet_cells": int(np.count_nonzero(arm != ov)),
            }

    update_scaling = [scaling["after_update"][field] for field in ("T", "S")]
    clears = all(row["oracle_transport_arm_residual"] == 0.0 for row in update_scaling)
    residual_scale = all(
        (row["faithful_residual"] == 0.0
         and row["oracle_transport_arm_movement"] == 0.0)
        or (row["movement_over_faithful_residual"] is not None
            and 0.9 <= row["movement_over_faithful_residual"] <= 1.1)
        for row in update_scaling
    )
    owner_label = (
        "CONFIRMED_STAGE1_TRACER_TRANSPORT_OPERAND"
        if clears and residual_scale
        else "REFUTED_STAGE1_TRACER_TRANSPORT_OPERAND"
    )
    mean_scaling = {}
    for field in ("T", "S"):
        fi = field_index[field]
        ov = oracle[f"Kaa_{field}"][..., :nlev][active]
        faithful = candidate["after_update"][fi][active]
        arm = legacy_mean_arm[fi][active]
        residual = float(np.max(np.abs(faithful - ov)))
        movement = float(np.max(np.abs(arm - faithful)))
        arm_residual = float(np.max(np.abs(arm - ov)))
        mean_scaling[field] = {
            "faithful_residual": residual,
            "legacy_reduced_mean_arm_movement": movement,
            "legacy_reduced_mean_arm_residual": arm_residual,
            "movement_over_faithful_residual": (
                movement / residual if residual else None),
            "arm_differing_wet_cells": int(np.count_nonzero(arm != ov)),
        }
    mean_clears = all(
        row["faithful_residual"] == 0.0 for row in mean_scaling.values())
    mean_scale = any(
        row["legacy_reduced_mean_arm_movement"] > 0.0
        and row["legacy_reduced_mean_arm_residual"] > 0.0
        for row in mean_scaling.values())

    first_boundaries = sorted({
        values["first_differing_boundary"]
        for cell in cells for values in cell["fields"].values()
    })
    report = {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round13-tracer-v1",
        "status": "AT-BAR" if not differing else "DEBT",
        "regime": "production-jit-cpu-fp64-x64",
        "oracle_root": str(oracle_root),
        "oracle_dump_sha256": sha256(tracer_path),
        "oracle_header": oracle["header"],
        "instrument_bit_identity": bit_identity,
        "initial_entry": initial_rows,
        "differing_location_count": len(distinct_locations),
        "differing_field_entry_count": len(differing),
        "differing_cells_by_field": field_counts,
        "briefing_count_disposition": (
            "the preregistered baseline has nine per field and disjoint T/S "
            "coordinate sets; post-arm counts are measured, not forced"
        ),
        "cells": cells,
        "aggregate": aggregate,
        "scaling_before_owner": scaling,
        "stored_barotropic_mean_scaling_before_owner": mean_scaling,
        "first_differing_boundaries": first_boundaries,
        "owner_label": owner_label,
        "transport_internal_owner_label": (
            "CONFIRMED_STORED_BAROTROPIC_MEAN_ASSOCIATION"
            if mean_clears and mean_scale
            else "REFUTED_STORED_BAROTROPIC_MEAN_AS_SOLE_OWNER"
        ),
        "source_dispositions": {
            "stage1_transcendentals": "REFUTED_BY_EXECUTED_SOURCE",
            "stage1_fct_limiter": "REFUTED_BY_EXECUTED_SOURCE",
            "tra_qsr_tra_ldf_tra_zdf": "STAGE3_ONLY",
            "tra_atf": "MLF_ONLY_NOT_CALLED_BY_STPRK3_STG",
        },
        "plant": plant,
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--control-root", type=Path, default=CONTROL_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument(
        "--mode", choices=("tracer", "transport-operands", "transport-candidate"),
        default="tracer")
    args = parser.parse_args(argv)
    try:
        if args.mode == "transport-operands":
            report = run_transport_operands(args.oracle_root, plant=args.plant)
        elif args.mode == "transport-candidate":
            report = run_transport_candidate(args.oracle_root, plant=args.plant)
        else:
            report = run(args.oracle_root, args.control_root, plant=args.plant)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
