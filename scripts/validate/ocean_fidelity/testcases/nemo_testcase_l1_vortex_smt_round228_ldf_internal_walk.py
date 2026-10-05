#!/usr/bin/env python3
"""Score Round 227's SMT-3 LDF internals through the production JIT step.

This is a thin extension of the Round-218 tracer-stage harness.  It reuses
that harness for the aggregate pre/post-LDF calibration and the shared
self-describing record parser for the new ldfslp/traldf_iso groups.  The
model's existing write-only ``tracer_ldf_diagnostics`` hook exposes the
values produced inside the ordinary production step; no isolated operator
closure is used for a claimed row.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "nemo_testcase_l1_vortex"))

from nemo_testcase_l1_vortex_kt2_walk import (  # noqa: E402
    _seed_from_record, _strip2, _strip3, _u_full, _v_full, read_bt_frame,
)
from nemo_testcase_l1_vortex_smt_round218_tracer_walk import (  # noqa: E402
    run as run_stage_walk,
)
from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    GateError, expected_masks, read_entry, require,
)

DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round228/"
    "oracle_vortex_smt3_ldf_internal")
CASE = "VORTEX_SMT3_VEC-zps"


def _checker():
    path = HERE / "nemo_testcase_l1_vortex" / "check_records.py"
    spec = importlib.util.spec_from_file_location("vortex_r228_checker", path)
    require(spec is not None and spec.loader is not None,
            "cannot load the shared self-describing parser")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_groups(path: Path) -> dict[str, np.ndarray]:
    """Parse a group record using only its self-described extents."""
    checker = _checker()
    checker.parse_record(path)
    return checker._read_group_values(path)


def _groups(path: Path) -> dict[str, np.ndarray]:
    """Parse and strip a group record using its own declared extents."""
    values = _raw_groups(path)
    out = {}
    for name, value in values.items():
        shape = value.shape
        flat = np.asarray(value).reshape(-1, order="F")
        if len(shape) == 3:
            out[name] = _strip3(flat, *shape)
        elif len(shape) == 2:
            out[name] = _strip2(flat, *shape)
        else:
            out[name] = flat.copy()
    return out


def _row(name, oracle, candidate, mask, *, plant: str | None):
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    use = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == use.shape,
            f"{name}: shape mismatch oracle={oracle.shape}, "
            f"candidate={candidate.shape}, mask={use.shape}")
    planted = plant == name
    if planted:
        locations = np.argwhere(use)
        require(locations.size > 0, f"{name}: plant support is empty")
        where = tuple(locations[len(locations) // 2])
        candidate = candidate.copy()
        candidate[where] = np.nextafter(candidate[where], np.inf)
    unequal = (candidate != oracle) & use
    delta = np.abs(candidate - oracle)
    peak = float(np.max(np.abs(oracle[use]))) if np.any(use) else 0.0
    return {
        "name": name,
        "cells_unequal": int(np.count_nonzero(unequal)),
        "max_abs": float(np.max(delta[use])) if np.any(use) else 0.0,
        "relative_max_abs": ((float(np.max(delta[use])) / peak)
                             if peak else 0.0),
        "bit_exact": not bool(np.any(unequal)),
        "planted": planted,
        "shape": list(oracle.shape),
        "active_cells": int(np.count_nonzero(use)),
        "dtype": str(candidate.dtype),
        "execution_regime": "production_step_jit",
    }


def run(root: Path, *, plant: str | None = None,
        corrected_factors: bool = False,
        divisor_arm: bool = False,
        same_stage_set: bool = False,
        include_live_faces: bool = False,
        pairwise_horizontal_flux: bool = False,
        literal_horizontal_flux: bool = False,
        allow_dirty: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "this walk must run on CPU")

    card = build_nemo_testcase_card(CASE)
    nlev = int(card.recipe.z_coord.n_levels)
    masks = expected_masks(card)
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    entry1 = read_entry(root / "oracle_step_entry_kt00000001.bin", CASE,
                        expect_interior=interior)
    entry2 = read_entry(root / "oracle_step_entry_kt00000002.bin", CASE,
                        expect_interior=interior)
    frame = read_bt_frame(root / "oracle_bt_frames_kt00000001.bin",
                          expect_step=1)
    external = (
        jnp.asarray(entry2["ssh"]),
        jnp.asarray(_u_full(frame["uu_b"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vv_b"][..., None])[..., 0]),
        jnp.asarray(_u_full(frame["un_adv"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vn_adv"][..., None])[..., 0]),
    )
    seed = _seed_from_record(card.recipe.initial_state, entry1, nlev)
    hooks = _NEMOWSRK3TestHooks(
        stage_barotropic_output_override=external,
        tracer_ldf_diagnostics="slope")
    trace = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks).step(seed, dt=card.dt_s)
    diagnostics = trace.ldf_diagnostics
    require(isinstance(diagnostics, dict) and "slope" in diagnostics,
            "production tracer step returned no LDF slope diagnostics")
    slope_diag = diagnostics["slope"]
    iso_diag = dict(diagnostics)
    iso_diag.update(diagnostics.get("zfw_operands", {}))
    print("live slope diagnostic keys:", sorted(slope_diag))
    print("live ISO diagnostic keys:", sorted(iso_diag))

    slope = _groups(root / "oracle_ldf_slope_kt00000001.bin")
    iso_path = root / "oracle_ldf_iso_kt00000001.bin"
    iso = _groups(iso_path)
    for family, groups in (("slope", slope), ("iso", iso)):
        print(f"{family} record groups={len(groups)}")
        for name in sorted(groups):
            print(f"  {name}: shape={groups[name].shape} "
                  f"dtype={groups[name].dtype}")

    rows = []
    def add(name, reference, candidate, support, *, skip_surface=False):
        support = np.asarray(support, dtype=bool).copy()
        if skip_surface:
            support[..., 0] = False
        rows.append(_row(name, np.asarray(reference)[..., :nlev],
                         np.asarray(candidate)[..., :nlev], support,
                         plant=plant))

    # Compiled ldfslp order: inputs -> raw face gradients -> bounded vertical
    # gradients -> mixed-layer raw slopes -> the four Shapiro-filtered outputs.
    slope_rows = (
        ("slope.prd", "prd", "prd", "T"),
        ("slope.pn2", "pn2", "pn2", "T"),
        ("slope.e3u", "e3u_kmm", "e3u_live", "u"),
        ("slope.e3v", "e3v_kmm", "e3v_live", "v"),
        ("slope.raw_u", "raw_u", "zau", "u"),
        ("slope.raw_v", "raw_v", "zav", "v"),
        ("slope.bound_u", "bound_u", "zbu_limited", "u"),
        ("slope.bound_v", "bound_v", "zbv_limited", "v"),
        ("slope.hraw_u", "hraw_u", "zwz", "u"),
        ("slope.hraw_v", "hraw_v", "zww", "v"),
        ("slope.uslp", "uslp", "uslp", "u"),
        ("slope.vslp", "vslp", "vslp", "v"),
        ("slope.wslpi", "wslpi", "wslpi", "T"),
        ("slope.wslpj", "wslpj", "wslpj", "T"),
    )
    for name, recorded, live, support in slope_rows:
        # ldfslp.f90's backward loop is jk=jpkm1..2.  The writer correctly
        # leaves its intermediate surface plane zero; it is not an executed
        # statement and therefore is excluded.  The final four arrays have a
        # defined zero surface and remain scored on the complete wet support.
        add(name, slope[recorded], slope_diag[live], masks[support],
            skip_surface=name not in ("slope.uslp", "slope.vslp",
                                      "slope.wslpi", "slope.wslpj"))

    # traldf_iso's compiled source order after its once-per-call A33 build.
    iso_rows = (
        ("iso.ah_wslp2", "ah_wslp2", "ah_wslp2_above", "T"),
        ("iso.akz", "akz", "akz_above", "T"),
        *((
            ("iso.tmask", "tmask", "tmask", "T"),
            ("iso.umask", "umask", "umask", "u"),
            ("iso.vmask", "vmask", "vmask", "v"),
            ("iso.wmask", "wmask", "wmask", "T"),
            ("iso.ahtu", "ahtu", "ahtu", "u"),
            ("iso.ahtv", "ahtv", "ahtv", "v"),
            ("iso.e3u_flux", "e3u_3d", "e3u_flux", "u"),
            ("iso.e3v_flux", "e3v_3d", "e3v_flux", "v"),
        ) if corrected_factors else ()),
        ("iso.dit", "dit", "dit", "u"),
        ("iso.djt", "djt", "djt", "v"),
        ("iso.dkt", "dkt", "dkt", "T"),
        *((
            ("iso.A11", "A11", "A11", "u"),
            ("iso.A22", "A22", "A22", "v"),
            ("iso.hmsku", "hmsku", "hmsku", "u"),
            ("iso.hmskv", "hmskv", "hmskv", "v"),
            ("iso.A13", "A13", "A13", "u"),
            ("iso.A23", "A23", "A23", "v"),
        ) if corrected_factors else ()),
        ("iso.fu", "fu", "zfu", "u"),
        ("iso.fv", "fv", "zfv", "v"),
        ("iso.vmsku", "vmsku", "vmsku", "T"),
        ("iso.vmskv", "vmskv", "vmskv", "T"),
        ("iso.ahu_w", "ahu_w", "ahu_w", "T"),
        ("iso.ahv_w", "ahv_w", "ahv_w", "T"),
        ("iso.A31", "A31", "A31", "T"),
        ("iso.A32", "A32", "A32", "T"),
        ("iso.fw_lower", "fw_lower", "zfw_top", "T"),
        ("iso.fw_upper", "fw_upper", "zfw_kp1", "T"),
        ("iso.rhs_increment", "rhs_increment", "tendency", "T"),
    )
    for name, recorded, live, support in iso_rows:
        add(name, iso[recorded], iso_diag[live], masks[support],
            skip_surface=name in ("iso.ah_wslp2", "iso.akz"))

    factor_reconstruction = []
    if corrected_factors:
        # Independent calibration of the corrected writer: rebuild the six
        # scalar factors on the full halo-bearing arrays, then strip them only
        # after evaluating NEMO's compiled source association.  This catches
        # the Round-227 defect where the writer copied one loop's final scalar
        # into every cell.
        raw = _raw_groups(iso_path)
        wmask = raw["wmask"]
        wm_ip1 = np.roll(wmask, -1, axis=0)
        wm_jp1 = np.roll(wmask, -1, axis=1)
        wm_kp1 = np.roll(wmask, -1, axis=2)
        hmsku = 1.0 / np.maximum(
            (wm_ip1 + wm_kp1) + (np.roll(wm_ip1, -1, axis=2) + wmask),
            1.0)
        hmskv = 1.0 / np.maximum(
            (wm_jp1 + wm_kp1) + (np.roll(wm_jp1, -1, axis=2) + wmask),
            1.0)
        A11 = (raw["e2_e1u"][..., None]
               * (raw["e3u_3d"]
                  * (1.0 + raw["r3u_kmm"][..., None] * raw["umask"])))
        A22 = (raw["e1_e2v"][..., None]
               * (raw["e3v_3d"]
                  * (1.0 + raw["r3v_kmm"][..., None] * raw["vmask"])))
        A13 = (-raw["e2u"][..., None] * raw["uslp"]) * hmsku
        A23 = (-raw["e1v"][..., None] * raw["vslp"]) * hmskv
        reconstructed = {
            "A11": A11, "A22": A22, "hmsku": hmsku, "hmskv": hmskv,
            "A13": A13, "A23": A23,
        }
        supports = {
            "A11": "u", "A22": "v", "hmsku": "u", "hmskv": "v",
            "A13": "u", "A23": "v",
        }
        for name in ("A11", "A22", "hmsku", "hmskv", "A13", "A23"):
            value = reconstructed[name]
            stripped = _strip3(
                value.reshape(-1, order="F"), *value.shape)
            factor_reconstruction.append(_row(
                f"record.{name}", iso[name][..., :nlev],
                stripped[..., :nlev], masks[supports[name]], plant=None))

    recorded_face_u = (
        np.asarray(iso["e3u_3d"])[..., :nlev]
        * (1.0 + np.asarray(iso["r3u_kmm"])[..., None]
           * np.asarray(iso["umask"])[..., :nlev]))
    recorded_face_v = (
        np.asarray(iso["e3v_3d"])[..., :nlev]
        * (1.0 + np.asarray(iso["r3v_kmm"])[..., None]
           * np.asarray(iso["vmask"])[..., :nlev]))
    face_thickness_arm = None
    if corrected_factors:
        # One-variable production-step arm through the existing fidelity hook:
        # replace only the two horizontal face-thickness arrays consumed by
        # zA11/zA22.  The ordinary full step still executes under _step_jitted;
        # this is not an isolated replay of the flux statement.
        override_hooks = _NEMOWSRK3TestHooks(
            stage_barotropic_output_override=external,
            tracer_ldf_diagnostics=(
                jnp.asarray(recorded_face_u),
                jnp.asarray(recorded_face_v)))
        override_trace = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=override_hooks).step(seed, dt=card.dt_s)
        override_diag = override_trace.ldf_diagnostics
        require(isinstance(override_diag, dict),
                "face-thickness production arm returned no LDF diagnostics")
        rhs_row = _row(
            "arm.face_thickness.rhs_increment",
            iso["rhs_increment"][..., :nlev],
            override_diag["tendency"][..., :nlev], masks["T"], plant=None)
        unchanged = []
        for name, support in (("A13", "u"), ("A23", "v"),
                              ("dit", "u"), ("djt", "v"),
                              ("dkt", "T")):
            unchanged.append(_row(
                f"arm.face_thickness.upstream.{name}", iso_diag[name],
                override_diag[name], masks[support], plant=None))
        baseline_rhs = next(
            row for row in rows if row["name"] == "iso.rhs_increment")
        face_thickness_arm = {
            "execution_regime": "production_step_jit",
            "rhs_row": rhs_row,
            "unchanged_rows": unchanged,
            "fraction_of_baseline_rhs_max_removed": (
                1.0 - rhs_row["max_abs"] / baseline_rhs["max_abs"]),
            "dt_scaled_residual_max": card.dt_s * rhs_row["max_abs"],
            "fraction_of_additional_ldf_max_remaining": (
                rhs_row["max_abs"] / baseline_rhs["max_abs"]),
        }

    # Post-hoc family split from the materialized production-JIT fluxes.  This
    # is not promoted to a production row: it only discriminates whether the
    # remaining RHS error is carried by the horizontal or vertical flux
    # family before the next round adds a one-variable production override.
    model_tendency = np.asarray(iso_diag["tendency"])[..., :nlev]
    area = np.asarray(iso_diag["e1e2t"])[..., None]
    thickness = np.asarray(iso_diag["e3t"])[..., :nlev]
    tmask = np.asarray(iso_diag["tmask"])[..., :nlev]
    delta_fu = (np.asarray(iso["fu"])[..., :nlev]
                - np.asarray(iso_diag["zfu"])[..., :nlev])
    delta_fv = (np.asarray(iso["fv"])[..., :nlev]
                - np.asarray(iso_diag["zfv"])[..., :nlev])
    delta_horizontal = (
        (delta_fu - np.roll(delta_fu, 1, axis=1))
        + (delta_fv - np.roll(delta_fv, 1, axis=0))) / area / thickness * tmask
    delta_vertical = (
        ((np.asarray(iso["fw_lower"])[..., :nlev]
          - np.asarray(iso_diag["zfw_top"])[..., :nlev])
         - (np.asarray(iso["fw_upper"])[..., :nlev]
            - np.asarray(iso_diag["zfw_kp1"])[..., :nlev]))
        / area / thickness * tmask)
    nemo_thickness = (
        np.asarray(iso["e3t_3d"])[..., :nlev]
        * (1.0 + np.asarray(iso["r3t_kmm"])[..., None]
           * np.asarray(iso["tmask"])[..., :nlev]))
    divisor_rows = [
        _row("iso.e3t_divisor", nemo_thickness, thickness,
             masks["T"], plant=None),
        _row("iso.r1_e1e2t", np.asarray(iso["r1_e1e2t"]),
             1.0 / np.asarray(iso_diag["e1e2t"]),
             np.asarray(masks["T"])[..., 0], plant=None),
    ]
    divisor_production_arm = None
    if divisor_arm:
        divisor_input = np.asarray(nemo_thickness).copy()
        divisor_plant_location = None
        if plant == "arm.live_divisor.input":
            support = np.asarray(masks["T"], dtype=bool)
            magnitude = np.where(
                support,
                np.abs(np.asarray(iso["rhs_increment"])[..., :nlev]), -1.0)
            divisor_plant_location = tuple(
                int(value) for value in np.unravel_index(
                    int(np.argmax(magnitude)), magnitude.shape))
            divisor_input[divisor_plant_location] = np.nextafter(
                divisor_input[divisor_plant_location], np.inf)
        hook = _NEMOWSRK3TestHooks(
            stage_barotropic_output_override=external,
            tracer_ldf_diagnostics={
                "divisor_thickness": jnp.asarray(divisor_input),
            })
        arm_trace = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=hook).step(seed, dt=card.dt_s)
        arm_diag = arm_trace.ldf_diagnostics
        require(isinstance(arm_diag, dict),
                "divisor production arm returned no LDF diagnostics")
        arm_row = _row(
            "arm.live_divisor.rhs_increment",
            iso["rhs_increment"][..., :nlev],
            arm_diag["tendency"][..., :nlev], masks["T"], plant=None)
        upstream = []
        for name, support in (("zfu", "u"), ("zfv", "v"),
                              ("zfw_kp1", "T"), ("zfw_top", "T")):
            upstream.append(_row(
                f"arm.live_divisor.upstream.{name}", iso_diag[name],
                arm_diag[name], masks[support], plant=None))
        divisor_production_arm = {
            "execution_regime": "production_step_jit",
            "rhs_row": arm_row,
            "upstream_rows": upstream,
            "fraction_of_baseline_rhs_max_removed": (
                1.0 - arm_row["max_abs"]
                / next(row for row in rows
                       if row["name"] == "iso.rhs_increment")["max_abs"]),
            "input_plant_location": divisor_plant_location,
            "tendency_sha256": hashlib.sha256(
                np.asarray(arm_diag["tendency"], dtype=np.float64)
                .tobytes(order="C")).hexdigest(),
        }

    same_stage_arms = None
    if same_stage_set:
        baseline_rhs = next(
            row for row in rows if row["name"] == "iso.rhs_increment")

        def run_same_stage_arm(name: str, *, use_divisor: bool,
                               use_faces: bool,
                               flux_evaluation: str | None = None) -> dict:
            divisor_input = np.asarray(nemo_thickness).copy()
            face_u = recorded_face_u.copy()
            face_v = recorded_face_v.copy()
            plant_location = None
            if plant == f"{name}.divisor_input":
                support = np.asarray(masks["T"], dtype=bool)
                magnitude = np.where(
                    support,
                    np.abs(np.asarray(iso["rhs_increment"])[..., :nlev]),
                    -1.0)
                plant_location = tuple(
                    int(value) for value in np.unravel_index(
                        int(np.argmax(magnitude)), magnitude.shape))
                divisor_input[plant_location] = np.nextafter(
                    divisor_input[plant_location], np.inf)
            elif plant == f"{name}.face_u_input":
                support = np.asarray(masks["u"], dtype=bool)
                magnitude = np.where(
                    support,
                    np.abs(np.asarray(iso["A11"])[..., :nlev]
                           * np.asarray(iso["dit"])[..., :nlev]), -1.0)
                plant_location = tuple(
                    int(value) for value in np.unravel_index(
                        int(np.argmax(magnitude)), magnitude.shape))
                face_u[plant_location] = np.nextafter(
                    face_u[plant_location], np.inf)
            elif plant == f"{name}.face_v_input":
                support = np.asarray(masks["v"], dtype=bool)
                magnitude = np.where(
                    support,
                    np.abs(np.asarray(iso["A22"])[..., :nlev]
                           * np.asarray(iso["djt"])[..., :nlev]), -1.0)
                plant_location = tuple(
                    int(value) for value in np.unravel_index(
                        int(np.argmax(magnitude)), magnitude.shape))
                face_v[plant_location] = np.nextafter(
                    face_v[plant_location], np.inf)

            operands = {"closed_bottom_wmask": True}
            if use_divisor:
                operands["divisor_thickness"] = jnp.asarray(divisor_input)
            if use_faces:
                operands["face_thickness"] = (
                    jnp.asarray(face_u), jnp.asarray(face_v))
            if flux_evaluation is not None:
                operands["horizontal_flux_evaluation"] = flux_evaluation
            hook = _NEMOWSRK3TestHooks(
                stage_barotropic_output_override=external,
                tracer_ldf_diagnostics=operands)
            arm_trace = LatLonCGridOceanModel(
                card.recipe.grid, card.recipe.z_coord,
                card.recipe.model_config,
                _nemo_ws_test_hooks=hook).step(seed, dt=card.dt_s)
            arm_diag = arm_trace.ldf_diagnostics
            require(isinstance(arm_diag, dict),
                    f"{name} returned no LDF diagnostics")
            source_rows = []
            for row_name, recorded, live, support in (
                    ("hmsku", "hmsku", "hmsku", "u"),
                    ("hmskv", "hmskv", "hmskv", "v"),
                    ("A11", "A11", "A11", "u"),
                    ("A22", "A22", "A22", "v"),
                    ("A13", "A13", "A13", "u"),
                    ("A23", "A23", "A23", "v"),
                    ("zfu", "fu", "zfu", "u"),
                    ("zfv", "fv", "zfv", "v")):
                source_rows.append(_row(
                    f"{name}.{row_name}", iso[recorded][..., :nlev],
                    arm_diag[live][..., :nlev], masks[support], plant=None))
            unchanged_rows = []
            for row_name, support in (("dit", "u"), ("djt", "v"),
                                      ("dkt", "T"), ("zfw_kp1", "T"),
                                      ("zfw_top", "T")):
                unchanged_rows.append(_row(
                    f"{name}.unchanged.{row_name}",
                    iso_diag[row_name][..., :nlev],
                    arm_diag[row_name][..., :nlev], masks[support],
                    plant=None))
            rhs_row = _row(
                f"{name}.rhs_increment",
                iso["rhs_increment"][..., :nlev],
                arm_diag["tendency"][..., :nlev], masks["T"], plant=None)
            return {
                "execution_regime": "production_step_jit",
                "uses_live_divisor": use_divisor,
                "uses_live_face_thickness": use_faces,
                "horizontal_flux_evaluation": flux_evaluation,
                "source_rows": source_rows,
                "unchanged_rows": unchanged_rows,
                "rhs_row": rhs_row,
                "fraction_of_baseline_rhs_max_removed": (
                    1.0 - rhs_row["max_abs"] / baseline_rhs["max_abs"]),
                "input_plant_location": plant_location,
                "tendency_sha256": hashlib.sha256(
                    np.asarray(arm_diag["tendency"], dtype=np.float64)
                    .tobytes(order="C")).hexdigest(),
            }

        mask_only = run_same_stage_arm(
            "arm.closed_bottom_wmask", use_divisor=False, use_faces=False)
        pair = run_same_stage_arm(
            "arm.pair", use_divisor=True, use_faces=False)
        triple = None
        if (include_live_faces
                or pair["fraction_of_baseline_rhs_max_removed"] < 0.9):
            triple = run_same_stage_arm(
                "arm.triple", use_divisor=True, use_faces=True)
        pairwise = None
        if pairwise_horizontal_flux:
            pairwise = run_same_stage_arm(
                "arm.pairwise_flux", use_divisor=True, use_faces=True,
                flux_evaluation="nemo_pairwise")
        literal = None
        if literal_horizontal_flux:
            literal = run_same_stage_arm(
                "arm.literal_flux", use_divisor=True, use_faces=True,
                flux_evaluation="nemo_literal")
        same_stage_arms = {
            "closed_bottom_wmask_control": mask_only,
            "pair": pair,
            "triple": triple,
            "pairwise_flux": pairwise,
            "literal_flux": literal,
        }
    oracle_fu = np.asarray(iso["fu"])[..., :nlev]
    oracle_fv = np.asarray(iso["fv"])[..., :nlev]
    oracle_hdiv = (
        (oracle_fu - np.roll(oracle_fu, 1, axis=1))
        + (oracle_fv - np.roll(oracle_fv, 1, axis=0)))
    oracle_vdiv = (
        np.asarray(iso["fw_lower"])[..., :nlev]
        - np.asarray(iso["fw_upper"])[..., :nlev])
    nemo_source_tendency = (
        (oracle_hdiv + oracle_vdiv)
        * np.asarray(iso["r1_e1e2t"])[..., None]
        / nemo_thickness)
    flux_family_reconstruction = []
    for name, candidate in (
            ("arm.horizontal_flux.rhs_increment",
             model_tendency + delta_horizontal),
            ("arm.vertical_flux.rhs_increment",
             model_tendency + delta_vertical),
            ("arm.all_fluxes.rhs_increment",
             model_tendency + delta_horizontal + delta_vertical),
            ("arm.nemo_flux_divisor.rhs_increment", nemo_source_tendency)):
        flux_family_reconstruction.append(_row(
            name, iso["rhs_increment"][..., :nlev], candidate,
            masks["T"], plant=None))

    # Post-hoc magnitude discriminator for the first genuine non-bit row.
    # Replace only the A33 coefficient in the already-materialized production
    # operands and rebuild the exact flux/divergence algebra below it.  This is
    # deliberately labelled a reconstructed one-variable bound, not another
    # production row.
    ah_model = np.asarray(iso_diag["ah_wslp2_above"])
    ah_oracle = np.asarray(iso["ah_wslp2"])[..., :nlev]
    delta_ah_below = np.roll(ah_oracle - ah_model, -1, axis=2)
    delta_flux_below = (
        np.asarray(iso_diag["e1e2t"])[..., None]
        / np.asarray(iso_diag["e3w_kp1"])
        * delta_ah_below * np.asarray(iso_diag["qdiff_kp1"])
        * np.asarray(iso_diag["act_below"])
    )
    delta_flux_above = np.roll(delta_flux_below, 1, axis=2)
    delta_flux_above[..., 0] = 0.0
    delta_rate = (
        (delta_flux_above - delta_flux_below)
        * np.asarray(iso_diag["e1e2t"])[..., None] ** -1
        / np.asarray(iso_diag["e3t"])
        * np.asarray(iso_diag["tmask"])
    )
    a33_rate_max = float(np.max(np.abs(delta_rate[masks["T"]])))
    a33_stage_max = card.dt_s * a33_rate_max

    aggregate = run_stage_walk(
        root, "smt3", allow_dirty=allow_dirty, stage=3)
    aggregate_rows = {
        row["name"].split(".", 1)[1]: row for row in aggregate["rows"]}
    pre = aggregate_rows["stage3.pre_ldf.T"]
    post = aggregate_rows["stage3.post_ldf.T"]
    first = next((row for row in rows if not row["bit_exact"]), None)
    report = {
        "case": CASE, "legoesm_git_sha": sha, "oracle_root": str(root),
        "plant": plant, "rows": rows,
        "record_defect": ({
            "invalid_groups": [],
            "reason": "R16 writer copies scalar factors inside their producing loop",
        } if corrected_factors else {
            "invalid_groups": ["A11", "A22", "A13", "A23", "hmsku",
                               "hmskv"],
            "reason": ("compiled writer lines 262-268 copy scalar "
                       "temporaries after their producing loop")}),
        "factor_reconstruction": factor_reconstruction,
        "face_thickness_arm": face_thickness_arm,
        "flux_family_reconstruction": flux_family_reconstruction,
        "divisor_rows": divisor_rows,
        "divisor_production_arm": divisor_production_arm,
        "same_stage_arms": same_stage_arms,
        "a33_one_variable_reconstruction": {
            "execution_regime": "post_hoc_from_production_operands",
            "max_abs_tendency_change": a33_rate_max,
            "dt_scaled_upper_bound": a33_stage_max,
            "fraction_of_additional_ldf_max": (
                a33_stage_max
                / (post["max_abs"] - pre["max_abs"]))},
        "first_non_bit": None if first is None else first["name"],
        "aggregate_reproduction": {
            "pre_ldf_max_abs": pre["max_abs"],
            "post_ldf_max_abs": post["max_abs"],
            "additional_ldf_max_abs": post["max_abs"] - pre["max_abs"],
        },
        "status": "PLANT" if plant else "MEASURED",
    }
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant")
    parser.add_argument("--corrected-factors", action="store_true")
    parser.add_argument("--divisor-arm", action="store_true")
    parser.add_argument("--same-stage-set", action="store_true")
    parser.add_argument("--include-live-faces", action="store_true")
    parser.add_argument("--pairwise-horizontal-flux", action="store_true")
    parser.add_argument("--literal-horizontal-flux", action="store_true")
    parser.add_argument("--clean-report", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    try:
        require(not args.include_live_faces or args.same_stage_set,
                "--include-live-faces requires --same-stage-set")
        require(not args.pairwise_horizontal_flux or args.same_stage_set,
                "--pairwise-horizontal-flux requires --same-stage-set")
        require(not args.literal_horizontal_flux or args.same_stage_set,
                "--literal-horizontal-flux requires --same-stage-set")
        report = run(args.oracle_dir, plant=args.plant,
                     corrected_factors=args.corrected_factors,
                     divisor_arm=args.divisor_arm,
                     same_stage_set=args.same_stage_set,
                     include_live_faces=args.include_live_faces,
                     pairwise_horizontal_flux=args.pairwise_horizontal_flux,
                     literal_horizontal_flux=args.literal_horizontal_flux,
                     allow_dirty=args.allow_dirty)
    except GateError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    for row in report["rows"]:
        print(f"{row['name']:22s} bit={str(row['bit_exact']):5s} "
              f"cells={row['cells_unequal']:7d} "
              f"max_abs={row['max_abs']:.16e}")
    for row in report["factor_reconstruction"]:
        print(f"{row['name']:22s} bit={str(row['bit_exact']):5s} "
              f"cells={row['cells_unequal']:7d} "
              f"max_abs={row['max_abs']:.16e}")
    if report["face_thickness_arm"] is not None:
        arm = report["face_thickness_arm"]
        row = arm["rhs_row"]
        print(f"{row['name']:22s} bit={str(row['bit_exact']):5s} "
              f"cells={row['cells_unequal']:7d} "
              f"max_abs={row['max_abs']:.16e}")
        print("face-thickness fraction removed:",
              f"{arm['fraction_of_baseline_rhs_max_removed']:.16e}")
        print("face-thickness remaining LDF fraction:",
              f"{arm['fraction_of_additional_ldf_max_remaining']:.16e}")
    for row in report["flux_family_reconstruction"]:
        print(f"{row['name']:35s} bit={str(row['bit_exact']):5s} "
              f"cells={row['cells_unequal']:7d} "
              f"max_abs={row['max_abs']:.16e}")
    for row in report["divisor_rows"]:
        print(f"{row['name']:35s} bit={str(row['bit_exact']):5s} "
              f"cells={row['cells_unequal']:7d} "
              f"max_abs={row['max_abs']:.16e}")
    if report["divisor_production_arm"] is not None:
        arm = report["divisor_production_arm"]
        for row in [*arm["upstream_rows"], arm["rhs_row"]]:
            print(f"{row['name']:35s} bit={str(row['bit_exact']):5s} "
                  f"cells={row['cells_unequal']:7d} "
                  f"max_abs={row['max_abs']:.16e}")
        print("live-divisor fraction removed:",
              f"{arm['fraction_of_baseline_rhs_max_removed']:.16e}")
    if report["same_stage_arms"] is not None:
        for arm_name in ("closed_bottom_wmask_control", "pair", "triple",
                         "pairwise_flux", "literal_flux"):
            arm = report["same_stage_arms"][arm_name]
            if arm is None:
                continue
            for row in [*arm["source_rows"], *arm["unchanged_rows"],
                        arm["rhs_row"]]:
                print(f"{row['name']:35s} bit={str(row['bit_exact']):5s} "
                      f"cells={row['cells_unequal']:7d} "
                      f"max_abs={row['max_abs']:.16e}")
            print(f"{arm_name} fraction removed:",
                  f"{arm['fraction_of_baseline_rhs_max_removed']:.16e}")
    print(json.dumps(report["aggregate_reproduction"], sort_keys=True))
    print("first non-bit:", report["first_non_bit"])
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True)
                               + "\n")
    if args.plant:
        require(args.clean_report is not None,
                "a plant requires --clean-report")
        clean = json.loads(args.clean_report.read_text())
        if args.plant == "arm.live_divisor.input":
            require(report["divisor_production_arm"] is not None,
                    "divisor input plant requires --divisor-arm")
            clean_arm = clean.get("divisor_production_arm")
            require(clean_arm is not None,
                    "clean report has no divisor production arm")
            fired = (report["divisor_production_arm"]["tendency_sha256"]
                     != clean_arm["tendency_sha256"])
            print(f"STATUS {'PLANT-FIRED' if fired else 'PLANT-MISSED'}")
            return 1 if fired else 0
        planted_arm = next((name for name in (
            "pair", "triple", "pairwise_flux", "literal_flux")
            if args.plant.startswith(f"arm.{name}.")), None)
        if planted_arm is not None:
            require(report["same_stage_arms"] is not None,
                    "same-stage input plant requires --same-stage-set")
            arm = report["same_stage_arms"][planted_arm]
            clean_arm = clean.get("same_stage_arms", {}).get(planted_arm)
            require(arm is not None and clean_arm is not None,
                    f"clean report has no {planted_arm} arm")
            fired = arm["tendency_sha256"] != clean_arm["tendency_sha256"]
            print(f"STATUS {'PLANT-FIRED' if fired else 'PLANT-MISSED'}")
            return 1 if fired else 0
        before = {row["name"]: row for row in clean["rows"]}
        planted = [row for row in report["rows"] if row["planted"]]
        require(len(planted) == 1, "plant did not select exactly one row")
        row = planted[0]
        visible = (row["name"] in before and
                   (row["cells_unequal"], row["max_abs"])
                   != (before[row["name"]]["cells_unequal"],
                       before[row["name"]]["max_abs"]))
        print(f"STATUS {'PLANT-FIRED' if visible else 'PLANT-MISSED'}")
        return 1 if visible else 0
    print("STATUS MEASURED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
