#!/usr/bin/env python3
"""Low-memory, one-static-variant GYRE stage-3 completion gate.

This is a process-split companion to the lane-2 phase-3 gate.  It reuses that
gate's readers, masks, forcing transcription and immutable fp64 score; each
invocation compiles exactly one diagnostic variant so the 32x22x30 GYRE graph
does not exhaust host memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

import numpy as np

from nemo_testcase_l2_gyre_phase3_gate import (
    BAR,
    BIT_IDENTITY_EXPECTED,
    CASE,
    STAGE3_ROOT,
    STAGE2_ROOT,
    _registered,
    _surface_forcings,
    expected_masks,
    lego_fields,
    read_stage,
    read_stage2_terms,
    read_tracer_stage3,
    read_transport,
    require,
    score,
    sha256,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp

POST_TRA_ADV_TRP_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "gyre_kt1_10_stage3_posttrp")
STAGE2_ENE_OPERAND_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "gyre_kt1_10_stage2_ene_operands")
STAGE2_HPG_OPERAND_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "gyre_kt1_10_stage2_hpg_operands")
STAGE3_WZV_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "gyre_kt1_10_stage3_wzv_v7")


def read_stage2_ene_operands(path: Path) -> dict[str, np.ndarray | int]:
    """Read stage-2 ``vor_ene`` operands written once per vertical slab."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        version, kt, kmm, nx, ny, nz, bits = header
        require(magic == "NEMO_L2_ENEOP_1", f"{path}: bad magic {magic!r}")
        require(
            (version, kt, kmm, nx, ny, nz, bits) ==
            (1, 1, 3, 36, 26, 31, 64),
            f"{path}: bad header {header}")
        rows = {name: [] for name in ("zwz", "zwx", "zwy")}
        count = nx * ny
        for level in range(nz - 1):
            for name in rows:
                values = np.fromfile(handle, dtype=np.float64, count=count)
                require(values.size == count, f"{path}: truncated {name} k={level + 1}")
                full = values.reshape((nx, ny), order="F").T
                owned = full[2:-2, 2:-2]
                require(np.all(np.isfinite(owned)), f"{path}: non-finite owned {name}")
                rows[name].append(owned)
        require(handle.read(1) == b"", f"{path}: trailing payload")
    return {
        "kt": kt,
        "Kmm": kmm,
        **{name: np.stack(slabs, axis=-1) for name, slabs in rows.items()},
    }


def read_stage2_hpg_operands(path: Path) -> dict[str, np.ndarray | int]:
    """Read the stage-2 ``hpg_sco`` operands at its executing call site."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=7i", handle.read(28))
        version, kt, kmm, nx, ny, nz, bits = header
        require(magic == "NEMO_L2_HPGOP_1", f"{path}: bad magic {magic!r}")
        require(
            (version, kt, kmm, nx, ny, nz, bits) ==
            (1, 1, 3, 36, 26, 31, 64),
            f"{path}: bad header {header}")
        rows = {name: [] for name in ("rhd", "e3w", "gdept_z0")}
        count = nx * ny
        for level in range(nz):
            for name in rows:
                values = np.fromfile(handle, dtype=np.float64, count=count)
                require(values.size == count, f"{path}: truncated {name} k={level + 1}")
                full = values.reshape((nx, ny), order="F").T
                owned = full[2:-2, 2:-2]
                require(np.all(np.isfinite(owned)), f"{path}: non-finite owned {name}")
                rows[name].append(owned)
        require(handle.read(1) == b"", f"{path}: trailing payload")
    return {
        "kt": kt,
        "Kmm": kmm,
        **{name: np.stack(slabs, axis=-1) for name, slabs in rows.items()},
    }


def read_stage3_wzv(path: Path) -> dict[str, np.ndarray | int | str]:
    """Read the live stage-3 ``wzv -> wAimp -> pFw`` WRITE-only split."""
    level = _registered(path, "now")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kmm, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_WZVOP_1", f"{path}: bad magic {magic!r}")
    require(
        (version, kt, stage, kmm, nx, ny, nz, bits)
        == (1, 1, 3, 2, 36, 26, 31, 64),
        f"{path}: bad header {header}",
    )
    count = nx * ny * nz
    # GYRE resolves ln_zad_Aimp=F, so NEMO's ``wi`` allocatable has zero
    # extent and the WRITE list contributes no payload for it.  Keep the
    # logical boundary explicit as a zero array in the parsed record.
    require(values.size == 3 * count, f"{path}: bad payload")
    names = ("ww_pre_aimp", "ww_post_aimp", "pFw")
    result = {
        "kt": kt,
        "stage": stage,
        "Kmm": kmm,
        "registry_level": level,
        **{
            name: values[index * count:(index + 1) * count]
            .reshape((nx, ny, nz), order="F")[2:-2, 2:-2]
            .transpose(1, 0, 2)
            for index, name in enumerate(names)
        },
    }
    result["wi_post_aimp"] = np.zeros_like(result["ww_post_aimp"])
    return result


def _oracle_pre_zdf(record: dict, nlev: int, dt: float) -> dict[str, np.ndarray]:
    r3bb = record["r3t_Kbb"][..., None]
    r3mm = record["r3t_Kmm"][..., None]
    r3aa = record["r3t_Kaa"][..., None]
    return {
        name: np.asarray((
            (1.0 + r3bb) * record[f"Kbb_{name}"]
            + dt * (1.0 + r3mm) * record[f"after_ldf_{name}"]
        ) / (1.0 + r3aa))[..., :nlev]
        for name in ("T", "S")
    }


def _oracle_content_rhs(
    record: dict, e3t_0: np.ndarray, nlev: int, dt: float,
) -> dict[str, np.ndarray]:
    """Literal ``trazdf.F90:271-278`` right-hand-side content."""
    e3bb = e3t_0 * (1.0 + record["r3t_Kbb"][..., None])
    e3mm = e3t_0 * (1.0 + record["r3t_Kmm"][..., None])
    return {
        name: (
            e3bb * record[f"Kbb_{name}"][..., :nlev]
            + dt * e3mm * record[f"after_ldf_{name}"][..., :nlev]
        )
        for name in ("T", "S")
    }


def _oracle_advection_content(
    record: dict, e3t_0: np.ndarray, nlev: int, dt: float,
) -> dict[str, np.ndarray]:
    e3bb = e3t_0 * (1.0 + record["r3t_Kbb"][..., None])
    e3mm = e3t_0 * (1.0 + record["r3t_Kmm"][..., None])
    return {
        name: (
            e3bb * record[f"Kbb_{name}"][..., :nlev]
            + dt * e3mm * record[f"after_advection_{name}"][..., :nlev]
        )
        for name in ("T", "S")
    }


def run(mode: str, output_npz: Path, faithful_npz: Path | None,
        plant: bool = False, oracle_root: Path | None = None) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
        _nemo_ws_qco_stage_faces,
    )
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        neumann_fill_cgrid,
        nemo_qco_wzv_operands,
    )
    from legoesm.ocean.dynamics.latlon_cgrid_operators import (
        compute_face_masks_3d,
        curl_vertex_cgrid,
        pv_flux_ene,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from legoesm.ocean.vertical import (
        compute_layer_thickness,
        nemo_wicker_aimp_partition_transport,
        nemo_qco_live_vorticity_e3f_cgrid,
        nemo_qco_vorticity_f_cgrid,
    )
    from legoesm import constants
    from legoesm.ocean.eos import (
        nemo_r3t_stretch,
        nemo_teos10_density_anomaly_ratio,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit),
            "certification requires production JIT; JAX_DISABLE_JIT is forbidden")
    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    masks = expected_masks(card)
    nlev = card.recipe.z_coord.n_levels
    stage3_root = oracle_root or STAGE3_ROOT
    post_transport_root = oracle_root or POST_TRA_ADV_TRP_ROOT
    stage2_ene_root = oracle_root or STAGE2_ENE_OPERAND_ROOT
    stage2_hpg_root = oracle_root or STAGE2_HPG_OPERAND_ROOT
    stage3_wzv_root = oracle_root or STAGE3_WZV_ROOT
    stage2_root = oracle_root or STAGE2_ROOT
    tracer_path = stage3_root / "oracle_rktracer_stage3_kt00000001.bin"
    stage_path = stage3_root / "oracle_stage_kt00000001_s3.bin"
    kt2_path = stage3_root / "oracle_step_entry_kt00000002.bin"
    transport_path = (
        post_transport_root / "oracle_tracer_transport_kt00000001_s3.bin")
    ene_operand_path = (
        stage2_ene_root / "oracle_rkstage2_ene_operands_kt00000001.bin")
    hpg_operand_path = (
        stage2_hpg_root / "oracle_rkstage2_hpg_operands_kt00000001.bin")
    wzv_path = stage3_wzv_root / "oracle_rkstage3_wzv_kt00000001.bin"
    for path in (
            tracer_path, stage_path, kt2_path, transport_path,
            ene_operand_path, hpg_operand_path, wzv_path):
        require(path.is_file(), f"missing {path}")
    require(
        sha256(stage_path) == BIT_IDENTITY_EXPECTED[stage_path.name],
        f"{stage_path}: WRITE-only instrumentation changed stage 3")
    require(
        sha256(kt2_path) == BIT_IDENTITY_EXPECTED[kt2_path.name],
        f"{kt2_path}: WRITE-only instrumentation changed kt=2 entry")
    record = read_tracer_stage3(tracer_path)
    stage3 = read_stage(stage_path, 3)
    transport3 = read_transport(transport_path, 3)
    stage2_terms = read_stage2_terms(
        stage2_root / "oracle_rkstage2_terms_kt00000001.bin")
    stage2 = read_stage(
        stage3_root / "oracle_stage_kt00000001_s2.bin", 2)
    ene_operands = read_stage2_ene_operands(ene_operand_path)
    hpg_operands = read_stage2_hpg_operands(hpg_operand_path)
    wzv_record = read_stage3_wzv(wzv_path)
    oracle_pre = _oracle_pre_zdf(record, nlev, card.dt_s)
    oracle_content = _oracle_content_rhs(
        record, np.asarray(card.recipe.z_coord.nemo_e3t_0), nlev, card.dt_s)
    oracle_advection_content = _oracle_advection_content(
        record, np.asarray(card.recipe.z_coord.nemo_e3t_0), nlev, card.dt_s)
    freshwater, surface = _surface_forcings(card, card.recipe.initial_state, 1)

    hooks = _NEMOWSRK3TestHooks()
    if mode == "pre":
        hooks = hooks._replace(expose_pre_implicit_state=True)
    elif mode == "content":
        hooks = hooks._replace(expose_pre_implicit_content=True)
    elif mode == "advection_content":
        hooks = hooks._replace(expose_stage3_advection_content=True)
    elif mode == "transport":
        hooks = hooks._replace(expose_tracer_transport_stage=3)
    elif mode == "transport_zub_arm":
        hooks = hooks._replace(
            expose_tracer_transport_stage=3,
            legacy_reduced_stage_transport_mean_arm=True)
    elif mode == "transport_wzv_legacy":
        hooks = hooks._replace(
            expose_tracer_transport_stage=3,
            legacy_wzv_rederived_transport=True)
    elif mode == "transport_arm":
        hooks = hooks._replace(
            expose_stage3_advection_content=True,
            stage3_transport_override=(
                jnp.asarray(transport3["zFu"][..., :nlev]),
                jnp.asarray(transport3["zFv"][..., :nlev]),
                jnp.asarray(transport3["zFw"]),
            ),
        )
    elif mode in ("stage2_rhs_arm", "stage2_rhs_transport_arm"):
        rhs_u = jnp.asarray(stage2_terms["after_advection_u"])[..., :nlev]
        rhs_v = jnp.asarray(stage2_terms["after_advection_v"])[..., :nlev]
        oracle_rhs_u = jnp.concatenate([
            rhs_u[:, -1:, :], rhs_u,
        ], axis=1)
        oracle_rhs_v = jnp.concatenate([
            jnp.zeros_like(rhs_v[:1]), rhs_v,
        ], axis=0)
        hooks = hooks._replace(
            stage2_momentum_rhs_override=(oracle_rhs_u, oracle_rhs_v),
            expose_momentum_stage=(
                2 if mode == "stage2_rhs_arm" else 0),
            expose_tracer_transport_stage=(
                3 if mode == "stage2_rhs_transport_arm" else 0),
        )
    elif mode == "stage2_rhs":
        hooks = hooks._replace(expose_stage2_momentum_rhs=True)
    elif mode == "stage2_vorticity_legacy":
        hooks = hooks._replace(
            expose_momentum_operator="vorticity",
            legacy_ene_vertex_coriolis=True)
    elif mode in ("stage2_hpg", "stage2_vorticity", "stage2_advection"):
        hooks = hooks._replace(expose_momentum_operator=mode.removeprefix("stage2_"))
    elif mode in ("stage2_ene_operands", "stage2_hpg_operands"):
        # Kmm=3 is the completed stage-1 state.  Both substitutions occur
        # only after the normal step has finished, so this remains WRITE-only.
        hooks = hooks._replace(expose_momentum_stage=1, expose_tracer_stage=1)
    elif mode in (
            "stage3_wzv", "stage3_wzv_nemo_e3w",
            "stage3_wzv_oracle_transport_arm",
            "stage3_wzv_oracle_eta_arm"):
        hooks = hooks._replace(expose_tracer_transport_stage=3)
    elif mode == "arm":
        hooks = hooks._replace(
            pre_implicit_tracer_content_override=(
                jnp.asarray(oracle_content["T"]),
                jnp.asarray(oracle_content["S"]),
            ))
    elif mode == "legacy_eta":
        hooks = hooks._replace(legacy_zdf_entry_kmm_eta=True)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=hooks)
    state = model.step(
        card.recipe.initial_state, dt=card.dt_s,
        freshwater=freshwater, surface_forcing=surface)
    fields = lego_fields(state)

    if mode in (
            "stage3_wzv", "stage3_wzv_nemo_e3w",
            "stage3_wzv_oracle_transport_arm",
            "stage3_wzv_oracle_eta_arm"):
        # The production transport hook returns zFu/zFv/zFw in u/v/T without
        # changing the completed eta.  A second WRITE-only hook returns the
        # stage-2 Kaa velocity that feeds the stage-3 Kmm transport.
        zfu = jnp.asarray(state.u.data)
        zfv = jnp.asarray(state.v.data)
        eta_before = card.recipe.initial_state.eta.data
        eta_after = state.eta.data
        eta_kmm = 0.5 * (eta_before + eta_after)
        if mode == "stage3_wzv_oracle_eta_arm":
            # Stage 3 has Kbb=N, Kmm=N+1/2, Kaa=N+1
            # (stprk3_stg.F90:218-235).  Replace that one coupled SSH
            # operand group while retaining the faithful candidate pFu/pFv.
            eta_kmm = jnp.asarray(stage2["ssh"])
            eta_after = jnp.asarray(stage3["ssh"])
        active = jnp.asarray(card.recipe.z_coord.is_active)
        u_mask_3d, v_mask_3d = compute_face_masks_3d(
            active, card.recipe.grid)
        u_mask_3d = u_mask_3d.astype(state.eta.data.dtype)
        v_mask_3d = v_mask_3d.astype(state.eta.data.dtype)
        if mode == "stage3_wzv_oracle_transport_arm":
            oracle_zfu = jnp.asarray(transport3["zFu"][..., :nlev])
            oracle_zfv = jnp.asarray(transport3["zFv"][..., :nlev])
            volume_pair = (
                jnp.concatenate([oracle_zfu[:, -1:, :], oracle_zfu], axis=1),
                jnp.concatenate([jnp.zeros_like(oracle_zfv[:1]), oracle_zfv], axis=0),
            )
        else:
            volume_pair = (zfu, zfv)
        ww_pre, _, _ = nemo_qco_wzv_operands(
            eta_kmm, eta_before, jnp.zeros_like(zfu), jnp.zeros_like(zfv),
            card.recipe.grid, card.recipe.z_coord, u_mask_3d, v_mask_3d,
            active.astype(state.eta.data.dtype), card.dt_s,
            eta_after_override=eta_after,
            volume_transport_override=volume_pair,
        )
        h_kmm = compute_layer_thickness(
            eta_kmm, state.H_bathy.data, card.recipe.z_coord,
            min_water_column_m=cfg.min_water_column_m)
        if mode == "stage3_wzv_nemo_e3w" and cfg.adaptive_implicit_vertadv:
            stretch = nemo_r3t_stretch(
                card.recipe.z_coord, eta_kmm, state.H_bathy.data,
                evaluation="nemo_reciprocal")
            e3w_kmm = (
                jnp.asarray(card.recipe.z_coord.nemo_e3w_0)
                * stretch[..., None])
        else:
            e3w_kmm = jnp.concatenate([
                h_kmm[..., :1],
                0.5 * (h_kmm[..., :-1] + h_kmm[..., 1:]),
                h_kmm[..., -1:],
            ], axis=-1)
        if cfg.adaptive_implicit_vertadv:
            mf_u = volume_pair[0] / jnp.asarray(card.recipe.grid.dy_u)[..., None]
            mf_v = volume_pair[1] / jnp.asarray(card.recipe.grid.dx_v)[..., None]
            partition = nemo_wicker_aimp_partition_transport(
                mf_u, mf_v, ww_pre, h_kmm, e3w_kmm,
                card.recipe.grid.area_T, card.recipe.grid.dy_u,
                card.recipe.grid.dx_v, card.dt_s)
            ww_post = partition.w_explicit
            wi_post = partition.w_implicit
        else:
            ww_post = ww_pre
            wi_post = jnp.zeros_like(ww_pre)
        candidates = {
            "ww_pre_aimp": np.asarray(ww_pre),
            "ww_post_aimp": np.asarray(ww_post),
            "wi_post_aimp": np.asarray(wi_post),
            "pFw": np.asarray(
                jnp.asarray(card.recipe.grid.area_T)[..., None]
                * ww_post),
        }
        w_mask = np.broadcast_to(
            masks["ssh"][..., None], candidates["ww_pre_aimp"].shape)
        rows = []
        for index, name in enumerate(
                ("ww_pre_aimp", "ww_post_aimp", "wi_post_aimp", "pFw")):
            candidate = candidates[name].copy()
            if plant and index == 0:
                candidate[tuple(np.argwhere(w_mask)[0])] += 1.0
            rows.append(score(
                f"{CASE}.kt1.stage3.{mode}.{name}",
                wzv_record[name], candidate, w_mask))
        arrays = {f"candidate_{name}": value for name, value in candidates.items()}
    elif mode == "stage2_hpg_operands":
        active = jnp.asarray(card.recipe.z_coord.is_active)
        stretch = nemo_r3t_stretch(
            card.recipe.z_coord, state.eta.data, state.H_bathy.data,
            evaluation="nemo_reciprocal")[..., None]
        live_gdept = jnp.asarray(card.recipe.z_coord.nemo_gdept_0) * stretch
        T_filled = neumann_fill_cgrid(
            state.T.data, state.land_mask.data, grid=card.recipe.grid)
        S_filled = neumann_fill_cgrid(
            state.S.data, state.land_mask.data, grid=card.recipe.grid)
        p_eos = (cfg.rho_0 * constants.g) * live_gdept
        candidates = {
            "rhd": np.asarray(jnp.where(
                active,
                nemo_teos10_density_anomaly_ratio(
                    T_filled, S_filled, p_eos, rho0=cfg.rho_0,
                    geometric_depth_m=live_gdept),
                0.0)),
            "e3w": np.asarray(
                jnp.asarray(card.recipe.z_coord.nemo_e3w_0) * stretch),
            "gdept_z0": np.asarray(
                jnp.asarray(card.recipe.z_coord.nemo_gdept_0) * stretch
                - state.eta.data[..., None]),
        }
        full_mask = np.ones_like(np.asarray(hpg_operands["rhd"])[..., :nlev], dtype=bool)
        rows = []
        for index, name in enumerate(("rhd", "e3w", "gdept_z0")):
            candidate = candidates[name].copy()
            if plant and index == 0:
                candidate.flat[0] += 1.0
            rows.append(score(
                f"{CASE}.kt1.stage2.hpg_operand.{name}",
                hpg_operands[name][..., :nlev], candidate, full_mask))
        arrays = {f"candidate_{name}": value for name, value in candidates.items()}
        arrays.update({
            "candidate_stage1_T": np.asarray(state.T.data),
            "candidate_stage1_S": np.asarray(state.S.data),
            "candidate_stage1_ssh": np.asarray(state.eta.data),
        })
    elif mode == "stage2_ene_operands":
        geometry = card.recipe.grid
        active = jnp.asarray(card.recipe.z_coord.is_active)
        u_mask_3d, v_mask_3d = compute_face_masks_3d(active, card.recipe.grid)
        u_mask_3d = u_mask_3d.astype(state.u.data.dtype)
        v_mask_3d = v_mask_3d.astype(state.v.data.dtype)
        h_k = compute_layer_thickness(
            state.eta.data, state.H_bathy.data, card.recipe.z_coord,
            min_water_column_m=cfg.min_water_column_m)
        h_ref = compute_layer_thickness(
            jnp.zeros_like(state.eta.data), state.H_bathy.data,
            card.recipe.z_coord, min_water_column_m=cfg.min_water_column_m)
        h_u, h_v, _, _ = _nemo_ws_qco_stage_faces(
            state.eta.data, h_ref, u_mask_3d, v_mask_3d, card.recipe.grid)
        h_vtx = nemo_qco_live_vorticity_e3f_cgrid(
            state.eta.data, card.recipe.z_coord, state.eta.data.dtype,
            nn_e3f_typ=0)
        _, _, candidate_operands = pv_flux_ene(
            curl_vertex_cgrid(state.u.data, state.v.data, card.recipe.grid),
            h_vtx, h_v, state.v.data, h_u, state.u.data,
            u_mask_3d, v_mask_3d,
            jnp.ones_like(h_vtx),
            f_vtx=nemo_qco_vorticity_f_cgrid(
                card.recipe.z_coord, state.eta.data.dtype),
            q_boundary=cfg.een_q_boundary,
            return_operands=True,
            metric_widths=(geometry.dx_u, geometry.dx_v,
                           geometry.dy_u, geometry.dy_v),
        )
        candidate_q, candidate_fu, candidate_fv = candidate_operands
        candidates = {
            # NEMO (i,j) F is the northeast vertex of T(i,j).
            "zwz": np.asarray(candidate_q)[1:, 1:, :nlev],
            # legoESM stores redundant west/south faces; native NEMO owns
            # the east/north faces, selected by dropping index zero.
            "zwx": np.asarray(candidate_fu)[:, 1:, :nlev],
            "zwy": np.asarray(candidate_fv)[1:, :, :nlev],
        }
        rows = []
        full_mask = np.ones_like(np.asarray(ene_operands["zwz"]), dtype=bool)
        for index, name in enumerate(("zwz", "zwx", "zwy")):
            candidate = candidates[name].copy()
            if plant and index == 0:
                candidate.flat[0] += 1.0
            rows.append(score(
                f"{CASE}.kt1.stage2.ene_operand.{name}",
                ene_operands[name], candidate, full_mask))
        arrays = {f"candidate_{name}": value for name, value in candidates.items()}
    elif mode in ("transport", "transport_zub_arm", "transport_wzv_legacy",
                  "stage2_rhs_transport_arm"):
        rows = []
        for index, (name, field, mask_name) in enumerate((
            ("zFu", "u", "u"), ("zFv", "v", "v"), ("zFw", "T", "T"),
        )):
            candidate = np.asarray(fields[field]).copy()
            if plant and index == 0:
                first = tuple(np.argwhere(masks[mask_name])[0])
                candidate[first] += 1.0
            rows.append(score(
                f"{CASE}.kt1.{mode}.{name}",
                transport3[name][..., :nlev], candidate, masks[mask_name]))
        arrays = {
            "candidate_zFu": np.asarray(fields["u"]),
            "candidate_zFv": np.asarray(fields["v"]),
            "candidate_zFw": np.asarray(fields["T"]),
        }
    elif mode == "stage2_rhs":
        references = {
            "u": stage2_terms["after_advection_u"][..., :nlev],
            "v": stage2_terms["after_advection_v"][..., :nlev],
        }
        names = {
            name: f"{CASE}.kt1.stage2.raw_rhs.{name}" for name in ("u", "v")}
    elif mode in (
            "stage2_hpg", "stage2_vorticity", "stage2_advection",
            "stage2_vorticity_legacy"):
        boundary = mode.removeprefix("stage2_").removesuffix("_legacy")
        before = {
            "hpg": "before", "vorticity": "after_hpg",
            "advection": "after_vorticity",
        }[boundary]
        after = {
            "hpg": "after_hpg", "vorticity": "after_vorticity",
            "advection": "after_advection",
        }[boundary]
        references = {
            name: (
                stage2_terms[f"{after}_{name}"]
                - stage2_terms[f"{before}_{name}"]
            )[..., :nlev]
            for name in ("u", "v")
        }
        names = {
            name: f"{CASE}.kt1.stage2.{boundary}.{name}"
            for name in ("u", "v")
        }
    elif mode == "stage2_rhs_arm":
        references = {name: stage2[name][..., :nlev] for name in ("u", "v")}
        names = {
            name: f"{CASE}.kt1.stage2.oracle_rhs_arm.{name}"
            for name in ("u", "v")}
    elif mode == "pre":
        references = oracle_pre
        names = {name: f"{CASE}.kt1.stage3.pre_zdf.{name}" for name in ("T", "S")}
    elif mode == "content":
        references = oracle_content
        names = {
            name: f"{CASE}.kt1.stage3.content_rhs.{name}"
            for name in ("T", "S")
        }
    elif mode in ("advection_content", "transport_arm"):
        references = oracle_advection_content
        names = {
            name: f"{CASE}.kt1.stage3.{mode}.{name}"
            for name in ("T", "S")
        }
    else:
        references = {name: stage3[name][..., :nlev] for name in ("T", "S")}
        names = {name: f"{CASE}.kt1.stage3.{mode}.{name}" for name in ("T", "S")}
    if mode not in (
            "transport", "transport_zub_arm", "transport_wzv_legacy",
            "stage2_rhs_transport_arm",
            "stage2_ene_operands",
            "stage2_hpg_operands", "stage3_wzv", "stage3_wzv_nemo_e3w",
            "stage3_wzv_oracle_transport_arm", "stage3_wzv_oracle_eta_arm"):
        rows = []
        score_names = (("u", "v") if mode in (
            "stage2_rhs", "stage2_rhs_arm", "stage2_hpg",
            "stage2_vorticity", "stage2_advection",
            "stage2_vorticity_legacy")
                       else ("T", "S"))
        for name in score_names:
            candidate = np.asarray(fields[name]).copy()
            if plant and name == "T":
                first = tuple(np.argwhere(masks[name])[0])
                candidate[first] += 1.0
            rows.append(score(names[name], references[name], candidate, masks[name]))
        arrays = {
            f"candidate_{name}": np.asarray(fields[name])
            for name in score_names
        }
    if plant:
        require(rows[0]["status"] == "DEBT" and rows[0]["absolute_max"] >= 0.9,
                "planted pre-ZDF violation did not fire")

    arrays.update({f"oracle_pre_{name}": oracle_pre[name] for name in ("T", "S")})
    np.savez(output_npz, **arrays)
    report = {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-stage3-completion-v1",
        "case": CASE,
        "mode": mode,
        "bar": BAR,
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "execution_regime": "production_jit",
        "oracle_root": str(oracle_root) if oracle_root is not None else "legacy split roots",
        "rows": rows,
        "status": "AT-BAR" if all(row["status"] == "AT-BAR" for row in rows) else "DEBT",
        "oracle_record": str(
            wzv_path if mode in (
                "stage3_wzv", "stage3_wzv_nemo_e3w",
                "stage3_wzv_oracle_transport_arm",
                "stage3_wzv_oracle_eta_arm") else
            hpg_operand_path if mode == "stage2_hpg_operands" else
            ene_operand_path if mode == "stage2_ene_operands" else
            transport_path if mode in (
                "transport", "transport_zub_arm", "transport_wzv_legacy",
                "stage2_rhs_transport_arm") else tracer_path),
        "oracle_record_sha256": sha256(
            wzv_path if mode in (
                "stage3_wzv", "stage3_wzv_nemo_e3w",
                "stage3_wzv_oracle_transport_arm",
                "stage3_wzv_oracle_eta_arm") else
            hpg_operand_path if mode == "stage2_hpg_operands" else
            ene_operand_path if mode == "stage2_ene_operands" else
            transport_path if mode in (
                "transport", "transport_zub_arm", "transport_wzv_legacy",
                "stage2_rhs_transport_arm") else tracer_path),
        "instrumentation_bit_identity": {
            stage_path.name: sha256(stage_path), kt2_path.name: sha256(kt2_path)},
        "output_npz": str(output_npz),
        "output_npz_sha256": sha256(output_npz),
        "source_order": ["advection", "sbc", "qsr", "ldf", "zdf"],
        "oracle_accumulator_term_max_abs": {
            "advection_T": float(np.max(np.abs(record["after_advection_T"]))),
            "advection_S": float(np.max(np.abs(record["after_advection_S"]))),
            "sbc_T": float(np.max(np.abs(
                record["after_sbc_T"] - record["after_advection_T"]))),
            "sbc_S": float(np.max(np.abs(
                record["after_sbc_S"] - record["after_advection_S"]))),
            "qsr_T": float(np.max(np.abs(
                record["after_qsr_T"] - record["after_sbc_T"]))),
            "qsr_S": float(np.max(np.abs(
                record["after_qsr_S"] - record["after_sbc_S"]))),
            "ldf_T": float(np.max(np.abs(
                record["after_ldf_T"] - record["after_qsr_T"]))),
            "ldf_S": float(np.max(np.abs(
                record["after_ldf_S"] - record["after_qsr_S"]))),
        },
        "planted_control": "VERIFIED" if plant else "NOT_REQUESTED",
    }
    if mode in ("arm", "legacy_eta", "transport_arm"):
        require(faithful_npz is not None and faithful_npz.is_file(),
                "arm mode requires --faithful-npz")
        faithful = np.load(faithful_npz)
        movement = residual = 0.0
        for name in ("T", "S"):
            active = masks[name]
            ref = references[name]
            scale = max(float(np.max(np.abs(ref[active]))), 1.0)
            movement = max(movement, float(np.max(np.abs(
                fields[name][active] - faithful[f"candidate_{name}"][active]))) / scale)
            residual = max(residual, float(np.max(np.abs(
                faithful[f"candidate_{name}"][active] - ref[active]))) / scale)
        report["scaling_check_before_owner_label"] = True
        report["causal_movement"] = movement
        report["faithful_residual"] = residual
        report["movement_over_faithful_residual"] = (
            movement / residual if residual else None)
        if mode == "legacy_eta":
            report["owner_label"] = (
                "ZDF_KMM_TIME_LEVEL_CONFIRMED_CAUSAL_OWNER"
                if movement >= 0.99 * residual else
                "ZDF_KMM_TIME_LEVEL_NOT_SOLE_OWNER")
        elif mode == "transport_arm":
            report["owner_label"] = (
                "STAGE3_TRANSPORT_CONFIRMED_CAUSAL_OWNER"
                if report["status"] == "AT-BAR" else
                "STAGE3_TRANSPORT_NOT_SOLE_OWNER")
        else:
            report["owner_label"] = (
                "PRE_ZDF_INPUT_CONFIRMED_CAUSAL_OWNER"
                if report["status"] == "AT-BAR" else
                "PRE_ZDF_INPUT_NOT_SOLE_OWNER")
        report["faithful_npz_sha256"] = sha256(faithful_npz)
    if mode == "stage2_vorticity_legacy":
        require(faithful_npz is not None and faithful_npz.is_file(),
                "stage2_vorticity_legacy requires --faithful-npz")
        faithful = np.load(faithful_npz)
        movement = 0.0
        faithful_residual = 0.0
        legacy_residual = 0.0
        for name in ("u", "v"):
            active = masks[name]
            ref = references[name]
            scale = max(float(np.max(np.abs(ref[active]))), 1.0)
            faithful_value = faithful[f"candidate_{name}"]
            movement = max(movement, float(np.max(np.abs(
                fields[name][active] - faithful_value[active]))) / scale)
            faithful_residual = max(faithful_residual, float(np.max(np.abs(
                faithful_value[active] - ref[active]))) / scale)
            legacy_residual = max(legacy_residual, float(np.max(np.abs(
                fields[name][active] - ref[active]))) / scale)
        report["scaling_check_before_owner_label"] = True
        report["causal_movement"] = movement
        report["faithful_residual"] = faithful_residual
        report["legacy_residual"] = legacy_residual
        report["movement_over_legacy_residual"] = (
            movement / legacy_residual if legacy_residual else None)
        report["owner_label"] = (
            "ENE_VERTEX_CORIOLIS_MAPPING_CONFIRMED_CAUSAL_OWNER"
            if faithful_residual <= BAR and movement >= 0.99 * legacy_residual
            else "ENE_VERTEX_CORIOLIS_MAPPING_NOT_SOLE_OWNER")
        report["faithful_npz_sha256"] = sha256(faithful_npz)
    if mode == "transport_zub_arm":
        require(faithful_npz is not None and faithful_npz.is_file(),
                "transport_zub_arm requires --faithful-npz")
        faithful = np.load(faithful_npz)
        movement = 0.0
        faithful_residual = 0.0
        arm_residual = 0.0
        for name, mask_name in (("zFu", "u"), ("zFv", "v"), ("zFw", "T")):
            active = masks[mask_name]
            oracle = transport3[name][..., :nlev]
            scale = max(float(np.max(np.abs(oracle[active]))), 1.0)
            arm = np.asarray(fields[{"zFu": "u", "zFv": "v", "zFw": "T"}[name]])
            faithful_value = faithful[f"candidate_{name}"]
            movement = max(movement, float(np.max(np.abs(
                arm[active] - faithful_value[active]))) / scale)
            faithful_residual = max(faithful_residual, float(np.max(np.abs(
                faithful_value[active] - oracle[active]))) / scale)
            arm_residual = max(arm_residual, float(np.max(np.abs(
                arm[active] - oracle[active]))) / scale)
        report["scaling_check_before_owner_label"] = True
        report["causal_movement"] = movement
        report["faithful_residual"] = faithful_residual
        report["arm_residual"] = arm_residual
        report["movement_over_faithful_residual"] = (
            movement / faithful_residual if faithful_residual else None)
        report["owner_label"] = (
            "STAGE_TRANSPORT_STORED_BAROTROPIC_MEAN_CONFIRMED_CAUSAL_OWNER"
            if arm_residual <= BAR and movement >= 0.99 * faithful_residual
            else "STAGE_TRANSPORT_STORED_BAROTROPIC_MEAN_NOT_SOLE_OWNER")
        report["faithful_npz_sha256"] = sha256(faithful_npz)
    if mode == "transport_wzv_legacy":
        require(faithful_npz is not None and faithful_npz.is_file(),
                "transport_wzv_legacy requires --faithful-npz")
        faithful = np.load(faithful_npz)
        movement = 0.0
        faithful_residual = 0.0
        legacy_residual = 0.0
        for name, mask_name in (("zFu", "u"), ("zFv", "v"), ("zFw", "T")):
            active = masks[mask_name]
            oracle = transport3[name][..., :nlev]
            scale = max(float(np.max(np.abs(oracle[active]))), 1.0)
            legacy = np.asarray(
                fields[{"zFu": "u", "zFv": "v", "zFw": "T"}[name]])
            faithful_value = faithful[f"candidate_{name}"]
            movement = max(movement, float(np.max(np.abs(
                legacy[active] - faithful_value[active]))) / scale)
            faithful_residual = max(faithful_residual, float(np.max(np.abs(
                faithful_value[active] - oracle[active]))) / scale)
            legacy_residual = max(legacy_residual, float(np.max(np.abs(
                legacy[active] - oracle[active]))) / scale)
        report["scaling_check_before_owner_label"] = True
        report["causal_movement"] = movement
        report["faithful_residual"] = faithful_residual
        report["legacy_residual"] = legacy_residual
        report["movement_over_legacy_residual"] = (
            movement / legacy_residual if legacy_residual else None)
        report["owner_label"] = (
            "WZV_SHARED_TRANSPORT_CONFIRMED_CAUSAL_OWNER"
            if faithful_residual <= BAR and movement >= 0.99 * legacy_residual
            else "WZV_SHARED_TRANSPORT_NOT_SOLE_OWNER")
        report["faithful_npz_sha256"] = sha256(faithful_npz)
    if mode == "stage3_wzv_nemo_e3w":
        require(faithful_npz is not None and faithful_npz.is_file(),
                "stage3_wzv_nemo_e3w requires --faithful-npz")
        faithful = np.load(faithful_npz)
        movement = 0.0
        faithful_residual = 0.0
        arm_residual = 0.0
        w_mask = np.broadcast_to(
            masks["ssh"][..., None], candidates["ww_pre_aimp"].shape)
        for name in ("ww_pre_aimp", "ww_post_aimp", "wi_post_aimp", "pFw"):
            oracle = np.asarray(wzv_record[name])
            scale = max(float(np.max(np.abs(oracle[w_mask]))), 1.0)
            old = faithful[f"candidate_{name}"]
            new = candidates[name]
            movement = max(movement, float(np.max(np.abs(
                new[w_mask] - old[w_mask]))) / scale)
            faithful_residual = max(faithful_residual, float(np.max(np.abs(
                old[w_mask] - oracle[w_mask]))) / scale)
            arm_residual = max(arm_residual, float(np.max(np.abs(
                new[w_mask] - oracle[w_mask]))) / scale)
        report["scaling_check_before_owner_label"] = True
        report["causal_movement"] = movement
        report["faithful_residual"] = faithful_residual
        report["arm_residual"] = arm_residual
        report["movement_over_faithful_residual"] = (
            movement / faithful_residual if faithful_residual else None)
        report["owner_label"] = (
            "WAIMP_E3W_KMM_CONFIRMED_CAUSAL_OWNER"
            if report["status"] == "AT-BAR"
            and movement >= 0.99 * faithful_residual
            else "WAIMP_E3W_KMM_NOT_SOLE_OWNER")
        report["faithful_npz_sha256"] = sha256(faithful_npz)
    if mode == "stage3_wzv_oracle_transport_arm":
        require(faithful_npz is not None and faithful_npz.is_file(),
                "stage3_wzv_oracle_transport_arm requires --faithful-npz")
        faithful = np.load(faithful_npz)
        movement = 0.0
        faithful_residual = 0.0
        arm_residual = 0.0
        w_mask = np.broadcast_to(
            masks["ssh"][..., None], candidates["ww_pre_aimp"].shape)
        for name in ("ww_pre_aimp", "ww_post_aimp", "wi_post_aimp", "pFw"):
            oracle = np.asarray(wzv_record[name])
            scale = max(float(np.max(np.abs(oracle[w_mask]))), 1.0)
            old = faithful[f"candidate_{name}"]
            new = candidates[name]
            movement = max(movement, float(np.max(np.abs(
                new[w_mask] - old[w_mask]))) / scale)
            faithful_residual = max(faithful_residual, float(np.max(np.abs(
                old[w_mask] - oracle[w_mask]))) / scale)
            arm_residual = max(arm_residual, float(np.max(np.abs(
                new[w_mask] - oracle[w_mask]))) / scale)
        report["scaling_check_before_owner_label"] = True
        report["causal_movement"] = movement
        report["faithful_residual"] = faithful_residual
        report["arm_residual"] = arm_residual
        report["movement_over_faithful_residual"] = (
            movement / faithful_residual if faithful_residual else None)
        report["owner_label"] = (
            "WZV_INPUT_TRANSPORT_CONFIRMED_CAUSAL_OWNER"
            if arm_residual <= BAR and movement >= 0.99 * faithful_residual
            else "WZV_INPUT_TRANSPORT_NOT_SOLE_OWNER")
        report["faithful_npz_sha256"] = sha256(faithful_npz)
    if mode == "stage3_wzv_oracle_eta_arm":
        require(faithful_npz is not None and faithful_npz.is_file(),
                "stage3_wzv_oracle_eta_arm requires --faithful-npz")
        faithful = np.load(faithful_npz)
        movement = 0.0
        faithful_residual = 0.0
        arm_residual = 0.0
        w_mask = np.broadcast_to(
            masks["ssh"][..., None], candidates["ww_pre_aimp"].shape)
        for name in ("ww_pre_aimp", "ww_post_aimp", "wi_post_aimp", "pFw"):
            oracle = np.asarray(wzv_record[name])
            scale = max(float(np.max(np.abs(oracle[w_mask]))), 1.0)
            old = faithful[f"candidate_{name}"]
            new = candidates[name]
            movement = max(movement, float(np.max(np.abs(
                new[w_mask] - old[w_mask]))) / scale)
            faithful_residual = max(faithful_residual, float(np.max(np.abs(
                old[w_mask] - oracle[w_mask]))) / scale)
            arm_residual = max(arm_residual, float(np.max(np.abs(
                new[w_mask] - oracle[w_mask]))) / scale)
        report["scaling_check_before_owner_label"] = True
        report["causal_movement"] = movement
        report["faithful_residual"] = faithful_residual
        report["arm_residual"] = arm_residual
        report["movement_over_faithful_residual"] = (
            movement / faithful_residual if faithful_residual else None)
        report["owner_label"] = (
            "WZV_SSH_OPERAND_TRIPLET_CONFIRMED_CAUSAL_OWNER"
            if arm_residual <= BAR and movement >= 0.99 * faithful_residual
            else "WZV_SSH_OPERAND_TRIPLET_NEAR_NULL"
            if movement < 0.01 * faithful_residual
            else "WZV_SSH_OPERAND_TRIPLET_NOT_SOLE_OWNER")
        report["faithful_npz_sha256"] = sha256(faithful_npz)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "pre", "content", "advection_content", "transport",
            "transport_zub_arm", "transport_wzv_legacy", "faithful",
            "arm", "legacy_eta", "transport_arm", "stage2_rhs",
            "stage2_rhs_arm", "stage2_rhs_transport_arm",
            "stage2_ene_operands", "stage2_hpg_operands", "stage2_hpg", "stage2_vorticity",
            "stage2_advection", "stage2_vorticity_legacy",
            "stage3_wzv", "stage3_wzv_nemo_e3w",
            "stage3_wzv_oracle_transport_arm", "stage3_wzv_oracle_eta_arm"),
        required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--faithful-npz", type=Path)
    parser.add_argument(
        "--oracle-root", type=Path,
        help="single complete oracle root (Round 19 uses scalar-math V2)")
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    report = run(
        args.mode, args.output_npz, args.faithful_npz, args.plant,
        args.oracle_root)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
