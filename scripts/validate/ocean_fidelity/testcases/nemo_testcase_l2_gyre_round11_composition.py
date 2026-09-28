#!/usr/bin/env python3
"""Low-memory production-JIT GYRE stage-2 composition boundary gate."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from nemo_testcase_l2_gyre_phase3_gate import (
    CASE,
    DIMS,
    _surface_forcings,
    expected_masks,
    lego_fields,
    read_stage,
    read_stage2_terms,
    require,
    score,
    sha256,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp

ORACLE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round11_oracle_stage2_v4"
)

ORACLE_IDENTITY_PAIRS = {
    # Original vector-math oracle retained for reproducibility.
    (
        "55e780b8d56eb249e5387123b20aeae8f735325200714c02a816ef719d6b3351",
        "3271da17716957c2c043f16af62b77d3aad3213bffd83ddce433d65edb6794ba",
    ): "V1-vector-math",
    # Round-15 scalar-math oracle V2 (-fno-tree-vectorize).
    (
        "e29972359b9fe9929d38dfc58ca0d5f0f84c9a0a7ce65905481349a8a52ef875",
        "6245d06d9b9f477e08d4d0b7b77f42fdb0ef6d86188332811898880f96233a68",
    ): "V2-scalar-math",
}


def _xyz(values: np.ndarray, nx: int, ny: int, nz: int) -> np.ndarray:
    return values.reshape((nx, ny, nz), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def _xy_full(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx, ny), order="F")[2:-2, 2:-2].T


def _xy_interior(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    return values.reshape((nx - 4, ny - 4), order="F").T


def read_stage2_composition(path: Path) -> dict:
    """Read the WRITE-only `NEMO_L2_RKSTG_1` stage-2 composition record."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_RKSTG_1", f"{path}: bad magic")
    require(
        (version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits)
        == (1, 1, 2, 1, 3, 2, 2, *DIMS, 64),
        f"{path}: bad header {header}",
    )
    n3 = nx * ny * nz
    n2 = nx * ny
    ni = (nx - 4) * (ny - 4)
    require(values.size == 10 * n3 + 2 * n2 + 2 * ni, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    out: dict[str, object] = {}
    offset = 0
    for name in (
        "Kbb_u", "Kbb_v", "Kmm_u", "Kmm_v", "Krhs_u", "Krhs_v",
        "raw_Kaa_u", "raw_Kaa_v",
    ):
        out[name] = _xyz(values[offset:offset + n3], nx, ny, nz)
        offset += n3
    for name in ("barotropic_Kaa_u", "barotropic_Kaa_v"):
        out[name] = _xy_full(values[offset:offset + n2], nx, ny)
        offset += n2
    for name in ("zub", "zvb"):
        out[name] = _xy_interior(values[offset:offset + ni], nx, ny)
        offset += ni
    for name in ("post_mean_Kaa_u", "post_mean_Kaa_v"):
        out[name] = _xyz(values[offset:offset + n3], nx, ny, nz)
        offset += n3
    require(offset == values.size, f"{path}: unread payload")
    out.update({"Kbb": kbb, "Kmm": kmm, "Krhs": krhs, "Kaa": kaa})
    return out


def read_stage2_preupdate(path: Path) -> dict[str, np.ndarray]:
    """Read the pre-update record made before the stage-2 Krhs/Kaa alias."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_RKPRE_1", f"{path}: bad magic")
    require(
        (version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits)
        == (1, 1, 2, 1, 3, 2, 2, *DIMS, 64),
        f"{path}: bad header {header}",
    )
    n3 = nx * ny * nz
    require(values.size == 2 * n3, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    return {
        "Krhs_u": _xyz(values[:n3], nx, ny, nz),
        "Krhs_v": _xyz(values[n3:], nx, ny, nz),
    }


def _redundant_faces(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map NEMO east/north native faces to legoESM west/south redundancy."""
    # NEMO's jpk includes its terminal zero-velocity level; legoESM stores the
    # 30 active GYRE layers only.
    u = u[..., :-1]
    v = v[..., :-1]
    return (
        np.concatenate([u[:, -1:, :], u], axis=1),
        np.concatenate([v[-1:, :, :], v], axis=0),
    )


def run(mode: str, oracle_root: Path, plant: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"), "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is required")
    oracle_path = oracle_root / "oracle_rkstage2_operands_kt00000001.bin"
    preupdate_path = oracle_root / "oracle_rkstage2_preupdate_kt00000001.bin"
    identity_path = oracle_root / "oracle_stage_kt00000001_s2.bin"
    restart_path = oracle_root / "GYRE_OMIP_L2_P3_00000010_restart.nc"
    oracle = read_stage2_composition(oracle_path)
    preupdate = read_stage2_preupdate(preupdate_path)
    identity_pair = (sha256(identity_path), sha256(restart_path))
    require(identity_pair in ORACLE_IDENTITY_PAIRS,
            "WRITE-only instrumentation changed the stage-2/restart identity")
    oracle_generation = ORACLE_IDENTITY_PAIRS[identity_pair]

    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    masks = expected_masks(card)
    oracle_rhs = _redundant_faces(preupdate["Krhs_u"], preupdate["Krhs_v"])
    hook_args: dict[str, object] = {}
    reference_names: tuple[str, str]
    if mode == "rhs":
        hook_args["expose_stage2_momentum_rhs"] = True
        oracle = {**oracle, **preupdate}
        reference_names = ("Krhs_u", "Krhs_v")
    elif mode == "raw":
        hook_args["expose_stage2_raw_momentum"] = True
        reference_names = ("raw_Kaa_u", "raw_Kaa_v")
    elif mode == "corrected":
        hook_args["expose_momentum_stage"] = 2
        reference_names = ("post_mean_Kaa_u", "post_mean_Kaa_v")
    elif mode == "oracle_rhs_raw":
        hook_args.update(
            stage2_momentum_rhs_override=oracle_rhs,
            expose_stage2_raw_momentum=True,
        )
        reference_names = ("raw_Kaa_u", "raw_Kaa_v")
    elif mode == "oracle_rhs_corrected":
        hook_args.update(
            stage2_momentum_rhs_override=oracle_rhs,
            expose_momentum_stage=2,
        )
        reference_names = ("post_mean_Kaa_u", "post_mean_Kaa_v")
    elif mode == "legacy_update":
        hook_args.update(
            legacy_vector_stage_qco_weights=True,
            expose_momentum_stage=2,
        )
        reference_names = ("post_mean_Kaa_u", "post_mean_Kaa_v")
    elif mode == "nemo_order_accumulation_arm":
        hook_args.update(
            nemo_stage_rhs_accumulation_order_arm=True,
            expose_stage2_raw_momentum=True,
        )
        reference_names = ("raw_Kaa_u", "raw_Kaa_v")
    elif mode.startswith("oracle_input_"):
        operator = mode.removeprefix("oracle_input_")
        if operator not in ("hpg", "vorticity", "advection"):
            raise ValueError(f"unknown source operator {operator!r}")
        stage1 = read_stage(
            oracle_root / "oracle_stage_kt00000001_s1.bin", 1)
        stage2_terms = read_stage2_terms(
            oracle_root / "oracle_rkstage2_terms_kt00000001.bin")
        if operator == "hpg":
            from nemo_testcase_l2_gyre_round11_hpg import read_hpg_literal
            literal = read_hpg_literal(
                oracle_root / "oracle_rkstage2_hpg_literal_kt00000001.bin")
        oracle = {
            **oracle,
            f"{operator}_u": (
                literal["sum_u"] if operator == "hpg" else
                stage2_terms["after_vorticity_u"]
                if operator == "vorticity" else
                stage2_terms["after_advection_u"]),
            f"{operator}_v": (
                literal["sum_v"] if operator == "hpg" else
                stage2_terms["after_vorticity_v"]
                if operator == "vorticity" else
                stage2_terms["after_advection_v"]),
        }
        cumulative_base = (
            None if operator == "hpg" else
            (stage2_terms["after_hpg_u"], stage2_terms["after_hpg_v"])
            if operator == "vorticity" else
            (stage2_terms["after_vorticity_u"],
             stage2_terms["after_vorticity_v"])
        )
        hook_args.update(
            stage2_thermodynamic_override=(
                jnp.asarray(stage1["T"][..., :-1]),
                jnp.asarray(stage1["S"][..., :-1]),
                jnp.asarray(stage1["ssh"]),
            ),
            expose_momentum_operator=operator,
        )
        reference_names = (f"{operator}_u", f"{operator}_v")
    elif mode == "legacy_mean":
        hook_args.update(
            legacy_live_stage_mean_weights=True,
            expose_momentum_stage=2,
        )
        reference_names = ("post_mean_Kaa_u", "post_mean_Kaa_v")
    else:
        raise ValueError(f"unknown mode {mode!r}")

    freshwater, surface = _surface_forcings(card, card.recipe.initial_state, 1)
    state = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hook_args),
    ).step(
        card.recipe.initial_state, dt=card.dt_s,
        freshwater=freshwater, surface_forcing=surface,
    )
    state = jax.tree_util.tree_map(
        lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, state)
    fields = lego_fields(state)
    rows = []
    for component, reference_name in zip(("u", "v"), reference_names):
        candidate = fields[component]
        if mode in ("oracle_input_vorticity", "oracle_input_advection"):
            # Score the accumulated Krhs boundary, not a subtraction-recovered
            # tendency contaminated by cancellation.  The preceding NEMO
            # boundary is exact for this ordered walk; one host float64 add is
            # the literal Fortran ``puu(Krhs) = puu(Krhs) + term`` statement.
            candidate = cumulative_base[0 if component == "u" else 1][
                ..., :candidate.shape[-1]] + candidate
        if plant and component == "u":
            candidate = candidate.copy()
            candidate[tuple(np.argwhere(masks[component])[0])] += 1.0
        rows.append(score(
            f"{CASE}.kt1.stage2.composition.{mode}.{component}",
            oracle[reference_name][..., :candidate.shape[-1]],
            candidate, masks[component],
        ))
    status = "AT-BAR" if all(row["status"] == "AT-BAR" for row in rows) else "DEBT"
    report = {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round11-composition-v1",
        "case": CASE,
        "mode": mode,
        "status": status,
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "oracle": str(oracle_path),
        "oracle_sha256": sha256(oracle_path),
        "preupdate_oracle": str(preupdate_path),
        "preupdate_oracle_sha256": sha256(preupdate_path),
        "instrumentation_identity": {
            "stage2_sha256": sha256(identity_path),
            "restart_sha256": sha256(restart_path),
            "oracle_generation": oracle_generation,
        },
        "rows": rows,
        "planted_control": plant,
    }
    if plant:
        # The selected live value is nonzero, so adding 1.0 can produce a
        # residual infinitesimally below one.  Require the material DEBT, not
        # an impossible exact lower bound on the subtraction result.
        require(status == "DEBT" and rows[0]["absolute_max"] >= 0.9,
                "planted stage-2 composition violation did not fire")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "rhs", "raw", "corrected", "oracle_rhs_raw",
        "oracle_rhs_corrected", "legacy_update", "legacy_mean",
        "nemo_order_accumulation_arm",
        "oracle_input_hpg", "oracle_input_vorticity",
        "oracle_input_advection",
    ))
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.mode, args.oracle_root, args.plant)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.plant else 0


if __name__ == "__main__":
    raise SystemExit(main())
