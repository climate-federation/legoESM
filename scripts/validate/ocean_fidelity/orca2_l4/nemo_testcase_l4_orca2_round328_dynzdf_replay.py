#!/usr/bin/env python3
"""Offline kt-1 stage-3 dyn_zdf replay from the passive round-328 record."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10,
    nemo_testcase_l4_orca2_round328_dynzdf_gate as admission,
)

DT = np.float64(10800.0)
RHO0 = np.float64(1026.0)
COMPONENTS = ("u", "v")


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def bits(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values).view(np.uint64)


def score(actual, expected, support=None) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    require(actual.shape == expected.shape, f"shape {actual.shape} != {expected.shape}")
    selected = np.ones(actual.shape, dtype=bool) if support is None else np.asarray(support, bool)
    require(selected.shape == actual.shape, "score support shape differs")
    unequal = (bits(actual) != bits(expected)) & selected
    count = int(np.count_nonzero(unequal))
    if count:
        locations = np.argwhere(unequal)
        first = [int(value) for value in locations[0]]
        delta = np.abs(actual[unequal] - expected[unequal])
        maximum = float(np.max(delta))
        max_at = [int(value) for value in locations[int(np.argmax(delta))]]
    else:
        first = max_at = None
        maximum = 0.0
    return {
        "bit_identical": count == 0,
        "unequal": count,
        "compared": int(np.count_nonzero(selected)),
        "first_unequal_jik": first,
        "max_abs": maximum,
        "max_abs_jik": max_at,
    }


def native(values: np.ndarray) -> np.ndarray:
    """Fortran (i,j,k) record layout to legoESM/Numpy (j,i,k)."""
    return np.asarray(values).transpose(1, 0, 2)


def mesh_fields(root: Path, rank: int) -> dict[str, np.ndarray]:
    with Dataset(root / f"mesh_mask_{rank:04d}.nc") as dataset:
        return {
            name: np.asarray(dataset.variables[name][0]).transpose(1, 2, 0).astype(np.float64)
            for name in ("umask", "vmask", "e3u_0", "e3v_0")
        } | {"e3w_1d": np.asarray(dataset.variables["e3w_1d"][0], dtype=np.float64)}


def pair_sum(record_field: np.ndarray, component: str, nx: int, ny: int) -> np.ndarray:
    if component == "u":
        result = record_field[1:nx + 1, :ny, 0] + record_field[:nx, :ny, 0]
    else:
        result = record_field[:nx, 1:ny + 1, 0] + record_field[:nx, :ny, 0]
    return result.T


def avm_face(avm: np.ndarray, component: str, nx: int, ny: int, nz: int) -> np.ndarray:
    if component == "u":
        total = avm[1:nx + 1, :ny, 1:nz] + avm[:nx, :ny, 1:nz]
    else:
        total = avm[:nx, 1:ny + 1, 1:nz] + avm[:nx, :ny, 1:nz]
    return (np.float64(0.5) * total).transpose(1, 0, 2)


def level_masks(wet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top = wet.copy()
    top[..., 1:] *= (wet[..., :-1] == 0.0)
    bottom = wet.copy()
    bottom[..., :-1] *= (wet[..., 1:] == 0.0)
    return top, bottom


def replay_rank(record: dict, mesh: dict[str, np.ndarray]) -> dict[str, object]:
    import jax
    import jax.numpy as jnp

    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import rk3_stage_velocity_update
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import nemo_dynzdf_drag_rhs
    from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
        implicit_vertical_diffusion_nemo_momentum,
    )

    fields = record["fields"]
    header = record["header"]
    nx, ny, nz = (header[name] for name in ("nx", "ny", "nz"))
    rows: dict[str, object] = {}

    for component in COMPONENTS:
        wet = mesh[f"{component}mask"][..., :nz]
        before = native(fields[f"before_{component}"])
        rhs = native(fields[f"rhs_{component}"])
        explicit_expected = native(fields[f"explicit_{component}"])
        explicit = jax.device_get(jax.jit(lambda a, b, m: rk3_stage_velocity_update(
            a, b, DT, m, vector_form=True, written_order=True,
        ))(jnp.asarray(before), jnp.asarray(rhs), jnp.asarray(wet)))
        rows[f"{component}_explicit"] = score(explicit, explicit_expected, wet > 0.0)

        bt = native(fields[f"barotropic_{component}"])
        baro_expected = native(fields[f"baro_sub_{component}"])
        baro = jax.device_get(jax.jit(lambda a, b, m: (a - b) * m)(
            jnp.asarray(explicit), jnp.asarray(bt), jnp.asarray(wet)))
        rows[f"{component}_barotropic_subtraction"] = score(
            baro, baro_expected, wet > 0.0)

        e3_ref = mesh[f"e3{component}_0"][..., :nz]
        r3_kaa = native(fields[f"r3{component}_kaa"])[..., 0]
        r3_kmm = native(fields[f"r3{component}_kmm"])[..., 0]
        dz_after = e3_ref * (np.float64(1.0) + r3_kaa[..., None] * wet)
        e3w_now = mesh["e3w_1d"][1:nz][None, None, :] * (
            np.float64(1.0) + r3_kmm[..., None])
        top_mask, bottom_mask = level_masks(wet)
        zdt2 = jnp.asarray(DT * np.float64(0.5), dtype=jnp.float64)

        bot_pair = pair_sum(fields["rCdU_bot"], component, nx, ny)
        bot_delta = jax.device_get(nemo_dynzdf_drag_rhs(
            zdt2, jnp.asarray(bot_pair), jnp.asarray(bt), jnp.asarray(dz_after)))
        bottom = jax.device_get(nemo_source_round(
            jnp.asarray(baro) + jnp.asarray(bot_delta) * jnp.asarray(bottom_mask)))
        bottom_expected = native(fields[f"bottom_rhs_{component}"])
        rows[f"{component}_bottom_rhs"] = score(bottom, bottom_expected, wet > 0.0)

        top_pair = pair_sum(fields["rCdU_top"], component, nx, ny)
        top_delta = jax.device_get(nemo_dynzdf_drag_rhs(
            zdt2, jnp.asarray(top_pair), jnp.asarray(bt), jnp.asarray(dz_after)))
        top_rhs = jax.device_get(nemo_source_round(
            jnp.asarray(bottom) + jnp.asarray(top_delta) * jnp.asarray(top_mask)))
        top_expected = native(fields[f"top_rhs_{component}"])
        rows[f"{component}_top_rhs"] = score(top_rhs, top_expected, wet > 0.0)

        base = tuple(native(fields[f"{component}_base_zdf_{term}"])
                     for term in ("low", "diag", "up"))
        after_zad = tuple(native(fields[f"{component}_after_zad_{term}"])
                          for term in ("low", "diag", "up"))
        after_ldf = tuple(native(fields[f"{component}_after_ldf_{term}"])
                          for term in ("low", "diag", "up"))
        rows[f"{component}_zad_inactive"] = {
            "bit_identical": all(np.array_equal(bits(a), bits(b))
                                 for a, b in zip(base, after_zad)),
        }
        rows[f"{component}_ldf_inactive"] = {
            "bit_identical": all(np.array_equal(bits(a), bits(b))
                                 for a, b in zip(after_zad, after_ldf)),
        }

        avm = avm_face(fields["avm_input"], component, nx, ny, nz)
        bottom_diag = native(fields[f"{component}_bottom_diag_diag"])
        top_diag = native(fields[f"{component}_top_diag_diag"])
        extra_diag = top_diag - base[1]
        stress = native(fields["utauU" if component == "u" else "vtauV"])[..., 0]
        r = nemo_source_round
        numerator = r(jnp.asarray(DT) * jnp.asarray(stress))
        divisor = r(jnp.asarray(dz_after[..., 0]) * jnp.asarray(RHO0))
        divisor = jnp.where(divisor == 0.0, 1.0, divisor)
        wind = jax.device_get(r(r(numerator / divisor) * jnp.asarray(wet[..., 0])))
        solve_in = np.array(top_rhs, copy=True)
        solve_in[..., 0] += wind
        solve = jax.device_get(jax.jit(implicit_vertical_diffusion_nemo_momentum)(
            jnp.asarray(solve_in), jnp.asarray(avm), jnp.asarray(dz_after),
            jnp.asarray(e3w_now), DT, jnp.asarray(wet > 0.0),
            extra_diag=jnp.asarray(extra_diag)))
        solve_expected = native(fields[f"solve_{component}"])
        rows[f"{component}_base_plus_drag_ordered_solve"] = score(
            solve, solve_expected, wet > 0.0)
        rows[f"{component}_matrix_phase_changes"] = {
            "bottom_diag_unequal": int(np.count_nonzero(bits(bottom_diag) != bits(after_ldf[1]))),
            "top_diag_unequal": int(np.count_nonzero(bits(top_diag) != bits(bottom_diag))),
        }

    order = [
        *(f"{c}_explicit" for c in COMPONENTS),
        *(f"{c}_barotropic_subtraction" for c in COMPONENTS),
        *(f"{c}_bottom_rhs" for c in COMPONENTS),
        *(f"{c}_top_rhs" for c in COMPONENTS),
        *(f"{c}_base_plus_drag_ordered_solve" for c in COMPONENTS),
    ]
    first = next((name for name in order if not rows[name]["bit_identical"]), None)
    return {"rows": rows, "first_non_bit_boundary": first}


def assemble_record_avm(records: dict[int, dict]) -> np.ndarray:
    slabs = []
    for rank in (0, 1):
        record = records[rank]
        nx, ny, nz = (record["header"][name] for name in ("nx", "ny", "nz"))
        slabs.append(record["fields"]["avm_input"][:nx, :ny, 1:nz])
    return np.concatenate(slabs, axis=0).transpose(1, 0, 2)


def model_avm(root: Path, record_avm: np.ndarray) -> dict[str, object]:
    import jax

    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    _, shipped = rung10.card_fields(root)
    card = rung10.comparison_card(shipped)
    surface_fields = rung10.assemble_surface_fields(root, 1)
    freshwater, surface = rung10._surface_forcings(card, root, surface_fields, 1)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        iwm_forcing=card.recipe.iwm_forcing)
    state_after = jax.device_get(model.step(
        card.recipe.initial_state, card.dt_s, freshwater=freshwater,
        surface_forcing=surface, **rung10.damping_step_kwargs(card, root, 1)))
    require(state_after.tke_avm is not None,
            "ordinary comparison step did not return its consumed avm")
    candidate = np.asarray(state_after.tke_avm.data, dtype=np.float64)
    require(candidate.shape == record_avm.shape,
            f"model avm {candidate.shape} != record {record_avm.shape}")
    active = np.array(card.recipe.z_coord.is_active[..., 1:], dtype=bool, copy=True)
    active &= np.asarray(card.recipe.z_coord.is_active[..., :-1], dtype=bool)
    require(active.shape == candidate.shape, "active-interface shape differs")
    row = score(candidate, record_avm, active)
    ice = np.asarray(surface.ice_concentration) > 0.0
    unequal = (bits(candidate) != bits(record_avm)) & active
    under_ice = unequal & ice[..., None]
    row["unequal_under_ice"] = int(np.count_nonzero(under_ice))
    row["unequal_open_water"] = int(np.count_nonzero(unequal & ~ice[..., None]))
    if np.any(under_ice):
        row["first_under_ice_jik"] = [int(v) for v in np.argwhere(under_ice)[0]]
    else:
        row["first_under_ice_jik"] = None
    row["backend"] = jax.default_backend()
    row["dtype"] = str(candidate.dtype)
    return row


def measure(root: Path, expect_commit: str, expect_tree: str) -> dict[str, object]:
    stamp = worktree_stamp(repo=REPO_ROOT)
    require(stamp["commit"] == expect_tree, "worktree commit differs from --expect-tree")
    require(stamp["clean"], "replay requires a clean tracked tree")
    admitted = admission.run_gate(root, expect_commit)
    refusals = {}
    for plant in admission.PLANTS:
        try:
            admission.run_gate(root, expect_commit, plant)
        except admission.GateError:
            refusals[plant] = True
        else:
            refusals[plant] = False
    require(all(refusals.values()), "one or more admission plants stayed green")
    records = {rank: admission.read_record(root / admission.record_name(rank))
               for rank in (0, 1)}
    replay = {str(rank): replay_rank(records[rank], mesh_fields(root, rank))
              for rank in (0, 1)}
    record_avm = assemble_record_avm(records)
    return {
        "format": "nemo-testcase-l4-orca2-round328-dynzdf-replay-v1",
        "worktree": stamp,
        "record_admission": admitted,
        "admission_refusals": refusals,
        "rank_replays": replay,
        "model_avm_vs_record": model_avm(root, record_avm),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", required=True, type=Path)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--expect-tree", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = measure(args.record_dir, args.expect_commit, args.expect_tree)
    except (GateError, admission.GateError, OSError, ValueError) as error:
        print(f"REFUSE: {error}")
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")
    print("STATUS PASS_R328_DYNZDF_OFFLINE_REPLAY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
