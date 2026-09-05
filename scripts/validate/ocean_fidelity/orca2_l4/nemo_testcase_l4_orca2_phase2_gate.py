#!/usr/bin/env python3
"""Boundary-1/2 entry gate for the NEMO 5.0.2 ORCA2 Lane-4 card."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

EXPECTED_INPUT_SHA256 = {
    "ORCA_R2_zps_domcfg.nc":
        "7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839",
    "data_1m_potential_temperature_nomask.nc":
        "ca00905c27078305e80130ea6dd07fc575de2ae3257fad38003b329458cf7f7a",
    "data_1m_salinity_nomask.nc":
        "ad648d972f0631bde7e1b598d98470a979b7f2649271fc9cf5ccfca158e31d0c",
}
EXPECTED_UNMEASURED = {
    "staged_gm_eiv",
    "linear_implicit_bottom_drag",
    "internal_wave_mixing",
    "spatial_lateral_viscosity",
    "freshwater_budget_carry",
    "si3_jpl5_layered_prather_state",
}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mesh_field(root: Path, name: str) -> np.ndarray:
    from netCDF4 import Dataset

    shards = []
    for rank in (0, 1):
        with Dataset(root / f"mesh_mask_{rank:04d}.nc", "r") as ds:
            value = np.asarray(ds.variables[name][0, :30])
            shards.append(np.moveaxis(value, 0, -1))
    return np.concatenate(shards, axis=1)


def _read_entry(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, dtype=np.float64)
    require(magic == "NEMO_L1_ENTRY_1", "step-entry magic")
    require(header == (1, 1, 1, 94, 152, 31, 2, 64), "step-entry header")
    nx, ny, nz = 94, 152, 31
    count = nx * ny * nz
    require(values.size == 4 * count + nx * ny, "step-entry payload size")

    def xyz(block: np.ndarray) -> np.ndarray:
        local = block.reshape((nx, ny, nz), order="F")
        return local[2:-2, 2:-2].transpose(1, 0, 2)

    def xy(block: np.ndarray) -> np.ndarray:
        local = block.reshape((nx, ny), order="F")
        return local[2:-2, 2:-2].T

    return {
        "T": xyz(values[:count]),
        "S": xyz(values[count:2 * count]),
        "u": xyz(values[2 * count:3 * count]),
        "v": xyz(values[3 * count:4 * count]),
        "ssh": xy(values[4 * count:]),
    }


def run_gate(
    deck_root: Path,
    oracle_root: Path,
    *,
    plant: str | None = None,
) -> dict[str, object]:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        NEMO_CONSTANTS_CONFIG,
        build_orca2_zps_card,
        validate_nemo_testcase_card_for_execution,
    )
    from netCDF4 import Dataset

    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")
    require(jnp.ones(1).dtype == jnp.float64, "JAX x64 not active")
    require(not jax.config.jax_disable_jit, "production JIT is disabled")

    hashes = {}
    for name, expected in EXPECTED_INPUT_SHA256.items():
        path = deck_root / name
        require(path.is_file(), f"missing input {path}")
        hashes[name] = sha256(path)
        require(hashes[name] == expected, f"input hash mismatch: {name}")

    card = build_orca2_zps_card(deck_root)
    require(card.precision_policy == "fp64", "card precision stamp")
    require(card.transcendentals == "libm", "card transcendental stamp")
    require((card.recipe.grid.n_lat, card.recipe.grid.n_lon) == (148, 180),
            "owned horizontal shape")
    require(card.recipe.z_coord.n_levels == 30, "prognostic vertical shape")
    require(card.dummy_bottom_records == 1, "dummy jpk record registry")
    require(card.recipe.model_config.eos == "nemo_eos80", "EOS-80 selector")
    require(card.recipe.model_config.vorticity_scheme == "een_total", "EEN selector")
    require(card.recipe.model_config.tracer_time_integrator == "rk3_ws", "WS-RK3")
    require(card.recipe.model_config.barotropic.n_barotropic_substeps == 65,
            "barotropic substep count")
    require(card.icebergs_enabled is False, "card must resolve ln_icebergs=F")
    require(card.iceberg_inputs == (), "icebergs-off card must have no inputs")
    require(card.surface_input_operator == "nemo_fld_read",
            "card must select the shared fld_read map")

    unresolved = set(card.unmeasured_features)
    if plant == "coverage":
        unresolved.remove("si3_jpl5_layered_prather_state")
    if plant == "iceberg_option":
        card = card._replace(icebergs_enabled=True, iceberg_inputs=("planted",))
        require(card.icebergs_enabled is False, "card must resolve ln_icebergs=F")
    require(unresolved == EXPECTED_UNMEASURED, "resolved-feature coverage registry")
    try:
        validate_nemo_testcase_card_for_execution(card)
    except ValueError as exc:
        execution_guard = str(exc)
    else:
        raise GateError("ORCA2 card executed despite unresolved selected arms")

    grid = card.recipe.grid
    z_coord = card.recipe.z_coord
    operands = z_coord.nemo_een_barotropic
    with Dataset(deck_root / "ORCA_R2_zps_domcfg.nc", "r") as ds:
        raw = {name: np.asarray(ds.variables[name][:]) for name in (
            "glamt", "gphit", "e1t", "e2t", "e1u", "e2u", "e1v", "e2v",
            "ff_t", "ff_f", "e3t_0", "e3u_0", "e3v_0", "e3f_0",
        )}

    geometry = {
        "glamt_rad": (np.asarray(grid.lon_T), np.asarray(jnp.deg2rad(raw["glamt"]))),
        "gphit_rad": (np.asarray(grid.lat_T), np.asarray(jnp.deg2rad(raw["gphit"]))),
        "e1t": (np.asarray(grid.dx_T), raw["e1t"]),
        "e2t": (np.asarray(grid.dy_T), raw["e2t"]),
        # NEMO U(i) is the east face of T(i); legoESM's redundant U[0] is
        # its periodic west image, so the complete native array maps to 1:.
        "e1u": (np.asarray(grid.dx_u)[:, 1:], raw["e1u"]),
        "e2u": (np.asarray(grid.dy_u)[:, 1:], raw["e2u"]),
        "e1v": (np.asarray(grid.dx_v)[1:], raw["e1v"]),
        "e2v": (np.asarray(grid.dy_v)[1:], raw["e2v"]),
        "ff_t": (np.asarray(grid.f_T), raw["ff_t"]),
        "ff_f_grid": (np.asarray(grid.ff_f), raw["ff_f"]),
        "ff_f": (np.asarray(operands.ff_f), raw["ff_f"]),
        "e3t_0": (np.asarray(z_coord.nemo_e3t_0), np.moveaxis(raw["e3t_0"][:30], 0, -1)),
        "e3u_0": (np.asarray(operands.e3u_0), np.moveaxis(raw["e3u_0"][:30], 0, -1)),
        "e3v_0": (np.asarray(operands.e3v_0), np.moveaxis(raw["e3v_0"][:30], 0, -1)),
        "e3f_0": (np.asarray(operands.e3f_0), np.moveaxis(raw["e3f_0"][:30], 0, -1)),
    }
    # Pin the pre-Phase-2 tripole v-face Coriolis byte-for-byte.  Native NEMO
    # ff_f is an F-point field and must never replace this generic V-point
    # storage on the curvilinear ORCA2 mesh.
    legacy_f_t = (
        2.0 * float(NEMO_CONSTANTS_CONFIG.Omega) * jnp.sin(grid.lat_T)
    ).astype(grid.f_v.dtype)
    legacy_f_v = jnp.concatenate(
        [
            legacy_f_t[0:1],
            0.5 * (legacy_f_t[:-1] + legacy_f_t[1:]),
            legacy_f_t[-1:],
        ],
        axis=0,
    )
    geometry["f_v_legacy"] = (np.asarray(grid.f_v), np.asarray(legacy_f_v))
    if plant == "grid":
        actual, expected = geometry["e1t"]
        actual = actual.copy()
        actual[0, 0] = np.nextafter(actual[0, 0], np.inf)
        geometry["e1t"] = (actual, expected)
    if plant == "coriolis_swap":
        geometry["ff_f_grid"] = (np.asarray(grid.f_v)[1:], raw["ff_f"])
    for name, (actual, expected) in geometry.items():
        require(np.array_equal(actual, expected), f"geometry mismatch: {name}")

    masks = {
        "tmask": np.asarray(z_coord.is_active),
        "umask": np.asarray(operands.umask),
        "vmask": np.asarray(operands.vmask),
        "fmask": np.asarray(operands.fmask),
    }
    if plant == "fold":
        masks["vmask"] = masks["vmask"].copy()
        masks["vmask"][-1] = masks["vmask"][-1, ::-1]
    for name, actual in masks.items():
        require(np.array_equal(actual, _mesh_field(oracle_root, name)),
                f"NEMO mesh mismatch: {name}")

    if plant == "dummy":
        require(card.recipe.z_coord.n_levels == 31, "dummy-level plant")

    entry = _read_entry(oracle_root / "oracle_step_entry_kt00000001.bin")
    state = card.recipe.initial_state
    candidate = {
        "T": np.asarray(state.T.data)[:, :90],
        "S": np.asarray(state.S.data)[:, :90],
        "u": np.asarray(state.u.data)[:, 1:91],
        "v": np.asarray(state.v.data)[1:, :90],
        "ssh": np.asarray(state.eta.data)[:, :90],
    }
    if plant == "interp":
        with Dataset(
            deck_root / "data_1m_potential_temperature_nomask.nc", "r"
        ) as ds:
            before = np.asarray(ds.variables["votemper"][11, :30], np.float64)
            after = np.asarray(ds.variables["votemper"][0, :30], np.float64)
        # Deliberate reassociation: mathematically equivalent, not NEMO's
        # fld_read multiply-add program.
        alternative = before + np.float64(249.0 / 496.0) * (after - before)
        alternative = np.moveaxis(alternative, 0, -1)
        alternative = np.where(np.asarray(z_coord.is_active), alternative, 0.0)
        candidate["T"] = alternative[:, :90]
    if plant in {"T", "S", "zero"}:
        name = "u" if plant == "zero" else plant
        candidate[name] = candidate[name].copy()
        candidate[name][0, 0, 0] = np.nextafter(candidate[name][0, 0, 0], np.inf)
    if plant == "halo":
        candidate["T"] = np.asarray(state.T.data)[:, :94]

    rows = {}
    for name, actual in candidate.items():
        expected = entry[name]
        if name != "ssh":
            require(not np.count_nonzero(expected[..., 30]),
                    f"nonzero dummy record: {name}")
            expected = expected[..., :30]
        unequal = (
            int(np.count_nonzero(actual != expected))
            if actual.shape == expected.shape
            else -1
        )
        rows[name] = {"unequal": unequal, "count": int(expected.size)}
        require(actual.shape == expected.shape and unequal == 0,
                f"kt=1 identity mismatch: {name} ({unequal}/{expected.size})")

    return {
        "status": "PASS",
        "boundary_1": "VERIFIED_WITH_EXPLICIT_UNMEASURED",
        "boundary_2_ocean": "VERIFIED_BIT_EXACT",
        "boundary_2_ice": "UNMEASURED_STOP",
        "card": card.case,
        "precision": card.precision_policy,
        "transcendentals": card.transcendentals,
        "execution_ready": False,
        "execution_guard": execution_guard,
        "icebergs_enabled": card.icebergs_enabled,
        "iceberg_inputs": list(card.iceberg_inputs),
        "comparison_domain": "rank0-owned y=148,x=90 after 2-cell halo strip",
        "input_sha256": hashes,
        "unmeasured_features": sorted(unresolved),
        "entry_identity": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--plant",
        choices=(
            "grid",
            "coriolis_swap",
            "fold",
            "dummy",
            "interp",
            "T",
            "S",
            "zero",
            "halo",
            "coverage",
            "iceberg_option",
        ),
    )
    args = parser.parse_args()
    try:
        result = run_gate(args.deck_root, args.oracle_root, plant=args.plant)
    except (GateError, ValueError, IndexError) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
