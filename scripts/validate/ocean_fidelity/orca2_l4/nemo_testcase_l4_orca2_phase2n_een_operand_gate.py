#!/usr/bin/env python3
"""First-divergence gate for ORCA2's frozen EEN primitive operands."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.core.source_rounding import nemo_source_round  # noqa: E402
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_orca2_zps_card  # noqa: E402
from legoesm.ocean.vertical import nemo_qco_live_vorticity_e3f_cgrid  # noqa: E402

LOCAL_NX, LOCAL_NY, NZ = 94, 152, 31
HALO = 2
GLOBAL_NX, GLOBAL_NY, ACTIVE_NZ = 180, 148, 30
FIELDS = (
    "u_nw", "u_ne", "u_sw", "u_se",
    "v_nw", "v_ne", "v_sw", "v_se",
)


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


def read_stream(path: Path, magic: str, fields: int) -> list[np.ndarray]:
    with path.open("rb") as handle:
        got_magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=10i", handle.read(40))
        values = np.fromfile(handle, np.float64)
    n3 = LOCAL_NX * LOCAL_NY * NZ
    require(got_magic == magic, f"{path.name}: bad magic {got_magic!r}")
    require(header == (1, 1, 1, 3, LOCAL_NX, LOCAL_NY, NZ, fields, 64,
                       fields * n3), f"{path.name}: bad header {header}")
    require(values.size == fields * n3, f"{path.name}: bad payload size")
    arrays = []
    for index in range(fields):
        local = values[index * n3:(index + 1) * n3].reshape(
            (LOCAL_NX, LOCAL_NY, NZ), order="F")
        arrays.append(local[HALO:-HALO, HALO:-HALO, :].transpose(1, 0, 2))
    return arrays


def _triads(q: jax.Array) -> tuple[jax.Array, ...]:
    b = nemo_source_round

    def shift(value, di=0, dj=0):
        out = jnp.roll(value, di, axis=1) if di else value
        return jnp.roll(out, dj, axis=0) if dj else out

    def triad(a, c, d):
        return b(b(a + c) + d)

    return (
        triad(shift(q, 1, 0), q, shift(q, 0, 1)),
        triad(shift(q, 0, 1), q, shift(q, -1, 0)),
        triad(q, shift(q, 0, 1), shift(q, 1, 1)),
        triad(shift(q, -1, 1), shift(q, 0, 1), q),
        triad(q, shift(q, 1, 0), shift(q, 1, -1)),
        triad(shift(q, 0, -1), q, shift(q, 1, 0)),
        triad(shift(q, 1, 1), shift(q, 1, 0), q),
        triad(shift(q, 1, 0), q, shift(q, 0, 1)),
    )


def _production_primitives(eta, z_coord):
    """Mirror the currently selected shared builder through its zpvo triads."""
    b = nemo_source_round
    dtype = jnp.float64
    raw = z_coord.nemo_een_barotropic
    one = jnp.asarray(1.0, dtype=dtype)
    quarter = jnp.asarray(0.25, dtype=dtype)
    eta = jnp.asarray(eta, dtype=dtype)
    area_eta = b(b(jnp.asarray(raw.e1t) * jnp.asarray(raw.e2t)) * eta)
    east = jnp.roll(area_eta, -1, axis=1)
    north = jnp.roll(area_eta, -1, axis=0)
    northeast = jnp.roll(north, -1, axis=1)
    quad = b(b(area_eta + east) + b(north + northeast))
    hf0 = jnp.asarray(raw.hf_0, dtype=dtype)
    wet_f = (hf0 > 0.0).astype(dtype)
    r1_hf0 = b(wet_f / b(hf0 + one - wet_f))
    area_f = b(jnp.asarray(raw.e1f) * jnp.asarray(raw.e2f))
    r3f = b(b(quarter * quad) * r1_hf0 / area_f)
    e3f0 = jnp.asarray(raw.e3f_0, dtype=dtype)
    live = b(e3f0 * b(one + r3f[..., None] * jnp.asarray(raw.fmask)))
    q = b(jnp.asarray(raw.ff_f)[..., None] / live)
    return (e3f0, live, q, *_triads(q))


def _round21_live_divisor_primitives(eta, z_coord):
    """Reproduce the GYRE Round-21 one-variable live-divisor arm."""
    # The committed arm called this existing shared helper.  Its uncommitted
    # probe diff adds source-round/order/reciprocal experiments; Round 21 says
    # those extra arms made zero further movement, so they are not folded in.
    raw = z_coord.nemo_een_barotropic
    dtype = jnp.float64
    b0 = jax.lax.optimization_barrier
    one = jnp.asarray(1.0, dtype=dtype)
    quarter = jnp.asarray(0.25, dtype=dtype)
    e3t0 = jnp.asarray(z_coord.nemo_e3t_0, dtype=dtype)
    tmask = jnp.asarray(z_coord.is_active, dtype=dtype)

    def east(value):
        return jnp.roll(value, -1, axis=1)

    def north_closed(value):
        return jnp.concatenate([value[1:], jnp.zeros_like(value[:1])], axis=0)

    masked = b0(e3t0 * tmask)
    masked_n = north_closed(masked)
    ref_sum = b0(b0(masked + east(masked))
                 + b0(masked_n + east(masked_n)))
    e3f0 = b0(ref_sum / jnp.asarray(4.0, dtype=dtype))
    e3f0 = jnp.where(e3f0 == 0.0, jnp.asarray(raw.e3f_0), e3f0)
    live_vertex = nemo_qco_live_vorticity_e3f_cgrid(
        eta, z_coord, dtype, nn_e3f_typ=0)
    live = live_vertex[1:, 1:]
    q = nemo_source_round(jnp.asarray(raw.ff_f)[..., None] / live)
    return (e3f0, live, q, *_triads(q))


def _orca_fold_primitives(eta, z_coord, grid):
    """Round-21 divisor plus NEMO's T-pivot F-field fold overwrite."""
    base = _round21_live_divisor_primitives(eta, z_coord)
    e3f0 = base[0]
    perm_f = jnp.arange(e3f0.shape[1] - 1, -1, -1, dtype=jnp.int32)
    e3f0 = e3f0.at[-1].set(e3f0[-2, perm_f])
    live_vertex = nemo_qco_live_vorticity_e3f_cgrid(
        eta, z_coord, jnp.float64, nn_e3f_typ=0, grid=grid)
    live = live_vertex[1:, 1:]
    q = nemo_source_round(
        jnp.asarray(z_coord.nemo_een_barotropic.ff_f)[..., None] / live)
    return (e3f0, live, q, *_triads(q))


def _source_masks(raw) -> tuple[np.ndarray, np.ndarray]:
    umask = np.asarray(raw.umask, dtype=bool)[:, :90]
    vmask = np.asarray(raw.vmask, dtype=bool)[:, :90]
    levels = np.arange(ACTIVE_NZ)[None, None, :]
    # NEMO mbku/mbkv retain level one even for dry columns.
    u_owned = levels < np.maximum(umask.sum(axis=2), 1)[:, :, None]
    v_owned = levels < np.maximum(vmask.sum(axis=2), 1)[:, :, None]
    return u_owned, v_owned


def _classify(index: tuple[int, int, int], wet: np.ndarray) -> str:
    j, i, k = index
    if j == GLOBAL_NY - 1:
        return "north_fold_row"
    here = wet[j, i, min(k, ACTIVE_NZ - 1)]
    if not here:
        return "coast_or_land_loop_cell"
    column_bottom = max(int(wet[j, i].sum()), 1) - 1
    if k == column_bottom and column_bottom < ACTIVE_NZ - 1:
        return "partial_cell_bottom"
    neighbours = (
        wet[j, (i - 1) % 90, min(k, ACTIVE_NZ - 1)],
        wet[j, (i + 1) % 90, min(k, ACTIVE_NZ - 1)],
        wet[max(j - 1, 0), i, min(k, ACTIVE_NZ - 1)],
        wet[min(j + 1, GLOBAL_NY - 1), i, min(k, ACTIVE_NZ - 1)],
    )
    if not all(neighbours):
        return "coast_or_land_loop_cell"
    return "interior"


def score(candidate, oracle, mask, wet) -> dict[str, object]:
    actual = np.asarray(candidate, np.float64)[:, :90, :ACTIVE_NZ]
    expected = np.asarray(oracle, np.float64)[:, :, :ACTIVE_NZ]
    require(actual.shape == expected.shape == mask.shape, "operand shape mismatch")
    unequal_grid = actual.view(np.uint64) != expected.view(np.uint64)
    unequal = unequal_grid & mask
    points = np.argwhere(unequal)
    result = {
        "status": "AT_BAR" if not len(points) else "DEBT",
        "unequal": int(unequal.sum()),
        "count": int(mask.sum()),
        "max_abs": float(np.abs(actual[mask] - expected[mask]).max(initial=0.0)),
    }
    if len(points):
        index = tuple(int(value) for value in points[0])
        result["first_global_jik_zero_based"] = list(index)
        result["first_nemo_ijk_one_based"] = [index[1] + 1, index[0] + 1, index[2] + 1]
        result["first_location_class"] = _classify(index, wet)
        classes: dict[str, int] = {}
        for point in points:
            label = _classify(tuple(int(value) for value in point), wet)
            classes[label] = classes.get(label, 0) + 1
        result["location_counts"] = classes
    return result


def validate(deck_root: Path, oracle_root: Path, plant: bool) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "EEN operand gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")

    paths = {
        "e3f_0vor": oracle_root / "oracle_een_e3f0vor_kt00000001.bin",
        "live_e3f_vor": oracle_root / "oracle_een_e3fvor_kt00000001.bin",
        "q": oracle_root / "oracle_een_q_kt00000001.bin",
        "zpvo": oracle_root / "oracle_een_zpvo_kt00000001.bin",
    }
    oracle = (
        read_stream(paths["e3f_0vor"], "NEMO_L4_E3F0_1", 1)[0],
        read_stream(paths["live_e3f_vor"], "NEMO_L4_E3FV_1", 1)[0],
        read_stream(paths["q"], "NEMO_L4_QEEN_1", 1)[0],
        *read_stream(paths["zpvo"], "NEMO_L4_ZPVO_1", 8),
    )
    names = ("e3f_0vor", "live_e3f_vor", "q", *FIELDS)
    card = build_orca2_zps_card(deck_root)
    require(card.recipe.model_config.vorticity_scheme == "een_total",
            "ORCA2 card does not select EEN")
    eta = card.recipe.initial_state.eta.data
    production = jax.jit(lambda value: _production_primitives(
        value, card.recipe.z_coord))(eta)
    live_arm = jax.jit(lambda value: _round21_live_divisor_primitives(
        value, card.recipe.z_coord))(eta)
    orca_fold = jax.jit(lambda value: _orca_fold_primitives(
        value, card.recipe.z_coord, card.recipe.grid))(eta)
    raw = card.recipe.z_coord.nemo_een_barotropic
    u_source, v_source = _source_masks(raw)
    full = np.ones_like(u_source, dtype=bool)
    masks = (full, full, full, u_source, u_source, u_source, u_source,
             v_source, v_source, v_source, v_source)
    wet = (np.asarray(raw.fmask, dtype=bool)[:, :90],) * 3 + (
        (np.asarray(raw.umask, dtype=bool)[:, :90],) * 4
        + (np.asarray(raw.vmask, dtype=bool)[:, :90],) * 4)

    rows = []
    for index, name in enumerate(names):
        expected = oracle[index].copy()
        if plant and index == 0:
            expected[0, 0, 0] = np.nextafter(expected[0, 0, 0], np.inf)
        rows.append({
            "operand": name,
            "production": score(production[index], expected, masks[index], wet[index]),
            "round21_live_divisor": score(live_arm[index], expected, masks[index], wet[index]),
            "orca_fold_fixed": score(orca_fold[index], expected, masks[index], wet[index]),
        })
    if plant:
        # An oracle/oracle arm proves the scorer rejects one changed bit.
        planted_expected = oracle[0][:, :, :ACTIVE_NZ].copy()
        planted_expected[0, 0, 0] = np.nextafter(
            planted_expected[0, 0, 0], np.inf)
        exact = score(oracle[0][:, :, :ACTIVE_NZ], planted_expected,
                      full, wet[0])
        require(exact["unequal"] == 1, "one-bit EEN operand plant did not fire")
        raise GateError("planted EEN operand bit rejected through production scorer")

    def first(arm: str):
        return next((row for row in rows if row[arm]["status"] == "DEBT"), None)

    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "boundary": "O4-EXT-A/EEN-primitive-first-divergence",
        "production_first": first("production"),
        "round21_live_divisor_first": first("round21_live_divisor"),
        "orca_fold_fixed_first": first("orca_fold_fixed"),
        "rows": rows,
        "records": {name: {"path": str(path), "sha256": sha256(path)}
                    for name, path in paths.items()},
        "execution": {
            "backend": jax.default_backend(), "jax_disable_jit": False,
            "dtype": "float64", "transcendentals": get_policy().transcendentals,
            "domain": "rank0 owned cells; global j=0:148, i=0:90, k=0:30",
        },
        "source": {
            "e3f_0vor": "dynvor.F90:918-950",
            "live_e3f_vor": "domqco.F90:233-246; domzgr_substitute.h90:130",
            "q_and_zpvo": "dynspg_ts.F90:1520-1569",
            "production_builder": "barotropic_latlon_cgrid.py:_nemo_literal_een_coefficients",
            "round21_arm": "vertical.py:nemo_qco_live_vorticity_e3f_cgrid",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(args.deck_root, args.oracle_root, args.plant)
    except (GateError, OSError, ValueError, IndexError, struct.error) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
