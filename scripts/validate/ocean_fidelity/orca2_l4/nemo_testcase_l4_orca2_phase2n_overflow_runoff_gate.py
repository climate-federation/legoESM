#!/usr/bin/env python3
"""Reduced-level Rule-12 gate for OVERFLOW across the runoff refactor."""

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
from netCDF4 import Dataset

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (  # noqa: E402
    nemo_transport_wzv_divergence_level,
)

NX, NY, NZ = 206, 7, 101
# Surface, shallow interior, mid-column, and the deepest level carrying a
# nonzero kt=1 transport in the shipped OVERFLOW gravity current.
LEVELS = (0, 1, 12, 24)


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _xyz(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY, NZ), order="F").transpose(1, 0, 2)


def read_transport(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=8i", handle.read(32))
        values = np.fromfile(handle, np.float64)
    require(magic == "NEMO_L1_TRANSP_1", f"bad transport magic {magic!r}")
    require(header == (1, 1, 1, 1, NX, NY, NZ, 64), f"bad header {header}")
    n3 = NX * NY * NZ
    require(values.size == 3 * n3, "bad transport payload")
    return _xyz(values[:n3]), _xyz(values[n3:2 * n3])


def pre_refactor(flux_u, west, flux_v, south, r1_area, live_e3t, tmask):
    """The e4f62e664 inline statements replaced by 020a5043b."""
    b = jax.lax.optimization_barrier
    zonal = b(flux_u - west)
    meridional = b(flux_v - south)
    numerator = b(zonal + meridional)
    transport_div = b(numerator * r1_area) * tmask
    safe_e3t = jnp.where(tmask > 0.5, live_e3t, 1.0)
    hdiv = b(transport_div / safe_e3t)
    return b(live_e3t * hdiv) * tmask


def validate(root: Path, plant: bool) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "OVERFLOW Rule-12 gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    transport_path = root / "oracle_transport_kt00000001_s1.bin"
    mesh_path = root / "mesh_mask.nc"
    zfu, zfv = read_transport(transport_path)
    with Dataset(mesh_path) as dataset:
        def field(name):
            return np.asarray(dataset[name][:]).squeeze()
        e1e2t = field("e1t") * field("e2t")
        tmask = field("tmask")
        e3t0 = field("e3t_0")
    require(e3t0.shape == (NZ, NY - 4, NX - 4),
            f"bad de-haloed e3t shape {e3t0.shape}")
    e3_inner = e3t0.transpose(1, 2, 0)
    mask_inner = tmask.transpose(1, 2, 0)
    e3t0 = np.ones((NY, NX, NZ), np.float64)
    tmask = np.zeros((NY, NX, NZ), np.float64)
    r1_area = np.ones((NY, NX), np.float64)
    e3t0[2:-2, 2:-2] = e3_inner
    tmask[2:-2, 2:-2] = mask_inner
    r1_area[2:-2, 2:-2] = 1.0 / np.asarray(e1e2t)
    rows = []
    compiled = jax.jit(lambda fu, fw, fv, fs, area, e3, mask: (
        pre_refactor(fu, fw, fv, fs, area, e3, mask),
        nemo_transport_wzv_divergence_level(
            fu, fw, fv, fs, area, e3, mask, runoff_mass_flux=None),
    ))
    for level in LEVELS:
        fu = zfu[..., level]
        fv = zfv[..., level]
        west = np.roll(fu, 1, axis=1)
        south = np.concatenate([np.zeros_like(fv[:1]), fv[:-1]], axis=0)
        before, after = compiled(
            jnp.asarray(fu), jnp.asarray(west), jnp.asarray(fv),
            jnp.asarray(south), jnp.asarray(r1_area),
            jnp.asarray(e3t0[..., level]), jnp.asarray(tmask[..., level]))
        before, after = np.asarray(before), np.asarray(after)
        owned = np.asarray(tmask[..., level], bool)
        owned[:2] = False; owned[-2:] = False
        owned[:, :2] = False; owned[:, -2:] = False
        require(owned.any(), f"level {level + 1} has no owned wet cells")
        if plant and level == LEVELS[0]:
            after = after.copy()
            index = tuple(np.argwhere(owned)[0])
            after[index] = np.nextafter(after[index], np.inf)
        unequal = before[owned].view(np.uint64) != after[owned].view(np.uint64)
        rows.append({
            "level_one_based": level + 1,
            "unequal": int(unequal.sum()), "count": int(unequal.size),
            "max_abs": float(np.abs(before[owned] - after[owned]).max(initial=0.0)),
            "nonzero_operand_cells": int(np.count_nonzero(fu[owned])
                                         + np.count_nonzero(fv[owned])),
            "status": "AT_BAR" if not unequal.any() else "DEBT",
        })
    require(all(row["status"] == "AT_BAR" for row in rows),
            "runoff None-branch moved an OVERFLOW owned cell")
    if plant:
        raise GateError("planted OVERFLOW runoff None-branch bit rejected")
    return {
        "status": "PASS", "result": "0_ULP",
        "case": "OVERFLOW-zps", "boundary": "Rule12/runoff_mass_flux=None",
        "rows": rows,
        "pre_revision": "e4f62e664 (inline div_hor statements)",
        "post_revision": "020a5043b+ (nemo_transport_wzv_divergence_level)",
        "records": {"transport": {"path": str(transport_path),
                                    "sha256": sha256(transport_path)},
                    "mesh": {"path": str(mesh_path), "sha256": sha256(mesh_path)}},
        "execution": {"backend": jax.default_backend(), "production_jit": True,
                      "dtype": "float64", "transcendentals": get_policy().transcendentals,
                      "scope": "kt=1 owned wet cells at four representative levels"},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    try:
        result = validate(args.oracle_root, args.plant)
    except (GateError, OSError, ValueError, struct.error) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
