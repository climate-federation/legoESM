#!/usr/bin/env python3
"""EOS-first gate for NEMO testcase phase 3.

The oracle is parsed directly from NEMO 5.0.2 ``eosbn2.F90``.  Candidate
density is evaluated on the actual registered Nbb/before T/S/SSH dump states;
no analytic test state substitutes for the trajectory inputs.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

BAR = 1.0e-15
NEMO_EOS = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/src/OCE/TRA/eosbn2.F90"
)
ROOTS = {
    "LOCK_EXCHANGE-zco": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/lock_exchange_zco"
    ),
    "OVERFLOW-zps": Path(
        "/data/abyssal/dbalwada/nemo-testcases-l1/overflow_zps"
    ),
}
DUMPS = {
    "LOCK_EXCHANGE-zco": (1, 30600, 61200),
    "OVERFLOW-zps": (1, 3060, 6120),
}


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def parse_teos10_density_coefficients(path: Path = NEMO_EOS) -> dict[str, float]:
    """Parse the executed TEOS-10 EOS### assignments from NEMO source."""
    text = path.read_text()
    init = text.index("SUBROUTINE eos_init")
    start = text.index("CASE( np_teos10 )", init)
    end = text.index("CASE( np_eos80 )", start)
    block = text[start:end]
    found: dict[str, float] = {}
    pattern = re.compile(
        r"^\s*(EOS\d{3})\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eEdD][+-]?\d+)?)_wp\s*$",
        re.MULTILINE,
    )
    for name, raw in pattern.findall(block):
        found[name] = float(raw.replace("D", "E").replace("d", "e"))
    require(len(found) == 52, f"parsed {len(found)} TEOS-10 EOS coefficients, want 52")
    found.update(
        rdeltaS=32.0,
        r1_S0=0.875 / 35.16504,
        r1_T0=1.0 / 40.0,
        r1_Z0=1.0e-4,
    )
    return found


def nemo_literal_density(T, S, depth_m, c: dict[str, float]) -> np.ndarray:
    """NumPy transliteration of ``eosbn2.F90:260-288`` association."""
    T = np.asarray(T, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    depth_m = np.asarray(depth_m, dtype=np.float64)
    zt = T * c["r1_T0"]
    zs = np.sqrt(np.abs(S + c["rdeltaS"]) * c["r1_S0"])
    zh = depth_m * c["r1_Z0"]
    zn3 = c["EOS013"] * zt + c["EOS103"] * zs + c["EOS003"]
    zn2 = ((c["EOS022"] * zt + c["EOS112"] * zs + c["EOS012"]) * zt
           + (c["EOS202"] * zs + c["EOS102"]) * zs + c["EOS002"])
    zn1 = (((c["EOS041"] * zt + c["EOS131"] * zs + c["EOS031"]) * zt
            + (c["EOS221"] * zs + c["EOS121"]) * zs + c["EOS021"]) * zt
           + ((c["EOS311"] * zs + c["EOS211"]) * zs + c["EOS111"]) * zs
           + c["EOS011"]) * zt \
        + (((c["EOS401"] * zs + c["EOS301"]) * zs + c["EOS201"]) * zs
           + c["EOS101"]) * zs + c["EOS001"]
    zn0 = (((((c["EOS060"] * zt + c["EOS150"] * zs + c["EOS050"]) * zt
             + (c["EOS240"] * zs + c["EOS140"]) * zs + c["EOS040"]) * zt
            + ((c["EOS330"] * zs + c["EOS230"]) * zs + c["EOS130"]) * zs
            + c["EOS030"]) * zt
           + (((c["EOS420"] * zs + c["EOS320"]) * zs + c["EOS220"]) * zs
              + c["EOS120"]) * zs + c["EOS020"]) * zt
          + ((((c["EOS510"] * zs + c["EOS410"]) * zs + c["EOS310"]) * zs
              + c["EOS210"]) * zs + c["EOS110"]) * zs + c["EOS010"]) * zt \
        + (((((c["EOS600"] * zs + c["EOS500"]) * zs + c["EOS400"]) * zs
             + c["EOS300"]) * zs + c["EOS200"]) * zs + c["EOS100"]) * zs \
        + c["EOS000"]
    return ((zn3 * zh + zn2) * zh + zn1) * zh + zn0


def read_entry(path: Path, case: str) -> dict[str, np.ndarray | int]:
    with path.open("rb") as fh:
        magic = fh.read(16).decode("ascii").rstrip()
        version, step, nbb, nx, ny, nz, ntr, bits = struct.unpack(
            "=8i", fh.read(32)
        )
        data = np.fromfile(fh, dtype=np.float64)
    require(magic == "NEMO_L1_ENTRY_1", f"{path}: bad magic")
    expected = (206, 7, 101) if case == "OVERFLOW-zps" else (134, 7, 21)
    require((version, nx, ny, nz, ntr, bits) == (1, *expected, 2, 64), "bad header")
    count = nx * ny * nz
    require(data.size == 4 * count + nx * ny, f"{path}: bad payload length")

    def xyz(values):
        return values.reshape((nx, ny, nz), order="F")[2:-2, 2:-2].transpose(1, 0, 2)

    return {
        "step": step,
        "Nbb": nbb,
        "T": xyz(data[:count]),
        "S": xyz(data[count : 2 * count]),
        "ssh": data[4 * count :].reshape((nx, ny), order="F")[2:-2, 2:-2].T,
    }


def score(name: str, oracle, candidate, mask) -> dict:
    oracle = np.asarray(oracle, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    use = np.asarray(mask, dtype=bool)
    require(oracle.shape == candidate.shape == use.shape, f"{name}: shape mismatch")
    require(bool(use.any()), f"{name}: empty mask")
    require(np.all(np.isfinite(candidate[use])), f"{name}: nonfinite candidate")
    scale = max(float(np.max(np.abs(oracle[use]))), 1.0)
    error = float(np.max(np.abs(candidate[use] - oracle[use]))) / scale
    return {
        "name": name,
        "status": "AT-BAR" if error <= BAR else "DEBT",
        "normalized_max_abs": error,
        "bar": BAR,
        "n": int(use.sum()),
        "oracle_dtype": str(oracle.dtype),
        "candidate_dtype": str(candidate.dtype),
    }


def run(case: str, *, plant: bool = False) -> dict:
    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.eos import _ROQUET_TEOS10, make_eos_fn, nemo_roquet_eos
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    set_policy(PrecisionPolicy.fp64())
    card = build_nemo_testcase_card(case)
    require(get_policy() == PrecisionPolicy.fp64(), "precision policy is not fp64")
    cfg = card.recipe.model_config
    zc = card.recipe.z_coord
    require(zc.t_depth_ref is not None, "card lacks explicit NEMO T-depth ladder")
    source_coeffs = parse_teos10_density_coefficients()
    coeff_names = sorted(k for k in source_coeffs if k.startswith("EOS"))
    coefficient_max_abs = max(
        abs(source_coeffs[k] - float(_ROQUET_TEOS10[k])) for k in coeff_names
    )
    require(coefficient_max_abs == 0.0, "canonical TEOS-10 coefficient table differs from NEMO")

    eos_fn = make_eos_fn(cfg.eos, rho0=cfg.rho_0)
    root = ROOTS[case]
    active = np.asarray(zc.is_active)
    H = np.asarray(card.recipe.initial_state.H_bathy.data)
    t_depth = np.asarray(zc.t_depth_ref)
    rows = []
    canonical_rows = []
    for step in DUMPS[case]:
        dump = read_entry(root / f"oracle_step_entry_kt{step:08d}.bin", case)
        require(dump["step"] == step, f"step header mismatch for {step}")
        T = np.asarray(dump["T"])[..., : zc.n_levels]
        S = np.asarray(dump["S"])[..., : zc.n_levels]
        ssh = np.asarray(dump["ssh"])
        stretch = np.where(H > 0.0, 1.0 + ssh / np.maximum(H, 1.0), 1.0)
        depth = t_depth[None, None, :] * stretch[..., None]
        oracle = nemo_literal_density(T, S, depth, source_coeffs)
        pressure = cfg.rho_0 * constants.g * depth
        candidate = np.asarray(eos_fn(jnp.asarray(T), jnp.asarray(S), jnp.asarray(pressure)))
        if plant and step == DUMPS[case][0]:
            candidate = candidate.copy()
            candidate[tuple(np.argwhere(active)[0])] += 1.0
        rows.append(score(f"{case}.kt{step}.card_eos.{cfg.eos}", oracle, candidate, active))
        canonical = np.asarray(nemo_roquet_eos(
            jnp.asarray(T), jnp.asarray(S), jnp.asarray(pressure),
            coeffs=_ROQUET_TEOS10, rho0=cfg.rho_0,
        ))
        canonical_rows.append(
            score(f"{case}.kt{step}.canonical_nemo_teos10", oracle, canonical, active)
        )
    all_rows = rows + canonical_rows
    return {
        "format": "nemo-testcase-l1-phase3-eos-v1",
        "case": case,
        "status": "AT-BAR" if all(r["status"] == "AT-BAR" for r in rows) else "DEBT",
        "card_eos": cfg.eos,
        "precision_policy": "fp64",
        "coefficient_count": len(coeff_names),
        "coefficient_max_abs": coefficient_max_abs,
        "card_rows": rows,
        "canonical_rows": canonical_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=tuple(ROOTS), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    report = run(args.case, plant=args.plant)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        print(f"DEBT: {exc}", file=sys.stderr)
        raise SystemExit(1)
