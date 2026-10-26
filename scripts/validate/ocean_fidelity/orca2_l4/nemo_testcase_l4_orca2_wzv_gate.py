#!/usr/bin/env python3
"""ORCA2 stage-1 WZV/runoff operand-substitution fidelity gate."""

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

from legoesm import constants  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy  # noqa: E402
from legoesm.core.source_rounding import nemo_source_round  # noqa: E402
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (  # noqa: E402
    nemo_qco_wzv_recurrence,
    nemo_transport_wzv_divergence_level,
)
from legoesm.ocean.fidelity.time_levels import time_level_for_dump  # noqa: E402

RECORD = "oracle_stage1_wzv_operands_kt00000001.bin"
MAGIC = "NEMO_L4_WZVS1_1"
NX, NY, NZ = 94, 152, 31
NLEV = NZ - 1
SX, SY = 91, 149
OX, OY = NX - 4, NY - 4
RN_DT = 10800.0
STAGE_DT = RN_DT / 3.0


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


def _full3(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY, NZ), order="F")[2:-2, 2:-2].transpose(1, 0, 2)


def _full2(values: np.ndarray) -> np.ndarray:
    return values.reshape((NX, NY), order="F")[2:-2, 2:-2].T


def _stencil(values: np.ndarray) -> np.ndarray:
    return values.reshape((SX, SY, NZ), order="F").transpose(1, 0, 2)


def read_record(path: Path) -> dict[str, np.ndarray | dict[str, object]]:
    require(time_level_for_dump(path.name) == "now", "WZV registry is not Kmm/now")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=14i", handle.read(56))
        values = np.fromfile(handle, dtype=np.float64)
    expected_header = (1, 1, 1, 1, 1, 3, NX, NY, NZ, 2, NX - 2, 2, NY - 2, 64)
    require(magic == MAGIC, f"bad WZV magic {magic!r}")
    require(header == expected_header, f"bad WZV header {header}")
    n2, n3, ns = NX * NY, NX * NY * NZ, SX * SY * NZ
    expected_f64 = 2 * ns + 5 * n3 + 4 * n2
    require(values.size == expected_f64,
            f"bad derived WZV payload {values.size} != {expected_f64}")
    require(np.isfinite(values).all(), "non-finite WZV payload")
    result: dict[str, np.ndarray | dict[str, object]] = {}
    offset = 0
    for name, count, convert in (
        ("pFu_stencil", ns, _stencil), ("pFv_stencil", ns, _stencil),
        ("e3t_kmm", n3, _full3), ("e3t_0", n3, _full3),
        ("tmask", n3, _full3), ("r3t_kbb", n2, _full2),
        ("r3t_kaa", n2, _full2), ("r1_e1e2t", n2, _full2),
        ("rnf", n2, _full2), ("ww", n3, _full3), ("pFw", n3, _full3),
    ):
        result[name] = convert(values[offset:offset + count])
        offset += count
    require(offset == values.size, "WZV schema walk did not reach EOF")
    result["header"] = {
        "magic": magic, "version": 1, "kt": 1, "stage": 1,
        "Kbb": 1, "Kmm": 1, "Kaa": 3,
        "jpi": NX, "jpj": NY, "jpk": NZ,
        "stencil_i": [2, NX - 2], "stencil_j": [2, NY - 2],
        "bits": 64, "payload_f64": expected_f64,
        "bytes": path.stat().st_size, "registry_level": "now",
    }
    return result


def _np_divergence(
    fu: np.ndarray, fuw: np.ndarray, fv: np.ndarray, fvs: np.ndarray,
    r1_area: np.ndarray, e3t: np.ndarray, mask: np.ndarray,
    runoff: np.ndarray | None,
) -> np.ndarray:
    """Independent NumPy walk of divhor/sbcrnf statement boundaries."""
    zonal = np.subtract(fu, fuw)
    meridional = np.subtract(fv, fvs)
    numerator = np.add(zonal, meridional)
    transport_div = np.multiply(np.multiply(numerator, r1_area), mask)
    safe = np.where(mask > 0.5, e3t, np.float64(1.0))
    hdiv = np.divide(transport_div, safe)
    if runoff is not None:
        r1_rho0 = np.divide(np.float64(1.0), np.float64(constants.rho_ocean_nemo))
        runoff_div = np.divide(np.multiply(runoff, r1_rho0), safe)
        hdiv = np.subtract(hdiv, runoff_div)
    return np.multiply(np.multiply(e3t, hdiv), mask)


def score(candidate: np.ndarray, oracle: np.ndarray, mask: np.ndarray) -> dict[str, object]:
    actual = np.asarray(candidate, np.float64)[mask]
    expected = np.asarray(oracle, np.float64)[mask]
    require(actual.size and actual.shape == expected.shape, "empty WZV score")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(),
            "non-finite defined WZV cell")
    unequal = actual.view(np.uint64) != expected.view(np.uint64)
    ia, ie = actual.view(np.int64), expected.view(np.int64)
    oa = ia ^ ((ia >> 63) & 0x7fffffffffffffff)
    oe = ie ^ ((ie >> 63) & 0x7fffffffffffffff)
    return {
        "status": "AT_BAR" if not unequal.any() else "DEBT",
        "unequal": int(unequal.sum()), "count": int(unequal.size),
        "max_abs": float(np.abs(actual - expected).max(initial=0.0)),
        "max_ulp": int(np.abs(oa - oe).max(initial=0)),
    }


def validate(oracle_root: Path, *, plant: str | None) -> dict[str, object]:
    jax.config.update("jax_enable_x64", True)
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(jax.default_backend() == "cpu", "WZV gate is CPU-only")
    require(not jax.config.jax_disable_jit, "production JIT disabled")
    require(get_policy() == policy, "fp64 + scalar-libm policy not active")

    path = oracle_root / RECORD
    require(path.is_file(), f"missing {path}")
    record = read_record(path)
    fu = np.asarray(record["pFu_stencil"])
    fv = np.asarray(record["pFv_stencil"])
    e3t = np.asarray(record["e3t_kmm"])
    e3t0 = np.asarray(record["e3t_0"])
    tmask = np.asarray(record["tmask"])
    r3bb = np.asarray(record["r3t_kbb"])
    r3aa = np.asarray(record["r3t_kaa"])
    r1_area = np.asarray(record["r1_e1e2t"])
    runoff = np.asarray(record["rnf"])
    ww_oracle = np.asarray(record["ww"])
    pfw_oracle = np.asarray(record["pFw"])

    # Fortran owned cells are i=3..jpi-2,j=3..jpj-2.  The raw writer begins
    # at i=j=2, retaining exactly the west/south source stencil.
    fu_c, fu_w = fu[1:, 1:, :], fu[1:, :-1, :]
    fv_c, fv_s = fv[1:, 1:, :], fv[:-1, 1:, :]
    live = tmask != 0.0

    def program(fu_, fuw_, fv_, fvs_, r1a_, e3_, e30_, mb_, ma_, mask_, rnf_):
        base, adjusted = [], []
        for jk in range(NLEV):
            b = nemo_transport_wzv_divergence_level(
                fu_[..., jk], fuw_[..., jk], fv_[..., jk], fvs_[..., jk],
                r1a_, e3_[..., jk], mask_[..., jk])
            a = nemo_transport_wzv_divergence_level(
                fu_[..., jk], fuw_[..., jk], fv_[..., jk], fvs_[..., jk],
                r1a_, e3_[..., jk], mask_[..., jk],
                runoff_mass_flux=(rnf_ if jk == 0 else None))
            base.append(b)
            adjusted.append(a)
        base3 = jnp.stack(base, axis=-1)
        adjusted3 = jnp.stack(adjusted, axis=-1)
        ww = nemo_qco_wzv_recurrence(
            adjusted3, e30_[..., :NLEV], mb_, ma_, mask_[..., :NLEV],
            jnp.asarray(STAGE_DT, e3_.dtype))
        return base3, adjusted3, ww

    base, adjusted, ww = (
        np.asarray(value) for value in jax.jit(program)(
            *map(jnp.asarray, (fu_c, fu_w, fv_c, fv_s, r1_area, e3t,
                               e3t0, r3bb, r3aa, tmask, runoff)))
    )
    base_ref = np.empty_like(base)
    adjusted_ref = np.empty_like(adjusted)
    for jk in range(NLEV):
        args = (fu_c[..., jk], fu_w[..., jk], fv_c[..., jk], fv_s[..., jk],
                r1_area, e3t[..., jk], tmask[..., jk])
        base_ref[..., jk] = _np_divergence(*args, None)
        adjusted_ref[..., jk] = _np_divergence(
            *args, runoff if jk == 0 else None)

    from netCDF4 import Dataset
    with Dataset(oracle_root / "mesh_mask_0000.nc") as dataset:
        e1t = np.asarray(dataset["e1t"][0], np.float64)
        e2t = np.asarray(dataset["e2t"][0], np.float64)

    @jax.jit
    def area_product(e1, e2, w):
        area = nemo_source_round(e1 * e2)
        return nemo_source_round(area[..., None] * w)

    pfw = np.asarray(area_product(jnp.asarray(e1t), jnp.asarray(e2t), jnp.asarray(ww)))
    candidates = {
        "divergence": base, "runoff": adjusted,
        "wzv": ww, "pfw": pfw,
    }
    expected = {
        "divergence": base_ref, "runoff": adjusted_ref,
        "wzv": ww_oracle, "pfw": pfw_oracle,
    }
    masks = {
        "divergence": live[..., :NLEV], "runoff": live[..., :NLEV],
        "wzv": live, "pfw": live,
    }
    if plant is not None:
        trial = candidates[plant].copy()
        index = tuple(np.argwhere(masks[plant])[0])
        trial[index] = np.nextafter(trial[index], np.inf)
        candidates[plant] = trial

    rows = {name: score(candidates[name], expected[name], masks[name])
            for name in ("divergence", "runoff", "wzv", "pfw")}
    if plant is not None:
        require(rows[plant]["unequal"] == 1, f"{plant} plant did not fire once")
        raise GateError(
            f"planted {plant} cell rejected through scorer "
            f"({rows[plant]['unequal']}/{rows[plant]['count']})")

    no_runoff_ww = np.asarray(jax.jit(nemo_qco_wzv_recurrence)(
        jnp.asarray(base), jnp.asarray(e3t0[..., :NLEV]), jnp.asarray(r3bb),
        jnp.asarray(r3aa), jnp.asarray(tmask[..., :NLEV]),
        jnp.asarray(STAGE_DT)))
    no_runoff_ablation = score(no_runoff_ww, ww_oracle, live)
    # Retained only as a Rule-11 retraction witness.  This is not a production
    # clock arm: it combines NEMO's explicitly rounded stage-1 HYB SSH level
    # with the full-step denominator.  Production instead combines the
    # full-step SSH endpoint with the full-step denominator, while NEMO pairs
    # HYB with rn_Dt/3.  The mixed pair is formed by neither implementation.
    invalid_mixed_state_dt_ww = np.asarray(jax.jit(nemo_qco_wzv_recurrence)(
        jnp.asarray(adjusted), jnp.asarray(e3t0[..., :NLEV]),
        jnp.asarray(r3bb), jnp.asarray(r3aa),
        jnp.asarray(tmask[..., :NLEV]), jnp.asarray(RN_DT)))
    invalid_mixed_state_dt_ablation = score(
        invalid_mixed_state_dt_ww, ww_oracle, live)
    source_first = next((name for name, row in rows.items()
                         if row["status"] != "AT_BAR"), None)
    first = source_first
    return {
        "status": "PASS_MEASUREMENT_COMPLETE",
        "boundary": "O5-B/stage1-divhor-runoff-QCO-WZV-pFw",
        "result": "AT_BAR" if first is None else "DEBT",
        "first_over_bar_subboundary": first,
        "owner": ("CONFIRMED_ORCA2_RUNOFF_AND_SHARED_WZV_SOURCE_PROGRAM_AT_BAR"
                  if first is None else "ORCA2_OWNER_RUNOFF"),
        "rows": rows,
        "runoff_omission_ablation": no_runoff_ablation,
        "retracted_invalid_mixed_state_full_dt_ablation": {
            **invalid_mixed_state_dt_ablation,
            "disposition": "RETRACTED_RULE_11_NOT_A_PRODUCTION_CONFIGURATION",
            "reason": (
                "pairs NEMO's materialized stage-1 HYB SSH delta with a "
                "full-step denominator; neither NEMO nor legoESM production "
                "forms that state/clock pair"
            ),
        },
        "record": {"path": str(path), "sha256": sha256(path),
                   "schema": record["header"]},
        "execution": {
            "backend": jax.default_backend(), "production_jit": True,
            "dtype": "float64", "transcendentals": get_policy().transcendentals,
            "comparison_domain": "rank0 two-halo-stripped live T cells; jpkm1",
        },
        "levels": {
            "pFu_pFv_e3t": "Kmm=1", "r3t_before": "Kbb=1",
            "r3t_after": "Kaa=3", "ww_pFw": "stage-1 transient",
        },
        "source": {
            "transport": "traadv.F90:221-250; divhor.F90:116-126,140-141",
            "runoff": "sbcrnf.F90:253-260 (ln_rnf_depth=F)",
            "recurrence": "sshwzv.F90:330-336",
            "stage_clock": "stprk3_stg.F90:123-124 (rDt=rn_Dt/3)",
            "area_product": "traadv.F90:264-269",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--plant", choices=("divergence", "runoff", "wzv", "pfw"))
    args = parser.parse_args()
    try:
        result = validate(args.oracle_root, plant=args.plant)
    except (GateError, OSError, UnicodeError, struct.error, ValueError, IndexError) as exc:
        print(f"FAIL: {exc}")
        return 1
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
