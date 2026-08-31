#!/usr/bin/env python
"""Discriminate row-30 ``zau`` metric value from reciprocal evaluation form."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

import jax
import jax.numpy as jnp
import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).parent))
import bn2_alpha_compare as loaders  # noqa: E402
import dump_lane  # noqa: E402
import ldf_slp_per_element as ldf  # noqa: E402
import zdf_chain_sweep as sweep  # noqa: E402
from legoesm.core.precision import PrecisionPolicy, set_policy  # noqa: E402

BAR = 1.0e-15
EXPECTED_ARTIFACT_SHA = "37c5565786d614597f8cd4b1d5f8e2b5d9656b5cf04ef5601d12aefe34a3203d"
EXPECTED_BINARY_SHA = "a77fbaa9e302e699acb76b89a4c4d8e04d4ec0a77e6b0bbee8b8a45180a6d20f"
EXPECTED_RESTART_SHA = "33c0c1a2e998161afdc9d4b71c5606f5cc5d869e54d53058fc0f64eeac7a115c"
EXPECTED_ZGRU_SHA = "d9d8d91277ae3766d9240642fae2abe5199552fa95e59c6254eb8a6b50543136"
EXPECTED_ZAU_SHA = "78f5d2159baad245f6c2fbb29f4cfbc24bc2724cfbd8b4077953e5b3576398b1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def _capture_zau(state: dict) -> tuple[np.ndarray, np.ndarray]:
    uv_code = next(c for c in ldf.compute_nemo_native_slopes.__code__.co_consts
                   if isinstance(c, types.CodeType) and c.co_name == "_uv_slp")
    _, calls = ldf.capture_return_locals(state["recall"], uv_code)
    if len(calls) != 2:
        raise SystemExit(f"expected U/V calls, got {len(calls)}")
    u_loc, _ = calls[0]
    return np.asarray(u_loc["zg"]), np.asarray(u_loc["zau"])


def _ulp_distance(a: np.ndarray, b: np.ndarray) -> int:
    ai = np.asarray(a, dtype=np.float64).view(np.int64)
    bi = np.asarray(b, dtype=np.float64).view(np.int64)
    ao = np.where(ai < 0, np.iinfo(np.int64).min - ai, ai)
    bo = np.where(bi < 0, np.iinfo(np.int64).min - bi, bi)
    return int(np.max(np.abs(ao.astype(object) - bo.astype(object))))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--mld-maps", type=Path, required=True)
    p.add_argument("--parent-artifact", type=Path, required=True)
    p.add_argument("--nemo-source", type=Path, required=True)
    p.add_argument("--nemo-binary", type=Path, required=True)
    p.add_argument("--expected-repo-sha", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    set_policy(PrecisionPolicy.fp64())
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise SystemExit("row30 zau probe requires CPU fp64")
    if os.environ.get("DINO_1226_LANE") != "d180":
        raise SystemExit("row30 zau probe requires DINO_1226_LANE=d180")
    if _git("rev-parse", "HEAD") != args.expected_repo_sha:
        raise SystemExit("repository HEAD differs from preregistered probe HEAD")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("tracked worktree must be clean")
    run = args.run_dir.resolve()
    if Path(dump_lane.RUN_DIR).resolve() != run:
        raise SystemExit("lane selector does not resolve to registered run")
    fixed = {
        args.parent_artifact: EXPECTED_ARTIFACT_SHA,
        args.nemo_binary: EXPECTED_BINARY_SHA,
        run / "nemo": EXPECTED_BINARY_SHA,
        run / "DINO_00005761_restart.nc": EXPECTED_RESTART_SHA,
        run / "eiv_dump_zgru_iik.bin": EXPECTED_ZGRU_SHA,
        run / "eiv_dump_zau.bin": EXPECTED_ZAU_SHA,
    }
    for path, expected in fixed.items():
        if sha256(path.resolve()) != expected:
            raise SystemExit(f"SHA receipt changed: {path}")
    source_text = args.nemo_source.read_text(errors="strict")
    quote = "zau = zgru(ji,jj,iik) * r1_e1u(ji,jj)"
    if quote not in source_text:
        raise SystemExit("active NEMO line-242 quote changed")

    state = ldf.build_state()
    zgru, production = _capture_zau(state)
    jpi, jpj, _, hls = loaders._read_dims(str(run))
    nemo_zgru = loaders._load_haloed(str(run / "eiv_dump_zgru_iik.bin"), jpi, jpj, hls)
    nemo_zau = loaders._load_haloed(str(run / "eiv_dump_zau.bin"), jpi, jpj, hls)
    nk = min(zgru.shape[-1], nemo_zgru.shape[-1], nemo_zau.shape[-1])
    wet = np.zeros_like(ldf.wet_u_mask(state["active"], state["u_mask"])[..., :nk])
    wet[..., ldf.KLO:min(ldf.KHI, nk)] = ldf.wet_u_mask(
        state["active"], state["u_mask"])[..., ldf.KLO:min(ldf.KHI, nk)]
    if not np.array_equal(zgru[..., :nk][wet], nemo_zgru[..., :nk][wet]):
        raise SystemExit("registered exact zgru baseline changed")

    e1_lego = np.asarray(state["grid"].dx_u[:, 1:], dtype=np.float64)
    with Dataset(run / "mesh_mask.nc") as ds:
        e1_halo = np.asarray(ds.variables["e1u"][0], dtype=np.float64)
    # mesh_mask is the global domain product and already omits the local
    # decomposition halos carried by the write-only stream buffers.
    e1_nemo = e1_halo
    if e1_nemo.shape != e1_lego.shape:
        raise SystemExit(f"metric shapes differ: {e1_nemo.shape} {e1_lego.shape}")

    zg = zgru[..., :nk]
    one = np.float64(1.0)
    np_leg_r = one / e1_lego
    np_nem_r = one / e1_nemo

    @jax.jit
    def jax_literal(z, e):
        r = jax.lax.optimization_barrier(jnp.asarray(1.0, z.dtype) / e)
        return jax.lax.optimization_barrier(z * r[:, :, None])

    candidates = {
        "production_capture": production[..., :nk],
        "lego_metric_division": zg / e1_lego[:, :, None],
        "lego_metric_reciprocal_multiply": zg * np_leg_r[:, :, None],
        "nemo_metric_division": zg / e1_nemo[:, :, None],
        "nemo_metric_numpy_reciprocal_multiply": zg * np_nem_r[:, :, None],
        "nemo_metric_jax_explicit_compile_reciprocal_multiply": np.asarray(
            jax_literal.lower(jnp.asarray(zg), jnp.asarray(e1_nemo)).compile()(jnp.asarray(zg), jnp.asarray(e1_nemo))),
        "nemo_metric_jax_jit_reciprocal_multiply": np.asarray(
            jax_literal(jnp.asarray(zg), jnp.asarray(e1_nemo))),
    }
    focus = sweep.focus_from_maps(args.mld_maps)
    scores = {name: sweep.metrics(val, nemo_zau[..., :nk], wet, focus, BAR)
              for name, val in candidates.items()}

    # Red-capable controls on the exact input and the literal reciprocal.
    controls = sweep.planted_controls(zg, nemo_zgru[..., :nk], wet, BAR)
    idx = tuple(np.argwhere(wet & (zg != 0.0))[0])
    j, i, k = idx
    r0 = np_nem_r[j, i]
    r1 = np.nextafter(r0, np.inf)
    controls["reciprocal_one_ulp_changes_product"] = bool(
        zg[idx] * r0 != zg[idx] * r1)
    controls["reciprocal_control_ji_k"] = [int(j), int(i), int(k)]
    if not all(v for v in controls.values() if isinstance(v, bool)):
        raise SystemExit("a planted row30 zau control did not fire")

    metric = {
        "exact_unequal": int(np.count_nonzero(e1_lego != e1_nemo)),
        "max_abs": float(np.max(np.abs(e1_lego - e1_nemo))),
        "max_ulp": _ulp_distance(e1_lego, e1_nemo),
    }
    confirmed = (
        scores["production_capture"]["n_diverged_columns"] == 190
        and not any(not x["pass"] for x in scores["production_capture"]["focus"])
        and scores["nemo_metric_jax_jit_reciprocal_multiply"]["pass"]
        and not any(not x["pass"] for x in scores["nemo_metric_jax_jit_reciprocal_multiply"]["focus"])
    )
    artifact = {
        "schema": "dino-zdf-row30-zau-operands-v1",
        "disposition": "CONFIRM-NEMO-RECIPROCAL" if confirmed else "REFUTE",
        "bar": BAR,
        "lane": dump_lane.banner(),
        "nemo_quote": quote,
        "metric_value": metric,
        "scores": scores,
        "controls": controls,
        "focus_columns_ji": [list(x) for x in focus],
        "repo_sha": args.expected_repo_sha,
        "provenance_sha256": {str(path.resolve()): sha256(path.resolve()) for path in (
            args.parent_artifact, args.nemo_source, args.nemo_binary, args.mld_maps,
            run / "mesh_mask.nc", run / "eiv_dump_zgru_iik.bin", run / "eiv_dump_zau.bin",
            run / "run.attempt1.log", run / "DINO_00005760_restart.nc",
            run / "DINO_00005761_restart.nc", Path(__file__), Path(ldf.__file__),
        )},
    }
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(f"metric exact_unequal={metric['exact_unequal']} max_abs={metric['max_abs']:.17g} max_ulp={metric['max_ulp']}")
    for name, score in scores.items():
        print(f"{name}: {score['n_diverged_columns']}/{score['n_wet_columns']} fail; focus_fail={sum(not x['pass'] for x in score['focus'])}")
    print(artifact["disposition"])
    print(f"wrote {args.output}")
    return 0 if confirmed else 30


if __name__ == "__main__":
    raise SystemExit(main())
