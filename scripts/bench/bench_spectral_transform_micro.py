"""Spectral-transform microbenchmark (feasibility inputs, audit item 9).

Times the spherical-harmonic ANALYSIS/SYNTHESIS pair (FFT-in-longitude +
dense Legendre GEMM) in isolation at several truncations, and reports the
GEMM problem SHAPES + FLOP counts + arithmetic intensity — the numbers the
GPU-native-transform go/no-go note
(docs/performance/scaling/spectral_gpu_feasibility.md) is grounded in.

MICROBENCHMARK ONLY: no dycore, no multi-device, no rewrite.  Honest about
its backend: rows carry the live backend + the shared metadata, so a
CPU-only laptop row can never masquerade as the missing GPU measurement
(the note lists the exact command to reproduce on a CUDA node).

Run:
  JAX_ENABLE_X64=1 python scripts/bench/bench_spectral_transform_micro.py \
      --truncations 42,85,170 --nlev 30 --out results/spectral_micro.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metadata import annotate_incomplete, scaling_metadata  # noqa: E402


def transform_flops(n_lat: int, n_lon: int, n_sh: int, nlev: int) -> dict:
    """Analytic cost model of ONE analysis+synthesis round trip.

    Legendre leg = dense GEMM pair, ``(n_lat x n_sh) @ (n_sh x
    nlev-batch)`` per longitudinal wavenumber block (one einsum over the
    packed ``(n_lat, n_sh, nlev)`` tensor).  The Legendre matrices are
    REAL and the spectral fields complex: a real x complex MAC is 4 real
    FLOPs (2 mul + 2 add), not 8 (codex).  FFT leg: ``5 * n * log2(n)``
    per real transform row.
    Arithmetic intensity vs the ``(n_lat, n_sh, nlev)`` working set is the
    GPU-viability number (fp64 GEMM runs at tensor-core-less rates on most
    consumer parts, 1:2..1:64 of fp32 — shapes must be fat enough to be
    compute-bound to benefit at all).
    """
    gemm_macs = 2 * n_lat * n_sh * nlev          # analysis + synthesis
    gemm_flops = 4 * gemm_macs                    # real x complex MAC = 4
    fft_flops = 2 * nlev * n_lat * 5 * n_lon * int(np.log2(max(n_lon, 2)))
    bytes_ws = (n_lat * n_sh + n_sh * nlev + n_lat * nlev) * 16  # complex128
    return {
        "gemm_flops": int(gemm_flops),
        "fft_flops": int(fft_flops),
        "gemm_fraction_of_flops": float(
            gemm_flops / max(gemm_flops + fft_flops, 1)),
        "arithmetic_intensity_flop_per_byte": float(
            gemm_flops / max(bytes_ws, 1)),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--truncations", type=str, default="42,85")
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--sh-gemm", choices=["on", "off", "both"],
                   default="both",
                   help="Legendre path: 'on' = the opt-in batched-GEMM "
                        "(LEGOESM_SH_GEMM=1, the cuBLAS-lowering path the "
                        "feasibility note is about), 'off' = the legacy "
                        "gather/segment-sum default, 'both' = one row per "
                        "mode (the mode is recorded on every row — a "
                        "legacy row can never masquerade as the GEMM "
                        "measurement; codex).")
    p.add_argument("--out", type=str,
                   default="results/a1/spectral_transform_micro.json")
    args = p.parse_args()

    truncs = [int(x) for x in args.truncations.split(",") if x]
    if not truncs or any(t < 10 for t in truncs):
        raise SystemExit("--truncations needs integers >= 10")
    if args.repeats < 1 or args.warmup < 0:
        raise SystemExit("--repeats >= 1, --warmup >= 0 required")

    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm.grids.gaussian import (
        create_gaussian_grid,
        sh_analysis_3d,
        sh_synthesis_3d,
    )

    modes = {"on": [True], "off": [False],
             "both": [False, True]}[args.sh_gemm]
    _prev_gemm_env = os.environ.get("LEGOESM_SH_GEMM")
    rows = []
    for T in truncs:
        grid = create_gaussian_grid(n_max=T)
        n_lat, n_lon = int(grid.n_lat), int(grid.n_lon)
        n_sh = int(grid.n_sh)
        rng = np.random.default_rng(0)
        field = jnp.asarray(
            rng.standard_normal((n_lat, n_lon, args.nlev)))

        for gemm_mode in modes:
            # The Legendre-path switch is read at TRACE time — set it
            # before building the jitted round trip (a fresh closure per
            # mode forces a fresh trace).
            os.environ["LEGOESM_SH_GEMM"] = "1" if gemm_mode else "0"

            @jax.jit
            def round_trip(f, _grid=grid):
                return sh_synthesis_3d(_grid, sh_analysis_3d(_grid, f))

            # Band-limit FIRST: synthesis∘analysis is a PROJECTION, so a
            # random grid field's first round trip includes truncation
            # error by construction.  The SECOND application measures the
            # transform's own error on an exactly band-limited input
            # (codex).
            f_band = round_trip(field)
            jax.block_until_ready(f_band)
            out = round_trip(f_band)
            jax.block_until_ready(out)
            times = []
            for _ in range(args.warmup + args.repeats):
                t0 = time.perf_counter()
                out = round_trip(f_band)
                jax.block_until_ready(out)
                times.append(time.perf_counter() - t0)
            med_s = float(np.median(times[args.warmup:]))
            cost = transform_flops(n_lat, n_lon, n_sh, args.nlev)
            achieved = cost["gemm_flops"] / med_s / 1e9
            mode_name = "gemm" if gemm_mode else "legacy_segment_sum"
            row = {
                "truncation": T,
                "legendre_path": mode_name,
                # Live backend ON THE ROW (not only in the shared metadata
                # block): a CPU row must be self-labeling even when a
                # consumer copies the rows table alone (codex).
                "backend": str(jax.default_backend()),
                "device": str(jax.devices()[0]),
                "n_lat": n_lat, "n_lon": n_lon, "n_sh": n_sh,
                "nlev": args.nlev,
                "round_trip_median_ms": round(med_s * 1e3, 3),
                "equivalent_gemm_gflops": round(achieved, 2),
                "band_limited_round_trip_error_max": float(
                    jnp.abs(out - f_band).max()),
                **cost,
            }
            rows.append(row)
            print(f"T{T:4d} [{mode_name:18s}] | {n_lat}x{n_lon}, "
                  f"n_sh={n_sh} | "
                  f"round-trip {row['round_trip_median_ms']:9.3f} ms | "
                  f"GEMM-model {row['gemm_flops'] / 1e9:7.2f} GF "
                  f"({100 * row['gemm_fraction_of_flops']:.0f}%) | "
                  f"equiv {achieved:7.2f} GF/s | "
                  f"AI {row['arithmetic_intensity_flop_per_byte']:.1f} F/B "
                  f"| err {row['band_limited_round_trip_error_max']:.1e}")

    # Restore the caller's Legendre-path env (the loop mutates a
    # process-global switch; an importing test must not inherit the last
    # mode; codex).
    if _prev_gemm_env is None:
        os.environ.pop("LEGOESM_SH_GEMM", None)
    else:
        os.environ["LEGOESM_SH_GEMM"] = _prev_gemm_env

    payload = {
        "rows": rows,
        "metadata": annotate_incomplete(scaling_metadata(
            grid="spectral",
            component="atmosphere",
            resolution=",".join(f"T{t}" for t in truncs),
            n_levels=args.nlev,
            precision="float64",
            decomposition="none",
            solver_variant="sh_transform_round_trip",
            scaling_kind="throughput",
            transport="none",
            extra={"repeats": args.repeats, "warmup": args.warmup,
                   "sh_gemm_modes": args.sh_gemm},
        )),
    }
    outdir = os.path.dirname(args.out)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"JSON: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
