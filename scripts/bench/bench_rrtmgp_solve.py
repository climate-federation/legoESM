"""Time one RRTMGP solve, forward and optionally reverse mode."""
import argparse
import json
import subprocess
import time

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig  # noqa: E402
from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP  # noqa: E402


def inputs(ncol, nlay):
    ph = jnp.tile(jnp.linspace(225.0, 1.0e5, nlay + 1), (ncol, 1))
    pf = 0.5 * (ph[:, 1:] + ph[:, :-1])
    temperature = 200.0 + 90.0 * pf / 1e5
    on = (jnp.arange(ncol) % 2 == 0)[:, None]
    liq = jnp.where(on & (pf > 8.5e4) & (pf < 9.2e4), 0.05, 0.0)
    ice = jnp.where(on & (pf > 2.0e4) & (pf < 3.0e4), 0.01, 0.0)
    return temperature, dict(
        p_full=pf, p_half=ph, q_v=1e-2 * (pf / 1e5) ** 3, cloud_path_liq=liq, cloud_path_ice=ice,
        cloud_r_eff_liq=jnp.full_like(pf, 1e-5), cloud_r_eff_ice=jnp.full_like(pf, 4e-5),
        cos_zenith=jnp.full((ncol,), 0.5), sfc_albedo=0.1, sfc_emissivity=0.98)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncol", type=int, default=4096)
    ap.add_argument("--nlay", type=int, default=32)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--no-checkpoint", action="store_true", help="plain g-point scan, as the CPU deck")
    ap.add_argument("--grad", action="store_true", help="also time reverse mode")
    a = ap.parse_args()
    temperature, kw = inputs(a.ncol, a.nlay)
    solver = RRTMGP.from_legoesm_config(RRTMGPConfig(
        compute_fp32=True, overhead_layer=True, use_scan=True, include_clouds=True,
        gpoint_batch_size=a.batch, gpoint_checkpoint=not a.no_checkpoint))
    f = jax.jit(lambda t: solver.solve_columns(T=t, sfc_temperature=t[:, -1], **kw).heating_rate)
    t0 = time.time()
    f(temperature).block_until_ready()
    compile_s = time.time() - t0
    ts = []
    for i in range(a.reps):
        t0 = time.time()
        f(temperature + 0.01 * i).block_until_ready()
        ts.append(time.time() - t0)
    extra = {}
    if a.grad:
        g = jax.jit(jax.grad(lambda t: jnp.sum(f(t))))
        g(temperature).block_until_ready()
        gs = []
        for i in range(3):
            t0 = time.time()
            r = g(temperature + 0.01 * i)
            r.block_until_ready()
            gs.append(time.time() - t0)
        extra = dict(grad_ms=min(gs) * 1e3, grad_finite=bool(jnp.isfinite(r).all()))
    stats = jax.devices()[0].memory_stats() or {}
    extra["peak_gb"] = stats.get("peak_bytes_in_use", 0) / 2**30
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    print("RESULT " + json.dumps(dict(vars(a), step_ms=min(ts) * 1e3, median_ms=sorted(ts)[len(ts) // 2] * 1e3,
                                      compile_s=compile_s, device=str(jax.devices()[0]), sha=sha, **extra)), flush=True)


if __name__ == "__main__":
    main()
