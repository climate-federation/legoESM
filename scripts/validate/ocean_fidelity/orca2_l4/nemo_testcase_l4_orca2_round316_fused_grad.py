#!/usr/bin/env python3
"""Round 316: one fused multi-step gradient of an ORCA2 card (measurement only).

ONE ``jax.jit(jax.grad(L o step^N))`` with respect to the initial T field:
step 1 (the cold-start state structure) followed by a ``lax.scan`` of
per-step ``jax.checkpoint`` bodies over steps 2..N (one compiled body, so
the reverse executable stays under the per-process memory-map limit).
Checked against forward mode over the SAME N-step function along the round-315
direction T_dir0, and the forward value against the production trajectory.
Round 317 ``--cost``: per-N T-loss reverse cost (compile, warm run, run peak
RSS, XLA buffer estimate, gradient norm) with or without per-step remat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round315_grad_probe as probe,
    nemo_testcase_l4_orca2_round315_nan_locate as locate,
)


def setup(rung: str):
    """(card, model, state0, forcing, losses, tmask, body) for one card."""

    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel

    probe.require(bool(jax.config.jax_enable_x64) and jax.default_backend() == "cpu",
                  "needs x64 on CPU")
    card, state0, forcing = probe.build(rung)
    r = card.recipe
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config,
                                  iwm_forcing=r.iwm_forcing)
    losses, tmask = probe.make_losses(model, r.z_coord)
    raw = type(model).__dict__["_step_jitted"].__wrapped__
    dt = card.dt_s

    def body(state, f):
        return raw(model, model._seed_tke_preclosure_carry(state), dt, *f)

    return card, model, state0, forcing, losses, tmask, body


def make_run(body, state0, forcing, n: int, remat: bool = True):
    """T0 -> state after n steps: step 1, then a scan over steps 2..n whose body
    is ``jax.checkpoint``-ed (remat) or plain (every step's residuals kept)."""

    import jax
    import jax.numpy as jnp

    rest = jax.tree.map(lambda *xs: jnp.stack(xs), *forcing[1:n]) if n > 1 else None
    step = jax.checkpoint(body) if remat else body

    def run(t0):
        st = body(probe._with_t(state0, t0), forcing[0])
        if n > 1:
            st, _ = jax.lax.scan(lambda s, f: (step(s, f), None), st, rest)
        return st

    return run


def fused(rung: str, n: int, losses_wanted: tuple[str, ...]) -> dict:
    import jax
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    card, model, state0, forcing, losses, tmask, body = setup(rung)
    n = min(n, len(forcing))
    dt = card.dt_s
    run = make_run(body, state0, forcing, n)

    # control: the fused forward equals the production trajectory, every leaf
    st = state0
    for k in range(n):
        st = model.step(st, dt, freshwater=forcing[k][0], surface_forcing=forcing[k][1])
    mine = jax.jit(run)(state0.T.data)
    pairs = zip(jax.tree.leaves(st), jax.tree.leaves(mine))
    same = all(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True) for a, b in pairs)
    d = locate.probe_direction(state0, tmask)
    out = {"rung": rung, "n": n, "worktree": worktree_stamp(allow_dirty=True),
           "forward_equals_production_every_leaf": same, "losses": {}}
    for name in losses_wanted:
        f = lambda t, name=name: losses[name](run(t))  # noqa: E731
        t0 = time.perf_counter()
        g = jax.jit(jax.grad(f))(state0.T.data)
        jax.block_until_ready(g)
        t_rev = time.perf_counter() - t0
        t0 = time.perf_counter()
        fwd = float(jax.jit(lambda t, v: jax.jvp(f, (t,), (v,))[1])(state0.T.data, d))
        t_fwd = time.perf_counter() - t0
        g = np.asarray(g)
        rev = float(np.vdot(g, np.asarray(d)))
        out["losses"][name] = {
            "grad_finite": bool(np.all(np.isfinite(g))),
            "nonfinite_entries": int(np.sum(~np.isfinite(g))),
            "grad_l2": float(np.linalg.norm(np.nan_to_num(g))),
            "rev_dot_dir0": rev, "fwd_dir0": fwd,
            "rev_vs_fwd_rel": abs(rev - fwd) / max(abs(fwd), 1e-300),
            "compile_plus_run_s": {"reverse": round(t_rev, 1), "forward": round(t_fwd, 1)}}
        print(name, out["losses"][name], flush=True)
    return out


class RssSampler:
    """Peak resident set size (bytes) of this process while the block runs."""

    def __init__(self, period_s: float = 0.02):
        import threading

        self.period_s, self.peak, self._stop = period_s, rss_bytes(), threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, rss_bytes())
            self._stop.wait(self.period_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, rss_bytes())


def rss_bytes() -> int:
    with open("/proc/self/statm") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def cost(rung: str, n_list: tuple[int, ...], remat: bool, jvp_at: tuple[int, ...],
         output: Path) -> dict:
    """T-loss reverse cost per N: compile and (warm) run seconds, the run's
    sampled peak RSS and XLA's own buffer estimate, gradient norm and finiteness.
    The JSON is rewritten after every N, so a stopped run keeps finished rows."""

    import jax
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    card, model, state0, forcing, losses, tmask, body = setup(rung)
    t0 = state0.T.data
    d = locate.probe_direction(state0, tmask)
    out = {"rung": rung, "remat": remat, "loss": "T", "n_list": list(n_list),
           "worktree": worktree_stamp(allow_dirty=True), "rows": {}}
    for n in n_list:
        probe.require(n <= len(forcing), f"N={n} > {len(forcing)} acquired steps")
        jax.clear_caches()
        run = make_run(body, state0, forcing, n, remat)
        f = lambda t, run=run: losses["T"](run(t))  # noqa: E731
        tic = time.perf_counter()
        compiled = jax.jit(jax.value_and_grad(f)).lower(t0).compile()
        t_compile = time.perf_counter() - tic
        mem = compiled.memory_analysis()
        base = rss_bytes()
        with RssSampler() as sampler:
            tic = time.perf_counter()
            value, g = jax.block_until_ready(compiled(t0))
            t_run = time.perf_counter() - tic
        g = np.asarray(g)
        row = {"compile_s": round(t_compile, 1), "backward_run_s": round(t_run, 2),
               "rss_before_run_gb": base / 1e9, "run_peak_rss_gb": sampler.peak / 1e9,
               "run_rss_growth_gb": (sampler.peak - base) / 1e9,
               "xla_temp_gb": mem.temp_size_in_bytes / 1e9,
               "xla_argument_gb": mem.argument_size_in_bytes / 1e9,
               "xla_output_gb": mem.output_size_in_bytes / 1e9,
               "loss": float(value), "loss_hex": float(value).hex(),
               "grad_finite": bool(np.all(np.isfinite(g))),
               "nonfinite_entries": int(np.sum(~np.isfinite(g))),
               "grad_l2": float(np.linalg.norm(np.nan_to_num(g))),
               "rev_dot_dir0": float(np.vdot(g, np.asarray(d)))}
        np.save(output.with_name(f"{output.stem}_grad_n{n}.npy"), g)
        if n in jvp_at:
            del compiled
            jax.clear_caches()
            tic = time.perf_counter()
            primal, tangent = jax.jit(lambda t, v: jax.jvp(f, (t,), (v,)))(t0, d)
            row["jvp_compile_plus_run_s"] = round(time.perf_counter() - tic, 1)
            row["jvp_loss_hex"] = float(primal).hex()
            row["jvp_primal_equals_reverse_primal"] = float(primal) == float(value)
            row["fwd_dir0"] = float(tangent)
            row["rev_vs_fwd_rel"] = probe.rel_err(row["rev_dot_dir0"], float(tangent))
        out["rows"][str(n)] = row
        output.write_text(json.dumps(out, indent=1, default=str) + "\n")
        print(f"N={n}", row, flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", choices=tuple(probe.ROOTS), required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--losses", nargs="+", default=["T", "ssh"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cost", type=int, nargs="+", metavar="N",
                        help="cost mode: T-loss reverse cost at each N (one process)")
    parser.add_argument("--no-remat", action="store_true",
                        help="cost mode: keep every step's residuals (no checkpoint)")
    parser.add_argument("--jvp-at", type=int, nargs="*", default=(), metavar="N",
                        help="cost mode: also forward mode (primal + tangent) at these N")
    args = parser.parse_args()
    if args.cost:
        cost(args.rung, tuple(args.cost), not args.no_remat, tuple(args.jvp_at), args.output)
        return 0
    result = fused(args.rung, args.steps, tuple(args.losses))
    args.output.write_text(json.dumps(result, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
