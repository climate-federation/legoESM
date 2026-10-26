"""Gate: the FV3-duo land step split over N devices (``solve_land_on_devices``)
equals the one-call step to a per-leaf tolerance, on a real land state.

Setup is production's: ``run_amip.main`` with a run's command line plus
``--restart-from <checkpoint>``; only ``ModelDriver.run`` is replaced.  The
inputs are the loop's land call on that checkpoint: the packed land columns,
the checkpoint's pending forcing window meaned as the loop means it, the land
tile's config/params/lat/carbon.  Each N in ``--devices`` runs on the first N
devices; the reference is the same step compiled on one device.  Exit 1 when
any leaf is outside ``TOL_REL`` / ``TOL_ABS`` or a counter differs.

    XLA_FLAGS=--xla_force_host_platform_device_count=6 JAX_PLATFORMS=cpu \\
    python scripts/validate/validate_duo_land_device_split.py \\
        --checkpoint CKPT --devices 2 6 -- <run_amip args>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# One land call on the day-60 C24 state, #2039's contiguous split emulated on
# one A100 at 2 and 6 shards (user-approved rule: 10x the worst measured): the
# worst leaf relative to its peak was 1.6e-11; the two energy-budget residual
# diagnostics are themselves rounding (peak ~1e-13) and are bounded in
# absolute terms (worst measured 3.4e-13).
TOL_REL = 1.6e-10
TOL_ABS = {"residual_int": 3.4e-12, "residual_ext": 3.4e-12}


def _finite_pair(a, b):
    """``(fin, mismatch)``: entries finite on both sides, and whether the
    non-finite entries differ in placement or value (NaN vs inf, sign)."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    bad = bool(np.any(fa != fb)) or not np.array_equal(
        a[~fa], b[~fb], equal_nan=True)
    return fa & fb, bad


def leaf_abs(ref, got) -> dict:
    """``{leaf path: max|got-ref|}`` over entries finite on both sides
    (``inf`` when the non-finite entries differ)."""
    import jax
    out = {}
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(ref)[0],
                            jax.tree_util.tree_leaves(got)):
        a, b = np.asarray(a, float), np.asarray(b, float)
        fin, bad = _finite_pair(a, b)
        d = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
        out[jax.tree_util.keystr(path)] = float("inf") if bad else d
    return out


def _leaf_name(path: str) -> str:
    """Last field name of a pytree key path (``"[3].residual_int"``)."""
    return path.rsplit(".", 1)[-1].strip("[]'")


def over_tolerance(rel: dict, absd: dict) -> dict:
    """Leaves outside the gate: residual diagnostics against their absolute
    bound, every other leaf against ``TOL_REL``."""
    bad = {}
    for k, r in rel.items():
        name = _leaf_name(k)
        if name in TOL_ABS:
            if absd[k] > TOL_ABS[name]:
                bad[k] = absd[k]
        elif r > TOL_REL:
            bad[k] = r
    return bad


def leaf_rel(ref, got) -> dict:
    """``{leaf path: max|got-ref| / max|ref|}`` over floating leaves, entries
    finite on both sides; differing non-finite entries or a nonzero diff on a
    zero-peak leaf is ``inf`` (never passes a tolerance)."""
    import jax
    out = {}
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(ref)[0],
                            jax.tree_util.tree_leaves(got)):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"{jax.tree_util.keystr(path)}: {a.shape} "
                             f"{a.dtype} vs {b.shape} {b.dtype}")
        if a.dtype.kind != "f":
            out[jax.tree_util.keystr(path)] = (0.0 if np.array_equal(a, b)
                                               else float("inf"))
            continue
        fin, bad = _finite_pair(a, b)
        peak = float(np.max(np.abs(a[fin]))) if fin.any() else 0.0
        d = float(np.max(np.abs(a[fin] - b[fin]))) if fin.any() else 0.0
        rel = d / peak if peak > 0 else (0.0 if d == 0 else float("inf"))
        out[jax.tree_util.keystr(path)] = float("inf") if bad else rel
    return out


def _gate_run(opts):
    def run(self, start_step: int = 0, start_day=None, **_kw):
        import jax
        import jax.numpy as jnp
        from jax.sharding import Mesh
        from legoesm.driver.model_driver import (
            land_window_from_checkpoint, mpas_land_step_params,
            shard_land_columns, solve_land_on_devices)
        from legoesm.land.multilayer_land import (
            gather_land_columns, step_multilayer_land_with_diagnostics)

        cfg = self.config
        dt = float(cfg.dycore.dt)
        land_k = (max(1, int(round(float(cfg.land_update_seconds) / dt)))
                  if float(getattr(cfg, "land_update_seconds", 0.0)) > 0.0
                  else 1)
        ph = self.physics
        f_land = np.asarray(self._f_land).reshape(-1)
        ncol = int(f_land.shape[0])
        cols = np.nonzero(f_land > 0.0)[0]
        payload = {k: v for k, v in (self._carry_aux or {}).items()
                   if k.startswith(("land_win_", "land_held_"))}
        a2s_sum, n, cz, w, _ = land_window_from_checkpoint(payload, land_k,
                                                           ncol)
        if n == 0:
            raise SystemExit("gate: checkpoint has no pending land window")
        a2s = jax.tree_util.tree_map(lambda s: s / n, a2s_sum)
        a2s = a2s._replace(cos_zenith=cz / w)
        doy, _ = self._calendar_for_radiation(float(start_day))
        refresh = bool(getattr(cfg, "land_params_refresh", False))
        upd = None
        if refresh:
            upd = getattr(ph, "land_ml_params_update", None)
            if upd is None:
                upd = ph.land_ml_params_update_factory()
        year0 = float(getattr(cfg, "start_year", 0.0) or 0.0)
        umin = float(getattr(ph, "land_ml_u_min", 1.0))
        dt_land = land_k * dt

        def step(st, a, c, p, la, cb, d):
            return step_multilayer_land_with_diagnostics(
                st, a, c, umin, dt_land, lat=la, doy=d, land_params=p,
                carbon_state=cb)

        # As the loop's land closure: one jit over the FULL state, forcing
        # and day; params refreshed inside it when the run refreshes them,
        # then packed to the slots; config, latitude, carbon closed over.
        def call(idx, split=None):
            idx = jnp.asarray(idx)

            def g(st, a, d, lai_doy, lai_year):
                p = (mpas_land_step_params(upd, ph.land_ml_params,
                                           st.theta_soil[:, 0], lai_doy,
                                           lai_year)
                     if refresh else ph.land_ml_params)
                full = (st, a, ph.land_ml_cfg, p, ph.land_ml_lat,
                        ph.land_ml_carbon, d)
                args = gather_land_columns(full, idx, ncol)
                if split is None:
                    return step(*args)
                is_col = jax.tree_util.tree_map(
                    lambda x: (hasattr(x, "shape") and len(x.shape) >= 1
                               and x.shape[0] == ncol), full)
                return solve_land_on_devices(step, args, is_col, *split)
            return jax.jit(g)(
                self._land_ml_state, a2s, jnp.asarray(doy, jnp.float64),
                jnp.asarray(doy - 1.0),
                jnp.asarray(year0 + start_day / 365.0))

        ref = jax.device_get(call(cols))      # device 0
        report, failed = {}, False
        for nd in opts.devices:
            if jax.device_count() < nd:
                raise SystemExit(f"gate: need {nd} devices, have "
                                 f"{jax.device_count()}")
            slots, keep = shard_land_columns(cols, nd)
            mesh = Mesh(np.array(jax.devices()[:nd]), ("land",))
            raw = call(slots, (mesh, keep))
            # report only: the leaves not per column, and whether every
            # device's copy of each is identical
            shared = []
            for path, x in jax.tree_util.tree_flatten_with_path(raw)[0]:
                if x.ndim and x.shape[0] == len(cols):
                    continue
                c = [np.asarray(s.data).tobytes()
                     for s in x.addressable_shards]
                shared.append(f"{jax.tree_util.keystr(path)}:"
                              f"{'agree' if len(set(c)) == 1 else 'DIFFER'}")
            print(f"SHARED devices={nd}: {len(shared)} leaves: "
                  f"{' '.join(shared)}", flush=True)
            got = jax.device_get(raw)
            # whole-run counters are recounted on the split side: compare
            # the per-column results and those counts, not the step's own
            # scalars (identical by construction on the one-call side)
            strip = lambda t: t[:3] + (t[3]._replace(  # noqa: E731
                n_held=None, n_fallback=None, n_fallback_rejected=None),)
            rel = leaf_rel(strip(ref), strip(got))
            absd = leaf_abs(strip(ref), strip(got))
            bad = over_tolerance(rel, absd)
            # the recounted whole-run counters must match exactly
            for k in ("n_held", "n_fallback", "n_fallback_rejected"):
                a, b = getattr(ref[3], k), getattr(got[3], k)
                if (a is None) != (b is None) or (
                        a is not None and int(a) != int(b)):
                    bad[f"surface.{k}"] = float("inf")
            report[nd] = {"rel": rel, "abs": absd, "bad": bad}
            failed = failed or bool(bad)
            res = {k for k in rel if _leaf_name(k) in TOL_ABS}
            print(f"GATE devices={nd}: {len(rel)} leaves, worst rel "
                  f"{max(v for k, v in rel.items() if k not in res):.3e} "
                  f"(tol {TOL_REL:.1e}), residuals worst abs "
                  f"{max((absd[k] for k in res), default=0.0):.3e} "
                  f"(tol {max(TOL_ABS.values()):.1e}): "
                  f"{'PASS' if not bad else 'FAIL ' + json.dumps(bad)}",
                  flush=True)
        if opts.out:
            Path(opts.out).write_text(json.dumps(
                {"tol_rel": TOL_REL, "tol_abs": TOL_ABS,
                 "checkpoint": opts.checkpoint,
                 "devices": {str(k): v for k, v in report.items()}},
                indent=1))
        raise SystemExit(1 if failed else 0)
    return run


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: ... --checkpoint C -- <run_amip args>")
    cut = argv.index("--")
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--devices", type=int, nargs="+", default=[2, 6])
    p.add_argument("--out", default=None)
    opts = p.parse_args(argv[:cut])
    sys.path.insert(0, str(Path.cwd()))
    from legoesm.driver.model_driver import ModelDriver
    ModelDriver.run = _gate_run(opts)
    from scripts.run import run_amip
    run_amip.main(argv[cut + 1:] + ["--restart-from", opts.checkpoint])


if __name__ == "__main__":
    main()
