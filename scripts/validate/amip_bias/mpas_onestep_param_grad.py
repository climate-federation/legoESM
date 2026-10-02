#!/usr/bin/env python
"""Multi-step parameter adjoint on the PRODUCTION MPAS AMIP lane (feasibility spike).

Question: can ``jax.grad`` flow from scalars built on K model steps (dynamics +
full physics, radiation on its production cadence) back to EVERY tier-1/2
float tunable of the schemes this run resolves, so that ONE backward pass
replaces a finite-difference ladder for screening levers?

HOW THE PRODUCTION SETUP IS REUSED, NOT RE-DERIVED: the script runs
``run_amip.main`` with the run's own launch arguments (read from its slurm
log), intercepts ``make_physics`` to capture the resolved ``PhysicsConfig`` and
the builder kwargs of BOTH variants (full radiation / held radiation), and
intercepts the model's FIRST ``step`` call to capture the exact (state, dt,
forcing, phys_state) the driver passes.  The driver's step loop is then
abandoned (host callbacks, eager side channels) and the UN-JITTED step
``_step_jit.__wrapped__`` is rolled K steps under ``jax.checkpoint`` per step
inside ``jax.jit(jax.value_and_grad(...))``, with the physics rebuilt from a
config whose tunable leaves are TRACED (``apply_param_overrides`` inside the
loss, SegmentForcing doctrine).

CAVEATS, stated: the FORCING IS HELD at its first-step value (SST, diurnal
angle, land fluxes) for all K steps -- the driver's per-step forcing is built
by ~2000 lines of orchestration and is not threaded here; TOA fluxes exist
only on radiation steps, so the "window mean" is the mean over the radiation
steps inside the window (K=24 -> one sample).

Losses (end state / window): tropical (20S-20N) free-troposphere (300-700 hPa)
q_v [g/kg] and RH, window-mean rsut/rlut global and tropical.  Every knob
enters as production_value * exp(theta), so a gradient is "per e-fold of the
knob"; two are checked against +-5 % central differences (accept <= 5 %:
fp32 physics at 41k columns).  Not a test: prints, exit 1 on failure.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path("/work/bd1083/b309178/diffESM/legoesm_pg/amip_runs")
FD_KNOBS = ("atm.conv.BechtoldConfig.epsilon_deep", "atm.clouds.CloudConfig.rh_crit")


class _Captured(BaseException):
    """Escapes run_amip.main's ``except Exception`` handlers."""


def launch_argv(run: str, day: int, out: Path) -> list[str]:
    logs = sorted((ROOT / "slurm_logs").glob(f"{run}-*.out"), key=os.path.getmtime)
    # packed campaign bundles log into the run directory instead
    logs = logs or sorted((ROOT / run).glob("slurm-*.out"), key=os.path.getmtime)
    line = None
    for lg in reversed(logs):
        for ln in open(lg, errors="replace"):
            if "[chain] launching" in ln and "run_amip.py" in ln:
                line = ln
                break
        if line:
            break
    if line is None:
        raise SystemExit(f"{run}: no '[chain] launching' line in its slurm logs")
    argv = shlex.split(line.split("run_amip.py", 1)[1])

    def drop(flag, n=2):
        while flag in argv:
            i = argv.index(flag)
            del argv[i:i + n]
    for f in ("--output", "--restart-from", "--days", "--max-wallclock-seconds",
              "--checkpoint-days"):
        drop(f)
    ckpt = ROOT / run / f"checkpoint_day_{day:04d}.npz"
    if not ckpt.exists():
        raise SystemExit(f"missing {ckpt}")
    argv += ["--output", str(out), "--restart-from", str(ckpt), "--days", "1"]
    return argv


def get_path(cfg, path):
    for a in path.split("."):
        cfg = getattr(cfg, a)
    return cfg


def set_path(cfg, path, value):
    head, _, rest = path.partition(".")
    from legoesm.core.param_overrides import apply_param_overrides
    if not rest:
        return apply_param_overrides(cfg, {head: value})
    return cfg._replace(**{head: set_path(getattr(cfg, head), rest, value)})


def walk_configs(cfg, prefix=""):
    """(path, NamedTuple) for every nested NamedTuple config."""
    out = [(prefix, cfg)]
    for f in getattr(cfg, "_fields", ()):
        sub = getattr(cfg, f)
        if hasattr(sub, "_fields"):
            out += walk_configs(sub, f"{prefix}.{f}" if prefix else f)
    return out


def enumerate_knobs(phys_cfg, tier_max=2):
    """{qualified_name: (config path, production value)} for every scalar
    float tunable of tier 1..tier_max whose config class is ACTIVE in this run
    (the category's selected scheme, or a non-scheme sub-config such as the
    cloud config)."""
    from legoesm.training.param_collector import build_registry
    active_paths = {}
    for path, sub in walk_configs(phys_cfg):
        parts = path.split(".") if path else []
        ok = True
        for depth in range(1, len(parts)):
            parent = get_path(phys_cfg, ".".join(parts[:depth]))
            sel = getattr(parent, "scheme", None)
            name = parts[depth]
            if sel is not None and name in _scheme_named_subconfigs(parent) \
                    and name not in str(sel).split("+"):
                ok = False
        if ok:
            active_paths.setdefault(type(sub).__name__, []).append(path)
    knobs, skipped = {}, []
    for m in build_registry():
        if not (1 <= m.tunable_tier <= tier_max) or m.shape_key is not None:
            continue
        paths = active_paths.get(m.config_class)
        if not paths:
            skipped.append((m.qualified_name, "inactive scheme"))
            continue
        for p in paths:
            full = f"{p}.{m.field}" if p else m.field
            try:
                v = get_path(phys_cfg, full)
            except AttributeError:
                skipped.append((m.qualified_name, f"no field at {full}"))
                continue
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                skipped.append((m.qualified_name, f"non-scalar leaf {type(v).__name__}"))
                continue
            if float(v) == 0.0:
                skipped.append((m.qualified_name, "production value 0 (multiplier cannot move it)"))
                continue
            knobs[m.qualified_name if len(paths) == 1 else f"{m.qualified_name}@{p}"] = (full, float(v))
    return knobs, skipped


def _scheme_named_subconfigs(parent):
    """Sub-config field names that are alternative SCHEMES of ``parent``
    (their class name ends in Config and they are named like a scheme)."""
    names = set()
    for f in parent._fields:
        sub = getattr(parent, f)
        if hasattr(sub, "_fields") and f not in ("cloud_config", "ozone", "orbit"):
            names.add(f)
    return names


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="dd_ctl")
    ap.add_argument("--day", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--fd-rel", type=float, default=0.05)
    ap.add_argument("--fd-tol", type=float, default=0.05)
    ap.add_argument("--no-fd", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    K = args.steps

    scratch = Path(args.out or f"/work/bd1083/b309178/diffESM/legoesm_pg/_wave3_grad/_onestep_{args.run}")
    scratch.mkdir(parents=True, exist_ok=True)
    argv_run = launch_argv(args.run, args.day, scratch)
    run_cfg = json.load(open(ROOT / args.run / "experiment_config.json"))
    RAD_EVERY = max(1, int(run_cfg.get("rad_update_steps", 1)))

    # ---- intercept the production builders --------------------------------
    import legoesm.atmosphere.physics.combined as combined
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel)
    real_make_physics = combined.make_physics
    builds: list[tuple] = []

    def spy_make_physics(config, *a, **kw):
        builds.append((config, a, dict(kw)))
        return real_make_physics(config, *a, **kw)

    cap: dict = {}

    def spy_step(self, state, dt, physics_fn=None, forcing=None, phys_state=None):
        cap.update(model=self, state=state, dt=dt, physics_fn=physics_fn,
                   forcing=forcing, phys_state=phys_state)
        raise _Captured()

    combined.make_physics = spy_make_physics
    MPASPrimitiveEquationModel.step = spy_step
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "run"))
    import run_amip
    try:
        run_amip.main(argv_run)
    except _Captured:
        pass
    else:
        raise SystemExit("driver returned without ever calling model.step")
    if not cap:
        raise SystemExit("step was never intercepted")
    full = [b for b in builds if b[2].get("need_rad", True) is not False]
    held = [b for b in builds if b[2].get("need_rad", True) is False]
    if not full:
        raise SystemExit("no full-radiation make_physics call captured")
    phys_cfg, pargs, pkw = full[-1]
    hkw = held[-1][2] if held else dict(pkw, need_rad=False)
    model, state, DT = cap["model"], cap["state"], float(cap["dt"])
    forcing, phys_state = cap["forcing"], cap["phys_state"]
    print(f"captured: dt={DT} K={K} rad_every={RAD_EVERY} builds={len(builds)} "
          f"forcing_keys={sorted(forcing) if isinstance(forcing, dict) else type(forcing)} "
          f"phys_state={'None' if phys_state is None else 'present'}", flush=True)
    eps_path = "convection.bechtold.epsilon_deep"
    if abs(float(get_path(phys_cfg, eps_path)) - 3.0e-3) > 1e-12:
        raise SystemExit("captured PhysicsConfig does not carry the run's --params "
                         f"(epsilon_deep={get_path(phys_cfg, eps_path)})")

    knobs, skipped = enumerate_knobs(phys_cfg)
    print(f"knobs ({len(knobs)}):")
    for k, (p, v) in knobs.items():
        print(f"  {k:55s} {p:50s} {v:.6g}")
    atm_skipped = [(k, why) for k, why in skipped if k.startswith("atm.")]
    print(f"skipped ({len(skipped)}, {len(atm_skipped)} atmospheric; only those listed):")
    for k, why in atm_skipped:
        print(f"  {k:55s} {why}")
    for f in FD_KNOBS:
        if f not in knobs:
            raise SystemExit(f"FD knob {f} not in the enumerated set")

    target_mass = model._target_mass
    if model.config.fix_mass and model.config.anchor_mass_to_initial and target_mass is None:
        target_mass = model.compute_mass(state)
    step_raw = type(model)._step_jit.__wrapped__

    mesh, sig = model.mesh, model.sigma_coord
    lat = jnp.asarray(mesh.latCell)
    area = jnp.asarray(mesh.areaCell)
    trop = (jnp.abs(lat) <= jnp.deg2rad(20.0)).astype(area.dtype)
    sigma_full = jnp.asarray(sig.sigma_full)
    dsigma = jnp.asarray(sig.dsigma)
    w_glob = area / jnp.sum(area)
    w_trop = trop * area / jnp.sum(trop * area)
    from legoesm.core.state import MPAS_SFC_DIAG_EXTRA_KEYS
    from legoesm.thermo import saturation_mixing_ratio
    i_lw = 3 + MPAS_SFC_DIAG_EXTRA_KEYS.index("lw_up_toa")
    i_sw = 3 + MPAS_SFC_DIAG_EXTRA_KEYS.index("sw_up_toa")

    def _arr(x):
        return getattr(x, "data", x)

    def ft_means(st):
        q = st.tracers["q_v"].data
        T = st.T.data
        p_full = sigma_full[None, :] * st.p_s.data[:, None]
        ft = ((p_full >= 300e2) & (p_full <= 700e2)).astype(area.dtype)
        w = ft * dsigma[None, :] * (trop * area * st.p_s.data)[:, None]   # air-mass weights
        w = w / jnp.sum(w)
        rh = q / jnp.maximum(saturation_mixing_ratio(T, p_full), 1e-12)
        return jnp.sum(w * q) * 1e3, jnp.sum(w * rh)

    def rollout(theta):
        cfg = phys_cfg
        for k, (p, v) in knobs.items():
            cfg = set_path(cfg, p, v * jnp.exp(theta[k]))
        pf_full = real_make_physics(cfg, *pargs, **pkw)
        pf_held = real_make_physics(cfg, *pargs, **hkw)

        def _step(carry, pf):
            st, ps = carry
            new, pout, sfc, _led = step_raw(model, st, DT, pf, target_mass,
                                            forcing, ps)
            return (new, pout), sfc

        rad_step = jax.checkpoint(lambda c, _: _step(c, pf_full))
        held_step = jax.checkpoint(lambda c, _: _step(c, pf_held))

        carry = (state, phys_state)
        toa = []
        remaining = K
        while remaining > 0:
            carry, sfc = rad_step(carry, None)
            toa.append((jnp.sum(w_glob * _arr(sfc[i_sw])), jnp.sum(w_glob * _arr(sfc[i_lw])),
                        jnp.sum(w_trop * _arr(sfc[i_sw])), jnp.sum(w_trop * _arr(sfc[i_lw]))))
            remaining -= 1
            n_held = min(RAD_EVERY - 1, remaining)
            if n_held > 0:
                carry, _ = jax.lax.scan(held_step, carry, None, length=n_held)
                remaining -= n_held
        q_ft, rh_ft = ft_means(carry[0])
        # The step-0 radiation sample is computed BEFORE any tendency of the
        # window applies (a zero-step sensitivity: only radiation-facing knobs
        # can move it), so once later samples exist it is dropped (GLM).
        toa = jnp.asarray(toa[1:] if len(toa) > 1 else toa).mean(axis=0)
        return {"q_ft_trop": q_ft, "rh_ft_trop": rh_ft,
                "rsut_glob": toa[0], "rlut_glob": toa[1],
                "rsut_trop": toa[2], "rlut_trop": toa[3]}

    theta0 = {k: jnp.zeros(()) for k in knobs}
    res = {"run": args.run, "day": args.day, "dt": DT, "K": K, "rad_every": RAD_EVERY,
           "n_knobs": len(knobs), "value": {}, "grad": {}, "fd": {}}

    def mem():
        try:
            return jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30
        except Exception:
            return float("nan")

    names = ["q_ft_trop", "rh_ft_trop", "rsut_glob", "rlut_glob", "rsut_trop", "rlut_trop"]
    print(f"TOA window: {'step-0 sample only (zero-step sensitivity)' if K <= RAD_EVERY else 'radiation steps after step 0'}")

    def rollout_vec(th):
        r = rollout(th)
        return jnp.stack([r[n] for n in names])

    # ONE compiled program: forward + one VJP; the six losses are six
    # cotangent basis vectors through the same executable (no recompile).
    @jax.jit
    def value_and_vjp(th, ct):
        y, f_vjp = jax.vjp(rollout_vec, th)
        return y, f_vjp(ct)[0]

    for i, name in enumerate(names):
        t1 = time.time()
        ct = jnp.zeros((len(names),)).at[i].set(1.0)
        y, g = value_and_vjp(theta0, ct)
        v = float(y[i]); g = {k: float(x) for k, x in g.items()}
        res["value"][name] = v
        res["grad"][name] = g
        top = sorted(g.items(), key=lambda kv: -abs(kv[1]))[:5]
        print(f"{name}: value={v:.4f} [{time.time() - t1:.0f}s, peak {mem():.1f} GiB] "
              f"top5={[(k.split('.')[-1], round(x, 5)) for k, x in top]}", flush=True)
        with open(scratch / f"multistep_grad_K{K}.json", "w") as fh:
            json.dump(res, fh, indent=1)

    # ---- ranked tables + sign agreement ------------------------------------
    gq, gs, gt = res["grad"]["rh_ft_trop"], res["grad"]["rsut_glob"], res["grad"]["rsut_trop"]
    gl = res["grad"]["rlut_glob"]
    print(f"\nRANKED |dL/dln theta| (K={K})   L=RH_ft_trop   L=rsut_glob   L=rsut_trop   L=rlut_glob")
    for k in sorted(knobs, key=lambda k: -abs(gq[k]))[:15]:
        print(f"  {k:55s} {gq[k]:+.3e}   {gs[k]:+.3e}   {gt[k]:+.3e}   {gl[k]:+.3e}")
    print(f"\nRANKED by |d rsut_glob|:")
    for k in sorted(knobs, key=lambda k: -abs(gs[k]))[:15]:
        print(f"  {k:55s} {gq[k]:+.3e}   {gs[k]:+.3e}   {gt[k]:+.3e}   {gl[k]:+.3e}")
    # A move of the knob that LOWERS both RH_ft and rsut: the direction is
    # -sign(gradient), and it must be the same for both losses.
    def _agree(ga, gb):
        return {k: ("decrease" if ga[k] > 0 else "increase") for k in knobs
                if ga[k] != 0 and gb[k] != 0 and np.sign(ga[k]) == np.sign(gb[k])
                and np.isfinite(ga[k]) and np.isfinite(gb[k])}
    agree_g, agree_t = _agree(gq, gs), _agree(gq, gt)
    print("SIGN-AGREEMENT (move lowers RH_ft AND global rsut):", agree_g)
    print("SIGN-AGREEMENT (move lowers RH_ft AND tropical rsut):", agree_t, flush=True)
    res["sign_agreement_global"] = agree_g
    res["sign_agreement_tropical"] = agree_t

    # ---- central finite differences on two knobs, all losses ---------------
    ok = bool(np.all(np.isfinite([x for g in res["grad"].values() for x in g.values()])))
    if not args.no_fd:
        fwd = jax.jit(rollout)
        eps = float(np.log(1.0 + args.fd_rel))
        for kn in FD_KNOBS:
            thp = dict(theta0); thp[kn] = jnp.asarray(eps)
            thm = dict(theta0); thm[kn] = jnp.asarray(-eps)
            fp, fm = fwd(thp), fwd(thm)
            res["fd"][kn] = {}
            for name in names:
                fd = (float(fp[name]) - float(fm[name])) / (2 * eps)
                ad = res["grad"][name][kn]
                rel = abs(fd - ad) / max(abs(fd), 1e-12)
                # fp32 loss noise floor ~1e-6 relative; a difference below
                # that, divided by 2*eps, is UNRESOLVED (not evidence either way)
                noise = 1e-6 * max(abs(res["value"][name]), 1e-12) / (2 * eps)
                resolvable = abs(fd) > 10 * noise
                if not (np.isfinite(fd) and np.isfinite(ad)):
                    verdict = "FAIL"
                elif not resolvable:
                    verdict = "UNRESOLVED" if abs(ad) > 10 * noise else "DEAD"
                else:
                    verdict = "PASS" if rel <= args.fd_tol else "FAIL"
                res["fd"][kn][name] = {"fd": fd, "ad": ad, "rel_err": rel,
                                       "noise": noise, "verdict": verdict}
                if verdict == "FAIL":
                    ok = False
                print(f"FD {kn.split('.')[-1]:14s} {name:11s} fd={fd:+.4e} ad={ad:+.4e} rel={rel:.3e} {verdict}")
    res["peak_gib"] = mem()
    res["wall_s"] = time.time() - t0
    with open(scratch / f"multistep_grad_K{K}.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"peak {res['peak_gib']:.1f} GiB, wall {res['wall_s']:.0f} s, OK={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
