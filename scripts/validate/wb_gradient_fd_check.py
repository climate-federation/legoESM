"""Is the WeatherBench training gradient CORRECT? Autodiff versus finite differences.

Before any conclusion about an objective, the derivative of that objective has
to be shown right: an underperforming gradient result is a bug in the method
until proven otherwise. This compares the autodiff gradient against a central
finite difference of the SAME loss over the SAME scenes, sweeping the step size
across decades.

Three outcomes separate here, and the earlier probe could not tell the last two
apart because it never measured the forward's own repeatability:

* **AD correct** -- the ratio approaches 1 over a plateau of step sizes.
* **AD wrong** -- the ratio settles somewhere else, or on the wrong sign, with
  the finite difference itself stable across neighbouring step sizes.
* **UNRESOLVED** -- the finite difference is smaller than what the forward's own
  jitter can resolve at that step size. Reporting this as disagreement is how a
  rough loss surface gets mistaken for a broken adjoint.

The noise floor is measured, not assumed: the same loss is evaluated twice on
identical inputs, and the difference divided by 2h is the smallest slope a
central difference can see at that h. A run under jit on one device should give
exactly zero; anything else is itself a finding.

Usage::

    python scripts/validate/wb_gradient_fd_check.py \\
        --config config/wb/campaign/spectral_t63_amip.yaml \\
        --n-scenes 4 --n-params 6 --out results/wb_fd/amip_seed.json
"""
from __future__ import annotations

import argparse

# Six decades: x64 gives roughly that many usable ones before round-off in the
# difference of two nearly-equal losses dominates.
_DEFAULT_STEPS = "1e-6,1e-5,1e-4,1e-3,1e-2"


def build_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--mode", default="physics")
    p.add_argument("--training-core", default="spectral", dest="training_core")
    p.add_argument("--checkpoint", default=None,
                   help="parameters to probe; default = the config's own "
                        "starting values")
    p.add_argument("--n-scenes", type=int, default=4, dest="n_scenes")
    p.add_argument("--n-params", type=int, default=6, dest="n_params",
                   help="probe this many parameters, largest |gradient| first")
    p.add_argument("--steps", default=_DEFAULT_STEPS,
                   help="finite-difference step sizes, in UNCONSTRAINED space")
    p.add_argument("--rollout-steps", type=int, default=None,
                   dest="rollout_steps",
                   help="override the number of model steps differentiated "
                        "through. THE decisive control: if autodiff agrees "
                        "with finite differences at one step and stops "
                        "agreeing at twelve, the derivative is right and the "
                        "rollout's non-smoothness is the problem; if it "
                        "disagrees at one step too, the derivative is wrong.")
    p.add_argument("--params", default=None,
                   help="Comma-separated SUBSTRINGS of parameter names to "
                        "probe, instead of the largest-gradient ones. Use it "
                        "to contrast a smooth scheme's knob against a gated "
                        "one's. A substring matching nothing is a hard error.")
    p.add_argument("--adjoint-steps", type=int, default=0,
                   dest="adjoint_steps",
                   help="check the TRAINER's gradient: the full forecast to "
                        "the real target, differentiated through the last n "
                        "steps only. The finite difference then perturbs the "
                        "parameters in those n steps only (the first leg runs "
                        "the unperturbed parameters and hands on its full "
                        "carry). Exclusive with --rollout-steps.")
    p.add_argument("--column-devices", type=int, default=0,
                   dest="column_devices",
                   help="split each scene's columns over this many GPUs, "
                        "exactly as the trainer's --column-devices (one "
                        "process per node). 0 = one device.")
    p.add_argument("--out", default="results/wb_fd/fd_check.json")
    return p.parse_args(argv)


def classify(ad, rows, tol=0.25, plateau_tol=0.15):
    """Verdict for one parameter from its step-size sweep.

    ``rows`` are ``{"h", "fd", "noise_floor", "finite"}``. Two rules, both
    earned in review:

    * A row carries information only if its finite difference clears TEN times
      the floor and is finite. A difference the forward's own noise could have
      produced is not evidence, and an overflowing perturbation is evidence
      about the FORWARD, never about the adjoint.
    * The verdict comes from a PLATEAU -- adjacent step sizes whose finite
      differences agree with each other -- not from whichever row happens to
      sit closest to the autodiff value. Selecting on closeness to 1 is
      circular: it certifies a match whenever any step size produces one, and
      a curved loss will always produce one somewhere.

    With no plateau there is no verdict, and saying so is the honest answer:
    a 12-step moist-physics rollout full of gates can genuinely fail to have
    one, which is itself a finding about the objective.
    """
    # TEN times the floor, not merely above it: a difference within a small
    # multiple of the resolution is dominated by it, and at five decades some
    # row will clear a bare floor by luck.
    usable = [r for r in rows
              if r.get("finite", True)
              and abs(r["fd"]) > 10.0 * r["noise_floor"]]
    if len(usable) < 2:
        return ("UNRESOLVED" if not usable else "NO_PLATEAU"), None
    usable.sort(key=lambda r: r["h"])
    best = None
    for lo, hi in zip(usable, usable[1:]):
        scale = max(abs(lo["fd"]), abs(hi["fd"]))
        if scale == 0.0:
            continue
        spread = abs(lo["fd"] - hi["fd"]) / scale
        if spread <= plateau_tol and (best is None or spread < best[0]):
            best = (spread, 0.5 * (lo["fd"] + hi["fd"]))
    if best is None:
        return "NO_PLATEAU", None
    fd = best[1]
    ratio = ad / fd if fd else float("nan")
    if abs(ratio - 1.0) <= tol:
        return "AD_MATCHES_FD", ratio
    if ratio < 0.0:
        return "SIGN_DISAGREES", ratio
    # A central difference straddling a where-branch returns the AVERAGE of the
    # two one-sided slopes. When one side is flat that average is half the true
    # slope, so a stable plateau near ratio 2 is the signature of a kink in the
    # loss, NOT of a wrong adjoint -- the distinction that decides whether to
    # fix the derivative or change the objective.
    if abs(ratio - 2.0) <= tol:
        return "KINK_SUSPECTED", ratio
    return "MAGNITUDE_DISAGREES", ratio


def noise_floor(h, jitter, baseline, n_terms=1):
    """Smallest slope a central difference can resolve at step ``h``.

    Two independent error sources, and the review caught that only the first
    was counted:

    * measured non-repeatability of the forward (``jitter``);
    * ROUND-OFF in the difference of two nearly-equal losses, which a repeat
      evaluation cannot see because it is deterministic. A loss of magnitude
      ``|L|`` accumulated over ``n_terms`` summands carries roughly
      ``n_terms * eps * |L|`` of representation error in binary64, and that
      error does not cancel between the plus and minus evaluations.
    """
    import numpy as np

    eps = float(np.finfo(np.float64).eps)
    return (abs(jitter) + n_terms * eps * abs(baseline)) / (2.0 * h)


def main(argv=None):
    a = build_args(argv)
    if a.adjoint_steps and a.mode != "physics":
        raise SystemExit("--adjoint-steps checks the classical (physics) arm "
                         "only; its two-leg forecast needs the radiation carry")
    if a.adjoint_steps and a.rollout_steps:
        raise SystemExit("--adjoint-steps and --rollout-steps measure different "
                         "things; pass one")

    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "run"))
    from train_weatherbench_scale import (
        _column_placers, _init_column_mesh, _init_distributed,
        build_scale_config_from_args,
    )
    if a.column_devices:
        _init_distributed(a.column_devices)   # before any JAX backend use

    import json
    import logging

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import yaml

    jax.config.update("jax_enable_x64", True)
    mesh = _init_column_mesh(a.column_devices) if a.column_devices else None
    rank0 = jax.process_index() == 0
    logging.basicConfig(level=logging.INFO if rank0 else logging.WARNING)
    log = logging.getLogger("wb_fd")

    from legoesm.training.losses import combined_loss
    from legoesm.training.scale_build import (
        build_mode_components, load_era5_samples, rollout_hours,
        validate_wb_campaign_yaml,
    )

    cfg = build_scale_config_from_args(
        ["--mode", a.mode, "--training-core", a.training_core,
         "--config", a.config])
    yml = yaml.safe_load(open(a.config))
    validate_wb_campaign_yaml(yml)

    _, grid, sigma, params, make_run_seg, loss_config, dt = \
        build_mode_components(cfg, yml, column_mesh=mesh)
    make_run_seg(params)
    place, rep_in_trace = jax.device_put, (lambda t: t)
    if mesh is not None:
        place, _, rep_in_trace = _column_placers(
            mesh, int(grid.n_lat), int(grid.n_lat) * int(grid.n_lon))
    if a.checkpoint:
        params = eqx.tree_deserialise_leaves(a.checkpoint, params)
    samples = load_era5_samples(cfg, yml, grid, sigma, rank=0, nproc=1,
                                host_resident=True)
    if mesh is not None:
        # Every process builds its own copy; collective forwards need them equal.
        from train_weatherbench_scale import (
            _SAMPLE_AGREE_RTOL, _assert_samples_agree,
        )
        _assert_samples_agree(samples, "fd-check samples", _SAMPLE_AGREE_RTOL)
    roll = (a.rollout_steps if a.rollout_steps
            else int(rollout_hours(cfg, yml) * 3600.0 / dt))
    if a.rollout_steps:
        log.warning("rollout SHORTENED to %d step(s) (%.2f h): the forecast "
                    "no longer reaches the target, so the loss VALUE is not "
                    "comparable to a full run. This is a derivative control, "
                    "not a forecast.", roll, roll * dt / 3600.0)
    sigma_full = jnp.asarray(sigma.sigma_full)

    arr, static = eqx.partition(params, eqx.is_inexact_array)
    leaves, treedef = jax.tree_util.tree_flatten(arr)
    names = [jax.tree_util.keystr(p) for p, _ in jax.tree.leaves_with_path(arr)]

    if a.adjoint_steps and not 0 < a.adjoint_steps < roll:
        raise SystemExit(f"--adjoint-steps must be in 1..{roll - 1}; the full "
                         "window is the default run without the flag")

    def _seg(flat):
        tree = rep_in_trace(jax.tree_util.tree_unflatten(treedef, flat))
        return make_run_seg(eqx.combine(tree, static))

    def one_loss(flat, s):
        ic, target, forcing = s
        kw = {"adjoint_steps": a.adjoint_steps} if a.adjoint_steps else {}
        return combined_loss(_seg(flat).raw(ic, roll, forcing, **kw),
                             target, sigma_full, grid=grid, config=loss_config)

    def suffix_loss(flat, s):
        # Leg 1 with the UNPERTURBED parameters, leg 2 with ``flat``: the
        # function whose derivative the windowed adjoint computes.
        ic, target, forcing = s
        pred = _seg(leaves).raw_two_leg(_seg(flat), ic, roll, forcing,
                                        a.adjoint_steps)
        return combined_loss(pred, target, sigma_full, grid=grid,
                             config=loss_config)

    vg = eqx.filter_jit(jax.value_and_grad(one_loss))
    fwd = eqx.filter_jit(suffix_loss if a.adjoint_steps else one_loss)

    # Scene selection is RECORDED, not silent. A scene whose forward is finite
    # but whose gradient is not is itself an adjoint failure; dropping it and
    # quietly taking the next healthy one would let the instrument report
    # nothing but matches on a model that produces NaN gradients (codex).
    scenes, grads, rejected = [], None, []
    for k, smp in enumerate(samples):
        sd = place(smp)
        v, g = vg(leaves, sd)
        bad_value = not bool(jnp.isfinite(v))
        bad_grad = not all(bool(jnp.all(jnp.isfinite(x)))
                           for x in jax.tree.leaves(g))
        if bad_value or bad_grad:
            rejected.append({"index": k, "loss_nonfinite": bad_value,
                             "gradient_nonfinite": bad_grad})
            continue
        scenes.append(sd)
        gl = jax.tree.leaves(g)
        grads = gl if grads is None else [x + y for x, y in zip(grads, gl)]
        if len(scenes) == a.n_scenes:
            break
    if not scenes:
        raise SystemExit("no scene had a finite loss AND gradient")
    if rejected:
        log.warning("%d scene(s) rejected before the sweep: %s. A non-finite "
                    "GRADIENT on a scene with a finite loss is an adjoint "
                    "failure in its own right, not a bad sample.",
                    len(rejected), rejected)

    def total(flat):
        return sum(float(fwd(flat, s)) for s in scenes)

    # --- forward determinism, measured with the SAME compiled program -------
    # Comparing a value from the gradient program against one from the forward
    # program measures the difference between two executables, not the
    # forward's repeatability (codex).
    if a.adjoint_steps:
        # The two-leg forecast at unperturbed parameters IS the trainer's
        # forecast; a mismatch means the chaining dropped part of the carry.
        v_one = float(eqx.filter_jit(one_loss)(leaves, scenes[0]))
        v_two = float(fwd(leaves, scenes[0]))
        log.info("two-leg vs single forecast loss: %.17g vs %.17g", v_two, v_one)
        # Two executables may round differently (fusion order, amplified over
        # the forecast); a dropped carry moves the loss by far more than 1e-6.
        if abs(v_two - v_one) > 1e-6 * max(abs(v_one), 1e-300):
            raise SystemExit("two-leg forecast differs from the single one; "
                             "the finite difference would test another model")
    base = total(leaves)
    repeat = total(leaves)
    jitter = abs(repeat - base)
    log.info("forward repeatability: |L - L| = %.3e over %d scene(s) "
             "(baseline %.10g)", jitter, len(scenes), base)
    if jitter > 0.0:
        log.warning("the forward is NOT bit-reproducible; every finite "
                    "difference carries a floor of |dL|/2h on top of round-off")

    steps = [float(x) for x in a.steps.split(",")]

    # ONE ELEMENT per leaf, never the whole leaf. Perturbing every element by
    # the same h and comparing against the SUMMED gradient tests a single
    # directional derivative, which two wrong components can satisfy by
    # cancelling (codex). The element with the largest |gradient| is probed,
    # and for a leaf with no gradient at all, element 0.
    #
    # ZERO-gradient leaves are probed DELIBERATELY: a detached path reports
    # exactly 0.0, which is the failure this whole check exists to catch, and
    # the first version excluded them for having nothing to compare.
    mags = [float(jnp.max(jnp.abs(g))) for g in grads]
    # Counted for the report either way -- the named-parameter branch below
    # does not build it, and the summary reads it unconditionally.
    dead = [i for i in range(len(grads)) if mags[i] == 0.0]
    if a.params:
        # Named parameters instead of the largest-gradient ones, so a SMOOTH
        # scheme's knob can be contrasted with a GATED one's. If the agreement
        # survives the full rollout for parameters that enter no threshold
        # while it dies for parameters behind a trigger, the amplifier is the
        # switch and not the trajectory -- worth separating, because a two-day
        # error-doubling time predicts a factor of about 1.2 over six hours
        # and the measured figure is 1.8e6.
        want = [w.strip() for w in a.params.split(",") if w.strip()]
        missing = [w for w in want if not any(w in nm for nm in names)]
        if missing:
            raise SystemExit(
                f"--params matched nothing for {missing!r}; a selector that "
                "selects nothing would silently probe the wrong parameters")
        probe = [i for i in range(len(names))
                 if any(w in names[i] for w in want)][:max(1, a.n_params)]
        log.info("probing %d NAMED parameter(s): %s", len(probe),
                 ", ".join(names[i] for i in probe))
    else:
        live = sorted((i for i in range(len(grads)) if mags[i] > 0.0),
                      key=lambda i: -mags[i])
        n_dead = min(len(dead), max(1, a.n_params // 3)) if dead else 0
        probe = live[:max(1, a.n_params - n_dead)] + dead[:n_dead]

    out = {"config": a.config, "checkpoint": a.checkpoint,
           "n_scenes": len(scenes), "rejected_scenes": rejected,
           "rollout_steps": roll, "adjoint_steps": a.adjoint_steps,
           "column_devices": a.column_devices, "baseline_loss": base,
           "forward_jitter": jitter, "n_zero_gradient_leaves": len(dead),
           "parameters": {}}

    for i in probe:
        g_i = np.asarray(grads[i], dtype=np.float64)
        j = int(np.argmax(np.abs(g_i))) if g_i.size else 0
        ad = float(g_i.reshape(-1)[j]) if g_i.size else 0.0
        base_leaf = np.asarray(leaves[i], dtype=np.float64)
        rows = []
        for h in steps:
            up, dn = list(leaves), list(leaves)
            xu, xd = base_leaf.copy(), base_leaf.copy()
            if xu.shape:
                xu.reshape(-1)[j] += h
                xd.reshape(-1)[j] -= h
            else:
                xu, xd = xu + h, xd - h
            up[i] = jnp.asarray(xu, dtype=leaves[i].dtype)
            dn[i] = jnp.asarray(xd, dtype=leaves[i].dtype)
            lp, lm = total(up), total(dn)
            finite = bool(np.isfinite(lp) and np.isfinite(lm))
            fd = (lp - lm) / (2.0 * h) if finite else float("nan")
            rows.append({"h": h, "fd": fd, "finite": finite,
                         "noise_floor": noise_floor(h, jitter, base,
                                                    n_terms=len(scenes))})
        verdict, ratio = classify(ad, rows)
        out["parameters"][names[i]] = {
            "autodiff": ad, "element": j, "leaf_size": int(g_i.size),
            "verdict": verdict, "best_ratio": ratio, "rows": rows}
        log.info("%s[%d]: autodiff %.6g -> %s (ratio %s)", names[i], j, ad,
                 verdict, "n/a" if ratio is None else f"{ratio:.4f}")
        for r in rows:
            why = ("non-finite" if not r["finite"]
                   else "usable" if abs(r["fd"]) > r["noise_floor"]
                   else "BELOW FLOOR")
            log.info("    h=%.0e  fd=%12.5g  floor=%9.3g  %s",
                     r["h"], r["fd"], r["noise_floor"], why)

    verdicts = [v["verdict"] for v in out["parameters"].values()]
    out["summary"] = {v: verdicts.count(v) for v in set(verdicts)}
    if rank0:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".",
                    exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=2)
    log.info("verdicts: %s -> %s", out["summary"], a.out)
    return out


if __name__ == "__main__":
    main()
