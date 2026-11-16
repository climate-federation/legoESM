#!/usr/bin/env python3
"""End-to-end training test on the forced ORCA2 K0 card: a synthetic twin.

Truth: the normal-mode K0 card's own parameter values.  Start: the same
parameters perturbed by ``--perturb`` (physical value times the factor,
reflected inside the bounds).  Each step differentiates, through the loopable
composed step (``legoesm.training.orca2_k0_rollout``) with deep restoring on
(below ``--restore-depth-m`` at ``--restore-years``), the misfit between the
end-of-window state and the truth run's:

    L = sum over T, S (top ``--depth-m``) of volume-mean (X - X*)^2 / var(X*)
        + area-mean (SSH - SSH*)^2 / var(SSH*)

and takes one optimizer step (repo standard ``ml.training.create_optimizer``).
The first gradient must reach every parameter (``assert_no_inert``).  Steps 1
and 2 run eagerly at the truth values (the cold start), then the loop.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for _p in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean",
           REPO_ROOT / "packages/ml"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

TKE = "ocean.vm.tke"
GM = "ocean.lat.gm_redi"
TREG = "ocean.lat.treguier"
TKE_FIELDS = ("surface_flux_coeff", "Prandtl_tke0", "lc_coeff", "etau_frac", "alpha_tke",
              "bg_diff_scale", "c_eps", "c_k")
GM_FIELDS = ("kappa_redi_diffusive_velocity",)
TREG_FIELDS = ("aei0",)


def card_values(config) -> dict[str, float]:
    """The card's own value of every trained parameter (the twin's truth)."""
    tke = config.physics.vertical_mixing.tke
    out = {f"{TKE}.{f}": float(getattr(tke, f)) for f in TKE_FIELDS}
    out.update({f"{GM}.{f}": float(getattr(config.gm_redi, f)) for f in GM_FIELDS})
    out.update({f"{TREG}.{f}": float(getattr(config.gm_redi.treguier, f)) for f in TREG_FIELDS})
    return out


def perturbed(values: dict, bounds: dict, factor: float) -> dict:
    """value * factor, reflected back inside (lo, hi) if it would leave them."""
    out = {}
    for name, v in values.items():
        lo, hi = bounds[name]
        p = v * factor
        if not lo < p < hi:
            p = v / factor
        if not lo < p < hi:
            raise ValueError(f"{name}={v} cannot be perturbed by {factor} inside {lo, hi}")
        out[name] = p
    return out


def with_params(model, overrides: dict):
    """A shallow copy of ``model`` whose config carries the TKE / GM / Treguier overrides."""
    from legoesm.core.param_overrides import apply_param_overrides

    unknown = set(overrides) - {TKE, GM, TREG}
    if unknown:
        raise ValueError(f"no injection path for {sorted(unknown)}")
    cfg = model.config
    if TKE in overrides:
        vm = cfg.physics.vertical_mixing
        vm = vm._replace(tke=apply_param_overrides(vm.tke, overrides[TKE]))
        cfg = cfg._replace(physics=cfg.physics._replace(vertical_mixing=vm))
    if GM in overrides or TREG in overrides:
        gm = cfg.gm_redi
        if GM in overrides:
            gm = apply_param_overrides(gm, overrides[GM])
        if TREG in overrides:
            gm = gm._replace(treguier=apply_param_overrides(gm.treguier, overrides[TREG]))
        cfg = cfg._replace(gm_redi=gm)
    m = copy.copy(model)
    m.config = cfg
    return m


def misfit(z_coord, area, depth_m):
    """``(state, truth) -> L`` with per-field variance normalisation from the truth."""
    wet = jnp.asarray(z_coord.is_active, dtype=jnp.float64)
    if z_coord.t_depth_ref is None:
        raise ValueError("misfit needs the NEMO level-centre depths (t_depth_ref)")
    level_depth = jnp.asarray(z_coord.t_depth_ref)
    w3 = (jnp.asarray(area)[..., None] * jnp.asarray(z_coord.h_partial) * wet
          * (level_depth <= depth_m))
    w2 = jnp.asarray(area) * wet[..., 0]

    def wmean(w, x):
        return jnp.sum(w * x) / jnp.sum(w)

    def wvar(w, x):
        return wmean(w, (x - wmean(w, x)) ** 2)

    def loss(state, truth):
        total = 0.0
        for name in ("T", "S"):
            x, xt = getattr(state, name).data, getattr(truth, name).data
            total = total + wmean(w3, (x - xt) ** 2) / wvar(w3, xt)
        e, et = state.eta.data, truth.eta.data
        return total + wmean(w2, (e - et) ** 2) / wvar(w2, et)

    return loss


def main(argv=None) -> int:
    from legoesm.ml.training import TrainingConfig, create_optimizer
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.sponge import deep_restoring_gamma, with_deep_restoring
    from legoesm.training.inert_params import assert_no_inert
    from legoesm.training.orca2_k0_rollout import (
        carry_from_result, make_k0_rollout, rollout_operands, split_controls)
    from legoesm.training.param_collector import build_orca2_rung10_trainable_params
    from legoesm.training.trainable_params import TrainablePhysicsParams
    from scripts.validate.ocean_fidelity.orca2_l4.orca2_k0_rollout_equivalence import build, eager

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--opt-steps", type=int, default=5)
    p.add_argument("--perturb", type=float, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--depth-m", type=float, required=True)
    p.add_argument("--restore-depth-m", type=float, required=True)
    p.add_argument("--restore-years", type=float, required=True)
    p.add_argument("--checkpoints", type=int, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if not jax.config.jax_enable_x64:
        raise SystemExit("needs JAX_ENABLE_X64=1")
    stamp = worktree_stamp(repo=REPO_ROOT)
    log = {"args": vars(args) | {"output": str(args.output)}, "provenance": stamp, "steps": []}

    def save():
        args.output.write_text(json.dumps(log, indent=1, default=str) + "\n")

    n_steps = 2 + 8 * args.days                         # kt 1, 2 eager; the rest looped
    card, model, controls = build("normal", n_steps)
    z = card.ocean.recipe.z_coord
    deep = deep_restoring_gamma(z.t_depth_ref, np.asarray(z.is_active), args.restore_depth_m,
                                args.restore_years * 365 * 86400.0)   # the card's no-leap year
    controls = [dict(c, sponge=with_deep_restoring(c["sponge"], deep)) for c in controls]
    ref = eager(card, model, controls[:2], card.ocean.recipe.initial_state)
    start = carry_from_result(ref[-1])
    split = split_controls(controls[2:], kt0=3)
    ops = rollout_operands(split)
    del controls, ref

    truth_vals = card_values(model.config)
    names = sorted(truth_vals)
    params0 = build_orca2_rung10_trainable_params(
        active_scheme_keys={TKE, GM, TREG}, tier=0, include=tuple(names),
        init_values=truth_vals, dtype=jnp.float64)
    bounds = {c.name: (c.min_val, c.max_val) for c in params0.constraints}
    start_vals = perturbed(truth_vals, bounds, args.perturb)
    params = build_orca2_rung10_trainable_params(
        active_scheme_keys={TKE, GM, TREG}, tier=0, include=tuple(names),
        init_values=start_vals, dtype=jnp.float64)
    constraints = params.constraints
    log.update(truth=truth_vals, start=start_vals)

    truth_state = jax.jit(make_k0_rollout(card, model, split, checkpoints=args.checkpoints))(
        start, ops).state
    loss_of_state = misfit(z, model.grid.area, args.depth_m)

    def loss(raw, carry, operands, truth):
        tp = TrainablePhysicsParams(raw_values=raw, constraints=constraints)
        m = with_params(model, tp.to_overrides())
        out = make_k0_rollout(card, m, split, schedule="binomial",
                              checkpoints=args.checkpoints)(carry, operands)
        return loss_of_state(out.state, truth)

    vg = jax.jit(jax.value_and_grad(loss))
    opt = create_optimizer(TrainingConfig(lr=args.lr, warmup_steps=0,
                                          total_steps=args.opt_steps, optimizer="adam"))
    raw = dict(params.raw_values)
    opt_state = opt.init(raw)
    save()
    for k in range(args.opt_steps):
        t0 = time.perf_counter()
        value, g = vg(raw, start, ops, truth_state)
        g = jax.device_get(g)
        if k == 0:
            assert_no_inert(g)
        cur = TrainablePhysicsParams(raw_values=raw, constraints=constraints).as_dict()
        log["steps"].append({
            "step": k, "loss": float(value), "seconds": time.perf_counter() - t0,
            "params": {n: float(v) for n, v in cur.items()},
            "rel_error": {n: float(cur[n]) / truth_vals[n] - 1.0 for n in names},
            "grad": {n: float(v) for n, v in g.items()}})
        save()
        print(f"step {k} loss {float(value):.6e} "
              f"max|rel err| {max(abs(v) for v in log['steps'][-1]['rel_error'].values()):.3f}",
              flush=True)
        updates, opt_state = opt.update(g, opt_state, raw)
        raw = jax.tree.map(lambda a, u: a + u, raw, updates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
