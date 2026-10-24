#!/usr/bin/env python3
"""ORCA2X round 2: multi-step rung-10 gradients through ``integrate_scan``.

Rung 10 as the NEMO comparison runs it, from its own initial state (round 329),
N steps through the model's ``integrate_scan`` with per-step inputs
(``nemo_kt0 = 1``).  Trainable: rn_Ud only, collected by the rung-10 trainer
entry (DECISION 149) with ORCA2's start value and spliced into the config
inside the loss, so its leaf is traced.  Losses: area/volume-weighted T^2 and
SSH^2 of the final state (round 315's).

The step body is the model's own jit body (``_step_jitted.__wrapped__``) on a
copy of the model whose config carries the traced rn_Ud, set as that copy's
``_step_jitted``; ``--remat`` wraps it in ``jax.checkpoint`` (every step,
step 1 included).  One N per process (per-process memory-map limit).

Program A = ``jit(vjp)`` of (L_T, L_SSH) w.r.t. every float leaf of the
initial state and the raw rn_Ud leaf, run once per loss (output cotangent is
a runtime argument): finiteness, norms, timing and peak RSS.  Program B
(``--checks``) = ``jit(jvp)``: forward mode along T_dir0 and along the raw
leaf, and central differences on the raw leaf.  Production control:
``model.step`` N times, its final losses vs program A's primal, bit for bit.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean",
                REPO_ROOT / "packages/ml"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round315_grad_probe as probe,
    nemo_testcase_l4_orca2_round315_nan_locate as locate,
    nemo_testcase_l4_orca2_round316_fused_grad as fused,
)

GM = "ocean.lat.gm_redi"
RN_UD = f"{GM}.kappa_redi_diffusive_velocity"
ORCA2_RN_UD = 0.018  # ORCA2_OMIP_L4/EXP00/namelist_cfg:306
FD_EPS = (1e-2, 1e-4, 1e-6)  # on the raw (unconstrained) rn_Ud leaf
LOSSES = ("T", "ssh")


def split_static(tree):
    """(array leaves, rebuild): Python ints/bools stay static (host step checks)."""

    import jax

    leaves, treedef = jax.tree_util.tree_flatten(tree)
    static = [isinstance(x, (bool, int, np.integer, str)) for x in leaves]
    dyn = [x for x, s in zip(leaves, static) if not s]

    def rebuild(dyn_leaves):
        it = iter(dyn_leaves)
        return treedef.unflatten([x if s else next(it) for x, s in zip(leaves, static)])

    return dyn, rebuild


def setup(n: int):
    import jax
    import jax.numpy as jnp
    from legoesm.core.param_overrides import apply_param_overrides
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.training.param_collector import build_orca2_rung10_trainable_params
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as gate,
    )

    probe.require(bool(jax.config.jax_enable_x64) and jax.default_backend() == "cpu",
                  "needs x64 on CPU")
    card, _, _ = probe.build("rung10")  # also sets the fp64/libm policy
    card = gate.comparison_card(card)
    r, root = card.recipe, probe.ROOTS["rung10"]
    inputs = []
    for kt in range(1, n + 1):
        fw, sf = gate._surface_forcings(card, root, gate.assemble_surface_fields(root, kt), kt)
        inputs.append({"freshwater": fw, "surface_forcing": sf,
                       "sponge": gate.damping_step_kwargs(card, root, kt)["sponge"]})
        probe.require(inputs[-1]["sponge"].nemo_kt == kt, f"damping target kt {kt}")
    model = LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config, iwm_forcing=r.iwm_forcing)
    base = model.config
    probe.require(float(base.gm_redi.kappa_redi_diffusive_velocity) == ORCA2_RN_UD,
                  "card rn_Ud is not ORCA2's")
    params = build_orca2_rung10_trainable_params(
        active_scheme_keys={GM}, tier=0, include=(RN_UD,),
        init_values={RN_UD: ORCA2_RN_UD}, dtype=jnp.float64)
    probe.require([c.name for c in params.constraints] == [RN_UD],
                  "the rung-10 entry returned more than rn_Ud")
    losses, tmask = probe.make_losses(model, r.z_coord)
    return dict(card=card, model=model, base=base, params=params, inputs=inputs,
                losses=losses, tmask=tmask, state0=r.initial_state, dt=card.dt_s,
                apply=apply_param_overrides)


def make_final(ctx, n: int, remat: bool):
    """(state float leaves, state int leaves, input leaves, raw) -> final state."""

    import jax
    from legoesm.training.trainable_params import TrainablePhysicsParams

    model, base, params, dt = ctx["model"], ctx["base"], ctx["params"], ctx["dt"]
    raw_step = type(model).__dict__["_step_jitted"].__wrapped__
    s_leaves, s_def = jax.tree_util.tree_flatten(ctx["state0"])
    is_f = [np.issubdtype(np.asarray(x).dtype, np.inexact) for x in s_leaves]
    in_dyn, in_rebuild = split_static(ctx["inputs"][:n])

    def final(fl, il, in_leaves, raw):
        fi, ii = iter(fl), iter(il)
        state = s_def.unflatten([next(fi) if f else next(ii) for f in is_f])
        tp = TrainablePhysicsParams(raw_values={RN_UD: raw}, constraints=params.constraints)
        m = copy.copy(model)
        m.config = base._replace(gm_redi=ctx["apply"](base.gm_redi, tp.to_overrides()[GM]))

        def body(st, fw, sf, sp):
            return raw_step(m, st, dt, fw, sf, sp)

        body = jax.checkpoint(body) if remat else body

        def step_jitted(st, dt_, fw=None, sf=None, sp=None, **kw):
            probe.require(dt_ == dt and all(v is None for v in kw.values()),
                          f"unexpected step arguments {sorted(k for k, v in kw.items() if v is not None)}")
            return body(st, fw, sf, sp)

        m._step_jitted = step_jitted
        out, _ = m.integrate_scan(state, n, dt, step_inputs=in_rebuild(in_leaves), nemo_kt0=1)
        return out

    fl = [x for x, f in zip(s_leaves, is_f) if f]
    il = [x for x, f in zip(s_leaves, is_f) if not f]
    return final, fl, il, in_dyn


def write(out, path):
    path.write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")


def measure(n: int, remat: bool, checks: bool, output: Path, smoke: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.training.trainable_params import TrainablePhysicsParams

    t_start = time.perf_counter()
    out = {"n": n, "remat": remat, "checks": checks, "rows": {}, "log": []}
    try:
        out["worktree"] = worktree_stamp(allow_dirty=True)
    except Exception as e:  # informational
        out["worktree"] = f"stamp failed: {e}"

    def log(phase, **kw):
        out["log"].append({"phase": phase, "t_s": round(time.perf_counter() - t_start, 1),
                           "rss_gib": fused.rss_bytes() / 2 ** 30, **kw})
        print(out["log"][-1], flush=True)
        write(out, output)

    ctx = setup(n)
    final, fl, il, in_dyn = make_final(ctx, n, remat)
    losses, tmask = ctx["losses"], ctx["tmask"]
    raw0 = ctx["params"].raw_values[RN_UD]
    dud_draw = float(jax.grad(lambda r: TrainablePhysicsParams(
        raw_values={RN_UD: r}, constraints=ctx["params"].constraints).as_dict()[RN_UD])(raw0))
    out["rn_ud"] = {"value": ORCA2_RN_UD, "raw": float(raw0), "d_rn_ud_d_raw": dud_draw}
    float_names = [jax.tree_util.keystr(p) for p, x in
                   jax.tree_util.tree_leaves_with_path(ctx["state0"])
                   if np.issubdtype(np.asarray(x).dtype, np.inexact)]
    i_T = float_names.index(".T[<flat index 0>]")
    i_eta = float_names.index(".eta[<flat index 0>]")

    def both(fl_, raw):
        st = final(fl_, il, in_dyn, raw)
        return jnp.stack([losses["T"](st), losses["ssh"](st)])

    if smoke:
        for name, f in (("vjp", lambda: jax.vjp(both, fl, raw0)[1](jnp.ones(2))),
                        ("jvp", lambda: jax.jvp(both, (fl, raw0), (fl, raw0)))):
            jax.eval_shape(f)
            log(f"smoke {name} traced")
        return out

    # production control: model.step N times (its own executable)
    st = ctx["state0"]
    for kt in range(1, n + 1):
        st = ctx["model"].step(st, ctx["dt"], nemo_kt=kt, **ctx["inputs"][kt - 1])
    prod = {k: float(losses[k](st)) for k in LOSSES}
    del st
    jax.clear_caches()
    log("production", prod_loss=prod)

    # program A: reverse mode, one run per loss
    def rev(fl_, raw, ct):
        val, pb = jax.vjp(both, fl_, raw)
        dfl, draw = pb(ct)
        return val, dfl, draw

    tic = time.perf_counter()
    compiled = jax.jit(rev).lower(fl, raw0, jnp.zeros(2)).compile()
    compile_s = time.perf_counter() - tic
    mem = compiled.memory_analysis()
    out["program_A"] = {"compile_s": round(compile_s, 1),
                        "xla_temp_gib": mem.temp_size_in_bytes / 2 ** 30,
                        "xla_argument_gib": mem.argument_size_in_bytes / 2 ** 30}
    log("program A compiled")
    wet = np.asarray(tmask) > 0.5
    grads = {}
    for j, name in enumerate(LOSSES):
        ct = jnp.zeros(2).at[j].set(1.0)
        base_rss = fused.rss_bytes()
        with fused.RssSampler() as sampler:
            tic = time.perf_counter()
            val, dfl, draw = jax.block_until_ready(compiled(fl, raw0, ct))
            run_s = time.perf_counter() - tic
        dfl = [np.asarray(x) for x in dfl]
        nonfinite = {float_names[i]: int(np.size(x) - np.count_nonzero(np.isfinite(x)))
                     for i, x in enumerate(dfl) if not np.all(np.isfinite(x))}
        gT = dfl[i_T]
        g_raw = float(draw)
        out["rows"][name] = {
            "loss": float(val[j]), "loss_hex": float(val[j]).hex(),
            "loss_equals_production_bits": float(val[j]) == prod[name],
            "loss_rel_vs_production": probe.rel_err(float(val[j]), prod[name]),
            "nonfinite_cotangent_leaves": nonfinite,
            "nonfinite_cotangent_total": int(sum(nonfinite.values())),
            "raw_finite": bool(np.isfinite(g_raw)),
            "dT0_l2": float(np.linalg.norm(np.nan_to_num(gT))),
            "deta0_l2": float(np.linalg.norm(np.nan_to_num(dfl[i_eta]))),
            "state_cotangent_l2": float(np.sqrt(sum(np.sum(np.nan_to_num(x) ** 2) for x in dfl))),
            "dT0_nonzero_dry": int(np.count_nonzero(gT[~wet])),
            "dL_draw": g_raw, "dL_dln_rn_ud": g_raw / dud_draw * ORCA2_RN_UD,
            "run_s": round(run_s, 2), "run_peak_rss_gib": sampler.peak / 2 ** 30,
            "run_rss_growth_gib": (sampler.peak - base_rss) / 2 ** 30}
        grads[name] = (gT, g_raw)
        np.save(output.with_name(f"{output.stem}_dT0_{name}.npy"), gT)
        log(f"program A run {name}")
    del compiled
    jax.clear_caches()
    if not checks:
        out["complete"] = True
        log("done")
        return out

    # program B: forward mode (+ primal for the central differences)
    d = locate.probe_direction(ctx["state0"], tmask)
    zero_fl = [jnp.zeros_like(x) for x in fl]
    t_dir = list(zero_fl)
    t_dir[i_T] = d

    def fwd(fl_, raw, tfl, traw):
        return jax.jvp(both, (fl_, raw), (tfl, traw))

    tic = time.perf_counter()
    cfwd = jax.jit(fwd).lower(fl, raw0, zero_fl, jnp.float64(0.0)).compile()
    out["program_B"] = {"compile_s": round(time.perf_counter() - tic, 1)}
    log("program B compiled")
    checks_out = {}
    tic = time.perf_counter()
    primal, tan_T = cfwd(fl, raw0, t_dir, jnp.float64(0.0))
    out["program_B"]["run_s"] = round(time.perf_counter() - tic, 2)
    _, tan_raw = cfwd(fl, raw0, zero_fl, jnp.float64(1.0))
    for j, name in enumerate(LOSSES):
        gT, g_raw = grads[name]
        rev_T = float(np.vdot(gT, np.asarray(d)))
        checks_out[name] = {
            "primal_equals_program_A_bits": float(primal[j]) == out["rows"][name]["loss"],
            "T_dir0": {"reverse": rev_T, "forward": float(tan_T[j]),
                       "rel": probe.rel_err(rev_T, float(tan_T[j]))},
            "raw": {"reverse": g_raw, "forward": float(tan_raw[j]),
                    "rel": probe.rel_err(g_raw, float(tan_raw[j])), "fd": {}, "fd_rel": {}}}
    log("program B jvp")
    for eps in FD_EPS:
        up, _ = cfwd(fl, raw0 + eps, zero_fl, jnp.float64(0.0))
        dn, _ = cfwd(fl, raw0 - eps, zero_fl, jnp.float64(0.0))
        for j, name in enumerate(LOSSES):
            fd = (float(up[j]) - float(dn[j])) / (2 * eps)
            row = checks_out[name]["raw"]
            row["fd"][f"{eps:g}"] = fd
            row["fd_rel"][f"{eps:g}"] = probe.rel_err(row["reverse"], fd)
        log(f"program B fd {eps:g}")
    out["checks_rows"] = checks_out
    out["complete"] = True
    log("done")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True, choices=range(1, 11),
                        metavar="N", help="steps (1..10: the record has 10)")
    parser.add_argument("--remat", action="store_true", help="jax.checkpoint every step")
    parser.add_argument("--checks", action="store_true",
                        help="also forward mode and central differences (program B)")
    parser.add_argument("--smoke", action="store_true",
                        help="trace both programs abstractly (eval_shape) and stop")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    out = measure(args.n, args.remat, args.checks, args.output, smoke=args.smoke)
    write(out, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
