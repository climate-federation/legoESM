#!/usr/bin/env python3
"""Round 319: rung-10 trainable scalars, traced through the production step.

MEASUREMENT ONLY. Each scalar a closure trainer would expose on rung 10 is a
traced multiplicative scale s (= 1) on its config field (the copy the step
reads). ``--scan``: one jitted step body, run at s = 1 (bit parity with the
production ``model.step``) and with each scalar alone at 1 + ``--rel``; per
kt, the number of state values that moved (each scalar at 1 + rel and
1 - rel). ``--linearity``: one step, responses and forward-mode tangent.
``--floor-cells``: NEMO's diffusivity statement on the floor cells.
``--trace-cd0``: trace
``jit(grad)`` of a one-step loss w.r.t. the cd0 scale (no compile).
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
for package in (REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

DRAG = {"cd0": "bottom_drag_cd0", "cd_max": "bottom_drag_cdmax", "z0": "bottom_drag_z0",
        "ke0": "bottom_drag_ke0", "uc0": "bottom_drag_uc0"}
TKE = {"kappaH_min": "kappaH_min", "tke_background": "tke_background"}
SCALARS = (*DRAG, "kappa_redi", "visc_scale", *TKE)


def scaled(config, z_coord, s: dict):
    """(config, z_coord) with each scalar in ``s`` multiplied by its scale.

    Absent keys are untouched; x * 1.0 == x in IEEE, so s = 1 is the nominal model.
    """
    unknown = set(s) - set(SCALARS)
    if unknown:
        raise ValueError(f"unknown scalars {sorted(unknown)}")
    bd = config.bottom_drag._replace(**{f: getattr(config.bottom_drag, f) * s[k]
                                        for k, f in DRAG.items() if k in s})
    redi = config.gm_redi  # the lat-lon step reads the top-level copy
    if "kappa_redi" in s:
        if redi is None:
            raise ValueError("card runs no isoneutral mixing (gm_redi is None)")
        redi = redi._replace(kappa_Redi=redi.kappa_Redi * s["kappa_redi"])
    vm = config.physics.vertical_mixing
    if any(k in s for k in TKE):
        if vm.scheme != "tke":
            raise ValueError(f"TKE scalars need scheme 'tke', got {vm.scheme!r}")
        vm = vm._replace(tke=vm.tke._replace(**{f: getattr(vm.tke, f) * s[k]
                                                for k, f in TKE.items() if k in s}))
    config = config._replace(bottom_drag=bd, gm_redi=redi,
                             physics=config.physics._replace(vertical_mixing=vm))
    if "visc_scale" in s:
        z_coord = z_coord._replace(nemo_ldf_ahmt=z_coord.nemo_ldf_ahmt * s["visc_scale"],
                                   nemo_ldf_ahmf=z_coord.nemo_ldf_ahmf * s["visc_scale"])
    return config, z_coord


def step_body(model):
    """The production step body with traced scales (not jitted)."""
    raw = type(model).__dict__["_step_jitted"].__wrapped__
    base_cfg, base_z = model.config, model.z_coord

    def step(state, s, dt, freshwater, surface):
        m = copy.copy(model)
        m.config, m.z_coord = scaled(base_cfg, base_z, s)
        return raw(m, m._seed_tke_preclosure_carry(state), dt, freshwater, surface)

    return step


def moved_counts(a, b) -> dict:
    """Per state leaf: values that differ (NaN == NaN; -0.0 vs 0.0 differ)."""
    import jax

    out = {}
    for (path, x), y in zip(jax.tree_util.tree_leaves_with_path(a),
                            jax.tree_util.tree_leaves(b), strict=True):
        x, y = np.asarray(x), np.asarray(y)
        if np.issubdtype(x.dtype, np.floating):
            same = ((x == y) & (np.signbit(x) == np.signbit(y))) | (np.isnan(x) & np.isnan(y))
        else:
            same = x == y
        n = int(np.size(same) - np.count_nonzero(same))
        if n:
            out[jax.tree_util.keystr(path)] = n
    return out


def _model(rung):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round315_grad_probe as probe,
    )
    card, state, forcing = probe.build(rung)
    r = card.recipe
    return card, LatLonCGridOceanModel(r.grid, r.z_coord, r.model_config,
                                       iwm_forcing=r.iwm_forcing), state, forcing


def trace_cd0(rung: str = "rung10"):
    """jaxpr of jit(grad) of the one-step volume T^2 sum w.r.t. the cd0 scale."""
    import jax
    import jax.numpy as jnp

    card, model, state, forcing = _model(rung)
    body = step_body(model)

    def loss(s_cd0):
        st = body(state, {"cd0": s_cd0}, card.dt_s, *forcing[0])
        return jnp.sum(st.T.data ** 2)

    return jax.make_jaxpr(jax.jit(jax.grad(loss)))(jnp.float64(1.0))


def scan(rung: str, rel: float, steps: int, log=print) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    card, model, state0, forcing = _model(rung)
    steps = min(steps, len(forcing))
    step = jax.jit(step_body(model))
    ones = {k: jnp.float64(1.0) for k in SCALARS}

    def run(s):
        st, out = state0, []
        for kt in range(steps):
            st = step(st, s, card.dt_s, *forcing[kt])
            out.append(jax.device_get(st))
        return out

    t0 = time.time()
    prod = jax.device_get(model.step(state0, card.dt_s, freshwater=forcing[0][0],
                                     surface_forcing=forcing[0][1]))
    base = run(ones)
    log(f"[{time.time() - t0:7.1f}s] base")
    result = {"rung": rung, "rel": rel, "steps": steps, "dt_s": card.dt_s,
              "worktree": worktree_stamp(allow_dirty=True), "jax": jax.__version__,
              "nominal": _nominal(model), "parity_kt1_vs_production": moved_counts(prod, base[0]),
              "scalars": {}}
    for k in SCALARS:
        row = {}
        for sign, r in (("up", rel), ("down", -rel)):
            bumped = run({**ones, k: jnp.float64(1.0 + r)})
            rows = [moved_counts(b, p) for b, p in zip(base, bumped, strict=True)]
            row[sign] = {"moved_by_kt": rows, "moved_total_by_kt": [sum(x.values()) for x in rows]}
        result["scalars"][k] = row
        log(f"[{time.time() - t0:7.1f}s] {k}: up {row['up']['moved_total_by_kt']}"
            f" down {row['down']['moved_total_by_kt']}")
    return result


def _max_abs(a, b) -> dict:
    import jax

    return {jax.tree_util.keystr(p): float(np.max(np.abs(np.asarray(x, np.float64)
                                                         - np.asarray(y, np.float64))))
            for (p, x), y in zip(jax.tree_util.tree_leaves_with_path(a),
                                 jax.tree_util.tree_leaves(b), strict=True)
            if np.issubdtype(np.asarray(x).dtype, np.floating) and np.size(x)}


def linearity(rung: str, keys, rel: float, log=print) -> dict:
    """One step: max |state(1 + r) - state(1)| at r = rel and 10 rel, and the
    forward-mode tangent times rel, per leaf, for each key."""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    card, model, state0, forcing = _model(rung)
    body = step_body(model)
    ones = {k: jnp.float64(1.0) for k in SCALARS}
    # state, dt and forcing are ARGUMENTS (as in the production step): closed
    # over, they become constants and XLA folds them into a different program.
    args = (card.dt_s, *forcing[0])
    jstep = jax.jit(body)
    jtan = jax.jit(lambda st, t, *a: jax.jvp(lambda s: body(st, s, *a), (ones,), (t,))[1])
    base = jax.device_get(jstep(state0, ones, *args))
    prod = jax.device_get(model.step(state0, card.dt_s, freshwater=forcing[0][0],
                                     surface_forcing=forcing[0][1]))
    out = {"rung": rung, "rel": rel, "worktree": worktree_stamp(allow_dirty=True),
           "parity_vs_production": moved_counts(prod, base), "keys": {}}
    log("parity vs production:", out["parity_vs_production"])
    for k in keys:
        row = {}
        for r in (rel, 10 * rel):
            row[f"diff@{r:g}"] = _max_abs(
                jax.device_get(jstep(state0, {**ones, k: jnp.float64(1 + r)}, *args)), base)
        t = jax.device_get(jtan(state0, {j: jnp.float64(1.0 if j == k else 0.0)
                                         for j in SCALARS}, *args))
        row[f"jvp*{rel:g}"] = {p: v * rel for p, v in _max_abs(t, jax.tree_util.tree_map(
            np.zeros_like, t)).items()}
        out["keys"][k] = row
        log(k, {name: {p: v for p, v in d.items() if p.startswith(".T[")}
                for name, d in row.items()})
    return out


def floor_cells(rung: str, log=print) -> dict:
    """The TKE diffusivity cells within 0.1 % above the background floor.

    Records the closure's inputs from inside the jitted production step
    (ordered callbacks; control: the recorded step's state equals the
    unrecorded one) and evaluates there, in fp64 numpy:
    NEMO ``zdftke.f90:707-717``: avt = MAX(pdlr * MAX(zav, avtb), avtb),
    zav = (rn_ediff * zmxlm) * SQRT(en); the closure eager, avtb vs
    zav / (1 / pdlr); and the closure jitted alone, where XLA may rewrite
    the reciprocal division.
    """
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.provenance import worktree_stamp
    from legoesm.ocean.physics.vertical_mixing import tke as tke_mod

    card, model, state0, forcing = _model(rung)
    tcfg = model.config.physics.vertical_mixing.tke
    fl = np.float64(tcfg.kappaH_min)
    args = (card.dt_s, *forcing[0])
    body = step_body(model)
    ones = {k: jnp.float64(1.0) for k in SCALARS}
    base = jax.device_get(jax.jit(body)(state0, ones, *args))

    calls = []
    orig_k, orig_pr = tke_mod.compute_K_from_tke, tke_mod._prandtl_number

    def rec(tag):
        return lambda **a: calls.append((tag, {k: np.asarray(v) for k, v in a.items()}))

    def pr_rec(*a, **kw):
        pr = orig_pr(*a, **kw)
        jax.debug.callback(rec("Pr"), Pr=pr, ordered=True)
        return pr

    def k_rec(e, l_k, cfg, **kw):
        k_m, k_h = orig_k(e, l_k, cfg, **kw)
        jax.debug.callback(rec("K"), e=e, l_k=l_k, K_H=k_h, ordered=True)
        return k_m, k_h

    tke_mod.compute_K_from_tke, tke_mod._prandtl_number = k_rec, pr_rec
    try:
        recorded = jax.device_get(jax.jit(step_body(model))(state0, ones, *args))
    finally:
        tke_mod.compute_K_from_tke, tke_mod._prandtl_number = orig_k, orig_pr
    out = {"rung": rung, "worktree": worktree_stamp(allow_dirty=True), "floor": float(fl),
           "recorded_vs_unrecorded_step": moved_counts(base, recorded),
           "calls": [t for t, _ in calls]}
    avt = np.asarray(base.tke_avt.data)
    cells = (avt >= fl) & (avt < fl * 1.001)
    # the closure call whose K_H is the stored diffusivity on the floor cells,
    # with its own Prandtl number (recorded just before it)
    agree = {i: (a["K_H"].shape == avt.shape and int(np.sum(a["K_H"][cells] == avt[cells])))
             for i, (t, a) in enumerate(calls) if t == "K"}
    out["closure_call_shapes"] = [list(a["K_H"].shape) for t, a in calls if t == "K"]
    out["stored_avt_shape"] = list(avt.shape)
    out["closure_call_equal_on_floor_cells"] = agree
    hits = [i for i, n in agree.items() if n == int(cells.sum())]
    if not hits or calls[hits[-1] - 1][0] != "Pr":
        raise SystemExit(f"no recorded closure call reproduces tke_avt on the floor cells: {out}")
    a, pr = calls[hits[-1]][1], calls[hits[-1] - 1][1]["Pr"]
    out["closure_call_used"] = hits[-1]
    out["closure_call_equal_wet"] = int(np.sum((a["K_H"] == avt) & (avt > 0)))
    out["wet"] = int(np.sum(avt > 0))
    e, lk, pr_c = a["e"][cells], a["l_k"][cells], pr[cells]
    zav = (np.float64(tcfg.c_k) * lk) * np.sqrt(e)
    # every pdlr whose reciprocal rounds to the recorded Pr (Pr = 1 / pdlr)
    pdlr_set = sorted({float(1.0 / p) for p in np.unique(pr_c)})
    cand = [np.nextafter(x, d) for x in pdlr_set for d in (0.0, 1.0)] + pdlr_set
    pdlr_ok = sorted({float(c) for c in cand if np.float64(1.0) / c in set(np.unique(pr_c))})

    def ulps(x):
        return np.asarray(x, np.float64).view(np.int64) - fl.view(np.int64)

    def hist(x):
        u, c = np.unique(ulps(x), return_counts=True)
        return {int(k): int(v) for k, v in zip(u[:8], c[:8], strict=True)}

    eager = np.asarray(jnp.maximum(fl, jnp.asarray(zav) / jnp.asarray(pr_c)))
    jit_alone = np.asarray(jax.jit(lambda z, p: jnp.maximum(fl, z / (1.0 / p)))(
        jnp.asarray(zav), jnp.asarray(1.0 / pr_c)))
    out.update({
        "cells": int(cells.sum()),
        "stored_avt_ulps_above_floor": hist(avt[cells]),
        "e_unique": [float(x) for x in np.unique(e)[:5]],
        "e_n_unique": int(np.unique(e).size),
        "l_k_unique": [float(x) for x in np.unique(lk)[:5]],
        "Pr_unique": [float(x) for x in np.unique(pr_c)],
        "pdlr_consistent_with_Pr": pdlr_ok,
        "closure_eager_ulps": hist(eager),
        "closure_jit_alone_ulps": hist(jit_alone),
        "closure_eager_equals_stored": int(np.sum(eager == avt[cells])),
        "nemo": {},
    })
    for p in pdlr_ok:
        nemo = np.maximum(np.float64(p) * np.maximum(zav, fl), fl)
        out["nemo"][repr(p)] = {"ulps": hist(nemo),
                                "equals_stored_jit": int(np.sum(nemo == avt[cells])),
                                "equals_closure_eager": int(np.sum(nemo == eager))}
    log(json.dumps(out, indent=1, default=str))
    return out


def _nominal(model) -> dict:
    c = model.config
    return {**{k: getattr(c.bottom_drag, f) for k, f in DRAG.items()},
            "bottom_drag_scheme": c.bottom_drag.bottom_drag_scheme,
            "kappa_redi": None if c.gm_redi is None else float(np.max(c.gm_redi.kappa_Redi)),
            "visc_ahmt_max": float(np.max(np.asarray(model.z_coord.nemo_ldf_ahmt))),
            **{k: getattr(c.physics.vertical_mixing.tke, f) for k, f in TKE.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", default="rung10")
    parser.add_argument("--rel", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trace-cd0", action="store_true")
    parser.add_argument("--linearity", nargs="+", choices=SCALARS)
    parser.add_argument("--floor-cells", action="store_true")
    args = parser.parse_args()
    import jax
    if not (jax.config.jax_enable_x64 and jax.default_backend() == "cpu"):
        raise SystemExit("needs JAX_ENABLE_X64=1 on CPU")
    if args.trace_cd0:
        print(f"TRACED cd0: {len(str(trace_cd0(args.rung)).splitlines())} jaxpr lines")
        return 0
    if args.output is None:
        parser.error("--output is required with a scan")
    if args.floor_cells:
        args.output.write_text(json.dumps(floor_cells(args.rung), indent=1, default=str) + "\n")
        return 0
    if args.linearity:
        args.output.write_text(json.dumps(linearity(args.rung, args.linearity, args.rel),
                                          indent=1, default=str) + "\n")
        return 0
    args.output.write_text(json.dumps(scan(args.rung, args.rel, args.steps),
                                      indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
