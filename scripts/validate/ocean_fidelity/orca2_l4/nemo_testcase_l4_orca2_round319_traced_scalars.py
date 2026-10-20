#!/usr/bin/env python3
"""Round 319: rung-10 trainable scalars, traced through the production step.

MEASUREMENT ONLY. Each scalar a closure trainer would expose on rung 10 is a
traced multiplicative scale s (= 1) on its config field (the copy the step
reads). ``--scan``: one jitted step body, run at s = 1 (bit parity with the
production ``model.step``) and with each scalar alone at 1 + ``--rel``; per
kt, the number of state values that moved. ``--trace-cd0``: trace
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
        bumped = run({**ones, k: jnp.float64(1.0 + rel)})
        rows = [moved_counts(b, p) for b, p in zip(base, bumped, strict=True)]
        result["scalars"][k] = {"moved_by_kt": rows,
                                "moved_total_by_kt": [sum(r.values()) for r in rows]}
        log(f"[{time.time() - t0:7.1f}s] {k}: {result['scalars'][k]['moved_total_by_kt']}")
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
    step = jax.jit(lambda s: body(state0, s, card.dt_s, *forcing[0]))
    tangent = jax.jit(lambda t: jax.jvp(lambda s: body(state0, s, card.dt_s, *forcing[0]),
                                        (ones,), (t,))[1])
    base = jax.device_get(step(ones))
    out = {"rung": rung, "rel": rel, "worktree": worktree_stamp(allow_dirty=True), "keys": {}}
    for k in keys:
        row = {}
        for r in (rel, 10 * rel):
            row[f"diff@{r:g}"] = _max_abs(jax.device_get(step({**ones, k: jnp.float64(1 + r)})),
                                          base)
        t = jax.device_get(tangent({j: jnp.float64(1.0 if j == k else 0.0) for j in SCALARS}))
        row[f"jvp*{rel:g}"] = {p: v * rel for p, v in _max_abs(t, jax.tree_util.tree_map(
            np.zeros_like, t)).items()}
        out["keys"][k] = row
        log(k, {name: {p: v for p, v in d.items() if p.startswith(".T[")}
                for name, d in row.items()})
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
    args = parser.parse_args()
    import jax
    if not (jax.config.jax_enable_x64 and jax.default_backend() == "cpu"):
        raise SystemExit("needs JAX_ENABLE_X64=1 on CPU")
    if args.trace_cd0:
        print(f"TRACED cd0: {len(str(trace_cd0(args.rung)).splitlines())} jaxpr lines")
        return 0
    if args.output is None:
        parser.error("--output is required with a scan")
    if args.linearity:
        args.output.write_text(json.dumps(linearity(args.rung, args.linearity, args.rel),
                                          indent=1, default=str) + "\n")
        return 0
    args.output.write_text(json.dumps(scan(args.rung, args.rel, args.steps),
                                      indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
