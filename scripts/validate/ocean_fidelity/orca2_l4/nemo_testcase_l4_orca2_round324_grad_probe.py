#!/usr/bin/env python3
"""Round 324: rung-10 one-step (N = 1) gradients on the comparison card
(DECISION 150: implicit ice-ocean top drag with NEMO's recorded rCdU_ice).

Round 323's probe (on round 315's build/make_losses/chain_reverse/
chain_forward) with the ladder's ``comparison_card``; the scaled scalar is
rn_Ud (GMRediConfig.kappa_redi_diffusive_velocity), a traced scale s = 1.
Usage: nemo_testcase_l4_orca2_round324_grad_probe.py <repo> <out.json> <on|off>
"""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1]).resolve()
OUT = Path(sys.argv[2])
for p in (REPO, REPO / "packages/core", REPO / "packages/ocean"):
    sys.path.insert(0, str(p))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round315_grad_probe as probe,
)

KEYS = ("rn_ud",)
DRAG = {"on": True, "off": False}[sys.argv[3]]
res = {"repo": str(REPO), "log": []}
t_start = time.perf_counter()


def save(phase):
    res["log"].append({"phase": phase, "t_s": round(time.perf_counter() - t_start, 1),
                       "peak_rss_gib": probe._rss_gb()})
    print(f"[{time.perf_counter() - t_start:8.1f}s] {phase} rss={probe._rss_gb():.1f}", flush=True)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")


def scaled(config, s):
    g = config.gm_redi
    g = g._replace(
        kappa_redi_diffusive_velocity=g.kappa_redi_diffusive_velocity * s.get("rn_ud", 1.0))
    return config._replace(gm_redi=g)


def traced_step(model):
    raw = type(model).__dict__["_step_jitted"].__wrapped__
    base = model.config

    def step(state, s, dt, freshwater, surface):
        m = copy.copy(model)
        m.config = scaled(base, s)
        state = m._seed_tke_preclosure_carry(state)
        return raw(m, state, dt, freshwater, surface, **SPONGE)

    return jax.jit(step)


def nonfinite_leaves(tree):
    out = {}
    for p, x in jax.tree_util.tree_leaves_with_path(tree):
        x = np.asarray(x)
        if np.issubdtype(x.dtype, np.inexact):
            n = int(np.size(x) - np.count_nonzero(np.isfinite(x)))
            if n:
                out[jax.tree_util.keystr(p)] = n
    return out


assert jax.config.jax_enable_x64 and jax.default_backend() == "cpu"
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as rung10_gate,
)
from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402

try:
    res["worktree"] = worktree_stamp(allow_dirty=True)
except Exception as e:  # stamp is informational here
    res["worktree"] = f"stamp failed: {e}"
card, state0, _ = probe.build("rung10")
if DRAG:
    card = rung10_gate.comparison_card(card)
cfg = card.recipe.model_config
z, dt = card.recipe.z_coord, card.dt_s
root = probe.ROOTS["rung10"]
forcing = [rung10_gate._surface_forcings(
    card, root, rung10_gate.assemble_surface_fields(root, 1), 1)]
SPONGE = {k: v for k, v in rung10_gate.damping_step_kwargs(card, root, 1).items()
          if k == "sponge"}
res["implicit_ice_ocean_top_drag"] = bool(cfg.implicit_ice_ocean_top_drag)
res["rCdU_ice_passed"] = forcing[0][1].ice_ocean_top_drag_rCdu is not None
res["nn_mxlice"] = int(cfg.physics.vertical_mixing.tke.nemo_mxlice)
res["hm_i_passed"] = forcing[0][1].ice_mean_thickness_hm_i is not None
res["damping_target_kt"] = None if not SPONGE else SPONGE["sponge"].nemo_kt
res["geothermal_forced"] = forcing[0][1].geothermal_heat_flux_mwm2 is not None
res["rn_ud"] = float(cfg.gm_redi.kappa_redi_diffusive_velocity)
fw, sf = forcing[0]
PROD_KW = rung10_gate.damping_step_kwargs(card, root, 1)


def new_model(c):
    return LatLonCGridOceanModel(card.recipe.grid, z, c, iwm_forcing=card.recipe.iwm_forcing)


model = new_model(cfg)
losses, tmask = probe.make_losses(model, z)
wet = np.asarray(tmask) > 0.5
prod = jax.device_get(model.step(state0, dt, freshwater=fw, surface_forcing=sf, **PROD_KW))
res["prod_loss"] = {k: float(f(prod)) for k, f in losses.items()}
save("production")

step = traced_step(model)
trace = {}
for k in KEYS:
    try:
        jax.eval_shape(step, state0, {k: jnp.float64(1.0)}, dt, fw, sf)
        trace[k] = "traceable"
    except Exception as e:  # record, do not crash
        trace[k] = f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"
res["traceability"] = trace
keys = tuple(k for k in KEYS if trace[k] == "traceable")
ones = {k: jnp.float64(1.0) for k in keys}
st = jax.device_get(step(state0, ones, dt, fw, sf))
res["forward_parity_non_bit_leaves"] = [
    jax.tree_util.keystr(p) for (p, x), y in zip(jax.tree_util.tree_leaves_with_path(st),
                                                jax.tree_util.tree_leaves(prod))
    if not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)]
del st
save("forward_parity")

rev = probe.chain_reverse(step, state0, ones, forcing, (1,), losses, dt,
                          log=lambda m: print("  " + m, flush=True))
res["reverse"] = {}
for (name, n), r in rev.items():
    g = np.asarray(r["dstate0"].T.data)
    res["reverse"][f"{name}@{n}"] = {
        "loss": r["loss"], "dT0_finite": bool(np.all(np.isfinite(g))),
        "dT0_nonfinite_count": int(np.size(g) - np.count_nonzero(np.isfinite(g))),
        "dT0_l2": float(np.linalg.norm(np.where(np.isfinite(g), g, 0.0))),
        "dT0_nonzero_wet_frac": float(np.mean(g[wet] != 0.0)),
        "dT0_nonzero_dry": int(np.count_nonzero(g[~wet])),
        "nonfinite_cotangent_leaves": nonfinite_leaves(r["dstate0"]),
        "dscales": r["dscales"], "peak_rss_gib": r["peak_rss_gb"]}
save("reverse")

# forward mode along one T direction (round 315's direction 0) and each scalar
rng = np.random.default_rng(315)
d = rng.standard_normal(np.shape(state0.T.data)) * np.asarray(tmask)
d = jnp.asarray(d / np.sqrt(np.mean(d[wet] ** 2)))
jv = probe.chain_forward(step, state0, ones, forcing, (1,), losses, dt,
                         tangent_state=probe._with_t(probe._zero_tangent(state0), d))
res["T_dir0"] = {}
for name in losses:
    ad = float(jnp.vdot(rev[(name, 1)]["dstate0"].T.data, d))
    fwd = jv[(name, 1)][1]
    res["T_dir0"][name] = {"reverse": ad, "forward": fwd, "rel": probe.rel_err(ad, fwd)}
save("T_dir0 jvp")
res["scalars"] = {}
for k in keys:
    tan = {kk: jnp.float64(1.0 if kk == k else 0.0) for kk in keys}
    jv = probe.chain_forward(step, state0, ones, forcing, (1,), losses, dt, tangent_scales=tan)
    row = {}
    for name in losses:
        ad = rev[(name, 1)]["dscales"][k]
        fwd = jv[(name, 1)][1]
        fd = {}
        for eps in (1e-2, 1e-4, 1e-6):
            up = {kk: jnp.float64(1.0 + (eps if kk == k else 0.0)) for kk in keys}
            dn = {kk: jnp.float64(1.0 - (eps if kk == k else 0.0)) for kk in keys}
            p = probe.chain_forward(step, state0, up, forcing, (1,), {name: losses[name]}, dt)
            m = probe.chain_forward(step, state0, dn, forcing, (1,), {name: losses[name]}, dt)
            fd[f"{eps:g}"] = (p[(name, 1)] - m[(name, 1)]) / (2 * eps)
        row[name] = {"reverse": ad, "forward": fwd, "rev_fwd_rel": probe.rel_err(ad, fwd),
                     "fd": fd, "fd_rel": {e: probe.rel_err(ad, v) for e, v in fd.items()}}
    res["scalars"][k] = row
    save(f"scalar {k}")

# construction parity: production model REBUILT with the scaled config
jax.clear_caches()
eps = 1e-4
res["construction"] = {}
for k in keys:
    c2 = scaled(cfg, {k: 1.0 + eps})
    m2 = new_model(c2)
    p2 = jax.device_get(m2.step(state0, dt, freshwater=fw, surface_forcing=sf, **PROD_KW))
    traced = jax.device_get(step(state0, {kk: jnp.float64(1.0 + (eps if kk == k else 0.0))
                                          for kk in keys}, dt, fw, sf))
    res["construction"][k] = {
        "eps": eps,
        "rebuilt_vs_traced_non_bit_leaves": [
            jax.tree_util.keystr(p) for (p, x), y in zip(
                jax.tree_util.tree_leaves_with_path(p2), jax.tree_util.tree_leaves(traced))
            if not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)],
        "rebuilt_vs_nominal_T_cells_changed": int(np.count_nonzero(
            np.asarray(p2.T.data) != np.asarray(prod.T.data))),
        "rebuilt_vs_nominal_S_cells_changed": int(np.count_nonzero(
            np.asarray(p2.S.data) != np.asarray(prod.S.data))),
        "rebuilt_loss": {kk: float(f(p2)) for kk, f in losses.items()}}
    del m2, p2, traced
    jax.clear_caches()
    save(f"construction {k}")
res["complete"] = True
save("done")
print("STATUS DONE")
