#!/usr/bin/env python3
"""Round 315: reverse-mode differentiability of the NEMO-exact ORCA2 step.

MEASUREMENT ONLY: no model file is touched and no forward value changes.

For one card (rung 0, OMT-1..4, rung 10) and N = 1, 2, 5, 10 steps:
  losses   L_ssh = area-weighted mean of SSH^2 over wet T points,
           L_T   = volume-weighted mean of T^2 over wet T cells (static
                   reference volume area * h_partial * tmask);
  inputs   (a) the initial T field, (b) four multiplicative scales s (= 1)
           on the isoneutral diffusivity, the lateral momentum viscosity
           (the 3-D ahmt/ahmf arrays), the background vertical diffusivity
           and the linear bottom-drag coefficient, passed TRACED.
Reverse mode is one jax.vjp per step of the production step body
(``_step_jitted.__wrapped__`` under this script's own jax.jit), chained by
hand so every N reuses one forward.  Checks: finite, nonzero, central finite
differences with an epsilon sweep, and forward-mode (jvp) agreement (the
kink-immune transpose check of tests/ocean/validation/test_step_gradient_matrix.py).

Controls: (1) forward parity — the traced-scale path at s = 1 reproduces the
production ``model.step`` trajectory bit for bit; (2) construction parity —
for each scalar, the traced path at s = 1 +- eps reproduces a model REBUILT
with the scaled configuration (else the scalar is folded at construction and
the traced gradient is not the model's gradient).
"""

from __future__ import annotations

import argparse
import copy
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

ROUNDS = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds")
ROOTS = {
    "rung0": ROUNDS / "round90/acquisition/orca2_rung0_entry_stage_runoff_guarded_10step_np2",
    "omt1": ROUNDS / "round211/acquisition/orca2_omt1_frames_8step_a_np2",
    "omt2": ROUNDS / "round218/acquisition/orca2_omt2_frames_10step_a_np2",
    "omt3": ROUNDS / "round220/acquisition/orca2_omt3_frames_10step_a_np2",
    "omt4": ROUNDS / "round222/acquisition/orca2_omt4_frames_10step_a_np2",
    "rung10": ROUNDS / "round5/acquisition/orca1ice_surface_entry_every_step_a_np2",
}
SCALARS = ("kappa_redi", "visc_scale", "kv_background", "bottom_cd0")
N_LIST = (1, 2, 5, 10)
T_EPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)          # K, along a unit-RMS direction
S_EPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)          # relative, on s = 1
# Accepted |AD - FD| / max(|AD|, |FD|) at the best epsilon of the sweep: fp64,
# central differences, truncation O(eps^2) and rounding O(1e-16 L / eps) are
# both <= ~1e-8 for a smooth loss at mid-sweep, so 1e-6 leaves 100x margin; a
# miss means the function is not smooth there (or the gradient is wrong, which
# the jvp check separates).
FD_TOL = 1e-6
REVFWD_TOL = 1e-10


class GateError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise GateError(message)


def rel_err(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return 0.0 if scale == 0.0 else abs(a - b) / scale


def best_fd(ad: float, fd_by_eps: dict) -> dict:
    """Best relative error over the epsilon sweep (and the epsilon that gave it)."""

    errs = {eps: rel_err(ad, fd) for eps, fd in fd_by_eps.items()}
    eps = min(errs, key=errs.get)
    return {"eps": eps, "rel_err": errs[eps], "pass": errs[eps] <= FD_TOL}


# --- the four scalars -------------------------------------------------------

def _vm_background_path(config) -> str:
    scheme = config.physics.vertical_mixing.scheme
    require(scheme in ("tke", "constant"), f"unexpected vertical mixing {scheme}")
    return "tke.kappaH_min" if scheme == "tke" else "constant.K_v"


def scaled_parts(config, z_coord, s: dict):
    """Return (config, z_coord) with every scalar multiplied by its scale.

    At s = 1 every product is exactly the nominal value (x * 1.0 == x in IEEE),
    traced or not.
    """

    s = {k: s.get(k, 1.0) for k in SCALARS}  # absent = held static (untraceable)
    ph = config.physics
    lm = ph.lateral_mixing
    vm = ph.vertical_mixing
    redi = lm.gm_redi._replace(kappa_Redi=lm.gm_redi.kappa_Redi * s["kappa_redi"])
    if _vm_background_path(config) == "tke.kappaH_min":
        vm = vm._replace(tke=vm.tke._replace(
            kappaH_min=vm.tke.kappaH_min * s["kv_background"]))
    else:
        vm = vm._replace(constant=vm.constant._replace(
            K_v=vm.constant.K_v * s["kv_background"]))
    config = config._replace(
        physics=ph._replace(lateral_mixing=lm._replace(gm_redi=redi),
                            vertical_mixing=vm),
        bottom_drag=config.bottom_drag._replace(
            bottom_drag_cd0=config.bottom_drag.bottom_drag_cd0 * s["bottom_cd0"]))
    z_coord = z_coord._replace(
        nemo_ldf_ahmt=z_coord.nemo_ldf_ahmt * s["visc_scale"],
        nemo_ldf_ahmf=z_coord.nemo_ldf_ahmf * s["visc_scale"])
    return config, z_coord


def nominal_scalars(config, z_coord) -> dict:
    vm = config.physics.vertical_mixing
    bg = (vm.tke.kappaH_min if _vm_background_path(config) == "tke.kappaH_min"
          else vm.constant.K_v)
    return {"kappa_redi": float(config.physics.lateral_mixing.gm_redi.kappa_Redi),
            "visc_scale": float(np.max(np.asarray(z_coord.nemo_ldf_ahmt))),
            "kv_background": float(bg),
            "kv_background_field": _vm_background_path(config),
            "bottom_cd0": float(config.bottom_drag.bottom_drag_cd0)}


# --- card, forcing, losses --------------------------------------------------

def build(rung: str):
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round1_ladder_gate as rung10_gate,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round92_rung0_card_gate as rung0_gate,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round103_rung0_ladder_gate as rung0_ladder,
    )
    from scripts.validate.ocean_fidelity.orca2_l4 import (
        nemo_testcase_l4_orca2_round269_g1_decision96_gate as prior,
    )

    root = ROOTS[rung]
    steps = 8 if rung == "omt1" else 10
    if rung == "rung10":
        _, card = rung10_gate.card_fields(root)
        state = card.recipe.initial_state
        forcing = [rung10_gate._surface_forcings(
            card, root, rung10_gate.assemble_surface_fields(root, kt), kt)
            for kt in range(1, steps + 1)]
    else:
        card = prior._selected_card(rung, root)
        state = (rung0_gate.bridge_entry(card, rung0_gate.assemble_frame(root, 1, 0))
                 if rung == "rung0" else card.recipe.initial_state)
        forcing = [rung0_ladder._zero_forcing((148, 180))] * steps
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy, "fp64/libm policy is not active")
    return card, state, forcing


def make_losses(model, z_coord):
    import jax.numpy as jnp

    area = jnp.asarray(model.grid.area)
    tmask = jnp.asarray(z_coord.is_active, dtype=jnp.float64)
    require(area.shape == tmask.shape[:2], f"area {area.shape} vs tmask {tmask.shape}")
    w2 = area * tmask[..., 0]
    w3 = area[..., None] * jnp.asarray(z_coord.h_partial) * tmask

    def loss_ssh(state):
        return jnp.sum(w2 * state.eta.data ** 2) / jnp.sum(w2)

    def loss_t(state):
        return jnp.sum(w3 * state.T.data ** 2) / jnp.sum(w3)

    return {"ssh": loss_ssh, "T": loss_t}, tmask


def traced_step(model):
    """The production jitted step body with traced scales (one jax.jit)."""

    import jax

    raw = type(model).__dict__["_step_jitted"].__wrapped__
    base_cfg, base_z = model.config, model.z_coord

    def step(state, s, dt, freshwater, surface):
        m = copy.copy(model)
        m.config, m.z_coord = scaled_parts(base_cfg, base_z, s)
        state = m._seed_tke_preclosure_carry(state)
        return raw(m, state, dt, freshwater, surface)

    return jax.jit(step)


def traceability(model, state0, forcing, dt) -> dict:
    """Trace the step with ONE scalar traced at a time; name the first refusal.

    A scalar the step converts to a Python value (``float(x)``, ``if x``) cannot
    carry a gradient at all; the innermost repository frame is recorded.
    """

    import traceback

    import jax
    import jax.numpy as jnp

    step = traced_step(model)
    fw, sf = forcing[0]
    out = {}
    for key in SCALARS:
        try:
            jax.eval_shape(step, state0, {key: jnp.float64(1.0)}, dt, fw, sf)
            out[key] = {"traceable": True}
        except (jax.errors.ConcretizationTypeError,
                jax.errors.TracerBoolConversionError,
                jax.errors.TracerIntegerConversionError) as error:
            frames = [f for f in traceback.extract_tb(error.__traceback__)
                      if "/packages/" in f.filename]
            f = frames[-1]
            out[key] = {"traceable": False, "error": type(error).__name__,
                        "first_statement": f"{f.filename.split('/packages/')[1]}:{f.lineno}",
                        "code": f.line, "function": f.name}
    return out


# --- chained reverse / forward mode ------------------------------------------

def _zero_tangent(tree):
    import jax
    import jax.numpy as jnp

    def zero(x):
        if getattr(x, "dtype", None) == jax.dtypes.float0:
            return x
        x = jnp.asarray(x)
        if jnp.issubdtype(x.dtype, jnp.inexact):
            return jnp.zeros_like(x)
        return np.zeros(x.shape, dtype=jax.dtypes.float0)

    return jax.tree.map(zero, tree)


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2


def _maps() -> int:
    try:
        with open("/proc/self/maps") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return -1


def chain_reverse(step, state0, scales, forcing, n_list, losses, dt, log=print):
    """Gradients of every loss at every N in ``n_list`` w.r.t. state0 and scales.

    One forward per step with its pullback kept; each loss cotangent is pulled
    back through steps N..1.  Compiled executables are dropped
    (``jax.clear_caches``) between stages so that at most one reverse-mode
    executable is resident: one ORCA2 reverse executable holds tens of
    thousands of memory maps against a 65,530 per-process limit.  Step 1 has
    its own input structure (the cold-start state), so its pullback is applied
    last, to every stored cotangent at once.  Returns per (loss, N): dstate0,
    dscales, backward seconds (sum over stages, compile excluded), peak RSS.
    """

    import jax

    jax.clear_caches()
    pullbacks, state, at_n = [], state0, {}
    for n in range(1, max(n_list) + 1):
        fw, sf = forcing[n - 1]
        t0 = time.perf_counter()
        state, pb = jax.vjp(lambda st, sc: step(st, sc, dt, fw, sf), state, scales)
        jax.block_until_ready(state)
        pullbacks.append(pb)
        log(f"vjp forward step {n}: {time.perf_counter() - t0:.1f}s maps={_maps()}")
        if n == 1:
            jax.clear_caches()
        if n in n_list:
            for name, loss in losses.items():
                value, lpb = jax.vjp(loss, state)
                at_n[(name, n)] = (float(value), lpb(jax.numpy.ones_like(value))[0])
    jax.clear_caches()

    def run_pullbacks(ct, steps):
        dscale = jax.tree.map(jax.numpy.zeros_like, scales)
        for k in steps:
            ct, ds = pullbacks[k - 1](ct)
            dscale = jax.tree.map(lambda a, b: a + b, dscale, ds)
        jax.block_until_ready((ct, dscale))
        return ct, dscale

    # warm (compile) each stage once so the timed runs exclude compilation
    out = {}
    if max(n_list) > 1:
        t0 = time.perf_counter()
        run_pullbacks(_zero_tangent(state), (2,))
        log(f"transpose compile (steps>=2): {time.perf_counter() - t0:.1f}s maps={_maps()}")
    for key, (value, ct) in at_n.items():
        t0 = time.perf_counter()
        ct, ds = run_pullbacks(ct, range(key[1], 1, -1))
        out[key] = {"loss": value, "ct1": ct, "ds": ds,
                    "backward_s": time.perf_counter() - t0}
        log(f"backward {key} steps {key[1]}..2: {out[key]['backward_s']:.1f}s maps={_maps()}")
    jax.clear_caches()
    t0 = time.perf_counter()
    jax.block_until_ready(pullbacks[0](_zero_tangent(next(iter(out.values()))["ct1"])))
    log(f"transpose compile (step 1): {time.perf_counter() - t0:.1f}s maps={_maps()}")
    result = {}
    for key, r in out.items():
        t0 = time.perf_counter()
        ct0, ds = run_pullbacks(r["ct1"], (1,))
        dscale = jax.tree.map(lambda a, b: a + b, r["ds"], ds)
        result[key] = {"loss": r["loss"], "dstate0": ct0,
                       "dscales": {k: float(v) for k, v in dscale.items()},
                       "backward_s": r["backward_s"] + time.perf_counter() - t0,
                       "peak_rss_gb": _rss_gb()}
    log(f"backward step 1 for all: maps={_maps()}")
    jax.clear_caches()
    return result


def chain_forward(step, state0, scales, forcing, n_list, losses, dt,
                  tangent_state=None, tangent_scales=None):
    """Losses (and, with tangents, their jvp) at every N from one trajectory."""

    import jax

    state = state0
    ts = _zero_tangent(state0) if tangent_state is None else tangent_state
    tsc = (jax.tree.map(jax.numpy.zeros_like, scales)
           if tangent_scales is None else tangent_scales)
    want_jvp = tangent_state is not None or tangent_scales is not None
    out = {}
    for n in range(1, max(n_list) + 1):
        fw, sf = forcing[n - 1]
        if want_jvp:
            # the scale tangent enters every step; the state tangent is carried
            state, ts = jax.jvp(lambda st, sc: step(st, sc, dt, fw, sf),
                                (state, scales), (ts, tsc))
        else:
            state = step(state, scales, dt, fw, sf)
        if n in n_list:
            for name, loss in losses.items():
                if want_jvp:
                    value, dot = jax.jvp(loss, (state,), (ts,))
                    out[(name, n)] = (float(value), float(dot))
                else:
                    out[(name, n)] = float(loss(state))
    return out


def _with_t(state, t_data):
    return state._replace(T=state.T.replace(data=t_data))


# --- measurement ----------------------------------------------------------

def measure(rung: str, n_dirs: int, full_sweep: bool, output: Path | None = None) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.provenance import worktree_stamp

    require(bool(jax.config.jax_enable_x64) and jax.default_backend() == "cpu"
            and not jax.config.jax_disable_jit, "needs x64, CPU, production JIT")
    card, state0, forcing = build(rung)
    cfg, z = card.recipe.model_config, card.recipe.z_coord
    dt = card.dt_s
    n_list = tuple(n for n in N_LIST if n <= len(forcing))

    def new_model(config, z_coord):
        return LatLonCGridOceanModel(card.recipe.grid, z_coord, config,
                                     iwm_forcing=card.recipe.iwm_forcing)

    model = new_model(cfg, z)
    losses, tmask = make_losses(model, z)
    result = {"format": "nemo-testcase-l4-orca2-round315-grad-probe-v1",
              "rung": rung, "worktree": worktree_stamp(allow_dirty=True),
              "jax": jax.__version__, "n_list": list(n_list), "dt_s": dt,
              "nominal": nominal_scalars(cfg, z),
              "fd_tol": FD_TOL, "revfwd_tol": REVFWD_TOL, "log": []}
    clock = time.perf_counter()

    def save(phase: str) -> None:
        # partial results survive a long run; the final write is the record
        result["log"].append({"phase": phase, "t_s": time.perf_counter() - clock,
                              "peak_rss_gb": _rss_gb()})
        print(f"[{time.perf_counter() - clock:8.1f}s] {phase}", flush=True)
        if output is not None:
            output.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")

    # production trajectory (also primes the model's step caches)
    t0 = time.perf_counter()
    prod, st = [], state0
    for n in range(1, max(n_list) + 1):
        fw, sf = forcing[n - 1]
        st = model.step(st, dt, freshwater=fw, surface_forcing=sf)
        prod.append(jax.device_get(st))
    result["production_forward_s"] = time.perf_counter() - t0
    result["production_losses"] = {f"{k}@{n}": float(f(prod[n - 1]))
                                   for k, f in losses.items() for n in n_list}
    save("production")

    trace = traceability(model, state0, forcing, dt)
    result["traceability"] = trace
    save("traceability")
    keys = tuple(k for k in SCALARS if trace[k]["traceable"])
    step = traced_step(model)
    ones = {k: jnp.float64(1.0) for k in keys}
    t0 = time.perf_counter()
    st, parity = state0, []
    for n in range(1, max(n_list) + 1):
        fw, sf = forcing[n - 1]
        st = step(st, ones, dt, fw, sf)
        a, b = jax.device_get(st), prod[n - 1]
        bad = [jax.tree_util.keystr(p) for (p, x), y in zip(
            jax.tree_util.tree_leaves_with_path(a), jax.tree_util.tree_leaves(b))
            if not np.array_equal(np.asarray(x), np.asarray(y), equal_nan=True)]
        parity.append({"kt": n, "non_bit_leaves": bad})
    result["traced_forward_s_incl_compile"] = time.perf_counter() - t0
    result["forward_parity"] = parity
    result["forward_parity_bit_identical"] = all(not p["non_bit_leaves"] for p in parity)
    save("forward_parity")

    # reverse mode
    a_tree = jax.tree_util.tree_structure(state0)
    b_tree = jax.tree_util.tree_structure(prod[0])
    result["step1_structure_differs"] = a_tree != b_tree
    if a_tree == b_tree:
        result["step1_aval_diffs"] = [
            f"{jax.tree_util.keystr(pa)} {jnp.asarray(x).dtype}/{getattr(x, 'weak_type', None)}"
            f" -> {jnp.asarray(y).dtype}/{getattr(y, 'weak_type', None)}"
            for (pa, x), y in zip(jax.tree_util.tree_leaves_with_path(state0),
                                  jax.tree_util.tree_leaves(prod[0]))
            if (np.shape(x), jnp.asarray(x).dtype) != (np.shape(y), jnp.asarray(y).dtype)][:20]
    del prod
    rev = chain_reverse(step, state0, ones, forcing, n_list, losses, dt,
                        log=lambda msg: print(f"  {msg}", flush=True))
    jax.clear_caches()
    wet = np.asarray(tmask) > 0.5
    grads = {}
    for (name, n), r in rev.items():
        gt = np.asarray(r["dstate0"].T.data)
        leaves = jax.tree_util.tree_leaves_with_path(r["dstate0"])
        nonfinite = [jax.tree_util.keystr(p) for p, x in leaves
                     if jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact)
                     and not np.all(np.isfinite(np.asarray(x)))]
        grads[f"{name}@{n}"] = {
            "loss": r["loss"], "backward_s": r["backward_s"],
            "peak_rss_gb": r["peak_rss_gb"],
            "dT0_finite": bool(np.all(np.isfinite(gt))),
            "dT0_l2": float(np.linalg.norm(np.where(np.isfinite(gt), gt, 0.0))),
            "dT0_nonzero_wet_frac": float(np.mean(gt[wet] != 0.0)),
            "dT0_nonzero_dry": int(np.count_nonzero(gt[~wet])),
            "nonfinite_state_cotangents": nonfinite,
            "dscales": r["dscales"],
        }
    result["reverse"] = grads
    result["checks"] = checks = []
    save("reverse")

    # directional checks: T directions (FD sweep + jvp), scalars (FD sweep + jvp)
    rng = np.random.default_rng(315)
    t_eps = T_EPS if full_sweep else T_EPS[2:4]
    s_eps = S_EPS if full_sweep else S_EPS[2:4]
    t0_data = state0.T.data
    for d_i in range(n_dirs):
        d = rng.standard_normal(np.shape(t0_data)) * np.asarray(tmask)
        d = d / np.sqrt(np.mean(d[wet] ** 2))
        d = jnp.asarray(d)
        jvp = _try_jvp(step, state0, ones, forcing, n_list, losses, dt,
                       tangent_state=_with_t(_zero_tangent(state0), d))
        fd = {eps: {k: (p - m) / (2 * eps) for (k, p), m in zip(
            chain_forward(step, _with_t(state0, t0_data + eps * d), ones, forcing,
                          n_list, losses, dt).items(),
            chain_forward(step, _with_t(state0, t0_data - eps * d), ones, forcing,
                          n_list, losses, dt).values())} for eps in t_eps}
        for (name, n) in rev:
            ad = float(jnp.vdot(rev[(name, n)]["dstate0"].T.data, d))
            checks.append(_row(f"T_dir{d_i}", name, n, ad, jvp.get((name, n), (0, None))[1],
                               {e: fd[e][(name, n)] for e in t_eps}))
        save(f"T_dir{d_i}")
    for key in SCALARS:
        if key not in keys:
            row = _rebuilt_fd(key, cfg, z, new_model, state0, forcing,
                              n_list, losses, dt, s_eps[len(s_eps) // 2])
            row.pop("_plus")
            checks.append(row)
            save(f"scalar {key} (rebuilt only)")
            continue
        tan = {k: jnp.float64(1.0 if k == key else 0.0) for k in keys}
        jvp = _try_jvp(step, state0, ones, forcing, n_list, losses, dt,
                       tangent_scales=tan)
        fd = {}
        for eps in s_eps:
            up = {k: jnp.float64(1.0 + (eps if k == key else 0.0)) for k in keys}
            dn = {k: jnp.float64(1.0 - (eps if k == key else 0.0)) for k in keys}
            p = chain_forward(step, state0, up, forcing, n_list, losses, dt)
            m = chain_forward(step, state0, dn, forcing, n_list, losses, dt)
            fd[eps] = {k: (p[k] - m[k]) / (2 * eps) for k in p}
        for (name, n) in rev:
            checks.append(_row(key, name, n, rev[(name, n)]["dscales"][key],
                               jvp.get((name, n), (0, None))[1],
                               {e: fd[e][(name, n)] for e in s_eps}))
        # construction parity: a model REBUILT with the scaled config at the
        # sweep's middle epsilon, run through production model.step.
        eps = s_eps[len(s_eps) // 2]
        row = _rebuilt_fd(key, cfg, z, new_model, state0, forcing, n_list,
                          losses, dt, eps)
        traced_up = chain_forward(step, state0, {k: jnp.float64(1.0 + (eps if k == key else 0.0))
                                                 for k in keys}, forcing, n_list, losses, dt)
        row["rebuilt_vs_traced_plus_bit_identical"] = all(
            row["_plus"][kn] == traced_up[kn] for kn in row["_plus"])
        row.pop("_plus")
        checks.append(row)
        save(f"scalar {key}")
    result["peak_rss_gb"] = _rss_gb()
    result["complete"] = True
    save("done")
    return result


def _rebuilt_fd(key, cfg, z, new_model, state0, forcing, n_list, losses, dt, eps):
    """Central FD of a model REBUILT with the scaled config (production step)."""

    rebuilt = {}
    for sign in (+1, -1):
        m_cfg, m_z = scaled_parts(cfg, z, {key: 1.0 + sign * eps})
        mm, st = new_model(m_cfg, m_z), state0
        vals = {}
        for n in range(1, max(n_list) + 1):
            fw, sf = forcing[n - 1]
            st = mm.step(st, dt, freshwater=fw, surface_forcing=sf)
            if n in n_list:
                vals.update({(k, n): float(f(st)) for k, f in losses.items()})
        rebuilt[sign] = vals
    return {"input": key, "construction_parity_eps": eps, "_plus": rebuilt[1],
            "rebuilt_fd": {f"{k}@{n}": (rebuilt[1][(k, n)] - rebuilt[-1][(k, n)]) / (2 * eps)
                           for (k, n) in rebuilt[1]}}


def _try_jvp(*args, **kwargs) -> dict:
    """Forward mode, or {} when the step holds a custom_vjp (no JVP rule)."""

    try:
        return chain_forward(*args, **kwargs)
    except TypeError as error:
        if "custom_vjp" not in str(error):
            raise
        return {}


def _row(inp, loss, n, ad, fwd, fd_by_eps):
    revfwd = None if fwd is None else rel_err(ad, fwd)
    return {"input": inp, "loss": loss, "N": n, "ad_reverse": ad, "ad_forward": fwd,
            "revfwd_rel": revfwd,
            "revfwd_pass": None if fwd is None else revfwd <= REVFWD_TOL,
            "fd": {f"{e:g}": v for e, v in fd_by_eps.items()},
            "fd_rel": {f"{e:g}": rel_err(ad, v) for e, v in fd_by_eps.items()},
            "best": best_fd(ad, fd_by_eps),
            "finite": bool(np.isfinite(ad)), "nonzero": ad != 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rung", choices=tuple(ROOTS), required=True)
    parser.add_argument("--directions", type=int, default=3)
    parser.add_argument("--short-sweep", action="store_true",
                        help="two epsilons per input instead of six")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = measure(args.rung, args.directions, not args.short_sweep, args.output)
    except GateError as error:
        print(f"STATUS REFUSE: {error}")
        return 2
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n")
    print(f"STATUS MEASURED {args.rung} parity={result['forward_parity_bit_identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
