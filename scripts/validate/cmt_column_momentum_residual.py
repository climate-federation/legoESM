"""Column momentum residual of the convective momentum tendency on a run checkpoint.

    cmt_column_momentum_residual.py --config <deck> --restart <ckpt.npz>
        [--expect on|off] [--label X] [--json out.json] [--extra <run_amip flags>]

Issue #2266 pre-registration, item 3: the scheme's column momentum conservation is
unit-tested on synthetic columns, but the run publishes no convective momentum
tendency.  This probe restores a run checkpoint through the production driver,
calls the deck's OWN convection bridge (the MPAS factory the run uses) with the
run's convective carry (``physstate_conv_prog_profile``), and captures the
scheme's cell-column (FU, FV) exactly as the bridge passed them on.

Gate, per column, for FU and FV:   |sum f dp| <= TOL * sum |f| dp,  TOL = 1e-12.
``--expect on``: at least one column must carry a nonzero tendency, and a control
(one column deliberately broken by a uniform offset) must be flagged, or the gate
is vacuous.  ``--expect off``: the scheme must return no tendency and the bridge's
edge-wind tendency must be exactly zero.

Runs on CPU in float64.  Exit 0 = gate held; 1 = gate violated; SystemExit text
otherwise.  NUMBERS ONLY -- the verdict belongs to the analysis.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ["JAX_ENABLE_X64"] = "1"   # forced: the 1e-12 gate is a float64 statement
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
TOL = 1e-12          # pre-registered (#2266, user-approved 2026-10-31)
CONTROL_OFFSET = 1e-6  # control: uniform offset, as a fraction of the column's sum|f|dp


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--restart", required=True)
    ap.add_argument("--expect", choices=("on", "off"), required=True)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--json", default=None)
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    return ap


def column_residual(f, dp):
    """Per-column (|sum f dp|, sum |f| dp) for a (ncol, nlev) tendency."""
    f = np.asarray(f, dtype=np.float64)
    dp = np.asarray(dp, dtype=np.float64)
    return np.abs(np.sum(f * dp, axis=1)), np.sum(np.abs(f) * dp, axis=1)


def violations(f, dp, tol=TOL):
    """Indices of columns with |sum f dp| > tol * sum |f| dp."""
    resid, scale = column_residual(f, dp)
    return np.flatnonzero(resid > tol * scale)


def broken_control(f, dp, frac=CONTROL_OFFSET):
    """Copy of ``f`` whose largest column gets a uniform offset carrying
    ``frac`` of its sum|f|dp as net column momentum, and that column's index."""
    f = np.array(f, dtype=np.float64)
    _, scale = column_residual(f, dp)
    i = int(np.argmax(scale))
    f[i] += frac * scale[i] / np.sum(dp[i])
    return f, i


def summarize(f, dp, tol=TOL):
    resid, scale = column_residual(f, dp)
    active = scale > 0.0
    rel = np.where(active, resid / np.where(active, scale, 1.0), 0.0)
    return {"n_columns": int(f.shape[0]), "n_active": int(active.sum()),
            "max_rel_residual": float(rel.max()), "max_scale": float(scale.max()),
            "n_violations": int(violations(f, dp, tol).size)}


def gate(fu, fv, dp, expect, edge_du=None, tol=TOL):
    """Apply the pre-registered gate; return (ok, report)."""
    if expect == "off":
        if fu is not None or fv is not None:
            return False, {"error": "off arm: scheme returned a momentum tendency"}
        if edge_du is None:
            return False, {"error": "off arm: no edge-wind tendency to check"}
        n_nonzero = int(np.count_nonzero(np.asarray(edge_du)))
        return n_nonzero == 0, {"edge_du_nonzero": n_nonzero}
    if fu is None or fv is None:
        return False, {"error": "on arm: scheme returned no momentum tendency"}
    if not (np.isfinite(fu).all() and np.isfinite(fv).all()):
        return False, {"error": "on arm: non-finite momentum tendency"}
    rep, ok = {}, True
    for name, f in (("FU", fu), ("FV", fv)):
        s = summarize(f, dp, tol)
        ok &= s["n_violations"] == 0
        if s["n_active"] > 0:      # the control needs a nonzero column to break
            bad, i = broken_control(f, dp)
            s["control_column"] = i
            s["control_flagged"] = bool(i in violations(bad, dp, tol))
            ok &= s["control_flagged"]
        rep[name] = s
    ok &= rep["FU"]["n_active"] + rep["FV"]["n_active"] > 0   # a nonzero tendency exists
    return bool(ok), rep


def capture_bridge(state, mesh, sc, conv_cfg, dt, phys_state):
    """Call the MPAS convection bridge; return (FU, FV, p_half, edge du_dt,
    carry, prog_in): FU/FV, p_half and the convective carry exactly as the
    bridge passed them to / received them from the scheme."""
    from legoesm.atmosphere.physics.convection import integration as integ
    if conv_cfg.scheme != "emanuel":
        raise SystemExit(f"probe covers the Emanuel scheme only; deck selects {conv_cfg.scheme}")
    seen = {}
    orig = integ.emanuel_convection

    def _spy(**kw):
        out, prog = orig(**kw)
        seen.update(out=out, p_half=kw["p_half"], prog_in=kw["conv_prog_profile"])
        return out, prog

    integ.emanuel_convection = _spy
    try:
        fn = integ.make_convection_physics(conv_cfg, model_type="mpas", dt=dt)
        res = fn(state, mesh, sc, phys_state, None)
    finally:
        integ.emanuel_convection = orig
    if "out" not in seen:
        raise SystemExit("the bridge never called the Emanuel scheme")
    tend, carry = res
    out = seen["out"]
    as_np = (lambda x: None if x is None else np.asarray(x))
    return (as_np(out.du_dt_conv), as_np(out.dv_dt_conv),
            np.asarray(seen["p_half"]), np.asarray(tend.du_dt.data), carry,
            np.asarray(seen["prog_in"]))


def main():
    a = build_arg_parser().parse_args()
    import jax.numpy as jnp
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "run_amip_cmtprobe", _ROOT / "scripts" / "run" / "run_amip.py")
    ra = importlib.util.module_from_spec(spec)
    sys.modules["run_amip_cmtprobe"] = ra
    spec.loader.exec_module(ra)
    argv = ["--config", a.config] + list(a.extra)
    parser = ra.build_arg_parser()
    from legoesm.driver.run_config_yaml import load_yaml_config
    keys = load_yaml_config(a.config, parser)
    parser.set_defaults(**keys)
    parser.set_defaults(_config_keys=frozenset(keys))
    args = parser.parse_args(argv)
    args = ra._postprocess_args(args, parser, argv)
    ra._apply_spectral_scheme_fallback(args, argv, parser)
    config = ra.build_config_from_args(args)

    from legoesm.driver.model_driver import ModelDriver
    driver = ModelDriver(config)
    driver.setup()
    step0, day0 = driver.load_checkpoint(a.restart)
    state, model = driver.state, driver.model
    mesh, sc = model.mesh, model.sigma_coord
    ncol, nlev = np.asarray(state.T.data).shape

    from legoesm.driver.physics_pipeline import convection_config_for
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.physics_state import (
        init_physics_state, update_physics_state)
    conv_cfg = convection_config_for(config)
    if "physstate_conv_prog_profile" not in driver._carry_aux:
        raise SystemExit("checkpoint carries no physstate_conv_prog_profile "
                         "(Emanuel's cloud-base mass flux); refusing a fresh carry")
    ps = update_physics_state(
        init_physics_state(ncol, nlev, PhysicsConfig(convection=conv_cfg)),
        {"conv_prog_profile": jnp.asarray(driver._carry_aux["physstate_conv_prog_profile"])})
    dt_phys = float(config.dycore.dt) * int(getattr(config, "physics_update_steps", 1) or 1)

    fu, fv, p_half, edge_du, _, prog_in = capture_bridge(state, mesh, sc, conv_cfg, dt_phys, ps)
    # The bridge silently substitutes zeros for a carry of the wrong shape:
    # the scheme must have seen the checkpoint's carry, bit for bit.
    if not np.array_equal(prog_in, np.asarray(driver._carry_aux["physstate_conv_prog_profile"])):
        raise SystemExit("the bridge did not hand the checkpoint's conv_prog_profile to the scheme")
    dp = p_half[:, 1:] - p_half[:, :-1]
    if p_half.dtype != np.float64 or (fu is not None and fu.dtype != np.float64):
        raise SystemExit(f"the scheme ran in {p_half.dtype}, not float64: the 1e-12 gate needs fp64")
    if not (dp > 0).all():
        raise SystemExit("p_half is not surface-last (non-positive layer thickness)")
    ok, rep = gate(fu, fv, dp, a.expect, edge_du=edge_du)
    try:
        sha = subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = "unknown"
    rec = {"label": a.label, "restart": str(a.restart), "config": str(a.config),
           "extra": a.extra, "expect": a.expect, "step": int(step0), "day": float(day0),
           "scheme": conv_cfg.scheme, "enable_cmt": bool(conv_cfg.emanuel.enable_cmt),
           "dt_phys": dt_phys, "tol": TOL, "git_sha": sha, "gate_held": ok, **rep}
    print(json.dumps(rec, indent=1), flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rec, indent=1))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
