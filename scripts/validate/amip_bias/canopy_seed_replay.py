"""Replay every cached canopy state in an AMIP checkpoint as a warm-start seed.

Seeds: ``land_ml_canopy_x`` rows of a checkpoint (real).  Forcing: SYNTHETIC
winter bundle (test_canopy_warm_start._bundle) with each cell's own lowest-level
air temperature, humidity and surface pressure; LAI and night/day are fixed per
pass.  Compares the relative-only gate (the pre-fix contract) with the
certified gate (relative + absolute ceiling + physical box).

Usage: scripts/validate/amip_bias/canopy_seed_replay.py <checkpoint.npz> [--lai 0.1] [--day]
Run with JAX_ENABLE_X64=1 and tests/land/unit on PYTHONPATH.
"""
import argparse
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.core.nonlinear import make_implicit_newton_solver
from legoesm.land.canopy import solver as S
from test_canopy_warm_start import _CFG, _bundle


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--lai", type=float, default=0.1)
    ap.add_argument("--day", action="store_true")
    a = ap.parse_args(argv)
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = "?"
    d = np.load(a.checkpoint)
    x = d["land_ml_canopy_x"]
    rows = np.where(np.all(np.isfinite(x), axis=1))[0]
    Ta = d["T"][rows, -1]
    q = np.maximum(d["trc_q_v"][rows, -1], 1e-6)
    ps = d["p_s"][rows]
    rho = ps / (constants.R_d * Ta)
    kw = dict(LAI=jnp.full(rows.size, a.lai), Ta=jnp.asarray(Ta), Tv_atm=jnp.asarray(Ta * (1 + (1 / constants.epsilon - 1) * q)),
              Ts_bc=jnp.asarray(Ta + 1.0), q_atm=jnp.asarray(q), Ca=jnp.full(rows.size, 373.5),
              La=jnp.asarray(0.8 * constants.sigma_sb * Ta**4), rhoa=jnp.asarray(rho),
              Ps=jnp.asarray(ps), ur=jnp.full(rows.size, 3.0))
    n = rows.size
    if a.day:
        kw.update(SZA=jnp.full(n, 80.0), fSun=jnp.full(n, 0.3), APAR_Sun=jnp.full(n, 40.0),
                  APAR_Sh=jnp.full(n, 10.0), ASW_Sun=jnp.full(n, 20.0 * a.lai),
                  ASW_Sh=jnp.full(n, 5.0 * a.lai), ASW_Soil=jnp.full(n, 30.0))
    else:
        z = jnp.zeros(n)
        kw.update(SZA=jnp.full(n, 95.0), fSun=z, APAR_Sun=z, APAR_Sh=z,
                  ASW_Sun=z, ASW_Sh=z, ASW_Soil=z)
    base = _bundle()
    bun = base._replace(**{k: v for k, v in kw.items()})
    bun = jax.tree.map(lambda v: jnp.broadcast_to(v, (n,)) if jnp.ndim(v) == 0 else v, bun)

    def resid(xx, b):
        return S._canopy_residual(xx, b, _CFG.LE_module, _CFG.stomatal_model,
                                  _CFG.le_cap_mode, _CFG.use_ta_for_photosynthesis,
                                  _CFG.rh_cap_smoothing_width,
                                  _CFG.zeta_cap_smoothing_width, _CFG.most_n_iters)
    old = make_implicit_newton_solver(resid, x_scale=S._LM_XSCALE, f_scale=S._LM_FSCALE,
                                      max_iters=_CFG.max_iters, rtol=S._LM_RTOL, atol=S._LM_ATOL)
    new = make_implicit_newton_solver(resid, x_scale=S._LM_XSCALE, f_scale=S._LM_FSCALE,
                                      max_iters=_CFG.max_iters, rtol=S._LM_RTOL, atol=S._LM_ATOL,
                                      n_sq_max=S._ROOT_NSQ_MAX, admissible=S.canopy_state_admissible)
    seeds = jnp.asarray(x[rows])
    cold = jnp.stack([bun.Ta, bun.Ta, 0.7 * bun.Ca, 0.7 * bun.Ca, bun.Ta, bun.q_atm], -1)
    adm_seed = np.asarray(S.canopy_state_admissible(seeds))
    xo, _, co, nso, *_ = jax.vmap(old)(seeds, bun)
    xn, _, cn, nsn, *_ = jax.vmap(new)(jnp.where(adm_seed[:, None], seeds, cold), bun)
    co, cn = np.asarray(co), np.asarray(cn)
    ao = np.asarray(S.canopy_state_admissible(xo))
    print(f"git {sha} | {a.checkpoint} | LAI {a.lai} {'day' if a.day else 'night'} | seeds {n}"
          f" | inadmissible seeds {int((~adm_seed).sum())}")
    print(f"OLD gate, raw seed:        converged {int(co.sum())}, of which inadmissible {int((co & ~ao).sum())},"
          f" n_sq > {S._ROOT_NSQ_MAX:g}: {int((co & (np.asarray(nso) > S._ROOT_NSQ_MAX)).sum())}")
    print(f"NEW gate, sanitised seed:  converged {int(cn.sum())}, NOT converged (held) {int((~cn).sum())}")
    bad = np.where(~cn)[0][:10]
    if bad.size:
        print("held rows (cell, Ta, seed Tf_Sun/Tf_Sh):",
              [(int(rows[i]), round(float(Ta[i]), 1), round(float(x[rows[i], 0]), 1),
                round(float(x[rows[i], 1]), 1)) for i in bad])


if __name__ == "__main__":
    main()
