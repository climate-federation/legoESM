#!/usr/bin/env python3
"""Oracle gate: Morrison ``warm_rain_incloud`` vs a transcription of CAM6 MG2,
on real model columns.

Whole columns (so the precip-fraction inheritance runs over real vertical
structure) are read from a checkpoint: T, q_v, q_c, q_r, q_i, rho from the
hybrid levels, and the CLUBB cloud fraction carry (``physstate_cloud_fraction``)
as MG2's lcldm.  The applied autoconversion / accretion of
``morrison_microphysics(..., cloud_fraction=cf)`` (``publish_qc_budget``, a
1e-3 s step so the donor clamp cannot bind) must equal the MG2 in-cloud
transcription in ``tests/unit/test_morrison_warm_rain_incloud.py``
(``mg2_oracle``) to rtol 1e-12, for both warm-rain laws and several specified
droplet numbers (production droplet number is specified, not stored).  Also
reported: cf=1 vs the switch-off path (must be bitwise on real states within
the documented domain), and the grid-mean/in-cloud rate ratio.

    mg2_incloud_oracle.py <checkpoint.npz> [--ncol 64]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

NC_CM3 = (30.0, 100.0, 300.0)


def load_columns(path, ncol):
    from legoesm import constants
    z = np.load(path, allow_pickle=True)
    T = np.asarray(z["T"], dtype=np.float64)
    ps = np.asarray(z["p_s"], dtype=np.float64)
    vg = np.asarray(z["meta_vgrid"], dtype=np.float64)
    ph = vg[0][None, :] * constants.p_ref + vg[1][None, :] * ps[:, None]
    p = 0.5 * (ph[:, 1:] + ph[:, :-1])
    get = {k: np.asarray(z[k], dtype=np.float64)
           for k in ("trc_q_v", "trc_q_c", "trc_q_r", "trc_q_i",
                     "physstate_cloud_fraction")}
    qv = get["trc_q_v"]
    rho = p / (constants.R_d * T * (1.0 + (constants.R_v / constants.R_d - 1.0) * qv))
    for k, v in get.items():
        if v.shape != T.shape or not np.isfinite(v).all():
            raise SystemExit(f"FATAL: {k} shape {v.shape} vs {T.shape} or non-finite")
    cf = get["physstate_cloud_fraction"]
    if not ((cf >= 0) & (cf <= 1)).all():
        raise SystemExit(f"FATAL: cloud fraction outside [0,1]: {cf.min()}..{cf.max()}")
    # cloudiest-and-rainiest columns first, then a deterministic stride
    score = (get["trc_q_c"] > 1e-6).sum(1) + (get["trc_q_r"] > 1e-6).sum(1)
    order = np.argsort(-score, kind="stable")
    pick = np.unique(np.concatenate([order[: ncol // 2],
                                     np.linspace(0, T.shape[0] - 1, ncol - ncol // 2).astype(int)]))
    return tuple(x[pick] for x in (T, qv, get["trc_q_c"], get["trc_q_r"],
                                   get["trc_q_i"], cf, rho, p, ph))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--ncol", type=int, default=64)
    a = ap.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "tests" / "unit"))
    from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
    from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    from test_morrison_warm_rain_incloud import mg2_oracle
    sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"git {sha}; checkpoint {a.checkpoint}; ncol {a.ncol}")
    T, qv, qc, qr, qi, cf, rho, p, ph = load_columns(a.checkpoint, a.ncol)
    from legoesm import constants
    dz = (ph[:, 1:] - ph[:, :-1]) / (rho * constants.g)
    print(f"{T.shape[0]} columns x {T.shape[1]} levels; cloudy cells (qc>=1e-18) "
          f"{int((qc >= 1e-18).sum())}; cf in cloud median "
          f"{np.median(cf[qc >= 1e-6]):.2f}; rainy cells {int((qr >= 1e-18).sum())}")

    def run(cfg, cfv, nc, qv_in=qv):
        z = jnp.zeros(T.shape)
        f = dict(q_c=qc, q_r=qr, q_i=qi, N_r=np.full(T.shape, 1e5))
        hyd = HydrometeorState(**{k: jnp.asarray(f[k]) if k in f else z
                                  for k in HydrometeorState._fields})
        kw = {} if cfv is None else {"cloud_fraction": jnp.asarray(cfv)}
        return morrison_microphysics(
            jnp.asarray(T), jnp.asarray(qv_in), hyd, jnp.asarray(p), jnp.asarray(ph),
            jnp.asarray(rho), jnp.asarray(dz), 1.0e-3,
            MorrisonConfig(Nc_0=nc * 1e6, publish_qc_budget=True, **cfg), **kw)

    worst = 0.0
    # Two vapour states: the stored one, and 0.05 kg/kg (supersaturated
    # everywhere) so saturation adjustment adds no cloud-evaporation sink and
    # the clamp stops hiding cells.  Warm-rain kernels do not read q_v.
    cases = [(qn, qvi, sch, law, nc)
             for qn, qvi in (("real-qv", qv), ("supersat", np.full(T.shape, 0.05)))
             for sch, law in (("kk2000_cam6", "mg2"), ("kk2000", "sam"))
             for nc in NC_CM3]
    for qv_name, qv_in, scheme, law, nc in cases:
        b = run(dict(warm_rain_scheme=scheme, warm_rain_incloud=True), cf, nc,
                qv_in).qc_budget
        prc, pra, _ = mg2_oracle(qc, qr, qi, np.full(T.shape, nc * 1e6), cf, rho, law=law)
        # the q_c donor clamp scales every sink where their sum empties the
        # cell within the step (on real, sub-saturated cloud the
        # saturation-adjustment evaporation does): gate only cells where
        # it cannot bind, i.e. applied sinks x dt < q_c; count the rest.
        sinks = sum(np.minimum(np.asarray(v), 0.0) for v in b.values())
        # donor_clamp_scale = min(1, q_c / max(sink dt, 1e-15)): exactly 1
        # only if q_c >= 1e-15 as well
        free = ((-sinks * 1.0e-3 < np.clip(qc, 0, None) * (1 - 1e-9))
                & (qc >= 1e-15))
        ours = np.stack([-np.asarray(b["autoconversion"]), -np.asarray(b["accretion"])])[:, free]
        ref = np.stack([prc, pra])[:, free]
        clamped = int(((prc > 0) | (pra > 0))[~free].sum())
        gated = f"gated prc {int((prc > 0)[free].sum())} pra {int((pra > 0)[free].sum())}"
        if not np.isfinite(ours).all():
            raise SystemExit("FATAL: non-finite rate")
        rel = np.where(ref != 0, np.abs(ours - ref) / np.abs(np.where(ref != 0, ref, 1)),
                       np.abs(ours))
        worst = max(worst, float(rel.max()))
        g, g_a, _ = mg2_oracle(qc, qr, qi, np.full(T.shape, nc * 1e6),
                               np.ones_like(cf), rho, law=law)
        m = (qc >= 1e-6) & (cf > 0.01) & (cf < 0.99)
        print(f"{qv_name} {scheme:12s} Nc {nc:5.0f}/cm3: max rel diff prc {rel[0].max():.2e} "
              f"pra {rel[1].max():.2e}; active cells prc {int((prc > 0).sum())} "
              f"pra {int((pra > 0).sum())}, {gated}, clamp-bound {clamped}; "
              "grid-mean/in-cloud (oracle, 0.01<cf<0.99, qc>=1e-6) autoconv median "
              f"{np.median(g[m] / prc[m]):.3f}, accretion "
              f"{np.median(g_a[m & (pra > 0)] / pra[m & (pra > 0)]):.3f}")
    # cf = 1 bitwise vs the switch-off path, on columns inside the documented
    # domain (q_c in {0} U [1e-18, 5e-3], q_r in {0} U [1e-18, 0.01))
    out_dom = (((qc < 1e-18) & (qc != 0)) | (qc > 5e-3)
               | ((qr < 1e-18) & (qr != 0)) | (qr >= 0.01))
    keep = ~out_dom.any(1)
    bit = bool(keep.any())
    for scheme in ("kk2000", "kk2000_cam6"):
        on = run(dict(warm_rain_scheme=scheme, warm_rain_incloud=True), np.ones_like(cf), 100.0)
        off = run(dict(warm_rain_scheme=scheme), None, 100.0)
        for x, y in zip(jax.tree_util.tree_leaves(on), jax.tree_util.tree_leaves(off)):
            x, y = np.asarray(x), np.asarray(y)
            if x.ndim and x.shape[0] == T.shape[0]:
                x, y = x[keep], y[keep]
            bit &= bool(np.array_equal(x, y))
    print(f"cf=1 vs switch off: {'BITWISE' if bit else 'DIFFERS'} on {int(keep.sum())} of "
          f"{keep.size} columns (the rest hold q_c/q_r outside the documented domain)")
    ok = worst <= 1e-12 and bit
    print(("ORACLE PASS" if ok else "ORACLE FAIL")
          + f": worst relative difference {worst:.2e} (tol 1e-12)")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
