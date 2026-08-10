"""#1455 dyn_vor EEN bisection: rerun row corr/ratio + mbkt-relative binning.

PROMOTED (#1492 evidence audit) from ``scripts/tmp/_probe_1455_een_vor_
bottom_bisect.py`` (gitignored, `_probe_*` convention) into this committed
canonical location -- this is the "dyn_vor EEN u"/"dyn_vor EEN v" gate rows'
own re-runnable instrument; PROVENANCE_SCRIPT previously cited only the
gitignored path. Logic UNCHANGED from the scripts/tmp/ version except for
this docstring + explicit dtype/time-level prints added below (task rule:
every restored probe must print fp64 explicit, dtypes, registry time level,
e3t mode, population/mask explicitly).

These rows are ALSO flagged STALE-SUSPECT with an UNATTRIBUTED drift cause
in fidelity_bar_gate.py -- this script records whatever the fresh numbers
are without forcing a match to the stored tuple (Rule 1e: reconcile, don't
silently overwrite).

Reproduces the recorded #1226 item-10 EEN numbers (u corr 0.999896/|x|ratio
1.00118, v corr 0.999932/1.000717) against the campaign's own oracle dump
(vor_dump_du.bin/vor_dump_dv.bin, dynvor.F90:151-155 vor_een Krhs increment,
NEMO time level "now" per the same registration as zu_frc_term_walk.py /
zu_frc_momentum_row_reconstruction.py / v_unification_timelevel_retest.py --
DINO_00057600_restart, RUN_GDB), then bins the per-element residual by
(mbkt - k) to localize the bottom-heavy bias precisely, per issue #1455.

Bridge/dump-reading conventions + model construction copied verbatim from
acc_momentum_budget.py (same restart, same recipe, same _load_full_3d
convention, same explicit LEGOESM_NEMO_E3T gate) -- no ad hoc oracle
re-derivation.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both \\
      .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/dyn_vor_een_bottom_bisect.py
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import dataclasses
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask, read_nemo_restart, read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (
    bridge_nemo_to_legoesm_topo, bridge_before_state_topo,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
    dino_lat_lon_surface_forcing_arrays, dino_step_surface_forcing,
)

RUN_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB"
RESTART_FILE = "DINO_00057600_restart.nc"


def _load_full_3d(path, jpi, jpj, jpkm1, hls):
    a = np.fromfile(path, dtype="<f8").reshape(jpkm1, jpj, jpi)
    if hls:
        a = a[:, hls:-hls, hls:-hls]
    return np.moveaxis(a, 0, -1)


def _u_to_nemo(a):
    return np.asarray(a)[:, 1:]


def corr_ratio(lego, nemo, mask):
    l = np.asarray(lego)[mask]
    n = np.asarray(nemo)[mask]
    c = float(np.corrcoef(l, n)[0, 1])
    r = float(np.sum(np.abs(l)) / np.sum(np.abs(n)))
    return c, r


def main() -> int:
    set_policy(PrecisionPolicy.fp64())
    e3t_mode = require_explicit_e3t_mode()
    assert e3t_mode == "both", (
        f"LEGOESM_NEMO_E3T={e3t_mode!r}: must be 'both' (NEMO's real e3t_0 "
        "ladder) for any oracle comparison -- see precision_gate docstring")
    print("=" * 78)
    print("#1455 dyn_vor EEN bisection -- rerun + mbkt-relative binning")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r}")
    print("=" * 78)

    g = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    s = read_nemo_restart(os.path.join(RUN_DIR, RESTART_FILE), nn_hls=0)
    print(f"dtypes: g.tmask={np.asarray(g.tmask).dtype}  s.T={np.asarray(s.T).dtype}  "
          f"g.e3t_0={np.asarray(g.e3t_0).dtype}")
    # TIME LEVEL: vor_dump_du.bin/vor_dump_dv.bin are dynvor.F90:151-155's
    # vor_een Krhs increment, dumped at NEMO time level "now" -- same
    # registration as zu_frc_term_walk.py/zu_frc_momentum_row_reconstruction.py/
    # v_unification_timelevel_retest.py (register_dump("vor_dump_du.bin", "now", ...)).
    # dyn_vor's own consumer (dynvor.F90 vor_een) reads velocity at Kmm (now),
    # matching this script's bridge_nemo_to_legoesm_topo (NOW-state bridge).
    print("time_level: vor_dump_du.bin/vor_dump_dv.bin = 'now' (dynvor.F90:151-155)")
    br = bridge_nemo_to_legoesm_topo(g, s, periodic_i=True, full_step=True)
    before = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART_FILE), nn_hls=0)
    st = bridge_before_state_topo(br._replace(state=br.state), g, before, periodic_i=True)

    cfg = dataclasses.replace(dino_config_for_recipe("nemo_dino_kamm_mlf"),
                              lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
    require_fp64(br.geometry, br.z_coord, st, context="1455 EEN bisect")

    model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
    forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)
    sf = dino_step_surface_forcing(forcing)
    if hasattr(st, "tau_x_prev"):
        st = st._replace(tau_x_prev=sf.tau_x, tau_y_prev=sf.tau_y)

    print(f"recipe=nemo_dino_kamm_mlf  vorticity_scheme={getattr(mc, 'vorticity_scheme', None)!r} "
          f"een_e3f_scheme={getattr(mc, 'een_e3f_scheme', None)!r} "
          f"een_q_boundary={getattr(mc, 'een_q_boundary', None)!r}")

    DT = 90.0  # DINO rn_Dt baroclinic step
    _tend, diag = model.tendencies_with_diagnostics(st, surface_forcing=sf, dt=DT)

    lego_u = np.asarray(diag.vortcor_u.data)
    lego_v = np.asarray(diag.vortcor_v.data)

    tmask_full = np.asarray(g.tmask) > 0.5   # (n_lat, n_lon, jpk=36) -- last level always dry
    n_lat, n_lon, jpk = tmask_full.shape
    jpkm1 = jpk - 1  # NEMO jpkm1=35: the dyn_vor dump only writes jk=1..jpkm1
    tmask = tmask_full[..., :jpkm1]
    # dump files carry the FULL model dims incl. nn_hls=2 halo (56 x 203 x 35);
    # mesh_mask/restart were read haloless (nn_hls=0) -> n_lat=199, n_lon=52.
    hls = 2
    jpi, jpj = n_lon + 2 * hls, n_lat + 2 * hls
    assert (jpi, jpj) == (56, 203), f"unexpected dims jpi={jpi} jpj={jpj} (n_lon={n_lon} n_lat={n_lat})"

    D_u = _load_full_3d(os.path.join(RUN_DIR, "vor_dump_du.bin"), jpi, jpj, jpkm1, hls)
    D_v = _load_full_3d(os.path.join(RUN_DIR, "vor_dump_dv.bin"), jpi, jpj, jpkm1, hls)

    umask3 = np.asarray(g.umask) > 0.5
    vmask3 = np.asarray(g.vmask) > 0.5
    lego_u_n = _u_to_nemo(lego_u)[..., :jpkm1]
    lego_v_n = np.asarray(lego_v)[1:, :, :jpkm1]  # NEMO v-point j = lego v-face (north) j+1

    mu = umask3[..., :jpkm1]
    mv = vmask3[..., :jpkm1]
    print(f"dtypes: lego_u_n={lego_u_n.dtype}  D_u={D_u.dtype}  mu={mu.dtype}")
    print(f"population: u_mask wet={int(mu.sum())}/{mu.size}  "
          f"v_mask wet={int(mv.sum())}/{mv.size}  (jpi={jpi} jpj={jpj} jpkm1={jpkm1})")

    cu, ru = corr_ratio(lego_u_n, D_u, mu)
    cv, rv = corr_ratio(lego_v_n, D_v, mv)
    print(f"\nRERUN  u: corr={cu:.6f}  |x|ratio={ru:.6f}   (recorded: 0.999896 / 1.00118)")
    print(f"RERUN  v: corr={cv:.6f}  |x|ratio={rv:.6f}   (recorded: 0.999932 / 1.000717)")

    # ---- mbkt-relative binning ---------------------------------------------
    mbkt = tmask.sum(axis=2).astype(np.int64)  # (n_lat,n_lon), 1-based deepest wet index
    e3t_0 = np.asarray(g.e3t_0)[..., :jpkm1]  # (n_lat,n_lon,jpkm1) reference thickness
    modal_e3t = np.array([np.median(e3t_0[..., k][tmask[..., k]]) if tmask[..., k].any() else np.nan
                           for k in range(jpkm1)])

    def bin_report(name, lego, nemo, mask3):
        print(f"\n--- {name}: residual binned by (mbkt - k), 0=bottommost wet cell ---")
        header = (f"{'d=mbkt-k':>8} {'n_pts':>8} {'median|resid/nemo|':>20} {'p90':>8} "
                   f"{'|x|ratio':>10} {'partial-cell frac':>18} {'topo-step frac':>15}")
        print(header)
        max_d = 6
        for d in range(0, max_d + 1):
            pts_l, pts_n, pts_partial, pts_step = [], [], [], []
            for j in range(mbkt.shape[0]):
                for i in range(mbkt.shape[1]):
                    kb = mbkt[j, i]
                    if kb <= 0:
                        continue
                    k = kb - 1 - d
                    if k < 0:
                        continue
                    if k >= mask3.shape[2] or j >= mask3.shape[0] or i >= mask3.shape[1]:
                        continue
                    if not mask3[j, i, k]:
                        continue
                    pts_l.append(lego[j, i, k])
                    pts_n.append(nemo[j, i, k])
                    is_partial = bool(e3t_0[j, i, k] < 0.999 * modal_e3t[k]) if not np.isnan(modal_e3t[k]) else False
                    pts_partial.append(is_partial)
                    neighs = [mbkt[max(j - 1, 0), i], mbkt[min(j + 1, mbkt.shape[0] - 1), i],
                              mbkt[j, max(i - 1, 0)], mbkt[j, min(i + 1, mbkt.shape[1] - 1)]]
                    pts_step.append(any(n != kb for n in neighs))
            if not pts_l:
                continue
            l = np.asarray(pts_l); n = np.asarray(pts_n)
            rel = np.abs(l - n) / np.maximum(np.abs(n), 1e-12)
            xr = np.sum(np.abs(l)) / max(np.sum(np.abs(n)), 1e-30)
            frac_partial = float(np.mean(pts_partial))
            frac_step = float(np.mean(pts_step))
            print(f"{d:>8} {len(pts_l):>8} {np.median(rel):>20.4e} {np.percentile(rel, 90):>8.3f} "
                  f"{xr:>10.4f} {frac_partial:>18.3f} {frac_step:>15.3f}")

    bin_report("u", lego_u_n, D_u, mu)
    bin_report("v", lego_v_n, D_v, mv)

    # Whole-column (not just near-bottom) topo-step vs non-step split, to test
    # whether the residual correlates with bathymetry steps AT ALL (not just
    # near mbkt).  step_col[j,i] = True if ANY horizontal T-neighbour has a
    # different mbkt (a genuine bathymetry step at that column).
    step_col = np.zeros_like(mbkt, dtype=bool)
    for j in range(mbkt.shape[0]):
        for i in range(mbkt.shape[1]):
            kb = mbkt[j, i]
            neighs = [mbkt[max(j - 1, 0), i], mbkt[min(j + 1, mbkt.shape[0] - 1), i],
                      mbkt[j, max(i - 1, 0)], mbkt[j, min(i + 1, mbkt.shape[1] - 1)]]
            step_col[j, i] = any(n != kb for n in neighs)

    def step_split(name, lego, nemo, mask3):
        step3 = np.broadcast_to(step_col[:, :, None], mask3.shape)
        for label, sel in (("topo-step cols", mask3 & step3), ("flat cols", mask3 & ~step3)):
            l = lego[sel]; n = nemo[sel]
            rel = np.abs(l - n) / np.maximum(np.abs(n), 1e-12)
            xr = np.sum(np.abs(l)) / max(np.sum(np.abs(n)), 1e-30)
            print(f"  {name} {label:>15}: n={sel.sum():>7} median_rel={np.median(rel):.4e} "
                  f"p90={np.percentile(rel,90):.3f} |x|ratio={xr:.4f}")

    print("\n--- whole-column topo-step vs flat split (all levels) ---")
    step_split("u", lego_u_n, D_u, mu)
    step_split("v", lego_v_n, D_v, mv)

    # Whole-domain per-absolute-level sanity: is there ANY residual depth
    # structure left (not just near mbkt), post-#1418?
    print("\n--- u: |x|ratio by ABSOLUTE level (sanity, not mbkt-relative) ---")
    for k in range(0, jpkm1, 5):
        m = mu[..., k]
        if not m.any():
            continue
        l = lego_u_n[..., k][m]; n = D_u[..., k][m]
        print(f"  k={k:>3} n={int(m.sum()):>6} |x|ratio={np.sum(np.abs(l))/max(np.sum(np.abs(n)),1e-30):.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
