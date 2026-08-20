#!/usr/bin/env python
"""#1226 SEQDUMP seam walk: NAME the intra-step seam that first creates the
barotropic/longitude-uniform eta(Naa) injection (target: deta max ~4.2e-3 m,
de3t ~5.4e-4 m, broad du; carry_injection_discriminator / join_seam_growth).

Two phases, each with its own planted control (skill Rule 3/1d/2).

PHASE A -- NEMO's OWN intra-step r3/ssh chain (pure NEMO arithmetic; no
  legoESM quantity enters, so there is NO staggering/units/time-level mapping
  to get wrong on one side). Walks the after-ssh commit through its three
  successive NEMO commit points, in NEMO execution order (stpmlf.F90):
    r3c_dump_r3t   (:244, 1st dom_qco_r3c, from ssh_nxt first-guess ssh)
      --[dyn_spg barotropic replacement + 2nd div_hor]-->
    seq_dump_r3t_aaa (:367, 2nd dom_qco_r3c, from post-barotropic ssh)
      --[ssh_atf Asselin filter]-->
    seq_dump_r3t_f (:442, from filtered ssh -> the NEXT step's geometry)
  r3t = ssh*r1_ht_0 (domqco.F90:160), so ssh = r3t*ht_0; the r3 chain IS the
  ssh commit chain. Reports each seam's own d(ssh) magnitude so the barotropic
  replacement's size is visible against the filter's.

PHASE B -- ONE legoESM step from the bridged y20 state, final eta compared
  against EACH NEMO seam (first-guess / post-barotropic / filtered). Which
  NEMO seam legoESM's committed eta matches BEST localises the injection: if
  lego matches r3t_aaa (post-barotropic) but the target deta is between
  r3t_aaa and r3t_f, the injection is at the filter; if lego sits off ALL
  three, the divergence is in the barotropic commit itself.

CONTROLS
  * PLANTED (Phase A): compare seq_dump_r3t_aaa_kt230401 against
    seq_dump_r3t_aaa_kt230402 (one step apart). The walker MUST flag this as
    a large change -- if it reports ~0 the loader is silently reading the same
    file / a constant.
  * DAY-0 bit-identity (Phase B): build_replay_ic already asserts
    max|dT|=max|d_eta|=0 vs the restart at kt=230400 (verify_day0_matches_
    restart); re-printed.
  * HALO/land convention: postlbc_tem land cells (tmask==0) must be 0/fill.

fp64 via run_fp64.py wrapper; LEGOESM_NEMO_E3T=both. dtypes printed.

RETRACTION POINTER (2026-08-19).  This file's premise cites the "~0.88 m^2/s
wall-concentrated transport bias" and the "~4.2e-3 m/step eta injection".  BOTH
WERE WIND-OFF ARTIFACTS and are RETRACTED: the probe that produced them ran with
surface_forcing=None on a card that threads the wind THROUGH model.step.  Wind-on
the transport residual is 5.99e-02 m^2/s and INTERIOR-peaked, and the eta
increment is 1.95e-04 m.  The associated claim that the transport diff "closes
the injection to 3 sig figs" is also retracted -- it compared against the wrong
reference.  See the HISTORY block in hu_avg_perface_diff.py before using any
number below.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_OCEAN_FIDELITY = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _SCRIPTS_OCEAN_FIDELITY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEQDUMP = os.environ.get(
    "DINO_NEMO_RUN_SEQDUMP",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_SEQDUMP_Y20_1R")

JPI, JPJ, JPK, HLS = 56, 203, 36, 2   # haloed 1-rank global, nn_hls=2
NI, NJ = JPI - 2 * HLS, JPJ - 2 * HLS  # 52 x 199 interior (== mesh_mask)

from legoesm.ocean.fidelity.time_levels import time_level_for_dump  # noqa: E402


def _load2d(base: str, kt: int) -> np.ndarray:
    """(jpk?,jpj,jpi) stream 2-D dump -> (nj,ni) interior. Registry lookup
    forces a time-level disposition (Rule 1d) before the array is used."""
    fn = f"{base}_kt{kt:08d}.bin"
    time_level_for_dump(fn)  # raises if unregistered
    path = os.path.join(SEQDUMP, fn)
    a = np.fromfile(path, dtype="<f8")
    if a.size != JPI * JPJ:
        raise SystemExit(f"{fn}: size {a.size} != {JPI*JPJ} (2-D haloed)")
    return a.reshape(JPJ, JPI)[HLS:-HLS, HLS:-HLS]


def _load3d(base: str, kt: int, *, strip_ring: bool = False) -> np.ndarray:
    """(nlev,jpj,jpi) -> (nlev,nj,ni). nlev inferred from size."""
    fn = f"{base}_kt{kt:08d}.bin"
    time_level_for_dump(fn)
    path = os.path.join(SEQDUMP, fn)
    a = np.fromfile(path, dtype="<f8")
    nlev = a.size // (JPI * JPJ)
    if nlev * JPI * JPJ != a.size:
        raise SystemExit(f"{fn}: size {a.size} not a whole # of {JPI}x{JPJ}")
    a = a.reshape(nlev, JPJ, JPI)[:, HLS:-HLS, HLS:-HLS]
    return a


def _ht0_tmask():
    import netCDF4 as nc
    d = nc.Dataset(os.path.join(SEQDUMP, "mesh_mask.nc"))
    e3t_0 = np.asarray(d.variables["e3t_0"][0], dtype=np.float64)   # (jpk,nj,ni)
    tmask = np.asarray(d.variables["tmask"][0], dtype=np.float64)
    ht_0 = (e3t_0 * tmask).sum(axis=0)   # (nj,ni) total rest column depth
    wet2 = tmask[0] > 0.5                # surface wet mask
    d.close()
    return ht_0, tmask, wet2


def _stats(d: np.ndarray, mask: np.ndarray, label: str) -> dict:
    m = np.broadcast_to(mask, d.shape)
    a = np.abs(d)
    sel = a[m]
    mx = float(sel.max()) if sel.size else 0.0
    idx = np.unravel_index(int(np.argmax(np.where(m, a, -np.inf))), a.shape)
    return {"label": label, "max": mx, "argmax": tuple(int(v) for v in idx),
            "p99.9": float(np.percentile(sel, 99.9)) if sel.size else 0.0,
            "p50": float(np.percentile(sel, 50.0)) if sel.size else 0.0,
            "n": int(sel.size)}


def _p(s):
    print(f"  {s['label']:<34s} max={s['max']:.4e} p99.9={s['p99.9']:.4e} "
          f"p50={s['p50']:.4e} argmax={s['argmax']} n={s['n']}")


def phase_a(kts):
    ht_0, tmask, wet2 = _ht0_tmask()
    print(f"\n=== PHASE A: NEMO's own after-ssh commit chain (r3*ht_0) ===")
    print(f"  ht_0 dtype={ht_0.dtype} shape={ht_0.shape}  wet cells={int(wet2.sum())}")

    # PLANTED CONTROL: two steps of r3t_aaa must differ a lot.
    c1 = _load2d("seq_dump_r3t_aaa", kts[0])
    c2 = _load2d("seq_dump_r3t_aaa", kts[1])
    dctrl = (c2 - c1) * ht_0
    sc = _stats(dctrl, wet2, f"PLANT r3t_aaa[{kts[1]}]-[{kts[0]}] (m)")
    _p(sc)
    if sc["max"] < 1e-9:
        raise SystemExit("PLANTED CONTROL FAILED: two steps of r3t_aaa are "
                         "identical -> loader is reading a constant/same file")
    print("  PLANT OK (steps differ) \n")

    # land-cell convention check on a 3-D postlbc dump
    tem = _load3d("seq_dump_postlbc_tem_aaa", kts[0])   # (35,nj,ni)
    dry = tmask[:tem.shape[0]] < 0.5
    n_dry_nonzero = int((np.abs(tem[dry]) > 0).sum())
    print(f"  HALO/land check: postlbc_tem dry cells nonzero = {n_dry_nonzero}"
          f" of {int(dry.sum())} (expect 0 if land is fill)\n")

    for kt in kts:
        r3t_fg = _load2d("r3c_dump_r3t", kt)          # first-guess (pre-spg)
        r3t_bt = _load2d("seq_dump_r3t_aaa", kt)      # post-barotropic
        r3t_f = _load2d("seq_dump_r3t_f", kt)        # filtered
        ssh_fg, ssh_bt, ssh_f = r3t_fg * ht_0, r3t_bt * ht_0, r3t_f * ht_0
        print(f"--- kt={kt} ---")
        _p(_stats(ssh_bt - ssh_fg, wet2,
                  "SEAM barotropic: ssh_aaa - ssh_firstguess (m)"))
        _p(_stats(ssh_f - ssh_bt, wet2,
                  "SEAM filter: ssh_f - ssh_aaa (m)"))
        _p(_stats(ssh_f - ssh_fg, wet2,
                  "  net: ssh_f - ssh_firstguess (m)"))
    return ht_0, wet2


def phase_b(kts, ht_0, wet2):
    """ONE legoESM step from bridged y20 state; final eta vs each NEMO seam."""
    import multistep_replay as mr
    from legoesm.core.precision import get_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel)
    from legoesm.ocean.experiments.dino import (
        apply_dino_lat_lon_surface_forcing,
        dino_lat_lon_model_config,
        dino_lat_lon_surface_forcing_arrays,
        dino_step_surface_forcing)

    if not mr.have_step1_artifacts():
        print("SKIP phase B: RUN_TWIN_STEP1 artifacts absent")
        return
    print(f"\n=== PHASE B: legoESM one step, eta vs NEMO seams ===")
    print(f"  control dtype = {get_policy().control}  "
          f"E3T={os.environ.get('LEGOESM_NEMO_E3T')!r}")

    DT = 2700.0
    import jax
    base = 230400
    model = forcing = None
    # RESET arm: re-bridge from NEMO's own restart at each step entry and take
    # ONE step, so the per-step injection carries ZERO inherited history
    # (task step 3: confirm the SAME seam owns it every step, same magnitude).
    per_step = []
    for jj, kt in enumerate(kts):
        mr.IC_STEP = base + jj
        jax.clear_caches()   # jitted-closure trap: force retrace per bridge
        g, br, cfg, st0 = mr.build_replay_ic()   # asserts day-0 bit-identity
        if model is None:
            print(f"  recipe surface_tendency_placement="
                  f"{getattr(cfg,'surface_tendency_placement',None)!r} "
                  f"barotropic_reconcile_target="
                  f"{getattr(cfg,'barotropic_reconcile_target',None)!r}")
            print(f"  eta dtype={st0.eta.data.dtype} u={st0.u.data.dtype}")
        mc, _ = dino_lat_lon_model_config(br.geometry, cfg)
        model = LatLonCGridOceanModel(br.geometry, br.z_coord, mc)
        forcing = dino_lat_lon_surface_forcing_arrays(br.geometry, cfg)

        # #1455 retraction fix: the card routes the WIND MOMENTUM through
        # model.step(surface_forcing=sf) (run_dino.py:665-670,763-767), NOT
        # through the analytic applicator. surface_forcing=None dropped the
        # wind and fabricated the 4.2e-3 m/step eta injection this walk
        # localises. Mirror production: sf iff wind_through_step.
        # DINO_SEAM_WIND=0 forces the wind OFF -- the Rule-1e continuity
        # control that reproduces the retracted (wind-off) measurement so the
        # wind-on flip is a single-variable change. Default (unset/"1") = the
        # faithful card behavior (wind on iff wind_through_step).
        _wind = bool(getattr(cfg, "wind_through_step", False))
        if os.environ.get("DINO_SEAM_WIND", "1") == "0":
            _wind = False
        sf_step = dino_step_surface_forcing(forcing) if _wind else None
        if jj == 0:
            _tlo = float(np.min(np.asarray(sf_step.tau_x))) if sf_step is not None else 0.0
            _thi = float(np.max(np.asarray(sf_step.tau_x))) if sf_step is not None else 0.0
            print(f"  FORCING: wind_through_step={_wind} "
                  f"surface_stress_implicit={getattr(cfg,'surface_stress_implicit',None)} "
                  f"tau_x[Pa] range=[{_tlo:.4f},{_thi:.4f}]")

        placement = getattr(cfg, "surface_tendency_placement", "applied_now")
        t_sec = (jj + 1) * DT
        if placement == "leapfrog_rhs":
            st, rate = apply_dino_lat_lon_surface_forcing(
                st0, forcing, br.z_coord, cfg, DT, t_seconds=t_sec,
                return_rate=True)
            st = model.step(st, DT, surface_forcing=sf_step,
                            external_tracer_rate=rate)
        else:
            st = apply_dino_lat_lon_surface_forcing(
                st0, forcing, br.z_coord, cfg, DT, t_seconds=t_sec)
            st = model.step(st, DT, surface_forcing=sf_step,
                            external_tracer_rate=None)
        eta_k = np.asarray(st.eta.data)
        ssh_bt_k = _load2d("seq_dump_r3t_aaa", kt) * ht_0
        gap = np.abs((eta_k - ssh_bt_k)[wet2])
        per_step.append((kt, float(gap.max()), float(np.percentile(gap, 99.9))))
        if jj == 0:
            st_first, st0_first, kt0 = st, st0, kt

    print("\n  DETERMINISM across steps (RESET arm, eta gap vs committed ssh):")
    for kt, mx, p in per_step:
        print(f"    kt={kt}: max={mx:.4e} m  p99.9={p:.4e} m")
    gaps = np.array([g[1] for g in per_step])
    print(f"    spread (max-min)/mean = "
          f"{(gaps.max()-gaps.min())/gaps.mean()*100:.2f}%  "
          f"-> {'STATE-INDEPENDENT' if (gaps.max()-gaps.min())/gaps.mean() < 0.05 else 'state-varying'}")

    # --- detailed seam localization on step 1 (kt=230401) ----------------
    st, st0, kt = st_first, st0_first, kt0
    eta = np.asarray(st.eta.data)   # (nj,ni) lego final committed eta
    ssh_fg = _load2d("r3c_dump_r3t", kt) * ht_0
    ssh_bt = _load2d("seq_dump_r3t_aaa", kt) * ht_0
    ssh_f = _load2d("seq_dump_r3t_f", kt) * ht_0
    # NEMO's restart sshn at kt (what run_replay compares against = the swap
    # result). Nnn after swap == filtered ssh, so restart sshn ~ ssh_f.
    ns = mr.nemo_now_state_at(kt)
    print(f"  lego eta shape={eta.shape}  NEMO sshn shape={ns.ssh.shape}")
    _p(_stats(eta - ssh_fg, wet2, "lego_eta - NEMO first-guess ssh (m)"))
    _p(_stats(eta - ssh_bt, wet2, "lego_eta - NEMO post-barotropic ssh (m)"))
    _p(_stats(eta - ssh_f, wet2, "lego_eta - NEMO filtered ssh (m)"))
    _p(_stats(eta - ns.ssh, wet2, "lego_eta - NEMO restart sshn (m)"))
    # sanity: NEMO's restart sshn should equal filtered ssh_f (both = Nnn swap)
    _p(_stats(ns.ssh - ssh_f, wet2, "  NEMO sshn - filtered ssh_f (must ~0)"))

    # --- LOCALIZE WITHIN THE SEAM (task step 4) --------------------------
    # eta_before is bit-identical at day 0 (verify_day0 asserted d_eta=0), so
    # the ssh INCREMENT (eta_after - eta_before) isolates the barotropic
    # transport source: ssh(Naa)-ssh(Nbb) = -2dt*div(H*u_bar). Comparing the
    # lego vs NEMO increment against the first-guess seam shows whether the
    # ~4.2e-3 m gap is BORN in the transport divergence (before any ssh
    # commit) or in a later commit. If the gap == the raw lego-NEMO ssh gap,
    # it is entirely upstream of the ssh chain.
    eta_b = np.asarray(st0.eta.data)                     # == NEMO sshb (day0)
    sshb = mr.nemo_before_state_at(kt).ssh               # NEMO Nbb ssh
    print(f"  [chk] lego eta_before vs NEMO sshb: "
          f"max={float(np.abs((eta_b - sshb)[wet2]).max()):.3e} (day0 -> ~0)")
    d_inc = (eta - eta_b) - (ssh_fg - sshb)              # increment divergence
    _p(_stats(d_inc, wet2, "INCREMENT: lego d_ssh - NEMO d_ssh_firstguess (m)"))

    # SPATIAL structure of the eta residual: WHERE (col census, not a zonal
    # metric -- the DINO domain has meridional walls, so a zonal-mean-removal
    # test is dominated by the boundary and is NOT a clean uniformity check).
    resid = eta - ssh_bt
    absr = np.where(wet2, np.abs(resid), 0.0)
    colmax = absr.max(axis=0)   # (ni,) max over latitude per longitude column
    order = np.argsort(colmax)[::-1]
    print("  eta-resid column census (x-index: colmax_m):  "
          + "  ".join(f"x{int(i)}={colmax[i]:.2e}" for i in order[:6]))
    print(f"    interior (x>=4) colmax max = {float(colmax[4:].max()):.3e}  "
          f"vs west-2-cols max = {float(colmax[:2].max()):.3e}")

    # du confirmation + VERTICAL structure (barotropic = depth-uniform?).
    u_face = np.asarray(st.u.data)[:, 1:, :]
    umask3 = None
    try:
        import netCDF4 as nc
        dd = nc.Dataset(os.path.join(SEQDUMP, "mesh_mask.nc"))
        umask3 = np.moveaxis(np.asarray(dd.variables["umask"][0]), 0, -1) > 0.5
        dd.close()
    except Exception as e:
        print(f"  (umask load failed: {e})")
    if umask3 is not None:
        du = u_face - ns.u
        _p(_stats(du, umask3, "du: lego u(Naa) - NEMO un (m/s)"))
        # depth-mean vs deviation-from-depth-mean of du at each column:
        duw = np.where(umask3, du, np.nan)
        with np.errstate(invalid="ignore"):
            dubar = np.nanmean(duw, axis=2, keepdims=True)   # (nj,ni,1)
            dudev = duw - dubar
            bt = float(np.nanmax(np.abs(dubar)))
            bc = float(np.nanmax(np.abs(dudev)))
        print(f"  du VERTICAL: |depth-mean|max={bt:.4e}  "
              f"|dev-from-depth-mean|max={bc:.4e}  "
              f"-> {'BAROTROPIC (depth-uniform)' if bt > 3*bc else 'has baroclinic part'}"
              f" (ratio {bt/max(bc,1e-30):.1f})")


def main(argv):
    kts = [230401, 230402, 230403, 230404]
    ht_0, wet2 = phase_a(kts)
    if "--phase-a-only" not in argv:
        phase_b(kts, ht_0, wet2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
