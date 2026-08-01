#!/usr/bin/env python
"""#1226 tier-2: ``bn2 (rn2b)`` + ``eos_rab alpha`` vs NEMO's own kt==nit000 dumps.

Measures the two gate rows that the z* depth-ladder fix moves.  NEMO evaluates
``rab_3d_t``/``bn2_t`` at the LIVE ``gdept(Kmm)``, which under ``key_qco``
(no ``key_isf``) is the PURE multiplicative stretch ``gdept_0*(1+r3t)``
(``domzgr_substitute.h90:139`` with ``Tisf -> Time()`` at ``:56``).  legoESM's
``nemo_bn2_depth_ladders`` returns the STATIC reference ladder; the production
call sites (``k_profiles.py``, ``enhanced_diffusion.py``) apply the ``*J``
stretch.  This script measures both the stretched and unstretched ladders
against the oracle so the fix's effect is visible, not asserted.

NEMO side (all written at ``kt == nit000``, i.e. the state of the restart the
run STARTED from -- so the dumps pair with ``cn_ocerst_in``, NOT the restart
the run wrote out):
  * ``eiv_dump_gdept.bin``  = ``gdept(ji,jj,jk,Kmm)``   (ldftra.F90:899)
  * ``dump_alpha_b.bin``    = ``rab_b(...,jp_tem)``     (ldftra.F90:900)
  * ``tke_dump_rn2b.bin``   = ``rn2b``                  (zdftke.F90:204)
``eiv_*``/``dump_*`` carry the full ``(jpj,jpi)`` halo; ``tke_*`` is interior.

Both ``rab_b`` and ``rn2b`` are BEFORE-level (``eos_rab``/``bn2`` are called on
``ts(:,:,:,:,Kbb)`` in ``step.F90``), so the matching legoESM input is the
restart's ``tb/sb/sshb`` -- the script scans BOTH time levels and reports which
aligns rather than assuming.

Run::

    .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/bn2_alpha_compare.py \
        --run-dir /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB
"""
from __future__ import annotations

import argparse
import os
import re

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")
# DINO is FULL-STEP (ln_zps_nam=.false.; ocean.output "partial steps l_zps = F"),
# so NEMO's gdept_0 is horizontally UNIFORM and a 1-D ladder represents it
# EXACTLY.  But mesh_mask carries TWO different 1-D ladders: e3t_1d (sums to
# 4506.375 m) and e3t_0 (deepest wet column exactly 4000.0 m); they diverge by
# up to 105 m at depth.  bridge_nemo_to_legoesm_topo DEFAULTS to the e3t_1d one
# ("off") for a documented dynamical reason, which is wrong for a geometry
# comparison -- pin the NEMO ladder here.  (Static t=0 comparison, so the
# documented multi-day instability of "both" does not apply.)
os.environ.setdefault("LEGOESM_NEMO_E3T", "both")

from legoesm.ocean.eos import (  # noqa: E402
    compute_buoyancy_frequency_nemo_bn2,
    nemo_bn2_depth_ladders,
    nemo_seos_alpha_beta,
)
from legoesm.ocean.fidelity.nemo_io import (  # noqa: E402
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import (  # noqa: E402
    bridge_nemo_to_legoesm_topo,
)
from legoesm.ocean.eos import nemo_bn2_live_ladders  # noqa: E402
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG  # noqa: E402


def _read_dims(run_dir: str) -> tuple[int, int, int]:
    with open(os.path.join(run_dir, "ocean.output")) as f:
        text = f.read()
    jpi = int(re.search(r"jpi\s*:\s*(\d+)", text).group(1))
    jpj = int(re.search(r"jpj\s*:\s*(\d+)", text).group(1))
    jpk = int(re.search(r"jpk\s*:\s*(\d+)", text).group(1))
    hls = int(re.search(r"nn_hls\s*=\s*(\d+)", text).group(1))
    return jpi, jpj, jpk, hls


def _load_haloed(path: str, jpi: int, jpj: int, hls: int) -> np.ndarray:
    """(nlev, jpj, jpi) stream dump -> (y, x, nlev) interior, vertical last."""
    a = np.fromfile(path, dtype="<f8")
    nlev = a.size // (jpi * jpj)
    a = a.reshape(nlev, jpj, jpi)
    if hls:
        a = a[:, hls:-hls, hls:-hls]
    return np.moveaxis(a, 0, -1)


def _load_interior(path: str, ni: int, nj: int) -> np.ndarray:
    a = np.fromfile(path, dtype="<f8")
    nlev = a.size // (ni * nj)
    return np.moveaxis(a.reshape(nlev, nj, ni), 0, -1)


def _report(name: str, lego: np.ndarray, nemo: np.ndarray,
            mask: np.ndarray) -> tuple[float, float]:
    """Gate convention (matches ldftra_ahtv_compare.py): ratio = mean(lego/nemo)
    over the wet mask, corr = Pearson on the same masked points."""
    m = mask & np.isfinite(lego) & np.isfinite(nemo) & (np.abs(nemo) > 0)
    lo, ne = lego[m], nemo[m]
    ratio = lo / ne
    corr = float(np.corrcoef(lo, ne)[0, 1])
    ratio_mean = float(ratio.mean())
    print(f"  {name:<52s} corr={corr:.10f}  ratio_mean={ratio_mean:.10f}  "
          f"ratio_med={float(np.median(ratio)):.10f}  "
          f"max|rel|={float(np.max(np.abs(ratio - 1.0))):.3e}  n={int(m.sum())}")
    return corr, ratio_mean


def _shift_scan(name: str, lego: np.ndarray, nemo: np.ndarray,
                mask: np.ndarray) -> None:
    """Mandatory alignment scan: a wrong index/halo offset can fake a residual."""
    best = None
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            for dk in (-1, 0, 1):
                L = np.roll(lego, (dj, di, dk), axis=(0, 1, 2))
                m = mask & np.isfinite(L) & np.isfinite(nemo) & (np.abs(nemo) > 0)
                if m.sum() < 100:
                    continue
                err = float(np.median(np.abs(L[m] / nemo[m] - 1.0)))
                if best is None or err < best[0]:
                    best = (err, dj, di, dk)
    print(f"  [align scan] {name}: best (dj,di,dk)={best[1:]} "
          f"median|rel|={best[0]:.3e}  (expect (0,0,0) -- nonzero = misalignment)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--restart", default="DINO_00057600_restart.nc")
    args = ap.parse_args()
    rd = args.run_dir

    jpi, jpj, jpk, hls = _read_dims(rd)
    print(f"NEMO dims jpi={jpi} jpj={jpj} jpk={jpk} nn_hls={hls}")

    g = read_nemo_mesh_mask(os.path.join(rd, "mesh_mask.nc"), nn_hls=0)
    s_now = read_nemo_restart(os.path.join(rd, args.restart), nn_hls=0)
    s_bef = read_nemo_restart_before(os.path.join(rd, args.restart), nn_hls=0)
    br = bridge_nemo_to_legoesm_topo(g, s_now, periodic_i=True)
    z = br.z_coord
    nj, ni, nk = g.tmask.shape
    tmask = g.tmask > 0.5
    print(f"lego bridged interior (y,x,z) = ({nj},{ni},{nk})")

    # NEMO's OWN ht_0 = sum_k(e3t_0*tmask) (domain.F90) -- the r3t denominator.
    ht_0 = (g.e3t_0 * g.tmask).sum(axis=-1)

    nemo_gdept = _load_haloed(os.path.join(rd, "eiv_dump_gdept.bin"), jpi, jpj, hls)
    nemo_alpha = _load_haloed(os.path.join(rd, "dump_alpha_b.bin"), jpi, jpj, hls)
    nemo_rn2b = _load_interior(os.path.join(rd, "tke_dump_rn2b.bin"), ni, nj)

    # --- NEMO-internal control: confirms the stretch formula + time level ---
    nk_g = nemo_gdept.shape[-1]
    for nm, ssh in (("sshn (Kmm=now)", s_now.ssh), ("sshb (before)", s_bef.ssh)):
        pred = g.gdept_0[..., :nk_g] * (1.0 + (ssh / np.where(ht_0 > 0, ht_0, 1))[..., None])
        m = tmask[..., :nk_g]
        rel = np.abs(pred[m] / nemo_gdept[m] - 1.0)
        print(f"[control] gdept_0*(1+ssh/ht_0) [{nm}] vs NEMO gdept(Kmm) dump: "
              f"median|rel|={np.median(rel):.3e} max={rel.max():.3e}")

    # CONTROL: legoESM's reference ladder must BE NEMO's gdept_0 (full-step ->
    # horizontally uniform, so a 1-D ladder is exact). A nonzero value here
    # means the harness is on the wrong e3t ladder and every number below is
    # contaminated.
    _t_ref_ctl = np.asarray(nemo_bn2_depth_ladders(z)[0])
    _gd0_col = g.gdept_0[..., :][tmask].reshape(-1)  # uniformity checked below
    _spread = max(float(np.ptp(g.gdept_0[..., k][tmask[..., k]]))
                  for k in range(nk) if tmask[..., k].any())
    _col = np.array([float(g.gdept_0[..., k][tmask[..., k]].mean())
                     if tmask[..., k].any() else np.nan for k in range(nk)])
    print(f"[control] gdept_0 horizontal spread = {_spread:.3e} m "
          f"(0 => full-step, 1-D ladder is exact)")
    print(f"[control] max|lego t_depth_ref - NEMO gdept_0| = "
          f"{float(np.nanmax(np.abs(_t_ref_ctl - _col))):.3e} m")

    results: dict[str, tuple[float, float]] = {}
    for lvl, T, S, ssh in (("now", s_now.T, s_now.S, s_now.ssh),
                           ("before", s_bef.T, s_bef.S, s_bef.ssh)):
        if ssh is None:
            continue
        T = np.asarray(T); S = np.asarray(S)
        static = nemo_bn2_depth_ladders(z)
        live = nemo_bn2_live_ladders(z, np.asarray(ssh), ht_0)
        for tag, (td, wd) in (("PRE-FIX static", static), ("POST-FIX live", live)):
            td = np.broadcast_to(np.asarray(td), (nj, ni, nk))
            wd = np.broadcast_to(np.asarray(wd), (nj, ni, nk - 1))
            a = np.asarray(nemo_seos_alpha_beta(T, S, td)[0])
            # NEMO grav = 9.80665 (phycst.F90:38), NOT legoesm.constants.g
            # = 9.80616 -- a 5.0e-5 scale on EVERY N^2. The DINO recipe pins
            # this via NEMO_CONSTANTS_CONFIG; the harness must too.
            n2 = np.asarray(compute_buoyancy_frequency_nemo_bn2(
                T, S, td, wd, g=NEMO_CONSTANTS_CONFIG.g))
            nka = nemo_alpha.shape[-1]
            print(f"\n=== [{lvl}] {tag} ===")
            c, r = _report("alpha vs rab_b", a[..., :nka], nemo_alpha,
                           tmask[..., :nka])
            results[f"alpha/{lvl}/{tag}"] = (c, r)
            c, r = _report("bn2 vs rn2b[k+1]", n2, nemo_rn2b[..., 1:nk],
                           tmask[..., 1:nk])
            results[f"bn2/{lvl}/{tag}"] = (c, r)
            if tag.startswith("POST") and lvl == "before":
                _shift_scan("alpha", a[..., :nka], nemo_alpha, tmask[..., :nka])
                _shift_scan("bn2", n2, nemo_rn2b[..., 1:nk], tmask[..., 1:nk])

    print("\n" + "=" * 78)
    print("GATE CANDIDATES over ALL wet cells "
          "(bar: corr>=1-1e-9, |ratio-1|<=1e-6)")
    for k, (c, r) in sorted(results.items()):
        at = c >= 1.0 - 1e-9 and abs(r - 1.0) <= 1e-6
        print(f"  {k:<38s} corr={c:.10f} ratio={r:.10f} "
              f"{'AT BAR' if at else 'off-bar'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
