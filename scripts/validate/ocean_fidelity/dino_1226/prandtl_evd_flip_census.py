#!/usr/bin/env python
"""#1226 JOB 2 support: PURE-NEMO census of Prandtl-branch and EVD-trigger
flips between the two y20 RUN_TWIN_STEP1 restarts (kt=230400 -> 230401).

No legoESM state, no model run -- reads NEMO's OWN ``avm_k``/``avt_k`` from the
two consecutive restarts and counts, per w-interface cell:

  * EVD-fired cells (avm_k >= 0.95*rn_evd = 95, since rn_evd=100) at EACH step,
    and cells that FLIP EVD state between the two steps.
  * Prandtl-branch state: NEMO's TKE closure applies a Prandtl number
    pdlr in [0.1, 1] to get avt_k = pdlr * avm_k (zdftke.F90 nn_pdl=1).  The
    avt_k/avm_k RATIO is therefore pdlr; it saturates at 0.1 (strongly
    stratified) or 1.0 (unstratified).  Count cells where this ratio crosses
    the 0.1<->1.0 branch between the two steps (a "Prandtl flip").

The OUTPUT is the flip-cell (y,x,z) index sets, saved to an .npz, so the
carry_injection_discriminator's per-state dT argmax can be checked against
them (Rule: keep argmax metadata; "the argmax cells correlate with the flip
cells" must be a COUNTED overlap, not a vibe).

This is DIAGNOSTIC-ONLY tooling for the "if B_now wins" branch of JOB 2 and
touches no model code.  fp64 throughout (NEMO arrays are f64 on disk).

Run::
    cd /home/dbalwada/legoESM && \\
      .venv/bin/python scripts/validate/ocean_fidelity/dino_1226/prandtl_evd_flip_census.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_OCEAN_FIDELITY = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _SCRIPTS_OCEAN_FIDELITY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RUN_TWIN_STEP1 = os.environ.get(
    "DINO_NEMO_RUN_TWIN_STEP1",
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TWIN_STEP1")
KT0 = 230400
KT1 = 230401
RN_EVD = 100.0            # ocean.output rn_evd
EVD_THR = 0.95 * RN_EVD   # a cell at >=95 is EVD-pinned (closure Kz p99.9 ~0.75)
PDL_LO, PDL_HI = 0.1, 1.0  # Prandtl saturation bounds (zdftke nn_pdl)


def _rebuild_avmt(kt):
    from rebuild_nemo_restart import rebuild
    raw = rebuild(f"{RUN_TWIN_STEP1}/DINO_{kt:08d}_restart_*.nc",
                  ["avm_k", "avt_k"])
    # (z,y,x) -> (y,x,z), f64
    out = {}
    for n in ("avm_k", "avt_k"):
        a = np.asarray(raw[n], dtype=np.float64)
        out[n] = np.moveaxis(a, 0, -1) if a.ndim == 3 else a
    return out


def main() -> int:
    f0 = _rebuild_avmt(KT0)
    f1 = _rebuild_avmt(KT1)
    avm0, avt0 = f0["avm_k"], f0["avt_k"]
    avm1, avt1 = f1["avm_k"], f1["avt_k"]
    print(f"avm_k shape {avm0.shape} dtype {avm0.dtype}")
    # w-interface levels only (drop surface w-level 0, exactly as the
    # discriminator does with [...,1:] before injecting).
    def wlv(a):
        return a[..., 1:]
    avm0, avt0, avm1, avt1 = wlv(avm0), wlv(avt0), wlv(avm1), wlv(avt1)

    # wet = finite & positive avm at BOTH steps (dry columns carry 0/fill).
    wet = np.isfinite(avm0) & np.isfinite(avm1) & (avm0 > 0) & (avm1 > 0)
    nwet = int(wet.sum())
    print(f"wet w-interface cells (both steps): {nwet}")
    if nwet == 0:
        raise SystemExit("no wet cells -- restart read is broken")

    # --- EVD census -------------------------------------------------------
    evd0 = (avm0 >= EVD_THR) & wet
    evd1 = (avm1 >= EVD_THR) & wet
    evd_flip = (evd0 ^ evd1) & wet
    print(f"\n[EVD] fired kt0={int(evd0.sum())}  kt1={int(evd1.sum())}  "
          f"FLIP(kt0^kt1)={int(evd_flip.sum())}  "
          f"({100*evd_flip.sum()/nwet:.4f}% of wet)")

    # --- Prandtl census ---------------------------------------------------
    # pdlr = avt_k/avm_k, saturates at [0.1, 1.0].  A "flip" = the branch
    # (near-floor vs near-ceiling) changes between the two steps.  Use a mid
    # split at 0.55 with a small dead-band to avoid counting numerical
    # chatter as a flip.
    with np.errstate(divide="ignore", invalid="ignore"):
        pdl0 = np.where(wet, avt0 / avm0, np.nan)
        pdl1 = np.where(wet, avt1 / avm1, np.nan)
    # census of how binary the field is (task: "near-binary, floor 0.1 vs 1.0")
    near_lo0 = wet & (pdl0 < 0.2)
    near_hi0 = wet & (pdl0 > 0.9)
    mid0 = wet & ~near_lo0 & ~near_hi0
    print(f"[Prandtl] kt0 pdlr distribution over wet: "
          f"near-0.1 (<0.2)={int(near_lo0.sum())}  "
          f"mid=[0.2,0.9]={int(mid0.sum())}  "
          f"near-1.0 (>0.9)={int(near_hi0.sum())}")
    lo0 = wet & (pdl0 < 0.55)
    lo1 = wet & (pdl1 < 0.55)
    pdl_flip = (lo0 ^ lo1) & wet
    # exclude dead-band chatter: require the ratio to move by >0.1 across 0.55
    moved = np.abs(np.where(wet, pdl1 - pdl0, 0.0)) > 0.1
    pdl_flip_strong = pdl_flip & moved
    print(f"[Prandtl] branch-flip (0.1<->1.0 across 0.55): "
          f"raw={int(pdl_flip.sum())}  strong(|dpdlr|>0.1)="
          f"{int(pdl_flip_strong.sum())}  "
          f"({100*pdl_flip_strong.sum()/nwet:.4f}% of wet)")

    # union of the two branch-flip mechanisms
    any_flip = (evd_flip | pdl_flip_strong) & wet
    print(f"\n[union] EVD-flip OR Prandtl-flip = {int(any_flip.sum())} cells "
          f"({100*any_flip.sum()/nwet:.4f}% of wet)")

    # --- per-level flip counts (where in the column do flips live?) -------
    nz = avm0.shape[-1]
    print(f"\n[per-level flip counts]  (0-based w-interface, [...,1:] already)")
    print(f"{'zlev':>5s} {'EVDflip':>9s} {'PDLflip':>9s} {'wet':>8s}")
    for z in range(min(nz, 20)):
        we = int(wet[..., z].sum())
        if we == 0:
            continue
        print(f"{z:5d} {int(evd_flip[...,z].sum()):9d} "
              f"{int(pdl_flip_strong[...,z].sum()):9d} {we:8d}")

    out = os.path.join(_THIS_DIR, "prandtl_evd_flip_census.npz")
    np.savez(out, evd_flip=evd_flip, pdl_flip_strong=pdl_flip_strong,
             any_flip=any_flip, wet=wet,
             pdl0=np.where(wet, pdl0, np.nan),
             pdl1=np.where(wet, pdl1, np.nan))
    print(f"\nsaved flip masks -> {out}")
    print("  overlay on carry_injection_discriminator dT argmax (y,x,z-1) "
          "to test the correlation claim.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
