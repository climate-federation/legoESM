"""Zero-step closure test: our K and mixing length against NEMO's OWN, same instant.

WHY THIS EXISTS.  Stage A2 (`compare_tendencies_nemo.py::run_stage_a2_mode_a`)
seeds the kernel with NEMO's `en` and takes ONE 3600 s step before comparing.
That leaves the measured excess (3.09x Arctic, 1.71x Antarctic calm, job
9403798) split between two causes it cannot separate:

    (i)  our MIXING LENGTH differs from NEMO's zmxlm, so K is wrong before a
         single step is taken; or
    (ii) the lengths agree and our TKE EVOLVES differently over the step.

This probe removes the step entirely.  NEMO's restart carries, at ONE instant
(step 8760), `en`, `avm_k`, `avt_k`, `dissl` and the full `tn/sn/un/vn` state.
NEMO's coefficient is `avm = rn_ediff * zmxlm * sqrt(en)` (zdftke tke_avn), and
legoESM's is `K_M = c_k * l_k * sqrt(e)` with the same c_k = rn_ediff = 0.1.
Fed the SAME `en`, the K ratio is therefore EXACTLY the length ratio:

    K_M(ours) / avm_k(NEMO)  ==  l_k(ours) / zmxlm(NEMO)

so a single number decides (i) vs (ii).  No integration, no time alignment, no
Mode-B equilibrium, no target contamination from EVD or zdfiwm -- `avm_k` is
the momentum coefficient the TKE closure itself produced and stored.

CONTROLS, all asserted before any number is reported:
  C1  Recompute `c_k*l_k*sqrt(en)` by hand and require it to match the kernel's
      K_M wherever no floor or ceiling binds.  If that fails we are not reading
      the quantities we think we are, and every ratio below is meaningless.
  C2  Invert NEMO's own relation: zmxlm_implied = avm_k / (c_k*sqrt(en)).  The
      inversion is INVALID wherever `avm_k` sits at its floor `avmb` (NEMO
      applies MAX(zav, avmb)), so those interfaces are EXCLUDED and the
      excluded fraction is printed next to every ratio.
  C3  Cross-check the inversion against NEMO's stored `dissl`.  `dissl` is NOT
      a length: zdftke.F90:717 sets `dissl = SQRT(en) / zmxld`, so the length
      is `zmxld = sqrt(en)/dissl` (the first revision of this probe compared
      against `dissl` directly and its own C3 fired at a median ratio of 757 —
      the control worked, the assumption did not).  ORCA1 runs nn_mxl=2, where
      NEMO sets zmxld = zmxlm (zdftke.F90:680), so the two NEMO-side lengths
      must AGREE on the non-floored set.  That gives an independent second
      estimate valid EVERYWHERE dissl>0, not only off the avmb floor.
  C4  Everything is volume-weighted on the eORCA1 metrics and split by the same
      latitude bands the MLD scorecard uses, so the numbers sit next to it.

NOTE ON THE N-SQUARED INSTRUMENT.  The mixing length needs N2.  This mirrors
`compare_tendencies_nemo.py::run_stage_a` exactly -- in-situ density from
`nemo_seos_eos` at cell-centre pressure, then `n2_mode="insitu"` -- so the two
probes are the same instrument on this axis.  NEMO's own rn2 is NOT in the
restart; where the length is set by the buoyancy scale rather than the sweeps,
an N2 difference propagates, and that is called out in the output.

Usage (CPU, minutes; sbatch it per the login-node policy):
    python scripts/validate/ocean_fidelity/nemo_zero_step_closure.py \
        --restart-npz .../restart_8760_full.npz \
        --mesh-mask data/grids/eORCA1.2_mesh_mask.nc
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_HERE.parent))
for _p in ("ocean", "core"):
    sys.path.insert(0, str(_REPO / "packages" / _p))
sys.path.insert(0, str(_REPO))

from global_tracer_content import (  # noqa: E402
    load_mesh_latitude, load_mesh_metrics,
)

_BANDS = (
    ("antarctic_S_of_45S", -91.0, -45.0),
    ("SH_midlat_45S_23S", -45.0, -23.0),
    ("tropics_23S_23N", -23.0, 23.0),
    ("NH_midlat_23N_45N", 23.0, 45.0),
    ("arctic_N_of_45N", 45.0, 91.0),
)
_REQUIRED = ("tn", "sn", "en", "avm_k")


def _wmean(x, w):
    d = w.sum()
    if d <= 0:
        return float("nan")
    return float((x * w).sum() / d)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--restart-npz", required=True,
                   help="rebuild_nemo_restart.py output; MUST include "
                        "avm_k (and ideally dissl) as well as en/tn/sn.")
    p.add_argument("--mesh-mask", required=True)
    p.add_argument("--out-json", default=None)
    a = p.parse_args()

    import jax.numpy as jnp
    from legoesm import constants
    from legoesm.ocean.eos import nemo_seos_eos
    from legoesm.ocean.physics.vertical_mixing._shared import compute_N2
    from legoesm.ocean.physics.vertical_mixing.tke import (
        compute_K_from_tke, compute_mixing_lengths,
    )
    sys.path.insert(0, str(_REPO / "scripts" / "run"))
    from run_omip_core2 import orca1_zdftke_config

    rst = dict(np.load(a.restart_npz))
    missing = [k for k in _REQUIRED if k not in rst]
    if missing:
        raise SystemExit(
            f"FATAL: restart npz lacks {missing}. Re-run rebuild_nemo_restart.py "
            "with --fields tn sn un vn sshn en avm_k avt_k dissl -- the default "
            "field list drops the zdftke coefficients this probe needs.")

    cfg = orca1_zdftke_config()          # the production card (veros_sqrte)
    print(f"[card] c_k={cfg.c_k} kappa_convention={cfg.kappa_convention!r} "
          f"tke_mxl_choice={cfg.tke_mxl_choice} kappaM_min={cfg.kappaM_min:g}")

    e1t, e2t, e3t, tmask = load_mesh_metrics(a.mesh_mask)     # (nlev,nj,ni)
    lat = load_mesh_latitude(a.mesh_mask)
    T3, S3, en3 = rst["tn"], rst["sn"], rst["en"]
    avm3 = rst["avm_k"]
    z, ny, nx = T3.shape
    if (ny, nx) != lat.shape:
        raise SystemExit(f"FATAL: restart {(ny, nx)} vs mesh {lat.shape}")
    ncol = ny * nx

    def cols(x):
        return np.transpose(x.reshape(z, ncol), (1, 0))

    T_c = np.nan_to_num(cols(T3), nan=0.0)
    S_c = np.where(np.isfinite(cols(S3)) & (cols(S3) > 0), cols(S3), 35.0)
    dz_c = np.where(np.isfinite(cols(np.transpose(e3t, (0, 1, 2)))), 1.0, 1.0)
    dz_c = np.transpose(e3t.reshape(z, ncol), (1, 0)).astype(np.float64)
    dz_half = 0.5 * (dz_c[:, :-1] + dz_c[:, 1:])
    zc = np.cumsum(dz_c, axis=1) - 0.5 * dz_c
    rho = np.asarray(nemo_seos_eos(
        jnp.asarray(T_c), jnp.asarray(S_c),
        jnp.asarray(constants.rho_ocean * constants.g * zc)))
    N2 = np.asarray(compute_N2(jnp.asarray(rho), jnp.asarray(dz_half),
                               constants.rho_ocean, g=constants.g,
                               n2_mode="insitu"))
    # NEMO stores en / avm_k on W-levels (index 0 = surface); the legoESM
    # kernel works on the nlev-1 INTERIOR interfaces, i.e. levels 1..z-1.
    en_i = np.nan_to_num(cols(en3)[:, 1:], nan=0.0)
    avm_i = np.nan_to_num(cols(avm3)[:, 1:], nan=0.0)

    dz_ref_1d = np.nanmax(dz_c, axis=0)
    l_k, l_eps = compute_mixing_lengths(
        jnp.asarray(en_i), jnp.asarray(N2), jnp.asarray(dz_half), cfg,
        signed_n2=False, dz_cell=jnp.asarray(dz_c), boundary_cap=None,
        l_surface_anchor=jnp.full((ncol,), cfg.mxl0_min_m))
    l_k = np.asarray(l_k)
    K_M, K_H = compute_K_from_tke(
        jnp.asarray(en_i), jnp.asarray(l_k), cfg, N2=jnp.asarray(N2),
        shear_sq=jnp.zeros_like(jnp.asarray(N2)))
    K_M = np.asarray(K_M)

    # ---- C1: the kernel's K_M must BE c_k*l_k*sqrt(en) off the clamps ------
    hand = cfg.c_k * l_k * np.sqrt(np.maximum(en_i, 0.0))
    free = (hand > cfg.kappaM_min * 1.01) & (hand < cfg.kappaM_max * 0.99)
    if free.sum() == 0:
        raise SystemExit("FATAL C1: every interface is clamped; nothing to test.")
    rel = np.abs(K_M[free] - hand[free]) / np.maximum(hand[free], 1e-30)
    if float(rel.max()) > 1e-9:
        raise SystemExit(
            f"FATAL C1: kernel K_M departs from c_k*l_k*sqrt(en) by "
            f"{float(rel.max()):.3e} on unclamped interfaces -- the amplitude "
            "is not what this probe assumes and no ratio below is meaningful.")
    print(f"[C1] kernel K_M == c_k*l_k*sqrt(en) to {float(rel.max()):.2e} on "
          f"{int(free.sum())} unclamped interfaces — PASS")

    # ---- C2: invert NEMO, excluding its own avmb floor ---------------------
    avmb = float(np.nanmin(avm_i[avm_i > 0])) if (avm_i > 0).any() else 0.0
    sqrt_en = np.sqrt(np.maximum(en_i, 0.0))
    ok = (en_i > 0) & (avm_i > avmb * 1.01) & np.isfinite(avm_i)
    wet_c = np.transpose(tmask.reshape(z, ncol), (1, 0)) > 0.5
    wet_i = wet_c[:, :-1] & wet_c[:, 1:]
    ok &= wet_i
    zmxlm_nemo = np.where(ok, avm_i / np.maximum(cfg.c_k * sqrt_en, 1e-30),
                          np.nan)
    print(f"[C2] NEMO avmb floor detected at {avmb:.3e} m2/s; inversion valid "
          f"on {100.0 * ok.sum() / max(wet_i.sum(), 1):.1f}% of wet interfaces "
          f"({int(ok.sum())} of {int(wet_i.sum())})")

    # ---- C3: NEMO's OWN second length, zmxld = sqrt(en)/dissl --------------
    # dissl is sqrt(en)/zmxld (zdftke.F90:717), NOT a length. ORCA1's nn_mxl=2
    # sets zmxld = zmxlm (:680), so this is an independent estimate of the SAME
    # length the avm_k inversion gives -- and it is valid wherever dissl>0,
    # including the 59% of interfaces where avm_k sits on its avmb floor.
    zmxld_nemo = None
    if "dissl" in rst:
        dissl_i = np.nan_to_num(cols(rst["dissl"])[:, 1:], nan=0.0)
        good = wet_i & (dissl_i > 0) & (en_i > 0)
        zmxld_nemo = np.where(good, sqrt_en / np.maximum(dissl_i, 1e-30), np.nan)
        m = ok & good
        if m.any():
            r = zmxlm_nemo[m] / zmxld_nemo[m]
            med = float(np.median(r))
            print(f"[C3] zmxlm(from avm_k) / zmxld(from dissl): median "
                  f"{med:.4f} p10 {np.percentile(r, 10):.4f} "
                  f"p90 {np.percentile(r, 90):.4f} on {int(m.sum())} interfaces "
                  f"— nn_mxl=2 makes these the SAME length, so ~1 is required")
            if not (0.9 < med < 1.1):
                raise SystemExit(
                    f"FATAL C3: NEMO's two self-consistent lengths disagree by "
                    f"{med:.3f}x. Either the avm_k inversion, the dissl "
                    f"identity, or the nn_mxl=2 assumption is wrong. No length "
                    f"ratio below would be interpretable.")
        print(f"[C3] dissl-based length covers "
              f"{100.0 * good.sum() / max(wet_i.sum(), 1):.1f}% of wet "
              f"interfaces vs {100.0 * ok.sum() / max(wet_i.sum(), 1):.1f}% "
              f"for the avm_k inversion")
    else:
        print("[C3] SKIPPED: no `dissl` in the restart npz — rebuild with it "
              "to cross-check the inversion.")

    # ---- the measurement ---------------------------------------------------
    dV = np.transpose((e1t[None] * e2t[None] * e3t * (tmask > 0.5)
                       ).reshape(z, ncol), (1, 0))
    w_i = 0.5 * (dV[:, :-1] + dV[:, 1:])
    out = {"restart": a.restart_npz, "mesh_mask": a.mesh_mask,
           "card": {"c_k": cfg.c_k, "kappa_convention": cfg.kappa_convention,
                    "tke_mxl_choice": cfg.tke_mxl_choice},
           "avmb_detected": avmb, "bands": {}}
    try:
        out["git_sha"] = subprocess.check_output(
            ["git", "-C", str(_REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        out["git_sha"] = "unknown"

    l_eps = np.asarray(l_eps)
    print()
    print("%-22s %9s %11s %11s %8s %8s %8s %8s" % (
        "band", "n_iface", "K_M_ours", "NEMO_avm_k", "K_rat", "lk/zmxlm",
        "lk/zmxld", "leps/zd"))
    for bname, lo, hi in _BANDS:
        inb = ((lat >= lo) & (lat < hi)).reshape(ncol)
        m = ok & inb[:, None]
        if not m.any():
            continue
        ww = w_i[m]
        km, an = _wmean(K_M[m], ww), _wmean(avm_i[m], ww)
        lr = _wmean(l_k[m] / np.maximum(zmxlm_nemo[m], 1e-30), ww)
        row = {"n_interfaces": int(m.sum()), "K_M_ours": km,
               "nemo_avm_k": an,
               "K_ratio_volmean": km / an if an > 0 else float("nan"),
               "lk_over_zmxlm_inv_volmean": lr,
               "lk_over_zmxlm_inv_median": float(np.median(
                   l_k[m] / np.maximum(zmxlm_nemo[m], 1e-30)))}
        lrd = leps_r = float("nan")
        if zmxld_nemo is not None:
            md = wet_i & inb[:, None] & np.isfinite(zmxld_nemo)
            if md.any():
                wd = w_i[md]
                lrd = _wmean(l_k[md] / np.maximum(zmxld_nemo[md], 1e-30), wd)
                # our l_eps = sqrt(lup*ldn) (choice 3) vs NEMO nn_mxl=2's
                # zmxld = min(lup,ldn): this ratio IS the nn_mxl gap, measured
                # on NEMO's own state rather than on one of our snapshots.
                leps_r = _wmean(l_eps[md] / np.maximum(zmxld_nemo[md], 1e-30), wd)
                row["lk_over_zmxld_dissl_volmean"] = lrd
                row["leps_over_zmxld_dissl_volmean"] = leps_r
                row["n_interfaces_dissl"] = int(md.sum())
        out["bands"][bname] = row
        print("%-22s %9d %11.4e %11.4e %8.3f %8.3f %8.3f %8.3f" % (
            bname, m.sum(), km, an, km / an if an > 0 else float("nan"),
            lr, lrd, leps_r))
    print()
    print("READ: K_rat and lk/zmxlm must agree by construction (same en, same "
          "c_k) — printed separately as a redundancy check; a gap means a "
          "clamp binds inside the band. lk/zmxld uses NEMO's dissl-derived "
          "length, which covers the floored interfaces the inversion drops, "
          "so it is the broader-coverage number. leps/zd compares OUR "
          "dissipation length sqrt(lup*ldn) (tke_mxl_choice=3) against NEMO's "
          "nn_mxl=2 min(lup,ldn) — that ratio IS the nn_mxl gap, measured "
          "here on NEMO's own state.")

    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out_json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[json] {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
