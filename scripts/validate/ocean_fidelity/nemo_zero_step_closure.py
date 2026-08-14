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

WHAT THE ZERO-STEP RATIO IS AND IS NOT (codex 9405117 #4).  It is a genuine
mismatch of the closure AS THIS PROBE RECONSTRUCTS IT.  It is NOT a clean
"our mixing length is 15% short" diagnosis, because the reconstruction differs
from the model's own tripole path in three named ways: N2 comes from
``n2_mode="insitu"`` rather than NEMO's ``rn2``; the stress-dependent NEMO
surface anchor is replaced by the windless ``mxl0_min_m`` floor (the snapshot
carries no taum at the restart instant); and the bottom row uses the legacy
``e3t`` proxy because NEMO's extra terminal slot is unavailable here.  Quote
the number as "the probe's reconstructed closure", never as the model's.

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
    load_mesh_depth_1d, load_mesh_latitude, load_mesh_metrics,
)


def _mesh_gdepw(mesh_mask_path):
    """gdepw_1d interface ladder — same accessor arctic_n2_compare.py uses."""
    import netCDF4 as nc
    ds = nc.Dataset(mesh_mask_path)
    try:
        return np.asarray(ds.variables["gdepw_1d"][:],
                          dtype=np.float64).squeeze()
    finally:
        ds.close()

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
    p.add_argument("--trd-tfile", default=None,
                   help="RUN_TRD2 hourly trd1h_T file. Enables the ONE-STEP "
                        "mode: seed with NEMO's en, take one 3600 s step, and "
                        "compare K_M against NEMO's `avm`. avm is the CLEAN "
                        "target — ORCA1 sets nn_evdm=0 so EVD never touches "
                        "momentum (namelist_cfg:429), and no Prandtl number "
                        "stands between the closure and avm. Every Stage-A "
                        "number to date used K_H vs avt, which carries both.")
    p.add_argument("--trd-rec", type=int, default=0,
                   help="record of --trd-tfile to score the one step against "
                        "(0 = the step the restart runs into).")
    p.add_argument("--cfg-override", action="append", default=[],
                   metavar="FIELD=VALUE",
                   help="repeatable TKEConfig override applied to the ORCA1 "
                        "card, for one-variable source-term ablation of the "
                        "one-step TKE growth (e.g. lc=False, etau_mode=none, "
                        "surface_bc=veros_flux, tke_mxl_choice=4). Values are "
                        "parsed as bool / int / float / str in that order. An "
                        "unknown FIELD raises rather than being ignored.")
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
    for spec in a.cfg_override:
        if "=" not in spec:
            raise SystemExit(f"--cfg-override {spec!r} must be FIELD=VALUE")
        k, _, raw = spec.partition("=")
        if k not in cfg._fields:
            raise SystemExit(
                f"--cfg-override {k!r} is not a TKEConfig field. Silently "
                "ignoring it would make an ablation look like a null result.")
        if raw in ("True", "False"):
            val = (raw == "True")
        else:
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw)
                except ValueError:
                    val = raw
        if getattr(cfg, k) == val:
            raise SystemExit(
                f"--cfg-override {k}={val!r} equals the card value, so this "
                "arm is a NO-OP and would be reported as 'no effect'.")
        cfg = cfg._replace(**{k: val})
        print(f"[cfg] OVERRIDE {k} -> {val!r}")
    print(f"[card] c_k={cfg.c_k} kappa_convention={cfg.kappa_convention!r} "
          f"tke_mxl_choice={cfg.tke_mxl_choice} kappaM_min={cfg.kappaM_min:g} "
          f"lc={cfg.lc} etau_mode={cfg.etau_mode!r} "
          f"surface_bc={cfg.surface_bc!r}")

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
    # REAL ln_mxl0 SURFACE ANCHOR (codex 9405117 #4 named the windless floor as
    # a reconstruction gap, and it is a big one). NEMO: zraug = vkarmn*2e5/
    # (rho0*grav); zmxlm(1) = zraug*taum, floored at rn_mxl0 (zdftke.F90:575,
    # 602) -- LINEAR in the stress modulus, not a square root. zraug is ~7.95,
    # so a 0.2 N/m2 wind gives ~1.6 m and a Southern Ocean storm ~2.8 m,
    # against the 0.04 m floor this probe used before. A too-short anchor
    # shortens lup near the surface and therefore UNDERSTATES our l_k -- which
    # is the direction of the 0.86 zero-step ratio, so the ratio may have been
    # measuring the probe rather than the model.
    _anchor = np.full((ncol,), cfg.mxl0_min_m)
    _anchor_src = f"windless floor rn_mxl0={cfg.mxl0_min_m:g} m"
    if a.trd_tfile:
        import netCDF4 as _nc
        _ds = _nc.Dataset(a.trd_tfile)
        try:
            _tm = np.ma.filled(np.ma.masked_invalid(
                _ds.variables["taum"][a.trd_rec]), 0.0).astype(np.float64)
        finally:
            _ds.close()
        _zraug = 0.4 * 2.0e5 / (constants.rho_ocean * constants.g)
        _anchor = np.maximum(cfg.mxl0_min_m,
                             _zraug * np.maximum(_tm.reshape(ncol), 0.0))
        _anchor_src = (f"NEMO zraug={_zraug:.3f} x taum  "
                       f"(median {np.median(_anchor):.3f} m, "
                       f"p90 {np.percentile(_anchor, 90):.3f} m)")
    print(f"[anchor] ln_mxl0 surface length: {_anchor_src}")
    l_k, l_eps = compute_mixing_lengths(
        jnp.asarray(en_i), jnp.asarray(N2), jnp.asarray(dz_half), cfg,
        signed_n2=False, dz_cell=jnp.asarray(dz_c), boundary_cap=None,
        l_surface_anchor=jnp.asarray(_anchor))
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

    # ---- ONE-STEP mode: where does the Stage-A2 excess actually appear? ----
    if a.trd_tfile:
        import netCDF4 as nc
        from legoesm.ocean.physics.vertical_mixing.tke import tke_vertical_mixing
        ds = nc.Dataset(a.trd_tfile)
        try:
            def _v(n):
                x = ds.variables[n][a.trd_rec]
                return np.ma.filled(np.ma.masked_invalid(x), np.nan).astype(
                    np.float64)
            avm_nemo = _v("avm")
            taum_n = _v("taum")
        finally:
            ds.close()
        if avm_nemo.shape != (z, ny, nx):
            raise SystemExit(f"FATAL: avm {avm_nemo.shape} vs {(z, ny, nx)}")
        avm_next = np.nan_to_num(cols(avm_nemo)[:, 1:], nan=0.0)
        # SET MATCHING (2026-08-13 retraction): the zero-step table scores on
        # `ok` (avm_k off its floor) and an earlier revision of this block
        # scored on avm_next off ITS floor -- 491743 vs 1192195 interfaces.
        # Comparing 0.861 (zero step) against 1.755 (one step) across those two
        # populations is not a comparison at all. Both tables now use the
        # INTERSECTION, and the zero-step K ratio is recomputed on it and
        # printed here so the two rows sit on identical ocean.
        taum_c = np.nan_to_num(taum_n.reshape(ncol), nan=0.0)
        u_c = np.nan_to_num(cols(rst["un"]), nan=0.0)
        v_c = np.nan_to_num(cols(rst["vn"]), nan=0.0)
        step = tke_vertical_mixing(
            jnp.asarray(u_c), jnp.asarray(v_c), jnp.asarray(T_c),
            jnp.asarray(S_c), jnp.asarray(rho), jnp.asarray(dz_half),
            tke_old=jnp.asarray(en_i),
            tau_x_surface=jnp.asarray(taum_c),
            tau_y_surface=jnp.zeros_like(jnp.asarray(taum_c)),
            taum_surface=jnp.asarray(taum_c),
            dt=3600.0, cfg=cfg, rho_0=constants.rho_ocean, g=constants.g,
            n_iterations=1,
            z_interface=jnp.asarray(-np.cumsum(dz_ref_1d)[:-1]),
            lat_deg=jnp.asarray(lat.reshape(ncol)), ice_frac=None,
            dz_ref=jnp.asarray(dz_ref_1d),
            jacobian=jnp.asarray(np.ones((ncol,), dtype=dz_c.dtype)),
            # Geometric depth ladders, required by n2_mode="nemo_bn2" and
            # ignored by the "insitu" default -- passed unconditionally so
            # every ablation arm stays ONE variable.
            t_depth=jnp.asarray(load_mesh_depth_1d(a.mesh_mask)),
            w_depth=jnp.asarray(_mesh_gdepw(a.mesh_mask)[1:]),
            # BEFORE velocities go in ONLY when the shear form consumes them.
            # tke_vertical_mixing has a silent-no-op guard that RAISES if they
            # are supplied under 'squared_centered' -- passing them
            # unconditionally (to keep arms one-variable) therefore broke every
            # arm at once, 12 failures. The guard is right and the probe was
            # wrong. A NEMO 5 RK3 restart carries no ub/vb (checked: only
            # un/vn); at a restart before == now, which is what NEMO starts
            # from, so the now x before product degenerates to the square.
            **({"u_before_cell": jnp.asarray(u_c),
                "v_before_cell": jnp.asarray(v_c)}
               if cfg.tke_shear_production in ("nemo_face_native",
                                               "nemo_burchard") else {}),
            # z=0-to-first-cell-centre distance, required by
            # tke_surface_bc_level="nemo_z0" (tke.py:1268) and ignored by the
            # "interior_pinned" default, so passing it unconditionally keeps
            # every arm on ONE variable.  Full top cell => gdept(1)=e3t(1)/2.
            dz_surface=jnp.asarray(0.5 * dz_c[:, 0]))
        e_after = np.asarray(step.tke_new)
        KM_after = np.asarray(step.K_M)
        okn = (ok & wet_i & (avm_next > avmb * 1.01) & np.isfinite(avm_next))
        print()
        print("ONE STEP (3600 s) from NEMO's en, scored against NEMO `avm` — "
              "the EVD-free, Prandtl-free target (nn_evdm=0). Scored on the "
              "INTERSECTION with the zero-step set, so K0_rat below is the "
              "zero-step ratio recomputed on these SAME interfaces and the "
              "two are directly comparable.")
        # TKE growth as a RATIO OF VOLUME-WEIGHTED MEANS, never a mean of
        # ratios.  The first revision printed <e_after/en> and returned
        # 9000-11700, which is not a measurement: NEMO floors `en` at
        # rn_emin=1e-6, so any interface sitting on that floor contributes a
        # ratio of order 1e4 and the mean is entirely those cells.  The
        # ratio-of-means is bounded by the fields themselves, and the
        # floor-excluded subset (en > 10*rn_emin) is reported beside it so the
        # reader can see how much of the domain is even eligible.
        emin = float(cfg.tke_background)
        print("%-22s %9s %8s %8s %10s %10s %8s %8s" % (
            "band", "n_iface", "e_grow", "e_gr_act", "KM_after",
            "NEMO_avm", "K0_rat", "KM_rat"))
        out["one_step"] = {"trd_tfile": a.trd_tfile, "trd_rec": a.trd_rec,
                           "rn_emin": emin, "bands": {}}
        for bname, lo, hi in _BANDS:
            inb = ((lat >= lo) & (lat < hi)).reshape(ncol)
            m = okn & inb[:, None] & (en_i > 0)
            if not m.any():
                continue
            ww = w_i[m]
            e_r_all = _wmean(e_after[m], ww) / max(_wmean(en_i[m], ww), 1e-30)
            act = m & (en_i > 10.0 * emin)
            if act.any():
                wa = w_i[act]
                e_r_act = _wmean(e_after[act], wa) / max(
                    _wmean(en_i[act], wa), 1e-30)
                f_act = float(w_i[act].sum() / max(ww.sum(), 1e-30))
            else:
                e_r_act, f_act = float("nan"), 0.0
            km, an = _wmean(KM_after[m], ww), _wmean(avm_next[m], ww)
            # zero-step ratio recomputed on THIS set, so the before/after pair
            # is a comparison rather than two different populations.
            k0 = _wmean(K_M[m], ww) / max(_wmean(avm_i[m], ww), 1e-30)
            out["one_step"]["bands"][bname] = {
                "n_interfaces": int(m.sum()),
                "our_tke_growth_all": e_r_all,
                "our_tke_growth_active": e_r_act,
                "active_volume_fraction": f_act,
                "K0_ratio_same_set": k0,
                "KM_after": km, "nemo_avm_next": an,
                "KM_ratio": km / an if an > 0 else float("nan")}
            print("%-22s %9d %8.3f %8.3f %10.4e %10.4e %8.3f %8.3f" % (
                bname, m.sum(), e_r_all, e_r_act, km, an, k0,
                km / an if an > 0 else float("nan")))
        print()
        print("READ, and note two labels that were WRONG in earlier revisions.")
        print(" * K0_rat -> KM_rat is the real before/after pair: both are "
              "ratios of volume-weighted means of OUR K_M to NEMO's own "
              "momentum diffusivity, on the SAME interfaces. What one step "
              "creates is the change between them.")
        print(" * e_grow / e_gr_act are OUR TKE after the step over the TKE we "
              "were HANDED (NEMO's en at the seed instant). There is no "
              "NEMO-after value in them, so they are a diagnostic of our own "
              "growth and NOT an error: NEMO's en evolves over the same hour "
              "too. An earlier revision reported them as if they were an "
              "error against NEMO. They are not.")
        print(f"   (e_gr_act restricts to en > 10x rn_emin={emin:g}; a mean of "
              "RATIOS was tried before that and is meaningless here — a "
              "floored denominator makes it read ~1e4 whatever the model does.)")

    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out_json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"[json] {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
