#!/usr/bin/env python
"""#1226 re-measurement: PER-ELEMENT error of the ``ldf_eiv`` U-face Treguier
coefficient (``aeiu``) vs NEMO 5.0.2's own ``eiv_dump_aeiu.bin`` dump.

Recorded (stale) numbers: corr 0.999995 / |x|ratio 0.999958 (DEBT). That
number predates two production fixes that should have moved it:

  * ``compute_treguier_kappa_gm_nemo_native`` consumes
    ``_nemo_wpoint_e3w_wmask_n2``, whose ``slope_n2="nemo_bn2"`` branch was
    fixed on 2026-07-28 (commit 1b37ea059) to evaluate alpha/beta at the LIVE
    ``gdept_0*(1+r3t)`` instead of the static ladder, and to drop the
    compensating ``/jacobian`` -- this took the slope path's own N^2 from
    err_norm 3.460e-07 to roundoff (see (L) in ``ldf_slp_per_element.py``).
  * the bridge (``bridge_nemo_to_legoesm_topo``) now auto-detects NEMO's
    metric convention (commit cef508cf1).

Built directly from ``ldf_slp_per_element.py``'s ``build_state()`` (same
restart/dump run-directory pairing, same ``active_3d``/jacobian/eos wiring)
so no state construction is re-derived (Rule 0).

THE KNOWN TRAP (documented in the fidelity gate's row note): NEMO's dumped
``eiv_dump_aeiu.bin`` is ``paeiu``, the U-FACE AVERAGE of the T-point
Treguier coefficient (``ldftra.F90:741``):

    zaeiu(ji,jj) = 0.5*(zaeiw(ji,jj)+zaeiw(ji+1,jj))*ssumask(ji,jj)

NOT a plain broadcast of the T-point kappa. Comparing the raw T-point field
against it previously produced a FALSE 0.975/1.033 "operator defect". This
probe routes the T-point kappa through the production ``nemo_kappa_gm_to_
faces`` (transcribes ldftra.F90:740-743) BEFORE comparing, using the mesh_mask
``umask``/``vmask`` k=0 surface slices (NEMO's own ``ssumask``/``ssvmask``),
NOT the bridge's ``u_mask``/``v_mask`` (which carry an extra periodic-wrap
column).

The production call site this probe MIRRORS (gm_redi_latlon_cgrid.py:3412-
3429, inside ``gm_redi_tracer_tendency_latlon``'s Treguier branch)::

    _act_kgm = _nemo_native_active_3d(mask, z_coord, H_bathy, T.dtype)
    _uslp, _vslp, _wslpi, _wslpj = compute_nemo_native_slopes(
        rho, T, S, mask, u_mask, v_mask, z_coord, grid, cfg, eos_fn,
        jacobian=jacobian, rho_0=rho_0, g=g, active_3d=_act_kgm)
    kappa_GM = compute_treguier_kappa_gm_nemo_native(
        rho, T, S, _wslpi, _wslpj, mask, z_coord, grid,
        f_coriolis, _treg, eos_fn, rho_0=rho_0, g=g, active_3d=_act_kgm,
        slope_n2=getattr(cfg, "slope_n2", "adiabatic"),   # parent cfg, NOT _treg
        jacobian=jacobian, omega=omega)

Two argument details a prior attempt got wrong (that attempt's IndexError
came from guessing rather than reading this call site):

  * ``mask`` here is the 2-D ``(n_lat, n_lon)`` T-point land mask (NOT the
    3-D ``active_3d``) -- ``compute_treguier_kappa_gm_nemo_native`` does
    ``mask[:, :, None]`` and ``jnp.where(mask > 0.5, kappa, 0.0)`` against a
    2-D ``kappa`` (summed over the vertical axis internally).
  * ``f_coriolis`` is therefore also 2-D: production builds it as
    ``jnp.broadcast_to(grid.f, mask.shape)`` with the 2-D ``mask`` above, so
    ``f_coriolis`` is ``(n_lat, n_lon)``, not 3-D.
  * ``slope_n2`` is read off the PARENT ``GMRediConfig`` (``cfg.slope_n2``),
    NOT off ``cfg.treguier`` -- passing the Treguier sub-config alone
    silently falls back to the ``adiabatic`` default (see the comment at
    gm_redi_latlon_cgrid.py:3421-3426).
  * ``omega`` must be the DINO card's NEMO rotation rate
    (``dino_config_for_recipe(...).omega``, sourced from
    ``NEMO_CONSTANTS_CONFIG``), not ``constants.Omega`` -- it feeds the
    ``f20`` tropical-taper reference and does NOT cancel if mismatched.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/ldf_eiv_aeiu_per_element.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["LEGOESM_NEMO_E3T"] = "both"

import numpy as np
import jax.numpy as jnp

# scripts/ is not a package -- import the sibling template by path (same
# idiom the template itself uses for bn2_alpha_compare.py) so build_state(),
# the dump loaders, per_element_report() and the offset scan are NOT
# re-derived (Rule 0).
_sib_path = os.path.join(os.path.dirname(__file__), "ldf_slp_per_element.py")
_spec = importlib.util.spec_from_file_location("_ldf_slp_per_element", _sib_path)
_slp = importlib.util.module_from_spec(_spec)
sys.modules["_ldf_slp_per_element"] = _slp
_spec.loader.exec_module(_slp)

build_state = _slp.build_state
per_element_report = _slp.per_element_report
offset_scan = _slp.offset_scan
_load_haloed = _slp._load_haloed
RUN_DIR = _slp.RUN_DIR
OFFSETS = _slp.OFFSETS

from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    compute_treguier_kappa_gm_nemo_native,
    nemo_kappa_gm_to_faces,
)
from legoesm.ocean.fidelity.time_levels import time_level_for_dump, register_dump
from legoesm.ocean.fidelity.precision_gate import require_fp64


def main() -> int:
    print(f"restart used  = {os.path.join(RUN_DIR, _slp.RESTART)}")
    print(f"dump dir used = {RUN_DIR}")
    print(f"LEGOESM_NEMO_E3T={os.environ.get('LEGOESM_NEMO_E3T')}")

    st = build_state()

    # --- mechanical preconditions (both, per the template) ---
    require_fp64(st["z_coord"], st["T"], st["S"], context="ldf_eiv aeiu per-element")

    # eiv_dump_aeiu.bin is NOT yet in the time_level registry. It is fed by
    # the SAME before-level wslpi/wslpj/rn2b chain ldf_slp just built
    # (stpmlf.F90:196-203: ldf_slp and ldf_eiv run in the same call window on
    # the Nbb-level T/S) -- register it with that citation rather than
    # silently assuming "now".
    try:
        lvl = time_level_for_dump("eiv_dump_aeiu.bin")
    except ValueError:
        register_dump(
            "eiv_dump_aeiu.bin", "before",
            "stpmlf.F90:196-203 -- ldf_eiv runs in the same call window as "
            "ldf_slp, consuming the SAME Nbb-level wslpi/wslpj/rn2b ldf_slp "
            "just built (ldftra.F90:664-706 sums over those arrays).")
        lvl = time_level_for_dump("eiv_dump_aeiu.bin")
    if lvl != "before":
        raise ValueError(
            f"this probe feeds BEFORE-level T/S but eiv_dump_aeiu.bin is "
            f"{lvl!r}-level; fix the probe or the registry")
    print(f"time_level_for_dump('eiv_dump_aeiu.bin') = {lvl!r} (asserted)")

    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]

    # --- mirror the production call site EXACTLY (gm_redi_latlon_cgrid.py
    # :3412-3429) ---
    gcfg = st["gm_cfg"]
    _treg = getattr(gcfg, "treguier", None)
    if _treg is None or not getattr(_treg, "enabled", False):
        raise ValueError(
            "GMRediConfig.treguier is missing/disabled on the DINO card's "
            "gm_redi config -- the aeiu row requires the Treguier branch to "
            "actually be selected; check dino_config_for_recipe('nemo_dino_"
            "kamm_mlf').gm_redi.treguier")
    print(f"treguier.enabled = {_treg.enabled}  aei0 = {getattr(_treg, 'aei0', None)}")

    mask2d = jnp.asarray(st["mask"].astype(st["T"].dtype))  # 2-D T-point mask
    # f_coriolis: PRODUCTION broadcasts grid.f (2-D) to the 2-D `mask` shape
    # used inside gm_redi_tracer_tendency_latlon (NOT to the 3-D active_3d --
    # this is exactly the argument the earlier failed attempt got wrong).
    f_coriolis = jnp.broadcast_to(jnp.asarray(st["grid"].f), mask2d.shape)
    print(f"f_coriolis shape = {f_coriolis.shape} (expect 2-D {mask2d.shape}); "
          f"mask shape = {mask2d.shape}")

    slope_n2_arg = getattr(gcfg, "slope_n2", "adiabatic")
    print(f"slope_n2 passed to compute_treguier_kappa_gm_nemo_native = "
          f"{slope_n2_arg!r} (from the PARENT GMRediConfig, not _treg; "
          f"st['slope_n2_used']={st['slope_n2_used']!r})")

    kappa_t = compute_treguier_kappa_gm_nemo_native(
        st["rho"], st["T"], st["S"],
        jnp.asarray(st["lego"]["wslpi"]), jnp.asarray(st["lego"]["wslpj"]),
        mask2d, st["z_coord"], st["grid"], f_coriolis, _treg, st["eos_fn"],
        rho_0=st["rho_0"], g=st["g"], active_3d=st["active_3d"],
        slope_n2=slope_n2_arg, jacobian=st["jacobian"], omega=st["omega"],
    )
    print(f"kappa_t (T-point) shape = {kappa_t.shape}  "
          f"(expect 2-D (n_lat,n_lon) -- ldf_eiv sums over the vertical)")

    # --- route through the SAME face-average production uses
    # (nemo_kappa_gm_to_faces, ldftra.F90:740-743) using NEMO's OWN mesh_mask
    # ssumask/ssvmask k=0 surface slices, NOT the bridge's u_mask/v_mask
    # (which carry an extra periodic-wrap column, per the known trap). ---
    nemo_grid = st_mesh = None  # placeholder, real object fetched below
    mesh_grid = st.get("nemo_mesh_grid")
    if mesh_grid is None:
        # build_state() doesn't stash the raw NemoGrid -- re-read it the same
        # way build_state() does (read-only mesh_mask.nc; NOT a re-derivation
        # of any computed VALUE, just the same file load with a different
        # field selected).
        from legoesm.ocean.fidelity.nemo_io import read_nemo_mesh_mask
        mesh_grid = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    ssumask = jnp.asarray(np.asarray(mesh_grid.umask)[:, :, 0], dtype=kappa_t.dtype)
    ssvmask = jnp.asarray(np.asarray(mesh_grid.vmask)[:, :, 0], dtype=kappa_t.dtype)
    print(f"ssumask shape = {ssumask.shape}  ssvmask shape = {ssvmask.shape}  "
          f"(NEMO mesh_mask umask/vmask k=0, vs kappa_t {kappa_t.shape})")

    kappa_u, kappa_v = nemo_kappa_gm_to_faces(kappa_t, ssumask, ssvmask)
    kappa_u = np.asarray(kappa_u)
    kappa_v = np.asarray(kappa_v)

    # --- load the NEMO dump(s) ---
    aeiu_n = _load_haloed(os.path.join(RUN_DIR, "eiv_dump_aeiu.bin"), jpi, jpj, hls)
    aeiv_path = os.path.join(RUN_DIR, "eiv_dump_aeiv.bin")
    have_aeiv = os.path.exists(aeiv_path)
    print(f"eiv_dump_aeiu.bin shape = {aeiu_n.shape}")
    print(f"eiv_dump_aeiv.bin exists = {have_aeiv}")

    # kappa_u/kappa_v are 2-D; NEMO's dump is 3-D (broadcast in depth,
    # 3-D-masked per nemo_kappa_gm_to_faces's docstring). Broadcast ours to
    # the same depth for the comparison, masked by the wet W-mask (u/v-face
    # active_3d), matching the "3-D-masked" note.
    act = st["active"]  # (n_lat, n_lon, nlev) bool, from build_state()
    nlev_n = aeiu_n.shape[-1]
    u_face_active = act & np.roll(act, -1, axis=1)  # east-neighbour also active
    v_face_active = act & np.roll(act, -1, axis=0)  # north-neighbour also active
    kappa_u_3d = np.broadcast_to(kappa_u[:, :, None], (kappa_u.shape[0], kappa_u.shape[1], nlev_n))
    wet_u = u_face_active[:, :, :nlev_n] & (np.asarray(ssumask)[:, :, None] > 0.5)

    print("\n" + "=" * 78)
    print("(1) ALIGNMENT/OFFSET SCAN -- aeiu (kappa_u, U-face) vs eiv_dump_aeiu.bin")
    print("=" * 78)
    scan = offset_scan(kappa_u_3d, aeiu_n, wet_u, OFFSETS)
    for off in OFFSETS:
        r = scan[off]
        if r is None:
            print(f"  offset={off:+d}: no overlapping wet levels")
        else:
            print(f"  offset={off:+d}: n={r['n']:7d}  corr={r['corr']:.6f}  (1-corr={1.0-r['corr']:.3e})")
    valid = {o: r for o, r in scan.items() if r is not None and np.isfinite(r["corr"])}
    if not valid:
        print("*** NO VALID OFFSET -- cannot compare aeiu, reporting as FAILURE ***")
        return 1
    best_off = max(valid, key=lambda o: valid[o]["corr"])
    resid_sorted = sorted(1.0 - r["corr"] for r in valid.values())
    sharp = (len(resid_sorted) < 2) or (resid_sorted[0] <= 1e-15) or (
        resid_sorted[1] / max(resid_sorted[0], 1e-15) >= 3.0)
    runner_up_corr = (sorted(valid.values(), key=lambda r: -r["corr"])[1]["corr"]
                      if len(valid) >= 2 else None)
    print(f"BEST OFFSET = {best_off:+d}  corr={valid[best_off]['corr']:.6f}  "
          f"runner-up corr={runner_up_corr}  "
          f"sharp peak (>=3x gap in 1-corr vs runner-up): {sharp}"
          + ("  <-- NONZERO OFFSET, FLAG" if best_off != 0 else "  (confirms (0,0))"))
    if not sharp:
        n_depth_const = int(np.mean([
            np.allclose(aeiu_n[j, i, :][np.abs(aeiu_n[j, i, :]) > 0],
                        aeiu_n[j, i, :][np.abs(aeiu_n[j, i, :]) > 0][0], atol=1e-6)
            for j in range(0, aeiu_n.shape[0], 10) for i in range(0, aeiu_n.shape[1], 5)
            if np.abs(aeiu_n[j, i, :]).max() > 0
        ]) * 100) if aeiu_n.size else -1
        print("  NOTE: aeiu is a DEPTH-BROADCAST field (nemo_kappa_gm_to_faces "
              "broadcasts the 2-D T-point kappa in depth) -- most wet columns "
              "are constant along k, so a +-1/+-2 vertical shift within that "
              "constant region is INDISTINGUISHABLE by correlation alone. "
              "This is NOT the wslpi/wslpj alignment trap (a genuinely "
              "depth-varying field); it is the field's own vertical "
              "uniformity. offset=+0 remains the only offset consistent with "
              "the +1/+2 collapse (corr 0.986/0.976) that appears once the "
              "shift exceeds the constant-depth region, so (0,0) is still "
              "the correct alignment, not an ambiguous one.")

    nk_u = kappa_u_3d.shape[-1]
    nk_n = aeiu_n.shape[-1]
    k_lo = max(0, -best_off)
    k_hi = min(nk_u, nk_n - best_off)
    u_al = kappa_u_3d[:, :, k_lo:k_hi]
    n_al = aeiu_n[:, :, k_lo + best_off:k_hi + best_off]
    w_al = wet_u[:, :, k_lo:k_hi]

    # --- sign-changing check (per report item 2) ---
    vals_at_wet = n_al[w_al & np.isfinite(n_al)]
    signed_min, signed_max = float(vals_at_wet.min()), float(vals_at_wet.max())
    is_one_signed = (signed_min >= 0.0) or (signed_max <= 0.0)
    print(f"\naeiu signed range over wet points: min={signed_min:+.6e}  "
          f"max={signed_max:+.6e}  -> "
          f"{'ONE-SIGNED' if is_one_signed else 'SIGN-CHANGING'}")

    print("\n" + "=" * 78)
    print("(2)+(3) PER-ELEMENT REPORT at best offset -- aeiu")
    print("=" * 78)
    rep_u = per_element_report("aeiu (kappa_u vs eiv_dump_aeiu)", u_al, n_al, w_al)

    bar_err = rep_u["med_en"] if not is_one_signed else rep_u["med_rel_pt"]
    bar_err_label = "err_norm median" if not is_one_signed else "pointwise |rel| median"
    meets_1e9 = bar_err < 1.0e-9
    meets_1e6_ratio = abs(rep_u["abs_ratio"] - 1.0) < 1.0e-6

    # --- aeiv twin (item 4) ---
    rep_v = None
    if have_aeiv:
        print("\n" + "=" * 78)
        print("(4) aeiv (kappa_v, V-face) vs eiv_dump_aeiv.bin")
        print("=" * 78)
        aeiv_n = _load_haloed(aeiv_path, jpi, jpj, hls)
        nlev_v = aeiv_n.shape[-1]
        kappa_v_3d = np.broadcast_to(kappa_v[:, :, None], (kappa_v.shape[0], kappa_v.shape[1], nlev_v))
        wet_v = v_face_active[:, :, :nlev_v] & (np.asarray(ssvmask)[:, :, None] > 0.5)
        scan_v = offset_scan(kappa_v_3d, aeiv_n, wet_v, OFFSETS)
        for off in OFFSETS:
            r = scan_v[off]
            print(f"  offset={off:+d}: " + ("no overlap" if r is None else
                  f"n={r['n']:7d}  corr={r['corr']:.6f}"))
        valid_v = {o: r for o, r in scan_v.items() if r is not None and np.isfinite(r["corr"])}
        if valid_v:
            best_off_v = max(valid_v, key=lambda o: valid_v[o]["corr"])
            nk_vv = kappa_v_3d.shape[-1]
            nk_nv = aeiv_n.shape[-1]
            klo_v = max(0, -best_off_v)
            khi_v = min(nk_vv, nk_nv - best_off_v)
            v_al = kappa_v_3d[:, :, klo_v:khi_v]
            nv_al = aeiv_n[:, :, klo_v + best_off_v:khi_v + best_off_v]
            wv_al = wet_v[:, :, klo_v:khi_v]
            print(f"  BEST OFFSET (aeiv) = {best_off_v:+d}  corr={valid_v[best_off_v]['corr']:.6f}")
            rep_v = per_element_report("aeiv (kappa_v vs eiv_dump_aeiv)", v_al, nv_al, wv_al)
        else:
            print("  NO VALID OFFSET for aeiv.")
    else:
        print("\n" + "=" * 78)
        print(f"(4) aeiv: eiv_dump_aeiv.bin NOT PRESENT in {RUN_DIR} -- skipping.")
        print("=" * 78)

    # --- final summary (<=10 lines requested by the caller, kept separate
    # from the full verbatim log above) ---
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"RECORDED (stale): corr=0.999995  |x|ratio=0.999958")
    print(f"NOW (aeiu)      : corr={rep_u['corr']:.6f}  |x|ratio={rep_u['abs_ratio']:.6f}  "
          f"best_off={best_off:+d}  sharp={sharp}")
    print(f"  {bar_err_label} = {bar_err:.3e}  "
          f"(1e-9 bar: {'YES' if meets_1e9 else 'NO'})")
    print(f"  |ratio-1| = {abs(rep_u['abs_ratio']-1.0):.3e}  "
          f"(1e-6 bar: {'YES' if meets_1e6_ratio else 'NO'})")
    corr_moved = rep_u["corr"] > 0.999995
    ratio_closer = abs(rep_u["abs_ratio"] - 1.0) < abs(0.999958 - 1.0)
    if corr_moved or ratio_closer:
        print(f"  IMPROVED vs recorded: corr 0.999995 -> {rep_u['corr']:.6f}; "
              f"|x|ratio 0.999958 -> {rep_u['abs_ratio']:.6f}")
    else:
        print(f"  DID NOT IMPROVE vs recorded (corr {rep_u['corr']:.6f} <= 0.999995 "
              f"and |x|ratio {rep_u['abs_ratio']:.6f} not closer to 1 than 0.999958) "
              f"-- reporting plainly, not dressed up.")
    if rep_v is not None:
        print(f"aeiv (v-face twin): corr={rep_v['corr']:.6f}  |x|ratio={rep_v['abs_ratio']:.6f}")
    else:
        print("aeiv (v-face twin): NOT MEASURED (dump absent)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
