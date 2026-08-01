#!/usr/bin/env python
"""#1226 work-order #2: PER-LEVEL residual walk of the ``eiv transport`` u/v
rows (recorded corr/ratio 0.999617/0.996109 and 0.999103/0.990501) vs NEMO
5.0.2's own ``eiv_dump_u.bin``/``eiv_dump_v.bin`` dumps -- the actual eddy-
induced (GM bolus) TRANSPORT increment NEMO adds to the advecting velocity,
NOT the kappa_GM coefficient (that is the already-resolved ``ldf_eiv aeiu``
row; its hypothesis space, T3/T23/T6, is EXCLUDED here per the task brief).

Built directly on ``ldf_eiv_aeiu_per_element.py`` / ``ldf_slp_per_element.py``
(same ``build_state()``, same restart/dump pairing, same offset-scan/
per_element_report machinery) -- Rule 0, no re-derivation.

NEMO SOURCE (``cfgs/DINO/MY_SRC/ldftra.F90``, the DINO-active instrumented
copy -- MY_SRC overrides ``src/OCE/LDF/ldftra.F90`` per the standard
override rule), subroutine ``ldf_eiv_trp_MLF`` (the DINO-active MLF time-
stepping branch; DINO uses ``stpmlf.F90``, confirmed by the ocean.output
call trace and the ``eiv_dump_u.bin`` write only firing under
``cdtype=='TRA'`` inside that subroutine)::

    zpsi_uw(ji,jj,2) = -1/4 * e2u(ji,jj) * ( wslpi(ji,jj,jk+1) + wslpi(ji+1,jj,jk+1) )
                             * ( aeiu(ji,jj,jk) + aeiu(ji,jj,jk+1) ) * wumask(ji,jj,jk+1)
    zpsi_vw(ji,jj,2) = -1/4 * e1v(ji,jj) * ( wslpj(ji,jj,jk+1) + wslpj(ji,jj+1,jk+1) )
                             * ( aeiv(ji,jj,jk) + aeiv(ji,jj,jk+1) ) * wvmask(ji,jj,jk+1)
    puu(ji,jj,jk) += -( zpsi_uw(1) - zpsi_uw(2) )        [ldftra.F90:826-834]
    pvv(ji,jj,jk) += -( zpsi_vw(1) - zpsi_vw(2) )

``eiv_dump_u.bin``/``eiv_dump_v.bin`` (ldftra.F90:864-865) are dumped
DIRECTLY as ``zpsi_uw(1)-zpsi_uw(2)`` / ``zpsi_vw(1)-zpsi_vw(2)`` -- but the
actual increment NEMO adds is ``puu(jk) = puu(jk) - (zpsi_uw(1)-zpsi_uw(2))``
(ldftra.F90:833, a MINUS, not a plus -- the in-code comment above the dump
calling it "the eiv transport added to the advecting flux" undersells its
own sign). So the dump is the NEGATIVE of the true momentum/transport
increment. legoESM's ``nemo_eiv_bolus_transport`` returns
``u_eiv = psi_uw - psi_uw_top`` = (interface BELOW cell k) - (interface
ABOVE cell k). Tracking NEMO's slot swap (``zpsi_uw(1)`` = the value from
the PREVIOUS iteration = the interface ABOVE the current cell; ``zpsi_uw(2)``
= the interface BELOW, just built) shows
``-(zpsi_uw(1)-zpsi_uw(2)) = psi_below - psi_above = u_eiv`` exactly -- i.e.
legoESM's ``u_eiv``/``v_eiv`` (the true added increment) equals
``-1 * eiv_dump_u.bin`` / ``-1 * eiv_dump_v.bin``. Confirmed empirically
below (offset=0 correlation was exactly -1.000000 before this sign was
applied -- a perfect anti-correlation is the fingerprint of a sign bug, not
noise, and is the FIRST thing this probe checks before trusting any other
number).

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/eiv_transport_walk.py
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

# scripts/ is not a package -- import the sibling templates by path (same
# idiom every dino_1226 probe uses) so build_state(), the dump loaders,
# per_element_report() and the offset scan are NOT re-derived (Rule 0).
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
RESTART = _slp.RESTART
OFFSETS = _slp.OFFSETS

from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    compute_nemo_native_slopes,
    compute_treguier_kappa_gm_nemo_native,
    nemo_eiv_bolus_transport,
)
from legoesm.ocean.fidelity.time_levels import time_level_for_dump, register_dump
from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.grids.latlon import ensure_geometry


# NEMO source citations for the newly-dumped fields, all written inside the
# SAME kt==kit000, cdtype=='TRA' block as eiv_dump_aeiu.bin (ldftra.F90:857-
# 921), which is already registered "before" -- these are the transport-walk
# siblings of that same instrumentation pass.
_NEW_DUMPS = {
    "eiv_dump_u.bin": "ldftra.F90:864 WRITE(8801) zpsi_uw(1)-zpsi_uw(2) -- the "
                       "eiv U-transport increment, same kt==kit000 block as "
                       "eiv_dump_aeiu.bin.",
    "eiv_dump_v.bin": "ldftra.F90:865 WRITE(8802) zpsi_vw(1)-zpsi_vw(2) -- the "
                       "eiv V-transport increment, same block.",
    "eiv_dump_psi_uw.bin": "ldftra.F90:847 WRITE(8830) zpsi_uw(2) -- the raw "
                            "bolus streamfunction at the interface BELOW cell "
                            "jk, BEFORE the vertical difference (isolates psi "
                            "assembly from the differencing/masking after).",
    "eiv_dump_psi_vw.bin": "ldftra.F90:848 WRITE(8831) zpsi_vw(2) -- same, "
                            "V-direction.",
    "eiv_dump_e3w.bin": "ldftra.F90:898 WRITE(8806) e3w(...,Kmm) -- consumed "
                         "only as slope-chain geometry, not by ldf_eiv_trp "
                         "itself (see report item 1: the transport formula "
                         "has NO direct e3w/gdept read).",
    "eiv_dump_gdept.bin": "ldftra.F90:899 WRITE(8807) gdept(...,Kmm) -- same "
                           "caveat as e3w above.",
}
for _name, _src in _NEW_DUMPS.items():
    try:
        time_level_for_dump(_name)
    except ValueError:
        register_dump(_name, "before", _src)


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="eiv transport u/v walk")
    print(f"restart used  = {os.path.join(RUN_DIR, RESTART)}")
    print(f"dump dir used = {RUN_DIR}")
    print(f"LEGOESM_NEMO_E3T={e3t_mode}")

    st = build_state()
    require_fp64(st["z_coord"], st["T"], st["S"], context="eiv transport walk")
    for _name in ("eiv_dump_u.bin", "eiv_dump_v.bin"):
        lvl = time_level_for_dump(_name)
        print(f"time_level_for_dump({_name!r}) = {lvl!r} (asserted)")
        if lvl != "before":
            raise ValueError(f"{_name} is {lvl!r}-level, probe feeds BEFORE-level T/S")

    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    dtype = st["T"].dtype

    gcfg = st["gm_cfg"]
    _treg = getattr(gcfg, "treguier", None)
    if _treg is None or not getattr(_treg, "enabled", False):
        raise ValueError("GMRediConfig.treguier must be enabled for this row")
    print(f"treguier.enabled = {_treg.enabled}  aei0 = {getattr(_treg, 'aei0', None)}")
    print(f"gm_bolus_kappa_face_average (kamm card) = "
          f"{getattr(gcfg, 'gm_bolus_kappa_face_average', None)!r} (expect True)")
    print(f"gm_bolus_advection (kamm card) = "
          f"{getattr(gcfg, 'gm_bolus_advection', None)!r} (expect 'through_fct')")

    mask2d = jnp.asarray(st["mask"].astype(dtype))
    f_coriolis = jnp.broadcast_to(jnp.asarray(st["grid"].f), mask2d.shape)
    geom = ensure_geometry(st["grid"])
    e2u = geom.dy_u[:, 1:]
    e1v = geom.dx_v[1:, :]
    u_mask = jnp.asarray(st["u_mask"])
    v_mask = jnp.asarray(st["v_mask"])
    act = st["active"]  # (n_lat, n_lon, nlev) bool numpy
    act_j = jnp.asarray(act.astype(dtype))
    act_below = jnp.concatenate(
        [act_j[:, :, 1:], jnp.zeros_like(act_j[:, :, :1])], axis=2)

    def build_transport(*, live_ladder: bool):
        """Rebuild wslpi/wslpj + kappa_GM + the bolus transport, toggling
        ONLY the jacobian argument passed to the two production slope/kappa
        builders (jacobian=None -> static ladder branch inside
        _nemo_wpoint_e3w_wmask_n2; jacobian=st['jacobian'] -> live z*-stretch
        branch, gm_redi_latlon_cgrid.py:787-790).  Everything else
        (T, S, rho, eos_fn, masks, cfg) is byte-identical -- the ONE-VARIABLE
        A/B the task requires."""
        _jac = st["jacobian"] if live_ladder else None
        uslp, vslp, wslpi, wslpj = compute_nemo_native_slopes(
            st["rho"], st["T"], st["S"], mask2d, u_mask, v_mask,
            st["z_coord"], st["grid"], gcfg, st["eos_fn"],
            rho_0=st["rho_0"], g=st["g"], active_3d=st["active_3d"],
            jacobian=_jac,
        )
        kappa_t = compute_treguier_kappa_gm_nemo_native(
            st["rho"], st["T"], st["S"], wslpi, wslpj,
            mask2d, st["z_coord"], st["grid"], f_coriolis, _treg, st["eos_fn"],
            rho_0=st["rho_0"], g=st["g"], active_3d=st["active_3d"],
            slope_n2=getattr(gcfg, "slope_n2", "adiabatic"),
            jacobian=_jac, omega=st["omega"],
        )
        wslpi_kp1 = jnp.roll(wslpi, -1, axis=2)
        wslpj_kp1 = jnp.roll(wslpj, -1, axis=2)
        u_eiv, v_eiv, _w_eiv = nemo_eiv_bolus_transport(
            kappa_t, wslpi_kp1, wslpj_kp1, e2u, e1v,
            u_mask, v_mask, act_j, act_below, st["T"].shape, dtype,
            kappa_face_average=gcfg.gm_bolus_kappa_face_average,
        )
        return np.asarray(u_eiv), np.asarray(v_eiv), np.asarray(wslpi), np.asarray(wslpj)

    u_live, v_live, wslpi_live, wslpj_live = build_transport(live_ladder=True)
    print(f"\nu_eiv (live ladder) shape={u_live.shape}  v_eiv shape={v_live.shape}")

    # wet masks: u_eiv/v_eiv are nonzero only where the u/v wumask/wvmask 3-D
    # face mask is open (nemo_eiv_bolus_transport's wumask_uw/wvmask_vw), same
    # convention as ldf_slp_per_element.py's wet_u_mask/wet_v_mask (east/north
    # neighbour also active), reused verbatim.
    wet_u_full = _slp.wet_u_mask(act, st["u_mask"])
    wet_v_full = _slp.wet_v_mask(act, st["v_mask"])

    # --- load NEMO dumps ---
    # SIGN (ldftra.F90:833-834,864-865): NEMO adds
    #   puu(jk) = puu(jk) - (zpsi_uw(1)-zpsi_uw(2))
    # (a MINUS), but the dump writes (zpsi_uw(1)-zpsi_uw(2)) directly -- the
    # NEGATIVE of the true increment.  legoESM's u_eiv/v_eiv (from
    # nemo_eiv_bolus_transport) IS the true added increment (psi_below -
    # psi_above), so the oracle-comparable quantity is -1 * the raw dump.
    # Self-check: without this negation the offset=0 correlation is EXACTLY
    # -1.000000 (a perfect anti-correlation -- the fingerprint of a sign
    # bug, verified empirically before this negation was added).
    u_n_raw = _load_haloed(os.path.join(RUN_DIR, "eiv_dump_u.bin"), jpi, jpj, hls)
    v_n_raw = _load_haloed(os.path.join(RUN_DIR, "eiv_dump_v.bin"), jpi, jpj, hls)
    # [self-check 1 of 2] verify the sign empirically at offset 0 BEFORE
    # trusting the negation: raw-dump corr must be strongly NEGATIVE
    # (anti-correlated), confirming ldftra.F90:833's minus sign rather than
    # assuming it from the source read alone.
    _wu0 = wet_u_full[:, :, :u_n_raw.shape[-1]]
    _m0 = _wu0 & np.isfinite(u_live[:, :, :u_n_raw.shape[-1]]) & np.isfinite(u_n_raw)
    _raw_corr0 = float(np.corrcoef(u_live[:, :, :u_n_raw.shape[-1]][_m0],
                                    u_n_raw[_m0])[0, 1])
    print(f"[self-check 1/2] RAW (un-negated) eiv_dump_u.bin vs u_eiv, offset=0, "
          f"corr = {_raw_corr0:.6f} (expect approx -1.0, confirming the "
          f"ldftra.F90:833 MINUS sign empirically, not just by source-reading)")
    if _raw_corr0 > -0.9:
        print("  WARNING: raw corr is not strongly negative -- the sign "
              "hypothesis is NOT confirmed by this data; do not apply the "
              "negation below blindly.")
    u_n = -1.0 * u_n_raw
    v_n = -1.0 * v_n_raw
    print(f"eiv_dump_u.bin shape={u_n.shape}  eiv_dump_v.bin shape={v_n.shape}  "
          f"(NEGATED per ldftra.F90:833 sign -- see module docstring)")

    # [self-check 2 of 2] independent cross-check via the intermediate
    # streamfunction dump eiv_dump_psi_uw.bin (ldftra.F90:847, WRITE(8830)
    # zpsi_uw(...,2) -- the raw psi at the interface BELOW cell jk, dumped
    # with NO sign ambiguity: it is written verbatim, not subtracted).
    # nemo_eiv_bolus_transport does not expose psi_uw directly, but
    # u_eiv[k] = psi_uw[k] - psi_uw[k-1] (psi_uw[-1]=0 at the surface) is a
    # plain mathematical identity (not a re-derivation of any numerics), so
    # psi_uw = cumsum(u_eiv, axis=k) must reproduce the dump exactly wherever
    # both are wet. This checks the SIGN/INDEXING chain via a route
    # completely independent of the corr/ratio machinery above.
    psi_uw_path = os.path.join(RUN_DIR, "eiv_dump_psi_uw.bin")
    if os.path.exists(psi_uw_path):
        psi_uw_n = _load_haloed(psi_uw_path, jpi, jpj, hls)
        psi_uw_reconstructed = np.cumsum(u_live, axis=2)
        nk_p = psi_uw_n.shape[-1]
        wet_p = wet_u_full[:, :, :nk_p]
        mp = wet_p & np.isfinite(psi_uw_reconstructed[:, :, :nk_p]) & np.isfinite(psi_uw_n)
        if mp.sum() >= 2:
            corr_p = float(np.corrcoef(psi_uw_reconstructed[:, :, :nk_p][mp],
                                        psi_uw_n[mp])[0, 1])
            rel_p = np.median(np.abs(
                psi_uw_reconstructed[:, :, :nk_p][mp] - psi_uw_n[mp])
                / np.maximum(np.abs(psi_uw_n[mp]), 1.0))
            print(f"[self-check 2/2] cumsum(u_eiv) vs eiv_dump_psi_uw.bin: "
                  f"corr={corr_p:.6f}  median|rel|={rel_p:.3e} "
                  f"(independent route via the intermediate streamfunction; "
                  f"corr near 1.0 confirms the u_eiv sign/indexing chain, not "
                  f"just the endpoint comparison above)")
        else:
            print("[self-check 2/2] eiv_dump_psi_uw.bin: no overlapping wet points")
    else:
        print("[self-check 2/2] eiv_dump_psi_uw.bin not found -- skipped")

    # =========================================================================
    # (1) ALIGNMENT/OFFSET SCAN, live-ladder variant (the production config)
    # =========================================================================
    print("\n" + "=" * 78)
    print("(1) OFFSET SCAN -- u_eiv vs eiv_dump_u.bin (live ladder, production)")
    print("=" * 78)
    nlev_n = u_n.shape[-1]
    wet_u = wet_u_full[:, :, :nlev_n]
    scan_u = offset_scan(u_live[:, :, :nlev_n], u_n, wet_u, OFFSETS)
    for off in OFFSETS:
        r = scan_u[off]
        print(f"  offset={off:+d}: " + ("no overlap" if r is None else
              f"n={r['n']:7d}  corr={r['corr']:.6f}  (1-corr={1.0-r['corr']:.3e})"))
    valid_u = {o: r for o, r in scan_u.items() if r is not None and np.isfinite(r["corr"])}
    if not valid_u:
        print("*** NO VALID OFFSET for u_eiv -- FAILURE ***")
        return 1
    best_u = max(valid_u, key=lambda o: valid_u[o]["corr"])
    resid_u = sorted(1.0 - r["corr"] for r in valid_u.values())
    sharp_u = (len(resid_u) < 2) or (resid_u[0] <= 1e-15) or (
        resid_u[1] / max(resid_u[0], 1e-15) >= 3.0)
    print(f"BEST OFFSET (u) = {best_u:+d}  corr={valid_u[best_u]['corr']:.6f}  "
          f"sharp={sharp_u}" + ("  <-- NONZERO, FLAG" if best_u != 0 else "  (confirms 0)"))

    print("\n" + "=" * 78)
    print("(1b) OFFSET SCAN -- v_eiv vs eiv_dump_v.bin (live ladder, production)")
    print("=" * 78)
    wet_v = wet_v_full[:, :, :nlev_n]
    scan_v = offset_scan(v_live[:, :, :nlev_n], v_n, wet_v, OFFSETS)
    for off in OFFSETS:
        r = scan_v[off]
        print(f"  offset={off:+d}: " + ("no overlap" if r is None else
              f"n={r['n']:7d}  corr={r['corr']:.6f}  (1-corr={1.0-r['corr']:.3e})"))
    valid_v = {o: r for o, r in scan_v.items() if r is not None and np.isfinite(r["corr"])}
    if not valid_v:
        print("*** NO VALID OFFSET for v_eiv -- FAILURE ***")
        return 1
    best_v = max(valid_v, key=lambda o: valid_v[o]["corr"])
    resid_v = sorted(1.0 - r["corr"] for r in valid_v.values())
    sharp_v = (len(resid_v) < 2) or (resid_v[0] <= 1e-15) or (
        resid_v[1] / max(resid_v[0], 1e-15) >= 3.0)
    print(f"BEST OFFSET (v) = {best_v:+d}  corr={valid_v[best_v]['corr']:.6f}  "
          f"sharp={sharp_v}" + ("  <-- NONZERO, FLAG" if best_v != 0 else "  (confirms 0)"))

    def _align(lego, nemo, wet, off):
        nk_l, nk_n = lego.shape[-1], nemo.shape[-1]
        k_lo, k_hi = max(0, -off), min(nk_l, nk_n - off)
        return (lego[:, :, k_lo:k_hi], nemo[:, :, k_lo + off:k_hi + off],
                wet[:, :, k_lo:k_hi])

    u_al, un_al, wu_al = _align(u_live, u_n, wet_u_full, best_u)
    v_al, vn_al, wv_al = _align(v_live, v_n, wet_v_full, best_v)

    print("\n" + "=" * 78)
    print("(2)+(3) PER-ELEMENT REPORT at best offset -- u_eiv (live ladder)")
    print("=" * 78)
    rep_u = per_element_report("u_eiv (live)", u_al, un_al, wu_al)

    print("\n" + "=" * 78)
    print("(2)+(3) PER-ELEMENT REPORT at best offset -- v_eiv (live ladder)")
    print("=" * 78)
    rep_v = per_element_report("v_eiv (live)", v_al, vn_al, wv_al)

    # =========================================================================
    # (4) THE LADDER A/B -- static vs live, ONE variable (jacobian=None vs
    #     jacobian=st['jacobian']), everything else byte-identical.
    # =========================================================================
    print("\n" + "=" * 78)
    print("(4) LADDER A/B -- static (jacobian=None) vs live (jacobian=jacobian)")
    print("=" * 78)
    u_static, v_static, wslpi_static, wslpj_static = build_transport(live_ladder=False)
    ladder_identical_wslpi = np.array_equal(wslpi_static, wslpi_live)
    ladder_identical_u = np.array_equal(u_static, u_live)
    print(f"  [self-check] static wslpi bit-identical to live wslpi: "
          f"{ladder_identical_wslpi} (expect False -- else the ladder toggle "
          f"is INERT and the A/B tests nothing)")
    print(f"  [self-check] static u_eiv bit-identical to live u_eiv: "
          f"{ladder_identical_u} (expect False, same reason)")

    def _ab_report(label, lego_static, lego_live, nemo, wet, off):
        s_al, sn_al, sw_al = _align(lego_static, nemo, wet, off)
        l_al, ln_al, lw_al = _align(lego_live, nemo, wet, off)
        print(f"\n  --- {label}: STATIC ladder ---")
        rs = per_element_report(f"{label} STATIC", s_al, sn_al, sw_al)
        print(f"\n  --- {label}: LIVE ladder ---")
        rl = per_element_report(f"{label} LIVE", l_al, ln_al, lw_al)
        return rs, rl

    rep_u_static, rep_u_live_ab = _ab_report(
        "u_eiv", u_static, u_live, u_n, wet_u_full, best_u)
    rep_v_static, rep_v_live_ab = _ab_report(
        "v_eiv", v_static, v_live, v_n, wet_v_full, best_v)

    # =========================================================================
    # (3) PER-LEVEL RESIDUAL TABLE (35 levels) -- already printed inside
    # per_element_report's "(4) per-level median err_norm" block for BOTH u
    # and v (live-ladder variant, item 2/3 above); pull the verdict strings
    # explicitly here so the ladder-shape verdict is unambiguous in the log.
    # =========================================================================
    print("\n" + "=" * 78)
    print("PER-LEVEL LADDER-SIGNATURE VERDICT (from the per_element_report "
          "tables above)")
    print("=" * 78)
    print("  See 'per-level spread' lines printed just above for u_eiv/v_eiv "
          "(live ladder): FLAT (max/min<10x) => REFUTED (matches the aeiu "
          "row's own flat signature, which excluded the ladder there); "
          "STRUCTURED with the worst level(s) at the DEEPEST wet levels "
          "=> CONFIRMED in shape.")

    # --- final summary ---
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"RECORDED (stale): u_eiv corr=0.999617 ratio=0.996109; "
          f"v_eiv corr=0.999103 ratio=0.990501")
    print(f"NOW (live ladder, production): u_eiv corr={rep_u['corr']:.6f} "
          f"ratio={rep_u['abs_ratio']:.6f}  best_off={best_u:+d} sharp={sharp_u}")
    print(f"                                v_eiv corr={rep_v['corr']:.6f} "
          f"ratio={rep_v['abs_ratio']:.6f}  best_off={best_v:+d} sharp={sharp_v}")
    print(f"A/B (u_eiv): static med_en={rep_u_static['med_en']:.3e} "
          f"ratio={rep_u_static['abs_ratio']:.6f}  |  "
          f"live med_en={rep_u_live_ab['med_en']:.3e} "
          f"ratio={rep_u_live_ab['abs_ratio']:.6f}")
    print(f"A/B (v_eiv): static med_en={rep_v_static['med_en']:.3e} "
          f"ratio={rep_v_static['abs_ratio']:.6f}  |  "
          f"live med_en={rep_v_live_ab['med_en']:.3e} "
          f"ratio={rep_v_live_ab['abs_ratio']:.6f}")
    u_v_ratio = (rep_v["med_en"] / rep_u["med_en"]
                 if rep_u["med_en"] > 0 else float("nan"))
    print(f"u/v med_en asymmetry: v/u = {u_v_ratio:.3f}x "
          f"(recorded ratio residual asymmetry was ~2.4x)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
