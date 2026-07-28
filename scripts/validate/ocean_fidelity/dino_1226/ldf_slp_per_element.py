#!/usr/bin/env python
"""#1226 tier-2/3: PER-ELEMENT error of the four ``ldf_slp`` isopycnal-slope
rows (``wslpi``, ``wslpj``, ``uslp``, ``vslp``) vs NEMO 5.0.2's own
``eiv_dump_*.bin`` dumps.

Modeled directly on ``eos_rab_bn2_per_element.py`` (same ``build_state()``
shape, same two mechanical preconditions, same restart/dump run-directory
pairing) and reuses the offset-scan idea from the throwaway
``probe_all4_slopes.py`` -- but rebuilt from the trusted template rather than
from that older probe's setup (it read mesh_mask/restart from a DIFFERENT run
directory than the dumps -- see below).

No numerics are re-implemented here (Rule 0): the slopes come from the real
production entry point ``compute_nemo_native_slopes`` in
``gm_redi_latlon_cgrid.py``, called with the same ``gm_redi_density_and_
jacobian`` + ``_nemo_native_active_3d`` helpers the template already uses.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/ldf_slp_per_element.py
"""
from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
# Per eos_rab_bn2_per_element.py / bn2_alpha_compare.py: DINO is full-step;
# pin the NEMO e3t ladder so gdept_0 matches NEMO's own dump exactly.
os.environ["LEGOESM_NEMO_E3T"] = "both"

import importlib.util
import sys

import numpy as np
import jax.numpy as jnp

# scripts/ is not a package -- import the sibling probe by path (same idiom
# eos_rab_bn2_per_element.py uses) to reuse _read_dims/_load_haloed without
# duplicating them.
_sib_path = os.path.join(os.path.dirname(__file__), "bn2_alpha_compare.py")
_spec = importlib.util.spec_from_file_location("_bn2_alpha_compare", _sib_path)
_bn2_alpha_compare = importlib.util.module_from_spec(_spec)
sys.modules["_bn2_alpha_compare"] = _bn2_alpha_compare
_spec.loader.exec_module(_bn2_alpha_compare)
_read_dims = _bn2_alpha_compare._read_dims
_load_haloed = _bn2_alpha_compare._load_haloed

from legoesm.ocean.eos import make_eos_fn
from legoesm.ocean.experiments.dino import (
    dino_config_for_recipe, dino_lat_lon_model_config,
)
from legoesm.ocean.fidelity.nemo_io import (
    read_nemo_mesh_mask,
    read_nemo_restart,
    read_nemo_restart_before,
)
from legoesm.ocean.fidelity.nemo_state_bridge import bridge_nemo_to_legoesm_topo
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    compute_nemo_native_slopes, gm_redi_density_and_jacobian,
    _nemo_native_active_3d,
)

# STATE / DUMP CONSISTENCY: read the restart from the SAME run directory as
# the dumps. probe_all4_slopes.py (the throwaway predecessor) read mesh_mask
# from RUN_TRAJ and the restart from RUN_Y5_REBUILD against RUN_GDB's dumps --
# a cross-run-directory mismatch. mesh_mask.nc also exists in RUN_GDB itself
# (verified), so there is no reason to reach outside it.
RUN_DIR = "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_GDB"
RESTART = "DINO_00057600_restart.nc"

FLOOR = 1.0e-12
OFFSETS = (-2, -1, 0, 1, 2)

DUMP_META = {
    # component: (dump basename, lego-array-role)
    "wslpi": "eiv_dump_wslpi.bin",
    "wslpj": "eiv_dump_wslpj.bin",
    "uslp": "eiv_dump_uslp.bin",
    "vslp": "eiv_dump_vslp.bin",
}


def build_state():
    # PRECISION (#1226): see eos_rab_bn2_per_element.py -- fp32 control policy
    # silently rounds the fp64 depth ladder; JAX_ENABLE_X64=1 does NOT change
    # legoESM's own precision policy.
    if os.environ.get("LEGOESM_FIDELITY_FP64", "1") == "1":
        from legoesm.core.precision import PrecisionPolicy, set_policy
        set_policy(PrecisionPolicy.fp64())

    jpi, jpj, jpk, hls = _read_dims(RUN_DIR)
    grid = read_nemo_mesh_mask(os.path.join(RUN_DIR, "mesh_mask.nc"), nn_hls=0)
    now = read_nemo_restart(os.path.join(RUN_DIR, RESTART), nn_hls=0)
    # TIME LEVEL: ldfslp.F90 (ldf_slp) consumes rn2b/rab_b -- BEFORE-level
    # T/S -- on the NOW geometry (eta/H_bathy), exactly the eos_rab/bn2
    # pairing the template already established. Fed here explicitly rather
    # than assumed; asserted below via time_level_for_dump.
    bef = read_nemo_restart_before(os.path.join(RUN_DIR, RESTART), nn_hls=0)

    cfg = dataclasses.replace(
        dino_config_for_recipe("nemo_dino_kamm_mlf"),
        lon_west_deg=1.0, lon_east_deg=49.0, sill_lon_m_deg=1.0,
    )
    br = bridge_nemo_to_legoesm_topo(grid, now, periodic_i=True, full_step=True,
                                      omega=cfg.omega)
    mc, _ = dino_lat_lon_model_config(br.geometry, cfg)

    z_coord = br.z_coord
    state = br.state
    mask = state.land_mask.data
    u_mask = state.u_mask.data
    v_mask = state.v_mask.data
    H_bathy = state.H_bathy.data
    _tb = np.asarray(bef.T); _sb = np.asarray(bef.S)
    T = jnp.asarray(_tb.reshape(state.T.data.shape))
    S = jnp.asarray(_sb.reshape(state.S.data.shape))
    eta = jnp.asarray(state.eta.data)

    rho, jacobian = gm_redi_density_and_jacobian(
        T, S, eta, H_bathy, br.geometry, z_coord,
        eos=mc.eos, eos_linear=mc.eos_linear, mask=mask,
        rho_0=mc.rho_0, g=mc.g,
    )
    active_3d = _nemo_native_active_3d(mask, z_coord, H_bathy, T.dtype)

    # --- oracle-comparison preconditions, both MECHANICAL (#1226) ---
    from legoesm.ocean.fidelity.precision_gate import require_fp64
    require_fp64(z_coord, T, S, context="ldf_slp per-element")

    from legoesm.ocean.fidelity.time_levels import time_level_for_dump
    for dump_name in DUMP_META.values():
        lvl = time_level_for_dump(dump_name)
        if lvl != "before":
            raise ValueError(
                f"this probe feeds BEFORE-level T/S but {dump_name!r} is "
                f"{lvl!r}-level; fix the probe or the registry")

    eos_fn = make_eos_fn(mc.eos, mc.eos_linear)
    card_slope_n2 = mc.gm_redi.slope_n2
    env_override = os.environ.get("SLOPE_N2")
    slope_n2_used = env_override if env_override is not None else card_slope_n2
    gm_cfg = mc.gm_redi._replace(slope_n2=slope_n2_used)

    uslp, vslp, wslpi, wslpj = compute_nemo_native_slopes(
        rho, T, S, mask, u_mask, v_mask, z_coord, br.geometry, gm_cfg, eos_fn,
        rho_0=mc.rho_0, g=mc.g, active_3d=active_3d, jacobian=jacobian,
    )

    return dict(
        jpi=jpi, jpj=jpj, jpk=jpk, hls=hls,
        mask=np.asarray(mask) > 0.5,
        u_mask=np.asarray(u_mask), v_mask=np.asarray(v_mask),
        active=np.asarray(active_3d) > 0.5,
        lego=dict(wslpi=np.asarray(wslpi), wslpj=np.asarray(wslpj),
                   uslp=np.asarray(uslp), vslp=np.asarray(vslp)),
        card_slope_n2=card_slope_n2, slope_n2_used=slope_n2_used,
        env_override=env_override,
    )


# ---------------------------------------------------------------------------
# wet-point selectors -- reproduce the EXACT umask3/vmask3/wmask3 conventions
# compute_nemo_native_slopes itself builds internally (gm_redi_latlon_cgrid.py
# :962-963, :790-791), using the SAME active_3d this script fed it. This is a
# stats-only wet mask, not a re-implementation of the slope numerics (Rule 0
# is about not re-deriving the computed VALUES; selecting which points are
# wet for a stats report is unavoidable bookkeeping and must match the
# production convention exactly, which is why it is derived from `act`
# rather than a hand re-guessed mask).
def wet_w_mask(act):
    """W-point wet mask at level k: wmask3 = act[k] & act[k-1] (k=0: act[0])."""
    nlev = act.shape[-1]
    above = np.concatenate([act[:, :, :1], act[:, :, :-1]], axis=-1)
    return act & above


def wet_u_mask(act, u_mask):
    east_active = np.roll(act, -1, axis=1)
    face2d = u_mask[:, 1:]
    return (face2d[:, :, None] > 0.5) & act & east_active


def wet_v_mask(act, v_mask):
    north_active = np.roll(act, -1, axis=0)
    face2d = v_mask[1:, :]
    return (face2d[:, :, None] > 0.5) & act & north_active


MASK_FN = {
    "wslpi": lambda st: wet_w_mask(st["active"]),
    "wslpj": lambda st: wet_w_mask(st["active"]),
    "uslp": lambda st: wet_u_mask(st["active"], st["u_mask"]),
    "vslp": lambda st: wet_v_mask(st["active"], st["v_mask"]),
}


def offset_scan(lego, nemo, wet, offsets=OFFSETS):
    """corr at each vertical offset: lego[...,k] vs nemo[...,k+offset]."""
    nlev_lego = lego.shape[-1]
    nlev_nemo = nemo.shape[-1]
    out = {}
    for off in offsets:
        L_parts, N_parts = [], []
        for k in range(nlev_lego):
            kn = k + off
            if not (0 <= kn < nlev_nemo):
                continue
            wk = wet[:, :, k]
            if not wk.any():
                continue
            L_parts.append(lego[:, :, k][wk])
            N_parts.append(nemo[:, :, kn][wk])
        if not L_parts:
            out[off] = None
            continue
        L = np.concatenate(L_parts)
        N = np.concatenate(N_parts)
        finite = np.isfinite(L) & np.isfinite(N)
        L, N = L[finite], N[finite]
        if L.size < 2 or L.std() == 0.0 or N.std() == 0.0:
            out[off] = dict(n=int(L.size), corr=float("nan"))
            continue
        out[off] = dict(n=int(L.size), corr=float(np.corrcoef(L, N)[0, 1]))
    return out


def per_element_report(name, lego, nemo, wet):
    """Full per-element report at a fixed (already-chosen) offset alignment.

    lego/nemo/wet must already share the SAME level axis (caller has sliced
    to the offset-aligned overlap range)."""
    m = wet & np.isfinite(lego) & np.isfinite(nemo)
    lo, ne = lego[m], nemo[m]
    n = int(m.sum())
    rms_nemo = float(np.sqrt(np.mean(ne ** 2))) if n else float("nan")

    corr = (float(np.corrcoef(lo, ne)[0, 1])
            if n >= 2 and lo.std() > 0 and ne.std() > 0 else float("nan"))
    abs_ratio = (float(np.abs(lo).sum() / np.abs(ne).sum())
                 if np.abs(ne).sum() > 0 else float("nan"))

    # (2) pointwise |lego-nemo|/|nemo| -- REFERENCE ONLY, not trustworthy for
    # a sign-changing field (see (3) below).
    denom_pt = np.maximum(np.abs(ne), FLOOR)
    rel_pt = np.abs(lo - ne) / denom_pt
    med_rel_pt = float(np.median(rel_pt))
    p99_rel_pt = float(np.percentile(rel_pt, 99))
    max_rel_pt = float(np.max(rel_pt))

    # conditioning-robust metric: err_norm = |lego-nemo| / RMS(nemo over wet)
    err_norm = np.abs(lo - ne) / max(rms_nemo, FLOOR)
    med_en = float(np.median(err_norm))
    p99_en = float(np.percentile(err_norm, 99))
    max_en = float(np.max(err_norm))

    # location of max err_norm, in the FULL (unmasked-shape) array
    en_full = np.full(wet.shape, -1.0)
    en_full[m] = err_norm
    idx = tuple(int(v) for v in np.unravel_index(np.argmax(en_full), en_full.shape))
    nemo_at_max = float(nemo[idx])

    # (3) conditioning check: fraction of wet points with |nemo| < 1e-3*RMS
    near0 = np.abs(ne) < 1.0e-3 * max(rms_nemo, FLOOR)
    frac_near0 = float(near0.mean()) if n else float("nan")
    trustworthy_pointwise = frac_near0 < 0.05

    print(f"\n--- {name}: at chosen offset ---")
    print(f"  n wet elements       = {n}")
    print(f"  corr                 = {corr:.6f}")
    print(f"  |x| ratio (sum|lego|/sum|nemo|) = {abs_ratio:.6f}")
    print(f"  RMS(nemo over wet)   = {rms_nemo:.6e}")
    print(f"  pointwise |rel| (REFERENCE ONLY, sign-changing field):")
    print(f"    median={med_rel_pt:.3e}  p99={p99_rel_pt:.3e}  max={max_rel_pt:.3e}")
    print(f"  err_norm = |lego-nemo|/RMS(nemo) (CONDITIONING-ROBUST):")
    print(f"    median={med_en:.3e}  p99={p99_en:.3e}  max={max_en:.3e}")
    print(f"    max err_norm at (j,i,level)={idx}  |nemo| there = {abs(nemo_at_max):.6e} "
          f"(nemo={nemo_at_max:+.6e}, lego={float(lego[idx]):+.6e})")
    print(f"  CONDITIONING: fraction of wet points with |nemo| < 1e-3*RMS(nemo) "
          f"= {frac_near0 * 100:.3f}%  ({int(near0.sum())}/{n})")
    if trustworthy_pointwise:
        print(f"  -> pointwise-relative stats for {name} are LIKELY TRUSTWORTHY "
              f"(near-zero fraction < 5%).")
    else:
        print(f"  -> pointwise-relative stats for {name} are NOT TRUSTWORTHY "
              f"(large near-zero-denominator fraction); use err_norm.")

    # (4) per-level median err_norm
    nlev = wet.shape[-1]
    print(f"  per-level median err_norm ({nlev} levels):")
    level_meds = []
    line = []
    for k in range(nlev):
        mk = m[:, :, k]
        if not mk.any():
            line.append(f"k{k}:--")
            continue
        ek = np.abs(lego[:, :, k][mk] - nemo[:, :, k][mk]) / max(rms_nemo, FLOOR)
        med_k = float(np.median(ek))
        level_meds.append((k, med_k))
        line.append(f"k{k}:{med_k:.2e}")
    print("    " + "  ".join(line))
    if level_meds:
        meds = np.array([v for _, v in level_meds])
        kbest = level_meds[int(np.argmin(meds))][0]
        kworst = level_meds[int(np.argmax(meds))][0]
        # floor the denominator at a small ABSOLUTE err_norm (not 1e-300):
        # a level whose median err_norm is itself ~0 (e.g. surface w-level,
        # legitimately near-exact) must not blow the ratio up to a
        # meaningless astronomical number.
        spread = float(meds.max() / max(meds.min(), 1e-8))
        print(f"  per-level spread: min={meds.min():.3e} (k={kbest})  "
              f"max={meds.max():.3e} (k={kworst})  max/min={spread:.1f}x  "
              f"-> {'FLAT (roundoff-like)' if spread < 10 else 'STRUCTURED (points at a term)'}")

    return dict(n=n, corr=corr, abs_ratio=abs_ratio, rms_nemo=rms_nemo,
                med_rel_pt=med_rel_pt, p99_rel_pt=p99_rel_pt, max_rel_pt=max_rel_pt,
                med_en=med_en, p99_en=p99_en, max_en=max_en,
                frac_near0=frac_near0, trustworthy_pointwise=trustworthy_pointwise)


def main() -> int:
    print(f"restart used  = {os.path.join(RUN_DIR, RESTART)}")
    print(f"dump dir used = {RUN_DIR}")
    print(f"LEGOESM_NEMO_E3T={os.environ.get('LEGOESM_NEMO_E3T')}")

    st = build_state()
    jpi, jpj, hls = st["jpi"], st["jpj"], st["hls"]
    print(f"card default GMRediConfig.slope_n2 = {st['card_slope_n2']!r}")
    print(f"SLOPE_N2 env override = {st['env_override']!r}")
    print(f"==> slope_n2 ACTUALLY USED = {st['slope_n2_used']!r}")
    nj, ni = st["mask"].shape
    print(f"grid interior (nj,ni) = ({nj},{ni})  wet T-columns = {int(st['mask'].sum())}")

    summary = {}
    for comp, dump_name in DUMP_META.items():
        print("\n" + "=" * 78)
        print(f"COMPONENT: {comp}  (dump={dump_name})")
        print("=" * 78)
        lego = st["lego"][comp]
        dump_path = os.path.join(RUN_DIR, dump_name)
        nemo = _load_haloed(dump_path, jpi, jpj, hls)
        wet3d_full = MASK_FN[comp](st)
        print(f"lego shape={lego.shape}  nemo dump shape={nemo.shape}  "
              f"wet3d shape={wet3d_full.shape}  total wet (lego-level-count)="
              f"{int(wet3d_full.sum())}")

        # --- (1) offset scan ---
        print("\n--- offset scan (corr at each vertical offset) ---")
        scan = offset_scan(lego, nemo, wet3d_full, OFFSETS)
        for off in OFFSETS:
            r = scan[off]
            if r is None:
                print(f"  offset={off:+d}: no overlapping wet levels")
            else:
                print(f"  offset={off:+d}: n={r['n']:7d}  corr={r['corr']:.6f}")
        valid = {o: r for o, r in scan.items() if r is not None and np.isfinite(r["corr"])}
        if not valid:
            print(f"*** {comp}: NO VALID OFFSET -- cannot compare, reporting as FAILURE ***")
            summary[comp] = None
            continue
        best_off = max(valid, key=lambda o: valid[o]["corr"])
        # corr is bounded above by 1, so near the peak it compresses (0.9999
        # vs 0.987 IS a decisive gap even though the raw difference is only
        # 0.012) -- compare on (1-corr) ("residual"), where a >=3x gap
        # between best and runner-up counts as a sharp peak.
        resid_sorted = sorted(1.0 - r["corr"] for r in valid.values())
        sharp = (len(resid_sorted) < 2) or (resid_sorted[0] <= 1e-15) or (
            resid_sorted[1] / max(resid_sorted[0], 1e-15) >= 3.0)
        print(f"BEST OFFSET = {best_off:+d}  corr={valid[best_off]['corr']:.6f}  "
              f"sharp peak vs runner-up (>=3x gap in 1-corr): {sharp}"
              + ("  <-- NONZERO OFFSET, FLAG" if best_off != 0 else ""))
        if not sharp:
            print("  WARNING: best-offset corr is NOT a sharp peak over the runner-up -- "
                  "alignment is ambiguous; treat downstream stats with caution.")

        # slice lego/nemo/wet to the offset-aligned common level range for (2)-(4)
        nlev_lego = lego.shape[-1]
        nlev_nemo = nemo.shape[-1]
        k_lo = max(0, -best_off)
        k_hi = min(nlev_lego, nlev_nemo - best_off)
        lego_al = lego[:, :, k_lo:k_hi]
        nemo_al = nemo[:, :, k_lo + best_off:k_hi + best_off]
        wet_al = wet3d_full[:, :, k_lo:k_hi]

        rep = per_element_report(comp, lego_al, nemo_al, wet_al)
        rep["best_off"] = best_off
        rep["sharp"] = sharp
        summary[comp] = rep

    print("\n" + "=" * 78)
    print("SUMMARY (<=12-line)")
    print("=" * 78)
    for comp in DUMP_META:
        r = summary[comp]
        if r is None:
            print(f"{comp:8s}: FAILED (no valid offset)")
            continue
        trust = "trustworthy" if r["trustworthy_pointwise"] else "NOT trustworthy"
        print(f"{comp:8s}: offset={r['best_off']:+d}  corr={r['corr']:.6f}  "
              f"|x|ratio={r['abs_ratio']:.6f}  err_norm(median/p99/max)="
              f"{r['med_en']:.3e}/{r['p99_en']:.3e}/{r['max_en']:.3e}  "
              f"pointwise-rel {trust} ({r['frac_near0']*100:.1f}% near-zero)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
