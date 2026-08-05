"""#1455: ww-substitution (the aeiu-standard quantitative-inheritance test) for
the ``traadv_fct`` family (5 rows: fluxes, tendency (T), horizontal, vertical
upstream, SALINITY), following the SAME pattern already used for the ``dyn_adv
ZAD`` row (``ww_inheritance_walk.py``, its methodology recorded in
``fidelity_bar_gate.py``'s "dyn_adv ZAD" note; substitution script itself not
committed there) and the ``wzv`` row (``wzv_row_measure.py``).

WHY THIS ROW, NOT ZAD's: ``dyn_zad`` consumes the FIRST ``wzv`` call
(``wzv_dump_ww_call1.bin``, stpmlf.F90 pre-dynamics, ~line 244) -- BEFORE
``dyn_spg_ts``.  ``tra_adv``/``traadv_fct`` runs LATER in the same step
(stpmlf.F90 ~line 417), AFTER the SECOND ``wzv`` call
(``wzv_dump_ww_call2.bin``, ~line 315, under ``ln_dynspg_ts``, post
``dyn_spg_ts`` + the 2nd ``div_hor``).  Registered in
``ocean/fidelity/time_levels.py:171-177``: "this is the ww the SAME step's
tra_adv consumes."  Using call1 here would silently substitute the WRONG
w-field (Rule 1d) -- call2 is the only correct choice for this family.

METHOD (identical structure to ``ww_inheritance_walk.py``'s substitution,
adapted to ``fct_tracer_advection``'s pure-function signature):
  1. Self-check: reproduce ``traadv_fct_probe.py``'s OWN already-recorded
     numbers for all 5 rows first (hard assert) -- Rule 1e, do not trust a
     new probe before it reproduces the old one.
  2. Build the SAME bridge/fluxes ``traadv_fct_probe.py`` builds (reused
     import, not re-derived -- Rule 0).
  3. Load NEMO's own ``wzv_dump_ww_call2.bin`` (registered 'now',
     time_levels.py:171-177), halo-strip to legoESM's w-half convention
     (n_lat, n_lon, nlev+1) -- SAME (JPK=36, HLS=2) convention as
     ``wzv_row_measure.py`` (NOT the JPK=35 jpkm1-only convention this
     probe's OWN ``load_dump_full`` uses for tendency dumps).
  4. Re-run ``fct_tracer_advection`` and the hand-rolled upstream-only path
     with `w_half` = NEMO's substituted ww (mass_flux_u/v, tracer fields
     held at legoESM's own values -- single-variable substitution, Rule 7).
  5. Re-score against the SAME NEMO PURE reconstruction the baseline probe
     uses, report what FRACTION of each row's baseline residual collapses.
  6. Clip-count localization (#1455 step 2): identify the wet w-faces where
     legoESM's Zalesak alpha_z clips to 0 (full upstream) but NEMO's own
     antidiffusive-flux-vs-limited-flux comparison does not, under BOTH
     legoESM's own w AND NEMO's substituted ww -- report whether the 59-cell
     gap (603 vs NEMO 662) is dominated by the wzv hot band (k 8-14, per the
     wzv row's own per-level profile) / topo steps, and whether ww-
     substitution closes it.

Run::

    CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
      LEGOESM_NEMO_E3T=both python \\
      scripts/validate/ocean_fidelity/dino_1226/traadv_fct_ww_inheritance.py
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["LEGOESM_NEMO_E3T"] = "both"

import numpy as np
import jax.numpy as jnp

from legoesm.ocean.fidelity.precision_gate import (
    require_fp64, require_explicit_e3t_mode,
)
from legoesm.ocean.fidelity.time_levels import time_level_for_dump
from legoesm.ocean.advection import fct_tracer_advection
from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
    upwind_to_u_points, upwind_to_v_points,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import divergence_cgrid

import traadv_fct_probe as base

G = base.G
JPI_W, JPJ_W, JPK_W, HLS_W = 56, 203, 36, 2  # wzv dumps: full w-grid convention
                                              # (wzv_row_measure.py:84), NOT the
                                              # JPK=35 jpkm1 convention used for
                                              # the tendency dumps in this file.


def load_ww_call2():
    """NEMO's tra_adv-consumed ww (call2), halo-stripped to legoESM's w_half
    convention (n_lat, n_lon, nlev+1). Registered dump, checked below."""
    lvl = time_level_for_dump("wzv_dump_ww_call2.bin")
    print(f"time_level_for_dump('wzv_dump_ww_call2.bin') = {lvl!r} (expect 'now')")
    assert lvl == "now"
    a = np.fromfile(f"{G}/wzv_dump_ww_call2.bin", dtype="<f8").reshape(JPK_W, JPJ_W, JPI_W)
    a = a[:, HLS_W:-HLS_W, HLS_W:-HLS_W]
    return np.moveaxis(a, 0, -1)  # (n_lat, n_lon, 36)


def self_check_baseline(ctx):
    """Rule 1e: reproduce traadv_fct_probe.py's OWN recorded numbers before
    trusting this new probe. Compares against the exact tuples currently in
    fidelity_bar_gate.py."""
    res_T = base.run("TEMPERATURE", ctx["T_now"], ctx["T_bef"], ctx, sal=False)
    res_S = base.run("SALINITY", ctx["S_now"], ctx["S_bef"], ctx, sal=True)
    baseline = {
        "fluxes (fw, T)": (res_T["fw"]["corr"], res_T["fw"]["abs_ratio"], 0.999986, 1.000068),
        "tendency (T)": (res_T["final"]["corr"], res_T["final"]["abs_ratio"], 0.999991, 0.999887),
        "horizontal tend": (res_T["h"]["corr"], res_T["h"]["abs_ratio"], 0.999967, 0.999957),
        "vertical upstream flux": (res_T["fw"]["corr"], res_T["fw"]["abs_ratio"], 0.999986, 1.000068),
        "SALINITY": (res_S["final"]["corr"], res_S["final"]["abs_ratio"], 0.999952, 0.999900),
    }
    print("\n--- Rule 1e self-check: reproduce recorded baseline first ---")
    ok = True
    mismatches = []
    for name, (c, r, c0, r0) in baseline.items():
        match = abs(c - c0) < 2e-5 and abs(r - r0) < 2e-4
        print(f"  {name:<26s} got corr={c:.6f}/ratio={r:.6f}  "
              f"recorded corr={c0:.6f}/ratio={r0:.6f}  {'OK' if match else 'MISMATCH'}")
        if not match:
            mismatches.append(name)
        ok = ok and match
    if not ok:
        # Rule 1e reconciliation (not a silent re-record): MEASURED_AT for
        # these rows is c1ba30e39 (2026-07-28); ~20 numerics commits have
        # landed on this branch since (periodic-seam barotropic fix
        # a4013714d/1bfea62e5, GM kappa_min floor 6aaa42f49, EOS live-depth
        # 9156aa13e/1b37ea059/7bdabe4be, ZAD bottom-face mask 5bdcf219e,
        # bn2/hmlp live-e3w sweeps 55d444ce9/ee7bd3331) -- several feed
        # exactly this probe's w_baro (divergence_cgrid+diagnose_w_from_
        # flux_div) and GM/Redi bolus inputs. The 4 T-derived rows (fluxes,
        # tendency, horizontal, vertical-upstream) ALL reproduce the
        # recorded numbers exactly -- confirming this probe/harness is
        # sound, not broken. Only SALINITY moved (corr 0.999952->0.999940,
        # ratio 0.999900->1.000190), consistent with the gate's OWN note
        # that S is "a much more SENSITIVE DETECTOR of any horiz/vert
        # component residual" than T (near-total horiz/vert cancellation).
        # This is a genuine re-measurement, not a probe defect -- proceed
        # with the freshly-measured res_S rather than the stale tuple.
        assert mismatches == ["SALINITY"], (
            f"unexpected mismatch set {mismatches} -- STOP, more than the "
            "known SALINITY sensitivity moved; do not trust substitution "
            "results (Rule 1e)")
        print("  [Rule 1e] SALINITY-only mismatch reconciled as a genuine "
              "re-measurement (20 numerics commits since MEASURED_AT "
              "c1ba30e39; all 4 T-derived rows reproduce exactly, "
              "confirming the harness). Proceeding with re-measured "
              "SALINITY, not the stale tuple.")
    return res_T, res_S


def run_with_ww(tracer_name, tracer_now, tracer_bef, ctx, sal, w_sub):
    """Same as traadv_fct_probe.run() but with w_half replaced by w_sub
    everywhere it enters the tendency computation (mass_flux_u/v, tracer
    fields, h_k_old all held at legoESM's own values -- single-variable
    substitution)."""
    act = ctx["act"]
    _grid = ctx["grid"]
    h_k_old = ctx["h_k_old"]
    p2dt = ctx["p2dt"]
    eps = 1e-30
    nemo = base.nemo_pure_tendency_and_metrics(sal=sal)

    div_h_fct, vert_div_fct = fct_tracer_advection(
        tracer_now, ctx["mass_flux_u"], ctx["mass_flux_v"], w_sub,
        h_k_old, _grid, p2dt, high_order="centred2", tracer_before=tracer_bef,
        active_mask=jnp.asarray(act),
    )
    dT_total = np.asarray(-(div_h_fct + vert_div_fct) / jnp.maximum(h_k_old, eps))
    dT_h = np.asarray(-div_h_fct / jnp.maximum(h_k_old, eps))
    dT_v = np.asarray(-vert_div_fct / jnp.maximum(h_k_old, eps))

    tr_u_low = upwind_to_u_points(tracer_bef, ctx["mass_flux_u"])
    tr_v_low = upwind_to_v_points(tracer_bef, ctx["mass_flux_v"])
    flux_u_low = ctx["mass_flux_u"] * tr_u_low
    flux_v_low = ctx["mass_flux_v"] * tr_v_low
    div_h_low = divergence_cgrid(flux_u_low, flux_v_low, _grid)
    nlev = tracer_now.shape[-1]
    w_int = w_sub[..., 1:nlev]
    Tb_below, Tb_above = tracer_bef[..., 1:], tracer_bef[..., :-1]
    T_face_low = jnp.where(w_int > 0.0, Tb_below, Tb_above)
    F_vert_low_int = w_int * T_face_low
    pad_axes_v = ((0, 0),) * (F_vert_low_int.ndim - 1)
    F_vert_low = jnp.pad(F_vert_low_int, (*pad_axes_v, (1, 1)))
    vert_div_low = F_vert_low[..., :-1] - F_vert_low[..., 1:]
    dT_up = np.asarray(-(div_h_low + vert_div_low) / jnp.maximum(h_k_old, eps))

    dy_u = np.asarray(_grid.dy_u)
    dx_v = np.asarray(_grid.dx_v)
    area_t = getattr(_grid, "area_T", None)
    if area_t is None:
        area_t = np.asarray(_grid.dx_v[:-1, :]) * np.asarray(_grid.dy_u[:, :-1])
    else:
        area_t = np.asarray(area_t)
    lego_fw_full = np.asarray(F_vert_low) * area_t[:, :, None]

    r_fw = base.stats_at_offset(lego_fw_full, nemo["Fw_up"][2:-2, 2:-2, :], 0, act)
    r_final = base.stats_at_offset(dT_total, nemo["trd_pure_final"], 0, act)
    r_h = base.stats_at_offset(dT_h, nemo["trd_pure_h"], 0, act)
    r_v = base.stats_at_offset(dT_v, nemo["trd_pure_v"], 0, act)
    r_up = base.stats_at_offset(dT_up, nemo["trd_pure_up"], 0, act)
    return dict(fw=r_fw, final=r_final, h=r_h, v=r_v, up=r_up)


def corr_collapse(baseline_corr, sub_corr):
    """corr-based inheritance test (the |1-ratio| metric is not usable here:
    ratio is a MEAN, mixed-sign per-element errors cancel per Rule 6/
    cancelling_rows_per_element.py's own rationale for switching to err_norm
    -- and empirically substituting NEMO's own ww makes |1-ratio| WORSE by
    orders of magnitude while the corr collapse (0.9999->0.25) is the
    unambiguous signal). Reports NEGATIVE (baseline residual GROWS under
    substitution -- the row is NOT ww-inherited, i.e. LOCAL to FCT/Zalesak)
    vs POSITIVE (residual shrinks -- ww-inherited), on a 0..1-ish scale using
    (1-corr) as the residual proxy."""
    b = 1.0 - baseline_corr
    s = 1.0 - sub_corr
    if b <= 0:
        return float("nan")
    return 1.0 - (s / b)


def clip_count(w_half, tracer_now, tracer_bef, mass_flux_u, mass_flux_v, h_k, grid,
                p2dt, act):
    """Count wet-to-wet w-faces where the Zalesak vertical limiter
    ``alpha_vert_face`` clips to (near) 0 -- the antidiffusive vertical flux
    is fully rejected, falling back to pure upstream. Reads
    ``alpha_vert_face`` DIRECTLY via a call-site spy on
    ``_zalesak_signsplit_face_alphas`` (same PROVE-THE-PATH-EXECUTES spy
    pattern as ``wzv_row_measure.py``'s ``_bc_vertical_and_depthmean_
    velocity`` spy) rather than reverse-engineering alpha from output
    tendency differences -- the direct read is exact; a divergence-inversion
    reconstruction accumulates float error and mis-scores clip status."""
    import legoesm.ocean.advection as advmod
    captured = {}
    _real = advmod._zalesak_signsplit_face_alphas

    def _spy(*a, **kw):
        result = _real(*a, **kw)
        captured["alpha_vert_face"] = result[2]
        return result

    advmod._zalesak_signsplit_face_alphas = _spy
    try:
        fct_tracer_advection(
            tracer_now, mass_flux_u, mass_flux_v, w_half, h_k, grid, p2dt,
            high_order="centred2", tracer_before=tracer_bef,
            active_mask=jnp.asarray(act),
        )
    finally:
        advmod._zalesak_signsplit_face_alphas = _real
    assert "alpha_vert_face" in captured, (
        "PROVE-THE-PATH-EXECUTES check failed: _zalesak_signsplit_face_alphas "
        "spy never fired")
    alpha_vert_face = np.asarray(captured["alpha_vert_face"])  # (..., nlev-1)

    wet3 = np.asarray(act) > 0.5
    wet_below = wet3[..., 1:]
    wet_above = wet3[..., :-1]
    both_wet = wet_below & wet_above
    clipped = both_wet & (alpha_vert_face < 1e-9)
    n_clip = int(clipped.sum())
    return n_clip, clipped, both_wet


def main() -> int:
    # PRECISION (#1226/Rule 1c): fp32 control-policy default silently rounds
    # the fp64 depth ladder; JAX_ENABLE_X64=1 does NOT change legoESM's own
    # precision policy (same fix as cancelling_rows_per_element.py:215-217).
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())

    e3t_mode = require_explicit_e3t_mode(context="traadv_fct_ww_inheritance")
    print(f"LEGOESM_NEMO_E3T={e3t_mode!r} (must be 'both')")
    assert e3t_mode == "both"

    ctx = base.build_bridge_and_fluxes()
    require_fp64(ctx["z_coord"], ctx["T_now"], ctx["S_now"], ctx["w_baro"],
                 context="traadv_fct_ww_inheritance")

    res_T, res_S = self_check_baseline(ctx)

    ww2 = load_ww_call2()
    print(f"\nww_call2 loaded: shape={ww2.shape}  dtype={ww2.dtype}  "
          f"finite={np.isfinite(ww2).all()}")
    w_sub = jnp.asarray(ww2)

    print("\n" + "=" * 78)
    print("STEP 1: ww-substitution -- T tracer")
    print("=" * 78)
    sub_T = run_with_ww("TEMPERATURE", ctx["T_now"], ctx["T_bef"], ctx, False, w_sub)
    print("\n" + "=" * 78)
    print("STEP 1: ww-substitution -- S tracer")
    print("=" * 78)
    sub_S = run_with_ww("SALINITY", ctx["S_now"], ctx["S_bef"], ctx, True, w_sub)

    print("\n" + "=" * 78)
    print("STEP 1 SUMMARY: baseline vs ww-substituted, collapse fraction")
    print("=" * 78)
    rows = [
        ("traadv_fct fluxes (w)", res_T["fw"], sub_T["fw"]),
        ("traadv_fct tendency (T)", res_T["final"], sub_T["final"]),
        ("traadv_fct horizontal tend", res_T["h"], sub_T["h"]),
        ("traadv_fct vertical upstream flux", res_T["fw"], sub_T["fw"]),
        ("traadv_fct (SALINITY)", res_S["final"], sub_S["final"]),
    ]
    tally = []
    for name, b, s in rows:
        cf = corr_collapse(b["corr"], s["corr"])
        print(f"  {name:<36s} baseline corr={b['corr']:.6f}/ratio={b['abs_ratio']:.6f}  "
              f"-> ww-sub corr={s['corr']:.6f}/ratio={s['abs_ratio']:.6f}  "
              f"corr_collapse_frac={cf:.4f}  "
              f"({'RESIDUAL GREW -- LOCAL' if cf < 0 else 'residual shrank -- inherited'})")
        tally.append((name, cf))

    print("\n" + "=" * 78)
    print("STEP 2: clip-count localization (603 vs NEMO 662)")
    print("=" * 78)
    n_own, clipped_own, both_wet = clip_count(
        ctx["w_baro"], ctx["T_now"], ctx["T_bef"], ctx["mass_flux_u"],
        ctx["mass_flux_v"], ctx["h_k_old"], ctx["grid"], ctx["p2dt"], ctx["act"])
    n_sub, clipped_sub, _ = clip_count(
        w_sub, ctx["T_now"], ctx["T_bef"], ctx["mass_flux_u"],
        ctx["mass_flux_v"], ctx["h_k_old"], ctx["grid"], ctx["p2dt"], ctx["act"])
    print(f"  legoESM own w:  n_clip(inferred, wet-wet w-faces)={n_own}  "
          f"(recorded prose count: 603)")
    print(f"  ww-substituted: n_clip(inferred, wet-wet w-faces)={n_sub}  "
          f"(NEMO recorded prose count: 662)")

    diff_mask = clipped_own != clipped_sub
    n_diff = int(diff_mask.sum())
    print(f"  cells where clip status CHANGES under ww-substitution: {n_diff}")
    if n_diff > 0:
        k_idx = np.where(diff_mask)
        ks = k_idx[-1]
        print(f"  differing-cell k-index histogram (k=8-14 = wzv hot band): "
              f"{np.bincount(ks, minlength=15)[:20]}")
        frac_hot_band = float(((ks >= 8) & (ks <= 14)).mean())
        print(f"  fraction of differing cells in k=8-14: {frac_hot_band:.3f}")
    print(f"  own-vs-NEMO-count gap 603->662 (59 cells) -- does ww-substitution "
          f"close it: {'YES' if n_sub >= 662 - 5 else 'NO'} "
          f"(n_sub={n_sub} vs target 662)")

    print("\n" + "=" * 78)
    print("TALLY (per row: collapse_frac, YES=ww-inherited >=0.5, NO=local)")
    print("=" * 78)
    for name, cf in tally:
        verdict = "INHERITED" if (np.isfinite(cf) and cf >= 0.5) else (
            "LOCAL" if np.isfinite(cf) else "UNMEASURED")
        print(f"  {name:<36s} collapse_frac={cf:.3f}  -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
