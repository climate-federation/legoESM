#!/usr/bin/env python
"""#1226 GM-bolus non-divergence hypothesis, ladder A/B (steps 1-3 only, per
task brief -- MEASUREMENT ONLY, no fix).

HYPOTHESIS (coordinator's): the GM bolus transport (u_eiv, v_eiv) is the curl
of a streamfunction psi = kappa_GM * slope (``nemo_eiv_bolus_transport``,
gm_redi_latlon_cgrid.py:1655-1729) and is therefore non-divergent BY
CONSTRUCTION only if the metrics used to build psi are consistent with the
divergence operator that later consumes it
(``add_bolus_to_advecting_flux``, ocean_model_latlon_cgrid.py:114-176, which
calls ``divergence_cgrid`` at :166 and re-diagnoses w via
``diagnose_w_from_flux_div`` at :175). On NEMO's true 3-D e3t ladder,
thicknesses vary 9-13% between adjacent abyssal levels and (the hypothesis
says) this consistency breaks, giving the bolus a spurious divergent
component that explains the restart-start instability (addenda 33/35/36).

PRODUCTION DIVERGENCE OPERATOR USED (traced, not re-derived -- Rule 0):
``divergence_cgrid`` at
packages/core/legoesm/grids/operators_latlon_cgrid.py:846-975. Confirmed as
the SAME function object ``add_bolus_to_advecting_flux`` calls at
ocean_model_latlon_cgrid.py:166 (`from legoesm.ocean.dynamics.
latlon_cgrid_operators import divergence_cgrid` at :64-67 there, itself a
bare re-export of the core function -- `grep -n "divergence_cgrid" ...
latlon_cgrid_operators.py:51`, one name, one implementation). This probe
calls THAT function directly on the isolated bolus transport (linearity of
divergence: div(mass_flux + bolus) = div(mass_flux) + div(bolus), so
isolating the bolus's own divergence is not a re-derivation, just dropping
the additive Eulerian term) -- NOT a freshly-written divergence, closing the
"self-consistent-but-meaningless" trap this campaign has hit before.

``divergence_cgrid`` takes TRANSPORT [m^3/s] on cell-face staggering and
returns a per-level 2-D horizontal divergence [m/s] (`(net_zonal+net_merid)/
area`, area is horizontal-only, :970-974) -- confirming the thickness is
carried IN the transport (via psi's vertical difference), not applied by
this operator, matching ``diagnose_w_from_flux_div``'s
``thickness_weighted=True`` contract (vertical.py:900-901,917-918).

Harness reused verbatim from ``ldf_slp_per_element.py`` (``build_state()``,
``wet_u_mask``/``wet_v_mask``, RUN_GDB/kt=57601 restart pairing) and
``eiv_transport_walk.py`` (bolus-rebuild pattern, live-vs-static ladder A/B
via the ``jacobian=`` toggle) -- Rule 0, no re-derivation of restart loading
or slope/kappa numerics.

Run::

    cd /home/dbalwada/legoESM && CUDA_VISIBLE_DEVICES="" JAX_ENABLE_X64=1 \\
      .venv/bin/python \\
      scripts/validate/ocean_fidelity/dino_1226/bolus_nondivergence_ladder_ab.py
"""
from __future__ import annotations

import importlib.util
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp

# scripts/ is not a package -- import the sibling templates by path (same
# idiom every dino_1226 probe uses), Rule 0.
_sib_path = os.path.join(os.path.dirname(__file__), "ldf_slp_per_element.py")
_spec = importlib.util.spec_from_file_location("_ldf_slp_per_element", _sib_path)
_slp = importlib.util.module_from_spec(_spec)
sys.modules["_ldf_slp_per_element"] = _slp
_spec.loader.exec_module(_slp)

build_state = _slp.build_state
wet_u_mask = _slp.wet_u_mask
wet_v_mask = _slp.wet_v_mask
RUN_DIR = _slp.RUN_DIR
RESTART = _slp.RESTART

from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    compute_nemo_native_slopes,
    compute_treguier_kappa_gm_nemo_native,
    nemo_eiv_bolus_transport,
)
from legoesm.grids.operators_latlon_cgrid import divergence_cgrid
from legoesm.ocean.fidelity.precision_gate import require_fp64, require_explicit_e3t_mode
from legoesm.grids.latlon import ensure_geometry


def build_bolus_and_div(st, *, live_ladder: bool):
    """Rebuild wslpi/wslpj + kappa_GM + bolus transport, toggling ONLY the
    ``jacobian`` argument passed to the two production slope/kappa builders
    (same one-variable toggle as eiv_transport_walk.py:161-190, reused
    verbatim) -- then run the bolus faces through the PRODUCTION
    ``divergence_cgrid`` (the same call ``add_bolus_to_advecting_flux``
    makes at ocean_model_latlon_cgrid.py:166).

    Returns (u_eiv, v_eiv, div_bolus, wslpi, wslpj, kappa_t) with div_bolus
    the per-level 2-D horizontal divergence [m/s] of the bolus TRANSPORT
    alone (mass_flux terms are additive/linear in divergence_cgrid and are
    NOT included -- isolating the bolus is what the hypothesis is about).
    """
    mask2d = jnp.asarray(st["mask"].astype(st["T"].dtype))
    dtype = st["T"].dtype
    gcfg = st["gm_cfg"]
    _treg = getattr(gcfg, "treguier", None)
    if _treg is None or not getattr(_treg, "enabled", False):
        raise ValueError("GMRediConfig.treguier must be enabled for this row")

    geom = ensure_geometry(st["grid"])
    e2u = geom.dy_u[:, 1:]
    e1v = geom.dx_v[1:, :]
    u_mask = jnp.asarray(st["u_mask"])
    v_mask = jnp.asarray(st["v_mask"])
    act = st["active"]
    act_j = jnp.asarray(act.astype(dtype))
    act_below = jnp.concatenate(
        [act_j[:, :, 1:], jnp.zeros_like(act_j[:, :, :1])], axis=2)

    _jac = st["jacobian"] if live_ladder else None
    uslp, vslp, wslpi, wslpj = compute_nemo_native_slopes(
        st["rho"], st["T"], st["S"], mask2d, u_mask, v_mask,
        st["z_coord"], st["grid"], gcfg, st["eos_fn"],
        rho_0=st["rho_0"], g=st["g"], active_3d=st["active_3d"],
        jacobian=_jac,
    )
    f_coriolis = jnp.broadcast_to(jnp.asarray(st["grid"].f), mask2d.shape)
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

    # --- exact production wiring: cell-indexed E/N faces -> staggered face
    # arrays -- reproduced verbatim from add_bolus_to_advecting_flux
    # (ocean_model_latlon_cgrid.py:150-163), the ONLY reshaping this probe
    # performs before calling the production divergence_cgrid. No mass_flux
    # term is added (linearity - see module docstring), and u_mask_3d_tracer/
    # v_mask_3d_tracer collapse to the plain 2-D face mask broadcast for a
    # pure z* column (ocean_model_latlon_cgrid.py:3550-3552, the branch this
    # DINO card takes -- no partial cells), reused identically here.
    bolus_mfu = jnp.concatenate([u_eiv[:, -1:, :], u_eiv], axis=1)
    bolus_mfv = jnp.concatenate([v_eiv[-1:, :, :], v_eiv], axis=0)
    u_mask_3d_tracer = u_mask[:, :, jnp.newaxis]
    v_mask_3d_tracer = v_mask[:, :, jnp.newaxis]
    bolus_mfu = bolus_mfu * u_mask_3d_tracer
    bolus_mfv = bolus_mfv * v_mask_3d_tracer

    div_bolus = divergence_cgrid(bolus_mfu, bolus_mfv, st["grid"])

    return (np.asarray(u_eiv), np.asarray(v_eiv), np.asarray(div_bolus),
            np.asarray(wslpi), np.asarray(wslpj),
            np.asarray(uslp), np.asarray(vslp))


def rms(x, w):
    xm = x[w]
    if xm.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(xm.astype(np.float64) ** 2)))


def column_integral_report(label, div, wet_t, area, dz_ref):
    """Column-integrated |div(bolus transport)| stats + vertical profile.

    div is [m/s] per level (divergence_cgrid's native units); multiplying by
    area recovers the TRANSPORT divergence [m^3/s] at that level, and summing
    over k gives the column-integrated transport divergence [m^3/s] -- the
    quantity the non-divergence hypothesis is about (net volume source/sink
    a column would see if this flux were applied uncorrected).
    """
    nlev = div.shape[-1]
    col = np.full(div.shape[:2], np.nan)
    wet_col = wet_t.any(axis=-1)
    # column integral: sum_k div[k] * area (area is horizontal-only, k-const)
    col_vals = np.nansum(np.where(wet_t, div, 0.0), axis=-1) * area
    col[wet_col] = col_vals[wet_col]
    finite = np.isfinite(col) & wet_col
    absvals = np.abs(col[finite])
    print(f"  [{label}] column-integrated div(bolus transport) [m^3/s], "
          f"n_wet_columns={finite.sum()}")
    if absvals.size == 0:
        print("    NO WET COLUMNS -- cannot report")
        return None
    print(f"    median|.|={np.median(absvals):.6e}  "
          f"p95|.|={np.percentile(absvals, 95):.6e}  "
          f"max|.|={absvals.max():.6e}")

    # vertical profile of |div| (per-level, not column-integrated): median
    # over wet (i,j) at each k, so "concentrated below ~2000m" is checkable.
    print(f"    per-level median|div| (per-level, [m/s]; k=0 surface):")
    prof = []
    depth = np.cumsum(dz_ref) - 0.5 * dz_ref  # cell-centre depth, 1-D ladder
    for k in range(nlev):
        wk = wet_t[:, :, k]
        if not wk.any():
            prof.append(np.nan)
            continue
        prof.append(float(np.median(np.abs(div[:, :, k][wk]))))
    for k in range(0, nlev, 4):
        chunk = ", ".join(f"k={kk}(~{depth[kk]:.0f}m):{prof[kk]:.3e}"
                           for kk in range(k, min(k + 4, nlev))
                           if np.isfinite(prof[kk]))
        if chunk:
            print(f"      {chunk}")
    return dict(col=col, absvals=absvals, profile=np.array(prof), depth=depth)


def slope_cap_report(label, uslp, vslp, wslpi, wslpj, wet_u, wet_v, wet_w,
                      depth_w, slpmax):
    """Step-3 fallback: abyssal (>2000m) slope population + cap-hit fraction."""
    below = depth_w > 2000.0
    caphit_frac = {}
    for name, arr, wet in (("uslp", uslp, wet_u), ("vslp", vslp, wet_v),
                            ("wslpi", wslpi, wet_w), ("wslpj", wslpj, wet_w)):
        nlev = arr.shape[-1]
        sel = wet & below[:nlev]
        vals = np.abs(arr[sel])
        if vals.size == 0:
            caphit_frac[name] = (float("nan"), 0)
            continue
        frac = float(np.mean(vals >= slpmax * (1.0 - 1e-9)))
        caphit_frac[name] = (frac, int(vals.size))
        print(f"  [{label}] {name} below 2000m: n={vals.size}  "
              f"median|slope|={np.median(vals):.4e}  "
              f"p95|slope|={np.percentile(vals, 95):.4e}  "
              f"max|slope|={vals.max():.4e}  "
              f"cap_hit_frac(>=rn_slpmax={slpmax:g})={frac:.4f}")
    return caphit_frac


def main() -> int:
    e3t_mode = require_explicit_e3t_mode(context="bolus non-divergence ladder A/B")
    if e3t_mode not in ("off", "both"):
        raise ValueError(
            f"this probe compares 'off' vs 'both' specifically; "
            f"LEGOESM_NEMO_E3T={e3t_mode!r} is neither")
    print(f"restart used  = {os.path.join(RUN_DIR, RESTART)}  (kt=57601, "
          f"DINO_00057600_restart.nc -> read_nemo_restart_before)")
    print(f"dump dir used = {RUN_DIR}")
    print(f"LEGOESM_NEMO_E3T (this process) = {e3t_mode} "
          f"(the A/B below builds BOTH branches in-process via the "
          f"jacobian= toggle, matching eiv_transport_walk.py's method, so "
          f"the env var only fixes the geometry used for wet-mask/grid "
          f"metadata -- both 'off' and 'both' arms are covered here in one "
          f"run since build_bolus_and_div's live_ladder flag IS the ladder "
          f"toggle)")

    st = build_state()
    require_fp64(st["z_coord"], st["T"], st["S"], context="bolus non-div ladder A/B")

    act = st["active"]
    wet_u = wet_u_mask(act, st["u_mask"])
    wet_v = wet_v_mask(act, st["v_mask"])
    wet_t = act  # column mask for the column-integral report
    dz_ref = np.asarray(st["z_coord"].dz_ref)
    area = np.asarray(ensure_geometry(st["grid"]).area)

    print("\n" + "=" * 78)
    print("STEP 1 -- column-integrated |div(bolus transport)|, OFF vs BOTH ladder")
    print("=" * 78)

    u_off, v_off, div_off, wslpi_off, wslpj_off, uslp_off, vslp_off = build_bolus_and_div(st, live_ladder=False)
    u_both, v_both, div_both, wslpi_both, wslpj_both, uslp_both, vslp_both = build_bolus_and_div(st, live_ladder=True)

    # [self-check 1/2] the two ladder branches must NOT be trivially identical
    # -- else the A/B toggles nothing (same guard eiv_transport_walk.py uses).
    ladder_inert = np.array_equal(wslpi_off, wslpi_both) and np.array_equal(u_off, u_both)
    print(f"[self-check 1/2] static(off) vs live(both) bolus bit-identical: "
          f"{ladder_inert}  (expect False -- else the ladder toggle is INERT "
          f"and this A/B tests nothing)")
    if ladder_inert:
        print("*** FAILURE: ladder toggle is inert -- aborting, the A/B below "
              "would be meaningless ***")
        return 1

    # [self-check 2/2] force the two ladder branches to use the SAME jacobian
    # (i.e. call build_bolus_and_div(live_ladder=True) TWICE) and confirm
    # bit-identical output -- proves the harness itself introduces no
    # incidental nondeterminism/RNG/ordering difference, so any OFF-vs-BOTH
    # difference measured below is attributable ONLY to the ladder.
    u_both_repeat, v_both_repeat, div_both_repeat, _, _, _, _ = build_bolus_and_div(st, live_ladder=True)
    repeat_identical = np.array_equal(div_both, div_both_repeat) and np.array_equal(u_both, u_both_repeat)
    print(f"[self-check 2/2] live(both) rebuilt twice, bit-identical: "
          f"{repeat_identical}  (expect True -- confirms the ladder is the "
          f"ONLY variable; a mismatch here would mean the harness itself is "
          f"non-deterministic and no A/B conclusion could be drawn)")
    if not repeat_identical:
        print("*** FAILURE: harness is non-deterministic -- aborting ***")
        return 1

    rep_off = column_integral_report("OFF (1-D ladder, wrong e3t_1d)", div_off, wet_t, area, dz_ref)
    rep_both = column_integral_report("BOTH (true 3-D e3t_0 ladder)", div_both, wet_t, area, dz_ref)

    # relative scale: RMS of the bolus transport itself and of a plausible
    # Eulerian transport scale (h*u ~ dz_ref * area^0.5 * O(0.1 m/s) is not
    # available without building the full model step here, so report the
    # bolus transport's own RMS at u/v faces as the honest available scale --
    # a column-integrated divergence should be judged against the TRANSPORT
    # magnitude that produced it).
    u_rms_both = rms(u_both, wet_u[:, :, :u_both.shape[-1]])
    v_rms_both = rms(v_both, wet_v[:, :, :v_both.shape[-1]])
    u_rms_off = rms(u_off, wet_u[:, :, :u_off.shape[-1]])
    v_rms_off = rms(v_off, wet_v[:, :, :v_off.shape[-1]])
    print(f"\n  RMS(u_eiv) [m^3/s]: off={u_rms_off:.6e}  both={u_rms_both:.6e}  "
          f"ratio(both/off)={u_rms_both / u_rms_off if u_rms_off else float('nan'):.4f}")
    print(f"  RMS(v_eiv) [m^3/s]: off={v_rms_off:.6e}  both={v_rms_both:.6e}  "
          f"ratio(both/off)={v_rms_both / v_rms_off if v_rms_off else float('nan'):.4f}")

    if rep_off is not None and rep_both is not None:
        scale_off = 0.5 * (u_rms_off + v_rms_off)
        scale_both = 0.5 * (u_rms_both + v_rms_both)
        print(f"\n  median column|div| / RMS(bolus face transport):")
        print(f"    off:  {np.median(rep_off['absvals']) / scale_off if scale_off else float('nan'):.4e}")
        print(f"    both: {np.median(rep_both['absvals']) / scale_both if scale_both else float('nan'):.4e}")
        print(f"\n  max column|div| / RMS(bolus face transport):")
        print(f"    off:  {rep_off['absvals'].max() / scale_off if scale_off else float('nan'):.4e}")
        print(f"    both: {rep_both['absvals'].max() / scale_both if scale_both else float('nan'):.4e}")
        below2k_off = rep_off['profile'][rep_off['depth'][:len(rep_off['profile'])] > 2000.0]
        below2k_both = rep_both['profile'][rep_both['depth'][:len(rep_both['profile'])] > 2000.0]
        above2k_off = rep_off['profile'][rep_off['depth'][:len(rep_off['profile'])] <= 2000.0]
        above2k_both = rep_both['profile'][rep_both['depth'][:len(rep_both['profile'])] <= 2000.0]
        print(f"\n  vertical concentration check (median per-level |div|, "
              f"nanmedian above vs below 2000m):")
        print(f"    off:  above2000m={np.nanmedian(above2k_off):.4e}  "
              f"below2000m={np.nanmedian(below2k_off):.4e}")
        print(f"    both: above2000m={np.nanmedian(above2k_both):.4e}  "
              f"below2000m={np.nanmedian(below2k_both):.4e}")

    step1_verdict_off_roundoff = (rep_off is not None and
                                   rep_off['absvals'].max() < 1e-6 * max(u_rms_off, v_rms_off, 1e-30))
    step1_verdict_both_material = (rep_both is not None and
                                    rep_both['absvals'].max() > 1e-3 * max(u_rms_both, v_rms_both, 1e-30))

    if step1_verdict_off_roundoff and step1_verdict_both_material:
        print("\n>>> STEP 1 pattern consistent with HYPOTHESIS (off~roundoff, "
              "both material) -- proceeding to STEP 2 localisation is the "
              "next step (not run automatically; see report).")
        return 0

    print("\n" + "=" * 78)
    print("STEP 3 (triggered: STEP 1 did not show off~roundoff/both~material) "
          "-- abyssal slope population + rn_slpmax cap-hit fraction")
    print("=" * 78)
    gcfg = st["gm_cfg"]
    # cfg field is S_max (gm_redi_latlon_cgrid.py:1015 `slpmax = cfg.S_max`,
    # NEMO's rn_slpmax) -- not re-derived, read straight off the config the
    # production slope builder itself reads.
    slpmax = float(gcfg.S_max)
    print(f"  S_max / rn_slpmax (card) = {slpmax}")

    depth_w = np.cumsum(dz_ref)  # w-point (interface) depth, 1-D ladder, k=0 is first interior interface
    wet_w = _slp.wet_w_mask(act)

    print("\n  -- OFF ladder (wrong 1-D e3t_1d) --")
    slope_cap_report("OFF", uslp_off, vslp_off, wslpi_off, wslpj_off,
                      wet_u, wet_v, wet_w, depth_w, slpmax)
    print("\n  -- BOTH ladder (true 3-D e3t_0) --")
    slope_cap_report("BOTH", uslp_both, vslp_both, wslpi_both, wslpj_both,
                      wet_u, wet_v, wet_w, depth_w, slpmax)

    return 2 if not (step1_verdict_off_roundoff and step1_verdict_both_material) else 0


if __name__ == "__main__":
    raise SystemExit(main())
