#!/usr/bin/env python
"""Why the equatorial closure is starved: too little energy, or too much
stratification, or too short a length?

WHY.  ``equatorial_diffusivity_oracle.py`` measured the symptom: below roughly
12 m our vertical heat diffusivity sits on its background floor while NEMO's
stays three orders of magnitude above it, and the viscosity is starved by the
same factor.  Both being starved rules out a Prandtl PARTITION error and puts
the defect in the mixing coefficient itself.

A TKE closure builds that coefficient as ``K ~ C_k * l * sqrt(e)``, so exactly
three things can produce it, and they call for different fixes:

  * the turbulent energy ``e`` is too small   -> a PRODUCTION problem
    (surface input, shear production) ;
  * the stratification ``N2`` is too strong   -> a FEEDBACK
    (our own warm surface restratifying the column and shutting the closure
    off, which would make the bias self-sustaining) ;
  * the mixing length ``l`` is too short      -> a LENGTH-SCALE problem
    (the buoyancy length sqrt(2e)/N, its limiters, or the surface value).

This measures ``e`` and ``N2`` directly on both sides and reports the length
each side's own K implies, so the three are separated rather than ranked.

MATCHED, deliberately, and this is the whole point of the probe:

  * BOTH sides' N2 goes through the model's own
    ``compute_buoyancy_frequency_nemo_bn2`` (NEMO eosbn2.F90 bn2_t), so the
    stratification is one instrument, not two.  Reading NEMO's archived
    ``bn2`` instead would compare its convention against ours.
  * both sides use the SAME mesh, the same native frame, the same interface
    ladder and the same box.
  * the 5-day record is DERIVED from the day (D/5 - 1) and never defaulted.

LIMITS, stated rather than discovered later:
  * our side is an instantaneous snapshot and NEMO's a five-day mean, so the
    top ~10 m carries a diurnal-phase difference the band medians cannot
    remove.  Read the 15-65 m band, which has little diurnal signal.
  * an interface is used only where BOTH touching cells are wet: NEMO's fill
    values are read as T=0, S=0, which at a wet/dry interface differences real
    water against fresh and reads as violently unstable.
  * the implied length is a DIAGNOSIS of each side's own K and e, not a field
    either model stores; it is printed as a ratio, which is the only part of
    it that means anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
for _p in ("ocean", "core"):
    sys.path.insert(0, str(_HERE.parents[3] / "packages" / _p))

from global_tracer_content import (  # noqa: E402
    _native, load_mesh_depth_1d, load_mesh_latitude, load_mesh_longitude,
    load_mesh_metrics,
)
# The MATCHED pair, already written and already paired: our staggered u/v carry
# an extra face column/row, NEMO's carry none, and each needs its own transform
# onto T points. Comparing a face velocity against a cell-centre average is the
# exact staggering error that put a retracted "1.2 % agreement" into a merged
# PR, so both sides are moved to T points here rather than one of them.
from frozen_column_tke_twin import (  # noqa: E402
    centre_uv_collocated, centre_uv_extra_column,
)

#: Bands matched to the diffusivity probe so the two can be read side by side.
_BANDS = (("surface 15-65 m", 15.0, 65.0),
          ("entrainment 65-105 m", 65.0, 105.0),
          ("background >105 m", 105.0, 1.0e9))

#: NEMO zdftke rn_ediff; K_m = rn_ediff * l * sqrt(2e) in zdftke.F90.  Only the
#: RATIO of the two implied lengths is reported, so this cancels -- it is named
#: so the formula in the code is the closure's, not an invented one.
_C_K = 0.1


def mesh_w_1d(mesh_mask_path):
    """Interface depths and RAW w-cell thicknesses (never reconstructed)."""
    import netCDF4 as nc
    ds = nc.Dataset(mesh_mask_path)
    try:
        g = np.asarray(ds.variables["gdepw_1d"][:], dtype=np.float64).squeeze()
        if "e3w_1d" not in ds.variables:
            raise SystemExit(f"FATAL: {mesh_mask_path} carries no e3w_1d; the "
                             "N2 closure requires raw-mesh w thicknesses")
        e = np.asarray(ds.variables["e3w_1d"][:], dtype=np.float64).squeeze()
    finally:
        ds.close()
    return g, e


#: ORCA1 runs ``ln_teos10 = .true.`` (namelist_cfg:308) and the ORCA1 card
#: sets ``n2_eos_form="teos10"`` to match (run_omip_core2.py:987).  The bn2
#: routine's own default is ``"seos"``, the 3-term simplified equation of
#: state, so calling it WITHOUT this argument silently computes the
#: stratification on an equation of state NEITHER model runs.  That exact
#: omission was found and fixed once before in a sibling probe (2026-08-15)
#: and was reintroduced here; it is named as a constant so the next reader
#: sees the choice instead of inheriting a default.
_N2_EOS_FORM = "teos10"


def bn2(T, S, gdept, gdepw_int, e3w_int):
    """The MODEL'S OWN NEMO bn2 on the RUN'S equation of state (TEOS-10)."""
    import jax.numpy as jnp
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2
    return np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S),
        jnp.asarray(gdept), jnp.asarray(gdepw_int),
        eos_form=_N2_EOS_FORM,
        e3w_int=jnp.asarray(e3w_int)))


def shear2(u, v, e3w_int):
    """|du/dz|^2 + |dv/dz|^2 at interior interfaces, level-LAST arrays.

    Differenced across the SAME interfaces N2 uses and divided by the SAME raw
    e3w, so the two quantities land on one ladder and their ratio is a real
    Richardson number rather than two fields on neighbouring half-levels.

    Both sides go through this one function from their own u/v, so the C-grid
    staggering -- identical on our tripole and on NEMO's eORCA1 -- cancels.
    No rotation is applied: the box straddles no fold and the squared sum of
    the two components is invariant under the grid rotation anyway.

    Both sides arrive here on T POINTS, each through its own centring
    transform, so neither is a face velocity paired against a cell average.
    """
    dz = e3w_int[None, None, :]
    du = (u[..., :-1] - u[..., 1:]) / dz
    dv = (v[..., :-1] - v[..., 1:]) / dz
    return du * du + dv * dv


def band_median(field, wet, box, z_int, lo, hi):
    """Median over the box and depth band, wet interfaces only."""
    sel = (z_int >= lo) & (z_int < hi)
    if not sel.any():
        return float("nan"), 0
    vals = field[box][:, sel]
    ok = wet[box][:, sel] > 0.5
    if not ok.any():
        return float("nan"), 0
    return float(np.median(vals[ok])), int(ok.sum())


# NEMO zdftke nn_pdl=1: Pr = clamp(zri/ri_cri, 1, 10) with ri_cri deduced from
# the two namelist coefficients (zdftke.F90:772, :399).  Written as the
# closure's own arithmetic rather than the number 4.5 so the provenance of the
# slope is on the page.
_RN_EDIFF, _RN_EDISS = 0.1, 0.7
_RI_CRI = 2.0 / (2.0 + _RN_EDISS / _RN_EDIFF)
_PR_FLOOR, _PR_CEIL = 1.0, 10.0


def _band_vals(field, wet, box, z_int, lo, hi):
    """Wet values inside the box and depth band, flattened (no reduction)."""
    sel = (z_int >= lo) & (z_int < hi)
    if not sel.any():
        return np.empty(0)
    ok = wet[box][:, sel] > 0.5
    return field[box][:, sel][ok]


def _pr_from_ri(ri):
    return np.clip(ri / _RI_CRI, _PR_FLOOR, _PR_CEIL)


def _prandtl_check(n2_our, s2_our, n2_nemo, s2_nemo, z, nemo_K,
                   wet_i, box, gdepw_int):
    """Score each model's archived K_M/K_H against its own Prandtl mapping.

    Non-circular by construction: each side is tested against ITS OWN
    archived coefficients, so nothing here depends on the other model or on
    the equilibrium algebra the mapping came from.
    """
    out = {"ri_cri": _RI_CRI, "pr_slope": 1.0 / _RI_CRI, "bands": {}}
    have_ours = ("K_M_diag" in z) and ("K_H_diag" in z)
    kmo = kho = None
    if have_ours:
        kmo = np.transpose(_native(z["K_M_diag"]), (1, 2, 0))
        kho = np.transpose(_native(z["K_H_diag"]), (1, 2, 0))
    have_nemo = ("avm" in nemo_K) and ("avt" in nemo_K)

    print(f"\n[prandtl] Pr = clamp(Ri/{_RI_CRI:.4f}, {_PR_FLOOR:.0f}, "
          f"{_PR_CEIL:.0f}); each model scored on its OWN archived "
          f"coefficients.")
    print(f"{'band':22s} {'side':6s} {'Pr pred':>9s} {'Pr arch':>9s} "
          f"{'within1.5':>10s} {'N2<=0':>8s} {'at ceil':>8s} {'n':>7s}")

    for name, lo, hi in _BANDS:
        row = {}
        for side, n2, s2, km, kh in (
                ("ours", n2_our, s2_our, kmo, kho),
                ("nemo", n2_nemo, s2_nemo,
                 nemo_K.get("avm"), nemo_K.get("avt"))):
            if n2 is None or s2 is None or km is None or kh is None:
                continue
            if km.shape != n2.shape or kh.shape != n2.shape:
                print(f"{name:22s} {side:6s} SKIPPED: coefficient shape "
                      f"{km.shape}/{kh.shape} vs N2 {n2.shape}")
                continue
            vn2 = _band_vals(n2, wet_i, box, gdepw_int, lo, hi)
            vs2 = _band_vals(s2, wet_i, box, gdepw_int, lo, hi)
            vkm = _band_vals(km, wet_i, box, gdepw_int, lo, hi)
            vkh = _band_vals(kh, wet_i, box, gdepw_int, lo, hi)
            if vn2.size == 0:
                continue
            # Convective occupancy first: NEMO forces zri=0 (Pr=1) wherever
            # the BEFORE-level N2 is non-positive, so these interfaces are
            # governed by a different branch and are counted, not averaged in
            # silently.
            conv = float(np.mean(vn2 <= 0.0))
            pr_pred = _pr_from_ri(np.where(vn2 > 0.0,
                                           vn2 / np.maximum(vs2, 1e-12), 0.0))
            # Both Pr are strictly positive and bounded in [1, 10] by the
            # clamp, so a POINTWISE agreement fraction is well posed here --
            # unlike the unbounded coefficient ratios elsewhere in this file,
            # where a pointwise quotient is meaningless.
            pr_arch = vkm / np.maximum(vkh, 1e-30)
            good = np.isfinite(pr_arch) & (pr_arch > 0.0)
            agree = (float(np.mean(np.maximum(pr_arch[good] / pr_pred[good],
                                              pr_pred[good] / pr_arch[good])
                                   <= 1.5)) if good.any() else float("nan"))
            ceil_frac = float(np.mean(pr_pred >= _PR_CEIL - 1e-9))
            mp, ma = float(np.median(pr_pred)), (
                float(np.median(pr_arch[good])) if good.any() else float("nan"))
            row[side] = {"pr_pred_median": mp, "pr_arch_median": ma,
                         "frac_within_1p5": agree, "frac_n2_nonpos": conv,
                         "frac_pred_at_ceiling": ceil_frac,
                         "n_interfaces": int(vn2.size)}
            print(f"{name:22s} {side:6s} {mp:9.3f} {ma:9.3f} {agree:10.3f} "
                  f"{conv:8.3f} {ceil_frac:8.3f} {vn2.size:7d}")
        out["bands"][name] = row

    if not have_nemo:
        print("[prandtl] NEMO side SKIPPED: grid_W carries no avm/avt pair.")
    if not have_ours:
        print("[prandtl] our side SKIPPED: snapshot lacks K_M_diag/K_H_diag.")
    print("READ IT AS: a side whose archived ratio does NOT match its own "
          "predicted Pr has something other than the Prandtl mapping setting "
          "its heat mixing. A side that DOES match is faithful to its own "
          "formula, which moves the question upstream to the Richardson "
          "number itself.")
    print("THE N2<=0 COLUMN CANNOT BOUND THE CONVECTIVE ROUTE, and an earlier "
          "version of this text wrongly said it could. NEMO's N2 here is "
          "built from FIVE-DAY-MEAN temperature and salinity, while its "
          "enhanced-diffusion branch fires on INSTANTANEOUS unstable "
          "episodes that such a mean erases. This column reading 0.000 is "
          "therefore evidence about the mean state only, and is consistent "
          "with frequent firing; the archived diffusivity, not this column, "
          "is what settles it.")
    print("Our side is one INSTANT and NEMO's a five-day MEAN, so the two "
          "'arch' columns are NOT comparable to each other; each is only "
          "comparable to the 'pred' beside it.")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--nemo-gridt", required=True)
    p.add_argument("--nemo-day", type=float, required=True,
                   help="Day the 5-day record ENDS on; the record index is "
                        "D/5 - 1 and is never defaulted.")
    p.add_argument("--nemo-gridw", default=None,
                   help="Optional grid_W, for NEMO's archived avm/avt so the "
                        "implied mixing lengths can be compared.")
    p.add_argument("--nemo-gridu", default=None,
                   help="NEMO grid_U. With --nemo-gridv this adds the VERTICAL "
                        "SHEAR on both sides, computed the same way from each "
                        "side's own u/v, so the staggering cancels in the "
                        "ratio. Shear production is the dominant TKE source at "
                        "the equator, so this is what separates a production "
                        "failure from a defect inside the TKE equation.")
    p.add_argument("--nemo-gridv", default=None)
    p.add_argument("--mesh-mask", required=True)
    p.add_argument("--lat-halfwidth", type=float, default=2.0)
    p.add_argument("--lon-west", type=float, default=200.0)
    p.add_argument("--lon-east", type=float, default=260.0)
    p.add_argument("--out-json", default=None)
    p.add_argument("--tke-budget", action="store_true",
                   help="Per-band TKE budget from the snapshot: shear "
                        "production K_m*S^2, buoyancy destruction -K_h*N^2, "
                        "and vertical transport d/dz(K_m de/dz). Dissipation "
                        "is NOT computed -- the dissipation length is not "
                        "stored, and reconstructing it would assume the "
                        "mixing-length relation this is meant to test -- so it "
                        "is reported as the RESIDUAL, with the storage term "
                        "the snapshot cannot see bounded and printed beside "
                        "it.")
    p.add_argument("--plot-out", default=None,
                   help="PNG of the per-interface profile. Drawn from the SAME "
                        "numbers the table prints, in the same function, so a "
                        "figure can never disagree with its own table.")
    p.add_argument("--prandtl-check", action="store_true",
                   help="Score EACH model's archived viscosity/diffusivity "
                        "ratio against the Prandtl mapping the ORCA1 card "
                        "runs, on that model's OWN state. Three independent "
                        "questions, none of which needs the other model: "
                        "(1) does our archived K_M/K_H equal what our own "
                        "formula predicts from our own N2 and shear -- an "
                        "implementation self-check that a coding defect would "
                        "fail; (2) does the SAME formula reproduce NEMO's "
                        "archived avm/avt on NEMO's own fields -- if not, "
                        "something other than the Prandtl mapping is setting "
                        "the oracle's heat mixing; (3) how far apart are the "
                        "card's plain gradient-Richardson argument and NEMO's "
                        "viscosity-weighted one on our snapshot. Convective "
                        "occupancy (the fraction of interfaces with N2<=0, "
                        "where NEMO forces Pr=1) is printed beside each, "
                        "because a five-day mean hides those episodes.")
    a = p.parse_args()

    if a.nemo_day <= 0 or a.nemo_day % 5 != 0:
        raise SystemExit(f"--nemo-day wants a positive multiple of 5 (the "
                         f"output is 5-day means), got {a.nemo_day}")
    rec = int(a.nemo_day // 5) - 1

    _, _, e3t, tmask = load_mesh_metrics(a.mesh_mask)
    gdept = load_mesh_depth_1d(a.mesh_mask)
    gdepw, e3w = mesh_w_1d(a.mesh_mask)
    gdepw_int, e3w_int = gdepw[1:], e3w[1:]          # interior interfaces
    lat = load_mesh_latitude(a.mesh_mask)
    lon = load_mesh_longitude(a.mesh_mask) % 360.0
    box = (np.abs(lat) <= a.lat_halfwidth) & (lon >= a.lon_west) & (lon <= a.lon_east)
    if not box.any():
        raise SystemExit("no cells in the requested box")
    print(f"box |lat| <= {a.lat_halfwidth}, {a.lon_west:g}-{a.lon_east:g}E "
          f"-> {int(box.sum())} columns; NEMO record {rec} (day {a.nemo_day:g})")

    z = np.load(a.snapshot)
    for k in ("T", "S", "tke"):
        if k not in z:
            raise SystemExit(f"FATAL: {a.snapshot} lacks '{k}'. TKE is written "
                             "by the tripole lane only; this probe refuses to "
                             "substitute a reconstruction for the stored field.")
    T_our = np.transpose(_native(z["T"]), (1, 2, 0))       # (nj, ni, nlev)
    S_our = np.transpose(_native(z["S"]), (1, 2, 0))
    tke_our = np.transpose(_native(z["tke"]), (1, 2, 0))   # (nj, ni, nlev-1)

    import netCDF4 as nc
    ds = nc.Dataset(a.nemo_gridt)
    try:
        def pick(names, what):
            for n in names:
                if n in ds.variables:
                    return n
            raise SystemExit(f"FATAL: no {what} in {a.nemo_gridt}")
        tn = pick(("to", "bigthetao", "thetao", "votemper"), "3-D temperature")
        sn = pick(("so", "so_abs", "vosaline"), "3-D salinity")
        print(f"[nemo] T={tn} S={sn}")
        def v(n):
            return np.ma.filled(np.ma.masked_invalid(ds.variables[n][rec]),
                                0.0).astype(np.float64)
        T_nemo = np.transpose(v(tn), (1, 2, 0))
        S_nemo = np.transpose(v(sn), (1, 2, 0))
    finally:
        ds.close()
    if T_nemo.shape != T_our.shape:
        raise SystemExit(f"NEMO {T_nemo.shape} vs ours {T_our.shape}")

    n2_our = bn2(T_our, S_our, gdept, gdepw_int, e3w_int)
    n2_nemo = bn2(T_nemo, S_nemo, gdept, gdepw_int, e3w_int)

    s2_our = s2_nemo = None
    if a.nemo_gridu and a.nemo_gridv:
        for k in ("u", "v"):
            if k not in z:
                raise SystemExit(f"FATAL: {a.snapshot} lacks '{k}'; the shear "
                                 "comparison needs both velocity components")
        u_c, v_c = centre_uv_extra_column(z["u"], z["v"])
        s2_our = shear2(np.transpose(_native(u_c), (1, 2, 0)),
                        np.transpose(_native(v_c), (1, 2, 0)), e3w_int)
        def _vel(path, names, what):
            d = nc.Dataset(path)
            try:
                nm = next((n for n in names if n in d.variables), None)
                if nm is None:
                    raise SystemExit(f"FATAL: no {what} in {path}; looked for "
                                     + ", ".join(names))
                print(f"[nemo] {what}={nm}")
                arr = np.ma.filled(
                    np.ma.masked_invalid(d.variables[nm][rec]), 0.0)
            finally:
                d.close()
            return np.transpose(np.asarray(arr, dtype=np.float64), (1, 2, 0))
        un, vn = centre_uv_collocated(
            _vel(a.nemo_gridu, ("uo", "vozocrtx", "uoce"), "u"),
            _vel(a.nemo_gridv, ("vo", "vomecrty", "voce"), "v"))
        s2_nemo = shear2(un, vn, e3w_int)
        if s2_nemo.shape != s2_our.shape:
            raise SystemExit(f"NEMO shear {s2_nemo.shape} vs ours {s2_our.shape}")

    # Both cells touching an interface must be wet, or a wet/dry interface
    # differences real water against NEMO's T=0/S=0 fill and reads unstable.
    wet_c = (tmask > 0.5)                                   # (nlev, nj, ni)
    wet_i = np.transpose(wet_c[:-1] & wet_c[1:], (1, 2, 0))  # (nj, ni, nlev-1)

    nemo_K = {}
    if a.nemo_gridw:
        dsw = nc.Dataset(a.nemo_gridw)
        try:
            for nm in ("avm", "avt"):
                if nm in dsw.variables:
                    arr = np.ma.filled(
                        np.ma.masked_invalid(dsw.variables[nm][rec]), 0.0)
                    nemo_K[nm] = np.transpose(np.asarray(arr, dtype=np.float64),
                                              (1, 2, 0))[..., 1:]
        finally:
            dsw.close()
        print(f"[nemo] grid_W carries: {sorted(nemo_K) or 'neither avm nor avt'}")

    report = {"snapshot": str(a.snapshot), "nemo_gridt": str(a.nemo_gridt),
              "nemo_record": rec, "nemo_day": a.nemo_day,
              "n2_eos_form": _N2_EOS_FORM, "bands": {}}
    print(f"[n2] equation of state: {_N2_EOS_FORM} (the run's and ORCA1's; the "
          f"bn2 routine defaults to seos, which is neither)")
    print(f"\n{'band':22s} {'N2 ours':>10s} {'N2 NEMO':>10s} {'xN2':>6s}"
          f" {'e ours':>10s} {'sqrt(e) x':>10s} {'n':>8s}")
    for name, lo, hi in _BANDS:
        n2o, n = band_median(n2_our, wet_i, box, gdepw_int, lo, hi)
        n2n, _ = band_median(n2_nemo, wet_i, box, gdepw_int, lo, hi)
        eo, _ = band_median(tke_our, wet_i, box, gdepw_int, lo, hi)
        rn2 = n2o / n2n if n2n else float("nan")
        row = {"n2_ours": n2o, "n2_nemo": n2n, "n2_ratio": rn2,
               "tke_ours": eo, "n_interfaces": n}
        print(f"{name:22s} {n2o:10.3e} {n2n:10.3e} {rn2:6.2f}"
              f" {eo:10.3e} {'':>10s} {n:8d}")
        if s2_our is not None:
            so, _ = band_median(s2_our, wet_i, box, gdepw_int, lo, hi)
            sn, _ = band_median(s2_nemo, wet_i, box, gdepw_int, lo, hi)
            rs2 = so / sn if sn else float("nan")
            row.update({"shear2_ours": so, "shear2_nemo": sn,
                        "shear2_ratio": rs2,
                        "ri_ours": n2o / so if so else float("nan"),
                        "ri_nemo": n2n / sn if sn else float("nan")})
            print(f"{'':22s} shear^2 {so:10.3e} {sn:10.3e} (x{rs2:.2f})"
                  f"   Ri {row['ri_ours']:8.3f} {row['ri_nemo']:8.3f}")
        report["bands"][name] = row

    if s2_our is not None:
        print("\nSHEAR: a ratio near 0.25 (half the velocity difference) makes "
              "the collapse SELF-CONSISTENT -- weak current, weak production, "
              "weak coefficient, weaker current. A ratio near 1 puts the "
              "defect INSIDE the TKE equation, because production is then "
              "available and the energy still is not there.")
        print("Ri here is N2/shear^2 from the two MEASURED fields on one "
              "ladder; it is not the Prandtl inversion and the two are "
              "independent estimates of the same number.")
    print("\nREAD IT AS: N2 ratio near 1 exonerates stratification, so a "
          "starved K is then an ENERGY or LENGTH problem. A ratio well above 1 "
          "means our own warm surface is restratifying the column and shutting "
          "the closure off -- a feedback, and the bias would be self-sustaining.")
    print("Our snapshot is INSTANTANEOUS and NEMO's record a five-day MEAN; "
          "read the 15-65 m band, not the surface one.")
    if not nemo_K:
        print("NEMO's TKE is not archived in the 5-day output, so `e` cannot be "
              "compared directly; ours is printed for the length arithmetic.")

    if a.prandtl_check:
        report["prandtl"] = _prandtl_check(
            n2_our, s2_our, n2_nemo, s2_nemo, z, nemo_K,
            wet_i, box, gdepw_int)

    # ---- per-interface profile: WHERE do the two energy profiles part? ----
    # The energy ratio is built so the closure constant CANCELS: with
    # l = sqrt(2e)/N the coefficient goes as K ~ e/N, so
    #     e_ours/e_NEMO = (avm_ours/avm_NEMO) * (N_ours/N_NEMO).
    # Nothing here is inferred from a tuning coefficient, and NEMO's TKE never
    # has to be archived.
    if "avm" in nemo_K and "K_M_diag" in z:
        avm_o = np.transpose(_native(z["K_M_diag"]), (1, 2, 0))
        avm_n = nemo_K["avm"]
        if avm_o.shape != avm_n.shape:
            print(f"[profile] SKIPPED: our avm {avm_o.shape} vs NEMO "
                  f"{avm_n.shape}")
        else:
            # The HEAT side is carried alongside because the tracer
            # diffusivity, not the viscosity, is what the temperature bias
            # actually rides on -- and it was previously reconstructed by
            # dividing a point viscosity by a whole-band Prandtl median,
            # which mixes two different reductions.  Both are measured
            # directly here.
            avt_o = (np.transpose(_native(z["K_H_diag"]), (1, 2, 0))
                     if "K_H_diag" in z else None)
            avt_n = nemo_K.get("avt")
            heat = (avt_o is not None and avt_n is not None
                    and avt_o.shape == avm_o.shape
                    and avt_n.shape == avm_n.shape)
            print(f"\n{'depth':>8s} {'e ours':>11s} {'avm ours':>11s} "
                  f"{'avm NEMO':>11s} {'e ours/NEMO':>12s}"
                  + (f" {'avt ours':>11s} {'avt NEMO':>11s} {'avt x':>8s}"
                     if heat else ""))
            top = gdepw_int <= 120.0
            prof = []
            for k in np.nonzero(top)[0]:
                col = lambda f: band_median(f, wet_i, box, gdepw_int,
                                            gdepw_int[k] - 1e-9,
                                            gdepw_int[k] + 1e-9)[0]
                ao, an = col(avm_o), col(avm_n)
                to = col(avt_o) if heat else float("nan")
                tn = col(avt_n) if heat else float("nan")
                no, nn = col(n2_our), col(n2_nemo)
                # Built from the BAND MEDIANS, not as the median of a pointwise
                # ratio: the pointwise form goes NaN in any column where N2 is
                # momentarily negative, and a single NaN makes the median NaN.
                # The first version of this table printed NaN through the whole
                # 9-65 m band -- the exact band the question is about -- and
                # that was my instrument, not the ocean.
                er = ((ao / an) * np.sqrt(no / nn)
                      if (an and nn > 0 and no > 0) else float("nan"))
                # Both band medians on the SAME reduction, so this ratio is a
                # like-for-like comparison of the two heat diffusivities.
                tr = to / tn if (heat and tn) else float("nan")
                print(f"{gdepw_int[k]:8.1f} {col(tke_our):11.3e} "
                      f"{ao:11.3e} {an:11.3e} {er:12.3e}"
                      + (f" {to:11.3e} {tn:11.3e} {tr:8.4f}" if heat else ""))
                prof.append((gdepw_int[k], col(tke_our), ao, an, er, to, tn))
            report["profile"] = [
                {"depth_m": d, "tke_ours": e, "avm_ours": ao, "avm_nemo": an,
                 "e_ratio": er, "avt_ours": to, "avt_nemo": tn}
                for d, e, ao, an, er, to, tn in prof]
            if heat:
                print("The avt columns are the HEAT diffusivity, measured on "
                      "both sides with the same reduction -- this is the "
                      "coefficient the temperature bias rides on, and the "
                      "viscosity beside it is not a substitute for it.")
            if a.plot_out and prof:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                d = [r[0] for r in prof]
                fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 5.2),
                                              sharey=True)
                ax.semilogx([r[2] for r in prof], d, "o-", label="ours")
                ax.semilogx([r[3] for r in prof], d, "s-", label="NEMO")
                ax.set_xlabel("vertical viscosity [m$^2$/s]")
                ax.set_ylabel("depth [m]")
                ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend()
                ax.set_title("the coefficient")
                ax2.semilogx([r[4] for r in prof], d, "o-", color="crimson")
                ax2.axvline(1.0, color="k", lw=0.8)
                ax2.set_xlabel("turbulent energy, ours / NEMO")
                ax2.grid(alpha=0.3)
                ax2.set_title("1 = agreement")
                fig.suptitle("equatorial Pacific, day 30: the surface value is "
                             "right, the decay is not")
                fig.tight_layout()
                Path(a.plot_out).parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(a.plot_out, dpi=140)
                print(f"[plot] {a.plot_out}")
            print("A ratio that is already small at the TOP interface is a "
                  "SURFACE SOURCE problem; one that starts near 1 and falls "
                  "with depth is a TRANSPORT or DISSIPATION problem. That is "
                  "the whole question this profile answers.")

    if a.tke_budget:
        if s2_our is None:
            raise SystemExit("--tke-budget needs the shear, so it needs "
                             "--nemo-gridu and --nemo-gridv")
        if "K_H_diag" not in z or "K_M_diag" not in z:
            raise SystemExit("--tke-budget needs K_M_diag and K_H_diag; re-run "
                             "the model with --kprofile-snapshots")
        km = np.transpose(_native(z["K_M_diag"]), (1, 2, 0))
        kh = np.transpose(_native(z["K_H_diag"]), (1, 2, 0))
        prod = km * s2_our                      # shear production  [m2/s3]
        buoy = -kh * n2_our                     # buoyancy work (sink if N2>0)
        # Vertical transport of TKE, d/dz( K_m de/dz ), on the SAME interface
        # ladder. NEMO diffuses TKE with avm itself, so K_m is the right
        # coefficient; the ladder is one shorter at each end, so the two
        # outermost interfaces are left out rather than one-sided-differenced.
        dz = e3w_int
        # The face coefficient is NEMO's max(avm(k)+avm(k+1), 2e-5) halved,
        # NOT the plain average. Read out of the model's own tridiagonal
        # assembly (tke.py, `avm_min` = 2.0e-5, zdftke.F90:503,505). It
        # matters here: our avm at these depths is 3e-6 to 6e-6, so the SUM
        # sits BELOW the floor and the real coefficient is the floor -- two to
        # three times what a plain average gives. The first version of this
        # term omitted the floor and understated transport by that factor.
        _AVM_SUM_FLOOR = 2.0e-5
        kface = 0.5 * np.maximum(km[..., :-1] + km[..., 1:], _AVM_SUM_FLOOR)
        flux = kface * (tke_our[..., :-1] - tke_our[..., 1:]) / \
            (0.5 * (dz[:-1] + dz[1:]))[None, None, :]
        trans = np.full_like(tke_our, np.nan)
        trans[..., 1:-1] = (flux[..., :-1] - flux[..., 1:]) / dz[None, None, 1:-1]
        # DISSIPATION from the CLOSURE'S OWN mixing-length routine, not from
        # the buoyancy relation. Codex's first objection was that inverting
        # K ~ e/N assumes the active length IS the buoyancy length; calling
        # compute_mixing_lengths instead runs the real code path, sweeps and
        # limiters included, so the length is measured rather than assumed and
        # eps = c_eps * e^(3/2) / l_eps stops being a residual.
        #
        # The card is rebuilt from the run's RECORDED command line, never
        # retyped: --tke-mxl-choice 3, --tke-surface-bc nemo_dirichlet,
        # --tke-prognostic, --tke-kappa-convention veros_sqrte, --iwm.
        eps_model = None
        try:
            import jax.numpy as jnp
            sys.path.insert(0, str(_HERE.parents[3] / "scripts" / "run"))
            from run_omip_core2 import orca1_zdftke_config
            from legoesm.ocean.physics.vertical_mixing.tke import (
                compute_mixing_lengths,
            )
            vcfg = orca1_zdftke_config(
                iwm_enabled=True, surface_bc="nemo_dirichlet", mxl_choice=3,
                prognostic=True, kappa_convention="veros_sqrte")
            tcfg = getattr(vcfg, "tke", vcfg)
            # dz_cell is REQUIRED for mxl_choice 3: the routine refuses
            # without it rather than silently skipping NEMO's |dl/dz| <= e3t
            # bounding sweeps, which is exactly the limiter codex said could
            # not be assumed away. Reference thicknesses (e3t_0, partial cells
            # included); the z-star dilation by eta is neglected, which is at
            # most a percent on these levels and is stated rather than hidden.
            l_k, l_eps = compute_mixing_lengths(
                jnp.asarray(tke_our), jnp.asarray(n2_our),
                jnp.asarray(e3w_int), tcfg,
                dz_cell=jnp.asarray(np.transpose(e3t, (1, 2, 0))))
            l_eps = np.asarray(l_eps)
            eps_model = (tcfg.c_eps * tke_our ** 1.5
                         / np.maximum(l_eps, tcfg.mxl_min))
            print(f"[closure] l_eps and dissipation rebuilt with the model's "
                  f"own compute_mixing_lengths (c_eps={tcfg.c_eps}, "
                  f"mxl_min={tcfg.mxl_min}). NO surface wind anchor is passed "
                  f"(taum is not in the snapshot), so the top few interfaces "
                  f"are not faithful; read the 15-65 m band.")
        except Exception as _exc:   # pragma: no cover - config-dependent
            print(f"[closure] SKIPPED rebuilding l_eps: "
                  f"{type(_exc).__name__}: {_exc}")

        omega = 2.0 * np.pi / 86400.0
        print(f"\n{'band':22s} {'production':>12s} {'buoyancy':>12s} "
              f"{'transport':>12s} {'residual':>12s} {'|de/dt| max':>12s}"
              + ("" if eps_model is None else
                 f" {'eps model':>12s} {'l_eps m':>9s} {'closure':>12s}"))
        for name, lo, hi in _BANDS:
            pr, _ = band_median(prod, wet_i, box, gdepw_int, lo, hi)
            bu, _ = band_median(buoy, wet_i, box, gdepw_int, lo, hi)
            tr, _ = band_median(np.nan_to_num(trans, nan=0.0), wet_i, box,
                                gdepw_int, lo, hi)
            eo, _ = band_median(tke_our, wet_i, box, gdepw_int, lo, hi)
            line = (f"{name:22s} {pr:12.3e} {bu:12.3e} {tr:12.3e} "
                    f"{pr + bu + tr:12.3e} {omega * eo:12.3e}")
            row = {"production": pr, "buoyancy": bu, "transport": tr,
                   "residual_is_dissipation": pr + bu + tr,
                   "storage_bound": omega * eo}
            if eps_model is not None:
                ep, _ = band_median(eps_model, wet_i, box, gdepw_int, lo, hi)
                le, _ = band_median(l_eps, wet_i, box, gdepw_int, lo, hi)
                # CLOSURE TEST: P + B + T - eps should be ~0 in a steady
                # column. If it is not, the reconstruction is wrong and NO
                # number from it may be quoted.
                resid = pr + bu + tr - ep
                line += f" {ep:12.3e} {le:9.3f} {resid:12.3e}"
                row.update({"eps_from_closure": ep, "l_eps_m": le,
                            "closure_residual": resid})
            print(line)
            report["bands"][name].update(row)
        print("The residual is production + buoyancy + transport, which in a "
              "steady column equals the DISSIPATION. The column is not steady "
              "-- TKE has a strong diurnal cycle -- but the storage term is "
              "bounded by omega*e, printed in the last column (GLM). Where the "
              "residual exceeds that bound by a wide margin, its magnitude and "
              "sign hold regardless of the diurnal phase.")
        if eps_model is not None:
            print("CLOSURE column = P + B + T - eps. Near zero means every "
                  "term is now measured and the budget closes; a large value "
                  "means the reconstruction is wrong and nothing from it may "
                  "be quoted. Read it BEFORE reading anything else here.")
        print("TRANSPORT is the term nobody has measured yet, and it is the "
              "one that decides between too little energy arriving from above "
              "and too much being destroyed in place.")

    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(report, indent=2))
        print(f"[report] {a.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
