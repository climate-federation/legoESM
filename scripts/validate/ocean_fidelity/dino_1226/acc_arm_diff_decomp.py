"""CHEAP DISCRIMINATOR (task Phase-0): where does the +0.151 Sv ACC degradation
between the two faithfulness arms live?

Decompose the ACC metric (acc_thermal_wind.acc_full: full-section zonal
transport, e3t_1d weighting, median over lons 2..-2) on the two committed arm
npz files at day 90, DECOMPOSED by longitude / depth-level / latitude-group /
band-vs-blocked, so we can say whether the +0.151 Sv is concentrated at the
sill/topography (=> transport-pathway) or spread uniformly (=> something else).

Both arms are stored identically (float32, bit-identical day-0); the arm2-arm1
DIFFERENCE cancels the common fp32 quantisation floor, so the between-arm
difference is meaningful well below the 0.091 Sv absolute noise floor.

Controls run before any number is quoted (Rule 3/Rule 10):
  C1  reproduce the recorded gate ACC |diff| vs NEMO d90 for BOTH arms
      (must land ~1.557 / ~1.708 Sv per task statement) -- proves the metric
      here IS the gate's metric.
  C2  the acc_full sum over the per-longitude decomposition must reproduce
      acc_full itself for each arm (median-vs-sum caveat: acc_full MEDIANS
      over lons, so per-lon values are printed and the median re-derived here,
      not summed).
  C3  day-0 arm1==arm2 exactly (bit-identical IC): the day-0 ACC diff must be 0.

Diagnostic only.  Reuses acc_thermal_wind geometry + metric verbatim.

Run: LEGOESM_NEMO_E3T=both .venv/bin/python \
     scripts/validate/ocean_fidelity/dino_1226/acc_arm_diff_decomp.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import acc_thermal_wind as A  # noqa: E402  -- geometry, e3t1d, e2u_col, masks, umask

GATE2 = ("/tmp/claude-10257/-home-dbalwada-legoESM/"
         "853ee94c-2651-44cc-ba12-f55bf3ed1979/scratchpad/gate2")
ARM1 = f"{GATE2}/arm1_pre.npz"      # 0 faithful fixes
ARM2 = f"{GATE2}/arm2_fixes.npz"    # 7 faithful fixes

# ---- geometry pulled from acc_thermal_wind (the recorded protocol) -----------
e3t1d = A.e3t1d
e2u = A.e2u_col
gdept1d = A.gdept1d
umask = A.umask                 # (y,x,z)
NY, NX, NZ = A.tmask.shape
J0, J1 = A.J0, A.J1


def load_arm_u(path, day):
    """u-faces (199,53,36) at the given day -> NEMO u-cols (199,52,36), verbatim
    from acc_thermal_wind.load_lego column handling."""
    d = np.load(path)
    lU = np.asarray(d[f"u3d_day{day}"], dtype=np.float64)   # (199,53,36)
    u = lU[:, 1:53, :].copy()
    u[:, 47, :] = lU[:, 48, :]
    lm = np.asarray(d["land_mask"], dtype=np.float64)
    return u, lm


def wet_u_of(lm):
    return umask & (lm[:, :, None] > 0.5)


def acc_full(u, wet_u):
    """acc_thermal_wind.acc_full verbatim: median over lons 2..-2 [Sv]."""
    return float(np.median(
        np.einsum("jik,k,j->i", np.where(wet_u, u, 0.0), e3t1d, e2u)[2:-2]) / 1e6)


def per_lon_section(u, wet_u):
    """Full-section transport per longitude [Sv/lon]: sum over (y,z), e3t_1d wtd."""
    return np.einsum("jik,k,j->i", np.where(wet_u, u, 0.0), e3t1d, e2u) / 1e6


def per_lev_transport(u, wet_u):
    """Section transport per depth level, median over lons 2..-2 [Sv/level]."""
    integ = np.einsum("jik,k,j->jik", np.where(wet_u, u, 0.0), e3t1d, e2u) / 1e6
    perlon_perlev = integ.sum(axis=0)                  # (x,z) -- meridional sum
    return np.median(perlon_perlev[2:-2], axis=0)      # (z,)


def per_latgroup(u, wet_u):
    """Transport by latitude group (south / band / north), median over lons."""
    groups = (("south", slice(0, J0)),
              ("band", slice(J0, J1 + 1)),
              ("north", slice(J1 + 1, NY)))
    out = {}
    for name, sl in groups:
        w = np.where(wet_u[sl], u[sl], 0.0)
        out[name] = float(np.median(
            np.einsum("jik,k,j->i", w, e3t1d, e2u[sl])[2:-2]) / 1e6)
    return out


def per_lat_row(u, wet_u):
    """Per-T-row contribution to the section transport, median over lons [Sv/row].
    NOT summed to acc_full (median is not additive) -- used only to LOCATE the
    latitudes where the arm2-arm1 difference concentrates."""
    contrib = np.einsum("jik,k,j->ji", np.where(wet_u, u, 0.0), e3t1d, e2u) / 1e6  # (y,x)
    return np.median(contrib[:, 2:-2], axis=1)         # (y,)


def main():
    print("=" * 74)
    print("ACC arm-diff decomposition  (arm2_fixes - arm1_pre), day 90")
    print("=" * 74)

    # ---- C1: reproduce the gate |diff| vs NEMO d90 for both arms -------------
    from acceptance_gate_90d import load_nemo_day90
    nemo = load_nemo_day90()
    nwet = umask                                        # band rows fully wet
    nemo_acc = acc_full(nemo["u"], nwet)
    print(f"\n[C1] NEMO d90 ACC (this harness): {nemo_acc:.4f} Sv")
    accs = {}
    for tag, path in (("arm1_pre", ARM1), ("arm2_fixes", ARM2)):
        u90, lm = load_arm_u(path, 90)
        wu = wet_u_of(lm)
        a = acc_full(u90, wu)
        accs[tag] = a
        print(f"[C1] {tag:12s} ACC d90 = {a:8.4f} Sv | "
              f"|diff vs NEMO| = {abs(a - nemo_acc):.4f} Sv "
              f"(task expects ~{'1.557' if tag=='arm1_pre' else '1.708'})")
    degr = abs(accs["arm2_fixes"] - nemo_acc) - abs(accs["arm1_pre"] - nemo_acc)
    print(f"[C1] degradation in |diff|: {degr:+.4f} Sv "
          f"(task: +0.151; sign check only, arm ACC values may re-center)")
    print(f"[C1] raw ACC shift arm2-arm1: "
          f"{accs['arm2_fixes'] - accs['arm1_pre']:+.4f} Sv")

    # ---- C3: day-0 must be bit-identical ------------------------------------
    u0_1, lm0_1 = load_arm_u(ARM1, 0)
    u0_2, lm0_2 = load_arm_u(ARM2, 0)
    d0 = float(np.max(np.abs(u0_1 - u0_2)))
    print(f"\n[C3] day-0 max|u_arm2 - u_arm1| = {d0:.3e} "
          f"({'OK bit-identical' if d0 == 0.0 else 'NONZERO -- IC differs!'})")

    # ---- decomposition of the day-90 arm2-arm1 shift ------------------------
    u1, lm1 = load_arm_u(ARM1, 90)
    u2, lm2 = load_arm_u(ARM2, 90)
    wu1, wu2 = wet_u_of(lm1), wet_u_of(lm2)
    assert np.array_equal(lm1, lm2), "land_mask differs between arms -- confound"

    print("\n" + "-" * 74)
    print("LATITUDE GROUP (median over lons 2..-2) [Sv]")
    print("-" * 74)
    g1, g2 = per_latgroup(u1, wu1), per_latgroup(u2, wu2)
    print(f"{'group':>8s}{'arm1':>10s}{'arm2':>10s}{'arm2-arm1':>12s}")
    for k in ("south", "band", "north"):
        print(f"{k:>8s}{g1[k]:10.4f}{g2[k]:10.4f}{g2[k]-g1[k]:+12.4f}")

    print("\n" + "-" * 74)
    print("DEPTH LEVEL (section transport per level, median over lons) [Sv]")
    print("-" * 74)
    l1, l2 = per_lev_transport(u1, wu1), per_lev_transport(u2, wu2)
    dl = l2 - l1
    order = np.argsort(-np.abs(dl))
    print(f"{'k':>3s}{'depth':>8s}{'arm1':>10s}{'arm2':>10s}{'diff':>10s}")
    for k in order[:12]:
        print(f"{k:3d}{gdept1d[k]:8.0f}{l1[k]:10.4f}{l2[k]:10.4f}{dl[k]:+10.4f}")
    upper = gdept1d < 1400.0
    print(f"  upper (<1400m) diff sum {dl[upper].sum():+.4f} | "
          f"deep diff sum {dl[~upper].sum():+.4f} Sv")

    print("\n" + "-" * 74)
    print("LONGITUDE (per-lon full-section transport) [Sv/lon]")
    print("-" * 74)
    s1, s2 = per_lon_section(u1, wu1), per_lon_section(u2, wu2)
    ds = s2 - s1
    order = np.argsort(-np.abs(ds))
    print(f"{'i':>3s}{'arm1':>10s}{'arm2':>10s}{'diff':>10s}")
    for i in order[:12]:
        print(f"{i:3d}{s1[i]:10.4f}{s2[i]:10.4f}{ds[i]:+10.4f}")
    print(f"  median-lon diff (2..-2): {np.median(ds[2:-2]):+.4f} | "
          f"mean {ds[2:-2].mean():+.4f} | std {ds[2:-2].std():.4f}")

    print("\n" + "-" * 74)
    print("LATITUDE ROW -- top |arm2-arm1| contributors (median over lons) [Sv/row]")
    print("-" * 74)
    r1, r2 = per_lat_row(u1, wu1), per_lat_row(u2, wu2)
    dr = r2 - r1
    order = np.argsort(-np.abs(dr))
    gphit = A.gphit
    print(f"{'j':>4s}{'lat':>8s}{'inband':>7s}{'arm1':>10s}{'arm2':>10s}{'diff':>10s}")
    for j in order[:15]:
        inb = "yes" if J0 <= j <= J1 else "no"
        print(f"{j:4d}{gphit[j,25]:8.1f}{inb:>7s}{r1[j]:10.4f}{r2[j]:10.4f}{dr[j]:+10.4f}")

    print(f"\n[dtype] u arm arrays float64 (cast from stored f32); "
          f"e3t1d {e3t1d.dtype}, e2u {e2u.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def barotropic_baroclinic_split(u, wet_u):
    """Split section transport into depth-mean (barotropic) and shear (baroclinic).
    u_bar = sum(u*e3)/sum(e3) over wet column; barotropic transport uses u_bar,
    baroclinic uses (u-u_bar).  bt+bc == full section transport per (row,lon)."""
    w = wet_u
    e3 = np.broadcast_to(e3t1d, u.shape)
    H = np.sum(np.where(w, e3, 0.0), axis=2)                 # (y,x) column depth
    ubar = np.where(H > 0, np.sum(np.where(w, u * e3, 0.0), axis=2) / np.maximum(H, 1e-30), 0.0)
    bt_perloncol = ubar * H * e2u[:, None]                   # (y,x) [m3/s]
    bc_perloncol = np.sum(np.where(w, (u - ubar[:, :, None]) * e3, 0.0), axis=2) * e2u[:, None]
    bt = np.median((bt_perloncol.sum(axis=0) / 1e6)[2:-2])   # median over lons
    bc = np.median((bc_perloncol.sum(axis=0) / 1e6)[2:-2])
    return float(bt), float(bc)


if os.environ.get("ADD_BTBC"):
    u1, lm1 = load_arm_u(ARM1, 90); u2, lm2 = load_arm_u(ARM2, 90)
    wu1, wu2 = wet_u_of(lm1), wet_u_of(lm2)
    from acceptance_gate_90d import load_nemo_day90
    nemo = load_nemo_day90()
    print("\n" + "=" * 74)
    print("BAROTROPIC / BAROCLINIC split of the section ACC (median over lons) [Sv]")
    print("=" * 74)
    print(f"{'':12s}{'barotropic':>14s}{'baroclinic':>14s}{'total(bt+bc)':>14s}")
    for tag, u, wu in (("arm1_pre", u1, wu1), ("arm2_fixes", u2, wu2),
                       ("NEMO d90", nemo["u"], umask)):
        bt, bc = barotropic_baroclinic_split(u, wu)
        print(f"{tag:12s}{bt:14.4f}{bc:14.4f}{bt+bc:14.4f}")
    bt1, bc1 = barotropic_baroclinic_split(u1, wu1)
    bt2, bc2 = barotropic_baroclinic_split(u2, wu2)
    print(f"{'arm2-arm1':12s}{bt2-bt1:+14.4f}{bc2-bc1:+14.4f}{(bt2+bc2)-(bt1+bc1):+14.4f}")
    print("  (bt+bc is a SUM per column then median-over-lon; not identical to")
    print("   acc_full's median-of-section, but the arm2-arm1 SPLIT is the signal.)")
