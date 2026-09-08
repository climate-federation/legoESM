"""CONFIRMATION EXPERIMENT for the barotropic-reconcile term (task Phase-4).

Controlled A/B at the arm2 state (all 7 faithful fixes), ONE variable =
``barotropic_reconcile_target`` (velocity_avg vs transport_avg). Reuses the
OFFICIAL 90-day gate (``acceptance_gate_90d``: metrics / classify / print_gate /
FLOORS / load_candidate / load_nemo_day90) VERBATIM -- no re-derived numerics,
no re-defined metric. That gate's metric bundle IS ``acc_thermal_wind`` (ACC =
acc_full, median over lons 2..-2; density = band meridional sigma contrast
up/deep; smax/smean = south surface sigma).

Prediction if the diagnosis is right (task statement):
  * |ACC(transport) - NEMO| recovers from ~1.708 toward <= ~1.557
  * density metrics (up/deep/smax/smean) do NOT degrade
Also decompose the ACC recovery (barotropic/baroclinic + longitude uniformity)
via acc_arm_diff_decomp's functions to check the ~100%-barotropic, near-uniform
signature.

If ACC does NOT recover -> diagnosis REFUTED. Report loudly; do not tune.

Diagnostic only. Run:
  LEGOESM_NEMO_E3T=both .venv/bin/python \
    scripts/validate/ocean_fidelity/dino_1226/reconcile_confirm.py \
    <arm_velocity.npz> <arm_transport.npz> [arm1_pre.npz]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import acceptance_gate_90d as G  # noqa: E402  official gate: metrics/classify/floors
import acc_arm_diff_decomp as D  # noqa: E402  bt/bc split + per-lon (reused)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    p_vel, p_tra = sys.argv[1], sys.argv[2]
    p_arm1 = sys.argv[3] if len(sys.argv) > 3 else None
    wet = G.A.tmask.astype(bool)

    print("=" * 78)
    print("BAROTROPIC-RECONCILE CONFIRMATION  (velocity_avg vs transport_avg), day 90")
    print("  same IC (arm2 = 7 fixes), one variable = barotropic_reconcile_target")
    print("  gate = acceptance_gate_90d (reduction: median-over-lons ACC + band")
    print("         sigma contrast); reference = NEMO DINO day-90 twin (tn/sn/un)")
    print("=" * 78)

    G.instrument_self_checks(wet)          # Rule 3: harness controls FIRST

    nemo_m = G.metrics(G.load_nemo_day90(), G.A.umask)

    # dtype gate (Rule 1c) + day-0 IC identity (Rule 7: one variable)
    dv = np.load(p_vel)
    print(f"[dtype] stored u3d_day90 = {np.asarray(dv['u3d_day90']).dtype} "
          f"-> f64 for metric; e3t1d {D.e3t1d.dtype}")
    uv0, _ = D.load_arm_u(p_vel, 0)
    ut0, _ = D.load_arm_u(p_tra, 0)
    d0 = float(np.max(np.abs(uv0 - ut0)))
    print(f"[C3] day-0 max|u_transport - u_velocity| = {d0:.3e} "
          f"({'OK bit-identical IC' if d0 == 0.0 else 'NONZERO -- IC CONFOUND!'})")

    # ---- OFFICIAL GATE table for each arm (Part 5) --------------------------
    cand = {"velocity_avg": G.metrics(G.load_candidate(p_vel), G.A.umask),
            "transport_avg": G.metrics(G.load_candidate(p_tra), G.A.umask)}
    if p_arm1:
        cand["arm1_pre(0fix)"] = G.metrics(G.load_candidate(p_arm1), G.A.umask)
    for tag, m in cand.items():
        print("\n" + "#" * 78)
        print(f"# ARM: {tag}")
        print("#" * 78)
        rows = G.classify(m, nemo_m, level=1)
        G.print_gate(rows, level=1, tag=f"[{tag}] ")

    # ---- headline: does ACC recover? ----------------------------------------
    acc_v = abs(cand["velocity_avg"]["acc"] - nemo_m["acc"])
    acc_t = abs(cand["transport_avg"]["acc"] - nemo_m["acc"])
    print("\n" + "=" * 78)
    print("HEADLINE  (ACC = acc_full, median over lons 2..-2 [Sv]; ref = NEMO d90)")
    print("=" * 78)
    print(f"  |ACC diff| velocity_avg  = {acc_v:.4f} Sv   (task arm2 baseline ~1.708)")
    print(f"  |ACC diff| transport_avg = {acc_t:.4f} Sv")
    recov = acc_v - acc_t
    verdict = "RECOVERS toward NEMO" if recov > 0 else "does NOT recover -> REFUTED"
    print(f"  recovery (velocity - transport) = {recov:+.4f} Sv  [{verdict}]")
    if p_arm1:
        acc_1 = abs(cand["arm1_pre(0fix)"]["acc"] - nemo_m["acc"])
        print(f"  arm1_pre |ACC diff| = {acc_1:.4f} Sv (task ~1.557); transport "
              f"{'reaches/passes arm1' if acc_t <= acc_1 + 1e-9 else 'still above arm1'}")

    # ---- density degradation check ------------------------------------------
    print("\n" + "-" * 78)
    print("DENSITY metrics -- do they degrade? (|diff vs NEMO|; smaller = better)")
    print("-" * 78)
    print(f"{'metric':<10}{'velocity':>12}{'transport':>12}{'change':>12}")
    for k in ("up", "deep", "smax", "smean"):
        dvk = abs(cand["velocity_avg"][k] - nemo_m[k])
        dtk = abs(cand["transport_avg"][k] - nemo_m[k])
        chg = dtk - dvk
        tag = "worse" if chg > 1e-6 else ("better" if chg < -1e-6 else "same")
        print(f"{k:<10}{dvk:12.5f}{dtk:12.5f}{chg:+12.5f}  {tag}")

    # ---- barotropic/baroclinic + longitude decomposition of the shift -------
    uv, lmv = D.load_arm_u(p_vel, 90)
    ut, lmt = D.load_arm_u(p_tra, 90)
    assert np.array_equal(lmv, lmt), "land_mask differs -- confound"
    wuv, wut = D.wet_u_of(lmv), D.wet_u_of(lmt)
    btv, bcv = D.barotropic_baroclinic_split(uv, wuv)
    btt, bct = D.barotropic_baroclinic_split(ut, wut)
    print("\n" + "-" * 78)
    print("SHIFT decomposition (transport_avg - velocity_avg) [Sv]")
    print("-" * 78)
    print(f"  barotropic {btt-btv:+.4f} | baroclinic {bct-bcv:+.4f}")
    tot = (btt - btv) + (bct - bcv)
    if abs(tot) > 1e-9:
        print(f"  barotropic fraction of the shift: {(btt-btv)/tot*100:.1f}% "
              f"(predicted ~100%)")
    ds = D.per_lon_section(ut, wut) - D.per_lon_section(uv, wuv)
    print(f"  longitude: median(2..-2) {np.median(ds[2:-2]):+.4f} | "
          f"mean {ds[2:-2].mean():+.4f} | std {ds[2:-2].std():.4f} "
          f"({'UNIFORM (transport-pathway)' if ds[2:-2].std() < 0.05 else 'localized'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
