#!/usr/bin/env python
"""ORACLE-FRAME replay of the port's d_sw3: is the 1.44e-7 S12 kee edge
signal a FORMULATION difference inside d_sw3 (ytp_v/xtp_u), or
input-noise amplification?

The stage ladder (compare_dyncore_stages) showed, under the FROZEN
basis-derived B transform, edge lines of ~5e-7 rel on the d_sw3
advected m/s pair (ubbtemp/vbb) feeding the S12 kee scalar at 1.44e-7,
on oracle tiles 1/2/5 only, while every d_sw3 INPUT (u/v at S02, uc/vc
at S07, the Courant pair ubb/vbbtemp) is clean at ~1e-14.  Two
mechanisms fit; this probe discriminates them:

  H-FORM  the port's ytp_v/xtp_u differ from the oracle's in some
          edge-row term.  Prediction: replaying the PORT d_sw3 on
          ORACLE-EXACT inputs (this probe: oracle dumps in, oracle
          metric dumps in, NO face mapping at all -- everything stays
          in the oracle tile frame) reproduces the ~5e-7 edge lines
          against the oracle's own S10 ubbtemp/vbb dumps.
  H-AMP   the routines are identical and the port-lane diff comes from
          ~1e-14 input differences flipped through the upwind c>0
          branch where the advective Courant number is a numerical
          zero (J&W symmetry).  Prediction: this replay matches the
          oracle at ~1e-13, and the flagged edge strips carry
          near-zero Courant selectors (|vbbtemp| / |ubb| ~ roundoff)
          with an O(1e-5) donor-branch gap.

Instrument controls (run before the verdict is believed):
  C1  grid identity: the metric dump dir (extchain) and the stage dump
      dir (dynstage) must carry BITWISE-identical AG/GR lon/lat per
      tile -- else the metrics do not belong to the dumped state.
  C2  input sensitivity (teeth): perturb one v halo row by a known
      relative eps on one tile and confirm the replayed ubbtemp North
      strip moves by O(eps * |v|) -- the baseline row max is printed
      first (a perturbed zero is not a control).
  C3  NaN anywhere in a replayed output is FATAL.

Everything is offline CPU on existing dumps; no Fortran is rebuilt.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_cmp():
    spec = importlib.util.spec_from_file_location(
        "compare_dyncore_stages",
        os.path.join(_HERE, "compare_dyncore_stages.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def edge_strip_maxima(d: np.ndarray, width: int = 3) -> dict:
    """Max |d| on the four edge strips and the interior of a 2-D/3-D
    array (leading two axes are the spatial ones)."""
    ni, nj = d.shape[0], d.shape[1]
    out = {"W": float(np.abs(d[:width]).max()),
           "E": float(np.abs(d[ni - width:]).max()),
           "S": float(np.abs(d[:, :width]).max()),
           "N": float(np.abs(d[:, nj - width:]).max())}
    out["interior"] = float(
        np.abs(d[width:ni - width, width:nj - width]).max())
    out["edge_max"] = max(out["W"], out["E"], out["S"], out["N"])
    return out


def _pad_embed(arr: np.ndarray, full_shape: tuple, off: int) -> np.ndarray:
    """Embed a compute-window dump into a NaN-filled padded array at
    offset ``off`` on both axes (fort arrays are origin-isd)."""
    out = np.full(full_shape, np.nan)
    out[off:off + arr.shape[0], off:off + arr.shape[1]] = arr
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", required=True,
                    help="dynstage run dir (dyncore_stage_t*.mf/.dat)")
    ap.add_argument("--metrics-dir", required=True,
                    help="extchain run dir (extchain_t*.mf/.dat with "
                         "the M_* metric families)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--perturb", action="store_true",
                    help="H-AMP mechanism test: rerun the replay with "
                         "amplitude-matched roundoff noise (the measured "
                         "port-lane input level) on u/v/uc/vc and report "
                         "the OUTPUT diff vs the unperturbed replay, at "
                         "1x and 0.1x noise plus a gate-free hord_mt=2 "
                         "variant.  PREREGISTERED: a smooth mechanism "
                         "scales the max diff linearly with the noise "
                         "and survives hord 2; a discontinuous limiter-"
                         "gate/donor flip keeps a ~1e-5-class max at "
                         "0.1x and collapses at hord 2.")
    ap.add_argument("--perturb-rel", type=float, default=3e-14,
                    help="noise amplitude as a fraction of each field's "
                         "max |value| (default: the measured S02/S07 "
                         "port-lane relative level)")
    ap.add_argument("--perturb-seed", type=int, default=20260811)
    args = ap.parse_args(argv)

    CMP = _load_cmp()
    from legoesm.core.fv3_native_duo_sw_core import d_sw3_duo
    from legoesm.core.fv3_native_sw_core import Bounds

    N, NG = CMP.N, CMP.NG
    bd = Bounds.single_tile(N, NG)
    npx = N + 1
    m_a = N + 2 * NG            # 54
    m_b = m_a + 1               # 55
    ring_off = NG               # index of i=is in a padded array

    stage, metric, notes0 = [], [], None
    for t in range(1, 7):
        f, notes, _ = CMP.read_tile(args.dump_dir, t)
        stage.append(f)
        if notes0 is None:
            notes0 = notes
        fm, _, _ = CMP.read_tile(args.metrics_dir, t, prefix="extchain_t")
        metric.append(fm)
    for ln in notes0 or []:
        print(ln)

    # dt of one acoustic substep from the NOTE lines, not hardcoded.
    kv = {}
    for ln in notes0:
        if "dt_atmos" in ln:
            kv["dt_atmos"] = float(ln.split("=")[1].split()[0])
        if "nsplit" in ln:
            parts = ln.split("=")[1].split()
            kv["nsplit"], kv["ksplit"] = int(parts[3]), int(parts[4])
    dt_sub = kv["dt_atmos"] / kv["ksplit"] / kv["nsplit"]
    print(f"dt_sub = {dt_sub} s  (dt_atmos {kv['dt_atmos']}, nsplit "
          f"{kv['nsplit']}, ksplit {kv['ksplit']})")

    # ---- C1: the two dump dirs describe the SAME grid, bitwise -------
    worst_grid = 0.0
    for t in range(6):
        for nm in ("AG_LON", "AG_LAT", "GR_LON", "GR_LAT"):
            worst_grid = max(worst_grid, float(np.abs(
                stage[t][nm] - metric[t][nm]).max()))
    if worst_grid != 0.0:
        raise SystemExit(
            f"C1 GRID-IDENTITY CONTROL FAILED: stage and metric dumps "
            f"disagree on tile lon/lat by {worst_grid:.3e} (must be "
            f"bitwise 0) -- the metric dumps do not belong to this "
            f"deck; refusing to replay.")
    print("C1 grid-identity control: stage vs metric dump lon/lat "
          "bitwise identical (24 tile-fields)")

    def build_gs(t: int) -> dict:
        m = metric[t]
        if m["M_RSINA"].shape != (npx, npx):
            raise SystemExit(f"tile {t+1}: M_RSINA shape "
                             f"{m['M_RSINA'].shape} != {(npx, npx)}")
        return {
            "cosa": np.asarray(m["M_COSA"], dtype=np.float64),
            "rsina": _pad_embed(np.asarray(m["M_RSINA"], np.float64),
                                (m_b, m_b), ring_off),
            "dx": np.asarray(m["M_DX"], dtype=np.float64),
            "rdx": np.asarray(m["M_RDX"], dtype=np.float64),
            "dy": np.asarray(m["M_DY"], dtype=np.float64),
            "rdy": np.asarray(m["M_RDY"], dtype=np.float64),
        }

    def replay_tile(t: int, v_override=None) -> dict:
        s = stage[t]
        gs = build_gs(t)
        km = s["S02_extuv_u"].shape[2]
        outs = {nm: np.empty((npx, npx, km)) for nm in
                ("ubbtemp", "vbbtemp", "ubb", "vbb")}
        for k in range(km):
            v_in = s["S02_extuv_v"][:, :, k]
            if v_override is not None and k == 0:
                v_in = v_override
            got = d_sw3_duo(s["S02_extuv_u"][:, :, k], v_in,
                            s["S07_extucvc_uc"][:, :, k],
                            s["S07_extucvc_vc"][:, :, k],
                            gs, bd, npx, npx, dt=dt_sub, hord_mt=6)
            for nm in outs:
                outs[nm][:, :, k] = got[nm]
        return outs

    # ---- the replay -------------------------------------------------
    rows = []
    print("\nORACLE-FRAME REPLAY: port d_sw3 on oracle-exact inputs vs "
          "the oracle's own S10 dumps (rel = |diff|/max|oracle field|)")
    print(f"{'tile':>4s} {'field':>8s} {'interior':>10s} {'W':>10s} "
          f"{'E':>10s} {'S':>10s} {'N':>10s} {'scale':>10s}")
    worst_cells = []
    for t in range(6):
        outs = replay_tile(t)
        for nm in ("ubbtemp", "vbbtemp", "ubb", "vbb"):
            ref = stage[t][f"S10_dsw23_{nm}"]
            if np.isnan(outs[nm]).any():
                raise SystemExit(f"C3: NaN in replayed {nm} tile {t+1}")
            d = outs[nm] - ref
            scale = float(np.abs(ref).max())
            sm = edge_strip_maxima(d)
            rows.append({"tile": t + 1, "field": nm, "scale": scale,
                         **{k: v / scale for k, v in sm.items()}})
            print(f"{t+1:4d} {nm:>8s} "
                  + " ".join(f"{sm[k]/scale:10.2e}"
                             for k in ("interior", "W", "E", "S", "N"))
                  + f" {scale:10.3g}")
            # per-cell census of the worst offenders, with the upwind
            # Courant selector at that B point (vbbtemp for ytp_v's
            # ubbtemp, ubb for xtp_u's vbb -- both dumped pre-barrier).
            if sm["edge_max"] / scale > 1e-9:
                sel = stage[t]["S10_dsw23_"
                               + ("vbbtemp" if nm == "ubbtemp" else
                                  "ubb" if nm == "vbb" else nm)]
                flat = np.argsort(np.abs(d).ravel())[::-1][:6]
                for fi in flat:
                    i, j, k = np.unravel_index(fi, d.shape)
                    worst_cells.append(
                        (t + 1, nm, int(i) + 1, int(j) + 1, int(k),
                         float(d[i, j, k]), float(sel[i, j, k])))

    if worst_cells:
        print("\nWORST REPLAY CELLS (tile, field, i, j, k, diff, "
              "upwind selector c at that B point):")
        for rec in worst_cells[:24]:
            t_, nm_, i_, j_, k_, d_, c_ = rec
            print(f"  t{t_} {nm_:8s} ({i_:2d},{j_:2d},k{k_}) "
                  f"diff={d_:+.3e}  c={c_:+.3e}")

    # ---- zero-Courant census on the edge strips ----------------------
    print("\nZERO-COURANT CENSUS (upwind selector magnitude on the "
          "B-ring edge strips; a ~1e-13 selector means the c>0 branch "
          "is decided by roundoff there):")
    for t in range(6):
        for nm, sel_nm in (("ubbtemp", "S10_dsw23_vbbtemp"),
                           ("vbb", "S10_dsw23_ubb")):
            sel = np.abs(stage[t][sel_nm])
            sm = edge_strip_maxima(sel)
            mins = {"W": float(sel[:3].min()),
                    "E": float(sel[-3:].min()),
                    "S": float(sel[:, :3].min()),
                    "N": float(sel[:, -3:].min())}
            n_tiny = int((sel[[0, -1], :] < 1e-9).sum()
                         + (sel[:, [0, -1]] < 1e-9).sum())
            print(f"  t{t+1} {nm:8s} strip |c| min "
                  + " ".join(f"{k}={mins[k]:.1e}" for k in "WESN")
                  + f"  outer-ring cells |c|<1e-9: {n_tiny}")

    # ---- C2: input-sensitivity mutation (teeth) ----------------------
    t0 = 0
    v0 = np.array(stage[t0]["S02_extuv_v"][:, :, 0], copy=True)
    # v is fort (isd:ied+1, jsd:jed); perturb the fort j = npy+1 halo
    # row, which the ytp_v North edge block reads directly.
    row_idx = (npx + 1) - bd.jsd
    base_row = float(np.abs(v0[:, row_idx]).max())
    print(f"\nC2 mutation control: tile 1 v halo row fort-j={npx+1} "
          f"baseline max|v| = {base_row:.6e} (must be nonzero)")
    if base_row == 0.0:
        raise SystemExit("C2 would perturb an all-zero row -- control "
                         "void")
    eps = 1e-6
    v0[:, row_idx] *= (1.0 + eps)
    ref0 = replay_tile(t0)
    mut0 = replay_tile(t0, v_override=v0)
    dmut = np.abs(mut0["ubbtemp"][:, :, 0] - ref0["ubbtemp"][:, :, 0])
    moved = edge_strip_maxima(dmut)
    predicted = eps * base_row
    ok = moved["N"] > 0.01 * predicted and moved["N"] > 0.0
    print(f"C2: ubbtemp North strip moved {moved['N']:.3e} "
          f"(predicted O({predicted:.1e})) -> "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("C2 FAILED: the replay is not sensitive to "
                         "the v halo row the ytp_v edge block reads -- "
                         "the instrument does not exercise the path "
                         "under test.")

    # ---- SELECTOR-AMPLITUDE SWEEP -----------------------------------
    # Discriminates the two mechanisms that both produce a large output
    # difference from a tiny input difference.  They have DIFFERENT
    # FUNCTIONAL FORMS, which is what is measured here, not magnitudes:
    #
    #   branch tie-break  the upwind selector is a numerical zero, so a
    #                     perturbation that crosses it swaps the donor
    #                     cell.  d|Out| SATURATES: once eps exceeds the
    #                     selector magnitude the response stops growing
    #                     with eps (it is the fixed donor gap).
    #   stiff-but-smooth  d|Out| stays PROPORTIONAL to eps over every
    #                     decade, with no plateau.
    #
    # Built-in control: tile 3 has ZERO near-zero selector cells in the
    # census, so its strips MUST stay linear in eps whatever tile 1
    # does.  A sweep where BOTH tiles saturate would indict the probe,
    # not the physics.
    #
    # THE PERTURBATION MUST BE ADDITIVE.  A first version of this sweep
    # used v *= (1 + eps), which is SIGN-PRESERVING: scaling can never
    # move a selector across zero, so it cannot flip an upwind branch
    # and the tie-break hypothesis was untestable by construction (both
    # tiles came back perfectly linear, ratios flat over four decades --
    # a test that could not fail).  An ADDITIVE shift of size
    # eps * max|v| can cross zero once eps * max|v| exceeds the local
    # selector magnitude, which on the flagged strips is ~1e-17..1e-19.
    sweep = []
    print("\nSELECTOR-AMPLITUDE SWEEP (max|d ubbtemp| on the W/E strips; "
          "tile 1 = flagged by the census, tile 3 = unflagged control):")
    print(f"{'eps':>10s} {'t1_flagged':>14s} {'t1/eps':>12s} "
          f"{'t3_control':>14s} {'t3/eps':>12s}")
    base = {}
    for t_s in (0, 2):
        base[t_s] = replay_tile(t_s)
    for eps_s in (1e-20, 1e-18, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8):
        row = {"eps": eps_s}
        for t_s, lbl in ((0, "t1"), (2, "t3")):
            v_s = np.array(stage[t_s]["S02_extuv_v"][:, :, 0], copy=True)
            vmax = float(np.abs(v_s).max())
            if vmax == 0.0:
                raise SystemExit(f"sweep would perturb an all-zero v on "
                                 f"tile {t_s+1} -- control void")
            v_s = v_s + eps_s * vmax   # ADDITIVE: can cross zero
            out_s = replay_tile(t_s, v_override=v_s)
            d_s = np.abs(out_s["ubbtemp"][:, :, 0]
                         - base[t_s]["ubbtemp"][:, :, 0])
            if np.isnan(d_s).any():
                raise SystemExit(f"C3: NaN in swept ubbtemp tile {t_s+1}")
            sm_s = edge_strip_maxima(d_s)
            row[lbl] = max(sm_s["W"], sm_s["E"])
        print(f"{eps_s:10.0e} {row['t1']:14.4e} "
              f"{row['t1'] / eps_s:12.3e} {row['t3']:14.4e} "
              f"{row['t3'] / eps_s:12.3e}")
        sweep.append(row)
    print("  (a FLAT response/eps column = saturation; a CONSTANT "
          "response/eps ratio = linear. No verdict printed here.)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows,
                       "worst_cells": worst_cells,
                       "c2_moved_north": moved["N"],
                       "c2_predicted": predicted,
                       "sweep": sweep}, fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
