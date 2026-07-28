#!/usr/bin/env python
"""Sharp-state corner-wedge capture-compare (codex deficiency-r1 endgame).

Feeds REAL C36 vertex-run lattice states (run_duo_stepper_modon
--dump-lattice-days output) through both wedge implementations:

  ours   : `_CornerLagrange.fill` exactly as the live exchanges call it
  oracle : the verbatim Fortran `compute_lagrange_coeff` +
           `fill_corner_region_2d` (fv3_cornerlag_extract.F90 driver)

on the SAME pre-wedge arrays (strips + k2e rings re-run, wedge slots
left stale — the live fill overwrites wedges wholesale and reads only
strips/interior, so the pre-wedge content is exactly what the live op
sees).  Families: A scalars (delp, pt), D-vector components (u, v).

If sharp-state wedge diffs stay at the smooth-cert scale (~5e-11 rel),
the wedge arithmetic is fully exonerated (deficiency-r1 P0 dead) and
the injector must be a dynamic-interaction effect; a large or
structured diff localizes the injecting family directly.

Usage:
  wedge_sharp_compare.py --dump vdump_ctl_lat_day3.npz --build-dir DIR
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def _driver(build_dir: Path) -> Path:
    exe = build_dir / "cornerlag_driver"
    if not exe.exists():
        src = REPO / "scripts/validate/fv3_native"
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["gfortran", "-O0", "-g", "-fdefault-real-8",
             "-fdefault-double-8", "-o", str(exe),
             str(src / "fv3_cornerlag_shim.F90"),
             str(src / "fv3_cornerlag_extract.F90"),
             str(src / "fv3_cornerlag_driver.F90")],
            check=True, cwd=build_dir)
    return exe


def _run_oracle(exe: Path, n: int, ng: int, a_lon_w, a_lat_w,
                fields: dict) -> dict:
    """fields: {(istag,jstag): pre-wedge array} -> {(istag,jstag):
    {(fort_i, fort_j): oracle value}} for the wedge slots."""
    tagf = {(0, 0): "F00", (1, 1): "F11", (0, 1): "F01", (1, 0): "F10"}
    lines = [f"{n} {ng}"]
    lo_w = 1 - (ng + 1)
    m = a_lon_w.shape[0]
    for i in range(m):
        for j in range(m):
            lines.append(f"APT {i + lo_w} {j + lo_w} "
                         f"{a_lon_w[i, j]:.17e} {a_lat_w[i, j]:.17e}")
    for (istag, jstag), f in fields.items():
        ni, nj = f.shape
        for i in range(ni):
            for j in range(nj):
                lines.append(f"{tagf[(istag, jstag)]} {i + 1 - ng} "
                             f"{j + 1 - ng} {f[i, j]:.17e} 0.0")
    lines.append("COEF 0 0 0 0")
    for (istag, jstag) in fields:
        lines.append(f"RUN {istag} {jstag} 0 0")
    run = subprocess.run([str(exe)], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, check=True)
    out: dict = {k: {} for k in fields}
    for line in run.stdout.splitlines():
        p = line.split()
        if p and p[0] == "OUT":
            out[(int(p[1]), int(p[2]))][(int(p[3]), int(p[4]))] = \
                float(p[5])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default=None,
                    help="lattice-dump npz (run_duo_stepper_modon "
                         "--dump-lattice-days); omit with --ic")
    ap.add_argument("--ic", default=None, metavar="LON,LAT",
                    help="skip the dump: build the single-burst IC at "
                         "LON,LAT in-process (the day-0 state is the "
                         "SHARPEST field of the run — max|u| 47.9 vs "
                         "day-2's 20.6) and compare on that")
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--ng", type=int, default=3)
    ap.add_argument("--build-dir", required=True)
    args = ap.parse_args()
    if not (args.dump or args.ic):
        ap.error("need --dump or --ic")

    sys.path.insert(0, str(REPO / "packages/core"))
    from legoesm.grids.fv3_native_ext_vector import (
        ext_parity_lonlat_ref,
        c2l_ord2_face,
        _pack_p1,
        _geo_lattice_exchange,
        _a2d_project,
        _write_d_strips,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        exchange_agrid_scalar_halos,
        exchange_dgrid_vector_halos,
        k2e_remap_halo_rings,
    )
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )

    n, ng = args.n, args.ng
    ctx = build_six_face_duo_context(n, ng, use_ext_bundle=True,
                                     oracle_conventions=True, omega=0.0)
    if args.ic:
        from legoesm.grids.cubed_sphere import great_circle_distance
        from legoesm.grids.fv3_native_gridstruct import (
            FV3_RADIUS_M,
            analytic_swcore_state,
        )
        sys.path.insert(0, str(REPO))
        from tests.test_cases.colliding_modons import (
            _MODON_H0,
            _MODON_SIZE,
            _MODON_UMAX,
        )

        lon0, lat0 = (np.deg2rad(float(x)) for x in args.ic.split(","))

        def wind_fn(ll):
            r = great_circle_distance(ll[..., 0], ll[..., 1], lon0,
                                      lat0, FV3_RADIUS_M)
            u_e = _MODON_UMAX * np.exp(-(np.asarray(r)
                                         / _MODON_SIZE) ** 2)
            return u_e, np.zeros_like(u_e)

        def scalars_fn(ll):
            delp = np.full(ll.shape[:-1], 9.80665 * _MODON_H0)
            return delp, np.ones_like(delp)

        d = {}
        for t, gs in enumerate(ctx["gs6"], start=1):
            st = analytic_swcore_state(gs, wind_fn=wind_fn,
                                       scalars_fn=scalars_fn)
            for k in ("delp", "pt", "u", "v"):
                d[f"{k}_t{t}"] = np.asarray(st[k])
        src_desc = f"IC single burst at {args.ic}"
    else:
        d = np.load(args.dump)
        src_desc = args.dump
    ectx = ctx["ectx"]
    exe = _driver(Path(args.build_dir))
    a_lon_w, a_lat_w = (x[0] for x in
                        ext_parity_lonlat_ref(n, ng + 1, "A"))

    # --- family A: strips + rings (NO fill) on the dumped scalars ----
    def pre_wedge_a(name):
        f6 = [np.array(d[f"{name}_t{t}"], copy=True)
              for t in range(1, 7)]
        for t in range(1, 7):
            exchange_agrid_scalar_halos(f6, t, n, ng)
        # thread the run's actual order (codex r5 P2: a bare call
        # silently selects the historical nord=4)
        k2e_remap_halo_rings(f6, "A", n, ng,
                             k2e_nord=int(d.get("k2e_nord", 4)))
        return f6

    # --- family D: full vector strip path (NO fill) ------------------
    def pre_wedge_d():
        u6 = [np.array(d[f"u_t{t}"], copy=True) for t in range(1, 7)]
        v6 = [np.array(d[f"v_t{t}"], copy=True) for t in range(1, 7)]
        for t in range(1, 7):
            exchange_dgrid_vector_halos(u6, v6, t, n, ng)
        ug6, vg6 = [], []
        for t in range(6):
            ua, va = c2l_ord2_face(u6[t], v6[t], ectx["dx6"][t],
                                   ectx["dy6"][t], ectx["amat6"][t],
                                   n, ng)
            ug6.append(_pack_p1(ua, n, ng))
            vg6.append(_pack_p1(va, n, ng))
        _geo_lattice_exchange(ug6, ectx)
        _geo_lattice_exchange(vg6, ectx)
        for t in range(6):
            ud4, vd4 = _a2d_project(ug6[t], vg6[t], t, ectx)
            _write_d_strips(u6[t], v6[t], ud4, vd4, n, ng)
        return u6, v6

    fams = []
    for name in ("delp", "pt"):
        f6 = pre_wedge_a(name)
        fams.append((name, (0, 0), f6, "corner_a3"))
    u6, v6 = pre_wedge_d()
    fams.append(("u", (0, 1), u6, "corner_du3"))
    fams.append(("v", (1, 0), v6, "corner_dv3"))

    print(f"state={src_desc}")
    worst_overall = 0.0
    for name, (istag, jstag), f6, opkey in fams:
        worst = 0.0
        scale = max(float(np.max(np.abs(f6[t]))) for t in range(6))
        for t in range(6):
            pre = np.array(f6[t], copy=True)
            ours = np.array(pre, copy=True)
            ectx[opkey][t].fill(ours)
            oracle = _run_oracle(exe, n, ng, a_lon_w, a_lat_w,
                                 {(istag, jstag): pre})
            for (fi, fj), val in oracle[(istag, jstag)].items():
                got = ours[fi - 1 + ng, fj - 1 + ng]
                worst = max(worst, abs(got - val))
        rel = worst / scale if scale else 0.0
        print(f"family {name:5s} stag({istag},{jstag}): "
              f"max|ours-oracle| {worst:.3e}  scale {scale:.3e}  "
              f"rel {rel:.3e}")
        worst_overall = max(worst_overall, rel)
    verdict = ("WEDGE-EXONERATED (smooth-cert scale)"
               if worst_overall < 1e-8 else "WEDGE-DISCREPANT")
    print(f"VERDICT: {verdict}  worst-rel {worst_overall:.3e}")


if __name__ == "__main__":
    main()
