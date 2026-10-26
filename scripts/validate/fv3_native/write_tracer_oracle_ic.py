"""Write a NONZERO second tracer into a copy of the oracle's zero-step
tracer restarts, in the oracle's own tile layout.

The pinned Fortran oracle cold-starts from its analytic test case, which
sets ``liq_wat = 0`` everywhere, so a two-tracer comparison against it
perturbs a zero and proves nothing.  A warm-started oracle run can carry
any tracer field its restart holds.  This writes ``liq_wat`` as the port's
tracer 1 -- ``sphum * (1 + 0.5 sin(lon))``, the same field
``FV3DuoDynamicsModel.dcmip16_initial_state(n_tracers=2)`` builds -- so
the port and the oracle start from the SAME second tracer by construction.

The face map (which port face is which oracle tile, and under which
dihedral / transpose) is DERIVED here from the initial condition exactly
as ``full_step_oracle_parity.py`` derives it, and the run refuses unless
that derivation sits at the IC floor.  The map is then applied in
reverse: the port's (i, j, k) window -> the oracle's stored (k, j, i).
The parity harness re-derives the map from u/v/pt/delp when it scores the
tracer deck, so a wrong map here shows up there as a liq_wat IC mismatch
rather than passing silently.

Usage::

    python write_tracer_oracle_ic.py --zerostep-run <oracle zero-step dir> \
        --out-dir <dir receiving fv_tracer.res.tile{1..6}.nc>
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "full_step_oracle_parity", os.path.join(_HERE, "full_step_oracle_parity.py"))
parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(parity)

IC_FLOOR = 1e-12       # the harness's own quad-geometry floor is ~1e-14
TRACER_NAME = "liq_wat"


def to_oracle_layout(port_ijk, meta_entry):
    """Port (i, j, k) window -> oracle stored (k, j, i) under the derived
    map.  Exact inverse of ``parity.oracle_ij`` composed with the
    dihedral (every dihedral here is an involution)."""
    transposed, nm, _su, _sv = meta_entry
    a = parity.DIHEDRAL[nm](np.asarray(port_ijk))        # port orientation
    b = a if transposed else a.transpose(1, 0, 2)         # -> (j, i, k)
    return np.ascontiguousarray(np.moveaxis(b, -1, 0))    # -> (k, j, i)


def build_tracer_ic(ctx, ak, bk, iq: int = 1):
    """Port faces of tracer ``iq`` on the compute window, (i, j, k)."""
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        lon_modulated_tracer)
    n, ng = ctx["n"], ctx["ng"]
    cs = slice(ng, ng + n)
    _state, sphum6 = parity.build_port_ic(ctx, ak, bk, nh=False, zvir=0.0)
    return [lon_modulated_tracer(sphum6[t], ctx["gs6"][t]["agrid_lon"],
                                 n, ng, iq)[cs, cs, :] for t in range(6)]


def derive_map_or_refuse(ctx, ak, bk, zerostep_run: str):
    orc = parity.load_oracle(zerostep_run, nh=False)
    state, _sphum6 = parity.build_port_ic(ctx, ak, bk, nh=False, zvir=0.0)
    port = parity.port_window(state, ctx)
    _cost, meta, perm, worst, _pf, _w = parity.derive_face_map(port, orc)
    if not worst < IC_FLOOR:
        raise SystemExit(
            f"REFUSED: derived face map sits at {worst:.3e} > {IC_FLOOR:g} "
            f"on the IC -- the map is not trustworthy, nothing is written")
    return meta, perm, worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zerostep-run", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--iq", type=int, default=1,
                    help="which port tracer index becomes liq_wat (>= 1)")
    args = ap.parse_args(argv)
    if args.iq < 1:
        raise SystemExit("--iq must be >= 1: tracer 0 is sphum itself")
    import netCDF4

    from legoesm.core.fv3_native_duo_stepper import build_six_face_duo_context
    from legoesm.core.fv3_native_eta import set_eta_analytic

    ctx = build_six_face_duo_context(parity.N, parity.NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    ak, bk, _ptop, _ks = set_eta_analytic(parity.KM)
    meta, perm, worst = derive_map_or_refuse(ctx, ak, bk, args.zerostep_run)
    print(f"face map derived from the IC at {worst:.3e}: port face -> tile "
          f"{[(pf, perm[pf] + 1) for pf in range(6)]}")

    tracer6 = build_tracer_ic(ctx, ak, bk, args.iq)
    os.makedirs(args.out_dir, exist_ok=True)
    src_dir = os.path.join(args.zerostep_run, "RESTART")
    written = []
    for pf in range(6):
        tile = perm[pf] + 1
        name = f"fv_tracer.res.tile{tile}.nc"
        dst = os.path.join(args.out_dir, name)
        shutil.copyfile(os.path.join(src_dir, name), dst)
        field = to_oracle_layout(tracer6[pf], meta[pf][perm[pf]])
        with netCDF4.Dataset(dst, "r+") as ds:
            var = ds.variables[TRACER_NAME]
            if var.shape[1:] != field.shape:
                raise SystemExit(
                    f"{name}: {TRACER_NAME} is {var.shape[1:]}, the mapped "
                    f"port field is {field.shape}")
            # sphum is the control on the SCALAR map: the port's sphum
            # mapped the same way must land on the stored sphum
            s_port = to_oracle_layout(
                parity.build_port_ic(ctx, ak, bk, nh=False, zvir=0.0)[1][pf]
                [ctx["ng"]:ctx["ng"] + ctx["n"],
                 ctx["ng"]:ctx["ng"] + ctx["n"], :],
                meta[pf][perm[pf]])
            s_orc = np.asarray(ds.variables["sphum"][0])
            r = parity.rel(s_port, s_orc)
            if not r < IC_FLOOR:
                raise SystemExit(
                    f"{name}: mapped port sphum vs stored sphum {r:.3e} > "
                    f"{IC_FLOOR:g} -- scalar map wrong, nothing written")
            var[0] = field
            if "checksum" in var.ncattrs():
                # FMS reads with checksum_required=.false.; the old value
                # would be a lie about the new bits
                var.delncattr("checksum")
        written.append((name, float(field.min()), float(field.max())))
        print(f"  port face {pf} -> {name}: {TRACER_NAME} "
              f"[{field.min():.4g}, {field.max():.4g}]  sphum control {r:.2e}")
    if len({w[0] for w in written}) != 6:
        raise SystemExit("the face map did not cover six distinct tiles")
    print(f"wrote 6 tracer restarts -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
