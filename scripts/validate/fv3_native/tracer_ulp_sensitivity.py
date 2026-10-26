"""One-ulp sensitivity of the tracer transport on a flat plateau.

The terminator cl2 one-step residual against the Fortran oracle (8e-4 of
qcly, 2026-09-14) lives only on the exactly-flat 2e-6 night-side plateau
next to the front, and the port's eager and compiled arms disagree there
by the same amount while sphum and cl are bit-identical between them.
The reading "a limiter branch is decided by rounding noise on the
plateau" is INFERRED from that.  This measures it: perturb the plateau by
ONE ulp in the IC and step once.  A smooth scheme moves the output by
~1 ulp; a branch flipping on the perturbation moves it by the difference
between its two branches.  Same for cl's zero plateau, nudged to +1 ulp
of qcly, to show whether cl matched the oracle only by virtue of exact
zeros.

Usage::

    python tracer_ulp_sensitivity.py [--n 48] [--km 5] [--n-split 8] [--dt 1920]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


# Fortran run_hydro_1step_term_gfs, tiles 1 and 2 (front-free edges):
# max |Cl + 2 Cl2 - qcly| / qcly within 3 cells of the tile edge.
# Re-measurable with --oracle-ic-run/--oracle-step-run (below).
ORACLE_EDGE_NONCONSTANCY = 1.8177e-4
# The config every pinned number here was measured on.
PINNED_CONFIG = (48, 5, 8, 1920.0)
# --assert-envelope band for the cl2 response: [0.5, 1.2] x pinned. The
# response is ONE limiter tie-break quantum: nudges of one and two ulp gave
# the same 3.297e-09 to four digits, so drift beyond a few percent is a
# scheme change, not noise; 1.2 leaves headroom for that, 0.5 catches a
# halved response (fewer cells flipping) that would otherwise pass as
# "smaller is better". A chosen band, stated as such.
ENVELOPE_BAND = (0.5, 1.2)


def measure_oracle_edge_nonconstancy(ic_run: str, step_run: str,
                                     edge_cells: int = 3) -> float:
    """max |Cl + 2 Cl2 - qcly| / qcly within ``edge_cells`` of a tile edge,
    over the tiles with NO terminator front (cl constant at the IC), from
    the two oracle restart sets. The committed path behind the pinned
    constant."""
    import netCDF4
    from legoesm.core.fv3_native_dcmip16_ic import TERM_QCLY
    worst = 0.0
    for t in range(1, 7):
        with netCDF4.Dataset(f"{ic_run}/RESTART/fv_tracer.res.tile{t}.nc") as z, \
                netCDF4.Dataset(f"{step_run}/RESTART/fv_tracer.res.tile{t}.nc") as o:
            cl0 = np.asarray(z["cl"][0])
            # a FULLY DAYLIT tile: cl above the transition band everywhere.
            # The all-night tile also has no front inside it, but the front
            # lies just across its edges in the neighbouring tiles and its
            # edge cells see it through the halo (1.056e-3 there, measured
            # 2026-09-14) -- that is transport, not the edge non-constancy.
            # The daylit tile's figure equals the port's constant-tracer
            # figure to four digits, which is the evidence it is the
            # right one.
            if not (cl0.min() > 3.99e-6):
                continue
            dev = np.abs(np.asarray(o["cl"][0]) + 2.0 * np.asarray(o["cl2"][0])
                         - TERM_QCLY) / TERM_QCLY
            n = dev.shape[-1]
            edge = np.zeros((n, n), bool)
            edge[:edge_cells, :] = edge[-edge_cells:, :] = True
            edge[:, :edge_cells] = edge[:, -edge_cells:] = True
            worst = max(worst, float(dev[:, edge].max()))
    if worst == 0.0:
        raise SystemExit("no fully daylit tile found in the oracle decks")
    return worst


def plateau_mask(q, value, rtol=1e-12):
    return np.abs(np.asarray(q) - value) <= rtol * value


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--km", type=int, default=5, choices=(5, 10))
    ap.add_argument("--n-split", type=int, default=8)
    ap.add_argument("--dt", type=float, default=1920.0)
    ap.add_argument("--assert-envelope", action="store_true",
                    help="fail unless cl2's one-ulp response lies within "
                         "[0.5, 1.2] x the envelope the parity harness pins "
                         "(CL2_ULP_ENVELOPE_ABS) and cl's stays <= 10 ulp -- "
                         "drift in either direction, or a scheme that "
                         "became smooth, re-opens the ceiling")
    ap.add_argument("--oracle-ic-run", default=None,
                    help="with --oracle-step-run: RE-MEASURE the oracle's "
                         "edge non-constancy from its restarts instead of "
                         "trusting ORACLE_EDGE_NONCONSTANCY (terminator decks)")
    ap.add_argument("--oracle-step-run", default=None)
    args = ap.parse_args(argv)
    if args.assert_envelope and (args.n, args.km, args.n_split, args.dt) != PINNED_CONFIG:
        raise SystemExit(
            f"--assert-envelope is pinned to C{PINNED_CONFIG[0]} km={PINNED_CONFIG[1]} "
            f"n_split={PINNED_CONFIG[2]} dt={PINNED_CONFIG[3]} (the config the "
            f"envelope and the edge figure were measured on); got "
            f"({args.n}, {args.km}, {args.n_split}, {args.dt})")
    oracle_edge = ORACLE_EDGE_NONCONSTANCY
    if args.oracle_ic_run or args.oracle_step_run:
        if not (args.oracle_ic_run and args.oracle_step_run):
            raise SystemExit("--oracle-ic-run and --oracle-step-run go together")
        oracle_edge = measure_oracle_edge_nonconstancy(args.oracle_ic_run,
                                                       args.oracle_step_run)
        print(f"oracle edge non-constancy RE-MEASURED: {oracle_edge:.4e} "
              f"(pinned {ORACLE_EDGE_NONCONSTANCY:.4e})")
        if abs(oracle_edge - ORACLE_EDGE_NONCONSTANCY) > 1e-2 * ORACLE_EDGE_NONCONSTANCY:
            print("  PINNED VALUE IS STALE -- update ORACLE_EDGE_NONCONSTANCY")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
    FV3DuoConfig,
    FV3DuoDynamicsModel,
    ORACLE_DAMPING,
)
    from legoesm.core.fv3_native_dcmip16_ic import TERM_QCLY
    from legoesm.grids.factory import create_fv3_duo_grid

    grid = create_fv3_duo_grid(args.n)
    model = FV3DuoDynamicsModel(grid, FV3DuoConfig(**ORACLE_DAMPING, km=args.km, hydrostatic=True,
                                                   n_split=args.n_split))
    ic = model.dcmip16_initial_state(do_pert=True, terminator=True)
    # q = [sphum, cl, cl2]
    ng, n = grid.ng, grid.n
    cs = slice(ng, ng + n)
    base = model.step(ic, args.dt)
    ulp = np.spacing(TERM_QCLY)             # one ulp at the qcly scale
    results = {}
    print(f"C{args.n} km={args.km} n_split={args.n_split} dt={args.dt}; "
          f"1 ulp at qcly = {ulp:.3e}")
    for iq, name, value in ((2, "cl2", TERM_QCLY / 2), (1, "cl", 0.0)):
        q0 = np.asarray(ic["q"][iq])
        mask = np.zeros_like(q0, dtype=bool)
        mask[:, cs, cs, :] = (plateau_mask(q0[:, cs, cs, :], value)
                              if value else q0[:, cs, cs, :] == 0.0)
        # ONE ulp OF THE PLATEAU VALUE (codex 2026-09-14: spacing(qcly)
        # is two ulp at the 2e-6 plateau). cl's plateau is exactly 0,
        # where an ulp is 5e-324 and meaningless; it is nudged by one ulp
        # of qcly instead, as a "smallest positive value" control.
        nudge = np.spacing(value) if value else ulp
        q1 = q0.copy()
        q1[mask] += nudge
        pert = dict(ic)
        pert["q"] = list(ic["q"])
        pert["q"][iq] = jnp.asarray(q1)
        out = model.step(pert, args.dt)
        a = np.asarray(base["q"][iq])[:, cs, cs, :]
        b = np.asarray(out["q"][iq])[:, cs, cs, :]
        d = np.abs(a - b)
        moved = d > 10 * nudge
        print(f"{name}: plateau cells perturbed {int(mask.sum())} by "
              f"{nudge:.3e}; one-step output max |change| {d.max():.3e} = "
              f"{d.max() / nudge:.3g} nudges; cells moved > 10x: "
              f"{int(moved.sum())}; per-level max "
              f"{[f'{d[..., k].max():.1e}' for k in range(d.shape[-1])]}")
        # the OTHER tracers must not move at all: passengers are independent
        for jq, other in ((0, "sphum"), (1, "cl"), (2, "cl2")):
            if jq == iq:
                continue
            dd = float(np.abs(np.asarray(base["q"][jq]) - np.asarray(out["q"][jq])).max())
            print(f"   {other} changed by {dd:.3e} (must be 0)")
            if dd != 0.0:
                print("PASSENGERS ARE NOT INDEPENDENT -- refusing")
                return 2
        results[name] = float(d.max())
    # CONSTANCY (GLM 2026-09-14): a uniform tracer must come back uniform
    # under any consistent flux-form transport + remap -- an index or
    # deposit bug shows up here with no envelope to hide under. Uses the
    # cl2 slot at the qcly value, so it rides the same code path.
    const = dict(ic)
    const["q"] = list(ic["q"])
    qc = np.zeros_like(np.asarray(ic["q"][2]))
    qc[:, cs, cs, :] = TERM_QCLY
    const["q"][2] = jnp.asarray(qc)
    outc = np.asarray(model.step(const, args.dt)["q"][2])[:, cs, cs, :]
    dev = np.abs(outc - TERM_QCLY) / TERM_QCLY
    # MEASURED 2026-09-14 (job 9769402 + oracle restarts): the deviation
    # lives ONLY within 3 cells of a face edge (5640 of 69120 cells at
    # C48), interior at 6.353e-16 -- and the Fortran oracle's own
    # Cl + 2 Cl2 breaks at the edges of its front-free tiles by the SAME
    # 1.818e-4 with the SAME 6.353e-16 interior. The duo-grid edge
    # treatment does not preserve a constant tracer in either code; the
    # port reproduces the oracle's edge figure to four digits. So the
    # interior is gated at rounding and the edge is pinned to the oracle.
    dev_int = float(dev[:, 3:-3, 3:-3, :].max())
    dev_edge = float(dev.max())
    print(f"constancy: uniform tracer after one step -- interior deviation "
          f"{dev_int:.3e} (rounding ~1e-15); face-edge deviation {dev_edge:.3e} "
          f"(oracle's own edge non-constancy {oracle_edge:.3e})")
    results["constancy_interior"] = dev_int
    results["constancy_edge"] = dev_edge
    if args.assert_envelope:
        import importlib.util
        import os
        spec = importlib.util.spec_from_file_location(
            "full_step_oracle_parity",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "full_step_oracle_parity.py"))
        parity = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parity)
        env = parity.CL2_ULP_ENVELOPE_ABS
        ok_cl2 = ENVELOPE_BAND[0] * env <= results["cl2"] <= ENVELOPE_BAND[1] * env
        ok_cl = results["cl"] <= 10 * ulp        # cl was nudged by ulp(qcly)
        ok_const = (results["constancy_interior"] <= 1e-12
                    and abs(results["constancy_edge"] - oracle_edge)
                    <= 1e-2 * oracle_edge)
        print(f"envelope check: cl2 {results['cl2']:.3e} vs pinned {env:.3e} "
              f"-> {'OK' if ok_cl2 else 'DRIFTED'}; cl {results['cl'] / ulp:.1f} ulp "
              f"-> {'OK' if ok_cl else 'NOT SMOOTH'}; constancy interior "
              f"{results['constancy_interior']:.1e}, edge {results['constancy_edge']:.4e} "
              f"vs oracle {oracle_edge:.4e} -> "
              f"{'OK' if ok_const else 'VIOLATED'}")
        return 0 if (ok_cl2 and ok_cl and ok_const) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
