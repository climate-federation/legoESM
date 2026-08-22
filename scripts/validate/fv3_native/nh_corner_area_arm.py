#!/usr/bin/env python
"""Opt-in arm: substitute the REFERENCE's corner-diagonal halo areas
for the port's own Lagrange fill, at exactly the cells the port fills,
and re-run the existing height probe against the measured baseline.

WHY THIS ARM. Job 9466634 placed the seat inside ``update_dz_d``: the
height enters at the parity floor everywhere and leaves the COMPUTE
WINDOW wrong at corner-adjacent cells (3.67e-05 on faces 1/2/4/5,
1.90e-05 on 3/6, zh interface 1). The only input of that kernel with a
horizontal stencil whose halo the port constructs itself is the cached
``nh_area6``/``nh_rarea6`` built by ``nh_exchanged_area6``: it ext_scalar
exchanges the port's area (whose corner-diagonal halo cells hold the
BIG_NUMBER sentinel and are therefore NOT supplied by that exchange)
and then runs its own Lagrange corner-region fill there. The gridstruct
docstring predicted, before any of this was measured, that this
substitution is where corner-adjacent disagreement would localise. The
earlier metric-transplant arm cannot have tested it: it wrote
``gs6[t]["area"]`` (sentinels preserved) and the consumed array is
derived from that and corner-filled by the port's own routine.

WHAT THIS DOES. Pre-seeds ``ctx["nh_area6"]``/``ctx["nh_rarea6"]`` with
the exchanged array, then OVERWRITES exactly the sentinel-marked cells
of the port area with the reference's mirror-symmetric values (tag
``M_AREA``, ``M_RAREA`` in the retro_gsmetrics dumps). Because
``nh_exchanged_area6`` returns the cache when the key is present, the
patched arrays are the ones ``update_dz_d`` receives -- no other code
path is touched. Then the height probe (the instrument that produced
the 9466634 baseline) runs unmodified via its own ``main``.

PRIVACY OF DEFAULT BEHAVIOUR. This whole script is the opt-in: nothing
in it is imported by the deck. Without ``--on`` (and this file is never
run by anything else) the port's behaviour is byte-identical.

PRE-REGISTERED READING (declared before the run, against the 9466634
baseline, zh1 post-update compute-interior per face):
  CONFIRMS the substitution is the seat:
    faces 1/2/4/5 fall from ~3.67e-05 and faces 3/6 from ~1.90e-05 to
    <= 1.0e-09 on BOTH interfaces (the parity floor the pre-state
    already shows), AND the pre-state floor and all instrument
    controls still pass.
  REFUTES it:
    the post-update interior on every face remains within a factor of
    2 of the baseline (>= 1.8e-05 on faces 1/2/4/5), i.e. the
    reference areas moved the corner cells by a measured nonzero amount
    (printed below before the run) and the answer did not move.
  NEITHER:
    a partial drop -- any face falls by more than 10x yet stays above
    1.0e-09 -- or a drop on one face class only. That would say the
    substitution contributes to the seat but is not the whole of it,
    and the two face classes (3.67e-05 vs 1.90e-05) point at a second
    corner mechanism whose signature this arm does not isolate.

WHAT THIS ARM DOES NOT COVER, stated:
  * Only ``area``/``rarea`` corner-diagonal cells. The side-strip halo
    areas come from the ext_scalar exchange and were measured at the
    parity floor on entry (9466634); they are not re-tested here.
  * No other metric family (dx/dy/.xx/.yy etc.) and nothing inside
    ``Riem_Solver3``, which is column-local and therefore only
    guilty through its inputs.
  * The oracle's own corner-area CONSTRUCTION is not derived or
    checked for symmetry here; it is transplanted as a black box.
  * Acoustic sub-step 1 only, inherited from the probe: from sub-step
    2 on, a bad interior has already become a bad halo and the seat is
    no longer separable.
  * Growth of the error over the sub-step loop.

NO VERDICT IS PRINTED. The pre-registered criteria above are the
reading; the run outputs are the probe's own table plus the per-face
disagreement lines printed before anything runs.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import full_step_oracle_parity as fsp      # noqa: E402
from full_step_oracle_parity import (      # noqa: E402
    DIHEDRAL, IC_CONTROL_MAX_REL, N, NG, ORACLE_ROOT, build_port_ic,
    derive_face_map, load_oracle, oracle_ij, port_window,
)

# The instrument that produced the measured baseline, re-run verbatim.
PROBE_FILE = os.path.join(_HERE, "nh_halo_zh_parity.py")

# The dumped reference metric trees live here, one per tile.
METRICS_ROOT = ("/burg-archive/glab/users/pg2328/fv3_duo_gaps/"
                "retro_gsmetrics/run_c48")

# update_dz_d's horizontal stencil over the area never exceeds one ring
# beyond the compute window per pass (the nord_v diffusion ring is
# applied inside that band, nh_utils.F90:194-311); a sentinel cell
# entirely outside this band is read by nothing in this kernel.
READ_WINDOW = (slice(NG - 1, NG + N + 1), slice(NG - 1, NG + N + 1))

# Pre-registered thresholds; single source of truth for the reading.
CONFIRM_MAX = 1.0e-09
REFUTE_MIN = 1.8e-05
NEITHER_DROP = 1.0e-06


def load_ref_plane(run_dir: str, tile: int, tag: str, m_a: int) -> np.ndarray:
    """One ``M_*`` plane from ``extchain_t<tile>.dat`` via its manifest.

    THE FORMAT IS READ, NOT ASSUMED. ``extchain_dump2`` writes
    ``<TAG> <ni> <nj> 1 <byte_offset>`` (fv3_extchain_extract.F90:73-80)
    and the plane is Fortran-ordered. A record that does not match that
    shape, a missing tag, or a plane that is not the padded square is
    FATAL -- guessing an offset here would transplant something this arm
    never verified.
    """
    mf_path = os.path.join(run_dir, f"extchain_t{tile}.mf")
    dat_path = os.path.join(run_dir, f"extchain_t{tile}.dat")
    for p in (mf_path, dat_path):
        if not os.path.exists(p):
            raise SystemExit(f"arm: missing {p}: the reference metric "
                             f"dump this arm transplants is absent")
    off = cnt = None
    with open(mf_path) as fh:
        for line in fh:
            parts = line.split()
            if parts and parts[0] == tag:
                if len(parts) != 5:
                    raise SystemExit(
                        f"{mf_path}: {tag} record {parts!r} is not "
                        f"<TAG> <ni> <nj> 1 <offset>; refusing to guess")
                ni, nj, off = int(parts[1]), int(parts[2]), int(parts[4])
                cnt = ni * nj
                break
    if off is None:
        raise SystemExit(f"{mf_path}: no {tag} record")
    if cnt != m_a * m_a:
        raise SystemExit(
            f"{mf_path}: {tag} count {cnt} != {m_a}x{m_a}; the dumped "
            f"plane is not the padded square this arm maps onto")
    with open(dat_path, "rb") as fh:
        fh.seek(off)
        plane = np.fromfile(fh, dtype="<f8", count=cnt)
    if plane.size != cnt:
        raise SystemExit(f"{dat_path}: short read at {tag}")
    # Fortran order: the writer streams the array as declared.
    return plane.reshape((ni, nj), order="F")


def build_patched_context(metrics_run: str, ic_run: str):
    """Wrap the deck's context builder so the cached area arrays carry
    the reference's corner values BEFORE any consumer can snapshot
    them. ``nh_exchanged_area6`` returns the cache when the key exists,
    so the write is the only difference and no other code path sees a
    change. Returns (builder, disagreement_rows).
    """
    import legoesm.core.fv3_native_duo_stepper as duo
    from legoesm.core.fv3_native_nh_core import (  # noqa: F401  presence check
        update_dz_d,
    )
    # The port's own corner fill is what this arm replaces; import it
    # lazily inside the builder below, not here, so nothing runs twice.
    orig_ctx = duo.build_six_face_duo_context
    m_a = N + 2 * NG

    ref_area, ref_rarea = {}, {}
    for tile in range(6):
        ref_area[tile] = load_ref_plane(metrics_run, tile + 1,
                                         "M_AREA", m_a)
        ref_rarea[tile] = load_ref_plane(metrics_run, tile + 1,
                                         "M_RAREA", m_a)
        # The reciprocal dump must be the reciprocal of the area dump;
        # if it is not, the two transplants are not one consistent
        # metric and writing both would be writing two metrics.
        bad = float(np.abs(1.0 / ref_area[tile] - ref_rarea[tile]).max())
        if bad > 1.0e-12 * float(np.abs(ref_rarea[tile]).max()):
            raise SystemExit(
                f"tile {tile + 1}: |1/M_AREA - M_RAREA| = {bad:.3e}; "
                f"the dumped reciprocal is not derived from the dumped "
                f"area; refusing to transplant an inconsistent pair")

    rows: list = []

    def builder(n, ng, **kw):
        ctx = orig_ctx(n, ng, **kw)
        if (n, ng) != (N, NG):
            raise SystemExit(
                f"arm: ctx built at n={n} ng={ng}, this arm's mapping "
                f"and baselines are for n={N} ng={NG}")
        # Face map: same derivation and same control floor the height
        # probe enforces, so the transplant lands on the same tile
        # pairing the baseline was measured with.
        from legoesm.core.fv3_native_eta import set_eta_analytic
        ak, bk, ptop, _ks = set_eta_analytic(fsp.KM)
        orc_ic = load_oracle(ic_run, nh=True)
        state, _s = build_port_ic(ctx, ak, bk, nh=True)
        p_ic = port_window(state, ctx)
        _c, meta, perm, worst, _pf, _wo = derive_face_map(p_ic, orc_ic)
        if worst > IC_CONTROL_MAX_REL:
            raise SystemExit(
                "arm: INSTRUMENT CONTROL FAILED: face-map worst rel "
                f"{worst:.3e} > floor {IC_CONTROL_MAX_REL:.0e}; the "
                "transplant would aim at the wrong cells")

        # Trigger the port's own exchange+fill ONCE, then overwrite.
        # Import here, not at module top, so merely reading this file
        # cannot half-build a cache.
        from legoesm.core.fv3_native_dynamics import (  # noqa: F401
            fv_dynamics_step,
        )
        from legoesm.core.fv3_native_dsw_tail_3d import (
            nh_exchanged_area6,
        )
        area6 = nh_exchanged_area6(ctx)
        rarea6 = ctx["nh_rarea6"]

        for pf in range(6):
            unex = np.asarray(ctx["gs6"][pf]["area"], dtype=np.float64)
            # The port gridstruct marks its unfilled corner-diagonal
            # cells with a single BIG_NUMBER sentinel. Take the value
            # at [0, 0] -- guaranteed sentinel by construction of the
            # single-tile gridstruct -- and REFUSE if it does not look
            # like a sentinel, because then the mask below would be a
            # plausible-looking lie.
            s = float(unex[0, 0])
            interior_med = float(np.median(
                unex[NG:NG + N, NG:NG + N]))
            if not (s > 1.0e3 * interior_med):
                raise SystemExit(
                    f"arm: face {pf + 1}: unexchanged area[0,0] = {s:.6e} "
                    f"is not a sentinel against interior median "
                    f"{interior_med:.6e}; the sentinel contract this "
                    f"arm's mask depends on does not hold")
            mask = unex == s
            cnt = int(mask.sum())
            if cnt == 0:
                raise SystemExit(
                    f"arm: face {pf + 1}: ZERO sentinel cells -- the "
                    f"port's own corner fill supplied nothing here, so "
                    f"this arm would write nothing and would return "
                    f"'unchanged' by construction")
            if cnt >= N * N:
                raise SystemExit(
                    f"arm: face {pf + 1}: sentinel mask covers {cnt} "
                    f"cells (>= the {N}x{N} compute window); the mask "
                    f"has matched real values, refusing to overwrite "
                    f"the interior")
            in_win = int(mask[READ_WINDOW].sum())
            if in_win == 0:
                raise SystemExit(
                    f"arm: face {pf + 1}: none of the {cnt} sentinel "
                    f"cells lies inside the window update_dz_d reads "
                    f"(compute window grown by one ring); a write "
                    f"nothing reads cannot move the answer")

            # Map the reference tile plane into this port face's index
            # order through the SAME machinery the baseline probe
            # scored with: oracle_ij for storage order, DIHEDRAL[nm]
            # for orientation. A dropped stage here was caught once
            # before in this campaign by exactly this reuse.
            ot = perm[pf]
            transposed, nm, _su, _sv = meta[pf][ot]
            ra = DIHEDRAL[nm](oracle_ij(ref_area[ot], transposed))
            rr = DIHEDRAL[nm](oracle_ij(ref_rarea[ot], transposed))
            if ra.shape != unex.shape:
                raise SystemExit(
                    f"arm: face {pf + 1}: mapped reference {ra.shape} "
                    f"vs port {unex.shape}")
            if not np.all(np.isfinite(ra[mask])) or \
                    np.any(ra[mask] <= 0.0):
                raise SystemExit(
                    f"arm: face {pf + 1}: reference area at the "
                    f"sentinel cells is not finite-positive; this is "
                    f"not a metric, refusing")
            # Report the pre-write disagreement BEFORE anything runs.
            dis = float(np.abs(area6[pf][mask] - ra[mask]).max())
            rel = dis / float(np.abs(ra[mask]).max())
            rows.append((pf + 1, ot + 1, cnt, in_win, dis, rel))

            # WRITE, in place: these arrays are the objects gs_nh and
            # update_dz_d receive. Recompute the reciprocal from the
            # patched area so the pair stays one metric, and verify
            # against the dumped M_RAREA rather than trusting either.
            area6[pf][mask] = ra[mask]
            rarea6[pf][:] = 1.0 / area6[pf]
            rbad = float(np.abs(rarea6[pf][mask] - rr[mask]).max())
            if rbad > 1.0e-10 * float(np.abs(rr[mask]).max()):
                raise SystemExit(
                    f"arm: face {pf + 1}: patched 1/area vs M_RAREA "
                    f"max {rbad:.3e} at the sentinel cells; the "
                    f"reciprocal was not written consistently")
            # Read-back control: the write must still be in the cache
            # the consumer will fetch, not a copy that died here.
            if not np.array_equal(ctx["nh_area6"][pf][mask], ra[mask]):
                raise SystemExit(
                    f"arm: face {pf + 1}: read-back of the patched "
                    f"cache disagrees; the write did not land")
        return ctx

    return builder, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--on", action="store_true",
                    help="the opt-in; without it this script exits 0 "
                         "having done nothing at all")
    ap.add_argument("--metrics-run", default=METRICS_ROOT)
    ap.add_argument("--ic-run", default=f"{ORACLE_ROOT}/run_nh_zerostep_gfs")
    ap.add_argument("--hdump",
                    default="/burg-archive/glab/users/pg2328/fv3_wsubstep/"
                            "run_nh_1step")
    ap.add_argument("--dt", type=float, default=1920.0)
    ap.add_argument("--n-split", type=int, default=8)
    args, probe_rest = ap.parse_known_args(argv)
    if not args.on:
        print("arm: not enabled (--on absent); no run performed")
        return 0

    if not os.path.exists(PROBE_FILE):
        raise SystemExit(f"arm: height probe not found at {PROBE_FILE}; "
                         f"this arm re-runs THAT instrument, it does "
                         f"not carry a second copy of it")
    spec = importlib.util.spec_from_file_location("zh_probe_arm_target",
                                                  PROBE_FILE)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    builder, rows = build_patched_context(args.metrics_run, args.ic_run)

    # Report the pre-write disagreement before anything runs (req. 4).
    print("pre-write disagreement at the port-filled corner cells "
          "(|port fill - reference|, per face):")
    for pf, ot, cnt, in_win, dis, rel in rows:
        print(f"  face {pf} -> tile {ot}: cells {cnt} (in read window "
              f"{in_win})  max|d| {dis:.6e}  rel {rel:.3e}")

    # Install the wrapper at the SOURCE module: the probe imports the
    # builder inside its own main at call time, so the package
    # attribute is the single interception point and no other import
    # of the package is perturbed.
    import legoesm.core.fv3_native_duo_stepper as duo
    duo.build_six_face_duo_context = builder
    probe.main(["--hdump", args.hdump, "--ic-run", args.ic_run,
                "--dt", str(args.dt),
                "--n-split", str(args.n_split)] + probe_rest)

    print("\npre-registered reading (declared in this file's header, "
          "before the run): compare the POST zh1 interior column above "
          "against baseline 3.67e-05 (faces 1/2/4/5) and 1.90e-05 "
          "(faces 3/6):")
    print(f"  CONFIRMS  : all faces <= {CONFIRM_MAX:.1e} on both "
          f"interfaces")
    print(f"  REFUTES   : all faces >= {REFUTE_MIN:.1e} (baseline "
          f"within a factor of 2) despite the nonzero pre-write "
          f"disagreement printed above")
    print(f"  NEITHER   : any face between {CONFIRM_MAX:.1e} and "
          f"{NEITHER_DROP:.1e}, or a drop on one face class only")
    print("NO VERDICT IS PRINTED BY THIS ARM.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
