#!/usr/bin/env python3
"""Round-104 record-consistency probe for NEMO's shear-production routine.

This checks the RECORDS, not legoESM.  It rebuilds NEMO's own shear
production ``p_sh2`` from NEMO's own recorded operands, in the compiled
source association, and compares it against NEMO's own recorded ``sh2``.  If
that replay is not bit-exact then the round's production rows cannot be
interpreted at all, because the reference side of the comparison would
already be wrong.  Round 103 named ``p_sh2`` as the sole magnitude owner of
the TKE right-hand-side miss and stopped there; this is the reference side of
the walk that follows it into its own routine.

Compiled source, Round-59 build
``GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90``.  On this card
``cpl_sdrftx .AND. ln_stshear`` is false, so the executed branch is the ELSE
at ``:97-109`` and the routine's three output-bearing assignments inside
``DO jk = 2, jpkm1`` (``:83``) are

  S1  ``zsh2u``  ``zdfsh2.f90:99-103``
  S2  ``zsh2v``  ``zdfsh2.f90:104-108``
  S3  ``p_sh2``  ``zdfsh2.f90:112-113``

with the surface/bottom zeroing at ``:116-119``.  The consumed viscosity is
``avm_k`` as passed at ``zdfphy.f90:319``, i.e. the Round-59 record's
``avm_entry``, NOT the post-``zdf_phy`` ``tke_avm_k`` of the round-46 stage
record (those two differ in 6,294 of 20,416 owned cells, which is why the
distinction is made explicitly rather than assumed).  ``wumask``/``wvmask``
are rebuilt from ``umask``/``vmask`` exactly as ``dommsk.f90:237-242``
defines them.  ``e3w_1d(jk)`` is read from the recorded ``e3w_0``, whose
horizontal uniformity this module asserts rather than assumes (zco).

LAYOUT.  Everything here is in legoESM's ``(lat, lon, level)`` orientation on
two windows chosen so that each array has EXACTLY the shape the model's own
shear helper produces, which is what lets a model-side array be substituted
for a recorded one without a second transposition:

  u window   NEMO ji = 1..33, jj = 2..23   -> (22, 33, .)
  v window   NEMO ji = 2..33, jj = 1..23   -> (23, 32, .)
  output     NEMO ji = 2..33, jj = 2..23   -> (22, 32, 29) over jk = 2..30

Both windows reach one cell outside the owned domain.  Every face there is
LAND on this card (``umask`` is identically zero at NEMO ji = 1, 2, 33, 34
and ``vmask`` at jj = 1, 2, 23, 24; asserted below), so the viscosity halo
the u/v face sums would need is annihilated by ``wumask``/``wvmask`` and a
zero pad is exact rather than merely adequate.

The plant arm advances one bit of a single operand and must make the replay
fail, so a green run is not green by construction.  A plant run never prints
a success word.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from legoesm.ocean.fidelity.provenance import worktree_stamp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nemo_testcase_l2_gyre_round54_tke_operands import (  # noqa: E402
    read_record as read_operands,
)

R59 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round59/"
           "oracle_tke_operands/oracle_tke_operands_kt00000002.bin")
R46 = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round46/"
           "oracle_kt2_stage")

# NEMO index windows, 0-based, inclusive-exclusive.
U_I, U_J = slice(1, 34), slice(2, 24)
V_I, V_J = slice(2, 34), slice(1, 24)
O_I, O_J = slice(2, 34), slice(2, 24)
K = slice(1, 30)                      # NEMO jk = 2..jpkm1


class ReplayError(RuntimeError):
    pass


def _refuse(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _unequal(a: np.ndarray, b: np.ndarray) -> tuple[int, float]:
    ne = (np.ascontiguousarray(a).view(np.uint64)
          != np.ascontiguousarray(b).view(np.uint64))
    return int(np.count_nonzero(ne)), float(np.max(np.abs(a - b)))


def _lat_lon(v: np.ndarray) -> np.ndarray:
    """NEMO ``(i, j[, k])`` -> legoESM ``(lat, lon[, level])``."""
    v = np.asarray(v, dtype=np.float64)
    return v.transpose(1, 0, 2) if v.ndim == 3 else v.T


def recorded_operands(
    stage_arrays: dict, avm_owned: np.ndarray, *, return_face_metrics=False,
):
    """Assemble NEMO's own ``zdf_sh2`` operands on the two windows.

    ``stage_arrays`` is the round-46 kt=2 stage-1 record (full halo, reader
    orientation ``(lat, lon, level)``); ``avm_owned`` is the Round-59
    ``avm_entry`` on the owned 32 x 22 x 31 domain.
    """
    def nemo(name):                       # reader gives (lat, lon, .)
        v = np.asarray(stage_arrays[name], dtype=np.float64)
        return v.transpose(1, 0, 2) if v.ndim == 3 else v.T

    uu, vv = nemo("u_Kmm"), nemo("v_Kmm")
    uu_b, vv_b = nemo("u_Kbb"), nemo("v_Kbb")
    r3u, r3v = nemo("r3u_Kmm"), nemo("r3v_Kmm")
    r3u_b, r3v_b = nemo("r3u_Kbb"), nemo("r3v_Kbb")
    umask, vmask = nemo("umask"), nemo("vmask")
    e3w0 = nemo("e3w_0")

    # zco: e3w_0(ji,jj,jk) is e3w_1d(jk).  Asserted, not assumed.
    _refuse(bool(np.all(e3w0 == e3w0[2, 2, :][None, None, :])),
            "REFUSE: recorded e3w_0 is not horizontally uniform, so it is "
            "not NEMO's e3w_1d and zdfsh2.f90:102 cannot be replayed from it")
    e3w1d = e3w0[2, 2, :]

    # Every window face is land on this card, so the viscosity halo is
    # annihilated.  Refuse rather than silently zero-pad a live face.
    _refuse(all(int(np.count_nonzero(umask[i, U_J, :])) == 0
                for i in (1, 2, 33, 34))
            and all(int(np.count_nonzero(vmask[V_I, j, :])) == 0
                    for j in (1, 2, 23, 24)),
            "REFUSE: a window-boundary face is wet, so the zero viscosity "
            "pad is not exact and a halo acquisition is required")

    avm = np.zeros(umask.shape, dtype=np.float64)
    avm[2:-2, 2:-2, :] = np.asarray(avm_owned, dtype=np.float64)

    # dommsk.f90:237-242
    wumask = np.empty_like(umask)
    wvmask = np.empty_like(vmask)
    wumask[:, :, 0], wvmask[:, :, 0] = umask[:, :, 0], vmask[:, :, 0]
    wumask[:, :, 1:] = umask[:, :, 1:] * umask[:, :, :-1]
    wvmask[:, :, 1:] = vmask[:, :, 1:] * vmask[:, :, :-1]

    def w(v, i, j, k=None):
        out = v[i, j] if k is None else v[i, j, k]
        return _lat_lon(out) if out.ndim == 3 else out.T

    e3u = e3w1d[K][None, None, :] * (1.0 + w(r3u, U_I, U_J)[..., None])
    e3u_b = e3w1d[K][None, None, :] * (1.0 + w(r3u_b, U_I, U_J)[..., None])
    e3v = e3w1d[K][None, None, :] * (1.0 + w(r3v, V_I, V_J)[..., None])
    e3v_b = e3w1d[K][None, None, :] * (1.0 + w(r3v_b, V_I, V_J)[..., None])
    operands = {
        # S1/S2 velocity operands, full column so the k-difference is taken
        # in the replay exactly as the compiled statement takes it.
        "u_now": w(uu, U_I, U_J, slice(0, 30)),
        "u_before": w(uu_b, U_I, U_J, slice(0, 30)),
        "v_now": w(vv, V_I, V_J, slice(0, 30)),
        "v_before": w(vv_b, V_I, V_J, slice(0, 30)),
        # group (d): the avm face SUM, zdfsh2.f90:99 / :104
        "avm_face_u": (w(avm, slice(2, 35), U_J, K)
                       + w(avm, U_I, U_J, K)),
        "avm_face_v": (w(avm, V_I, slice(2, 25), K)
                       + w(avm, V_I, V_J, K)),
        # group (a): the live face-metric divisor, zdfsh2.f90:102 / :107
        "divisor_u": e3u * e3u_b,
        "divisor_v": e3v * e3v_b,
        # group (b): zdfsh2.f90:103 / :108
        "wumask": w(wumask, U_I, U_J, K),
        "wvmask": w(wvmask, V_I, V_J, K),
        # group (c): the coast factors, zdfsh2.f90:112-113
        "coast_u": 2.0 - (w(umask, slice(1, 33), O_J, K)
                          * w(umask, slice(2, 34), O_J, K)),
        "coast_v": 2.0 - (w(vmask, O_I, slice(1, 23), K)
                          * w(vmask, O_I, slice(2, 24), K)),
    }
    if return_face_metrics:
        return operands, (e3u, e3u_b, e3v, e3v_b)
    return operands


def rebuild_sh2(op: dict, **overrides) -> np.ndarray:
    """Replay ``zdfsh2.f90:99-113`` in the compiled association.

    Fortran and Python both evaluate a same-precedence ``*``/``/`` chain left
    to right, so ``a * b * c / d * m`` below is the compiled order of
    ``:99-103`` statement for statement; no parenthesis is added that the
    source does not have, and the divisor keeps the source's own bracketing
    ``( e3w*(1+r3u(Kmm)) ) * ( e3w*(1+r3u(Kbb)) )``.
    """
    o = dict(op)
    for key, value in overrides.items():
        if value is None:
            continue
        _refuse(key in o, f"REFUSE: unknown zdf_sh2 operand {key!r}")
        _refuse(np.shape(value) == np.shape(o[key]),
                f"REFUSE: {key} override has shape {np.shape(value)}, "
                f"expected {np.shape(o[key])}")
        o[key] = np.asarray(value, dtype=np.float64)

    du_n = o["u_now"][..., :-1] - o["u_now"][..., 1:]
    du_b = o["u_before"][..., :-1] - o["u_before"][..., 1:]
    dv_n = o["v_now"][..., :-1] - o["v_now"][..., 1:]
    dv_b = o["v_before"][..., :-1] - o["v_before"][..., 1:]

    zsh2u = o["avm_face_u"] * du_n * du_b / o["divisor_u"] * o["wumask"]
    zsh2v = o["avm_face_v"] * dv_n * dv_b / o["divisor_v"] * o["wvmask"]
    return 0.25 * ((zsh2u[:, :-1, :] + zsh2u[:, 1:, :]) * o["coast_u"]
                   + (zsh2v[:-1, :, :] + zsh2v[1:, :, :]) * o["coast_v"])


def replay(r59: Path, r46: Path, plant: str | None) -> dict:
    from nemo_testcase_l2_gyre_round46_kt2_stage_gate import read_stage

    a59 = read_operands(r59)["arrays"]
    stage = read_stage(r46 / "oracle_momstage_kt00000002_s1.bin",
                       expected_kt=2, expected_stage=1)
    _refuse((stage["header"]["Kbb"], stage["header"]["Kmm"]) == (3, 3),
            "REFUSE: the kt=2 stage-1 record is not the Kbb=Kmm=3 slot "
            "zdf_phy passes to zdf_sh2")
    op = recorded_operands(stage["arrays"], a59["avm_entry"])

    planted_at = None
    planted_baseline = None
    planted_term = None
    if plant == "operand-ulp":
        # A control that perturbs a zero is not a control, and neither is one
        # whose perturbation is absorbed by the 0.25 sum at :112-113.  Plant
        # on the u-face carrying the LARGEST |zsh2u|, where the u term is the
        # one that sets the output's last bit, and report the baseline the
        # perturbation multiplies.
        du_n = op["u_now"][..., :-1] - op["u_now"][..., 1:]
        du_b = op["u_before"][..., :-1] - op["u_before"][..., 1:]
        zsh2u = (op["avm_face_u"] * du_n * du_b / op["divisor_u"]
                 * op["wumask"])
        if not np.any(zsh2u != 0.0):
            raise ReplayError(
                "REFUSE: no live u-face to plant; the record is all land")
        # One ULP of a single face is only HALF an ULP of the two-face sum at
        # :112, so round-to-nearest can absorb it.  Walk the live faces in
        # descending |zsh2u| and take the first whose one-bit corruption
        # actually reaches the output; refuse loudly if none does, because
        # that would mean the replay is insensitive to its own operand.
        clean = rebuild_sh2(op)
        order = np.argsort(np.abs(zsh2u), axis=None)[::-1]
        for flat in order:
            idx = np.unravel_index(int(flat), zsh2u.shape)
            if zsh2u[idx] == 0.0:
                raise ReplayError(
                    "REFUSE: no live u-face perturbation reaches p_sh2; the "
                    "replay is insensitive to its own viscosity operand")
            faces = op["avm_face_u"].copy()
            faces[idx] = np.nextafter(faces[idx], np.float64(np.inf))
            trial = dict(op, avm_face_u=faces)
            if _unequal(rebuild_sh2(trial), clean)[0]:
                planted_at = tuple(int(v) for v in idx)
                planted_baseline = float(op["avm_face_u"][idx])
                planted_term = float(zsh2u[idx])
                op = trial
                break

    rebuilt = rebuild_sh2(op)
    recorded = _lat_lon(np.asarray(a59["sh2"]))[..., K]
    _refuse(rebuilt.shape == recorded.shape,
            f"REFUSE: replay shape {rebuilt.shape} != recorded "
            f"{recorded.shape}")
    n, mx = _unequal(rebuilt, recorded)
    rows = [{
        "field": "p_sh2",
        "nemo_statement": "R59TKE zdfsh2.f90:99-113",
        "cells": int(recorded.size),
        "n_unequal": n,
        "absolute_max": mx,
        "classification": "BIT" if n == 0 else "NOT-BIT",
    }]
    ok = n == 0
    return {
        "format": "nemo-testcase-l2-gyre-round104-shear-record-replay-v1",
        "worktree": worktree_stamp(),
        "record_round59": str(r59),
        "record_round46": str(r46),
        "viscosity_operand": (
            "avm_k as passed at zdfphy.f90:319 = Round-59 avm_entry; the "
            "round-46 post-zdf_phy tke_avm_k is a DIFFERENT array"),
        "domain": "NEMO levels 2:jpkm1, every owned cell, no mask exception",
        "plant": plant,
        "plant_index": planted_at,
        "plant_baseline_operand": planted_baseline,
        "plant_baseline_term": planted_term,
        "rows": rows,
        "status": "PASS" if ok else "FAIL",
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--round59", type=Path, default=R59)
    p.add_argument("--round46", type=Path, default=R46)
    p.add_argument("--plant", choices=("operand-ulp",))
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    try:
        report = replay(args.round59, args.round46, args.plant)
    except ReplayError as exc:
        print(str(exc))
        return 3
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    if args.plant:
        if report["status"] == "PASS":
            print("REFUSE: the operand plant left the replay green")
            return 2
        print("STATUS PLANT-FIRED")
        return 1
    print("STATUS", report["status"])
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())


def model_operands(
    z_coord, u_face, v_face, eta, avm, u_mask, v_mask, *,
    return_intermediates=False,
):
    """MIRROR of legoESM's own ``zdf_sh2`` operand construction.

    This is deliberately a mirror and not a second implementation to be
    trusted on its own: it exists so that ONE operand at a time can be
    substituted into :func:`rebuild_sh2`, and the gate PROVES it is the
    model's by scoring ``rebuild_sh2(**model_operands(...))`` against the
    production step itself (the ``model_operand_replay`` row).  If that row
    is not bit-identical the mirror is wrong, or the production step is
    reading an operand the mirror does not know about, and the gate says so
    rather than reporting an attribution.

    It follows ``ocean_model_latlon_cgrid.py``'s ``nemo_qco_live_face``
    branch and ``_shared.avm_weighted_shear_production``: the free-surface
    ratio of ``domqco.f90:214-217``, the reference ladder from the raw mesh
    ``nemo_e3w_0``, the viscosity face SUM of ``zdfsh2.f90:99``, the wet-face
    masks of ``dommsk.f90:237-242`` and the coast factors of
    ``zdfsh2.f90:112-113``.  ``eta`` is supplied by the caller precisely
    because WHICH free-surface field reaches this construction is the
    question the round is asking.
    """
    f64 = np.float64
    eta = np.asarray(eta, dtype=f64)
    hu0 = np.asarray(z_coord.nemo_hu_0, dtype=f64)
    hv0 = np.asarray(z_coord.nemo_hv_0, dtype=f64)
    a_t = np.asarray(z_coord.nemo_e1e2t, dtype=f64)
    a_u = np.asarray(z_coord.nemo_e1e2u, dtype=f64)
    a_v = np.asarray(z_coord.nemo_e1e2v, dtype=f64)
    u_face = np.asarray(u_face, dtype=f64)
    v_face = np.asarray(v_face, dtype=f64)
    avm = np.asarray(avm, dtype=f64)
    u_mask = np.asarray(u_mask, dtype=f64)
    v_mask = np.asarray(v_mask, dtype=f64)

    wet_u = (hu0 > 0.0).astype(f64)
    wet_v = (hv0 > 0.0).astype(f64)
    num_u = 0.5 * (a_t * eta + np.roll(a_t * eta, -1, axis=1))
    num_v = 0.5 * (a_t * eta + np.roll(a_t * eta, -1, axis=0))
    r3u = num_u * (wet_u / (hu0 + 1.0 - wet_u)) / a_u
    r3v = num_v * (wet_v / (hv0 + 1.0 - wet_v)) / a_v
    r3u = np.concatenate([r3u[:, -1:], r3u], axis=1)
    r3v = np.concatenate([r3v[:1, :], r3v], axis=0)
    ref = np.asarray(z_coord.nemo_e3w_0, dtype=f64)[..., 1:]
    ref_u = np.concatenate([ref[:, -1:, :], ref], axis=1)
    ref_v = np.concatenate([ref[:1, :, :], ref], axis=0)
    eps = np.float64(1e-30)               # coeff-ok: the helper's own floor
    e3u = np.maximum(ref_u * (1.0 + r3u[..., None]), eps)
    e3v = np.maximum(ref_v * (1.0 + r3v[..., None]), eps)
    operands = {
        # nemo_face_native_now2: Kbb and Kmm are the SAME slot, so the two
        # factors of zdfsh2.f90:100-102 are the same array.
        "u_now": u_face, "u_before": u_face,
        "v_now": v_face, "v_before": v_face,
        "avm_face_u": (np.concatenate([avm[:, -1:, :], avm], axis=1)
                       + np.concatenate([avm, avm[:, :1, :]], axis=1)),
        "avm_face_v": (np.concatenate([avm[:1, :, :], avm], axis=0)
                       + np.concatenate([avm, avm[-1:, :, :]], axis=0)),
        "divisor_u": e3u * e3u,
        "divisor_v": e3v * e3v,
        "wumask": u_mask[..., :-1] * u_mask[..., 1:],
        "wvmask": v_mask[..., :-1] * v_mask[..., 1:],
        "coast_u": 2.0 - u_mask[:, :-1, 1:] * u_mask[:, 1:, 1:],
        "coast_v": 2.0 - v_mask[:-1, :, 1:] * v_mask[1:, :, 1:],
    }
    if not return_intermediates:
        return operands
    du = u_face[..., :-1] - u_face[..., 1:]
    dv = v_face[..., :-1] - v_face[..., 1:]
    zsh2u = (operands["avm_face_u"] * du * du
             / operands["divisor_u"] * operands["wumask"])
    zsh2v = (operands["avm_face_v"] * dv * dv
             / operands["divisor_v"] * operands["wvmask"])
    return operands, {
        "r3u": r3u,
        "r3v": r3v,
        "e3u": e3u,
        "e3v": e3v,
        "du": du,
        "dv": dv,
        "zsh2u": zsh2u,
        "zsh2v": zsh2v,
    }


OPERAND_GROUPS = {
    "velocity": ("u_now", "u_before", "v_now", "v_before"),
    "viscosity_face_sum": ("avm_face_u", "avm_face_v"),
    "live_face_divisor": ("divisor_u", "divisor_v"),
    "wet_face_mask": ("wumask", "wvmask"),
    "coast_factor": ("coast_u", "coast_v"),
}
