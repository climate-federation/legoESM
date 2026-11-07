#!/usr/bin/env python
"""Can a DRY cell's diagonal reach a WET cell's solved tracer?

``nemo_tracer_tridiagonal`` (``implicit_solver.py:503``) ends with
``diagonal = where(wet, diagonal, 1.0)``.  NEMO writes ``trazdf.f90:445``
unconditionally and its dry rows keep their own ``e3t``.  Round 37 registered
that substitution as a DEVIATION rather than a transcription, because
legoESM's layer thickness is exactly ``0.0`` below the seafloor where NEMO's
``e3t_3d`` is the positive reference thickness -- removing it divided by a
zero diagonal and OVERFLOW's kt=2 tracers went non-finite.

The row it leaves behind (3120 dry cells of GYRE's scored box, absmax 299.71)
is DEBT only if a dry diagonal can influence a wet answer.  This gate PLANTS
into the dry diagonals and measures whether it can.

FOUR ARMS, because an independent claim review broke the first version -- a
``x1e3`` plant on a card with no dry cell above a wet one is inert by
TOPOLOGY, not by identity, and proves nothing about a card that has one:

* ``finite``     every dry diagonal replaced by a large random value.
* ``nan``        every dry diagonal replaced by NaN.  NaN is the one value a
                 multiply by exact zero cannot absorb.
* ``synthetic``  the same two plants on a hand-built column that HAS a dry
                 cell directly ABOVE a wet one, which is the topology the
                 real cards may not provide.
* ``negative``   the control that proves the test can fail: the off-diagonal
                 coupling ACROSS a dry interface is set non-zero, and the wet
                 answer must then move.

The census of dry-above-wet cells is reported per card, because it is what
decides whether a card's own result is an identity or an accident of its
bathymetry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from nemo_testcase_l2_gyre_phase3_gate import require  # noqa: E402

CARDS = ("GYRE-zco", "LOCK_EXCHANGE-zco", "OVERFLOW-zps")
SEED = 20260906


def _card_geometry(case: str):
    """The card's own wet mask and after-thickness, from its own recipe."""
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card)
    card = build_nemo_testcase_card(case)
    state = card.recipe.initial_state
    z = card.recipe.z_coord
    wet = (np.asarray(state.land_mask.data) > 0.5)[..., None] & np.asarray(
        z.is_active)
    dz = np.asarray(getattr(z, "h_partial", None)
                    if getattr(z, "h_partial", None) is not None
                    else z.dz_ref)
    if dz.ndim == 1:
        dz = np.broadcast_to(dz, wet.shape)
    return wet, np.ascontiguousarray(dz, dtype=np.float64), float(card.dt_s)


def _solve(K, dz, e3w, dt, wet, rhs, *, diag_plant=None, break_mask=False):
    """legoESM's own assembly and ordered solve, with an optional plant."""
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing import (
        nemo_ordered_tridiagonal_solve, nemo_tracer_tridiagonal)
    lower, diagonal, upper = nemo_tracer_tridiagonal(
        jnp.asarray(K), jnp.asarray(dz), jnp.asarray(e3w), dt,
        jnp.asarray(wet), dtype=jnp.float64)
    lower, diagonal, upper = (np.asarray(lower), np.asarray(diagonal),
                              np.asarray(upper))
    if diag_plant is not None:
        diagonal = np.where(wet, diagonal, diag_plant)
    if break_mask:
        # THE NEGATIVE CONTROL.  The whole identity rests on the off-diagonal
        # coupling across a dry interface being EXACTLY zero; give it a value
        # and the wet answer must move, or this gate cannot fail.
        dry_face = ~(wet[..., 1:] & wet[..., :-1])
        lower = lower.copy()
        upper = upper.copy()
        lower[..., 1:] = np.where(dry_face, -0.25, lower[..., 1:])
        upper[..., :-1] = np.where(dry_face, -0.25, upper[..., :-1])
    out = np.asarray(nemo_ordered_tridiagonal_solve(
        jnp.asarray(lower), jnp.asarray(diagonal), jnp.asarray(upper),
        jnp.asarray(rhs)))
    return out * wet


def _arms(wet, dz, dt, rng) -> dict:
    """Baseline plus every plant, scored on WET cells only."""
    nlev = wet.shape[-1]
    face_wet = wet[..., 1:] & wet[..., :-1]
    K = np.where(face_wet, 1.0e-4 + rng.uniform(0.0, 1.0e-2, face_wet.shape),
                 0.0)
    e3w = np.where(face_wet, 10.0 + rng.uniform(0.0, 200.0, face_wet.shape),
                   1.0)
    dz_safe = np.where(wet, np.maximum(dz, 1.0), 0.0)
    rhs = np.where(wet, rng.uniform(-20.0, 20.0, wet.shape), 0.0) * dz_safe
    base = _solve(K, dz_safe, e3w, dt, wet, rhs)
    require(bool(np.isfinite(base[wet]).all()),
            "the baseline solve is not finite on wet cells")
    require(int(np.count_nonzero(base[wet])) > 0,
            "the baseline solve is identically zero; this plant would be "
            "vacuous")
    rows = {}
    for label, plant, broken in (
            ("finite", rng.uniform(-1.0e6, 1.0e6, wet.shape), False),
            ("nan", np.full(wet.shape, np.nan), False),
            ("negative_control", None, True)):
        got = _solve(K, dz_safe, e3w, dt, wet, rhs,
                     diag_plant=plant, break_mask=broken)
        moved = int(np.count_nonzero(
            (base[wet].view(np.uint64) != got[wet].view(np.uint64))
            | ~np.isfinite(got[wet])))
        rows[label] = {
            "wet_cells": int(wet.sum()),
            "wet_cells_moved_or_nonfinite": moved,
            "wet_nonfinite": int(np.count_nonzero(~np.isfinite(got[wet]))),
        }
    # The topology that decides whether a card's result is an identity or an
    # accident of its bathymetry -- and whether this arm can fail at all on it.
    rows["dry_above_wet_cells"] = int(np.count_nonzero(
        (~wet[..., :-1]) & wet[..., 1:]))
    rows["dry_cells"] = int((~wet).sum())
    # A dry cell with NO wet vertical neighbour has no path to a wet answer
    # even when the coupling is BROKEN, so on such a card a clean plant is
    # vacuous.  This is measured, not assumed: it is exactly what the negative
    # control reports.
    rows["wet_cells_vertically_adjacent_to_dry"] = int(np.count_nonzero(
        (wet[..., :-1] & ~wet[..., 1:]) | (~wet[..., :-1] & wet[..., 1:])))
    rows["arm_informative"] = bool(
        rows["negative_control"]["wet_cells_moved_or_nonfinite"] > 0)
    return rows


def run() -> dict:
    rng = np.random.default_rng(SEED)
    cards = {}
    for case in CARDS:
        wet, dz, dt = _card_geometry(case)
        cards[case] = _arms(wet, dz, dt, rng)

    # THE SYNTHETIC COLUMN the real cards may not provide: a dry cell directly
    # ABOVE a wet one, which is the only topology through which a dry row's
    # value could enter the forward recurrence of a wet one.
    wet = np.ones((1, 1, 6), dtype=bool)
    wet[0, 0, 2] = False
    dz = np.full(wet.shape, 50.0)
    synthetic = _arms(wet, dz, 14400.0, rng)
    require(synthetic["dry_above_wet_cells"] == 1,
            "the synthetic column does not carry the topology it exists for")

    every = list(cards.items()) + [("synthetic", synthetic)]
    finite_clean = all(v["finite"]["wet_cells_moved_or_nonfinite"] == 0
                       for _, v in every)
    informative = [name for name, v in every if v["arm_informative"]]
    vacuous = [name for name, v in every if not v["arm_informative"]]
    # The synthetic column exists precisely so that at least one arm carries
    # the dry-above-wet topology; if IT is vacuous the gate proves nothing.
    control_fires = ("synthetic" in informative) and bool(informative)
    nan_leaks = {name: v["nan"]["wet_nonfinite"] for name, v in every}
    return {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round38-dry-slot-plant-v1",
        "cards": cards,
        "synthetic_dry_above_wet_column": synthetic,
        "finite_plant_never_reaches_a_wet_cell": finite_clean,
        "arms_where_the_control_fires": informative,
        "arms_where_this_gate_is_VACUOUS": vacuous,
        "nan_leaks_into_wet_cells": nan_leaks,
        "verdict": (
            "A dry diagonal's FINITE value is an UNCONSUMED SLOT: the "
            "off-diagonal coupling across a dry interface is exactly 0.0 "
            "(nemo_tracer_tridiagonal multiplies each face coefficient by "
            "interface_wet), and 0.0 * finite is 0.0, so no dry row can enter "
            "a wet row's recurrence.  It is NOT NaN-safe, because 0.0 * NaN "
            "is NaN; the synthetic column measures exactly that, and any card "
            "whose dry_above_wet_cells is 0 could not have shown it.  On a "
            "zco card every dry cell is a whole LAND COLUMN with no wet "
            "vertical neighbour, so GYRE's and LOCK_EXCHANGE's clean plants "
            "are VACUOUS -- their own negative control moves nothing either, "
            "and this gate says so rather than counting them as evidence."),
        "status": ("AT-BAR" if finite_clean and control_fires else "DEBT"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    report = run()
    text = json.dumps(report, indent=1, sort_keys=True, default=str)
    if args.json:
        args.json.write_text(text + "\n")
    print(text)
    for name, row in list(report["cards"].items()) + [
            ("synthetic", report["synthetic_dry_above_wet_column"])]:
        print(f"{name:<20} dry {row['dry_cells']:<6} dry-above-wet "
              f"{row['dry_above_wet_cells']:<5} "
              f"finite-plant moved {row['finite']['wet_cells_moved_or_nonfinite']} "
              f"nan-plant nonfinite {row['nan']['wet_nonfinite']} "
              f"control moved {row['negative_control']['wet_cells_moved_or_nonfinite']}"
              + ("" if row["arm_informative"] else "   [VACUOUS on this card]"))
    print(f"VACUOUS-ARMS {report['arms_where_this_gate_is_VACUOUS'] or 'none'}")
    print(f"STATUS {report['status']}")
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    sys.exit(main())
