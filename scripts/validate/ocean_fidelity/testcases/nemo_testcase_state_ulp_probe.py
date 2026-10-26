#!/usr/bin/env python3
"""Per-cell ulp distance between two revisions' PROGNOSTIC STATE.

The phase-3 gates score REDUCTIONS -- one L-infinity number per field per step.
That is the right instrument for "did we move away from NEMO", and the wrong
one for "did the field itself move": a reduction is unchanged whenever the
cells that moved are not the cell attaining the maximum.  This probe closes
that blind spot by comparing the arrays.

It exists because the reduction and the field disagreed.  Over ten OVERFLOW-zps
steps the gate's T rows were bit-identical while the T FIELD departed from
kt=7 onward at five cells, and a receipt had already been written saying "T is
bit-identical" on the strength of the rows alone.

Usage (run once per revision, then compare):

    nemo_testcase_state_ulp_probe.py dump  <case> <n_steps> <out.npz>
    nemo_testcase_state_ulp_probe.py compare <case> <before.npz> <after.npz>

``compare`` reports, per field and per step, on the SAME wet mask and the same
staggering slice the trajectory gate uses: how many cells differ, the largest
absolute difference, and the largest ulp distance among cells whose magnitude
is at least ``REL_FLOOR`` of the field maximum.

Read the ulp column with its floor in mind.  A perturbation entering through a
column-uniform barotropic shift is ABSOLUTE, so it lands on a cell of
magnitude 1e-4 of the field maximum as 1e4 times as many ulps as it does on
the largest cell.  Cells below the floor are excluded outright because an ulp
distance measured against a value near zero (or across zero) is a number in
the 1e18 range that means nothing at all -- the first version of this probe
reported exactly that and it had to be thrown away.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

#: Cells below this fraction of the field maximum are excluded from the ulp
#: column (their ulp distance is dominated by how close they sit to zero).
REL_FLOOR = 1.0e-10

FIELDS = ("T", "S", "u", "v", "eta")


def _ordered(values: np.ndarray) -> np.ndarray:
    """IEEE-754 sign-magnitude -> monotone two's complement, so that the
    absolute difference of two of these IS the ulp distance."""
    bits = np.ascontiguousarray(values, dtype=np.float64).view(np.int64)
    return np.where(bits < 0, np.int64(-0x8000000000000000) - bits, bits)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(_ordered(a) - _ordered(b))


def gate_slice(field: str, array: np.ndarray) -> np.ndarray:
    """The staggering slice the trajectory gate scores (``lego_fields``)."""
    if field == "u":
        return array[:, 1:, :]
    if field == "v":
        return array[1:, :, :]
    return array


def wet_masks(card) -> dict[str, np.ndarray]:
    """The trajectory gate's own ``expected_masks``, re-derived from the card."""
    wet = np.asarray(card.recipe.initial_state.land_mask.data) > 0.5
    active = np.asarray(card.recipe.z_coord.is_active) & wet[..., None]
    u = active & np.roll(active, -1, axis=1)
    u[:, -1] = False
    v = active & np.roll(active, -1, axis=0)
    v[-1] = False
    return {"T": active, "S": active, "u": u, "v": v, "eta": wet}


def _card(case: str):
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    if get_policy() != PrecisionPolicy.fp64(transcendentals="libm"):
        raise SystemExit("precision policy is not fp64")
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    return build_nemo_testcase_card(case)


def dump(case: str, n_steps: int, out: str) -> int:
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )

    card = _card(case)
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    state = card.recipe.initial_state
    snapshots = {}
    for step in range(1, n_steps + 1):
        state = model.step(state, dt=card.dt_s)
        for field, array in (
            ("T", state.T.data), ("S", state.S.data), ("u", state.u.data),
            ("v", state.v.data), ("eta", state.eta.data),
        ):
            values = np.asarray(array)
            if values.dtype != np.float64:
                raise SystemExit(f"{field} is {values.dtype}, not float64")
            # kt indexing matches the gate: the entry state of step kt is the
            # state after kt-1 completed steps, so one step lands on kt=2.
            snapshots[f"kt{step + 1}_{field}"] = values
    np.savez(out, **snapshots)
    print(f"wrote {out} ({len(snapshots)} arrays, {n_steps} steps, fp64)")
    return 0


def compare(case: str, before_path: str, after_path: str) -> int:
    card = _card(case)
    masks = wet_masks(card)
    before, after = np.load(before_path), np.load(after_path)
    if set(before.files) != set(after.files):
        raise SystemExit("the two dumps do not hold the same arrays")
    steps = sorted({int(k.split("_")[0][2:]) for k in before.files})
    print(f"{case}: per-cell state difference on the wet mask, "
          f"ulp column floored at {REL_FLOOR:g} of the field maximum")
    worst_any = False
    for field in FIELDS:
        mask = masks[field]
        rows = []
        for kt in steps:
            key = f"kt{kt}_{field}"
            a = gate_slice(field, after[key])[mask]
            b = gate_slice(field, before[key])[mask]
            delta = np.abs(a - b)
            if not delta.any():
                rows.append((kt, 0, 0.0, 0))
                continue
            scale = max(float(np.max(np.abs(b))), np.finfo(np.float64).tiny)
            # BOTH sides must clear the floor.  A cell that is 4.8e-28 on one
            # side and exactly 0.0 on the other has an ulp distance of 4.2e18,
            # which is a true statement about the representation and a useless
            # one about the physics; the count and max|delta| columns still
            # report that cell.  LOCK's ssh at kt=2 is exactly this case.
            keep = (np.abs(b) > REL_FLOOR * scale) & (np.abs(a) > REL_FLOOR * scale)
            ulps = (int(ulp_distance(a[keep], b[keep]).max())
                    if keep.any() else 0)
            rows.append((kt, int((delta > 0).sum()), float(delta.max()), ulps))
        differing = sum(row[1] for row in rows)
        worst_any = worst_any or bool(differing)
        print(f"  {field:4s} bit-identical: {'YES' if not differing else 'NO'}"
              f"   cells ever differing {differing:7d}"
              f"   max|delta| {max(r[2] for r in rows):.4e}"
              f"   max ulp {max(r[3] for r in rows)}")
        if differing:
            detail = "  ".join(
                f"kt{r[0]}:{r[1]}/{r[2]:.1e}/{r[3]}u" for r in rows if r[1])
            print(f"       {detail}")
    # Exit 0 whether or not the states differ: this probe MEASURES, it does not
    # adjudicate.  The bar lives in ulp_move_gate, and what counts as an
    # admissible field difference is a decision for the receipt, not for this.
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dump")
    d.add_argument("case")
    d.add_argument("n_steps", type=int)
    d.add_argument("out")
    c = sub.add_parser("compare")
    c.add_argument("case")
    c.add_argument("before")
    c.add_argument("after")
    args = parser.parse_args(argv)
    if args.mode == "dump":
        return dump(args.case, args.n_steps, args.out)
    return compare(args.case, args.before, args.after)


if __name__ == "__main__":
    sys.exit(main())
