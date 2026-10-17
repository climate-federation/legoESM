#!/usr/bin/env python3
"""Round 303: the hooked OMT-4 run with the fold-band stage pair, per arm.

Base run = round 302's ``unit`` arm (round 249's private atomic fold unit +
round 245's barotropic surface-mask arm).  Arms add, one at a time and
together, two NEMO statements on top of it:

* ``topv``: the card's existing ``rk3_fold_exchange_top_v_row`` selector
  (the end-of-stage sign-minus V exchange, ``stprk3_stg.f90:826`` R299W7 /
  ``:795`` ORCA2_OMIP_L4, SUBROUTINE ``stp_RK3_stg``).
* ``tmask``: round 258's ``_resolved_stage_vmask_arm``.  It replaces the
  ``v_mask_3d`` argument of EVERY call of the stage-transport function, at
  all three stages, with the D114 resolved V mask; inside that function the
  mask feeds the qco V face thickness, the corrected stage V, the 3-D V mass
  flux and the continuity W -- i.e. everything the stage transport builds,
  not only the ``zFv`` line (``stprk3_stg.f90:291`` / ``:283``).

Unlike round 302's probe, a refusal keeps the history.  ``--mirror`` instead
checks, on NEMO's own frames, that the top V row is the fold image of the row
below (value equality on every cell; bit differences listed separately).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in (REPO_ROOT, REPO_ROOT / "packages/core",
                REPO_ROOT / "packages/ocean"):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from legoesm.ocean.fidelity.provenance import worktree_stamp  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402, I001
    nemo_testcase_l4_orca2_round258_v_transport_pivot_gate as r258,
    nemo_testcase_l4_orca2_round302_region_history as r302,
)

ARMS = {"hooked": (False, False), "topv": (True, False),
        "tmask": (False, True), "pair": (True, True)}
require = r302.require
GateError = r302.GateError


def fold_mirror(v_frame: np.ndarray, perm: np.ndarray, sign: float) -> dict:
    """Compare the top V row with ``sign * row[-2][perm]`` (the fold image)."""
    top, image = v_frame[-1], sign * v_frame[-2][perm]
    value_unequal = int(np.count_nonzero(top != image))
    bits = top.view(np.uint64) != image.view(np.uint64)
    signed_zero_only = bool(np.all((top[bits] == 0.0) & (image[bits] == 0.0)))
    return {"value_unequal": value_unequal, "bit_unequal": int(bits.sum()),
            "bit_diffs_are_signed_zeros": signed_zero_only}


def _resolved_vmask(card):
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks_3d
    from legoesm.ocean.vertical import nemo_qco_resolved_mesh_operands

    zc, grid = card.recipe.z_coord, card.recipe.grid
    cu, cv = compute_face_masks_3d(zc.is_active, grid)
    dt = jnp.asarray(card.recipe.initial_state.eta.data).dtype
    res = nemo_qco_resolved_mesh_operands(
        zc, grid, jnp.asarray(cu, dt), jnp.asarray(cv, dt), dt,
        zc.is_active.shape[-1])
    return jnp.concatenate([jnp.zeros_like(res.vmask3[:1]), res.vmask3], axis=0)


def run(card, record_root: Path, arm: str, kts: int) -> dict[str, object]:
    import jax

    require(arm in ARMS, f"unknown arm {arm}")
    topv, tmask = ARMS[arm]
    if topv:
        cfg = card.recipe.model_config
        require(not cfg.rk3_fold_exchange_top_v_row, "card already selects top-V")
        card = card._replace(recipe=card.recipe._replace(
            model_config=cfg._replace(rk3_fold_exchange_top_v_row=True)))
    fields = r302.r300.omt4.rung0.candidate_fields
    frame = r302.r300.omt4.rung0.assemble_frame
    masks = r302.nemo_masks(record_root)
    forcing = r302.r300.omt0.rung0_ladder._zero_forcing((148, 180))
    models = r302._models(card, "unit")
    state, history, refusal = card.recipe.initial_state, [], None
    with contextlib.ExitStack() as stack:
        stack.enter_context(r302.r300.r245._barotropic_surface_mask_arm(
            card, True, v_only=True, t_pivot_north_neighbor=True))
        if tmask:
            stack.enter_context(r258._resolved_stage_vmask_arm(_resolved_vmask(card)))
        for kt in range(1, kts + 1):
            history.append({"kt": kt, "stage": 0, "fields": r302.score_stage(
                fields(state), frame(record_root, kt, 0), masks)})
            try:
                outs = [jax.device_get(m.step(state, card.dt_s, freshwater=forcing[0],
                                              surface_forcing=forcing[1]))
                        for m in models]
            except Exception as error:  # noqa: BLE001 - the refusal is the result
                refusal = {"kt": kt, "error": str(error)[:4000]}
                break
            for stage, out in enumerate(outs, start=1):
                history.append({"kt": kt, "stage": stage, "fields": r302.score_stage(
                    fields(out), frame(record_root, kt, stage), masks)})
            print(f"PROGRESS {arm} kt={kt}", file=sys.stderr, flush=True)
            state = outs[-1]
    return {"arm": arm, "refusal": refusal, "history": history,
            "first_over_bar": r302.first_over_bar(history)}


def mirror(card, record_root: Path, kts: int) -> dict[str, object]:
    fold = card.recipe.grid.fold
    perm, sign = np.asarray(fold.perm_v), float(fold.vector_sign_v)
    frame = r302.r300.omt4.rung0.assemble_frame
    rows = [{"kt": kt, "stage": s, **fold_mirror(
        np.asarray(frame(record_root, kt, s)["v"], np.float64), perm, sign)}
        for kt in range(1, kts + 1) for s in range(4)]
    return {"frames": len(rows), "rows": rows,
            "value_unequal": sum(r["value_unequal"] for r in rows),
            "bit_unequal": sum(r["bit_unequal"] for r in rows),
            "bit_diffs_are_signed_zeros": all(r["bit_diffs_are_signed_zeros"] for r in rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck-root", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), default="hooked")
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--kts", type=int, default=10)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import jax
        from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

        stamp = worktree_stamp()
        require(stamp["clean"] and stamp["commit"].lower() == args.expect_commit.lower(),
                "measurement requires its clean committed tree")
        policy = PrecisionPolicy.fp64(transcendentals="libm")
        set_policy(policy)
        require(get_policy() == policy and bool(jax.config.jax_enable_x64)
                and jax.default_backend() == "cpu", "fp64/libm CPU not active")
        card = r302.r300.omt4.build_omt4_card(args.deck_root)
        r302.r300.omt4.validate_omt4_card(args.deck_root, card)
        report = (mirror(card, args.deck_root, args.kts) if args.mirror
                  else run(card, args.deck_root, args.arm, args.kts))
    except (GateError, OSError, ValueError) as error:
        print(f"STATUS REFUSE: {error}")
        return r302.EXIT_REFUSE
    report["worktree"] = stamp
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    if args.mirror:
        print("MIRROR", json.dumps({k: report[k] for k in (
            "frames", "value_unequal", "bit_unequal", "bit_diffs_are_signed_zeros")}))
    else:
        print("REFUSAL", json.dumps(report["refusal"]))
        print("FIRST_OVER_BAR", json.dumps(report["first_over_bar"], sort_keys=True))
    print("STATUS MEASURED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
