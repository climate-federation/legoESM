#!/usr/bin/env python3
"""R8-P1: split ``sshwzv.f90:297-298`` into its two operands on the seamount.

Round 218 named the first non-bit tracer statement as the RK3 continuity
solve

    pww(ji,jj,jk) = pww(ji,jj,jk+1)
       - (  ze3div(ji,jj,jk)
          + r1_Dt * e3t_3d(ji,jj,jk) * ( r3t(ji,jj,Kaa) - r3t(ji,jj,Kbb) ) )
         * tmask(ji,jj,jk)

at ``1.99e-13`` (vector) / ``3.37e-13`` (flux) relative, and left it
UNATTRIBUTED between the horizontal divergence ``ze3div`` and the
thickness-tendency term.  legoESM forms the same statement in
``nemo_qco_wzv_recurrence`` (``ocean_pe_latlon_cgrid.py``), whose second
operand is built from exactly three numbers: the reference thickness
``e3t_0``, the step clock, and ``r3_after - r3_before``.

This probe decides the split WITHOUT a model seam, because every input of
the second operand is recorded:

1. the round-218 tracer record carries NEMO's own ``r3t`` at ``Kbb``,
   ``Kmm`` and ``Kaa`` for each stage;
2. NEMO builds those as ``r3t = ssh * r1_ht_0`` (``domqco.F90``), and the
   step-entry / bt-frame records carry the same ``ssh`` arrays legoESM is
   handed;
3. so ``ssh / r3t`` recovers NEMO's own ``ht_0`` cell by cell, and legoESM's
   column depth is the card's ``sum_k e3t_0 * tmask`` accumulated by the same
   helper's own loop.

If NEMO's implied ``ht_0`` and the card's column depth agree to the bit,
BOTH ``r3_before`` and ``r3_after`` are bitwise NEMO's, the thickness
tendency operand is bitwise NEMO's, and the whole residual belongs to the
horizontal divergence ``ze3div``.  If they do not, the thickness operand is
a candidate and the probe says by how much.

Bit equality is the standard; the probe prints what it measured and never
its own verdict.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "nemo_testcase_l1_vortex"))

from nemo_testcase_l1_vortex_kt2_walk import (  # noqa: E402
    _strip2, _strip3, read_bt_frame,
)
from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    GateError, read_entry, require,
)
from nemo_testcase_l1_vortex_smt_round218_tracer_walk import (  # noqa: E402
    CARDS, DEFAULT_ROOTS, read_tracer_terms,
)


def run(root: Path, card_key: str, *, allow_dirty: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(card_key in CARDS, f"unknown card key {card_key!r}")

    case = CARDS[card_key]
    card = build_nemo_testcase_card(case)
    zc = card.recipe.z_coord
    nlev = int(zc.n_levels)
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    entry1 = read_entry(root / "oracle_step_entry_kt00000001.bin", case,
                        expect_interior=interior)
    entry2 = read_entry(root / "oracle_step_entry_kt00000002.bin", case,
                        expect_interior=interior)

    # The card's own column depth, by the SAME accumulation the production
    # helper runs (nemo_qco_wzv_operands: h0 += e3t_0[...,jk]*tmask[...,jk]),
    # on the SAME operand set that helper reads -- not a rebuilt ladder.
    from legoesm.ocean.vertical import nemo_qco_resolved_mesh_operands
    state0 = card.recipe.initial_state
    u_mask_3d = jnp.asarray(state0.u_mask.data)[..., None] * jnp.ones(
        (1, 1, nlev), dtype=jnp.float64)
    v_mask_3d = jnp.asarray(state0.v_mask.data)[..., None] * jnp.ones(
        (1, 1, nlev), dtype=jnp.float64)
    ops = nemo_qco_resolved_mesh_operands(
        zc, card.recipe.grid, u_mask_3d, v_mask_3d, jnp.float64, nlev)
    e3t0 = np.asarray(ops.e3t_0, dtype=np.float64)
    tmask = np.asarray(zc.is_active, dtype=np.float64)
    require(e3t0.shape[-1] == nlev,
            f"the card's e3t_0 is {e3t0.shape[-1]} deep, the card has {nlev}")
    h0 = np.zeros(e3t0.shape[:2], dtype=np.float64)
    for jk in range(nlev):
        h0 = h0 + e3t0[..., jk] * tmask[..., jk]

    rows = []
    for stage in (1, 2, 3):
        groups = read_tracer_terms(root, stage)
        for slot, ssh, label in (
                ("r3t_kbb", entry1["ssh"],
                 "r3_before, the Kbb surface ratio (domqco.F90 r3t=ssh*r1_ht_0)"),
                ("r3t_kaa", entry2["ssh"],
                 "r3_after, the Kaa surface ratio the stage wzv reads")):
            r3 = np.asarray(groups[slot], dtype=np.float64)
            ssh = np.asarray(ssh, dtype=np.float64)
            # NEMO's own ht_0, implied by its own two arrays.  Only where the
            # ratio is defined: a zero ssh (land, or an unstarted column)
            # carries no information about ht_0 and is excluded by its own
            # test, not by a mask guessed here.
            live = (r3 != 0.0) & (ssh != 0.0)
            require(int(live.sum()) > 0,
                    f"{case} stage {stage} {slot}: no column has a nonzero "
                    "ssh AND a nonzero r3t; the implied depth is untestable")
            implied = np.full_like(h0, np.nan)
            implied[live] = ssh[live] / r3[live]
            delta = np.abs(implied[live] - h0[live])
            rel = delta / np.maximum(np.abs(h0[live]), 1.0e-300)
            # The bit test that actually decides the split: does NEMO's own
            # product rebuild its own r3t from the card's depth?
            rebuilt = ssh[live] * (1.0 / h0[live])
            rows.append({
                "case": case, "stage": stage, "operand": slot,
                "describes": label,
                "columns_tested": int(live.sum()),
                "implied_ht0_max_abs_diff": float(delta.max()),
                "implied_ht0_max_rel_diff": float(rel.max()),
                "r3_cells_unequal": int(np.count_nonzero(rebuilt != r3[live])),
                "r3_max_abs_diff": float(np.max(np.abs(rebuilt - r3[live]))),
                "r3_peak": float(np.max(np.abs(r3[live]))),
            })

    bt = read_bt_frame(root / "oracle_bt_frames_kt00000001.bin", expect_step=1)
    return {
        "case": case, "oracle_root": str(root), "legoesm_git_sha": sha,
        "nlev": nlev, "bt_frame_fields": sorted(bt),
        "column_depth_min": float(h0[h0 > 0].min()),
        "column_depth_max": float(h0.max()),
        "rows": rows,
        "all_r3_bit_identical": all(r["r3_cells_unequal"] == 0 for r in rows),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--card", choices=sorted(CARDS), default="vec")
    parser.add_argument("--oracle-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)
    root = args.oracle_dir or DEFAULT_ROOTS[args.card]
    try:
        report = run(root, args.card, allow_dirty=args.allow_dirty)
    except GateError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True)
                               + "\n")
    for row in report["rows"]:
        print("s{stage} {operand:9s} cols={c:5d} r3_unequal={u:5d} "
              "r3_max_abs={m:.6e} implied_ht0_rel={h:.3e}".format(
                  stage=row["stage"], operand=row["operand"],
                  c=row["columns_tested"], u=row["r3_cells_unequal"],
                  m=row["r3_max_abs_diff"],
                  h=row["implied_ht0_max_rel_diff"]))
    print("all r3 bit-identical:", report["all_r3_bit_identical"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
