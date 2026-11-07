#!/usr/bin/env python3
"""Does the barotropic Coriolis OPERATOR differ, or only its operand?

Round 197's substitution arm shows that handing legoESM NEMO's recorded
per-substep 2-D Coriolis trend takes the end of the barotropic window from
``1.2459e-08`` to ``1.94e-16``.  That alone does NOT name the operator: the
trend is inside a recurrence, so clamping it at every substep also removes
whatever feedback an operand difference would have produced.  This probe
separates the two readings with the only test that can:

    apply legoESM's OWN frozen coefficients to NEMO's OWN recorded
    mid-step velocity, and compare against NEMO's recorded ``cor_u``.

* bit-equal (or at the rounding floor) -> the coefficients and the stencil
  are NEMO's; the trend is the CHANNEL the difference grows through, and the
  owner is upstream of ``dyn_cor_2D``.
* not equal -> the frozen coefficients differ, and the statement is named.

The instrument is self-checking against a KNOWN answer before it is used on
the unknown one: the same call on legoESM's own mid-step velocity must
reproduce legoESM's own recorded trend bit for bit.  If it does not, the
index convention is wrong and the probe refuses instead of reporting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "nemo_testcase_l1_vortex"))

from nemo_testcase_l1_vortex_kt2_walk import (  # noqa: E402
    _seed_from_record, _u_full, _v_full,
)
from nemo_testcase_l1_vortex_round196_spgts_walk import (  # noqa: E402
    CASE, DEFAULT_ROOT, _lego_plane, read_spgts,
)
from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    BAR, GateError, expected_masks, read_entry, require,
)

_COEFF_NAMES = ("ffu_nw", "ffu_ne", "ffu_sw", "ffu_se",
                "ffv_sw", "ffv_se", "ffv_nw", "ffv_ne")


def run(root: Path, *, kt: int = 1, allow_dirty: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        _nemo_literal_barotropic_coriolis,
    )
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
    from legoesm.ocean.fidelity.provenance import (
        allow_dirty_stamps, git_sha, worktree_stamp,
    )

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    # With a held patch applied the bare SHA says "-dirty" and nothing more;
    # the worktree stamp carries the patch's own diff hash.
    tree = worktree_stamp(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit), "production JIT is disabled")

    card = build_nemo_testcase_card(CASE)
    nlev = int(card.recipe.z_coord.n_levels)
    masks3 = expected_masks(card)
    mask_u = np.asarray(masks3["u"])[..., 0].astype(bool)
    mask_v = np.asarray(masks3["v"])[..., 0].astype(bool)
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    entry = read_entry(root / f"oracle_step_entry_kt{kt:08d}.bin", CASE,
                       expect_interior=interior)
    seed = _seed_from_record(card.recipe.initial_state, entry, nlev)
    meta, groups = read_spgts(root, kt)

    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True))
    trace = {k: np.asarray(v) for k, v in
             jax.device_get(model.step(seed, dt=card.dt_s)).substeps.items()}
    n_loop = meta["icycle"]

    # The coefficients are frozen at dyn_cor_2D_init; prove it from the trace
    # rather than assuming it, or the probe is using one substep's operand
    # with another substep's map.
    coeffs = {}
    for name in _COEFF_NAMES:
        stack = np.asarray(trace[name])
        require(stack.shape[0] == n_loop, f"{name}: not one frame per substep")
        drift = float(np.max(np.abs(stack - stack[0])))
        require(drift == 0.0,
                f"{name} is not frozen across the window (max drift {drift})")
        require(float(np.max(np.abs(stack[0]))) > 0.0,
                f"{name} is identically zero; this card does not carry the "
                f"literal coefficients and the probe cannot run")
        coeffs[name] = jnp.asarray(stack[0])

    rows = []
    for jn in range(1, n_loop + 1):
        prefix = f"j{jn:03d}_"
        u_lego = jnp.asarray(trace["u_mid"][jn - 1])
        v_lego = jnp.asarray(trace["v_mid"][jn - 1])
        u_nemo = jnp.asarray(_u_full(groups[prefix + "ua_ext"][..., None])[..., 0])
        v_nemo = jnp.asarray(_v_full(groups[prefix + "va_ext"][..., None])[..., 0])
        self_u, self_v = _nemo_literal_barotropic_coriolis(
            u_lego, v_lego, coeffs)
        cross_u, cross_v = _nemo_literal_barotropic_coriolis(
            u_nemo, v_nemo, coeffs)
        self_u = _lego_plane(np.asarray(self_u), "u")
        self_v = _lego_plane(np.asarray(self_v), "v")
        cross_u = _lego_plane(np.asarray(cross_u), "u")
        cross_v = _lego_plane(np.asarray(cross_v), "v")
        lego_u = _lego_plane(trace["cor_u"][jn - 1], "u")
        lego_v = _lego_plane(trace["cor_v"][jn - 1], "v")
        nemo_u = groups[prefix + "cor_u"]
        nemo_v = groups[prefix + "cor_v"]
        rows.append({
            "substep": jn,
            # KNOWN-ANSWER control: the reconstruction against the model's own
            # operand must reproduce the model's own trend bit for bit.
            "selfcheck_u": float(np.max(np.abs((self_u - lego_u)[mask_u]))),
            "selfcheck_v": float(np.max(np.abs((self_v - lego_v)[mask_v]))),
            # The question: the SAME operator on NEMO's OWN operand.
            "cross_u": float(np.max(np.abs((cross_u - nemo_u)[mask_u]))),
            "cross_v": float(np.max(np.abs((cross_v - nemo_v)[mask_v]))),
            "cross_cells_u": int(np.count_nonzero(
                (cross_u != nemo_u)[mask_u])),
            "cross_cells_v": int(np.count_nonzero(
                (cross_v != nemo_v)[mask_v])),
            # for scale: what the production trend difference is at the same
            # substep, which the cross row has to be compared against.
            "production_u": float(np.max(np.abs((lego_u - nemo_u)[mask_u]))),
            "production_v": float(np.max(np.abs((lego_v - nemo_v)[mask_v]))),
            "operand_u": float(np.max(np.abs(
                (np.asarray(_lego_plane(np.asarray(u_lego), "u"))
                 - groups[prefix + "ua_ext"])[mask_u]))),
        })
        # WHERE the operator disagrees, and against what local magnitude.  A
        # difference of 1e-53 in a cell whose own value is 1e-53 is the
        # rounding floor of an exponentially small far field, not a defect;
        # the same absolute number in a cell whose value is 1e-05 is one.
        delta = np.where(mask_u, np.abs(cross_u - nemo_u), 0.0)
        rowsum = delta > 0.0
        if rowsum.any():
            at = np.unravel_index(int(np.argmax(delta)), delta.shape)
            jj, ii = np.nonzero(rowsum)
            rows[-1].update({
                "cross_argmax_cell": [int(at[0]), int(at[1])],
                "cross_argmax_nemo_value": float(nemo_u[at]),
                "cross_argmax_lego_value": float(cross_u[at]),
                "cross_argmax_relative": (
                    float(delta[at] / abs(nemo_u[at]))
                    if nemo_u[at] != 0.0 else None),
                "cross_rows": [int(jj.min()), int(jj.max())],
                "cross_cols": [int(ii.min()), int(ii.max())],
                "cross_distinct_rows": int(len(set(jj.tolist()))),
                "cross_distinct_cols": int(len(set(ii.tolist()))),
                "nemo_peak": float(np.max(np.abs(nemo_u[mask_u]))),
            })

    selfcheck = max(max(r["selfcheck_u"], r["selfcheck_v"]) for r in rows)
    require(selfcheck == 0.0,
            "the reconstruction does not reproduce the model's own trend "
            f"(max {selfcheck:.3e}); the index convention is wrong and the "
            "cross row would be meaningless")
    cross = max(max(r["cross_u"], r["cross_v"]) for r in rows)
    production = max(max(r["production_u"], r["production_v"]) for r in rows)
    return {
        "case": CASE, "kt": kt, "git_sha": sha, "worktree": tree,
        "bar": BAR,
        "oracle_root": str(root), "substeps": n_loop,
        "selfcheck_max_abs": selfcheck,
        "cross_max_abs": cross,
        "production_max_abs": production,
        "verdict": ("OPERATOR-FAITHFUL" if cross <= BAR
                    else "OPERATOR-DIFFERS"),
        "rows": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--kt", type=int, default=1)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        report = run(args.oracle_root, kt=args.kt, allow_dirty=args.allow_dirty)
    except GateError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for row in report["rows"]:
        if row["substep"] in (1, 8, 16, 24, 28, 32, 40, report["substeps"]):
            print(f"j{row['substep']:03d} selfcheck={row['selfcheck_u']:.3e} "
                  f"cross_u={row['cross_u']:.3e} cells={row['cross_cells_u']:<5d} "
                  f"production_u={row['production_u']:.3e} "
                  f"operand_u={row['operand_u']:.3e}")
    print("selfcheck_max_abs:", report["selfcheck_max_abs"])
    print("cross_max_abs:", report["cross_max_abs"])
    print("production_max_abs:", report["production_max_abs"])
    print("VERDICT", report["verdict"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
