#!/usr/bin/env python3
"""Round-42 GYRE ``zdf_mxl``/``ldf_slp`` given-input statement gate.

NEMO's compiled GYRE path builds ``nmln`` by the positive-N2 integral in
``zdfmxl.f90:109-124``.  ``ldf_slp`` then consumes Kmm live depths, stored
metric reciprocals and live face thicknesses (``ldfslp.f90:145-330``).  This
gate feeds the shared legoESM implementation NEMO's recorded T/S, prd, pn2,
e3w and Kmm SSH.  It scores the first discriminating arithmetic statement and
all four final slope fields at the immutable ``1e-15`` bar; exactness remains
the stronger zero-unequal-cell condition.

The record owns the 32x22 physical domain.  Its x neighbours are periodic and
its missing y neighbours are zero, the same convention used by the shared
model path.  No oracle output is injected as a model result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from legoesm.ocean.fidelity.provenance import worktree_stamp
from nemo_testcase_l2_gyre_phase3_gate import require, sha256
from nemo_testcase_l2_gyre_round35_trazdf_matrix import (
    _box,
    read_trazdf_matrix,
)
from nemo_testcase_l2_gyre_round38_matrix_operands import _oracle
from nemo_testcase_l2_gyre_round40_stage3_operators import (
    SLOPES_RECORD,
    SLOPES_ROOT,
    read_ldfslp,
)

CASE = "GYRE-zco"
BAR = 1.0e-15
TRACER_RECORD = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round38_oracle_trazdf_kt2/oracle_trazdf_matrix_kt00000002.bin"
)
EXPECTED_SELECTORS = {
    "mld_criterion": "n2_integral",
    "slope_metric_evaluation": "nemo_reciprocal",
    "slope_face_thickness_evaluation": "nemo_qco_live",
    "slope_depth_evaluation": "nemo_qco_live_literal",
    "redi_a33_evaluation": "nemo_literal",
}


def _row(name: str, reference: np.ndarray, actual: np.ndarray,
         mask: np.ndarray, statement: str) -> dict:
    reference = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    require(reference.shape == actual.shape == mask.shape,
            f"{name}: shape mismatch {reference.shape}, {actual.shape}, "
            f"{mask.shape}")
    require(mask.any(), f"{name}: empty score mask")
    delta = np.abs(actual - reference)
    refmax = float(np.max(np.abs(reference[mask])))
    absolute = float(np.max(delta[mask]))
    relative = absolute / max(refmax, np.finfo(np.float64).tiny)
    unequal = int(np.count_nonzero((actual != reference) & mask))
    return {
        "name": name,
        "statement": statement,
        "n": int(mask.sum()),
        "n_unequal": unequal,
        "absolute_max": absolute,
        "reference_max_abs": refmax,
        "relative_max_abs": relative,
        "bar": BAR,
        "exact": unequal == 0,
        "status": "AT-BAR" if relative <= BAR else "DEBT",
    }


def run(*, expect_commit: str, plant: bool = False) -> dict:
    import jax
    import jax.numpy as jnp

    from legoesm.core.precision import (
        PrecisionPolicy,
        get_policy,
        set_policy,
    )
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.experiments.dino import dino_config_for_recipe
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        _nemo_mld,
        compute_nemo_native_slopes,
        nemo_iso_a33,
        nemo_iso_face_masks,
    )

    stamp = worktree_stamp()
    expected = expect_commit.lower()
    actual = str(stamp["commit"]).lower()
    require(len(expected) == 40 and all(c in "0123456789abcdef" for c in expected),
            f"expected commit must be a full hexadecimal SHA: {expected!r}")
    require(actual == expected,
            f"commit stamp mismatch: report {actual}, expected {expected}")

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64/libm")
    require(jax.default_backend() == "cpu", "this gate is CPU-only")

    slope_path = SLOPES_ROOT / SLOPES_RECORD
    rec = read_ldfslp(slope_path)
    require(rec["kt"] == 2, f"{slope_path}: expected kt=2")
    a = rec["arrays"]
    tracer = read_trazdf_matrix(TRACER_RECORD, expect_kt=2)
    nlev = tracer["header"]["jpkm1"]
    require(nlev == 30, f"{TRACER_RECORD}: expected jpkm1=30")
    transpose = lambda x: np.ascontiguousarray(x.transpose(1, 0, 2))
    T = transpose(_box(tracer, "T_Kbb_in", nlev))
    S = transpose(_box(tracer, "S_Kbb_in", nlev))

    card = build_nemo_testcase_card(CASE)
    cfg = card.recipe.model_config
    gm = cfg.gm_redi
    resolved = {key: getattr(gm, key) for key in EXPECTED_SELECTORS}
    require(resolved == EXPECTED_SELECTORS,
            f"NEMO identity selector drift: {resolved}")
    dino = dino_config_for_recipe("nemo_dino_kamm")
    require(dino.gm_redi_mld_criterion == "n2_integral",
            "DINO's pre-existing NEMO criterion moved")

    init = card.recipe.initial_state
    mask2 = np.asarray(a["ssmask"]) > 0.0
    active = np.asarray(card.recipe.z_coord.is_active)
    r3t = np.asarray(a["r3t_Kmm"], dtype=np.float64)
    jacobian = 1.0 + r3t
    H = np.asarray(init.H_bathy.data, dtype=np.float64)
    eta = r3t * H
    eos_fn = make_eos_fn(cfg.eos, cfg.eos_linear)

    hml, m_base = _nemo_mld(
        gm.mld_criterion, jnp.asarray(T), jnp.asarray(S),
        jnp.asarray(init.land_mask.data), card.recipe.z_coord, eos_fn,
        gm.mld_rho_c, g=cfg.constants.g, rho_0=cfg.constants.rho_0,
        active_3d=jnp.asarray(active), jacobian=jnp.asarray(jacobian),
    )
    rows = [
        _row(
            f"{CASE}.kt2.zdfmxl.nmln",
            a["nmln"], np.asarray(m_base, dtype=np.float64) + 2.0, mask2,
            "zdfmxl.f90:115-124 positive-N2 integral and nmln/hmlp",
        ),
        _row(
            f"{CASE}.kt2.zdfmxl.hmlp",
            a["hmlp"], np.asarray(hml), mask2,
            "zdfmxl.f90:121-124 gdepw_1d(nmln)*(1+r3t(Kmm))",
        ),
    ]

    # First non-bit statement after selecting NEMO's MLD criterion.  NEMO
    # multiplies the stored reciprocal; the historical card divided here.
    # Both rows are computed by JAX at the same precision as the model path.
    level = slice(1, nlev)  # compiled loop jk=jpkm1..2; surface is unassigned
    zu = jnp.asarray(a["zgru_iik"][..., level])
    zv = jnp.asarray(a["zgrv_iik"][..., level])
    r1u = jax.lax.optimization_barrier(
        jnp.asarray(a["r1_e1u"])[..., None])
    r1v = jax.lax.optimization_barrier(
        jnp.asarray(a["r1_e2v"])[..., None])
    model_zau = jax.lax.optimization_barrier(zu * r1u)
    model_zav = jax.lax.optimization_barrier(zv * r1v)
    um = np.asarray(a["umask"][..., level]) > 0.0
    vm = np.asarray(a["vmask"][..., level]) > 0.0
    rows.extend([
        _row(
            f"{CASE}.kt2.ldfslp.zau",
            a["zau"][..., level], np.asarray(model_zau), um,
            "ldfslp.f90:225 zgru*r1_e1u",
        ),
        _row(
            f"{CASE}.kt2.ldfslp.zav",
            a["zav"][..., level], np.asarray(model_zav), vm,
            "ldfslp.f90:226 zgrv*r1_e2v",
        ),
    ])

    # Rule 10: the final score is through the shared production function,
    # with NEMO operands replacing inputs rather than outputs.
    rho = cfg.constants.rho_0 * (np.asarray(a["prd"][..., :nlev]) + 1.0)
    native = compute_nemo_native_slopes(
        jnp.asarray(rho), jnp.asarray(T), jnp.asarray(S),
        jnp.asarray(init.land_mask.data), jnp.asarray(init.u_mask.data),
        jnp.asarray(init.v_mask.data), card.recipe.z_coord, card.recipe.grid,
        gm, eos_fn, rho_0=cfg.constants.rho_0, g=cfg.constants.g,
        active_3d=jnp.asarray(active), jacobian=jnp.asarray(jacobian),
        eta=jnp.asarray(eta), H_bathy=jnp.asarray(H),
        prd_override=jnp.asarray(a["prd"][..., :nlev]),
        pn2_override=jnp.asarray(a["pn2"][..., :nlev]),
        e3w_override=jnp.asarray(a["e3w_Kmm"][..., :nlev]),
    )
    final_masks = {
        "uslp": np.asarray(a["umask"][..., :nlev]) > 0.0,
        "vslp": np.asarray(a["vmask"][..., :nlev]) > 0.0,
        "wslpi": np.asarray(a["wmask"][..., :nlev]) > 0.0,
        "wslpj": np.asarray(a["wmask"][..., :nlev]) > 0.0,
    }
    for field, actual in zip(("uslp", "vslp", "wslpi", "wslpj"), native):
        reference = np.array(a[field][..., :nlev], copy=True)
        score_mask = final_masks[field]
        score_mask[..., 0] = False
        if plant and field == "uslp":
            planted = np.where(score_mask, np.abs(reference), -1.0)
            index = np.unravel_index(np.argmax(planted), planted.shape)
            reference[index] += np.max(planted) * 1.0e-12
        rows.append(_row(
            f"{CASE}.kt2.ldfslp.{field}", reference, np.asarray(actual),
            score_mask,
            "ldfslp.f90:262-330 Shapiro-filtered native slope output",
        ))

    # The round-38 matrix record is the same kt=2 before state and carries
    # NEMO's resulting ah_wslp2.  Fold the four NEMO-recorded slopes through
    # the shared production a33 function: this isolates the downstream square
    # association from the still-open model-side prd/rn2 producer boundary.
    um3, vm3, wm3 = nemo_iso_face_masks(
        jnp.asarray(init.u_mask.data), jnp.asarray(init.v_mask.data),
        jnp.asarray(active))
    aht = jnp.ones_like(jnp.asarray(active), dtype=jnp.float64) * 1000.0
    geom = card.recipe.grid
    ah_wslp2, _ = nemo_iso_a33(
        aht, um3, vm3, wm3,
        jnp.asarray(a["wslpi"][..., :nlev]),
        jnp.asarray(a["wslpj"][..., :nlev]),
        jnp.asarray(geom.dx_u[:, 1:]), jnp.asarray(geom.dy_v[1:]),
        jnp.ones_like(aht), evaluation=gm.redi_a33_evaluation)
    matrix_oracle = _oracle(tracer)
    wet = np.asarray(matrix_oracle["wet"]) > 0.0
    wet_face = wet[..., 1:] & wet[..., :-1]
    rows.append(_row(
        f"{CASE}.kt2.traldf_iso.ah_wslp2",
        matrix_oracle["K33"], np.asarray(ah_wslp2[..., 1:]), wet_face,
        "traldf_iso.f90:793-794 (zahu*wslpi)*wslpi + "
        "(zahv*wslpj)*wslpj",
    ))

    # A measured counterfactual, not an accepted row: it identifies the first
    # post-decision statement which the former association failed.
    old_zau = np.asarray(zu / jnp.asarray(a["e1u"])[..., None])
    old_zav = np.asarray(zv / jnp.asarray(a["e2v"])[..., None])
    baseline = {
        "statement": "ldfslp.f90:225-226 stored reciprocal multiply",
        "zau_cells_unequal": int(np.count_nonzero(
            (old_zau != a["zau"][..., level]) & um)),
        "zav_cells_unequal": int(np.count_nonzero(
            (old_zav != a["zav"][..., level]) & vm)),
    }
    require(baseline["zau_cells_unequal"] > 0,
            "the preregistered first non-bit statement is not discriminating")
    require(baseline["zav_cells_unequal"] > 0,
            "the preregistered first non-bit statement is not discriminating")

    status = "AT-BAR" if all(r["status"] == "AT-BAR" for r in rows) else "DEBT"
    exact = all(r["exact"] for r in rows)
    report = {
        "format": "nemo-testcase-l2-gyre-round42-ldfslp-walk-v1",
        "worktree": stamp,
        "case": CASE,
        "bar": BAR,
        "execution_regime": "production_jit",
        "precision_policy": "fp64/libm",
        "jax_backend": jax.default_backend(),
        "record": str(slope_path),
        "record_sha256": sha256(slope_path),
        "tracer_record": str(TRACER_RECORD),
        "tracer_record_sha256": sha256(TRACER_RECORD),
        "resolved_selectors": resolved,
        "dino_resolved_mld_criterion": dino.gm_redi_mld_criterion,
        "first_nonbit_after_decision23": baseline,
        "rows": rows,
        "all_rows_exact": exact,
        "status": status,
        "planted_control": plant,
    }
    if plant:
        require(not exact and status == "DEBT",
                "the planted final-slope violation did not fire")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expect-commit", required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    report = run(expect_commit=args.expect_commit, plant=args.plant)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    for row in report["rows"]:
        print(f"{row['status']:<8} {row['name']:<38} "
              f"unequal {row['n_unequal']}/{row['n']} "
              f"max {row['absolute_max']:.17g}")
    first = report["first_nonbit_after_decision23"]
    print("FIRST-NONBIT", first["statement"],
          f"zau={first['zau_cells_unequal']}",
          f"zav={first['zav_cells_unequal']}")
    print("STATUS", report["status"])
    if args.plant:
        return 1
    return 0 if report["status"] == "AT-BAR" and report["all_rows_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
