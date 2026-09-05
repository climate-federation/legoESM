#!/usr/bin/env python3
"""Round-15 causal gate for the one-layer RK3/ZDF momentum owner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.core.source_rounding import nemo_source_round as rnd
from scripts.validate.ocean_fidelity.testcases.nemo_rung36_ocean_gate import (
    _records,
    _row,
    evaluate_continuous,
)


OWNER_FIELDS = (
    "u_stage0", "u_after_baro_strip", "u_after_bottom", "u_after_top",
    "u_diag0", "u_diag_bottom", "u_diag_top", "u_lower", "u_upper",
    "u_rhs0", "u_rhs_stress", "u_solution", "ubar_after", "e3u_after",
    "utauU", "rCdU_bot", "rCdU_bot_east", "rCdU_top",
    "rCdU_top_east", "umask",
    "v_stage0", "v_after_baro_strip", "v_after_bottom", "v_after_top",
    "v_diag0", "v_diag_bottom", "v_diag_top", "v_lower", "v_upper",
    "v_rhs0", "v_rhs_stress", "v_solution", "vbar_after", "e3v_after",
    "vtauV", "rCdU_bot", "rCdU_bot_north", "rCdU_top",
    "rCdU_top_north", "vmask", "dt", "half_dt", "rho0",
)
RHS_FIELDS = (
    "u_before", "u_rhs", "r3u_before", "r3u_now", "r3u_after",
    "u_after", "v_before", "v_rhs", "r3v_before", "r3v_now",
    "r3v_after", "v_after", "dt",
)
STAGE_FIELDS = (
    "u_post_zdf", "v_post_zdf", "ubar_after", "vbar_after",
    "u_correction", "v_correction", "u_after_correction",
    "v_after_correction", "u_after_lbc", "v_after_lbc", "u_final",
    "v_final", "ssh_after", "r3u_after", "r3v_after", "temperature",
    "salinity", "e3u_after",
)


def _qco_rhs(before, rhs, r3_before, r3_now, r3_after, dt, mask):
    """Replay dynzdf.F90:127-132 with one rounding guard per statement."""
    one = jnp.asarray(1.0, dtype=jnp.asarray(before).dtype)
    before_term = rnd(rnd(one + r3_before) * before)
    rhs_term = rnd(rnd(dt * rnd(one + r3_now)) * rhs)
    numerator = rnd(before_term + rhs_term)
    return rnd(rnd(numerator / rnd(one + r3_after)) * mask)


def evaluate(root: Path, plant: bool = False) -> dict[str, object]:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    owner_header, owner = _records(
        root / "oracle_rung36_dynzdf_owner_frames.bin",
        b"NEMO_L3ZOWN_001 ", "8i", len(OWNER_FIELDS))[0]
    rhs_header, rhs = _records(
        root / "oracle_rung36_dynzdf_rhs_frames.bin",
        b"NEMO_L3ZRHS_001 ", "8i", len(RHS_FIELDS))[0]
    stage_header, stage = _records(
        root / "oracle_rung36_stage_end_frames.bin",
        b"NEMO_L3SEND_001 ", "8i", len(STAGE_FIELDS))[0]

    got_u_rhs = _qco_rhs(
        *[jnp.asarray(rhs[index]) for index in (0, 1, 2, 3, 4)],
        jnp.asarray(rhs[12]), jnp.asarray(1.0))
    got_v_rhs = _qco_rhs(
        *[jnp.asarray(rhs[index]) for index in (6, 7, 8, 9, 10)],
        jnp.asarray(rhs[12]), jnp.asarray(1.0))
    got_u_after = rnd(jnp.asarray(stage[0]) + jnp.asarray(stage[4]))
    got_v_after = rnd(jnp.asarray(stage[1]) + jnp.asarray(stage[5]))
    if plant:
        got_u_after = got_u_after + jnp.asarray(1.0e-8)

    rows = [
        _row("DYN_ZDF.QCO_RHS.u", got_u_rhs, rhs[5]),
        _row("DYN_ZDF.QCO_RHS.v", got_v_rhs, rhs[11]),
        _row("POST_ZDF_MEAN.u", got_u_after, stage[6]),
        _row("POST_ZDF_MEAN.v", got_v_after, stage[7]),
        _row("POST_LBC.u", stage[8], stage[10]),
        _row("POST_LBC.v", stage[9], stage[11]),
    ]
    scales = {}
    for scale in (0.0, 0.5, 1.0):
        report = evaluate_continuous(
            root, post_zdf_mean_scale=scale)
        scales[f"{scale:g}"] = {
            row["name"].rsplit(".", 1)[-1]: row["max_abs"]
            for row in report["rows"][:2]
        }
    first = next((row for row in rows if row["status"] == "DEBT"), None)
    return {
        "verdict": "AT_BAR" if first is None else "DEBT",
        "headers": {
            "dynzdf_owner": list(owner_header),
            "dynzdf_rhs": list(rhs_header),
            "stage_end": list(stage_header),
        },
        "rows": rows,
        "first_over_bar": first,
        "discriminator": {
            "u_Kbb": float(rhs[0]),
            "u_Krhs": float(rhs[1]),
            "v_Kbb": float(rhs[6]),
            "v_Krhs": float(rhs[7]),
            "first_nonzero_owner": "dynzdf.F90:127-132 Krhs term",
        },
        "one_variable_scaling": {
            "hook": "_NEMOWSRK3TestHooks.post_zdf_mean_scale",
            "nemo_switch": False,
            "rows": scales,
        },
        "plant": "POST_ZDF_MEAN.u + 1e-8" if plant else None,
        "plant_binding": None if not plant else {
            "row": "POST_ZDF_MEAN.u",
            "red": rows[2]["status"] == "DEBT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.root, args.plant)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.plant:
        return 1 if report["plant_binding"]["red"] else 2
    return 0 if report["verdict"] == "AT_BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
