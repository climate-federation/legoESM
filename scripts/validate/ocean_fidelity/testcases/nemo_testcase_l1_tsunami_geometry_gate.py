#!/usr/bin/env python3
"""TSUNAMI round 3 -- the card's mesh IS the RK3 run's mesh_mask, bit for bit.

VORTEX's pattern (nemo_testcase_l1_vortex_smt_round2_geometry_gate.py): before
any time step is scored, every mesh array the card carries must equal the one
NEMO resolved and wrote to ``mesh_mask.nc`` (the case sets ``ln_meshmask``).
Zero ULP, not a tolerance.  Under key_vco_1d the file carries the vertical
ladders as ``*_1d`` only, so the card's per-column thickness/depth arrays are
compared with those at the executed level; the dummy bottom record (jpk = 2)
is reported, not scored, because the card does not carry it.

Pre-impl search (RULE 4): grepped scripts/validate/ocean_fidelity for a
mesh_mask comparison: the VORTEX gate above is the template and is case-
specific (zps, mbathy); ``require`` comes from the shared trajectory gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

DEFAULT_MESH = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round2/"
    "oracle_tsunami_r2/p3/mesh_mask.nc")


def card_rows(card, source) -> list[tuple[str, np.ndarray, str]]:
    """(row name, the card's array, the mesh_mask variable it must equal)."""
    g = card.recipe.grid
    z = card.recipe.z_coord
    ops = z.nemo_een_barotropic
    f64 = lambda a: np.asarray(a, dtype=np.float64)       # noqa: E731
    rows = [(n, f64(source[n]), n) for n in
            ("glamt", "glamu", "glamv", "glamf",
             "gphit", "gphiu", "gphiv", "gphif")]
    for n in ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f"):
        rows.append((f"card operand {n}", f64(getattr(ops, n)), n))
    # the grid object the solver also reads (u/v carry one redundant west row)
    rows += [
        ("grid dx_T", f64(g.dx_T), "e1t"), ("grid dy_T", f64(g.dy_T), "e2t"),
        ("grid dx_u", f64(g.dx_u)[:, 1:], "e1u"),
        ("grid dy_u", f64(g.dy_u)[:, 1:], "e2u"),
        ("grid dx_v", f64(g.dx_v)[1:, :], "e1v"),
        ("grid dy_v", f64(g.dy_v)[1:, :], "e2v"),
        ("source ff_t", f64(source["ff_t"]), "ff_t"),
        ("source ff_f", f64(source["ff_f"]), "ff_f"),
        ("grid f_T", f64(g.f_T), "ff_t"),
        ("card operand ff_f", f64(ops.ff_f), "ff_f"),
        ("grid ff_f", f64(g.ff_f), "ff_f"),
        ("card tmask (is_active)", f64(z.is_active)[..., 0], "tmask"),
        ("card umask", f64(ops.umask)[..., 0], "umask"),
        ("card vmask", f64(ops.vmask)[..., 0], "vmask"),
        ("card fmask", f64(ops.fmask)[..., 0], "fmask"),
        ("card e3t_0 (every column)", f64(z.nemo_e3t_0), "e3t_1d"),
        ("card e3w_0 (every column)", f64(z.nemo_e3w_0), "e3w_1d"),
        ("card gdept_0 (every column)", f64(z.nemo_gdept_0), "gdept_1d"),
        ("card gdepw_0 (every column)", f64(z.nemo_gdepw_0), "gdepw_1d"),
    ]
    return rows


def compare(mesh_path: Path, plant: str | None = None) -> dict:
    import netCDF4
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_tsunami_zco_card, tsunami_horizontal_coordinates)
    card = build_tsunami_zco_card()
    nlev = int(card.recipe.z_coord.n_levels)
    rows_in = card_rows(card, tsunami_horizontal_coordinates())
    names = [r[0] for r in rows_in]
    require(plant is None or plant in names,
            f"unknown plant {plant!r}; expected one of {names}")
    out = []
    with netCDF4.Dataset(mesh_path) as h:
        for name, lego, var in rows_in:
            nemo = np.asarray(h.variables[var][0], dtype=np.float64)
            if var in ("tmask", "umask", "vmask", "fmask"):
                nemo = nemo[:nlev][0] if nlev == 1 else nemo[:nlev]
            elif nemo.ndim == 1:      # the 1-d ladder, broadcast to every column
                nemo = np.broadcast_to(nemo[:nlev], lego.shape)
            lego = np.array(lego, dtype=np.float64)
            nemo = np.array(nemo)
            if plant == name:
                lego.flat[0] += 1.0
            require(lego.shape == nemo.shape,
                    f"{name}: card {lego.shape} vs NEMO {var} {nemo.shape}")
            diff = np.abs(lego - nemo)
            bad = np.argwhere(diff != 0)
            out.append({
                "row": name, "nemo_variable": var, "shape": list(lego.shape),
                "bit_identical": bool(np.array_equal(lego, nemo)),
                "n_differing": int(bad.shape[0]),
                "first_unequal_index": bad[0].tolist() if bad.size else None,
                "max_abs_difference": float(diff.max()), "planted": plant == name,
            })
        dummy = {v: np.asarray(h.variables[v][0]).reshape(-1).tolist()
                 for v in ("e3t_1d", "e3w_1d", "gdept_1d", "gdepw_1d")}
    return {"mesh_mask": str(mesh_path), "executed_levels": nlev,
            "rows": out, "nemo_1d_ladders_not_scored_beyond_executed": dummy,
            "all_bit_identical": all(r["bit_identical"] for r in out)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh-mask", type=Path, default=DEFAULT_MESH)
    p.add_argument("--json", type=Path)
    p.add_argument("--plant", help="perturb one element of the named card row")
    a = p.parse_args()
    res = compare(a.mesh_mask, a.plant)
    for r in res["rows"]:
        print(f"{'EXACT ' if r['bit_identical'] else 'DIFFERS'} "
              f"{r['row']:28s} n_diff={r['n_differing']:6d}")
    verdict = "GEOMETRY IDENTICAL" if res["all_bit_identical"] else "GEOMETRY DIFFERS"
    res["verdict"] = verdict
    print(verdict)
    if a.json:
        a.json.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    return 0 if res["all_bit_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
