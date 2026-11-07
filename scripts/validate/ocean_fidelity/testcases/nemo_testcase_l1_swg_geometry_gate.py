#!/usr/bin/env python3
"""SWG round-1 zero-ULP geometry gate against the operator probe mesh."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nemo_testcase_phase3_trajectory_gate import require  # noqa: E402

DEFAULT_MESH = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/"
    "round0_probe/kt10/mesh_mask.nc")
PLANTS = ("signed_y_coriolis", "wet_cell", "layer_thickness")


def card_rows(card, source) -> list[tuple[str, np.ndarray, str]]:
    """Return card/source rows and their authoritative mesh variables."""
    g, z = card.recipe.grid, card.recipe.z_coord
    ops = z.nemo_een_barotropic
    f64 = lambda a: np.asarray(a, dtype=np.float64)  # noqa: E731
    rows = [(n, f64(source[n]), n) for n in
            ("glamt", "glamu", "glamv", "glamf",
             "gphit", "gphiu", "gphiv", "gphif")]
    for n in ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f"):
        rows.append((f"card operand {n}", f64(getattr(ops, n)), n))
    rows += [
        ("grid dx_T", f64(g.dx_T), "e1t"),
        ("grid dy_T", f64(g.dy_T), "e2t"),
        ("grid dx_u", f64(g.dx_u)[:, 1:], "e1u"),
        ("grid dy_u", f64(g.dy_u)[:, 1:], "e2u"),
        ("grid dx_v", f64(g.dx_v)[1:], "e1v"),
        ("grid dy_v", f64(g.dy_v)[1:], "e2v"),
        ("source ff_t", f64(source["ff_t"]), "ff_t"),
        ("source ff_f", f64(source["ff_f"]), "ff_f"),
        ("grid f_T", f64(g.f_T), "ff_t"),
        ("grid ff_f", f64(g.ff_f), "ff_f"),
        ("card tmask", f64(z.is_active)[..., 0], "tmask"),
        ("card umask", f64(ops.umask)[..., 0], "umask"),
        ("card vmask", f64(ops.vmask)[..., 0], "vmask"),
        ("card fmask", f64(ops.fmask)[..., 0], "fmask"),
        ("card e3t_0", f64(z.nemo_e3t_0), "e3t_1d"),
        ("card e3w_0", f64(z.nemo_e3w_0), "e3w_1d"),
        ("card gdept_0", f64(z.nemo_gdept_0), "gdept_1d"),
        ("card gdepw_0", f64(z.nemo_gdepw_0), "gdepw_1d"),
    ]
    return rows


def compare(mesh_path: Path, plant: str | None = None, *,
            allow_dirty: bool = False) -> dict:
    """Compare every preregistered row; plants alter card-side operands."""
    import netCDF4
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        SWG_NAMELIST, build_swg_card, swg_horizontal_coordinates)
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha
    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    require(plant is None or plant in PLANTS,
            f"unknown plant {plant!r}; expected one of {PLANTS}")
    card = build_swg_card()
    source = swg_horizontal_coordinates()
    if plant == "signed_y_coriolis":
        source = dict(source)
        source["ff_t"] = (SWG_NAMELIST.rn_f0_s
                          + SWG_NAMELIST.rn_beta_m_s * source["gphit"])
    rows = card_rows(card, source)
    nlev = int(card.recipe.z_coord.n_levels)
    out = []
    with netCDF4.Dataset(mesh_path) as ds:
        for name, lego, var in rows:
            nemo = np.asarray(ds.variables[var][0], dtype=np.float64)
            if var in ("tmask", "umask", "vmask", "fmask"):
                nemo = nemo[0]
            elif nemo.ndim == 1:
                nemo = np.broadcast_to(nemo[:nlev], lego.shape)
            lego = np.array(lego, copy=True, dtype=np.float64)
            if plant == "wet_cell" and name == "card tmask":
                lego[1, 1] = 0.0
            if plant == "layer_thickness" and name == "card e3t_0":
                lego.flat[0] += 1.0
            require(lego.shape == nemo.shape,
                    f"{name}: card {lego.shape} vs {var} {nemo.shape}")
            diff = np.abs(lego - nemo)
            bad = np.argwhere(diff != 0.0)
            out.append({
                "row": name, "nemo_variable": var,
                "bit_identical": bool(np.array_equal(lego, nemo)),
                "n_differing": int(bad.shape[0]),
                "first_unequal_index": bad[0].tolist() if bad.size else None,
                "max_abs_difference": float(diff.max()),
            })
        ladders = {n: np.asarray(ds.variables[n][0]).tolist() for n in
                   ("e3t_1d", "e3w_1d", "gdept_1d", "gdepw_1d")}
    return {
        "mesh_mask": str(mesh_path), "dtype": "float64",
        "legoesm_git_sha": sha,
        "executed_levels": nlev, "rows": out,
        "nemo_full_ladders": ladders,
        "all_bit_identical": all(r["bit_identical"] for r in out),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mesh-mask", type=Path, default=DEFAULT_MESH)
    p.add_argument("--json", type=Path)
    p.add_argument("--plant", choices=PLANTS)
    p.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args(argv)
    try:
        result = compare(a.mesh_mask, a.plant, allow_dirty=a.allow_dirty)
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    result["verdict"] = ("GEOMETRY IDENTICAL" if result["all_bit_identical"]
                         else "GEOMETRY DIFFERS")
    for row in result["rows"]:
        print(f"{'EXACT' if row['bit_identical'] else 'DIFFERS'} "
              f"{row['row']} n_diff={row['n_differing']}")
    print(result["verdict"])
    if a.json:
        a.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["all_bit_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
