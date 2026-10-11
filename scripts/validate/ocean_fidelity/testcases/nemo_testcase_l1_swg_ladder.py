#!/usr/bin/env python3
"""SWG kt=1..10 ladder: restart after-states now, stage entries when acquired."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nemo_testcase_phase3_trajectory_gate import BAR, require, score  # noqa: E402

DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/"
    "round0_probe/kt10")
DEFAULT_STAGE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/"
    "round1/oracle_swg_r1/run")
RESTART_FIELDS = (
    "sshn", "un", "vn", "uu_n", "vv_n", "utau_b", "vtau_b",
    "emp_b", "qns_b", "sfx_b",
)
U_FIELDS = frozenset(("un", "uu_n"))
V_FIELDS = frozenset(("vn", "vv_n"))


def _stage_checker():
    path = HERE / "nemo_testcase_l1_swg/check_records.py"
    spec = importlib.util.spec_from_file_location("swg_check_records", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def read_restarts(root: Path) -> dict[int, dict[str, np.ndarray]]:
    """Read the ten restart AFTER states, refusing omissions or hidden casts."""
    import netCDF4
    files = sorted(root.glob("SWG_????????_restart.nc"))
    expected = [root / f"SWG_{kt:08d}_restart.nc" for kt in range(1, 11)]
    require(files == expected, f"restart sequence is {[p.name for p in files]}")
    out = {}
    for kt, path in enumerate(files, 1):
        with netCDF4.Dataset(path) as ds:
            ds.set_auto_mask(False)
            file_kt = int(np.asarray(ds["kt"][:]).item())
            require(file_kt == kt, f"{path.name}: kt={file_kt}")
            require(float(np.asarray(ds["rdt"][:]).item()) == 3600.0,
                    f"{path.name}: rdt is not 3600 s")
            row = {}
            for name in RESTART_FIELDS:
                require(name in ds.variables, f"{path.name}: missing {name}")
                var = ds[name]
                require(var.dtype == np.dtype("float64"),
                        f"{path.name}.{name}: dtype {var.dtype}")
                a = np.asarray(var[0], dtype=np.float64)
                if name in ("un", "vn"):
                    require(a.shape[0] == 2, f"{path.name}.{name}: no dummy record")
                    a = a[0]                 # executed level; crop dummy bottom
                require(a.shape == (84, 84) and np.all(np.isfinite(a)),
                        f"{path.name}.{name}: bad shape/non-finite")
                row[name] = a
            out[kt] = row
    return out


def read_stage_entries(root: Path) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    """Read all acquired stage entries through the admission parser."""
    cr = _stage_checker()
    out = {}
    for kt in range(1, 11):
        for stage in (1, 2, 3):
            path = root / f"oracle_swgstage_kt{kt:08d}_s{stage}.bin"
            require(path.is_file(), f"missing {path}")
            head, groups = cr.parse(path)
            require((head[1], head[2]) == (kt, stage),
                    f"{path.name}: header {(head[1], head[2])}")
            is0, js0, ie0, je0 = head[cr.BOUNDS_AT:cr.BOUNDS_AT + 4]
            out[kt, stage] = {
                name: np.asarray(a[js0 - 1:je0, is0 - 1:ie0])
                for name, a in groups.items()
            }
    return out


def _card_masks(card) -> dict[str, np.ndarray]:
    """Return native NEMO T/U/V masks from the card and shared face builder."""
    from legoesm.ocean.dynamics.latlon_cgrid_operators import compute_face_masks

    tmask = np.asarray(card.recipe.land_mask, dtype=bool)
    u_full, v_full = compute_face_masks(card.recipe.land_mask, card.recipe.grid)
    return {
        "T": tmask,
        "U": np.asarray(u_full[:, 1:], dtype=bool),
        "V": np.asarray(v_full[1:, :], dtype=bool),
    }


def _field_mask(field: str, masks: dict[str, np.ndarray]) -> np.ndarray:
    return masks["U" if field in U_FIELDS else "V" if field in V_FIELDS else "T"]


def _row(name, ref, cand, mask):
    ref, cand = np.asarray(ref), np.asarray(cand)
    use = np.asarray(mask, dtype=bool)
    result = score(name, ref, cand, use)
    delta = cand - ref
    bad = np.argwhere((ref != cand) & use)
    result.update({
        "bit_identical": bool(bad.size == 0),
        "n_unequal": int(bad.shape[0]),
        "first_unequal_cell": bad[0].tolist() if bad.size else None,
        "max_abs": float(np.max(np.abs(delta[use]))),
    })
    return result


def _gate_rows(out: dict) -> dict:
    rows = [row for step in out.get("per_kt", ()) for row in step["rows"]]
    out["all_at_bar"] = all(row["status"] == "AT-BAR" for row in rows)
    out["all_bit_identical"] = all(row["bit_identical"] for row in rows)
    out["first_kt_over_bar"] = next(
        (step["kt"] for step in out.get("per_kt", ())
         if any(row["status"] != "AT-BAR" for row in step["rows"])), None)
    return out


def _exit_code(out: dict) -> int:
    return 0 if out.get("all_at_bar", True) else 1


def self_check(root: Path, *, plant: bool = False, allow_dirty: bool = False) -> dict:
    """Calibrate restart decoding only; this is not a fidelity measurement."""
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha
    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_swg_card
    masks = _card_masks(build_swg_card())
    records = read_restarts(root)
    per_kt = []
    for kt, fields in records.items():
        rows = []
        for name, ref in fields.items():
            cand = ref.copy()
            if plant and kt == 1 and name == "sshn":
                cand[42, 42] += 1.0e-3
            rows.append(_row(f"self_check.kt{kt}.{name}", ref, cand,
                             _field_mask(name, masks)))
        per_kt.append({"kt": kt, "rows": rows})
    return _gate_rows({
        "label": "RESTART-READER-SELF-CHECK (not fidelity)",
        "legoesm_git_sha": sha, "bar": BAR, "plant": plant,
        "per_kt": per_kt,
    })


def independent(root: Path, *, allow_dirty: bool = False) -> dict:
    """Run the real chained ladder; round 1 must refuse at the execution gate."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_swg_card, validate_nemo_testcase_card_for_execution)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(jax.default_backend() == "cpu", "ladder must run on CPU")
    card = build_swg_card()
    masks = _card_masks(card)
    validate_nemo_testcase_card_for_execution(card)
    records = read_restarts(root)
    model = LatLonCGridOceanModel(card.recipe.grid, card.recipe.z_coord,
                                  card.recipe.model_config)
    state = card.recipe.initial_state
    per_kt = []
    for kt in range(1, 11):
        state = model.step(state, card.dt_s)
        cand = {
            "sshn": np.asarray(state.eta.data),
            "un": np.asarray(state.u.data)[:, 1:, 0],
            "vn": np.asarray(state.v.data)[1:, :, 0],
            "uu_n": np.asarray(state.uu_b.data)[:, 1:],
            "vv_n": np.asarray(state.vv_b.data)[1:, :],
        }
        rows = [_row(f"independent.kt{kt}.{n}", records[kt][n], cand[n],
                     _field_mask(n, masks))
                for n in cand]
        per_kt.append({"kt": kt, "rows": rows})
    return _gate_rows({"label": "INDEPENDENT", "bar": BAR,
                       "per_kt": per_kt})


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("self-check", "independent", "stage-entry-read"),
                   required=True)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    p.add_argument("--plant", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--json", type=Path)
    a = p.parse_args(argv)
    try:
        if a.arm == "self-check":
            out = self_check(a.root, plant=a.plant, allow_dirty=a.allow_dirty)
        elif a.arm == "independent":
            out = independent(a.root, allow_dirty=a.allow_dirty)
        else:
            entries = read_stage_entries(a.stage_root)
            out = {"label": "STAGE-ENTRY-READ", "records": len(entries),
                   "keys": [list(k) for k in entries]}
    except Exception as exc:  # noqa: BLE001 - fail-closed CLI boundary
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 2
    if a.json:
        a.json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(out["label"])
    if "all_at_bar" in out:
        print(f"all_at_bar={out['all_at_bar']} bit_identical={out['all_bit_identical']}")
        return _exit_code(out)
    print(f"records={out.get('records', len(out.get('per_kt', [])))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
