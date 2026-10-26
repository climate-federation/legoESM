#!/usr/bin/env python3
"""Round-30 GYRE stage-3 momentum-RHS boundary gate: who owns the divergence.

WHY THIS EXISTS.  GYRE's kt=1 momentum divergence is localised per stage ---
stage 1 ``2.710505431213761e-19`` (AT-BAR), stage 2 ``4.740083109008864e-13``,
stage 3 ``9.481924730527598e-07`` --- so essentially all of it enters at stage
3.  Exactly one momentum call sits inside ``stprk3_stg.F90``'s ``CASE ( 3 )``
on this deck, ``dyn_ldf`` at ``stprk3_stg.F90:400``; ``dyn_zdf`` follows it
outside the ``SELECT CASE`` at ``stprk3_stg.F90:430``.  (``dyn_osm``,
``bdy_dyn3d_dmp`` and ``dyn_dmp`` at ``:402``, ``:404`` and ``:406`` are gated
on ``ln_zdfosm``, ``ln_bdy`` and ``ln_dyndmp .AND. ln_c1d``, all resolved
false on GYRE.)  The round-29 instrument dumped the two frames that separate
them, and this gate scores legoESM against both:

* ``pre_ldf`` --- ``oracle_rkstage3_preldf_kt00000001.bin``
  (``NEMO_L2_RKPLD_1``), ``uu/vv(:,:,:,Krhs)`` immediately BEFORE
  ``dyn_ldf``.  A bit-unequal cell here puts the owner at or before
  ``dyn_hpg``/``dyn_vor``/``dyn_adv``, i.e. UPSTREAM of both stage-3-only
  operators.
* ``post_ldf`` --- ``uu_Krhs_in``/``vv_Krhs_in`` inside
  ``oracle_zdf_matrix_kt00000001.bin`` (``NEMO_L2_ZDFMX_1``), the same array
  as ``dyn_zdf`` receives it.  Exact ``pre_ldf`` with non-exact ``post_ldf``
  names ``dyn_ldf``; exact on both puts the owner inside ``dyn_zdf``.

The model side is the production step under production JIT with one WRITE-only
hook (``expose_stage3_momentum_rhs``); the ordinary step completes before the
diagnostic arrays are substituted, so this scores the compiled production
arithmetic and not an eager re-evaluation.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
from nemo_testcase_l2_gyre_phase3_gate import (
    CASE,
    DIMS,
    _surface_forcings,
    _xyz,
    expected_masks,
    lego_fields,
    require,
    score,
    sha256,
)
from nemo_testcase_l2_gyre_round29_zdf_matrix import read_zdf_matrix
from legoesm.ocean.fidelity.provenance import worktree_stamp

ORACLE_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round29_oracle_v2_zdf_matrix")
PRE_LDF_RECORD = "oracle_rkstage3_preldf_kt00000001.bin"
ZDF_MATRIX_RECORD = "oracle_zdf_matrix_kt00000001.bin"
# The round-29 acquisition is admitted against this round-19 twin by
# ``nemo_testcase_l2_gyre_round21_admission.py``; the restart identity is the
# one-line proof that the two added WRITE-only records changed no model state.
ADMISSION_BASELINE = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
    "round19_oracle_v2_external")


def read_pre_ldf(path: Path) -> dict[str, np.ndarray]:
    """Read the stage-3 pre-``dyn_ldf`` momentum RHS frame."""
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump

    require(time_level_for_dump(path.name) == "now",
            f"{path}: registered time level is not the stage's live Kmm")
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        header = struct.unpack("=11i", handle.read(44))
        values = np.fromfile(handle, dtype=np.float64)
    version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_RKPLD_1", f"{path}: bad magic {magic!r}")
    require(
        (version, kt, stage, kbb, kmm, krhs, kaa, nx, ny, nz, bits)
        == (1, 1, 3, 1, 2, 3, 3, *DIMS, 64),
        f"{path}: bad header {header}")
    n3 = nx * ny * nz
    require(values.size == 2 * n3, f"{path}: bad payload")
    require(np.all(np.isfinite(values)), f"{path}: non-finite payload")
    return {"u": _xyz(values[:n3], nx, ny, nz),
            "v": _xyz(values[n3:], nx, ny, nz)}


def read_post_ldf(path: Path) -> dict[str, np.ndarray]:
    """The Krhs ``dyn_zdf`` receives, out of the round-29 matrix record."""
    rec = read_zdf_matrix(path)
    nx, ny, nz = DIMS
    require(tuple(rec["header"][k] for k in ("jpi", "jpj", "jpk")) == DIMS,
            f"{path}: record dimensions are not GYRE's")

    def strip(name: str) -> np.ndarray:
        return _xyz(np.asarray(rec["arrays"][name]).ravel(order="F"), nx, ny, nz)

    return {"u": strip("uu_Krhs_in"), "v": strip("vv_Krhs_in")}


def run(mode: str, oracle_root: Path, *, plant: bool = False) -> dict:
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
        _NEMOWSRK3TestHooks,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

    require(mode in ("pre_ldf", "post_ldf"), f"unknown mode {mode!r}")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")
    require(bool(jax.config.jax_enable_x64), "JAX x64 is disabled")
    require(not bool(jax.config.jax_disable_jit),
            "certification requires production JIT; JAX_DISABLE_JIT is forbidden")

    record = oracle_root / (
        PRE_LDF_RECORD if mode == "pre_ldf" else ZDF_MATRIX_RECORD)
    oracle = (read_pre_ldf(record) if mode == "pre_ldf"
              else read_post_ldf(record))
    restart = oracle_root / "GYRE_OMIP_L2_P3_00000010_restart.nc"
    baseline_restart = ADMISSION_BASELINE / restart.name
    require(sha256(restart) == sha256(baseline_restart),
            "round-29 instrumentation moved the final restart; the record is "
            "not WRITE-only and no number from it may be quoted")

    card = build_nemo_testcase_card(CASE)
    # Same resolved configuration the trajectory rows are scored under
    # (nemo_testcase_l2_gyre_phase3_gate.run): NEMO QCO carries E-P as volume
    # with sfx = 0 (usrdef_sbc.F90:138-145).
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    masks = expected_masks(card)
    freshwater, surface = _surface_forcings(card, card.recipe.initial_state, 1)
    state = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_stage3_momentum_rhs=mode),
    ).step(
        card.recipe.initial_state, dt=card.dt_s,
        freshwater=freshwater, surface_forcing=surface,
    )
    state = jax.tree_util.tree_map(
        lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, state)
    fields = lego_fields(state)

    rows = []
    for component in ("u", "v"):
        candidate = fields[component]
        rows.append(score(
            f"{CASE}.kt1.stage3.{mode}_rhs.{component}",
            oracle[component][..., :candidate.shape[-1]],
            candidate, masks[component], plant=plant and component == "u"))
    status = "AT-BAR" if all(r["status"] == "AT-BAR" for r in rows) else "DEBT"
    exact = all(r["exact"] for r in rows)
    report = {
        "worktree": worktree_stamp(),
        "format": "nemo-testcase-l2-gyre-round30-stage3-owner-v1",
        "case": CASE,
        "mode": mode,
        "boundary": ("stprk3_stg.F90:400 (the Krhs dyn_ldf receives)"
                     if mode == "pre_ldf" else
                     "stprk3_stg.F90:430 (the Krhs dyn_zdf receives)"),
        "status": status,
        "bit_exact": exact,
        "execution_regime": "production_jit",
        "precision_policy": "fp64",
        "jax_backend": jax.default_backend(),
        "record": str(record),
        "record_sha256": sha256(record),
        "restart_sha256": sha256(restart),
        "rows": rows,
        "planted_control": plant,
    }
    if plant:
        require(status == "DEBT" and rows[0]["absolute_max"] >= 0.9,
                "planted stage-3 RHS violation did not fire")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("pre_ldf", "post_ldf"))
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args(argv)
    report = run(args.mode, args.oracle_root, plant=args.plant)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)
    for row in report["rows"]:
        print(f"{row['status']:<8} {row['name']:<42} "
              f"bit_unequal {row['n_unequal']}/{row['n']} "
              f"max {row['absolute_max']:.17g}")
    print(f"STATUS {report['status']}")
    if args.plant:
        return 1
    return 0 if report["status"] == "AT-BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
