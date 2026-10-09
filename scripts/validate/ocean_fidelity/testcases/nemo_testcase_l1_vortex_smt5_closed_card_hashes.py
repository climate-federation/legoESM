#!/usr/bin/env python
"""SHA-256 of each closed card's prognostic state after N production steps.

Run once at the parent commit and once after a card-selected option lands;
identical digests prove the option leaves every card that does not select it
bit-identical.  fp64, scalar libm, the production ``model.step`` closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

CLOSED_CARDS = (
    "GYRE-zco", "VORTEX-zco", "VORTEX_VEC-zco",
    "VORTEX_SMT-zps", "VORTEX_SMT_VEC-zps",
    "VORTEX_SMT1_VEC-zps", "VORTEX_SMT2_VEC-zps",
    "VORTEX_SMT3_VEC-zps", "VORTEX_SMT4_VEC-zps", "VORTEX_SMT5_VEC-zps",
)
# SMT-5 reads the three inputs NEMO dumped in the admitted SMT-5 record.
SMT5_DECK_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                      "smtrungs_rounds/round2/oracle_vortex_smt5/kt1_10")
STATE_FIELDS = ("u", "v", "T", "S", "eta")


def state_digest(state) -> str:
    handle = hashlib.sha256()
    for name in STATE_FIELDS:
        data = np.ascontiguousarray(np.asarray(getattr(state, name).data,
                                               dtype=np.float64))
        if not np.all(np.isfinite(data)):
            raise SystemExit(f"non-finite {name}")
        handle.update(name.encode())
        handle.update(repr(data.shape).encode())
        handle.update(data.tobytes())
    return handle.hexdigest()


def run_case(case: str, steps: int) -> str:
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    if not jax.config.jax_enable_x64:
        raise SystemExit("JAX x64 is disabled")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    smt5 = case == "VORTEX_SMT5_VEC-zps"
    card = build_nemo_testcase_card(
        case, **({"deck_root": SMT5_DECK_ROOT} if smt5 else {}))
    model = LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config)
    state = card.recipe.initial_state
    for n in range(steps):
        state = model.step(state, dt=card.dt_s,
                           **({"t_seconds": n * card.dt_s} if smt5 else {}))
    return state_digest(state)


ORCA2_DECK_ROOT = Path("/data/abyssal/dbalwada/nemo-testcases-l4/inputs/"
                       "ORCA2_ICE_v5.0.0")


def orca2_bbl_digest() -> str:
    """ORCA2 cannot step without its forcing pipeline; digest instead the
    diffusive-BBL chain the model runs (geometry, gate, bottom-cell trend) on
    the ORCA2 card's own geometry, initial T/S and EOS."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from legoesm.ocean.physics.bbl_adv import (
        apply_bbl_diffusive_tendency,
        nemo_bbl_diffusive_coefficients,
        nemo_bbl_diffusive_geometry,
    )

    if not jax.config.jax_enable_x64:
        raise SystemExit("JAX x64 is disabled")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_nemo_testcase_card("ORCA2-zps", deck_root=ORCA2_DECK_ROOT)
    cfg, zc, st = card.recipe.model_config, card.recipe.z_coord, card.recipe.initial_state
    if cfg.bbl_diffusive_option != 1:
        raise SystemExit("ORCA2 card no longer selects the diffusive BBL")
    raw = zc.nemo_een_barotropic
    geom = nemo_bbl_diffusive_geometry(
        zc.h_partial, st.land_mask.data, zc.nemo_gdept_0, zc.nemo_bbl_e3u_0,
        zc.nemo_bbl_e3v_0, raw.e1u, raw.e2u, raw.e1v, raw.e2v, raw.umask,
        raw.vmask, aht_m2_s=cfg.bbl_aht_m2_s, grid=card.recipe.grid)
    ahu, ahv = nemo_bbl_diffusive_coefficients(
        st.T.data, st.S.data, geom, bottom_depth_m=geom.dep_bot_ref,
        rho_0=cfg.rho_0, grid=card.recipe.grid, eos_form=cfg.eos)
    zero = np.zeros(np.shape(st.T.data))
    dT, dS = apply_bbl_diffusive_tendency(
        zero, zero, st.T.data, st.S.data, zc.h_partial, card.recipe.grid.area_T,
        geom, ahu, ahv, grid=card.recipe.grid)
    handle = hashlib.sha256()
    for name, arr in (("ahu", ahu), ("ahv", ahv), ("dT", dT), ("dS", dS)):
        data = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        handle.update(name.encode() + repr(data.shape).encode() + data.tobytes())
    print("ORCA2 BBL open faces", int(np.count_nonzero(np.asarray(ahu))),
          int(np.count_nonzero(np.asarray(ahv))), flush=True)
    return handle.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CLOSED_CARDS)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--orca2-bbl", action="store_true",
                        help="also digest the ORCA2 diffusive-BBL chain")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path,
                        help="baseline json; exit 1 on any differing digest")
    args = parser.parse_args(argv)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True,
                                check=True).stdout.strip())
    result = {"git_sha": sha, "worktree_dirty": dirty, "steps": args.steps,
              "digests": {}}
    for case in args.case or CLOSED_CARDS:
        result["digests"][case] = run_case(case, args.steps)
        print(case, result["digests"][case], flush=True)
    if args.orca2_bbl:
        result["digests"]["ORCA2-zps:bbl_chain"] = orca2_bbl_digest()
        print("ORCA2-zps:bbl_chain", result["digests"]["ORCA2-zps:bbl_chain"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.compare is not None:
        base = json.loads(args.compare.read_text())["digests"]
        bad = [c for c, d in result["digests"].items() if base.get(c) != d]
        print("CLOSED_CARDS_" + ("DIFFER " + ",".join(bad) if bad
                                 else "BIT_IDENTICAL"))
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
