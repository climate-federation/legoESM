#!/usr/bin/env python3
"""Split the barotropic loop-entry forcing into its two compiled statements.

Round 215's substep walk names the loop-entry depth-averaged slow forcing
``zu_frc``/``zv_frc`` as the owner of the seamount cards' kt=2 sea-surface
height error, and round 213's held transcription of NEMO's depth-average
statement removes only part of it.  The operand NEMO records is the forcing
AFTER the barotropic Coriolis trend has been subtracted from it, so two
compiled statements can carry the remainder:

    dynspg_ts.f90:275   zu_frc(:,:) = Ue_rhs(:,:)          the depth average
    dynspg_ts.f90:292   zu_frc = zu_frc - zu_trd*ssumask   the Coriolis sub.

This probe separates them WITHOUT a new acquisition, using only records
already admitted:

* the per-term pre-stage record's LAST cumulative dump is NEMO's completed
  ``uu(:,:,:,Krhs)``, the array ``stp2d.f90:178`` averages;
* the card carries NEMO's own ``e3u_0``/``e3v_0``, ``umask``/``vmask`` and
  ``hu_0``/``hv_0`` (round 212 proved them 0 ULP against ``mesh_mask.nc``),
  so ``Ue_rhs`` can be rebuilt from NEMO's own operands;
* the substep record's ``i000_zu_frc`` is the finished operand.

``max |Ue_rhs(rebuilt) - zu_frc(recorded)|`` is therefore the SIZE of the
Coriolis subtraction at this step, and ``max |Ue_rhs(rebuilt) -
Ue_rhs(NEMO's own, unavailable)|`` is not claimed: the rebuild uses NEMO's
operands and NEMO's right-hand side, so a residual against the recorded
``zu_frc`` is the subtraction, not a transcription error.

The probe REFUSES rather than reports if the right-hand side it was handed
is not the completed one (the control: the per-term record's own ordering).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nemo_testcase_l1_vortex_prestage_terms import (  # noqa: E402
    BOUNDARIES, read_terms,
)
from nemo_testcase_l1_vortex_round196_spgts_walk import (  # noqa: E402
    read_spgts,
)
from nemo_testcase_phase3_trajectory_gate import GateError, require  # noqa: E402

HALO = 2


def _interior(plane: np.ndarray, nlev: int) -> np.ndarray:
    """NEMO's interior of a halo-padded (i, j, k) record array, as (j, i, k).

    The same slice ``nemo_testcase_l1_vortex_prestage_terms`` uses on this
    record family, written once here rather than inferred.
    """
    return plane[HALO:-HALO, HALO:-HALO].transpose(1, 0, 2)[..., :nlev]


def run(terms_root: Path, spgts_root: Path, *, case: str,
        allow_dirty: bool = False, write_operand: Path | None = None) -> dict:
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    require(get_policy() == PrecisionPolicy.fp64(transcendentals="libm"),
            "precision policy is not fp64")

    terms = read_terms(terms_root)
    last = BOUNDARIES[-1]
    require(last == "zad", "the last pre-stage boundary is no longer zad")
    _meta, groups = read_spgts(spgts_root, 1)
    zu_frc = groups["i000_zu_frc"]
    zv_frc = groups["i000_zv_frc"]
    un_e = groups["i000_un_e"]
    vn_e = groups["i000_vn_e"]

    card = build_nemo_testcase_card(case)
    zc = card.recipe.z_coord
    raw = getattr(zc, "nemo_een_barotropic", None)
    require(raw is not None,
            "this probe needs the card's NEMO-built mesh bundle")
    nlev = int(zc.n_levels)
    uu = _interior(terms[last]["fields"]["uu_rhs"], nlev)
    vv = _interior(terms[last]["fields"]["vv_rhs"], nlev)
    e3u_0 = np.asarray(raw.e3u_0, dtype=np.float64)[..., :nlev]
    e3v_0 = np.asarray(raw.e3v_0, dtype=np.float64)[..., :nlev]
    umask = np.asarray(raw.umask, dtype=np.float64)[..., :nlev]
    vmask = np.asarray(raw.vmask, dtype=np.float64)[..., :nlev]
    hu_0 = np.asarray(raw.hu_0, dtype=np.float64)
    hv_0 = np.asarray(raw.hv_0, dtype=np.float64)
    require(uu.shape == e3u_0.shape,
            f"the record's RHS is {uu.shape}, the card's faces {e3u_0.shape}")

    ssu = (hu_0 > 0.0).astype(np.float64)
    ssv = (hv_0 > 0.0).astype(np.float64)
    r1_hu_0 = ssu / (hu_0 + 1.0 - ssu)          # domain.F90
    r1_hv_0 = ssv / (hv_0 + 1.0 - ssv)
    ue_rhs = np.sum(e3u_0 * uu * umask, axis=-1) * r1_hu_0   # stp2d.f90:178
    ve_rhs = np.sum(e3v_0 * vv * vmask, axis=-1) * r1_hv_0   # stp2d.f90:179

    du = np.abs(ue_rhs - zu_frc)
    dv = np.abs(ve_rhs - zv_frc)
    report = {
        "case": case, "kt": 1, "git_sha": sha,
        "terms_root": str(terms_root), "spgts_root": str(spgts_root),
        "rhs_peak_u": float(np.max(np.abs(uu))),
        "rhs_peak_v": float(np.max(np.abs(vv))),
        "ue_rhs_peak": float(np.max(np.abs(ue_rhs))),
        "zu_frc_peak": float(np.max(np.abs(zu_frc))),
        "entry_barotropic_velocity_peak_u": float(np.max(np.abs(un_e))),
        "entry_barotropic_velocity_peak_v": float(np.max(np.abs(vn_e))),
        "coriolis_subtraction_max_u": float(np.max(du)),
        "coriolis_subtraction_max_v": float(np.max(dv)),
        "coriolis_subtraction_cells_u": int(np.count_nonzero(du)),
        "coriolis_subtraction_cells_v": int(np.count_nonzero(dv)),
    }
    if write_operand is not None:
        # NEMO's OWN depth average, rebuilt from NEMO's own right-hand side
        # and NEMO's own mesh operands.  The substep walk substitutes it at
        # the boundary legoESM forms the same quantity, which is what turns
        # the split below into a measurement instead of an inference.
        np.savez(write_operand, ue_rhs=ue_rhs, ve_rhs=ve_rhs)
        report["operand_written"] = str(write_operand)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms-dir", type=Path, required=True)
    parser.add_argument("--spgts-dir", type=Path, required=True)
    parser.add_argument("--case", default="VORTEX_SMT_VEC-zps")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-operand", type=Path,
                        help="save NEMO's own rebuilt depth average as an "
                             "npz the substep walk can substitute")
    args = parser.parse_args(argv)
    try:
        report = run(args.terms_dir, args.spgts_dir, case=args.case,
                     allow_dirty=args.allow_dirty,
                     write_operand=args.write_operand)
    except GateError as error:
        print(f"REFUSE: {error}", file=sys.stderr)
        return 2
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True)
                               + "\n")
    for key, value in sorted(report.items()):
        print(f"{key:<36} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
