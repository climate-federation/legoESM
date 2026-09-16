"""Global per-process budget table from ``budget_ledger_columns.npz`` (#1354).

The artifact holds per-column mean rates (ncol, 7, 2) over the LAST diagnostic
window: rows = LEDGER_PROCESSES, columns = (water [kg/m2/s], dry-enthalpy
energy [W/m2]).

REDUCTION.  The rows are PER-COLUMN, so the consumer picks the weighting, and
the wrong pick silently produces a number that cannot be compared to anything.
An unweighted cell mean is a global mean only on an equal-area mesh; the
production SCVT mesh has areaCell max/min = 1.471.  Worse, the quantity these
rows get differenced against -- the energy-budget tracker's dE/dt -- is
ALREADY area-weighted, so an unweighted row and that store tendency are
different global operators and their difference is not interpretable.  That
mismatch produced a wrong published attribution on #1354 (a "+33 W/m2 dynamics
row" whose own worst-case reweighting was +-43 W/m2), which is why this script
now REFUSES rather than prints when it cannot establish the reduction -- the
same posture ``mpas_energy_budget_leak.assert_interval_means`` takes when it is
handed snapshot fluxes instead of interval means.

Reference values printed alongside are the PRE-REGISTERED healthy expectations
from the run card (scripts/cluster/mpas_energy_1354/ledger_3d_ginsburg.sbatch),
so the deltas are readable in place.  Numbers only, no verdicts.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from legoesm import constants
from legoesm.diagnostics.process_ledger import reduce_ledger_global
from legoesm.parallel.geometry_consistency import content_hash48


def resolve_area_weights(d, areacell_path, allow_unweighted):
    """Return ``(weights_or_None, provenance_string)``; raise if undecidable.

    Order: an explicit ``--areacell`` file, then the ``area_cell`` array the
    driver ships inside the artifact, then the escape hatch.  A zero-length
    ``area_cell`` is how the first version of the writer recorded "this run
    had no area weights"; the writer now omits the key instead, but those
    artifacts exist and are treated as absent rather than as an empty mesh.

    An explicit override is CHECKED against the fingerprint the writer ships
    whenever there is one.  Length alone cannot see a permutation, and a
    permuted weight vector produces a quietly wrong global mean rather than
    an error -- both reviewers flagged this as the top defect in the first
    draft of this script.
    """
    stored_hash = float(d["area_hash48"]) if "area_hash48" in d else None
    if areacell_path is not None:
        raw = np.load(areacell_path)
        if not isinstance(raw, np.ndarray):
            raise SystemExit(
                f"--areacell {areacell_path} is a {type(raw).__name__}, not a "
                "plain array; pass a .npy holding areaCell, not an archive.")
        w = np.asarray(raw, dtype=np.float64).ravel()
        prov = f"area-weighted (weights from {areacell_path})"
        if stored_hash is not None:
            if content_hash48(w) != stored_hash:
                raise SystemExit(
                    f"--areacell {areacell_path} does NOT match the weights "
                    "this run used: the artifact ships a fingerprint and the "
                    "supplied array has a different one.  Same cell count is "
                    "not the same mesh, and the same mesh in a different "
                    "ORDER is not the same weights either.  Drop --areacell "
                    "to use the shipped weights.")
            prov += " (fingerprint matches the run)"
        else:
            prov += " (UNVERIFIED -- artifact ships no fingerprint)"
        return w, prov
    w = d["area_cell"] if "area_cell" in d else None
    if w is not None and np.asarray(w).size > 0:
        return (np.asarray(w, dtype=np.float64).ravel(),
                "area-weighted (weights from the artifact)")
    if allow_unweighted:
        return None, ("UNWEIGHTED cell mean -- NOT a global mean on a "
                      "non-equal-area mesh, and NOT comparable to the "
                      "area-weighted dE/dt (--allow-unweighted was passed)")
    raise SystemExit(
        "REFUSING to print: this artifact carries no 'area_cell' weights and "
        "no --areacell file was given, so the reduction is undecidable.  An "
        "unweighted mean is not a global mean on the SCVT mesh and is not "
        "comparable to the area-weighted energy-budget dE/dt.  Fix it one of "
        "these ways:\n"
        "  * re-run the model -- the driver now writes the weights, and the "
        "fingerprint that proves they are the ones dE/dt used;\n"
        "  * for an existing run, extract this mesh's areaCell to a .npy and "
        "pass --areacell <file>;\n"
        "  * pass --allow-unweighted if you want the unweighted table anyway "
        "and will NOT difference it against the tracker (exit code 3).")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact", type=pathlib.Path,
                   help="budget_ledger_columns.npz from the run")
    p.add_argument("--areacell", type=pathlib.Path, default=None,
                   help="areaCell weights [.npy] for artifacts written before "
                        "the driver shipped them; must match this run's mesh")
    p.add_argument("--allow-unweighted", action="store_true",
                   help="print the unweighted table anyway, stamped as such")
    p.add_argument("--hfss", type=float, default=20.5,
                   help="reference sensible heat flux [W/m2] (CMOR mean)")
    p.add_argument("--hfls", type=float, default=58.1,
                   help="reference latent heat flux [W/m2] (CMOR mean)")
    args = p.parse_args(argv)

    d = np.load(args.artifact, allow_pickle=True)
    rates = np.asarray(d["ledger_rates"], dtype=np.float64)   # (ncol, 7, 2)
    procs = [str(x) for x in d["processes"]]
    nstep = int(d["n_steps"]) if "n_steps" in d else -1
    day = float(d["day"]) if "day" in d else float("nan")

    if nstep == 0:
        raise SystemExit(
            "REFUSING to print: n_steps is 0, so every rate in this artifact "
            "is a zero divided into an empty interval, not a measurement.")
    if "cell_partitioned" in d and bool(d["cell_partitioned"]):
        raise SystemExit(
            "REFUSING to print: this artifact was written under MPAS cell "
            "partitioning, so its rows are ONE RANK's columns.  A per-rank "
            "table is not a global budget however it is weighted, and the "
            "energy tracker it would be compared against does not even run "
            "in that configuration.")
    if not np.isfinite(rates).all():
        raise SystemExit(
            f"REFUSING to print: {int((~np.isfinite(rates)).sum())} non-finite "
            "entries in the ledger.  A weighted mean over them is NaN for the "
            "whole row, and a nan-skipping mean would silently reduce over a "
            "different set of columns per row -- which is the same class of "
            "defect this script exists to prevent.")

    area, provenance = resolve_area_weights(
        d, args.areacell, args.allow_unweighted)
    if area is not None:
        if area.shape[0] != rates.shape[0]:
            raise SystemExit(
                f"area weights have {area.shape[0]} cells but the ledger has "
                f"{rates.shape[0]} columns -- wrong mesh for this artifact.")
        if not (np.isfinite(area).all() and (area > 0).all()):
            raise SystemExit(
                "area weights must be finite and positive; got "
                f"min={np.nanmin(area):.3e}.")
        provenance += (f"; sum={area.sum():.6e} m^2 over {area.size} cells")
    g_mean = np.asarray(reduce_ledger_global(rates, area))     # (7, 2)

    print(f"{args.artifact}")
    print(f"last diagnostic window ending day {day:g}, {nstep} steps, "
          f"{rates.shape[0]} columns")
    print(f"reduction: {provenance}")
    print()
    hdr = (f"{'process':>14s} {'water [mm/day]':>15s} {'L_v*water':>10s} "
           f"{'energy [W/m2]':>14s}")
    print(hdr)
    print("-" * len(hdr))
    for i, name in enumerate(procs):
        w, e = g_mean[i]
        print(f"{name:>14s} {w * 86400.0:15.4f} {constants.L_v * w:10.3f} "
              f"{e:14.3f}")
    tw, te = g_mean.sum(axis=0)
    print("-" * len(hdr))
    print(f"{'TOTAL':>14s} {tw * 86400.0:15.4f} {constants.L_v * tw:10.3f} "
          f"{te:14.3f}")
    print()
    print("pre-registered healthy references (run card):")
    print(f"  turbulence energy ~ +hfss = {args.hfss:+.1f}   "
          f"(~{2 * args.hfss:.0f} = Louis double-count hypothesis)")
    print(f"  turbulence L_v*water ~ +hfls = {args.hfls:+.1f}")
    print("  radiation energy ~ toa_net - sfc_net ~ -97")
    print("  convection/microphysics: energy ~ -L_v*water (heating paid by "
          "drying); excess = unpaid heating")
    print("  dynamics: the energy column is DRY ENTHALPY, so adiabatic "
          "enthalpy<->geopotential<->KE conversion lands here even for a "
          "perfectly conserving dycore.  A large row is NOT evidence of "
          "creation; localising an MSE leak per stage needs a total-energy "
          "column (#1354).  A row large relative to the net top-of-"
          "atmosphere flux is still worth a look -- filter dissipation lands "
          "here too -- it is just not by itself evidence of creation.)")
    print("  other_physics/clips: residual buckets, an unledgered creation "
          "lands THERE unnamed")
    # Exit 3, not 0, when the table is unweighted: a text stamp is invisible
    # to anything that greps the numbers, so the escape hatch has to be
    # visible to a caller as well as to a human (GLM).
    return 3 if area is None else 0


if __name__ == "__main__":
    raise SystemExit(main())
