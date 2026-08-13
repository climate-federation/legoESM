"""Is the eady_uniform/mpas_channel blow-up grid-scale noise or a resolved mode?

WHY. Centring the barotropic averaging window on t+dt (a correctness fix:
external gravity waves ran ~1.9x too slow) breaks exactly one case of 65 --
eady_uniform on mpas_channel, where the free surface reaches 5192 m against a
100 m threshold. Two explanations were proposed and BOTH are refuted: the new
window damps MORE at the coupling period, and the barotropic CFL is 0.033, a
30x margin (more substeps make it worse, not better).

What is left is the SPATIAL structure of eta just before onset, which the test
matrix does not dump at that step. This script gets it. The discriminator, on
an unstructured mesh where a Fourier spectrum is awkward:

  CHEQUERBOARD INDEX  mean over edges of -sign(eta_c1 * eta_c2), i.e. how often
      neighbouring cells carry OPPOSITE signs. A 2*dx null-space mode (the
      known hexagonal C-grid divergence chequerboard) drives this toward +1.
      A mode resolved over many cells leaves it near 0 or negative.

  CONCENTRATION      fraction of wet cells carrying 10% or more of max|eta|.
      A single hot cell or a few-cell cluster is numerical; a resolved
      baroclinic mode occupies a large fraction of the channel.

Together these separate "TRiSK/Perot noise on distorted cells" from "the Eady
mode growing faster under a correctly-phased free surface". Neither number is
a verdict on its own -- they are reported with the mesh's own reference values
so a reader can see the scale.

Run AFTER building the mesh the case uses; writes a JSON record per sampled
step so the onset can be walked rather than inferred from one frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def chequerboard_index(eta, cells_on_edge, wet=None):
    """Mean over interior edges of -sign(eta_a * eta_b), in [-1, 1].

    +1 means every neighbouring pair has opposite signs (a 2*dx mode); 0 means
    signs are uncorrelated across edges; negative means neighbours agree, i.e.
    a smooth large-scale field.

    Edges touching a dry cell are dropped: a land neighbour holds a constant
    and would bias the index toward whatever that constant's sign is.
    """
    coe = np.asarray(cells_on_edge)
    if coe.shape[0] != 2:                       # (nEdges, 2) -> (2, nEdges)
        coe = coe.T
    a, b = coe[0], coe[1]
    ok = (a >= 0) & (b >= 0)
    if wet is not None:
        wet = np.asarray(wet, dtype=bool)
        ok &= wet[a] & wet[b]
    if not ok.any():
        return float("nan"), 0
    ea, eb = np.asarray(eta)[a[ok]], np.asarray(eta)[b[ok]]
    good = np.isfinite(ea) & np.isfinite(eb)
    if not good.any():
        return float("nan"), 0
    return float(np.mean(-np.sign(ea[good] * eb[good]))), int(good.sum())


def concentration(eta, wet=None, frac=0.1):
    """Fraction of wet cells at or above ``frac`` of max|eta|."""
    e = np.abs(np.asarray(eta, dtype=np.float64))
    if wet is not None:
        e = e[np.asarray(wet, dtype=bool)]
    e = e[np.isfinite(e)]
    if e.size == 0 or e.max() <= 0:
        return float("nan"), 0
    return float(np.mean(e >= frac * e.max())), int(e.size)


def _controls(cells_on_edge, n_cells, wet=None, seed=0):
    """The two indices on fields whose answer is KNOWN.

    Without these the numbers above are unanchored: a chequerboard index of
    0.3 means nothing until you have seen what an actual chequerboard and an
    actual smooth field score on THIS mesh.
    """
    rng = np.random.default_rng(seed)
    out = {}
    alt = rng.standard_normal(n_cells)
    alt = np.abs(alt) * np.where(np.arange(n_cells) % 2 == 0, 1.0, -1.0)
    out["synthetic_alternating"] = chequerboard_index(alt, cells_on_edge, wet)[0]
    out["synthetic_random"] = chequerboard_index(
        rng.standard_normal(n_cells), cells_on_edge, wet)[0]
    out["synthetic_constant_sign"] = chequerboard_index(
        np.abs(rng.standard_normal(n_cells)) + 1.0, cells_on_edge, wet)[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npz", type=Path, required=True,
                    help="eta snapshots on the NATIVE mesh, one row per "
                         "sampled step (key 'eta', shape (n_samples, nCells))")
    ap.add_argument("--mesh-npz", type=Path, required=True,
                    help="mesh arrays: cellsOnEdge, and optionally land_mask")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    z = np.load(a.npz)
    m = np.load(a.mesh_npz)
    eta = np.asarray(z["eta"], dtype=np.float64)
    if eta.ndim == 1:
        eta = eta[None, :]
    coe = m["cellsOnEdge"]
    wet = None
    if "land_mask" in m.files:
        lm = np.asarray(m["land_mask"], dtype=np.float64)
        wet = (lm[-1] if lm.ndim == 2 else lm) > 0.5
    steps = (np.asarray(z["steps"]).tolist() if "steps" in z.files
             else list(range(eta.shape[0])))

    rec = {"controls": _controls(coe, eta.shape[1], wet), "samples": []}
    for k, s in enumerate(steps[:eta.shape[0]]):
        cb, n_edges = chequerboard_index(eta[k], coe, wet)
        cc, n_cells = concentration(eta[k], wet)
        rec["samples"].append({
            "step": int(s),
            "max_abs_eta_m": float(np.nanmax(np.abs(eta[k]))),
            "chequerboard_index": cb, "edges_used": n_edges,
            "concentration_frac": cc, "cells_used": n_cells})
        print(f"  step {int(s):>7d}  max|eta| {np.nanmax(np.abs(eta[k])):10.3f} m"
              f"   chequerboard {cb:+.3f}   concentration {cc:.4f}")

    print("\n  CONTROLS on this mesh (what the index reads for known fields):")
    for k, v in rec["controls"].items():
        print(f"    {k:26s} {v:+.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2))
    print(f"\nCOMPLETED: {a.out}")


if __name__ == "__main__":
    main()
