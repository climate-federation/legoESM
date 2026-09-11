#!/usr/bin/env python
"""THE IMPLICIT TRACER SOLVE, given NEMO's own matrix -- kt=1 and kt=2.

Until this record existed, legoESM's tracer tridiagonal could only be observed
through a SOLVED column, which conflates the ASSEMBLY with the ORDERED SWEEP:
a residual there cannot say which of the two owns it.  The record
(``nemo_dino_zdf_matrix/run.sh``) dumps NEMO's own ``zwi``/``zwd``/``zws``, so
each statement is scored on its own.

THE ORACLE'S STATEMENTS (``cfgs/DINO/BLD/ppsrc/nemo/trazdf.f90``, the branches
this card's namelist selects -- ``ln_zad_Aimp = F``
(``RUN_FROMREST_KT1/ocean.output:786``), ``ln_traldf_msc = T`` (``:890``),
``ln_zdfmfc`` absent)::

    231  DO jk = 1, jpkm1
    232     zwi(ji,jk) = - p2dt * zwt(ji,jk  ) / (e3w_3d(ji,jj,jk  )*(1+r3t(ji,jj,Kmm)))
    233     zws(ji,jk) = - p2dt * zwt(ji,jk+1) / (e3w_3d(ji,jj,jk+1)*(1+r3t(ji,jj,Kmm)))
    234     zwd(ji,jk) = (e3t_3d(ji,jj,jk)*(1+r3t(ji,jj,Kaa)*tmask(ji,jj,jk)))
    234                  - ( zwi(ji,jk) + zws(ji,jk) )
    270  zwt(ji,1) = zwd(ji,1)
    273  zwt(ji,jk) = zwd(ji,jk) - zwi(ji,jk) * zws(ji,jk-1) / zwt(ji,jk-1)

Note ``:234``: ``tmask`` multiplies ``r3t``, NOT the whole product, so a DRY
cell's diagonal is its REFERENCE thickness ``e3t_3d`` -- not 1, and not 0.
And note the three different time levels in three adjacent statements
(``Kmm`` on the face divisors, ``Kaa`` on the diagonal's thickness, ``Kbb``
and ``Kmm`` on the right-hand side at ``:285-290``): Rule 1d lives here.

WHAT THIS GATE CAN SEE, AND WHAT IT CANNOT (Rule 2, stated before the rows):

* The dump sits inside the ``jj`` i-k-slice loop with ``STATUS='REPLACE'`` on
  one filename per rank, so each rank's file holds its LAST ``jj`` slice --
  SIXTEEN rows of 199, 8.0% of the domain, and the gate says so on every row
  rather than implying a global comparison.  The rows ARE attributable: the
  index is derived from ``layout.dat`` (``njmpp``, ``jpj``) and ``nn_hls``,
  and row C1 is what proves the derivation, because a wrong row would not
  reproduce NEMO's own diagonal from NEMO's own off-diagonals.
* ``zwt`` itself is NOT dumped, so the sweep is scored against a literal
  transcription of ``:270-273`` fed NEMO's three dumped diagonals -- a
  NEMO-from-NEMO reference, not a second model.
* The RHS (``:285-290``) and the MOMENTUM diagonals are NOT IN THE RECORD.
  The acquisition script says so in its own header and so does this one; they
  are UNMEASURED here, not waived.

Usage::

    CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\
        python scripts/validate/ocean_fidelity/dino_1226/\\
            zdf_matrix_assembly_gate.py [--kt 1|2] [--plant] [--plant-lego]
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

MATRIX = "/data/abyssal/dbalwada/dino_fromrest_y1/nemo_zdf_matrix"
KT1 = "/data/abyssal/dbalwada/dino_fromrest_y1/nemo_kt1_rankdump"
PLANT_REL = 1.0e-9

#: NEMO-from-NEMO, one pointwise statement, no intrinsic whose order numpy
#: cannot reproduce: expected EXACTLY zero.
CALIB_MAX = 1.0e-18
CALIB_REL = 1.0e-15


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _table_cls():
    return _load("_spg_gate2",
                 os.path.join(_HERE, "spg_kt1_frozen_forcing_hpg.py")).Table


def rank_tiles(kt1_dir):
    """``rank -> (i0, ncore, j_last)``, 0-based haloless, from the mesh tiles.

    NEMO's own ``DOMAIN_position_first/last`` on each ``mesh_mask`` tile is
    already the HALOLESS core, so the strip's row is read off the record
    rather than reconstructed from ``layout.dat``'s haloed ``njmpp``/``jpj``
    (an earlier version did that and silently parsed the WRONG table, putting
    a row at global 200 on a 199-row domain).  ``trazdf.f90:231`` runs
    ``jj = ntsj .. ntej`` and the dump's ``STATUS='REPLACE'`` keeps the last,
    so the strip is ``ntej`` = the tile's LAST CORE ROW.
    """
    import netCDF4 as nc
    out = {}
    for path in sorted(glob.glob(os.path.join(kt1_dir, "mesh_mask_*.nc"))):
        rank = int(os.path.basename(path).split("_")[-1].split(".")[0])
        d = nc.Dataset(path)
        x0, y0 = (int(v) for v in d.DOMAIN_position_first)
        x1, y1 = (int(v) for v in d.DOMAIN_position_last)
        out[rank] = dict(i0=x0 - 1, ncore=x1 - x0 + 1, j=y1 - 1)
        d.close()
    if not out:
        raise SystemExit(f"no mesh_mask tiles under {kt1_dir}")
    return out


def read_strip(path, jpk, jpi):
    a = np.fromfile(path, dtype="<f8")
    if a.size != jpk * jpi:
        raise SystemExit(
            f"{os.path.basename(path)} holds {a.size} doubles, expected "
            f"{jpk}*{jpi}={jpk * jpi}; the dump's geometry is not what this "
            "gate assumes and nothing below it would mean anything")
    return a.reshape(jpk, jpi)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-dir", default=MATRIX)
    ap.add_argument("--kt1-dir", default=KT1)
    ap.add_argument("--kt", type=int, default=1, choices=(1, 2))
    ap.add_argument("--plant", action="store_true")
    ap.add_argument("--plant-lego", action="store_true")
    args = ap.parse_args()

    from rebuild_nemo_restart import rebuild

    md = args.matrix_dir
    print(f"RECORD: {md}   kt={args.kt}")
    tiles = rank_tiles(args.kt1_dir)
    mesh = rebuild(os.path.join(args.kt1_dir, "mesh_mask_*.nc"),
                   ["e3t_0", "e3w_0", "tmask"])
    e3t0, e3w0, tmask = mesh["e3t_0"], mesh["e3w_0"], mesh["tmask"]
    nz, ny, nx = e3t0.shape
    ht0 = (e3t0 * tmask).sum(axis=0)
    # ``Kaa`` is the AFTER level of the step being solved, and the restart
    # written at the END of that step carries it as ``sshn`` (the swap at
    # stpmlf.f90:577-579 precedes rst_write at :590).  For kt it therefore
    # comes from THIS step's own restart.
    rst_glob = os.path.join(md, "DINO_%08d_restart_*.nc" % args.kt)
    if not glob.glob(rst_glob):
        if args.kt == 1:
            rst_glob = os.path.join(args.kt1_dir,
                                    "DINO_00000001_restart_*.nc")
        if not glob.glob(rst_glob):
            raise SystemExit(f"no kt={args.kt} restart matching {rst_glob}")
    rst = rebuild(rst_glob, ["sshn", "sshb"])
    ssh_aa = rst["sshn"]
    r3t_aa = np.where(ht0 > 0.0, ssh_aa / np.where(ht0 > 0.0, ht0, 1.0), 0.0)
    print(f"  r3t(Kaa) from {os.path.basename(glob.glob(rst_glob)[0])} "
          f"'sshn' (stpmlf.f90:577-579 then :590); ssh range "
          f"{ssh_aa.min():+.6f} .. {ssh_aa.max():+.6f} m")

    # ---- which global row each rank's strip is --------------------------
    strips = {}
    for rank, t in sorted(tiles.items()):
        f = os.path.join(md, "trazdf_zwd_kt%08d_rank%02d.bin"
                         % (args.kt, rank))
        if not os.path.exists(f):
            raise SystemExit(f"missing {f}")
        jpi = os.path.getsize(f) // (8 * nz)
        if jpi * nz * 8 != os.path.getsize(f):
            raise SystemExit(
                f"{os.path.basename(f)} is {os.path.getsize(f)} bytes, not a "
                f"whole multiple of {nz} levels x 8 bytes")
        nn_hls = (jpi - t["ncore"]) // 2
        if jpi != t["ncore"] + 2 * nn_hls:
            raise SystemExit(
                f"rank {rank}: dump width {jpi} is not the core {t['ncore']} "
                "plus a symmetric halo; the strip cannot be placed")
        strips[rank] = dict(jpi=jpi, nn_hls=nn_hls, **t)
    js = sorted({v["j"] for v in strips.values()})
    if min(js) < 0 or max(js) >= ny:
        raise SystemExit(f"strip rows {js} fall outside 0..{ny - 1}")
    print(f"  {len(strips)} rank strips -> {len(js)} distinct global rows "
          f"{js} of {ny}  ({100.0 * len(js) / ny:.1f}% of the domain; the "
          "dump is one jj slice per rank by construction, "
          "trazdf.f90 STATUS='REPLACE')")

    def gather(name):
        """(nz, ny, nx) with NaN everywhere the record does not cover."""
        out = np.full((nz, ny, nx), np.nan)
        for rank, st in strips.items():
            a = read_strip(
                os.path.join(md, "trazdf_%s_kt%08d_rank%02d.bin"
                             % (name, args.kt, rank)), nz, st["jpi"])
            h, n = st["nn_hls"], st["ncore"]
            out[:, st["j"], st["i0"]:st["i0"] + n] = a[:, h:h + n]
        return out

    zwi, zwd, zws = (gather(n) for n in ("zwi", "zwd", "zws"))
    cov = ~np.isnan(zwd)
    # NEMO's assembly loop is jk = 1..jpkm1; level jpk is untouched memory.
    cov[nz - 1:, :, :] = False
    print(f"  covered cells (levels 1..jpkm1): {int(cov.sum())} of "
          f"{int((tmask > 0.5).sum())} wet")
    wet_cov = cov & (tmask > 0.5)
    dry_cov = cov & (tmask <= 0.5)
    print(f"    of which wet {int(wet_cov.sum())}, dry {int(dry_cov.sum())}")

    Table = _table_cls()
    tab = Table()
    ref_zwd = zwd
    if args.plant:
        ref_zwd = zwd.copy()
        j = sorted({v["j"] for v in strips.values()})[0]
        i = int(np.argwhere(cov[3, j])[0][0])
        ref_zwd[3, j, i] *= (1.0 + PLANT_REL)
        print(f"  PLANT: NEMO's own zwd at (3,{j},{i}) scaled by "
              f"1+{PLANT_REL:g}; C1 must be REFUSED")

    e3t_aa = e3t0 * (1.0 + r3t_aa[None, :, :] * tmask)
    print()
    print("1. CALIBRATION -- NEMO's zwd rebuilt from NEMO's own zwi/zws")
    tab.calibrate("C1 trazdf.f90:234 from the record",
                  e3t_aa - (zwi + zws), ref_zwd, cov)

    # ---- legoESM's own assembly statement -------------------------------
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())
    import jax.numpy as jnp
    from legoesm.ocean.physics.vertical_mixing import implicit_solver as iv

    def L(a):
        return np.moveaxis(np.asarray(a, dtype=np.float64), 0, -1)

    lower = L(np.nan_to_num(zwi, nan=0.0))
    upper = L(np.nan_to_num(zws, nan=0.0))
    e3t_l = L(e3t_aa)
    wet_l = L((tmask > 0.5).astype(np.float64))
    if args.plant_lego:
        j = sorted({v["j"] for v in strips.values()})[0]
        i = int(np.argwhere(cov[3, j])[0][0])
        e3t_l = e3t_l.copy()
        e3t_l[j, i, 3] *= (1.0 + PLANT_REL)
        print(f"  PLANT-LEGO: legoESM's e3t_after at ({j},{i},3) scaled by "
              f"1+{PLANT_REL:g}; the assembly rows must MOVE")

    # The statement, exactly as implicit_vertical_diffusion_nemo_tracer_pair
    # writes it (implicit_solver.py:506-525): the ordered sum, then the
    # subtraction, then the DRY substitution.
    lo = jnp.asarray(lower)
    up = jnp.asarray(upper)
    coeff_sum = jnp.asarray(np.asarray(lo) + np.asarray(up))
    diag_lego = np.asarray(jnp.asarray(e3t_l) - coeff_sum)
    diag_dry1 = np.where(wet_l > 0.0, diag_lego, 1.0)

    tab.score("A1 legoESM diagonal, WET cells",
              np.moveaxis(diag_dry1, -1, 0), ref_zwd, wet_cov)
    tab.score("A2 legoESM diagonal, DRY cells",
              np.moveaxis(diag_dry1, -1, 0), ref_zwd, dry_cov)
    tab.score("A2b same without the 1.0 substitution",
              np.moveaxis(diag_lego, -1, 0), ref_zwd, dry_cov)

    # ---- the sign of zero in the two boundary slots ---------------------
    print()
    print("3. THE TWO BOUNDARY SLOTS -- value AND sign of zero")
    for nm, arr, k in (("zwi(:,1)   trazdf.f90:232 with zwt(:,1)=0", zwi, 0),
                       ("zws(:,jpkm1) trazdf.f90:233", zws, nz - 2)):
        a = arr[k][cov[k]]
        neg = int(np.signbit(a).sum())
        print(f"  NEMO {nm:44s} nonzero {int((a != 0).sum())} of {a.size}, "
              f"negative-signed {neg}")
    lego_top = np.zeros_like(zwi[0][cov[0]])
    print(f"  legoESM's zwi(:,1) is jnp.zeros_like -> negative-signed "
          f"{int(np.signbit(lego_top).sum())} of {lego_top.size}")
    tab.note("the two boundary slots are read by NO recurrence "
             "(trazdf.f90:273 starts at jk=2 and :298 stops at jk=1), so a "
             "sign-of-zero difference there changes no answer -- it is "
             "reported because a gate at the exact bar must say what it saw")

    # ---- does the dry diagonal change an ANSWER?  Measured, not argued --
    print()
    print("4. THE DRY DIAGONAL AND THE SIGN OF ZERO -- do they change an "
          "answer?")
    rng = np.random.default_rng(1728)
    rhs = jnp.asarray(rng.standard_normal(diag_lego.shape) * wet_l)
    solve = iv.nemo_ordered_tridiagonal_solve
    # NEMO's own dry diagonal is its REFERENCE thickness; legoESM's is 1.0.
    diag_nemo_dry = np.where(wet_l > 0.0, diag_lego, e3t_l)
    x_1 = np.asarray(solve(jnp.asarray(lower), jnp.asarray(diag_dry1),
                           jnp.asarray(upper), rhs)) * wet_l
    x_n = np.asarray(solve(jnp.asarray(lower), jnp.asarray(diag_nemo_dry),
                           jnp.asarray(upper), rhs)) * wet_l
    d = np.abs(x_1 - x_n)
    print(f"  solved column with legoESM's 1.0 vs NEMO's e3t on DRY cells: "
          f"max|d| {d.max():.4e} on {int((x_1 != x_n).sum())} of {x_1.size} "
          "values")
    # And the sign of zero on the two slots NEMO writes as -0.0.
    lower_neg = lower.copy()
    lower_neg[..., 0] = -0.0
    upper_neg = upper.copy()
    upper_neg[..., -1] = -0.0
    x_z = np.asarray(solve(jnp.asarray(lower_neg), jnp.asarray(diag_dry1),
                           jnp.asarray(upper_neg), rhs)) * wet_l
    print(f"  solved column with NEMO's NEGATIVE zeros in the two boundary "
          f"slots: max|d| {np.abs(x_1 - x_z).max():.4e} on "
          f"{int((x_1 != x_z).sum())} of {x_1.size} values")
    if d.max() == 0.0 and np.abs(x_1 - x_z).max() == 0.0:
        tab.note("MEASURED: neither the dry diagonal nor the sign of zero "
                 "changes a solved value -- trazdf.f90:273 starts at jk=2 "
                 "and :295/:298 multiply by tmask, so the dry slots are "
                 "written and never read.  They are REGISTERED as "
                 "representation differences, not fixed: making legoESM's "
                 "dry diagonal NEMO's own means changing the THICKNESS the "
                 "caller hands the solver (h_partial is exactly 0 below the "
                 "seafloor, which is why the 1.0 substitution exists), and "
                 "that is a separate change with its own measurement")
    else:
        tab.note("the dry diagonal or the sign of zero DOES change a solved "
                 "value; it is a live defect, not a representation "
                 "difference")
        tab.fail = True

    tab.note("THE RHS (trazdf.f90:285-290) and the MOMENTUM diagonals "
             "(dynzdf) are NOT IN THIS RECORD -- UNMEASURED, not waived")
    tab.render()
    if tab.rows[0]["status"] != "CALIBRATED":
        print("\nthe NEMO-from-NEMO calibration is not CALIBRATED, so no row "
              "below it is underwritten and none may be quoted")
        return 1
    scored = [r for r in tab.rows if r["status"] in ("AT BAR", "DEBT")]
    first = next((r for r in scored if r["status"] == "DEBT"), None)
    print(f"\nRESULT: {sum(1 for r in scored if r['status'] == 'AT BAR')} of "
          f"{len(scored)} scored rows AT BAR against a floor of "
          f"{tab.floor:.4e}")
    if first is not None:
        print(f"FIRST NON-BIT STATEMENT: {first['name']} at "
              f"{first['rel']:.4e} relative on {first['cells']} of "
              f"{first['n']} cells")
    return 1 if tab.fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
