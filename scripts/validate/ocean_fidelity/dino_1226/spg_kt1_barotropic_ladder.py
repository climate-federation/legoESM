#!/usr/bin/env python
"""LADDER: where inside NEMO's kt=1 barotropic solve does legoESM leave it?

The step-1 gate says sea level is the largest fraction of NEMO's own first
step (about 1%), and neither the surface-forcing arm nor ``mlf_baro_corr``
moved it.  Sea level after the step IS the barotropic loop's own output:
``ssh_nxt`` writes ``ssh(Kaa)`` at stpmlf.f90:248 and ``dyn_spg_ts``
OVERWRITES it at dynspg_ts.f90:1006 with the boxcar average, and the Euler
step skips ``ssh_atf`` (sshwzv.f90:443), so the restart's now-level ``sshn``
is exactly ``pssh(:,:,Kaa)``.

So this walks the loop from its inputs to its outputs and scores each rung
that a RACE-FREE oracle exists for.

WHY "RACE-FREE" IS THE WHOLE PROBLEM HERE.  The record's per-substep streams
cannot be used, and that is a property of the record, not of the comparison.
``ll_spg_dump`` (dynspg_ts.F90:206) and the substep-trajectory guard (:926)
test only ``kt == nit000``; neither tests ``narea``.  This record ran on 16
MPI ranks in one directory, so all 16 OPEN the same filename with
STATUS='REPLACE' and write concurrently.  Section 1 proves the damage three
ways rather than assuming it.  The rungs those streams would have scored are
printed UNMEASURED with the record that would close them -- never silently
dropped (Rule 1).

The rungs that ARE scorable come from the restart tiles, which are written
through XIOS with DOMAIN attributes and are not raced:

  R0  the layer thicknesses  every rung below is a depth-weighted mean or a
      the state sits in        face-column depth, so a wrong thickness reaches
                               all of them (Rule 2: the state can be right
                               while the boxes holding it are the wrong size).
                               NEMO's DINO carries TWO ladders -- the 1-D
                               reference e3t_1d and the 3-D e3t_0 that
                               zgr_sco_mi96 returns for its flat 4000 m column
                               -- and they part company below level 25.
                               legoESM runs on the first, NEMO on the second.
  R2  loop-entry seed        pssh/puu_b/pvv_b at Kbb (dynspg_ts.f90:494-503).
                             From rest these are identically zero, and NEMO's
                             own ub/vb confirm it.
  R4a barotropic velocity    puu_b(:,:,Kaa) -- the boxcar average of the
      average                substep velocities (dynspg_ts.f90:1005), sampled
                             into the restart as ``ubtacc_aa`` = uu_b(Naa) *
                             r1_Dt right after the dyn_spg call
                             (stpmlf.F90 trddump_acc_baro, trddump.F90:161).
  R4b barotropic sea level   pssh(:,:,Kaa) == restart ``sshn`` (above).
      average

Both R4 rungs are the loop's EXIT, so a break there says "somewhere in the
45 substeps" and no more.  Localising it needs the per-substep record, which
``nemo_dino_kt1_rankdump/run.sh`` acquires as a makenemo CONFIG COPY that
cannot write inside the read-only oracle.  WHEN THAT RECORD IS THE
``--run-dir``, section 4 runs and scores R1 and R3 substep by substep; the
audit decides which of the two paths this gate takes, and neither is
hard-coded.

REMOVED 2026-09-10, and it is worth saying why rather than letting it vanish
from the diff: this script used to EMIT that re-run itself, and the script it
emitted edited ``cfgs/DINO/MY_SRC/dynspg_ts.F90`` IN PLACE behind an exit trap
and then ran ``makenemo -r DINO -n DINO``, which overwrites the oracle's own
``cfgs/DINO/BLD`` and ``bin/nemo.exe`` -- and no trap restores those.  A
reviewer found it still reachable after the config-copy acquisition landed.
The safe acquisition replaces it; this script no longer writes anything under
the oracle.

Rule 10: the legoESM side is the production solver's own arrays, captured
through the private ``_nemo_substep_trace_test_hook`` (no numerical selector
is introduced) and through a wrapper that records the solver's arguments.
The loop's time-averaged velocity is not returned by the solver, so it is
rebuilt from the traced per-substep velocities; section 3 CALIBRATES that
rebuild by applying the identical method to sea level, whose average the
solver DOES return, and requiring bit equality.

Usage (GPU only -- this card's XLA CPU compile crashes on this machine):

    CUDA_VISIBLE_DEVICES=<uuid> JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \\
    python scripts/validate/ocean_fidelity/dino_1226/\\
        spg_kt1_barotropic_ladder.py [--plant]

``--plant`` moves one wet cell of EVERY rung that is currently AT BAR -- the
carried mesh ladder and all three loop-entry seeds -- by 1 ulp.  All four MUST
then read DEBT.  Planting into a row that already reads DEBT would prove
nothing, which is why the plant does not touch one.
"""
from __future__ import annotations

import argparse
import glob
import os
import struct
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from rebuild_nemo_restart import rebuild  # noqa: E402

RUN_KT1 = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
           "RUN_FROMREST_KT1")
RN_DT = 2700.0                      # namdom rn_Dt
R1_DT = 1.0 / RN_DT                 # stpmlf.F90:131-133 on the Euler step
NN_E = 23                           # ln_bt_auto resolved value (ocean.output:1102)
ICYCLE = 45                         # forward-start window (ocean.output:1265)
_RDT_E = RN_DT / NN_E               # dynspg_ts.f90:1203, rn_Dt not rDt


# ---------------------------------------------------------------- section 1
def audit_record(run_dir: str) -> dict:
    """Is this record's dyn_spg_ts instrumentation usable?  Three checks.

    FAILS CLOSED.  ``out["checks_run"]`` counts how many of the three actually
    produced evidence; a missing stream, an unreadable file or a netCDF4
    failure leaves its check unrun, and the caller must read fewer than three
    as INDETERMINATE, never as "coherent".  An audit that says a record is
    usable when it simply could not look is the same defect as a gate that
    passes because it scored nothing.
    """
    out = {"checks_run": 0}
    # (a) the substep trajectory's own header disagrees with its length.
    path = os.path.join(run_dir, "substep_dump.bin")
    if os.path.exists(path):
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            jpi, jpj, icy = struct.unpack("<3i", fh.read(12))
        need = 12 + icy * (4 + 7 * jpi * jpj * 8)
        # the jn markers must read 1..icy under the header's own stride
        with open(path, "rb") as fh:
            raw = fh.read()
        markers = []
        stride = 4 + 7 * jpi * jpj * 8
        off = 12
        for _ in range(icy):
            if off + 4 > len(raw):
                break
            markers.append(struct.unpack("<i", raw[off:off + 4])[0])
            off += stride
        out["substep_header"] = (jpi, jpj, icy)
        out["substep_size_bytes"] = size
        out["substep_size_implied"] = need
        out["substep_markers_ok"] = markers == list(range(1, icy + 1))
        out["checks_run"] += 1
    # (b) two streams that must satisfy a closed identity, do not.
    #     From rest at jn=1: ssh_frc == 0 (no E-P) so ssha_e(1) == 0, hence
    #     zu_spg == 0; the seed is rest so dyn_cor_2D and the bottom stress
    #     are 0 too.  dynspg_ts.f90:839-845 then reduces to
    #         un_e(after jn=1) = rDt_e * zu_frc * ssumask.
    rdt_e = RN_DT / NN_E
    try:
        zu = np.fromfile(os.path.join(run_dir, "spg_dump_zu_frc.bin"),
                         dtype="<f8")
        ub1 = np.fromfile(os.path.join(run_dir, "spg_dump_ub_substep1.bin"),
                          dtype="<f8")
        sfrc = np.fromfile(os.path.join(run_dir, "spg_dump_ssh_frc.bin"),
                           dtype="<f8")
        ssh1 = np.fromfile(os.path.join(run_dir, "spg_dump_ssh_substep1.bin"),
                           dtype="<f8")
        nj = 29
        ni = 30
        ub1 = ub1.reshape(nj, ni)[2:-2, 2:-2].ravel()
        out["ssh_frc_zero"] = bool((sfrc == 0).all())
        out["ssh_substep1_zero"] = bool((ssh1 == 0).all())
        pred = rdt_e * zu
        out["identity_cells_unequal"] = int((pred != ub1).sum())
        out["identity_n"] = int(pred.size)
        out["identity_max_abs"] = float(np.abs(pred - ub1).max())
        out["identity_field_max"] = float(np.abs(ub1).max())
        out["checks_run"] += 1
    except (OSError, ValueError) as exc:                 # pragma: no cover
        out["identity_error"] = str(exc)
    # (c) different streams carry different ranks' tiles.  Land patterns of
    #     the DINO basin differ between the western and eastern tile columns;
    #     a coherent record cannot show one stream on each.
    try:
        import netCDF4 as nc
        masks = {}
        for p in sorted(glob.glob(os.path.join(run_dir, "mesh_mask_*.nc"))):
            r = int(p[-7:-3])
            ds = nc.Dataset(p)
            masks[r] = {k: np.squeeze(ds.variables[k][:]).astype(bool)
                        for k in ("tmaskutil", "umaskutil")}
            ds.close()

        def which(fname, key, interior):
            a = np.fromfile(os.path.join(run_dir, fname), dtype="<f8")
            a = (a.reshape(25, 26) if interior
                 else a.reshape(29, 30)[2:-2, 2:-2])
            nz = a != 0.0
            return [r for r in sorted(masks)
                    if np.array_equal(nz, masks[r][key])]
        out["tiles_zu_frc"] = which("spg_dump_zu_frc.bin", "umaskutil", True)
        out["tiles_qns"] = which("sbc_dump_qns.bin", "tmaskutil", False)
        out["checks_run"] += 1
    except Exception as exc:                             # pragma: no cover
        out["tile_error"] = str(exc)
    # (d) THE RECORD MAY ALREADY BE RANK-TAGGED.  Checks (a)-(c) all look at
    #     the OLD, un-suffixed filenames; on a record acquired by the config
    #     copy they simply do not exist, and the audit would then report
    #     INDETERMINATE -- "could not look" -- for a record that is in fact
    #     clean.  That is the wrong verdict and it would keep the two rungs
    #     waived forever.  So the rank-tagged layout is detected explicitly,
    #     and it is only accepted when the file COUNT matches the run's own
    #     decomposition (layout.dat's jpnij) times the substep count.
    tagged = sorted(glob.glob(os.path.join(run_dir, "substep_r*_s*.bin")))
    if tagged:
        jpnij = None
        lay = os.path.join(run_dir, "layout.dat")
        if os.path.exists(lay):
            with open(lay) as fh:
                for ln in fh:
                    parts = ln.split()
                    if len(parts) >= 6 and parts[0].isdigit():
                        jpnij = int(parts[0])
                        break
        ranks = sorted({os.path.basename(t).split("_")[1] for t in tagged})
        steps = sorted({os.path.basename(t).split("_")[2] for t in tagged})
        out["rank_tagged_files"] = len(tagged)
        out["rank_tagged_ranks"] = len(ranks)
        out["rank_tagged_substeps"] = len(steps)
        out["rank_tagged_jpnij"] = jpnij
        out["rank_tagged_complete"] = bool(
            jpnij is not None and len(ranks) == jpnij
            and len(tagged) == len(ranks) * len(steps))
        out["checks_run"] += 1
    return out


#: The acquisition that produces a race-free record.  A CONFIG COPY: it never
#: writes inside cfgs/DINO or src/, and it refuses rather than trying.
ACQUISITION = ("scripts/validate/ocean_fidelity/dino_1226/"
               "nemo_dino_kt1_rankdump/run.sh")


# ---------------------------------------------------------------- helpers
def _mesh_field(run_dir: str, name: str) -> np.ndarray:
    """Stitch one mesh_mask variable over the 16 tiles (haloless, no race)."""
    import netCDF4 as nc
    out = None
    for p in sorted(glob.glob(os.path.join(run_dir, "mesh_mask_*.nc"))):
        ds = nc.Dataset(p)
        x0, y0 = (int(v) for v in ds.DOMAIN_position_first)
        x1, y1 = (int(v) for v in ds.DOMAIN_position_last)
        nxg, nyg = (int(v) for v in ds.DOMAIN_size_global)
        a = np.squeeze(np.asarray(ds.variables[name][:], dtype=np.float64))
        if out is None:
            out = np.full(a.shape[:-2] + (nyg, nxg), np.nan)
        out[..., y0 - 1:y1, x0 - 1:x1] = a
        ds.close()
    if out is None:
        raise SystemExit(f"no mesh_mask tiles under {run_dir}")
    return out


class Row:
    def __init__(self):
        self.rows = []
        self.fail = False

    def score(self, name, built, oracle, mask, floor=None):
        b = np.asarray(built, dtype=np.float64)[mask]
        o = np.asarray(oracle, dtype=np.float64)[mask]
        if b.size == 0:
            # An empty comparison is not agreement.  A mask that selects
            # nothing would otherwise read AT BAR at n=0.
            self.fail = True
            self.rows.append(dict(name=name, n=0, cells=None, maxabs=None,
                                  rms=None, floor=None, status="UNMEASURED",
                                  reason="the mask selected no cells"))
            return self.rows[-1]
        ne = int((b != o).sum())
        d = b - o
        row = dict(name=name, n=int(b.size), cells=ne,
                   maxabs=float(np.abs(d).max()) if b.size else 0.0,
                   rms=float(np.sqrt(np.mean(d ** 2))) if b.size else 0.0,
                   floor=floor, status="AT BAR" if ne == 0 else "DEBT")
        if ne:
            self.fail = True
        self.rows.append(row)
        return row

    def unmeasured(self, name, reason):
        self.rows.append(dict(name=name, n=0, cells=None, maxabs=None,
                              rms=None, floor=None, status="UNMEASURED",
                              reason=reason))

    def render(self):
        print(f"{'rung':38s}{'cells !=':>10s}{'max|d|':>13s}"
              f"{'rms':>13s}{'NEMO step':>13s}  status")
        for r in self.rows:
            if r["status"] == "UNMEASURED":
                print(f"{r['name']:38s}{'-':>10s}{'-':>13s}{'-':>13s}"
                      f"{'-':>13s}  UNMEASURED")
                print(f"{'':38s}  needs: {r['reason']}")
                continue
            fl = "-" if r["floor"] is None else f"{r['floor']:13.4e}"
            rm = "-" if r["rms"] is None else f"{r['rms']:13.4e}"
            print(f"{r['name']:38s}{r['cells']:>10d}{r['maxabs']:13.4e}"
                  f"{rm:>13s}{fl:>13s}  {r['status']}")


# ---------------------------------------------------------------- section 4
#: NEMO's half-step-back interpolation coefficients, read from
#: ``ts_bck_interp`` (dynspg_ts.f90, the SUBROUTINE near the end of the file)
#: for this card's ``rn_bt_alpha = 0`` (namelist_cfg:360).  ``ll_init`` is
#: TRUE on the Euler first step (stpmlf.f90:241), so the jn=1 and jn=2 rows
#: are the forward-backward and AB2-AM3 ramps and every later substep is
#: AB3-AM4.  Transcribed, not derived: a coefficient guessed here would make
#: the calibration below fail for the wrong reason.
_BCK = {1: (1.0, 0.0, 0.0, 0.0),
        2: (1.0833333333333, -0.1666666666666, 0.0833333333333, 0.0)}
_BCK_AB3AM4 = (0.614, 0.285, 0.088, 0.013)


def _load_stitch():
    """``stitch`` from the rank-tagged record's own reader, not a second copy.

    RULE 4 (search before you build): the tile placement, the header parsing
    and the double-cover check already exist in
    ``nemo_dino_kt1_rankdump/read_rankdump.py`` and are shared with the kt=1
    slope record.  A second implementation here could drift from it, and the
    placement rule is exactly the part that must not.
    """
    import importlib.util
    p = os.path.join(_HERE, "nemo_dino_kt1_rankdump", "read_rankdump.py")
    if not os.path.exists(p):                              # pragma: no cover
        raise SystemExit(f"the rank-dump reader is missing: {p}")
    spec = importlib.util.spec_from_file_location("_read_rankdump", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.stitch


def stitch_all(run_dir: str):
    """Every substep of the rank-tagged record, stitched, in substep order."""
    stitch = _load_stitch()
    names = sorted(glob.glob(os.path.join(run_dir, "substep_r*_s*.bin")))
    steps = sorted({int(os.path.basename(t).split("_s")[1][:-4])
                    for t in names})
    if steps != list(range(1, len(steps) + 1)):
        raise SystemExit(f"substeps present are {steps} -- not 1..N")
    return [stitch(run_dir, n) for n in steps]


def nemo_self_calibration(subs, twet, uwet, vwet, tab, plant=False,
                          land_ref=None):
    """Rebuild NEMO's substep n+1 operands from NEMO's substep n, in NEMO's
    own numbers.  legoESM does not appear in this section at all.

    Two statements, both transcribed:

    * the SWAP, dynspg_ts.f90:830 ``un_e = ua_e`` and :838 ``sshn_e = ssha_e``
      (with :828/:836 rotating the older levels one slot down).  So substep
      n+1's ENTRY must be substep n's EXIT, bit for bit.
    * the HALF-STEP BACK INTERPOLATION, dynspg_ts.f90:662-665
      ``zsshp2_e = za0*ssha_e + za1*sshn_e + za2*sshb_e + za3*sshbb_e``
      where the swap makes ``sshb_e(n) = sshn_e(n-1)`` and
      ``sshbb_e(n) = sshn_e(n-2)``, both zero before they exist
      (:464-467 under ``ll_init``).

    WHAT THESE TWO CAN AND CANNOT SEE (Rule 2, and an independent review
    made this correction).  Both are POINTWISE: ``ts_bck_interp`` combines
    four values at one ``(i,j)`` and the swap is a copy.  Both sides come
    from the same stitched arrays, so both identities are invariant under any
    permutation of cells -- **a systematic mis-stitch cancels exactly**.  They
    calibrate the header parsing, the substep ordering, the coefficient rows
    and the fact that ``:664`` OVERWRITES ``:574`` (the dump at ``:926`` sees
    only the second).  They do NOT calibrate tile PLACEMENT.

    C3 below is the placement check, and it is a different kind of statement:
    ``ssha_e`` is multiplied by ``ssmask`` (``:627``), so it is exactly zero
    on land.  A tile placed at the wrong offset moves the land pattern, which
    no pointwise identity can notice and this one cannot miss.
    """
    n = len(subs)
    # AN EMPTY MASK IS NOT AGREEMENT.  Row.score already refuses n=0; these
    # two loops did not, so all-False masks made every identity vacuously
    # true and the calibration reported "reproduces NEMO's own statements
    # exactly" on a record that violated both.  Found by an independent
    # review.
    if n == 0 or not (twet.any() and uwet.any() and vwet.any()):
        print(f"  CALIBRATION REFUSED: masks select {int(twet.sum())}/"
              f"{int(uwet.sum())}/{int(vwet.sum())} cells over {n} substeps; "
              "an identity checked on no cells is not an identity")
        tab.unmeasured("CALIBRATION (NEMO from NEMO)",
                       "a mask that selects some cells and a record with "
                       "some substeps")
        tab.fail = True
        return False
    swap_bad = bck_bad = 0
    _planted = _swap_planted = False
    swap_first = bck_first = None
    swap_max = bck_max = 0.0
    for i in range(n):
        ssha, sshn = subs[i]["ssha_e"], subs[i]["sshn_e"]
        sshb = subs[i - 1]["sshn_e"] if i >= 1 else np.zeros_like(sshn)
        sshbb = subs[i - 2]["sshn_e"] if i >= 2 else np.zeros_like(sshn)
        za = _BCK.get(i + 1, _BCK_AB3AM4)
        pred = za[0] * ssha + za[1] * sshn + za[2] * sshb + za[3] * sshbb
        if plant:
            # A NONZERO cell, deliberately.  nextafter(0.0) is 4.94e-324, so
            # planting into a zero would be detected by a comparison that
            # only distinguishes zero from subnormal -- a much weaker claim
            # than "one ulp of a live value is visible".  Substep 1's
            # zsshp2_e is za0*ssha_e and ssha_e(1) is identically zero from
            # rest, so the first substep with a nonzero prediction is used.
            nz = (np.argwhere(twet & (pred != 0.0))
                  if not _planted else np.empty((0, 2), int))
            if nz.size:
                _planted = True
                pred = pred.copy()
                pred[nz[0][0], nz[0][1]] = np.nextafter(
                    pred[nz[0][0], nz[0][1]], np.inf)
        bad = int((pred[twet] != subs[i]["zsshp2_e"][twet]).sum())
        if bad:
            bck_bad += bad
            bck_first = bck_first if bck_first is not None else i + 1
            bck_max = max(bck_max, float(np.abs(
                (pred - subs[i]["zsshp2_e"])[twet]).max()))
        if i + 1 < n:
            for k_exit, k_entry, m in (("ssha_e", "sshn_e", twet),
                                       ("ua_e", "un_e", uwet),
                                       ("va_e", "vn_e", vwet)):
                _exit = subs[i][k_exit]
                if plant and not _swap_planted and k_exit == "ua_e":
                    _nzs = np.argwhere(m & (_exit != 0.0))
                    if _nzs.size:
                        _swap_planted = True
                        _exit = _exit.copy()
                        _exit[_nzs[0][0], _nzs[0][1]] = np.nextafter(
                            _exit[_nzs[0][0], _nzs[0][1]], np.inf)
                d = _exit[m] != subs[i + 1][k_entry][m]
                if d.any():
                    swap_bad += int(d.sum())
                    swap_first = swap_first if swap_first is not None else i + 1
                    swap_max = max(swap_max, float(np.abs(
                        (_exit - subs[i + 1][k_entry])[m]).max()))
    print(f"  C1 swap identity        exit(n) == entry(n+1) over {n - 1} "
          f"joins x 3 fields: {swap_bad} cells unequal"
          + (f", first at substep {swap_first}, max|d| {swap_max:.3e}"
             if swap_bad else ""))
    print(f"  C2 ts_bck_interp        zsshp2_e rebuilt from NEMO's own "
          f"sshn_e/ssha_e over {n} substeps: {bck_bad} cells unequal"
          + (f", first at substep {bck_first}, max|d| {bck_max:.3e}"
             if bck_bad else ""))
    if swap_bad or bck_bad:
        tab.unmeasured(
            "CALIBRATION (NEMO from NEMO)",
            "a reader that reproduces NEMO's own statements; it does not, so "
            "every legoESM row below it is uninterpretable")
        tab.fail = True
        return False
    # C3 -- PLACEMENT.  ssha_e carries ssmask (:627), so its zero set is the
    # land mask.  Compared against the stitched mesh_mask that the R0 rungs
    # already use, on the last substep (by which point ssha_e is nonzero
    # everywhere wet; at substep 1 from rest it is identically zero and the
    # check would be vacuous, which is why the substep is chosen and not
    # defaulted).
    land_bad = None
    if land_ref is not None:
        wetpat = np.zeros_like(land_ref, dtype=bool)
        for i in range(n):
            wetpat |= subs[i]["ssha_e"] != 0.0
        land_bad = int((wetpat != land_ref).sum())
        print(f"  C3 placement            the zero set of ssha_e (ssmask, "
              f":627) against the stitched mesh_mask: {land_bad} cells "
              "disagree")
        if land_bad:
            tab.unmeasured(
                "CALIBRATION (NEMO from NEMO)",
                "a reader whose tiles land where the mesh says they do; the "
                "land pattern of ssha_e does not match tmask, so the stitch "
                "is misplaced and nothing below can be read")
            tab.fail = True
            return False
    else:
        print("  C3 placement            UNMEASURED: no mesh mask was passed, "
              "so a systematic mis-stitch would cancel in C1 and C2 and "
              "nothing here would see it")
    print("  -> the reader reproduces NEMO's own two statements exactly and "
          "its tiles land where the mesh says they do")
    return True


#: Per-substep row -> the NEMO statement it stands for.  A row that breaks is
#: reported as this statement, never as "the barotropic solver".
_ROW_STATEMENT = {
    "eta_entry": ("dynspg_ts.f90:489 sshn_e = pssh(Kbb) (substep 1) / "
                  ":838 sshn_e = ssha_e (the swap)"),
    "u_entry":   ("dynspg_ts.f90:490 un_e = puu_b(Kbb) (substep 1) / "
                  ":830 un_e = ua_e (the swap)"),
    "v_entry":   ("dynspg_ts.f90:491 vn_e = pvv_b(Kbb) (substep 1) / "
                  ":832 vn_e = va_e (the swap)"),
    "eta_pgf":   "dynspg_ts.f90:662-665 ts_bck_interp half-step back eta",
    "eta_exit":  ("dynspg_ts.f90:627 ssha_e = (sshn_e - rDt_e*(ssh_frc + "
                  "zhdiv))*ssmask"),
    "u_exit":    ("dynspg_ts.f90:732-736 ua_e = (un_e + rDt_e*(zu_spg + "
                  "zu_trd + zu_frc))*ssumask"),
    "v_exit":    ("dynspg_ts.f90:738-742 va_e = (vn_e + rDt_e*(zv_spg + "
                  "zv_trd + zv_frc))*ssvmask"),
}
#: legoESM trace slot -> (NEMO dump name, how to slice the lego array, mask key)
_ROWS = (("eta_entry", "sshn_e",   "t"),
         ("u_entry",   "un_e",     "u"),
         ("v_entry",   "vn_e",     "v"),
         ("eta_pgf",   "zsshp2_e", "t"),
         ("eta_exit",  "ssha_e",   "t"),
         ("u_exit",    "ua_e",     "u"),
         ("v_exit",    "va_e",     "v"))


def frozen_forcing_identity(subs, fro, tr, dt_s, twet, uwet, vwet):
    """From rest at substep 1 the whole substep collapses to the forcing.

    NEMO, dynspg_ts.f90:732-736 with every other operand identically zero at
    jn=1 from rest (ssh_frc == 0 makes ssha_e(1) == 0 so zu_spg == 0; the rest
    seed makes dyn_cor_2D and the bottom stress 0):

        ua_e(1) = rDt_e * zu_frc * ssumask

    legoESM's substep body reduces the same way.  So BOTH sides are checked
    against their own arithmetic before the two are compared to each other.
    An R1 row that scored legoESM's ``slow_u`` against NEMO's ``zu_frc``
    without this is comparing two names, not two quantities -- and this
    campaign has published a residual built on exactly that mistake before.
    """
    out = {}
    for side, u, v, mu, mv, dt in (
            ("NEMO", subs[0]["ua_e"], subs[0]["va_e"], uwet, vwet, None),
            ("lego", np.asarray(tr["u_exit"][0])[:, 1:],
             np.asarray(tr["v_exit"][0])[1:, :], uwet, vwet, dt_s)):
        if side == "NEMO":
            fu, fv, dt = fro["zu_frc"], fro["zv_frc"], None
        else:
            fu = np.asarray(tr["slow_u"][0])[:, 1:]
            fv = np.asarray(tr["slow_v"][0])[1:, :]
        step = dt if dt is not None else _RDT_E
        out[side] = (int((step * fu[mu] != u[mu]).sum()),
                     int((step * fv[mv] != v[mv]).sum()))
        print(f"  identity  {side}: u_exit(1) == dt*forcing on "
              f"{out[side][0]} cells unequal (u), {out[side][1]} (v)")
    return out


def _read_tile_header(path):
    """One tile's self-describing header, through the record's own reader."""
    import importlib.util
    q = os.path.join(_HERE, "nemo_dino_kt1_rankdump", "read_rankdump.py")
    spec = importlib.util.spec_from_file_location("_read_rankdump3", q)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.read_tile(path)["header"]


def _render_substeps(per, first, n):
    """The ordered table: one line per substep, cells unequal per statement."""
    cols = [r[0] for r in _ROWS]
    print()
    print("  " + "jn".rjust(4) + "".join(c.rjust(13) for c in cols))
    for i in range(n):
        line = "  " + str(i + 1).rjust(4)
        for c in cols:
            line += str(per[i][c][0]).rjust(13)
        print(line)
        if first is not None and i + 1 == first[0]:
            print("  " + "-" * (4 + 13 * len(cols))
                  + "  <- first non-bit substep")
    print("  (cells unequal, out of the wet cells of that field's grid; "
          "0 is the bar)")
    print()
    print("  the same table as max|d|, because 'how many cells' cannot say "
          "whether the gap GROWS across the window:")
    print("  " + "jn".rjust(4) + "".join(c.rjust(13) for c in cols))
    for i in [j for j in (0, 1, 2, 4, 9, 19, n - 1) if j < n]:
        print("  " + str(i + 1).rjust(4)
              + "".join(f"{per[i][c][1]:13.4e}" for c in cols))


def substep_ladder(subs, tr, n_loop, twet, uwet, vwet, plant=False):
    """legoESM's substep operator against NEMO's, substep by substep.

    WHY THIS IS A GIVEN-INPUTS COMPARISON AND NOT JUST A FREE RUN, which is
    the whole reason the table is read from the top down: legoESM is driven
    from its own state, so at substep n its inputs are its own substep n-1
    outputs.  But the three ENTRY rows are scored too, so at the FIRST substep
    where any row breaks, the entry rows above it are known to be 0 cells --
    i.e. that substep's operator ran on NEMO's own operands, bit for bit, and
    the break belongs to the operator and not to inherited drift.  The gate
    states which of the two it is rather than leaving it to be assumed; the
    claim is only made when the entry rows at that substep are in fact 0.
    """
    masks = {"t": twet, "u": uwet, "v": vwet}
    if plant:
        # Pass 1 with no plant, to find a row that is AT BAR.  Planting into
        # a row that already reads DEBT proves nothing about the gate, which
        # this function's own docstring claimed and did not enforce.
        _per0, _, _, _ = substep_ladder(subs, tr, n_loop, twet, uwet,
                                        vwet, plant=False)
        plant = next((((i + 1), k) for i in range(len(_per0))
                      for k, _, _ in _ROWS if _per0[i][k][0] == 0), None)
        if plant is None:
            raise SystemExit(
                "--plant found no row at 0 cells to plant into; every row is "
                "already DEBT, so a plant here would prove nothing")
        print(f"  PLANT: 1 ulp into substep {plant[0]} row {plant[1]!r}, "
              "which is AT BAR without it")
    n = min(len(subs), int(n_loop))
    # A COUNT MISMATCH IS A FINDING, NOT A SLICE.  If legoESM runs a different
    # number of substeps than NEMO, min() would quietly score the overlap and
    # the table would look complete -- which is exactly the defect this round
    # found at kt=2 (91 against NEMO's 68).  Say it out loud instead.
    mismatch = int(n_loop) != len(subs)
    if mismatch:
        print(f"  COUNT MISMATCH: NEMO's record has {len(subs)} substeps and "
              f"legoESM runs {int(n_loop)}. Only the first {n} can be scored, "
              "and the two loops are NOT the same loop -- this is a FAILING "
              "row on its own, not a slice.")
    if n == 0:
        raise SystemExit("the record and the loop do not overlap at all")
    first = None
    per = []
    for i in range(n):
        row = {}
        for slot, nemo_name, mk in _ROWS:
            m = masks[mk]
            a = np.asarray(tr[slot][i])
            a = a[:, 1:] if mk == "u" else (a[1:, :] if mk == "v" else a)
            if plant and (i + 1, slot) == plant:
                a = a.copy()
                q = np.argwhere(m)[0]
                a[q[0], q[1]] = np.nextafter(a[q[0], q[1]], np.inf)
            o = subs[i][nemo_name]
            d = a[m] - o[m]
            bad = int((~(a[m] == o[m])).sum())     # non-finite counts UNEQUAL
            row[slot] = (bad, float(np.abs(d).max()) if d.size else 0.0)
            if bad and first is None:
                first = (i + 1, slot)
        per.append(row)
    return per, first, n, mismatch


def stitch_frozen_forcing(run_dir: str, hdrs):
    """NEMO's frozen barotropic forcing, stitched from the rank-tagged streams.

    These three streams have NO header of their own -- the writer is the
    oracle's own (dynspg_ts.f90:441-443) and predates the rank tagging, which
    only renamed the files.  ``zu_frc``/``zv_frc`` are written over the INNER
    region ``ji=Nis0..Nie0, jj=Njs0..Nje0``; ``ssh_frc`` over the full local
    ``(jpi,jpj)``.  So the geometry comes from the SUBSTEP files' headers for
    the same ranks -- the same per-rank decomposition, read from the record
    rather than assumed -- and a length that disagrees with it is fatal.
    """
    out = {}
    for name, inner in (("zu_frc", True), ("zv_frc", True),
                        ("ssh_frc", False)):
        field = None
        for rank, h in sorted(hdrs.items()):
            p = os.path.join(run_dir, f"spg_dump_{name}_r{rank:04d}.bin")
            if not os.path.exists(p):
                return {}
            a = np.fromfile(p, dtype="<f8")
            gj0, gj1, gi0, gi1, j0, j1, i0, i1 = _tile_slices(h)
            if field is None:
                field = np.full((h["jpjglo"] - 2 * h["nn_hls"],
                                 h["jpiglo"] - 2 * h["nn_hls"]), np.nan)
            if inner:
                want = (j1 - j0) * (i1 - i0)
                if a.size != want:
                    raise SystemExit(
                        f"{p}: {a.size} doubles but this rank's own substep "
                        f"header gives an inner region of {want}")
                field[gj0:gj1, gi0:gi1] = a.reshape(j1 - j0, i1 - i0)
            else:
                if a.size != h["jpi"] * h["jpj"]:
                    raise SystemExit(
                        f"{p}: {a.size} doubles but its rank's header says "
                        f"jpi*jpj = {h['jpi'] * h['jpj']}")
                field[gj0:gj1, gi0:gi1] = a.reshape(
                    h["jpj"], h["jpi"])[j0:j1, i0:i1]
        if field is None or np.isnan(field).any():
            raise SystemExit(f"{name}: the stitched field has holes")
        out[name] = field
    return out


def _tile_slices(h):
    """Placement of one rank's inner block, from the record's own reader."""
    import importlib.util
    p = os.path.join(_HERE, "nemo_dino_kt1_rankdump", "read_rankdump.py")
    spec = importlib.util.spec_from_file_location("_read_rankdump2", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    shape = mod.haloless_shape(h)
    return mod.tile_slices(h, shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=RUN_KT1)
    ap.add_argument("--plant", action="store_true")
    ap.add_argument("--plant-reader", action="store_true",
                    help="move one cell of the CALIBRATION's own prediction "
                         "by 1 ulp; C2 must then report cells unequal and "
                         "the gate must refuse to score anything below it")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--save-trace", default=None,
                    help="npz path for legoESM's per-substep trajectory")
    args = ap.parse_args()

    print("=" * 78)
    print("1. RECORD AUDIT -- are this run's dyn_spg_ts streams usable?")
    print("=" * 78)
    a = audit_record(args.run_dir)
    raced = False
    if "substep_header" in a:
        jpi, jpj, icy = a["substep_header"]
        print(f"  substep_dump.bin header says jpi={jpi} jpj={jpj} "
              f"icycle={icy}")
        print(f"    -> that header implies {a['substep_size_implied']} bytes; "
              f"the file is {a['substep_size_bytes']}")
        print(f"    -> substep markers read 1..{icy} in order: "
              f"{a['substep_markers_ok']}")
        if (a["substep_size_implied"] != a["substep_size_bytes"]
                or not a["substep_markers_ok"]):
            raced = True
    if "identity_cells_unequal" in a:
        print(f"  ssh_frc identically zero: {a['ssh_frc_zero']}   "
              f"ssh after substep 1 identically zero: "
              f"{a['ssh_substep1_zero']}")
        print("  closed identity  un_e(after jn=1) = rDt_e * zu_frc * ssumask")
        print(f"    -> cells unequal {a['identity_cells_unequal']}"
              f"/{a['identity_n']}   max|d| {a['identity_max_abs']:.4e}"
              f"   (the field's own max is {a['identity_field_max']:.4e})")
        if a["identity_cells_unequal"]:
            raced = True
    if "tiles_zu_frc" in a:
        print(f"  land pattern of spg_dump_zu_frc.bin matches ranks "
              f"{a['tiles_zu_frc']}")
        print(f"  land pattern of sbc_dump_qns.bin  matches ranks "
              f"{a['tiles_qns']}")
        if set(a["tiles_zu_frc"]).isdisjoint(a["tiles_qns"]):
            raced = True
    if "rank_tagged_files" in a:
        print(f"  RANK-TAGGED substep streams present: "
              f"{a['rank_tagged_files']} files = {a['rank_tagged_ranks']} "
              f"ranks x {a['rank_tagged_substeps']} substeps; layout.dat "
              f"says jpnij={a['rank_tagged_jpnij']}; complete: "
              f"{a['rank_tagged_complete']}")
    print()
    _n = int(a.get("checks_run", 0))
    _tagged_ok = bool(a.get("rank_tagged_complete"))
    if raced and not _tagged_ok:
        _verdict = ("RACED -- these streams are a 16-rank interleave and "
                    "cannot be scored")
    elif _tagged_ok:
        # A rank-tagged record does not need checks (a)-(c): those exist to
        # detect an interleave, and one rank per file cannot interleave.  The
        # completeness check above is what stands in their place, and it is
        # against the run's OWN decomposition rather than a constant.
        _verdict = ("RANK-TAGGED and complete -- one file per (rank, "
                    "substep), so there is no interleave to detect")
        raced = False
    elif _n < 3:
        _verdict = (f"INDETERMINATE -- only {_n}/3 checks could run, so this "
                    "audit did not look; treat the streams as unusable")
        raced = True
    else:
        _verdict = "coherent -- all 3 checks ran and none found damage"
    print("  VERDICT: " + _verdict)
    if raced:
        print("  cause: dynspg_ts.F90:206 and :926 guard on kt only, never "
              "on narea;")
        print("         all 16 ranks OPEN the same path with "
              "STATUS='REPLACE'.")
        print("  fix:   " + ACQUISITION + " acquires a clean record")
    if args.audit_only:
        return 0

    # ------------------------------------------------------------ oracle
    print()
    print("=" * 78)
    print("2. THE LADDER")
    print("=" * 78)
    import jax
    if not jax.config.jax_enable_x64:
        # BEFORE set_policy: PrecisionPolicy.fp64() turns x64 on itself, so a
        # check after it can never fire.
        raise SystemExit("JAX x64 is disabled; re-run with JAX_ENABLE_X64=1")
    from legoesm.core.precision import PrecisionPolicy, set_policy
    set_policy(PrecisionPolicy.fp64())                            # Rule 1c
    import jax.numpy as jnp
    import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as ocean_model
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        compute_layer_thickness,
        min_cell_to_uface,
    )
    from legoesm.ocean.experiments import dino as dm
    from legoesm.ocean.fidelity import nemo_dino_mesh as ndm

    pattern = os.path.join(args.run_dir, "DINO_00000001_restart_*.nc")
    if not glob.glob(pattern):
        raise SystemExit(f"no restart tiles match {pattern}")
    R = rebuild(pattern, ["sshn", "ubtacc_aa", "vbtacc_aa", "ub", "vb",
                          "sshb"])
    for k in ("sshn", "ubtacc_aa", "vbtacc_aa"):
        if k not in R:
            raise SystemExit(f"restart carries no {k!r}: this record predates "
                             "the trddump barotropic accumulator")
    # ``ubtacc_aa`` is a SUM over steps (trddump.F90:161).  It is this step's
    # puu_b(Kaa)*r1_Dt only if exactly one step was folded in; the file's own
    # counter says so, and a multi-step record must not be read as a kt=1
    # oracle.
    import netCDF4 as _nc
    _ns = set()
    for _p in sorted(glob.glob(pattern)):
        _ds = _nc.Dataset(_p)
        if "nacc_steps" in _ds.variables:
            _ns.add(float(np.squeeze(_ds.variables["nacc_steps"][:])))
        _ds.close()
    print(f"  trddump accumulator folded {sorted(_ns)} step(s) "
          "(must be exactly [1.0])")
    if _ns != {1.0}:
        raise SystemExit("ubtacc_aa is not a single-step accumulator here")

    # ------------------------------------------------------- legoESM side
    cfg = dm.nemo_faithful_dino_config(
        base=dm.dino_config_for_recipe("nemo_dino_kamm_mlf"))
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    state0 = dm.dino_lat_lon_state(grid, z, cfg)
    forcing = dm.dino_lat_lon_surface_forcing_arrays(grid, cfg)
    sf_step = (dm.dino_step_surface_forcing(forcing)
               if getattr(cfg, "wind_through_step", False) else None)
    mc, _ = dm.dino_lat_lon_model_config(grid, cfg, physics=True)
    print(f"  card: filter={mc.barotropic.barotropic_time_filter!r} "
          f"n_substeps={mc.barotropic.n_barotropic_substeps} "
          f"face_depth={mc.barotropic.barotropic_face_depth!r} "
          f"continuity="
          f"{mc.barotropic.barotropic_continuity_evaluation!r} "
          f"pgf={mc.barotropic.barotropic_pgf_evaluation!r} "
          f"seed={mc.barotropic.barotropic_seed_evaluation!r}")

    original = ocean_model.barotropic_substeps_latlon_cgrid
    cap = {}

    import inspect
    sig = inspect.signature(original)

    def wrapper(*a, **k):
        if "trace" in cap:
            return original(*a, **k)
        ba = sig.bind(*a, **k)
        ba.apply_defaults()
        for nm in ("dt_s", "n_substeps", "substep_scale", "F_slow_eta",
                   "F_slow_u", "F_slow_v", "add_barotropic_coriolis"):
            if nm in ba.arguments:
                cap[nm] = ba.arguments[nm]
        k2 = dict(k)
        k2["_nemo_substep_trace_test_hook"] = True
        state_new, transports, trace = original(*a, **k2)
        cap["trace"] = [np.asarray(t) for t in trace]
        cap["eta_out"] = np.asarray(state_new.eta.data)
        return state_new, transports

    model = LatLonCGridOceanModel(grid, z, mc)
    st, rate = dm.apply_dino_lat_lon_surface_forcing(
        state0, forcing, z, cfg, RN_DT, t_seconds=RN_DT, return_rate=True)
    ocean_model.barotropic_substeps_latlon_cgrid = wrapper
    try:
        with jax.disable_jit():
            model.step(st, dt=RN_DT, surface_forcing=sf_step,
                       external_tracer_rate=rate)
    finally:
        ocean_model.barotropic_substeps_latlon_cgrid = original
    if "trace" not in cap:
        raise SystemExit("the barotropic solver was never called")

    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import _compute_weights
    n_sub = int(cap["n_substeps"])
    # substep_scale COMES FROM THE CALL.  This used to read
    # ``mc.barotropic.barotropic_substep_scale`` -- a field that exists
    # nowhere in the package -- so the getattr silently returned 1.  It is
    # right at kt=1 only because the Euler path really does pass 1; the
    # leap-frog path passes 2 and the same line in the kt=2 gate produced a
    # window the model never ran.  An independent review caught it there.
    _scale = int(cap.get("substep_scale", 1))
    w_filter, w_total, w_transport, n_loop = _compute_weights(
        mc, n_sub, np.float64, substep_scale=_scale)
    w_filter = np.asarray(w_filter, dtype=np.float64)
    w_total = float(np.asarray(w_total))
    print(f"  loop: dt_s={float(np.asarray(cap['dt_s'])):.9f} s  "
          f"n_substeps={n_sub}  substep_scale={_scale}  n_loop={n_loop}  "
          f"sum(w_filter)={w_filter.sum():.17g}  w_total={w_total:.17g}")
    print(f"  NEMO: rDt_e={RN_DT / NN_E:.9f} s  nn_e={NN_E}  icycle={ICYCLE}")

    # trace layout is pinned by _run_substep_loop's `trace` tuple
    TR = ("eta_entry", "u_entry", "v_entry", "eta_mid", "u_mid", "v_mid",
          "flux_u", "flux_v", "eta_continuity", "eta_pgf", "pgf_u", "pgf_v",
          "slow_u", "slow_v", "drag_u", "drag_v", "u_exit", "v_exit",
          "eta_exit")
    tr = dict(zip(TR, cap["trace"]))
    if tr["eta_exit"].shape[0] != n_loop:
        raise SystemExit(f"trace has {tr['eta_exit'].shape[0]} substeps, "
                         f"loop count is {n_loop}")

    # The rebuild below is the VELOCITY-average form.  The solver has a second
    # form (barotropic_latlon_cgrid.py:1683-1689) that accumulates
    # w*U*H_face and divides the completed mean by a face depth; sea level has
    # no such branch, so section 3's calibration would still pass bit-exactly
    # while the velocity rebuild was wrong.  Refuse rather than report.
    from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
        nemo_flux_form_update_active,  # noqa: F401  (import-time contract)
    )
    _primary_transport = (
        getattr(mc, "momentum_time_integrator", "euler") == "rk3_ws"
        and getattr(mc, "momentum_advection", "vector_invariant") == "flux_form"
        and mc.barotropic.barotropic_time_filter in (
            "nemo_boxcar_ab3", "nemo_boxcar1_ab3"))
    if _primary_transport:
        raise SystemExit(
            "this card resolves to the solver's TRANSPORT primary average; "
            "the velocity rebuild below would be the wrong formula and "
            "section 3's sea-level calibration cannot see it")

    def boxcar(stack):
        """Reproduce the solver's own sequential accumulate-then-divide."""
        acc = np.zeros(stack.shape[1:], dtype=np.float64)
        for i in range(stack.shape[0]):
            acc = acc + w_filter[i] * stack[i]
        return acc / w_total

    # --- section 3: calibrate the rebuild on a quantity the solver returns
    eta_avg_rebuilt = boxcar(tr["eta_exit"])
    eta_cells = int((eta_avg_rebuilt != cap["eta_out"]).sum())
    print()
    print("3. REBUILD CALIBRATION (the method that produces the R4a number)")
    print(f"   sea-level average rebuilt from the trace vs the solver's own "
          f"returned array: {eta_cells} cells unequal, "
          f"max|d| {np.abs(eta_avg_rebuilt - cap['eta_out']).max():.3e}")
    if eta_cells:
        raise SystemExit(
            "the rebuild does not reproduce the solver's own average, so the "
            "velocity number it would produce is not the model's")

    u_avg = boxcar(tr["u_exit"])
    v_avg = boxcar(tr["v_exit"])

    g = ndm.nemo_dino_mesh()
    twet = g.tmask[:, :, 0] > 0.5
    uwet = g.umask[:, :, 0] > 0.5
    vwet = g.vmask[:, :, 0] > 0.5

    # ---------------------------------------------------------- R0
    # THE BOXES BEFORE THE STATE IN THEM (Rule 2).  Every rung below is a
    # depth-weighted mean or a face-column depth, so a wrong layer thickness
    # reaches all of them.
    #
    # NEMO's DINO has TWO vertical ladders and they are not the same one.
    # usr_def_zgr takes the ld_zco branch (ln_zco_nam = .true.) and calls
    # zgr_sco_mi96 on a FLAT column, zflat(:,:) = zHmax = 4000 m
    # (usrdef_zgr.F90:107-118) -- so the 3-D e3t_0 it returns is horizontally
    # UNIFORM (measured: column-to-column spread exactly 0.0 at every level)
    # but it is NOT the 1-D reference ladder e3t_1d.  The two agree for
    # k = 1..25 and then part company: e3t_0/e3t_1d = 0.979, 0.945, 0.924,
    # 0.915, 0.919, 0.935, 0.965, 1.009, 1.070, 1.148 at k = 26..35.
    # legoESM runs on e3t_1d.  NEMO runs on e3t_0.
    #
    # RETRACTED, 2026-09-10: an earlier reading of this called it per-column
    # stretching to the local bathymetry, citing ocean.output:402 ("zgr_lib:
    # zgr_sco : define full vertical s-coord. system using the 2d bathymetry
    # field") and ln_hpg_sco = T.  A reviewer refuted it and the measurement
    # agrees: that log line is printed INSIDE zgr_sco_mi96, which both
    # branches call, and ln_hpg_sco names the pressure-gradient scheme, not
    # the coordinate.  Prose is a pointer, never a citable fact.  The
    # NUMBERS below are unchanged; only the mechanism was wrong, and the
    # correct one makes the gap a 1-D ladder mismatch -- horizontally
    # uniform, deep levels only.
    e3t0 = np.moveaxis(_mesh_field(args.run_dir, "e3t_0"), 0, -1)
    tmask3 = np.moveaxis(_mesh_field(args.run_dir, "tmask"), 0, -1) > 0.5
    e3u0 = np.moveaxis(_mesh_field(args.run_dir, "e3u_0"), 0, -1)
    umask3 = np.moveaxis(_mesh_field(args.run_dir, "umask"), 0, -1) > 0.5
    hu0 = (e3u0 * umask3).sum(axis=-1)
    _h_live = compute_layer_thickness(
        st.eta.data, st.H_bathy.data, z,
        min_water_column_m=mc.min_water_column_m)
    h_live = np.asarray(_h_live)
    Hu_live = np.asarray(jnp.sum(min_cell_to_uface(_h_live), axis=-1))[:, 1:]

    tab = Row()
    tab.score("R0a live layer thickness  e3t_0",
              h_live, e3t0, tmask3)
    tab.score("R0b U-face column depth   hu_0",
              Hu_live, hu0, uwet, floor=float(np.sqrt(np.mean(hu0[uwet] ** 2))))
    # The card DOES carry NEMO's own ladder -- it just does not run on it.
    ne3t = getattr(z, "nemo_e3t_0", None)
    if ne3t is not None:
        ne3t = np.asarray(ne3t)
        if args.plant:
            # Non-vacuity.  Plant into the two rungs that are AT BAR -- a
            # plant on a rung that is already DEBT proves nothing about the
            # gate.  Both must flip.
            ne3t = ne3t.copy()
            _p = np.argwhere(tmask3)[0]
            ne3t[_p[0], _p[1], _p[2]] = np.nextafter(
                ne3t[_p[0], _p[1], _p[2]], np.inf)
        tab.score("R0c carried mesh ladder   e3t_0", ne3t, e3t0, tmask3)
    # R2 -- the loop-entry seed.  From rest NEMO's is identically zero.
    _seed_eta = np.asarray(tr["eta_entry"][0])
    if args.plant:
        _seed_eta = _seed_eta.copy()
        _q = np.argwhere(twet)[0]
        _seed_eta[_q[0], _q[1]] = np.nextafter(_seed_eta[_q[0], _q[1]], np.inf)
    tab.score("R2 seed eta   pssh(Kbb)", _seed_eta, R["sshb"], twet)
    # NEMO does not write puu_b(Kbb); it writes the 3-D ub/vb the depth mean
    # is built from (dynatf_qco.F90:222-235).  Build the mean here with
    # NEMO's own e3u_0/hu_0 weights rather than asserting zero -- a hard-wired
    # zero would read AT BAR even if the record's ub were not zero.
    e3v0 = np.moveaxis(_mesh_field(args.run_dir, "e3v_0"), 0, -1)
    vmask3 = np.moveaxis(_mesh_field(args.run_dir, "vmask"), 0, -1) > 0.5
    hv0 = (e3v0 * vmask3).sum(axis=-1)
    ub3 = np.moveaxis(R["ub"], 0, -1)
    vb3 = np.moveaxis(R["vb"], 0, -1)
    seed_u_oracle = np.where(
        hu0 > 0, (e3u0 * ub3 * umask3).sum(axis=-1) / np.where(hu0 > 0, hu0, 1.0), 0.0)
    seed_v_oracle = np.where(
        hv0 > 0, (e3v0 * vb3 * vmask3).sum(axis=-1) / np.where(hv0 > 0, hv0, 1.0), 0.0)
    _seed_u = np.asarray(tr["u_entry"][0])[:, 1:]
    _seed_v = np.asarray(tr["v_entry"][0])[1:, :]
    if args.plant:
        _seed_u = _seed_u.copy()
        _seed_v = _seed_v.copy()
        _a = np.argwhere(uwet)[0]
        _b = np.argwhere(vwet)[0]
        _seed_u[_a[0], _a[1]] = np.nextafter(_seed_u[_a[0], _a[1]], np.inf)
        _seed_v[_b[0], _b[1]] = np.nextafter(_seed_v[_b[0], _b[1]], np.inf)
    tab.score("R2 seed U     puu_b(Kbb)", _seed_u, seed_u_oracle, uwet)
    tab.score("R2 seed V     pvv_b(Kbb)", _seed_v, seed_v_oracle, vwet)
    # R1/R3 -- their disposition is DECIDED BY the audit, not hard-coded.
    # If a later record passes the audit these rungs become scoreable, and a
    # gate that kept printing UNMEASURED would hide that.
    if raced:
        reason = ("a rank-tagged spg dump (" + ACQUISITION + "); "
                  "this record's "
                  "streams are a 16-rank interleave (section 1)")
        tab.unmeasured("R1 frozen forcing zu_frc/zv_frc/ssh_frc", reason)
        tab.unmeasured("R3 per-substep ssha_e/ua_e/va_e (x45)", reason)
    else:
        # R1/R3 ARE SCOREABLE on a rank-tagged record, and section 4 scores
        # them.  The calibration runs FIRST and its failure disqualifies
        # every row below it (Rule 3: never report a residual before the
        # instrument's noise floor is known -- here the floor is exactly
        # zero, because the reader must reproduce NEMO's own arithmetic).
        print()
        print("=" * 78)
        print("4. THE PER-SUBSTEP LADDER (rank-tagged record)")
        print("=" * 78)
        subs = stitch_all(args.run_dir)
        hdrs = {}
        for _t in sorted(glob.glob(os.path.join(args.run_dir,
                                                "substep_r*_s001.bin"))):
            _h = _read_tile_header(_t)
            hdrs[_h["narea"] - 1] = _h
        print(f"  stitched {len(subs)} substeps x {subs[0]['ranks']} "
              f"ranks onto {subs[0]['sshn_e'].shape} (haloless global)")
        print("  CALIBRATION -- NEMO rebuilt from NEMO, legoESM absent:")
        ok = nemo_self_calibration(subs, twet, uwet, vwet, tab,
                                   plant=args.plant_reader,
                                   land_ref=twet)
        if ok:
            fro = stitch_frozen_forcing(args.run_dir, hdrs)
            if fro:
                tab.score("R1 frozen forcing zu_frc",
                          np.asarray(tr["slow_u"][0])[:, 1:], fro["zu_frc"],
                          uwet,
                          floor=float(np.sqrt(np.mean(
                              fro["zu_frc"][uwet] ** 2))))
                tab.score("R1 frozen forcing zv_frc",
                          np.asarray(tr["slow_v"][0])[1:, :], fro["zv_frc"],
                          vwet,
                          floor=float(np.sqrt(np.mean(
                              fro["zv_frc"][vwet] ** 2))))
                frozen_forcing_identity(
                    subs, fro, tr, float(np.asarray(cap["dt_s"])),
                    twet, uwet, vwet)
                print(f"  ssh_frc is identically zero in the record: "
                      f"{bool((fro['ssh_frc'] == 0).all())} (no E-P on this "
                      "card), so it cannot carry a residual")
            else:
                tab.unmeasured("R1 frozen forcing zu_frc/zv_frc",
                               "the rank-tagged spg_dump_z[uv]_frc_r*.bin "
                               "streams; this record has none")
            per, first, n_scored, _mismatch = substep_ladder(
                subs, tr, n_loop, twet, uwet, vwet, plant=args.plant)
            _render_substeps(per, first, n_scored)
            if _mismatch:
                tab.rows.append(dict(
                    name=f"R3 substep COUNT {int(n_loop)} vs {len(subs)}",
                    n=len(subs), cells=abs(int(n_loop) - len(subs)),
                    maxabs=float(abs(int(n_loop) - len(subs))), rms=None,
                    floor=None, status="DEBT"))
                tab.fail = True
            elif first is None:
                # NOT a zero compared to itself.  The row carries the real
                # comparison count and the worst residual over every substep
                # and every scored slot, so AT BAR here is a statement about
                # the data.  The first version was
                # score(zeros(1), zeros(1), ones(1)) -- tautological, and an
                # independent review said so.
                _worst = max(per[i][k][1] for i in range(n_scored)
                             for k, _, _ in _ROWS)
                tab.rows.append(dict(
                    name=f"R3 per-substep ladder (x{n_scored})",
                    n=n_scored * len(_ROWS), cells=0, maxabs=_worst,
                    rms=None, floor=None, status="AT BAR"))
            else:
                _sub, _slot = first
                entry_clean = all(
                    per[_sub - 1][k][0] == 0
                    for k in ("eta_entry", "u_entry", "v_entry"))
                tab.rows.append(dict(
                    name=f"R3 substep {_sub} {_slot}",
                    n=int(twet.sum()), cells=per[_sub - 1][_slot][0],
                    maxabs=per[_sub - 1][_slot][1], rms=None, floor=None,
                    status="DEBT"))
                tab.fail = True
                print()
                print(f"  FIRST NON-BIT: substep {_sub}, row {_slot!r}")
                print(f"    NEMO statement: {_ROW_STATEMENT[_slot]}")
                # The headline row is the first in STATEMENT ORDER, which is
                # not the largest: at substep 1 u_exit breaks by 7.6e-21 and
                # v_exit by 9.9e-09.  Reporting only the first would make the
                # break look thirteen orders smaller than it is, so every row
                # that breaks at that substep is listed, largest first.
                _also = sorted(
                    ((per[_sub - 1][k][1], k) for k, _, _ in _ROWS
                     if per[_sub - 1][k][0]), reverse=True)
                print(f"    every row that breaks at substep {_sub}, "
                      "largest first:")
                for _d, _k in _also:
                    print(f"      {_k:11s} {per[_sub - 1][_k][0]:6d} cells  "
                          f"max|d| {_d:.4e}   {_ROW_STATEMENT[_k]}")
                print("    the entry rows are the STATE operands only; the "
                      "frozen forcing (R1) and the face-column depths (R0) "
                      "enter the same statement and carry their own rows "
                      "above -- read the three together")
                print(f"    STATE operands at substep {_sub} are 0 cells: "
                      f"{entry_clean}"
                      + ("  -> this substep's operator ran on NEMO's OWN "
                         "operands, so the break is the operator's, not "
                         "inherited drift.  NOT the same as exonerating the "
                         "loop: from rest at jn=1 the Coriolis, surface-"
                         "pressure-gradient and bottom-stress terms are "
                         "multiplied by zero, so they are UNTESTED here, and "
                         "once jn=1 breaks no later substep is a given-"
                         "inputs comparison either"
                         if entry_clean else
                         "  -> the operands were ALREADY different entering "
                         "this substep, so this row is inherited; read the "
                         "first broken ENTRY row above instead"))
    # R4 -- the loop exit, from race-free restart fields
    floor_u = float(np.sqrt(np.mean((R["ubtacc_aa"][uwet]) ** 2)))
    floor_v = float(np.sqrt(np.mean((R["vbtacc_aa"][vwet]) ** 2)))
    floor_e = float(np.sqrt(np.mean((R["sshn"][twet]) ** 2)))
    tab.score("R4a barotropic U avg  puu_b(Kaa)",
              u_avg[:, 1:] * R1_DT, R["ubtacc_aa"], uwet, floor=floor_u)
    tab.score("R4a barotropic V avg  pvv_b(Kaa)",
              v_avg[1:, :] * R1_DT, R["vbtacc_aa"], vwet, floor=floor_v)
    tab.score("R4b barotropic eta avg pssh(Kaa)",
              cap["eta_out"], R["sshn"], twet, floor=floor_e)
    print()
    tab.render()
    print()
    print("  'NEMO step' is the rms of NEMO's own answer for that field, so a "
          "residual is read as a fraction of the motion it belongs to.")

    if args.save_trace:
        os.makedirs(os.path.dirname(args.save_trace) or ".", exist_ok=True)
        np.savez_compressed(
            args.save_trace,
            w_filter=w_filter, w_total=w_total,
            dt_s=float(np.asarray(cap["dt_s"])), n_loop=n_loop,
            **{k: v for k, v in tr.items()})
        print(f"  legoESM per-substep trajectory -> {args.save_trace}")

    if tab.fail:
        print("\nGATE FAIL: at least one rung is off the bar (0 cells "
              "unequal).")
        return 1
    print("\nGATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
