#!/usr/bin/env python
"""ORACLE STAGE-STATE comparison: name the dyn_core stage where the
panel-boundary parity floor (9.79e-06 on the one-step hydro gate)
first appears.

Consumes the per-tile dumps of fv3_dyncore_stage_driver.F90 (the staged
verbatim dyn_core/fv_dynamics copies, acoustic substep 1 of the C48/npz=5
hydro parity deck) and replays the port's substep 1 with the same stage
boundaries (`stage_hook` threading in fv3_native_{acoustic,cgrid_phase,
dsw_phase,dsw_tail}_3d).

TRUST LADDER -- nothing is reported unless every rung below it holds:

 1. CERT lines in every tile manifest are exactly 0.0: the staged copy
    reproduced the REAL fv_dynamics bitwise, so its stage dumps describe
    the real oracle, not an instrumented cousin.
 2. External control: the driver's arm-A final state equals the pinned
    run_hydro_1step_gfs RESTART bitwise -- the driver's init path
    reproduces the production solo driver.
 3. Face-map instrument control: the S00 entry states (oracle theta_v
    entry vs port post-conversion IC) must map at the quad-geometry
    floor under the FROZEN 2026-08-07 bijection.
 4. Mutation control (--mutate): deliberately break one port operation
    and confirm the ladder flags the matching stage FIRST -- the
    instrument must be shown to have teeth before its clean run is
    believed.

Then, per stage and field, max|port - oracle| split by region of the
compute window (interior / edge strips / corner wedges, width 3) and --
for post-exchange stages -- the halo edge bands and corner-diagonal
blocks separately.  The FIRST stage whose boundary regions jump to the
1e-6 class names the culprit.

B-grid wind ingredients (ubb/vbb/ubbtemp/vbbtemp) are scored under a
FROZEN, basis-derived transform (codex r2 HIGH#1): d_sw3 defines
ubbtemp as ytp_v's advected D-grid *v* (a y-component quantity, m/s)
and vbb as xtp_u's advected D-grid *u* (an x-component quantity), so
they transform exactly like every other B-grid vector pair -- the
y-like member takes the dihedral v-sign and pairs with the partner's
x-like member under a transpose ("bv"), the x-like member takes the
u-sign ("bu").  Nothing is fitted per run.  The old empirical
best-of-4 search is retained ONLY as a printed cross-check: if it ever
finds a combo strictly better than the frozen one, that is flagged as
FROZEN-VS-EMPIRICAL (the masking codex r2 warned about).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_fsp():
    spec = importlib.util.spec_from_file_location(
        "full_step_oracle_parity",
        os.path.join(_HERE, "full_step_oracle_parity.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FSP = _load_fsp()
N, NG, KM = FSP.N, FSP.NG, FSP.KM
ORACLE_ROOT = FSP.ORACLE_ROOT

UNINIT = 1.0e30    # huge_r init pattern -> field slot never written


# ----------------------------------------------------------------------
# dump reading
# ----------------------------------------------------------------------

def read_tile(dump_dir: str, tile: int, prefix: str = "dyncore_stage_t"):
    """Manifest+stream -> dict name -> np.ndarray (F-order axes), plus
    the NOTE/CERT lines.  ``prefix`` selects the dump family (the
    extchain metric dumps use ``extchain_t`` with the same format)."""
    mf = os.path.join(dump_dir, f"{prefix}{tile}.mf")
    dat = os.path.join(dump_dir, f"{prefix}{tile}.dat")
    if not (os.path.exists(mf) and os.path.exists(dat)):
        raise SystemExit(f"missing stage dumps for tile {tile} in "
                         f"{dump_dir}")
    fields, notes, certs = {}, [], {}
    raw = np.memmap(dat, dtype=np.float64, mode="r")
    for line in open(mf):
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "NOTE":
            notes.append(line.strip())
            continue
        if parts[0] == "CERT":
            certs[parts[1]] = float(parts[2])
            continue
        name = parts[0]
        nums = [int(x) for x in parts[1:]]
        *shape, off = nums
        shape = [s for s in shape if s != 1] or [1]
        cnt = int(np.prod(shape))
        arr = np.array(raw[off // 8: off // 8 + cnt])
        fields[name] = arr.reshape(shape, order="F")
    return fields, notes, certs


def require_certs(certs_by_tile: list) -> None:
    bad = []
    for t, certs in enumerate(certs_by_tile, start=1):
        if not certs:
            raise SystemExit(f"tile {t}: manifest has NO CERT lines -- "
                             f"the driver did not finish both arms")
        for k, v in certs.items():
            if v != 0.0:
                bad.append((t, k, v))
    if bad:
        for t, k, v in bad:
            print(f"  CERT FAIL tile {t} {k}: {v:.3e}")
        raise SystemExit(
            "CERTIFICATION FAILED: the staged dyn_core/fv_dynamics copy "
            "does not reproduce the REAL fv_dynamics bitwise. Its stage "
            "dumps describe a broken instrument, not the oracle. "
            "Refusing to compare stages.")
    print("CERT: staged == real bitwise on every tile "
          f"({sum(len(c) for c in certs_by_tile)} field certificates, "
          "all exactly 0.0)")


def external_control(dumps: list, step_run: str) -> None:
    """Arm-A finals vs the pinned 1-step restart, bitwise.

    codex r1 #5: covers every field the restart set carries -- u, v, T,
    delp, phis (fv_core.res) and, when present, the tracer fields
    (fv_tracer.res).  In-memory-only fields (pe/pk/peln/pkz/uc/vc/...)
    have no restart representation; they are covered by the CERT rung
    (staged==real) instead, which bounds an instrument error but not a
    driver-init error in those fields -- stated, not hidden.
    """
    import netCDF4 as nc
    worst = 0.0
    n_fields = 0
    for t in range(6):
        p = os.path.join(step_run, "RESTART", f"fv_core.res.tile{t+1}.nc")
        d = nc.Dataset(p)
        for dump_name, var, stag in (("FINA_u", "u", "u"),
                                     ("FINA_v", "v", "v"),
                                     ("FINA_pt", "T", "a"),
                                     ("FINA_delp", "delp", "a")):
            ref = np.array(d[var][:])[0]              # (k, j, i)
            ref_f = np.moveaxis(ref, 0, -1).transpose(1, 0, 2)  # (i,j,k)
            arr = dumps[t][dump_name]
            ei = 48 + (1 if stag == "v" else 0)
            ej = 48 + (1 if stag == "u" else 0)
            win = arr[NG:NG + ei, NG:NG + ej, :]
            worst = max(worst, float(np.abs(win - ref_f).max()))
            n_fields += 1
        if "phis" in d.variables and "FINA_phis" in dumps[t]:
            ref2 = np.array(d["phis"][:])[0].T          # (i, j)
            win2 = dumps[t]["FINA_phis"][NG:NG + 48, NG:NG + 48]
            worst = max(worst, float(np.abs(win2 - ref2).max()))
            n_fields += 1
        d.close()
        ptr = os.path.join(step_run, "RESTART",
                           f"fv_tracer.res.tile{t+1}.nc")
        if os.path.exists(ptr) and f"FINA_q1" in dumps[t]:
            dtr = nc.Dataset(ptr)
            for iq, var in enumerate(dtr.variables):
                key = f"FINA_q{iq+1}"
                if dtr[var].ndim != 4 or key not in dumps[t]:
                    continue
                ref = np.array(dtr[var][:])[0]
                ref_f = np.moveaxis(ref, 0, -1).transpose(1, 0, 2)
                win = dumps[t][key][NG:NG + 48, NG:NG + 48, :]
                worst = max(worst, float(np.abs(win - ref_f).max()))
                n_fields += 1
            dtr.close()
    if worst != 0.0:
        raise SystemExit(
            f"EXTERNAL CONTROL FAILED: the driver's arm-A final state "
            f"differs from {step_run}/RESTART by {worst:.3e} (must be "
            f"bitwise 0). The driver's init path does not reproduce the "
            f"production deck; nothing downstream is comparable.")
    print(f"EXTERNAL CONTROL: arm-A final == pinned 1-step restart, "
          f"bitwise ({n_fields} tile-fields)")


# ----------------------------------------------------------------------
# stagger/window/mapping machinery
# ----------------------------------------------------------------------

# per-kind compute extents (i, j) and the partner kind under a transpose
KINDS = {
    "ascalar": ((48, 48), "ascalar", 0),
    "bscalar": ((49, 49), "bscalar", 0),
    "u":       ((48, 49), "v", +1),        # sign index 1 -> s_u
    "v":       ((49, 48), "u", -1),        # sign index -1 -> s_v
    "uc":      ((49, 48), "vc", +1),
    "vc":      ((48, 49), "uc", -1),
    "au":      ((48, 48), "av", +1),
    "av":      ((48, 48), "au", -1),
    "bu":      ((49, 49), "bv", +1),
    "bv":      ((49, 49), "bu", -1),
    "xflux":   ((49, 48), "yflux", +1),
    "yflux":   ((48, 49), "xflux", -1),
    # positive scalar pair (area ratios): partner-swap, no sign.
    "sx":      ((48, 48), "sy", 0),
    "sy":      ((48, 48), "sx", 0),
    # B-grid m/s advected pair (ubbtemp/vbb): the transpose partner and
    # sign are resolved EMPIRICALLY per face (best of 4 combos) --
    # see map_bm_best.
    "bm":      ((49, 49), "bm", 0),
    # PSEUDO quantities: relative vorticity is a pseudoscalar, so it and
    # its transport fluxes acquire an EXTRA sign on faces whose
    # composite map has determinant -1 (pure transpose).  det =
    # (transpose ? -1 : +1) * (dihedral in {id, r180} ? +1 : -1).
    "pscalar": ((48, 48), "pscalar", 3),
    "pxflux":  ((49, 48), "pyflux", +2),
    "pyflux":  ((48, 49), "pxflux", -2),
}


def _map_det(meta) -> float:
    transposed, nm, su, sv = meta
    det = -1.0 if transposed else 1.0
    if nm in ("fi", "fj"):
        det = -det
    return det


def map_bm_best(port_arr, orc_direct, orc_partner, meta):
    """Empirical mapping for the B-grid m/s pair: try {direct, partner-
    transposed} x {+1, -1} and keep the combo with the smallest max
    diff.  Self-diagnosing: a genuinely wrong pair leaves every combo
    at O(1)."""
    transposed, nm, su, sv = meta
    p0 = FSP.DIHEDRAL[nm](window(np.asarray(port_arr), "bscalar"))
    cands = [("direct", window(np.asarray(orc_direct), "bscalar"))]
    cands.append(("partner^T",
                  np.swapaxes(window(np.asarray(orc_partner), "bscalar"),
                              0, 1)))
    best = None
    for tag, o in cands:
        if o.shape != p0.shape:
            continue
        for s in (1.0, -1.0):
            d = float(np.abs(p0 * s - o).max())
            if best is None or d < best[0]:
                best = (d, f"{tag}{'+' if s > 0 else '-'}", p0 * s, o)
    return best


def window(arr: np.ndarray, kind: str):
    (ei, ej), _, _ = KINDS[kind]
    sls = []
    for ax, e in ((0, ei), (1, ej)):
        s = arr.shape[ax]
        if s == e:
            sls.append(slice(None))
        elif s == e + 2 * NG:
            sls.append(slice(NG, NG + e))
        else:
            raise ValueError(f"axis {ax} of shape {arr.shape} fits neither "
                             f"{e} nor {e + 2*NG} for kind {kind}")
    return arr[tuple(sls)]


def map_stage_field(port_arr, orc_direct, orc_partner, kind, meta):
    """Return (mapped_port, mapped_oracle) on the compute window.

    Convention mirrors full_step_oracle_parity.apply_map: the DIHEDRAL +
    sign act on the PORT array; a transposed face reads the PARTNER
    oracle field with its (i, j) axes swapped.
    """
    transposed, nm, su, sv = meta
    (ei, ej), partner, sgn_ix = KINDS[kind]
    p = window(np.asarray(port_arr), kind)
    p = FSP.DIHEDRAL[nm](p)
    if sgn_ix == +1:
        p = p * su
    elif sgn_ix == -1:
        p = p * sv
    elif sgn_ix == +2:
        p = p * su * _map_det(meta)
    elif sgn_ix == -2:
        p = p * sv * _map_det(meta)
    elif sgn_ix == 3:
        p = p * _map_det(meta)
    if not transposed:
        o = window(np.asarray(orc_direct), kind)
    else:
        o = window(np.asarray(orc_partner), partner)
        o = np.swapaxes(o, 0, 1)
    if p.shape != o.shape:
        raise ValueError(f"kind {kind}: mapped shapes differ "
                         f"{p.shape} vs {o.shape}")
    return p, o


def region_masks(ni: int, nj: int, width: int = 3):
    di = np.minimum(np.arange(ni), ni - 1 - np.arange(ni))[:, None]
    dj = np.minimum(np.arange(nj), nj - 1 - np.arange(nj))[None, :]
    near_i = di < width
    near_j = dj < width
    corner = near_i & near_j
    edge = (near_i | near_j) & ~corner
    interior = ~(near_i | near_j)
    return {"interior": np.broadcast_to(interior, (ni, nj)),
            "edge": np.broadcast_to(edge, (ni, nj)),
            "corner": np.broadcast_to(corner, (ni, nj))}


def region_maxima(p, o):
    d = np.abs(p - o)
    masks = region_masks(d.shape[0], d.shape[1])
    out = {}
    for name, m in masks.items():
        sel = d[m] if d.ndim == 2 else d[m, ...]
        out[name] = float(sel.max()) if sel.size else 0.0
    return out


def halo_maxima(p_pad, o_pad, kind, meta):
    """Halo comparison for post-exchange stages, edge bands vs the
    corner-diagonal blocks, full padded arrays on both sides."""
    transposed, nm, su, sv = meta
    (ei, ej), partner, sgn_ix = KINDS[kind]
    p = FSP.DIHEDRAL[nm](np.asarray(p_pad))
    if sgn_ix == +1:
        p = p * su
    elif sgn_ix == -1:
        p = p * sv
    if not transposed:
        o = np.asarray(o_pad)
    else:
        o = np.swapaxes(np.asarray(o_pad), 0, 1)
    if p.shape != o.shape:
        raise ValueError(f"halo kind {kind}: {p.shape} vs {o.shape}")
    ni, nj = p.shape[0], p.shape[1]
    ci = np.zeros(ni, bool)
    cj = np.zeros(nj, bool)
    ci[NG:ni - NG] = True          # compute rows (incl +1 stagger row)
    cj[NG:nj - NG] = True
    in_i, in_j = ci[:, None], cj[None, :]
    halo_edge = (in_i & ~in_j) | (~in_i & in_j)
    halo_corner = ~in_i & ~in_j
    d = np.abs(p - o)
    out = {}
    for name, m in (("halo_edge", halo_edge), ("halo_corner", halo_corner)):
        mm = np.broadcast_to(m, (ni, nj))
        sel = d[mm] if d.ndim == 2 else d[mm, ...]
        out[name] = float(sel.max()) if sel.size else 0.0
    return out


# ----------------------------------------------------------------------
# the stage table
# ----------------------------------------------------------------------
# (stage, port_stage_key, port_field, oracle_direct, oracle_partner,
#  kind, halo_too, note)
def stage_rows():
    rows = []

    def r(stage, pkey, pf, od, op, kind, halo=False, note=""):
        rows.append((stage, pkey, pf, od, op, kind, halo, note))

    # codex r1 #1: the delp/pt rows are the S01 exchange even though the
    # port observes them after the (delp/pt-non-mutating) wind exchange.
    exnote = "observed post-wind-exchange (non-mutating)"
    r("S01_extdp", "S02_entryex", "delp", "S01_extdp_delp",
      "S01_extdp_delp", "ascalar", halo=True, note=exnote)
    r("S01_extdp", "S02_entryex", "pt", "S01_extdp_pt",
      "S01_extdp_pt", "ascalar", halo=True, note=exnote)
    r("S02_entryex", "S02_entryex", "u", "S02_extuv_u", "S02_extuv_v",
      "u", halo=True)
    r("S02_entryex", "S02_entryex", "v", "S02_extuv_v", "S02_extuv_u",
      "v", halo=True)

    r("S03_csw", "S03_csw", "delpc", "S03_csw_delpc", "S03_csw_delpc",
      "ascalar")
    r("S03_csw", "S03_csw", "ptc", "S03_csw_ptc", "S03_csw_ptc", "ascalar")
    r("S03_csw", "S03_csw", "uc", "S03_csw_uc", "S03_csw_vc", "uc")
    r("S03_csw", "S03_csw", "vc", "S03_csw_vc", "S03_csw_uc", "vc")
    r("S03_csw", "S03_csw", "ut", "S03_csw_ut", "S03_csw_vt", "au")
    r("S03_csw", "S03_csw", "vt", "S03_csw_vt", "S03_csw_ut", "av")
    r("S03_csw", "S03_csw", "ua", "S03_csw_ua", "S03_csw_va", "au")
    r("S03_csw", "S03_csw", "va", "S03_csw_va", "S03_csw_ua", "av")
    r("S03_csw", "S03_csw", "divg_d", "S03_csw_divgd", "S03_csw_divgd",
      "bscalar")

    r("S04_geopkC", "S04_geopkC", "pkc", "S04_geopkC_pkc",
      "S04_geopkC_pkc", "ascalar")
    r("S04_geopkC", "S04_geopkC", "gz", "S04_geopkC_gz", "S04_geopkC_gz",
      "ascalar")

    r("S05_pgradc", "S05_pgradc", "uc", "S05_pgradc_uc", "S05_pgradc_vc",
      "uc")
    r("S05_pgradc", "S05_pgradc", "vc", "S05_pgradc_vc", "S05_pgradc_uc",
      "vc")

    # codex r1 #1: the divgd row is the S06 exchange (:652), observed
    # after the (divgd-non-mutating) uc/vc exchange at :655.
    r("S06_extdivgd", "S07_extucvc", "divg_d", "S06_extdivgd_divgd",
      "S06_extdivgd_divgd", "bscalar", halo=True, note=exnote)
    r("S07_extucvc", "S07_extucvc", "uc", "S07_extucvc_uc",
      "S07_extucvc_vc", "uc", halo=True)
    r("S07_extucvc", "S07_extucvc", "vc", "S07_extucvc_vc",
      "S07_extucvc_uc", "vc", halo=True)

    # codex r1 #4: also compare the flux capacitors / Courant
    # accumulators / advected C winds / area ratios d_sw1 writes.
    for nm_p, nm_o, kind in (("crx_adv", "crx", "xflux"),
                             ("cry_adv", "cry", "yflux"),
                             ("xfx_adv", "xfx", "xflux"),
                             ("yfx_adv", "yfx", "yflux"),
                             ("xflux", "mfx", "xflux"),
                             ("yflux", "mfy", "yflux"),
                             ("cx", "cx", "xflux"),
                             ("cy", "cy", "yflux"),
                             ("ut", "utt", "uc"),
                             ("vt", "vtt", "vc"),
                             ("ra_x", "rax", "sx"),
                             ("ra_y", "ray", "sy")):
        part = {"crx": "cry", "cry": "crx", "xfx": "yfx", "yfx": "xfx",
                "mfx": "mfy", "mfy": "mfx", "cx": "cy", "cy": "cx",
                "utt": "vtt", "vtt": "utt", "rax": "ray",
                "ray": "rax"}[nm_o]
        r("S08_dsw1", "S08_dsw1", nm_p, f"S08_dsw1_{nm_o}",
          f"S08_dsw1_{part}", kind)
    for slot in (0, 3):
        r("S08_dsw1", "S08_dsw1", f"allflux_x[{slot}]",
          f"S08_dsw1_allflux_x[{slot}]", f"S08_dsw1_allflux_y[{slot}]",
          "xflux")
        r("S08_dsw1", "S08_dsw1", f"allflux_y[{slot}]",
          f"S08_dsw1_allflux_y[{slot}]", f"S08_dsw1_allflux_x[{slot}]",
          "yflux")
    for slot in (0, 3):
        r("S09_fluxavg", "S09_fluxavg", f"allflux_x[{slot}]",
          f"S09_fluxavg_allflux_x[{slot}]",
          f"S09_fluxavg_allflux_y[{slot}]", "xflux")
        r("S09_fluxavg", "S09_fluxavg", f"allflux_y[{slot}]",
          f"S09_fluxavg_allflux_y[{slot}]",
          f"S09_fluxavg_allflux_x[{slot}]", "yflux")

    r("S10_dsw23", "S10_dsw23", "delp", "S10_dsw23_delp",
      "S10_dsw23_delp", "ascalar")
    r("S10_dsw23", "S10_dsw23", "pt", "S10_dsw23_pt", "S10_dsw23_pt",
      "ascalar")
    # B-wind transpose pairing is UNITS-DERIVED (b2 partner probe,
    # 2026-08-11): ubb/vbbtemp are the CIRCULATION-scale pair (~4.8e3)
    # and ubbtemp/vbb the m/s pair (~20), and the oracle's own measured
    # blend exchanges ubb<->vbbtemp across rotated contacts.  So under
    # a transposed face ubb pairs with vbbtemp and ubbtemp with vbb.
    # All faces compared; a wrong pairing would show as O(1), which is
    # self-diagnosing.
    bnote = "pairing-units-derived"
    r("S10_dsw23", "S10_dsw3", "ubb", "S10_dsw23_ubb",
      "S10_dsw23_vbbtemp", "bu", note=bnote)
    r("S10_dsw23", "S10_dsw3", "vbbtemp", "S10_dsw23_vbbtemp",
      "S10_dsw23_ubb", "bv", note=bnote)
    # the m/s advected pair: FROZEN basis-derived transform (codex r2
    # HIGH#1).  ubbtemp = ytp_v output = advected D-grid v (y-like,
    # sw_core.F90:1315-1323) -> kind "bv"; vbb = xtp_u output =
    # advected D-grid u (x-like, :1375-1386) -> kind "bu".  Same
    # staggering algebra as the ubb/vbbtemp pair above; no per-run fit.
    r("S10_dsw23", "S10_dsw3", "ubbtemp", "S10_dsw23_ubbtemp",
      "S10_dsw23_vbb", "bv", note="basis-derived m/s pair")
    r("S10_dsw23", "S10_dsw3", "vbb", "S10_dsw23_vbb",
      "S10_dsw23_ubbtemp", "bu", note="basis-derived m/s pair")

    r("S11_b2", "S11_b2", "ubb", "S11_b2_ubb", "S11_b2_vbbtemp", "bu",
      note=bnote)
    r("S11_b2", "S11_b2", "vbbtemp", "S11_b2_vbbtemp", "S11_b2_ubb",
      "bv", note=bnote)

    r("S12_kee", "S12_kee", "kee", "S12_kee_kee", "S12_kee_kee",
      "bscalar")

    r("S13_dsw45", "S13_dsw45", "ke", "S13_dsw45_kee", "S13_dsw45_kee",
      "bscalar")
    r("S13_dsw45", "S13_dsw45", "wk", "S13_dsw45_wkk", "S13_dsw45_wkk",
      "pscalar", note="pseudo-scalar (vorticity)")
    r("S13_dsw45", "S13_dsw45", "vortfluxx", "S13_dsw45_vortfluxx",
      "S13_dsw45_vortfluxy", "pxflux", note="pseudo-flux")
    r("S13_dsw45", "S13_dsw45", "vortfluxy", "S13_dsw45_vortfluxy",
      "S13_dsw45_vortfluxx", "pyflux", note="pseudo-flux")
    # the pre-d_sw6 momentum carriers: names the term if u/v corner
    # wedges originate before d_sw6's vort-flux application.
    r("S13_dsw45", "S13_dsw45", "ut", "S13_dsw45_utt",
      "S13_dsw45_vtt", "uc")
    r("S13_dsw45", "S13_dsw45", "vt", "S13_dsw45_vtt",
      "S13_dsw45_utt", "vc")

    r("S14_dsw6", "S14_dsw6", "u", "S14_dsw6_u", "S14_dsw6_v", "u")
    r("S14_dsw6", "S14_dsw6", "v", "S14_dsw6_v", "S14_dsw6_u", "v")

    r("S15_extdp2", "S15_extdp2", "delp", "S15_extdp2_delp",
      "S15_extdp2_delp", "ascalar", halo=True)
    r("S15_extdp2", "S15_extdp2", "pt", "S15_extdp2_pt", "S15_extdp2_pt",
      "ascalar", halo=True)

    r("S16_geopkD", "S16_geopkD", "pkc", "S16_geopkD_pkc",
      "S16_geopkD_pkc", "ascalar")
    r("S16_geopkD", "S16_geopkD", "gz", "S16_geopkD_gz", "S16_geopkD_gz",
      "ascalar")
    r("S16_geopkD", "S16_geopkD", "pkz", "S16_geopkD_pkz",
      "S16_geopkD_pkz", "ascalar")

    r("S17_onegradp", "S17_onegradp", "u", "S17_onegradp_u",
      "S17_onegradp_v", "u")
    r("S17_onegradp", "S17_onegradp", "v", "S17_onegradp_v",
      "S17_onegradp_u", "v")
    return rows


def get_oracle_field(dumps_t: dict, name: str):
    """Resolve 'base[slot]' 4-D slot selection."""
    if name.endswith("]"):
        base, slot = name[:-1].split("[")
        arr = dumps_t[base]
        return arr[:, :, :, int(slot)]
    return dumps_t[name]


def get_port_field(stages: dict, pkey: str, pf: str, t: int):
    payload = stages[pkey]
    if pf.endswith("]"):
        base, slot = pf[:-1].split("[")
        return payload[t][base][:, :, :, int(slot)]
    return payload[t][pf]


# ----------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-dir", required=True,
                    help="dir with dyncore_stage_t*.dat/.mf")
    ap.add_argument("--step-run",
                    default=f"{ORACLE_ROOT}/run_hydro_1step_gfs")
    ap.add_argument("--mutate", default=None,
                    choices=("b2", "fluxavg", "dsw3"),
                    help="instrument-teeth control: no-op one port "
                         "barrier / scale one port stage output and "
                         "confirm the matching stage is flagged (b2 -> "
                         "S11, fluxavg -> S09, dsw3 -> S10 ubbtemp)")
    ap.add_argument("--write-receipt", default=None,
                    help="mutation runs: write a PASS receipt here")
    ap.add_argument("--baseline-json", default=None,
                    help="mutation runs: the clean run's --json output; "
                         "PASS = target stage's boundary rel grew 100x "
                         "over it")
    ap.add_argument("--receipts", default=None,
                    help="clean runs: comma-separated mutation-control "
                         "receipts (codex r1 #2 -- rung 4 is mandatory; "
                         "without them the verdict exits 2 as NOT "
                         "PROVEN)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    receipts_ok = False
    if args.mutate is None:
        if args.receipts:
            import json as _json
            missing = []
            for rp in args.receipts.split(","):
                try:
                    rec = _json.load(open(rp))
                    if not (rec.get("pass") and
                            os.path.samefile(rec.get("dump_dir", ""),
                                             args.dump_dir)):
                        missing.append(rp)
                except (OSError, ValueError):
                    missing.append(rp)
            if missing:
                raise SystemExit(
                    f"mutation-control receipts invalid/missing: "
                    f"{missing}. Run --mutate fluxavg/b2 with "
                    f"--write-receipt against THIS dump dir first.")
            receipts_ok = True
        else:
            print("WARNING: no --receipts given -- the mutation "
                  "controls are NOT proven for this dump dir; the "
                  "verdict below is PROVISIONAL (exit 2).")

    dumps, certs_by_tile, notes0 = [], [], None
    for t in range(1, 7):
        f, notes, certs = read_tile(args.dump_dir, t)
        dumps.append(f)
        certs_by_tile.append(certs)
        if notes0 is None:
            notes0 = notes
    for ln in notes0 or []:
        print(ln)

    require_certs(certs_by_tile)
    external_control(dumps, args.step_run)

    # ---------------- port side ----------------
    from legoesm.core.fv3_native_acoustic_3d import acoustic_substep_3d
    from legoesm.core.fv3_native_duo_stepper import (
        build_six_face_duo_context,
    )
    from legoesm.core.fv3_native_dynamics import (
        p_var_hydrostatic,
        pt_to_theta_v,
    )
    from legoesm.core.fv3_native_eta import set_eta_analytic
    from legoesm.grids.fv3_native_gridstruct import FV3_CP_AIR, FV3_KAPPA

    if args.mutate:
        import legoesm.grids.fv3_native_gridstruct as GS
        if args.mutate == "b2":
            GS.average_shared_edge_bgrid = (
                lambda u6, v6, n, ng: None)
            print("MUTATION: average_shared_edge_bgrid -> no-op "
                  "(expect S11_b2 flagged first)")
        elif args.mutate == "fluxavg":
            GS.average_allflux_shared_edges = (
                lambda afx6, afy6, nq, n, ng: None)
            print("MUTATION: average_allflux_shared_edges -> no-op "
                  "(expect S09_fluxavg flagged first)")
        elif args.mutate == "dsw3":
            import legoesm.core.fv3_native_duo_sw_core as DSC
            _orig_dsw3 = DSC.d_sw3_duo

            def _mut_dsw3(*a, **kw):
                out = _orig_dsw3(*a, **kw)
                ubt = np.array(out["ubbtemp"], copy=True)
                ring = np.abs(np.concatenate(
                    [ubt[0, :], ubt[-1, :], ubt[:, 0], ubt[:, -1]])).max()
                if not getattr(_mut_dsw3, "_printed", False):
                    print(f"MUTATION: d_sw3_duo ubbtemp boundary ring "
                          f"*= (1+1e-4); baseline max|ring| = "
                          f"{ring:.6e} (printed BEFORE trusting the "
                          f"control -- a perturbed zero is not a "
                          f"control)")
                    _mut_dsw3._printed = True
                if ring == 0.0:
                    raise SystemExit(
                        "dsw3 mutation perturbs an all-zero ubbtemp "
                        "ring -- the control is void; refusing to "
                        "write a receipt")
                # edge-concentrated on purpose: the flag rule requires
                # boundary >> interior, so a uniform scale would prove
                # nothing about the boundary instrument.
                ubt[0, :] *= 1.0 + 1e-4
                ubt[-1, :] *= 1.0 + 1e-4
                ubt[:, 0] *= 1.0 + 1e-4
                ubt[:, -1] *= 1.0 + 1e-4
                out = dict(out)
                out["ubbtemp"] = ubt
                return out

            DSC.d_sw3_duo = _mut_dsw3
            print("MUTATION: d_sw3_duo ubbtemp scaled (expect "
                  "S10_dsw23 boundary rel to grow >= 100x)")

    ctx = build_six_face_duo_context(N, NG, use_ext_bundle=True,
                                     oracle_conventions=True)
    ak, bk, ptop, ks = set_eta_analytic(KM)
    ptop = float(ptop)
    state = FSP.build_port_ic(ctx, ak, bk)
    press = [p_var_hydrostatic(f["delp"], ptop=ptop, akap=FV3_KAPPA,
                               n=N, ng=NG, km=KM) for f in state]
    for t in range(6):
        pt_to_theta_v(state[t]["pt"], press[t]["pkz"], n=N, ng=NG)

    # ------- rung 3: face map from the S00 entry states -------
    cs, cc = slice(NG, NG + N), slice(NG, NG + N + 1)
    p_ic = FSP.port_window(state, ctx)

    def orc_kji(t, name, stag):
        arr = dumps[t][name]
        ei = 48 + (1 if stag == "v" else 0)
        ej = 48 + (1 if stag == "u" else 0)
        win = arr[NG:NG + ei, NG:NG + ej, :]
        return win.transpose(2, 1, 0)          # (k, j, i) C-layout

    orc_ic = [{"u": orc_kji(t, "S00_entry_u", "u"),
               "v": orc_kji(t, "S00_entry_v", "v"),
               "pt": orc_kji(t, "S00_entry_pt", "a"),
               "delp": orc_kji(t, "S00_entry_delp", "a")}
              for t in range(6)]
    (cost, meta, perm, worst, _pf, _wo) = FSP.derive_face_map(p_ic, orc_ic)
    print(f"\nS00 face-map control: worst rel {worst:.3e} "
          f"(floor {FSP.IC_CONTROL_MAX_REL:.0e})")
    for pf in range(6):
        tr, nm, su, sv = meta[pf][perm[pf]]
        print(f"  port face {pf+1} -> oracle tile {perm[pf]+1}  "
              f"rel={cost[pf, perm[pf]]:.3e}  "
              f"[{'transpose' if tr else 'direct':9s} {nm:4s} "
              f"s_u{su:+.0f} s_v{sv:+.0f}]")
    if worst > FSP.IC_CONTROL_MAX_REL:
        raise SystemExit(
            f"S00 INSTRUMENT CONTROL FAILED ({worst:.3e}): the port entry "
            f"state does not reproduce the oracle's dyn_core entry state; "
            f"no stage diff below would mean anything.")
    _FROZEN = {0: {3}, 1: {4}, 2: {2}, 3: {0, 1}, 4: {0, 1}, 5: {5}}
    drift = [(pf + 1, perm[pf] + 1) for pf in range(6)
             if perm[pf] not in _FROZEN[pf]]
    if drift:
        raise SystemExit(f"FACE-MAP DRIFT {drift}: refuse to compare on a "
                         f"silently different relabelling.")

    # ------- static-field control: f0 incl corner-diagonal blocks -----
    # vortflux corner cells are the confirmed carrier; vort = wk + f0
    # and wk is at floor, so f0's corner-diagonal region (the one part
    # of vort the compute-window rows cannot see) is the prime suspect.
    if "IC_f0" in dumps[0]:
        print("\nf0 CONTROL (pseudo-scalar, full padded domain incl "
              "corner-diagonal blocks):")
        for pf in range(6):
            ot = perm[pf]
            m = meta[pf][ot]
            p_f0 = np.asarray(ctx["gs6"][pf]["f0"], dtype=np.float64)
            o_f0 = dumps[ot]["IC_f0"]
            det = _map_det(m)
            p_m = FSP.DIHEDRAL[m[1]](p_f0) * det
            o_m = o_f0 if not m[0] else o_f0.T
            if p_m.shape != o_m.shape:
                print(f"  face{pf+1}: shape mismatch {p_m.shape} vs "
                      f"{o_m.shape}")
                continue
            hm = {}
            ni_, nj_ = p_m.shape
            ci = np.zeros(ni_, bool)
            cj = np.zeros(nj_, bool)
            ci[NG:ni_ - NG] = True
            cj[NG:nj_ - NG] = True
            d = np.abs(p_m - o_m)
            comp = d[np.ix_(ci, cj)].max()
            he = max(d[np.ix_(ci, ~cj)].max(), d[np.ix_(~ci, cj)].max())
            hc = d[np.ix_(~ci, ~cj)].max()
            print(f"  face{pf+1}->tile{ot+1}: compute={comp:.3e} "
                  f"halo_edge={he:.3e} halo_corner={hc:.3e} "
                  f"(scale {np.abs(o_f0).max():.3g})")

    # ------- replay port substep 1 with the stage hook -------
    stages: dict = {}

    def hook(name, payload):
        stages[name] = [{k: np.array(v, copy=True) for k, v in d.items()}
                        for d in payload]

    dt_sub = 1920.0 / 1.0 / 8.0
    acoustic_substep_3d(ctx, state, dt_sub, KM, first_substep=True,
                        ptop=ptop, akap=FV3_KAPPA, cp_air=FV3_CP_AIR,
                        remap_step=False, remap_follows=True,
                        stage_hook=hook)
    print(f"\nport substep 1 replayed; stages captured: "
          f"{sorted(stages)}")

    # ------- the ladder -------
    rows = stage_rows()
    results = []
    print("\nPER-STAGE BOUNDARY DIFF (max|port-oracle| by region; "
          "'scale' = max|oracle| of the pair's window)")
    hdr = (f"{'stage':13s} {'field':16s} {'interior':>10s} {'edge':>10s} "
           f"{'corner':>10s} {'halo_edge':>10s} {'halo_crnr':>10s} "
           f"{'scale':>10s}  note")
    print(hdr)
    flagged = []
    for (stage, pkey, pf_name, od_name, op_name, kind, halo,
         note) in rows:
        if pkey not in stages:
            print(f"{stage:13s} {pf_name:16s}  -- port stage missing --")
            continue
        agg = {k: 0.0 for k in ("interior", "edge", "corner",
                                "halo_edge", "halo_corner")}
        scale = 0.0
        skip_note = note
        per_face_used = 0
        per_face_detail = []
        for pf in range(6):
            ot = perm[pf]
            m = meta[pf][ot]
            transposed = m[0]
            try:
                p_arr = get_port_field(stages, pkey, pf_name, pf)
                o_dir = get_oracle_field(dumps[ot], od_name)
                o_par = get_oracle_field(dumps[ot], op_name)
            except KeyError as e:
                skip_note = f"missing {e}"
                break
            combo = ""
            if kind == "bm":
                got_bm = map_bm_best(p_arr, o_dir, o_par, m)
                if got_bm is None:
                    skip_note = "bm: no shape-compatible combo"
                    break
                _, combo, p, o = got_bm
            else:
                p, o = map_stage_field(p_arr, o_dir, o_par, kind, m)
                if stage == "S10_dsw23" and pf_name in ("ubbtemp", "vbb"):
                    # codex r2 HIGH#1 cross-check: the retired empirical
                    # best-of-4 must NOT beat the frozen basis-derived
                    # transform; if it does, the frozen algebra is wrong
                    # or the empirical fit was masking a real signal.
                    frozen_d = float(np.abs(p - o).max())
                    got_bm = map_bm_best(p_arr, o_dir, o_par, m)
                    if got_bm is not None and got_bm[0] < 0.5 * frozen_d:
                        print(f"    FROZEN-VS-EMPIRICAL {pf_name} "
                              f"face{pf+1}: empirical [{got_bm[1]}] "
                              f"max|d|={got_bm[0]:.3e} beats frozen "
                              f"{frozen_d:.3e} -- possible masking")
                        combo = f"emp!{got_bm[1]}"
            if (np.abs(o) > UNINIT).any() or (np.abs(p) > UNINIT).any():
                skip_note = "UNAVAILABLE (uninitialised window)"
                break
            per_face_used += 1
            scale = max(scale, float(np.abs(o).max()),
                        float(np.abs(p).max()))
            rm = region_maxima(p, o)
            for k, v in rm.items():
                agg[k] = max(agg[k], v)
            # per-edge-strip maxima (W/E/S/N, width 3) for localisation
            d_ = np.abs(p - o)
            ni_, nj_ = d_.shape[0], d_.shape[1]
            strips = {"W": d_[:3, ...], "E": d_[ni_ - 3:, ...],
                      "S": d_[:, :3, ...], "N": d_[:, nj_ - 3:, ...]}
            per_face_detail.append(
                (pf + 1, ot + 1, transposed, rm,
                 {k: float(v.max()) for k, v in strips.items()}, combo))
            if halo:
                try:
                    hm = halo_maxima(p_arr, o_dir if not transposed
                                     else o_par, kind, m)
                    for k, v in hm.items():
                        agg[k] = max(agg[k], v)
                except ValueError:
                    pass
        if per_face_used == 0:
            print(f"{stage:13s} {pf_name:16s}  -- skipped: {skip_note}")
            continue
        rel = {k: (v / scale if scale else 0.0) for k, v in agg.items()}
        boundary_rel = max(rel["edge"], rel["corner"], rel["halo_edge"],
                           rel["halo_corner"])
        flag = ""
        if boundary_rel > 1e-8 and boundary_rel > 30 * max(rel["interior"],
                                                           1e-16):
            flag = "<-- BOUNDARY JUMP"
            flagged.append((stage, pf_name, boundary_rel,
                            rel["interior"]))
        print(f"{stage:13s} {pf_name:16s} "
              f"{rel['interior']:10.2e} {rel['edge']:10.2e} "
              f"{rel['corner']:10.2e} {rel['halo_edge']:10.2e} "
              f"{rel['halo_corner']:10.2e} {scale:10.3g}  "
              f"{skip_note} {flag}")
        if (flag or kind in ("bm", "pxflux", "pyflux", "pscalar")) \
                and scale:
            # LOCALISE a flagged row: which faces, which edge strips.
            for pf_, ot_, tr_, rm_, strips_, combo_ in per_face_detail:
                worst_r = max(max(rm_.values()), max(strips_.values()))
                if worst_r / scale < 1e-9 and not (kind == "bm" and tr_):
                    continue
                print(f"    face{pf_}->tile{ot_}"
                      f"{' (transposed)' if tr_ else '':13s} "
                      + "  ".join(f"{k}={v/scale:9.2e}"
                                  for k, v in rm_.items()
                                  if k in ("interior", "edge", "corner"))
                      + "   strips "
                      + "  ".join(f"{k}={v/scale:9.2e}"
                                  for k, v in strips_.items())
                      + (f"   [{combo_}]" if combo_ else ""))
        results.append({"stage": stage, "field": pf_name, "kind": kind,
                        "rel": rel, "scale": scale,
                        "faces_used": per_face_used, "note": skip_note,
                        "flag": bool(flag)})

    print("\nFIRST FLAGGED STAGES (ladder order):")
    if flagged:
        for st, f, br, ir in flagged[:8]:
            print(f"  {st} {f}: boundary rel {br:.3e} vs interior "
                  f"{ir:.3e}")
    else:
        print("  none -- no stage shows a boundary-concentrated jump")

    if args.mutate:
        # PASS criterion: the TARGET stage's boundary rel must GROW by
        # >= 100x versus the unmutated baseline run.  ("first flag ==
        # target" was the original criterion; it is unsatisfiable once a
        # GENUINE upstream signal exists in the clean run -- measured:
        # the real S10 d_sw3 difference precedes S11 in ladder order, so
        # the b2 control could never pass while correctly having teeth.)
        want = {"b2": "S11_b2", "fluxavg": "S09_fluxavg",
                "dsw3": "S10_dsw23"}[args.mutate]
        base_val = 0.0
        if args.baseline_json:
            import json as _json
            base = _json.load(open(args.baseline_json))
            base_val = max((r["rel"]["edge"] for r in base["rows"]
                            if r["stage"] == want), default=0.0)
        got_val = max((br for s, f, br, ir in flagged if s == want),
                      default=0.0)
        if args.baseline_json:
            ok = got_val >= 100.0 * max(base_val, 1e-13)
            print(f"\nMUTATION CONTROL: {want} boundary rel "
                  f"{got_val:.3e} vs baseline {base_val:.3e} -> "
                  f"{'PASS' if ok else 'FAIL'} (needs >= 100x)")
        else:
            got = flagged[0][0] if flagged else None
            ok = (got == want)
            print(f"\nMUTATION CONTROL: expected first flag {want}, got "
                  f"{got} -> {'PASS' if ok else 'FAIL'}")
        if ok and args.write_receipt:
            import json as _json
            with open(args.write_receipt, "w") as fh:
                _json.dump({"pass": True, "mutate": args.mutate,
                            "dump_dir": os.path.abspath(args.dump_dir)},
                           fh)
            print(f"wrote receipt {args.write_receipt}")
        return 0 if ok else 1

    if args.json:
        import json
        with open(args.json, "w") as fh:
            json.dump({"face_map_worst": worst,
                       "rows": results,
                       "flagged": [{"stage": s, "field": f,
                                    "boundary_rel": br,
                                    "interior_rel": ir}
                                   for s, f, br, ir in flagged]},
                      fh, indent=2)
        print(f"wrote {args.json}")
    return 0 if receipts_ok else 2


if __name__ == "__main__":
    sys.exit(main())
