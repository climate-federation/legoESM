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

B-grid wind ingredients (ubb/vbb/ubbtemp/vbbtemp) have an UNPROVEN
component pairing under transposed faces; their rows are labelled
"direct-faces-authoritative" and only untransposed faces carry a claim.
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

def read_tile(dump_dir: str, tile: int):
    """Manifest+stream -> dict name -> np.ndarray (F-order axes), plus
    the NOTE/CERT lines."""
    mf = os.path.join(dump_dir, f"dyncore_stage_t{tile}.mf")
    dat = os.path.join(dump_dir, f"dyncore_stage_t{tile}.dat")
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
    """Arm-A finals vs the pinned 1-step restart, bitwise."""
    import netCDF4 as nc
    worst = 0.0
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
        d.close()
    if worst != 0.0:
        raise SystemExit(
            f"EXTERNAL CONTROL FAILED: the driver's arm-A final state "
            f"differs from {step_run}/RESTART by {worst:.3e} (must be "
            f"bitwise 0). The driver's init path does not reproduce the "
            f"production deck; nothing downstream is comparable.")
    print("EXTERNAL CONTROL: arm-A final == pinned 1-step restart, "
          "bitwise, all 6 tiles / 4 fields")


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
}


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

    r("S02_entryex", "S02_entryex", "delp", "S01_extdp_delp",
      "S01_extdp_delp", "ascalar", halo=True)
    r("S02_entryex", "S02_entryex", "pt", "S01_extdp_pt",
      "S01_extdp_pt", "ascalar", halo=True)
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

    r("S07_extucvc", "S07_extucvc", "uc", "S07_extucvc_uc",
      "S07_extucvc_vc", "uc", halo=True)
    r("S07_extucvc", "S07_extucvc", "vc", "S07_extucvc_vc",
      "S07_extucvc_uc", "vc", halo=True)
    r("S07_extucvc", "S07_extucvc", "divg_d", "S06_extdivgd_divgd",
      "S06_extdivgd_divgd", "bscalar", halo=True)

    for nm_p, nm_o, kind in (("crx_adv", "crx", "xflux"),
                             ("cry_adv", "cry", "yflux"),
                             ("xfx_adv", "xfx", "xflux"),
                             ("yfx_adv", "yfx", "yflux")):
        part = {"crx": "cry", "cry": "crx",
                "xfx": "yfx", "yfx": "xfx"}[nm_o]
        r("S08_dsw1", "S08_dsw1", nm_p, f"S08_dsw1_{nm_o}",
          f"S08_dsw1_{part}", kind)
    for slot, tag in ((0, "delp"), (3, "temp")):
        r("S08_dsw1", "S08_dsw1", f"allflux_x[{slot}]",
          f"S08_dsw1_allflux_x[{slot}]", f"S08_dsw1_allflux_y[{slot}]",
          "xflux")
        r("S08_dsw1", "S08_dsw1", f"allflux_y[{slot}]",
          f"S08_dsw1_allflux_y[{slot}]", f"S08_dsw1_allflux_x[{slot}]",
          "yflux")
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
    bnote = "direct-faces-authoritative"
    r("S10_dsw23", "S10_dsw3", "ubb", "S10_dsw23_ubb", "S10_dsw23_vbb",
      "bu", note=bnote)
    r("S10_dsw23", "S10_dsw3", "vbb", "S10_dsw23_vbb", "S10_dsw23_ubb",
      "bv", note=bnote)
    r("S10_dsw23", "S10_dsw3", "ubbtemp", "S10_dsw23_ubbtemp",
      "S10_dsw23_vbbtemp", "bu", note=bnote)
    r("S10_dsw23", "S10_dsw3", "vbbtemp", "S10_dsw23_vbbtemp",
      "S10_dsw23_ubbtemp", "bv", note=bnote)

    r("S11_b2", "S11_b2", "ubb", "S11_b2_ubb", "S11_b2_vbbtemp", "bu",
      note=bnote)
    r("S11_b2", "S11_b2", "vbbtemp", "S11_b2_vbbtemp", "S11_b2_ubb",
      "bv", note=bnote)

    r("S12_kee", "S12_kee", "kee", "S12_kee_kee", "S12_kee_kee",
      "bscalar")

    r("S13_dsw45", "S13_dsw45", "ke", "S13_dsw45_kee", "S13_dsw45_kee",
      "bscalar")
    r("S13_dsw45", "S13_dsw45", "wk", "S13_dsw45_wkk", "S13_dsw45_wkk",
      "ascalar")
    r("S13_dsw45", "S13_dsw45", "vortfluxx", "S13_dsw45_vortfluxx",
      "S13_dsw45_vortfluxy", "xflux")
    r("S13_dsw45", "S13_dsw45", "vortfluxy", "S13_dsw45_vortfluxy",
      "S13_dsw45_vortfluxx", "yflux")

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
                    choices=("b2", "fluxavg"),
                    help="instrument-teeth control: no-op one port "
                         "barrier and confirm the matching stage is "
                         "flagged first (b2 -> S11, fluxavg -> S09)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

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
        for pf in range(6):
            ot = perm[pf]
            m = meta[pf][ot]
            transposed = m[0]
            if note == "direct-faces-authoritative" and transposed:
                continue
            try:
                p_arr = get_port_field(stages, pkey, pf_name, pf)
                o_dir = get_oracle_field(dumps[ot], od_name)
                o_par = get_oracle_field(dumps[ot], op_name)
            except KeyError as e:
                skip_note = f"missing {e}"
                break
            p, o = map_stage_field(p_arr, o_dir, o_par, kind, m)
            if (np.abs(o) > UNINIT).any() or (np.abs(p) > UNINIT).any():
                skip_note = "UNAVAILABLE (uninitialised window)"
                break
            per_face_used += 1
            scale = max(scale, float(np.abs(o).max()),
                        float(np.abs(p).max()))
            for k, v in region_maxima(p, o).items():
                agg[k] = max(agg[k], v)
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
        want = {"b2": "S11_b2", "fluxavg": "S09_fluxavg"}[args.mutate]
        got = flagged[0][0] if flagged else None
        ok = (got == want)
        print(f"\nMUTATION CONTROL: expected first flag {want}, got "
              f"{got} -> {'PASS' if ok else 'FAIL'}")
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
