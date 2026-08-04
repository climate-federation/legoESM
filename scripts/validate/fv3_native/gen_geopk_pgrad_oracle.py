#!/usr/bin/env python
"""Gen the multilevel geopk -> p_grad_c -> geopk -> one_grad_p oracle
inputs, or pack the fixture.

Brick ``geopk_pgrad``: the first genuine 3-D boundary crossing of the
FV3-duo campaign.  ``geopk`` is the only column-recursive kernel in the
acoustic loop, and ``p_grad_c``/``one_grad_p`` are the only consumers of
interface-level ``pk``/``gz``.

SCOPE: ROUTINE-TRANSLATION CERTIFICATE on ONE face.  Every inter-stage
and cross-face quantity arrives as EXPLICIT serialized input — this
certifies neither the six-face exchange cadence nor the ``ext_scalar``
schedule.  Build lane: NO ``-DSW_DYNAMICS``, NO ``-DUSE_COND``,
hydrostatic, ``beta<=0``, ``a2b_ord=4``, duogrid, ``d_ext>0``.

Two modes (no regeneration on pack — the fixture inputs, the hash and
the Fortran output all come from ONE generation, so "bit-exact on
identical inputs" is actually enforced):

  gen_geopk_pgrad_oracle.py <W> --km 2         build inputs + smoke-run the
                                               python port; write
                                               geopk_pgrad_input.txt (the
                                               Fortran driver reads it) +
                                               staging_km2.npz.
  gen_geopk_pgrad_oracle.py <W> --km 2 --pack  read the SAME staging npz +
                                               the first-run input.txt (hash
                                               it, reject on drift) + the
                                               driver's
                                               geopk_pgrad_output_km2.txt,
                                               and write the fixture npz.

The byte layout of ``geopk_pgrad_input.txt`` is produced by ONE canonical
writer, :func:`serialize_geopk_pgrad_inputs`, which the oracle test
imports and re-runs on the fixture's STORED arrays to prove
``input_sha256`` is the hash of exactly those bytes (provenance
ENFORCED, not merely recorded).  Importing this module has no side
effects.
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np
from legoesm import constants

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "..", "..", "..")

RES, NG = 12, 3
NPROBE = 64

# ---------------------------------------------------------------------
# Configuration scalars.  These live in the input HEADER, so the Fortran
# oracle and the python port read the SAME number and the certificate is
# INSENSITIVE to their true production values.
#
# UNCERTAIN U1: dyn_core.F90:24 imports rdgas/radius/cp_air/pi from FMS
# ``constants_mod`` and NO constants*.F90 exists anywhere under the
# Zenodo 8327578 symmetryclean tree, so the production cp_air/akap are
# NOT verifiable from the sources.  ``AKAP = 2/7`` is the FMS-convention
# TEST value; note that it is deliberately NOT
# ``legoesm.constants.kappa`` (this repo's R_d/c_pd ratio differs from
# 2/7 in the 5th decimal), i.e. the repo's thermodynamic pair is not the
# FMS pair and silently substituting it would change what the
# certificate is about.
# UNCERTAIN U6: PTOP/DT/DT2/D_EXT are DISCRIMINATION choices, not the
# Zenodo duo run's namelist values (which were not read).  Re-pin and
# regenerate before citing this fixture as production fidelity.
# ---------------------------------------------------------------------
PTOP = 100.0            # Pa, model-top pressure (test value)
AKAP = 2.0 / 7.0        # dyn_core `akap` dummy (FMS convention, see U1)
CP_AIR = float(constants.c_pd)   # J/(kg K); coincides with FMS rdgas*7/2
DT = 225.0              # one_grad_p `dt`
DT2 = 112.5             # p_grad_c `dt2` (matches the family's c_sw dt2)
D_EXT = 0.02            # > 0 so the wk1/wk2 external-mode branch is live
P_SFC_REF = 1.0e5       # Pa, reference surface pressure of the test column
P_PROBE_HI = 1.1e5      # Pa, top of the LOGEXP probe span

# Layer thickness partitions.  DELIBERATELY UNEQUAL, and the km=3
# profile is NOT a refinement that reproduces the km=2 interfaces
# (0.30 vs 0.20 of the column at the k=2 interface) — that is what makes
# `test_km2_km3_differ` a real check rather than a tautology.
KM_PROFILES = {2: (0.30, 0.70), 3: (0.20, 0.45, 0.35)}

# ---------------------------------------------------------------------
# Canonical field order the Fortran driver reads AND the fixture stores;
# (UPPER driver token, fixture npz key, index-origin code).  ONE order,
# ONE writer.  Origin codes: "h" = 1-ng on both horizontal axes (k always
# from 1), "1" = Fortran index 1, "s" = scalar (no indices).
# ---------------------------------------------------------------------
INPUT_FIELDS = (
    # grid metrics (k-independent)
    ("RDXC", "rdxc", "h"), ("RDYC", "rdyc", "h"),
    ("RDX", "rdx", "h"), ("RDY", "rdy", "h"),
    ("DXA", "dxa", "h"), ("DYA", "dya", "h"),
    ("GRID_LON", "grid_lon", "h"), ("GRID_LAT", "grid_lat", "h"),
    ("AGRID_LON", "agrid_lon", "h"), ("AGRID_LAT", "agrid_lat", "h"),
    ("EDGE_W", "edge_w", "1"), ("EDGE_E", "edge_e", "1"),
    ("EDGE_S", "edge_s", "1"), ("EDGE_N", "edge_n", "1"),
    ("DA_MIN_C", "da_min_c", "s"),
    # C-grid stage state
    ("DELPC", "delpc", "h"), ("PTC", "ptc", "h"),
    ("UC", "uc", "h"), ("VC", "vc", "h"),
    # D-grid stage state
    ("DELP", "delp", "h"), ("PT", "pt", "h"),
    ("U", "u", "h"), ("V", "v", "h"),
    ("HS", "hs", "h"),
    ("DIVG2", "divg2", "1"),
    ("Q_CON", "q_con", "h"),
    ("LOGEXP_PROBE", "logexp_probe", "1"),
)

# Inputs that are provably NOT read on this lane.  Each entry is a CLAIM
# checked two-sidedly by `test_every_serialized_input_is_consumed`, and
# each reason is echoed into the fixture's `input_lineage`.
_A2B_DEAD = (
    "dxa/dya/grid_lon/grid_lat/agrid_lon/agrid_lat/edge_w/e/s/n: on the "
    "DUO branch a2b_ord4 takes the interior-everywhere arms (a2b_edge.F90 "
    "gates 98/185/241) and reads NONE of them; the Fortran still "
    "associates the pointers unconditionally, so they must exist and are "
    "serialized for interface fidelity")
DEAD_INPUTS = {
    "q_con": "q_con: unreferenced without -DUSE_COND (dyn_core.F90:2721-"
             "2724/2747-2750/2772-2773); serialized to keep the argument "
             "position and the assumed-shape interface in the certificate",
    "da_min_c": "da_min_c: read by NEITHER geopk, p_grad_c NOR one_grad_p; "
                "serialized because the gridstruct member exists and the "
                "divg2 construction scales by it",
    "dxa": _A2B_DEAD, "dya": _A2B_DEAD,
    "grid_lon": _A2B_DEAD, "grid_lat": _A2B_DEAD,
    "agrid_lon": _A2B_DEAD, "agrid_lat": _A2B_DEAD,
    "edge_w": _A2B_DEAD, "edge_e": _A2B_DEAD,
    "edge_s": _A2B_DEAD, "edge_n": _A2B_DEAD,
    "logexp_probe": "logexp_probe: consumed ONLY by the LOGEXP_OUT "
                    "transcendental-attribution probe, not by the chain",
}

# Authoritative Fortran block SHAs, computed at extraction time from the
# Zenodo 8327578 symmetryclean tree with the family convention
# ('\n'.join(lines[a-1:b]) + '\n').  The a2b_ord4 value is IDENTICAL to
# the one recorded by gen_dsw5_duo_oracle.py — a live self-check that the
# hashing convention did not drift.
AUTH_BLOCK_SHA256 = ";".join((
    "geopk:dyn_core.F90:2660-2790:"
    "faa6bdcec132712bbc15505890c015f3e887c9dd2c25725e31bebaeeb650b38e",
    "p_grad_c:dyn_core.F90:2073-2132:"
    "5e21f4b1ef8a5640aff643193d9c1ae36a48508dd17aef172f840ead11b9e353",
    "one_grad_p:dyn_core.F90:2347-2480:"
    "09c2082f6ca2e2fb75d65ce8908b6ac8470cce1833f6ae5da02ffe8bf77f01fb",
    "a2b_ord4:a2b_edge.F90:50-330:"
    "b1d4038993561f73a5bfb52b35318f560381ecfe04ad5d16346e85434560ab58",
    "a2b_ord2:a2b_edge.F90:332-453:"
    "077db6d0bf2073553d93f36e356faa727b47de334b832040108a665340a1b0af",
    "a2b_params:a2b_edge.F90:33-43:"
    "66c491a6baff9e46293d0811360993e886275dd1a8aa5d92f01aa3f01cbdf98e",
))

EXTRACT = os.path.join(HERE, "fv3_geopk_pgrad_extract.F90")
DRIVER = os.path.join(HERE, "fv3_geopk_pgrad_driver.F90")
SHIM = os.path.join(HERE, "fv3_geopk_pgrad_shim.F90")

SENTINEL = 1.0e30

# dumped-token layout (spec 1.3).  shapes/origins are declared
# EXPLICITLY per token so a driver/packer disagreement cannot be papered
# over by inference.
TOKEN2KEY = {
    "PKC_C": "pkc_c", "GZ_C": "gz_c", "PE_C": "pe_c", "PELN_C": "peln_c",
    "PKZ_C": "pkz_c", "UC_PGC": "uc_pgc", "VC_PGC": "vc_pgc",
    "UC_PGC_DPPOISON": "uc_pgc_dppoison",
    "VC_PGC_DPPOISON": "vc_pgc_dppoison",
    "PKC_D": "pkc_d", "GZ_D": "gz_d", "PE_D": "pe_d", "PELN_D": "peln_d",
    "PKZ_D": "pkz_d", "U_OGP": "u_ogp", "V_OGP": "v_ogp",
    "PK_OGP": "pk_ogp", "GZ_OGP": "gz_ogp",
    "U_OGP_DPPOISON": "u_ogp_dppoison",
    "V_OGP_DPPOISON": "v_ogp_dppoison",
    "LOGEXP_OUT": "logexp_out",
}


def output_layout(res: int = RES, ng: int = NG, km: int = 2) -> tuple:
    """``(shapes, origins)`` for every dumped token (spec 1.3)."""
    lo = 1 - ng
    m_a = res + 2 * ng
    m_b = m_a + 1
    cell3 = (m_a, m_a, km + 1)
    shapes = {
        "pkc_c": cell3, "gz_c": cell3, "pkc_d": cell3, "gz_d": cell3,
        "pk_ogp": cell3, "gz_ogp": cell3,
        "pe_c": (res + 2, km + 1, res + 2),
        "pe_d": (res + 2, km + 1, res + 2),
        "peln_c": (res, km + 1, res), "peln_d": (res, km + 1, res),
        "pkz_c": (res, res, km), "pkz_d": (res, res, km),
        "uc_pgc": (m_b, m_a, km), "vc_pgc": (m_a, m_b, km),
        "uc_pgc_dppoison": (m_b, m_a, km),
        "vc_pgc_dppoison": (m_a, m_b, km),
        "u_ogp": (m_a, m_b, km), "v_ogp": (m_b, m_a, km),
        "u_ogp_dppoison": (m_a, m_b, km),
        "v_ogp_dppoison": (m_b, m_a, km),
        "logexp_out": (NPROBE,),
    }
    origins = {k: (lo, lo, 1) for k in
               ("pkc_c", "gz_c", "pkc_d", "gz_d", "pk_ogp", "gz_ogp",
                "uc_pgc", "vc_pgc", "uc_pgc_dppoison", "vc_pgc_dppoison",
                "u_ogp", "v_ogp", "u_ogp_dppoison", "v_ogp_dppoison")}
    # pe/peln keep the upstream (i, k, j) axis order
    origins["pe_c"] = origins["pe_d"] = (0, 1, 0)
    origins["peln_c"] = origins["peln_d"] = (1, 1, 1)
    origins["pkz_c"] = origins["pkz_d"] = (1, 1, 1)
    origins["logexp_out"] = (1,)
    return shapes, origins


def _dump_field(name: str, a: np.ndarray, lo: int) -> str:
    a = np.asarray(a, dtype=np.float64)
    out = []
    if a.ndim == 1:
        for i in range(a.shape[0]):
            out.append(f"{name} {i + lo} {a[i]:.17e}\n")
    elif a.ndim == 2:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                out.append(f"{name} {i + lo} {j + lo} {a[i, j]:.17e}\n")
    elif a.ndim == 3:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for k in range(a.shape[2]):
                    out.append(f"{name} {i + lo} {j + lo} {k + 1} "
                               f"{a[i, j, k]:.17e}\n")
    else:
        raise ValueError(f"{name}: unsupported rank {a.ndim}")
    return "".join(out)


def serialize_geopk_pgrad_inputs(fields: dict, res: int = RES, ng: int = NG,
                                 km: int = 2) -> bytes:
    """The exact ``geopk_pgrad_input.txt`` byte stream for ``fields``.

    ``fields`` is keyed by the lowercase npz names of
    :data:`INPUT_FIELDS`.  The Fortran driver reads these bytes and
    ``input_sha256`` is their SHA-256, so gen and the test both call THIS
    function — gen on the freshly built arrays, the test on the fixture's
    stored arrays — and get identical bytes iff the stored arrays are the
    ones hashed.
    """
    lo = 1 - ng
    s = (f"# res {res}\n# ng {ng}\n# km {km}\n"
         f"# dt {DT:.17e}\n# dt2 {DT2:.17e}\n# ptop {PTOP:.17e}\n"
         f"# akap {AKAP:.17e}\n# cpair {CP_AIR:.17e}\n"
         f"# dext {D_EXT:.17e}\n")
    for name, key, origin in INPUT_FIELDS:
        if origin == "s":
            s += f"{name} {float(fields[key]):.17e}\n"
        else:
            s += _dump_field(name, fields[key], lo if origin == "h" else 1)
    return s.encode()


def _index_ripple(m: int) -> np.ndarray:
    """Deterministic, finite, non-separable (i,j) ripple in ~[0.998, 1.002].

    Applied to every constructed field so that NO cell — including the
    four corner-diagonal ghost blocks, where the analytic state carries
    the ``fv3_native_gridstruct.BIG_NUMBER`` sentinel rather than data —
    is constant in (i, j).  A constant halo would let a level-broadcast
    or window bug hide there, and geopk's ``computehalo=.true.`` D pass
    writes the FULL data domain.
    """
    ii, jj = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    return 1.0 + 0.002 * np.cos(2.0 * np.pi * ii / m) \
        * np.sin(2.0 * np.pi * (jj + 0.5) / m)


def _normalised(a: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """``a / mean(a[ok])`` on ``ok``, 1.0 elsewhere (sentinel-safe)."""
    base = float(np.asarray(a)[ok].mean())
    return np.where(ok, np.asarray(a, dtype=np.float64) / base, 1.0)


def build_inputs(km: int) -> dict:
    """Deterministic input set for one ``km`` fixture (no RNG)."""
    from legoesm.core.fv3_native_sw_core import Bounds, c_sw
    from legoesm.grids.fv3_native_gridstruct import (
        BIG_NUMBER as GS_SENTINEL,
    )
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA,
        FV3_RADIUS_M,
        analytic_swcore_state,
        build_fv3_native_gridstruct,
    )

    if km not in KM_PROFILES:
        raise SystemExit(f"unknown km={km!r}; expected one of "
                         f"{sorted(KM_PROFILES)}")
    frac = KM_PROFILES[km]
    if abs(sum(frac) - 1.0) > 1e-12:
        raise SystemExit(f"km={km} profile does not partition the column")

    gs = build_fv3_native_gridstruct(RES, NG, tile=1, radius=FV3_RADIUS_M,
                                     omega=FV3_OMEGA)
    st = analytic_swcore_state(gs)
    bd = Bounds.single_tile(RES, NG)
    m_a = RES + 2 * NG
    ng = NG
    cell_ok = np.asarray(gs["cell_ok"], dtype=bool)
    ripple = _index_ripple(m_a)

    # --- D-grid column: p_sfc modulated horizontally, UNEQUAL layers
    p_sfc = P_SFC_REF * (1.0 + 0.03 * (_normalised(st["delp"], cell_ok)
                                       - 1.0)) * ripple
    delp = np.stack([f * (p_sfc - PTOP) for f in frac], axis=-1)

    # --- pt: different per level AND per (i,j) so a level-swapped bug
    # cannot survive.  Strictly positive => gz decreases downward.
    coslat = np.where(cell_ok, np.cos(np.asarray(gs["agrid_lat"], float)),
                      0.0)
    pt = np.stack([(300.0 - 10.0 * (k + 1) + 5.0 * coslat) * ripple
                   for k in range(km)], axis=-1)

    # --- C-grid stage state from the CERTIFIED c_sw (one call on the
    # analytic state, then mapped onto the same column partition).
    # PLAIN c_sw (duogrid default) exactly as gen_divduo_oracle.py — the
    # C-stage fields are serialized INPUTS here, so the branch c_sw took
    # is a pattern choice, not part of the certificate.
    csw = c_sw(delp=st["delp"], pt=st["pt"],
               w=np.zeros_like(np.asarray(st["delp"])),
               u=st["u"], v=st["v"], gs=gs, bd=bd, npx=RES + 1,
               npy=RES + 1, dt2=DT2, nord=1, hydrostatic=True,
               dord4=True, grid_type=0)
    dc = np.nan_to_num(np.asarray(csw["delpc"], float), nan=0.0,
                       posinf=0.0, neginf=0.0)
    tc = np.nan_to_num(np.asarray(csw["ptc"], float), nan=0.0,
                       posinf=0.0, neginf=0.0)
    p_sfc_c = P_SFC_REF * (1.0 + 0.03 * (_normalised(dc, dc > 0.0) - 1.0)) \
        * ripple
    delpc = np.stack([f * (p_sfc_c - PTOP) for f in frac], axis=-1)
    mod_t = _normalised(tc, tc > 0.0)
    ptc = np.stack([(300.0 - 10.0 * (k + 1)) * mod_t * ripple
                    for k in range(km)], axis=-1)

    def _desentinel(a):
        """Zero the analytic state's BIG_NUMBER slots.  They are a
        SENTINEL, not data (unreachable corner-diagonal wind slots), and
        they are never read — the PG updates run on the compute box —
        but leaving them in the dumped tokens would wreck the physical
        band checks."""
        a = np.asarray(a, dtype=np.float64)
        return np.where(np.abs(a) >= 0.5 * GS_SENTINEL, 0.0, a)

    uc2 = np.nan_to_num(_desentinel(csw["uc"]), nan=0.0)
    vc2 = np.nan_to_num(_desentinel(csw["vc"]), nan=0.0)
    u2 = np.nan_to_num(_desentinel(st["u"]), nan=0.0)
    v2 = np.nan_to_num(_desentinel(st["v"]), nan=0.0)
    # level-dependent factors: no two levels identical
    uc = np.stack([uc2 * (1.0 + 0.35 * (k + 1)) for k in range(km)], axis=-1)
    vc = np.stack([vc2 * (1.0 - 0.23 * (k + 1)) for k in range(km)], axis=-1)
    u = np.stack([u2 * (1.0 + 0.27 * (k + 1)) for k in range(km)], axis=-1)
    v = np.stack([v2 * (1.0 - 0.19 * (k + 1)) for k in range(km)], axis=-1)

    # --- hs: NONZERO analytic surface geopotential, max|hs| > 1e4 m^2/s^2
    lat_ok = np.where(cell_ok, np.asarray(gs["agrid_lat"], float), 0.0)
    lon_ok = np.where(cell_ok, np.asarray(gs["agrid_lon"], float), 0.0)
    hs = 1.5e4 * (1.0 + np.cos(lat_ok) * np.cos(lon_ok)) * ripple

    # --- divg2 on the B nodes bd%is:bd%ie+1 x bd%js:bd%je+1, origin (1,1).
    # UNCERTAIN U9: here divg2 is an INPUT, so its MAGNITUDE is a
    # discrimination choice, not a fidelity claim.  The km>1 upstream
    # formula (dyn_core.F90:1310-1326) exists in this repo only as the
    # jax `column_d_ext_field` — port THAT (do not re-derive, and do not
    # reuse the km=1 delpc collapse) when the stepper is generalised.
    sl_b = slice(ng, ng + RES + 1)
    blon = np.asarray(gs["grid_lon"], float)[sl_b, sl_b]
    blat = np.asarray(gs["grid_lat"], float)[sl_b, sl_b]
    shape_b = (1.0 + 0.4 * np.sin(2.0 * blon) * np.cos(blat)
               + 0.2 * np.cos(3.0 * blat) + 0.1 * np.sin(blon + 2.0 * blat))
    divg2 = D_EXT * float(gs["da_min_c"]) * 1.0e-2 * shape_b

    # --- LOGEXP probe: log-spaced over [ptop, P_PROBE_HI] PLUS the exact
    # interface pressures of the (0,0) column, accumulated exactly the
    # way geopk accumulates them.
    p_int = [PTOP]
    for k in range(km):
        p_int.append(p_int[-1] + float(delp[0, 0, k]))
    n_log = NPROBE - len(p_int)
    probe = np.concatenate([
        np.logspace(np.log10(PTOP), np.log10(P_PROBE_HI), n_log),
        np.asarray(p_int, dtype=np.float64)])

    # --- gridstruct sub-blocks.  UNCERTAIN U4: python rdyc is
    # (m_a, m_b); the p_grad_c dummy is (isd:ied, jsd:jed) = (m_a, m_a),
    # so only that sub-block is serialized.  Assert the dropped column is
    # unreachable: the vc loop reads rdyc(i, j) with j <= je+1, i.e.
    # python index j-jsd <= RES+NG <= m_a-1.
    rdyc_full = np.asarray(gs["rdyc"], float)
    if RES + NG > m_a - 1:
        raise SystemExit("rdyc sub-block would drop a column the vc loop "
                         "reads — do NOT truncate")
    fields = {
        "rdxc": np.asarray(gs["rdxc"], float),
        "rdyc": rdyc_full[:, :m_a],
        "rdx": np.asarray(gs["rdx"], float),
        "rdy": np.asarray(gs["rdy"], float),
        "dxa": np.asarray(gs["dxa"], float),
        "dya": np.asarray(gs["dya"], float),
        "grid_lon": np.asarray(gs["grid_lon"], float),
        "grid_lat": np.asarray(gs["grid_lat"], float),
        "agrid_lon": np.asarray(gs["agrid_lon"], float),
        "agrid_lat": np.asarray(gs["agrid_lat"], float),
        "edge_w": np.asarray(gs["edge_w"], float),
        "edge_e": np.asarray(gs["edge_e"], float),
        "edge_s": np.asarray(gs["edge_s"], float),
        "edge_n": np.asarray(gs["edge_n"], float),
        "da_min_c": float(gs["da_min_c"]),
        "delpc": delpc, "ptc": ptc, "uc": uc, "vc": vc,
        "delp": delp, "pt": pt, "u": u, "v": v,
        "hs": hs, "divg2": divg2,
        "q_con": np.zeros((m_a, m_a, km)),
        "logexp_probe": probe,
    }
    for key in ("edge_w", "edge_e", "edge_s", "edge_n"):
        if fields[key].shape != (RES + 1,):
            raise SystemExit(f"{key}: expected length {RES + 1}, got "
                             f"{fields[key].shape}")
    # every column must be strictly positive over the FULL data domain:
    # geopk's computehalo D pass takes log() there.
    for key in ("delp", "delpc", "pt", "ptc"):
        if not np.all(np.asarray(fields[key]) > 0.0):
            raise SystemExit(f"{key}: non-positive entry — geopk would "
                             f"log() a non-positive pressure")
    flat = np.concatenate([np.ravel(arr) for arr in fields.values()
                           if isinstance(arr, np.ndarray)])
    if not np.isfinite(flat).all():
        raise SystemExit("non-finite serialized input")
    if float(np.abs(hs).max()) <= 1.0e4:
        raise SystemExit("hs is too flat to make gz(km+1)=hs nontrivial")
    return fields


def run_port(fields: dict, km: int) -> dict:
    """Run the python port on ``fields``; returns every dumped token.

    Used as the gen-time smoke run AND, at test time, as the thing the
    fixture is compared against.  Both call the SAME function, so the
    test cannot accidentally certify a different call sequence.
    """
    from legoesm.core.fv3_native_pgrad import geopk, one_grad_p, p_grad_c
    from legoesm.core.fv3_native_sw_core import Bounds

    bd = Bounds.single_tile(RES, NG)
    npx = npy = RES + 1
    gs = {
        "rdxc": fields["rdxc"], "rdyc": fields["rdyc"],
        "rdx": fields["rdx"], "rdy": fields["rdy"],
        "dxa": fields["dxa"], "dya": fields["dya"],
        "grid_lon": fields["grid_lon"], "grid_lat": fields["grid_lat"],
        "agrid_lon": fields["agrid_lon"], "agrid_lat": fields["agrid_lat"],
        "edge_w": fields["edge_w"], "edge_e": fields["edge_e"],
        "edge_s": fields["edge_s"], "edge_n": fields["edge_n"],
        "bounded_domain": False, "grid_type": 0,
        "sw_corner": True, "se_corner": True,
        "nw_corner": True, "ne_corner": True,
    }
    out = {}

    def _geo(delp, pt, *, cg, computehalo):
        return geopk(delp, pt, fields["hs"], bd, km=km, ptop=PTOP,
                     akap=AKAP, cp_air=CP_AIR, cg=cg, duogrid=True,
                     computehalo=computehalo, npx=npx, npy=npy, a2b_ord=4,
                     bounded_domain=False, sw_dynamics=False,
                     q_con=fields["q_con"], use_cond=False,
                     unwritten_fill=SENTINEL)

    # 1. geopk, C-grid site (dyn_core.F90:533)
    gc = _geo(fields["delpc"], fields["ptc"], cg=True, computehalo=False)
    out.update(pkc_c=gc["pk"], gz_c=gc["gz"], pe_c=gc["pe"],
               peln_c=gc["peln"], pkz_c=gc["pkz"])

    # 2. p_grad_c (dyn_core.F90:629) — mutates uc/vc in place
    uc = np.array(fields["uc"], dtype=np.float64, copy=True)
    vc = np.array(fields["vc"], dtype=np.float64, copy=True)
    p_grad_c(DT2, fields["delpc"], gc["pk"], gc["gz"], uc, vc, gs, bd,
             npz=km, hydrostatic=True)
    out.update(uc_pgc=uc, vc_pgc=vc)
    # the port structurally never reads delpc on the hydrostatic branch
    # (it is `del`-eted), so the poison control is the SAME array on this
    # side by construction; the discriminating comparison is the FORTRAN
    # poison re-run, which is what the fixture's *_DPPOISON tokens hold.
    out.update(uc_pgc_dppoison=uc, vc_pgc_dppoison=vc)

    # 3. geopk, D-grid site (dyn_core.F90:1401), computehalo=.true.
    gd = _geo(fields["delp"], fields["pt"], cg=False, computehalo=True)
    out.update(pkc_d=np.array(gd["pk"], copy=True),
               gz_d=np.array(gd["gz"], copy=True),
               pe_d=gd["pe"], peln_d=gd["peln"], pkz_d=gd["pkz"])

    # 4. one_grad_p (dyn_core.F90:1531) — mutates u/v AND pk/gz in place
    u = np.array(fields["u"], dtype=np.float64, copy=True)
    v = np.array(fields["v"], dtype=np.float64, copy=True)
    one_grad_p(u, v, gd["pk"], gd["gz"], fields["divg2"], None, gs, bd,
               npx=npx, npy=npy, npz=km, dt=DT, ptop=PTOP, akap=AKAP,
               hydrostatic=True, a2b_ord=4, d_ext=D_EXT, ng=NG,
               duogrid=True)
    out.update(u_ogp=u, v_ogp=v, pk_ogp=gd["pk"], gz_ogp=gd["gz"])
    out.update(u_ogp_dppoison=u, v_ogp_dppoison=v)

    # 5. transcendental attribution probe
    out["logexp_out"] = np.exp(AKAP * np.log(
        np.asarray(fields["logexp_probe"], dtype=np.float64)))
    return out


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _gen(work: str, km: int) -> None:
    fields = build_inputs(km)
    blob = serialize_geopk_pgrad_inputs(fields, RES, NG, km)
    with open(f"{work}/geopk_pgrad_input.txt", "wb") as f:
        f.write(blob)
    # stage the EXACT arrays the fixture will carry, so --pack does NOT
    # regenerate them
    np.savez_compressed(f"{work}/staging_km{km}.npz", **fields)
    run_port(fields, km)          # smoke: the port runs on these inputs
    print(f"gen: geopk_pgrad_input.txt + staging_km{km}.npz written "
          f"(res={RES} ng={NG} km={km}); "
          f"sha256 {hashlib.sha256(blob).hexdigest()}")


def _pack(work: str, km: int) -> None:
    stag = dict(np.load(f"{work}/staging_km{km}.npz"))
    canon = serialize_geopk_pgrad_inputs(stag, RES, NG, km)
    on_disk = open(f"{work}/geopk_pgrad_input.txt", "rb").read()
    if canon != on_disk:
        raise SystemExit("geopk_pgrad_input.txt disagrees with "
                         f"staging_km{km}.npz — regenerate; refusing to "
                         "pack a drifted fixture")
    inp_hash = hashlib.sha256(canon).hexdigest()

    shapes, origins = output_layout(RES, NG, km)
    outs = {k: np.full(v, np.nan) for k, v in shapes.items()}
    with open(f"{work}/geopk_pgrad_output_km{km}.txt") as fh:
        for line in fh:
            pp = line.split()
            if not pp:
                continue
            key = TOKEN2KEY[pp[0]]
            org = origins[key]
            if len(org) == 1:
                outs[key][int(pp[1]) - org[0]] = float(pp[2])
            else:
                outs[key][int(pp[1]) - org[0], int(pp[2]) - org[1],
                          int(pp[3]) - org[2]] = float(pp[4])
    for k, a in outs.items():
        if np.isnan(a).any():
            raise SystemExit(f"dump under-writes token {k}")

    lineage = (
        "ROUTINE-TRANSLATION CERTIFICATE, one face, single acoustic "
        "substep: geopk(CG=T,computehalo=F) -> p_grad_c -> "
        "geopk(CG=F,computehalo=T) -> one_grad_p, DUO branch.  "
        "EXCLUSIONS: no six-face exchange or averaging cadence, no "
        "ext_scalar schedule, no USE_COND (dry), no SW_DYNAMICS, "
        "hydrostatic only, beta<=0, a2b_ord=4, d_ext>0.  Inputs: "
        "certified c_sw (plain branch) delpc/ptc/uc/vc mapped onto an "
        "UNEQUAL km-layer partition, analytic D-grid state, nonzero "
        "analytic hs, smooth non-separable divg2.  Sentinels: 1e30 = "
        "never-written output region (write window certified), -9.e9 = "
        "poisoned dead-branch input.  ptop/akap/cp_air are HEADER "
        "values fed identically to both sides (UNCERTAIN U1/U6: FMS "
        "constants_mod is absent from the Zenodo tree, so these are "
        "TEST values, NOT production-fidelity claims).  DEAD INPUTS "
        "(serialized, provably unread on this lane) -- "
        + " | ".join(sorted(set(DEAD_INPUTS.values()))))

    fix = os.path.join(REPO, "tests", "grids", "fixtures",
                       f"geopk_pgrad_oracle_c12_km{km}.npz")
    np.savez_compressed(
        fix, **outs,
        **{f"in_{k}": v for k, v in stag.items()},
        res=RES, ng=NG, km=km, dt=DT, dt2=DT2, ptop=PTOP, akap=AKAP,
        cp_air=CP_AIR, d_ext=D_EXT, npx=RES + 1, npy=RES + 1,
        input_sha256=inp_hash,
        geopk_pgrad_extract_sha256=_sha(EXTRACT),
        driver_sha256=_sha(DRIVER), shim_sha256=_sha(SHIM),
        auth_block_sha256=AUTH_BLOCK_SHA256,
        input_lineage=lineage)
    print("fixture packed;", fix)
    print("input_sha256", inp_hash)
    print("extract sha256", _sha(EXTRACT))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        raise SystemExit("usage: gen_geopk_pgrad_oracle.py <WORKDIR> "
                         "[--km 2|3] [--pack]")
    _work = argv[0]
    _flags = argv[1:]
    _km = 2
    if "--km" in _flags:
        _km = int(_flags[_flags.index("--km") + 1])
    if "--pack" in _flags:
        _pack(_work, _km)
    else:
        _gen(_work, _km)
