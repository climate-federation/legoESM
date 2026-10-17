#!/usr/bin/env python3
"""Grid ladder rung G2: the ORCA2G2-zps card against NEMO's record, by level.

The G1 ladder (nemo_testcase_g1_orca2grid_ladder.py) is written for one
level; this script reuses its readers, regions and row scorer and adds what
30 levels need:

  geometry     every mesh array the card carries against NEMO's mesh_mask.nc
               on ALL levels (NEMO's level 31 must be dry).
  independent  the card's own chain from its own initial state.  Level 1 and
               the 2-D fields are scored at every step from NEMO's step
               frames; all levels where NEMO wrote a restart.  The chain is
               resumable (--state-in/--state-out): one step costs about a
               minute, so 100 steps are run in chunks.
  given_entry  NEMO's restart of step kt - 1 as the entry of step kt (all
               prognostics on all levels, the six barotropic histories and,
               on G6, the carried after-ssh slot 'ssha'), then one step (level 1 and 2-D rows) and a second step
               (all-level rows against NEMO's next restart: TWO-STEP rows).
  substeps     step 1: the split-explicit solver's loop, every recorded
               operand in compiled order against the model's write-only
               trace (G1's reader and table), and the traced loop's exit and
               the ordinary step's handoff each against NEMO's exit frame.
  rhs_frame    step 1: the traced solver's 3-D right-hand side, level 1,
               against NEMO's frame after stp_2D (wet and dry faces apart),
               and an OFFLINE replay of its depth mean (stp2d.f90:178-179,
               ascending k) against the card's.
  stp2d_terms  G6, kt 1..3: stp_2D in compiled order -- the 3-D RHS after
               dyn_hpg, dyn_vor, dyn_keg, dyn_zad (r42 record), its depth mean,
               after drag, after wind (slow-forcing record) and dyn_spg_ts's
               exit -- against the card's write-only observers.  kt 1 from
               the initial state, kt 2 the card's own chain (INDEPENDENT),
               kt >= 3 NEMO's restart of kt - 1 (GIVEN-NEMO-ENTRY).
  trazdf_replay  step 1, OFFLINE numpy replay of NEMO's implicit vertical
               tracer solve (trazdf.f90:231-233, :269-272, :284-298) from
               NEMO's recorded thickness ratios and the mesh file: level 1
               against NEMO (must be 0 unequal: that makes it NEMO's), then
               the card's T and S after one step against it on ALL levels.

All-level rows are split by level and by DEPTH CLASS, read from NEMO's own
mesh file: FULL = wet cell whose rest thickness equals the 1-D ladder's,
PARTIAL = wet cell whose rest thickness differs (a bottom cell or a face
beside one).  Labels are never mixed (D52).
"""

from __future__ import annotations

import argparse
import json
import pickle
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import nemo_testcase_g1_orca2grid_ladder as g1  # noqa: E402
from nemo_testcase_g1_orca2grid_ladder import BAR, require  # noqa: E402

CASE = "ORCA2G2-zps"
CASES = ("ORCA2G2-zps", "ORCA2G3-ts", "ORCA2G4-sbc", "ORCA2G5-ice",
         "ORCA2G6-vec", "ORCA2G7-drg", "ORCA2G8-ldf")
ICE_CASES = ("ORCA2G5-ice", "ORCA2G6-vec", "ORCA2G7-drg", "ORCA2G8-ldf")
FORCED = ("ORCA2G4-sbc", *ICE_CASES)   # NEMO-supplied surface fields each kt (D123)
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round10/"
    "oracle_g2_r10/p3_a_100step")
ARMS = ("geometry", "independent", "given_entry", "stages", "operators",
        "barotropic_exit",
        "zdf_replay", "substeps", "trazdf_replay", "rhs_frame",
        "restart_independent", "carried", "stp2d_terms")
# stage exposures of one step: the private hook that returns it, and the
# level-1 NEMO frame fields (u, v, T, S, ssh) it must equal
EXPOSURES = {
    "rhs": ({"expose_stage1_momentum_rhs": True},
            {"u": "b_uu_rhs_k1", "v": "b_vv_rhs_k1"}),
    "stage1": ({"expose_momentum_stage": 1, "expose_tracer_stage": 1},
               {"u": "1_uu_k1_aa", "v": "1_vv_k1_aa", "T": "1_tn_k1_aa",
                "S": "1_sn_k1_aa", "ssh": "1_ssh_aa",
                "uu_b": "1_uu_b_aa", "vv_b": "1_vv_b_aa"}),
    "stage2": ({"expose_momentum_stage": 2, "expose_tracer_stage": 2},
               {"u": "2_uu_k1_aa", "v": "2_vv_k1_aa", "T": "2_tn_k1_aa",
                "S": "2_sn_k1_aa", "ssh": "2_ssh_aa",
                "uu_b": "2_uu_b_aa", "vv_b": "2_vv_b_aa"}),
    "stage3": ({}, {"u": "3_uu_k1_aa", "v": "3_vv_k1_aa", "T": "3_tn_k1_aa",
                    "S": "3_sn_k1_aa", "ssh": "3_ssh_aa",
                    "uu_b": "3_uu_b_aa", "vv_b": "3_vv_b_aa"}),
}
FRAME = {"ssh": "f_ssh_bb", "uu_b": "f_uu_b_bb", "vv_b": "f_vv_b_bb",
         "u": "f_uu_k1_bb", "v": "f_vv_k1_bb"}
RESTART = {"u": "un", "v": "vn", "T": "tn", "S": "sn"}
POINT = {"u": "u", "v": "v", "T": "t", "S": "t"}


def restart_path(roots, kt: int):
    for root in roots:
        path = Path(root) / f"ORCA2G1_{kt:08d}_restart.nc"
        if path.is_file():
            return path
    return None


def levels_last(a, nk: int) -> np.ndarray:
    """NEMO's (jpk, y, x) as the card's (y, x, nk); the dropped levels must
    hold nothing."""
    a = np.asarray(a, dtype=np.float64)
    require(not np.any(a[nk:]), "NEMO holds a value below the card's last level")
    return np.ascontiguousarray(np.moveaxis(a[:nk], 0, -1))


def read_stage2_terms(path: Path) -> dict[str, np.ndarray]:
    """Read NEMO's source-ordered stage-2 momentum accumulators.

    The writer records native ``(i, j, k)`` arrays including the two-cell
    halo and dummy bottom level.  Keep that native extent here; the caller
    strips it only after validating the header and payload.
    """
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        raw_header = handle.read(struct.calcsize("=9i"))
        require(len(raw_header) == struct.calcsize("=9i"),
                f"{path}: truncated stage-2 header")
        header = struct.unpack("=9i", raw_header)
        values = np.fromfile(handle, dtype=np.float64)
        require(handle.read(1) == b"", f"{path}: trailing stage-2 payload")
    version, kt, stage, kmm, krhs, nx, ny, nz, bits = header
    require(magic == "NEMO_L2_RKTRM_1", f"{path}: bad stage-2 magic")
    require((version, kt, stage, kmm, krhs, bits) == (1, 1, 2, 3, 2, 64),
            f"{path}: bad stage-2 header {header}")
    count = nx * ny * nz
    names = (
        "before_u", "before_v", "after_hpg_u", "after_hpg_v",
        "after_vorticity_u", "after_vorticity_v",
        "after_advection_u", "after_advection_v",
    )
    require(values.size == len(names) * count,
            f"{path}: bad stage-2 payload count {values.size}")
    require(bool(np.isfinite(values).all()), f"{path}: non-finite stage-2 payload")
    out = {}
    for index, name in enumerate(names):
        raw = values[index * count:(index + 1) * count]
        out[name] = raw.reshape((nx, ny, nz), order="F").transpose(1, 0, 2).copy()
    return out


G6_STGTRM_3D = ("ww",)
G6_STGTRM_2D = ("r3t_bb", "r3t_mm", "r3t_aa", "r3u_mm", "r3v_mm",
                "ssh_bb", "ssh_mm", "ssh_aa")
G6_STGTRM_KRHS = ("before_hpg", "after_hpg", "after_vor", "after_keg", "after_zad")


def read_g6_stage_terms(path: Path, kt: int, stage: int) -> dict:
    """NEMO's write-only stage-term dump (run_r44_stage_terms.sh): the stage
    momentum wzv's ww and its ratio / sea-level slots, then the 3-D Krhs at
    five source-ordered boundaries.  Native (y, x, z) with the halo."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii")
        head = struct.unpack("=12i", handle.read(48))
        values = np.fromfile(handle, dtype=np.float64)
    ver, k, s, kbb, kmm, kaa, krhs, nx, ny, nz, hls, bits = head
    n3, n2 = nx * ny * nz, nx * ny
    require((magic, ver, k, s, hls, bits) == ("NEMO_G6_STGTRM_1", 1, kt, stage, 2, 64)
            and values.size == (1 + 2 * len(G6_STGTRM_KRHS)) * n3 + 8 * n2,
            f"{path}: bad stage-term header {magic!r} {head} {values.size}")
    require(bool(np.isfinite(values).all()), f"{path}: non-finite payload")
    names = (*G6_STGTRM_3D, *G6_STGTRM_2D,
             *(f"{b}_{f}" for b in G6_STGTRM_KRHS for f in "uv"))
    out, at = {"levels": (kbb, kmm, kaa, krhs)}, 0
    for name in names:
        n = n2 if name in G6_STGTRM_2D else n3
        raw = values[at:at + n]
        at += n
        out[name] = (raw.reshape((nx, ny), order="F").T.copy() if n == n2 else
                     raw.reshape((nx, ny, nz), order="F").transpose(1, 0, 2).copy())
    return out


def depth_classes(mesh_path: Path, nk: int) -> dict:
    """{point: {"full": mask, "partial": mask}} on (y, x, nk), from NEMO's
    mesh file alone."""
    import netCDF4
    out = {}
    with netCDF4.Dataset(mesh_path) as h:
        ladder = np.asarray(h.variables["e3t_1d"][0], dtype=np.float64)[:nk]
        for p in "tuv":
            wet = levels_last(h.variables[f"{p}mask"][0], nk) > 0.5
            e3 = levels_last(h.variables[f"e3{p}_0"][0] * h.variables[f"{p}mask"][0], nk)
            same = e3 == ladder[None, None, :]
            out[p] = {"full": wet & same, "partial": wet & ~same}
    return out


def level_rows(ref, cand, cls: dict, *, plant=False) -> list[dict]:
    """One row per level: the wet cells split into the two depth classes.
    ``relative`` divides by NEMO's own maximum over the whole field."""
    ref, cand = np.asarray(ref, np.float64), np.asarray(cand, np.float64)
    require(ref.shape == cand.shape == cls["full"].shape, "level rows: shape mismatch")
    if plant:
        cand = cand.copy()
        cand[tuple(np.argwhere(cls["partial"])[0])] += 1.0
    d, scale = np.abs(cand - ref), max(float(np.max(np.abs(ref))), 1.0)
    rows = []
    for k in range(ref.shape[-1]):
        r = {"level": k + 1}
        for name, mask in cls.items():
            m = mask[..., k]
            n = int(m.sum())
            r[name] = {"n_cells": n,
                       "n_unequal": int(np.count_nonzero(d[..., k][m])),
                       "max_abs": float(d[..., k][m].max()) if n else 0.0,
                       "relative": (float(d[..., k][m].max()) / scale) if n else 0.0,
                       "ref_max_abs": float(np.abs(ref[..., k][m]).max()) if n else 0.0}
        dry = ~(cls["full"][..., k] | cls["partial"][..., k])
        r["dry_n_unequal"] = int(np.count_nonzero(d[..., k][dry]))
        rows.append(r)
    return rows


def class_summary(rows: list[dict]) -> dict:
    out = {}
    for name in ("full", "partial"):
        out[name] = {
            "n_cells": sum(r[name]["n_cells"] for r in rows),
            "n_unequal": sum(r[name]["n_unequal"] for r in rows),
            "max_abs": max(r[name]["max_abs"] for r in rows),
            "relative": max(r[name]["relative"] for r in rows),
            "worst_level": max(rows, key=lambda r: r[name]["max_abs"])["level"]}
    out["dry_n_unequal"] = sum(r["dry_n_unequal"] for r in rows)
    out["levels_partial_worse"] = sum(
        r["partial"]["relative"] > r["full"]["relative"] for r in rows
        if r["partial"]["n_cells"] and r["full"]["n_cells"])
    out["levels_full_worse"] = sum(
        r["full"]["relative"] > r["partial"]["relative"] for r in rows
        if r["partial"]["n_cells"] and r["full"]["n_cells"])
    out["status"] = "AT-BAR" if (max(out["full"]["relative"],
                                     out["partial"]["relative"]) <= BAR
                                 and not out["dry_n_unequal"]) else "DEBT"
    return out


STP2D_3D = ("after_hpg_u", "after_hpg_v", "after_vor_u", "after_vor_v", "ww")
STP2D_2D = ("ssh_aa", "ssh_bb", "r3t_aa", "r3t_bb", "r3u_bb", "r3v_bb")
STP2D_3D_LATE = ("after_keg_u", "after_keg_v", "after_zad_u", "after_zad_v")


def read_stp2d_terms(path: Path, kt: int, nk: int) -> dict[str, np.ndarray]:
    """NEMO's write-only stp_2D term dump (run_r42_stp2d.sh): the 3-D Krhs
    after dyn_hpg, dyn_vor, the first wzv's ww with its sea-level operands,
    after dyn_keg and after dyn_zad, cut to the inner domain and nk levels."""
    with path.open("rb") as handle:
        magic = handle.read(16).decode("ascii").rstrip()
        version, k, kbb, kaa, krhs, nx, ny, nz, bits = struct.unpack(
            "=9i", handle.read(36))
        values = np.fromfile(handle, dtype=np.float64)
    names = (*STP2D_3D, *STP2D_2D, *STP2D_3D_LATE)
    n3, n2 = nx * ny * nz, nx * ny
    require((magic, version, k, bits) == ("NEMO_G6_STP2D_1", 1, kt, 64)
            and values.size == 9 * n3 + 6 * n2,
            f"{path}: bad stp_2D term record {(magic, version, k, bits, values.size)}")
    require(bool(np.isfinite(values).all()), f"{path}: non-finite payload")
    require(nz == nk + 1, f"{path}: {nz} levels, the card has {nk}")
    out, at = {}, 0
    for name in names:
        n = n2 if name in STP2D_2D else n3
        raw = values[at:at + n]
        at += n
        if name in STP2D_2D:
            out[name] = raw.reshape((nx, ny), order="F").T[2:-2, 2:-2].copy()
        else:
            a = raw.reshape((nx, ny, nz), order="F").transpose(1, 0, 2)
            require(not np.any(a[..., nk:]) or name == "ww",
                    f"{name}: NEMO holds a value on the dummy level")
            out[name] = a[2:-2, 2:-2, :nk].copy()
    out["levels"] = (kt, kbb, krhs)
    return out


def bit_rows(name: str, ref, cand, wet, *, plant=False) -> dict:
    """Bitwise census on NEMO's wet points (dry points reported apart):
    unequal count, max |difference|, max ulps, worst level and cell."""
    from nemo_testcase_state_ulp_probe import ulp_distance
    ref, cand = np.asarray(ref, np.float64), np.array(cand, np.float64)
    wet = np.asarray(wet, bool)
    require(ref.shape == cand.shape == wet.shape, f"{name}: shape mismatch "
            f"{ref.shape} {cand.shape} {wet.shape}")
    if plant:
        cand[tuple(np.argwhere(wet & (ref != 0))[0])] = np.nextafter(
            cand[tuple(np.argwhere(wet & (ref != 0))[0])], np.inf)
    unequal = (cand.view(np.uint64) != ref.view(np.uint64)) & wet
    d = np.where(wet, np.abs(cand - ref), 0.0)
    scale = float(np.max(np.abs(ref[wet]), initial=0.0))
    worst = tuple(int(i) for i in np.unravel_index(int(np.argmax(d)), d.shape))
    first = np.argwhere(unequal)
    return {"name": name, "n_wet": int(wet.sum()), "n_unequal": int(unequal.sum()),
            "max_abs": float(d.max()), "ref_max_abs": scale,
            "normalised": float(d.max()) / scale if scale else 0.0,
            "max_ulps": int(np.max(ulp_distance(cand[wet], ref[wet]), initial=0)),
            "worst_cell": worst, "first_cell": first[0].tolist() if first.size else None,
            "levels_touched": (sorted({int(i) + 1 for i in first[:, -1]})
                               if first.size and ref.ndim == 3 else None),
            "dry_card_nonzero": int(np.count_nonzero(cand[~wet])),
            "dry_nemo_nonzero": int(np.count_nonzero(ref[~wet]))}


def stp2d_terms(out, root, kt_from, kt_to, *, card, cfg, z, s0, nk, cls, reg,
                nemo_entry, plant=None, dump=None) -> dict:
    """stp_2D boundaries in compiled order (stp2d.f90 of the r42 build) at
    kt_from..kt_to; every card value comes from a write-only observer of the
    ordinary step, no operator is re-implemented here."""
    import jax
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from nemo_testcase_l1_vortex_kt2_walk import read_bt_frame
    from nemo_testcase_l2_gyre_round16_slow_forcing import read_slow_forcing
    sys.path.insert(0, str(HERE / "nemo_testcase_g5_orca2ice"))
    import g5_records as g5  # noqa: E402
    require(out["case"] == "ORCA2G6-vec", "stp2d_terms is registered for G6")
    seen, zad = {}, {}
    observed = LatLonCGridOceanModel(
        card.recipe.grid, z, cfg, _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True,
            slow_forcing_rhs_term_observer=lambda d: seen.update(
                {k: np.asarray(v) for k, v in d.items()}),
            stage1_zad_operand_observer=lambda d: zad.update(
                {k: np.asarray(v) for k, v in d.items()})))
    plain = LatLonCGridOceanModel(card.recipe.grid, z, cfg)
    wet = {f: cls[f]["full"] | cls[f]["partial"] for f in "uv"}
    wet["t"] = cls["t"]["full"] | cls["t"]["partial"]
    cut = {"u": lambda a: np.asarray(a)[:, 1:], "v": lambda a: np.asarray(a)[1:]}
    out["label"] = ("kt 1: INDEPENDENT = GIVEN-NEMO-ENTRY; kt 2: INDEPENDENT "
                    "(card chain); kt >= 3: GIVEN-NEMO-ENTRY (NEMO restart kt-1)")
    chain = s0
    for kt in range(kt_from, kt_to + 1):
        fw, sf = g5.nemo_supplied_forcing(card, root, kt)
        if kt <= 2:
            entry, label = chain, ("INDEPENDENT = GIVEN" if kt == 1 else "INDEPENDENT")
        else:
            entry, label = nemo_entry(kt), "GIVEN-NEMO-ENTRY"
        seen.clear()
        zad.clear()
        res = jax.device_get(observed.step(entry, dt=card.dt_s, freshwater=fw,
                                           surface_forcing=sf))
        jax.effects_barrier()           # the observer is an async callback
        require(bool(seen) and bool(zad), "a stp_2D observer never fired")
        terms = read_stp2d_terms(root / f"oracle_stp2d_terms_kt{kt:08d}.bin", kt, nk)
        kbb, krhs = terms["levels"][1], terms["levels"][2]
        slow = read_slow_forcing(root / f"oracle_slow_forcing_kt{kt:08d}.bin",
                                 dims=(184, 152, nk + 1), levels=(kt, kbb, krhs))
        ops = {k: np.asarray(v) for k, v in res.slow_forcing_operands.items()}
        bt = read_bt_frame(root / f"oracle_bt_frames_kt{kt:08d}.bin", expect_step=kt)
        rows = []
        w = zad["w"]
        require(w.shape[:2] == terms["ww"].shape[:2] and w.shape[-1] >= nk,
                f"card w {w.shape} vs NEMO ww {terms['ww'].shape}")
        # ww(:,:,1) is the surface face; level k of both is the top of cell k
        rows.append(bit_rows("first_wzv.ww", terms["ww"], w[..., :nk], wet["t"]))
        # level 1 carries NEMO's runoff divergence (registered); 2..nk alone
        rows.append(bit_rows("first_wzv.ww_lev2+", terms["ww"][..., 1:],
                             w[..., 1:nk], wet["t"][..., 1:]))
        for f in "uv":
            for b, card_key in (("after_hpg", "after_hpg"), ("after_vor", "after_vor"),
                                ("after_keg", "after_keg"), ("after_zad", "after_adv")):
                rows.append(bit_rows(f"{b}.{f}", terms[f"{b}_{f}"],
                                     cut[f](seen[f"{card_key}_{f}"]), wet[f],
                                     plant=plant == "score" and b == "after_hpg"
                                     and f == "u"))
            rows.append(bit_rows(f"krhs_into_mean.{f}", slow[f"krhs_{f}"],
                                 cut[f](ops[f"d{f}_dt"]), wet[f]))
        for f in "uv":
            w2 = wet[f][..., 0]
            for b, k in (("Ue_rhs_mean", "depth"), ("Ue_rhs_post_drag", "post_drag"),
                         ("Ue_rhs_post_wind", "post_wind")):
                rows.append(bit_rows(f"{b}.{f}", slow[f"{k}_{f}"],
                                     cut[f](ops[f"{k}_{f}"]), w2))
        for f, key in (("uu_b", "u"), ("vv_b", "v")):
            rows.append(bit_rows(f"spg_ts_exit.{f}", bt[f],
                                 cut[key](getattr(res.state_after_barotropic, f).data),
                                 wet[key][..., 0]))
        if dump is not None:
            Path(dump).mkdir(parents=True, exist_ok=True)
            np.savez(Path(dump) / f"stp2d_kt{kt}.npz",
                     **{f"nemo_{k}": v for k, v in terms.items() if k != "levels"},
                     **{f"card_zad_{k}": v for k, v in zad.items()},
                     **{f"card_{k}": v for k, v in seen.items()
                        if k.startswith("after_")})
        first = next((r for r in rows if r["n_unequal"]), None)
        out["per_kt"].append({"kt": kt, "label": label, "stp2d_rows": rows,
                              "first_unequal": first and first["name"]})
        if kt < kt_to and kt <= 1:
            chain = jax.device_get(plain.step(chain, dt=card.dt_s, freshwater=fw,
                                              surface_forcing=sf))
    return out


def _pickle_fields() -> None:
    """Field forbids attribute assignment, so pickle rebuilds it through its
    constructor (the chain's saved state holds numpy data only)."""
    import copyreg
    from legoesm.core.field import Field
    copyreg.pickle(Field, lambda f: (Field, (
        np.asarray(f.data), f.name, f.dims, f.units, f.long_name, f.staggering)))


def nemo_dynzdf_replay(explicit, barotropic, *, e3_0, mask, avm_pair, r3_aa,
                       r3_mm, r1_h0, e3w_1d, dt, interface="r3_mm"):
    """NEMO's dyn_zdf and the stage's barotropic replacement, statement by
    statement in the written operand order, for one velocity component on
    (y, x, k) arrays (drag and wind stress zero, as on this rung):

      dynzdf.f90:160       remove the barotropic velocity
      dynzdf.f90:191-205   the matrix: (avm + avm) * dt/2 over
                           e3(Kaa) * (e3w_1d * (1 + r3(Kmm))), closed by wmask
      dynzdf.f90:332-349   the three recurrences
      stprk3_stg.f90:413-417  put the barotropic velocity back

    ``interface`` names the stretch of the interface thickness: NEMO's
    "r3_mm"; "r3_aa" and "none" are measurement variants.  Returns
    (after the solve, after the replacement)."""
    nk = explicit.shape[-1]
    u = (explicit - barotropic[..., None]) * mask
    half = dt * 0.5
    e3a = e3_0 * (1.0 + r3_aa[..., None] * mask)
    stretch = {"r3_mm": 1.0 + r3_mm, "r3_aa": 1.0 + r3_aa,
               "none": np.ones_like(r3_mm)}[interface]
    wmask = mask.copy()
    wmask[..., 1:] = mask[..., 1:] * mask[..., :-1]
    zwi, zws, zwd = (np.zeros_like(u) for _ in range(3))
    for k in range(nk):
        if k + 1 < nk:
            zws[..., k] = -half * avm_pair[..., k + 1] / (
                e3a[..., k] * (e3w_1d[k + 1] * stretch)) * wmask[..., k + 1]
        # below the card's last level NEMO's level jpk is dry: zws = 0
        if k > 0:
            zwi[..., k] = -half * avm_pair[..., k] / (
                e3a[..., k] * (e3w_1d[k] * stretch)) * wmask[..., k]
        zwd[..., k] = 1.0 - zwi[..., k] - zws[..., k] if k > 0 else 1.0 - zws[..., k]
    for k in range(1, nk):
        zwd[..., k] = zwd[..., k] - zwi[..., k] * zws[..., k - 1] / zwd[..., k - 1]
    for k in range(1, nk):
        u[..., k] = u[..., k] - zwi[..., k] / zwd[..., k - 1] * u[..., k - 1]
    u[..., nk - 1] = u[..., nk - 1] / zwd[..., nk - 1]
    for k in range(nk - 2, -1, -1):
        u[..., k] = (u[..., k] - zws[..., k] * u[..., k + 1]) / zwd[..., k]
    total = np.zeros_like(barotropic)
    for k in range(nk):                       # SUM( e3_3d * uu ), level order
        total = total + e3_0[..., k] * u[..., k]
    zub = barotropic - total * r1_h0
    return u, u + zub[..., None] * mask


def nemo_trazdf_replay(before, *, tmask, e3t_0, e3w_1d, avt0, r3t_bb, r3t_mm,
                       r3t_aa, dt) -> np.ndarray:
    """NEMO's tra_zdf_imp on one tracer with a zero trend, every statement in
    the written order with one rounding per operation (numpy has no fused
    multiply-add).  ``before`` and the masks are (y, x, nk); the ratios are
    (y, x).  avt = avt0 * wmask (zdfphy.f90:227, :349)."""
    nk = before.shape[-1]
    wmask = tmask.copy()
    wmask[..., 1:] = tmask[..., 1:] * tmask[..., :-1]
    wmask[..., 0] = 0.0                                   # trazdf.f90:216
    avt = np.concatenate([avt0 * wmask, np.zeros_like(wmask[..., :1])], axis=-1)
    zwi, zws, zwd = (np.empty_like(tmask) for _ in range(3))
    for k in range(nk):                                   # trazdf.f90:231-233
        zwi[..., k] = (-dt * avt[..., k]) / (e3w_1d[k] * (1.0 + r3t_mm))
        zws[..., k] = (-dt * avt[..., k + 1]) / (e3w_1d[k + 1] * (1.0 + r3t_mm))
        zwd[..., k] = (e3t_0[..., k] * (1.0 + r3t_aa * tmask[..., k])) - (
            zwi[..., k] + zws[..., k])
    zwt = np.empty_like(tmask)
    zwt[..., 0] = zwd[..., 0]                             # trazdf.f90:269
    for k in range(1, nk):                                # trazdf.f90:272
        zwt[..., k] = zwd[..., k] - zwi[..., k] * zws[..., k - 1] / zwt[..., k - 1]
    pt = np.empty_like(tmask)
    for k in range(nk):                                   # trazdf.f90:284-290
        zrhs = (e3t_0[..., k] * (1.0 + r3t_bb * tmask[..., k])) * before[..., k]
        pt[..., k] = zrhs if k == 0 else (
            zrhs - zwi[..., k] / zwt[..., k - 1] * pt[..., k - 1])
    pt[..., -1] = pt[..., -1] / zwt[..., -1] * tmask[..., -1]   # trazdf.f90:294
    for k in range(nk - 2, -1, -1):                       # trazdf.f90:297-298
        pt[..., k] = (pt[..., k] - zws[..., k] * pt[..., k + 1]) / zwt[..., k] * tmask[..., k]
    return pt


def nemo_ue_rhs_replay(rhs, e3_0, mask, *, lanes: int = 1) -> np.ndarray:
    """OFFLINE stp2d.f90:178-179: SUM(e3*rhs*mask over levels) * r1_h_0,
    with hu_0 summed as domain.f90 does and r1_hu_0 = ss/(h0 + 1 - ss).
    lanes = 1 is Fortran's ascending k; lanes > 1 splits k by residue and
    adds the partial sums pairwise (a vector order, for contrast only)."""
    nk = mask.shape[-1]
    h0 = np.zeros(mask.shape[:-1])
    for k in range(nk):
        h0 = h0 + e3_0[..., k] * mask[..., k]
    ss = mask[..., 0]
    prod = e3_0 * rhs * mask
    part = [np.zeros(mask.shape[:-1]) for _ in range(lanes)]
    for k in range(nk):
        part[k % lanes] = part[k % lanes] + prod[..., k]
    while len(part) > 1:
        part = [part[i] + part[i + 1] for i in range(0, len(part), 2)]
    return part[0] * (ss / (h0 + 1.0 - ss))


ABORT = {"ssh": "sossheig", "u": "vozocrtx", "v": "vomecrty", "T": "votemper",
         "S": "vosaline"}


def abort_witness(roots, kt: int):
    """NEMO's output.abort.nc when stp_ctl stopped the run AT step kt: the
    fp64 Kmm state the check judged (stpctl.f90:181, :245)."""
    for root in roots:
        path, ts = Path(root) / "output.abort.nc", Path(root) / "time.step"
        if path.is_file() and ts.is_file() and int(ts.read_text()) == kt:
            return path
    return None


def abort_rows(path: Path, kt: int, state, nk: int, reg, cls) -> dict:
    """The card at NEMO's stop step: 2-D and level-1 rows by region, every
    level by depth class, and stp_ctl's maxima (wet cells) on both sides."""
    import netCDF4
    cand = card_levels(state)
    cand["ssh"] = np.asarray(state.eta.data)
    with netCDF4.Dataset(path) as h:
        ref = {f: np.asarray(h.variables[n][0], np.float64) for f, n in ABORT.items()}
    require(all(a.dtype == np.float64 for a in ref.values()), "abort file is not fp64")
    ref.update({f: levels_last(ref[f], nk) for f in RESTART})
    rows = [g1.row(f"abort.kt{kt}.{f}", ref[f] if f == "ssh" else ref[f][..., 0],
                   cand[f] if f == "ssh" else cand[f][..., 0], reg)
            for f in ("ssh", "u", "v", "T", "S")]
    levels = {f: {"summary": class_summary(level_rows(ref[f], cand[f], cls[POINT[f]]))}
              for f in RESTART}

    def peak(a, wet):
        a = np.where(wet, np.abs(a), -1.0)
        at = np.unravel_index(int(np.argmax(a)), a.shape)
        # NEMO's (ji, jj, jk), 1-based, as stp_ctl prints them
        return {"max": float(a[at]), "ji_jj_jk": [int(at[1]) + 1, int(at[0]) + 1]
                + ([int(at[2]) + 1] if a.ndim == 3 else [])}
    wet = {p: cls[p]["full"] | cls[p]["partial"] for p in "tuv"}
    stp = {side: {"|ssh|": peak(src["ssh"], wet["t"][..., 0]),
                  "|U|": peak(src["u"], wet["u"]), "|V|": peak(src["v"], wet["v"])}
           for side, src in (("nemo", ref), ("card", cand))}
    return {"kt": kt, "witness": str(path), "rows": rows,
            "first_unequal": g1.first_unequal(rows), "all_levels": levels,
            "stp_ctl_maxima": stp}


def hand_edit_cells(root: Path, card) -> dict[str, np.ndarray]:
    """Decision 120: the wet cells ORCA2's hand edits of the initial state
    would change (dtatsd.f90:219), on the card's layout; u and v faces touch
    an edited T or S cell on either side."""
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    wet = np.asarray(card.recipe.z_coord.is_active) > 0.5
    paths = (root / "data_1m_potential_temperature_nomask.nc",
             root / "data_1m_salinity_nomask.nc")
    on = recipe.build_orca2_initial_ts(*paths, wet, apply_hand_alterations=True)
    off = recipe.build_orca2_initial_ts(*paths, wet, apply_hand_alterations=False)
    t, s = (np.asarray(a) != np.asarray(b) for a, b in zip(on, off))
    ts = t | s
    return {"T": t, "S": s, "u": ts | np.roll(ts, -1, axis=1),
            "v": ts | np.concatenate([ts[1:], np.zeros_like(ts[:1])], axis=0)}


def card_levels(state) -> dict[str, np.ndarray]:
    return {"u": np.asarray(state.u.data)[:, 1:], "v": np.asarray(state.v.data)[1:],
            "T": np.asarray(state.T.data), "S": np.asarray(state.S.data)}


def card_surface(state) -> dict[str, np.ndarray]:
    lev = card_levels(state)
    return {"ssh": np.asarray(state.eta.data),
            "uu_b": np.asarray(state.uu_b.data)[:, 1:],
            "vv_b": np.asarray(state.vv_b.data)[1:, :],
            "u": lev["u"][..., 0], "v": lev["v"][..., 0],
            "T": lev["T"][..., 0], "S": lev["S"][..., 0]}


def geometry(root: Path, plant: str | None = None, case: str = CASE) -> dict:
    import netCDF4
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    g1._fp64()                 # a direct caller must not get the f32 default
    card = recipe.build_nemo_testcase_card(case, deck_root=root)
    nk = int(card.recipe.z_coord.n_levels)
    dom = recipe._orca2g1_domain(root / "domain_cfg.nc", nk + 1)
    rows_in = g1.card_geometry_rows(card, dom, all_levels=True)
    names = [r[0] for r in rows_in]
    require(plant is None or plant in names, f"unknown geometry plant {plant!r}")
    out, below = [], {}
    with netCDF4.Dataset(root / "mesh_mask.nc") as h:
        def nemo(var, shape):
            if var == "e3t_0*tmask":
                return nemo("e3t_0", shape) * nemo("tmask", shape)
            a = np.asarray(h.variables[var][0], dtype=np.float64)
            if a.ndim == 3:
                # NEMO writes the strait value into fmask on every level,
                # the dry level 31 included (dommsk.f90:293): counted
                below[var] = int(np.count_nonzero(a[nk:]))
                require(var == "fmask" or not var.endswith("mask") or not below[var],
                        f"NEMO's {var} is wet below level {nk}")
                return np.ascontiguousarray(np.moveaxis(a[:nk], 0, -1))
            if a.ndim == 1:                      # the 1-d ladder, every column
                return np.broadcast_to(a[:nk], shape).copy()
            return a
        for name, lego, var in rows_in:
            lego = np.array(lego, dtype=np.float64)
            ref = nemo(var, lego.shape)
            if plant == name:
                lego.flat[lego.size // 2] += 1.0
            require(lego.shape == ref.shape,
                    f"{name}: card {lego.shape} vs NEMO {var} {ref.shape}")
            bad = np.argwhere(lego != ref)
            out.append({
                "row": name, "nemo_variable": var, "shape": list(lego.shape),
                "bit_identical": bool(bad.shape[0] == 0),
                "n_differing": int(bad.shape[0]),
                "first_unequal_index": bad[0].tolist() if bad.size else None,
                "levels_touched": (sorted({int(b[2]) + 1 for b in bad})
                                   if bad.size and lego.ndim == 3 else []),
                "max_abs_difference": float(np.max(np.abs(lego - ref))),
                "planted": plant == name})
    return {"arm": "geometry", "case": case, "mesh_mask": str(root / "mesh_mask.nc"),
            "card_levels": nk, "rows": out, "n_rows": len(out),
            "nemo_nonzero_below_card_levels": below,
            "n_bit_identical": sum(r["bit_identical"] for r in out),
            "all_bit_identical": all(r["bit_identical"] for r in out)}


def run(root: Path, *, arm: str, kt_from: int = 1, kt_to: int = 10,
        restart_roots=(), state_in: Path | None = None,
        state_out: Path | None = None, plant: str | None = None,
        geometry_plant: str | None = None, allow_dirty: bool = False,
        entries=(1, 3, 5, 7, 9), exposures=tuple(EXPOSURES),
        dump: Path | None = None, without_zdf_faces: bool = False,
        without_tracer_rounding: bool = False,
        without_rhs_ascending_sum: bool = False,
        without_zdf_stage_r3u: bool = False,
        without_top_drag_consumer: str | None = None,
        without_stage2_stored_thickness: bool = False,
        without_stage3_salt: bool = False,
        without_hpg_fold_geometry: bool = False,
        without_vector_raw_vmask: bool = False,
        case: str = CASE) -> dict:
    require(case in CASES, f"unknown case {case!r}")
    require(arm in ARMS, f"unknown arm {arm!r}")
    require(plant in (None, "score", "level"), f"unknown plant {plant!r}")
    require(without_top_drag_consumer in (None, "residual", "substeps", "dyn_zdf"),
            f"unknown top-drag consumer {without_top_drag_consumer!r}")
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha
    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    # the model code's own identity: a resumed chain may cross commits that
    # touch documents, tests or scripts, never one that touches packages/
    packages = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "HEAD:packages"], check=True,
        capture_output=True, text=True).stdout.strip()
    g1._fp64()
    _pickle_fields()
    if arm == "geometry":
        return {**geometry(root, geometry_plant, case), "legoesm_git_sha": sha}

    import jax
    import jax.numpy as jnp
    import netCDF4
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

    card = recipe.build_nemo_testcase_card(case, deck_root=root)
    cfg, z, s0 = card.recipe.model_config, card.recipe.z_coord, card.recipe.initial_state
    if case in FORCED:
        require(arm in ("independent", "given_entry")
                or (arm in ("stages", "operators", "barotropic_exit", "carried",
                            "stp2d_terms")
                    and case in ICE_CASES),
                "G4/G5/G6 NEMO-supplied per-step forcing is registered only for "
                "the independent and given-entry ladders")
        require(cfg.rk3_stage_tracer_written_order is True,
                "the G4 card no longer selects NEMO's tracer update order")
    require(cfg.zdf_momentum_nemo_face_thickness is True,
            "the G2 card no longer selects NEMO's dyn_zdf divisors")
    if without_zdf_faces:      # ROUND-2 BEFORE ARM / plant: the two-cell mean
        cfg = cfg._replace(zdf_momentum_nemo_face_thickness=False)
    require(cfg.zdf_tracer_nemo_rounded_thickness is True,
            "the G2 card no longer selects NEMO's tra_zdf diagonal rounding")
    if without_tracer_rounding:   # ROUND-3 BEFORE ARM / plant: the fused form
        cfg = cfg._replace(zdf_tracer_nemo_rounded_thickness=False)
    require(cfg.barotropic_rhs_ascending_level_sum is True,
            "the G2 card no longer selects NEMO's ascending Ue_rhs SUM")
    if without_rhs_ascending_sum:  # ROUND-4 BEFORE ARM / plant: XLA's order
        cfg = cfg._replace(barotropic_rhs_ascending_level_sum=False)
    require(cfg.zdf_momentum_stage_r3u_nemo_literal is
            (case in ("ORCA2G3-ts", *FORCED)),
            "only the G3/G4 cards select NEMO's stored r3u(Kmm) in dyn_zdf")
    if without_zdf_stage_r3u:      # G3 ROUND-3 BEFORE ARM / plant
        cfg = cfg._replace(zdf_momentum_stage_r3u_nemo_literal=False)
    require(cfg.rk3_stage2_tracer_nemo_stored_thickness is (case in ICE_CASES)
            and cfg.rk3_stage3_salt_nemo_written is (case in ICE_CASES),
            "only the G5/G6 cards select the stage-2 e3t(Kmm) and stage-3 sfx options")
    if without_stage2_stored_thickness:   # G5 ROUND-4 BEFORE ARM
        cfg = cfg._replace(rk3_stage2_tracer_nemo_stored_thickness=False)
    if without_stage3_salt:               # G5 ROUND-4 BEFORE ARM
        cfg = cfg._replace(rk3_stage3_salt_nemo_written=False)
    if without_hpg_fold_geometry:         # G6 ROUND-2 PAIR HALF
        cfg = cfg._replace(nemo_hpg_sco_fold_geometry=False)
    if without_vector_raw_vmask:          # G6 ROUND-2 PAIR HALF
        cfg = cfg._replace(rk3_vector_raw_vmask=False)
    nk = int(z.n_levels)
    roots = (root, *restart_roots)
    reg = g1.regions(np.asarray(z.is_active)[..., 0])
    cls = depth_classes(root / "mesh_mask.nc", nk)
    # no step frames (an acquisition that stopped before the record binary
    # ran): only the restart-scored arm can run
    # G5's record build writes the ORCA2 lane's single-rank frames instead
    has_frames = (root / ("oracle_step_entry_kt00000001.bin" if case in ICE_CASES
                          else "oracle_g1step_kt00000001.bin")).is_file()
    require(has_frames or arm == "restart_independent",
            f"{root} holds no step frames; only restart_independent can run")
    if has_frames and case in ICE_CASES:
        n_stage = 0
        while (root / f"oracle_step_entry_kt{n_stage + 1:08d}.bin").is_file():
            n_stage += 1
    elif has_frames and case == "ORCA2G4-sbc":
        # G4 records the NCAR bulk-iteration operands on odd forcing steps,
        # so its otherwise complete kt=1..10 stage files alternate sizes.
        # Count the contiguous files; G1-G3's equal-size rule is not valid.
        checker = g1._checker()
        n_stage = 0
        while checker.step_path(root, n_stage + 1).is_file():
            n_stage += 1
    else:
        n_stage = g1.stage_steps(root) if has_frames else 0
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks
    consumer_hooks = _NEMOWSRK3TestHooks(
        omit_ice_top_drag_residual=without_top_drag_consumer == "residual",
        omit_ice_top_drag_substeps=without_top_drag_consumer == "substeps",
        omit_ice_top_drag_zdf=without_top_drag_consumer == "dyn_zdf",
    )
    model = LatLonCGridOceanModel(
        card.recipe.grid, z, cfg, _nemo_ws_test_hooks=consumer_hooks)
    # from rest NEMO's after-ssh slot is the initial height (restart.f90:406):
    # the model's unseeded slot
    require(s0.eta_rk3_after is None, "the card's initial after-ssh slot is seeded")
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import nemo_rk3_after_ssh_is_carried
    carries_after_ssh = nemo_rk3_after_ssh_is_carried(cfg)
    require(carries_after_ssh is (case == "ORCA2G6-vec"),
            "only the G6 card carries NEMO's after-ssh slot (Decision 132)")
    if case == "ORCA2G4-sbc":
        sys.path.insert(0, str(HERE / "nemo_testcase_g4_orca2sbc"))
        import g4_forcing_steps as g4  # noqa: E402

        def read_step(kt):
            header, fields = g4.frames(root, kt)
            return {name: g4.inner(value, header)
                    for name, value in fields.items()}
    elif case in ICE_CASES:
        sys.path.insert(0, str(HERE / "nemo_testcase_g5_orca2ice"))
        import g5_records as g5  # noqa: E402
        read_step = lambda kt: g5.read_step(root, kt, roots, nk)  # noqa: E731
    else:
        read_step = lambda kt: g1.read_step(root, kt)  # noqa: E731
    out = {"arm": arm, "case": case, "legoesm_git_sha": sha, "root": str(root),
           "restart_roots": [str(r) for r in restart_roots], "bar": BAR,
           "plant": plant, "card_levels": nk, "packages_tree": packages,
           "zdf_momentum_nemo_face_thickness": bool(
               cfg.zdf_momentum_nemo_face_thickness),
           "zdf_tracer_nemo_rounded_thickness": bool(
               cfg.zdf_tracer_nemo_rounded_thickness),
           "barotropic_rhs_ascending_level_sum": bool(
               cfg.barotropic_rhs_ascending_level_sum),
           "zdf_momentum_stage_r3u_nemo_literal": bool(
               cfg.zdf_momentum_stage_r3u_nemo_literal),
           "rk3_stage_tracer_written_order": bool(
               cfg.rk3_stage_tracer_written_order),
           "rk3_stage3_heat_nemo_written": bool(
               cfg.rk3_stage3_heat_nemo_written),
           "rk3_stage3_tracer_nemo_stored_thickness": bool(
               cfg.rk3_stage3_tracer_nemo_stored_thickness),
           "without_top_drag_consumer": without_top_drag_consumer,
           "rk3_stage2_tracer_nemo_stored_thickness": bool(
               cfg.rk3_stage2_tracer_nemo_stored_thickness),
           "rk3_stage3_salt_nemo_written": bool(cfg.rk3_stage3_salt_nemo_written),
           "nemo_hpg_sco_fold_geometry": bool(cfg.nemo_hpg_sco_fold_geometry),
           "rk3_vector_raw_vmask": bool(cfg.rk3_vector_raw_vmask),
           "forcing": ("NEMO-supplied surface fields at each kt"
                       if case == "ORCA2G4-sbc" else
                       "NEMO's post-SI3 ocean surface input at each kt"
                       if case in ICE_CASES else "none"),
           "n_barotropic_substeps": int(cfg.barotropic.n_barotropic_substeps),
           "region_cells": {k: int(m.sum()) for k, m in reg.items()},
           "depth_class_cells": {p: {k: int(m.sum()) for k, m in c.items()}
                                 for p, c in cls.items()},
           "per_kt": []}

    def surface_rows(tag, kt, state):
        g = read_step(kt)
        ref = {f: g[n] for f, n in FRAME.items()}
        if kt < n_stage:           # T, S after the step = the next step's entry
            nxt = read_step(kt + 1)
            ref["T"], ref["S"] = nxt["e_tn_k1_bb"], nxt["e_sn_k1_bb"]
            for f, n in (("ssh", "e_ssh_bb"), ("u", "e_uu_k1_bb"), ("v", "e_vv_k1_bb")):
                require(np.array_equal(ref[f], nxt[n]),
                        f"kt {kt}: after-step {f} != entry of kt {kt + 1}")
        elif (path := restart_path(roots, kt)) is not None:
            # The final admitted step has no following entry frame.  Its
            # restart is already tied to the after-step frame above, and is
            # the only recorded source for the endpoint tracer surfaces.
            with netCDF4.Dataset(path) as h:
                ref["T"] = levels_last(h.variables["tn"][0], nk)[..., 0]
                ref["S"] = levels_last(h.variables["sn"][0], nk)[..., 0]
        cand = card_surface(state)
        return [g1.row(f"{tag}.kt{kt}.{f}", ref[f], cand[f], reg,
                       plant=(plant == "score" and f == "ssh"))
                for f in ("ssh", "uu_b", "vv_b", "u", "v", "T", "S") if f in ref]

    def all_level_rows(kt, state):
        path = restart_path(roots, kt)
        if path is None:
            return None
        cand, rows = card_levels(state), {}
        with netCDF4.Dataset(path) as h:
            if has_frames:
                frame = read_step(kt)
                require(np.array_equal(np.asarray(h.variables["sshn"][0], np.float64),
                                       frame["f_ssh_bb"]),
                        f"kt {kt}: the restart is not the recorded after-step state")
            for f, var in RESTART.items():
                ref = levels_last(h.variables[var][0], nk)
                by_level = level_rows(ref, cand[f], cls[POINT[f]],
                                      plant=(plant == "level" and f == "u"))
                rows[f] = {"summary": class_summary(by_level), "levels": by_level,
                           "ref_max_abs": float(np.max(np.abs(ref))),
                           "restart": str(path)}
        return rows

    def nemo_entry(kt):
        """NEMO's state entering step kt: the restart written after kt - 1."""
        path = restart_path(roots, kt - 1)
        require(path is not None, f"no NEMO restart of step {kt - 1}")
        with netCDF4.Dataset(path) as h:
            var = lambda n: np.asarray(h.variables[n][0], np.float64)  # noqa: E731
            lev = lambda n: levels_last(var(n), nk)                     # noqa: E731
            return s0._replace(
                T=s0.T.replace(data=jnp.asarray(lev("tn"))),
                S=s0.S.replace(data=jnp.asarray(lev("sn"))),
                u=s0.u.replace(data=jnp.asarray(g1.pad_u(lev("un")))),
                v=s0.v.replace(data=jnp.asarray(g1.pad_v(lev("vn")))),
                eta=s0.eta.replace(data=jnp.asarray(var("sshn"))),
                uu_b=s0.uu_b.replace(data=jnp.asarray(g1.pad_u(var("uu_n")))),
                vv_b=s0.vv_b.replace(data=jnp.asarray(g1.pad_v(var("vv_n")))),
                bt_hist=g1.nemo_histories(path.parent, kt),
                # rst_read_ssh reads 'ssha' into ssh(Kaa) (restart.f90:367)
                eta_rk3_after=(s0.eta.replace(data=jnp.asarray(var("ssha")))
                               if carries_after_ssh else None))

    def step(state, kt):
        t0 = time.time()
        kwargs = {}
        if case in FORCED:
            freshwater, surface = (g4 if case == "ORCA2G4-sbc" else g5).nemo_supplied_forcing(
                card, root, kt)
            kwargs = {"freshwater": freshwater, "surface_forcing": surface}
        new = jax.device_get(model.step(state, dt=card.dt_s, **kwargs))
        return new, round(time.time() - t0, 1)

    if arm == "stages":
        # one step, each stage exposed by the model's private hook; the entry
        # is the card's initial state (kt = 1) or NEMO's restart (other kt)
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks
        kt = kt_from
        out["label"] = ("INDEPENDENT (the card's own chain; stage-local)"
                        if kt == 1 or state_in is not None else
                        "GIVEN-NEMO-ENTRY (NEMO's restart; stage-local)")
        g = read_step(kt)
        if state_in is not None:
            saved = pickle.loads(Path(state_in).read_bytes())
            require(saved["after_kt"] == kt - 1 and saved["packages"] == packages,
                    "--state-in is not this model code's chain after step kt - 1")
            entry = saved["state"]
        else:
            entry = s0 if kt == 1 else nemo_entry(kt)
        for name in exposures:
            hooks, frame = EXPOSURES[name]
            kwargs = {}
            if case in FORCED:
                fw, sf = (g4 if case == "ORCA2G4-sbc" else g5).nemo_supplied_forcing(
                    card, root, kt)
                kwargs = {"freshwater": fw, "surface_forcing": sf}
            st = jax.device_get(LatLonCGridOceanModel(
                card.recipe.grid, z, cfg,
                _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hooks)).step(
                    entry, dt=card.dt_s, **kwargs))
            cand = card_surface(st)
            rows = [g1.row(f"{name}.kt{kt}.{f}", g[n], cand[f], reg)
                    for f, n in frame.items() if n in g]
            levels = None
            if name.startswith("stage"):
                stage = int(name[-1])
                card_all = card_levels(st)
                levels = {}
                for f, n in (("u", "uu"), ("v", "vv"), ("T", "tn"), ("S", "sn")):
                    by_level = level_rows(
                        g[f"{stage}_{n}_all_aa"], card_all[f], cls[POINT[f]])
                    levels[f] = {"summary": class_summary(by_level), "levels": by_level}
            out["per_kt"].append({"kt": kt, "exposure": name, "rows": rows,
                                  "first_unequal": g1.first_unequal(rows),
                                  "all_levels": levels})
            if dump is not None:
                np.savez(Path(dump) / f"card_kt{kt}_{name}.npz", **card_levels(st),
                         ssh=np.asarray(st.eta.data))
        return out
    if arm == "operators":
        # G6's first multilevel vector-form statement walk.  The record is
        # source ordered: zero accumulator -> HPG -> VOR -> KEG+ZAD.  Publish
        # the production accumulators through the existing write-only hook;
        # no operator is reimplemented here.
        require(case == "ORCA2G6-vec" and kt_from == 1,
                "the operator walk is registered for G6 step 1 only")
        terms = read_stage2_terms(
            root / "oracle_rkstage2_terms_kt00000001.bin")
        require(all(not np.any(terms[name]) for name in ("before_u", "before_v")),
                "NEMO's stage-2 accumulator did not start at zero")
        out["label"] = (
            "INDEPENDENT = GIVEN-NEMO-ENTRY at kt 1 (source-ordered stage-2 "
            "momentum accumulators)"
        )
        fw, sf = g5.nemo_supplied_forcing(card, root, 1)
        boundaries = (
            ("after_hpg", "after_hpg"),
            ("after_vor", "after_vorticity"),
            ("after_zad", "after_advection"),
        )
        # Round-44 record: KEG split from ZAD, and the stage momentum ww.
        # Its three shared boundaries must equal the older record bit for bit.
        stg_path = root / "oracle_g6_stage_terms_kt00000001_s2.bin"
        stg = read_g6_stage_terms(stg_path, 1, 2) if stg_path.is_file() else None
        stg3_path = root / "oracle_g6_stage_terms_kt00000001_s3.bin"
        stg3 = read_g6_stage_terms(stg3_path, 1, 3) if stg3_path.is_file() else None
        if stg3 is not None:
            require(stg3["levels"][:3] == (1, 2, 3),
                    f"stage-3 slots {stg3['levels']} are not NEMO's (Nbb, Nnn, Naa)")
        if stg is not None:
            require(stg["levels"] == (1, 3, 2, 2),
                    f"stage-2 slots {stg['levels']} are not NEMO's (Nbb, Nnn, Naa)")
            # the raw dump keeps NEMO's dry-point and halo values; the older
            # record is l4_canon_3d (inner domain, wet umask/vmask only)
            for f in "uv":
                wet_f = np.zeros(stg[f"after_hpg_{f}"].shape, bool)
                wet_f[2:-2, 2:-2, :nk] = cls[f]["full"] | cls[f]["partial"]
                for b in G6_STGTRM_KRHS:
                    stg[f"{b}_{f}"] = np.where(wet_f, stg[f"{b}_{f}"], 0.0)
                for b, rec in boundaries:
                    require(np.array_equal(stg[f"{b}_{f}"], terms[f"{rec}_{f}"]),
                            f"{b}.{f}: the two NEMO stage-2 records disagree")
                terms[f"after_keg_{f}"] = stg[f"after_keg_{f}"]
            boundaries = boundaries[:2] + (("after_keg", "after_keg"),) + boundaries[2:]
        ww_seen, ww3_seen = {}, {}
        selected = set(exposures)
        if selected & {name for name, _ in boundaries}:
            boundaries = tuple(pair for pair in boundaries if pair[0] in selected)
        for hook_name, record_name in boundaries:
            exposed = jax.device_get(LatLonCGridOceanModel(
                card.recipe.grid, z, cfg,
                _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                    expose_momentum_operator=hook_name,
                    expose_momentum_operator_stage=2,
                    stage2_zad_operand_observer=(
                        (lambda d: ww_seen.update(w=np.asarray(d["w_consumed"]), u=np.asarray(d["u"]), h_u=np.asarray(d["h_u_consumed"])))
                        if stg is not None and hook_name == "after_zad" else None),
                    stage3_zad_operand_observer=(
                        (lambda d: ww3_seen.update(w=np.asarray(d["w_consumed"])))
                        if stg3 is not None and hook_name == "after_zad" else None),
                )).step(s0, dt=card.dt_s, freshwater=fw, surface_forcing=sf))
            jax.effects_barrier()
            cand = card_levels(exposed)
            rows, all_levels, differences = [], {}, {}
            for field in ("u", "v"):
                native = terms[f"{record_name}_{field}"]
                require(native.shape[-1] == nk + 1 and native.shape[:2] == (
                    cand[field].shape[0] + 4, cand[field].shape[1] + 4),
                    f"{field}: unexpected native stage-2 shape {native.shape}")
                ref = native[2:-2, 2:-2, :nk]
                delta = np.abs(cand[field] - ref)
                unequal = cand[field].view(np.uint64) != ref.view(np.uint64)
                first = np.argwhere(unequal)
                argmax = tuple(int(i) for i in np.unravel_index(
                    int(np.argmax(delta)), delta.shape))
                first_index = tuple(int(i) for i in first[0]) if first.size else None
                differences[field] = {
                    "n_unequal": int(np.count_nonzero(unequal)),
                    "first_index": first_index,
                    "first_nemo": (float(ref[first_index]) if first_index else None),
                    "first_card": (float(cand[field][first_index]) if first_index else None),
                    "argmax": argmax,
                    "argmax_nemo": float(ref[argmax]),
                    "argmax_card": float(cand[field][argmax]),
                    "max_abs": float(delta[argmax]),
                }
                rows.append(g1.row(
                    f"stage2.{hook_name}.{field}", ref[..., 0],
                    cand[field][..., 0], reg,
                    plant=(plant == "score" and hook_name == "after_hpg"
                           and field == "u"),
                ))
                by_level = level_rows(
                    ref, cand[field], cls[field],
                    plant=(plant == "level" and hook_name == "after_hpg"
                           and field == "u"),
                )
                all_levels[field] = {
                    "summary": class_summary(by_level), "levels": by_level,
                }
            out["per_kt"].append({
                "kt": 1, "boundary": hook_name, "rows": rows,
                "first_unequal": g1.first_unequal(rows),
                "all_levels": all_levels, "differences": differences,
            })
            if dump is not None:
                Path(dump).mkdir(parents=True, exist_ok=True)
                np.savez(Path(dump) / f"g6_stage2_{hook_name}.npz",
                         **{f"card_{name}": cand[name] for name in ("u", "v")},
                         **{f"nemo_{name}": terms[f"{record_name}_{name}"][
                             2:-2, 2:-2, :nk] for name in ("u", "v")})
        if stg is not None:
            require("w" in ww_seen, "the stage-2 dyn_zad observer never fired")
            wet_t = cls["t"]["full"] | cls["t"]["partial"]
            # level k of both is the top face of cell k
            out["stage2_ww"] = bit_rows("stage2_momentum_wzv.ww",
                                        stg["ww"][2:-2, 2:-2, :nk],
                                        ww_seen["w"][..., :nk], wet_t)
            if dump is not None:
                np.savez(Path(dump) / "g6_stage2_ww.npz", card_w=ww_seen["w"],
                         card_u=ww_seen["u"], card_h_u=ww_seen["h_u"],
                         **{f"nemo_{k}": v for k, v in stg.items() if k != "levels"})
        if stg3 is not None and "after_zad" in {h for h, _ in boundaries}:
            require("w" in ww3_seen, "the stage-3 dyn_zad observer never fired")
            out["stage3_ww"] = bit_rows("stage3_momentum_wzv.ww",
                                        stg3["ww"][2:-2, 2:-2, :nk],
                                        ww3_seen["w"][..., :nk],
                                        cls["t"]["full"] | cls["t"]["partial"])
        return out
    if arm == "stp2d_terms":
        return stp2d_terms(out, root, kt_from, kt_to, card=card, cfg=cfg, z=z,
                           s0=s0, nk=nk, cls=cls, reg=reg, nemo_entry=nemo_entry,
                           plant=plant, dump=dump)
    if arm == "carried":
        # G5 round 4: every leaf the card carries into step kt against what
        # NEMO's restart of kt - 1 hands it, then one step from swapped states
        require(case in FORCED, "the carried arm scores a forced rung's chain")
        require(state_in is not None, "carried needs --state-in")
        kt = kt_from
        saved = pickle.loads(Path(state_in).read_bytes())
        require(saved["after_kt"] == kt - 1 and saved["packages"] == packages,
                "--state-in is not this model code's chain after step kt - 1")
        carried, given = saved["state"], nemo_entry(kt)
        out["label"] = "INDEPENDENT carried state vs GIVEN-NEMO-ENTRY, swap arms"
        diff = {}
        for name in carried._fields:
            la = jax.tree_util.tree_leaves(getattr(carried, name))
            lb = jax.tree_util.tree_leaves(getattr(given, name))
            require(len(la) == len(lb), f"{name}: leaf count differs")
            for i, (a, b) in enumerate(zip(la, lb)):
                a, b = np.asarray(a), np.asarray(b)
                key = f"{name}[{i}]" if len(la) > 1 else name
                if a.shape != b.shape:
                    diff[key] = {"shape": [list(a.shape), list(b.shape)]}
                    continue
                d = a != b
                diff[key] = {"n_unequal": int(np.count_nonzero(d)),
                             "max_abs": float(np.max(np.abs(a - b))) if a.size else 0.0,
                             "cells": np.argwhere(d)[:8].tolist()}
        out["carried_vs_restart"] = diff
        nemo = ("T", "S", "u", "v", "eta", "uu_b", "vv_b", "bt_hist")
        arms = {"carried": carried, "restart": given,
                "carried_with_nemo_S": carried._replace(S=given.S),
                "restart_with_carried_S": given._replace(S=carried.S),
                "carried_with_nemo_restart_fields": carried._replace(
                    **{f: getattr(given, f) for f in nemo})}
        for name, st in arms.items():
            new, seconds = step(st, kt)
            rows = surface_rows(f"carried.{name}", kt, new)
            out["per_kt"].append({"kt": kt, "arm": name, "seconds": seconds,
                                  "rows": rows, "first_unequal": g1.first_unequal(rows),
                                  "all_levels": all_level_rows(kt, new)})
        return out
    if arm == "barotropic_exit":
        # G5 round 3 follow-on walk.  The admitted record has a compact
        # after-window uu_b/vv_b frame at every kt, but its operand-complete
        # external-mode stream only at kt=1.  Expose the ordinary solver's
        # exit without changing it, so a post-kt1 card chain can locate a
        # later leave on the barotropic side of the stage-1 handoff.
        require(case in ICE_CASES, "barotropic_exit is a G5/G6-only arm")
        require(state_in is not None, "barotropic_exit needs --state-in")
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks
        from nemo_testcase_l1_vortex_kt2_walk import read_bt_frame
        saved = pickle.loads(Path(state_in).read_bytes())
        require(saved["after_kt"] == kt_from - 1 and saved["packages"] == packages,
                "--state-in is not this model code's chain after step kt - 1")
        fw, sf = g5.nemo_supplied_forcing(card, root, kt_from)
        result = jax.device_get(LatLonCGridOceanModel(
            card.recipe.grid, z, cfg,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_barotropic_substeps=True)).step(
                    saved["state"], dt=card.dt_s, freshwater=fw,
                    surface_forcing=sf))
        ref = read_bt_frame(
            root / f"oracle_bt_frames_kt{kt_from:08d}.bin",
            expect_step=kt_from)
        cand = {"uu_b": np.asarray(result.state_after_barotropic.uu_b.data)[:, 1:],
                "vv_b": np.asarray(result.state_after_barotropic.vv_b.data)[1:, :]}
        rows = [g1.row(f"barotropic_exit.kt{kt_from}.{name}", ref[name],
                       cand[name], reg,
                       plant=(plant == "score" and name == "uu_b"))
                for name in ("uu_b", "vv_b")]
        out.update({"label": "INDEPENDENT (card chain; barotropic exit)",
                    "kt": kt_from, "rows": rows,
                    "first_unequal": g1.first_unequal(rows)})
        return out
    if arm == "substeps":
        # the split-explicit solver's loop at step 1 (the exact initial state
        # is its entry): every recorded operand in compiled order against the
        # model's write-only trace.  Reader and table are G1's.
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks
        require(kt_from == 1, "the G2 substep walk starts from the initial state")
        out["label"] = "INDEPENDENT = GIVEN-NEMO-ENTRY at step 1 (solver substep loop)"
        head, sub = g1.read_substep_record(root, 1)
        n_sub = head["icycle"]
        require(n_sub == int(cfg.barotropic.n_barotropic_substeps),
                "NEMO and the card ran different substep counts")
        res = jax.device_get(LatLonCGridOceanModel(
            card.recipe.grid, z, cfg, _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_barotropic_substeps=True)).step(s0, dt=card.dt_s))
        trace = {k: np.asarray(v) for k, v in res.substeps.items()}
        raw = z.nemo_een_barotropic
        wet = {"t": np.asarray(z.is_active)[..., 0] > 0.5,
               "u": np.asarray(raw.umask)[..., 0] > 0,
               "v": np.asarray(raw.vmask)[..., 0] > 0}
        table = g1.substep_table(sub, trace, n_sub, wet, reg, plant)
        # passive: the traced loop's exit is the ordinary stage-1 handoff
        plain = card_surface(jax.device_get(LatLonCGridOceanModel(
            card.recipe.grid, z, cfg, _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_momentum_stage=1, expose_tracer_stage=1)).step(
                    s0, dt=card.dt_s)))
        traced = card_surface(res.state_after_barotropic)
        # ... and both must sit at the same distance from NEMO's exit frame
        out["trace_passive"] = {f: {
            "ordinary_vs_traced_unequal": int(np.count_nonzero(plain[f] != traced[f])),
            "ordinary_vs_traced_max_abs": float(np.max(np.abs(plain[f] - traced[f]))),
            "ordinary_vs_nemo_unequal": int(np.count_nonzero(
                (plain[f] != sub["o000_" + nm]) & wet[sg])),
            "traced_vs_nemo_unequal": int(np.count_nonzero(
                (traced[f] != sub["o000_" + nm]) & wet[sg]))}
            for f, nm, sg in (("ssh", "ssh_aa", "t"), ("uu_b", "uu_b_aa", "u"),
                              ("vv_b", "vv_b_aa", "v"))}
        out.update({"kt": 1, "n_substeps": n_sub, **table})
        return out
    if arm == "rhs_frame":
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks
        require(kt_from == 1, "the right-hand-side frame is scored at step 1")
        out["label"] = "INDEPENDENT = GIVEN-NEMO-ENTRY at step 1 (3-D RHS after stp_2D)"
        g = read_step(1)
        res = jax.device_get(LatLonCGridOceanModel(
            card.recipe.grid, z, cfg, _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                expose_barotropic_substeps=True)).step(s0, dt=card.dt_s))
        ops = {k: np.asarray(v) for k, v in res.slow_forcing_operands.items()}
        rows = {}
        with netCDF4.Dataset(root / "mesh_mask.nc") as h:
            for f, cut, e3n, mn in (("u", lambda a: a[:, 1:], "e3u_0", "umask"),
                                    ("v", lambda a: a[1:], "e3v_0", "vmask")):
                rhs, card_mean = cut(ops[f"d{f}_dt"]), cut(ops[f"depth_{f}"])
                mask = levels_last(h.variables[mn][0], nk)
                e3 = np.ascontiguousarray(np.moveaxis(np.asarray(
                    h.variables[e3n][0], np.float64)[:nk], 0, -1))
                ref, wet = g[f"b_{f}{f}_rhs_k1"], mask[..., 0] > 0
                # the top V row included: the fold unit sums NEMO's raw mesh
                # operands, which are the mesh file's
                scored = wet
                rows[f] = {
                    "level1_wet_faces": int(wet.sum()),
                    "level1_wet_unequal": int(np.count_nonzero((rhs[..., 0] != ref) & wet)),
                    "level1_dry_unequal": int(np.count_nonzero((rhs[..., 0] != ref) & ~wet)),
                    "level1_dry_card_nonzero": int(np.count_nonzero((rhs[..., 0] != 0) & ~wet)),
                    "level1_dry_nemo_max_abs": float(np.max(np.abs(ref[~wet]))),
                    "depth_mean_faces": int(scored.sum()),
                    "ascending_replay_vs_card_unequal": int(np.count_nonzero(
                        (nemo_ue_rhs_replay(rhs, e3, mask) != card_mean) & scored)),
                    "lane4_replay_vs_card_unequal": int(np.count_nonzero(
                        (nemo_ue_rhs_replay(rhs, e3, mask, lanes=4) != card_mean) & scored))}
        out["per_kt"].append({"kt": 1, "rhs_frame": rows})
        return out
    if arm == "trazdf_replay":
        require(kt_from == 1, "the tracer replay starts from the initial state")
        out["label"] = ("OFFLINE replay of NEMO's tra_zdf from NEMO's recorded "
                        "ratios; the card from its own initial state (step 1)")
        g = read_step(1)
        with netCDF4.Dataset(root / "mesh_mask.nc") as h:
            geom = dict(
                tmask=levels_last(h.variables["tmask"][0], nk),
                # NEMO's e3t_0 is nonzero on its level 31; the card has 30
                e3t_0=np.ascontiguousarray(np.moveaxis(np.asarray(
                    h.variables["e3t_0"][0], np.float64)[:nk], 0, -1)),
                e3w_1d=np.asarray(h.variables["e3w_1d"][0], np.float64),
                avt0=float(recipe.read_orca2g1_deck(root).rn_avt0_m2_s),
                r3t_bb=g["e_r3t_bb"], r3t_mm=g["2_r3t_aa"], r3t_aa=g["3_r3t_aa"],
                dt=card.dt_s)
        cand, rows = card_levels(step(s0, 1)[0]), {}
        for f in ("T", "S"):
            name = f"3_{f.lower()}n_k1_aa"
            rep = nemo_trazdf_replay(np.asarray(getattr(s0, f).data, np.float64), **geom)
            by_level = level_rows(rep, cand[f], cls["t"])
            rows[f] = {
                "level1_replay_vs_nemo": g1.row(
                    f"trazdf_replay.kt1.{f}", g[name], rep[..., 0], reg),
                "level1_card_vs_nemo": g1.row(
                    f"trazdf_replay.card.kt1.{f}", g[name], cand[f][..., 0], reg),
                "all_levels_card_vs_replay": class_summary(by_level),
                "unequal_by_level": [r["full"]["n_unequal"] + r["partial"]["n_unequal"]
                                     for r in by_level]}
        out["per_kt"].append({"kt": 1, "trazdf_replay": rows})
        return out
    if arm == "zdf_replay":
        require(case != "ORCA2G7-drg",
                "the dyn_zdf replay transcribes zero drag; G7 has bottom drag")
        # OFFLINE replay (rule 5): the card's own passive stage-3 arrays, from
        # its write-only observer, are pushed through NEMO's statements.
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks
        kt, seen = kt_from, []
        out["label"] = ("OFFLINE replay of NEMO's dyn_zdf from the card's own "
                        "stage-3 inputs; " + ("INDEPENDENT entry" if kt == 1
                                              else "GIVEN-NEMO-ENTRY"))
        g, entry = read_step(kt), s0 if kt == 1 else nemo_entry(kt)
        st = jax.device_get(LatLonCGridOceanModel(
            card.recipe.grid, z, cfg, _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(
                zdf_momentum_observer=lambda *a: seen.append(
                    [np.array(x) for x in a]))).step(entry, dt=card.dt_s))
        require(len(seen) == 1, f"the observer fired {len(seen)} times, not once")
        plain = jax.device_get(model.step(entry, dt=card.dt_s))
        final = card_levels(st)
        out["observer_is_passive"] = all(
            np.array_equal(final[f], card_levels(plain)[f]) for f in final)
        require(out["observer_is_passive"], "the observer moved the result")
        with netCDF4.Dataset(root / "mesh_mask.nc") as h:
            lev = lambda n: levels_last(h.variables[n][0], nk)   # noqa: E731
            tmask = lev("tmask")
            e3w = np.asarray(h.variables["e3w_1d"][0], np.float64)
            wmask_t = tmask.copy()
            wmask_t[..., 1:] = tmask[..., 1:] * tmask[..., :-1]
            # zdfphy.f90:228,350 with nn_avb = 0: avm = rn_avm0 * wmask
            avm = float(recipe.read_orca2g1_deck(root).rn_avm0_m2_s) * wmask_t
            rows = {}
            for f, col, e3n, mn, pair in (
                    ("u", 0, "e3u_0", "umask", avm + np.roll(avm, -1, axis=1)),
                    ("v", 1, "e3v_0", "vmask",
                     avm + np.concatenate([avm[1:], avm[-1:]], axis=0))):
                cut = (lambda a: a[:, 1:]) if f == "u" else (lambda a: a[1:])
                explicit, solved = cut(seen[0][col]), cut(seen[0][6 + col])
                e3_0, mask = np.ascontiguousarray(np.moveaxis(np.asarray(
                    h.variables[e3n][0], np.float64)[:nk], 0, -1)), lev(mn)
                h0 = np.zeros(mask.shape[:2])
                for k in range(nk):                     # domain.f90:199
                    h0 = h0 + e3_0[..., k] * mask[..., k]
                ss = mask[..., 0]
                r1_h0 = ss / (h0 + 1.0 - ss)            # domain.f90:213
                keep = np.ones(mask.shape[:2], bool)
                if f == "v":
                    keep[-1] = False     # the top V row's north T row is a fold halo
                per = {}
                for variant in ("r3_mm", "r3_aa", "none"):
                    after_solve, after = nemo_dynzdf_replay(
                        explicit, g[f"3_{f}{f}_b_aa"], e3_0=e3_0, mask=mask,
                        avm_pair=pair, r3_aa=g[f"3_r3{f}_aa"], r3_mm=g[f"2_r3{f}_aa"],
                        r1_h0=r1_h0, e3w_1d=e3w, dt=card.dt_s, interface=variant)
                    ref1 = np.where(keep, g[f"3_{f}{f}_k1_aa"], 0.0)
                    per[variant] = {
                        "level1_replay_vs_nemo": g1.row(
                            f"zdf_replay.{variant}.kt{kt}.{f}", ref1,
                            np.where(keep, after[..., 0], 0.0), reg),
                        "all_levels_card_vs_replay": class_summary(level_rows(
                            after * keep[..., None], final[f] * keep[..., None],
                            cls[f])),
                        "all_levels_card_solve_vs_replay_solve": class_summary(
                            level_rows(after_solve * keep[..., None],
                                       solved * keep[..., None], cls[f]))}
                per["level1_card_vs_nemo"] = g1.row(
                    f"zdf_replay.card.kt{kt}.{f}", g[f"3_{f}{f}_k1_aa"],
                    final[f][..., 0], reg)
                rows[f] = per
        out["per_kt"].append({"kt": kt, "zdf_replay": rows})
        if dump is not None:
            np.savez(Path(dump) / f"card_kt{kt}_zdf_observer.npz",
                     **{f"a{i}": a for i, a in enumerate(seen[0])})
        return out
    if arm == "restart_independent":
        # the card's own chain, scored ONLY where a NEMO restart exists: the
        # 2-D fields and level 1 by region, every level by depth class
        out["label"] = "INDEPENDENT (restart-scored; no step frames)"
        require((kt_from == 1) == (state_in is None),
                "a chain starts at step 1 or resumes from --state-in")
        edits = hand_edit_cells(root, card) if case == "ORCA2G3-ts" else None
        state = s0
        if state_in is not None:
            saved = pickle.loads(Path(state_in).read_bytes())
            require(saved["after_kt"] == kt_from - 1 and saved["packages"] == packages,
                    "--state-in is not this model code's chain after step kt_from - 1")
            state = saved["state"]
        for kt in range(kt_from, kt_to + 1):
            state, seconds = step(state, kt)
            path = restart_path(roots, kt)
            if path is None:
                stop = abort_witness(roots, kt)
                if stop is not None:
                    out["per_kt"].append(abort_rows(stop, kt, state, nk, reg, cls))
                continue
            cand = card_surface(state)
            with netCDF4.Dataset(path) as h:
                var = lambda n: np.asarray(h.variables[n][0], np.float64)  # noqa: E731
                ref = {"ssh": var("sshn"), "uu_b": var("uu_n"), "vv_b": var("vv_n"),
                       **{f: levels_last(var(n), nk)[..., 0] for f, n in RESTART.items()}}
            rows = [g1.row(f"restart.kt{kt}.{f}", ref[f], cand[f], reg,
                           plant=(plant == "score" and f == "ssh"))
                    for f in ("ssh", "uu_b", "vv_b", "u", "v", "T", "S")]
            levels = all_level_rows(kt, state)
            if edits is not None:
                lev = card_levels(state)
                with netCDF4.Dataset(path) as h:
                    for f, var in RESTART.items():
                        d = levels_last(h.variables[var][0], nk) != lev[f]
                        levels[f]["hand_edit_cells"] = int(edits[f].sum())
                        levels[f]["hand_edit_unequal"] = int(np.count_nonzero(d & edits[f]))
                # P3: an unequal T/S cell in a column whose sea level is equal
                        if f in ("T", "S"):
                            levels[f]["unequal_in_equal_ssh_columns"] = int(
                                np.count_nonzero(d & (cand["ssh"] == ref["ssh"])[..., None]))
            out["per_kt"].append({"kt": kt, "seconds": seconds, "restart": str(path),
                                  "rows": rows, "first_unequal": g1.first_unequal(rows),
                                  "all_levels": levels,
                                  "history_rows": g1.history_rows(path.parent, kt, state, reg)})
        require(bool(out["per_kt"]), f"no NEMO restart or abort witness at kt {kt_from}..{kt_to}")
        if state_out is not None:
            Path(state_out).write_bytes(pickle.dumps(
                {"after_kt": kt_to, "packages": packages, "state": state}))
        return out
    if arm == "independent":
        out["label"] = "INDEPENDENT"
        require(1 <= kt_from <= kt_to <= 100, "steps must lie in 1..100")
        require((kt_from == 1) == (state_in is None),
                "a chain starts at step 1 or resumes from --state-in")
        state = s0
        if state_in is not None:
            saved = pickle.loads(Path(state_in).read_bytes())
            require(saved["after_kt"] == kt_from - 1 and saved["packages"] == packages,
                    "--state-in is not this model code's chain after step kt_from - 1")
            state = saved["state"]
        for kt in range(kt_from, kt_to + 1):
            state, seconds = step(state, kt)
            rows = surface_rows("independent", kt, state)
            out["per_kt"].append({"kt": kt, "seconds": seconds, "rows": rows,
                                  "first_unequal": g1.first_unequal(rows),
                                  "all_levels": all_level_rows(kt, state)})
            if kt % 10 == 0:
                at = restart_path(roots, kt)
                if at is not None:
                    out.setdefault("history_rows", []).extend(
                        g1.history_rows(at.parent, kt, state, reg))
        if state_out is not None:
            Path(state_out).write_bytes(pickle.dumps(
                {"after_kt": kt_to, "packages": packages, "state": state}))
    else:
        out["label"] = ("GIVEN-NEMO-ENTRY (NEMO's restart of step kt - 1: all "
                        "prognostics on all levels and the six barotropic "
                        "histories; w is the card's rest value; G4 uses "
                        "NEMO's surface fields at each kt)")
        for kt in entries:
            first, s1 = step(s0 if kt == 1 else nemo_entry(kt), kt)
            rows = surface_rows("given_entry", kt, first)
            result = {"kt": kt, "seconds": [s1], "rows": rows,
                      "first_unequal": g1.first_unequal(rows),
                      "all_levels": all_level_rows(kt, first)}
            # The G6 acquisition completed kt=7 but its certification record
            # ends at kt=6 (nn_fsbc=2).  Do not manufacture an unrecorded kt=7
            # two-step row merely because kt=6 is a requested entry point.
            if kt < n_stage:
                second, s2 = step(first, kt + 1)
                result.update({
                    "seconds": [s1, s2], "two_step_kt": kt + 1,
                    "two_step_rows": surface_rows(
                        "given_entry_two_step", kt + 1, second),
                    "two_step_all_levels": all_level_rows(kt + 1, second)})
            out["per_kt"].append(result)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--restart-root", type=Path, action="append", default=[])
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--kt-from", type=int, default=1)
    ap.add_argument("--kt-to", type=int, default=10)
    ap.add_argument("--entries", default="1,3,5,7,9")
    ap.add_argument("--exposures", default=",".join(EXPOSURES))
    ap.add_argument("--dump", type=Path, help="directory for the card's "
                    "all-level stage arrays (evidence, never git)")
    ap.add_argument("--state-in", type=Path)
    ap.add_argument("--state-out", type=Path)
    ap.add_argument("--plant", choices=("score", "level"))
    ap.add_argument("--geometry-plant")
    ap.add_argument("--without-zdf-faces", action="store_true",
                    help="BEFORE ARM: the card without NEMO's dyn_zdf divisors")
    ap.add_argument("--without-tracer-rounding", action="store_true",
                    help="BEFORE ARM: the card without NEMO's tra_zdf diagonal rounding")
    ap.add_argument("--without-rhs-ascending-sum", action="store_true",
                    help="BEFORE ARM: the card's Ue_rhs depth sum in XLA's order")
    ap.add_argument("--without-zdf-stage-r3u", action="store_true",
                    help="BEFORE ARM: dyn_zdf's r3u(Kmm) as the ratio of the mean ssh")
    ap.add_argument("--without-top-drag-consumer",
                    choices=("residual", "substeps", "dyn_zdf"),
                    help="G5 CONTROL: omit exactly one downstream top-drag consumer")
    ap.add_argument("--without-stage2-stored-thickness", action="store_true",
                    help="G5 round-4 before arm: stage-2 thickness interpolant")
    ap.add_argument("--without-stage3-salt", action="store_true",
                    help="G5 round-4 before arm: rescaled Kbb salt tendency")
    ap.add_argument("--without-hpg-fold-geometry", action="store_true",
                    help="G6 round-2 pair half: cyclic north HPG geometry")
    ap.add_argument("--without-vector-raw-vmask", action="store_true",
                    help="G6 round-2 pair half: compact vector V mask")
    ap.add_argument("--case", choices=CASES, default=CASE)
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args(argv)
    out = run(a.root, arm=a.arm, kt_from=a.kt_from, kt_to=a.kt_to,
              restart_roots=tuple(a.restart_root), state_in=a.state_in,
              state_out=a.state_out, plant=a.plant,
              geometry_plant=a.geometry_plant, allow_dirty=a.allow_dirty,
              entries=tuple(int(k) for k in a.entries.split(",")),
              exposures=tuple(a.exposures.split(",")), dump=a.dump,
              without_zdf_faces=a.without_zdf_faces,
              without_tracer_rounding=a.without_tracer_rounding,
              without_rhs_ascending_sum=a.without_rhs_ascending_sum,
              without_zdf_stage_r3u=a.without_zdf_stage_r3u,
              without_top_drag_consumer=a.without_top_drag_consumer,
              without_stage2_stored_thickness=a.without_stage2_stored_thickness,
              without_stage3_salt=a.without_stage3_salt,
              without_hpg_fold_geometry=a.without_hpg_fold_geometry,
              without_vector_raw_vmask=a.without_vector_raw_vmask,
              case=a.case)
    a.output.write_text(json.dumps(out, indent=1) + "\n")
    if a.arm == "geometry":
        print(f"GEOMETRY {out['n_bit_identical']} of {out['n_rows']} rows bit-identical")
        for r in out["rows"]:
            if not r["bit_identical"]:
                print("  UNEQUAL", r["row"], r["n_differing"], r["max_abs_difference"],
                      r["first_unequal_index"], r["levels_touched"][:8])
        return 0 if out["all_bit_identical"] else 1
    if a.arm == "zdf_replay":
        for f, per in out["per_kt"][0]["zdf_replay"].items():
            c = per["level1_card_vs_nemo"]
            print(f"{f} card vs NEMO level 1: {c['status']} max {c['max_abs']:.3e} "
                  f"unequal {c['n_unequal']}")
            for v in ("r3_mm", "r3_aa", "none"):
                r, s2 = per[v]["level1_replay_vs_nemo"], per[v]["all_levels_card_vs_replay"]
                s3 = per[v]["all_levels_card_solve_vs_replay_solve"]
                print(f"  replay[{v}] vs NEMO level 1: {r['status']} max {r['max_abs']:.3e} "
                      f"unequal {r['n_unequal']} | card vs replay all levels: full "
                      f"{s2['full']['max_abs']:.2e} partial {s2['partial']['max_abs']:.2e}"
                      f" | solve only: full {s3['full']['max_abs']:.2e} partial "
                      f"{s3['partial']['max_abs']:.2e}")
        return 0
    if a.arm == "trazdf_replay":
        bad = False
        for f, per in out["per_kt"][0]["trazdf_replay"].items():
            r, c, s2 = (per["level1_replay_vs_nemo"], per["level1_card_vs_nemo"],
                        per["all_levels_card_vs_replay"])
            n = s2["full"]["n_unequal"] + s2["partial"]["n_unequal"]
            bad |= bool(r["n_unequal"] or c["n_unequal"] or n or s2["dry_n_unequal"])
            print(f"{f} replay vs NEMO level 1: unequal {r['n_unequal']} | card vs NEMO "
                  f"level 1: unequal {c['n_unequal']} max {c['max_abs']:.3e} | card vs "
                  f"replay all levels: unequal {n} max "
                  f"{max(s2['full']['max_abs'], s2['partial']['max_abs']):.3e}")
        return 1 if bad else 0
    if a.arm == "rhs_frame":
        bad = False
        for f, r in out["per_kt"][0]["rhs_frame"].items():
            bad |= bool(r["level1_wet_unequal"] or r["level1_dry_card_nonzero"]
                        or r["ascending_replay_vs_card_unequal"])
            print(f"{f} RHS level 1 vs NEMO: wet unequal {r['level1_wet_unequal']} of "
                  f"{r['level1_wet_faces']}, dry unequal {r['level1_dry_unequal']} (NEMO "
                  f"max {r['level1_dry_nemo_max_abs']:.2e}) | depth mean: ascending "
                  f"replay unequal {r['ascending_replay_vs_card_unequal']}, 4-lane "
                  f"{r['lane4_replay_vs_card_unequal']}")
        return 1 if bad else 0
    if a.arm == "stp2d_terms":
        for p in out["per_kt"]:
            print(f"kt {p['kt']} ({p['label']}): first unequal {p['first_unequal']}")
            for r in p["stp2d_rows"]:
                print(f"  {r['name']:24s} unequal {r['n_unequal']:6d}/{r['n_wet']:6d} "
                      f"max {r['max_abs']:.3e} norm {r['normalised']:.2e} "
                      f"ulps {r['max_ulps']} first {r['first_cell']} "
                      f"dry card/nemo {r['dry_card_nonzero']}/{r['dry_nemo_nonzero']}")
        return 0
    if a.arm == "substeps":
        f = out["first_unequal"]
        print("SUBSTEPS first unequal operand:", f and (f["name"], f["n_unequal"]))
        print("  rows unequal:", sorted({r["operand"] for r in out["rows"]
                                         if not r["bit_identical"]}))
        for k, v in out["trace_passive"].items():
            print(f"  {k}: traced exit vs NEMO unequal {v['traced_vs_nemo_unequal']}, "
                  f"ordinary handoff vs NEMO unequal {v['ordinary_vs_nemo_unequal']}")
        return 0
    bad = False
    if "stage2_ww" in out:
        r = out["stage2_ww"]
        print(f"{r['name']:34s} unequal {r['n_unequal']}/{r['n_wet']} max {r['max_abs']:.3e} "
              f"norm {r['normalised']:.2e} ulps {r['max_ulps']} first {r['first_cell']}")
    for p in out["per_kt"]:
        for r in p["rows"] + p.get("two_step_rows", []) + p.get("history_rows", []):
            bad |= r["status"] != "AT-BAR"
            print(f"{r['name']:34s} {r['status']:6s} max {r['max_abs']:.3e} "
                  f"rms {r['rms']:.3e} unequal {r['n_unequal']}")
        lv = p.get("all_levels") or p.get("two_step_all_levels")
        for f, t in (lv or {}).items():
            s = t["summary"]
            bad |= s["status"] != "AT-BAR"
            print(f"  levels kt{p.get('two_step_kt', p['kt'])} {f}: {s['status']} "
                  f"full {s['full']['relative']:.2e} ({s['full']['n_unequal']}) "
                  f"partial {s['partial']['relative']:.2e} ({s['partial']['n_unequal']}) "
                  f"dry unequal {s['dry_n_unequal']}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
