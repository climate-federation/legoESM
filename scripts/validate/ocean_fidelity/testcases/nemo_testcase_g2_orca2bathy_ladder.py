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
               prognostics on all levels and the six barotropic histories),
               then one step (level 1 and 2-D rows) and a second step
               (all-level rows against NEMO's next restart: TWO-STEP rows).
  substeps     step 1: the split-explicit solver's loop, every recorded
               operand in compiled order against the model's write-only
               trace (G1's reader and table), and the traced loop's exit and
               the ordinary step's handoff each against NEMO's exit frame.
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
DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round10/"
    "oracle_g2_r10/p3_a_100step")
ARMS = ("geometry", "independent", "given_entry", "stages", "zdf_replay",
        "substeps", "trazdf_replay")
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


def geometry(root: Path, plant: str | None = None) -> dict:
    import netCDF4
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    g1._fp64()                 # a direct caller must not get the f32 default
    card = recipe.build_nemo_testcase_card(CASE, deck_root=root)
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
    return {"arm": "geometry", "case": CASE, "mesh_mask": str(root / "mesh_mask.nc"),
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
        without_rhs_ascending_sum: bool = False) -> dict:
    require(arm in ARMS, f"unknown arm {arm!r}")
    require(plant in (None, "score", "level"), f"unknown plant {plant!r}")
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
        return {**geometry(root, geometry_plant), "legoesm_git_sha": sha}

    import jax
    import jax.numpy as jnp
    import netCDF4
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

    card = recipe.build_nemo_testcase_card(CASE, deck_root=root)
    cfg, z, s0 = card.recipe.model_config, card.recipe.z_coord, card.recipe.initial_state
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
    nk = int(z.n_levels)
    roots = (root, *restart_roots)
    reg = g1.regions(np.asarray(z.is_active)[..., 0])
    cls = depth_classes(root / "mesh_mask.nc", nk)
    n_stage = g1.stage_steps(root)
    model = LatLonCGridOceanModel(card.recipe.grid, z, cfg)
    require(s0.eta_rk3_after is None, "the card carries an after-ssh slot")
    out = {"arm": arm, "case": CASE, "legoesm_git_sha": sha, "root": str(root),
           "restart_roots": [str(r) for r in restart_roots], "bar": BAR,
           "plant": plant, "card_levels": nk, "packages_tree": packages,
           "zdf_momentum_nemo_face_thickness": bool(
               cfg.zdf_momentum_nemo_face_thickness),
           "zdf_tracer_nemo_rounded_thickness": bool(
               cfg.zdf_tracer_nemo_rounded_thickness),
           "barotropic_rhs_ascending_level_sum": bool(
               cfg.barotropic_rhs_ascending_level_sum),
           "n_barotropic_substeps": int(cfg.barotropic.n_barotropic_substeps),
           "region_cells": {k: int(m.sum()) for k, m in reg.items()},
           "depth_class_cells": {p: {k: int(m.sum()) for k, m in c.items()}
                                 for p, c in cls.items()},
           "per_kt": []}

    def surface_rows(tag, kt, state):
        g = g1.read_step(root, kt)
        ref = {f: g[n] for f, n in FRAME.items()}
        if kt < n_stage:           # T, S after the step = the next step's entry
            nxt = g1.read_step(root, kt + 1)
            ref["T"], ref["S"] = nxt["e_tn_k1_bb"], nxt["e_sn_k1_bb"]
            for f, n in (("ssh", "e_ssh_bb"), ("u", "e_uu_k1_bb"), ("v", "e_vv_k1_bb")):
                require(np.array_equal(ref[f], nxt[n]),
                        f"kt {kt}: after-step {f} != entry of kt {kt + 1}")
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
            frame = g1.read_step(root, kt)
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
                bt_hist=g1.nemo_histories(path.parent, kt))

    def step(state):
        t0 = time.time()
        new = jax.device_get(model.step(state, dt=card.dt_s))
        return new, round(time.time() - t0, 1)

    if arm == "stages":
        # one step, each stage exposed by the model's private hook; the entry
        # is the card's initial state (kt = 1) or NEMO's restart (other kt)
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks
        kt = kt_from
        out["label"] = ("INDEPENDENT (the card's own chain; stage-local)"
                        if kt == 1 or state_in is not None else
                        "GIVEN-NEMO-ENTRY (NEMO's restart; stage-local)")
        g = g1.read_step(root, kt)
        if state_in is not None:
            saved = pickle.loads(Path(state_in).read_bytes())
            require(saved["after_kt"] == kt - 1 and saved["packages"] == packages,
                    "--state-in is not this model code's chain after step kt - 1")
            entry = saved["state"]
        else:
            entry = s0 if kt == 1 else nemo_entry(kt)
        for name in exposures:
            hooks, frame = EXPOSURES[name]
            st = jax.device_get(LatLonCGridOceanModel(
                card.recipe.grid, z, cfg,
                _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hooks)).step(
                    entry, dt=card.dt_s))
            cand = card_surface(st)
            rows = [g1.row(f"{name}.kt{kt}.{f}", g[n], cand[f], reg)
                    for f, n in frame.items()]
            out["per_kt"].append({"kt": kt, "exposure": name, "rows": rows,
                                  "first_unequal": g1.first_unequal(rows)})
            if dump is not None:
                np.savez(Path(dump) / f"card_kt{kt}_{name}.npz", **card_levels(st),
                         ssh=np.asarray(st.eta.data))
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
    if arm == "trazdf_replay":
        require(kt_from == 1, "the tracer replay starts from the initial state")
        out["label"] = ("OFFLINE replay of NEMO's tra_zdf from NEMO's recorded "
                        "ratios; the card from its own initial state (step 1)")
        g = g1.read_step(root, 1)
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
        cand, rows = card_levels(step(s0)[0]), {}
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
        # OFFLINE replay (rule 5): the card's own passive stage-3 arrays, from
        # its write-only observer, are pushed through NEMO's statements.
        from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import _NEMOWSRK3TestHooks
        kt, seen = kt_from, []
        out["label"] = ("OFFLINE replay of NEMO's dyn_zdf from the card's own "
                        "stage-3 inputs; " + ("INDEPENDENT entry" if kt == 1
                                              else "GIVEN-NEMO-ENTRY"))
        g, entry = g1.read_step(root, kt), s0 if kt == 1 else nemo_entry(kt)
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
            state, seconds = step(state)
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
                        "histories; w is the card's rest value)")
        for kt in entries:
            first, s1 = step(s0 if kt == 1 else nemo_entry(kt))
            rows = surface_rows("given_entry", kt, first)
            second, s2 = step(first)
            out["per_kt"].append({
                "kt": kt, "seconds": [s1, s2], "rows": rows,
                "first_unequal": g1.first_unequal(rows),
                "two_step_kt": kt + 1,
                "two_step_rows": surface_rows("given_entry_two_step", kt + 1, second),
                "two_step_all_levels": all_level_rows(kt + 1, second)})
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
              without_rhs_ascending_sum=a.without_rhs_ascending_sum)
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
    for p in out["per_kt"]:
        for r in p["rows"] + p.get("two_step_rows", []):
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
