#!/usr/bin/env python3
"""Grid ladder G1 -- geometry identity and the ladders against NEMO's record.

Two labels, never mixed (Decision 52):
  INDEPENDENT      the card's step chained from the card's own initial state;
  GIVEN-NEMO-ENTRY the card handed NEMO's recorded entry state of each step
                   (ssh, uu_b, vv_b, u, v, T, S).  The step record does NOT
                   hold the six barotropic histories NEMO carries between
                   steps (dynspg_ts.f90:799-805 reads them, :817-822 writes them),
                   so those, and every other unrecorded carried field, come
                   from the card's own chain at that kt.  Seeding the initial
                   state's empty history at kt >= 2 instead would run the
                   cold-start ramp on a moving ocean: an instrument error.

Arms (``--arm``):
  geometry     every mesh array the card carries vs NEMO's mesh_mask.nc (gate)
  independent  kt = 1..kt-max (up to 100) against the after-step frames
  given_entry  kt = 1..10, whole step from NEMO's entry
  rhs          kt = 1..10, the stp_2D right-hand side
  handoff      kt = 1..10, the external handoff read at stage 1
  stages       kt = 1..10, each stage from NEMO's handoff and stage entry
  restart_entry  ONE step from NEMO's restart state at kt = 10, 20, .., 90:
               every prognostic field and the six barotropic histories are
               NEMO's, so no evolving field comes from the card's chain.
               NOT seeded: NEMO's after-ssh slot (restart.f90:184, :362-370);
               the card's ``nemo_first_wzv_after_ssh`` selection does not read
               the carried slot, and the arm refuses a card that does
  rhs_terms    kt = 1..10, the stp_2D right-hand side term by term: the
               card's pressure-gradient and vorticity terms (the model's own
               WRITE-only per-term observer) against the OFFLINE replay of
               NEMO's two statements from NEMO's entry frame
               (rhs_term_replay); the replay's sum against NEMO's recorded
               right-hand side is the control and must be 0 unequal
  vor_fold_replay  OFFLINE numpy replay of NEMO's vor_een Coriolis statement
               (dynvor.f90:720, :727, :767, :772-775, :779-784) on the U faces
               of the pivot row and the row below, from NEMO's recorded entry
               fields with their halos: once as compiled, once with the
               fold-image V row's transport set to zero
  fold_v_depth  OFFLINE operand table, kt = 1..10: the split-explicit solver's
               sea level on V faces (the ssh term of zhvp2_e,
               dynspg_ts.f90:419-421) from the card's production helper applied
               to NEMO's entry ssh, on the fold-image V row and the row below,
               against NEMO's own stored weighted mean (r3v, domqco.f90:268-269,
               times hv_0); no model step
  stage_replay  OFFLINE numpy replay, from NEMO's recorded frames alone, of
               the RK3 stage statements that carry no solver arithmetic: the
               stage sea level and thickness ratios (stprk3_stg.f90:148,
               :171-173, :211, :215-217, domqco.f90:257) and the stage tracer
               update (stprk3_stg.f90:503-505; trazdf.f90:284-294 at stage 3; trend zero on this
               deck), each in NEMO's order (must be 0 unequal) and in the
               order the card uses; no model step
  interior_events  census of sea-level cells over the bar in the INDEPENDENT
               chain, kt = 1..kt-max (up to 30): count, largest cell, its
               distance to land and to the fold, whether NEMO's sea level
               changes sign there during the step (with the base rate over
               all wet cells), and the same cell after ONE card step from
               NEMO's previous after-frame (T, S from the card's chain)

  substeps     the split-explicit solver's substep loop at one step holding a
               substep record (--kt-min = --kt-max = that step), every
               operand in compiled order (the VORTEX round-196 walk's table)
               against the model's own WRITE-only substep trace, on wet
               points, split by region.  GIVEN-NEMO-ENTRY: NEMO's entry state
               and, past step 1, NEMO's six histories from the record's own
               loop-entry frame
  substeps_independent  the same table from the card's own chain state
  flux_update_replay  OFFLINE numpy replay, from NEMO's substep record and
               mesh file alone, of the flux-form velocity update
               (dynspg_ts.f90:557-568) at every substep of one step: once as
               compiled (times the inverse exit depth; must be 0 unequal) and
               once dividing by the exit depth; no model step
  ratio_fold_replay  OFFLINE leaf table at one step (--kt-min = --kt-max), from
               NEMO's record alone: the U-, V- and F-point thickness ratios
               on the pivot row and the row below (domqco.f90:268-269,
               :281-285) against NEMO's recorded entry ratios, as L0 the
               bare stencil with the cyclic wrap, L1 with the fold halo read
               as the north neighbour of the top row, L2 = L1 passed through
               the fold exchange (stprk3_stg.f90:163), L3 = L0 passed through
               it.  Where the step holds a substep record, the PRODUCTION
               coefficient builder is also run offline on NEMO's entry sea
               level against NEMO's eight recorded coefficient sets, four
               ways: neither half, the ratio exchange alone, the mesh
               fe3mask pivot row alone (dommsk.f90:232, :258), both; no
               model step

  ratio_order_replay  OFFLINE, kt = kt-min..kt-max, from NEMO's record and
               the card's mesh arrays: the U-, V- and F-point thickness
               ratios from NEMO's entry sea level in NEMO's written order
               (domqco.f90:266-269, :281-285; must be 0 unequal against
               NEMO's recorded entry ratios off the top row) and divided by
               the face area; the solver's four V-face Coriolis coefficients
               (dynspg_ts.f90:987-1008) replayed from NEMO's recorded ratios
               on open faces (0 unequal against NEMO's recorded coefficients
               where the step holds a substep record) against the PRODUCTION
               builder run on NEMO's entry sea level with the written-order
               option on and off; no model step

  stage_operand_replay  OFFLINE, kt = kt-min..kt-max, from NEMO's stage frames:
               the density anomaly each stage's pressure gradient reads
               (eosbn2.f90 at gdept(Kmm) = gdept_0 * (1 + r3t)), from the
               shared library function on NEMO's T and S, at NEMO's stored
               ratio (the reference; stage_momentum_replay's control is what
               makes it NEMO's) against the PRODUCTION live depth ladder on
               the stage's sea level, as the card calls it (quotient) and
               with the reciprocal product; and one plus the U and V face
               ratio the stage reads as Kmm (stprk3_stg.f90:172-173,
               :216-217, NEMO's recorded one) against one plus the ratio of
               that frame's sea level (domqco.f90:266-269), the form the
               card's stage faces take; no model step

  subnormal_environment  no record, no model: how this process's arithmetic
               treats numbers below the smallest normal double.  A product of
               two normal numbers whose exact value is subnormal, and a
               subnormal operand times 1e10 (a normal result, so only the
               operand's treatment can zero it), in numpy before any compiled
               call, in a compiled XLA function (4 and 1,000,000 elements),
               in uncompiled jax.numpy, and in numpy again afterwards

Every scored row of the whole-step arms also carries ``unequal_census``: the
unequal cells split into those where NEMO holds a nonzero number below the
smallest normal double and the card holds exactly zero, and all others.

Every scored row is split into the preregistered regions (fold band, seam,
south, interior; interior again into coast-adjacent and open).  The scorer
and its 1e-15 bar are the shared trajectory gate's; the record reader is the
admission parser (check_records.read_stream).  The arm layout follows the
TSUNAMI lane's ladder, which is not on this branch (PR #1920).
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nemo_testcase_phase3_trajectory_gate import BAR, require, score  # noqa: E402

DEFAULT_ROOT = Path(
    "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gridladder_rounds/round1/"
    "oracle_g1_r1/p3_a_100step")
ARMS = ("geometry", "independent", "given_entry", "rhs", "handoff", "stages",
        "vor_fold_replay", "restart_entry", "fold_v_depth", "rhs_terms",
        "interior_events", "stage_replay", "substeps", "substeps_independent",
        "flux_update_replay", "ratio_fold_replay", "ratio_order_replay",
        "stage_operand_replay", "subnormal_environment")
PLANTS = ("score", "entry")
CASES = ("ORCA2G1-flat", "ORCA2G1b-vec")   # G1b = G1 + ORCA2's vector-form advection
CASE = CASES[0]           # --case: the card every arm builds
STAGE_STEPS = 10          # round 1's record: stage frames at kt = 1..10
FOLD_ROWS, SEAM_COLS, SOUTH_ROWS = 3, 2, 3
FIELDS = ("ssh", "uu_b", "vv_b", "u", "v")
FOLD_ROW_NAMES = (("fold_pivot_row", -1), ("fold_row_below", -2), ("fold_third_row", -3))


def _checker():
    spec = importlib.util.spec_from_file_location(
        "g1_check_records", HERE / "nemo_testcase_g1_orca2grid" / "check_records.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stage_steps(root: Path) -> int:
    """How many leading steps of this record hold stage frames (the step
    files with stage frames are the size of the first one)."""
    cr = _checker()
    size = cr.step_path(root, 1).stat().st_size
    n = 1
    while cr.step_path(root, n + 1).is_file() and \
            cr.step_path(root, n + 1).stat().st_size == size:
        n += 1
    return n


def read_step(root: Path, kt: int) -> dict[str, np.ndarray]:
    """One step file, cut to the inner domain its own header states."""
    cr = _checker()
    path = cr.step_path(root, kt)
    require(path.is_file(), f"missing {path}")
    head, fields = cr.read_stream(path.read_bytes(), path.name)
    require(head["step"] == kt, f"{path.name}: header step {head['step']}")
    js, je, i_s, ie = head["Njs0"] - 1, head["Nje0"], head["Nis0"] - 1, head["Nie0"]
    return {k: np.array(v[js:je, i_s:ie]) for k, v in fields.items()}


def regions(tmask: np.ndarray) -> dict[str, np.ndarray]:
    """The preregistered partition, plus the coast split of the interior."""
    wet = np.asarray(tmask) > 0.5
    nj, ni = wet.shape
    jj, ii = np.indices(wet.shape)
    fold = jj >= nj - FOLD_ROWS
    seam = ((ii < SEAM_COLS) | (ii >= ni - SEAM_COLS)) & ~fold
    south = (jj < SOUTH_ROWS) & ~seam
    interior = ~(fold | seam | south)
    require(bool(np.all(fold.astype(int) + seam + south + interior == 1)),
            "regions do not partition the domain exactly once")
    land_near = np.zeros_like(wet)
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            shifted = np.roll(~wet, (dj, di), axis=(0, 1))
            if dj == 1:
                shifted[0] = False
            if dj == -1:
                shifted[-1] = False
            land_near |= shifted
    edge = wet & land_near
    coast = edge.copy()                 # the wet edge and the land it touches
    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            shifted = np.roll(edge, (dj, di), axis=(0, 1))
            if dj == 1:
                shifted[0] = False
            if dj == -1:
                shifted[-1] = False
            coast |= shifted & ~wet
    out = {"fold": fold, "seam": seam, "south": south, "interior": interior,
           "interior_coast": interior & coast, "interior_open": interior & ~coast}
    # the fold band row by row: each row is scored against NEMO's OWN row
    # maximum (the signal falls ~100-fold per row towards the pivot, so a
    # band-wide maximum hides the pivot row)
    for name, j in FOLD_ROW_NAMES:
        out[name] = jj == nj + j
    return out


CELL: list = []            # --cell j,i: every row also reports this cell


def row(name, ref, cand, reg, *, plant=False) -> dict:
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    at_cell = None
    if CELL:
        j, i = CELL
        d0 = cand - ref
        near = d0[j - 1:j + 2, i - 1:i + 2]
        at_cell = {"cell": [j, i], "error": float(d0[j, i]),
                   "nemo": float(ref[j, i]),
                   "ulps": float(d0[j, i] / np.spacing(abs(ref[j, i]))),
                   "neighbourhood_max_abs": float(np.max(np.abs(near))),
                   "neighbourhood_errors": near.tolist()}
    r = score(name, ref, cand, np.ones(ref.shape, dtype=bool), plant=plant)
    if plant:
        cand = cand.copy()
        cand.flat[0] += 1.0
    d = cand - ref
    scale = max(float(np.max(np.abs(ref))), 1.0)
    bad = np.argwhere(d != 0)
    r.update({
        "n_unequal": int(bad.shape[0]),
        "first_unequal_cell": bad[0].tolist() if bad.size else None,
        "max_abs": float(np.max(np.abs(d))),
        "max_abs_cell": [int(x) for x in np.unravel_index(
            int(np.argmax(np.abs(d))), d.shape)],
        "rms": float(np.sqrt(np.mean(d * d))),
        "ref_max_abs": float(np.max(np.abs(ref))),
        "bit_identical": bool(bad.shape[0] == 0),
        "unequal_census": unequal_census(ref, cand),
        "at_cell": at_cell,
        "regions": {
            k: {"n_cells": int(m.sum()),
                "n_unequal": int(np.count_nonzero(d[m])),
                "max_abs": float(np.max(np.abs(d[m]))),
                "rms": float(np.sqrt(np.mean(d[m] ** 2))),
                "normalized_max_abs": float(np.max(np.abs(d[m]))) / scale,
                "ref_max_abs": float(np.max(np.abs(ref[m]))),
                "n_ref_nonzero": int(np.count_nonzero(ref[m]))}
            for k, m in reg.items()},
    })
    return r


def first_unequal(rows):
    for r in rows:
        if not r["bit_identical"]:
            return {k: r[k] for k in ("name", "n_unequal", "first_unequal_cell",
                                      "max_abs", "normalized_max_abs", "status")}
    return None


def state_fields(st) -> dict[str, np.ndarray]:
    return {"ssh": np.asarray(st.eta.data),
            "uu_b": np.asarray(st.uu_b.data)[:, 1:],
            "vv_b": np.asarray(st.vv_b.data)[1:, :],
            "u": np.asarray(st.u.data)[:, 1:, 0],
            "v": np.asarray(st.v.data)[1:, :, 0],
            "T": np.asarray(st.T.data)[..., 0],
            "S": np.asarray(st.S.data)[..., 0]}


def pad_u(a):
    """The redundant west face record: the cyclic image of the east face."""
    a = np.asarray(a)
    return np.concatenate([a[:, -1:], a], axis=1)


def pad_v(a):
    """The redundant south face record: the closed wall, zero."""
    a = np.asarray(a)
    return np.concatenate([np.zeros_like(a[:1]), a], axis=0)


def history_rows(root: Path, kt: int, state, reg) -> list[dict]:
    """The card's six carried barotropic histories vs NEMO's restart file."""
    path = root / f"ORCA2G1_{kt:08d}_restart.nc"
    if not path.is_file() or state.bt_hist is None:
        return []
    import netCDF4
    ub, ubb, vb, vbb, sb, sbb = (np.asarray(a) for a in state.bt_hist)
    mine = {"ub_e": ub[:, 1:], "ubb_e": ubb[:, 1:], "vb_e": vb[1:], "vbb_e": vbb[1:],
            "sshb_e": sb, "sshbb_e": sbb}
    with netCDF4.Dataset(path) as h:
        return [row(f"history.kt{kt}.{n}",
                    np.asarray(h.variables[n][0], dtype=np.float64), a, reg)
                for n, a in mine.items()]


def nemo_histories(root: Path, kt: int):
    """NEMO's six barotropic histories entering step ``kt`` (the restart
    written after step kt - 1), padded to the card's layout; None where no
    restart exists."""
    path = root / f"ORCA2G1_{kt - 1:08d}_restart.nc"
    if not path.is_file():
        return None
    import jax.numpy as jnp
    import netCDF4
    with netCDF4.Dataset(path) as h:
        var = lambda n: np.asarray(h.variables[n][0], dtype=np.float64)  # noqa: E731
        return tuple(jnp.asarray(a) for a in (
            pad_u(var("ub_e")), pad_u(var("ubb_e")), pad_v(var("vb_e")),
            pad_v(var("vbb_e")), var("sshb_e"), var("sshbb_e")))


# ----------------------------------------------------------------- geometry
def card_geometry_rows(card, dom, all_levels=False) -> list[tuple[str, np.ndarray, str]]:
    """(row, the card's array, the mesh_mask variable it must equal)."""
    g, z = card.recipe.grid, card.recipe.z_coord
    ops = z.nemo_een_barotropic
    f64 = lambda a: np.asarray(a, dtype=np.float64)       # noqa: E731
    lev = lambda a: f64(a) if all_levels else f64(a)[..., 0]   # noqa: E731
    rows = [(f"operand {n}", f64(getattr(ops, n)), n) for n in
            ("e1t", "e2t", "e1u", "e2u", "e1v", "e2v", "e1f", "e2f")]
    rows += [
        ("operand ff_f", f64(ops.ff_f), "ff_f"),
        ("grid ff_f", f64(g.ff_f), "ff_f"),
        ("grid f_T", f64(g.f_T), "ff_t"),
        ("grid dx_T", f64(g.dx_T), "e1t"), ("grid dy_T", f64(g.dy_T), "e2t"),
        ("grid dx_u", f64(g.dx_u)[:, 1:], "e1u"),
        ("grid dy_u", f64(g.dy_u)[:, 1:], "e2u"),
        ("grid dx_v", f64(g.dx_v)[1:, :], "e1v"),
        ("grid dy_v", f64(g.dy_v)[1:, :], "e2v"),
        ("coordinate nemo_e2u", f64(z.nemo_e2u), "e2u"),
        ("coordinate nemo_e1v", f64(z.nemo_e1v), "e1v"),
        ("tmask (is_active)", lev(z.is_active), "tmask"),
        ("operand umask", lev(ops.umask), "umask"),
        ("operand vmask", lev(ops.vmask), "vmask"),
        ("operand fmask", lev(ops.fmask), "fmask"),
        ("coordinate e3t_0", lev(z.nemo_e3t_0), "e3t_0"),
        ("operand e3u_0", lev(ops.e3u_0), "e3u_0"),
        ("operand e3v_0", lev(ops.e3v_0), "e3v_0"),
        ("operand e3f_0", lev(ops.e3f_0), "e3f_0"),
        ("drag e3u_0", lev(z.nemo_bbl_e3u_0), "e3u_0"),
        ("drag e3v_0", lev(z.nemo_bbl_e3v_0), "e3v_0"),
        ("coordinate e3w_0", lev(z.nemo_e3w_0), "e3w_1d"),
        ("coordinate gdept_0", lev(z.nemo_gdept_0), "gdept_1d"),
        ("coordinate gdepw_0", lev(z.nemo_gdepw_0), "gdepw_1d"),
        # NEMO writes mbkt = MAX(k_bot, 1) (domzgr.f90:575, domwri.f90:164-165):
        # 1 on land too, so the file cannot tell land from one wet level.
        ("max(bottom_level + 1, 1)", np.maximum(f64(z.bottom_level) + 1.0, 1.0),
         "mbathy"),
        ("h_partial", lev(z.h_partial), "e3t_0*tmask"),
        ("bump input glamt", f64(dom["glamt"]), "glamt"),
        ("bump input gphit", f64(dom["gphit"]), "gphit"),
    ]
    return rows


def _fp64() -> None:
    """The oracle's precision, set and verified (never the caller's default)."""
    import jax
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    policy = PrecisionPolicy.fp64(transcendentals="libm")
    set_policy(policy)
    require(get_policy() == policy and bool(jax.config.jax_enable_x64),
            "the fp64 policy is not active")
    require(jax.default_backend() == "cpu", "the ladder must run on CPU")


def geometry(root: Path, plant: str | None = None) -> dict:
    import netCDF4
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    _fp64()
    card = recipe.build_nemo_testcase_card(CASE, deck_root=root)
    dom = recipe._orca2g1_domain(root / "domain_cfg.nc")
    rows_in = card_geometry_rows(card, dom)
    names = [r[0] for r in rows_in]
    require(plant is None or plant in names,
            f"unknown geometry plant {plant!r}; expected one of {names}")
    out = []
    with netCDF4.Dataset(root / "mesh_mask.nc") as h:
        def nemo(var):
            if var == "e3t_0*tmask":
                return nemo("e3t_0") * nemo("tmask")
            a = np.asarray(h.variables[var][0], dtype=np.float64)
            if a.ndim == 3:                 # (jpk, y, x): the executed level
                return a[0]
            if a.ndim == 1:                 # 1-d ladder at the executed level
                return np.full(rows_in[0][1].shape, a[0])
            return a
        for name, lego, var in rows_in:
            lego = np.array(lego, dtype=np.float64)
            ref = nemo(var)
            if plant == name:
                lego.flat[lego.size // 2] += 1.0
            require(lego.shape == ref.shape,
                    f"{name}: card {lego.shape} vs NEMO {var} {ref.shape}")
            bad = np.argwhere(lego != ref)
            out.append({
                "row": name, "nemo_variable": var,
                "bit_identical": bool(bad.shape[0] == 0),
                "n_differing": int(bad.shape[0]),
                "first_unequal_index": bad[0].tolist() if bad.size else None,
                "rows_touched": sorted({int(b[0]) for b in bad})[:12],
                "max_abs_difference": float(np.max(np.abs(lego - ref))),
                "planted": plant == name})
    return {"arm": "geometry", "mesh_mask": str(root / "mesh_mask.nc"),
            "rows": out, "n_rows": len(out),
            "n_bit_identical": sum(r["bit_identical"] for r in out),
            "all_bit_identical": all(r["bit_identical"] for r in out)}


def rhs_term_replay(root: Path, kt: int, fe3mask, *, stage: int = 1,
                    zero_image_row: bool = False, rhd=None,
                    vector: bool = False) -> dict:
    """NEMO's 3-D momentum right-hand side of one stage, term by term, offline.

    The pressure gradient on the one level (dynhpg.f90:380-394) and the
    planetary vor_een trend (dynvor.f90:720 z1_e3f, :727 zwz, :766-767 the
    transports, :772-775 the triads, :779-784 zua and zva), each in NEMO's
    literal operand order, on the whole inner domain.  Everything is NEMO's:
    the frame the stage reads WITH its halos (stage 1: the step entry; stage
    2 and 3: the previous stage's frame, with the F ratio the stage reads),
    the mesh file, and the T fold image read off NEMO's own halo row.
    ``fe3mask`` is the card's (the mesh file does not carry it).  ``rhd`` is
    the density anomaly on the haloed grid; None runs the pressure gradient
    with the shared library function on NEMO's recorded T and S at NEMO's
    stretched depth (eosbn2.f90 reads gdept at Kmm).  The top V row is not
    replayed: its north operands are fold halos the mesh file does not hold.
    ``vector`` (G1b): zwz is NEMO's np_CRV, planetary plus relative vorticity
    (dynvor.f90:744-748, unmasked: ln_dynvor_msk = F), and the KE gradient
    (dynkeg.f90:117-125, nkeg_C2) is subtracted after the vorticity trend
    (stp2d.f90:142, :155); dyn_zad adds nothing on one level (dynzad.f90:105).
    """
    import netCDF4
    cr = _checker()
    path = cr.step_path(root, kt)
    head, f = cr.read_stream(path.read_bytes(), path.name)
    require(head["Njs0"] == 3 and head["Nis0"] == 3, "two halo points expected")
    require(stage in (1, 2, 3), "stage must be 1, 2 or 3")
    nj, ni = head["jpj"], head["jpi"]
    jp = head["Nje0"] - 1                       # 0-based halo row of the pivot
    pre, suf = ("e_", "_bb") if stage == 1 else (f"{stage - 1}_", "_aa")
    ssh, r3t, r3u, r3v, u, v, tn, sn = (f[pre + n + suf] for n in (
        "ssh", "r3t", "r3u", "r3v", "uu_k1", "vv_k1", "tn_k1", "sn_k1"))
    # the F ratio the stage's vorticity term reads: the step-entry one at
    # stage 1; the one stage 2 set (stprk3_stg.f90:218), which its frame
    # holds; at stage 3 the mean of the entry and after ratios (:239) -- the
    # stage-3 frame holds the after ratio, written back at :546
    r3f = (f["e_r3f"], f["2_r3f"], 0.5 * (f["e_r3f"] + f["3_r3f"]))[stage - 1]

    def halo(a):                                # cyclic east-west halo
        out = np.zeros((nj, ni))
        out[2:2 + a.shape[0], 2:-2] = a
        out[:, :2], out[:, -2:] = out[:, -4:-2], out[:, 2:4]
        return out
    with netCDF4.Dataset(root / "mesh_mask.nc") as h:
        lev = lambda n: np.asarray(h.variables[n][0], dtype=np.float64)  # noqa: E731
        m2 = {n: halo(lev(n)) for n in ("ff_f", "e1v", "e1u", "e2u", "e2v", "e1f", "e2f")}
        m3 = {n: halo(lev(n)[0]) for n in (
            "tmask", "vmask", "umask", "e3t_0", "e3u_0", "e3v_0", "e3f_0")}
        gdept, e3w = (float(lev(n)[0]) for n in ("gdept_1d", "e3w_1d"))
    # T fold image of the first halo row, read off NEMO's own sea level at
    # the last stage step (the first steps have no signal on the fold)
    last = cr.step_path(root, STAGE_STEPS)
    ssh_l = cr.read_stream(last.read_bytes(), last.name)[1]["e_ssh_bb"]
    require(bool(np.any(ssh_l[jp + 1, 2:-2])), "no sea-level signal on the fold halo row")
    cand = [s for s in range(-4, 5) if np.array_equal(
        ssh_l[jp + 1, 2:-2], ssh_l[jp - 1, (ni - np.arange(2, ni - 2) + s) % ni])]
    require(len(cand) == 1, f"T fold image not unique: shifts {cand}")
    perm = (ni - np.arange(ni) + cand[0]) % ni
    te3 = m3["e3t_0"] * m3["tmask"]
    te3[jp + 1] = te3[jp - 1, perm]             # halo row jj = Nje0 + 1
    e3f0 = np.zeros((nj, ni))
    e3f0[:-1, :-1] = ((te3[1:, :-1] + te3[1:, 1:]) + (te3[:-1, :-1] + te3[:-1, 1:])) * 0.25
    e3f0 = np.where(e3f0 == 0.0, m3["e3f_0"], e3f0)      # dynvor.f90:919
    fe3 = halo(np.asarray(fe3mask, dtype=np.float64))
    at = lambda a, dj, di: np.roll(a, (-dj, -di), (0, 1))  # noqa: E731  a(ji+di, jj+dj)
    with np.errstate(divide="ignore", invalid="ignore"):
        z1_e3f = 1.0 / (e3f0 * (1.0 + r3f * fe3))
        if vector:                              # np_CRV, dynvor.f90:744-748
            r1_e1e2f = 1.0 / (m2["e1f"] * m2["e2f"])
            zwz = (m2["ff_f"] + ((at(m2["e2v"], 0, 1) * at(v, 0, 1) - m2["e2v"] * v)
                                 - (at(m2["e1u"], 1, 0) * at(u, 1, 0) - m2["e1u"] * u))
                   * r1_e1e2f) * z1_e3f
        else:
            zwz = m2["ff_f"] * z1_e3f
        r1_e1u, r1_e2v = 1.0 / m2["e1u"], 1.0 / m2["e2v"]
    zwx = m2["e2u"] * (m3["e3u_0"] * (1.0 + r3u * m3["umask"])) * u
    zwy = m2["e1v"] * (m3["e3v_0"] * (1.0 + r3v * m3["vmask"])) * v
    if zero_image_row:
        zwy[jp] = 0.0
    with np.errstate(invalid="ignore"):
        ztne = at(zwz, 0, -1) + zwz + at(zwz, -1, 0)
        ztnw = at(zwz, -1, -1) + at(zwz, 0, -1) + zwz
        ztse = zwz + at(zwz, -1, 0) + at(zwz, -1, -1)
        ztsw = at(zwz, -1, 0) + at(zwz, -1, -1) + at(zwz, 0, -1)
        vor_u = (1.0 / 12.0) * r1_e1u * (
            (ztne * zwy + at(ztnw, 0, 1) * at(zwy, 0, 1))
            + (ztse * at(zwy, -1, 0) + at(ztsw, 0, 1) * at(zwy, -1, 1)))
        vor_v = -((1.0 / 12.0) * r1_e2v * (
            (at(ztsw, 1, 0) * at(zwx, 1, -1) + at(ztse, 1, 0) * at(zwx, 1, 0))
            + (ztnw * at(zwx, 0, -1) + ztne * zwx)))
    from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG as C
    if rhd is None:
        from legoesm.ocean.eos import nemo_roquet_density_anomaly_ratio
        depth = (gdept * (1.0 + r3t))[..., None]
        rhd = np.asarray(nemo_roquet_density_anomaly_ratio(
            tn[..., None], sn[..., None], depth, rho0=float(C.rho_0),
            geometric_depth_m=depth, tmask=m3["tmask"][..., None]))[..., 0]
    zcoef0 = -float(C.g) * 0.5
    col = e3w * (1.0 + r3t)                    # e3w(Kmm), one level
    dep = gdept * (1.0 + r3t) - ssh            # gde3w(Kmm)
    with np.errstate(invalid="ignore"):
        hpg_u = zcoef0 * r1_e1u * (at(col, 0, 1) * at(rhd, 0, 1) - col * rhd) \
            + -zcoef0 * (at(rhd, 0, 1) + rhd) * (at(dep, 0, 1) - dep) * r1_e1u
        hpg_v = zcoef0 * r1_e2v * (at(col, 1, 0) * at(rhd, 1, 0) - col * rhd) \
            + -zcoef0 * (at(rhd, 1, 0) + rhd) * (at(dep, 1, 0) - dep) * r1_e2v
    cut = lambda a: np.array(a[2:jp + 1, 2:-2])  # noqa: E731
    wet_u, wet_v = cut(m3["umask"]) > 0, cut(m3["vmask"]) > 0
    wet_v[-1] = False
    clean = lambda a, w: np.where(w, cut(a), 0.0)  # noqa: E731
    out = {"hpg_u": clean(hpg_u, wet_u), "hpg_v": clean(hpg_v, wet_v),
           "vor_u": clean(vor_u, wet_u), "vor_v": clean(vor_v, wet_v),
           "wet_u": wet_u, "wet_v": wet_v, "t_fold_shift": cand[0],
           "rhd": cut(rhd), "r3t": cut(r3t), "r3u": cut(r3u), "r3v": cut(r3v),
           "r3f": cut(r3f), "ssh": cut(ssh), "u": cut(u), "v": cut(v),
           "zwz": cut(zwz), "e3f": cut(e3f0 * (1.0 + r3f * fe3)),
           "ff_f": cut(m2["ff_f"]), "depth": cut(gdept * (1.0 + r3t))}
    out["rhs_u"] = out["hpg_u"] + out["vor_u"]
    out["rhs_v"] = out["hpg_v"] + out["vor_v"]
    if vector:                                  # dynkeg.f90:117-125
        zhke = 0.25 * ((at(v, -1, 0) * at(v, -1, 0) + v * v)
                       + (at(u, 0, -1) * at(u, 0, -1) + u * u))
        with np.errstate(invalid="ignore"):
            keg_u = (at(zhke, 0, 1) - zhke) * r1_e1u
            keg_v = (at(zhke, 1, 0) - zhke) * r1_e2v
        out["keg_u"], out["keg_v"] = clean(keg_u, wet_u), clean(keg_v, wet_v)
        out["rhs_u"] = out["rhs_u"] - out["keg_u"]
        out["rhs_v"] = out["rhs_v"] - out["keg_v"]
    return out


def vor_een_u_replay(root: Path, kt: int, fe3mask, *, zero_image_row: bool) -> dict:
    """NEMO's planetary vor_een trend on U, rows jj = Nje0-1 and Nje0: the two
    top rows of ``rhs_term_replay``'s vorticity term at the step entry."""
    full = rhs_term_replay(root, kt, fe3mask, zero_image_row=zero_image_row)
    return {"pivot_row": full["vor_u"][-1], "row_below": full["vor_u"][-2],
            "wet": {"pivot_row": full["wet_u"][-1], "row_below": full["wet_u"][-2]},
            "t_fold_shift": full["t_fold_shift"]}


def land_distance(wet: np.ndarray, cap: int = 40) -> np.ndarray:
    """Cells (Chebyshev, cyclic east-west, closed north-south) to the nearest
    dry cell; ``cap`` where none lies within ``cap - 1``."""
    land = ~np.asarray(wet, dtype=bool)
    dist = np.where(land, 0, cap)
    reach = land.copy()
    for d in range(1, cap):
        grown = reach.copy()
        for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1),
                       (-1, 1), (-1, -1)):
            sh = np.roll(reach, (dj, di), axis=(0, 1))
            if dj == 1:
                sh[0] = False
            if dj == -1:
                sh[-1] = False
            grown |= sh
        dist = np.where((dist == cap) & grown, d, dist)
        reach = grown
    return dist


def fold_v_depth_rows(root: Path, kt: int, card, cfg) -> list[dict]:
    """The ssh term of ``zhvp2_e`` on the two top V rows, card vs NEMO.

    NEMO side: ``r3v`` as recorded at step entry times ``hv_0`` -- the same
    ``e1e2t``-weighted two-cell mean (``domqco.f90:268-269``) that
    ``dynspg_ts.f90:419-421`` adds to ``hv_0``; on the fold-image row it reads
    the T fold halo.  Card side: the solver's own helper on NEMO's entry ssh,
    with the reference depths and the fold operands ``cfg`` selects.  The two
    differ by the rounding of one multiply-divide pair where they agree.
    """
    import jax.numpy as jnp
    from legoesm.grids.latlon import ensure_geometry
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as bt
    zc, grid, s0 = card.recipe.z_coord, card.recipe.grid, card.recipe.initial_state
    g = read_step(root, kt)
    f64 = jnp.float64
    prep = bt._nemo_ssh_avg_reference_depth_override(
        bt._nemo_ssh_avg_prep(s0.H_bathy.data, jnp.asarray(card.recipe.land_mask),
                              grid, f64),
        bt._nemo_literal_reference_face_depths(zc, f64), f64)
    out = bt._nemo_ssh_avg_apply(
        jnp.asarray(g["e_ssh_bb"]), s0.u_mask.data, s0.v_mask.data, grid,
        ensure_geometry(grid).area_T, prep, return_literal_inverse=True,
        return_ssh_average=True,
        fold_top_v_wet=bt._nemo_fold_top_v_wet(cfg, zc, f64))
    card_v = np.asarray(out[5])[1:]
    nemo_v = g["e_r3v_bb"] * np.asarray(zc.nemo_een_barotropic.hv_0)
    rows = []
    for name, j in (("fold_image_v_row", -1), ("row_below", -2)):
        ref = float(np.max(np.abs(nemo_v[j])))
        err = float(np.max(np.abs(card_v[j] - nemo_v[j])))
        rows.append({"row": name, "nemo_max_abs": ref,
                     "n_nemo_nonzero": int(np.count_nonzero(nemo_v[j])),
                     "card_max_abs": float(np.max(np.abs(card_v[j]))),
                     "max_abs": err, "relative": err / ref if ref else None})
    return rows


def stage_replay_rows(root: Path, kt: int, tmask) -> dict:
    """NEMO's stage statements replayed from its own frames (inner domain).

    ``nemo``: the statement in NEMO's order against NEMO's record (a
    transliteration check: 0 unequal or the replay is wrong).  ``card_order``:
    the same operands combined the way the card combines them, against NEMO's
    record (cells where the two orders round differently)."""
    g = read_step(root, kt)
    wet = np.asarray(tmask) > 0.5
    r1_3, r2_3, r1_2 = 1.0 / 3.0, 2.0 / 3.0, 1.0 / 2.0   # stprk3_stg.f90:51-53
    r1_ht_0 = wet / (100.0 * wet + 1.0 - wet)   # one 100 m level (geometry gate)
    sb, sa = g["e_ssh_bb"], g["3_ssh_aa"]

    def cmp(mine, ref):
        d = (mine - ref)[wet]
        return {"n_unequal": int(np.count_nonzero(d)),
                "max_abs": float(np.max(np.abs(d))),
                "ref_max_abs": float(np.max(np.abs(ref[wet])))}
    rows = {
        "stage1 ssh (:148)": {
            "nemo": cmp(r2_3 * sb + r1_3 * sa, g["1_ssh_aa"]),
            "card_order": cmp(sb + (sa - sb) / 3.0, g["1_ssh_aa"])},
        "stage2 ssh (:211)": {
            "nemo": cmp(r1_2 * (sb + sa), g["2_ssh_aa"]),
            "card_order": cmp(0.5 * (sb + sa), g["2_ssh_aa"])},
        "after r3t (domqco:257)": {
            "nemo": cmp(sa * r1_ht_0, g["3_r3t_aa"]),
            "card_order": cmp(sa * r1_ht_0, g["3_r3t_aa"])},
    }
    for f in ("t", "u", "v"):
        b, a = g[f"e_r3{f}_bb"], g[f"3_r3{f}_aa"]
        m = wet if f == "t" else np.ones_like(wet)
        for name, mine, ref in (
                (f"stage1 r3{f} (:171-173)", r2_3 * b + r1_3 * a, g[f"1_r3{f}_aa"]),
                (f"stage2 r3{f} (:215-217)", r1_2 * (b + a), g[f"2_r3{f}_aa"])):
            d = (mine - ref)[m]
            rows[name] = {"nemo": {"n_unequal": int(np.count_nonzero(d)),
                                   "max_abs": float(np.max(np.abs(d))),
                                   "ref_max_abs": float(np.max(np.abs(ref[m])))}}
    # the card forms the stage-2 T ratio from the interpolated sea level
    q2_card = 1.0 + (0.5 * (sb + sa)) * r1_ht_0
    q2_nemo = 1.0 + g["2_r3t_aa"]
    rows["stage2 1+r3t"] = {"card_order": cmp(q2_card, q2_nemo)}
    qb = 1.0 + g["e_r3t_bb"]
    for trc in ("tn", "sn"):
        tb = g[f"e_{trc}_k1_bb"]
        for st, q_card in ((1, 1.0 + g["1_r3t_aa"]), (2, q2_card)):
            ref = g[f"{st}_{trc}_k1_aa"]
            rows[f"stage{st} {trc} (:503-505)"] = {
                "nemo": cmp((qb * tb) / (1.0 + g[f"{st}_r3t_aa"]), ref),
                "card_order": cmp((qb * tb) / q_card, ref)}
        # stage 3 is tra_zdf's one-level solve (trazdf.f90:284-285, :294, with
        # the diagonal of :233 and no mixing across the single level): the
        # same quotient with the 100 m reference thickness on both sides
        rows[f"stage3 {trc} (trazdf:284-294)"] = {"nemo": cmp(
            ((100.0 * qb) * tb) / (100.0 * (1.0 + g["3_r3t_aa"])),
            g[f"3_{trc}_k1_aa"])}
    return rows


def read_substep_record(root: Path, kt: int):
    """One solver substep stream, every plane cut to the inner domain."""
    cr = _checker()
    path = cr.substep_path(root, kt)
    require(path.is_file(), f"missing {path}")
    head, fields = cr.read_substeps(path.read_bytes(), path.name)
    require(head["step"] == kt, f"{path.name}: header step {head['step']}")
    js, je, i_s, ie = head["Njs0"] - 1, head["Nje0"], head["Nis0"] - 1, head["Nie0"]
    full, inner = (head["jpj"], head["jpi"]), (je - js, ie - i_s)
    out = {}
    for name, a in fields.items():
        require(a.ndim == 1 or a.shape in (full, inner),
                f"{path.name}: {name} has extent {a.shape}")
        out[name] = np.array(a[js:je, i_s:ie] if a.shape == full else a)
    return head, out


REPLAY_ORDERS = ("nemo_order", "divide_by_exit_depth", "entry_depth_as_sum")


def flux_update_replay_rows(root: Path, kt: int, reg) -> list[dict]:
    """NEMO's flux-form external velocity update replayed from NEMO's own
    recorded operands (dynspg_ts.f90:557-568), U and V, every substep.  The
    one difference between the two orders is the last operation: NEMO
    multiplies the bracket by the inverse exit depth z1_hu (:557-558), the
    other order divides the bracket by the exit depth.  The top V row and
    the fold are not replayed (their north neighbour is outside the mesh
    file's inner domain)."""
    import netCDF4
    head, g = read_substep_record(root, kt)
    with netCDF4.Dataset(root / "mesh_mask.nc") as h:
        var = lambda n: np.asarray(h.variables[n][0], dtype=np.float64)  # noqa: E731
        e1e2t = var("e1t") * var("e2t")
        mesh = {"u": (var("umask")[0], var("e3u_0")[0], var("e1u") * var("e2u"),
                      np.roll(e1e2t, -1, axis=1)),
                "v": (var("vmask")[0], var("e3v_0")[0], var("e1v") * var("e2v"),
                      np.roll(e1e2t, -1, axis=0))}
    rdt = float(g["i000_entry_sc"][0])
    rows = []
    for jn in range(1, head["icycle"] + 1):
        now, prev = f"j{jn:03d}_", f"j{jn - 1:03d}_"
        ssh = g[now + "sshp2_bck"]
        for face, axis in (("u", 1), ("v", 0)):
            mask, e3, e1e2, e1e2t_next = mesh[face]
            wet = mask > 0
            if face == "v":
                wet[-1] = False                 # north neighbour not in the file
            h0 = e3 * mask
            vel = g[f"i000_{face}n_e"] if jn == 1 else g[prev + f"{face}a_new"]
            h_in = g[f"i000_h{face}_e"] if jn == 1 else g[prev + f"h{face}_e"]
            h_bck = h0 + 0.5 * (1.0 / e1e2) * (
                e1e2t * ssh + e1e2t_next * np.roll(ssh, -1, axis=axis)) * mask
            bracket = h_in * vel + rdt * (
                h_bck * g[now + f"z{face}_spg"]
                + g[now + f"h{face}p2_e"] * g[now + f"trd_{face}"]
                + g[f"i000_h{face}_e"] * g[f"i000_z{face}_frc"])
            h_exit = h0 + g[now + f"ssh{face}_a"]
            inverse = mask / (h_exit + 1.0 - mask)
            # the other entry depth: rest depth PLUS the face-mean entry sea
            # level, where NEMO's hu_e / hu(Kmm) is hu_0 * (1 + r3u) (:329)
            ssh0 = g["i000_sshn_e"]
            h_sum = h0 + 0.5 * (1.0 / e1e2) * (
                e1e2t * ssh0 + e1e2t_next * np.roll(ssh0, -1, axis=axis)) * mask
            bracket_sum = (h_sum if jn == 1 else h_in) * vel + rdt * (
                h_bck * g[now + f"z{face}_spg"]
                + g[now + f"h{face}p2_e"] * g[now + f"trd_{face}"]
                + h_sum * g[f"i000_z{face}_frc"])
            with np.errstate(divide="ignore", invalid="ignore"):
                orders = {"nemo_order": bracket * inverse,
                          "divide_by_exit_depth": np.where(
                              wet, bracket / h_exit * mask, 0.0),
                          "entry_depth_as_sum": bracket_sum * inverse}
            ref = g[now + f"{face}a_new"]
            rows.append({"substep": jn, "face": face, **{
                name: substep_row(f"j{jn:03d}.{face}a.{name}", ref, cand, wet, reg)
                for name, cand in orders.items()}})
    return rows


RATIO_VARIANTS = ("L0_stencil_cyclic_wrap", "L1_fold_halo_read",
                  "L2_halo_read_then_exchange", "L3_exchange_only",
                  "L4_exchange_only_times_inverse_area")


def _inverse_depth(h):
    """r1_hu_0 and its siblings: ssumask / (hu_0 + 1 - ssumask), domain.f90:213."""
    wet = (h > 0.0).astype(np.float64)
    return wet / (h + 1.0 - wet)


def ratio_stencil(area_ssh, north, a, nemo_order):
    """dom_qco_r3c_RK3's three face ratios (domqco.f90:266-269, :281-285) from
    the area-weighted sea level and its north neighbour.  ``nemo_order``
    multiplies by the inverse face area as NEMO writes it; otherwise the
    product is divided by the face area (the solver coefficient builder's
    order, one last place apart on about a quarter of the points)."""
    east, north_east = np.roll(area_ssh, -1, axis=1), np.roll(north, -1, axis=1)
    return tuple(lead * _inverse_depth(a(h)) * (1.0 / (a(e1) * a(e2))) if nemo_order
                 else lead * _inverse_depth(a(h)) / (a(e1) * a(e2))
                 for lead, h, e1, e2 in (
                     (0.5 * (area_ssh + east), "hu_0", "e1u", "e2u"),
                     (0.5 * (area_ssh + north), "hv_0", "e1v", "e2v"),
                     (0.25 * ((area_ssh + east) + (north + north_east)),
                      "hf_0", "e1f", "e2f")))


STAGE1_RATIOS = ("nemo_recorded_ratio", "ratios_interpolated",
                 "ratio_of_interpolated_ssh")
# how the two statements are evaluated: as written (every product and sum
# rounded once), or with ONE product fused into the sum that follows it
STAGE1_ORDERS = ("written", "fused_rhs_product", "fused_before_product",
                 "fused_correction", "fused_rhs_product_and_correction")


def fused_multiply_add(x, y, z):
    """x * y + z rounded ONCE (exact rational arithmetic, then one rounding
    to binary64): what a fused multiply-add instruction returns."""
    from fractions import Fraction
    x, y, z = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in (x, y, z)))
    require(bool(np.all(np.isfinite(x) & np.isfinite(y) & np.isfinite(z))),
            "fused_multiply_add: a non-finite operand (the replay has none)")
    out = np.zeros(x.shape)
    for idx in zip(*np.nonzero((x != 0) & (y != 0) | (z != 0))):
        out[idx] = float(Fraction(x[idx]) * Fraction(y[idx]) + Fraction(z[idx]))
    return out


def stage1_momentum_replay(g: dict, card, card_fields: dict, reg,
                           card_rhs=None, card_raw=None) -> dict:
    """NEMO's stage-1 velocity from NEMO's own recorded operands, offline.

    The thickness-weighted update (stprk3_stg.f90:373-378) and the barotropic
    correction (:413-414, :417-418), with the after ratio taken three ways:
    NEMO's recorded one; two thirds of the before ratio plus one third of the
    ratio of the solver's sea level (:171-173); the ratio of the stage's
    interpolated sea level.  Each is scored against NEMO's stage-1 frame and
    against the card's own stage-1 velocity from the same entry.  Stage 1
    only: the later stages' right-hand sides are not in the record.  The top
    V row is not replayed (its north neighbour is the fold)."""
    raw = card.recipe.z_coord.nemo_een_barotropic
    a = lambda n: np.asarray(getattr(raw, n), dtype=np.float64)  # noqa: E731
    area = a("e1t") * a("e2t")
    rdt = (1.0 / 3.0) * float(card.dt_s)             # rDt = r1_3 * rn_Dt

    def ratios(ssh):
        x = area * ssh
        return ratio_stencil(x, np.roll(x, -1, axis=0), a, True)[:2]

    out = {"rDt": rdt, "rows": []}
    for k, face in enumerate("uv"):
        mask, e3 = a(f"{face}mask")[..., 0], a(f"e3{face}_0")[..., 0]
        wet = mask > 0
        if face == "v":
            wet[-1] = False
        r3b, r3a = g[f"e_r3{face}_bb"], g[f"1_r3{face}_aa"]
        after = dict(zip(STAGE1_RATIOS, (
            r3a,
            (2.0 / 3.0) * r3b + (1.0 / 3.0) * ratios(g["b_ssh_aa"])[k],
            ratios(g["1_ssh_aa"])[k])))
        ref, mine = g[f"1_{face}{face}_k1_aa"], card_fields[face]
        before, rhs = g[f"e_{face}{face}_k1_bb"], g[f"b_{face}{face}_rhs_k1"]
        target, r1 = g[f"1_{face}{face}_b_aa"], _inverse_depth(a(f"h{face}_0"))
        for name, order in ((n, o) for n in after for o in STAGE1_ORDERS):
            ratio = after[name]
            if order in ("fused_rhs_product", "fused_rhs_product_and_correction"):
                top = fused_multiply_add(rdt * (1.0 + r3b), rhs, (1.0 + r3b) * before)
            elif order == "fused_before_product":
                top = fused_multiply_add(1.0 + r3b, before, rdt * (1.0 + r3b) * rhs)
            else:
                top = (1.0 + r3b) * before + rdt * (1.0 + r3b) * rhs
            with np.errstate(divide="ignore", invalid="ignore"):
                vel = np.where(wet, top / (1.0 + ratio) * mask, 0.0)
            if order in ("fused_correction", "fused_rhs_product_and_correction"):
                vel = vel + fused_multiply_add(-(e3 * vel), r1, target) * mask
            else:
                vel = vel + (target - (e3 * vel) * r1) * mask
            out["rows"].append({
                "face": face, "after_ratio": name, "order": order,
                "ratio_vs_nemo_unequal": int(np.count_nonzero((ratio != r3a) & wet)),
                "ratio_vs_nemo_max_abs": float(np.max(np.abs((ratio - r3a)[wet]))),
                "vs_nemo": substep_row(f"stage1.{face}.{name}", ref, vel, wet, reg),
                "vs_card": substep_row(f"stage1.{face}.{name}", mine, vel, wet, reg)})
        out["rows"].append({"face": face, "after_ratio": "card_live_row",
                            "vs_nemo": substep_row(f"stage1.{face}.card", ref, mine,
                                                   wet, reg)})
        if card_rhs is None:
            continue
        # the card's OWN stage-1 operands (write-only exposures of the same
        # entry): its right-hand side against NEMO's, and the written
        # statements re-run in numpy on that right-hand side
        mine_rhs, mine_raw = card_rhs[face], card_raw[face]
        with np.errstate(divide="ignore", invalid="ignore"):
            raw_nemo, raw_card = (np.where(wet, (
                (1.0 + r3b) * before + rdt * (1.0 + r3b) * x) / (1.0 + r3a) * mask, 0.0)
                for x in (rhs, mine_rhs))
        done = raw_card + (target - (e3 * raw_card) * r1) * mask
        done_from_raw = mine_raw + (target - (e3 * mine_raw) * r1) * mask
        out.setdefault("card_operand_rows", []).append({
            "face": face,
            "card_rhs_vs_nemo_rhs": substep_row("rhs", rhs, mine_rhs, wet, reg),
            "card_raw_vs_written_update_of_nemo_rhs": substep_row(
                "raw", raw_nemo, mine_raw, wet, reg),
            "card_raw_vs_written_update_of_card_rhs": substep_row(
                "raw", raw_card, mine_raw, wet, reg),
            "card_final_vs_written_statements_on_card_rhs": substep_row(
                "final", done, mine, wet, reg),
            "card_final_vs_written_correction_of_card_raw": substep_row(
                "final", done_from_raw, mine, wet, reg)})
    return out


def stage_momentum_replay(root: Path, kt: int, card, stage: int, g: dict,
                          card_fields: dict, reg, card_rhs=None,
                          card_raw=None, card_terms=None) -> dict:
    """NEMO's stage velocity from NEMO's own recorded frames, offline.

    The stage's right-hand side is ``rhs_term_replay`` (the pressure gradient
    and the vorticity term from the frame the stage reads); the update is the
    thickness-weighted statement (stprk3_stg.f90:373-378 at stages 1 and 2,
    dynzdf.f90:145-150 at stage 3, followed there by the removal of the
    barotropic velocity, :160-161, which the deck's resolved ln_drgimp = T
    switches on; the one-level implicit solve after it is the identity on
    this deck: no stress, no drag, and the interface below the only level is
    masked) and the barotropic correction
    (stprk3_stg.f90:413-414, :417-418).  The replay against NEMO's stage
    frame is the control: the stage's right-hand side is not in the record,
    so 0 unequal there is what makes the replayed right-hand side NEMO's.
    The top V row is not replayed."""
    raw = card.recipe.z_coord.nemo_een_barotropic
    a = lambda n: np.asarray(getattr(raw, n), dtype=np.float64)  # noqa: E731
    rep = rhs_term_replay(root, kt, a("fe3mask")[..., 0], stage=stage)
    rdt = {1: 1.0 / 3.0, 2: 0.5, 3: 1.0}[stage] * float(card.dt_s)
    pre, suf = ("e_", "_bb") if stage == 1 else (f"{stage - 1}_", "_aa")
    out = {"stage": stage, "rDt": rdt, "rows": []}
    for face in "uv":
        mask, e3 = a(f"{face}mask")[..., 0], a(f"e3{face}_0")[..., 0]
        wet = rep["wet_" + face]
        r3b, r3m = g[f"e_r3{face}_bb"], g[f"{pre}r3{face}{suf}"]
        r3a = g[f"{stage}_r3{face}_aa"]
        before, rhs = g[f"e_{face}{face}_k1_bb"], rep["rhs_" + face]
        target, r1 = g[f"{stage}_{face}{face}_b_aa"], _inverse_depth(a(f"h{face}_0"))
        with np.errstate(divide="ignore", invalid="ignore"):
            vel_raw = np.where(wet, (
                (1.0 + r3b) * before + rdt * (1.0 + r3m) * rhs) / (1.0 + r3a) * mask, 0.0)
        if stage == 3:      # dynzdf.f90:160-161 (ln_drgimp and ln_dynspg_ts)
            vel_raw = (vel_raw - target) * mask
        vel = vel_raw + (target - (e3 * vel_raw) * r1) * mask
        row = {"face": face,
               "control_replay_vs_nemo": substep_row(
                   f"stage{stage}.{face}", g[f"{stage}_{face}{face}_k1_aa"], vel, wet, reg),
               "card_vs_replay": substep_row(
                   f"stage{stage}.{face}", vel, card_fields[face], wet, reg)}
        if card_rhs is not None:
            row["card_rhs_vs_replay_rhs"] = substep_row(
                "rhs", rhs, card_rhs[face], wet, reg)
            row["card_raw_vs_replay_raw"] = substep_row(
                "raw", vel_raw, card_raw[face], wet, reg)
            with np.errstate(divide="ignore", invalid="ignore"):
                own = np.where(wet, ((1.0 + r3b) * before + rdt * (1.0 + r3m)
                                     * card_rhs[face]) / (1.0 + r3a) * mask, 0.0)
            if stage == 3:
                own = (own - target) * mask
            row["card_raw_vs_written_update_of_card_rhs"] = substep_row(
                "raw", own, card_raw[face], wet, reg)
            row["card_final_vs_written_correction_of_card_raw"] = substep_row(
                "final", card_raw[face] + (target - (e3 * card_raw[face]) * r1) * mask,
                card_fields[face], wet, reg)
        if card_terms is not None:      # round 9: the term split (model hook
            # expose_momentum_operator); the sum row is the split's control
            hpg, vor = card_terms["hpg"][face], card_terms["vorticity"][face]
            row["card_pressure_gradient_vs_replay"] = substep_row(
                "hpg", rep["hpg_" + face], hpg, wet, reg)
            row["card_vorticity_vs_replay"] = substep_row(
                "vor", rep["vor_" + face], vor, wet, reg)
            if card_rhs is not None:
                row["control_card_terms_sum_vs_card_rhs"] = substep_row(
                    "rhs", card_rhs[face], hpg + vor, wet, reg)
            for n, mine, got in (("hpg", rep["hpg_" + face], hpg),
                                 ("vor", rep["vor_" + face], vor)):
                # lists, not dicts: every dict in a row is a score row
                row["unequal_faces_" + n] = [[int(j), int(i)] for j, i in np.argwhere(
                    np.where(wet, mine, 0.0) != np.where(wet, got, 0.0))[:8]]
        out["rows"].append(row)
    return out


def eos_depth_rows(root: Path, kt: int, card) -> list[dict]:
    """The density anomaly of each stage at three depths, offline.

    Reference: the library density function at ``gdept_0 * (1 + r3t)`` with
    NEMO's recorded ratio of the frame the stage reads.  Candidates: the same
    function at the production ladder ``nemo_bn2_live_ladders`` evaluated on
    that frame's sea level, in its two evaluations.  T, S and the function
    are identical on both sides, so an unequal cell is the depth operand's."""
    import jax.numpy as jnp
    from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG as C
    from legoesm.ocean.eos import (
        nemo_bn2_depth_ladders, nemo_bn2_live_ladders,
        nemo_roquet_density_anomaly_ratio)
    zc, g = card.recipe.z_coord, read_step(root, kt)
    H = card.recipe.initial_state.H_bathy.data
    tm = np.asarray(zc.is_active, dtype=np.float64)
    wet = tm[..., 0] > 0.5
    gdept = np.asarray(nemo_bn2_depth_ladders(zc)[0], dtype=np.float64)
    rows = []
    for stage in (1, 2, 3):
        pre, suf = ("e_", "_bb") if stage == 1 else (f"{stage - 1}_", "_aa")
        ssh, r3t, tn, sn = (g[pre + n + suf] for n in ("ssh", "r3t", "tn_k1", "sn_k1"))

        def rhd(depth):
            return np.asarray(nemo_roquet_density_anomaly_ratio(
                jnp.asarray(tn[..., None]), jnp.asarray(sn[..., None]),
                jnp.asarray(depth), rho0=float(C.rho_0),
                geometric_depth_m=jnp.asarray(depth), tmask=jnp.asarray(tm)))[..., 0]
        ref_depth = gdept * (1.0 + r3t)[..., None]
        ref = rhd(ref_depth)
        row = {"stage": stage, "n_wet": int(wet.sum()),
               "rhd_max_abs": float(np.max(np.abs(ref[wet])))}
        for name, ev in (("card_ladder", "quotient"),
                         ("reciprocal_of_stage_ssh", "nemo_reciprocal")):
            depth = np.asarray(nemo_bn2_live_ladders(
                zc, jnp.asarray(ssh), H, r3t_evaluation=ev)[0])
            dd, dr = (depth - ref_depth)[..., 0], rhd(depth) - ref
            bad = np.argwhere((dr != 0.0) & wet)
            row[name] = {
                "depth_n_unequal": int(np.count_nonzero(dd[wet])),
                "rhd_n_unequal": int(len(bad)),
                "rhd_max_abs": float(np.max(np.abs(dr[wet]))),
                "rhd_cells": [[int(j), int(i)] for j, i in bad[:8]]}
        rows.append(row)
    return rows


def stage_face_ratio_rows(root: Path, kt: int, card) -> list[dict]:
    """One plus the U and V ratio stages 2 and 3 read as Kmm, two ways.

    NEMO's recorded ratio (the interpolation of two stored ratios) against
    the ratio of the same frame's sea level in NEMO's written stencil order.
    The top V row is not scored (its north neighbour is the fold)."""
    raw = card.recipe.z_coord.nemo_een_barotropic
    a = lambda n: np.asarray(getattr(raw, n), dtype=np.float64)  # noqa: E731
    area, g, rows = a("e1t") * a("e2t"), read_step(root, kt), []
    for stage in (2, 3):
        x = area * g[f"{stage - 1}_ssh_aa"]
        mine = ratio_stencil(x, np.roll(x, -1, axis=0), a, True)[:2]
        row = {"stage": stage}
        for k, face in enumerate("uv"):
            wet = a(f"{face}mask")[..., 0] > 0
            if face == "v":
                wet[-1] = False
            ref = g[f"{stage - 1}_r3{face}_aa"]
            bad = np.argwhere(((1.0 + mine[k]) != (1.0 + ref)) & wet)
            row[face] = {"ratio_n_unequal": int(np.count_nonzero((mine[k] - ref)[wet])),
                         "one_plus_n_unequal": int(len(bad)),
                         "faces": [[int(j), int(i)] for j, i in bad[:8]]}
        rows.append(row)
    return rows


TINY = float(np.finfo(np.float64).tiny)     # smallest normal double


def unequal_census(ref, cand) -> dict:
    """The unequal cells of one row, split by the tiny-number class."""
    ref, cand = np.asarray(ref, dtype=np.float64), np.asarray(cand, dtype=np.float64)
    bad = ref != cand
    sub_ref = (np.abs(ref) < TINY) & (ref != 0.0)
    flushed = bad & sub_ref & (cand == 0.0)
    mag = np.abs(ref[bad])
    return {"n_unequal": int(bad.sum()),
            "nemo_subnormal_card_zero": int(flushed.sum()),
            "other": int(bad.sum() - flushed.sum()),
            "nemo_subnormal_anywhere": int(sub_ref.sum()),
            "card_subnormal_anywhere": int(((np.abs(cand) < TINY) & (cand != 0.0)).sum()),
            "unequal_nemo_abs_min": float(mag.min()) if mag.size else None,
            "unequal_nemo_abs_max": float(mag.max()) if mag.size else None,
            "unequal_card_abs_max": float(np.abs(cand[bad]).max()) if mag.size else None}


def subnormal_environment() -> dict:
    """How numpy and XLA on this host treat subnormal results and operands."""
    import jax
    import jax.numpy as jnp
    a, b, s = 1.0e-200, 1.0e-110, 5.0e-310        # a * b = 1e-310 exactly-ish
    out = {"smallest_normal": TINY, "product_operands": [a, b],
           "subnormal_operand": s}

    def numpy_pair(n):
        return [float((np.full(n, a) * np.full(n, b))[0]),
                float((np.full(n, s) * np.full(n, 1.0e10))[0])]
    out["numpy_before_xla"] = numpy_pair(4)
    mul = jax.jit(lambda x, y: x * y)
    for n in (4, 1_000_000):
        out[f"xla_compiled_n{n}"] = [
            float(mul(jnp.full(n, a), jnp.full(n, b))[0]),
            float(mul(jnp.full(n, s), jnp.full(n, 1.0e10))[0])]
    out["jax_numpy_uncompiled_n4"] = [
        float((jnp.full(4, a) * jnp.full(4, b))[0]),
        float((jnp.full(4, s) * jnp.full(4, 1.0e10))[0])]
    out["numpy_after_xla"] = numpy_pair(4)
    out["xla_flushes_result"] = out["xla_compiled_n1000000"][0] == 0.0
    out["xla_flushes_operand"] = out["xla_compiled_n1000000"][1] == 0.0
    out["numpy_keeps"] = all(v != 0.0 for v in out["numpy_before_xla"] + out["numpy_after_xla"])
    return out


def ratio_fold_rows(root: Path, kt: int, card, reg) -> dict:
    """The thickness ratios on the fold rows, four ways, against NEMO's own
    recorded entry ratios; and the production coefficient builder offline."""
    import jax.numpy as jnp
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as bt
    cr = _checker()
    path = cr.step_path(root, kt)
    head, full = cr.read_stream(path.read_bytes(), path.name)
    js, je, i_s, ie = head["Njs0"] - 1, head["Nje0"], head["Nis0"] - 1, head["Nie0"]
    raw, grid = card.recipe.z_coord.nemo_een_barotropic, card.recipe.grid
    fold = grid.fold
    a = lambda n: np.asarray(getattr(raw, n), dtype=np.float64)  # noqa: E731
    ssh = np.array(full["e_ssh_bb"][js:je, i_s:ie])
    area_ssh = (a("e1t") * a("e2t")) * ssh
    perm_t = np.asarray(fold.perm_T)

    def stencil(north, nemo_order=False):
        return ratio_stencil(area_ssh, north, a, nemo_order)

    def exchange(r3):
        return tuple(np.asarray(x) for x in bt._nemo_een_ratio_fold_exchange(
            *(jnp.asarray(x) for x in r3), grid, True))

    wrap = np.roll(area_ssh, -1, axis=0)
    halo = wrap.copy()
    halo[-1] = area_ssh[-2][perm_t]
    variants = dict(zip(RATIO_VARIANTS, (
        stencil(wrap), stencil(halo), exchange(stencil(halo)), exchange(stencil(wrap)),
        exchange(stencil(wrap, nemo_order=True)))))
    nemo = {n: np.array(full[f"e_{k}"][js:je, i_s:ie])
            for n, k in (("r3u", "r3u_bb"), ("r3v", "r3v_bb"), ("r3f", "r3f"))}
    bands = {"pivot_row": np.s_[-1], "row_below": np.s_[-2], "all_rows_below": np.s_[:-2]}
    rows = []
    for name, r3 in variants.items():
        for (ratio, ref), cand in zip(nemo.items(), r3):
            d = cand - ref
            rows.append({"variant": name, "ratio": ratio, **{
                band: {"n_unequal": int(np.count_nonzero(d[sl])),
                       "max_abs": float(np.max(np.abs(d[sl]))),
                       "nemo_max_abs": float(np.max(np.abs(ref[sl]))),
                       "n_nemo_nonzero": int(np.count_nonzero(ref[sl]))}
                for band, sl in bands.items()}})
    out = {"kt": kt, "ratio_rows": rows,
           "nemo_halo_ssh_equals_mirrored_row_below": bool(np.array_equal(
               full["e_ssh_bb"][je, i_s:ie], ssh[-2][perm_t])),
           "L3_equals_L2_bitwise": all(np.array_equal(x, y) for x, y in zip(
               variants[RATIO_VARIANTS[2]], variants[RATIO_VARIANTS[3]]))}
    if cr.substep_path(root, kt).is_file():
        _, sub = read_substep_record(root, kt)
        wet = {"u": a("umask")[..., 0] > 0, "v": a("vmask")[..., 0] > 0}
        out["coefficient_rows"] = []
        for pair in ((False, False), (True, False), (False, True), (True, True)):
            built = bt._nemo_literal_een_coefficients(
                jnp.asarray(ssh), card.recipe.z_coord, jnp.float64, scheme="een",
                grid=grid, fold_pivot_row=pair)
            out["coefficient_rows"] += [dict(substep_row(
                f"i000.{n}", sub["i000_" + n], np.asarray(built[n]), wet[n[2]], reg),
                ratio_exchange=pair[0], fe3mask_pivot_row=pair[1],
                operand=n) for n in (
                "ffu_nw", "ffu_ne", "ffu_sw", "ffu_se",
                "ffv_nw", "ffv_ne", "ffv_sw", "ffv_se")]
    return out


def ratio_order_rows(root: Path, kt: int, card, plant: str | None = None) -> dict:
    """The solver's thickness ratios and V-face Coriolis coefficients at one
    step: NEMO's statements replayed in numpy against NEMO's record (the
    controls) and against the production coefficient builder."""
    import jax.numpy as jnp
    from legoesm.ocean.dynamics import barotropic_latlon_cgrid as bt
    g = read_step(root, kt)
    raw, grid = card.recipe.z_coord.nemo_een_barotropic, card.recipe.grid
    a = lambda n: np.asarray(getattr(raw, n), dtype=np.float64)  # noqa: E731
    ssh = np.array(g["e_ssh_bb"])
    area_ssh = (a("e1t") * a("e2t")) * ssh
    north = np.roll(area_ssh, -1, axis=0)        # the top row is not scored
    nemo = (g["e_r3u_bb"], g["e_r3v_bb"], g["e_r3f"])
    written = ratio_stencil(area_ssh, north, a, True)
    divided = ratio_stencil(area_ssh, north, a, False)
    below = np.s_[:-1]
    ratios = [{"ratio": name,
               "written_order_vs_nemo": int(np.count_nonzero((w != ref)[below])),
               "divided_vs_nemo": int(np.count_nonzero((d != ref)[below])),
               "one_plus_ratio_divided_vs_nemo": int(np.count_nonzero(
                   ((1.0 + d) != (1.0 + ref))[below])),
               "n": int(ref[below].size)}
              for name, w, d, ref in zip(("r3u", "r3v", "r3f"), written, divided, nemo)]
    # dyn_cor_2D_init, V faces (dynspg_ts.f90:987-1008), one level, from
    # NEMO's recorded ratios; open faces only: every T cell within two cells
    # wet, three rows off the south wall and the fold, so that e3f_0vor is
    # e3f_0, fe3mask is 1 and no neighbour is a halo
    um, vm = (a(n)[..., 0] for n in ("umask", "vmask"))
    tm = np.asarray(card.recipe.z_coord.is_active, dtype=np.float64)[..., 0]
    w_, s_, n_ = (lambda x: np.roll(x, 1, 1)), (lambda x: np.roll(x, 1, 0)), (
        lambda x: np.roll(x, -1, 0))
    open_faces = np.ones(tm.shape, dtype=bool)
    for dj in range(-2, 3):
        for di in range(-2, 3):
            open_faces &= np.roll(np.roll(tm, dj, 0), di, 1) > 0
    open_faces[:3] = open_faces[-3:] = False
    q = a("ff_f") / (a("e3f_0")[..., 0] * (1.0 + nemo[2]))
    e3u = a("e3u_0")[..., 0] * (1.0 + nemo[0] * um)
    e3v = a("e3v_0")[..., 0] * (1.0 + nemo[1] * vm)
    lead = (1.0 / 12.0) * (1.0 / a("e2v")) * (_inverse_depth(a("hv_0")) / (1.0 + nemo[1]))
    triads = {"ffv_nw": (q + w_(q)) + n_(w_(q)), "ffv_ne": (n_(q) + q) + w_(q),
              "ffv_sw": (s_(w_(q)) + w_(q)) + q, "ffv_se": (w_(q) + q) + s_(q)}
    at = {"ffv_nw": lambda x: n_(w_(x)), "ffv_ne": n_, "ffv_sw": w_, "ffv_se": lambda x: x}
    replay = {k: lead * at[k](a("e2u")) * (e3v * at[k](e3u) * at[k](um) * z)
              for k, z in triads.items()}
    if plant == "score":
        j, i = np.argwhere(open_faces)[0]
        replay["ffv_nw"][j, i] *= 2.0
    recorded = (read_substep_record(root, kt)[1]
                if _checker().substep_path(root, kt).is_file() else None)
    coefficients = []
    for selected in (False, True):
        built = bt._nemo_literal_een_coefficients(
            jnp.asarray(ssh), card.recipe.z_coord, jnp.float64, scheme="een",
            grid=grid, fold_pivot_row=(True, True, selected))
        for k, ref in replay.items():
            d = (np.asarray(built[k]) != ref) & open_faces
            coefficients.append({
                "written_order_option": selected, "operand": k,
                "builder_vs_replay": int(d.sum()),
                "cells": [[int(x) for x in c] for c in np.argwhere(d)[:8]],
                "replay_vs_nemo_recorded": None if recorded is None else int(
                    np.count_nonzero((recorded["i000_" + k] != ref) & open_faces))})
    return {"kt": kt, "ratio_rows": ratios, "n_open_faces": int(open_faces.sum()),
            "coefficient_rows": coefficients}


SUB_REGIONS = ("fold", "seam", "south", "interior")


def substep_row(name, ref, cand, wet, reg, *, plant=False) -> dict:
    """One operand on its own wet points: unequal count and largest error,
    whole domain and per region (the fold band also row by row)."""
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.array(cand, dtype=np.float64)
    require(ref.shape == cand.shape == wet.shape,
            f"{name}: extents {ref.shape} / {cand.shape} / {wet.shape}")
    if plant:
        j, i = np.argwhere(wet & reg["interior"])[0]
        cand[j, i] += 1.0
    d = np.where(wet, cand - ref, 0.0)
    worst = np.unravel_index(int(np.argmax(np.abs(d))), d.shape)

    def part(m):
        return {"n_unequal": int(np.count_nonzero(d[m])),
                "max_abs": float(np.max(np.abs(d[m]))),
                "ref_max_abs": float(np.max(np.abs(ref[m & wet]), initial=0.0))}
    return {"name": name, "n_wet": int(wet.sum()),
            "n_unequal": int(np.count_nonzero(d)),
            "max_abs": float(np.max(np.abs(d))),
            "max_abs_cell": [int(x) for x in worst],
            "ref_max_abs": float(np.max(np.abs(ref[wet]))),
            "bit_identical": not np.any(d),
            "regions": {k: part(reg[k]) for k in (
                *SUB_REGIONS, *(n for n, _ in FOLD_ROW_NAMES))}}


def substep_table(g: dict, trace: dict, n_sub: int, wet: dict, reg,
                  plant: str | None = None) -> dict:
    """Every recorded operand of the loop against the model's trace, in
    NEMO's compiled order (dynspg_ts.f90:359 on); the order and the trace
    keys are the VORTEX walk's."""
    from nemo_testcase_l1_vortex_round196_spgts_walk import (
        ENTRY_ORDER, SCALAR_ORDER, SUBSTEP_ORDER, _lego_plane)
    g = dict(g)
    for jn in range(1, n_sub + 1):      # the compiled swap, dynspg_ts.f90:817-822
        prev = f"j{jn - 1:03d}_"
        for eff, at_entry, later in (("sshn_e_eff", "sshn_e", "ssha_e"),
                                     ("un_e_eff", "un_e", "ua_new"),
                                     ("vn_e_eff", "vn_e", "va_new")):
            g[f"j{jn:03d}_{eff}"] = g["i000_" + at_entry] if jn == 1 else g[prev + later]
    rows = []

    def add(name, group, key, stagger, jn):
        rows.append(dict(substep_row(
            f"j{jn:03d}.{name}", g[group],
            _lego_plane(trace[key][max(jn, 1) - 1], stagger), wet[stagger], reg,
            plant=(plant == "score" and jn == 1 and name == "mid.eta")),
            substep=jn, operand=name))

    for name, group, key, stagger in (
            *ENTRY_ORDER,
            ("entry.sshb", "sshb_e", "eta_history_b", "t"),
            ("entry.sshbb", "sshbb_e", "eta_history_bb", "t"),
            ("entry.ub", "ub_e", "u_history_b", "u"),
            ("entry.ubb", "ubb_e", "u_history_bb", "u"),
            ("entry.vb", "vb_e", "v_history_b", "v"),
            ("entry.vbb", "vbb_e", "v_history_bb", "v"),
            # the EEN triads: U-point sets on U faces, V-point sets on V faces
            *((f"entry.{n}", n, n, "t") for n in (
                "ffu_nw", "ffu_ne", "ffu_sw", "ffu_se",
                "ffv_nw", "ffv_ne", "ffv_sw", "ffv_se"))):
        mask_key = {"ffu": "u", "ffv": "v"}.get(name[6:9], stagger)
        rows.append(dict(substep_row(
            f"i000.{name}", g["i000_" + group],
            _lego_plane(trace[key][0], stagger), wet[mask_key], reg),
            substep=0, operand=name))
    scalars = []
    for jn in range(1, n_sub + 1):
        for name, group, index, key in SCALAR_ORDER:
            a, b = float(g[f"j{jn:03d}_{group}"][index]), float(trace[key][jn - 1])
            scalars.append({"substep": jn, "name": name, "nemo": a, "card": b,
                            "bit_identical": a == b})
        for name, group, key, stagger in SUBSTEP_ORDER:
            add(name, f"j{jn:03d}_{group}", key, stagger, jn)
    first = next((r for r in rows if not r["bit_identical"]), None)
    return {"rows": rows, "scalars_unequal": [x for x in scalars if not x["bit_identical"]],
            "n_scalars": len(scalars),
            "first_unequal": first and {k: first[k] for k in (
                "name", "n_unequal", "max_abs", "max_abs_cell", "ref_max_abs", "regions")}}


@contextlib.contextmanager
def vor_fold_image_arm(card, calls: list):
    """MEASUREMENT ARM, not the card.  Inside the 3-D vorticity trend only,
    hand the triad operator NEMO's fold-image V row: velocity
    v(jj = Nje0) = -v(mirror, Nje0 - 1), its face thickness (same mirror,
    sign +) and NEMO's raw wet mask on that row (dynvor.f90:767 reads
    vv, e3v and vmask there).  The barotropic solver's own triad call is a
    different module binding and is not touched.  ``calls`` counts traces."""
    import jax.numpy as jnp
    from legoesm.ocean.dynamics import ocean_pe_latlon_cgrid as pe
    top = jnp.asarray(np.asarray(card.recipe.z_coord.nemo_een_barotropic.vmask)[-1])
    n = top.shape[0]
    perm = (n - np.arange(n)) % n
    original = pe.pv_flux_al81_partial_cell

    def wrapped(zeta, h_vtx, h_v, v, h_u, u, u_mask_3d, v_mask_3d, vtx_mask, **kw):
        calls.append(1)
        v = v.at[-1].set(-v[-2][perm] * top)
        h_v = h_v.at[-1].set(h_v[-2][perm])
        v_mask_3d = v_mask_3d.at[-1].set(top.astype(v_mask_3d.dtype))
        return original(zeta, h_vtx, h_v, v, h_u, u, u_mask_3d, v_mask_3d,
                        vtx_mask, **kw)

    pe.pv_flux_al81_partial_cell = wrapped
    try:
        yield
    finally:
        pe.pv_flux_al81_partial_cell = original


# ------------------------------------------------------------------ ladders
def run(root: Path, *, arm: str, kt_max: int = 10, kt_min: int = 1,
        plant: str | None = None,
        allow_dirty: bool = False, geometry_plant: str | None = None,
        fold_fill_v: bool = False, vor_fold_image: bool = False,
        before_arm: bool = False,
        hook_arm: tuple = (), without_fold_v_depth: bool = False,
        ts_from_chain: bool = False, without_stage1_ssh: bool = False,
        without_flux_literal: bool = False,
        without_substep_dt: bool = False,
        without_entry_depth: bool = False,
        without_ratio_fold: bool = False,
        stage1_rhs_from_nemo: bool = False,
        without_stage1_rhs: bool = False,
        without_written_order: bool = False,
        without_vor_order: bool = False,
        without_zdf_residual: bool = False,
        without_fold_exchange_v: bool = False,
        without_stage_r3f: bool = False, without_stage_r3t: bool = False,
        without_eos_gdept: bool = False, without_stage_r3uv: bool = False,
        without_ratio_order: bool = False,
        without_r3t_product: bool = False,
        without_nemo_zeta: bool = False,
        without_fold_unit_vec: bool = False) -> dict:
    if vor_fold_image:
        from legoesm.ocean.fidelity import nemo_testcase_recipe as _recipe
        calls: list = []
        with vor_fold_image_arm(_recipe.build_nemo_testcase_card(
                CASE, deck_root=root), calls):
            out = run(root, arm=arm, kt_max=kt_max, kt_min=kt_min, plant=plant,
                      allow_dirty=allow_dirty, geometry_plant=geometry_plant,
                      fold_fill_v=fold_fill_v, before_arm=before_arm,
                      hook_arm=hook_arm,
                      without_fold_v_depth=without_fold_v_depth,
                      ts_from_chain=ts_from_chain,
                      without_stage1_ssh=without_stage1_ssh,
                      without_flux_literal=without_flux_literal,
                      without_substep_dt=without_substep_dt,
                      without_entry_depth=without_entry_depth,
                      without_ratio_fold=without_ratio_fold,
                      without_stage1_rhs=without_stage1_rhs,
                      without_written_order=without_written_order,
                      without_vor_order=without_vor_order,
                      without_zdf_residual=without_zdf_residual,
                      without_fold_exchange_v=without_fold_exchange_v,
                      without_stage_r3f=without_stage_r3f, without_stage_r3t=without_stage_r3t,
                      without_eos_gdept=without_eos_gdept,
                      without_stage_r3uv=without_stage_r3uv,
                      without_ratio_order=without_ratio_order,
                      without_r3t_product=without_r3t_product,
                      without_nemo_zeta=without_nemo_zeta,
                      without_fold_unit_vec=without_fold_unit_vec)
        require(len(calls) > 0, "the vorticity fold-image arm was never traced")
        out["vor_fold_image_measurement_arm"] = {"traces": len(calls)}
        out["label"] = f"{out['label']} + MEASUREMENT ARM (fold-image V row "\
                       "in the 3-D vorticity trend; not the card)"
        return out
    require(arm in ARMS, f"unknown arm {arm!r}; expected one of {ARMS}")
    if arm == "subnormal_environment":
        _fp64()
        return {"arm": arm, "label": "this process's arithmetic (no record, "
                "no model step)", **subnormal_environment()}
    require(not stage1_rhs_from_nemo or arm == "stages",
            "--stage1-rhs-from-nemo is a measurement arm of the stages table")
    require(plant is None or plant in PLANTS, f"unknown plant {plant!r}")
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.fidelity.provenance import allow_dirty_stamps, git_sha

    allow_dirty_stamps(allow_dirty)
    sha = git_sha(allow_dirty=allow_dirty)
    _fp64()
    if arm == "geometry":
        out = geometry(root, geometry_plant)
        out["legoesm_git_sha"] = sha
        return out

    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe

    card = recipe.build_nemo_testcase_card(CASE, deck_root=root)
    if without_r3t_product:   # ROUND-8 BEFORE ARM / plant: the bare product
        require(card.recipe.z_coord.nemo_r3t_stored_product is True,
                "the card no longer selects the stored T-point ratio")
        card = card._replace(recipe=card.recipe._replace(
            z_coord=card.recipe.z_coord._replace(nemo_r3t_stored_product=False)))
    cfg = card.recipe.model_config
    if before_arm:      # BEFORE ARM / plant: the card without its fold-image row
        require(cfg.een_fold_image_v_row is True, "the card no longer selects the row")
        cfg = cfg._replace(een_fold_image_v_row=False)
    if without_fold_v_depth:  # ROUND-3 BEFORE ARM / plant: the solver's face
        # depth on the fold-image V row without sea level (hv_0 only)
        require(cfg.barotropic.barotropic_fold_top_v_face_ssh is True,
                "the card no longer selects the fold-row V face depth")
        cfg = cfg._replace(barotropic=cfg.barotropic._replace(
            barotropic_fold_top_v_face_ssh=False))
    if without_stage1_ssh:    # ROUND-4 BEFORE ARM / plant: the legacy order
        require(cfg.rk3_stage1_ssh_nemo_literal is True,
                "the card no longer selects NEMO's stage-1 sea level")
        cfg = cfg._replace(rk3_stage1_ssh_nemo_literal=False)
    if without_entry_depth or without_flux_literal:
        # ROUND-5 BEFORE ARM / plant: rest depth plus face-mean sea level
        # (the update's own before arm drops it too: only the update reads it)
        require(cfg.barotropic.barotropic_entry_depth_nemo_literal is True,
                "the card no longer selects NEMO's solver entry depth")
        cfg = cfg._replace(barotropic=cfg.barotropic._replace(
            barotropic_entry_depth_nemo_literal=False))
    if without_flux_literal:  # ROUND-5 BEFORE ARM / plant: divide by the depth
        require(cfg.barotropic.barotropic_flux_update_nemo_literal is True,
                "the card no longer selects NEMO's flux-form velocity update")
        cfg = cfg._replace(barotropic=cfg.barotropic._replace(
            barotropic_flux_update_nemo_literal=False))
    if without_substep_dt:    # ROUND-5 BEFORE ARM / plant: the compiled product
        require(cfg.barotropic.barotropic_substep_dt_nemo_literal is True,
                "the card no longer selects NEMO's substep length")
        cfg = cfg._replace(barotropic=cfg.barotropic._replace(
            barotropic_substep_dt_nemo_literal=False))
    for off, name in ((without_stage1_rhs, "rk3_stage1_rhs_nemo_literal"),
                      (without_written_order, "rk3_stage_velocity_written_order"),
                      (without_vor_order, "een_vorticity_written_order"),
                      (without_zdf_residual, "zdf_barotropic_removal_nemo_literal"),
                      (without_fold_exchange_v, "rk3_fold_exchange_top_v_row"),
                      (without_stage_r3f, "rk3_stage_r3f_nemo_literal"),
                      (without_stage_r3t, "rk3_stage_r3t_nemo_literal"),  # round 9
                      (without_eos_gdept, "nemo_eos_gdept_rounded_product"),
                      (without_stage_r3uv, "rk3_stage_r3uv_nemo_literal"),
                      (without_nemo_zeta, "een_nemo_relative_vorticity")):  # G1b round 2
        if off:               # ROUND-6 BEFORE ARMS / plants, one option each
            require(getattr(cfg, name) is True, f"the card no longer selects {name}")
            cfg = cfg._replace(**{name: False})
    if without_fold_unit_vec:  # G1b ROUND-2 BEFORE ARM / plant
        require(cfg.barotropic.barotropic_literal_fold_unit_vector_form is True,
                "the card no longer selects the fold unit under vector form")
        cfg = cfg._replace(barotropic=cfg.barotropic._replace(
            barotropic_literal_fold_unit_vector_form=False))
    if without_ratio_order:   # ROUND-8 BEFORE ARM / plant: divide by the area
        require(cfg.barotropic.barotropic_een_ratio_written_order is True,
                "the card no longer selects the written-order ratios")
        cfg = cfg._replace(barotropic=cfg.barotropic._replace(
            barotropic_een_ratio_written_order=False))
    if without_ratio_fold:    # ROUND-6 BEFORE ARM / plant: the bare stencil
        require(cfg.barotropic.barotropic_een_ratio_fold_exchange is True,
                "the card no longer selects the ratio fold exchange")
        cfg = cfg._replace(barotropic=cfg.barotropic._replace(
            barotropic_een_ratio_fold_exchange=False,
            barotropic_een_fe3mask_pivot_row=False))
    s0 = card.recipe.initial_state
    require(int(card.recipe.z_coord.n_levels) == 1, "the G1 card executes one level")
    n_stage = stage_steps(root)
    if arm == "fold_v_depth":
        require(1 <= kt_max <= n_stage, f"kt_max must lie in 1..{n_stage}")
        return {"arm": arm, "legoesm_git_sha": sha, "root": str(root),
                "label": "GIVEN-NEMO-ENTRY (offline operand: ssh term of the "
                         "solver's V face depth; no model step)",
                "een_fold_image_v_row": bool(cfg.een_fold_image_v_row),
                "barotropic_fold_top_v_face_ssh": bool(
                    cfg.barotropic.barotropic_fold_top_v_face_ssh),
                "per_kt": [{"kt": kt, "operand_rows": fold_v_depth_rows(
                    root, kt, card, cfg)} for kt in range(1, kt_max + 1)]}
    if arm == "stage_replay":
        require(1 <= kt_min <= kt_max <= n_stage, f"steps must lie in 1..{n_stage}")
        tm = np.asarray(card.recipe.z_coord.is_active)[..., 0]
        return {"arm": arm, "legoesm_git_sha": sha, "root": str(root),
                "label": "NEMO's record only (offline replay of the stage "
                         "statements; no model step)",
                "per_kt": [{"kt": kt, "replay": stage_replay_rows(root, kt, tm)}
                           for kt in range(kt_min, kt_max + 1)]}
    if arm == "stage_operand_replay":
        require(1 <= kt_min <= kt_max <= n_stage, f"steps must lie in 1..{n_stage}")
        return {"arm": arm, "legoesm_git_sha": sha, "root": str(root),
                "label": "NEMO's record only (offline: the stage density "
                         "anomaly at NEMO's stored-ratio depth against the "
                         "production depth ladder; no model step)",
                "per_kt": [{"kt": kt, "rows": eos_depth_rows(root, kt, card),
                            "face_ratio_rows": stage_face_ratio_rows(root, kt, card)}
                           for kt in range(kt_min, kt_max + 1)]}
    if arm == "flux_update_replay":
        require(kt_min == kt_max, "one step: --kt-min equal to --kt-max")
        reg = regions(np.asarray(card.recipe.z_coord.is_active)[..., 0])
        rows = flux_update_replay_rows(root, kt_max, reg)
        return {"arm": arm, "legoesm_git_sha": sha, "root": str(root), "kt": kt_max,
                "label": "NEMO's record only (offline replay of the flux-form "
                         "velocity update; no model step)",
                "replay_rows": rows,
                "totals": {name: {"n_unequal": sum(r[name]["n_unequal"] for r in rows),
                                  "max_abs": max(r[name]["max_abs"] for r in rows)}
                           for name in REPLAY_ORDERS}}
    if arm == "ratio_order_replay":
        require(1 <= kt_min <= kt_max <= n_stage, f"steps must lie in 1..{n_stage}")
        return {"arm": arm, "legoesm_git_sha": sha, "root": str(root), "plant": plant,
                "label": "NEMO's record only (offline replay of the solver's "
                         "thickness ratios and V-face coefficients; no model step)",
                "per_kt": [ratio_order_rows(root, kt, card, plant)
                           for kt in range(kt_min, kt_max + 1)]}
    if arm == "ratio_fold_replay":
        require(1 <= kt_min == kt_max <= n_stage, "one step holding stage frames")
        reg = regions(np.asarray(card.recipe.z_coord.is_active)[..., 0])
        return {"arm": arm, "legoesm_git_sha": sha, "root": str(root),
                "label": "NEMO's record only (offline leaf table of the "
                         "thickness ratios on the fold rows; no model step)",
                **ratio_fold_rows(root, kt_max, card, reg)}
    if arm == "interior_events":
        require(2 <= kt_max <= 30, "kt_max must lie in 2..30")
        wet = np.asarray(card.recipe.z_coord.is_active)[..., 0] > 0.5
        dist, nj = land_distance(wet), wet.shape[0]
        reg = regions(wet.astype(float))
        m, st, events = LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg), s0, []
        prev = None
        for kt in range(1, kt_max + 1):
            entry = st
            st = jax.device_get(m.step(st, dt=card.dt_s))
            g = read_step(root, kt)
            nemo, mine = g["f_ssh_bb"], np.asarray(st.eta.data)
            err = np.where(wet, mine - nemo, 0.0)
            over = np.abs(err) > BAR
            ev = {"kt": kt, "n_over_bar": int(over.sum()),
                  "max_abs": float(np.max(np.abs(err)))}
            if over.any() and prev is not None:
                j, i = (int(x) for x in np.unravel_index(
                    int(np.argmax(np.abs(err))), err.shape))
                flip = np.sign(nemo) != np.sign(prev["f_ssh_bb"])
                seeded = entry._replace(
                    u=s0.u.replace(data=jnp.asarray(pad_u(prev["f_uu_k1_bb"])[..., None])),
                    v=s0.v.replace(data=jnp.asarray(pad_v(prev["f_vv_k1_bb"])[..., None])),
                    eta=s0.eta.replace(data=jnp.asarray(prev["f_ssh_bb"])),
                    uu_b=s0.uu_b.replace(data=jnp.asarray(pad_u(prev["f_uu_b_bb"]))),
                    vv_b=s0.vv_b.replace(data=jnp.asarray(pad_v(prev["f_vv_b_bb"]))))
                one = np.asarray(jax.device_get(
                    m.step(seeded, dt=card.dt_s)).eta.data) - nemo
                e0 = np.asarray(entry.eta.data) - prev["f_ssh_bb"]
                ev.update({
                    "cell": [j, i], "error": float(err[j, i]),
                    "nemo_ssh": float(nemo[j, i]),
                    "entry_error_at_cell": float(e0[j, i]),
                    "cells_to_land": int(dist[j, i]),
                    "rows_to_fold": nj - 1 - j,
                    "region": next(k for k in ("fold", "seam", "south",
                                               "interior_coast", "interior_open")
                                   if reg[k][j, i]),
                    "one_step_from_nemo_entry_error_at_cell": float(one[j, i]),
                    "over_bar_with_sign_change": int((over & flip).sum()),
                    "wet_cells_with_sign_change": int((wet & flip).sum()),
                    "n_wet": int(wet.sum()),
                    "over_bar_by_region": {k: int((over & reg[k]).sum()) for k in (
                        "fold", "seam", "south", "interior_coast", "interior_open")},
                    "min_cells_to_land_over_bar": int(dist[over].min())})
            events.append(ev)
            prev = g
        return {"arm": arm, "legoesm_git_sha": sha, "root": str(root), "bar": BAR,
                "label": "INDEPENDENT (census of sea-level cells over the bar; "
                         "one-step re-run from NEMO's previous after-frame at "
                         "the largest cell)",
                "een_fold_image_v_row": bool(cfg.een_fold_image_v_row),
                "barotropic_fold_top_v_face_ssh": bool(
                    cfg.barotropic.barotropic_fold_top_v_face_ssh),
                "per_kt": events}
    cap = 100 if arm in ("independent", "restart_entry") else n_stage
    require(1 <= kt_max <= cap, f"kt_max must lie in 1..{cap}")
    require(1 <= kt_min <= kt_max, "kt_min must lie in 1..kt_max")
    require(kt_min == 1 or arm not in ("independent", "restart_entry"),
            "--kt-min is for the stage-frame arms (the chain always starts at 1)")
    kts = range(1, kt_max + 1)
    scored = range(kt_min, kt_max + 1)   # the chain runs over kts; rows over scored
    reg = regions(np.asarray(card.recipe.z_coord.is_active)[..., 0])
    raw = card.recipe.z_coord.nemo_een_barotropic
    wet_face = {"u": np.asarray(raw.umask)[..., 0] > 0,
                "v": np.asarray(raw.vmask)[..., 0] > 0}
    require(not fold_fill_v or arm == "independent",
            "--fold-fill-v is a measurement arm of the independent chain only")
    n_lon = wet_face["v"].shape[1]
    fold_perm = (n_lon - np.arange(n_lon)) % n_lon

    def fill_top_v(st):
        """MEASUREMENT ARM, not the card: between steps, write NEMO's fold
        image into the top V row of the 3-D velocity -- v(jj = Nje0) =
        -v(mirror, jj = Nje0 - 1) on NEMO's wet faces, the identity NEMO's
        own record satisfies bit for bit (checked below)."""
        v = np.array(st.v.data)
        v[-1, :, 0] = -v[-2, fold_perm, 0] * wet_face["v"][-1]
        return st._replace(v=st.v.replace(data=jnp.asarray(v)))

    def model(hooks=None):
        if hooks is None and hook_arm:
            # MEASUREMENT ARM, not the card: private boolean test hooks of the
            # model switched on for every step of this arm
            hooks = _NEMOWSRK3TestHooks(**{n: True for n in hook_arm})
        return LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            **({} if hooks is None else {"_nemo_ws_test_hooks": hooks}))

    def seed(g, base, kt=None):
        hist = None if kt is None else nemo_histories(root, kt)
        if hist is not None:        # NEMO's own histories where a restart exists
            base = base._replace(bt_hist=hist)
            if kt not in out.setdefault("nemo_histories_at_kt", []):
                out["nemo_histories_at_kt"].append(kt)
        eta = np.array(g["e_ssh_bb"])
        if plant == "entry":
            j, i = np.unravel_index(int(np.argmax(np.abs(eta))), eta.shape)
            eta[j, i] += 1.0e-3
        if not ts_from_chain:
            base = base._replace(
                T=s0.T.replace(data=jnp.asarray(g["e_tn_k1_bb"][..., None])),
                S=s0.S.replace(data=jnp.asarray(g["e_sn_k1_bb"][..., None])))
        return base._replace(
            u=s0.u.replace(data=jnp.asarray(pad_u(g["e_uu_k1_bb"])[..., None])),
            v=s0.v.replace(data=jnp.asarray(pad_v(g["e_vv_k1_bb"])[..., None])),
            eta=s0.eta.replace(data=jnp.asarray(eta)),
            uu_b=s0.uu_b.replace(data=jnp.asarray(pad_u(g["e_uu_b_bb"]))),
            vv_b=s0.vv_b.replace(data=jnp.asarray(pad_v(g["e_vv_b_bb"]))))

    out = {"arm": arm, "case": CASE, "label": None, "legoesm_git_sha": sha, "plant": plant,
           "bar": BAR, "root": str(root), "kts": list(scored),
           "stage_steps": n_stage,
           "fold_fill_v_measurement_arm": fold_fill_v,
           "een_fold_image_v_row": bool(cfg.een_fold_image_v_row),
           "barotropic_fold_top_v_face_ssh": bool(
               cfg.barotropic.barotropic_fold_top_v_face_ssh),
           "n_barotropic_substeps": int(cfg.barotropic.n_barotropic_substeps),
           "region_cells": {k: int(m.sum()) for k, m in reg.items()},
           "hook_measurement_arm": list(hook_arm),
           "ts_from_chain_measurement_arm": ts_from_chain,
           "rk3_stage1_ssh_nemo_literal": bool(cfg.rk3_stage1_ssh_nemo_literal),
           "barotropic_flux_update_nemo_literal": bool(
               cfg.barotropic.barotropic_flux_update_nemo_literal),
           "barotropic_substep_dt_nemo_literal": bool(
               cfg.barotropic.barotropic_substep_dt_nemo_literal),
           "barotropic_entry_depth_nemo_literal": bool(
               cfg.barotropic.barotropic_entry_depth_nemo_literal),
           "barotropic_een_ratio_fold_exchange": bool(
               cfg.barotropic.barotropic_een_ratio_fold_exchange),
           "stage1_rhs_from_nemo_measurement_arm": stage1_rhs_from_nemo,
           "rk3_stage1_rhs_nemo_literal": bool(cfg.rk3_stage1_rhs_nemo_literal),
           "rk3_stage_velocity_written_order": bool(
               cfg.rk3_stage_velocity_written_order),
           "een_vorticity_written_order": bool(cfg.een_vorticity_written_order),
           "zdf_barotropic_removal_nemo_literal": bool(
               cfg.zdf_barotropic_removal_nemo_literal),
           "rk3_fold_exchange_top_v_row": bool(cfg.rk3_fold_exchange_top_v_row),
           "rk3_stage_r3f_nemo_literal": bool(cfg.rk3_stage_r3f_nemo_literal),
           "rk3_stage_r3t_nemo_literal": bool(cfg.rk3_stage_r3t_nemo_literal),
           "nemo_eos_gdept_rounded_product": bool(
               cfg.nemo_eos_gdept_rounded_product),
           "rk3_stage_r3uv_nemo_literal": bool(cfg.rk3_stage_r3uv_nemo_literal),
           "een_nemo_relative_vorticity": bool(cfg.een_nemo_relative_vorticity),
           "barotropic_literal_fold_unit_vector_form": bool(
               cfg.barotropic.barotropic_literal_fold_unit_vector_form),
           "barotropic_een_ratio_written_order": bool(
               cfg.barotropic.barotropic_een_ratio_written_order),
           "nemo_r3t_stored_product": bool(
               card.recipe.z_coord.nemo_r3t_stored_product),
           "per_kt": []}
    records = {kt: read_step(root, kt) for kt in scored if kt <= n_stage}
    before = {}             # the card's own chain: the state entering step kt
    if arm == "restart_entry":
        import netCDF4
        out["label"] = ("GIVEN-NEMO-ENTRY from NEMO's restart (one step: all "
                        "prognostics and the six barotropic histories; the "
                        "after-ssh slot is not seeded and not read by this card)")
        require(getattr(cfg, "nemo_first_wzv_after_ssh", "") != "rk3_extrapolated_carried",
                "this card reads the carried after-ssh slot; seed NEMO's ssha first")
        m, chain = model(), s0
        for kt in kts:
            # every step whose previous restart the record holds (G1's
            # record: every tenth; G1b's merged root: every step NEMO wrote)
            if kt > 1 and (root / f"ORCA2G1_{kt - 1:08d}_restart.nc").is_file():
                k0 = kt - 1
                path = root / f"ORCA2G1_{k0:08d}_restart.nc"
                require(path.is_file(), f"missing {path}")
                g0 = read_step(root, k0)
                with netCDF4.Dataset(path) as h:
                    def var(n):
                        return np.asarray(h.variables[n][0], dtype=np.float64)
                    # the restart IS the recorded after-step state, bit for bit
                    for n, a, b in (("sshn", var("sshn"), g0["f_ssh_bb"]),
                                    ("un", var("un")[0], g0["f_uu_k1_bb"]),
                                    ("vn", var("vn")[0], g0["f_vv_k1_bb"]),
                                    ("uu_n", var("uu_n"), g0["f_uu_b_bb"]),
                                    ("vv_n", var("vv_n"), g0["f_vv_b_bb"])):
                        require(np.array_equal(a, b), f"kt {k0}: restart {n} != frame")
                    require(chain.eta_rk3_after is None,
                            "the card carries an after-ssh slot the seed does not set")
                    entry = chain._replace(
                        T=s0.T.replace(data=jnp.asarray(var("tn")[0][..., None])),
                        S=s0.S.replace(data=jnp.asarray(var("sn")[0][..., None])),
                        u=s0.u.replace(data=jnp.asarray(pad_u(var("un")[0])[..., None])),
                        v=s0.v.replace(data=jnp.asarray(pad_v(var("vn")[0])[..., None])),
                        eta=s0.eta.replace(data=jnp.asarray(var("sshn"))),
                        uu_b=s0.uu_b.replace(data=jnp.asarray(pad_u(var("uu_n")))),
                        vv_b=s0.vv_b.replace(data=jnp.asarray(pad_v(var("vv_n")))),
                        bt_hist=nemo_histories(root, kt))
                g = read_step(root, kt)
                cand = state_fields(jax.device_get(m.step(entry, dt=card.dt_s)))
                rows = [row(f"restart_entry.kt{kt}.{f}", g[nm], cand[f], reg)
                        for f, nm in (("ssh", "f_ssh_bb"), ("uu_b", "f_uu_b_bb"),
                                      ("vv_b", "f_vv_b_bb"), ("u", "f_uu_k1_bb"),
                                      ("v", "f_vv_k1_bb"))]
                out["per_kt"].append({"kt": kt, "rows": rows,
                                      "first_unequal": first_unequal(rows)})
            chain = jax.device_get(m.step(chain, dt=card.dt_s))
        out["unrecorded_carry"] = "w only (3.7e-21 at kt 2); layout from the card's chain"
        return out
    if arm in ("substeps", "substeps_independent"):
        require(kt_min == kt_max, "one step: --kt-min equal to --kt-max")
        kt = kt_max
        head, sub = read_substep_record(root, kt)
        n_sub = head["icycle"]
        require(n_sub == int(cfg.barotropic.n_barotropic_substeps),
                f"NEMO ran {n_sub} substeps, the card "
                f"{cfg.barotropic.n_barotropic_substeps}")
        plain, st = model(), s0
        for _ in range(1, kt):
            st = jax.device_get(plain.step(st, dt=card.dt_s))
        g = read_step(root, kt)
        if arm == "substeps":
            out["label"] = ("GIVEN-NEMO-ENTRY (solver substep loop; past step "
                            "1 the six histories are the record's own)")
            base = st
            if kt > 1:      # NEMO's histories as the loop was handed them
                require(not sub["i000_entry_sc"][4], "NEMO restarted its histories here")
                base = st._replace(bt_hist=tuple(jnp.asarray(a) for a in (
                    pad_u(sub["i000_ub_e"]), pad_u(sub["i000_ubb_e"]),
                    pad_v(sub["i000_vb_e"]), pad_v(sub["i000_vbb_e"]),
                    sub["i000_sshb_e"], sub["i000_sshbb_e"])))
                rst = nemo_histories(root, kt)
                if rst is not None:   # the record's frame IS the restart's
                    require(all(np.array_equal(a, b) for a, b in zip(rst, base.bt_hist)),
                            f"kt {kt}: loop-entry histories != restart histories")
                    out["histories_equal_restart"] = True
            entry = seed(g, base)
        else:
            out["label"] = "INDEPENDENT (solver substep loop, the card's own chain)"
            entry = st
        res = jax.device_get(model(_NEMOWSRK3TestHooks(
            expose_barotropic_substeps=True)).step(entry, dt=card.dt_s))
        trace = {k: np.asarray(v) for k, v in res.substeps.items()}
        require(next(iter(trace.values())).shape[0] == n_sub, "trace length")
        tmask = np.asarray(card.recipe.z_coord.is_active)[..., 0] > 0.5
        wet = {"t": tmask, "u": wet_face["u"], "v": wet_face["v"]}
        table = substep_table(sub, trace, n_sub, wet, reg, plant)
        # the trace must be passive: the ordinary step from the same entry
        # gives the same handoff the traced step returned, bit for bit
        ref_state = state_fields(jax.device_get(model(_NEMOWSRK3TestHooks(
            expose_momentum_stage=1, expose_tracer_stage=1)).step(entry, dt=card.dt_s)))
        traced = state_fields(res.state_after_barotropic)
        out["trace_passive"] = {f: bool(np.array_equal(ref_state[f], traced[f]))
                                for f in ("ssh", "uu_b", "vv_b")}
        # the same control against NEMO: the ordinary step's handoff and the
        # traced step's exit must sit at the same distance from NEMO's frames
        # (which are one array: the stage-1 handoff IS the loop's exit)
        out["trace_passive_detail"] = {f: {
            "nemo_handoff_equals_exit_frame": bool(np.array_equal(
                g["1_" + nm], sub["o000_" + nm])) if "1_" + nm in g else None,
            "ordinary_vs_traced_unequal": int(np.count_nonzero(ref_state[f] != traced[f])),
            "ordinary_vs_traced_max_abs": float(np.max(np.abs(ref_state[f] - traced[f]))),
            "ordinary_vs_nemo_unequal": int(np.count_nonzero(
                (ref_state[f] != sub["o000_" + nm]) & wet[sg])),
            "traced_vs_nemo_unequal": int(np.count_nonzero(
                (traced[f] != sub["o000_" + nm]) & wet[sg]))}
            for f, nm, sg in (("ssh", "ssh_aa", "t"), ("uu_b", "uu_b_aa", "u"),
                              ("vv_b", "vv_b_aa", "v"))}
        out["exit_rows"] = [substep_row(f"o000.{f}", sub["o000_" + nm], traced[f],
                                        wet[sg], reg)
                            for f, nm, sg in (("ssh", "ssh_aa", "t"),
                                              ("uu_b", "uu_b_aa", "u"),
                                              ("vv_b", "vv_b_aa", "v"))]
        out.update({"kt": kt, "n_substeps": n_sub,
                    "record_scalars": {k: sub[k].tolist() for k in (
                        "i000_entry_sc", "j003_ext_coef", "j003_bck_coef")},
                    **table})
        return out
    if arm != "independent":
        m0, st = model(), s0
        for kt in kts:
            before[kt] = st
            st = jax.device_get(m0.step(st, dt=card.dt_s))
        out["unrecorded_carry"] = (
            "bt_hist (six barotropic histories; NEMO's own at the steps "
            "listed in nemo_histories_at_kt), w and the after-ssh slot "
            "come from the card's own chain at each kt")

    if arm in ("independent", "given_entry"):
        out["label"] = ("GIVEN-NEMO-ENTRY (whole step)" if arm == "given_entry"
                        else "INDEPENDENT" if not fold_fill_v else
                        "INDEPENDENT + MEASUREMENT ARM (fold image written "
                        "into the top V row between steps; not the card)")
        m = model()
        state = s0
        pads = {}
        for kt in (kts if arm == "independent" else scored):
            g = records.get(kt) or read_step(root, kt)
            ref = {"ssh": g["f_ssh_bb"], "uu_b": g["f_uu_b_bb"],
                   "vv_b": g["f_vv_b_bb"], "u": g["f_uu_k1_bb"],
                   "v": g["f_vv_k1_bb"]}
            if kt < n_stage:
                nxt = records.get(kt + 1) or read_step(root, kt + 1)
                ref["T"], ref["S"] = nxt["e_tn_k1_bb"], nxt["e_sn_k1_bb"]
                # the reference is the entry of the next step, bit for bit
                for f, nm in (("ssh", "e_ssh_bb"), ("uu_b", "e_uu_b_bb"),
                              ("vv_b", "e_vv_b_bb"), ("u", "e_uu_k1_bb"),
                              ("v", "e_vv_k1_bb")):
                    require(np.array_equal(ref[f], nxt[nm]),
                            f"kt {kt}: after-step {f} != entry of kt {kt + 1}")
            top = g["f_vv_k1_bb"]      # NEMO's own top V row is the fold image
            require(np.array_equal(top[-1], -top[-2][fold_perm]),
                    f"kt {kt}: NEMO's top V row is not the fold image")
            state = jax.device_get(m.step(
                state if arm == "independent" else seed(g, before[kt], kt),
                dt=card.dt_s))
            if fold_fill_v:
                state = fill_top_v(state)
            if arm == "independent" and kt % 10 == 0:
                hist = history_rows(root, kt, state, reg)
                if hist:
                    out.setdefault("history_rows", []).extend(hist)
            u_raw, v_raw = np.asarray(state.u.data), np.asarray(state.v.data)
            pads[kt] = {
                "west_u_equals_east_u": bool(np.array_equal(u_raw[:, 0], u_raw[:, -1])),
                "west_u_all_zero": bool(not np.any(u_raw[:, 0])),
                "south_v_all_zero": bool(not np.any(v_raw[0]))}
            cand = state_fields(state)
            rows = [row(f"{arm}.kt{kt}.{f}", ref[f], cand[f], reg,
                        plant=(plant == "score" and kt == 1 and f == "ssh"))
                    for f in (*FIELDS, "T", "S") if f in ref]
            out["per_kt"].append({"kt": kt, "rows": rows,
                                  "first_unequal": first_unequal(rows)})
        out["redundant_face_records_after_step"] = pads
        out["first_kt_over_bar"] = {
            f: next((p["kt"] for p in out["per_kt"] for r in p["rows"]
                     if r["name"].endswith("." + f) and r["status"] != "AT-BAR"),
                    None) for f in FIELDS}
    elif arm == "rhs":
        out["label"] = "GIVEN-NEMO-ENTRY (stp_2D right-hand side)"
        seen = {}
        observed = {face: model(_NEMOWSRK3TestHooks(
            slow_forcing_rhs_observer=(
                lambda v, face=face: seen.__setitem__(face, np.asarray(v))),
            slow_forcing_rhs_observer_face=face)) for face in ("u", "v")}
        for kt in scored:
            g = records[kt]
            seen.clear()
            for face in ("u", "v"):
                jax.block_until_ready(
                    observed[face].step(seed(g, before[kt], kt), dt=card.dt_s))
                jax.effects_barrier()       # the observer is an async callback
            require(set(seen) == {"u", "v"}, "the right-hand-side observer never fired")
            # scored on NEMO's wet faces; land faces are reported apart
            rows = [row(f"rhs.kt{kt}.{f}", g[f"b_{f}{f}_rhs_k1"], cand,
                        {**{k: m & wet_face[f] for k, m in reg.items()},
                         "masked_faces": ~wet_face[f]})
                    for f, cand in (("u", seen["u"][:, 1:, 0]),
                                    ("v", seen["v"][1:, :, 0]))]
            out["per_kt"].append({"kt": kt, "rows": rows,
                                  "first_unequal": first_unequal(rows)})
    elif arm == "rhs_terms":
        out["label"] = ("GIVEN-NEMO-ENTRY (stp_2D right-hand side, term by term, "
                        "against the offline replay of NEMO's statements)")
        fe3 = np.asarray(raw.fe3mask)[..., 0]
        seen = {}
        observed = model(_NEMOWSRK3TestHooks(
            slow_forcing_rhs_term_observer=lambda d: seen.update(
                {k: np.asarray(v) for k, v in d.items()})))
        cut = {"u": lambda a: a[:, 1:, 0], "v": lambda a: a[1:, :, 0]}
        for kt in scored:
            g = records[kt]
            seen.clear()
            jax.block_until_ready(observed.step(seed(g, before[kt], kt), dt=card.dt_s))
            jax.effects_barrier()           # the observer is an async callback
            require(bool(seen), "the right-hand-side term observer never fired")
            vec = CASE == "ORCA2G1b-vec"
            rep = rhs_term_replay(root, kt, fe3, vector=vec)
            rows = []
            for f in "uv":
                wet = rep["wet_" + f]
                terms = {k[:-2]: cut[f](v) for k, v in seen.items()
                         if k.endswith("_" + f)}
                total = terms["total"]
                others = sum(v for k, v in terms.items()
                             if k not in ("KE_PGF", "vortcor", "total"))
                rows.append({
                    "face": f,
                    "card_term_max_abs": {k: float(np.max(np.abs(v[wet])))
                                          for k, v in terms.items()},
                    "other_terms_max_abs": float(np.max(np.abs(others[wet]))),
                    "control_replay_vs_nemo": substep_row(
                        f"rhs.{f}", g[f"b_{f}{f}_rhs_k1"], rep["rhs_" + f], wet, reg,
                        plant=plant == "score"),
                    "card_total_vs_nemo": substep_row(
                        f"rhs.{f}", g[f"b_{f}{f}_rhs_k1"], total, wet, reg),
                    # the card's KE_PGF term holds the KE gradient too (G1b)
                    "card_pressure_gradient_vs_replay": substep_row(
                        f"hpg.{f}", rep["hpg_" + f] - rep["keg_" + f] if vec
                        else rep["hpg_" + f], terms["KE_PGF"], wet, reg),
                    "card_vorticity_vs_replay": substep_row(
                        f"vor.{f}", rep["vor_" + f], terms["vortcor"], wet, reg),
                    "replay_sum_of_card_terms_vs_nemo": substep_row(
                        f"rhs.{f}", g[f"b_{f}{f}_rhs_k1"],
                        terms["KE_PGF"] + terms["vortcor"], wet, reg)})
            out["per_kt"].append({"kt": kt, "rows": rows})
    elif arm == "vor_fold_replay":
        out["label"] = ("GIVEN-NEMO-ENTRY (offline replay of vor_een on the "
                        "two top U rows; no model step enters the NEMO side)")
        fe3 = np.asarray(raw.fe3mask)[..., 0]
        seen = {}
        observed = model(_NEMOWSRK3TestHooks(
            slow_forcing_rhs_observer=lambda v: seen.__setitem__("u", np.asarray(v)),
            slow_forcing_rhs_observer_face="u"))
        for kt in scored:
            g = records[kt]
            seen.clear()
            jax.block_until_ready(observed.step(seed(g, before[kt], kt), dt=card.dt_s))
            jax.effects_barrier()
            require("u" in seen, "the right-hand-side observer never fired")
            card_u = seen["u"][:, 1:, 0]
            full = vor_een_u_replay(root, kt, fe3, zero_image_row=False)
            cut = vor_een_u_replay(root, kt, fe3, zero_image_row=True)
            rows = []
            for name, j in (("pivot_row", -1), ("row_below", -2)):
                w = full["wet"][name]

                def cmp(a, b):
                    d = (a - b)[w]
                    ref = float(np.max(np.abs(b[w]))) if w.any() else 0.0
                    return {"n_wet": int(w.sum()), "n_unequal": int(np.count_nonzero(d)),
                            "max_abs": float(np.max(np.abs(d))) if d.size else 0.0,
                            "ref_max_abs": ref,
                            "relative": (float(np.max(np.abs(d))) / ref) if ref else 0.0}
                rows.append({
                    "row": name,
                    "replay_as_compiled_vs_NEMO_record": cmp(full[name], g["b_uu_rhs_k1"][j]),
                    "replay_image_row_zeroed_vs_card": cmp(cut[name], card_u[j]),
                    "card_vs_NEMO_record": cmp(card_u[j], g["b_uu_rhs_k1"][j]),
                    "replay_image_row_zeroed_vs_NEMO_record": cmp(
                        cut[name], g["b_uu_rhs_k1"][j])})
            out["per_kt"].append({"kt": kt, "replay_rows": rows,
                                  "t_fold_shift": full["t_fold_shift"]})
    elif arm == "handoff":
        out["label"] = "GIVEN-NEMO-ENTRY (external handoff read at stage 1)"
        m = model(_NEMOWSRK3TestHooks(expose_momentum_stage=1,
                                      expose_tracer_stage=1))
        for kt in scored:
            g = records[kt]
            cand = state_fields(jax.device_get(
                m.step(seed(g, before[kt], kt), dt=card.dt_s)))
            rows = [row(f"handoff.kt{kt}.{f}", g[nm], cand[f], reg)
                    for f, nm in (("uu_b", "1_uu_b_aa"), ("vv_b", "1_vv_b_aa"),
                                  ("ssh", "1_ssh_aa"))]
            out["per_kt"].append({"kt": kt, "rows": rows,
                                  "first_unequal": first_unequal(rows)})
    else:   # stages
        out["label"] = ("GIVEN-NEMO-ENTRY (stage-local: NEMO's external "
                        "handoff and NEMO's stage entry)")
        for kt in scored:
            g = records[kt]
            ext = (jnp.asarray(g["b_ssh_aa"]), jnp.asarray(pad_u(g["b_uu_b_aa"])),
                   jnp.asarray(pad_v(g["b_vv_b_aa"])),
                   jnp.asarray(pad_u(g["b_un_adv"])),
                   jnp.asarray(pad_v(g["b_vn_adv"])))
            stages = []
            for stage in (1, 2, 3):
                ent = None
                if stage > 1:
                    p = f"{stage - 1}_"
                    ent = (stage, jnp.asarray(pad_u(g[p + "uu_k1_aa"])[..., None]),
                           jnp.asarray(pad_v(g[p + "vv_k1_aa"])[..., None]),
                           jnp.asarray(g[p + "tn_k1_aa"][..., None]),
                           jnp.asarray(g[p + "sn_k1_aa"][..., None]),
                           jnp.asarray(g[p + "ssh_aa"]))
                last = stage == 3
                h = _NEMOWSRK3TestHooks(
                    **({"stage1_momentum_rhs_override": (
                        jnp.asarray(pad_u(g["b_uu_rhs_k1"])[..., None]),
                        jnp.asarray(pad_v(g["b_vv_rhs_k1"])[..., None]))}
                       if stage1_rhs_from_nemo and stage == 1 else {}),
                    stage_barotropic_output_override=ext,
                    stage_entry_override=ent,
                    expose_momentum_stage=0 if last else stage,
                    expose_tracer_stage=0 if last else stage)
                cand = state_fields(jax.device_get(
                    model(h).step(seed(g, before[kt], kt), dt=card.dt_s)))
                p = f"{stage}_"
                rows = [row(f"stage{stage}.kt{kt}.{f}", g[p + nm], cand[f], reg)
                        for f, nm in (("ssh", "ssh_aa"), ("u", "uu_k1_aa"),
                                      ("v", "vv_k1_aa"), ("T", "tn_k1_aa"),
                                      ("S", "sn_k1_aa"))]
                stages.append({"stage": stage, "rows": rows,
                               "first_unequal": first_unequal(rows)})
                if stage == 1:
                    exposed = [state_fields(jax.device_get(model(_NEMOWSRK3TestHooks(
                        stage_barotropic_output_override=ext,
                        **{flag: True})).step(seed(g, before[kt], kt), dt=card.dt_s)))
                        for flag in ("expose_stage1_momentum_rhs",
                                     "expose_stage1_raw_momentum")] \
                        if not stage1_rhs_from_nemo else [None, None]
                    stages[-1]["momentum_replay"] = stage1_momentum_replay(
                        g, card, cand, reg, *exposed)
                else:
                    exposed = [None, None]
                    if last and not stage1_rhs_from_nemo:
                        exposed = [state_fields(jax.device_get(model(
                            _NEMOWSRK3TestHooks(
                                stage_barotropic_output_override=ext,
                                stage_entry_override=ent, **flag)).step(
                                    seed(g, before[kt], kt), dt=card.dt_s)))
                            for flag in (
                                {"expose_stage3_momentum_rhs": "post_ldf"},
                                {"expose_stage3_raw_momentum": True})]
                    terms = None if stage1_rhs_from_nemo else {
                        name: state_fields(jax.device_get(model(
                            _NEMOWSRK3TestHooks(
                                stage_barotropic_output_override=ext,
                                stage_entry_override=ent,
                                expose_momentum_operator=name,
                                expose_momentum_operator_stage=stage)).step(
                                    seed(g, before[kt], kt), dt=card.dt_s)))
                        for name in ("hpg", "vorticity")}
                    stages[-1]["stage_replay"] = stage_momentum_replay(
                        root, kt, card, stage, g, cand, reg, *exposed,
                        card_terms=terms)
            out["per_kt"].append({"kt": kt, "stages": stages})
    return out


def _print(out: dict) -> None:
    if out["arm"] == "geometry":
        for r in out["rows"]:
            print(f"{'EXACT  ' if r['bit_identical'] else 'DIFFERS'} "
                  f"{r['row']:24s} vs {r['nemo_variable']:12s} "
                  f"n_diff={r['n_differing']:6d} max={r['max_abs_difference']:.3e}")
        print(f"GEOMETRY {out['n_bit_identical']}/{out['n_rows']} bit-identical")
        return
    print(out["label"], "| een_fold_image_v_row =", out.get("een_fold_image_v_row"),
          "| T,S FROM THE CARD'S CHAIN" if out.get("ts_from_chain_measurement_arm") else "")
    if out["arm"] == "subnormal_environment":
        for k, v in out.items():
            print(f"{k}: {v}")
        return
    if out["arm"] == "stage_operand_replay":
        for e in out["per_kt"]:
            for r in e["rows"]:
                print(f"kt{e['kt']} stage{r['stage']} rhd " + " | ".join(
                    f"{n}: depth {r[n]['depth_n_unequal']} rhd "
                    f"{r[n]['rhd_n_unequal']} max={r[n]['rhd_max_abs']:.1e} "
                    f"{r[n]['rhd_cells']}"
                    for n in ("card_ladder", "reciprocal_of_stage_ssh")))
            for r in e["face_ratio_rows"]:
                print(f"kt{e['kt']} stage{r['stage']} 1+r3(Kmm) of the frame's "
                      "sea level vs recorded: " + " | ".join(
                          f"{f}: ratio {r[f]['ratio_n_unequal']} one_plus "
                          f"{r[f]['one_plus_n_unequal']} {r[f]['faces']}" for f in "uv"))
        return
    if out["arm"] == "rhs_terms":
        for e in out["per_kt"]:
            for r in e["rows"]:
                print(f"kt{e['kt']} {r['face']} card terms (max abs):",
                      {k: f"{v:.2e}" for k, v in r["card_term_max_abs"].items() if v})
                for k in ("control_replay_vs_nemo", "card_total_vs_nemo",
                          "card_pressure_gradient_vs_replay",
                          "card_vorticity_vs_replay",
                          "replay_sum_of_card_terms_vs_nemo"):
                    x = r[k]
                    print(f"   {k:36s} unequal {x['n_unequal']:5d} of {x['n_wet']} "
                          f"max {x['max_abs']:.2e} (ref {x['ref_max_abs']:.2e}) "
                          + " ".join(f"{n}={v['n_unequal']}"
                                     for n, v in x["regions"].items()))
        return
    if out["arm"] == "ratio_order_replay":
        print(out["label"])
        for k in out["per_kt"]:
            print(f"kt {k['kt']}: " + "; ".join(
                f"{r['ratio']} written {r['written_order_vs_nemo']} divided "
                f"{r['divided_vs_nemo']} (1+r {r['one_plus_ratio_divided_vs_nemo']})"
                f" of {r['n']}" for r in k["ratio_rows"]))
            for r in k["coefficient_rows"]:
                print(f"   option {str(r['written_order_option']):5s} {r['operand']} "
                      f"builder vs replay {r['builder_vs_replay']} {r['cells']} | "
                      f"replay vs NEMO {r['replay_vs_nemo_recorded']}")
        return
    if out["arm"] == "ratio_fold_replay":
        print("NEMO halo ssh == mirrored row below:",
              out["nemo_halo_ssh_equals_mirrored_row_below"],
              "| L3 == L2 bitwise:", out["L3_equals_L2_bitwise"])
        for r in out["ratio_rows"]:
            print(f"{r['variant']:27s} {r['ratio']} " + " | ".join(
                f"{b}: {r[b]['n_unequal']} max={r[b]['max_abs']:.1e} "
                f"nemo={r[b]['nemo_max_abs']:.1e}"
                for b in ("pivot_row", "row_below", "all_rows_below")))
        for r in out.get("coefficient_rows", []):
            print(f"exchange={r['ratio_exchange']!s:5s} "
                  f"fe3mask={r['fe3mask_pivot_row']!s:5s} {r['operand']} "
                  f"unequal={r['n_unequal']}/{r['n_wet']} max={r['max_abs']:.1e} "
                  f"nemo={r['ref_max_abs']:.1e} " + " ".join(
                      f"{n}={v['n_unequal']}/{v['max_abs']:.1e}"
                      for n, v in r["regions"].items()))
        return
    if out["arm"] == "flux_update_replay":
        for r in out["replay_rows"]:
            print(f"j{r['substep']:03d} {r['face']}a " + " | ".join(
                f"{n}: {r[n]['n_unequal']}/{r[n]['n_wet']} max={r[n]['max_abs']:.1e}"
                for n in REPLAY_ORDERS))
        print("TOTALS", out["totals"])
        return
    if "rows" in out:       # the substep table: first unequal row per operand
        for f, d in out["trace_passive_detail"].items():
            print("passivity", f, d)
        print("trace passive:", out["trace_passive"], "| scalars unequal:",
              len(out["scalars_unequal"]), "of", out["n_scalars"])
        seen = set()
        for r in (*out["rows"], *out["exit_rows"]):
            op = r.get("operand", r["name"])
            if r["bit_identical"] or op in seen:
                continue
            seen.add(op)
            print(f"{r['name']:22s} unequal={r['n_unequal']:5d}/{r['n_wet']} "
                  f"max={r['max_abs']:.2e} nemo={r['ref_max_abs']:.2e} "
                  f"at {r['max_abs_cell']} " + " ".join(
                      f"{n}={v['n_unequal']}/{v['max_abs']:.1e}"
                      for n, v in r["regions"].items()))
        print("FIRST UNEQUAL:", out["first_unequal"] and out["first_unequal"]["name"])
        return
    for k in out["per_kt"]:
        for name, r in k.get("replay", {}).items():
            print(f"kt{k['kt']} {name:26s} " + " | ".join(
                f"{n}: {v['n_unequal']} unequal max={v['max_abs']:.1e}"
                for n, v in r.items()))
    for k in out["per_kt"]:
        for r in k.get("replay_rows", []):
            print(f"kt{k['kt']} {r['row']:9s} " + " | ".join(
                f"{n}: {v['n_unequal']}/{v['n_wet']} rel={v['relative']:.1e}"
                for n, v in r.items() if n != "row"))
    for k in out["per_kt"]:
        for r in k.get("operand_rows", []):
            print(f"kt{k['kt']} {r['row']:17s} NEMO max={r['nemo_max_abs']:.3e} "
                  f"({r['n_nemo_nonzero']} nonzero) card max={r['card_max_abs']:.3e} "
                  f"relative={r['relative']}")
    for k in out["per_kt"]:
        if "n_over_bar" in k and k["n_over_bar"]:
            print({a: b for a, b in k.items() if a != "over_bar_by_region"},
                  k.get("over_bar_by_region"))
    for r in out.get("history_rows", []):
        print(f"{r['name']:24s} {r['status']:6s} max={r['max_abs']:.3e} "
              f"n_unequal={r['n_unequal']}")
    for k in out["per_kt"]:
        for grp in k.get("stages", []):
            for r in grp.get("momentum_replay", {}).get("rows", []):
                print(f"kt{k['kt']} stage1 replay {r['face']} {r['after_ratio']:26s} "
                      f"{r.get('order', ''):33s}"
                      + (f"ratio!=nemo {r['ratio_vs_nemo_unequal']:5d} "
                         f"({r['ratio_vs_nemo_max_abs']:.1e}) " if "vs_card" in r else "")
                      + f"vs NEMO {r['vs_nemo']['n_unequal']:5d} "
                      f"max={r['vs_nemo']['max_abs']:.1e}"
                      + (f" | vs card {r['vs_card']['n_unequal']:5d} "
                         f"max={r['vs_card']['max_abs']:.1e}" if "vs_card" in r else ""))
            for r in grp.get("momentum_replay", {}).get("card_operand_rows", []):
                print(f"kt{k['kt']} stage1 card operands {r['face']}: " + " | ".join(
                    f"{n}: {v['n_unequal']} ({v['max_abs']:.1e} of {v['ref_max_abs']:.1e})"
                    for n, v in r.items() if n != "face"))
    for k in out["per_kt"]:
        groups = k["stages"] if "stages" in k else [k] if "rows" in k else []
        for grp in groups:
            for r in grp["rows"]:
                reg = r["regions"]
                if r.get("at_cell"):
                    c = r["at_cell"]
                    print(f"{r['name']:24s} cell {c['cell']} error={c['error']:.3e} "
                          f"nemo={c['nemo']:.3e} ulps={c['ulps']:.1f} "
                          f"near={c['neighbourhood_max_abs']:.2e}")
                print(f"{r['name']:24s} {r['status']:6s} max={r['max_abs']:.3e} "
                      f"norm={r['normalized_max_abs']:.3e} rms={r['rms']:.3e} "
                      "n_unequal " + " ".join(
                          f"{n}={reg[n]['n_unequal']}/{reg[n]['max_abs']:.1e}"
                          for n in ("fold", "seam", "south", "interior_coast",
                                    "interior_open", "masked_faces")
                          if n in reg))


def main(argv=None) -> int:
    global CASE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--case", choices=CASES, default=CASE)
    ap.add_argument("--kt-max", type=int, default=10)
    ap.add_argument("--kt-min", type=int, default=1,
                    help="first scored step of a stage-frame arm")
    ap.add_argument("--cell", help="j,i (inner domain, 0-based): every row "
                    "also reports the error at this cell and its neighbours")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--plant", choices=PLANTS)
    ap.add_argument("--geometry-plant", help="the geometry row to perturb")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--before-arm", action="store_true",
                    help="the card with een_fold_image_v_row switched off "
                         "(the round-2 before arm; also the option's plant)")
    ap.add_argument("--vor-fold-image", action="store_true",
                    help="measurement arm: the 3-D vorticity trend reads the "
                         "fold image on the top V row")
    ap.add_argument("--fold-fill-v", action="store_true",
                    help="measurement arm (independent only): write the fold "
                         "image into the top V row between steps")
    ap.add_argument("--hook", action="append", default=[],
                    help="measurement arm, not the card: a private boolean "
                         "model test hook switched on (repeatable)")
    ap.add_argument("--ts-from-chain", action="store_true",
                    help="measurement arm, not GIVEN-NEMO-ENTRY proper: the "
                         "entry T and S stay the card's own chain's; every "
                         "other entry field is NEMO's")
    ap.add_argument("--without-stage1-ssh", action="store_true",
                    help="the card with rk3_stage1_ssh_nemo_literal switched "
                         "off (the round-4 before arm; also the option's plant)")
    ap.add_argument("--without-entry-depth", action="store_true",
                    help="the card with barotropic_entry_depth_nemo_literal "
                         "switched off (a round-5 before arm; also the "
                         "option's plant)")
    ap.add_argument("--without-substep-dt", action="store_true",
                    help="the card with barotropic_substep_dt_nemo_literal "
                         "switched off (a round-5 before arm; also the "
                         "option's plant)")
    ap.add_argument("--without-flux-literal", action="store_true",
                    help="the card with barotropic_flux_update_nemo_literal "
                         "switched off (the round-5 before arm; also the "
                         "option's plant)")
    ap.add_argument("--stage1-rhs-from-nemo", action="store_true",
                    help="measurement arm (stages only), not the card: stage "
                         "1's momentum right-hand side is NEMO's recorded one")
    ap.add_argument("--without-stage1-rhs", action="store_true",
                    help="the card with rk3_stage1_rhs_nemo_literal switched "
                         "off (a round-6 before arm; also the option's plant)")
    ap.add_argument("--without-vor-order", action="store_true",
                    help="BEFORE ARM: the vorticity trend in the operator's "
                         "own operation order (round 6's card)")
    ap.add_argument("--without-zdf-residual", action="store_true",
                    help="BEFORE ARM: the barotropic velocity re-added after "
                         "the implicit vertical solve (round 6's card)")
    ap.add_argument("--without-fold-exchange-v", action="store_true",
                    help="BEFORE ARM: the stored top V row left at the "
                         "compact mask's zero (round 6's card)")
    ap.add_argument("--without-r3t-product", action="store_true",
                    help="the card's coordinate with nemo_r3t_stored_product "
                         "off (round 8's before arm / plant)")
    ap.add_argument("--without-ratio-order", action="store_true",
                    help="the card with barotropic_een_ratio_written_order "
                         "off (round 8's before arm / plant)")
    ap.add_argument("--without-stage-r3f", action="store_true",
                    help="BEFORE ARM: the stage F ratio taken from the "
                         "stage's interpolated sea level")
    ap.add_argument("--without-stage-r3t", action="store_true",
                    help="the card with rk3_stage_r3t_nemo_literal off "
                         "(round 9's before arm / plant)")
    ap.add_argument("--without-eos-gdept", action="store_true",
                    help="the card with nemo_eos_gdept_rounded_product off "
                         "(round 9's before arm / plant)")
    ap.add_argument("--without-stage-r3uv", action="store_true",
                    help="the card with rk3_stage_r3uv_nemo_literal off "
                         "(round 9's before arm / plant)")
    ap.add_argument("--without-nemo-zeta", action="store_true",
                    help="G1b: een_total's relative vorticity from the generic "
                         "curl and the grid's F area (round 2's before arm / plant)")
    ap.add_argument("--without-fold-unit-vec", action="store_true",
                    help="G1b: the solver's literal fold unit off under vector "
                         "form (round 2's before arm / plant)")
    ap.add_argument("--without-written-order", action="store_true",
                    help="the card with rk3_stage_velocity_written_order "
                         "switched off (a round-6 before arm; also its plant)")
    ap.add_argument("--without-ratio-fold", action="store_true",
                    help="the card with barotropic_een_ratio_fold_exchange "
                         "switched off (the round-6 before arm; also the "
                         "option's plant)")
    ap.add_argument("--without-fold-v-depth", action="store_true",
                    help="the card with barotropic_fold_top_v_face_ssh switched "
                         "off (the round-3 before arm; also the option's plant)")
    a = ap.parse_args(argv)
    CASE = a.case
    CELL[:] = [int(x) for x in a.cell.split(",")] if a.cell else []
    try:
        out = run(a.root, arm=a.arm, kt_max=a.kt_max, kt_min=a.kt_min, plant=a.plant,
                  allow_dirty=a.allow_dirty, geometry_plant=a.geometry_plant,
                  fold_fill_v=a.fold_fill_v, vor_fold_image=a.vor_fold_image,
                  before_arm=a.before_arm,
                  hook_arm=tuple(a.hook),
                  without_fold_v_depth=a.without_fold_v_depth,
                  ts_from_chain=a.ts_from_chain,
                  without_stage1_ssh=a.without_stage1_ssh,
                  without_flux_literal=a.without_flux_literal,
                  without_substep_dt=a.without_substep_dt,
                  without_entry_depth=a.without_entry_depth,
                  without_ratio_fold=a.without_ratio_fold,
                  stage1_rhs_from_nemo=a.stage1_rhs_from_nemo,
                  without_stage1_rhs=a.without_stage1_rhs,
                  without_written_order=a.without_written_order,
                  without_vor_order=a.without_vor_order,
                  without_zdf_residual=a.without_zdf_residual,
                  without_fold_exchange_v=a.without_fold_exchange_v,
                  without_stage_r3f=a.without_stage_r3f, without_stage_r3t=a.without_stage_r3t,
                  without_eos_gdept=a.without_eos_gdept,
                  without_stage_r3uv=a.without_stage_r3uv,
                  without_ratio_order=a.without_ratio_order,
                  without_r3t_product=a.without_r3t_product,
                  without_nemo_zeta=a.without_nemo_zeta,
                  without_fold_unit_vec=a.without_fold_unit_vec)
    except Exception as e:      # noqa: BLE001 -- refusals print and exit 2
        print(f"REFUSE: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    _print(out)
    if a.arm == "geometry":
        return 0 if out["all_bit_identical"] else 1
    if a.arm == "ratio_order_replay":   # the two controls gate the arm
        bad = [k["kt"] for k in out["per_kt"] if any(
            r["written_order_vs_nemo"] for r in k["ratio_rows"]) or any(
            r["replay_vs_nemo_recorded"] for r in k["coefficient_rows"])]
        if bad:
            print(f"STATUS CONTROL-FAILED at kt {bad}", file=sys.stderr)
        return 1 if bad else 0
    if a.plant:     # a plant must move a row that is AT-BAR when unplanted
        # (kt = 1 only: all five rows are at the bar there in both labels;
        # T and S are excluded, NEMO's kt = 1 entry holds T, S on land)
        if a.arm == "substeps":     # the table's own plant: one interior cell
            fired = any(r["name"] == "j001.mid.eta" and r["n_unequal"] == 1
                        for r in out["rows"])   # bit-identical when unplanted
            print("STATUS PLANT-FIRED" if fired else "REFUSE: plant stayed green",
                  file=sys.stderr)
            return 1 if fired else 3
        if a.kt_max != 1 or a.arm not in ("independent", "given_entry"):
            print("REFUSE: plants run at --kt-max 1 on the two whole-step arms",
                  file=sys.stderr)
            return 2
        fired = any(r["status"] != "AT-BAR" for k in out["per_kt"]
                    for r in k["rows"] if r["name"].split(".")[-1] in FIELDS)
        print("STATUS PLANT-FIRED" if fired else "REFUSE: plant stayed green",
              file=sys.stderr)
        return 1 if fired else 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
