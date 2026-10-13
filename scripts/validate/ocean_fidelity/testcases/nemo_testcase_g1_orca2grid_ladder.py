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
  vor_fold_replay  OFFLINE numpy replay of NEMO's vor_een Coriolis statement
               (dynvor.f90:720, :727, :767, :772-775, :779-783) on the U faces
               of the pivot row and the row below, from NEMO's recorded entry
               fields with their halos: once as compiled, once with the
               fold-image V row's transport set to zero

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
        "vor_fold_replay", "restart_entry")
PLANTS = ("score", "entry")
STAGE_STEPS = 10          # stage frames exist for kt = 1..10
FOLD_ROWS, SEAM_COLS, SOUTH_ROWS = 3, 2, 3
FIELDS = ("ssh", "uu_b", "vv_b", "u", "v")


def _checker():
    spec = importlib.util.spec_from_file_location(
        "g1_check_records", HERE / "nemo_testcase_g1_orca2grid" / "check_records.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    return {"fold": fold, "seam": seam, "south": south, "interior": interior,
            "interior_coast": interior & coast, "interior_open": interior & ~coast}


def row(name, ref, cand, reg, *, plant=False) -> dict:
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
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


# ----------------------------------------------------------------- geometry
def card_geometry_rows(card, dom) -> list[tuple[str, np.ndarray, str]]:
    """(row, the card's array, the mesh_mask variable it must equal)."""
    g, z = card.recipe.grid, card.recipe.z_coord
    ops = z.nemo_een_barotropic
    f64 = lambda a: np.asarray(a, dtype=np.float64)       # noqa: E731
    lev = lambda a: f64(a)[..., 0]                         # noqa: E731
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
    card = recipe.build_nemo_testcase_card("ORCA2G1-flat", deck_root=root)
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


def vor_een_u_replay(root: Path, kt: int, fe3mask, *, zero_image_row: bool) -> dict:
    """NEMO's planetary vor_een trend on U, rows jj = Nje0-1 and Nje0.

    Literal operand order of dynvor.f90:720 (z1_e3f), :727 (zwz), :767 (zwy),
    :772-775 (triads), :779-783 (zua).  Everything is NEMO's: the entry frame
    WITH halos, the mesh file, and the T fold image read off NEMO's own halo
    row.  ``fe3mask`` is the card's (the mesh file does not carry it).
    """
    import netCDF4
    cr = _checker()
    path = cr.step_path(root, kt)
    head, f = cr.read_stream(path.read_bytes(), path.name)
    require(head["Njs0"] == 3 and head["Nis0"] == 3, "two halo points expected")
    nj, ni = head["jpj"], head["jpi"]
    jp = head["Nje0"] - 1                       # 0-based halo row of the pivot

    def halo(a):                                # cyclic east-west halo
        out = np.zeros((nj, ni))
        out[2:2 + a.shape[0], 2:-2] = a
        out[:, :2], out[:, -2:] = out[:, -4:-2], out[:, 2:4]
        return out
    with netCDF4.Dataset(root / "mesh_mask.nc") as h:
        lev = lambda n: np.asarray(h.variables[n][0], dtype=np.float64)  # noqa: E731
        m2 = {n: halo(lev(n)) for n in ("ff_f", "e1v", "e1u")}
        m3 = {n: halo(lev(n)[0]) for n in (
            "tmask", "vmask", "umask", "e3t_0", "e3v_0", "e3f_0")}
    # T fold image of the first halo row, read off NEMO's own sea level at
    # the last stage step (the first steps have no signal on the fold)
    last = cr.step_path(root, STAGE_STEPS)
    ssh = cr.read_stream(last.read_bytes(), last.name)[1]["e_ssh_bb"]
    require(bool(np.any(ssh[jp + 1, 2:-2])), "no sea-level signal on the fold halo row")
    cand = [s for s in range(-4, 5) if np.array_equal(
        ssh[jp + 1, 2:-2], ssh[jp - 1, (ni - np.arange(2, ni - 2) + s) % ni])]
    require(len(cand) == 1, f"T fold image not unique: shifts {cand}")
    perm = (ni - np.arange(ni) + cand[0]) % ni
    te3 = m3["e3t_0"] * m3["tmask"]
    te3[jp + 1] = te3[jp - 1, perm]             # halo row jj = Nje0 + 1
    e3f0 = np.zeros((nj, ni))
    e3f0[:-1, :-1] = ((te3[1:, :-1] + te3[1:, 1:]) + (te3[:-1, :-1] + te3[:-1, 1:])) * 0.25
    e3f0 = np.where(e3f0 == 0.0, m3["e3f_0"], e3f0)      # dynvor.f90:919
    fe3 = halo(np.asarray(fe3mask, dtype=np.float64))
    with np.errstate(divide="ignore", invalid="ignore"):
        zwz = m2["ff_f"] * (1.0 / (e3f0 * (1.0 + f["e_r3f"] * fe3)))
    zwy = m2["e1v"] * (m3["e3v_0"] * (1.0 + f["e_r3v_bb"] * m3["vmask"])) * f["e_vv_k1_bb"]
    if zero_image_row:
        zwy[jp] = 0.0
    out = {}
    i = np.arange(2, ni - 2)
    for name, j in (("pivot_row", jp), ("row_below", jp - 1)):
        ztne = zwz[j, i - 1] + zwz[j, i] + zwz[j - 1, i]
        ztnw1 = zwz[j - 1, i] + zwz[j, i] + zwz[j, i + 1]
        ztse = zwz[j, i] + zwz[j - 1, i] + zwz[j - 1, i - 1]
        ztsw1 = zwz[j - 1, i + 1] + zwz[j - 1, i] + zwz[j, i]
        with np.errstate(invalid="ignore"):
            out[name] = (1.0 / 12.0) * (1.0 / m2["e1u"][j, i]) * (
                (ztne * zwy[j, i] + ztnw1 * zwy[j, i + 1])
                + (ztse * zwy[j - 1, i] + ztsw1 * zwy[j - 1, i + 1]))
    out["wet"] = {"pivot_row": m3["umask"][jp, 2:-2] > 0,
                  "row_below": m3["umask"][jp - 1, 2:-2] > 0}
    out["t_fold_shift"] = cand[0]
    return out


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
def run(root: Path, *, arm: str, kt_max: int = 10, plant: str | None = None,
        allow_dirty: bool = False, geometry_plant: str | None = None,
        fold_fill_v: bool = False, vor_fold_image: bool = False,
        before_arm: bool = False) -> dict:
    if vor_fold_image:
        from legoesm.ocean.fidelity import nemo_testcase_recipe as _recipe
        calls: list = []
        with vor_fold_image_arm(_recipe.build_nemo_testcase_card(
                "ORCA2G1-flat", deck_root=root), calls):
            out = run(root, arm=arm, kt_max=kt_max, plant=plant,
                      allow_dirty=allow_dirty, geometry_plant=geometry_plant,
                      fold_fill_v=fold_fill_v, before_arm=before_arm)
        require(len(calls) > 0, "the vorticity fold-image arm was never traced")
        out["vor_fold_image_measurement_arm"] = {"traces": len(calls)}
        out["label"] = f"{out['label']} + MEASUREMENT ARM (fold-image V row "\
                       "in the 3-D vorticity trend; not the card)"
        return out
    require(arm in ARMS, f"unknown arm {arm!r}; expected one of {ARMS}")
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

    card = recipe.build_nemo_testcase_card("ORCA2G1-flat", deck_root=root)
    cfg = card.recipe.model_config
    if before_arm:      # BEFORE ARM / plant: the card without its fold-image row
        require(cfg.een_fold_image_v_row is True, "the card no longer selects the row")
        cfg = cfg._replace(een_fold_image_v_row=False)
    s0 = card.recipe.initial_state
    require(int(card.recipe.z_coord.n_levels) == 1, "the G1 card executes one level")
    cap = 100 if arm in ("independent", "restart_entry") else STAGE_STEPS
    require(1 <= kt_max <= cap, f"kt_max must lie in 1..{cap}")
    kts = range(1, kt_max + 1)
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
        return LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, cfg,
            **({} if hooks is None else {"_nemo_ws_test_hooks": hooks}))

    def seed(g, base):
        eta = np.array(g["e_ssh_bb"])
        if plant == "entry":
            j, i = np.unravel_index(int(np.argmax(np.abs(eta))), eta.shape)
            eta[j, i] += 1.0e-3
        return base._replace(
            T=s0.T.replace(data=jnp.asarray(g["e_tn_k1_bb"][..., None])),
            S=s0.S.replace(data=jnp.asarray(g["e_sn_k1_bb"][..., None])),
            u=s0.u.replace(data=jnp.asarray(pad_u(g["e_uu_k1_bb"])[..., None])),
            v=s0.v.replace(data=jnp.asarray(pad_v(g["e_vv_k1_bb"])[..., None])),
            eta=s0.eta.replace(data=jnp.asarray(eta)),
            uu_b=s0.uu_b.replace(data=jnp.asarray(pad_u(g["e_uu_b_bb"]))),
            vv_b=s0.vv_b.replace(data=jnp.asarray(pad_v(g["e_vv_b_bb"]))))

    out = {"arm": arm, "label": None, "legoesm_git_sha": sha, "plant": plant,
           "bar": BAR, "root": str(root), "kts": list(kts),
           "fold_fill_v_measurement_arm": fold_fill_v,
           "een_fold_image_v_row": bool(cfg.een_fold_image_v_row),
           "n_barotropic_substeps": int(cfg.barotropic.n_barotropic_substeps),
           "region_cells": {k: int(m.sum()) for k, m in reg.items()},
           "per_kt": []}
    records = {kt: read_step(root, kt) for kt in kts if kt <= STAGE_STEPS}
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
            if kt % 10 == 1 and kt > 1:
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
                        bt_hist=tuple(jnp.asarray(a) for a in (
                            pad_u(var("ub_e")), pad_u(var("ubb_e")),
                            pad_v(var("vb_e")), pad_v(var("vbb_e")),
                            var("sshb_e"), var("sshbb_e"))))
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
    if arm != "independent":
        m0, st = model(), s0
        for kt in kts:
            before[kt] = st
            st = jax.device_get(m0.step(st, dt=card.dt_s))
        out["unrecorded_carry"] = (
            "bt_hist (six barotropic histories), w and the after-ssh slot "
            "come from the card's own chain at each kt")

    if arm in ("independent", "given_entry"):
        out["label"] = ("GIVEN-NEMO-ENTRY (whole step)" if arm == "given_entry"
                        else "INDEPENDENT" if not fold_fill_v else
                        "INDEPENDENT + MEASUREMENT ARM (fold image written "
                        "into the top V row between steps; not the card)")
        m = model()
        state = s0
        pads = {}
        for kt in kts:
            g = records.get(kt) or read_step(root, kt)
            ref = {"ssh": g["f_ssh_bb"], "uu_b": g["f_uu_b_bb"],
                   "vv_b": g["f_vv_b_bb"], "u": g["f_uu_k1_bb"],
                   "v": g["f_vv_k1_bb"]}
            if kt < STAGE_STEPS:
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
                state if arm == "independent" else seed(g, before[kt]),
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
        for kt in kts:
            g = records[kt]
            seen.clear()
            for face in ("u", "v"):
                jax.block_until_ready(
                    observed[face].step(seed(g, before[kt]), dt=card.dt_s))
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
    elif arm == "vor_fold_replay":
        out["label"] = ("GIVEN-NEMO-ENTRY (offline replay of vor_een on the "
                        "two top U rows; no model step enters the NEMO side)")
        fe3 = np.asarray(raw.fe3mask)[..., 0]
        seen = {}
        observed = model(_NEMOWSRK3TestHooks(
            slow_forcing_rhs_observer=lambda v: seen.__setitem__("u", np.asarray(v)),
            slow_forcing_rhs_observer_face="u"))
        for kt in kts:
            g = records[kt]
            seen.clear()
            jax.block_until_ready(observed.step(seed(g, before[kt]), dt=card.dt_s))
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
        for kt in kts:
            g = records[kt]
            cand = state_fields(jax.device_get(
                m.step(seed(g, before[kt]), dt=card.dt_s)))
            rows = [row(f"handoff.kt{kt}.{f}", g[nm], cand[f], reg)
                    for f, nm in (("uu_b", "1_uu_b_aa"), ("vv_b", "1_vv_b_aa"),
                                  ("ssh", "1_ssh_aa"))]
            out["per_kt"].append({"kt": kt, "rows": rows,
                                  "first_unequal": first_unequal(rows)})
    else:   # stages
        out["label"] = ("GIVEN-NEMO-ENTRY (stage-local: NEMO's external "
                        "handoff and NEMO's stage entry)")
        for kt in kts:
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
                    stage_barotropic_output_override=ext,
                    stage_entry_override=ent,
                    expose_momentum_stage=0 if last else stage,
                    expose_tracer_stage=0 if last else stage)
                cand = state_fields(jax.device_get(
                    model(h).step(seed(g, before[kt]), dt=card.dt_s)))
                p = f"{stage}_"
                rows = [row(f"stage{stage}.kt{kt}.{f}", g[p + nm], cand[f], reg)
                        for f, nm in (("ssh", "ssh_aa"), ("u", "uu_k1_aa"),
                                      ("v", "vv_k1_aa"), ("T", "tn_k1_aa"),
                                      ("S", "sn_k1_aa"))]
                stages.append({"stage": stage, "rows": rows,
                               "first_unequal": first_unequal(rows)})
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
    print(out["label"], "| een_fold_image_v_row =", out.get("een_fold_image_v_row"))
    for k in out["per_kt"]:
        for r in k.get("replay_rows", []):
            print(f"kt{k['kt']} {r['row']:9s} " + " | ".join(
                f"{n}: {v['n_unequal']}/{v['n_wet']} rel={v['relative']:.1e}"
                for n, v in r.items() if n != "row"))
    for r in out.get("history_rows", []):
        print(f"{r['name']:24s} {r['status']:6s} max={r['max_abs']:.3e} "
              f"n_unequal={r['n_unequal']}")
    for k in out["per_kt"]:
        groups = k["stages"] if "stages" in k else [k] if "rows" in k else []
        for grp in groups:
            for r in grp["rows"]:
                reg = r["regions"]
                print(f"{r['name']:24s} {r['status']:6s} max={r['max_abs']:.3e} "
                      f"norm={r['normalized_max_abs']:.3e} rms={r['rms']:.3e} "
                      "n_unequal " + " ".join(
                          f"{n}={reg[n]['n_unequal']}/{reg[n]['max_abs']:.1e}"
                          for n in ("fold", "seam", "south", "interior_coast",
                                    "interior_open", "masked_faces")
                          if n in reg))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--kt-max", type=int, default=10)
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
    a = ap.parse_args(argv)
    try:
        out = run(a.root, arm=a.arm, kt_max=a.kt_max, plant=a.plant,
                  allow_dirty=a.allow_dirty, geometry_plant=a.geometry_plant,
                  fold_fill_v=a.fold_fill_v, vor_fold_image=a.vor_fold_image,
                  before_arm=a.before_arm)
    except Exception as e:      # noqa: BLE001 -- refusals print and exit 2
        print(f"REFUSE: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    _print(out)
    if a.arm == "geometry":
        return 0 if out["all_bit_identical"] else 1
    if a.plant:     # a plant must move a row that is AT-BAR when unplanted
        # (kt = 1 only: all five rows are at the bar there in both labels;
        # T and S are excluded, NEMO's kt = 1 entry holds T, S on land)
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
