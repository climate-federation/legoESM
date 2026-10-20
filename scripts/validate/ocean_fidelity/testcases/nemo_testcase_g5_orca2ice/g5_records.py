#!/usr/bin/env python3
"""G5 record (round-31 acquisition): admission, forcing identity, ladder frames.

The G5 record build writes the ORCA2 lane's single-rank frames
(``oracle_step_entry``, ``oracle_stage``, ``oracle_bt_frames``) and its
post-SI3 ``oracle_ocean_surface_input`` frame, not G1-G4's ``oracle_g1step``
stream.  This module reads them with the existing readers, admits the record
(passivity against the writer-free reference, twin bytes, internal
consistency), scores the card's surface inputs against NEMO's record, and
serves the G2 ladder the frame keys it already scores.

``admit``    passivity: ``check_records.restarts_identical`` of the reference
             against twin A at the last common step (G4's admission), twin
             bytes of every per-step frame, and the frame/restart consistency
             the ladder relies on.
``identity`` per step, per ocean-read field: the card's input against NEMO's
             record (unequal cells, max abs), the U/V stress statement, and
             the ice-ocean top-drag coefficient the card has no input for.

Refusals exit 73; a fired plant exits 1.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TESTCASES = HERE.parent
REPO = HERE.parents[4]
for p in (TESTCASES, TESTCASES / "nemo_testcase_g1_orca2grid",
          TESTCASES / "nemo_testcase_g4_orca2sbc", HERE.parents[1] / "orca2_l4", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import check_records as cr  # noqa: E402
from nemo_testcase_l1_vortex_kt2_walk import read_bt_frame, read_stage  # noqa: E402
from nemo_testcase_phase3_trajectory_gate import read_entry  # noqa: E402
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_phase2b_exchange_gate as schema,
)

STEPS = 10
TWINS = ("twin_a", "twin_b", "twin_c")
PER_STEP = ("oracle_step_entry_kt{kt:08d}.bin", "oracle_ocean_surface_input_kt{kt:08d}.bin",
            "oracle_bt_frames_kt{kt:08d}.bin", "oracle_stage_kt{kt:08d}_s1.bin",
            "oracle_stage_kt{kt:08d}_s2.bin", "oracle_stage_kt{kt:08d}_s3.bin")
# record fields the G5 ocean reads (D123 route, round-1 receipt table)
ROUTE_FIELDS = ("utau", "vtau", "qns", "qsr", "emp", "sfx", "rnf", "taum", "fr_i")


class Refuse(RuntimeError):
    pass


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise Refuse(msg)


def restart(roots, kt: int):
    for root in roots:
        path = Path(root) / f"ORCA2G1_{kt:08d}_restart.nc"
        if path.is_file():
            return path
    return None


def surface(root: Path, kt: int, *, keep_halo: bool = False) -> dict[str, np.ndarray]:
    """One post-sbc frame on the inner domain (y, x), shape read from its header;
    ``keep_halo``: "full" fields keep NEMO's halos (the U/V stress mirror)."""
    path = Path(root) / f"oracle_ocean_surface_input_kt{kt:08d}.bin"
    with path.open("rb") as fh:
        magic = fh.read(16).decode("ascii").rstrip()
        header = struct.unpack(schema.HEADER_FMT, fh.read(struct.calcsize(schema.HEADER_FMT)))
        data = np.fromfile(fh, dtype=np.float64)
    require(magic == schema.MAGIC, f"{path.name}: bad magic {magic!r}")
    require(header[:3] == (1, kt, 1 if kt % 2 else 3), f"{path.name}: bad header {header}")
    nx, ny, ntr, halo = header[3], header[4], header[5], header[7]
    require(data.size == schema.payload_count(header), f"{path.name}: payload size")
    require(bool(np.isfinite(data).all()), f"{path.name}: non-finite payload")
    sizes = {"full": (nx, ny), "reduced": (nx - 2 * halo, ny - 2 * halo),
             "halo1": (nx - 2, ny - 2), "reduced3d": (nx - 2 * halo, ny - 2 * halo, ntr)}
    cut = {"full": halo, "reduced": 0, "halo1": halo - 1, "reduced3d": 0}
    out, at = {}, 0
    for name, kind in schema.FIELDS:
        shape = sizes[kind]
        n = int(np.prod(shape))
        a = data[at:at + n].reshape(shape, order="F")
        at += n
        c = 0 if keep_halo and kind in ("full", "halo1") else cut[kind]
        if c:
            a = a[c:-c, c:-c]
        out[name] = np.ascontiguousarray(np.swapaxes(a, 0, 1))
    require(at == data.size, f"{path.name}: schema walk {at}/{data.size}")
    return out


def read_v1_hm_i(root: Path, kt: int) -> np.ndarray:
    """V1 record: SI3's hm_i after sbc at kt (run.sh ``v1_dump_hm_i``), inner
    domain (y, x).  Header magic + version, kt, nx, ny, bits; the round-1
    legacy frames declare the haloed 184 x 152 over a 180 x 148 payload."""
    path = Path(root) / f"oracle_v1_hm_i_kt{kt:08d}.bin"
    with path.open("rb") as fh:
        magic = fh.read(16).decode("ascii").rstrip()
        version, k, nx, ny, bits = struct.unpack("=5i", fh.read(20))
        data = np.fromfile(fh, dtype=np.float64)
    require(magic == "NEMO_V1_HMI_1" and (version, k, bits) == (1, kt, 64)
            and data.size in (nx * ny, (nx - 4) * (ny - 4)),
            f"{path.name}: {magic!r} {version} {k} {nx}x{ny} {bits} {data.size}")
    if data.size != nx * ny:
        nx, ny = nx - 4, ny - 4
    require(bool(np.isfinite(data).all()) and bool((data >= 0).all()),
            f"{path.name}: non-finite or negative hm_i")
    return np.ascontiguousarray(data.reshape((nx, ny), order="F").T)


def _lev(a: np.ndarray, nk: int) -> np.ndarray:
    a = np.asarray(a, np.float64)
    require(not np.any(a[..., nk:]), "NEMO holds a value below the card's last level")
    return np.ascontiguousarray(a[..., :nk])


def read_step(root: Path, kt: int, roots, nk: int = 30) -> dict[str, np.ndarray]:
    """The G2 ladder's frame keys from G5's frames: ``e_*`` entering kt,
    ``f_*`` after kt (stage 3; barotropic velocity from the bt frame),
    ``{s}_*_aa`` after stage s.  Level-1 keys plus ``*_all`` all-level T/S/u/v."""
    root = Path(root)
    e = read_entry(root / f"oracle_step_entry_kt{kt:08d}.bin", "ORCA2G5-ice")
    bt = read_bt_frame(root / f"oracle_bt_frames_kt{kt:08d}.bin", expect_step=kt)
    out = {"e_ssh_bb": np.asarray(e["ssh"], np.float64)}
    for f, n in (("T", "tn"), ("S", "sn"), ("u", "uu"), ("v", "vv")):
        out[f"e_{n}_k1_bb"] = _lev(e[f], nk)[..., 0]
    for s in (1, 2, 3):
        st = read_stage(root / f"oracle_stage_kt{kt:08d}_s{s}.bin", expect_step=kt, expect_stage=s)
        out[f"{s}_ssh_aa"] = np.asarray(st["ssh"], np.float64)
        out[f"{s}_uu_b_aa"] = np.asarray(bt["uu_b"], np.float64)
        out[f"{s}_vv_b_aa"] = np.asarray(bt["vv_b"], np.float64)
        for f, n in (("T", "tn"), ("S", "sn"), ("u", "uu"), ("v", "vv")):
            lev = _lev(st[f], nk)
            out[f"{s}_{n}_k1_aa"] = lev[..., 0]
            out[f"{s}_{n}_all_aa"] = lev
    out["f_ssh_bb"] = out["3_ssh_aa"]
    out["f_uu_k1_bb"], out["f_vv_k1_bb"] = out["3_uu_k1_aa"], out["3_vv_k1_aa"]
    out["f_uu_b_bb"], out["f_vv_b_bb"] = (np.asarray(bt["uu_b"], np.float64),
                                         np.asarray(bt["vv_b"], np.float64))
    return out


def nemo_supplied_forcing(card, root: Path, kt: int, *, zero: bool = False):
    """NEMO's post-SI3 fields at kt through the ORCA2 lane's route, exactly as
    G4 (``g4_forcing_steps.nemo_supplied_forcing``), plus NEMO's own qns."""
    from nemo_testcase_l4_orca2_round1_ladder_gate import _surface_forcings
    rec = surface(root, kt)
    fields = {n: rec[n] for n in ROUTE_FIELDS}
    fields["rnf_tsc"] = rec["rnf_tsc"]
    if zero:
        fields = {k: np.zeros_like(v) for k, v in fields.items()}
    freshwater, forcing = _surface_forcings(card, root, fields, kt)
    forcing = forcing._replace(
        q_nonsolar=fields["qns"],
        ice_ocean_top_drag_rCdu=(
            np.zeros_like(rec["rCdU_ice"]) if zero else rec["rCdU_ice"]),
        salt_flux_sfx=fields["sfx"],
    )
    if card.case == "ORCA2L4-bbc":
        path = Path(root) / "geothermal_heating.nc"
        heatflow = read_l4_geothermal_heatflow(
            path, np.asarray(card.recipe.initial_state.eta.data).shape)
        forcing = forcing._replace(
            geothermal_heat_flux_mwm2=(np.zeros_like(heatflow) if zero else heatflow))
    return freshwater, forcing


def read_l4_geothermal_heatflow(path: Path, expected_shape) -> np.ndarray:
    """Read native float32 heatflow and widen each value exactly to float64."""
    import netCDF4

    path = Path(path)
    require(path.is_file(), f"missing L4 geothermal forcing: {path}")
    with netCDF4.Dataset(path) as dataset:
        variable = dataset.variables["heatflow"]
        variable.set_auto_maskandscale(False)
        native = np.asarray(variable[0])
    require(native.dtype.kind == "f" and native.dtype.itemsize == 4,
            f"L4 heatflow dtype {native.dtype} is not native float32")
    require(native.shape == tuple(expected_shape),
            f"L4 heatflow shape {native.shape} != card grid {tuple(expected_shape)}")
    # astype performs IEEE binary32 -> binary64 widening.  Never format these
    # values as decimal: doing so changes 170 of the file's 183 distinct values.
    return native.astype(np.float64, copy=True)


def _cmp(card, nemo, sel) -> dict:
    card, nemo = np.asarray(card, np.float64), np.asarray(nemo, np.float64)
    d = (card.view("<u8") != nemo.view("<u8")) & sel
    return {"unequal": int(d.sum()),
            "max_abs": float(np.max(np.abs(card - nemo)[sel])) if sel.any() else 0.0}


def admit(ev: Path, plant: str | None = None, steps: int = STEPS) -> dict:
    ev = Path(ev)
    ref, a = ev / "ref_100step", ev / "twin_a"
    roots = [ev / t for t in TWINS]
    shared = [s for s in range(2, steps + 1)
              if (ref / f"ORCA2G1_{s:08d}_restart.nc").is_file()
              and (a / f"ORCA2G1_{s:08d}_restart.nc").is_file()]
    require(bool(shared), f"no restart step 2..{steps} common to the reference and twin A")
    common = max(shared)
    cand = a
    if plant == "passivity":         # one value moved in a copy of twin A's restart
        import shutil
        import tempfile
        import netCDF4
        cand = Path(tempfile.mkdtemp(prefix="g5_passivity_plant_"))
        name = f"ORCA2G1_{common:08d}_restart.nc"
        shutil.copyfile(a / name, cand / name)
        with netCDF4.Dataset(cand / name, "a") as h:
            v = h.variables["sshn"]
            x = np.array(v[:])
            x.flat[x.size // 2] = np.nextafter(x.flat[x.size // 2], np.inf)
            v[:] = x
    try:
        passive = cr.restarts_identical(ref, cand, [common])
    except cr.Refuse as exc:
        raise Refuse(f"PASSIVITY FAILED: {exc}") from exc
    extra = {}
    for t in ("twin_b", "twin_c"):
        common_t = [s for s in range(2, steps + 1)
                    if (ref / f"ORCA2G1_{s:08d}_restart.nc").is_file()
                    and (ev / t / f"ORCA2G1_{s:08d}_restart.nc").is_file()]
        # Twin C writes odd steps, the reference even ones: say so, never
        # report an empty comparison as identity.
        extra[t] = (cr.restarts_identical(ref, ev / t, common_t) if common_t
                    else "no restart step in common with the reference")
    twin = {}
    for kt in range(1, steps + 1):
        for pat in PER_STEP:
            name = pat.format(kt=kt)
            blobs = [(r / name).read_bytes() for r in roots]
            require(all(b == blobs[0] for b in blobs[1:]), f"twins differ: {name}")
            twin[name] = len(blobs[0])
    cons = {}
    for kt in range(1, steps + 1):
        g = read_step(a, kt, roots)
        if kt < steps:
            n = read_step(a, kt + 1, roots)
            for k in ("ssh", "tn_k1", "sn_k1", "uu_k1", "vv_k1"):
                fk, ek = (f"3_{k}_aa", f"e_{k}_bb")
                require(np.array_equal(g[fk], n[ek]), f"kt {kt}: stage 3 {k} != entry of kt {kt + 1}")
        path = restart(roots, kt)
        if path is not None:
            import netCDF4
            with netCDF4.Dataset(path) as h:
                var = lambda v: np.asarray(h.variables[v][0], np.float64)  # noqa: E731
                lev = lambda v: _lev(np.moveaxis(var(v), 0, -1), 30)  # noqa: E731
                ok = {"sshn": np.array_equal(var("sshn"), g["f_ssh_bb"]),
                      "uu_n": np.array_equal(var("uu_n"), g["f_uu_b_bb"]),
                      "vv_n": np.array_equal(var("vv_n"), g["f_vv_b_bb"]),
                      **{v: np.array_equal(lev(v), g[f"3_{k}_all_aa"])
                         for v, k in (("tn", "tn"), ("sn", "sn"), ("un", "uu"), ("vn", "vv"))}}
            require(all(ok.values()), f"kt {kt}: restart != stage-3/bt frame: {ok}")
            cons[kt] = {"restart": str(path), **{k: bool(v) for k, v in ok.items()}}
    return {"status": "PASS_G5_ADMISSION", "passivity": {"reference": str(ref),
            "candidate": "twin_a", "step": common, "restart_fields": passive},
            "reference_vs_twin_b_c": extra, "twin_identical_files": len(twin),
            "stage3_equals_next_entry_kt": list(range(1, steps)),
            "restart_consistency": cons}


def identity(ev: Path, deck_root: Path, plant: bool = False) -> dict:
    """Card surface inputs vs NEMO's record, every step."""
    import netCDF4
    sys.path.insert(0, str(TESTCASES))
    import nemo_testcase_g1_orca2grid_ladder as g1
    from g4_forcing_steps import t_to_uv
    from legoesm.core.source_rounding import nemo_source_round
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    import jax.numpy as jnp
    g1._fp64()
    root = Path(ev) / "twin_a"
    card = recipe.build_nemo_testcase_card("ORCA2G5-ice", deck_root=deck_root)
    with netCDF4.Dataset(root / "mesh_mask.nc") as h:
        tm, um, vm = (np.asarray(h.variables[n][0, 0], np.float64) for n in ("tmask", "umask", "vmask"))
    wet = tm > 0.5
    out = {}
    for kt in range(1, STEPS + 1):
        rec = surface(root, kt)
        fw, sf = nemo_supplied_forcing(card, root, kt)
        if plant and kt == 1:
            rec["emp"] = rec["emp"].copy()
            rec["emp"][np.argwhere(wet)[0][0], np.argwhere(wet)[0][1]] += 1e-12
        rows = {
            "utau (native i)": _cmp(sf.tau_i_native, rec["utau"], wet),
            "vtau (native j)": _cmp(sf.tau_j_native, rec["vtau"], wet),
            "taum": _cmp(sf.taum, rec["taum"], wet),
            "qsr": _cmp(sf.sw_down, rec["qsr"], wet),
            "qns": _cmp(sf.q_nonsolar, rec["qns"], wet),
            "q_net = round(qns+qsr)": _cmp(sf.q_net, np.asarray(nemo_source_round(
                jnp.asarray(rec["qns"]) + jnp.asarray(rec["qsr"]))), wet),
            "emp": _cmp(fw.evap, rec["emp"], wet),
            "rnf": _cmp(fw.runoff, rec["rnf"], wet),
            "sfx (card salt_flux vs 1e-3*sfx)": _cmp(sf.salt_flux, np.float64(1.0e-3) * rec["sfx"], wet),
            "fr_i": _cmp(sf.ice_concentration, rec["fr_i"], wet),
            "rnf_tsc T": _cmp(sf.runoff_tracer_content[0], rec["rnf_tsc"][..., 0], wet),
            "rnf_tsc S": _cmp(sf.runoff_tracer_content[1], rec["rnf_tsc"][..., 1], wet),
            "rCdU_ice": _cmp(sf.ice_ocean_top_drag_rCdu, rec["rCdU_ice"], wet),
            "ice_fw / precip / restoring (card 0)": {
                "card_nonzero": int(sum(np.count_nonzero(np.asarray(x)) for x in (
                    fw.ice_fw, fw.precip, fw.restoring)))},
        }
        # U/V stress (sbcmod.f90 T -> U/V): the instrument mirror on NEMO's own
        # full T stress (halos as NEMO filled them) against NEMO's recorded
        # utauU/vtauV.  The card's inner T stress is NEMO's (rows above); its
        # own T -> U/V statement is G4's certified one, scored by the ladders.
        fullrec = surface(root, kt, keep_halo=True)
        pad = {k: np.pad(m, 2) for k, m in (("t", tm), ("u", um), ("v", vm))}
        for name, src, along in (("utauU", "utau", True), ("vtauV", "vtau", False)):
            mirror = t_to_uv(fullrec[src], pad["t"], pad["u"], pad["v"], along)
            sel = (pad["u"] if along else pad["v"]) > 0     # wet faces: the ocean's support
            sel[:2] = False
            sel[-3:] = False
            sel[:, :2] = False
            sel[:, -3:] = False
            rows[f"{name} (mirror on NEMO T stress; instrument)"] = _cmp(mirror, fullrec[name], sel)
        cd = rec["rCdU_ice"]
        ice = wet & (rec["fr_i"] > 0)
        rows["rCdU_ice activity"] = {
            "ice_wet_cells": int(ice.sum()),
            "nonzero_on_ice_wet": int(np.count_nonzero(cd[ice])),
            "range_on_ice_wet": [float(cd[ice].min()), float(cd[ice].max())] if ice.any() else None,
            "finite": bool(np.isfinite(cd).all())}
        rows["fr_i>0 wet cells"] = int(ice.sum())
        out[kt] = rows
    bad = [(kt, n) for kt, rows in out.items() for n, r in rows.items()
           if isinstance(r, dict) and r.get("unequal", 0)]
    no_drag = [kt for kt, rows in out.items()
               if rows["rCdU_ice activity"]["nonzero_on_ice_wet"] == 0]
    # preregistered falsifiers: any field mismatch, or no ice-ocean drag on ice
    require(not bad, f"{'plant fired' if plant else 'forcing identity FAILED'}: {bad[:3]}")
    require(not no_drag, f"rCdU_ice is zero on every ice-covered wet cell at kt {no_drag}")
    return {"label": "card surface inputs vs NEMO's post-SI3 record (twin A), kt 1..10",
            "wet_cells": int(wet.sum()), "unequal_rows": bad, "steps": out}


def top_drag(ev: Path, plant: bool = False) -> dict:
    """kt = 1: NEMO's recorded external-mode drag coefficient against the
    card's jitted production T-to-U/V coefficient path, including the north
    fold and IEEE signed zeros."""
    root = Path(ev) / "twin_a"
    path = root / "oracle_bt_drag_operands_kt00000001.bin"
    with path.open("rb") as fh:
        magic = fh.read(16).decode("ascii").rstrip()
        _, kt, _, nx, ny, bits = struct.unpack("=6i", fh.read(24))
        cdu = np.fromfile(fh, dtype=np.float64, count=2 * nx * ny)
    require(magic == "NEMO_L2_BTDRG_1" and kt == 1 and bits == 64, f"{path.name}: header")
    cdu_u, cdu_v = (cdu[k * nx * ny:(k + 1) * nx * ny].reshape((nx, ny), order="F").T
                    for k in (0, 1))
    rec = surface(root, 1)
    if plant:
        rec["rCdU_ice"] = rec["rCdU_ice"].copy()
        j, i = np.argwhere(rec["rCdU_ice"] != 0)[0]
        rec["rCdU_ice"][j, i] += 1e-9
    from legoesm.ocean.dynamics.ocean_pe_latlon_cgrid import (
        nemo_ice_ocean_top_drag_coeff_faces,
    )
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    import jax
    import jax.numpy as jnp
    card = recipe.build_nemo_testcase_card("ORCA2G5-ice", deck_root=root)
    # The input is a tracer here, not a closed-over eager constant.  This is
    # the production compilation mode in which XLA can eliminate ``zeros + x``.
    rep_u, rep_v, _, _, _ = jax.jit(
        lambda x: nemo_ice_ocean_top_drag_coeff_faces(
            x, card.recipe.grid, dtype=np.float64)
    )(jnp.asarray(rec["rCdU_ice"], dtype=jnp.float64))
    # NEMO stores the east/north native face; lego stores redundant west/south
    # boundaries, hence [:,1:] / [1:,:].  The G5-selected production path
    # carries NEMO's signed (non-positive) pCdU coefficient directly.
    card_faces = {"zCdU_u": np.asarray(rep_u)[:, 1:],
                  "zCdU_v": np.asarray(rep_v)[1:, :]}
    nemo_faces = {"zCdU_u": cdu_u[2:-2, 2:-2],
                  "zCdU_v": cdu_v[2:-2, 2:-2]}
    faces = {"zCdU_u": np.asarray(card.recipe.initial_state.u_mask.data)[:, 1:] > 0.5,
             "zCdU_v": np.asarray(card.recipe.initial_state.v_mask.data)[1:, :] > 0.5}
    rows, ice = {}, rec["fr_i"] > 0
    for name in ("zCdU_u", "zCdU_v"):
        nemo, rep, face = nemo_faces[name], card_faces[name], faces[name]
        wet = nemo != 0
        rows[name] = {"nemo_nonzero_faces": int(wet.sum()),
                      "nemo_range": [float(nemo[wet].min()), float(nemo[wet].max())] if wet.any() else None,
                      "card_vs_nemo_on_nemo_nonzero": _cmp(rep, nemo, wet),
                      "card_vs_nemo_on_all_wet_faces": _cmp(rep, nemo, face),
                      "wet_faces_scored": int(face.sum()),
                      "fold_row_nonzero_faces_scored": int(wet[-1].sum()) if name == "zCdU_v" else 0}
        rows[name]["card_negative_zero_faces"] = int(
            np.count_nonzero((rep == 0.0) & np.signbit(rep)))
    rows["fr_i>0_cells"] = int(ice.sum())
    bad = [n for n in ("zCdU_u", "zCdU_v")
           if rows[n]["card_vs_nemo_on_all_wet_faces"]["unequal"]
           or rows[n]["nemo_nonzero_faces"] == 0]
    require(not bad, f"{'plant fired' if plant else 'top-drag replay FAILED'}: {bad}")
    return {"label": "kt 1 jitted external-mode drag coefficient, NEMO record vs replay vs card",
            "jit_traced_input": True, **rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("admit", "identity", "top_drag"))
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--deck-root", type=Path)
    ap.add_argument("--plant", action="store_true")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--steps", type=int, default=STEPS, help="admitted record length (admit)")
    args = ap.parse_args()
    try:
        if args.mode == "admit":
            res = admit(args.evidence, "passivity" if args.plant else None, args.steps)
        elif args.mode == "top_drag":
            res = top_drag(args.evidence, args.plant)
        else:
            res = identity(args.evidence, args.deck_root or args.evidence / "twin_a", args.plant)
    except (Refuse, cr.Refuse) as exc:
        print(f"REFUSE: {exc}" + ("  STATUS PLANT-FIRED" if args.plant else ""), file=sys.stderr)
        return 1 if args.plant else 73
    if args.plant:
        print("REFUSE: plant stayed green", file=sys.stderr)
        return 2
    if args.json:
        args.json.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(res.get("status", f"{args.mode.upper()}_DONE"), "unequal_rows=%s" % res.get("unequal_rows"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
