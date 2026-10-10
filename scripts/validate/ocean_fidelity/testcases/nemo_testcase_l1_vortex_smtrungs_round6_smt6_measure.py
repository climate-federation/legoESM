#!/usr/bin/env python3
"""SMT-RUNGS round 6: SMT-6 / SMT-6b (BBL + geothermal) admission-side
identity, NEMO-side BBL gate counts, ladders, increments and the BBL footprint.

Sections (one JSON report, fp64 libm, CPU):

``inputs``     mesh_mask / dumped T,S / resto identity across SMT-5, SMT-6,
               SMT-6b; the SMT-6b dumped T equals the card's analytical initial
               T plus the anomaly bit for bit (the card builds only if its analytic T
               equals the dump, and the equality is re-read here, for S as
               well); the anomaly cells and values.
``gate``       NEMO's BBL criterion evaluated on NEMO's OWN T/S (initial dump,
               the ten kt entries, every daily restart): open U/V faces and the
               bottom cells they touch (shelf-side / deep-side).
``ladder``     kt=1..10, per field first unequal cell / n unequal / max abs /
               rms, both labels (INDEPENDENT and GIVEN-NEMO-ENTRY, never
               mixed), for SMT-6 (control SMT-5) and SMT-6b (control SMT-6);
               SMT-6b also with the BBL switched off in legoESM only.
``geothermal`` SMT-6: NEMO's SMT-6 entry minus NEMO's SMT-5 entry where the
               two are bit-equal (the increment of the one namelist change)
               against legoESM geothermal ON minus OFF, cell by cell; and the
               trabbc statement replayed in numpy on NEMO's mesh_mask e3t_0 and
               NEMO's stage-2 ssh against legoESM's exposed increment.

``--plant`` moves one wet T cell one ULP in the independent SMT-6 kt=1 row
(bit-exact unplanted); the run must exit nonzero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nemo_testcase_phase3_trajectory_gate import (  # noqa: E402
    expected_masks, lego_fields, read_entry, require,
)
from nemo_testcase_l1_vortex_kt2_walk import (  # noqa: E402
    _seed_from_record, read_stage,
)
from nemo_testcase_l1_vortex_smtrungs_round3_smt5_measure import (  # noqa: E402
    FIELDS, field_stats, sha256,
)

ROUNDS = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/smtrungs_rounds")
ROOTS = {
    "smt5": ROUNDS / "round2/oracle_vortex_smt5",
    "smt6": ROUNDS / "round5/oracle_vortex_smt6",
    "smt6b": ROUNDS / "round5/oracle_vortex_smt6b",
}
CASES = {"smt5": "VORTEX_SMT5_VEC-zps", "smt6": "VORTEX_SMT6_VEC-zps",
         "smt6b": "VORTEX_SMT6B_VEC-zps"}
INPUT_FILES = ("data_1m_potential_temperature_nomask.nc",
               "data_1m_salinity_nomask.nc", "resto.nc", "mesh_mask.nc")


def _cards():
    from legoesm.core.precision import PrecisionPolicy, set_policy
    from legoesm.ocean.fidelity.nemo_testcase_recipe import (
        build_nemo_testcase_card,
    )

    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    return {t: build_nemo_testcase_card(CASES[t], deck_root=ROOTS[t] / "kt1_10")
            for t in CASES}


def _model(card, cfg=None, hooks=None):
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel,
    )
    kw = {} if hooks is None else {"_nemo_ws_test_hooks": hooks}
    return LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord,
        card.recipe.model_config if cfg is None else cfg, **kw)


def bbl_gate(card):
    """Return (coefficients(T, S) -> (ahu>0, ahv>0), geom): the card's
    transcription of trabbl.f90:409-431/584-600 on the card's geometry."""
    from legoesm.ocean.physics.bbl_adv import (
        nemo_bbl_diffusive_coefficients, nemo_bbl_diffusive_geometry,
    )

    zc, cfg = card.recipe.z_coord, card.recipe.model_config
    raw = zc.nemo_een_barotropic
    geom = nemo_bbl_diffusive_geometry(
        zc.h_partial, card.recipe.initial_state.land_mask.data,
        zc.nemo_gdept_0, zc.nemo_bbl_e3u_0, zc.nemo_bbl_e3v_0,
        raw.e1u, raw.e2u, raw.e1v, raw.e2v, raw.umask, raw.vmask,
        aht_m2_s=cfg.bbl_aht_m2_s, grid=card.recipe.grid)

    def coeffs(T, S):
        ahu, ahv = nemo_bbl_diffusive_coefficients(
            T, S, geom, bottom_depth_m=geom.dep_bot_ref, rho_0=cfg.rho_0,
            grid=card.recipe.grid, eos_form=cfg.eos, seos_cfg=cfg.eos_nemo_seos)
        return np.asarray(ahu) > 0.0, np.asarray(ahv) > 0.0

    return coeffs, geom


def touched_columns(ahu, ahv):
    """Bottom columns whose cell takes a BBL trend: both sides of an open
    face (trabbl.f90:258-263 updates ji and ji+1 / jj and jj+1)."""
    up = np.zeros_like(ahu)
    up[:, 1:] = ahu[:, :-1]
    vp = np.zeros_like(ahv)
    vp[1:, :] = ahv[:-1, :]
    return ahu | up | ahv | vp


def inputs_section(cards) -> dict:
    import netCDF4

    out = {"file_sha256": {}}
    for t, root in ROOTS.items():
        out["file_sha256"][t] = {
            arm: {f: sha256(root / arm / f)[:16] for f in INPUT_FILES}
            for arm in ("smoke", "kt1_10", "day100")}
    for t in ("smt6", "smt6b"):
        out[f"{t}_mesh_equals_smt5"] = all(
            sha256(ROOTS[t] / arm / "mesh_mask.nc")
            == sha256(ROOTS["smt5"] / "kt1_10" / "mesh_mask.nc")
            for arm in ("smoke", "kt1_10", "day100"))
    with netCDF4.Dataset(ROOTS["smt6"] / "kt1_10/mesh_mask.nc") as a, \
            netCDF4.Dataset(ROOTS["smt5"] / "kt1_10/mesh_mask.nc") as b:
        out["mesh_variables_differing_smt6_vs_smt5"] = [
            n for n in sorted(a.variables)
            if not np.array_equal(np.asarray(a[n][:]), np.asarray(b[n][:]))]

    def target(t):
        with netCDF4.Dataset(ROOTS[t] / "kt1_10" / INPUT_FILES[0]) as h:
            raw = np.asarray(h.variables["votemper"][:], dtype=np.float64)
        nlev = int(cards[t].recipe.z_coord.n_levels)
        return raw, np.moveaxis(raw[0, :nlev], 0, -1)

    raw5, t5 = target("smt5")
    for t in ("smt6", "smt6b"):
        raw, rec1 = target(t)
        card = cards[t]
        with netCDF4.Dataset(ROOTS[t] / "kt1_10" / INPUT_FILES[1]) as h:
            rawS = np.asarray(h.variables["vosaline"][:], dtype=np.float64)
        s1 = np.moveaxis(rawS[0, :int(card.recipe.z_coord.n_levels)], 0, -1)
        out[f"{t}_dump_S_records_identical"] = bool(np.all(rawS == rawS[:1]))
        out[f"{t}_dump_vs_card_initial_S_wet"] = field_stats(
            np.asarray(card.recipe.initial_state.S.data), s1,
            np.asarray(card.recipe.z_coord.is_active))
        T0 = np.asarray(card.recipe.initial_state.T.data)
        wet = np.asarray(card.recipe.z_coord.is_active)
        out[f"{t}_dump_records_identical"] = bool(np.all(raw == raw[:1]))
        out[f"{t}_dump_vs_card_initial_T_wet"] = field_stats(T0, rec1, wet)
        out[f"{t}_dump_vs_smt5_dump_wet"] = field_stats(t5, rec1, wet)
    T6 = np.asarray(cards["smt6"].recipe.initial_state.T.data)
    T6b = np.asarray(cards["smt6b"].recipe.initial_state.T.data)
    moved = T6b != T6
    wetcol = np.asarray(cards["smt6"].recipe.z_coord.is_active).any(-1)
    bottom = np.asarray(cards["smt6"].recipe.z_coord.bottom_level)
    k = np.arange(T6.shape[-1])
    is_bottom = (k[None, None, :] == bottom[..., None]) & wetcol[..., None]
    out["anomaly"] = {
        "n_moved": int(moved.sum()),
        "n_moved_not_bottom": int((moved & ~is_bottom).sum()),
        "n_bottom_cells": int(is_bottom.sum()),
        "values_K": sorted(float(v) for v in np.unique(
            np.round((T6 - T6b)[moved], 12))),
        "n_per_value": {str(v): int((np.round(T6 - T6b, 12)[moved] == v).sum())
                        for v in np.unique(np.round((T6 - T6b)[moved], 12))},
    }
    for t, c in cards.items():
        s = np.asarray(c.recipe.initial_state.S.data)
        out[f"{t}_initial_S_is_35_wet"] = bool(np.all(
            s[np.asarray(c.recipe.z_coord.is_active)] == 35.0))
    ok = (out["smt6_mesh_equals_smt5"] and out["smt6b_mesh_equals_smt5"]
          and not out["mesh_variables_differing_smt6_vs_smt5"]
          and out["smt6_dump_vs_card_initial_T_wet"]["n_unequal"] == 0
          and out["smt6b_dump_vs_card_initial_T_wet"]["n_unequal"] == 0
          and out["smt6_dump_vs_card_initial_S_wet"]["n_unequal"] == 0
          and out["smt6b_dump_vs_card_initial_S_wet"]["n_unequal"] == 0
          and out["smt6_dump_vs_smt5_dump_wet"]["n_unequal"] == 0
          and out["anomaly"]["n_moved_not_bottom"] == 0)
    out["status"] = "IDENTICAL" if ok else "DIFFERS"
    return out


def gate_section(cards) -> dict:
    import netCDF4

    out = {}
    for t in ("smt6", "smt6b"):
        card = cards[t]
        coeffs, geom = bbl_gate(card)
        nlev = int(card.recipe.z_coord.n_levels)
        interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
        root = ROOTS[t]
        nu = int(np.count_nonzero(np.asarray(geom.mgrhu)
                                  * np.asarray(geom.u_active)))
        nv = int(np.count_nonzero(np.asarray(geom.mgrhv)
                                  * np.asarray(geom.v_active)))
        dep = np.asarray(card.recipe.z_coord.bottom_level)

        def count(T, S):
            ahu, ahv = coeffs(T, S)
            tc = touched_columns(ahu, ahv)
            return {"open_u": int(ahu.sum()), "open_v": int(ahv.sum()),
                    "touched_columns": int(tc.sum())}, tc

        rows = {"sloped_faces_u": nu, "sloped_faces_v": nv}
        T0 = np.asarray(card.recipe.initial_state.T.data)
        S0 = np.asarray(card.recipe.initial_state.S.data)
        rows["initial_state"], tc0 = count(T0, S0)
        # shelf-side = the shallower bottom of the two cells of an open face
        ahu, ahv = coeffs(T0, S0)
        shallow = np.zeros_like(ahu)
        deep = np.zeros_like(ahu)
        for ahx, step in ((ahu, (0, 1)), (ahv, (1, 0))):
            for j, i in np.argwhere(ahx):
                a_, b_ = (j, i), (j + step[0], i + step[1])
                s_, d_ = (a_, b_) if dep[a_] <= dep[b_] else (b_, a_)
                shallow[s_] = True
                deep[d_] = True
        rows["initial_state"]["shelf_side_cells"] = int(shallow.sum())
        rows["initial_state"]["deep_side_cells"] = int(deep.sum())
        rows["initial_state"]["cells_both_roles"] = int((shallow & deep).sum())
        kt = {}
        for n in range(1, 11):
            e = read_entry(root / "kt1_10" / f"oracle_step_entry_kt{n:08d}.bin",
                           CASES[t], expect_interior=interior)
            c, _ = count(e["T"][..., :nlev], e["S"][..., :nlev])
            kt[str(n)] = c
        rows["entries_kt1_10"] = kt
        daily = {}
        for day in range(1, 101):
            with netCDF4.Dataset(
                    root / "day100" /
                    f"VORTEX_SMT_VEC_OMIP_L1_ZPS_{day * 30:08d}_restart.nc") as h:
                T = np.asarray(h["tn"][0], dtype=np.float64).transpose(
                    1, 2, 0)[..., :nlev]
                S = np.asarray(h["sn"][0], dtype=np.float64).transpose(
                    1, 2, 0)[..., :nlev]
            c, _ = count(T, S)
            daily[str(day)] = c
        rows["daily_restarts"] = daily
        out[t] = rows
    return out


def _entries(t, card, nlev):
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    dummy = card.dummy_bottom_records
    ents = {}
    for kt in range(1, 11):
        e = read_entry(ROOTS[t] / "kt1_10" / f"oracle_step_entry_kt{kt:08d}.bin",
                       CASES[t], expect_interior=interior)
        require(e["step"] == kt and e["nz"] == nlev + dummy, f"entry {kt}")
        ents[kt] = e
    return ents


def _ref(e, f, nlev):
    r = np.asarray(e[f])
    return r if f == "ssh" else r[..., :nlev]


def ladder_section(cards, plant: bool) -> dict:
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    out = {}
    entries = {}
    for t in ("smt5", "smt6", "smt6b"):
        card = cards[t]
        nlev = int(card.recipe.z_coord.n_levels)
        masks = expected_masks(card)
        ents = entries[t] = _entries(t, card, nlev)
        dt = card.dt_s
        arms = {"card": _model(card)}
        if t == "smt6b":
            arms["lego_bbl_off"] = _model(
                card, hooks=_NEMOWSRK3TestHooks(disable_bbl=True))
        out[t] = {}
        for arm, model in arms.items():
            def advance(state, kt, model=model):
                return model.step(state, dt=dt, t_seconds=(kt - 1) * dt)

            ind, giv = {}, {}
            state = card.recipe.initial_state
            for kt in range(1, 11):
                cand = lego_fields(state)
                if plant and t == "smt6" and kt == 1 and arm == "card":
                    cand = dict(cand)
                    T = np.array(cand["T"])
                    where = tuple(np.argwhere(masks["T"])[0])
                    T[where] = np.nextafter(T[where], np.inf)
                    cand["T"] = T
                ind[str(kt)] = {f: field_stats(_ref(ents[kt], f, nlev),
                                               cand[f], masks[f])
                                for f in FIELDS}
                if kt < 10:
                    state = advance(state, kt)
            carried = card.recipe.initial_state
            for kt in range(1, 10):
                seed = _seed_from_record(carried, ents[kt], nlev)
                carried = advance(seed, kt)
                cand = lego_fields(carried)
                giv[str(kt)] = {f: field_stats(_ref(ents[kt + 1], f, nlev),
                                               cand[f], masks[f])
                                for f in FIELDS}
            out[t][arm] = {"independent": ind, "given_entry": giv}
    return out, entries


def footprint_section(cards, entries) -> dict:
    """SMT-6b kt=1 -> kt=2 residual on the cells the BBL touches vs elsewhere,
    BBL on / off in legoESM, and the SMT-6 control on the same cells."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    card = cards["smt6b"]
    nlev = int(card.recipe.z_coord.n_levels)
    masks = expected_masks(card)
    coeffs, _ = bbl_gate(card)
    ahu, ahv = coeffs(np.asarray(card.recipe.initial_state.T.data),
                      np.asarray(card.recipe.initial_state.S.data))
    tc = touched_columns(ahu, ahv)
    wetcol = np.asarray(card.recipe.z_coord.is_active).any(-1)
    bottom = np.asarray(card.recipe.z_coord.bottom_level)
    k = np.arange(nlev)
    is_bottom = (k[None, None, :] == bottom[..., None]) & wetcol[..., None]
    foot = is_bottom & tc[..., None]
    out = {"n_footprint_cells": int(foot.sum())}
    dt = card.dt_s
    for tag, c, ents, hooks in (
            ("smt6b_bbl_on", card, entries["smt6b"], None),
            ("smt6b_bbl_off", card, entries["smt6b"],
             _NEMOWSRK3TestHooks(disable_bbl=True)),
            ("smt6_control", cards["smt6"], entries["smt6"], None)):
        seed = c.recipe.initial_state
        after = lego_fields(_model(c, hooks=hooks).step(
            seed, dt=dt, t_seconds=0.0))
        ref = _ref(ents[2], "T", nlev)
        d = after["T"] - ref
        row = {}
        for name, m in (("footprint", foot),
                        ("same_bottom_row_off_footprint",
                         is_bottom & ~tc[..., None]),
                        ("interior_not_bottom", masks["T"] & ~is_bottom)):
            v = np.abs(d[m & masks["T"]])
            row[name] = {"n": int(v.size), "max_abs": float(v.max()),
                         "rms": float(np.sqrt(np.mean(v ** 2))),
                         "n_unequal": int(np.count_nonzero(v))}
        out[tag] = row
    on = lego_fields(_model(card).step(card.recipe.initial_state, dt=dt,
                                       t_seconds=0.0))["T"]
    off = lego_fields(_model(card, hooks=_NEMOWSRK3TestHooks(
        disable_bbl=True)).step(card.recipe.initial_state, dt=dt,
                                t_seconds=0.0))["T"]
    inc = on - off
    out["lego_bbl_increment"] = {
        "n_nonzero": int(np.count_nonzero(inc[masks["T"]])),
        "n_nonzero_footprint": int(np.count_nonzero(inc[foot])),
        "n_nonzero_off_footprint": int(np.count_nonzero(inc[~foot])),
        "max_abs": float(np.max(np.abs(inc[masks["T"]]))),
        "n_nonzero_S": int(np.count_nonzero(np.asarray(lego_fields(_model(
            card).step(card.recipe.initial_state, dt=dt, t_seconds=0.0))["S"]
            - lego_fields(_model(card, hooks=_NEMOWSRK3TestHooks(
                disable_bbl=True)).step(card.recipe.initial_state, dt=dt,
                                        t_seconds=0.0))["S"])[masks["S"]])),
    }
    out["footprint_mask_n_columns"] = int(tc.sum())
    return out


def geothermal_section(cards, entries) -> dict:
    card = cards["smt6"]
    nlev = int(card.recipe.z_coord.n_levels)
    masks = expected_masks(card)
    dt = card.dt_s
    e6, e5 = entries["smt6"], entries["smt5"]
    off_cfg = card.recipe.model_config._replace(nemo_geothermal_qgh_wm2=None)
    m_on, m_off = _model(card), _model(card, cfg=off_cfg)
    out = {}
    carried = card.recipe.initial_state
    for kt in range(1, 10):
        seed = _seed_from_record(carried, e6[kt], nlev)
        carried = m_on.step(seed, dt=dt, t_seconds=(kt - 1) * dt)
        on = lego_fields(carried)
        off = lego_fields(m_off.step(seed, dt=dt, t_seconds=(kt - 1) * dt))
        equal_entry = all(np.array_equal(e6[kt][f], e5[kt][f]) for f in FIELDS)
        row = {"nemo_entry_smt6_equals_smt5": equal_entry}
        d_lego = on["T"] - off["T"]
        row["lego_increment_T"] = {
            "n_nonzero": int(np.count_nonzero(d_lego[masks["T"]])),
            "max_abs": float(np.max(np.abs(d_lego[masks["T"]]))),
            "min_nonzero_abs": float(np.min(np.abs(
                d_lego[masks["T"]][d_lego[masks["T"]] != 0.0]))),
        }
        if equal_entry:
            d_nemo = (e6[kt + 1]["T"] - e5[kt + 1]["T"])[..., :nlev]
            row["nemo_increment_T"] = {
                "n_nonzero": int(np.count_nonzero(d_nemo[masks["T"]])),
                "max_abs": float(np.max(np.abs(d_nemo[masks["T"]]))),
            }
            row["increment_lego_vs_nemo_T"] = field_stats(
                d_nemo, d_lego, masks["T"])
            spacing = np.spacing(np.abs(e6[kt + 1]["T"][..., :nlev]))
            ratio = np.abs(d_nemo - d_lego) / spacing
            row["increment_diff_ulps_max"] = float(ratio[masks["T"]].max())
            row["n_cells_over_2ulp"] = int(((ratio > 2.0) & masks["T"]).sum())
            row["nemo_other_fields_equal"] = {
                f: bool(np.array_equal(e6[kt + 1][f], e5[kt + 1][f]))
                for f in ("S", "u", "v", "ssh")}
        out[str(kt)] = row
    return out


def replay_section(cards) -> dict:
    """trabbc.f90:158-159 replayed on NEMO's mesh_mask e3t_0 / ht_0 and NEMO's
    stage-2 ssh against legoESM's exposed stage-3 increment."""
    import netCDF4
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    card = cards["smt6"]
    nlev = int(card.recipe.z_coord.n_levels)
    root = ROOTS["smt6"] / "kt1_10"
    with netCDF4.Dataset(root / "mesh_mask.nc") as h:
        e3t0 = np.asarray(h["e3t_0"][0], dtype=np.float64).transpose(
            1, 2, 0)[..., :nlev]
        mbkt = np.asarray(h["mbathy"][0], dtype=np.int64)
    s2 = read_stage(root / "oracle_stage_kt00000001_s2.bin",
                    expect_step=1, expect_stage=2)
    out = {}
    cfg = card.recipe.model_config
    rho0 = cfg.rho_0
    rcp = cfg.physics.constants.c_sw
    r1_rho0_rcp = 1.0 / (rho0 * rcp)
    qgh_trd0 = r1_rho0_rcp * cfg.nemo_geothermal_qgh_wm2
    with netCDF4.Dataset(root / "mesh_mask.nc") as h:
        tm = np.asarray(h["tmask"][0], dtype=np.float64).transpose(
            1, 2, 0)[..., :nlev]
    ht0 = np.zeros(e3t0.shape[:2])
    for kk in range(nlev):
        ht0 = ht0 + e3t0[..., kk] * tm[..., kk]
    ssh2 = np.asarray(s2["ssh"], dtype=np.float64)
    r1_ht0 = np.where(ht0 != 0.0, 1.0 / np.where(ht0 != 0.0, ht0, 1.0), 0.0)
    r3t = ssh2 * r1_ht0
    wet = np.asarray(card.recipe.z_coord.is_active)
    k = np.arange(nlev)
    bottom = np.asarray(card.recipe.z_coord.bottom_level)
    is_bottom = (k[None, None, :] == bottom[..., None]) & wet.any(-1)[..., None]
    rep = np.zeros(e3t0.shape)
    denom = e3t0 * (1.0 + r3t[..., None] * tm)
    rep[is_bottom] = qgh_trd0 / denom[is_bottom]
    # legoESM's own ssh at the stage-3 level differs from NEMO's by the
    # inherited ssh residual, which moves (1+r3t) at 1e-14 relative; start
    # stage 3 from NEMO's stage-2 bundle so r3t is NEMO's.
    import jax.numpy as jnp
    from nemo_testcase_l1_vortex_kt2_walk import (
        _u_full, _v_full, read_bt_frame,
    )
    interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
    e1 = read_entry(root / "oracle_step_entry_kt00000001.bin", CASES["smt6"],
                    expect_interior=interior)
    e2 = read_entry(root / "oracle_step_entry_kt00000002.bin", CASES["smt6"],
                    expect_interior=interior)
    frame = read_bt_frame(root / "oracle_bt_frames_kt00000001.bin",
                          expect_step=1)
    external = (
        jnp.asarray(e2["ssh"]),
        jnp.asarray(_u_full(frame["uu_b"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vv_b"][..., None])[..., 0]),
        jnp.asarray(_u_full(frame["un_adv"][..., None])[..., 0]),
        jnp.asarray(_v_full(frame["vn_adv"][..., None])[..., 0]),
    )
    entry3 = (3, jnp.asarray(_u_full(s2["u"][..., :nlev])),
              jnp.asarray(_v_full(s2["v"][..., :nlev])),
              jnp.asarray(s2["T"][..., :nlev]),
              jnp.asarray(s2["S"][..., :nlev]), jnp.asarray(s2["ssh"]))
    seed = _seed_from_record(card.recipe.initial_state, e1, nlev)
    trace = _model(card, hooks=_NEMOWSRK3TestHooks(
        expose_stage3_tracer_damping=True,
        stage_barotropic_output_override=external,
        stage_entry_override=entry3)).step(seed, dt=card.dt_s, t_seconds=0.0)
    geo = np.asarray(trace[2][0])
    out["n_bottom_cells"] = int(is_bottom.sum())
    out["n_nonzero_replay"] = int(np.count_nonzero(rep))
    out["n_nonzero_lego"] = int(np.count_nonzero(geo))
    out["same_support"] = bool(np.array_equal(rep != 0.0, geo != 0.0))
    out["n_unequal"] = int(np.count_nonzero(rep != geo))
    out["max_abs_diff"] = float(np.max(np.abs(rep - geo)))
    nz = rep[is_bottom]
    out["per_step_increment_K_min_max"] = [
        float(card.dt_s * nz.min()), float(card.dt_s * nz.max())]
    out["max_abs_r3t_at_bottom"] = float(np.max(np.abs(
        np.broadcast_to(r3t[..., None], e3t0.shape)[is_bottom])))
    return out


def stage_section(cards) -> dict:
    """kt=1 tracer after each RK3 stage vs NEMO's stage records (T, S), and
    where the SMT-6b stage-3 T residual sits relative to the anomaly cells."""
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )

    out = {}
    for t in ("smt6", "smt6b"):
        card = cards[t]
        nlev = int(card.recipe.z_coord.n_levels)
        masks = expected_masks(card)
        root = ROOTS[t] / "kt1_10"
        out[t] = {}
        finals = {}
        for stage in (1, 2, 3):
            hooks = (_NEMOWSRK3TestHooks(expose_tracer_stage=stage)
                     if stage < 3 else None)
            st = _model(card, hooks=hooks).step(
                card.recipe.initial_state, dt=card.dt_s, t_seconds=0.0)
            cand = lego_fields(st)
            rec = read_stage(root / f"oracle_stage_kt00000001_s{stage}.bin",
                             expect_step=1, expect_stage=stage)
            row = {}
            for f in ("T", "S"):
                row[f] = field_stats(np.asarray(rec[f])[..., :nlev], cand[f],
                                     masks[f])
            out[t][f"stage{stage}"] = row
            finals[stage] = (cand["T"], np.asarray(rec["T"])[..., :nlev])
        if t == "smt6b":
            cand, ref = finals[3]
            d = np.abs(cand - ref)
            d[~masks["T"]] = 0.0
            j, i, k = np.unravel_index(np.argmax(d), d.shape)
            T0 = np.asarray(card.recipe.initial_state.T.data)
            T6 = np.asarray(cards["smt6"].recipe.initial_state.T.data)
            anomaly = T0 != T6
            bottom = np.asarray(card.recipe.z_coord.bottom_level)
            out[t]["argmax_stage3_T"] = {
                "cell_jik": [int(j), int(i), int(k)], "abs": float(d[j, i, k]),
                "is_anomaly_cell": bool(anomaly[j, i, k]),
                "is_bottom_level_of_column": bool(bottom[j, i] == k),
                "column_has_anomaly": bool(anomaly[j, i].any()),
                "bottom_level": int(bottom[j, i]),
                "anomaly_cells_with_abs_gt_1e-6": int(
                    (d[anomaly] > 1e-6).sum()),
                "n_anomaly_cells": int(anomaly.sum()),
                "n_nonanomaly_cells_gt_1e-6": int((d[~anomaly] > 1e-6).sum()),
                "max_abs_nonanomaly_cells": float(d[~anomaly].max()),
                "max_abs_by_level": [float(d[..., kk].max())
                                     for kk in range(nlev)],
            }
    return out


def bbl_replay_section(cards) -> dict:
    """trabbl.f90 replayed in numpy from NEMO's OWN mesh_mask, namelist S-EOS
    coefficients and kt=1 entry T/S (Kbb), against the card's transcription.

    409-431: bottom T/S at Kbb, alpha/beta (eosbn2.f90:1344,1347), zgdrho,
    ``SIGN(0.5, -zgdrho*mgrh)`` (copysign: gfortran honours -0.0);
    584-600: mgrh, e3u_bbl_0, ahu_bbl_0; 258-263: the bottom-cell trend.
    """
    import netCDF4
    from legoesm.ocean.physics.bbl_adv import (
        apply_bbl_diffusive_tendency, nemo_bbl_diffusive_coefficients,
        nemo_bbl_diffusive_geometry,
    )

    out = {}
    for t in ("smt6", "smt6b"):
        card = cards[t]
        cfg = card.recipe.model_config
        zc = card.recipe.z_coord
        root = ROOTS[t] / "kt1_10"
        nlev = int(zc.n_levels)
        with netCDF4.Dataset(root / "mesh_mask.nc") as h:
            def f2(n):
                return np.asarray(h[n][0], dtype=np.float64)

            def f3(n):
                return np.asarray(h[n][0], dtype=np.float64).transpose(
                    1, 2, 0)[..., :nlev]
            mbkt = np.maximum(np.asarray(h["mbathy"][0], dtype=np.int64), 1)
            e1t, e2t, e1u, e2u, e1v, e2v = (f2(n) for n in (
                "e1t", "e2t", "e1u", "e2u", "e1v", "e2v"))
            e3t0, e3u0, e3v0 = f3("e3t_0"), f3("e3u_0"), f3("e3v_0")
            tmask, umask, vmask = f3("tmask"), f3("umask"), f3("vmask")
            gdept = np.asarray(h["gdept_1d"][0], dtype=np.float64)
        ssu, ssv = umask.max(-1), vmask.max(-1)
        nj, ni = mbkt.shape
        jj, ii = np.indices((nj, ni))
        east = np.minimum(ii + 1, ni - 1)
        north = np.minimum(jj + 1, nj - 1)
        k0 = mbkt - 1
        dep = gdept[k0]
        # mgrhu/mgrhv (584-585); the closed-box edge faces carry ssumask = 0
        dU = dep[jj, east] - dep
        dV = dep[north, ii] - dep
        mgu = np.where(dU != 0.0, np.sign(dU), 0.0)
        mgv = np.where(dV != 0.0, np.sign(dV), 0.0)
        e3u_bbl = np.minimum(e3u0[jj, ii, k0[jj, east]], e3u0[jj, ii, k0])
        e3v_bbl = np.minimum(e3v0[jj, ii, k0[north, ii]], e3v0[jj, ii, k0])
        ahu0 = cfg_aht = 1000.0 * (e2u / e1u) * e3u_bbl * ssu
        ahv0 = 1000.0 * (e1v / e2v) * e3v_bbl * ssv
        # the deck's namelist: rn_ahtbbl, S-EOS a0 with all other terms zero
        rho0 = cfg.rho_0
        r1_rho0 = 1.0 / rho0
        alpha = 0.28 * (1.0 + 0.0 * 0.0 + 0.0 * 0.0) + 0.0 * 0.0
        alpha = alpha * r1_rho0
        beta = (0.0 * (1.0 - 0.0 * 0.0 - 0.0 * 0.0) - 0.0 * 0.0) * r1_rho0
        e = read_entry(root / "oracle_step_entry_kt00000001.bin", CASES[t],
                       expect_interior=(nj, ni))
        Tb = np.take_along_axis(e["T"][..., :nlev], k0[..., None], 2)[..., 0]
        Sb = np.take_along_axis(e["S"][..., :nlev], k0[..., None], 2)[..., 0]
        za = alpha + alpha
        zb = beta + beta
        gu = (za * (Tb[jj, east] - Tb) - zb * (Sb[jj, east] - Sb)) * ssu
        gv = (za * (Tb[north, ii] - Tb) - zb * (Sb[north, ii] - Sb)) * ssv
        # SIGN(0.5, x): closed for x >= 0 INCLUDING a signed zero (measured
        # below: NEMO closes the flat faces; copysign would open half of them)
        ahu = (0.5 - np.where(-gu * mgu >= 0.0, 0.5, -0.5)) * ahu0
        ahv = (0.5 - np.where(-gv * mgv >= 0.0, 0.5, -0.5)) * ahv0
        ahu_alt = (0.5 - np.copysign(0.5, -gu * mgu)) * ahu0
        ahv_alt = (0.5 - np.copysign(0.5, -gv * mgv)) * ahv0
        # stage-2 ssh -> r3t(Kmm); ht_0 = sum e3t_0*tmask, r3t = ssh*r1_ht_0
        s2 = read_stage(root / "oracle_stage_kt00000001_s2.bin",
                        expect_step=1, expect_stage=2)
        ht0 = np.zeros((nj, ni))
        for k in range(nlev):
            ht0 = ht0 + e3t0[..., k] * tmask[..., k]
        r1_ht0 = np.where(ht0 != 0.0, 1.0 / np.where(ht0 != 0.0, ht0, 1.0), 0.0)
        r3t = np.asarray(s2["ssh"], dtype=np.float64) * r1_ht0
        r1_e1e2t = 1.0 / (e1t * e2t)
        div = e3t0[jj, ii, k0] * (1.0 + r3t * tmask[jj, ii, k0])
        ahu_m = np.where(ii > 0, ahu[jj, np.maximum(ii - 1, 0)], 0.0)
        ahv_m = np.where(jj > 0, ahv[np.maximum(jj - 1, 0), ii], 0.0)
        Tw = Tb[jj, np.maximum(ii - 1, 0)]
        Ts = Tb[np.maximum(jj - 1, 0), ii]
        wet_b = tmask[jj, ii, k0] > 0.5
        trend_b = ((ahu * (Tb[jj, east] - Tb) - ahu_m * (Tb - Tw))
                   + (ahv * (Tb[north, ii] - Tb) - ahv_m * (Tb - Ts))
                   ) * r1_e1e2t / div
        trend = np.zeros(e3t0.shape)
        trend[jj, ii, k0] = np.where(wet_b, trend_b, 0.0)
        ahu_m_a = np.where(ii > 0, ahu_alt[jj, np.maximum(ii - 1, 0)], 0.0)
        ahv_m_a = np.where(jj > 0, ahv_alt[np.maximum(jj - 1, 0), ii], 0.0)
        trend_alt_b = ((ahu_alt * (Tb[jj, east] - Tb) - ahu_m_a * (Tb - Tw))
                       + (ahv_alt * (Tb[north, ii] - Tb) - ahv_m_a * (Tb - Ts))
                       ) * r1_e1e2t / div
        trend_alt = np.zeros(e3t0.shape)
        trend_alt[jj, ii, k0] = np.where(wet_b, trend_alt_b, 0.0)

        # the card's transcription on the same operands
        raw = zc.nemo_een_barotropic
        geom = nemo_bbl_diffusive_geometry(
            zc.h_partial, card.recipe.initial_state.land_mask.data,
            zc.nemo_gdept_0, zc.nemo_bbl_e3u_0, zc.nemo_bbl_e3v_0,
            raw.e1u, raw.e2u, raw.e1v, raw.e2v, raw.umask, raw.vmask,
            aht_m2_s=cfg.bbl_aht_m2_s, grid=card.recipe.grid)
        T3, S3 = e["T"][..., :nlev], e["S"][..., :nlev]
        lu, lv = nemo_bbl_diffusive_coefficients(
            T3, S3, geom, bottom_depth_m=geom.dep_bot_ref, rho_0=cfg.rho_0,
            grid=card.recipe.grid, eos_form=cfg.eos, seos_cfg=cfg.eos_nemo_seos)
        lu, lv = np.asarray(lu), np.asarray(lv)
        zero = np.zeros_like(T3)
        h_kmm = e3t0 * (1.0 + r3t[..., None] * tmask)
        area = e1t * e2t
        dT, dS = apply_bbl_diffusive_tendency(
            zero, zero, T3, S3, h_kmm, area, geom, lu, lv,
            grid=card.recipe.grid)
        dT, dS = np.asarray(dT), np.asarray(dS)
        face_u = (ssu > 0) & (mgu != 0)
        face_v = (ssv > 0) & (mgv != 0)
        out[t] = {
            "replay_open_u": int(((ahu > 0) & face_u).sum()),
            "replay_open_v": int(((ahv > 0) & face_v).sum()),
            "lego_open_u": int((lu > 0).sum()),
            "lego_open_v": int((lv > 0).sum()),
            "ahu_unequal_on_sloped_faces": int(((ahu != lu) & face_u).sum()),
            "ahv_unequal_on_sloped_faces": int(((ahv != lv) & face_v).sum()),
            "trend_T_nonzero_replay": int(np.count_nonzero(trend)),
            "trend_T_nonzero_lego": int(np.count_nonzero(dT)),
            "trend_T_unequal_cells": int(np.count_nonzero(trend != dT)),
            "trend_T_max_abs_diff": float(np.max(np.abs(trend - dT))),
            "trend_T_max_abs": float(np.max(np.abs(trend))),
            "trend_S_nonzero_lego": int(np.count_nonzero(dS)),
            "r1_rho0_alpha": [float(alpha), float(beta)],
            "flat_wet_u_faces_with_nonzero_dT": int(
                ((Tb[jj, east] - Tb != 0.0) & (mgu == 0) & (ssu > 0)).sum()),
            "signed_zero_open_reading": {
                "n_trend_cells": int(np.count_nonzero(trend_alt)),
                "max_step_increment_K": float(
                    card.dt_s * np.max(np.abs(trend_alt))),
            },
        }
        # the gate alone (trabbl.f90:409-431) on every NEMO entry kt=1..10 and
        # every daily restart: replay vs the card, sloped faces
        def gate_replay(Tk, Sk):
            tk = np.take_along_axis(Tk, k0[..., None], 2)[..., 0]
            sk = np.take_along_axis(Sk, k0[..., None], 2)[..., 0]
            g_u = (za * (tk[jj, east] - tk) - zb * (sk[jj, east] - sk)) * ssu
            g_v = (za * (tk[north, ii] - tk) - zb * (sk[north, ii] - sk)) * ssv
            r_u = (0.5 - np.where(-g_u * mgu >= 0.0, 0.5, -0.5)) * ahu0
            r_v = (0.5 - np.where(-g_v * mgv >= 0.0, 0.5, -0.5)) * ahv0
            c_u, c_v = nemo_bbl_diffusive_coefficients(
                Tk, Sk, geom, bottom_depth_m=geom.dep_bot_ref, rho_0=cfg.rho_0,
                grid=card.recipe.grid, eos_form=cfg.eos,
                seos_cfg=cfg.eos_nemo_seos)
            c_u, c_v = np.asarray(c_u), np.asarray(c_v)
            return (int(((r_u != c_u) & face_u).sum()
                        + ((r_v != c_v) & face_v).sum()),
                    int(((r_u > 0) & face_u).sum()
                        + ((r_v > 0) & face_v).sum()))

        gate_rows = {}
        for n in range(1, 11):
            en = read_entry(root / f"oracle_step_entry_kt{n:08d}.bin",
                            CASES[t], expect_interior=(nj, ni))
            gate_rows[f"kt{n}"] = gate_replay(en["T"][..., :nlev],
                                              en["S"][..., :nlev])
        for day in range(1, 101):
            with netCDF4.Dataset(
                    ROOTS[t] / "day100" /
                    f"VORTEX_SMT_VEC_OMIP_L1_ZPS_{day * 30:08d}_restart.nc") as h:
                Tk = np.asarray(h["tn"][0], dtype=np.float64).transpose(
                    1, 2, 0)[..., :nlev]
                Sk = np.asarray(h["sn"][0], dtype=np.float64).transpose(
                    1, 2, 0)[..., :nlev]
            gate_rows[f"day{day}"] = gate_replay(Tk, Sk)
        out[t]["gate_replay_unequal_faces_all_samples"] = int(
            sum(v[0] for v in gate_rows.values()))
        out[t]["gate_replay_open_faces_by_sample"] = {
            k: v[1] for k, v in gate_rows.items()
            if k.startswith("kt") or k in ("day1", "day10", "day100")}
        out[t]["gate_replay_n_samples"] = len(gate_rows)
        if t == "smt6":
            # NEMO's own SMT-6 minus SMT-5 entry at kt=2 (one namelist change:
            # geothermal + BBL) vs legoESM geothermal-only: the signed-zero
            # reading predicts extra increments at the cells counted above.
            c5 = cards["smt5"]
            e5 = read_entry(ROOTS["smt5"] / "kt1_10" /
                            "oracle_step_entry_kt00000002.bin", CASES["smt5"],
                            expect_interior=(nj, ni))
            e6 = read_entry(root / "oracle_step_entry_kt00000002.bin",
                            CASES[t], expect_interior=(nj, ni))
            d_nemo = (e6["T"] - e5["T"])[..., :nlev]
            off = card.recipe.model_config._replace(
                nemo_geothermal_qgh_wm2=None)
            dt = card.dt_s
            on_T = lego_fields(_model(card).step(
                card.recipe.initial_state, dt=dt, t_seconds=0.0))["T"]
            off_T = lego_fields(_model(card, cfg=off).step(
                card.recipe.initial_state, dt=dt, t_seconds=0.0))["T"]
            gap = np.abs(d_nemo - (on_T - off_T))
            alt = np.abs(trend_alt) > 0.0
            out[t]["nemo_minus_lego_geothermal_only"] = {
                "max_abs_all_cells": float(gap.max()),
                "max_abs_at_signed_zero_open_cells": float(gap[alt].max()),
                "n_signed_zero_open_cells": int(alt.sum()),
            }
    return out


def stage3_arms_section(cards) -> dict:
    """legoESM stage 3 started from NEMO's own stage-2 state and NEMO's
    barotropic handoff (the stage-twin arm of rounds 218/238), scored against
    NEMO's kt=2 entry T; then one term at a time switched off or changed in
    legoESM only (NEMO is never re-run).  The residual of the unchanged arm
    says whether the owner sits inside stage 3; the others size each term."""
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        _NEMOWSRK3TestHooks,
    )
    from nemo_testcase_l1_vortex_kt2_walk import (
        _u_full, _v_full, read_bt_frame,
    )

    out = {}
    for t in ("smt6", "smt6b"):
        card = cards[t]
        nlev = int(card.recipe.z_coord.n_levels)
        masks = expected_masks(card)
        root = ROOTS[t] / "kt1_10"
        interior = np.asarray(card.recipe.initial_state.T.data).shape[:2]
        e1 = read_entry(root / "oracle_step_entry_kt00000001.bin", CASES[t],
                        expect_interior=interior)
        e2 = read_entry(root / "oracle_step_entry_kt00000002.bin", CASES[t],
                        expect_interior=interior)
        s2 = read_stage(root / "oracle_stage_kt00000001_s2.bin",
                        expect_step=1, expect_stage=2)
        frame = read_bt_frame(root / "oracle_bt_frames_kt00000001.bin",
                              expect_step=1)
        seed = _seed_from_record(card.recipe.initial_state, e1, nlev)
        external = (
            jnp.asarray(e2["ssh"]),
            jnp.asarray(_u_full(frame["uu_b"][..., None])[..., 0]),
            jnp.asarray(_v_full(frame["vv_b"][..., None])[..., 0]),
            jnp.asarray(_u_full(frame["un_adv"][..., None])[..., 0]),
            jnp.asarray(_v_full(frame["vn_adv"][..., None])[..., 0]),
        )
        entry3 = (3, jnp.asarray(_u_full(s2["u"][..., :nlev])),
                  jnp.asarray(_v_full(s2["v"][..., :nlev])),
                  jnp.asarray(s2["T"][..., :nlev]),
                  jnp.asarray(s2["S"][..., :nlev]), jnp.asarray(s2["ssh"]))
        base = card.recipe.model_config
        arms = {
            "unchanged": base,
            "bbl_off": base._replace(bbl_diffusive_option=0,
                                     bbl_aht_m2_s=0.0),
            "redi_off": base._replace(
                gm_redi=base.gm_redi._replace(kappa_Redi=0.0)),
            "redi_smax_1.0": base._replace(
                gm_redi=base.gm_redi._replace(S_max=1.0)),
        }
        T0 = np.asarray(card.recipe.initial_state.T.data)
        T6 = np.asarray(cards["smt6"].recipe.initial_state.T.data)
        anomaly = T0 != T6
        ref = np.asarray(e2["T"])[..., :nlev]
        out[t] = {}
        res = {}
        for name, cfg in arms.items():
            st = _model(card, cfg=cfg, hooks=_NEMOWSRK3TestHooks(
                stage_barotropic_output_override=external,
                stage_entry_override=entry3)).step(
                    seed, dt=card.dt_s, t_seconds=0.0)
            T = lego_fields(st)["T"]
            res[name] = T
            row = field_stats(ref, T, masks["T"])
            d = np.abs(T - ref)
            d[~masks["T"]] = 0.0
            row["max_abs_anomaly_cells"] = float(d[anomaly].max()) \
                if anomaly.any() else None
            row["max_abs_nonanomaly_cells"] = float(d[~anomaly].max())
            out[t][name] = row
        for name in arms:
            if name != "unchanged":
                inc = np.abs(res["unchanged"] - res[name])
                inc[~masks["T"]] = 0.0
                out[t][name]["term_size_max_abs"] = float(inc.max())
    return out


SECTIONS = ("inputs", "gate", "ladder", "footprint_smt6b", "geothermal",
            "replay_trabbc", "stages_kt1", "bbl_replay", "stage3_from_nemo_stage2")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    parser.add_argument(
        "--sections", required=True,
        help=("comma-separated subset of " + ", ".join(SECTIONS) + ".  Every "
              "section jit-compiles its own models; run the heavy ones in "
              "separate processes (one process holding all of them exhausts "
              "the JIT code memory)"))
    args = parser.parse_args(argv)
    wanted = [x for x in args.sections.split(",") if x]
    require(all(x in SECTIONS for x in wanted), f"unknown section in {wanted}")
    require(not args.plant or "ladder" in wanted, "--plant needs the ladder")
    cards = _cards()
    report = {"format": "nemo-testcase-l1-vortex-smtrungs-round6-v2",
              "sections": wanted}
    entries = None

    def need_entries():
        return {t: _entries(t, cards[t], int(cards[t].recipe.z_coord.n_levels))
                for t in ("smt5", "smt6", "smt6b")}

    for name in wanted:
        if name == "inputs":
            report[name] = inputs_section(cards)
        elif name == "gate":
            report[name] = gate_section(cards)
        elif name == "ladder":
            report[name], entries = ladder_section(cards, args.plant)
        elif name == "footprint_smt6b":
            report[name] = footprint_section(cards, entries or need_entries())
        elif name == "geothermal":
            report[name] = geothermal_section(cards, entries or need_entries())
        elif name == "replay_trabbc":
            report[name] = replay_section(cards)
        elif name == "stages_kt1":
            report[name] = stage_section(cards)
        elif name == "bbl_replay":
            report[name] = bbl_replay_section(cards)
        elif name == "stage3_from_nemo_stage2":
            report[name] = stage3_arms_section(cards)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.plant:
        kt1 = report["ladder"]["smt6"]["card"]["independent"]["1"]["T"][
            "n_unequal"]
        require(kt1 == 1, f"plant moved {kt1} cells, expected exactly 1")
        print("PLANT-FIRED independent smt6 kt=1 T n_unequal", kt1)
        return 1
    status = report.get("inputs", {}).get("status", "NOT-RUN")
    print("SECTIONS", ",".join(wanted), "INPUTS-" + status)
    return 1 if status == "DIFFERS" else 0


if __name__ == "__main__":
    sys.exit(main())
