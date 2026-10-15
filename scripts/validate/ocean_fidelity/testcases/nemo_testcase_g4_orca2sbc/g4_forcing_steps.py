#!/usr/bin/env python3
"""G4 forcing at every recorded step (round-19 record, kt = 1..10).

``identity``: the shared fld_read + NCAR bulk evaluated on the CORE records
NEMO's log names at each bulk step, with NEMO's own surface mean (the
round-18 convention: the card cannot step forced), scored against every
surface field NEMO's 's' frame holds.  Fields with no card-side operator are
listed as such with NEMO's range.

``walk``: GIVEN-NEMO-ENTRY replay of turb_ncar.  Every statement of the
shared NCAR iteration is fed NEMO's own recorded operands (entry frame 'n',
previous iteration frame) and compared to NEMO's recorded result, so the
first unequal operand is that statement's own.

Refusals exit 73; a fired plant exits 1.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "nemo_testcase_g1_orca2grid"))
sys.path.insert(0, str(HERE.parents[1] / "orca2_l4"))
import check_records as cr  # noqa: E402
import nemo_testcase_l4_orca2_o1_numerical_gate as o1  # noqa: E402
from kt11_forcing_witness import ALBO, require  # noqa: E402
from nemo_testcase_phase3_trajectory_gate import score  # noqa: E402

FSBC = 2
ITER = "ABCDE"
# 's' fields with no card-side shared operator (Decision 123's subject)
NO_CARD_OPERATOR = ("taum", "wndm", "qns", "emp", "rnf", "erp", "qrp",
                    "rnf_tsc_tem", "rnf_tsc_sal")
ZERO_WITHOUT_ICE = ("sfx", "fr_i")          # no sea ice in G4: the card's value is 0
DT_BINS = ((-np.inf, 5.0), (5.0, 20.0), (20.0, np.inf))   # SST minus air T [K]


def require_program(run: Path, fields: dict) -> None:
    """The constants this instrument assumes are the run's: nn_hls = 2,
    nn_fsbc = FSBC, rn_zqt = rn_zu = 10 m (turb_ncar's height shift is
    skipped), no wave drag, and exactly len(ITER) recorded NCAR iterations."""
    log = (run / "ocean.output").read_text()
    for pattern, what in ((r"halo width .* nn_hls = +2$", "nn_hls = 2"),
                          (rf"nn_fsbc += +{FSBC}$", f"nn_fsbc = {FSBC}"),
                          (r"rn_zqt += +10\.0+ *$", "rn_zqt = 10"), (r"rn_zu += +10\.0+ *$", "rn_zu = 10")):
        require(re.search(pattern, log, re.M) is not None, f"{run.name}: the run is not {what}")
    decks = [(run / n).read_text() for n in ("namelist_cfg", "namelist_ref")]
    cdgw = [re.search(r"^ *ln_cdgw *= *\.(true|false)\.", d, re.M | re.I) for d in decks]
    require(next(m for m in cdgw if m).group(1).lower() == "false", f"{run.name}: wave drag on")
    letters = {k[0] for k in fields if len(k) > 1 and k[1] == "_" and k[0].isupper()}
    require(letters == set(ITER), f"{run.name}: recorded NCAR iterations {sorted(letters)}")


def _inner_slices(header: dict) -> tuple[slice, slice]:
    require(1 <= header["Nis0"] <= header["Nie0"] <= header["jpi"]
            and 1 <= header["Njs0"] <= header["Nje0"] <= header["jpj"],
            "invalid inner-domain bounds in the frame header")
    return (slice(header["Njs0"] - 1, header["Nje0"]),
            slice(header["Nis0"] - 1, header["Nie0"]))


def frames(run: Path, kt: int) -> tuple[dict, dict]:
    header, fields = cr.read_stream(
        cr.step_path(run, kt).read_bytes(), f"kt={kt}", allow_inner=True)
    require(header["step"] == kt,
            f"kt={kt}: frame header holds step {header['step']}")
    _inner_slices(header)
    return header, fields


def inner(a: np.ndarray, header: dict) -> np.ndarray:
    """Cut a full plane to the inner-domain bounds stored in its header."""
    slices = _inner_slices(header)
    full = (header["jpj"], header["jpi"])
    inner_shape = tuple(bound.stop - bound.start for bound in slices)
    require(a.shape in (full, inner_shape),
            f"field shape {a.shape} is neither full {full} nor inner {inner_shape}")
    return a[slices] if a.shape == full else a


def records(log: str, kt: int) -> dict:
    """0-based record per CORE field NEMO's log reads at bulk step kt."""
    out = {}
    for field, _, variable, _ in o1.FIELD_SPECS:
        hit = re.search(rf"fld_read: var {variable} kt = +{kt} .*record: +(\d+)", log)
        require(bool(hit), f"{variable}: NEMO's log has no record at kt = {kt}")
        out[field] = int(hit.group(1)) - 1
    return out


def relerr(card, nemo, wet) -> dict:
    card, nemo = np.asarray(card, np.float64), np.asarray(nemo, np.float64)
    unequal = (card.view("<u8") != nemo.view("<u8")) & wet
    rel = np.abs(card - nemo) / np.maximum(np.abs(nemo), np.finfo(float).tiny)
    return {"unequal_cells": int(unequal.sum()),
            "max_rel": float(rel[wet].max()) if wet.any() else 0.0,
            "argmax_jj_ji_inner_1based": [int(v) + 1 for v in
                                         np.unravel_index(np.argmax(np.where(wet, rel, -1)), rel.shape)]}


def split(card, nemo, wet, dtemp) -> dict:
    """Unequal cells and max relative error by SST minus air temperature."""
    out = {}
    unequal = (np.asarray(card).view("<u8") != np.asarray(nemo).view("<u8")) & wet
    rel = np.abs(card - nemo) / np.maximum(np.abs(nemo), np.finfo(float).tiny)
    for lo, hi in DT_BINS:
        sel = wet & (dtemp >= lo) & (dtemp < hi)
        out[f"{lo:g}..{hi:g}K"] = {"wet": int(sel.sum()), "unequal": int((unequal & sel).sum()),
                                   "max_rel": float(rel[sel].max()) if sel.any() else 0.0}
    return out


def card_bulk(run: Path, rec: dict, sst_c: np.ndarray, cache) -> dict:
    import jax
    maps = {}
    for field, filename, variable, bicubic in o1.FIELD_SPECS:
        with Dataset(run / filename) as ds:
            source = np.asarray(ds[variable][rec[field]].data, dtype=np.float64)
        idx, wgt = cache[bicubic]
        maps[field] = np.asarray(jax.jit(
            lambda v, s, w, b=bicubic: o1.nemo_fld_interp(v, s, w, bicubic=b))(source, idx, wgt))
    with Dataset(run / "domain_cfg.nc") as ds:
        cos, sin = o1.nemo_t_rotation_from_domain(*(
            np.asarray(ds[n][:].data, dtype=np.float64) for n in ("glamt", "gphit", "glamv", "gphiv")))
    maps["wndi"], maps["wndj"] = map(np.asarray, jax.jit(o1.rotate_en_to_ij)(
        maps["wndi"], maps["wndj"], cos, sin))
    tx, ty, sen, lat, evp = map(np.asarray, jax.jit(
        lambda u, v, ta, qa, ts, ps: o1.air_sea_fluxes(
            u, v, ta, qa, ts + o1.constants.T_freeze, slp_Pa=ps, algo="ncar"))(
        maps["wndi"], maps["wndj"], maps["tair"], maps["humi"], sst_c, maps["slp"]))
    return {"maps": maps, "utau": -tx, "vtau": -ty, "sen": sen, "lat": lat, "evp": evp}


def t_to_uv(tau_t, tmask, umask, vmask, along_i: bool):
    """Instrument mirror of sbcmod.f90:557-560 (T -> U/V stress); validated
    against NEMO's own recorded T and U/V stresses before use."""
    if along_i:
        nb, m, mk = np.roll(tau_t, -1, 1), umask, np.maximum(tmask, np.roll(tmask, -1, 1))
    else:
        nb, m, mk = np.roll(tau_t, -1, 0), vmask, np.maximum(tmask, np.roll(tmask, -1, 0))
    return 0.5 * (tau_t + nb) * (2.0 - m) * mk


def identity(run: Path, steps: int, plant: bool = False) -> dict:
    import jax
    jax.config.update("jax_enable_x64", True)
    o1.set_policy(o1.PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() == "cpu", "CPU only")
    log = (run / "ocean.output").read_text()
    cache = {kind: o1.load_weights(run, kind) for kind in (False, True)}
    with Dataset(run / "mesh_mask.nc") as ds:
        wet = np.asarray(ds["tmask"][0, 0].data, dtype=bool)
    out, card, kb_prev, tu_ok = {}, None, None, True
    _, first_fields = frames(run, 1)
    require_program(run, first_fields)
    for kt in range(1, steps + 1):
        header, f = frames(run, kt)
        full = (header["jpj"], header["jpi"])
        inner_slices = _inner_slices(header)
        kb = kt - (kt - 1) % FSBC
        if kb != kb_prev:      # the bulk runs at MOD(kt-1, nn_fsbc) == 0 (sbcblk.f90:554)
            rec = records(log, kb)
            card = card_bulk(run, rec, inner(f["s_sst_m"], header), cache)
            card["qsr"] = (1.0 - ALBO) * card["maps"]["qsr_down"] * wet
            dtemp = inner(f["s_sst_m"], header) + o1.constants.T_freeze - card["maps"]["tair"]
            kb_prev = kb
        rows = {}
        for name in ("qsr", "utau", "vtau"):
            nemo = inner(f[f"s_{name}"], header)
            row = score(f"kt{kt}.{name}", nemo, np.asarray(card[name], np.float64), wet,
                        plant=plant and kt == 1 and name == "qsr")
            row["unequal_cells"] = int(((np.asarray(card[name]).view("<u8")
                                         != nemo.view("<u8")) & wet).sum())
            if row["unequal_cells"]:
                row["by_sst_minus_tair"] = split(np.asarray(card[name]), nemo, wet, dtemp)
            rows[name] = {k: row[k] for k in ("status", "normalized_max_abs", "unequal_cells",
                                              "by_sst_minus_tair") if k in row}
        # U/V-point stress: the sbcmod statement first reproduces NEMO from NEMO's T stress
        tm = np.zeros(full); um = np.zeros(full); vm = np.zeros(full)
        with Dataset(run / "mesh_mask.nc") as ds:
            for arr, n in ((tm, "tmask"), (um, "umask"), (vm, "vmask")):
                arr[inner_slices] = ds[n][0, 0]
        for name, src, ref_mask, along in (("utauU", "utau", um, True), ("vtauV", "vtau", vm, False)):
            nemo_full = f[f"s_{name}"]
            # every face NEMO's statement can make nonzero: MAX(tmask here, tmask next)
            sel = t_to_uv(np.ones(full), tm, um, vm, along) != 0
            js, i_s = inner_slices
            sel[:js.start, :] = False; sel[js.stop - 1:, :] = False
            sel[:, :i_s.start] = False; sel[:, i_s.stop - 1:] = False
            mirror = t_to_uv(f[f"s_{src}"], tm, um, vm, along)
            tu_ok &= bool(np.array_equal(mirror[sel].view("<u8"), nemo_full[sel].view("<u8")))
            card_full = np.zeros(full); card_full[inner_slices] = card[src] * wet   # taum * smask0 (sbcblk.f90:795)
            cand = t_to_uv(card_full, tm, um, vm, along)
            row = score(f"kt{kt}.{name}", nemo_full, cand, sel)
            row["unequal_cells"] = int(((cand.view("<u8") != nemo_full.view("<u8")) & sel).sum())
            rows[name] = {k: row[k] for k in ("status", "normalized_max_abs", "unequal_cells")}
        for name in ZERO_WITHOUT_ICE:
            nemo = inner(f[f"s_{name}"], header)
            rows[name] = {"card": "0 (no sea ice)", "nemo_nonzero_wet_cells": int((nemo[wet] != 0).sum())}
        for name in NO_CARD_OPERATOR:
            nemo = inner(f[f"s_{name}"], header)
            rows[name] = {"card": "NO CARD OPERATOR",
                          "nemo_min_max_wet": [float(nemo[wet].min()), float(nemo[wet].max())]}
        for name in ("ssh_frc", "zu_frc", "zv_frc"):
            rows[name] = {"card": "NO CARD OPERATOR (composite of the above)"}
        out[kt] = {"bulk_step_used": kb, "records_0based": rec, "rows": rows}
    require(tu_ok, "the sbcmod T->U/V mirror does not reproduce NEMO's own stresses")
    return {"label": "INDEPENDENT forcing operators on NEMO's surface mean (round-18 convention)",
            "wet_cells": int(wet.sum()), "t_to_uv_mirror_validated": tu_ok, "steps": out}


# ---- the GIVEN-NEMO-ENTRY walk of turb_ncar -------------------------------
NCAR = "ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/sbcblk_algo_ncar.f90"
STATEMENTS = {   # recorded name -> NEMO statement it is the result of
    "ubzu": f"{NCAR}:130", "dtv": f"{NCAR}:133", "stab": f"{NCAR}:134", "cdn": f"{NCAR}:141",
    "ce": f"{NCAR}:148", "ch": f"{NCAR}:149",
    "one_on_l": f"{NCAR}:174", "zeta_u": f"{NCAR}:177-178", "psi_m": f"{NCAR}:195",
    "un10": f"{NCAR}:207", "it_cdn": f"{NCAR}:209", "sqrt_cdn": f"{NCAR}:210",
    "cd": f"{NCAR}:215-216", "sqrt_cd": f"{NCAR}:219", "psi_h_term": f"{NCAR}:220",
    "cd_ratio": f"{NCAR}:221", "it_stab": f"{NCAR}:223", "chn": f"{NCAR}:224", "cen": f"{NCAR}:225",
    "it_ch": f"{NCAR}:227", "it_ce": f"{NCAR}:228", "t_zu": f"{NCAR}:158", "q_zu": f"{NCAR}:159",
    "chain_cd": f"{NCAR}:130-228", "chain_ch": f"{NCAR}:130-228", "chain_ce": f"{NCAR}:130-228",
}


def walk(run: Path, steps: int, plant: bool = False) -> dict:
    import jax
    import jax.numpy as jnp
    from legoesm.core.bulk_flux import large_yeager_neutral_cd, psi_h, psi_m
    from legoesm.ocean import bulk_flux_omip as bof
    jax.config.update("jax_enable_x64", True)
    o1.set_policy(o1.PrecisionPolicy.fp64(transcendentals="libm"))
    require(jax.default_backend() == "cpu", "CPU only")
    _, first_fields = frames(run, 1)
    require_program(run, first_fields)
    k = o1.constants.kappa_von_karman
    with Dataset(run / "mesh_mask.nc") as ds:
        wet = np.asarray(ds["tmask"][0, 0].data, dtype=bool)
    a = np.asarray
    out = {}
    for kt in range(1, steps + 1, FSBC):
        _, f = frames(run, kt)
        n = {key[2:]: v for key, v in f.items() if key.startswith("n_")}
        dtemp = f["k_sspt"] - f["k_tair"]      # potential SST minus the bulk's air temperature
        rows = {}

        def row(name, card, nemo, x=None):
            card = a(card, np.float64).copy()
            if plant and not rows:
                card[tuple(np.argwhere(wet)[0])] += 1.0
            r = relerr(card, nemo, wet)
            r["statement"] = STATEMENTS[name.split(".")[-1]]
            if r["unequal_cells"]:
                r["by_sst_minus_tair"] = split(card, a(nemo, np.float64), wet, dtemp)
            rows[name] = r

        ub = a(jnp.maximum(n["u_zu"], bof._U10_FLOOR_M_S))
        row("ubzu", ub, n["ubzu"])
        row("dtv", a(bof._virt_temp(n["t_zt"], n["q_zt"]) - bof._virt_temp(n["sst"], n["ssq"])), n["dtv"])
        row("stab", np.where(n["dtv"] > 0.0, 1.0, 0.0), n["stab"])
        row("cdn", a(large_yeager_neutral_cd(n["ubzu"], nemo_parity=True)), n["cdn"])
        sq = np.sqrt(n["cdn"])
        row("ce", a(bof._ce_n10(sq)), n["ce"])
        row("ch", a(bof._ch_n10(sq, n["stab"])), n["ch"])
        floors = {"t_zu_ne_t_zt": int(((n["t_zu"] != n["t_zt"]) & wet).sum()),
                  "q_zu_ne_q_zt": int(((n["q_zu"] != n["q_zt"]) & wet).sum())}
        # the shared iteration uses theta and q as given (no NEMO entry floors)
        row("t_zu", n["t_zt"], n["t_zu"])
        row("q_zu", n["q_zt"], n["q_zu"])
        # whole shared NCAR on NEMO's entry operands vs NEMO's final coefficients
        cd, ch, ce, _ = map(a, bof.ncar_transfer_coefficients(n["t_zt"], n["q_zt"], n["sst"], n["ssq"],
                                                               n["u_zu"]))
        for name, c in (("chain_cd", cd), ("chain_ch", ch), ("chain_ce", ce)):
            row(name, c, f[f"E_{name[-2:]}"])
        prev = {"cd": n["cd"], "sqrt_cd": sq, "ch": n["ch"], "ce": n["ce"]}
        for letter in ITER:
            it = {key[2:]: v for key, v in f.items() if key.startswith(f"{letter}_")}
            p = f"{letter}."
            ts = prev["ch"] / prev["sqrt_cd"] * (n["t_zu"] - n["sst"])
            qs = prev["ce"] / prev["sqrt_cd"] * (n["q_zu"] - n["ssq"])
            row(p + "one_on_l", a(bof._one_on_obukhov(n["t_zu"], n["q_zu"], prev["sqrt_cd"] * n["ubzu"],
                                                      ts, qs)), it["one_on_l"])
            row(p + "zeta_u", a(jnp.clip(bof._ZU_M * it["one_on_l"], -bof._ZETA_ABS_MAX,
                                         bof._ZETA_ABS_MAX)), it["zeta_u"])
            row(p + "psi_m", a(psi_m(it["zeta_u"])), it["psi_m"])
            # shared: Ub (1 + sqrt(Cd)/k psi_m), floored (bulk_flux_omip.py ncar iteration)
            row(p + "un10", a(jnp.maximum(bof._UN10_FLOOR_M_S, n["ubzu"] * (
                1.0 + prev["sqrt_cd"] / k * it["psi_m"]))), it["un10"])
            row(p + "it_cdn", a(large_yeager_neutral_cd(it["un10"], nemo_parity=True)), it["cdn"])
            row(p + "sqrt_cdn", np.sqrt(it["cdn"]), it["sqrt_cdn"])
            den = 1.0 - it["sqrt_cdn"] / k * it["psi_m"]
            row(p + "cd", np.maximum(it["cdn"] / (den * den), bof._CX_MIN), it["cd"])
            row(p + "sqrt_cd", np.sqrt(it["cd"]), it["sqrt_cd"])
            row(p + "psi_h_term", a(-psi_h(it["zeta_u"]) / k / it["sqrt_cdn"]), it["psi_h_term"])
            row(p + "cd_ratio", it["sqrt_cd"] / it["sqrt_cdn"], it["cd_ratio"])
            row(p + "it_stab", np.where(it["zeta_u"] > 0.0, 1.0, 0.0), it["stab"])
            row(p + "chn", a(bof._ch_n10(it["sqrt_cdn"], it["stab"])), it["chn"])
            row(p + "cen", a(bof._ce_n10(it["sqrt_cdn"])), it["cen"])
            row(p + "it_ch", np.maximum(it["chn"] * it["cd_ratio"] / (1.0 + it["chn"] * it["psi_h_term"]),
                                        bof._CX_MIN), it["ch"])
            row(p + "it_ce", np.maximum(it["cen"] * it["cd_ratio"] / (1.0 + it["cen"] * it["psi_h_term"]),
                                        bof._CX_MIN), it["ce"])
            prev = {"cd": it["cd"], "sqrt_cd": it["sqrt_cd"], "ch": it["ch"], "ce": it["ce"]}
        first = next((name for name, r in rows.items() if r["unequal_cells"]), None)
        out[kt] = {"first_unequal": first, "entry_floors_binding": floors, "rows": rows}
    require(Path(bof.__file__).resolve().is_relative_to(HERE.parents[4]),
            f"shared code imported from another checkout: {bof.__file__}")
    return {"label": "GIVEN-NEMO-ENTRY (each statement on NEMO's recorded operands)",
            "shared_module": str(Path(bof.__file__).resolve().relative_to(HERE.parents[4])),
            "steps": out}


# Decision 123: the fields NEMO's ocean reads after sbc (round-19 's' frame),
# each consumed where NEMO consumes it -- stress: stp2d.f90:198-199 and
# dynzdf.f90:338, :506; emp/rnf: stp2d.f90:242-245 and trasbc.f90:285;
# qns/sfx: trasbc.f90:307-308; runoff content: trasbc.f90:323; qsr:
# traqsr.f90:371-373 (P5 build line numbers, the P6 build adds writer lines).
ROUTE_FIELDS = ("utau", "vtau", "qns", "qsr", "emp", "sfx", "rnf", "taum", "fr_i")
STAGE_ROWS = {"ssh": "ssh_aa", "uu_b": "uu_b_aa", "vv_b": "vv_b_aa",
              "u": "uu_k1_aa", "v": "vv_k1_aa", "T": "tn_k1_aa", "S": "sn_k1_aa"}


def nemo_supplied_forcing(card, run: Path, kt: int, *, zero: bool = False):
    """NEMO's own surface fields at step kt as the card's (freshwater, surface)
    inputs, through the ORCA2 lane's route unchanged (round-1 ladder gate
    ``_surface_forcings``).  ``zero``: every field 0 (the non-vacuity control)."""
    from nemo_testcase_l4_orca2_round1_ladder_gate import _surface_forcings
    header, f = frames(run, kt)
    fields = {name: inner(f[f"s_{name}"], header) for name in ROUTE_FIELDS}
    fields["rnf_tsc"] = np.stack([inner(f["s_rnf_tsc_tem"], header),
                                  inner(f["s_rnf_tsc_sal"], header)], axis=-1)
    if zero:
        fields = {k: np.zeros_like(v) for k, v in fields.items()}
    freshwater, surface = _surface_forcings(card, run, fields, kt)
    # NEMO's qns itself: the stage-3 heat statement reads it, and
    # round(qns + qsr) - qsr is not bitwise qns.
    return freshwater, surface._replace(q_nonsolar=fields["qns"])


# NEMO's surface-stress statements the G4 card selects (round 23); ``without``
# deselects them (the before arm).
STRESS_OPTIONS = ("barotropic_wind_nemo_stored_depth", "surface_stress_nemo_fold_vector_sign",
                  "zdf_surface_stress_nemo_face_thickness")
TRACER_UPDATE_OPTION = "rk3_stage_tracer_written_order"
STAGE3_HEAT_OPTION = "rk3_stage3_heat_nemo_written"
STAGE3_KMM_OPTION = "rk3_stage3_tracer_nemo_stored_thickness"   # round 30


def route(run: Path, deck_root: Path, *, zero: bool = False, stages: str = "123",
          dump: Path | None = None, without_stress_options: bool = False,
          without_tracer_update_option: bool = False,
          without_stage3_heat_option: bool = False,
          without_stage3_kmm_option: bool = False) -> dict:
    """GIVEN-NEMO-ENTRY, kt = 1: the G4 card from NEMO's as-built initial state
    (the record ran ln_tsd_dmp = F: no hand edits), forced by NEMO's kt = 1
    surface fields, each RK3 stage compared with NEMO's stage frame."""
    import jax
    import jax.numpy as jnp
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanModel, _NEMOWSRK3TestHooks)
    from legoesm.ocean.fidelity import nemo_testcase_recipe as recipe
    import nemo_testcase_g1_orca2grid_ladder as g1
    import nemo_testcase_g2_orca2bathy_ladder as g2
    g1._fp64()
    require(jax.default_backend() == "cpu", "CPU only")
    log = (run / "ocean.output").read_text()
    as_built = "ln_tsd_dmp  =  F" in log
    require(as_built or "ln_tsd_dmp  =  T" in log, "ln_tsd_dmp not in NEMO's log")
    card = recipe.build_nemo_testcase_card("ORCA2G4-sbc", deck_root=deck_root)
    require(all(getattr(card.recipe.model_config, o) is True for o in STRESS_OPTIONS),
            "the G4 card no longer selects NEMO's surface-stress statements")
    if without_stress_options:
        card = card._replace(recipe=card.recipe._replace(model_config=card.recipe.model_config._replace(
            **{o: False for o in STRESS_OPTIONS})))
    require(getattr(card.recipe.model_config, TRACER_UPDATE_OPTION) is True,
            "the G4 card no longer selects NEMO's tracer-update statement")
    if without_tracer_update_option:
        card = card._replace(recipe=card.recipe._replace(
            model_config=card.recipe.model_config._replace(
                **{TRACER_UPDATE_OPTION: False})))
    require(getattr(card.recipe.model_config, STAGE3_HEAT_OPTION) is True,
            "the G4 card no longer selects NEMO's stage-3 heat statement")
    if without_stage3_heat_option:
        card = card._replace(recipe=card.recipe._replace(
            model_config=card.recipe.model_config._replace(
                **{STAGE3_HEAT_OPTION: False})))
    require(getattr(card.recipe.model_config, STAGE3_KMM_OPTION) is True,
            "the G4 card no longer selects NEMO's stage-3 Kmm thickness")
    if without_stage3_kmm_option:
        card = card._replace(recipe=card.recipe._replace(
            model_config=card.recipe.model_config._replace(
                **{STAGE3_KMM_OPTION: False})))
    st = card.recipe.initial_state
    entry = st          # Decision 120 record: the card's state is NEMO's (hand edits)
    if as_built:        # rounds 17-19 records: NEMO ran without the hand edits
        wet = np.asarray(card.recipe.z_coord.is_active) > 0.5
        t0, s0 = recipe.build_orca2_initial_ts(
            run / "data_1m_potential_temperature_nomask.nc", run / "data_1m_salinity_nomask.nc",
            wet, apply_hand_alterations=False)
        entry = st._replace(T=st.T.replace(data=jnp.asarray(t0)), S=st.S.replace(data=jnp.asarray(s0)))
    freshwater, surface = nemo_supplied_forcing(card, run, 1, zero=zero)
    header, nemo = frames(run, 1)
    out = {"label": "GIVEN-NEMO-ENTRY (%s initial state), kt = 1" % (
               "as-built" if as_built else "Decision-120"), "zero_forcing": zero,
           "without_stress_options": without_stress_options,
           "without_tracer_update_option": without_tracer_update_option,
           "without_stage3_heat_option": without_stage3_heat_option,
           "without_stage3_kmm_option": without_stage3_kmm_option,
           "stages": {}}
    for stage, (hooks, _) in (("1", g2.EXPOSURES["stage1"]), ("2", g2.EXPOSURES["stage2"]),
                              ("3", g2.EXPOSURES["stage3"])):
        if stage not in stages:
            continue
        new = jax.device_get(LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=_NEMOWSRK3TestHooks(**hooks)).step(
                entry, dt=card.dt_s, freshwater=freshwater, surface_forcing=surface))
        cand = g2.card_surface(new)
        rows = {}
        for name, key in STAGE_ROWS.items():
            ref = np.asarray(inner(nemo[f"{stage}_{key}"], header), np.float64)
            got = np.asarray(cand[name], np.float64)
            rows[name] = {"unequal": int((ref.view("<u8") != got.view("<u8")).sum()),
                          "unequal_values": int((ref != got).sum()),
                          "max_abs": float(np.max(np.abs(ref - got)))}
        out["stages"][stage] = rows
        if dump is not None:
            np.savez(Path(dump) / f"route_stage{stage}.npz",
                     **{f"card_{n}": np.asarray(cand[n]) for n in STAGE_ROWS},
                     **{f"nemo_{n}": inner(nemo[f"{stage}_{k}"], header) for n, k in STAGE_ROWS.items()})
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("identity", "walk", "route"))
    p.add_argument("--deck-root", type=Path, help="route: the G4 deck with the record's inputs")
    p.add_argument("--zero", action="store_true", help="route: zero forcing (control)")
    p.add_argument("--stages", default="123", help="route: which RK3 stages")
    p.add_argument("--without-stage3-heat", action="store_true",
                   help="route: deselect the stage-3 heat statement (control)")
    p.add_argument("--without-stage3-kmm", action="store_true",
                   help="route: deselect the stage-3 stored Kmm thickness (control)")
    p.add_argument("--dump", type=Path, help="route: save card and NEMO rows (npz)")
    p.add_argument("--run", type=Path, required=True, help="twin a of the round-19 record")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--json", type=Path)
    p.add_argument("--plant", action="store_true")
    a = p.parse_args()
    if a.mode == "route":
        out = route(a.run, a.deck_root, zero=a.zero, stages=a.stages, dump=a.dump,
                    without_stage3_heat_option=a.without_stage3_heat,
                    without_stage3_kmm_option=a.without_stage3_kmm)
        if a.json:
            a.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(json.dumps(out, sort_keys=True))
        same = all(r["unequal"] == 0 for rows in out["stages"].values() for r in rows.values())
        return 0 if same else 1
    measure = identity if a.mode == "identity" else walk
    if a.plant:
        if a.mode == "identity":
            fired = measure(a.run, 1, plant=True)["steps"][1]["rows"]["qsr"]["status"] != "AT-BAR"
        else:
            r = measure(a.run, 1, plant=True)["steps"][1]["rows"]["ubzu"]
            fired = r["unequal_cells"] == 1
        print("STATUS PLANT-FIRED" if fired else "REFUSE: plant did not fire")
        return 1 if fired else 2
    out = measure(a.run, a.steps)
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
