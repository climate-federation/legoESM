#!/usr/bin/env python3
"""Exact-entry gate for the first rung-3.6 coupled-column boundaries."""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
from pathlib import Path

import jax
import numpy as np

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.core.source_rounding import nemo_source_round
from legoesm.coupler.ocean_forcing import (
    NemoSI3ExchangeConfig,
    nemo_si3_exchange_forcing,
    nemo_si3_fwb_step,
    nemo_si3_ssm_step,
    nemo_si3_tra_sbc_rk3,
)
from legoesm.ocean.eos import nemo_eos_fzp
from legoesm.ocean.physics.shortwave_penetration import nemo_rgb_one_layer_rhs
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
from legoesm.ice.sea_ice import (
    _nemo_si3_ice_update_flux,
    _nemo_si3_ice_update_tau,
)


BAR = 1.0e-15
SSM_MAGIC = b"NEMO_L3SSM__001 "
ZIN_MAGIC = b"NEMO_L3ZIN_002  "
SSM_FIELDS = ("u", "v", "sst", "sss", "ssh", "e3t", "fraqsr")
UPDATE_MAGIC = b"NEMO_L3UPD__001 "
UPDATE_PRE_FIELDS = (
    "at_i_b", "at_i", "a_i_b", "qns_oce", "qsr_oce", "qemp_oce", "emp_oce",
    "qcn_ice", "qml_ice", "qtr_ice_top", "qemp_ice", "qevap_ice",
    "qsr_tot", "qns_tot", "qtr_ice_bot", "qsr_ice", "fhld",
    "hfx_sum", "hfx_bom", "hfx_bog", "hfx_dif", "hfx_opw", "hfx_snw",
    "hfx_thd", "hfx_dyn", "hfx_res", "hfx_sub", "hfx_spr",
    "wfx_bog", "wfx_bom", "wfx_sum", "wfx_sni", "wfx_opw", "wfx_dyn",
    "wfx_res", "wfx_lam", "wfx_snw_sni", "wfx_snw_dyn", "wfx_snw_sum",
    "wfx_pnd", "wfx_err_sub", "wfx_snw_sub", "wfx_ice_sub",
    "sfx_bog", "sfx_bom", "sfx_sum", "sfx_sni", "sfx_opw", "sfx_res",
    "sfx_dyn", "sfx_bri", "sfx_sub", "sfx_lam",
    "vt_s", "vt_i", "vt_ip", "vt_il", "snwice_mass",
    "snwice_mass_b", "frq_m",
)
UPDATE_POST_FIELDS = (
    "qsr", "qns", "emp", "sfx", "qt_atm_oi", "qt_oce_ai", "fwfice",
    "snwice_mass_b", "snwice_mass", "snwice_fmass", "fr_i", "tn_ice",
    "alb_ice", "wfx_ice", "wfx_snw", "wfx_sub",
)
UPDATE_TAU_FIELDS = (
    "u_ocean_instant", "v_ocean_instant", "refresh", "tmod_io", "at_i",
    "u_ice", "u_ice_west", "v_ice", "v_ice_south", "drag_io",
    "utau_oce", "vtau_oce", "rCdU_ice", "utau", "vtau", "taum",
    "taum_before",
)
FWB_MAGIC = b"NEMO_L3FWB__001 "
FWB_FIELDS = (
    "emp_pre", "qns_pre", "snwice_fmass", "area", "mask", "emp_ext",
    "emp_corr_pre", "rcp", "sst", "active", "emp", "qns", "emp_corr",
)
TRASBC_MAGIC = b"NEMO_L3TSB__001 "
TRASBC_FIELDS = (
    "t_pre", "s_pre", "t_post", "s_post", "emp", "qns", "sfx",
    "e3t", "rho0", "r1_rho0", "rcp", "r1_rcp", "t_bb", "s_bb", "mask",
)
QSR_MAGIC = b"NEMO_L3QSR__001 "
QSR_FIELDS = (
    "rhs_pre", "rhs_post", "qsr", "chl", "e3t", "r1_rho0_rcp",
    "fraqsr_1lev", "itab", "rn_abs", "rn_si0", "bottom_wmask",
)

_DRIFT_PATH = Path(__file__).with_name("nemo_si3_exchange_drift_gate.py")
_SPEC = importlib.util.spec_from_file_location("rung36_exchange_layout", _DRIFT_PATH)
assert _SPEC and _SPEC.loader
drift = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(drift)

_BULK_PATH = Path(__file__).with_name("nemo_si3_bulk_flux_gate.py")
_BULK_SPEC = importlib.util.spec_from_file_location("rung36_bulk_layout", _BULK_PATH)
assert _BULK_SPEC and _BULK_SPEC.loader
bulk = importlib.util.module_from_spec(_BULK_SPEC)
_BULK_SPEC.loader.exec_module(bulk)


def _ssm(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    instant, before, after, steps, stages = [], [], [], [], []
    with path.open("rb") as stream:
        while magic := stream.read(16):
            if magic != SSM_MAGIC:
                raise ValueError("bad SSM magic")
            version, kt, _kbb, _kmm, stage, count, bits = struct.unpack(
                "=7i", stream.read(28))
            if (version, count, bits) != (1, 15, 64):
                raise ValueError("bad SSM header")
            z = np.fromfile(stream, np.float64, count)
            steps.append(kt); stages.append(stage); instant.append(z[:7]);
            (before if stage == 0 else after).append(z[7:14])
    if stages != [value for _ in range(len(stages)//2) for value in (0, 1)]:
        raise ValueError("SSM frame order")
    return (np.asarray(instant[::2]), np.asarray(before), np.asarray(after))


def _zdf_inputs(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps, salinity, t_bo = [], [], []
    with path.open("rb") as stream:
        while magic := stream.read(16):
            if magic != ZIN_MAGIC:
                raise ValueError("bad ZDF-input magic")
            version, kt, _cat, npti, bits, count = struct.unpack("=6i", stream.read(24))
            if version != 2 or npti != 1 or bits != 64:
                raise ValueError("bad ZDF-input header")
            z = np.fromfile(stream, np.float64, count)
            steps.append(kt); t_bo.append(z[4]); salinity.append(z[5])
    return np.asarray(steps), np.asarray(salinity), np.asarray(t_bo)


def _exchange(path: Path) -> dict[str, np.ndarray]:
    info = drift.inspect_stream(path)
    nx, ny, nc = (info["geometry"][key] for key in ("nx", "ny", "nc"))
    layout, record_bytes = drift._layout(nx, ny, nc)
    data = np.memmap(path, np.uint8, "r", shape=(info["steps"], record_bytes))
    active_index = {"reduced": 0, "halo1": 4, "full": 12}
    result = {}
    for spec, start, _count in layout:
        index = active_index[spec.size]
        offset = start + 8*index
        result[spec.name] = np.ndarray(
            (info["steps"],), dtype="=f8", buffer=data,
            offset=offset, strides=(record_bytes,)).copy()
    return result


def _update(path: Path) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    pre, post, tau = [], [], []
    with path.open("rb") as stream:
        while magic := stream.read(16):
            if magic != UPDATE_MAGIC:
                raise ValueError("bad update-frame magic")
            version, _kt, stage, count, bits = struct.unpack("=5i", stream.read(20))
            if version != 1 or bits != 64:
                raise ValueError("bad update-frame header")
            z = np.fromfile(stream, np.float64, count)
            if stage == 0:
                if count != len(UPDATE_PRE_FIELDS):
                    raise ValueError("bad PRE_UPDATE_FLX count")
                pre.append(z)
            elif stage == 1:
                if count != len(UPDATE_POST_FIELDS):
                    raise ValueError("bad POST_UPDATE_FLX count")
                post.append(z)
            elif stage == 2 and count == len(UPDATE_TAU_FIELDS):
                tau.append(z)
            else:
                raise ValueError("bad POST_UPDATE_TAU count")
    return (
        {name: np.asarray(pre)[:, index] for index, name in enumerate(UPDATE_PRE_FIELDS)},
        {name: np.asarray(post)[:, index] for index, name in enumerate(UPDATE_POST_FIELDS)},
        {name: np.asarray(tau)[:, index] for index, name in enumerate(UPDATE_TAU_FIELDS)},
    )


def _bulk_stage2(path: Path) -> dict[str, np.ndarray]:
    """Read the every-fourth-step ice-flux operands from the coupled oracle."""
    rows = []
    expected_kt = 1
    with path.open("rb") as stream:
        while magic := stream.read(16):
            if magic != bulk.MAGIC:
                raise ValueError("bad bulk magic")
            version, kt, stage, count, bits = struct.unpack("=5i", stream.read(20))
            if (version, kt, count, bits) != (
                1, expected_kt, len((bulk.STAGE0_NAMES, bulk.STAGE1_NAMES,
                                    bulk.STAGE2_NAMES)[stage]), 64,
            ):
                raise ValueError("bad coupled bulk header")
            values = np.fromfile(stream, np.float64, count)
            if stage == 2:
                rows.append(values)
                expected_kt += 4
    return bulk._as_dict(np.asarray(rows), bulk.STAGE2_NAMES)


def _fwb(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open("rb") as stream:
        expected_kt = 1
        while magic := stream.read(16):
            if magic != FWB_MAGIC:
                raise ValueError("bad FWB magic")
            version, kt, count, bits = struct.unpack("=4i", stream.read(16))
            if (version, kt, count, bits) != (
                1, expected_kt, len(FWB_FIELDS), 64,
            ):
                raise ValueError("bad FWB header")
            rows.append(np.fromfile(stream, np.float64, count))
            expected_kt += 1
    values = np.asarray(rows)
    return {name: values[:, index] for index, name in enumerate(FWB_FIELDS)}


def _trasbc(path: Path) -> dict[str, np.ndarray]:
    rows, stages = [], []
    with path.open("rb") as stream:
        while magic := stream.read(16):
            if magic != TRASBC_MAGIC:
                raise ValueError("bad tra_sbc magic")
            h = struct.unpack("=8i", stream.read(32))
            version, _kt, stage, _kbb, _kmm, _krhs, count, bits = h
            if (version, count, bits) != (1, len(TRASBC_FIELDS), 64):
                raise ValueError("bad tra_sbc header")
            stages.append(stage)
            rows.append(np.fromfile(stream, np.float64, count))
    values = np.asarray(rows)
    result = {name: values[:, index] for index, name in enumerate(TRASBC_FIELDS)}
    result["stage"] = np.asarray(stages)
    return result


def _qsr(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open("rb") as stream:
        while magic := stream.read(16):
            if magic != QSR_MAGIC:
                raise ValueError("bad QSR magic")
            version, _kt, _kmm, _krhs, count, bits = struct.unpack(
                "=6i", stream.read(24))
            if (version, count, bits) != (1, len(QSR_FIELDS), 64):
                raise ValueError("bad QSR header")
            rows.append(np.fromfile(stream, np.float64, count))
    values = np.asarray(rows)
    return {name: values[:, index] for index, name in enumerate(QSR_FIELDS)}


def _summary(name: str, got, wanted) -> dict[str, object]:
    got = np.asarray(got, np.float64); wanted = np.asarray(wanted, np.float64)
    error = np.abs(got - wanted)
    norm = error / np.maximum(np.abs(wanted), 1.0)
    bit = got.view(np.uint64) == wanted.view(np.uint64)
    first = np.flatnonzero(norm > BAR)
    return {
        "name": name,
        "rows": int(got.size),
        "bit_identical": int(np.sum(bit)),
        "non_bit": int(np.sum(~bit)),
        "max_normalized_error": float(np.max(norm, initial=0.0)),
        "first_over_bar_index": int(first[0]) if first.size else None,
        "status": "AT_BAR" if not first.size else "DEBT",
    }


def evaluate(root: Path, plant: str | None = None) -> dict[str, object]:
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    config = NemoSI3ExchangeConfig()
    instant, before, after = _ssm(root / "oracle_rung36_ssm_frames.bin")
    steps = np.arange(1, after.shape[0] + 1, dtype=np.int64)
    if plant == "ssm_sample":
        instant[4, 2] += 1.0e-8
    got_ssm = np.stack([
        np.asarray(value) for value in nemo_si3_ssm_step(
            tuple(jax.numpy.asarray(before[:, k]) for k in range(7)),
            tuple(jax.numpy.asarray(instant[:, k]) for k in range(7)),
            kt=jax.numpy.asarray(steps), config=config)
    ], axis=1)
    rows = [_summary(f"POST_SSM.{name}", got_ssm[:, k], after[:, k])
            for k, name in enumerate(SSM_FIELDS)]

    ice_steps, salinity, t_bo = _zdf_inputs(root / "oracle_si3_zdf_inputs.bin")
    if plant == "fzp_operand":
        salinity[0] += 1.0e-8
    got_tbo = np.asarray(nemo_eos_fzp(jax.numpy.asarray(salinity))) + constants.T_freeze
    fzp_row = _summary("POST_FZP.t_bo", got_tbo, t_bo)
    rows.append(fzp_row)

    update_path = root / "oracle_rung36_update_frames.bin"
    if update_path.exists():
        update_pre, update_post, update_tau = _update(update_path)
        if plant == "update_heat":
            update_pre["hfx_sum"][0] += 1.0e-8
        got_update = _nemo_si3_ice_update_flux(
            {name: jax.numpy.asarray(value) for name, value in update_pre.items()},
            ice_constants=NEMO_SI3_CONSTANTS_CONFIG, dt=14400.0)
        for name in (
            "qsr", "qns", "emp", "sfx", "qt_atm_oi", "qt_oce_ai",
            "fwfice", "snwice_mass_b", "snwice_mass", "snwice_fmass",
            "fr_i", "wfx_ice", "wfx_snw", "wfx_sub",
        ):
            rows.append(_summary(
                f"POST_UPDATE_FLX.{name}", np.asarray(got_update[name]),
                update_post[name]))

        bulk_stage2 = _bulk_stage2(root / "oracle_si3_bulk_operands.bin")
        refresh = update_tau["refresh"] != 0.0
        tmod_before = np.concatenate((
            update_tau["tmod_io"][:1], update_tau["tmod_io"][:-1]))
        expected_utau_oce = np.empty_like(update_tau["utau_oce"])
        expected_vtau_oce = np.empty_like(update_tau["vtau_oce"])
        ice_index = 0
        for index, is_refresh in enumerate(refresh):
            if is_refresh:
                expected_utau_oce[index] = bulk_stage2["utau"][ice_index]
                expected_vtau_oce[index] = bulk_stage2["vtau"][ice_index]
                ice_index += 1
            else:
                expected_utau_oce[index] = expected_utau_oce[index - 1]
                expected_vtau_oce[index] = expected_vtau_oce[index - 1]
        rows.append(_summary(
            "POST_UPDATE_TAU.utau_oce", expected_utau_oce,
            update_tau["utau_oce"]))
        rows.append(_summary(
            "POST_UPDATE_TAU.vtau_oce", expected_vtau_oce,
            update_tau["vtau_oce"]))
        got_tau = _nemo_si3_ice_update_tau(
            u_ocean=jax.numpy.asarray(after[:, 0]),
            v_ocean=jax.numpy.asarray(after[:, 1]),
            u_ice=jax.numpy.asarray(update_tau["u_ice"]),
            u_ice_west=jax.numpy.asarray(update_tau["u_ice_west"]),
            v_ice=jax.numpy.asarray(update_tau["v_ice"]),
            v_ice_south=jax.numpy.asarray(update_tau["v_ice_south"]),
            drag_io=jax.numpy.asarray(update_tau["drag_io"]),
            ice_fraction=jax.numpy.asarray(update_tau["at_i"]),
            tmod_before=jax.numpy.asarray(tmod_before),
            taum_before=jax.numpy.asarray(update_tau["taum_before"]),
            utau_ocean=jax.numpy.asarray(update_tau["utau_oce"]),
            vtau_ocean=jax.numpy.asarray(update_tau["vtau_oce"]),
            refresh=jax.numpy.asarray(refresh),
            rho_ocean=NEMO_SI3_CONSTANTS_CONFIG.rho_ocean,
        )
        for name in ("tmod_io", "rCdU_ice", "utau", "vtau"):
            rows.append(_summary(
                f"POST_UPDATE_TAU.{name}", np.asarray(got_tau[name]),
                update_tau[name]))
        rows.append(_summary(
            "POST_UPDATE_TAU.taum", np.asarray(got_tau["taum"]),
            update_tau["taum"]))

    fwb_path = root / "oracle_rung36_fwb_frames.bin"
    if fwb_path.exists():
        fwb = _fwb(fwb_path)
        if plant == "fwb_mass":
            fwb["snwice_fmass"][0] += 1.0e-8
        current_domain_sum = np.asarray(nemo_source_round(
            jax.numpy.asarray(fwb["area"]) * nemo_source_round(
                jax.numpy.asarray(fwb["emp_pre"])
                - jax.numpy.asarray(fwb["snwice_fmass"]))))
        delayed_domain_sum = current_domain_sum.copy()
        active_indices = np.flatnonzero(fwb["active"] != 0.0)
        if plant != "fwb_immediate":
            delayed_domain_sum[active_indices[1:]] = current_domain_sum[
                active_indices[:-1]]
        got_emp, got_qns, got_corr = nemo_si3_fwb_step(
            emp=jax.numpy.asarray(fwb["emp_pre"]),
            qns=jax.numpy.asarray(fwb["qns_pre"]),
            snwice_fmass=jax.numpy.asarray(fwb["snwice_fmass"]),
            area=jax.numpy.asarray(fwb["area"]),
            mask=jax.numpy.asarray(fwb["mask"]),
            emp_ext=jax.numpy.asarray(fwb["emp_ext"]),
            emp_corr=jax.numpy.asarray(fwb["emp_corr_pre"]),
            domain_sum=jax.numpy.asarray(delayed_domain_sum),
            heat_capacity=jax.numpy.asarray(fwb["rcp"]),
            sst=jax.numpy.asarray(fwb["sst"]),
            active=jax.numpy.asarray(fwb["active"] != 0.0), config=config,
        )
        for name, got in (
            ("emp", got_emp), ("qns", got_qns), ("emp_corr", got_corr),
        ):
            rows.append(_summary(
                f"POST_FWB.{name}", np.asarray(got), fwb[name]))

    trasbc_path = root / "oracle_rung36_trasbc_frames.bin"
    if trasbc_path.exists():
        trasbc = _trasbc(trasbc_path)
        if plant == "trasbc_heat":
            first_stage3 = int(np.flatnonzero(trasbc["stage"] == 3)[0])
            trasbc["qns"][first_stage3] += 1.0e-4
        got_t, got_s = nemo_si3_tra_sbc_rk3(
            tendency_t=jax.numpy.asarray(trasbc["t_pre"]),
            tendency_s=jax.numpy.asarray(trasbc["s_pre"]),
            emp=jax.numpy.asarray(trasbc["emp"]),
            qns=jax.numpy.asarray(trasbc["qns"]),
            salt_flux_pss=jax.numpy.asarray(trasbc["sfx"]),
            layer_thickness=jax.numpy.asarray(trasbc["e3t"]),
            inverse_density=jax.numpy.asarray(trasbc["r1_rho0"]),
            inverse_heat_capacity=jax.numpy.asarray(trasbc["r1_rcp"]),
            temperature=jax.numpy.asarray(trasbc["t_bb"]),
            salinity=jax.numpy.asarray(trasbc["s_bb"]),
            stage=jax.numpy.asarray(trasbc["stage"]), config=config,
        )
        rows.append(_summary("POST_TRA_SBC_RK3.temperature", np.asarray(got_t),
                             trasbc["t_post"]))
        rows.append(_summary("POST_TRA_SBC_RK3.salinity", np.asarray(got_s),
                             trasbc["s_post"]))

    qsr_path = root / "oracle_rung36_qsr_frames.bin"
    if qsr_path.exists():
        qsr = _qsr(qsr_path)
        if plant == "qsr_flux":
            qsr["qsr"][0] += 1.0e-4
        got_rhs = nemo_rgb_one_layer_rhs(
            jax.numpy.asarray(qsr["rhs_pre"]), jax.numpy.asarray(qsr["qsr"]),
            jax.numpy.asarray(qsr["e3t"]),
            jax.numpy.asarray(qsr["r1_rho0_rcp"]),
        )
        rows.append(_summary("POST_TRA_QSR.temperature", np.asarray(got_rhs),
                             qsr["rhs_post"]))
        rows.append(_summary(
            "POST_TRA_QSR.fraqsr_1lev", np.ones_like(qsr["fraqsr_1lev"]),
            qsr["fraqsr_1lev"]))

    fields = _exchange(root / "oracle_si3_exchange_frames.bin")
    emp = fields["emp"].copy()
    if plant == "freshwater_sign":
        emp = -emp
    freshwater, surface = nemo_si3_exchange_forcing(
        qsr=jax.numpy.asarray(fields["qsr"]),
        qns=jax.numpy.asarray(fields["qns"]),
        emp=jax.numpy.asarray(emp), sfx=jax.numpy.asarray(fields["sfx"]),
        utau=jax.numpy.asarray(fields["utau"]),
        vtau=jax.numpy.asarray(fields["vtau"]),
        chl=jax.numpy.ones_like(jax.numpy.asarray(fields["qsr"])),
        rCdU_ice=jax.numpy.asarray(fields["rCdU_ice"]),
        snwice_fmass=jax.numpy.asarray(fields["snwice_fmass"]), config=config)
    mapped = {
        "freshwater": np.asarray(freshwater.ice_fw),
        "q_net": np.asarray(surface.q_net), "sw_down": np.asarray(surface.sw_down),
        "tau_x": np.asarray(surface.tau_x), "tau_y": np.asarray(surface.tau_y),
        "salt_flux": np.asarray(surface.salt_flux),
        "rCdU_top": np.asarray(surface.rCdU_top),
        "snwice_fmass": np.asarray(surface.snwice_fmass),
    }
    expected = {
        "freshwater": -fields["emp"], "q_net": fields["qsr"] + fields["qns"],
        "sw_down": fields["qsr"], "tau_x": -fields["utau"],
        "tau_y": -fields["vtau"],
        "salt_flux": fields["sfx"] * constants.pss_to_mass_fraction,
        "rCdU_top": fields["rCdU_ice"], "snwice_fmass": fields["snwice_fmass"],
    }
    rows.extend(_summary(f"EXCHANGE_CARD.{name}", mapped[name], expected[name])
                for name in mapped)
    first = next((row for row in rows if row["status"] == "DEBT"), None)
    promoted = {
        "tn_ice": "POST_UPDATE_FLX registered output",
        "qml_ice": "PRE_UPDATE_FLX exact-entry operand from ice_thd",
        "qcn_ice": "PRE_UPDATE_FLX exact-entry operand from ice_thd",
        "sstfrz": "POST_FZP scored output",
        "rCdU_ice": "POST_UPDATE_TAU scored output",
        "snwice_mass": "POST_UPDATE_FLX scored output",
        "snwice_mass_b": "POST_UPDATE_FLX scored output",
        "snwice_fmass": "POST_UPDATE_FLX scored output and FWB input",
        "utau": "POST_UPDATE_TAU scored output",
        "vtau": "POST_UPDATE_TAU scored output",
        "taum": "POST_UPDATE_TAU scored output",
        "qns": "POST_UPDATE_FLX and POST_FWB scored output",
        "sfx": "POST_UPDATE_FLX scored output",
    }
    coverage = []
    for name in bulk.EXPECTED_EXCHANGE_FIELDS:
        status, reason = bulk.COVERAGE[name]
        if name in promoted:
            status, reason = "VERIFIED", promoted[name]
        coverage.append({"name": name, "status": status, "reason": reason})
    if len(coverage) != 36 or len({row["name"] for row in coverage}) != 36:
        raise ValueError("exchange coverage register is not one-to-one")
    plant_target = {
        "ssm_sample": "POST_SSM.sst", "fzp_operand": "POST_FZP.t_bo",
        "update_heat": "POST_UPDATE_FLX.qns", "fwb_mass": "POST_FWB.emp",
        "fwb_immediate": "POST_FWB.emp",
        "trasbc_heat": "POST_TRA_SBC_RK3.temperature",
        "qsr_flux": "POST_TRA_QSR.temperature",
        "freshwater_sign": "EXCHANGE_CARD.freshwater",
    }.get(plant)
    return {
        "verdict": "AT_BAR" if first is None else "DEBT",
        "bar": BAR,
        "backend": jax.default_backend(),
        "dtype": str(got_ssm.dtype),
        "precision_policy": {"mode": "fp64", "transcendentals": "libm"},
        "oracle_steps": int(after.shape[0]), "ice_steps": int(ice_steps.size),
        "rows": rows, "first_over_bar": first, "coverage": coverage,
        "plant": plant,
        "plant_binding": None if plant is None else {
            "target": plant_target,
            "red": next(row for row in rows if row["name"] == plant_target)["status"] == "DEBT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--plant", choices=(
        "ssm_sample", "fzp_operand", "update_heat", "fwb_mass",
        "fwb_immediate",
        "trasbc_heat",
        "qsr_flux",
        "freshwater_sign"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.root, args.plant)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.plant is not None:
        return 1 if report["plant_binding"]["red"] else 2
    return 0 if report["verdict"] == "AT_BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
