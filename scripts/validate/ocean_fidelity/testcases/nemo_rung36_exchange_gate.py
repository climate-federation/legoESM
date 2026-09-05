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
from netCDF4 import Dataset

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
from legoesm.ice.c1d_omip_l3 import build_c1d_omip_l3_coupled_card
from legoesm.ice.sea_ice import (
    _nemo_si3_ice_update_flux,
    _nemo_si3_ice_update_tau,
)


BAR = 1.0e-15


class GateError(RuntimeError):
    """A fail-closed rung-3.6 input or schema error."""


REQUIRED_STREAMS = (
    "oracle_rung36_ssm_frames.bin",
    "oracle_si3_zdf_inputs.bin",
    "oracle_rung36_update_frames.bin",
    "oracle_si3_bulk_operands.bin",
    "oracle_rung36_fwb_frames.bin",
    "oracle_rung36_trasbc_frames.bin",
    "oracle_rung36_qsr_frames.bin",
    "oracle_si3_exchange_frames.bin",
)
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


def _ssm(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    instant, before, after, registry = [], [], [], []
    with path.open("rb") as stream:
        while magic := stream.read(16):
            if magic != SSM_MAGIC:
                raise ValueError("bad SSM magic")
            version, kt, kbb, kmm, stage, count, bits = struct.unpack(
                "=7i", stream.read(28))
            if (version, count, bits) != (1, 15, 64):
                raise ValueError("bad SSM header")
            z = np.fromfile(stream, np.float64, count)
            registry.append((kt, kbb, kmm, stage)); instant.append(z[:7]);
            (before if stage == 0 else after).append(z[7:14])
    expected = []
    for kt in range(1, len(registry) // 2 + 1):
        level = 1 if kt % 2 else 3
        expected.extend(((kt, level, level, 0), (kt, level, level, 1)))
    if registry != expected:
        raise ValueError("SSM kt/Kbb/Kmm/stage registry")
    info = {
        "records": len(registry),
        "rule": "two stages per kt; odd Kbb=Kmm=1, even Kbb=Kmm=3",
        "first": list(registry[0]), "last": list(registry[-1]),
    }
    return (np.asarray(instant[::2]), np.asarray(before), np.asarray(after), info)


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
    expected = list(range(1, 4 * len(steps), 4))
    if steps != expected:
        raise ValueError("ZDF-input kt cadence registry")
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
    pre, post, tau, sequence = [], [], [], []
    with path.open("rb") as stream:
        while magic := stream.read(16):
            if magic != UPDATE_MAGIC:
                raise ValueError("bad update-frame magic")
            version, kt, stage, count, bits = struct.unpack("=5i", stream.read(20))
            if version != 1 or bits != 64:
                raise ValueError("bad update-frame header")
            z = np.fromfile(stream, np.float64, count)
            sequence.append((kt, stage))
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
    expected = []
    for kt in range(1, len(tau) + 1):
        if (kt - 1) % 4 == 0:
            expected.extend(((kt, 0), (kt, 1)))
        expected.append((kt, 2))
    if sequence != expected:
        raise ValueError("update-frame kt/stage registry")
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
    rows, stages, registry = [], [], []
    with path.open("rb") as stream:
        while magic := stream.read(16):
            if magic != TRASBC_MAGIC:
                raise ValueError("bad tra_sbc magic")
            h = struct.unpack("=8i", stream.read(32))
            version, kt, stage, kbb, kmm, krhs, count, bits = h
            if (version, count, bits) != (1, len(TRASBC_FIELDS), 64):
                raise ValueError("bad tra_sbc header")
            stages.append(stage)
            registry.append((kt, stage, kbb, kmm, krhs))
            rows.append(np.fromfile(stream, np.float64, count))
    values = np.asarray(rows)
    expected = []
    for kt in range(1, len(registry) // 3 + 1):
        kbb = 1 if kt % 2 else 3
        krhs = 3 if kt % 2 else 1
        expected.extend((
            (kt, 1, kbb, kbb, krhs),
            (kt, 2, kbb, krhs, 2),
            (kt, 3, kbb, 2, krhs),
        ))
    if registry != expected:
        raise ValueError("tra_sbc exact kt/stage/Kbb/Kmm/Krhs registry")
    result = {name: values[:, index] for index, name in enumerate(TRASBC_FIELDS)}
    result["stage"] = np.asarray(stages)
    result["registry"] = np.asarray(registry, dtype=np.int64)
    return result


def _qsr(path: Path) -> dict[str, np.ndarray]:
    rows, registry = [], []
    with path.open("rb") as stream:
        while magic := stream.read(16):
            if magic != QSR_MAGIC:
                raise ValueError("bad QSR magic")
            version, kt, kmm, krhs, count, bits = struct.unpack(
                "=6i", stream.read(24))
            if (version, count, bits) != (1, len(QSR_FIELDS), 64):
                raise ValueError("bad QSR header")
            rows.append(np.fromfile(stream, np.float64, count))
            registry.append((kt, kmm, krhs))
    expected = [
        (kt, 2, 3 if kt % 2 else 1) for kt in range(1, len(registry) + 1)
    ]
    if registry != expected:
        raise ValueError("QSR exact kt/Kmm/Krhs registry")
    values = np.asarray(rows)
    result = {name: values[:, index] for index, name in enumerate(QSR_FIELDS)}
    result["registry"] = np.asarray(registry, dtype=np.int64)
    return result


def _nemo_bilinear_months(
    chlorophyll: np.ndarray, source_indices: list[int], weights: list[np.float64]
) -> np.ndarray:
    """Replay fldread.F90:1475-1484's four ordered assignments."""
    months = []
    nx = chlorophyll.shape[2]
    for month in range(chlorophyll.shape[0]):
        value = np.float64(0.0)
        for source_index, weight in zip(source_indices, weights, strict=True):
            y, x = divmod(source_index - 1, nx)
            value = np.float64(
                value + np.float64(weight * np.float64(chlorophyll[month, y, x])))
        months.append(value)
    return np.asarray(months)


def _nemo_monthly_interp(months: np.ndarray, steps: int) -> np.ndarray:
    """Replay fldread.F90:181-186,225-228,890-917 for a 2018 hourly run."""
    month_beg_days = np.asarray(
        (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365),
        dtype=np.int64,
    )
    month_beg = month_beg_days * 86400
    anchors = (
        month_beg[:-1] // 2 + month_beg[1:] // 2
        + np.maximum(month_beg[:-1] % 2, month_beg[1:] % 2)
    )
    times = np.concatenate((
        np.asarray((-31 * 86400 // 2,), dtype=np.int64),
        anchors,
        np.asarray((365 * 86400 + 31 * 86400 // 2,), dtype=np.int64),
    ))
    values = np.concatenate((months[-1:], months, months[:1]))
    result = []
    for kt in range(1, steps + 1):
        isecsbc = np.int64((2 * kt - 1) * 1800)
        iaa = int(np.searchsorted(times, isecsbc, side="left"))
        ibb = iaa - 1
        ztinta = np.float64(
            np.float64(isecsbc - times[ibb])
            / np.float64(times[iaa] - times[ibb]))
        ztintb = np.float64(1.0 - ztinta)
        result.append(np.float64(
            np.float64(ztintb * values[ibb])
            + np.float64(ztinta * values[iaa])))
    return np.asarray(result)


def _expected_chlorophyll(root: Path, steps: int) -> np.ndarray:
    """Read immutable source plus NEMO-WEIGHTS output and replay fld_read."""
    with Dataset(root / "merged_ESACCI_BIOMER4V1R1_CHL_REG05.nc") as dataset:
        chlorophyll = np.asarray(dataset.variables["CHLA"][:])
    with Dataset(root / "weights_reg05_C1D_OMIP_L3_bilinear.nc") as dataset:
        source_indices = [
            int(np.asarray(dataset.variables[f"src{index:02d}"][:]).item())
            for index in range(1, 5)
        ]
        weights = [
            np.float64(np.asarray(dataset.variables[f"wgt{index:02d}"][:]).item())
            for index in range(1, 5)
        ]
    return _nemo_monthly_interp(
        _nemo_bilinear_months(chlorophyll, source_indices, weights), steps)


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
    missing = [name for name in REQUIRED_STREAMS if not (root / name).is_file()]
    if missing:
        raise GateError(f"missing required rung-3.6 stream(s): {missing}")
    set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
    card = build_c1d_omip_l3_coupled_card(oracle_root=root)
    config = card.exchange
    if (card.ice_cadence != config.nn_fsbc
            or card.ice_dt_seconds != card.ocean_dt_seconds * card.ice_cadence):
        raise ValueError("coupled card ocean/ice clock relation is inconsistent")
    instant, before, after, ssm_registry = _ssm(
        root / "oracle_rung36_ssm_frames.bin")
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
    if not np.array_equal(ice_steps, steps[::card.ice_cadence]):
        raise ValueError("ZDF cadence does not cross-link to ocean steps")
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
            ice_constants=NEMO_SI3_CONSTANTS_CONFIG, dt=card.ice_dt_seconds)
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
        if trasbc_path.exists():
            stage3_registry = trasbc["registry"][trasbc["stage"] == 3]
            if (not np.array_equal(qsr["registry"][:, 0], stage3_registry[:, 0])
                    or not np.array_equal(qsr["registry"][:, 1:],
                                          stage3_registry[:, (3, 4)])):
                raise ValueError("QSR does not cross-link to TRA stage-3 Kmm/Krhs")
        expected_chl = _expected_chlorophyll(root, len(qsr["chl"]))
        if plant == "chl_input":
            expected_chl = np.ones_like(expected_chl)
        rows.append(_summary(
            "POST_FLD_READ_CHL.chl", expected_chl, qsr["chl"]))
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
    bridge_diagnostics = None
    if fwb_path.exists():
        bridge_diagnostics = {
            "source": "pre-FWB icestp exchange versus POST_FWB operands",
            "max_abs_emp": float(np.max(np.abs(fields["emp"] - fwb["emp"]))),
            "max_abs_qns": float(np.max(np.abs(fields["qns"] - fwb["qns"]))),
        }
    card_emp = fwb["emp"].copy() if fwb_path.exists() else fields["emp"].copy()
    card_qns = fwb["qns"].copy() if fwb_path.exists() else fields["qns"].copy()
    if plant == "fwb_bridge":
        card_emp = fields["emp"].copy()
        card_qns = fields["qns"].copy()
    if plant == "freshwater_sign":
        card_emp = -card_emp
    freshwater, surface = nemo_si3_exchange_forcing(
        qsr=jax.numpy.asarray(fields["qsr"]),
        qns=jax.numpy.asarray(card_qns),
        emp=jax.numpy.asarray(card_emp), sfx=jax.numpy.asarray(fields["sfx"]),
        utau=jax.numpy.asarray(fields["utau"]),
        vtau=jax.numpy.asarray(fields["vtau"]),
        chl=jax.numpy.asarray(qsr["chl"]) if qsr_path.exists()
        else jax.numpy.ones_like(jax.numpy.asarray(fields["qsr"])),
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
        "freshwater": -fwb["emp"] if fwb_path.exists() else -fields["emp"],
        "q_net": fields["qsr"] + (fwb["qns"] if fwb_path.exists()
                                      else fields["qns"]),
        "sw_down": fields["qsr"], "tau_x": -fields["utau"],
        "tau_y": -fields["vtau"],
        "salt_flux": fields["sfx"] * constants.pss_to_mass_fraction,
        "rCdU_top": fields["rCdU_ice"], "snwice_fmass": fields["snwice_fmass"],
    }
    rows.extend(_summary(f"EXCHANGE_CARD.{name}", mapped[name], expected[name])
                for name in mapped)

    # Explicit producer-to-exchange bridges.  Ice-flux values refresh at
    # kt=1,5,... and are carried for four ocean steps; qns/emp are the sole
    # exception because sbc_fwb mutates them after each exchange record.
    def carry_four(values: np.ndarray) -> np.ndarray:
        return np.repeat(np.asarray(values), card.ice_cadence)[:len(steps)]

    bridge_expected = {
        name: np.asarray(update_tau[name])
        for name in ("rCdU_ice", "utau", "vtau", "taum")
    }
    bridge_expected.update({
        name: carry_four(update_post[name])
        for name in (
            "tn_ice", "alb_ice", "snwice_mass", "snwice_mass_b",
            "snwice_fmass", "sfx", "fr_i",
        )
    })
    for name in ("qns", "emp"):
        value = np.empty_like(fields[name])
        for index in range(len(value)):
            if index % card.ice_cadence == 0:
                value[index] = update_post[name][index // card.ice_cadence]
            else:
                value[index] = fwb[name][index - 1]
        bridge_expected[name] = value
    if plant == "producer_bridge":
        bridge_expected["utau"] = bridge_expected["utau"].copy()
        bridge_expected["utau"][0] += 1.0e-8
    for name, wanted in bridge_expected.items():
        rows.append(_summary(f"BRIDGE_EXCHANGE.{name}", fields[name], wanted))
    for name in ("qml_ice", "qcn_ice"):
        rows.append(_summary(
            f"BRIDGE_EXCHANGE.{name}", fields[name][::card.ice_cadence],
            update_pre[name]))
    first = next((row for row in rows if row["status"] == "DEBT"), None)
    promoted = {
        "tn_ice": "POST_UPDATE_FLX producer and four-step exchange carry scored",
        "alb_ice": "POST_UPDATE_FLX producer and four-step exchange carry scored",
        "qml_ice": "PRE_UPDATE_FLX ice_thd operand bridged at each refresh",
        "qcn_ice": "PRE_UPDATE_FLX ZDF operand bridged at each refresh",
        "rCdU_ice": "POST_UPDATE_TAU producer-to-exchange bridge scored every step",
        "snwice_mass": "POST_UPDATE_FLX producer and four-step exchange carry scored",
        "snwice_mass_b": "POST_UPDATE_FLX producer and four-step exchange carry scored",
        "snwice_fmass": "POST_UPDATE_FLX producer and four-step exchange carry scored",
        "utau": "POST_UPDATE_TAU producer-to-exchange bridge scored every step",
        "vtau": "POST_UPDATE_TAU producer-to-exchange bridge scored every step",
        "taum": "POST_UPDATE_TAU producer-to-exchange bridge scored every step",
        "qns": "POST_UPDATE_FLX/FWB cadence bridge and post-FWB card input scored",
        "emp": "POST_UPDATE_FLX/FWB cadence bridge and post-FWB card input scored",
        "sfx": "POST_UPDATE_FLX producer and four-step exchange carry scored",
        "fr_i": "POST_UPDATE_FLX producer and four-step exchange carry scored",
    }
    coverage = []
    for name in bulk.EXPECTED_EXCHANGE_FIELDS:
        status, reason = bulk.COVERAGE[name]
        if name in promoted:
            status, reason = "VERIFIED", promoted[name]
        if name == "sstfrz":
            status = "WAIVED"
            reason = (
                "allocated zero in this in-process C1D run; only the inactive "
                "external-coupler send path sets it (sbccpl.F90:2724-2726); "
                "the active ice-base eos_fzp operand is scored separately"
            )
        coverage.append({"name": name, "status": status, "reason": reason})
    if len(coverage) != 36 or len({row["name"] for row in coverage}) != 36:
        raise ValueError("exchange coverage register is not one-to-one")
    plant_target = {
        "ssm_sample": "POST_SSM.sst", "fzp_operand": "POST_FZP.t_bo",
        "update_heat": "POST_UPDATE_FLX.qns", "fwb_mass": "POST_FWB.emp",
        "fwb_immediate": "POST_FWB.emp",
        "trasbc_heat": "POST_TRA_SBC_RK3.temperature",
        "chl_input": "POST_FLD_READ_CHL.chl",
        "qsr_flux": "POST_TRA_QSR.temperature",
        "producer_bridge": "BRIDGE_EXCHANGE.utau",
        "fwb_bridge": "EXCHANGE_CARD.freshwater",
        "freshwater_sign": "EXCHANGE_CARD.freshwater",
    }.get(plant)
    unmeasured_boundaries = (
        "POST_SBC_STAGGER", "POST_ZDF_DRG_COEFF", "PRE_DYN_SPG_TS",
        "SSH_SUBSTEP", "POST_STP2D", "PRE_DYN_ZDF_SOLVE",
    )
    return {
        "verdict": "MEASURED_PREFIX_AT_BAR" if first is None else "DEBT",
        "scope_complete": False,
        "unmeasured_boundaries": unmeasured_boundaries,
        "bar": BAR,
        "backend": jax.default_backend(),
        "dtype": str(got_ssm.dtype),
        "precision_policy": {"mode": "fp64", "transcendentals": "libm"},
        "clock": {
            "ocean_dt_seconds": card.ocean_dt_seconds,
            "ice_dt_seconds": card.ice_dt_seconds,
            "ice_cadence": card.ice_cadence,
        },
        "time_level_registry": {
            "SSM": ssm_registry,
            "ZDF": {
                "records": int(ice_steps.size),
                "rule": "kt=1,5,9,... (nn_fsbc=4)",
                "first": int(ice_steps[0]), "last": int(ice_steps[-1]),
            },
            "UPDATE": {
                "rule": "FLX stages 0/1 at kt=1,5,...; TAU stage 2 every kt",
            },
            "TRA_SBC": {
                "records": int(trasbc["registry"].shape[0]),
                "rule": "exact alternating Kbb/Kmm/Krhs three-stage cycle",
                "first": trasbc["registry"][0].tolist(),
                "last": trasbc["registry"][-1].tolist(),
            },
            "QSR": {
                "records": int(qsr["registry"].shape[0]),
                "rule": "Kmm=2; Krhs=3 odd kt, 1 even kt; equals TRA stage 3",
                "first": qsr["registry"][0].tolist(),
                "last": qsr["registry"][-1].tolist(),
            },
        },
        "oracle_steps": int(after.shape[0]), "ice_steps": int(ice_steps.size),
        "rows": rows, "first_over_bar": first, "coverage": coverage,
        "bridge_diagnostics": bridge_diagnostics,
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
        "chl_input",
        "qsr_flux",
        "producer_bridge",
        "fwb_bridge",
        "freshwater_sign"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.root, args.plant)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.plant is not None:
        return 1 if report["plant_binding"]["red"] else 2
    return 0 if report["verdict"] == "MEASURED_PREFIX_AT_BAR" else 1


if __name__ == "__main__":
    raise SystemExit(main())
