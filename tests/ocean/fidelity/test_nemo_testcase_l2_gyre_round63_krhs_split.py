from __future__ import annotations

import hashlib
import importlib.util
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
    / "nemo_testcase_l2_gyre_round54_tracer_decomposition.py"
)
R64_RUN = (
    ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
    / "nemo_testcase_l2_gyre_round64_krhs_split" / "run.sh"
)


def _module():
    spec = importlib.util.spec_from_file_location("round63_krhs_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _put_field(payload: bytearray, name: str, value) -> None:
    array = np.asarray(value, dtype="=f8")
    payload.extend(name.ljust(16).encode("ascii"))
    if array.ndim == 0:
        shape = (1, 1, 1)
    elif array.ndim == 2:
        shape = (*array.shape, 1)
    else:
        assert array.ndim == 3
        shape = array.shape
    payload.extend(struct.pack("=4i", array.ndim, *shape))
    payload.extend(np.asfortranarray(array).tobytes(order="F"))


def _records(tmp_path: Path):
    module = _module()
    shape = (32, 22, 30)
    ones = np.ones(shape, dtype=np.float64)
    tmask = ones.copy()
    r3b = np.full(shape[:2], 0.01)
    r3m = np.full(shape[:2], 0.02)
    base = np.full(shape, 2.0)
    e3b = base * (1.0 + r3b[..., None] * tmask)
    e3m = base * (1.0 + r3m[..., None] * tmask)
    after_t = np.full(shape, 3.0e-7)
    after_s = np.full(shape, -2.0e-8)
    t_kbb = np.full(shape, 10.0)
    s_kbb = np.full(shape, 35.0)
    p2dt = np.float64(4800.0)
    content_t = e3b * t_kbb + p2dt * e3m * after_t
    content_s = e3b * s_kbb + p2dt * e3m * after_s
    krhs_values = {
        "p2dt": p2dt,
        "krhs_zero_T": np.zeros(shape), "krhs_zero_S": np.zeros(shape),
        "adv_up1_T": after_t / 2, "adv_up1_S": after_s / 2,
        "after_adv_T": after_t, "after_adv_S": after_s,
        "after_sbc_T": after_t, "after_sbc_S": after_s,
        "after_qsr_T": after_t, "after_qsr_S": after_s,
        "after_ldf_T": after_t, "after_ldf_S": after_s,
        "T_Kbb": t_kbb, "S_Kbb": s_kbb,
        "e3t_Kbb": e3b, "e3t_Kmm": e3m,
        "r3t_Kbb": r3b, "r3t_Kmm": r3m, "e3t_3d": base,
        "tmask": tmask, "content_T": content_t, "content_S": content_s,
    }
    krhs_payload = bytearray(b"NEMO_L2_R63KRS1 ")
    krhs_payload.extend(struct.pack(
        "=13i", 1, 2, 3, 1, 2, 3, 32, 22, 30, 1, 1, 64,
        len(module.R63_KRHS_FIELDS)))
    for name in module.R63_KRHS_FIELDS:
        _put_field(krhs_payload, name, krhs_values[name])

    entry = np.full(shape, 1.0e-4)
    shear = np.full(shape, 2.0e-7)
    avt = np.full(shape, 1.0e-5)
    rn2 = np.full(shape, 3.0e-4)
    dissl = np.full(shape, 4.0e-3)
    zfact3 = np.float64(0.35)
    rn_dt = np.float64(14400.0)
    strat = avt * rn2
    diss = zfact3 * dissl * entry
    post = entry + rn_dt * ((shear - strat) + diss) * ones
    tke_values = {
        "rn_Dt": rn_dt, "zfact3": zfact3, "en_rhs_entry": entry,
        "shear": shear, "avt": avt, "rn2": rn2, "dissl": dissl,
        "strat_product": strat, "diss_product": diss, "wmask": ones,
        "en_rhs_post": post,
    }
    tke_payload = bytearray(b"NEMO_L2_R63TKR1 ")
    tke_payload.extend(struct.pack(
        "=9i", 1, 2, 32, 22, 30, 1, 1, 64,
        len(module.R63_TKE_FIELDS)))
    for name in module.R63_TKE_FIELDS:
        _put_field(tke_payload, name, tke_values[name])

    commit = "a" * 40
    paths = {}
    for name, payload in (("krhs", krhs_payload), ("tke", tke_payload)):
        record = tmp_path / f"{name}.bin"
        record.write_bytes(payload)
        stamp = tmp_path / f"{name}.stamp"
        stamp.write_text(
            f"{hashlib.sha256(payload).hexdigest()} {commit} {record.name}\n")
        paths[name] = (record, stamp)
    producer = tmp_path / "producer_commit.txt"
    producer.write_text(commit + "\n")
    resolved = tmp_path / "ocean.output"
    resolved.write_text(
        "number of the last time step nn_itend = 2\n"
        "Assimilation cycle nn_no = 0\n"
        "ln_tile = F\nln_traqsr = T\nln_bdy = F\nln_isfcav = F\n"
        "ln_traadv_fct = T\nnn_fct_h = 2\nnn_fct_v = 2\n"
        "nn_fct_imp = 1\nln_zad_Aimp = F\nln_traldf_msc = F\n"
        "ln_trabbc = F\nln_trabbl = F\nln_tradmp = F\n"
        "ln_zdfmfc = F\nln_zdfosm = F\nln_zdfnpc = F\n")
    return module, paths, producer, resolved, commit


def test_round63_both_records_calibrate_bit_exact(tmp_path):
    module, paths, producer, resolved, commit = _records(tmp_path)
    report = module.r63_calibrate(
        krhs_record=paths["krhs"][0], tke_record=paths["tke"][0],
        krhs_stamp=paths["krhs"][1], tke_stamp=paths["tke"][1],
        expect_commit=commit, producer_commit=producer,
        resolved_output=resolved)
    assert report["status"] == "PASS"
    assert all(
        count == 0
        for family in report["calibration"].values()
        for count in family.values()
    )


def test_provisional_boundary_walk_can_fail_on_one_consumed_value():
    module = _module()
    shape = (4, 3, 2)  # record order i,j,k; model order j,i,k
    record_shape = shape
    tmask = np.ones(record_shape, dtype=np.float64)
    arrays = {
        "p2dt": np.float64(4.0),
        "e3t_Kbb": np.full(record_shape, 2.0),
        "e3t_Kmm": np.full(record_shape, 3.0),
        "T_Kbb": np.full(record_shape, 10.0),
        "S_Kbb": np.full(record_shape, 35.0),
        "krhs_zero_T": np.zeros(record_shape),
        "krhs_zero_S": np.zeros(record_shape),
        "after_adv_T": np.full(record_shape, 2.0e-6),
        "after_adv_S": np.full(record_shape, -3.0e-7),
    }
    def transpose(value):
        return np.ascontiguousarray(value.transpose(1, 0, 2))
    live = {
        tracer: transpose(
            arrays["e3t_Kbb"] * arrays[f"{tracer}_Kbb"]
            + arrays["p2dt"] * arrays["e3t_Kmm"]
            * arrays[f"after_adv_{tracer}"])
        for tracer in ("T", "S")
    }
    wet = {name: transpose(tmask).astype(bool) for name in ("T", "S")}
    exact = module._provisional_krhs_boundary_rows(live, arrays, wet)
    assert all(row["after_complete_fct_advection_content"]["cells_unequal"] == 0
               for row in exact.values())

    planted = {name: value.copy() for name, value in live.items()}
    planted["T"][0, 0, 0] = np.nextafter(
        planted["T"][0, 0, 0], np.float64(np.inf))
    detected = module._provisional_krhs_boundary_rows(planted, arrays, wet)
    assert detected["T"]["after_complete_fct_advection_content"][
        "cells_unequal"] == 1


def test_round64_runner_preserves_the_ten_step_source_horizon():
    run = R64_RUN.read_text()
    assert "TARGET_CFG=GYRE_OMIP_L2_P3_SM_R64KRHS" in run
    assert "round64/oracle_krhs_split" in run
    assert "--expect-itend 10" in run
    assert "nn_itend *= *10" in run
    assert "sed -i" not in run
    assert "cmp \"$SOURCE_ROOT/EXP00/namelist_cfg\"" in run


@pytest.mark.parametrize(
    "plant", ["stamp", "krhs-ulp", "tke-ulp",
              "krhs-truncation", "tke-truncation"])
def test_round63_plants_are_nonzero(tmp_path, plant):
    module, paths, producer, resolved, commit = _records(tmp_path)
    argv = [
        "--mode", "krhs-calibrate",
        "--krhs-record", str(paths["krhs"][0]),
        "--tke-rhs-record", str(paths["tke"][0]),
        "--krhs-stamp", str(paths["krhs"][1]),
        "--tke-rhs-stamp", str(paths["tke"][1]),
        "--producer-commit", str(producer),
        "--resolved-output", str(resolved),
        "--expect-commit", commit,
        "--plant", plant,
    ]
    assert module.main(argv) != 0
