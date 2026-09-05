import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
sys.path.insert(0, str(SCRIPTS))
import nemo_orca2_si3_exact_input_gate as gate  # noqa: E402


def _output(**changes):
    values = gate.supported_selectors() | changes

    def logical(value):
        return "T" if value else "F"

    return "\n".join(
        (
            f"jpl = {values['n_categories']}",
            f"nlay_i = {values['n_ice_layers']}",
            f"nlay_s = {values['n_snow_layers']}",
            f"ln_cndi_P07 = {logical(values['conductivity'] == 'p07')}",
            f"nn_icesal = {values['salinity_scheme']}",
            f"rn_sinew = {values['new_ice_salinity_fraction']}",
            f"ln_drainage = {logical(values['drainage'])}",
            f"ln_flushing = {logical(values['flushing'])}",
            f"ln_pnd = {logical(values['ponds'])}",
            f"ln_pnd_LEV = {logical(values['pond_level'])}",
            f"ln_icedA = {logical(values['lateral_melt'])}",
            f"ln_virtual_itd = {logical(values['virtual_itd'])}",
            f"ln_icedH = {logical(values['thermodynamic_growth'])}",
            f"ln_icedO = {logical(values['open_water_growth'])}",
            f"ln_landfast_L16 = {logical(values['landfast_l16'])}",
            f"ln_rhg_EVP = {logical(values['evp'])}",
            f"ln_aEVP = {logical(values['aevp'])}",
            f"ln_str_H79 = {logical(values['strength_h79'])}",
            f"ln_ridging = {logical(values['ridging'])}",
            f"ln_rafting = {logical(values['rafting'])}",
            f"ln_adv_Pra = {logical(values['prather'])}",
        )
    )


def test_supported_resolved_identity_is_admitted():
    resolved = gate.resolve_selectors(_output())
    assert all(row["status"] == "VERIFIED" for row in gate.selector_rows(resolved))


def test_one_selector_plant_fails_closed_at_its_row():
    resolved = gate.resolve_selectors(_output(n_categories=5))
    rows = gate.selector_rows(resolved)
    planted = [row for row in rows if row["status"] == "UNSUPPORTED"]
    assert planted == [
        {
            "selector": "n_categories",
            "oracle": 5,
            "implemented_identity": 1,
            "status": "UNSUPPORTED",
        }
    ]


def test_synthetic_schema_and_every_registered_row_plant_bind():
    result = gate._self_test()
    assert result["status"] == "PASS"
    assert result["valid_synthetic_frames"] == 6
    assert all(result["schema_plants"].values())
    assert set(result["schema_plants"]) == {
        "magic",
        "version",
        "field_count",
        "payload_count",
        "extent",
        "step",
        "truncated",
        "trailing",
        "frame_sequence",
        "selector",
        "cadence",
    }
    assert result["row_plants"] == {field: True for field in gate.COMMON_STATE_FIELDS}


def test_parser_uses_self_described_state_and_flux_extents(tmp_path):
    path = tmp_path / "oracle_orca1ice_thd_kt00000001_f0.bin"
    path.write_bytes(gate._synthetic_raw(kt=1, frame=0))
    frame = gate._parse_l4_thd(path)
    assert frame.fields["a_i"].shape == (7, 8, 1, 1)
    assert frame.fields["e_i"].shape == (7, 8, 3, 1)
    assert frame.fields["qns_ice"].shape == (3, 4, 1, 1)
    assert frame.fields["qprec_ice"].shape == (3, 4, 1, 1)


def test_score_reports_bits_and_row_scale_ulp_and_plant():
    values = np.asarray([1.0, 2.0], dtype=np.float64)
    clean = gate._score_row(values, values, name="clean")
    planted = gate._score_row(values, values, name="plant", plant=True)
    assert clean["bit_identical"] == 2
    assert clean["non_bit"] == 0
    assert planted["bit_identical"] == 1
    assert planted["non_bit"] == 1
    assert planted["max_row_scale_ulp"] > 0.0


def test_cadence_mismatch_is_a_named_gate_error():
    with pytest.raises(gate.GateError, match="cadence mismatch"):
        gate._validate_cadence(4, 21600.0)


def test_resolved_ice_timestep_uses_nemos_final_numeric_value():
    line = "ice timestep rDt_ice = nn_fsbc*rn_Dt =    21600.000000000000"
    assert gate._resolved_numeric_tail(line, "rDt_ice") == 21600.0
