from pathlib import Path
import sys


ROOT = Path(__file__).parents[3]
SCRIPTS = ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
sys.path.insert(0, str(SCRIPTS))
import nemo_orca2_si3_exact_input_gate as gate


def _output(**changes):
    values = gate.supported_selectors() | changes
    logical = lambda value: "T" if value else "F"
    return "\n".join((
        f"jpl = {values['n_categories']}",
        f"nlay_i = {values['n_ice_layers']}",
        f"nlay_s = {values['n_snow_layers']}",
        f"ln_cndi_P07 = {logical(values['conductivity'] == 'p07')}",
        f"nn_icesal = {values['salinity_scheme']}",
        f"rn_sinew = {values['new_ice_salinity_fraction']}",
        f"ln_drainage = {logical(values['drainage'])}",
        f"ln_flushing = {logical(values['flushing'])}",
        f"ln_pnd = {logical(values['ponds'])}",
        f"ln_icedA = {logical(values['lateral_melt'])}",
    ))


def test_supported_resolved_identity_is_admitted():
    resolved = gate.resolve_selectors(_output())
    assert all(row["status"] == "VERIFIED" for row in gate.selector_rows(resolved))


def test_one_selector_plant_fails_closed_at_its_row():
    resolved = gate.resolve_selectors(_output(n_categories=5))
    rows = gate.selector_rows(resolved)
    planted = [row for row in rows if row["status"] == "UNSUPPORTED"]
    assert planted == [{
        "selector": "n_categories",
        "oracle": 5,
        "implemented_identity": 1,
        "status": "UNSUPPORTED",
    }]
