"""Direct controls for the decision-independent coupled C-ladder foundation."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
ACQ = REPO / "scripts/validate/ocean_fidelity/testcases/nemo_coupled_c_ladder_acquisition"
SPEC = importlib.util.spec_from_file_location("c_ladder_preflight", ACQ / "preflight.py")
assert SPEC and SPEC.loader
preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)


def test_real_candidates_and_live_coupled_call_order() -> None:
    report = preflight.source_contract(preflight.NEMO_DEFAULT)
    assert report["status"] == "PASS_COUPLED_C_SOURCE_CONTRACT"
    assert report["nemo_run"] is False
    coupled = report["candidates"][preflight.COUPLED]
    standalone = report["candidates"][preflight.STANDALONE]
    assert coupled["driver"] == "stp_RK3"
    assert coupled["key_si3"] and not coupled["key_top"]
    assert coupled["nn_fsbc"] == 2 and coupled["nn_ice"] == 2
    assert standalone["key_si3"] and standalone["nn_ice"] == 2
    assert report["facts"]["standalone_cfg_currently_has_parent_si3_profile"]
    assert report["call_order"]["ice_stp"][-1] == "ice_update_tau-every-ocean-step"
    assert report["facts"]["sbc_ice_si3_exists"] is False


@pytest.mark.parametrize("plant", ("drop-sbc-ssm", "swap-flux-update", "pretend-mlf"))
def test_source_contract_plants_fail(plant: str) -> None:
    with pytest.raises(preflight.ContractError):
        preflight.source_contract(preflight.NEMO_DEFAULT, plant)


def test_shell_is_preflight_only_and_delegates_the_admitted_framework() -> None:
    text = (ACQ / "run.sh").read_text()
    assert "--run" not in text and "makenemo" not in text and "mpirun" not in text
    assert '"$master" --preflight-only' in text
    assert "PASS_REUSED_REAL_PPSRC_OBSERVER_AND_STUB_NEMO_PREFLIGHT" in text
    assert "timeout 1500" in text
    proc = subprocess.run(
        ["bash", "-n", str(ACQ / "run.sh")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0
