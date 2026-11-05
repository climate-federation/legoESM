"""Direct controls for every coupled C-ladder acquisition artifact."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.ice.si3_finalize import SI3StressInputs, si3_ice_update_tau
from legoesm.ice.si3_orca2_state import SI3ORCA2StateConfig
from legoesm.ocean.forcing.nemo_surface import NemoSurfaceMeans, nemo_sbc_ssm_step

REPO = Path(__file__).resolve().parents[3]
ACQ = REPO / "scripts/validate/ocean_fidelity/testcases/nemo_coupled_c_ladder_acquisition"
REAL = Path(
    "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo"
)


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


preflight = _module("c_ladder_preflight", ACQ / "preflight.py")
deck = _module("c_ladder_deck", ACQ / "deck.py")
scorer = _module("c_ladder_scorer", ACQ / "scorer.py")
link_objects = _module("c_ladder_link_objects", ACQ / "link_objects.py")


def test_real_candidates_and_live_coupled_call_order() -> None:
    report = preflight.source_contract(preflight.NEMO_DEFAULT)
    assert report["status"] == "PASS_COUPLED_C_SOURCE_CONTRACT"
    coupled = report["candidates"][preflight.COUPLED]
    assert coupled["driver"] == "stp_RK3"
    assert coupled["key_si3"] and not coupled["key_top"]
    assert coupled["nn_fsbc"] == 2 and coupled["nn_ice"] == 2
    assert report["call_order"]["ice_stp"][-1] == "ice_update_tau-every-ocean-step"


@pytest.mark.parametrize("plant", ("drop-sbc-ssm", "swap-flux-update", "pretend-mlf"))
def test_source_contract_plants_fail(plant: str) -> None:
    with pytest.raises(preflight.ContractError):
        preflight.source_contract(preflight.NEMO_DEFAULT, plant)


def test_surface_mean_both_parities_and_numeric_plant(tmp_path: Path) -> None:
    instant = np.arange(6, dtype=np.float64).reshape(2, 3) + 1.0
    zero = jnp.zeros_like(jnp.asarray(instant))
    prior = NemoSurfaceMeans(*(zero for _ in range(7)))
    first = nemo_sbc_ssm_step(
        prior,
        kt=1,
        nit000=1,
        forcing_stride=2,
        restart_means_present=False,
        use_conservative_temperature=False,
        remove_inverse_barometer=False,
        u_before=instant,
        v_before=instant + 1,
        temperature_now=instant + 2,
        salinity_now=instant + 3,
        ssh_now=instant + 4,
        ssh_inverse_barometer=zero,
        e3t_reference=instant + 5,
        r3t_now=zero,
        surface_mask=jnp.ones_like(zero),
        shortwave_fraction_top=instant + 6,
    )
    np.testing.assert_array_equal(np.asarray(first.ssu).view(np.uint64), instant.view(np.uint64))
    second = nemo_sbc_ssm_step(
        first,
        kt=2,
        nit000=1,
        forcing_stride=2,
        restart_means_present=False,
        use_conservative_temperature=False,
        remove_inverse_barometer=False,
        u_before=instant * 2,
        v_before=instant * 2,
        temperature_now=instant * 2,
        salinity_now=instant * 2,
        ssh_now=instant * 2,
        ssh_inverse_barometer=zero,
        e3t_reference=instant * 2,
        r3t_now=zero,
        surface_mask=jnp.ones_like(zero),
        shortwave_fraction_top=instant * 2,
    )
    np.testing.assert_array_equal(
        np.asarray(second.ssu).view(np.uint64), (instant * 2).view(np.uint64)
    )
    with pytest.raises(ValueError, match="not 0 ULP"):
        scorer.self_test("numeric", tmp_path)


def test_tau_cadence_is_scorer_only_and_preserves_production_api(tmp_path: Path) -> None:
    full = jnp.ones((6, 6), dtype=jnp.float64)
    interior = jnp.ones((2, 2), dtype=jnp.float64)
    inputs = SI3StressInputs(
        full,
        full,
        full,
        full,
        full * 2,
        full * 2,
        full,
        full,
        full,
        full * 0.5,
        full * 3,
        full * 4,
        full * 7,
        interior * 8,
        interior * 9,
        interior * 10,
    )
    cfg = SI3ORCA2StateConfig.for_orca2_sas_ice(nn_hls=2, nn_fsbc=2)
    assert "update_cadence" not in inspect.signature(si3_ice_update_tau).parameters
    with pytest.raises(ValueError, match="nn_fsbc=1"):
        si3_ice_update_tau(inputs, config=cfg, ln_drgice_imp=False)
    off = scorer._nemo_cadenced_tau_step(inputs, config=cfg, ln_drgice_imp=False, update_now=False)
    np.testing.assert_array_equal(off.tmod_io, inputs.tmod_io)
    np.testing.assert_array_equal(off.taum, inputs.taum)
    np.testing.assert_array_equal(off.utau_oce, inputs.utau_oce)
    on = scorer._nemo_cadenced_tau_step(inputs, config=cfg, ln_drgice_imp=False, update_now=True)
    assert not np.array_equal(on.tmod_io, inputs.tmod_io)
    with pytest.raises(ValueError, match="parity"):
        scorer.self_test("wrong-parity", tmp_path)


def test_schema_overlays_writer_prereg_and_truncation(tmp_path: Path) -> None:
    schema = json.loads((ACQ / "record_schema.json").read_text())
    overlays = json.loads((ACQ / "namelist_overlays.json").read_text())
    assert schema["magic"] == "CLAD0001" and set(schema["kinds"]) == set("123456")
    assert overlays["C4"]["nn_stock"] == 8 and overlays["C4"]["expected_restart_shards"] == 1460
    writer = (ACQ / "c_ladder_writer.F90").read_text()
    assert "PUBLIC :: c_record_ssm, c_record_sbc, c_record_step" in writer
    assert "INTENT(out)" not in "\n".join(
        line for line in writer.splitlines() if "c_record_" in line
    )
    prereg = (REPO / "docs/ocean/fidelity/PREREG_nemo_coupled_c_ladder_acquisition.md").read_text()
    assert "NOT_YET_SCORABLE_NO_COMPOSED_LEGOESM_COUPLED_STEP" in prereg
    assert "CONFIRM" in prereg and "REFUTE" in prereg
    with pytest.raises(ValueError, match="truncated"):
        scorer.self_test("truncation", tmp_path)


def test_scorer_main_reads_c0_from_staged_rec_c3(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    def fake_c0(path, **_kwargs):
        seen["c0"] = path
        return {"status": "PASS"}

    monkeypatch.setattr(scorer, "score_c0", fake_c0)
    monkeypatch.setattr(scorer, "score_c1", lambda *_args, **_kwargs: {"status": "PASS"})
    monkeypatch.setattr(
        sys,
        "argv",
        ["scorer.py", "--evidence", str(tmp_path), "--ppsrc", str(tmp_path / "ppsrc")],
    )
    assert scorer.main() == 0
    assert seen["c0"] == tmp_path / "REC_C3/c_ladder_records"


@pytest.mark.parametrize("unit", ("stprk3", "sbcssm", "sbcmod", "icestp", "iceupdate"))
def test_each_observer_patch_applies_to_real_ppsrc(tmp_path: Path, unit: str) -> None:
    target = tmp_path / f"{unit}.f90"
    target.write_bytes((REAL / f"{unit}.f90").read_bytes())
    subprocess.run(
        ["patch", "--batch", "--fuzz=0", str(target), str(ACQ / f"{unit}_c.patch")],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    calls = [
        line
        for line in target.read_text().splitlines()
        if "CALL c_record_" in line or "CALL o8o16_" in line
    ]
    assert calls and all("=" not in line for line in calls)


def test_exact_overlays_and_shell_control_flow(tmp_path: Path) -> None:
    source = Path(deck.OVERLAYS["source_deck"])
    text = (source / "namelist_cfg").read_text()
    for rung, expected in (("C2", (2, 2)), ("C3", (100, 100)), ("C4", (2920, 8))):
        rendered = deck._replace(
            deck._replace(text, "nn_itend", expected[0]), "nn_stock", expected[1]
        )
        assert deck._last_int(rendered, "nn_itend") == expected[0]
        assert deck._last_int(rendered, "nn_stock") == expected[1]
    for shell in (ACQ / "run.sh", ACQ / "c_profile.sh"):
        subprocess.run(["bash", "-n", str(shell)], check=True, timeout=30)
    profile = (ACQ / "c_profile.sh").read_text()
    assert "C_LADDER_OPERATOR_ACK" in profile
    assert "STUB_REACHED_EXACT_NEMO_START" in profile
    assert "c_legacy_observer_gate" in profile
    assert "for unit in $C_OBSERVER_UNITS" in profile
    assert "mpirun -np 2" in profile and "makenemo" not in profile
    objects = link_objects.active_objects(
        Path(
            "/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/Makefile"
        )
    )
    assert "nemo.o" in objects and "stprk3.o" in objects and "trcopt.o" not in objects


def test_real_run_refuses_noncanonical_evidence_path(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["TMPDIR"] = str(tmp_path)
    proc = subprocess.run(
        [str(ACQ / "run.sh"), "--run", str(tmp_path / "evidence")],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 64
    assert "evidence must be under /data/abyssal" in proc.stderr
    launcher = (ACQ / "run.sh").read_text()
    assert "phase3/si3_rounds/round30" in launcher
    assert "scratch_root/c_ladder_acquisitions" not in launcher


def test_bounded_master_preflight_exercises_exact_run_stub(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "TMPDIR": str(tmp_path),
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    proc = subprocess.run(
        [str(ACQ / "run.sh"), "--preflight-only"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert "PREFLIGHT_OK: exact --run stub reached C2 REF NEMO start" in proc.stdout
    assert (
        "PASS_C_MASTER_PLANTS_SOURCE_NUMERIC_PASSIVITY_TRUNCATION_PARITY_DISK_RAM_RETENTION"
        in proc.stdout
    )
