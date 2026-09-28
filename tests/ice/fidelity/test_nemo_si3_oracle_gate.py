"""Direct and planted controls for the SI3 lane-3 oracle gate."""

from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pytest

GATE_PATH = (
    Path(__file__).parents[3] / "scripts/validate/ocean_fidelity/testcases/nemo_si3_oracle_gate.py"
)
SPEC = importlib.util.spec_from_file_location("nemo_si3_oracle_gate", GATE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _minimal_run(root: Path, *, nx: int = 59, ny: int = 59, nz: int = 2) -> None:
    with netCDF4.Dataset(root / "mesh_mask.nc", "w") as ds:
        ds.createDimension("t", 1)
        ds.createDimension("x", nx)
        ds.createDimension("y", ny)
        ds.createDimension("nav_lev", nz)
        ds.createVariable("e1t", "f8", ("t", "y", "x"))[:] = 4.0
    with netCDF4.Dataset(root / "CASE_00000040_restart_ice.nc", "w") as ds:
        ds.createDimension("x", 1)
        ds.createVariable("state", "f8", ("x",))[:] = 1.0
    (root / "output.namelist.dyn").write_text("&namrun\n nn_itend = 40\n/\n")
    (root / "output.namelist.ice").write_text(
        "&nampar\n nlay_i = 3\n nlay_s = 3\n/\n"
        "&namthd_sal\n nn_icesal = 4\n/\n"
        "&namthd_pnd\n ln_pnd = .true.\n/\n"
        "&namdia\n ln_icediachk = .true.\n/\n"
    )
    (root / "ocean.output").write_text("run complete\n")


INSTRUMENT = GATE_PATH.parent / "nemo502_MY_SRC/si3_l3/icestp.F90"


def test_frame_registry_is_complete_unique_and_sourced() -> None:
    assert len(gate.FRAME_REGISTRY) == 19
    assert len({row[0] for row in gate.FRAME_REGISTRY}) == len(gate.FRAME_REGISTRY)
    assert all(
        row[2] in {"STEP_ENTRY_CURRENT", "CARRIED_PREVIOUS_STEP", "CARRIED_PREVIOUS_BEFORE_LEVEL"}
        for row in gate.FRAME_REGISTRY
    )
    assert all("icestp.F90:154-171" in row[3] for row in gate.FRAME_REGISTRY)


def test_frame_registry_order_matches_the_committed_instrument() -> None:
    """The registry is what NAMES each array, so its ORDER is load-bearing.

    Reordering the instrument's WRITE statements without reordering the registry
    would silently mislabel every field and every other test would still pass.
    """
    lines = INSTRUMENT.read_text().splitlines()
    payload = [
        line.split(")", 1)[1] for line in lines if "WRITE(itraj)" in line and "cl_magic" not in line
    ]
    written = [name.strip() for line in payload for name in line.split(",")]
    assert written == [row[0] for row in gate.FRAME_REGISTRY]
    header = next(line for line in lines if "WRITE(itraj)" in line and "cl_magic" in line)
    assert header.rstrip().endswith(f", {len(gate.FRAME_REGISTRY)}")


def test_frame_round_trip_uses_fp64_and_fortran_order(tmp_path: Path) -> None:
    dims = {"jpi": 5, "jpj": 6, "jpl": 2, "nlay_i": 3, "nlay_s": 3}
    expected: dict[str, np.ndarray] = {}
    path = tmp_path / "oracle_ice_step_entry_kt00000001.bin"
    with path.open("wb") as fh:
        fh.write(gate.MAGIC.encode().ljust(16))
        fh.write(struct.pack("=9i", 1, 1, 5, 6, 2, 3, 3, 64, len(gate.FRAME_REGISTRY)))
        offset = 0
        for name, axes, _, _ in gate.FRAME_REGISTRY:
            shape = tuple(dims[axis] for axis in axes)
            values = np.arange(offset, offset + int(np.prod(shape)), dtype=np.float64).reshape(
                shape, order="F"
            )
            expected[name] = values
            values.ravel(order="F").tofile(fh)
            offset += values.size

    header, actual = gate.read_frame(path)
    assert header["storage_bits"] == 64
    for name in expected:
        assert actual[name].dtype == np.float64
        assert np.array_equal(actual[name], expected[name])

    _, selected = gate.read_frame(path, frozenset(("a_i", "u_ice")))
    assert set(selected) == {"a_i", "u_ice"}
    assert np.array_equal(selected["a_i"], expected["a_i"])
    with pytest.raises(gate.GateError, match="unknown requested frame fields"):
        gate.read_frame(path, frozenset(("not_a_registered_field",)))


def test_rung34_sishea_formula_and_mask_plant_bind() -> None:
    shape = (9, 9)
    metrics = {
        name: np.ones(shape, dtype=np.float64)
        for name in ("e1u", "e2u", "e1v", "e2v", "e1f", "e2f", "e1t", "e2t", "fmask")
    }
    u_ice = np.zeros(shape, dtype=np.float64)
    v_ice = np.zeros(shape, dtype=np.float64)
    area = np.ones(shape, dtype=np.float64)
    assert np.array_equal(gate._rung34_sishea(u_ice, v_ice, area, metrics), u_ice)

    # A single U-point perturbation exercises both the F-point shear and
    # T-point tension branches; removing the perturbation or either branch
    # makes this planted control fail.
    u_ice[4, 4] = 1.0
    perturbed = gate._rung34_sishea(u_ice, v_ice, area, metrics)
    assert float(np.max(perturbed)) > 0.0
    area.fill(0.0)
    assert np.array_equal(gate._rung34_sishea(u_ice, v_ice, area, metrics), np.zeros(shape))


def test_restart_contract_disposes_active_and_inactive_appendix_a_fields() -> None:
    adv = gate.restart_contract("3.1", 3, 3, 4, True)
    rhg = gate.restart_contract("3.3", 3, 3, 4, True)
    assert adv["stress1_i"]["status"] == "WAIVED"
    assert rhg["stress1_i"]["status"] == "VERIFIED"
    assert rhg["sxsi_l03"]["status"] == "VERIFIED"
    assert rhg["sxsal"]["status"] == "WAIVED"
    assert rhg["sxvl"]["status"] == "VERIFIED"
    assert rhg["cnd_ice"]["status"] == "WAIVED"
    assert rhg["t_s_l03"]["status"] == "WAIVED"


def test_rung34_contract_includes_layers_ponds_and_ephemeral_ridging_state() -> None:
    contract = gate.restart_contract("3.4", 10, 5, 4, True)
    assert contract["stress12_i"]["status"] == "VERIFIED"
    assert contract["sxe_l10"]["status"] == "VERIFIED"
    assert contract["sxc0_l05"]["status"] == "VERIFIED"
    assert contract["sxap"]["status"] == "VERIFIED"
    assert contract["closing_net"]["status"] == "WAIVED"
    assert "ephemeral" in contract["closing_net"]["reason"]
    assert contract["araft"]["status"] == "WAIVED"


def test_planted_unaccounted_mesh_array_goes_red(tmp_path: Path) -> None:
    _minimal_run(tmp_path)
    manifest = gate.disposition_template(tmp_path, "3.1")
    with pytest.raises(
        gate.GateError,
        match=r"missing=\['PLANTED_UNACCOUNTED_FILE_ARRAY'\].*extra=\[\]",
    ):
        gate.check_manifest(tmp_path, "3.1", manifest, plant_unaccounted=True)


def test_planted_geometry_field_perturbation_goes_red(tmp_path: Path) -> None:
    _minimal_run(tmp_path)
    with netCDF4.Dataset(tmp_path / "mesh_mask.nc", "a") as ds:
        for name in ("e1u", "e1v", "e1f", "e2t", "e2u", "e2v", "e2f"):
            ds.createVariable(name, "f8", ("t", "y", "x"))[:] = 4.0
        for name in ("ff_t", "ff_f"):
            ds.createVariable(name, "f8", ("t", "y", "x"))[:] = 0.0
        ds.createDimension("z", 1)
        for name in ("tmask", "umask", "vmask"):
            ds.createVariable(name, "i1", ("t", "z", "y", "x"))[:] = 1
    with pytest.raises(gate.GateError, match="e1t metric"):
        gate.geometry(tmp_path, "3.1", plant_field=True)


def test_manifest_contract_omission_and_status_downgrade_go_red(tmp_path: Path) -> None:
    _minimal_run(tmp_path)
    manifest = gate.disposition_template(tmp_path, "3.1")
    manifest["restart_contract"].pop("t_s_l01")
    with pytest.raises(gate.GateError, match="exhaustive Appendix-A contract"):
        gate.check_manifest(tmp_path, "3.1", manifest)

    # A VERIFIED row downgraded to WAIVED would SKIP its numeric/finite/fp64
    # check, so every disposition is regenerated rather than trusted.
    manifest = gate.disposition_template(tmp_path, "3.1")
    manifest["entries"]["mesh"]["e1t"] = {"status": "WAIVED", "reason": "reviewer waived this"}
    with pytest.raises(gate.GateError, match="mesh dispositions differ from the regenerated"):
        gate.check_manifest(tmp_path, "3.1", manifest)

    manifest = gate.disposition_template(tmp_path, "3.1")
    manifest["git_sha"] = "not-a-real-commit"
    with pytest.raises(gate.GateError, match="git_sha is not a 40-hex commit"):
        gate.check_manifest(tmp_path, "3.1", manifest)

    # ocean.output carries the conservation verdict, so it is hash-pinned too.
    manifest = gate.disposition_template(tmp_path, "3.1")
    (tmp_path / "ocean.output").write_text("run complete, edited\n")
    with pytest.raises(gate.GateError, match="ocean_output SHA256 mismatch"):
        gate.check_manifest(tmp_path, "3.1", manifest)
    (tmp_path / "ocean.output").write_text("run complete\n")

    template = gate.disposition_template(tmp_path, "3.1")
    assert {"VERIFIED", "WAIVED"} >= {
        item["status"]
        for namespace in ("mesh", "restart")
        for item in template["entries"][namespace].values()
    }


def test_maximum_trajectory_diagnostic_is_non_vacuous() -> None:
    equal = [
        {"phase": "step_entry", "kt": 1.0, "a_i_max": 0.9},
        {"phase": "step_entry", "kt": 2.0, "a_i_max": 0.9},
        {"phase": "post_step_final_restart", "kt": 2.0, "a_i_max": 0.9},
    ]
    report = gate.maximum_trajectory_diagnostic(equal)
    assert report["all_sampled_maxima_exactly_equal"] is True
    assert report["worst_abs_relative_excursion"] == 0.0

    intermediate = [dict(row) for row in equal]
    intermediate[1]["a_i_max"] = 1.0
    report = gate.maximum_trajectory_diagnostic(intermediate)
    assert report["all_sampled_maxima_exactly_equal"] is False
    assert report["worst_sample_phase"] == "step_entry"
    assert report["worst_sample_kt"] == 2.0
    assert report["worst_abs_relative_excursion"] > 0.1

    restart = [dict(row) for row in equal]
    restart[-1]["a_i_max"] = 0.8
    report = gate.maximum_trajectory_diagnostic(restart)
    assert report["all_sampled_maxima_exactly_equal"] is False
    assert report["worst_sample_phase"] == "post_step_final_restart"


def test_phenomenology_does_not_invent_a_maximum_preservation_band() -> None:
    shape3 = (9, 9, 1)
    final_shape3 = (5, 5, 1)
    final_shape2 = (5, 5)
    initial = {
        "a_i": np.ones(shape3),
        "v_i": np.ones(shape3),
        "u_ice": np.zeros((9, 9)),
    }
    final = {
        "a_i": np.ones(final_shape3),
        "v_i": np.ones(final_shape3),
        "u_ice": np.zeros(final_shape2),
    }
    maximum = gate.maximum_trajectory_diagnostic(
        [
            {"phase": "step_entry", "kt": 1.0, "a_i_max": 1.0},
            {"phase": "step_entry", "kt": 2.0, "a_i_max": 1.25},
            {"phase": "post_step_final_restart", "kt": 2.0, "a_i_max": 1.0},
        ]
    )
    verdict = gate.phenomenology("3.2", initial, final, maximum_trajectory=maximum)
    assert verdict["status"] == "REFUTE"
    claims = verdict["refuted_predicates"]
    assert not any("maximum-concentration" in row for row in claims)
    assert verdict["maximum_concentration_documentation_status"].startswith("UNMEASURED")
    assert verdict["maximum_concentration_step_boundary_diagnostic"][
        "worst_abs_relative_excursion"
    ] == 0.25


def test_unmeasured_maximum_conformance_cannot_go_green() -> None:
    initial = {
        "a_i": np.ones((9, 9, 1)),
        "v_i": np.ones((9, 9, 1)),
        "u_ice": np.zeros((9, 9)),
    }
    final = {
        "a_i": np.ones((5, 5, 1)),
        "v_i": np.full((5, 5, 1), 2.0),
        "u_ice": np.zeros((5, 5)),
    }
    maximum = gate.maximum_trajectory_diagnostic(
        [
            {"phase": "step_entry", "kt": 1.0, "a_i_max": 1.0},
            {"phase": "post_step_final_restart", "kt": 1.0, "a_i_max": 1.0},
        ]
    )
    verdict = gate.phenomenology("3.2", initial, final, maximum_trajectory=maximum)
    assert verdict["refuted_predicates"] == []
    assert verdict["h_i_overshoot"] == 1.0
    assert verdict["status"] == "UNMEASURED"


def test_free_drift_identity_control_goes_red() -> None:
    """Rung 3.3's bar is the free-drift identity, not merely "the ice moved"."""
    initial = {"a_i": np.ones((9, 9, 1)), "v_i": np.ones((9, 9, 1)), "u_ice": np.zeros((9, 9))}
    final = {
        "a_i": np.ones((5, 5, 1)),
        "v_i": np.full((5, 5, 1), 2.0),
        "u_ice": np.full((5, 5), 0.5033997477580665),
    }
    constants = {"u_free_drift_m_s": 0.5033997477580665, "relative_tolerance": 1.0e-2}
    assert gate.phenomenology("3.3", initial, final, free_drift=constants)["status"] == "CONFIRM"

    off = dict(constants, u_free_drift_m_s=0.6)
    verdict = gate.phenomenology("3.3", initial, final, free_drift=off)
    assert verdict["status"] == "REFUTE"
    assert any("free-drift identity" in row for row in verdict["refuted_predicates"])


def test_rung34_readme_phenomenology_cannot_be_promoted_without_a_band() -> None:
    initial = {
        "a_i": np.ones((9, 9, 1)),
        "v_i": np.ones((9, 9, 1)),
        "u_ice": np.zeros((9, 9)),
    }
    final = {
        "a_i": np.ones((5, 5, 1)),
        "v_i": np.full((5, 5, 1), 2.0),
        "u_ice": np.full((5, 5), 0.1),
        "v_ice": np.zeros((5, 5)),
    }
    shear = {
        "maximum_s-1": 1.0e-4,
        "classification": "MEASURED-UNCLASSIFIED: no numeric README band",
    }
    verdict = gate.phenomenology("3.4", initial, final, rung34_shear=shear)
    assert verdict["status"] == "MEASURED-UNCLASSIFIED"
    assert verdict["shear"] == shear
    assert verdict["refuted_predicates"] == []
    assert str(verdict["eap_angle_contrast"]).startswith("OUT-OF-SCOPE")


def test_rung34_missing_output_and_reconstruction_inputs_is_loud(tmp_path: Path) -> None:
    path = tmp_path / "CASE_6h_00010101_00010101_gr_0000.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("x", 2)
        dataset.createDimension("y", 2)
        dataset.createVariable("ssv_m", "f8", ("time", "y", "x"))[:] = 0.0
    with pytest.raises(gate.GateError, match="mesh_mask"):
        gate.rung34_shear_diagnostic(tmp_path, (slice(None), slice(None)))


def test_rung34_documented_shear_field_is_measured_when_present(tmp_path: Path) -> None:
    path = tmp_path / "CASE_6h_00010101_00010101_icemod.nc"
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        dataset.createDimension("x", 2)
        dataset.createDimension("y", 2)
        dataset.createVariable("sishea", "f8", ("time", "y", "x"))[:] = (
            (1.0, 2.0),
            (3.0, 4.0),
        )
    diagnostic = gate.rung34_shear_diagnostic(
        tmp_path, (slice(None), slice(None))
    )
    assert diagnostic["classification"].startswith("MEASURED-UNCLASSIFIED")
    assert diagnostic["maximum_s-1"] == 4.0


def test_native_conservation_violation_is_refuted(tmp_path: Path) -> None:
    _minimal_run(tmp_path)
    assert gate.conservation_diagnostics(tmp_path)["status"] == "CONFIRM"
    (tmp_path / "ocean.output").write_text("icedyn_adv : violation v_i < 0 = -1\n")
    verdict = gate.conservation_diagnostics(tmp_path)
    assert verdict["status"] == "REFUTE"
    assert verdict["violation_count"] == 1


def test_disabled_native_conservation_check_is_not_called_clean(tmp_path: Path) -> None:
    _minimal_run(tmp_path)
    ice_namelist = tmp_path / "output.namelist.ice"
    ice_namelist.write_text(
        ice_namelist.read_text().replace(
            "ln_icediachk = .true.", "ln_icediachk = .false."
        )
    )
    verdict = gate.conservation_diagnostics(tmp_path)
    assert verdict["status"] == "WAIVED-INACTIVE"
    assert verdict["enabled"] is False


RUN_ROOTS = {
    "3.1": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv1d/final"),
        Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/ICE_ADV1D_OMIP_L3"),
        "ice_adv1d_l3",
    ),
    "3.2": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d/final"),
        Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/ICE_ADV2D_OMIP_L3"),
        "ice_adv2d_l3",
    ),
    "3.3": (
        Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_adv2d_rhg/final"),
        Path("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/tests/ICE_ADV2D_RHG_OMIP_L3"),
        "ice_adv2d_rhg_l3",
    ),
}
MANIFEST_DIR = GATE_PATH.parent / "manifests"


def _cli(rung: str, *extra: str) -> subprocess.CompletedProcess[str]:
    run_dir, case_dir, manifest = RUN_ROOTS[rung]
    return subprocess.run(
        [
            sys.executable,
            str(GATE_PATH),
            "--run-dir",
            str(run_dir),
            "--case-dir",
            str(case_dir),
            "--rung",
            rung,
            "--manifest",
            str(MANIFEST_DIR / f"{manifest}.json"),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


PLANT_MESSAGE = {
    "--plant-unaccounted": "mesh coverage mismatch: missing=['PLANTED_UNACCOUNTED_FILE_ARRAY']",
    "--plant-field": "e1t metric",
}


@pytest.mark.parametrize("rung", sorted(RUN_ROOTS))
@pytest.mark.parametrize("control", sorted(PLANT_MESSAGE))
def test_cli_planted_controls_exit_nonzero(rung: str, control: str) -> None:
    """End-to-end: each preregistered plant must make the shipped CLI exit red.

    Rungs 3.1 and 3.2 already exit 1 unplanted, so a bare returncode assertion
    would pass on four of the six arms even if the plants were no-ops.  The
    plant's own message is what discriminates.
    """
    run_dir, case_dir, _ = RUN_ROOTS[rung]
    if not run_dir.is_dir() or not case_dir.is_dir():
        pytest.skip(f"oracle run root absent: {run_dir}")
    result = _cli(rung, control)
    assert result.returncode != 0
    assert PLANT_MESSAGE[control] in result.stderr, result.stderr[-400:]


def test_cli_rejects_a_run_whose_input_deck_is_not_the_committed_one() -> None:
    """The committed input decks are bound to the run, not just documentation."""
    run_dir, case_dir, _ = RUN_ROOTS["3.1"]
    if not run_dir.is_dir() or not case_dir.is_dir():
        pytest.skip(f"oracle run root absent: {run_dir}")
    assert gate.input_namelists(run_dir, "3.1")
    with pytest.raises(gate.GateError, match="differs from committed ice_adv2d_l3_namelist_cfg"):
        gate.input_namelists(run_dir, "3.2")


def test_rung34_binds_its_deliberately_different_deck_stems() -> None:
    run_dir = Path("/data/abyssal/dbalwada/nemo-testcases-l3/ice_rheo/final")
    if not run_dir.is_dir():
        pytest.skip(f"oracle run root absent: {run_dir}")
    digests = gate.input_namelists(run_dir, "3.4")
    assert set(digests) == {"namelist_cfg", "namelist_ice_cfg"}
    assert gate.CONFIG_STEM_OVERRIDE[("3.4", "namelist_ice_cfg")] == "ice_rheo_l3"
    assert gate.final_restart_names("3.4") == ("a_i", "v_i", "u_ice", "v_ice")


def test_cli_clean_rung_is_green_and_refuted_rungs_are_red() -> None:
    """The controls only prove anything if the unplanted arms are not always red.

    Rungs 3.1 and 3.2 are expected DEBT because of MEASURED oracle behaviour
    (19 native heat-conservation violations; no positive thickness overshoot).
    If SI3 ever stops producing those, this test goes red -- that is a signal to
    re-read section 4 of the receipt, not a regression in this lane.
    """
    for rung, expected_green in (("3.3", True), ("3.1", False), ("3.2", False)):
        run_dir, case_dir, _ = RUN_ROOTS[rung]
        if not run_dir.is_dir() or not case_dir.is_dir():
            pytest.skip(f"oracle run root absent: {run_dir}")
        result = _cli(rung)
        assert (result.returncode == 0) is expected_green, result.stderr[-400:]
        payload = json.loads(result.stdout)
        assert payload["status"] == ("VERIFIED" if expected_green else "DEBT")


def test_cli_rung_3_1_phenomenology_confirms_on_the_wet_window() -> None:
    """Pins the land-rim fix that this rung's phenomenology verdict rests on.

    ICE_ADV1D's one-cell land rim is identically zero, so a whole-array
    reduction reports a y-spread of ~2.687 for `a_i` and refutes the documented
    y-homogeneity.  Over the wet window it is exactly zero.  Without this
    assertion, reverting `wet_window()` leaves the whole suite green.
    """
    run_dir, case_dir, _ = RUN_ROOTS["3.1"]
    if not run_dir.is_dir() or not case_dir.is_dir():
        pytest.skip(f"oracle run root absent: {run_dir}")
    report = json.loads(_cli("3.1").stdout)["trajectory"]["phenomenology"]
    assert report["status"] == "CONFIRM", report["refuted_predicates"]
    assert report["y_homogeneity_max_abs"] == 0.0
    assert report["refuted_predicates"] == []
