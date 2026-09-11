"""Direct tests for the GYRE day-30 OWNER harness.

Every assertion is paired with a synthetic violation that must FAIL, so none
of them can pass vacuously.  The harness is loaded by path because it is a
script, not an installed module.

The forcing-gate and switch-trace modes need the NEMO record on /data, so the
tests that touch them are skipped when it is absent; the STATEMENT-level
properties they rest on -- that the literal transcription's new arguments
change nothing at their defaults, and that each plant moves a number -- are
tested unconditionally, because those are the parts that can rot.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
HARNESS = (ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
           / "nemo_testcase_l2_gyre_year_owners.py")
RUN_SH = (ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
          / "nemo_testcase_l2_gyre_earlydays" / "run.sh")
NEMO_RECORD = Path("/data/abyssal/dbalwada/nemo-testcases-l2/phase3/"
                   "year_fromrest/nemo_seed0")


@pytest.fixture(scope="module")
def harness():
    assert HARNESS.is_file(), HARNESS
    spec = importlib.util.spec_from_file_location("gyre_year_owners", HARNESS)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("gyre_year_owners", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def literal(harness):
    return harness._round16()._literal_sbc


def test_self_check_passes_as_a_subprocess():
    """The harness's own self-check is the gate; run it the way CI would."""
    result = subprocess.run([sys.executable, str(HARNESS), "--self-check"],
                            capture_output=True, text=True, cwd=str(ROOT))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "self-check: all checks passed" in result.stdout


def _toy():
    lat = np.array([[18.0, 30.0, 42.0]])
    wet = np.ones_like(lat, dtype=bool)
    ct = np.array([[22.0, 18.0, 8.0]])
    pt = ct - 0.1
    return lat, wet, ct, pt


def test_literal_sbc_default_is_byte_unchanged(literal):
    """Round 16's own numbers must not move because the owner round needed
    an argument.  Byte equality, not a tolerance."""
    lat, wet, ct, pt = _toy()
    base, _ = literal(lat, wet, ct, pt)
    again, _ = literal(lat, wet, ct, pt, kt=1, nyear=1, qsr_pi=None)
    for field in ("qsr", "qns", "emp", "utau", "vtau"):
        assert np.array_equal(np.asarray(base[field]).view(np.uint64),
                              np.asarray(again[field]).view(np.uint64)), field


def test_every_forcing_field_moves_with_the_clock(literal):
    """Non-vacuity of the PHASE plant: if a field did not move between two
    values of ztime, a wrong phase could not be seen on it."""
    lat, wet, ct, pt = _toy()
    base, _ = literal(lat, wet, ct, pt, kt=1)
    later, _ = literal(lat, wet, ct, pt, kt=181)
    for field in ("qsr", "qns", "emp", "utau", "vtau"):
        assert not np.array_equal(np.asarray(base[field]).view(np.uint64),
                                  np.asarray(later[field]).view(np.uint64)), field


def test_the_nyear_term_is_a_no_op_inside_year_one_and_not_beyond(literal):
    """usrdef_sbc.f90:107-108 subtracts (nyear-1)*rjjhh*zyydd = one full
    8640 h period per year.  Inside year 1 it subtracts exactly 0.0, which is
    exact; at nyear=2 it moves the argument and the result."""
    lat, wet, ct, pt = _toy()
    inside, _ = literal(lat, wet, ct, pt, kt=181, nyear=1)
    beyond, _ = literal(lat, wet, ct, pt, kt=181, nyear=2)
    assert not np.array_equal(np.asarray(inside["qsr"]).view(np.uint64),
                              np.asarray(beyond["qsr"]).view(np.uint64))
    # and it is MATHEMATICALLY a no-op: one period, so the value barely moves
    assert float(np.max(np.abs(inside["qsr"] - beyond["qsr"]))) < 1.0e-11


def test_the_qsr_pi_plant_moves_qsr(literal):
    """usrdef_sbc.f90:136 writes the literal 3.1415, NOT rpi.  A transcription
    that reaches for pi is a real defect, so the plant must be visible."""
    lat, wet, ct, pt = _toy()
    base, _ = literal(lat, wet, ct, pt)
    swapped, _ = literal(lat, wet, ct, pt, qsr_pi=3.141592653589793)
    moved = float(np.max(np.abs(base["qsr"] - swapped["qsr"])))
    assert moved > 1.0e-4, moved


def test_regions_partition_the_wet_surface(harness):
    lat = np.tile(np.linspace(12.0, 50.0, 8)[:, None], (1, 8))
    wet = np.zeros((8, 8), dtype=bool)
    wet[1:-1, 1:-1] = True
    regions = harness._regions(lat, wet)
    thirds = sum(regions[name].astype(int)
                 for name in ("west_third", "interior_third", "east_third"))
    assert np.array_equal(thirds, wet.astype(int))
    # non-vacuity: every cut must actually select something
    for name, mask in regions.items():
        assert mask.any(), name
    # and a cut must never reach land
    for name, mask in regions.items():
        assert not (mask & ~wet).any(), name


def test_the_step_entry_registry_covers_the_card_writer_and_no_more():
    """The card writes sixty per-step entry dumps
    (cfgs/GYRE_OMIP_L2_P3_SM_YRPERT/MY_SRC/stprk3.F90:90).  The registry must
    read all sixty and still FAIL CLOSED on the sixty-first."""
    from legoesm.ocean.fidelity.time_levels import time_level_for_dump
    for kt in (1, 2, 10, 11, 59, 60):
        assert time_level_for_dump(
            f"oracle_step_entry_kt{kt:08d}.bin") == "before"
    with pytest.raises(ValueError):
        time_level_for_dump("oracle_step_entry_kt00000061.bin")


def test_member_snapshot_cadence_refuses_a_partial_day():
    """run_member's new cadence names its files by DAY, so a cadence that is
    not a whole number of days would fold several steps onto one filename."""
    spec = importlib.util.spec_from_file_location(
        "gyre_year_fromrest_cadence",
        ROOT / "scripts" / "validate" / "ocean_fidelity" / "testcases"
        / "nemo_testcase_l2_gyre_year_fromrest.py")
    year = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(year)
    with pytest.raises(year.GateError):
        year.run_member(0, Path("/nonexistent"), days=1, snap_steps=1)
    # non-vacuity: the whole-day cadences pass this guard and fail LATER, on
    # something else, which proves the guard is what rejected snap_steps=1
    with pytest.raises(Exception) as info:
        year.run_member(0, Path("/nonexistent"), days=1, snap_steps=6)
    assert "snap_steps" not in str(info.value)


def test_run_sh_is_a_config_free_staging_script_and_never_builds():
    """The acquisition reuses the certified binary.  If it ever grows a
    makenemo call, the preregistration's ASKED item 1 is no longer describing
    the script and the receipt's admission argument is stale."""
    text = RUN_SH.read_text()
    assert "makenemo" not in text
    assert "cp -r" not in text          # never copy a whole cfgs/ directory
    assert "BYTE-IDENTICAL" in text
    assert "GYRE_OMIP_L2_P3_00000180_restart.nc" in text
    # the refusal must be a refusal, not a warning
    assert "exit 71" in text


@pytest.mark.skipif(not NEMO_RECORD.is_dir(),
                    reason="the NEMO year record is not on this machine")
def test_forcing_gate_plants_all_exit_non_zero():
    """Each plant must turn the BIT-EXACT forcing gate red.  Without this the
    gate's green is unfalsifiable."""
    for plant in ("forcing-phase", "forcing-qsr-pi", "forcing-nyear"):
        result = subprocess.run(
            [sys.executable, str(HARNESS), "--forcing-gate", "--days", "30",
             "--plant", plant],
            capture_output=True, text=True, cwd=str(ROOT))
        assert result.returncode == 1, (plant, result.stdout, result.stderr)
        assert "DEBT" in result.stdout, plant
