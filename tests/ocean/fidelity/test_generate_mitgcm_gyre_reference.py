"""Unit tests for the MITgcm barotropic-gyre reference generator's pure helpers.

The full build+run pipeline needs a Fortran toolchain (a cluster node); these
tests cover the deterministic, dependency-free helpers so the script's logic is
verified even where MITgcm cannot be built.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts" / "data" / "generate_mitgcm_barotropic_gyre_reference.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_gen_mitgcm_gyre", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load_module()


_MINIMAL_DATA = """\
 &PARM01
 viscAh=4.E2,
 &

 &PARM03
 nIter0=0,
 nTimeSteps=10,
 deltaT=1200.0,
 &
"""


def test_patch_data_adds_global_files_in_parm01():
    """globalFiles=.TRUE. (the key fix) must land inside PARM01."""
    patched = gen.patch_data_for_field_dumps(_MINIMAL_DATA)
    assert "globalFiles=.TRUE." in patched
    p01 = patched.index("&PARM01")
    term = patched.index("&", p01 + 1)  # PARM01's terminator
    assert p01 < patched.index("globalFiles") < term


def test_patch_data_adds_dump_init_and_last_in_parm03():
    patched = gen.patch_data_for_field_dumps(_MINIMAL_DATA)
    assert "dumpInitAndLast=.TRUE." in patched
    p03 = patched.index("&PARM03")
    term = patched.index("&", p03 + 1)
    assert p03 < patched.index("dumpInitAndLast") < term


def test_patch_data_is_idempotent():
    once = gen.patch_data_for_field_dumps(_MINIMAL_DATA)
    twice = gen.patch_data_for_field_dumps(once)
    assert once == twice
    assert twice.count("dumpInitAndLast") == 1
    assert twice.count("globalFiles") == 1


def test_patch_data_overrides_ntimesteps():
    patched = gen.patch_data_for_field_dumps(_MINIMAL_DATA, n_timesteps=500)
    assert "nTimeSteps=500" in patched
    assert "nTimeSteps=10" not in patched


def test_patch_data_without_parm03_raises():
    with pytest.raises(ValueError, match="PARM03"):
        gen.patch_data_for_field_dumps(" &PARM01\n &\n")


def test_archive_plan_covers_fields_grid_and_monitor(tmp_path):
    run, ref = tmp_path / "run", tmp_path / "ref"
    pairs = gen.archive_plan(run, ref, iteration=10)
    names = {src.name for src, _ in pairs}
    # Prognostic field dumps at the iteration, grid descriptors, monitor log.
    assert {"U.0000000010.data", "V.0000000010.meta", "Eta.0000000010.data"} <= names
    assert {"XC.data", "RC.meta", "hFacC.data", "Depth.meta"} <= names
    assert "output.txt" in names
    # Destinations are all under the reference dir.
    assert all(dst.parent == ref for _, dst in pairs)


def test_resolve_mitgcm_root_explicit(tmp_path):
    deck = tmp_path / "verification" / "tutorial_barotropic_gyre" / "input"
    deck.mkdir(parents=True)
    (deck / "data").write_text("&PARM03\n&\n")
    assert gen.resolve_mitgcm_root(str(tmp_path)) == tmp_path


def test_resolve_mitgcm_root_missing_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("MITGCM_ROOTDIR", raising=False)
    # Point fallbacks at nonexistent paths so the probe genuinely fails.
    monkeypatch.setattr(gen, "_FALLBACK_ROOTS", (str(tmp_path / "nope"),))
    with pytest.raises(FileNotFoundError, match="tutorial_barotropic_gyre"):
        gen.resolve_mitgcm_root(str(tmp_path / "absent"))
