"""Decision 19: the tanks remove the barotropic mode before the implicit solve.

The user answered this one "Do as NEMO does".  Both tanks resolve
``ln_drgimp = T`` and ``ln_dynspg_ts = T``, so NEMO takes the branch at
``dynzdf.F90:148`` that subtracts ``uu_b(Kaa)`` from every level before the
implicit vertical solve.  legoESM did not.  These arms pin the resolution --
which is a CONFIGURATION fact, so the test reads it off the built card rather
than off the source -- and pin the two premises the change rests on.
"""

from __future__ import annotations

import numpy as np
import pytest
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card

TANKS = ("LOCK_EXCHANGE-zco", "OVERFLOW-zps")


@pytest.mark.parametrize("case", TANKS)
def test_the_tanks_remove_the_barotropic_mode_before_the_implicit_solve(case):
    config = build_nemo_testcase_card(case).recipe.model_config
    assert config.zdf_baroclinic_only is True


@pytest.mark.parametrize("case", TANKS)
def test_the_flip_is_the_ONLY_flag_that_moved_on_these_cards(case):
    """Decision 19 authorised one field.  Nothing else may ride along."""
    config = build_nemo_testcase_card(case).recipe.model_config
    assert config.zdf_drag_in_matrix is False
    assert config.barotropic_drag_substep is False
    assert config.barotropic.nemo_stage_mean_imposition is False


def test_gyres_resolution_is_UNCHANGED():
    """GYRE already had the bundle; decision 19 must not have touched it."""
    config = build_nemo_testcase_card("GYRE-zco").recipe.model_config
    assert config.zdf_baroclinic_only is True
    assert config.zdf_drag_in_matrix is True
    assert config.barotropic_drag_substep is True
    assert config.barotropic.nemo_stage_mean_imposition is True


@pytest.mark.parametrize(
    "case,root,expected_wet",
    [("LOCK_EXCHANGE-zco",
      "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round33_lock_zdf_matrix",
      390),
     ("OVERFLOW-zps",
      "/data/abyssal/dbalwada/nemo-testcases-l2/phase3"
      "/round33_overflow_zdf_matrix", 606)])
def test_nemos_barotropic_bottom_stress_is_EXACTLY_zero_on_these_cards(
        case, root, expected_wet):
    """The premise that lets the companion statement go untranscribed.

    ``dynzdf.F90:156-159`` adds the barotropic bottom stress back at the
    deepest wet level, and legoESM does not transcribe it here.  That is
    bit-exact only because ``rCdU_bot`` is EXACTLY zero on every owned cell of
    both tanks.  A whole-array maximum would report a denormal near 6.9e-310,
    which is uninitialised halo memory -- so this arm strips the halo, the way
    every reader in this campaign does.
    """
    import importlib.util
    import sys
    from pathlib import Path

    gates = (Path(__file__).parents[3]
             / "scripts/validate/ocean_fidelity/testcases")
    if str(gates) not in sys.path:
        sys.path.insert(0, str(gates))
    spec = importlib.util.spec_from_file_location(
        "r29_zdf_matrix", gates / "nemo_testcase_l2_gyre_round29_zdf_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    record = Path(root) / "oracle_zdf_matrix_kt00000001.bin"
    if not record.is_file():
        pytest.skip(f"{record} has not been acquired")
    arrays = module.read_zdf_matrix(record)["arrays"]
    header = module.read_zdf_matrix(record)["header"]
    nx, ny = header["jpi"], header["jpj"]
    drag = np.asarray(arrays["rCdU_bot"], dtype=np.float64).reshape(
        (nx, ny), order="F")
    owned = drag[2:-2, 2:-2]
    assert owned.size == expected_wet
    assert np.count_nonzero(owned) == 0
    # ... and the halo really does carry the denormal that a naive maximum
    # would report, so this arm is not measuring a field that is zero anyway.
    assert np.max(np.abs(drag)) > 0.0


# --------------------------------------------------------------------------
# Round-34, SECOND diff review.  Two findings, each with the arm it bought.
# --------------------------------------------------------------------------

def test_both_tanks_switch_bottom_drag_OFF_so_the_zero_holds_at_EVERY_step():
    """The premise is stronger than a kt=1 record could establish.

    The commit argued from the round-33 record that ``rCdU_bot`` is zero on
    every owned cell -- true, but that record is kt=1 only, so on its own it
    cannot say whether the companion statement stays inert later.  Both tanks
    resolve ``ln_drg_OFF = T`` (lock_kt1_10/ocean.output:556,
    overflow_kt1_10/ocean.output:668), which is what makes the bottom-drag
    rate identically zero at EVERY step.  An independent review found this.
    """
    from pathlib import Path
    import re

    runs = {
        "LOCK_EXCHANGE": Path("/data/abyssal/dbalwada/nemo-testcases-l1"
                              "/phase3/lock_kt1_10/ocean.output"),
        "OVERFLOW": Path("/data/abyssal/dbalwada/nemo-testcases-l1"
                         "/phase3/overflow_kt1_10/ocean.output"),
    }
    for card, log in runs.items():
        if not log.is_file():
            pytest.skip(f"{log} is not available")
        hits = [(n, line.strip()) for n, line
                in enumerate(log.read_text(errors="replace").splitlines(), 1)
                if re.search(r"\bln_drg_OFF\s*=\s*[TF]\b", line)]
        assert hits, f"{card}: ocean.output does not print ln_drg_OFF"
        for _, line in hits:
            assert re.search(r"\bln_drg_OFF\s*=\s*T\b", line), (
                f"{card}: bottom drag is NOT off ({line}); the untranscribed "
                "barotropic bottom-stress statement is no longer inert")


@pytest.mark.parametrize(
    "card,root",
    [("LOCK_EXCHANGE",
      "/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round33_lock_zdf_matrix"),
     ("OVERFLOW",
      "/data/abyssal/dbalwada/nemo-testcases-l2/phase3"
      "/round33_overflow_zdf_matrix")])
def test_no_dry_level_can_reach_a_wet_one_through_nemos_own_matrix(card, root):
    """The review's second finding, refuted at the oracle's own matrix.

    NEMO writes the removal as ``(puu - uu_b)*umask``; legoESM subtracts
    unmasked, so a wet column's DRY levels hold ``-uu_b`` rather than zero.
    That only matters if the implicit momentum matrix couples the deepest wet
    level to the level below it.  In NEMO's OWN matrix -- copied by the
    round-29 instrument after every contribution, with ln_zad_Aimp = T live on
    both cards -- the super-diagonal at the deepest wet level is EXACTLY zero
    on every wet column, so no dry level can reach a wet one.
    """
    import importlib.util
    import sys
    from pathlib import Path

    gates = (Path(__file__).parents[3]
             / "scripts/validate/ocean_fidelity/testcases")
    if str(gates) not in sys.path:
        sys.path.insert(0, str(gates))
    spec = importlib.util.spec_from_file_location(
        "r29_zdf_matrix_bed",
        gates / "nemo_testcase_l2_gyre_round29_zdf_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    record = Path(root) / "oracle_zdf_matrix_kt00000001.bin"
    if not record.is_file():
        pytest.skip(f"{record} has not been acquired")
    parsed = module.read_zdf_matrix(record)
    header, arrays = parsed["header"], parsed["arrays"]
    nx, ny, nz = header["jpi"], header["jpj"], header["jpk"]
    nzm1 = header["jpkm1"]

    def owned(name, levels):
        return np.asarray(arrays[name], dtype=np.float64).reshape(
            (nx, ny, levels), order="F")[2:-2, 2:-2, :]

    bed = np.asarray(arrays["mbku"], dtype=np.float64).reshape(
        (nx, ny), order="F")[2:-2, 2:-2].astype(int)
    wet_i, wet_j = np.nonzero(bed > 0)
    assert wet_i.size > 0
    deepest = np.clip(bed[wet_i, wet_j] - 1, 0, nzm1 - 1)
    super_diagonal = owned("zws_u", nzm1)[wet_i, wet_j, deepest]
    assert np.count_nonzero(super_diagonal) == 0, (
        f"{card}: NEMO's own matrix couples {np.count_nonzero(super_diagonal)}"
        " wet bed cells to the level below")
    # ... and the arm is not vacuous: the same matrix's OTHER coefficient at
    # the same cells is nonzero, so a zero here is a fact about the geometry
    # and not about an array the instrument failed to fill.
    assert np.max(np.abs(owned("zwi_u", nzm1)[wet_i, wet_j, deepest])) > 0.0
    # the level below the bed really is dry, which is what makes the question
    # meaningful at all
    below = owned("umask", nz)[wet_i, wet_j,
                               np.clip(bed[wet_i, wet_j], 0, nz - 1)]
    assert np.max(np.abs(below)) == 0.0
