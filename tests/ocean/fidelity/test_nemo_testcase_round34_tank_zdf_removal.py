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
