"""Direct checks of the RRTMGP oracle audit's own comparison arithmetic."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from legoesm import constants

pytest.importorskip("netCDF4")
_PATH = Path(__file__).parents[2] / "scripts/validate/oracle/rrtmgp_faithfulness/audit.py"
_SPEC = importlib.util.spec_from_file_location("rrtmgp_faithfulness_audit", _PATH)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def test_cmp_records_worst_cell_and_flags_nonfinite():
    audit.cmp("a", [1.0, 5.0], [1.0, 4.0])
    r = audit.RESULTS["a"]
    assert (r["max_abs"], r["at"], r["ours"], r["ref"]) == (1.0, [1], 5.0, 4.0)
    assert r["max_rel"] == pytest.approx(0.25)
    audit.cmp("b", [np.nan, 1.0], [1.0, 1.0])
    assert audit.RESULTS["b"] == dict(nonfinite_ours=1, nonfinite_oracle=0)


def test_cmp_refuses_mismatched_shapes():
    with pytest.raises(AssertionError):
        audit.cmp("c", [1.0], [1.0, 2.0])


def test_heating_sign_warms_a_layer_gaining_energy():
    """Levels run surface first: net downward flux falling toward the surface warms."""
    p = np.array([[1.0e5, 0.9e5]])
    up, dn = np.array([[0.0, 0.0]]), np.array([[0.0, 10.0]])
    expected = constants.g * 10.0 / 1.0e4 / constants.c_pd * audit.SECONDS_PER_DAY
    assert audit.heating(up, dn, p)[0, 0] == pytest.approx(expected)
