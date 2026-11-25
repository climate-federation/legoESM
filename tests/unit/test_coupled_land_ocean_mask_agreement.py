import importlib.util
from pathlib import Path

import numpy as np
import pytest

_p = Path(__file__).resolve().parents[2] / "scripts/validate/coupled_land_ocean_mask_agreement.py"
_s = importlib.util.spec_from_file_location("_mask_agree", _p)
mod = importlib.util.module_from_spec(_s)
_s.loader.exec_module(mod)


def test_classify_counts_both_disagreement_classes():
    f_land = np.array([[1.0, 1.0, 0.5, 0.0]])
    wet = np.array([[1.0, 0.0, 0.0, 1.0]])
    r = mod.classify(f_land, wet)
    assert (r["wet_and_full_land"], r["dry_and_some_water"]) == (1, 1)
    assert (r["n_full_land"], r["n_partial_land"], r["n_ocean_wet"]) == (2, 1, 2)


def test_classify_refuses_mismatched_grids():
    with pytest.raises(ValueError, match="grid mismatch"):
        mod.classify(np.zeros((2, 3)), np.zeros((3, 2)))
