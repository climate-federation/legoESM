"""Per-column NaN-revert guard for the LMIP driver (run_lmip_biophys).

Columns are independent in the offline land model, so a diverging boreal/Arctic
cell must be reverted to its previous state WITHOUT touching the finite cells and
WITHOUT poisoning the run-level PASS/FAIL.  ``_nonfinite_per_col`` is the detector
that makes the revert per-column; these tests pin its contract.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import jax.numpy as jnp

_ROOT = Path(__file__).resolve().parents[2]
_DRIVER = _ROOT / "scripts" / "run" / "run_lmip_biophys.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("run_lmip_biophys", _DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flags_only_the_bad_columns():
    mod = _load_driver()
    tree = {
        # (4, 2): column 1 has a NaN in a sub-layer
        "T_soil": jnp.array([[1.0, 2.0], [jnp.nan, 3.0], [4.0, 5.0], [6.0, 7.0]]),
        # (4,): column 2 is inf
        "snow": jnp.array([1.0, 2.0, jnp.inf, 4.0]),
    }
    flags = np.asarray(mod._nonfinite_per_col(tree, 4))
    assert flags.dtype == bool
    assert flags.tolist() == [False, True, True, False]


def test_all_finite_flags_none():
    mod = _load_driver()
    tree = {"a": jnp.ones((3, 5)), "b": jnp.zeros(3)}
    assert not np.asarray(mod._nonfinite_per_col(tree, 3)).any()


def test_revert_reshape_reverts_only_flagged_columns():
    """The reshape-broadcast used in the scan body must revert only flagged
    columns and leave the rest (and other sub-layers) untouched."""
    ncol = 3
    reverted = jnp.array([False, True, False])
    new = jnp.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]])   # (3, 2)
    old = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    m = reverted.reshape((ncol,) + (1,) * (new.ndim - 1))
    out = np.asarray(jnp.where(m, old, new))
    assert out.tolist() == [[10.0, 11.0], [2.0, 2.0], [30.0, 31.0]]


def test_non_column_leaves_are_skipped():
    mod = _load_driver()
    # a leaf whose leading dim != ncol (e.g. a shared scalar-ish array) is ignored
    tree = {"percol": jnp.array([1.0, jnp.nan, 3.0]), "shared": jnp.array([9.0, 9.0])}
    flags = np.asarray(mod._nonfinite_per_col(tree, 3))
    assert flags.tolist() == [False, True, False]
