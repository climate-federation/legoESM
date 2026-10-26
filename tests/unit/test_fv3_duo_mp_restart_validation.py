"""``_validate_fv3_duo_broadcast_source`` is the guard that turns a
corrupt/foreign multi-process fv3_duo checkpoint into a LOUD error on
every rank BEFORE any ``broadcast_one_to_all`` collective runs (a
mismatch discovered mid-broadcast would otherwise hang or raise
confusingly on a subset of ranks).

These are single-process unit tests of that pure validator (no
``jax.distributed`` / multiprocess needed): they build a template leaf
dict shaped like ``_fv3_duo_flatten_bundle``'s output and assert the
validator ACCEPTS a matching source and REJECTS each corruption class
the multi-process loader must catch.  The full multi-process
uninterrupted-vs-restart bitwise round-trip lives in
``scripts/validate/fv3_native/spmd_multiprocess_restart_roundtrip.py``
(needs a real srun launch, not a pytest fixture).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from legoesm.driver.model_driver import ModelDriver


def _template(nq=1, km=5, n=6):
    """A leaf dict shaped like ``_fv3_duo_flatten_bundle`` output:
    ``state_*``, ``press_*``, ``q_i``, ``omga`` (all float64).  Shapes
    need only be self-consistent between template and source for this
    validator; the exact face layout is irrelevant to the checks."""
    m = n + 6
    t = {
        "state_delp": np.zeros((6, m, m, km)),
        "state_pt": np.zeros((6, m, m, km)),
        "state_u": np.zeros((6, m, m + 1, km)),
        "state_v": np.zeros((6, m + 1, m, km)),
        "press_ps": np.zeros((6, m, m)),
        "press_pe": np.zeros((6, n + 2, km + 1, n + 2)),
        "press_peln": np.zeros((6, n, km + 1, n)),
        "press_pk": np.zeros((6, n, n, km + 1)),
        "press_pkz": np.zeros((6, n, n, km)),
        "omga": np.zeros((6, m, m, km)),
    }
    for i in range(nq):
        t[f"q_{i}"] = np.zeros((6, m, m, km))
    return t


def _validate(src, template, nq):
    from pathlib import Path
    fake = MagicMock()
    # Bind the REAL shared leaf-spec helper (the validator calls
    # self._fv3_duo_broadcast_leaf_spec) so it returns a genuine
    # (shape, dtype) tuple, not a MagicMock that fails to unpack.
    fake._fv3_duo_broadcast_leaf_spec = (
        lambda nm, tmpl, qs, qd: ModelDriver._fv3_duo_broadcast_leaf_spec(
            fake, nm, tmpl, qs, qd))
    return ModelDriver._validate_fv3_duo_broadcast_source(
        fake, src, template, nq, Path("ckpt.npz"))


def test_matching_source_accepted():
    t = _template(nq=1)
    # A well-formed source == the template's own leaf set/shapes/dtypes.
    _validate(dict(t), t, nq=1)   # must not raise


def test_wrong_tracer_count_rejected():
    """nq disagreeing with the deck's own tracer count (the nq=0
    silent-tracer-loss hole) is refused."""
    t = _template(nq=1)
    src = dict(t)
    del src["q_0"]                # a 0-tracer source vs a 1-tracer deck
    with pytest.raises(ValueError, match="tracer count"):
        _validate(src, t, nq=0)


def test_missing_leaf_rejected():
    t = _template(nq=1)
    src = dict(t)
    del src["state_u"]
    with pytest.raises(ValueError, match="leaf set"):
        _validate(src, t, nq=1)


def test_extra_leaf_rejected():
    t = _template(nq=1)
    src = dict(t)
    src["state_bogus"] = np.zeros((6, 12, 12, 5))
    with pytest.raises(ValueError, match="leaf set"):
        _validate(src, t, nq=1)


def test_wrong_shape_rejected():
    t = _template(nq=1)
    src = dict(t)
    src["state_pt"] = np.zeros((6, 12, 12, 4))   # km 4 not 5
    with pytest.raises(ValueError, match="mismatch|shape"):
        _validate(src, t, nq=1)


def test_wrong_dtype_rejected():
    t = _template(nq=1)
    src = dict(t)
    src["omga"] = np.zeros((6, 12, 12, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="mismatch|shape|dtype"):
        _validate(src, t, nq=1)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
