"""Cross-precision restart digest verification (_verify_state_digest).

An fp32-written checkpoint resumed under --precision fp64 used to hard-fail
"State digest mismatch": the validator hashed the (up-cast) loaded arrays at
the ACTIVE dtype instead of the SAVED one.  Up-casts are lossless, so
re-hashing at the saved dtype verifies exactly; lossy down-casts cannot be
verified and must warn, not raise; genuine corruption must still raise.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.driver.restart import RestartMetadata, _verify_state_digest
from legoesm.io.state_digest import compute_state_digest


def _meta(digest: str, storage_dtype: str = "") -> RestartMetadata:
    return RestartMetadata(
        model_version="t", creation_time="t", platform="t", jax_version="t",
        jax_x64_enabled=True, numpy_version="t", config_hash="c",
        state_digest=digest, resolution=4, nlev=3, step=0, day=0.0,
        git_hash="", storage_dtype=storage_dtype,
    )


def _arrays(dtype):
    rng = np.random.default_rng(0)
    return {"T": rng.normal(280, 10, (4, 3)).astype(dtype),
            "q": rng.random((4, 3)).astype(dtype)}


def _saved_digest(dtype):
    return compute_state_digest(_arrays(dtype))


def test_same_dtype_passes_silently():
    _verify_state_digest(_arrays(np.float32), _meta(_saved_digest(np.float32),
                                                    "float32"), np.float32)


def test_fp32_saved_fp64_loaded_verifies_with_warning():
    # loader up-cast the fp32 arrays to fp64 (lossless)
    loaded = {k: v.astype(np.float64) for k, v in _arrays(np.float32).items()}
    with pytest.warns(UserWarning, match="Cross-precision"):
        _verify_state_digest(loaded, _meta(_saved_digest(np.float32),
                                           "float32"), np.float64)


def test_fp64_saved_fp32_loaded_warns_not_raises():
    # loader DOWN-cast (lossy) -> unverifiable, warn only
    loaded = {k: v.astype(np.float32) for k, v in _arrays(np.float64).items()}
    with pytest.warns(UserWarning, match="discarded precision"):
        _verify_state_digest(loaded, _meta(_saved_digest(np.float64),
                                           "float64"), np.float32)


def test_corruption_still_raises():
    bad = _arrays(np.float32)
    bad["T"] = bad["T"] + 1.0
    with pytest.raises(ValueError, match="State digest mismatch"):
        _verify_state_digest(bad, _meta(_saved_digest(np.float32),
                                        "float32"), np.float32)


def test_legacy_same_dtype_passes():
    _verify_state_digest(_arrays(np.float32),
                         _meta(_saved_digest(np.float32)), np.float32)


def test_legacy_cross_precision_sibling_verifies():
    # legacy meta (no dtype recorded): fp32-saved, loaded under fp64
    loaded = {k: v.astype(np.float64) for k, v in _arrays(np.float32).items()}
    with pytest.warns(UserWarning, match="legacy metadata"):
        _verify_state_digest(loaded, _meta(_saved_digest(np.float32)),
                             np.float64)


def test_legacy_corruption_raises():
    bad = {k: v.astype(np.float64) + 1.0
           for k, v in _arrays(np.float32).items()}
    with pytest.raises(ValueError, match="State digest mismatch"):
        _verify_state_digest(bad, _meta(_saved_digest(np.float32)), np.float64)
