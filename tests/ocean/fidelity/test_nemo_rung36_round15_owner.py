"""Unit controls for the round-15 coupled-slab momentum owner."""

from __future__ import annotations

import struct

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.dynamics.barotropic_common import (
    compute_nemo_forward_raw_primary_weights,
)
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    nemo_literal_temporal_combination,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    _NEMOWSRK3TestHooks,
)
from scripts.validate.ocean_fidelity.testcases.nemo_rung36_ocean_gate import (
    _records,
)
from scripts.validate.ocean_fidelity.testcases.nemo_rung36_round15_owner_gate import (
    _qco_rhs,
)


def test_raw_primary_boxcar_matches_nemo_window() -> None:
    weights, divisor, nloop = compute_nemo_forward_raw_primary_weights(
        631, jnp.float64)
    values = np.asarray(weights)
    assert nloop == 946
    assert float(divisor) == 631.0
    assert set(np.unique(values)) == {0.0, 1.0}


def test_temporal_combination_preserves_left_to_right_products() -> None:
    coefficients = tuple(map(jnp.float64, (1.781105, -1.06221, 0.281105)))
    levels = tuple(map(jnp.float64, (0.125, -0.03125, 0.015625)))
    got = nemo_literal_temporal_combination(coefficients, levels)
    expected = np.float64(coefficients[0] * levels[0])
    expected = np.float64(expected + np.float64(coefficients[1] * levels[1]))
    expected = np.float64(expected + np.float64(coefficients[2] * levels[2]))
    assert np.asarray(got).view(np.uint64) == expected.view(np.uint64)


def test_qco_rhs_replays_registered_source_statement() -> None:
    got = _qco_rhs(
        *map(jnp.float64, (
            0.0, 1.0845180127162992e-08,
            -1.0 / 6.0, -0.16667574316802025,
            -0.1666848196693738,
        )),
        jnp.float64(3600.0), jnp.float64(1.0),
    )
    expected = np.float64.fromhex("0x1.47845cc3e00d2p-15")
    assert np.asarray(got).view(np.uint64) == expected.view(np.uint64)


def test_post_zdf_mean_scaling_is_private_and_defaults_to_nemo() -> None:
    assert _NEMOWSRK3TestHooks().post_zdf_mean_scale == 1.0
    assert "post_zdf_mean_scale" not in (
        __import__(
            "legoesm.ocean.state", fromlist=["BarotropicConfig"]
        ).BarotropicConfig._fields
    )


def test_owner_stream_header_validation_fails_closed(tmp_path) -> None:
    path = tmp_path / "bad.bin"
    path.write_bytes(
        b"NEMO_L3ZRHS_001 "
        + struct.pack("=8i", 1, 1, 2, 3, 3, 3, 12, 64)
        + np.zeros(13, dtype=np.float64).tobytes()
    )
    with pytest.raises(ValueError, match="untrue header"):
        _records(path, b"NEMO_L3ZRHS_001 ", "8i", 13)
