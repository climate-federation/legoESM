"""AIMIP Phase 2.6: 3-way surface-albedo blend numerical sentinel.

Pins the iter-257 surface-albedo blend in
:meth:`legoesm.driver.physics_pipeline.PhysicsPipeline.compute_radiation_core`:

    albedo = lf * albedo_land + (1 - lf) * (sic * albedo_ice + (1 - sic) * albedo_ocean)

at the four canonical surface types (pure ocean, pure ice, pure land,
50/50 land+ocean) so any silent change to the blend (mis-weighted
terms, dropped land contribution, etc.) is caught at unit-test time.

Also pins that ``SegmentForcing.land_fraction`` defaults to all-zero
in :func:`legoesm.driver.compiled_segments.pack_forcing` so analytical
AMIP runs are bit-for-bit identical to pre-iter-257 behaviour.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.driver.compiled_segments import (
    SegmentForcing,
    pack_forcing,
)


_ALBEDO_LAND = 0.30
_ALBEDO_ICE = 0.65
_ALBEDO_OCEAN = 0.06


def _blend(lf: float, sic: float) -> float:
    """Reference 3-way blend in pure Python for the test oracle."""
    ocean_ice = sic * _ALBEDO_ICE + (1.0 - sic) * _ALBEDO_OCEAN
    return lf * _ALBEDO_LAND + (1.0 - lf) * ocean_ice


@pytest.mark.parametrize(
    "lf, sic, expected",
    [
        (0.0, 0.0, _ALBEDO_OCEAN),                     # pure ocean
        (0.0, 1.0, _ALBEDO_ICE),                       # pure ice
        (1.0, 0.0, _ALBEDO_LAND),                      # pure land
        (1.0, 1.0, _ALBEDO_LAND),                      # land beats ice when lf=1
        (0.5, 0.0, 0.5 * _ALBEDO_LAND + 0.5 * _ALBEDO_OCEAN),  # 50/50 land+ocean
        (0.3, 0.4, _blend(0.3, 0.4)),                  # generic mix
    ],
)
def test_three_way_albedo_blend_oracle(lf: float, sic: float, expected: float):
    """The 3-way blend matches the analytical oracle at canonical points."""
    _sic = jnp.asarray(sic)
    _lf = jnp.asarray(lf)
    ocean_ice = _sic * _ALBEDO_ICE + (1.0 - _sic) * _ALBEDO_OCEAN
    got = _lf * _ALBEDO_LAND + (1.0 - _lf) * ocean_ice
    # JAX defaults to float32 unless ``JAX_ENABLE_X64=1`` is set;
    # the float32 epsilon (~1.2e-7) bounds the achievable accuracy
    # for these scalar blends.
    np.testing.assert_allclose(float(got), expected, rtol=1e-6, atol=1e-7)


def test_pack_forcing_default_land_fraction_is_zero():
    """``pack_forcing`` without ``land_fraction`` produces all-zero
    so analytical AMIP runs reduce to the pre-iter-257 ice/ocean blend
    bit-for-bit.  Pinning this prevents a silent default-change that
    would alter analytical AMIP baselines."""
    f = pack_forcing(
        sst=jnp.ones((4, 8)),
        sic=jnp.zeros((4, 8)),
        day_of_year=1.0,
        seconds_of_day=0.0,
        solar_weights=jnp.ones((4, 8)),
        s_0=1361.0,
        o3_vmr=jnp.zeros((4, 8, 10)),
        aerosol_od=jnp.zeros((4, 8)),
    )
    assert isinstance(f, SegmentForcing)
    np.testing.assert_array_equal(np.asarray(f.land_fraction), np.zeros((4, 8)))
    assert f.land_fraction.shape == (4, 8)


def test_pack_forcing_round_trips_provided_land_fraction():
    """A user-supplied ``land_fraction`` round-trips through
    ``pack_forcing`` unchanged (modulo float32 truncation: JAX defaults
    to float32 unless ``JAX_ENABLE_X64=1`` is set, so we use
    ``assert_allclose`` rather than ``assert_array_equal``)."""
    lf_in = 0.3 * jnp.ones((4, 8))
    f = pack_forcing(
        sst=jnp.ones((4, 8)),
        sic=jnp.zeros((4, 8)),
        day_of_year=1.0,
        seconds_of_day=0.0,
        solar_weights=jnp.ones((4, 8)),
        s_0=1361.0,
        o3_vmr=jnp.zeros((4, 8, 10)),
        aerosol_od=jnp.zeros((4, 8)),
        land_fraction=lf_in,
    )
    np.testing.assert_allclose(
        np.asarray(f.land_fraction), 0.3 * np.ones((4, 8)),
        rtol=1e-6, atol=1e-7,
    )


def test_segment_forcing_has_land_fraction_field():
    """``land_fraction`` is the 10th NamedTuple field — pinned so a
    silent drop trips this sentinel rather than blowing up at the
    next ``forcing.land_fraction`` site inside the scan body."""
    assert "land_fraction" in SegmentForcing._fields, (
        "SegmentForcing lost its ``land_fraction`` field.  See "
        "iter-257 (AIMIP Phase 2.6) in new_test_dycores.md for "
        "the rationale; that channel is consumed by "
        "physics_pipeline.compute_radiation_core's 3-way albedo "
        "blend, so dropping it silently breaks ERA5/AIMIP runs."
    )
    assert len(SegmentForcing._fields) == 10, (
        f"SegmentForcing has {len(SegmentForcing._fields)} fields, "
        f"expected 10 (pre-iter-257 had 9 + land_fraction)."
    )
