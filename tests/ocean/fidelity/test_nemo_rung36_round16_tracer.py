"""Focused controls for the round-16 coupled tracer owner."""

from __future__ import annotations

import struct

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.coupler.ocean_forcing import (
    NemoSI3ExchangeConfig,
    nemo_si3_exchange_forcing,
)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    _NEMOWSRK3TestHooks,
)
from legoesm.ocean.fidelity.nemo_testcase_recipe import (
    build_c1d_omip_l3_slab_ocean_card,
)
from legoesm.ocean.physics.surface_forcing.external import (
    nemo_tra_sbc_rk3_source,
)
from scripts.validate.ocean_fidelity.testcases.nemo_rung36_ocean_gate import (
    _records,
)


def test_si3_mapper_preserves_raw_nemo_rk3_operands() -> None:
    value = jnp.asarray([[0.125]], dtype=jnp.float64)
    freshwater, surface = nemo_si3_exchange_forcing(
        qsr=value, qns=value + 1, emp=value + 2, sfx=value + 3,
        utau=value, vtau=value, chl=value, rCdU_ice=-value,
        snwice_fmass=value, config=NemoSI3ExchangeConfig())
    raw = surface.nemo_rk3_surface
    np.testing.assert_array_equal(np.asarray(raw.qsr), np.asarray(value))
    np.testing.assert_array_equal(np.asarray(raw.qns), np.asarray(value + 1))
    np.testing.assert_array_equal(np.asarray(raw.emp), np.asarray(value + 2))
    np.testing.assert_array_equal(np.asarray(raw.sfx_pss), np.asarray(value + 3))
    np.testing.assert_array_equal(
        np.asarray(freshwater.ice_fw), np.asarray(-(value + 2)))


def test_source_statement_uses_kmm_thickness_before_kaa_division() -> None:
    h_kbb = jnp.float64(8.333333333333332)
    h_kmm = jnp.float64(8.333242568319797)
    h_kaa = jnp.float64(8.333151803306261)
    tracer = jnp.float64(-1.690032958984375)
    rhs, _ = nemo_tra_sbc_rk3_source(
        tendency_t=0.0, tendency_s=0.0,
        emp=jnp.float64(5.173605771912232e-5),
        qns=jnp.float64(-402.6598769905362),
        salt_flux_pss=jnp.float64(-5.668532644228933e-4),
        layer_thickness=h_kmm,
        inverse_density=jnp.float64(1.0 / 1026.0),
        inverse_heat_capacity=jnp.float64(1.0 / 3991.86795711963),
        temperature=tracer, salinity=jnp.float64(34.0), stage=3)
    # Literal trazdf.F90:271-286 replay; the fixture is the kt=1 oracle row.
    content = np.float64(h_kbb) * np.float64(tracer)
    content = np.float64(content + np.float64(
        np.float64(3600.0 * np.float64(h_kmm)) * np.float64(rhs)))
    got = np.float64(content / np.float64(h_kaa))
    expected = np.float64(-1.7325422954166647)
    assert got.view(np.uint64) == expected.view(np.uint64)


def test_slab_card_selects_resolved_off_advection_and_real_volume() -> None:
    cfg = build_c1d_omip_l3_slab_ocean_card().recipe.model_config
    assert cfg.tracer_advection == "off"
    assert cfg.freshwater_closure == "real_freshwater"
    assert cfg.barotropic.barotropic_continuity_evaluation == "nemo_literal"


def test_round16_causal_controls_are_private() -> None:
    hooks = _NEMOWSRK3TestHooks()
    assert hooks.tracer_surface_source_scale == 1.0
    assert hooks.expose_pre_implicit_tracer is False
    assert "tracer_surface_source_scale" not in (
        build_c1d_omip_l3_slab_ocean_card().recipe.model_config._fields)


def test_round16_schema_rejects_false_derived_count(tmp_path) -> None:
    path = tmp_path / "bad_tracer.bin"
    path.write_bytes(
        b"NEMO_L3TR16_001 "
        + struct.pack("=9i", 1, 3, 1, 2, 3, 3, 1, 17, 64)
        + np.zeros(18, dtype=np.float64).tobytes())
    with pytest.raises(ValueError, match="untrue header"):
        _records(path, b"NEMO_L3TR16_001 ", "9i", 18)
