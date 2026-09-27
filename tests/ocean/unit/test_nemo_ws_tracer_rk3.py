"""Stage-program tests for NEMO's key_RK3 active tracers."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from types import SimpleNamespace

from legoesm import constants
import legoesm.ocean.dynamics.ocean_model_latlon_cgrid as model_module
import legoesm.ocean.dynamics.ocean_pe_latlon_cgrid as pe_module
import legoesm.ocean.vertical as vertical_module
import legoesm.ocean.eos as eos_module
from legoesm.ocean.dynamics.barotropic_latlon_cgrid import (
    _nemo_flux_form_external_velocity_update,
)
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    nemo_hpg_sco_literal_cgrid,
)
from legoesm.ocean.eos import (
    nemo_roquet_density_anomaly_ratio,
    nemo_teos10_density_anomaly_ratio,
)
from legoesm.core.precision import PrecisionPolicy, set_policy
from legoesm.grids.latlon import create_beta_plane_cgrid_geometry, create_latlon_grid
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_lock_exchange_zco_card
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_nemo_testcase_card
from legoesm.ocean.fidelity.nemo_testcase_recipe import build_overflow_zps_card
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


def test_nemo_ws_stage_transport_preserves_fortran_product_association():
    """The metric-bearing zF is materialized in NEMO source order."""
    set_policy(PrecisionPolicy.fp64())
    rng = np.random.default_rng(1301)
    metric = jnp.asarray(rng.uniform(1000.0, 9000.0, (2, 3)))
    thickness = jnp.asarray(rng.uniform(1.0, 250.0, (2, 3, 7)))
    velocity = jnp.asarray(rng.uniform(-0.25, 0.25, (2, 3, 7)))
    expected = np.asarray(
        (np.asarray(metric)[..., None] * np.asarray(thickness))
        * np.asarray(velocity))
    actual = np.asarray(jax.jit(model_module._nemo_metric_stage_transport)(
        metric, thickness, velocity))
    assert np.array_equal(actual, expected)

    # Non-vacuity: the pre-round-13 regrouping first formed e3*velocity and
    # only then multiplied by the horizontal metric; these inputs move bits.
    reassociated = (
        np.asarray(metric)[..., None]
        * (np.asarray(thickness) * np.asarray(velocity)))
    assert not np.array_equal(reassociated, expected)


def test_nemo_r3t_stage1_interpolates_endpoint_ratios_in_source_order():
    """The executing RK3 Kaa ratio uses NEMO's endpoint association."""
    set_policy(PrecisionPolicy.fp64())
    depth = jnp.asarray([[1.5528150381393142]], dtype=jnp.float64)
    before = jnp.asarray([[0.366016802486834]], dtype=jnp.float64)
    after = jnp.asarray([[0.26495575608265054]], dtype=jnp.float64)
    z_coord = SimpleNamespace(linear_free_surface=False)

    def evaluate(a, b):
        return eos_module.nemo_r3t_rk3_stage1_stretch(
            z_coord, a, b, depth)

    eager = evaluate(before, after)
    compiled = jax.jit(evaluate)(before, after)
    np.testing.assert_array_equal(np.asarray(compiled), np.asarray(eager))

    r1_depth = np.float64(1.0) / np.asarray(depth)
    r3_before = np.asarray(before) * r1_depth
    r3_after = np.asarray(after) * r1_depth
    expected = np.float64(1.0) + (
        (np.float64(2.0) / np.float64(3.0)) * r3_before
        + (np.float64(1.0) / np.float64(3.0)) * r3_after)
    np.testing.assert_array_equal(np.asarray(compiled), expected)

    interpolated_ssh = np.asarray(before) + (
        np.asarray(after) - np.asarray(before)) / np.float64(3.0)
    wrong = np.float64(1.0) + interpolated_ssh * r1_depth
    assert not np.array_equal(np.asarray(compiled), wrong)

    gradients = jax.grad(lambda a, b: jnp.sum(evaluate(a, b)), argnums=(0, 1))(
        before, after)
    assert all(np.isfinite(np.asarray(value)).all() for value in gradients)
    assert all(np.any(np.asarray(value) != 0.0) for value in gradients)

    source = inspect.getsource(model_module.LatLonCGridOceanModel._step_impl)
    assert "_qt_13 = nemo_r3t_rk3_stage1_stretch(" in source


def test_nemo_qco_live_t_thickness_matches_literal_source_bits():
    """Pin domain/domqco/e3t macro association on awkward partial cells."""
    set_policy(PrecisionPolicy.fp64())
    e3t0 = np.asarray([
        [[1.234567890123, 3.456789012345],
         [7.654321098765, 0.456789012345]],
        [[2.345678901234, 6.789012345678],
         [9.876543210987, 0.123456789012]],
    ], dtype=np.float64)
    active = np.asarray([
        [[1, 1], [1, 0]],
        [[1, 1], [1, 1]],
    ], dtype=bool)
    eta = np.asarray([
        [0.1234567890123, -0.0123456789012],
        [0.2345678901234, -0.0234567890123],
    ], dtype=np.float64)
    H = np.asarray([
        [123.4567890123, 45.6789012345],
        [234.5678901234, 67.8901234567],
    ], dtype=np.float64)
    z_coord = SimpleNamespace(
        nemo_e3t_0=jnp.asarray(e3t0), is_active=jnp.asarray(active),
        linear_free_surface=False,
    )

    ssmask = active[..., 0].astype(np.float64)
    denominator = (H + np.float64(1.0)) - ssmask
    r1_ht0 = ssmask / denominator
    r3t = eta * r1_ht0
    expected = e3t0 * (
        np.float64(1.0) + r3t[..., None] * active.astype(np.float64))
    actual = np.asarray(jax.jit(
        lambda ssh, depth: vertical_module.nemo_qco_live_t_thickness(
            ssh, depth, z_coord, jnp.float64))(jnp.asarray(eta), jnp.asarray(H)))
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))

    # Non-vacuity: the old generic Jacobian divides the live water column by
    # depth and multiplies afterward; it is real-equivalent but not bitwise.
    legacy = e3t0 * ((eta + H) / H)[..., None]
    assert not np.array_equal(legacy.view(np.uint64), expected.view(np.uint64))


def test_nemo_cen2_tracer_rhs_matches_literal_source_bits():
    """Pin traadv_cen CEN2 face, metric and live-e3 association."""
    set_policy(PrecisionPolicy.fp64())
    rng = np.random.default_rng(2201)
    ny, nx, nz = 3, 4, 2
    base = create_latlon_grid(ny, nx, dtype=jnp.float64)
    area = rng.uniform(1.0e5, 9.0e6, (ny, nx)).astype(np.float64)
    grid = SimpleNamespace(**base._asdict(), area_T=jnp.asarray(area))
    tracer = rng.uniform(-3.0, 37.0, (ny, nx, nz)).astype(np.float64)
    p_u = rng.uniform(-8.0e5, 8.0e5, (ny, nx + 1, nz)).astype(np.float64)
    p_v = rng.uniform(-8.0e5, 8.0e5, (ny + 1, nx, nz)).astype(np.float64)
    p_w = rng.uniform(-2.0e3, 2.0e3, (ny, nx, nz + 1)).astype(np.float64)
    p_w[..., 0] = 0.0
    p_w[..., -1] = 0.0
    e3t = rng.uniform(0.2, 250.0, (ny, nx, nz)).astype(np.float64)
    tmask = np.ones((ny, nx, nz), dtype=np.float64)
    tmask[0, 0, 1] = 0.0

    sum_u_core = np.roll(tracer, 1, axis=1) + tracer
    sum_u = np.concatenate([sum_u_core, sum_u_core[:, :1]], axis=1)
    sum_v = np.zeros((ny + 1, nx, nz), dtype=np.float64)
    sum_v[1:-1] = tracer[:-1] + tracer[1:]
    flux_u = (np.float64(0.5) * p_u) * sum_u
    flux_v = (np.float64(0.5) * p_v) * sum_v
    delta_u = flux_u[:, 1:] - flux_u[:, :-1]
    delta_v = flux_v[1:] - flux_v[:-1]
    r1_area = np.float64(1.0) / area[..., None]
    expected = -(((delta_u + delta_v) * r1_area) / e3t)
    sum_w = tracer[..., :-1] + tracer[..., 1:]
    wmask_inner = tmask[..., 1:] * tmask[..., :-1]
    flux_w = np.zeros((ny, nx, nz + 1), dtype=np.float64)
    flux_w[..., 1:-1] = (
        (np.float64(0.5) * p_w[..., 1:-1]) * sum_w) * wmask_inner
    expected = expected - (
        (flux_w[..., :-1] - flux_w[..., 1:]) * r1_area) / e3t

    actual = np.asarray(jax.jit(
        lambda tr, pu, pv, pw, h, tm: model_module._nemo_cen2_tracer_rhs(
            tr, pu, pv, pw, h, tm, grid))(
        jnp.asarray(tracer), jnp.asarray(p_u), jnp.asarray(p_v),
        jnp.asarray(p_w), jnp.asarray(e3t), jnp.asarray(tmask)))
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))

    # Non-vacuity: the old path materialised averages, cancelled them with 2,
    # and multiplied a precomputed inverse thickness.
    avg_u_core = np.float64(0.5) * (
        np.roll(tracer, 1, axis=1) + tracer)
    avg_u = np.concatenate([avg_u_core, avg_u_core[:, :1]], axis=1)
    avg_v = np.zeros_like(sum_v)
    avg_v[1:-1] = np.float64(0.5) * (tracer[:-1] + tracer[1:])
    legacy_fu = (np.float64(0.5) * p_u) * (np.float64(2.0) * avg_u)
    legacy_fv = (np.float64(0.5) * p_v) * (np.float64(2.0) * avg_v)
    legacy = -(
        ((legacy_fu[:, 1:] - legacy_fu[:, :-1])
         + (legacy_fv[1:] - legacy_fv[:-1])) * r1_area
    ) * (np.float64(1.0) / e3t)
    unmasked_flux_w = np.pad(
        (np.float64(0.5) * p_w[..., 1:-1]) * sum_w,
        ((0, 0), (0, 0), (1, 1)),
    )
    legacy = legacy - (
        (unmasked_flux_w[..., :-1] - unmasked_flux_w[..., 1:])
        * r1_area
    ) * (np.float64(1.0) / e3t)
    assert not np.array_equal(legacy.view(np.uint64), expected.view(np.uint64))


def test_nemo_ws_stage_corrected_velocity_matches_oracle_bits():
    """Pin GYRE V2 ``stprk3_stg`` zub/zvb source operands under JIT."""
    set_policy(PrecisionPolicy.fp64())
    transport = jnp.asarray([
        float.fromhex("-0x1.eb51b28e0faccp-5"),
        float.fromhex("-0x1.bac2b9be438a8p-2"),
    ])
    inverse_depth = jnp.asarray([
        float.fromhex("0x1.e7a1168688fe6p-13"),
        float.fromhex("0x1.e7a1168688fe6p-13"),
    ])
    actual = np.asarray(jax.jit(model_module._nemo_stage_corrected_velocity)(
        jnp.zeros((2, 1)), transport, inverse_depth, jnp.zeros((2,)),
        jnp.ones((2, 1)),
    ))[:, 0]
    expected = np.asarray([
        float.fromhex("-0x1.d3eeca2462186p-17"),
        float.fromhex("-0x1.a5af856296fbfp-14"),
    ])
    np.testing.assert_array_equal(
        actual.view(np.uint64), expected.view(np.uint64))


def test_nemo_ws_eos_hpg_transport_are_bitwise_equal_eager_and_jit(monkeypatch):
    """The new literal operand chain has one fp64 result in both regimes."""
    set_policy(PrecisionPolicy.fp64())
    dtype = jnp.float64
    grid = create_beta_plane_cgrid_geometry(
        3, 4, dx_m=8.0, dy_m=8.0, f0=0.0, beta=0.0,
        cartesian_pseudo_lat=True)
    rng = np.random.default_rng(4)

    def fixed(shape, lower, upper):
        return jnp.asarray(rng.uniform(lower, upper, shape), dtype=dtype)

    eta_transport = fixed((3, 4), -0.7, 0.7)
    e3u_transport = fixed((3, 4, 2), 0.1, 100.0)
    e3v_transport = fixed((3, 4, 2), 0.1, 100.0)
    hu_transport = fixed((3, 4), 1.0, 300.0)
    hv_transport = fixed((3, 4), 1.0, 300.0)
    area_t = fixed((3, 4), 0.1, 1000.0)
    area_u = fixed((3, 4), 0.1, 1000.0)
    area_v = fixed((3, 4), 0.1, 1000.0)

    def operand_chain():
        # EOS: live NEMO rho0 and nonzero geometric depths make the subtractive
        # density-anomaly cancellation active.  The former rho0=1 smoke input
        # could not see the production GYRE eager/JIT defect.
        temperature = jnp.asarray([-2.0, 4.25, 11.0], dtype=dtype)
        salinity = jnp.asarray([34.0, 35.0, 36.5], dtype=dtype)
        depth = jnp.asarray([0.0, 1000.0, 4321.0], dtype=dtype)
        tmask = jnp.asarray([1.0, 1.0, 0.0], dtype=dtype)
        density = nemo_roquet_density_anomaly_ratio(
            temperature, salinity, jnp.zeros_like(depth), rho0=1026.0,
            geometric_depth_m=depth, tmask=tmask,
            return_intermediates=True)

        # HPG: exercise the complete top-down recurrence on two levels.
        rhd = jnp.arange(24, dtype=dtype).reshape(3, 4, 2) / 8.0
        e3w = jnp.full_like(rhd, 2.0)
        gdept = jnp.broadcast_to(jnp.asarray([1.0, 3.0], dtype=dtype), rhd.shape)
        hpg_u, hpg_v = nemo_hpg_sco_literal_cgrid(
            rhd, e3w, gdept, grid, 8.0)

        # Transport geometry: awkward fp64 literals make the association live
        # while remaining deterministic and small.
        shape3 = (3, 4, 2)
        mask3 = jnp.ones(shape3, dtype=dtype)
        face = vertical_module.nemo_qco_live_face_geometry_from_operands(
            eta_transport, e3u_transport, e3v_transport,
            mask3, mask3,
            hu_transport, hv_transport, area_t, area_u, area_v,
        )
        return (*density, hpg_u, hpg_v, *face)

    with jax.disable_jit():
        eager = tuple(np.asarray(value) for value in operand_chain())
    compiled = tuple(np.asarray(value) for value in jax.jit(operand_chain)())
    assert all(np.array_equal(a, b) for a, b in zip(eager, compiled, strict=True))
    assert compiled[12][2] == 0.0

    # EOS non-vacuity: replacing the source-operation materialization with
    # the HLO-only barrier reproduces the contraction/reassociation defect.
    jax.clear_caches()
    with monkeypatch.context() as patch:
        patch.setattr(
            eos_module, "nemo_source_round", jax.lax.optimization_barrier)
        reassociated = tuple(
            np.asarray(value) for value in jax.jit(operand_chain)())
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(compiled[:13], reassociated[:13], strict=True)
    )

    # Planted mutation: removing the shared source-round materialization
    # changes live r3u/r3v bits.  The old optimization_barrier-only plant is
    # obsolete because XLA strips those barriers from optimized HLO.
    jax.clear_caches()
    with monkeypatch.context() as patch:
        patch.setattr(vertical_module, "nemo_source_round", lambda x: x)
        unbarriered = tuple(
            np.asarray(value) for value in jax.jit(operand_chain)())
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(compiled, unbarriered, strict=True)
    )


def test_nemo_teos10_prd_oracle_bit_patterns_without_external_data():
    """Pin production-JIT prd bits sampled from the GYRE oracle record.

    The five tuples are literal wet-cell inputs from the config-local
    round-12 stage-2 ``eos_insitu`` record.  Keeping the inputs and expected
    IEEE-754 words here makes this a hermetic CI guard: it needs neither the
    oracle executable nor ``/data`` and will catch a future XLA peephole that
    sees through the source-rounding identity.
    """
    set_policy(PrecisionPolicy.fp64())
    temperature = jnp.asarray([
        23.460942698185985,
        23.460942084268694,
        22.130311366989037,
        13.42386253495835,
        4.000532963209061,
    ], dtype=jnp.float64)
    salinity = jnp.asarray([
        36.83805761571642,
        36.83805759257495,
        36.79622408154826,
        35.74386965269002,
        35.12000012396704,
    ], dtype=jnp.float64)
    depth = jnp.asarray([
        4.9752647763236375,
        4.975265189781595,
        60.73198233881616,
        547.0295551893229,
        4150.281785429398,
    ], dtype=jnp.float64)
    expected_words = np.asarray([
        0xBF4E8AA111773000,
        0xBF4E8AA0B9A4D800,
        0xBF43E05B3BCB1400,
        0x3F442E7694E65800,
        0x3F5BF8D85E2A7400,
    ], dtype=np.uint64)

    prd = np.asarray(jax.jit(lambda t, s, z:
        nemo_teos10_density_anomaly_ratio(
            t, s, jnp.zeros_like(z), rho0=1026.0,
            geometric_depth_m=z, tmask=jnp.ones_like(z))
    )(temperature, salinity, depth))
    np.testing.assert_array_equal(prd.view(np.uint64), expected_words)


def test_nemo_hpg_literal_component_exposure_is_the_production_sum():
    """The WRITE-only HPG seam exposes, but cannot re-evaluate, zhpi+zuap."""
    set_policy(PrecisionPolicy.fp64())
    grid = create_beta_plane_cgrid_geometry(
        3, 4, dx_m=7.0, dy_m=11.0, f0=0.0, beta=0.0,
        cartesian_pseudo_lat=True)
    rhd = jnp.arange(36, dtype=jnp.float64).reshape(3, 4, 3) / 13.0
    e3w = jnp.arange(36, dtype=jnp.float64).reshape(3, 4, 3) / 17.0 + 1.0
    gdept = jnp.cumsum(e3w, axis=-1)

    values = jax.jit(lambda: nemo_hpg_sco_literal_cgrid(
        rhd, e3w, gdept, grid, 8.0, return_components=True))()
    sum_u, sum_v, zhpi_u, zhpj_v, zuap_u, zvap_v = (
        np.asarray(value) for value in values)
    np.testing.assert_array_equal(sum_u, zhpi_u + zuap_u)
    np.testing.assert_array_equal(sum_v, zhpj_v + zvap_v)


def test_nemo_hpg_consumer_keeps_direct_acceleration_bits():
    """dynhpg writes acceleration; a rho multiply/divide round trip is forbidden."""
    direct = jnp.asarray([3.812364514570218e-06], dtype=jnp.float64)
    pressure = -jnp.float64(1026.0) * direct
    reconstructed = -pressure / jnp.float64(1026.0)
    assert not np.array_equal(
        np.asarray(reconstructed).view(np.uint64),
        np.asarray(direct).view(np.uint64),
    )
    got = jax.jit(
        pe_module._nemo_hpg_tendency_from_pressure_or_direct,
    )(pressure, jnp.float64(1026.0), direct)
    np.testing.assert_array_equal(
        np.asarray(got).view(np.uint64), np.asarray(direct).view(np.uint64))


def test_nemo_qco_gdept_z0_oracle_bit_pattern():
    """Pin a live GYRE stage-2 cell where XLA fused multiply-subtract."""
    t_depth = jnp.asarray([60.731977023408945], dtype=jnp.float64)
    stretch = jnp.asarray([[0.9999998965725583]], dtype=jnp.float64)
    eta = jnp.asarray([[-0.0004448114346508057]], dtype=jnp.float64)
    got = jax.jit(pe_module._nemo_qco_gdept_z0)(t_depth, stretch, eta)
    assert np.asarray(got).view(np.uint64).item() == 0x404E5DBFCAF8A971
    fused = jax.jit(lambda z, r, ssh: z * r - ssh[..., None])(
        t_depth, stretch, eta)
    assert np.asarray(fused).view(np.uint64).item() == 0x404E5DBFCAF8A972


def test_nemo_hpg_source_rounding_arm_is_non_vacuous():
    """The private pre-fix association changes live HPG bits."""
    set_policy(PrecisionPolicy.fp64())
    grid = create_beta_plane_cgrid_geometry(
        3, 4, dx_m=7.123456789, dy_m=11.987654321,
        f0=0.0, beta=0.0, cartesian_pseudo_lat=True,
    )
    rng = np.random.default_rng(0)
    rhd = jnp.asarray(rng.normal(0.0, 1.0e-3, (3, 4, 5)), dtype=jnp.float64)
    e3w = jnp.asarray(rng.uniform(0.1, 100.0, (3, 4, 5)), dtype=jnp.float64)
    gdept = jnp.cumsum(e3w, axis=-1) + jnp.asarray(
        rng.normal(0.0, 0.1, (3, 4, 5)), dtype=jnp.float64)

    rounded = jax.jit(lambda r, e, d: nemo_hpg_sco_literal_cgrid(
        r, e, d, grid, constants.g))(rhd, e3w, gdept)
    pre_fix = jax.jit(lambda r, e, d: nemo_hpg_sco_literal_cgrid(
        r, e, d, grid, constants.g, _source_round=False))(rhd, e3w, gdept)

    assert any(
        not np.array_equal(np.asarray(actual), np.asarray(control))
        for actual, control in zip(rounded, pre_fix, strict=True)
    )


def test_nemo_hpg_literal_consumer_bit_patterns():
    """Pin source-rounded HPG bits on a fixed, non-vacuous operand set."""
    set_policy(PrecisionPolicy.fp64())
    grid = create_beta_plane_cgrid_geometry(
        3, 4, dx_m=7.123456789, dy_m=11.987654321,
        f0=0.0, beta=0.0, cartesian_pseudo_lat=True,
    )
    rhd = jnp.asarray(
        np.arange(1, 37, dtype=np.float64).reshape(3, 4, 3) * 1.0e-4)
    e3w = jnp.asarray(np.asarray([
        [[1, 2, 4], [3, 5, 7], [11, 13, 17], [19, 23, 29]],
        [[31, 37, 41], [43, 47, 53], [59, 61, 67], [71, 73, 79]],
        [[83, 89, 97], [101, 103, 107], [109, 113, 127], [131, 137, 139]],
    ], dtype=np.float64) / 7.0)
    gdept = jnp.cumsum(e3w, axis=-1) + jnp.asarray(
        np.linspace(-0.03, 0.04, 36).reshape(3, 4, 3))
    hpg_u, hpg_v = jax.jit(lambda r, e, d: nemo_hpg_sco_literal_cgrid(
        r, e, d, grid, constants.g))(rhd, e3w, gdept)
    selected = np.asarray([
        hpg_u[0, 1, 0], hpg_u[1, 2, 1], hpg_u[2, 3, 2],
        hpg_v[1, 0, 0], hpg_v[2, 2, 1], hpg_v[3, 3, 2],
    ], dtype=np.float64)
    expected_words = np.asarray([
        0xBEE04A64EC095760, 0xBF581374E675D8B0, 0xBF84809308568370,
        0x3F1F2C61EDE1BB80, 0xBF842AAC3393E35A, 0x3FA2D2A9AB3190BE,
    ], dtype=np.uint64)
    np.testing.assert_array_equal(selected.view(np.uint64), expected_words)

    # The former unrounded association changes five of these six words, so
    # this pin fails rather than silently accepting a removed helper.
    raw_u, raw_v = jax.jit(lambda r, e, d: nemo_hpg_sco_literal_cgrid(
        r, e, d, grid, constants.g, _source_round=False))(rhd, e3w, gdept)
    raw_selected = np.asarray([
        raw_u[0, 1, 0], raw_u[1, 2, 1], raw_u[2, 3, 2],
        raw_v[1, 0, 0], raw_v[2, 2, 1], raw_v[3, 3, 2],
    ], dtype=np.float64)
    assert not np.array_equal(raw_selected.view(np.uint64), expected_words)


def test_nemo_ws_public_step_is_same_production_kernel_under_outer_disable_jit():
    """A diagnostic outer context cannot bypass the production step JIT."""
    set_policy(PrecisionPolicy.fp64())
    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config._replace(
        freshwater_closure="real_freshwater", fix_eta_drift=True)
    model = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg)
    model.prime_step_caches(card.recipe.initial_state)

    with jax.disable_jit():
        eager_context = model.step(card.recipe.initial_state, dt=card.dt_s)
    production_context = model.step(card.recipe.initial_state, dt=card.dt_s)

    for field in ("T", "S", "u", "v", "eta"):
        np.testing.assert_array_equal(
            np.asarray(getattr(eager_context, field).data),
            np.asarray(getattr(production_context, field).data),
        )


def test_rk3_tke_nbb_mapping_is_independent_of_evd_selector():
    """stprk3 Nbb is the entry tracer for each independent N2 consumer."""
    card = build_nemo_testcase_card("GYRE-zco")
    cfg = card.recipe.model_config
    tke = cfg.physics.vertical_mixing.tke._replace(
        tke_n2_time_level="nemo_before")
    evd = cfg.physics.convection.enhanced_diffusion._replace(
        evd_n2_time_level="solver_state")
    cfg = cfg._replace(
        momentum_time_integrator="rk3_ws",
        physics=cfg.physics._replace(
            vertical_mixing=cfg.physics.vertical_mixing._replace(tke=tke),
            convection=cfg.physics.convection._replace(
                enhanced_diffusion=evd)))
    model = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg)
    T_nbb, S_nbb = model._n2_nemo_before_tracers(
        card.recipe.initial_state)
    np.testing.assert_array_equal(T_nbb, card.recipe.initial_state.T.data)
    np.testing.assert_array_equal(S_nbb, card.recipe.initial_state.S.data)


def test_nemo_ws_microselectors_are_not_public_config():
    """NEMO exposes no internal RK3 switches; neither may legoESM."""
    set_policy(PrecisionPolicy.fp64())
    grid = create_latlon_grid(4, 8, dtype=jnp.float64)
    z_coord = create_ocean_z_star(n_levels=2, H_max=20.0)

    fields = LatLonCGridOceanConfig._fields
    assert "tracer_rk3_transport_time_levels" not in fields
    assert "tracer_fct_low_order_predictor" not in fields
    assert "rk3_ws_stage_barotropic_correction" not in fields
    assert "rk3_ws_momentum_transport_reconcile" not in fields
    assert "primary_transport_average" not in fields


def test_nemo_ws_private_stage_velocity_exposure_is_diagnostic_only():
    """The fidelity seam returns Kaa stage velocity without changing tracers."""
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    normal = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config).step(
            card.recipe.initial_state, dt=card.dt_s)
    stage1 = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            expose_momentum_stage=1)).step(
                card.recipe.initial_state, dt=card.dt_s)
    assert np.max(np.abs(
        np.asarray(stage1.u.data) - np.asarray(normal.u.data))) > 1.0e-12
    # T-unchanged guards tracer wiring only: the hook writes u/v slots alone;
    # its structural post-step placement is the stage-noninterference guarantee.
    np.testing.assert_array_equal(
        np.asarray(stage1.T.data), np.asarray(normal.T.data))
    with pytest.raises(ValueError, match="expose_momentum_stage"):
        model_module.LatLonCGridOceanModel(
            card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
            _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
                expose_momentum_stage=4))


def test_nemo_sco_missing_raw_e3w_fails_closed():
    """A NEMO SCO card may not reconstruct W geometry from T midpoints."""
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    missing = card.recipe.z_coord._replace(
        nemo_e3w_0=None, nemo_e3w_mesh_reference=False)
    model = model_module.LatLonCGridOceanModel(
        card.recipe.grid, missing, card.recipe.model_config)
    with jax.disable_jit(), pytest.raises(
            ValueError, match="requires the raw NEMO nemo_e3w_0"):
        model.step(card.recipe.initial_state, dt=card.dt_s)


def test_nemo_overflow_primary_transport_average_is_source_bound_and_live():
    """Flux-form RK3 uses NEMO's transport primary; only the test hook ablates."""
    set_policy(PrecisionPolicy.fp64())
    card = build_overflow_zps_card()
    faithful = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config).step(
            card.recipe.initial_state, dt=card.dt_s)
    ablated = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            primary_transport_average=False)).step(
                card.recipe.initial_state, dt=card.dt_s)
    assert np.max(np.abs(
        np.asarray(faithful.u.data) - np.asarray(ablated.u.data))) > 1.0e-12
    assert np.max(np.abs(
        np.asarray(faithful.T.data) - np.asarray(ablated.T.data))) > 0.0


def test_nemo_flux_form_external_update_is_literal_and_depth_sensitive():
    """dynspg_ts.F90:731-761 uses five distinct face-depth operands."""
    velocity = jnp.asarray([[0.25, -0.5]], dtype=jnp.float64)
    h_entry = jnp.asarray([[10.0, 11.0]], dtype=jnp.float64)
    h_pgf = jnp.asarray([[9.0, 12.0]], dtype=jnp.float64)
    h_mid = jnp.asarray([[8.0, 13.0]], dtype=jnp.float64)
    h_kmm = jnp.asarray([[7.0, 14.0]], dtype=jnp.float64)
    h_exit = jnp.asarray([[6.0, 15.0]], dtype=jnp.float64)
    pgf = jnp.asarray([[0.1, -0.2]], dtype=jnp.float64)
    transport = jnp.asarray([[0.3, 0.4]], dtype=jnp.float64)
    slow = jnp.asarray([[-0.5, 0.6]], dtype=jnp.float64)
    wet = jnp.asarray([[1.0, 0.0]], dtype=jnp.float64)
    dt = jnp.asarray(2.0, dtype=jnp.float64)
    actual = _nemo_flux_form_external_velocity_update(
        velocity, h_entry, h_pgf, h_mid, h_kmm, h_exit,
        pgf, transport, slow, dt, wet, jnp.asarray(1.0e-10))
    expected = (
        h_entry * velocity
        + dt * (h_pgf * pgf + h_mid * transport + h_kmm * slow)
    ) / h_exit * wet
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    # Synthetic violation: replacing the midpoint depth by the entry depth is
    # not an algebraic no-op and must move a live wet face.
    violated = _nemo_flux_form_external_velocity_update(
        velocity, h_entry, h_pgf, h_entry, h_kmm, h_exit,
        pgf, transport, slow, dt, wet, jnp.asarray(1.0e-10))
    assert not np.array_equal(np.asarray(actual), np.asarray(violated))


def test_nemo_two_step_fct_is_a_live_real_flux_arm():
    """The source-ordered FCT predictor must move a nontrivial WS result."""
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    cfg_two = card.recipe.model_config
    state_two = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_two).step(
            card.recipe.initial_state, dt=card.dt_s)
    state_one = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_two,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            two_step_fct_predictor=False)).step(
            card.recipe.initial_state, dt=card.dt_s)
    movement = np.max(np.abs(
        np.asarray(state_two.T.data) - np.asarray(state_one.T.data)))
    assert movement > 1.0e-12


def test_nemo_ws_tracer_stage_polynomial(monkeypatch):
    rate = 0.2

    def linear_flux_pair(a, b, *args, **kwargs):
        zeros_a = jnp.zeros_like(a)
        zeros_b = jnp.zeros_like(b)
        return (rate * a, zeros_a), (rate * b, zeros_b)

    monkeypatch.setattr(
        model_module, "compute_advection_flux_div_pair", linear_flux_pair)
    a0 = jnp.array([[[2.0]]], dtype=jnp.float64)
    b0 = jnp.array([[[3.0]]], dtype=jnp.float64)
    ones = jnp.ones_like(a0)
    a, b = model_module._nemo_ws_rk3_tracer_pair_step(
        a0, b0, "centered", ones, ones, jnp.ones((1, 1, 2)),
        ones, ones, ones, ones, object(), 0.5, ones,
    )
    z = -0.5 * rate
    amplification = 1.0 + z + z * z / 2.0 + z * z * z / 6.0
    np.testing.assert_allclose(np.asarray(a), 2.0 * amplification, rtol=0, atol=2e-16)
    np.testing.assert_allclose(np.asarray(b), 3.0 * amplification, rtol=0, atol=2e-16)


def test_nemo_ws_tracer_zero_flux_is_exact_identity(monkeypatch):
    def zero_flux_pair(a, b, *args, **kwargs):
        return (jnp.zeros_like(a), jnp.zeros_like(a)), (
            jnp.zeros_like(b), jnp.zeros_like(b))

    monkeypatch.setattr(
        model_module, "compute_advection_flux_div_pair", zero_flux_pair)
    tracer = jnp.array([[[1.25, -2.0]]], dtype=jnp.float64)
    h = jnp.ones_like(tracer)
    got_a, got_b = model_module._nemo_ws_rk3_tracer_pair_step(
        tracer, tracer, "centered", h, h, jnp.ones((1, 1, 3)),
        h, h, h, h, object(), 2.0, h,
    )
    np.testing.assert_array_equal(np.asarray(got_a), np.asarray(tracer))
    np.testing.assert_array_equal(np.asarray(got_b), np.asarray(tracer))


def test_nemo_ws_stage1_trace_exposes_live_rhs_boundaries(monkeypatch):
    """The WRITE-only trace returns post-adv and post-SBC Krhs, not copies."""
    rate = 0.125

    def linear_flux_pair(a, b, *args, **kwargs):
        zeros = jnp.zeros_like(a)
        return (rate * a, zeros), (rate * b, zeros)

    monkeypatch.setattr(
        model_module, "compute_advection_flux_div_pair", linear_flux_pair)
    a0 = jnp.array([[[2.0]]], dtype=jnp.float64)
    b0 = jnp.array([[[3.0]]], dtype=jnp.float64)
    ones = jnp.ones_like(a0)
    source_a = jnp.full_like(a0, 0.75)
    source_b = jnp.full_like(b0, -0.5)
    result = model_module._nemo_ws_rk3_tracer_pair_step(
        a0, b0, "centered", ones, ones, jnp.ones((1, 1, 2)),
        ones, ones, ones, ones, object(), 0.5, ones,
        stage_source_rates=((source_a, source_b),) * 3,
        stop_after_stage=1, return_stage1_trace=True,
    )
    a1, b1, adv_a, adv_b, sbc_a, sbc_b = (
        np.asarray(value) for value in result)
    np.testing.assert_array_equal(adv_a, -rate * np.asarray(a0))
    np.testing.assert_array_equal(adv_b, -rate * np.asarray(b0))
    np.testing.assert_array_equal(sbc_a, adv_a + np.asarray(source_a))
    np.testing.assert_array_equal(sbc_b, adv_b + np.asarray(source_b))
    assert not np.array_equal(a1, adv_a)
    assert not np.array_equal(b1, adv_b)


def test_nemo_ws_stage1_trace_preserves_source_statement_order(monkeypatch):
    """EMP and runoff must enter Krhs as two ordered additions."""
    def large_advective_rhs(a, b, *args, **kwargs):
        fd = jnp.full_like(a, -1.0e16)
        zero = jnp.zeros_like(a)
        return (fd, zero), (fd, zero)

    monkeypatch.setattr(
        model_module, "compute_advection_flux_div_pair", large_advective_rhs)
    tracer = jnp.ones((1, 1, 1), dtype=jnp.float64)
    ones = jnp.ones_like(tracer)
    emp = jnp.full_like(tracer, -1.0e16)
    runoff = jnp.ones_like(tracer)
    combined = emp + runoff
    source_rates = ((combined, combined),) * 3
    source_terms = (((emp, runoff), (emp, runoff)),) * 3

    ordered = model_module._nemo_ws_rk3_tracer_pair_step(
        tracer, tracer, "centered", ones, ones, jnp.ones((1, 1, 2)),
        ones, ones, ones, ones, object(), 1.0, ones,
        stage_source_rates=source_rates, stage_source_terms=source_terms,
        stop_after_stage=1, return_stage1_trace=True,
    )
    regrouped = model_module._nemo_ws_rk3_tracer_pair_step(
        tracer, tracer, "centered", ones, ones, jnp.ones((1, 1, 2)),
        ones, ones, ones, ones, object(), 1.0, ones,
        stage_source_rates=source_rates,
        stop_after_stage=1, return_stage1_trace=True,
    )
    np.testing.assert_array_equal(np.asarray(ordered[4]), 1.0)
    np.testing.assert_array_equal(np.asarray(ordered[5]), 1.0)
    np.testing.assert_array_equal(np.asarray(regrouped[4]), 0.0)
    np.testing.assert_array_equal(np.asarray(regrouped[5]), 0.0)


def test_nemo_ws_qco_tracer_assignment_preserves_source_order(monkeypatch):
    """The executing stage assignment keeps NEMO's multiply/add barriers."""
    set_policy(PrecisionPolicy.fp64())
    kbb = jnp.asarray([[[-0.6333231084168918]]], dtype=jnp.float64)
    krhs = jnp.asarray([[[1.5175620838095593e-08]]], dtype=jnp.float64)
    active = jnp.ones_like(kbb)
    qbb = jnp.asarray([[1.0 - 0.00014940245890372877]], dtype=jnp.float64)
    qmm = qbb
    qaa = jnp.asarray([[1.0 - 0.00020466918852310662]], dtype=jnp.float64)
    weights = ((qbb, qmm, qaa),) * 3
    grid = SimpleNamespace(
        dy_u=jnp.ones((1, 2), dtype=jnp.float64),
        dx_v=jnp.ones((2, 1), dtype=jnp.float64),
        area_T=jnp.ones((1, 1), dtype=jnp.float64),
    )
    mass_u = jnp.ones((1, 2, 1), dtype=jnp.float64)
    mass_v = jnp.ones((2, 1, 1), dtype=jnp.float64)
    w = jnp.ones((1, 1, 2), dtype=jnp.float64)

    monkeypatch.setattr(
        model_module, "_nemo_cen2_tracer_rhs",
        lambda *args, **kwargs: krhs)

    def evaluate(base):
        return model_module._nemo_ws_rk3_tracer_pair_step(
            base, base, "fct2", mass_u, mass_v, w,
            active, active, mass_u, mass_v, grid, 10800.0, active,
            stage_qco_weights=weights, stop_after_stage=1,
        )[0]

    eager = evaluate(kbb)
    compiled = jax.jit(evaluate)(kbb)
    expected = np.asarray([[[-0.6333034820241417]]], dtype=np.float64)
    np.testing.assert_array_equal(np.asarray(eager), expected)
    np.testing.assert_array_equal(np.asarray(compiled), expected)

    fused = jax.jit(lambda base, rhs: (
        qbb[..., None] * base
        + jnp.float64(3600.0) * qmm[..., None] * rhs
    ) / qaa[..., None])(kbb, krhs)
    assert not np.array_equal(np.asarray(compiled), np.asarray(fused))

    gradient = jax.grad(lambda value: jnp.sum(evaluate(value)))(kbb)
    assert np.isfinite(np.asarray(gradient)).all()
    assert np.any(np.asarray(gradient) != 0.0)


def test_nemo_ws_stage1_boundary_hook_is_private_and_validated():
    with pytest.raises(ValueError, match="expose_tracer_stage1_boundary"):
        _lock_model(model_module._NEMOWSRK3TestHooks(
            expose_tracer_stage1_boundary="not-a-boundary"))


def test_nemo_ws_real_fct_flux_changes_on_wrong_transport_time_level():
    """Exercise real FCT geometry; a frozen final velocity must fail this pin."""
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    cfg_kmm = card.recipe.model_config
    model_kmm = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_kmm)
    model_wrong = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg_kmm,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            stage_barotropic_correction=False,
            momentum_transport_reconcile=False,
            kmm_tracer_transports=False))
    state_kmm = model_kmm.step(card.recipe.initial_state, dt=card.dt_s)
    state_wrong = model_wrong.step(card.recipe.initial_state, dt=card.dt_s)
    delta = np.max(np.abs(
        np.asarray(state_kmm.T.data) - np.asarray(state_wrong.T.data)))
    # Wrong Kaa reuse changes the real limiter/flux path by ~2.70e-5 K.
    assert delta > 2.0e-5
    np.testing.assert_array_equal(
        np.asarray(state_kmm.S.data), np.asarray(state_wrong.S.data))


def test_overflow_bbl_is_live_inside_real_rk3_stage3():
    """A planted dense shelf must activate BBL in the full RK3 solver."""
    set_policy(PrecisionPolicy.fp64())
    from legoesm.ocean.physics.bbl_adv import nemo_bbl_static_geometry
    card = build_overflow_zps_card()
    # Use the geometry the MODEL runs.  ocean_model_latlon_cgrid selects
    # nemo_bbl_static_geometry unless the legacy_bbl_partial_geometry test
    # hook is set, and the two disagree about which faces are BBL-active:
    # trabbl.F90:519-527 signs gdept_0(...,mbkt) differences and leaves
    # mgrhu = 0 when they are equal, so a ku_s == ku_d face is NEVER active
    # under NEMO's rule.  Measured on this card: nemo_bbl_static_geometry
    # gives 29 active U faces and 0 of them degenerate, while the legacy
    # partial-cell builder gives 141 active of which 112 are degenerate.  An
    # earlier version of this test called the legacy builder, drew a
    # degenerate face, measured 0.0 movement, and recorded "why is a
    # ku_s == ku_d face BBL-active at all" as an open model question.  It is
    # not a model question: the model never marks one.  Retracted, and the
    # ku_s != ku_d filter it motivated is removed -- a filter would hide the
    # disagreement instead of testing the geometry the model uses.
    zc = card.recipe.z_coord
    h_ref = jnp.asarray(zc.h_partial)
    if h_ref.ndim == 1:
        h_ref = jnp.broadcast_to(
            h_ref, card.recipe.initial_state.T.data.shape)
    geom = nemo_bbl_static_geometry(
        h_ref, card.recipe.initial_state.land_mask.data,
        zc.nemo_gdept_0, zc.nemo_bbl_e3u_0, zc.nemo_bbl_e3v_0)
    active = np.asarray(geom.u_active) > 0.5
    # Honest about what this assertion can and cannot fail on: NEMO signs a
    # 3-D gdept_0, so on a real zps mesh two columns with equal mbkt but
    # different partial-cell centroids WOULD be active.  This card feeds a 1-D
    # ladder (nemo_testcase_recipe.py OVERFLOW card) and usrdef_zgr keeps
    # pdept uniform, so here it cannot fail -- it is a theorem, kept as a
    # regression pin against a future 3-D gdept and as documentation of the
    # census.  It would bind on ORCA2.
    assert not (active & (np.asarray(geom.ku_s) == np.asarray(geom.ku_d))).any()
    active_faces = np.argwhere(active)
    assert active_faces.size
    j, i = active_faces[len(active_faces) // 2]
    slope = int(np.asarray(geom.mgrhu)[j, i])
    shelf_i, deep_i = ((i, i + 1) if slope > 0 else (i + 1, i))
    ks = int(np.asarray(geom.ku_s)[j, i])
    kd = int(np.asarray(geom.ku_d)[j, i])
    initial = card.recipe.initial_state
    T = initial.T.data.at[j, shelf_i, ks].set(0.0)
    T = T.at[j, deep_i, kd].set(20.0)
    S = initial.S.data.at[j, shelf_i, ks].set(36.0)
    S = S.at[j, deep_i, kd].set(35.0)
    initial = initial._replace(
        T=initial.T.replace(data=T), S=initial.S.replace(data=S))
    on = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config).step(
            initial, dt=card.dt_s)
    off = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=model_module._NEMOWSRK3TestHooks(
            disable_bbl=True)).step(initial, dt=card.dt_s)
    movement = np.max(np.abs(
        np.asarray(on.T.data) - np.asarray(off.T.data)))
    assert movement > 1.0e-12
    # BBL is a closed three-leg exchange; its isolated stage contribution
    # must not change the thickness-weighted domain tracer content.
    h = np.asarray(card.recipe.z_coord.h_partial)
    area = np.asarray(card.recipe.grid.area_T)[..., None]
    content_delta = np.sum(
        area * h * (np.asarray(on.T.data) - np.asarray(off.T.data)))
    content_scale = np.sum(
        np.abs(area * h * np.asarray(initial.T.data)))
    assert abs(content_delta) <= 1024.0 * np.finfo(np.float64).eps * content_scale


def test_nemo_ws_fct_limits_stage3_only_and_centres_stages_1_2(monkeypatch):
    """traadv.F90:281-282,361-364: FCT at kstg=3 only; cen2 at stages 1-2."""
    seen = []

    def recording_flux_pair(a, b, scheme, *args, **kwargs):
        seen.append((scheme, "tr_a_before" in kwargs))
        return (jnp.zeros_like(a), jnp.zeros_like(a)), (
            jnp.zeros_like(b), jnp.zeros_like(b))

    monkeypatch.setattr(
        model_module, "compute_advection_flux_div_pair", recording_flux_pair)
    base_grid = create_latlon_grid(2, 2, dtype=jnp.float64)
    grid = SimpleNamespace(
        **base_grid._asdict(),
        dy_u=jnp.ones((2, 3), dtype=jnp.float64),
        dx_v=jnp.ones((3, 2), dtype=jnp.float64),
        area_T=jnp.asarray(base_grid.area),
    )
    tracer = jnp.arange(8, dtype=jnp.float64).reshape(2, 2, 2) + 1.0
    h = jnp.ones_like(tracer)
    mf_u = jnp.ones((2, 3, 2), dtype=jnp.float64)
    mf_v = jnp.ones((3, 2, 2), dtype=jnp.float64)
    w = jnp.ones((2, 2, 3), dtype=jnp.float64)
    model_module._nemo_ws_rk3_tracer_pair_step(
        tracer, tracer, "fct2", mf_u, mf_v, w,
        h, h, mf_u, mf_v, grid, 2.0, h,
    )
    # Stages 1-2 execute the literal inline CEN2 accumulator; only stage 3
    # reaches the shared FCT dispatcher.
    assert seen == [("fct2", True)]
    seen.clear()
    model_module._nemo_ws_rk3_tracer_pair_step(
        tracer, tracer, "centered", mf_u, mf_v, w,
        h, h, mf_u, mf_v, grid, 2.0, h,
    )
    # Non-FCT NEMO schemes run the same operator at every stage.
    assert [scheme for scheme, _ in seen] == ["centered"] * 3


def test_nemo_ws_tracer_resume_reuses_the_handed_stages(monkeypatch):
    """One stage ladder: resume=(k, a_k, b_k) skips stages <= k exactly."""
    rate = 0.2
    calls = []

    def linear_flux_pair(a, b, *args, **kwargs):
        calls.append(1)
        return (rate * a, jnp.zeros_like(a)), (rate * b, jnp.zeros_like(b))

    monkeypatch.setattr(
        model_module, "compute_advection_flux_div_pair", linear_flux_pair)
    a0 = jnp.array([[[2.0]]], dtype=jnp.float64)
    b0 = jnp.array([[[3.0]]], dtype=jnp.float64)
    ones = jnp.ones_like(a0)
    common = (ones, ones, jnp.ones((1, 1, 2)), ones, ones, ones, ones, object(), 0.5, ones)
    full = model_module._nemo_ws_rk3_tracer_pair_step(a0, b0, "centered", *common)
    assert len(calls) == 3
    calls.clear()
    s1 = model_module._nemo_ws_rk3_tracer_pair_step(
        a0, b0, "centered", *common, stop_after_stage=1)
    s2 = model_module._nemo_ws_rk3_tracer_pair_step(
        a0, b0, "centered", *common, stop_after_stage=2, resume=(1, *s1))
    out = model_module._nemo_ws_rk3_tracer_pair_step(
        a0, b0, "centered", *common, resume=(2, *s2))
    assert len(calls) == 3          # 1 + 1 + 1: no stage evaluated twice
    np.testing.assert_array_equal(np.asarray(out[0]), np.asarray(full[0]))
    np.testing.assert_array_equal(np.asarray(out[1]), np.asarray(full[1]))
    # A planted wrong stage-2 operand must change the answer (non-vacuous).
    planted = model_module._nemo_ws_rk3_tracer_pair_step(
        a0, b0, "centered", *common, resume=(2, s2[0] + 1.0, s2[1]))
    assert float(np.abs(np.asarray(planted[0]) - np.asarray(full[0])).max()) > 0.0
    with pytest.raises(ValueError, match="resume"):
        model_module._nemo_ws_rk3_tracer_pair_step(
            a0, b0, "centered", *common, stop_after_stage=1, resume=(1, *s1))


def _lock_step(hooks=None, **cfg_overrides):
    set_policy(PrecisionPolicy.fp64())
    card = build_lock_exchange_zco_card()
    cfg = card.recipe.model_config._replace(**cfg_overrides)
    model = model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, cfg,
        _nemo_ws_test_hooks=hooks or model_module._NEMOWSRK3TestHooks())
    return card, model.step(card.recipe.initial_state, dt=card.dt_s)


def test_nemo_ws_stage_vertical_up3_is_applied_only_under_aimp():
    """ln_zad_Aimp=.false.: the explicit vertical term already sits inside
    tendencies() (ocean_pe: ``if not _aimp_vertadv: du_dt += diag_vertadv_u``),
    so the per-stage UP3 must not add it again; the omit hook is then a no-op."""
    omit = model_module._NEMOWSRK3TestHooks(omit_stage_vertical_up3=True)
    card, on_faithful = _lock_step()
    assert card.recipe.model_config.adaptive_implicit_vertadv is True
    _, on_omit = _lock_step(omit)
    assert float(np.abs(np.asarray(on_omit.u.data) - np.asarray(on_faithful.u.data)).max()) > 0.0
    _, off_faithful = _lock_step(adaptive_implicit_vertadv=False)
    _, off_omit = _lock_step(omit, adaptive_implicit_vertadv=False)
    np.testing.assert_array_equal(np.asarray(off_omit.u.data), np.asarray(off_faithful.u.data))
    np.testing.assert_array_equal(np.asarray(off_omit.v.data), np.asarray(off_faithful.v.data))


def test_nemo_ws_exposed_tracer_stage_carries_the_stage_ssh():
    """expose_tracer_stage returns the HYB stage eta handed to eos+dyn_hpg
    (stprk3_stg.F90:146 N+1/3, :209 N+1/2), not the final eta."""
    card, entry = _tilted_entry_after_one_step()
    normal = _lock_model().step(entry, dt=card.dt_s)
    eta0 = np.asarray(entry.eta.data)
    eta_new = np.asarray(normal.eta.data)
    assert float(np.abs(eta_new - eta0).max()) > 1.0e-9
    for stage, expected in ((1, eta0 + (eta_new - eta0) / 3.0), (2, 0.5 * (eta0 + eta_new))):
        exposed = _lock_model(model_module._NEMOWSRK3TestHooks(
            expose_tracer_stage=stage)).step(entry, dt=card.dt_s)
        np.testing.assert_allclose(np.asarray(exposed.eta.data), expected, rtol=1e-14, atol=0)
        assert float(np.abs(np.asarray(exposed.eta.data) - eta_new).max()) > 1.0e-9


def test_nemo_ws_stage2_raw_exposure_precedes_mean_replacement():
    """The round-11 seam returns raw Kaa, while the ordinary step still runs."""
    card, entry = _tilted_entry_after_one_step()
    raw = _lock_model(model_module._NEMOWSRK3TestHooks(
        expose_stage2_raw_momentum=True)).step(entry, dt=card.dt_s)
    corrected = _lock_model(model_module._NEMOWSRK3TestHooks(
        expose_momentum_stage=2)).step(entry, dt=card.dt_s)
    assert float(np.max(np.abs(
        np.asarray(raw.u.data) - np.asarray(corrected.u.data)))) > 1.0e-12
    # LOCK_EXCHANGE is meridionally uniform, so its v correction is the
    # expected structural zero; the live zonal correction proves the seam.
    np.testing.assert_array_equal(
        np.asarray(raw.v.data), np.asarray(corrected.v.data))


def _lock_model(hooks=None):
    card = build_lock_exchange_zco_card()
    return model_module.LatLonCGridOceanModel(
        card.recipe.grid, card.recipe.z_coord, card.recipe.model_config,
        _nemo_ws_test_hooks=hooks or model_module._NEMOWSRK3TestHooks())


_TILTED = {}


def _tilted_entry_after_one_step():
    """LOCK's own eta stays ~1e-28 (symmetric front) and a forward-backward
    step from rest leaves eta unchanged (continuity runs on u=0 first), so a
    1 m tilt is planted and ONE step taken: the returned entry state has a
    barotropic flow, and the next step's stage ladder is non-degenerate."""
    if not _TILTED:
        set_policy(PrecisionPolicy.fp64())
        card = build_lock_exchange_zco_card()
        init = card.recipe.initial_state
        mask = np.asarray(init.land_mask.data)
        tilt = 1.0 * np.linspace(-1.0, 1.0, mask.shape[1])[None, :] * mask
        tilted = init._replace(eta=init.eta.replace(data=jnp.asarray(tilt, dtype=jnp.float64)))
        _TILTED["card"] = card
        _TILTED["entry"] = _lock_model().step(tilted, dt=card.dt_s)
    return _TILTED["card"], _TILTED["entry"]


def test_nemo_ws_split_freeze_hooks_compose_to_the_freeze_arm():
    """freeze_stage_hpg_tracers + freeze_stage_hpg_eta == freeze_stage_hpg_operands
    bit for bit, and each alone moves the step (so each is a live one-variable
    arm).  The eta arm is inert on LOCK's own ~1e-28 eta (1 + eta/H == 1.0 in
    fp64), so it is exercised on the tilted, once-stepped entry state."""
    H = model_module._NEMOWSRK3TestHooks
    _, faithful = _lock_step()
    _, frozen = _lock_step(H(freeze_stage_hpg_operands=True))
    _, both = _lock_step(H(freeze_stage_hpg_tracers=True, freeze_stage_hpg_eta=True))
    np.testing.assert_array_equal(np.asarray(both.u.data), np.asarray(frozen.u.data))
    np.testing.assert_array_equal(np.asarray(both.T.data), np.asarray(frozen.T.data))
    _, tracers = _lock_step(H(freeze_stage_hpg_tracers=True))
    assert float(np.abs(np.asarray(tracers.u.data) - np.asarray(faithful.u.data)).max()) > 0.0
    card, entry = _tilted_entry_after_one_step()
    live = _lock_model().step(entry, dt=card.dt_s)
    eta_arm = _lock_model(H(freeze_stage_hpg_eta=True)).step(entry, dt=card.dt_s)
    assert float(np.abs(np.asarray(eta_arm.u.data) - np.asarray(live.u.data)).max()) > 0.0
