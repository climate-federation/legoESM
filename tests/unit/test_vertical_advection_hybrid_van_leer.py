"""Conservative limited tracer vertical transport on the hybrid (CAM L32) lane.

The advective upwind operator the MPAS hybrid lane used for every tracer
creates water: measured +0.32..+0.36 kg/m2/day globally on the production
CAM6 run (days 30/45/55), the size of that run's own P - E + dW/dt budget gap.
These tests pin the replacement's defining properties and that the MPAS
tracer path runs it.
"""
from __future__ import annotations

import importlib.util
import pathlib

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

# Reuse the existing scoped science-test precision fixture; x64 alone does not
# choose the geometry/storage precision used by the MPAS mesh constructor.
from tests.atmosphere.hydrostatic.unit.test_mpas_vert4_flux_form import fp64_policy as fp64_policy

from legoesm.grids.vertical import (
    compute_mass_flux_from_cumsum, dp_from_hybrid, make_cam6_l32_levels,
    vertical_advection_hybrid, vertical_advection_hybrid_van_leer)


def _column(seed=0, ncol=6):
    """CAM L32 columns with varying p_s and a continuity-built mass flux."""
    rng = np.random.default_rng(seed)
    coord = make_cam6_l32_levels()
    nlev = coord.A_full.shape[0]
    p_s = jnp.asarray(1.0e5 + 3.0e3 * rng.standard_normal(ncol))
    div = jnp.asarray(1.0e-5 * rng.standard_normal((ncol, nlev)))
    cums = jnp.cumsum(div * dp_from_hybrid(coord, p_s), axis=-1)
    F = compute_mass_flux_from_cumsum(cums, cums[..., -1:], coord)
    q = jnp.asarray(1.0e-2 * np.exp(-0.5 * ((np.arange(nlev) - 24) / 3.0) ** 2)
                    * (1.0 + 0.3 * rng.random((ncol, nlev))))
    return coord, p_s, F, q


def product_rule_residual(op, coord, p_s, F, q):
    dp = dp_from_hybrid(coord, p_s)
    col = jnp.sum(dp * op(q, F, p_s, coord) - q * (F[..., 1:] - F[..., :-1]), axis=-1)
    scale = jnp.sum(jnp.abs(F[..., 1:-1] * jnp.diff(q, axis=-1)), axis=-1)
    return float(jnp.max(jnp.abs(col) / scale))


def test_column_water_telescopes_and_the_upwind_operator_does_not():
    coord, p_s, F, q = _column()
    assert product_rule_residual(vertical_advection_hybrid_van_leer, coord, p_s, F, q) < 1e-12
    assert product_rule_residual(vertical_advection_hybrid, coord, p_s, F, q) > 1e-3


def test_constant_field_has_zero_tendency():
    coord, p_s, F, _ = _column(1)
    c = jnp.full((p_s.shape[0], coord.A_full.shape[0]), 7.3e-3)
    assert float(jnp.max(jnp.abs(vertical_advection_hybrid_van_leer(c, F, p_s, coord)))) < 1e-18


def test_positive_and_bounded_under_repeated_steps():
    """A non-negative layer with sharp edges stays in [0, max] over many
    forward-Euler steps at a per-face Courant sum of ~0.3 (production max 0.14)."""
    coord, p_s, F, _ = _column(2)
    nlev = coord.A_full.shape[0]
    dp = dp_from_hybrid(coord, p_s)
    q = jnp.zeros((p_s.shape[0], nlev)).at[:, 18:22].set(1.0e-3)
    rate = (jnp.abs(F[..., :-1]) + jnp.abs(F[..., 1:])) / dp
    dt = 0.3 / float(jnp.max(rate))
    for _ in range(200):
        q = q + dt * vertical_advection_hybrid_van_leer(q, F, p_s, coord, smoothing_width=0.0, flux_smoothing_width=0.0)
    assert float(jnp.min(q)) >= -1e-18
    assert float(jnp.max(q)) <= 1.0e-3 * (1 + 1e-12)


def test_mpas_hybrid_tracer_path_runs_the_conservative_operator(monkeypatch):
    """Names the symbol the MPAS hybrid tracer path calls, and fails if the
    lane goes back to the upwind operator."""
    path = pathlib.Path(__file__).with_name("test_mpas_atmosphere.py")
    spec = importlib.util.spec_from_file_location("_mpas_atm_helpers_vl", path)
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_mpas as pe
    from legoesm.core.field import Field

    coord = make_cam6_l32_levels()
    nlev = coord.A_full.shape[0]
    mesh = h._make_mesh(level=2)
    state = h._add_perturbation_hydro(h._make_hydrostatic_state(mesh, nlev), mesh, nlev)
    from legoesm.atmosphere.standard_atmosphere import standard_atmosphere_temperature
    from legoesm.driver.config import ExperimentConfig
    from legoesm.thermo import saturation_mixing_ratio

    # Reuse the shared realistic lapse-rate/stratosphere profile and the
    # production humidity initializer, with existing seeded wind perturbations.
    temperature = standard_atmosphere_temperature(mesh.latCell, coord.sigma_full)
    state = state._replace(T=state.T.replace(data=temperature))
    pressure = state.p_s.data[..., None] * coord.sigma_full
    qv = (ExperimentConfig().rh_init * saturation_mixing_ratio(temperature, pressure)
          * coord.sigma_full**2)
    state = state._replace(tracers={"q_v": Field(
        data=qv, name="q_v", dims=("nCells", "nlev"), units="kg/kg")})
    calls = []
    real = pe.vertical_advection_hybrid_van_leer

    def spy(*a, **k):
        calls.append((a, k))
        return real(*a, **k)

    monkeypatch.setattr(pe, "vertical_advection_hybrid_van_leer", spy)
    out = pe.mpas_model_tendencies(
        state, mesh, coord, pe.MPASPrimitiveEquationConfig(nu_del2=1.0e5))
    assert calls, "the MPAS hybrid tracer path no longer calls the conservative operator"
    assert np.all(np.isfinite(np.asarray(out.tracer_tendencies["q_v"].data)))
    assert calls[0][1]["flux_smoothing_width"] == pe.MPASPrimitiveEquationConfig().tracer_van_leer_flux_width
    assert calls[0][1]["smoothing_width"] == pe.MPASPrimitiveEquationConfig().tracer_van_leer_width
    # Diagnose only the shared vertical column operator: horizontal MPAS
    # transport is outside this instantaneous paired-column claim.
    args, kwargs = calls[0]
    _, F_actual, ps_actual, coord_actual = args
    q_actual = qv
    residual = product_rule_residual(real, coord_actual, ps_actual, F_actual, q_actual)
    assert residual < 1e-12
    constant = jnp.full_like(q_actual, 7.3e-3)
    assert float(jnp.max(jnp.abs(real(constant, F_actual, ps_actual, coord_actual)))) < 1e-18
    hard = pe.mpas_model_tendencies(
        state, mesh, coord, pe.MPASPrimitiveEquationConfig(nu_del2=1.0e5, tracer_van_leer_width=0.0, tracer_van_leer_flux_width=0.0))
    smooth_t = np.asarray(out.tracer_tendencies["q_v"].data)
    hard_t = np.asarray(hard.tracer_tendencies["q_v"].data)
    gap = smooth_t - hard_t
    root_area = np.sqrt(np.asarray(mesh.areaCell))[:, None]
    norm_hard = np.linalg.norm(root_area * hard_t)
    assert norm_hard > 0.0
    relative_l2 = np.linalg.norm(root_area * gap) / norm_hard
    print(f"MPAS realistic full tendency max_abs={np.max(np.abs(gap)):.17e} "
          f"area_weighted_relative_l2={relative_l2:.17e} "
          f"paired={residual:.17e}")
    assert np.max(np.abs(gap)) > 0.0
    np.testing.assert_allclose(
        [np.max(np.abs(gap)), relative_l2],
        [3.670506864982424e-13, 6.766825774837968e-05],
        rtol=1e-6, atol=1e-12)



def test_boundary_donor_faces_keep_a_full_column_in_range():
    """Load-bearing: without the donor faces next to the top and bottom layers
    a non-constant column under steady descent leaves its starting range
    (independent review measured 1.13..1.99 -> 0.44 after 2000 steps)."""
    coord = make_cam6_l32_levels()
    nlev = coord.A_full.shape[0]
    p_s = jnp.array([1.0e5])
    dp = dp_from_hybrid(coord, p_s)
    q0 = jnp.asarray(1.13 + 0.86 * np.random.default_rng(3).random((1, nlev)))
    for sign in (1.0, -1.0):
        F = jnp.concatenate([jnp.zeros((1, 1)), jnp.full((1, nlev - 1), sign),
                             jnp.zeros((1, 1))], axis=-1)
        dt = 0.4 / float(jnp.max((jnp.abs(F[..., :-1]) + jnp.abs(F[..., 1:])) / dp))
        q = q0
        for _ in range(2000):
            q = q + dt * vertical_advection_hybrid_van_leer(q, F, p_s, coord, smoothing_width=0.0, flux_smoothing_width=0.0)
        assert float(jnp.min(q)) >= float(jnp.min(q0)) - 1e-12
        assert float(jnp.max(q)) <= float(jnp.max(q0)) + 1e-12


# Frozen output from unpatched T14 8bece12646691fa7623f88f3cceb2180c056a35b,
# generated on PBS in fp64. No second implementation of the hard operator.
_HARD_FIXTURE = pathlib.Path(__file__).with_name("fixtures") / "hybrid_van_leer_t14.npz"


def _limiter_case(case):
    coord, p_s, F, q = _column(case)
    if case == 2:
        q = jnp.zeros_like(q).at[:, 18:22].set(1.0e-3)
    elif case == 3:
        q = jnp.asarray(1.13 + 0.86 * np.random.default_rng(3).random(q.shape))
    return coord, p_s, F, q


# T18 re-measurement: W4 separates q=1e-10 kg/kg from flux=1e-6 Pa/s
# because the old 1e-6 kg/kg width produced material negative condensate faces.
# Measured max absolute tendency mismatch per column, not a worst-case bound.
_SMOOTH_GAPS = {0: [2.9747168260550255e-13,
     3.267784190844157e-13,
     3.551931092884434e-13,
     6.687833828611003e-13,
     3.17129376737286e-13,
     2.8815014650937645e-13],
 1: [2.294923465566201e-13,
     3.448407858274477e-13,
     3.8762665109210754e-13,
     6.243330938622849e-13,
     3.502406629791996e-13,
     2.3467837825354595e-13],
 2: [9.598781885805938e-14,
     9.892948730148268e-14,
     9.756718913378074e-14,
     1.0654859157285657e-13,
     9.035158346011574e-14,
     9.304165483193451e-14],
 3: [3.6982261041430196e-10,
     8.347549396535928e-10,
     1.140703555968979e-09,
     5.51852590507939e-10,
     4.797702480929074e-10,
     8.289120110598247e-10]}


@pytest.mark.parametrize("case", range(4))
def test_hard_limit_matches_t14_snapshot(case):
    coord, p_s, F, q = _limiter_case(case)
    actual = vertical_advection_hybrid_van_leer(q, F, p_s, coord, smoothing_width=0.0, flux_smoothing_width=0.0)
    with np.load(_HARD_FIXTURE) as reference:
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint64),
                                      reference[str(case)].view(np.uint64))


@pytest.mark.parametrize("case", range(4))
def test_smooth_default_recorded_gap_per_column(case):
    coord, p_s, F, q = _limiter_case(case)
    hard = vertical_advection_hybrid_van_leer(q, F, p_s, coord, smoothing_width=0.0, flux_smoothing_width=0.0)
    smooth = vertical_advection_hybrid_van_leer(q, F, p_s, coord)
    gap = np.max(np.abs(np.asarray(smooth - hard)), axis=-1)
    assert np.max(gap) > 0.0  # Detect a hard default even below the absolute comparison floor.
    np.testing.assert_allclose(gap, _SMOOTH_GAPS[case], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(jax.jit(vertical_advection_hybrid_van_leer)(q, F, p_s, coord),
                               smooth, rtol=1e-12, atol=1e-18)


def test_default_limiter_check_grads_order_two_at_plateau_and_zero_flux():
    """Perturb in width units so FD resolves the default-width transition."""
    from jax.test_util import check_grads

    coord, p_s, F, q = _limiter_case(2)
    width, flux_width = 1e-10, 1e-6
    # Keep finite differences inside the default transitions: the previous
    # 1e-3 plateau is 1e7 q-width units and loses FD resolution at eps=1e-6.
    q = q[:1] * (width / 1e-3)
    F = F[:1] * (flux_width / jnp.max(jnp.abs(F[:1])))
    F = F.at[:, 15].set(0.0)
    p_s = p_s[:1]
    # Keep a nonzero exchange to scale the scalar, even at the zero face.
    scale = jnp.max(jnp.abs(F)) / jnp.min(dp_from_hybrid(coord, p_s))
    weights = jnp.asarray(np.random.default_rng(0).normal(size=q.shape))

    def scalar(q_units, F_units):
        return jnp.sum(weights * vertical_advection_hybrid_van_leer(
            q_units * width, F_units * flux_width, p_s, coord)) / (width * scale)

    args = (q / width, F / flux_width)
    grad = jax.grad(scalar, argnums=(0, 1))(*args)
    hess = jax.jacfwd(jax.grad(scalar, argnums=0), argnums=0)(*args)
    assert all(np.all(np.isfinite(x)) for x in (*grad, hess))
    assert float(jnp.max(jnp.abs(grad[0]))) > 0.0
    # A hard limiter has no curvature inside these plateaus; pin smoothing.
    assert float(jnp.max(jnp.abs(hess))) > 0.0
    check_grads(scalar, args, order=2, modes=("fwd", "rev"), eps=1e-6)


@pytest.mark.parametrize("name", ["smoothing_width", "flux_smoothing_width"])
@pytest.mark.parametrize("width", [-1.0, float("nan"), float("inf")])
def test_limiter_width_rejects_invalid(name, width):
    coord, p_s, F, q = _column()
    with pytest.raises(ValueError, match=name):
        vertical_advection_hybrid_van_leer(q, F, p_s, coord, **{name: width})


def test_latlon_tracer_scheme_rejects_unknown_on_empty_state():
    from legoesm.atmosphere.dynamics.gcm import primitive_eq_latlon_cgrid as pe
    config = pe.CGridLatLonPrimitiveEquationConfig(tracer_vertical_scheme="typo")
    with pytest.raises(ValueError, match="unknown tracer vertical scheme"):
        pe.latlon_tracer_tendencies({}, *([None] * 12), config, True)


@pytest.mark.parametrize("module_name,class_name", [
    ("primitive_eq_mpas", "MPASPrimitiveEquationConfig"),
    ("primitive_eq_latlon_cgrid", "CGridLatLonPrimitiveEquationConfig"),
])
def test_smoothing_width_is_excluded_from_training(module_name, class_name):
    import importlib
    module = importlib.import_module("legoesm.atmosphere.dynamics.gcm." + module_name)
    spec = module.__param_spec__[class_name]
    for name in ("tracer_van_leer_width", "tracer_van_leer_flux_width"):
        assert name not in spec["params"]
        assert spec["excluded"][name] == "tunable_tier 0: numerics smoothing width"


def test_positive_width_approaches_hard_clipped_limit_on_stretched_grid():
    coord, p_s, F, q = _limiter_case(3)
    hard = vertical_advection_hybrid_van_leer(q, F, p_s, coord, smoothing_width=0.0, flux_smoothing_width=0.0)
    smooth = vertical_advection_hybrid_van_leer(q, F, p_s, coord)
    narrow = vertical_advection_hybrid_van_leer(q, F, p_s, coord, smoothing_width=1e-12, flux_smoothing_width=1e-8)
    # A 100x narrower transition must reduce the gap by at least 10x.
    assert float(jnp.max(jnp.abs(narrow - hard))) < 0.1 * float(jnp.max(jnp.abs(smooth - hard)))
    # Reflecting the tracer about zero interchanges bounds and reverses all
    # corrections; this also exercises the independently reconstructed q_neg.
    np.testing.assert_allclose(vertical_advection_hybrid_van_leer(-q, F, p_s, coord),
                               -smooth, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("name", ["smoothing_width", "flux_smoothing_width"])
@pytest.mark.parametrize("dtype,width", [(jnp.float32, 1e-30), (jnp.float32, 1e30),
                                         (jnp.float64, 1e-200), (jnp.float64, 1e200)])
def test_positive_width_must_be_representable(name, dtype, width):
    coord, p_s, F, q = _column()
    with pytest.raises(ValueError, match=name + ".*representable"):
        vertical_advection_hybrid_van_leer(q.astype(dtype), F.astype(dtype), p_s, coord,
                                         **{name: width})


@pytest.mark.parametrize("q_width,flux_width", [(0.0, 1e-6), (1e-10, 0.0), (1e-10, 1e-6)])
def test_independent_widths_conserve_and_preserve_constants(q_width, flux_width):
    coord, ps, F, q = _limiter_case(2)
    def op(q, F, ps, coord):
        return vertical_advection_hybrid_van_leer(
            q, F, ps, coord, smoothing_width=q_width, flux_smoothing_width=flux_width)
    assert product_rule_residual(op, coord, ps, F, q) < 1e-12
    assert float(jnp.max(jnp.abs(op(jnp.full_like(q, 7.3e-3), F, ps, coord)))) < 1e-18
    hard = vertical_advection_hybrid_van_leer(
        q, F, ps, coord, smoothing_width=0.0, flux_smoothing_width=0.0)
    assert bool(jnp.any(op(q, F, ps, coord) != hard))
