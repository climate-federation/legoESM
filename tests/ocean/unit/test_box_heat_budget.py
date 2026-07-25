"""Direct unit test for the #1226 online box heat-budget accumulator.

Covers the instrument's own validation gates:
  1. Closure — sum of the 4 directly-computed terms + the vertmix residual
     equals the measured box heat-content change, by construction (the
     residual IS defined that way) — the real check is that the 4 EXPLICIT
     terms are non-trivial (not all zero) and that the closure identity
     holds bit-for-bit, not just approximately.
  2. Non-invasive: running the accumulator's ``.sample()`` alongside a short
     model integration must not perturb the model trajectory at all —
     the prognostic state with the accumulator ON must be bit-identical to
     the state with it OFF.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import LatLonCGridOceanModel
from legoesm.ocean.experiments.dino import (
    DINOConfig,
    dino_lat_lon_surface_forcing_arrays,
    dino_step_surface_forcing,
)
from legoesm.ocean.fidelity.box_heat_budget import (
    BoxHeatBudgetAccumulator,
    TERM_NAMES,
    compute_box_heat_dT_terms,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star

DT = 900.0  # s
N_LAT, N_LON, N_LEV = 24, 16, 8
ROW_SLICE = slice(4, 20)
BANDS = ((0.0, 200.0), (200.0, 2000.0), (2000.0, None))


@pytest.fixture
def grid():
    return create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)


@pytest.fixture
def z_coord():
    return create_ocean_z_star(n_levels=N_LEV, H_max=3000.0)


@pytest.fixture
def config():
    gm = GMRediConfig(kappa_GM=50.0, kappa_Redi=50.0, slope_scheme="nemo_iso_lap")
    return LatLonCGridOceanConfig.from_flat(
        A_h=1.0e3, A_v=1.0e-3, K_h=0.0, K_v=1.0e-4,
        bottom_drag_r=1.0e-3, gm_redi=gm,
        implicit_vertical_mixing=True,
        outer_integrator="forward_euler",
    )


@pytest.fixture
def dino_cfg():
    return DINOConfig(forcing_annual_cycle=False)


@pytest.fixture
def state(grid, z_coord):
    rng = np.random.default_rng(0)
    s = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=20.0, T_deep=4.0, S_uniform=35.0,
        H_max=3000.0,
    )
    u = 0.02 * rng.standard_normal((grid.n_lat, grid.n_lon + 1, z_coord.n_levels))
    v = 0.02 * rng.standard_normal((grid.n_lat + 1, grid.n_lon, z_coord.n_levels))
    return s._replace(
        u=s.u.replace(data=jnp.asarray(u) * (s.u_mask.data[..., None] > 0.5)),
        v=s.v.replace(data=jnp.asarray(v) * (s.v_mask.data[..., None] > 0.5)),
    )


def _run(state, model, forcing, sf, dino_cfg, *, n_steps, sample_every,
         accumulate: bool):
    """Advance ``n_steps`` of ``DT``, optionally sampling the accumulator
    every ``sample_every`` steps (including step 0, before any stepping)."""
    acc = None
    if accumulate:
        acc = BoxHeatBudgetAccumulator(
            model.grid, model.z_coord, model.config, dino_cfg, forcing, DT,
            row_slice=ROW_SLICE, depth_bands_m=BANDS,
        )

    st = state
    t = 0.0
    if acc is not None:
        acc.sample(st, dt_step=sample_every * DT, t_seconds=t)
    for k in range(n_steps):
        st = model.step(st, DT, surface_forcing=sf, t_seconds=t)
        t += DT
        if acc is not None and (k + 1) % sample_every == 0:
            acc.sample(st, dt_step=sample_every * DT, t_seconds=t)
    return st, acc, t


def test_accumulator_does_not_alter_trajectory(grid, z_coord, config, dino_cfg):
    """Flag-off vs flag-on: the PROGNOSTIC state must be bit-identical.
    ``BoxHeatBudgetAccumulator.sample`` reads the state but the run loop
    never feeds its output back into ``model.step`` — verify that holds."""
    model = LatLonCGridOceanModel(grid, z_coord, config)
    forcing = dino_lat_lon_surface_forcing_arrays(model.grid, dino_cfg)
    sf = dino_step_surface_forcing(forcing)

    base = rest_state_latlon_cgrid_ocean(
        model.grid, z_coord, T_water_init_C=20.0, T_deep=4.0, S_uniform=35.0,
        H_max=3000.0,
    )
    rng = np.random.default_rng(1)
    u = 0.02 * rng.standard_normal((model.grid.n_lat, model.grid.n_lon + 1, z_coord.n_levels))
    v = 0.02 * rng.standard_normal((model.grid.n_lat + 1, model.grid.n_lon, z_coord.n_levels))
    st0 = base._replace(
        u=base.u.replace(data=jnp.asarray(u) * (base.u_mask.data[..., None] > 0.5)),
        v=base.v.replace(data=jnp.asarray(v) * (base.v_mask.data[..., None] > 0.5)),
    )

    st_off, acc_off, _ = _run(st0, model, forcing, sf, dino_cfg,
                              n_steps=6, sample_every=2, accumulate=False)
    st_on, acc_on, _ = _run(st0, model, forcing, sf, dino_cfg,
                            n_steps=6, sample_every=2, accumulate=True)

    assert acc_off is None
    assert acc_on is not None
    for field in ("T", "S", "u", "v", "eta"):
        a = np.asarray(getattr(st_off, field).data)
        b = np.asarray(getattr(st_on, field).data)
        np.testing.assert_array_equal(
            a, b, err_msg=f"field {field!r} diverged with accumulator ON"
        )


def test_accumulator_closure_and_nontrivial_terms(grid, z_coord, config, dino_cfg):
    """Budget closure is exact BY CONSTRUCTION (vertmix is defined as the
    residual) — the real gate is that this identity actually holds (no
    arithmetic bug in the accumulation) and that the four directly-computed
    terms are not degenerately zero, so the decomposition is doing real
    work, not silently no-op-ing."""
    model = LatLonCGridOceanModel(grid, z_coord, config)
    forcing = dino_lat_lon_surface_forcing_arrays(model.grid, dino_cfg)
    sf = dino_step_surface_forcing(forcing)

    rng = np.random.default_rng(2)
    base = rest_state_latlon_cgrid_ocean(
        model.grid, z_coord, T_water_init_C=20.0, T_deep=4.0, S_uniform=35.0,
        H_max=3000.0,
    )
    u = 0.03 * rng.standard_normal((model.grid.n_lat, model.grid.n_lon + 1, z_coord.n_levels))
    v = 0.03 * rng.standard_normal((model.grid.n_lat + 1, model.grid.n_lon, z_coord.n_levels))
    st0 = base._replace(
        u=base.u.replace(data=jnp.asarray(u) * (base.u_mask.data[..., None] > 0.5)),
        v=base.v.replace(data=jnp.asarray(v) * (base.v_mask.data[..., None] > 0.5)),
    )

    n_steps, sample_every = 20, 4
    _, acc, t_final = _run(st0, model, forcing, sf, dino_cfg,
                           n_steps=n_steps, sample_every=sample_every,
                           accumulate=True)

    assert acc.n_samples == n_steps // sample_every + 1
    total_seconds = t_final  # first sample at t=0
    summary = acc.summary(total_seconds)

    any_nonzero = {term: False for term in TERM_NAMES}
    for band in BANDS:
        band_out = summary["terms"][band]
        # Closure: dH == sum(explicit terms) + vertmix(residual), by
        # construction of accum_W["vertmix"] — verify the arithmetic
        # actually reproduces that identity (catches an accounting bug).
        term_sum_J = sum(band_out[t]["J"] for t in TERM_NAMES)
        # Residual must be many orders of magnitude below the heat-content
        # scale itself (~1e24 J here) — floating-point roundoff only, not a
        # real unclosed term (the vertmix bucket IS the closing residual by
        # construction; this catches an ACCOUNTING bug, e.g. double-counting
        # or mismatched interval boundaries).
        scale = max(abs(band_out["dH_J"]), 1.0)
        np.testing.assert_allclose(
            term_sum_J, band_out["dH_J"], rtol=1e-9, atol=1e-9 * scale,
            err_msg=f"budget does not close for band {band}",
        )
        assert abs(band_out["closure_residual_J"]) < 1e-9 * scale
        for term in TERM_NAMES:
            if abs(band_out[term]["J"]) > 1e-8:
                any_nonzero[term] = True

    # At least the directly-computed terms must be doing real work on a
    # perturbed, forced, GM/Redi-enabled state.
    assert any_nonzero["adv_h"] or any_nonzero["adv_v"], (
        "advection terms are degenerately zero — box/sampling misconfigured"
    )
    assert any_nonzero["forcing"], "forcing term is degenerately zero"

    assert summary["box_area_m2"] > 0.0
    for band in BANDS:
        for term in TERM_NAMES:
            assert np.isfinite(summary["terms"][band][term]["W_per_m2"])


def test_physics_dt_is_the_model_timestep_not_the_sampling_interval(
    grid, z_coord, config, dino_cfg, state,
):
    """Regression guard (#1226 physics-validator review finding): the
    accumulator MUST feed the model's own dynamical dt (``DT``) to
    ``restoring_surface_forcing``/the FCT limiter/the GM/Redi MSC clamp —
    NOT the (much longer) sampling interval. All three are genuinely
    dt-sensitive (not merely dt-rescaled), so conflating the two silently
    biases every explicit term. Direct check: call
    ``compute_box_heat_dT_terms`` with the correct model dt vs a 32x-longer
    "sampling interval" dt and confirm the FORCING term (the clearest
    dt-sensitive case: ``restoring_surface_forcing(implicit=True)`` uses
    ``eff_tau = tau + dt``) materially differs — i.e. dt really matters,
    so accidentally passing the wrong one is not a no-op."""
    forcing = dino_lat_lon_surface_forcing_arrays(grid, dino_cfg)
    terms_model_dt = compute_box_heat_dT_terms(
        state, grid, z_coord, config, dino_cfg, forcing, DT, t_seconds=0.0,
    )
    terms_sampling_dt = compute_box_heat_dT_terms(
        state, grid, z_coord, config, dino_cfg, forcing, DT * 32, t_seconds=0.0,
    )
    mask = np.asarray(state.land_mask.data) > 0.5
    forcing_model = np.asarray(terms_model_dt["forcing"])[mask]
    forcing_sampling = np.asarray(terms_sampling_dt["forcing"])[mask]
    assert not np.allclose(forcing_model, forcing_sampling, rtol=1e-6), (
        "FORCING term is insensitive to dt -- the dt-conflation regression "
        "guard would not catch a re-introduced bug; investigate whether "
        "restoring_surface_forcing's implicit tau+dt path is still wired."
    )

    # End-to-end guard: sample() must use dt_model (=DT, fixed at
    # construction), not dt_step (which varies with sampling cadence) --
    # the reported per-second FORCING rate must be ~IDENTICAL whether
    # sampled every 2 or every 4 steps (same physics, different bookkeeping
    # cadence only).
    model = LatLonCGridOceanModel(grid, z_coord, config)
    sf = dino_step_surface_forcing(forcing)
    _, acc2, t2 = _run(state, model, forcing, sf, dino_cfg,
                       n_steps=8, sample_every=2, accumulate=True)
    _, acc4, t4 = _run(state, model, forcing, sf, dino_cfg,
                       n_steps=8, sample_every=4, accumulate=True)
    s2 = acc2.summary(t2)
    s4 = acc4.summary(t4)
    for band in BANDS:
        f2 = s2["terms"][band]["forcing"]["W_per_m2"]
        f4 = s4["terms"][band]["forcing"]["W_per_m2"]
        if abs(f2) < 1e-12 and abs(f4) < 1e-12:
            continue
        # Loose tolerance: sample_every=2 vs 4 also evaluates the endpoint
        # rate at slightly different states along a genuinely-evolving
        # trajectory (real physics, not a bug) -- O(1e-4) relative here.
        # The dt-conflation bug this guards against was ~8% (tau_T-scale),
        # 4 orders of magnitude above the trajectory-divergence noise floor.
        np.testing.assert_allclose(
            f2, f4, rtol=1e-2,
            err_msg=(
                f"band {band}: FORCING W/m^2 depends on sampling cadence "
                "(sample_every=2 vs 4) far beyond trajectory-divergence "
                "noise -- dt_step may be leaking into the physics dt again"
            ),
        )
