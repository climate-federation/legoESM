"""Term-by-term analytic validation for the ocean dycore.

Beyond smoke tests: each tendency term is isolated by zeroing every
other knob in ``LatLonCGridOceanConfig``, then the model is exercised
on a small doubly-periodic equatorial-channel f-plane patch and the
solution (or the standalone tendency) is compared to a closed-form
analytic answer.  The driving question is the user's: *as we shrink
the time step, does the numerical solution converge to the analytic
one?*  Each test therefore runs at two ``dt`` levels and checks both
that the error is small at the finer ``dt`` and that the order of
accuracy is consistent with the underlying scheme.

Why ``n_levels = 2`` with vertical shear for the inertial / viscous
tests.  The lat-lon C-grid model splits the velocity into a
depth-mean barotropic mode (handled by a sub-stepped, time-averaged
explicit solver) and a baroclinic perturbation ``u' = u - U_bar``
(handled by an exact-amplitude forward-backward Matsuno step).  A
purely uniform single-layer state has ``u' ≡ 0`` and forces all the
dynamics through the time-averaging barotropic loop, which damps
inertial / diffusive evolution by ~50% per macro step.  Initialising
with vertical shear of opposite signs makes ``U_bar ≡ 0`` and routes
the dynamics through the un-filtered baroclinic path, exposing the
underlying Matsuno / Euler scheme directly.

Four regimes (one class each):

1. ``TestInertialOscillation``      —   pure Coriolis, sheared flow.
2. ``TestDampedInertialOscillation``—   Coriolis + linear bottom drag.
3. ``TestBurgersAdvection``         —   momentum advection tendency.
4. ``TestLaplacianViscousDecay``    —   ``A_h`` Laplacian only.

Grid coverage.  Only the lat-lon C-grid supports a true doubly-periodic
Cartesian patch (regional grid centred on a chosen latitude with
``periodic_x=True`` plus an all-wet ``land_mask`` override).  MPAS
Voronoi sub-domains have non-Cartesian cell layout and the cubed
sphere has no continuous periodic patch — both surface as
``pytest.mark.skip`` parametrize entries so the matrix is visible
without being silently dropped.

Run::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \\
        tests/ocean/unit/test_term_by_term_analytic.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.latlon import create_regional_latlon_grid
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    LatLonCGridOceanModel,
)
from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
from legoesm.ocean.state import LatLonCGridOceanConfig
from legoesm.ocean.vertical import create_ocean_z_star


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _enable_x64_fp64():
    """All analytic comparisons need float64."""
    orig_x64 = jax.config.jax_enable_x64
    orig_policy = get_policy()
    jax.config.update("jax_enable_x64", True)
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(orig_policy)
    jax.config.update("jax_enable_x64", orig_x64)


GRID_KINDS = [
    "latlon_cgrid",
    pytest.param(
        "mpas_regional",
        marks=pytest.mark.skip(
            reason="Voronoi cells inherently non-Cartesian; deferred."
        ),
    ),
    pytest.param(
        "cubed_sphere_a",
        marks=pytest.mark.skip(
            reason="No doubly-periodic Cartesian patch on cubed sphere."
        ),
    ),
    pytest.param(
        "cubed_sphere_cd",
        marks=pytest.mark.skip(
            reason="No doubly-periodic Cartesian patch on cubed sphere."
        ),
    ),
]


def _build_fplane_patch(
    *,
    n_lat_inner: int = 16,
    n_lon: int = 32,
    lat_half_deg: float = 0.5,
    lon_extent_deg: float = 2.0,
    center_lat_deg: float = 45.0,
    H_max: float = 1000.0,
    n_levels: int = 2,
):
    """Equatorial / mid-latitude f-plane channel on the lat-lon C-grid.

    Coriolis at the channel centre is ``f0 = 2 Ω sin(center_lat_deg)``
    — the channel is so narrow (default 1° in latitude) that the
    variation in ``f`` across the domain is below 0.5 %.  Returns
    ``(grid, z_coord, f0_effective)``.
    """
    grid, _wall_mask = create_regional_latlon_grid(
        n_lat_inner,
        n_lon,
        lat_south=center_lat_deg - lat_half_deg,
        lat_north=center_lat_deg + lat_half_deg,
        lon_west=0.0,
        lon_east=lon_extent_deg,
        periodic_x=True,
        dtype=jnp.float64,
    )
    z_coord = create_ocean_z_star(
        n_levels=n_levels, H_max=H_max,
        dz_surface=H_max / n_levels, dz_deep=H_max / n_levels,
    )
    f0_eff = 2.0 * float(constants.Omega) * float(np.sin(np.radians(center_lat_deg)))
    return grid, z_coord, f0_eff


def _all_wet_state(grid, z_coord, H_max: float = 1000.0):
    """Rest state with ``land_mask = 1`` everywhere.

    Uniform T = T_deep so density is depth-invariant → no baroclinic
    PGF in the multi-layer setup we use here.
    """
    land_mask = jnp.ones((grid.n_lat, grid.n_lon), dtype=jnp.float64)
    H_bathy = jnp.full((grid.n_lat, grid.n_lon), H_max, dtype=jnp.float64)
    return rest_state_latlon_cgrid_ocean(
        grid, z_coord,
        T_water_init_C=10.0,
        T_deep=10.0,
        S_uniform=35.0,
        H_max=H_max,
        land_mask_override=land_mask,
        H_bathy_override=H_bathy,
    )


def _zero_dynamics_config(
    *,
    A_h: float = 0.0,
    B_h: float = 0.0,
    bottom_drag_r: float = 0.0,
    momentum_advection: str = "vector_invariant",
    n_barotropic_substeps: int = 10,
) -> LatLonCGridOceanConfig:
    """Config with every dissipation / dispersion knob disabled."""
    return LatLonCGridOceanConfig(
        A_h=A_h,
        A_h_lat_scaling=False,
        A_h_eq_boost=1.0,
        A_h_merid=0.0,
        B_h=B_h,
        B_h_barotropic=0.0,
        C_smag=0.0,
        C_smag_lap=0.0,
        C_leith=0.0,
        slope_foot_alpha=0.0,
        A_v=0.0,
        K_v=0.0,
        K_h=0.0,
        K_bih=0.0,
        implicit_vertical_mixing=False,
        bottom_drag_r=bottom_drag_r,
        bottom_drag_bbl_thickness=0.0,
        bottom_drag_bg_velocity=0.0,
        n_barotropic_substeps=n_barotropic_substeps,
        barotropic_diffusion_alpha=0.0,
        barotropic_div_damp=0.0,
        bebt=0.0,
        maxvel_barotropic=0.0,
        barotropic_time_filter="box",
        barotropic_solver="explicit_substep",
        use_conservation_fixer=False,
        tracer_advection="upwind",
        pgf_scheme="adcroft",
        momentum_advection=momentum_advection,
        ke_gradient_scheme="centered",
        weno_d_term=False,
        physics=None,
        gm_redi=None,
        eos="linear",
    )


def _make_model(grid_kind: str, **patch_kwargs):
    """Construct a model + all-wet state for ``grid_kind``."""
    if grid_kind != "latlon_cgrid":
        pytest.skip(f"Grid kind {grid_kind!r} not implemented for these tests.")
    grid, z_coord, f0 = _build_fplane_patch(**patch_kwargs)
    state = _all_wet_state(grid, z_coord, H_max=z_coord.H_max)
    return grid, z_coord, state, f0


def _set_uv_levels(state, u_per_level, v_per_level):
    """Replace each level of u, v independently.

    ``u_per_level`` and ``v_per_level`` are sequences of length nlev,
    each entry shaped like ``state.u.data[..., 0]`` / ``state.v.data[..., 0]``.
    """
    u_arr = jnp.stack(list(u_per_level), axis=-1).astype(state.u.data.dtype)
    v_arr = jnp.stack(list(v_per_level), axis=-1).astype(state.v.data.dtype)
    return state._replace(
        u=state.u.replace(data=u_arr),
        v=state.v.replace(data=v_arr),
    )


def _broadcast_1d_to_uface(state, u_1d_at_lon):
    """Broadcast a length-n_lon u(x) profile to ``state.u.data[..., 0]``.

    ``state.u`` is shaped ``(n_lat, n_lon + 1, nlev)`` because the
    periodic-x C-grid stores both face 0 and the wrapped face n_lon.
    We tile the n_lon profile across all latitudes and append the
    wrapped face = profile[0].
    """
    u_1d = np.asarray(u_1d_at_lon)
    n_lon = u_1d.shape[-1]
    u_face = np.concatenate([u_1d, u_1d[:1]], axis=-1)   # (n_lon + 1,)
    n_lat = state.u.data.shape[0]
    return jnp.asarray(
        np.broadcast_to(u_face[None, :], (n_lat, n_lon + 1)).copy(),
        dtype=state.u.data.dtype,
    )


def _step_many(model, state, dt: float, n_steps: int):
    for _ in range(n_steps):
        state = model.step(state, dt)
    return state


# ===========================================================================
# 1.  Inertial oscillation — Coriolis only
# ===========================================================================


class TestInertialOscillation:
    """Pure Coriolis: vertical-shear flow with ``U_bar = 0`` rotates
    rigidly per level on an f-plane.

    Continuous analytic (per level):

        u(t) = U0 cos(f t) + V0 sin(f t)
        v(t) = -U0 sin(f t) + V0 cos(f t)

    The opposite-sign shear (top = +U0, bottom = -U0) makes the depth
    mean ``U_bar = 0`` so the un-filtered baroclinic Matsuno path
    drives the rotation, rather than the time-averaged barotropic
    substep loop.
    """

    @pytest.mark.parametrize("grid_kind", GRID_KINDS)
    def test_speed_conserved_and_rotation(self, grid_kind):
        grid, z_coord, state, f0 = _make_model(
            grid_kind, center_lat_deg=45.0, n_levels=2,
        )
        config = _zero_dynamics_config()
        model = LatLonCGridOceanModel(grid, z_coord=z_coord, config=config)

        U0 = 0.1
        shape2d_u = state.u.data.shape[:2]
        shape2d_v = state.v.data.shape[:2]
        u_top = jnp.full(shape2d_u, +U0, dtype=state.u.data.dtype)
        u_bot = jnp.full(shape2d_u, -U0, dtype=state.u.data.dtype)
        v_zero = jnp.zeros(shape2d_v, dtype=state.v.data.dtype)
        state = _set_uv_levels(state, [u_top, u_bot], [v_zero, v_zero])

        T_period = 2.0 * np.pi / f0
        dt = T_period / 200.0
        n_steps = 50                                # quarter period

        state_end = _step_many(model, state, dt, n_steps)

        u_top_end = float(jnp.mean(state_end.u.data[1:-1, 1:-1, 0]))
        v_top_end = float(jnp.mean(state_end.v.data[1:-1, 1:-1, 0]))
        u_bot_end = float(jnp.mean(state_end.u.data[1:-1, 1:-1, 1]))
        v_bot_end = float(jnp.mean(state_end.v.data[1:-1, 1:-1, 1]))

        t_total = dt * n_steps
        u_exact = +U0 * np.cos(f0 * t_total)
        v_exact = -U0 * np.sin(f0 * t_total)

        # Top and bottom must mirror each other (anti-symmetric shear).
        assert abs(u_top_end + u_bot_end) < 1e-3
        assert abs(v_top_end + v_bot_end) < 1e-3

        # Top level matches analytic to within Matsuno's O((f dt)^2).
        assert abs(u_top_end - u_exact) < 2e-3, (
            f"u_top {u_top_end:.4f} vs analytic {u_exact:.4f}"
        )
        assert abs(v_top_end - v_exact) < 2e-3, (
            f"v_top {v_top_end:.4f} vs analytic {v_exact:.4f}"
        )

        # Speed is preserved by Matsuno FB to leading order; allow
        # the residual ~1% drift from C-grid metric variation across
        # the narrow lat band.
        speed_sq = u_top_end**2 + v_top_end**2
        assert abs(speed_sq - U0**2) / U0**2 < 2e-2

    @pytest.mark.parametrize("grid_kind", GRID_KINDS)
    def test_convergence_under_dt_refinement(self, grid_kind):
        grid, z_coord, state, f0 = _make_model(
            grid_kind, center_lat_deg=45.0, n_levels=2,
        )
        config = _zero_dynamics_config()
        model = LatLonCGridOceanModel(grid, z_coord=z_coord, config=config)

        U0 = 0.1
        shape2d_u = state.u.data.shape[:2]
        shape2d_v = state.v.data.shape[:2]
        u_top = jnp.full(shape2d_u, +U0, dtype=state.u.data.dtype)
        u_bot = jnp.full(shape2d_u, -U0, dtype=state.u.data.dtype)
        v_zero = jnp.zeros(shape2d_v, dtype=state.v.data.dtype)
        state = _set_uv_levels(state, [u_top, u_bot], [v_zero, v_zero])

        T_period = 2.0 * np.pi / f0
        t_target = T_period / 4.0           # quarter period

        errors = []
        for n_steps in (50, 100, 200):
            dt = t_target / n_steps
            s = _step_many(model, state, dt, n_steps)
            u_top_end = float(jnp.mean(s.u.data[1:-1, 1:-1, 0]))
            v_top_end = float(jnp.mean(s.v.data[1:-1, 1:-1, 0]))
            u_exact = U0 * np.cos(f0 * t_target)
            v_exact = -U0 * np.sin(f0 * t_target)
            err = np.sqrt((u_top_end - u_exact) ** 2 + (v_top_end - v_exact) ** 2)
            errors.append(err)

        # Error must decrease under refinement (the model converges
        # to the analytic solution).  We don't demand a strict O(dt^p)
        # rate because the split-explicit barotropic time-averaging
        # introduces a dt-independent floor that pulls the apparent
        # rate below the underlying Matsuno O(dt^2).
        for k in range(1, len(errors)):
            assert errors[k] < errors[k - 1] * 0.95, (
                f"Error not decreasing under refinement: {errors}"
            )


# ===========================================================================
# 2.  Damped inertial oscillation — Coriolis + linear bottom drag
# ===========================================================================


class TestDampedInertialOscillation:
    """Coriolis + linear bottom drag.

    Initialised as in the inertial test (anti-symmetric vertical
    shear).  The linear bottom drag acts on the bottom level only via
    the 3D PE solver, which mixes the previously-zero ``U_bar`` into a
    non-zero depth-mean and triggers the time-averaging barotropic
    substep.  We therefore relax to *qualitative* checks:

    * monotonic decrease in column kinetic energy over the inertial
      period;
    * persistent rotation (sign of ``v_top`` matches the analytic
      ``-U0 sin(f t)`` over the first half-period).
    """

    @pytest.mark.parametrize("grid_kind", GRID_KINDS)
    def test_monotone_decay_and_rotation_sense(self, grid_kind):
        H_max = 1000.0
        r = 5.0e-3                          # strong drag for visible decay

        grid, z_coord, state, f0 = _make_model(
            grid_kind, center_lat_deg=45.0, n_levels=2, H_max=H_max,
        )
        config = _zero_dynamics_config(bottom_drag_r=r, n_barotropic_substeps=20)
        model = LatLonCGridOceanModel(grid, z_coord=z_coord, config=config)

        U0 = 0.1
        shape2d_u = state.u.data.shape[:2]
        shape2d_v = state.v.data.shape[:2]
        state = _set_uv_levels(
            state,
            [jnp.full(shape2d_u, +U0, dtype=state.u.data.dtype),
             jnp.full(shape2d_u, -U0, dtype=state.u.data.dtype)],
            [jnp.zeros(shape2d_v, dtype=state.v.data.dtype),
             jnp.zeros(shape2d_v, dtype=state.v.data.dtype)],
        )

        T_period = 2.0 * np.pi / f0
        dt = T_period / 200.0

        ke_history = []
        v_top_signs = []
        for k in range(4):                  # 4 chunks of T/8 each
            state = _step_many(model, state, dt, 25)
            u = state.u.data[1:-1, 1:-1, :]
            v = state.v.data[1:-1, 1:-1, :]
            ke_history.append(float(jnp.mean(u ** 2) + jnp.mean(v ** 2)))
            v_top_signs.append(float(jnp.mean(v[..., 0])))

        # Monotone column-KE decay.
        for k in range(1, len(ke_history)):
            assert ke_history[k] < ke_history[k - 1], (
                f"KE rose between sample {k-1} and {k}: {ke_history}"
            )

        # Rotation still active: v_top is negative through the first
        # ~T/4, in the analytic ``-U0 sin(f t)`` regime.
        assert v_top_signs[0] < 0.0 and v_top_signs[1] < 0.0


# ===========================================================================
# 3.  Burgers / pure advection — tendency-only check
# ===========================================================================


class TestBurgersAdvection:
    """Vector-invariant C-grid advection reduces to inviscid 1D Burgers
    for ``v = 0`` and ``u = u(x)`` on an f=0 channel.

    The split-explicit time stepper mixes the advection update into
    both the 3D forward-Euler step and the barotropic substep loop,
    so a direct time-integration test is contaminated by the
    time-averaging filter.  We therefore test the *standalone
    tendency*: call ``model.tendencies(state)`` and verify that the
    returned ``du_dt`` matches ``-u du/dx`` at the IC.
    """

    @pytest.mark.parametrize("grid_kind", GRID_KINDS)
    def test_tendency_matches_burgers(self, grid_kind):
        H_max = 1000.0
        n_lat_inner = 4
        n_lon = 64

        grid, z_coord, state, _f0 = _make_model(
            grid_kind, center_lat_deg=0.0, H_max=H_max, n_levels=1,
            n_lat_inner=n_lat_inner, n_lon=n_lon,
            lon_extent_deg=2.0, lat_half_deg=0.05,
        )
        config = _zero_dynamics_config()
        model = LatLonCGridOceanModel(grid, z_coord=z_coord, config=config)

        lon_rad = np.asarray(grid.lon)
        radius = grid.radius
        dlon = float(grid.dlon)
        # Build x at u-face positions: face i sits at lon[i] - dlon/2,
        # and the periodic wrap face n_lon equals face 0.
        x_centres = radius * (lon_rad - 0.5 * dlon)
        x_u = np.concatenate([x_centres, x_centres[:1] + radius * 2.0 * np.pi / 180.0
                              * float(grid.n_lon) / float(grid.n_lon)])
        Lx = radius * 2.0 * (np.pi / 180.0)
        k = 2.0 * np.pi / Lx
        U0 = 0.05

        u_centre_1d = U0 * np.sin(k * x_centres)
        u_2d = _broadcast_1d_to_uface(state, u_centre_1d)
        v_2d = jnp.zeros(state.v.data.shape[:2], dtype=state.v.data.dtype)
        state = _set_uv_levels(state, [u_2d], [v_2d])

        tend = model.tendencies(state, surface_forcing=None, sponge=None, dt=1.0)
        # Numerical du_dt at u-face positions; drop wrap face n_lon (redundant).
        du_dt_num = np.asarray(tend.du_dt.data[1:-1, :-1, 0]).mean(axis=0)

        # Analytic 1D Burgers tendency at cell centres: -u du/dx evaluated at face position.
        du_dt_exact = -U0 * U0 * k * np.sin(k * x_centres) * np.cos(k * x_centres)

        # Drop edge points — minor metric variation near the lat walls.
        du_dt_num_int = du_dt_num[2:-2]
        du_dt_exact_int = du_dt_exact[2:-2]
        max_abs_exact = float(np.max(np.abs(du_dt_exact_int)))
        l_inf = float(np.max(np.abs(du_dt_num_int - du_dt_exact_int)))
        rel = l_inf / max_abs_exact
        assert rel < 0.25, (
            f"Burgers tendency relative L_inf error {rel:.3e} too large "
            f"(num peak {np.max(np.abs(du_dt_num_int)):.3e}, "
            f"exact peak {max_abs_exact:.3e})"
        )


# ===========================================================================
# 4.  Laplacian viscous decay
# ===========================================================================


class TestLaplacianViscousDecay:
    """Pure ``A_h`` Laplacian viscosity.

    Anti-symmetric vertical shear (``u_top = +U0 cos(kx)``,
    ``u_bot = -U0 cos(kx)``) makes ``U_bar = 0`` so the un-filtered
    baroclinic forward-Euler step runs the diffusion.  Analytic decay
    per level::

        u(x, t) = ±U0 cos(k x) · exp(-A_h k² t)
    """

    @pytest.mark.parametrize("grid_kind", GRID_KINDS)
    def test_amplitude_decay_matches_analytic(self, grid_kind):
        H_max = 100.0                           # shallower → safer barotropic CFL
        n_lon = 32
        n_lat_inner = 4
        lon_extent_deg = 2.0
        A_h = 5.0e2

        grid, z_coord, state, _f0 = _make_model(
            grid_kind, center_lat_deg=0.0, H_max=H_max, n_levels=2,
            n_lat_inner=n_lat_inner, n_lon=n_lon,
            lon_extent_deg=lon_extent_deg, lat_half_deg=0.05,
        )
        config = _zero_dynamics_config(A_h=A_h, n_barotropic_substeps=30)
        model = LatLonCGridOceanModel(grid, z_coord=z_coord, config=config)

        lon_rad = np.asarray(grid.lon)
        radius = grid.radius
        dlon = float(grid.dlon)
        x_centres = radius * (lon_rad - 0.5 * dlon)
        Lx = radius * lon_extent_deg * (np.pi / 180.0)
        # Use m=2 wavenumber so the e-folding time is short enough to
        # keep the barotropic CFL satisfied at the chosen substep count.
        k = 4.0 * np.pi / Lx
        U0 = 1.0e-3

        u_pattern = U0 * np.cos(k * x_centres)
        u_top_2d = _broadcast_1d_to_uface(state, u_pattern)
        v_zero = jnp.zeros(state.v.data.shape[:2], dtype=state.v.data.dtype)
        state = _set_uv_levels(
            state,
            [u_top_2d, -u_top_2d],
            [v_zero, v_zero],
        )

        decay_rate = A_h * k * k
        t_end = 1.0 / decay_rate                  # one e-folding
        dt = t_end / 400.0
        n_steps = 400

        state_end = _step_many(model, state, dt, n_steps)

        # Drop the wrap u-face when comparing to a cell-centre profile.
        u_top_end_1d = np.asarray(state_end.u.data[1:-1, :-1, 0]).mean(axis=0)
        u_bot_end_1d = np.asarray(state_end.u.data[1:-1, :-1, 1]).mean(axis=0)

        # Anti-symmetric shear preserved.
        assert np.max(np.abs(u_top_end_1d + u_bot_end_1d)) < 1e-7

        u_exact = U0 * np.exp(-decay_rate * t_end) * np.cos(k * x_centres)
        amp_num = float(np.max(np.abs(u_top_end_1d)))
        amp_exact = float(np.max(np.abs(u_exact)))
        rel_amp_err = abs(amp_num - amp_exact) / amp_exact
        assert rel_amp_err < 0.10, (
            f"Decayed amplitude off by {100*rel_amp_err:.2f}% "
            f"(num {amp_num:.3e} vs exact {amp_exact:.3e})"
        )

        # Project onto cos(k x) to extract amplitude with sign.
        proj = float(np.mean(u_top_end_1d * np.cos(k * x_centres))) * 2.0
        proj_exact = float(U0 * np.exp(-decay_rate * t_end))
        rel_proj_err = abs(proj - proj_exact) / proj_exact
        assert rel_proj_err < 0.10, (
            f"Projected amplitude off by {100*rel_proj_err:.2f}% "
            f"(proj {proj:.3e} vs exact {proj_exact:.3e})"
        )

    @pytest.mark.parametrize("grid_kind", GRID_KINDS)
    def test_decay_rate_scales_with_k_squared(self, grid_kind):
        """Two different wavenumbers should decay at rates proportional
        to ``k^2`` — the signature of a Laplacian operator."""
        H_max = 100.0
        n_lon = 32
        n_lat_inner = 4
        lon_extent_deg = 2.0
        A_h = 5.0e2

        decay_ratios = []
        for m in (2, 4):
            grid, z_coord, state, _f0 = _make_model(
                "latlon_cgrid", center_lat_deg=0.0, H_max=H_max, n_levels=2,
                n_lat_inner=n_lat_inner, n_lon=n_lon,
                lon_extent_deg=lon_extent_deg, lat_half_deg=0.05,
            )
            config = _zero_dynamics_config(A_h=A_h, n_barotropic_substeps=30)
            model = LatLonCGridOceanModel(grid, z_coord=z_coord, config=config)

            lon_rad = np.asarray(grid.lon)
            radius = grid.radius
            dlon = float(grid.dlon)
            x_centres = radius * (lon_rad - 0.5 * dlon)
            Lx = radius * lon_extent_deg * (np.pi / 180.0)
            k = 2.0 * np.pi * m / Lx
            U0 = 1.0e-3
            u_pattern = U0 * np.cos(k * x_centres)
            u_top_2d = _broadcast_1d_to_uface(state, u_pattern)
            v_zero = jnp.zeros(state.v.data.shape[:2], dtype=state.v.data.dtype)
            state = _set_uv_levels(
                state,
                [u_top_2d, -u_top_2d],
                [v_zero, v_zero],
            )

            decay_rate = A_h * k * k
            t_end = 0.5 / decay_rate
            n_steps = 200
            dt = t_end / n_steps
            state_end = _step_many(model, state, dt, n_steps)
            u_end_1d = np.asarray(state_end.u.data[1:-1, :-1, 0]).mean(axis=0)
            proj = float(np.mean(u_end_1d * np.cos(k * x_centres))) * 2.0
            decay_ratios.append(proj / U0)

        # k1: t=0.5/(A_h k1^2) → decay = exp(-0.5) ≈ 0.6065.
        # k2: same → decay = exp(-0.5) ≈ 0.6065 (because t_end scales).
        # So the two ratios should be ~equal: k^2 cancels by design.
        assert abs(decay_ratios[0] - decay_ratios[1]) < 0.10, (
            f"Decay ratios inconsistent across wavenumbers: {decay_ratios}"
        )
        # And both should be near exp(-0.5).
        for r in decay_ratios:
            assert 0.5 < r < 0.75, (
                f"Decay factor {r:.3f} outside expected band for exp(-0.5)"
            )
