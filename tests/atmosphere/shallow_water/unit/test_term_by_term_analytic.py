"""Term-by-term analytic validation for the atmosphere shallow-water dycore.

Atmosphere analogue of ``tests/ocean/unit/test_term_by_term_analytic.py``.
Each tendency in the lat-lon C-grid shallow-water solver is isolated by
zeroing the other knobs, then the *standalone tendency* returned by
``cgrid_latlon_sw_tendencies`` is compared to a closed-form analytic
answer.

Why tendency-only rather than time integration.  Atmosphere shallow
water has no barotropic / baroclinic split.  Any time-integrated
``uniform-flow + f-plane`` setup couples Coriolis to the free-surface
height through the divergence the rotation generates, exciting an
inertia-gravity wave whose period (``ω² = f² + g H k²``) is comparable
to the inertial period at the scales involved.  The tendency
formulation cuts through that contamination: it probes the operator
math directly, without the IGW feedback or the gravity-wave CFL.

Four regimes:

1. ``TestCoriolisTendencySW``        —   Coriolis term in isolation.
2. ``TestRayleighDampingSW``         —   linear drag *NOT* implemented;
                                            documented skip.
3. ``TestBurgersAdvectionSW``        —   momentum advection tendency
                                            at the equator (f = 0).
4. ``TestLaplacianViscousTendencySW``—   isolated ``A_h`` Laplacian
                                            tendency.

Grid coverage.  Only the lat-lon C-grid supports a true doubly-periodic
Cartesian patch; cubed-sphere FV3, MPAS Voronoi, and spectral appear
as documented ``pytest.mark.skip`` parametrize entries.

Run::

    JAX_ENABLE_X64=1 .venv/bin/python -m pytest \\
        tests/atmosphere/shallow_water/unit/test_term_by_term_analytic.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.dynamics.shallow_water_latlon_cgrid import (
    CGridLatLonShallowWaterConfig,
    CGridLatLonShallowWaterState,
    cgrid_latlon_sw_tendencies,
)
from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
from legoesm.grids.latlon import create_regional_latlon_grid


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _enable_x64_fp64():
    """Analytic comparisons require float64."""
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
        "cubed_sphere_fv3",
        marks=pytest.mark.skip(
            reason="No doubly-periodic Cartesian patch on cubed sphere."
        ),
    ),
    pytest.param(
        "mpas_voronoi",
        marks=pytest.mark.skip(
            reason="Voronoi cells inherently non-Cartesian; deferred."
        ),
    ),
    pytest.param(
        "spectral",
        marks=pytest.mark.skip(
            reason="Spectral solver global only; no periodic Cartesian patch."
        ),
    ),
]


H_REST = 100.0       # rest layer depth [m]; absolute value unimportant
                     # for tendency-only tests as long as h stays uniform.


def _build_fplane_patch(
    *,
    n_lat_inner: int = 16,
    n_lon: int = 32,
    lat_half_deg: float = 0.5,
    lon_extent_deg: float = 2.0,
    center_lat_deg: float = 45.0,
):
    """Equatorial / mid-latitude f-plane channel on the lat-lon C-grid.

    Coriolis at the channel centre is ``f0 = 2 Ω sin(center_lat_deg)``;
    over a 1° lat band the variation in ``f`` is below 0.5 %.  Returns
    ``(grid, f0_effective)``.
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
    f0 = 2.0 * float(constants.Omega) * float(np.sin(np.radians(center_lat_deg)))
    return grid, f0


def _rest_state(grid, H=H_REST):
    """Rest state: uniform layer depth, zero winds, flat surface."""
    return CGridLatLonShallowWaterState(
        h=jnp.full((grid.n_lat, grid.n_lon), H, dtype=jnp.float64),
        u=jnp.zeros((grid.n_lat, grid.n_lon + 1), dtype=jnp.float64),
        v=jnp.zeros((grid.n_lat + 1, grid.n_lon), dtype=jnp.float64),
        h_s=jnp.zeros((grid.n_lat, grid.n_lon), dtype=jnp.float64),
    )


def _zero_dynamics_config(*, A_h: float = 0.0) -> CGridLatLonShallowWaterConfig:
    """Config with every dissipation knob off."""
    return CGridLatLonShallowWaterConfig(
        A_h=A_h,
        time_integrator="ssp_rk3",
        fix_mass=False,
        anchor_mass_to_initial=False,
        use_ppm_transport=True,
        use_polar_filter=False,
    )


def _broadcast_1d_to_uface(state, u_1d_at_lon):
    """Broadcast a length-n_lon u(x) profile to ``state.u`` (n_lat, n_lon+1).

    Periodic-x stores both face 0 and the wrapped face n_lon.
    """
    u_1d = np.asarray(u_1d_at_lon)
    n_lon = u_1d.shape[-1]
    u_face = np.concatenate([u_1d, u_1d[:1]], axis=-1)   # (n_lon + 1,)
    n_lat = state.u.shape[0]
    return jnp.asarray(
        np.broadcast_to(u_face[None, :], (n_lat, n_lon + 1)).copy(),
        dtype=state.u.dtype,
    )


def _make_grid_or_skip(grid_kind: str, **patch_kwargs):
    if grid_kind != "latlon_cgrid":
        pytest.skip(f"Grid kind {grid_kind!r} not implemented for these tests.")
    return _build_fplane_patch(**patch_kwargs)


# ===========================================================================
# 1.  Coriolis tendency check
# ===========================================================================


class TestCoriolisTendencySW:
    """Vector-invariant Coriolis tendency on uniform flow.

    For ``v = 0`` and ``u = U0`` everywhere, relative vorticity
    ``ζ = ∂v/∂x − ∂u/∂y = 0`` (uniform u in latitude over the narrow
    channel).  Absolute vorticity ``η = ζ + f ≈ f``.  The C-grid
    momentum tendency reduces to:

        du/dt = +(ζ + f) · v_avg ≈ 0
        dv/dt = -(ζ + f) · u_avg ≈ -f · U0

    A symmetric setup with ``u = 0, v = V0`` gives ``du/dt ≈ +f V0``,
    ``dv/dt ≈ 0``.  We test both.
    """

    @pytest.mark.parametrize("grid_kind", GRID_KINDS)
    def test_coriolis_u(self, grid_kind):
        grid, f0 = _make_grid_or_skip(grid_kind, center_lat_deg=45.0)
        config = _zero_dynamics_config()
        U0 = 0.1
        state = _rest_state(grid)
        state = state._replace(
            u=jnp.full(state.u.shape, U0, dtype=state.u.dtype),
        )

        _dh, _du, dv = cgrid_latlon_sw_tendencies(state, grid, config)
        dv_mid = float(jnp.mean(dv[1:-1, 1:-1]))
        expected = -f0 * U0
        rel = abs(dv_mid - expected) / abs(expected)
        assert rel < 5e-3, (
            f"dv/dt {dv_mid:.4e} vs expected {expected:.4e} "
            f"(rel err {rel:.2e})"
        )

    @pytest.mark.parametrize("grid_kind", GRID_KINDS)
    def test_coriolis_v(self, grid_kind):
        grid, f0 = _make_grid_or_skip(grid_kind, center_lat_deg=45.0)
        config = _zero_dynamics_config()
        V0 = 0.1
        state = _rest_state(grid)
        state = state._replace(
            v=jnp.full(state.v.shape, V0, dtype=state.v.dtype),
        )

        _dh, du, _dv = cgrid_latlon_sw_tendencies(state, grid, config)
        du_mid = float(jnp.mean(du[1:-1, 1:-1]))
        expected = +f0 * V0
        rel = abs(du_mid - expected) / abs(expected)
        assert rel < 5e-3, (
            f"du/dt {du_mid:.4e} vs expected {expected:.4e} "
            f"(rel err {rel:.2e})"
        )


# ===========================================================================
# 2.  Rayleigh / linear damping — not implemented in atmosphere SW
# ===========================================================================


class TestRayleighDampingSW:
    """Linear (Rayleigh) drag is *not* implemented in the atmosphere
    shallow-water solver.  ``CGridLatLonShallowWaterConfig`` carries no
    drag coefficient (see ``src/legoesm/atmosphere/dynamics/
    shallow_water_latlon_cgrid.py:80``); nor do the cubed-sphere FV3,
    MPAS, or spectral configs.  Surface as documented skip so the gap
    is visible in the matrix.
    """

    @pytest.mark.parametrize("grid_kind", GRID_KINDS)
    @pytest.mark.skip(reason="Linear drag not implemented in atmosphere SW.")
    def test_monotone_decay_and_rotation_sense(self, grid_kind):
        pass


# ===========================================================================
# 3.  Burgers / pure momentum advection — tendency check
# ===========================================================================


class TestBurgersAdvectionSW:
    """Vector-invariant C-grid advection reduces to inviscid 1D Burgers
    for ``v = 0``, ``u = u(x)`` on an f = 0 channel placed at the
    equator (where ``sin(lat) = 0`` makes the Coriolis term vanish).

    With a flat free surface (``h = H``, ``h_s = 0``) the PGF is
    identically zero, so the only active momentum term is advection.
    Analytic tendency:

        du/dt = -u du/dx = -U0² k sin(kx) cos(kx)
    """

    @pytest.mark.parametrize("grid_kind", GRID_KINDS)
    def test_tendency_matches_burgers(self, grid_kind):
        grid, _f0 = _make_grid_or_skip(
            grid_kind, center_lat_deg=0.0,
            n_lat_inner=4, n_lon=64,
            lon_extent_deg=2.0, lat_half_deg=0.05,
        )
        config = _zero_dynamics_config()

        lon_rad = np.asarray(grid.lon)
        radius = grid.radius
        dlon = float(grid.dlon)
        x_centres = radius * (lon_rad - 0.5 * dlon)
        Lx = radius * 2.0 * (np.pi / 180.0)
        k = 2.0 * np.pi / Lx
        U0 = 0.05

        state = _rest_state(grid)
        u_1d = U0 * np.sin(k * x_centres)
        state = state._replace(u=_broadcast_1d_to_uface(state, u_1d))

        _dh, du, _dv = cgrid_latlon_sw_tendencies(state, grid, config)
        du_num = np.asarray(du[1:-1, :-1]).mean(axis=0)

        du_exact = -U0 * U0 * k * np.sin(k * x_centres) * np.cos(k * x_centres)

        du_num_int = du_num[2:-2]
        du_exact_int = du_exact[2:-2]
        max_abs_exact = float(np.max(np.abs(du_exact_int)))
        l_inf = float(np.max(np.abs(du_num_int - du_exact_int)))
        rel = l_inf / max_abs_exact
        assert rel < 0.25, (
            f"Burgers tendency relative L_inf error {rel:.3e} too large "
            f"(num peak {np.max(np.abs(du_num_int)):.3e}, "
            f"exact peak {max_abs_exact:.3e})"
        )


# ===========================================================================
# 4.  Laplacian viscous tendency
# ===========================================================================


class TestLaplacianViscousTendencySW:
    """Pure ``A_h`` Laplacian viscosity tendency.

    IC: ``u(x) = U0 cos(kx)``, ``v = 0``, flat ``h``, equator
    (Coriolis = 0).  The Laplacian term contributes:

        du/dt |_visc = A_h · d²u/dx² = -A_h k² · U0 cos(kx)

    Two checks: (a) the tendency at the IC matches the analytic
    formula; (b) varying ``k`` shows the expected ``k²`` scaling.
    """

    @pytest.mark.parametrize("grid_kind", GRID_KINDS)
    def test_tendency_matches_laplacian(self, grid_kind):
        n_lon = 32
        A_h = 5.0e2

        grid, _f0 = _make_grid_or_skip(
            grid_kind, center_lat_deg=0.0,
            n_lat_inner=4, n_lon=n_lon,
            lon_extent_deg=2.0, lat_half_deg=0.05,
        )
        config = _zero_dynamics_config(A_h=A_h)

        lon_rad = np.asarray(grid.lon)
        radius = grid.radius
        dlon = float(grid.dlon)
        x_centres = radius * (lon_rad - 0.5 * dlon)
        Lx = radius * 2.0 * (np.pi / 180.0)
        k = 4.0 * np.pi / Lx                       # m = 2
        U0 = 1.0e-3

        state = _rest_state(grid)
        u_1d = U0 * np.cos(k * x_centres)
        state = state._replace(u=_broadcast_1d_to_uface(state, u_1d))

        _dh, du, _dv = cgrid_latlon_sw_tendencies(state, grid, config)
        du_num = np.asarray(du[1:-1, :-1]).mean(axis=0)

        # Subtract any small advection contribution by also computing
        # du with A_h = 0 (gives the pure advection residue).
        config_no_visc = _zero_dynamics_config(A_h=0.0)
        _dh0, du0, _dv0 = cgrid_latlon_sw_tendencies(state, grid, config_no_visc)
        du_visc_num = du_num - np.asarray(du0[1:-1, :-1]).mean(axis=0)

        du_visc_exact = -A_h * k * k * U0 * np.cos(k * x_centres)

        # Interior comparison.
        du_visc_num_int = du_visc_num[2:-2]
        du_visc_exact_int = du_visc_exact[2:-2]
        max_abs_exact = float(np.max(np.abs(du_visc_exact_int)))
        l_inf = float(np.max(np.abs(du_visc_num_int - du_visc_exact_int)))
        rel = l_inf / max_abs_exact
        assert rel < 0.20, (
            f"Viscous tendency relative L_inf error {rel:.3e} too large "
            f"(num peak {np.max(np.abs(du_visc_num_int)):.3e}, "
            f"exact peak {max_abs_exact:.3e})"
        )

    @pytest.mark.parametrize("grid_kind", GRID_KINDS)
    def test_tendency_scales_with_k_squared(self, grid_kind):
        """Two wavenumbers, tendency amplitude should scale as ``k²``."""
        n_lon = 64
        A_h = 5.0e2

        amplitudes = []
        ks = []
        for m in (2, 4):
            grid, _f0 = _make_grid_or_skip(
                grid_kind, center_lat_deg=0.0,
                n_lat_inner=4, n_lon=n_lon,
                lon_extent_deg=2.0, lat_half_deg=0.05,
            )
            config = _zero_dynamics_config(A_h=A_h)

            lon_rad = np.asarray(grid.lon)
            radius = grid.radius
            dlon = float(grid.dlon)
            x_centres = radius * (lon_rad - 0.5 * dlon)
            Lx = radius * 2.0 * (np.pi / 180.0)
            k = 2.0 * np.pi * m / Lx
            U0 = 1.0e-3

            state = _rest_state(grid)
            u_1d = U0 * np.cos(k * x_centres)
            state = state._replace(u=_broadcast_1d_to_uface(state, u_1d))

            _dh, du, _dv = cgrid_latlon_sw_tendencies(state, grid, config)
            _dh0, du0, _dv0 = cgrid_latlon_sw_tendencies(
                state, grid, _zero_dynamics_config(A_h=0.0))
            du_visc = (np.asarray(du[1:-1, :-1]).mean(axis=0)
                       - np.asarray(du0[1:-1, :-1]).mean(axis=0))
            amplitudes.append(float(np.max(np.abs(du_visc))))
            ks.append(k)

        # Ratio amp2/amp1 should equal (k2/k1)^2 = 4.
        ratio_num = amplitudes[1] / amplitudes[0]
        ratio_exact = (ks[1] / ks[0]) ** 2
        rel = abs(ratio_num - ratio_exact) / ratio_exact
        assert rel < 0.10, (
            f"Tendency amplitude ratio {ratio_num:.3f} vs k² ratio "
            f"{ratio_exact:.3f} (rel err {rel:.2e})"
        )
