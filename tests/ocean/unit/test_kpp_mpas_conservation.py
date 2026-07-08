"""Column heat/salt conservation of MPAS-KPP vertical mixing on partial cells.

Regression for the partial-cell non-conservation bug in
``ocean/physics/vertical_mixing/mpas_integration.py::make_kpp_physics_mpas``:

``kpp_vertical_mixing`` builds the tracer tendencies ``dT/dt``, ``dS/dt`` as
flux-form vertical divergences divided by the REFERENCE-grid thickness
``dz_ref * J`` (kpp.py ``dz_actual`` — used by BOTH the local down-gradient
diffusion and the non-local counter-gradient term).  The MPAS dycore, however,
advances heat/salt content weighted by the LIVE partial-cell thickness
``h_k = compute_layer_thickness(eta, H_bathy, z_coord)``
(``ocean_model_mpas.py``: ``T_new = T + dt*dT_dt``, budget measured as
``sum(dT_dt * h_k * area)``).  On a partial bottom cell ``dz_ref*J > h_k``, so
the LIVE-thickness column integral of a purely REDISTRIBUTIVE interior-mixing
tendency was NONZERO — a spurious source/sink on partial/live cells.

The fix rescales the tendency by ``dz_used / h_k`` (turning ``D/dz_used`` into
``D/h_k`` for the same interface-flux divergence ``D``), so the live-thickness
column integral ``sum_k h_k * dT/dt = sum_k D = F_surface - F_seafloor = 0``
is conserved to machine precision (flux-form, zero flux at surface + seafloor).

These tests build a small MPAS Voronoi column with a genuine partial bottom
cell (``h_partial = 0.5 * dz_ref``) and assert:

1. Interior redistribution (no surface flux): the live-thickness-weighted
   column heat ``sum(rho*cp*T*h_partial*area)`` and salt ``sum(S*h_partial*area)``
   change is ~0 to machine epsilon, while the reference-thickness (pre-fix)
   weighting is NOT — proving the test discriminates the fix (non-vacuous).
2. With a convective surface flux (B_f>0 activates the non-local term): the
   surface forcing reaches the tendency (it differs from the no-flux case) AND
   the live-weighted column integral is STILL conserved — KPP REDISTRIBUTES;
   the net surface injection is a separate dycore source (the floored-h_k term
   in ``mpas_ocean_baroclinic_tendencies``), not part of ``dT_dt``.

Notes on the "flux*dt" expectation: because the KPP non-local flux vanishes at
both the surface (G(sigma=0)=0) and the boundary-layer base (G(sigma=1)=0), it
injects NO net column heat — so a surface heat flux does NOT change the KPP
``dT_dt`` column integral by ``flux*dt``.  The genuine, code-faithful invariant
is column conservation of the redistribution, which is what is asserted here.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.core.field import Field
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.eos import c_sw as _C_SW, rho_0 as _RHO_0
from legoesm.ocean.init_mpas import rest_state_mpas_ocean
from legoesm.ocean.physics.vertical_mixing.config import (
    KPPConfig,
    VerticalMixingConfig,
)
from legoesm.ocean.physics.vertical_mixing.mpas_integration import (
    make_kpp_physics_mpas,
)
from legoesm.ocean.vertical import (
    OceanPartialCellCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
    create_ocean_z_star,
    create_partial_cell_coordinate,
)

_H_MAX = 4000.0
_N_LEV = 10
_K_BOT = 6  # target bottom level -> n_active = 7 (>= the KPP 5-level floor)
_DT = 900.0  # physics step [s] for the "content change = dt * tendency" framing


@pytest.fixture(scope="module", autouse=True)
def _fp64_policy():
    """Machine-eps conservation requires fp64 STATE arrays.  ``JAX_ENABLE_X64``
    alone does not set the legoESM precision policy (whose default storage is
    fp32); without this the flux-form telescoping only cancels to fp32 (~1e-7)
    and the conservation assertion would be a precision artifact, not physics.
    """
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy
    orig = get_policy()
    set_policy(PrecisionPolicy.fp64())
    yield
    set_policy(orig)


@pytest.fixture(scope="module")
def mesh():
    return create_voronoi_mesh(subdivision_level=2)


@pytest.fixture(scope="module")
def z_coord(_fp64_policy):
    return create_ocean_z_star(n_levels=_N_LEV, H_max=_H_MAX)


@pytest.fixture(scope="module")
def partial_setup(_fp64_policy, mesh, z_coord):
    """All-ocean MPAS state on a partial-cell coord with a mid-bin bottom cell.

    Every wet column gets the SAME bathymetry ``H`` placed at the midpoint of
    reference bin ``_K_BOT`` so ``bottom_level == _K_BOT`` and
    ``h_partial[_K_BOT] == 0.5 * dz_ref[_K_BOT]`` (a genuine partial cell,
    strictly thinner than the reference thickness).  T and S carry real
    vertical gradients so the down-gradient diffusion produces nonzero,
    non-vacuous tendencies for BOTH tracers.
    """
    abs_z_half = np.abs(np.asarray(z_coord.z_half_ref))  # (nlev+1,)
    H_val = 0.5 * (abs_z_half[_K_BOT] + abs_z_half[_K_BOT + 1])

    # Base rest state (all ocean: land_lat_threshold=90 -> no land cells).
    state = rest_state_mpas_ocean(
        mesh, z_coord, H_max=_H_MAX, land_lat_threshold=90.0,
    )
    nCells = mesh.nCells
    H_bathy = jnp.full((nCells,), H_val, dtype=state.H_bathy.data.dtype)
    pc_coord = create_partial_cell_coordinate(z_coord, H_bathy)

    # Stratified T (warm over cold) and S (fresh over salty): both have a
    # vertical gradient at the partial bottom cell so dT/dt and dS/dt != 0.
    k = jnp.arange(_N_LEV, dtype=jnp.float64)
    T_prof = 20.0 - 1.5 * k          # degC, decreasing with depth
    S_prof = 34.0 + 0.10 * k         # psu, increasing with depth
    T_2d = jnp.broadcast_to(T_prof, (nCells, _N_LEV))
    S_2d = jnp.broadcast_to(S_prof, (nCells, _N_LEV))

    state = state._replace(
        H_bathy=Field(
            data=H_bathy, name="H_bathy", dims=("nCells",), units="m",
        ),
        T=Field(
            data=T_2d.astype(state.T.data.dtype),
            name="T", dims=("nCells", "nlev"), units="degC",
        ),
        S=Field(
            data=S_2d.astype(state.S.data.dtype),
            name="S", dims=("nCells", "nlev"), units="psu",
        ),
    )
    return state, pc_coord


@pytest.fixture
def kpp_cfg():
    return VerticalMixingConfig(scheme="kpp", kpp=KPPConfig())


def _live_thickness_and_area(state, pc_coord, mesh):
    """LIVE partial-cell thickness ``h_k`` and per-cell horizontal area."""
    h_live = compute_layer_thickness(
        state.eta.data, state.H_bathy.data, pc_coord,
    )  # (nCells, nlev)
    area = jnp.asarray(mesh.areaCell)[:, jnp.newaxis]  # (nCells, 1)
    return h_live, area


def _column_content_change(tend, h_live, area, rho_cp):
    """Live-thickness-weighted column-integrated content change over ``_DT``.

    ``Delta = rho_cp * dt * sum_over(cells, levels)(tendency * h_live * area)``.
    For a conservative interior redistribution this is 0 to machine epsilon.
    """
    return float(rho_cp * _DT * jnp.sum(tend * h_live * area))


def test_partial_setup_is_a_real_partial_cell(partial_setup, z_coord):
    """Guard: the fixture must actually build a genuine partial bottom cell
    with >= 5 active levels (else the conservation test is vacuous or the
    shallow-cell mask would zero the whole column)."""
    _, pc_coord = partial_setup
    assert isinstance(pc_coord, OceanPartialCellCoordinate)
    bot = np.asarray(pc_coord.bottom_level)
    assert np.all(bot == _K_BOT), f"expected bottom_level={_K_BOT}, got {set(bot)}"
    dz_ref = np.asarray(z_coord.dz_ref)
    h_partial = np.asarray(pc_coord.h_partial)
    # Partial bottom cell is strictly thinner than the reference thickness.
    np.testing.assert_allclose(
        h_partial[:, _K_BOT], 0.5 * dz_ref[_K_BOT], rtol=1e-6,
    )
    assert dz_ref[_K_BOT] - h_partial[0, _K_BOT] > 1.0  # meaningful (>1 m) gap
    assert _K_BOT + 1 >= 5  # n_active clears the 5-level KPP floor


class TestInteriorRedistributionConserves:
    """No surface flux => the KPP tracer tendency is a pure interior
    redistribution and MUST conserve the live-thickness column integral."""

    def test_heat_and_salt_conserved_to_machine_eps(
        self, mesh, z_coord, partial_setup, kpp_cfg,
    ):
        state, pc_coord = partial_setup
        fn = make_kpp_physics_mpas(kpp_cfg)
        _, dT_dt, dS_dt = fn(state, mesh, pc_coord, None)

        assert bool(jnp.all(jnp.isfinite(dT_dt)))
        assert bool(jnp.all(jnp.isfinite(dS_dt)))

        h_live, area = _live_thickness_and_area(state, pc_coord, mesh)
        rho_cp = _RHO_0 * _C_SW

        dHeat = _column_content_change(dT_dt, h_live, area, rho_cp)
        dSalt = _column_content_change(dS_dt, h_live, area, _RHO_0)

        # Absolute scale of the redistribution (total |flux| moved), used as
        # the relative-conservation reference.
        heat_scale = float(
            rho_cp * _DT * jnp.sum(jnp.abs(dT_dt) * h_live * area)
        )
        salt_scale = float(
            _RHO_0 * _DT * jnp.sum(jnp.abs(dS_dt) * h_live * area)
        )

        # Non-vacuous: interior diffusion actually moved heat and salt.
        assert heat_scale > 0.0
        assert salt_scale > 0.0

        # Live-thickness column integral conserved to machine precision.
        assert abs(dHeat) <= 1e-10 * heat_scale, (
            f"heat not conserved: dHeat={dHeat:.3e} J, "
            f"scale={heat_scale:.3e} J (ratio {abs(dHeat)/heat_scale:.1e})"
        )
        assert abs(dSalt) <= 1e-10 * salt_scale, (
            f"salt not conserved: dSalt={dSalt:.3e}, scale={salt_scale:.3e}"
        )

    def test_reference_thickness_weighting_would_not_conserve(
        self, mesh, z_coord, partial_setup, kpp_cfg,
    ):
        """Discriminating counter-check (proves the fix is real): the PRE-FIX
        tendency ``D/dz_used`` (reconstructed from the shipped ``D/h_k`` output)
        weighted by the LIVE thickness has a NONZERO column integral — i.e. the
        old reference-thickness code was non-conservative on the partial cell.
        """
        state, pc_coord = partial_setup
        fn = make_kpp_physics_mpas(kpp_cfg)
        _, dT_dt, _ = fn(state, mesh, pc_coord, None)

        h_live, area = _live_thickness_and_area(state, pc_coord, mesh)
        J = compute_ocean_jacobian(state.eta.data, state.H_bathy.data, pc_coord)
        dz_used = z_coord.dz_ref * J[:, jnp.newaxis]  # what KPP divided by

        # D = shipped(D/h_k) * h_k ; pre-fix tendency = D / dz_used.
        D = dT_dt * h_live
        buggy_tend = D / dz_used  # exactly the old make_kpp_physics_mpas output
        rho_cp = _RHO_0 * _C_SW
        dHeat_buggy = _column_content_change(buggy_tend, h_live, area, rho_cp)
        heat_scale = float(
            rho_cp * _DT * jnp.sum(jnp.abs(dT_dt) * h_live * area)
        )
        assert abs(dHeat_buggy) > 1e-6 * heat_scale, (
            "reference-thickness (pre-fix) weighting was already conservative "
            "on this column — the test does not discriminate the fix; "
            f"dHeat_buggy={dHeat_buggy:.3e}, scale={heat_scale:.3e}"
        )


class TestConvectiveSurfaceFluxStillRedistributes:
    """A convective surface flux (B_f>0) activates the non-local counter-
    gradient term.  It threads through to the tendency, yet KPP still only
    REDISTRIBUTES: the live-weighted column integral stays conserved."""

    def test_surface_flux_reaches_tendency_and_conserves(
        self, mesh, z_coord, partial_setup, kpp_cfg,
    ):
        state, pc_coord = partial_setup
        fn = make_kpp_physics_mpas(kpp_cfg)

        # No-flux baseline.
        _, dT_dt_ref, _ = fn(state, mesh, pc_coord, None)

        # Surface cooling (q_net < 0) => destabilising => B_f > 0 => the
        # non-local term switches on.
        nCells = mesh.nCells
        q_net = jnp.full((nCells,), -50.0, dtype=jnp.float64)  # W/m^2, cooling
        forcing = types.SimpleNamespace(
            tau_x=None, tau_y=None, q_net=q_net,
            freshwater=None, salt_flux=None,
        )
        _, dT_dt, dS_dt = fn(state, mesh, pc_coord, forcing)

        assert bool(jnp.all(jnp.isfinite(dT_dt)))
        # The surface flux genuinely reached the tendency (non-local term on).
        assert not bool(jnp.allclose(dT_dt, dT_dt_ref)), (
            "convective surface flux did not change dT_dt — the non-local "
            "counter-gradient term never activated"
        )

        h_live, area = _live_thickness_and_area(state, pc_coord, mesh)
        rho_cp = _RHO_0 * _C_SW
        dHeat = _column_content_change(dT_dt, h_live, area, rho_cp)
        heat_scale = float(
            rho_cp * _DT * jnp.sum(jnp.abs(dT_dt) * h_live * area)
        )
        assert heat_scale > 0.0
        # KPP redistributes: net column heat change from dT_dt is ~0 (the
        # surface injection is a separate dycore source, not part of dT_dt).
        assert abs(dHeat) <= 1e-10 * heat_scale, (
            f"convective KPP redistribution not conserved: dHeat={dHeat:.3e} J, "
            f"scale={heat_scale:.3e} J"
        )


def test_ad_gradient_finite_through_conservation_fix(
    mesh, z_coord, partial_setup, kpp_cfg,
):
    """The partial-cell rescale must keep the tendency differentiable: a
    scalar loss on the KPP tracer tendency has finite gradients w.r.t. T."""
    state, pc_coord = partial_setup
    fn = make_kpp_physics_mpas(kpp_cfg)

    def loss(T_data):
        s = state._replace(
            T=Field(
                data=T_data, name="T", dims=("nCells", "nlev"), units="degC",
            ),
        )
        _, dT_dt, dS_dt = fn(s, mesh, pc_coord, None)
        return jnp.sum(dT_dt ** 2) + jnp.sum(dS_dt ** 2)

    g = jax.grad(loss)(state.T.data)
    assert bool(jnp.all(jnp.isfinite(g))), (
        "jax.grad through the partial-cell conservation rescale produced "
        "NaN/Inf gradients"
    )
