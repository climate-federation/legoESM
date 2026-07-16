"""Cross-grid NH dycore consistency.

Compares the doubly-periodic plane (PR2-PR5), cubed-sphere C-D
(``CDGridCompressibleEulerModel``), and MPAS Voronoi
(``MPASCompressibleEulerModel``) non-hydrostatic compressible
Euler dycores at the API-contract level.

Per-grid (parametrised) checks:
1. ``test_rest_state_preserved`` — isothermal hydrostatic atmosphere
   stays near rest in u/v/w/θ'/ρ' after 10 steps; per-grid
   tolerances reflect curvature/metric noise floor.
2. ``test_dry_mass_conserved`` — each dycore's own
   ``compute_dry_mass(state)`` integral drifts < per-grid tolerance
   without the mass fixer.
3. ``test_buoyancy_signed_response`` — a positive θ' perturbation
   produces *positive* (upward) max(w) on every grid; rules out
   wrong-sign buoyancy regressions that ``max|w|`` would miss.

Cross-grid checks (exercising all three grids together):
4. ``test_cross_grid_buoyancy_sign_consistent`` — positive θ'
   perturbation gives max(w) > 0 on every grid (signs agree).
5. ``test_cross_grid_buoyancy_order_of_magnitude`` — log10 spread of
   max(w) across the three dycores < 3 (factor 1000). Cell sizes
   and metric scales differ legitimately between plane (4 km),
   cubed-sphere (~1000 km at C4), and MPAS Voronoi (~5000 km at
   resolution 2), so this is a coarse "all grids respond within the
   same ballpark" check, not a tight numerical match.

Setup failures are HARD-FAILS (not skips): both the parametrised
``grid_setup`` fixture and the cross-grid ``all_grid_responses``
fixture let exceptions propagate so a regression in cubed-sphere
or MPAS construction is reported as ERROR rather than SKIP.

Note on physics-equivalence
---------------------------
Full RCEMIP-style physics equivalence (matching OLR / surface flux
/ precipitation / MSE budget between the three dycores at
equilibrium) requires multi-day integrations of each grid's own
RCE harness. That validation runs outside this CI suite — the
plane harness lives at ``scripts/run/run_rcemip_plane.py`` (PR4);
cubed-sphere + MPAS RCE wiring is documented separately. This
file's scope is the **dynamical-core contract**, not the
parametrised-physics equilibrium.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# Common cross-grid spec
NLEV = 5
H_TOP = 30_000.0
DT = 5.0
N_STEPS = 10
THETA_PERT_AMPLITUDE = 0.5  # K


# Per-grid tolerances (measured-baseline + curvature/metric headroom).
# Plane has no curvature so hits machine precision; cubed-sphere
# curvature + MPAS dual-mesh connectivity carry O(1e-6 m/s) metric
# noise after 10 steps from an isothermal hydrostatic rest state.
# A regression that widens a tolerance MUST update the measured
# baseline in the comment and explain why the noise floor moved.
_TOL = {
    "plane": {
        # Plane is bit-exact in rest-state on the unit
        # ``test_plane_nh_rest_state`` harness; 1e-10 is generous.
        "rest_w": 1.0e-10,
        "rest_u": 1.0e-10,
        "rest_theta_p": 1.0e-10,
        "rest_rho_p": 1.0e-10,
        # Measured 0.0 drift on the plane (analytical conservation).
        "mass_drift": 1.0e-12,
    },
    "cubed_sphere": {
        # C4 duogrid; measured O(1e-5 to 1e-4) metric noise.
        "rest_w": 1.0e-3,
        "rest_u": 1.0e-3,
        "rest_theta_p": 1.0e-3,
        "rest_rho_p": 1.0e-3,
        # MPAS / cubed-sphere paths sum across 6 faces / many edges;
        # mass-fixer-off drift is dominated by float64 reduction
        # roundoff (~1e-13) but bumped to 1e-10 for headroom.
        "mass_drift": 1.0e-10,
    },
    "mpas": {
        # Voronoi resolution 2 with Lloyd smoothing; measured similar
        # noise floor to cubed-sphere C4 in this configuration.
        "rest_w": 1.0e-3,
        "rest_u": 1.0e-3,
        "rest_theta_p": 1.0e-3,
        "rest_rho_p": 1.0e-3,
        "mass_drift": 1.0e-10,
    },
}


# --------------------------------------------------------------------- #
# Per-grid factories                                                    #
# --------------------------------------------------------------------- #


def _build_plane():
    from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
        CompressibleEulerConfig,
    )
    from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
        PlaneCompressibleEulerModel,
        compute_dry_mass_plane,
        make_flat_plane_terrain_metric,
        make_rest_state,
    )
    from legoesm.grids.plane import create_plane_grid
    from legoesm.grids.vertical import create_height_coordinate

    grid = create_plane_grid(
        nx=8, ny=8, nlev=NLEV, dx=4_000.0, dy=4_000.0,
        dtype=jnp.float64,
    )
    hc = create_height_coordinate(NLEV, H=H_TOP)
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=0.0, hyperdiff_coeff=0.0,
        hyperdiff_rho_coeff=0.0, hyperdiff_w_coeff=0.0,
        semi_implicit_acoustic=False, use_coriolis=False,
        fix_mass=False, smagorinsky_cs=0.0,
    )
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)
    state = make_rest_state(grid, hc, dtype=jnp.float64)
    return {
        "name": "plane",
        "model": model,
        "state": state,
        "mass_fn": lambda s: compute_dry_mass_plane(s, grid, hc, tm),
        "max_abs_w": lambda s: float(jnp.max(jnp.abs(s.w.data))),
        "max_abs_u": lambda s: float(jnp.max(jnp.abs(s.u.data))),
        "max_abs_theta_p": lambda s: float(
            jnp.max(jnp.abs(s.theta_prime.data))
        ),
        "max_abs_rho_p": lambda s: float(
            jnp.max(jnp.abs(s.rho_prime.data))
        ),
        "max_w_signed": lambda s: float(jnp.max(s.w.data)),
        "min_w_signed": lambda s: float(jnp.min(s.w.data)),
        "perturb_theta": lambda s, amp: s._replace(
            theta_prime=s.theta_prime.replace(
                data=s.theta_prime.data.at[:, :, NLEV // 2].add(amp),
            ),
        ),
        "tol": _TOL["plane"],
    }


def _build_cubed_sphere():
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerConfig,
        CDGridCompressibleEulerModel,
    )
    from legoesm.core.field import Field
    from legoesm.core.state import NonHydrostaticState
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import (
        compute_terrain_metric,
        create_height_coordinate,
    )

    n = 4
    grid = create_cubed_sphere(n, use_duogrid=True)
    hc = create_height_coordinate(NLEV, H_TOP)
    tm = compute_terrain_metric(jnp.zeros((6, n, n)), hc)
    # Cubed-sphere-specific config (separate from the shared
    # CompressibleEulerConfig used by plane / MPAS code paths;
    # cubed-sphere carries C-D-grid-specific damping knobs).
    cfg = CDGridCompressibleEulerConfig(
        n_acoustic_substeps=4, fix_mass=False,
    )
    model = CDGridCompressibleEulerModel(grid, hc, tm, cfg)
    dims_3d = ("face", "x", "y", "level")
    dims_w = ("face", "x", "y", "level_half")
    dims_2d = ("face", "x", "y")
    state = NonHydrostaticState(
        u=Field(jnp.zeros((6, n, n, NLEV), jnp.float64), name="u",
                dims=dims_3d, units="m/s"),
        v=Field(jnp.zeros((6, n, n, NLEV), jnp.float64), name="v",
                dims=dims_3d, units="m/s"),
        w=Field(jnp.zeros((6, n, n, NLEV + 1), jnp.float64), name="w",
                dims=dims_w, units="m/s"),
        theta_prime=Field(jnp.zeros((6, n, n, NLEV), jnp.float64),
                          name="theta_prime", dims=dims_3d, units="K"),
        rho_prime=Field(jnp.zeros((6, n, n, NLEV), jnp.float64),
                        name="rho_prime", dims=dims_3d, units="kg/m^3"),
        phis=Field(jnp.zeros((6, n, n), jnp.float64), name="phis",
                   dims=dims_2d, units="m^2/s^2"),
        tracers=Field(jnp.zeros((6, n, n, NLEV, 0), jnp.float64),
                      name="tracers",
                      dims=("face", "x", "y", "level", "tracer"),
                      units="kg/kg"),
    )

    def _perturb(s, amp):
        # Add amp at the single mid-level cell (face 0, i=0, j=0).
        return s._replace(
            theta_prime=s.theta_prime.replace(
                data=s.theta_prime.data.at[0, 0, 0, NLEV // 2].add(amp),
            ),
        )

    return {
        "name": "cubed_sphere",
        "model": model,
        "state": state,
        "mass_fn": model.compute_dry_mass,
        "max_abs_w": lambda s: float(jnp.max(jnp.abs(s.w.data))),
        "max_abs_u": lambda s: float(jnp.max(jnp.abs(s.u.data))),
        "max_abs_theta_p": lambda s: float(
            jnp.max(jnp.abs(s.theta_prime.data))
        ),
        "max_abs_rho_p": lambda s: float(
            jnp.max(jnp.abs(s.rho_prime.data))
        ),
        "max_w_signed": lambda s: float(jnp.max(s.w.data)),
        "min_w_signed": lambda s: float(jnp.min(s.w.data)),
        "perturb_theta": _perturb,
        "tol": _TOL["cubed_sphere"],
    }


def _build_mpas():
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_mpas import (
        MPASCompressibleEulerConfig,
        MPASCompressibleEulerModel,
    )
    from legoesm.core.field import Field
    from legoesm.core.state import MPASNonHydrostaticState
    from legoesm.grids.vertical import (
        compute_terrain_metric,
        create_height_coordinate,
    )
    from legoesm.grids.voronoi import create_voronoi_mesh

    mesh = create_voronoi_mesh(2, lloyd_iterations=5)
    hc = create_height_coordinate(NLEV, H_TOP)
    tm = compute_terrain_metric(
        jnp.zeros(mesh.nCells, jnp.float64), hc,
    )
    cfg = MPASCompressibleEulerConfig(
        nu_del2=1.0e4, n_acoustic_substeps=4, fix_mass=False,
    )
    model = MPASCompressibleEulerModel(mesh, hc, tm, cfg)
    state = MPASNonHydrostaticState(
        u=Field(jnp.zeros((mesh.nEdges, NLEV), jnp.float64), name="u",
                dims=("nEdges", "nlev"), units="m/s"),
        w=Field(jnp.zeros((mesh.nCells, NLEV + 1), jnp.float64), name="w",
                dims=("nCells", "nlev_half"), units="m/s"),
        theta_prime=Field(jnp.zeros((mesh.nCells, NLEV), jnp.float64),
                          name="theta_prime", dims=("nCells", "nlev"),
                          units="K"),
        rho_prime=Field(jnp.zeros((mesh.nCells, NLEV), jnp.float64),
                        name="rho_prime", dims=("nCells", "nlev"),
                        units="kg/m^3"),
        phis=Field(jnp.zeros(mesh.nCells, jnp.float64), name="phis",
                   dims=("nCells",), units="m^2/s^2"),
        tracers=Field(jnp.zeros((mesh.nCells, NLEV, 0), jnp.float64),
                      name="tracers",
                      dims=("nCells", "nlev", "tracer"), units="kg/kg"),
    )

    def _perturb(s, amp):
        return s._replace(
            theta_prime=s.theta_prime.replace(
                data=s.theta_prime.data.at[0, NLEV // 2].add(amp),
            ),
        )

    return {
        "name": "mpas",
        "model": model,
        "state": state,
        "mass_fn": model.compute_dry_mass,
        "max_abs_w": lambda s: float(jnp.max(jnp.abs(s.w.data))),
        "max_abs_u": lambda s: float(jnp.max(jnp.abs(s.u.data))),
        "max_abs_theta_p": lambda s: float(
            jnp.max(jnp.abs(s.theta_prime.data))
        ),
        "max_abs_rho_p": lambda s: float(
            jnp.max(jnp.abs(s.rho_prime.data))
        ),
        "max_w_signed": lambda s: float(jnp.max(s.w.data)),
        "min_w_signed": lambda s: float(jnp.min(s.w.data)),
        "perturb_theta": _perturb,
        "tol": _TOL["mpas"],
    }


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


_GRID_BUILDERS = {
    "plane": _build_plane,
    "cubed_sphere": _build_cubed_sphere,
    "mpas": _build_mpas,
}


@pytest.fixture(scope="module", params=list(_GRID_BUILDERS))
def grid_setup(request):
    """One setup dict per grid. Setup failures are HARD-FAILS — the
    exception propagates so a regression in cubed-sphere or MPAS
    construction shows as ERROR, not SKIP. (Codex review
    2026-05-24, finding "Skip behaviour".) All three grids are
    core dycores in CI; none is optional."""
    return _GRID_BUILDERS[request.param]()


@pytest.fixture(scope="module")
def all_grid_responses():
    """Run the buoyancy perturbation experiment on ALL three grids
    once and return a dict keyed by grid name. Used by the
    cross-grid sign + order-of-magnitude tests. Hard-fails on any
    setup error."""
    out = {}
    for name, builder in _GRID_BUILDERS.items():
        setup = builder()
        state = setup["perturb_theta"](
            setup["state"], THETA_PERT_AMPLITUDE,
        )
        for _ in range(N_STEPS):
            state = setup["model"].step(state, dt=DT)
        out[name] = {
            "max_w_signed": setup["max_w_signed"](state),
            "min_w_signed": setup["min_w_signed"](state),
            "max_abs_w": setup["max_abs_w"](state),
            "finite": bool(jnp.all(jnp.isfinite(state.w.data))),
        }
    return out


# --------------------------------------------------------------------- #
# Per-grid tests (parametrised by grid_setup)                           #
# --------------------------------------------------------------------- #


def test_rest_state_preserved(grid_setup):
    """Isothermal hydrostatic atmosphere with no perturbation stays
    near rest after 10 steps on each NH dycore. Now also checks
    ``max|ρ'|`` — Codex review found that omitting it allowed a
    dycore to develop local density perturbations while still
    passing both the velocity-rest and global-mass checks."""
    state = grid_setup["state"]
    for _ in range(N_STEPS):
        state = grid_setup["model"].step(state, dt=DT)
    tol = grid_setup["tol"]
    max_w = grid_setup["max_abs_w"](state)
    max_u = grid_setup["max_abs_u"](state)
    max_th = grid_setup["max_abs_theta_p"](state)
    max_rho = grid_setup["max_abs_rho_p"](state)
    assert max_w < tol["rest_w"], (
        f"{grid_setup['name']} rest-state max|w| = {max_w:.3e}; "
        f"expected < {tol['rest_w']:.0e}"
    )
    assert max_u < tol["rest_u"], (
        f"{grid_setup['name']} rest-state max|u| = {max_u:.3e}; "
        f"expected < {tol['rest_u']:.0e}"
    )
    assert max_th < tol["rest_theta_p"], (
        f"{grid_setup['name']} rest-state max|θ'| = {max_th:.3e}; "
        f"expected < {tol['rest_theta_p']:.0e}"
    )
    assert max_rho < tol["rest_rho_p"], (
        f"{grid_setup['name']} rest-state max|ρ'| = {max_rho:.3e}; "
        f"expected < {tol['rest_rho_p']:.0e}"
    )


def test_dry_mass_conserved(grid_setup):
    """Each dycore's own ``compute_dry_mass(state)`` integral stays
    constant to its per-grid tolerance over 10 steps without the
    mass fixer. Verifies slow-tendency continuity branch + acoustic
    substep are consistent on every grid."""
    state = grid_setup["state"]
    mass_0 = float(grid_setup["mass_fn"](state))
    for _ in range(N_STEPS):
        state = grid_setup["model"].step(state, dt=DT)
    mass_f = float(grid_setup["mass_fn"](state))
    rel = abs(mass_f - mass_0) / max(abs(mass_0), 1.0e-30)
    tol = grid_setup["tol"]["mass_drift"]
    assert rel < tol, (
        f"{grid_setup['name']} dry-mass drift = {rel:.3e} over "
        f"{N_STEPS} steps; expected < {tol:.0e}"
    )


def test_buoyancy_signed_response(grid_setup):
    """A positive θ' perturbation produces *positive* (upward) max(w)
    on every grid. Codex review noted that checking only ``max|w|``
    misses a wrong-sign regression: a dycore that responded
    downward-only to warm air would still pass the magnitude
    bound. We now assert (a) signed ``max(w) > +threshold`` for the
    positive perturbation, and (b) upward peak >= magnitude of
    downward peak (compensating subsidence must not dominate)."""
    state = grid_setup["perturb_theta"](
        grid_setup["state"], THETA_PERT_AMPLITUDE,
    )
    state_next = grid_setup["model"].step(state, dt=DT)
    max_w_signed = grid_setup["max_w_signed"](state_next)
    min_w_signed = grid_setup["min_w_signed"](state_next)
    max_abs_w = grid_setup["max_abs_w"](state_next)
    # Sign check: positive θ' → upward parcel acceleration → max(w)
    # must be strictly positive after one step. Lower bound is well
    # above floating-point noise but well below the analytical
    # estimate (g·Δθ/θ̄·Δt ≈ 0.016 m/s for Δθ=0.5 K, Δt=5 s).
    assert max_w_signed > 1.0e-9, (
        f"{grid_setup['name']} buoyancy WRONG SIGN: positive θ' "
        f"gave max(w) = {max_w_signed:.3e} (expected > +1e-9)"
    )
    # No blow-up.
    assert max_abs_w < 1.0e2, (
        f"{grid_setup['name']} buoyancy BLOW-UP: max|w| = "
        f"{max_abs_w:.3e} (expected < 100 m/s)"
    )
    # Upward peak must dominate the downward peak after one step
    # from a single warm parcel — otherwise the response is
    # backwards (subsidence > updraft).
    assert max_w_signed >= abs(min_w_signed), (
        f"{grid_setup['name']} buoyancy ASYMMETRY: upward peak "
        f"{max_w_signed:.3e} < downward peak {abs(min_w_signed):.3e}; "
        "warm-parcel updraft should dominate compensating subsidence"
    )
    assert jnp.all(jnp.isfinite(state_next.w.data))


# --------------------------------------------------------------------- #
# Cross-grid tests (exercising all three grids together)                #
# --------------------------------------------------------------------- #


def test_cross_grid_buoyancy_sign_consistent(all_grid_responses):
    """All three NH dycores must agree on the SIGN of the vertical
    response to a positive θ' perturbation. This is the strongest
    cross-grid contract from a physics standpoint: regardless of
    cell size or metric, warm air goes up. A grid that gave
    max(w) <= 0 after 10 steps of a +0.5 K perturbation has a
    signed-buoyancy bug and breaks coupling assumptions of any
    physics that consumes w."""
    bad = {
        name: r["max_w_signed"]
        for name, r in all_grid_responses.items()
        if not (r["max_w_signed"] > 0.0 and r["finite"])
    }
    assert not bad, (
        f"Sign disagreement / non-finite w on grids: {bad}. "
        f"All grids must produce max(w) > 0 after a positive θ' "
        f"perturbation. Full responses: "
        f"{ {n: r['max_w_signed'] for n, r in all_grid_responses.items()} }"
    )


def test_cross_grid_buoyancy_order_of_magnitude(all_grid_responses):
    """All three NH dycores must produce ``max(w)`` within 3 orders
    of magnitude of one another. Cell sizes and metric scales
    differ legitimately (plane: 4 km cell; cubed-sphere C4:
    ~1000 km cell; MPAS resolution 2: ~5000 km cell) so an exact
    match is not expected, but a 10⁴+ spread would signal a
    coupling-strength or unit bug in one of the dycores. Bound
    `3` (factor 1000) is loose enough to absorb cell-area scaling
    yet tight enough to catch a sign-of-magnitude unit error."""
    responses = {
        n: r["max_w_signed"] for n, r in all_grid_responses.items()
    }
    # All > 0 guaranteed by the companion sign test, but guard the
    # log10 against degenerate inputs in case this test runs alone.
    for name, value in responses.items():
        assert value > 0.0, (
            f"{name} max(w) = {value:.3e} <= 0; cannot compare "
            f"order-of-magnitude. See sign test for diagnosis."
        )
    log_vals = {n: math.log10(v) for n, v in responses.items()}
    spread = max(log_vals.values()) - min(log_vals.values())
    assert spread < 3.0, (
        f"Cross-grid buoyancy response spread = {spread:.2f} "
        f"orders of magnitude (> 3). max(w) per grid: "
        f"{ {n: f'{v:.3e}' for n, v in responses.items()} }. "
        f"log10: { {n: f'{v:.2f}' for n, v in log_vals.items()} }."
    )
