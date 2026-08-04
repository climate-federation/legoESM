"""omega = Dp/Dt (CMIP6 ``wap``) on the MPAS hydrostatic lane.

Two tiers, both non-vacuous:

* the SHARED assembler ``grids.vertical.compute_omega_total`` against
  hand-arithmetic on both coordinate families -- every one of the three
  terms of the material derivative is pinned by a number written out in
  the test, so a dropped/mis-scaled term cannot pass;
* the DYCORE export ``MPASPrimitiveEquationModel.diagnose_omega`` (the
  symbol ``model_driver._feed_mpas_cmip_accumulators`` actually calls)
  against two configurations whose exact answer is derivable by hand from
  the continuity closure alone:

  1. horizontal mass divergence confined to the TOP layer over a UNIFORM
     surface pressure.  Then ``v.grad(ln p_s) = 0`` identically (ln p_s is
     constant, so ``div(u ln_ps_e) - ln_ps div(u)`` cancels exactly) and the
     whole column's mass change is generated above every level below the
     source layer, so ``omega[k] == dp_s/dt`` EXACTLY for k >= 1.  (Uses
     only ``sigma_full = 0.5*(sigma_half[k] + sigma_half[k+1])``, which
     ``create_sigma_coordinate`` guarantees for uniform AND refined grids.)
  2. a VERTICALLY UNIFORM wind over a SLOPING surface pressure.  Then
     ``div(u dp_e)_k = dsigma_k * div(u p_s_e)``, the continuity closure's
     ``frac_k * D_total`` cancels the cumsum term for term, and sigma-dot is
     ZERO at every interface -- so omega collapses to
     ``sigma_full * (dp_s/dt + p_s * v.grad(ln p_s))`` and the horizontal
     pressure-advection term is exposed on its own.  The test computes the
     no-term-2 alternative too and asserts the dycore does NOT match it, so
     it is provably red if that term is dropped.

Sign convention under test throughout: CMIP6 ``wap`` is positive DOWNWARD
(pressure increasing following the flow); an ASCENDING column is NEGATIVE.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.core.field import Field
from legoesm.core.operators_voronoi import cell_to_edge_avg, divergence_cell
from legoesm.core.state import MPASHydrostaticState
from legoesm.grids.factory import create_grid
from legoesm.grids.vertical import (
    compute_omega_total,
    create_hybrid_coordinate,
    create_sigma_coordinate,
)

NLEV = 6


# =====================================================================
# Tier 1 -- the shared assembler, hand arithmetic
# =====================================================================

def test_compute_omega_total_sigma_hand_arithmetic():
    """Pure sigma, every term written out by hand.

    sigma_half = [.2 .4 .6 .8 1.]  ->  sigma_full = [.3 .5 .7 .9]
    p_s = 1e5 Pa,  dp_s/dt = -20 Pa/s,  v.grad(ln p_s) = 1e-5 1/s
    sigma_dot_half = [0, -1e-4, -2e-4, -1e-4, 0] 1/s

      term 1  sigma_full * dp_s/dt        = [ -6,   -10,   -14,   -18  ]
      term 3  p_s * mean(sigma_dot_half)  = [ -5,   -15,   -15,   -5   ]
      term 2  sigma_full * p_s * 1e-5     = [ +0.3, +0.5,  +0.7,  +0.9 ]
                                     omega = [-10.7, -24.5, -28.3, -22.1]
    """
    coord = create_sigma_coordinate(4, sigma_top=0.2, dtype=jnp.float64)
    np.testing.assert_allclose(
        np.asarray(coord.sigma_full), [0.3, 0.5, 0.7, 0.9], atol=1e-12)

    p_s = jnp.asarray([1.0e5])
    dp_s_dt = jnp.asarray([-20.0])
    sigma_dot = jnp.asarray([[0.0, -1.0e-4, -2.0e-4, -1.0e-4, 0.0]])
    v_grad_lnps = jnp.full((1, 4), 1.0e-5)

    omega = compute_omega_total(sigma_dot, p_s, dp_s_dt, v_grad_lnps, coord)
    np.testing.assert_allclose(
        np.asarray(omega)[0], [-10.7, -24.5, -28.3, -22.1], atol=1e-10)


def test_compute_omega_total_hybrid_hand_arithmetic():
    """Hybrid, every term written out by hand.

    B_half = [0 .2 .5 .8 1.]  ->  B_full = [.1 .35 .65 .9]
    (A contributes NOTHING to omega: A*p_ref is constant in space and time.)
    p_s = 1e5 Pa,  dp_s/dt = -20 Pa/s,  v.grad(ln p_s) = 1e-5 1/s
    mass_flux_half = [0, -10, -20, -10, 0] Pa/s

      term 1  B_full * dp_s/dt      = [ -2,   -7,     -13,    -18  ]
      term 3  mean(mass_flux_half)  = [ -5,   -15,    -15,    -5   ]
      term 2  B_full * p_s * 1e-5   = [ +0.1, +0.35,  +0.65,  +0.9 ]
                                omega = [ -6.9, -21.65, -27.35, -22.1]
    """
    coord = create_hybrid_coordinate(
        4,
        A_half=[0.01, 0.008, 0.005, 0.002, 0.0],
        B_half=[0.0, 0.2, 0.5, 0.8, 1.0],
        dtype=jnp.float64,
    )
    np.testing.assert_allclose(
        np.asarray(coord.B_full), [0.1, 0.35, 0.65, 0.9], atol=1e-12)

    p_s = jnp.asarray([1.0e5])
    dp_s_dt = jnp.asarray([-20.0])
    mass_flux = jnp.asarray([[0.0, -10.0, -20.0, -10.0, 0.0]])
    v_grad_lnps = jnp.full((1, 4), 1.0e-5)

    omega = compute_omega_total(mass_flux, p_s, dp_s_dt, v_grad_lnps, coord)
    np.testing.assert_allclose(
        np.asarray(omega)[0], [-6.9, -21.65, -27.35, -22.1], atol=1e-10)


def test_compute_omega_total_hybrid_ignores_a_half():
    """A_half must not move omega: ``A*p_ref`` has no space or time
    dependence, so it drops out of every term of Dp/Dt.  Red if a future
    edit reaches for ``A_full`` (or ``p_full``) instead of ``B_full``."""
    kw = dict(B_half=[0.0, 0.2, 0.5, 0.8, 1.0], dtype=jnp.float64)
    c1 = create_hybrid_coordinate(
        4, A_half=[0.01, 0.008, 0.005, 0.002, 0.0], **kw)
    c2 = create_hybrid_coordinate(
        4, A_half=[0.30, 0.200, 0.100, 0.050, 0.0], **kw)
    args = (jnp.asarray([[0.0, -10.0, -20.0, -10.0, 0.0]]),
            jnp.asarray([1.0e5]), jnp.asarray([-20.0]),
            jnp.full((1, 4), 1.0e-5))
    np.testing.assert_allclose(
        np.asarray(compute_omega_total(*args, c1)),
        np.asarray(compute_omega_total(*args, c2)), atol=1e-12)


def test_compute_omega_total_ascent_is_negative():
    """The sign contract, stated as physics rather than as arithmetic.

    A column with NO surface-pressure change and NO pressure advection but
    a purely UPWARD vertical transport (sigma decreasing following the
    flow, sigma_dot < 0 -- sigma increases downward here) must report a
    NEGATIVE wap; reversing sigma_dot must flip it positive and nothing
    else.  A convention flip anywhere on this path goes red.
    """
    coord = create_sigma_coordinate(4, sigma_top=0.2, dtype=jnp.float64)
    p_s = jnp.asarray([1.0e5])
    zero = jnp.zeros((1,))
    up = jnp.asarray([[0.0, -1.0e-4, -2.0e-4, -1.0e-4, 0.0]])
    v0 = jnp.zeros((1, 4))

    rising = np.asarray(compute_omega_total(up, p_s, zero, v0, coord))
    sinking = np.asarray(compute_omega_total(-up, p_s, zero, v0, coord))
    assert (rising < 0).all(), f"ascent gave a non-negative wap: {rising}"
    assert (sinking > 0).all(), f"descent gave a non-positive wap: {sinking}"
    np.testing.assert_allclose(rising, -sinking, atol=1e-12)


def test_compute_omega_total_rejects_unknown_coordinate():
    """Dispatch hardening: never a silent default on an unrecognised
    coordinate -- that would publish a wrong-coordinate omega."""
    class _NotACoord:
        pass

    with pytest.raises(TypeError, match="unsupported vertical coordinate"):
        compute_omega_total(
            jnp.zeros((1, 5)), jnp.asarray([1.0e5]), jnp.asarray([0.0]),
            jnp.zeros((1, 4)), _NotACoord())


# =====================================================================
# Tier 2 -- the dycore export
# =====================================================================

@pytest.fixture(scope="module")
def mesh():
    # Level-2 SCVT mesh = 162 cells; small but a REAL Voronoi mesh with
    # real TRiSK operators (not a synthetic stand-in).
    return create_grid("mpas", 2, lloyd_iterations=10)


@pytest.fixture(scope="module")
def coord():
    return create_sigma_coordinate(NLEV, dtype=jnp.float64)


def _model(mesh, coord, **cfg_kw):
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        MPASPrimitiveEquationModel,
    )
    return MPASPrimitiveEquationModel(
        mesh, coord, MPASPrimitiveEquationConfig(**cfg_kw))


def _state(mesh, u, p_s, t_uniform=260.0):
    n, nlev = int(mesh.nCells), u.shape[-1]
    return MPASHydrostaticState(
        u=Field(data=jnp.asarray(u, dtype=jnp.float64), name="u",
                dims=("nEdges", "nlev"), units="m/s"),
        T=Field(data=jnp.full((n, nlev), t_uniform, dtype=jnp.float64), name="T",
                dims=("nCells", "nlev"), units="K"),
        p_s=Field(data=jnp.asarray(p_s, dtype=jnp.float64), name="p_s",
                  dims=("nCells",), units="Pa"),
        phis=Field(data=jnp.zeros(n, dtype=jnp.float64), name="phis",
                   dims=("nCells",), units="m^2/s^2"),
    )


def _top_layer_only_wind(mesh, nlev, seed=3, amp=5.0):
    """Wind confined to the TOP model layer -- puts ALL the horizontal mass
    divergence in layer 0."""
    rng = np.random.default_rng(seed)
    u = np.zeros((int(mesh.nEdges), nlev))
    u[:, 0] = amp * rng.standard_normal(int(mesh.nEdges))
    return u


def test_diagnose_omega_is_the_symbol_the_driver_calls(mesh, coord):
    """Attribution gate: the CMOR feed reaches omega through
    ``getattr(self.model, "diagnose_omega")`` on the dycore model."""
    import inspect

    from legoesm.driver import model_driver
    m = _model(mesh, coord)
    assert callable(getattr(m, "diagnose_omega", None))
    src = inspect.getsource(model_driver.ModelDriver._feed_mpas_cmip_accumulators)
    assert '"diagnose_omega"' in src, (
        "the MPAS CMOR feed no longer resolves the dycore omega export")
    assert "omega=omega" in src, "omega is not forwarded to the CMOR feed"


def test_omega_below_a_top_layer_mass_source_equals_dps_dt(mesh, coord):
    """ANALYTIC: with the horizontal mass divergence confined to layer 0 and
    a UNIFORM p_s, every level below layer 0 sees the ENTIRE column mass
    change above it, so omega == dp_s/dt exactly.

    Red if term 1 or term 3 is dropped or mis-weighted (they cancel to
    exactly 1 * dp_s/dt only when both are present with the right signs).
    """
    m = _model(mesh, coord)
    n = int(mesh.nCells)
    st = _state(mesh, _top_layer_only_wind(mesh, NLEV), np.full(n, 1.0e5))

    omega = np.asarray(m.diagnose_omega(st))
    dps_dt = np.asarray(m.tendencies(st).dp_s_dt.data)
    assert np.abs(dps_dt).max() > 1e-3, "degenerate fixture: dp_s/dt ~ 0"

    np.testing.assert_allclose(
        omega[:, 1:], np.broadcast_to(dps_dt[:, None], omega[:, 1:].shape),
        rtol=0, atol=1e-12 * np.abs(dps_dt).max())

    # Layer 0 itself sits INSIDE the source layer and sees only part of it:
    #   omega[0] = dp_s/dt * (sigma_full[0] + 0.5*(1 - sigma_half[1]))
    # (from omega_ps = sigma_full[0]*dp_s/dt and the half-level flux pair
    #  (0, (1 - sigma_half[1])*dp_s/dt) averaged to the full level).
    sf = np.asarray(coord.sigma_full)
    sh = np.asarray(coord.sigma_half)
    frac = sf[0] + 0.5 * (1.0 - sh[1])
    assert 0.0 < frac < 1.0
    np.testing.assert_allclose(
        omega[:, 0], dps_dt * frac, rtol=0,
        atol=1e-12 * np.abs(dps_dt).max())


def test_omega_sign_is_negative_for_an_ascending_column(mesh, coord):
    """Mass EXPORTED aloft (dp_s/dt < 0) means the air below rises to
    replace it -> NEGATIVE wap; mass IMPORTED aloft loads the column ->
    POSITIVE wap.  Both branches must be populated by the fixture, so this
    cannot pass vacuously."""
    m = _model(mesh, coord)
    n = int(mesh.nCells)
    st = _state(mesh, _top_layer_only_wind(mesh, NLEV), np.full(n, 1.0e5))

    omega = np.asarray(m.diagnose_omega(st))
    dps_dt = np.asarray(m.tendencies(st).dp_s_dt.data)
    up = dps_dt < 0.0
    down = dps_dt > 0.0
    assert up.sum() > 10 and down.sum() > 10, "fixture covers only one sign"

    assert (omega[up] < 0.0).all(), (
        "an ASCENDING column reported a non-negative wap -- CMIP6 wap is "
        "positive DOWNWARD")
    assert (omega[down] > 0.0).all(), (
        "a SUBSIDING column reported a non-positive wap")


def test_omega_carries_the_horizontal_pressure_advection_term(mesh, coord):
    """ANALYTIC + provably non-vacuous.

    A VERTICALLY UNIFORM wind makes the flux-form layer-mass divergence
    ``div(u dp_e)_k = dsigma_k * div(u p_s_e)``, which the continuity
    closure's ``frac_k * D_total - cumsum_k`` cancels EXACTLY -> sigma-dot
    is zero at every interface and term 3 vanishes.  omega must then be

        omega_k = sigma_full[k] * (dp_s/dt + p_s * v.grad(ln p_s))

    with ``v.grad(ln p_s)`` rebuilt here from the PUBLIC Voronoi operators,
    independently of the dycore's batched-divergence assembly.  The test
    also builds the no-term-2 alternative and asserts the dycore does NOT
    match it -- i.e. it is red by many orders of magnitude if the
    horizontal pressure-advection term is dropped.
    """
    m = _model(mesh, coord)
    ne = int(mesh.nEdges)
    rng = np.random.default_rng(11)
    u_col = 8.0 * rng.standard_normal(ne)
    u = np.repeat(u_col[:, None], NLEV, axis=1)      # uniform in the vertical
    # fp64 THROUGHOUT: the mesh coordinate arrays are float32, and
    # ``v.grad(ln p_s) = div(u ln_ps_e) - ln_ps div(u)`` is a ~4-digit
    # cancellation (G ~ 5e-8 from two ~2e-4 terms).  Rebuilding it in
    # float32 costs ~1e-6 Pa/s and would make this test look like a dycore
    # defect; the dycore itself takes ln p_s from the fp64 state.
    lat = np.asarray(mesh.latCell, dtype=np.float64)
    lon = np.asarray(mesh.lonCell, dtype=np.float64)
    p_s = 1.0e5 + 3.0e3 * np.cos(lat) * np.cos(lon)  # SLOPING surface pressure

    st = _state(mesh, u, p_s)
    omega = np.asarray(m.diagnose_omega(st))
    dps_dt = np.asarray(m.tendencies(st).dp_s_dt.data)

    # Independent v.grad(ln p_s) = div(u ln_ps_e) - ln_ps div(u).
    ln_ps = jnp.log(jnp.asarray(p_s, dtype=jnp.float64))
    u0 = jnp.asarray(u_col, dtype=jnp.float64)
    v_grad_lnps = np.asarray(
        divergence_cell(u0 * cell_to_edge_avg(ln_ps, mesh), mesh)
        - ln_ps * divergence_cell(u0, mesh))

    sf = np.asarray(coord.sigma_full)
    term2 = p_s * v_grad_lnps                        # [Pa/s], per column
    expected = sf[None, :] * (dps_dt + term2)[:, None]
    without_term2 = sf[None, :] * dps_dt[:, None]

    scale = np.abs(omega).max()
    np.testing.assert_allclose(omega, expected, rtol=0, atol=1e-10 * scale)

    # Non-vacuity: the term the test is here to protect is >= 6 orders of
    # magnitude above the tolerance just used.
    miss = np.abs(omega - without_term2).max()
    assert miss > 1e-4 * scale, (
        "the horizontal pressure-advection term is too small in this "
        f"fixture to prove anything (miss={miss:.3e}, scale={scale:.3e})")


def test_omega_runs_on_the_hybrid_coordinate(mesh):
    """The hybrid branch must produce a finite omega with the same sign
    law (the production AMIP deck runs hybrid, not pure sigma)."""
    from legoesm.grids.vertical import standard_hybrid_levels
    _std = standard_hybrid_levels(NLEV)
    hyb = create_hybrid_coordinate(
        NLEV, _std.A_half, _std.B_half, p_ref=_std.p_ref, dtype=jnp.float64)
    m = _model(mesh, hyb)
    n = int(mesh.nCells)
    st = _state(mesh, _top_layer_only_wind(mesh, NLEV), np.full(n, 1.0e5))

    omega = np.asarray(m.diagnose_omega(st))
    dps_dt = np.asarray(m.tendencies(st).dp_s_dt.data)
    assert omega.shape == (n, NLEV)
    assert np.isfinite(omega).all()
    up = dps_dt < 0.0
    assert up.sum() > 10
    assert (omega[up] < 0.0).all(), "hybrid ascent reported a positive wap"


# =====================================================================
# Contract: the gate is static and the hot path is untouched
# =====================================================================

def test_default_tendency_call_is_unchanged(mesh, coord):
    """``return_omega`` defaults to False and does NOT change the return
    type -- every existing caller (RK stages, the SPMD sharder, the MPI
    step) keeps its ABI."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASHydrostaticTendencies,
        MPASPrimitiveEquationConfig,
        mpas_hydrostatic_tendencies,
    )
    n = int(mesh.nCells)
    st = _state(mesh, _top_layer_only_wind(mesh, NLEV), np.full(n, 1.0e5))
    cfg = MPASPrimitiveEquationConfig()

    plain = mpas_hydrostatic_tendencies(st, mesh, coord, cfg)
    # NamedTuples ARE tuples, so assert the concrete type, not tuple-ness.
    assert type(plain) is MPASHydrostaticTendencies, (
        f"the default return type changed to {type(plain).__name__} -- "
        "existing callers would break")

    pair = mpas_hydrostatic_tendencies(
        st, mesh, coord, cfg, return_omega=True)
    assert type(pair) is tuple and len(pair) == 2
    tend, omega = pair
    assert type(tend) is MPASHydrostaticTendencies
    assert omega.shape == (n, NLEV)
    # The gate must not perturb the tendencies themselves.
    for name in ("du_dt", "dT_dt", "dp_s_dt"):
        np.testing.assert_array_equal(
            np.asarray(getattr(plain, name).data),
            np.asarray(getattr(tend, name).data))


def test_omega_jit_parity_and_grad(mesh, coord):
    """JIT parity (no tracer-dependent Python branching) and a finite,
    non-zero reverse-mode gradient through the whole omega path."""
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        mpas_hydrostatic_tendencies,
    )
    n = int(mesh.nCells)
    st = _state(mesh, _top_layer_only_wind(mesh, NLEV), np.full(n, 1.0e5))
    cfg = MPASPrimitiveEquationConfig()

    def _omega(u_data, ps_data):
        s = st._replace(u=st.u.replace(data=u_data),
                        p_s=st.p_s.replace(data=ps_data))
        return mpas_hydrostatic_tendencies(
            s, mesh, coord, cfg, return_omega=True)[1]

    eager = np.asarray(_omega(st.u.data, st.p_s.data))
    jitted = np.asarray(jax.jit(_omega)(st.u.data, st.p_s.data))
    np.testing.assert_allclose(eager, jitted, rtol=0,
                               atol=1e-12 * np.abs(eager).max())

    g_u, g_ps = jax.grad(
        lambda a, b: jnp.sum(_omega(a, b) ** 2), argnums=(0, 1),
    )(st.u.data, st.p_s.data)
    assert np.isfinite(np.asarray(g_u)).all()
    assert np.isfinite(np.asarray(g_ps)).all()
    assert np.abs(np.asarray(g_u)).max() > 0.0, "no gradient reaches u"
    assert np.abs(np.asarray(g_ps)).max() > 0.0, "no gradient reaches p_s"


def test_omega_is_independent_of_dt(mesh, coord):
    """``diagnose_omega`` passes ``dt=0.0``.  That is only safe because
    ``dt`` enters the RHS solely through the APVM PV-flux upwinding (a
    du_dt term); omega is built from ``div(u dp_e)``, its cumsum and
    ``v.grad(ln p_s)``, all read off the raw state.  Proven, not argued:
    with APVM ACTIVE omega must be bit-identical at dt=0 and dt=600, and
    the control confirms du_dt does respond (otherwise the first
    assertion would be vacuous).
    """
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        mpas_hydrostatic_tendencies,
    )
    n, ne = int(mesh.nCells), int(mesh.nEdges)
    rng = np.random.default_rng(5)
    lat = np.asarray(mesh.latCell, dtype=np.float64)
    lon = np.asarray(mesh.lonCell, dtype=np.float64)
    st = _state(mesh, 6.0 * rng.standard_normal((ne, NLEV)),
                1.0e5 + 3.0e3 * np.cos(lat) * np.cos(lon))
    assert n == int(mesh.nCells)
    cfg = MPASPrimitiveEquationConfig(apvm_scale=0.5)

    o0 = np.asarray(mpas_hydrostatic_tendencies(
        st, mesh, coord, cfg, dt=0.0, return_omega=True)[1])
    o6 = np.asarray(mpas_hydrostatic_tendencies(
        st, mesh, coord, cfg, dt=600.0, return_omega=True)[1])
    np.testing.assert_array_equal(o0, o6)

    d0 = np.asarray(mpas_hydrostatic_tendencies(
        st, mesh, coord, cfg, dt=0.0).du_dt.data)
    d6 = np.asarray(mpas_hydrostatic_tendencies(
        st, mesh, coord, cfg, dt=600.0).du_dt.data)
    assert np.abs(d0 - d6).max() > 0.0, (
        "APVM is inactive in this fixture — the dt-independence assertion "
        "above proves nothing")


def test_omega_excludes_the_p_s_hyperdiffusion(mesh, coord):
    """omega is pinned to the CONTINUITY-closure dp_s/dt, i.e. BEFORE the
    ``nu_del4_ps`` filter: that filter moves p_s with no matching vertical
    mass flux, so folding it in would leave omega's three terms mutually
    inconsistent.

    nu_del4_ps = 1e20 is far outside any physical setting — it is chosen
    so the CONTROL is decisive: it shifts ``dp_s_dt`` by ~35 % of its own
    magnitude on this mesh while omega must stay BIT-IDENTICAL.  A rougher
    p_s (grid-scale noise) is needed for del4 to bite at all on a level-2
    mesh.
    """
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationConfig,
        mpas_hydrostatic_tendencies,
    )
    n, ne = int(mesh.nCells), int(mesh.nEdges)
    rng = np.random.default_rng(5)
    st = _state(mesh, 6.0 * rng.standard_normal((ne, NLEV)),
                1.0e5 + 500.0 * rng.standard_normal(n))
    off = MPASPrimitiveEquationConfig(nu_del4_ps=0.0)
    on = MPASPrimitiveEquationConfig(nu_del4_ps=1.0e20)

    o_off = np.asarray(mpas_hydrostatic_tendencies(
        st, mesh, coord, off, return_omega=True)[1])
    o_on = np.asarray(mpas_hydrostatic_tendencies(
        st, mesh, coord, on, return_omega=True)[1])
    np.testing.assert_array_equal(o_off, o_on)

    p_off = np.asarray(
        mpas_hydrostatic_tendencies(st, mesh, coord, off).dp_s_dt.data)
    p_on = np.asarray(
        mpas_hydrostatic_tendencies(st, mesh, coord, on).dp_s_dt.data)
    moved = np.abs(p_off - p_on).max() / np.abs(p_off).max()
    assert moved > 0.05, (
        f"the p_s hyperdiffusion barely moved dp_s_dt ({100 * moved:.2f} %) "
        "— the exclusion assertion above proves nothing")
