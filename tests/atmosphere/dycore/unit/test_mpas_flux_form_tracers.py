"""#1354: mass-CONSISTENT flux-form tracer transport on the MPAS hydrostatic PE.

The MPAS PE carries layer mass with the flux-form continuity
``div_dp = div(u δp_edge)`` (``primitive_eq_mpas`` step 4) but advected tracers
with the ADVECTIVE ``-(div(q u) - q div(u))``, which differences ``q`` against
the UNWEIGHTED ``div(u)``.  That mismatch makes the tracer update non-CONSERVATIVE
on this dycore's own discrete mass budget.

``MPASPrimitiveEquationConfig.moisture_flux_form=True`` routes the per-MASS
tracers through :func:`tracer_flux_form_tendency`, which transports ``q·δp``
with the SAME edge mass flux and the SAME half-level vertical mass flux ``F``.

WHICH PROPERTY ACTUALLY SEPARATES THE TWO — read this before adding a test.
Both operators are consistent discretisations of the SAME continuous equation
(``∂q/∂t = -v·∇q - σ̇ ∂q/∂σ`` follows from the flux form minus ``q``×
continuity), and BOTH are free-stream preserving: ``-(div(q u) - q div(u))``
is identically zero for constant ``q``, measured at 2.1e-20 /s here.  So a
constant-``q`` test can NOT distinguish them — it is a necessary check on the
new kernel, not evidence of the defect.  The property the advective form lacks
is DISCRETE CONSERVATION of ``∫ q δp dA``: measured below at 2.1e+10 (σ) /
4.0e+10 (hybrid) Pa·m²/s against a total tracer mass of ~1.3e+17 Pa·m², i.e.
~1.5e-7 per second — a spurious source ~15 orders of magnitude larger than the
flux form's round-off residual.

What is asserted here (each an independent property, not a restatement):

1. FREE-STREAM: a spatially + vertically constant ``q`` has an exactly zero
   tendency under a nontrivial wind over nontrivial terrain.  NECESSARY (a
   conservative-but-not-free-stream operator manufactures extrema out of a
   uniform field), but NOT a discriminator — see above.
2. CONSERVATION (the distinguishing property): the AREA-weighted (``areaCell``
   varies ~1.5x on an SCVT mesh), δp-weighted global tracer mass tendency
   vanishes to round-off, and the advective form's does NOT — asserted as a
   contrast so the test is non-vacuous and the defect is measured, not assumed.
3. TRUNCATION: on a smooth field the two operators agree to a small relative
   difference — flux form is a re-discretisation, not different physics.
4. DEFAULT OFF is bit-identical to the pre-#1354 operator.
5. Per-VOLUME numbers ``N_c``/``N_r`` are EXCLUDED (kept advective).
6. jit parity, and ``jax.grad`` / ``check_grads`` differentiability.

x64 is required: these are round-off-level conservation assertions.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.field import Field
from legoesm.core.state import MPASHydrostaticState
from legoesm.grids.vertical import (
    create_hybrid_coordinate,
    create_sigma_coordinate,
)
from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
    MPASPrimitiveEquationConfig,
    mpas_hydrostatic_tendencies,
)
from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
    tracer_flux_form_tendency,
)

NLEV = 6
_F64 = jnp.float64


# =========================================================================
# Fixtures
# =========================================================================

def _mesh(level: int = 2):
    """Level-2 icosahedral SCVT mesh (162 cells) — small but non-uniform."""
    return create_voronoi_mesh(level, lloyd_iterations=5)


def _sigma(nlev: int = NLEV):
    return create_sigma_coordinate(nlev, dtype=_F64)


def _hybrid(nlev: int = NLEV):
    """A genuinely hybrid A/B pair (A ≠ 0 aloft) so ``δp`` is NOT ∝ p_s."""
    eta = np.linspace(0.0, 1.0, nlev + 1)
    # B ramps from 0 at the top to 1 at the surface; A carries the remainder,
    # so the upper layers are ~pure pressure levels.
    B = eta ** 3
    A = (eta - B) * 0.9 + 0.01 * (1.0 - eta)
    A = A - A[0] + 0.002          # A_half[0] > 0 (finite p_top)
    return create_hybrid_coordinate(nlev, A, B, dtype=_F64)


def _state(mesh, coord, *, q_const=None, seed=0, nlev=NLEV):
    """Nontrivial state: sheared wind, terrain-modulated p_s, structured T.

    ``q_const`` -> a single ``q_v`` tracer set to that constant everywhere
    (the free-stream probe).  Otherwise a smooth, strongly varying ``q_v``.
    """
    rng = np.random.default_rng(seed)
    n_cells, n_edges = mesh.nCells, mesh.nEdges
    lat_c = np.asarray(mesh.latCell)
    lon_c = np.asarray(mesh.lonCell)

    # Surface pressure with a large (±80 hPa) terrain-like signal: ∇δp ≠ 0 is
    # exactly the regime where advective and flux form diverge.
    p_s = 1.0e5 + 8.0e3 * np.cos(3.0 * lon_c) * np.cos(lat_c) ** 2
    phis = constants.g * 1500.0 * np.cos(3.0 * lon_c) * np.cos(lat_c) ** 2
    T = 250.0 + 30.0 * np.cos(lat_c)[:, None] * np.linspace(
        1.0, -1.0, nlev)[None, :]
    # Divergent + rotational edge wind, sheared in the vertical.
    u = (20.0 * rng.standard_normal((n_edges, nlev))
         * np.linspace(0.4, 1.6, nlev)[None, :])

    if q_const is None:
        q = 1.0e-2 * (
            0.5 + 0.4 * np.cos(2.0 * lon_c)[:, None] * np.cos(lat_c)[:, None]
        ) * np.linspace(1.0, 0.05, nlev)[None, :]
    else:
        q = np.full((n_cells, nlev), float(q_const))

    def _f(a, name, dims, units):
        return Field(data=jnp.asarray(a, dtype=_F64), name=name, dims=dims,
                     units=units)

    return MPASHydrostaticState(
        u=_f(u, "u", ("nEdges", "nlev"), "m/s"),
        T=_f(T, "T", ("nCells", "nlev"), "K"),
        p_s=_f(p_s, "p_s", ("nCells",), "Pa"),
        phis=_f(phis, "phis", ("nCells",), "m2/s2"),
        tracers={"q_v": _f(q, "q_v", ("nCells", "nlev"), "kg/kg")},
    )


def _cfg(**kw):
    """Config with every *extra* term off, so the tracer block is isolated."""
    base = dict(nu_del2=0.0, nu_del4=0.0, nu_del4_ps=0.0, K_h=0.0,
                nu_vert4_T=0.0, fix_mass=False)
    base.update(kw)
    return MPASPrimitiveEquationConfig(**base)


def _dq(state, mesh, coord, *, flux_form, **cfgkw):
    tend = mpas_hydrostatic_tendencies(
        state, mesh, coord, _cfg(moisture_flux_form=flux_form, **cfgkw))
    return {k: v.data for k, v in tend.tracer_tendencies.items()}


def _layer_mass(state, coord):
    """δp at cells [Pa] for either coordinate — the model's own definition."""
    return coord.layer_thickness_dp(state.p_s.data)


def _d_dp_dt(coord, dps_dt):
    """d(δp)/dt implied by the model's own dp_s/dt (σ: Δσ·ṗ_s; hybrid: dB·ṗ_s)."""
    w = coord.dB if hasattr(coord, "dB") else coord.dsigma
    return dps_dt[:, None] * jnp.asarray(w, dtype=_F64)[None, :]


_COORDS = [("sigma", _sigma), ("hybrid", _hybrid)]


# =========================================================================
# 1. Free-stream preservation (NECESSARY, not a discriminator)
# =========================================================================

@pytest.mark.parametrize("cname,cbuild", _COORDS)
def test_constant_q_is_exactly_preserved(cname, cbuild):
    """A spatially AND vertically constant q must have dq/dt == 0 to round-off
    under an arbitrary wind over terrain.  Tolerance is stated relative to the
    tracer value: 1e-14 * q per second (fp64 eps is 2.2e-16; the divergence
    reduction sums up to ``maxEdges`` terms, so a couple of decades of
    headroom).

    NOT a discriminator: the advective operator is free-stream preserving too
    (it discretises ``-v·∇q``, identically zero here) and its residual is
    printed alongside to keep that on the record.  This is a NECESSARY check on
    the new kernel — a flux-form scheme whose ``q_edge``/``q_upwind`` did not
    reduce exactly to ``q`` would manufacture extrema out of a uniform field —
    but the property that separates the two operators is conservation, asserted
    in section 2 below.
    """
    mesh, coord = _mesh(), cbuild()
    q0 = 1.0e-2
    st = _state(mesh, coord, q_const=q0)

    dq_flux = _dq(st, mesh, coord, flux_form=True)["q_v"]
    dq_adv = _dq(st, mesh, coord, flux_form=False)["q_v"]

    res_flux = float(jnp.max(jnp.abs(dq_flux))) / q0
    res_adv = float(jnp.max(jnp.abs(dq_adv))) / q0
    print(f"[{cname}] constant-q max|dq/dt|/q: flux={res_flux:.3e} /s, "
          f"advective={res_adv:.3e} /s (both free-stream; see docstring)")

    assert res_flux < 1e-14, (
        f"flux form must preserve a constant tracer: {res_flux:.3e} /s")


@pytest.mark.parametrize("cname,cbuild", _COORDS)
def test_constant_q_preserved_with_flat_terrain_too(cname, cbuild):
    """Guard against a fixture that only works because p_s varies: with a
    uniform p_s the advective form is (nearly) fine, so this asserts the flux
    form is not *worse* in the easy regime."""
    mesh, coord = _mesh(), cbuild()
    q0 = 5.0e-3
    st = _state(mesh, coord, q_const=q0)
    st = st._replace(
        p_s=st.p_s.replace(data=jnp.full_like(st.p_s.data, 1.0e5)),
        phis=st.phis.replace(data=jnp.zeros_like(st.phis.data)),
    )
    res = float(jnp.max(jnp.abs(_dq(st, mesh, coord, flux_form=True)["q_v"])))
    assert res / q0 < 1e-14, f"[{cname}] flat-terrain residual {res:.3e}"


# =========================================================================
# 2. Global column-tracer-mass conservation, AREA weighted
#    -- the property that actually separates flux form from advective
# =========================================================================

@pytest.mark.parametrize("cname,cbuild", _COORDS)
def test_global_tracer_mass_tendency_vanishes(cname, cbuild):
    """d/dt Σ_cells areaCell · Σ_k q δp must vanish to round-off.

    The conserved quantity is the tracer MASS ``∫ q δp/g dA``.  Its rate is
    ``Σ area Σ_k [δp·dq/dt + q·d(δp)/dt]``, with ``d(δp)/dt`` taken from the
    model's own ``dp_s/dt``.  ``areaCell`` varies by ~1.5x on this SCVT mesh,
    so an UNWEIGHTED sum would be meaningless (CLAUDE.md area-weighting gate).
    """
    mesh, coord = _mesh(), cbuild()
    st = _state(mesh, coord, seed=3)
    tend = mpas_hydrostatic_tendencies(
        st, mesh, coord, _cfg(moisture_flux_form=True))

    q = st.tracers["q_v"].data
    dq_dt = tend.tracer_tendencies["q_v"].data
    dp = _layer_mass(st, coord)
    ddp_dt = _d_dp_dt(coord, tend.dp_s_dt.data)

    area = mesh.areaCell.astype(_F64)[:, None]
    rate = float(jnp.sum(area * (dp * dq_dt + q * ddp_dt)))
    scale = float(jnp.sum(area * dp * q))
    gross = float(jnp.sum(jnp.abs(area * dp * dq_dt)))
    print(f"[{cname}] d(mass)/dt = {rate:.6e}, mass = {scale:.6e}, "
          f"gross |terms| = {gross:.6e}, rate/gross = {rate / gross:.3e}")

    assert abs(rate) < 1e-12 * gross, (
        f"[{cname}] tracer mass not conserved: {rate:.3e} vs gross {gross:.3e}")


@pytest.mark.parametrize("cname,cbuild", _COORDS)
def test_column_mass_tendency_beats_advective(cname, cbuild):
    """Non-vacuity for the conservation test: the advective operator leaves a
    much larger global residual on the SAME state."""
    mesh, coord = _mesh(), cbuild()
    st = _state(mesh, coord, seed=3)
    q = st.tracers["q_v"].data
    dp = _layer_mass(st, coord)
    area = mesh.areaCell.astype(_F64)[:, None]

    out = {}
    for ff in (True, False):
        tend = mpas_hydrostatic_tendencies(
            st, mesh, coord, _cfg(moisture_flux_form=ff))
        dq_dt = tend.tracer_tendencies["q_v"].data
        ddp_dt = _d_dp_dt(coord, tend.dp_s_dt.data)
        out[ff] = abs(float(jnp.sum(area * (dp * dq_dt + q * ddp_dt))))
    print(f"[{cname}] |d(mass)/dt|: flux={out[True]:.3e} "
          f"advective={out[False]:.3e}")
    assert out[True] < 1e-6 * out[False]


# =========================================================================
# 3. Truncation-level agreement on a smooth field
# =========================================================================

@pytest.mark.parametrize("cname,cbuild", _COORDS)
def test_agrees_with_advective_to_truncation(cname, cbuild):
    """The two operators discretise the SAME continuous transport, so on a
    smooth tracer they must agree to a truncation-level difference — a large
    disagreement would mean one of them is wrong, not merely less conservative.
    """
    mesh, coord = _mesh(), cbuild()
    st = _state(mesh, coord, seed=1)
    a = _dq(st, mesh, coord, flux_form=False)["q_v"]
    b = _dq(st, mesh, coord, flux_form=True)["q_v"]
    rel = float(jnp.linalg.norm(b - a) / jnp.linalg.norm(a))
    print(f"[{cname}] ||flux - advective|| / ||advective|| = {rel:.4f}")
    # Coarse level-2 mesh + a strong 80 hPa p_s signal: a few tens of percent
    # is the expected truncation spread; an order-of-magnitude difference is
    # not.
    assert rel < 0.5, (
        f"[{cname}] operators disagree by {rel:.3f} (not truncation)")
    assert rel > 1e-6, "operators identical — the flag did nothing"


# =========================================================================
# 4. Default OFF is bit-identical
# =========================================================================

@pytest.mark.parametrize("cname,cbuild", _COORDS)
def test_default_off_is_bit_identical(cname, cbuild):
    """The whole new path sits behind a Python ``if`` on a STATIC bool, so with
    ``moisture_flux_form=False`` not one operation changes.  Compared against
    an INDEPENDENT reconstruction of the legacy operator
    (``tracer_horizontal_advection`` + the dycore's own vertical advection)."""
    from legoesm.atmosphere.dynamics.gcm.tracer_transport_mpas import (
        tracer_horizontal_advection,
    )
    from legoesm.grids.vertical import (
        HybridSigmaPressureCoordinate, compute_mass_flux_from_cumsum,
        compute_sigma_dot_from_cumsum, vertical_advection,
        vertical_advection_hybrid,
    )
    from legoesm.core.operators_voronoi import (
        cell_to_edge_avg_3d, divergence_cell_3d,
    )

    mesh, coord = _mesh(), cbuild()
    st = _state(mesh, coord, seed=7)
    got = _dq(st, mesh, coord, flux_form=False)["q_v"]

    p_s = st.p_s.data
    u = st.u.data
    q = st.tracers["q_v"].data[..., None]
    hyb = isinstance(coord, HybridSigmaPressureCoordinate)
    if hyb:
        dp_e = cell_to_edge_avg_3d(coord.layer_thickness_dp(p_s), mesh)
    else:
        dp_e = (cell_to_edge_avg_3d(p_s[:, None], mesh)
                * coord.dsigma.astype(p_s.dtype)[None, :])
    cum = jnp.cumsum(divergence_cell_3d(u * dp_e, mesh), axis=-1)
    tot = cum[..., -1:]
    horiz = tracer_horizontal_advection(q, u, mesh)
    if hyb:
        vert_vel = compute_mass_flux_from_cumsum(cum, tot, coord)
        vert = jax.vmap(
            lambda qk: vertical_advection_hybrid(qk, vert_vel, p_s, coord),
            in_axes=-1, out_axes=-1)(q)
    else:
        vert_vel = compute_sigma_dot_from_cumsum(cum, tot, p_s, coord)
        vert = jax.vmap(lambda qk: vertical_advection(qk, vert_vel, coord),
                        in_axes=-1, out_axes=-1)(q)
    expect = (horiz + vert)[..., 0]

    assert jnp.array_equal(got, expect), (
        f"[{cname}] default path changed; max diff "
        f"{float(jnp.max(jnp.abs(got - expect))):.3e}")


# =========================================================================
# 5. Per-VOLUME number concentrations are EXCLUDED
# =========================================================================

def test_per_volume_numbers_stay_advective():
    """``N_c``/``N_r`` are per-VOLUME [#/m³] (``HydrometeorState``), so a
    δp-weighted transport conserves ``∫ N dp/g``, which has no physical meaning
    for them.  They must keep the advective operator even with the flag on —
    the same exclusion ``conservation.BORROW_ELIGIBLE_TRACERS`` encodes, and
    the exact class of error the 2026-07 conserving-borrow defect was.
    """
    mesh, coord = _mesh(), _sigma()
    st = _state(mesh, coord, seed=2)
    qv = st.tracers["q_v"].data
    st = st._replace(tracers={
        "q_v": st.tracers["q_v"],
        "q_i": st.tracers["q_v"].replace(data=0.3 * qv, name="q_i"),
        "N_i": st.tracers["q_v"].replace(data=1.0e7 * qv, name="N_i"),
        "N_c": st.tracers["q_v"].replace(data=1.0e8 * qv, name="N_c"),
        "N_r": st.tracers["q_v"].replace(data=1.0e6 * qv, name="N_r"),
    })
    on = _dq(st, mesh, coord, flux_form=True)
    off = _dq(st, mesh, coord, flux_form=False)

    for k in ("N_c", "N_r"):
        assert jnp.array_equal(on[k], off[k]), (
            f"{k} is per-VOLUME and must NOT be routed through the "
            "mass-weighted flux form")
    for k in ("q_v", "q_i", "N_i"):
        assert not jnp.allclose(on[k], off[k]), (
            f"{k} is per-MASS and must be routed through the flux form")


def test_all_water_species_are_free_stream_preserved():
    """Every per-mass water species (not just q_v) gets the flux form."""
    mesh, coord = _mesh(), _sigma()
    st = _state(mesh, coord, q_const=1.0e-2)
    names = ("q_v", "q_c", "q_r", "q_i", "q_s", "q_g", "N_i", "N_s", "N_g")
    base = st.tracers["q_v"]
    st = st._replace(tracers={
        k: base.replace(data=jnp.full_like(base.data, 1.0e-2), name=k)
        for k in names
    })
    out = _dq(st, mesh, coord, flux_form=True)
    for k in names:
        r = float(jnp.max(jnp.abs(out[k]))) / 1.0e-2
        assert r < 1e-14, f"{k} constant-tracer residual {r:.3e} /s"


# =========================================================================
# 6. jit parity + differentiability
# =========================================================================

@pytest.mark.parametrize("cname,cbuild", _COORDS)
def test_jit_parity(cname, cbuild):
    mesh, coord = _mesh(), cbuild()
    st = _state(mesh, coord, seed=4)
    cfg = _cfg(moisture_flux_form=True)

    def f(s):
        return mpas_hydrostatic_tendencies(
            s, mesh, coord, cfg).tracer_tendencies["q_v"].data

    eager = f(st)
    jitted = jax.jit(f)(st)
    d = float(jnp.max(jnp.abs(eager - jitted)))
    scale = float(jnp.max(jnp.abs(eager)))
    print(f"[{cname}] jit-vs-eager max|Δ| = {d:.3e} (field max {scale:.3e})")
    assert d <= 1e-13 * scale, f"[{cname}] jit divergence {d:.3e}"


@pytest.mark.parametrize("cname,cbuild", _COORDS)
def test_grad_flows_through_flux_form(cname, cbuild):
    """The lane must stay ``jax.grad``-compatible, and the gradient must be
    non-degenerate w.r.t. BOTH the tracer and the wind (a zero would mean the
    new term dropped out of the graph)."""
    mesh, coord = _mesh(), cbuild()
    st = _state(mesh, coord, seed=5)
    cfg = _cfg(moisture_flux_form=True)

    def loss(q, u):
        s = st._replace(
            u=st.u.replace(data=u),
            tracers={"q_v": st.tracers["q_v"].replace(data=q)},
        )
        return jnp.sum(
            mpas_hydrostatic_tendencies(
                s, mesh, coord, cfg).tracer_tendencies["q_v"].data ** 2)

    gq, gu = jax.grad(loss, argnums=(0, 1))(
        st.tracers["q_v"].data, st.u.data)
    for name, g in (("q", gq), ("u", gu)):
        assert jnp.all(jnp.isfinite(g)), f"[{cname}] non-finite d/d{name}"
        assert float(jnp.max(jnp.abs(g))) > 0.0, f"[{cname}] zero d/d{name}"


def test_kernel_check_grads_order2():
    """``check_grads`` on the standalone kernel (fwd + rev, 2nd order).  It is
    pure ``jnp`` — no ``custom_vjp`` — but the gate is cheap and would catch a
    ``jnp.where`` / ``pad`` that silently kills a cotangent path."""
    from legoesm.core.operators_voronoi import (
        cell_to_edge_avg_3d, divergence_cell_3d,
    )
    from legoesm.grids.vertical import compute_sigma_dot_from_cumsum

    mesh, coord = _mesh(), _sigma()
    st = _state(mesh, coord, seed=6)
    p_s = st.p_s.data
    dp_cell = coord.layer_thickness_dp(p_s)
    dp_edge = (cell_to_edge_avg_3d(p_s[:, None], mesh)
               * coord.dsigma.astype(_F64)[None, :])
    q = st.tracers["q_v"].data[..., None]
    # Scale the wind down: check_grads' finite-difference reference needs the
    # perturbed upwind SELECTION to stay put; a huge |u| makes some interface
    # F sit within eps of 0 and the (correctly) non-smooth switch trips.
    u = 0.1 * st.u.data

    def f(q_, u_):
        # Recompute the mass fluxes from u_ so the derivative covers the wind
        # path too, not just the (linear) tracer path.
        dd = divergence_cell_3d(u_ * dp_edge, mesh)
        cc = jnp.cumsum(dd, axis=-1)
        ff = p_s[:, None] * compute_sigma_dot_from_cumsum(
            cc, cc[..., -1:], p_s, coord)
        return tracer_flux_form_tendency(
            q_, u_, dp_edge, dp_cell, dd, ff, mesh)

    jax.test_util.check_grads(f, (q, u), order=2, modes=("fwd", "rev"),
                              atol=2e-4, rtol=2e-4)


# =========================================================================
# 7. Kernel-level unit checks (boundaries, direction, single level)
# =========================================================================

def test_kernel_zero_wind_zero_flux_gives_zero_tendency():
    """No mass flux at all ⇒ no transport, for ANY tracer field."""
    mesh, coord = _mesh(), _sigma()
    st = _state(mesh, coord, seed=8)
    p_s = st.p_s.data
    dp_cell = coord.layer_thickness_dp(p_s)
    q = st.tracers["q_v"].data[..., None]
    n_e = mesh.nEdges
    out = tracer_flux_form_tendency(
        q,
        jnp.zeros((n_e, NLEV), _F64),
        jnp.ones((n_e, NLEV), _F64),
        dp_cell,
        jnp.zeros((mesh.nCells, NLEV), _F64),
        jnp.zeros((mesh.nCells, NLEV + 1), _F64),
        mesh,
    )
    assert float(jnp.max(jnp.abs(out))) == 0.0


def test_kernel_single_level_column():
    """nlev == 1: there are no interior interfaces; the vertical term must
    degenerate cleanly (no shape error, F=0 at both ends)."""
    from legoesm.core.operators_voronoi import divergence_cell_3d

    mesh = _mesh()
    n_c, n_e = mesh.nCells, mesh.nEdges
    q = jnp.full((n_c, 1, 1), 3.0e-3, _F64)
    u = jnp.asarray(np.random.default_rng(0).standard_normal((n_e, 1)), _F64)
    dp_e = jnp.full((n_e, 1), 1.0e5, _F64)
    dp_c = jnp.full((n_c, 1), 1.0e5, _F64)
    div_dp = divergence_cell_3d(u * dp_e, mesh)
    out = tracer_flux_form_tendency(
        q, u, dp_e, dp_c, div_dp, jnp.zeros((n_c, 2), _F64), mesh)
    assert out.shape == (n_c, 1, 1)
    # Constant q, F ≡ 0 ⇒ exactly free-stream.
    assert float(jnp.max(jnp.abs(out))) / 3.0e-3 < 1e-14


def test_kernel_vertical_only_conserves_and_points_downward():
    """Isolate the VERTICAL term (zero horizontal flux).

    (a) CONSERVATION: with ΔF = 0 in the interior, d(δp)/dt = 0, so the column
        sum of ``δp·dq/dt`` must telescope to exactly zero — that is what
        ``F_0 = F_nlev = 0`` buys.
    (b) DIRECTION (the sign-convention gate): a flipped upwind side or a
        shifted interface index still CONSERVES, so conservation alone cannot
        certify the sign.  ``F > 0`` is DOWNWARD (increasing σ / increasing k),
        so a tracer spike at level 2 must deplete level 2 and fill level 3.
    """
    mesh = _mesh()
    n_c, n_e = mesh.nCells, mesh.nEdges
    dp_c = jnp.full((n_c, NLEV), 1.0e5 / NLEV, _F64)
    q = jnp.zeros((n_c, NLEV, 1), _F64).at[:, 2, 0].set(1.0e-2)
    F = jnp.zeros((n_c, NLEV + 1), _F64).at[:, 1:-1].set(50.0)
    out = tracer_flux_form_tendency(
        q, jnp.zeros((n_e, NLEV), _F64), jnp.ones((n_e, NLEV), _F64),
        dp_c, jnp.zeros((n_c, NLEV), _F64), F, mesh)

    col = jnp.sum(dp_c * out[..., 0], axis=-1)
    assert float(jnp.max(jnp.abs(col))) < 1e-12 * 1.0e-2 * 1.0e5

    assert float(out[0, 2, 0]) < 0.0, "downward flux must deplete the source"
    assert float(out[0, 3, 0]) > 0.0, "downward flux must fill the level BELOW"
    assert float(jnp.max(jnp.abs(out[:, :2, 0]))) == 0.0, (
        "nothing above the source may change under a purely downward flux")


def test_grad_flows_through_the_mixed_flux_advective_scatter():
    """With BOTH per-mass and per-volume tracers present the tendency is
    assembled by a ``.at[..., idx].set(...)`` scatter of two separately
    computed groups.  A cotangent must reach BOTH groups — a scatter that
    dropped one would silently produce a zero gradient for those tracers."""
    mesh, coord = _mesh(), _sigma()
    st = _state(mesh, coord, seed=2)
    qv = st.tracers["q_v"].data
    names = ("q_v", "N_c")
    st = st._replace(tracers={
        "q_v": st.tracers["q_v"],
        "N_c": st.tracers["q_v"].replace(data=1.0e8 * qv, name="N_c"),
    })
    cfg = _cfg(moisture_flux_form=True)

    def loss(qs):
        s = st._replace(tracers={
            k: st.tracers[k].replace(data=qs[i]) for i, k in enumerate(names)})
        t = mpas_hydrostatic_tendencies(s, mesh, coord, cfg).tracer_tendencies
        return jnp.sum(t["q_v"].data ** 2) + jnp.sum(t["N_c"].data ** 2)

    gs = jax.grad(loss)([st.tracers[k].data for k in names])
    for k, g in zip(names, gs):
        assert jnp.all(jnp.isfinite(g)), f"non-finite gradient for {k}"
        assert float(jnp.max(jnp.abs(g))) > 0.0, f"zero gradient for {k}"


@pytest.mark.parametrize("cname,cbuild", _COORDS)
def test_ps_hyperdiffusion_degrades_conservation_only_linearly(cname, cbuild):
    """Pin the ONE known consistency limit, so it cannot silently grow.

    ``d(δp)/dt`` inside the operator is the ADIABATIC continuity closure (what
    ``σ̇``/``F`` are diagnosed from).  ``nu_del4_ps`` then adds a
    surface-pressure hyperdiffusion that moves mass with no matching tracer
    flux, so the model's true ``d(δp)/dt`` exceeds the closure by that term.

    Asserted: (a) FREE-STREAM is untouched by ν — the cancellation is internal
    to the operator; (b) the conservation residual is LINEAR in ν (10x ν ⇒ 10x
    residual, which is what an omitted linear term must do — a superlinear
    growth would mean something else is wrong); (c) even at the production
    ``nu_del4_ps`` the residual stays far below the advective operator's.
    """
    mesh, coord = _mesh(), cbuild()
    st = _state(mesh, coord, seed=3)
    q = st.tracers["q_v"].data
    dp = _layer_mass(st, coord)
    area = mesh.areaCell.astype(_F64)[:, None]
    st_c = _state(mesh, coord, q_const=1.0e-2)

    res, gross = {}, {}
    for nu in (0.0, 1.0e15, 1.0e16):
        cfg = _cfg(moisture_flux_form=True, nu_del4_ps=nu)
        t = mpas_hydrostatic_tendencies(st, mesh, coord, cfg)
        dq = t.tracer_tendencies["q_v"].data
        res[nu] = abs(float(jnp.sum(
            area * (dp * dq + q * _d_dp_dt(coord, t.dp_s_dt.data)))))
        gross[nu] = float(jnp.sum(jnp.abs(area * dp * dq)))
        # (a) free-stream is ν-independent
        fs = float(jnp.max(jnp.abs(mpas_hydrostatic_tendencies(
            st_c, mesh, coord, cfg).tracer_tendencies["q_v"].data))) / 1.0e-2
        assert fs < 1e-14, f"[{cname}] nu={nu:.0e}: free-stream broke ({fs:.3e})"
        print(f"[{cname}] nu_del4_ps={nu:8.1e} residual/gross="
              f"{res[nu] / gross[nu]:.3e} free-stream={fs:.3e}/s")

    # (b) linear in nu (allow a factor-2 band around the 10x expectation)
    ratio = res[1.0e16] / res[1.0e15]
    assert 5.0 < ratio < 20.0, (
        f"[{cname}] residual scales as {ratio:.2f}x for a 10x nu — not the "
        "linear behaviour an omitted linear closure term must show")

    # (c) still vastly better than advective at the production setting
    t_adv = mpas_hydrostatic_tendencies(
        st, mesh, coord, _cfg(moisture_flux_form=False, nu_del4_ps=1.0e15))
    dq_adv = t_adv.tracer_tendencies["q_v"].data
    res_adv = abs(float(jnp.sum(
        area * (dp * dq_adv + q * _d_dp_dt(coord, t_adv.dp_s_dt.data)))))
    assert res[1.0e15] < 1e-6 * res_adv, (
        f"[{cname}] with p_s hyperdiffusion on, flux form ({res[1.0e15]:.3e}) "
        f"is no longer decisively better than advective ({res_adv:.3e})")
