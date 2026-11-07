"""Advective bottom boundary layer (Campin-Goosse / NEMO trabbl nn_bbl_adv=2).

Idealized acceptance battery (the physics-contract test):
  * analytic 2-column dense-shelf overflow: transport magnitude matches the
    closed-form ``width*e3_bbl*g*gamma*max(0, drho/rho0)`` with the canonical
    EOS, sign is DOWN-slope, and a light shelf gives exactly zero;
  * exact tracer conservation: sum(area*h*d(pt)/dt) telescopes to 0 (fp tol);
  * tendency direction: the deep bottom cell moves TOWARD the dense shelf
    water (cools/salinifies for a cold-salty shelf);
  * flat bottom -> zero everywhere; land-adjacent faces inactive;
  * host wrapper integrates one Euler step and leaves land/dry cells alone.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import (
    nemo_roquet_alpha_beta,
    wright_eos,
)
from legoesm.ocean.physics.bbl_adv import (
    BBLDiffusiveGeometry,
    apply_bbl_diffusive_tendency,
    apply_bbl_adv_tendency,
    bbl_static_geometry,
    bbl_transports,
    nemo_bbl_diffusive_coefficients,
    nemo_bbl_static_geometry,
)

RHO0 = 1025.0
GAMMA = 20.0


def _two_column_setup(nlev=6, dz=100.0, shelf_levels=2):
    """1x2 horizontal domain: column 0 = shelf (shelf_levels wet),
    column 1 = deep (all nlev wet).  Returns (h_ref, mask, T, S)."""
    h = np.zeros((1, 2, nlev))
    h[0, 0, :shelf_levels] = dz
    h[0, 1, :] = dz
    mask = np.ones((1, 2))
    T = np.full((1, 2, nlev), 10.0)
    S = np.full((1, 2, nlev), 35.0)
    return (jnp.asarray(h), jnp.asarray(mask),
            jnp.asarray(T), jnp.asarray(S))


def test_dense_shelf_transport_matches_closed_form():
    h, mask, T, S = _two_column_setup()
    # cold + salty shelf bottom -> denser than the deep neighbour
    T = T.at[0, 0, 1].set(4.0)
    S = S.at[0, 0, 1].set(36.0)
    geom = bbl_static_geometry(h, mask)
    dy_u = jnp.full((1, 1), 5.0e4)   # 50 km face
    dx_v = jnp.zeros((0, 2))
    utr, vtr = bbl_transports(T, S, geom, dy_u, dx_v,
                              gamma_s=GAMMA, rho_0=RHO0)
    # closed form: the EXACT NEMO eos_rab gating — alpha/beta per column at
    # ITS OWN bottom pressure, averaged across the face
    a_sh, b_sh = nemo_roquet_alpha_beta(
        jnp.asarray(4.0), jnp.asarray(36.0), jnp.asarray(150.0), rho0=RHO0)
    a_dp, b_dp = nemo_roquet_alpha_beta(
        jnp.asarray(10.0), jnp.asarray(35.0), jnp.asarray(550.0), rho0=RHO0)
    zgdrho = max(0.0, 0.5 * (a_sh + a_dp) * (10.0 - 4.0)
                 - 0.5 * (b_sh + b_dp) * (35.0 - 36.0))
    expect = 5.0e4 * 100.0 * constants.g * GAMMA * zgdrho * 1.0
    assert zgdrho > 0.0
    assert float(utr[0, 0]) == pytest.approx(expect, rel=1e-10)
    # sign: deeper column is i=1 -> mgrhu=+1 -> transport positive (down-slope)
    assert float(geom.mgrhu[0, 0]) == 1.0
    assert float(utr[0, 0]) > 0.0


def test_thermobaric_gating_uses_own_pressure_rab():
    """Steep-contrast regression (codex HIGH): the alpha/beta-at-own-pressure
    NEMO form differs measurably from a naive direct-density difference at
    the face-MEAN pressure — the implementation must follow the former."""
    nlev = 12
    h = np.zeros((1, 2, nlev))
    h[0, 0, :1] = 150.0                   # shelf: single 150 m cell
    h[0, 1, :] = 350.0                    # deep: 4200 m
    mask = jnp.ones((1, 2))
    T = jnp.asarray(np.full((1, 2, nlev), 2.0))
    S = jnp.asarray(np.full((1, 2, nlev), 34.7))
    T = T.at[0, 0, 0].set(1.0)            # slightly colder shelf
    S = S.at[0, 0, 0].set(34.75)          # slightly saltier shelf
    geom = bbl_static_geometry(jnp.asarray(h), mask)
    utr, _ = bbl_transports(T, S, geom, jnp.full((1, 1), 5.0e4),
                            jnp.zeros((0, 2)), gamma_s=GAMMA, rho_0=RHO0)
    # NEMO-form expectation
    dep_sh, dep_dp = 75.0, float(np.sum(h[0, 1]) - 0.5 * 350.0)
    a_sh, b_sh = nemo_roquet_alpha_beta(
        jnp.asarray(1.0), jnp.asarray(34.75), jnp.asarray(dep_sh), rho0=RHO0)
    a_dp, b_dp = nemo_roquet_alpha_beta(
        jnp.asarray(2.0), jnp.asarray(34.7), jnp.asarray(dep_dp), rho0=RHO0)
    a_bar = 0.5 * (float(a_sh) + float(a_dp))
    b_bar = 0.5 * (float(b_sh) + float(b_dp))
    zg_nemo = max(0.0, a_bar * (2.0 - 1.0) - b_bar * (34.7 - 34.75))
    expect = 5.0e4 * 150.0 * constants.g * GAMMA * zg_nemo
    assert float(utr[0, 0]) == pytest.approx(expect, rel=1e-10)
    # and the naive mean-pressure direct-density form is measurably DIFFERENT
    p_mean = RHO0 * constants.g * 0.5 * (dep_sh + dep_dp)
    drho_naive = float(wright_eos(jnp.asarray(1.0), jnp.asarray(34.75), jnp.asarray(p_mean))
                       - wright_eos(jnp.asarray(2.0), jnp.asarray(34.7), jnp.asarray(p_mean)))
    zg_naive = max(0.0, drho_naive / RHO0)
    assert abs(zg_naive - zg_nemo) > 1e-3 * max(zg_nemo, 1e-12)


def test_light_shelf_gives_zero_transport():
    h, mask, T, S = _two_column_setup()
    T = T.at[0, 0, 1].set(20.0)                 # WARM shelf bottom -> lighter
    geom = bbl_static_geometry(h, mask)
    utr, _ = bbl_transports(T, S, geom, jnp.full((1, 1), 5.0e4),
                            jnp.zeros((0, 2)), gamma_s=GAMMA, rho_0=RHO0)
    assert float(utr[0, 0]) == 0.0


def test_flat_bottom_inactive():
    nlev = 5
    h = jnp.asarray(np.full((2, 3, nlev), 80.0))
    mask = jnp.ones((2, 3))
    geom = bbl_static_geometry(h, mask)
    assert float(jnp.sum(geom.u_active)) == 0.0
    assert float(jnp.sum(geom.v_active)) == 0.0


def test_land_adjacent_faces_inactive():
    h, mask, T, S = _two_column_setup()
    mask = mask.at[0, 0].set(0.0)
    geom = bbl_static_geometry(h, mask)
    assert float(jnp.sum(geom.u_active)) == 0.0


def test_nemo_reference_slope_mask_ignores_partial_centroid_with_same_bottom_level():
    """trabbl.F90:517-533 keys mgrhu to gdept_0(mbkt), not bathymetry.

    Two adjacent partial cells can have different thickness/centroid while
    sharing one bottom index.  The legacy continuous-depth builder activates
    that face; NEMO's reference-depth builder must leave it exactly inactive.
    This is the planted synthetic violation behind the OVERFLOW census arm.
    """
    h = np.zeros((1, 2, 4), dtype=np.float64)
    h[0, 0, :3] = (20.0, 20.0, 5.0)
    h[0, 1, :3] = (20.0, 20.0, 15.0)
    mask = np.ones((1, 2), dtype=np.float64)
    gdept = np.asarray([10.0, 30.0, 50.0, 70.0])
    e3u = np.full((1, 1, 4), 20.0)
    e3v = np.empty((0, 2, 4))

    legacy = bbl_static_geometry(jnp.asarray(h), jnp.asarray(mask))
    nemo = nemo_bbl_static_geometry(
        jnp.asarray(h), jnp.asarray(mask), jnp.asarray(gdept),
        jnp.asarray(e3u), jnp.asarray(e3v))

    assert float(legacy.u_active[0, 0]) == 1.0
    assert float(nemo.mgrhu[0, 0]) == 0.0
    assert float(nemo.u_active[0, 0]) == 0.0


def test_nemo_reference_bbl_thickness_gathers_unmasked_face_metric():
    """The deeper bottom index may be dry in the shelf T column.

    trabbl.F90:529-531 nevertheless gathers the unmasked e3u_0 at both
    bottom indices.  A min of masked T-cell thicknesses would return zero (or
    the wrong partial thickness) and is therefore not an equivalent operand.
    """
    h = np.zeros((1, 2, 4), dtype=np.float64)
    h[0, 0, :2] = (20.0, 7.0)
    h[0, 1, :3] = (20.0, 20.0, 13.0)
    mask = np.ones((1, 2), dtype=np.float64)
    gdept = np.asarray([10.0, 30.0, 50.0, 70.0])
    # Exact unmasked U-face metric at the shelf and deep bottom indices.
    e3u = np.asarray([[[20.0, 7.0, 7.0, 20.0]]])
    e3v = np.empty((0, 2, 4))
    geom = nemo_bbl_static_geometry(
        jnp.asarray(h), jnp.asarray(mask), jnp.asarray(gdept),
        jnp.asarray(e3u), jnp.asarray(e3v))

    assert int(geom.ku_s[0, 0]) == 1
    assert int(geom.ku_d[0, 0]) == 2
    assert float(geom.dep_bot[0, 0]) == 30.0
    assert float(geom.dep_bot[0, 1]) == 50.0
    assert float(geom.e3u_bbl[0, 0]) == 7.0
    assert float(geom.u_active[0, 0]) == 1.0


def test_diffusive_bbl_coefficient_selector_uses_requested_nemo_eos():
    """``bbl`` must use the card's EOS80/TEOS10 ``eos_rab`` arm.

    ORCA2 selects EOS80; retaining the helper's historical TEOS10 default
    changes the density gate on real bottom faces.  This synthetic fixture
    makes the selector observable without relying on an external deck.
    """
    from types import SimpleNamespace

    shape = (1, 2)
    geom = BBLDiffusiveGeometry(
        bot_k=jnp.zeros(shape, dtype=jnp.int32),
        dep_bot_ref=jnp.full(shape, 3000.0),
        mgrhu=jnp.ones(shape, dtype=jnp.int32),
        mgrhv=jnp.ones(shape, dtype=jnp.int32),
        ahu_bbl_0=jnp.ones(shape),
        ahv_bbl_0=jnp.ones(shape),
        t_active=jnp.ones(shape, dtype=bool),
        u_active=jnp.ones(shape, dtype=bool),
        v_active=jnp.ones(shape, dtype=bool),
    )
    # Near the gate boundary, the two source coefficient families select
    # opposite directions on this face.  These values are deliberately not
    # rounded so a replacement with a generic density comparison also fires.
    T = jnp.asarray([[[14.73718664657434], [13.687022703777936]]])
    S = jnp.asarray([[[39.99958086478045], [39.613727777084314]]])
    grid = SimpleNamespace(fold=None)
    eos80 = nemo_bbl_diffusive_coefficients(
        T, S, geom, bottom_depth_m=geom.dep_bot_ref, rho_0=1026.0,
        grid=grid, eos_form="eos80")
    teos10 = nemo_bbl_diffusive_coefficients(
        T, S, geom, bottom_depth_m=geom.dep_bot_ref, rho_0=1026.0,
        grid=grid, eos_form="teos10")
    assert any(not np.array_equal(np.asarray(a), np.asarray(b))
               for a, b in zip(eos80, teos10))


def test_diffusive_bbl_rhs_is_jittable_and_differentiable():
    """The source-literal bottom RHS remains production-JIT/grad safe."""
    from types import SimpleNamespace

    ny, nx, nk = 2, 3, 2
    shape2 = (ny, nx)
    geom = BBLDiffusiveGeometry(
        bot_k=jnp.ones(shape2, dtype=jnp.int32),
        dep_bot_ref=jnp.full(shape2, 150.0),
        mgrhu=jnp.zeros(shape2, dtype=jnp.int32),
        mgrhv=jnp.zeros(shape2, dtype=jnp.int32),
        ahu_bbl_0=jnp.ones(shape2),
        ahv_bbl_0=jnp.ones(shape2),
        t_active=jnp.ones(shape2, dtype=bool),
        u_active=jnp.ones(shape2, dtype=bool),
        v_active=jnp.ones(shape2, dtype=bool),
    )
    tracer = jnp.arange(ny * nx * nk, dtype=jnp.float64).reshape(ny, nx, nk)
    h = jnp.full_like(tracer, 100.0)
    area = jnp.full(shape2, 2.0e8)
    ahu = jnp.asarray([[2.0, 3.0, 0.0], [4.0, 5.0, 0.0]])
    ahv = jnp.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    grid = SimpleNamespace(fold=None)

    def loss(x):
        out, _ = apply_bbl_diffusive_tendency(
            jnp.zeros_like(x), jnp.zeros_like(x), x, x, h, area,
            geom, ahu, ahv, grid=grid)
        return jnp.sum(out * out)

    eager = loss(tracer)
    compiled = jax.jit(loss)(tracer)
    tangent = jax.jit(jax.grad(loss))(tracer)
    assert np.array_equal(np.asarray(eager), np.asarray(compiled))
    assert np.isfinite(np.asarray(tangent)).all()
    assert np.any(np.asarray(tangent) != 0.0)


def test_exact_tracer_conservation_and_direction():
    """Random multi-column domain: the 3-leg cell conserves area*h-weighted
    tracer exactly; the deep bottom cell moves toward the shelf water."""
    rng = np.random.default_rng(11)
    ny, nx, nlev = 4, 6, 7
    # random staircase bathymetry, all wet
    nlev_col = rng.integers(2, nlev + 1, size=(ny, nx))
    h = np.zeros((ny, nx, nlev))
    for j in range(ny):
        for i in range(nx):
            h[j, i, :nlev_col[j, i]] = 90.0
    mask = np.ones((ny, nx))
    T = 8.0 + rng.standard_normal((ny, nx, nlev))
    S = 35.0 + 0.3 * rng.standard_normal((ny, nx, nlev))
    h_j, m_j = jnp.asarray(h), jnp.asarray(mask)
    T_j, S_j = jnp.asarray(T), jnp.asarray(S)
    geom = bbl_static_geometry(h_j, m_j)
    dy_u = jnp.full((ny, nx - 1), 4.0e4)
    dx_v = jnp.full((ny - 1, nx), 4.0e4)
    utr, vtr = bbl_transports(T_j, S_j, geom, dy_u, dx_v,
                              gamma_s=GAMMA, rho_0=RHO0)
    area = jnp.full((ny, nx), 1.6e9)
    dT, dS = apply_bbl_adv_tendency(
        jnp.zeros_like(T_j), jnp.zeros_like(S_j), T_j, S_j, h_j, area,
        geom, utr, vtr, nlev=nlev)
    w = np.asarray(area)[..., None] * np.maximum(h, 1.0e-3)
    # exact conservation (telescoping): compare against the LOCAL tendency
    # magnitude so the tolerance is scale-aware
    tot_T = float(np.sum(np.asarray(dT) * w))
    scale = float(np.sum(np.abs(np.asarray(dT)) * w)) + 1e-30
    assert abs(tot_T) < 1e-9 * scale
    tot_S = float(np.sum(np.asarray(dS) * w))
    scale_S = float(np.sum(np.abs(np.asarray(dS)) * w)) + 1e-30
    assert abs(tot_S) < 1e-9 * scale_S
    # at least one active face moved tracers
    assert scale > 0.0


def test_deep_bottom_moves_toward_shelf_water():
    h, mask, T, S = _two_column_setup()
    T = T.at[0, 0, 1].set(2.0)                   # very cold dense shelf
    S = S.at[0, 0, 1].set(36.5)
    geom = bbl_static_geometry(h, mask)
    dy_u = jnp.full((1, 1), 5.0e4)
    dx_v = jnp.zeros((0, 2))
    utr, vtr = bbl_transports(T, S, geom, dy_u, dx_v,
                              gamma_s=GAMMA, rho_0=RHO0)
    area = jnp.full((1, 2), 1.0e9)
    dT, dS = apply_bbl_adv_tendency(
        jnp.zeros_like(T), jnp.zeros_like(S), T, S, h, area,
        geom, utr, vtr, nlev=6)
    # deep bottom (col 1, k=5) cools + salinifies toward the shelf water
    assert float(dT[0, 1, 5]) < 0.0
    assert float(dS[0, 1, 5]) > 0.0
    # shelf bottom (col 0, k=1) warms (receives deep water at shelf level)
    assert float(dT[0, 0, 1]) > 0.0


def test_host_wrapper_step_and_gating():
    from typing import NamedTuple

    from legoesm.ocean.physics.bbl_adv import apply_bbl_adv_step
    from legoesm.core.field import Field

    class _S(NamedTuple):  # minimal state stand-in (NamedTuple's own _replace)
        T: object
        S: object

    h, mask, T, S = _two_column_setup()
    T = T.at[0, 0, 1].set(4.0)
    geom = bbl_static_geometry(h, mask)
    st = _S(T=Field(T, name="T"), S=Field(S, name="S"))
    out = apply_bbl_adv_step(
        st, geom, dt=600.0, gamma_s=GAMMA, rho_0=RHO0,
        area_2d=jnp.full((1, 2), 1.0e9),
        dy_u_faces=jnp.full((1, 1), 5.0e4),
        dx_v_faces=jnp.zeros((0, 2)), nlev=6)
    dT = np.asarray(out.T.data) - np.asarray(T)
    # exchange happened, bounded (|dT| << shelf-deep contrast), dry cells 0
    assert np.abs(dT).max() > 0.0
    assert np.abs(dT).max() < 6.0                  # << the 6 K contrast
    assert np.all(dT[0, 0, 2:] == 0.0)             # below shelf seafloor


# ---------------------------------------------------------------------------
# S-42 branch-isomorphism collapse: one transcription, no oracle-less clamp
# ---------------------------------------------------------------------------

def _host_step_pieces(area_value):
    """Active-slope fixture + the pieces the host wrapper is built from."""
    h, mask, T, S = _two_column_setup()
    T = T.at[0, 0, 1].set(4.0)                  # cold dense shelf bottom
    geom = bbl_static_geometry(h, mask)
    dy_u = jnp.full((1, 1), 5.0e4)
    dx_v = jnp.zeros((0, 2))
    area = jnp.full((1, 2), area_value)
    utr, vtr = bbl_transports(T, S, geom, dy_u, dx_v,
                              gamma_s=GAMMA, rho_0=RHO0)
    return h, mask, T, S, geom, dy_u, dx_v, area, utr, vtr


def _run_host_step(T, S, geom, dy_u, dx_v, area, dt):
    from typing import NamedTuple

    from legoesm.core.field import Field
    from legoesm.ocean.physics.bbl_adv import apply_bbl_adv_step

    class _S(NamedTuple):
        T: object
        S: object

    st = _S(T=Field(T, name="T"), S=Field(S, name="S"))
    out = apply_bbl_adv_step(
        st, geom, dt=dt, gamma_s=GAMMA, rho_0=RHO0,
        area_2d=area, dy_u_faces=dy_u, dx_v_faces=dx_v, nlev=6)
    return np.asarray(out.T.data), np.asarray(out.S.data)


def _deleted_cap_transports(geom, area, utr, vtr, dt):
    """The 0.25*V_min/dt clamp that used to live inside the host wrapper.

    Reproduced here (and ONLY here) so the deletion has a control: NEMO's
    ``tra_bbl_adv`` (trabbl.F90:243-284) clamps neither ``utr_bbl`` nor
    ``vtr_bbl``, so this arm exists as the thing the model must no longer do.
    """
    e3_bot = jnp.take_along_axis(
        geom.h_ref, geom.bot_k[..., None], axis=-1)[..., 0]
    V_bot = area * jnp.maximum(e3_bot, 1.0e-3)
    cap_u = 0.25 * jnp.minimum(V_bot[:, :-1], V_bot[:, 1:]) / dt
    cap_v = 0.25 * jnp.minimum(V_bot[:-1, :], V_bot[1:, :]) / dt
    return (jnp.sign(utr) * jnp.minimum(jnp.abs(utr), cap_u),
            jnp.sign(vtr) * jnp.minimum(jnp.abs(vtr), cap_v))


def test_host_step_adds_no_arithmetic_of_its_own():
    """The host post-step wrapper IS the shared operator plus one Euler step.

    S-42's two paths must reduce to ONE transcription of
    ``trabbl.F90:243-284``.  This pins the half that is checkable in
    isolation: on a synthetic slope where the BBL is ACTIVE, calling
    ``bbl_transports`` + ``apply_bbl_adv_tendency`` directly and stepping
    ``pt += dt*d(pt)/dt`` reproduces ``apply_bbl_adv_step`` BIT-FOR-BIT, so
    the wrapper contributes no arithmetic of its own once the non-NEMO
    transport cap is gone.

    What this does NOT show, stated so the name cannot be over-read: the
    in-stage site is not bit-identical to the host step in a RUN, because it
    feeds the operator LIVE stage thickness and LIVE bottom depth
    (``ocean_model_latlon_cgrid.py:1290-1298``) where the host wrapper feeds
    the REFERENCE ladder ``geom.h_ref`` and ``geom.dep_bot``.  That is the
    placement difference S-42 still carries; it is not a second
    transcription, and it is not what this test is about.
    """
    dt = 600.0
    h, mask, T, S, geom, dy_u, dx_v, area, utr, vtr = _host_step_pieces(1.0e9)
    assert float(jnp.sum(geom.u_active)) > 0.0        # the slope is active
    assert float(jnp.abs(utr).max()) > 0.0            # and the BBL is running

    # the in-stage operator, called directly, then advanced one Euler step
    dT, dS = apply_bbl_adv_tendency(
        jnp.zeros_like(T), jnp.zeros_like(S), T, S, geom.h_ref, area,
        geom, utr, vtr, nlev=6)
    T_ref = np.asarray(T) + dt * np.asarray(dT)
    S_ref = np.asarray(S) + dt * np.asarray(dS)

    T_host, S_host = _run_host_step(T, S, geom, dy_u, dx_v, area, dt)
    assert np.array_equal(T_host, T_ref)
    assert np.array_equal(S_host, S_ref)

    # ... and again on the PATHOLOGICAL face, where the deleted clamp WOULD
    # have bound.  Without this the assertion above is inert against the cap
    # (at ocean cell volumes the clamp never engages, so a restored clamp
    # leaves it green) and the whole guard rests on one other test.
    _, _, T2, S2, geom2, dy_u2, dx_v2, area2, utr2, vtr2 = _host_step_pieces(
        1.0e6)
    dT2, dS2 = apply_bbl_adv_tendency(
        jnp.zeros_like(T2), jnp.zeros_like(S2), T2, S2, geom2.h_ref, area2,
        geom2, utr2, vtr2, nlev=6)
    T2_host, S2_host = _run_host_step(T2, S2, geom2, dy_u2, dx_v2, area2, dt)
    assert np.array_equal(T2_host, np.asarray(T2) + dt * np.asarray(dT2))
    assert np.array_equal(S2_host, np.asarray(S2) + dt * np.asarray(dS2))


def test_survivor_matches_trabbl_three_leg_formula():
    """The surviving tendency IS ``tra_bbl_adv``'s three legs, term by term.

    trabbl.F90:255-265 (i-direction), with ``zbtr = r1_e1e2t/e3t`` and
    ``zu_bbl = ABS(utr_bbl)``:
      shelf bottom  (iis, ikus): += zu*(pt[iid,ikus] - pt[iis,ikus])*zbtr
      deep interior (iid, jk)  : += zu*(pt[iid,jk+1] - pt[iid,jk])*zbtr
      deep bottom   (iid, ikud): += zu*(pt[iis,ikus] - pt[iid,ikud])*zbtr
    """
    _, _, T, S, geom, dy_u, dx_v, area, utr, vtr = _host_step_pieces(1.0e9)
    dT, _ = apply_bbl_adv_tendency(
        jnp.zeros_like(T), jnp.zeros_like(S), T, S, geom.h_ref, area,
        geom, utr, vtr, nlev=6)
    dT = np.asarray(dT)
    t = np.asarray(T)
    zu = abs(float(utr[0, 0]))
    zbtr = 1.0 / (1.0e9 * 100.0)                    # r1_e1e2t / e3t
    iis, iid, ikus, ikud = 0, 1, 1, 5               # shelf/deep, shelf/deep bottom
    assert dT[0, iis, ikus] == pytest.approx(
        zu * (t[0, iid, ikus] - t[0, iis, ikus]) * zbtr, rel=1e-12)
    for jk in range(ikus, ikud):
        assert dT[0, iid, jk] == pytest.approx(
            zu * (t[0, iid, jk + 1] - t[0, iid, jk]) * zbtr, rel=1e-12)
    assert dT[0, iid, ikud] == pytest.approx(
        zu * (t[0, iis, ikus] - t[0, iid, ikud]) * zbtr, rel=1e-12)

    # NEMO uses zu_bbl = ABS(utr_bbl) and picks shelf/deep off mgrh, so the
    # tendency must be INVARIANT under flipping the slope (which flips both
    # the sign of utr and the shelf/deep assignment).  With a down-slope
    # transport of one sign only, `abs` is the identity and the mgrh<0 branch
    # never runs, so neither is pinned by the rows above.
    h_m = jnp.asarray(np.flip(np.asarray(geom.h_ref), axis=1))
    T_m = jnp.asarray(np.flip(np.asarray(T), axis=1))
    S_m = jnp.asarray(np.flip(np.asarray(S), axis=1))
    geom_m = bbl_static_geometry(h_m, jnp.ones((1, 2)))
    assert float(geom_m.mgrhu[0, 0]) == -float(geom.mgrhu[0, 0]) != 0.0
    utr_m, vtr_m = bbl_transports(T_m, S_m, geom_m, dy_u, dx_v,
                                  gamma_s=GAMMA, rho_0=RHO0)
    assert float(utr_m[0, 0]) == pytest.approx(-float(utr[0, 0]), rel=1e-12)
    dT_m, _ = apply_bbl_adv_tendency(
        jnp.zeros_like(T_m), jnp.zeros_like(S_m), T_m, S_m, h_m, area,
        geom_m, utr_m, vtr_m, nlev=6)
    np.testing.assert_allclose(np.asarray(dT_m), np.flip(dT, axis=1),
                               rtol=1e-12, atol=0.0)


def test_transport_cap_is_gone_and_moved_nothing_at_ocean_scales():
    """Rule 9: the removed clamp has no NEMO arm, and removing it is inert
    where the driver actually runs.

    * at ORCA1-like cell volumes the old clamp never bound, so the survivor
      reproduces the pre-deletion state EXACTLY (the deletion is a no-op for
      every production configuration);
    * on a pathological tiny-area face the old clamp DID bind, and the
      survivor now carries the uncapped NEMO transport — which is what makes
      the first assertion non-vacuous.
    """
    dt = 600.0
    # (a) ORCA1-like: 1e9 m^2 cells -> cap never binds -> deletion inert
    _, _, T, S, geom, dy_u, dx_v, area, utr, vtr = _host_step_pieces(1.0e9)
    utr_c, vtr_c = _deleted_cap_transports(geom, area, utr, vtr, dt)
    assert np.array_equal(np.asarray(utr_c), np.asarray(utr))
    dT_c, dS_c = apply_bbl_adv_tendency(
        jnp.zeros_like(T), jnp.zeros_like(S), T, S, geom.h_ref, area,
        geom, utr_c, vtr_c, nlev=6)
    T_host, S_host = _run_host_step(T, S, geom, dy_u, dx_v, area, dt)
    assert np.array_equal(T_host, np.asarray(T) + dt * np.asarray(dT_c))
    assert np.array_equal(S_host, np.asarray(S) + dt * np.asarray(dS_c))

    # (b) pathological 1e6 m^2 face: the deleted clamp WOULD have bound
    _, _, T2, S2, geom2, dy_u2, dx_v2, area2, utr2, vtr2 = _host_step_pieces(
        1.0e6)
    utr2_c, _ = _deleted_cap_transports(geom2, area2, utr2, vtr2, dt)
    assert float(jnp.abs(utr2_c).max()) < float(jnp.abs(utr2).max())
    dT2_c, _ = apply_bbl_adv_tendency(
        jnp.zeros_like(T2), jnp.zeros_like(S2), T2, S2, geom2.h_ref, area2,
        geom2, utr2_c, jnp.zeros_like(vtr2), nlev=6)
    T2_host, _ = _run_host_step(T2, S2, geom2, dy_u2, dx_v2, area2, dt)
    assert not np.array_equal(
        T2_host, np.asarray(T2) + dt * np.asarray(dT2_c))
