"""#930: MPAS single-column 2Δσ vertical-checkerboard cure + loud floor.

Root cause (proven by ``test_2dz_is_undamped_null_mode_without_filter``): the
MPAS hydrostatic thermodynamic equation has NO vertical damping of the
grid-scale 2Δσ (Nyquist) vertical mode in T.  Vertical advection is upwind but
vanishes where the vertical mass flux is weak; ω is smoothed (its 2Δσ part is
killed by the half→full average).  So a 2Δσ mode is a *null mode* of the
resolved T-tendency — a resting column with a ``(-1)^k`` T perturbation
produces EXACTLY zero tendency.  The adiabatic term κ·T·ω/p then amplifies it
in subsidence (1/p explodes at the low-pressure top levels) with nothing to
oppose it, until it rides the silent ``T_min=50 K`` floor and rectifies into a
checkerboard (#915 autopsy: even levels pinned at 50 K, odd exploding to 8e8 K).

Cure: ``nu_vert4_T`` — a scale-selective vertical biharmonic (∂⁴/∂σ⁴)
hyperdiffusion of T (the vertical analogue of the horizontal ``nu_del4``).  It
removes the 2Δσ null mode at its source while leaving resolved vertical
structure ~untouched and conserving column-integrated T.  With the cure on, the
``T_min=50 K`` floor never binds (asserted below); it stays only as a last-
resort guard, no longer the silent masker — #915's daily ``T < 100`` bounds
guard aborts on any activation.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.grids.vertical import create_sigma_coordinate
from legoesm.core.state import MPASHydrostaticState
from legoesm.core.field import Field
from legoesm.atmosphere.dynamics.primitive_eq_mpas import (
    MPASPrimitiveEquationModel,
    MPASPrimitiveEquationConfig,
    mpas_hydrostatic_tendencies,
    vertical_del4_T_tendency,
)

pytestmark = pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="dycore science test needs float64 (run with JAX_ENABLE_X64=1)",
)


NLEV = 24


def _mesh():
    return create_voronoi_mesh(2, lloyd_iterations=5)


def _resting_state(mesh, nlev, seed_2dz=0.0, T0=250.0):
    """Resting (u=0) isothermal column + optional 2Δσ T perturbation."""
    k = np.arange(nlev)
    Tcol = T0 + seed_2dz * (-1.0) ** k
    nC, nE = mesh.nCells, mesh.nEdges
    return MPASHydrostaticState(
        u=Field(data=jnp.zeros((nE, nlev)), name="u",
                dims=("nEdges", "nlev"), units="m/s"),
        T=Field(data=jnp.asarray(np.tile(Tcol, (nC, 1))), name="T",
                dims=("nCells", "nlev"), units="K"),
        p_s=Field(data=jnp.full((nC,), 1.0e5), name="p_s",
                  dims=("nCells",), units="Pa"),
        phis=Field(data=jnp.zeros((nC,)), name="phis",
                   dims=("nCells",), units="m^2/s^2"),
    )


def _amp_2dz(T_col):
    k = np.arange(T_col.shape[-1])
    return float(np.abs((np.asarray(T_col) * (-1.0) ** k).mean()))


# ---------------------------------------------------------------------------
# Root cause: the 2Δσ mode is an undamped null mode without the filter.
# ---------------------------------------------------------------------------

def test_2dz_is_undamped_null_mode_without_filter():
    """A resting column with a 2Δσ T perturbation has ZERO tendency (the mode
    is a null mode of the resolved T-tendency); the filter damps it."""
    mesh = _mesh()
    coord = create_sigma_coordinate(NLEV, sigma_top=1e-3)
    state = _resting_state(mesh, NLEV, seed_2dz=1.0)

    cfg0 = MPASPrimitiveEquationConfig(
        nu_vert4_T=0.0, nu_del2=0.0, nu_del4=0.0, K_h=0.0)
    tend0 = mpas_hydrostatic_tendencies(state, mesh, coord, cfg0, dt=60.0)
    # No vertical damping of the 2Δσ mode: dT/dt is machine-zero.
    assert float(jnp.abs(tend0.dT_dt.data).max()) < 1e-10

    nu = 2.0e-6
    cfg1 = MPASPrimitiveEquationConfig(
        nu_vert4_T=nu, nu_del2=0.0, nu_del4=0.0, K_h=0.0)
    tend1 = mpas_hydrostatic_tendencies(state, mesh, coord, cfg1, dt=60.0)
    dT = tend1.dT_dt.data[0]
    # The filter now produces a nonzero tendency that OPPOSES the 2Δσ mode
    # (negative projection onto (-1)^k), i.e. it damps it.
    k = np.arange(NLEV)
    proj = float((np.asarray(dT) * (-1.0) ** k).mean())
    assert proj < 0.0
    # Interior response is ≈ -16·nu·amplitude (edge levels slightly weaker).
    assert proj == pytest.approx(-16.0 * nu, rel=0.15)


# ---------------------------------------------------------------------------
# Cure property: the vertical del4 filter is grid-scale selective.
# ---------------------------------------------------------------------------

def test_vertical_del4_is_scale_selective():
    nlev = 32
    k = np.arange(nlev)
    nu = 1.0e-3

    T2 = jnp.asarray((-1.0) ** k)                     # 2Δσ (Nyquist)
    t2 = vertical_del4_T_tendency(T2, nu)
    rate_2dz = float(jnp.abs(t2[nlev // 2] / T2[nlev // 2]))
    assert rate_2dz == pytest.approx(16.0 * nu, rel=1e-6)  # interior -16·nu

    T8 = jnp.asarray(np.cos(np.pi * k / 4))           # 8Δσ resolved wave
    t8 = vertical_del4_T_tendency(T8, nu)
    rate_8dz = float(jnp.abs(t8[8] / T8[8]))
    # del4 damps the 2Δσ mode ≥40× faster than an 8Δσ wave.
    assert rate_2dz / rate_8dz > 40.0

    # A smooth (quadratic) profile is essentially untouched.
    sig = np.linspace(0.02, 0.99, nlev)
    Tsm = jnp.asarray(200.0 + 100.0 * sig + 50.0 * sig ** 2)
    tsm = vertical_del4_T_tendency(Tsm, nu)
    smooth_rate = float(jnp.abs(tsm).max() / Tsm.max())
    assert smooth_rate < rate_2dz / 100.0

    # nu == 0 is exactly zero (bit-identical pre-fix path) — even for an
    # arbitrary (random) profile, so nu_vert4_T=0 is a true no-op.
    rng = np.random.RandomState(0)
    Trand = jnp.asarray(rng.randn(nlev) * 40.0 + 250.0)
    assert float(jnp.abs(vertical_del4_T_tendency(Trand, 0.0)).max()) == 0.0


def test_vertical_del4_boundary_damped_and_column_conservative():
    """The top/bottom boundary levels (where the #930 blowup is worst) are
    damped at ½ the interior rate — a no-flux boundary constraint of any
    conservative biharmonic — and the operator conserves column-integrated T
    to machine precision for ANY profile (flux form, not accidental)."""
    nlev = 32
    k = np.arange(nlev)
    nu = 1.0e-3
    T2 = jnp.asarray((-1.0) ** k)
    t2 = vertical_del4_T_tendency(T2, nu)
    # Boundary 2Δσ response is -8·nu (½ the interior -16·nu) — reduced but
    # strong, and NOT in the null space (would be 0 without the reflect inner BC).
    assert float(jnp.abs(t2[0] / T2[0])) == pytest.approx(8.0 * nu, rel=1e-6)
    assert float(jnp.abs(t2[-1] / T2[-1])) == pytest.approx(8.0 * nu, rel=1e-6)

    # Column-integrated-T conservation for arbitrary profiles (even + odd nlev,
    # random amplitudes) — refutes the "only accidental for symmetric fields"
    # concern: the outer edge (no-flux) Laplacian makes Σ_k tendency ≡ 0.
    rng = np.random.RandomState(3)
    for n in (24, 31, 17):
        Tr = jnp.asarray(rng.randn(n) * 40.0 + 250.0)
        col_sum = float(jnp.sum(vertical_del4_T_tendency(Tr, nu)))
        scale = float(jnp.abs(vertical_del4_T_tendency(Tr, nu)).max())
        assert abs(col_sum) < 1e-10 * max(scale, 1.0)


def test_vertical_del4_is_differentiable():
    """AD-safe: the cure helper has a finite gradient (end-to-end grad goal)."""
    k = np.arange(NLEV)
    T = jnp.asarray(250.0 + (-1.0) ** k)
    g = jax.grad(lambda x: jnp.sum(vertical_del4_T_tendency(x, 2e-6) ** 2))(T)
    assert bool(jnp.all(jnp.isfinite(g)))


# ---------------------------------------------------------------------------
# Integration: the real dycore step preserves the 2Δσ mode without the cure
# (undamped null mode) and decays it with the cure — and the floor never fires.
# ---------------------------------------------------------------------------

def test_real_step_damps_2dz_and_floor_stays_dead():
    mesh = _mesh()
    coord = create_sigma_coordinate(NLEV, sigma_top=1e-3)
    nsteps = 50
    seed = 2.0

    def integrate(nu):
        state = _resting_state(mesh, NLEV, seed_2dz=seed)
        model = MPASPrimitiveEquationModel(
            mesh=mesh, sigma_coord=coord,
            config=MPASPrimitiveEquationConfig(
                nu_vert4_T=nu, nu_del2=0.0, nu_del4=0.0, K_h=0.0,
                fix_mass=True),
        )
        a0 = _amp_2dz(state.T.data[0])
        tmin_hist = []
        for _ in range(nsteps):
            state = model.step(state, dt=60.0)
            tmin_hist.append(float(state.T.data.min()))
        return a0, _amp_2dz(state.T.data[0]), min(tmin_hist), state

    a0_off, a_off, tmin_off, _ = integrate(0.0)
    a0_on, a_on, tmin_on, st_on = integrate(2.0e-4)

    # Without the cure: 2Δσ mode is a null mode — preserved (undamped).
    assert a_off / a0_off > 0.99
    # With the cure: strongly damped.
    assert a_on / a0_on < 0.1
    # The T_min=50 K floor never fires in either resting-column run
    # (no clamp activations — Tmin stays far above 50 K).
    assert tmin_off > 100.0 and tmin_on > 100.0
    assert bool(jnp.all(jnp.isfinite(st_on.T.data)))


# ---------------------------------------------------------------------------
# Regression guards on the config defaults (off in the dycore, on in production).
# ---------------------------------------------------------------------------

def test_config_defaults():
    # Base dycore config: off by default => pre-#930 bit-identical.
    assert MPASPrimitiveEquationConfig().nu_vert4_T == 0.0
    # Production (coupled/AMIP) default: the cure is on.
    from legoesm.driver.config import DycoreConfig
    assert DycoreConfig().mpas_nu_vert4_T > 0.0


# ---------------------------------------------------------------------------
# Shapiro-form per-step filter (vert4_T_filter) — the ERA5-IC lane cure.
# The explicit rate form is stability-capped (nu*dt*16 < 1) below the
# physics-forced checkerboard growth at production dt; the filter form is
# dt-independent and unconditionally stable for s in (0, 1].
# ---------------------------------------------------------------------------

def _shapiro_apply(T, s):
    return T + vertical_del4_T_tendency(T, s / 16.0)


def test_vert4_filter_kills_2dz_in_one_step_at_full_strength():
    nlev = 20
    smooth = 250.0 + 30.0 * jnp.linspace(1.0, 0.0, nlev)[None, :]
    cb = 5.0 * ((-1.0) ** jnp.arange(nlev))[None, :]
    T = smooth + cb
    T1 = _shapiro_apply(T, 1.0)
    # Project onto the (-1)^k mode over interior levels, REFERENCED to the
    # smooth profile's own projection (a sloped profile projects to slope/2,
    # not zero — the raw projection or an even/odd mean both pick that up).
    sign = ((-1.0) ** jnp.arange(nlev))[None, :]
    proj_ref = jnp.mean((smooth * sign)[:, 4:-4], axis=1)
    proj_before = jnp.mean((T * sign)[:, 4:-4], axis=1) - proj_ref
    proj_after = jnp.mean((T1 * sign)[:, 4:-4], axis=1) - proj_ref
    assert float(jnp.abs(proj_before).min()) > 4.9      # mode present (5 K)
    # interior 2Δσ response is -16·(s/16) => full removal at s=1 (the smooth
    # interior is untouched, so referencing to `smooth` is exact there)
    assert float(jnp.abs(proj_after).max()) < 1e-6
    # column-integrated T conserved (flux-form outer Laplacian)
    assert float(jnp.abs(jnp.sum(T1) - jnp.sum(T))) < 1e-8


def test_vert4_filter_monotone_stable_at_any_strength():
    # amplification factor per mode is 1 - s*lambda/16 with lambda in [0,16]
    # => |factor| <= 1 for s <= 1: repeated application never grows ANY profile.
    nlev = 20
    key = jax.random.PRNGKey(0)
    T = 250.0 + 25.0 * jax.random.normal(key, (4, nlev))
    for s in (0.25, 0.5, 1.0):
        Tk = T
        prev_var = float(jnp.var(Tk))
        for _ in range(50):
            Tk = _shapiro_apply(Tk, s)
            v = float(jnp.var(Tk))
            assert v <= prev_var + 1e-10   # monotone variance decay
            prev_var = v
        assert bool(jnp.all(jnp.isfinite(Tk)))


def test_vert4_filter_preserves_smooth_profile():
    nlev = 20
    # smooth tropospheric profile: no grid-scale content
    T = 300.0 - 60.0 * jnp.sin(
        jnp.pi * jnp.arange(nlev)[None, :] / (2 * (nlev - 1)))
    T1 = _shapiro_apply(T, 0.5)
    # INTERIOR resolved structure essentially untouched (<0.01 K change);
    # the two BOUNDARY levels see the reflect-pad slope response (the price
    # of keeping the full 2Δσ response at the boundary, where the #930
    # checkerboard is worst) — bounded (<0.5 K/application) and column-
    # conservative, but not zero.  Documented in the config docstring.
    d = jnp.abs(T1 - T)
    assert float(jnp.max(d[:, 2:-2])) < 0.01
    assert float(jnp.max(d)) < 0.5


def test_vert4_filter_zero_is_exact_noop_in_config():
    assert MPASPrimitiveEquationConfig().vert4_T_filter == 0.0
    from legoesm.driver.config import DycoreConfig
    assert DycoreConfig().mpas_vert4_t_filter == 0.0
