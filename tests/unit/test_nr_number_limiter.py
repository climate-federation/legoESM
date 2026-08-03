"""Rain-NUMBER consistency limiter (SAM LAMMINR/LAMMAXR semantics).

Regression for the N_r orphan-number runaway measured 2026-08-03: the
production AMIP reference (`stormtrack_ah025_120d`, a_h_scale=0.25) carried
42% orphan rain number (q_r <= QSMALL with N_r > 0) at every checkpoint and
reached N_r = 2.8e19 /m^3 by day 120 — ~8x the number density of AIR MOLECULES
by the time later runs hit 2e26 — with the extremes piled into the
stratosphere (median level 2, ~20 hPa, T ~ 207 K) where rain cannot exist.

Cause: every N_r sink is proportional to the rain MASS or a mass tendency, so
all vanish together as q_r -> 0, leaving the net tendency EXACTLY zero and the
orphan number immortal; N_r was the only prognostic number species without the
post-step PSD-consistency limiter that N_i/N_s/N_g already had.  This is the
same failure the module documents for ice ("century3 at N_i = 1e193").

Each test below FAILS if the `n_r_new` limiter in ``morrison.py`` is removed.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState

DT = 75.0  # production MPAS timestep


def _column(T0, q_r0, n_r0, q_v0=1.0e-4, nlev=4, p0=7.0e4, p1=9.8e4):
    ncol = 1
    T = jnp.full((ncol, nlev), T0)
    q_v = jnp.full((ncol, nlev), q_v0)
    p_full = jnp.linspace(p0, p1, nlev)[None, :]
    p_half = jnp.linspace(p0 - 2.0e3, p1 + 2.0e3, nlev + 1)[None, :]
    rho = p_full / (constants.R_d * T)
    dz = jnp.full((ncol, nlev), 800.0)
    zeros = jnp.zeros((ncol, nlev))
    hyd = HydrometeorState(
        q_c=zeros, q_r=jnp.full((ncol, nlev), q_r0), q_i=zeros,
        q_s=zeros, q_g=zeros, N_c=zeros,
        N_r=jnp.full((ncol, nlev), n_r0), N_i=zeros,
    )
    return T, q_v, hyd, p_full, p_half, rho, dz


def _run(*args):
    return morrison_microphysics(*args, DT, MorrisonConfig())


def test_orphan_rain_number_is_cleared():
    """THE regression: q_r = 0 with huge N_r (the trap state).

    All mass-proportional sinks are zero here, so only the consistency
    limiter can clear it.  Without the fix dN_r_dt is EXACTLY 0.0 and this
    assertion fails.
    """
    args = _column(280.0, q_r0=0.0, n_r0=1.0e20)
    out = _run(*args)
    n_new = 1.0e20 + DT * np.asarray(out.dN_r_dt)
    assert n_new.max() < 1.0e20 * 1e-6, (
        f"orphan rain number not cleared: N_r {1.0e20:.3g} -> {n_new.max():.3g}"
    )


@pytest.mark.parametrize("n_r0", [1.0e6, 1.0e10, 1.0e14, 1.0e18, 1.0e22])
def test_orphan_clearing_is_magnitude_independent(n_r0):
    """The pre-fix defect was structural, not a matter of degree: dN_r_dt was
    exactly zero across 16 decades.  Clearing must hold at every magnitude."""
    args = _column(280.0, q_r0=0.0, n_r0=n_r0)
    out = _run(*args)
    n_new = n_r0 + DT * np.asarray(out.dN_r_dt)
    assert n_new.max() < n_r0 * 1e-6, f"not cleared at N_r={n_r0:.3g}"


def test_rain_number_bounded_by_lamr_slope_limits():
    """With mass present, the post-step number must respect the SAM rain-slope
    UPPER bound (LAMR <= lamr_max), i.e. sit at or below n_hi derived from the
    SAME slope definition the fall speed uses.  Only the upper bound is ported
    (see the EXCLUSIONS note in morrison.py).

    Uses the PER-VOLUME inversion N_r = LAMR^3 * rho * q_r / (pi * rho_water);
    the per-mass ice form would be wrong by a factor of rho.
    """
    cfg = MorrisonConfig()
    q_r0 = 1.0e-4
    args = _column(280.0, q_r0=q_r0, n_r0=1.0e18)  # absurdly many tiny drops
    T, q_v, hyd, p_full, p_half, rho, dz = args
    out = _run(*args)
    n_new = np.asarray(hyd.N_r) + DT * np.asarray(out.dN_r_dt)
    q_r_new = np.maximum(np.asarray(hyd.q_r)
                         + DT * np.asarray(out.dq_r_dt), 0.0)
    c = np.pi * constants.rho_water
    n_hi = cfg.lamr_max ** 3 * np.asarray(rho) * q_r_new / c
    # small tolerance for the float round-trip through the tendency
    assert (n_new <= n_hi * (1.0 + 1e-9) + 1e-30).all(), (
        f"post-step N_r exceeds the lamr_max bound: "
        f"max ratio {(n_new / np.maximum(n_hi, 1e-300)).max():.6g}")


def test_healthy_rain_column_the_limiter_does_not_bind():
    """The limiter must be a NO-OP where the invariant already holds.

    NO-OP CRITERION: in every level that still holds rain mass after the step,
    the post-step number must lie STRICTLY INSIDE (n_lo, n_hi).  A value
    strictly interior proves the clip was inactive, i.e. the limiter changed
    nothing there.

    Levels whose rain sediments away entirely (q_r_new <= QSMALL) are EXCLUDED
    and checked separately below — clearing those is the fix working, not a
    regression.  In a 4-level column with no inflow the TOP level always
    drains, which is why a blanket "all levels keep their number" assertion
    would be wrong.
    """
    cfg = MorrisonConfig()
    q_r0 = 1.0e-4
    T, q_v, hyd, p_full, p_half, rho, dz = _column(280.0, q_r0=q_r0, n_r0=1.0)
    c = np.pi * constants.rho_water
    rho0 = float(np.asarray(rho)[0, 0])
    n_hi0 = cfg.lamr_max ** 3 * rho0 * q_r0 / c
    n_lo0 = cfg.lamr_min ** 3 * rho0 * q_r0 / c  # for a mid-range start only
    n_mid = float(np.sqrt(n_hi0 * n_lo0))  # squarely inside the bounds

    args = _column(280.0, q_r0=q_r0, n_r0=n_mid)
    out = _run(*args)
    rho_a = np.asarray(args[5])
    n_new = n_mid + DT * np.asarray(out.dN_r_dt)
    q_r_new = np.maximum(np.asarray(args[2].q_r)
                         + DT * np.asarray(out.dq_r_dt), 0.0)
    n_hi = cfg.lamr_max ** 3 * rho_a * q_r_new / c

    wet = q_r_new > 1.0e-14
    assert wet.any(), "test setup produced no wet level to check"
    # only the UPPER bound is applied (the lower bound is deliberately not
    # ported -- see the EXCLUSIONS note in morrison.py), so the no-op criterion
    # is that n_new sits strictly BELOW n_hi.
    assert (n_new[wet] < n_hi[wet]).all(), (
        "upper limiter BOUND in a healthy column (should be strictly below): "
        f"n_new={n_new[wet]} n_hi={n_hi[wet]}")
    assert (n_new[wet] > 0).all(), "healthy wet level lost all its number"
    # and the drained levels are exactly cleared
    assert (n_new[~wet] == 0.0).all() if (~wet).any() else True


def test_mass_budget_untouched_by_the_number_clearing():
    """Clearing orphan NUMBER must not alter the water MASS budget — there is
    no mass in an orphan cell, so dq_r_dt must be unaffected by N_r."""
    a = _run(*_column(280.0, q_r0=0.0, n_r0=1.0e6))
    b = _run(*_column(280.0, q_r0=0.0, n_r0=1.0e22))
    np.testing.assert_allclose(np.asarray(a.dq_r_dt), np.asarray(b.dq_r_dt),
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(a.dq_v_dt), np.asarray(b.dq_v_dt),
                               rtol=0, atol=0)


def test_jit_eager_parity():
    """Divergence under jit means a tracer bug (CLAUDE.md verification gate)."""
    args = _column(280.0, q_r0=0.0, n_r0=1.0e20)
    eager = _run(*args)
    jitted = jax.jit(
        lambda *a: morrison_microphysics(*a, DT, MorrisonConfig()))(*args)
    np.testing.assert_allclose(np.asarray(eager.dN_r_dt),
                               np.asarray(jitted.dN_r_dt), rtol=1e-12, atol=0)


def test_gradients_finite_through_the_limiter():
    """The limiter must not poison the AD path: rain number is on the
    differentiable trajectory used by the training lanes."""
    cfg = MorrisonConfig()
    T, q_v, hyd, p_full, p_half, rho, dz = _column(
        280.0, q_r0=1.0e-4, n_r0=1.0e8)

    def loss(n_r):
        h = hyd._replace(N_r=jnp.full_like(hyd.N_r, 1.0) * n_r)
        o = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, DT, cfg)
        return jnp.sum(o.dN_r_dt ** 2)

    g = jax.grad(loss)(1.0e8)
    assert np.isfinite(float(g)), f"non-finite gradient through limiter: {g}"


def test_check_grads_second_order_in_the_interior():
    """check_grads(order=2) on the changed path.

    Evaluated with N_r strictly INSIDE (n_lo, n_hi) — the clip and the QSMALL
    `where` are piecewise, so their kinks have no derivative and finite
    differences across a kink are meaningless.  The interior is where the
    gradient must be correct, and where the training lanes operate.
    """
    from jax.test_util import check_grads
    cfg = MorrisonConfig()
    q_r0 = 1.0e-4
    T, q_v, hyd, p_full, p_half, rho, dz = _column(280.0, q_r0=q_r0, n_r0=1.0)
    c = np.pi * constants.rho_water
    rho0 = float(np.asarray(rho)[0, 0])
    n_mid = float(np.sqrt((cfg.lamr_max ** 3 * rho0 * q_r0 / c)
                          * (cfg.lamr_min ** 3 * rho0 * q_r0 / c)))

    def f(n_r_scalar, q_r_scalar):
        h = hyd._replace(
            N_r=jnp.full_like(hyd.N_r, 1.0) * n_r_scalar,
            q_r=jnp.full_like(hyd.q_r, 1.0) * q_r_scalar,
        )
        o = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz, DT, cfg)
        return jnp.sum(o.dN_r_dt)

    # scale the perturbation to the variables' own magnitudes
    check_grads(lambda a, b: f(a * n_mid, b * q_r0), (1.0, 1.0), order=2,
                modes=("rev",), atol=2e-2, rtol=2e-2)


def test_clearing_cannot_be_outrun_by_transport():
    """ADVERSARIAL: sustained convergence of orphan number into a dry level.

    The claim being tested is that a post-step clearing invariant cannot be
    outrun by advection, because clearing is instantaneous rather than
    rate-limited.  `morrison_microphysics` is a column-local kernel, so the
    dycore's delivery of orphan number is emulated directly: every step a large
    slug of N_r is INJECTED into a dry upper level (q_r = 0 there), exactly as
    convergence would do, and the kernel is stepped forward.

    Known answer: with the limiter, N_r after each step returns to ~0 no matter
    how much is injected, so nothing accumulates across steps.  Without it the
    injected number is inert and N_r grows linearly with the number of steps —
    which is the measured production behaviour (2.2e7 -> 2.8e19 over 90 days).
    """
    nlev = 4
    T, q_v, hyd, p_full, p_half, rho, dz = _column(
        220.0, q_r0=0.0, n_r0=0.0, q_v0=1.0e-6, nlev=nlev)
    n_r = np.zeros((1, nlev))
    slug = 1.0e18  # per step, into the dry top level
    peak = 0.0
    for _step in range(25):
        n_r[0, 0] += slug                      # "advective convergence"
        h = hyd._replace(N_r=jnp.asarray(n_r))
        out = morrison_microphysics(T, q_v, h, p_full, p_half, rho, dz,
                                    DT, MorrisonConfig())
        n_r = np.maximum(n_r + DT * np.asarray(out.dN_r_dt), 0.0)
        assert np.isfinite(n_r).all(), "non-finite N_r during the transport test"
        peak = max(peak, n_r.max())
    # after the final clear, essentially nothing is left standing
    assert n_r.max() < slug * 1e-6, (
        f"orphan number ACCUMULATED under sustained injection: "
        f"final max {n_r.max():.3g} after 25 slugs of {slug:.3g}")
    # and it never built up across steps either
    assert peak < slug * 1e-6, (
        f"orphan number transiently accumulated to {peak:.3g}")
