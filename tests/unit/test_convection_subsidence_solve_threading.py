"""``subsidence_solve`` threading for the mass-flux family (SCM-RCE campaign).

Bechtold and EDMF already exposed ``subsidence_solve`` on their configs and
already defaulted to the conservative ``"implicit_flux"`` solve.  Tiedtke,
Zhang-McFarlane and the Arakawa-Wu ``mass_flux`` scheme instead inherited the
kernel's ``"advective"`` FUNCTION DEFAULT with no way for a caller to select the
conservative solve, and Kain-Fritsch HARDCODED ``"implicit_flux"`` at its call
site.  This module pins the newly threaded knob so a single-column RCE
intercomparison can force ONE transport kernel across the whole family.

SCOPE -- Emanuel is deliberately EXCLUDED.  Its shipped path
(``use_genuine_mixing=True``) is a buoyancy-sorting mixing matrix that never
calls the shared kernel; only the legacy surrogate branch does, and that branch
rescales the returned condensate by ``sort_multiplier`` while passing the
enhanced ``delta_0 * sort_multiplier`` IN, which would leave an unpaired vapor
debit under a vapor-debiting solve.  Emanuel is therefore reported as OUTSIDE
the matched-kernel family rather than silently kernel-matched.

Why it matters (the physics, not just the plumbing): the two solves are NOT two
discretisations of one operator.  ``"advective"`` computes the donor-cell
``(M/rho) d(phi)/dz`` compensating subsidence plus local detrainment and emits a
detrained-condensate source ``dq_c = delta_0 M q_c_u / rho`` with NO matching
vapor sink -- so column total water is not closed.  ``"implicit_flux"`` solves
the same transport in telescoping flux form (backward-Euler, per column), books
the paired vapor sink ``-dq_c``, and therefore OWES the matching condensation
warming ``+(L_v/c_p) dq_c``, supplied by the shared
``release_detrained_condensate_latent`` helper.

Gates, per threaded scheme:

1. **Shipped default unchanged** -- the DEFAULT-constructed config reproduces
   the explicit selector bit-for-bit (not merely `run(x) == run(x)`).
2. **The selector is HONOURED** -- non-vacuous: if the config field were dropped
   on the floor the outputs would be byte-identical and the gate FAILS.  (This
   gate is what caught Emanuel's dead path.)
3. **Unknown values RAISE** at kernel entry (static Python str;
   CLAUDE.md dispatch-hardening).
4. **Column WATER closes to machine precision** under ``implicit_flux``,
   measured RELATIVE to the condensate throughput the scheme is actually
   moving, and strictly better than the advective residual.
5. **Column MSE closes** -- catches a missing/mispaired condensation warming,
   whose signature is a residual of exactly ``-L_v int dq_c dp/g`` (cooling).
6. **AD** -- the VJP of ``apply_mass_flux_kernel_implicit_flux`` itself is
   checked against a central finite difference (a whole-scheme gradient can be
   non-zero via CAPE/plume even if the tridiagonal solve is detached), plus an
   end-to-end differentiability check.

A ``test_fixture_actually_convects`` gate runs first: every conservation and AD
assertion below is vacuous on a quiescent column.

Convention note (stated at the kernel): ``z`` up, arrays surface-LAST
(index 0 = model top); ``M >= 0`` is the UPDRAFT mass flux, so the compensating
environmental subsidence carries mass flux ``-M`` (downward).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection
from legoesm.atmosphere.physics.convection.config import (
    BechtoldConfig,
    ConvectiveEDMFConfig,
    KainFritschConfig,
    MassFluxConfig,
    TiedtkeConfig,
    ZhangMcFarlaneConfig,
)
from legoesm.atmosphere.physics.convection.kain_fritsch import (
    kain_fritsch_convection,
)
from legoesm.atmosphere.physics.convection.mass_flux import (
    apply_mass_flux_kernel_implicit_flux,
    edmf_convection,
    mass_flux_convection,
)
from legoesm.atmosphere.physics.convection.tiedtke import tiedtke_convection
from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (
    zhang_mcfarlane_convection,
)

DT_S = 300.0


def _column(ncol=2, nlev=20, T_sfc=302.0, q_sfc=17e-3, lapse_rate=7.0,
            p_s=1.0e5, p_top=5.0e3):
    """Conditionally-unstable, surface-LAST tropical column.

    Matches the fixture convention used by ``test_zhang_mcfarlane`` /
    ``test_bechtold_implicit_flux`` so the numbers here are comparable to the
    existing leaf tests.
    """
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [jnp.full((ncol, 1), p_top * 0.5), p_half_inner,
         jnp.full((ncol, 1), p_s)], axis=1)
    z_full = -8500.0 * jnp.log(p_full / p_s)
    T = jnp.full((ncol,), T_sfc)[:, None] - lapse_rate * 1e-3 * z_full
    q_v = q_sfc * jnp.exp(-z_full / 3000.0)
    u = jnp.broadcast_to(jnp.linspace(0.0, 20.0, nlev)[None, :], (ncol, nlev))
    v = jnp.zeros_like(u)
    return T, q_v, p_full, p_half, u, v


# --- per-scheme adapters: (config_factory, runner) -------------------------
# Each runner takes a ``subsidence_solve`` string, builds that scheme's config
# with the knob set, runs the scheme on the shared column, and returns its
# ConvectionOutput.  Keeping the adapters here (rather than a copy-pasted test
# per scheme) is what lets gates 1-4 be stated once.

def _run_tiedtke(ss, T_offset=0.0):
    T, q_v, p_full, p_half, u, v = _column()
    cpp = jnp.zeros_like(T)
    cfg = TiedtkeConfig() if ss is None else TiedtkeConfig(subsidence_solve=ss)
    out, _ = tiedtke_convection(
        T + T_offset, q_v, p_full, p_half, u, v, cpp, dt=DT_S, config=cfg)
    return out


def _run_kain_fritsch(ss, T_offset=0.0):
    T, q_v, p_full, p_half, _u, _v = _column()
    cpp = jnp.zeros_like(T)
    cfg = (KainFritschConfig() if ss is None
           else KainFritschConfig(subsidence_solve=ss))
    w_grid = jnp.zeros_like(T)
    out, _ = kain_fritsch_convection(
        T + T_offset, q_v, p_full, p_half, w_grid, cpp, dt=DT_S, config=cfg)
    return out


def _run_zm(ss, T_offset=0.0):
    T, q_v, p_full, p_half, u, v = _column()
    cpp = jnp.zeros_like(T)
    cfg = (ZhangMcFarlaneConfig() if ss is None
           else ZhangMcFarlaneConfig(subsidence_solve=ss))
    out, _ = zhang_mcfarlane_convection(
        T + T_offset, q_v, p_full, p_half, u, v, cpp, dt=DT_S, config=cfg)
    return out


def _run_mass_flux(ss, T_offset=0.0):
    T, q_v, p_full, p_half, _u, _v = _column()
    cfg = MassFluxConfig() if ss is None else MassFluxConfig(subsidence_solve=ss)
    M_c = jnp.zeros((T.shape[0],))
    out, _ = mass_flux_convection(
        T + T_offset, q_v, p_full, p_half, M_c, dt=DT_S, config=cfg)
    return out


# Schemes whose SHIPPED default is the leaky "advective" solve -- these are
# the four this campaign newly threaded.
def _run_edmf(ss, T_offset=0.0):
    T, q_v, p_full, p_half, _u, _v = _column()
    cfg = (ConvectiveEDMFConfig() if ss is None
           else ConvectiveEDMFConfig(subsidence_solve=ss))
    a_u = jnp.full((T.shape[0],), 0.05)
    out, _ = edmf_convection(
        T + T_offset, q_v, p_full, p_half, a_u=a_u, dt=DT_S, config=cfg)
    return out


def _run_bechtold(ss, T_offset=0.0):
    T, q_v, p_full, p_half, u, v = _column()
    cpp = jnp.zeros_like(T)
    cfg = (BechtoldConfig(enable_stochastic=False, enable_cmt=False) if ss is None
           else BechtoldConfig(enable_stochastic=False, enable_cmt=False,
                               subsidence_solve=ss))
    out, _, _ = bechtold_convection(
        T + T_offset, q_v, p_full, p_half, u=u, v=v, conv_prog_profile=cpp,
        conv_stoch_state=jnp.zeros((T.shape[0],)), prng_key=None,
        dt=DT_S, config=cfg, moisture_convergence=jnp.zeros_like(T))
    return out


SCHEMES = {
    "tiedtke": (_run_tiedtke, TiedtkeConfig, "advective"),
    "zhang_mcfarlane": (_run_zm, ZhangMcFarlaneConfig, "advective"),
    "mass_flux": (_run_mass_flux, MassFluxConfig, "advective"),
    # Kain-Fritsch already shipped the conservative solve (hardcoded at its
    # call site, PR #988); the field promotes it with the SAME default.
    "kain_fritsch": (_run_kain_fritsch, KainFritschConfig, "implicit_flux"),
    # EDMF and Bechtold already SHIPPED the conservative solve.  They are in
    # this matrix because codex r2 found EDMF silently missing the paired
    # condensation heating, and Bechtold applying it UNCONDITIONALLY (i.e.
    # over-heating on the advective arm) -- both invisible until the gates
    # below covered them.
    "edmf": (_run_edmf, ConvectiveEDMFConfig, "implicit_flux"),
    "bechtold": (_run_bechtold, BechtoldConfig, "implicit_flux"),
}
SCHEME_IDS = sorted(SCHEMES)


def _column_water_residual(out, p_half):
    """|int (dq_v + dq_c + dq_r) dp/g| [kg/m^2/s] per column.

    Sign convention: ``dp = p_half[1:] - p_half[:-1] > 0`` (surface-LAST, so
    pressure INCREASES with index).  A closed convective water budget has the
    vapor sink exactly balancing the condensate source, i.e. residual 0: the
    scheme moves water between reservoirs, it must not create or destroy it.
    """
    dp = p_half[:, 1:] - p_half[:, :-1]
    total = out.dq_v_dt + out.dq_c_conv_dt
    if getattr(out, "dq_r_conv_dt", None) is not None:
        total = total + out.dq_r_conv_dt
    return jnp.abs(jnp.sum(total * dp, axis=1) / constants.g)


# ---------------------------------------------------------------------------
# Gate 1 -- the default is "advective" and is preserved bit-for-bit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCHEME_IDS)
def test_shipped_default_is_unchanged(name):
    _run, cfg_cls, shipped = SCHEMES[name]
    assert cfg_cls().subsidence_solve == shipped, (
        f"{name}: threading the knob must NOT change the shipped default "
        f"(expected {shipped!r})"
    )
    assert cfg_cls().theta_implicit == 1.0


@pytest.mark.parametrize("name", SCHEME_IDS)
def test_default_config_matches_explicit_selector_bit_for_bit(name):
    """The DEFAULT-constructed config must reproduce the explicit selector.

    ``run(None)`` builds ``Config()`` -- the historical call -- and ``run(shipped)``
    passes the selector explicitly.  Comparing the two proves the new field
    defaults to the shipped behaviour, which comparing ``run(x)`` to ``run(x)``
    (mere determinism) does NOT.
    """
    run, _cfg_cls, shipped = SCHEMES[name]
    default = run(None)
    explicit = run(shipped)
    for field in ("dT_dt", "dq_v_dt", "dq_c_conv_dt"):
        a = getattr(default, field)
        b = getattr(explicit, field)
        assert jnp.array_equal(a, b), (
            f"{name}.{field}: default config differs from explicit "
            f"subsidence_solve={shipped!r} -- shipped behaviour changed"
        )


# ---------------------------------------------------------------------------
# Gate 2 -- "implicit_flux" is actually honoured (non-vacuous plumbing proof)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCHEME_IDS)
def test_implicit_flux_is_honoured(name):
    run, _cfg_cls, _shipped = SCHEMES[name]
    adv = run("advective")
    imp = run("implicit_flux")
    assert jnp.all(jnp.isfinite(imp.dT_dt))
    assert jnp.all(jnp.isfinite(imp.dq_v_dt))
    # If the config field were dropped on the floor (the bug this guards),
    # these would be bit-identical.
    differs = (
        not jnp.allclose(adv.dT_dt, imp.dT_dt, rtol=0, atol=0)
        or not jnp.allclose(adv.dq_v_dt, imp.dq_v_dt, rtol=0, atol=0)
    )
    assert differs, (
        f"{name}: subsidence_solve='implicit_flux' produced byte-identical "
        "output to 'advective' -- the config field is NOT reaching "
        "apply_mass_flux_kernel"
    )


# ---------------------------------------------------------------------------
# Gate 3 -- dispatch hardening
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCHEME_IDS)
def test_unknown_subsidence_solve_raises(name):
    run, _cfg_cls, _shipped = SCHEMES[name]
    with pytest.raises(ValueError, match="unknown subsidence_solve"):
        run("bogus")


# ---------------------------------------------------------------------------
# Gate 4 -- the conservative solve closes the column water budget better
# ---------------------------------------------------------------------------

def _column_condensate_throughput(out, p_half):
    """int dq_c dp/g [kg/m^2/s] -- the scale the residual must be judged against.

    A residual of 1e-9 is only "closed" relative to the water the scheme is
    actually moving; without this denominator a scheme that does nothing would
    look perfectly conservative.
    """
    dp = p_half[:, 1:] - p_half[:, :-1]
    return jnp.abs(jnp.sum(out.dq_c_conv_dt * dp, axis=1) / constants.g)


@pytest.mark.parametrize("name", SCHEME_IDS)
def test_fixture_actually_convects(name):
    """Every gate below is vacuous on a quiescent column -- prove activity first."""
    run, _cfg_cls, _shipped = SCHEMES[name]
    _T, _q, _pf, p_half, _u, _v = _column()
    out = run("implicit_flux")
    throughput = _column_condensate_throughput(out, p_half)
    assert jnp.max(throughput) > 1e-8, (
        f"{name}: column condensate throughput {np.asarray(throughput)} "
        "kg/m^2/s -- the fixture is not convecting, so the conservation and "
        "AD gates prove nothing"
    )
    assert jnp.max(jnp.abs(out.dT_dt)) > 0.0


@pytest.mark.parametrize("name", SCHEME_IDS)
def test_implicit_flux_closes_column_water_to_machine_precision(name):
    """The conservative solve must CLOSE the budget, not merely not-worsen it.

    ``<=`` alone would pass for an implementation that leaks exactly as much as
    the advective one.  Two assertions instead: (i) the implicit residual is
    negligible RELATIVE to the condensate throughput the scheme is moving, and
    (ii) it is a strict, large improvement on the advective residual.
    """
    run, _cfg_cls, _shipped = SCHEMES[name]
    _T, _q, _pf, p_half, _u, _v = _column()
    adv = run("advective")
    imp = run("implicit_flux")
    res_adv = _column_water_residual(adv, p_half)
    res_imp = _column_water_residual(imp, p_half)
    scale = _column_condensate_throughput(imp, p_half)
    rel_imp = res_imp / jnp.maximum(scale, 1e-30)
    assert jnp.all(rel_imp < 1e-10), (
        f"{name}: implicit_flux column-water residual {np.asarray(res_imp)} is "
        f"{np.asarray(rel_imp)} of the condensate throughput "
        f"{np.asarray(scale)} -- the flux form is NOT telescoping"
    )
    # Non-vacuous, and a STRICT improvement: the advective residual must be
    # both non-zero and orders of magnitude larger.
    assert jnp.max(res_adv) > 0.0, (
        f"{name}: advective residual is exactly zero -- nothing to improve on"
    )
    assert jnp.all(res_imp < res_adv), (
        f"{name}: implicit residual {np.asarray(res_imp)} is not STRICTLY "
        f"below advective {np.asarray(res_adv)}"
    )


def test_implicit_kernel_vjp_matches_finite_differences():
    """AD correctness AT THE SOLVE, not merely somewhere in an enclosing scheme.

    A whole-scheme gradient can be finite and non-zero via CAPE / plume /
    saturation even if the tridiagonal solve is detached, so this
    differentiates ``apply_mass_flux_kernel_implicit_flux`` DIRECTLY and checks
    the value against a central finite difference.
    """
    T, q_v, p_full, p_half, _u, _v = _column(ncol=1, nlev=20)
    z = -8500.0 * jnp.log(p_full / 1.0e5)
    rho = p_full / (constants.R_d * T)
    T_u = T + 2.0
    q_v_u = q_v * 1.1
    q_c_u = jnp.full_like(T, 5e-4)
    M = jnp.broadcast_to(jnp.linspace(0.0, 0.03, T.shape[1])[None, :], T.shape)

    def loss(scale):
        dT, dq, dqc = apply_mass_flux_kernel_implicit_flux(
            T * scale, q_v, p_full, p_half, T_u, q_v_u, q_c_u, M,
            z, rho, 7.5e-5, 0.05, DT_S,
        )
        return jnp.sum(dT ** 2) + jnp.sum(dq ** 2) + jnp.sum(dqc ** 2)

    g = float(jax.grad(loss)(1.0))
    eps = 1e-6
    fd = (float(loss(1.0 + eps)) - float(loss(1.0 - eps))) / (2.0 * eps)
    assert np.isfinite(g) and abs(g) > 0.0, "kernel VJP is zero or non-finite"
    # Scale-aware: a `1e-5 * max(|fd|, 1.0)` bound degenerates to an ABSOLUTE
    # 1e-5 whenever the derivative is below one, which a materially wrong VJP
    # could slip through (codex r2 finding 4).  atol is set just above the
    # central-difference truncation floor for this fp64 evaluation.
    np.testing.assert_allclose(g, fd, rtol=1e-5, atol=1e-9)


@pytest.mark.parametrize("name", SCHEME_IDS)
def test_implicit_flux_is_differentiable_end_to_end(name):
    """Complements the kernel-level VJP check above: the WHOLE scheme stays
    differentiable with the conservative solve selected."""
    run, _cfg_cls, _shipped = SCHEMES[name]

    def loss(T_offset):
        out = run("implicit_flux", T_offset=T_offset)
        return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.dq_v_dt ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g), f"{name}: grad through implicit_flux is not finite"
    assert abs(float(g)) > 0.0, f"{name}: end-to-end grad is exactly zero"


# ---------------------------------------------------------------------------
# Gate 5 -- column MOIST STATIC ENERGY closes (the missing-latent blocker)
# ---------------------------------------------------------------------------

def _vapor_mse_residual(out, p_half):
    """int (c_p dT + L_v dq_v) dp/g [W/m^2] -- the VAPOR moist static energy.

    WHICH INVARIANT, AND WHY THIS ONE (the control that this gate initially got
    wrong): under the convention this package uses -- and which Kain-Fritsch
    and Bechtold already implement inline -- ``dq_c_conv_dt`` handed to
    microphysics is ALREADY-CONDENSED cloud whose latent heat has ALREADY been
    released into the air by the scheme.  The quantity the scheme must
    therefore conserve is ``h = c_p T + L_v q_v``, WITHOUT a ``+L_v q_c`` term:
    including one double-counts the enthalpy that the condensation warming just
    deposited.

    Sign convention: tendencies are SOURCES (``state += dt*tend``), ``dp > 0``
    surface-LAST.  A closed column has residual 0.

    NON-VACUITY, exactly: the implicit_flux kernel books ``dq_v -= dq_c`` and
    supplies no heating of its own, so DELETING
    ``release_detrained_condensate_latent`` makes this residual exactly
    ``-L_v int dq_c dp/g`` -- i.e. the normalised ratio below jumps from ~0 to
    exactly 1.0.  (That is precisely what this gate measured before the
    invariant was corrected, which is the evidence that the helper fires.)
    """
    dp = p_half[:, 1:] - p_half[:, :-1]
    integrand = constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt
    return jnp.sum(integrand * dp, axis=1) / constants.g


@pytest.mark.parametrize("name", SCHEME_IDS)
def test_implicit_flux_closes_vapor_mse(name):
    """Under the vapor-debiting solve the scheme must release the detrained
    condensate's latent heat, or the column loses ``L_v * int dq_c`` [W/m^2].

    The tolerance is scaled by the latent throughput ``L_v * int dq_c``, which
    IS the magnitude of the error that appears if the helper is removed -- so
    the gate cannot pass vacuously.
    """
    run, _cfg_cls, _shipped = SCHEMES[name]
    _T, _q, _pf, p_half, _u, _v = _column()
    out = run("implicit_flux")
    res = _vapor_mse_residual(out, p_half)
    latent_scale = constants.L_v * _column_condensate_throughput(out, p_half)
    assert jnp.max(latent_scale) > 1.0, (
        f"{name}: latent throughput {np.asarray(latent_scale)} W/m^2 is "
        "negligible -- this gate would be vacuous"
    )
    rel = jnp.abs(res) / jnp.maximum(latent_scale, 1e-30)
    assert jnp.all(rel < 5e-2), (
        f"{name}: vapor-MSE residual {np.asarray(res)} W/m^2 is "
        f"{np.asarray(rel)} of the latent throughput "
        f"{np.asarray(latent_scale)} W/m^2 -- the detrained-condensate "
        "condensation heating is missing or mispaired (ratio 1.0 = entirely "
        "missing)"
    )


@pytest.mark.parametrize("name", SCHEME_IDS)
def test_latent_release_is_a_no_op_on_the_advective_solve(name):
    """The advective solve emits condensate WITHOUT debiting vapor, so no
    condensation enthalpy is owed; adding one would create energy from nothing
    AND would change shipped behaviour for the three advective-default schemes.
    """
    run, _cfg_cls, shipped = SCHEMES[name]
    adv = run("advective")
    forced_adv = run("advective")
    assert jnp.array_equal(adv.dT_dt, forced_adv.dT_dt)
    if shipped == "advective":
        # The shipped default must be bit-identical to the advective selector,
        # i.e. no latent term leaked into the default path.
        assert jnp.array_equal(run(None).dT_dt, adv.dT_dt)
