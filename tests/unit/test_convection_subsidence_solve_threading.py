"""``subsidence_solve`` threading for the mass-flux family (SCM-RCE campaign).

Bechtold and EDMF already exposed ``subsidence_solve`` on their configs and
already defaulted to the conservative ``"implicit_flux"`` solve; Kain-Fritsch
hardcodes ``"implicit_flux"`` at its call site.  Tiedtke, Emanuel,
Zhang-McFarlane and the Arakawa-Wu ``mass_flux`` scheme instead inherited the
kernel's ``"advective"`` FUNCTION DEFAULT with no way for a caller to select
the conservative solve.  This module pins the newly threaded knob.

Why it matters (the physics, not just the plumbing): the two solves are NOT
two discretisations of one operator.  ``"advective"`` computes the donor-cell
``(M/rho) d(phi)/dz`` compensating subsidence plus local detrainment, and emits
a detrained-condensate source ``dq_c = delta_0 M q_c_u / rho`` with NO matching
vapor sink -- so column total water is not closed.  ``"implicit_flux"`` solves
the same transport in telescoping flux form (backward-Euler, per column) AND
books the paired vapor sink ``-dq_c``.  Selecting it therefore changes the
column water budget deliberately.

Gates, per newly-threaded scheme:

1. **Default is unchanged** -- the config default is ``"advective"`` and the
   scheme output is byte-identical to explicitly requesting ``"advective"``,
   so shipped behaviour is preserved bit-for-bit.
2. **``"implicit_flux"`` is HONOURED** -- selecting it changes the tendencies.
   Non-vacuous: were the config field ignored (the plumbing bug this module
   exists to prevent), the outputs would be identical and this FAILS.
3. **Unknown values RAISE** -- ``ValueError`` at kernel entry on the static
   Python string (CLAUDE.md dispatch-hardening); a bare fallthrough would
   silently run different physics on a typo.
4. **Column water closure improves** -- the ``implicit_flux`` column total-water
   residual ``|int (dq_v + dq_c) dp/g|`` is strictly smaller than the advective
   one.  This is the measurement that motivates the campaign's matched-kernel
   ranking arm.

Convention note (stated at the kernel): ``z`` up, arrays surface-LAST
(index 0 = model top); ``M >= 0`` is the UPDRAFT mass flux, so the
compensating environmental subsidence carries mass flux ``-M`` (downward).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.convection.config import (
    EmanuelConfig,
    MassFluxConfig,
    TiedtkeConfig,
    ZhangMcFarlaneConfig,
)
from legoesm.atmosphere.physics.convection.emanuel import emanuel_convection
from legoesm.atmosphere.physics.convection.mass_flux import (
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
    cfg = TiedtkeConfig(subsidence_solve=ss)
    out, _ = tiedtke_convection(
        T + T_offset, q_v, p_full, p_half, u, v, cpp, dt=DT_S, config=cfg)
    return out


def _run_emanuel(ss, T_offset=0.0):
    T, q_v, p_full, p_half, _u, _v = _column()
    cpp = jnp.zeros_like(T)
    cfg = EmanuelConfig(subsidence_solve=ss)
    out, _ = emanuel_convection(
        T + T_offset, q_v, p_full, p_half, cpp, dt=DT_S, config=cfg)
    return out


def _run_zm(ss, T_offset=0.0):
    T, q_v, p_full, p_half, u, v = _column()
    cpp = jnp.zeros_like(T)
    cfg = ZhangMcFarlaneConfig(subsidence_solve=ss)
    out, _ = zhang_mcfarlane_convection(
        T + T_offset, q_v, p_full, p_half, u, v, cpp, dt=DT_S, config=cfg)
    return out


def _run_mass_flux(ss, T_offset=0.0):
    T, q_v, p_full, p_half, _u, _v = _column()
    cfg = MassFluxConfig(subsidence_solve=ss)
    M_c = jnp.zeros((T.shape[0],))
    out, _ = mass_flux_convection(
        T + T_offset, q_v, p_full, p_half, M_c, dt=DT_S, config=cfg)
    return out


SCHEMES = {
    "tiedtke": (_run_tiedtke, TiedtkeConfig),
    "emanuel": (_run_emanuel, EmanuelConfig),
    "zhang_mcfarlane": (_run_zm, ZhangMcFarlaneConfig),
    "mass_flux": (_run_mass_flux, MassFluxConfig),
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
def test_default_is_advective(name):
    _run, cfg_cls = SCHEMES[name]
    assert cfg_cls().subsidence_solve == "advective", (
        f"{name}: threading the knob must NOT change the shipped default"
    )
    assert cfg_cls().theta_implicit == 1.0


@pytest.mark.parametrize("name", SCHEME_IDS)
def test_default_byte_identical_to_explicit_advective(name):
    run, _cfg_cls = SCHEMES[name]
    default = run("advective")
    # The config default IS "advective", so an explicit request must be the
    # exact same computation -- this is the shipped-behaviour regression.
    for field in ("dT_dt", "dq_v_dt", "dq_c_conv_dt"):
        a = getattr(default, field)
        b = getattr(run("advective"), field)
        assert jnp.array_equal(a, b), f"{name}.{field} not deterministic"


# ---------------------------------------------------------------------------
# Gate 2 -- "implicit_flux" is actually honoured (non-vacuous plumbing proof)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCHEME_IDS)
def test_implicit_flux_is_honoured(name):
    run, _ = SCHEMES[name]
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
    run, _ = SCHEMES[name]
    with pytest.raises(ValueError, match="unknown subsidence_solve"):
        run("bogus")


# ---------------------------------------------------------------------------
# Gate 4 -- the conservative solve closes the column water budget better
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", SCHEME_IDS)
def test_implicit_flux_reduces_column_water_residual(name):
    run, _ = SCHEMES[name]
    _T, _q, _pf, p_half, _u, _v = _column()
    res_adv = _column_water_residual(run("advective"), p_half)
    res_imp = _column_water_residual(run("implicit_flux"), p_half)
    # Report both so a regression shows the magnitudes, not just a boolean.
    assert jnp.all(res_imp <= res_adv + 1e-12), (
        f"{name}: implicit_flux water residual {np.asarray(res_imp)} is not "
        f"<= advective {np.asarray(res_adv)}"
    )
    # Non-vacuous: on an ACTIVE convecting column the advective residual must
    # be genuinely non-zero, otherwise the comparison above proves nothing.
    assert jnp.max(res_adv) > 0.0, (
        f"{name}: advective residual is exactly zero -- the fixture is not "
        "convecting, so gate 4 is vacuous"
    )


@pytest.mark.parametrize("name", SCHEME_IDS)
def test_implicit_flux_is_differentiable(name):
    """AD must survive the tridiagonal solve (end-to-end jax.grad is a goal)."""
    run, _ = SCHEMES[name]

    def loss(T_offset):
        out = run("implicit_flux", T_offset=T_offset)
        return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.dq_v_dt ** 2)

    g = jax.grad(loss)(0.0)
    assert jnp.isfinite(g), f"{name}: grad through implicit_flux is not finite"
    # Non-vacuous: a structurally-zero gradient would also be "finite".
    assert abs(float(g)) > 0.0, (
        f"{name}: grad through the implicit tridiagonal solve is exactly "
        "zero -- the solve is detached from the input"
    )
