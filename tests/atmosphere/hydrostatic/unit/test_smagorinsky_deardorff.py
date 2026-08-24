"""Deardorff (1980) stable-length option for the Smagorinsky closure (#1508-class).

The Lilly form multiplies by ``√(max(0, 1 − Ri/Pr_t))``, guarded only where the
argument goes NEGATIVE. As it approaches zero from ABOVE the slope ``0.5/√x``
diverges, which is why the scheme scored |dLoss/dparam| = 7.4e18 on the stable
GABLS1 case and could not take a single reducing step, against 0.013-0.375 on
the seven other LES cases.
"""
import jax
import jax.test_util
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics._shared import (  # noqa: E402
    deardorff_stable_eddy_viscosity,
    lilly_buoyancy_factor,
)

# A stable column: strong positive N², weak shear — the GABLS1 regime, and the
# one that parks the Lilly argument next to its cutoff.
_S2_STABLE = 1.0e-4
_N2_STABLE = 0.99e-4          # 1 - Ri/Pr_t = 1e-2 at Pr_t = 1
_LENGTH = 50.0


def _km_lilly(S2, N2, length, c_s, Pr_t):
    Ri = N2 / S2
    return (c_s * length) ** 2 * jnp.sqrt(S2) * lilly_buoyancy_factor(Ri, Pr_t)


@pytest.mark.parametrize("argnum,name", [(0, "S2"), (1, "N2")])
def test_deardorff_gradient_is_bounded_where_lilly_diverges(argnum, name):
    """BOUNDED, not merely finite — 0.5/√1e-36 = 5e17 is finite and useless.

    The Lilly divergence is in the gradient w.r.t. the STRATIFICATION-bearing
    inputs (S2 and N2, which enter Ri), NOT w.r.t. C_s: d Km/d C_s =
    2·C_s·l²·S·f vanishes as f→0 at the cutoff, so a C_s probe would not
    reproduce the defect and the control would be vacuous.
    """
    # Sit just above the Lilly cutoff, where its slope blows up.
    n2 = _S2_STABLE * (1.0 - 1.0e-14)
    g_deard = jax.grad(
        lambda *a: deardorff_stable_eddy_viscosity(*a).sum(), argnums=argnum
    )(_S2_STABLE, n2, _LENGTH, 0.2, 1.0)
    g_lilly = jax.grad(
        lambda *a: _km_lilly(*a).sum(), argnums=argnum
    )(_S2_STABLE, n2, _LENGTH, 0.2, 1.0)

    assert np.isfinite(g_deard), f"{name}: deardorff gradient non-finite"
    assert abs(g_deard) < 1.0e6, (
        f"{name}: deardorff |grad| = {abs(g_deard):.3g}, not bounded")
    # Non-vacuous: the SAME probe must show the old form is pathological, so
    # this test cannot pass by accident if the fix is reverted.
    assert abs(g_lilly) > 1.0e6, (
        f"{name}: lilly |grad| = {abs(g_lilly):.3g} — the control did not "
        "reproduce the defect, so the bound above proves nothing")


def test_reduces_to_plain_smagorinsky_when_not_stable():
    """N² ≤ 0 must collapse EXACTLY to (c_s·l)²·√(S² − N²/Pr_t).

    Includes S² at the PRODUCTION strain floor (1e-10, the caller's
    ``|∂V/∂z|² + 1e-10`` at zero shear): the reviewer found an earlier
    ``_STRAIN_FLOOR = 1e-9`` broke exactly this corner (√1e-9 vs √1e-10) while a
    comfortable S² = 1e-4 hid it. The floor must sit BELOW the caller's, so it
    does not bite at neutral zero shear.
    """
    for s2 in (_S2_STABLE, 1.0e-10):        # comfortable AND the production floor
        for n2 in (0.0, -1.0e-4, -1.0e-3):
            got = deardorff_stable_eddy_viscosity(s2, n2, _LENGTH, 0.2, 1.0)
            want = (0.2 * _LENGTH) ** 2 * jnp.sqrt(s2 - n2 / 1.0)
            np.testing.assert_allclose(got, want, rtol=1e-12, atol=0.0,
                                       err_msg=f"s2={s2:g} n2={n2:g}")


def test_check_grads_first_order_in_smooth_region():
    """First-order reverse-mode check AWAY from the clip kink.

    The Deardorff length has a ``clip(·, 0.1·l, l)`` whose crossovers are
    non-C¹ by construction (a design feature: the clip is what BOUNDS the
    gradient), so a 2nd-order check across a crossover legitimately fails —
    that is smoothness the closure never claims. Pick strong stable
    stratification where ``smix`` sits strictly inside the clip, and check the
    FIRST derivative the optimizer actually consumes.
    """
    # smix_raw = sqrt(0.76*tk/(Ck*sqrt(N2))); at S2=1e-3, N2=5e-4, l=50 this is
    # comfortably inside (0.1*l, l), so the function is locally smooth.
    s2, n2 = 1.0e-3, 5.0e-4
    jax.test_util.check_grads(
        lambda a, b: deardorff_stable_eddy_viscosity(a, b, _LENGTH, 0.2, 1.0),
        (s2, n2), order=1, modes=("rev",), rtol=3e-2)


def test_stable_stratification_suppresses_mixing():
    """Sanity: more stable => less mixing. N²>0 is STABLE."""
    weak = deardorff_stable_eddy_viscosity(_S2_STABLE, 1e-6, _LENGTH, 0.2, 1.0)
    strong = deardorff_stable_eddy_viscosity(_S2_STABLE, 0.9e-4, _LENGTH, 0.2, 1.0)
    assert float(strong) < float(weak)


def test_unknown_stability_form_raises():
    from legoesm.atmosphere.physics.turbulence.config import SmagorinskyConfig
    assert SmagorinskyConfig().stability_form == "lilly"   # default unchanged
    cfg = SmagorinskyConfig(stability_form="deardorf")     # typo
    ncol, nlev = 1, 8
    from legoesm.atmosphere.physics.turbulence.smagorinsky import (
        smagorinsky_turbulence,
    )
    with pytest.raises(ValueError, match="stability_form"):
        smagorinsky_turbulence(
            jnp.full((ncol, nlev), 5.0),                    # u
            jnp.zeros((ncol, nlev)),                        # v
            jnp.full((ncol, nlev), 280.0),                  # T
            jnp.full((ncol, nlev), 5e-3),                   # q_v
            jnp.linspace(9e4, 2e4, nlev)[None, :],          # p_full (ncol,nlev)
            jnp.linspace(9.5e4, 1.5e4, nlev + 1)[None, :],  # p_half (ncol,nlev+1)
            jnp.linspace(1000.0, 10.0, nlev)[None, :],      # z_full (ncol,nlev)
            jnp.linspace(1100.0, 0.0, nlev + 1)[None, :],   # z_half (ncol,nlev+1)
            jnp.full((ncol,), 285.0),                       # T_sfc (ncol,)
            jnp.full((ncol,), 6e-3),                        # q_sfc (ncol,)
            jnp.full((ncol, nlev), 1.1),                    # rho (ncol,nlev)
            60.0, cfg,                                      # dt, config
        )
