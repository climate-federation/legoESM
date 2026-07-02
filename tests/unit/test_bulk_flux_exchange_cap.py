"""Unit tests for the optional ``max_exchange_coeff`` ceiling in
``compute_most_fluxes`` (the land cold-start SEB stabiliser).

Root cause it guards: at extreme surface-layer instability the MOST log-law
denominators ``ln(z/z0) - psi`` collapse toward the historical ``0.5`` floor,
so the neutral-equivalent transfer coefficient ``C = kappa^2/(denom_m*denom_h)``
reaches ``kappa^2/0.25 ~ 0.64`` — ~190x the neutral ~3.4e-3.  That spurious
coefficient turns a trivial surface/air gradient into an O(1e3) W/m^2 flux shock
that drives the stiff thin top soil layer NaN (dt-independent — it is the
coefficient, not the time integrator).  ``max_exchange_coeff=C_max`` floors each
denominator at ``max(0.5, kappa/sqrt(C_max))`` so ``C_d, C_h, C_e <= C_max``,
while the fluxes stay linear in the T/q gradients (the SEB keeps its
self-limiting feedback).  ``None`` (default) must be byte-identical to the prior
behaviour for every existing ocean/atmosphere caller.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.core.bulk_flux import KAPPA, compute_most_fluxes  # noqa: E402


def _extreme_instability_inputs():
    """A strongly unstable surface layer (hot surface, weak wind, very rough
    land) that drives BOTH MOST denominators onto their 0.5 floor — the singular
    regime where the uncapped coefficient reaches kappa^2/0.25 ~ 0.64.

    The floor is hit when ``ln(z/z0) - psi`` <= 0.5.  With z0 = 2 m (dense
    forest/urban roughness) at z = 10 m: ln(10/2) = 1.61 and psi_m(-10) ~ 2.5 ->
    denom_m floored; ln(10/0.2) = 3.91 (z0_t = z0/10) and psi_h(-10) ~ 3.85 ->
    denom_h floored.  So C_h = kappa^2/(0.5*0.5) = 0.64 without the cap."""
    T_atm = jnp.asarray(280.0)
    T_sfc = jnp.asarray(340.0)          # surface 60 K warmer -> very unstable
    q_atm = jnp.asarray(2.0e-3)
    q_sfc = jnp.asarray(6.0e-3)
    rho = jnp.asarray(1.2)
    u_rel = jnp.asarray(1.0)            # weak mechanical mixing -> zeta -> -10
    v_rel = jnp.asarray(0.0)
    return dict(
        u_rel=u_rel, v_rel=v_rel, T_atm=T_atm, q_atm=q_atm,
        T_sfc=T_sfc, q_sfc=q_sfc, rho=rho,
        z_ref=10.0, z0_init=2.0, scheme="most", n_iter=8,
    )


def _implied_ch(shflx, kw):
    """Back out the sensible-heat transfer coefficient from the flux:
    shflx = rho * c_pd * C_h * U * (T_sfc - T_atm).  Uses the SAME wind-speed
    regulariser the solver applies internally (``sqrt(u^2+v^2+1e-4)``) so the
    inversion is exact — otherwise the ~1e-4 calm-wind floor shows up as a ~5e-5
    relative overshoot in the recovered coefficient."""
    U = float(jnp.sqrt(kw["u_rel"] ** 2 + kw["v_rel"] ** 2 + 1e-4))
    dT = float(kw["T_sfc"] - kw["T_atm"])
    return float(shflx) / (float(kw["rho"]) * constants.c_pd * U * dT)


def test_uncapped_coefficient_blows_up_at_extreme_instability():
    """Without the cap the implied C_h reaches ~kappa^2/0.25 (~0.64) — the
    documented singularity — and the sensible flux is O(1e3) W/m^2."""
    kw = _extreme_instability_inputs()
    _, _, shflx, _, _ = compute_most_fluxes(**kw)          # default None
    ch = _implied_ch(shflx, kw)
    assert ch > 0.4, f"expected C_h near the 0.5-floor singularity, got {ch:.4g}"
    assert ch <= KAPPA ** 2 / 0.25 + 1e-9                  # bounded by the 0.5 floor
    assert float(shflx) > 500.0                            # a large flux shock


def test_cap_bounds_the_coefficient():
    """With ``max_exchange_coeff=C_max`` the implied C_h cannot exceed C_max."""
    kw = _extreme_instability_inputs()
    c_max = 0.02
    _, _, shflx, lhflx, _ = compute_most_fluxes(**kw, max_exchange_coeff=c_max)
    ch = _implied_ch(shflx, kw)
    assert ch <= c_max * (1.0 + 1e-6), f"C_h {ch:.4g} exceeds cap {c_max}"
    # The latent flux is likewise bounded (same denominators): the O(1e3) W/m^2
    # shock is removed.
    assert abs(float(lhflx)) < 1000.0


def test_cap_is_much_smaller_than_uncapped():
    """The cap must materially reduce the flux in the singular regime (not a
    no-op)."""
    kw = _extreme_instability_inputs()
    _, _, shflx_none, _, _ = compute_most_fluxes(**kw)
    _, _, shflx_cap, _, _ = compute_most_fluxes(**kw, max_exchange_coeff=0.02)
    assert float(shflx_cap) < 0.25 * float(shflx_none)


def test_default_none_byte_identical_to_explicit_half_floor():
    """``None`` == the historical 0.5 floor.  A ``C_max`` whose implied floor is
    <= 0.5 (i.e. C_max >= kappa^2/0.25) must be bit-for-bit identical to the
    default, proving the default path is unchanged."""
    kw = _extreme_instability_inputs()
    out_none = compute_most_fluxes(**kw)
    # kappa/sqrt(C_max) <= 0.5  <=>  C_max >= (kappa/0.5)^2 = 0.64
    c_max_noop = (KAPPA / 0.5) ** 2
    out_noop = compute_most_fluxes(**kw, max_exchange_coeff=c_max_noop)
    for a, b in zip(out_none, out_noop):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_cap_inert_in_benign_regime():
    """In a well-behaved (near-neutral, windy) surface layer the denominators
    are far above the floor, so the cap does not bind — capped == uncapped."""
    kw = dict(
        u_rel=jnp.asarray(8.0), v_rel=jnp.asarray(0.0),
        T_atm=jnp.asarray(288.0), q_atm=jnp.asarray(6.0e-3),
        T_sfc=jnp.asarray(290.0), q_sfc=jnp.asarray(8.0e-3),
        rho=jnp.asarray(1.2), z_ref=10.0, z0_init=1e-3,
        scheme="most", n_iter=8,
    )
    out_none = compute_most_fluxes(**kw)
    out_cap = compute_most_fluxes(**kw, max_exchange_coeff=0.02)
    for a, b in zip(out_none, out_cap):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10)


def test_cap_preserves_gradient_monotonicity():
    """The cap must keep the flux linear in the surface/air gradient (so the SEB
    self-limits): a larger T_sfc-T_atm gradient still gives a larger flux."""
    base = _extreme_instability_inputs()
    kw_hi = dict(base)
    kw_hi["T_sfc"] = jnp.asarray(345.0)     # even larger gradient
    _, _, shflx_lo, _, _ = compute_most_fluxes(**base, max_exchange_coeff=0.02)
    _, _, shflx_hi, _, _ = compute_most_fluxes(**kw_hi, max_exchange_coeff=0.02)
    assert float(shflx_hi) > float(shflx_lo)


def test_cap_differentiable():
    """The capped path stays jax.grad-friendly (feature gate is a static Python
    float, so no traced control flow is introduced)."""
    kw = _extreme_instability_inputs()

    def _flux_of_tsfc(T_sfc):
        out = compute_most_fluxes(
            kw["u_rel"], kw["v_rel"], kw["T_atm"], kw["q_atm"],
            T_sfc, kw["q_sfc"], kw["rho"],
            z_ref=kw["z_ref"], z0_init=kw["z0_init"], scheme=kw["scheme"],
            n_iter=kw["n_iter"], max_exchange_coeff=0.02,
        )
        return out[2]  # shflx

    g = jax.grad(_flux_of_tsfc)(jnp.asarray(340.0))
    assert np.isfinite(float(g))
    assert float(g) > 0.0  # hotter surface -> more upward sensible heat


# ---------------------------------------------------------------------------
# large_yeager coefficient-space branch: the cap must ALSO bind there (it
# bypasses the log-law denominators), else the "C <= C_max" guarantee is false
# for that valid land scheme.
# ---------------------------------------------------------------------------

def _ly_inputs():
    return dict(
        u_rel=jnp.asarray(8.0), v_rel=jnp.asarray(0.0),
        T_atm=jnp.asarray(288.0), q_atm=jnp.asarray(6.0e-3),
        T_sfc=jnp.asarray(292.0), q_sfc=jnp.asarray(9.0e-3),
        rho=jnp.asarray(1.2), z_ref=10.0, z0_init=1e-4,
        scheme="large_yeager", n_iter=8,
    )


def test_large_yeager_cap_binds_in_coefficient_space():
    """A tight ceiling below LY's natural C_h (~1.3e-3) must cap the
    coefficient-space rd/rh/re and reduce the flux — proving the LY branch
    honours max_exchange_coeff (codex MED)."""
    kw = _ly_inputs()
    c_max = 5.0e-4
    _, _, shflx_none, _, _ = compute_most_fluxes(**kw)
    _, _, shflx_cap, _, _ = compute_most_fluxes(**kw, max_exchange_coeff=c_max)
    ch_cap = _implied_ch(shflx_cap, kw)
    assert ch_cap <= c_max * (1.0 + 1e-6), f"LY C_h {ch_cap:.4g} exceeds cap"
    assert abs(float(shflx_cap)) < abs(float(shflx_none))  # cap actually binds


def test_large_yeager_cap_inert_when_ceiling_high():
    """A ceiling well above LY's natural coefficients must not bind — capped ==
    uncapped, so the default LY (ocean/OMIP) physics is untouched."""
    kw = _ly_inputs()
    out_none = compute_most_fluxes(**kw)
    out_cap = compute_most_fluxes(**kw, max_exchange_coeff=1.0)  # >> LY C ~1e-3
    for a, b in zip(out_none, out_cap):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-10)


def test_traced_max_exchange_coeff_rejected():
    """A traced (non-static) ceiling must fail loudly, not silently mis-trace
    the Python `max`/`if` gate (codex LOW)."""
    kw = _extreme_instability_inputs()

    def _run(c):
        return compute_most_fluxes(**kw, max_exchange_coeff=c)

    with pytest.raises(TypeError):
        jax.jit(_run)(jnp.asarray(0.02))


@pytest.mark.parametrize("bad", [0.0, -0.01])
def test_nonpositive_max_exchange_coeff_rejected(bad):
    """C_max <= 0 must fail loudly: 0 -> infinite floor (silent zero fluxes),
    negative -> sqrt of a negative (NaN). Not silent garbage."""
    kw = _extreme_instability_inputs()
    with pytest.raises(ValueError):
        compute_most_fluxes(**kw, max_exchange_coeff=bad)


def test_numpy_float_scalar_accepted():
    """A numpy real scalar (e.g. np.float64 read from a config) is a valid
    static ceiling — must be accepted and give the same result as the
    equivalent Python float."""
    kw = _extreme_instability_inputs()
    out_py = compute_most_fluxes(**kw, max_exchange_coeff=0.02)
    out_np = compute_most_fluxes(**kw, max_exchange_coeff=np.float64(0.02))
    for a, b in zip(out_py, out_np):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
