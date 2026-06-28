"""fp32 marginal-supersaturation recovery in saturation_adjustment (issue #618).

The warm-Cu condensate is the small residual ``q_v - q_sat`` riding on a large
``q_sat`` base.  The precision loss is NOT the subtraction (``q_v`` and ``q_sat``
are close ⇒ Sterbenz-exact in fp32) but ``q_sat``'s fp32 COMPUTATION (the
exp / softplus / division chain in ``saturation_mixing_ratio`` carries
~1.7e-8 kg/kg absolute error at BOMEX conditions).  For a marginal residual
~1e-6 kg/kg that is a ~2 % per-cell under-formation, which the sigmoid
condensation threshold + the conserving-positive rescale amplify into the
~65 % LWP deficit reported in fp32 LES.

The fix computes ``q_sat`` and the residual in fp64 inside the otherwise-fp32
step.  These tests need real fp64, so x64 is enabled at import.
"""

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)  # before any array is created

import jax.numpy as jnp  # noqa: E402

from legoesm.atmosphere.physics.microphysics._warm_rain import (  # noqa: E402
    saturation_adjustment,
)
from legoesm.thermo import saturation_mixing_ratio  # noqa: E402

# BOMEX-like marginal warm cumulus: T ~ 298 K, p ~ 970 hPa, q_sat ~ 21 g/kg.
_T = 298.0
_P = 9.7e4


def _cond(T, q_v, p, dt=20.0):
    """Scalar condensation rate from saturation_adjustment (1x1 column)."""
    cond, _ = saturation_adjustment(
        jnp.asarray([[T]]), jnp.asarray([[q_v]]), jnp.asarray([[p]]), dt,
    )
    return float(cond[0, 0])


def test_fp32_matches_fp64_on_marginal_supersaturation():
    """The fp32 step recovers the fp64 condensation for a marginal residual.

    q_sat computed in fp64 inside the step (the fix); without it the fp32 q_sat
    computation error (~1.7e-8) would leave the ~1e-6 residual ~2 % low and this
    ratio would fall well below the tolerance.
    """
    q_sat64 = float(saturation_mixing_ratio(
        jnp.asarray(_T, jnp.float64), jnp.asarray(_P, jnp.float64)))
    residual = 1.0e-6  # marginal supersaturation [kg/kg]
    q_v = q_sat64 + residual

    # fp32 state (the production LES dtype) vs fp64 ground truth.
    cond_f32 = _cond(jnp.float32(_T), jnp.float32(q_v), jnp.float32(_P))
    cond_f64 = _cond(jnp.float64(_T), jnp.float64(q_v), jnp.float64(_P))

    assert cond_f64 > 0.0
    ratio = cond_f32 / cond_f64
    assert ratio > 0.99, (
        f"fp32 condensation under-forms vs fp64 (ratio={ratio:.4f}); the fp64 "
        f"q_sat path did not recover the marginal residual. "
        f"cond_f32={cond_f32:.4e} cond_f64={cond_f64:.4e}"
    )


def test_naive_fp32_qsat_would_under_form_residual():
    """Non-vacuous guard: a naive all-fp32 q_sat (what the fix removes) really

    does mis-resolve the marginal residual, so the test above is meaningful.
    """
    q_sat64 = saturation_mixing_ratio(
        jnp.asarray(_T, jnp.float64), jnp.asarray(_P, jnp.float64))
    q_sat32 = saturation_mixing_ratio(
        jnp.asarray(_T, jnp.float32), jnp.asarray(_P, jnp.float32))
    residual = 1.0e-6
    q_v32 = jnp.float32(float(q_sat64) + residual)

    naive_excess = float(q_v32 - q_sat32)            # all-fp32 (pre-fix)
    fixed_excess = float(q_v32.astype(jnp.float64) - q_sat64)  # fp64 q_sat (fix)
    truth = residual

    naive_err = abs(naive_excess - truth) / truth
    fixed_err = abs(fixed_excess - truth) / truth
    # The naive fp32 path is materially wrong; the fp64-q_sat path is ~exact.
    assert naive_err > 0.005, f"expected naive fp32 to mis-resolve, got {naive_err:.2%}"
    assert fixed_err < 1.0e-3, f"fp64 q_sat should be ~exact, got {fixed_err:.2%}"


def test_condensation_is_differentiable_wrt_qv():
    """The fp64 promotion keeps the step differentiable (astype VJP is identity)."""
    q_sat64 = float(saturation_mixing_ratio(
        jnp.asarray(_T, jnp.float64), jnp.asarray(_P, jnp.float64)))

    def loss(q_v):
        cond, _ = saturation_adjustment(
            jnp.asarray([[_T]]), jnp.reshape(q_v, (1, 1)), jnp.asarray([[_P]]),
            20.0,
        )
        return jnp.sum(cond)

    g = jax.grad(loss)(jnp.asarray(q_sat64 + 1.0e-6))
    assert jnp.isfinite(g) and float(g) > 0.0  # more vapour ⇒ more condensation


if __name__ == "__main__":
    test_fp32_matches_fp64_on_marginal_supersaturation()
    test_naive_fp32_qsat_would_under_form_residual()
    test_condensation_is_differentiable_wrt_qv()
    print("ok")
