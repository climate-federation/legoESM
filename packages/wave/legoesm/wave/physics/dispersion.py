"""Linear wave dispersion on a regular lat-lon grid.

Adapted from WADEN 46aadc7, ``src/waden/physics.py``. Fixed-count
bisection has an implicit root derivative. Wet support is held fixed;
depth cutoffs and numerical clips retain the oracle's piecewise AD contract.

References
----------
WW3 Manual v5.16, Eq. (2.6) (group velocity).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.wave.constants import GRAVITY, WAVENUM_MAX_ITER

__physics_contract__ = {
    "summary": (
        "Linear surface-gravity dispersion: solve k from omega^2 = g k tanh(k d) "
        "by a fixed-count bisection, then the group velocity Cg = n sigma / k with "
        "n = 0.5 + k d / sinh(2 k d)."
    ),
    "inputs": {
        "freq": "wave frequency [Hz], (1,1,1,1,F)",
        "depth": "water depth [m], (B,Nlon,Nlat,1,1)",
    },
    "outputs": {
        "wavenum": "wavenumber [rad/m], (B,Nlon,Nlat,1,F)",
        "group_velocity": "group velocity [m/s], same shape",
    },
    "sign_convention": (
        "All quantities positive; Cg carries the direction through the "
        "caller's sin/cos(theta) projection."),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "WADEN 46aadc7 src/waden/physics.py (v2 solver); WW3 manual v5.16 Eq. (2.6).",
    "idealized_test": "tests/wave/unit/test_dispersion.py::test_deep_water_limit",
}

# Clip bounds for kd in the group-velocity expression.  At kd >= 20, n is 0.5
# to float32 precision; the floor keeps the expression finite at kd = 0.
_MIN_KD = 1e-8
_MAX_KD = 20.0


@jax.custom_jvp
def _dispersion_tanh(z):
    """Keep tanh's primal, with an accurate derivative on nonnegative kd."""
    return jnp.tanh(z)


@_dispersion_tanh.defjvp
def _dispersion_tanh_jvp(primals, tangents):
    # sech²(z) = 4 exp(-2z)/(1 + exp(-2z))² avoids subtracting nearly
    # equal float32 values in 1-tanh(z)². This is tanh's mathematical
    # derivative, not a surrogate for the root or its former rounding.
    z, = primals
    z_dot, = tangents
    decay = jnp.exp(-2 * z)
    sech2 = 4 * decay / (1 + decay)**2
    return _dispersion_tanh(z), sech2 * z_dot



def solve_wavenumber(freq, depth, gravity=GRAVITY, max_iter=WAVENUM_MAX_ITER):
    """Solve the gravity-wave dispersion relation with an implicit derivative.

    The forward solve is a fixed-iteration bisection on the wet support
    ``depth > 0`` and ``freq > 0``.  ``jax.lax.custom_root`` supplies the
    implicit derivative, so ``jax.grad``/``jax.jvp`` see the derivative of
    ``g k tanh(k d) = (2*pi*f)^2`` rather than the derivative of a rounded or
    branch-selected bisection output.  Land and zero-frequency points return
    ``k = 0`` with zero derivative by explicit safe-input substitution.
    The derivative approximates that of the converged continuous root within
    the measured solve tolerance; it is not a derivative of floating-point
    branch decisions. Positive depth/frequency support is held fixed for AD.
    See WAVEWATCH III (2016), Eq. (2.1). The residual retains tanh's forward
    value and uses its stable analytic sech-squared derivative near saturation.

    Parameters
    ----------
    freq : jax.Array
        Wave frequency (Hz). Shape: [F] or broadcastable.
    depth : jax.Array
        Water depth (m). Shape: [B, H, W, 1, 1].
    gravity : float
        Gravitational acceleration (m/s^2).
    max_iter : int
        Number of bisection iterations.

    Returns
    -------
    jax.Array
        Wave number (rad/m). Shape: [B, H, W, 1, F].
    """
    # depth: [B, H, W, 1, 1] -> [B, H, W, 1, 1, 1]
    depth_ext = jnp.expand_dims(depth, axis=-1)
    omega = 2 * jnp.pi * freq
    wet = (depth_ext > 0) & (freq > 0)
    # Safe substitution keeps the unselected branch finite; output masking alone
    # is not enough because the unselected inf/NaN can poison reverse mode.
    safe_depth = jnp.where(wet, depth_ext, 1.0)
    safe_omega = jnp.where(wet, omega, 1.0)

    def residual(k):
        z = k * safe_depth
        return gravity * k * _dispersion_tanh(z) - safe_omega**2

    deep = safe_omega**2 / gravity
    shallow = safe_omega / jnp.sqrt(gravity * safe_depth)
    initial = 2.0 * jnp.maximum(deep, shallow) + 1e-12

    def bisect(f, initial_guess):
        lower = jnp.zeros_like(initial_guess)
        upper = initial_guess

        def body_fn(_, bounds):
            low, high = bounds
            mid = 0.5 * (low + high)
            value = f(mid)
            low = jnp.where(value <= 0, mid, low)
            high = jnp.where(value > 0, mid, high)
            return low, high

        lower, upper = jax.lax.fori_loop(0, max_iter, body_fn, (lower, upper))
        return 0.5 * (lower + upper)

    def tangent_solve(g, y):
        return y / g(jnp.ones_like(y))

    k = jax.lax.custom_root(residual, initial, bisect, tangent_solve)
    k = jnp.where(wet, k, 0.0)
    # Remove the extra depth dimension: [B,H,W,1,1,F] -> [B,H,W,1,F]
    return jnp.squeeze(k, axis=-2)



def compute_group_velocity(freq, wavenum, depth):
    """
    Calculate group velocity of the wave field.

    Cg = n * (sigma / k)
    where n = 0.5 + k*d / sinh(2*k*d),  sigma = 2*pi*f

    See WW3 manual v5.16, Page 12, Equation (2.6).
    JAX equivalent of waden_hpc.physics_calculation.GroupVelocityCalculator.

    Parameters
    ----------
    freq : jax.Array
        Wave frequency (Hz). Shape: [F] or broadcastable.
    wavenum : jax.Array
        Wave number (rad/m). Shape: [B, H, W, 1, F].
    depth : jax.Array
        Water depth (m). Shape: [B, H, W, 1, 1].

    Returns
    -------
    jax.Array
        Group velocity (m/s). Same shape as wavenum.
    """
    sigma = 2 * jnp.pi * freq
    wet = wavenum > 1e-7
    safe_wavenum = jnp.where(wet, wavenum, 1.0)
    # Bound kd before sinh so the unselected deep-water branch cannot overflow
    # in either forward or reverse mode.  At kd >= 20, n is 0.5 to float32
    # precision; the band below uses the exact z/sinh(2z) expression.
    z = jnp.clip(wavenum * depth, min=_MIN_KD, max=_MAX_KD)
    n = 0.5 + z / jnp.sinh(2.0 * z)
    Cg = n * (sigma / safe_wavenum)
    return jnp.where(wet, Cg, 0.0)



def calculate_depth_dispersion_derivative(cg, wavenum, freq, depth):
    """
    Calculate the depth dispersion derivative (DsDd).

    DsDd = (cg * k - 0.5 * f * 2*pi) / depth
    Set to 0 where k*d >= 5 (deep water limit) and clamped to non-negative.

    Used in theta/k tendency calculations for depth refraction.
    JAX equivalent of waden_hpc.physics_calculation.calculate_depth_dispersion_derivative.

    Parameters
    ----------
    Cg : jax.Array
        Group velocity.
    wavenum : jax.Array
        Wave number.
    freq : jax.Array
        Wave frequency (Hz).
    depth : jax.Array
        Water depth (m).

    Returns
    -------
    jax.Array
        Depth dispersion derivative. Non-negative, zero in deep water.

    Notes
    -----
    The retained ``k*d >= 5`` cutoff generally introduces a value jump,
    not just a derivative kink. AD describes the selected smooth branch
    away from the cutoff and other masks/clips; it does not differentiate
    crossings of the cutoff or changes to wet support.
    """
    # Deep water mask: k*d >= 5 means deep water (no bottom effect)
    mk_dsdd = depth * wavenum >= 5

    safe_depth = jnp.where(depth > 0, depth, 1.0)
    sigma = freq * 2 * jnp.pi
    DsDd = (cg * wavenum - 0.5 * sigma) / safe_depth

    # The deep branch is intentionally zeroed by the existing contract; the
    # unselected value above is still finite because Cg uses the stable z form.
    DsDd = jnp.where((depth > 0) & (wavenum > 0) & (~mk_dsdd), DsDd, 0.0)
    # Physical constraint: DsDd must be non-negative.  This clip is a declared
    # piecewise-smooth point; it is kept to preserve the existing forward rule.
    return jnp.clip(DsDd, min=0.0)
