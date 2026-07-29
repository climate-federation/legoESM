"""
JAX translation of MLWaterVaporMod Fortran module.

Compute saturation vapour pressure and latent heat of vaporization.
Polynomial approximations for saturation vapour pressure follow
Flatau et al. (1992) J. Appl. Meteor. 31:1507-1513.

Original Fortran module: MLWaterVaporMod
Fortran lines 1-115

Differentiability notes
-----------------------
Both ``SatVap`` and ``LatVap`` accept plain Python floats **or** JAX
arrays (scalars or batched).  All branching on temperature uses
``jnp.where`` so the functions are fully differentiable and
``jax.jit``-compatible.
"""

import jax.numpy as jnp

from legoesm.land.canopy.clm_ml_backend.clm_src_main.clm_varcon import hsub, hvap, tfrz  # noqa: F401
from legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLclm_varcon import mmh2o  # noqa: F401

# ---------------------------------------------------------------------------
# Public: saturation vapour pressure and its temperature derivative
# ---------------------------------------------------------------------------

# Polynomial coefficients for water vapour (0 °C to 100 °C) — Fortran lines 35-43
_a0: float = 6.11213476
_a1: float = 0.444007856
_a2: float = 0.143064234e-1
_a3: float = 0.264461437e-3
_a4: float = 0.305903558e-5
_a5: float = 0.196237241e-7
_a6: float = 0.892344772e-10
_a7: float = -0.373208410e-12
_a8: float = 0.209339997e-15

# Derivative coefficients for water vapour — Fortran lines 45-53
_b0: float = 0.444017302
_b1: float = 0.286064092e-1
_b2: float = 0.794683137e-3
_b3: float = 0.121211669e-4
_b4: float = 0.103354611e-6
_b5: float = 0.404125005e-9
_b6: float = -0.788037859e-12
_b7: float = -0.114596802e-13
_b8: float = 0.381294516e-16

# Polynomial coefficients for ice (-75 °C to 0 °C) — Fortran lines 55-63
_c0: float = 6.11123516
_c1: float = 0.503109514
_c2: float = 0.188369801e-1
_c3: float = 0.420547422e-3
_c4: float = 0.614396778e-5
_c5: float = 0.602780717e-7
_c6: float = 0.387940929e-9
_c7: float = 0.149436277e-11
_c8: float = 0.262655803e-14

# Derivative coefficients for ice — Fortran lines 65-73
_d0: float = 0.503277922
_d1: float = 0.377289173e-1
_d2: float = 0.126801703e-2
_d3: float = 0.249468427e-4
_d4: float = 0.313703411e-6
_d5: float = 0.257180651e-8
_d6: float = 0.133268878e-10
_d7: float = 0.394116744e-13
_d8: float = 0.498070196e-16


def SatVap(t):
    """
    Compute saturation vapour pressure and its temperature derivative.

    Mirrors Fortran subroutine ``SatVap`` (lines 22-95).

    Uses 8th-order polynomial fits from Flatau et al. (1992):

    - **Water** (``tc >= 0``, valid 0 °C – 100 °C): coefficients ``a0–a8``
      (es) and ``b0–b8`` (d(es)/dT).
    - **Ice** (``tc < 0``, valid -75 °C – 0 °C): coefficients ``c0–c8``
      (es) and ``d0–d8`` (d(es)/dT).

    Temperature in Celsius is clamped to ``[-75, 100]`` before evaluation
    (Fortran lines 75-76). Results are converted from millibars to Pa
    by multiplying by 100 (Fortran lines 91-92).

    Differentiable: ``jnp.where`` replaces Python ``if`` on temperature
    so the function is ``jax.jit``- and ``jax.grad``-compatible.

    Args:
        t: Temperature (K) — Python float or JAX scalar/array.

    Returns:
        Tuple of ``(es, desdt)`` where:

        - ``es``: Saturation vapour pressure (Pa).
        - ``desdt``: d(es)/dT (Pa/K).
    """
    # Temperature in Celsius, clamped to valid polynomial range — Fortran lines 74-76
    tc = jnp.clip(jnp.asarray(t) - tfrz, -75.0, 100.0)

    # Water vapour polynomials — Fortran lines 78-83 (evaluated for all t)
    es_w = _a0 + tc * (
        _a1 + tc * (_a2 + tc * (_a3 + tc * (_a4 + tc * (_a5 + tc * (_a6 + tc * (_a7 + tc * _a8))))))
    )
    desdt_w = _b0 + tc * (
        _b1 + tc * (_b2 + tc * (_b3 + tc * (_b4 + tc * (_b5 + tc * (_b6 + tc * (_b7 + tc * _b8))))))
    )

    # Ice polynomials — Fortran lines 84-89 (evaluated for all t)
    es_i = _c0 + tc * (
        _c1 + tc * (_c2 + tc * (_c3 + tc * (_c4 + tc * (_c5 + tc * (_c6 + tc * (_c7 + tc * _c8))))))
    )
    desdt_i = _d0 + tc * (
        _d1 + tc * (_d2 + tc * (_d3 + tc * (_d4 + tc * (_d5 + tc * (_d6 + tc * (_d7 + tc * _d8))))))
    )

    # Select water vs ice branch without Python if — differentiable
    above_zero = tc >= 0.0
    es = jnp.where(above_zero, es_w, es_i) * 100.0  # mb → Pa
    desdt = jnp.where(above_zero, desdt_w, desdt_i) * 100.0

    return es, desdt


def SatVap_py(t: float):
    """Pure-Python version of :func:`SatVap` for plain float inputs.

    Uses the same 8th-order polynomial fits but with Python arithmetic
    and ``if/else`` branching.  No JAX dispatch overhead — safe for use
    inside per-layer Python loops.

    Returns ``(es, desdt)`` as plain Python floats (Pa and Pa/K).
    """
    tc = t - tfrz
    if tc < -75.0:
        tc = -75.0
    elif tc > 100.0:
        tc = 100.0

    if tc >= 0.0:
        es = _a0 + tc * (
            _a1
            + tc * (_a2 + tc * (_a3 + tc * (_a4 + tc * (_a5 + tc * (_a6 + tc * (_a7 + tc * _a8))))))
        )
        desdt = _b0 + tc * (
            _b1
            + tc * (_b2 + tc * (_b3 + tc * (_b4 + tc * (_b5 + tc * (_b6 + tc * (_b7 + tc * _b8))))))
        )
    else:
        es = _c0 + tc * (
            _c1
            + tc * (_c2 + tc * (_c3 + tc * (_c4 + tc * (_c5 + tc * (_c6 + tc * (_c7 + tc * _c8))))))
        )
        desdt = _d0 + tc * (
            _d1
            + tc * (_d2 + tc * (_d3 + tc * (_d4 + tc * (_d5 + tc * (_d6 + tc * (_d7 + tc * _d8))))))
        )

    return es * 100.0, desdt * 100.0


# ---------------------------------------------------------------------------
# Public: molar latent heat of vaporization
# ---------------------------------------------------------------------------


def LatVap(t):
    """
    Compute the molar latent heat of vaporization as a function of
    temperature, following the CLM5 formulation.

    Mirrors Fortran function ``LatVap`` (lines 97-115).

    Above the freezing point ``tfrz`` the latent heat of vaporization
    (``hvap``) is used; at or below freezing the latent heat of
    sublimation (``hsub``) is used. The mass-specific value (J/kg) is
    then converted to a molar value (J/mol) by multiplying by the
    molecular mass of water ``mmh2o``:

    .. code-block:: none

        lambda = hvap * mmh2o    if t > tfrz
        lambda = hsub * mmh2o    if t <= tfrz

    Differentiable: ``jnp.where`` replaces Python ``if`` on temperature.

    Args:
        t: Temperature (K) — Python float or JAX scalar/array.

    Returns:
        Molar latent heat of vaporization (J/mol).
    """
    # Select vaporization or sublimation without Python if — differentiable
    lam = jnp.where(jnp.asarray(t) > tfrz, hvap, hsub)
    # Convert J/kg → J/mol — Fortran line 114
    return lam * mmh2o
