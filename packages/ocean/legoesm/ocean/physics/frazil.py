"""Conservative local frazil equilibrium and optional bottom-to-top remelting.

The public interface takes and returns surface-referenced POTENTIAL temperature
in Celsius, matching the ocean tracer. Conversion to in-situ temperature wraps
the private equilibrium core, using the updated salinity on return. Sea pressure
is in Pa and is held fixed during this adjustment.

For initial liquid mass M = rho*dz and signed ice production m (negative on
melting), use M' = M-m, M'S' = MS-m*Si and
M'*cp*T' - m*Lf = M*cp*T. Thus liquid enthalpy is cp*T_C and the transported
ice has constant specific enthalpy -Lf relative to liquid water at zero Celsius.
Ice sensible heat, pressure work and density changes are omitted. This is an
explicit constant-enthalpy ice approximation, not a full ice thermodynamics
model; the recipient must use that same enthalpy and the configured ice salinity.
The recipient must account for exported salt as
ice_mass_per_area_kg_m2 * config.ice_salinity_psu * 1e-3 [kg salt m^-2];
a fresh-ice recipient must configure ice_salinity_psu=0 on both sides.
These budgets use in-situ cp*T_C, not cp*potential_temperature.
The returned potential temperature is the forward EOS image of the final
state. Its salinity dependence adds a haline contribution (about -5e-3 K/PSU
at depth, pressure dependent) absent from the in-situ enthalpy budget; a
driver auditing cp*theta can therefore see residuals of order 1e-3 K.

The nonlinear liquidus is solved with 16 fixed-point iterations (also on
remelting). Supported seawater regime: initial Si <= S <= 50 PSU, 0 <= Si <= 4,
sea pressure 0..1e8 Pa, supercooling <= 1.34 K, cp/Lf <= 0.02 K^-1.
Within this regime the iteration is contractive; tests require equilibrium
within 1e-10 K in float64 and 1e-4 K in float32. Warm layers with insufficient
incoming ice remain above freezing. A half-mass numerical freeze cap prevents
division by zero outside this regime; equilibrium is not promised there.

Unlike the draft, salt and liquid mass use the full denominators, without a
small-fraction approximation. The returned thickness MUST replace the input
thickness: retaining old thickness invalidates mass, salt and energy budgets.
Replace the liquid state's layer thickness with result.dz_m together with both
returned tracers, and update any dependent geometry before the next tendency.
The final axis is levels, index 0 at the surface; active positive-thickness
levels must be contiguous from the surface (the rise carry skips inactive cells).
Full-depth cell sea pressure must increase down each active column; broadcasting
a surface pressure along levels is invalid. Only horizontal broadcasting is
allowed for pressure. A driver compiled with jax.jit(checkify.checkify(step))
with user checks enabled (the default), followed by err.throw() outside JIT,
enforces value checks for
pressure range/order, contiguous activity, nonfinite state, and residual
supercooling and the EOS inverse residual. These debug checks are inert in
eager and ordinary JIT execution; this module does not enable them for a driver.
Static shapes and config checks raise at entry, including during JIT tracing.
The pressure bound catches negative, nonfinite, and >1e8 Pa values, not units:
6000 dbar mistakenly supplied as 6000 Pa passes. Units cannot be certified
without independent depth/density information; callers must supply Pa.
No driver or sea-ice coupling is performed by this module.
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import checkify

from legoesm import constants
from legoesm.ocean.eos import (VALID_FREEZE_SCHEMES, freezing_point,
                               potential_temperature, in_situ_temperature)

__physics_contract__ = {
    "summary": "Iterated in-situ frazil equilibrium with exact liquid mass, salt and cp*in_situ_T_C enthalpy accounting. Value guards are inert unless the driver is compiled with jax.jit(checkify.checkify(step)), user checks enabled, and err.throw() called outside JIT. Static shape/config checks always raise. Validity domain in module docstring.",
    "inputs": {"T_C": "surface-referenced potential degC (..., nlev), surface index 0",
               "S_psu": "PSU, broadcast to T_C", "dz_m": "m, broadcast to T_C",
               "active_mask": "bool, broadcast to T_C; nonpositive thickness is inactive; active levels contiguous from surface",
               "p_pa": "full-depth cell sea pressure Pa, explicit nlev axis, increasing downward; horizontal broadcast only; [0, 1e8] range check cannot detect dbar supplied as Pa"},
    "outputs": {"T_C": "surface-referenced potential degC (..., nlev)", "S_psu": "PSU (..., nlev)",
                "ice_mass_per_area_kg_m2": "kg m^-2 (...), exported at surface",
                "dz_m": "updated liquid thickness m (..., nlev); MUST be applied"},
    "sign_convention": "Positive ice export removes ocean mass and salt and releases latent heat; melting freshens and cools. Ice enthalpy=-Lf; liquid enthalpy=cp*in_situ_T_C. Recipient salt export=ice_mass_per_area_kg_m2*config.ice_salinity_psu*1e-3 kg salt m^-2.",
    # The schema has no enthalpy category; do not promise tracer cp*theta energy.
    "conserves": ["mass", "salt"],
    "differentiable": True,
    "reference": "Algebraic mass, salt and in-situ enthalpy balances documented above; liquidus delegated to ocean.eos.freezing_point. The exact returned potential temperature includes a pressure-dependent haline contribution (about -5e-3 K/PSU at depth), absent from the cp*T enthalpy budget; cp*theta audits can show order 1e-3 K residuals.",
    "idealized_test": "tests/ocean/unit/test_frazil.py",
}

__param_spec__ = {
    "FrazilConfig": {
        "scheme_key": "frazil",
        "params": {},
        "excluded": {
            "ice_salinity_psu": "Fixed composition convention shared with receiving sea ice.",
            "rho_sw_kg_m3": "Reference density convention, not a closure tuning parameter.",
            "c_p_sw_j_kg_k": "Specific heat convention shared with receiving sea ice.",
            "L_f_j_kg": "Latent enthalpy convention shared with receiving sea ice.",
        },
    },
}

# Numerical solver controls, not physical/tunable coefficients.
_EQUILIBRIUM_ITERATIONS = 16
# Residual-supercooling tolerance of the fixed-point equilibrium, per dtype.
_EQUILIBRIUM_TOL_K = {"float64": 1e-10, "float32": 1e-4}
# Sea-pressure sanity bound [Pa]; catches negative/non-finite/absurd values,
# not dbar supplied as Pa (module docstring).
_MAX_SEA_PRESSURE_PA = 1e8
_DBAR_PER_PA = 1e-4  # exact unit conversion


class FrazilConfig(NamedTuple):
    """Static selectors and fixed thermodynamic conventions; disabled by default."""

    enabled: bool = False
    freezing_scheme: str = "constant"
    # No canonical frazil-specific 0.5 PSU constant exists: use the repository's
    # bulk ice default, or explicitly supply the receiving model's composition.
    ice_salinity_psu: float = constants.S_ice_bulk_default
    rising_frazil: str = "remelt"
    rho_sw_kg_m3: float = constants.rho_ocean
    c_p_sw_j_kg_k: float = constants.c_sw
    L_f_j_kg: float = constants.L_f


class FrazilResult(NamedTuple):
    """Potential temperature [degC], salinity, surface ice mass, and new thickness."""

    T_C: jax.Array
    S_psu: jax.Array
    ice_mass_per_area_kg_m2: jax.Array
    dz_m: jax.Array


def apply_frazil(T_C, S_psu, dz_m, active_mask, p_pa, config):
    """Adjust the model's surface-referenced potential-temperature tracer [degC].

    Returns potential temperature at the UPDATED salinity, never in-situ T.
    Config is static under JIT. Array inputs are differentiable, with piecewise
    derivatives at phase/ice boundaries. See the module contract for pressure,
    level ordering, recipient salt/enthalpy and mandatory geometry updates.
    Use ``jax.jit(checkify.checkify(fn))`` and ``err.throw()`` outside JIT to
    enforce value contracts, including in production; user checks must remain
    enabled (the checkify default). Without this, value guards are inert in
    both eager and plain JIT execution. Static pressure shape checks
    always run. Disabled mode returns the input state without conversion.
    """
    if config.freezing_scheme not in VALID_FREEZE_SCHEMES:
        raise ValueError(f"Unknown freezing_scheme: {config.freezing_scheme!r}")
    if config.rising_frazil not in ("direct", "remelt"):
        raise ValueError(f"Unknown rising_frazil: {config.rising_frazil!r}")
    for name in ("rho_sw_kg_m3", "c_p_sw_j_kg_k", "L_f_j_kg"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(config.ice_salinity_psu) or not 0 <= config.ice_salinity_psu <= 4:
        raise ValueError("ice_salinity_psu must be finite and in [0, 4]")

    dtype = jnp.result_type(T_C, S_psu, dz_m, p_pa, jnp.float32)
    T = jnp.asarray(T_C, dtype=dtype)
    if T.ndim < 1 or T.shape[-1] == 0:
        raise ValueError("T_C must have a nonempty final level axis")
    pressure_shape = jnp.shape(p_pa)
    if not pressure_shape or pressure_shape[-1] != T.shape[-1]:
        raise ValueError("p_pa must have an explicit full-depth nlev axis")
    S, dz, p = (jnp.broadcast_to(jnp.asarray(a, dtype=dtype), T.shape)
                for a in (S_psu, dz_m, p_pa))
    requested_active = jnp.broadcast_to(jnp.asarray(active_mask, dtype=bool), T.shape)
    active = requested_active & (dz > 0)
    zero = jnp.zeros(T.shape[:-1], dtype=dtype)
    if not config.enabled:
        return FrazilResult(T, S, zero, dz)

    checkify.debug_check(jnp.all(~requested_active | jnp.isfinite(dz)),
                         "frazil: nonfinite active thickness")
    checkify.debug_check(jnp.all(~active | (jnp.isfinite(p) & (p >= 0) & (p <= _MAX_SEA_PRESSURE_PA))),
                         "frazil: invalid sea pressure")
    checkify.debug_check(jnp.all(~active[..., 1:] | active[..., :-1]),
                         "frazil: active levels must be contiguous from surface")
    checkify.debug_check(jnp.all(~(active[..., 1:] & active[..., :-1])
                                | (jnp.diff(p, axis=-1) > 0)),
                         "frazil: full-depth pressure must increase downward")
    checkify.debug_check(jnp.all(~active | (jnp.isfinite(T) & jnp.isfinite(S)
                                           & jnp.isfinite(dz))),
                         "frazil: nonfinite active state")
    p_dbar = p * _DBAR_PER_PA
    # Inactive storage may be nonfinite; the inverse checks all supplied cells.
    insitu = in_situ_temperature(jnp.where(active, S, 0),
                                 jnp.where(active, T, 0),
                                 jnp.where(active, p_dbar, 0))
    out = _apply_frazil_in_situ(insitu, S, dz, active, p, config)
    theta = potential_temperature(out.S_psu, out.T_C, p_dbar)
    # Preserve exact identity for untouched cells, avoiding conversion roundoff.
    changed = active & ((out.T_C != insitu) | (out.S_psu != S))
    return out._replace(T_C=jnp.where(changed, theta, T))


def _apply_frazil_in_situ(T, S, dz, active, p, config):
    """Private constant-enthalpy core; all arrays already broadcast/promoted."""
    dtype = T.dtype
    zero = jnp.zeros(T.shape[:-1], dtype=dtype)
    rho, cp, latent = config.rho_sw_kg_m3, config.c_p_sw_j_kg_k, config.L_f_j_kg
    si = config.ice_salinity_psu
    latent_over_cp = jnp.asarray(latent / cp, dtype=dtype)
    mass = rho * jnp.where(active, dz, jnp.ones_like(dz))

    def equilibrium_fraction(t, s, pressure, lower, upper):
        # x = m/M. Solving T'(x)=Tf(S'(x),p) by fixed-point substitution
        # preserves derivatives, unlike differentiating bisection comparisons.
        def step(_, x):
            salinity = (s - x * si) / (1 - x)
            tf = (freezing_point(salinity, pressure, scheme=config.freezing_scheme)
                  - constants.T_freeze).astype(dtype)
            updated = (tf - t) / (latent_over_cp + tf)
            return jnp.clip(updated, lower, upper)

        return jax.lax.fori_loop(0, _EQUILIBRIUM_ITERATIONS, step, jnp.zeros_like(t))

    formed_fraction = equilibrium_fraction(T, S, p, 0, 0.5)
    formed_fraction = jnp.where(active, formed_fraction, 0)
    formed = mass * formed_fraction
    liquid_mass = mass - formed
    T1 = (T + formed_fraction * latent_over_cp) / (1 - formed_fraction)
    S1 = (S - formed_fraction * si) / (1 - formed_fraction)
    tf1 = freezing_point(S1, p, scheme=config.freezing_scheme) - constants.T_freeze
    tolerance = _EQUILIBRIUM_TOL_K["float64" if dtype == jnp.float64 else "float32"]
    checkify.debug_check(jnp.all(~active | (jnp.isfinite(T1) & jnp.isfinite(S1)
                                           & jnp.isfinite(tf1) & (T1 >= tf1 - tolerance))),
                         "frazil: formation failed to remove supercooling")
    if config.rising_frazil == "direct":
        return FrazilResult(T1, S1, jnp.sum(formed, axis=-1),
                            jnp.where(active, liquid_mass / rho, dz))

    def rise(carry, cell):
        t, s, pressure, m, new_ice, wet, old_dz = cell
        x = equilibrium_fraction(t, s, pressure, -carry / m, 0)
        x = jnp.where(wet, x, 0)
        melted = -x * m
        t_new = (t + x * latent_over_cp) / (1 - x)
        s_new = (s - x * si) / (1 - x)
        dz_new = jnp.where(wet, (m + melted) / rho, old_dz)
        return jnp.maximum(carry - melted, 0) + new_ice, (t_new, s_new, dz_new)

    def front(a):
        return jnp.flip(jnp.moveaxis(a, -1, 0), axis=0)

    cells = tuple(front(a) for a in (T1, S1, p, liquid_mass, formed, active, dz))
    ice, updated = jax.lax.scan(rise, zero, cells)
    t_new, s_new, dz_new = (jnp.moveaxis(jnp.flip(a, axis=0), 0, -1) for a in updated)
    return FrazilResult(t_new, s_new, ice, dz_new)
