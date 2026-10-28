"""Plug-in point for EXPERIMENTAL turbulence closures (the co-scientist harness).

A candidate closure is NOT a production scheme: it never enters the turbulence
dispatch (``physics.turbulence.integration.get_turbulence_fn``). It is a Python
file the harness loads by path and drives through the single-column model.

Candidate contract (a module with these names)::

    KIND = "scalar"        # owns θ_l and q_t transport; momentum stays with the baseline
    # or KIND = "momentum" # owns u, v transport; scalars stay with the baseline
    # or KIND = "both"     # owns θ_l, q_t, u and v transport, incl. the surface stress
    PARAMS = {"name": (default, lo, hi, "units"), ...}   # every free parameter
    def init_aux(inputs: ColumnInputs): -> pytree       # candidate prognostic state, or ()
    def fluxes(params: dict, inputs: ColumnInputs, aux) -> (flux_a, flux_b, aux_new)

``fluxes`` returns KINEMATIC interface fluxes, shape ``(n+1,)``, surface-first,
positive upward: ``(w'θ_l', w'q_t')`` for ``KIND="scalar"``, ``(u'w', v'w')`` for
``KIND="momentum"``; for ``KIND="both"`` ``flux_a`` is the pair ``(w'θ_l', w'q_t')``
and ``flux_b`` the pair ``(u'w', v'w')``. Index 0 is the surface interface and must
equal the prescribed surface flux in ``inputs`` exactly, except the surface stress of
a ``"both"`` candidate, which the candidate computes itself (the baseline's
turbulence is computed and subtracted whole, so no surface stress is handed over and
``inputs.uw_s``/``vw_s`` are NaN for that kind); index ``n`` is the model top and must be zero
(use :func:`absorb_top` to deposit a top-reaching wave flux into the mean flow
inside the column). ``params`` values are traced scalars, so ``jax.grad`` reaches
them. The operator may be implicit in time internally (``inputs.dt`` is given):
whatever it does, it reports the flux it applied, and the harness converts flux to
tendency, so the transport is conservative by construction.

Conservation (flux form, CLAUDE.md RULE 0). With ``M_i = ρ_i F_i`` the interface
mass flux and ``m_k = Δp_k/g`` the SCM layer mass per unit area,
``dφ_k/dt = −(M_{k+1} − M_k)/m_k`` (surface-first). The column integral
``Σ m_k dφ_k/dt = M_0 − M_n`` telescopes exactly in the SCM's own mass measure,
so the change of column content equals the surface input minus the top outflow.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp

from legoesm import constants

Array = jnp.ndarray

KINDS = ("scalar", "momentum", "both")


def owns_scalars(kind: str) -> bool:
    return kind in ("scalar", "both")


def owns_momentum(kind: str) -> bool:
    return kind in ("momentum", "both")


def owned_fluxes(kind: str, flux_a, flux_b) -> dict:
    """The candidate's fluxes by field name (``wtheta``, ``wqt``, ``uw``, ``vw``),
    owned fields only."""
    if kind == "scalar":
        return {"wtheta": flux_a, "wqt": flux_b}
    if kind == "momentum":
        return {"uw": flux_a, "vw": flux_b}
    if kind == "both":
        if not (isinstance(flux_a, tuple | list) and isinstance(flux_b, tuple | list)
                and len(flux_a) == 2 and len(flux_b) == 2):
            raise ValueError('a "both" candidate returns ((wtheta, wqt), (uw, vw), aux)')
        return {"wtheta": flux_a[0], "wqt": flux_a[1], "uw": flux_b[0], "vw": flux_b[1]}
    raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
#: Heat-flux conversions the harness offers. ``theta``: the flux moves mass-weighted
#: θ_l (what every roster closure and the LES conserve; column enthalpy is NOT
#: conserved). ``enthalpy``: the flux is converted with the interface Exner function
#: into an enthalpy flux, so column c_p·T is conserved exactly and θ_l only to
#: O(ΔΠ). The choice is recorded in every output; there is no default.
HEAT_FORMS = ("theta", "enthalpy")


class ColumnInputs(NamedTuple):
    """One column, surface-first (index 0 at the ground), SI units.

    ``z_full`` (n,), ``z_half`` (n+1,) with ``z_half[0] = 0``; ``rho_full`` (n,),
    ``rho_half`` (n+1,); ``mass`` (n,) layer mass ``Δp/g`` [kg m⁻²]; ``exner`` (n,),
    ``exner_half`` (n+1,); pressures ``p_full`` (n,), ``p_half`` (n+1,) [Pa];
    profiles ``theta_l``, ``q_t``, ``u``, ``v`` (n,);
    surface kinematic fluxes (+up) ``w_theta_s`` [K m/s], ``w_qt_s`` [m/s],
    ``uw_s``, ``vw_s`` [m²/s²]; Coriolis ``f_c`` [1/s]; geostrophic wind
    ``u_geo``, ``v_geo`` (n,); step ``dt`` [s]; the actual temperature ``temp`` [K],
    vapour ``q_v`` and cloud liquid ``q_c`` [kg/kg] (n,) the column carries
    (``theta_l`` = θ − L_v q_c/(c_pd Π); ``q_t`` is the total water q_v + q_c + q_r,
    rain included as the LES records it); the surface roughness ``z0_m``, ``z0h_m``
    [m] of the LES case the column runs (NaN when the column has none); the LES
    case's wall-drag law ``sfc_drag_most``: 1 when its surface stress was
    Businger-Dyer stability-corrected at the prescribed flux's Obukhov length, 0 for
    the plain neutral log law (NaN when the column has none).
    """

    z_full: Array
    z_half: Array
    rho_full: Array
    rho_half: Array
    mass: Array
    exner: Array
    exner_half: Array
    p_full: Array
    p_half: Array
    theta_l: Array
    q_t: Array
    u: Array
    v: Array
    w_theta_s: Array
    w_qt_s: Array
    uw_s: Array
    vw_s: Array
    f_c: Array
    u_geo: Array
    v_geo: Array
    dt: Array
    temp: Array
    q_v: Array
    q_c: Array
    z0_m: Array
    z0h_m: Array
    sfc_drag_most: Array


class Candidate(NamedTuple):
    """A loaded candidate: its kind, parameter table and the two functions."""

    name: str
    kind: str
    params: dict          # {name: (default, lo, hi, units)}
    init_aux: Any
    fluxes: Any
    source_path: str

    def default_params(self) -> dict:
        return {k: jnp.asarray(float(v[0])) for k, v in self.params.items()}

    def bounds(self) -> tuple[list[str], Array, Array]:
        names = list(self.params)
        lo = jnp.asarray([float(self.params[k][1]) for k in names])
        hi = jnp.asarray([float(self.params[k][2]) for k in names])
        return names, lo, hi


def load_candidate(path: str | Path) -> Candidate:
    """Import a candidate file by path and validate its contract (raises on any gap)."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"cosci_candidate_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import candidate {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    missing = [n for n in ("KIND", "PARAMS", "init_aux", "fluxes") if not hasattr(mod, n)]
    if missing:
        raise ValueError(f"{path.name}: candidate is missing {missing}")
    if mod.KIND not in KINDS:
        raise ValueError(f"{path.name}: KIND must be one of {KINDS}, got {mod.KIND!r}")
    for k, spec_ in mod.PARAMS.items():
        if len(spec_) != 4:
            raise ValueError(f"{path.name}: PARAMS[{k!r}] must be (default, lo, hi, units)")
        d, lo, hi, _u = spec_
        if not (float(lo) < float(hi) and float(lo) <= float(d) <= float(hi)):
            raise ValueError(
                f"{path.name}: PARAMS[{k!r}] needs lo < hi and lo <= default <= hi")
    return Candidate(name=path.stem, kind=mod.KIND, params=dict(mod.PARAMS),
                     init_aux=mod.init_aux, fluxes=mod.fluxes, source_path=str(path))


# --- flux <-> tendency (the conservative conversion) --------------------------------

def flux_to_tendency(flux_kin: Array, rho_half: Array, mass: Array) -> Array:
    """Tendency ``dφ_k/dt = −(ρF_{k+1} − ρF_k)/m_k`` from kinematic interface fluxes.

    Surface-first: ``flux_kin`` (n+1,), ``mass`` (n,). Column-integral identity:
    ``Σ m_k dφ_k/dt = ρ_0 F_0 − ρ_n F_n``.
    """
    m_flux = rho_half * flux_kin
    return -(m_flux[1:] - m_flux[:-1]) / mass


def tendency_to_flux(tend: Array, rho_half: Array, mass: Array) -> Array:
    """Inverse of :func:`flux_to_tendency` with a zero top flux: the kinematic interface
    flux a flux-form scheme applied, recovered from its tendency (``M_n = 0``,
    ``M_k = M_{k+1} + m_k dφ_k/dt``). Index 0 is the surface flux the scheme implied."""
    m_from_top = jnp.cumsum((mass * tend)[::-1])[::-1]          # M_k for k=0..n-1
    m_flux = jnp.concatenate([m_from_top, jnp.zeros((1,), m_from_top.dtype)])
    return m_flux / rho_half


def absorb_top(flux_kin: Array, z_half: Array, z_base: float) -> Array:
    """Deposit a top-reaching flux into the mean flow between ``z_base`` and the top.

    Multiplies the interface flux by a taper that is 1 below ``z_base`` and falls as
    ``cos²`` to exactly 0 at the top interface. The flux divergence across the taper
    is the deposition, so it stays in flux form: column momentum changes by exactly
    the surface flux, and the absorbed amount is ``ρF`` at ``z_base`` (reported by
    :func:`absorbed_mass_flux`). ``z_base`` at or above the top is invalid: the
    result is NaN (traceable), never a silently zeroed or untapered flux.
    """
    z_top = z_half[-1]
    frac = jnp.clip((z_half - z_base) / (z_top - z_base), 0.0, 1.0)
    # cos²(π/2) is ~4e-33, not 0: the lid value is set exactly so nothing leaks out.
    out = jnp.where(frac < 1.0, flux_kin * jnp.cos(0.5 * jnp.pi * frac) ** 2, 0.0)
    return jnp.where(z_base < z_top, out, jnp.nan)


def absorbed_mass_flux(flux_kin: Array, rho_half: Array, z_half: Array, z_base: float) -> Array:
    """Mass flux ``ρF`` entering the absorbing layer at ``z_base`` (interpolated)."""
    return jnp.interp(z_base, z_half, rho_half * flux_kin)


def ke_dissipation_heating(uw: Array, vw: Array, u: Array, v: Array,
                           rho_half: Array, mass: Array) -> Array:
    """Heating rate [W kg⁻¹] equal to the kinetic energy the INTERIOR momentum fluxes
    remove, split half to each layer adjacent to an interface.

    Summation by parts of ``Σ m_k u_k du_k/dt`` with :func:`flux_to_tendency` gives
    ``Σ_{i=1}^{n-1} M_i (u_i − u_{i−1}) + u_0 M_0 − u_{n−1} M_n``. The interior sum
    is the shear term; its negative, returned as heat, makes interior KE + enthalpy
    exact. The surface term ``u_0 M_0`` (surface-stress work) is a boundary energy
    flux and is NOT heated here; :func:`column_energy_budget` reports it.
    """
    mu = rho_half * uw
    mv = rho_half * vw
    d_int = mu[1:-1] * (u[1:] - u[:-1]) + mv[1:-1] * (v[1:] - v[:-1])   # (n-1,) W m^-2
    heat = jnp.zeros_like(mass)
    heat = heat.at[1:].add(-0.5 * d_int)
    heat = heat.at[:-1].add(-0.5 * d_int)
    return heat / mass


def heat_flux_to_theta_tendency(w_theta: Array, inputs: ColumnInputs, heat_form: str) -> Array:
    """θ_l tendency from the candidate heat flux under the chosen ``heat_form``."""
    if heat_form == "theta":
        return flux_to_tendency(w_theta, inputs.rho_half, inputs.mass)
    if heat_form == "enthalpy":
        d_temp = flux_to_tendency(w_theta * inputs.exner_half, inputs.rho_half, inputs.mass)
        return d_temp / inputs.exner
    raise ValueError(f"heat_form must be one of {HEAT_FORMS}, got {heat_form!r}")


def step_midpoint_winds(du: Array, dv: Array, inputs: ColumnInputs) -> tuple[Array, Array]:
    """``u + ½ dt du``: with it, ``Σ m u_mid du dt`` is the EXACT kinetic-energy change
    of a forward-Euler step of the candidate's own momentum tendency."""
    half = 0.5 * inputs.dt
    return inputs.u + half * du, inputs.v + half * dv


def _total_accel(du: Array, dv: Array, other_accel) -> tuple[Array, Array]:
    if other_accel is None:
        return du, dv
    return du + other_accel[0], dv + other_accel[1]


class CandidateTendencies(NamedTuple):
    """Candidate tendencies on the column (surface-first): θ_l [K/s] (incl. any
    dissipation heating), q_t [1/s], u, v [m/s²]. Non-owned entries are ``None``."""

    dtheta: Array | None
    dqt: Array | None
    du: Array | None
    dv: Array | None


def candidate_tendencies(kind: str, flux_a: Array, flux_b: Array, inputs: ColumnInputs,
                         *, heat_form: str, ke_heating: bool,
                         other_accel: tuple[Array, Array] | None = None
                         ) -> CandidateTendencies:
    """Convert a candidate's interface fluxes into column tendencies.

    ``other_accel``: the (du, dv) every OTHER process adds to the winds in the same
    forward-Euler step (``None`` = the candidate stepped alone). Dissipation heating
    is evaluated at the full step's midpoint winds so the stepped energy budget
    closes exactly."""
    fl = owned_fluxes(kind, flux_a, flux_b)
    dtheta = dqt = du = dv = None
    if owns_scalars(kind):
        dtheta = heat_flux_to_theta_tendency(fl["wtheta"], inputs, heat_form)
        dqt = flux_to_tendency(fl["wqt"], inputs.rho_half, inputs.mass)
    if owns_momentum(kind):
        du = flux_to_tendency(fl["uw"], inputs.rho_half, inputs.mass)
        dv = flux_to_tendency(fl["vw"], inputs.rho_half, inputs.mass)
        if ke_heating:
            u_mid, v_mid = step_midpoint_winds(*_total_accel(du, dv, other_accel), inputs)
            heat = ke_dissipation_heating(fl["uw"], fl["vw"], u_mid, v_mid,
                                          inputs.rho_half, inputs.mass)
            d_heat = heat / (constants.c_pd * inputs.exner)
            dtheta = d_heat if dtheta is None else dtheta + d_heat
    return CandidateTendencies(dtheta=dtheta, dqt=dqt, du=du, dv=dv)


class EnergyBudget(NamedTuple):
    """Column total-energy rate budget [W m⁻²]: ``d/dt Σ m (c_p T + L_v q_t + ½|u|²)``
    (the candidate's own tendencies; latent term only for a candidate that owns scalars)
    vs the boundary inputs (surface enthalpy flux + surface-stress work − top
    fluxes); ``residual`` = rate − inputs."""

    rate: Array
    boundary_input: Array
    residual: Array


def column_energy_budget(tend: CandidateTendencies, flux_a: Array, flux_b: Array,
                         inputs: ColumnInputs, kind: str, *,
                         other_accel: tuple[Array, Array] | None = None) -> EnergyBudget:
    """The candidate's share of the total-energy budget of one forward-Euler step
    (momentum: its KE work and surface/top work at the full step's midpoint winds;
    ``other_accel`` as in :func:`candidate_tendencies`)."""
    m = inputs.mass
    fl = owned_fluxes(kind, flux_a, flux_b)
    d_temp = (jnp.zeros_like(m) if tend.dtheta is None else tend.dtheta * inputs.exner)
    rate = jnp.sum(m * constants.c_pd * d_temp)
    boundary = jnp.asarray(0.0, m.dtype)
    if owns_scalars(kind):
        mh = inputs.rho_half * fl["wtheta"] * inputs.exner_half * constants.c_pd
        # latent-ok: column moist-enthalpy diagnostic at the constant L_v
        mq = inputs.rho_half * fl["wqt"] * constants.L_v  # latent-ok: see above
        rate = rate + jnp.sum(m * constants.L_v * tend.dqt)  # latent-ok: see above
        boundary = boundary + mh[0] - mh[-1] + mq[0] - mq[-1]
    if owns_momentum(kind):
        u_mid, v_mid = step_midpoint_winds(*_total_accel(tend.du, tend.dv, other_accel),
                                           inputs)
        rate = rate + jnp.sum(m * (u_mid * tend.du + v_mid * tend.dv))
        mu = inputs.rho_half * fl["uw"]
        mv = inputs.rho_half * fl["vw"]
        boundary = boundary + u_mid[0] * mu[0] + v_mid[0] * mv[0] \
            - u_mid[-1] * mu[-1] - v_mid[-1] * mv[-1]
    return EnergyBudget(rate=rate, boundary_input=boundary, residual=rate - boundary)


# --- column inputs ----------------------------------------------------------------

def column_inputs(*, z_full, z_half, rho_full, rho_half, mass, exner, exner_half,
                  p_full, p_half, theta_l, q_t, u, v, w_theta_s, w_qt_s, uw_s, vw_s,
                  f_c=0.0, u_geo=None, v_geo=None, dt, temp=None, q_v=None,
                  q_c=None, z0_m=None, z0h_m=None,
                  sfc_drag_most=None) -> ColumnInputs:
    """Assemble :class:`ColumnInputs` (surface-first arrays; scalars broadcast).

    ``temp``/``q_v``/``q_c`` given together, or all omitted for a cloud-free column
    (``q_c = 0``, ``q_v = q_t``, ``temp = θ_l Π``)."""
    given = [x is not None for x in (temp, q_v, q_c)]
    if any(given) and not all(given):
        raise ValueError("give temp, q_v and q_c together (or none: cloud-free)")
    dtype = jnp.asarray(theta_l).dtype

    def _s(x):
        return jnp.asarray(x, dtype)

    zeros = jnp.zeros_like(jnp.asarray(theta_l))
    return ColumnInputs(
        z_full=_s(z_full), z_half=_s(z_half), rho_full=_s(rho_full),
        rho_half=_s(rho_half), mass=_s(mass), exner=_s(exner),
        exner_half=_s(exner_half), p_full=_s(p_full), p_half=_s(p_half),
        theta_l=_s(theta_l), q_t=_s(q_t), u=_s(u), v=_s(v),
        w_theta_s=_s(w_theta_s), w_qt_s=_s(w_qt_s), uw_s=_s(uw_s), vw_s=_s(vw_s),
        f_c=_s(f_c), u_geo=zeros if u_geo is None else _s(u_geo),
        v_geo=zeros if v_geo is None else _s(v_geo), dt=_s(dt),
        temp=_s(theta_l) * _s(exner) if temp is None else _s(temp),
        q_v=_s(q_t) if q_v is None else _s(q_v),
        q_c=zeros if q_c is None else _s(q_c),
        z0_m=_s(jnp.nan if z0_m is None else z0_m),
        z0h_m=_s(jnp.nan if z0h_m is None else z0h_m),
        sfc_drag_most=_s(jnp.nan if sfc_drag_most is None else sfc_drag_most))


def half_level_average(full: Array, surface_value: Array) -> Array:
    """Interfaces (n+1,) from full levels (n,): mean of the two neighbours inside,
    ``surface_value`` at index 0, the top full level copied to index n."""
    full = jnp.asarray(full)
    inner = 0.5 * (full[1:] + full[:-1])
    return jnp.concatenate([jnp.asarray(surface_value, full.dtype)[None], inner, full[-1:]])
