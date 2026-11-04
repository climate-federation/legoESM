"""Evaluation stages for experimental closures (the co-scientist experimenter).

E0 static checks, L idealized limits, E1 a-priori fluxes, E2 a-posteriori SCM runs
with AD tuning. Every stage drives the SAME single-column model the roster is scored
in (:mod:`~legoesm.atmosphere.les_suite.scm_runner`); a candidate replaces only the
part it owns (scalars, momentum, or both), and any other part is a named baseline
closure. A ``"both"`` candidate replaces the whole baseline turbulence tendency and
computes its own surface stress; the baseline then only sets up the column.

How a candidate enters the column (:func:`rollout`): the SCM step calls a substitute
physics function that returns ``baseline_total − baseline_turbulence[owned] +
candidate[owned]``. Combined physics is a parallel split (every process sees the same
input state, tendencies summed), so that subtraction removes the baseline closure's
owned part exactly. The candidate gets the surface heat/moisture flux the forcing
injected for this stage; a scalar or momentum candidate also gets the surface stress
the baseline surface layer applied, so both arms of such a comparison see the same
surface boundary condition. A ``"both"`` candidate gets NaN for the stress and
applies its own, so its surface stress can differ from any baseline's.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from legoesm.atmosphere.forcing.scm.scm_forcing import compute_forcing_tendencies
from legoesm.atmosphere.physics import MicrophysicsConfig
from legoesm.atmosphere.physics._shared import (
    compute_heights_from_sigma,
    compute_rho,
    exner_function,
)
from legoesm.atmosphere.physics.combined import make_physics
from legoesm.core.profile_metrics import safe_sqrt

from legoesm import constants

from .bridge import LESReferenceArtifact, total_turbulent_flux
from .candidate import (
    Candidate,
    ColumnInputs,
    candidate_tendencies,
    column_energy_budget,
    column_inputs,
    half_level_average,
    ke_dissipation_heating,
    owned_fluxes,
    owns_momentum,
    owns_scalars,
    step_midpoint_winds,
    tendency_to_flux,
)
from .scm_coupling import interp_profile, liquid_water_theta
from .score import (
    PROFILE_FLOOR_FLUX,
    PROFILE_FLOOR_QFLUX,
    PROFILE_FLOOR_QT,
    PROFILE_FLOOR_THETA,
    PROFILE_FLOOR_WIND,
    case_score_weights,
    normalized_rmse_series,
)

Array = jnp.ndarray

#: Adam budget shared by candidates and baselines (the existing LES-suite AD tuner's
#: CLI defaults, ``scripts/run/tune_scm_to_les.py --ad-steps/--ad-lr``).
AD_STEPS = 20
AD_LR = 0.08
#: SCM column used by the LES-suite tuner CLI (``tune_scm_to_les.py --nlev/--dt``).
SCM_NLEV = 32
SCM_DT_S = 10.0


# --- column inputs from the SCM state ---------------------------------------------

def _col(field) -> Array:
    """``(1,1,1,n)`` top-first SCM field -> ``(n,)`` surface-first."""
    return jnp.asarray(field)[0, 0, 0][::-1]


def scm_column_inputs(state, sigma_coord, phys_state, *, f_c, u_geo, v_geo, dt,
                      uw_s, vw_s, z0_m=None, z0h_m=None,
                      sfc_drag_most=None) -> ColumnInputs:
    """:class:`ColumnInputs` for the SCM column, built with the SAME height/density
    helpers the roster's turbulence wrapper uses (moist hypsometric heights, moist
    density). Surface heat/moisture fluxes are the kinematic overrides the forcing
    injected into ``phys_state`` for this stage; ``uw_s``/``vw_s`` are passed in."""
    temp = jnp.asarray(state.T.data)[0, 0, 0]                 # top-first
    p_s = jnp.asarray(state.p_s.data).reshape(())
    p_full = sigma_coord.pressure_at_full(p_s)
    p_half = sigma_coord.pressure_at_half(p_s)
    tr = state.tracers or {}
    q_v = jnp.asarray(tr["q_v"].data)[0, 0, 0] if "q_v" in tr else jnp.zeros_like(temp)
    q_c = jnp.asarray(tr["q_c"].data)[0, 0, 0] if "q_c" in tr else jnp.zeros_like(temp)
    q_r = jnp.asarray(tr["q_r"].data)[0, 0, 0] if "q_r" in tr else jnp.zeros_like(temp)
    z_full, z_half = compute_heights_from_sigma(temp[None], p_half[None], q_v=q_v[None])
    rho = compute_rho(temp[None], p_full[None], q_v=q_v[None])[0]
    exner = exner_function(p_full)
    theta_l = liquid_water_theta(temp / exner, q_c, exner)
    mass = (p_half[1:] - p_half[:-1]) / constants.g
    w_th = getattr(phys_state, "surface_wth_override", None)
    w_qv = getattr(phys_state, "surface_wqv_override", None)
    zero = jnp.zeros((), temp.dtype)
    rho_sf = rho[::-1]
    return column_inputs(
        z_full=z_full[0][::-1], z_half=z_half[0][::-1], rho_full=rho_sf,
        rho_half=half_level_average(rho_sf, rho_sf[0]), mass=mass[::-1],
        exner=exner[::-1], exner_half=exner_function(p_half)[::-1],
        p_full=p_full[::-1], p_half=p_half[::-1],
        theta_l=theta_l[::-1], q_t=(q_v + q_c + q_r)[::-1],
        u=_col(state.u.data), v=_col(state.v.data),
        w_theta_s=zero if w_th is None else jnp.reshape(w_th, ()),
        w_qt_s=zero if w_qv is None else jnp.reshape(w_qv, ()),
        uw_s=uw_s, vw_s=vw_s, f_c=f_c,
        u_geo=u_geo, v_geo=v_geo,
        dt=dt, temp=temp[::-1], q_v=q_v[::-1], q_c=q_c[::-1], z0_m=z0_m, z0h_m=z0h_m,
        sfc_drag_most=sfc_drag_most)


class Fluxes(NamedTuple):
    """Kinematic interface fluxes, surface-first ``(n+1,)``, positive up."""

    wtheta: Array
    wqt: Array
    uw: Array
    vw: Array


def implied_fluxes(tend, inputs: ColumnInputs) -> Fluxes:
    """The interface fluxes a flux-form closure applied, from its SCM tendencies
    (``dθ = dT/Π``; moisture from the ``q_v`` tendency)."""
    d_temp = _col(tend.dT_dt.data)
    dq = (_col(tend.tracer_tendencies["q_v"].data)
          if tend.tracer_tendencies and "q_v" in tend.tracer_tendencies
          else jnp.zeros_like(d_temp))
    rh, m = inputs.rho_half, inputs.mass
    return Fluxes(
        wtheta=tendency_to_flux(d_temp / inputs.exner, rh, m),
        wqt=tendency_to_flux(dq, rh, m),
        uw=tendency_to_flux(_col(tend.du_dt.data), rh, m),
        vw=tendency_to_flux(_col(tend.dv_dt.data), rh, m))


def _swap(field, full_new: Array):
    return field.replace(data=full_new[::-1].reshape(field.data.shape))


def _owned_tendencies(tend_full, tend_turb, cand, inputs: ColumnInputs, kind: str):
    """Replace the baseline closure's owned part by the candidate's tendencies."""
    out = tend_full
    if owns_scalars(kind):
        d_temp = (_col(tend_full.dT_dt.data) - _col(tend_turb.dT_dt.data)
                  + cand.dtheta * inputs.exner)
        tr = dict(tend_full.tracer_tendencies or {})
        if "q_v" in tr:
            q_full = _col(tr["q_v"].data)
            q_turb = _col(tend_turb.tracer_tendencies["q_v"].data)
            tr["q_v"] = _swap(tr["q_v"], q_full - q_turb + cand.dqt)
        out = out._replace(dT_dt=_swap(tend_full.dT_dt, d_temp), tracer_tendencies=tr)
    elif cand.dtheta is not None:  # KE-dissipation heating of a momentum candidate
        d_temp = _col(tend_full.dT_dt.data) + cand.dtheta * inputs.exner
        out = out._replace(dT_dt=_swap(tend_full.dT_dt, d_temp))
    if owns_momentum(kind):
        du = _col(tend_full.du_dt.data) - _col(tend_turb.du_dt.data) + cand.du
        dv = _col(tend_full.dv_dt.data) - _col(tend_turb.dv_dt.data) + cand.dv
        out = out._replace(du_dt=_swap(tend_full.du_dt, du),
                           dv_dt=_swap(tend_full.dv_dt, dv))
    return out


#: Roster closures with a prognostic turbulence carry (TKE / moments) that the
#: baseline updates from its OWN fluxes; refused as the other half of a candidate.
PROGNOSTIC_TURBULENCE = ("tke", "mynn25", "clubb_lite", "clubb", "edmf")


class RolloutRecord(NamedTuple):
    """``inputs``/``fluxes``: the column the closure saw at each step and the fluxes
    applied in that step (leading axis = step, surface-first); ``final_inputs``: the
    column after the last step; ``other_accel``: per step, the (du, dv) of everything
    but a momentum candidate with dissipation heating (else ``None``); ``budget``:
    per step, the column budget terms of :func:`step_budget` when the column runs a
    surface law (else ``None``)."""

    inputs: ColumnInputs
    fluxes: Fluxes
    final_inputs: ColumnInputs
    other_accel: tuple | None = None
    budget: dict | None = None

    def at(self, step: int) -> ColumnInputs:
        """The column state after ``step`` steps (``step == nsteps`` → final)."""
        n = self.fluxes.wtheta.shape[0]
        if step == n:
            return self.final_inputs
        return jax.tree_util.tree_map(lambda a: a[step], self.inputs)

    def fluxes_of_step(self, step: int) -> Fluxes:
        return jax.tree_util.tree_map(lambda a: a[step], self.fluxes)


def _with_surface(kind: str, flux_a, flux_b, inputs: ColumnInputs):
    """Owned fluxes with their surface value set to the column's surface law."""
    def s0(f, v):
        return jnp.asarray(f).at[0].set(v)
    if kind == "scalar":
        return s0(flux_a, inputs.w_theta_s), s0(flux_b, inputs.w_qt_s)
    if kind == "momentum":
        return s0(flux_a, inputs.uw_s), s0(flux_b, inputs.vw_s)
    return ((s0(flux_a[0], inputs.w_theta_s), s0(flux_a[1], inputs.w_qt_s)),
            (s0(flux_b[0], inputs.uw_s), s0(flux_b[1], inputs.vw_s)))


#: Quantities of :func:`step_budget`: column water (q_v + q_c + q_r), moist energy
#: ``c_pd T + L_v q_v + ½|u|²`` and the two momentum components.
BUDGET_QUANTITIES = ("water", "energy", "u", "v")


def _tend_arrays(t, like):
    """Surface-first tendency columns of ``t`` (zeros for an absent tracer)."""
    tr = t.tracer_tendencies or {}
    out = {"T": _col(t.dT_dt.data), "u": _col(t.du_dt.data), "v": _col(t.dv_dt.data)}
    for k in ("q_v", "q_c", "q_r"):
        out[k] = _col(tr[k].data) if k in tr else jnp.zeros_like(like)
    return out


#: The processes :func:`step_budget` splits a step into.
BUDGET_PROCESSES = ("turbulence", "microphysics", "forcing")


def step_budget(out, tend_full, tend_turb, ftend, inputs: ColumnInputs, record) -> dict:
    """One forward-Euler step's column budget, per unit area per second.

    The step's tendency is split by process: ``turbulence`` (what the harness
    applied for the closure: ``out − (tend_full − tend_turb)``), ``microphysics``
    (the rest of the physics, ``tend_full − tend_turb``) and ``forcing`` (the SCM
    forcing ``ftend``). For each process and quantity (water q_v + q_c + q_r, moist
    energy c_pd T + L_v q_v + ½|u|², u, v): ``<process>_<q>`` the mass-weighted
    column integral of its tendency (kinetic energy at the step's midpoint winds,
    so the total is the exact change over the step) and ``gross_<process>_<q>`` the
    integral of its absolute terms (the round-off scale). ``rate_<q>`` is the total.
    Declared supply: ``sfc_<q>`` the surface law at the surface interface (enthalpy
    ``ρ_s c_pd Π_s w'θ_l'``, latent ``L_v ρ_s w'q_t'``, stress ``ρ_s u'w'`` and its
    work), ``precip`` the rain reaching the ground. ``record_rho_rel`` /
    ``record_exner_rel``: how far the record's ``rho_s``/``exner_s`` are from the
    column's surface interface values, which the closures and this budget use.
    No turbulent flux leaves through the column top (the turbulence share is held
    to the surface law alone); the forcing's own column change is its declared
    supply."""
    m, rh, ex = inputs.mass, inputs.rho_half[0], inputs.exner_half[0]
    full, turb_t = _tend_arrays(tend_full, m), _tend_arrays(tend_turb, m)
    applied, forc = _tend_arrays(out, m), _tend_arrays(ftend, m)
    parts = {
        "turbulence": {k: applied[k] - full[k] + turb_t[k] for k in full},
        "microphysics": {k: full[k] - turb_t[k] for k in full},
        "forcing": forc,
    }
    # a process share that is a difference of tendencies carries the round-off of
    # its operands, so its scale is the operands' magnitude, not the difference's
    mag = {
        "turbulence": {k: jnp.abs(applied[k]) + jnp.abs(full[k]) + jnp.abs(turb_t[k])
                       for k in full},
        "microphysics": {k: jnp.abs(full[k]) + jnp.abs(turb_t[k]) for k in full},
        "forcing": {k: jnp.abs(v) for k, v in forc.items()},
    }
    du = sum(p["u"] for p in parts.values())
    dv = sum(p["v"] for p in parts.values())
    u_mid, v_mid = step_midpoint_winds(du, dv, inputs)
    lv, cp = constants.L_v, constants.c_pd  # latent-ok: column moist-energy budget at the constant L_v
    res = {}
    for name, p in parts.items():
        g = mag[name]
        res[name + "_water"] = jnp.sum(m * (p["q_v"] + p["q_c"] + p["q_r"]))
        res["gross_" + name + "_water"] = jnp.sum(m * (g["q_v"] + g["q_c"] + g["q_r"]))
        res[name + "_energy"] = jnp.sum(
            m * (cp * p["T"] + lv * p["q_v"] + u_mid * p["u"] + v_mid * p["v"]))
        res["gross_" + name + "_energy"] = jnp.sum(
            m * (cp * g["T"] + lv * g["q_v"] + jnp.abs(u_mid) * g["u"]
                 + jnp.abs(v_mid) * g["v"]))
        for c in ("u", "v"):
            res[name + "_" + c] = jnp.sum(m * p[c])
            res["gross_" + name + "_" + c] = jnp.sum(m * g[c])
    for q in BUDGET_QUANTITIES:
        res["rate_" + q] = sum(res[n + "_" + q] for n in BUDGET_PROCESSES)
    precip = tend_full.precip
    res["precip"] = (jnp.reshape(jnp.asarray(precip.data), ()) if precip is not None
                     else jnp.zeros((), m.dtype))
    res["sfc_water"] = rh * inputs.w_qt_s
    res["sfc_energy"] = (rh * (cp * ex * inputs.w_theta_s + lv * inputs.w_qt_s)
                         + rh * (u_mid[0] * inputs.uw_s + v_mid[0] * inputs.vw_s))
    res["sfc_u"] = rh * inputs.uw_s
    res["sfc_v"] = rh * inputs.vw_s
    res["record_rho_rel"] = jnp.abs(record.rho_s / rh - 1.0)
    res["record_exner_rel"] = jnp.abs(record.exner_s / ex - 1.0)
    return res


def rollout(scm, physics_config, nsteps: int, *, candidate: Candidate | None = None,
            params: dict | None = None, heat_form: str | None = None,
            ke_heating: bool | None = None, f_c: float = 0.0, u_geo=None, v_geo=None,
            sfc_drag_most: bool | None = None, energy_form_baseline: bool = False):
    """Free-run ``scm`` for ``nsteps`` with ``candidate`` owning its part (or the pure
    baseline when ``candidate is None``). Returns ``(RolloutRecord, aux_final)``.

    The baseline closure is ``physics_config.turbulence``; ``scm`` must have been
    built with the same config (its physics is the baseline total). Forward Euler
    only (one physics call per step, so the candidate sees the same stage state).
    ``sfc_drag_most`` (the LES case's wall-drag law) reaches the candidate as
    ``ColumnInputs.sfc_drag_most`` (NaN when not given).

    A column with a surface law (``SCMForcing.surface_flux_fn``, the CloudBench
    columns) gives every closure the law's heat, moisture and momentum fluxes: they
    are the ``ColumnInputs`` surface values for every kind, and the surface value of
    every flux the harness applies (a candidate's, or a converted baseline's) is set
    to them; a native baseline applies them through its own surface override. Such
    a run also records :func:`step_budget` per step, with each closure's own surface
    flux before the harness set it (``own_sfc_*``).

    ``energy_form_baseline`` (baseline only): the baseline's turbulence is applied
    through the candidate conversion as a ``"both"`` closure in enthalpy form with
    dissipation heating, its interface fluxes being those its tendencies imply, so
    it conserves the column moist energy (PI 2026-11-03). Its prognostic turbulence
    state, if any, still evolves from its own fluxes.
    """
    if scm.time_integrator != "forward_euler":
        raise ValueError("rollout supports forward_euler only (one stage per step)")
    if energy_form_baseline and candidate is not None:
        raise ValueError("energy_form_baseline converts the baseline; a candidate "
                         "brings its own heat_form")
    if candidate is not None:
        if params is None:
            raise ValueError("a candidate rollout needs params")
        if owns_scalars(candidate.kind) and heat_form is None:
            raise ValueError("a scalar-owning candidate rollout needs an explicit heat_form")
        if owns_momentum(candidate.kind) and ke_heating is None:
            raise ValueError("a momentum-owning candidate rollout needs an explicit "
                             "ke_heating")
        if physics_config.turbulence.scheme in PROGNOSTIC_TURBULENCE:
            raise ValueError(
                f"baseline {physics_config.turbulence.scheme!r} carries prognostic "
                "turbulence state built from its own fluxes; swapping part of those fluxes "
                "for a candidate's has no defined ownership of that state. Use a "
                "diagnostic baseline.")
    turb_only = physics_config._replace(microphysics=MicrophysicsConfig(scheme="none"))
    physics_full = scm.physics_fn
    physics_turb = (physics_full if physics_config.microphysics.scheme == "none"
                    else make_physics(turb_only, model_type="hydrostatic", dt=scm.dt))
    sigma = scm.sigma_coord
    dt = scm.dt
    kind = None if candidate is None else candidate.kind
    conv_kind = "both" if energy_form_baseline else kind
    conv_heat = "enthalpy" if energy_form_baseline else heat_form
    conv_ke = True if energy_form_baseline else ke_heating

    forcing = scm.forcing_arg
    record_fn = None if forcing is None else forcing.surface_flux_fn

    def _geo(t):
        """The geostrophic wind the SCM forcing applies at time ``t`` (surface-first;
        time-varying for e.g. GABLS3), else the ``u_geo``/``v_geo`` given."""
        if forcing is None or forcing.u_geo is None:
            return u_geo, v_geo
        vg = (None if forcing.v_geo is None
              else jnp.asarray(forcing.v_geo(t))[::-1])
        return jnp.asarray(forcing.u_geo(t))[::-1], vg

    # the roughness the baseline surface layer runs with (the LES case's)
    _surf = getattr(physics_config.turbulence, physics_config.turbulence.scheme).surface
    z0_m, z0h_m = _surf.z0, _surf.z0 * _surf.z0h_z0_ratio

    def _inputs(state, phys_state, tend_turb, t):
        zero = jnp.zeros((), jnp.asarray(state.T.data).dtype)
        ug, vg = _geo(t)
        base = scm_column_inputs(state, sigma, phys_state, f_c=f_c, u_geo=ug,
                                 v_geo=vg, dt=dt, uw_s=zero, vw_s=zero, z0_m=z0_m,
                                 z0h_m=z0h_m, sfc_drag_most=sfc_drag_most)
        imp = implied_fluxes(tend_turb, base)
        if record_fn is not None:  # the surface law gives every closure its fluxes
            rec = record_fn(state, sigma, t)
            return base._replace(w_theta_s=rec.w_theta_l, w_qt_s=rec.w_q_t,
                                 uw_s=rec.uw, vw_s=rec.vw), imp
        if kind == "both":  # the candidate computes its own surface stress
            nan = jnp.full((), jnp.nan, zero.dtype)
            return base._replace(uw_s=nan, vw_s=nan), imp
        return base._replace(uw_s=imp.uw[0], vw_s=imp.vw[0]), imp

    def _other_accel(tend_full, tend_turb, st, sc, t):
        """Wind tendencies of everything but the candidate in this step: the
        baseline's non-turbulent physics plus the SCM forcing the step adds."""
        du = _col(tend_full.du_dt.data) - _col(tend_turb.du_dt.data)
        dv = _col(tend_full.dv_dt.data) - _col(tend_turb.dv_dt.data)
        if forcing is not None:
            ftend = compute_forcing_tendencies(st, sc, forcing, t)
            du = du + _col(ftend.du_dt.data)
            dv = dv + _col(ftend.dv_dt.data)
        return du, dv

    def _turb_tend(st, grid, sc, phys_state, tend_full):
        return (tend_full if physics_turb is physics_full else
                physics_turb(st, grid, sc, phys_state=phys_state)[0])

    aux0 = None
    if candidate is not None:
        # One throw-away baseline step so init_aux sees the forcing-prepared column
        # (prescribed surface fluxes injected) that the first closure call sees.
        probe = {}

        def probe_fn(st, grid, sc, phys_state=None):
            tend_full, phys_out = physics_full(st, grid, sc, phys_state=phys_state)
            probe["inputs"], _ = _inputs(st, phys_state,
                                         _turb_tend(st, grid, sc, phys_state, tend_full),
                                         scm.t_seconds)
            return tend_full, phys_out

        scm.pure_step_with_physics(scm.state, scm.phys_state, scm.t_seconds, probe_fn)
        aux0 = candidate.init_aux(probe["inputs"])

    def body(carry, _):
        state, phys, t, aux = carry
        stash = {}

        def physics_fn(st, grid, sc, phys_state=None):
            if stash:
                raise RuntimeError("rollout: physics called twice in one step")
            tend_full, phys_out = physics_full(st, grid, sc, phys_state=phys_state)
            tend_turb = _turb_tend(st, grid, sc, phys_state, tend_full)
            inputs, imp = _inputs(st, phys_state, tend_turb, t)
            stash["inputs"] = inputs
            stash["other"] = None
            stash["budget"] = None

            def _done(out, own):
                if record_fn is not None:
                    ftend = compute_forcing_tendencies(st, sc, forcing, t)
                    stash["budget"] = step_budget(out, tend_full, tend_turb, ftend, inputs,
                                                  record_fn(st, sc, t))
                    # the surface value of each flux the closure itself produced,
                    # before the harness set it to the surface law (diagnostic)
                    for k, f in own.items():
                        stash["budget"]["own_sfc_" + k] = jnp.asarray(f)[0]
                return out, phys_out

            if conv_kind is None:
                stash["fluxes"], stash["aux"] = imp, aux
                return _done(tend_full, imp._asdict())
            if candidate is None:
                fa, fb, aux_new = (imp.wtheta, imp.wqt), (imp.uw, imp.vw), aux
            else:
                fa, fb, aux_new = candidate.fluxes(params, inputs, aux)
            own = {**imp._asdict(), **owned_fluxes(conv_kind, fa, fb)}
            if record_fn is not None:
                fa, fb = _with_surface(conv_kind, fa, fb, inputs)
            other = (_other_accel(tend_full, tend_turb, st, sc, t)
                     if owns_momentum(conv_kind) and conv_ke else None)
            ct = candidate_tendencies(conv_kind, fa, fb, inputs, heat_form=conv_heat,
                                      ke_heating=conv_ke, other_accel=other)
            stash["other"] = other
            stash["fluxes"] = imp._replace(**owned_fluxes(conv_kind, fa, fb))
            stash["aux"] = aux_new
            return _done(_owned_tendencies(tend_full, tend_turb, ct, inputs, conv_kind), own)

        new_state, new_phys = scm.pure_step_with_physics(state, phys, t, physics_fn)
        new_phys = new_phys if new_phys is not None else phys
        return ((new_state, new_phys, t + dt, stash["aux"]),
                (stash["inputs"], stash["fluxes"], stash["other"], stash["budget"]))

    (fstate, fphys, t_f, aux_f), (inputs_rec, flux_rec, other_rec, budget_rec) = lax.scan(
        body, (scm.state, scm.phys_state, scm.t_seconds, aux0), None, length=nsteps)
    tend_f, _ = physics_turb(fstate, scm.grid, sigma, phys_state=fphys)
    final_inputs, _ = _inputs(fstate, fphys, tend_f, t_f)
    return RolloutRecord(inputs=inputs_rec, fluxes=flux_rec, final_inputs=final_inputs,
                         other_accel=other_rec, budget=budget_rec), aux_f


# --- E2 scoring --------------------------------------------------------------------

#: Variables E2 scores, each weighted equally (user decision 2026-10-15).
E2_PROFILE_VARS = ("theta", "qt", "u", "v")
E2_FLUX_VARS = ("wtheta", "wqt", "uw", "vw")


def e2_nsteps(artifact: LESReferenceArtifact, dt: float) -> int:
    """Rollout length E2 needs: one step past the last frame, so the flux at every
    frame (incl. the last) is the closure's flux evaluated AT that frame's state."""
    return _frames(artifact, dt)[-1] + 1


def _frames(artifact: LESReferenceArtifact, dt: float) -> list[int]:
    """SCM step of each LES frame after t=0: the NEAREST step (offset <= dt/2; user
    decision 2026-10-15). Distinct frames must land on distinct steps."""
    times = [float(t) for t in np.asarray(artifact.times_s)]  # host-side: static under jit
    steps = [int(round(t / dt)) for t in times[1:]]
    if not steps:
        raise ValueError(f"{artifact.case_name}: no output frame after t=0 to score")
    if min(steps) < 1:
        raise ValueError(f"{artifact.case_name}: a frame after t=0 rounds to step 0 "
                         f"(the initial column) at dt={dt}")
    if len(set(steps)) != len(steps):
        raise ValueError(f"{artifact.case_name}: frames closer than dt={dt} collide on "
                         "one SCM step")
    return steps


#: Momentum-flux normalisation floor as a fraction of the case's LES u*^2
#: (user decision 2026-10-15).
MOMFLUX_FLOOR_USTAR2_FRAC = 0.05


def momentum_flux_floor(artifact: LESReferenceArtifact) -> float | None:
    """Floor for scoring u'w', v'w' in ``artifact``, or ``None`` = not scored.

    Momentum fluxes are scored only in a case with a geostrophic wind (``u_geo`` /
    ``v_geo``, or their time integrals ``u_geo_cum`` / ``v_geo_cum``, present and not
    all zero); the floor is ``MOMFLUX_FLOOR_USTAR2_FRAC``
    times the case's LES ``u*^2`` = mean over the frames after t=0 of the surface
    stress magnitude ``|(tau_x_s, tau_y_s)|``. Such a case without LES momentum
    fluxes or surface stress is an error (user decision 2026-10-15)."""
    geo = [np.asarray(g) for g in (artifact.u_geo, artifact.v_geo, artifact.u_geo_cum,
                                   artifact.v_geo_cum) if g is not None]
    if not any(bool(np.any(g != 0.0)) for g in geo):
        return None
    missing = [n for n in ("uw_resolved", "vw_resolved", "tau_x_s", "tau_y_s")
               if getattr(artifact, n) is None]
    if missing:
        raise ValueError(f"{artifact.case_name}: geostrophic-wind case without LES "
                         f"momentum data {missing}; its momentum fluxes must be scored")
    tau = np.hypot(np.asarray(artifact.tau_x_s)[1:], np.asarray(artifact.tau_y_s)[1:])
    ustar2 = float(np.mean(tau))
    if not ustar2 > 0.0:
        raise ValueError(f"{artifact.case_name}: LES u*^2 = {ustar2}; no momentum floor")
    return MOMFLUX_FLOOR_USTAR2_FRAC * ustar2


class E2Score(NamedTuple):
    """Per-variable normalized RMSE (over all output frames after t=0) + combined."""

    per_var: dict
    combined: Array


class E2Profiles(NamedTuple):
    """What :func:`e2_score` compares: per variable, the SCM (``model``) and LES
    (``truth``) profiles on the LES heights, shape ``(frames after t=0, nz)``, with
    the case's height weights and per-variable floors."""

    steps: list
    heights: Array
    truth: dict
    model: dict
    weights: Array
    floors: dict


def e2_profiles(artifact: LESReferenceArtifact, rec: RolloutRecord,
                dt: float) -> E2Profiles:
    """The SCM and LES profiles :func:`e2_score` scores (see there)."""
    steps = _frames(artifact, dt)
    momflux_floor = momentum_flux_floor(artifact)
    if rec.fluxes.wtheta.shape[0] <= steps[-1]:
        raise ValueError(f"{artifact.case_name}: rollout has "
                         f"{rec.fluxes.wtheta.shape[0]} steps; E2 needs e2_nsteps = "
                         f"{steps[-1] + 1} (the flux AT the last frame)")
    z = jnp.asarray(artifact.heights_m)
    w = case_score_weights(artifact.case_name, z)
    sl = slice(1, None)
    truth = {
        "theta": jnp.asarray(artifact.theta)[sl], "u": jnp.asarray(artifact.u)[sl],
        "v": jnp.asarray(artifact.v)[sl],
        "wtheta": total_turbulent_flux(artifact.wtheta_resolved, artifact.wtheta_sgs)[sl],
    }
    floors = {"theta": PROFILE_FLOOR_THETA, "u": PROFILE_FLOOR_WIND,
              "v": PROFILE_FLOOR_WIND, "wtheta": PROFILE_FLOOR_FLUX,
              "qt": PROFILE_FLOOR_QT, "wqt": PROFILE_FLOOR_QFLUX,
              "uw": momflux_floor, "vw": momflux_floor}
    if artifact.is_moist:
        truth["qt"] = jnp.asarray(artifact.qt)[sl]
        truth["wqt"] = total_turbulent_flux(artifact.wqt_resolved, artifact.wqt_sgs)[sl]
    if momflux_floor is not None:
        truth["uw"] = total_turbulent_flux(artifact.uw_resolved, artifact.uw_sgs)[sl]
        truth["vw"] = total_turbulent_flux(artifact.vw_resolved, artifact.vw_sgs)[sl]
    model = {k: [] for k in truth}
    for s in steps:
        col = rec.at(s)
        for name, key in (("theta_l", "theta"), ("q_t", "qt"), ("u", "u"), ("v", "v")):
            if key in truth:
                model[key].append(interp_profile(getattr(col, name), col.z_full, z))
        fl = rec.fluxes_of_step(s)
        zh = col.z_half
        for key in ("wtheta", "wqt", "uw", "vw"):
            if key in truth:
                model[key].append(interp_profile(getattr(fl, key), zh, z))
    return E2Profiles(steps=steps, heights=z, truth=truth,
                      model={k: jnp.stack(v) for k, v in model.items()},
                      weights=w, floors={k: floors[k] for k in truth})


def e2_score(artifact: LESReferenceArtifact, rec: RolloutRecord, dt: float) -> E2Score:
    """Score a rollout against every LES frame after t=0: θ_l, q_t (moist), u, v
    profiles and heat, moisture (moist), u'w', v'w' fluxes, equally weighted (RMS of
    the per-variable normalized RMSEs). Fluxes are LES resolved+SGS totals, which are
    snapshots at the frame time, so they are compared with the SCM flux evaluated on
    the frame's own column (``rec`` must run :func:`e2_nsteps` steps). Momentum
    fluxes are scored per :func:`momentum_flux_floor`."""
    p = e2_profiles(artifact, rec, dt)
    per = {k: normalized_rmse_series(p.model[k], p.truth[k], p.weights, p.floors[k])
           for k in p.truth}
    stacked = jnp.stack([per[k] for k in per])
    return E2Score(per_var=per, combined=safe_sqrt(jnp.mean(stacked ** 2)))


# --- shared AD descent ---------------------------------------------------------------

def adam_descend_normalized(loss_of, lo: Array, hi: Array, x0: Array, *,
                            steps: int = AD_STEPS, lr: float = AD_LR):
    """The LES-suite AD tuner's loop: Adam on parameters normalised to ``[0, 1]``
    over their bounds, clipped back each step, stopped at the first non-finite loss
    or gradient; returns ``(best_x, best_loss, default_loss, history)``.

    ``loss_of`` maps PHYSICAL parameter values (a ``(p,)`` array) to a scalar loss.
    """
    import optax  # noqa: PLC0415

    span = hi - lo
    if not bool(jnp.all(span > 0.0)):
        raise ValueError("every parameter needs hi > lo")
    n0 = jnp.clip((x0 - lo) / span, 0.0, 1.0)

    def loss_n(norm):
        return loss_of(lo + jnp.clip(norm, 0.0, 1.0) * span)

    value_and_grad = jax.value_and_grad(loss_n)
    opt = optax.adam(lr)
    opt_state = opt.init(n0)
    norm = n0
    default_loss = float(loss_n(n0))
    best_loss = default_loss if math.isfinite(default_loss) else math.inf
    best = n0
    history = [(default_loss, [float(x) for x in lo + n0 * span])]
    for _ in range(max(1, steps)):
        loss, grad = value_and_grad(norm)
        if not (math.isfinite(float(loss)) and bool(jnp.all(jnp.isfinite(grad)))):
            break
        updates, opt_state = opt.update(grad, opt_state)
        norm = jnp.clip(optax.apply_updates(norm, updates), 0.0, 1.0)
        cur = float(loss_n(norm))
        history.append((cur, [float(x) for x in lo + norm * span]))
        if math.isfinite(cur) and cur < best_loss:
            best_loss, best = cur, norm
    return lo + best * span, best_loss, default_loss, history


# --- synthetic columns (L stage, E0 test columns) -----------------------------------

def synthetic_artifact(name: str, heights_m, theta, u, v, *, w_theta_s: float,
                       hours: float, nframes: int, z0_m: float, f_c: float = 0.0,
                       u_geo=None, v_geo=None) -> LESReferenceArtifact:
    """An LES-free reference holding only an initial state and its forcing (constant
    surface heat flux, optional geostrophic wind) — the input the SCM builder needs
    for an idealized run. Truth profiles beyond t=0 are the initial state repeated
    and are never scored."""
    z = jnp.asarray(heights_m)
    nt = int(nframes)

    def rep(p):
        return jnp.broadcast_to(jnp.asarray(p, z.dtype), (nt, z.shape[0]))

    def prof(p):
        return None if p is None else jnp.broadcast_to(jnp.asarray(p, z.dtype), z.shape)

    times = jnp.linspace(0.0, hours * 3600.0, nt)
    return LESReferenceArtifact(
        case_name=name, sgs="none", heights_m=z, times_s=times, theta=rep(theta),
        u=rep(u), v=rep(v), wtheta_resolved=rep(jnp.zeros_like(z)),
        wtheta_sgs=rep(jnp.zeros_like(z)), prescribe="fluxes",
        w_theta_s=jnp.full((nt,), w_theta_s, z.dtype),
        w_theta_s_cum=w_theta_s * times, f_c=f_c, u_geo=prof(u_geo), v_geo=prof(v_geo),
        les_setup={"z0_m": float(z0_m), "sfc_flux_drag": "most"})


# --- E0: static checks -------------------------------------------------------------

def _rel(residual, scale) -> float:
    r, s = float(jnp.abs(residual)), float(scale)
    return r / s if s > 0.0 else r


def e0_column_checks(candidate: Candidate, params: dict, inputs: ColumnInputs, *,
                     heat_form: str, ke_heating: bool) -> dict:
    """Contract + conservation checks of the candidate on one column (no time step).

    * ``finite``: fluxes finite with shape ``(n+1,)``;
    * ``surface_exact``: flux index 0 equals the prescribed surface flux bit-for-bit;
    * ``top_zero``: flux index n is exactly 0 (nothing leaves through the lid);
    * ``closure_rel``: relative residual of ``Σ m·tendency − (ρF)_0 + (ρF)_n`` for
      each owned field, in the conserved variable of the chosen ``heat_form``;
    * ``energy_rel``: relative residual of the total-energy rate budget
      (:func:`candidate.column_energy_budget`), dissipation heating included when on;
    * ``jit_parity``: max |eager − jit| flux difference.

    A ``"both"`` candidate computes its own surface stress, so ``surface_exact``
    covers its scalar fluxes only; ``surface_stress_power`` [W m⁻²] records
    ``ρ_0 (u_0 u'w'_0 + v_0 v'w'_0)`` at the lowest level's winds (negative = the
    stress removes kinetic energy from the column, as surface drag does).
    """
    kind = candidate.kind
    if kind == "both":  # as in the rollout: no other closure supplies a surface stress
        nan = jnp.full((), jnp.nan, inputs.u.dtype)
        inputs = inputs._replace(uw_s=nan, vw_s=nan)
    aux = candidate.init_aux(inputs)
    fa, fb, _ = candidate.fluxes(params, inputs, aux)
    fl = owned_fluxes(kind, fa, fb)
    n = inputs.mass.shape[0]
    out: dict = {}
    out["finite"] = all(f.shape == (n + 1,) and bool(jnp.all(jnp.isfinite(f)))
                        for f in fl.values())
    prescribed = {"wtheta": inputs.w_theta_s, "wqt": inputs.w_qt_s}
    if kind == "momentum":
        prescribed = {"uw": inputs.uw_s, "vw": inputs.vw_s}
    owned_prescribed = {k: v for k, v in prescribed.items() if k in fl}
    out["surface_exact"] = all(bool(fl[k][0] == s) for k, s in owned_prescribed.items())
    out["surface_mismatch"] = [float(fl[k][0] - s) for k, s in owned_prescribed.items()]
    if kind == "both":
        out["surface_stress_power"] = float(
            inputs.rho_half[0] * (inputs.u[0] * fl["uw"][0] + inputs.v[0] * fl["vw"][0]))
    out["top_zero"] = all(bool(f[-1] == 0.0) for f in fl.values())
    out["top_flux"] = [float(f[-1]) for f in fl.values()]
    ct = candidate_tendencies(kind, fa, fb, inputs, heat_form=heat_form,
                              ke_heating=ke_heating)
    m, rh = inputs.mass, inputs.rho_half
    pairs = []
    if owns_scalars(kind):
        # the heat the candidate applied, less the dissipation heating a "both"
        # candidate adds (an internal source, closed by the energy budget), must
        # close on its scalar transport
        dtheta = ct.dtheta
        if kind == "both" and ke_heating:
            heat = ke_dissipation_heating(fl["uw"], fl["vw"], *step_midpoint_winds(
                ct.du, ct.dv, inputs), rh, inputs.mass)
            dtheta = dtheta - heat / (constants.c_pd * inputs.exner)
        if heat_form == "theta":
            heat_tend, heat_flux = dtheta, rh * fl["wtheta"]
        else:
            heat_tend = dtheta * inputs.exner
            heat_flux = rh * fl["wtheta"] * inputs.exner_half
        pairs += [("heat", heat_tend, heat_flux), ("qt", ct.dqt, rh * fl["wqt"])]
    if owns_momentum(kind):
        pairs += [("u", ct.du, rh * fl["uw"]), ("v", ct.dv, rh * fl["vw"])]
    closures = {}
    for name, tend, mflux in pairs:
        resid = jnp.sum(m * tend) - (mflux[0] - mflux[-1])
        scale = jnp.sum(jnp.abs(m * tend)) + jnp.abs(mflux[0]) + jnp.abs(mflux[-1])
        closures[name] = _rel(resid, scale)
    out["closure_rel"] = closures
    eb = column_energy_budget(ct, fa, fb, inputs, kind)
    out["energy_rel"] = _rel(eb.residual, jnp.abs(eb.rate) + jnp.abs(eb.boundary_input))
    jfa, jfb, _ = jax.jit(candidate.fluxes)(params, inputs, aux)
    jfl = owned_fluxes(kind, jfa, jfb)
    out["jit_parity"] = float(jnp.max(jnp.stack([jnp.max(jnp.abs(jfl[k] - fl[k]))
                                                 for k in fl])))
    return out


def gradient_activity(loss_of_params, params: dict) -> dict:
    """``∂loss/∂p`` for every parameter; ``inert`` lists those with exactly zero
    gradient (CLAUDE.md: no inert parameters), ``nonfinite`` those with NaN/inf."""
    g = jax.grad(loss_of_params)(params)
    vals = {k: float(v) for k, v in g.items()}
    return {"grad": vals,
            "inert": sorted(k for k, v in vals.items() if v == 0.0),
            "nonfinite": sorted(k for k, v in vals.items() if not math.isfinite(v))}


# --- L stage: idealized limits ---------------------------------------------------

def implied_diffusivity(flux: Array, prof: Array, z_full: Array) -> Array:
    """``K = −F/(∂φ/∂z)`` at the interior interfaces (n-1,) from interface fluxes
    (n+1,) and full-level values (n,)."""
    grad = (prof[1:] - prof[:-1]) / (z_full[1:] - z_full[:-1])
    return -flux[1:-1] / grad


def neutral_surface_layer_metrics(inputs: ColumnInputs, fluxes: Fluxes, *,
                                  z_max: float) -> dict:
    """Surface-layer (interfaces below ``z_max``) implied von Kármán constant from
    momentum, ``κ_m = K_m/(u_* z)``, turbulent Prandtl number ``Pr_t = κ u_* z/K_h``
    (``κ = constants.kappa_von_karman``, ``u_*`` from the applied surface stress) and
    the constant-flux ratio ``u'w'(z)/u'w'_0``. Medians over the selected interfaces."""
    zi = inputs.z_half[1:-1]
    sel = zi < z_max
    ustar = jnp.sqrt(jnp.hypot(fluxes.uw[0], fluxes.vw[0]))
    km = implied_diffusivity(fluxes.uw, inputs.u, inputs.z_full)
    kh = implied_diffusivity(fluxes.wtheta, inputs.theta_l, inputs.z_full)
    prt = constants.kappa_von_karman * ustar * zi / kh
    return {"ustar": float(ustar), "n_levels": int(jnp.sum(sel)),
            "kappa_m_median": float(jnp.median((km / (ustar * zi))[sel])),
            "Pr_t_median": float(jnp.median(prt[sel])),
            "const_flux_ratio_median": float(jnp.median(
                (fluxes.uw[1:-1] / fluxes.uw[0])[sel]))}


def stable_metrics(inputs: ColumnInputs, fluxes: Fluxes, heights: tuple) -> dict:
    """K_m, K_h at the interfaces nearest ``heights``, their z-dependence
    ``K(top)/K(bottom)`` (1 for a z-less closure) and ``Pr_t = K_m/K_h``."""
    zi = inputs.z_half[1:-1]
    km = implied_diffusivity(fluxes.uw, inputs.u, inputs.z_full)
    kh = implied_diffusivity(fluxes.wtheta, inputs.theta_l, inputs.z_full)
    idx = [int(jnp.argmin(jnp.abs(zi - h))) for h in heights]
    return {"z": [float(zi[i]) for i in idx], "K_m": [float(km[i]) for i in idx],
            "K_h": [float(kh[i]) for i in idx],
            "K_m_ratio_top_bottom": float(km[idx[-1]] / km[idx[0]]),
            "K_h_ratio_top_bottom": float(kh[idx[-1]] / kh[idx[0]]),
            "Pr_t": [float(km[i] / kh[i]) for i in idx]}


def free_convection_metrics(theta: Array, z_full: Array, wtheta: Array, z_flux: Array,
                            *, q0: float, theta0: float) -> dict:
    """Mixed-layer diagnostics: ``z_i`` = height of the most negative heat flux,
    entrainment ratio ``min(wθ)/Q0``, mid-mixed-layer lapse rate ``∂θ/∂z`` averaged
    over 0.3–0.7 z_i [K/km], and that lapse rate in Deardorff units
    ``(∂θ/∂z) z_i/θ_*`` (``w_* = (g Q0 z_i/θ0)^{1/3}``, ``θ_* = Q0/w_*``)."""
    k = int(jnp.argmin(wtheta))
    zi = float(z_flux[k])
    grad = jnp.gradient(theta, z_full)
    sel = (z_full > 0.3 * zi) & (z_full < 0.7 * zi)
    lapse = float(jnp.mean(grad[sel])) if bool(jnp.any(sel)) else float("nan")
    wstar = (constants.g * q0 * zi / theta0) ** (1.0 / 3.0)
    tstar = q0 / wstar
    return {"z_i": zi, "entrainment_ratio": float(wtheta[k]) / q0,
            "lapse_mid_ml_K_per_km": 1e3 * lapse, "lapse_deardorff": lapse * zi / tstar,
            "w_star": wstar}


def ekman_metrics(u: Array, v: Array, uw: Array, vw: Array, z_half: Array,
                  *, ug: float, vg: float) -> dict:
    """Neutral-Ekman diagnostics from (time-mean) surface-first profiles: turning
    angle between the lowest-level wind and the geostrophic wind [deg], ``u_*/G``,
    and the height where the stress magnitude falls to 5 % of its surface value."""
    alpha = math.degrees(math.atan2(float(v[0]), float(u[0])) - math.atan2(vg, ug))
    tau = jnp.hypot(uw, vw)
    frac = tau / tau[0]
    k = int(jnp.argmax(frac < 0.05))
    h = (float(z_half[k - 1] + (0.05 - frac[k - 1]) * (z_half[k] - z_half[k - 1])
               / (frac[k] - frac[k - 1])) if k > 0 else float("nan"))
    return {"turning_angle_deg": alpha,
            "ustar_over_G": float(jnp.sqrt(tau[0])) / math.hypot(ug, vg),
            "h_stress_5pct_m": h}

