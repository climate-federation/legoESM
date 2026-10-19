"""Evaluation stages for experimental closures (the co-scientist experimenter).

E0 static checks, L idealized limits, E1 a-priori fluxes, E2 a-posteriori SCM runs
with AD tuning. Every stage drives the SAME single-column model the roster is scored
in (:mod:`~legoesm.atmosphere.les_suite.scm_runner`); a candidate replaces only the
part it owns (scalars OR momentum), and the other part is a named baseline closure.

How a candidate enters the column (:func:`rollout`): the SCM step calls a substitute
physics function that returns ``baseline_total − baseline_turbulence[owned] +
candidate[owned]``. Combined physics is a parallel split (every process sees the same
input state, tendencies summed), so that subtraction removes the baseline closure's
owned part exactly. The candidate gets the surface heat/moisture flux the forcing
injected for this stage and the surface stress the baseline surface layer applied,
so both arms of a comparison see the same surface boundary condition.
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
from legoesm.core.profile_metrics import layer_weights_from_heights, safe_sqrt

from legoesm import constants

from .bridge import LESReferenceArtifact, total_turbulent_flux
from .candidate import (
    Candidate,
    ColumnInputs,
    candidate_tendencies,
    column_energy_budget,
    column_inputs,
    half_level_average,
    tendency_to_flux,
)
from .scm_coupling import interp_profile, liquid_water_theta
from .score import (
    PROFILE_FLOOR_FLUX,
    PROFILE_FLOOR_QFLUX,
    PROFILE_FLOOR_QT,
    PROFILE_FLOOR_THETA,
    PROFILE_FLOOR_WIND,
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
                      uw_s, vw_s, z0_m=None, z0h_m=None) -> ColumnInputs:
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
        dt=dt, temp=temp[::-1], q_v=q_v[::-1], q_c=q_c[::-1], z0_m=z0_m, z0h_m=z0h_m)


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
    if kind == "scalar":
        d_temp = (_col(tend_full.dT_dt.data) - _col(tend_turb.dT_dt.data)
                  + cand.dtheta * inputs.exner)
        tr = dict(tend_full.tracer_tendencies or {})
        if "q_v" in tr:
            q_full = _col(tr["q_v"].data)
            q_turb = _col(tend_turb.tracer_tendencies["q_v"].data)
            tr["q_v"] = _swap(tr["q_v"], q_full - q_turb + cand.dqt)
        return out._replace(dT_dt=_swap(tend_full.dT_dt, d_temp), tracer_tendencies=tr)
    du = _col(tend_full.du_dt.data) - _col(tend_turb.du_dt.data) + cand.du
    dv = _col(tend_full.dv_dt.data) - _col(tend_turb.dv_dt.data) + cand.dv
    out = out._replace(du_dt=_swap(tend_full.du_dt, du), dv_dt=_swap(tend_full.dv_dt, dv))
    if cand.dtheta is not None:  # KE-dissipation heating of a momentum candidate
        d_temp = _col(tend_full.dT_dt.data) + cand.dtheta * inputs.exner
        out = out._replace(dT_dt=_swap(tend_full.dT_dt, d_temp))
    return out


#: Roster closures with a prognostic turbulence carry (TKE / moments) that the
#: baseline updates from its OWN fluxes; refused as the other half of a candidate.
PROGNOSTIC_TURBULENCE = ("tke", "mynn25", "clubb_lite", "clubb", "edmf")


class RolloutRecord(NamedTuple):
    """``inputs``/``fluxes``: the column the closure saw at each step and the fluxes
    applied in that step (leading axis = step, surface-first); ``final_inputs``: the
    column after the last step; ``other_accel``: per step, the (du, dv) of everything
    but a momentum candidate with dissipation heating (else ``None``)."""

    inputs: ColumnInputs
    fluxes: Fluxes
    final_inputs: ColumnInputs
    other_accel: tuple | None = None

    def at(self, step: int) -> ColumnInputs:
        """The column state after ``step`` steps (``step == nsteps`` → final)."""
        n = self.fluxes.wtheta.shape[0]
        if step == n:
            return self.final_inputs
        return jax.tree_util.tree_map(lambda a: a[step], self.inputs)

    def fluxes_of_step(self, step: int) -> Fluxes:
        return jax.tree_util.tree_map(lambda a: a[step], self.fluxes)


def rollout(scm, physics_config, nsteps: int, *, candidate: Candidate | None = None,
            params: dict | None = None, heat_form: str | None = None,
            ke_heating: bool | None = None, f_c: float = 0.0, u_geo=None, v_geo=None):
    """Free-run ``scm`` for ``nsteps`` with ``candidate`` owning its part (or the pure
    baseline when ``candidate is None``). Returns ``(RolloutRecord, aux_final)``.

    The baseline closure is ``physics_config.turbulence``; ``scm`` must have been
    built with the same config (its physics is the baseline total). Forward Euler
    only (one physics call per step, so the candidate sees the same stage state).
    """
    if scm.time_integrator != "forward_euler":
        raise ValueError("rollout supports forward_euler only (one stage per step)")
    if candidate is not None:
        if params is None:
            raise ValueError("a candidate rollout needs params")
        if candidate.kind == "scalar" and heat_form is None:
            raise ValueError("a scalar candidate rollout needs an explicit heat_form")
        if candidate.kind == "momentum" and ke_heating is None:
            raise ValueError("a momentum candidate rollout needs an explicit ke_heating")
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

    forcing = scm.forcing_arg

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
                                 z0h_m=z0h_m)
        imp = implied_fluxes(tend_turb, base)
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
            if candidate is None:
                stash["fluxes"], stash["aux"] = imp, aux
                return tend_full, phys_out
            fa, fb, aux_new = candidate.fluxes(params, inputs, aux)
            other = (_other_accel(tend_full, tend_turb, st, sc, t)
                     if kind == "momentum" and ke_heating else None)
            ct = candidate_tendencies(kind, fa, fb, inputs, heat_form=heat_form,
                                      ke_heating=ke_heating, other_accel=other)
            stash["other"] = other
            stash["fluxes"] = (imp._replace(wtheta=fa, wqt=fb) if kind == "scalar"
                               else imp._replace(uw=fa, vw=fb))
            stash["aux"] = aux_new
            return _owned_tendencies(tend_full, tend_turb, ct, inputs, kind), phys_out

        new_state, new_phys = scm.pure_step_with_physics(state, phys, t, physics_fn)
        new_phys = new_phys if new_phys is not None else phys
        return ((new_state, new_phys, t + dt, stash["aux"]),
                (stash["inputs"], stash["fluxes"], stash["other"]))

    (fstate, fphys, t_f, aux_f), (inputs_rec, flux_rec, other_rec) = lax.scan(
        body, (scm.state, scm.phys_state, scm.t_seconds, aux0), None, length=nsteps)
    tend_f, _ = physics_turb(fstate, scm.grid, sigma, phys_state=fphys)
    final_inputs, _ = _inputs(fstate, fphys, tend_f, t_f)
    return RolloutRecord(inputs=inputs_rec, fluxes=flux_rec, final_inputs=final_inputs,
                         other_accel=other_rec), aux_f


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


def e2_score(artifact: LESReferenceArtifact, rec: RolloutRecord, dt: float) -> E2Score:
    """Score a rollout against every LES frame after t=0: θ_l, q_t (moist), u, v
    profiles and heat, moisture (moist), u'w', v'w' fluxes, equally weighted (RMS of
    the per-variable normalized RMSEs). Fluxes are LES resolved+SGS totals, which are
    snapshots at the frame time, so they are compared with the SCM flux evaluated on
    the frame's own column (``rec`` must run :func:`e2_nsteps` steps). Momentum
    fluxes are scored per :func:`momentum_flux_floor`."""
    steps = _frames(artifact, dt)
    momflux_floor = momentum_flux_floor(artifact)
    if rec.fluxes.wtheta.shape[0] <= steps[-1]:
        raise ValueError(f"{artifact.case_name}: rollout has "
                         f"{rec.fluxes.wtheta.shape[0]} steps; E2 needs e2_nsteps = "
                         f"{steps[-1] + 1} (the flux AT the last frame)")
    z = jnp.asarray(artifact.heights_m)
    w = layer_weights_from_heights(z)
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
    per = {k: normalized_rmse_series(jnp.stack(model[k]), truth[k], w, floors[k])
           for k in truth}
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
        les_setup={"z0_m": float(z0_m)})


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
    """
    aux = candidate.init_aux(inputs)
    fa, fb, _ = candidate.fluxes(params, inputs, aux)
    n = inputs.mass.shape[0]
    out: dict = {}
    out["finite"] = bool(fa.shape == (n + 1,) and fb.shape == (n + 1,)
                         and jnp.all(jnp.isfinite(fa)) and jnp.all(jnp.isfinite(fb)))
    if candidate.kind == "scalar":
        sa, sb = inputs.w_theta_s, inputs.w_qt_s
    else:
        sa, sb = inputs.uw_s, inputs.vw_s
    out["surface_exact"] = bool(fa[0] == sa) and bool(fb[0] == sb)
    out["surface_mismatch"] = [float(fa[0] - sa), float(fb[0] - sb)]
    out["top_zero"] = bool(fa[-1] == 0.0) and bool(fb[-1] == 0.0)
    out["top_flux"] = [float(fa[-1]), float(fb[-1])]
    ct = candidate_tendencies(candidate.kind, fa, fb, inputs, heat_form=heat_form,
                              ke_heating=ke_heating)
    m, rh = inputs.mass, inputs.rho_half
    if candidate.kind == "scalar":
        if heat_form == "theta":
            heat_tend, heat_flux = ct.dtheta, rh * fa
        else:
            heat_tend, heat_flux = ct.dtheta * inputs.exner, rh * fa * inputs.exner_half
        pairs = (("heat", heat_tend, heat_flux), ("qt", ct.dqt, rh * fb))
    else:
        pairs = (("u", ct.du, rh * fa), ("v", ct.dv, rh * fb))
    closures = {}
    for name, tend, mflux in pairs:
        resid = jnp.sum(m * tend) - (mflux[0] - mflux[-1])
        scale = jnp.sum(jnp.abs(m * tend)) + jnp.abs(mflux[0]) + jnp.abs(mflux[-1])
        closures[name] = _rel(resid, scale)
    out["closure_rel"] = closures
    eb = column_energy_budget(ct, fa, fb, inputs, candidate.kind)
    out["energy_rel"] = _rel(eb.residual, jnp.abs(eb.rate) + jnp.abs(eb.boundary_input))
    jfa, jfb, _ = jax.jit(candidate.fluxes)(params, inputs, aux)
    out["jit_parity"] = float(jnp.maximum(jnp.max(jnp.abs(jfa - fa)),
                                          jnp.max(jnp.abs(jfb - fb))))
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

