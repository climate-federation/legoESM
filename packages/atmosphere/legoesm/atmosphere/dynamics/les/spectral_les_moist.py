"""Moist coupling for the pseudo-spectral incompressible plane LES.

Connects the (Boussinesq) spectral LES core to the SWAPPABLE legoESM
microphysics dispatch — the same per-scheme tendency interface the plane CRM
uses (``microphysics/integration._get_microphysics_fn``), so Morrison (default),
Kessler, Thompson, P3, SDM, … all swap by ``MicrophysicsConfig.scheme`` with no
core change. Two pieces:

* :func:`make_anelastic_reference` — a FIXED hydrostatic reference column
  (p, Π, ρ on the LES levels) from the case sounding. The incompressible core
  has no prognostic pressure/density; the microphysics thermodynamics (T from
  θ, saturation, fall speeds) use this reference column — the standard
  anelastic/Boussinesq LES approximation for shallow boundary-layer clouds
  (BOMEX/DYCOMS/RICO depth ≲ 3 km ⇒ reference-state errors are small).

* :func:`make_les_microphysics_fn` — builds a jitted ``(theta, tracers) →
  (dθ/dt, dtracers/dt, surface_precip)`` adapter that reshapes the LES fields
  to microphysics columns, calls the dispatched scheme, and maps the
  tendencies back. The latent heating is applied to θ via the reference Exner.

* :func:`step_lagrangian_sdm_les` — opt-in stateful coupling for the advected
  Lagrangian SDM path. It carries a persistent super-droplet state beside the
  LES state and overwrites q_c/q_r with particle-binned diagnostics after each
  split SDM update.

VERTICAL ORDER: the microphysics column convention is TOP-DOWN (index 0 = model
top; ``sedimentation_tendency`` propagates flux from index k−1 INTO k, and the
"surface" precip flux leaves the LAST index). The spectral LES is BOTTOM-UP
(index 0 = lowest cell). The adapter flips the z-axis on entry and flips every
tendency back on exit — getting this wrong would rain upward.

The tracer slot layout is the STANDARD microphysics layout ([0]=q_v, [1]=q_c,
[2]=q_r, [3]=q_i, [4]=q_s, [5]=q_g, [6]=N_c, [7]=N_r, [8]=N_i), identical to
the plane CRM's, enforced against ``_PLANE_MIN_TRACER_SLOTS`` at build time.
"""
from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.microphysics.integration import (
    get_microphysics_fn,
    min_tracer_slots,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState


def moist_diagnostics(u, v, w_centre, theta, tracers, rho_c, dz, dx,
                      dt, dy=None, qc_thresh=1.0e-5):
    """Bundle of moist-LES diagnostics for the stabilization test matrix.

    Host (numpy-on-jax) reductions; cheap, call per print. ``w_centre`` is the
    cell-centred w. Returns a plain dict:

    * ``cloud_frac``  projected cloud cover (any q_c > qc_thresh in a column)
    * ``lwp``         domain-mean liquid-water path [g/m²]
    * ``max_w``       max |w| [m/s]            ``w_var`` ⟨w'²⟩ peak [m²/s²]
    * ``tke``         peak resolved 0.5⟨u'²+v'²+w'²⟩ [m²/s²]
    * ``max_cfl``     max(|u|,|v|)·dt/dx, |w|·dt/dz
    * ``total_water`` domain-mean column-integrated q_t = ∫ρ(q_v+q_c+q_r)dz [kg/m²]
                      — track its drift for the total-water conservation error
    * ``qv_min/qv_max/qc_max`` scalar bounds [kg/kg] (positivity monitor)
    """
    import numpy as np
    u = np.asarray(u); v = np.asarray(v); wc = np.asarray(w_centre)
    tr = np.asarray(tracers); rho = np.asarray(rho_c)
    qv, qc = tr[..., 0], tr[..., 1]
    qr = tr[..., 2] if tr.shape[-1] > 2 else np.zeros_like(qv)
    up = u - u.mean((0, 1)); vp = v - v.mean((0, 1)); wp = wc - wc.mean((0, 1))
    uu = (up * up).mean((0, 1)); vv = (vp * vp).mean((0, 1))
    ww = (wp * wp).mean((0, 1))
    cloudy = np.any(qc > qc_thresh, axis=-1)
    lwp = float((qc * rho[None, None, :]).sum(-1).mean()) * dz * 1.0e3
    qt_col = ((qv + qc + qr) * rho[None, None, :]).sum(-1) * dz  # (ny,nx) [kg/m²]
    dy = dx if dy is None else dy
    max_w = float(np.abs(wc).max())                          # cell-centred w
    cfl = max(float(np.abs(u).max()) * dt / dx,
              float(np.abs(v).max()) * dt / dy,
              max_w * dt / dz)
    return dict(
        cloud_frac=float(cloudy.mean()), lwp=lwp, max_w=max_w,
        w_var=float(ww.max()), tke=float((0.5 * (uu + vv + ww)).max()),
        max_cfl=cfl, total_water=float(qt_col.mean()),
        qv_min=float(qv.min()), qv_max=float(qv.max()), qc_max=float(qc.max()))


def conserving_positive(tracers, rho_c, dz, n_water=6):
    """MASS-CONSERVING positivity fixer for the water tracers (slots 0..n_water-1).

    The pseudo-spectral scalar transport is NON-MONOTONE: at the sharp trade /
    stratocumulus moisture inversion it overshoots/undershoots (Gibbs), driving
    small NEGATIVE q. A plain ``clip(q, 0)`` removes those negatives but ADDS the
    deficit as spurious water — a continuous moisture source that feeds runaway
    condensation (whole-column saturation). Instead, per column per species,
    clip to zero THEN rescale the positive cells so the column-integrated
    ``∫ρq dz`` is unchanged (hole-filling / borrowing; standard for
    positive-definite-but-non-monotone scalar transport). Number slots
    (≥ n_water) clip freely (their conservation is not physically required).

    Returns ``(tracers_fixed, created_water)`` where ``created_water`` is the
    column-summed water the OLD clip WOULD have created — a near-zero
    monotonicity diagnostic, kept for the run log.
    """
    w = rho_c * dz                                       # (nz,) mass weight
    q = tracers[..., :n_water]                           # (ny,nx,nz,n_water)
    q_clip = jnp.clip(q, 0.0, None)
    col_before = jnp.sum(q * w[None, None, :, None], axis=2, keepdims=True)
    col_clip = jnp.sum(q_clip * w[None, None, :, None], axis=2, keepdims=True)
    created = jnp.sum(jnp.clip(-q, 0.0, None) * w[None, None, :, None])
    # factor ≤ 1 removes the borrowed mass from the positives; guard an
    # all-nonpositive column (col_clip→0) by leaving it at zero.
    factor = jnp.where(col_clip > 1e-30, col_before / col_clip, 0.0)
    q_fixed = q_clip * jnp.clip(factor, 0.0, 1.0)
    numbers = jnp.clip(tracers[..., n_water:], 0.0, None)
    return jnp.concatenate([q_fixed, numbers], axis=-1), created


class SpectralRefState(NamedTuple):
    """Fixed hydrostatic reference column on the LES grid (BOTTOM-UP, matching
    ``SpectralLESGrid.z_c``/``z_f``)."""
    p_c: jax.Array        # (nz,)   pressure at cell centres [Pa]
    p_f: jax.Array        # (nz+1,) pressure at faces [Pa]
    exner_c: jax.Array    # (nz,)   Exner Π = (p/p_ref)^κ
    rho_c: jax.Array      # (nz,)   density [kg/m³]


class LagrangianSDMSegmentDiagnostics(NamedTuple):
    """Scalar diagnostics accumulated by a compiled Lagrangian-SDM segment."""

    max_abs_total_water_error: jax.Array
    total_water_error: jax.Array
    n_active: jax.Array
    u_star: jax.Array


def make_anelastic_reference(z_c, z_f, p_sfc, theta_prof, qv_prof=None,
                             dtype=jnp.float64) -> SpectralRefState:
    """Hydrostatic reference column from a θ(z) (+ optional q_v(z)) profile.

    Integrates the Exner hydrostatic balance ``dΠ/dz = −g/(c_pd·θ_v)`` upward
    from ``Π_sfc = (p_sfc/p_ref)^κ`` with trapezoidal θ_v, then
    ``p = p_ref·Π^(1/κ)``, ``ρ = p/(R_d·T_v)`` (shared constants; no literals).

    ``theta_prof``/``qv_prof`` are (nz,) on the cell centres (bottom-up).
    """
    z_c = jnp.asarray(z_c, dtype=dtype)
    z_f = jnp.asarray(z_f, dtype=dtype)
    th = jnp.asarray(theta_prof, dtype=dtype)
    qv = (jnp.zeros_like(th) if qv_prof is None
          else jnp.asarray(qv_prof, dtype=dtype))
    eps_v = 1.0 / constants.epsilon - 1.0
    th_v = th * (1.0 + eps_v * qv)                       # (nz,)

    kappa = constants.kappa
    exner_sfc = (p_sfc / constants.p_ref) ** kappa
    dz_f = jnp.diff(z_f)                                 # (nz,) layer thickness
    # Π at faces: Π_f[0]=Π_sfc; downward accumulation of the layer increments
    # using the CENTRE θ_v of each layer (midpoint rule on the layer).
    dpi = -constants.g / (constants.c_pd * th_v) * dz_f  # (nz,) per-layer ΔΠ
    exner_f = exner_sfc + jnp.concatenate(
        [jnp.zeros((1,), dtype=dtype), jnp.cumsum(dpi)])  # (nz+1,)
    # Π at centres: same integral to z_c (half-layer from the lower face).
    exner_c = exner_f[:-1] + 0.5 * dpi
    if bool(jnp.any(exner_f <= 0.0)):
        raise ValueError("make_anelastic_reference: Π ≤ 0 aloft — the column "
                         "is taller than the reference atmosphere supports.")
    p_c = constants.p_ref * exner_c ** (1.0 / kappa)
    p_f = constants.p_ref * exner_f ** (1.0 / kappa)
    T_v = th_v * exner_c
    rho_c = p_c / (constants.R_d * T_v)
    return SpectralRefState(p_c=p_c, p_f=p_f, exner_c=exner_c, rho_c=rho_c)


def make_les_microphysics_fn(micro_config: MicrophysicsConfig,
                             ref: SpectralRefState, dz: float, dt: float):
    """Build the swappable microphysics adapter for the spectral LES.

    Returns ``micro(theta, tracers) -> (dtheta_dt, dtracers_dt, precip_sfc)``
    with LES shapes ``(ny,nx,nz)`` / ``(ny,nx,nz,nt)`` / ``(ny,nx)`` (bottom-up;
    the top-down flip happens inside). Apply forward-Euler OUTSIDE
    ``spectral_les_plane.step`` (mirrors the CRM's radiation/microphysics
    cadence pattern; keeps the dycore step scheme-agnostic).

    ``dz`` is the (uniform) LES layer thickness, ``dt`` the LES step (used by
    the scheme's sedimentation positivity limiter).
    """
    scheme_name, micro_fn, scheme_config = get_microphysics_fn(micro_config)
    if scheme_name == "ml_emulator":
        raise NotImplementedError(
            "spectral-LES microphysics adapter: 'ml_emulator' takes an extra "
            "model argument (stateful cache) — use the CRM integration path, "
            "or extend this adapter with the model cache if needed.")
    if scheme_name == "sundqvist":
        raise ValueError(
            "spectral-LES microphysics adapter: 'sundqvist' is a LARGE-SCALE "
            "diagnostic condensation scheme (Sundqvist 1978/SBK89) that gates "
            "condensation on a subgrid RH>RH_crit partial-cloud-fraction basis "
            "designed for coarse GCM grid boxes, NOT resolved-cloud LES. On the "
            "LES grid it condenses in every cell above RH_crit -> spurious "
            "near-100% cloud cover and inflated LWP (BOMEX/DYCOMS/RICO measured "
            "cloud_cover ~0.9-1.0, LWP 20-43 g/m^2 vs 0.03-0.24 / 0.4-5 g/m^2 for "
            "the resolved-cloud schemes). Use a resolved-cloud microphysics: "
            "kessler, seifert_beheng, morrison, thompson, p3, sdm, or fast_sbm.")
    if micro_fn is None:                                   # scheme "none"
        raise ValueError("microphysics scheme 'none' — build no adapter; run "
                         "the dry driver instead.")
    min_slots = min_tracer_slots(scheme_name, scheme_config)

    def micro(theta, tracers):
        ny, nx, nz = theta.shape
        nt = tracers.shape[-1]
        if nt < min_slots:
            raise ValueError(
                f"microphysics scheme {scheme_name!r} writes up to {min_slots} "
                f"tracer slots (standard layout) but the LES state carries "
                f"{nt}. Allocate n_tracers >= {min_slots}.")
        ncol = ny * nx
        # LES (bottom-up) → microphysics columns (TOP-DOWN): flip z, flatten.
        flip = lambda a: a[..., ::-1]                       # noqa: E731
        th_td = flip(theta).reshape(ncol, nz)
        tr_td = tracers[..., ::-1, :].reshape(ncol, nz, nt)
        exner = ref.exner_c[::-1][None, :]                  # (1, nz) top-down
        T_col = th_td * exner
        p_full = jnp.broadcast_to(ref.p_c[::-1][None, :], (ncol, nz))
        p_half = jnp.broadcast_to(ref.p_f[::-1][None, :], (ncol, nz + 1))
        rho = jnp.broadcast_to(ref.rho_c[::-1][None, :], (ncol, nz))
        dz_col = jnp.full((ncol, nz), dz, dtype=theta.dtype)

        def slot(idx):
            if nt > idx:
                return tr_td[..., idx]
            return jnp.zeros((ncol, nz), dtype=theta.dtype)

        # Cloud and rain number are STORED per MASS [#/kg] so the transport
        # operator is right for them; the schemes work per VOLUME.  Ice number
        # is per mass on both sides.  See microphysics/integration.py.
        from legoesm.atmosphere.physics.microphysics.integration import (
            number_per_mass_to_per_volume,
            number_per_volume_to_per_mass,
        )
        hyd = HydrometeorState(
            q_c=slot(1), q_r=slot(2), q_i=slot(3), q_s=slot(4), q_g=slot(5),
            N_c=number_per_mass_to_per_volume(slot(6), rho),
            N_r=number_per_mass_to_per_volume(slot(7), rho),
            N_i=slot(8),
            N_s=(slot(9) if nt > 9 else None),
            N_g=(slot(10) if nt > 10 else None),
        )
        out = micro_fn(T_col, slot(0), hyd, p_full, p_half, rho, dz_col, dt,
                       scheme_config)

        # Latent heating → θ tendency via the reference Exner; flip back.
        dth = (out.dT_dt / exner).reshape(ncol, nz)
        dtheta_dt = flip(dth.reshape(ny, nx, nz))
        tend = [out.dq_v_dt, out.dq_c_dt, out.dq_r_dt, out.dq_i_dt,
                out.dq_s_dt, out.dq_g_dt,
                number_per_volume_to_per_mass(out.dN_c_dt, rho),
                number_per_volume_to_per_mass(out.dN_r_dt, rho),
                out.dN_i_dt, out.dN_s_dt, out.dN_g_dt]
        dtr = jnp.zeros((ncol, nz, nt), dtype=theta.dtype)
        for idx, f in enumerate(tend):
            if f is not None and nt > idx:
                dtr = dtr.at[..., idx].set(f)
        dtracers_dt = dtr.reshape(ny, nx, nz, nt)[..., ::-1, :]
        precip = out.precipitation.reshape(ny, nx)          # [kg/m²/s]
        return dtheta_dt, dtracers_dt, precip

    micro.scheme_name = scheme_name
    return micro


def step_lagrangian_sdm_les(
    state,
    sdm_state,
    g,
    ref: SpectralRefState,
    dt: float,
    sdm_config,
    u_geo,
    f_cor: float,
    *,
    first: bool = False,
    force=(0.0, 0.0),
    sfc_theta_flux=0.0,
    t_sfc=None,
    sfc_qv_flux=0.0,
    do_condensation: bool = True,
    do_coalescence: bool = True,
):
    """One opt-in spectral LES step coupled to persistent Lagrangian SDM.

    This is a driver hook, not part of the default microphysics dispatch. The
    normal spectral LES step advances the Eulerian velocity, θ and vapor tracer;
    then the persistent super-droplets are advected/sedimented, collided within
    cells, condensed/evaporated, and binned back to diagnostic q_c/q_r slots.

    Returns ``(state_new, sdm_state_new, u_star, diagnostics)``. Existing dry and
    Eulerian microphysics paths are unchanged unless callers explicitly use this
    function and carry ``sdm_state``.

    JIT usage: ``g``, ``ref`` and ``sdm_config`` are static Python/NestedTuple
    configuration objects, so direct ``jax.jit(step_lagrangian_sdm_les)`` is not
    the supported entry point. Use :func:`make_lagrangian_sdm_les_step`, which
    closes over those objects and returns a jitted ``(state, sdm_state, dt)``
    step. With the default ``sdm_config.collision_mode="stochastic"``,
    coalescence is forward-only because pair order and integer collision counts
    are random/discontinuous. With
    ``sdm_config.collision_mode="deterministic"``, collision uses a mean-field
    expected pair increment; for fixed particle-cell membership the full split
    update has JAX VJPs with respect to droplet radii/multiplicities and
    Eulerian thermodynamic fields. The floor-based particle-cell assignment is
    still discrete with respect to particle positions.
    """
    from legoesm.atmosphere.dynamics.les import spectral_les_plane as sl
    from legoesm.atmosphere.physics.microphysics.sdm.lagrangian import (
        apply_lagrangian_sdm_to_les_state,
    )

    state_new, u_star = sl.step(
        state, g=g, dt=dt, u_geo=u_geo, f_cor=f_cor, first=first, force=force,
        sfc_theta_flux=sfc_theta_flux, t_sfc=t_sfc, sfc_qv_flux=sfc_qv_flux)
    state_new, sdm_state_new, diagnostics = apply_lagrangian_sdm_to_les_state(
        state_new, sdm_state, g, ref, dt, sdm_config,
        do_condensation=do_condensation, do_coalescence=do_coalescence)
    return state_new, sdm_state_new, u_star, diagnostics


def make_lagrangian_sdm_les_step(
    g,
    ref: SpectralRefState,
    sdm_config,
    *,
    u_geo,
    f_cor: float,
    force=(0.0, 0.0),
    sfc_theta_flux=0.0,
    t_sfc=None,
    sfc_qv_flux=0.0,
    do_condensation: bool = True,
    do_coalescence: bool = True,
):
    """Return the supported jitted Lagrangian-SDM spectral-LES step closure.

    The returned function has signature ``step(state, sdm_state, dt, *,
    first=False)``. ``first`` is static because the AB2 LES integrator uses it
    in a Python branch. Use ``SDMConfig(collision_mode="deterministic")`` for
    reverse-mode sensitivities through Lagrangian coalescence; the stochastic
    default is a forward simulation path.
    """

    @functools.partial(jax.jit, static_argnames=("first",))
    def step(state, sdm_state, dt, *, first: bool = False):
        return step_lagrangian_sdm_les(
            state,
            sdm_state,
            g,
            ref,
            dt,
            sdm_config,
            u_geo=u_geo,
            f_cor=f_cor,
            first=first,
            force=force,
            sfc_theta_flux=sfc_theta_flux,
            t_sfc=t_sfc,
            sfc_qv_flux=sfc_qv_flux,
            do_condensation=do_condensation,
            do_coalescence=do_coalescence,
        )

    return step


def update_lagrangian_sdm_segment_diagnostics(
    diagnostics: LagrangianSDMSegmentDiagnostics,
    u_star,
    step_diagnostics: dict,
) -> LagrangianSDMSegmentDiagnostics:
    """Fold one step's scalar SDM diagnostics into a segment accumulator."""

    err = step_diagnostics["total_water_error"]
    return LagrangianSDMSegmentDiagnostics(
        max_abs_total_water_error=jnp.maximum(
            diagnostics.max_abs_total_water_error, jnp.abs(err)),
        total_water_error=err,
        n_active=step_diagnostics["n_active"],
        u_star=u_star,
    )


def make_lagrangian_sdm_step_segment(
    step_fn,
    *,
    segment_steps: int,
    donate_args: bool = False,
):
    """Return a jitted bounded-memory runner for Lagrangian SDM step chunks.

    ``step_fn`` must have the same call signature as
    :func:`make_lagrangian_sdm_les_step`'s return value and is always called with
    ``first=False``. Callers should run the AB2/RK startup step separately when
    needed. ``steps_to_run`` is a traced scalar, so one compiled executable is
    reused for full chunks and short tail chunks without rebuilding closures or
    changing static arguments. ``segment_steps`` is the caller-side chunk size
    used by drivers to cap synchronization intervals; the compiled loop itself
    runs exactly ``steps_to_run`` iterations.

    Prefer passing a raw (non-jitted) step body. If ``step_fn`` is itself a
    separately jitted function, XLA may have to materialize its full diagnostic
    output tuple before this wrapper drops the large arrays. With a raw body, the
    segment returns only the final state and scalar diagnostics; large per-step
    arrays in the one-step diagnostics stay internal to the compiled body and are
    not transferred to Python.

    This is a forward driver helper. Reverse-mode sensitivities should continue
    to use the non-donating one-step deterministic path; the dynamic loop trip
    count here is chosen to avoid production-driver retracing and memory growth.
    """

    segment_steps = int(segment_steps)
    if segment_steps < 1:
        raise ValueError(f"segment_steps must be >= 1, got {segment_steps}")

    def run_segment(state, sdm_state, dt, steps_to_run, diagnostics):
        steps_to_run = jnp.asarray(steps_to_run, dtype=jnp.int32)

        def body(_k, carry):
            state_i, sdm_i, diag_i = carry
            state_i, sdm_i, u_star, step_diag = step_fn(
                state_i, sdm_i, dt, first=False)
            diag_i = update_lagrangian_sdm_segment_diagnostics(
                diag_i, u_star, step_diag)
            return state_i, sdm_i, diag_i

        state, sdm_state, diagnostics = lax.fori_loop(
            0,
            steps_to_run,
            body,
            (state, sdm_state, diagnostics),
        )
        return state, sdm_state, diagnostics

    donate_argnums = (0, 1) if donate_args else ()
    compiled = jax.jit(run_segment, donate_argnums=donate_argnums)
    compiled.raw = run_segment
    return compiled
