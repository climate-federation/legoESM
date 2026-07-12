"""Large-scale forcing for the plane (double-periodic f-plane) CRM.

Provides the GATE/LBA-style external forcing that SAM applies through
``forcing.f90`` + ``subsidence.f90`` (see
:mod:`legoesm.atmosphere.large_scale_forcing` for the shared upwind operator
and the SAM background):

* **Subsidence** — first-order upwind vertical advection ``-w_ls·∂φ/∂z`` of the
  full potential temperature and **every** tracer (q_v + hydrometeors +
  numbers), exactly as SAM's ``subsidence.f90`` advects all ``micro_field``s,
  not just water vapour.
* **Prescribed horizontal-advection tendencies** ``dθ/dt|_ls`` (onto θ) and
  ``dq_v/dt|_ls`` (onto the q_v tracer slot 0).
* **Wind nudging** (SAM ``donudging_uv``) — relax the DOMAIN-MEAN wind toward a
  target profile on a timescale ``τ`` (``du/dt = -(ū−u_tgt)/τ``, ū = horizontal
  mean), leaving the eddies free.

The forcing is returned as a :class:`PlaneNonHydrostaticTendencies` so the
RCEMIP/GATE/LBA driver simply appends it to the physics-tendency sum
(:func:`legoesm.atmosphere.idealized.land_rce.sum_plane_tendencies`).

Profiles are captured as static ``(nlev,)`` arrays in the returned closure —
correct for the constant / steady forcing used by RCE and the early GATE/LBA
spin-up.  Genuinely time-varying forcing (the full GATE/LBA file forcing) must
thread the current-time profile as a traced argument to avoid recompilation;
that ingestion layer (cases C1-C3) is built on top of this operator.

Positivity: like SAM, this returns raw tendencies and relies on the downstream
integrator/microphysics to keep moisture and hydrometeors non-negative — the
Morrison/Thompson plane path clips every tracer to ``>= 0`` on read each step
(``microphysics/integration.py``), so a subsidence/advection step that would
nudge q_v slightly negative is clamped before it can corrupt a rate (codex
iter-31 F).  The forcing therefore does NOT pre-clamp (which would break the
SSP-RK3 tendency-sum consistency with the other physics sources).
"""

from __future__ import annotations

import jax.numpy as jnp

import numpy as np

from legoesm.atmosphere.large_scale_forcing import (
    subsidence_tendency_top2bottom,
)
from legoesm.atmosphere.sam_case_forcing import (
    interp_forcing_to_levels,
    read_sam_lsf,
)
from legoesm.core.field import Field
from legoesm.core.state import (
    PlaneNonHydrostaticState,
    PlaneNonHydrostaticTendencies,
)
from legoesm.grids.vertical import HeightCoordinate


def _as_profile(arr, nlev, name):
    """Validate + return a ``(nlev,)`` float array, or ``None``."""
    if arr is None:
        return None
    out = jnp.asarray(arr)
    if out.shape != (nlev,):
        raise ValueError(
            f"plane large-scale forcing profile {name!r} has shape "
            f"{tuple(out.shape)}, expected ({nlev},)."
        )
    return out


def surface_kinematic_flux_tendency(flux_s, height_coord: HeightCoordinate):
    """``(nlev,)`` scalar tendency [φ/s] from a PRESCRIBED surface kinematic flux
    ``flux_s`` [φ·m/s], POSITIVE = UPWARD (surface → atmosphere): a positive
    sensible-heat / moisture flux WARMS / MOISTENS the surface cell.

    Injected ONLY on the SURFACE boundary cell — the LAST index in the height
    coordinate's TOP-TO-BOTTOM storage (``z_half[-1] = 0``).  The interior SGS
    vertical diffusion (``compressible_euler_plane._vertical_K_diffusion_full``)
    uses a NO-FLUX (``F=0``) surface interface and documents that "surface/top
    fluxes are injected SEPARATELY by the surface scheme" (SAM
    ``diffuse_scalar_z.f90:61``); this boundary-cell source is EXACTLY that
    injection — NOT a double-count of the interior stencil.  Mass-weighted to
    match the SGS form ``∂_t φ = (1/ρ_ref)·∂_z(ρ_w·F)``::

        tend_sfc = flux_s · ρ_w_sfc / (ρ_ref_sfc · dz_sfc)

    with ``ρ_w_sfc = rho_ref_half[-1]`` the surface-interface reference density,
    ``ρ_ref_sfc = rho_ref[-1]`` the surface-cell density, ``dz_sfc = dz[-1]`` the
    surface-cell thickness.  Pure + AD-safe: ``flux_s`` flows LINEARLY into a
    single cell and the only divisions are by STATIC grid metric (no division by
    a traced value).  This is a boundary SOURCE — it intentionally changes the
    column heat/moisture budget by the surface flux (it is NOT internally
    conservative, unlike the interior SGS).
    """
    rho_ref = jnp.asarray(height_coord.rho_ref)
    rho_w_sfc = jnp.asarray(height_coord.rho_ref_half)[-1]
    dz_sfc = jnp.asarray(height_coord.dz)[-1]
    tend_sfc = jnp.asarray(flux_s) * (rho_w_sfc / rho_ref[-1]) / dz_sfc
    out = jnp.zeros((height_coord.n_levels,), dtype=rho_ref.dtype)
    return out.at[-1].set(tend_sfc.astype(rho_ref.dtype))


def make_plane_ls_forcing_physics(
    height_coord: HeightCoordinate,
    *,
    w_ls=None,
    theta_adv=None,
    qv_adv=None,
    w_theta_sfc=None,
    w_qv_sfc=None,
    u_nudge=None,
    v_nudge=None,
    tau_nudge=None,
    subsidence_on_tracers: bool = True,
):
    """Build a plane physics_fn applying SAM-style large-scale forcing.

    Parameters
    ----------
    height_coord
        Column geometry; supplies ``z_full`` and the reference ``theta_ref``
        used to form the full θ whose vertical gradient subsidence advects.
    w_ls
        ``(nlev,)`` large-scale vertical velocity [m/s], **positive upward**
        (subsidence < 0).  ``None`` disables subsidence.

        FAITHFULNESS CAVEAT (codex iter-55, MEDIUM): this advects the full
        POTENTIAL temperature ``-w_ls·∂θ/∂z`` whereas SAM ``subsidence.f90``
        advects the liquid/ice STATIC-ENERGY ``t`` (``-w_sub·∂t/∂z``). In clear
        air the two are the same adiabatic warming (dry-static-energy and θ
        subsidence agree to the nonlinear exner weighting of the perturbation,
        ~O(0.1%)); they DIVERGE where condensate gradients are large (the
        ``-L·∂qcond/∂z`` term SAM carries in ``t`` but θ does not). The current
        SAM comparison cases all set ``w_ls≡0`` (GATE lsf wls column is zero;
        RCEMIP/RCE impose no large-scale subsidence — it is resolved), so this is
        currently inactive; a case with prescribed cloudy subsidence (e.g.
        DYNAMO) would need a static-energy subsidence path (see FORCING-SUB).
    theta_adv, qv_adv
        ``(nlev,)`` prescribed horizontal-advection tendencies of θ [K/s] and
        q_v [(kg/kg)/s].  ``None`` disables that channel.
    w_theta_sfc, w_qv_sfc
        Scalar PRESCRIBED surface kinematic fluxes of θ [K·m/s] and q_v
        [(kg/kg)·m/s], POSITIVE = UPWARD (surface → atmosphere).  Applied via
        :func:`surface_kinematic_flux_tendency` ONLY to the surface (last) cell —
        the separate surface-scheme injection the no-flux interior SGS leaves room
        for (SAM ``diffuse_scalar_z.f90:61``).  ``None`` disables that channel.
        (The prescribed-FLUX surface BC; a prescribed surface-TEMPERATURE bulk
        path is a separate future channel.)
    u_nudge, v_nudge
        ``(nlev,)`` target wind profiles [m/s] for DOMAIN-MEAN relaxation (SAM
        ``donudging_uv`` — only the horizontal-mean wind is nudged, eddies are
        untouched).  Require ``tau_nudge``.
    tau_nudge
        Nudging timescale [s] (> 0).  ``None`` disables nudging.
    subsidence_on_tracers
        Apply subsidence to every tracer (SAM behaviour).  ``False`` restricts
        it to θ + q_v only (slot 0), e.g. for dry/idealised runs.

    Returns
    -------
    physics_fn : callable
        ``physics_fn(state, grid, height_coord, terrain_metric)`` →
        :class:`PlaneNonHydrostaticTendencies`.
    """
    nlev = height_coord.n_levels
    w_ls = _as_profile(w_ls, nlev, "w_ls")
    theta_adv = _as_profile(theta_adv, nlev, "theta_adv")
    qv_adv = _as_profile(qv_adv, nlev, "qv_adv")
    u_nudge = _as_profile(u_nudge, nlev, "u_nudge")
    v_nudge = _as_profile(v_nudge, nlev, "v_nudge")

    nudging = (u_nudge is not None) or (v_nudge is not None)
    if nudging:
        if tau_nudge is None or tau_nudge <= 0.0:
            raise ValueError(
                "plane large-scale forcing: u_nudge/v_nudge require a "
                f"positive tau_nudge (got {tau_nudge!r})."
            )
    inv_tau = (1.0 / tau_nudge) if nudging else 0.0

    if (w_ls is None and theta_adv is None and qv_adv is None and not nudging
            and w_theta_sfc is None and w_qv_sfc is None):
        raise ValueError(
            "make_plane_ls_forcing_physics: no forcing channel is active "
            "(all of w_ls, theta_adv, qv_adv, w_theta_sfc, w_qv_sfc, u_nudge, "
            "v_nudge are None) — this would add a pure-zero tendency every step. "
            "Configure at least one channel or omit the forcing physics entirely."
        )

    # Prescribed surface kinematic fluxes → a static (nlev,) surface-cell tendency
    # profile baked into the closure (flux is steady; compute once, not per step).
    sfc_theta_tend = (None if w_theta_sfc is None
                      else surface_kinematic_flux_tendency(w_theta_sfc, height_coord))
    sfc_qv_tend = (None if w_qv_sfc is None
                   else surface_kinematic_flux_tendency(w_qv_sfc, height_coord))

    z_full = height_coord.z_full            # (nlev,), top-to-bottom
    theta_ref = height_coord.theta_ref      # (nlev,)

    # Validate the grid host-side (codex iter-31 G): the upwind operator floors
    # dz at 1 m, which would otherwise silently mask a non-monotonic / duplicate
    # vertical grid. The factory runs at setup on a concrete coordinate.
    if w_ls is not None:
        import numpy as _np
        _zf = _np.asarray(z_full)
        if not _np.all(_np.diff(_zf) < 0.0):
            raise ValueError(
                "make_plane_ls_forcing_physics: height_coord.z_full must be "
                "strictly DECREASING with level index (top-to-bottom storage) "
                "for the upwind subsidence operator; got non-monotonic z_full."
            )
    dims_3d = ("y", "x", "z")
    dims_w = ("y", "x", "z_half")
    dims_2d = ("y", "x")
    dims_tr = ("y", "x", "z", "tracer")

    def physics_fn(
        state: PlaneNonHydrostaticState,
        grid,
        hc: HeightCoordinate,
        terrain_metric,
    ) -> PlaneNonHydrostaticTendencies:
        theta_p = state.theta_prime.data            # (ny, nx, nlev)
        tracers = state.tracers.data                # (ny, nx, nlev, n_tr)
        u = state.u.data
        v = state.v.data if state.v is not None else None
        sd = theta_p.dtype
        pd = state.phis.data.dtype
        shape_3d = theta_p.shape
        shape_w = state.w.data.shape
        shape_2d = state.phis.data.shape
        n_tr = tracers.shape[-1] if tracers.ndim >= 4 else 0

        z3 = z_full.reshape(1, 1, nlev)
        d_theta = jnp.zeros(shape_3d, dtype=sd)
        d_tracers = jnp.zeros_like(tracers)
        d_u = jnp.zeros(shape_3d, dtype=sd)
        d_v = jnp.zeros(shape_3d, dtype=sd)

        # --- Subsidence: -w_ls · ∂φ/∂z (upwind) on full θ and all tracers ---
        if w_ls is not None:
            w3 = w_ls.reshape(1, 1, nlev).astype(sd)
            theta_total = theta_ref.reshape(1, 1, nlev).astype(sd) + theta_p
            d_theta = d_theta + subsidence_tendency_top2bottom(
                theta_total, z3, w3,
            )
            if n_tr > 0:
                if subsidence_on_tracers:
                    # ∂/∂z along the nlev axis (-2): move it last, advect, back.
                    tr_t = jnp.moveaxis(tracers, -2, -1)   # (ny,nx,n_tr,nlev)
                    zb = z_full.reshape(1, 1, 1, nlev)
                    wb = w_ls.reshape(1, 1, 1, nlev).astype(sd)
                    sub_t = subsidence_tendency_top2bottom(tr_t, zb, wb)
                    d_tracers = d_tracers + jnp.moveaxis(sub_t, -1, -2)
                else:
                    qv = tracers[..., 0]
                    sub_qv = subsidence_tendency_top2bottom(qv, z3, w3)
                    d_tracers = d_tracers.at[..., 0].add(sub_qv)

        # --- Prescribed horizontal-advection tendencies --------------------
        if theta_adv is not None:
            d_theta = d_theta + theta_adv.reshape(1, 1, nlev).astype(sd)
        if qv_adv is not None and n_tr > 0:
            d_tracers = d_tracers.at[..., 0].add(
                qv_adv.reshape(1, 1, nlev).astype(sd)
            )

        # --- Prescribed surface kinematic fluxes (surface-cell source) -----
        # The no-flux interior SGS leaves the surface flux to this scheme (SAM
        # diffuse_scalar_z.f90:61); only the surface (last) cell is nonzero.
        if sfc_theta_tend is not None:
            d_theta = d_theta + sfc_theta_tend.reshape(1, 1, nlev).astype(sd)
        if sfc_qv_tend is not None and n_tr > 0:
            d_tracers = d_tracers.at[..., 0].add(
                sfc_qv_tend.reshape(1, 1, nlev).astype(sd)
            )

        # --- Wind nudging (SAM donudging_uv) -------------------------------
        # SAM relaxes the DOMAIN-MEAN wind toward the target (nudging.f90:425
        # ``dudt -= (u0(k)-ug0(k))/τ`` with u0 = horizontal mean), applying the
        # SAME correction to every column so the eddies (perturbations from the
        # mean) are left FREE. Nudging the full local wind would spuriously damp
        # resolved convective momentum transport. (jnp.mean is the single-device
        # domain mean; an MPI-sharded plane would need a global reduction.)
        if u_nudge is not None:
            u_mean = jnp.mean(u, axis=(0, 1), keepdims=True)
            d_u = d_u - inv_tau * (
                u_mean - u_nudge.reshape(1, 1, nlev).astype(sd))
        if v_nudge is not None and v is not None:
            v_mean = jnp.mean(v, axis=(0, 1), keepdims=True)
            d_v = d_v - inv_tau * (
                v_mean - v_nudge.reshape(1, 1, nlev).astype(sd))

        return PlaneNonHydrostaticTendencies(
            du_dt=Field(data=d_u, name="du_dt_lsf", dims=dims_3d,
                        units="m/s^2"),
            dv_dt=Field(data=d_v, name="dv_dt_lsf", dims=dims_3d,
                        units="m/s^2"),
            dw_dt=Field(data=jnp.zeros(shape_w, dtype=sd), name="dw_dt_lsf",
                        dims=dims_w, units="m/s^2"),
            dtheta_prime_dt=Field(data=d_theta, name="dtheta_prime_dt_lsf",
                                  dims=dims_3d, units="K/s"),
            drho_prime_dt=Field(data=jnp.zeros(shape_3d, dtype=sd),
                                name="drho_prime_dt_lsf", dims=dims_3d,
                                units="kg/m^3/s"),
            dphis_dt=Field(data=jnp.zeros(shape_2d, dtype=pd),
                           name="dphis_dt_lsf", dims=dims_2d, units="m^2/s^3"),
            dtracers_dt=Field(data=d_tracers, name="dtracers_dt_lsf",
                              dims=dims_tr, units="1/s"),
        )

    return physics_fn


def make_plane_ls_forcing_from_sam_case(
    height_coord: HeightCoordinate,
    lsf_path,
    *,
    day: float = 0.0,
    include_subsidence: bool = True,
    include_advective: bool = True,
    nudge_winds: bool = False,
    tau_nudge=None,
    subsidence_on_tracers: bool = True,
):
    """Build a plane large-scale-forcing physics_fn from a SAM ``lsf`` file.

    Reads + interpolates the SAM large-scale-forcing file onto
    ``height_coord.z_full`` at time ``day`` (steady for GATE_IDEAL; linearly
    time-interpolated for multi-block files) and wires:

    * ``w_ls`` → subsidence,
    * ``dT/dt|_ls`` (``tls``) → θ advective tendency (converted T→θ via exner),
    * ``dq_v/dt|_ls`` (``qls``) → q_v advective tendency.

    The lsf ``u_ls``/``v_ls`` columns are SAM's GEOSTROPHIC reference wind
    (consumed by the Coriolis term via ``HeightCoordinate.u_geo0/v_geo0`` — see
    the D2 plane Coriolis), NOT a nudging target, so they are applied as wind
    nudging ONLY when ``nudge_winds=True`` (then ``tau_nudge`` is required).

    FORCING-T (iter-55): the lsf ``tls`` column is an ABSOLUTE-temperature
    tendency (SAM ``forcing.f90`` adds it to the static-energy ``t``≈tabs with no
    ``/prespot``), so it is converted to legoESM's potential-temperature tendency
    here — ``dθ/dt = (dT/dt)/exner_ref`` — before reaching the operator, whose
    ``theta_adv`` channel is (like the SCM forcing) a genuine θ tendency.
    """
    z_model = np.asarray(height_coord.z_full, dtype=np.float64)
    lsf = read_sam_lsf(lsf_path)
    fp = interp_forcing_to_levels(lsf, z_model, day=day)
    w_ls = jnp.asarray(fp["w_ls"]) if include_subsidence else None
    # T→θ: SAM tls is dT_abs/dt; θ=T/π ⇒ dθ/dt=(dT/dt)/π (π=exner_ref, the
    # large-scale/reference value — pressure perturbations are negligible here).
    exner_ref = np.asarray(height_coord.exner_ref, dtype=np.float64)
    theta_adv = (jnp.asarray(fp["T_adv"] / exner_ref)
                 if include_advective else None)
    qv_adv = jnp.asarray(fp["qv_adv"]) if include_advective else None
    u_nudge = jnp.asarray(fp["u_ls"]) if nudge_winds else None
    v_nudge = jnp.asarray(fp["v_ls"]) if nudge_winds else None
    return make_plane_ls_forcing_physics(
        height_coord,
        w_ls=w_ls, theta_adv=theta_adv, qv_adv=qv_adv,
        u_nudge=u_nudge, v_nudge=v_nudge, tau_nudge=tau_nudge,
        subsidence_on_tracers=subsidence_on_tracers,
    )

