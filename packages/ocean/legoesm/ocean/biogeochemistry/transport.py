"""Operator-split transport of biogeochemical tracers on the lat-lon / tripole C-grid.

The physics step (``LatLonCGridOceanModel.step``) advects and mixes T and S
only.  This module applies the SAME two operators to an arbitrary stack of
passive tracers AFTER the physics step, using what the step exports on the
state:

* ``state.mass_flux_u`` / ``mass_flux_v`` / ``mass_flux_w`` — the h*u, h*v, w
  the step used for its own tracer transport (populated when
  ``LatLonCGridOceanConfig.store_mass_flux`` is True; the standard tripole
  card sets it).
* ``state.vmix_K_v`` — the composed vertical tracer diffusivity the step's
  implicit solve used (closure + EVD + IWM + DDM) at interior W-levels
  ``(n_lat, n_lon, nlev-1)``; populated when ``store_vmix_K`` is True (the
  driver's ``--biogeo`` turns it on).  Falls back to ``tke_avt`` on cards
  that carry NEMO's closure coefficients.

so the BGC sees the physics' own advecting velocities and mixing, with
first-order operator splitting (the NEMO/TOP arrangement).  Nothing in the
physics step is touched: a run with BGC off is bit-identical to a run without
this module.

Numerics are the ocean step's own, imported from it:
``_compute_advection_flux_div`` / ``_ssp_rk3_tracer_step`` (flux form, the
card's ``tracer_advection`` scheme) and ``implicit_vertical_diffusion_ocean``
(backward Euler), on partial-cell layer thicknesses, so ``sum(h * c)`` is
conserved to round-off by each operator.

Ported from the in-step implementation on ``ggn/bgc_modelling`` (the
``_step_impl`` hooks of 2026-10-13 and ``_transport_marbl_tracers``) into a
standalone, jit-compiled class.

Debug switches, read when the transporter is built:
    LEGOESM_BGC_NO_ADV=1    skip advection
    LEGOESM_BGC_NO_VMIX=1   skip vertical mixing
"""
from __future__ import annotations

import os
from functools import partial

import jax
import jax.numpy as jnp

from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    min_cell_to_uface, min_cell_to_vface)
from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
    _compute_advection_flux_div, _ssp_rk3_tracer_step)
from legoesm.ocean.physics.vertical_mixing.implicit_solver import (
    build_dz_half, implicit_vertical_diffusion_ocean)
from legoesm.ocean.vertical import compute_layer_thickness


def _as_array(x, dtype):
    return jnp.asarray(getattr(x, "data", x), dtype=dtype)


class TracerTransporter:
    """Advect + vertically mix a ``(n_tracers, n_lat, n_lon, nlev)`` stack.

    Build ONCE per run (it holds the grid, the coordinate and the card's
    scheme choices; ``self`` is the static jit argument, as in the model).
    """

    def __init__(self, grid, z_coord, model_config, land_mask_2d, *,
                 k_bg: float = 1.0e-5):
        self.grid = grid
        self.z_coord = z_coord
        self.config = model_config
        self.tracer_advection = str(model_config.tracer_advection)
        self.tracer_time_integrator = str(
            getattr(model_config, "tracer_time_integrator", "euler"))
        self.min_wc = getattr(model_config, "min_water_column_m", None)
        self.k_bg = float(k_bg)
        self.do_adv = os.environ.get("LEGOESM_BGC_NO_ADV", "0") != "1"
        self.do_vmix = os.environ.get("LEGOESM_BGC_NO_VMIX", "0") != "1"

        lm = jnp.asarray(land_mask_2d) > 0.5
        nlev = int(z_coord.n_levels) if hasattr(z_coord, "n_levels") \
            else int(jnp.asarray(z_coord.z_full_ref).shape[-1])
        is_active = getattr(z_coord, "is_active", None)
        if is_active is None:
            act = jnp.broadcast_to(lm[..., None], lm.shape + (nlev,))
        else:
            act = jnp.asarray(is_active, dtype=bool)
            if act.shape[-1] == 1:
                act = jnp.broadcast_to(act, lm.shape + (nlev,))
            act = act & lm[..., None]
        self.active_3d = act
        nwet = jnp.sum(act, axis=-1).astype(jnp.int32)
        self.k_bottom = jnp.clip(nwet - 1, 0, nlev - 1)
        self.dz_ref = jnp.asarray(z_coord.dz_ref)
        self.n_wet_cells = int(jnp.sum(act))

    # -- geometry helpers -------------------------------------------------
    def thickness(self, state):
        return compute_layer_thickness(state.eta.data, state.H_bathy.data,
                                       self.z_coord,
                                       min_water_column_m=self.min_wc)

    # -- the operator -------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def step(self, tracers, h_old, state_new, dt):
        """One operator-split transport step.

        tracers   (n_tracers, n_lat, n_lon, nlev); land / sub-bottom values pass through
        h_old     layer thickness BEFORE the physics step (self.thickness(state_old))
        state_new ocean state AFTER the physics step (fluxes, K_H, eta)
        """
        dtype = tracers.dtype
        act = self.active_3d
        act_f = act.astype(dtype)
        h_old = jnp.asarray(h_old, dtype=dtype)
        h_new = self.thickness(state_new).astype(dtype)

        if self.do_adv:
            if (state_new.mass_flux_u is None or state_new.mass_flux_v is None
                    or state_new.mass_flux_w is None):
                raise ValueError(
                    "TracerTransporter: the physics step exported no mass fluxes; "
                    "the card needs LatLonCGridOceanConfig.store_mass_flux=True.")
            mfu = _as_array(state_new.mass_flux_u, dtype)
            mfv = _as_array(state_new.mass_flux_v, dtype)
            mfw = _as_array(state_new.mass_flux_w, dtype)
            h_u_old = min_cell_to_uface(h_old)
            h_v_old = min_cell_to_vface(h_old, self.grid)

            def _advect(c):
                if self.tracer_time_integrator in ("rk3", "rk3_ws"):
                    c_new = _ssp_rk3_tracer_step(
                        c, self.tracer_advection, mfu, mfv, mfw,
                        h_old, h_new, h_u_old, h_v_old, self.grid, dt, act_f)
                else:
                    dh, dv = _compute_advection_flux_div(
                        c, self.tracer_advection, mfu, mfv, mfw,
                        h_old, h_u_old, h_v_old, self.grid, dt,
                        recon_fill_mask=act_f)
                    c_new = (h_old * c - dt * (dh + dv)) / jnp.maximum(h_new, 1e-10)
                return jnp.where(act, c_new, c)

            tracers = jax.lax.map(_advect, tracers)

        if self.do_vmix:
            K_src = getattr(state_new, "vmix_K_v", None)
            if K_src is None:
                K_src = getattr(state_new, "tke_avt", None)     # NEMO carried-coefficient cards
            if K_src is None:
                raise ValueError(
                    "TracerTransporter: no vertical diffusivity on the state. The "
                    "card needs LatLonCGridOceanConfig.store_vmix_K=True (the "
                    "driver's --biogeo sets it) or a TKE card carrying tke_avt.")
            K = jnp.maximum(_as_array(K_src, dtype), self.k_bg)
            K = K * (act[..., :-1] & act[..., 1:]).astype(dtype)   # no flux across the floor
            dz = jnp.where(act, jnp.maximum(h_new, 1e-3), self.dz_ref.astype(dtype))
            dzh = build_dz_half(dz)
            kb = self.k_bottom[..., None]

            def _mix(c):
                bottom = jnp.take_along_axis(c, kb, axis=-1)
                filled = jnp.where(act, c, bottom)          # flat below the floor
                m = implicit_vertical_diffusion_ocean(filled, K, dz, dzh, dt)
                return jnp.where(act, m, c)

            tracers = jax.lax.map(_mix, tracers)

        return tracers

    # -- diagnostics --------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def wet_volume(self, state, area2d):
        """sum(h * area) over wet cells [m^3] — the denominator of the mean concentration."""
        h = self.thickness(state)
        return jnp.sum(jnp.where(self.active_3d, h, 0.0) * jnp.asarray(area2d)[..., None])

    @partial(jax.jit, static_argnums=(0,))
    def inventory(self, tracers, state, area2d):
        """sum(h * c * area) per tracer, wet cells only — conservation check."""
        h = self.thickness(state)
        w = jnp.where(self.active_3d, h, 0.0) * jnp.asarray(area2d)[..., None]
        return jnp.einsum("tijk,ijk->t", tracers, w)
