"""Couple the SFNO to the differentiable dycore for training.

Two coupling modes:
- "correction": SFNO adds corrections on top of traditional physics.
- "replacement": SFNO replaces physics entirely.

``make_sfno_step_unified`` returns a drop-in replacement for the
``step_unified`` callable produced by ``PhysicsPipeline.build_step_unified()``.
SFNO parameters (eqx.Module leaves) flow through ``jax.grad``.
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import jax
import jax.numpy as jnp
import equinox as eqx

# Radiation fluxes are O(100 W/m^2); the flux head's raw output (O(1)) is
# scaled by this so it can reach observed magnitudes (mirrors the column
# NN's flux_output_scale).  Untrained -> ~0 (stable).
_SFNO_FLUX_OUTPUT_SCALE = 100.0
if TYPE_CHECKING:
    from legoesm.grids.gaussian import GaussianGrid

from legoesm.ml.sfno import SFNO
from legoesm.ml.channel_packing import PE3DChannelSpec
from legoesm.driver.physics_pipeline import PhysicsOutput


# Re-use the canonical helpers from ``atmosphere.physics.neural_physics``
# rather than maintaining a duplicate parse + PhysicsOutput-kwargs
# logic here.  The two had identical behavior and silently risked
# drift as the ``step_unified`` contract evolved.
from legoesm.atmosphere.physics.neural_physics import (
    build_physics_output_kwargs as _physics_output_kwargs,
    parse_step_unified_tail as _parse_step_unified_tail,
)

_PHYSICS_OUTPUT_FIELDS = set(getattr(PhysicsOutput, "_fields", ()))


class SFNOPhysics(eqx.Module):
    """SFNO wrapper producing PhysicsOutput from grid-space fields.

    Packs prognostic fields into PE3DChannelSpec layout, runs the SFNO,
    and unpacks output as tendencies.

    ``flux_head`` (optional) is a per-column MLP mapping the packed
    normalized state -> 4 radiation fluxes (sw_net_sfc, lw_net_sfc,
    sw_up_toa, lw_up_toa) so the radiation-flux loss can supervise the SFNO
    (replacement) variant.  None -> zeros (state-only training, the legacy
    behavior).  It is a separate float module so it trains alongside the
    SFNO without dragging the Gaussian grid's int leaves into the gradient.
    """

    sfno: SFNO
    grid: "GaussianGrid"  # noqa: F821
    nlev: int = eqx.field(static=True)
    flux_head: eqx.Module = None
    flux_output_scale: float = eqx.field(static=True, default=_SFNO_FLUX_OUTPUT_SCALE)

    def __call__(
        self,
        T: jnp.ndarray,
        u: jnp.ndarray,
        v: jnp.ndarray,
        q_v: jnp.ndarray,
        p_s: jnp.ndarray,
        phis: jnp.ndarray,
        dt: jnp.ndarray,
    ) -> PhysicsOutput:
        """Predict physics tendencies from Gaussian-grid prognostic fields.

        3D fields have shape (n_lat, n_lon, nlev); 2D fields (n_lat, n_lon).
        Radiation flux outputs are zero (SFNO predicts combined tendencies).
        """
        spec = PE3DChannelSpec(nlev=self.nlev)

        # Pack [u, v, T, q_v, lnps, phis] NORMALISED to O(1) so the encoder
        # sees standardised inputs — packing raw fields (T~300, lnps~11)
        # makes the encoder activations (and hence the output tendencies)
        # huge, blowing up the multi-step forward at init even after the
        # output tendency_scale.  Same standardisation the column NN uses.
        lnps = jnp.log(jnp.maximum(p_s, 1.0))
        packed = jnp.concatenate(
            [u / 20.0, v / 20.0, T / 300.0, q_v * 1.0e3,
             ((lnps - 11.5))[..., None], (phis / 5.0e4)[..., None]],
            axis=-1,
        )

        # SFNO forward pass
        out = self.sfno(packed, self.grid)  # (n_lat, n_lon, n_channels)

        # Unpack tendency channels
        dT_dt = out[..., spec.T_slice]
        dq_v_dt = out[..., spec.q_slice]

        zeros_3d = jnp.zeros(T.shape, dtype=T.dtype)
        zeros_2d = jnp.zeros(p_s.shape, dtype=p_s.dtype)

        # Flux head: per-column MLP on the packed normalized state -> 4
        # radiation fluxes (sw_net_sfc, lw_net_sfc, sw_up_toa, lw_up_toa),
        # scaled to W/m^2.  None -> zeros (state-only).  precip / sw_down_toa
        # stay zero (insolation is an external forcing, not predicted here).
        if self.flux_head is not None:
            n_lat_g, n_lon_g, n_ch = packed.shape
            flat = packed.reshape(n_lat_g * n_lon_g, n_ch)
            f = jax.vmap(self.flux_head)(flat) * self.flux_output_scale
            f = f.reshape(n_lat_g, n_lon_g, 4)
            sw_net_sfc, lw_net_sfc = f[..., 0], f[..., 1]
            sw_up_toa, lw_up_toa = f[..., 2], f[..., 3]
        else:
            sw_net_sfc = lw_net_sfc = sw_up_toa = lw_up_toa = zeros_2d

        return PhysicsOutput(
            **_physics_output_kwargs(
                dT_dt=dT_dt,
                dq_v_dt=dq_v_dt,
                dq_c_dt=zeros_3d,
                dq_r_dt=zeros_3d,
                precip=zeros_2d,
                sw_net_sfc=sw_net_sfc,
                lw_net_sfc=lw_net_sfc,
                sw_up_toa=sw_up_toa,
                lw_up_toa=lw_up_toa,
                sw_down_toa=zeros_2d,
                reference_3d=T,
            )
        )


def make_sfno_step_unified(
    sfno_physics: SFNOPhysics,
    mode: Literal["correction", "replacement"] = "correction",
    traditional_step_unified=None,
):
    """Build a step_unified callable that incorporates the SFNO.

    Same signature as ``PhysicsPipeline.build_step_unified()`` output,
    so it can be passed directly to ``build_segment_fn``.

    Parameters
    ----------
    sfno_physics : SFNOPhysics
    mode : "correction" or "replacement"
    traditional_step_unified : callable or None
        Required when mode="correction".
    """
    if mode == "correction" and traditional_step_unified is None:
        raise ValueError(
            "traditional_step_unified is required for mode='correction'"
        )

    def step_unified(need_rad, T, p_s, q_v, q_c, q_r, *args, **kwargs):
        conv_prog, tail = _parse_step_unified_tail(args)
        (
            u,
            v,
            sst,
            sic,
            lat,
            lon,
            day_of_year,
            seconds_of_day,
            dt,
            solar_weights,
            s_0,
            o3_vmr,
            aerosol_od,
            held_dT_rad,
            held_sw_net_sfc,
            held_lw_net_sfc,
            held_sw_up_toa,
            held_lw_up_toa,
            held_sw_down_toa,
        ) = tail
        # SFNO tendency prediction from prognostic fields
        phis = kwargs.get("phis", jnp.zeros_like(p_s))
        sfno_out = sfno_physics(T, u, v, q_v, p_s, phis, dt)
        if "conv_prog" in _PHYSICS_OUTPUT_FIELDS and conv_prog is not None:
            sfno_out = sfno_out._replace(conv_prog=conv_prog)

        if mode == "replacement":
            held_new = (
                held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
                held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
            )
            return sfno_out, held_new

        # mode == "correction": traditional physics + SFNO correction
        trad_args = [need_rad, T, p_s, q_v, q_c, q_r]
        if conv_prog is not None:
            trad_args.append(conv_prog)
        trad_args.extend([
            u, v, sst, sic, lat, lon,
            day_of_year, seconds_of_day, dt,
            solar_weights, s_0,
            o3_vmr, aerosol_od,
            held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
            held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
        ])
        _trad = traditional_step_unified(*trad_args, **kwargs)
        trad_out, held_new = _trad[0], _trad[1]
        # The traditional PhysicsPipeline step returns a 3rd value (the
        # slab-land skin temperature, #325); carry it through so an
        # SFNO-correction run with an active land tile still evolves
        # T_land.  Older 2-tuple steps leave it None (land inert).
        _trad_T_land = _trad[2] if len(_trad) > 2 else None

        corrected = PhysicsOutput(
            **_physics_output_kwargs(
                dT_dt=trad_out.dT_dt + sfno_out.dT_dt,
                dq_v_dt=trad_out.dq_v_dt + sfno_out.dq_v_dt,
                dq_c_dt=trad_out.dq_c_dt + sfno_out.dq_c_dt,
                dq_r_dt=trad_out.dq_r_dt + sfno_out.dq_r_dt,
                precip=trad_out.precip,
                sw_net_sfc=trad_out.sw_net_sfc,
                lw_net_sfc=trad_out.lw_net_sfc,
                sw_up_toa=trad_out.sw_up_toa,
                lw_up_toa=trad_out.lw_up_toa,
                sw_down_toa=trad_out.sw_down_toa,
                reference_3d=trad_out.dT_dt,
                template=trad_out,
                conv_prog=conv_prog,
            )
        )
        return corrected, held_new, _trad_T_land

    return step_unified


def make_sfno_step_unified_latlon(
    sfno_physics: SFNOPhysics,
    w_latlon_to_gauss,
    w_gauss_to_latlon,
):
    """step_unified for a LAT-LON model grid with the SFNO on a Gaussian grid.

    The SFNO's spherical-harmonic transforms need Gaussian quadrature
    latitudes, so the model's lat-lon prognostics are remapped onto the
    SFNO's own Gaussian grid, the SFNO predicts tendencies there, and the
    dT/dt, dq_v/dt tendencies are remapped back to the lat-lon grid.
    Replacement mode only (the SFNO IS the physics; there is no lat-lon
    traditional pipeline to correct against in the WB scale trainer).

    Parameters
    ----------
    sfno_physics : SFNOPhysics
        Wrapper whose ``grid`` is the Gaussian grid the weights target.
    w_latlon_to_gauss, w_gauss_to_latlon : RegridWeights
        Precomputed IDW weights (``compute_latlon_to_voronoi_weights``)
        with ``target_shape`` set to the 2-D destination grid shape, so
        ``regrid_scalar`` returns (n_lat, n_lon[, nlev]) fields directly.
    """
    from legoesm.grids.regridding import regrid_scalar

    def step_unified(need_rad, T, p_s, q_v, q_c, q_r, *args, **kwargs):
        conv_prog, tail = _parse_step_unified_tail(args)
        (
            u, v, sst, sic, lat, lon,
            day_of_year, seconds_of_day, dt,
            solar_weights, s_0, o3_vmr, aerosol_od,
            held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
            held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
        ) = tail

        phis = kwargs.get("phis", jnp.zeros_like(p_s))

        def to_gauss(f):
            return regrid_scalar(f, w_latlon_to_gauss)

        sfno_out = sfno_physics(
            to_gauss(T), to_gauss(u), to_gauss(v), to_gauss(q_v),
            to_gauss(p_s), to_gauss(phis), dt,
        )

        def to_latlon(f):
            return regrid_scalar(f, w_gauss_to_latlon)

        zeros_3d = jnp.zeros_like(T)
        zeros_2d = jnp.zeros_like(p_s)
        out = PhysicsOutput(
            **_physics_output_kwargs(
                dT_dt=to_latlon(sfno_out.dT_dt),
                dq_v_dt=to_latlon(sfno_out.dq_v_dt),
                dq_c_dt=zeros_3d,
                dq_r_dt=zeros_3d,
                precip=zeros_2d,
                sw_net_sfc=zeros_2d,
                lw_net_sfc=zeros_2d,
                sw_up_toa=zeros_2d,
                lw_up_toa=zeros_2d,
                sw_down_toa=zeros_2d,
                reference_3d=T,
            )
        )
        if "conv_prog" in _PHYSICS_OUTPUT_FIELDS and conv_prog is not None:
            out = out._replace(conv_prog=conv_prog)
        held_new = (
            held_dT_rad, held_sw_net_sfc, held_lw_net_sfc,
            held_sw_up_toa, held_lw_up_toa, held_sw_down_toa,
        )
        return out, held_new

    return step_unified
