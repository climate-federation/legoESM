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

if TYPE_CHECKING:
    from legoesm.grids.gaussian import GaussianGrid

from legoesm.ml.sfno import SFNO, SFNOConfig
from legoesm.ml.channel_packing import PE3DChannelSpec
from legoesm.driver.physics_pipeline import PhysicsOutput


_OPTIONAL_3D_OUTPUT_FIELDS = (
    "du_dt",
    "dv_dt",
    "dq_i_dt",
    "dq_s_dt",
    "dq_g_dt",
    "dN_c_dt",
    "dN_r_dt",
    "dN_i_dt",
)
_PHYSICS_OUTPUT_FIELDS = set(getattr(PhysicsOutput, "_fields", ()))


def _parse_step_unified_tail(args):
    """Support both legacy and conv_prog-extended step_unified signatures."""
    if len(args) == 19:
        return None, args
    if len(args) == 20:
        return args[0], args[1:]
    raise TypeError(
        "step_unified expected 19 positional tail arguments "
        "(legacy) or 20 (with conv_prog)"
    )


def _physics_output_kwargs(
    *,
    dT_dt,
    dq_v_dt,
    dq_c_dt,
    dq_r_dt,
    precip,
    sw_net_sfc,
    lw_net_sfc,
    sw_up_toa,
    lw_up_toa,
    sw_down_toa,
    reference_3d,
    template=None,
    conv_prog=None,
):
    kwargs = dict(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        precip=precip,
        sw_net_sfc=sw_net_sfc,
        lw_net_sfc=lw_net_sfc,
        sw_up_toa=sw_up_toa,
        lw_up_toa=lw_up_toa,
        sw_down_toa=sw_down_toa,
    )
    zeros_3d = jnp.zeros_like(reference_3d)
    for field_name in _OPTIONAL_3D_OUTPUT_FIELDS:
        if field_name in _PHYSICS_OUTPUT_FIELDS:
            kwargs[field_name] = (
                getattr(template, field_name, zeros_3d)
                if template is not None else zeros_3d
            )
    if "conv_prog" in _PHYSICS_OUTPUT_FIELDS:
        conv_prog_value = (
            getattr(template, "conv_prog", conv_prog)
            if template is not None else conv_prog
        )
        if conv_prog_value is None:
            conv_prog_value = jnp.asarray(0.0, dtype=reference_3d.dtype)
        kwargs["conv_prog"] = conv_prog_value
    return kwargs


class SFNOPhysics(eqx.Module):
    """SFNO wrapper producing PhysicsOutput from grid-space fields.

    Packs prognostic fields into PE3DChannelSpec layout, runs the SFNO,
    and unpacks output as tendencies.
    """

    sfno: SFNO
    grid: "GaussianGrid"  # noqa: F821
    nlev: int = eqx.field(static=True)

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

        # Pack: [u(nlev), v(nlev), T(nlev), q_v(nlev), lnps, phis]
        lnps = jnp.log(jnp.maximum(p_s, 1.0))
        packed = jnp.concatenate(
            [u, v, T, q_v, lnps[..., None], phis[..., None]],
            axis=-1,
        )

        # SFNO forward pass
        out = self.sfno(packed, self.grid)  # (n_lat, n_lon, n_channels)

        # Unpack tendency channels
        dT_dt = out[..., spec.T_slice]
        dq_v_dt = out[..., spec.q_slice]

        zeros_3d = jnp.zeros(T.shape, dtype=T.dtype)
        zeros_2d = jnp.zeros(p_s.shape, dtype=p_s.dtype)

        return PhysicsOutput(
            **_physics_output_kwargs(
                dT_dt=dT_dt,
                dq_v_dt=dq_v_dt,
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


def build_sfno_physics(
    grid: "GaussianGrid",
    nlev: int,
    config: SFNOConfig | None = None,
    *,
    key: jax.Array,
) -> SFNOPhysics:
    """Build an SFNOPhysics from a grid and vertical resolution.

    If *config* is None, a default SFNOConfig is created with channel
    counts derived from PE3DChannelSpec(nlev) and residual_prediction
    disabled (output = tendencies, not states).
    """
    spec = PE3DChannelSpec(nlev=nlev)
    n_ch = spec.n_channels
    if config is None:
        config = SFNOConfig(
            in_channels=n_ch,
            out_channels=n_ch,
            residual_prediction=False,
        )
    sfno = SFNO(config, grid, key=key)
    return SFNOPhysics(sfno=sfno, grid=grid, nlev=nlev)


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
        trad_out, held_new = traditional_step_unified(*trad_args, **kwargs)

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
        return corrected, held_new

    return step_unified
