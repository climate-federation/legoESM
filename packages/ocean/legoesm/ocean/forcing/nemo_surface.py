"""NEMO-literal coupled surface means and native-face stress conversion."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm.core.source_rounding import nemo_source_round as _sr
from legoesm.ocean.eos import nemo_potential_temperature_from_conservative


class NemoSurfaceMeans(NamedTuple):
    ssu: jax.Array
    ssv: jax.Array
    sst: jax.Array
    sss: jax.Array
    ssh: jax.Array
    e3t: jax.Array
    frq: jax.Array


def _surface_temperature(temperature, salinity, use_conservative_temperature):
    temperature = _sr(jnp.asarray(temperature, dtype=jnp.float64))
    salinity = _sr(jnp.asarray(salinity, dtype=jnp.float64))
    if use_conservative_temperature:
        return _sr(nemo_potential_temperature_from_conservative(temperature, salinity))
    return temperature


def nemo_sbc_ssm_step(
    prior,
    *,
    kt,
    nit000,
    forcing_stride,
    restart_means_present,
    use_conservative_temperature,
    remove_inverse_barometer,
    u_before,
    v_before,
    temperature_now,
    salinity_now,
    ssh_now,
    ssh_inverse_barometer,
    e3t_reference,
    r3t_now,
    surface_mask,
    shortwave_fraction_top,
):
    """Execute one stateful ``sbc_ssm`` update in NEMO source order."""
    if not isinstance(forcing_stride, jax.core.Tracer) and forcing_stride < 2:
        raise ValueError("stateful sbc_ssm requires forcing_stride >= 2")
    u = _sr(jnp.asarray(u_before, dtype=jnp.float64))
    v = _sr(jnp.asarray(v_before, dtype=jnp.float64))
    salinity = _sr(jnp.asarray(salinity_now, dtype=jnp.float64))
    temperature = _surface_temperature(temperature_now, salinity, use_conservative_temperature)
    ssh = _sr(jnp.asarray(ssh_now, dtype=jnp.float64))
    ssh_ib = _sr(jnp.asarray(ssh_inverse_barometer, dtype=jnp.float64))
    ssh_instant = _sr(ssh - ssh_ib) if remove_inverse_barometer else ssh
    e3t_0 = _sr(jnp.asarray(e3t_reference, dtype=jnp.float64))
    r3t = _sr(jnp.asarray(r3t_now, dtype=jnp.float64))
    tmask = _sr(jnp.asarray(surface_mask, dtype=jnp.float64))
    e3t_instant = _sr(e3t_0 * _sr(_sr(1.0) + _sr(r3t * tmask)))
    frq = _sr(jnp.asarray(shortwave_fraction_top, dtype=jnp.float64))
    first = jnp.logical_and(
        jnp.asarray(kt) == jnp.asarray(nit000),
        jnp.logical_not(jnp.asarray(restart_means_present)),
    )
    reset = jnp.mod(jnp.asarray(kt) - 2, jnp.asarray(forcing_stride)) == 0
    initial = _sr(jnp.asarray(forcing_stride - 1, dtype=jnp.float64))
    zero = _sr(jnp.zeros_like(temperature))
    instant = NemoSurfaceMeans(u, v, temperature, salinity, ssh_instant, e3t_instant, frq)
    base = NemoSurfaceMeans(
        *(
            _sr(jnp.where(first, _sr(initial * value), jnp.where(reset, zero, _sr(previous))))
            for value, previous in zip(instant, prior, strict=True)
        )
    )
    accumulated = NemoSurfaceMeans(
        _sr(base.ssu + u),
        _sr(base.ssv + v),
        _sr(base.sst + temperature),
        _sr(base.sss + salinity),
        _sr(_sr(base.ssh + ssh) - ssh_ib) if remove_inverse_barometer else _sr(base.ssh + ssh),
        _sr(base.e3t + e3t_instant),
        _sr(base.frq + frq),
    )
    divide = jnp.mod(jnp.asarray(kt) - 1, jnp.asarray(forcing_stride)) == 0
    coefficient = _sr(_sr(1.0) / _sr(jnp.asarray(forcing_stride, dtype=jnp.float64)))
    return NemoSurfaceMeans(
        *(_sr(jnp.where(divide, _sr(value * coefficient), value)) for value in accumulated)
    )


def nemo_t_stress_to_uv(
    utau, vtau, utau_east, vtau_north, tmask, tmask_east, tmask_north, umask, vmask
):
    """Convert same-step T-point stresses to NEMO native U/V faces."""
    values = [
        _sr(jnp.asarray(value, dtype=jnp.float64))
        for value in (
            utau,
            vtau,
            utau_east,
            vtau_north,
            tmask,
            tmask_east,
            tmask_north,
            umask,
            vmask,
        )
    ]
    utau, vtau, utau_east, vtau_north, tmask, tmask_east, tmask_north, umask, vmask = values
    out_u = _sr(_sr(_sr(0.5) * _sr(utau + utau_east)) * _sr(2.0 - umask))
    out_v = _sr(_sr(_sr(0.5) * _sr(vtau + vtau_north)) * _sr(2.0 - vmask))
    return _sr(out_u * _sr(jnp.maximum(tmask, tmask_east))), _sr(
        out_v * _sr(jnp.maximum(tmask, tmask_north))
    )


__all__ = ("NemoSurfaceMeans", "nemo_sbc_ssm_step", "nemo_t_stress_to_uv")
