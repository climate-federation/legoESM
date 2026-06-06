"""SAM RCEMIP ozone: the mid-latitude-summer (MLS) standard profile.

gSAM's RRTM radiation reads its trace-gas profiles from the MLS standard
atmosphere stored in ``RUNDATA/rrtmg_lw.nc`` (variable ``AbsorberAmountMLS``)
and scales ozone by a namelist factor (1.0 for RCEMIP). This module bundles
that exact O3 volume-mixing-ratio profile (read directly from the gSAM data
file) so legoESM's RRTMGP radiation can use the SAME ozone as the oracle,
instead of the built-in skewed-Gaussian ``_standard_o3_profile`` (which
over-estimates lower-stratospheric O3 by ~3x — 52 hPa: 4.9 vs MLS 1.49 ppm).

The 59 levels span 1054 hPa (surface) to 0.0097 hPa (mesosphere). Ozone is
interpolated to the model levels LINEARLY in log(O3) vs log(p) and clamped to
the table endpoints outside the tabulated range.

Provenance: extracted from ``gSAM/.../RUNDATA/rrtmg_lw.nc`` with::

    ds = xarray.open_dataset("rrtmg_lw.nc")
    names = [decode(n) for n in ds["AbsorberNames"].values]
    o3 = ds["AbsorberAmountMLS"].values[:, names.index("O3")]   # (59,) VMR
    p  = ds["Pressure"].values[:59]                             # (59,) hPa

This is the SAM/gSAM MLS ozone (what ``nxco2/factor=1`` RCEMIP uses) — NOT the
RCEMIP-protocol analytic ozone (Wing et al. 2018, ``g1·p^g2·exp(-p/g3)``);
the oracle is gSAM, which reads this MLS file. Caveat (codex iter-19): SAM
builds layer-mean O3 from column-integrated trace-gas PATHS (mass-conserving
differencing), whereas this does pointwise log-log VMR interpolation; the two
agree at the table nodes and to high accuracy for a smooth profile on CRM
levels, but are not bit-identical at arbitrary intermediate pressures.
"""

from __future__ import annotations

import jax.numpy as jnp

# MLS standard-atmosphere pressure [hPa] and O3 volume mixing ratio [mol/mol],
# surface -> top, read from gSAM RUNDATA/rrtmg_lw.nc (AbsorberAmountMLS, O3).
_MLS_P_HPA = (
    1053.63, 862.642, 706.272, 578.246, 473.428, 387.61, 317.348, 259.823,
    212.725, 174.164, 142.594, 116.746, 95.584, 78.257, 64.072, 52.457,
    42.948, 35.163, 28.789, 23.571, 19.298, 15.8, 12.936, 10.591, 8.671,
    7.099, 5.812, 4.759, 3.896, 3.19, 2.612, 2.138, 1.751, 1.433, 1.174,
    0.961, 0.787, 0.644, 0.527, 0.432, 0.353, 0.289, 0.237, 0.194, 0.159,
    0.13, 0.106, 0.087, 0.071, 0.058, 0.048, 0.039, 0.032, 0.026, 0.021,
    0.018, 0.014, 0.012, 0.01,
)
_MLS_O3_VMR = (
    1.7351e-08, 1.8594e-08, 2.316e-08, 2.9776e-08, 3.6161e-08, 4.0377e-08,
    4.0419e-08, 3.9723e-08, 3.9278e-08, 3.9966e-08, 4.3242e-08, 6.2232e-08,
    1.4327e-07, 3.8332e-07, 8.6365e-07, 1.4865e-06, 2.2225e-06, 3.2583e-06,
    4.6561e-06, 6.2635e-06, 7.6946e-06, 8.617e-06, 9.335e-06, 9.7664e-06,
    9.9645e-06, 9.8541e-06, 9.4685e-06, 8.8641e-06, 8.0659e-06, 7.1325e-06,
    6.1535e-06, 5.2523e-06, 4.5005e-06, 3.8327e-06, 3.2429e-06, 2.7352e-06,
    2.3146e-06, 1.9814e-06, 1.7157e-06, 1.5011e-06, 1.3296e-06, 1.1901e-06,
    1.0665e-06, 9.6507e-07, 8.8057e-07, 8.114e-07, 7.3923e-07, 6.7913e-07,
    6.2578e-07, 5.7923e-07, 5.371e-07, 4.6702e-07, 4.0964e-07, 3.4091e-07,
    2.7114e-07, 2.1521e-07, 1.8115e-07, 1.5325e-07, 1.596e-07,
)

# Pre-sorted ASCENDING in pressure (top -> surface) for jnp.interp, with the
# interpolation done in log space: x = log(p), y = log(O3).
_LOGP_ASC = jnp.log(jnp.asarray(_MLS_P_HPA[::-1]))
_LOGO3_ASC = jnp.log(jnp.asarray(_MLS_O3_VMR[::-1]))


def mls_ozone_vmr(p_full: jnp.ndarray) -> jnp.ndarray:
    """Interpolate the SAM MLS O3 profile to ``p_full`` [Pa].

    Linear in log(O3) vs log(p); ``jnp.interp`` clamps to the table endpoints
    outside [0.0097, 1054] hPa. Returns O3 VMR [mol/mol], same shape as
    ``p_full``. AD-finite everywhere (log of clamped-positive pressure;
    interp is piecewise-linear).
    """
    log_p = jnp.log(jnp.clip(p_full, 1.0e-2) / 100.0)   # Pa -> hPa -> log
    log_o3 = jnp.interp(log_p, _LOGP_ASC, _LOGO3_ASC)
    return jnp.exp(log_o3)
