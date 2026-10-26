"""BechtoldConfig.use_ifs_ascent: the IFS convection as CUMASTRN runs it.

Wraps the faithful OpenIFS 8f6f722 port (:mod:`._ifs_cumastrn`, oracle-matched
in tests/atmosphere/test_bechtold_ifs_oracle.py) for the legoESM physics
pipeline.  Everything that is NOT the oracle lives here, at the seam:

* inputs: PTENT/PTENQ (total tendency before convection) = the dynamics +
  radiation pair the process-split pipeline has (no vertical diffusion);
  PTENTA/PTENQA = dynamics; PLITOT = 0 (environmental condensate is not
  passed); omega None -> 0; surface heat/moisture fluxes turned into the IFS
  sign (PAHFS = -SHF, PQHFL = -LHF/L_v); one grid length for every column
  (PGAW from ``config.dx_m``); geopotential surface-relative and hydrostatic;
  PQSEN from SATUR (mixed phase); LDSHCV = .TRUE.;
* outputs: convective T/q/u/v tendency = CUMASTRN's increment of
  PTENT/PTENQ/PTENU/PTENV; detrained condensate (liquid + ice, PLUDE) -> the
  cloud-water source; precipitation generated in the column (ZDMFUP) plus the
  detrained rain and snow (PSNDE) -> the rain source, so the host gets back
  exactly the water CUDTDQN removes from the vapour.

No other BechtoldConfig field reaches this path: a non-default value of any
of them raises (they configure the legacy reduction only).
"""
from __future__ import annotations

import math

import jax.numpy as jnp

from legoesm.atmosphere.physics._shared import virtual_temperature
from legoesm.constants import R_d

from ._ifs_cumastrn import cumastrn
from ._ifs_cuthermo import lego_thermo, satur_qsat

__physics_contract__ = {
    "summary": (
        "Seam between the legoESM pipeline and the faithful CUMASTRN port "
        "(IFS cumulus convection, OpenIFS 8f6f722): input/output mapping only."
    ),
    "inputs": {
        "T": "K", "q_v": "kg kg-1", "p_full": "Pa", "p_half": "Pa",
        "u": "m s-1", "v": "m s-1", "omega": "Pa s-1",
        "shf_w_m2": "W m-2 (positive upward)", "lhf_w_m2": "W m-2 (positive upward)",
        "land_frac": "1", "dT_dt_adv": "K s-1", "dq_dt_adv": "kg kg-1 s-1",
        "dT_dt_other": "K s-1", "dq_dt_other": "kg kg-1 s-1", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K s-1", "dq_v_dt": "kg kg-1 s-1", "dq_c_conv_dt": "kg kg-1 s-1",
        "dq_r_conv_dt": "kg kg-1 s-1", "du_dt_conv": "m s-2", "dv_dt_conv": "m s-2",
        "cape": "J kg-1", "convective_mask": "1 (bool)", "M_u": "kg m-2 s-1",
    },
    "sign_convention": (
        "pipeline surface fluxes positive upward are negated into the IFS "
        "PAHFS/PQHFL sign (negative = upward); mass fluxes positive upward."
    ),
    "conserves": ["moisture"],
    "differentiable": True,
    "reference": "OpenIFS 8f6f722 cumastrn.F90 and callees",
    "idealized_test": "tests/atmosphere/test_bechtold_ifs_oracle.py",
}

# BechtoldConfig fields this path reads; every other field must stay default.
_READ_FIELDS = ("use_ifs_ascent", "dx_m")


def _hydrostatic_geopotential(T, q_v, p_half):
    """Surface-relative hydrostatic geopotential, surface-last.

    geo_half[:, nlev] = 0; layer increments R_d*Tv*ln(p_lo/p_hi) are summed
    from the surface upward; geo_full is the half-level midpoint.
    """
    geo_half = jnp.zeros_like(p_half)
    Tv = virtual_temperature(T, q_v)
    dphi = R_d * Tv * jnp.log(p_half[:, 1:] / p_half[:, :-1])
    geo_half = geo_half.at[:, :-1].set(jnp.cumsum(dphi[:, ::-1], axis=1)[:, ::-1])
    geo_full = 0.5 * (geo_half[:, :-1] + geo_half[:, 1:])
    return geo_full, geo_half


def _check_config(config):
    default = type(config)()
    set_ = []
    for f in config._fields:
        if f in _READ_FIELDS:
            continue
        a, b = getattr(config, f), getattr(default, f)
        try:
            same = bool(a == b) if isinstance(a, (bool, int, float, str, tuple, type(None))) else a is b
        except Exception:   # traced leaf: cannot be shown to be the default
            same = False
        if not same:
            set_.append(f)
    if set_:
        raise ValueError(
            "BechtoldConfig.use_ifs_ascent runs the IFS scheme as CUMASTRN does; "
            f"these fields configure only the legacy reduction and must stay default: {set_}")
    if not (isinstance(config.dx_m, (int, float)) and config.dx_m > 0.0):
        raise ValueError("BechtoldConfig.use_ifs_ascent needs the grid length dx_m > 0 "
                         "(cumastrn's ZDX); got %r" % (config.dx_m,))


def ifs_faithful_convection(T, q_v, p_full, p_half, u, v, dt, config, shf_w_m2,
                            lhf_w_m2, land_frac, dT_dt_adv, dq_dt_adv, dT_dt_other,
                            dq_dt_other, omega, out_ctor):
    """Run CUMASTRN on the pipeline's columns.  Returns (ConvectionOutput, M_u)."""
    _check_config(config)
    if shf_w_m2 is None or lhf_w_m2 is None or land_frac is None:
        raise ValueError("use_ifs_ascent needs shf_w_m2, lhf_w_m2 and land_frac "
                         "(cubasen's surface parcel and cumastrn's land switches)")
    th = lego_thermo()
    zero = jnp.zeros_like(T)
    z = lambda a: zero if a is None else a
    geo_full, geo_half = _hydrostatic_geopotential(T, q_v, p_half)
    ptent, ptenq = z(dT_dt_other), z(dq_dt_other)
    ncol = T.shape[0]
    pgaw = jnp.full((ncol,), (config.dx_m / (2.0 * th.RA)) ** 2 / math.pi, T.dtype)
    o = cumastrn(
        th, dt, pgaw, T, q_v, u, v, zero, z(omega), satur_qsat(th, T, p_full),
        -shf_w_m2, -lhf_w_m2 / th.RLVTT, p_full, p_half, geo_full, geo_half,
        jnp.asarray(land_frac, T.dtype), ptent, ptenq, zero, zero, z(dT_dt_adv),
        z(dq_dt_adv))
    g_dp = th.RG / (p_half[:, 1:] - p_half[:, :-1])
    out = out_ctor(
        dT_dt=o.ptent - ptent,
        dq_v_dt=o.ptenq - ptenq,
        dq_c_conv_dt=g_dp * o.plude,
        cape=o.pcape,
        convective_mask=o.ldcum,
        du_dt_conv=o.ptenu,
        dv_dt_conv=o.ptenv,
        dq_r_conv_dt=g_dp * (o.pdmfup + o.psnde[:, 0] + o.psnde[:, 1]),
    )
    return out, o.pmfu
