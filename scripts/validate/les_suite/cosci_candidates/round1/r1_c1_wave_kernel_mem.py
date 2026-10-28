"""Co-scientist round-1 candidate C1_wave_kernel with plume memory (KIND="both",
EXPERIMENTAL).

Identical to ``r1_c1_wave_kernel.py`` (implicit upwind mass flux), except that the
whole face mass-flux profile M(z) is prognostic (user 2026-10-28): each step it
relaxes toward the profile the plume diagnoses,

    M^{n+1} = (M^n + (dt/tau_M) M_diag) / (1 + dt/tau_M),

backward Euler in time, so M^{n+1} stays between M^n and M_diag (>= 0). The relaxed
M is what the closure uses everywhere M enters (transport, plume buoyancy flux in
the TKE, wave kernel). The first step takes M = M_diag (no spin-up). tau_M is a
hypothesis parameter: default 900 s (about one convective turnover), range
300-3600 s. Conservation is unchanged: M only scales flux-form transport.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax.numpy as jnp

_spec = importlib.util.spec_from_file_location(
    "_r1_c1_wave_kernel_base", Path(__file__).with_name("r1_c1_wave_kernel.py"))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

KIND = _base.KIND
PARAMS = {**_base.PARAMS,
          "tau_M": (900.0, 300.0, 3600.0, "s")}   # plume mass-flux memory timescale


def init_aux(inputs):
    return {"e": _base.init_aux(inputs),
            "M": jnp.zeros_like(inputs.z_half),
            "started": jnp.zeros((), inputs.theta_l.dtype)}


def fluxes(params, inputs, aux):
    base_params = {k: v for k, v in params.items() if k != "tau_M"}
    r = inputs.dt / params["tau_M"]

    def relax(m_diag):
        m_mem = (aux["M"] + r * m_diag) / (1.0 + r)
        return jnp.where(aux["started"] > 0.0, m_mem, m_diag)

    fa, fb, e_new, m_used = _base.fluxes_with_mass_flux(base_params, inputs, aux["e"],
                                                        mass_flux_hook=relax)
    return fa, fb, {"e": e_new, "M": m_used,
                    "started": jnp.ones((), inputs.theta_l.dtype)}
