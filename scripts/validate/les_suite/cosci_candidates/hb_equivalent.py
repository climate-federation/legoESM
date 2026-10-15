"""Harness self-check candidate: Holtslag-Boville routed through the plug-in.

Calls the roster's own ``holtslag_boville_turbulence`` on the column the harness
hands it, then reports the interface fluxes HB applied (recovered from its
tendencies). Driven with ``--baseline holtslag_boville`` the column must reproduce
the plain HB single-column run to round-off; any difference is a harness defect.
It is handed the column's actual temperature and vapour (as the SCM turbulence
wrapper is), so cloudy columns are covered too. ``Ri_crit`` (HB's tier-1 parameter)
is traced so the AD path is exercised.
"""
from __future__ import annotations

import sys
from pathlib import Path

from legoesm.atmosphere.les_suite.candidate import tendency_to_flux
from legoesm.atmosphere.physics.turbulence.holtslag_boville import (
    holtslag_boville_turbulence,
)
from legoesm.atmosphere.physics.turbulence.integration import (
    fold_prescribed_surface_fluxes,
)
from legoesm.core.param_overrides import apply_param_overrides
from legoesm.thermo import saturation_specific_humidity

from legoesm import constants

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts" / "run"))
from tune_scm_to_les import _base_turbulence  # noqa: E402

KIND = "scalar"
_BASE = _base_turbulence("holtslag_boville")[0].holtslag_boville
PARAMS = {"Ri_crit": (float(_BASE.Ri_crit), 0.1, 0.9, "1")}


def init_aux(inputs):
    return ()


def fluxes(params, inputs, aux):
    rev = lambda a: a[::-1][None]  # noqa: E731  surface-first (n,) -> top-first (1, n)
    temp = rev(inputs.temp)
    q_v = rev(inputs.q_v)
    p_full, p_half = rev(inputs.p_full), rev(inputs.p_half)
    rho = rev(inputs.rho_full)
    temp_sfc = temp[:, -1]
    q_sfc = saturation_specific_humidity(temp_sfc, p_full[:, -1])
    rho_sfc = rho[:, -1]
    # Same kinematic -> energetic conversion the SCM turbulence wrapper applies to
    # the per-step prescribed flux (lowest-level density).
    cfg = fold_prescribed_surface_fluxes(
        apply_param_overrides(_BASE, params),
        shflx_w_m2=rho_sfc * constants.c_pd * inputs.w_theta_s,
        # latent-ok: mirrors the SCM kinematic-override fold (constant L_v)
        lhflx_w_m2=rho_sfc * constants.L_v * inputs.w_qt_s,  # latent-ok: see above
        evap_kg_m2_s=rho_sfc * inputs.w_qt_s)
    out = holtslag_boville_turbulence(
        rev(inputs.u), rev(inputs.v), temp, q_v, p_full, p_half,
        rev(inputs.z_full), rev(inputs.z_half), temp_sfc, q_sfc, rho, inputs.dt, cfg)
    dtheta = out.dT_dt[0][::-1] / inputs.exner
    dq = out.dq_v_dt[0][::-1]
    return (tendency_to_flux(dtheta, inputs.rho_half, inputs.mass),
            tendency_to_flux(dq, inputs.rho_half, inputs.mass), aux)
