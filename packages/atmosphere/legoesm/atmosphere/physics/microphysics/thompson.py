"""Thompson microphysics: host adapter of the faithful WRF v4.8.0 port.

``microphysics: thompson`` runs :func:`.thompson_wrf.thompson_wrf_step`, the
port of WRF ``phys/module_mp_thompson.F`` (tag v4.8.0, commit
06d4240ae989cc3e50af412bb472df3d9048783c, ``mp_physics=8``) matched to the
running oracle in ``tests/atmosphere/microphysics/unit/
test_thompson_wrf_oracle.py``.  This module only maps legoESM's column state
onto the oracle's representation and back (votes H1-H9, J1-J10):

* species: specific humidities [kg/kg moist air] <-> mixing ratios per kg DRY
  air, ``r_x = q_x / (1 - q_tot)`` and back ``q_x = r_x * (1 - q_tot)`` with
  the INPUT total water of the six species: the host holds its layer mass
  fixed through physics, so this is the conversion that keeps the host's
  water budget equal to the port's own (vote V1b, measured; supersedes H3);
* numbers: per-volume ``N_r`` [1/m^3] and per-kg ``N_i`` -> per kg dry air with the scheme's OWN air density
  (:func:`.thompson_wrf.thompson_air_density`, input state), so the scheme's
  ``n * rho`` recovers the host number exactly (J4); back with the same
  input factors, the basis of the returned species (vote V2b);
* droplet number: WRF's prescribed ``Nt_c`` = 100e6 /m^3 as in mp_physics=8
  (user decision 2026-11-04); the carried ``N_c`` is not read and
  ``dN_c_dt`` is zero (no prognostic droplet number, J5);
* layer mass: the port's layer thickness is derived from the HOST layer mass
  ``dp/g`` (``p_half``; on the non-hydrostatic lanes their ``rho*dz`` by
  construction), ``dz = dp / (g * rho_dry * (1 + r_tot))``, so the port's
  masses ``rho_dry * dz`` are the host's dry masses exactly (vote V3b,
  supersedes J9; the lane's ``dz`` is not read);
* orientation: top-down host columns <-> the oracle's bottom-up ones, the
  resolved vertical velocity ``w``
  [m/s] (H2/J6; the oracle reads it only for its low-level cloud-water
  sedimentation switch);
* tendencies: ``(new - old) / dt`` of the converted updated state, T from the
  scheme's updated temperature (H4/J10); surface precipitation per step / dt,
  rain liquid and snow + graupel + ice frozen (H5).

The scheme keeps WRF's hard switches (user-approved exception, 2026-11-03).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import (
    SED_CFL_SUBSTEPS_MAX_LIMIT,
    ThompsonConfig,
)
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
)
from legoesm.atmosphere.physics.microphysics.thompson_wrf import (
    thompson_air_density,
    thompson_wrf_step,
)

__physics_contract__ = {
    "summary": (
        "Host adapter of the WRF v4.8.0 Thompson port (mp_physics=8): "
        "specific humidity <-> dry mixing ratio, per-volume <-> per-kg "
        "numbers, top-down <-> bottom-up, WRF's fixed droplet number."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "q_c..q_g": "kg/kg",
        "N_r": "1/m^3", "N_i": "1/kg", "p_full": "Pa", "p_half": "Pa",
        "w": "m/s", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_*_dt": "kg/kg/s", "dN_r_dt": "1/(m^3 s)",
        "dN_i_dt": "1/(kg s)", "precipitation": "kg/m^2/s",
    },
    "sign_convention": (
        "Tendencies positive into the species; precipitation >= 0 leaves the "
        "column at the bottom (positive into the surface)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "WRF v4.8.0 phys/module_mp_thompson.F (Thompson et al. 2008)",
    "idealized_test": "tests/unit/test_thompson_adapter.py",
}


def _flip(x):
    return x[:, ::-1]


def thompson_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt,
    config: ThompsonConfig = ThompsonConfig(),
    *,
    w: jax.Array | None = None,
) -> MicrophysicsOutput:
    """One WRF Thompson step on ``(ncol, nlev)`` top-down host columns.

    ``rho``, ``dz`` and ``N_c`` are not read: the scheme uses its own density
    and the layer thickness of the host layer mass ``p_half`` (module
    docstring).  ``w`` and ``p_half`` are required (no fallback).
    """
    del rho, dz
    n_sed = int(config.sed_substeps_max)
    if not 1 <= n_sed <= SED_CFL_SUBSTEPS_MAX_LIMIT:
        raise ValueError(
            f"ThompsonConfig.sed_substeps_max={n_sed} must lie in "
            f"[1, {SED_CFL_SUBSTEPS_MAX_LIMIT}]")
    if w is None:
        raise ValueError(
            "microphysics='thompson' (WRF port) needs the resolved vertical "
            "velocity w; this lane did not supply it")
    if p_half is None:
        raise ValueError(
            "microphysics='thompson' (WRF port) needs p_half, the host layer "
            "mass it derives the port's layer thickness from")
    if not jax.config.read("jax_enable_x64"):
        raise ValueError(
            "microphysics='thompson' runs the fp64-certified WRF port as an "
            "fp64 island, which needs jax_enable_x64 (vote U9a)")
    h = hydrometeors
    if h.N_r is None or h.N_i is None:
        raise ValueError(
            "microphysics='thompson' needs the carried N_r and N_i")
    # Run the oracle-certified fp64 port as an fp64 island (vote J11a): its
    # NumPy float64 constants would otherwise promote an fp32 state mid-step.
    # Tendencies are cast back to the state dtype.
    out_dtype = T.dtype
    up = lambda x: jnp.asarray(x, jnp.float64)                 # noqa: E731
    T, q_v, p_full, p_half, w = up(T), up(q_v), up(p_full), up(p_half), up(w)
    dt = up(dt)                     # may be traced (coupled pipeline jit, X1a)
    h = HydrometeorState(*(None if x is None else up(x) for x in h))
    qs = (q_v, h.q_c, h.q_i, h.q_r, h.q_s, h.q_g)
    dry = 1.0 - sum(qs)                       # 1 - q_tot (H3)
    r = [_flip(q / dry) for q in qs]
    Tb, pb = _flip(T), _flip(p_full)
    rho_p = thompson_air_density(Tb, pb, r[0])
    # host layer mass dp/g = rho_dry * dz * (1 + r_tot) = rho_dry * dz / dry
    dzb = _flip(p_half[:, 1:] - p_half[:, :-1]) * _flip(dry) / (constants.g * rho_p)
    o = thompson_wrf_step(
        Tb, pb, dzb, _flip(w), *r,
        _flip(h.N_i / dry),
        _flip(h.N_r) / rho_p, dt,
        n_sed_max=n_sed)
    r_new = (o.qv, o.qc, o.qi, o.qr, o.qs, o.qg)
    q_new = [_flip(x) * dry for x in r_new]   # input dry factor (V1b)
    rate = [(qn - qo) / dt for qn, qo in zip(q_new, qs)]
    dq_v, dq_c, dq_i, dq_r, dq_s, dq_g = rate
    # numbers back: per kg dry -> per kg moist (N_i) / per m^3 (N_r) with the
    # input factors, the basis of the returned species (vote V2b)
    N_i_new = _flip(o.ni) * dry
    N_r_new = _flip(o.nr * rho_p)
    frozen = (o.pptsnow + o.pptgraul + o.pptice) / dt
    cast = lambda x: x.astype(out_dtype)                       # noqa: E731
    return jax.tree_util.tree_map(cast, MicrophysicsOutput(
        dT_dt=(_flip(o.T) - T) / dt,
        dq_v_dt=dq_v, dq_c_dt=dq_c, dq_r_dt=dq_r, dq_i_dt=dq_i,
        dq_s_dt=dq_s, dq_g_dt=dq_g,
        dN_c_dt=jnp.zeros_like(T),
        dN_r_dt=(N_r_new - h.N_r) / dt,
        dN_i_dt=(N_i_new - h.N_i) / dt,
        precipitation=o.pptrain / dt + frozen,
        precip_frozen=frozen,
    ))
