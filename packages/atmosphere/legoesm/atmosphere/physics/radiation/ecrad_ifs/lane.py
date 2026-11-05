"""ecRad 49r1 on a model lane: one physics step of IFS radiation.

Composes the ported IFS interface the way callpar.F90 (2.4-2.5) and
radflux_layer.F90:143/159 run it: on a full-radiation step,
:func:`.ifs_interface.radiation_scheme` refreshes the between-call carry;
on EVERY step (full or not) RADHEATN (:func:`.ifs_interface.radheatn`) turns
the carry into this step's fluxes (votes L3a, L4a, L5a, L7a).

Lane-side choices (votes, oracle_audit/ifs_port/vote_1103/VOTES.md):

* heating from the RADHEATN net fluxes with the LANE's ``g`` and ``c_pd``
  (L6a), so the column heating integral equals the net-flux divergence
  that the dycore's energy budget sees;
* half-level temperature by radintg.F90:800-818, surface half level = skin
  temperature (PTH(KLEV+1) = PTS);
* one broadband diffuse and direct albedo in all six SW albedo bands, one
  emissivity in both LW emissivity bands (L10a, L11a);
* McICA seed from the run start date and the elapsed model time, traced
  (L8a, S3a), uniforms from :func:`.ecrad.draw_uniforms` (L9a);
* TSI, gases and ozone come from :class:`.config.EcradIFSSources` (pending
  the user, L12-L14);
* the carry is packed into one (ncol, 5*(nlev+1)+13) array for the model's
  physics state (S1a): :func:`pack_carry` / :func:`unpack_carry`.

Arrays are column first, levels top-down (index 0 = model top).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.atmosphere.physics.radiation.ecrad_ifs import ecrad
from legoesm.atmosphere.physics.radiation.ecrad_ifs import ifs_interface as ifs
from legoesm.atmosphere.physics.radiation.ecrad_ifs.config import (
    ECRAD_IFS_GASES, EcradIFSConfig, EcradIFSSources)

__physics_contract__ = {
    "summary": (
        "ecRad 49r1 lane step: full IFS radiation call (when due) into the "
        "between-call carry, then RADHEATN fluxes and heating every step."
    ),
    "inputs": {
        "T, p_full": "K, Pa (ncol, nlev)", "p_half": "Pa (ncol, nlev+1)",
        "q, q_liquid, q_ice, q_rain, q_snow": "kg/kg (grid mean)",
        "cloud_fraction, land_fraction, albedo, emissivity": "1",
        "mu0": "1 (cos solar zenith, >= 0)", "T_skin": "K",
        "sources": "W/m^2 (TSI x distance factor), kg/kg (gases, ozone)",
    },
    "outputs": {
        "dT_dt": "K/s", "fluxes": "W/m^2 (CMOR: up/down positive magnitudes)",
        "carry": "packed RadiationCarry",
    },
    "sign_convention": (
        "net flux = down minus up (positive downward) on half levels; "
        "dT/dt = -(g/c_pd) d(net)/dp, so flux convergence warms."
    ),
    "conserves": ["energy"],
    "differentiable": True,
    "reference": (
        "OpenIFS 49r1 (8f6f722) callpar.F90 2.4-2.5, radflux_layer.F90, "
        "radintg.F90:800-818, radiation_scheme.F90:390-405, radheatn.F90."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ecrad_ifs_lane.py: column energy closure, "
        "carry round trip, eager vs jit, gradients."
    ),
}

_N_SW_ALBEDO = len(ecrad.I_SW_ALBEDO_INDEX)            # 6 (NSW)
_N_LW_EMISS = max(ecrad.I_EMISS_FROM_BAND_LW)          # 2 (NLWEMISS)
_N_TEGEN = 6                                           # radiation_scheme.F90 aerosol_old
_N_PROFILE = ("emtd", "trso", "trsc", "emtc", "lwderivative")
_N_SCALAR = ("trsod", "trsodc", "emtdc", "emit", "sudu", "uvdf", "parf",
             "parcf", "tincf", "fdir", "cdir", "mu0_rad")


def carry_width(nlev: int) -> int:
    """Packed width of :class:`.ifs_interface.RadiationCarry` (NLWOUT=1)."""
    return len(_N_PROFILE) * (nlev + 1) + 1 + len(_N_SCALAR)


def pack_carry(carry: ifs.RadiationCarry):
    """RadiationCarry -> (ncol, carry_width(nlev)), lossless."""
    parts = [getattr(carry, k) for k in _N_PROFILE] + [carry.frted]
    parts += [getattr(carry, k)[..., None] for k in _N_SCALAR]
    return jnp.concatenate(parts, axis=-1)


def unpack_carry(packed, nlev: int) -> ifs.RadiationCarry:
    """Inverse of :func:`pack_carry`; raises on a wrong width."""
    if packed.shape[-1] != carry_width(nlev):
        raise ValueError(
            f"ecrad_carry width {packed.shape[-1]} != {carry_width(nlev)} "
            f"for nlev={nlev}")
    out, i = {}, 0
    for k in _N_PROFILE:
        out[k] = packed[..., i:i + nlev + 1]
        i += nlev + 1
    out["frted"] = packed[..., i:i + 1]
    i += 1
    for k in _N_SCALAR:
        out[k] = packed[..., i]
        i += 1
    return ifs.RadiationCarry(**out)


def half_level_temperature(T, p_full, p_half, T_skin):
    """radintg.F90:800-818: interior half levels from the two adjacent full
    levels, the top by extrapolation, the surface = skin temperature."""
    pf0, pf1 = p_full[..., :-1], p_full[..., 1:]
    ph = p_half[..., 1:-1]
    t0, t1 = T[..., :-1], T[..., 1:]
    inner = (t0 * pf0 * (pf1 - ph) + t1 * pf1 * (ph - pf0)) * (1.0 / (ph * (pf1 - pf0)))
    top = T[..., 0] - p_full[..., 0] * (T[..., 0] - inner[..., 0]) / (p_full[..., 0] - p_half[..., 1])
    return jnp.concatenate([top[..., None], inner, T_skin[..., None]], axis=-1)


def step(cfg: EcradIFSConfig, sources: EcradIFSSources, *, full_call: bool,
         packed_carry, T, p_full, p_half, T_skin, lat, lon, q, q_liquid, q_ice,
         q_rain, q_snow, cloud_fraction, land_fraction, mu0, albedo_dif,
         albedo_dir, emissivity, iday, elapsed_s, g: float, c_pd: float):
    """One physics step.  ``full_call`` is static; ``iday`` is NDD(NINDAT),
    the run start date's day of month, ``elapsed_s`` NSTEP*TSTEP [s].  Returns
    ``(dT_dt (ncol, nlev), fluxes dict (ncol,), packed carry)``."""
    ncol, nlev = T.shape
    sharp = cfg.sharpness
    if full_call:
        for k in ECRAD_IFS_GASES:
            if k not in sources.gas_mmr:
                raise ValueError(f"ecrad_ifs: no {k!r} in sources.gas_mmr (vote L16b)")
        prof = lambda x: jnp.broadcast_to(jnp.asarray(x, T.dtype), T.shape)  # noqa: E731
        col = lambda x: jnp.broadcast_to(jnp.asarray(x, T.dtype), (ncol,))   # noqa: E731
        dp = p_half[..., 1:] - p_half[..., :-1]
        alb_d = jnp.broadcast_to(col(albedo_dif)[:, None], (ncol, _N_SW_ALBEDO))
        alb_p = jnp.broadcast_to(col(albedo_dir)[:, None], (ncol, _N_SW_ALBEDO))
        emis = jnp.broadcast_to(col(emissivity)[:, None], (ncol, _N_LW_EMISS))
        gemu = jnp.sin(lat)
        seed = ifs.mcica_seed_traced(iday, elapsed_s, lon, gemu)
        u_lw, u_sw = ecrad.draw_uniforms(seed, nlev)
        gas = {k: prof(sources.gas_mmr[k]) for k in ECRAD_IFS_GASES}
        _, carry = ifs.radiation_scheme(
            mu0=mu0, uniforms_lw=u_lw, uniforms_sw=u_sw, sharpness=sharp,
            solar_irradiance=sources.solar_irradiance, skin_temperature=T_skin,
            albedo_dif=alb_d, albedo_dir=alb_p, spectral_emiss=emis,
            ccn_land=col(cfg.ccn_land), ccn_sea=col(cfg.ccn_sea),
            gelam=lon, gemu=gemu, land_sea_mask=col(land_fraction),
            pressure_fl=p_full, temperature_fl=T, pressure_hl=p_half,
            temperature_hl=half_level_temperature(T, p_full, p_half, T_skin),
            q=q, o3_dp=prof(sources.o3_mmr) * dp,
            cloud_fraction=cloud_fraction, q_liquid=q_liquid, q_ice=q_ice,
            q_rain=q_rain, q_snow=q_snow,
            aerosol_old=jnp.zeros((ncol, _N_TEGEN, nlev), T.dtype),
            # prepare_inputs' host-side seed lands in an output that
            # radiation_scheme discards; the traced seed above is the one used
            nindat=0, nstep=0, tstep=0.0, **gas)
    else:
        carry = unpack_carry(packed_carry, nlev)
    rh = ifs.radheatn(carry, p_half, q, T_skin, mu0, sources.solar_irradiance,
                      sharpness=sharp)
    net = rh["frso"] + rh["frth"]
    dT_dt = -(g / c_pd) * (net[..., 1:] - net[..., :-1]) / (p_half[..., 1:] - p_half[..., :-1])
    zi0 = rh["incsr"]
    fluxes = {
        "sw_net_sfc": rh["frso"][..., -1], "lw_net_sfc": rh["frth"][..., -1],
        "sw_down_toa": zi0, "sw_up_toa": zi0 - rh["frso"][..., 0],
        # ecRad's LW down flux at the model top is zero
        "lw_up_toa": -rh["frth"][..., 0],
        "sw_down_sfc": rh["frsod"], "lw_down_sfc": rh["frthd"],
        "sw_up_toa_clr": zi0 - rh["frsoc"][..., 0],
        "lw_up_toa_clr": -rh["frthc"][..., 0],
        "net_flux": net,
    }
    return dT_dt, fluxes, pack_carry(carry)


__all__ = ["carry_width", "pack_carry", "unpack_carry", "half_level_temperature",
           "step"]
