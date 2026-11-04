"""Run configuration of the ecRad 49r1 lane (``radiation='ecrad_ifs'``).

Only the values the lane hands the IFS interface are fields (vote L21a);
every other 49r1 namelist value stays a module constant of the ported
routine it belongs to.  Defaults are the 49r1 values, so the oracle tests
are unchanged.

* ``ccn_land`` / ``ccn_sea``: RCCNLND / RCCNSEA, suecrad.F90:841-842, as
  callpar.F90:757-758 passes them (the LERADS=F arm).  The 49r1 default
  LERADS=T takes PCCNL/PCCNO from SURFRAD, which is not in our source clone:
  a recorded departure (vote L25a).
* ``sharpness``: the one static smoothing argument of the port (100 in
  production, ``math.inf`` = the oracle's hard form).

:class:`EcradIFSSources` holds the three inputs that still wait for the user
(votes L12-L14: IFS TSI and GHG climatologies, A15; Cariolle ozone, A11).
The lane refuses to build without them (vote S2a); tests inject them.
"""

from __future__ import annotations

from typing import NamedTuple

import jax

__param_spec__ = {
    "EcradIFSConfig": {
        "scheme_key": "atm.rad.EcradIFSConfig",
        "excluded": {
            "ccn_land": "49r1 oracle setting (RCCNLND, suecrad.F90:841), fixed pending an IFS tuning plan (vote L22a)",
            "ccn_sea": "49r1 oracle setting (RCCNSEA, suecrad.F90:842), fixed pending an IFS tuning plan (vote L22a)",
            "sharpness": "numerics: smoothing sharpness of the ported switches, not a physical closure",
        },
        "params": {},
    },
}


class EcradIFSConfig(NamedTuple):
    """ecRad 49r1 lane settings (see module docstring)."""

    ccn_land: float = 900.0     # [cm-3] RCCNLND, suecrad.F90:841
    ccn_sea: float = 50.0       # [cm-3] RCCNSEA, suecrad.F90:842
    sharpness: float = 100.0    # [1] port smoothing sharpness (inf = hard)


class EcradIFSSources(NamedTuple):
    """The inputs pending the user's answer (votes L12-L14).

    ``solar_irradiance`` [W/m^2] is RII0 (TSI times the Earth-Sun distance
    factor, radintg.F90:513); ``gas_mmr`` maps each of co2, ch4, n2o, cfc11,
    cfc12, hcfc22, ccl4 to a mass mixing ratio [kg/kg], scalar or (ncol,
    nlev); ``o3_mmr`` [kg/kg] (ncol, nlev).  Every gas is required (vote
    L16b)."""

    solar_irradiance: jax.Array | float
    gas_mmr: dict
    o3_mmr: jax.Array


ECRAD_IFS_GASES = ("co2", "ch4", "n2o", "cfc11", "cfc12", "hcfc22", "ccl4")
