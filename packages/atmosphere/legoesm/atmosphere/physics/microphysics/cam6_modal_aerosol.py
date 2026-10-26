"""CAM6 4-mode (MAM4) aerosol properties and ``loadaer`` for droplet activation.

Mode composition, species property files and mode files are CAM6's defaults:
``bld/build-namelist:1105-1123`` (4mode: accum so4 pom soa bc dst ncl, aitken so4
soa ncl dst, coarse dst ncl so4, primary_carbon pom bc; species order as there,
which ``ChemNamelist.pm::set_aero_modes_info`` keeps) and
``bld/namelist_files/namelist_defaults_cam.xml:260-267, 285-288``.  The netCDF
files live outside the repo and are sha256-pinned (the caller passes the
directory).  ``voltonumblo/hi`` follow ``ndrop.F90:158-161``; ``loadaer``
(``ndrop.F90:1860-1962``) is ported for the interstitial phase only: with the
cloud-borne aerosol set to zero (user decision 2026-10-16) phases 1 and 3 give
the same numbers.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

__physics_contract__ = {
    "summary": "CAM6 MAM4 mode/species properties and loadaer (mode number, dry volume, hygroscopicity).",
    "inputs": {
        "species_mmr": "kg/kg (interstitial species mass mixing ratios of one mode)",
        "number_mmr": "1/kg (interstitial number mixing ratio of the mode)",
        "cs": "kg/m^3 (air density p/(R_d T))",
    },
    "outputs": {
        "naerosol": "1/m^3", "vaerosol": "m^3/m^3", "hygro": "1 (volume-mean hygroscopicity)",
    },
    "sign_convention": "all outputs >= 0; number bounded by the mode's size limits",
    "conserves": ["none"],
    "differentiable": True,
    "reference": "CAM6 ndrop.F90 loadaer / ndrop_init (Ghan et al. 2011, Liu et al. 2012 MAM)",
    "idealized_test": "gfortran build of the verbatim loadaer vs this port at fp64 rtol 1e-12",
}

# --- CAM6 MAM4 layout (bld/build-namelist:1115-1123) -------------------------
MAM4_MODES = ("accum", "aitken", "coarse", "primary_carbon")
MAM4_SPECIES = (
    ("so4", "pom", "soa", "bc", "dst", "ncl"),
    ("so4", "soa", "ncl", "dst"),
    ("dst", "ncl", "so4"),
    ("pom", "bc"),
)
# --- CAM6 default property files (namelist_defaults_cam.xml) ----------------
MAM4_MODE_FILES = (
    "mam4_mode1_rrtmg_aeronetdust_c141106.nc",
    "mam4_mode2_rrtmg_aitkendust_c141106.nc",
    "mam4_mode3_rrtmg_aeronetdust_c141106.nc",
    "mam4_mode4_rrtmg_c130628.nc",
)
MAM4_SPECIES_FILES = {
    "so4": "sulfate_rrtmg_c080918.nc",
    "pom": "ocpho_rrtmg_c130709.nc",
    "soa": "ocphi_rrtmg_c100508.nc",
    "bc": "bcpho_rrtmg_c100508.nc",
    "dst": "dust_aeronet_rrtmg_c141106.nc",
    "ncl": "ssam_rrtmg_c100508.nc",
}
# sha256 pins (manifest ppe000_sha256.txt in the data folder, 2026-10-16)
MAM4_SHA256 = {
    "mam4_mode1_rrtmg_aeronetdust_c141106.nc": "a48521293bfc21ed2f71a21e2de81fc1634e028e48b88d1428cde3a0d8ae90a1",
    "mam4_mode2_rrtmg_aitkendust_c141106.nc": "ef42a9139cfe69e1220dd9af607c041fc98657ecafba4d32e26b5911725add12",
    "mam4_mode3_rrtmg_aeronetdust_c141106.nc": "730f4da004f7bec4d89cbc7f9dc93adfd534e78cff613a472df0070456ec8ea2",
    "mam4_mode4_rrtmg_c130628.nc": "5bc964842d5ef88e287373bb651a269d4c88151cf6836de66625140390e2039d",
    "sulfate_rrtmg_c080918.nc": "9d29f5c30e99e958014551542e25899c8083f014825acb61c5a56876e695c051",
    "ocpho_rrtmg_c130709.nc": "41b539924ce0c651a51933fd12a7b13ef978bd407fa700b8aa3e07ea7e846490",
    "ocphi_rrtmg_c100508.nc": "c25de9e502cc97a5ddc696ee3529f35b0b2f42ab648d670781aa9a24b91dd234",
    "bcpho_rrtmg_c100508.nc": "03b5a15b43b7bac93984314dec0e4c8bc2a694f44a5283d5fbece13198c076ea",
    "dust_aeronet_rrtmg_c141106.nc": "bf5207fa0ad4cd29909540e2008f220314ee60b22e1f77d0535e78655b0a12cf",
    "ssam_rrtmg_c100508.nc": "42985d7334d2f1f230fb9470cd160006e99d8c4ec5d4c42fdeb86a5b2181af0a",
}
# --- loadaer threshold (ndrop.F90:1932) --------------------------------------
_LOADAER_VOL_MIN = 1.0e-30


class Cam6ModeProps(NamedTuple):
    """Per-mode constants (all float64 numpy): ``sigmag``, ``dgnumlo``,
    ``dgnumhi``, ``voltonumblo``, ``voltonumbhi`` shape (4,); ``specdens`` and
    ``spechygro`` are tuples (per mode) of per-species arrays in MAM4_SPECIES order."""
    sigmag: np.ndarray
    dgnumlo: np.ndarray
    dgnumhi: np.ndarray
    voltonumblo: np.ndarray
    voltonumbhi: np.ndarray
    specdens: tuple
    spechygro: tuple


def read_pinned_netcdf(path: Path, pins: dict):
    """Open path with netCDF4 after checking its sha256 against pins[path.name]."""
    import netCDF4
    with open(path, "rb") as f:
        got = hashlib.file_digest(f, "sha256").hexdigest()
    want = pins[path.name]
    if got != want:
        raise ValueError(f"CAM6 aerosol input file {path}: sha256 {got} != pinned {want}")
    return netCDF4.Dataset(path)


def voltonumb(dgnum, sigmag):
    """``1/((pi/6) dgnum**3 exp(4.5 alogsig**2))`` (ndrop.F90:158-161)."""
    alogsig = np.log(sigmag)
    return 1.0 / ((np.pi / 6.0) * (dgnum ** 3.0) * np.exp(4.5 * alogsig ** 2.0))  # coeff-ok: exact lognormal 3rd moment


def load_cam6_mode_props(data_dir) -> Cam6ModeProps:
    """Read the four MAM4 mode files and six species files (sha256-checked)."""
    data_dir = Path(data_dir)
    sig, lo, hi = [], [], []
    for name in MAM4_MODE_FILES:
        with read_pinned_netcdf(data_dir / name, MAM4_SHA256) as d:
            sig.append(float(d["sigmag"][...]))
            lo.append(float(d["dgnumlo"][...]))
            hi.append(float(d["dgnumhi"][...]))
    dens, hyg = {}, {}
    for spec, name in MAM4_SPECIES_FILES.items():
        with read_pinned_netcdf(data_dir / name, MAM4_SHA256) as d:
            dens[spec] = float(d["density"][...])
            hyg[spec] = float(d["hygroscopicity"][...])
    sigmag, dgnumlo, dgnumhi = (np.array(x, dtype=np.float64) for x in (sig, lo, hi))
    return Cam6ModeProps(
        sigmag=sigmag, dgnumlo=dgnumlo, dgnumhi=dgnumhi,
        voltonumblo=voltonumb(dgnumlo, sigmag), voltonumbhi=voltonumb(dgnumhi, sigmag),
        specdens=tuple(np.array([dens[s] for s in sp]) for sp in MAM4_SPECIES),
        spechygro=tuple(np.array([hyg[s] for s in sp]) for sp in MAM4_SPECIES),
    )


def cam6_loadaer(species_mmr, number_mmr, cs, specdens, spechygro, voltonumblo, voltonumbhi):
    """``loadaer`` phase 1 for one mode (ndrop.F90:1916-1960), statement order kept.

    ``species_mmr`` (nspec,) + F [kg/kg], ``number_mmr`` F [1/kg], ``cs`` F [kg/m^3];
    ``specdens``/``spechygro`` (nspec,); ``voltonumblo/hi`` scalars of this mode.
    Returns ``naerosol`` [1/m^3], ``vaerosol`` [m^3/m^3], ``hygro`` [-].
    """
    vaerosol = jnp.zeros_like(cs)
    hygro = jnp.zeros_like(cs)
    for l in range(species_mmr.shape[0]):
        vol = jnp.maximum(species_mmr[l], 0.0) / specdens[l]
        vaerosol = vaerosol + vol
        hygro = hygro + vol * spechygro[l]
    big = vaerosol > _LOADAER_VOL_MIN
    vsafe = jnp.where(big, vaerosol, 1.0)
    hygro = jnp.where(big, hygro / vsafe, 0.0)
    vaerosol = jnp.where(big, vaerosol * cs, 0.0)
    naerosol = number_mmr * cs
    naerosol = jnp.maximum(naerosol, vaerosol * voltonumbhi)
    naerosol = jnp.minimum(naerosol, vaerosol * voltonumblo)
    return naerosol, vaerosol, hygro
