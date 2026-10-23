"""CAM6 Mitchell ice (and snow) optics: table loader and band optics.

Port of CESM2.1 CAM6 ``rrtmg/cloud_rad_props.F90`` (tag cam_cesm2_1_rel):
the ice optics file ``iceoptics_c080917.nc`` (namelist ``iceopticsfile``,
``icecldoptics = mitchell``, bld/namelist_files/namelist_defaults_cam.xml:310-311)
read at :195-260, and the per-cell optics of ``interpolate_ice_optics_sw``
(:472-520) and ``interpolate_ice_optics_lw`` (:630-660).  CAM6 uses the SAME
table and routines for cloud ice (path ICIWP, diameter DEI) and for snow
(path ICSWP, diameter DES; :424-443, :594-608).

Ported behaviour, guards included:

* a cell with water path < 1e-80 or diameter == 0 gets ZERO optics
  (:491, :641);
* linear interpolation in effective diameter with ``extrap_method_bndry``
  (values outside the table take the end value);
* SW: ``tau = path*ext``, ``tau_w = tau*ssa``, ``tau_w_g = tau_w*g``,
  ``tau_w_f = tau_w_g*g``;
* LW: ``abs_od = path*lw_abs`` capped at 50 per band (:654).

The table is on RRTMG bands.  The JAX solver runs RRTMGP bands; the mapping is
band-to-band (user decision 2026-10-16, no overlap remap): LW in the same order
(RRTMG 10-350/350-500/.../2380-2600/2600-3250 cm-1 against RRTMGP
10-250/250-500/.../2390-2680/2680-3250 cm-1); SW RRTMG puts 820-2600 cm-1 last
(rrtmg/radconstants.F90:25-30) while RRTMGP puts 820-2680 cm-1 first, so the
SW table rows are rotated by one.  The loader checks the file's own band
sequence attribute against the RRTMG band limits before rotating.

Units: water path [kg/m2], diameter [micron], table [m2/kg] (file attributes).
The file is NOT shipped with the repo: its path is given explicitly and its
sha256 must match ``ICEOPTICS_SHA256`` (NCAR inputdata
atm/cam/physprops/iceoptics_c080917.nc, 72,756 bytes).
"""
from __future__ import annotations

import hashlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.radiation.rrtmgp.optics import cam_lininterp

# --- table identity (CESM inputdata atm/cam/physprops/iceoptics_c080917.nc) ---
ICEOPTICS_SHA256 = "e95e8b0de6a66f24bad66f75cecfd28e15fca0abc4d5452412c636b8b865d638"
_N_SW_BANDS = 14   # RRTMG nbndsw; the file must match (cloud_rad_props.F90:204)
_N_LW_BANDS = 16   # RRTMG nbndlw (cloud_rad_props.F90:200)
# RRTMG SW band upper wavenumbers [cm-1] in RRTMG order (radconstants.F90:28-30);
# the file's SW_band_sequence_based_on_shortest_wavelength attribute is
# 1e4/these [micron].
_RRTMG_SW_WAVENUM_HIGH = (3250.0, 4000.0, 4650.0, 5150.0, 6150.0, 7700.0, 8050.0,
                          12850.0, 16000.0, 22650.0, 29000.0, 38000.0, 50000.0,
                          2600.0)
_CM_TO_MICRON = 1.0e4
# The file prints its band sequence to 0.01 micron, so a match is within half that.
_SW_SEQ_ATOL_MICRON = 0.006

# --- guard of interpolate_ice_optics_lw (cloud_rad_props.F90:654); the 1e-80
# path guard (:491, :641) lives in cam_lininterp.path_on ---
_LW_ABS_OD_MAX = 50.0    # per-band LW absorption optical depth cap


class MitchellIceOptics(NamedTuple):
    """Mitchell table on RRTMGP band order.

    d_eff (n_d,) [micron], ascending; sw_ext/sw_ssa/sw_asm (14, n_d) with
    sw_ext [m2/kg]; lw_abs (16, n_d) [m2/kg].
    """
    d_eff: jax.Array
    sw_ext: jax.Array
    sw_ssa: jax.Array
    sw_asm: jax.Array
    lw_abs: jax.Array


def rrtmg_to_rrtmgp_sw(table_rows):
    """Rotate RRTMG-ordered SW rows (820-2600 cm-1 last) to RRTMGP order
    (820-2680 cm-1 first)."""
    t = np.asarray(table_rows)
    if t.shape[0] != _N_SW_BANDS:
        raise ValueError(f"expected {_N_SW_BANDS} SW bands, got {t.shape[0]}")
    return np.concatenate([t[-1:], t[:-1]], axis=0)


def _sha256(path) -> str:
    with open(path, "rb") as f:          # 72 KB file: one read
        return hashlib.sha256(f.read()).hexdigest()


def load_mitchell_ice_optics(path, expected_sha256: str = ICEOPTICS_SHA256,
                             dtype=None) -> MitchellIceOptics:
    """Read the Mitchell table, refusing a file whose checksum, band counts or
    band sequence differ from CAM6's (cloud_rad_props.F90:195-260)."""
    import xarray as xr
    got = _sha256(path)
    if got != expected_sha256:
        raise ValueError(f"Mitchell ice optics {path}: sha256 {got} != pinned "
                         f"{expected_sha256}")
    with xr.open_dataset(path) as ds:
        if ds.sizes.get("sw_band") != _N_SW_BANDS or ds.sizes.get("lw_band") != _N_LW_BANDS:
            raise ValueError(f"Mitchell ice optics {path}: band counts "
                             f"{dict(ds.sizes)} != {_N_SW_BANDS} SW / {_N_LW_BANDS} LW")
        seq = ds.attrs.get("SW_band_sequence_based_on_shortest_wavelength")
        if seq is None:
            raise ValueError(f"Mitchell ice optics {path}: no SW band sequence attribute")
        seq = np.array([float(s) for s in str(seq).replace(",", " ").split()])
        want = _CM_TO_MICRON / np.asarray(_RRTMG_SW_WAVENUM_HIGH)
        if seq.shape != want.shape or not np.allclose(seq, want, rtol=0.0, atol=_SW_SEQ_ATOL_MICRON):
            raise ValueError(f"Mitchell ice optics {path}: SW band sequence {seq} is not "
                             f"RRTMG's {np.round(want, 2)}")
        d = np.asarray(ds["d_eff"].values, dtype=np.float64)
        if d.ndim != 1 or d.size < 2 or not np.all(np.diff(d) > 0):
            raise ValueError(f"Mitchell ice optics {path}: d_eff not strictly ascending")
        get = lambda n: np.asarray(ds[n].transpose(n.split("_")[0] + "_band", "d_eff").values,
                                   dtype=np.float64)
        sw_ext, sw_ssa, sw_asm, lw_abs = (get(n) for n in ("sw_ext", "sw_ssa", "sw_asm",
                                                              "lw_abs"))
    if not all(np.all(np.isfinite(a)) for a in (d, sw_ext, sw_ssa, sw_asm, lw_abs)):
        raise ValueError(f"Mitchell ice optics {path}: non-finite grid or table values")
    arrs = (d, rrtmg_to_rrtmgp_sw(sw_ext), rrtmg_to_rrtmgp_sw(sw_ssa),
            rrtmg_to_rrtmgp_sw(sw_asm), lw_abs)
    return MitchellIceOptics(*(jnp.asarray(a, dtype=dtype) for a in arrs))


def _on(path, d_eff):
    """Cells CAM6 gives optics: path >= 1e-80 and D != 0 (cloud_rad_props.F90:491, :641)."""
    return cam_lininterp.path_on(path) & (d_eff != 0.0)


def mitchell_sw_optics(table: MitchellIceOptics, path, d_eff):
    """SW band optics of ``interpolate_ice_optics_sw`` for in-cloud ``path``
    [kg/m2] and effective diameter ``d_eff`` [micron], both (...).
    Returns (tau, tau_w, tau_w_g, tau_w_f), each (14, ...) in RRTMGP order."""
    path = jnp.asarray(path)
    d_eff = jnp.asarray(d_eff)
    on = _on(path, d_eff)
    # Double where: off cells read a finite in-table diameter and a zero path, so
    # neither the forward nor the reverse pass sees their (possibly NaN) inputs.
    d_eff = jnp.where(on, d_eff, table.d_eff[0])
    path = jnp.where(on, path, 0.0)
    ext, ssa, asm = (jax.vmap(lambda row: cam_lininterp.interp1(table.d_eff, row, d_eff))(t)
                     for t in (table.sw_ext, table.sw_ssa, table.sw_asm))
    # where() on the product, not on path: an off cell never touches the table
    # (the Fortran skips it), so a non-finite entry cannot leak (GLM P1a review).
    tau = path * ext
    tau_w = tau * ssa
    tau_w_g = tau_w * asm
    return tuple(jnp.where(on, x, 0.0) for x in (tau, tau_w, tau_w_g, tau_w_g * asm))


def mitchell_lw_abs_od(table: MitchellIceOptics, path, d_eff):
    """LW absorption optical depth of ``interpolate_ice_optics_lw``, (16, ...)."""
    path = jnp.asarray(path)
    d_eff = jnp.asarray(d_eff)
    on = _on(path, d_eff)
    # Double where: off cells read a finite in-table diameter and a zero path, so
    # neither the forward nor the reverse pass sees their (possibly NaN) inputs.
    d_eff = jnp.where(on, d_eff, table.d_eff[0])
    path = jnp.where(on, path, 0.0)
    absor = jax.vmap(lambda row: cam_lininterp.interp1(table.d_eff, row, d_eff))(table.lw_abs)
    return jnp.where(on, jnp.minimum(path * absor, _LW_ABS_OD_MAX), 0.0)
