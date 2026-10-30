"""CAM6 gammadist cloud-liquid optics: table loader and band optics.

Port of CESM2.1 CAM6 ``rrtmg/cloud_rad_props.F90`` (tag cam_cesm2_1_rel),
``liqcldoptics = gammadist`` with ``liqopticsfile =
atm/cam/physprops/F_nwvl200_mu20_lam50_res64_t298_c080428.nc``
(bld/namelist_files/namelist_defaults_cam.xml:309, :312):

* loader (:112-179): g_mu (nmu), g_lambda (nmu, nlambda), k_ext_sw / ssa_sw /
  asm_sw (nmu, nlambda, 14), k_abs_lw (nmu, nlambda, 16); k_ext_sw and k_abs_lw
  divided by 0.9970449e3 ("I forgot to convert kext from m^2/Volume to m^2/Kg",
  :178-179);
* ``get_mu_lambda_weights`` (:745-769): weights in mu on g_mu, the lambda grid
  interpolated to that mu, then weights in lambda on it (both
  ``extrap_method_bndry``); ``lininterp2d1d`` bilinear combination
  (interpolate_data.F90:400-438);
* ``gam_liquid_sw`` / ``gam_liquid_lw`` (:668-741): zero optics if the
  in-cloud path < 1e-80; tau = path*ext, tau_w = tau*ssa, tau_w_g = tau_w*g,
  tau_w_f = tau_w_g*g; LW abs_od = path*abs (no cap, unlike ice);
* the callers skip a cell with lamc <= 0 (get_liquid_optics_sw :546,
  liquid_cloud_get_rad_props_lw :580): zero optics.

Bands: the file is on RRTMG bands; mapped band-to-band onto RRTMGP (user
decision 2026-10-16): SW rotated by one (RRTMG 820-2600 cm-1 last -> RRTMGP
820-2680 first), LW same order.  The loader checks the file's own band
wavelength limits against RRTMG's before mapping.
Units: path [kg/m2]; lamc [1/m]; mu dimensionless; tables [m2/kg] after the
conversion.  The file stays outside the repo, sha256-pinned.
"""
from __future__ import annotations

import hashlib
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.radiation.rrtmgp.optics import cam_lininterp
from legoesm.atmosphere.physics.radiation.rrtmgp.optics.mitchell_ice_optics import (
    rrtmg_to_rrtmgp_sw)

# --- table identity (CESM inputdata atm/cam/physprops/F_nwvl200_mu20_lam50_res64_t298_c080428.nc) ---
LIQOPTICS_SHA256 = "19b31cbb4b47fcebe55702cf3b92d24bcd217c23c9ca4d19611c95bc7e932507"
_N_SW_BANDS = 14
_N_LW_BANDS = 16
# m2/volume -> m2/kg divisor, cloud_rad_props.F90:178-179 (the table's water density)
_KEXT_DENSITY = 0.9970449e3
# RRTMG band limits [cm-1], rrtmg/radconstants.F90:25-30 (SW) and :90-96 (LW)
_RRTMG_SW_LOW = (2600., 3250., 4000., 4650., 5150., 6150., 7700., 8050., 12850., 16000.,
                 22650., 29000., 38000., 820.)
_RRTMG_SW_HIGH = (3250., 4000., 4650., 5150., 6150., 7700., 8050., 12850., 16000., 22650.,
                  29000., 38000., 50000., 2600.)
_RRTMG_LW_LOW = (10., 350., 500., 630., 700., 820., 980., 1080., 1180., 1390., 1480., 1800.,
                 2080., 2250., 2390., 2600.)
_RRTMG_LW_HIGH = (350., 500., 630., 700., 820., 980., 1080., 1180., 1390., 1480., 1800., 2080.,
                  2250., 2390., 2600., 3250.)
_WVL_M_PER_INV_CM = 1.0e-2
_BAND_RTOL = 1.0e-6


class Cam6LiquidOptics(NamedTuple):
    """gammadist table on RRTMGP band order.  g_mu (nmu,) increasing; g_lambda
    (nmu, nlambda) [1/m], decreasing along lambda; sw_ext/sw_ssa/sw_asm
    (14, nmu, nlambda) with sw_ext [m2/kg]; lw_abs (16, nmu, nlambda) [m2/kg]."""
    g_mu: jax.Array
    g_lambda: jax.Array
    sw_ext: jax.Array
    sw_ssa: jax.Array
    sw_asm: jax.Array
    lw_abs: jax.Array


def load_cam6_liquid_optics(path, expected_sha256: str = LIQOPTICS_SHA256,
                            dtype=None) -> Cam6LiquidOptics:
    """Read the gammadist table, refusing a file whose checksum, band counts,
    band limits or grid monotonicity differ from what CAM6 assumes."""
    import xarray as xr
    with open(path, "rb") as f:          # 1.9 MB file: one read
        got = hashlib.sha256(f.read()).hexdigest()
    if got != expected_sha256:
        raise ValueError(f"CAM6 liquid optics {path}: sha256 {got} != pinned {expected_sha256}")
    with xr.open_dataset(path) as ds:
        if ds.sizes.get("sw_band") != _N_SW_BANDS or ds.sizes.get("lw_band") != _N_LW_BANDS:
            raise ValueError(f"CAM6 liquid optics {path}: band counts {dict(ds.sizes)}")
        for lo, hi, wlo, whi, name in (
                (_RRTMG_SW_LOW, _RRTMG_SW_HIGH, "wvl_sw_max", "wvl_sw_min", "SW"),
                (_RRTMG_LW_LOW, _RRTMG_LW_HIGH, "wvl_max_lw", "wvl_min_lw", "LW")):
            got_lo = _WVL_M_PER_INV_CM / np.asarray(ds[wlo].values, dtype=np.float64)
            got_hi = _WVL_M_PER_INV_CM / np.asarray(ds[whi].values, dtype=np.float64)
            if not (np.allclose(got_lo, lo, rtol=_BAND_RTOL)
                    and np.allclose(got_hi, hi, rtol=_BAND_RTOL)):
                raise ValueError(f"CAM6 liquid optics {path}: {name} band limits are not RRTMG's")
        mu = np.asarray(ds["mu"].values, dtype=np.float64)
        lam = np.asarray(ds["lambda"].transpose("mu", "lambda_scale").values, dtype=np.float64)
        if mu.size < 2 or lam.shape[1] < 2:
            raise ValueError(f"CAM6 liquid optics {path}: mu/lambda grids need >= 2 points")
        if not np.all(np.diff(mu) > 0):
            raise ValueError(f"CAM6 liquid optics {path}: mu not strictly increasing")
        if not np.all(np.diff(lam, axis=1) < 0):
            raise ValueError(f"CAM6 liquid optics {path}: lambda not strictly decreasing")
        get = lambda n, b: np.asarray(ds[n].transpose(b, "mu", "lambda_scale").values,
                                      dtype=np.float64)
        ext = get("k_ext_sw", "sw_band") / _KEXT_DENSITY
        ssa = get("ssa_sw", "sw_band")
        asm = get("asm_sw", "sw_band")
        absl = get("k_abs_lw", "lw_band") / _KEXT_DENSITY
    if not all(np.all(np.isfinite(a)) for a in (mu, lam, ext, ssa, asm, absl)):
        raise ValueError(f"CAM6 liquid optics {path}: non-finite grid or table values")
    arrs = (mu, lam, rrtmg_to_rrtmgp_sw(ext), rrtmg_to_rrtmgp_sw(ssa),
            rrtmg_to_rrtmgp_sw(asm), absl)
    return Cam6LiquidOptics(*(jnp.asarray(a, dtype=dtype) for a in arrs))


def _bilinear(table: Cam6LiquidOptics, mu, lamc):
    """``get_mu_lambda_weights`` + ``lininterp2d1d``: a function mapping a
    (nmu, nlambda) table to its value at each (mu, lamc) cell."""
    im, ip, ww, we = cam_lininterp.bndry_weights(table.g_mu, mu, increasing=True)
    lam_at_mu = table.g_lambda[im] * ww[..., None] + table.g_lambda[ip] * we[..., None]
    jm, jp, ws, wn = cam_lininterp.bndry_weights(lam_at_mu, lamc, increasing=False)

    def at(t):
        return (t[im, jm] * ww * ws + t[ip, jm] * we * ws
                + t[im, jp] * ww * wn + t[ip, jp] * we * wn)
    return at


def _on_and_safe(table, path, lamc, mu):
    path, lamc, mu = jnp.asarray(path), jnp.asarray(lamc), jnp.asarray(mu)
    on = cam_lininterp.path_on(path) & (lamc > 0.0)
    # Double where: an off cell reads finite in-table (mu, lambda) and a zero
    # path, so neither pass sees its (possibly NaN) inputs.
    return (on, jnp.where(on, path, 0.0), jnp.where(on, lamc, table.g_lambda[0, 0]),
            jnp.where(on, mu, table.g_mu[0]))


def cam6_liquid_sw_optics(table: Cam6LiquidOptics, path, lamc, mu):
    """SW band optics of ``gam_liquid_sw`` for in-cloud ``path`` [kg/m2], slope
    ``lamc`` [1/m] and shape ``mu``, all (...).  Returns (tau, tau_w, tau_w_g,
    tau_w_f), each (14, ...) in RRTMGP band order."""
    on, path, lamc, mu = _on_and_safe(table, path, lamc, mu)
    at = _bilinear(table, mu, lamc)
    ext, ssa, asm = (jax.vmap(at)(t) for t in (table.sw_ext, table.sw_ssa, table.sw_asm))
    tau = path * ext
    tau_w = tau * ssa
    tau_w_g = tau_w * asm
    return tuple(jnp.where(on, x, 0.0) for x in (tau, tau_w, tau_w_g, tau_w_g * asm))


def cam6_liquid_lw_abs_od(table: Cam6LiquidOptics, path, lamc, mu):
    """LW absorption optical depth of ``gam_liquid_lw``, (16, ...); no cap."""
    on, path, lamc, mu = _on_and_safe(table, path, lamc, mu)
    at = _bilinear(table, mu, lamc)
    return jnp.where(on, path * jax.vmap(at)(table.lw_abs), 0.0)
