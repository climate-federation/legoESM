"""ecRad 49r1 McICA longwave and shortwave solvers, JAX port.

Line-by-line port (OpenIFS 49r1, 8f6f722, compiled sources in
``oracle_audit/ifs_port/ecrad/bld49/src``) of

* ``radiation_mcica_lw.F90`` solver_mcica_lw, for do_lw_aerosol_scattering =
  false, do_lw_cloud_scattering = true, do_lw_derivatives = true: clear sky by
  ``calc_no_scattering_transmittance_lw`` + ``calc_fluxes_no_scattering_lw``,
  cloudy layers by ``calc_ref_trans_lw``, fluxes by ``fast_adding_ica_lw``,
  Hogan-Bozzo derivatives by ``calc_lw_derivatives_ica`` /
  ``modify_lw_derivatives_ica``;
* ``radiation_mcica_sw.F90`` solver_mcica_sw, for
  do_sw_delta_scaling_with_gases = false, do_sw_direct = true, do_clear = true
  (``calc_ref_trans_sw`` + ``adding_ica_sw``);
* ``radiation_flux.F90`` calc_surface_spectral (SW surface band fluxes and the
  canopy fluxes through ``sw_albedo_weights``), and the broadband sums.

The cloud generator is NOT here: ``od_scaling`` (McICA optical-depth scaling
per g-point and layer) and ``total_cloud_cover`` are inputs, exactly as the
oracle solvers receive them from ``cloud_generator``.  Cloud optics per band
are inputs too (mapped to g-points here via ``i_band_from_reordered_g``).

Array layout: one column is ``(nlev, ng)`` (Fortran ``(ng, nlev)`` transposed),
half levels ``(nlev+1, ng)``; the public functions are vmapped over a leading
column axis.  Levels are top-down (index 0 = TOA), as the oracle.

Smoothing (RULE 5; votes M1-M5): one static ``sharpness`` (default 100, inf =
oracle).  Smoothed traced switches: cloud_fraction >= threshold per layer
(cloudy vs clear layer optics, and the fast-adding cloud-top recurrence),
total_cloud_cover >= threshold (cloudy vs clear column), total_cloud_cover <
1 - threshold (derivative modification, in the clear-fraction variable), and
the kernels' switches in ``two_stream.py``.  Width of each band = 0.01 x the
threshold.  The od_total > 0 / scat_od > 0 division guards divide by an inactive-exact
floor at 2^-500 (vote N1'); the cos_sza > 0 night branch is a smoothstep
whose band lies wholly on the night side (vote N2).  KNOWN DIFFERENCE at the default: only
inputs inside a band differ from the oracle (measured gaps are pinned in the
oracle test).

Fortran SUM/indexed_sum/canopy loops are emulated as sequential left-to-right
adds (vote M6a).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from legoesm.atmosphere.physics.radiation.ecrad_ifs.two_stream import (
    blend,
    calc_no_scattering_transmittance_lw,
    calc_ref_trans_lw,
    calc_ref_trans_sw,
    lower_clamp,
    step,
)
from legoesm.core.smooth import smoothstep_cubic

# Floor of the guarded divisions and of mu0 (votes N1', N2'): 2^-500 in fp64,
# 2^-60 in fp32, so the floor, its 1% band and 1/floor^2 are all normal
# (XLA flushes subnormals).
_DEN_FLOOR = {np.dtype(np.float64): float.fromhex("0x1p-500"),
              np.dtype(np.float32): float.fromhex("0x1p-60")}


def den_floor(x):
    return _DEN_FLOOR[np.dtype(jnp.result_type(x))]

# --- RRTMG band limits [cm-1] given to allocate_bands_only (radiation_ifs_rrtm.F90:103-106) ---
_SW_WAVENUMBER1_BAND = (2600.0, 3250.0, 4000.0, 4650.0, 5150.0, 6150.0, 7700.0,
                        8050.0, 12850.0, 16000.0, 22650.0, 29000.0, 38000.0, 820.0)
_SW_WAVENUMBER2_BAND = (3250.0, 4000.0, 4650.0, 5150.0, 6150.0, 7700.0, 8050.0,
                        12850.0, 16000.0, 22650.0, 29000.0, 38000.0, 50000.0, 2600.0)
# Trapezium weights of calc_mapping_from_bands (radiation_spectral_definition.F90).
_MAPPING_SAMPLE_WEIGHTS = (0.5, 1.0, 1.0, 1.0, 0.5)
# Metres -> cm-1 conversion factor 0.01/wavelength (radiation_spectral_definition.F90).
_M_TO_INV_CM = 0.01

__physics_contract__ = {
    "summary": (
        "ecRad 49r1 McICA longwave (cloud scattering, no aerosol scattering, "
        "Hogan-Bozzo approximate-update surface derivatives) and shortwave "
        "(direct + diffuse, no gas delta scaling) solvers with broadband, "
        "surface-band and canopy flux outputs. Port of radiation_mcica_lw/sw, "
        "radiation_adding_ica_lw/sw, radiation_lw_derivatives and "
        "radiation_flux calc_surface_spectral (OpenIFS 49r1)."
    ),
    "inputs": {
        "od_lw": "1 (gas, per g-point)", "planck_hl": "W/m^2 (per g-point)",
        "lw_emission": "W/m^2", "lw_albedo": "1",
        "od_sw": "1", "ssa_sw": "1", "asymmetry_sw": "1",
        "incoming_sw": "W/m^2 (TOA normal to beam, per g-point)",
        "sw_albedo": "1", "sw_albedo_direct": "1", "cos_sza": "1",
        "od/ssa/asymmetry_*_cloud": "1 (per band)", "cloud_fraction": "1",
        "od_scaling": "1 (McICA generator)", "total_cloud_cover": "1",
    },
    "outputs": {
        "flux_up_lw/flux_dn_lw (+_clear)": "W/m^2",
        "lw_derivative": "1 (d flux_up / d surface flux_up)",
        "flux_up_sw/flux_dn_sw/flux_dn_direct_sw (+_clear)": "W/m^2",
        "spectral_flux_dn_(direct_)sw_surf(_clear)": "W/m^2 per band",
        "canopy_flux_dn_diffuse/direct_sw_surf": "W/m^2 per albedo interval",
        "cloud_cover_lw/sw": "1 (-1 for SW night columns, oracle fill)",
    },
    "sign_convention": (
        "All fluxes positive in their named direction on half levels, top-down "
        "(index 0 = TOA). SW direct flux is per unit horizontal area (times "
        "cos_sza). Net flux dn - up; layer heating = g/cp * divergence. Energy: flux "
        "continuity across every interface; column heating = TOA net - "
        "surface net."
    ),
    "conserves": ["energy"],
    "differentiable": True,
    "reference": (
        "Pincus, Barker & Morcrette (2003) JGR 108 (McICA); Hogan & Bozzo (2015, "
        "2018) JAMES; oracle radiation_mcica_lw.F90 / radiation_mcica_sw.F90 "
        "(OpenIFS 49r1 8f6f722)."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ecrad_ifs_mcica_solver_oracle.py: oracle match "
        "at 1e-12 (hard form), conservative-scattering column, energy closure "
        "and gradient checks."
    ),
}


# --------------------------------------------------------------------------
# Static configuration (numpy, built once)
# --------------------------------------------------------------------------

def sw_albedo_weights(wavelength_bound, i_intervals):
    """calc_mapping_from_bands, use_bands branch, Planck = 1 (RRTMG leaves
    reference_temperature = -1).  Returns mapping (ninput, nband), Fortran
    ``sw_albedo_weights(jalbedoband, jband)``; i_intervals one-based."""
    wn1b = np.asarray(_SW_WAVENUMBER1_BAND, dtype=np.float64)
    wn2b = np.asarray(_SW_WAVENUMBER2_BAND, dtype=np.float64)
    nband = wn1b.size
    ninterval = len(i_intervals)
    ninput = max(i_intervals)
    sample_weight_sum = 0.0
    for w in _MAPPING_SAMPLE_WEIGHTS:  # sum(planck_sample*weight_sample), planck = 1
        sample_weight_sum = sample_weight_sum + 1.0 * w
    mapping = np.zeros((ninput, nband))
    for jband in range(nband):
        for jint in range(ninterval):
            wn2 = wn2b[jband] if jint == 0 else min(wn2b[jband], _M_TO_INV_CM / wavelength_bound[jint - 1])
            wn1 = wn1b[jband] if jint == ninterval - 1 else max(wn1b[jband], _M_TO_INV_CM / wavelength_bound[jint])
            if wn2 > wn1:
                i = i_intervals[jint] - 1
                mapping[i, jband] = mapping[i, jband] + sample_weight_sum * (wn2 - wn1)
    for jband in range(nband):
        total = 0.0
        for v in mapping[:, jband]:
            total = total + v
        mapping[:, jband] = mapping[:, jband] * (1.0 / total)
    return mapping


# --------------------------------------------------------------------------
# Sequential sums (Fortran order, vote M6a)
# --------------------------------------------------------------------------

def seqsum(x, axis):
    """Left-to-right sum along ``axis`` (gfortran SUM at -O1, no reassociation)."""
    x = jnp.moveaxis(x, axis, 0)
    total, _ = jax.lax.scan(lambda c, v: (c + v, None), x[0], x[1:])
    return total


def _band_sums(x_g, band_of_g):
    """indexed_sum: dest(band) = 0 + sum of its g-points in g order; x_g (..., ng)."""
    nband = int(band_of_g.max()) + 1
    out = []
    for b in range(nband):
        idx = np.nonzero(band_of_g == b)[0]
        acc = jnp.zeros_like(x_g[..., 0])
        for g in idx:
            acc = acc + x_g[..., g]
        out.append(acc)
    return jnp.stack(out, axis=-1)


# --------------------------------------------------------------------------
# Adding methods (radiation_adding_ica_lw/sw.F90), one column (nlev, ng)
# --------------------------------------------------------------------------

def _scan_up_albedo_source(R, T, s_up_term, s_dn_term, albedo_surf, source_surf):
    """Upward adding pass shared by LW and SW.

    albedo(h) = R + T*T*albedo(h+1)*inv;  source(h) = s_up_term(h) +
    T*(source(h+1) + albedo(h+1)*s_dn_term(h))*inv.  Returns albedo, source
    (nlev+1, ng) and inv_denominator (nlev, ng)."""

    def body(carry, xs):
        a_below, s_below = carry
        r, t, su, sd = xs
        inv = 1.0 / (1.0 - a_below * r)
        a = r + t * t * a_below * inv
        s = su + t * (s_below + a_below * sd) * inv
        return (a, s), (a, s, inv)

    _, (a, s, inv) = jax.lax.scan(
        body, (albedo_surf, source_surf), (R, T, s_up_term, s_dn_term), reverse=True
    )
    albedo = jnp.concatenate([a, albedo_surf[None]], axis=0)
    source = jnp.concatenate([s, source_surf[None]], axis=0)
    return albedo, source, inv


def _scan_down_diffuse(R, T, src_dn, source, inv):
    """flux_dn(h+1) = (T*flux_dn(h) + R*source(h+1) + src_dn(h))*inv; flux_dn(1) = 0."""

    def body(f, xs):
        r, t, sd, s_next, iv = xs
        f_new = (t * f + r * s_next + sd) * iv
        return f_new, f_new

    f0 = jnp.zeros_like(T[0])
    _, f = jax.lax.scan(body, f0, (R, T, src_dn, source[1:], inv))
    return jnp.concatenate([f0[None], f], axis=0)


def fluxes_no_scattering_lw(T, su, sd, emission_surf, albedo_surf):
    """calc_fluxes_no_scattering_lw (radiation_adding_ica_lw.F90)."""

    def down(f, xs):
        t, d = xs
        f_new = t * f + d
        return f_new, f_new

    f0 = jnp.zeros_like(T[0])
    _, fdn = jax.lax.scan(down, f0, (T, sd))
    fdn = jnp.concatenate([f0[None], fdn], axis=0)
    fup_surf = emission_surf + albedo_surf * (fdn[-1])

    def up(f, xs):
        t, u = xs
        f_new = t * f + u
        return f_new, f_new

    _, fup = jax.lax.scan(up, fup_surf, (T, su), reverse=True)
    return jnp.concatenate([fup, fup_surf[None]], axis=0), fdn


def fast_adding_lw(R, T, su, sd, emission_surf, albedo_surf, w_cloud):
    """fast_adding_ica_lw with a soft cloud-top (vote M5a).

    Clear layers carry R = 0 exactly (full-adding formulas then equal the
    oracle's clear-layer formulas bitwise).  Above the cloud top the oracle
    uses flux_up(h) = T(h)*flux_up(h+1) + source_up(h); this is blended with
    albedo(h)*flux_dn(h) + source(h) by a(h) = prod_{k<=h} (1 - w_cloud(k)),
    which is 1 exactly above the (hard) cloud top and 0 from it down.
    """
    albedo, source, inv = _scan_up_albedo_source(R, T, su, sd, albedo_surf, emission_surf)
    fdn = _scan_down_diffuse(R, T, sd, source, inv)
    a_clear_above = jnp.cumprod(1.0 - w_cloud)  # (nlev,): a(h), h = 1..nlev
    fup_adding = albedo * fdn + source  # (nlev+1, ng)

    def up(f, xs):
        t, u, fa, a = xs
        f_new = blend(a, t * f + u, fa)
        return f_new, f_new

    _, fup = jax.lax.scan(
        up, fup_adding[-1], (T, su, fup_adding[:-1], a_clear_above), reverse=True
    )
    return jnp.concatenate([fup, fup_adding[-1:]], axis=0), fdn


def adding_sw(incoming_toa, albedo_diffuse, albedo_direct, mu0, R, T, ref_dir, t_dir_diff, t_dir_dir):
    """adding_ica_sw (radiation_adding_ica_sw.F90).  Returns flux_up,
    flux_dn_diffuse, flux_dn_direct on half levels (direct times mu0)."""

    def down_dir(f, t):
        f_new = f * t
        return f_new, f_new

    _, fdir = jax.lax.scan(down_dir, incoming_toa, t_dir_dir)
    fdir = jnp.concatenate([incoming_toa[None], fdir], axis=0)  # normal to beam
    source_surf = albedo_direct * fdir[-1] * mu0
    albedo, source, inv = _sw_albedo_source(R, T, ref_dir, t_dir_diff, fdir, albedo_diffuse, source_surf)
    fdiff = _scan_down_diffuse(R, T, t_dir_diff * fdir[:-1], source, inv)
    fup = albedo * fdiff + source  # flux_up(1) = source(1): albedo*0 + s is exact
    return fup, fdiff, fdir * mu0


def _sw_albedo_source(R, T, ref_dir, t_dir_diff, fdir, albedo_surf, source_surf):
    def body(carry, xs):
        a_below, s_below = carry
        r, t, rd, td, fd = xs
        inv = 1.0 / (1.0 - a_below * r)
        a = r + t * t * a_below * inv
        s = rd * fd + t * (s_below + a_below * td * fd) * inv
        return (a, s), (a, s, inv)

    _, (a, s, inv) = jax.lax.scan(
        body, (albedo_surf, source_surf), (R, T, ref_dir, t_dir_diff, fdir[:-1]), reverse=True
    )
    return (
        jnp.concatenate([a, albedo_surf[None]], axis=0),
        jnp.concatenate([s, source_surf[None]], axis=0),
        inv,
    )


# --------------------------------------------------------------------------
# LW derivatives (radiation_lw_derivatives.F90)
# --------------------------------------------------------------------------

def lw_derivatives_ica(T, flux_up_surf, live=True):
    """calc_lw_derivatives_ica: (nlev+1,) d flux_up(h) / d flux_up(surface).

    ``live`` False marks a call whose result carries exactly zero weight: its
    normalising sum is replaced by 1 so a zero surface flux there cannot put
    0/0 into the gradient (dead-branch guard, as the oracle skips the call).
    Known limit: a live surface up-flux sum below ~1e-154 W/m2 (unphysical)
    overflows the normalisation's VJP."""
    total = seqsum(flux_up_surf, 0)
    g0 = flux_up_surf / jnp.where(live, total, 1.0)

    def body(g, t):
        g_new = g * t
        return g_new, seqsum(g_new, 0)

    _, d = jax.lax.scan(body, g0, T, reverse=True)
    return jnp.concatenate([d, jnp.ones((1,), d.dtype)])


def modify_lw_derivatives_ica(T, flux_up_surf, weight, lw_derivatives, live=True):
    """modify_lw_derivatives_ica: (1-weight)*old + weight*new, surface stays 1."""
    new = lw_derivatives_ica(T, flux_up_surf, live)
    mod = (1.0 - weight) * (lw_derivatives[:-1]) + weight * (new[:-1])
    return jnp.concatenate([mod, jnp.ones((1,), mod.dtype)])


# --------------------------------------------------------------------------
# Solvers, one column
# --------------------------------------------------------------------------

def _cloud_threshold_weights(cloud_fraction, total_cloud_cover, thr, sharpness):
    w_layer = step(cloud_fraction, thr, thr, sharpness, strict=False)
    w_col = step(total_cloud_cover, thr, thr, sharpness, strict=False)
    return w_layer, w_col


def guarded_div(num, den, sharpness, live=True):
    """Oracle ``if (den > 0) x = num/den else x = 0`` with num = 0 wherever
    den = 0: divide by the inactive-exact floor max(den, 2^-500) (vote N1').
    Where ``live`` is False (zero-weight cloudy layer) the result is 0 and the
    divisor 1 (dead-branch guard: valid unused optics, finite VJP).
    Known limit: at sharpness=inf a LIVE positive divisor below ~1e-154 can
    overflow the quotient's VJP (the finite-sharpness floor removes this)."""
    den_eff = jnp.where(live, den, 1.0)
    if sharpness == math.inf:  # exactly the oracle for every den >= 0
        pos = den_eff > 0.0
        q = jnp.where(pos, num / jnp.where(pos, den_eff, 1.0), 0.0)
    else:
        f = den_floor(den_eff)
        q = num / lower_clamp(den_eff, f, f, sharpness)
    return jnp.where(live, q, 0.0)


def _solver_lw_column(od, od_cloud, ssa_cloud, g_cloud, planck_hl, emission, albedo,
                      cloud_fraction, od_scaling, total_cloud_cover, thr, sharpness, band):
    pt, pb = planck_hl[:-1], planck_hl[1:]
    # Clear sky: no scattering (do_lw_aerosol_scattering = false).
    t_clr, su_clr, sd_clr = calc_no_scattering_transmittance_lw(od, pt, pb, sharpness)
    fup_clr, fdn_clr = fluxes_no_scattering_lw(t_clr, su_clr, sd_clr, emission, albedo)
    up_clear = seqsum(fup_clr, 1)
    dn_clear = seqsum(fdn_clr, 1)

    # Cloudy layers (do_lw_cloud_scattering = true, no aerosol scattering).
    w_layer, w_col = _cloud_threshold_weights(cloud_fraction, total_cloud_cover, thr, sharpness)
    # Dead-branch guard: where a cloudy layer's weight is exactly 0 (layer or
    # column clear) its cloud optics are replaced by none, so an unused
    # singular case (e.g. ssa = g = 1) cannot put NaN in the gradient.
    live = ((w_layer > 0.0) & (w_col > 0.0))[:, None]
    od_scaling = jnp.where(live, od_scaling, 0.0)
    od_cloud_new = od_scaling * od_cloud[:, band]
    od_total = od + od_cloud_new
    scat_od = ssa_cloud[:, band] * od_cloud_new
    ssa_total = guarded_div(scat_od, od_total, sharpness, live)
    g_total = guarded_div(g_cloud[:, band] * ssa_cloud[:, band] * od_cloud_new, scat_od, sharpness, live)
    r_cld, t_cld, su_cld, sd_cld = calc_ref_trans_lw(od_total, ssa_total, g_total, pt, pb, sharpness)
    wl = w_layer[:, None]
    R = blend(wl, r_cld, 0.0)
    T = blend(wl, t_cld, t_clr)
    su = blend(wl, su_cld, su_clr)
    sd = blend(wl, sd_cld, sd_clr)
    fup, fdn = fast_adding_lw(R, T, su, sd, emission, albedo, w_layer)

    tcc = total_cloud_cover
    up_cld = tcc * (seqsum(fup, 1)) + (1.0 - tcc) * up_clear
    dn_cld = tcc * (seqsum(fdn, 1)) + (1.0 - tcc) * dn_clear

    deriv_cld = lw_derivatives_ica(T, fup[-1], w_col > 0.0)
    # Oracle: if (tcc < 1 - thr) modify; smoothed in the clear fraction (vote M2a).
    one_minus_thr = 1.0 - thr
    w_mod = 1.0 - step(tcc, one_minus_thr, thr, sharpness, strict=False)
    deriv_mod = modify_lw_derivatives_ica(t_clr, fup_clr[-1], 1.0 - tcc, deriv_cld,
                                          (w_mod > 0.0) & (w_col > 0.0))
    deriv_cld = blend(w_mod, deriv_mod, deriv_cld)
    deriv_clr = lw_derivatives_ica(t_clr, fup_clr[-1], w_col < 1.0)

    return {
        "flux_up_lw": blend(w_col, up_cld, up_clear),
        "flux_dn_lw": blend(w_col, dn_cld, dn_clear),
        "flux_up_lw_clear": up_clear,
        "flux_dn_lw_clear": dn_clear,
        "lw_derivative": blend(w_col, deriv_cld, deriv_clr),
        "cloud_cover_lw": tcc,
    }


def _solver_sw_column(od, ssa, g, od_cloud, ssa_cloud, g_cloud, albedo_direct, albedo_diffuse,
                      incoming, cos_sza, cloud_fraction, od_scaling, total_cloud_cover,
                      thr, sharpness, band):
    # Oracle: if (cos_sza > 0) day else all fluxes 0.  Vote N2: weight with
    # its band wholly on the night side (exactly 1 for cos_sza >= 0), day
    # formula evaluated at max(cos_sza, 2^-500).
    if sharpness == math.inf:  # exactly the oracle for every cos_sza
        w_day = (cos_sza > 0.0).astype(cos_sza.dtype)
        mu0 = jnp.where(cos_sza > 0.0, cos_sza, 1.0)
    else:
        w_day = smoothstep_cubic(1.0 + cos_sza * sharpness)
        f = den_floor(cos_sza)
        mu0 = lower_clamp(cos_sza, f, f, sharpness)

    def column_fluxes(rd, td, rdir, tdd_iff, tdd):
        fup, fdiff, fdir = adding_sw(incoming, albedo_diffuse, albedo_direct, mu0, rd, td, rdir, tdd_iff, tdd)
        up = seqsum(fup, 1)
        dn_direct = seqsum(fdir, 1)
        dn = seqsum(fdiff, 1) + dn_direct
        return up, dn, dn_direct, fdiff[-1], fdir[-1]

    clear_props = calc_ref_trans_sw(mu0, od, ssa, g, sharpness)
    up_c, dn_c, dir_c, sdiff_c, sdir_c = column_fluxes(*clear_props)

    w_layer, w_col = _cloud_threshold_weights(cloud_fraction, total_cloud_cover, thr, sharpness)
    live = ((w_layer > 0.0) & (w_col > 0.0) & (w_day > 0.0))[:, None]  # dead-branch guard, as LW
    od_scaling = jnp.where(live, od_scaling, 0.0)
    od_cloud_new = od_scaling * od_cloud[:, band]
    od_total = od + od_cloud_new
    scat_od = ssa * od + ssa_cloud[:, band] * od_cloud_new
    ssa_total = guarded_div(scat_od, od_total, sharpness, live)
    g_total = guarded_div(
        (g * ssa) * od + (g_cloud[:, band] * ssa_cloud[:, band]) * od_cloud_new,
        scat_od, sharpness, live,
    )
    cloudy_props = calc_ref_trans_sw(mu0, od_total, ssa_total, g_total, sharpness)
    wl = w_layer[:, None]
    props = [blend(wl, a, b) for a, b in zip(cloudy_props, clear_props)]
    up, dn, dn_dir, sdiff, sdir = column_fluxes(*props)

    tcc = total_cloud_cover
    mix = lambda x, xc: tcc * x + (1.0 - tcc) * xc  # noqa: E731
    out = {
        "flux_up_sw": blend(w_col, mix(up, up_c), up_c),
        "flux_dn_sw": blend(w_col, mix(dn, dn_c), dn_c),
        "flux_dn_direct_sw": blend(w_col, mix(dn_dir, dir_c), dir_c),
        "flux_up_sw_clear": up_c,
        "flux_dn_sw_clear": dn_c,
        "flux_dn_direct_sw_clear": dir_c,
        "sw_dn_diffuse_surf_g": blend(w_col, mix(sdiff, sdiff_c), sdiff_c),
        "sw_dn_direct_surf_g": blend(w_col, mix(sdir, sdir_c), sdir_c),
        "sw_dn_diffuse_surf_clear_g": sdiff_c,
        "sw_dn_direct_surf_clear_g": sdir_c,
    }
    out = {k: blend(w_day, v, 0.0) for k, v in out.items()}
    out["cloud_cover_sw"] = blend(w_day, tcc, -1.0)  # -1 = flux_type init value
    return out


# --------------------------------------------------------------------------
# Public entry points (leading column axis)
# --------------------------------------------------------------------------

def solver_mcica_lw(od, od_cloud, ssa_cloud, g_cloud, planck_hl, emission, albedo,
                    cloud_fraction, od_scaling, total_cloud_cover, *,
                    band, cloud_fraction_threshold, sharpness=100.0):
    """solver_mcica_lw.  Shapes: od (ncol, nlev, ng); *_cloud (ncol, nlev, nband);
    planck_hl (ncol, nlev+1, ng); emission/albedo (ncol, ng); cloud_fraction
    (ncol, nlev); od_scaling (ncol, nlev, ng); total_cloud_cover (ncol,).
    ``od_scaling`` must be finite (its value is unused where the column or
    layer is clear).  ``band``: band of each g-point, (ng,)."""
    f = lambda *a: _solver_lw_column(*a, cloud_fraction_threshold, sharpness, band)  # noqa: E731
    return jax.vmap(f)(od, od_cloud, ssa_cloud, g_cloud, planck_hl, emission, albedo,
                       cloud_fraction, od_scaling, total_cloud_cover)


def solver_mcica_sw(od, ssa, g, od_cloud, ssa_cloud, g_cloud, albedo_direct, albedo_diffuse,
                    incoming_sw, cos_sza, cloud_fraction, od_scaling, total_cloud_cover, *,
                    band, cloud_fraction_threshold, sharpness=100.0):
    """solver_mcica_sw.  Shapes as solver_mcica_lw; albedos and
    incoming_sw (ncol, ng); cos_sza (ncol,).  Returns broadband profiles, the
    g-point surface fluxes and cloud_cover_sw."""
    f = lambda *a: _solver_sw_column(*a, cloud_fraction_threshold, sharpness, band)  # noqa: E731
    return jax.vmap(f)(od, ssa, g, od_cloud, ssa_cloud, g_cloud, albedo_direct, albedo_diffuse,
                       incoming_sw, cos_sza, cloud_fraction, od_scaling, total_cloud_cover)


def surface_spectral_sw(sw_out, sw_albedo_wavelength_bound, i_sw_albedo_index, band):
    """flux_type%calc_surface_spectral, SW part, for do_surface_sw_spectral_flux
    and do_canopy_fluxes_sw with the weighted (not nearest) albedo mapping."""
    direct = _band_sums(sw_out["sw_dn_direct_surf_g"], band)
    total = _band_sums(sw_out["sw_dn_diffuse_surf_g"], band) + direct
    direct_c = _band_sums(sw_out["sw_dn_direct_surf_clear_g"], band)
    total_c = _band_sums(sw_out["sw_dn_diffuse_surf_clear_g"], band) + direct_c
    weights = sw_albedo_weights(sw_albedo_wavelength_bound, i_sw_albedo_index)
    ninput, nband = weights.shape
    can_diff = [jnp.zeros_like(total[..., 0]) for _ in range(ninput)]
    can_dir = [jnp.zeros_like(total[..., 0]) for _ in range(ninput)]
    for jband in range(nband):
        for jalb in range(ninput):
            w = weights[jalb, jband]
            if w != 0.0:
                can_diff[jalb] = can_diff[jalb] + w * total[..., jband]
                can_dir[jalb] = can_dir[jalb] + w * direct[..., jband]
    can_dir = jnp.stack(can_dir, axis=-1)
    can_diff = jnp.stack(can_diff, axis=-1) - can_dir
    return {
        "spectral_flux_dn_sw_surf": total,
        "spectral_flux_dn_direct_sw_surf": direct,
        "spectral_flux_dn_sw_surf_clear": total_c,
        "spectral_flux_dn_direct_sw_surf_clear": direct_c,
        "canopy_flux_dn_diffuse_sw_surf": can_diff,
        "canopy_flux_dn_direct_sw_surf": can_dir,
    }


def sw_albedo_to_gpoints(albedo_intervals, sw_albedo_wavelength_bound, i_sw_albedo_index, band):
    """get_albedos weighted branch: (ncol, ninput) -> (ncol, ng_sw).  Used to
    check the ported weights against the oracle's saved g-point albedo."""
    weights = sw_albedo_weights(sw_albedo_wavelength_bound, i_sw_albedo_index)
    ninput, nband = weights.shape
    bands = []
    for jband in range(nband):
        acc = jnp.zeros_like(albedo_intervals[..., 0])
        for jalb in range(ninput):
            if weights[jalb, jband] != 0.0:
                acc = acc + weights[jalb, jband] * albedo_intervals[..., jalb]
        bands.append(acc)
    return jnp.stack(bands, axis=-1)[..., band]
