"""ecRad 49r1 Tripleclouds longwave and shortwave solvers, JAX port.

Line-by-line port (OpenIFS 49r1, 8f6f722, compiled sources in
``oracle_audit/ifs_port/ecrad/bld49/src``) of

* ``radiation_regions.F90`` calc_region_properties, nregions = 3, gamma PDF
  (``cloud_pdf_shape_name = "Gamma"``, the 49r1 default lower-fraction rule,
  not OLD_GAMMA_REGION_BEHAVIOUR);
* ``radiation_overlap.F90`` calc_overlap_matrices with the alpha (not beta)
  overlap matrix (``use_beta_overlap = false``) and the cloud cover it returns;
* ``radiation_tripleclouds_lw.F90`` for do_lw_aerosol_scattering = false,
  do_lw_cloud_scattering = true, do_lw_derivatives = true (with
  ``radiation_lw_derivatives.F90`` calc_lw_derivatives_region);
* ``radiation_tripleclouds_sw.F90`` for do_sw_delta_scaling_with_gases = false,
  do_sw_direct = true, do_clear = true.

The SW surface band and canopy fluxes are solver-independent and come from
:func:`.mcica_solvers.surface_spectral_sw` on this solver's g-point surface
fluxes.  Inputs are the oracle solver inputs (radiative properties per
g-point, cloud optics per band, cloud fraction, fractional standard deviation
and overlap parameter); no reordering of g-points (Tripleclouds keeps the
RRTMG order, radiation_ifs_rrtm.F90:130-141).

Layout: one column is ``(nlev, ng)``; region-resolved arrays are ``(nlev, 3,
ng)`` with region 0 = clear, 1 = optically thin, 2 = thick cloud; overlap
matrices are ``(nlev+1, 3, 3)`` on half levels.  Levels top-down (0 = TOA).

Restructuring (vote T11): the oracle loops only from the first cloudy layer
(``i_cloud_top``) and skips the region work in clear layers
(``is_clear_sky_layer``).  Here every layer is computed: clear layers give
the cloudy regions R = 0, T = 1, S = 0 and zero fluxes (blends with exact 0/1
weights at ``sharpness = inf``), and above the cloud top the cloudy regions
carry no upward flux.  The remaining oracle branches change rounding only and
are not reproduced (measured, mutation runs): between two clear layers the
overlap matrices are the identity on the clear region, and above the cloud
top the adding solution equals the oracle's clear recursion algebraically
(clear layers have R = 0).
Oracle quirks kept (votes T12, T17; both measured on the oracle):

* in a column with no cloudy layer the LW derivative is normalised by the TOA
  (not surface) spectral upward flux, because the oracle's ``flux_up`` array
  then still holds its last (TOA) values;
* the SW TOA upward flux multiplies the top layer's region direct fluxes by
  the albedo of the regions ABOVE TOA (clear only), so the cloudy part of the
  top layer reflects nothing at TOA (overcast top layer: 0 W/m2 at TOA, 157
  W/m2 one level down).

Smoothing (user rule; one static ``sharpness``, 100 in production, ``inf`` =
oracle): votes T1-T10 in the PR body.  Region switch cf >= threshold and the
overlap divisions frac >= threshold are centred steps of relative width;
the lower-fraction clamps and the 1e-6 cover floor are inactive-exact clamps;
the pair cover max is :func:`max_kink_exact` (exact where the two
fractions are equal); layer presence cf > 0
and op >= 0 use :func:`.mcica_generator.step_pos` (the McICA generator keeps it
only for op; its cloudy-range presence is exact, part of the sampling); the
SW night test mu0 < 1e-10 is a band wholly on the night side; the LW
ssa/asymmetry divisions use :func:`.mcica_solvers.guarded_div`.

Fortran SUM / matrix-vector products are emulated as sequential left-to-right
adds (as vote M6a for the McICA solvers).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.radiation.ecrad_ifs._libm import exp_libm
from legoesm.atmosphere.physics.radiation.ecrad_ifs.mcica_generator import step_pos
from legoesm.atmosphere.physics.radiation.ecrad_ifs.mcica_solvers import (
    den_floor,
    fluxes_no_scattering_lw,
    guarded_div,
    seqsum,
)
from legoesm.atmosphere.physics.radiation.ecrad_ifs.two_stream import (
    blend,
    calc_no_scattering_transmittance_lw,
    calc_ref_trans_lw,
    calc_ref_trans_sw,
    lower_clamp,
    step,
    upper_clamp,
)
from legoesm.core.smooth import smoothstep_cubic

N_REGIONS = 3

# --- Gamma-PDF region properties (radiation_regions.F90, 49r1 parameters) ---
_MIN_GAMMA_OD_SCALING = 0.025
_MIN_LOWER_FRAC = 0.5
_MAX_LOWER_FRAC = 0.9
_FSD_AT_MIN_LOWER_FRAC = 1.5
_FSD_AT_MAX_LOWER_FRAC = 3.725
_LOWER_FRAC_FSD_GRADIENT = ((_MAX_LOWER_FRAC - _MIN_LOWER_FRAC)
                            / (_FSD_AT_MAX_LOWER_FRAC - _FSD_AT_MIN_LOWER_FRAC))
_LOWER_FRAC_FSD_INTERCEPT = _MIN_LOWER_FRAC - _FSD_AT_MIN_LOWER_FRAC * _LOWER_FRAC_FSD_GRADIENT
# Scale of the lower-fraction clamps (vote T2): a fraction, scale 1.
_LOWER_FRAC_CLAMP_SCALE = 1.0
# --- Alpha overlap matrix (radiation_overlap.F90 calc_alpha_overlap_matrix) ---
# Floor on the cloud cover a region fraction is divided by.
_OVERLAP_CF_FLOOR = 1.0e-6
# --- Night test of radiation_tripleclouds_sw.F90 (mu0 < 1e-10 -> no SW) ---
_MIN_MU0 = 1.0e-10

__physics_contract__ = {
    "summary": (
        "ecRad 49r1 Tripleclouds longwave (cloud scattering, no aerosol "
        "scattering, Hogan-Bozzo region derivatives) and shortwave (direct + "
        "diffuse, no gas delta scaling) solvers: one clear and two cloudy "
        "regions per layer (gamma-PDF split), exponential-random alpha overlap "
        "matrices between layers, adding method per region. Port of "
        "radiation_regions/overlap/tripleclouds_lw/tripleclouds_sw.F90 and "
        "calc_lw_derivatives_region (OpenIFS 49r1)."
    ),
    "inputs": {
        "od_lw": "1 (gas, per g-point)", "planck_hl": "W/m^2 (per g-point)",
        "lw_emission": "W/m^2", "lw_albedo": "1",
        "od_sw": "1", "ssa_sw": "1", "asymmetry_sw": "1",
        "incoming_sw": "W/m^2 (TOA normal to beam, per g-point)",
        "sw_albedo": "1", "sw_albedo_direct": "1", "cos_sza": "1",
        "od/ssa/asymmetry_*_cloud": "1 (per band)", "cloud_fraction": "1",
        "fractional_std": "1 (in-cloud water content FSD)",
        "overlap_param": "1 (between adjacent layers, nlev-1)",
    },
    "outputs": {
        "flux_up_lw/flux_dn_lw (+_clear)": "W/m^2",
        "lw_derivative": "1 (d flux_up / d surface flux_up)",
        "flux_up_sw/flux_dn_sw/flux_dn_direct_sw (+_clear)": "W/m^2",
        "sw_dn_(diffuse|direct)_surf(_clear)_g": "W/m^2 per g-point",
        "cloud_cover_lw/sw": "1",
    },
    "sign_convention": (
        "All fluxes positive in their named direction on half levels, top-down "
        "(index 0 = TOA). SW direct flux is per unit horizontal area (times "
        "cos_sza). Region fractions sum to 1 per layer; the overlap matrices "
        "move flux between regions at half levels without creating it "
        "(columns of v sum to 1). Energy: column heating = TOA net - surface net."
    ),
    "conserves": ["energy"],
    "differentiable": True,
    "reference": (
        "Shonk & Hogan (2008) J. Climate 21, 2352-2370 (Tripleclouds); Hogan "
        "et al. (2016) GMD 9 (gamma regions, ecRad); oracle "
        "radiation_tripleclouds_lw/sw.F90, radiation_regions.F90, "
        "radiation_overlap.F90 (OpenIFS 49r1 8f6f722)."
    ),
    "idealized_test": (
        "tests/atmosphere/test_ecrad_ifs_tripleclouds_solver_oracle.py: oracle "
        "match at 1e-12 (hard form), overcast/clear limits, overlap-matrix "
        "flux conservation and gradient checks."
    ),
}


# --------------------------------------------------------------------------
# Region properties and overlap matrices (one column)
# --------------------------------------------------------------------------

def region_properties(cloud_fraction, fractional_std, threshold, sharpness):
    """calc_region_properties, nreg = 3, do_gamma.  Returns region fractions
    (nlev, 3) and the optical-depth scalings of regions 1-2 (nlev, 2)."""
    cf = cloud_fraction
    w = step(cf, threshold, threshold, sharpness, strict=False)  # oracle: clear if cf < thr
    live = w > 0.0
    cfl = jnp.where(live, cf, 1.0)  # dead-branch guard: cloudy formulas at a safe cf
    fsd = fractional_std
    lower = upper_clamp(_LOWER_FRAC_FSD_INTERCEPT + fsd * _LOWER_FRAC_FSD_GRADIENT,
                        _MAX_LOWER_FRAC, _LOWER_FRAC_CLAMP_SCALE, sharpness)
    lower = lower_clamp(lower, _MIN_LOWER_FRAC, _LOWER_FRAC_CLAMP_SCALE, sharpness)
    r1 = 1.0 - cfl
    r2 = cfl * lower
    s2 = _MIN_GAMMA_OD_SCALING + (1.0 - _MIN_GAMMA_OD_SCALING) * exp_libm(
        -(fsd * (1.0 + 0.5 * fsd * (1.0 + 0.5 * fsd))))
    r3 = 1.0 - r1 - r2
    s3 = (cfl - r2 * s2) / r3
    fracs = jnp.stack([blend(w, r1, 1.0), blend(w, r2, 0.0), blend(w, r3, 0.0)], axis=-1)
    scaling = jnp.stack([blend(w, s2, 1.0), blend(w, s3, 1.0)], axis=-1)
    return fracs, scaling


def max_kink_exact(a, b, sharpness):
    """``max(a, b)`` of two fractions in [0, 1] (vote T4'): ``(a + b + sabs(a - b))/2``
    with the C1 absolute value ``sabs(d) = d^2 (2w - |d|)/w^2`` for ``|d| < w``,
    ``|d|`` outside, ``w = min(a + b, 2 - a - b)/sharpness`` (the band of the
    generator's max_c1).  Exact at ``a = b`` (where the thin/thick shares of
    adjacent layers usually sit), at 0, at 1 and for ``|a - b| >= w``; inside
    the band at most ``4w/27`` below the max.  The Huber form max_c1 is exact
    only for ``|a - b| >= w`` and biased by ``w/4`` at ``a = b`` (measured:
    7.7e-2 worst flux gap at sharpness 100 vs 1.4e-6 with this form)."""
    if sharpness == math.inf:
        return jnp.maximum(a, b)
    d = a - b
    w = jnp.minimum(a + b, 2.0 - a - b) / sharpness
    band = jnp.abs(d) < w
    ws = jnp.where(band, w, 1.0)
    sabs = jnp.where(band, d * d * (2.0 * ws - jnp.abs(d)) / (ws * ws), jnp.abs(d))
    return 0.5 * (a + b + sabs)


def _floor_cover(x, sharpness):
    return lower_clamp(x, _OVERLAP_CF_FLOOR, _OVERLAP_CF_FLOOR, sharpness)


def alpha_overlap_matrix(op, op_inhom, frac_upper, frac_lower, sharpness):
    """calc_alpha_overlap_matrix, nreg = 3.  frac_* (..., 3); returns M (..., 3, 3)
    with M[..., upper region, lower region]."""
    fu, fl = frac_upper, frac_lower
    cf_upper = fu[..., 1] + fu[..., 2]
    cf_lower = fl[..., 1] + fl[..., 2]
    pair = (op * max_kink_exact(cf_upper, cf_lower, sharpness)
            + (1.0 - op) * (cf_upper + cf_lower - cf_upper * cf_lower))
    m11 = 1.0 - pair
    one_over_cf = 1.0 / _floor_cover(cf_lower, sharpness)
    m12 = (pair - cf_upper) * fl[..., 1] * one_over_cf
    m13 = (pair - cf_upper) * fl[..., 2] * one_over_cf
    one_over_cf = 1.0 / _floor_cover(cf_upper, sharpness)
    m21 = (pair - cf_lower) * fu[..., 1] * one_over_cf
    m31 = (pair - cf_lower) * fu[..., 2] * one_over_cf
    frac_both = cf_upper + cf_lower - pair
    a = fu[..., 2] / _floor_cover(cf_upper, sharpness)
    b = fl[..., 2] / _floor_cover(cf_lower, sharpness)
    pair = op_inhom * max_kink_exact(a, b, sharpness) + (1.0 - op_inhom) * (a + b - a * b)
    m22 = frac_both * (1.0 - pair)
    m23 = frac_both * (pair - a)
    m32 = frac_both * (pair - b)
    m33 = frac_both * (a + b - pair)
    rows = [jnp.stack(r, axis=-1) for r in ((m11, m12, m13), (m21, m22, m23), (m31, m32, m33))]
    return jnp.stack(rows, axis=-2)


def overlap_matrices(fracs, overlap_param, decorrelation_scaling, threshold, sharpness):
    """calc_overlap_matrices (alpha overlap).  fracs (nlev, 3), overlap_param
    (nlev-1,).  Returns u, v (nlev+1, 3, 3) as the oracle's
    u_matrix(upper, lower, k) and v_matrix(lower, upper, k), and the cloud
    cover 1 - prod_k v(clear, clear, k)."""
    clear = jnp.zeros((1, N_REGIONS), fracs.dtype).at[0, 0].set(1.0)
    fu = jnp.concatenate([clear, fracs], axis=0)  # half level k: layer k-1 above
    fl = jnp.concatenate([fracs, clear], axis=0)  # layer k below
    one = jnp.ones((1,), fracs.dtype)
    op = jnp.concatenate([one, overlap_param, one])  # op = 1 at TOA and surface
    # op(2:nreg) = op**(1/decorrelation_scaling) where op >= 0 (value-identical
    # to the generator's op > 0 at op = 0), else op; = 1 at TOA and surface.
    use_pow = step_pos(op, 1.0 / sharpness, sharpness)
    op_pow = jnp.power(jnp.where(op > 0.0, op, 1.0), 1.0 / decorrelation_scaling)
    op_inhom = use_pow * op_pow + (1.0 - use_pow) * op
    m = alpha_overlap_matrix(op, op_inhom, fu, fl, sharpness)  # [k, upper, lower]

    def divide(frac):
        w = step(frac, threshold, threshold, sharpness, strict=False)
        return w, jnp.where(w > 0.0, frac, 1.0)

    wl, fl_safe = divide(fl)
    wu, fu_safe = divide(fu)
    u = blend(wl[:, None, :], m / fl_safe[:, None, :], 0.0)
    v = jnp.swapaxes(blend(wu[:, :, None], m / fu_safe[:, :, None], 0.0), 1, 2)
    cover = 1.0 - _seqprod(v[:, 0, 0])
    return u, v, cover


def _seqprod(x):
    """Fortran PRODUCT, left to right."""
    total, _ = jax.lax.scan(lambda c, y: (c * y, None), x[0], x[1:])
    return total


def _matvec(a, b):
    """singlemat_x_vec: out(i) = sum_j a(i, j) b(j), j ascending; a (3, 3), b (3, ng)."""
    return jnp.stack([(a[i, 0] * b[0] + a[i, 1] * b[1]) + a[i, 2] * b[2]
                      for i in range(N_REGIONS)])


def _vec_mat(b, v):
    """total_albedo(jreg) = sum_jreg2 below(jreg2) * v(jreg2, jreg), jreg2 ascending."""
    return jnp.stack([(b[0] * v[0, j] + b[1] * v[1, j]) + b[2] * v[2, j]
                      for j in range(N_REGIONS)])


def _region_sum(x):
    """sum(sum(x, 1)) of a (3, ng) region array: per-region g sums, then regions."""
    s = seqsum(x, 1)
    return (s[0] + s[1]) + s[2]


def _sum_regions(x):
    """sum(x, 2) of a (3, ng) array: regions summed per g-point."""
    return (x[0] + x[1]) + x[2]


def _cloud_presence(cloud_fraction, threshold, sharpness):
    """Layer presence: cloud_fraction > 0 (the oracle's not is_clear_sky_layer)."""
    return step_pos(cloud_fraction, threshold / sharpness, sharpness)


def _keep(w_layer):
    """(3, 1) weights: region 0 always, cloudy regions only in a cloudy layer."""
    return jnp.stack([jnp.ones_like(w_layer), w_layer, w_layer])[:, None]


def _with_regions(x, r1, r2):
    return jnp.stack([x, r1, r2], axis=1)


# --------------------------------------------------------------------------
# Longwave (one column)
# --------------------------------------------------------------------------

def _solver_lw_column(od, od_cloud, ssa_cloud, g_cloud, planck_hl, emission, albedo,
                      cloud_fraction, fractional_std, overlap_param,
                      threshold, decorrelation_scaling, sharpness, band):
    pt, pb = planck_hl[:-1], planck_hl[1:]
    fracs, scaling = region_properties(cloud_fraction, fractional_std, threshold, sharpness)
    u, v, cover = overlap_matrices(fracs, overlap_param, decorrelation_scaling, threshold,
                                   sharpness)
    w = _cloud_presence(cloud_fraction, threshold, sharpness)
    wl = w[:, None]
    live = wl > 0.0

    # Clear sky (do_lw_aerosol_scattering = false).
    t_clr, su_clr, sd_clr = calc_no_scattering_transmittance_lw(od, pt, pb, sharpness)
    fup_clr, fdn_clr = fluxes_no_scattering_lw(t_clr, su_clr, sd_clr, emission, albedo)

    # Cloudy regions 1-2 (do_lw_cloud_scattering = true).
    def cloudy(jr):
        s = jnp.where(live, scaling[:, jr][:, None], 0.0)  # dead-branch guard
        od_cloud_new = od_cloud[:, band] * s
        od_total = od + od_cloud_new
        ssa_total = guarded_div(ssa_cloud[:, band] * od_cloud_new, od_total, sharpness, live)
        g_total = guarded_div(g_cloud[:, band] * ssa_cloud[:, band] * od_cloud_new,
                              ssa_total * od_total, sharpness, live)
        r, t, su, sd = calc_ref_trans_lw(od_total, ssa_total, g_total, pt, pb, sharpness)
        f = fracs[:, jr + 1][:, None]
        return (blend(wl, r, 0.0), blend(wl, t, 1.0),
                blend(wl, f * su, 0.0), blend(wl, f * sd, 0.0))

    c1, c2 = cloudy(0), cloudy(1)
    f0 = fracs[:, 0][:, None]
    zero = jnp.zeros_like(od)
    R = _with_regions(zero, c1[0], c2[0])
    T = _with_regions(t_clr, c1[1], c2[1])
    SU = _with_regions(blend(wl, f0 * su_clr, su_clr), c1[2], c2[2])
    SD = _with_regions(blend(wl, f0 * sd_clr, sd_clr), c1[3], c2[3])

    # Upward adding pass with overlap at half levels.
    a_surf = jnp.broadcast_to(albedo, (N_REGIONS,) + albedo.shape)
    s_surf = fracs[-1][:, None] * emission[None, :]

    def up(carry, xs):
        a_b, s_b = carry
        r, t, su, sd, wlay, uk, vk = xs
        inv = 1.0 / (1.0 - a_b * r)
        alb = r + t * t * a_b * inv
        src = su + t * (s_b + a_b * sd) * inv
        keep = _keep(wlay)
        alb = blend(keep, alb, 0.0)
        src = blend(keep, src, 0.0)
        a_h = _vec_mat(alb, vk)
        s_h = _matvec(uk, src)
        return (a_h, s_h), (a_h, s_h)

    _, (a_h, s_h) = jax.lax.scan(up, (a_surf, s_surf),
                                 (R, T, SU, SD, w, u[:-1], v[:-1]), reverse=True)
    alb = jnp.concatenate([a_h, a_surf[None]], axis=0)  # (nlev+1, 3, ng)
    src = jnp.concatenate([s_h, s_surf[None]], axis=0)

    # Downward pass.
    def down(fd, xs):
        r, t, sd, a_n, s_n, wlay, vk = xs
        fd = (t * fd + r * s_n + sd) / (1.0 - r * a_n)
        fu = s_n + fd * a_n
        keep = _keep(wlay)
        fd = blend(keep, fd, 0.0)
        fu = blend(keep, fu, 0.0)
        fd = _matvec(vk, fd)
        return fd, (fu, fd)

    fd0 = jnp.zeros_like(a_surf)
    _, (fu_h, fd_h) = jax.lax.scan(down, fd0, (R, T, SD, alb[1:], src[1:], w, v[1:]))
    fu_all = jnp.concatenate([src[:1], fu_h], axis=0)  # half level 0: source only (fd = 0)
    fd_all = jnp.concatenate([fd0[None], fd_h], axis=0)

    # Above the cloud top the cloudy regions carry no upward flux (oracle).
    clear_to = jnp.cumprod(1.0 - w)  # 1 where layers 0..h are all clear
    clear_above = jnp.concatenate([jnp.ones((1,), w.dtype), clear_to])  # layers 0..h-1
    m23 = (1.0 - clear_above)[:, None]
    fu_all = jnp.stack([fu_all[:, 0], blend(m23, fu_all[:, 1], 0.0),
                        blend(m23, fu_all[:, 2], 0.0)], axis=1)

    flux_up = jax.vmap(_region_sum)(fu_all)
    flux_dn = jax.vmap(_region_sum)(fd_all)

    # Derivatives: flux_up at the surface, or at TOA in a cloud-free column (oracle quirk).
    no_cloud = clear_to[-1]
    flux_up_surf = blend(no_cloud, fu_all[0, 0], _sum_regions(fu_all[-1]))
    deriv = _lw_derivatives_region(T, u, flux_up_surf)

    return {
        "flux_up_lw": flux_up,
        "flux_dn_lw": flux_dn,
        "flux_up_lw_clear": seqsum(fup_clr, 1),
        "flux_dn_lw_clear": seqsum(fdn_clr, 1),
        "lw_derivative": deriv,
        "cloud_cover_lw": cover,
    }


def _lw_derivatives_region(T, u, flux_up_surf):
    """calc_lw_derivatives_region, nreg = 3: (nlev+1,) d flux_up(h) / d flux_up(surface)."""
    g0 = jnp.zeros((N_REGIONS,) + flux_up_surf.shape, flux_up_surf.dtype)
    g0 = g0.at[0].set(flux_up_surf / seqsum(flux_up_surf, 0))

    def body(g, xs):
        t, uk = xs
        g = _matvec(uk, g) * t
        partial = (g[0] + g[1]) + g[2]
        return g, seqsum(partial, 0)

    _, d = jax.lax.scan(body, g0, (T, u[1:]), reverse=True)
    return jnp.concatenate([d, jnp.ones((1,), d.dtype)])


# --------------------------------------------------------------------------
# Shortwave (one column)
# --------------------------------------------------------------------------

def _day_weight(cos_sza, sharpness):
    """Oracle: no SW if mu0 < 1e-10.  Band wholly on the night side (vote T8);
    the day formulas run at max(mu0, 2^-500) as the McICA solver (vote N2)."""
    if sharpness == math.inf:
        day = cos_sza >= _MIN_MU0
        return day.astype(cos_sza.dtype), jnp.where(day, cos_sza, 1.0)
    w = smoothstep_cubic(1.0 + (cos_sza - _MIN_MU0) * (sharpness / _MIN_MU0))
    f = den_floor(cos_sza)
    return w, lower_clamp(cos_sza, f, f, sharpness)


def _solver_sw_column(od, ssa, g, od_cloud, ssa_cloud, g_cloud, albedo_direct, albedo_diffuse,
                      incoming, cos_sza, cloud_fraction, fractional_std, overlap_param,
                      threshold, decorrelation_scaling, sharpness, band):
    w_day, mu0 = _day_weight(cos_sza, sharpness)
    fracs, scaling = region_properties(cloud_fraction, fractional_std, threshold, sharpness)
    _, v, cover = overlap_matrices(fracs, overlap_param, decorrelation_scaling, threshold,
                                   sharpness)
    w = _cloud_presence(cloud_fraction, threshold, sharpness)
    wl = w[:, None]
    live = (wl > 0.0) & (w_day > 0.0)  # dead-branch guard: clear layer or night (oracle skips both)

    clear = calc_ref_trans_sw(mu0, od, ssa, g, sharpness)  # R, T, ref_dir, t_dir_diff, t_dir_dir

    def cloudy(jr):
        s = jnp.where(live, scaling[:, jr][:, None], 0.0)
        scat_od = od * ssa
        scat_od_cloud = od_cloud[:, band] * ssa_cloud[:, band] * s
        od_total = od + od_cloud[:, band] * s
        scat_total = scat_od + scat_od_cloud
        # No guard in the oracle (cloudy layers only); safe divisors where unused.
        ssa_total = scat_total / jnp.where(live, od_total, 1.0)
        g_total = (scat_od * g + scat_od_cloud * g_cloud[:, band]) / jnp.where(live, scat_total, 1.0)
        props = calc_ref_trans_sw(mu0, od_total, ssa_total, g_total, sharpness)
        return [blend(wl, p, q) for p, q in zip(props, clear)]

    regions = [clear, cloudy(0), cloudy(1)]
    R, T, RDIR, TDD, TDIR = (jnp.stack([reg[i] for reg in regions], axis=1) for i in range(5))
    rc, tc, rdc, tddc, tdc = clear

    a_surf1 = albedo_diffuse
    ad_surf1 = mu0 * albedo_direct
    w_last = w[-1]
    a_surf = jnp.stack([a_surf1, blend(w_last, a_surf1, 0.0), blend(w_last, a_surf1, 0.0)])
    ad_surf = jnp.stack([ad_surf1, blend(w_last, ad_surf1, 0.0), blend(w_last, ad_surf1, 0.0)])

    def up(carry, xs):
        a_b, ad_b, ac_b, adc_b = carry
        r, t, rd, tdd, tdir, r0, t0, rd0, tdd0, tdir0, wlay, vk = xs
        inv = 1.0 / (1.0 - ac_b * r0)
        ac = r0 + t0 * t0 * ac_b * inv
        adc = rd0 + (tdir0 * adc_b + tdd0 * ac_b) * t0 * inv
        inv = 1.0 / (1.0 - a_b * r)
        alb = r + t * t * a_b * inv
        albd = rd + (tdir * ad_b + tdd * a_b) * t * inv
        keep = _keep(wlay)
        a_h = _vec_mat(blend(keep, alb, 0.0), vk)
        ad_h = _vec_mat(blend(keep, albd, 0.0), vk)
        return (a_h, ad_h, ac, adc), (a_h, ad_h, ac, adc)

    _, (a_h, ad_h, ac_h, adc_h) = jax.lax.scan(
        up, (a_surf, ad_surf, a_surf1, ad_surf1),
        (R, T, RDIR, TDD, TDIR, rc, tc, rdc, tddc, tdc, w, v[:-1]), reverse=True)
    alb = jnp.concatenate([a_h, a_surf[None]], axis=0)
    albd = jnp.concatenate([ad_h, ad_surf[None]], axis=0)
    alb_c = jnp.concatenate([ac_h, a_surf1[None]], axis=0)
    albd_c = jnp.concatenate([adc_h, ad_surf1[None]], axis=0)

    dd0 = incoming[None, :] * fracs[0][:, None]
    fu_top = dd0 * albd[0]
    fuc_top = incoming * albd_c[0]

    def down(carry, xs):
        fd, dd, fdc, ddc = carry
        r, t, tdd, tdir, r0, t0, tdd0, tdir0, a_n, ad_n, ac_n, adc_n, wlay, vk = xs
        fdc = (t0 * fdc + ddc * (tdir0 * adc_n * r0 + tdd0)) / (1.0 - r0 * ac_n)
        ddc = tdir0 * ddc
        fuc = ddc * adc_n + fdc * ac_n
        fd = (t * fd + dd * (tdir * ad_n * r + tdd)) / (1.0 - r * a_n)
        dd = tdir * dd
        fu = dd * ad_n + fd * a_n
        keep = _keep(wlay)
        fd, dd, fu = (blend(keep, x, 0.0) for x in (fd, dd, fu))
        fd = _matvec(vk, fd)
        dd = _matvec(vk, dd)
        return (fd, dd, fdc, ddc), (fu, fd, dd, fuc, fdc, ddc)

    zero = jnp.zeros_like(dd0)
    (fd_s, dd_s, fdc_s, ddc_s), (fu_h, fd_h, dd_h, fuc_h, fdc_h, ddc_h) = jax.lax.scan(
        down, (zero, dd0, zero[0], incoming),
        (R, T, TDD, TDIR, rc, tc, tddc, tdc, alb[1:], albd[1:], alb_c[1:], albd_c[1:],
         w, v[1:]))

    up_reg = jax.vmap(_region_sum)(fu_h)
    direct = mu0 * jax.vmap(_region_sum)(dd_h)
    flux_up = jnp.concatenate([_region_sum(fu_top)[None], up_reg])
    dn_direct = jnp.concatenate([(mu0 * _region_sum(dd0))[None], direct])
    flux_dn = dn_direct.at[1:].add(jax.vmap(_region_sum)(fd_h))
    up_c = jnp.concatenate([seqsum(fuc_top, 0)[None], seqsum(fuc_h, 1)])
    dir_c = jnp.concatenate([(mu0 * seqsum(incoming, 0))[None], mu0 * seqsum(ddc_h, 1)])
    dn_c = dir_c.at[1:].add(seqsum(fdc_h, 1))

    out = {
        "flux_up_sw": flux_up,
        "flux_dn_sw": flux_dn,
        "flux_dn_direct_sw": dn_direct,
        "flux_up_sw_clear": up_c,
        "flux_dn_sw_clear": dn_c,
        "flux_dn_direct_sw_clear": dir_c,
        "sw_dn_diffuse_surf_g": _sum_regions(fd_s),
        "sw_dn_direct_surf_g": mu0 * _sum_regions(dd_s),
        "sw_dn_diffuse_surf_clear_g": fdc_s,
        "sw_dn_direct_surf_clear_g": mu0 * ddc_s,
    }
    out = {k: blend(w_day, x, 0.0) for k, x in out.items()}
    out["cloud_cover_sw"] = cover
    return out


# --------------------------------------------------------------------------
# Public entry points (leading column axis)
# --------------------------------------------------------------------------

def solver_tripleclouds_lw(od, od_cloud, ssa_cloud, g_cloud, planck_hl, emission, albedo,
                           cloud_fraction, fractional_std, overlap_param, *,
                           band, cloud_fraction_threshold, decorrelation_scaling, sharpness=100.0):
    """solver_tripleclouds_lw.  Shapes: od (ncol, nlev, 140); *_cloud (ncol, nlev, 16);
    planck_hl (ncol, nlev+1, 140); emission/albedo (ncol, 140); cloud_fraction,
    fractional_std (ncol, nlev); overlap_param (ncol, nlev-1)."""
    f = lambda *a: _solver_lw_column(*a, cloud_fraction_threshold,  # noqa: E731
                                     decorrelation_scaling, sharpness, band)
    return jax.vmap(f)(od, od_cloud, ssa_cloud, g_cloud, planck_hl, emission, albedo,
                       cloud_fraction, fractional_std, overlap_param)


def solver_tripleclouds_sw(od, ssa, g, od_cloud, ssa_cloud, g_cloud, albedo_direct,
                           albedo_diffuse, incoming_sw, cos_sza, cloud_fraction,
                           fractional_std, overlap_param, *, band, cloud_fraction_threshold,
                           decorrelation_scaling, sharpness=100.0):
    """solver_tripleclouds_sw.  Shapes as solver_tripleclouds_lw with ng = 112 and
    14 bands; albedos and incoming_sw (ncol, 112); cos_sza (ncol,).  Returns the
    broadband profiles, the g-point surface fluxes (input of
    :func:`.mcica_solvers.surface_spectral_sw`) and cloud_cover_sw."""
    f = lambda *a: _solver_sw_column(*a, cloud_fraction_threshold,  # noqa: E731
                                     decorrelation_scaling, sharpness, band)
    return jax.vmap(f)(od, ssa, g, od_cloud, ssa_cloud, g_cloud, albedo_direct, albedo_diffuse,
                       incoming_sw, cos_sza, cloud_fraction, fractional_std, overlap_param)
