"""Faithfulness pins for the RRTMGP cloud-optics combination.

Target: ``compute_optical_properties`` in
``legoesm.atmosphere.physics.radiation.rrtmgp.optics.cloud_optics`` — the
lookup-table cloud optics used by the RRTMGP radiation scheme (swirl_jatmos /
rte-rrtmgp port).  For each phase it linearly interpolates the extinction /
single-scattering-albedo / asymmetry tables by effective particle size, scales
by the (masked) cloud path, and combines the two phases with the standard
optical-property mixing rule:

    tau   = tau_liq + tau_ice                              (extinction adds)
    ssa   = (ssa_liq*tau_liq + ssa_ice*tau_ice) / tau      (tau-weighted)
    g     = (g_liq*ssa_liq*tau_liq + g_ice*ssa_ice*tau_ice)
            / (ssa_liq*tau_liq + ssa_ice*tau_ice)          (ssa-weighted)

Most-trustful source
--------------------
The rte-rrtmgp / ecRad cloud-optics kernel (Pincus et al. 2019; the swirl_jatmos
``cloud_optics`` this file is ported from).  The mixing rule above is the
standard optical-property combination (scattering optical depth is ``ssa*tau``,
so total ssa is tau-weighted; the scattering first moment is ``g*ssa*tau``, so
total ``g`` is ssa-weighted).  The per-phase step is an even-grid linear
interpolation of the extinction (m^2/g), single-scattering-albedo, and asymmetry
LUTs (the latter two dimensionless).

Certification (test-only; the module is a closed-form LUT combination):
1. EXACT LUT linear interpolation for ALL of {ext, ssa, asy} x {liquid, ice} at a
   NON-midpoint, NON-affine table (so reversed or wrong interpolation weights and
   broken per-phase/per-coefficient interpolation fail), vs hand-computed values
   and an independent numpy reimplementation typed from create_linear_interpolant.
2. Unit conventions: effective radius m->micron (1e6), ICE uses DIAMETER = 2*r,
   cloud path kg/m^2 -> g/m^2 (1e3) — each pinned by a discriminating placement;
   out-of-range sizes clamp to the table bounds.
3. EXACT phase combination (tau adds, ssa tau-weighted, g ssa-weighted) with
   asymmetric per-phase values so a wrong weighting (arithmetic mean, tau- vs
   ssa-weighting swap) fails.
4. TRUTH-TIER: single-phase reduction, ssa/g in [0,1], the >= 1e-6 g/m^2 cloud
   mask pinned AT its boundary, mixed masking, and the safe_divide STRICT (> eps)
   fill (empty/near-empty cloud -> ssa=g=0, finite grad).
5. Band-index (liquid AND ice) and ice-roughness table selection; differentiability.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics.radiation.rrtmgp.optics import cloud_optics
from legoesm.atmosphere.physics.radiation.rrtmgp.optics.lookup_cloud_optics import (
    IceRoughness,
    LookupCloudOptics,
)

jax.config.update("jax_enable_x64", True)

_M_TO_UM = 1e6
_KG_TO_G = 1e3
_EPS = 1e-6            # cloud_optics._EPSILON: mask threshold (>=) and safe_divide eps (>)
_MEDIUM = 1           # IceRoughness.MEDIUM index into the ice-table roughness axis

# Synthetic lookup grids (even spacing -> known linear-interp weights):
#   liquid radius   [micron]: linspace(2, 8, 4)   = [2, 4, 6, 8],   delta 2
#   ice    diameter [micron]: linspace(10, 40, 4) = [10,20,30,40],  delta 10
# 2 bands; ice roughness axis length 3 (SMOOTH/MEDIUM/ROUGH).
_N_LIQ = 4
_N_ICE = 4
_R_LIQ_LO, _R_LIQ_HI = 2.0, 8.0
_D_ICE_LO, _D_ICE_HI = 10.0, 40.0

# r_ice giving an ON-node ice diameter of 20 micron (used where ice is a passive
# constant); r_liq giving an on-node 4 micron.
_R_ICE_20UM = 20.0 / (2 * _M_TO_UM)
_R_LIQ_4UM = 4.0 / _M_TO_UM


def _make_lookup(ext_liq, ssa_liq, asy_liq, ext_ice, ssa_ice, asy_ice):
    """Frozen dataclass with the given (nbnd, nsize) / (nrgh, nbnd, nsize) tables."""
    f = jnp.asarray
    return LookupCloudOptics(
        n_size_liq=_N_LIQ, n_size_ice=_N_ICE,
        radius_liq_lower=_R_LIQ_LO, radius_liq_upper=_R_LIQ_HI,
        diameter_ice_lower=_D_ICE_LO, diameter_ice_upper=_D_ICE_HI,
        ext_liq=f(ext_liq), ssa_liq=f(ssa_liq), asy_liq=f(asy_liq),
        ext_ice=f(ext_ice), ssa_ice=f(ssa_ice), asy_ice=f(asy_ice),
        ice_roughness=IceRoughness.MEDIUM,
    )


def _const_lookup(el, sl, al, ei, si, ai):
    """Lookup whose tables are CONSTANT across size (interp -> the constant).

    Band 0 carries the given per-phase (ext, ssa, asy) EXACTLY; band 1 is
    distinct (band0 + 100 liquid, +1100 ice MEDIUM) so a band-index pin can
    discriminate.  Ice roughness MEDIUM (row 1) at band 0 carries the ice value
    EXACTLY; the other roughness rows / band 1 are distinct so a wrong
    roughness/band read fails.
    """
    def liq_tbl(v0):                                                     # (nbnd, nsize)
        return np.array([[v0] * _N_LIQ, [v0 + 100.0] * _N_LIQ])

    def ice_tbl(v0):                                                     # (nrgh, nbnd, nsize)
        t = np.zeros((3, 2, _N_ICE))
        t[0, 0, :] = v0 - 50.0                                           # SMOOTH / band0
        t[1, 0, :] = v0                                                  # MEDIUM / band0 = value
        t[2, 0, :] = v0 + 50.0                                           # ROUGH  / band0
        t[0, 1, :] = v0 + 1000.0                                         # band1 rows distinct
        t[1, 1, :] = v0 + 1100.0
        t[2, 1, :] = v0 + 1150.0
        return t

    return _make_lookup(liq_tbl(el), liq_tbl(sl), liq_tbl(al),
                        ice_tbl(ei), ice_tbl(si), ice_tbl(ai))


def _liq_only_lookup(ext_row, ssa_row, asy_row):
    """Liquid band-0 tables = the given size-varying rows; ice all zero."""
    z2 = np.zeros((2, _N_LIQ))
    el = np.array([ext_row, [0.0] * _N_LIQ])
    sl = np.array([ssa_row, [0.0] * _N_LIQ])
    al = np.array([asy_row, [0.0] * _N_LIQ])
    z3 = np.zeros((3, 2, _N_ICE))
    return _make_lookup(el, sl, al, z3, z3, z3), z2


def _ice_only_lookup(ext_row, ssa_row, asy_row):
    """Ice MEDIUM/band-0 tables = the given size-varying rows; liquid all zero."""
    def ice(row):
        t = np.zeros((3, 2, _N_ICE))
        t[_MEDIUM, 0, :] = row
        return t
    z2 = np.zeros((2, _N_LIQ))
    return _make_lookup(z2, z2, z2, ice(ext_row), ice(ssa_row), ice(asy_row))


def _interp1d(table_row, s, lo, hi, n):
    """Independent even-grid linear interpolation (matches create_linear_interpolant)."""
    fref = np.linspace(lo, hi, n)
    delta = fref[1] - fref[0]
    s = float(np.clip(s, lo, hi))
    j = int(np.clip(np.floor((s - lo) / delta), 0, n - 1))
    j2 = min(j + 1, n - 1)
    lower = fref[0] + delta * j
    w2 = abs((s - lower) / delta)
    w1 = 1.0 - w2
    return w1 * table_row[j] + w2 * table_row[j2]


def _oracle(lk, path_liq, path_ice, r_liq, r_ice, ibnd, rough=_MEDIUM):
    """Independent reimplementation of compute_optical_properties."""
    s_liq = np.clip(_M_TO_UM * r_liq, _R_LIQ_LO, _R_LIQ_HI)
    s_ice = np.clip(_M_TO_UM * 2.0 * r_ice, _D_ICE_LO, _D_ICE_HI)   # ICE = diameter = 2r
    el = np.asarray(lk.ext_liq)[ibnd]
    sl = np.asarray(lk.ssa_liq)[ibnd]
    al = np.asarray(lk.asy_liq)[ibnd]
    ei = np.asarray(lk.ext_ice)[rough, ibnd]
    si = np.asarray(lk.ssa_ice)[rough, ibnd]
    ai = np.asarray(lk.asy_ice)[rough, ibnd]
    ext_l = _interp1d(el, s_liq, _R_LIQ_LO, _R_LIQ_HI, _N_LIQ)
    ssa_l = _interp1d(sl, s_liq, _R_LIQ_LO, _R_LIQ_HI, _N_LIQ)
    asy_l = _interp1d(al, s_liq, _R_LIQ_LO, _R_LIQ_HI, _N_LIQ)
    ext_i = _interp1d(ei, s_ice, _D_ICE_LO, _D_ICE_HI, _N_ICE)
    ssa_i = _interp1d(si, s_ice, _D_ICE_LO, _D_ICE_HI, _N_ICE)
    asy_i = _interp1d(ai, s_ice, _D_ICE_LO, _D_ICE_HI, _N_ICE)
    pl, pi = path_liq * _KG_TO_G, path_ice * _KG_TO_G
    ml = 1.0 if pl >= _EPS else 0.0                                 # mask: >= (inclusive)
    mi = 1.0 if pi >= _EPS else 0.0
    tau_l, tau_i = ext_l * pl * ml, ext_i * pi * mi
    ts_l, ts_i = ssa_l * tau_l, ssa_i * tau_i
    tsg_l, tsg_i = asy_l * ts_l, asy_i * ts_i
    tau, ts, tsg = tau_l + tau_i, ts_l + ts_i, tsg_l + tsg_i
    ssa = ts / tau if abs(tau) > _EPS else 0.0                      # safe_divide: STRICT >
    g = tsg / ts if abs(ts) > _EPS else 0.0
    return tau, ssa, g


def _call(lk, path_liq, path_ice, r_liq, r_ice, ibnd, out):
    a = jnp.atleast_1d
    res = cloud_optics.compute_optical_properties(
        lk, a(jnp.float64(path_liq)), a(jnp.float64(path_ice)),
        a(jnp.float64(r_liq)), a(jnp.float64(r_ice)),
        jnp.int32(ibnd),          # scalar band index (production loops over bands)
    )
    return float(res[out][0])


# ---------------------------------------------------------------------------
# 1. EXACT LUT linear interpolation — ALL tables, NON-affine, OFF-midpoint.
# ---------------------------------------------------------------------------
def test_lut_interp_liquid_all_tables_off_midpoint():
    # NON-affine tables over radii [2,4,6,8] micron; size 4.5 micron -> between
    # nodes 1 (=4) and 2 (=6) with weights w1=0.75, w2=0.25 (NOT 0.5/0.5, so
    # reversed weights are detectable; NOT affine, so a broken interpolation is).
    ext = [1.0, 3.0, 9.0, 27.0]
    ssa = [0.20, 0.40, 0.80, 0.90]
    asy = [0.10, 0.30, 0.70, 0.85]
    lk, _ = _liq_only_lookup(ext, ssa, asy)
    r = 4.5 / _M_TO_UM
    od = _call(lk, 1e-3, 0.0, r, _R_ICE_20UM, 0, "optical_depth")
    ssa_out = _call(lk, 1e-3, 0.0, r, _R_ICE_20UM, 0, "ssa")
    g_out = _call(lk, 1e-3, 0.0, r, _R_ICE_20UM, 0, "asymmetry_factor")
    # w1=0.75, w2=0.25 on nodes 1,2 ; path 1 g/m^2 -> tau = ext_interp.
    np.testing.assert_allclose(od, 0.75 * 3.0 + 0.25 * 9.0, rtol=1e-12)      # 4.5
    np.testing.assert_allclose(ssa_out, 0.75 * 0.40 + 0.25 * 0.80, rtol=1e-12)  # 0.5
    np.testing.assert_allclose(g_out, 0.75 * 0.30 + 0.25 * 0.70, rtol=1e-12)    # 0.4
    o = _oracle(lk, 1e-3, 0.0, r, _R_ICE_20UM, 0)
    np.testing.assert_allclose([od, ssa_out, g_out], list(o), rtol=1e-12)


def test_lut_interp_ice_all_tables_off_midpoint():
    # ICE diameter grid [10,20,30,40]; r_ice = 11.25 micron -> diameter 22.5 ->
    # between nodes 1 (=20) and 2 (=30), w1=0.75, w2=0.25.  Non-affine tables.
    ext = [2.0, 5.0, 11.0, 23.0]
    ssa = [0.15, 0.35, 0.60, 0.75]
    asy = [0.20, 0.45, 0.65, 0.80]
    lk = _ice_only_lookup(ext, ssa, asy)
    r_ice = 11.25 / _M_TO_UM
    od = _call(lk, 0.0, 1e-3, _R_LIQ_4UM, r_ice, 0, "optical_depth")
    ssa_out = _call(lk, 0.0, 1e-3, _R_LIQ_4UM, r_ice, 0, "ssa")
    g_out = _call(lk, 0.0, 1e-3, _R_LIQ_4UM, r_ice, 0, "asymmetry_factor")
    np.testing.assert_allclose(od, 0.75 * 5.0 + 0.25 * 11.0, rtol=1e-12)     # 6.5
    np.testing.assert_allclose(ssa_out, 0.75 * 0.35 + 0.25 * 0.60, rtol=1e-12)  # 0.4125
    np.testing.assert_allclose(g_out, 0.75 * 0.45 + 0.25 * 0.65, rtol=1e-12)    # 0.5
    o = _oracle(lk, 0.0, 1e-3, _R_LIQ_4UM, r_ice, 0)
    np.testing.assert_allclose([od, ssa_out, g_out], list(o), rtol=1e-12)


def test_lut_interpolation_on_node_is_table_value():
    ext = [10.0, 20.0, 30.0, 40.0]
    lk, _ = _liq_only_lookup(ext, [1.0] * _N_LIQ, [0.0] * _N_LIQ)
    # 6 micron lands exactly on node index 2 -> ext = 30.
    od = _call(lk, 1e-3, 0.0, 6.0 / _M_TO_UM, _R_ICE_20UM, 0, "optical_depth")
    np.testing.assert_allclose(od, 30.0, rtol=1e-12)


def test_size_below_lower_bound_clamps_not_extrapolates():
    # NON-affine tables; a size BELOW the lower bound must CLAMP to node 0, not
    # extrapolate.  floor_idx already pins the index at 0, but WITHOUT the value
    # clip the weight w2 = |s - node0|/delta > 0 would give w1*t0 + w2*t1 (= 2.5 /
    # 4.4 here), so pinning the exact node-0 value is a discriminating clip canary.
    # (The UPPER bound is NOT discriminating: floor_idx caps the index at the last
    # node, collapsing both interpolation endpoints, so the last table value is
    # returned with or without the clip.)
    lk_l, _ = _liq_only_lookup([1.0, 3.0, 9.0, 27.0], [1.0] * _N_LIQ, [0.0] * _N_LIQ)
    od_l = _call(lk_l, 1e-3, 0.0, 0.5 / _M_TO_UM, _R_ICE_20UM, 0, "optical_depth")
    np.testing.assert_allclose(od_l, 1.0, rtol=1e-12)          # liquid node 0, not 2.5

    lk_i = _ice_only_lookup([2.0, 5.0, 11.0, 23.0], [1.0] * _N_ICE, [0.0] * _N_ICE)
    # r_ice = 1 micron -> diameter 2 micron < 10 lower bound -> clamp to ice node 0.
    od_i = _call(lk_i, 0.0, 1e-3, _R_LIQ_4UM, 1.0 / _M_TO_UM, 0, "optical_depth")
    np.testing.assert_allclose(od_i, 2.0, rtol=1e-12)          # ice node 0, not 4.4


# ---------------------------------------------------------------------------
# 2. Unit conventions (m->micron, ice DIAMETER=2r, kg->g).
# ---------------------------------------------------------------------------
def test_micron_conversion_selects_right_liquid_node():
    # A radius of 4 micron must select node index 1 (=20); a missing 1e6 factor
    # would clamp to the lower bound and read node 0 (=10).
    ext = [10.0, 20.0, 30.0, 40.0]
    lk, _ = _liq_only_lookup(ext, [1.0] * _N_LIQ, [0.0] * _N_LIQ)
    od = _call(lk, 1e-3, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "optical_depth")
    np.testing.assert_allclose(od, 20.0, rtol=1e-12)


def test_ice_uses_diameter_twice_radius():
    # r_ice = 10 micron -> diameter 20 micron -> node index 1.  Using radius (not
    # diameter) would read node 0.
    ext = [100.0, 200.0, 300.0, 400.0]
    lk = _ice_only_lookup(ext, [1.0] * _N_ICE, [0.0] * _N_ICE)
    r_ice = 10.0 / _M_TO_UM                                        # -> diameter 20 micron
    od = _call(lk, 0.0, 1e-3, _R_LIQ_4UM, r_ice, 0, "optical_depth")
    np.testing.assert_allclose(od, 200.0, rtol=1e-12)              # node index 1


def test_cloud_path_kg_to_g_scaling():
    # tau linear in path*1e3.  ext=10, path=2e-3 kg/m^2 -> 2 g/m^2 -> tau=20.
    lk = _const_lookup(10.0, 1.0, 0.0, 0.0, 1.0, 0.0)
    od = _call(lk, 2e-3, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "optical_depth")
    np.testing.assert_allclose(od, 20.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. EXACT phase combination (asymmetric values -> weighting is discriminating).
# ---------------------------------------------------------------------------
def test_phase_combination_exact():
    # liquid: ext=4, ssa=0.9, g=0.85 ; ice: ext=6, ssa=0.5, g=0.7
    # path_liq=path_ice=1e-3 (1 g/m^2) -> tau_l=4, tau_i=6.
    lk = _const_lookup(4.0, 0.9, 0.85, 6.0, 0.5, 0.7)
    od = _call(lk, 1e-3, 1e-3, _R_LIQ_4UM, _R_ICE_20UM, 0, "optical_depth")
    ssa = _call(lk, 1e-3, 1e-3, _R_LIQ_4UM, _R_ICE_20UM, 0, "ssa")
    g = _call(lk, 1e-3, 1e-3, _R_LIQ_4UM, _R_ICE_20UM, 0, "asymmetry_factor")
    tau_l, tau_i = 4.0, 6.0
    exp_ssa = (0.9 * tau_l + 0.5 * tau_i) / (tau_l + tau_i)            # tau-weighted
    ts_l, ts_i = 0.9 * tau_l, 0.5 * tau_i
    exp_g = (0.85 * ts_l + 0.7 * ts_i) / (ts_l + ts_i)                 # ssa-weighted
    np.testing.assert_allclose(od, 10.0, rtol=1e-12)
    np.testing.assert_allclose(ssa, exp_ssa, rtol=1e-12)
    np.testing.assert_allclose(g, exp_g, rtol=1e-12)
    np.testing.assert_allclose([od, ssa, g],
                               list(_oracle(lk, 1e-3, 1e-3, _R_LIQ_4UM, _R_ICE_20UM, 0)),
                               rtol=1e-12)


def test_ssa_is_tau_weighted_not_arithmetic_mean():
    lk = _const_lookup(2.0, 0.9, 0.8, 8.0, 0.4, 0.6)   # tau_l=2, tau_i=8
    ssa = _call(lk, 1e-3, 1e-3, _R_LIQ_4UM, _R_ICE_20UM, 0, "ssa")
    tau_weighted = (0.9 * 2.0 + 0.4 * 8.0) / (2.0 + 8.0)              # 0.5
    arithmetic = 0.5 * (0.9 + 0.4)                                    # 0.65
    np.testing.assert_allclose(ssa, tau_weighted, rtol=1e-12)
    assert abs(ssa - arithmetic) > 0.1                               # NOT the mean


def test_g_is_ssa_weighted_not_tau_weighted():
    lk = _const_lookup(5.0, 0.9, 0.85, 5.0, 0.3, 0.55)   # tau_l=tau_i=5
    g = _call(lk, 1e-3, 1e-3, _R_LIQ_4UM, _R_ICE_20UM, 0, "asymmetry_factor")
    ts_l, ts_i = 0.9 * 5.0, 0.3 * 5.0
    ssa_weighted = (0.85 * ts_l + 0.55 * ts_i) / (ts_l + ts_i)
    tau_weighted = 0.5 * (0.85 + 0.55)                               # equal tau -> plain mean
    np.testing.assert_allclose(g, ssa_weighted, rtol=1e-12)
    assert abs(g - tau_weighted) > 1e-3                              # NOT tau-weighted


def test_single_phase_reduction_liquid_only():
    lk = _const_lookup(4.0, 0.88, 0.82, 6.0, 0.5, 0.7)
    od = _call(lk, 1e-3, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "optical_depth")
    ssa = _call(lk, 1e-3, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "ssa")
    g = _call(lk, 1e-3, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "asymmetry_factor")
    np.testing.assert_allclose(od, 4.0, rtol=1e-12)
    np.testing.assert_allclose(ssa, 0.88, rtol=1e-12)
    np.testing.assert_allclose(g, 0.82, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. Masks + safe_divide (boundary-pinned).
# ---------------------------------------------------------------------------
def test_cloud_mask_boundary_at_1e_minus_6_gm2():
    # Mask is ``path*1e3 >= 1e-6`` (INCLUSIVE).  path = 1e-9 kg/m^2 -> exactly
    # 1e-6 g/m^2 -> NOT masked; path just below -> masked.
    lk = _const_lookup(10.0, 0.9, 0.8, 6.0, 0.5, 0.7)
    at = _call(lk, 1e-9, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "optical_depth")
    below = _call(lk, 0.99e-9, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "optical_depth")
    above = _call(lk, 1.01e-9, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "optical_depth")
    np.testing.assert_allclose(at, 10.0 * 1e-6, rtol=1e-9)           # included at boundary
    assert below == 0.0                                             # masked just below
    np.testing.assert_allclose(above, 10.0 * 1.01e-6, rtol=1e-9)


def test_mixed_mask_one_phase_active():
    # Liquid active, ice sub-threshold -> result is liquid-only.
    lk = _const_lookup(4.0, 0.88, 0.82, 6.0, 0.5, 0.7)
    od = _call(lk, 1e-3, 1e-10, _R_LIQ_4UM, _R_ICE_20UM, 0, "optical_depth")
    ssa = _call(lk, 1e-3, 1e-10, _R_LIQ_4UM, _R_ICE_20UM, 0, "ssa")
    np.testing.assert_allclose(od, 4.0, rtol=1e-12)                 # ice contributes nothing
    np.testing.assert_allclose(ssa, 0.88, rtol=1e-12)


def test_safe_divide_strict_threshold_on_tau_and_tau_ssa():
    # General fill behaviour: tau straddling eps (path 2 g/m^2, not masked) ->
    # fill 0 below, divide above.
    below = _const_lookup(0.4, 0.9, 0.8, 6.0, 0.5, 0.7)   # tau = 0.4 * 2e-6 = 0.8e-6
    above = _const_lookup(0.6, 0.9, 0.8, 6.0, 0.5, 0.7)   # tau = 0.6 * 2e-6 = 1.2e-6
    assert _call(below, 2e-9, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "ssa") == 0.0
    np.testing.assert_allclose(
        _call(above, 2e-9, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "ssa"), 0.9, rtol=1e-12)
    # STRICT ``> eps`` boundary (the >= vs > regression canary): ext=eps and
    # path 1e-3 kg/m^2 -> path_g = 1.0 exactly -> tau = eps*1.0 = eps EXACTLY, and
    # with ssa=1 the g-divide denominator tau_ssa = eps EXACTLY too.  ``|eps| > eps``
    # is False, so BOTH ssa and g take the fill 0; a ``>=`` regression would give
    # ssa=1 and g=0.5 and fail here.
    at = _const_lookup(_EPS, 1.0, 0.5, 6.0, 0.5, 0.7)
    assert _call(at, 1e-3, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "optical_depth") == _EPS
    assert _call(at, 1e-3, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "ssa") == 0.0
    assert _call(at, 1e-3, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "asymmetry_factor") == 0.0


def test_empty_cloud_safe_divide_fill_zero():
    lk = _const_lookup(10.0, 0.9, 0.8, 6.0, 0.5, 0.7)
    ssa = _call(lk, 0.0, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "ssa")
    g = _call(lk, 0.0, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "asymmetry_factor")
    assert ssa == 0.0 and g == 0.0


def test_ssa_and_g_within_unit_interval():
    lk = _const_lookup(4.0, 0.9, 0.85, 6.0, 0.5, 0.7)
    ssa = _call(lk, 1e-3, 1e-3, _R_LIQ_4UM, _R_ICE_20UM, 0, "ssa")
    g = _call(lk, 1e-3, 1e-3, _R_LIQ_4UM, _R_ICE_20UM, 0, "asymmetry_factor")
    assert 0.0 <= ssa <= 1.0 and 0.0 <= g <= 1.0


# ---------------------------------------------------------------------------
# 5. Band / ice-roughness selection; differentiability.
# ---------------------------------------------------------------------------
def test_band_index_selects_liquid_table_row():
    lk = _const_lookup(4.0, 0.9, 0.85, 6.0, 0.5, 0.7)
    od0 = _call(lk, 1e-3, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 0, "optical_depth")
    od1 = _call(lk, 1e-3, 0.0, _R_LIQ_4UM, _R_ICE_20UM, 1, "optical_depth")
    np.testing.assert_allclose(od0, 4.0, rtol=1e-12)
    np.testing.assert_allclose(od1, 104.0, rtol=1e-12)   # liquid band0 + 100


def test_band_index_selects_ice_table_row():
    # Ice-only: MEDIUM/band0 = ei, MEDIUM/band1 = ei + 1100 (see _const_lookup).
    lk = _const_lookup(4.0, 0.9, 0.85, 6.0, 0.5, 0.7)
    od0 = _call(lk, 0.0, 1e-3, _R_LIQ_4UM, _R_ICE_20UM, 0, "optical_depth")
    od1 = _call(lk, 0.0, 1e-3, _R_LIQ_4UM, _R_ICE_20UM, 1, "optical_depth")
    np.testing.assert_allclose(od0, 6.0, rtol=1e-12)
    np.testing.assert_allclose(od1, 1106.0, rtol=1e-12)  # ice MEDIUM band1 = 6 + 1100


def test_ice_roughness_medium_row_selected():
    # ice tables encode value = base + 10*roughness at band 0; MEDIUM row (1)
    # must be read (base + 10).
    ei = np.zeros((3, 2, _N_ICE))
    for r in range(3):
        ei[r, 0, :] = 300.0 + 10.0 * r
    si = np.zeros((3, 2, _N_ICE))
    si[:, 0, :] = 1.0
    lk = _make_lookup(np.zeros((2, _N_LIQ)), np.zeros((2, _N_LIQ)), np.zeros((2, _N_LIQ)),
                      ei, si, np.zeros((3, 2, _N_ICE)))
    od = _call(lk, 0.0, 1e-3, _R_LIQ_4UM, _R_ICE_20UM, 0, "optical_depth")
    np.testing.assert_allclose(od, 310.0, rtol=1e-12)    # MEDIUM row: 300 + 10*1


def test_differentiable_including_empty_cloud():
    lk = _const_lookup(4.0, 0.9, 0.85, 6.0, 0.5, 0.7)

    def loss(path_liq):
        a = jnp.atleast_1d
        res = cloud_optics.compute_optical_properties(
            lk, a(path_liq), a(jnp.float64(0.0)),
            a(jnp.float64(_R_LIQ_4UM)), a(jnp.float64(_R_ICE_20UM)),
            jnp.int32(0),         # scalar band index
        )
        return res["optical_depth"][0] + res["ssa"][0] + res["asymmetry_factor"][0]

    g_active = jax.grad(loss)(jnp.float64(1e-3))     # cloudy
    g_empty = jax.grad(loss)(jnp.float64(0.0))       # empty -> through safe_divide
    assert jnp.isfinite(g_active) and jnp.isfinite(g_empty)
