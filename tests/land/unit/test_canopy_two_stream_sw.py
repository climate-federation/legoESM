"""Two-stream canopy shortwave (CanopyConfig.sw_scheme="two_stream").

Oracle: an independent float64 NumPy transcription of DifferBESS v10
``CanopyShortwaveRadiationTwoStream.two_stream_fractions`` / ``_band`` /
``canopy_geometry`` (checkout bd38955c7b working tree, file md5
89755866fd49ead196440691f2818cac), with its own literal constants (canaried
against the module).  RULE 5 for the smoothed sunlit clip: hard limit
(clip_width=0) matches the oracle per case; the default width's per-case gap is
recorded; the gradient through the default is finite and non-zero.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.land.canopy import radiative_transfer as rt
from legoesm.land.canopy.config import CanopyConfig, PFT_LEAF_OPTICS

jax.config.update("jax_enable_x64", True)

# ---- oracle literals (DifferBESS canopy_geometry.py / TwoStream module) ----
KD_CHI = np.array([-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
KD_VAL = np.array([0.6500, 0.6640, 0.6815, 0.7020, 0.7240, 0.7485, 0.7735, 0.7990, 0.8245,
                   0.8475, 0.8675, 0.8805, 0.8775])
SOIL = (0.15, 0.30)
UV_SCAT = 0.05
APAR = 4.56


def o_geom(chi):
    phi1 = 0.5 - 0.633 * chi - 0.33 * chi ** 2
    return phi1, 0.877 * (1.0 - 2.0 * phi1), float(np.interp(chi, KD_CHI, KD_VAL)), ((1.0 + chi) / 2.0) ** 2


def o_optics(pft):
    if pft == "SAV":
        t, g, w = PFT_LEAF_OPTICS_ROWS["DBF"], PFT_LEAF_OPTICS_ROWS["GRA"], 0.45
        tg, gg = o_geom(t[4]), o_geom(g[4])
        return [w * t[i] + (1 - w) * g[i] for i in range(4)] + [w * tg[i] + (1 - w) * gg[i] for i in range(4)]
    r = PFT_LEAF_OPTICS_ROWS[pft]
    return list(r[:4]) + list(o_geom(r[4]))


PFT_LEAF_OPTICS_ROWS = {  # CLM5 Table 2.3.1 as DifferBESS DEFAULT_LEAF_OPTICS
    "ENF": (0.07, 0.05, 0.35, 0.10, 0.01), "DBF": (0.10, 0.05, 0.45, 0.25, 0.25),
    "GRA": (0.11, 0.05, 0.35, 0.34, -0.30), "EBF": (0.10, 0.05, 0.45, 0.25, 0.10),
}


def o_fractions(SZA, L, rho, tau, alb_soil, phi1, phi2, kd, cos2, w):
    mu = math.cos(math.radians(min(SZA, 89.0)))
    omega = rho + tau
    mubar = 1.0 / kd
    ob = 0.5 * (rho + tau + (rho - tau) * cos2)
    b = 1.0 - omega + ob
    c = max(ob, 1e-6)
    h = math.sqrt(b * b - c * c) / mubar
    K = 50.0 if SZA > 89.0 else (phi1 + phi2 * mu) / mu
    sigma = (mubar * K) ** 2 + c * c - b * b
    if abs(sigma) < 1e-4:
        K *= 1.001
        sigma = (mubar * K) ** 2 + c * c - b * b
    G = phi1 + phi2 * mu
    a_s = 0.5 * G / (mu * phi2 + G) * (1 - (mu * phi1 / (mu * phi2 + G)) * math.log((mu * phi1 + mu * phi2 + G) / (mu * phi1)))
    beta0 = (1 + mubar * K) / (mubar * K) * a_s
    d = omega * mubar * K * beta0
    f = omega * mubar * K * (1 - beta0)
    vm, vp = (b + mubar * h) / c, (b - mubar * h) / c
    eL, eK = math.exp(-h * L), math.exp(-K * L)
    hk = h - K
    cross = L * eK * (1 - 0.5 * hk * L) if abs(hk) < 1e-3 else (eK - eL) / hk
    out = {}
    for kind in "bd":
        if kind == "b":
            p1 = (d * (mubar * K - b) - c * f) / sigma
            p2 = (-f * (b + mubar * K) - c * d) / sigma
            I0, Tb, c0 = 0.0, eK, K * (p2 - p1) + omega * K
        else:
            p1 = p2 = 0.0
            I0, Tb, c0 = 1.0, 0.0, 0.0
        a11, a12, r1 = vm, vp * eL, I0 - p2
        a21, a22 = eL * (1 - alb_soil * vm), 1 - alb_soil * vp
        r2 = alb_soil * (p2 * eK + Tb) - p1 * eK
        det = a11 * a22 - a12 * a21
        A = (r1 * a22 - a12 * r2) / det
        B = (a11 * r2 - a21 * r1) / det
        albedo = p1 + A + B * eL
        soil = (1 - alb_soil) * (p2 * eK + A * vm * eL + B * vp + Tb)
        veg = 1 - albedo - soil
        sd = c0 * (1 - eK * eK) / (2 * K) + h * A * (vm - 1) * (1 - eK * eL) / (K + h) - h * B * (vp - 1) * cross
        sun = min(max(((1 - omega) * (1 - eK) if kind == "b" else 0.0) + w * sd, 0.0), veg)
        out[kind] = (sun, veg - sun, soil, albedo)
    return out


def o_band(Idir, Idif, SZA, L, rho, tau, alb_soil, alb_modis, g, w):
    fr = o_fractions(SZA, L, rho, tau, alb_soil, *g, w)
    parts = [Idir * fr["b"][i] + Idif * fr["d"][i] for i in range(3)]
    tot = sum(parts)
    s = (1 - alb_modis) * (Idir + Idif) / tot if tot > 1e-9 else 1.0
    return [p * s for p in parts]


def o_sw(case, pft):
    PARd, PARf, NIRd, NIRf, UV, SZA, LAI, CI, AV, AN = case
    o = o_optics(pft)
    g = o[4:]
    L = max(LAI * CI, 1e-4)
    P = o_band(PARd, PARf, SZA, L, o[0], o[1], SOIL[0], AV, g, CI)
    N = o_band(NIRd, NIRf, SZA, L, o[2], o[3], SOIL[1], AN, g, CI)
    UVd = UV * PARd / (PARd + PARf + 1e-5)
    U = o_band(UVd, UV - UVd, SZA, L, UV_SCAT / 2, UV_SCAT / 2, SOIL[0], AV, g, CI)
    return dict(ASW_Sun=P[0] + N[0] + U[0], ASW_Sh=P[1] + N[1] + U[1], ASW_Soil=P[2] + N[2] + U[2],
                APAR_Sun=P[0] * APAR, APAR_Sh=P[1] * APAR)


CASES = [  # PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV, SZA, LAI, CI, ALB_VIS, ALB_NIR
    (300.0, 60.0, 320.0, 60.0, 15.0, 20.0, 4.0, 0.8, 0.05, 0.30),
    (150.0, 80.0, 160.0, 80.0, 10.0, 55.0, 1.5, 0.7, 0.06, 0.28),
    (20.0, 60.0, 22.0, 60.0, 3.0, 80.0, 6.0, 0.9, 0.04, 0.25),
    (250.0, 40.0, 260.0, 40.0, 12.0, 35.0, 0.3, 1.0, 0.10, 0.30),
    (0.0, 5.0, 0.0, 5.0, 0.2, 88.5, 2.5, 0.6, 0.05, 0.27),
    (200.0, 50.0, 210.0, 50.0, 10.0, 45.0, 0.02, 0.5, 0.12, 0.30),
    (220.0, 40.0, 230.0, 40.0, 10.0, 30.0, 0.01, 1.0, 0.08, 0.30),  # sunlit share in the band
]
PFTS = ["ENF", "DBF", "SAV", "GRA"]


def _optics(pft, n=1):
    v = rt.two_stream_leaf_optics(pft)
    return rt.TwoStreamLeafOptics(*(jnp.full(n, v[f"leaf_{k}"]) for k in (
        "rho_vis", "tau_vis", "rho_nir", "tau_nir", "phi1", "phi2", "kd", "cos2_theta")))


def _run(case, pft, width):
    a = [jnp.array([x]) for x in case]
    return rt.canopy_shortwave_two_stream(*a, jnp.array([60.0]), jnp.array([40.0]), jnp.array([0.4]),
                                          _optics(pft), width)


def test_constants_canary():
    assert rt._TS_KD_CHI == tuple(KD_CHI) and rt._TS_KD_VAL == tuple(KD_VAL)
    assert (rt._RHO_PAR_SOIL, rt._RHO_NIR_SOIL, rt._TS_UV_LEAF_SCATTERING, rt._APAR_CONVERSION) == (*SOIL, UV_SCAT, APAR)
    for k, row in PFT_LEAF_OPTICS_ROWS.items():
        assert PFT_LEAF_OPTICS[k] == row


@pytest.mark.parametrize("pft", PFTS)
@pytest.mark.parametrize("ci", range(len(CASES)))
def test_hard_limit_matches_oracle(pft, ci):
    out, ref = _run(CASES[ci], pft, 0.0), o_sw(CASES[ci], pft)
    for k, v in ref.items():
        np.testing.assert_allclose(float(getattr(out, k)[0]), v, rtol=1e-9, atol=1e-12, err_msg=k)


# Measured per-case (default - oracle) of ASW_Sun at clip_width=0.01 [W m-2], job
# 11246591 at the ratio-space smooth clip.  Rounding level (<4e-13) wherever the
# sunlit share of canopy absorption is outside the band (cases 0-5: share <= 0.988);
# case 6 (LAI 0.01, CI 1, share 0.997-0.9995) -0.0019..-0.0026 W m-2.
# Re-measure and update with the reason if the smooth form or its width changes.
DEFAULT_GAP = {
    ('ENF', 0): 1.1368683772161603e-13,
    ('ENF', 1): 0.0,
    ('ENF', 2): 0.0,
    ('ENF', 3): 9.947598300641403e-14,
    ('ENF', 4): 0.0,
    ('ENF', 5): 1.4210854715202004e-14,
    ('ENF', 6): -0.0025609565686188773,
    ('DBF', 0): 0.0,
    ('DBF', 1): 0.0,
    ('DBF', 2): -7.105427357601002e-15,
    ('DBF', 3): 0.0,
    ('DBF', 4): 0.0,
    ('DBF', 5): 1.021405182655144e-14,
    ('DBF', 6): -0.002297587390699185,
    ('SAV', 0): 0.0,
    ('SAV', 1): -2.842170943040401e-14,
    ('SAV', 2): 7.105427357601002e-15,
    ('SAV', 3): 1.4210854715202004e-14,
    ('SAV', 4): 0.0,
    ('SAV', 5): 2.6201263381153694e-14,
    ('SAV', 6): -0.0020631818985004102,
    ('GRA', 0): 3.410605131648481e-13,
    ('GRA', 1): -2.842170943040401e-14,
    ('GRA', 2): 7.105427357601002e-15,
    ('GRA', 3): 1.4210854715202004e-14,
    ('GRA', 4): -2.7755575615628914e-17,
    ('GRA', 5): -1.9984014443252818e-14,
    ('GRA', 6): -0.001861332159471818,
}


@pytest.mark.parametrize("pft", PFTS)
@pytest.mark.parametrize("ci", range(len(CASES)))
def test_default_width_gap_recorded(pft, ci):
    out, ref = _run(CASES[ci], pft, 0.01), o_sw(CASES[ci], pft)
    gap = float(out.ASW_Sun[0]) - ref["ASW_Sun"]
    np.testing.assert_allclose(gap, DEFAULT_GAP[(pft, ci)], rtol=1e-6, atol=1e-12)


# A forcing case whose sunlit/canopy ratio sits INSIDE the smoothing band at the
# default width (measured; see GRAD_CASE_NOTE), so the gradient test exercises the
# smooth switch rather than an exact branch.
GRAD_PFT, GRAD_CI = "DBF", 6


def _sun(width, pft, ci):
    def f(lai, sza):
        a = [jnp.array([x]) for x in CASES[ci]]
        a[6], a[5] = lai, sza
        o = rt.canopy_shortwave_two_stream(*a, jnp.array([60.0]), jnp.array([40.0]), jnp.array([0.4]),
                                           _optics(pft), width)
        return jnp.sum(o.ASW_Sun) + 0.5 * jnp.sum(o.ASW_Sh)
    return f


def test_gradient_through_default():
    f = _sun(0.01, GRAD_PFT, GRAD_CI)
    lai, sza = jnp.array([CASES[GRAD_CI][6]]), jnp.array([CASES[GRAD_CI][5]])
    check_grads(f, (lai, sza), order=1, modes=["rev"], atol=1e-6, rtol=1e-5)
    g = jax.grad(f)(lai, sza)
    assert np.isfinite(float(g[0])) and float(g[0]) != 0.0
    # control: the case is inside the band, so the hard clip has a different slope
    g_hard = jax.grad(_sun(0.0, GRAD_PFT, GRAD_CI))(lai, sza)
    assert abs(float(g[0]) - float(g_hard[0])) > 1e-6 * abs(float(g_hard[0])) + 1e-9


@pytest.mark.parametrize("width", [0.0, 0.01])
def test_sunlit_bounded_in_sparse_canopies(width):
    """0 <= sunlit <= canopy absorbed for every L down to the floor, both widths;
    the control shows the grid reaches the regime where a flux-space smooth clip
    (the reviewed defect) goes negative."""
    from legoesm.core.smooth import smooth_max_c1, smooth_min_c1
    worst_flux_space = 0.0
    for pft in PFTS:
        opt = _optics(pft)
        for sza in (10.0, 45.0, 75.0, 88.0):
            for L in (1e-4, 1e-3, 5e-3, 0.02, 0.1):
                for rho, tau in ((opt.rho_vis, opt.tau_vis), (opt.rho_nir, opt.tau_nir)):
                    fr = rt._ts_fractions(jnp.array([sza]), jnp.array([L]), rho, tau, 0.2, opt,
                                          jnp.array([0.7]), width)
                    for kind in "bd":
                        sun, sh, _, _ = (float(x[0]) for x in fr[kind])
                        assert sun >= 0.0 and sh >= -1e-15
                        veg = sun + sh
                        old = float(smooth_min_c1(smooth_max_c1(jnp.array(0.3 * veg), 0.0, 0.01), veg, 0.01))
                        worst_flux_space = min(worst_flux_space, old)
    assert worst_flux_space < 0.0


@pytest.mark.parametrize("width", [0.0, 0.01])
def test_energy_closure(width):
    """Pre-rescale: albedo + sunlit + shaded + soil = 1 per incidence, no negative
    part; post-rescale: absorbed SW = (1 - ALB) x incident, summed over bands."""
    for pft in PFTS:
        opt = _optics(pft)
        for sza in (5.0, 30.0, 60.0, 80.0, 88.0):
            for L in (0.05, 0.5, 2.0, 6.0):
                for rho, tau in ((opt.rho_vis, opt.tau_vis), (opt.rho_nir, opt.tau_nir)):
                    fr = rt._ts_fractions(jnp.array([sza]), jnp.array([L]), rho, tau, 0.2, opt,
                                          jnp.array([0.7]), width)
                    for kind in "bd":
                        sun, sh, soil, alb = (float(x[0]) for x in fr[kind])
                        assert min(sun, sh, soil) >= -1e-12
                        assert abs(alb + sun + sh + soil - 1.0) < 1e-12
    for case in CASES[:4] + [CASES[0][:6] + (0.0,) + CASES[0][7:]]:
        out = _run(case, "DBF", width)
        PARd, PARf, NIRd, NIRf, UV, *_, AV, AN = case
        incoming_abs = (1 - AV) * (PARd + PARf + UV) + (1 - AN) * (NIRd + NIRf)
        tot = float(out.ASW_Sun[0] + out.ASW_Sh[0] + out.ASW_Soil[0])
        assert abs(tot - incoming_abs) < 1e-9 * incoming_abs


def test_mutant_fails_oracle(monkeypatch):
    """Non-vacuity: a 2% error in the beam upscatter must break the oracle match."""
    orig = rt._ts_single_scattering_albedo
    monkeypatch.setattr(rt, "_ts_single_scattering_albedo", lambda mu, p1, p2: 1.02 * orig(mu, p1, p2))
    jax.clear_caches()
    out, ref = _run(CASES[0], "DBF", 0.0), o_sw(CASES[0], "DBF")
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(float(out.ASW_Sun[0]), ref["ASW_Sun"], rtol=1e-9)
    jax.clear_caches()


def test_weight_mutant_fails_oracle():
    """Non-vacuity of the CI weighting: the v9 weight 1 must not match the v10 oracle."""
    case = CASES[1]
    ref = o_sw(case, "DBF")
    o = o_optics("DBF")
    PARd, PARf, *_ = case
    alt = o_band(PARd, PARf, case[5], case[6] * case[7], o[0], o[1], SOIL[0], case[8], o[4:], 1.0)[0]
    assert abs(alt * APAR - ref["APAR_Sun"]) > 1e-3 * ref["APAR_Sun"]


def test_leaf_optics_table():
    v = rt.two_stream_leaf_optics("SAV")
    np.testing.assert_allclose(v["leaf_rho_nir"], 0.45 * 0.45 + 0.55 * 0.35)
    assert rt._ts_geometry(0.0)[:2] == (0.5, 0.0)
    with pytest.raises(ValueError, match="no two-stream leaf optics"):
        rt.two_stream_leaf_optics("XXX")


def test_config_default_and_validation():
    assert CanopyConfig().sw_scheme == "ryu"
    with pytest.raises(ValueError, match="unknown sw_scheme"):
        CanopyConfig(sw_scheme="bogus").validate()
    with pytest.raises(ValueError, match="two_stream_clip_width"):
        CanopyConfig(two_stream_clip_width=-1.0).validate()
    CanopyConfig(sw_scheme="two_stream").validate()


def test_two_stream_without_leaf_optics_raises():
    from legoesm.core.coupling_fields import AtmToSurface
    from legoesm.land.config import MultiLayerLandConfig
    from legoesm.land.surface_scheme import TwoLeafCanopyConfig
    from legoesm.land.surface_scheme.two_leaf_canopy import compute_two_leaf_canopy_fluxes
    n = 1
    forcing = AtmToSurface(
        T_lowest=jnp.full(n, 295.0), q_lowest=jnp.full(n, 0.01), u_lowest=jnp.full(n, 2.0),
        v_lowest=jnp.zeros(n), p_lowest=jnp.full(n, 98000.0), p_surface=jnp.full(n, 101325.0),
        rho_lowest=jnp.full(n, 1.15), sw_down=jnp.full(n, 600.0), lw_down=jnp.full(n, 350.0),
        cos_zenith=jnp.full(n, 0.8), precip_total=jnp.zeros(n), precip_snow=jnp.zeros(n),
        co2_ppmv=jnp.full(n, 410.0), has_radiation=True, has_precipitation=False)
    T = jnp.full(n, 293.0)
    with pytest.raises(ValueError, match="needs per-column leaf optics"):
        compute_two_leaf_canopy_fluxes(
            T_soil_top=T, forcing=forcing,
            canopy_config=TwoLeafCanopyConfig(max_iters=30, sw_scheme="two_stream"),
            land_config=MultiLayerLandConfig(), canopy_params=None,
            w_frac_rz=jnp.ones(n), wind_speed=jnp.full(n, 2.0),
            wind_dir_x=jnp.ones(n), wind_dir_y=jnp.zeros(n),
            soil_thermal_fn=lambda G, dt_: T, dt=1800.0)
