"""CLM5 big-leaf two-stream albedo (canopy over snow): closed form vs an
independent numerical solve of the two-stream ODEs.

The oracle integrates the Sellers (1985) / CLM5 tech-note two-stream equations

    -avmu dU/dx + b U - c1 D = d  exp(-K x)      (upward diffuse U)
     avmu dD/dx + b D - c1 U = f  exp(-K x)      (downward diffuse D)

over x in [0, VAI] by shooting (scipy ``solve_ivp``), with D(0) = 0 and
U(VAI) = a_g (D(VAI) + exp(-K VAI)) for a unit direct beam, and D(0) = 1,
no source, U(VAI) = a_g D(VAI) for unit diffuse; the albedo is U(0).  Only the
optical-parameter preamble (gdir, avmu, betad, betai, snow weighting) is shared
with the code under test, transcribed again here from CTSM SurfaceAlbedoMod.F90.
"""
import math

import jax
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from legoesm.land.canopy.radiative_transfer import clm5_two_stream_albedo

pytestmark = pytest.mark.skipif(not jax.config.jax_enable_x64,
                                reason="needs JAX_ENABLE_X64=1")


def _params(vai, fl, rl, tl, rs, ts, xl, cosz, fcs, oms):
    rho = max(rl * fl + rs * (1 - fl), 1e-6)
    tau = max(tl * fl + ts * (1 - fl), 1e-6)
    cz = max(cosz, 0.001)
    chil = min(max(xl, -0.4), 0.6)
    chil = 0.01 if abs(chil) <= 0.01 else chil
    phi1 = 0.5 - 0.633 * chil - 0.330 * chil ** 2
    phi2 = 0.877 * (1 - 2 * phi1)
    gdir = phi1 + phi2 * cz
    K = gdir / cz
    avmu = (1 - phi1 / phi2 * math.log((phi1 + phi2) / phi1)) / phi2
    t0 = max(gdir + phi2 * cz, 1e-6)
    t1 = phi1 * cz
    t2 = 1 - t1 / t0 * math.log((t1 + t0) / t1)
    oml = rho + tau
    asu = 0.5 * oml * gdir / t0 * t2
    bdl = (1 + avmu * K) / (oml * avmu * K) * asu
    bil = 0.5 * ((rho + tau) + (rho - tau) * ((1 + chil) / 2) ** 2) / oml
    om = (1 - fcs) * oml + fcs * oms
    bd = ((1 - fcs) * oml * bdl + fcs * oms * 0.5) / om
    bi = ((1 - fcs) * oml * bil + fcs * oms * 0.5) / om
    b = 1 - om + om * bi
    c1 = om * bi
    return K, avmu, b, c1, avmu * K * om * bd, avmu * K * om * (1 - bd)


def _oracle(vai, fl, rl, tl, rs, ts, xl, cosz, ag, fcs=0.0, oms=0.0):
    K, mu, b, c1, d, f = _params(vai, fl, rl, tl, rs, ts, xl, cosz, fcs, oms)

    def run(U0, D0, beam):
        def rhs(x, y):
            s = math.exp(-K * x) if beam else 0.0
            U, D = y
            return [(b * U - c1 * D - d * s) / mu, (c1 * U - b * D + f * s) / mu]
        r = solve_ivp(rhs, (0.0, vai), [U0, D0], method="DOP853",
                      rtol=1e-12, atol=1e-14)
        U, D = r.y[:, -1]
        return U - ag * (D + (math.exp(-K * vai) if beam else 0.0))

    out = []
    for beam, D0 in ((True, 0.0), (False, 1.0)):
        r0, r1 = run(0.0, D0, beam), run(1.0, D0, beam)
        out.append(-r0 / (r1 - r0))       # residual is linear in U(0)
    return out


# (vai, f_leaf, rhol, taul, rhos, taus, xl, cosz, ground, fcansno, omega_snow)
CASES = [
    (1.95, 0.77, 0.07, 0.05, 0.16, 0.001, 0.01, 0.15, 0.75, 0.0, 0.8),   # NET boreal vis, Dec
    (1.95, 0.77, 0.35, 0.10, 0.39, 0.001, 0.01, 0.30, 0.75, 0.0, 0.4),   # NET boreal nir
    (0.55, 0.0, 0.07, 0.05, 0.16, 0.001, 0.01, 0.20, 0.70, 0.0, 0.8),    # larch, stems only
    (0.32, 0.06, 0.35, 0.34, 0.53, 0.25, -0.30, 0.50, 0.60, 0.0, 0.4),   # grass stems nir
    (4.5, 0.9, 0.10, 0.05, 0.16, 0.001, 0.25, 0.70, 0.12, 0.0, 0.8),     # dense broadleaf, snow-free
    (1.95, 0.77, 0.07, 0.05, 0.16, 0.001, 0.01, 0.15, 0.75, 0.6, 0.8),   # snowy canopy
    (1.95, 0.77, 0.35, 0.10, 0.39, 0.001, 0.01, 0.15, 0.75, 1.0, 0.4),   # fully snowy canopy
]


@pytest.mark.parametrize("case", CASES)
def test_closed_form_matches_ode_solve(case):
    vai, fl, rl, tl, rs, ts, xl, cz, ag, fcs, oms = case
    albd, albi = clm5_two_stream_albedo(
        np.array([vai]), np.array([fl]), rl, tl, rs, ts, xl, np.array([cz]),
        np.array([ag]), fcansno=fcs, omega_snow=oms)
    od, oi = _oracle(vai, fl, rl, tl, rs, ts, xl, cz, ag, fcs, oms)
    np.testing.assert_allclose(float(albd[0]), od, rtol=1e-8)
    np.testing.assert_allclose(float(albi[0]), oi, rtol=1e-8)
    assert 0.0 < float(albd[0]) < 1.0 and 0.0 < float(albi[0]) < 1.0



# Reference values from EXECUTING the verbatim CTSM 5.1 SurfaceAlbedoMod.F90
# lines 823-824 and 830-831 (leaf/stem weighting), 1320-1338 and 1381-1518
# (TwoStream albedo), gfortran, one patch, snowveg on; harness
# scripts/validate/ctsm_twostream_oracle.F90 + ..._cases.txt (same rows).
# (elai, esai, xl, coszen, fcansno, rhol_v, rhol_n, taul_v, taul_n, rhos_v,
#  rhos_n, taus_v, taus_n, ground_v, ground_n)
#   -> (albd_vis, albi_vis, albd_nir, albi_nir)
FORTRAN = [
    ((1.51, 0.44, 0.01, 0.15, 0.0, 0.07, 0.35, 0.05, 0.10, 0.16, 0.39, 0.001, 0.001, 0.75, 0.75),
     (4.9571345752056475E-02, 5.6454576426496553E-02, 2.0106690286741402E-01, 1.9064478753149044E-01)),
    ((1.51, 0.44, 0.01, 0.30, 0.0, 0.07, 0.35, 0.05, 0.10, 0.16, 0.39, 0.001, 0.001, 0.75, 0.75),
     (4.5675123851123364E-02, 5.6454576426496553E-02, 1.7783639559774189E-01, 1.9064478753149044E-01)),
    ((0.0, 0.55, 0.01, 0.20, 0.0, 0.07, 0.35, 0.05, 0.10, 0.16, 0.39, 0.001, 0.001, 0.70, 0.70),
     (1.7638087673267722E-01, 2.8977285821363896E-01, 2.9778662086858310E-01, 3.8278231839045324E-01)),
    ((0.02, 0.30, -0.30, 0.50, 0.0, 0.11, 0.35, 0.05, 0.34, 0.31, 0.53, 0.12, 0.25, 0.60, 0.60),
     (4.2661468988808698E-01, 4.3671772638670370E-01, 5.3889559077522131E-01, 5.4539470544492241E-01)),
    ((4.05, 0.45, 0.25, 0.70, 0.0, 0.10, 0.45, 0.05, 0.25, 0.16, 0.39, 0.001, 0.001, 0.12, 0.12),
     (3.3187318412708881E-02, 4.6809060320339449E-02, 2.2980739802767661E-01, 2.8796786798139012E-01)),
    ((1.51, 0.44, 0.01, 0.15, 0.6, 0.07, 0.35, 0.05, 0.10, 0.16, 0.39, 0.001, 0.001, 0.75, 0.75),
     (2.7566303715402962E-01, 2.3148597887500627E-01, 1.9517975684778313E-01, 1.7209337143057937E-01)),
    ((1.51, 0.44, 0.01, 0.15, 1.0, 0.07, 0.35, 0.05, 0.10, 0.16, 0.39, 0.001, 0.001, 0.75, 0.75),
     (5.2617930981239891E-01, 4.5614119854441160E-01, 1.9167008254406520E-01, 1.5969733826126509E-01)),
    ((0.30, 0.40, -0.30, 0.05, 0.0, 0.11, 0.35, 0.05, 0.34, 0.31, 0.53, 0.12, 0.25, 0.65, 0.55),
     (1.8782200761871046E-01, 2.9159566439368728E-01, 5.0751217169742657E-01, 4.4765209618737845E-01)),
]


@pytest.mark.parametrize("inp,ref", FORTRAN)
def test_matches_executed_ctsm_fortran(inp, ref):
    el, es, xl, cz, fcs, rlv, rln, tlv, tln, rsv, rsn, tsv, tsn, gv, gn = inp
    vai = el + es
    got = []
    for rl, tl, rs, ts, g, oms in ((rlv, tlv, rsv, tsv, gv, 0.8),
                                   (rln, tln, rsn, tsn, gn, 0.4)):
        albd, albi = clm5_two_stream_albedo(
            np.array([vai]), np.array([el / max(vai, 1e-6)]), rl, tl, rs, ts, xl,
            np.array([cz]), np.array([g]), fcansno=fcs, omega_snow=oms)
        got += [float(albd[0]), float(albi[0])]
    np.testing.assert_allclose(got, ref, rtol=1e-12)

def test_no_canopy_is_the_ground():
    ag = np.array([0.73, 0.2])
    albd, albi = clm5_two_stream_albedo(
        np.zeros(2), np.zeros(2), 0.07, 0.05, 0.16, 0.001, 0.01,
        np.array([0.2, 0.2]), ag)
    np.testing.assert_array_equal(np.asarray(albd), ag)
    np.testing.assert_array_equal(np.asarray(albi), ag)


def test_canopy_hides_bright_ground():
    """More plant area over snow -> darker column (diffuse and beam)."""
    vai = np.array([0.1, 0.5, 1.0, 2.0, 4.0])
    n = vai.size
    albd, albi = clm5_two_stream_albedo(
        vai, np.full(n, 0.8), 0.07, 0.05, 0.16, 0.001, 0.01,
        np.full(n, 0.2), np.full(n, 0.8))
    assert np.all(np.diff(np.asarray(albd)) < 0)
    assert np.all(np.diff(np.asarray(albi)) < 0)
    assert float(albi[-1]) < 0.15


def test_diffuse_albedo_does_not_depend_on_sun_angle():
    """The diffuse column albedo carries no zenith dependence (only the beam
    one does), so the diffuse-only masking needs no sun angle."""
    cz = np.array([0.05, 0.2, 0.6, 1.0])
    _, albi = clm5_two_stream_albedo(
        np.full(4, 1.9), np.full(4, 0.77), 0.35, 0.10, 0.39, 0.001, 0.01, cz,
        np.full(4, 0.7))
    np.testing.assert_allclose(np.asarray(albi), float(albi[0]), rtol=1e-13)


# ---- the delta form and the land step -------------------------------------
from legoesm.land.canopy.radiative_transfer import canopy_masked_snow_albedo  # noqa: E402


def _masked(alb_sf, alb_sn, **kw):
    a = dict(band=0, LAI=np.array([1.51]), SAI=np.array([0.44]),
             htop=np.array([16.4]), hbot=np.array([8.2]),
             pft_index=np.array([2.0]), f_snow=np.array([1.0]),
             snow_depth=np.array([0.3]), cosz=np.array([0.3]),
             f_diffuse=np.array([1.0]))
    a.update(kw)
    return float(canopy_masked_snow_albedo(np.array([alb_sf]), np.array([alb_sn]), **a)[0])


def test_delta_form_limits_are_exact():
    # no snow: the snow-free albedo, bit for bit
    assert _masked(0.11, 0.11, f_snow=np.array([0.0])) == 0.11
    # no plant area (bare / glacier / fully buried): the unmasked snowy value
    assert _masked(0.11, 0.74, LAI=np.array([0.0]), SAI=np.array([0.0])) == 0.74
    # grass (PFT 13, 0.5 m) under 0.5 m of snow is buried -> unmasked
    # (CLM5: burial height 0.8 htop = 0.4 m < 0.5 m of snow): no exposed area
    # left, so the column is the plain snowy ground (CTSM non-vegetated filter)
    v = _masked(0.11, 0.74, pft_index=np.array([13.0]), htop=np.array([0.5]),
                hbot=np.array([0.0]), LAI=np.array([0.3]), SAI=np.array([0.4]),
                snow_depth=np.array([0.5]))
    assert v == 0.74
    # half-buried grass (0.2 m of snow): masked, darker than the bare snow
    h = _masked(0.11, 0.74, pft_index=np.array([13.0]), htop=np.array([0.5]),
                hbot=np.array([0.0]), LAI=np.array([0.3]), SAI=np.array([0.4]),
                snow_depth=np.array([0.2]))
    assert 0.3 < h < 0.74, h


def test_boreal_forest_hides_snow():
    """Needleleaf evergreen boreal in December over deep snow: the unmasked
    0.74 drops to the dark canopy (bare-branch bound, no canopy snow)."""
    vis = _masked(0.06, 0.74)
    nir = _masked(0.12, 0.74, band=1)
    assert 0.0 < vis < 0.15 and 0.05 < nir < 0.35, (vis, nir)
    # larch in winter: stems only, still masks part of the snow
    larch = _masked(0.06, 0.74, LAI=np.array([0.0]), SAI=np.array([0.55]),
                    pft_index=np.array([3.0]))
    assert vis < larch < 0.6, larch


# ---- land step: absorbed == exported, switch off untouched ------------------
import jax.numpy as jnp  # noqa: E402

from legoesm.core.coupling_fields import AtmToSurface  # noqa: E402
from legoesm.land.boundary_data.gap_fill import bare_canopy_params  # noqa: E402
from legoesm.land.canopy import radiative_transfer as rt  # noqa: E402
from legoesm.land.multilayer_land import (  # noqa: E402
    MultiLayerLandConfig, init_multilayer_land_state,
    step_multilayer_land_with_diagnostics)
from legoesm.land.surface_scheme import TwoLeafCanopyConfig  # noqa: E402

NCOL, SW = 4, 300.0


def _forcing(sw=SW):
    n = NCOL
    return AtmToSurface(
        sw_down=jnp.full(n, sw), lw_down=jnp.full(n, 250.0),
        precip_total=jnp.zeros(n), precip_snow=jnp.zeros(n),
        T_lowest=jnp.full(n, 268.0), q_lowest=jnp.full(n, 0.002),
        u_lowest=jnp.full(n, 5.0), v_lowest=jnp.full(n, 2.0),
        p_lowest=jnp.full(n, 95000.0), p_surface=jnp.full(n, 100000.0),
        rho_lowest=jnp.full(n, 1.2), cos_zenith=jnp.full(n, 0.5),
        co2_ppmv=jnp.full(n, 400.0), has_radiation=jnp.ones(n),
        has_precipitation=jnp.ones(n))


def _step(masking, sw=SW):
    cfg = MultiLayerLandConfig(snow_albedo_feedback=True,
                               surface_scheme=TwoLeafCanopyConfig(),
                               canopy_snow_masking=masking)
    state = init_multilayer_land_state(NCOL, cfg, T_init=265.0)
    # forest no snow / forest deep snow / bare deep snow / forest thin snow
    state = state._replace(snow_depth=jnp.asarray([0.0, 200.0, 200.0, 5.0]),
                           snow_age=jnp.full(NCOL, 2.0 * 86400.0))
    lp = bare_canopy_params(NCOL, canopy_structure=True)._replace(
        LAI=jnp.asarray([1.5, 1.5, 0.0, 1.5]), SAI_dom=jnp.asarray([0.44, 0.44, 0.0, 0.44]),
        hc=jnp.full(NCOL, 16.0), hbot_dom=jnp.full(NCOL, 8.0),
        pft_dom=jnp.asarray([2.0, 2.0, 0.0, 2.0]),
        ALB_VIS=jnp.full(NCOL, 0.06), ALB_NIR=jnp.full(NCOL, 0.12))
    new, resp, _, sfc = step_multilayer_land_with_diagnostics(
        state, _forcing(sw), cfg, 1.0, 600.0, lat=jnp.full(NCOL, 1.1),
        land_params=lp)
    _step.n_held = int(np.asarray(sfc.n_held).sum())
    return np.asarray(resp.albedo), np.asarray(sfc.sw_net)


def test_land_step_at_night_is_finite():
    """No sunlight: the beam/diffuse split is 0/0, so the exported albedo must
    fall back to the diffuse value, not NaN (it feeds the next radiation call)."""
    a_on, _ = _step(True, sw=0.0)
    # a NaN would be contained by the land step (column held, albedo replaced
    # by the 0.2 fallback): so test that nothing was held and the value is
    # the masked diffuse one, not the fallback
    assert _step.n_held == 0
    assert np.all(np.isfinite(a_on)), a_on
    assert 0.05 < a_on[1] < 0.18, a_on


def test_land_step_switch_on_masks_forest_snow_only():
    a_off, sw_off = _step(False)
    a_on, sw_on = _step(True)
    # snow-free forest and snowy bare ground: identical, bit for bit
    assert a_on[0] == a_off[0] and a_on[2] == a_off[2]
    assert sw_on[0] == sw_off[0] and sw_on[2] == sw_off[2]
    # deep-snow forest: unmasked ~0.7, masked dark canopy
    assert a_off[1] > 0.6 and a_on[1] < 0.2, (a_off[1], a_on[1])
    assert a_on[3] < a_off[3]
    # absorbed SW = (1 - masked broadband on the PRE-step state) * SW: the canopy
    # RT absorbed with exactly the masked bands
    np.testing.assert_allclose(sw_on, (1.0 - _pre_step_masked()) * SW, rtol=1e-9)


def _pre_step_masked():
    from legoesm import constants
    from legoesm.land.multilayer_land import compute_land_albedo
    from legoesm.surface_albedo import LandAlbedoConfig, snow_cover_fraction  # noqa: F401
    la = LandAlbedoConfig()
    swe = jnp.asarray([0.0, 200.0, 200.0, 5.0])
    age = jnp.full(NCOL, 2.0 * 86400.0)
    lai, sai = jnp.asarray([1.5, 1.5, 0.0, 1.5]), jnp.asarray([0.44, 0.44, 0.0, 0.44])
    pft = jnp.asarray([2.0, 2.0, 0.0, 2.0])
    cz = jnp.full(NCOL, 0.5)
    pd, pf, nd, nf, _ = rt.split_sw_components(jnp.full(NCOL, SW), cz)
    f_dif = (pf / (pd + pf), nf / (nd + nf))
    out = []
    for ib, a in ((0, 0.06), (1, 0.12)):
        base = jnp.full(NCOL, a)
        snowy = compute_land_albedo(jnp.full(NCOL, 1.1), swe, age, la, base_albedo=base)
        out.append(canopy_masked_snow_albedo(
            base, snowy, ib, lai, sai, jnp.full(NCOL, 16.0), jnp.full(NCOL, 8.0),
            pft, snow_cover_fraction(swe, la), swe / constants.rho_snow_land,
            cz, f_dif[ib]))
    return np.asarray(rt.broadband_albedo(*out))


def test_switch_without_canopy_structure_raises():
    cfg = MultiLayerLandConfig(snow_albedo_feedback=True,
                               surface_scheme=TwoLeafCanopyConfig(),
                               canopy_snow_masking=True)
    state = init_multilayer_land_state(NCOL, cfg, T_init=265.0)
    lp = bare_canopy_params(NCOL)
    with pytest.raises(ValueError, match="SAI_dom"):
        step_multilayer_land_with_diagnostics(
            state, _forcing(), cfg, 1.0, 600.0, lat=jnp.full(NCOL, 1.1),
            land_params=lp)


def test_switch_on_other_scheme_raises():
    from legoesm.land.surface_scheme import SimpleSEBConfig
    cfg = MultiLayerLandConfig(snow_albedo_feedback=True, canopy_snow_masking=True,
                               surface_scheme=SimpleSEBConfig())
    state = init_multilayer_land_state(NCOL, cfg, T_init=265.0)
    with pytest.raises(ValueError, match="canopy_snow_masking"):
        step_multilayer_land_with_diagnostics(
            state, _forcing(), cfg, 1.0, 600.0, lat=jnp.full(NCOL, 1.1))


def test_beam_blend_and_night_are_finite():
    """Beam light at low sun sees a darker sparse canopy than diffuse light; at
    night (no light) the diffuse value is used and nothing is NaN."""
    kw = dict(LAI=np.array([0.0]), SAI=np.array([0.55]), pft_index=np.array([3.0]),
              cosz=np.array([0.2]))
    beam = _masked(0.06, 0.74, f_diffuse=np.array([0.0]), **kw)
    diff = _masked(0.06, 0.74, f_diffuse=np.array([1.0]), **kw)
    half = _masked(0.06, 0.74, f_diffuse=np.array([0.5]), **kw)
    assert beam < diff and abs(half - 0.5 * (beam + diff)) < 1e-12
    night = _masked(0.06, 0.74, f_diffuse=np.array([1.0]),
                    **{**kw, "cosz": np.array([0.0])})
    assert np.isfinite(night) and night == diff


def test_gradients_finite_including_no_canopy_and_night():
    """Reverse-mode gradients stay finite on the masked branch, at VAI = 0
    (where() branch) and at night (cosz = 0, diffuse only)."""
    def f(lai, sai, snowy, depth, cz):
        return jnp.sum(canopy_masked_snow_albedo(
            jnp.full(4, 0.06), snowy, 0, lai, sai, jnp.full(4, 16.0),
            jnp.full(4, 8.0), jnp.asarray([2.0, 2.0, 13.0, 3.0]),
            jnp.full(4, 0.9), depth, cz, jnp.asarray([0.3, 1.0, 0.5, 0.0])))
    args = (jnp.asarray([1.5, 0.0, 0.3, 0.0]), jnp.asarray([0.44, 0.0, 0.4, 0.06]),
            jnp.full(4, 0.74), jnp.asarray([0.3, 0.3, 0.2, 0.1]),
            jnp.asarray([0.3, 0.0, 0.5, 0.2]))
    g = jax.grad(f, argnums=(0, 1, 2, 3, 4))(*args)
    for gi in g:
        assert np.all(np.isfinite(np.asarray(gi))), g
    # snowier ground -> brighter column wherever the snow shows (d/d snowy > 0)
    assert np.all(np.asarray(g[2]) > 0.0)


def _boreal_gsd():
    """2 columns: needleleaf evergreen boreal (dominant, 70%) + grass, and an
    uncovered column (gap-filled bare)."""
    from legoesm.land.global_surface_data import (
        GlobalSurfaceData, GlobalSurfaceDataConfig)
    from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5
    nb = CLM5_PFT_NAMES.index("needleleaf_evergreen_boreal")
    gr = CLM5_PFT_NAMES.index("c3_arctic_grass")
    ncol, npft = 2, N_PFT_CLM5
    pft = np.zeros((1, ncol, npft))
    pft[0, 0, nb], pft[0, 0, gr] = 0.7, 0.3
    m = lambda a, b: np.broadcast_to(np.where(np.arange(npft) == nb, a,
                                              np.where(np.arange(npft) == gr, b, 0.0)),
                                     (12, ncol, npft)).copy()
    z = np.zeros((ncol, 4))
    return nb, GlobalSurfaceData(
        sand_frac=jnp.asarray(np.full((ncol, 4), 0.4)),
        clay_frac=jnp.asarray(np.full((ncol, 4), 0.2)),
        organic=jnp.asarray(z), bulk_density=jnp.asarray(z),
        soil_color=jnp.asarray(np.array([5, 5])), cell_area=jnp.ones(ncol),
        years=jnp.asarray(np.array([2000.0])),
        f_land=jnp.ones((1, ncol)), f_lake=jnp.zeros((1, ncol)),
        f_glacier=jnp.zeros((1, ncol)), pft_frac=jnp.asarray(pft),
        months=jnp.arange(12.0),
        lai_monthly=jnp.asarray(m(1.5, 0.1)), sai_monthly=jnp.asarray(m(0.44, 0.3)),
        htop_monthly=jnp.asarray(m(16.0, 0.5)), hbot_monthly=jnp.asarray(m(8.0, 0.0)),
        config=GlobalSurfaceDataConfig())


def test_builders_carry_dominant_pft_structure():
    """Both the start-of-run builder and the per-step updater (the MPAS lane)
    hand the land step the DOMINANT plant's stem area, canopy bottom and type."""
    from legoesm.land.boundary_data.builders import build_canopy_params
    from legoesm.land.boundary_data.step_updater import make_step_land_params_updater
    from legoesm.land.canopy import CanopyConfig
    nb, gsd = _boreal_gsd()
    theta = jnp.asarray([0.28, 0.28])
    lp_b = build_canopy_params(gsd, 350.0, theta)
    lp_u, _ = make_step_land_params_updater(gsd, CanopyConfig())(
        theta, jnp.asarray(350.0), jnp.asarray(2000.0))
    for lp in (lp_b, lp_u):
        np.testing.assert_allclose(np.asarray(lp.SAI_dom)[0], 0.44, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(lp.hbot_dom)[0], 8.0, rtol=1e-12)
        assert float(np.asarray(lp.pft_dom)[0]) == float(nb)
        # uncovered column: bare (no stems, so the snow is never masked)
        assert float(np.asarray(lp.SAI_dom)[1]) == 0.0


def test_optics_cache_is_safe_under_jit():
    """The PFT optics table is cached; filling the cache from inside a jit trace
    must not store a tracer (a second, differently-shaped trace would then fail)."""
    from legoesm.land.canopy import radiative_transfer as _rt
    _rt._clm5_pft_optics_cached.cache_clear()
    f = jax.jit(lambda a, s: canopy_masked_snow_albedo(
        a, s, 1, jnp.full(a.shape, 1.5), jnp.full(a.shape, 0.44),
        jnp.full(a.shape, 16.0), jnp.full(a.shape, 8.0), jnp.full(a.shape, 2.0),
        jnp.full(a.shape, 0.9), jnp.full(a.shape, 0.3), jnp.full(a.shape, 0.3),
        jnp.full(a.shape, 0.5)))
    v1 = f(jnp.full(2, 0.12), jnp.full(2, 0.7))
    v2 = f(jnp.full(3, 0.12), jnp.full(3, 0.7))
    np.testing.assert_allclose(np.asarray(v2)[:2], np.asarray(v1), rtol=1e-13)


def test_switch_without_land_params_raises():
    """Two-leaf with no canopy parameters skips the snow layering altogether, so
    the switch would be silently inert: refused."""
    cfg = MultiLayerLandConfig(snow_albedo_feedback=True,
                               surface_scheme=TwoLeafCanopyConfig(),
                               canopy_snow_masking=True)
    state = init_multilayer_land_state(NCOL, cfg, T_init=265.0)
    with pytest.raises(ValueError, match="canopy_snow_masking"):
        step_multilayer_land_with_diagnostics(
            state, _forcing(), cfg, 1.0, 600.0, lat=jnp.full(NCOL, 1.1))



# Reference values from EXECUTING the verbatim CTSM 5.1 SatellitePhenologyMod.F90
# burial lines 173-178 and 185-188 (gfortran; scripts/validate/ctsm_burial_oracle.F90
# + ..._cases.txt).  (pft, htop, hbot, snow_depth, frac_sno, tlai, tsai) -> (elai, esai)
BURIAL = [
    ((2, 16.4, 8.2, 0.30, 1.0, 1.51, 0.44), (1.5100000000000000E+00, 4.4000000000000000E-01)),
    ((2, 16.4, 8.2, 9.00, 0.8, 1.51, 0.44), (1.3921463414634143E+00, 4.0565853658536583E-01)),
    ((11, 0.5, 0.1, 0.30, 0.9, 0.00, 0.31), (0.0000000000000000E+00, 1.7050000000000001E-01)),
    ((11, 0.5, 0.1, 0.60, 1.0, 0.20, 0.31), (0.0000000000000000E+00, 0.0000000000000000E+00)),
    ((12, 0.5, 0.0, 0.20, 1.0, 0.02, 0.30), (0.0000000000000000E+00, 1.4999999999999999E-01)),
    ((13, 0.5, 0.0, 0.50, 1.0, 0.30, 0.40), (0.0000000000000000E+00, 0.0000000000000000E+00)),
    ((15, 0.1, 0.0, 0.0908, 0.6, 0.40, 0.10), (1.6000000000000003E-01, 0.0000000000000000E+00)),
    ((13, 0.5, 0.0, 0.10, 0.3, 0.30, 0.40), (2.7749999999999997E-01, 3.7000000000000000E-01)),
    ((0, 0.0, 0.0, 0.30, 1.0, 0.00, 0.00), (0.0000000000000000E+00, 0.0000000000000000E+00)),
]


@pytest.mark.parametrize("inp,ref", BURIAL)
def test_burial_matches_executed_ctsm_fortran(inp, ref):
    from legoesm.land.canopy.radiative_transfer import clm5_exposed_area
    pft, ht, hb, sd, fs, tl, ts = inp
    a = lambda v: np.array([float(v)])
    elai, esai = clm5_exposed_area(a(tl), a(ts), a(ht), a(hb), a(pft), a(fs), a(sd))
    np.testing.assert_allclose([float(elai[0]), float(esai[0])], ref, rtol=1e-12, atol=0)


def test_tower_override_refused_even_after_cache_fill(monkeypatch):
    from legoesm.land.canopy import radiative_transfer as _rt
    from legoesm.land.canopy.clm_ml_backend.multilayer_canopy import MLclm_varctl
    _rt._clm5_pft_optics()                      # fill the cache with defaults
    monkeypatch.setattr(MLclm_varctl, "pftcon_val", 1)
    with pytest.raises(ValueError, match="pftcon_val"):
        _rt._clm5_pft_optics()


def test_finite_and_bounded_over_all_plants_and_light():
    """Sweep every PFT, sun angle, plant area, ground albedo and diffuse share:
    a NaN here would be silently replaced by the land step's held-column
    fallback, so it must never be produced."""
    pft = np.arange(17.0)
    cz = np.array([0.0, 0.02, 0.1, 0.4, 1.0])
    vai = np.array([0.0, 0.06, 0.5, 2.0, 6.0])
    grd = np.array([0.05, 0.5, 0.95])
    fd = np.array([0.0, 0.5, 1.0])
    depth = np.array([0.0, 0.3, 1.0, 3.0])   # reaches short plants AND hbot = 2 m
    fsn = np.array([0.3, 1.0])
    P, C, V, G, F, D, S = (x.ravel() for x in np.meshgrid(
        pft, cz, vai, grd, fd, depth, fsn, indexing="ij"))
    for band in (0, 1):
        out = np.asarray(canopy_masked_snow_albedo(
            np.minimum(G, 0.3), G, band, 0.7 * V, 0.3 * V, np.full_like(V, 10.0),
            np.full_like(V, 2.0), P, S, D, C, F))
        assert np.all(np.isfinite(out)), band
        assert np.all((out > 0.0) & (out < 1.0)), (out.min(), out.max())


def test_switch_off_ignores_the_new_structure_fields():
    """With the switch off, carrying SAI_dom/hbot_dom/pft_dom or not gives the
    same land step bit for bit (the fields are read only by the masking)."""
    cfg = MultiLayerLandConfig(snow_albedo_feedback=True,
                               surface_scheme=TwoLeafCanopyConfig())
    state = init_multilayer_land_state(NCOL, cfg, T_init=265.0)
    state = state._replace(snow_depth=jnp.asarray([0.0, 200.0, 200.0, 5.0]))
    base = bare_canopy_params(NCOL)._replace(
        LAI=jnp.full(NCOL, 1.5), hc=jnp.full(NCOL, 16.0),
        ALB_VIS=jnp.full(NCOL, 0.06), ALB_NIR=jnp.full(NCOL, 0.12))
    full = bare_canopy_params(NCOL, canopy_structure=True)._replace(
        LAI=base.LAI, hc=base.hc, ALB_VIS=base.ALB_VIS, ALB_NIR=base.ALB_NIR,
        SAI_dom=jnp.full(NCOL, 0.44), hbot_dom=jnp.full(NCOL, 8.0),
        pft_dom=jnp.full(NCOL, 2.0))
    outs = [step_multilayer_land_with_diagnostics(
        state, _forcing(), cfg, 1.0, 600.0, lat=jnp.full(NCOL, 1.1), land_params=lp)
        for lp in (base, full)]
    la, lb = jax.tree.leaves(outs[0][:2]), jax.tree.leaves(outs[1][:2])
    assert len(la) == len(lb) > 10
    for x, y in zip(la, lb):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()



def test_burial_uses_physical_snow_cover_not_the_albedo_scale():
    """User 2026-10-03: the calibrated snow_cover_scale is a brightness knob; the
    burial sees the physical cover.  Grass under deep snow with a scale of 0.5:
    physical cover ~1 -> fully buried -> the column is the plain snowy ground."""
    from legoesm.land.multilayer_land import _canopy_masked_band
    from legoesm.surface_albedo import LandAlbedoConfig
    la = LandAlbedoConfig(snow_cover_scale=jnp.full(NCOL, 0.5))
    cfg = MultiLayerLandConfig(snow_albedo_feedback=True, land_albedo=la,
                               surface_scheme=TwoLeafCanopyConfig(),
                               canopy_snow_masking=True)
    lp = bare_canopy_params(NCOL, canopy_structure=True)._replace(
        LAI=jnp.full(NCOL, 0.3), SAI_dom=jnp.full(NCOL, 0.4), hc=jnp.full(NCOL, 0.5),
        hbot_dom=jnp.zeros(NCOL), pft_dom=jnp.full(NCOL, 13.0))
    swe = jnp.full(NCOL, 300.0)                      # 1.2 m of snow
    snowy = lambda a: a + 0.6
    f = _canopy_masked_band(snowy, lp, None, swe, cfg, 0, _forcing())
    out = np.asarray(f(jnp.full(NCOL, 0.1)))
    np.testing.assert_array_equal(out, np.full(NCOL, 0.7))
