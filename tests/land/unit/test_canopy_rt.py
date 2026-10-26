"""Unit tests for canopy/radiative_transfer.py.

Checks:
- sw decomposition: PAR + NIR + UV ≈ sw_down during day
- Night guard: all components = 0 when cos_zenith < 0.01
- Two-leaf RT: fSun ∈ [0, 1], APAR_Sun + APAR_Sh = total absorbed PAR
- Energy conservation: ASW_Sun + ASW_Sh + ASW_Soil ≤ sw_down
- Vcmax25_Sun + Vcmax25_Sh > 0 when LAI > 0
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.land.canopy.radiative_transfer import (
    canopy_longwave_rt,
    canopy_shortwave_rt,
    split_sw_components,
)


def test_split_sw_partitions_daytime():
    sw_down = jnp.array([800.0, 400.0])
    cos_zenith = jnp.array([0.8, 0.3])
    PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV = split_sw_components(
        sw_down, cos_zenith)
    total = PAR_dir + PAR_diff + NIR_dir + NIR_diff + UV
    assert jnp.allclose(total, sw_down, rtol=1e-5, atol=1e-3)


def test_split_sw_night_guard():
    sw_down = jnp.array([10.0])
    cos_zenith = jnp.array([0.001])  # below 0.01 night threshold
    comps = split_sw_components(sw_down, cos_zenith)
    for c in comps:
        assert float(c[0]) == 0.0


def test_canopy_sw_absorption_bounded():
    sw_down = jnp.array([800.0])
    cos_zenith = jnp.array([0.85])
    PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV = split_sw_components(
        sw_down, cos_zenith)
    SZA = jnp.degrees(jnp.arccos(cos_zenith))
    out = canopy_shortwave_rt(
        PAR_dir, PAR_diff, NIR_dir, NIR_diff, UV,
        SZA=SZA,
        LAI=jnp.array([3.0]),
        CI=jnp.array([0.75]),
        ALB_VIS=jnp.array([0.08]),
        ALB_NIR=jnp.array([0.25]),
        Vcmax25_C3_leaf=jnp.array([60.0]),
        Vcmax25_C4_leaf=jnp.array([0.0]),
        kn=jnp.array([0.3]),
    )
    # Energy conservation: absorbed ≤ incoming
    absorbed = out.ASW_Sun + out.ASW_Sh + out.ASW_Soil
    assert float(absorbed[0]) > 0.0
    assert float(absorbed[0]) <= float(sw_down[0]) + 1e-3
    # Sunlit fraction in [0, 1]
    assert 0.0 <= float(out.fSun[0]) <= 1.0
    # Vcmax distributed to both fractions
    assert float(out.Vcmax25_C3Sun[0]) > 0.0
    assert float(out.Vcmax25_C3Sh[0]) > 0.0


def test_canopy_sw_night_zero():
    out = canopy_shortwave_rt(
        jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]),
        jnp.array([0.0]), jnp.array([0.0]),
        SZA=jnp.array([89.5]),
        LAI=jnp.array([3.0]), CI=jnp.array([0.75]),
        ALB_VIS=jnp.array([0.08]), ALB_NIR=jnp.array([0.25]),
        Vcmax25_C3_leaf=jnp.array([60.0]),
        Vcmax25_C4_leaf=jnp.array([0.0]),
        kn=jnp.array([0.3]),
    )
    assert float(out.APAR_Sun[0] + out.APAR_Sh[0]) == 0.0
    assert float(out.ASW_Sun[0] + out.ASW_Sh[0] + out.ASW_Soil[0]) == 0.0


# ---------------------------------------------------------------------------
# Longwave radiative transfer
# ---------------------------------------------------------------------------

def test_longwave_isothermal_blackbody_zero():
    """Fully isothermal blackbody (leaves=soil=air, eps=1) → zero net LW.

    Radiative-equilibrium / detailed-balance sanity: if every emitter is at
    the same temperature and emissivity 1, the down-welling La exactly equals
    every body's own emission, so each net absorbed flux is zero.  This relies
    on the kernel identities ``W_*_sky + W_*_soil`` summing correctly and
    ``W_tot + gap_LW == 1``.
    """
    T = jnp.array([295.0, 280.0])
    La = constants.sigma_sb * T ** 4
    out = canopy_longwave_rt(
        LAI=jnp.array([3.0, 1.0]), CI=jnp.array([0.75, 0.9]),
        SZA=jnp.array([30.0, 70.0]), Ts=T, Tf_Sun=T, Tf_Sh=T, La=La,
        epsf=1.0, epss=1.0,
    )
    for v in (out.ALW_Sun, out.ALW_Sh, out.ALW_Soil):
        assert jnp.allclose(v, 0.0, atol=1e-6)


def test_longwave_global_conservation():
    """Conservative gray-body scheme: ΣALW == La − LW_out for any εf, εs.

    The near-black scheme this replaces fails this for ε<1 (it has no reflected
    term).  Checked at the physical εf=0.97/εs=0.96 and at the εf=εs=1 limit.
    """
    LAI = jnp.array([3.0, 1.0, 5.0]); CI = jnp.array([0.75, 0.6, 0.9])
    SZA = jnp.array([30.0, 70.0, 10.0]); Ts = jnp.array([305.0, 315.0, 298.0])
    TfS = jnp.array([300.0, 308.0, 297.0]); TfH = jnp.array([297.0, 303.0, 296.0])
    La = jnp.array([340.0, 300.0, 380.0])
    for epsf, epss in ((0.97, 0.96), (1.0, 1.0), (0.9, 0.85)):
        out = canopy_longwave_rt(LAI, CI, SZA, Ts, TfS, TfH, La, epsf, epss)
        sigma_alw = out.ALW_Sun + out.ALW_Sh + out.ALW_Soil
        assert jnp.allclose(sigma_alw, La - out.LW_out, rtol=0, atol=1e-9)


def test_longwave_isothermal_graybody_zero():
    """Isothermal equilibrium at εf<1, εs<1 → all net LW = 0 (the gray-body fix).

    With every emitter at T and a black sky (La=σT⁴), the conservative scheme
    gives exactly zero net flux for ANY emissivity — the property the near-black
    scheme violated (it gained O(1−ε)·σT⁴).
    """
    T = jnp.array([295.0, 280.0])
    La = constants.sigma_sb * T ** 4
    out = canopy_longwave_rt(
        LAI=jnp.array([3.0, 1.0]), CI=jnp.array([0.75, 0.9]),
        SZA=jnp.array([30.0, 70.0]), Ts=T, Tf_Sun=T, Tf_Sh=T, La=La,
        epsf=0.97, epss=0.96,
    )
    for v in (out.ALW_Sun, out.ALW_Sh, out.ALW_Soil):
        assert jnp.allclose(v, 0.0, atol=1e-9)
    # and the column looks like a blackbody at T from above: LW_out == σT⁴
    assert jnp.allclose(out.LW_out, La, atol=1e-9)


def test_longwave_clumping_reduces_canopy_absorption():
    """Smaller clumping index → smaller effective LAI → less canopy LW exchange.

    Net canopy LW exchange magnitude must shrink monotonically as CI falls
    (clumped canopies are more transparent), and in the CI→0 limit the soil
    sees nearly the full atmospheric LW (gap_LW→1).
    """
    LAI = jnp.array([3.0]); SZA = jnp.array([30.0])
    Ts = jnp.array([305.0]); Tf_Sun = jnp.array([300.0]); Tf_Sh = jnp.array([297.0])
    La = jnp.array([340.0])
    mags = []
    for ci in (1.0, 0.6, 0.2):
        out = canopy_longwave_rt(
            LAI, jnp.array([ci]), SZA, Ts, Tf_Sun, Tf_Sh, La, 0.97, 0.96)
        mags.append(float(jnp.abs(out.ALW_Sun + out.ALW_Sh)[0]))
    assert mags[0] > mags[1] > mags[2]


def test_longwave_finite_and_smooth_across_kb_equals_kd():
    """No NaN/Inf and C0-continuity through the SZA where kb == kd (~50°).

    ``denom = kd - kb`` crosses zero near SZA≈50°; the guarded L'Hôpital
    branch must keep the result finite and smooth (no spike).
    """
    SZA = jnp.linspace(45.0, 55.0, 41)
    n = SZA.shape[0]
    out = canopy_longwave_rt(
        LAI=jnp.full(n, 3.0), CI=jnp.full(n, 0.75), SZA=SZA,
        Ts=jnp.full(n, 305.0), Tf_Sun=jnp.full(n, 300.0),
        Tf_Sh=jnp.full(n, 297.0), La=jnp.full(n, 340.0), epsf=0.97, epss=0.96,
    )
    for v in (out.ALW_Sun, out.ALW_Sh, out.ALW_Soil):
        assert bool(jnp.all(jnp.isfinite(v)))
        # No grid-scale spike: successive differences stay small/smooth.
        assert float(jnp.max(jnp.abs(jnp.diff(v)))) < 5.0


def test_longwave_differentiable_at_singular_sza():
    """grad wrt Tf_Sun is finite even at kb≈kd (denom guard) and at night."""
    def lw_sum(Tf_Sun_scalar, sza):
        out = canopy_longwave_rt(
            LAI=jnp.array([3.0]), CI=jnp.array([0.75]), SZA=jnp.array([sza]),
            Ts=jnp.array([305.0]), Tf_Sun=jnp.array([Tf_Sun_scalar]),
            Tf_Sh=jnp.array([297.0]), La=jnp.array([340.0]),
            epsf=0.97, epss=0.96,
        )
        return jnp.sum(out.ALW_Sun + out.ALW_Sh + out.ALW_Soil)

    for sza in (50.14, 89.0, 30.0):
        g = jax.grad(lw_sum)(300.0, sza)
        assert jnp.isfinite(g)


def test_longwave_sza_gradient_matches_fd_through_kb_eq_kd():
    """d(ALW_Sun)/d(SZA) matches finite differences across kb=kd (~50.14°).

    The exprel formulation must give the correct, continuous SZA derivative
    through the removable singularity (the old constant-limit guard zeroed it).
    """
    def alw_sun(sza):
        out = canopy_longwave_rt(
            LAI=jnp.array([3.0]), CI=jnp.array([0.75]), SZA=jnp.array([sza]),
            Ts=jnp.array([305.0]), Tf_Sun=jnp.array([300.0]),
            Tf_Sh=jnp.array([297.0]), La=jnp.array([340.0]), epsf=0.97, epss=0.96)
        return out.ALW_Sun[0]

    for sza in (50.14, 49.5, 50.8):
        g_ad = jax.grad(alw_sun)(sza)
        g_fd = (alw_sun(sza + 1e-3) - alw_sun(sza - 1e-3)) / 2e-3
        assert jnp.isfinite(g_ad)
        assert jnp.allclose(g_ad, g_fd, rtol=1e-4, atol=1e-6)


def test_longwave_atmosphere_equivalent_reconstruction():
    """eps_col + LW_emit reproduce LW_out via the atmosphere's property-coupling.

    The coupler hands RRTMGP (eps_col, T_surface); RRTMGP forms the surface LW
    boundary eps_col*sigma*T^4 + (1-eps_col)*La.  With T_surface derived from
    LW_emit = LW_out - R_col*La and eps_col = 1 - R_col, this MUST equal the
    canopy's conservative LW_out EXACTLY for any eps_f, eps_s, LAI — closing the
    land->atmosphere LW consistency gap.
    """
    LAI = jnp.array([0.1, 1.0, 3.0, 7.0]); CI = jnp.array([0.7, 0.8, 0.6, 0.9])
    SZA = jnp.array([20.0, 40.0, 60.0, 10.0]); Ts = jnp.array([305.0, 310.0, 300.0, 295.0])
    TfS = jnp.array([300.0, 305.0, 298.0, 294.0]); TfH = jnp.array([297.0, 301.0, 296.0, 293.0])
    La = jnp.array([340.0, 300.0, 360.0, 380.0])
    for epsf, epss in ((0.97, 0.96), (0.9, 0.85), (1.0, 1.0)):
        out = canopy_longwave_rt(LAI, CI, SZA, Ts, TfS, TfH, La, epsf, epss)
        T_surface = (out.LW_emit / (out.eps_col * constants.sigma_sb)) ** 0.25
        recon = (out.eps_col * constants.sigma_sb * T_surface ** 4
                 + (1.0 - out.eps_col) * La)
        assert jnp.allclose(recon, out.LW_out, rtol=0, atol=1e-8)
        # physical: column emissivity in (0,1], emission-only flux positive
        assert bool(jnp.all(out.eps_col > 0.0)) and bool(jnp.all(out.eps_col <= 1.0))
        assert bool(jnp.all(out.LW_emit > 0.0))
