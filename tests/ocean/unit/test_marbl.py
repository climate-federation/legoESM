"""MARBL port: carbonate chemistry, conversions, interior budgets, gradients.

Run with JAX_ENABLE_X64=1 (budget tolerances assume float64).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.biogeochemistry.carbonate import (
    _alkalinity_residual,
    carbonate_coefficients,
    carbonate_speciation,
    co3_saturation,
    solve_htotal,
)
from legoesm.ocean.biogeochemistry.gas_exchange import o2_saturation
from legoesm.ocean.biogeochemistry.marbl import (
    MarblConfig,
    MarblInteriorForcing,
    MarblSurfaceForcing,
    MarblTracers,
    marbl_interior_tendencies,
    marbl_surface_fluxes,
)
from legoesm.ocean.biogeochemistry.marbl.config import TRACER_NAMES
from legoesm.ocean.eos import adiabatic_temperature_gradient, potential_temperature

X64 = jax.config.jax_enable_x64
CFG = MarblConfig()


# ---------------------------------------------------------------------------
# carbonate chemistry
# ---------------------------------------------------------------------------

def test_constants_match_doe_check_values():
    """DOE (1994) handbook check values, S = 35, T = 25 degC, total scale."""
    c = carbonate_coefficients(jnp.asarray(25.0), jnp.asarray(35.0))
    ln = lambda k: float(jnp.log(k))  # noqa: E731
    assert ln(c.k0) == pytest.approx(-3.5617, abs=2e-3)
    assert -np.log10(float(c.k1)) == pytest.approx(5.8472, abs=2e-3)
    assert -np.log10(float(c.k2)) == pytest.approx(8.9660, abs=2e-3)
    assert ln(c.kb) == pytest.approx(-19.7964, abs=2e-3)
    assert ln(c.k1p) == pytest.approx(-3.71, abs=1e-2)
    assert ln(c.k2p) == pytest.approx(-13.727, abs=1e-2)
    assert ln(c.k3p) == pytest.approx(-20.24, abs=1e-2)
    assert ln(c.ksi) == pytest.approx(-21.61, abs=1e-2)
    assert ln(c.kw) == pytest.approx(-30.434, abs=2e-2)
    assert ln(c.ks) == pytest.approx(-2.30, abs=1e-2)
    assert ln(c.kf) == pytest.approx(-6.09, abs=2e-2)


def test_calcite_aragonite_solubility_mucci():
    """Mucci (1983): log10 Ksp = -6.3693 (calcite), -6.1883 (aragonite) at S=35, 25 degC."""
    co3_c, co3_a = co3_saturation(jnp.asarray(25.0), jnp.asarray(35.0))
    Ca = 0.01028  # mol/kg at S = 35
    assert np.log10(float(co3_c) * Ca) == pytest.approx(-6.3693, abs=1e-3)
    assert np.log10(float(co3_a) * Ca) == pytest.approx(-6.1883, abs=1e-3)


def test_pressure_correction_identity_at_zero_pressure():
    T, S = jnp.asarray(10.0), jnp.asarray(34.5)
    c0 = carbonate_coefficients(T, S)
    cp = carbonate_coefficients(T, S, jnp.asarray(0.0))
    for a, b in zip(c0, cp):
        assert float(b) == pytest.approx(float(a), rel=1e-12)
    # pressure increases K1 and the calcite solubility product
    c400 = carbonate_coefficients(T, S, jnp.asarray(400.0))
    assert float(c400.k1) > float(c0.k1)
    assert float(co3_saturation(T, S, jnp.asarray(400.0))[0]) > float(co3_saturation(T, S)[0])


@pytest.mark.parametrize("T", [-1.5, 10.0, 28.0])
def test_alkalinity_balance_solved(T):
    T = jnp.asarray(T)
    S = jnp.asarray(35.0)
    c = carbonate_coefficients(T, S, jnp.asarray(200.0))
    dic, ta, pt, sit = (jnp.asarray(v) for v in (2.1e-3, 2.3e-3, 2e-6, 50e-6))
    H = solve_htotal(dic, ta, pt, sit, c)
    resid = _alkalinity_residual(H, dic, ta, pt, sit, c)
    assert abs(float(resid)) < 1e-12
    pH = -np.log10(float(H))
    assert 7.3 < pH < 8.4


def test_speciation_sums_to_dic_and_gradient():
    c = carbonate_coefficients(jnp.asarray(15.0), jnp.asarray(35.0))
    pt, sit = jnp.asarray(1e-6), jnp.asarray(10e-6)

    def co3(dic):
        return carbonate_speciation(dic, jnp.asarray(2.3e-3), pt, sit, c).CO3

    spec = carbonate_speciation(jnp.asarray(2.05e-3), jnp.asarray(2.3e-3), pt, sit, c)
    assert float(spec.H2CO3 + spec.HCO3 + spec.CO3) == pytest.approx(2.05e-3, rel=1e-12)
    g = jax.grad(co3)(jnp.asarray(2.05e-3))
    eps = 1e-9
    fd = (co3(jnp.asarray(2.05e-3 + eps)) - co3(jnp.asarray(2.05e-3 - eps))) / (2 * eps)
    assert float(g) < 0.0  # more DIC at fixed TA lowers carbonate ion
    assert float(g) == pytest.approx(float(fd), rel=1e-5)


# ---------------------------------------------------------------------------
# conversions shared with MARBL
# ---------------------------------------------------------------------------

def test_o2_saturation_check_value():
    """Garcia & Gordon (1992) check value used by MARBL: 282.015 mmol/m^3."""
    assert float(o2_saturation(jnp.asarray(10.0), jnp.asarray(35.0))) * 1e3 == pytest.approx(
        282.015, rel=1e-5)


def test_potential_temperature_unesco_check_values():
    S, T, p = jnp.asarray(40.0), jnp.asarray(40.0), jnp.asarray(1000.0)
    assert float(adiabatic_temperature_gradient(S, T, p)) == pytest.approx(3.255976e-3, rel=1e-6)
    assert float(potential_temperature(S, T, p, jnp.asarray(0.0))) == pytest.approx(36.89073, abs=1e-5)


# ---------------------------------------------------------------------------
# MARBL interior tendencies on an idealized column
# ---------------------------------------------------------------------------

def _column(nlev=30, depth=4000.0, omz=True, biology=True):
    edges = depth * (np.linspace(0.0, 1.0, nlev + 1) ** 1.8)
    dz = np.diff(edges)
    zt = 0.5 * (edges[:-1] + edges[1:])
    zw = edges[1:]
    surf = np.exp(-zt / 60.0)
    deep = 1.0 - np.exp(-zt / 400.0)
    O2 = 230.0 - 120.0 * deep
    if omz:
        O2 = np.where((zt > 400.0) & (zt < 900.0), 3.0, O2)
    bio = surf if biology else 0.0 * surf
    tr = dict(
        PO4=0.1 + 2.1 * deep, NO3=0.8 + 32.0 * deep, SiO3=2.0 + 60.0 * deep,
        NH4=0.1 * surf + 0.01, Fe=2e-4 + 5e-4 * deep, Lig=2.5e-3 + 0.0 * zt, O2=O2,
        DIC=2000.0 + 250.0 * deep, DIC_ALT_CO2=2000.0 + 250.0 * deep,
        ALK=2350.0 + 50.0 * deep, ALK_ALT_CO2=2350.0 + 50.0 * deep,
        DOC=40.0 + 30.0 * surf, DON=2.0 + 3.0 * surf, DOP=0.05 + 0.2 * surf,
        DOPr=0.1 + 0.0 * zt, DONr=2.0 + 0.0 * zt, DOCr=40.0 + 0.0 * zt,
        zooC=0.6 * bio,
        spChl=0.04 * bio, spC=1.2 * bio, spP=1.2 * bio / 117.0, spFe=1.2 * bio * 2e-5,
        spCaCO3=0.06 * bio,
        diatChl=0.03 * bio, diatC=0.6 * bio, diatP=0.6 * bio / 117.0, diatFe=0.6 * bio * 2e-5,
        diatSi=0.6 * bio * 0.15,
        diazChl=0.002 * bio, diazC=0.05 * bio, diazP=0.05 * bio / 300.0, diazFe=0.05 * bio * 4e-5,
    )
    tracers = MarblTracers(**{k: jnp.asarray(v) for k, v in tr.items()})
    T = 26.0 * np.exp(-zt / 300.0) + 2.0
    forcing = MarblInteriorForcing(
        potemp=jnp.asarray(T), salinity=jnp.asarray(35.0 + 0.0 * zt),
        pressure_bar=jnp.asarray(zt * constants.rho_ocean * constants.g / 1e5),
        surf_shortwave=jnp.asarray([250.0]), PAR_col_frac=jnp.asarray([1.0]),
        dust_flux=jnp.asarray(1.0e-11),
        fesedflux=jnp.zeros(nlev).at[-1].set(1e-9),
        feventflux=jnp.zeros(nlev).at[-2].set(5e-10),
    )
    return tracers, forcing, jnp.asarray(dz), jnp.asarray(zt), jnp.asarray(zw)


def _tendencies(cfg=CFG, **kw):
    tracers, forcing, dz, zt, zw = _column(**kw)
    tend, diag = marbl_interior_tendencies(tracers, forcing, dz, zt, zw, jnp.asarray(dz.shape[0]), cfg)
    return tracers, forcing, dz, tend, diag


def _integral(dz, *fields):
    return float(sum(jnp.sum(f * dz) for f in fields))


def _gross(dz, *fields):
    return float(sum(jnp.sum(jnp.abs(f) * dz) for f in fields))


@pytest.mark.skipif(not X64, reason="budget closure needs float64")
def test_carbon_budget_closes():
    _, _, dz, d, g = _tendencies()
    C = (d.DIC, d.DOC, d.DOCr, d.zooC, d.spC, d.diatC, d.diazC, d.spCaCO3)
    residual = _integral(dz, *C) + float(g.POC_sed_loss + g.CaCO3_sed_loss)
    assert abs(residual) < 1e-11 * _gross(dz, *C)
    C_alt = (d.DIC_ALT_CO2, d.DOC, d.DOCr, d.zooC, d.spC, d.diatC, d.diazC, d.spCaCO3)
    residual_alt = _integral(dz, *C_alt) + float(g.POC_sed_loss + g.CaCO3_ALT_CO2_sed_loss)
    assert abs(residual_alt) < 1e-11 * _gross(dz, *C_alt)


@pytest.mark.skipif(not X64, reason="budget closure needs float64")
def test_nitrogen_budget_closes():
    _, _, dz, d, g = _tendencies()
    Q = CFG.Q
    N = (d.NO3, d.NH4, d.DON, d.DONr, Q * d.zooC, Q * d.spC, Q * d.diatC, Q * d.diazC)
    sources = _integral(dz, g.Nfix) - _integral(dz, g.denitrif, g.sed_denitrif)
    residual = _integral(dz, *N) + float(g.PON_sed_loss) - sources
    assert _integral(dz, g.denitrif) > 0.0  # the OMZ exercises denitrification
    assert abs(residual) < 1e-11 * _gross(dz, *N)


@pytest.mark.skipif(not X64, reason="budget closure needs float64")
def test_phosphorus_silicon_budgets_close():
    _, _, dz, d, g = _tendencies()
    P = (d.PO4, d.DOP, d.DOPr, CFG.Qp_zoo * d.zooC, d.spP, d.diatP, d.diazP)
    residual_P = _integral(dz, *P) + float(g.POP_sed_loss)
    assert abs(residual_P) < 1e-11 * _gross(dz, *P)
    Si = (d.SiO3, d.diatSi)
    residual_Si = _integral(dz, *Si) + float(g.SiO2_sed_loss)
    assert abs(residual_Si) < 1e-11 * _gross(dz, *Si)


@pytest.mark.skipif(not X64, reason="budget closure needs float64")
def test_iron_budget_closes():
    tracers, forcing, dz, d, g = _tendencies()
    Fe = (d.Fe, d.spFe, d.diatFe, d.diazFe, CFG.Qfe_zoo * d.zooC)
    external = (_integral(dz, g.dust_Fe_release)
                + float(jnp.sum(forcing.fesedflux) + jnp.sum(forcing.feventflux)))
    residual = _integral(dz, *Fe) - external + float(g.P_iron_sed_loss)
    assert abs(residual) < 1e-11 * _gross(dz, *Fe)


def test_no_biology_is_finite_and_quiet():
    _, _, dz, d, g = _tendencies(biology=False, omz=False)
    for name, v in zip(TRACER_NAMES, d):
        assert bool(jnp.all(jnp.isfinite(v))), name
    for name in ("spC", "diatC", "diazC", "zooC", "spChl"):
        assert float(jnp.max(jnp.abs(getattr(d, name)))) == 0.0, name


def test_production_is_light_and_depth_structured():
    _, _, dz, d, g = _tendencies()
    npp = jnp.sum(g.photoC, axis=0)
    assert float(npp[0]) > 0.0
    assert float(npp[-1]) == 0.0
    # sinking POC flux attenuates below the euphotic zone
    assert float(g.POC_flux_out[10]) < float(jnp.max(g.POC_flux_out))


def test_gradient_wrt_parameters_matches_fd():
    tracers, forcing, dz, zt, zw = _column()
    kmt = jnp.asarray(dz.shape[0])

    def export(scale_pcref, scale_kfe):
        cfg = CFG._replace(
            PCref_per_day=tuple(scale_pcref * v for v in CFG.PCref_per_day),
            kFe=tuple(scale_kfe * v for v in CFG.kFe))
        d, g = marbl_interior_tendencies(tracers, forcing, dz, zt, zw, kmt, cfg)
        return jnp.sum(jnp.sum(g.photoC, axis=0) * dz)

    grads = jax.grad(export, argnums=(0, 1))(1.0, 1.0)
    for i, g in enumerate(grads):
        assert bool(jnp.isfinite(g))
        h = 1e-6
        args_p = [1.0, 1.0]
        args_m = [1.0, 1.0]
        args_p[i] += h
        args_m[i] -= h
        fd = (export(*args_p) - export(*args_m)) / (2 * h)
        assert float(g) == pytest.approx(float(fd), rel=1e-4, abs=1e-14)
    assert float(grads[0]) > 0.0   # faster growth -> more production
    assert float(grads[1]) < 0.0   # higher half-saturation -> less production


def test_surface_fluxes_signs():
    tracers, *_ = _column()
    surf = jax.tree_util.tree_map(lambda a: a[0], tracers)
    forcing = MarblSurfaceForcing(
        sst=jnp.asarray(26.0), sss=jnp.asarray(35.0), u10_sqr=jnp.asarray(64.0),
        atm_press=jnp.asarray(1.0), ifrac=jnp.asarray(0.0),
        xco2=jnp.asarray(420.0), xco2_alt_co2=jnp.asarray(280.0),
        dust_flux=jnp.asarray(1e-11), iron_flux=jnp.asarray(1e-9),
        nox_flux=jnp.asarray(1e-7), nhy_flux=jnp.asarray(1e-7),
    )
    flux, diag = marbl_surface_fluxes(surf, forcing, CFG)
    # undersaturated O2 at the surface (230 mmol/m^3 at 26 degC is supersaturated)
    assert float(diag.o2sat) == pytest.approx(
        float(o2_saturation(jnp.asarray(26.0), jnp.asarray(35.0))) * 1e3, rel=1e-12)
    assert np.sign(float(flux.O2)) == np.sign(float(diag.o2sat) - 230.0)
    # alternative CO2 at 280 ppm outgasses relative to 420 ppm
    assert float(flux.DIC_ALT_CO2) < float(flux.DIC)
    assert float(diag.nhx_surface_emis) > 0.0
    assert float(flux.ALK) == pytest.approx(float(flux.NH4 - flux.NO3), rel=1e-12)
    for name in ("DOC", "spC", "zooC", "Lig"):
        assert float(getattr(flux, name)) == 0.0
