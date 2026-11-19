"""Issue #1816: implicit CaCO3 pool + sea-floor return in the NPZD ecosystem.

Oracle: MITgcm pkg/dic (OCMIP-2 biotic protocol), commit c3f71ac8:
``car_flux.F`` (implicit CaCO3 dissolution), ``dic_biotic_forcing.F``
(CAR_S = rain_ratio x particulate organic production, GDIC += cflux - CAR_S,
GALK += 2*(cflux - CAR_S)) and ``dic_init_fixed.F`` (production levels
``nlev`` from ``zcrit``). The two oracle loops are transcribed literally
below (1-based Fortran indices, shifted by -1 on each array access).

Conserved column integrals of the ecosystem tendencies (no air-sea flux):
carbon DIC + R_CN*(P+Z+D), nitrogen NO3+P+Z+D, alkalinity ALK + NO3.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

from legoesm.ocean.biogeochemistry.carbon_cycle import (
    compute_biogeo_tendencies,
)
from legoesm.ocean.biogeochemistry.config import (
    BiogeoConfig,
    OceanBiogeoState,
)
from legoesm.ocean.biogeochemistry.npzd import (
    implicit_caco3_dissolution,
    npzd_source_sink,
)


@pytest.fixture(autouse=True)
def _x64():
    """fp64 for this module only (does not leak into the session)."""
    with jax.enable_x64(True):
        yield

_DAY_S = 86400.0  # s per day (the rate unit of BiogeoConfig)
_CANCEL_RTOL = 1e-14  # fp64 rounding of the organic terms subtracted out


# ---------------------------------------------------------------------------
# Literal oracle transcriptions
# ---------------------------------------------------------------------------


def _oracle_nlev(drF, zcrit):
    """dic_init_fixed.F: k=1; DO WHILE (k.LT.Nr .AND. rC(k).GE.-zcrit) k=k+1."""
    Nr = len(drF)
    rF = np.concatenate([[0.0], -np.cumsum(drF)])
    rC = 0.5 * (rF[:-1] + rF[1:])
    k = 1
    while k < Nr and rC[k - 1] >= -zcrit:
        k += 1
    return k


def _car_flux_oracle(car_s, drF, maskC, zca, nlev):
    """car_flux.F for ONE column (linear free surface, hFacC = maskC)."""
    Nr = len(drF)
    rF = np.concatenate([[0.0], -np.cumsum(drF)])  # rF(1..Nr+1)
    recip_hFacC = np.where(maskC > 0, 1.0, 0.0)
    cflux = np.zeros(Nr)
    for k in range(1, nlev + 1):  # DO k=1,nlev
        caexport = 0.0
        if maskC[k - 1] > 0.0:
            if k == Nr:
                cflux[k - 1] += car_s[k - 1]
            elif maskC[k] == 0.0:  # hFacC(k+1).EQ.0
                cflux[k - 1] += car_s[k - 1]
            else:
                caexport = car_s[k - 1] * drF[k - 1] * maskC[k - 1]
        zbase = -rF[k]  # -rF(k+1)
        flux_u = caexport
        for ko in range(k + 1, Nr):  # DO ko=k+1,Nr-1
            kop1 = min(Nr, ko + 1)
            depth_l = -rF[ko - 1] + drF[ko - 1]
            reminFac = np.exp(-(depth_l - zbase) / zca)
            if caexport != 0.0:
                flux_l = caexport * reminFac * maskC[kop1 - 1]
                cflux[ko - 1] += (flux_u - flux_l) / drF[ko - 1] * recip_hFacC[ko - 1]
                flux_u = flux_l
        cflux[Nr - 1] += (flux_u - 0.0) / drF[Nr - 1] * recip_hFacC[Nr - 1]
    return cflux


# ---------------------------------------------------------------------------
# Fixture: 3 columns x 14 levels with a 3-D sea mask
# ---------------------------------------------------------------------------

_NLEV = 14
_DZ = np.array([10, 15, 25, 40, 60, 80, 100, 150, 200, 300, 400, 500, 700, 900.0])
# full-depth column, a column whose floor is at level 6, a 1-level shelf.
_N_WET = (14, 6, 1)


def _mask():
    return np.array([[1.0 if k < n else 0.0 for k in range(_NLEV)] for n in _N_WET])


def _tracers():
    rng = np.random.default_rng(1816)
    shape = (len(_N_WET), _NLEV)
    f = lambda lo, hi: jnp.asarray(rng.uniform(lo, hi, shape))  # noqa: E731
    return dict(
        NO3=f(2e-3, 30e-3), Phyto=f(0.05e-3, 1e-3), Zoo=f(0.02e-3, 0.5e-3),
        Det=f(0.01e-3, 0.3e-3), DIC=f(2.0, 2.2), ALK=f(2.25, 2.4),
        T_degC=f(2.0, 25.0), PAR=f(0.0, 150.0),
    )


def _tend(cfg, tr, mask):
    return npzd_source_sink(
        tr["NO3"], tr["Phyto"], tr["Zoo"], tr["Det"], tr["DIC"], tr["ALK"],
        tr["T_degC"], tr["PAR"], jnp.asarray(_DZ), cfg, jnp.asarray(mask),
    )


def _rel(resid, *terms):
    scale = sum(np.abs(np.asarray(t)) for t in terms)
    return np.abs(np.asarray(resid)) / np.sum(scale * _DZ, axis=-1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_column_budgets_close_x64():
    """C, N and ALK column integrals of the tendencies vanish (no gas flux)."""
    cfg = BiogeoConfig(scheme="npzd")
    tr, mask = _tracers(), _mask()
    dN, dP, dZ, dD, dC, dA = (np.asarray(t) for t in _tend(cfg, tr, mask))
    org = dP + dZ + dD
    c_res = np.sum((dC + cfg.R_CN * org) * _DZ, axis=-1)
    n_res = np.sum((dN + org) * _DZ, axis=-1)
    a_res = np.sum((dA + dN) * _DZ, axis=-1)
    assert np.all(_rel(c_res, dC, cfg.R_CN * org) < 1e-14), c_res
    assert np.all(_rel(n_res, dN, org) < 1e-14), n_res
    assert np.all(_rel(a_res, dA, dN) < 1e-14), a_res
    # Nothing happens below the sea floor.
    for t in (dN, dP, dZ, dD, dC, dA):
        assert np.all(t[mask == 0.0] == 0.0)


def test_compute_tendencies_3d_mask_closes_with_gas_flux():
    """With a 3-D mask the full tendency path closes up to the air-sea flux."""
    cfg = BiogeoConfig(scheme="npzd")
    tr, mask = _tracers(), _mask()
    state = OceanBiogeoState(DIC=tr["DIC"], ALK=tr["ALK"], NO3=tr["NO3"],
                             Phyto=tr["Phyto"], Zoo=tr["Zoo"], Det=tr["Det"])
    S = jnp.full(tr["DIC"].shape, 35.0)
    z_i = np.concatenate([[0.0], np.cumsum(_DZ)])
    z_full = -0.5 * (z_i[:-1] + z_i[1:])
    tend, diag = compute_biogeo_tendencies(
        state, tr["T_degC"], S, jnp.asarray(_DZ), jnp.asarray(z_full),
        jnp.asarray(mask), cfg)
    org = np.asarray(tend.dPhyto_dt + tend.dZoo_dt + tend.dDet_dt)
    dC = np.asarray(tend.dDIC_dt)
    c_res = np.sum((dC + cfg.R_CN * org) * _DZ, axis=-1) - np.asarray(diag.flux_co2)
    assert np.all(_rel(c_res, dC, cfg.R_CN * org) < 1e-14), c_res
    a_res = np.sum(np.asarray(tend.dALK_dt + tend.dNO3_dt) * _DZ, axis=-1)
    assert np.all(_rel(a_res, tend.dALK_dt, tend.dNO3_dt) < 1e-14), a_res


def test_production_levels_match_oracle_nlev():
    """CaCO3 forms in exactly the oracle's levels 1..nlev."""
    # 405 m is exactly a level centre: the oracle's >= keeps going past it.
    for zcrit in (5.0, 20.0, 100.0, 405.0, 500.0, 1e5):
        cfg = BiogeoConfig(scheme="npzd", caco3_zcrit=zcrit)
        tr = _tracers()
        mask = np.ones((len(_N_WET), _NLEV))
        # With zca huge almost nothing dissolves above the array bottom, so
        # the sign of the CaCO3 term (cflux - CAR_S) marks where it forms.
        cfg_l = cfg._replace(caco3_remin_depth=1e12)
        dN, _, _, _, dC, _ = (np.asarray(t) for t in _tend(cfg_l, tr, mask))
        car = dC - cfg_l.R_CN * dN
        # The array-bottom level always dissolves everything that reaches
        # it, so it is excluded (nlev == Nr is covered by the oracle test).
        n_ours = int(np.sum(car[0, :-1] < 0.0))
        assert n_ours == min(_oracle_nlev(_DZ, zcrit), _NLEV - 1), (zcrit, n_ours)


@pytest.mark.parametrize("zcrit", [100.0, 500.0, 1e5])
def test_caco3_terms_match_car_flux_oracle(zcrit):
    """DIC and ALK CaCO3 terms equal the oracle's cflux - CAR_S, 2*(...)."""
    cfg = BiogeoConfig(scheme="npzd", caco3_zcrit=zcrit)
    tr, mask = _tracers(), _mask()
    dN, _, _, _, dC, dA = (np.asarray(t) for t in _tend(cfg, tr, mask))
    P, Z = np.asarray(tr["Phyto"]), np.asarray(tr["Zoo"])
    nlev = _oracle_nlev(_DZ, zcrit)
    for c in range(len(_N_WET)):
        # CAR_S = rain_ratio * particulate organic C production; in the
        # NPZD that is detritus formation (phyto + zoo mortality) x R_CN.
        det_form = (cfg.m_P * P[c] + cfg.m_Z * Z[c] ** 2) / _DAY_S
        car_s = cfg.R_CaP * cfg.R_CN * det_form * mask[c]
        car_s[nlev:] = 0.0
        cflux = _car_flux_oracle(car_s, _DZ, mask[c], cfg.caco3_remin_depth, nlev)
        want = cflux - car_s
        got_c = dC[c] - cfg.R_CN * dN[c]
        got_a = dA[c] + dN[c]
        # Isolating the CaCO3 term subtracts the organic terms, which are
        # ~100x larger: allow their fp64 rounding (1e-14 of their size).
        atol = _CANCEL_RTOL * np.max(np.abs(cfg.R_CN * dN[c]))
        np.testing.assert_allclose(got_c, want, rtol=1e-12, atol=atol)
        np.testing.assert_allclose(got_a, 2.0 * want, rtol=1e-12, atol=2.0 * atol)


def test_dissolution_function_matches_oracle_any_profile():
    """The redistribution alone matches car_flux.F for arbitrary production."""
    rng = np.random.default_rng(7)
    prod = rng.uniform(0.0, 1e-9, (len(_N_WET), _NLEV)) * _mask()
    got = np.asarray(implicit_caco3_dissolution(
        jnp.asarray(prod), jnp.asarray(_DZ), jnp.asarray(_mask()), 3500.0))
    for c in range(len(_N_WET)):
        want = _car_flux_oracle(prod[c], _DZ, _mask()[c], 3500.0, _NLEV)
        np.testing.assert_allclose(got[c], want, rtol=1e-12, atol=1e-25)


def test_sign_convention():
    """Formation lowers DIC/ALK in the top layer; the floor cell gains; ALK=2xDIC."""
    cfg = BiogeoConfig(scheme="npzd")
    tr, mask = _tracers(), _mask()
    dN, _, _, _, dC, dA = (np.asarray(t) for t in _tend(cfg, tr, mask))
    car = dC - cfg.R_CN * dN
    assert np.all(car[:2, 0] < 0.0)  # surface: formation dominates
    assert car[0, 13] > 0.0 and car[1, 5] > 0.0  # bottom wet cells dissolve
    atol = _CANCEL_RTOL * np.max(np.abs(cfg.R_CN * dN))
    np.testing.assert_allclose(dA + dN, 2.0 * car, rtol=1e-12, atol=2.0 * atol)


def test_detritus_reaching_floor_is_remineralised_in_bottom_cell():
    """Detritus sinking onto the floor returns NO3 to the bottom wet cell."""
    cfg = BiogeoConfig(scheme="npzd")
    tr, mask = _tracers(), _mask()
    cfg0 = cfg._replace(w_sink=0.0)
    dN1 = np.asarray(_tend(cfg, tr, mask)[0])
    dN0 = np.asarray(_tend(cfg0, tr, mask)[0])
    floor = np.asarray(tr["Det"])[1, 5] * cfg.w_sink / _DAY_S / _DZ[5]
    np.testing.assert_allclose(dN1[1, 5] - dN0[1, 5], floor, rtol=1e-12)
    np.testing.assert_allclose(dN1[1, :5], dN0[1, :5], rtol=1e-12)


def test_gradients_check_grads_order2():
    cfg = BiogeoConfig(scheme="npzd")
    tr, mask = _tracers(), jnp.asarray(_mask())

    def f(P, Z, D):
        out = npzd_source_sink(tr["NO3"], P, Z, D, tr["DIC"], tr["ALK"],
                               tr["T_degC"], tr["PAR"], jnp.asarray(_DZ),
                               cfg, mask)
        return out[4] * 1e6, out[5] * 1e6  # DIC, ALK tendencies

    check_grads(f, (tr["Phyto"], tr["Zoo"], tr["Det"]), order=2,
                modes=["fwd", "rev"], eps=1e-7)
    g = jax.grad(lambda P: jnp.sum(f(P, tr["Zoo"], tr["Det"])[0]))(tr["Phyto"])
    assert np.all(np.isfinite(np.asarray(g))) and float(jnp.max(jnp.abs(g))) > 0.0


def test_non_contiguous_mask_is_cut_at_first_dry_cell():
    """A wet cell under a dry one is treated as below the floor; budgets close."""
    cfg = BiogeoConfig(scheme="npzd")
    tr = _tracers()
    mask = _mask()
    mask[0, 4] = 0.0  # hole in the full-depth column
    dN, dP, dZ, dD, dC, dA = (np.asarray(t) for t in _tend(cfg, tr, mask))
    for t in (dN, dP, dZ, dD, dC, dA):
        assert np.all(t[0, 4:] == 0.0)
    org = dP + dZ + dD
    c_res = np.sum((dC + cfg.R_CN * org) * _DZ, axis=-1)
    assert np.all(_rel(c_res, dC, cfg.R_CN * org) < 1e-14), c_res


def test_jit_parity_with_traced_geometry():
    """Production levels and the profile work with dz/mask as traced inputs."""
    cfg = BiogeoConfig(scheme="npzd")
    tr, mask = _tracers(), jnp.asarray(_mask())

    def f(dz, m):
        return npzd_source_sink(tr["NO3"], tr["Phyto"], tr["Zoo"], tr["Det"],
                                tr["DIC"], tr["ALK"], tr["T_degC"], tr["PAR"],
                                dz, cfg, m)

    eager = f(jnp.asarray(_DZ), mask)
    jitted = jax.jit(f)(jnp.asarray(_DZ), mask)
    for a, b in zip(eager, jitted):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-13, atol=1e-25)


def test_thin_layers_conserve_in_physical_thickness():
    """Sub-metre layers: budgets close with the physical dz (no 1 m floor)."""
    cfg = BiogeoConfig(scheme="npzd")
    dz = np.array([0.5, 0.8, 2.0, 999.0, 100.0])
    rng = np.random.default_rng(3)
    tr = {k: jnp.asarray(rng.uniform(lo, hi, (1, 5))) for k, lo, hi in (
        ("NO3", 2e-3, 3e-2), ("Phyto", 5e-5, 1e-3), ("Zoo", 2e-5, 5e-4),
        ("Det", 1e-5, 3e-4), ("DIC", 2.0, 2.2), ("ALK", 2.25, 2.4),
        ("T_degC", 2.0, 25.0), ("PAR", 0.0, 150.0))}
    out = npzd_source_sink(tr["NO3"], tr["Phyto"], tr["Zoo"], tr["Det"],
                           tr["DIC"], tr["ALK"], tr["T_degC"], tr["PAR"],
                           jnp.asarray(dz), cfg, jnp.ones((1, 5)))
    dN, dP, dZ, dD, dC, dA = (np.asarray(t)[0] for t in out)
    org = dP + dZ + dD
    for res, scale in ((np.sum((dN + org) * dz), np.sum((abs(dN) + abs(org)) * dz)),
                       (np.sum((dC + cfg.R_CN * org) * dz),
                        np.sum((abs(dC) + abs(cfg.R_CN * org)) * dz)),
                       (np.sum((dA + dN) * dz), np.sum((abs(dA) + abs(dN)) * dz))):
        assert abs(res) / scale < 1e-14, res
    # Oracle levels with the physical thickness (centres 0.25, 0.9, 2.3,
    # 502.8 -> levels 1..4 for zcrit=500).
    assert _oracle_nlev(dz, 500.0) == 4
    car = dC - cfg.R_CN * dN
    assert np.all(car[:4] < 0.0)
