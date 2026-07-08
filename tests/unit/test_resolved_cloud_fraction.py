"""CRM 'resolved' cloud-fraction scheme tests (iter-6 RAD-5).

At cloud-resolving resolution a grid cell is fully cloudy where it holds
condensate (SAM convention), unlike the Xu-Randall sub-grid fraction used
for coarse GCM grids. The ``resolved`` scheme returns ``cf ≈ 1`` wherever
explicit ``q_cloud``/``q_ice`` is present (smooth, AD-safe) and feeds the
resolved condensate to the RRTMGP cloud optics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.clouds.cloud_fraction import (
    compute_cloud_properties,
)
from legoesm.atmosphere.physics.clouds.config import CloudConfig


jax.config.update("jax_enable_x64", True)


def _column():
    # Radiation column convention: index 0 = model TOP (low p, cold),
    # index nlev-1 = surface (high p, warm) ⇒ pressure increases with
    # index so dp = p_half[1:] - p_half[:-1] > 0.
    nlev = 20
    T = jnp.linspace(220.0, 300.0, nlev)[None, :]
    p_full = jnp.linspace(1.0e4, 1.0e5, nlev)[None, :]
    p_half = jnp.linspace(9.0e3, 1.013e5, nlev + 1)[None, :]
    q_v = jnp.full((1, nlev), 5.0e-3)
    dp = p_half[:, 1:] - p_half[:, :-1]
    # Thick liquid cloud (levels 12-14), THIN ice cloud (levels 6-7) —
    # the thin cloud is where resolved cf ≈ 1 but Xu-Randall ≪ 1.
    q_c = jnp.zeros((1, nlev)).at[0, 12:15].set(4.0e-4)
    q_i = jnp.zeros((1, nlev)).at[0, 6:8].set(1.5e-5)
    return T, p_full, p_half, q_v, dp, q_c, q_i


def test_resolved_cf_unity_in_cloud_zero_clear():
    """cf ≈ 1 wherever condensate is present, exactly 0 in clear air."""
    T, p_full, _, q_v, dp, q_c, q_i = _column()
    cp = compute_cloud_properties(
        T=T, p_full=p_full, q_v=q_v, dp=dp,
        config=CloudConfig(scheme="resolved"), q_cloud=q_c, q_ice=q_i,
    )
    cf = np.asarray(cp.cloud_fraction)[0]
    assert float(np.min(cf[[6, 7, 12, 13, 14]])) > 0.9   # cloudy ⇒ cf≈1
    assert float(np.max(cf[[0, 1, 2, 17, 19]])) == 0.0   # clear ⇒ cf=0
    # LWP / IWP follow the explicit condensate (q·dp/g), strictly positive.
    assert float(np.asarray(cp.lwp).sum()) > 0.0
    assert float(np.asarray(cp.iwp).sum()) > 0.0


def test_resolved_requires_explicit_condensate():
    """The resolved scheme cannot diagnose condensate from RH — it must
    be handed explicit q_cloud/q_ice (else raise, no silent clear-sky)."""
    T, p_full, _, q_v, dp, _, _ = _column()
    with pytest.raises(ValueError, match="requires explicit"):
        compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp,
            config=CloudConfig(scheme="resolved"),
            q_cloud=None, q_ice=None,
        )


def test_resolved_cf_depends_only_on_condensate_not_rh():
    """CRM-faithful defining property: the resolved cloud fraction is a
    function of resolved CONDENSATE only — independent of RH — whereas
    Xu-Randall's sub-grid fraction varies with RH for the same
    condensate. Same q_c, two very different humidities ⇒ identical
    resolved cf but different Xu-Randall cf."""
    from legoesm.thermo import saturation_mixing_ratio
    T, p_full, _, _, dp, q_c, _ = _column()
    q_sat = saturation_mixing_ratio(T, p_full)
    q_v_dry = 0.30 * q_sat
    q_v_wet = 0.90 * q_sat
    zeros = jnp.zeros_like(q_c)

    def cf(scheme, q_v):
        return np.asarray(compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp,
            config=CloudConfig(scheme=scheme), q_cloud=q_c, q_ice=zeros,
        ).cloud_fraction)[0]

    # Resolved: identical for dry vs wet (RH-independent).
    assert np.allclose(cf("resolved", q_v_dry), cf("resolved", q_v_wet))
    # Xu-Randall: RH-dependent ⇒ the two differ in the cloudy layer.
    cloud_lev = 13
    assert cf("xu_randall", q_v_dry)[cloud_lev] != pytest.approx(
        cf("xu_randall", q_v_wet)[cloud_lev]
    )


def test_resolved_uses_sam_liquid_r_eff():
    """The driver passes SAM's 14 µm ocean liquid r_eff (RAD-1)."""
    T, p_full, _, q_v, dp, q_c, q_i = _column()
    cp = compute_cloud_properties(
        T=T, p_full=p_full, q_v=q_v, dp=dp,
        config=CloudConfig(scheme="resolved", r_eff_liq=14.0e-6),
        q_cloud=q_c, q_ice=q_i,
    )
    assert float(np.asarray(cp.r_eff_liq)[0, 0]) == pytest.approx(14.0e-6)


def test_resolved_cf_ad_safe():
    """cf = q_cond/(q_cond+ref) is smooth — grad wrt condensate finite
    everywhere incl. clear air (q_cond=0)."""
    T, p_full, _, q_v, dp, _, _ = _column()

    def lwp_cf_sum(q_c):
        cp = compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp,
            config=CloudConfig(scheme="resolved"),
            q_cloud=q_c, q_ice=jnp.zeros_like(q_c),
        )
        return jnp.sum(cp.cloud_fraction) + jnp.sum(cp.lwp)

    g = jax.grad(lwp_cf_sum)(jnp.zeros_like(q_v))   # clear-air gradient
    assert bool(jnp.all(jnp.isfinite(g)))


def test_rrtmgp_resolved_clouds_change_heating_vs_clear():
    """End-to-end guard (codex iter-6): with ``include_clouds=True`` and the
    resolved scheme, RRTMGP cloud optics MUST engage — the cloudy heating
    rate differs materially from clear-sky (catches a silent clear-sky
    fallback if the bundled cloud-optics tables are missing)."""
    from legoesm.atmosphere.physics.radiation.config import (
        RadiationConfig, RRTMGPConfig,
    )
    from legoesm.atmosphere.physics.radiation.integration import (
        _call_radiation_backend,
    )
    nlev = 20
    T = jnp.linspace(230.0, 300.0, nlev)[None, :]
    p_full = jnp.linspace(2.0e4, 1.0e5, nlev)[None, :]
    p_half = jnp.linspace(1.5e4, 1.013e5, nlev + 1)[None, :]
    q_v = jnp.full((1, nlev), 5.0e-3)
    q_c = jnp.zeros((1, nlev)).at[0, 12:15].set(5.0e-4)   # liquid cloud deck

    def heating(clouds):
        cfg = RadiationConfig(
            scheme="rrtmgp",
            rrtmgp=RRTMGPConfig(include_clouds=clouds),
            cloud_scheme="resolved" if clouds else "none",
            cloud_config=CloudConfig(scheme="resolved", r_eff_liq=14.0e-6),
        )
        out = _call_radiation_backend(
            radiation_config=cfg, T=T, p_full=p_full, p_half=p_half,
            sfc_temperature=jnp.array([300.0]), lat=jnp.array([0.0]),
            q_v=q_v, insolation=jnp.array([425.0]),
            cos_sza=jnp.array([0.62]),
            q_cloud=q_c if clouds else None, q_ice=None,
            rrtmgp_solver=None,
        )
        return np.asarray(out.heating_rate)[0]

    hr_clear = heating(False)
    hr_cloud = heating(True)
    assert np.all(np.isfinite(hr_clear)) and np.all(np.isfinite(hr_cloud))
    # Cloud optics engaged ⇒ heating changes by ≫ numerical noise.
    assert float(np.max(np.abs(hr_cloud - hr_clear)) * 86400.0) > 1.0  # K/day


def test_rrtmgp_clear_sky_toa_cre_sign():
    """#843 clear-sky diagnostic: the clear-sky pass (``cloud_scheme='none'``,
    no condensate) is exactly what ``compute_radiation_core`` runs for its
    second RRTMGP call, and its TOA up-fluxes become ``rsutcs``/``rlutcs``.
    Assert the physical cloud-radiative-effect signs those diagnostics must
    satisfy against the all-sky pass, so ``rsut``/``rsutcs`` can never be
    swapped or nulled silently:

      * SW: a reflective liquid deck raises reflected SW at TOA ⇒
        ``rsut`` (all-sky) > ``rsutcs`` (clear-sky), i.e. SW_CRE > 0.
      * LW: the same mid-tropospheric deck emits at a colder-than-surface
        temperature ⇒ it traps OLR ⇒ ``rlut`` (all-sky) < ``rlutcs``
        (clear-sky), i.e. LW_CRE > 0.

    ``compute_radiation_core`` reads TOA up-flux at level index 0
    (``sw_flux_up[:, 0]`` / ``lw_flux_up[:, 0]``), matched here.
    """
    from legoesm.atmosphere.physics.radiation.config import (
        RadiationConfig, RRTMGPConfig,
    )
    from legoesm.atmosphere.physics.radiation.integration import (
        _call_radiation_backend,
    )
    nlev = 20
    # index 0 = TOA (low p, cold), index nlev-1 = surface (high p, warm).
    T = jnp.linspace(230.0, 300.0, nlev)[None, :]
    p_full = jnp.linspace(2.0e4, 1.0e5, nlev)[None, :]
    p_half = jnp.linspace(1.5e4, 1.013e5, nlev + 1)[None, :]
    q_v = jnp.full((1, nlev), 5.0e-3)
    # Mid-tropospheric liquid cloud deck (~700 hPa, T≈275 K < T_sfc=300 K):
    # reflective in SW, colder-than-surface emitter in LW ⇒ both CRE > 0.
    q_c = jnp.zeros((1, nlev)).at[0, 12:15].set(5.0e-4)

    def toa_up(clouds):
        cfg = RadiationConfig(
            scheme="rrtmgp",
            rrtmgp=RRTMGPConfig(include_clouds=clouds),
            cloud_scheme="resolved" if clouds else "none",
            cloud_config=CloudConfig(scheme="resolved", r_eff_liq=14.0e-6),
        )
        out = _call_radiation_backend(
            radiation_config=cfg, T=T, p_full=p_full, p_half=p_half,
            sfc_temperature=jnp.array([300.0]), lat=jnp.array([0.0]),
            q_v=q_v, insolation=jnp.array([425.0]),
            cos_sza=jnp.array([0.62]),
            q_cloud=q_c if clouds else None, q_ice=None,
            rrtmgp_solver=None,
        )
        # TOA (level index 0): SW-up (rsut/rsutcs), LW-up (rlut/rlutcs).
        return (float(np.asarray(out.sw_flux_up)[0, 0]),
                float(np.asarray(out.lw_flux_up)[0, 0]))

    rsutcs, rlutcs = toa_up(False)   # clear-sky pass (the #843 2nd call)
    rsut, rlut = toa_up(True)        # all-sky pass
    for v in (rsutcs, rlutcs, rsut, rlut):
        assert np.isfinite(v)
    # SW cloud-radiative effect > 0: clouds reflect ⇒ all-sky SW-up higher.
    assert rsut > rsutcs + 1.0, f"SW_CRE={rsut - rsutcs} not > 0 (W/m^2)"
    # LW cloud-radiative effect > 0: clouds trap OLR ⇒ all-sky LW-up lower.
    assert rlutcs > rlut + 1.0, f"LW_CRE={rlutcs - rlut} not > 0 (W/m^2)"


def test_psd_ice_effective_radius_m2005():
    """RAD-1-ice: with explicit ``n_ice``, r_eff_ice = 1.5/LAMI (SAM M2005
    EFFI), LAMI=(ρ_ci·π·N_i/q_i)^(1/3) ⇒ scales q_i^⅓·N_i^−⅓ [m]; 25 µm
    where no ice. No explicit clamp here (the RRTMGP cloud-optics clamps to
    its ice table). Constant config value when n_ice is absent."""
    nlev = 6
    T = jnp.full((1, nlev), 240.0)
    p = jnp.full((1, nlev), 3.0e4)
    ph = jnp.linspace(2.5e4, 3.5e4, nlev + 1)[None, :]
    qv = jnp.full((1, nlev), 1.0e-3)
    dp = ph[:, 1:] - ph[:, :-1]
    cfg = CloudConfig(scheme="resolved", rho_cloud_ice=500.0)
    q_i = jnp.array([[0.0, 1.0e-5, 8.0e-5, 1.0e-5, 1.0e-5, 1.0e-4]])
    n_i = jnp.array([[1.0e5, 1.0e5, 1.0e5, 8.0e5, 1.0e6, 1.0e5]])
    re = np.asarray(compute_cloud_properties(
        T=T, p_full=p, q_v=qv, dp=dp, config=cfg,
        q_cloud=jnp.zeros((1, nlev)), q_ice=q_i, n_ice=n_i,
    ).r_eff_ice)[0]
    assert re[0] == pytest.approx(25.0e-6)               # no ice ⇒ 25 µm
    assert re[2] / re[1] == pytest.approx(8.0 ** (1.0 / 3.0), rel=1e-3)  # q_i^⅓
    assert re[3] / re[1] == pytest.approx(8.0 ** (-1.0 / 3.0), rel=1e-3)  # N_i^−⅓
    # Absolute value = 1.5/LAMI: q_i=1e-5,N_i=1e5 ⇒ ~60 µm.
    cons12 = 500.0 * np.pi
    lami_ref = (cons12 * 1.0e5 / 1.0e-5) ** (1.0 / 3.0)
    assert float(re[1]) == pytest.approx(1.5 / lami_ref, rel=1e-6)

    # Without n_ice ⇒ the constant config value (backward compatible).
    re_const = np.asarray(compute_cloud_properties(
        T=T, p_full=p, q_v=qv, dp=dp,
        config=CloudConfig(scheme="resolved", r_eff_ice=30.0e-6),
        q_cloud=jnp.zeros((1, nlev)), q_ice=q_i,
    ).r_eff_ice)
    assert np.allclose(re_const, 30.0e-6)


def test_psd_ice_radius_ad_safe():
    """jax.grad of the PSD ice radius wrt q_i stays finite incl. q_i=0."""
    nlev = 4
    T = jnp.full((1, nlev), 240.0)
    p = jnp.full((1, nlev), 3.0e4)
    ph = jnp.linspace(2.5e4, 3.5e4, nlev + 1)[None, :]
    qv = jnp.full((1, nlev), 1.0e-3)
    dp = ph[:, 1:] - ph[:, :-1]
    n_i = jnp.full((1, nlev), 1.0e5)

    def loss(q_i):
        return jnp.sum(compute_cloud_properties(
            T=T, p_full=p, q_v=qv, dp=dp,
            config=CloudConfig(scheme="resolved"),
            q_cloud=jnp.zeros((1, nlev)), q_ice=q_i, n_ice=n_i,
        ).r_eff_ice)

    g = jax.grad(loss)(jnp.array([[0.0, 1.0e-5, 5.0e-5, 1.0e-4]]))
    assert bool(jnp.all(jnp.isfinite(g)))


def test_unknown_cloud_scheme_raises():
    """Typo'd scheme must raise (no silent clear-sky dispatch)."""
    T, p_full, _, q_v, dp, q_c, q_i = _column()
    with pytest.raises(ValueError, match="Unknown cloud scheme"):
        compute_cloud_properties(
            T=T, p_full=p_full, q_v=q_v, dp=dp,
            config=CloudConfig(scheme="resolvd"),  # typo
            q_cloud=q_c, q_ice=q_i,
        )


# --- Coarse-GCM cloud-radiation condensate floor (planetary-albedo fix) -------
# With microphysics active, compute_cloud_properties receives the PROGNOSTIC
# grid-mean q_c/q_i, which is ~0 in a coarse model (grid-mean rarely saturates)
# even where the diagnostic cloud fraction cf>0 — giving optically INERT clouds
# and a clear-sky planetary albedo (~12% vs Earth ~30%).  The floor restores a
# radiatively-active in-cloud condensate (cf*q_c_diagnostic) for the
# diagnostic-fraction schemes, BOTH liquid and ice, without touching the
# water/energy budget; 'resolved' (CRM) is excluded.

def test_cloud_radiation_floors_both_liquid_and_ice():
    """The floor wires BOTH phases into cloud radiation: with prognostic q_c/q_i
    ~= 0, sundqvist clouds carry radiative condensate floored to cf*q_c_diagnostic,
    temperature-partitioned into LIQUID (warm cloudy layers) and ICE (cold cloudy
    layers) — neither phase optically inert (user req: liquid + ice both wired)."""
    from legoesm.thermo import saturation_mixing_ratio
    nlev = 20
    T = jnp.linspace(240.0, 300.0, nlev)[None, :]        # spans ice + liquid
    p_full = jnp.linspace(1.0e4, 1.0e5, nlev)[None, :]
    p_half = jnp.linspace(9.0e3, 1.013e5, nlev + 1)[None, :]
    dp = p_half[:, 1:] - p_half[:, :-1]
    q_v = 0.9 * saturation_mixing_ratio(T, p_full)       # RH=0.9 > rh_crit ⇒ cf>0
    zeros = jnp.zeros_like(T)
    cp = compute_cloud_properties(
        T=T, p_full=p_full, q_v=q_v, dp=dp,
        config=CloudConfig(scheme="sundqvist"),
        q_cloud=zeros, q_ice=zeros,                      # microphysics drained both
    )
    lwp = np.asarray(cp.lwp); iwp = np.asarray(cp.iwp)
    assert float(lwp.sum()) > 0.0, "liquid condensate not wired into cloud radiation"
    assert float(iwp.sum()) > 0.0, "ice condensate not wired into cloud radiation"
    # The kwargs handed to RRTMGP must carry BOTH paths.
    kw = cp.to_rrtmg_kwargs()
    assert float(np.asarray(kw["cloud_path_liq"]).sum()) > 0.0
    assert float(np.asarray(kw["cloud_path_ice"]).sum()) > 0.0


def test_floor_not_applied_to_resolved_scheme():
    """The floor is gated to diagnostic-fraction schemes; 'resolved' (CRM, q_c IS
    the truth) is NOT floored — zero explicit condensate ⇒ zero LWP/IWP."""
    T, p_full, _, q_v, dp, _, _ = _column()
    zeros = jnp.zeros_like(T)
    cp = compute_cloud_properties(
        T=T, p_full=p_full, q_v=q_v, dp=dp,
        config=CloudConfig(scheme="resolved"), q_cloud=zeros, q_ice=zeros,
    )
    assert float(np.asarray(cp.lwp).sum()) == 0.0
    assert float(np.asarray(cp.iwp).sum()) == 0.0


def test_floor_preserves_large_prognostic_condensate():
    """jnp.maximum: where the prognostic condensate exceeds the diagnostic floor
    it is preserved (no double-count). A thick prognostic cloud gives MORE LWP
    than the floor-only (zero-q_c) case at the same level."""
    T, p_full, _, q_v, dp, q_c, _ = _column()      # q_c=4e-4 at lev 12-14
    zeros = jnp.zeros_like(T)
    cfg = CloudConfig(scheme="sundqvist")
    lwp_big = np.asarray(compute_cloud_properties(
        T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg,
        q_cloud=q_c, q_ice=zeros).lwp)[0]
    lwp_floor = np.asarray(compute_cloud_properties(
        T=T, p_full=p_full, q_v=q_v, dp=dp, config=cfg,
        q_cloud=zeros, q_ice=zeros).lwp)[0]
    # Thick prognostic cloud (4e-4) exceeds the cf*q_c_diagnostic floor (<=2e-4).
    assert lwp_big[13] > lwp_floor[13]
