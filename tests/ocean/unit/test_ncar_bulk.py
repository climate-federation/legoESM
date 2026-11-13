"""Unit tests for the NEMO-parity NCAR bulk algorithm (bulk_flux_omip).

Strategy
--------
1. An INDEPENDENT NumPy transcription of NEMO 5.0.1
   ``sbcblk_algo_ncar.F90``/``sbc_phy.F90`` (written directly from the
   Fortran, no JAX, no shared code) cross-checks the JAX implementation on a
   seeded random ocean-like input field — a transcription bug in either copy
   shows up as a mismatch.
2. Physical-limit checks: neutral reduction to the LY09 neutral coefficients,
   stability monotonicity, Goff curve / moist density / L_vap reference
   values, flux sign conventions, evap-lh consistency.
3. JAX health: jit + grad (differentiability through the fixed-point loop).
4. Legacy scheme pinning + dispatch hardening (unknown algo raises).
"""

from __future__ import annotations

import inspect
import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax.test_util import check_grads  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.ocean.bulk_flux_omip import (  # noqa: E402
    air_sea_fluxes,
    blk_oce_2,
    exner_potential_temperature,
    latent_heat_vaporization_sst as latent_heat_vaporization,
    moist_air_cp,
    nemo_blk_oce_ncar,
    ncar_transfer_coefficients,
    potential_air_temperature_10m,
    pressure_at_height,
    rho_air_moist,
    seawater_q_sat,
)
from legoesm.thermo import saturation_vapor_pressure_goff  # noqa: E402


def test_standalone_k0_config_requires_every_explicit_selector():
    """All six record-free K0 fields participate in fail-closed validation."""
    from legoesm.ocean.bulk_flux_omip import (
        NemoStandaloneK0Config,
        validate_nemo_standalone_k0,
    )

    cfg = NemoStandaloneK0Config(
        nn_ice=1, ln_drgice_imp=False, ln_mxl0=True, nn_mxlice=0,
        ln_icebergs=False, initial_state="nemo_k0_istate")
    validate_nemo_standalone_k0(cfg)
    plants = {
        "nn_ice": 0,
        "ln_drgice_imp": True,
        "ln_mxl0": False,
        "nn_mxlice": 2,
        "ln_icebergs": True,
        "initial_state": "si3_iceistate",
    }
    for field, value in plants.items():
        with pytest.raises(ValueError, match="Decision-175 K0 selectors"):
            validate_nemo_standalone_k0(cfg._replace(**{field: value}))

    from types import SimpleNamespace
    import legoesm.ocean.bulk_flux_omip as bfo
    from legoesm.ocean.fidelity import nemo_testcase_recipe as ntr

    seen_init = []

    def _fake_orca2_card(deck_root, *, initial_state="file_ts"):
        seen_init.append(initial_state)
        return SimpleNamespace(
            recipe=SimpleNamespace(
                model_config=SimpleNamespace(
                    implicit_ice_ocean_top_drag=False,
                    physics=SimpleNamespace(
                        vertical_mixing=SimpleNamespace(
                            tke=SimpleNamespace(
                                tke_mxl_choice=3,
                                nemo_mxl0_surface_tmask=True,
                                nemo_mxl0_rmxl_min_overwrite=True,
                                nemo_mxlice=0,
                            )
                        )
                    ),
                ),
                initial_state=None,
            ),
            icebergs_enabled=False,
        )

    real_builder = ntr.build_orca2_zps_card
    ntr.build_orca2_zps_card = _fake_orca2_card
    try:
        k0_card = bfo.build_nemo_orca2_k0_ocean_card("/unused")
    finally:
        ntr.build_orca2_zps_card = real_builder
    assert seen_init == ["nemo_k0_istate"]

    for bad_k0 in (
        cfg._replace(ln_drgice_imp=True),
        cfg._replace(ln_mxl0=False),
        cfg._replace(nn_mxlice=2),
        cfg._replace(ln_icebergs=True),
    ):
        with pytest.raises(ValueError, match="Decision-175 K0 selectors"):
            bfo._validate_k0_ocean_card(k0_card._replace(k0=bad_k0))

    bad_drg = _fake_orca2_card("/unused")
    bad_drg.recipe.model_config.implicit_ice_ocean_top_drag = True
    with pytest.raises(ValueError, match="standalone K0 ocean selectors"):
        bfo._validate_k0_ocean_card(k0_card._replace(ocean=bad_drg))

    bad_mxl_tmask = _fake_orca2_card("/unused")
    bad_mxl_tmask.recipe.model_config.physics.vertical_mixing.tke.nemo_mxl0_surface_tmask = False
    with pytest.raises(ValueError, match="standalone K0 ocean selectors"):
        bfo._validate_k0_ocean_card(k0_card._replace(ocean=bad_mxl_tmask))

    bad_mxl_rmxl = _fake_orca2_card("/unused")
    bad_mxl_rmxl.recipe.model_config.physics.vertical_mixing.tke.nemo_mxl0_rmxl_min_overwrite = False
    with pytest.raises(ValueError, match="standalone K0 ocean selectors"):
        bfo._validate_k0_ocean_card(k0_card._replace(ocean=bad_mxl_rmxl))

    bad_mxlice = _fake_orca2_card("/unused")
    bad_mxlice.recipe.model_config.physics.vertical_mixing.tke.nemo_mxlice = 2
    with pytest.raises(ValueError, match="standalone K0 ocean selectors"):
        bfo._validate_k0_ocean_card(k0_card._replace(ocean=bad_mxlice))

    bad_icb = _fake_orca2_card("/unused")
    bad_icb.icebergs_enabled = True
    with pytest.raises(ValueError, match="standalone K0 ocean selectors"):
        bfo._validate_k0_ocean_card(k0_card._replace(ocean=bad_icb))

    seen_nn_ice = []
    real_blk = bfo.nemo_blk_oce_ncar
    bfo.nemo_blk_oce_ncar = lambda *a, **kw: seen_nn_ice.append(kw.get("nn_ice"))
    try:
        bfo.nemo_orca2_k0_surface_card(
            *(0.0,) * 10, 0.066, 1.0, 1.0, 5,
            k0_config=cfg, ice_if=None, s0=None, s2=None)
    finally:
        bfo.nemo_blk_oce_ncar = real_blk
    assert seen_nn_ice == [1]


def test_k0_compiled_programs_trace_once_per_branch(monkeypatch):
    """The K0 surface program traces once per nn_fsbc branch and the fraqsr
    program once, however the per-step values change.  The physics they call
    is stubbed with trace-time counters; dropping either ``eqx.filter_jit``
    (running the uncompiled glue) calls the stub on every step and fails."""
    from typing import NamedTuple

    import legoesm.ocean.bulk_flux_omip as bfo
    import legoesm.ocean.eos as eos
    import legoesm.ocean.physics.shortwave_penetration as swp
    from legoesm.core.precision import get_policy

    traces = {"surface": 0, "fraqsr": 0}

    class K0(NamedTuple):
        nn_ice: int

    class Card(NamedTuple):
        albedo: float
        precip_factor: float
        evap_factor: float
        nb_iter: int
        k0: K0

    class Z(NamedTuple):
        nemo_e3t_0: jax.Array
        z_half_ref: jax.Array
        is_active: jax.Array

    class S0(NamedTuple):
        kt: jax.Array
        u_before: jax.Array
        v_before: jax.Array
        temperature_now: jax.Array
        salinity_now: jax.Array
        ssh_now: jax.Array
        e3t_reference: jax.Array
        r3t_now: jax.Array
        surface_mask: jax.Array

    class IceIf(NamedTuple):
        ts_surface: jax.Array
        clock: jax.Array

    class S2(NamedTuple):
        runoff_now: jax.Array
        sss_target: jax.Array
        surface_mask: jax.Array

    class Clock(NamedTuple):
        seconds: jax.Array

    class Surface(NamedTuple):
        ice_if: IceIf

    def s0f_stub(s0f):
        traces["surface"] += 1
        field = s0f.seconds * jnp.ones((3, 4))
        return (field,) * 9, field, field

    def fraqsr_stub(qsr, *_args):
        traces["fraqsr"] += 1
        return qsr

    monkeypatch.setattr(bfo, "nemo_orca2_k0_s0f", s0f_stub)
    monkeypatch.setattr(
        bfo, "_nemo_blk_oce_ncar", lambda *a, **kw: Surface(kw["ice_if"]))
    monkeypatch.setattr(eos, "nemo_r3t_stretch", lambda z, ssh, hb, **kw: 1.0 + ssh)
    monkeypatch.setattr(
        eos, "nemo_r3t_rk3_stage1_stretch", lambda z, a, b, hb, **kw: 1.0 + a + b)
    monkeypatch.setattr(swp, "nemo_fraqsr_1lev", fraqsr_stub)

    ny, nx, nz = 3, 4, 2
    z = Z(jnp.ones((ny, nx, nz)), jnp.asarray([0.0, -1.0, -2.0]),
          jnp.ones((ny, nx, nz)))
    card = Card(0.066, 1.0, 1.0, 5, K0(1))
    T = jnp.ones((ny, nx, nz))
    zeros2 = jnp.zeros((ny, nx))
    s0 = S0(*(jnp.zeros((ny, nx)) for _ in range(9)))
    s2 = S2(zeros2, zeros2, zeros2)
    prior = None
    for kt in range(1, 5):
        forcing_update = (kt - 1) % 2 == 0
        _, prior, _, _ = bfo._nemo_orca2_k0_surface_compiled(
            card, z, get_policy(), T + kt, T, jnp.ones((ny, nx + 1, nz)),
            jnp.ones((ny + 1, nx, nz)), zeros2 + kt, zeros2, zeros2 + 1.0,
            s0f=Clock(jnp.asarray(kt * 5400, dtype=jnp.int64)),
            ice_if=IceIf(zeros2, jnp.asarray(kt, dtype=jnp.int64)),
            s0=s0._replace(kt=jnp.asarray(kt, dtype=jnp.int64)), s2=s2,
            chl=zeros2, geothermal_heat_flux_mwm2=zeros2,
            prior_surface=None if forcing_update else prior,
            forcing_update=forcing_update,
        )
        bfo._nemo_orca2_k0_next_fraqsr_compiled(
            z, get_policy(), "stub", zeros2 + kt, zeros2 - kt, zeros2,
            zeros2 + 1.0, zeros2 + kt, zeros2)
    assert traces == {"surface": 2, "fraqsr": 1}


# ---------------------------------------------------------------------------
# Independent NumPy mirror of NEMO sbcblk_algo_ncar + sbc_phy preprocessing
# (zt = zu = 10 m), transcribed directly from the Fortran.  Deliberately
# uses NEMO's literal values (NOT legoesm.constants) so a wrong constant in
# the implementation cannot hide.
# ---------------------------------------------------------------------------

_VK = 0.4
_GRAV = 9.80665                       # const-ok: NEMO phycst grav, test mirror
_RCTV0 = 461.495 / 287.05 - 1.0       # const-ok: NEMO R_vap/R_dry, test mirror
_REPS0 = 287.05 / 461.495             # const-ok: NEMO R_dry/R_vap, test mirror
_RGAS = 8.314510                      # NEMO R_gas
_MDRY = 28.9647e-3                    # NEMO rmm_dryair
_MWAT = 18.0153e-3                    # NEMO rmm_water
_GAMMA_DRY = _RGAS / (_MDRY * 1005.0)  # NEMO rgamma_dry
_RT0 = 273.15                         # const-ok: NEMO rt0 in the test mirror


@pytest.mark.parametrize("backend", ["host", "scan"])
def test_legacy_blk_oce_2_preserves_parent_reduction_order(backend):
    """Host and scan backends keep the pre-refactor q_net word order.

    The deliberately flattened plant is algebraically identical but differs
    by ulps, so this is a live bitwise compatibility check rather than an
    allclose physics test.
    """
    rng = np.random.default_rng(335)
    shape = (37, 43)
    theta = rng.uniform(245.0, 310.0, shape)
    tsk_c = rng.uniform(-2.0, 32.0, shape)
    lw = rng.uniform(180.0, 480.0, shape)
    precip = rng.uniform(0.0, 2.0e-4, shape)
    snow = precip * rng.uniform(0.0, 1.0, shape)
    evap = rng.uniform(-2.0e-4, 2.0e-4, shape)
    sensible = rng.uniform(-400.0, 400.0, shape)
    latent = rng.uniform(-800.0, 800.0, shape)

    xp = np if backend == "host" else jnp
    theta = xp.asarray(theta)
    tsk_c = xp.asarray(tsk_c)
    lw = xp.asarray(lw)
    precip = xp.asarray(precip)
    snow = xp.asarray(snow)
    evap = xp.asarray(evap)
    sensible = xp.asarray(sensible)
    latent = xp.asarray(latent)
    theta_c = theta - constants.T_freeze
    tsk_K = tsk_c + constants.T_freeze
    qlwn = constants.emissivity_seawater_lw * (
        lw - constants.sigma_sb * tsk_K ** 4
    )
    liquid = xp.maximum(precip - snow, 0.0)
    q_precip_evap = (
        -snow * constants.L_fus_nemo
        - evap * constants.c_p_seawater * tsk_c
        + liquid * constants.c_p_seawater * theta_c
        + snow * constants.c_p_ice_nemo * xp.minimum(theta_c, 0.0)
    )
    parent_qns = sensible + latent + qlwn + q_precip_evap

    got = blk_oce_2(
        theta, tsk_c, lw, precip, snow, evap, sensible, latent,
        array_namespace=xp,
    )
    np.testing.assert_array_equal(np.asarray(got.qns), parent_qns)

    flattened_plant = (
        qlwn + sensible + latent
        - snow * constants.L_fus_nemo
        - evap * constants.c_p_seawater * tsk_c
        + liquid * constants.c_p_seawater * theta_c
        + snow * constants.c_p_ice_nemo * xp.minimum(theta_c, 0.0)
    )
    flattened_plant = np.asarray(flattened_plant)
    parent_qns = np.asarray(parent_qns)
    assert np.count_nonzero(
        flattened_plant.view(np.uint64) != parent_qns.view(np.uint64)
    ) > 0


def _np_e_sat_goff(T):
    """NEMO sbc_phy e_sat (Goff 1957), straight from the Fortran."""
    T = np.maximum(T, 180.0)
    rt = _RT0 / T
    return 100.0 * 10.0 ** (  # satcurve-ok: independent NEMO-Goff test mirror
        10.79574 * (1.0 - rt) - 5.028 * np.log10(T / _RT0)
        + 1.50475e-4 * (1.0 - 10.0 ** (-8.2969 * (T / _RT0 - 1.0)))
        + 0.42873e-3 * (10.0 ** (4.76955 * (1.0 - rt)) - 1.0) + 0.78614)


def _np_q_sat(T, p):
    e = _np_e_sat_goff(T)
    return _REPS0 * e / (p - (1.0 - _REPS0) * e)


def _np_pres_temp(q, slp, z, T_abs):
    """NEMO pres_temp (absolute-T branch, 3 iterations) — UNCLIPPED
    w = q/q_sat exactly like the Fortran, so it can falsify the
    implementation's garbage-cell guard on the physical domain."""
    T = np.maximum(T_abs, 180.0)
    p = np.asarray(slp, dtype=np.float64).copy()
    for _ in range(3):
        qs = _np_q_sat(T, p)
        w = q / qs
        xm = (1.0 - w) * _MDRY + w * _MWAT
        p = slp * np.exp(-_GRAV * xm * z / (_RGAS * T))
    return p


def _np_theta_exner(T, p):
    return T * (1.0e5 / p) ** _GAMMA_DRY


def _np_rho_air(T_abs, q, p):
    R_dry = 287.05  # const-ok: NEMO R_dry in the independent test mirror
    return np.maximum(p / (R_dry * T_abs * (1.0 + _RCTV0 * q)), 0.8)


def _np_cd_n10(w):
    # NEMO cd_n10_ncar (sbcblk_algo_ncar.F90:242-255) has no wind floor: the
    # caller floors (0.5 first guess, 0.25 iterated UN10).  A 0.5 floor here
    # mirrored legoESM's tier-3 BLK D1 defect instead of NEMO.
    cd = (2.7 / w + 0.142 + w / 13.09 - 3.14807e-10 * w ** 6) * 1e-3
    cd = np.where(w >= 33.0, 2.34e-3, cd)
    return np.maximum(cd, 0.1e-3)


def _np_psi_m(z):
    z = np.clip(z, -10.0, 10.0)
    x2 = np.sqrt(np.abs(1.0 - 16.0 * z))
    x2 = np.maximum(x2, 1.0)
    x = np.sqrt(x2)
    unst = (2.0 * np.log(0.5 * (1.0 + x)) + np.log(0.5 * (1.0 + x2))
            - 2.0 * np.arctan(x) + 0.5 * np.pi)
    stab = -5.0 * z
    return np.where(z > 0.0, stab, unst)


def _np_psi_h(z):
    z = np.clip(z, -10.0, 10.0)
    x2 = np.sqrt(np.abs(1.0 - 16.0 * z))
    x2 = np.maximum(x2, 1.0)
    unst = 2.0 * np.log(0.5 * (1.0 + x2))
    stab = -5.0 * z
    return np.where(z > 0.0, stab, unst)


def _np_turb_ncar(theta, q, sst, ssq, wind, nb_iter=5):
    """Direct Fortran transcription (vector NumPy), zt == zu == 10 m."""
    Ub = np.maximum(wind, 0.5)
    dTv = theta * (1.0 + _RCTV0 * q) - sst * (1.0 + _RCTV0 * ssq)
    stab = np.where(dTv > 0.0, 1.0, 0.0)
    CdN = _np_cd_n10(Ub)
    sqCdN = np.sqrt(CdN)
    Cd = CdN.copy()
    Ce = np.maximum(34.6e-3 * sqCdN, 0.1e-3)
    Ch = np.maximum(1e-3 * sqCdN * (18.0 * stab + 32.7 * (1.0 - stab)), 0.1e-3)
    sqCd = sqCdN.copy()
    dt_ = theta - sst
    dq_ = q - ssq
    for _ in range(nb_iter):
        us = sqCd * Ub
        ts = Ch / sqCd * dt_
        qs = Ce / sqCd * dq_
        zqa = 1.0 + _RCTV0 * q
        invL = (_GRAV * _VK * (ts * zqa + _RCTV0 * theta * qs)
                / np.maximum(us * us * theta * zqa, 1e-9))
        invL = np.clip(invL, -200.0, 200.0)
        zeta = np.clip(10.0 * invL, -10.0, 10.0)
        psim = _np_psi_m(zeta)
        UN10 = np.maximum(0.25, Ub * (1.0 + sqCd / _VK * psim))
        CdN = _np_cd_n10(UN10)
        sqCdN = np.sqrt(CdN)
        denom = 1.0 - sqCdN / _VK * psim
        Cd = np.maximum(CdN / (denom * denom), 0.1e-3)
        sqCd = np.sqrt(Cd)
        zh = -_np_psi_h(zeta) / _VK / sqCdN
        ratio = sqCd / sqCdN
        stab = np.where(zeta > 0.0, 1.0, 0.0)
        ChN = np.maximum(1e-3 * sqCdN * (18.0 * stab + 32.7 * (1.0 - stab)),
                         0.1e-3)
        CeN = np.maximum(34.6e-3 * sqCdN, 0.1e-3)
        Ch = np.maximum(ChN * ratio / (1.0 + ChN * zh), 0.1e-3)
        Ce = np.maximum(CeN * ratio / (1.0 + CeN * zh), 0.1e-3)
    return Cd, Ch, Ce, Ub


def _np_nemo_l_vap(theta_sst):
    """NEMO sbc_phy ``L_vap`` verbatim (empirical slope 2370 J/kg/K).  The model
    uses the Kirchhoff slope c_pw - c_pv = 2372 instead; the parity test below
    rescales the mirror's latent flux by the known ratio, and
    ``test_l_vap_deviation_from_nemo_is_bounded`` pins the deviation itself."""
    return (2.501 - 0.00237 * (theta_sst - _RT0)) * 1e6  # const-ok: NEMO L_vap mirror


def _np_fluxes(u, v, T_air, q_air, sst, slp):
    """Full NEMO open-water flux path in the NumPy mirror: preprocessing
    (pres_temp/theta_exner/ssq) -> turb_ncar -> BULK_FORMULA."""
    wind = np.sqrt(u * u + v * v + 1e-12)
    ssq = 0.98 * _np_q_sat(sst, slp)  # satcurve-ok: NEMO rdct_qsat_salt mirror
    p10 = _np_pres_temp(q_air, slp, 10.0, T_air)
    theta_air = _np_theta_exner(T_air, p10)
    theta_sst = _np_theta_exner(sst, slp)
    Cd, Ch, Ce, Ub = _np_turb_ncar(theta_air, q_air, theta_sst, ssq, wind)
    rho = _np_rho_air(T_air, q_air, p10)
    Urho = Ub * np.maximum(rho, 1.0)
    tau_x = -Urho * Cd * u
    tau_y = -Urho * Cd * v
    L_vap = _np_nemo_l_vap(theta_sst)
    cp_a = 1005.0 + 1860.0 * q_air
    zevap = Urho * Ce * (q_air - ssq)
    sh = Urho * Ch * (theta_air - theta_sst) * cp_a
    lh = L_vap * zevap
    return tau_x, tau_y, sh, lh, -zevap


def _random_marine_inputs(n=4096, seed=7):
    rng = np.random.default_rng(seed)
    sst = rng.uniform(271.3, 303.0, n)                  # K
    theta = sst + rng.uniform(-8.0, 8.0, n)             # K (potential, at 10 m)
    q = rng.uniform(1e-4, 0.022, n)                     # kg/kg
    slp = rng.uniform(95000.0, 104000.0, n)             # Pa
    # ssq from the INDEPENDENT mirror (not the implementation under test)
    ssq = 0.98 * _np_q_sat(sst, slp)  # satcurve-ok: NEMO mirror
    wind = rng.uniform(0.0, 38.0, n)                    # m/s incl. calm+cyclone
    return theta, q, sst, ssq, wind, slp


# ---------------------------------------------------------------------------
# 1. Cross-implementation equivalence
# ---------------------------------------------------------------------------

def test_ncar_coefficients_match_independent_numpy_mirror():
    theta, q, sst, ssq, wind, _slp = _random_marine_inputs()
    Cd_np, Ch_np, Ce_np, Ub_np = _np_turb_ncar(theta, q, sst, ssq, wind)
    # The NumPy mirror is NEMO's hard 33 m/s plateau and hard stable/unstable
    # Stanton step: compare at the hard limit (all three widths 0); the
    # default's gaps are pinned in the #2241 tests.
    Cd, Ch, Ce, Ub = ncar_transfer_coefficients(
        jnp.asarray(theta), jnp.asarray(q), jnp.asarray(sst),
        jnp.asarray(ssq), jnp.asarray(wind), cd_cutoff_width=0.0,
        dthv_width=0.0, neutral_width=0.0)
    assert np.allclose(np.asarray(Ub), Ub_np, rtol=0, atol=0)
    assert np.allclose(np.asarray(Cd), Cd_np, rtol=1e-12, atol=1e-15)
    assert np.allclose(np.asarray(Ch), Ch_np, rtol=1e-12, atol=1e-15)
    assert np.allclose(np.asarray(Ce), Ce_np, rtol=1e-12, atol=1e-15)


def test_full_flux_path_matches_independent_numpy_mirror():
    """END-TO-END parity: air_sea_fluxes(algo='ncar') vs the NumPy Fortran
    mirror INCLUDING NEMO's preprocessing (pres_temp 10-m pressure, Exner
    potential air temperature, Exner potential SST, mirror ssq/rho/L_vap).
    Catches a transcription bug anywhere in the chain (codex round-1
    finding: the earlier gamma_moist shortcut + absolute-SST mix)."""
    rng = np.random.default_rng(23)
    n = 2048
    sst = rng.uniform(271.5, 302.0, n)
    T_air = sst + rng.uniform(-6.0, 6.0, n)             # ABSOLUTE at 10 m
    q = rng.uniform(2e-4, 0.020, n)
    slp = rng.uniform(96000.0, 104000.0, n)
    # Keep q within w = q/q_sat <= 1.5 (up to 150% RH): the random draw can
    # otherwise pair cold air with tropical humidity (w ~ 9, unphysical).
    # Supersaturation parity has its own dedicated test (q = 1.3 and 7 q_sat).
    q = np.minimum(q, 1.5 * _np_q_sat(T_air, slp))
    u = rng.uniform(-20.0, 20.0, n)
    v = rng.uniform(-20.0, 20.0, n)
    ref = _np_fluxes(u, v, T_air, q, sst, slp)
    out = air_sea_fluxes(
        u10=jnp.asarray(u), v10=jnp.asarray(v), T_air_K=jnp.asarray(T_air),
        q_air=jnp.asarray(q), T_sfc_K=jnp.asarray(sst),
        slp_Pa=jnp.asarray(slp), dthv_width=0.0, neutral_width=0.0)
    names = ("tau_x", "tau_y", "sh", "lh", "evap")
    theta_sst = _np_theta_exner(sst, slp)
    l_ratio = np.asarray(latent_heat_vaporization(jnp.asarray(theta_sst))) / _np_nemo_l_vap(theta_sst)
    for nm, a, b in zip(names, out, ref):
        if nm == "lh":
            # The ONE intentional departure from the Fortran: Kirchhoff L_v(T)
            # in place of NEMO's empirical slope.  Exact parity after rescaling
            # by that known ratio; evap itself is untouched by the choice of L.
            b = b * l_ratio
        assert np.allclose(np.asarray(a), b, rtol=1e-10, atol=1e-12), nm


def test_ncar_coefficient_magnitudes_physical():
    """Coefficients stay in the physically-plausible bulk range."""
    theta, q, sst, ssq, wind, _slp = _random_marine_inputs(seed=11)
    Cd, Ch, Ce, _ = ncar_transfer_coefficients(
        jnp.asarray(theta), jnp.asarray(q), jnp.asarray(sst),
        jnp.asarray(ssq), jnp.asarray(wind))
    for C in (Cd, Ch, Ce):
        a = np.asarray(C)
        assert np.isfinite(a).all()
        assert (a >= 0.1e-3 - 1e-12).all()
        # Calm-wind CdN reaches 5.54e-3 (NEMO 2.7/U term) and strong
        # instability amplifies it by up to 1/(1-sqrt(CdN)/k psi_m)^2 ~ 3.6x
        # at zeta=-10 -> bound ~2e-2; anything above is a real blunder.
        assert (a <= 2.5e-2).all()


# ---------------------------------------------------------------------------
# 2. Physical limits
# ---------------------------------------------------------------------------

def test_near_neutral_reduces_to_neutral_coefficients():
    """theta -> SST, q -> ssq  =>  Cd -> CdN, Ce -> 34.6e-3 sqrt(CdN)."""
    from legoesm.core.bulk_flux import large_yeager_neutral_cd
    wind = jnp.asarray([2.0, 5.0, 8.0, 12.0, 20.0])
    sst = jnp.full(wind.shape, 288.0)
    ssq = seawater_q_sat(sst)
    # tiny UNSTABLE offset (exact zero sits on the Fortran SIGN() knife edge)
    theta = sst - 1e-7
    # NEMO's hard stable/unstable step (#2241 widths 0): the limit is one-sided.
    Cd, Ch, Ce, Ub = ncar_transfer_coefficients(theta, ssq, sst, ssq, wind,
                                                dthv_width=0.0, neutral_width=0.0)
    CdN = large_yeager_neutral_cd(Ub, nemo_parity=True)
    assert np.allclose(np.asarray(Cd), np.asarray(CdN), rtol=1e-5)
    assert np.allclose(np.asarray(Ce),
                       34.6e-3 * np.sqrt(np.asarray(CdN)), rtol=1e-5)
    # near-neutral from the unstable side -> 32.7 coefficient
    assert np.allclose(np.asarray(Ch),
                       32.7e-3 * np.sqrt(np.asarray(CdN)), rtol=1e-5)


def test_stability_monotonicity():
    """Stable boundary layer suppresses transfer; unstable enhances it."""
    wind = jnp.asarray(7.0)
    sst = jnp.asarray(288.0)
    ssq = seawater_q_sat(sst)
    Cd_u, Ch_u, Ce_u, _ = ncar_transfer_coefficients(
        sst - 4.0, ssq, sst, ssq, wind)      # unstable (air colder)
    Cd_n, Ch_n, Ce_n, _ = ncar_transfer_coefficients(
        sst - 1e-7, ssq, sst, ssq, wind)     # ~neutral
    Cd_s, Ch_s, Ce_s, _ = ncar_transfer_coefficients(
        sst + 4.0, ssq, sst, ssq, wind)      # stable (air warmer)
    assert float(Cd_u) > float(Cd_n) > float(Cd_s)
    assert float(Ce_u) > float(Ce_n) > float(Ce_s)
    assert float(Ch_u) > float(Ch_s)


def test_goff_curve_reference_values():
    """Goff (1957) curve: ~611 Pa at 0 degC, ~3537 Pa at 300 K (Goff values),
    and within 0.5% of the Tetens default over the marine range."""
    e0 = float(saturation_vapor_pressure_goff(jnp.asarray(constants.T_freeze)))
    assert abs(e0 - 611.0) < 2.5
    e300 = float(saturation_vapor_pressure_goff(jnp.asarray(300.0)))
    assert abs(e300 - 3537.0) < 35.0
    from legoesm.thermo import saturation_vapor_pressure
    T = jnp.linspace(271.0, 305.0, 35)
    rel = np.abs(np.asarray(saturation_vapor_pressure_goff(T))
                 / np.asarray(saturation_vapor_pressure(T)) - 1.0)
    assert rel.max() < 0.005


def test_seawater_q_sat_salt_factor_and_pressure():
    T = jnp.asarray(293.0)
    q_std = float(seawater_q_sat(T))
    e_s = float(saturation_vapor_pressure_goff(T))
    # NEMO reps0 = R_dry/R_vap with the NEMO gas constants (NOT
    # constants.epsilon, which uses legoESM's R_v and differs in the 5th
    # digit -- the implementation is NEMO-parity by design).
    q_pure = _REPS0 * e_s / (constants.p_atm_std - (1.0 - _REPS0) * e_s)
    assert q_std == pytest.approx(0.98 * q_pure, rel=1e-12)
    # lower pressure -> higher specific humidity at saturation
    assert float(seawater_q_sat(T, 95000.0)) > q_std


def test_rho_air_moist_reference_and_floor():
    rho_dry = float(rho_air_moist(288.15, 0.0, constants.p_atm_std))
    # rel 2e-6: the implementation floors q at 1e-6 like NEMO (rctv0*1e-6
    # ~ 6e-7 relative shift vs the exact dry-air value).
    assert rho_dry == pytest.approx(
        constants.p_atm_std / (constants.R_d * 288.15), rel=2e-6)
    # moisture lightens air
    assert float(rho_air_moist(288.15, 0.015, constants.p_atm_std)) < rho_dry
    # NEMO floor at 0.8
    assert float(rho_air_moist(400.0, 0.0, 50000.0)) >= 0.8


def test_l_vap_and_cp_air_reference():
    assert float(latent_heat_vaporization(constants.T_freeze)) == (
        pytest.approx(constants.L_v, rel=1e-12))
    assert float(latent_heat_vaporization(constants.T_freeze + 25.0)) == (
        pytest.approx(constants.L_v - (constants.c_pw - constants.c_pv) * 25.0, rel=1e-12))
    # Independent pin (codex round 2): a literal, not the constants the
    # implementation reads, so a drift in c_pw/c_pv cannot pass unnoticed.
    assert float(latent_heat_vaporization(constants.T_freeze + 25.0)) == (
        pytest.approx(2441700.0, rel=1e-12))   # const-ok: 2.501e6 - 2372*25 pinned
    assert float(moist_air_cp(0.0)) == pytest.approx(
        constants.c_p_dry_air_nemo, rel=1e-12)
    assert float(moist_air_cp(0.01)) == pytest.approx(
        constants.c_p_dry_air_nemo + 0.01 * constants.c_p_vapor_nemo,
        rel=1e-12)


def test_l_vap_deviation_from_nemo_is_bounded():
    """Kirchhoff (2372 J/kg/K) vs NEMO's empirical 2370: identical at 0 degC,
    within 3e-5 relative over the whole ocean SST range (user decision 2026-09-28,
    one latent-heat formula for the codebase)."""
    T = constants.T_freeze + np.array([0.0, 10.0, 25.0, 32.0])
    ours = np.asarray(latent_heat_vaporization(jnp.asarray(T)))
    nemo = _np_nemo_l_vap(T)
    assert ours[0] == pytest.approx(nemo[0], rel=1e-14)
    assert np.all(np.abs(ours / nemo - 1.0) < 3e-5)
    assert np.all(ours[1:] < nemo[1:])   # Kirchhoff is the slightly steeper slope, so smaller L when warm


_NO_ICE = {"nn_ice": 0, "ice_if": None}


def _literal_scalar_inputs():
    return (
        jnp.asarray(6.25), jnp.asarray(-2.5), jnp.asarray(286.4),
        jnp.asarray(0.0065), jnp.asarray(100800.0), jnp.asarray(14.2),
        jnp.asarray(0.1), jnp.asarray(-0.2), jnp.asarray(125.0),
        jnp.asarray(310.0), jnp.asarray(2.3e-5), jnp.asarray(3.0e-6),
        jnp.asarray(1.0), 0.066, 1.0, 1.0, 5,
    )


def test_literal_blk_oce_inexact_binary64_pins_and_eager_jit():
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    prior = get_policy()
    try:
        set_policy(PrecisionPolicy.fp64(transcendentals="libm"))
        eager = nemo_blk_oce_ncar(
            *_literal_scalar_inputs(), **_NO_ICE, s0=None, s2=None)
        inputs = _literal_scalar_inputs()
        compiled = jax.jit(
            lambda *values: nemo_blk_oce_ncar(
                *values, 5, **_NO_ICE, s0=None, s2=None)
        )(*inputs[:-1])
        assert all(np.array_equal(np.asarray(a).view(np.uint64),
                                  np.asarray(b).view(np.uint64))
                   for a, b in zip(eager, compiled) if a is not None)
        pins = {
            "theta_air": 4643735832083737147,
            "ssq": 4576833549615626132,
            "qns": 13863530862587990243,
            "utau": 4588492273966355432,
            "taum": 4589134070458585377,
        }
        for name, word in pins.items():
            assert int(np.asarray(getattr(eager, name)).view(np.uint64)) == word
    finally:
        set_policy(prior)


def test_literal_blk_oce_path_plants_and_physical_inputs_have_no_defaults():
    signature = inspect.signature(nemo_blk_oce_ncar)
    optional_composition = {
        "prior_surface", "chl", "geothermal_heat_flux_mwm2"}
    assert all(parameter.default is inspect.Parameter.empty
               for name, parameter in signature.parameters.items()
               if name not in optional_composition)
    base_args = list(_literal_scalar_inputs())
    base = nemo_blk_oce_ncar(*base_args, **_NO_ICE, s0=None, s2=None)
    longwave_args = list(base_args)
    longwave_args[9] = longwave_args[9] + 1.0
    longwave = nemo_blk_oce_ncar(
        *longwave_args, **_NO_ICE, s0=None, s2=None)
    assert float(longwave.qlwn - base.qlwn) == pytest.approx(
        constants.emissivity_seawater_lw, rel=0.0, abs=1e-14)
    assert np.asarray(longwave.qns).view(np.uint64) != np.asarray(base.qns).view(np.uint64)
    snow_args = list(base_args)
    snow_args[11] = snow_args[11] + 1.0e-6
    snow = nemo_blk_oce_ncar(*snow_args, **_NO_ICE, s0=None, s2=None)
    assert np.asarray(snow.qns).view(np.uint64) != np.asarray(base.qns).view(np.uint64)
    assert np.asarray(snow.emp).view(np.uint64) == np.asarray(base.emp).view(np.uint64)


def test_literal_blk_oce_finite_difference_gradient():
    args = _literal_scalar_inputs()
    objective = lambda sst: nemo_blk_oce_ncar(
        *args[:5], sst, *args[6:], **_NO_ICE, s0=None, s2=None
    ).qns
    value = args[5]
    tangent = float(jax.grad(objective)(value))
    step = 1.0e-4
    finite_difference = float(
        (objective(value + step) - objective(value - step)) / (2.0 * step)
    )
    assert np.isfinite(tangent)
    assert tangent == pytest.approx(finite_difference, rel=2e-5, abs=2e-5)


def test_literal_blk_oce_zero_wind_and_masked_fill_gradients_are_finite():
    """Inactive fills and exactly calm wind are safe for reverse-mode AD."""
    args = list(_literal_scalar_inputs())
    args[0] = jnp.asarray(0.0)
    args[1] = jnp.asarray(0.0)

    def wind_objective(u_wind):
        values = list(args)
        values[0] = u_wind
        result = nemo_blk_oce_ncar(
            *values, **_NO_ICE, s0=None, s2=None)
        return result.qns + result.utau + result.vtau + result.taum

    assert np.isfinite(float(jax.grad(wind_objective)(jnp.asarray(0.0))))

    def masked_temperature_objective(temperature):
        values = list(args)
        values[2] = temperature
        values[12] = jnp.asarray(0.0)
        result = nemo_blk_oce_ncar(
            *values, **_NO_ICE, s0=None, s2=None)
        return result.theta_air + result.qns

    def masked_qns_objective(temperature):
        values = list(args)
        values[2] = temperature
        values[12] = jnp.asarray(0.0)
        result = nemo_blk_oce_ncar(
            *values, **_NO_ICE, s0=None, s2=None)
        return result.qns

    value = masked_temperature_objective(jnp.asarray(250.0))
    tangent = jax.grad(masked_temperature_objective)(jnp.asarray(250.0))
    qns_tangent = jax.grad(masked_qns_objective)(jnp.asarray(250.0))
    assert np.isfinite(float(value))
    assert np.isfinite(float(tangent)) and float(tangent) != 0.0
    assert float(qns_tangent) == 0.0


def test_record_object_minmax_second_operand_wins_signed_zero_ties():
    from legoesm.ocean.bulk_flux_omip import _fmax, _fmin

    positive = jnp.asarray(0.0)
    negative = jnp.asarray(-0.0)
    for operation in (_fmax, _fmin):
        assert np.signbit(float(operation(positive, negative)))
        assert not np.signbit(float(operation(negative, positive)))


def test_record_object_sign_treats_negative_zero_as_positive():
    from legoesm.core.source_rounding import nemo_fortran_sign

    magnitude = jnp.asarray(2.0)
    assert float(nemo_fortran_sign(magnitude, jnp.asarray(-0.0))) == 2.0
    assert float(nemo_fortran_sign(magnitude, jnp.asarray(-1.0))) == -2.0


def test_pressure_at_height_supersaturated_matches_unclipped_nemo():
    """SUPERSATURATED air: the implementation matches NEMO's UNCLIPPED
    pres_temp (sbc_phy.F90:275).  q = 1.3 q_sat (RH 130%) and q = 7 q_sat
    (the largest q/q_sat on the tier-3 ORCA1 record, wet or land; the old
    w <= 2 clip departed from NEMO there, BLK D3)."""
    T, slp = 275.0, 101000.0
    q_sat0 = float(_np_q_sat(np.float64(T), np.float64(slp)))
    q_super = 1.3 * q_sat0
    p_impl = float(pressure_at_height(
        jnp.asarray(q_super), jnp.asarray(slp), 10.0, jnp.asarray(T)))
    p_ref = float(_np_pres_temp(np.float64(q_super), np.float64(slp), 10.0,
                                np.float64(T)))
    assert p_impl == pytest.approx(p_ref, rel=1e-13)
    q_record_max = 7.0 * q_sat0
    p_impl = float(pressure_at_height(
        jnp.asarray(q_record_max), jnp.asarray(slp), 10.0, jnp.asarray(T)))
    p_ref = float(_np_pres_temp(np.float64(q_record_max), np.float64(slp),
                                10.0, np.float64(T)))
    assert p_impl == pytest.approx(p_ref, rel=1e-13)


def test_pressure_at_height_and_exner_theta():
    """NEMO preprocessing reference values: the 10-m pressure sits ~120 Pa
    below slp; theta at 10 m exceeds the absolute T by ~0.09-0.11 K (the
    adiabatic 10-m lift); the potential SST at slp>1e5 is COLDER than the
    absolute SST (Exner reference 1e5 Pa)."""
    T, q, slp = 288.0, 0.008, 101325.0
    p10 = float(pressure_at_height(jnp.asarray(q), jnp.asarray(slp), 10.0,
                                   jnp.asarray(T)))
    assert 80.0 < slp - p10 < 160.0
    th_air = float(exner_potential_temperature(jnp.asarray(T),
                                               jnp.asarray(p10)))
    th_sst = float(exner_potential_temperature(jnp.asarray(T),
                                               jnp.asarray(slp)))
    # both referenced to 1e5 Pa: at slp > 1e5 the Exner factor < 1...
    assert th_sst < T
    # ...but the AIR-SEA pair (same Exner reference) keeps the ~+0.1 K
    # 10-m adiabatic separation that gamma*z used to approximate.
    assert 0.05 < th_air - th_sst < 0.16
    # cross-check vs the independent mirror
    assert p10 == pytest.approx(
        float(_np_pres_temp(np.float64(q), np.float64(slp), 10.0,
                            np.float64(T))), rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Flux-level checks
# ---------------------------------------------------------------------------

def test_fluxes_signs_and_consistency():
    tau_x, tau_y, sh, lh, evap = air_sea_fluxes(
        u10=jnp.asarray(8.0), v10=jnp.asarray(-3.0),
        T_air_K=jnp.asarray(285.0), q_air=jnp.asarray(0.006),
        T_sfc_K=jnp.asarray(291.0), slp_Pa=jnp.asarray(101000.0))
    # atmospheric convention: tau opposes the wind components
    assert float(tau_x) < 0.0 and float(tau_y) > 0.0
    # warm ocean under cold dry air: ocean loses sensible + latent heat
    assert float(sh) < 0.0 and float(lh) < 0.0 and float(evap) > 0.0
    # evap-lh consistency: NEMO evaluates L_vap at the POTENTIAL SST
    # (BULK_FORMULA pTs = zsspt), not the absolute SST.
    theta_sst = exner_potential_temperature(jnp.asarray(291.0),
                                            jnp.asarray(101000.0))
    L = float(latent_heat_vaporization(theta_sst))
    assert float(evap) == pytest.approx(-float(lh) / L, rel=1e-12)


def test_flux_magnitudes_tropical_reference():
    """Trade-wind regime: latent O(100 W/m2), sensible O(10 W/m2),
    stress O(0.05 Pa) -- guards against unit / coefficient blunders."""
    tau_x, _, sh, lh, evap = air_sea_fluxes(
        u10=jnp.asarray(7.0), v10=jnp.asarray(0.0),
        T_air_K=jnp.asarray(299.0), q_air=jnp.asarray(0.016),
        T_sfc_K=jnp.asarray(300.5), slp_Pa=jnp.asarray(101000.0))
    assert 0.02 < abs(float(tau_x)) < 0.15
    assert 40.0 < -float(lh) < 250.0
    assert 0.0 < -float(sh) < 40.0
    assert 1.5e-5 < float(evap) < 1.0e-4


def test_air_sea_fluxes_jit_and_grad():
    """The fixed-point loop is unrolled + jnp.where-branched: jit and grad
    must work and give finite gradients (AD safety for DA/tuning)."""
    def loss(sst):
        _, _, sh, lh, _ = air_sea_fluxes(
            u10=jnp.asarray(6.0), v10=jnp.asarray(2.0),
            T_air_K=jnp.asarray(287.5),  # const-ok: air temperature [K]
            q_air=jnp.asarray(0.008),
            T_sfc_K=sst, slp_Pa=jnp.asarray(101325.0))
        return sh + lh

    g = jax.grad(loss)(jnp.asarray(290.0))
    assert np.isfinite(float(g))
    # heat loss increases (more negative q) as the ocean warms
    assert float(g) < 0.0
    lj = jax.jit(loss)(jnp.asarray(290.0))
    assert np.isfinite(float(lj))
    assert float(lj) == pytest.approx(float(loss(jnp.asarray(290.0))),
                                      rel=1e-12)


# ---------------------------------------------------------------------------
# 4. Legacy scheme + dispatch hardening
# ---------------------------------------------------------------------------

def test_legacy_two_coeff_scheme_pinned():
    """algo='ly09_2coeff' reproduces the exact pre-NCAR formula."""
    u, v = 10.0, 0.0
    T_air, q_air, sst, q_sfc, rho = 290.0, 0.005, 295.0, 0.012, 1.2
    tau_x, tau_y, sh, lh, evap = air_sea_fluxes(
        u10=jnp.asarray(u), v10=jnp.asarray(v), T_air_K=jnp.asarray(T_air),
        q_air=jnp.asarray(q_air), T_sfc_K=jnp.asarray(sst),
        q_sfc=jnp.asarray(q_sfc), rho_air=jnp.asarray(rho),
        algo="ly09_2coeff")
    from legoesm.core.bulk_flux import large_yeager_neutral_cd
    w = np.sqrt(u * u + v * v + 1e-12)
    Cd = float(large_yeager_neutral_cd(jnp.asarray(w)))   # legacy clip bounds
    Ch = 1.46e-3  # unstable: sst > T_air
    assert float(tau_x) == pytest.approx(-rho * Cd * w * u, rel=1e-12)
    assert float(sh) == pytest.approx(
        rho * constants.c_pd * Ch * w * (T_air - sst), rel=1e-12)
    assert float(lh) == pytest.approx(
        rho * constants.L_v * Ch * w * (q_air - q_sfc), rel=1e-12)
    assert float(evap) == pytest.approx(-float(lh) / constants.L_v, rel=1e-12)


def test_unknown_algo_raises():
    with pytest.raises(ValueError, match="Unknown OMIP bulk algo"):
        air_sea_fluxes(
            u10=jnp.asarray(1.0), v10=jnp.asarray(0.0),
            T_air_K=jnp.asarray(280.0), q_air=jnp.asarray(0.005),
            T_sfc_K=jnp.asarray(281.0), algo="nope")


def test_legacy_requires_q_sfc():
    with pytest.raises(ValueError, match="requires an explicit"):
        air_sea_fluxes(
            u10=jnp.asarray(1.0), v10=jnp.asarray(0.0),
            T_air_K=jnp.asarray(280.0), q_air=jnp.asarray(0.005),
            T_sfc_K=jnp.asarray(281.0), algo="ly09_2coeff")


# --- #2241 RULE 5: the NEMO-parity 33 m/s neutral-drag plateau -------------
# The production OMIP path (air_sea_fluxes -> ncar_transfer_coefficients ->
# large_yeager_neutral_cd(nemo_parity=True)) blends the LY09 polynomial into
# the 2.34e-3 plateau over [33 - w, 33] m/s, w = cd_cutoff_width (default
# NCAR_CD_CUTOFF_WIDTH = 0.5); 0.0 is NEMO's hard switch.

def test_nemo_parity_cd_hard_limit_matches_nemo_source_exact():
    """Width 0 reproduces NEMO's cd_n10_ncar (the rounding-exact branch,
    itself pinned to NEMO) to round-off, eager and under jit, on both sides
    of U = 33 (the bit-exact claim belongs to nemo_source_exact only)."""
    from legoesm.core.bulk_flux import large_yeager_neutral_cd as cd
    U = jnp.asarray([0.25, 5.0, 30.0, 32.5, 32.75, 33.0 - 1e-12, 33.0, 35.0])
    ref = np.asarray(cd(U, nemo_parity=True, nemo_source_exact=True))
    hard = lambda u: cd(u, nemo_parity=True, cd_cutoff_width=0.0)  # noqa: E731
    for got in (hard(U), jax.jit(hard)(U)):
        np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-15, atol=0.0)


# Relative gap of the default vs the hard plateau, (Cd(0.5) - Cd(0)) / Cd(0),
# per neutral wind [m/s].  Measured 2026-11-07 (#2241); 0 outside the band.
_CD_GAP = {32.5: 0.0, 32.6: 6.452073249229509e-05, 32.75: 0.0004441758241635477,
           32.9: 0.0007216142895876101, 33.0: 0.0}


@pytest.mark.parametrize("U", sorted(_CD_GAP))
def test_nemo_parity_cd_default_gap_per_wind(U):
    from legoesm.core.bulk_flux import large_yeager_neutral_cd as cd
    u = jnp.asarray(U)
    hard = cd(u, nemo_parity=True, cd_cutoff_width=0.0)
    gap = float((cd(u, nemo_parity=True) - hard) / hard)
    np.testing.assert_allclose(gap, _CD_GAP[U], rtol=1e-6, atol=1e-12)


def test_check_grads_omip_drag_through_the_plateau_at_default():
    """Scheme level: Cd of ncar_transfer_coefficients vs wind in a neutral
    column (air = sea theta and q, so zeta = 0 and UN10 = wind in every
    iteration): the hard switch jumps at 33 m/s; the default is C2 there
    (order 1 at the join, as for the core helper), smooth mid-band."""
    def cd(u, width=None):
        return ncar_transfer_coefficients(
            jnp.asarray([290.0]), jnp.asarray([0.010]), jnp.asarray([290.0]),
            jnp.asarray([0.010]), jnp.atleast_1d(u), cd_cutoff_width=width)[0][0]
    check_grads(cd, (jnp.asarray(33.0),), order=1, modes=("fwd", "rev"), eps=1e-4)
    check_grads(cd, (jnp.asarray(32.75),), order=2, modes=("fwd", "rev"), eps=1e-4)
    assert float(jax.grad(cd)(jnp.asarray(32.75))) != 0.0
    with pytest.raises(AssertionError):          # the hard limit is not smooth
        check_grads(lambda u: cd(u, 0.0), (jnp.asarray(33.0),), order=1,
                    modes=("rev",), eps=1e-4)


def test_omip_cd_cutoff_width_reaches_the_drag():
    """The keyword is not inert: inside the band the two widths differ."""
    args = (jnp.asarray([32.75]), jnp.zeros(1), jnp.asarray([290.0]),
            jnp.asarray([0.010]), jnp.asarray([291.0]))
    assert float(air_sea_fluxes(*args)[0][0]) != float(
        air_sea_fluxes(*args, cd_cutoff_width=0.0)[0][0])


def test_every_large_yeager_drag_branch_shares_one_polynomial(monkeypatch):
    """#2242: the legacy (clipped) and NEMO-parity drag branches evaluate the
    same LY09 polynomial, so a change to it reaches both."""
    from legoesm.core import bulk_flux as bf
    U = jnp.array([5.0, 20.0])
    base_legacy = bf.large_yeager_neutral_cd(U)
    base_parity = bf.large_yeager_neutral_cd(U, nemo_parity=True)
    monkeypatch.setattr(bf, "_ly09_cdn_poly", lambda w: jnp.full_like(w, 1.5))
    np.testing.assert_allclose(bf.large_yeager_neutral_cd(U), 1.5e-3, rtol=1e-15)
    np.testing.assert_allclose(
        bf.large_yeager_neutral_cd(U, nemo_parity=True), 1.5e-3, rtol=1e-15)
    assert not np.allclose(base_legacy, 1.5e-3) and not np.allclose(base_parity, 1.5e-3)


# --- #2241 RULE 5: the stable/unstable neutral Stanton select ---------------
# ncar_transfer_coefficients blends the stable (18) and unstable (32.7)
# neutral Stanton numbers with core's ncar_stable_weight: half-width
# dthv_width [K] on the first-guess dTv, neutral_width on zeta in the
# iteration; 0.0 is NEMO's hard ``> 0`` step.  States: theta_air - SST in K
# below, q_air = q_sfc (so dTv ~ that difference), SST 290 K.
_STANTON_DTH = (-0.03, -0.008, -0.002, 0.0, 0.002, 0.008, 0.03)


def _stanton_args(U, dth=_STANTON_DTH):
    th = jnp.asarray([290.0 + d for d in dth])
    n = th.shape
    return (th, jnp.full(n, 0.010), jnp.full(n, 290.0), jnp.full(n, 0.010),
            jnp.full(n, U))


@pytest.mark.parametrize("U", (3.0, 8.0, 15.0))
def test_omip_stanton_hard_limit_matches_numpy_mirror(U):
    """Widths 0 reproduce the independent NumPy NEMO mirror on near-neutral
    states on both sides of the switch, eager and under jit."""
    args = _stanton_args(U)
    ref = _np_turb_ncar(*(np.asarray(a) for a in args))
    hard = lambda *a: ncar_transfer_coefficients(  # noqa: E731
        *a, dthv_width=0.0, neutral_width=0.0)
    for got in (hard(*args), jax.jit(hard)(*args)):
        for a, b in zip(got, ref):
            np.testing.assert_allclose(np.asarray(a), b, rtol=1e-12, atol=1e-15)


# Relative gap of the default vs the hard step, (X(default) - X(0)) / X(0),
# per state of _STANTON_DTH.  Measured 2026-11-12 (#2241); a change to the
# blend or its widths re-measures these (CLAUDE.md RULE 5).
_STANTON_GAP = {
    (3.0, 'Cd'): [0.0, 0.0, -2.3366027648952347e-06, 0.0, -2.770797659503093e-05, 0.0, 0.0],
    (3.0, 'Ch'): [0.0, 0.0, -0.005800862717823097, -0.2247706422018349, 0.0988916791323193, 0.0, 0.0],
    (3.0, 'Ce'): [0.0, 0.0, -4.244781781660209e-06, 0.0, -3.2117140483581586e-05, 0.0, 0.0],
    (8.0, 'Cd'): [0.0, -1.9180640298511565e-05, -1.7064710272036337e-05, 0.0, -2.1862356180547596e-05, -4.52598206322089e-05, 0.0],
    (8.0, 'Ch'): [0.0, -0.05118353832474979, -0.18138513554341992, -0.22477064220183487, 0.3372789370783426, 0.17467753279580744, 0.0],
    (8.0, 'Ce'): [0.0, -2.7433526968469796e-05, -2.4418183235609444e-05, 0.0, -2.1109609184077847e-05, -4.3702571816134735e-05, 0.0],
    (15.0, 'Cd'): [-3.35668016405842e-05, -2.0440278195622704e-05, -5.865839852003554e-06, 0.0, -7.340718352222034e-06, -2.599836026475332e-05, -5.950613098510515e-05],
    (15.0, 'Ch'): [-0.08271099527959236, -0.1882409494939306, -0.21590828040781762, -0.22477064220183499, 0.39257013656989764, 0.34756020090464396, 0.2121875520336049],
    (15.0, 'Ce'): [-3.971561625183429e-05, -2.4192601948759753e-05, -6.943256850536766e-06, 0.0, -6.179842313426353e-06, -2.1887155530135814e-05, -5.0097896816933336e-05],
}


@pytest.mark.parametrize("U,name", sorted(_STANTON_GAP))
def test_omip_stanton_default_gap_per_state(U, name):
    args = _stanton_args(U)
    i = ("Cd", "Ch", "Ce").index(name)
    d = np.asarray(ncar_transfer_coefficients(*args)[i])
    h = np.asarray(ncar_transfer_coefficients(
        *args, dthv_width=0.0, neutral_width=0.0)[i])
    np.testing.assert_allclose((d - h) / h, _STANTON_GAP[(U, name)],
                               rtol=1e-6, atol=1e-12)


def test_check_grads_omip_stanton_at_default():
    """Ch vs theta_air: the hard step jumps at the neutral point; the default
    is smooth (order 2, non-zero derivative) inside both blends.  The point
    is 3 mK on the stable side, off the psi_m / psi_h zeta = 0 kink that
    stays hard (as in the core TURB_NCAR port)."""
    def ch(th, dthv_width=None, neutral_width=None):
        a = _stanton_args(8.0, (0.0,))
        return ncar_transfer_coefficients(
            jnp.atleast_1d(th), *a[1:], dthv_width=dthv_width,
            neutral_width=neutral_width)[1][0]
    th0 = jnp.asarray(290.003)
    # both blends are live at th0 (each width alone moves Ch there)
    assert float(ch(th0)) != float(ch(th0, dthv_width=0.0))
    assert float(ch(th0)) != float(ch(th0, neutral_width=0.0))
    check_grads(ch, (th0,), order=2, modes=("fwd", "rev"), eps=1e-5)
    assert float(jax.grad(ch)(th0)) != 0.0
    with pytest.raises(AssertionError):          # the hard limit is not smooth
        check_grads(lambda t: ch(t, 0.0, 0.0), (jnp.asarray(290.0),), order=1,
                    modes=("rev",), eps=1e-4)


@pytest.mark.parametrize("U", (3.0, 8.0, 15.0))
def test_omip_stanton_default_is_continuous_at_neutral(U):
    """The hard step jumps Ch by ~1.8x across theta_air = SST; the default
    blend is continuous there (fails if the blend is reverted)."""
    def ch(dth, **w):
        return float(ncar_transfer_coefficients(*_stanton_args(U, (dth,)), **w)[1][0])
    hard = ch(1e-9, dthv_width=0.0, neutral_width=0.0) / ch(-1e-9, dthv_width=0.0, neutral_width=0.0)
    assert abs(hard - 18.0 / 32.7) < 1e-3
    np.testing.assert_allclose(ch(1e-9), ch(-1e-9), rtol=1e-5)


def test_omip_stanton_widths_reach_the_fluxes(monkeypatch):
    """air_sea_fluxes forwards both widths to the coefficient solve."""
    import legoesm.ocean.bulk_flux_omip as bfo
    seen = {}
    real = bfo.ncar_transfer_coefficients

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(bfo, "ncar_transfer_coefficients", spy)
    bfo.air_sea_fluxes(jnp.asarray([8.0]), jnp.zeros(1), jnp.asarray([290.0]),
                       jnp.asarray([0.008]), jnp.asarray([290.0]),
                       dthv_width=0.03, neutral_width=0.004)
    assert (seen["dthv_width"], seen["neutral_width"]) == (0.03, 0.004)
