"""Faithfulness canaries for the E3SM/CAM gravity-wave drag scheme.

Pins the audited faithfulness scope and the honest conservation declaration of
``e3sm_cam.py`` (see its ``__physics_contract__`` energy note and the
"Faithfulness scope & departures" docstring section). These are non-behavioral
guards: each asserts a documented fact so a silent regression — a re-inflated
``conserves`` claim, a flipped default heating frame, a lost energy fixer —
fails loudly.

``conserves == ["none"]`` is the static INTERSECTION over all selectable
sources; it does NOT mean the default runtime path fails to conserve:

* **Orographic** (c=0, the DEFAULT source) IS energy-conserving in-atmosphere:
  a stationary mountain exchanges momentum without mechanical work, so the code
  returns the mean-flow KE it removes as heat BY CONSTRUCTION
  (``c_pd*sum(rho*dT*dz) == eps_gwd`` definitionally). Momentum is a surface
  sink (mountain drag), never conserved.
* **Frontal / convective** launch NONSTATIONARY spectral components from an
  EXTERNAL, unbudgeted reservoir (so momentum is not conserved), and their
  default thermal term is the ground-relative wave-energy-flux-divergence term
  ``sum_l c_l*gwut_l``, which is NOT the mean-flow KE tendency
  ``sum_l ubm*gwut_l`` (whose negated column integral is ``eps_gwd``), nor the
  irreversible ``(c-ubm)*gwut`` conversion. So the orographic KE->heat closure
  does not extend to spectra, and a static ["energy"] claim is wrong.

The irreversible intrinsic heating frame remains opt-in
(``dttke_use_intrinsic``); since 2026-07-17 the C.-C. Chen fixer, the E3SM
spectral thermal term (dttdf + band), and the discrete-KE orographic closure
default ON (the flip campaign) — legacy arms are pinned explicitly below.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    E3SMCAMConfig,
    E3SMFrontalConfig,
    E3SMOrographicConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.e3sm_cam import (
    __physics_contract__,
    e3sm_cam_gwd,
)

from legoesm import constants as C


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _driver_column(ncol=3, nlev=40):
    """A hydrostatic westerly-jet column (same construction as the oracle column
    in test_gwd_e3sm_cam.py; replicated to keep this canary file standalone).

    Reads the live ``legoesm.constants`` R_d/g so rho and dz are self-consistent
    with the driver (which uses ``constants.R_d``/``constants.g``), making the
    orographic KE->heat identity exact.
    """
    g = float(C.g)
    rair = float(C.R_d)
    p_top, p_surf = 100.0, 1.0e5
    pint = np.linspace(p_top, p_surf, nlev + 1)
    pmid = 0.5 * (pint[:-1] + pint[1:])
    T = np.linspace(230.0, 290.0, nlev)
    dp = np.diff(pint)
    dz = rair * T * dp / (g * pmid)
    z_half = np.zeros(nlev + 1)
    for k in range(nlev, 0, -1):
        z_half[k - 1] = z_half[k] + dz[k - 1]
    zm = 0.5 * (z_half[:-1] + z_half[1:])
    u = np.linspace(5.0, 30.0, nlev)

    def rep(a):
        return jnp.broadcast_to(jnp.asarray(a)[None, :], (ncol, len(a)))

    pmid_c = rep(pmid)
    pint_c = jnp.broadcast_to(jnp.asarray(pint)[None, :], (ncol, nlev + 1))
    T_c = rep(T)
    zf_c = rep(zm)
    zh_c = jnp.broadcast_to(jnp.asarray(z_half)[None, :], (ncol, nlev + 1))
    rho_c = pmid_c / (rair * T_c)
    u_c = rep(u)
    v_c = jnp.zeros((ncol, nlev))
    lat = jnp.zeros(ncol)
    return u_c, v_c, T_c, pmid_c, pint_c, zf_c, zh_c, rho_c, lat


def _column_energy_imbalance(out, rho, z_half):
    """``c_pd*sum(rho*dT_dt*dz) - eps_gwd`` per column, i.e. (column thermal
    deposition rate) - (mean-flow KE removal rate).

    ~0 for a self-energy-conserving path (the thermal deposition == the KE
    removal rate ``eps_gwd``); a non-trivial residual means the deposited
    thermal quantity differs from the mean-flow KE removal (the ground-relative
    wave-energy-flux term vs the KE removal rate).
    """
    dz = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    heat = float(C.c_pd) * jnp.sum(rho * out.dT_dt * dz, axis=1)
    return heat - out.eps_gwd


# --------------------------------------------------------------------------
# 1. The honest conservation declaration.
# --------------------------------------------------------------------------

def test_conserves_is_none():
    """The scheme supports frontal/convective launched-wave sources, so a
    static energy claim is false; ``conserves`` must be the audited ["none"].
    Re-inflating it to ["energy"]/["momentum"] fails here."""
    assert __physics_contract__["conserves"] == ["none"]


# --------------------------------------------------------------------------
# 2. Orographic (c=0, DEFAULT source) IS energy-conserving in-atmosphere.
# --------------------------------------------------------------------------

def test_orographic_in_atmosphere_energy_closure():
    """Orographic waves are stationary (c=0): they carry no energy flux, so the
    mean-flow KE removed by the drag is returned as heat. This is the ONE source
    for which ``c_pd*sum(rho*dT*dz) == eps_gwd``, and it is the reason the
    orographic path alone is energy-conserving in-atmosphere.

    NOTE this identity is DEFINITIONAL, not independent physics: the code sets
    dT_dt = -(u*du+v*dv)/c_pd and then eps_gwd from the SAME inner product, so
    the equality is bookkeeping (it pins that the orographic KE->heat closure
    holds exactly as coded, and — the discriminating point — that a NONZERO
    residual would appear for a source whose thermal deposition is not the KE
    removal rate, which is exactly what the frontal canary below shows)."""
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column()
    cfg = E3SMCAMConfig(
        source="orographic",
        orographic=E3SMOrographicConfig(sgh_default=200.0),
        # The continuous-rate identity below holds for the CONTINUOUS
        # closure; the shipped default is now the E3SM discrete closure
        # (flipped 2026-07-17 after the AMIP A/B), so pin it explicitly.
        use_discrete_ke_heating=False,
    )
    out = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0, cfg)
    # Non-vacuous: the drag is actually active.
    assert float(jnp.max(jnp.abs(out.du_dt))) > 0.0
    imbalance = _column_energy_imbalance(out, rho, zh)
    ke_removal = out.eps_gwd
    rel = float(jnp.max(jnp.abs(imbalance) / jnp.clip(jnp.abs(ke_removal), 1e-12, None)))
    assert rel < 1e-10, f"orographic KE->heat did not close: rel={rel:.2e}"


# --------------------------------------------------------------------------
# 3. Frontal: the default ground-relative frame term is NOT the mean-flow KE
#    removal rate (so the orographic KE->heat closure does NOT extend to spectra).
# --------------------------------------------------------------------------

def test_frontal_default_frame_differs_from_mean_flow_ke_removal():
    """For the spectral path the default thermal term is the GROUND-RELATIVE
    ``dttke = sum_l c_l*gwut_l`` (the wave-energy-flux-divergence term), which is
    NOT the mean-flow KE tendency ``sum_l ubm*gwut_l`` (whose negated column
    integral is ``eps_gwd``). So ``c_pd*sum(rho*dT*dz)`` (the deposited thermal
    quantity) != ``eps_gwd`` (the mean-flow KE removal rate) --
    the orographic definitional KE->heat closure does NOT extend to a frontal
    spectrum, which is the direct reason a static ["energy"] claim is wrong.

    This is a SINGLE-COLUMN canary: the magnitude of the gap is spectrum- and
    column-dependent (cancellation across +/-c bins can shrink it for other
    soundings), so the >1e-2 bound is specific to this jet, not a universal
    lower bound. The frame difference itself is proven generally by the sibling
    test_gwd_e3sm_cam.test_frontal_dttke_intrinsic_switch (default vs intrinsic
    differ by the -ubm*gwut term)."""
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column()
    ncol, nlev = u.shape
    frontgf = jnp.full((ncol, nlev), 1e-9)
    cfg = E3SMCAMConfig(
        source="frontal", pgwv=8, dc=5.0,
        frontal=E3SMFrontalConfig(taubgnd=1.5e-3, frontgfc=1e-10),
        # Isolate the RAW ground-relative dttke frame: the shipped defaults
        # now add dttdf and run the C.-C. Chen fixer (which CLOSES the
        # budget, erasing exactly the gap this canary measures).
        use_e3sm_spectral_heating=False,
        do_energy_conservation=False,
    )
    out = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0, cfg,
                       frontgf_col=frontgf)
    assert float(jnp.max(jnp.abs(out.du_dt))) > 0.0
    imbalance = _column_energy_imbalance(out, rho, zh)
    ke_removal = out.eps_gwd
    rel = float(jnp.max(jnp.abs(imbalance) / jnp.clip(jnp.abs(ke_removal), 1e-12, None)))
    assert rel > 1e-2, (
        f"frontal thermal deposition matched the KE removal rate (rel={rel:.2e}) "
        "on this column; the ground-relative dttke should differ from the "
        "mean-flow KE removal by the ubm frame term"
    )


# --------------------------------------------------------------------------
# 4. Version choice: the driver's DEFAULT heating frame is ground-relative
#    (the irreversible intrinsic form is opt-in).
# --------------------------------------------------------------------------

def test_default_heating_frame_is_ground_relative():
    """The default matches the pinned E3SM-3.0.1 oracle (ground-relative
    ``dttke``); the irreversible intrinsic form is opt-in. Beyond the literal
    flag, tie the default to actual DRIVER behavior: the default-config output
    is bit-identical to an explicit ``dttke_use_intrinsic=False`` and DIFFERS
    from ``=True`` (so a silent flip of the default would change the heating)."""
    assert E3SMCAMConfig().dttke_use_intrinsic is False
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column(ncol=1)
    ncol, nlev = u.shape
    frontgf = jnp.full((ncol, nlev), 1e-9)
    base = dict(source="frontal", pgwv=8, dc=5.0,
                frontal=E3SMFrontalConfig(taubgnd=1.5e-3, frontgfc=1e-10))
    out_default = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0,
                               E3SMCAMConfig(**base), frontgf_col=frontgf)
    out_ground = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0,
                              E3SMCAMConfig(**base, dttke_use_intrinsic=False),
                              frontgf_col=frontgf)
    out_intrinsic = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0,
                                 E3SMCAMConfig(**base, dttke_use_intrinsic=True),
                                 frontgf_col=frontgf)
    # Default == explicit ground-relative (the driver honors the default).
    np.testing.assert_array_equal(np.array(out_default.dT_dt),
                                  np.array(out_ground.dT_dt))
    # ... and the intrinsic form is genuinely different (non-vacuous).
    assert float(jnp.max(jnp.abs(out_default.dT_dt - out_intrinsic.dT_dt))) > 0.0


# --------------------------------------------------------------------------
# 5. The energy fixer ships ON since 2026-07-17; the EXPLICIT opt-out arm
#    shows a spectral config does not close the column budget without it.
# --------------------------------------------------------------------------

def test_energy_fixer_explicit_off_leaves_budget_open():
    """``do_energy_conservation`` (C.-C. Chen fixer) now defaults ON (the
    oracle calls it unconditionally after each spectral gw_drag_prof;
    flipped 2026-07-17 after the AMIP/RCE A/B).  With the fixer explicitly
    OFF, a spectral (frontal, effgw=0.5) config leaves a non-trivial
    residual in E3SM's exact column total-energy metric — the behavior the
    old default shipped.  (Machine-zero closure with the fixer ON is
    covered by test_gwd_e3sm_ediff.test_driver_energy_closure_machine_zero.)"""
    assert E3SMCAMConfig().do_energy_conservation is True  # oracle default
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column(ncol=1)
    ncol, nlev = u.shape
    frontgf = jnp.full((ncol, nlev), 1e-9)
    cfg = E3SMCAMConfig(  # fixer + spectral heating explicitly OFF (legacy)
        source="frontal", pgwv=8, dc=5.0, effgw=0.5,
        frontal=E3SMFrontalConfig(taubgnd=1.5e-3, frontgfc=1e-10),
        do_energy_conservation=False,
        use_e3sm_spectral_heating=False,
    )
    out = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0, cfg,
                       frontgf_col=frontgf)
    assert float(jnp.max(jnp.abs(out.du_dt))) > 0.0  # non-vacuous: drag active
    # E3SM's exact column total-energy metric (dry-static + kinetic tendency,
    # including the quadratic 0.5*dt*(du/dt)^2 term); the fixer removes this
    # residual uniformly from dsdt below source when ON.
    dt = 1800.0
    pdel = np.abs(np.diff(np.array(ph[0])))
    u_np, v_np = np.array(u[0]), np.array(v[0])
    dudt, dvdt = np.array(out.du_dt[0]), np.array(out.dv_dt[0])
    dsdt = float(C.c_pd) * np.array(out.dT_dt[0])
    E = np.sum(pdel * (
        dsdt + dudt * (u_np + 0.5 * dt * dudt) + dvdt * (v_np + 0.5 * dt * dvdt)
    ))
    assert abs(E) > 1e-3, (
        f"frontal column energy budget unexpectedly closed with the fixer OFF "
        f"(E={E:.2e}); this config should leave a residual so the fixer flag has "
        "an observable effect"
    )


# --------------------------------------------------------------------------
# 6. E3SM driver-level orographic heating + landfrac (gw_drag.F90:902-915).
# --------------------------------------------------------------------------

def _oro_cfg(**kw):
    kw.setdefault("use_discrete_ke_heating", False)   # legacy arm explicit
    return E3SMCAMConfig(
        source="orographic",
        orographic=E3SMOrographicConfig(sgh_default=200.0),
        **kw,
    )


def test_discrete_ke_heating_closes_discrete_budget():
    """Oracle mirror (gw_drag.F90:908-913, no-energy-fix default branch,
    ttgw = 0 for the oro ngwv=0 call): with use_discrete_ke_heating=True the
    heat returned per step must equal EXACTLY the discrete resolved-KE loss,

        c_pd*dT_dt*dt == -0.5*[(u+dt*du)^2 - u^2 + (v+dt*dv)^2 - v^2],

    level-by-level to machine precision (the algebraic identity
    du*(u+0.5*dt*du)*dt == 0.5*[(u+dt*du)^2 - u^2])."""
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column()
    dt = 1800.0
    out = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, dt,
                       _oro_cfg(use_discrete_ke_heating=True))
    assert float(jnp.max(jnp.abs(out.du_dt))) > 0.0
    lhs = float(C.c_pd) * out.dT_dt * dt
    rhs = -0.5 * ((u + dt * out.du_dt) ** 2 - u ** 2
                  + (v + dt * out.dv_dt) ** 2 - v ** 2)
    err = float(jnp.max(jnp.abs(lhs - rhs)))
    scale = float(jnp.max(jnp.abs(rhs))) + 1e-30
    assert err / scale < 1e-12, f"discrete KE closure broken: rel={err/scale:.2e}"


def test_discrete_ke_flag_off_is_bit_identical_and_on_differs():
    """The EXPLICIT False arm must reproduce the continuous-rate identity
    exactly (legacy regression pin; True is the DEFAULT since 2026-07-17);
    True must differ by the 0.5*dt*(du^2+dv^2)/c_pd term — i.e. discrete
    heating is STRICTLY LESS where drag acts (the continuous form
    over-heats)."""
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column()
    dt = 1800.0
    out_off = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, dt, _oro_cfg())
    expected_off = -(u * out_off.du_dt + v * out_off.dv_dt) / float(C.c_pd)
    assert jnp.array_equal(out_off.dT_dt, expected_off)

    out_on = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, dt,
                          _oro_cfg(use_discrete_ke_heating=True))
    # Same momentum tendencies (the flag touches only the heating).
    assert jnp.array_equal(out_on.du_dt, out_off.du_dt)
    diff = out_off.dT_dt - out_on.dT_dt
    expected_gap = 0.5 * dt * (out_off.du_dt ** 2
                               + out_off.dv_dt ** 2) / float(C.c_pd)
    assert float(jnp.max(jnp.abs(diff - expected_gap))) < 1e-18
    assert float(jnp.max(expected_gap)) > 0.0


def test_landfrac_scales_oro_drag_and_heat_in_order():
    """gw_drag.F90:904-913 mirror: utgw *= landfrac happens BEFORE the heating
    closure, so (a) land_frac=0 zeroes drag AND heat, (b) land_frac=1 is
    identical to None, (c) at land_frac=0.5 the momentum tendencies scale by
    exactly 0.5 while the discrete heat is computed FROM the scaled
    tendencies (NOT 0.5x the unscaled heat — the order discriminant)."""
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column()
    ncol = u.shape[0]
    dt = 1800.0
    cfg = _oro_cfg(use_discrete_ke_heating=True)
    base = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, dt, cfg)

    zero = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, dt, cfg,
                        land_frac_col=jnp.zeros(ncol))
    assert float(jnp.max(jnp.abs(zero.du_dt))) == 0.0
    assert float(jnp.max(jnp.abs(zero.dT_dt))) == 0.0
    assert float(jnp.max(jnp.abs(zero.eps_gwd))) == 0.0

    ones = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, dt, cfg,
                        land_frac_col=jnp.ones(ncol))
    assert jnp.array_equal(ones.du_dt, base.du_dt)
    assert jnp.array_equal(ones.dT_dt, base.dT_dt)

    half = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, dt, cfg,
                        land_frac_col=jnp.full(ncol, 0.5))
    assert float(jnp.max(jnp.abs(half.du_dt - 0.5 * base.du_dt))) < 1e-18
    # Heat from the SCALED du (order discriminant): recompute the closure
    # from half.du_dt and compare; 0.5x the unscaled heat would be WRONG
    # (differs by the 0.5*dt*du^2 quadratic term).
    expected_heat = -(half.du_dt * (u + 0.5 * dt * half.du_dt)
                      + half.dv_dt * (v + 0.5 * dt * half.dv_dt)
                      ) / float(C.c_pd)
    assert float(jnp.max(jnp.abs(half.dT_dt - expected_heat))) < 1e-22
    wrong_heat = 0.5 * base.dT_dt
    assert float(jnp.max(jnp.abs(half.dT_dt - wrong_heat))) > 0.0


def test_landfrac_ignored_for_spectral_source():
    """E3SM applies landfrac ONLY to the orographic tendencies
    (gw_drag.F90:904-906 sits in the oro block; the spectral applications at
    :795/:859 have no landfrac).  Passing land_frac_col=0 with the frontal
    source must change nothing."""
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column()
    ncol, nlev = u.shape
    cfg = E3SMCAMConfig(source="frontal", pgwv=8)
    frontgf = jnp.full((ncol, nlev), 1.0e-1)
    out = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0, cfg,
                       frontgf_col=frontgf)
    out_lf = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0, cfg,
                          frontgf_col=frontgf,
                          land_frac_col=jnp.zeros(ncol))
    assert jnp.array_equal(out.du_dt, out_lf.du_dt)
    assert jnp.array_equal(out.dT_dt, out_lf.dT_dt)


def test_landfrac_path_is_differentiable():
    """jax.grad through land_frac (a traced continuous scaling) is finite and
    nonzero — the coupler can differentiate through the surface-type field."""
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column()
    ncol = u.shape[0]
    cfg = _oro_cfg(use_discrete_ke_heating=True)

    def loss(lf):
        out = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0, cfg,
                           land_frac_col=lf)
        return jnp.sum(out.du_dt ** 2) + jnp.sum(out.dT_dt ** 2)

    g = jax.grad(loss)(jnp.full(ncol, 0.7))
    assert bool(jnp.all(jnp.isfinite(g)))
    assert float(jnp.max(jnp.abs(g))) > 0.0


def test_extract_land_frac_grid_attr():
    """Integration wiring: the canonical grid attribute ``land_frac`` is
    picked up per column; absent attribute -> None (legacy, no scaling)."""
    from legoesm.atmosphere.physics.gravity_wave_drag.integration import (
        _extract_land_frac,
    )

    class _G:
        pass

    g = _G()
    assert _extract_land_frac(g, 4) is None
    g.land_frac = np.array([[0.0, 0.25], [0.5, 1.0]])
    got = _extract_land_frac(g, 4)
    assert got.shape == (4,)
    np.testing.assert_allclose(np.asarray(got), [0.0, 0.25, 0.5, 1.0])


# --------------------------------------------------------------------------
# 7. E3SM-faithful spectral thermal term (use_e3sm_spectral_heating;
#    gw_common.F90:708-731 — unconditional dttdf + dttke band ktop+1..kbotbg)
# --------------------------------------------------------------------------

def _frontal_cfg(**kw):
    kw.setdefault("use_e3sm_spectral_heating", False)  # legacy arm explicit
    kw.setdefault("do_energy_conservation", False)
    return E3SMCAMConfig(
        source="frontal", pgwv=8, dc=5.0,
        frontal=E3SMFrontalConfig(taubgnd=1.5e-3, frontgfc=1e-10),
        **kw,
    )


def test_spectral_heating_adds_dttdf_not_momentum():
    """Flag ON (frontal, where the dttke band is inert — tend_level IS
    kbotbg so gwut = 0 below): dT gains exactly the unconditional dttdf
    term while du/dv are BIT-IDENTICAL (E3SM never diffuses u/v with
    egwdffi — the discriminant vs the do_eddy_diffusion standalone
    addition, which changes du)."""
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column()
    ncol, nlev = u.shape
    frontgf = jnp.full((ncol, nlev), 1e-9)
    out_off = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0,
                           _frontal_cfg(), frontgf_col=frontgf)
    out_on = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0,
                          _frontal_cfg(use_e3sm_spectral_heating=True),
                          frontgf_col=frontgf)
    assert jnp.array_equal(out_on.du_dt, out_off.du_dt)
    assert jnp.array_equal(out_on.dv_dt, out_off.dv_dt)
    d = float(jnp.max(jnp.abs(out_on.dT_dt - out_off.dT_dt)))
    assert d > 0.0, "dttdf did not engage"

    out_ediff = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0,
                             _frontal_cfg(do_eddy_diffusion=True),
                             frontgf_col=frontgf)
    assert not jnp.array_equal(out_ediff.du_dt, out_off.du_dt), (
        "do_eddy_diffusion should change du (standalone u/v addition)")
    # dttdf itself is the SAME machinery: with the band inert the two
    # flags' dT paths agree exactly.
    assert jnp.array_equal(out_on.dT_dt, out_ediff.dT_dt)


def test_spectral_heating_composes_dttdf_once():
    """Both flags ON (fixer OFF, as here): dT identical to e3sm-only —
    dttdf added exactly once and the u/v diffusion does not feed dT; du
    gains the standalone diffusion.  With do_energy_conservation=True the
    changed du/dv WOULD feed the fixer's dse redistribution, so the
    equality below holds only on the fixer-off path (codex wave-7 P3)."""
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column()
    ncol, nlev = u.shape
    frontgf = jnp.full((ncol, nlev), 1e-9)
    on = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0,
                      _frontal_cfg(use_e3sm_spectral_heating=True),
                      frontgf_col=frontgf)
    both = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0,
                        _frontal_cfg(use_e3sm_spectral_heating=True,
                                     do_eddy_diffusion=True),
                        frontgf_col=frontgf)
    assert jnp.array_equal(both.dT_dt, on.dT_dt)
    assert not jnp.array_equal(both.du_dt, on.du_dt)


def test_spectral_heating_band_cuts_deep_beres_dttke():
    """Beres source with deep heating (top well below 500 hPa): the legacy
    all-level dttke deposits ground-relative wave heating below the E3SM
    band that the flag removes (gw_common.F90:726-728).  Compare the flag's
    dT against the OFF run MINUS its below-band dttke: with dttdf disabled
    by a zero-diffusivity cap, the two must agree — pinning that the flag's
    ONLY below-band effect is the band cut."""
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column()
    ncol, nlev = u.shape
    pmid_mean = np.mean(np.array(pf), axis=0)
    # Shear layer 700->500 hPa (u: +10 -> -10) puts critical levels for the
    # inner phase-speed bins BETWEEN the Beres launch (~650 hPa) and the
    # 500 hPa band edge, forcing legacy dttke deposition BELOW the band.
    u_prof = np.interp(pmid_mean, [1.0e4, 5.0e4, 7.0e4, 1.0e5],
                       [-12.0, -12.0, 12.0, 12.0])
    u = jnp.asarray(np.broadcast_to(u_prof, (ncol, nlev)).copy())
    # Deep convective heating 950-600 hPa (hdepth > the 2.5 km trigger; a
    # shallower band left hdepth under the minimum and launched nothing).
    heat = np.zeros((ncol, nlev))
    mask = (pmid_mean > 6.0e4) & (pmid_mean < 9.5e4)
    heat[:, mask] = 2e-3                                   # K/s, deep layer
    netdt = jnp.asarray(heat)
    base = dict(source="convective", pgwv=8, dc=2.5,
                do_energy_conservation=False)
    cfg_off = E3SMCAMConfig(**base, use_e3sm_spectral_heating=False)
    # prndl=0 zeroes egwdffm AT THE SOURCE -> egwdffi = min(cap, 0) = 0
    # everywhere -> dttdf == 0 exactly, isolating the band effect.  (NOT
    # egwd_max=0: the cap is min(egwd_max, x) with NO zero floor, so a
    # NEGATIVE diffusivity would survive it and dttdf would not null.)
    cfg_on = E3SMCAMConfig(**base, use_e3sm_spectral_heating=True,
                           prndl=0.0)
    out_off = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0,
                           cfg_off, netdt_col=netdt)
    out_on = e3sm_cam_gwd(u, v, T, pf, ph, zf, zh, rho, lat, 1800.0,
                          cfg_on, netdt_col=netdt)
    kbot = int(np.argmin(np.abs(pmid_mean - cfg_on.ediff_kbot_p)))
    below = np.array(out_off.dT_dt)[:, kbot:]
    assert np.max(np.abs(below)) > 0.0, (
        "fixture must deposit legacy dttke below the 500 hPa band")
    # Flag ON: below-band dT is exactly zero (band cut, dttdf nulled).
    np.testing.assert_allclose(np.array(out_on.dT_dt)[:, kbot:], 0.0,
                               atol=0.0)
    # Above the band the two agree exactly (the flag touches nothing else).
    np.testing.assert_allclose(np.array(out_on.dT_dt)[:, :kbot],
                               np.array(out_off.dT_dt)[:, :kbot],
                               rtol=0, atol=0)
    assert E3SMCAMConfig().use_e3sm_spectral_heating is True  # oracle default (flipped 2026-07-17)
