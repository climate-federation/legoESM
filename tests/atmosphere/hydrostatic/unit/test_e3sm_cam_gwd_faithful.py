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

The irreversible intrinsic heating frame and the C.-C. Chen momentum+energy
fixer both exist but are OFF by default.
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
# 5. The energy fixer FLAG is OFF by default, so a spectral config does not
#    close the column energy budget without it.
# --------------------------------------------------------------------------

def test_energy_fixer_off_by_default_so_budget_does_not_close():
    """``do_energy_conservation`` (C.-C. Chen fixer) is OFF by default. On a
    spectral (frontal, effgw=0.5) config -- one where the fixer HAS an effect --
    the shipped default leaves a non-trivial residual in E3SM's exact column
    total-energy metric, i.e. it does NOT close the budget. This isolates the
    fixer FLAG's default (the orographic default SOURCE separately self-closes,
    tested above; the point here is the spectral paths do not without the fixer).
    (The machine-zero closure when the fixer is ON is covered by
    test_gwd_e3sm_ediff.test_driver_energy_closure_machine_zero.)"""
    assert E3SMCAMConfig().do_energy_conservation is False
    u, v, T, pf, ph, zf, zh, rho, lat = _driver_column(ncol=1)
    ncol, nlev = u.shape
    frontgf = jnp.full((ncol, nlev), 1e-9)
    cfg = E3SMCAMConfig(  # default do_energy_conservation=False
        source="frontal", pgwv=8, dc=5.0, effgw=0.5,
        frontal=E3SMFrontalConfig(taubgnd=1.5e-3, frontgfc=1e-10),
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
