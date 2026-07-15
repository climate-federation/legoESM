"""Faithfulness / structural canaries for the prognostic spectral GWD scheme.

``prognostic_spectral.py`` is BESPOKE (honestly self-labelled experimental, not
a faithful port of a published spectral GWD), so this pins the honest
declarations + the specific FAITHFUL ingredients it does reuse, complementing
the physical-realism suite (``test_gwd_physical_realism.py`` already covers
drag-opposes-slow-spectrum, heating>=0 for the default spectrum, the energy
tie-back, and finiteness). Here we add the structural pins that suite omits:

* ``conserves == ["none"]`` (the honest contract);
* the PROGNOSTIC spectrum relaxation toward the launch source;
* the ``thermal_tendency`` gate;
* the intrinsic-MAGNITUDE heating (``|c-U|``) staying >=0 for a SLOW spectrum
  (``c < U`` => intrinsic < 0), which discriminates the intrinsic-magnitude
  ``|c-U|`` form (a positive dissipation diagnostic, NOT the E3SM dttke
  identity) from a signed ``(c-U)`` term;
* the CONSTANT launch-level sign ``s0`` (F-GWD-1 fix): the deposition sign is
  set at the launch level, not the local layer.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants as C
from legoesm.atmosphere.physics.gravity_wave_drag.config import (
    PrognosticSpectralConfig,
)
from legoesm.atmosphere.physics.gravity_wave_drag.prognostic_spectral import (
    __physics_contract__,
    prognostic_spectral_gwd,
)


@pytest.fixture(autouse=True)
def _x64():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", prev)


def _column(nlev=40, u_surface=10.0, u_top=10.0, T0=250.0, ncol=1):
    """Isothermal hydrostatic column with a linear wind ramp (top -> surface).

    k=0 is the model top, k=nlev-1 the surface (launch) level. ``u`` ramps
    linearly from ``u_top`` (k=0) to ``u_surface`` (k=nlev-1).
    """
    g = float(C.g)
    rair = float(C.R_d)
    p_top, p_surf = 100.0, 1.0e5
    pint = np.linspace(p_top, p_surf, nlev + 1)
    pmid = 0.5 * (pint[:-1] + pint[1:])
    T = np.full(nlev, T0)
    dp = np.diff(pint)
    dz = rair * T * dp / (g * pmid)
    z_half = np.zeros(nlev + 1)
    for k in range(nlev, 0, -1):
        z_half[k - 1] = z_half[k] + dz[k - 1]
    zm = 0.5 * (z_half[:-1] + z_half[1:])
    u = np.linspace(u_top, u_surface, nlev)  # k=0 top ... k=nlev-1 surface

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


def _run(cfg, launch, col=None):
    if col is None:
        col = _column()
    u, v, T, pf, ph, zf, zh, rho, lat = col
    spec = jnp.full((u.shape[0], cfg.n_azimuths, cfg.n_wavenumbers), launch)
    out, spec_new = prognostic_spectral_gwd(
        u, v, T, pf, ph, zf, zh, rho, lat, 1800.0, cfg, spec
    )
    return out, spec_new, col


# --------------------------------------------------------------------------
# 1. The honest conservation declaration.
# --------------------------------------------------------------------------

def test_conserves_is_none():
    """A relaxed-toward-launch spectrum + untracked top flux => the column
    budget is open; ``conserves`` must be the audited ["none"]."""
    assert __physics_contract__["conserves"] == ["none"]


# --------------------------------------------------------------------------
# 2. The PROGNOSTIC feature: the spectrum relaxes toward the launch source.
# --------------------------------------------------------------------------

def test_spectrum_relaxes_toward_launch_flux():
    """spectrum_new = spectrum + dt*(launch_flux - spectrum)/tau_decay, exactly
    (independent of the wind). This is the defining 'prognostic' memory term."""
    cfg = PrognosticSpectralConfig()
    dt = 1800.0
    spec_in_val = 5e-3  # != launch_flux (1e-3) so relaxation is non-trivial
    _, spec_new, _ = _run(cfg, spec_in_val)
    expected = spec_in_val + dt * (cfg.launch_flux - spec_in_val) / cfg.tau_decay
    np.testing.assert_allclose(np.asarray(spec_new), expected, rtol=1e-12)
    # Non-vacuous: the spectrum actually moved toward the (smaller) launch flux.
    assert float(jnp.max(spec_new)) < spec_in_val


# --------------------------------------------------------------------------
# 3. The thermal_tendency gate.
# --------------------------------------------------------------------------

def test_thermal_tendency_gate_and_pregate_eps():
    """dT_dt is zeroed when config.thermal_tendency is False and active when
    True. But eps_gwd is computed from the PRE-GATE heating, so it is UNCHANGED
    by the gate: with the gate OFF, eps_gwd stays > 0 while dT_dt == 0, i.e.
    eps_gwd is the diagnosed would-be dissipation, NOT the applied thermal
    deposition (documented in the contract / module Faithfulness note)."""
    base = dict(k_min=2.0 * math.pi / 3.0e3)  # slow spectrum -> strong breaking
    out_on, _, _ = _run(PrognosticSpectralConfig(**base, thermal_tendency=True), 0.1)
    out_off, _, _ = _run(PrognosticSpectralConfig(**base, thermal_tendency=False), 0.1)
    assert float(jnp.max(jnp.abs(out_on.dT_dt))) > 0.0
    assert float(jnp.max(jnp.abs(out_off.dT_dt))) == 0.0
    # eps_gwd is pre-gate: unchanged by the flag, and > 0 even with dT_dt == 0.
    np.testing.assert_allclose(np.asarray(out_off.eps_gwd),
                               np.asarray(out_on.eps_gwd), rtol=1e-12)
    assert float(jnp.max(out_off.eps_gwd)) > 0.0


# --------------------------------------------------------------------------
# 4. Intrinsic-MAGNITUDE heating: >=0 even for a SLOW spectrum (intrinsic<0).
# --------------------------------------------------------------------------

def test_slow_spectrum_heating_nonneg_discriminates_abs_intrinsic():
    """The heating uses |c-U| (intrinsic magnitude, not the E3SM dttke
    identity), not signed
    (c-U). For a spectrum SLOWER than the wind everywhere, c-U_proj < 0 on the
    along-wind azimuth, so a signed form could give NEGATIVE heating; |c-U|
    keeps dT_dt >= 0 and eps_gwd >= 0. Non-vacuous: breaking is active."""
    cfg = PrognosticSpectralConfig(k_min=2.0 * math.pi / 3.0e3)  # c <= ~9.4 m/s
    out, _, _ = _run(cfg, 0.1)
    assert float(jnp.max(jnp.abs(out.du_dt))) > 0.0     # deposition active
    assert float(jnp.min(out.dT_dt)) >= -1e-15
    assert float(jnp.min(out.eps_gwd)) >= -1e-12
    # And the heating is genuinely non-trivial (not a vacuous all-zero pass).
    assert float(jnp.max(out.dT_dt)) > 0.0


# --------------------------------------------------------------------------
# 5. CONSTANT launch-level sign s0 (F-GWD-1): the deposition sign is set at
#    the launch level, not the local layer.
# --------------------------------------------------------------------------

def test_deposition_sign_is_set_at_launch_not_local_layer():
    """F-GWD-1 fix: s0 = sign(c - U_launch) is FIXED at the surface. Two columns
    that share the SAME launch (surface) wind but differ in the wind ALOFT must
    produce deposition of the SAME sign at every level (the launch sign), even
    though a local-layer sign would flip where the aloft wind crosses c. Use a
    single azimuth (+x) so du_dt isolates that azimuth's projected deposition.

    Surface u=+10 with a slow spectrum (c<10) => s0<0 => du_dt<=0 everywhere.
    Column A keeps u>0 aloft; column B reverses to u<0 aloft (local sign would
    flip to +). Both must stay du_dt<=0, and they must DIFFER (non-vacuous)."""
    cfg = PrognosticSpectralConfig(n_azimuths=1, k_min=2.0 * math.pi / 3.0e3)
    col_a = _column(u_top=10.0, u_surface=10.0)
    col_b = _column(u_top=-10.0, u_surface=10.0)  # reverses aloft
    out_a, _, _ = _run(cfg, 0.1, col=col_a)
    out_b, _, _ = _run(cfg, 0.1, col=col_b)
    du_a = np.asarray(out_a.du_dt[0])
    du_b = np.asarray(out_b.du_dt[0])
    u_b = np.asarray(col_b[0][0])
    # Deposition is active in both.
    assert np.max(np.abs(du_a)) > 0.0 and np.max(np.abs(du_b)) > 0.0
    # Same (launch) sign everywhere: du_dt <= 0 in BOTH, incl. where u<0 aloft.
    assert np.all(du_a <= 1e-15)
    assert np.all(du_b <= 1e-15)
    # Non-vacuous discriminator: the aloft wind change DID change the magnitude
    # (so the two columns are genuinely different), yet the sign never flipped.
    assert np.max(np.abs(du_a - du_b)) > 1e-12
    # STRICT non-vacuity (codex): require an ACTIVE-deposition layer where the
    # LOCAL sign of the fastest wave OPPOSES the launch sign -- there the launch
    # sign (<=0) is what determines du_dt, not the local (>0) sign. Fastest
    # phase speed c = N/k_min for the isothermal T0=250 K column.
    N = math.sqrt(float(C.g) ** 2 / (float(C.c_pd) * 250.0))
    c_fast = N / cfg.k_min
    s_launch = np.sign(c_fast - 10.0)                     # launch sign (< 0)
    local_opposes = (np.sign(c_fast - u_b) != s_launch) & (np.abs(du_b) > 1e-14)
    assert np.any(local_opposes), (
        "no active-deposition layer where the local sign opposes the launch "
        "sign -> the constant-launch-sign design is not exercised (vacuous)"
    )
    assert np.all(du_b[local_opposes] <= 1e-15)          # launch sign wins there
