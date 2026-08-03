"""Truth-tier unit tests for the NEMO zdfmxl mixed-layer-depth criteria used
by the ldfslp isoneutral-slope ramp (``gm_redi_latlon_cgrid._nemo_mld``).

Node 6/7 of the DINO wiring match: NEMO ``zdfmxl.F90:91-105`` sets the MLD from
the N^2 INTEGRAL ``integral(MAX(N^2,0) dz) >= g*rho_c/rho0`` (``"n2_integral"``),
not a potential-density difference (``"rho_c"``, the legoESM default).  These
assert:

  1. the ``"n2_integral"`` MLD lands exactly at the depth where the cumulative
     ``integral(MAX(N^2,0) dz)`` — built from the SAME shared adiabatic-N^2 the
     native ldfslp slopes consume — first reaches the analytic threshold;
  2. the criteria genuinely differ (an unstable inversion inside the surface
     layer is ignored by the clamped N^2 integral but not by the raw
     pot-density difference);
  3. the dispatcher raises on an unknown criterion (dispatch hardening).
"""
from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic
from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    _NEMO_MLD_REF_DEPTH_M,
    _nemo_mld,
    _nemo_mld_from_n2_integral,
    _nemo_mld_from_potential_density,
)

RHO0 = 1026.0
ALPHA = 2.0e-4     # thermal expansion [1/degC]
T0 = 10.0
D = 25.0           # uniform layer thickness [m]
NLEV = 40


def _lin_eos(T, S, p):
    """Linear, pressure-independent EOS: rho = rho0 (1 - alpha (T - T0))."""
    return RHO0 * (1.0 - ALPHA * (T - T0))


def _uniform_zcoord():
    return types.SimpleNamespace(dz_ref=jnp.full((NLEV,), D))


def _column(T_1d):
    """Broadcast a 1-D T profile to a (1,1,nlev) wet single column."""
    T = jnp.asarray(T_1d, dtype=jnp.float64).reshape((1, 1, NLEV))
    S = jnp.full((1, 1, NLEV), 35.0)
    mask = jnp.ones((1, 1))
    return T, S, mask


def _independent_n2_integral_mld(T, S, z_coord):
    """Re-derive NEMO's zdfmxl criterion in NumPy from the public adiabatic N^2.

    Independent of the helper's internal bookkeeping: it re-computes N^2 with
    the shared EOS routine and replays the exact integral/threshold/index rule.
    """
    dz = np.asarray(z_coord.dz_ref)
    z_iface = np.cumsum(dz)
    z_centers = z_iface - 0.5 * dz
    p_cell = (RHO0 * constants.g * z_centers)[None, None, :] * np.ones_like(np.asarray(T))
    J1 = jnp.ones((1, 1))
    n2 = np.asarray(compute_buoyancy_frequency_adiabatic(
        T, S, jnp.asarray(p_cell), z_coord.dz_ref, J1,
        eos_fn=_lin_eos, rho_ref=RHO0, g=constants.g))[0, 0]
    e3w = z_centers[1:] - z_centers[:-1]
    iref = int(np.clip(np.searchsorted(z_iface, _NEMO_MLD_REF_DEPTH_M), 0, NLEV - 2))
    contrib = np.where(np.arange(NLEV - 1) < iref, 0.0, np.maximum(n2, 0.0) * e3w)
    cum = np.cumsum(contrib)
    thresh = constants.g * 0.01 / RHO0
    reached = cum >= thresh
    m_base = int(np.argmax(reached)) if reached.any() else NLEV - 2
    return z_iface[m_base], m_base, n2, thresh


def test_n2_integral_mld_matches_analytic_crossing():
    # Well-mixed to the base of cell (k_ml-1), then a WEAK fixed lapse below so
    # the column is stably stratified with a KNOWN constant N^2 per interface —
    # weak enough that the tiny rho_c=0.01 threshold takes SEVERAL interfaces to
    # reach (genuinely exercising the cumulative integral, not a 1-step cross).
    k_ml = 5
    lapse = 0.02  # degC colder per level below the ML -> contrib ~ thresh/2.4
    T = np.full(NLEV, 18.0)
    for k in range(k_ml, NLEV):
        T[k] = 18.0 - lapse * (k - (k_ml - 1))
    T_j, S_j, mask = _column(T)
    z_coord = _uniform_zcoord()

    hml_exp, m_base_exp, n2, thresh = _independent_n2_integral_mld(T_j, S_j, z_coord)
    hml, m_base = _nemo_mld_from_n2_integral(
        T_j, S_j, mask, z_coord, _lin_eos, 0.01, constants.g, RHO0)

    # PRIMARY truth-tier: helper matches the independent NumPy criterion exactly.
    assert int(m_base[0, 0]) == m_base_exp
    assert float(hml[0, 0]) == pytest.approx(hml_exp)

    # The first stratified interface is at index k_ml-1 (base of the last mixed
    # cell); the integral accumulates over several interfaces before crossing.
    first_strat = k_ml - 1
    n_accumulated = m_base_exp - first_strat + 1
    assert n_accumulated >= 2                       # multi-interface accumulation
    # Closed form: constant per-interface contribution g*alpha*lapse crosses the
    # g*rho_c/rho0 threshold after ceil(thresh/(g*alpha*lapse)) interfaces.
    contrib = constants.g * ALPHA * lapse
    count = int(np.ceil(thresh / contrib))
    assert n_accumulated == count
    # ML base is BELOW the well-mixed layer base z_iface[k_ml-1].
    z_iface = np.cumsum(np.asarray(z_coord.dz_ref))
    assert hml_exp > z_iface[first_strat - 1]


def test_dispatch_matches_helpers_and_defaults_to_rho_c():
    T = np.linspace(18.0, 2.0, NLEV)          # smoothly stratified
    T_j, S_j, mask = _column(T)
    z_coord = _uniform_zcoord()

    rho_c_ref = _nemo_mld_from_potential_density(T_j, S_j, mask, z_coord, _lin_eos, 0.01)
    n2_ref = _nemo_mld_from_n2_integral(
        T_j, S_j, mask, z_coord, _lin_eos, 0.01, constants.g, RHO0)

    hml_rho, mb_rho = _nemo_mld("rho_c", T_j, S_j, mask, z_coord, _lin_eos, 0.01,
                                g=constants.g, rho_0=RHO0)
    hml_n2, mb_n2 = _nemo_mld("n2_integral", T_j, S_j, mask, z_coord, _lin_eos,
                              0.01, g=constants.g, rho_0=RHO0)
    assert float(hml_rho[0, 0]) == pytest.approx(float(rho_c_ref[0][0, 0]))
    assert float(hml_n2[0, 0]) == pytest.approx(float(n2_ref[0][0, 0]))


def test_criteria_differ_on_surface_inversion():
    # Weak stratification (each interface accumulates ~1/5 of the threshold)
    # with a small UNSTABLE inversion (a warm bump at ~90 m).  For a linear EOS
    # the clamped integral is Sum(MAX(Drho,0)) and the pot-density difference is
    # Sum(Drho) (telescoping) -> the integral is >= the difference at every
    # depth, so it crosses the threshold at the SAME or a SHALLOWER interface.
    # The inversion's negative step delays the pot-density crossing (it must be
    # recovered by later positive steps) but is discarded by the clamp, so the
    # two MLDs disagree with mb_n2 strictly above mb_rho.
    T = np.zeros(NLEV)
    T[0] = 18.0
    lapse = 0.01
    for k in range(1, NLEV):
        T[k] = T[k - 1] - lapse
    T[3] += 0.06       # warm bump -> statically unstable inversion near cell 3
    T_j, S_j, mask = _column(T)
    z_coord = _uniform_zcoord()

    _, mb_rho = _nemo_mld("rho_c", T_j, S_j, mask, z_coord, _lin_eos, 0.01,
                          g=constants.g, rho_0=RHO0)
    _, mb_n2 = _nemo_mld("n2_integral", T_j, S_j, mask, z_coord, _lin_eos, 0.01,
                         g=constants.g, rho_0=RHO0)
    assert int(mb_n2[0, 0]) != int(mb_rho[0, 0])
    assert int(mb_n2[0, 0]) < int(mb_rho[0, 0])


def test_n2_integral_live_e3w_gated_on_partial_cell_coordinate():
    """#1226 bn2 live-e3w divisor (eosbn2.F90:1467, pn2 = ... / e3w(Kmm)).

    ``jacobian`` IS NEMO's live stretch ``(1+r3t) = (eta+H_bathy)/H_bathy``
    ONLY on an ``OceanPartialCellCoordinate`` (``compute_ocean_jacobian``);
    on a plain ``OceanZStarCoordinate`` the SAME-NAMED ``jacobian`` argument
    is a different quantity, ``(eta+H_bathy)/H_max``.  Synthetic-violation
    check: with ``t_depth_ref`` set and a jacobian far from 1.0,
      * on ``OceanPartialCellCoordinate`` the MLD must be sensitive to the
        stretch (bit-different from the unstretched/jacobian=None case) --
        proves the gate CAN fire when it should;
      * on ``OceanZStarCoordinate`` the result must be BIT-IDENTICAL to
        jacobian=None -- proves the gate does NOT misapply the wrong
        stretch there (the bug this test guards against: a bare
        ``if jacobian is not None`` would have applied it on both).
    """
    from legoesm.ocean.vertical import (
        OceanPartialCellCoordinate, OceanZStarCoordinate,
    )

    z_iface = jnp.cumsum(jnp.full((NLEV,), D))
    z_full_ref = -(z_iface - 0.5 * D)
    z_half_ref = jnp.concatenate([jnp.zeros((1,)), -z_iface])
    t_depth_ref = jnp.abs(z_full_ref) + 1.0   # deliberately off-midpoint

    z_star = OceanZStarCoordinate(
        n_levels=NLEV, H_max=float(z_iface[-1]),
        z_full_ref=z_full_ref, z_half_ref=z_half_ref,
        dz_ref=jnp.full((NLEV,), D), dz_half_ref=jnp.full((NLEV - 1,), D),
        t_depth_ref=t_depth_ref,
    )
    partial = OceanPartialCellCoordinate(
        n_levels=NLEV, H_max=z_star.H_max,
        z_full_ref=z_full_ref, z_half_ref=z_half_ref,
        dz_ref=z_star.dz_ref, dz_half_ref=z_star.dz_half_ref,
        h_partial=jnp.broadcast_to(z_star.dz_ref, (1, 1, NLEV)),
        bottom_level=jnp.full((1, 1), NLEV - 1, dtype=jnp.int32),
        is_active=jnp.ones((1, 1, NLEV), dtype=bool),
        t_depth_ref=t_depth_ref,
    )

    # Weak lapse (as in test_n2_integral_mld_matches_analytic_crossing) so the
    # integral accumulates over SEVERAL deep interfaces before crossing the
    # threshold -- giving depth-dependent alpha/beta (nemo_seos_alpha_beta's
    # mu1/mu2 pressure terms) leverage to move m_base under the stretch.
    k_ml = 5
    lapse = 0.02
    T = np.full(NLEV, 18.0)
    for k in range(k_ml, NLEV):
        T[k] = 18.0 - lapse * (k - (k_ml - 1))
    T_j, S_j, mask = _column(T)
    # far from 1.0 -- DINO's mu1=1.497e-4 thermobaric coefficient (eos.py
    # NemoSEOSConfig) needs a large stretch to move m_base by a whole
    # interface; 5.0 is unphysical but this test isolates gate LOGIC, not
    # a realistic eta/H_bathy ratio.
    jac = jnp.full((1, 1), 5.0)

    hml_none, mb_none = _nemo_mld_from_n2_integral(
        T_j, S_j, mask, z_star, _lin_eos, 0.01, constants.g, RHO0,
        jacobian=None)
    hml_zstar, mb_zstar = _nemo_mld_from_n2_integral(
        T_j, S_j, mask, z_star, _lin_eos, 0.01, constants.g, RHO0,
        jacobian=jac)
    hml_partial, mb_partial = _nemo_mld_from_n2_integral(
        T_j, S_j, mask, partial, _lin_eos, 0.01, constants.g, RHO0,
        jacobian=jac)

    # Plain z* coordinate: gate must NOT fire -- bit-identical to jacobian=None.
    assert float(hml_zstar[0, 0]) == float(hml_none[0, 0])
    assert int(mb_zstar[0, 0]) == int(mb_none[0, 0])

    # Partial-cell coordinate: gate MUST fire -- the stretch changes gdept/gdepw
    # fed to compute_buoyancy_frequency_nemo_bn2, so the result must differ from
    # the unstretched case (proves the branch is non-vacuous on this input).
    assert (float(hml_partial[0, 0]) != float(hml_none[0, 0])
            or int(mb_partial[0, 0]) != int(mb_none[0, 0]))


def test_dispatch_raises_on_unknown_criterion():
    T_j, S_j, mask = _column(np.linspace(18.0, 2.0, NLEV))
    z_coord = _uniform_zcoord()
    with pytest.raises(ValueError, match="unknown GMRediConfig.mld_criterion"):
        _nemo_mld("bogus", T_j, S_j, mask, z_coord, _lin_eos, 0.01,
                  g=constants.g, rho_0=RHO0)
