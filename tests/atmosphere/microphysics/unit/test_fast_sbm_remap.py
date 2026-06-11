"""Fast-SBM growth remap (oracle JERDFUN_KS/JERNEWF_KS).

Invariant set: identity at m_new == masses; exact one-bin shift under mass
doubling; KO conservation (post-remap spectrum number and mass equal the
post-GROWTH values — condensation adds vapor mass by design); evaporation
loses number monotonically, never negative, and disables the 3-point and
tail-merge passes; the drop-tail merge folds spurious tails; gradients flow
from f_new back to S_int.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.microphysics.fast_sbm import (
    discretize_lognormal,
    drop_growth_coefficient,
    mass_doubling_grid,
)
from legoesm.atmosphere.physics.microphysics.fast_sbm.remap import (
    COEFF_REMAPING,
    KRDROP_REMAP_HI,
    KRDROP_REMAP_LO,
    condensation_new_masses,
    remap_spectrum,
)

jax.config.update("jax_enable_x64", True)


def _spectrum():
    m = mass_doubling_grid()
    f = discretize_lognormal(m, 1.0e8, 10.0e-6, 1.4)
    return m, f


def _number(f, m):
    return float(jnp.sum(f * m) * np.log(2.0))


def _mass(f, m):
    return float(jnp.sum(f * m * m) * np.log(2.0))


def test_identity_when_masses_unchanged():
    m, f = _spectrum()
    out = remap_spectrum(f, m, m)
    np.testing.assert_array_equal(np.asarray(out.f_new), np.asarray(f))


def test_exact_one_bin_shift_on_doubling():
    m, f = _spectrum()
    # Keep the top empty so nothing hits the sentinel.
    f = f.at[-6:].set(0.0)
    out = remap_spectrum(f, m, 2.0 * m, three_point=False,
                         drop_tail_merge=False)
    psi_in = np.asarray(f * m)
    psi_out = np.asarray(out.f_new * m)
    np.testing.assert_allclose(psi_out[1:], psi_in[:-1], rtol=1e-12)
    assert psi_out[0] == 0.0


def test_ko_conserves_packets_and_grown_mass():
    m, f = _spectrum()
    f = f.at[-8:].set(0.0)                  # stay far from the grid top
    s_int = 8.0e-4                          # supersaturated substep [s]
    B = drop_growth_coefficient(m, jnp.asarray(283.0), jnp.asarray(9.0e4),
                                jnp.zeros_like(m))
    m_new = condensation_new_masses(m, B, jnp.asarray(s_int))
    assert np.all(np.asarray(m_new) > np.asarray(m))   # condensation grows
    for three_point in (False, True):
        out = remap_spectrum(f, m, m_new, three_point=three_point,
                             drop_tail_merge=False)
        # Number: Σψ conserved.
        np.testing.assert_allclose(
            float(jnp.sum(out.f_new * m)), float(jnp.sum(f * m)),
            rtol=1e-12)
        # Mass: post-remap mass equals post-GROWTH mass (vapor was added).
        np.testing.assert_allclose(
            float(jnp.sum(out.f_new * m * m)),
            float(jnp.sum(f * m * m_new)), rtol=1e-12)
        assert float(out.min_psi) >= 0.0


def test_evaporation_loses_number_never_negative():
    m, f = _spectrum()
    B = drop_growth_coefficient(m, jnp.asarray(283.0), jnp.asarray(9.0e4),
                                jnp.zeros_like(m))
    m_new = condensation_new_masses(m, B, jnp.asarray(-2.0e-3))  # subsat
    assert float(m_new[0]) < float(m[0])    # evaporation detected
    out = remap_spectrum(f, m, m_new)
    assert np.all(np.asarray(out.f_new) >= 0.0)
    assert _number(out.f_new, m) <= _number(f, m) * (1.0 + 1e-12)
    # Mass shrinks toward the grown (smaller) total.
    assert _mass(out.f_new, m) < _mass(f, m)


def test_condensation_new_masses_limits():
    m, _ = _spectrum()
    B = drop_growth_coefficient(m, jnp.asarray(283.0), jnp.asarray(9.0e4),
                                jnp.zeros_like(m))
    # (m^{2/3})^{3/2} round-trip costs ~2 ulp — within the remap's
    # exact-match shortcut (1e-19 kg) so downstream still sees identity.
    np.testing.assert_allclose(
        np.asarray(condensation_new_masses(m, B, jnp.asarray(0.0))),
        np.asarray(m), rtol=1e-13)
    grown = condensation_new_masses(m, B, jnp.asarray(1.0e-3))
    assert np.all(np.asarray(grown) > np.asarray(m))
    # Catastrophic evaporation overshoot floors at the oracle tiny mass.
    shrunk = condensation_new_masses(m, B, jnp.asarray(-1.0e3))
    assert np.all(np.asarray(shrunk) > 0.0)


def test_drop_tail_merge_folds_spurious_tail():
    m, _ = _spectrum()
    f = jnp.zeros_like(m)
    k = KRDROP_REMAP_LO + 2
    f = f.at[k].set(1.0e12)
    # Right neighbour far below the 1/150 mass-content threshold.
    f = f.at[k + 1].set(1.0e12 * COEFF_REMAPING * 0.05)
    # Nudge masses so the remap path actually runs (not the identity).
    m_new = m * (1.0 + 1.0e-12)
    out = remap_spectrum(f, m, m_new, three_point=False,
                         drop_tail_merge=True)
    fn = np.asarray(out.f_new)
    assert fn[k + 1] == 0.0                       # folded away
    # Folded INTO the left neighbour: window mass content conserved.
    lo, hi = KRDROP_REMAP_LO, KRDROP_REMAP_HI
    np.testing.assert_allclose(
        float(jnp.sum(out.f_new[lo:hi + 1] * m[lo:hi + 1])),
        float(jnp.sum(f[lo:hi + 1] * m[lo:hi + 1])), rtol=1e-9)


def test_exact_match_overwrite_semantics():
    # Oracle KO: an exact-match source ASSIGNS PSINEW(K)=FI(K)*RR(K),
    # overwriting earlier deposits into K; later sources still add on top.
    # Construct: source bin 4 grows onto bin 5's center region BEFORE the
    # exact source 5 is visited; oracle semantics → bin-5 packet equals
    # f5*m5 plus only the deposits from sources AFTER 5 (none here), i.e.
    # the bin-4 deposit into 5 is overwritten.
    m, _ = _spectrum()
    f = jnp.zeros_like(m).at[4].set(1.0e12).at[5].set(2.0e12)
    m_new = jnp.asarray(m)
    m_new = m_new.at[4].set(float(m[5]))      # bin 4 lands exactly on 5
    # (all other bins exact-match: unchanged masses)
    out = remap_spectrum(f, m, m_new, three_point=False,
                         drop_tail_merge=False)
    psi_out = np.asarray(out.f_new * m)
    # Bin 4's packet went to bin 5 first (boundary hit, then the
    # exact-match source 5 OVERWROTE it) → bin 5 holds only its own packet.
    assert psi_out[5] == pytest.approx(float(f[5] * m[5]), rel=1e-12)
    assert psi_out[4] == 0.0


def test_boundary_equality_lands_lower_interval():
    # m_new exactly on a grid node must use the LOWER interval (oracle
    # first-match linear search): all weight goes to that node, none below.
    m, _ = _spectrum()
    f = jnp.zeros_like(m).at[7].set(1.0e12)
    m_new = jnp.asarray(m).at[7].set(float(m[8]))
    out = remap_spectrum(f, m, m_new, three_point=False,
                         drop_tail_merge=False)
    psi_out = np.asarray(out.f_new * m)
    assert psi_out[8] == pytest.approx(float(f[7] * m[7]), rel=1e-12)
    assert psi_out[7] == 0.0


def test_negative_new_mass_kills_packet():
    # Oracle pre-step: RN < 0 → RN = tiny, FI = 0 (packet destroyed).
    m, _ = _spectrum()
    f = jnp.zeros_like(m).at[3].set(1.0e12)
    m_new = jnp.asarray(m).at[3].set(-1.0e-15)
    out = remap_spectrum(f, m, m_new, three_point=False,
                         drop_tail_merge=False)
    np.testing.assert_array_equal(np.asarray(out.f_new), 0.0)


def test_sentinel_overflow_fraction_is_lost():
    # A top-bin packet growing past m_top splits between the top bin and
    # the discarded sentinel slot — the sentinel fraction is LOST (oracle
    # drops PSINEW(NRX+1)); the retained fraction matches the KO weight.
    m, _ = _spectrum()
    n = m.shape[0]
    f = jnp.zeros_like(m).at[n - 1].set(1.0e6)
    grow = 2.0
    m_new = jnp.asarray(m).at[n - 1].set(grow * float(m[n - 1]))
    out = remap_spectrum(f, m, m_new, three_point=False,
                         drop_tail_merge=False)
    psi_in = float(f[n - 1] * m[n - 1])
    w_keep = (1024.0 - grow) / (1024.0 - 1.0)
    assert float(out.f_new[n - 1] * m[n - 1]) == pytest.approx(
        psi_in * w_keep, rel=1e-12)


def test_empty_tail_window_merge_noop():
    # Tail merge with an empty drop window must be a no-op (oracle KMAX
    # falls back to the window floor).
    m, _ = _spectrum()
    f = jnp.zeros_like(m).at[20].set(1.0e10)   # outside [5..11]
    m_new = jnp.asarray(m) * (1.0 + 1.0e-12)
    out = remap_spectrum(f, m, m_new, three_point=False,
                         drop_tail_merge=True)
    out_off = remap_spectrum(f, m, m_new, three_point=False,
                             drop_tail_merge=False)
    np.testing.assert_array_equal(np.asarray(out.f_new),
                                  np.asarray(out_off.f_new))


def test_evaporation_disables_three_point_and_merge():
    # Under evaporation (m_new[0] < masses[0]) the 3-point and tail-merge
    # paths must be inert: all four gate combinations agree exactly.
    m, f = _spectrum()
    m_new = jnp.asarray(m) * 0.95              # uniform shrink
    outs = [remap_spectrum(f, m, m_new, three_point=tp, drop_tail_merge=dm)
            for tp in (False, True) for dm in (False, True)]
    base = np.asarray(outs[0].f_new)
    for o in outs[1:]:
        np.testing.assert_array_equal(np.asarray(o.f_new), base)


def test_remap_differentiable_through_s_int():
    m, f = _spectrum()
    f = f.at[-8:].set(0.0)
    B = drop_growth_coefficient(m, jnp.asarray(283.0), jnp.asarray(9.0e4),
                                jnp.zeros_like(m))

    def grown_mass(s_int):
        m_new = condensation_new_masses(m, B, s_int)
        out = remap_spectrum(f, m, m_new)
        return jnp.sum(out.f_new * m * m)

    g = jax.grad(grown_mass)(jnp.asarray(5.0e-4))
    assert np.isfinite(float(g))
    assert float(g) > 0.0    # more integrated supersaturation → more mass


def test_jit_and_vmap_columns():
    m, f = _spectrum()
    B = drop_growth_coefficient(m, jnp.asarray(283.0), jnp.asarray(9.0e4),
                                jnp.zeros_like(m))
    s_int = jnp.array([2.0e-4, 0.0, -2.0e-4])
    fs = jnp.broadcast_to(f, (3, m.shape[0]))

    @jax.jit
    def step(fcol, s):
        m_new = condensation_new_masses(m, B, s)
        return remap_spectrum(fcol, m, m_new).f_new

    out = jax.vmap(step)(fs, s_int)
    assert out.shape == (3, m.shape[0])
    assert np.all(np.isfinite(np.asarray(out)))
    # Middle column (s_int = 0) is the identity.
    np.testing.assert_array_equal(np.asarray(out[1]), np.asarray(f))
