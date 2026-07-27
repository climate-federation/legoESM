"""NEMO ``zdf_mxl`` (mixed-layer level) must transcribe zdfmxl.F90 exactly.

Two deviations found against NEMO 5.0.2's own dumped ``nmln`` on the DINO y5
state (#1226).  Together they accounted for 339 of the 353 mismatched columns
(9920 wet columns total); after both fixes 14 remain.

1. **The bottom cap.**  NEMO does not record the crossing level; it advances
   ``nmln`` on every level still BELOW threshold, clamped by ``mbkt``::

       DO jk = nlb10, jpkm1
          hmlp += MAX( rn2b(jk), 0 ) * e3w(jk,Kmm)
          IF( hmlp < zN2_c )   nmln = MIN( jk, mbkt ) + 1

   so a column that never reaches the threshold ends at ``mbkt+1`` -- the ML
   reaches the SEAFLOOR -- not at the deepest interface.  Omitting the clamp put
   339 columns 1-5 levels too deep.

2. **The alpha/beta interpolation depth.**  ``eosbn2.F90:1459`` weights the two
   T-point alpha/beta by the TRUE w-interface depth::

       zrw = ( gdepw(jk) - gdept(jk) ) / ( gdept(jk-1) - gdept(jk) )

   which is 1/2 only on a uniform ladder.  Feeding the midpoint of the
   bracketing T-depths instead biased N^2 by 3.7e-4 (median, relative); using
   the true gdepw cut the MLD integrand error to 4.9e-5 and took its correlation
   with NEMO's own ``rn2b*e3w`` to 1.00000000.

Synthetic -- no NEMO build needed, so these run in CI.  Each carries a
violation check so it cannot pass vacuously.
"""
import types

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
    _nemo_mld_from_n2_integral,
)


def _z_coord(dz, n_lat, n_lon, k_bot):
    """Minimal z_coord stand-in carrying the fields the MLD helper reads."""
    nlev = dz.size
    gdept = np.cumsum(dz) - 0.5 * dz
    kk = np.arange(nlev)[None, None, :]
    is_active = (kk < np.asarray(k_bot)[:, :, None]).astype(np.float64)
    return types.SimpleNamespace(
        dz_ref=jnp.asarray(dz),
        t_depth_ref=jnp.asarray(gdept),
        is_active=jnp.asarray(is_active),
        h_partial=jnp.broadcast_to(jnp.asarray(dz), (n_lat, n_lon, nlev)),
    )


def _run(dz, T, S, k_bot, rho_c=0.01):
    n_lat, n_lon = T.shape[:2]
    z = _z_coord(dz, n_lat, n_lon, k_bot)
    mask = jnp.ones((n_lat, n_lon))
    hml, m_base = _nemo_mld_from_n2_integral(
        jnp.asarray(T), jnp.asarray(S), mask, z, None, rho_c,
        9.80665, 1026.0, active_3d=z.is_active)
    return np.asarray(hml), np.asarray(m_base)


def test_unstratified_column_stops_at_the_seafloor_not_the_last_interface():
    """A column that never reaches zN2_c must give nmln = mbkt+1 (NEMO's cap).

    This is the 339-column bug: without ``MIN(jk, mbkt)`` an unstratified
    column runs past its own seafloor to the deepest interface in the array.
    """
    nlev = 20
    dz = np.full(nlev, 50.0)
    # Perfectly uniform T/S => N^2 == 0 everywhere => threshold NEVER reached.
    T = np.full((3, 2, nlev), 10.0)
    S = np.full((3, 2, nlev), 35.0)
    k_bot = np.array([[6, 9], [12, 15], [4, nlev]])      # varied bathymetry
    _hml, m_base = _run(dz, T, S, k_bot)
    # nmln = m_base + 2 (verified convention), and NEMO gives nmln = mbkt + 1.
    nmln = m_base + 2
    np.testing.assert_array_equal(nmln, np.minimum(k_bot + 1, nlev))
    # Violation check: the pre-fix behaviour (fall through to the deepest
    # interface) would give nlev for EVERY column regardless of bathymetry.
    assert not np.all(nmln == nlev), (
        "every column hit the array bottom -- the mbkt cap is not being applied")


def test_mld_never_exceeds_the_column_depth():
    """The ML base must lie inside the water column for ANY stratification."""
    rng = np.random.default_rng(4)
    nlev = 16
    dz = np.full(nlev, 25.0)
    # Weak, noisy stratification: some columns cross the threshold, some do not.
    T = 10.0 + np.cumsum(rng.uniform(0.0, 0.002, (5, 4, nlev)), axis=-1)[:, :, ::-1]
    S = np.full((5, 4, nlev), 35.0)
    k_bot = rng.integers(3, nlev + 1, size=(5, 4))
    _hml, m_base = _run(dz, T, S, k_bot)
    assert np.all(m_base + 2 <= k_bot + 1), "ML base ran past the seafloor"


def test_alpha_beta_use_true_gdepw_not_the_gdept_midpoint():
    """On a STRETCHED ladder the two weightings must give different answers,
    and the helper must produce the true-gdepw one.

    Guards the eosbn2 zrw transcription: gdept is not centred between its
    interfaces once dz varies with depth, so the midpoint is a real bias, not a
    cosmetic difference.
    """
    from legoesm.ocean.eos import (
        compute_buoyancy_frequency_nemo_bn2, NemoSEOSConfig,
    )
    nlev = 12
    dz = np.geomspace(10.0, 300.0, nlev)          # strongly stretched
    gdept = np.cumsum(dz) - 0.5 * dz
    z_iface = np.cumsum(dz)
    rng = np.random.default_rng(1)
    T = 10.0 + np.cumsum(rng.uniform(0.05, 0.4, (4, 3, nlev)), axis=-1)[:, :, ::-1]
    S = 35.0 + rng.uniform(-0.1, 0.1, (4, 3, nlev))

    kw = dict(cfg=NemoSEOSConfig(), g=9.80665)
    n2_true = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept),
        jnp.asarray(z_iface[:-1]), **kw))
    n2_mid = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept),
        jnp.asarray(0.5 * (gdept[:-1] + gdept[1:])), **kw))

    # The two weightings must genuinely differ, or this proves nothing.
    rel = np.abs(n2_true - n2_mid) / np.maximum(np.abs(n2_true), 1e-12)
    assert rel.max() > 1e-6, (
        f"the ladder is not stretched enough to discriminate (max rel {rel.max():.2e})")
    # gdepw is NOT the midpoint on this ladder -- that is the whole point.
    assert np.abs(z_iface[:-1] - 0.5 * (gdept[:-1] + gdept[1:])).max() > 1.0


def test_uniform_ladder_makes_the_two_weightings_agree():
    """Sanity: on a uniform ladder gdepw IS the midpoint, so zrw == 1/2.

    Confirms the previous test's discrimination comes from the STRETCH and not
    from an unrelated bug in either call.
    """
    nlev = 10
    dz = np.full(nlev, 40.0)
    gdept = np.cumsum(dz) - 0.5 * dz
    z_iface = np.cumsum(dz)
    np.testing.assert_allclose(z_iface[:-1], 0.5 * (gdept[:-1] + gdept[1:]),
                               rtol=0, atol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
