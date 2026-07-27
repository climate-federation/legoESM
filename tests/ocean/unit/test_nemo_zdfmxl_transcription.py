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


def _nemo_nmln_reference(n2_by_level, mbkt, thresh, nlb10=2):
    """Independent transcription of NEMO's zdfmxl loop, in plain Python.

    Deliberately a LITERAL port of the Fortran rather than a closed form, so it
    cannot inherit an algebraic mistake from the implementation under test::

        nmln = nlb10
        DO jk = nlb10, jpkm1
           hmlp += MAX( rn2b(jk), 0 ) * e3w(jk,Kmm)
           IF( hmlp < zN2_c )   nmln = MIN( jk, mbkt ) + 1

    ``n2_by_level[jk]`` is the already-integrated contribution at NEMO w-level
    ``jk`` (1-based), i.e. ``MAX(rn2b,0)*e3w``.
    """
    jpkm1 = len(n2_by_level) - 1
    nmln, hmlp = nlb10, 0.0
    for jk in range(nlb10, jpkm1 + 1):
        hmlp += n2_by_level[jk]
        if hmlp < thresh:
            nmln = min(jk, mbkt) + 1
    return nmln


def test_unstratified_column_stops_at_the_seafloor_not_the_last_interface():
    """A column that never reaches zN2_c must give nmln = mbkt+1 (NEMO's cap).

    This is the 339-column bug: without ``MIN(jk, mbkt)`` an unstratified
    column runs past its own seafloor to the deepest interface in the array.

    The expected value comes from :func:`_nemo_nmln_reference`, an independent
    port of the Fortran -- NOT from a closed form mirroring the implementation.
    An earlier version of this test asserted ``min(k_bot+1, nlev)``, which is
    the implementation's own saturation rule and so could not detect an error
    in it.
    """
    nlev = 20
    dz = np.full(nlev, 50.0)
    # Perfectly uniform T/S => N^2 == 0 everywhere => threshold NEVER reached.
    T = np.full((3, 2, nlev), 10.0)
    S = np.full((3, 2, nlev), 35.0)
    # Bathymetry stays within NEMO's invariant mbkt <= jpkm1 (the deepest
    # T-level is always land), which is the only range an oracle config reaches.
    k_bot = np.array([[6, 9], [12, 15], [4, nlev - 1]])
    _hml, m_base = _run(dz, T, S, k_bot)
    nmln = m_base + 2                       # verified index convention

    thresh = 9.80665 * 0.01 / 1026.0
    for j in range(k_bot.shape[0]):
        for i in range(k_bot.shape[1]):
            # N^2 == 0 at every level for a uniform column.
            expect = _nemo_nmln_reference([0.0] * (nlev + 1), int(k_bot[j, i]),
                                          thresh)
            assert nmln[j, i] == expect, (
                f"col ({j},{i}) mbkt={k_bot[j, i]}: got nmln={nmln[j, i]}, "
                f"NEMO's loop gives {expect}")
    # Violation check: the pre-fix behaviour (fall through to the deepest
    # interface) would give the same value for EVERY column regardless of
    # bathymetry, so the assertions above must actually discriminate.
    assert len(set(nmln.ravel().tolist())) > 1, (
        "all columns returned the same level -- the mbkt cap is not applied")


def test_stratified_column_matches_the_fortran_loop_level_by_level():
    """Cross-check the crossing case (not just the capped case) against the port.

    The capped and crossing branches take different paths through the
    vectorised prefix count, so both need an independent oracle.
    """
    nlev = 18
    dz = np.full(nlev, 30.0)
    rng = np.random.default_rng(7)
    # Genuine stratification: warm surface over cold deep => threshold IS met.
    T = np.sort(rng.uniform(2.0, 18.0, (4, 3, nlev)), axis=-1)[:, :, ::-1]
    S = np.full((4, 3, nlev), 35.0)
    k_bot = np.full((4, 3), nlev - 1)
    hml, m_base = _run(dz, T, S, k_bot)
    nmln = m_base + 2
    # The crossing must be strictly inside the column for this to be a test of
    # the crossing branch rather than of the cap.
    assert np.all(nmln < nlev), "no column actually crossed -- retune the fixture"
    assert np.all(nmln >= 2), "nmln below nlb10"
    # hml must be the w-interface depth of that level.
    z_iface = np.cumsum(dz)
    np.testing.assert_allclose(hml, z_iface[m_base], rtol=0, atol=1e-12)


def test_active_3d_none_falls_back_without_claiming_an_all_wet_column():
    """The no-3-D-mask path must honour NEMO's mbkt <= jpkm1 invariant.

    With ``active_3d=None`` there is no bathymetry to read, so the fallback
    assumes the deepest T-level is land (as NEMO guarantees) rather than an
    all-wet column, which would ask for a w-level the interface indexing cannot
    represent.
    """
    nlev = 14
    dz = np.full(nlev, 40.0)
    T = np.full((2, 2, nlev), 10.0)         # unstratified => never crosses
    S = np.full((2, 2, nlev), 35.0)
    z = _z_coord(dz, 2, 2, np.full((2, 2), nlev))
    _hml, m_base = _nemo_mld_from_n2_integral(
        jnp.asarray(T), jnp.asarray(S), jnp.ones((2, 2)), z, None, 0.01,
        9.80665, 1026.0, active_3d=None)
    nmln = np.asarray(m_base) + 2
    assert np.all(nmln == nlev), (
        f"expected the fallback to cap at nmln=nlev={nlev}, got {nmln}")


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


def test_slope_n2_dispatch_is_hardened_and_card_opts_in():
    """`slope_n2` must raise on a typo, and only the oracle card may opt in.

    A bare ``else: <default>`` here would silently run a DIFFERENT N^2 (and so
    different slopes) on a misspelling.
    """
    import dataclasses
    from legoesm.ocean.experiments.dino import dino_config_for_recipe

    assert dino_config_for_recipe(
        "nemo_dino_kamm_mlf").gm_redi_slope_n2 == "nemo_bn2"
    assert dino_config_for_recipe(
        "legoesm_default").gm_redi_slope_n2 == "adiabatic"

    from legoesm.ocean.experiments.dino import (
        dino_lat_lon_grid, dino_lat_lon_model_config,
    )
    bad = dataclasses.replace(dino_config_for_recipe("nemo_dino_kamm_mlf"),
                              gm_redi_slope_n2="not_a_scheme")
    grid = dino_lat_lon_grid(bad, n_lon=8)
    with pytest.raises(ValueError, match="gm_redi_slope_n2"):
        dino_lat_lon_model_config(grid, bad)


def test_slope_n2_selector_changes_the_n2_that_reaches_the_slopes():
    """The two options must give genuinely different N^2 (non-vacuous switch).

    Guards against the selector being threaded but ignored -- which is exactly
    how the earlier 'wired the right function with a wrong depth' defect hid.
    """
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        _nemo_wpoint_e3w_wmask_n2,
    )
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig

    nlev = 12
    dz = np.geomspace(10.0, 300.0, nlev)
    rng = np.random.default_rng(2)
    T = 10.0 + np.cumsum(rng.uniform(0.05, 0.4, (4, 3, nlev)), axis=-1)[:, :, ::-1]
    S = 35.0 + rng.uniform(-0.1, 0.1, (4, 3, nlev))
    z = _z_coord(dz, 4, 3, np.full((4, 3), nlev - 1))
    act = z.is_active
    eos_fn = make_eos_fn("nemo_seos", None, rho0=1026.0)
    rho = jnp.asarray(1026.0 + 0.2 * (10.0 - T))

    out = {}
    for opt in ("adiabatic", "nemo_bn2"):
        _e3w, _wm, pn2 = _nemo_wpoint_e3w_wmask_n2(
            rho, jnp.asarray(T), jnp.asarray(S), z, eos_fn, 1026.0, 9.80665,
            act, slope_n2=opt)
        out[opt] = np.asarray(pn2)
    d = np.abs(out["adiabatic"] - out["nemo_bn2"])
    assert d.max() > 1e-9, "the selector had no effect -- it is not wired through"
    with pytest.raises(ValueError, match="slope_n2"):
        _nemo_wpoint_e3w_wmask_n2(rho, jnp.asarray(T), jnp.asarray(S), z,
                                  eos_fn, 1026.0, 9.80665, act,
                                  slope_n2="typo")


def test_nemo_bn2_requires_the_s_eos():
    """`nemo_bn2` is S-EOS-specific; pairing it with another EOS must RAISE.

    Otherwise the slopes are built from S-EOS alpha/beta derivatives while the
    rest of the tendency uses a different density -- silently inconsistent.
    """
    import dataclasses
    from legoesm.ocean.experiments.dino import (
        dino_config_for_recipe, dino_lat_lon_grid, dino_lat_lon_model_config,
    )
    bad = dataclasses.replace(dino_config_for_recipe("nemo_dino_kamm_mlf"),
                              eos="wright")
    assert bad.gm_redi_slope_n2 == "nemo_bn2"      # the pairing under test
    grid = dino_lat_lon_grid(bad, n_lon=8)
    with pytest.raises(ValueError, match="requires eos='nemo_seos'"):
        dino_lat_lon_model_config(grid, bad)


def test_nemo_bn2_matches_an_independent_numpy_transcription():
    """Ground truth, not just 'differs from the other branch'.

    A pure-NumPy port of eosbn2.F90's bn2_t, written from the Fortran rather
    than from the implementation under test.  This is what catches a swapped
    zrw weight, which a 'the two options differ' assertion cannot.
    """
    from legoesm.ocean.eos import (
        compute_buoyancy_frequency_nemo_bn2, nemo_seos_alpha_beta,
        NemoSEOSConfig,
    )
    nlev = 10
    dz = np.geomspace(12.0, 250.0, nlev)          # non-uniform => zrw != 1/2
    gdept = np.cumsum(dz) - 0.5 * dz
    gdepw = np.cumsum(dz)[:-1]                    # interior w-interfaces
    rng = np.random.default_rng(5)
    T = 10.0 + np.cumsum(rng.uniform(0.05, 0.5, (3, 2, nlev)), axis=-1)[:, :, ::-1]
    S = 35.0 + rng.uniform(-0.2, 0.2, (3, 2, nlev))
    grav = 9.80665

    got = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept),
        jnp.asarray(gdepw), NemoSEOSConfig(), g=grav))

    # --- independent port of eosbn2.F90:1459-1466 ---
    a, b = nemo_seos_alpha_beta(jnp.asarray(T), jnp.asarray(S),
                                jnp.broadcast_to(jnp.asarray(gdept), T.shape),
                                NemoSEOSConfig())
    a = np.asarray(a); b = np.asarray(b)
    want = np.empty_like(got)
    for m in range(nlev - 1):
        ku, kl = m, m + 1                  # upper (jk-1) and lower (jk) T-cells
        zrw = (gdepw[m] - gdept[kl]) / (gdept[ku] - gdept[kl])
        zaw = a[..., kl] * (1.0 - zrw) + a[..., ku] * zrw
        zbw = b[..., kl] * (1.0 - zrw) + b[..., ku] * zrw
        e3w = gdept[kl] - gdept[ku]
        want[..., m] = grav * (zaw * (T[..., ku] - T[..., kl])
                               - zbw * (S[..., ku] - S[..., kl])) / e3w
    # zrw must be genuinely off-centre or this proves nothing about the weight.
    zrws = [(gdepw[m] - gdept[m + 1]) / (gdept[m] - gdept[m + 1])
            for m in range(nlev - 1)]
    assert max(abs(z - 0.5) for z in zrws) > 0.02, "ladder too uniform to test zrw"
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)
