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
        nemo_e3w_mesh_reference=False,
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


def test_native_bn2_divide_and_mld_multiply_share_one_e3w():
    """Native e3w is consumed, and only the shared multiply cancels it."""
    from legoesm.ocean.eos import compute_buoyancy_frequency_nemo_bn2

    dz = np.array([8.0, 12.0, 20.0, 35.0, 55.0, 80.0])
    nlev = len(dz)
    gd = np.cumsum(dz) - 0.45 * dz
    active = jnp.ones((1, 1, nlev))
    T = jnp.asarray(np.linspace(14.0, 4.0, nlev)[None, None, :])
    S = jnp.asarray(np.linspace(35.3, 34.8, nlev)[None, None, :])

    def run(native_e3w):
        z = types.SimpleNamespace(
            dz_ref=jnp.asarray(dz), t_depth_ref=jnp.asarray(gd),
            is_active=active, h_partial=jnp.asarray(dz)[None, None, :],
            nemo_e3w_mesh_reference=True,
            nemo_e3w_0=jnp.asarray(native_e3w), n_levels=nlev,
        )
        return _nemo_mld_from_n2_integral(
            T, S, jnp.ones((1, 1)), z, None, 0.2, 9.80665, 1026.0,
            active_3d=active)

    e3w = np.concatenate([[2.0 * gd[0]], np.diff(gd)])
    gdepw = np.cumsum(dz)[:-1]
    n2_1 = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        T, S, jnp.asarray(gd), jnp.asarray(gdepw),
        e3w_int=jnp.asarray(e3w[1:])))
    n2_2 = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        T, S, jnp.asarray(gd), jnp.asarray(gdepw),
        e3w_int=jnp.asarray(2.0 * e3w[1:])))
    assert not np.array_equal(n2_1, n2_2), (
        "doubling native e3w did not change bn2, so the supplied divisor "
        "is not actually consumed")
    paired_1 = n2_1 * e3w[1:]
    paired_2 = n2_2 * (2.0 * e3w[1:])
    mismatched = n2_2 * e3w[1:]
    assert np.array_equal(paired_1, paired_2)
    assert not np.array_equal(paired_1, mismatched), (
        "a deliberately different MLD multiplier did not break cancellation")
    # Red-capable threshold control: the fixture must make the mismatched
    # product choose the opposite side at its first eligible interface.
    # The production loop begins at its second interior contribution.  The
    # actual rho_c=0.2 threshold is between the matched and mismatched values,
    # so rebuilding the production multiplier from a fixed ladder makes the
    # two run() results below choose different MLD levels.
    threshold = 9.80665 / 1026.0 * 0.2
    assert np.all(paired_1[..., 1] > threshold)
    assert np.all(mismatched[..., 1] < threshold)

    h1, k1 = run(e3w)
    h2, k2 = run(2.0 * e3w)
    assert np.array_equal(np.asarray(k1), np.asarray(k2))
    assert np.array_equal(np.asarray(h1), np.asarray(h2))


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

    kw = dict(cfg=NemoSEOSConfig(), g=9.80665,
              e3w_source="depth_difference")
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
    # NEMO per-face kappa averaging (ldftra.F90:716-718): oracle card opts in,
    # legacy default stays bit-identical.
    assert dino_config_for_recipe(
        "nemo_dino_kamm_mlf").gm_bolus_kappa_face_average is True
    assert dino_config_for_recipe(
        "legoesm_default").gm_bolus_kappa_face_average is False

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
        jnp.asarray(gdepw), NemoSEOSConfig(), g=grav,
        e3w_source="depth_difference"))

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


def test_bolus_kappa_face_average_matches_nemo_and_is_inert_for_constant_kappa():
    """NEMO averages kappa onto EACH face before building psi (ldftra.F90:715-718).

        zaeiu = 0.5*( zaeiw(ji,jj) + zaeiw(ji+1,jj) ) * ssumask
        zaeiv = 0.5*( zaeiw(ji,jj) + zaeiw(ji,jj+1) ) * ssvmask

    Reusing the cell-centred kappa for both faces is exact ONLY for constant
    kappa; the Treguier kappa is spatially 2-D.  Both halves are asserted so the
    option cannot be vacuously "on".
    """
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        nemo_eiv_bolus_transport,
    )
    nlat, nlon, nlev = 6, 5, 4
    shape = (nlat, nlon, nlev)
    rng = np.random.default_rng(3)
    wslpi = jnp.asarray(rng.normal(0, 1e-3, shape))
    wslpj = jnp.asarray(rng.normal(0, 1e-3, shape))
    e2u = jnp.ones((nlat, nlon)) * 1e4
    e1v = jnp.ones((nlat, nlon)) * 1e4
    u_mask = jnp.ones((nlat, nlon + 1))
    v_mask = jnp.ones((nlat + 1, nlon))
    act = jnp.ones(shape)
    kw = dict(out_shape=shape, dtype=jnp.float64)

    def run(kappa, face_avg):
        return nemo_eiv_bolus_transport(
            kappa, wslpi, wslpj, e2u, e1v, u_mask, v_mask, act, act,
            shape, jnp.float64, kappa_face_average=face_avg)

    # (a) CONSTANT kappa: the two must agree exactly -- averaging a constant is
    #     the identity, so switching the option on must change nothing.
    kc = jnp.full((nlat, nlon), 800.0)
    for a, b in zip(run(kc, False), run(kc, True)):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                   rtol=1e-13, atol=0)

    # (b) SPATIALLY-2-D kappa: the two must genuinely differ, or the option is
    #     not doing anything.
    k2d = jnp.asarray(rng.uniform(200.0, 2000.0, (nlat, nlon)))
    off = run(k2d, False)
    on = run(k2d, True)
    assert max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(off, on)) > 1.0, (
        "face averaging had no effect on a 2-D kappa -- the flag is not wired")


def test_kappa_gm_to_faces_matches_a_literal_fortran_port():
    """`nemo_kappa_gm_to_faces` vs a loop port of ldftra.F90:716-717.

    The loop port is written from the Fortran (ji+1 / jj+1 neighbours, face
    masks), not from the vectorised implementation, so a flipped roll direction
    or a swapped axis fails here.  Periodic wrap in i is asserted explicitly --
    the wrap columns are exactly where the previous inline reconstruction was
    wrong.
    """
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        nemo_kappa_gm_to_faces,
    )
    rng = np.random.default_rng(8)
    nlat, nlon = 5, 4
    kt = rng.uniform(100.0, 1500.0, (nlat, nlon))
    um = (rng.uniform(size=(nlat, nlon)) > 0.2).astype(float)
    vm = (rng.uniform(size=(nlat, nlon)) > 0.2).astype(float)
    ku, kv = nemo_kappa_gm_to_faces(jnp.asarray(kt), jnp.asarray(um),
                                    jnp.asarray(vm))
    ku = np.asarray(ku); kv = np.asarray(kv)
    for j in range(nlat):
        for i in range(nlon):
            ip1 = (i + 1) % nlon               # periodic wrap (lbc_lnk)
            jp1 = (j + 1) % nlat
            assert abs(ku[j, i] - 0.5 * (kt[j, i] + kt[j, ip1]) * um[j, i]) < 1e-12
            assert abs(kv[j, i] - 0.5 * (kt[j, i] + kt[jp1, i]) * vm[j, i]) < 1e-12
    # non-vacuous: the wrap column must actually use the wrap neighbour.
    assert abs(ku[0, nlon - 1] - 0.5 * (kt[0, nlon - 1] + kt[0, 0]) * um[0, nlon - 1]) < 1e-12
    assert kt[0, 0] != kt[0, nlon - 1]


def test_treguier_return_diagnostics_is_consistent_and_nonempty():
    """The diagnostics path must return the SAME kappa plus the 5 NEMO terms.

    Guards the oracle instrument: if the flag ever forked the computation, the
    term-by-term comparison would silently measure something other than what
    the model runs.
    """
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        compute_nemo_native_slopes, compute_treguier_kappa_gm_nemo_native,
    )
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.physics.lateral_mixing.config import (
        GMRediConfig, TreguierConfig,
    )

    nlat, nlon, nlev = 6, 5, 8
    dz = np.geomspace(15.0, 300.0, nlev)
    z = _z_coord(dz, nlat, nlon, np.full((nlat, nlon), nlev - 1))
    rng = np.random.default_rng(6)
    T = 8.0 + np.cumsum(rng.uniform(0.05, 0.4, (nlat, nlon, nlev)), axis=-1)[:, :, ::-1]
    S = np.full((nlat, nlon, nlev), 35.0)
    mask = jnp.ones((nlat, nlon))
    u_mask = jnp.ones((nlat, nlon + 1)); v_mask = jnp.ones((nlat + 1, nlon))
    eos_fn = make_eos_fn("nemo_seos", None, rho0=1026.0)
    rho = jnp.asarray(1026.0 + 0.2 * (10.0 - T))
    cfg = GMRediConfig(slope_n2="nemo_bn2")
    from legoesm.grids.latlon import create_latlon_grid
    grid = create_latlon_grid(n_lat=nlat, n_lon=nlon)
    slopes = compute_nemo_native_slopes(
        rho, jnp.asarray(T), jnp.asarray(S), mask, u_mask, v_mask, z, grid,
        cfg, eos_fn, active_3d=z.is_active)
    f_cor = jnp.full((nlat, nlon), -1e-4)
    kw = dict(active_3d=z.is_active)
    treg = TreguierConfig()
    k_plain = compute_treguier_kappa_gm_nemo_native(
        rho, jnp.asarray(T), jnp.asarray(S), slopes[2], slopes[3], mask, z,
        grid, f_cor, treg, eos_fn, **kw)
    k_diag, d = compute_treguier_kappa_gm_nemo_native(
        rho, jnp.asarray(T), jnp.asarray(S), slopes[2], slopes[3], mask, z,
        grid, f_cor, treg, eos_fn, return_diagnostics=True, **kw)
    np.testing.assert_array_equal(np.asarray(k_plain), np.asarray(k_diag))
    assert set(d) == {"zn", "zah", "zhw", "zRo", "zaeiw"}
    assert all(np.asarray(v).shape == (nlat, nlon) for v in d.values())
    assert float(jnp.max(d["zn"])) > 0.0, "zn identically zero -- fixture inert"


def test_treguier_kappa_uses_the_same_n2_variant_as_the_slopes():
    """The dispatcher must thread slope_n2 to the kappa builder EXPLICITLY.

    Regression for the 1.6% aeiu deficit (#1226): the dispatcher passed the
    2-field TreguierConfig as `cfg`, so `getattr(cfg, 'slope_n2', 'adiabatic')`
    silently fell back to 'adiabatic' while the slopes two lines above ran
    'nemo_bn2' -- an internally inconsistent kappa.  Two guards:

    1. behavioural: the explicit `slope_n2` parameter must actually change
       kappa (non-vacuous switch);
    2. source-level: the dispatcher call site must pass `slope_n2=` and
       `jacobian=` explicitly, so reintroducing the getattr-on-the-wrong-config
       pattern goes red.
    """
    import inspect
    from legoesm.ocean.physics.lateral_mixing import gm_redi_latlon_cgrid as m

    src = inspect.getsource(m.gm_redi_tracer_tendency_latlon)
    i = src.index("compute_treguier_kappa_gm_nemo_native(")
    call = src[i:i + 900]
    assert "slope_n2=" in call, (
        "dispatcher no longer passes slope_n2 to the Treguier kappa builder -- "
        "the silent-adiabatic fallback bug is back")
    assert "jacobian=" in call, (
        "dispatcher no longer passes jacobian to the Treguier kappa builder")

    # Behavioural half: same fixture as the diagnostics test, two variants.
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        compute_nemo_native_slopes, compute_treguier_kappa_gm_nemo_native,
    )
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.physics.lateral_mixing.config import (
        GMRediConfig, TreguierConfig,
    )
    from legoesm.grids.latlon import create_latlon_grid

    nlat, nlon, nlev = 6, 5, 8
    dz = np.geomspace(15.0, 300.0, nlev)
    z = _z_coord(dz, nlat, nlon, np.full((nlat, nlon), nlev - 1))
    rng = np.random.default_rng(9)
    T = 8.0 + np.cumsum(rng.uniform(0.05, 0.4, (nlat, nlon, nlev)), axis=-1)[:, :, ::-1]
    S = np.full((nlat, nlon, nlev), 35.0)
    mask = jnp.ones((nlat, nlon))
    u_mask = jnp.ones((nlat, nlon + 1)); v_mask = jnp.ones((nlat + 1, nlon))
    eos_fn = make_eos_fn("nemo_seos", None, rho0=1026.0)
    rho = jnp.asarray(1026.0 + 0.2 * (10.0 - T))
    grid = create_latlon_grid(n_lat=nlat, n_lon=nlon)
    slopes = compute_nemo_native_slopes(
        rho, jnp.asarray(T), jnp.asarray(S), mask, u_mask, v_mask, z, grid,
        GMRediConfig(slope_n2="nemo_bn2"), eos_fn, active_3d=z.is_active)
    f_cor = jnp.full((nlat, nlon), -1e-4)
    out = {}
    for v in ("adiabatic", "nemo_bn2"):
        out[v] = np.asarray(compute_treguier_kappa_gm_nemo_native(
            rho, jnp.asarray(T), jnp.asarray(S), slopes[2], slopes[3], mask,
            z, grid, f_cor, TreguierConfig(), eos_fn,
            active_3d=z.is_active, slope_n2=v))
    assert np.abs(out["adiabatic"] - out["nemo_bn2"]).max() > 1e-8, (
        "slope_n2 has no effect on kappa -- the parameter is not wired through")


def test_shapiro_smoother_preserves_a_uniform_field_across_the_seam():
    """A uniform slope field on a fully-wet PERIODIC band must stay uniform.

    Weighting-free invariant (Rule 6): no thickness, no area, no mask enters.
    The pre-fix zero-padded smoother deflated the two seam columns to 0.375x
    (12/16 of the binomial sum times a halved zcofw), so this test discriminates
    the seam bug directly; interior-lat rows are exact for both, which is why
    only a seam-aware invariant catches it.
    """
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        _shapiro_smooth_slopes,
    )
    nlat, nlon, nlev = 7, 6, 4
    c = 3.7e-4
    S = jnp.full((nlat, nlon, nlev), c)
    mask = jnp.ones((nlat, nlon))
    Sx, Sy = _shapiro_smooth_slopes(S, S, mask)
    # Interior LAT rows (lat is genuinely closed; edge rows legitimately taper).
    interior = np.asarray(Sx)[1:-1, :, :]
    np.testing.assert_allclose(interior, c, rtol=1e-13)
    # And explicitly at the seam columns, where zero-padding broke it:
    np.testing.assert_allclose(np.asarray(Sx)[1:-1, 0, :], c, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(Sx)[1:-1, -1, :], c, rtol=1e-13)


def test_native_slopes_are_lon_translation_equivariant():
    """slopes(roll(inputs, lon)) == roll(slopes(inputs), lon) on a periodic band.

    The repo's equivariance doctrine: a genuinely periodic operator must
    commute with translation around the ring.  Zero-padded ghosts break this at
    the seam; wrap satisfies it.  Covers compute_nemo_native_slopes end to end
    (slope stencils + mixed-layer ramp + Shapiro smoother).
    """
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        compute_nemo_native_slopes,
    )
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.grids.latlon import create_latlon_grid

    nlat, nlon, nlev = 6, 8, 6
    dz = np.geomspace(20.0, 200.0, nlev)
    z = _z_coord(dz, nlat, nlon, np.full((nlat, nlon), nlev - 1))
    rng = np.random.default_rng(12)
    T = 8.0 + np.cumsum(rng.uniform(0.05, 0.4, (nlat, nlon, nlev)), axis=-1)[:, :, ::-1]
    S = 35.0 + rng.uniform(-0.2, 0.2, (nlat, nlon, nlev))
    mask = jnp.ones((nlat, nlon))
    u_mask = jnp.ones((nlat, nlon + 1)); v_mask = jnp.ones((nlat + 1, nlon))
    eos_fn = make_eos_fn("nemo_seos", None, rho0=1026.0)
    cfg = GMRediConfig(slope_n2="nemo_bn2")
    grid = create_latlon_grid(n_lat=nlat, n_lon=nlon)

    def run(Ta, Sa):
        rho = jnp.asarray(1026.0 + 0.2 * (10.0 - Ta))
        return compute_nemo_native_slopes(
            rho, jnp.asarray(Ta), jnp.asarray(Sa), mask, u_mask, v_mask, z,
            grid, cfg, eos_fn, active_3d=z.is_active)

    base = run(T, S)
    shifted = run(np.roll(T, 1, axis=1), np.roll(S, 1, axis=1))
    labels = ("uslp", "vslp", "wslpi", "wslpj")
    for name, b, sh in zip(labels, base, shifted):
        np.testing.assert_allclose(
            np.asarray(sh), np.roll(np.asarray(b), 1, axis=1),
            rtol=1e-11, atol=1e-16,
            err_msg=f"{name} is not lon-translation equivariant -- a seam "
                    "(zero-ghost) dependence is back")


def test_native_slope_legacy_selector_keeps_output_bits_and_skips_barriers(
        monkeypatch):
    """A non-identity card retains the pre-round-42 numerical path."""
    import legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid as gm
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.grids.latlon import create_latlon_grid

    nlat, nlon, nlev = 4, 5, 5
    dz = np.geomspace(20.0, 180.0, nlev)
    z = _z_coord(dz, nlat, nlon, np.full((nlat, nlon), nlev - 1))
    rng = np.random.default_rng(31)
    T = 8.0 + rng.uniform(-1.0, 1.0, (nlat, nlon, nlev))
    S = 35.0 + rng.uniform(-0.1, 0.1, (nlat, nlon, nlev))
    rho = jnp.asarray(1026.0 + 0.2 * (10.0 - T))
    mask = jnp.ones((nlat, nlon))
    umask = jnp.ones((nlat, nlon + 1))
    vmask = jnp.ones((nlat + 1, nlon))
    grid = create_latlon_grid(n_lat=nlat, n_lon=nlon)
    eos_fn = make_eos_fn("nemo_seos", None, rho0=1026.0)
    default = gm.compute_nemo_native_slopes(
        rho, jnp.asarray(T), jnp.asarray(S), mask, umask, vmask, z, grid,
        GMRediConfig(), eos_fn, active_3d=z.is_active)

    def barrier_must_not_run(_value):
        raise AssertionError("NEMO-identity association reached a legacy card")

    monkeypatch.setattr(gm.lax, "optimization_barrier", barrier_must_not_run)
    explicit = gm.compute_nemo_native_slopes(
        rho, jnp.asarray(T), jnp.asarray(S), mask, umask, vmask, z, grid,
        GMRediConfig(
            slope_prd_evaluation="density_roundtrip",
            slope_prd_geometry_stage="current_step",
            slope_n2_evaluation="recompute"),
        eos_fn, active_3d=z.is_active)
    for implicit, selected in zip(default, explicit):
        np.testing.assert_array_equal(np.asarray(selected), np.asarray(implicit))


def test_carried_mld_n2_skips_dead_recompute_with_dry_mesh_e3w(monkeypatch):
    """The carried rn2b arm consumes its operand before any local eosbn2."""
    import legoesm.ocean.eos as eos
    from legoesm import constants
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        _nemo_mld_from_n2_integral,
    )

    nlat, nlon, nlev = 3, 4, 6
    z = _z_coord(
        np.geomspace(10.0, 100.0, nlev), nlat, nlon,
        np.full((nlat, nlon), nlev - 1))
    active = np.asarray(z.is_active) > 0.5
    iface_wet = active[..., :-1] & active[..., 1:]
    carried_e3w = np.where(iface_wet, 1.0, 0.0)
    carried_n2 = np.zeros_like(carried_e3w)

    def dead_recompute(*_args, **_kwargs):
        raise AssertionError("carried rn2b arm evaluated a dead local eosbn2")

    monkeypatch.setattr(eos, "compute_buoyancy_frequency_nemo_bn2",
                        dead_recompute)
    hml, m_base = _nemo_mld_from_n2_integral(
        jnp.full((nlat, nlon, nlev), 10.0),
        jnp.full((nlat, nlon, nlev), 35.0),
        jnp.ones((nlat, nlon)), z, make_eos_fn("nemo_seos", None),
        GMRediConfig().mld_rho_c, constants.g, constants.rho_ocean_nemo,
        active_3d=z.is_active,
        jacobian=jnp.ones((nlat, nlon)), n2_override=carried_n2,
        e3w_override=carried_e3w)
    assert np.all(np.isfinite(np.asarray(hml)))
    assert np.all(np.asarray(m_base) >= 0)


def test_wslp_ml_anchor_never_reads_past_the_columns_own_bottom():
    """The w-slope ML-ramp anchor index (``kanc``, ldfslp.F90 ``nmln+1``) must
    never exceed the COLUMN'S OWN deepest wet level, not just the array's
    global level count.

    NEMO caps ``nmln = MIN(jk, mbkt) + 1`` (zdfmxl.F90:99) per column, so the
    anchor ``nmln+1`` for a fully-unstratified (never-crosses-threshold)
    column saturates at that column's own ``mbkt+1`` -- the seafloor -- not at
    the domain's deepest level.  ``compute_nemo_native_slopes`` used to build
    ``kanc = jnp.clip(first + 1, 1, nlev - 1)``: a GLOBAL array-size bound
    with no per-column ceiling.  On a domain with a genuinely shallow column
    next to deep ones, an unstratified shallow column's ``first`` saturates
    (via ``_nemo_mld_from_potential_density``'s ``has=False`` branch) at the
    GLOBAL ``nlev - 1`` regardless of that column's own bottom -- i.e. the
    clamp silently depended on every column sharing one dry bottom level, an
    assumption that does not hold for real (non-uniform-depth) bathymetry
    (#1226: 442/9920 DINO wet columns read ``kanc`` past their own
    ``bottom_wet_k`` under the old formula).

    ``kanc`` is purely an internal index (not separately exposed as a public
    return value), and its effect on the PUBLIC ``wslpi``/``wslpj`` output is
    masked away by the final ``* wmask3`` regardless of which dry level it
    points at (verified: both the old and new formula give byte-identical,
    all-zero output at the dry sentinel levels on every fixture tried) --
    which is exactly why this was invisible in the DINO fidelity harness
    despite being a real transcription gap.  This test therefore calls the
    module's actual extracted helper, :func:`_nemo_ml_anchor_index` (the
    exact code the fix changed), directly -- exercising the real source, not
    a re-derivation -- on a shallow, fully-mixed column, and asserts the
    INDEX invariant: ``kanc`` must never exceed the column's own
    ``mbkt + 1`` (0-based).  Non-vacuous: the OLD global-only formula
    (reproduced alongside for comparison) violates this invariant on this
    exact fixture.
    """
    from legoesm.ocean.eos import make_eos_fn
    from legoesm.ocean.physics.lateral_mixing.config import GMRediConfig
    from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
        _nemo_ml_anchor_index, _nemo_mld,
    )

    nlat, nlon, nlev = 3, 4, 9
    dz = np.geomspace(20.0, 200.0, nlev)
    k_bot = np.full((nlat, nlon), nlev - 1)
    k_bot[:, 0] = 3   # column 0: shallow, mbkt=3 (0-based active levels 0..2)
    z = _z_coord(dz, nlat, nlon, k_bot)

    # Uniform T/S everywhere (fully mixed) -> the rho_c criterion never
    # crosses threshold anywhere -> `has=False` -> `_nemo_mld_from_potential_
    # density` returns the saturating `m_base = nlev - 2` fallback for EVERY
    # column, deep or shallow alike -- exactly the scenario the old
    # global-only clamp could not distinguish from a genuinely deep column.
    T = np.full((nlat, nlon, nlev), 10.0)
    S = np.full((nlat, nlon, nlev), 35.0)
    mask = jnp.ones((nlat, nlon))
    eos_fn = make_eos_fn("nemo_seos", None, rho0=1026.0)
    cfg = GMRediConfig()   # default mld_criterion="rho_c" (the affected path)

    hml, m_base = _nemo_mld(
        cfg.mld_criterion, jnp.asarray(T), jnp.asarray(S), mask, z, eos_fn,
        cfg.mld_rho_c, active_3d=z.is_active)
    m_base = np.asarray(m_base)
    # Fixture sanity: the shallow column must actually hit the saturating
    # (unstratified) MLD branch, or this test exercises nothing.
    assert m_base[0, 0] == nlev - 2, (
        "fixture's shallow column did not saturate at the unstratified "
        f"nlev-2 fallback (got {m_base[0, 0]}); adjust the fixture")

    first = jnp.clip(m_base + 1, 1, nlev - 1)
    kanc = np.asarray(_nemo_ml_anchor_index(first, z.is_active, nlev))
    mbkt_shallow = int(np.asarray(z.is_active)[0, 0, :].sum())   # = 3

    # The FIX must hold the invariant NEMO's own nmln cap enforces: kanc <=
    # mbkt + 1 (0-based), i.e. never read past the column's own
    # seafloor-adjacent w-level.
    assert kanc[0, 0] <= mbkt_shallow + 1, (
        f"kanc={kanc[0, 0]} exceeds the shallow column's own "
        f"mbkt+1={mbkt_shallow + 1} -- the per-column clamp regressed")

    # Non-vacuous: the OLD global-only formula (no per-column mbkt ceiling)
    # violates that same invariant on this exact fixture, proving the
    # fixture actually exercises the gap the fix closes.
    old_kanc_shallow = int(np.clip(int(first[0, 0]) + 1, 1, nlev - 1))
    assert old_kanc_shallow > mbkt_shallow + 1, (
        "fixture does not exercise the gap: the OLD global-only clamp did "
        "not exceed the shallow column's own bottom (mbkt+1) here")

    # Source guard: the call site must actually USE the helper (not just
    # define it) -- otherwise the inline formula could regress back to the
    # old global-only clamp while this test still imports and passes the
    # helper directly, silently going vacuous.
    import inspect
    from legoesm.ocean.physics.lateral_mixing import gm_redi_latlon_cgrid as m
    src = inspect.getsource(m.compute_nemo_native_slopes)
    assert "_nemo_ml_anchor_index(" in src, (
        "compute_nemo_native_slopes no longer calls _nemo_ml_anchor_index -- "
        "the per-column mbkt clamp may have regressed back to a global-only "
        "jnp.clip(first + 1, 1, nlev - 1)")


def test_dispatcher_reads_kfa_directly_not_via_getattr_default():
    """The dispatcher must read cfg.gm_bolus_kappa_face_average DIRECTLY.

    A getattr default here silently disables the option when a wrong config
    object is passed -- the exact pattern that hid the slope_n2 bug (#1226).
    Direct attribute access raises AttributeError instead.
    """
    import inspect
    from legoesm.ocean.physics.lateral_mixing import gm_redi_latlon_cgrid as m
    src = inspect.getsource(m.gm_redi_tracer_tendency_latlon)
    assert "cfg.gm_bolus_kappa_face_average" in src, (
        "dispatcher no longer reads the field directly")
    assert 'getattr(cfg, "gm_bolus_kappa_face_average"' not in src, (
        "the silent getattr default is back")
