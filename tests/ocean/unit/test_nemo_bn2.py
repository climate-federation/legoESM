"""NEMO ``bn2`` (S-EOS) Brunt-Väisälä ``N²`` — exact-transcription tests.

Validates :func:`legoesm.ocean.eos.compute_buoyancy_frequency_nemo_bn2` and its
``rab`` companion :func:`legoesm.ocean.eos.nemo_seos_alpha_beta` against an
independent NumPy transcription of NEMO ``eosbn2.F90`` (``rab_3d_t`` +
``bn2_t``, ``np_seos`` branch) on an analytic S-EOS column — to machine
precision — plus the ``n2_mode`` dispatch-raise hardening.
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.ocean.eos import (
    _nemo_bn2_zrw,
    NemoSEOSConfig,
    compute_buoyancy_frequency_nemo_bn2,
    nemo_r3t_stretch,
    nemo_seos_alpha_beta,
    nemo_seos_eos,
)


def _numpy_bn2(T, S, gdept, gdepw_int, cfg, g):
    """Independent NumPy transcription of NEMO eosbn2.F90 bn2_t (np_seos).

    Loops the NEMO w-point recurrence directly (no vectorised broadcasting)
    so a broadcasting/index slip in the JAX version cannot be masked.
    """
    T = np.asarray(T); S = np.asarray(S); gdept = np.asarray(gdept)
    nlev = T.shape[-1]
    # rab alpha/beta at each T-cell's own depth (rab_3d_t np_seos).
    zt = T - cfg.T0; zs = S - cfg.S0; zh = gdept
    alpha = (cfg.a0 * (1.0 + cfg.lambda1 * zt + cfg.mu1 * zh)
             + cfg.nu * zs) / cfg.rho0
    beta = (cfg.b0 * (1.0 - cfg.lambda2 * zs - cfg.mu2 * zh)
            - cfg.nu * zt) / cfg.rho0
    out = np.zeros(T.shape[:-1] + (nlev - 1,))
    for i in range(nlev - 1):
        jk_up, jk_lo = i, i + 1  # NEMO jk-1 (upper), jk (lower)
        zrw = (gdepw_int[i] - gdept[jk_lo]) / (gdept[jk_up] - gdept[jk_lo])
        zaw = alpha[..., jk_lo] * (1.0 - zrw) + alpha[..., jk_up] * zrw
        zbw = beta[..., jk_lo] * (1.0 - zrw) + beta[..., jk_up] * zrw
        e3w = gdept[jk_lo] - gdept[jk_up]
        out[..., i] = g * (
            zaw * (T[..., jk_up] - T[..., jk_lo])
            - zbw * (S[..., jk_up] - S[..., jk_lo])
        ) / e3w
    return out


def _column():
    """A stably-and-unstably layered analytic S-EOS column (nlev=8)."""
    rng = np.random.default_rng(0)
    nlev = 8
    gdepw = np.array([0., 10., 25., 55., 105., 190., 350., 650., 1200.])
    gdept = 0.5 * (gdepw[:-1] + gdepw[1:]) + rng.uniform(-2, 2, nlev)  # non-midpoint
    gdept = np.sort(gdept)
    gdepw_int = gdepw[1:-1]  # interior w-interfaces (nlev-1,)
    # Warm/salty surface, cool/fresh deep — plus one inverted (unstable) pair.
    T = np.linspace(18.0, 3.0, nlev) + rng.uniform(-0.5, 0.5, nlev)
    S = np.linspace(35.5, 34.6, nlev) + rng.uniform(-0.05, 0.05, nlev)
    # Force a statically unstable interface 3: cold-over-warm with the
    # salinity held neutral there so alpha*ΔT (destabilising) dominates.
    T[3] = T[4] - 3.0
    S[3] = S[4]
    return T, S, gdept, gdepw_int


def test_alpha_beta_matches_analytic_seos_derivative():
    """rab alpha,beta == analytic (T,S)-derivatives of the S-EOS density / rho0."""
    import jax
    cfg = NemoSEOSConfig()
    T, S, depth = 12.3, 34.7, 250.0
    p = cfg.rho0 * constants.g * depth  # so nemo_seos_eos recovers zh = depth
    drdT = float(jax.grad(lambda t: nemo_seos_eos(t, S, p, cfg))(T))
    drdS = float(jax.grad(lambda s: nemo_seos_eos(T, s, p, cfg))(S))
    a, b = nemo_seos_alpha_beta(jnp.array(T), jnp.array(S), jnp.array(depth), cfg)
    assert np.allclose(float(a), -drdT / cfg.rho0, rtol=0, atol=1e-15)
    assert np.allclose(float(b), drdS / cfg.rho0, rtol=0, atol=1e-15)


def test_alpha_beta_cross_terms_full_nemo_namelist():
    """Non-zero nu/lambda2/mu2 (NEMO source-default S-EOS set) still equal the
    analytic jax.grad of the density polynomial — locks the cross terms the
    DINO defaults (nu=lambda2=mu2=0) never exercise."""
    import jax
    cfg = NemoSEOSConfig(a0=0.1655, b0=0.76554, lambda1=0.05952,
                         lambda2=5.4914e-4, mu1=1.4970e-4, mu2=1.109e-5,
                         nu=2.4341e-3)
    T, S, depth = 7.6, 34.2, 812.0
    p = cfg.rho0 * constants.g * depth
    drdT = float(jax.grad(lambda t: nemo_seos_eos(t, S, p, cfg))(T))
    drdS = float(jax.grad(lambda s: nemo_seos_eos(T, s, p, cfg))(S))
    a, b = nemo_seos_alpha_beta(jnp.array(T), jnp.array(S), jnp.array(depth), cfg)
    assert np.allclose(float(a), -drdT / cfg.rho0, rtol=0, atol=1e-15)
    assert np.allclose(float(b), drdS / cfg.rho0, rtol=0, atol=1e-15)


def test_bn2_matches_numpy_transcription():
    """Vectorised JAX bn2 == looped NumPy NEMO transcription to machine eps."""
    cfg = NemoSEOSConfig()
    T, S, gdept, gdepw_int = _column()
    g = constants.g
    ref = _numpy_bn2(T, S, gdept, gdepw_int, cfg, g)
    got = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept),
        jnp.asarray(gdepw_int), cfg=cfg, g=g,
        e3w_source="depth_difference"))
    assert got.shape == (len(gdept) - 1,)
    assert np.allclose(got, ref, rtol=0, atol=1e-18), np.max(np.abs(got - ref))


def test_bn2_batched_broadcasts():
    """Leading batch dims + (nlev,) depth ladder broadcast correctly."""
    cfg = NemoSEOSConfig()
    T, S, gdept, gdepw_int = _column()
    Tb = np.broadcast_to(T, (4, 3, len(T))).copy()
    Sb = np.broadcast_to(S, (4, 3, len(S))).copy()
    got = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(Tb), jnp.asarray(Sb), jnp.asarray(gdept),
        jnp.asarray(gdepw_int), cfg=cfg,
        e3w_source="depth_difference"))
    ref = _numpy_bn2(Tb, Sb, gdept, gdepw_int, cfg, constants.g)
    assert got.shape == (4, 3, len(gdept) - 1)
    assert np.allclose(got, ref, atol=1e-18)


def test_bn2_sign_is_convection_trigger():
    """The forced inverted pair (interface 3) is statically UNSTABLE (N²<0)."""
    cfg = NemoSEOSConfig()
    T, S, gdept, gdepw_int = _column()
    n2 = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept),
        jnp.asarray(gdepw_int), cfg=cfg,
        e3w_source="depth_difference"))
    assert n2[3] < 0.0                       # unstable interface fires
    assert (n2[:3] > 0.0).all()              # stable stratification above


def test_convection_n2_mode_dispatch_raises():
    """Unknown enhanced-diffusion n2_mode raises (dispatch hardening)."""
    from legoesm.ocean.physics.convection.enhanced_diffusion import (
        convective_K_A_flag,
    )
    from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
    rho = jnp.ones((2, 2, 4))
    dz = jnp.ones((4,)); J = jnp.ones((2, 2))
    with pytest.raises(ValueError, match="n2_mode"):
        convective_K_A_flag(rho, dz, J,
                            EnhancedDiffusionConfig(n2_mode="bogus"))


def test_nemo_bn2_requires_depth_ladders():
    """enhanced-diffusion nemo_bn2 without the ladders raises (no silent fallback)."""
    from legoesm.ocean.physics.convection.enhanced_diffusion import (
        convective_K_A_flag,
    )
    from legoesm.ocean.physics.convection.config import EnhancedDiffusionConfig
    rho = jnp.ones((2, 2, 4)); dz = jnp.ones((4,)); J = jnp.ones((2, 2))
    T = jnp.ones((2, 2, 4)); S = jnp.ones((2, 2, 4))
    with pytest.raises(ValueError, match="nemo_bn2"):
        convective_K_A_flag(rho, dz, J,
                            EnhancedDiffusionConfig(n2_mode="nemo_bn2"),
                            T=T, S=S)  # t_depth / w_depth missing


def test_other_consumers_reject_nemo_bn2_loudly():
    """Richardson does not thread the bn2 depth ladders and must raise, and
    _shared.compute_N2 without the ladders must raise (catke pass-through) —
    no consumer may silently mis-handle nemo_bn2."""
    from legoesm.ocean.physics.vertical_mixing.richardson import (
        richardson_vertical_mixing,
    )
    from legoesm.ocean.physics.vertical_mixing.config import (
        RichardsonVerticalMixingConfig,
    )
    from legoesm.ocean.physics.vertical_mixing._shared import compute_N2
    with pytest.raises(ValueError, match="n2_mode"):
        richardson_vertical_mixing(
            jnp.ones((2, 2, 4)), jnp.ones((2, 2, 4)), jnp.ones((2, 2, 4)),
            jnp.ones((2, 2, 4)), jnp.ones((2, 2, 4)), None, jnp.ones((2, 2)),
            RichardsonVerticalMixingConfig(n2_mode="nemo_bn2"))
    with pytest.raises(ValueError, match="nemo_bn2"):
        compute_N2(jnp.ones((2, 2, 4)), jnp.ones((2, 2, 3)), 1026.0,
                   T_cell=jnp.ones((2, 2, 4)), S_cell=jnp.ones((2, 2, 4)),
                   n2_mode="nemo_bn2")  # ladders missing


def test_shared_compute_N2_nemo_bn2_branch():
    """_shared.compute_N2 routes nemo_bn2 to the eos bn2 and matches it."""
    from legoesm.ocean.physics.vertical_mixing._shared import compute_N2
    cfg = NemoSEOSConfig()
    T, S, gdept, gdepw_int = _column()
    Tj = jnp.asarray(T[None, :]); Sj = jnp.asarray(S[None, :])
    n2 = np.asarray(compute_N2(
        jnp.zeros_like(Tj), jnp.ones((1, len(T) - 1)), cfg.rho0,
        T_cell=Tj, S_cell=Sj,
        t_depth=jnp.asarray(gdept), w_depth=jnp.asarray(gdepw_int),
        e3w_int=jnp.diff(jnp.asarray(gdept)),
        n2_mode="nemo_bn2"))
    ref = _numpy_bn2(T[None, :], S[None, :], gdept, gdepw_int, cfg, constants.g)
    assert np.allclose(n2, ref, atol=1e-18)


def test_shared_compute_N2_nemo_bn2_jacobian_stretches_live_gdept():
    """#1226: NEMO's bn2_t/rab_3d_t evaluate alpha/beta/e3w at the LIVE
    gdept(Kmm) = gdept_0*J (z-star stretch), not the static reference
    ladder (eosbn2.F90 rab_3d_t ``zh = gdept(ji,jj,jk,Kmm)``; bn2_t divides
    by the LIVE ``e3w(ji,jj,jk,Kmm)``, domzgr_substitute.h90:131). The
    caller stretches ``t_depth``/``w_depth`` by the jacobian BEFORE calling
    ``compute_N2`` -- ``e3w`` is derived internally as ``diff(t_depth)``, so
    the stretch already propagates through alpha/beta AND e3w with no
    separate division needed. Passing a jacobian-stretched ladder must (a)
    match an independent NumPy transcription built on that SAME stretched
    ladder and (b) differ from the static-ladder result -- proving the
    stretch is not a no-op.
    """
    from legoesm.ocean.physics.vertical_mixing._shared import compute_N2
    cfg = NemoSEOSConfig()
    T, S, gdept, gdepw_int = _column()
    Tj = jnp.asarray(T[None, :]); Sj = jnp.asarray(S[None, :])
    J = 1.0003   # representative DINO |eta/H| ~ 1e-4 to 1e-3 z-star stretch
    Jj = jnp.asarray([J])

    n2_live = np.asarray(compute_N2(
        jnp.zeros_like(Tj), jnp.ones((1, len(T) - 1)), cfg.rho0,
        T_cell=Tj, S_cell=Sj,
        t_depth=jnp.asarray(gdept) * J, w_depth=jnp.asarray(gdepw_int) * J,
        e3w_int=jnp.diff(jnp.asarray(gdept)) * J,
        jacobian=Jj, n2_mode="nemo_bn2"))
    ref_live = _numpy_bn2(
        T[None, :], S[None, :], gdept * J, gdepw_int * J, cfg, constants.g,
    )
    assert np.allclose(n2_live, ref_live, atol=1e-15)

    n2_static = np.asarray(compute_N2(
        jnp.zeros_like(Tj), jnp.ones((1, len(T) - 1)), cfg.rho0,
        T_cell=Tj, S_cell=Sj,
        t_depth=jnp.asarray(gdept), w_depth=jnp.asarray(gdepw_int),
        e3w_int=jnp.diff(jnp.asarray(gdept)),
        n2_mode="nemo_bn2"))
    assert not np.allclose(n2_live, n2_static, atol=1e-8)
    # Dominant effect of the stretch is the 1/e3w scaling -> n2_live ~ n2_static/J,
    # with a small residual from alpha/beta's own (thermobaric) J-dependence.
    assert np.allclose(n2_live, n2_static / J, rtol=1e-4)


def test_nemo_bn2_ladder_is_pure_jacobian_stretch_not_eta_shift():
    """#1226: gdept(Kmm) is the PURE multiplicative z-star stretch
    gdept_0*(1+r3t) -- NOT gdept_0*(1+r3t) - eta.

    Traced through the actual Fortran macro NEMO's ``gdept(...)`` expands
    to: domzgr_substitute.h90:139 ``gdept(i,j,k,t) = (DEPt_0(i,j,k)
    Tisf(r3t,risfdep,i,j,t))``; under key_qco WITHOUT key_isf (DINO has no
    ice shelf), ``Tisf(r3,isf,i,j,t) = ) Time(r3,i,j,t)`` (:51), i.e. plain
    ``DEPt_0*(1+r3t)`` -- no ``-eta`` anywhere. ``gdept_z0 = gdept - ssh``
    (:145) is a DIFFERENT named quantity (depth relative to z=0, used only
    for diagnostics via the unused ``DEPT_z0`` macro, zero call sites under
    src/OCE/); eosbn2.F90's ``zh = gdept(ji,jj,jk,Kmm)`` (rab_3d_t) uses the
    plain macro, not ``gdept_z0``. Verified against NEMO's own dumped
    gdept(Kmm): the pure-stretch formula matches to ~1e-8; subtracting eta
    is off by 3-20% -- proving "-eta" would be a REGRESSION, not a fix.
    """
    cfg = NemoSEOSConfig()
    T, S, gdept, gdepw_int = _column()
    Tj = jnp.asarray(T[None, :]); Sj = jnp.asarray(S[None, :])

    ht_0 = 1200.0  # representative DINO column depth [m]
    eta = 0.5      # exaggerated SSH anomaly [m] for a clean test signal
    r3t = eta / ht_0
    J = 1.0 + r3t

    # TRUE NEMO ladder: gdept_0*(1+r3t), pure multiplicative stretch.
    gdept_true = gdept * J
    gdepw_true = gdepw_int * J
    ref_true = _numpy_bn2(T[None, :], S[None, :], gdept_true, gdepw_true,
                          cfg, constants.g)

    from legoesm.ocean.physics.vertical_mixing._shared import compute_N2

    # CORRECT (current code): t_depth*J, no eta shift.
    n2_correct = np.asarray(compute_N2(
        jnp.zeros_like(Tj), jnp.ones((1, len(T) - 1)), cfg.rho0,
        T_cell=Tj, S_cell=Sj,
        t_depth=jnp.asarray(gdept) * J, w_depth=jnp.asarray(gdepw_int) * J,
        e3w_int=jnp.diff(jnp.asarray(gdept)) * J,
        n2_mode="nemo_bn2"))
    assert np.allclose(n2_correct, ref_true, atol=1e-12)

    # An "-eta shift" construction is NOT a better match to the true ladder.
    n2_eta_shifted = np.asarray(compute_N2(
        jnp.zeros_like(Tj), jnp.ones((1, len(T) - 1)), cfg.rho0,
        T_cell=Tj, S_cell=Sj,
        t_depth=jnp.asarray(gdept) * J - eta,
        w_depth=jnp.asarray(gdepw_int) * J - eta,
        e3w_int=jnp.diff(jnp.asarray(gdept)) * J,
        n2_mode="nemo_bn2"))
    err_correct = np.max(np.abs(n2_correct - ref_true) / np.abs(ref_true))
    err_eta_shifted = np.max(
        np.abs(n2_eta_shifted - ref_true) / np.abs(ref_true))
    assert err_correct < 1e-10
    assert err_eta_shifted > 1e-4   # genuinely worse, not a rounding wash


def test_nemo_bn2_live_ladders_uses_local_column_not_zstar_jacobian():
    """#1226: the live ladder is ``gdept_0*(1 + eta/H_bathy)`` -- the LOCAL
    column stretch (NEMO ``r3t = ssh/ht_0``, domqco.F90:160) -- and NOT
    legoESM's z* Jacobian ``compute_ocean_jacobian = (eta + H)/H_max``,
    which is normalised by the GLOBAL maximum depth.

    Measured against NEMO's own ``kt==nit000`` ``gdept(Kmm)`` dump on the
    DINO y5 restart: the local form is off by median 2.5e-8 relative, the
    z*-Jacobian form by 1.1e-1 (a 4.4e6x regression). The two coincide only
    when ``H_bathy == H_max``, so this test uses a shallow column where they
    genuinely differ.
    """
    from legoesm.ocean.eos import nemo_bn2_depth_ladders, nemo_bn2_live_ladders
    from legoesm.ocean.vertical import compute_ocean_jacobian

    class _Z:                      # minimal z_coord stand-in for the ladders
        z_full_ref = -jnp.array([5.0, 20.0, 60.0, 150.0])
        z_half_ref = -jnp.array([0.0, 10.0, 35.0, 100.0, 200.0])
        t_depth_ref = jnp.array([5.0, 20.0, 60.0, 150.0])

    z = _Z()
    t_ref, w_ref = nemo_bn2_depth_ladders(z)
    eta = jnp.array([0.5])
    H = jnp.array([1200.0])        # local column, shallower than any H_max

    t_live, w_live = nemo_bn2_live_ladders(z, eta, H)
    expect = 1.0 + 0.5 / 1200.0
    assert np.allclose(np.asarray(t_live), np.asarray(t_ref) * expect, rtol=0,
                       atol=1e-13)
    assert np.allclose(np.asarray(w_live), np.asarray(w_ref) * expect, rtol=0,
                       atol=1e-13)

    # The z* Jacobian is a DIFFERENT number here -- guard the regression that
    # motivated this helper (using it stretched the ladder by ~0.89, not
    # ~1.0004).
    J = np.asarray(compute_ocean_jacobian(eta, H, _ZStar()))
    assert abs(float(J[0]) - expect) > 1e-3


class _ZStar:
    """OceanZStarCoordinate stand-in: J = (eta + H_bathy)/H_max."""
    linear_free_surface = False
    H_max = 4506.0


def test_nemo_bn2_live_ladders_dry_column_is_inert():
    """H_bathy == 0 (land) must give r3t = 0, not eta/0 -> inf/NaN."""
    from legoesm.ocean.eos import nemo_bn2_live_ladders

    class _Z:
        z_full_ref = -jnp.array([5.0, 20.0])
        z_half_ref = -jnp.array([0.0, 10.0, 30.0])
        t_depth_ref = jnp.array([5.0, 20.0])

    t_live, w_live = nemo_bn2_live_ladders(
        _Z(), jnp.array([0.3, 0.0]), jnp.array([0.0, 100.0]))
    assert np.all(np.isfinite(np.asarray(t_live)))
    assert np.all(np.isfinite(np.asarray(w_live)))
    assert np.allclose(np.asarray(t_live)[0], np.asarray(_Z().t_depth_ref))


def test_native_e3w_default_and_explicit_legacy_are_distinguishable():
    """Default bn2 consumes the independent mesh spacing; legacy is opt-in."""
    T, S, gdept, gdepw = _column()
    derived = np.diff(gdept)
    native = np.nextafter(derived, np.inf)
    default = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept), jnp.asarray(gdepw),
        e3w_int=jnp.asarray(native)))
    legacy = np.asarray(compute_buoyancy_frequency_nemo_bn2(
        jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept), jnp.asarray(gdepw),
        e3w_source="depth_difference"))
    assert not np.array_equal(default, legacy)
    with pytest.raises(ValueError, match="requires raw-mesh e3w_int"):
        compute_buoyancy_frequency_nemo_bn2(
            jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept),
            jnp.asarray(gdepw))
    with pytest.raises(ValueError, match="unknown e3w_source"):
        compute_buoyancy_frequency_nemo_bn2(
            jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept),
            jnp.asarray(gdepw), e3w_source="typo")


def test_bn2_literal_zrw_matches_hand_computed_nemo_source_order():
    """eosbn2:1459 macro products round before subtraction; cancellation is red."""
    upper = np.float64(25.95864807194812)
    lower = np.float64(37.01406258301722)
    wpoint = np.float64(31.428849032898142)
    stretch = np.float64(0.9996588545248404)
    expected = ((wpoint * stretch - lower * stretch)
                / (upper * stretch - lower * stretch))
    got = np.asarray(_nemo_bn2_zrw(
        jnp.asarray([upper * stretch, lower * stretch]),
        jnp.asarray([wpoint * stretch]), evaluation="nemo_literal",
        gdept_0=jnp.asarray([upper, lower]),
        gdepw_0=jnp.asarray([wpoint]),
        stretch=jnp.asarray(stretch)))[0]
    assert got == expected

    # Planted former association: algebraically cancel the common stretch.
    cancelled = (wpoint - lower) / (upper - lower)
    assert cancelled != expected


def test_bn2_literal_path_uses_raw_depth_for_eos_rab_too():
    """The raw-depth multiply must feed both eos_rab and bn2 interpolation."""
    T = jnp.asarray([[12.0, 8.0]])
    S = jnp.asarray([[35.1, 34.9]])
    gdept0 = jnp.asarray([25.95864807194812, 37.01406258301722])
    gdepw0 = jnp.asarray([31.428849032898142])
    stretch = jnp.asarray([0.9996588545248404])
    literal_depth = gdept0 * stretch[..., None]
    # Plant an incompatible preassembled depth: a regression that uses this
    # argument for eos_rab instead of raw_depth*stretch must turn red.
    preassembled = literal_depth.at[..., 0].set(
        literal_depth[..., 0] * (1.0 + 1.0e-10))
    e3w = jnp.asarray([[11.0]])
    got = compute_buoyancy_frequency_nemo_bn2(
        T, S, preassembled, gdepw0 * stretch[..., None],
        e3w_int=e3w, zrw_evaluation="nemo_literal",
        zrw_gdept_0=gdept0, zrw_gdepw_0=gdepw0,
        zrw_stretch=stretch)
    expected = compute_buoyancy_frequency_nemo_bn2(
        T, S, literal_depth, gdepw0 * stretch[..., None],
        e3w_int=e3w, zrw_evaluation="nemo_literal",
        zrw_gdept_0=gdept0, zrw_gdepw_0=gdepw0,
        zrw_stretch=stretch)
    legacy = compute_buoyancy_frequency_nemo_bn2(
        T, S, preassembled, gdepw0 * stretch[..., None],
        e3w_int=e3w, zrw_evaluation="preassembled_live")
    np.testing.assert_array_equal(got, expected)
    assert not np.array_equal(np.asarray(got), np.asarray(legacy))


def test_nemo_reciprocal_r3t_is_selectable_jittable_and_differentiable():
    """Only the explicit selector transcribes domain:158 -> domqco:160."""
    import jax

    class _Z:
        linear_free_surface = False

    eta = jnp.asarray([-0.7711205157415337])
    H = jnp.asarray([2260.386175078943])
    legacy = nemo_r3t_stretch(_Z(), eta, H)
    explicit_legacy = nemo_r3t_stretch(
        _Z(), eta, H, evaluation="quotient")
    assert np.array_equal(np.asarray(legacy), np.asarray(explicit_legacy))

    expected = np.maximum(
        1.0 + np.asarray(eta) * (1.0 / np.asarray(H)), 1.0e-6)
    faithful = jax.jit(lambda e: nemo_r3t_stretch(
        _Z(), e, H, evaluation="nemo_reciprocal"))(eta)
    assert np.array_equal(np.asarray(faithful), expected)
    grad = jax.grad(lambda e: jnp.sum(nemo_r3t_stretch(
        _Z(), e, H, evaluation="nemo_reciprocal")))(eta)
    assert np.all(np.isfinite(np.asarray(grad)))

    with pytest.raises(ValueError, match="unknown r3t evaluation"):
        nemo_r3t_stretch(_Z(), eta, H, evaluation="typo")


def test_native_e3w_coordinate_validation_wrappers_and_ad():
    """Raw mesh fields survive both wrappers; JIT and T/S/eta gradients work."""
    import jax
    from legoesm.ocean.eos import nemo_bn2_live_geometry
    from legoesm.ocean.vertical import (
        create_full_step_coordinate, create_partial_cell_coordinate,
        create_z_star_from_thicknesses,
    )

    dz = np.array([10.0, 20.0, 40.0, 80.0])
    gd = np.array([4.0, 18.0, 48.0, 108.0])
    gw0 = np.array([0.0, 10.0, 30.0, 70.0])
    ew = np.array([8.0, 14.0, 30.0, 60.0])
    with pytest.raises(ValueError, match="unknown nemo_e3w_source"):
        create_z_star_from_thicknesses(dz, nemo_e3w_source="typo")
    with pytest.raises(ValueError, match="trailing dimension"):
        create_z_star_from_thicknesses(dz, nemo_e3w_0_m=ew[:-1])
    with pytest.raises(ValueError, match="finite values > 0"):
        create_z_star_from_thicknesses(dz, nemo_e3w_0_m=ew.at[0].set(-1)
                                       if hasattr(ew, "at") else [-1, 14, 30, 60])
    with pytest.raises(ValueError, match="four-ULP"):
        create_z_star_from_thicknesses(
            dz, nemo_gdept_0_m=gd,
            nemo_e3w_0_m=np.array([8.0, 14.0, 30.0, 61.0]))

    # GYRE's independently evaluated source arrays miss their nominal
    # recurrence by four fp64 ULPs at one level; both raw operands must survive.
    ew_roundoff = ew.copy()
    ew_roundoff[3] = np.nextafter(
        np.nextafter(np.nextafter(np.nextafter(ew[3], np.inf), np.inf), np.inf),
        np.inf,
    )
    z_roundoff = create_z_star_from_thicknesses(
        dz,
        nemo_gdept_0_m=gd,
        nemo_e3w_0_m=ew_roundoff,
    )
    np.testing.assert_array_equal(
        np.asarray(z_roundoff.nemo_e3w_0), ew_roundoff
    )

    z = create_z_star_from_thicknesses(
        dz, t_depth_ref_m=gd, nemo_gdept_0_m=gd,
        nemo_gdepw_0_m=gw0,
        nemo_e3w_0_m=ew)
    partial = create_partial_cell_coordinate(z, jnp.array([150.0]))
    full = create_full_step_coordinate(z, jnp.array([3]))
    for wrapped in (partial, full):
        assert wrapped.nemo_e3w_mesh_reference is True
        assert np.array_equal(np.asarray(wrapped.nemo_gdept_0), gd)
        assert np.array_equal(np.asarray(wrapped.nemo_gdepw_0), gw0)
        assert np.array_equal(np.asarray(wrapped.nemo_e3w_0), ew)

    T = jnp.array([[12.0, 10.0, 7.0, 4.0]])
    S = jnp.array([[35.2, 35.1, 35.0, 34.9]])
    H = jnp.array([150.0])

    def total(Tv, Sv, eta):
        gt, gw, e3w = nemo_bn2_live_geometry(z, eta, H)
        return jnp.sum(compute_buoyancy_frequency_nemo_bn2(
            Tv, Sv, gt, gw, e3w_int=e3w))

    value = jax.jit(total)(T, S, jnp.array([0.2]))
    grads = jax.grad(total, argnums=(0, 1, 2))(T, S, jnp.array([0.2]))
    assert np.isfinite(np.asarray(value)).all()
    assert all(np.isfinite(np.asarray(g)).all() for g in grads)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_native_e3w_kernel_rejects_nonpositive_or_nonfinite(bad):
    """The low-level faithful operand fails closed, including under JIT."""
    import jax

    T, S, gdept, gdepw = _column()
    e3w = jnp.asarray(np.diff(gdept)).at[2].set(bad)

    @jax.jit
    def run(native_e3w):
        return compute_buoyancy_frequency_nemo_bn2(
            jnp.asarray(T), jnp.asarray(S), jnp.asarray(gdept),
            jnp.asarray(gdepw), e3w_int=native_e3w)

    with pytest.raises(Exception, match="finite values > 0"):
        run(e3w)


def test_enhanced_diffusion_nemo_bn2_requires_eta_and_H_bathy():
    """nemo_bn2 without the live-ladder inputs must RAISE, not silently fall
    back to the static (or z*-Jacobian) ladder."""
    from legoesm.ocean.physics.convection.enhanced_diffusion import (
        EnhancedDiffusionConfig, enhanced_diffusion_convection,
    )
    from legoesm.ocean.vertical import create_ocean_z_star

    n, nz = 2, 4
    z = create_ocean_z_star(n_levels=nz, H_max=100.0)
    T = jnp.zeros((n, nz)); S = jnp.full((n, nz), 35.0)
    rho = jnp.full((n, nz), 1025.0); J = jnp.ones((n,))
    cfg = EnhancedDiffusionConfig(n2_mode="nemo_bn2")
    with pytest.raises(ValueError, match="eta and H_bathy"):
        enhanced_diffusion_convection(T, S, rho, z, J, cfg)
