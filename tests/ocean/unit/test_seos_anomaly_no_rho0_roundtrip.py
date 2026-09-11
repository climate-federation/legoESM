"""The S-EOS anomaly must never be recovered by subtracting rho0 back off rho.

NEMO forms the density ANOMALY directly (``eosbn2.F90:365-367``, compiled at
``cfgs/DINO/BLD/ppsrc/nemo/eosbn2.f90:365-367``) and scales it by ``r1_rho0``
once at ``:369``.  It never materialises ``rho0 + zn``.  A consumer that does,
and then subtracts ``rho0`` back to get the anomaly, rounds the sum to 53 bits
of ~1026 and hands back an anomaly carrying ~1.1e-16 kg/m3 of that rounding.

Measured against NEMO's own ``rhd`` restart field over the DINO domain that is
1.3403e-13 of the anomaly's rms on 341964 of 342134 wet cells -- see
``scripts/validate/ocean_fidelity/dino_1226/seos_restart_density_gate.py``.

Each test below carries its own synthetic-violation check: the assertion is
shown to FAIL for the construction it exists to forbid, so none of them can
pass vacuously.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.eos import (
    NemoSEOSConfig,
    make_eos_fn,
    nemo_seos_anomaly,
    nemo_seos_eos,
    nemo_seos_prd_literal,
)

jax.config.update("jax_enable_x64", True)


def _state(n=4096, seed=0):
    """A T/S/depth field spanning DINO's own range."""
    rng = np.random.default_rng(seed)
    T = rng.uniform(-1.8, 28.0, n)
    S = rng.uniform(33.0, 37.5, n)
    d = rng.uniform(0.0, 4400.0, n)
    return jnp.asarray(T), jnp.asarray(S), jnp.asarray(d)


def _numpy_zn(T, S, d, cfg):
    """eosbn2.f90:365-367, written here from the source, not imported."""
    zt = np.asarray(T) - cfg.T0
    zs = np.asarray(S) - cfg.S0
    zh = np.asarray(d)
    return (-cfg.a0 * (1.0 + 0.5 * cfg.lambda1 * zt + cfg.mu1 * zh) * zt
            + cfg.b0 * (1.0 - 0.5 * cfg.lambda2 * zs - cfg.mu2 * zh) * zs
            - cfg.nu * zt * zs)


def test_anomaly_is_bit_exact_against_the_source_expression():
    cfg = NemoSEOSConfig()
    T, S, d = _state()
    got = np.asarray(nemo_seos_anomaly(T, S, d, cfg))
    want = _numpy_zn(T, S, d, cfg)
    assert got.dtype == np.float64
    assert np.array_equal(got, want), (
        f"{int((got != want).sum())} of {got.size} values differ; "
        f"max|d| = {np.abs(got - want).max():.3e}")


def test_subtracting_rho0_does_NOT_recover_the_anomaly():
    """The synthetic violation for the test above: prove the round trip loses
    bits, so ``array_equal`` there is a real constraint and not a tautology."""
    cfg = NemoSEOSConfig()
    T, S, d = _state()
    exact = np.asarray(nemo_seos_anomaly(T, S, d, cfg))
    p = (cfg.rho0 * constants.g) * d
    roundtrip = np.asarray(nemo_seos_eos(T, S, p, cfg)) - cfg.rho0
    n_bad = int((roundtrip != exact).sum())
    assert n_bad > 0.5 * exact.size, (
        "the rho0 round trip was expected to perturb most values; it "
        f"perturbed {n_bad} of {exact.size}. If this now passes bit-exactly "
        "the platform's arithmetic changed and the fix it guards is moot")
    rel = np.abs(roundtrip - exact).max() / np.sqrt(np.mean(exact ** 2))
    assert rel > 1e-16, f"round-trip error {rel:.3e} is implausibly small"


def test_eos_is_the_anomaly_plus_rho0_bit_for_bit():
    """One copy of the polynomial, not two: ``nemo_seos_eos`` must be exactly
    ``rho0 + nemo_seos_anomaly`` so the two can never drift apart."""
    cfg = NemoSEOSConfig()
    T, S, d = _state(seed=1)
    p = (cfg.rho0 * constants.g) * d
    rho = np.asarray(nemo_seos_eos(T, S, p, cfg))
    zh = np.asarray(p) / (cfg.rho0 * constants.g)
    assert np.array_equal(
        rho, cfg.rho0 + np.asarray(nemo_seos_anomaly(T, S, zh, cfg)))


def test_prd_literal_agrees_with_the_anomaly_scaled_by_r1_rho0():
    """The ordered/barriered transcription and the single expression are the
    SAME statement; ``eosbn2.f90:369`` is the only thing between them."""
    cfg = NemoSEOSConfig()
    T, S, d = _state(seed=2)
    prd = np.asarray(nemo_seos_prd_literal(T, S, d, cfg))
    want = np.asarray(nemo_seos_anomaly(T, S, d, cfg)) * (1.0 / cfg.rho0)
    assert np.array_equal(prd, want), (
        f"max|d| = {np.abs(prd - want).max():.3e} on "
        f"{int((prd != want).sum())} of {prd.size} values")


def test_make_eos_fn_publishes_the_seos_coefficients():
    """A consumer that wants the anomaly must be able to ASK for it.  Only the
    S-EOS branch may advertise this -- an unrelated EOS carrying the attribute
    would send a caller down the S-EOS path with the wrong polynomial."""
    cfg = NemoSEOSConfig(rho0=1030.0, a0=0.2)
    fn = make_eos_fn("nemo_seos", eos_nemo_seos=cfg)
    assert getattr(fn, "nemo_seos_cfg", None) is cfg
    for other in ("wright", "linear", "nemo_eos80", "nemo_teos10"):
        assert getattr(make_eos_fn(other), "nemo_seos_cfg", None) is None, (
            f"{other} advertises S-EOS coefficients it does not use")


def test_published_coefficients_are_the_ones_the_callable_evaluates():
    """The advertised coefficients must be the callable's OWN, or a consumer
    that asks for the anomaly gets a different polynomial than the one the
    density came from.

    Note the S-EOS takes its reference density through ``eos_nemo_seos``, NOT
    through ``make_eos_fn``'s ``rho0=`` -- that argument is documented for the
    ``nemo_eos80``/``nemo_teos10`` depth reconstruction and the S-EOS branch
    ignores it.  Pinned here because a caller passing ``rho0=`` and expecting
    the S-EOS to honour it would silently get 1026.
    """
    fn = make_eos_fn("nemo_seos", rho0=1035.0)
    assert fn.nemo_seos_cfg.rho0 == NemoSEOSConfig().rho0, (
        "make_eos_fn's rho0= now reaches the S-EOS branch; "
        "ocean_pe_latlon_cgrid's geometric path relies on it NOT doing so "
        "and must be re-read")
    cfg = NemoSEOSConfig(rho0=1035.0, a0=0.2)
    fn2 = make_eos_fn("nemo_seos", eos_nemo_seos=cfg)
    T, S, d = _state(seed=3)
    p = (cfg.rho0 * constants.g) * d
    assert np.array_equal(
        np.asarray(fn2(T, S, p)),
        cfg.rho0 + np.asarray(nemo_seos_anomaly(T, S, d, fn2.nemo_seos_cfg)))


def test_anomaly_is_jittable_and_differentiable():
    """JIT must not change the VALUE beyond the backend's own contraction.

    MEASURED on the DINO record (342134 wet cells), eager vs a numpy
    transcription of eosbn2.f90:365-367:

        GPU  eager 0.0          jit 0.0          -- bit-exact, and the GPU is
                                                    where the oracle runs
        CPU  eager 0.0          jit 9.9920e-16 on 110845 cells (1.1757e-15 of
                                                    the anomaly's rms)

    So XLA's CPU backend contracts this expression (FMA) and its GPU backend
    does not.  That is a property of the platform, not of the statement, and
    it is a floor on any CPU-side comparison -- never a licence to widen the
    GPU one.  The assertion is therefore a few-ulp bound, not bit-equality.
    """
    cfg = NemoSEOSConfig()
    T, S, d = _state(n=4096, seed=4)
    eager = np.asarray(nemo_seos_anomaly(T, S, d, cfg))
    jitted = np.asarray(jax.jit(
        lambda t, s, z: nemo_seos_anomaly(t, s, z, cfg))(T, S, d))
    scale = np.sqrt(np.mean(eager ** 2))
    rel = np.abs(jitted - eager).max() / scale
    assert rel < 1e-14, f"jit/eager disagree by {rel:.3e} of rms, far above a "\
                        "contraction: the traced expression is not the same one"
    assert np.array_equal(eager, _numpy_zn(T, S, d, cfg)), (
        "the EAGER value must be the source expression bit for bit on every "
        "backend; only the jitted one may contract")
    g = jax.grad(lambda t: jnp.sum(nemo_seos_anomaly(t, S, d, cfg)))(T)
    g = np.asarray(g)
    assert np.isfinite(g).all() and np.abs(g).max() > 0.0
    # d(zn)/dT = -a0 (1 + lambda1 zt + mu1 zh) - nu zs, i.e. -alpha*rho0.
    zt = np.asarray(T) - cfg.T0
    zs = np.asarray(S) - cfg.S0
    want = -(cfg.a0 * (1.0 + cfg.lambda1 * zt + cfg.mu1 * np.asarray(d))
             + cfg.nu * zs)
    np.testing.assert_allclose(g, want, rtol=1e-13, atol=0.0)


@pytest.mark.parametrize("rho0", [1026.0, 1000.0])
def test_pressure_anomaly_helper_uses_the_exact_anomaly(rho0):
    """``iterate_eos_and_pressure_anomaly``'s geometric branch must return the
    EXACT anomaly, not ``rho - rho_0``; and the synthetic violation is the
    same helper driven by an EOS that does NOT publish its coefficients, which
    must fall back to the subtraction and therefore differ."""
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        iterate_eos_and_pressure_anomaly,
    )
    rng = np.random.default_rng(5)
    ny, nx, nz = 3, 4, 6
    T = jnp.asarray(rng.uniform(-1.0, 25.0, (ny, nx, nz)))
    S = jnp.asarray(rng.uniform(33.5, 37.0, (ny, nx, nz)))
    mask = jnp.ones((ny, nx), dtype=bool)
    dz = jnp.full((nz,), 100.0)
    depth = jnp.cumsum(dz) - 0.5 * dz
    cfg = NemoSEOSConfig(rho0=rho0)

    def run(eos_fn):
        return iterate_eos_and_pressure_anomaly(
            T, S, mask, lambda f: f, eos_fn, dz, rho0, constants.g,
            n_iter=2, eos_depth="geometric", eos_geometric_depth_1d=depth,
        )

    published = make_eos_fn("nemo_seos", eos_nemo_seos=cfg)
    _rho, rho_prime, _p = run(published)
    want = np.asarray(nemo_seos_anomaly(T, S, depth, cfg))
    assert np.array_equal(np.asarray(rho_prime), want), (
        f"max|d| = {np.abs(np.asarray(rho_prime) - want).max():.3e}")

    # Synthetic violation: strip the published coefficients.  The helper can
    # no longer ask for the anomaly, falls back to rho - rho_0, and the row
    # this test pins must then FAIL.
    def anonymous(T_, S_, p_, compute_dtype=None):
        return published(T_, S_, p_, compute_dtype=compute_dtype)
    _rho2, rho_prime2, _p2 = run(anonymous)
    assert not np.array_equal(np.asarray(rho_prime2), want), (
        "the subtraction fallback reproduced the exact anomaly, so this test "
        "cannot distinguish the two and proves nothing")


def test_helper_refuses_a_published_rho0_that_disagrees_with_the_caller():
    """``rho_0`` and ``cfg.rho0`` are two names for one number.  If they
    disagree the anomaly and the depth reconstruction belong to different
    equations of state, and taking the exact path would silently mix them."""
    from legoesm.ocean.dynamics.ocean_tendency_common import (
        iterate_eos_and_pressure_anomaly,
    )
    rng = np.random.default_rng(6)
    ny, nx, nz = 2, 2, 4
    T = jnp.asarray(rng.uniform(0.0, 20.0, (ny, nx, nz)))
    S = jnp.asarray(rng.uniform(34.0, 36.0, (ny, nx, nz)))
    dz = jnp.full((nz,), 50.0)
    depth = jnp.cumsum(dz) - 0.5 * dz
    eos_fn = make_eos_fn("nemo_seos", eos_nemo_seos=NemoSEOSConfig(rho0=1030.))
    _rho, rho_prime, _p = iterate_eos_and_pressure_anomaly(
        T, S, jnp.ones((ny, nx), dtype=bool), lambda f: f, eos_fn, dz,
        1026.0, constants.g, n_iter=2, eos_depth="geometric",
        eos_geometric_depth_1d=depth)
    # The mismatched card falls back to the subtraction, which is what the
    # pre-existing behaviour was -- never to a mixed-rho0 anomaly.
    rho = eos_fn(T, S, (1026.0 * constants.g) * depth)
    assert np.array_equal(np.asarray(rho_prime), np.asarray(rho) - 1026.0)
