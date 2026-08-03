"""Direct unit tests for the Treguier-1997 adaptive GM coefficient
(NEMO 5.0.1 ldftra.F90::ldf_eiv, nn_aei_ijk_t=21 — the DINO/ORCA1 oracle
scaling): κ = min( min(1,|f/f20|)·Ro²·T⁻¹, aei0 ).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
    TREGUIER_RO_FACTOR,
    TREGUIER_RO_MAX_M,
    TREGUIER_RO_MIN_M,
    TREGUIER_ZHW_OFFSET_M,
    compute_treguier_kappa_gm,
)
from legoesm.ocean.physics.lateral_mixing.config import (
    GMRediConfig,
    TreguierConfig,
    VisbeckConfig,
)
from legoesm.ocean.vertical import create_ocean_z_star

jax.config.update("jax_enable_x64", True)


def _setup(nlev=10, H=1000.0, n=3, slope=1e-4, jac=1.0):
    z = create_ocean_z_star(n_levels=nlev, H_max=H)
    shape = (n, n, nlev)
    rho = jnp.broadcast_to(
        jnp.linspace(constants.rho_ocean, constants.rho_ocean + 2.0, nlev),
        shape)
    S_x = jnp.full(shape[:-1] + (nlev - 1,), slope)
    S_y = jnp.zeros_like(S_x)
    jacobian = jnp.full(shape[:-1], jac)
    return rho, S_x, S_y, z, jacobian


def _expected_kappa(rho, S_x, S_y, z, jacobian, f, aei0):
    """Independently reassemble the NEMO ldf_eiv formula using the model's
    own shared N² (the exact quantity the implementation consumes)."""
    from legoesm.ocean.eos import compute_buoyancy_frequency
    dz_actual = np.asarray(z.dz_ref) * np.asarray(jacobian)[..., None]
    dz_half = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
    N2 = np.asarray(compute_buoyancy_frequency(
        rho, z.dz_ref, jacobian, rho_ref=constants.rho_ocean, g=constants.g))
    N = np.sqrt(np.maximum(N2, 1e-30))
    S = np.sqrt(np.asarray(S_x) ** 2 + np.asarray(S_y) ** 2 + 1e-30)
    int_N_dz = np.sum(N * dz_half, axis=-1)
    ro = np.clip(TREGUIER_RO_FACTOR * int_N_dz / np.maximum(np.abs(f), 1e-10),
                 TREGUIER_RO_MIN_M, TREGUIER_RO_MAX_M)
    zah = np.sum((N * S) ** 2 * dz_half, axis=-1)
    zhw = TREGUIER_ZHW_OFFSET_M + np.sum(dz_half, axis=-1)
    t_inv = np.sqrt(zah / zhw)
    f20 = 2.0 * constants.Omega * np.sin(np.deg2rad(20.0))
    taper = np.minimum(1.0, np.abs(f) / f20)
    return np.minimum(taper * ro ** 2 * t_inv, aei0)


class TestTreguierKappa:
    def test_matches_independent_formula(self):
        rho, S_x, S_y, z, jac = _setup()
        f = jnp.full((3, 3), 1.0e-4)                 # midlatitude
        cfg = TreguierConfig(enabled=True, aei0=1.0e9)   # cap inert
        got = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f, cfg))
        want = _expected_kappa(rho, S_x, S_y, z, jac, np.asarray(f), 1.0e9)
        np.testing.assert_allclose(got, want, rtol=1e-6)
        assert (got > 0.0).all()

    def test_rossby_radius_clamps(self):
        rho, S_x, S_y, z, jac = _setup()
        cfg = TreguierConfig(enabled=True, aei0=1.0e12)
        # Tiny |f| -> Ro hits the 40 km cap; huge |f| -> the 2 km floor.
        k_lo_f = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, jnp.full((3, 3), 1.0e-9), cfg))
        k_hi_f = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, jnp.full((3, 3), 1.0), cfg))
        want_lo = _expected_kappa(rho, S_x, S_y, z, jac,
                                  np.full((3, 3), 1.0e-9), 1.0e12)
        want_hi = _expected_kappa(rho, S_x, S_y, z, jac,
                                  np.full((3, 3), 1.0), 1.0e12)
        np.testing.assert_allclose(k_lo_f, want_lo, rtol=1e-6)
        np.testing.assert_allclose(k_hi_f, want_hi, rtol=1e-6)
        # The clamps genuinely BIND in these regimes (non-vacuous): the raw
        # (unclamped) Ro = 0.4·∫N dz/|f| straddles the [2 km, 40 km] bounds.
        # (No directional κ assert: the tropical taper ∝|f| dominates, so the
        # tiny-f κ is SMALLER despite its 40 km radius.)
        from legoesm.ocean.eos import compute_buoyancy_frequency
        dz_actual = np.asarray(z.dz_ref) * np.asarray(jac)[..., None]
        dzh = 0.5 * (dz_actual[..., :-1] + dz_actual[..., 1:])
        N = np.sqrt(np.maximum(np.asarray(compute_buoyancy_frequency(
            rho, z.dz_ref, jac, rho_ref=constants.rho_ocean,
            g=constants.g)), 1e-30))
        int_N = np.sum(N * dzh, axis=-1)
        assert (TREGUIER_RO_FACTOR * int_N / 1.0e-9 > TREGUIER_RO_MAX_M).all()
        assert (TREGUIER_RO_FACTOR * int_N / 1.0 < TREGUIER_RO_MIN_M).all()

    def test_tropical_taper(self):
        """At |f| = ½f₂₀ the taper halves κ relative to the untapered value
        at the SAME f (cap inert, Ro un-clamped regime)."""
        rho, S_x, S_y, z, jac = _setup(slope=1e-5)
        f20 = 2.0 * constants.Omega * np.sin(np.deg2rad(20.0))
        f_half = jnp.full((3, 3), 0.5 * f20)
        cfg = TreguierConfig(enabled=True, aei0=1.0e12)
        got = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f_half, cfg))
        want = _expected_kappa(rho, S_x, S_y, z, jac,
                               np.asarray(f_half), 1.0e12)
        np.testing.assert_allclose(got, want, rtol=1e-6)
        # taper factor is exactly 0.5 in the expected formula — assert the
        # implementation reproduces it (ratio vs the taper-free value).
        untapered = want / 0.5
        np.testing.assert_allclose(got * 2.0, untapered, rtol=1e-6)

    def test_aei0_cap_binds(self):
        rho, S_x, S_y, z, jac = _setup()
        f = jnp.full((3, 3), 1.0e-4)
        k_capped = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f, TreguierConfig(enabled=True, aei0=100.0)))
        k_free = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f, TreguierConfig(enabled=True, aei0=1e12)))
        assert (k_free > 100.0).all()           # cap is genuinely binding
        np.testing.assert_allclose(k_capped, 100.0, rtol=1e-12)

    def test_dry_column_zero(self):
        rho, S_x, S_y, z, _ = _setup()
        jac = jnp.zeros((3, 3))                  # all-dry
        k = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, jnp.full((3, 3), 1e-4),
            TreguierConfig(enabled=True)))
        assert np.allclose(k, 0.0)

    def test_grad_finite(self):
        rho, S_x, S_y, z, jac = _setup()
        f = jnp.full((3, 3), 1e-4)
        cfg = TreguierConfig(enabled=True)

        def total(r):
            return jnp.sum(compute_treguier_kappa_gm(
                r, S_x, S_y, z, jac, f, cfg))

        g = jax.grad(total)(rho)
        assert bool(jnp.isfinite(g).all())

    def test_omega_override_changes_taper_only(self):
        """#1226: the ldf_eiv kappa (aeiu) amplitude bias (ratio 1.000608 at
        corr=1.0) traced to legoESM's canonical constants.Omega being a
        4-sig-fig rounding of the physical Earth rotation rate NEMO's own
        ldftra.F90 uses (verified by feeding NEMO's own dumped
        zn/zah/zhw/wslpi/wslpj through this exact formula: the relative gap
        enters zRo linearly and zaeiw quadratically, matching NEMO's own
        omega closes both to machine precision). ``omega`` feeds ONLY
        ``f20 = 2*omega*sin(20deg)`` inside this function -- an independent
        hand-computed f20 with a DIFFERENT omega must change kappa (a no-op
        parameter would silently defeat the whole fix)."""
        rho, S_x, S_y, z, jac = _setup(slope=1e-5)
        omega_a = constants.Omega
        from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
        omega_b = NEMO_CONSTANTS_CONFIG.Omega
        assert omega_a != omega_b
        # Pick |f| squarely inside the tropical taper's linear regime (not
        # at the min(1, .) clip) so a small omega shift is NOT masked by the
        # cap: |f| = 0.5*f20(omega_a) sits well under both f20(omega_a) and
        # f20(omega_b) (they differ by ~1.6e-5 relative).
        f20_a = 2.0 * omega_a * np.sin(np.deg2rad(20.0))
        f_val = jnp.full((3, 3), 0.5 * f20_a)
        cfg = TreguierConfig(enabled=True, aei0=1.0e12)   # cap inert
        k_a = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f_val, cfg, omega=omega_a))
        k_b = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f_val, cfg, omega=omega_b))
        # Independently hand-compute the taper ratio the omega swap implies:
        # taper = min(1, |f|/f20); f20 scales linearly with omega, so
        # taper_b/taper_a = f20_a/f20_b = omega_a/omega_b (both un-clipped
        # here since |f|=0.5*f20_a < f20_b too -- omega_b > omega_a).
        f20_b = 2.0 * omega_b * np.sin(np.deg2rad(20.0))
        expected_ratio = f20_a / f20_b
        assert not np.isclose(expected_ratio, 1.0)   # non-vacuous
        np.testing.assert_allclose(
            k_b / k_a, np.full_like(k_a, expected_ratio), rtol=1e-9)
        # Default (no omega kwarg) must equal the explicit constants.Omega
        # call -- zero-behaviour-change guarantee for every non-oracle caller.
        k_default = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f_val, cfg))
        np.testing.assert_allclose(k_default, k_a, rtol=1e-12)


class TestTreguierKappaMinFloor:
    """``TreguierConfig.kappa_min`` — the equatorial-taper floor.

    The taper ``min(1, |f/f20|)`` drives κ → 0 at the equator; the floor keeps a
    finite GM there.  Default 0.0 must be INERT (byte-identical), the floor must
    NOT leak into dry columns, and it must behave identically on the generic and
    the NEMO-native κ paths.
    """

    def test_default_is_inert(self):
        """The unfloored result must equal the INDEPENDENT ldf_eiv formula --
        a reference that predates the floor -- not merely another call through
        the same new code (which would pass even if the floor were wrong)."""
        assert TreguierConfig().kappa_min == 0.0
        rho, S_x, S_y, z, jac = _setup()
        f = jnp.full((3, 3), 1.0e-4)
        aei0 = 1.0e9
        got = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f,
            TreguierConfig(enabled=True, aei0=aei0)))
        want = _expected_kappa(rho, S_x, S_y, z, jac, np.asarray(f), aei0)
        np.testing.assert_allclose(got, want, rtol=1e-6)

    def test_floor_binds_at_the_equator(self):
        """f → 0 makes the taper → 0, so the unfloored κ collapses; the floor
        is what keeps GM alive in the equatorial band."""
        rho, S_x, S_y, z, jac = _setup()
        f_eq = jnp.full((3, 3), 1.0e-8)          # ~equatorial: taper ~ 2e-4
        unfloored = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f_eq, TreguierConfig(enabled=True)))
        floored = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f_eq,
            TreguierConfig(enabled=True, kappa_min=200.0)))
        assert (unfloored < 200.0).all()         # floor is genuinely binding
        np.testing.assert_allclose(floored, 200.0, rtol=1e-12)

    def test_floor_does_not_raise_midlatitude_kappa(self):
        rho, S_x, S_y, z, jac = _setup()
        f = jnp.full((3, 3), 1.0e-4)
        cfg_hi = TreguierConfig(enabled=True, aei0=1.0e9)
        free = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f, cfg_hi))
        assert (free > 1.0).all()
        floored = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f, cfg_hi._replace(kappa_min=1.0)))
        np.testing.assert_array_equal(free, floored)

    def test_floor_excluded_from_dry_columns(self):
        """The floor is applied BEFORE the wet mask, so a dry column must stay
        EXACTLY 0 even with a large kappa_min (else GM would switch on over
        land)."""
        rho, S_x, S_y, z, _ = _setup()
        jac = jnp.zeros((3, 3))                  # all-dry
        k = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, jnp.full((3, 3), 1e-4),
            TreguierConfig(enabled=True, kappa_min=200.0)))
        np.testing.assert_array_equal(k, 0.0)

    def test_grad_finite_with_floor(self):
        rho, S_x, S_y, z, jac = _setup()
        f = jnp.full((3, 3), 1e-4)
        cfg = TreguierConfig(enabled=True, kappa_min=200.0)

        def total(r):
            return jnp.sum(compute_treguier_kappa_gm(
                r, S_x, S_y, z, jac, f, cfg))

        assert bool(jnp.isfinite(jax.grad(total)(rho)).all())

    def test_floor_is_clamped_to_the_cap(self):
        """STRUCTURAL invariant: even with kappa_min > aei0 (reachable when a
        TRAINED, traced aei0 disables the Python validator) the kernel must
        never return more than the cap."""
        rho, S_x, S_y, z, jac = _setup()
        f = jnp.full((3, 3), 1.0e-4)
        k = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f,
            TreguierConfig(enabled=True, aei0=100.0, kappa_min=200.0)))
        assert (k <= 100.0 + 1e-9).all()
        np.testing.assert_allclose(k, 100.0, rtol=1e-12)

    def test_traced_aei0_does_not_raise_and_is_differentiable(self):
        """``aei0`` is tunable_tier=2, so param_collector splices it in as a
        TRACER inside the loss.  A Python ``kappa_min > aei0`` comparison would
        raise TracerBoolConversionError; the floor must stay trace-safe."""
        rho, S_x, S_y, z, jac = _setup()
        f = jnp.full((3, 3), 1.0e-4)

        def total(aei0):
            cfg = TreguierConfig(enabled=True, aei0=aei0, kappa_min=200.0)
            return jnp.sum(compute_treguier_kappa_gm(
                rho, S_x, S_y, z, jac, f, cfg))

        # forward under jit (aei0 traced) and reverse-mode through the cap
        val = jax.jit(total)(jnp.asarray(1800.0))
        assert bool(jnp.isfinite(val))
        g = jax.grad(total)(jnp.asarray(1800.0))
        assert bool(jnp.isfinite(g))

    def test_traced_kappa_min_is_differentiable(self):
        rho, S_x, S_y, z, jac = _setup()
        f = jnp.full((3, 3), 1.0e-8)          # equatorial: the floor binds

        def total(kmin):
            cfg = TreguierConfig(enabled=True, aei0=1800.0, kappa_min=kmin)
            return jnp.sum(compute_treguier_kappa_gm(
                rho, S_x, S_y, z, jac, f, cfg))

        g = jax.grad(total)(jnp.asarray(200.0))
        assert bool(jnp.isfinite(g))
        # d(sum)/d(kappa_min) = number of columns where the floor BINDS.
        # Establish that count from the fixture instead of hard-coding it, so
        # a fixture change gives a diagnostic failure rather than a bare 9.
        unfloored = np.asarray(compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f,
            TreguierConfig(enabled=True, aei0=1800.0)))
        n_binding = int((unfloored < 200.0).sum())
        assert n_binding == unfloored.size, "fixture must be fully floored here"
        assert float(g) == pytest.approx(float(n_binding))

    def test_validator_skips_traced_leaves(self):
        from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
            validate_treguier_cfg,
        )

        def f(aei0):
            # would raise TracerBoolConversionError if the validator compared
            validate_treguier_cfg(
                TreguierConfig(enabled=True, aei0=aei0, kappa_min=200.0))
            return aei0 * 2.0

        assert float(jax.jit(f)(jnp.asarray(100.0))) == pytest.approx(200.0)

    def test_validator_rejects_non_finite(self):
        from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
            validate_treguier_cfg,
        )
        with pytest.raises(ValueError, match="must be finite"):
            validate_treguier_cfg(
                TreguierConfig(enabled=True, aei0=float("nan")))
        with pytest.raises(ValueError, match="must be > 0"):
            validate_treguier_cfg(TreguierConfig(enabled=True, aei0=0.0))

    @pytest.mark.parametrize("bad", [
        np.float32("nan"), np.float64("nan"), np.float32(-1.0),
        jnp.asarray(float("nan")), jnp.asarray(-1.0),
    ])
    def test_validator_rejects_concrete_non_python_scalars(self, bad):
        """An ``isinstance(x, (int, float))`` guard would wave these through --
        they are CONCRETE (not tracers), so they must be validated, or a
        NaN/negative diffusivity reaches the kernel."""
        from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
            validate_treguier_cfg,
        )
        with pytest.raises(ValueError, match="must be finite|must be > 0"):
            validate_treguier_cfg(TreguierConfig(enabled=True, aei0=bad))

    def test_validator_rejects_bool(self):
        """bool is an int subclass: aei0=True would install a 1 m^2/s cap."""
        from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
            validate_treguier_cfg,
        )
        with pytest.raises(ValueError, match="must be a real number"):
            validate_treguier_cfg(TreguierConfig(enabled=True, aei0=True))

    def test_floor_precedes_resolution_scaling(self):
        """DOCUMENTED ordering (same as VisbeckConfig.kappa_min): the floor is
        on the raw Treguier coefficient, and the Hallberg resolution function
        scales it afterwards, so the EFFECTIVE kappa may fall below the floor.
        Pinning it so the behaviour cannot drift silently."""
        from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
            gm_resolution_scaled_kappa,
        )
        rho, S_x, S_y, z, jac = _setup()
        f = jnp.full((3, 3), 1.0e-8)
        kappa = compute_treguier_kappa_gm(
            rho, S_x, S_y, z, jac, f,
            TreguierConfig(enabled=True, kappa_min=200.0))
        np.testing.assert_allclose(np.asarray(kappa), 200.0, rtol=1e-12)
        scaled = np.asarray(gm_resolution_scaled_kappa(
            kappa, f, jnp.full((3, 3), 1.0e4), 400.0, 2.0))
        assert (scaled < 200.0).all()      # floor is pre-scaling, by design

    def test_validator_rejects_floor_above_cap(self):
        """kappa_min > aei0 would override the NEMO cap on every wet cell."""
        from legoesm.ocean.physics.lateral_mixing._gm_redi_common import (
            validate_treguier_cfg,
        )
        with pytest.raises(ValueError, match="exceeds the NEMO cap"):
            validate_treguier_cfg(
                TreguierConfig(enabled=True, aei0=1800.0, kappa_min=5000.0))
        with pytest.raises(ValueError, match="must be >= 0"):
            validate_treguier_cfg(
                TreguierConfig(enabled=True, kappa_min=-1.0))
        # disabled block is never validated; valid block passes
        validate_treguier_cfg(TreguierConfig(kappa_min=5000.0))
        validate_treguier_cfg(
            TreguierConfig(enabled=True, aei0=1800.0, kappa_min=200.0))

    def test_param_spec_classifies_kappa_min(self):
        """A new float field on a specced Config must be in params/excluded or
        tests/test_param_specs.py goes red."""
        from legoesm.ocean.physics.lateral_mixing import config as _lm_config
        entry = _lm_config.__param_spec__["TreguierConfig"]
        assert "kappa_min" in (set(entry["params"]) | set(entry["excluded"]))


class TestTreguierKappaNemoNative:
    """#1317: compute_treguier_kappa_gm_nemo_native must consume the SAME
    wslpi/wslpj (compute_nemo_native_slopes) NEMO's own ldf_eiv sums over
    (ldftra.F90:664-706) — not the simplified cell-centred S_x/S_y the
    generic compute_treguier_kappa_gm uses. Verified (day-0 DINO twin) the
    two formulations disagree materially (corr=0.28, mean 646 vs 196 m^2/s)
    and the fix raises the ADVECTION-bucket (bolus-inclusive) tracer-tendency
    corr vs the NEMO oracle from 0.9347 to 0.9889 (full3D)."""

    def _dino_fixture(self, n_lon=50):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, create_dino_z_star, dino_lat_lon_grid,
            dino_lat_lon_state,
        )
        dcfg = DINOConfig()
        z = create_dino_z_star(dcfg)
        g = dino_lat_lon_grid(dcfg, n_lon=n_lon)
        st = dino_lat_lon_state(g, z, dcfg)
        return dcfg, z, g, st

    def _independent_kappa(self, rho, T, S, mask, wslpi, wslpj, z, g_grid,
                            f, cfg, rho_0, g, eos_fn, act):
        """Reassemble ldftra.F90:664-706 straight from the definition
        (full jk=1..jpk column sum incl. the surface w-level e3w(1)),
        independent of the shared helper the implementation calls. Uses
        the SAME eos_fn AND the SAME topography-aware 3-D active mask
        (``act`` — DINO has a variable bathymetry, so a column's active
        depth is shallower than nlev below the shelf/ridge) as the
        implementation for the adiabatic N^2 (that part is validated on
        its own by test_mixed_layer_depth.py / test_isoneutral_slope_
        density.py); this test's job is the ldf_eiv REDUCTION (full-column
        sum, zhw offset, Ro clamp, tropical taper, aei0 cap), not
        re-deriving N^2/the wet-column mask from scratch.
        """
        dz = np.asarray(z.dz_ref)
        gdept = np.cumsum(dz) - 0.5 * dz
        e3w = np.concatenate([dz[:1], gdept[1:] - gdept[:-1]])
        nlat, nlon, nlev = np.asarray(rho).shape
        e3w_3d = np.broadcast_to(e3w, (nlat, nlon, nlev))
        m = np.asarray(mask)
        act = np.asarray(act)
        wmask3 = act * np.roll(act, 1, axis=2)
        wmask3[:, :, 0] = act[:, :, 0]

        from legoesm.ocean.eos import compute_buoyancy_frequency_adiabatic
        p_cell = (rho_0 * g * gdept)[None, None, :] * np.ones_like(np.asarray(rho))
        J1 = np.ones((nlat, nlon))
        n2_int = np.asarray(compute_buoyancy_frequency_adiabatic(
            T, S, jnp.asarray(p_cell), z.dz_ref, jnp.asarray(J1), eos_fn=eos_fn))
        pn2 = np.concatenate([np.zeros((nlat, nlon, 1)), n2_int], axis=-1) * wmask3

        zn2 = np.maximum(pn2, 0.0)
        zn = np.sum(np.sqrt(zn2) * e3w_3d, axis=-1)
        ze3w = e3w_3d * wmask3
        wi = np.asarray(wslpi)
        wj = np.asarray(wslpj)
        zah = np.sum(zn2 * (wi ** 2 + wj ** 2) * ze3w, axis=-1)
        zhw = TREGUIER_ZHW_OFFSET_M + np.sum(ze3w, axis=-1)
        f_abs = np.maximum(np.abs(np.asarray(f)), 1e-10)
        ro = np.clip(TREGUIER_RO_FACTOR * zn / f_abs,
                     TREGUIER_RO_MIN_M, TREGUIER_RO_MAX_M)
        t_inv = np.sqrt(zah / np.maximum(zhw, 1e-10))
        f20 = 2.0 * constants.Omega * np.sin(np.deg2rad(20.0))
        taper = np.minimum(1.0, np.abs(np.asarray(f)) / f20)
        return np.where(m > 0.5, np.minimum(taper * ro ** 2 * t_inv, cfg.aei0), 0.0)

    def test_matches_ldf_eiv_full_column_formula(self):
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            compute_nemo_native_slopes, compute_treguier_kappa_gm_nemo_native,
            gm_redi_density_and_jacobian,
        )
        from legoesm.ocean.eos import make_eos_fn
        dcfg, z, g, st = self._dino_fixture()
        gm_cfg = GMRediConfig(slope_scheme="nemo_iso_lap", slope_positions="nemo_native")
        eos_fn = make_eos_fn("nemo_seos")
        mask = st.land_mask.data
        rho, jacobian = gm_redi_density_and_jacobian(
            st.T.data, st.S.data, st.eta.data, st.H_bathy.data, g, z,
            eos="nemo_seos", mask=mask)
        f = jnp.broadcast_to(g.f, mask.shape)
        _z_top = jnp.cumsum(z.dz_ref) - z.dz_ref
        act = ((mask[:, :, None] > 0.5)
               & (_z_top[None, None, :] < st.H_bathy.data[:, :, None])).astype(st.T.data.dtype)
        uslp, vslp, wslpi, wslpj = compute_nemo_native_slopes(
            rho, st.T.data, st.S.data, mask, st.u_mask.data, st.v_mask.data,
            z, g, gm_cfg, eos_fn, active_3d=act)
        cfg = TreguierConfig(enabled=True, aei0=1500.0)
        got = np.asarray(compute_treguier_kappa_gm_nemo_native(
            rho, st.T.data, st.S.data, wslpi, wslpj, mask, z, g, f, cfg,
            eos_fn, active_3d=act))
        want = self._independent_kappa(
            rho, st.T.data, st.S.data, mask, wslpi, wslpj, z, g, f, cfg,
            rho_0=constants.rho_ocean, g=constants.g, eos_fn=eos_fn, act=act)
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)
        assert (got[np.asarray(mask) > 0.5] >= 0.0).all()

    def test_dry_column_zero(self):
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            compute_nemo_native_slopes, compute_treguier_kappa_gm_nemo_native,
            gm_redi_density_and_jacobian,
        )
        from legoesm.ocean.eos import make_eos_fn
        dcfg, z, g, st = self._dino_fixture()
        gm_cfg = GMRediConfig(slope_scheme="nemo_iso_lap", slope_positions="nemo_native")
        eos_fn = make_eos_fn("nemo_seos")
        mask = jnp.zeros_like(st.land_mask.data)  # all-dry
        rho, jacobian = gm_redi_density_and_jacobian(
            st.T.data, st.S.data, st.eta.data, st.H_bathy.data, g, z,
            eos="nemo_seos", mask=mask)
        f = jnp.broadcast_to(g.f, mask.shape)
        u_mask = jnp.zeros_like(st.u_mask.data)
        v_mask = jnp.zeros_like(st.v_mask.data)
        act = jnp.zeros_like(rho)
        uslp, vslp, wslpi, wslpj = compute_nemo_native_slopes(
            rho, st.T.data, st.S.data, mask, u_mask, v_mask, z, g, gm_cfg,
            eos_fn, active_3d=act)
        cfg = TreguierConfig(enabled=True, aei0=1500.0)
        got = np.asarray(compute_treguier_kappa_gm_nemo_native(
            rho, st.T.data, st.S.data, wslpi, wslpj, mask, z, g, f, cfg,
            eos_fn, active_3d=act))
        assert np.allclose(got, 0.0)
        # ...and a large floor must NOT switch GM on over a dry column: the
        # floor is applied before the wet mask on this path too.
        got_floored = np.asarray(compute_treguier_kappa_gm_nemo_native(
            rho, st.T.data, st.S.data, wslpi, wslpj, mask, z, g, f,
            cfg._replace(kappa_min=200.0), eos_fn, active_3d=act))
        np.testing.assert_array_equal(got_floored, 0.0)

    def test_kappa_min_floor_applies_on_this_path_too(self):
        """REGRESSION: the floor was added to the generic
        ``compute_treguier_kappa_gm`` only, so ``kappa_min`` was silently INERT
        on the nemo_iso_lap+nemo_native path — i.e. on exactly the most
        NEMO-faithful configuration.  Both κ paths run the SAME equatorial
        taper and so must honour the SAME floor."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            compute_nemo_native_slopes, compute_treguier_kappa_gm_nemo_native,
            gm_redi_density_and_jacobian,
        )
        from legoesm.ocean.eos import make_eos_fn
        dcfg, z, g, st = self._dino_fixture()
        gm_cfg = GMRediConfig(slope_scheme="nemo_iso_lap",
                              slope_positions="nemo_native")
        eos_fn = make_eos_fn("nemo_seos")
        mask = st.land_mask.data
        rho, jacobian = gm_redi_density_and_jacobian(
            st.T.data, st.S.data, st.eta.data, st.H_bathy.data, g, z,
            eos="nemo_seos", mask=mask)
        f = jnp.broadcast_to(g.f, mask.shape)
        _z_top = jnp.cumsum(z.dz_ref) - z.dz_ref
        act = ((mask[:, :, None] > 0.5)
               & (_z_top[None, None, :]
                  < st.H_bathy.data[:, :, None])).astype(st.T.data.dtype)
        uslp, vslp, wslpi, wslpj = compute_nemo_native_slopes(
            rho, st.T.data, st.S.data, mask, st.u_mask.data, st.v_mask.data,
            z, g, gm_cfg, eos_fn, active_3d=act)
        cfg = TreguierConfig(enabled=True, aei0=1500.0)
        wet = np.asarray(mask) > 0.5

        def _kappa(c):
            return np.asarray(compute_treguier_kappa_gm_nemo_native(
                rho, st.T.data, st.S.data, wslpi, wslpj, mask, z, g, f, c,
                eos_fn, active_3d=act))

        unfloored = _kappa(cfg)
        # the fixture must actually contain sub-floor wet cells, else the
        # assertion below would pass vacuously
        floor = 200.0
        assert (unfloored[wet] < floor).any()
        floored = _kappa(cfg._replace(kappa_min=floor))
        assert (floored[wet] >= floor - 1e-9).all()
        # default (0.0) stays byte-identical to the pre-floor behaviour
        np.testing.assert_array_equal(_kappa(cfg._replace(kappa_min=0.0)),
                                      unfloored)

    def test_grad_finite(self):
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            compute_nemo_native_slopes, compute_treguier_kappa_gm_nemo_native,
            gm_redi_density_and_jacobian,
        )
        from legoesm.ocean.eos import make_eos_fn
        dcfg, z, g, st = self._dino_fixture()
        gm_cfg = GMRediConfig(slope_scheme="nemo_iso_lap", slope_positions="nemo_native")
        eos_fn = make_eos_fn("nemo_seos")
        mask = st.land_mask.data
        f = jnp.broadcast_to(g.f, mask.shape)
        _z_top = jnp.cumsum(z.dz_ref) - z.dz_ref
        act = ((mask[:, :, None] > 0.5)
               & (_z_top[None, None, :] < st.H_bathy.data[:, :, None])).astype(st.T.data.dtype)
        cfg = TreguierConfig(enabled=True, aei0=1500.0)

        def total(T):
            rho, _ = gm_redi_density_and_jacobian(
                T, st.S.data, st.eta.data, st.H_bathy.data, g, z,
                eos="nemo_seos", mask=mask)
            uslp, vslp, wslpi, wslpj = compute_nemo_native_slopes(
                rho, T, st.S.data, mask, st.u_mask.data, st.v_mask.data,
                z, g, gm_cfg, eos_fn, active_3d=act)
            return jnp.sum(compute_treguier_kappa_gm_nemo_native(
                rho, T, st.S.data, wslpi, wslpj, mask, z, g, f, cfg,
                eos_fn, active_3d=act))

        grad = jax.grad(total)(st.T.data)
        # Finite on cells with a fully-WET 8-neighbourhood, EXCLUDING the
        # channel's non-periodic meridional (row) boundary: the grid is
        # periodic in longitude only, so row 0 / row -1 have no real
        # north/south neighbour: pad those with "dry" (False) rather than
        # wrapping, matching the domain's actual (non-periodic-in-lat)
        # topology (a plain np.roll on axis 0 would incorrectly treat row
        # 0's neighbour as row -1, an unrelated part of the channel).
        m = np.asarray(mask)
        wet_interior = np.ones_like(m, dtype=bool)
        for di in (-1, 0, 1):
            row_shifted = np.roll(m, di, axis=0) > 0.5
            if di == -1:
                row_shifted[-1, :] = False
            elif di == 1:
                row_shifted[0, :] = False
            for dj in (-1, 0, 1):
                wet_interior &= np.roll(row_shifted, dj, axis=1)
        wet3d = jnp.broadcast_to(jnp.asarray(wet_interior)[:, :, None], grad.shape)
        assert bool(wet_interior.any())  # non-vacuous
        assert bool(jnp.isfinite(jnp.where(wet3d, grad, 0.0)).all())

    def test_omega_override_changes_taper_only(self):
        """Same #1226 guarantee as the generic-path test above, for the
        nemo_native leaf (compute_treguier_kappa_gm_nemo_native) — the
        function the production nemo_dino_kamm_mlf card actually calls."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            compute_nemo_native_slopes, compute_treguier_kappa_gm_nemo_native,
            gm_redi_density_and_jacobian,
        )
        from legoesm.ocean.eos import make_eos_fn
        dcfg, z, g, st = self._dino_fixture()
        gm_cfg = GMRediConfig(slope_scheme="nemo_iso_lap", slope_positions="nemo_native")
        eos_fn = make_eos_fn("nemo_seos")
        mask = st.land_mask.data
        rho, jacobian = gm_redi_density_and_jacobian(
            st.T.data, st.S.data, st.eta.data, st.H_bathy.data, g, z,
            eos="nemo_seos", mask=mask)
        omega_a = constants.Omega
        from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
        omega_b = NEMO_CONSTANTS_CONFIG.Omega
        assert omega_a != omega_b
        f = jnp.broadcast_to(g.f, mask.shape)
        _z_top = jnp.cumsum(z.dz_ref) - z.dz_ref
        act = ((mask[:, :, None] > 0.5)
               & (_z_top[None, None, :] < st.H_bathy.data[:, :, None])).astype(st.T.data.dtype)
        uslp, vslp, wslpi, wslpj = compute_nemo_native_slopes(
            rho, st.T.data, st.S.data, mask, st.u_mask.data, st.v_mask.data,
            z, g, gm_cfg, eos_fn, active_3d=act)
        cfg = TreguierConfig(enabled=True, aei0=1.0e9)   # cap inert
        k_a = np.asarray(compute_treguier_kappa_gm_nemo_native(
            rho, st.T.data, st.S.data, wslpi, wslpj, mask, z, g, f, cfg,
            eos_fn, active_3d=act, omega=omega_a))
        k_b = np.asarray(compute_treguier_kappa_gm_nemo_native(
            rho, st.T.data, st.S.data, wslpi, wslpj, mask, z, g, f, cfg,
            eos_fn, active_3d=act, omega=omega_b))
        k_default = np.asarray(compute_treguier_kappa_gm_nemo_native(
            rho, st.T.data, st.S.data, wslpi, wslpj, mask, z, g, f, cfg,
            eos_fn, active_3d=act))
        wet = np.asarray(mask) > 0.5
        assert wet.any()
        # Different omega -> different taper -> different kappa somewhere in
        # the wet domain (non-vacuous: a no-op parameter would defeat the fix).
        assert not np.allclose(k_a[wet], k_b[wet])
        # Zero-behaviour-change guarantee: omitting omega == constants.Omega.
        np.testing.assert_allclose(k_default[wet], k_a[wet], rtol=1e-12)

    def test_dispatch_prefers_nemo_native_over_generic_treguier(self):
        """gm_redi_tracer_tendency_latlon must route through the
        nemo_native-consistent kappa_GM (not the generic simplified-slope
        path) when slope_scheme='nemo_iso_lap' + slope_positions='nemo_native'
        + treguier.enabled — the #1317 wiring fix."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            gm_redi_tracer_tendency_latlon,
        )
        dcfg, z, g, st = self._dino_fixture()
        mask = st.land_mask.data
        common = dict(
            kappa_Redi=100.0, slope_scheme="nemo_iso_lap",
            slope_density="neutral", treguier=TreguierConfig(enabled=True, aei0=1500.0),
        )
        cfg_native = GMRediConfig(slope_positions="nemo_native", **common)
        cfg_mode_b = GMRediConfig(slope_positions="mode_b", **common)
        kwargs = dict(
            eos="nemo_seos", mask=mask, u_mask=st.u_mask.data, v_mask=st.v_mask.data,
        )
        dT_native, dS_native = gm_redi_tracer_tendency_latlon(
            st.T.data, st.S.data, st.eta.data, st.H_bathy.data, g, z,
            cfg_native, **kwargs)
        dT_modeb, dS_modeb = gm_redi_tracer_tendency_latlon(
            st.T.data, st.S.data, st.eta.data, st.H_bathy.data, g, z,
            cfg_mode_b, **kwargs)
        # Different slope_positions -> different operator AND (post-fix)
        # different kappa_GM -> tendencies must differ (non-vacuous: the
        # pre-fix code would still differ here via the operator alone, but
        # a regression that silently drops the native-kappa branch would
        # only be caught by the corr-vs-NEMO oracle check, which is exactly
        # what this dispatch test is a cheap proxy for).
        wet3d = jnp.broadcast_to(mask[:, :, None] > 0.5, dT_native.shape)
        assert not np.allclose(
            np.asarray(jnp.where(wet3d, dT_native, 0.0)),
            np.asarray(jnp.where(wet3d, dT_modeb, 0.0)))
        assert bool(jnp.isfinite(jnp.where(wet3d, dT_native, 0.0)).all())


class TestDispatchAndWiring:
    def test_gm_redi_config_carries_treguier_default_off(self):
        cfg = GMRediConfig()
        assert cfg.treguier.enabled is False
        assert cfg.treguier.aei0 == 3000.0       # DINO rn_Ue*rn_Le

    def test_mutual_exclusion_raises_on_latlon_path(self):
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            gm_redi_tracer_tendency_latlon,
        )
        from legoesm.ocean.experiments.dino import (
            DINOConfig, create_dino_z_star, dino_lat_lon_grid,
            dino_lat_lon_state,
        )
        dcfg = DINOConfig()
        z = create_dino_z_star(dcfg)
        g = dino_lat_lon_grid(dcfg, n_lon=50)
        st = dino_lat_lon_state(g, z, dcfg)
        bad = GMRediConfig(
            visbeck=VisbeckConfig(enabled=True),
            treguier=TreguierConfig(enabled=True))
        with pytest.raises(ValueError, match="mutually exclusive"):
            gm_redi_tracer_tendency_latlon(
                st.T.data, st.S.data, st.eta.data, st.H_bathy.data,
                g, z, bad, mask=st.land_mask.data,
                u_mask=st.u_mask.data, v_mask=st.v_mask.data)

    def test_eke_override_with_treguier_raises(self):
        """A prognostic-EKE kappa_GM override is consumed BEFORE Treguier, so
        the combination would silently run the EKE coefficient (and its own
        [0, kappa_max] clip) while the user believes NEMO ldf_eiv is active."""
        from legoesm.ocean.physics.lateral_mixing.gm_redi_latlon_cgrid import (
            gm_redi_tracer_tendency_latlon,
        )
        from legoesm.ocean.experiments.dino import (
            DINOConfig, create_dino_z_star, dino_lat_lon_grid,
            dino_lat_lon_state,
        )
        dcfg = DINOConfig()
        z = create_dino_z_star(dcfg)
        g = dino_lat_lon_grid(dcfg, n_lon=50)
        st = dino_lat_lon_state(g, z, dcfg)
        cfg = GMRediConfig(treguier=TreguierConfig(enabled=True))
        override = jnp.full(st.land_mask.data.shape, 500.0)
        with pytest.raises(ValueError, match="prognostic-EKE"):
            gm_redi_tracer_tendency_latlon(
                st.T.data, st.S.data, st.eta.data, st.H_bathy.data,
                g, z, cfg, mask=st.land_mask.data,
                u_mask=st.u_mask.data, v_mask=st.v_mask.data,
                kappa_gm_override=override)

    def test_dino_treguier_rejects_non_finite_aei0(self):
        """DINO builds an enabled TreguierConfig directly; a NaN --treguier-aei0
        must fail at CONFIG BUILD, not turn the coefficient field into NaN
        mid-run."""
        import dataclasses as _dc

        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_model_config,
        )
        bad = _dc.replace(DINOConfig(), gm_kappa_scheme="treguier",
                          treguier_aei0=float("nan"))
        gg = dino_lat_lon_grid(bad, n_lon=50)
        with pytest.raises(ValueError, match="must be finite"):
            dino_lat_lon_model_config(gg, bad, physics=True)

    def test_dino_gm_kappa_scheme_wiring(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_model_config,
        )
        base = DINOConfig()
        assert base.gm_kappa_scheme == "visbeck"     # historical default
        g = dino_lat_lon_grid(base, n_lon=50)
        mc, _ = dino_lat_lon_model_config(g, base, physics=True)
        assert mc.gm_redi.visbeck.enabled is True
        assert mc.gm_redi.treguier.enabled is False
        treg = dataclasses.replace(base, gm_kappa_scheme="treguier")
        mc2, _ = dino_lat_lon_model_config(g, treg, physics=True)
        assert mc2.gm_redi.visbeck.enabled is False
        assert mc2.gm_redi.treguier.enabled is True
        # DINOConfig.treguier_aei0 default = 0.5*rn_Ue*rn_Le (ldftra.F90:332
        # explicit 1/2 factor) = 1500, not the un-halved rn_Ue*rn_Le = 3000.
        assert mc2.gm_redi.treguier.aei0 == pytest.approx(1500.0)

    def test_dino_omega_wiring_nemo_card(self):
        """#1226: the nemo_dino_kamm_mlf card pins NEMO's full-precision
        Omega (NEMO_CONSTANTS_CONFIG.Omega, phycst.F90:89) via
        DINOConfig.omega -> LatLonCGridOceanConfig.omega (the top-level
        field self.config.omega, NOT the currently-unwired self.config.
        constants.Omega default) -> create_mercator_grid(omega=...) ->
        grid.f. Every other recipe stays byte-identical to legoESM's
        canonical constants.Omega (zero-behaviour-change guarantee)."""
        from legoesm.ocean.experiments.dino import (
            dino_config_for_recipe, dino_lat_lon_grid,
            dino_lat_lon_model_config,
        )
        from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG

        default_cfg = dino_config_for_recipe("legoesm_default")
        assert default_cfg.omega == constants.Omega
        g_default = dino_lat_lon_grid(default_cfg, n_lon=50)
        mc_default, _ = dino_lat_lon_model_config(g_default, default_cfg)
        assert mc_default.omega == constants.Omega

        nemo_cfg = dino_config_for_recipe("nemo_dino_kamm_mlf")
        assert nemo_cfg.omega == NEMO_CONSTANTS_CONFIG.Omega
        assert nemo_cfg.omega != constants.Omega
        g_nemo = dino_lat_lon_grid(nemo_cfg, n_lon=50)
        mc_nemo, _ = dino_lat_lon_model_config(g_nemo, nemo_cfg)
        assert mc_nemo.omega == pytest.approx(NEMO_CONSTANTS_CONFIG.Omega)
        # The Coriolis field itself must reflect the pinned omega (not just
        # the config value) -- f = 2*omega*sin(lat) at grid construction.
        expected_f_ratio = nemo_cfg.omega / default_cfg.omega
        got_f_ratio = float(np.asarray(g_nemo.f)[50, 25]
                            / np.asarray(g_default.f)[50, 25])
        # rel=1e-6: sin(lat) itself differs at the ~1e-7 bit level between
        # two independently-constructed Mercator grids (lat-placement
        # rounding, unrelated to omega) -- loose enough to catch a
        # completely-unwired omega (which would give ratio 1.0, off by
        # 1.6e-5) yet tight enough not to mask that failure mode.
        assert got_f_ratio == pytest.approx(expected_f_ratio, rel=1e-6)

    def test_dino_unknown_scheme_raises(self):
        from legoesm.ocean.experiments.dino import (
            DINOConfig, dino_lat_lon_grid, dino_lat_lon_model_config,
        )
        base = dataclasses.replace(DINOConfig(), gm_kappa_scheme="bogus")
        g = dino_lat_lon_grid(base, n_lon=50)
        with pytest.raises(ValueError, match="gm_kappa_scheme"):
            dino_lat_lon_model_config(g, base, physics=True)
