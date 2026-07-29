"""Mixed-phase (ice-curve) hard-saturation drain — the TTL dehydration fix.

The liquid-only drain gates/lands on the liquid saturation curve; at TTL
temperatures (~195 K) that curve sits ~60% above the ice curve, so the TTL
legally holds permanent ice-supersaturation (20x-ERA5 vapour, +17.8 K warm
bias at 100 hPa — first ClimateEval scorecard).  ``ice_curve=True`` blends
liquid/ice saturation and latent heat with the standard mixed-phase w(T)
ramp [T_hom_freeze, T_freeze] and routes cold condensate to cloud ice.
"""

import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    hard_saturation_drain,
    mixed_phase_l_over_cp,
    mixed_phase_liquid_fraction,
    mixed_phase_saturation_mixing_ratio,
)
from legoesm.driver.model_driver import _mpas_hard_saturation_poststep
from legoesm.thermo import (
    saturation_mixing_ratio,
    saturation_mixing_ratio_ice,
)

DT = 75.0
P_TTL = 10000.0   # 100 hPa
T_TTL = 195.0


class TestBlend:
    def test_liquid_fraction_ramp(self):
        T = jnp.asarray([200.0, constants.T_hom_freeze, 253.15,
                         constants.T_freeze, 290.0])
        w = mixed_phase_liquid_fraction(T)
        np.testing.assert_allclose(np.asarray(w), [0.0, 0.0, 0.5, 1.0, 1.0])

    def test_blended_curve_matches_endpoints(self):
        p = jnp.full(2, 50000.0)
        T_cold = jnp.full(2, 200.0)
        T_warm = jnp.full(2, 285.0)
        np.testing.assert_allclose(
            np.asarray(mixed_phase_saturation_mixing_ratio(T_cold, p)),
            np.asarray(saturation_mixing_ratio_ice(T_cold, p)))
        np.testing.assert_allclose(
            np.asarray(mixed_phase_saturation_mixing_ratio(T_warm, p)),
            np.asarray(saturation_mixing_ratio(T_warm, p)))

    def test_ice_curve_below_liquid_when_cold(self):
        p = jnp.full(1, P_TTL)
        T = jnp.full(1, T_TTL)
        q_ice = float(saturation_mixing_ratio_ice(T, p)[0])
        q_liq = float(saturation_mixing_ratio(T, p)[0])
        assert q_ice < q_liq
        assert q_liq / q_ice > 1.3  # the supersaturation head-room being fixed

    def test_latent_heat_endpoints(self):
        assert float(mixed_phase_l_over_cp(jnp.asarray([200.0]))[0]) == \
            pytest.approx(constants.L_s / constants.c_pd)
        assert float(mixed_phase_l_over_cp(jnp.asarray([285.0]))[0]) == \
            pytest.approx(constants.L_v / constants.c_pd)


class TestDrain:
    def _ttl_state(self):
        """TTL cell: ice-supersaturated but liquid-SUBsaturated."""
        T = jnp.full((1, 1), T_TTL)
        p = jnp.full((1, 1), P_TTL)
        q_ice = saturation_mixing_ratio_ice(T, p)
        q_liq = saturation_mixing_ratio(T, p)
        q_v = 1.25 * q_ice  # 25% ice-supersat, below the liquid curve
        assert float(q_v[0, 0]) < float(q_liq[0, 0])
        return T, q_v, p, q_ice

    def test_liquid_only_blind_to_ice_supersaturation(self):
        T, q_v, p, _ = self._ttl_state()
        rate = hard_saturation_drain(T, q_v, p, DT)
        assert float(jnp.max(rate)) == 0.0  # the TTL bias mechanism

    def test_ice_curve_drains_to_blended_saturation(self):
        T, q_v, p, q_ice = self._ttl_state()
        rate = hard_saturation_drain(T, q_v, p, DT, ice_curve=True)
        assert float(jnp.min(rate)) > 0.0
        dq = rate * DT
        q_after = q_v - dq
        # Lands at/above the blended curve of the WARMED cell (deposition
        # heating raises q_sat), never below (pure drain).
        T_after = T + mixed_phase_l_over_cp(T) * dq
        q_target = mixed_phase_saturation_mixing_ratio(T_after, p)
        assert float(q_after[0, 0]) >= float(q_target[0, 0]) * (1 - 1e-9)
        # And it substantially removed the excess (>= half in one step here).
        assert float(dq[0, 0]) > 0.5 * float(q_v[0, 0] - q_ice[0, 0])

    def test_warm_cells_byte_identical(self):
        """Above the ramp both modes see the pure liquid curve: identical."""
        T = jnp.full((2, 3), 290.0)
        p = jnp.full((2, 3), 90000.0)
        q_v = 1.3 * saturation_mixing_ratio(T, p)
        r_liq = hard_saturation_drain(T, q_v, p, DT)
        r_ice = hard_saturation_drain(T, q_v, p, DT, ice_curve=True)
        np.testing.assert_array_equal(np.asarray(r_liq), np.asarray(r_ice))

    def test_gradient_finite(self):
        T = jnp.full((1, 2), T_TTL)
        p = jnp.full((1, 2), P_TTL)

        def loss(q):
            return jnp.sum(
                hard_saturation_drain(T, q, p, DT, ice_curve=True) ** 2)

        q_v = 1.4 * saturation_mixing_ratio_ice(T, p)
        g = jax.grad(loss)(q_v)
        assert bool(jnp.all(jnp.isfinite(g)))

    def test_gradient_matches_central_fd(self):
        """AD d(drain)/d{q_v,T,p} == centered FD (implicit-fn gradient exact,
        incl. the per-cell l_over_cp(T) dependence) at TTL and mixed cells."""
        def drain(qv, T, p):
            return hard_saturation_drain(
                T.reshape(1), qv.reshape(1), p.reshape(1), DT,
                ice_curve=True)[0]
        for Te, pe, mult in [(195.0, 10000.0, 1.4), (250.0, 40000.0, 1.5)]:
            qsat_i = float(saturation_mixing_ratio_ice(
                jnp.array([Te]), jnp.array([pe]))[0])
            args = [jnp.array(mult * qsat_i), jnp.array(Te), jnp.array(pe)]
            ad = [float(jax.grad(drain, i)(*args)) for i in range(3)]

            def fd(i, h):
                a = list(args)
                a[i] = args[i] + h
                fp = float(drain(*a))
                a[i] = args[i] - h
                fm = float(drain(*a))
                return (fp - fm) / (2.0 * h)

            for i, h in ((0, 1e-9), (1, 1e-4), (2, 1e-1)):
                n = fd(i, h)
                assert abs(ad[i] - n) <= 1e-4 * max(abs(n), 1e-12) + 1e-13, (
                    Te, i, ad[i], n)

    def test_heating_cap_bounds_actual_heating_on_ice_branch(self):
        """Per-step heating stays <= hard_max_heating_K with the ICE latent heat.

        A huge mixed-phase supersaturation pool binds the heating cap; the
        rate limit must cap ACTUAL heating (L_eff(T0)*dq), not a liquid-
        equivalent dq -- otherwise the ice branch (L_s > L_v) overshoots the
        stated K cap by ~13% (the pre-fix defect).
        """
        cap_K = 5.0
        for T_val in (250.0, 234.0):  # mixed and near-homogeneous-freezing
            T = jnp.full((1, 1), T_val)
            p = jnp.full((1, 1), 50000.0)
            q_v = jnp.full((1, 1), 0.05)  # 50 g/kg -- absurd pool, cap binds
            rate = hard_saturation_drain(T, q_v, p, DT, 1.1, cap_K,
                                         ice_curve=True)
            dq = rate * DT
            dT_actual = float(mixed_phase_l_over_cp(T)[0, 0] * dq[0, 0])
            # cap binds -> heating is at (not under) the K cap, and never over.
            assert dT_actual <= cap_K * (1 + 1e-9), (T_val, dT_actual)
            assert dT_actual == pytest.approx(cap_K, rel=1e-9), (T_val, dT_actual)

    def test_liquid_heating_cap_unchanged(self):
        """Warm liquid pool still caps ACTUAL heating at exactly L_v/c_pd*dq."""
        cap_K = 5.0
        T = jnp.full((1, 1), 295.0)
        p = jnp.full((1, 1), 90000.0)
        q_v = jnp.full((1, 1), 0.08)  # huge pool -> cap binds
        rate = hard_saturation_drain(T, q_v, p, DT, 1.1, cap_K)
        dT_actual = float((constants.L_v / constants.c_pd) * rate[0, 0] * DT)
        assert dT_actual == pytest.approx(cap_K, rel=1e-9), dT_actual


class TestPoststep:
    def _cold_inputs(self, with_qi=True):
        ncol, nlev = 3, 4
        T = jnp.full((ncol, nlev), T_TTL)
        p_s = jnp.full(ncol, 1.0e5)
        sigma = jnp.linspace(0.08, 0.12, nlev)  # ~TTL pressures
        p_full = p_s[:, None] * sigma[None, :]
        q_v = 1.3 * saturation_mixing_ratio_ice(T, p_full)
        q_c = jnp.zeros((ncol, nlev))
        q_i = jnp.zeros((ncol, nlev)) if with_qi else None
        return T, q_v, q_c, q_i, p_s, sigma

    def test_cold_condensate_routes_to_ice_and_conserves_water(self):
        T, q_v, q_c, q_i, p_s, sigma = self._cold_inputs()
        T2, qv2, qc2, qi2, dq = _mpas_hard_saturation_poststep(
            T, q_v, q_c, p_s, sigma, DT, 1.1, 5.0,
            ice_curve=True, q_i=q_i)
        assert float(jnp.max(dq)) > 0.0
        # All-cold: everything to q_i, nothing to q_c.
        np.testing.assert_allclose(np.asarray(qc2), 0.0, atol=1e-18)
        np.testing.assert_allclose(np.asarray(qi2), np.asarray(dq))
        # Total water exact.
        np.testing.assert_allclose(
            np.asarray(qv2 + qc2 + qi2), np.asarray(q_v + q_c + q_i),
            rtol=1e-14)

    def test_enthalpy_conserved_with_blended_latent_heat(self):
        T, q_v, q_c, q_i, p_s, sigma = self._cold_inputs()
        T2, qv2, _, _, dq = _mpas_hard_saturation_poststep(
            T, q_v, q_c, p_s, sigma, DT, 1.1, 5.0,
            ice_curve=True, q_i=q_i)
        l_cp = mixed_phase_l_over_cp(T)
        h0 = T + l_cp * q_v
        h1 = T2 + l_cp * qv2
        np.testing.assert_allclose(np.asarray(h1), np.asarray(h0), rtol=1e-12)

    def test_no_qi_reservoir_degrades_to_liquid_drain(self):
        """No cloud-ice reservoir -> ice_curve degrades to the energy-EXACT
        liquid drain (rate AND heating liquid), byte-identical to
        ice_curve=False.  Depositing the drained vapour as LIQUID q_c while
        heating with the blended (L_s-weighted) latent heat would inject
        (1-w)*L_f*dq of spurious energy -- the codex-flagged fallback bug.
        """
        # Warm supersaturated cell so the liquid drain actually fires.
        ncol, nlev = 2, 3
        T = jnp.full((ncol, nlev), 285.0)
        p_s = jnp.full(ncol, 1.0e5)
        sigma = jnp.linspace(0.8, 0.9, nlev)
        p_full = p_s[:, None] * sigma[None, :]
        q_v = 1.3 * saturation_mixing_ratio(T, p_full)
        q_c = jnp.zeros((ncol, nlev))
        ice = _mpas_hard_saturation_poststep(
            T, q_v, q_c, p_s, sigma, DT, 1.1, 5.0, ice_curve=True, q_i=None)
        liq = _mpas_hard_saturation_poststep(
            T, q_v, q_c, p_s, sigma, DT, 1.1, 5.0, ice_curve=False, q_i=None)
        assert ice[3] is None  # q_i_new
        # T, q_v, q_c, dq all identical to the liquid path (skip the None q_i).
        for a, b in zip(ice[:3] + ice[4:], liq[:3] + liq[4:]):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        # Enthalpy is the LIQUID invariant (L_v), not the blended one.
        np.testing.assert_allclose(
            np.asarray(ice[0] + (constants.L_v / constants.c_pd) * ice[1]),
            np.asarray(T + (constants.L_v / constants.c_pd) * q_v), rtol=1e-12)
        # At cold TTL with no reservoir the liquid drain correctly does nothing
        # (q_v is ice-supersaturated but liquid-SUBsaturated) -- no spurious
        # supercooled-liquid deposition.
        Tc, qvc, qcc, _, psc, sigc = self._cold_inputs(with_qi=False)
        _, _, _, _, dqc = _mpas_hard_saturation_poststep(
            Tc, qvc, qcc, psc, sigc, DT, 1.1, 5.0, ice_curve=True, q_i=None)
        np.testing.assert_allclose(np.asarray(dqc), 0.0, atol=1e-18)

    def test_legacy_mode_unchanged_arity_and_values(self):
        T, q_v, q_c, _, p_s, sigma = self._cold_inputs(with_qi=False)
        # Warm liquid supersaturation so the legacy path actually fires.
        T = jnp.full_like(T, 285.0)
        q_v = 1.3 * saturation_mixing_ratio(
            T, p_s[:, None] * jnp.asarray(sigma)[None, :])
        T2, qv2, qc2, qi2, dq = _mpas_hard_saturation_poststep(
            T, q_v, q_c, p_s, sigma, DT, 1.1, 5.0)
        assert qi2 is None
        np.testing.assert_allclose(
            np.asarray(T2), np.asarray(
                T + (constants.L_v / constants.c_pd) * dq))


def _ice_curve_cfg(**overrides):
    """A valid Morrison-on-MPAS ExperimentConfig with the ice curve on."""
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig,
    )
    base = dict(
        grid=GridConfig(grid_type="mpas", resolution=1, nlev=8,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(dt=600.0, discretization="mpas"),
        output=OutputConfig(diag_days=1),
        days=1, dataset="analytical", radiation="gray",
        microphysics="morrison", hard_saturation_adjustment=True,
        hard_sat_ice_curve=True,
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def test_validate_refuses_inert_ice_curve():
    # ice_curve on but the base drain gate off -> silently inert.
    cfg = _ice_curve_cfg(hard_saturation_adjustment=False)
    with pytest.raises(ValueError, match="hard_saturation_adjustment=True"):
        cfg.validate_strict()


def test_validate_refuses_non_mpas_ice_curve():
    from legoesm.driver.config import DycoreConfig, GridConfig
    # Only the MPAS post-step hook applies the ice curve -> inert elsewhere.
    cfg = _ice_curve_cfg(
        grid=GridConfig(grid_type="cubed_sphere", resolution=1, nlev=8),
        dycore=DycoreConfig(dt=600.0, discretization="centered"))
    with pytest.raises(ValueError, match="requires an MPAS grid"):
        cfg.validate_strict()


def test_validate_accepts_voronoi_alias_ice_curve():
    """Voronoi/icosahedral are MPAS aliases -> accepted (normalize_grid_type)."""
    from legoesm.driver.config import DycoreConfig, GridConfig
    cfg = _ice_curve_cfg(
        grid=GridConfig(grid_type="voronoi", resolution=1, nlev=8,
                        vertical_coord="sigma"),
        dycore=DycoreConfig(dt=600.0, discretization="mpas"))
    cfg.validate_strict()  # must not raise on the ice-curve grid guard


def test_validate_refuses_non_morrison_ice_curve():
    # Non-Morrison schemes lack the cloud-ice reservoir the drain deposits to.
    cfg = _ice_curve_cfg(microphysics="kessler")
    with pytest.raises(ValueError, match="microphysics='morrison'"):
        cfg.validate_strict()


def test_validate_accepts_morrison_mpas_ice_curve():
    _ice_curve_cfg().validate_strict()  # must not raise


def test_cli_round_trip():
    from scripts.run.run_amip import (
        _postprocess_args, build_arg_parser, build_config_from_args,
    )
    parser = build_arg_parser()
    cfg_default = build_config_from_args(_postprocess_args(
        parser.parse_args(["--dataset", "analytical"]), parser))
    assert cfg_default.hard_sat_ice_curve is False
    cfg = build_config_from_args(_postprocess_args(parser.parse_args([
        "--dataset", "analytical", "--grid-type", "mpas",
        "--microphysics", "morrison",
        "--hard-saturation-adjustment", "--hard-sat-ice-curve",
    ]), parser))
    assert cfg.hard_sat_ice_curve is True
    cfg.validate_strict()


def test_seed_nucleated_ice_number():
    """dN_i = dq_i / mi0 (Morrison's nucleation mass<->number closure),
    capped at the Cooper ceiling N_i_nuc_max/rho_air with rho_air=p/(R_d T)."""
    import math
    from legoesm.driver.model_driver import _seed_nucleated_ice_number
    rho_ci, r_nuc = 500.0, 10.0e-6
    mi0 = (4.0 / 3.0) * math.pi * rho_ci * r_nuc ** 3
    n_i_nuc_max = 5.0e5   # MorrisonConfig default [1/m^3]
    rho_floor = 0.1       # Morrison _RHO_FLOOR
    # Mid-tropo state (500 hPa, 260 K -> rho ~ 0.67 kg/m^3 >> floor): the cap
    # uses the true rho.
    p = jnp.full((2,), 5.0e4)
    T = jnp.full((2,), 260.0)
    rho_air = np.asarray(p) / (constants.R_d * np.asarray(T))
    assert float(rho_air[0]) > rho_floor
    ceil_perkg = n_i_nuc_max / rho_air

    # Small deposit: uncapped, each new crystal ~ the nucleation mass.
    dq_small = jnp.asarray([1.0e-9, 4.0e-9])
    out = _seed_nucleated_ice_number(
        jnp.zeros((2,)), dq_small, mi0, n_i_nuc_max, p, T)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(dq_small) / mi0, rtol=1e-12)
    assert float(out[0]) < float(ceil_perkg[0])   # below the ceiling

    # Cap-binding: a ~2 g/kg deposit would seed ~1e9/kg uncapped; the Cooper
    # ceiling bounds it -> crystals grow, not multiply.
    dq_big = jnp.asarray([2.0e-3, 2.0e-3])
    out2 = _seed_nucleated_ice_number(
        jnp.zeros((2,)), dq_big, mi0, n_i_nuc_max, p, T)
    np.testing.assert_allclose(np.asarray(out2), ceil_perkg, rtol=1e-12)
    assert float(dq_big[0] / mi0) > 300.0 * float(ceil_perkg[0])  # >> ceiling

    # GENUINE low density (15 hPa / 205 K -> rho ~ 0.025 << floor): the ceiling
    # must use Morrison's 0.1 floor, NOT the true rho (else it loosens ~4x).
    p_lo = jnp.full((2,), 1.5e3)          # 15 hPa (TTL/stratosphere)
    T_lo = jnp.full((2,), 205.0)
    rho_lo = float(np.asarray(p_lo)[0] / (constants.R_d * np.asarray(T_lo)[0]))
    assert rho_lo < rho_floor             # floor regime
    out_lo = _seed_nucleated_ice_number(
        jnp.zeros((2,)), dq_big, mi0, n_i_nuc_max, p_lo, T_lo)
    np.testing.assert_allclose(
        np.asarray(out_lo), n_i_nuc_max / rho_floor, rtol=1e-12)   # floored
    # Without the floor the cap would be ~4x looser -> assert the floor bit.
    assert float(out_lo[0]) < 0.5 * (n_i_nuc_max / rho_lo)

    # Already at/above the ceiling: no number added (deposition grows crystals).
    out3 = _seed_nucleated_ice_number(
        jnp.asarray([1.0e9, 1.0e9]), dq_big, mi0, n_i_nuc_max, p, T)
    np.testing.assert_array_equal(np.asarray(out3), [1.0e9, 1.0e9])

    # Negative/zero deposition adds no number.
    out4 = _seed_nucleated_ice_number(
        jnp.asarray([7.0, 7.0]), jnp.asarray([-1.0, 0.0]), mi0,
        n_i_nuc_max, p, T)
    np.testing.assert_array_equal(np.asarray(out4), [7.0, 7.0])


def test_fp32_ice_curve_branch_subprocess():
    """The GENUINE float32 (x64-OFF) drain branch: finite, positive, on-curve.

    A subprocess because this module enables x64 globally; the else (float32)
    branch of ``hard_saturation_drain`` is only reachable with x64 off.
    """
    script = textwrap.dedent('''
        import jax, jax.numpy as jnp
        assert not jax.config.jax_enable_x64
        from legoesm.thermo import saturation_mixing_ratio_ice
        from legoesm.atmosphere.physics.microphysics._warm_rain import (
            hard_saturation_drain, mixed_phase_saturation_mixing_ratio,
            mixed_phase_l_over_cp)
        T = jnp.full((1, 4), 205.0, jnp.float32)
        p = jnp.full((1, 4), 15000.0, jnp.float32)
        qv = (1.5 * saturation_mixing_ratio_ice(T, p)).astype(jnp.float32)
        r = hard_saturation_drain(T, qv, p, 75.0, ice_curve=True)
        assert r.dtype == jnp.float32, r.dtype
        assert bool(jnp.all(jnp.isfinite(r)))
        assert float(jnp.min(r)) > 0.0
        dq = r * 75.0
        Ta = T + mixed_phase_l_over_cp(T) * dq
        resid = float(((qv - dq) - mixed_phase_saturation_mixing_ratio(Ta, p))[0, 0])
        assert resid >= -1e-4, resid           # on/above the ice curve (fp32 tol)
        print("FP32_OK")
    ''')
    env = dict(os.environ)
    env.pop("JAX_ENABLE_X64", None)            # ensure x64 OFF in the child
    out = subprocess.run([sys.executable, "-c", script],
                         capture_output=True, text=True, env=env)
    assert "FP32_OK" in out.stdout, (out.stdout, out.stderr)
