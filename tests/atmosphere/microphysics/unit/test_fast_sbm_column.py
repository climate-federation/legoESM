"""fast_sbm as a switchable scheme: dispatch + column-operator physics.

Dispatch is exercised through the PUBLIC MicrophysicsConfig (CLAUDE.md
config-dispatch rule). Physics: condensation closure on (ncol, nlev)
fields, emergent autoconversion (cloud→rain mass transfer through the
resolved spectrum — no parameterized rate), clear-cell fixed point.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics import (
    FastSBMConfig,
    MicrophysicsConfig,
    fast_sbm_microphysics,
    make_zero_hydrometeors,
)
from legoesm.atmosphere.physics.microphysics.integration import (
    _get_microphysics_fn,
)
from legoesm.thermo import saturation_vapor_pressure

jax.config.update("jax_enable_x64", True)

NCOL, NLEV = 2, 3
P0, T0 = 9.0e4, 283.0
DT = 2.0


def _fields(rh, q_c=1.0e-3, q_r=0.0):
    T = jnp.full((NCOL, NLEV), T0)
    p = jnp.full((NCOL, NLEV), P0)
    e = rh * float(saturation_vapor_pressure(jnp.asarray(T0)))
    q_v = jnp.full((NCOL, NLEV), constants.epsilon * e / (P0 - e))
    rho = jnp.full((NCOL, NLEV), 1.1)
    hyd = make_zero_hydrometeors(NCOL, NLEV)
    hyd = hyd._replace(q_c=jnp.full((NCOL, NLEV), q_c),
                       q_r=jnp.full((NCOL, NLEV), q_r))
    p_half = jnp.zeros((NCOL, NLEV + 1))
    dz = jnp.full((NCOL, NLEV), 100.0)
    return T, q_v, hyd, p, p_half, rho, dz


def test_dispatch_via_public_config():
    cfg = MicrophysicsConfig(scheme="fast_sbm")
    name, fn, scheme_cfg = _get_microphysics_fn(cfg)
    assert name == "fast_sbm"
    assert fn is fast_sbm_microphysics
    assert isinstance(scheme_cfg, FastSBMConfig)
    with pytest.raises(ValueError, match="Unknown microphysics"):
        _get_microphysics_fn(MicrophysicsConfig(scheme="fast_sbmm"))


def test_resolves_through_production_registry():
    # Codex review item 17: validate_strict accepted fast_sbm but the
    # production MICROPHYSICS_REGISTRY omitted it → runtime KeyError.
    from legoesm.driver.kernel_registry import (
        MICROPHYSICS_REGISTRY, resolve_kernel)
    assert "fast_sbm" in MICROPHYSICS_REGISTRY
    assert resolve_kernel(MICROPHYSICS_REGISTRY, "fast_sbm") \
        is fast_sbm_microphysics


def test_unknown_collision_kernel_raises():
    T, q_v, hyd, p, p_half, rho, dz = _fields(1.02)
    with pytest.raises(ValueError, match="collision_kernel"):
        fast_sbm_microphysics(T, q_v, hyd, p, p_half, rho, dz, DT,
                              FastSBMConfig(collision_kernel="hal"))


def test_condensation_closure_on_fields():
    T, q_v, hyd, p, p_half, rho, dz = _fields(1.02)
    out = fast_sbm_microphysics(T, q_v, hyd, p, p_half, rho, dz, DT)
    assert out.dT_dt.shape == (NCOL, NLEV)
    # Heat closure per cell (condensation only — sedimentation moves
    # liquid without phase change).
    np.testing.assert_allclose(
        np.asarray(out.dT_dt),
        -(constants.L_v / constants.c_pd) * np.asarray(out.dq_v_dt),
        rtol=1e-10)
    # Column water closure: vapor loss = liquid gain + surface precip.
    col = lambda x: np.asarray(jnp.sum(x * rho * dz, axis=1))
    np.testing.assert_allclose(
        -col(out.dq_v_dt),
        col(out.dq_c_dt + out.dq_r_dt) + np.asarray(out.precipitation),
        rtol=1e-9)
    # Supersaturated: net condensation.
    assert np.all(np.asarray(out.dq_v_dt) < 0.0)
    # Warm-only: ice tendencies identically zero; precip nonnegative.
    np.testing.assert_array_equal(np.asarray(out.dq_i_dt), 0.0)
    assert np.all(np.asarray(out.precipitation) >= 0.0)


def test_emergent_autoconversion_dense_vs_thin():
    # Mass crossing KRDROP comes from resolved coalescence: a dense cloud
    # must convert far more cloud→rain than a thin one (no tuned
    # autoconversion threshold/rate anywhere in the scheme).
    T, q_v, hyd_thin, p, p_half, rho, dz = _fields(1.0, q_c=5.0e-5)
    _, _, hyd_dense, _, _, _, _ = _fields(1.0, q_c=2.0e-3)
    cfg = FastSBMConfig(collision_kernel="hall")
    out_thin = fast_sbm_microphysics(T, q_v, hyd_thin, p, p_half, rho, dz,
                                     DT, cfg)
    out_dense = fast_sbm_microphysics(T, q_v, hyd_dense, p, p_half, rho,
                                      dz, DT, cfg)
    # Rain production = column rain-mass gain + what already precipitated.
    col = lambda x: float(jnp.sum((x * rho * dz)[0]))
    rain_thin = col(out_thin.dq_r_dt) + float(out_thin.precipitation[0])
    rain_dense = col(out_dense.dq_r_dt) + float(out_dense.precipitation[0])
    assert rain_dense > 0.0
    assert rain_dense > 50.0 * max(rain_thin, 0.0) or rain_thin <= 0.0
    # At S = 0 coalescence+settling conserve liquid against precip:
    # column (dq_c + dq_r) + precip ≈ 0 (condensation contributes ~0).
    np.testing.assert_allclose(
        col(out_dense.dq_c_dt + out_dense.dq_r_dt)
        + float(out_dense.precipitation[0]),
        0.0, atol=5.0e-9 * float(jnp.sum((rho * dz)[0])))


def test_cloud_rain_boundary_matches_oracle():
    # Codex review item 6: oracle IF(KRR < KRDROP=15) (1-based) → bins
    # 1..14 cloud, 15.. rain ⇒ 0-based bins 0..13 cloud, 14.. rain. Put a
    # spectrum exactly at 0-based bin 14 (the 50 um bin) — its mass must
    # land in RAIN, not cloud. Probe the projection directly.
    from legoesm.atmosphere.physics.microphysics.fast_sbm import (
        bin_mixing_ratios_from_f, mass_density, mass_doubling_grid)
    from legoesm.atmosphere.physics.microphysics.fast_sbm.grid import KRDROP
    m = mass_doubling_grid()
    assert KRDROP == 15
    # Single delta at 0-based bin KRDROP-1 = 14 (1-based 15, the ~50um bin).
    f = jnp.zeros_like(m).at[KRDROP - 1].set(1.0e12)
    cloud_mask = jnp.arange(m.shape[0]) < (KRDROP - 1)
    qc = float(mass_density(jnp.where(cloud_mask, f, 0.0), m))
    qr = float(mass_density(jnp.where(~cloud_mask, f, 0.0), m))
    assert qc == 0.0 and qr > 0.0      # the 50um bin is RAIN
    # And bin 13 (1-based 14) is cloud.
    f2 = jnp.zeros_like(m).at[KRDROP - 2].set(1.0e12)
    qc2 = float(mass_density(jnp.where(cloud_mask, f2, 0.0), m))
    assert qc2 > 0.0


def test_prognostic_Nc_used_when_present():
    # Codex review item 10: a column carrying N_c should drive cloud
    # number, not the fixed cdnc. Two states, same q_c, different N_c →
    # different droplet sizes → measurably different rain production.
    T, q_v, hyd, p, p_half, rho, dz = _fields(1.0, q_c=1.0e-3)
    cfg = FastSBMConfig(collision_kernel="hall")
    low_N = hyd._replace(N_c=jnp.full((NCOL, NLEV), 3.0e7))   # big drops
    high_N = hyd._replace(N_c=jnp.full((NCOL, NLEV), 6.0e8))  # small drops
    out_low = fast_sbm_microphysics(T, q_v, low_N, p, p_half, rho, dz, DT,
                                    cfg)
    out_high = fast_sbm_microphysics(T, q_v, high_N, p, p_half, rho, dz, DT,
                                     cfg)
    col = lambda x, o: float(jnp.sum((x * rho * dz)[0])) \
        + float(o.precipitation[0])
    # Fewer/larger droplets coalesce faster → more rain.
    assert col(out_low.dq_r_dt, out_low) > col(out_high.dq_r_dt, out_high)


def test_clear_supersaturated_cell_activates_cloud():
    # With CCN activation a supersaturated CLEAR cell must form cloud
    # (number + condensed water) — not stay clear.
    T, q_v, hyd, p, p_half, rho, dz = _fields(1.05, q_c=0.0, q_r=0.0)
    out = fast_sbm_microphysics(T, q_v, hyd, p, p_half, rho, dz, DT)
    assert np.all(np.asarray(out.dN_c_dt) > 0.0)     # droplets nucleated
    assert np.all(np.asarray(out.dq_c_dt) > 0.0)     # cloud water grew
    assert np.all(np.asarray(out.dq_v_dt) < 0.0)     # vapor consumed
    # Total-water closure holds through activation + condensation + precip.
    col = lambda x: np.asarray(jnp.sum(x * rho * dz, axis=1))
    np.testing.assert_allclose(
        -col(out.dq_v_dt),
        col(out.dq_c_dt + out.dq_r_dt) + np.asarray(out.precipitation),
        rtol=1e-9)


def test_activation_self_limits_when_target_met():
    # Deficit activation (codex review iters 8-9, ADV-10-4): a cell whose
    # cloud number ALREADY meets/exceeds the Köhler target must NOT add new
    # droplets (an additive `ccn_number − n_existing` reservoir WOULD).
    # Feed N_c well above any plausible target directly — isolates the
    # self-limiting property from coalescence replenishment dynamics.
    T, q_v, hyd0, p, p_half, rho, dz = _fields(1.02, q_c=1.0e-3, q_r=0.0)
    cfg = FastSBMConfig()
    # Clear cell with NO existing cloud → full activation deficit.
    clear = _fields(1.02, q_c=0.0, q_r=0.0)[2]
    out_clear = fast_sbm_microphysics(T, q_v, clear, p, p_half, rho, dz, DT,
                                      cfg)
    dNc_clear = float(out_clear.dN_c_dt[0, 0])
    assert dNc_clear > 0.0
    # Same cell but already carrying 10× the CCN budget as cloud number.
    saturated = hyd0._replace(N_c=jnp.full((NCOL, NLEV), 10.0 * cfg.ccn_number))
    out_sat = fast_sbm_microphysics(T, q_v, saturated, p, p_half, rho, dz,
                                    DT, cfg)
    # Net dN_c may be negative (coalescence), but the ACTIVATION component is
    # zero: positive dN_c can't exceed a tiny fraction of the clear-cell one.
    assert float(out_sat.dN_c_dt[0, 0]) < 0.01 * dNc_clear


def test_subsaturated_clear_cell_fixed_point():
    # A clear SUBsaturated cell activates nothing and stays put.
    T, q_v, hyd, p, p_half, rho, dz = _fields(0.8, q_c=0.0, q_r=0.0)
    out = fast_sbm_microphysics(T, q_v, hyd, p, p_half, rho, dz, DT)
    for fld in (out.dT_dt, out.dq_v_dt, out.dq_c_dt, out.dq_r_dt):
        np.testing.assert_allclose(np.asarray(fld), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(out.dq_i_dt), 0.0, atol=1e-15)


def test_supercooled_cell_freezes_to_ice():
    # A SUPERSATURATED supercooled (T < 0 °C) cloudy cell: real condensation
    # (so the water budget isn't a near-zero cancellation) + freezing to ice
    # with fusion warming.
    T = jnp.full((NCOL, NLEV), constants.T_freeze - 20.0)
    p = jnp.full((NCOL, NLEV), P0)
    e = 1.05 * float(saturation_vapor_pressure(jnp.asarray(float(T[0, 0]))))
    q_v = jnp.full((NCOL, NLEV), constants.epsilon * e / (P0 - e))
    rho = jnp.full((NCOL, NLEV), 1.1)
    hyd = make_zero_hydrometeors(NCOL, NLEV)._replace(
        q_c=jnp.full((NCOL, NLEV), 1.0e-3),
        q_r=jnp.full((NCOL, NLEV), 5.0e-4))
    p_half = jnp.zeros((NCOL, NLEV + 1))
    dz = jnp.full((NCOL, NLEV), 100.0)
    out = fast_sbm_microphysics(T, q_v, hyd, p, p_half, rho, dz, DT)
    assert np.all(np.asarray(out.dq_i_dt) > 0.0)         # ice formed
    # Net heating exceeds the condensation-only part (fusion adds warming).
    assert np.all(np.asarray(out.dT_dt) > 0.0)
    # Total water closure incl. ice: −dq_v = dq_c+dq_r+dq_i+precip.
    col = lambda x: np.asarray(jnp.sum(x * rho * dz, axis=1))
    np.testing.assert_allclose(
        -col(out.dq_v_dt),
        col(out.dq_c_dt + out.dq_r_dt + out.dq_i_dt)
        + np.asarray(out.precipitation), rtol=1e-8)


def test_warm_cell_melts_carried_ice():
    # A warm (T > 0 °C) cell carrying q_i must MELT it: dq_i < 0, melt water
    # joins liquid, latent cooling, total water closes incl. ice.
    T, q_v, hyd, p, p_half, rho, dz = _fields(0.99, q_c=2.0e-4, q_r=0.0)
    hyd = hyd._replace(q_i=jnp.full((NCOL, NLEV), 5.0e-4))
    out = fast_sbm_microphysics(T, q_v, hyd, p, p_half, rho, dz, DT)
    assert np.all(np.asarray(out.dq_i_dt) < 0.0)         # ice melting away
    # Closure incl. ice: −dq_v = dq_c+dq_r+dq_i+precip.
    col = lambda x: np.asarray(jnp.sum(x * rho * dz, axis=1))
    np.testing.assert_allclose(
        -col(out.dq_v_dt),
        col(out.dq_c_dt + out.dq_r_dt + out.dq_i_dt)
        + np.asarray(out.precipitation), rtol=1e-8)


def test_supercooled_riming_grows_ice_from_cloud():
    # A supercooled cell carrying ICE + CLOUD liquid: riming (oracle
    # coll_xyx_lwf) collects cloud onto ice → more ice than the same cell
    # with no pre-existing ice (freezing alone), with extra fusion heat and
    # total-water closure incl. ice.
    T = jnp.full((NCOL, NLEV), constants.T_freeze - 15.0)
    p = jnp.full((NCOL, NLEV), P0)
    # Supersaturated → real condensation, so the closure isn't a near-zero
    # cancellation of the (much larger) riming/freezing internal transfers.
    e = 1.02 * float(saturation_vapor_pressure(jnp.asarray(float(T[0, 0]))))
    q_v = jnp.full((NCOL, NLEV), constants.epsilon * e / (P0 - e))
    rho = jnp.full((NCOL, NLEV), 1.1)
    base = make_zero_hydrometeors(NCOL, NLEV)._replace(
        q_c=jnp.full((NCOL, NLEV), 2.0e-3))
    with_ice = base._replace(q_i=jnp.full((NCOL, NLEV), 1.0e-3))
    p_half = jnp.zeros((NCOL, NLEV + 1))
    dz = jnp.full((NCOL, NLEV), 100.0)
    out_no_ice = fast_sbm_microphysics(T, q_v, base, p, p_half, rho, dz, DT)
    out_ice = fast_sbm_microphysics(T, q_v, with_ice, p, p_half, rho, dz, DT)
    col = lambda x: np.asarray(jnp.sum(x * rho * dz, axis=1))
    # More cloud→ice conversion (the ice collects cloud) than freezing-only.
    # Compare the cloud LOSS: with seed ice, more cloud is removed.
    assert np.all(col(out_ice.dq_c_dt) < col(out_no_ice.dq_c_dt))
    # Closure incl. ice.
    np.testing.assert_allclose(
        -col(out_ice.dq_v_dt),
        col(out_ice.dq_c_dt + out_ice.dq_r_dt + out_ice.dq_i_dt)
        + np.asarray(out_ice.precipitation), rtol=1e-7)


def test_warm_cell_grad_through_discarded_riming():
    # Riming runs unconditionally then where-selects on T<T_freeze (codex
    # iter-14 LOW): a WARM cell carrying ice+cloud must have finite
    # gradients even though its riming branch is discarded — guards the
    # where-trap on the unselected supercooled branch.
    T = jnp.full((NCOL, NLEV), constants.T_freeze + 5.0)
    p = jnp.full((NCOL, NLEV), P0)
    e = 1.0 * float(saturation_vapor_pressure(jnp.asarray(float(T[0, 0]))))
    q_v = jnp.full((NCOL, NLEV), constants.epsilon * e / (P0 - e))
    rho = jnp.full((NCOL, NLEV), 1.1)
    hyd = make_zero_hydrometeors(NCOL, NLEV)._replace(
        q_c=jnp.full((NCOL, NLEV), 1.0e-3),
        q_i=jnp.full((NCOL, NLEV), 5.0e-4))
    p_half = jnp.zeros((NCOL, NLEV + 1))
    dz = jnp.full((NCOL, NLEV), 100.0)

    def loss(Tx):
        out = fast_sbm_microphysics(Tx, q_v, hyd, p, p_half, rho, dz, DT)
        return jnp.sum(out.dT_dt ** 2)

    g = jax.grad(loss)(T)
    assert np.all(np.isfinite(np.asarray(g)))


def test_cold_cell_does_not_melt_ice():
    # A subfreezing cell leaves carried ice intact (no melt source).
    T = jnp.full((NCOL, NLEV), constants.T_freeze - 10.0)
    p = jnp.full((NCOL, NLEV), P0)
    e = 0.9 * float(saturation_vapor_pressure(jnp.asarray(float(T[0, 0]))))
    q_v = jnp.full((NCOL, NLEV), constants.epsilon * e / (P0 - e))
    rho = jnp.full((NCOL, NLEV), 1.1)
    hyd = make_zero_hydrometeors(NCOL, NLEV)._replace(
        q_i=jnp.full((NCOL, NLEV), 5.0e-4))
    out = fast_sbm_microphysics(T, q_v, hyd, p, jnp.zeros((NCOL, NLEV + 1)),
                                rho, jnp.full((NCOL, NLEV), 100.0), DT)
    np.testing.assert_allclose(np.asarray(out.dq_i_dt), 0.0, atol=1e-15)


def test_reconstruction_conserves_mass_when_floor_binds():
    # Codex iter-13 WARN: the thin-cloud mass floor changes the spectrum
    # SHAPE when it binds (mean droplet < 2 µm) but the exact-mass rescale
    # must still preserve q to roundoff. Drive the floor with tiny q + huge
    # N (sub-2µm mean) for cloud and ice.
    from legoesm.atmosphere.physics.microphysics.fast_sbm import (
        mass_density, mass_doubling_grid)
    from legoesm.atmosphere.physics.microphysics.fast_sbm.column import (
        _reconstruct_spectrum, _reconstruct_ice)
    m = mass_doubling_grid()
    cfg = FastSBMConfig()
    rho = jnp.asarray(1.1)
    z = jnp.asarray(0.0)
    for qc, Nc in [(1.0e-6, 1.0e10), (5.0e-5, 5.0e9), (1.0e-3, 1.0e8)]:
        f = _reconstruct_spectrum(jnp.asarray(qc), z, jnp.asarray(Nc), z,
                                  rho, m, cfg)
        assert float(mass_density(f, m) / rho) == pytest.approx(qc, rel=1e-12)
    for qr, Nr in [(1.0e-6, 1.0e7), (1.0e-4, 1.0e3)]:   # rain mode floor
        f = _reconstruct_spectrum(z, jnp.asarray(qr), z, jnp.asarray(Nr),
                                  rho, m, cfg)
        assert float(mass_density(f, m) / rho) == pytest.approx(qr, rel=1e-12)
    f_i = _reconstruct_ice(jnp.asarray(1.0e-6), rho, m, cfg)
    assert float(mass_density(f_i, m) / rho) == pytest.approx(1.0e-6,
                                                              rel=1e-12)


def test_float32_grad_dry_atmosphere():
    # Regression (iter 13): the end-to-end hydrostatic grad test surfaced
    # three float32 NaN-gradient traps over a DRY column — Köhler r_crit
    # (inf-branch + 1/u² reciprocal VJP overflow), the q_v=0 OPER2
    # singularity, and the thin-cloud reconstruction rescale (1/mass²). All
    # must stay finite at float32 with zero hydrometeors and zero vapor.
    # float32 arrays exercise the float32 numeric path (the overflow that
    # produced the NaN is a float32 range property, independent of the x64
    # config flag — under x64 these explicit-float32 ops still run in f32).
    ncol, nlev = 2, 5
    T = jnp.linspace(230.0, 300.0, nlev)[None, :].repeat(ncol, 0) \
        .astype(jnp.float32)
    p = jnp.full((ncol, nlev), 8.0e4, jnp.float32)
    dz = jnp.full((ncol, nlev), 100.0, jnp.float32)
    hyd = make_zero_hydrometeors(ncol, nlev, dtype=jnp.float32)
    ph = jnp.zeros((ncol, nlev + 1), jnp.float32)

    def loss(Tx):
        # rho(T) — reproduces the wrapper's T-dependence that triggered the
        # thin-cloud rescale gradient overflow.
        rho_x = (p / (constants.R_d * Tx)).astype(jnp.float32)
        out = fast_sbm_microphysics(Tx, jnp.zeros((ncol, nlev), jnp.float32),
                                    hyd, p, ph, rho_x, dz, 300.0)
        return jnp.sum(out.dT_dt ** 2)

    g = jax.grad(loss)(T)
    assert np.all(np.isfinite(np.asarray(g)))


def test_ice_aggregation_conserves_ice_mass():
    # Ice-ice aggregation (snow formation) redistributes ice to larger bins
    # but creates/destroys no ice mass and no phase change — a cold cell
    # carrying ice (no liquid, no vapor source) keeps q_i exactly (only
    # internal bin redistribution + any sedimentation).
    T = jnp.full((NCOL, NLEV), constants.T_freeze - 10.0)
    p = jnp.full((NCOL, NLEV), P0)
    e = 0.5 * float(saturation_vapor_pressure(jnp.asarray(float(T[0, 0]))))
    q_v = jnp.full((NCOL, NLEV), constants.epsilon * e / (P0 - e))
    rho = jnp.full((NCOL, NLEV), 1.1)
    hyd = make_zero_hydrometeors(NCOL, NLEV)._replace(
        q_i=jnp.full((NCOL, NLEV), 1.0e-3))
    p_half = jnp.zeros((NCOL, NLEV + 1))
    dz = jnp.full((NCOL, NLEV), 100.0)
    out = fast_sbm_microphysics(T, q_v, hyd, p, p_half, rho, dz, DT)
    # No vapor exchange (subsaturated, no liquid), no melt (cold): ice
    # change is aggregation (internal) — q_i tendency ~0 (aggregation
    # conserves total ice mass; ice does not sediment in this adapter).
    np.testing.assert_allclose(np.asarray(out.dq_i_dt), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(out.dq_v_dt), 0.0, atol=1e-12)


def test_multistep_total_water_conserved():
    # Run the full scheme (warm + ice + riming + aggregation) for many
    # steps feeding tendencies back, and verify total water (vapor + cloud
    # + rain + ice) minus accumulated surface precipitation is conserved
    # over the whole trajectory — validates the scheme as a stable,
    # conservative integrator, not just per-step.
    ncol, nlev = 1, 4
    T = jnp.full((ncol, nlev), constants.T_freeze + 2.0)
    p = jnp.full((ncol, nlev), P0)
    e = 1.04 * float(saturation_vapor_pressure(jnp.asarray(float(T[0, 0]))))
    q_v0 = jnp.full((ncol, nlev), constants.epsilon * e / (P0 - e))
    rho = jnp.full((ncol, nlev), 1.1)
    dz = jnp.full((ncol, nlev), 200.0)
    ph = jnp.zeros((ncol, nlev + 1))
    hyd = make_zero_hydrometeors(ncol, nlev)._replace(
        q_c=jnp.full((ncol, nlev), 5.0e-4))

    def total_water(qv, h):
        return float(jnp.sum((qv + h.q_c + h.q_r + h.q_i) * rho * dz))

    qv = q_v0
    tw0 = total_water(qv, hyd)
    accum_precip = 0.0
    dt = 5.0
    for _ in range(30):
        out = fast_sbm_microphysics(T, qv, hyd, p, ph, rho, dz, dt)
        qv = jnp.maximum(qv + dt * out.dq_v_dt, 0.0)
        hyd = hyd._replace(
            q_c=jnp.maximum(hyd.q_c + dt * out.dq_c_dt, 0.0),
            q_r=jnp.maximum(hyd.q_r + dt * out.dq_r_dt, 0.0),
            q_i=jnp.maximum(hyd.q_i + dt * out.dq_i_dt, 0.0))
        accum_precip += float(jnp.sum(out.precipitation * dt))
    tw1 = total_water(qv, hyd)
    # Closure over the trajectory: water now + what precipitated == start.
    # Clamps to nonnegative can only ADD water, so allow a small one-sided
    # slack but require tight two-sided agreement (no spurious source).
    assert (tw1 + accum_precip) == pytest.approx(tw0, rel=2e-3)


def test_column_jit_and_grad():
    T, q_v, hyd, p, p_half, rho, dz = _fields(1.02)

    @jax.jit
    def total_heating(qv):
        out = fast_sbm_microphysics(T, qv, hyd, p, p_half, rho, dz, DT)
        return jnp.sum(out.dT_dt)

    val = float(total_heating(q_v))
    g = jax.grad(total_heating)(q_v)
    assert np.isfinite(val) and val > 0.0
    assert np.all(np.isfinite(np.asarray(g)))
