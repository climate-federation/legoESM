"""CLUBB's two-sided cloud-liquid exchange with the host (CAM clubb_intr.F90).

The historical bridge hands the advanced total water back WHOLLY as vapour, so
the liquid the closure's own PDF diagnoses never reaches the host: the host then
takes its cloud FRACTION from CLUBB and its cloud WATER from a tracer CLUBB never
wrote.  ``CLUBBConfig.liquid_partition`` ports the reference's exchange:

* IN  (clubb_intr.F90:1546,1550) ``rt = q_v + q_c``, ``thl = (T - (L_v/c_pd) q_c)/exner``
* OUT (clubb_intr.F90:2159,2160) ``q_v = rt - rcm``, ``q_c := rcm`` (REPLACE)

The replace semantics are only reversible because of the seeding, so the two
halves are tested together.  Default off, and the off path must stay
byte-identical.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.physics.turbulence.clubb import (
    CLUBBConfig, clubb_turbulence_prognostic, init_clubb_moments,
    pack_clubb_moments,
)

_NCOL, _NLEV = 3, 32


def _column(seed=0, q_c_amp=0.0):
    """A stable L32-like column, top-down, with an optional cloud-liquid layer."""
    rng = np.random.default_rng(seed)
    sigma_full = (np.arange(_NLEV) + 0.5) / _NLEV            # top-down
    p_s = 1.0e5
    p_full = np.broadcast_to(sigma_full * p_s, (_NCOL, _NLEV)).copy()
    p_half = np.concatenate([[0.0], (np.arange(_NLEV) + 1.0) / _NLEV]) * p_s
    p_half = np.broadcast_to(p_half, (_NCOL, _NLEV + 1)).copy()
    z_half = 8000.0 * np.log(p_s / np.clip(p_half, 100.0, None))
    z_full = 8000.0 * np.log(p_s / p_full)
    T = 288.0 - 60.0 * (1.0 - sigma_full)
    T = np.broadcast_to(T, (_NCOL, _NLEV)).copy()
    # Near-saturated lower troposphere so the PDF actually makes cloud.
    q_v = np.broadcast_to(1.4e-2 * sigma_full ** 2, (_NCOL, _NLEV)).copy()
    u = np.broadcast_to(20.0 * (1.0 - sigma_full), (_NCOL, _NLEV)).copy()
    v = np.zeros((_NCOL, _NLEV))
    u += rng.normal(0.0, 0.1, u.shape)
    rho = p_full / (constants.R_d * T)
    # A cloud-liquid slab in the lower-middle troposphere.
    q_c = np.zeros((_NCOL, _NLEV))
    if q_c_amp:
        q_c[:, int(0.55 * _NLEV):int(0.8 * _NLEV)] = q_c_amp
    T_sfc = np.full(_NCOL, 289.0)
    q_sfc = np.full(_NCOL, 1.5e-2)
    j = jnp.asarray
    return dict(u=j(u), v=j(v), T=j(T), q_v=j(q_v), q_c=j(q_c),
                p_full=j(p_full), p_half=j(p_half), z_full=j(z_full),
                z_half=j(z_half), T_sfc=j(T_sfc), q_sfc=j(q_sfc), rho=j(rho))


def _run(config, col, dt=300.0, pass_qc=True):
    moments = pack_clubb_moments(
        init_clubb_moments(_NCOL, _NLEV, config, dtype=jnp.float64))
    kw = {"q_c": col["q_c"]} if (config.liquid_partition and pass_qc) else {}
    return clubb_turbulence_prognostic(
        col["u"], col["v"], col["T"], col["q_v"], moments,
        col["p_full"], col["p_half"], col["z_full"], col["z_half"],
        col["T_sfc"], col["q_sfc"], col["rho"], dt, config, **kw)


_OFF = CLUBBConfig(prognostic=True)
_ON = CLUBBConfig(prognostic=True, liquid_partition=True)


def test_default_is_off_and_emits_no_liquid():
    """The lever ships off, and the off path publishes no liquid tendency."""
    assert CLUBBConfig().liquid_partition is False
    out, _ = _run(_OFF, _column())
    assert out.dq_c_dt is None


def test_off_path_is_bit_identical_to_the_historical_bridge():
    """Nothing about an existing run changes: the off path is the old formula.

    The historical bridge is ``q_v_new = rtm``, ``T_new = thlm*exner``.  Recompute
    it here from the advanced moments and demand bitwise equality, so a future
    edit to the partition branch cannot leak into runs that never asked for it.
    """
    col = _column()
    out, packed = _run(_OFF, col)
    from legoesm.atmosphere.physics.turbulence.clubb import (
        exner_function, flip_vertical, unpack_clubb_moments,
    )
    st = unpack_clubb_moments(packed)
    exner = exner_function(col["p_full"])
    dt = 300.0
    q_ref = (flip_vertical(st.rtm) - col["q_v"]) / dt
    T_ref = (flip_vertical(st.thlm) * exner - col["T"]) / dt
    np.testing.assert_array_equal(np.asarray(out.dq_v_dt), np.asarray(q_ref))
    np.testing.assert_array_equal(np.asarray(out.dT_dt), np.asarray(T_ref))


def test_total_water_is_conserved_by_the_partition():
    """Vapour + liquid together move exactly the total water the closure did.

    This is the property that makes the split safe: the partition only decides
    WHERE the water sits, never how much there is.  A lane that routed dq_v_dt
    and dropped dq_c_dt would violate precisely this.
    """
    col = _column(q_c_amp=2.0e-4)
    dt = 300.0
    out_on, packed_on = _run(_ON, col, dt=dt)
    from legoesm.atmosphere.physics.turbulence.clubb import (
        flip_vertical, unpack_clubb_moments,
    )
    st = unpack_clubb_moments(packed_on)
    rt_new = flip_vertical(st.rtm)
    # d(q_v + q_c)/dt must equal d(rt)/dt about the SEEDED total water.
    lhs = np.asarray(out_on.dq_v_dt + out_on.dq_c_dt)
    rhs = np.asarray((rt_new - (col["q_v"] + col["q_c"])) / dt)
    np.testing.assert_allclose(lhs, rhs, rtol=0, atol=1e-18)


def test_liquid_is_replaced_not_accumulated():
    """``q_c`` after the step is the closure's rcm, as CAM writes it.

    CAM's ``ptend q(ixcldliq) = (rcm - q_cldliq)/hdtime`` REPLACES.  Applying the
    tendency must therefore land exactly on rcm regardless of what the host was
    carrying, which is also what makes the exchange reversible: a layer that
    dries out gets its liquid returned to vapour instead of ratcheting up.
    """
    from legoesm.atmosphere.physics.turbulence.clubb import (
        clubb_step, flip_vertical,
    )
    dt = 300.0
    wet = _column(q_c_amp=5.0e-4)
    dry = _column(q_c_amp=0.0)
    out_wet, _ = _run(_ON, wet, dt=dt)
    q_c_final = np.asarray(wet["q_c"] + dt * out_wet.dq_c_dt)
    assert np.all(q_c_final >= -1e-30), "replaced liquid must stay non-negative"

    # REPLACE, stated as an equality against the closure's own diagnosis rather
    # than as an inequality that accumulation could also satisfy.
    moments = init_clubb_moments(_NCOL, _NLEV, _ON, dtype=jnp.float64)
    _, _, _, _, _, diags = clubb_step(
        wet["u"], wet["v"], wet["T"], wet["q_v"], moments,
        wet["p_full"], wet["p_half"], wet["z_full"], wet["z_half"],
        wet["T_sfc"], wet["q_sfc"], wet["rho"], dt, _ON, q_c=wet["q_c"])
    rcm = np.asarray(flip_vertical(diags["rcm_grid"]))
    # 1 ULP, not exact: the host applies a TENDENCY, so the value round-trips
    # through q_c + dt*(rcm - q_c)/dt rather than being assigned.
    np.testing.assert_allclose(q_c_final, rcm, rtol=1e-15, atol=1e-20)
    # Accumulation would have landed on q_c + rcm, so show the two differ where
    # the host was already carrying liquid -- otherwise the equality above is
    # satisfied by both semantics and proves nothing.
    seeded = np.asarray(wet["q_c"]) > 0.0
    assert seeded.any()
    assert not np.allclose(rcm[seeded], (np.asarray(wet["q_c"]) + rcm)[seeded])

    # And a column with liquid must not be bit-identical to one without, or the
    # seeding never happened (this is the non-vacuity check for the IN half).
    out_dry, _ = _run(_ON, dry, dt=dt)
    assert not np.allclose(np.asarray(out_wet.dq_v_dt),
                           np.asarray(out_dry.dq_v_dt), atol=0, rtol=0)


def test_vapour_stays_non_negative():
    """``q_v = rt - rcm`` is safe because the closure clips rcm against rt."""
    col = _column(q_c_amp=3.0e-4)
    dt = 300.0
    out, _ = _run(_ON, col, dt=dt)
    q_v_final = np.asarray(col["q_v"] + dt * out.dq_v_dt)
    assert q_v_final.min() >= -1e-30


def test_moist_static_energy_round_trip_is_exact_without_cloud():
    """With no liquid anywhere the partition is the identity on T and q_v.

    Guards the convention: this port's ``thl`` satisfies ``T = exner*thl +
    (L_v/c_pd)*rcm`` (its PDF forms ``tl_i = thl_i*exner``), NOT CAM's
    ``T = exner*(thl + (L_v/c_pd)*rcm)``.  If the inverse used the wrong one the
    zero-cloud limit would still match, so this test is paired with the
    latent-heat test below, which does discriminate them.
    """
    # A deliberately dry, warm column: the PDF makes no cloud, so rcm == 0.
    col = _column()
    col["q_v"] = col["q_v"] * 1e-3
    col["q_sfc"] = col["q_sfc"] * 1e-3
    out_off, _ = _run(_OFF, col)
    out_on, _ = _run(_ON, col)
    np.testing.assert_allclose(np.asarray(out_on.dT_dt),
                               np.asarray(out_off.dT_dt), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(np.asarray(out_on.dq_v_dt),
                               np.asarray(out_off.dq_v_dt), rtol=1e-12, atol=1e-14)
    assert np.allclose(np.asarray(out_on.dq_c_dt), 0.0, atol=1e-18)


def test_latent_heat_accompanies_the_condensate():
    """Condensing liquid must warm the column by ``L_v/c_pd`` times its mass.

    The temperature and water halves of the partition are a single
    transformation; this pins the coefficient, so a sign flip or a stray
    ``exner`` on the latent term fails here rather than as a slow climate drift.
    """
    col = _column(q_c_amp=0.0)
    dt = 300.0
    out_off, packed_off = _run(_OFF, col, dt=dt)
    out_on, packed_on = _run(_ON, col, dt=dt)
    from legoesm.atmosphere.physics.turbulence.clubb import (
        flip_vertical, unpack_clubb_moments,
    )
    # Same seeding (q_c == 0), so the advanced moments are identical and the ONLY
    # difference between the two outputs is the outbound split.
    st_on = unpack_clubb_moments(packed_on)
    st_off = unpack_clubb_moments(packed_off)
    np.testing.assert_allclose(np.asarray(st_on.rtm), np.asarray(st_off.rtm),
                               rtol=1e-13, atol=1e-16)
    rcm = np.asarray(dt * out_on.dq_c_dt)          # q_c was zero, so this IS rcm
    assert rcm.max() > 1e-7, "fixture must actually make cloud to be meaningful"
    dT = np.asarray(out_on.dT_dt - out_off.dT_dt) * dt
    dq = np.asarray(out_on.dq_v_dt - out_off.dq_v_dt) * dt
    np.testing.assert_allclose(dq, -rcm, rtol=1e-12, atol=1e-18)
    np.testing.assert_allclose(
        dT, (constants.L_v / constants.c_pd) * rcm, rtol=1e-12, atol=1e-12)


def test_sub_cycle_carries_the_liquid_between_steps():
    """Each sub-step must seed from the liquid the previous one replaced.

    Without the carry the closure re-condenses from the host's ORIGINAL liquid
    every sub-step, which is the one-way ratchet the seeding exists to prevent.
    A sub-cycled run must therefore differ from a single-step one, and the
    difference must live where the cloud is.
    """
    from legoesm.atmosphere.physics.turbulence.clubb import (
        clubb_step, flip_vertical, virtual_temperature,
    )
    col = _column(q_c_amp=2.0e-4)
    dt = 1200.0
    n_sub = 4
    cfg_1 = _ON._replace(clubb_dt=dt)            # n_sub == 1
    cfg_n = _ON._replace(clubb_dt=dt / n_sub)    # n_sub == 4
    out_1, _ = _run(cfg_1, col, dt=dt)
    out_n, packed_n = _run(cfg_n, col, dt=dt)
    assert not np.allclose(np.asarray(out_1.dq_c_dt),
                           np.asarray(out_n.dq_c_dt), rtol=1e-6, atol=1e-12)

    # The real check: reproduce the scan by hand, sub-step by sub-step, seeding
    # each call from the liquid the previous one replaced. A carry that dropped
    # or reset the liquid would diverge from this immediately.
    dts = dt / n_sub
    u_c, v_c, T_c, q_c_col, ql = (col["u"], col["v"], col["T"], col["q_v"],
                                  col["q_c"])
    m = pack_clubb_moments(
        init_clubb_moments(_NCOL, _NLEV, cfg_n, dtype=jnp.float64))
    from legoesm.atmosphere.physics.turbulence.clubb import unpack_clubb_moments
    m = unpack_clubb_moments(m)
    for _ in range(n_sub):
        tv = jnp.maximum(virtual_temperature(T_c, q_c_col), cfg_n.T0 * 0.5)
        rho_c = col["p_full"] / (constants.R_d * tv)
        du, dv, dT, dq, m, dg = clubb_step(
            u_c, v_c, T_c, q_c_col, m, col["p_full"], col["p_half"],
            col["z_full"], col["z_half"], col["T_sfc"], col["q_sfc"],
            rho_c, dts, cfg_n, q_c=ql)
        u_c = u_c + dts * du
        v_c = v_c + dts * dv
        T_c = T_c + dts * dT
        q_c_col = q_c_col + dts * dq
        ql = ql + dts * dg["dq_c_dt"]
    # 1e-11, not bitwise: the scan and this Python loop are the same arithmetic
    # in a different fusion, so the last couple of digits move.  Still four
    # orders tighter than any carry bug, which would show up at O(1).
    np.testing.assert_allclose(np.asarray(col["q_c"] + dt * out_n.dq_c_dt),
                               np.asarray(ql), rtol=1e-11, atol=1e-18)
    np.testing.assert_allclose(np.asarray(col["q_v"] + dt * out_n.dq_v_dt),
                               np.asarray(q_c_col), rtol=1e-11, atol=1e-18)

    # ... and the sub-cycled answer still conserves total water against the
    # moments it returns, which "finite" never established.
    st = unpack_clubb_moments(packed_n)
    lhs = np.asarray(out_n.dq_v_dt + out_n.dq_c_dt)
    rhs = np.asarray(
        (flip_vertical(st.rtm) - (col["q_v"] + col["q_c"])) / dt)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-11, atol=1e-18)


def test_mismatched_pairing_raises_both_ways():
    """A caller that forgets q_c, or passes it with the lever off, is refused.

    Silence here is the dangerous outcome: the first would replace the host's
    liquid with a closure that never saw it, the second would tell the caller the
    liquid was exchanged when it was not.
    """
    from legoesm.atmosphere.physics.turbulence.clubb import clubb_step
    col = _column(q_c_amp=1.0e-4)
    moments = init_clubb_moments(_NCOL, _NLEV, _ON, dtype=jnp.float64)
    args = (col["u"], col["v"], col["T"], col["q_v"], moments,
            col["p_full"], col["p_half"], col["z_full"], col["z_half"],
            col["T_sfc"], col["q_sfc"], col["rho"], 300.0)
    with pytest.raises(ValueError, match="requires the host cloud"):
        clubb_step(*args, _ON)
    with pytest.raises(ValueError, match="liquid_partition=False"):
        clubb_step(*args, _OFF, q_c=col["q_c"])


def test_non_mpas_lanes_refuse_the_lever():
    """Lanes that cannot route the liquid refuse instead of destroying water."""
    from legoesm.atmosphere.physics.turbulence.integration import (
        make_turbulence_physics,
    )
    from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig
    tc = TurbulenceConfig(scheme="clubb", clubb=_ON)
    for lane in ("hydrostatic", "nonhydrostatic", "spectral_pe"):
        with pytest.raises(NotImplementedError, match="liquid_partition"):
            make_turbulence_physics(tc, model_type=lane)
    # The wired lane builds.
    make_turbulence_physics(tc, model_type="mpas")


def test_partition_uses_the_grid_mean_liquid_not_the_in_cloud_one():
    """The tracer written back must be GRID-MEAN water, not in-cloud water.

    ``compute_cloud_cover`` converts the closure's grid-mean ``rcm`` into the
    IN-LAYER value by dividing by a vertical cloud fraction <= 1 (the reference
    keeps the two as separate outputs; this port overwrites one with the other).
    The in-cloud value is therefore the LARGER number and is no longer bounded
    by total water the way ``clip_rcm`` bounded the grid mean, so writing it into
    a grid-mean host tracer would both overstate the liquid and let ``q_v = rt -
    rcm`` go negative.  This pins the right key.
    """
    from legoesm.atmosphere.physics.turbulence.clubb import (
        clubb_step, flip_vertical,
    )
    col = _column(q_c_amp=2.0e-4)
    dt = 300.0
    moments = init_clubb_moments(_NCOL, _NLEV, _ON, dtype=jnp.float64)
    _, _, _, _, _, diags = clubb_step(
        col["u"], col["v"], col["T"], col["q_v"], moments,
        col["p_full"], col["p_half"], col["z_full"], col["z_half"],
        col["T_sfc"], col["q_sfc"], col["rho"], dt, _ON, q_c=col["q_c"])
    out, _ = _run(_ON, col, dt=dt)
    grid = np.asarray(flip_vertical(diags["rcm_grid"]))
    in_cloud = np.asarray(flip_vertical(diags["rcm"]))
    applied = np.asarray(col["q_c"] + dt * out.dq_c_dt)
    # 1 ULP for the same reason as above: this is a tendency round trip.
    np.testing.assert_allclose(applied, grid, rtol=1e-15, atol=1e-20)
    # And the two really are different, or this test proves nothing.  They agree
    # in the cloud INTERIOR by construction -- compute_cloud_cover rewrites only
    # the levels at a cloud top, a cloud base or an isolated cloudy layer -- so
    # the column maxima can coincide.  The edges are where it bites.
    assert not np.allclose(grid, in_cloud, rtol=0, atol=0), (
        "fixture must contain a cloud edge for the in-layer conversion to bite")
    assert (in_cloud >= grid - 1e-18).all(), "in-cloud water is never the smaller"
    assert (in_cloud > grid + 1e-12).any(), "no level was actually rewritten"


def _mpas_cfg(**kw):
    """A minimal MPAS ExperimentConfig carrying the liquid partition."""
    from legoesm.driver.config import ExperimentConfig
    base = dict(clubb_liquid_partition=True, clubb_prognostic=True,
                turbulence="clubb", microphysics="morrison",
                cld_macmic_num_steps=3, cloud_q_c_diagnostic=None)
    base.update(kw)
    return base


def test_partition_requires_the_sequential_macro_micro_subcycle():
    """The replace is only correct if the microphysics reads the replaced water.

    CAM guarantees that by sequential-update splitting inside its macro/micro
    loop.  This model takes that order ONLY when the loop runs: at
    ``cld_macmic_num_steps=1`` the combined physics evaluates every module on the
    same start-of-step state and SUMS the tendencies, so the microphysical sinks
    would be computed from the pre-exchange liquid.  Autoconversion goes as
    roughly the 2.5th power of cloud water, so a 2-3x error there is nearly an
    order of magnitude in the sink, and nothing fails loudly because water is
    still conserved.  Hence a hard refusal rather than a warning.
    """
    import pytest
    from legoesm.driver.config import ExperimentConfig

    bad = ExperimentConfig(**_mpas_cfg(cld_macmic_num_steps=1))
    with pytest.raises(ValueError, match="cld_macmic_num_steps>=2"):
        bad.validate_strict()
    # ... and the same config with the sub-cycle on does not raise for THIS
    # reason, so the guard is about the sub-cycle and not about something else.
    good = ExperimentConfig(**_mpas_cfg(cld_macmic_num_steps=3))
    try:
        good.validate_strict()
    except ValueError as exc:
        assert "cld_macmic_num_steps>=2" not in str(exc)


def test_partition_refuses_the_diagnostic_condensate_floor():
    """The floor exists to compensate for the liquid this lever restores.

    Left on it would be added on top of the closure's own water, and because it
    materialises condensate with no matching vapour sink it also breaks the
    exact total-water pairing the exchange otherwise guarantees.
    """
    import pytest
    from legoesm.driver.config import ExperimentConfig

    bad = ExperimentConfig(**_mpas_cfg(cloud_q_c_diagnostic=5.0e-5))
    with pytest.raises(ValueError, match="cloud_q_c_diagnostic"):
        bad.validate_strict()


def test_the_host_applies_the_subcycled_modules_in_sequence():
    """Pin the ordering contract the guard above relies on.

    The refusal is only meaningful if the sub-cycle really is sequential-update
    with turbulence ahead of microphysics.  Read it off the module that owns the
    loop rather than trusting a comment: this is the property that makes the
    replace semantics correct, and a refactor that quietly made the sub-cycle
    parallel would leave the guard in place while removing what it guards.
    """
    import inspect
    from legoesm.atmosphere.physics import combined

    src = inspect.getsource(combined._make_hydrostatic_combined)
    # One module at a time, with the state advanced between them.
    assert 'for _fr in sel["sub"]:' in src
    assert "r_sub = _run([_fr], state_m, ps_m)" in src
    assert "state_m = _advance_state(state_m, r_sub[0]" in src
    # Turbulence is appended to the module list before microphysics, so the
    # closure writes the liquid the microphysics then consumes.
    i_turb = src.find('if config.turbulence.scheme != "none":')
    i_micro = src.find('if config.microphysics.scheme != "none":')
    assert 0 < i_turb < i_micro
    # And at N=1 the parallel branch really is taken, which is what the guard
    # exists for.
    assert "if _n_macmic == 1:" in inspect.getsource(combined)
