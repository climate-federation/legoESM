"""``resolved_inactive_fields``: the production trainable set freezes out, by
name, exactly the core knobs the resolved options never read (measured in the
production MPAS physics: AD = FD = 0 for Morrison k_au/k_ac under kk2000 and
McFarlane h_topo with the SSO height wired; Morrison autocon_fact /
accre_enhan_fact are read only by the two kk2000 laws).  Each frozen name's reason is
pinned against the real kernel, with a control that the same fixture moves
when the option that reads it is selected."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.combined import PhysicsConfig  # noqa: E402
from legoesm.atmosphere.physics.gravity_wave_drag.config import (  # noqa: E402
    GravityWaveDragConfig,
    McFarlaneConfig,
)
from legoesm.atmosphere.physics.microphysics.config import (  # noqa: E402
    MicrophysicsConfig,
    MorrisonConfig,
)
from legoesm.training.aimip_params import (  # noqa: E402
    aimip_active_scheme_keys,
    resolved_inactive_fields,
)

_MICRO = {"atm.micro.MorrisonConfig." + f for f in ("k_au", "k_ac", "x_star")}
_KK = {"atm.micro.MorrisonConfig." + f
       for f in ("autocon_fact", "accre_enhan_fact")}
_HTOPO = "atm.gwd.McFarlaneConfig.h_topo"


def _cfg(warm_rain="kk2000", gwd="mcfarlane"):
    return PhysicsConfig(
        microphysics=MicrophysicsConfig(
            scheme="morrison",
            morrison=MorrisonConfig(warm_rain_scheme=warm_rain)),
        gravity_wave_drag=GravityWaveDragConfig(scheme=gwd))


def test_sets_follow_the_resolved_options():
    assert resolved_inactive_fields(_cfg(), subgrid_orography=True) == (
        _MICRO | {_HTOPO})
    assert resolved_inactive_fields(_cfg(), subgrid_orography=False) == _MICRO
    assert resolved_inactive_fields(
        _cfg("kk2000_cam6"), subgrid_orography=False) == _MICRO
    assert resolved_inactive_fields(
        _cfg("seifert_beheng"), subgrid_orography=False) == _KK
    assert resolved_inactive_fields(
        _cfg("seifert_beheng_sb2001"), subgrid_orography=False) == _MICRO | _KK
    assert _HTOPO not in resolved_inactive_fields(
        _cfg(gwd="lindzen"), subgrid_orography=True)
    assert _HTOPO in resolved_inactive_fields(
        _cfg(gwd="mcfarlane+hines"), subgrid_orography=True)


def test_frozen_names_leave_the_core_trainable_set():
    from legoesm.training.param_collector import build_trainable_params
    cfg = _cfg()
    keys = aimip_active_scheme_keys(cfg)
    frozen = resolved_inactive_fields(cfg, subgrid_orography=True)
    full = build_trainable_params(active_scheme_keys=keys, tier="core")
    cut = build_trainable_params(active_scheme_keys=keys, tier="core",
                                 exclude=tuple(sorted(frozen)))
    core_frozen = {"atm.micro.MorrisonConfig.k_au", "atm.micro.MorrisonConfig.k_ac",
                   _HTOPO}
    assert core_frozen <= set(full.raw_values)        # non-vacuous: they WERE in
    assert not (frozen & set(cut.raw_values))
    assert set(full.raw_values) - set(cut.raw_values) == core_frozen
    # tier 1 (user decision 2026-10-14): the kk2000 knobs are core and stay
    # trainable under kk2000; under seifert_beheng they leave, k_au/k_ac stay
    assert _KK <= set(cut.raw_values)
    sb = _cfg("seifert_beheng")
    cut_sb = build_trainable_params(
        active_scheme_keys=keys, tier="core", exclude=tuple(sorted(
            resolved_inactive_fields(sb, subgrid_orography=True))))
    assert set(full.raw_values) - set(cut_sb.raw_values) == _KK | {_HTOPO}


def _morrison_out(cfg):
    from legoesm.atmosphere.physics.microphysics.morrison import (
        morrison_microphysics,
    )
    from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
    rng = np.random.default_rng(3)
    n = 16
    q_c = jnp.asarray(rng.uniform(1e-4, 8e-4, n))[None]
    q_r = jnp.asarray(rng.uniform(1e-6, 2e-4, n))[None]
    rho = jnp.asarray(rng.uniform(0.8, 1.2, n))[None]
    z = jnp.zeros(q_c.shape)
    fields = dict(q_c=q_c, q_r=q_r, N_r=jnp.full(q_c.shape, 1e5))
    hyd = HydrometeorState(**{f: fields.get(f, z)
                              for f in HydrometeorState._fields})
    out = morrison_microphysics(
        jnp.full(q_c.shape, 288.0), jnp.full(q_c.shape, 1.0e-2), hyd,
        jnp.full(q_c.shape, 9.0e4), jnp.full((1, n + 1), 9.0e4), rho,
        jnp.full(q_c.shape, 200.0), 60.0, cfg)
    return np.concatenate([np.ravel(x) for x in jax.tree_util.tree_leaves(out)
                           if hasattr(x, "dtype")])


def test_morrison_sb_knobs_unread_under_kk2000():
    """One knob at a time: bitwise inert under kk2000, each one moves the
    output under seifert_beheng (the control that the fixture can see it)."""
    for knob in ("k_au", "k_ac", "x_star"):
        for law, moves in (("kk2000", False), ("seifert_beheng", True)):
            base = MorrisonConfig(warm_rain_scheme=law)
            a = _morrison_out(base)
            b = _morrison_out(base._replace(
                **{knob: 2.0 * getattr(base, knob)}))
            assert (not np.array_equal(a, b)) is moves, (knob, law)


def test_morrison_kk2000_knobs_unread_under_seifert_beheng():
    """One knob at a time: bitwise inert under both Seifert-Beheng laws, each
    one moves the output under both kk2000 laws (the control that the fixture
    can see it)."""
    for knob in ("autocon_fact", "accre_enhan_fact"):
        for law, moves in (("seifert_beheng", False),
                           ("seifert_beheng_sb2001", False),
                           ("kk2000", True), ("kk2000_cam6", True)):
            base = MorrisonConfig(warm_rain_scheme=law)
            a = _morrison_out(base)
            b = _morrison_out(base._replace(
                **{knob: 2.0 * getattr(base, knob)}))
            assert (not np.array_equal(a, b)) is moves, (knob, law)


def _mcfarlane_drag(h_topo, h_topo_col):
    from legoesm.atmosphere.physics.gravity_wave_drag.mcfarlane import (
        mcfarlane_gwd,
    )
    ncol, nlev = 1, 30
    p_half = jnp.linspace(100e2, 1000e2, nlev + 1)[None, :]
    p_full = 0.5 * (p_half[:, 1:] + p_half[:, :-1])
    T = jnp.full((ncol, nlev), 280.0)
    z_full = -7500.0 * jnp.log(p_full / 1000e2)
    z_half = -7500.0 * jnp.log(jnp.clip(p_half, 1.0, None) / 1000e2)
    out = mcfarlane_gwd(
        u=jnp.full((ncol, nlev), 25.0), v=jnp.zeros((ncol, nlev)), T=T,
        p_full=p_full, p_half=p_half, z_full=z_full, z_half=z_half,
        rho=p_full / (constants.R_d * T), lat=jnp.zeros((ncol,)), dt=600.0,
        config=McFarlaneConfig(h_topo=h_topo), h_topo_col=h_topo_col)
    return jnp.sum(out.du_dt ** 2)


def test_mcfarlane_h_topo_unread_with_sso_column():
    h0 = McFarlaneConfig().h_topo
    # 300 m: uncapped launch with the E3SM 2*sgh displacement (50-200 m give
    # zero drag, >= 400 m saturates), so the fixture can see a height change
    col = jnp.full((1,), 300.0)
    assert float(jax.grad(_mcfarlane_drag, argnums=1)(h0, col)[0]) != 0.0
    assert float(jax.grad(_mcfarlane_drag)(h0, col)) == 0.0
    # forward too: the source DEPTH is a discrete level choice AD cannot see
    assert float(_mcfarlane_drag(h0, col)) == float(_mcfarlane_drag(8 * h0, col))
    assert float(jax.grad(_mcfarlane_drag)(h0, None)) != 0.0   # fallback reads it
