"""CAM's macro/micro sub-cycle and the CLUBB liquid partition on the spectral lane.

Production (MPAS) runs `cld_macmic_num_steps=3` with `clubb_liquid_partition`:
CLUBB is seeded with the host cloud water, returns its own diagnosed liquid,
and the microphysics runs on the REPLACED value inside the same step. The
spectral training lane evaluated every module in parallel on one state and
refused the partition outright. These tests pin the ported behaviour.

The N=1 and N=3 results are checked against a reference assembled here from
the standalone module factories (parallel sum at N=1; CAM's sequential
turbulence -> microphysics loop at dt/N, state and carry handed on, at N=3),
so a regression in combined.py's accumulate or sub-cycle cannot pass by
agreeing with itself. The partition's routing is checked against the CLUBB
kernel's own output.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.spectral_pe import (  # noqa: E402
    isothermal_rest_state_spectral)
from legoesm.atmosphere.physics import combined as cb  # noqa: E402
from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics  # noqa: E402
from legoesm.atmosphere.physics.physics_state import init_physics_state  # noqa: E402
from legoesm.atmosphere.physics.microphysics.config import (  # noqa: E402
    MicrophysicsConfig, MorrisonConfig)
from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig  # noqa: E402
from legoesm.grids.gaussian import create_gaussian_grid  # noqa: E402
from legoesm.grids.vertical import create_sigma_coordinate  # noqa: E402

_N_MAX, _N_LEV, _DT = 8, 6, 1800.0


def _state(grid, sigma):
    shp = (grid.n_lat, grid.n_lon, _N_LEV)
    return isothermal_rest_state_spectral(
        grid, sigma, T_init=285.0, p_s_init=1.0e5, perturbation_amplitude=0.5,
        tracers={"q_v": jnp.full(shp, 6.0e-3), "q_c": jnp.full(shp, 2.0e-4),
                 "q_i": jnp.full(shp, 1.0e-5), "q_r": jnp.zeros(shp)})


def _cfg(partition=False, macmic=1):
    return PhysicsConfig(
        radiation=type(PhysicsConfig().radiation)(scheme="none"),
        convection=type(PhysicsConfig().convection)(scheme="none"),
        turbulence=TurbulenceConfig(
            scheme="clubb", liquid_partition=partition,
            clubb=CLUBBConfig(prognostic=partition)),
        microphysics=MicrophysicsConfig(
            scheme="morrison",
            morrison=MorrisonConfig(liquid_from_closure=partition,
                                    warm_rain_scheme="kk2000")),
        gravity_wave_drag=type(PhysicsConfig().gravity_wave_drag)(scheme="none"),
    ), macmic


@pytest.fixture(scope="module")
def setup():
    grid = create_gaussian_grid(_N_MAX, dealiasing="quadratic")
    sigma = create_sigma_coordinate(_N_LEV)
    return grid, sigma, _state(grid, sigma)


def _run(cfg, macmic, grid, sigma, state):
    fn = make_physics(cfg, model_type="spectral_pe", dt=_DT,
                      cld_macmic_num_steps=macmic)
    # Prognostic CLUBB refuses a markerless call (its moments would re-seed
    # every step), so hand it the carry the rollout threads.
    ncol = int(grid.n_lat) * int(grid.n_lon)
    ps = init_physics_state(ncol, _N_LEV, cfg)
    out = fn(state, grid, sigma, phys_state=ps)
    return out[0] if isinstance(out, tuple) else out


def _arr(t, k):
    v = t.tracers[k]
    return np.asarray(getattr(v, "data", v))


def _d(x):
    return getattr(x, "data", x)


def _advance(st, t, dt):
    """``st + dt * t`` on the fields the sub-cycle advances (phis fixed)."""
    kw = {k: getattr(st, k).replace(
              data=getattr(st, k).data + dt * getattr(t, k).data)
          for k in ("vor_hat", "div_hat", "T_hat", "lnps_hat")}
    kw["tracers"] = {k: (f.replace(data=f.data + dt * _d(t.tracers[k]))
                         if hasattr(f, "replace") else f + dt * _d(t.tracers[k]))
                     for k, f in st.tracers.items()}
    return st._replace(**kw)


def _carry(c):
    return c if isinstance(c, dict) else {"clubb_moments": c}


def _reference(cfg, grid, sigma, st, ps, n):
    """Turbulence + microphysics from their own factories, CAM order.

    Per-call inputs (the prescribed surface-flux overrides) are handed to
    EVERY sub-step and cleared from the returned carry, as production does.
    """
    from legoesm.atmosphere.physics.microphysics.integration import (
        make_microphysics_physics)
    from legoesm.atmosphere.physics.physics_state import (
        PHYSSTATE_PER_CALL_INPUTS, update_physics_state)
    from legoesm.atmosphere.physics.turbulence.integration import (
        make_turbulence_physics)
    turb = make_turbulence_physics(cfg.turbulence, "spectral_pe", _DT / n)
    micro = make_microphysics_physics(cfg.microphysics, "spectral_pe", _DT / n)
    fields = ("vor_hat", "div_hat", "T_hat", "lnps_hat")
    keep = {k: getattr(ps, k) for k in PHYSSTATE_PER_CALL_INPUTS}
    if n == 1:   # the parallel accumulate: both on the start-of-step state
        t_t, c = turb(st, grid, sigma, phys_state=ps)
        t_m = micro(st, grid, sigma)
        tend = {k: getattr(t_t, k).data + getattr(t_m, k).data for k in fields}
        tend.update({k: _d(t_t.tracers[k]) + _d(t_m.tracers[k])
                     for k in st.tracers})
        return tend, update_physics_state(ps, _carry(c))
    tend = {k: 0.0 for k in (*fields, *st.tracers)}
    for _ in range(n):
        t_t, c = turb(st, grid, sigma, phys_state=ps)
        st = _advance(st, t_t, _DT / n)
        ps = update_physics_state(ps, {**keep, **_carry(c)})
        t_m = micro(st, grid, sigma)
        st = _advance(st, t_m, _DT / n)
        for t in (t_t, t_m):
            for k in fields:
                tend[k] = tend[k] + getattr(t, k).data / n
            for k in st.tracers:
                tend[k] = tend[k] + _d(t.tracers[k]) / n
    return tend, update_physics_state(ps, {})


def _run_full(cfg, macmic, grid, sigma, state):
    fn = make_physics(cfg, model_type="spectral_pe", dt=_DT,
                      cld_macmic_num_steps=macmic)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    # Nonzero per-call surface-flux overrides: every sub-step must see them.
    ps = init_physics_state(ncol, _N_LEV, cfg)._replace(
        surface_wth_override=jnp.full((ncol,), 0.05),
        surface_wqv_override=jnp.full((ncol,), 2.0e-5))
    return fn(state, grid, sigma, phys_state=ps), ps


def _close(a, b, rtol, name):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (name, a.shape, b.shape)
    np.testing.assert_allclose(a, b, rtol=rtol,
                               atol=rtol * max(np.abs(b).max(), 1e-300),
                               err_msg=name)


def _assert_matches(out, ps_out, tend, ps_ref, rtol):
    for k in ("vor_hat", "div_hat", "T_hat", "lnps_hat"):
        _close(getattr(out, k).data, tend[k], rtol, k)
    assert set(out.tracers) == set(tend) - {"vor_hat", "div_hat", "T_hat",
                                            "lnps_hat"}
    for k, f in out.tracers.items():
        _close(_d(f), tend[k], rtol, k)
    # the returned carry is the LAST sub-step's, same tree, per-call cleared
    for name in ("clubb_moments", "cloud_fraction"):
        a, b = getattr(ps_out, name), getattr(ps_ref, name)
        assert (jax.tree.structure(a) == jax.tree.structure(b)), name
        assert (a is None) == (b is None), name
        jax.tree.map(lambda x, y: _close(x, y, rtol, name), a, b)
    from legoesm.atmosphere.physics.physics_state import (
        PHYSSTATE_PER_CALL_INPUTS)
    for k in PHYSSTATE_PER_CALL_INPUTS:
        assert getattr(ps_out, k) is None, k


def _with_closure(cfg, prognostic):
    clubb = cfg.turbulence.clubb._replace(prognostic=prognostic)
    return cfg._replace(turbulence=cfg.turbulence._replace(clubb=clubb))


@pytest.mark.parametrize("prognostic", [False, True])
def test_n1_is_the_parallel_sum_of_the_modules(setup, prognostic):
    # The partition needs N>=2, so N=1 is pinned without it, on both closures.
    grid, sigma, state = setup
    cfg = _with_closure(_cfg(partition=False)[0], prognostic)
    (out, ps_out), ps = _run_full(cfg, 1, grid, sigma, state)
    tend, ps_ref = _reference(cfg, grid, sigma, state, ps, 1)
    assert ps_ref.cloud_fraction is not None
    _assert_matches(out, ps_out, tend, ps_ref, 1e-12)


@pytest.mark.parametrize("partition", [False, True])
def test_n3_is_the_sequential_reference(setup, partition):
    grid, sigma, state = setup
    cfg, _ = _cfg(partition=partition)
    cfg = _with_closure(cfg, True)
    (out, ps_out), ps = _run_full(cfg, 3, grid, sigma, state)
    tend, ps_ref = _reference(cfg, grid, sigma, state, ps, 3)
    assert ps_ref.clubb_moments is not None
    # summation order only (combined adds w*r per module, the reference /n)
    _assert_matches(out, ps_out, tend, ps_ref, 1e-10)


def test_checkpointed_subcycle_has_the_uncheckpointed_gradient(setup, monkeypatch):
    """The per-sub-step jax.checkpoint must change memory, never the answer."""
    grid, sigma, state = setup
    # Cheap modules: the property is the checkpoint's, not the schemes'
    # (reverse-mode through CLUBB + Morrison x3 takes >20 min to compile).
    cfg, _ = _cfg()
    cfg = cfg._replace(turbulence=TurbulenceConfig(scheme="tke"),
                       microphysics=MicrophysicsConfig(scheme="kessler"))
    ncol = int(grid.n_lat) * int(grid.n_lon)
    ps = init_physics_state(ncol, _N_LEV, cfg)

    def grad_of_scale():
        fn = make_physics(cfg, model_type="spectral_pe", dt=_DT,
                          cld_macmic_num_steps=3)

        def loss(a):
            tr = dict(state.tracers)
            tr["q_v"] = (tr["q_v"].replace(data=tr["q_v"].data * a)
                         if hasattr(tr["q_v"], "replace") else tr["q_v"] * a)
            t, _ = fn(state._replace(tracers=tr), grid, sigma, phys_state=ps)
            return (jnp.sum(jnp.abs(t.T_hat.data) ** 2)
                    + jnp.sum(_d(t.tracers["q_c"]) ** 2))
        return jax.jit(jax.value_and_grad(loss))(1.0)

    v_ck, g_ck = grad_of_scale()
    calls = []

    def _plain(f, static_argnums=()):
        calls.append(f)
        return f
    monkeypatch.setattr(cb.jax, "checkpoint", _plain)
    v_pl, g_pl = grad_of_scale()
    assert calls, "the sub-cycle no longer calls jax.checkpoint"
    assert np.isfinite(float(g_ck)) and float(g_ck) != 0.0
    np.testing.assert_allclose(float(v_ck), float(v_pl), rtol=1e-12)
    np.testing.assert_allclose(float(g_ck), float(g_pl), rtol=1e-10)


def test_the_subcycle_changes_the_answer_and_stays_finite(setup):
    grid, sigma, state = setup
    cfg, _ = _cfg(partition=False)
    one = _run(cfg, 1, grid, sigma, state)
    three = _run(cfg, 3, grid, sigma, state)
    for t in (one, three):
        assert np.all(np.isfinite(_arr(t, "q_v")))
        assert np.all(np.isfinite(np.asarray(t.T_hat.data)))
    # Sequential application is a DIFFERENT operator from the parallel sum:
    # if these agreed, the sub-cycle would not be running. Compared
    # elementwise, NOT with allclose: most cells sit on the same
    # remove-all-the-cloud-water clamp (q_c/dt), so a tolerance-based
    # comparison passes while the unclamped cells differ by an order of
    # magnitude -- which is exactly the ordering effect under test.
    d_qc = np.abs(_arr(one, "q_c") - _arr(three, "q_c"))
    d_T = np.abs(np.asarray(one.T_hat.data) - np.asarray(three.T_hat.data))
    assert d_qc.max() > 0.0 and d_T.max() > 0.0
    # And the difference is not rounding: the largest disagreement is a
    # sizeable fraction of the largest tendency.
    assert d_qc.max() > 1.0e-3 * np.abs(_arr(three, "q_c")).max()


def test_the_partition_runs_on_the_spectral_lane_and_needs_the_subcycle(setup):
    grid, sigma, state = setup
    cfg, _ = _cfg(partition=True)
    out = _run(cfg, 3, grid, sigma, state)
    assert np.all(np.isfinite(_arr(out, "q_c")))
    # The closure's liquid reached the cloud-water tracer.
    assert np.max(np.abs(_arr(out, "q_c"))) > 0.0
    with pytest.raises(ValueError, match="cld_macmic_num_steps>=2"):
        _run(cfg, 1, grid, sigma, state)


def test_the_partition_refuses_a_state_with_no_cloud_tracer(setup):
    grid, sigma, _ = setup
    shp = (grid.n_lat, grid.n_lon, _N_LEV)
    bare = isothermal_rest_state_spectral(
        grid, sigma, T_init=285.0, p_s_init=1.0e5, perturbation_amplitude=0.0,
        tracers={"q_v": jnp.full(shp, 6.0e-3)})
    cfg, _ = _cfg(partition=True)
    with pytest.raises(ValueError, match="liquid_partition exchanges water"):
        _run(cfg, 3, grid, sigma, bare)


def test_the_subcycle_needs_both_sub_cycled_modules(setup):
    grid, sigma, _ = setup
    cfg, _ = _cfg(partition=False)
    cfg = cfg._replace(microphysics=MicrophysicsConfig(scheme="none"))
    with pytest.raises(ValueError, match="re-run a single module"):
        make_physics(cfg, model_type="spectral_pe", dt=_DT,
                     cld_macmic_num_steps=3)


def test_the_partition_routes_both_halves_of_the_kernel_exchange(setup, monkeypatch):
    """Both of CLUBB's water tendencies land on their tracers, unchanged, and
    the column total they carry is the kernel's surface moisture flux."""
    from legoesm import constants
    from legoesm.atmosphere.physics.turbulence import integration as ti
    grid, sigma, state = setup
    cfg, _ = _cfg(partition=True)
    seen = []
    orig = ti.get_turbulence_fn

    def spying(conf):
        name, fn, sc = orig(conf)

        def wrapped(*a, **k):
            r = fn(*a, **k)
            seen.append(r[0] if isinstance(r, tuple) else r)
            return r
        return name, wrapped, sc
    monkeypatch.setattr(ti, "get_turbulence_fn", spying)
    turb = ti.make_turbulence_physics(cfg.turbulence, "spectral_pe", _DT)
    ncol = int(grid.n_lat) * int(grid.n_lon)
    # A prescribed kinematic moisture flux makes the surface source live
    # (the isothermal fixture's own bulk flux is ~1e-7 kg/m2/s).
    ps = init_physics_state(ncol, _N_LEV, cfg)._replace(
        surface_wth_override=jnp.full((ncol,), 0.05),
        surface_wqv_override=jnp.full((ncol,), 2.0e-5))
    t, _ = turb(state, grid, sigma, phys_state=ps)
    (k_out,) = seen
    shp = (grid.n_lat, grid.n_lon, _N_LEV)
    np.testing.assert_array_equal(_arr(t, "q_c"),
                                  np.asarray(k_out.dq_c_dt).reshape(shp))
    np.testing.assert_array_equal(_arr(t, "q_v"),
                                  np.asarray(k_out.dq_v_dt).reshape(shp))
    assert np.abs(_arr(t, "q_c")).max() > 0.0
    # column water: sum_k (dq_v + dq_c) dp/g against LH / L_v
    from legoesm.atmosphere.physics.combined import spectral_pe_to_grid
    p_s = np.asarray(spectral_pe_to_grid(state, grid, sigma)["p_s"])
    dp = np.diff(np.asarray(sigma.pressure_at_half(p_s)), axis=-1)
    col = ((_arr(t, "q_v") + _arr(t, "q_c")) * dp).sum(-1) / constants.g
    evap = np.asarray(k_out.lhflx).reshape(shp[:2]) / constants.L_v
    assert np.abs(evap).max() > 1.0e-6   # kg/m2/s: a live surface flux
    np.testing.assert_allclose(col, evap, rtol=1e-3,
                               atol=1e-3 * np.abs(evap).max())


def test_the_saline_refusal_still_applies_to_the_spectral_lane():
    from legoesm.atmosphere.physics.turbulence.integration import (
        make_turbulence_physics)
    from legoesm.atmosphere.physics.turbulence.config import (
        SurfaceLayerConfig)
    turb = TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(
        prognostic=True, surface=SurfaceLayerConfig(ocean_q_sfc_saline=True)))
    with pytest.raises(NotImplementedError, match="ocean_q_sfc_saline"):
        make_turbulence_physics(turb, "spectral_pe", _DT)


@pytest.mark.parametrize("bad", [3.9, True, 0])
def test_the_wb_deck_subcycle_count_is_not_coerced(bad):
    """scale_build passes the deck's value raw, so make_physics refuses it."""
    import pathlib
    from types import SimpleNamespace

    import yaml
    from legoesm.training.scale_build import build_mode_components
    repo = pathlib.Path(__file__).resolve().parents[2]
    yml = yaml.safe_load((repo / "config" / "wb" / "campaign"
                          / "spectral_smoke.yaml").read_text())
    yml["classical"] = dict(yml.get("classical", {}),
                            cld_macmic_num_steps=bad)
    cfg = SimpleNamespace(mode="physics", training_core="spectral",
                          smoke=True, multi_step_hours=(6,), n_days=1)
    *_, params, mk, _loss, _dt = build_mode_components(cfg, yml)
    with pytest.raises(ValueError, match="cld_macmic_num_steps must be an int"):
        mk(params)


def test_the_partition_refuses_the_diagnostic_closure(setup):
    grid, sigma, _ = setup
    cfg, _ = _cfg(partition=True)
    cfg = cfg._replace(turbulence=TurbulenceConfig(
        scheme="clubb", liquid_partition=True,
        clubb=CLUBBConfig(prognostic=False)))
    with pytest.raises(ValueError, match=r"(?i)needs (the )?prognostic CLUBB"):
        make_physics(cfg, model_type="spectral_pe", dt=_DT,
                     cld_macmic_num_steps=3)


@pytest.mark.parametrize("split_rad", [True, False])
@pytest.mark.parametrize("n", [None, 3])
def test_the_classical_factory_threads_the_count_on_both_paths(
        setup, monkeypatch, split_rad, n):
    """Split and combined radiation both hand make_physics the deck's count
    (the combined path used to drop it); omitted means 1."""
    import sys
    sys.path.insert(0, "tests/unit")
    from test_spectral_cam6_cloud_routing import _cam6_bundle

    from legoesm.training.aimip_params import (
        make_aimip_classical_spectral_physics)
    grid, _sigma, _state = setup
    seen = []

    class _Stop(Exception):
        pass

    def spy(*a, **kw):
        seen.append(kw.get("cld_macmic_num_steps", 1))
        raise _Stop
    monkeypatch.setattr(cb, "make_physics", spy)
    bundle, schemes = _cam6_bundle("cam6_clubb")
    kw = {} if n is None else {"cld_macmic_num_steps": n}
    with pytest.raises(_Stop):
        make_aimip_classical_spectral_physics(
            bundle, grid, _DT, radiation="rrtmgp", split_rad=split_rad,
            rad_update_interval_steps=6, rrtmgp_gpoint_batch_size=8,
            convection_scheme=schemes["convection"],
            turbulence_scheme=schemes["turbulence"], gwd_scheme=schemes["gwd"],
            microphysics_scheme=schemes["microphysics"],
            cloud_scheme=schemes["cloud"], surface_bulk_scheme="constant",
            **kw)
    assert seen == [1 if n is None else n]
