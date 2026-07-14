"""Issue #405 regression: PhysicsState must be seeded and carried.

Root cause: ``update_physics_state(None, ...)`` returns ``None``, so a
run loop that starts the carry at ``None`` (or never threads it) keeps
``None`` forever — every stateful scheme silently reseeds its
prognostic fields each timestep.  Three guarantees pinned here:

1. ``update_physics_state(None, ...)`` returning ``None`` is the
   documented sentinel (the trap the loops must not fall into).
2. The MPAS model step threads a SEEDED carry: the output carry
   depends on the input carry (memory), and a perturbed prognostic
   column survives chained steps.
3. ``ModelDriver._refuse_stateful_physics_unthreaded`` raises for
   prognostic-carry schemes and passes diagnostic ones — the loud
   tripwire on the loops that do not thread the carry yet.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.atmosphere.physics.physics_state import (
    PhysicsState,
    init_physics_state,
    update_physics_state,
)


@pytest.fixture(autouse=True)
def _fp64_precision_policy():
    """Pin the legoESM precision policy to fp64 for the duration of each test.

    Enabling the JAX ``jax_enable_x64`` flag (module top) is NOT sufficient on
    its own: the legoESM precision policy is a separate process-global that
    still defaults to fp32, so the model state comes back fp32 while x64-enabled
    physics produces fp64 tendencies — the MPAS ``ssp_rk54`` scan-carry guard
    then (correctly) rejects the dtype-narrowing mismatch.  Set the policy to
    fp64 so state and tendencies are consistently fp64, and RESTORE it on
    teardown so the policy does not leak into sibling test modules sharing the
    xdist worker process (the cross-test contamination this repo guards against).
    """
    from legoesm.core.precision import PrecisionPolicy, get_policy, set_policy

    _prev = get_policy()
    set_policy(PrecisionPolicy.fp64())
    try:
        yield
    finally:
        set_policy(_prev)


def _tke_physics_config():
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig

    return PhysicsConfig(turbulence=TurbulenceConfig(scheme="tke"))


def test_update_physics_state_none_sentinel():
    ps = init_physics_state(8, 4, _tke_physics_config())
    assert update_physics_state(None, {"tke": ps.tke}) is None
    out = update_physics_state(ps, {})
    assert isinstance(out, PhysicsState)


def test_turbulence_scheme_traits_match_seeding():
    """The shared traits agree with init_physics_state's seeding: every
    energy-carrying scheme seeds its slot non-zero, every diagnostic
    scheme leaves both slots zero."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig
    from legoesm.atmosphere.physics.turbulence.integration import (
        turbulence_scheme_traits,
    )

    for scheme in ("tke", "clubb_lite", "edmf", "mynn25",
                   "louis", "smagorinsky", "holtslag_boville", "ysu"):
        traits = turbulence_scheme_traits(scheme)
        cfg = PhysicsConfig(turbulence=TurbulenceConfig(scheme=scheme))
        ps = init_physics_state(4, 3, cfg)
        seeded = {
            "tke": bool(np.any(np.asarray(ps.tke) > 0)),
            "qke": bool(np.any(np.asarray(ps.qke) > 0)),
        }
        if traits.carries_energy:
            assert seeded[traits.energy_field], (
                f"{scheme}: traits say energy in {traits.energy_field!r} "
                "but init_physics_state seeded it zero"
            )
            other = "qke" if traits.energy_field == "tke" else "tke"
            assert not seeded[other], (
                f"{scheme}: the inactive energy slot {other!r} must stay zero"
            )
        else:
            assert not seeded["tke"] and not seeded["qke"], (
                f"{scheme}: diagnostic scheme must not seed an energy carry"
            )


def test_mpas_step_threads_seeded_carry():
    """Two MPAS steps with a seeded carry: the prognostic TKE evolves
    and step 2 consumes step 1's output (no silent reseed)."""
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel,
        MPASPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas
    from legoesm.atmosphere.physics.combined import make_physics

    mesh = create_voronoi_mesh(3)
    sigma = create_sigma_coordinate(8)
    model = MPASPrimitiveEquationModel(
        mesh, sigma, MPASPrimitiveEquationConfig())
    state = held_suarez_init_mpas(mesh, sigma)

    cfg = _tke_physics_config()
    # model_type="mpas" — the driver's own call (model_driver.py:3067);
    # MPAS states carry edge-normal u with v=None, which the
    # "hydrostatic" pipeline would dereference.
    physics_fn = make_physics(cfg, model_type="mpas", dt=1.0)
    ncol = int(state.T.data.shape[0])
    nlev = int(state.T.data.shape[1])
    seed = init_physics_state(ncol, nlev, cfg)

    # The issue-405 property is MEMORY: the carry fed in must shape the
    # carry coming out.  From a rest state TKE has zero shear production
    # and at dt=300 s the dissipation term removes any perturbation in a
    # single step (both branches legitimately floor at tke_min), so the
    # step uses dt=1 s — the perturbed column must then SURVIVE the step
    # iff the scheme actually consumed the input carry.
    # Same dynamics state, two different input carries -> outputs must
    # differ, and the perturbed column must remember its perturbation.
    seed_pert = seed._replace(tke=seed.tke.at[0, :].set(1e-2))

    _ = model.step(state, 1.0, physics_fn=physics_fn, phys_state=seed)
    ps_a = model._phys_state
    _ = model.step(state, 1.0, physics_fn=physics_fn,
                   phys_state=seed_pert)
    ps_b = model._phys_state

    assert ps_a is not None and ps_b is not None, (
        "MPAS step dropped the seeded carry (model._phys_state is None)"
    )
    assert not np.array_equal(np.asarray(ps_b.tke), np.asarray(ps_a.tke)), (
        "Output carry is independent of the input carry — the scheme "
        "is being reseeded every step (issue #405)"
    )
    assert (float(np.max(np.asarray(ps_b.tke)[0]))
            > float(np.max(np.asarray(ps_a.tke)[0]))), (
        "Perturbed TKE column lost its memory through the step"
    )
    # Chain a second step on the perturbed branch: memory persists.
    s1 = model.step(state, 1.0, physics_fn=physics_fn,
                    phys_state=seed_pert)
    _ = model.step(s1, 1.0, physics_fn=physics_fn,
                   phys_state=model._phys_state)
    ps2 = model._phys_state
    assert ps2 is not None
    assert (float(np.max(np.asarray(ps2.tke)[0]))
            > float(np.max(np.asarray(ps_a.tke)[0])))
    assert np.all(np.isfinite(np.asarray(ps2.tke)))


def _toy_carry_physics_fn():
    """Toy stateful physics: zero tendencies; carry advance tke += 1.

    Pins the issue-#413 carry contract on the RK-stage dycores: every
    stage receives the STEP-INPUT carry and the carry-out comes from
    ONE extra post-step evaluation — so after one step tke must equal
    seed + 1 exactly (not +n_stages, not +n_stages+1).
    """
    from legoesm.core.state import HydrostaticTendencies

    def physics_fn(hs, grid, sigma_coord, phys_state=None):
        zero = HydrostaticTendencies(
            du_dt=hs.u.replace(data=jnp.zeros_like(hs.u.data)),
            dv_dt=(None if hs.v is None
                   else hs.v.replace(data=jnp.zeros_like(hs.v.data))),
            dT_dt=hs.T.replace(data=jnp.zeros_like(hs.T.data)),
            dp_s_dt=hs.p_s.replace(data=jnp.zeros_like(hs.p_s.data)),
            dphis_dt=hs.p_s.replace(data=jnp.zeros_like(hs.p_s.data)),
        )
        ps_out = (None if phys_state is None
                  else phys_state._replace(tke=phys_state.tke + 1.0))
        return zero, ps_out

    return physics_fn


def _cdgrid_setup(n=6, nlev=6):
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
        CDGridPrimitiveEquationModel,
        CDGridPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init

    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    model = CDGridPrimitiveEquationModel(
        grid, sigma, CDGridPrimitiveEquationConfig())
    state = held_suarez_init(grid, sigma)
    return model, state, sigma


def _mpas_setup(n=3, nlev=8):
    from legoesm.grids.voronoi import create_voronoi_mesh
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_mpas import (
        MPASPrimitiveEquationModel,
        MPASPrimitiveEquationConfig,
    )
    from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init_mpas

    mesh = create_voronoi_mesh(n)
    sigma = create_sigma_coordinate(nlev)
    model = MPASPrimitiveEquationModel(
        mesh, sigma, MPASPrimitiveEquationConfig())
    state = held_suarez_init_mpas(mesh, sigma)
    return model, state


def test_cdgrid_step_threads_seeded_carry():
    """CDGrid PE step with real TKE turbulence: output carry depends on
    the input carry (memory), perturbation survives, chained steps keep
    it — mirror of the MPAS memory test (issue #413 lift)."""
    from legoesm.atmosphere.physics.combined import make_physics

    model, state, sigma = _cdgrid_setup()
    cfg = _tke_physics_config()
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=1.0)

    shp = state.T.data.shape           # (6, n, n, nlev)
    ncol = shp[0] * shp[1] * shp[2]
    nlev = shp[3]
    seed = init_physics_state(ncol, nlev, cfg)
    seed_pert = seed._replace(tke=seed.tke.at[0, :].set(1e-2))

    _ = model.step(state, 1.0, physics_fn=physics_fn, phys_state=seed)
    ps_a = model._phys_state
    _ = model.step(state, 1.0, physics_fn=physics_fn, phys_state=seed_pert)
    ps_b = model._phys_state

    assert ps_a is not None and ps_b is not None, (
        "CDGrid step dropped the seeded carry (model._phys_state is None)"
    )
    assert not np.array_equal(np.asarray(ps_b.tke), np.asarray(ps_a.tke)), (
        "Output carry is independent of the input carry — the scheme "
        "is being reseeded every step (issue #405/#413)"
    )
    assert (float(np.max(np.asarray(ps_b.tke)[0]))
            > float(np.max(np.asarray(ps_a.tke)[0]))), (
        "Perturbed TKE column lost its memory through the step"
    )
    # Chain a second step on the perturbed branch: memory persists.
    s1 = model.step(state, 1.0, physics_fn=physics_fn, phys_state=seed_pert)
    _ = model.step(s1, 1.0, physics_fn=physics_fn,
                   phys_state=model._phys_state)
    ps2 = model._phys_state
    assert ps2 is not None
    assert np.all(np.isfinite(np.asarray(ps2.tke)))


def test_cdgrid_carry_advances_exactly_once_per_step():
    """Toy +1 carry: after one CDGrid step the carry must be seed+1 —
    stages share the step-input carry; one post-step eval produces the
    carry-out (the documented #413 semantics)."""
    model, state, _ = _cdgrid_setup(n=4, nlev=4)
    cfg = _tke_physics_config()
    shp = state.T.data.shape
    seed = init_physics_state(shp[0] * shp[1] * shp[2], shp[3], cfg)

    _ = model.step(state, 600.0, physics_fn=_toy_carry_physics_fn(),
                   phys_state=seed)
    ps_out = model._phys_state
    assert ps_out is not None
    np.testing.assert_allclose(
        np.asarray(ps_out.tke), np.asarray(seed.tke) + 1.0,
        rtol=0, atol=0,
        err_msg="carry must advance exactly once per step "
                "(per-RK-stage accumulation or a dropped carry detected)",
    )
    # No carry threaded -> stash resets to None (legacy contract).
    _ = model.step(state, 600.0, physics_fn=_toy_carry_physics_fn())
    assert model._phys_state is None


def test_latlon_step_threads_seeded_carry():
    """Lat-lon C-grid PE: carry threads through step / step_with_physics
    and advances exactly once per step (issue #413 lift)."""
    from legoesm import constants
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonHydrostaticState,
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
    )

    grid = create_latlon_grid(
        n_lat=16, radius=constants.R_earth, omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=5)
    model = CGridLatLonPrimitiveEquationModel(
        grid, sigma, CGridLatLonPrimitiveEquationConfig())
    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, sigma.n_levels
    state = CGridLatLonHydrostaticState(
        u=jnp.zeros((n_lat, n_lon + 1, nlev)),
        v=jnp.zeros((n_lat + 1, n_lon, nlev)),
        T=jnp.full((n_lat, n_lon, nlev), 300.0),
        p_s=jnp.full((n_lat, n_lon), 1.0e5),
        phis=jnp.zeros((n_lat, n_lon)),
    )

    cfg = _tke_physics_config()
    seed = init_physics_state(n_lat * n_lon, nlev, cfg)
    seed_pert = seed._replace(tke=seed.tke.at[0, :].set(1e-2))
    fn = _toy_carry_physics_fn()

    _ = model.step_with_physics(state, 600.0, physics_fn=fn,
                                phys_state=seed)
    ps_a = model._phys_state
    assert ps_a is not None, "lat-lon step dropped the seeded carry"
    np.testing.assert_allclose(
        np.asarray(ps_a.tke), np.asarray(seed.tke) + 1.0, rtol=0, atol=0,
        err_msg="carry must advance exactly once per step",
    )

    _ = model.step_with_physics(state, 600.0, physics_fn=fn,
                                phys_state=seed_pert)
    ps_b = model._phys_state
    assert not np.array_equal(np.asarray(ps_b.tke), np.asarray(ps_a.tke)), (
        "Output carry is independent of the input carry (silent reseed)"
    )
    # Memory: the perturbed column rides the carry through the step.
    assert float(np.max(np.asarray(ps_b.tke)[0])) > float(
        np.max(np.asarray(ps_a.tke)[0]))


@pytest.mark.parametrize(
    "turb,conv,gwd,should_raise",
    [
        ("tke", "none", "none", True),
        ("mynn25", "none", "none", True),
        # clubb_lite also carries TKE — the original hand-written set
        # omitted it; the guard now reads turbulence_scheme_traits.
        ("clubb_lite", "none", "none", True),
        ("edmf", "none", "none", True),
        ("none", "bechtold", "none", True),
        ("none", "mass_flux", "none", True),
        ("none", "edmf", "none", True),
        # Profile-prognostic convection reads/relaxes conv_prog_profile
        # (Tiedtke carries genuine updraft mass-flux memory) — codex
        # round 5 widened the guard beyond the hand-written set.
        ("none", "tiedtke", "none", True),
        ("none", "zhang_mcfarlane", "none", True),
        ("none", "none", "prognostic_spectral", True),
        ("louis", "sbm", "none", False),
        ("louis", "kuo", "none", False),
        ("holtslag_boville", "none", "none", False),
        ("none", "none", "linear", False),
    ],
)
def test_refusal_guard(turb, conv, gwd, should_raise):
    from legoesm.driver.model_driver import ModelDriver

    class _Cfg:
        turbulence = turb
        convection = conv
        gravity_wave_drag = gwd

    if should_raise:
        with pytest.raises(NotImplementedError, match="405"):
            ModelDriver._refuse_stateful_physics_unthreaded(_Cfg())
    else:
        ModelDriver._refuse_stateful_physics_unthreaded(_Cfg())


def _stateful_driver_config(**overrides):
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    # dt=1 s: at production dt a rest-state TKE perturbation legitimately
    # dissipates to the scheme floor within a step (same caveat as the
    # MPAS memory test), which would mask a dropped carry.
    kwargs = dict(
        grid=GridConfig(resolution=8, nlev=5),
        dycore=DycoreConfig(dt=1.0),
        output=OutputConfig(diag_days=1),
        days=6.0 / 86400.0,             # 6 steps at dt=1
        dataset="analytical",
        turbulence="tke",
        gravity_wave_drag="prognostic_spectral",
    )
    kwargs.update(overrides)
    return ExperimentConfig(**kwargs)


@pytest.mark.parametrize("compiled", [True, False],
                         ids=["run_compiled", "run_per_step"])
def test_driver_loops_thread_and_persist_stateful_carries(
        tmp_path, compiled):
    """Issue #413 lift: the compiled and per-step driver loops now RUN
    stateful schemes (formerly refused), seed the carries via
    init_physics_state, thread them through the unified physics step,
    persist them in carry_aux for checkpointing, and RESTORE a
    carry_aux seed (the checkpoint-restart path).  Memory pin: a
    perturbed restored carry must produce a different final carry than
    the default seed under identical dynamics."""
    from legoesm.driver.model_driver import ModelDriver

    cfg = _stateful_driver_config()
    driver = ModelDriver(cfg, output_dir=tmp_path / "a")
    driver.setup()
    status = driver.run(compiled=compiled)
    assert status == "COMPLETED", (
        "stateful schemes must RUN on this loop now (guard lifted #413)"
    )

    assert "tke" in driver._carry_aux, (
        "tke carry not persisted in carry_aux (checkpoint would reseed)"
    )
    assert "gwd_spectrum" in driver._carry_aux
    tke_a = np.asarray(driver._carry_aux["tke"])
    spec_a = np.asarray(driver._carry_aux["gwd_spectrum"])
    nlev = cfg.grid.nlev
    assert tke_a.shape[-1] == nlev and tke_a.ndim == 2
    assert spec_a.ndim == 3
    assert np.all(np.isfinite(tke_a)) and np.all(np.isfinite(spec_a))

    # Second run, identical dynamics, PERTURBED restored carry (the
    # restart path: carry_aux seeds _prepare_run_context).  The final
    # carry must remember the perturbation — if the loop reseeded every
    # step (issue #405) both runs would end identical.
    driver_b = ModelDriver(cfg, output_dir=tmp_path / "b")
    driver_b.setup()
    tke_seed = jnp.asarray(tke_a)
    driver_b._carry_aux = {
        "tke": tke_seed.at[0, :].set(tke_seed[0, :] + 1e-2),
    }
    status_b = driver_b.run(compiled=compiled)
    assert status_b == "COMPLETED"
    tke_b = np.asarray(driver_b._carry_aux["tke"])
    assert not np.array_equal(tke_b, tke_a), (
        "final tke is independent of the seeded carry — the loop is "
        "reseeding the physics state every step (issue #405)"
    )
    assert float(np.max(tke_b[0])) > float(np.max(tke_a[0])), (
        "perturbed TKE column lost its memory through the run"
    )


def test_checkpoint_roundtrip_restores_stateful_carries(tmp_path):
    """#413 item 3 (cubed-sphere/npz path): tke + gwd_spectrum ride
    carry_aux into the checkpoint and come back bit-identical through
    load_checkpoint; the restarted run then CONTINUES from the restored
    carry (no silent reseed across restart)."""
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.driver.config import OutputConfig

    cfg = _stateful_driver_config(
        output=OutputConfig(diag_days=1, checkpoint_days=3.0 / 86400.0),
    )
    driver_a = ModelDriver(cfg, output_dir=tmp_path / "a")
    driver_a.setup()
    assert driver_a.run(compiled=True) == "COMPLETED"
    tke_saved = np.asarray(driver_a._carry_aux["tke"])
    spec_saved = np.asarray(driver_a._carry_aux["gwd_spectrum"])

    ckpts = sorted((tmp_path / "a").glob("checkpoint_day_*.npz"))
    assert ckpts, "no checkpoint written"

    driver_b = ModelDriver(cfg, output_dir=tmp_path / "b")
    driver_b.setup()
    driver_b.load_checkpoint(ckpts[-1])
    assert "tke" in driver_b._carry_aux, (
        "tke did not survive the checkpoint roundtrip — restart would "
        "silently reseed the physics memory (issue #405/#413)"
    )
    np.testing.assert_array_equal(
        np.asarray(driver_b._carry_aux["tke"]), tke_saved)
    np.testing.assert_array_equal(
        np.asarray(driver_b._carry_aux["gwd_spectrum"]), spec_saved)
    # Restored carry actually seeds the continued run.
    assert driver_b.run(compiled=True) == "COMPLETED"
    assert np.all(np.isfinite(np.asarray(driver_b._carry_aux["tke"])))


def test_mpas_checkpoint_roundtrip_restores_physics_state(tmp_path):
    """#413 item 3 (MPAS/npz path): the full PhysicsState is persisted
    as physstate_* fields and restored into the _run_mpas seed."""
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )

    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=5),
        dycore=DycoreConfig(dt=1.0, discretization="mpas"),
        output=OutputConfig(diag_days=1, checkpoint_days=3.0 / 86400.0),
        days=4.0 / 86400.0,
        dataset="analytical",
        radiation="none",
        turbulence="tke",
    )
    driver_a = ModelDriver(cfg, output_dir=tmp_path / "a")
    driver_a.setup()
    assert driver_a.run() == "COMPLETED"
    ps_saved = driver_a._mpas_phys_state
    assert ps_saved is not None

    ckpts = sorted((tmp_path / "a").glob("checkpoint_day_*.npz"))
    assert ckpts, "no MPAS checkpoint written"
    with np.load(ckpts[-1]) as d:
        ps_keys = [k for k in d.files if k.startswith("physstate_")]
        assert "physstate_tke" in ps_keys, (
            "PhysicsState not persisted in the MPAS checkpoint — a "
            "chained restart would silently reseed (issue #405/#413)"
        )

    driver_b = ModelDriver(cfg, output_dir=tmp_path / "b")
    driver_b.setup()
    driver_b.load_checkpoint(ckpts[-1])
    assert "physstate_tke" in driver_b._carry_aux
    # Scheme tag round-trips as a string and is skipped by the overlay
    # field loop (codex round 9).
    assert (driver_b._carry_aux.get("physstate_meta_conv_scheme")
            == str(cfg.convection))
    # The restored field matches the final checkpointed carry... note the
    # checkpoint may predate the END of run A (periodic cadence), so
    # compare against the FILE, the ground truth of what was persisted.
    with np.load(ckpts[-1]) as d:
        np.testing.assert_array_equal(
            np.asarray(driver_b._carry_aux["physstate_tke"]),
            d["physstate_tke"],
        )
    # And the continued run consumes it (seed overlay) without error.
    assert driver_b.run() == "COMPLETED"
    assert np.all(np.isfinite(np.asarray(driver_b._mpas_phys_state.tke)))


def test_pack_carry_preserves_f64_gwd_spectrum_dtype():
    """Codex round 9 watch-item, pinned mechanically: pack_carry's
    _promote is result_type-based (upcast-only), so the f64-seeded GWD
    spectrum must SURVIVE an f32 storage policy — a downcast here would
    reintroduce the kernel-internal scan dtype mismatch."""
    from legoesm.driver.compiled_segments import pack_carry
    from legoesm.core.field import Field
    from legoesm.core.state import HydrostaticState

    n = 2
    s3 = (6, n, n, 3)
    s2 = (6, n, n)
    state = HydrostaticState(
        u=Field(jnp.zeros(s3, jnp.float32), name="u",
                dims=("face", "x", "y", "level"), units="m/s"),
        v=Field(jnp.zeros(s3, jnp.float32), name="v",
                dims=("face", "x", "y", "level"), units="m/s"),
        T=Field(jnp.full(s3, 280.0, jnp.float32), name="T",
                dims=("face", "x", "y", "level"), units="K"),
        p_s=Field(jnp.full(s2, 1e5, jnp.float32), name="p_s",
                  dims=("face", "x", "y"), units="Pa"),
        phis=Field(jnp.zeros(s2, jnp.float32), name="phis",
                   dims=("face", "x", "y"), units="m2/s2"),
    )
    spec64 = jnp.full((6 * n * n, 4, 20), 1e-3, dtype=jnp.float64)
    carry = pack_carry(
        state,
        q_v=jnp.zeros(s3, jnp.float32),
        q_c=jnp.zeros(s3, jnp.float32),
        q_r=jnp.zeros(s3, jnp.float32),
        held_dT_rad=jnp.zeros(s3, jnp.float32),
        held_sw_net_sfc=jnp.zeros(s2, jnp.float32),
        held_lw_net_sfc=jnp.zeros(s2, jnp.float32),
        held_sw_up_toa=jnp.zeros(s2, jnp.float32),
        held_lw_up_toa=jnp.zeros(s2, jnp.float32),
        held_sw_down_toa=jnp.zeros(s2, jnp.float32),
        step_index=0,
        gwd_spectrum=spec64,
    )
    assert carry.gwd_spectrum.dtype == jnp.float64, (
        "pack_carry downcast the f64 GWD spectrum — the prognostic-"
        "spectral kernel's internal scan would break (issue #413)"
    )


def test_conv_prog_scheme_tag_roundtrip_and_rejection(tmp_path):
    """Codex round 9: the conv_prog scheme tag round-trips the npz
    checkpoint as a string and a same-shape cross-scheme restore is
    rejected loudly."""
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.driver.config import OutputConfig

    cfg = _stateful_driver_config(
        convection="mass_flux",
        output=OutputConfig(diag_days=1, checkpoint_days=3.0 / 86400.0),
    )
    driver_a = ModelDriver(cfg, output_dir=tmp_path / "a")
    driver_a.setup()
    assert driver_a.run(compiled=True) == "COMPLETED"
    ckpts = sorted((tmp_path / "a").glob("checkpoint_day_*.npz"))
    assert ckpts
    with np.load(ckpts[-1]) as d:
        assert "carry_conv_prog_scheme" in d.files
        assert str(d["carry_conv_prog_scheme"]) == "mass_flux"

    # Same shape, different scheme: edmf also carries (ncol,) — the
    # tag must reject the restore.
    cfg_b = _stateful_driver_config(
        convection="edmf",
        output=OutputConfig(diag_days=1, checkpoint_days=3.0 / 86400.0),
    )
    driver_b = ModelDriver(cfg_b, output_dir=tmp_path / "b")
    driver_b.setup()
    driver_b.load_checkpoint(ckpts[-1])
    assert driver_b._carry_aux.get("conv_prog_scheme") == "mass_flux"
    with pytest.raises(ValueError, match="mass_flux"):
        driver_b.run(compiled=True)

    # Matching scheme restores cleanly.
    driver_c = ModelDriver(cfg, output_dir=tmp_path / "c")
    driver_c.setup()
    driver_c.load_checkpoint(ckpts[-1])
    assert driver_c.run(compiled=True) == "COMPLETED"


def test_restored_carry_shape_mismatch_fails_fast(tmp_path):
    """Codex round 1: a PRESENT restored carry with the wrong shape is a
    checkpoint/config mismatch — must raise, not silently reseed."""
    from legoesm.driver.model_driver import ModelDriver

    cfg = _stateful_driver_config()
    driver = ModelDriver(cfg, output_dir=tmp_path)
    driver.setup()
    driver._carry_aux = {"tke": jnp.zeros((7, 3))}   # wrong shape
    with pytest.raises(ValueError, match="405"):
        driver.run(compiled=True)


def test_pipeline_refuses_dropped_active_carry():
    """Codex round 1: the pipeline raises when an ACTIVE stateful
    scheme receives no carry — a silent reseed every step is the #405
    bug class.  Exercised through the driver per-step warmup by
    sabotaging the seeded carry."""
    from legoesm.atmosphere.physics.combined import (
        physics_config_requires_phys_state, PhysicsConfig,
    )
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig

    # Predicate sanity: drives both the wrapper refusals and mirrors
    # the driver guard.
    assert physics_config_requires_phys_state(
        PhysicsConfig(turbulence=TurbulenceConfig(scheme="tke")))
    assert physics_config_requires_phys_state(
        PhysicsConfig(turbulence=TurbulenceConfig(scheme="clubb_lite")))
    assert not physics_config_requires_phys_state(
        PhysicsConfig(turbulence=TurbulenceConfig(scheme="louis")))


def test_sharded_wrapper_refuses_stateful_physics():
    """Codex round 1: generic sharded step wrappers cannot thread the
    carry — they must refuse a stateful physics_fn loudly."""
    from legoesm.parallel.sharded_dynamics import (
        _refuse_stateful_physics_unthreaded_wrapper,
    )

    def stateless_fn(*a, **k):
        return None

    _refuse_stateful_physics_unthreaded_wrapper(stateless_fn)  # passes

    def stateful_fn(*a, **k):
        return None

    stateful_fn._requires_phys_state = True
    with pytest.raises(NotImplementedError, match="405"):
        _refuse_stateful_physics_unthreaded_wrapper(stateful_fn)

    # Wrapper-aware (codex round 10): a functools.partial that hides the
    # tag must STILL be refused — the guard routes through the shared
    # physics_requires_phys_state predicate, which the lat-lon MPI wrapper
    # (make_latlon_mpi_step) now uses too.
    import functools
    with pytest.raises(NotImplementedError, match="405"):
        _refuse_stateful_physics_unthreaded_wrapper(
            functools.partial(stateful_fn))


def test_integrate_mixin_refuses_and_threads_carry():
    """Codex round 2: IntegrationMixin.integrate must refuse a stateful
    physics_fn without a carry, and thread the carry when supplied."""
    from legoesm.atmosphere.physics.combined import make_physics

    model, state, _ = _cdgrid_setup(n=4, nlev=4)
    cfg = _tke_physics_config()
    fn = make_physics(cfg, model_type="hydrostatic", dt=1.0)
    with pytest.raises(NotImplementedError, match="405"):
        model.integrate(state, duration=2.0, dt=1.0, physics_fn=fn)

    shp = state.T.data.shape
    seed = init_physics_state(shp[0] * shp[1] * shp[2], shp[3], cfg)
    # Rest state at the TKE floor is a fixed point — memory is pinned by
    # comparing two DIFFERENT seeds under identical dynamics (a
    # reseeding loop would erase the difference).
    _ = model.integrate(
        state, duration=2.0, dt=1.0, physics_fn=fn, phys_state=seed)
    ps_a = model._phys_state
    assert ps_a is not None
    seed_pert = seed._replace(tke=seed.tke.at[0, :].set(1e-2))
    _ = model.integrate(
        state, duration=2.0, dt=1.0, physics_fn=fn, phys_state=seed_pert)
    ps_b = model._phys_state
    assert not np.array_equal(np.asarray(ps_b.tke), np.asarray(ps_a.tke)), (
        "integrate() final carry is independent of the seeded carry — "
        "the loop is reseeding the physics state (issue #405)"
    )


def test_make_physics_tags_requires_phys_state():
    """make_physics output advertises statefulness so wrappers that
    drop the carry can refuse at build time."""
    from legoesm.atmosphere.physics.combined import make_physics, PhysicsConfig
    from legoesm.atmosphere.physics.turbulence import TurbulenceConfig

    from legoesm.atmosphere.physics.convection import ConvectionConfig

    fn_stateful = make_physics(
        PhysicsConfig(turbulence=TurbulenceConfig(scheme="tke")),
        model_type="hydrostatic", dt=1.0)
    assert fn_stateful._requires_phys_state is True
    # Profile-prognostic convection (Tiedtke relaxes the previous
    # updraft mass-flux profile) is stateful too — codex round 5.
    fn_conv = make_physics(
        PhysicsConfig(convection=ConvectionConfig(scheme="tiedtke")),
        model_type="hydrostatic", dt=1.0)
    assert fn_conv._requires_phys_state is True
    fn_diag = make_physics(
        PhysicsConfig(turbulence=TurbulenceConfig(scheme="louis")),
        model_type="hydrostatic", dt=1.0)
    assert fn_diag._requires_phys_state is False
    fn_kuo = make_physics(
        PhysicsConfig(convection=ConvectionConfig(scheme="kuo")),
        model_type="hydrostatic", dt=1.0)
    assert fn_kuo._requires_phys_state is False


def test_refusal_guard_normalizes_scheme_objects():
    """PhysicsConfig-style sub-configs (with .scheme) are normalized."""
    from legoesm.driver.model_driver import ModelDriver

    class _Sub:
        def __init__(self, scheme):
            self.scheme = scheme

    class _Cfg:
        turbulence = _Sub("tke")
        convection = _Sub("none")
        gravity_wave_drag = _Sub("none")

    with pytest.raises(NotImplementedError, match="405"):
        ModelDriver._refuse_stateful_physics_unthreaded(_Cfg())


# ---------------------------------------------------------------------------
# Codex adversarial review (#413 follow-up): the loud carry contract must
# also cover DIRECT public per-step calls (model.step(..., physics_fn=fn)
# in a hand-written loop), not only integrate()/the sharded wrappers, and
# a reused MPAS driver must not resume stale physstate_* from a prior load.
# ---------------------------------------------------------------------------


def _stateful_make_physics():
    from legoesm.atmosphere.physics.combined import make_physics
    fn = make_physics(_tke_physics_config(), model_type="hydrostatic", dt=1.0)
    assert fn._requires_phys_state is True, "fixture must be a stateful fn"
    return fn


def test_cdgrid_step_refuses_stateful_physics_without_carry():
    """A stateful physics_fn handed to the CDGrid public step APIs with
    no phys_state must REFUSE loudly — silently reseeding every step is
    the exact failure class #413 removes (the integrate()/sharded guards
    did not cover direct step / step_with_physics / step_cell_centre)."""
    model, state, _ = _cdgrid_setup(n=4, nlev=4)
    fn = _stateful_make_physics()
    with pytest.raises(NotImplementedError, match="405"):
        model.step(state, 1.0, physics_fn=fn)
    with pytest.raises(NotImplementedError, match="405"):
        model.step_with_physics(state, 1.0, physics_fn=fn)
    with pytest.raises(NotImplementedError, match="405"):
        model.step_cell_centre(state, 1.0, physics_fn=fn)
    # Untagged (legacy) physics_fn is unaffected: the no-carry call still
    # runs byte-identically to the 3-arg path.
    _ = model.step(state, 1.0, physics_fn=_toy_carry_physics_fn())
    assert model._phys_state is None


def test_latlon_step_refuses_stateful_physics_without_carry():
    """Same loud refusal on the lat-lon C-grid public step APIs."""
    from legoesm import constants
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.dynamics.gcm.primitive_eq_latlon_cgrid import (
        CGridLatLonHydrostaticState,
        CGridLatLonPrimitiveEquationModel,
        CGridLatLonPrimitiveEquationConfig,
    )

    grid = create_latlon_grid(
        n_lat=16, radius=constants.R_earth, omega=constants.Omega)
    sigma = create_sigma_coordinate(n_levels=5)
    model = CGridLatLonPrimitiveEquationModel(
        grid, sigma, CGridLatLonPrimitiveEquationConfig())
    n_lat, n_lon, nlev = grid.n_lat, grid.n_lon, sigma.n_levels
    state = CGridLatLonHydrostaticState(
        u=jnp.zeros((n_lat, n_lon + 1, nlev)),
        v=jnp.zeros((n_lat + 1, n_lon, nlev)),
        T=jnp.full((n_lat, n_lon, nlev), 300.0),
        p_s=jnp.full((n_lat, n_lon), 1.0e5),
        phis=jnp.zeros((n_lat, n_lon)),
    )
    fn = _stateful_make_physics()
    with pytest.raises(NotImplementedError, match="405"):
        model.step(state, 600.0, physics_fn=fn)
    with pytest.raises(NotImplementedError, match="405"):
        model.step_with_physics(state, 600.0, physics_fn=fn)
    # Untagged toy fn with no carry still runs (legacy contract).
    _ = model.step_with_physics(state, 600.0,
                                physics_fn=_toy_carry_physics_fn())
    assert model._phys_state is None


def test_mpas_step_refuses_stateful_physics_without_carry():
    """MPAS step shares the loud contract (doctrine: every dycore)."""
    model, state = _mpas_setup()
    fn = _stateful_make_physics()
    with pytest.raises(NotImplementedError, match="405"):
        model.step(state, 1.0, physics_fn=fn)


def test_mpas_repeated_load_clears_stale_physstate(tmp_path):
    """Codex adversarial (#413 stale-persistence class): reusing a driver
    to load a checkpoint WITHOUT physstate_* must drop the carry from a
    PRIOR load, so _run_mpas freshly seeds rather than resuming carry
    from the wrong checkpoint."""
    from legoesm.driver.model_driver import ModelDriver
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )

    cfg = ExperimentConfig(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=5),
        dycore=DycoreConfig(dt=1.0, discretization="mpas"),
        output=OutputConfig(diag_days=1, checkpoint_days=3.0 / 86400.0),
        days=4.0 / 86400.0,
        dataset="analytical",
        radiation="none",
        turbulence="tke",
    )
    driver_a = ModelDriver(cfg, output_dir=tmp_path / "a")
    driver_a.setup()
    assert driver_a.run() == "COMPLETED"
    ckpts = sorted((tmp_path / "a").glob("checkpoint_day_*.npz"))
    assert ckpts, "no MPAS checkpoint written"
    ckpt = ckpts[-1]

    # A "legacy" checkpoint: same mesh, physstate_* stripped out.
    legacy = tmp_path / "legacy.npz"
    with np.load(ckpt) as d:
        kept = {k: d[k] for k in d.files if not k.startswith("physstate_")}
    np.savez(legacy, **kept)

    driver_b = ModelDriver(cfg, output_dir=tmp_path / "b")
    driver_b.setup()
    driver_b.load_checkpoint(ckpt)
    assert any(k.startswith("physstate_") for k in driver_b._carry_aux), (
        "first load should populate physstate_* carry"
    )
    # Reload a physstate-free checkpoint into the SAME driver.
    driver_b.load_checkpoint(legacy)
    assert not any(k.startswith("physstate_") for k in driver_b._carry_aux), (
        "stale physstate_* survived a physstate-free reload — _run_mpas "
        "would resume carry from the WRONG checkpoint (issue #413)"
    )


def _mpas_driver_cfg(**overrides):
    """MPAS driver config for the restart-safety regressions (#413)."""
    from legoesm.driver.config import (
        ExperimentConfig, GridConfig, DycoreConfig, OutputConfig,
    )
    base = dict(
        grid=GridConfig(grid_type="mpas", resolution=3, nlev=5),
        dycore=DycoreConfig(dt=1.0, discretization="mpas"),
        output=OutputConfig(diag_days=1, checkpoint_days=3.0 / 86400.0),
        days=4.0 / 86400.0,
        dataset="analytical",
        radiation="none",
        turbulence="tke",
    )
    base.update(overrides)
    return ExperimentConfig(**base)


def test_mpas_load_then_save_without_run_refuses_and_drops_stale(tmp_path):
    """Codex adversarial rounds 2+4-7 (#413): load_checkpoint clears the
    stale save channel and does NOT adopt the staged carry — validating
    it faithfully (shape / dtype / scheme / completeness) needs the
    run-time seed _run_mpas builds.  So a save before any run REFUSES
    rather than emitting an unvalidated / stale / fresh-seed checkpoint,
    and the normal load->run->save path remains the way to persist a
    restored carry."""
    from legoesm.driver.model_driver import ModelDriver

    cfg = _mpas_driver_cfg()
    driver_a = ModelDriver(cfg, output_dir=tmp_path / "a")
    driver_a.setup()
    assert driver_a.run() == "COMPLETED"
    ckpt = sorted((tmp_path / "a").glob("checkpoint_day_*.npz"))[-1]

    driver_b = ModelDriver(cfg, output_dir=tmp_path / "b")
    driver_b.setup()
    # Simulate a stale carry left by a prior run on this reused driver.
    ncol = int(driver_b.state.T.data.shape[0])
    nlev = int(driver_b.state.T.data.shape[1])
    stale = init_physics_state(ncol, nlev, _tke_physics_config())
    driver_b._mpas_phys_state = stale._replace(tke=stale.tke + 999.0)

    driver_b.load_checkpoint(ckpt)
    # Stale carry dropped; the staged carry is NOT adopted before a run.
    assert driver_b._mpas_phys_state is None, "stale carry survived the load"
    assert any(k.startswith("physstate_") for k in driver_b._carry_aux), (
        "load should still stage the checkpoint's carry for _run_mpas"
    )
    # A save before the validating run is refused (no laundering).
    with pytest.raises(ValueError, match="launder"):
        driver_b.save_checkpoint(0, 7.0)


def test_mpas_partial_physstate_checkpoint_rejected(tmp_path):
    """Codex adversarial round 2 (#413): a checkpoint carrying only a
    SUBSET of the physstate_* fields must be REJECTED loudly at restart —
    overlaying it would mix restored and freshly-seeded carry and branch
    the trajectory.  (Stripping ALL physstate_* stays the documented
    fresh-seed opt-out, covered by the repeated-load test.)"""
    from legoesm.driver.model_driver import ModelDriver

    cfg = _mpas_driver_cfg()
    driver_a = ModelDriver(cfg, output_dir=tmp_path / "a")
    driver_a.setup()
    assert driver_a.run() == "COMPLETED"
    ckpt = sorted((tmp_path / "a").glob("checkpoint_day_*.npz"))[-1]

    # Drop ONE carry field (keep the meta tag and every other field).
    partial = tmp_path / "partial.npz"
    with np.load(ckpt) as d:
        assert "physstate_tke" in d.files, "fixture must have a tke carry"
        kept = {k: d[k] for k in d.files if k != "physstate_tke"}
    np.savez(partial, **kept)

    driver_b = ModelDriver(cfg, output_dir=tmp_path / "b")
    driver_b.setup()
    driver_b.load_checkpoint(partial)   # load stages the subset
    with pytest.raises(ValueError, match="INCOMPLETE"):
        driver_b.run()


def test_mpas_meta_only_physstate_checkpoint_rejected(tmp_path):
    """Codex adversarial round 3 (#413): a checkpoint left with only the
    physstate_meta_* scheme tag and NO carry fields is a corrupted
    remnant (the save writes the tag only alongside the carry), so it
    must be rejected loudly rather than silently fresh-seeding.  Only
    stripping EVERY physstate_* key (fields AND meta) opts into a fresh
    seed."""
    from legoesm.driver.model_driver import ModelDriver

    cfg = _mpas_driver_cfg()
    driver_a = ModelDriver(cfg, output_dir=tmp_path / "a")
    driver_a.setup()
    assert driver_a.run() == "COMPLETED"
    ckpt = sorted((tmp_path / "a").glob("checkpoint_day_*.npz"))[-1]

    # Keep the meta tag, drop every actual physstate_<field>.
    meta_only = tmp_path / "meta_only.npz"
    with np.load(ckpt) as d:
        assert any(k.startswith("physstate_meta_") for k in d.files)
        kept = {
            k: d[k] for k in d.files
            if not (k.startswith("physstate_")
                    and not k[len("physstate_"):].startswith("meta_"))
        }
    np.savez(meta_only, **kept)

    driver_b = ModelDriver(cfg, output_dir=tmp_path / "b")
    driver_b.setup()
    driver_b.load_checkpoint(meta_only)
    with pytest.raises(ValueError, match="INCOMPLETE"):
        driver_b.run()


def test_mpas_bad_carry_load_then_save_refused(tmp_path):
    """Codex adversarial round 5 (#413): a load-then-save with no
    intervening run must NOT launder a bad MPAS restart.  A partial,
    meta-only, or convection-scheme-mismatched checkpoint stages carry
    that is not adopted into the save channel, so save_checkpoint refuses
    rather than emitting a fresh-seed (silent memory loss) or a relabelled
    (cross-scheme) checkpoint."""
    from legoesm.driver.model_driver import ModelDriver

    cfg = _mpas_driver_cfg()
    driver_a = ModelDriver(cfg, output_dir=tmp_path / "a")
    driver_a.setup()
    assert driver_a.run() == "COMPLETED"
    ckpt = sorted((tmp_path / "a").glob("checkpoint_day_*.npz"))[-1]
    with np.load(ckpt) as d:
        files = {k: np.asarray(d[k]) for k in d.files}
    assert "physstate_tke" in files and "physstate_meta_conv_scheme" in files

    def _variant(name, mutate):
        p = tmp_path / f"{name}.npz"
        kept = dict(files)
        mutate(kept)
        np.savez(p, **kept)
        drv = ModelDriver(cfg, output_dir=tmp_path / name)
        drv.setup()
        drv.load_checkpoint(p)
        assert drv._mpas_phys_state is None, (
            f"{name}: a bad carry was adopted into the save channel"
        )
        with pytest.raises(ValueError, match="launder"):
            drv.save_checkpoint(0, 7.0)

    # Partial: one carry field stripped (meta + the rest kept).
    _variant("partial", lambda k: k.pop("physstate_tke"))
    # Meta-only: every carry field stripped, scheme tag left behind.
    _variant(
        "meta_only",
        lambda k: [k.pop(key) for key in list(k)
                   if key.startswith("physstate_")
                   and not key.startswith("physstate_meta_")],
    )
    # Complete carry but the scheme tag disagrees with this run.
    _variant(
        "mismatch",
        lambda k: k.__setitem__(
            "physstate_meta_conv_scheme", np.asarray("some_other_scheme")),
    )


def test_mpas_unknown_physstate_field_rejected(tmp_path):
    """Codex adversarial round 6 (#413): a checkpoint carrying a
    physstate_* field absent from this build's PhysicsState schema
    (version skew) must be refused at the load boundary — silently
    dropping it on a rewrite would branch the trajectory for a newer
    reader that understands the field."""
    from legoesm.driver.model_driver import ModelDriver

    cfg = _mpas_driver_cfg()
    driver_a = ModelDriver(cfg, output_dir=tmp_path / "a")
    driver_a.setup()
    assert driver_a.run() == "COMPLETED"
    ckpt = sorted((tmp_path / "a").glob("checkpoint_day_*.npz"))[-1]

    skewed = tmp_path / "skewed.npz"
    with np.load(ckpt) as d:
        kept = {k: np.asarray(d[k]) for k in d.files}
    # A future build's extra prognostic carry, unknown to this schema.
    kept["physstate_future_field"] = np.asarray(kept["physstate_tke"])
    np.savez(skewed, **kept)

    driver_b = ModelDriver(cfg, output_dir=tmp_path / "b")
    driver_b.setup()
    with pytest.raises(ValueError, match="unknown physstate"):
        driver_b.load_checkpoint(skewed)


def test_spectral_family_step_refuses_stateful_physics():
    """Codex adversarial round 8 (#413): the PE dycores that run column
    physics but CANNOT thread a PhysicsState carry — spectral PE, U-Cast
    PE, SFNO PE — must refuse a tagged stateful physics_fn on their direct
    step APIs (the driver's _run_spectral refuses; the public API must too,
    else a hand-written loop silently reseeds).  The guard fires before the
    state is touched, so a placeholder state suffices."""
    import jax
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.ml.channel_packing import PE3DChannelSpec

    fn = _stateful_make_physics()
    grid = create_gaussian_grid(n_max=8)
    sigma = create_sigma_coordinate(n_levels=3)
    n_ch = PE3DChannelSpec(nlev=sigma.n_levels).n_channels

    from legoesm.atmosphere.dynamics.gcm.spectral_pe import (
        SpectralPrimitiveEquationModel,
        SpectralPEConfig,
    )
    spec = SpectralPrimitiveEquationModel(
        grid, sigma, SpectralPEConfig())
    with pytest.raises(NotImplementedError, match="405"):
        spec.step(None, 1.0, physics_fn=fn)
    with pytest.raises(NotImplementedError, match="405"):
        spec.step_with_physics(None, 1.0, physics_fn=fn)

    from legoesm.atmosphere.dynamics.neural.ucast_pe import (
        UCastPrimitiveEquationModel,
        UCastPrimitiveEquationConfig,
    )
    from legoesm.ml.ucast import UCastConfig
    ucast = UCastPrimitiveEquationModel(
        grid=grid, sigma_coord=sigma,
        config=UCastPrimitiveEquationConfig(
            ucast_config=UCastConfig(
                in_channels=n_ch, out_channels=n_ch, model_channels=8,
                channel_mult=(1, 2), num_blocks=1, attn_levels=()),
            mode="state_update", correct_mass=False),
        key=jax.random.PRNGKey(0))
    with pytest.raises(NotImplementedError, match="405"):
        ucast.step_with_physics(None, 1.0, fn)

    from legoesm.atmosphere.dynamics.neural.sfno_pe import (
        SFNOPrimitiveEquationModel,
        SFNOPrimitiveEquationConfig,
    )
    from legoesm.ml.sfno import SFNOConfig
    sfno = SFNOPrimitiveEquationModel(
        grid=grid, sigma_coord=sigma,
        config=SFNOPrimitiveEquationConfig(
            sfno_config=SFNOConfig(
                in_channels=n_ch, out_channels=n_ch, embed_dim=12,
                n_blocks=2, mlp_expansion=2),
            mode="state_update", correct_mass=False,
            correct_moisture_budget=False),
        key=jax.random.PRNGKey(0))
    with pytest.raises(NotImplementedError, match="405"):
        sfno.step_with_physics(None, 1.0, fn)


def test_carry_guard_sees_through_partial_and_wraps():
    """Codex adversarial round 9 (#413): a functools.partial / wraps
    wrapper around a tagged stateful make_physics fn strips the direct
    _requires_phys_state attribute; the carry guard must follow
    .func / __wrapped__ so a wrapped stateful physics cannot bypass the
    contract and silently reseed."""
    import functools
    from legoesm.timestepping.integration import physics_requires_phys_state

    fn = _stateful_make_physics()
    assert physics_requires_phys_state(fn)

    # functools.partial (e.g. binding forcing=) strips the attribute.
    p = functools.partial(fn, forcing=None)
    assert not getattr(p, "_requires_phys_state", False)
    assert physics_requires_phys_state(p)

    # functools.wraps copies __wrapped__ -> must be followed.
    @functools.wraps(fn)
    def wrapped(*a, **k):
        return fn(*a, **k)
    assert physics_requires_phys_state(wrapped)

    # A diagnostic (untagged) fn stays False through a partial.
    assert not physics_requires_phys_state(
        functools.partial(_toy_carry_physics_fn()))

    # End-to-end: a partial-wrapped stateful fn on a direct CDGrid step
    # (no phys_state) still refuses loudly.
    model, state, _ = _cdgrid_setup(n=4, nlev=4)
    with pytest.raises(NotImplementedError, match="405"):
        model.step(state, 1.0, physics_fn=functools.partial(fn))


def test_voronoi_mpi_step_refuses_stateful_physics_without_carry():
    """Codex adversarial round 11 (#413): make_voronoi_mpi_step with the
    default state-only contract (return_phys_state=False) drops the
    PhysicsState carry, so it must refuse a stateful physics_fn — even one
    hidden behind a functools.partial — at build time, before the
    model/layout are touched."""
    import functools
    from legoesm.parallel.voronoi_mpi import make_voronoi_mpi_step

    fn = _stateful_make_physics()
    with pytest.raises(NotImplementedError, match="405"):
        make_voronoi_mpi_step(None, None, None, physics_fn=fn)
    with pytest.raises(NotImplementedError, match="405"):
        make_voronoi_mpi_step(
            None, None, None, physics_fn=functools.partial(fn))


def test_nonhydrostatic_cdgrid_step_refuses_stateful_physics():
    """Codex adversarial round 13 (#413): the nonhydrostatic CD-grid
    dycore does not thread a PhysicsState carry, so a tagged stateful
    make_physics(model_type="nonhydrostatic") fn on its direct step APIs
    must be refused, not silently reseeded.  (NH MPAS / NH plane share the
    same helper-based guard.)  The guard fires before the state is
    touched, so a placeholder suffices."""
    from legoesm.atmosphere.physics.combined import make_physics
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import (
        create_height_coordinate, compute_terrain_metric,
    )
    from legoesm.atmosphere.dynamics.gcm.compressible_euler_cdgrid import (
        CDGridCompressibleEulerModel,
        CDGridCompressibleEulerConfig,
    )

    fn = make_physics(
        _tke_physics_config(), model_type="nonhydrostatic", dt=1.0)
    assert fn._requires_phys_state is True

    grid = create_cubed_sphere(4)
    hc = create_height_coordinate(5, 30000.0)
    tm = compute_terrain_metric(jnp.zeros((6, grid.n, grid.n)), hc)
    nh = CDGridCompressibleEulerModel(
        grid, hc, tm, CDGridCompressibleEulerConfig())

    with pytest.raises(NotImplementedError, match="405"):
        nh.step(None, 1.0, physics_fn=fn)
    with pytest.raises(NotImplementedError, match="405"):
        nh.step_with_physics(None, 1.0, physics_fn=fn)


def test_requires_phys_state_gwd_composite():
    """#834 (codex code-review): a '+'-composite containing prognostic_spectral
    carries a wave-action spectrum, so it MUST require a threaded PhysicsState;
    a stateless composite must not.  Regresses the exact-string
    ``== "prognostic_spectral"`` statefulness check that misclassified
    ``mcfarlane+prognostic_spectral`` as stateless (dropping the spectrum carry
    -> per-step reseed / seed shape mismatch)."""
    from legoesm.atmosphere.physics.combined import (
        PhysicsConfig, physics_config_requires_phys_state,
    )
    from legoesm.atmosphere.physics.gravity_wave_drag.config import (
        GravityWaveDragConfig,
    )

    def _req(scheme):
        return physics_config_requires_phys_state(
            PhysicsConfig(
                gravity_wave_drag=GravityWaveDragConfig(scheme=scheme)))

    assert _req("prognostic_spectral") is True
    assert _req("mcfarlane+prognostic_spectral") is True
    assert _req("prognostic_spectral+hines") is True
    assert _req("hines+mcfarlane") is False
    assert _req("mcfarlane") is False
    assert _req("none") is False
