"""``run_omip_core2`` restart: CLI wiring + the warm-vs-fresh equivalence gate.

The driver can now ``--restart-save`` a resumable checkpoint and ``--restart-from``
it, so a wallclock-limited job chain integrates forward instead of re-paying the
~60-day cold-start spin-up every leg.

The load-bearing test here is :func:`test_two_leg_resume_matches_continuous`:
run N steps continuously, then run N/2 + restart + N/2 and compare.  It drives
the REAL ocean step (prognostic-TKE closure, so ``state.tke`` is a live carry),
the REAL ``step_sea_ice`` (12-field ``DynamicSeaIceState``, two-way coupled to
the ocean through the open-water fraction), and the REAL step-indexed CORE-II
forcing lookup ``run_omip_core2._idx_t`` — so the three things a restart can
silently lose (the TKE carry, the sea-ice pack, the step counter) all matter to
the answer.

NON-VACUITY is proven by mutation, not asserted: the three
``test_dropping_*_breaks_continuity`` cases deliberately corrupt one restored
piece each and assert the comparison FAILS.  If the equivalence test could pass
with a carry dropped, those tests go red.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.ice import SeaIceConfig, init_dynamic_ice_state, step_sea_ice
from legoesm.ice.state import DynamicSeaIceState
from legoesm.ocean.restart import load_run_restart, save_run_restart
from legoesm.ocean.state import OceanSurfaceForcing

from scripts.run.run_omip_core2 import (
    _build_arg_parser,
    _build_atm_to_surface_core2,
    _idx_t,
    orca1_zdftke_config,
)


N_LAT, N_LON, NLEV = 6, 8, 4
H_MAX = 1000.0
# dt=7200 s (2 h) with 8 steps spans 16 h, i.e. THREE of the 6-hourly CORE-II
# record bins that run_omip_core2._idx_t maps `step` onto: steps 1..8 give
# floor(step*7200/21600) = 0,0,1,1,1,2,2,2.
# A shorter dt or fewer steps would leave the record index pinned at 0 and the
# "drop the step counter" mutation below would be VACUOUS —
# test_the_carries_under_test_are_actually_live asserts the index really moves.
# Stability: c=sqrt(g*H)=100 m/s on a 45-deg lon spacing gives a barotropic
# Courant number of ~0.17 at this dt, and the vertical mixing is implicit.
DT = 7200.0
N_REC = 8                    # synthetic 6-hourly CORE-II-like records
U_MIN = 1.0                  # [m/s] wind-speed floor for the ice bulk fluxes
N_TOTAL = 8                  # continuous leg length (4 + 4 around the restart)


# ============================================================================
# argparse round-trip
# ============================================================================

def test_cli_round_trip_restart_flags():
    p = _build_arg_parser()
    d = p.parse_args([])
    assert d.restart_save is None and d.restart_from is None
    assert d.restart_every_days == 0.0
    a = p.parse_args(["--restart-save", "/tmp/r.npz",
                      "--restart-from", "/tmp/parent.npz",
                      "--restart-every-days", "5"])
    assert a.restart_save == "/tmp/r.npz"
    assert a.restart_from == "/tmp/parent.npz"
    assert a.restart_every_days == 5.0


def test_cli_restart_flags_are_documented():
    """Both flags carry help text (a bare flag is undiscoverable in --help)."""
    p = _build_arg_parser()
    helps = {a.dest: (a.help or "") for a in p._actions}
    assert "resumable" in helps["restart_save"].lower()
    assert "resume" in helps["restart_from"].lower()
    assert "cadence" in helps["restart_every_days"].lower()


# ============================================================================
# Source-revision provenance (codex r6 MEDIUM)
# ============================================================================

def _init_git_repo(root, content="x = 1\n"):
    """Create a one-commit git repo at ``root``; return a runner for it."""
    import subprocess

    def g(*a):
        return subprocess.run(["git", "-C", str(root), *a],
                              capture_output=True, text=True, check=True)

    g("init", "-q")
    g("config", "user.email", "t@example.invalid")
    g("config", "user.name", "restart test")
    g("config", "commit.gpgsign", "false")
    (root / "src.py").write_text(content)
    g("add", "src.py")
    g("commit", "-q", "-m", "init")
    return g


def test_source_revision_scopes_dirty_state_and_explicit_failure(tmp_path):
    """codex r6 MEDIUM: the old capture was ``git rev-parse HEAD`` with no
    ``-C``, so it resolved against the CWD — a job launched from ``$HOME`` or
    from a different worktree recorded ANOTHER repository's HEAD.  It also
    never detected a dirty working tree (a SHA describes committed content
    only) and failed SILENTLY to ``None``, which is indistinguishable from
    "this archive predates the field" — so both sides skipped the comparison
    and the resume looked checked when nothing had been checked.
    """
    from scripts.run.run_omip_core2 import (
        _SOURCE_REV_UNAVAILABLE, _source_revision,
    )

    a, b = tmp_path / "repo_a", tmp_path / "repo_b"
    a.mkdir()
    b.mkdir()
    ga = _init_git_repo(a, "x = 1\n")
    _init_git_repo(b, "x = 2\n")
    head_a = ga("rev-parse", "HEAD").stdout.strip()

    # SCOPING: each call describes the tree it was POINTED AT, independent of
    # the process CWD (the two repos have different HEADs by construction).
    assert _source_revision(a) == head_a
    assert _source_revision(b) != head_a

    # DIRTY: a modified TRACKED file means the SHA no longer describes the
    # code being executed.
    (a / "src.py").write_text("x = 999\n")
    assert _source_revision(a) == f"{head_a}-dirty"

    # ...but an UNTRACKED file is deliberately NOT dirty (git describe --dirty
    # semantics).  This repo always carries hundreds of untracked scratch
    # scripts, so counting them would pin the marker permanently on and make
    # the warning uninformative.
    ga("checkout", "--", "src.py")
    (a / "scratch.sbatch").write_text("#!/bin/bash\n")
    assert _source_revision(a) == head_a

    # EXPLICIT FAILURE: a path that cannot be resolved records a value, never
    # None — so the resume can say the check could not RUN.
    missing = _source_revision(tmp_path / "does_not_exist")
    assert missing == _SOURCE_REV_UNAVAILABLE
    assert missing is not None


def test_source_revision_of_this_checkout_is_recorded_and_scoped():
    """The production call takes no argument and must describe THIS driver's
    checkout — i.e. the tree that supplies ``run_omip_core2.py``."""
    import subprocess

    from scripts.run import run_omip_core2 as _c2

    here = Path(_c2.__file__).resolve().parent
    got = _c2._source_revision()
    probe = subprocess.run(["git", "-C", str(here), "rev-parse", "HEAD"],
                           capture_output=True, text=True)
    if probe.returncode != 0:              # not a checkout (e.g. installed)
        assert got == _c2._SOURCE_REV_UNAVAILABLE
    else:
        assert got.startswith(probe.stdout.strip())
        assert got == probe.stdout.strip() or got.endswith("-dirty")


def test_resume_drift_note_distinguishes_unknown_from_a_match():
    """codex r6 MEDIUM: with the revision stored as ``None`` on failure, "the
    check could NOT run" was indistinguishable from "the check ran and
    matched" — both sides were falsy, the comparison was skipped, and the
    resume looked verified.  The three outcomes must stay distinct.

    Behavioural, not source-inspecting: it calls the function ``main`` calls.
    """
    from scripts.run.run_omip_core2 import (
        _SOURCE_REV_UNAVAILABLE, _source_revision_drift_note as note,
    )

    sha, other = "a" * 40, "b" * 40

    # Equal + clean: the only case that may be silent.
    assert note(sha, sha) is None

    # Genuine drift: warns and names BOTH revisions.
    n = note(other, sha)
    assert n is not None and other[:20] in n and sha[:20] in n

    # Either side unknown -> must NOT read as "checked and equal".
    for pair in ((None, sha), ("", sha), (_SOURCE_REV_UNAVAILABLE, sha),
                 (sha, _SOURCE_REV_UNAVAILABLE),
                 (_SOURCE_REV_UNAVAILABLE, _SOURCE_REV_UNAVAILABLE)):
        n = note(*pair)
        assert n is not None, f"{pair} silently skipped the drift check"
        assert "SKIPPED" in n, f"{pair} did not say the check could not run"

    # Equal but DIRTY: identical markers do not identify identical code, so
    # this may not be silent either.
    n = note(f"{sha}-dirty", f"{sha}-dirty")
    assert n is not None and "dirty" in n


def test_scan_lane_refuses_restarts():
    """codex r2 HIGH: the --scan-block body steps model._step_impl directly and
    never seeds the scan carry, so it can PROMOTE an optional slot None->Field
    mid-block that the restart neither records nor advances.  Restarts on that
    lane are refused outright; the guard text must name the flag and the fix."""
    import inspect

    from scripts.run import run_omip_core2 as _c2

    src = inspect.getsource(_c2.main)
    assert "--restart-save/--restart-from is not supported with" in src, (
        "the scan-lane restart refusal is gone from main()")
    # And it must sit INSIDE the use_scan branch, before the block builder.
    lines = src.splitlines()
    scan = next(i for i, ln in enumerate(lines) if ln.strip() == "if use_scan:")
    guard = next(i for i, ln in enumerate(lines)
                 if "is not supported with" in ln)
    build = next(i for i, ln in enumerate(lines)
                 if "build_omip2_scan_block_fn(" in ln and "import" not in ln)
    assert scan < guard < build, (
        "the refusal must execute inside the scan branch before the lane runs")


# ============================================================================
# Two-leg warm-vs-fresh equivalence
# ============================================================================

def _setup():
    """Tiny lat-lon C-grid ocean with the PROGNOSTIC TKE closure + dynamic ice."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.dynamics.ocean_model_latlon_cgrid import (
        LatLonCGridOceanConfig, LatLonCGridOceanModel,
    )
    from legoesm.ocean.init_latlon_cgrid import rest_state_latlon_cgrid_ocean
    from legoesm.ocean.physics.combined import OceanPhysicsConfig
    from legoesm.ocean.physics.vertical_mixing.config import VerticalMixingConfig
    from legoesm.ocean.vertical import create_ocean_z_star

    grid = create_latlon_grid(n_lat=N_LAT, n_lon=N_LON)
    z_coord = create_ocean_z_star(n_levels=NLEV, H_max=H_MAX)
    # prognostic=True makes state.tke a LIVE carry (Mode-A: one backward-Euler
    # TKE solve per model step, seeded from the carried field) — without it the
    # "drop tke" mutation below would be vacuous.  The closure is read from
    # config.physics.vertical_mixing (what _tke_prognostic_active inspects),
    # with every OTHER physics module off so the test exercises the carry, not
    # a full physics stack.
    from legoesm.ocean.physics.bottom_drag.config import BottomDragConfig
    from legoesm.ocean.physics.lateral_mixing.config import LateralMixingConfig
    from legoesm.ocean.physics.surface_forcing.config import (
        SurfaceForcingConfig,
    )
    vmix = VerticalMixingConfig(scheme="tke",
                                tke=orca1_zdftke_config(prognostic=True))
    phys = OceanPhysicsConfig(
        vertical_mixing=vmix,
        lateral_mixing=LateralMixingConfig(scheme="none"),
        surface_forcing=SurfaceForcingConfig(scheme="none"),
        bottom_drag=BottomDragConfig(scheme="none"),
        shortwave_penetration=None,
    )
    cfg = LatLonCGridOceanConfig(implicit_vertical_mixing=True, physics=phys)
    model = LatLonCGridOceanModel(grid, z_coord, cfg)
    state = rest_state_latlon_cgrid_ocean(
        grid, z_coord, T_water_init_C=-1.0, T_deep=1.0, S_uniform=34.0,
        H_max=H_MAX,
    )
    # brine ON (as run_omip_core2 --prognostic-sea-ice does) so `step_sea_ice`
    # takes the NEW-PHYSICS path and keeps the 12-field DynamicSeaIceState.
    # With every gate off and dynamics="none" the dispatcher falls to the slab
    # branch, which DOWNGRADES the state to the 3-field SeaIceState — 9 of the
    # 12 fields under test would then not exist.  dynamics/transport stay
    # "none" so no grid-specific strain-rate operators are required.
    from legoesm.ice.config import BrineConfig, SnowConfig
    ice_cfg = SeaIceConfig(dynamics="none", transport="none",
                           brine=BrineConfig(enabled=True),
                           snow=SnowConfig(enabled=True))
    ice0 = init_dynamic_ice_state((N_LAT, N_LON))
    # Seed a real pack: a cold start (h=0, T_ice=260 K, S_ice=0) is exactly the
    # state a dropped ice restart would rebuild, so the reference must NOT be
    # that.  Thickness/concentration vary in latitude (polar-heavy).
    lat_w = np.linspace(1.0, 0.0, N_LAT)[:, None] * np.ones((1, N_LON))
    ice0 = ice0._replace(
        h_ice=ice0.h_ice.replace(data=jnp.asarray(0.8 * lat_w)),
        concentration=ice0.concentration.replace(
            data=jnp.asarray(0.7 * lat_w)),
        T_ice=ice0.T_ice.replace(
            data=jnp.asarray(constants.T_freeze - 8.0 * lat_w)),
        h_snow=ice0.h_snow.replace(data=jnp.asarray(0.1 * lat_w)),
    )
    return grid, z_coord, model, state, ice_cfg, ice0


def _forcing_stack(seed=3):
    """``N_REC`` synthetic CORE-II-like records, indexed per step by _idx_t."""
    rng = np.random.default_rng(seed)
    shape = (N_REC, N_LAT, N_LON)
    return {
        "u10": 6.0 + 3.0 * rng.standard_normal(shape),
        "v10": 2.0 + 3.0 * rng.standard_normal(shape),
        # Records differ strongly in T_air so the step->record mapping is
        # observable: a wrong step counter picks a visibly different record.
        "T_air": constants.T_freeze - 6.0
                 + 8.0 * np.linspace(-1.0, 1.0, N_REC)[:, None, None]
                 * np.ones((1, N_LAT, N_LON)),
        "q_air": 2.0e-3 * np.ones(shape),
        "sw_down": 120.0 + 100.0 * rng.random(shape),
        "lw_down": 250.0 + 20.0 * rng.random(shape),
        "precip": 1.0e-5 * rng.random(shape),
    }


def _leg(model, state, ice_state, ice_cfg, stack, step0, nsteps):
    """Integrate [step0+1, step0+nsteps] exactly as the driver's host loop does.

    Two-way ice<->ocean coupling: the ocean SST drives the ice basal exchange,
    and the ice concentration scales the open-water fraction that the surface
    stress / heat / salt reach the ocean through.  So a dropped ice restart
    perturbs the OCEAN answer too, not just the ice diagnostics.
    """
    for step in range(step0 + 1, step0 + nsteps + 1):
        it = _idx_t(step, DT, N_REC)
        forc = {k: v[it] for k, v in stack.items()}
        atm = _build_atm_to_surface_core2(forc, ramp=1.0)
        sst_K = state.T.data[:, :, 0] + constants.T_freeze
        ocn_u = state.u.data[:, :-1, 0]
        ocn_v = state.v.data[:-1, :, 0]
        ice_state, resp = step_sea_ice(ice_state, atm, sst_K, ocn_u, ocn_v,
                                       ice_cfg, U_MIN, DT)
        # step_sea_ice DOWNGRADES to the 3-field SeaIceState on its slab
        # branch; if that ever happened here the 12-field coverage this test
        # claims would silently shrink to 3.
        assert isinstance(ice_state, DynamicSeaIceState), (
            f"step_sea_ice returned {type(ice_state).__name__}; the test's "
            "SeaIceConfig no longer selects the 12-field dynamic state")
        f_ocn = 1.0 - ice_state.concentration.data
        # SIGN CONVENTION: TileResponse.shflx is positive UP (surface -> atm),
        # OceanSurfaceForcing.q_net is positive INTO the ocean, hence the
        # negation.  Only the open-water fraction reaches the ocean.
        sf = OceanSurfaceForcing(
            sw_down=jnp.asarray(forc["sw_down"]) * f_ocn,
            q_net=-jnp.asarray(resp.shflx) * f_ocn,
            tau_x=jnp.asarray(1.0e-3 * forc["u10"]) * f_ocn,
            tau_y=jnp.asarray(1.0e-3 * forc["v10"]) * f_ocn,
        )
        state = model.step(state, DT, surface_forcing=sf)
    return state, ice_state


def _assert_states_match(got, want, got_ice, want_ice, *, atol=1e-10):
    for name in ("T", "S", "u", "v", "eta"):
        np.testing.assert_allclose(
            np.asarray(getattr(got, name).data),
            np.asarray(getattr(want, name).data), atol=atol, rtol=0,
            err_msg=f"ocean {name} diverged after resume")
    assert (got.tke is None) == (want.tke is None)
    if want.tke is not None:
        np.testing.assert_allclose(np.asarray(got.tke.data),
                                   np.asarray(want.tke.data),
                                   atol=atol, rtol=0,
                                   err_msg="TKE carry diverged after resume")
    for name in DynamicSeaIceState._fields:
        np.testing.assert_allclose(
            np.asarray(getattr(got_ice, name).data),
            np.asarray(getattr(want_ice, name).data), atol=atol, rtol=0,
            err_msg=f"sea ice {name} diverged after resume")


@pytest.fixture(scope="module")
def _legs():
    """(continuous, restart-file inputs) — built once, reused by every case."""
    _, _, model, state0, ice_cfg, ice0 = _setup()
    stack = _forcing_stack()
    cont_state, cont_ice = _leg(model, state0, ice0, ice_cfg, stack, 0, N_TOTAL)
    half_state, half_ice = _leg(model, state0, ice0, ice_cfg, stack, 0,
                                N_TOTAL // 2)
    return dict(model=model, ice_cfg=ice_cfg, stack=stack, template=state0,
                ice_template=ice0, cont_state=cont_state, cont_ice=cont_ice,
                half_state=half_state, half_ice=half_ice)


def test_the_carries_under_test_are_actually_live(_legs):
    """Guard against a vacuous equivalence test: the TKE carry must really be
    populated, the ice must really have evolved, and the forcing must really
    depend on the step index."""
    assert _legs["half_state"].tke is not None, (
        "prognostic TKE is not carrying — the 'drop tke' mutation would be "
        "vacuous")
    assert not np.allclose(np.asarray(_legs["half_ice"].h_ice.data),
                           np.asarray(_legs["ice_template"].h_ice.data)), (
        "the ice pack did not evolve over the first leg")
    # The step->record map must actually move within one leg.
    idx = {_idx_t(s, DT, N_REC) for s in range(1, N_TOTAL + 1)}
    assert len(idx) > 1, "forcing record is constant; the step counter is inert"


def test_two_leg_resume_matches_continuous(tmp_path, _legs):
    """N steps continuous == N/2 + save/load + N/2, ocean AND ice AND carries."""
    path = tmp_path / "restart.npz"
    save_run_restart(path, _legs["half_state"], step=N_TOTAL // 2,
                     time_days=(N_TOTAL // 2) * DT / 86400.0,
                     grid_type="latlon", dt_seconds=DT,
                     n_forcing_records=N_REC, ice_state=_legs["half_ice"])
    warm_state, warm_ice, meta = load_run_restart(
        path, _legs["template"], ice_template=_legs["ice_template"],
        grid_type="latlon", dt_seconds=DT, n_forcing_records=N_REC)
    assert meta["step"] == N_TOTAL // 2

    got, got_ice = _leg(_legs["model"], warm_state, warm_ice, _legs["ice_cfg"],
                        _legs["stack"], meta["step"], N_TOTAL - meta["step"])
    _assert_states_match(got, _legs["cont_state"], got_ice, _legs["cont_ice"])


# --------------------------------------------------------------- mutations
# Each case breaks ONE restored piece and asserts the comparison goes RED.
# These are the non-vacuity proof for the test above.

def _resume_with(_legs, *, state, ice, step):
    return _leg(_legs["model"], state, ice, _legs["ice_cfg"], _legs["stack"],
                step, N_TOTAL - step)


def test_dropping_the_ice_state_breaks_continuity(_legs):
    """Resume on a COLD-START pack (h=0, T_ice=260 K, S_ice=0, h_snow=0) —
    what a restart that omits the ice would rebuild."""
    got, got_ice = _resume_with(_legs, state=_legs["half_state"],
                                ice=init_dynamic_ice_state(
                                    (N_LAT, N_LON), S_ice_init=0.0),
                                step=N_TOTAL // 2)
    with pytest.raises(AssertionError):
        _assert_states_match(got, _legs["cont_state"], got_ice,
                             _legs["cont_ice"])


def test_dropping_the_step_counter_breaks_continuity(_legs):
    """Resume with step=0 — what a restart that omits the counter would do:
    the CORE-II record index restarts, so the forcing is replayed."""
    got, got_ice = _leg(_legs["model"], _legs["half_state"], _legs["half_ice"],
                        _legs["ice_cfg"], _legs["stack"], 0,
                        N_TOTAL - N_TOTAL // 2)
    with pytest.raises(AssertionError):
        _assert_states_match(got, _legs["cont_state"], got_ice,
                             _legs["cont_ice"])


def test_dropping_the_tke_carry_breaks_continuity(_legs):
    """Resume with state.tke = None — the closure re-seeds at the background
    value and the mixing (hence T/S) differs."""
    got, got_ice = _resume_with(_legs,
                                state=_legs["half_state"]._replace(tke=None),
                                ice=_legs["half_ice"], step=N_TOTAL // 2)
    with pytest.raises(AssertionError):
        _assert_states_match(got, _legs["cont_state"], got_ice,
                             _legs["cont_ice"])
