"""M7: the FV3 duo COLUMN model on the closed lane's SPMD layouts, on 24
virtual CPU devices.  Run ALONE (the device count must be set before jax
is imported)::

    XLA_FLAGS=--xla_force_host_platform_device_count=24 pytest this_file

Every identity is against the SAME column model built on one device:
the FACE layout (6 devices, ``step_out_shardings = P("face")``) and the
WINDOW layout (kt=2, 24 devices, the window SPMD tests' certified pad)
stepped with a state-dependent synthetic physics (winds, heating, a
humidity sink and a cloud tendency that drives the positivity borrow
every step).  Claims B1-B4 of fv3_duo_gaps/q_review/m7_design.md: B1
(dynamics bitwise under the layout) was REFUTED by measurement (see
``REL`` below), so every identity is asserted at the closed lane's own
multi-process parity gate, 1e-11 of peak, and a leaf that differs names
itself with its number.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("XLA_FLAGS",
                      "--xla_force_host_platform_device_count=24")
jax = pytest.importorskip("jax")
jnp = jax.numpy
jax.config.update("jax_enable_x64", True)
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

from legoesm.core.state import HydrostaticTendencies  # noqa: E402

N, NG, KM, KT, PAD = 24, 3, 5, 2, 4
DT = 300.0
N_STEPS = 4
TAU = 6.0 * 3600.0


@pytest.fixture(scope="module")
def grid():
    if jax.device_count() < 6 * KT * KT:
        pytest.skip(f"need {6 * KT * KT} devices (XLA_FLAGS "
                    f"--xla_force_host_platform_device_count)")
    from legoesm.grids.factory import create_fv3_duo_grid
    return create_fv3_duo_grid(N, NG)


def _cfg():
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import FV3DuoConfig
    return FV3DuoConfig(km=KM, hydrostatic=True, moist=True)


def _column(dyn):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import (
        FV3DuoColumnModel)
    return FV3DuoColumnModel(dyn)


def _plain(grid):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoDynamicsModel)
    return _column(FV3DuoDynamicsModel(grid, _cfg()))


def _faces_model(grid):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoDynamicsModel)
    mesh = Mesh(np.array(jax.devices()[:6]), ("face",))
    return _column(FV3DuoDynamicsModel(
        grid, _cfg(), step_spmd_mesh=mesh,
        step_out_shardings=NamedSharding(mesh, P("face")),
        step_face_batched=True))


def _windows_model(grid):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoDynamicsModel)
    mesh = Mesh(np.array(jax.devices()[:6 * KT * KT]).reshape(6, KT, KT),
                ("face", "tile_i", "tile_j"))
    return _column(FV3DuoDynamicsModel(
        grid, _cfg(), step_spmd_mesh=mesh, step_windows=(KT, PAD),
        step_face_batched=True))


def _ic(model):
    """The DCMIP16 IC with a dash of cloud so the positivity stage has
    something to borrow against once the sink drives it negative."""
    b = model.dyn.dcmip16_initial_state(do_pert=True, n_tracers=3)
    ci = slice(NG, NG + N)
    q = [jnp.asarray(x) for x in b["q"]]
    q[1] = q[1].at[:, ci, ci].set(2e-7)
    return {**b, "q": q}


def _physics(state, mesh, coord, phys_state=None, forcing=None):
    """State-dependent, column-local, deterministic: Rayleigh winds,
    Newtonian heating towards a latitude profile, a humidity sink and
    a cloud sink larger than the cloud itself (negatives every step)."""
    lat = jnp.asarray(mesh.latCell)[:, None]
    T_ref = 250.0 + 40.0 * jnp.cos(lat) ** 2
    z = jnp.zeros_like(state.p_s.data)
    fld = lambda like, d, nm: like.replace(data=d, name=nm)  # noqa: E731
    return HydrostaticTendencies(
        du_dt=fld(state.u, -state.u.data / TAU, "du_dt"),
        dv_dt=fld(state.v, -state.v.data / TAU, "dv_dt"),
        dT_dt=fld(state.T, (T_ref - state.T.data) / TAU, "dT_dt"),
        dp_s_dt=fld(state.p_s, z, "dp_s"),
        dphis_dt=fld(state.phis, z, "dphis"),
        tracer_tendencies={
            "q_v": fld(state.tracers["q_v"],
                       -state.tracers["q_v"].data / TAU, "dq_v"),
            "q_c": fld(state.tracers["q_c"],
                       -jnp.full_like(state.tracers["q_c"].data, 1e-9),
                       "dq_c")})


def _run(model, n_steps, bundle=None):
    st = model.from_bundle(_ic(model) if bundle is None else bundle)
    for _ in range(n_steps):
        st = model.step(st, DT, physics_fn=_physics)
    return st


def _host_faces(model, state):
    return model.to_flat(model.to_bundle(state))


# The closed lane's sharded dynamics step is NOT bitwise with one device:
# MEASURED 2026-10-04 (job 10201205), dynamics only, 4 steps C24 km5:
# column u max |d| 8.3e-14 (faces) / 4.8e-14 (windows) of peak 20.6 m/s,
# i.e. the GSPMD-partitioned HLO fuses differently.  With physics the
# numbers are the same class (8.9e-14 / 5.5e-14), so the column seam adds
# nothing the dynamics did not already carry.  The gate is therefore the
# closed lane's own multi-process parity gate, 1e-11 of each leaf's peak
# (fv3duo_driver_hs_mp_parity.sbatch REL_PEAK_MAX), not bitwise.
REL = 1e-11
# TRACERS: a LIMITER-BRANCH flip, not rounding.  MEASURED (probe
# scripts/tmp/_m7_tracer_diff_probe.py, job 10201329, dynamics only,
# C24 km5, faces / windows vs one device): step 1 delp*q_c differs by
# 6e-18 (3e-15 of peak, 0 cells above 1e-16); at step 2 it jumps to
# 1.6e-13 / 4.8e-14 (8e-11 / 2.4e-11 of peak) in 36 / 10 of 17280 cells
# and then DECAYS (7.3e-11 at step 8, 34 cells): the 1e-14 wind
# difference flipped a monotonicity-limiter branch in a few cells once,
# at flux magnitude, and the scheme diffused it.  u/pt/delp stay at
# 1e-14 of peak throughout.  Gate for the mass-weighted tracers: 1e-9
# of peak -- a real seam defect (a dropped tendency, a wrong slot, a
# permuted column) is 1e-6 and up.
REL_TRACER = 1e-9


def _close(x, y, label, rel=REL):
    x, y = np.asarray(x), np.asarray(y)
    peak = max(float(np.abs(y).max()), 1e-300)
    d = float(np.abs(x - y).max())
    assert d <= rel * peak, (
        f"{label} differs, max |d| = {d:.3e} of peak {peak:.3e} "
        f"({d / peak:.2e} rel > {rel:.0e})")


def _assert_columns_equal(a, b, label, rel=REL):
    """The column leaves the driver sees; tracers are gated on the faces
    (mass-weighted, below), the columns being a slice of the same."""
    for nm in ("u", "v", "T", "p_s"):
        _close(getattr(a, nm).data, getattr(b, nm).data,
               f"{label}: column {nm}", rel)
    assert set(a.tracers) == set(b.tracers)


def _assert_faces_equal(a, b, label, rel=REL):
    ci = slice(NG, NG + N)
    ce = slice(NG, NG + N + 1)
    win = {"u": (slice(None), ci, ce), "v": (slice(None), ce, ci)}
    for k in ("u", "v", "pt", "delp"):
        w = win.get(k, (slice(None), ci, ci))
        _close(np.asarray(a["state"][k])[w], np.asarray(b["state"][k])[w],
               f"{label}: state {k}", rel)
    for k in a["press"]:
        _close(a["press"][k], b["press"][k], f"{label}: press {k}", rel)
    # tracers MASS-weighted (delp*q, what the dycore transports): a 2e-7
    # mixing ratio differed by 1.6e-17 absolute between layouts (job
    # 10201234) -- one ulp of delp*q, 8e-11 of the mixing ratio's peak
    dpa = np.asarray(a["state"]["delp"])[:, ci, ci]
    dpb = np.asarray(b["state"]["delp"])[:, ci, ci]
    for i, (x, y) in enumerate(zip(a["q"], b["q"])):
        _close(dpa * np.asarray(x)[:, ci, ci], dpb * np.asarray(y)[:, ci, ci],
               f"{label}: delp*q{i}",
               rel if rel == 0.0 else max(rel, REL_TRACER))


@pytest.fixture(scope="module")
def reference(grid):
    model = _plain(grid)
    st = _run(model, N_STEPS)
    return model, st, _host_faces(model, st)


def test_the_physics_moves_the_state_and_drives_the_borrow(grid, reference):
    """Non-vacuity: every column leaf moved, and the cloud sink produced
    negatives the positivity stage had to repair (q_c stays >= 0)."""
    model, st, faces = reference
    ic = _ic(model)
    st0 = model.from_bundle(ic)
    for nm in ("u", "v", "T", "p_s"):
        assert np.abs(np.asarray(getattr(st, nm).data)
                      - np.asarray(getattr(st0, nm).data)).max() > 0.0, nm
    qc = np.asarray(st.tracers["q_c"].data)
    assert qc.min() >= 0.0
    assert qc.max() < 2e-7           # the sink acted
    assert qc.min() == 0.0           # ... and the floor/borrow bound


@pytest.mark.parametrize("build", [_faces_model, _windows_model],
                         ids=["faces", "windows"])
def test_dynamics_only_on_the_layout_is_within_the_parity_gate(grid, build):
    """B1 in isolation: physics_fn=None (dynamics + the positivity stage),
    the layout vs one device, at the parity gate; a difference here is the
    closed lane's own sharded-step class, not the column seam's."""
    plain = _plain(grid)
    ref = plain.from_bundle(_ic(plain))
    model = build(grid)
    st = model.from_bundle(_ic(model))
    for _ in range(N_STEPS):
        ref = plain.step(ref, DT)
        st = model.step(st, DT)
    _assert_columns_equal(st, ref, f"{build.__name__} dynamics-only")
    _assert_faces_equal(_host_faces(model, st), _host_faces(plain, ref),
                        f"{build.__name__} dynamics-only")


def test_face_layout_is_the_single_device_lane_within_the_gate(grid, reference):
    _, ref_st, ref_faces = reference
    model = _faces_model(grid)
    st = _run(model, N_STEPS)
    # the band sharding took effect (not a silently replicated physics)
    assert st.u.data.sharding.is_equivalent_to(model._col_sharding, 2)
    assert st.native["state"]["pt"].sharding.is_equivalent_to(
        model.step_out_shardings, 4)
    _assert_columns_equal(st, ref_st, "face layout")
    _assert_faces_equal(_host_faces(model, st), ref_faces, "face layout")


def test_window_layout_is_the_single_device_lane_within_the_gate(grid, reference):
    _, ref_st, ref_faces = reference
    model = _windows_model(grid)
    assert model.window_layout is not None and model.window_layout.nb == 24
    st = _run(model, N_STEPS)
    assert st.u.data.sharding.is_equivalent_to(model._col_sharding, 2)
    assert int(st.native["state"]["pt"].shape[0]) == 24
    _assert_columns_equal(st, ref_st, "window layout")
    _assert_faces_equal(_host_faces(model, st), ref_faces, "window layout")


@pytest.mark.parametrize("build", [_faces_model, _windows_model],
                         ids=["faces", "windows"])
def test_restart_across_layouts_is_within_the_gate(grid, reference, build):
    """A host six-face bundle written by one layout resumes on another
    (and on one device) and lands on the uninterrupted run at the gate --
    the checkpoint file is layout-free (G3 at the model level)."""
    _, ref_st, ref_faces = reference
    a = build(grid)
    st_a = _run(a, 2)
    host = _host_faces(a, st_a)
    # the checkpoint writer's input: HOST arrays under windows (to_flat is a
    # host scatter), fully addressable arrays on the face layout (identity)
    for v in host["state"].values():
        if build is _windows_model:
            assert type(v) is np.ndarray, type(v)
        else:
            assert not isinstance(v, jax.Array) or v.is_fully_addressable
    plain = _plain(grid)
    st_p = _run(plain, N_STEPS - 2, bundle=host)
    _assert_columns_equal(st_p, ref_st, f"{build.__name__} -> one device")
    # and the other way: one device -> this layout
    p1 = _plain(grid)
    st_1 = _run(p1, 2)
    b = build(grid)
    st_b = _run(b, N_STEPS - 2, bundle=_host_faces(p1, st_1))
    _assert_faces_equal(_host_faces(b, st_b), ref_faces,
                        f"one device -> {build.__name__}")


def test_column_edits_write_back_on_the_window_layout(grid):
    """The driver's post-step T / tracer edit reaches the window bundle
    (the write-back scatters to faces and gathers back)."""
    model = _windows_model(grid)
    st = _run(model, 1)
    T_new = st.T.replace(data=st.T.data + 1.0)
    edited = st._replace(T=T_new)
    bundle = model.to_bundle(edited)
    faces = model.to_flat(bundle)
    ci = slice(NG, NG + N)
    before = np.asarray(model.to_flat(st.native)["state"]["pt"])[:, ci, ci]
    after = np.asarray(faces["state"]["pt"])[:, ci, ci]
    assert np.array_equal(after, before + 1.0)


def test_kt1_windows_are_refused(grid):
    from types import SimpleNamespace
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import (
        FV3DuoColumnModel)
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoDynamicsModel)
    dyn = FV3DuoDynamicsModel(grid, _cfg())
    dyn.window_layout = SimpleNamespace(nb=6)
    with pytest.raises(NotImplementedError, match="kt=1"):
        FV3DuoColumnModel(dyn)


# ---------------------------------------------------------------------
# the DRIVER on the window layout (S2/S3): the factory's shared layout
# policy, the checkpoint seam, the restart and the terrain rewrap
# ---------------------------------------------------------------------

# C24: on C12 a (nb, W, W, nq*km=15) tracer stack is AMBIGUOUS to the
# window layout's horizontal-extent rule (15+1 lies in [n, m_a+1] = [12, 19]),
# a pre-existing window-lane limit (fv3_duo_windows.horizontal_axes); C24
# km5 with three tracers is the window SPMD tests' own certified shape.
DN, DKM, DDT = 24, 5, 1920.0


def _driver_cfg(out_dir, *, windows, **over):
    from legoesm.driver.config import (
        DycoreConfig, ExperimentConfig, GridConfig, OutputConfig)
    dyc = dict(model_type="hydrostatic", discretization="fv3_duo",
               dt=over.pop("dt", DDT),
               column_lane=over.pop("column_lane", True))
    if windows:
        dyc.update(fv3_duo_windows=2, fv3_duo_window_pad=4)
    base = dict(
        grid=GridConfig(grid_type="cubed_sphere", resolution=DN, nlev=DKM),
        dycore=DycoreConfig(**dyc), days=over.pop("days", 1),
        radiation="none", convection="none", microphysics="none",
        turbulence="none", gravity_wave_drag="none", precision="fp64",
        output=OutputConfig(diag_days=0,
                            checkpoint_days=over.pop("checkpoint_days", 0),
                            output_dir=str(out_dir)))
    base.update(over)
    return ExperimentConfig(**base)


def _driver(out_dir, *, windows, run=True, **over):
    from legoesm.driver.model_driver import ModelDriver
    out_dir.mkdir(parents=True, exist_ok=True)
    drv = ModelDriver(_driver_cfg(out_dir, windows=windows, **over),
                      output_dir=out_dir)
    drv.setup()
    if run:
        assert drv.run() == "COMPLETED"
    return drv


def _driver_faces(drv):
    return drv.model.to_flat(drv.model.to_bundle(drv.state))


def _assert_driver_faces_close(a, b, label, n=DN, ng=3, rel=REL):
    ci, ce = slice(ng, ng + n), slice(ng, ng + n + 1)
    win = {"u": (slice(None), ci, ce), "v": (slice(None), ce, ci)}
    for k in ("u", "v", "pt", "delp"):
        w = win.get(k, (slice(None), ci, ci))
        _close(np.asarray(a["state"][k])[w], np.asarray(b["state"][k])[w],
               f"{label}: state {k}", rel)
    _close(a["press"]["ps"], b["press"]["ps"], f"{label}: ps", rel)
    dpa = np.asarray(a["state"]["delp"])[:, ci, ci]
    dpb = np.asarray(b["state"]["delp"])[:, ci, ci]
    for i, (x, y) in enumerate(zip(a["q"], b["q"])):
        _close(dpa * np.asarray(x)[:, ci, ci], dpb * np.asarray(y)[:, ci, ci],
               f"{label}: delp*q{i}",
               rel if rel == 0.0 else max(rel, REL_TRACER))


@pytest.mark.parametrize("phys", [{}, {"microphysics": "kessler"}],
                         ids=["dry", "kessler"])
def test_driver_column_lane_on_windows_is_the_single_device_run(tmp_path, phys):
    """``run()`` through the MPAS lane with the column model on the
    window layout (kt=2 pad=4, 24 devices, the factory's shared layout
    policy) equals the single-device column run at the parity gate."""
    win = _driver(tmp_path / "win", windows=True, **phys)
    assert win.model.window_layout is not None
    assert win.model.window_layout.nb == 24
    assert win.model.step_out_shardings is not None
    one = _driver(tmp_path / "one", windows=False, **phys)
    assert one.model.window_layout is None
    # the gate is the CLOSED lane's own windows-vs-one-device difference on
    # this deck (MEASURED 7.5e-11 of peak after a day at C24: the sharded
    # dynamics' class, not the column seam's), x10 -- the column seam must
    # add nothing to it
    c_win = _driver(tmp_path / "c_win", windows=True,
                    column_lane=False, **phys)
    c_one = _driver(tmp_path / "c_one", windows=False,
                    column_lane=False, **phys)
    ci = slice(3, 3 + DN)
    worst = 0.0
    for k in ("u", "v", "pt", "delp"):
        x = np.asarray(c_win.model.to_flat(c_win.state)["state"][k])[:, ci, ci]
        y = np.asarray(c_one.state["state"][k])[:, ci, ci]
        worst = max(worst, float(np.abs(x - y).max() / np.abs(y).max()))
    assert 0.0 < worst < 1e-9, worst       # the reference itself is bounded
    _assert_driver_faces_close(_driver_faces(win), _driver_faces(one),
                               f"driver windows vs one device ({phys}; "
                               f"closed lane's own {worst:.2e})",
                               rel=max(REL, 10.0 * worst))
    # the run moved (not a vacuous identity on the IC)
    ic = win.model.to_flat(win.model.dyn.dcmip16_initial_state(do_pert=True))
    assert np.abs(np.asarray(_driver_faces(win)["state"]["pt"])
                  - np.asarray(ic["state"]["pt"])).max() > 0.0


@pytest.mark.parametrize("resume_windows", [True, False],
                         ids=["resume-on-windows", "resume-on-one-device"])
def test_driver_column_lane_restart_across_layouts(tmp_path, resume_windows):
    """G3 at the driver level: a checkpoint written on the window layout
    (six host faces through ``_fv3_duo_host_faces``) resumes on the same
    layout or on one device and lands on the uninterrupted window run at
    the parity gate; the file carries the duo bundle + the column stamps."""
    from legoesm.driver.model_driver import ModelDriver
    a = _driver(tmp_path / "a", windows=True, days=2, checkpoint_days=1,
                microphysics="kessler")
    mid = tmp_path / "a" / "checkpoint_day_0001.npz"
    assert mid.is_file()
    with np.load(mid) as d:
        assert str(d["_schema"]) == "fv3duo_ckpt_v1"
        assert d["state_pt"].shape[0] == 6        # six faces, not windows
        assert "fv3duo_tracer_names" in d.files
    b_dir = tmp_path / "b"
    b_dir.mkdir()
    b = ModelDriver(_driver_cfg(b_dir, windows=resume_windows, days=1,
                                microphysics="kessler"), output_dir=b_dir)
    b.setup()
    step, day = b.load_checkpoint(mid)
    assert day == pytest.approx(1.0)
    assert b.run(start_step=step, start_day=day) == "COMPLETED"
    # same layout: the SAME jitted programs resume from the six host faces
    # -> BITWISE with the uninterrupted run (the layout-free file claim);
    # one device: the layouts' own 1e-10 difference at day 1 grows through
    # a day of dynamics (MEASURED 5.2e-10 of peak on u) -- bounded, not tight
    _assert_driver_faces_close(_driver_faces(b), _driver_faces(a),
                               f"restart resume_windows={resume_windows}",
                               rel=0.0 if resume_windows else 1e-8)


def test_driver_terrain_rewrap_keeps_the_layout_and_the_knobs(tmp_path):
    """codex M7 claim review: the terrain rebuild must not silently drop
    the decomposition or reset the positivity knobs."""
    from legoesm.driver.config import DycoreConfig
    # a NON-default knob (codex: both defaults would pass a dropped carry)
    cfg = _driver_cfg(tmp_path / "w", windows=True)
    cfg = cfg._replace(dycore=cfg.dycore._replace(
        mpas_conservative_tracer_clamp=False))
    assert DycoreConfig().mpas_conservative_tracer_clamp is True
    from legoesm.driver.model_driver import ModelDriver
    (tmp_path / "w").mkdir()
    drv = ModelDriver(cfg, output_dir=tmp_path / "w")
    drv.setup()
    old = drv.model
    assert old.window_layout is not None
    assert old.conservative_tracer_clamp is False
    drv._fv3_duo_column_rewrap(old.grid)
    new = drv.model
    assert new is not old
    assert new.window_layout is not None and new.window_layout.nb == 24
    assert new.step_spmd_mesh is not None
    assert new.step_out_shardings is not None
    assert new.conservative_tracer_clamp == old.conservative_tracer_clamp
    assert (new.energy_consistent_moisture_clip
            == old.energy_consistent_moisture_clip)
    assert drv.grid is new.mesh
